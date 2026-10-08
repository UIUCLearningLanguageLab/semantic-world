"""Explicit requirements from an event file, applied to the taxonomy's objects.

A requirement in an event file replaces a one-place event type's CAN rule, or a two-place event
type's own constraint, before the world is built. The affected derived values (the CAN column,
the node vectors, the projections, and the relation statistics) are recomputed with the
taxonomy's own functions. The taxonomy run written to ``taxonomy/`` is the unmodified one.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from semantic_world.common.boolean import dnf_literal_count
from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.constraints import (
    Constraint,
    ConstraintLiteral,
    Relations,
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
    leaf_pair_density,
)
from semantic_world.taxonomy.expressions import Cmp, Const, Expr, Gt, Not, Op, Var
from semantic_world.taxonomy.fixed import compute_node_vectors
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.projections import compute_projections
from semantic_world.taxonomy.relation_stats import compute_relation_stats
from semantic_world.taxonomy.rule_matrices import build_rule_matrices
from semantic_world.taxonomy.rules import (
    Input,
    Rule,
    RuleSet,
    Threshold,
    input_key,
    model_quantile_of,
)
from semantic_world.taxonomy.streams import stream_generator
from semantic_world.world.event_file import EventFile, ExplicitEventType
from semantic_world.world.labels import untranslate

_ROLE_BACK = {"agent": "a", "patient": "p"}


def _split_role(name: str, source: str, field: str) -> tuple[str, str]:
    role, _, label = name.partition(".")
    if role not in _ROLE_BACK or not label:
        raise ConfigError(
            source, field, f"{name} must name a role: agent.<label> or patient.<label>"
        )
    if label.startswith("BOOLFL."):
        raise ConfigError(
            source, field, f"{name} reads a fluent; a requirement reads static facts only"
        )
    return role, label


def _static_feature(label: str, taxonomy: TaxonomyResult, source: str, field: str):
    try:
        old = untranslate(label)
    except Exception:
        raise ConfigError(source, field, f"unknown label {label}") from None
    features = taxonomy.features
    if old not in features:
        raise ConfigError(source, field, f"unknown feature {label}")
    feature = features[old]
    if feature.type == "can":
        raise ConfigError(source, field, f"{label} is an event type, not a static feature")
    return feature


def _threshold(
    scalar_label: str, threshold: float, taxonomy: TaxonomyResult, source: str, field: str
) -> Threshold:
    prefix, _, number = scalar_label.partition(".")
    count = taxonomy.config.scalars.count
    if prefix != "SCALARDIM" or not number.isdigit() or not 1 <= int(number) <= count:
        raise ConfigError(
            source, field, f"unknown scalar {scalar_label}; the world has {count} scalars"
        )
    return Threshold(int(number), threshold, model_quantile_of(taxonomy.config.scalars, threshold))


def _to_old(expr: Expr, roles: bool) -> Expr:
    """The expression in the taxonomy's vocabulary: role prefixes shortened (``a.``, ``p.``) when
    ``roles`` is true, and dropped otherwise (a one-place rule over the agent alone)."""

    def name(new: str) -> str:
        role, _, label = new.partition(".")
        old = untranslate(label)
        return f"{_ROLE_BACK[role]}.{old}" if roles else old

    if isinstance(expr, Var):
        return Var(name(expr.name))
    if isinstance(expr, Gt):
        return Gt(name(expr.scalar), expr.threshold)
    if isinstance(expr, Cmp):
        return Cmp(name(expr.agent), name(expr.patient), expr.low, expr.high)
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Not):
        return Not(_to_old(expr.operand, roles))
    assert isinstance(expr, Op)
    return Op(expr.operator, tuple(_to_old(o, roles) for o in expr.operands))


def one_place_rule(entry: ExplicitEventType, taxonomy: TaxonomyResult, source: str) -> Rule:
    field = f"{entry.field}.requirement"
    assert entry.requirement is not None
    old_label = untranslate(entry.label)
    feature = taxonomy.features[old_label]
    inputs: list[Input] = []
    for atom in entry.requirement.atoms():
        if isinstance(atom, Cmp):
            raise ConfigError(source, field, "a one-place requirement holds no comparison")
        role, label = _split_role(atom.scalar if isinstance(atom, Gt) else atom.name, source, field)
        if role != "agent":
            raise ConfigError(source, field, f"{entry.label} has the role agent only")
        if isinstance(atom, Gt):
            inputs.append(_threshold(label, atom.threshold, taxonomy, source, field))
        else:
            inputs.append(_static_feature(label, taxonomy, source, field))
    expression = _to_old(entry.requirement, roles=False)
    table = expression.truth_table([input_key(i) for i in inputs])
    return Rule(
        output=feature,
        inputs=tuple(inputs),
        table=table,
        expression=expression,
        family="explicit",
        shj_type=None,
        nesting_depth=expression.depth(),
        min_dnf_literals=dnf_literal_count(table),
    )


def two_place_constraint(
    entry: ExplicitEventType, taxonomy: TaxonomyResult, source: str
) -> Constraint:
    field = f"{entry.field}.requirement"
    assert entry.requirement is not None
    literals: list[ConstraintLiteral] = []
    for atom in entry.requirement.atoms():
        if isinstance(atom, Cmp):
            role_a, a = _split_role(atom.agent, source, field)
            role_p, p = _split_role(atom.patient, source, field)
            if role_a != "agent" or role_p != "patient":
                raise ConfigError(
                    source, field, "a comparison is agent.SCALARDIM.i - patient.SCALARDIM.j"
                )
            ta = _threshold(a, 0.0, taxonomy, source, field)
            tp = _threshold(p, 0.0, taxonomy, source, field)
            literals.append(ScalarComparison(ta.scalar, tp.scalar, atom.low, atom.high))
        elif isinstance(atom, Gt):
            role, label = _split_role(atom.scalar, source, field)
            literals.append(
                RoleThreshold(
                    _ROLE_BACK[role], _threshold(label, atom.threshold, taxonomy, source, field)
                )
            )
        else:
            role, label = _split_role(atom.name, source, field)
            literals.append(
                RoleFeature(_ROLE_BACK[role], _static_feature(label, taxonomy, source, field))
            )
    expression = _to_old(entry.requirement, roles=True)
    table = expression.truth_table([item.key for item in literals])
    return Constraint(
        f"K.{untranslate(entry.label)}",
        "explicit",
        tuple(literals),
        table,
        expression,
        dnf_literal_count(table),
    )


def apply_requirements(taxonomy: TaxonomyResult, event_file: EventFile) -> TaxonomyResult:
    """The taxonomy result with every explicit requirement applied, and the derived values
    recomputed. Returns ``taxonomy`` itself when the file gives no requirement."""
    entries = [e for e in event_file.event_types.values() if e.requirement is not None]
    if not entries:
        return taxonomy
    source = event_file.source
    rules = list(taxonomy.rules.rules)
    relations = taxonomy.relations
    own = dict(relations.own_constraints) if relations is not None else {}
    changed_rules = False
    changed_constraints = False
    for entry in entries:
        old = untranslate(entry.label)
        if entry.label.startswith("EVENTTYPE1."):
            if old not in taxonomy.features or taxonomy.features[old].type != "can":
                raise ConfigError(
                    source, entry.field, "unknown event type; the world has none with this label"
                )
            rule = one_place_rule(entry, taxonomy, source)
            rules = [rule if r.output.label == old else r for r in rules]
            changed_rules = True
        else:
            verbs = {v.label for v in relations.verbs.verbs} if relations is not None else set()
            if old not in verbs:
                raise ConfigError(
                    source, entry.field, "unknown event type; the world has none with this label"
                )
            own[old] = two_place_constraint(entry, taxonomy, source)
            changed_constraints = True
    config = taxonomy.config
    rule_set = taxonomy.rules
    instances = taxonomy.instances
    vectors = taxonomy.vectors
    if changed_rules:
        rule_set = RuleSet(taxonomy.features, tuple(rules), taxonomy.rules.warnings)
        scalars = instances.scalars if taxonomy.features.scalar_count else None
        values = rule_set.compute(instances.free_values(taxonomy.features), scalars)
        instances = dataclasses.replace(instances, values=values)
        vectors = compute_node_vectors(rule_set, taxonomy.tree, instances, config.scalars)
    new_relations = relations
    projections = taxonomy.projections
    relation_stats = taxonomy.relation_stats
    if relations is not None and (changed_rules or changed_constraints):
        density = relations.constraint_density
        if density is not None and changed_constraints:
            leaf_values = np.stack([leaf.values for leaf in taxonomy.tree.leaves])
            leaf_scalars = np.stack([leaf.scalars for leaf in taxonomy.tree.leaves])
            density = dict(density)
            for constraint in own.values():
                density[constraint.label] = leaf_pair_density(
                    [constraint], leaf_values, leaf_scalars
                )
        new_relations = Relations(
            relations.verbs,
            relations.feature_constraints,
            own,
            instances,
            constraint_density=density,
            verb_density=relations.verb_density,
            verb_tries=relations.verb_tries,
            outside_range=relations.outside_range,
            warnings=relations.warnings,
        )
        projections = compute_projections(
            config, rule_set, new_relations, instances, np.random.default_rng(0)
        )
        relation_stats = compute_relation_stats(
            config,
            taxonomy.tree,
            instances,
            new_relations,
            vectors,
            stream_generator(config.seed, "taxonomy:pairs"),
        )
    return dataclasses.replace(
        taxonomy,
        rules=rule_set,
        instances=instances,
        vectors=vectors,
        relations=new_relations,
        projections=projections,
        relation_stats=relation_stats,
        matrices=build_rule_matrices(rule_set),
    )
