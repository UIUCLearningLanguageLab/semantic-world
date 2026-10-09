"""Explicit requirements from an event file.

A requirement in an event file replaces a one-place event type's sampled requirement, or a
two-place event type's own constraint (the constraints of its event-type features stay, so the
event type still entails its ancestors' base relations, REL.10). The replacement happens before
the capacities and the relation statistics are computed, in ``semantic_world.world.statics``.
"""

from __future__ import annotations

from semantic_world.common.boolean import dnf_literal_count
from semantic_world.taxonomy.config import ConfigError, ScalarsConfig
from semantic_world.taxonomy.expressions import Cmp, Const, Expr, Gt, Not, Op, Var
from semantic_world.taxonomy.features import Feature, FeatureSet
from semantic_world.taxonomy.rules import Input, Rule, Threshold, input_key, model_quantile_of
from semantic_world.world.constraints import (
    Constraint,
    ConstraintLiteral,
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
    constraint_label,
)
from semantic_world.world.dynamics import ROLES
from semantic_world.world.event_file import ExplicitEventType

STATIC_TYPES = ("is", "has")


def _split_role(name: str, source: str, field: str) -> tuple[str, str]:
    role, _, label = name.partition(".")
    if role not in ROLES or not label:
        raise ConfigError(
            source, field, f"{name} must name a role: agent.<label> or patient.<label>"
        )
    if label.startswith("BOOLFL."):
        raise ConfigError(
            source, field, f"{name} reads a fluent; a requirement reads static facts only"
        )
    return role, label


def _static_feature(label: str, features: FeatureSet, source: str, field: str) -> Feature:
    if label not in features:
        raise ConfigError(source, field, f"unknown feature {label}")
    feature = features[label]
    if feature.type not in STATIC_TYPES:
        raise ConfigError(source, field, f"{label} is an event type, not a static feature")
    return feature


def _threshold(
    scalar_label: str, threshold: float, scalars: ScalarsConfig, source: str, field: str
) -> Threshold:
    prefix, _, number = scalar_label.partition(".")
    count = scalars.count
    if prefix != "SCALARDIM" or not number.isdigit() or not 1 <= int(number) <= count:
        raise ConfigError(
            source, field, f"unknown scalar {scalar_label}; the world has {count} scalars"
        )
    return Threshold(int(number), threshold, model_quantile_of(scalars, threshold))


def _unprefixed(expr: Expr) -> Expr:
    """A one-place requirement over the agent alone, with the role prefixes dropped, as a rule
    over the entity's own features."""

    def name(new: str) -> str:
        return new.partition(".")[2]

    if isinstance(expr, Var):
        return Var(name(expr.name))
    if isinstance(expr, Gt):
        return Gt(name(expr.scalar), expr.threshold)
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Not):
        return Not(_unprefixed(expr.operand))
    assert isinstance(expr, Op)
    return Op(expr.operator, tuple(_unprefixed(o) for o in expr.operands))


def one_place_rule(
    entry: ExplicitEventType, features: FeatureSet, scalars: ScalarsConfig, source: str
) -> Rule:
    """The requirement of a one-place event type from its event-file entry, as a rule over the
    extended feature table (``semantic_world.world.unary``)."""
    field = f"{entry.field}.requirement"
    assert entry.requirement is not None
    output = features[entry.label]
    inputs: list[Input] = []
    for atom in entry.requirement.atoms():
        if isinstance(atom, Cmp):
            raise ConfigError(source, field, "a one-place requirement holds no comparison")
        role, label = _split_role(atom.scalar if isinstance(atom, Gt) else atom.name, source, field)
        if role != "agent":
            raise ConfigError(source, field, f"{entry.label} has the role agent only")
        if isinstance(atom, Gt):
            inputs.append(_threshold(label, atom.threshold, scalars, source, field))
        else:
            inputs.append(_static_feature(label, features, source, field))
    expression = _unprefixed(entry.requirement)
    table = expression.truth_table([input_key(i) for i in inputs])
    return Rule(
        output=output,
        inputs=tuple(inputs),
        table=table,
        expression=expression,
        family="explicit",
        shj_type=None,
        nesting_depth=expression.depth(),
        min_dnf_literals=dnf_literal_count(table),
    )


def two_place_constraint(
    entry: ExplicitEventType, features: FeatureSet, scalars: ScalarsConfig, source: str
) -> Constraint:
    """The own constraint of a two-place event type from its event-file entry."""
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
            ta = _threshold(a, 0.0, scalars, source, field)
            tp = _threshold(p, 0.0, scalars, source, field)
            literals.append(ScalarComparison(ta.scalar, tp.scalar, atom.low, atom.high))
        elif isinstance(atom, Gt):
            role, label = _split_role(atom.scalar, source, field)
            literals.append(
                RoleThreshold(role, _threshold(label, atom.threshold, scalars, source, field))
            )
        else:
            role, label = _split_role(atom.name, source, field)
            literals.append(RoleFeature(role, _static_feature(label, features, source, field)))
    table = entry.requirement.truth_table([item.key for item in literals])
    return Constraint(
        constraint_label(entry.label),
        "explicit",
        tuple(literals),
        table,
        entry.requirement,
        dnf_literal_count(table),
    )


__all__ = ["one_place_rule", "two_place_constraint"]
