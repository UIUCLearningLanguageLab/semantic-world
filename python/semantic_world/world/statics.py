"""The static side of a world: the taxonomy run, the one-place requirements, the event-type
tree with its constraints, the capacities, and the relation statistics.

``build_statics`` draws, in this order: the one-place requirements (``world:requirements``), the
event-type features' constraints and the event-type tree with each event type's own constraint
(``world:constraints`` and ``world:event_tree``), then replaces the requirements that an event
file gives explicitly, and computes the capacities (the local test samples from
``world:constraints``) and the relation statistics (``world:pairs``). Fluents, preconditions,
and effects are the dynamic side (``semantic_world.world.event_types``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.features import FeatureSet
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import Rule, RuleSet
from semantic_world.world.config import Config
from semantic_world.world.constraints import (
    Constraint,
    Relations,
    generate_relations,
    leaf_pair_density,
)
from semantic_world.world.event_file import EventFile
from semantic_world.world.event_tree import EventTree
from semantic_world.world.projections import Projections, compute_projections
from semantic_world.world.relation_stats import RelationStats, compute_relation_stats
from semantic_world.world.requirements import one_place_rule, two_place_constraint
from semantic_world.world.streams import WorldStreams
from semantic_world.world.unary import (
    ONE_PLACE,
    UnaryRequirements,
    assemble_unary,
    extended_features,
    generate_unary,
)


@dataclass(frozen=True)
class StaticWorld:
    taxonomy: TaxonomyResult
    unary: UnaryRequirements
    event_tree: EventTree | None
    relations: Relations | None
    projections: Projections | None
    relation_stats: RelationStats | None
    warnings: tuple[str, ...]

    @property
    def features(self) -> FeatureSet:
        """The taxonomy's features followed by the one-place event types."""
        return self.unary.features

    @property
    def rules(self) -> RuleSet:
        """The taxonomy's rules followed by the one-place requirements."""
        return self.unary.rules

    @property
    def instances(self) -> Instances:
        return self.taxonomy.instances

    @property
    def values(self) -> np.ndarray:
        """Every entity's static features and one-place capacities, by feature position."""
        return self.unary.values


def build_statics(
    taxonomy: TaxonomyResult, config: Config, streams: WorldStreams, event_file: EventFile | None
) -> StaticWorld:
    settings = config.event_types
    scalars = taxonomy.config.scalars
    explicit_rules, explicit_constraints = _explicit(taxonomy, settings.unary.count, event_file)
    unary = generate_unary(taxonomy, settings.unary, streams.requirements, explicit_rules)
    event_tree, relations = generate_relations(
        settings.binary,
        scalars,
        taxonomy.rules,
        taxonomy.tree,
        taxonomy.instances,
        streams.constraints,
        streams.event_tree,
    )
    warnings: list[str] = []
    if explicit_constraints and relations is None:
        assert event_file is not None
        label = next(iter(explicit_constraints))
        raise ConfigError(
            event_file.source,
            event_file.event_types[label].field,
            "unknown event type; the world has no two-place event types",
        )
    if event_tree is not None and relations is not None:
        warnings += list(event_tree.tree.warnings) + list(relations.warnings)
        if explicit_constraints:
            assert event_file is not None
            relations = _with_own_constraints(relations, explicit_constraints, taxonomy, event_file)
    projections = None
    relation_stats = None
    if relations is not None and settings.binary is not None:
        projections = compute_projections(
            unary.rules, relations, taxonomy.instances, streams.constraints
        )
        relation_stats = compute_relation_stats(
            settings.binary,
            taxonomy.config.analysis,
            taxonomy.tree,
            taxonomy.instances,
            relations,
            taxonomy.vectors,
            streams.pairs,
        )
    return StaticWorld(
        taxonomy, unary, event_tree, relations, projections, relation_stats, tuple(warnings)
    )


def _explicit(
    taxonomy: TaxonomyResult, count: int, event_file: EventFile | None
) -> tuple[dict[str, Rule], dict[str, Constraint]]:
    """The explicit requirements of an event file: the one-place ones as rules over the extended
    features, the two-place ones as own constraints."""
    rules: dict[str, Rule] = {}
    constraints: dict[str, Constraint] = {}
    if event_file is None:
        return rules, constraints
    features = extended_features(taxonomy, count)
    scalars = taxonomy.config.scalars
    for entry in event_file.event_types.values():
        if entry.requirement is None:
            continue
        if entry.label.startswith(f"{ONE_PLACE}."):
            if entry.label not in features:
                raise ConfigError(
                    event_file.source,
                    entry.field,
                    "unknown event type; the world has none with this label",
                )
            rules[entry.label] = one_place_rule(entry, features, scalars, event_file.source)
        else:
            constraints[entry.label] = two_place_constraint(
                entry, features, scalars, event_file.source
            )
    return rules, constraints


def _with_own_constraints(
    relations: Relations,
    explicit: dict[str, Constraint],
    taxonomy: TaxonomyResult,
    event_file: EventFile,
) -> Relations:
    """The relations with the explicit own constraints in place of the sampled ones. An
    explicit requirement is the whole requirement of its event type apart from the base
    relations of the event type's ancestors (REL.10): the constraints of the features that are
    defining at an ancestor stay, and the constraints of its other true features go with the
    sampled own constraint."""
    event_types = {v.label for v in relations.event_tree.event_types}
    own = dict(relations.own_constraints)
    for label, constraint in explicit.items():
        if label not in event_types:
            raise ConfigError(
                event_file.source,
                event_file.event_types[label].field,
                "unknown event type; the world has none with this label",
            )
        own[label] = constraint
    density = relations.constraint_density
    if density is not None:
        leaf_values = np.stack([leaf.values for leaf in taxonomy.tree.leaves])
        leaf_scalars = np.stack([leaf.scalars for leaf in taxonomy.tree.leaves])
        density = dict(density)
        for label in explicit:
            density[own[label].label] = leaf_pair_density([own[label]], leaf_values, leaf_scalars)
    return Relations(
        relations.event_tree,
        relations.feature_constraints,
        own,
        relations.instances,
        constraint_density=density,
        event_type_density=relations.event_type_density,
        event_type_tries=relations.event_type_tries,
        outside_range=relations.outside_range,
        warnings=relations.warnings,
        explicit=frozenset(explicit),
    )


def replace_requirements(statics: StaticWorld, rules: dict[str, Rule] | None = None) -> StaticWorld:
    """``statics`` with some one-place requirements replaced (tests use this)."""
    requirement_rules = tuple(
        (rules or {}).get(r.output.label, r) for r in statics.unary.requirement_rules
    )
    unary = assemble_unary(statics.taxonomy, statics.unary.features, requirement_rules)
    return StaticWorld(
        statics.taxonomy,
        unary,
        statics.event_tree,
        statics.relations,
        statics.projections,
        statics.relation_stats,
        statics.warnings,
    )


__all__ = ["StaticWorld", "build_statics", "replace_requirements"]
