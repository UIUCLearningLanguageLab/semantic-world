"""One-place event types: their requirements, sampled as the taxonomy sampled CAN rules.

A one-place event type ``EVENTTYPE1.<k>`` has a requirement over the agent's static facts: a rule
over PROPERTY and PART features of any layer and threshold literals on scalars, drawn with the
settings of ``event_types.unary.rules`` from the ``world:requirements`` stream
(``docs/specs/WORLD_AND_LANGUAGE.md``, "Event types"). The rule is sampled exactly as the taxonomy
samples a determined feature's rule, except that its pool is every PROPERTY and PART feature,
and it is never a duplicate of a taxonomy rule or of another requirement.

The requirements are kept as rules over an extended feature table: the taxonomy's features
followed by one feature per one-place event type, so that the taxonomy's machinery (evaluation,
cones, the fixed-by-rule test behind ``capacity_roles.csv`` and the corpus's ``NEC`` statements)
runs unchanged over them. The extra features are never written by the taxonomy; the world writes
their columns as the one-place capacities in ``derived/capacities.csv``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from semantic_world.common.boolean import dnf_literal_count
from semantic_world.taxonomy.config import RuleSampling
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.features import Feature, FeatureSet
from semantic_world.taxonomy.fixed import FIXED_NONE, fixed_by_rule
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.rules import (
    MAX_TRIES,
    Input,
    Rule,
    RuleSet,
    Threshold,
    _draw,
    build_function,
    canonical_key,
    expected_true_proportion,
    input_atom,
    input_key,
    model_quantile_threshold,
)
from semantic_world.world.config import UnaryConfig

ONE_PLACE = "EVENTTYPE1"
EVENT_TYPE1_TYPE = "event_type1"
"""The feature type of a one-place event type in the extended feature table."""


@dataclass(frozen=True)
class UnaryRequirements:
    """The one-place event types over the taxonomy's static facts."""

    features: FeatureSet
    """The taxonomy's features followed by one feature per one-place event type."""
    rules: RuleSet
    """The taxonomy's rules followed by the requirements, in event-type order."""
    values: np.ndarray
    """Every entity's static features and one-place capacities, shape ``(entities,
    features)``, uint8, by feature position."""
    category_values: np.ndarray
    """Every category's generative vector with its one-place capacities, in category order."""
    fixed: np.ndarray
    """Shape ``(categories, event types)``, bool: whether the capacity is fixed by rule at the
    category (the fixed-by-rule test over the extended rules)."""
    fixed_test: np.ndarray
    """Shape ``(categories, event types)``, int8: which test decided (``FIXED_EXACT``,
    ``FIXED_LOCAL``, or ``FIXED_NONE``)."""

    @property
    def event_features(self) -> tuple[Feature, ...]:
        return tuple(f for f in self.features.features if f.type == EVENT_TYPE1_TYPE)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.event_features)

    @property
    def requirement_rules(self) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules.rules if r.output.type == EVENT_TYPE1_TYPE)

    def rule_for(self, label: str) -> Rule:
        return self.rules.rule_for(label)

    def column(self, label: str) -> np.ndarray:
        """The one-place capacity of every entity."""
        return self.values[:, self.features[label].position]


def extended_features(taxonomy: TaxonomyResult, count: int) -> FeatureSet:
    """The taxonomy's feature table with ``count`` one-place event types appended, each a
    determined feature of type ``event_type1`` at the layer above the taxonomy's last."""
    base = taxonomy.features
    layer = base.layers + 1
    position = len(base.features)
    extra = tuple(
        Feature(
            label=f"{ONE_PLACE}.{k}",
            type=EVENT_TYPE1_TYPE,
            index=k,
            free=False,
            layer=layer,
            base_rate=None,
            position=position + k - 1,
        )
        for k in range(1, count + 1)
    )
    return FeatureSet(
        base.features + extra,
        max_chain_depth=base.max_chain_depth,
        warnings=(),
        scalar_count=base.scalar_count,
    )


class _RequirementSampler:
    """Samples one-place requirements as the taxonomy's rule builder sampled CAN rules."""

    def __init__(
        self,
        taxonomy: TaxonomyResult,
        features: FeatureSet,
        sampling: RuleSampling,
        rng: np.random.Generator,
    ) -> None:
        self.taxonomy = taxonomy
        self.features = features
        self.sampling = sampling
        self.rng = rng
        self.scalars = taxonomy.config.scalars
        self.allow_duplicates = taxonomy.config.rules.allow_duplicate_rules
        self.variance_bound = taxonomy.config.rules.variance_bound
        self.rules: dict[str, Rule] = {r.output.label: r for r in taxonomy.rules.rules}
        self.keys: dict[tuple, str] = {
            canonical_key(r.inputs, r.table): r.output.label for r in taxonomy.rules.rules
        }
        types = sampling.input_types
        self.pool: list[Feature | int] = [f for f in features.features if f.type in types]
        if self.scalars.count and sampling.scalar_weight > 0:
            self.pool.extend(range(1, self.scalars.count + 1))

    def sample(self, output: Feature) -> Rule:
        duplicates = 0
        closest: tuple[float, float] | None = None
        for _ in range(MAX_TRIES):
            rule = self._automatic(output)
            key = canonical_key(rule.inputs, rule.table)
            if key in self.keys and not self.allow_duplicates:
                duplicates += 1
                continue
            if self.variance_bound is not None:
                low, high = self.variance_bound
                proportion = expected_true_proportion(self.features, self.rules, rule, self.rng)
                if not low <= proportion <= high:
                    distance = max(low - proportion, proportion - high)
                    if closest is None or distance < closest[0]:
                        closest = (distance, proportion)
                    continue
            self.keys[key] = output.label
            self.rules[output.label] = rule
            return rule
        reasons = []
        if duplicates:
            reasons.append(
                f"{duplicates} candidates duplicated an existing rule (set "
                f"rules.allow_duplicate_rules in the taxonomy configuration, or enlarge the "
                f"arity range of event_types.unary.rules)"
            )
        if closest is not None:
            low, high = self.variance_bound  # type: ignore[misc]
            reasons.append(
                f"the closest expected proportion of true outputs was {closest[1]:.4f}, outside "
                f"rules.variance_bound [{low}, {high}]"
            )
        raise GenerationError(
            f"could not sample a requirement for {output.label} in {MAX_TRIES} tries: "
            + "; ".join(reasons)
        )

    def _automatic(self, output: Feature) -> Rule:
        arity = _draw(self.rng, self.sampling.arity)
        if len(self.pool) < arity:
            raise GenerationError(
                f"a requirement for {output.label} needs {arity} inputs but only "
                f"{len(self.pool)} inputs are eligible; lower the arity in "
                f"event_types.unary.rules or loosen its input_type_weights"
            )
        p = np.array(
            [
                self.sampling.scalar_weight
                if isinstance(c, int)
                else self.sampling.input_type_weights[c.type]
                for c in self.pool
            ],
            dtype=float,
        )
        picks = self.rng.choice(len(self.pool), size=arity, replace=False, p=p / p.sum())
        chosen = [self.pool[int(i)] for i in picks]
        binary = sorted((c for c in chosen if isinstance(c, Feature)), key=lambda f: f.position)
        thresholds = [
            self._draw_threshold(n) for n in sorted(c for c in chosen if isinstance(c, int))
        ]
        inputs: tuple[Input, ...] = tuple(binary) + tuple(thresholds)
        draw = build_function(
            self.rng,
            self.sampling,
            [input_atom(i) for i in inputs],
            [input_key(i) for i in inputs],
        )
        return Rule(
            output,
            inputs,
            draw.table,
            draw.expression,
            draw.family,
            draw.shj_type,
            draw.nesting_depth,
            dnf_literal_count(draw.table),
        )

    def _draw_threshold(self, scalar: int) -> Threshold:
        low, high = self.scalars.threshold_quantiles
        quantile = float(self.rng.uniform(low, high))
        return Threshold(scalar, model_quantile_threshold(self.scalars, quantile), quantile)


def generate_unary(
    taxonomy: TaxonomyResult,
    settings: UnaryConfig,
    rng: np.random.Generator,
    explicit: Mapping[str, Rule] | None = None,
) -> UnaryRequirements:
    """Sample every one-place requirement from ``rng`` (the ``world:requirements`` stream),
    then replace the ones given explicitly (``explicit`` maps an event-type label to its rule
    over the extended features), so that an explicit entry changes no other event type's draw.
    Compute every entity's and every category's capacities, and the fixed-by-rule test."""
    features = extended_features(taxonomy, settings.count)
    sampler = _RequirementSampler(taxonomy, features, settings.rules, rng)
    sampled: list[Rule] = [
        sampler.sample(f) for f in features.features if f.type == EVENT_TYPE1_TYPE
    ]
    rules = [explicit.get(r.output.label, r) if explicit else r for r in sampled]
    return assemble_unary(taxonomy, features, tuple(rules))


def assemble_unary(
    taxonomy: TaxonomyResult, features: FeatureSet, requirement_rules: tuple[Rule, ...]
) -> UnaryRequirements:
    """The requirements over the extended features, with the capacities computed."""
    rule_set = RuleSet(features, taxonomy.rules.rules + requirement_rules, ())
    instances = taxonomy.instances
    scalars = instances.scalars if features.scalar_count else None
    values = rule_set.compute(instances.free_values(taxonomy.features), scalars)
    tree = taxonomy.tree
    category_scalars = tree.scalar_matrix() if features.scalar_count else None
    category_values = rule_set.compute(tree.free_matrix(), category_scalars)
    event_features = [f for f in features.features if f.type == EVENT_TYPE1_TYPE]
    fixed = np.zeros((len(tree.categories), len(event_features)), dtype=bool)
    fixed_test = np.full((len(tree.categories), len(event_features)), FIXED_NONE, dtype=np.int8)
    positions = [f.position for f in event_features]
    for ci, category in enumerate(tree.categories):
        stand_in = dataclasses.replace(category, values=category_values[ci])
        all_fixed, all_tests = fixed_by_rule(rule_set, stand_in, scalars=taxonomy.config.scalars)
        fixed[ci] = all_fixed[positions]
        fixed_test[ci] = all_tests[positions]
    return UnaryRequirements(features, rule_set, values, category_values, fixed, fixed_test)


__all__ = [
    "EVENT_TYPE1_TYPE",
    "ONE_PLACE",
    "UnaryRequirements",
    "assemble_unary",
    "extended_features",
    "generate_unary",
]
