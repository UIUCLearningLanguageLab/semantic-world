"""Rules: sampling the global rule set, rule files, duplicate checks, the variance bound, and
evaluation of the determined features.

The rule set is global: one rule set holds for the whole world. Rules form an acyclic graph.
Free IS and HAS features are layer 0. A rule for a layer-k IS or HAS feature takes inputs from
layers below k, and at least one input comes from layer k−1. CAN rules read IS and HAS features
of any layer. No rule reads an ISA or CAN feature. Evaluation order is layer order, so every
input is known before a rule runs.

Every rule is stored as its input list and a truth table, and evaluated by table lookup,
vectorized over all objects with NumPy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.taxonomy.boolean import (
    TruthTable,
    apply_operator,
    dnf_literal_count,
    minimal_dnf,
    settings_array,
    shj_function,
    shj_type_of,
)
from semantic_world.taxonomy.config import Config, ConfigError, RuleSampling
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.expressions import Expr, Op, from_dnf, literal, random_read_once
from semantic_world.taxonomy.features import Feature, FeatureSet, build_features
from semantic_world.taxonomy.rule_files import ExplicitRule, RuleFile, Template, load_rule_file
from semantic_world.taxonomy.streams import Streams

MAX_TRIES = 1000
"""How many times a rule is resampled before the generator gives up on a feature."""

CONE_ENUMERATION_LIMIT = 20
"""Cones of at most this many free features are enumerated exactly; larger ones are sampled."""

MONTE_CARLO_SAMPLES = 1 << 16

RULE_FAMILIES = ("literal", "binary", "shj", "compositional", "fixed", "explicit")


# ---------------------------------------------------------------------------------------------
# Rules and rule sets
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One determined feature's rule: its inputs, truth table, and expression."""

    output: Feature
    inputs: tuple[Feature, ...]
    table: TruthTable
    expression: Expr
    family: str
    shj_type: str | None
    nesting_depth: int | None
    min_dnf_literals: int

    @property
    def arity(self) -> int:
        return len(self.inputs)

    @property
    def layer(self) -> int:
        return self.output.layer

    @property
    def relevant_inputs(self) -> tuple[Feature, ...]:
        return tuple(self.inputs[i] for i in self.table.relevant_inputs())

    @property
    def input_positions(self) -> np.ndarray:
        return np.array([f.position for f in self.inputs], dtype=np.intp)

    def record(self) -> dict[str, Any]:
        """The entry written to ``rules.yaml``."""
        return {
            "output": self.output.label,
            "layer": self.layer,
            "family": self.family,
            "shj_type": self.shj_type,
            "arity": self.arity,
            "nesting_depth": self.nesting_depth,
            "inputs": [f.label for f in self.inputs],
            "relevant_inputs": [f.label for f in self.relevant_inputs],
            "expression": str(self.expression),
            "truth_table": self.table.bit_string(),
            "min_dnf_literals": self.min_dnf_literals,
        }


def canonical_key(
    inputs: Sequence[Feature], table: TruthTable
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The function of the inputs in a form independent of input order: the inputs sorted by
    position, and the truth table over that order. Two rules with equal keys are identical
    columns."""
    order = sorted(range(len(inputs)), key=lambda i: inputs[i].position)
    new_index = {old: new for new, old in enumerate(order)}
    permutation = [new_index[i] for i in range(len(inputs))]
    return (tuple(inputs[i].position for i in order), table.permute_inputs(permutation).bits)


@dataclass(frozen=True)
class RuleSet:
    """The global rule set, in evaluation order, over a feature layout."""

    features: FeatureSet
    rules: tuple[Rule, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_output", {r.output.label: r for r in self.rules})

    def rule_for(self, feature: Feature | str) -> Rule:
        label = feature if isinstance(feature, str) else feature.label
        try:
            return self._by_output[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"{label} has no rule") from None

    def compute(self, free_values: np.ndarray) -> np.ndarray:
        """Compute every non-ISA feature from the free features.

        ``free_values`` has shape ``(n, free count)`` in free-feature order. The result has
        shape ``(n, feature count)`` in matrix order (IS, HAS, CAN), dtype ``uint8``.
        """
        return compute_features(self.features, self.rules, free_values)

    def cone(self, feature: Feature | str) -> tuple[Feature, ...]:
        """The free features a feature depends on, directly or through other determined
        features, in free-feature order."""
        return cone_of(
            feature if isinstance(feature, Feature) else self.features[feature], self._by_output
        )  # type: ignore[attr-defined]

    def records(self) -> list[dict[str, Any]]:
        return [rule.record() for rule in self.rules]

    def expected_true_proportion(self, rule: Rule | str) -> float:
        """See :func:`expected_true_proportion`. Sampling, when needed, uses a fixed seed."""
        if isinstance(rule, str):
            rule = self.rule_for(rule)
        return expected_true_proportion(
            self.features,
            self._by_output,
            rule,
            np.random.default_rng(0),  # type: ignore[attr-defined]
        )


def compute_features(
    features: FeatureSet, rules: Sequence[Rule], free_values: np.ndarray
) -> np.ndarray:
    values = np.asarray(free_values)
    free_positions = features.free_positions
    if values.ndim != 2 or values.shape[1] != len(free_positions):
        raise ValueError(
            f"expected an array of shape (n, {len(free_positions)}), got {values.shape}"
        )
    out = np.zeros((values.shape[0], len(features)), dtype=np.uint8)
    out[:, free_positions] = values.astype(np.uint8)
    for rule in rules:
        out[:, rule.output.position] = rule.table.evaluate(out[:, rule.input_positions])
    return out


def _stack(columns: list[np.ndarray], rows: int) -> np.ndarray:
    if not columns:
        return np.zeros((rows, 0), dtype=np.uint8)
    return np.stack(columns, axis=1)


def evaluate_feature(
    feature: Feature,
    free_values: np.ndarray,
    features: FeatureSet,
    rules_by_output: Mapping[str, Rule],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """One feature's column from a free-feature matrix, computing only the rules it needs."""
    if feature.free:
        return free_values[:, features.free.index(feature)]
    if feature.label not in cache:
        rule = rules_by_output[feature.label]
        columns = [
            evaluate_feature(f, free_values, features, rules_by_output, cache) for f in rule.inputs
        ]
        cache[feature.label] = rule.table.evaluate(_stack(columns, free_values.shape[0]))
    return cache[feature.label]


def expected_true_proportion(
    features: FeatureSet, rules_by_output: Mapping[str, Rule], rule: Rule, rng: np.random.Generator
) -> float:
    """The probability that the rule's output is 1 when every free feature is drawn
    independently at its base rate.

    The rule's cone (the free features it depends on, directly or through other determined
    features) is enumerated exactly when it has at most ``CONE_ENUMERATION_LIMIT`` features, and
    sampled with ``rng`` otherwise. ``rules_by_output`` must hold the rules of every determined
    input.
    """
    cone: dict[int, Feature] = {}
    for f in rule.inputs:
        for g in cone_of(f, rules_by_output):
            cone[g.position] = g
    cone_features = [cone[p] for p in sorted(cone)]
    rates = np.array([f.base_rate for f in cone_features], dtype=float)
    free = features.free
    columns = [free.index(f) for f in cone_features]
    if len(cone_features) <= CONE_ENUMERATION_LIMIT:
        grid = settings_array(len(cone_features))
        weights = np.prod(np.where(grid == 1, rates, 1 - rates), axis=1)
    else:
        grid = (rng.random((MONTE_CARLO_SAMPLES, len(cone_features))) < rates).astype(np.uint8)
        weights = np.full(grid.shape[0], 1 / grid.shape[0])
    free_values = np.zeros((grid.shape[0], len(free)), dtype=np.uint8)
    free_values[:, columns] = grid
    cache: dict[str, np.ndarray] = {}
    inputs = _stack(
        [evaluate_feature(f, free_values, features, rules_by_output, cache) for f in rule.inputs],
        grid.shape[0],
    )
    return float(np.sum(weights * rule.table.evaluate(inputs)))


def cone_of(feature: Feature, rules_by_output: Mapping[str, Rule]) -> tuple[Feature, ...]:
    found: dict[int, Feature] = {}
    stack = [feature]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current.label in seen:
            continue
        seen.add(current.label)
        if current.free:
            found[current.position] = current
        else:
            stack.extend(rules_by_output[current.label].inputs)
    return tuple(found[p] for p in sorted(found))


# ---------------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------------


def generate_rules(config: Config, streams: Streams) -> RuleSet:
    """Lay out the features and build the rule set. Draws from the ``rules`` stream (and the
    ``base_rates`` stream when base-rate heterogeneity is on)."""
    features = build_features(config, streams)
    return build_rules(config, features, streams)


def build_rules(config: Config, features: FeatureSet, streams: Streams) -> RuleSet:
    rule_file = None
    if config.rules.source == "file":
        path = config.rule_file_path()
        assert path is not None
        rule_file = load_rule_file(path)
    return _Builder(config, features, streams.rules, rule_file).build()


def _draw(rng: np.random.Generator, weights: Mapping[Any, float]) -> Any:
    keys = list(weights)
    values = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=values / values.sum()))]


class _Builder:
    def __init__(
        self,
        config: Config,
        features: FeatureSet,
        rng: np.random.Generator,
        rule_file: RuleFile | None,
    ) -> None:
        self.config = config
        self.features = features
        self.rng = rng
        self.rule_file = rule_file
        self.allow_duplicates = config.rules.allow_duplicate_rules
        self.variance_bound = config.rules.variance_bound
        self.rules: dict[str, Rule] = {}
        self.keys: dict[tuple, str] = {}
        self.warnings: list[str] = list(features.warnings)

    # Entry point -----------------------------------------------------------------------------

    def build(self) -> RuleSet:
        explicit = self._explicit_rules()
        self._check_templates()
        ordered: list[Rule] = []
        for feature in self.features.determined:
            rule = explicit.get(feature.label) or self._sample(feature)
            self.rules[feature.label] = rule
            ordered.append(rule)
        return RuleSet(self.features, tuple(ordered), tuple(self.warnings))

    # Explicit rules ----------------------------------------------------------------------------

    def _explicit_rules(self) -> dict[str, Rule]:
        if self.rule_file is None:
            return {}
        result: dict[str, Rule] = {}
        for explicit in self.rule_file.explicit:
            rule = self._explicit_rule(explicit)
            key = canonical_key(rule.inputs, rule.table)
            if key in self.keys and not self.allow_duplicates:
                raise ConfigError(
                    self.rule_file.source,
                    f"{explicit.field}.expression",
                    f"is the same function of the same inputs as the rule for {self.keys[key]}; "
                    f"set rules.allow_duplicate_rules to allow duplicate rules",
                )
            self.keys[key] = rule.output.label
            result[rule.output.label] = rule
        return result

    def _explicit_rule(self, explicit: ExplicitRule) -> Rule:
        assert self.rule_file is not None
        source = self.rule_file.source

        def error(key: str, message: str) -> ConfigError:
            return ConfigError(source, f"{explicit.field}.{key}", message)

        if explicit.output not in self.features:
            raise error("output", f"unknown feature {explicit.output}")
        output = self.features[explicit.output]
        if output.free:
            raise error(
                "output", f"{output.label} is a free feature; only determined features have rules"
            )
        inputs: list[Feature] = []
        for name in explicit.expression.variables():
            if name.startswith("ISA."):
                raise error("expression", f"reads {name}; no rule may read an ISA feature")
            if name not in self.features:
                raise error("expression", f"reads unknown feature {name}")
            feature = self.features[name]
            if feature.type == "can":
                raise error("expression", f"reads {name}; no rule may read a CAN feature")
            inputs.append(feature)
        if output.type != "can":
            above = [f.label for f in inputs if f.layer >= output.layer]
            if above:
                raise error(
                    "expression",
                    f"{output.label} is at layer {output.layer} but reads {above}, which are not "
                    f"below layer {output.layer}",
                )
            if not any(f.layer == output.layer - 1 for f in inputs):
                raise error(
                    "expression",
                    f"{output.label} is at layer {output.layer} but reads no feature from "
                    f"layer {output.layer - 1}",
                )
        labels = [f.label for f in inputs]
        table = explicit.expression.truth_table(labels)
        return Rule(
            output=output,
            inputs=tuple(inputs),
            table=table,
            expression=explicit.expression,
            family="explicit",
            shj_type=shj_type_of(table) if len(inputs) == 3 else None,
            nesting_depth=explicit.expression.depth(),
            min_dnf_literals=dnf_literal_count(table),
        )

    # Templates -----------------------------------------------------------------------------------

    def _check_templates(self) -> None:
        """Every template's arity must fit the smallest input pool of a type it applies to."""
        if self.rule_file is None:
            return
        for i, template in enumerate(self.rule_file.templates):
            if template.weight == 0:
                continue
            for feature_type in template.applies_to:
                outputs = [f for f in self.features.of_type(feature_type) if not f.free]
                if not outputs:
                    continue
                sampling = self.config.rules.sampling_for(feature_type)
                pool = self._pool(min(outputs, key=lambda f: f.layer), sampling)
                if template.max_arity > len(pool):
                    raise ConfigError(
                        self.rule_file.source,
                        f"templates[{i}].arity",
                        f"the template has arity {template.max_arity}, but a rule for a "
                        f"{feature_type.upper()} feature can read at most {len(pool)} features",
                    )

    # Sampling ------------------------------------------------------------------------------------

    def _sample(self, output: Feature) -> Rule:
        sampling = self.config.rules.sampling_for(output.type)
        duplicates = 0
        closest: tuple[float, float] | None = None
        for _ in range(MAX_TRIES):
            if self.rule_file is not None:
                rule = self._from_template(output, sampling)
            else:
                rule = self._automatic(output, sampling)
            key = canonical_key(rule.inputs, rule.table)
            if key in self.keys and not self.allow_duplicates:
                duplicates += 1
                continue
            if self.variance_bound is not None:
                low, high = self.variance_bound
                proportion = self.expected_true_proportion(rule)
                if not low <= proportion <= high:
                    distance = max(low - proportion, proportion - high)
                    if closest is None or distance < closest[0]:
                        closest = (distance, proportion)
                    continue
            self.keys[key] = output.label
            return rule
        reasons = []
        if duplicates:
            reasons.append(
                f"{duplicates} candidates duplicated an existing rule (set "
                f"rules.allow_duplicate_rules, or enlarge the input pool or the arity range)"
            )
        if closest is not None:
            low, high = self.variance_bound  # type: ignore[misc]
            reasons.append(
                f"the closest expected proportion of true outputs was {closest[1]:.4f}, outside "
                f"rules.variance_bound [{low}, {high}] (loosen the bound, or change the base rates "
                f"or the rule complexity)"
            )
        raise GenerationError(
            f"could not sample a rule for {output.label} in {MAX_TRIES} tries: "
            + "; ".join(reasons)
        )

    def _pool(self, output: Feature, sampling: RuleSampling) -> list[Feature]:
        """The features a rule for ``output`` may read, in feature order."""
        types = sampling.input_types
        if output.type == "can":
            return [f for f in self.features.features if f.type in types]
        return [f for f in self.features.features if f.type in types and f.layer < output.layer]

    def _choose_inputs(
        self, output: Feature, arity: int, sampling: RuleSampling
    ) -> tuple[Feature, ...]:
        pool = self._pool(output, sampling)
        weights = {t: sampling.input_type_weights[t] for t in sampling.input_types}
        if len(pool) < arity:
            raise GenerationError(
                f"a rule for {output.label} needs {arity} inputs but only {len(pool)} features are "
                f"eligible; lower the arity or loosen rules.input_type_weights"
            )
        chosen: list[Feature] = []
        if output.type != "can":
            previous = [f for f in pool if f.layer == output.layer - 1]
            if not previous:
                raise GenerationError(
                    f"a rule for {output.label} at layer {output.layer} needs an input from layer "
                    f"{output.layer - 1}, but no eligible feature is at that layer; loosen "
                    f"rules.input_type_weights or lower rules.max_chain_depth"
                )
            chosen.append(self._weighted_pick(previous, weights, 1)[0])
        remaining = [f for f in pool if f not in chosen]
        chosen.extend(self._weighted_pick(remaining, weights, arity - len(chosen)))
        return tuple(sorted(chosen, key=lambda f: f.position))

    def _weighted_pick(
        self, candidates: list[Feature], weights: Mapping[str, float], count: int
    ) -> list[Feature]:
        if count == 0:
            return []
        p = np.array([weights[f.type] for f in candidates], dtype=float)
        picks = self.rng.choice(len(candidates), size=count, replace=False, p=p / p.sum())
        return [candidates[int(i)] for i in picks]

    def _automatic(self, output: Feature, sampling: RuleSampling) -> Rule:
        arity = _draw(self.rng, sampling.arity)
        inputs = self._choose_inputs(output, arity, sampling)
        p = sampling.negation_probability
        if arity == 1:
            return self._literal(output, inputs, p)
        if arity == 2:
            return self._binary(output, inputs, _draw(self.rng, sampling.operator_mix), p, "binary")
        if arity == 3:
            family = _draw(self.rng, sampling.arity_3_families)
            if family != "compositional":
                return self._shj(output, inputs, family, p)
        depth = _draw(self.rng, sampling.nesting_depth)
        return self._compositional(output, inputs, depth, sampling.operator_mix, p)

    def _from_template(self, output: Feature, sampling: RuleSampling) -> Rule:
        assert self.rule_file is not None
        templates = self.rule_file.templates_for(output.type)
        if not templates or all(t.weight == 0 for t in templates):
            raise ConfigError(
                self.rule_file.source,
                "templates",
                f"no template with positive weight applies to {output.type.upper()} features, "
                f"and {output.label} has no explicit rule",
            )
        template: Template = templates[
            _draw(self.rng, {i: t.weight for i, t in enumerate(templates)})
        ]
        p = (
            sampling.negation_probability
            if template.negation_probability is None
            else template.negation_probability
        )
        inputs = self._choose_inputs(output, template.max_arity, sampling)
        if template.family == "literal":
            return self._literal(output, inputs, p)
        if template.family == "shj":
            assert template.shj_type is not None
            return self._shj(output, inputs, template.shj_type, p)
        if template.family == "fixed":
            assert template.operator is not None
            if len(inputs) == 2:
                return self._binary(output, inputs, template.operator, p, "fixed")
            return self._flat(output, inputs, template.operator, p)
        operators = template.operators or sampling.operator_mix
        depth = _draw(self.rng, template.nesting_depth or sampling.nesting_depth)
        return self._compositional(output, inputs, depth, operators, p)

    # Rule constructors ---------------------------------------------------------------------------

    def _finish(
        self,
        output: Feature,
        inputs: tuple[Feature, ...],
        expression: Expr,
        table: TruthTable,
        family: str,
        shj_type: str | None,
        nesting_depth: int | None,
    ) -> Rule:
        return Rule(
            output,
            inputs,
            table,
            expression,
            family,
            shj_type,
            nesting_depth,
            dnf_literal_count(table),
        )

    def _literal(self, output: Feature, inputs: tuple[Feature, ...], p: float) -> Rule:
        negated = bool(self.rng.random() < p)
        expression = literal(inputs[0].label, negated)
        table = TruthTable.from_function(1, lambda x: x[0] ^ int(negated))
        return self._finish(output, inputs, expression, table, "literal", None, None)

    def _binary(
        self, output: Feature, inputs: tuple[Feature, ...], operator: str, p: float, family: str
    ) -> Rule:
        negated = [bool(v) for v in self.rng.random(2) < p]
        expression = Op(
            operator, tuple(literal(f.label, n) for f, n in zip(inputs, negated, strict=True))
        )
        table = TruthTable.from_function(
            2,
            lambda x: apply_operator(
                operator, [v ^ int(n) for v, n in zip(x, negated, strict=True)]
            ),
        )
        return self._finish(output, inputs, expression, table, family, None, None)

    def _flat(self, output: Feature, inputs: tuple[Feature, ...], operator: str, p: float) -> Rule:
        negated = [bool(v) for v in self.rng.random(len(inputs)) < p]
        expression = Op(
            operator, tuple(literal(f.label, n) for f, n in zip(inputs, negated, strict=True))
        )
        table = expression.truth_table([f.label for f in inputs])
        return self._finish(output, inputs, expression, table, "fixed", None, 1)

    def _shj(self, output: Feature, inputs: tuple[Feature, ...], shj_type: str, p: float) -> Rule:
        permutation = [int(i) for i in self.rng.permutation(3)]
        negations = [bool(v) for v in self.rng.random(3) < p]
        table = shj_function(shj_type, permutation, negations)
        expression = from_dnf(minimal_dnf(table), [f.label for f in inputs])
        return self._finish(output, inputs, expression, table, "shj", shj_type, None)

    def _compositional(
        self,
        output: Feature,
        inputs: tuple[Feature, ...],
        depth: int,
        operators: Mapping[str, float],
        p: float,
    ) -> Rule:
        labels = [f.label for f in inputs]
        expression = random_read_once(labels, depth, operators, p, self.rng)
        table = expression.truth_table(labels)
        return self._finish(
            output, inputs, expression, table, "compositional", None, expression.depth()
        )

    # The variance bound ------------------------------------------------------------------------

    def expected_true_proportion(self, rule: Rule) -> float:
        return expected_true_proportion(self.features, self.rules, rule, self.rng)
