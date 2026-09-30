"""Rules: sampling the global rule set, rule files, duplicate checks, the variance bound, and
evaluation of the determined features.

The rule set is global: one rule set holds for the whole world. Rules form an acyclic graph.
Free IS and HAS features are layer 0. A rule for a layer-k IS or HAS feature takes inputs from
layers below k, and at least one input comes from layer k−1. CAN rules read IS and HAS features
of any layer. No rule reads an ISA or CAN feature. Evaluation order is layer order, so every
input is known before a rule runs.

A rule's input can also be a threshold literal on a scalar dimension, ``SC.<n> > threshold``,
which acts like a binary layer-0 input (``docs/specs/TAXONOMY_RELATIONS.md``, Part A). One rule
uses at most one literal per scalar. Thresholds are quantiles of the scalar's model distribution,
drawn from the ``rules`` stream and rounded to 4 decimal places.

Every rule is stored as its input list and a truth table, and evaluated by table lookup,
vectorized over all objects with NumPy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
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
from semantic_world.taxonomy.config import Config, ConfigError, RuleSampling, ScalarsConfig
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.expressions import (
    Expr,
    Gt,
    Op,
    Var,
    from_dnf,
    literal,
    random_read_once,
)
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
class Threshold:
    """A threshold literal ``SC.<scalar> > threshold`` used as a rule input. It is a layer-0
    input, always free, of type ``scalar``."""

    scalar: int
    """The 1-based index of the scalar dimension."""
    threshold: float
    quantile: float
    """The threshold's quantile in the scalar's model distribution: the literal is true with
    probability ``1 - quantile`` for an independent instance."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", round(float(self.threshold), 4))

    @property
    def label(self) -> str:
        return f"SC.{self.scalar}"

    @property
    def key(self) -> str:
        return f"{self.label}>{self.threshold:.4f}"

    free = True
    layer = 0
    type = "scalar"

    def atom(self) -> Gt:
        return Gt(self.label, self.threshold)

    def values(self, scalar_values: np.ndarray) -> np.ndarray:
        """The literal's value for every row of a scalar matrix."""
        return (np.asarray(scalar_values)[:, self.scalar - 1] > self.threshold).astype(np.uint8)


Input = Feature | Threshold


def input_key(item: Input) -> str:
    """The variable name of an input in expressions and truth tables."""
    return item.key if isinstance(item, Threshold) else item.label


def input_atom(item: Input) -> Expr:
    return item.atom() if isinstance(item, Threshold) else Var(item.label)


def input_sort_key(item: Input) -> tuple[int, int, float]:
    """Binary inputs first, by position; then threshold literals, by scalar and threshold."""
    if isinstance(item, Threshold):
        return (1, item.scalar, item.threshold)
    return (0, item.position, 0.0)


@dataclass(frozen=True)
class Rule:
    """One determined feature's rule: its inputs, truth table, and expression."""

    output: Feature
    inputs: tuple[Input, ...]
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
    def relevant_inputs(self) -> tuple[Input, ...]:
        return tuple(self.inputs[i] for i in self.table.relevant_inputs())

    @property
    def thresholds(self) -> tuple[Threshold, ...]:
        return tuple(i for i in self.inputs if isinstance(i, Threshold))

    @property
    def input_positions(self) -> np.ndarray:
        """The matrix columns of the binary inputs (threshold literals have no column)."""
        return np.array([f.position for f in self.inputs if isinstance(f, Feature)], dtype=np.intp)

    def input_matrix(self, values: np.ndarray, scalar_values: np.ndarray | None) -> np.ndarray:
        """The inputs' columns for every row of a feature matrix (and scalar matrix)."""
        columns = []
        for item in self.inputs:
            if isinstance(item, Threshold):
                if scalar_values is None:
                    raise ValueError(f"the rule for {self.output.label} needs scalar values")
                columns.append(item.values(scalar_values))
            else:
                columns.append(values[:, item.position])
        return _stack(columns, values.shape[0])

    def record(self, include_thresholds: bool = False) -> dict[str, Any]:
        """The entry written to ``rules.yaml``. With ``include_thresholds``, the entry lists
        its threshold literals with their thresholds (empty when it has none)."""
        entry = {
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
        if include_thresholds:
            entry["thresholds"] = {t.label: t.threshold for t in self.thresholds}
        return entry


def canonical_key(inputs: Sequence[Input], table: TruthTable) -> tuple[tuple, tuple[int, ...]]:
    """The function of the inputs in a form independent of input order: the inputs sorted (binary
    inputs by position, then threshold literals by scalar and threshold), and the truth table
    over that order. Two rules with equal keys are identical columns."""
    order = sorted(range(len(inputs)), key=lambda i: input_sort_key(inputs[i]))
    new_index = {old: new for new, old in enumerate(order)}
    permutation = [new_index[i] for i in range(len(inputs))]
    return (
        tuple(input_sort_key(inputs[i]) for i in order),
        table.permute_inputs(permutation).bits,
    )


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

    def compute(
        self, free_values: np.ndarray, scalar_values: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute every non-ISA feature from the free features (and the scalar values, when
        any rule reads a threshold literal).

        ``free_values`` has shape ``(n, free count)`` in free-feature order and ``scalar_values``
        shape ``(n, scalar count)``. The result has shape ``(n, feature count)`` in matrix order
        (IS, HAS, CAN), dtype ``uint8``.
        """
        return compute_features(self.features, self.rules, free_values, scalar_values)

    def cone(self, feature: Feature | str) -> tuple[Feature, ...]:
        """The free binary features a feature depends on, directly or through other determined
        features, in free-feature order."""
        return cone_of(
            feature if isinstance(feature, Feature) else self.features[feature], self._by_output
        )  # type: ignore[attr-defined]

    def thresholds_of(self, feature: Feature | str) -> tuple[Threshold, ...]:
        """The threshold literals a feature depends on, directly or through other determined
        features, by scalar and threshold."""
        return threshold_cone_of(
            feature if isinstance(feature, Feature) else self.features[feature], self._by_output
        )  # type: ignore[attr-defined]

    @property
    def has_thresholds(self) -> bool:
        return any(rule.thresholds for rule in self.rules)

    def records(self) -> list[dict[str, Any]]:
        """The ``rules.yaml`` entries. Threshold lists are included when the run has scalars."""
        include = self.features.scalar_count > 0
        return [rule.record(include_thresholds=include) for rule in self.rules]

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
    features: FeatureSet,
    rules: Sequence[Rule],
    free_values: np.ndarray,
    scalar_values: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(free_values)
    free_positions = features.free_positions
    if values.ndim != 2 or values.shape[1] != len(free_positions):
        raise ValueError(
            f"expected an array of shape (n, {len(free_positions)}), got {values.shape}"
        )
    if scalar_values is not None:
        scalar_values = np.asarray(scalar_values, dtype=float)
        if scalar_values.shape != (values.shape[0], features.scalar_count):
            raise ValueError(
                f"expected scalar values of shape ({values.shape[0]}, {features.scalar_count}), "
                f"got {scalar_values.shape}"
            )
    out = np.zeros((values.shape[0], len(features)), dtype=np.uint8)
    out[:, free_positions] = values.astype(np.uint8)
    for rule in rules:
        out[:, rule.output.position] = rule.table.evaluate(rule.input_matrix(out, scalar_values))
    return out


def _stack(columns: list[np.ndarray], rows: int) -> np.ndarray:
    if not columns:
        return np.zeros((rows, 0), dtype=np.uint8)
    return np.stack(columns, axis=1)


def evaluate_feature(
    feature: Input,
    free_values: np.ndarray,
    features: FeatureSet,
    rules_by_output: Mapping[str, Rule],
    cache: dict[str, np.ndarray],
    literal_values: Mapping[str, np.ndarray] | None = None,
    scalar_values: np.ndarray | None = None,
) -> np.ndarray:
    """One feature's column from a free-feature matrix, computing only the rules it needs.

    A threshold literal takes its column from ``literal_values`` (by its key) when given, and
    otherwise from ``scalar_values``.
    """
    if isinstance(feature, Threshold):
        if literal_values is not None and feature.key in literal_values:
            return np.asarray(literal_values[feature.key], dtype=np.uint8)
        if scalar_values is None:
            raise ValueError(f"no values for the threshold literal {feature.key}")
        return feature.values(scalar_values)
    if feature.free:
        return free_values[:, features.free.index(feature)]
    if feature.label not in cache:
        rule = rules_by_output[feature.label]
        columns = [
            evaluate_feature(
                f, free_values, features, rules_by_output, cache, literal_values, scalar_values
            )
            for f in rule.inputs
        ]
        cache[feature.label] = rule.table.evaluate(_stack(columns, free_values.shape[0]))
    return cache[feature.label]


def expected_true_proportion(
    features: FeatureSet, rules_by_output: Mapping[str, Rule], rule: Rule, rng: np.random.Generator
) -> float:
    """The probability that the rule's output is 1 when every free feature is drawn
    independently at its base rate, and every threshold literal is an independent input that is
    true with probability ``1 - quantile``.

    The rule's cone (the free features and threshold literals it depends on, directly or through
    other determined features) is enumerated exactly when it has at most
    ``CONE_ENUMERATION_LIMIT`` members, and sampled with ``rng`` otherwise. ``rules_by_output``
    must hold the rules of every determined input.
    """
    cone: dict[int, Feature] = {}
    literals: dict[str, Threshold] = {}
    for f in rule.inputs:
        for g in cone_of(f, rules_by_output):
            cone[g.position] = g
        for t in threshold_cone_of(f, rules_by_output):
            literals[t.key] = t
    cone_features = [cone[p] for p in sorted(cone)]
    thresholds = sorted(literals.values(), key=input_sort_key)
    rates = np.array(
        [f.base_rate for f in cone_features] + [1.0 - t.quantile for t in thresholds], dtype=float
    )
    free = features.free
    columns = [free.index(f) for f in cone_features]
    size = len(cone_features) + len(thresholds)
    if size <= CONE_ENUMERATION_LIMIT:
        grid = settings_array(size)
        weights = np.prod(np.where(grid == 1, rates, 1 - rates), axis=1)
    else:
        grid = (rng.random((MONTE_CARLO_SAMPLES, size)) < rates).astype(np.uint8)
        weights = np.full(grid.shape[0], 1 / grid.shape[0])
    free_values = np.zeros((grid.shape[0], len(free)), dtype=np.uint8)
    free_values[:, columns] = grid[:, : len(cone_features)]
    literal_values = {t.key: grid[:, len(cone_features) + j] for j, t in enumerate(thresholds)}
    cache: dict[str, np.ndarray] = {}
    inputs = _stack(
        [
            evaluate_feature(f, free_values, features, rules_by_output, cache, literal_values)
            for f in rule.inputs
        ],
        grid.shape[0],
    )
    return float(np.sum(weights * rule.table.evaluate(inputs)))


def cone_of(feature: Input, rules_by_output: Mapping[str, Rule]) -> tuple[Feature, ...]:
    """The free binary features a feature depends on, transitively, in free-feature order."""
    found: dict[int, Feature] = {}
    stack: list[Input] = [feature]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, Threshold):
            continue
        if current.label in seen:
            continue
        seen.add(current.label)
        if current.free:
            found[current.position] = current
        else:
            stack.extend(rules_by_output[current.label].inputs)
    return tuple(found[p] for p in sorted(found))


def threshold_cone_of(feature: Input, rules_by_output: Mapping[str, Rule]) -> tuple[Threshold, ...]:
    """The threshold literals a feature depends on, transitively, by scalar and threshold."""
    found: dict[str, Threshold] = {}
    stack: list[Input] = [feature]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, Threshold):
            found[current.key] = current
        elif current.label not in seen:
            seen.add(current.label)
            if not current.free:
                stack.extend(rules_by_output[current.label].inputs)
    return tuple(sorted(found.values(), key=input_sort_key))


def model_quantile_threshold(scalars: ScalarsConfig, quantile: float) -> float:
    """The threshold at a quantile of the scalar model distribution, rounded to 4 decimals."""
    return round(NormalDist(0.0, scalars.model_std).inv_cdf(quantile), 4)


def model_quantile_of(scalars: ScalarsConfig, threshold: float) -> float:
    """The quantile of a threshold in the scalar model distribution."""
    return NormalDist(0.0, scalars.model_std).cdf(threshold)


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


@dataclass(frozen=True)
class FunctionDraw:
    """A sampled Boolean function over given atoms: what the rule and constraint samplers share."""

    expression: Expr
    table: TruthTable
    family: str
    shj_type: str | None
    nesting_depth: int | None


def build_function(
    rng: np.random.Generator, sampling: RuleSampling, atoms: Sequence[Expr], keys: Sequence[str]
) -> FunctionDraw:
    """Sample a function of the given atoms by their number: a literal, a binary function with
    an operator from the mix, an SHJ type or a compositional formula for three, and a read-once
    formula for more. Each input is negated with the negation probability."""
    p = sampling.negation_probability
    n = len(atoms)
    if n == 1:
        negated = bool(rng.random() < p)
        return FunctionDraw(
            literal(atoms[0], negated),
            TruthTable.from_function(1, lambda x: x[0] ^ int(negated)),
            "literal",
            None,
            None,
        )
    if n == 2:
        operator = _draw(rng, sampling.operator_mix)
        negated = [bool(v) for v in rng.random(2) < p]
        expression = Op(
            operator, tuple(literal(a, ng) for a, ng in zip(atoms, negated, strict=True))
        )
        table = TruthTable.from_function(
            2,
            lambda x: apply_operator(
                operator, [v ^ int(ng) for v, ng in zip(x, negated, strict=True)]
            ),
        )
        return FunctionDraw(expression, table, "binary", None, None)
    if n == 3:
        family = _draw(rng, sampling.arity_3_families)
        if family != "compositional":
            permutation = [int(i) for i in rng.permutation(3)]
            negations = [bool(v) for v in rng.random(3) < p]
            table = shj_function(family, permutation, negations)
            return FunctionDraw(from_dnf(minimal_dnf(table), atoms), table, "shj", family, None)
    depth = _draw(rng, sampling.nesting_depth)
    expression = random_read_once(atoms, depth, sampling.operator_mix, p, rng)
    return FunctionDraw(
        expression, expression.truth_table(keys), "compositional", None, expression.depth()
    )


def _entry_layer(entry: Feature | int) -> int:
    """The layer of a pool entry: a scalar dimension is a layer-0 input."""
    return 0 if isinstance(entry, int) else entry.layer


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
        self.scalars = config.scalars

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
        inputs: list[Input] = []
        for atom in explicit.expression.atoms():
            if isinstance(atom, Gt):
                inputs.append(self._explicit_threshold(atom, error))
                continue
            name = atom.name
            if name.startswith("ISA."):
                raise error("expression", f"reads {name}; no rule may read an ISA feature")
            if name.startswith("SC."):
                raise error(
                    "expression",
                    f"reads the scalar {name} without a threshold; write {name} > x or {name} <= x",
                )
            if name not in self.features:
                raise error("expression", f"reads unknown feature {name}")
            feature = self.features[name]
            if feature.type == "can":
                raise error("expression", f"reads {name}; no rule may read a CAN feature")
            inputs.append(feature)
        scalars_used = [i.scalar for i in inputs if isinstance(i, Threshold)]
        if len(set(scalars_used)) != len(scalars_used):
            raise error("expression", "uses more than one threshold literal on the same scalar")
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
        labels = [input_key(i) for i in inputs]
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

    def _explicit_threshold(self, atom: Gt, error) -> Threshold:
        prefix, _, number = atom.scalar.partition(".")
        if prefix != "SC" or not number.isdigit() or not 1 <= int(number) <= self.scalars.count:
            raise error(
                "expression",
                f"reads unknown scalar {atom.scalar}; the run has {self.scalars.count} scalars",
            )
        return Threshold(
            int(number), atom.threshold, model_quantile_of(self.scalars, atom.threshold)
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

    def _pool(self, output: Feature, sampling: RuleSampling) -> list[Feature | int]:
        """The inputs a rule for ``output`` may read: features in feature order, then the
        scalar dimensions (as indices) when scalars are on with a positive weight."""
        types = sampling.input_types
        if output.type == "can":
            pool: list[Feature | int] = [f for f in self.features.features if f.type in types]
        else:
            pool = [f for f in self.features.features if f.type in types and f.layer < output.layer]
        if self.scalars.count and sampling.scalar_weight > 0:
            pool.extend(range(1, self.scalars.count + 1))
        return pool

    def _choose_inputs(
        self, output: Feature, arity: int, sampling: RuleSampling
    ) -> tuple[Input, ...]:
        pool = self._pool(output, sampling)
        if len(pool) < arity:
            raise GenerationError(
                f"a rule for {output.label} needs {arity} inputs but only {len(pool)} inputs are "
                f"eligible; lower the arity or loosen rules.input_type_weights"
            )
        chosen: list[Feature | int] = []
        if output.type != "can":
            previous = [c for c in pool if _entry_layer(c) == output.layer - 1]
            if not previous:
                raise GenerationError(
                    f"a rule for {output.label} at layer {output.layer} needs an input from layer "
                    f"{output.layer - 1}, but no eligible feature is at that layer; loosen "
                    f"rules.input_type_weights or lower rules.max_chain_depth"
                )
            chosen.append(self._weighted_pick(previous, sampling, 1)[0])
        remaining = [c for c in pool if c not in chosen]
        chosen.extend(self._weighted_pick(remaining, sampling, arity - len(chosen)))
        binary = sorted((c for c in chosen if isinstance(c, Feature)), key=lambda f: f.position)
        thresholds = [
            self._draw_threshold(n) for n in sorted(c for c in chosen if isinstance(c, int))
        ]
        return tuple(binary) + tuple(thresholds)

    def _draw_threshold(self, scalar: int) -> Threshold:
        low, high = self.scalars.threshold_quantiles
        quantile = float(self.rng.uniform(low, high))
        return Threshold(scalar, model_quantile_threshold(self.scalars, quantile), quantile)

    def _weighted_pick(
        self, candidates: list[Feature | int], sampling: RuleSampling, count: int
    ) -> list[Feature | int]:
        if count == 0:
            return []
        p = np.array(
            [
                sampling.scalar_weight
                if isinstance(c, int)
                else sampling.input_type_weights[c.type]
                for c in candidates
            ],
            dtype=float,
        )
        picks = self.rng.choice(len(candidates), size=count, replace=False, p=p / p.sum())
        return [candidates[int(i)] for i in picks]

    def _automatic(self, output: Feature, sampling: RuleSampling) -> Rule:
        arity = _draw(self.rng, sampling.arity)
        inputs = self._choose_inputs(output, arity, sampling)
        draw = build_function(
            self.rng, sampling, [input_atom(i) for i in inputs], [input_key(i) for i in inputs]
        )
        return self._finish(
            output,
            inputs,
            draw.expression,
            draw.table,
            draw.family,
            draw.shj_type,
            draw.nesting_depth,
        )

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
        inputs: tuple[Input, ...],
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

    def _literal(self, output: Feature, inputs: tuple[Input, ...], p: float) -> Rule:
        negated = bool(self.rng.random() < p)
        expression = literal(input_atom(inputs[0]), negated)
        table = TruthTable.from_function(1, lambda x: x[0] ^ int(negated))
        return self._finish(output, inputs, expression, table, "literal", None, None)

    def _binary(
        self, output: Feature, inputs: tuple[Input, ...], operator: str, p: float, family: str
    ) -> Rule:
        negated = [bool(v) for v in self.rng.random(2) < p]
        expression = Op(
            operator,
            tuple(literal(input_atom(f), n) for f, n in zip(inputs, negated, strict=True)),
        )
        table = TruthTable.from_function(
            2,
            lambda x: apply_operator(
                operator, [v ^ int(n) for v, n in zip(x, negated, strict=True)]
            ),
        )
        return self._finish(output, inputs, expression, table, family, None, None)

    def _flat(self, output: Feature, inputs: tuple[Input, ...], operator: str, p: float) -> Rule:
        negated = [bool(v) for v in self.rng.random(len(inputs)) < p]
        expression = Op(
            operator,
            tuple(literal(input_atom(f), n) for f, n in zip(inputs, negated, strict=True)),
        )
        table = expression.truth_table([input_key(f) for f in inputs])
        return self._finish(output, inputs, expression, table, "fixed", None, 1)

    def _shj(self, output: Feature, inputs: tuple[Input, ...], shj_type: str, p: float) -> Rule:
        permutation = [int(i) for i in self.rng.permutation(3)]
        negations = [bool(v) for v in self.rng.random(3) < p]
        table = shj_function(shj_type, permutation, negations)
        expression = from_dnf(minimal_dnf(table), [input_atom(f) for f in inputs])
        return self._finish(output, inputs, expression, table, "shj", shj_type, None)

    def _compositional(
        self,
        output: Feature,
        inputs: tuple[Input, ...],
        depth: int,
        operators: Mapping[str, float],
        p: float,
    ) -> Rule:
        atoms = [input_atom(f) for f in inputs]
        expression = random_read_once(atoms, depth, operators, p, self.rng)
        table = expression.truth_table([input_key(f) for f in inputs])
        return self._finish(
            output, inputs, expression, table, "compositional", None, expression.depth()
        )

    # The variance bound ------------------------------------------------------------------------

    def expected_true_proportion(self, rule: Rule) -> float:
        return expected_true_proportion(self.features, self.rules, rule, self.rng)
