"""Fluents: Boolean facts that events change (``docs/specs/WORLD_AND_LANGUAGE.md``, "Fluents").

A world has ``fluents.count`` fluents ``BOOLFL.1`` to ``BOOLFL.n``. The first ones are base
fluents, with an initial rate drawn from ``fluents.initial_rates`` and an initial value drawn per
entity at that rate. The last ``fluents.derived_proportion`` of them are derived: each has a rule
that reads at least one fluent of the layer below, and may also read static features and
threshold literals on scalars. Rule complexity comes from the taxonomy's rule settings. Rules and
rates draw from ``world:fluents``; initial values draw from ``world:initial``. Fluents named in an
event file are always base fluents, and take the file's initial rates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.common.boolean import TruthTable, dnf_literal_count
from semantic_world.taxonomy.config import Config as TaxonomyConfig
from semantic_world.taxonomy.config import ConfigError, RuleSampling
from semantic_world.taxonomy.expressions import Expr, Gt, Var
from semantic_world.taxonomy.features import Feature
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.rules import MAX_TRIES, _draw, build_function, model_quantile_threshold
from semantic_world.world.config import Config
from semantic_world.world.errors import WorldError
from semantic_world.world.streams import WorldStreams

FLUENT_PREFIX = "BOOLFL"
FLUENT_WEIGHT = 1.0
"""The weight of a fluent in a derived rule's input pool (features use the taxonomy's
``rules.input_type_weights``)."""


@dataclass(frozen=True)
class Fluent:
    label: str
    index: int
    derived: bool
    layer: int
    """0 for a base fluent, 1 and up for a derived fluent."""
    initial_rate: float | None
    """A base fluent's initial rate; None for a derived fluent."""


@dataclass(frozen=True)
class ThresholdLiteral:
    """A threshold literal on a scalar, in the world's vocabulary: ``SCALARDIM.2 > 0.4127``."""

    scalar: int
    threshold: float
    quantile: float

    @property
    def label(self) -> str:
        return f"SCALARDIM.{self.scalar}"

    @property
    def key(self) -> str:
        return f"{self.label}>{self.threshold:.4f}"

    def atom(self) -> Gt:
        return Gt(self.label, self.threshold)

    def values(self, scalars: np.ndarray) -> np.ndarray:
        return (np.asarray(scalars)[:, self.scalar - 1] > self.threshold).astype(np.uint8)


@dataclass(frozen=True)
class FluentRule:
    """A derived fluent's rule, over literal keys in the world's vocabulary."""

    output: str
    inputs: tuple[str, ...]
    table: TruthTable
    expression: Expr
    family: str
    nesting_depth: int | None
    thresholds: tuple[ThresholdLiteral, ...]

    @property
    def min_dnf_literals(self) -> int:
        return dnf_literal_count(self.table)

    def record(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "family": self.family,
            "arity": len(self.inputs),
            "nesting_depth": self.nesting_depth,
            "inputs": list(self.inputs),
            "expression": str(self.expression),
            "truth_table": self.table.bit_string(),
            "min_dnf_literals": self.min_dnf_literals,
        }


@dataclass(frozen=True)
class Fluents:
    fluents: tuple[Fluent, ...]
    rules: tuple[FluentRule, ...]
    """One rule per derived fluent, in layer order."""
    initial_values: np.ndarray
    """Shape ``(entities, base fluents)``, uint8: each entity's initial base fluents."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_label", {f.label: f for f in self.fluents})

    def __len__(self) -> int:
        return len(self.fluents)

    def __getitem__(self, label: str) -> Fluent:
        try:
            return self._by_label[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown fluent {label!r}") from None

    def __contains__(self, label: object) -> bool:
        return label in self._by_label  # type: ignore[attr-defined]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.fluents)

    @property
    def base(self) -> tuple[Fluent, ...]:
        return tuple(f for f in self.fluents if not f.derived)

    @property
    def derived(self) -> tuple[Fluent, ...]:
        return tuple(f for f in self.fluents if f.derived)

    @property
    def base_labels(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.base)

    @property
    def initial_rates(self) -> np.ndarray:
        return np.array([f.initial_rate for f in self.base], dtype=float)

    def rule_for(self, label: str) -> FluentRule:
        for rule in self.rules:
            if rule.output == label:
                return rule
        raise KeyError(f"{label} has no rule")

    def thresholds(self) -> tuple[ThresholdLiteral, ...]:
        """Every distinct threshold literal the rules read, by scalar and threshold."""
        found: dict[str, ThresholdLiteral] = {}
        for rule in self.rules:
            for t in rule.thresholds:
                found.setdefault(t.key, t)
        return tuple(sorted(found.values(), key=lambda t: (t.scalar, t.threshold)))


def fluent_label(index: int) -> str:
    return f"{FLUENT_PREFIX}.{index}"


def _even_split(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def generate_fluents(
    config: Config,
    taxonomy: TaxonomyResult,
    streams: WorldStreams,
    explicit_rates: Mapping[str, float] | None = None,
    explicit_source: str | None = None,
) -> Fluents:
    """Draw the fluents of a world. ``explicit_rates`` are the initial rates of fluents named
    in an event file; those fluents are base fluents."""
    settings = config.fluents
    explicit_rates = dict(explicit_rates or {})
    count = settings.count
    explicit_indices: set[int] = set()
    for label in explicit_rates:
        prefix, _, number = label.partition(".")
        if prefix != FLUENT_PREFIX or not number.isdigit() or not 1 <= int(number) <= count:
            raise ConfigError(
                explicit_source or "<event file>",
                f"fluents.{label}",
                f"unknown fluent; the world has fluents {FLUENT_PREFIX}.1 to "
                f"{FLUENT_PREFIX}.{count}",
            )
        explicit_indices.add(int(number))
    derived_count = settings.derived_count
    free_indices = [i for i in range(1, count + 1) if i not in explicit_indices]
    if len(free_indices) < derived_count:
        raise ConfigError(
            explicit_source or "<event file>",
            "fluents",
            f"{len(explicit_indices)} fluents are named in the file and must be base fluents, "
            f"but the world needs {derived_count} derived fluents among {count}",
        )
    derived_indices = set(free_indices[len(free_indices) - derived_count :])
    layer_sizes = _even_split(derived_count, settings.max_chain_depth if derived_count else 0)
    layer_of: dict[int, int] = {}
    ordered_derived = sorted(derived_indices)
    start = 0
    for layer, size in enumerate(layer_sizes, start=1):
        for index in ordered_derived[start : start + size]:
            layer_of[index] = layer
        start += size

    rng = streams.fluents
    rates = list(settings.initial_rates)
    fluents: list[Fluent] = []
    for index in range(1, count + 1):
        label = fluent_label(index)
        if index in derived_indices:
            fluents.append(Fluent(label, index, True, layer_of[index], None))
            continue
        rate = rates[int(rng.integers(len(rates)))]
        if label in explicit_rates:
            rate = float(explicit_rates[label])
        fluents.append(Fluent(label, index, False, 0, float(rate)))

    rules = _draw_rules(fluents, taxonomy, rng)
    base = [f for f in fluents if not f.derived]
    n = len(taxonomy.instances)
    initial = np.zeros((n, len(base)), dtype=np.uint8)
    if base:
        draws = streams.initial.random((n, len(base)))
        initial = (draws < np.array([f.initial_rate for f in base])[None, :]).astype(np.uint8)
    return Fluents(tuple(fluents), tuple(rules), initial)


def _draw_rules(
    fluents: Sequence[Fluent], taxonomy: TaxonomyResult, rng: np.random.Generator
) -> list[FluentRule]:
    config: TaxonomyConfig = taxonomy.config
    sampling: RuleSampling = config.rules.sampling
    features = taxonomy.features
    types = sampling.input_types
    static_pool: list[Feature] = [f for f in features.features if f.type in types]
    scalar_pool = (
        list(range(1, config.scalars.count + 1))
        if config.scalars.count and sampling.scalar_weight > 0
        else []
    )
    rules: list[FluentRule] = []
    seen: set[tuple] = set()
    for output in sorted((f for f in fluents if f.derived), key=lambda f: (f.layer, f.index)):
        below = [f for f in fluents if f.layer < output.layer]
        previous = [f for f in below if f.layer == output.layer - 1]
        if not previous:
            raise WorldError(
                f"the derived fluent {output.label} at layer {output.layer} has no fluent at "
                f"layer {output.layer - 1} to read"
            )
        pool: list[Fluent | Feature | int] = list(below) + list(static_pool) + list(scalar_pool)
        for _ in range(MAX_TRIES):
            arity = _draw(rng, sampling.arity)
            if arity > len(pool):
                raise WorldError(
                    f"a rule for {output.label} needs {arity} inputs but only {len(pool)} are "
                    f"eligible; lower rules.arity in the taxonomy configuration"
                )
            first = previous[int(rng.integers(len(previous)))]
            rest = [entry for entry in pool if entry is not first]
            chosen: list[Fluent | Feature | int] = [first]
            if arity > 1:
                weights = np.array([_weight(entry, sampling) for entry in rest], dtype=float)
                picks = rng.choice(
                    len(rest), size=arity - 1, replace=False, p=weights / weights.sum()
                )
                chosen.extend(rest[int(i)] for i in picks)
            inputs, atoms, thresholds = _inputs(chosen, config, rng)
            keys = [k for k in inputs]
            draw = build_function(rng, sampling, atoms, keys)
            key = (tuple(sorted(inputs)), draw.table.permute_inputs(_sort_permutation(inputs)).bits)
            if key in seen and not config.rules.allow_duplicate_rules:
                continue
            seen.add(key)
            rules.append(
                FluentRule(
                    output.label,
                    tuple(inputs),
                    draw.table,
                    draw.expression,
                    draw.family,
                    draw.nesting_depth,
                    tuple(thresholds),
                )
            )
            break
        else:
            raise WorldError(f"could not sample a rule for {output.label} in {MAX_TRIES} tries")
    return rules


def _sort_permutation(inputs: Sequence[str]) -> list[int]:
    order = sorted(range(len(inputs)), key=lambda i: inputs[i])
    new_index = {old: new for new, old in enumerate(order)}
    return [new_index[i] for i in range(len(inputs))]


def _weight(entry: Fluent | Feature | int, sampling: RuleSampling) -> float:
    if isinstance(entry, Fluent):
        return FLUENT_WEIGHT
    if isinstance(entry, int):
        return sampling.scalar_weight
    return sampling.input_type_weights[entry.type]


def _inputs(
    chosen: Sequence[Fluent | Feature | int], config: TaxonomyConfig, rng: np.random.Generator
) -> tuple[list[str], list[Expr], list[ThresholdLiteral]]:
    """Inputs in a fixed order: static features by position, fluents by index, then threshold
    literals by scalar (drawn in that order)."""
    features = sorted((e for e in chosen if isinstance(e, Feature)), key=lambda f: f.position)
    fluents = sorted((e for e in chosen if isinstance(e, Fluent)), key=lambda f: f.index)
    scalars = sorted(e for e in chosen if isinstance(e, int))
    keys: list[str] = []
    atoms: list[Expr] = []
    thresholds: list[ThresholdLiteral] = []
    for feature in features:
        keys.append(feature.label)
        atoms.append(Var(feature.label))
    for fluent in fluents:
        keys.append(fluent.label)
        atoms.append(Var(fluent.label))
    low, high = config.scalars.threshold_quantiles
    for scalar in scalars:
        quantile = float(rng.uniform(low, high))
        literal = ThresholdLiteral(
            scalar, model_quantile_threshold(config.scalars, quantile), quantile
        )
        thresholds.append(literal)
        keys.append(literal.key)
        atoms.append(literal.atom())
    return keys, atoms, thresholds


def derived_initial_values(fluents: Fluents, taxonomy: TaxonomyResult) -> dict[str, np.ndarray]:
    """Every derived fluent's value for every entity in the initial state, by truth-table
    lookup in layer order (the reference evaluation; the matrices are checked against it)."""
    features = taxonomy.features
    values = taxonomy.instances.values
    scalars = taxonomy.instances.scalars
    columns: dict[str, np.ndarray] = {}
    for i, fluent in enumerate(fluents.base):
        columns[fluent.label] = fluents.initial_values[:, i]
    for rule in sorted(
        fluents.rules, key=lambda r: (fluents[r.output].layer, fluents[r.output].index)
    ):
        inputs = []
        for key in rule.inputs:
            if key in columns:
                inputs.append(columns[key])
            elif ">" in key:
                literal = next(t for t in rule.thresholds if t.key == key)
                inputs.append(literal.values(scalars))
            else:
                inputs.append(values[:, features[key].position])
        stacked = (
            np.stack(inputs, axis=1) if inputs else np.zeros((values.shape[0], 0), dtype=np.uint8)
        )
        columns[rule.output] = rule.table.evaluate(stacked)
    return {f.label: columns[f.label] for f in fluents.derived}
