"""Numeric fluents: real-valued base fluents that events add to and subtract from, and that can
drift by themselves (``docs/specs/WORLD_AND_LANGUAGE.md``, "Numeric fluents", stage b2).

A world has ``numeric_fluents.count`` numeric fluents ``NUMFL.1`` to ``NUMFL.n``. Each has a
range (``[0, 1]`` by default), an initial distribution (``uniform`` over the range, ``fixed`` at a
value, or ``beta`` with a mean and a concentration), and a drift per step (0 for none). Values
are clamped to the range. Which fluents drift, and by how much, is drawn from ``world:numeric``;
each entity's initial values are drawn from ``world:numeric_initial``. With a count of 0 nothing
is drawn, and every other output of the world is as it was before the stage.

The order within a step is ``clamp(clamp(v + e) + d)``: the one effect that wrote the fluent,
then the drift, each clamped, in float64. ``clamp`` and ``step_value`` here are the reference
arithmetic that the runtime, the brute-force evaluator, and the histories' replay all use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.world.config import Config
from semantic_world.world.streams import WorldStreams

NUMERIC_PREFIX = "NUMFL"
NUMERIC_KIND = "numeric_fluent"
"""The symbol kind of a numeric fluent in ``definition.json``."""
INITIAL_KINDS = ("uniform", "fixed", "beta")


def numeric_label(index: int) -> str:
    return f"{NUMERIC_PREFIX}.{index}"


def is_numeric_label(label: str) -> bool:
    return label.startswith(NUMERIC_PREFIX + ".")


def clamp(value: float, low: float, high: float) -> float:
    """A value clamped to ``[low, high]``, as one float64 operation after another."""
    return min(max(value, low), high)


def step_value(value: float, effect: float | None, drift: float, low: float, high: float) -> float:
    """The value after one step: the effect (None for none) clamped, then the drift clamped."""
    after = clamp(value + effect, low, high) if effect is not None else value
    return clamp(after + drift, low, high)


@dataclass(frozen=True)
class InitialDistribution:
    """How an entity's initial value of a numeric fluent is drawn."""

    kind: str
    value: float | None = None
    """``fixed``: the value, inside the range."""
    mean: float | None = None
    """``beta``: the mean, inside the range."""
    concentration: float | None = None
    """``beta``: the sum of the two shape parameters."""

    def record(self) -> dict[str, Any]:
        if self.kind == "fixed":
            return {"kind": "fixed", "value": float(self.value)}  # type: ignore[arg-type]
        if self.kind == "beta":
            return {
                "kind": "beta",
                "mean": float(self.mean),  # type: ignore[arg-type]
                "concentration": float(self.concentration),  # type: ignore[arg-type]
            }
        return {"kind": "uniform"}

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> InitialDistribution:
        kind = str(data["kind"])
        if kind == "fixed":
            return cls("fixed", value=float(data["value"]))
        if kind == "beta":
            return cls("beta", mean=float(data["mean"]), concentration=float(data["concentration"]))
        if kind == "uniform":
            return cls("uniform")
        raise ValueError(f"unknown initial distribution {kind!r}")

    def draw(self, rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
        """``size`` initial values, in float64, inside ``[low, high]``."""
        if self.kind == "fixed":
            assert self.value is not None
            return np.full(size, float(self.value), dtype=np.float64)
        if self.kind == "beta":
            assert self.mean is not None and self.concentration is not None
            fraction = (self.mean - low) / (high - low)
            a, b = fraction * self.concentration, (1.0 - fraction) * self.concentration
            return low + (high - low) * rng.beta(a, b, size=size)
        return rng.uniform(low, high, size=size)

    def can_exceed(self, low: float, high: float, threshold: float) -> bool:
        """Whether some drawn value can be above the threshold (strictly)."""
        if self.kind == "fixed":
            return self.value > threshold  # type: ignore[operator]
        return high > threshold

    def can_reach_below(self, low: float, high: float, threshold: float) -> bool:
        """Whether some drawn value can be at or below the threshold."""
        if self.kind == "fixed":
            return self.value <= threshold  # type: ignore[operator]
        return low <= threshold


@dataclass(frozen=True)
class NumericFluent:
    label: str
    index: int
    low: float
    high: float
    initial: InitialDistribution
    drift: float
    """The change per step, 0 for a fluent that does not drift."""

    @property
    def range(self) -> tuple[float, float]:
        return (self.low, self.high)

    @property
    def drifts(self) -> bool:
        return self.drift != 0.0

    def point(self, fraction: float) -> float:
        """The point of the range at a fraction of the way through it."""
        return self.low + fraction * (self.high - self.low)

    def clamp(self, value: float) -> float:
        return clamp(value, self.low, self.high)

    def symbol_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": NUMERIC_KIND,
            "derived": False,
            "fluent": True,
            "arity": 1,
            "range": [float(self.low), float(self.high)],
            "initial": self.initial.record(),
            "drift": float(self.drift),
        }


@dataclass(frozen=True)
class NumericThreshold:
    """A threshold literal on a numeric fluent, read by a derived Boolean fluent's rule:
    ``NUMFL.1 > 0.75``."""

    fluent: str
    threshold: float

    @property
    def key(self) -> str:
        return f"{self.fluent}>{self.threshold:.4f}"

    def values(self, numeric: np.ndarray, index: int) -> np.ndarray:
        return (np.asarray(numeric, dtype=np.float64)[:, index] > self.threshold).astype(np.uint8)


@dataclass(frozen=True)
class NumericFluents:
    fluents: tuple[NumericFluent, ...]
    initial_values: np.ndarray
    """Shape ``(entities, numeric fluents)``, float64: each entity's initial values."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_label", {f.label: f for f in self.fluents})
        object.__setattr__(self, "_index", {f.label: i for i, f in enumerate(self.fluents)})

    def __len__(self) -> int:
        return len(self.fluents)

    def __iter__(self):
        return iter(self.fluents)

    def __getitem__(self, label: str) -> NumericFluent:
        try:
            return self._by_label[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown numeric fluent {label!r}") from None

    def __contains__(self, label: object) -> bool:
        return label in self._by_label  # type: ignore[attr-defined]

    def index(self, label: str) -> int:
        return self._index[label]  # type: ignore[attr-defined]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.fluents)

    @property
    def drift_vector(self) -> np.ndarray:
        return np.array([f.drift for f in self.fluents], dtype=np.float64)

    def achievable(
        self, label: str, above: bool, threshold: float, produced: set[tuple[str, int]]
    ) -> bool:
        """Whether a threshold literal on the fluent can hold: ``above`` (``> threshold``) when
        some entity starts above it, the fluent drifts upward, or some effect adds to it; the
        negation (``<= threshold``) when some entity starts at or below it, the fluent drifts
        downward, or some effect subtracts from it. ``produced`` holds ``(fluent, +1 or -1)``
        for the effects of the world."""
        fluent = self[label]
        if not fluent.low < threshold < fluent.high:
            return False  # a threshold at or outside the ends of the range is never a literal
        column = self.initial_values[:, self.index(label)]
        if above:
            return (
                bool((column > threshold).any())
                or fluent.drift > 0
                or (label, 1) in produced
                or (
                    column.size == 0
                    and fluent.initial.can_exceed(fluent.low, fluent.high, threshold)
                )
            )
        return (
            bool((column <= threshold).any())
            or fluent.drift < 0
            or (label, -1) in produced
            or (
                column.size == 0
                and fluent.initial.can_reach_below(fluent.low, fluent.high, threshold)
            )
        )


EMPTY = NumericFluents((), np.zeros((0, 0), dtype=np.float64))


def empty_numeric(entities: int) -> NumericFluents:
    return NumericFluents((), np.zeros((entities, 0), dtype=np.float64))


def generate_numeric_fluents(
    config: Config,
    taxonomy: TaxonomyResult,
    streams: WorldStreams,
    explicit: Mapping[str, Mapping[str, Any]] | None = None,
    explicit_source: str | None = None,
) -> NumericFluents:
    """Draw the numeric fluents of a world: for each, its range and initial distribution from
    the configuration (the per-fluent overrides first), its drift (drawn against
    ``numeric_fluents.drift.share`` and from ``drift.amounts``, unless the configuration or an
    event file fixes it), and every entity's initial value. ``explicit`` holds the entries of an
    event file's ``fluents`` block for numeric fluents (``initial``, ``drift``)."""
    settings = config.numeric_fluents
    n = len(taxonomy.instances)
    count = settings.count
    explicit = dict(explicit or {})
    for label in explicit:
        prefix, _, number = label.partition(".")
        if prefix != NUMERIC_PREFIX or not number.isdigit() or not 1 <= int(number) <= count:
            raise ConfigError(
                explicit_source or "<event file>",
                f"fluents.{label}",
                f"unknown numeric fluent; the world has {NUMERIC_PREFIX}.1 to "
                f"{NUMERIC_PREFIX}.{count}",
            )
    if count == 0:
        return empty_numeric(n)
    rng = streams.numeric
    fluents: list[NumericFluent] = []
    for index in range(1, count + 1):
        label = numeric_label(index)
        override = settings.fluents.get(label)
        low, high = settings.range if override is None or override.range is None else override.range
        initial_record = (
            settings.initial if override is None or override.initial is None else override.initial
        )
        initial = InitialDistribution.from_record(initial_record)
        if override is not None and override.drift_set:
            drift = float(override.drift or 0.0)
        else:
            drift = 0.0
            if rng.random() < settings.drift_share:
                drift = float(rng.uniform(settings.drift_amounts[0], settings.drift_amounts[1]))
        entry = explicit.get(label)
        if entry is not None:
            if entry.get("initial") is not None:
                initial = InitialDistribution.from_record(entry["initial"])
            if "drift" in entry:
                drift = float(entry["drift"] or 0.0)
        fluents.append(NumericFluent(label, index, float(low), float(high), initial, drift))
    rng_initial = streams.numeric_initial
    columns = [f.initial.draw(rng_initial, f.low, f.high, n) for f in fluents]
    initial_values = (
        np.stack(columns, axis=1).astype(np.float64)
        if columns
        else np.zeros((n, 0), dtype=np.float64)
    )
    return NumericFluents(tuple(fluents), initial_values)


def numeric_points(fluent: NumericFluent, fractions: Sequence[float]) -> tuple[float, ...]:
    """The threshold points of a fluent: the range at each fraction."""
    return tuple(fluent.point(f) for f in fractions)


__all__ = [
    "EMPTY",
    "INITIAL_KINDS",
    "NUMERIC_KIND",
    "NUMERIC_PREFIX",
    "InitialDistribution",
    "NumericFluent",
    "NumericFluents",
    "NumericThreshold",
    "clamp",
    "empty_numeric",
    "generate_numeric_fluents",
    "is_numeric_label",
    "numeric_label",
    "numeric_points",
    "step_value",
]
