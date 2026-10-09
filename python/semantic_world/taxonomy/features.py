"""The feature layout: labels, free and determined features, layers, and base rates.

Within each type, free features are numbered first, then determined features in layer order.
Determined PROPERTY and PART features are split as evenly as possible across layers 1 to
``max_chain_depth``, lowest layers first, and the assignment of types to layer slots is drawn
from the ``taxonomy:rules`` stream.

The matrix layout used by the generator puts the non-ISA features in the order PROPERTY, PART,
each by index. ISA features are determined by the tree alone and live outside this layout. The
world package appends its one-place event types to this layout when it computes their
requirements (``semantic_world.world.unary``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_world.taxonomy.config import (
    FEATURE_TYPES,
    FREE_FEATURE_TYPES,
    LABEL_PREFIX,
    SCALAR_PREFIX,
    Config,
)
from semantic_world.taxonomy.streams import Streams


@dataclass(frozen=True)
class Feature:
    label: str
    type: str
    """``is`` (a PROPERTY feature) or ``has`` (a PART feature). The world package uses other
    types for the features it appends (``event_type1``)."""
    index: int
    """The 1-based index within the type, as in the label."""
    free: bool
    layer: int
    """0 for free features; 1 and up for determined features."""
    base_rate: float | None
    """The base rate of a free feature; None for determined features."""
    position: int
    """The column of the feature in the non-ISA matrix (PROPERTY, PART, each by index)."""


@dataclass(frozen=True)
class FeatureSet:
    """Every non-ISA feature, in matrix order, with the derived views the generator needs."""

    features: tuple[Feature, ...]
    max_chain_depth: int
    warnings: tuple[str, ...]
    scalar_count: int = 0
    """The number of scalar dimensions ``SCALARDIM.1`` to ``SCALARDIM.<count>``. Scalars are not in
    ``features``: they are always free, have no rules, and live in separate float arrays."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_label", {f.label: f for f in self.features})
        object.__setattr__(self, "_free", tuple(f for f in self.features if f.free))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, label: str) -> Feature:
        try:
            return self._by_label[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown feature {label!r}") from None

    def __contains__(self, label: object) -> bool:
        return label in self._by_label  # type: ignore[attr-defined]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.features)

    @property
    def scalar_labels(self) -> tuple[str, ...]:
        return tuple(f"{SCALAR_PREFIX}.{i}" for i in range(1, self.scalar_count + 1))

    def of_type(self, feature_type: str) -> tuple[Feature, ...]:
        return tuple(f for f in self.features if f.type == feature_type)

    @property
    def free(self) -> tuple[Feature, ...]:
        """The free features, PROPERTY then PART, by index."""
        return self._free  # type: ignore[attr-defined]

    @property
    def free_positions(self) -> np.ndarray:
        return np.array([f.position for f in self.free], dtype=np.intp)

    @property
    def base_rates(self) -> np.ndarray:
        """The base rate of every free feature, in free-feature order."""
        return np.array([f.base_rate for f in self.free], dtype=float)

    @property
    def determined(self) -> tuple[Feature, ...]:
        """The determined features in evaluation order: layer by layer (PROPERTY before PART
        within a layer, each by index)."""
        determined = [f for f in self.features if not f.free]
        determined.sort(key=lambda f: (f.layer, FEATURE_TYPES.index(f.type), f.index))
        return tuple(determined)

    def layer(self, layer: int) -> tuple[Feature, ...]:
        """The features at a layer (layer 0 is the free features)."""
        return tuple(f for f in self.features if f.layer == layer)

    @property
    def layers(self) -> int:
        """The number of layers with at least one feature, not counting layer 0."""
        return max((f.layer for f in self.features), default=0)


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def build_features(config: Config, streams: Streams) -> FeatureSet:
    """Lay out the features of a configuration. Draws the layer assignment from the ``rules``
    stream and, with heterogeneity on, the base rates from the ``base_rates`` stream."""
    fc = config.features
    warnings: list[str] = []

    # Layers for the determined features.
    determined_counts = {t: fc.determined_count(t) for t in FREE_FEATURE_TYPES}
    total_determined = sum(determined_counts.values())
    depth = config.rules.max_chain_depth if total_determined else 0
    layer_sizes = _even_split(total_determined, depth)
    slots = [t for t in FREE_FEATURE_TYPES for _ in range(determined_counts[t])]
    order = streams.rules.permutation(len(slots)) if slots else np.array([], dtype=int)
    shuffled = [slots[i] for i in order]
    layers_by_type: dict[str, list[int]] = {t: [] for t in FREE_FEATURE_TYPES}
    start = 0
    for layer, size in enumerate(layer_sizes, start=1):
        for slot_type in shuffled[start : start + size]:
            layers_by_type[slot_type].append(layer)
        start += size
    for t in FREE_FEATURE_TYPES:
        layers_by_type[t].sort()
    empty = [layer for layer, size in enumerate(layer_sizes, start=1) if size == 0]
    if empty:
        warnings.append(
            f"rules.max_chain_depth is {depth} but only {total_determined} PROPERTY and PART "
            f"features are determined, so layers {empty} are empty"
        )

    # Base rates for the free features.
    base_rates: dict[str, list[float]] = {}
    for t in FREE_FEATURE_TYPES:
        free_count = fc.free_count(t)
        mean = fc.base_rate(t)
        if free_count == 0:
            base_rates[t] = []
        elif fc.base_rate_heterogeneity is None or mean in (0.0, 1.0):
            base_rates[t] = [float(mean)] * free_count
        else:
            kappa = fc.base_rate_heterogeneity
            draws = streams.base_rates.beta(mean * kappa, (1 - mean) * kappa, size=free_count)
            base_rates[t] = [float(r) for r in draws]

    features: list[Feature] = []
    position = 0
    for t in FREE_FEATURE_TYPES:
        free_count = fc.free_count(t)
        for index in range(1, free_count + 1):
            features.append(
                Feature(
                    label=f"{LABEL_PREFIX[t]}.{index}",
                    type=t,
                    index=index,
                    free=True,
                    layer=0,
                    base_rate=base_rates[t][index - 1],
                    position=position,
                )
            )
            position += 1
        for offset, layer in enumerate(layers_by_type[t]):
            index = free_count + offset + 1
            features.append(
                Feature(
                    label=f"{LABEL_PREFIX[t]}.{index}",
                    type=t,
                    index=index,
                    free=False,
                    layer=layer,
                    base_rate=None,
                    position=position,
                )
            )
            position += 1
    return FeatureSet(
        tuple(features),
        max_chain_depth=depth,
        warnings=tuple(warnings),
        scalar_count=config.scalars.count,
    )


def _even_split(total: int, parts: int) -> list[int]:
    """Split ``total`` items into ``parts`` sizes that differ by at most one, largest first."""
    if parts <= 0:
        return []
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
