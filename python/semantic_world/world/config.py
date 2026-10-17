"""The world configuration: ``data/world/default.yaml`` and its validation.

A world names a taxonomy configuration and a seed, the fluents, and the event-type settings. The
block ``event_types.unary`` holds the settings of the one-place event types (the taxonomy's old
``features.can`` count and ``rules.overrides.can`` sampling overrides), and ``event_types.binary``
the settings of the two-place event types (the taxonomy's old ``verbs`` block, without its
exposure keys). Both are read against the taxonomy configuration the world names, because the
sampling settings of requirements and constraints start from the taxonomy's ``rules`` block and
the comparison family needs its scalars. Validation follows the taxonomy's conventions: unknown
keys are errors, and every error names the file and the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_world.taxonomy.config import (
    _MISSING,
    SEED_MAX,
    ConfigError,
    InheritanceConfig,
    RuleSampling,
    ScalarsConfig,
    SimilarityBound,
    TaxonomyConfig,
    TreeSettings,
    _describe,
    _list_schedule,
    _Node,
    _read_inheritance,
    _read_open_interval,
    _read_sampling,
    _read_similarity_bound,
    _read_taxonomy,
    _resolved_bound,
    _resolved_inheritance,
)
from semantic_world.taxonomy.config import Config as TaxonomyRunConfig
from semantic_world.taxonomy.config import config_from_mapping as taxonomy_from_mapping
from semantic_world.taxonomy.streams import stream_seed

ROLES = ("agent", "patient")
REMOVED_BINARY_KEYS = ("projections",)
"""Keys of the old ``verbs`` block that the world does not take: exposure is replaced by views."""
CONSTRAINT_FAMILIES = ("agent", "patient", "cross", "key_lock", "comparison")
EVENT_FEATURE_PREFIX = "EVENTFEAT"
TWO_PLACE_PREFIX = "EVENTTYPE2."
DEFAULT_UNARY_COUNT = 20
DEFAULT_PLACES_CONFIG = "data/taxonomy/places_default.yaml"
"""The default place taxonomy (stage c1 of ``docs/specs/WORLD_PHASE_C.md``): the code's default
follows ``data/world/default.yaml``, as the entity taxonomy's does (WM.E41)."""
PLACES_STREAM = "world:places"
"""The stream whose seed the place taxonomy takes when ``places.taxonomy.seed`` is null."""
PLACE_CONSTRAINT_FAMILIES = CONSTRAINT_FAMILIES
DEFAULT_PLACE_CONSTRAINT_FAMILIES = {"agent": 1, "patient": 1, "cross": 1, "key_lock": 1}
"""The comparison family is off by default for place event types: an entity's scalar and a
place's scalar are different dimensions (WM.45)."""


@dataclass(frozen=True)
class TaxonomySetting:
    config: str
    """The taxonomy configuration file, as written (relative paths are relative to the current
    directory, as the corpus resolves them)."""
    seed: int | None
    """The taxonomy's seed, or None for the world's seed."""


@dataclass(frozen=True)
class FluentsConfig:
    count: int
    derived_proportion: float
    max_chain_depth: int
    initial_rates: tuple[float, ...]

    @property
    def derived_count(self) -> int:
        return math.floor(self.count * self.derived_proportion + 0.5)

    @property
    def base_count(self) -> int:
        return self.count - self.derived_count

    def resolved(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "derived_proportion": self.derived_proportion,
            "max_chain_depth": self.max_chain_depth,
            "initial_rates": list(self.initial_rates),
        }


DEFAULT_NUMERIC_THRESHOLDS = (0.25, 0.5, 0.75)
"""The fractions of a numeric fluent's range from which a numeric threshold literal's point is
drawn (stage b2): the quarter points by default."""


@dataclass(frozen=True)
class PreconditionsConfig:
    literals: dict[int, float]
    feature_rate: float
    roles: dict[str, float]
    enabled_share: float
    numeric_share: float = 0.3
    """The share of precondition literals that are numeric threshold literals, when the world
    has numeric fluents (stage b2)."""
    numeric_thresholds: tuple[float, ...] = DEFAULT_NUMERIC_THRESHOLDS

    def resolved(self) -> dict[str, Any]:
        return {
            "literals": dict(self.literals),
            "feature_rate": self.feature_rate,
            "roles": dict(self.roles),
            "enabled_share": self.enabled_share,
            "numeric_share": self.numeric_share,
            "numeric_thresholds": list(self.numeric_thresholds),
        }


CONDITION_KINDS = ("feature", "fluent", "numeric")
"""The kinds of a condition literal: a static feature, a Boolean fluent, or a numeric threshold
(stage b2)."""
DEFAULT_CONDITION_KINDS = {"feature": 0.4, "fluent": 0.4, "numeric": 0.2}


@dataclass(frozen=True)
class EffectsConfig:
    count: dict[int, float]
    feature_rate: float
    roles: dict[str, float]
    conditional_share: float = 0.2
    """The share of effects that get a condition (stage b1)."""
    condition_literals: dict[int, float] = None  # type: ignore[assignment]
    """Weights over the number of literals of a condition."""
    condition_kinds: dict[str, float] = None  # type: ignore[assignment]
    """Weights over the kind of a condition literal, ``feature``, ``fluent``, or ``numeric``
    (Jon's ruling 1 on stage b1): a choice of the generator, not a frequency of the world."""
    numeric_share: float = 0.3
    """The share of effects that are numeric (``+=`` or ``-=``), when the world has numeric
    fluents (stage b2)."""
    numeric_amounts: tuple[float, float] = (0.1, 0.5)
    """The interval a numeric effect's constant is drawn from, uniformly."""

    def __post_init__(self) -> None:
        if self.condition_literals is None:
            object.__setattr__(self, "condition_literals", {1: 0.7, 2: 0.3})
        if self.condition_kinds is None:
            object.__setattr__(self, "condition_kinds", dict(DEFAULT_CONDITION_KINDS))

    def resolved(self) -> dict[str, Any]:
        return {
            "count": dict(self.count),
            "feature_rate": self.feature_rate,
            "roles": dict(self.roles),
            "conditional_share": self.conditional_share,
            "condition_literals": dict(self.condition_literals),
            "condition_kinds": dict(self.condition_kinds),
            "numeric_share": self.numeric_share,
            "numeric_amounts": list(self.numeric_amounts),
        }


@dataclass(frozen=True)
class NumericFluentSetting:
    """The per-fluent overrides of ``numeric_fluents.fluents.NUMFL.<n>``."""

    range: tuple[float, float] | None
    initial: dict[str, Any] | None
    drift: float | None
    drift_set: bool
    """Whether the entry gives a drift (a number, or null for none), which replaces the draw."""

    def resolved(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.range is not None:
            data["range"] = list(self.range)
        if self.initial is not None:
            data["initial"] = dict(self.initial)
        if self.drift_set:
            data["drift"] = self.drift
        return data


@dataclass(frozen=True)
class NumericFluentsConfig:
    """The numeric fluents (stage b2): how many, the default range and initial distribution,
    the drift draw, the weight of a numeric threshold literal in a derived fluent's rule, and the
    per-fluent overrides."""

    count: int
    range: tuple[float, float]
    initial: dict[str, Any]
    drift_share: float
    drift_amounts: tuple[float, float]
    rule_weight: float
    fluents: dict[str, NumericFluentSetting]

    def resolved(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "range": list(self.range),
            "initial": dict(self.initial),
            "drift": {"share": self.drift_share, "amounts": list(self.drift_amounts)},
            "rule_weight": self.rule_weight,
            "fluents": {label: setting.resolved() for label, setting in self.fluents.items()},
        }


@dataclass(frozen=True)
class UnaryConfig:
    """The one-place event types: how many, and the sampling settings of their requirements
    (the taxonomy's ``rules`` settings with the ``event_types.unary.rules`` overrides)."""

    count: int
    rules: RuleSampling

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f"EVENTTYPE1.{i}" for i in range(1, self.count + 1))

    def resolved(self) -> dict[str, Any]:
        return {"count": self.count, "rules": self.rules.resolved()}


@dataclass(frozen=True)
class ComparisonConfig:
    window_probability: float
    cross_dimension_probability: float
    margin_quantiles: tuple[float, float]


@dataclass(frozen=True)
class DensityConfig:
    """The leaf-pair density range of every two-place event type, and how many tries the
    generator spends bringing an event type or a constraint into range. A density of 0 or 1 (a
    requirement that holds for no leaf pair, or for every leaf pair) is always outside the range,
    so the default ``[0, 1]`` redraws only those (REL.15 under "A principle for defaults")."""

    min: float
    max: float
    max_tries: int

    def inside(self, value: float) -> bool:
        return self.min <= value <= self.max and 0.0 < value < 1.0

    def distance(self, value: float) -> float:
        """0 inside the range; the distance to the range outside it, and at least ``1e-9`` for
        a degenerate density."""
        if self.inside(value):
            return 0.0
        if value <= 0.0:
            return max(self.min - value, 1e-9)
        if value >= 1.0:
            return max(value - self.max, 1e-9)
        return self.min - value if value < self.min else value - self.max

    def resolved(self) -> dict[str, Any]:
        return {"min": self.min, "max": self.max, "max_tries": self.max_tries}


@dataclass(frozen=True)
class BinaryConfig:
    """Two-place event types: the event-type features, the event-type tree, and the constraint
    families (Part B of ``docs/specs/TAXONOMY_RELATIONS.md``, with "verb" read as "event type")."""

    feature_count: int
    expected_true: float
    taxonomy: TaxonomyConfig
    similarity_bound: SimilarityBound | None
    inheritance: InheritanceConfig
    own_constraint: bool
    constraint_families: dict[str, float]
    key_lock_pairs: dict[int, float]
    comparison: ComparisonConfig
    rules: RuleSampling
    """The rule-complexity settings for constraints: the taxonomy's ``rules`` settings with the
    ``event_types.binary.rules`` overrides applied."""
    sampled_true: int
    sampled_false: int
    max_exact_pairs: int
    density: DensityConfig | None
    """The event-type density range; None turns the check off."""
    constraint_min_density: float | None
    """The smallest leaf-pair density allowed for any single constraint; None turns it off."""

    @property
    def density_tries(self) -> int:
        return self.density.max_tries if self.density is not None else 200

    @property
    def base_rate(self) -> float | None:
        """The base rate shared by the event-type features, or None without features."""
        return self.expected_true / self.feature_count if self.feature_count else None

    @property
    def feature_labels(self) -> tuple[str, ...]:
        return tuple(f"{EVENT_FEATURE_PREFIX}.{i}" for i in range(1, self.feature_count + 1))

    def tree_settings(self) -> TreeSettings:
        return TreeSettings(
            self.taxonomy, self.inheritance, self.similarity_bound, None, TWO_PLACE_PREFIX
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "features": {"count": self.feature_count, "expected_true": self.expected_true},
            "taxonomy": {
                "superordinates": self.taxonomy.superordinates,
                "depth": self.taxonomy.depth,
                "branching": _list_schedule([r.resolved() for r in self.taxonomy.branching]),
            },
            "superordinates": {"similarity_bound": _resolved_bound(self.similarity_bound)},
            "inheritance": _resolved_inheritance(self.inheritance),
            "own_constraint": self.own_constraint,
            "constraint_families": dict(self.constraint_families),
            "key_lock_pairs": dict(self.key_lock_pairs),
            "comparison": {
                "window_probability": self.comparison.window_probability,
                "cross_dimension_probability": self.comparison.cross_dimension_probability,
                "margin_quantiles": list(self.comparison.margin_quantiles),
            },
            "rules": self.rules.resolved(),
            "pairs": {
                "sampled_true": self.sampled_true,
                "sampled_false": self.sampled_false,
                "max_exact_pairs": self.max_exact_pairs,
            },
            "density": None if self.density is None else self.density.resolved(),
            "constraint_min_density": self.constraint_min_density,
        }


@dataclass(frozen=True)
class PlacesConfig:
    """The ``places`` block (stage c1): the place taxonomy, and the habitat strength that stage
    c3 uses."""

    taxonomy: TaxonomySetting
    habitat_strength: float
    taxonomy_run: TaxonomyRunConfig
    """The validated place taxonomy configuration, with its seed resolved."""

    @property
    def seed(self) -> int:
        return self.taxonomy_run.seed

    def resolved(self) -> dict[str, Any]:
        return {
            "taxonomy": {
                "config": self.taxonomy.config,
                "seed": self.taxonomy.seed,
                "resolved": self.taxonomy_run.resolved(),
            },
            "habitat_strength": self.habitat_strength,
        }


@dataclass(frozen=True)
class PlaceEventTypesConfig:
    """The ``place_event_types`` block (stage c1): how many, the move share that stage c3 uses,
    the number of constraints of a requirement, the constraint families, and the rule settings
    of the constraints (the taxonomy's ``rules`` settings with the overrides applied)."""

    count: int
    move_share: float
    constraints: dict[int, float]
    constraint_families: dict[str, float]
    rules: RuleSampling

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f"PLACEEVENTTYPE.{i}" for i in range(1, self.count + 1))

    def resolved(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "move_share": self.move_share,
            "constraints": dict(self.constraints),
            "constraint_families": dict(self.constraint_families),
            "rules": self.rules.resolved(),
        }


@dataclass(frozen=True)
class EventTypesConfig:
    unary: UnaryConfig
    binary: BinaryConfig | None
    """The two-place event types; None means none."""
    preconditions: PreconditionsConfig
    effects: EffectsConfig
    event_file: str | None

    def resolved(self) -> dict[str, Any]:
        return {
            "unary": self.unary.resolved(),
            "binary": None if self.binary is None else self.binary.resolved(),
            "preconditions": self.preconditions.resolved(),
            "effects": self.effects.resolved(),
            "event_file": self.event_file,
        }


@dataclass(frozen=True)
class Config:
    """A validated world configuration."""

    name: str
    seed: int
    taxonomy: TaxonomySetting
    fluents: FluentsConfig
    event_types: EventTypesConfig
    source: str
    taxonomy_run: TaxonomyRunConfig
    numeric_fluents: NumericFluentsConfig = None  # type: ignore[assignment]
    """The numeric fluents (stage b2); the code's defaults when the file has no block."""
    places: PlacesConfig | None = None
    """The places (stage c1); None turns them off. ``config_from_mapping`` gives a mapping
    without a ``places`` key the default place taxonomy."""
    place_event_types: PlaceEventTypesConfig | None = None
    """The place event types; None exactly when the world has no places."""

    def __post_init__(self) -> None:
        if self.numeric_fluents is None:
            object.__setattr__(
                self,
                "numeric_fluents",
                _read_numeric_fluents(_Node("<default>", "numeric_fluents", {})),
            )

    """The validated taxonomy configuration the world generates from."""

    @property
    def taxonomy_seed(self) -> int:
        return self.seed if self.taxonomy.seed is None else self.taxonomy.seed

    def event_file_path(self) -> Path | None:
        """The event file, resolved relative to the configuration file's folder."""
        if self.event_types.event_file is None:
            return None
        path = Path(self.event_types.event_file)
        if path.is_absolute():
            return path
        source = Path(self.source)
        base = source.parent if source.suffix else Path.cwd()
        return base / path

    def taxonomy_config(self) -> TaxonomyRunConfig:
        """The validated taxonomy configuration the world generates from."""
        return self.taxonomy_run

    def resolved(self) -> dict[str, Any]:
        """The configuration as a plain mapping, with the resolved taxonomy configuration under
        ``taxonomy.resolved``. The mapping loads back to an equal configuration."""
        return {
            "name": self.name,
            "seed": self.seed,
            "taxonomy": {
                "config": self.taxonomy.config,
                "seed": self.taxonomy.seed,
                "resolved": self.taxonomy_run.resolved(),
            },
            "fluents": self.fluents.resolved(),
            "numeric_fluents": self.numeric_fluents.resolved(),
            "event_types": self.event_types.resolved(),
            "places": None if self.places is None else self.places.resolved(),
            "place_event_types": (
                None if self.place_event_types is None else self.place_event_types.resolved()
            ),
        }


def _taxonomy_mapping(source: str, setting: TaxonomySetting, seed: int) -> dict[str, Any]:
    """The taxonomy configuration file as a mapping, with the world's seed for the taxonomy and
    without a run's provenance."""
    path = Path(setting.config)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            source, "taxonomy.config", f"cannot read {path}: {error.strerror}"
        ) from error
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(str(path), "<root>", f"expected a mapping, found {_describe(data)}")
    data = dict(data)
    data["seed"] = seed
    data.pop("provenance", None)
    return data


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


def _read_taxonomy_setting(node: _Node) -> TaxonomySetting:
    path = node.string("config", "data/taxonomy/relations.yaml")
    seed = (
        node.int("seed", None, min=0, max=SEED_MAX) if node.data.get("seed") is not None else None
    )
    node.seen.add("seed")
    node.get("resolved", None, nullable=True)  # written by a run's config.yaml; ignored
    node.finish()
    return TaxonomySetting(path, seed)


def _read_fluents(node: _Node) -> FluentsConfig:
    count = node.int("count", 8, min=0)
    proportion = node.probability("derived_proportion", 0.25)
    depth = node.int("max_chain_depth", 1, min=0)
    rates = node.get("initial_rates", [0.0, 0.2, 0.8, 1.0])
    if not isinstance(rates, list) or not rates:
        raise node.error("initial_rates", f"expected a non-empty list, found {_describe(rates)}")
    checked = tuple(float(node.check_number("initial_rates", r, min=0, max=1)) for r in rates)
    node.finish()
    config = FluentsConfig(count, proportion, depth, checked)
    if config.derived_count and depth == 0:
        raise node.error("max_chain_depth", "must be at least 1 when some fluent is derived")
    if config.derived_count and config.base_count == 0:
        raise node.error("derived_proportion", "every fluent would be derived; none would be base")
    return config


def _read_pair(node: _Node, key: str, default: list[float]) -> tuple[float, float]:
    """A pair ``[low, high]`` of numbers with ``low < high``."""
    value = node.get(key, default)
    if not isinstance(value, list) or len(value) != 2:
        raise node.error(key, f"expected a pair [low, high], found {_describe(value)}")
    low = float(node.check_number(key, value[0]))
    high = float(node.check_number(key, value[1]))
    if not low < high:
        raise node.error(key, f"expected low < high, found {value!r}")
    return (low, high)


def _read_initial(node: _Node, key: str, default: dict[str, Any] | None) -> dict[str, Any] | None:
    """An initial distribution of a numeric fluent: ``{kind: uniform}``, ``{kind: fixed, value}``,
    or ``{kind: beta, mean, concentration}``, as a validated plain mapping."""
    value = node.get(key, default, nullable=default is None)
    if value is None:
        return None
    if isinstance(value, str):
        value = {"kind": value}
    if not isinstance(value, dict):
        raise node.error(key, f"expected a mapping with a kind, found {_describe(value)}")
    inner = _Node(node.source, node.field(key), value)
    kind = inner.choice("kind", "uniform", ("uniform", "fixed", "beta"))
    result: dict[str, Any] = {"kind": kind}
    if kind == "fixed":
        result["value"] = float(inner.number("value", _MISSING))
    elif kind == "beta":
        result["mean"] = float(inner.number("mean", _MISSING))
        result["concentration"] = float(inner.number("concentration", 2.0, min=0))
        if result["concentration"] <= 0:
            raise inner.error("concentration", "must be above 0")
    inner.finish()
    return result


def _check_initial_in_range(
    node: _Node, key: str, initial: dict[str, Any], low: float, high: float
) -> None:
    for field in ("value", "mean"):
        if field in initial and not low <= initial[field] <= high:
            raise node.error(
                f"{key}.{field}",
                f"must be inside the range [{low}, {high}], found {initial[field]}",
            )
    if initial.get("kind") == "beta" and initial["mean"] in (low, high):
        raise node.error(f"{key}.mean", "a beta distribution's mean is strictly inside the range")


def _read_numeric_fluents(node: _Node) -> NumericFluentsConfig:
    count = node.int("count", 2, min=0)
    default_range = _read_pair(node, "range", [0.0, 1.0])
    initial = _read_initial(node, "initial", {"kind": "uniform"})
    assert initial is not None
    _check_initial_in_range(node, "initial", initial, *default_range)
    drift = node.mapping("drift")
    share = drift.probability("share", 0.5)
    amounts = _read_pair(drift, "amounts", [-0.1, 0.1])
    drift.finish()
    rule_weight = float(node.number("rule_weight", 21.0, min=0))
    fluents: dict[str, NumericFluentSetting] = {}
    entries = node.mapping("fluents")
    for label in list(entries.data):
        prefix, _, number = str(label).partition(".")
        if prefix != "NUMFL" or not number.isdigit() or not 1 <= int(number) <= count:
            raise entries.error(label, f"expected a numeric fluent label NUMFL.1 to NUMFL.{count}")
        entry = entries.mapping(label)
        fluent_range = (
            _read_pair(entry, "range", None) if entry.data.get("range") is not None else None
        )
        entry.seen.add("range")
        fluent_initial = _read_initial(entry, "initial", None)
        drift_set = "drift" in entry.data
        fluent_drift = entry.number("drift", None, nullable=True) if drift_set else None
        entry.finish()
        low, high = fluent_range or default_range
        _check_initial_in_range(entry, "initial", fluent_initial or initial, low, high)
        fluents[str(label)] = NumericFluentSetting(
            fluent_range,
            fluent_initial,
            None if fluent_drift is None else float(fluent_drift),
            drift_set,
        )
    entries.finish()
    node.finish()
    return NumericFluentsConfig(count, default_range, initial, share, amounts, rule_weight, fluents)


def _read_roles(node: _Node, key: str, default: dict[str, float]) -> dict[str, float]:
    return node.weights(key, default, allowed=ROLES)


def _count_weights(node: _Node, key: str, default: dict[int, float]) -> dict[int, float]:
    """Weights over counts: a mapping from non-negative integers to non-negative weights, at
    least one positive, returned in count order."""
    value = node.get(key, default)
    if not isinstance(value, dict) or not value:
        raise node.error(key, f"expected a mapping of weights, found {_describe(value)}")
    result: dict[int, float] = {}
    for count, weight in value.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise node.error(key, f"keys must be non-negative integers, found {count!r}")
        result[count] = float(node.check_number(f"{key}.{count}", weight, min=0))
    if all(w == 0 for w in result.values()):
        raise node.error(key, "at least one weight must be positive")
    return {k: result[k] for k in sorted(result)}


def _read_preconditions(node: _Node) -> PreconditionsConfig:
    literals = _count_weights(node, "literals", {0: 0.3, 1: 0.5, 2: 0.2})
    rate = node.probability("feature_rate", 0.3)
    roles = _read_roles(node, "roles", {"agent": 0.5, "patient": 0.5})
    enabled = node.probability("enabled_share", 0.7)
    numeric_share = node.probability("numeric_share", 0.3)
    thresholds = node.get("numeric_thresholds", list(DEFAULT_NUMERIC_THRESHOLDS))
    if not isinstance(thresholds, list) or not thresholds:
        raise node.error(
            "numeric_thresholds",
            f"expected a non-empty list of fractions, found {_describe(thresholds)}",
        )
    points = tuple(float(node.check_number("numeric_thresholds", v)) for v in thresholds)
    if any(not 0.0 < v < 1.0 for v in points):
        raise node.error("numeric_thresholds", "every fraction is strictly between 0 and 1")
    node.finish()
    return PreconditionsConfig(literals, rate, roles, enabled, numeric_share, points)


def _read_effects(node: _Node) -> EffectsConfig:
    count = _count_weights(node, "count", {1: 0.7, 2: 0.3})
    rate = node.probability("feature_rate", 0.5)
    roles = _read_roles(node, "roles", {"agent": 0.4, "patient": 0.6})
    conditional_share = node.probability("conditional_share", 0.2)
    condition_literals = _count_weights(node, "condition_literals", {1: 0.7, 2: 0.3})
    if 0 in condition_literals and condition_literals[0] > 0:
        raise node.error("condition_literals", "a condition has at least one literal")
    condition_kinds = node.weights(
        "condition_kinds", dict(DEFAULT_CONDITION_KINDS), allowed=CONDITION_KINDS
    )
    numeric_share = node.probability("numeric_share", 0.3)
    amounts = _read_pair(node, "numeric_amounts", [0.1, 0.5])
    if amounts[0] < 0:
        raise node.error("numeric_amounts", "the constants of numeric effects are positive")
    node.finish()
    return EffectsConfig(
        count,
        rate,
        roles,
        conditional_share,
        condition_literals,
        condition_kinds,
        numeric_share,
        amounts,
    )


def _read_unary(node: _Node, taxonomy: TaxonomyRunConfig) -> UnaryConfig:
    count = node.int("count", DEFAULT_UNARY_COUNT, min=0)
    rules_node = node.mapping("rules")
    sampling = _read_sampling(rules_node, taxonomy.rules.sampling)
    rules_node.finish()
    node.finish()
    if count:
        _check_requirement_pool(rules_node, sampling, taxonomy)
    return UnaryConfig(count, sampling)


def _check_requirement_pool(
    node: _Node, sampling: RuleSampling, taxonomy: TaxonomyRunConfig
) -> None:
    """The largest arity with nonzero weight must fit the pool of a one-place requirement:
    every PROPERTY and PART feature of the types with positive weight, and the scalars."""
    types = sampling.input_types
    pool = sum(taxonomy.features.count(t) for t in types)
    what = "PROPERTY or PART features"
    if taxonomy.scalars.count and sampling.scalar_weight > 0:
        pool += taxonomy.scalars.count
        what += f" plus {taxonomy.scalars.count} scalar threshold literals"
    if sampling.max_arity > pool:
        raise node.error(
            "arity",
            f"the largest arity with nonzero weight is {sampling.max_arity}, but a one-place "
            f"requirement can read at most {pool} {what}",
        )


def _read_binary(root: _Node, taxonomy: TaxonomyRunConfig) -> BinaryConfig | None:
    value = root.get("binary", {}, nullable=True)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise root.error("binary", f"expected a mapping or null, found {_describe(value)}")
    node = _Node(root.source, root.field("binary"), value)
    for key in REMOVED_BINARY_KEYS:
        if key in value:
            raise node.error(
                key,
                "exposure is removed from the world (REL.16): views choose the columns a model "
                "sees; drop this key",
            )
    scalars: ScalarsConfig = taxonomy.scalars
    features = node.mapping("features")
    count = features.int("count", 12, min=0)
    expected_true = features.number("expected_true", 3, min=0)
    features.finish()
    if count and expected_true > count:
        raise features.error(
            "expected_true",
            f"must be at most the number of event-type features ({count}), found {expected_true}",
        )
    shape = _read_taxonomy(node.mapping("taxonomy"), defaults=(3, 2, [2, 3]))
    similarity_bound = _read_similarity_bound(node.mapping("superordinates"), default_on=False)
    inheritance = _read_inheritance(
        node.mapping("inheritance"), shape.depth, defaults=(0.4, 0.4, 0.9)
    )
    own_constraint = node.bool("own_constraint", True)
    families = node.weights(
        "constraint_families",
        {"agent": 1, "patient": 1, "cross": 1, "key_lock": 1, "comparison": 1},
        allowed=CONSTRAINT_FAMILIES,
    )
    usable = [f for f, w in families.items() if w > 0 and (f != "comparison" or scalars.count > 0)]
    if not usable:
        raise node.error(
            "constraint_families",
            "no constraint family with positive weight can be used: the comparison family needs "
            "scalars.count above 0 in the taxonomy configuration",
        )
    key_lock_pairs = node.weights("key_lock_pairs", {1: 0.5, 2: 0.3, 3: 0.2}, int_keys=True)
    comparison_node = node.mapping("comparison")
    comparison = ComparisonConfig(
        comparison_node.probability("window_probability", 0.3),
        comparison_node.probability("cross_dimension_probability", 0.2),
        _read_open_interval(comparison_node, "margin_quantiles", [0.1, 0.9]),
    )
    comparison_node.finish()
    rules_node = node.mapping("rules")
    constraint_rules = _read_sampling(rules_node, taxonomy.rules.sampling)
    rules_node.finish()
    pairs = node.mapping("pairs")
    sampled_true = pairs.int("sampled_true", 1000, min=0)
    sampled_false = pairs.int("sampled_false", 1000, min=0)
    max_exact_pairs = pairs.int("max_exact_pairs", 50_000_000, min=1)
    pairs.finish()
    density_node = node.mapping("density", nullable=True)
    density = None
    if density_node is not None:
        low = density_node.probability("min", 0.0)
        high = density_node.probability("max", 1.0)
        tries = density_node.int("max_tries", 200, min=1)
        density_node.finish()
        if low > high:
            raise density_node.error("min", f"min {low} exceeds max {high}")
        density = DensityConfig(low, high, tries)
    constraint_min_density = node.probability("constraint_min_density", None, nullable=True)
    node.finish()
    return BinaryConfig(
        feature_count=count,
        expected_true=expected_true,
        taxonomy=shape,
        similarity_bound=similarity_bound,
        inheritance=inheritance,
        own_constraint=own_constraint,
        constraint_families=families,
        key_lock_pairs=key_lock_pairs,
        comparison=comparison,
        rules=constraint_rules,
        sampled_true=sampled_true,
        sampled_false=sampled_false,
        max_exact_pairs=max_exact_pairs,
        density=density,
        constraint_min_density=constraint_min_density,
    )


def _read_places(root: _Node, source: str, master_seed: int) -> PlacesConfig | None:
    """The ``places`` block: the default place taxonomy when absent, None when null."""
    value = root.get("places", {}, nullable=True)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise root.error("places", f"expected a mapping or null, found {_describe(value)}")
    node = _Node(root.source, root.field("places"), value)
    setting_node = node.mapping("taxonomy")
    path = setting_node.string("config", DEFAULT_PLACES_CONFIG)
    seed = (
        setting_node.int("seed", None, min=0, max=SEED_MAX)
        if setting_node.data.get("seed") is not None
        else None
    )
    setting_node.seen.add("seed")
    setting_node.get("resolved", None, nullable=True)  # written by a run's config.yaml; ignored
    setting_node.finish()
    strength = float(node.number("habitat_strength", 1.0, min=0))
    node.finish()
    setting = TaxonomySetting(path, seed)
    # The seed of the stream world:places, as the taxonomy takes it (its low 64 bits), when the
    # setting gives none, so that the place taxonomy never repeats the entities' draws.
    place_seed = stream_seed(master_seed, PLACES_STREAM) % (SEED_MAX + 1) if seed is None else seed
    run = taxonomy_from_mapping(
        _taxonomy_mapping(source, setting, place_seed), source=setting.config
    )
    if run.labels.kind != "place":
        raise node.error(
            "taxonomy.config",
            f"{path} must set labels.kind: place; a place taxonomy writes the place labels",
        )
    return PlacesConfig(setting, strength, run)


def _read_place_event_types(
    root: _Node, places: PlacesConfig | None, taxonomy: TaxonomyRunConfig
) -> PlaceEventTypesConfig | None:
    """The ``place_event_types`` block: the defaults when absent and the world has places; an
    error when given without places."""
    value = root.get("place_event_types", {}, nullable=True)
    if places is None:
        if value not in ({}, None):
            raise root.error(
                "place_event_types",
                "needs places; the world has places: null, so leave this block out or null",
            )
        return None
    if value is None:
        value = {"count": 0}
    if not isinstance(value, dict):
        raise root.error(
            "place_event_types", f"expected a mapping or null, found {_describe(value)}"
        )
    node = _Node(root.source, root.field("place_event_types"), value)
    count = node.int("count", 4, min=0)
    move_share = node.probability("move_share", 0.5)
    constraints = _count_weights(node, "constraints", {1: 0.6, 2: 0.4})
    if 0 in constraints and constraints[0] > 0:
        raise node.error("constraints", "a requirement has at least one constraint")
    families = node.weights(
        "constraint_families",
        dict(DEFAULT_PLACE_CONSTRAINT_FAMILIES),
        allowed=PLACE_CONSTRAINT_FAMILIES,
    )
    place_scalars = places.taxonomy_run.scalars.count
    usable = [
        f
        for f, w in families.items()
        if w > 0 and (f != "comparison" or (taxonomy.scalars.count > 0 and place_scalars > 0))
    ]
    if count and not usable:
        raise node.error(
            "constraint_families",
            "no constraint family with positive weight can be used: the comparison family needs "
            "scalars in both the entity and the place taxonomy",
        )
    rules_node = node.mapping("rules")
    sampling = _read_sampling(rules_node, taxonomy.rules.sampling)
    rules_node.finish()
    node.finish()
    return PlaceEventTypesConfig(count, move_share, constraints, families, sampling)


def _read_event_types(node: _Node, taxonomy: TaxonomyRunConfig) -> EventTypesConfig:
    unary = _read_unary(node.mapping("unary"), taxonomy)
    binary = _read_binary(node, taxonomy)
    preconditions = _read_preconditions(node.mapping("preconditions"))
    effects = _read_effects(node.mapping("effects"))
    event_file = node.string("event_file", None, nullable=True)
    node.finish()
    return EventTypesConfig(unary, binary, preconditions, effects, event_file)


def config_from_mapping(data: Any, *, source: str = "<mapping>", seed: int | None = None) -> Config:
    """Build a world configuration from a parsed mapping. ``seed`` overrides the file's."""
    if data is None:
        data = {}
    root = _Node(source, "", data)
    name = root.string("name", "default")
    master_seed = root.int("seed", 1, min=0, max=SEED_MAX)
    if seed is not None:
        master_seed = root.check_int("seed", seed, min=0, max=SEED_MAX)
    prov = root.get("provenance", None, nullable=True)
    if prov is not None and not isinstance(prov, dict):
        raise root.error("provenance", f"expected a mapping, found {_describe(prov)}")
    taxonomy = _read_taxonomy_setting(root.mapping("taxonomy"))
    taxonomy_seed = master_seed if taxonomy.seed is None else taxonomy.seed
    taxonomy_run = taxonomy_from_mapping(
        _taxonomy_mapping(source, taxonomy, taxonomy_seed), source=taxonomy.config
    )
    fluents = _read_fluents(root.mapping("fluents"))
    numeric = _read_numeric_fluents(root.mapping("numeric_fluents"))
    event_types = _read_event_types(root.mapping("event_types"), taxonomy_run)
    places = _read_places(root, source, master_seed)
    place_event_types = _read_place_event_types(root, places, taxonomy_run)
    root.finish()
    return Config(
        name,
        master_seed,
        taxonomy,
        fluents,
        event_types,
        source,
        taxonomy_run,
        numeric,
        places,
        place_event_types,
    )


def load_config(path: str | Path, *, seed: int | None = None) -> Config:
    """Load, validate, and resolve a world configuration file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(str(path), "<file>", f"cannot read the file: {error.strerror}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from error
    return config_from_mapping(data, source=str(path), seed=seed)
