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

ROLES = ("agent", "patient")
REMOVED_BINARY_KEYS = ("projections",)
"""Keys of the old ``verbs`` block that the world does not take: exposure is replaced by views."""
CONSTRAINT_FAMILIES = ("agent", "patient", "cross", "key_lock", "comparison")
EVENT_FEATURE_PREFIX = "EVENTFEAT"
TWO_PLACE_PREFIX = "EVENTTYPE2."
DEFAULT_UNARY_COUNT = 20


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


@dataclass(frozen=True)
class PreconditionsConfig:
    literals: dict[int, float]
    feature_rate: float
    roles: dict[str, float]
    enabled_share: float

    def resolved(self) -> dict[str, Any]:
        return {
            "literals": dict(self.literals),
            "feature_rate": self.feature_rate,
            "roles": dict(self.roles),
            "enabled_share": self.enabled_share,
        }


@dataclass(frozen=True)
class EffectsConfig:
    count: dict[int, float]
    feature_rate: float
    roles: dict[str, float]

    def resolved(self) -> dict[str, Any]:
        return {
            "count": dict(self.count),
            "feature_rate": self.feature_rate,
            "roles": dict(self.roles),
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
            "event_types": self.event_types.resolved(),
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
    node.finish()
    return PreconditionsConfig(literals, rate, roles, enabled)


def _read_effects(node: _Node) -> EffectsConfig:
    count = _count_weights(node, "count", {1: 0.7, 2: 0.3})
    rate = node.probability("feature_rate", 0.5)
    roles = _read_roles(node, "roles", {"agent": 0.4, "patient": 0.6})
    node.finish()
    return EffectsConfig(count, rate, roles)


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
    event_types = _read_event_types(root.mapping("event_types"), taxonomy_run)
    root.finish()
    return Config(name, master_seed, taxonomy, fluents, event_types, source, taxonomy_run)


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
