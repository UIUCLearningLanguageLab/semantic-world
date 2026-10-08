"""The world configuration: ``data/world/default.yaml`` and its validation.

A world names a taxonomy configuration and a seed, the fluents, and the event-type settings. The
blocks ``event_types.unary`` and ``event_types.binary`` are the taxonomy's old CAN and verbs
settings, moved: they are merged over the ``features.can`` and ``verbs`` blocks of the taxonomy
configuration before the taxonomy is generated. Validation follows the taxonomy's conventions:
unknown keys are errors, and every error names the file and the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_world.taxonomy.config import SEED_MAX, ConfigError, _describe, _Node
from semantic_world.taxonomy.config import Config as TaxonomyConfig
from semantic_world.taxonomy.config import config_from_mapping as taxonomy_from_mapping

ROLES = ("agent", "patient")
REMOVED_BINARY_KEYS = ("projections",)
"""Keys of the old ``verbs`` block that the world does not take: exposure is replaced by views."""


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
class EventTypesConfig:
    unary: dict[str, Any]
    """Overrides of the taxonomy's ``features.can`` block (``count``) and of its
    ``rules.overrides.can`` block (``rules``)."""
    binary: dict[str, Any] | None
    """The old ``verbs`` block, merged over the taxonomy's; None means no two-place event
    types."""
    preconditions: PreconditionsConfig
    effects: EffectsConfig
    event_file: str | None

    def resolved(self) -> dict[str, Any]:
        return {
            "unary": dict(self.unary),
            "binary": None if self.binary is None else dict(self.binary),
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

    def taxonomy_mapping(self) -> dict[str, Any]:
        """The taxonomy configuration as a mapping: the named file with the world's overrides
        applied (the seed, ``features.can``, ``rules.overrides.can``, and ``verbs``)."""
        path = Path(self.taxonomy.config)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ConfigError(
                self.source, "taxonomy.config", f"cannot read {path}: {error.strerror}"
            ) from error
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as error:
            raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from error
        if not isinstance(data, dict):
            raise ConfigError(str(path), "<root>", f"expected a mapping, found {_describe(data)}")
        data = dict(data)
        data["seed"] = self.taxonomy_seed
        data.pop("provenance", None)
        unary = dict(self.event_types.unary)
        if "count" in unary:
            features = dict(data.get("features") or {})
            features["can"] = {**(features.get("can") or {}), "count": unary["count"]}
            data["features"] = features
        if "rules" in unary:
            rules = dict(data.get("rules") or {})
            overrides = dict(rules.get("overrides") or {})
            overrides["can"] = {**(overrides.get("can") or {}), **unary["rules"]}
            rules["overrides"] = overrides
            data["rules"] = rules
        binary = self.event_types.binary
        if binary is None:
            data["verbs"] = None
        else:
            data["verbs"] = _merge(data.get("verbs") or {}, binary)
        return data

    def taxonomy_config(self) -> TaxonomyConfig:
        """The validated taxonomy configuration the world generates from."""
        return taxonomy_from_mapping(self.taxonomy_mapping(), source=self.taxonomy.config)

    def resolved(self) -> dict[str, Any]:
        """The configuration as a plain mapping, with the resolved taxonomy configuration under
        ``taxonomy.resolved``. The mapping loads back to an equal configuration."""
        return {
            "name": self.name,
            "seed": self.seed,
            "taxonomy": {
                "config": self.taxonomy.config,
                "seed": self.taxonomy.seed,
                "resolved": self.taxonomy_config().resolved(),
            },
            "fluents": self.fluents.resolved(),
            "event_types": self.event_types.resolved(),
        }


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Mappings merge key by key; any other value is replaced."""
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


def _read_taxonomy(node: _Node) -> TaxonomySetting:
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


def _read_unary(node: _Node) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "count" in node.data:
        result["count"] = node.int("count", 0, min=0)
    if "rules" in node.data:
        rules = node.get("rules")
        if not isinstance(rules, dict):
            raise node.error("rules", f"expected a mapping, found {_describe(rules)}")
        result["rules"] = dict(rules)
    node.finish()
    return result


def _read_binary(root: _Node) -> dict[str, Any] | None:
    value = root.get("binary", {}, nullable=True)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise root.error("binary", f"expected a mapping or null, found {_describe(value)}")
    for key in REMOVED_BINARY_KEYS:
        if key in value:
            raise ConfigError(
                root.source,
                f"{root.field('binary')}.{key}",
                "exposure is removed from the world (REL.16): views choose the columns a model "
                "sees; drop this key",
            )
    return dict(value)


def _read_event_types(node: _Node) -> EventTypesConfig:
    unary = _read_unary(node.mapping("unary"))
    binary = _read_binary(node)
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
    taxonomy = _read_taxonomy(root.mapping("taxonomy"))
    fluents = _read_fluents(root.mapping("fluents"))
    event_types = _read_event_types(root.mapping("event_types"))
    root.finish()
    config = Config(name, master_seed, taxonomy, fluents, event_types, source)
    config.taxonomy_config()  # validate the merged taxonomy configuration now
    return config


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
