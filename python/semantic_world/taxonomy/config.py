"""Configuration for the taxonomy generator.

This module loads the YAML configuration described in ``docs/specs/TAXONOMY_GENERATOR.md``,
fills in every default, validates every field, resolves the depth schedules to per-level
values, and writes the fully resolved configuration back out as YAML.

Unknown keys are errors. Every error is a :class:`ConfigError` whose message names the source
file and the dotted field path, for example ``data/taxonomy/x.yaml: features.is.count: ...``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SEED_MAX = 2**64 - 1

FEATURE_TYPES = ("is", "has", "can")
FREE_FEATURE_TYPES = ("is", "has")
INPUT_TYPES = ("is", "has", "scalar")
OPERATORS = ("AND", "OR", "XOR")
SHJ_TYPES = ("I", "II", "III", "IV", "V", "VI")
ARITY_3_FAMILIES = SHJ_TYPES + ("compositional",)
SIMILARITY_METRICS = ("phi", "cosine", "jaccard")
SIMILARITY_SCOPES = ("free", "is_has", "all")
SIMILARITY_FEATURE_SETS = ("non_isa", "all")
RULE_SOURCES = ("automatic", "file")
SCHEDULE_KINDS = ("linear", "exponential", "list")


class ConfigError(ValueError):
    """A configuration error. The message names the source file and the field."""

    def __init__(self, source: str, field: str, message: str) -> None:
        self.source = source
        self.field = field
        self.message = message
        super().__init__(f"{source}: {field}: {message}")


# ---------------------------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Range:
    """An inclusive integer range ``[min, max]``. A fixed number is the range ``[n, n]``."""

    min: int
    max: int

    @property
    def fixed(self) -> bool:
        return self.min == self.max

    def resolved(self) -> int | list[int]:
        """The range in configuration form: a number when fixed, else ``[min, max]``."""
        return self.min if self.fixed else [self.min, self.max]


@dataclass(frozen=True)
class FreeFeatureTypeConfig:
    """The IS or HAS feature settings: ``features.is`` and ``features.has``."""

    count: int
    proportion_determined: float
    expected_true_free: float

    @property
    def determined_count(self) -> int:
        """``proportion_determined × count``, rounded half up."""
        return _round_half_up(self.proportion_determined * self.count)

    @property
    def free_count(self) -> int:
        return self.count - self.determined_count


@dataclass(frozen=True)
class CanFeatureConfig:
    """The CAN feature settings: ``features.can``. Every CAN feature is determined."""

    count: int


@dataclass(frozen=True)
class FeaturesConfig:
    is_: FreeFeatureTypeConfig
    has: FreeFeatureTypeConfig
    can: CanFeatureConfig
    base_rate_override: float | None
    base_rate_heterogeneity: float | None

    def free_type(self, feature_type: str) -> FreeFeatureTypeConfig:
        if feature_type == "is":
            return self.is_
        if feature_type == "has":
            return self.has
        raise ValueError(f"{feature_type!r} is not a free feature type")

    def count(self, feature_type: str) -> int:
        if feature_type == "can":
            return self.can.count
        return self.free_type(feature_type).count

    def determined_count(self, feature_type: str) -> int:
        if feature_type == "can":
            return self.can.count
        return self.free_type(feature_type).determined_count

    def free_count(self, feature_type: str) -> int:
        if feature_type == "can":
            return 0
        return self.free_type(feature_type).free_count

    def base_rate(self, feature_type: str) -> float | None:
        """The base rate shared by the free features of a type (the Beta mean when
        heterogeneity is on), or ``None`` when the type has no free features."""
        free = self.free_count(feature_type)
        if free == 0:
            return None
        if self.base_rate_override is not None:
            return self.base_rate_override
        return self.free_type(feature_type).expected_true_free / free


@dataclass(frozen=True)
class RuleSampling:
    """The complexity parameters that automatic rule sampling draws from."""

    arity: dict[int, float]
    operator_mix: dict[str, float]
    negation_probability: float
    arity_3_families: dict[str, float]
    nesting_depth: dict[int, float]
    input_type_weights: dict[str, float]

    @property
    def max_arity(self) -> int:
        return max(arity for arity, weight in self.arity.items() if weight > 0)

    @property
    def input_types(self) -> tuple[str, ...]:
        """The binary input feature types with nonzero weight, in ``(is, has)`` order."""
        return tuple(t for t in FREE_FEATURE_TYPES if self.input_type_weights[t] > 0)

    @property
    def scalar_weight(self) -> float:
        """The weight of scalar threshold literals in the input pool (no effect without scalars)."""
        return self.input_type_weights.get("scalar", 1.0)

    def resolved(self) -> dict[str, Any]:
        return {
            "arity": dict(self.arity),
            "operator_mix": dict(self.operator_mix),
            "negation_probability": self.negation_probability,
            "arity_3_families": dict(self.arity_3_families),
            "nesting_depth": dict(self.nesting_depth),
            "input_type_weights": dict(self.input_type_weights),
        }


@dataclass(frozen=True)
class RulesConfig:
    source: str
    file: str | None
    max_chain_depth: int
    sampling: RuleSampling
    overrides: dict[str, RuleSampling]
    allow_duplicate_rules: bool
    variance_bound: tuple[float, float] | None

    def sampling_for(self, output_type: str) -> RuleSampling:
        """The effective sampling settings for rules whose output has the given type."""
        if output_type not in FEATURE_TYPES:
            raise ValueError(f"{output_type!r} is not a feature type")
        return self.overrides.get(output_type, self.sampling)


@dataclass(frozen=True)
class TaxonomyConfig:
    superordinates: int
    depth: int
    branching: tuple[Range, ...]
    """The branching range at each level from 1 to ``depth - 1``."""


@dataclass(frozen=True)
class SimilarityBound:
    metric: str
    scope: str
    min: float | None
    max: float | None
    max_tries: int
    local_search: bool


@dataclass(frozen=True)
class InheritanceConfig:
    proportion_defining: tuple[float, ...]
    proportion_characteristic: tuple[float, ...]
    characteristic_probability: tuple[float, ...]
    """Each schedule resolved to one value per level, from level 1 to ``depth``."""
    require_distinct_leaves: bool
    distinct_max_tries: int


@dataclass(frozen=True)
class InstancesConfig:
    per_leaf: Range
    characteristic_probability: float


@dataclass(frozen=True)
class AnalysisConfig:
    similarity_metric: str
    similarity_features: str
    max_pairs: int


@dataclass(frozen=True)
class ScalarsConfig:
    """Scalar dimensions (``docs/specs/TAXONOMY_RELATIONS.md``, Part A)."""

    count: int
    drift: tuple[float, ...]
    """The standard deviation of a child's change from its parent, for parent levels 1 to
    ``depth - 1``."""
    instance_drift: float
    threshold_quantiles: tuple[float, float]
    thermometer_bins: int

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f"SC.{i}" for i in range(1, self.count + 1))

    def fixed_below(self, level: int) -> bool:
        """Whether a scalar is fixed for the members of a category at ``level``: every drift
        below that level, including the instance drift, is 0."""
        return all(d == 0 for d in self.drift[level - 1 :]) and self.instance_drift == 0


@dataclass(frozen=True)
class Config:
    """A fully validated and resolved run configuration."""

    name: str
    seed: int
    features: FeaturesConfig
    rules: RulesConfig
    taxonomy: TaxonomyConfig
    similarity_bound: SimilarityBound | None
    inheritance: InheritanceConfig
    instances: InstancesConfig
    analysis: AnalysisConfig
    scalars: ScalarsConfig
    source: str
    """Where the configuration came from: the file path, or a label for an in-memory mapping."""

    def rule_file_path(self) -> Path | None:
        """The rule file, resolved relative to the configuration file's folder."""
        if self.rules.file is None:
            return None
        path = Path(self.rules.file)
        if path.is_absolute():
            return path
        source = Path(self.source)
        base = source.parent if source.suffix else Path.cwd()
        return base / path

    def resolved(self) -> dict[str, Any]:
        """The fully resolved configuration as a plain mapping.

        Every default is filled in and every schedule is written in its ``list`` form with one
        value per level. The mapping loads back through :func:`config_from_mapping` to an equal
        configuration.
        """
        features = self.features
        rules = self.rules
        return {
            "name": self.name,
            "seed": self.seed,
            "features": {
                "is": _resolved_free_type(features.is_),
                "has": _resolved_free_type(features.has),
                "can": {"count": features.can.count},
                "base_rate_override": features.base_rate_override,
                "base_rate_heterogeneity": features.base_rate_heterogeneity,
            },
            "rules": {
                "source": rules.source,
                "file": rules.file,
                "max_chain_depth": rules.max_chain_depth,
                **rules.sampling.resolved(),
                "overrides": {t: s.resolved() for t, s in rules.overrides.items()},
                "allow_duplicate_rules": rules.allow_duplicate_rules,
                "variance_bound": (
                    None if rules.variance_bound is None else list(rules.variance_bound)
                ),
            },
            "taxonomy": {
                "superordinates": self.taxonomy.superordinates,
                "depth": self.taxonomy.depth,
                "branching": _list_schedule([r.resolved() for r in self.taxonomy.branching]),
            },
            "superordinates": {
                "similarity_bound": (
                    None
                    if self.similarity_bound is None
                    else {
                        "metric": self.similarity_bound.metric,
                        "scope": self.similarity_bound.scope,
                        "min": self.similarity_bound.min,
                        "max": self.similarity_bound.max,
                        "max_tries": self.similarity_bound.max_tries,
                        "local_search": self.similarity_bound.local_search,
                    }
                ),
            },
            "inheritance": {
                "proportion_defining": _list_schedule(self.inheritance.proportion_defining),
                "proportion_characteristic": _list_schedule(
                    self.inheritance.proportion_characteristic
                ),
                "characteristic_probability": _list_schedule(
                    self.inheritance.characteristic_probability
                ),
                "require_distinct_leaves": self.inheritance.require_distinct_leaves,
                "distinct_max_tries": self.inheritance.distinct_max_tries,
            },
            "instances": {
                "per_leaf": self.instances.per_leaf.resolved(),
                "characteristic_probability": self.instances.characteristic_probability,
            },
            "analysis": {
                "similarity_metric": self.analysis.similarity_metric,
                "similarity_features": self.analysis.similarity_features,
                "max_pairs": self.analysis.max_pairs,
            },
            "scalars": {
                "count": self.scalars.count,
                "drift": _list_schedule(self.scalars.drift),
                "instance_drift": self.scalars.instance_drift,
                "threshold_quantiles": list(self.scalars.threshold_quantiles),
                "thermometer_bins": self.scalars.thermometer_bins,
            },
        }

    def to_yaml(self) -> str:
        """The resolved configuration as YAML text, in the order of the specification."""
        return yaml.safe_dump(self.resolved(), sort_keys=False, default_flow_style=None)


def _resolved_free_type(cfg: FreeFeatureTypeConfig) -> dict[str, Any]:
    return {
        "count": cfg.count,
        "proportion_determined": cfg.proportion_determined,
        "expected_true_free": cfg.expected_true_free,
    }


def _list_schedule(values) -> dict[str, Any]:
    return {"schedule": "list", "values": list(values)}


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


# ---------------------------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------------------------

DEFAULT_SAMPLING = RuleSampling(
    arity={1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2},
    operator_mix={"AND": 1, "OR": 1, "XOR": 1},
    negation_probability=0.2,
    arity_3_families={"I": 1, "II": 1, "III": 1, "IV": 1, "V": 1, "VI": 1, "compositional": 1},
    nesting_depth={1: 0.5, 2: 0.5},
    input_type_weights={"is": 1, "has": 1, "scalar": 1},
)

DEFAULT_SIMILARITY_BOUND = {
    "metric": "phi",
    "scope": "free",
    "min": None,
    "max": 0.3,
    "max_tries": 10000,
    "local_search": True,
}


# ---------------------------------------------------------------------------------------------
# Reading with validation
# ---------------------------------------------------------------------------------------------

_MISSING = object()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"the boolean {str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"the number {value!r}"
    if isinstance(value, str):
        return f"the string {value!r}"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return f"a value of type {type(value).__name__}"


class _Node:
    """One mapping of the configuration, read field by field with defaults and checks."""

    def __init__(self, source: str, path: str, data: Any) -> None:
        if not isinstance(data, dict):
            raise ConfigError(
                source, path or "<root>", f"expected a mapping, found {_describe(data)}"
            )
        self.source = source
        self.path = path
        self.data = data
        self.seen: set[Any] = set()

    def field(self, key: Any) -> str:
        return f"{self.path}.{key}" if self.path else str(key)

    def error(self, key: Any, message: str) -> ConfigError:
        return ConfigError(self.source, self.field(key), message)

    def get(self, key: str, default: Any = _MISSING, *, nullable: bool = False) -> Any:
        self.seen.add(key)
        if key not in self.data:
            if default is _MISSING:
                raise self.error(key, "is required")
            return default
        value = self.data[key]
        if value is None and not nullable:
            raise self.error(key, "must not be null")
        return value

    def finish(self) -> None:
        """Reject keys that were never read. Call after reading every valid key."""
        for key in self.data:
            if key not in self.seen:
                valid = ", ".join(sorted(str(k) for k in self.seen))
                raise self.error(key, f"unknown key; valid keys are: {valid}")

    # Scalars ---------------------------------------------------------------------------------

    def int(self, key: str, default: Any, *, min: int | None = None, max: int | None = None) -> int:
        value = self.get(key, default)
        return self.check_int(key, value, min=min, max=max)

    def check_int(
        self, key: Any, value: Any, *, min: int | None = None, max: int | None = None
    ) -> int:
        if not _is_int(value):
            raise self.error(key, f"expected an integer, found {_describe(value)}")
        if min is not None and value < min:
            raise self.error(key, f"must be at least {min}, found {value}")
        if max is not None and value > max:
            raise self.error(key, f"must be at most {max}, found {value}")
        return value

    def number(
        self,
        key: str,
        default: Any,
        *,
        min: float | None = None,
        max: float | None = None,
        nullable: bool = False,
    ) -> float | None:
        value = self.get(key, default, nullable=nullable)
        if value is None:
            return None
        return self.check_number(key, value, min=min, max=max)

    def check_number(
        self, key: Any, value: Any, *, min: float | None = None, max: float | None = None
    ) -> float:
        if not _is_number(value):
            raise self.error(key, f"expected a number, found {_describe(value)}")
        if min is not None and value < min:
            raise self.error(key, f"must be at least {min}, found {value}")
        if max is not None and value > max:
            raise self.error(key, f"must be at most {max}, found {value}")
        return value

    def probability(self, key: str, default: Any, *, nullable: bool = False) -> float | None:
        return self.number(key, default, min=0, max=1, nullable=nullable)

    def bool(self, key: str, default: Any) -> bool:
        value = self.get(key, default)
        if not isinstance(value, bool):
            raise self.error(key, f"expected true or false, found {_describe(value)}")
        return value

    def string(self, key: str, default: Any, *, nullable: bool = False) -> str | None:
        value = self.get(key, default, nullable=nullable)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise self.error(key, f"expected a non-empty string, found {_describe(value)}")
        return value

    def choice(self, key: str, default: Any, choices: tuple[str, ...]) -> str:
        value = self.get(key, default)
        if not isinstance(value, str) or value not in choices:
            options = ", ".join(choices)
            raise self.error(key, f"expected one of {options}, found {_describe(value)}")
        return value

    # Compound values -------------------------------------------------------------------------

    def mapping(self, key: str, *, nullable: bool = False) -> _Node | None:
        value = self.get(key, {}, nullable=nullable)
        if value is None:
            return None
        return _Node(self.source, self.field(key), value)

    def weights(
        self,
        key: str,
        default: dict,
        *,
        int_keys: bool = False,
        allowed: tuple[str, ...] | None = None,
    ) -> dict:
        """A mapping from names (or positive integers) to non-negative weights, at least one
        positive. Integer keys come back sorted, and named keys in their allowed order."""
        value = self.get(key, default)
        if not isinstance(value, dict):
            raise self.error(key, f"expected a mapping of weights, found {_describe(value)}")
        node = _Node(self.source, self.field(key), value)
        result: dict = {}
        for name, weight in value.items():
            if int_keys:
                if not _is_int(name) or name < 1:
                    raise node.error(name, "keys must be positive integers")
            elif allowed is not None and name not in allowed:
                raise node.error(name, f"unknown key; valid keys are: {', '.join(allowed)}")
            result[name] = node.check_number(name, weight, min=0)
        if not result:
            raise self.error(key, "must list at least one weight")
        if all(w == 0 for w in result.values()):
            raise self.error(key, "at least one weight must be positive")
        if int_keys:
            return {k: result[k] for k in sorted(result)}
        order = allowed if allowed is not None else tuple(result)
        return {k: result[k] for k in order if k in result}

    def range(
        self, key: str, default: Any, *, min: int = 0, nullable: bool = False
    ) -> Range | None:
        """A fixed number or a two-element list ``[min, max]`` of integers."""
        value = self.get(key, default, nullable=nullable)
        if value is None:
            return None
        return self.check_range(key, value, min=min)

    def check_range(self, key: Any, value: Any, *, min: int = 0) -> Range:
        if _is_int(value):
            self.check_int(key, value, min=min)
            return Range(value, value)
        if isinstance(value, list) and len(value) == 2:
            lo, hi = value
            lo = self.check_int(key, lo, min=min)
            hi = self.check_int(key, hi, min=min)
            if lo > hi:
                raise self.error(key, f"the range minimum {lo} exceeds the maximum {hi}")
            return Range(lo, hi)
        raise self.error(key, f"expected a number or a range [min, max], found {_describe(value)}")

    def unit_interval(self, key: str, default: Any) -> tuple[float, float] | None:
        """A nullable ``[low, high]`` pair of probabilities with ``low <= high``."""
        value = self.get(key, default, nullable=True)
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise self.error(key, f"expected null or a pair [low, high], found {_describe(value)}")
        low = self.check_number(key, value[0], min=0, max=1)
        high = self.check_number(key, value[1], min=0, max=1)
        if low > high:
            raise self.error(key, f"low {low} exceeds high {high}")
        return (low, high)

    def schedule(self, key: str, default: Any, depth: int) -> tuple[float, ...]:
        """A depth schedule resolved to one value per level. See :func:`resolve_schedule`."""
        value = self.get(key, default)
        return resolve_schedule(value, depth, self.source, self.field(key))


# ---------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------


def resolve_schedule(value: Any, depth: int, source: str, field: str) -> tuple[float, ...]:
    """Resolve one of the four schedule forms to a value for each level 1 to ``depth``.

    - a bare number: constant at every level;
    - ``{schedule: linear, start: a, end: b}``: ``a`` at level 1, ``b`` at level ``depth``;
    - ``{schedule: exponential, start: a, asymptote: c, rate: r}``:
      ``c + (a - c) * exp(-r * (L - 1))`` at level ``L``;
    - ``{schedule: list, values: [...]}``: one value per level, exactly ``depth`` values.
    """
    if _is_number(value):
        return (float(value),) * depth
    if not isinstance(value, dict):
        raise ConfigError(
            source, field, f"expected a number or a schedule mapping, found {_describe(value)}"
        )
    node = _Node(source, field, value)
    kind = node.choice("schedule", _MISSING, SCHEDULE_KINDS)
    if kind == "linear":
        start = node.number("start", _MISSING)
        end = node.number("end", _MISSING)
        node.finish()
        if depth == 1:
            return (float(start),)
        step = (end - start) / (depth - 1)
        return tuple(float(start + step * level) for level in range(depth))
    if kind == "exponential":
        start = node.number("start", _MISSING)
        asymptote = node.number("asymptote", _MISSING)
        rate = node.number("rate", _MISSING, min=0)
        node.finish()
        return tuple(
            float(asymptote + (start - asymptote) * math.exp(-rate * level))
            for level in range(depth)
        )
    values = node.get("values", _MISSING)
    node.finish()
    if not isinstance(values, list):
        raise node.error("values", f"expected a list, found {_describe(values)}")
    if len(values) != depth:
        raise node.error(
            "values", f"expected exactly {depth} values (one per level), found {len(values)}"
        )
    items = _Node(source, node.field("values"), dict(enumerate(values)))
    return tuple(float(items.check_number(i, v)) for i, v in enumerate(values))


def resolve_branching(value: Any, depth: int, source: str, field: str) -> tuple[Range, ...]:
    """Resolve the branching parameter to one range per level from 1 to ``depth - 1``.

    A bare number or a two-element list ``[min, max]`` applies to every level. The form
    ``{schedule: list, values: [...]}`` gives one number or range per level, with exactly
    ``depth - 1`` values. Every minimum is at least 1, so every leaf is at level ``depth``.
    """
    levels = depth - 1
    parent, _, key = field.rpartition(".")
    node = _Node(source, parent, {})
    if _is_int(value) or isinstance(value, list):
        return (node.check_range(key, value, min=1),) * levels if levels else ()
    if not isinstance(value, dict):
        raise ConfigError(
            source,
            field,
            f"expected a number, a range [min, max], or a list schedule, found {_describe(value)}",
        )
    inner = _Node(source, field, value)
    kind = inner.choice("schedule", _MISSING, ("list",))
    assert kind == "list"
    values = inner.get("values", _MISSING)
    inner.finish()
    if not isinstance(values, list):
        raise inner.error("values", f"expected a list, found {_describe(values)}")
    if len(values) != levels:
        raise inner.error(
            "values",
            f"expected exactly {levels} values (one per level 1..depth-1), found {len(values)}",
        )
    items = _Node(source, inner.field("values"), dict(enumerate(values)))
    return tuple(items.check_range(i, v, min=1) for i, v in enumerate(values))


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def _read_free_type(node: _Node) -> FreeFeatureTypeConfig:
    count = node.int("count", 40, min=0)
    proportion = node.probability("proportion_determined", 0.25)
    expected = node.number("expected_true_free", 6, min=0)
    node.finish()
    return FreeFeatureTypeConfig(count, proportion, expected)


def _read_features(node: _Node) -> FeaturesConfig:
    is_node = node.mapping("is")
    has_node = node.mapping("has")
    can_node = node.mapping("can")
    is_ = _read_free_type(is_node)
    has = _read_free_type(has_node)
    can = CanFeatureConfig(can_node.int("count", 20, min=0))
    can_node.finish()
    override = node.probability("base_rate_override", None, nullable=True)
    heterogeneity = node.number("base_rate_heterogeneity", None, nullable=True)
    if heterogeneity is not None and heterogeneity <= 0:
        raise node.error(
            "base_rate_heterogeneity",
            f"the Beta concentration must be positive, found {heterogeneity}",
        )
    node.finish()
    if override is None:
        for type_node, cfg in ((is_node, is_), (has_node, has)):
            if cfg.free_count > 0 and cfg.expected_true_free > cfg.free_count:
                raise type_node.error(
                    "expected_true_free",
                    f"must be at most the number of free features ({cfg.free_count}), "
                    f"found {cfg.expected_true_free}",
                )
    return FeaturesConfig(is_, has, can, override, heterogeneity)


def _read_sampling(node: _Node, base: RuleSampling) -> RuleSampling:
    return RuleSampling(
        arity=node.weights("arity", base.arity, int_keys=True),
        operator_mix=node.weights("operator_mix", base.operator_mix, allowed=OPERATORS),
        negation_probability=node.probability("negation_probability", base.negation_probability),
        arity_3_families=node.weights(
            "arity_3_families", base.arity_3_families, allowed=ARITY_3_FAMILIES
        ),
        nesting_depth=node.weights("nesting_depth", base.nesting_depth, int_keys=True),
        input_type_weights=_read_input_type_weights(node, base),
    )


def _read_input_type_weights(node: _Node, base: RuleSampling) -> dict[str, float]:
    """``is`` and ``has`` weights, plus a ``scalar`` weight that defaults to 1 when absent."""
    weights = node.weights("input_type_weights", base.input_type_weights, allowed=INPUT_TYPES)
    if all(weights.get(t, 0) == 0 for t in FREE_FEATURE_TYPES):
        raise node.error("input_type_weights", "at least one of is and has must be positive")
    weights.setdefault("scalar", 1.0)
    return {t: weights[t] for t in INPUT_TYPES}


def _check_arity_pool(
    node: _Node, output_type: str, sampling: RuleSampling, features: FeaturesConfig
) -> None:
    """The largest arity with nonzero weight must fit the smallest eligible input pool."""
    types = sampling.input_types
    if output_type == "can":
        pool = sum(features.count(t) for t in types)
        what = "IS or HAS features"
    else:
        pool = sum(features.free_count(t) for t in types)
        what = "free IS or HAS features (layer 0)"
    if len(types) == 1:
        what = what.replace("IS or HAS", types[0].upper())
    if sampling.max_arity > pool:
        raise node.error(
            "arity",
            f"the largest arity with nonzero weight is {sampling.max_arity}, but a rule for a "
            f"{output_type.upper()} feature can read at most {pool} {what}",
        )


def _read_rules(node: _Node, features: FeaturesConfig) -> RulesConfig:
    source = node.choice("source", "automatic", RULE_SOURCES)
    file = node.string("file", None, nullable=True)
    if source == "file" and file is None:
        raise node.error("file", "a rule file path is required when rules.source is file")
    max_chain_depth = node.int("max_chain_depth", 1, min=0)
    sampling = _read_sampling(node, DEFAULT_SAMPLING)
    overrides_node = node.mapping("overrides")
    overrides: dict[str, RuleSampling] = {}
    for feature_type in FEATURE_TYPES:
        if feature_type in overrides_node.data:
            type_node = overrides_node.mapping(feature_type)
            overrides[feature_type] = _read_sampling(type_node, sampling)
            type_node.finish()
    overrides_node.finish()
    allow_duplicates = node.bool("allow_duplicate_rules", False)
    variance_bound = node.unit_interval("variance_bound", None)
    node.finish()

    determined = features.determined_count("is") + features.determined_count("has")
    if determined > 0 and max_chain_depth < 1:
        raise node.error(
            "max_chain_depth",
            f"must be at least 1 when any IS or HAS feature is determined ({determined} are)",
        )
    for feature_type in FEATURE_TYPES:
        if features.determined_count(feature_type) == 0:
            continue
        if feature_type in overrides:
            _check_arity_pool(
                overrides_node.mapping(feature_type),
                feature_type,
                overrides[feature_type],
                features,
            )
        else:
            _check_arity_pool(node, feature_type, sampling, features)
    return RulesConfig(
        source, file, max_chain_depth, sampling, overrides, allow_duplicates, variance_bound
    )


def _read_taxonomy(node: _Node) -> TaxonomyConfig:
    superordinates = node.int("superordinates", 4, min=1)
    depth = node.int("depth", 3, min=1)
    branching = resolve_branching(
        node.get("branching", [2, 4]), depth, node.source, node.field("branching")
    )
    node.finish()
    return TaxonomyConfig(superordinates, depth, branching)


def _read_similarity_bound(node: _Node) -> SimilarityBound | None:
    bound = node.mapping("similarity_bound", nullable=True)
    node.finish()
    if bound is None:
        return None
    d = DEFAULT_SIMILARITY_BOUND
    metric = bound.choice("metric", d["metric"], SIMILARITY_METRICS)
    scope = bound.choice("scope", d["scope"], SIMILARITY_SCOPES)
    low = bound.number("min", d["min"], nullable=True)
    high = bound.number("max", d["max"], nullable=True)
    max_tries = bound.int("max_tries", d["max_tries"], min=1)
    local_search = bound.bool("local_search", d["local_search"])
    bound.finish()
    if low is not None and high is not None and low > high:
        raise bound.error("min", f"min {low} exceeds max {high}")
    return SimilarityBound(metric, scope, low, high, max_tries, local_search)


def _read_inheritance(node: _Node, depth: int) -> InheritanceConfig:
    defining = node.schedule("proportion_defining", 0.1, depth)
    characteristic = node.schedule("proportion_characteristic", 0.5, depth)
    copy_probability = node.schedule("characteristic_probability", 0.9, depth)
    require_distinct = node.bool("require_distinct_leaves", True)
    max_tries = node.int("distinct_max_tries", 1000, min=1)
    node.finish()
    for key, values in (
        ("proportion_defining", defining),
        ("proportion_characteristic", characteristic),
        ("characteristic_probability", copy_probability),
    ):
        for level, value in enumerate(values, start=1):
            if not 0 <= value <= 1:
                raise node.error(
                    key, f"must be between 0 and 1 at every level; level {level} is {value}"
                )
    for level, (d, c) in enumerate(zip(defining, characteristic, strict=True), start=1):
        if d + c > 1 + 1e-12:
            raise node.error(
                "proportion_characteristic",
                f"proportion_defining + proportion_characteristic must be at most 1 at every "
                f"level; level {level} is {d} + {c} = {d + c}",
            )
    return InheritanceConfig(
        defining, characteristic, copy_probability, require_distinct, max_tries
    )


def _read_instances(node: _Node) -> InstancesConfig:
    per_leaf = node.range("per_leaf", [5, 10], min=0)
    probability = node.probability("characteristic_probability", 0.9)
    node.finish()
    return InstancesConfig(per_leaf, probability)


def _read_scalars(node: _Node, depth: int) -> ScalarsConfig:
    count = node.int("count", 0, min=0)
    drift = node.schedule("drift", 0.5, depth - 1)
    for level, value in enumerate(drift, start=1):
        if value < 0:
            raise node.error(
                "drift", f"must be non-negative at every level; level {level} is {value}"
            )
    instance_drift = node.number("instance_drift", 0.2, min=0)
    quantiles = _read_open_interval(node, "threshold_quantiles", [0.2, 0.8])
    bins = node.int("thermometer_bins", 0, min=0)
    node.finish()
    return ScalarsConfig(count, drift, instance_drift, quantiles, bins)


def _read_open_interval(node: _Node, key: str, default: Any) -> tuple[float, float]:
    """A pair ``[low, high]`` of quantiles inside (0, 1) with ``low < high``."""
    value = node.get(key, default)
    if not isinstance(value, list) or len(value) != 2:
        raise node.error(key, f"expected a pair [low, high], found {_describe(value)}")
    low = node.check_number(key, value[0])
    high = node.check_number(key, value[1])
    if not 0 < low < high < 1:
        raise node.error(key, f"expected 0 < low < high < 1, found [{low}, {high}]")
    return (low, high)


def _read_analysis(node: _Node) -> AnalysisConfig:
    metric = node.choice("similarity_metric", "cosine", SIMILARITY_METRICS)
    feature_set = node.choice("similarity_features", "non_isa", SIMILARITY_FEATURE_SETS)
    max_pairs = node.int("max_pairs", 200000, min=1)
    node.finish()
    return AnalysisConfig(metric, feature_set, max_pairs)


# ---------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------


def config_from_mapping(data: Any, *, source: str = "<mapping>", seed: int | None = None) -> Config:
    """Build a configuration from an already parsed mapping.

    ``source`` names the origin in error messages. ``seed`` overrides the file's master seed
    (the command line's ``--seed``).
    """
    if data is None:
        data = {}
    root = _Node(source, "", data)
    name = root.string("name", "default")
    master_seed = root.int("seed", 1, min=0, max=SEED_MAX)
    # A run's config.yaml records its provenance (git commit, package version, stream seeds)
    # under this key, so that a run folder's configuration loads back unchanged.
    prov = root.get("provenance", None, nullable=True)
    if prov is not None and not isinstance(prov, dict):
        raise root.error("provenance", f"expected a mapping, found {_describe(prov)}")
    if seed is not None:
        master_seed = root.check_int("seed", seed, min=0, max=SEED_MAX)
    features = _read_features(root.mapping("features"))
    rules = _read_rules(root.mapping("rules"), features)
    taxonomy = _read_taxonomy(root.mapping("taxonomy"))
    similarity_bound = _read_similarity_bound(root.mapping("superordinates"))
    inheritance = _read_inheritance(root.mapping("inheritance"), taxonomy.depth)
    instances = _read_instances(root.mapping("instances"))
    analysis = _read_analysis(root.mapping("analysis"))
    scalars = _read_scalars(root.mapping("scalars"), taxonomy.depth)
    root.finish()
    return Config(
        name=name,
        seed=master_seed,
        features=features,
        rules=rules,
        taxonomy=taxonomy,
        similarity_bound=similarity_bound,
        inheritance=inheritance,
        instances=instances,
        analysis=analysis,
        scalars=scalars,
        source=source,
    )


def load_config(path: str | Path, *, seed: int | None = None) -> Config:
    """Load, validate, and resolve a YAML configuration file."""
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
