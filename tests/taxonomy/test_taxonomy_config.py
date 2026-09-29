"""Stage 1 acceptance tests: configuration loading, validation, and the resolved configuration.

The example configurations load. Deliberately broken configurations each fail with an error
that names the file and the field. The resolved configuration round-trips.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from semantic_world.taxonomy import ConfigError, config_from_mapping, load_config
from semantic_world.taxonomy.config import Range

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
EXAMPLES = sorted(DATA.glob("*.yaml"))


def test_examples_exist() -> None:
    names = {p.name for p in EXAMPLES}
    assert {"default.yaml", "tiny.yaml"} <= names


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_configurations_load(path: Path) -> None:
    config = load_config(path)
    assert config.source == str(path)
    assert config.name == path.stem


def test_default_values() -> None:
    config = load_config(DATA / "default.yaml")
    assert config.seed == 1
    features = config.features
    assert (features.is_.count, features.has.count, features.can.count) == (40, 40, 20)
    assert features.is_.determined_count == 10
    assert features.is_.free_count == 30
    assert features.base_rate("is") == pytest.approx(6 / 30)
    assert features.base_rate("has") == pytest.approx(6 / 30)
    assert features.base_rate_override is None
    assert features.base_rate_heterogeneity is None
    rules = config.rules
    assert rules.source == "automatic"
    assert rules.file is None
    assert config.rule_file_path() is None
    assert rules.max_chain_depth == 1
    assert rules.sampling.arity == {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}
    assert rules.sampling.max_arity == 4
    assert rules.sampling.operator_mix == {"AND": 1, "OR": 1, "XOR": 1}
    assert rules.sampling.negation_probability == 0.2
    assert set(rules.sampling.arity_3_families) == {
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "compositional",
    }
    assert rules.sampling.nesting_depth == {1: 0.5, 2: 0.5}
    assert rules.sampling.input_types == ("is", "has")
    assert rules.overrides == {}
    assert rules.sampling_for("can") is rules.sampling
    assert rules.allow_duplicate_rules is False
    assert rules.variance_bound is None
    assert config.taxonomy.superordinates == 4
    assert config.taxonomy.depth == 3
    assert config.taxonomy.branching == (Range(2, 4), Range(2, 4))
    bound = config.similarity_bound
    assert bound is not None
    assert (bound.metric, bound.scope, bound.min, bound.max) == ("phi", "free", None, 0.3)
    assert (bound.max_tries, bound.local_search) == (10000, True)
    inheritance = config.inheritance
    assert inheritance.proportion_defining == (0.1, 0.1, 0.1)
    assert inheritance.proportion_characteristic == (0.5, 0.5, 0.5)
    assert inheritance.characteristic_probability == (0.9, 0.9, 0.9)
    assert inheritance.require_distinct_leaves is True
    assert inheritance.distinct_max_tries == 1000
    assert config.instances.per_leaf == Range(5, 10)
    assert config.instances.characteristic_probability == 0.9
    assert config.analysis.similarity_metric == "cosine"
    assert config.analysis.similarity_features == "non_isa"
    assert config.analysis.max_pairs == 200000


def test_tiny_values() -> None:
    config = load_config(DATA / "tiny.yaml")
    features = config.features
    assert (features.is_.count, features.has.count, features.can.count) == (8, 8, 4)
    assert features.is_.determined_count == 2
    assert features.is_.free_count == 6
    assert features.base_rate("is") == pytest.approx(2 / 6)
    assert config.taxonomy.superordinates == 2
    assert config.taxonomy.depth == 2
    assert config.taxonomy.branching == (Range(2, 2),)
    assert config.instances.per_leaf == Range(3, 3)
    assert config.instances.per_leaf.fixed
    # Unlisted parameters keep their defaults.
    assert config.rules.sampling.arity == {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}
    assert config.inheritance.proportion_defining == (0.1, 0.1)


def test_default_file_equals_built_in_defaults() -> None:
    """default.yaml shows every parameter with its default value, so it equals an empty file."""
    from_file = load_config(DATA / "default.yaml").resolved()
    from_nothing = config_from_mapping({}).resolved()
    assert from_file == from_nothing


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_resolved_configuration_round_trips(path: Path, tmp_path: Path) -> None:
    config = load_config(path)
    text = config.to_yaml()
    out = tmp_path / "config.yaml"
    out.write_text(text, encoding="utf-8")
    reloaded = load_config(out)
    assert reloaded.resolved() == config.resolved()
    # Every schedule is written in its list form, with one value per level.
    resolved = config.resolved()
    depth = config.taxonomy.depth
    for key in ("proportion_defining", "proportion_characteristic", "characteristic_probability"):
        schedule = resolved["inheritance"][key]
        assert schedule["schedule"] == "list"
        assert len(schedule["values"]) == depth
    branching = resolved["taxonomy"]["branching"]
    assert branching["schedule"] == "list"
    assert len(branching["values"]) == depth - 1


def test_resolved_configuration_has_the_specified_sections_in_order() -> None:
    resolved = config_from_mapping({}).resolved()
    assert list(resolved) == [
        "name",
        "seed",
        "features",
        "rules",
        "taxonomy",
        "superordinates",
        "inheritance",
        "instances",
        "analysis",
        "scalars",
    ]


def test_seed_override() -> None:
    assert load_config(DATA / "default.yaml", seed=7).seed == 7
    assert load_config(DATA / "default.yaml", seed=2**64 - 1).seed == 2**64 - 1
    with pytest.raises(ConfigError) as info:
        load_config(DATA / "default.yaml", seed=-1)
    assert info.value.field == "seed"
    assert "default.yaml" in str(info.value)


def test_rule_file_path_is_relative_to_the_configuration_file(tmp_path: Path) -> None:
    (tmp_path / "rules.yaml").write_text("templates: []\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text("rules: {source: file, file: rules.yaml}\n", encoding="utf-8")
    config = load_config(path)
    assert config.rule_file_path() == tmp_path / "rules.yaml"
    absolute = config_from_mapping({"rules": {"source": "file", "file": "/abs/rules.yaml"}})
    assert absolute.rule_file_path() == Path("/abs/rules.yaml")


def test_overrides_merge_over_the_base_settings() -> None:
    config = config_from_mapping(
        {"rules": {"negation_probability": 0.5, "overrides": {"can": {"arity": {2: 1, 3: 1}}}}}
    )
    can = config.rules.sampling_for("can")
    assert can.arity == {2: 1, 3: 1}
    assert can.negation_probability == 0.5  # inherited from the base settings
    assert config.rules.sampling_for("is").arity == {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}
    assert config.resolved()["rules"]["overrides"]["can"]["negation_probability"] == 0.5


def test_base_rate_override_replaces_expected_true_free() -> None:
    config = config_from_mapping(
        {"features": {"is": {"expected_true_free": 100}, "base_rate_override": 0.4}}
    )
    assert config.features.base_rate("is") == 0.4
    assert config.features.base_rate("has") == 0.4


def test_no_free_features_has_no_base_rate() -> None:
    config = config_from_mapping(
        {
            "features": {
                "is": {"count": 4, "proportion_determined": 1.0},
                "has": {"count": 4, "proportion_determined": 0.0, "expected_true_free": 2},
            },
            "rules": {"arity": {1: 1, 2: 1, 3: 1, 4: 1}},
        }
    )
    assert config.features.free_count("is") == 0
    assert config.features.base_rate("is") is None
    assert config.features.base_rate("has") == pytest.approx(0.5)


def test_no_determined_features_allows_zero_chain_depth() -> None:
    config = config_from_mapping(
        {
            "features": {
                "is": {"proportion_determined": 0},
                "has": {"proportion_determined": 0},
                "can": {"count": 0},
            },
            "rules": {"max_chain_depth": 0},
        }
    )
    assert config.rules.max_chain_depth == 0


def test_role_proportions_may_sum_to_exactly_one() -> None:
    config = config_from_mapping(
        {"inheritance": {"proportion_defining": 0.4, "proportion_characteristic": 0.6}}
    )
    assert config.inheritance.proportion_defining == (0.4, 0.4, 0.4)


def test_similarity_bound_can_be_off() -> None:
    config = config_from_mapping({"superordinates": {"similarity_bound": None}})
    assert config.similarity_bound is None
    assert config.resolved()["superordinates"]["similarity_bound"] is None


def test_depth_one_has_no_branching_levels() -> None:
    config = config_from_mapping({"taxonomy": {"depth": 1, "branching": [2, 4]}})
    assert config.taxonomy.branching == ()
    assert config.inheritance.proportion_defining == (0.1,)


# ---------------------------------------------------------------------------------------------
# Broken configurations
# ---------------------------------------------------------------------------------------------


def _set(dotted: str, value: object) -> Callable[[dict], None]:
    def apply(data: dict) -> None:
        keys = dotted.split(".")
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    return apply


def _many(*patches: Callable[[dict], None]) -> Callable[[dict], None]:
    def apply(data: dict) -> None:
        for patch in patches:
            patch(data)

    return apply


BROKEN: list[tuple[str, Callable[[dict], None], str]] = [
    ("unknown top-level key", _set("colour", 1), "colour"),
    ("unknown nested key", _set("features.is.colour", 1), "features.is.colour"),
    ("null section", _set("features", None), "features"),
    ("section is a list", _set("features", []), "features"),
    ("empty name", _set("name", ""), "name"),
    ("negative seed", _set("seed", -1), "seed"),
    ("seed too large", _set("seed", 2**64), "seed"),
    ("fractional seed", _set("seed", 1.5), "seed"),
    ("negative count", _set("features.is.count", -1), "features.is.count"),
    ("fractional count", _set("features.is.count", 2.5), "features.is.count"),
    ("boolean count", _set("features.is.count", True), "features.is.count"),
    ("string count", _set("features.is.count", "40"), "features.is.count"),
    ("negative can count", _set("features.can.count", -3), "features.can.count"),
    (
        "proportion above 1",
        _set("features.is.proportion_determined", 1.5),
        "features.is.proportion_determined",
    ),
    (
        "expected true above free count",
        _set("features.has.expected_true_free", 31),
        "features.has.expected_true_free",
    ),
    (
        "base rate override above 1",
        _set("features.base_rate_override", 2),
        "features.base_rate_override",
    ),
    (
        "non-positive heterogeneity",
        _set("features.base_rate_heterogeneity", 0),
        "features.base_rate_heterogeneity",
    ),
    ("unknown rule source", _set("rules.source", "random"), "rules.source"),
    ("file source without a file", _set("rules.source", "file"), "rules.file"),
    (
        "chain depth 0 with determined features",
        _set("rules.max_chain_depth", 0),
        "rules.max_chain_depth",
    ),
    ("arity larger than the free pool", _set("rules.arity", {1: 0.1, 61: 0.9}), "rules.arity"),
    ("arity key 0", _set("rules.arity", {0: 1}), "rules.arity.0"),
    ("negative arity weight", _set("rules.arity", {2: -1}), "rules.arity.2"),
    ("all arity weights zero", _set("rules.arity", {2: 0, 3: 0}), "rules.arity"),
    ("empty arity weights", _set("rules.arity", {}), "rules.arity"),
    ("arity weights not a mapping", _set("rules.arity", [1, 2]), "rules.arity"),
    (
        "unknown operator",
        _set("rules.operator_mix", {"AND": 1, "NAND": 1}),
        "rules.operator_mix.NAND",
    ),
    (
        "negative negation probability",
        _set("rules.negation_probability", -0.1),
        "rules.negation_probability",
    ),
    ("unknown SHJ type", _set("rules.arity_3_families", {"VII": 1}), "rules.arity_3_families.VII"),
    (
        "no input types",
        _set("rules.input_type_weights", {"is": 0, "has": 0}),
        "rules.input_type_weights",
    ),
    (
        "override for a non-feature type",
        _set("rules.overrides", {"isa": {}}),
        "rules.overrides.isa",
    ),
    (
        "override arity larger than the CAN pool",
        _set("rules.overrides", {"can": {"arity": {81: 1}}}),
        "rules.overrides.can.arity",
    ),
    (
        "unknown override key",
        _set("rules.overrides", {"can": {"colour": 1}}),
        "rules.overrides.can.colour",
    ),
    ("string boolean", _set("rules.allow_duplicate_rules", "no"), "rules.allow_duplicate_rules"),
    ("reversed variance bound", _set("rules.variance_bound", [0.6, 0.4]), "rules.variance_bound"),
    ("short variance bound", _set("rules.variance_bound", [0.1]), "rules.variance_bound"),
    ("zero superordinates", _set("taxonomy.superordinates", 0), "taxonomy.superordinates"),
    ("zero depth", _set("taxonomy.depth", 0), "taxonomy.depth"),
    ("branching 0", _set("taxonomy.branching", 0), "taxonomy.branching"),
    ("branching range below 1", _set("taxonomy.branching", [0, 3]), "taxonomy.branching"),
    ("reversed branching range", _set("taxonomy.branching", [4, 2]), "taxonomy.branching"),
    ("three-element branching list", _set("taxonomy.branching", [1, 2, 3]), "taxonomy.branching"),
    ("string branching", _set("taxonomy.branching", "two"), "taxonomy.branching"),
    (
        "branching list schedule of the wrong length",
        _set("taxonomy.branching", {"schedule": "list", "values": [2]}),
        "taxonomy.branching.values",
    ),
    (
        "branching list schedule with a bad level",
        _set("taxonomy.branching", {"schedule": "list", "values": [2, [0, 2]]}),
        "taxonomy.branching.values.1",
    ),
    (
        "branching with a non-list schedule",
        _set("taxonomy.branching", {"schedule": "linear", "start": 2, "end": 4}),
        "taxonomy.branching.schedule",
    ),
    (
        "unknown similarity metric",
        _set("superordinates.similarity_bound.metric", "euclidean"),
        "superordinates.similarity_bound.metric",
    ),
    (
        "unknown similarity scope",
        _set("superordinates.similarity_bound.scope", "isa"),
        "superordinates.similarity_bound.scope",
    ),
    (
        "similarity min above max",
        _many(
            _set("superordinates.similarity_bound.min", 0.5),
            _set("superordinates.similarity_bound.max", 0.3),
        ),
        "superordinates.similarity_bound.min",
    ),
    (
        "zero max tries",
        _set("superordinates.similarity_bound.max_tries", 0),
        "superordinates.similarity_bound.max_tries",
    ),
    (
        "role proportion above 1",
        _set("inheritance.proportion_defining", 1.2),
        "inheritance.proportion_defining",
    ),
    (
        "role proportions sum above 1 at one level",
        _many(
            _set(
                "inheritance.proportion_defining", {"schedule": "list", "values": [0.1, 0.6, 0.1]}
            ),
            _set("inheritance.proportion_characteristic", 0.5),
        ),
        "inheritance.proportion_characteristic",
    ),
    (
        "schedule list of the wrong length",
        _set("inheritance.proportion_defining", {"schedule": "list", "values": [0.1, 0.1]}),
        "inheritance.proportion_defining.values",
    ),
    (
        "schedule list with a non-number",
        _set("inheritance.proportion_defining", {"schedule": "list", "values": [0.1, "x", 0.1]}),
        "inheritance.proportion_defining.values.1",
    ),
    (
        "unknown schedule kind",
        _set("inheritance.proportion_defining", {"schedule": "sigmoid", "start": 0, "end": 1}),
        "inheritance.proportion_defining.schedule",
    ),
    (
        "linear schedule missing end",
        _set("inheritance.proportion_defining", {"schedule": "linear", "start": 0.1}),
        "inheritance.proportion_defining.end",
    ),
    (
        "linear schedule with an extra key",
        _set(
            "inheritance.proportion_defining",
            {"schedule": "linear", "start": 0.1, "end": 0.2, "extra": 1},
        ),
        "inheritance.proportion_defining.extra",
    ),
    (
        "exponential schedule with a negative rate",
        _set(
            "inheritance.characteristic_probability",
            {"schedule": "exponential", "start": 0.9, "asymptote": 0.5, "rate": -1},
        ),
        "inheritance.characteristic_probability.rate",
    ),
    (
        "exponential schedule leaving the unit interval",
        _set(
            "inheritance.characteristic_probability",
            {"schedule": "exponential", "start": 2, "asymptote": 0.5, "rate": 1},
        ),
        "inheritance.characteristic_probability",
    ),
    (
        "schedule is a string",
        _set("inheritance.proportion_defining", "0.1"),
        "inheritance.proportion_defining",
    ),
    (
        "zero distinct tries",
        _set("inheritance.distinct_max_tries", 0),
        "inheritance.distinct_max_tries",
    ),
    ("negative instances", _set("instances.per_leaf", -1), "instances.per_leaf"),
    ("reversed instance range", _set("instances.per_leaf", [3, 2]), "instances.per_leaf"),
    (
        "instance probability above 1",
        _set("instances.characteristic_probability", 1.1),
        "instances.characteristic_probability",
    ),
    (
        "unknown similarity feature set",
        _set("analysis.similarity_features", "isa"),
        "analysis.similarity_features",
    ),
    ("zero max pairs", _set("analysis.max_pairs", 0), "analysis.max_pairs"),
]


@pytest.mark.parametrize(("label", "patch", "field"), BROKEN, ids=[b[0] for b in BROKEN])
def test_broken_configurations_name_the_file_and_the_field(
    label: str, patch: Callable[[dict], None], field: str, tmp_path: Path
) -> None:
    data = copy.deepcopy(yaml.safe_load((DATA / "default.yaml").read_text(encoding="utf-8")))
    patch(data)
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    error = info.value
    assert error.field == field, str(error)
    assert error.source == str(path)
    assert str(error).startswith(f"{path}: {field}: ")


def test_missing_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert info.value.field == "<file>"
    assert str(path) in str(info.value)


def test_invalid_yaml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("features: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert info.value.field == "<file>"
    assert "invalid YAML" in str(info.value)


def test_error_message_lists_valid_keys() -> None:
    with pytest.raises(ConfigError) as info:
        config_from_mapping({"analysis": {"max_pair": 5}})
    assert "max_pairs" in str(info.value)


def test_float_values_are_finite() -> None:
    with pytest.raises(ConfigError) as info:
        config_from_mapping({"rules": {"negation_probability": math.nan}})
    assert info.value.field == "rules.negation_probability"
