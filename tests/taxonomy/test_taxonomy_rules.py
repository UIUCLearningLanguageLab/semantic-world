"""Stage 3 acceptance tests: feature layout, rule sampling, duplicate checks, the variance
bound, and rule files.

Since stage a5b of the world model the taxonomy has PROPERTY and PART features only: the CAN rules
are the one-place requirements of the world package. Tests that read base rates set
``features.base_rate_heterogeneity`` to null, because the default (2) draws them from a Beta
distribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic_world.common.boolean import dnf_literal_count, minimal_dnf
from semantic_world.taxonomy import (
    ConfigError,
    GenerationError,
    Streams,
    build_features,
    config_from_mapping,
    generate_rules,
    load_config,
)
from semantic_world.taxonomy.expressions import from_dnf, parse_expression
from semantic_world.taxonomy.rule_files import load_rule_file, rule_file_from_mapping
from semantic_world.taxonomy.rules import RuleSet, canonical_key

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"

EQUAL_RATES = {"base_rate_heterogeneity": None}
CHAINED = {
    "features": {
        "property": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
        "part": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
    },
    "rules": {"max_chain_depth": 3},
}


def make(overrides: dict | None = None, seed: int = 1) -> RuleSet:
    config = config_from_mapping(overrides or {}, seed=seed)
    return generate_rules(config, Streams(config.seed))


# ---------------------------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------------------------


def test_default_feature_layout() -> None:
    config = config_from_mapping({"features": EQUAL_RATES})
    features = build_features(config, Streams(1))
    assert len(features) == 80
    assert features.labels[:3] == ("PROPERTY.1", "PROPERTY.2", "PROPERTY.3")
    assert features.labels[40:42] == ("PART.1", "PART.2")
    assert features.labels[-1] == "PART.40"
    assert [f.position for f in features.features] == list(range(80))
    # Free features first, then determined, within each type.
    assert all(features[f"PROPERTY.{i}"].free for i in range(1, 31))
    assert all(not features[f"PROPERTY.{i}"].free for i in range(31, 41))
    assert all(features[f"PROPERTY.{i}"].layer == 1 for i in range(31, 41))
    assert all(features[f"PART.{i}"].layer == (0 if i <= 30 else 1) for i in range(1, 41))
    assert features.layers == 1
    assert len(features.free) == 60
    assert features.base_rates.tolist() == [pytest.approx(0.2)] * 60
    assert all(f.base_rate is None for f in features.determined)
    order = [f.label for f in features.determined]
    assert order[:10] == [f"PROPERTY.{i}" for i in range(31, 41)]
    assert order[10:] == [f"PART.{i}" for i in range(31, 41)]
    assert features.warnings == ()
    assert not any(f.label.startswith("CAN.") for f in features.features)
    with pytest.raises(KeyError):
        features["ISA.CATEGORY.1"]
    with pytest.raises(KeyError):
        features["CAN.1"]


def test_layers_are_split_evenly_lowest_first() -> None:
    config = config_from_mapping(CHAINED)
    sizes_seen = set()
    type_splits = set()
    for seed in range(5):
        features = build_features(config, Streams(seed))
        sizes = tuple(len(features.layer(k)) for k in (1, 2, 3))
        sizes_seen.add(sizes)
        type_splits.add(
            tuple(sum(f.type == "property" for f in features.layer(k)) for k in (1, 2, 3))
        )
        # Determined features are numbered in layer order within each type.
        for t in ("PROPERTY", "PART"):
            layers = [features[f"{t}.{i}"].layer for i in range(11, 21)]
            assert layers == sorted(layers)
            assert all(features[f"{t}.{i}"].free for i in range(1, 11))
        assert features.layers == 3
        assert features.max_chain_depth == 3
    assert sizes_seen == {(7, 7, 6)}
    assert len(type_splits) > 1  # the assignment of types to layer slots is random


def test_empty_layers_warn() -> None:
    config = config_from_mapping(
        {
            "features": {
                "property": {"count": 4, "proportion_determined": 0.25, "expected_true_free": 1},
                "part": {"count": 4, "proportion_determined": 0.0, "expected_true_free": 1},
            },
            "rules": {"max_chain_depth": 3},
        }
    )
    features = build_features(config, Streams(1))
    assert features["PROPERTY.4"].layer == 1
    assert features.layers == 1
    assert len(features.warnings) == 1
    assert "layers [2, 3] are empty" in features.warnings[0]


def test_base_rates() -> None:
    override = build_features(
        config_from_mapping({"features": {"base_rate_override": 0.4, **EQUAL_RATES}}), Streams(1)
    )
    assert set(override.base_rates.tolist()) == {0.4}
    tiny = build_features(load_config(DATA / "tiny.yaml"), Streams(1))
    assert tiny.base_rates.tolist() == [pytest.approx(2 / 6)] * 12
    hetero_config = config_from_mapping({"features": {"base_rate_heterogeneity": 10}})
    hetero = build_features(hetero_config, Streams(1))
    rates = hetero.base_rates
    assert len(set(rates.tolist())) > 1
    assert 0.1 < rates.mean() < 0.3
    assert np.all((rates > 0) & (rates < 1))
    # The default heterogeneity (2) also draws the rates; they are not all equal.
    default = build_features(config_from_mapping({}), Streams(1))
    assert len(set(default.base_rates.tolist())) > 1
    assert np.all((default.base_rates > 0) & (default.base_rates < 1))
    # Only the base_rates stream is consumed for heterogeneity; equal rates consume nothing.
    fresh = Streams(1)
    used = Streams(1)
    build_features(config_from_mapping({"features": EQUAL_RATES}), used)
    assert used.base_rates.random() == fresh.base_rates.random()
    drawn = Streams(1)
    build_features(config_from_mapping({}), drawn)
    assert drawn.base_rates.random() != Streams(1).base_rates.random()


# ---------------------------------------------------------------------------------------------
# The rule graph
# ---------------------------------------------------------------------------------------------


def _check_graph(rules: RuleSet) -> None:
    """Acyclic, ordered, no ISA inputs, PROPERTY and PART outputs only, and the layer
    constraints."""
    known = {f.label for f in rules.features.free}
    for rule in rules.rules:
        for f in rule.inputs:
            assert f.label in known, f"{rule.output.label} reads {f.label} before it is computed"
            assert f.type in ("property", "part")
            assert f.label.startswith(("PROPERTY.", "PART."))
        assert len(set(rule.inputs)) == rule.arity
        assert rule.output.type in ("property", "part")
        k = rule.output.layer
        assert all(f.layer < k for f in rule.inputs)
        assert any(f.layer == k - 1 for f in rule.inputs)
        known.add(rule.output.label)
    assert known == set(rules.features.labels)
    assert len(rules.rules) == len(rules.features.determined)
    assert [r.output.label for r in rules.rules] == [f.label for f in rules.features.determined]


@pytest.mark.parametrize("overrides", [{}, CHAINED], ids=["default", "chained"])
@pytest.mark.parametrize("seed", range(4))
def test_rule_graph_is_acyclic_and_obeys_the_layer_constraints(overrides: dict, seed: int) -> None:
    _check_graph(make(overrides, seed))


def test_chain_depth_is_real() -> None:
    rules = make(CHAINED, seed=3)
    layered = [r for r in rules.rules if r.output.layer >= 2]
    assert layered
    for rule in layered:
        assert any(not f.free for f in rule.inputs)
        assert len(rules.cone(rule.output)) >= rule.arity
    default = make({}, seed=3)
    assert all(f.free for r in default.rules for f in r.inputs)


# ---------------------------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------------------------

TINY_POOL = {
    # Three free PROPERTY features feed six determined ones, with literal rules only.
    "features": {
        "property": {"count": 9, "proportion_determined": 6 / 9, "expected_true_free": 1},
        "part": {"count": 0, "proportion_determined": 0, "expected_true_free": 0},
        **EQUAL_RATES,
    },
    "rules": {"arity": {1: 1}, "input_type_weights": {"property": 1, "part": 0}},
}


def test_no_two_rules_share_inputs_and_truth_table() -> None:
    rules = make(TINY_POOL)
    assert len(rules.features.free) == 3 and len(rules.rules) == 6
    keys = {canonical_key(r.inputs, r.table) for r in rules.rules}
    assert len(keys) == 6  # the only six functions of one of three inputs
    for seed in range(4):
        rules = make({}, seed)
        keys = [canonical_key(r.inputs, r.table) for r in rules.rules]
        assert len(set(keys)) == len(keys)


def test_duplicates_exhaust_the_pool_or_are_allowed() -> None:
    seven = dict(
        TINY_POOL,
        features={
            **TINY_POOL["features"],
            "property": {"count": 10, "proportion_determined": 0.7, "expected_true_free": 1},
        },
    )
    with pytest.raises(GenerationError, match=r"PROPERTY\.10.*duplicated"):
        make(seven)
    allowed = dict(seven, rules={**TINY_POOL["rules"], "allow_duplicate_rules": True})
    rules = make(allowed)
    keys = [canonical_key(r.inputs, r.table) for r in rules.rules]
    assert len(rules.rules) == 7
    assert len(set(keys)) < 7


def test_canonical_key_ignores_input_order() -> None:
    features = build_features(config_from_mapping({}), Streams(1))
    a, b = features["PROPERTY.1"], features["PART.2"]
    ab = parse_expression("PROPERTY.1 AND NOT PART.2").truth_table(["PROPERTY.1", "PART.2"])
    ba = parse_expression("PROPERTY.1 AND NOT PART.2").truth_table(["PART.2", "PROPERTY.1"])
    assert canonical_key((a, b), ab) == canonical_key((b, a), ba)
    assert canonical_key((a, b), ab) != canonical_key((a, b), ba)


# ---------------------------------------------------------------------------------------------
# Rule records and families
# ---------------------------------------------------------------------------------------------

RECORD_KEYS = [
    "output",
    "layer",
    "family",
    "shj_type",
    "arity",
    "nesting_depth",
    "inputs",
    "relevant_inputs",
    "expression",
    "truth_table",
    "min_dnf_literals",
]


@pytest.mark.parametrize("seed", range(3))
def test_rule_records(seed: int) -> None:
    rules = make({}, seed)
    families = set()
    for rule in rules.rules:
        record = rule.record()
        assert list(record) == RECORD_KEYS
        families.add(record["family"])
        assert record["arity"] == len(record["inputs"]) == rule.arity
        assert len(record["truth_table"]) == 2**rule.arity
        assert set(record["relevant_inputs"]) <= set(record["inputs"])
        assert record["min_dnf_literals"] == dnf_literal_count(rule.table)
        assert parse_expression(record["expression"]).truth_table(record["inputs"]) == rule.table
        assert record["inputs"] == [f.label for f in sorted(rule.inputs, key=lambda f: f.position)]
        if rule.family == "literal":
            assert rule.arity == 1 and rule.shj_type is None and rule.nesting_depth is None
        elif rule.family == "binary":
            assert rule.arity == 2 and rule.table.relevant_inputs() == (0, 1)
            assert rule.nesting_depth is None
        elif rule.family == "shj":
            assert rule.arity == 3 and rule.shj_type in ("I", "II", "III", "IV", "V", "VI")
            assert rule.expression == from_dnf(minimal_dnf(rule.table), record["inputs"])
            assert rule.nesting_depth is None
        else:
            assert rule.family == "compositional"
            assert rule.arity in (3, 4)
            assert 1 <= rule.nesting_depth <= 2
            assert rule.expression.depth() == rule.nesting_depth
            assert rule.table.relevant_inputs() == tuple(range(rule.arity))
    assert families <= {"literal", "binary", "shj", "compositional"}
    assert len(families) >= 3


def test_records_are_yaml_serializable() -> None:
    text = yaml.safe_dump(make({}, 1).records(), sort_keys=False)
    assert "output: PROPERTY.31" in text
    assert "CAN." not in text


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [{}, CHAINED], ids=["default", "chained"])
def test_compute_matches_each_rule(overrides: dict) -> None:
    rules = make(overrides, seed=2)
    features = rules.features
    rng = np.random.default_rng(0)
    free_values = rng.integers(0, 2, size=(200, len(features.free)))
    out = rules.compute(free_values)
    assert out.shape == (200, len(features))
    assert out.dtype == np.uint8
    assert np.array_equal(out[:, features.free_positions], free_values)
    for rule in rules.rules:
        expected = rule.table.evaluate(out[:, rule.input_positions])
        assert np.array_equal(out[:, rule.output.position], expected)
    # Recomputing from the free columns reproduces the matrix exactly.
    assert np.array_equal(rules.compute(out[:, features.free_positions]), out)
    with pytest.raises(ValueError):
        rules.compute(free_values[:, :-1])


def test_cone() -> None:
    rules = make(CHAINED, seed=1)
    for rule in rules.rules:
        cone = rules.cone(rule.output)
        assert all(f.free for f in cone)
        direct_free = {f for f in rule.inputs if f.free}
        assert direct_free <= set(cone)
        assert list(cone) == sorted(cone, key=lambda f: f.position)
    free = rules.features.free[0]
    assert rules.cone(free) == (free,)


# ---------------------------------------------------------------------------------------------
# Determinism and stream independence
# ---------------------------------------------------------------------------------------------


def test_rules_are_deterministic_and_use_only_the_rules_stream() -> None:
    assert make({}, 5).records() == make({}, 5).records()
    assert make({}, 5).records() != make({}, 6).records()
    config = config_from_mapping({"features": EQUAL_RATES})
    used = Streams(5)
    fresh = Streams(5)
    generate_rules(config, used)
    for name in ("base_rates", "superordinates", "tree", "instances", "analysis"):
        assert getattr(used, name).random() == getattr(fresh, name).random(), name
    assert used.rules.random() != fresh.rules.random()
    # With the default heterogeneity, the base_rates stream is drawn too, and nothing else.
    hetero = Streams(5)
    generate_rules(config_from_mapping({}), hetero)
    assert hetero.base_rates.random() != Streams(5).base_rates.random()
    untouched = Streams(5)
    for name in ("superordinates", "tree", "instances", "analysis"):
        assert getattr(hetero, name).random() == getattr(untouched, name).random(), name


# ---------------------------------------------------------------------------------------------
# The variance bound
# ---------------------------------------------------------------------------------------------

AND_OF_FOUR = {
    # Ten free PROPERTY features at rate 0.5 feed twenty determined ones, each an AND of four.
    "features": {
        "property": {"count": 30, "proportion_determined": 2 / 3, "expected_true_free": 5},
        "part": {"count": 0, "proportion_determined": 0, "expected_true_free": 0},
        "base_rate_override": 0.5,
        **EQUAL_RATES,
    },
    "rules": {
        "arity": {4: 1},
        "operator_mix": {"AND": 1},
        "nesting_depth": {1: 1},
        "negation_probability": 0,
        "input_type_weights": {"property": 1, "part": 0},
    },
}


def test_expected_true_proportion() -> None:
    rules = make(AND_OF_FOUR)
    assert len(rules.features.free) == 10 and len(rules.rules) == 20
    for rule in rules.rules:
        assert rules.expected_true_proportion(rule) == pytest.approx(1 / 16)
    literal_rules = make(TINY_POOL)
    for rule in literal_rules.rules:
        expected = 1 / 3 if rule.table.bits == (0, 1) else 2 / 3
        assert literal_rules.expected_true_proportion(rule.output.label) == pytest.approx(expected)


def test_expected_true_proportion_matches_simulation_through_layers() -> None:
    rules = make(CHAINED, seed=4)
    rng = np.random.default_rng(1)
    free_values = (
        rng.random((200000, len(rules.features.free))) < rules.features.base_rates
    ).astype(np.uint8)
    out = rules.compute(free_values)
    for rule in rules.rules:
        simulated = out[:, rule.output.position].mean()
        assert rules.expected_true_proportion(rule) == pytest.approx(simulated, abs=0.01)


def test_variance_bound_rejects_and_accepts() -> None:
    infeasible = dict(AND_OF_FOUR, rules={**AND_OF_FOUR["rules"], "variance_bound": [0.3, 0.7]})
    with pytest.raises(GenerationError, match=r"closest expected proportion.*0\.0625"):
        make(infeasible)
    feasible = dict(AND_OF_FOUR, rules={**AND_OF_FOUR["rules"], "variance_bound": [0.05, 0.1]})
    rules = make(feasible)
    assert len(rules.rules) == 20
    mixed = dict(CHAINED, rules={**CHAINED["rules"], "variance_bound": [0.2, 0.8]})
    rules = make(mixed, seed=2)
    for rule in rules.rules:
        assert 0.2 <= rules.expected_true_proportion(rule) <= 0.8


# ---------------------------------------------------------------------------------------------
# Rule files
# ---------------------------------------------------------------------------------------------


def test_example_rule_file_loads() -> None:
    rule_file = load_rule_file(DATA / "rules" / "example.yaml")
    assert [t.family for t in rule_file.templates] == [
        "literal",
        "fixed",
        "shj",
        "shj",
        "compositional",
    ]
    assert rule_file.templates[1].operator == "XOR" and rule_file.templates[1].arity == 2
    assert rule_file.templates[3].applies_to == ("part",)
    assert rule_file.templates[2].applies_to == ("property", "part")
    assert rule_file.templates[4].operators == {"AND": 1, "OR": 1}
    assert rule_file.templates[4].nesting_depth == {2: 1.0}
    assert len(rule_file.explicit) == 1
    assert rule_file.explicit[0].output == "PROPERTY.35"


@pytest.mark.parametrize("seed", range(3))
def test_rules_from_the_example_rule_file(seed: int) -> None:
    config = load_config(DATA / "rule_file.yaml", seed=seed)
    rules = generate_rules(config, Streams(config.seed))
    _check_graph(rules)
    explicit = rules.rule_for("PROPERTY.35")
    assert explicit.family == "explicit"
    assert str(explicit.expression) == "(PART.2 AND NOT PROPERTY.5) OR PROPERTY.7"
    assert [f.label for f in explicit.inputs] == ["PART.2", "PROPERTY.5", "PROPERTY.7"]
    assert explicit.nesting_depth == 2
    families = {r.family for r in rules.rules}
    assert families <= {"literal", "fixed", "shj", "compositional", "explicit"}
    for rule in rules.rules:
        if rule.family == "fixed":
            assert rule.arity == 2 and rule.table.bit_string() in ("0110", "1001")
        elif rule.family == "shj":
            assert rule.shj_type in ("IV", "VI")
            if rule.shj_type == "VI":
                assert rule.output.type == "part"
        elif rule.family == "compositional":
            assert rule.arity == 4 and 1 <= rule.nesting_depth <= 2
            assert "XOR" not in str(rule.expression)
        elif rule.family == "literal":
            assert rule.arity == 1
    keys = [canonical_key(r.inputs, r.table) for r in rules.rules]
    assert len(set(keys)) == len(keys)


def _rule_file_config(tmp_path: Path, overrides: dict, rule_file: dict) -> dict:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rule_file, sort_keys=False), encoding="utf-8")
    return {
        **overrides,
        "rules": {**overrides.get("rules", {}), "source": "file", "file": str(path)},
    }


TEMPLATES = [{"family": "literal", "weight": 1}]


def _labels(config: dict, seed: int = 1) -> dict:
    features = build_features(config_from_mapping(config), Streams(seed))
    by_layer = {k: [f.label for f in features.layer(k)] for k in range(0, 4)}
    return {"features": features, "by_layer": by_layer}


def test_explicit_rules_that_break_the_layer_constraints_are_rejected(tmp_path: Path) -> None:
    info = _labels(CHAINED)
    layer1, layer2 = info["by_layer"][1], info["by_layer"][2]
    free = info["by_layer"][0]
    cases = [
        # A layer-1 feature reading a layer-1 feature.
        ({"output": layer1[0], "expression": f"{layer1[1]} AND {free[0]}"}, "not below layer 1"),
        # A layer-2 feature reading only free features.
        ({"output": layer2[0], "expression": f"{free[0]} OR {free[1]}"}, "no feature from layer 1"),
        # A layer-1 feature reading a layer-2 feature.
        ({"output": layer1[0], "expression": f"{layer2[0]}"}, "not below layer 1"),
        # Rules never read ISA features, and never a feature the taxonomy does not have.
        (
            {"output": layer1[0], "expression": "ISA.CATEGORY.1 OR PROPERTY.1"},
            "no rule may read an ISA feature",
        ),
        ({"output": layer1[0], "expression": "PROPERTY.99"}, "unknown feature PROPERTY.99"),
        ({"output": layer1[0], "expression": "CAN.2 AND PROPERTY.1"}, "unknown feature CAN.2"),
        # Only determined features have rules.
        ({"output": free[0], "expression": "PROPERTY.2"}, "free feature"),
        ({"output": "PROPERTY.99", "expression": "PROPERTY.2"}, "unknown feature PROPERTY.99"),
    ]
    for explicit, message in cases:
        config = _rule_file_config(
            tmp_path, CHAINED, {"templates": TEMPLATES, "explicit": [explicit]}
        )
        with pytest.raises(ConfigError) as error:
            make(config)
        assert message in str(error.value), explicit
        assert error.value.field.startswith("explicit[0].")
        assert str(tmp_path / "rules.yaml") in str(error.value)


def test_valid_explicit_rules_through_layers(tmp_path: Path) -> None:
    info = _labels(CHAINED)
    by_layer = info["by_layer"]
    free, layer1, layer2, layer3 = by_layer[0], by_layer[1], by_layer[2], by_layer[3]
    explicit = [
        {"output": layer2[0], "expression": f"{layer1[0]} XOR NOT {free[3]}"},
        {
            "output": layer3[0],
            "expression": f"({free[0]} AND NOT {layer1[2]}) OR (NOT {layer2[1]} AND NOT {free[0]})",
        },
        {"output": layer1[3], "expression": f"NOT {free[0]} OR NOT {free[1]} OR {free[2]}"},
    ]
    config = _rule_file_config(tmp_path, CHAINED, {"templates": TEMPLATES, "explicit": explicit})
    rules = make(config)
    _check_graph(rules)
    assert rules.rule_for(layer2[0]).family == "explicit"
    top = rules.rule_for(layer3[0])
    assert [f.label for f in top.inputs] == [free[0], layer1[2], layer2[1]]
    assert top.shj_type == "III"  # (C AND NOT B) OR (NOT A AND NOT C) is the type III form
    assert top.nesting_depth == 2
    assert rules.rule_for(layer1[3]).min_dnf_literals == 3


def test_duplicate_explicit_rules_are_rejected(tmp_path: Path) -> None:
    same_output = {
        "templates": TEMPLATES,
        "explicit": [{"output": "PROPERTY.31", "expression": "PROPERTY.1"}] * 2,
    }
    with pytest.raises(ConfigError, match="already has an explicit rule") as error:
        rule_file_from_mapping(same_output, source="x.yaml")
    assert error.value.field == "explicit[1].output"
    same_function = {
        "templates": TEMPLATES,
        "explicit": [
            {"output": "PROPERTY.31", "expression": "PROPERTY.1 AND PART.2"},
            {"output": "PROPERTY.32", "expression": "PART.2 AND PROPERTY.1"},
        ],
    }
    config = _rule_file_config(tmp_path, {}, same_function)
    with pytest.raises(ConfigError, match="allow_duplicate_rules") as error:
        make(config)
    assert error.value.field == "explicit[1].expression"
    allowed = dict(config, rules={**config["rules"], "allow_duplicate_rules": True})
    rules = make(allowed)
    assert rules.rule_for("PROPERTY.31").table == rules.rule_for("PROPERTY.32").table


@pytest.mark.parametrize(
    ("rule_file", "field"),
    [
        (
            {"templates": [{"family": "literal", "weight": 1, "applies_to": "can"}]},
            "templates[0].applies_to",
        ),
        (
            {"templates": [{"family": "literal", "weight": 1, "applies_to": ["property", "can"]}]},
            "templates[0].applies_to",
        ),
        (
            {"templates": TEMPLATES, "explicit": [{"output": "CAN.3", "expression": "PROPERTY.1"}]},
            "explicit[0].output",
        ),
        (
            {
                "templates": TEMPLATES,
                "explicit": [{"output": "EVENTTYPE1.3", "expression": "PROPERTY.1"}],
            },
            "explicit[0].output",
        ),
    ],
)
def test_can_rules_in_a_rule_file_name_the_event_file(
    rule_file: dict, field: str, tmp_path: Path
) -> None:
    """A template for CAN features, or an explicit CAN rule, is an error that points at the
    world's event file (``event_types.event_file``) and its sampling settings."""
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rule_file, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_rule_file(path)
    assert error.value.field == field
    assert "event_types.event_file" in str(error.value)
    assert str(error.value).startswith(f"{path}: {field}: ")


@pytest.mark.parametrize(
    ("value", "new_name"), [("is", "property"), ("has", "part"), (["property", "has"], "part")]
)
def test_old_type_names_in_applies_to_name_the_new_ones(value, new_name: str, tmp_path: Path):
    """Stage a6: ``applies_to`` takes ``property`` and ``part``; the old names are errors."""
    path = tmp_path / "rules.yaml"
    rule_file = {"templates": [{"family": "literal", "weight": 1, "applies_to": value}]}
    path.write_text(yaml.safe_dump(rule_file, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_rule_file(path)
    assert error.value.field == "templates[0].applies_to"
    assert new_name in str(error.value) and "stage a6" in str(error.value)


@pytest.mark.parametrize(
    ("rule_file", "field"),
    [
        ({"templates": [{"family": "shj", "type": "IV"}]}, "templates[0].weight"),
        ({"templates": [{"family": "nand", "weight": 1}]}, "templates[0].family"),
        ({"templates": [{"family": "fixed", "weight": 1, "arity": 2}]}, "templates[0].operator"),
        (
            {"templates": [{"family": "fixed", "weight": 1, "arity": 1, "operator": "AND"}]},
            "templates[0].arity",
        ),
        ({"templates": [{"family": "shj", "weight": 1, "type": "VII"}]}, "templates[0].type"),
        ({"templates": [{"family": "compositional", "weight": 1}]}, "templates[0].arity"),
        (
            {
                "templates": [
                    {"family": "compositional", "weight": 1, "arity": 4, "nesting_depth": 0}
                ]
            },
            "templates[0].nesting_depth",
        ),
        (
            {
                "templates": [
                    {"family": "compositional", "weight": 1, "arity": 4, "operators": {"NAND": 1}}
                ]
            },
            "templates[0].operators.NAND",
        ),
        (
            {"templates": [{"family": "literal", "weight": 1, "applies_to": "isa"}]},
            "templates[0].applies_to",
        ),
        ({"templates": [{"family": "literal", "weight": 1, "colour": 1}]}, "templates[0].colour"),
        ({"templates": {"family": "literal"}}, "templates"),
        (
            {
                "templates": TEMPLATES,
                "explicit": [{"output": "PROPERTY.31", "expression": "PROPERTY.1 AND"}],
            },
            "explicit[0].expression",
        ),
        (
            {"templates": TEMPLATES, "explicit": [{"output": "PROPERTY.31"}]},
            "explicit[0].expression",
        ),
        ({"templates": TEMPLATES, "rules": []}, "rules"),
    ],
)
def test_malformed_rule_files_name_the_field(rule_file: dict, field: str, tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rule_file, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_rule_file(path)
    assert error.value.field == field
    assert str(error.value).startswith(f"{path}: {field}: ")


def test_template_arity_must_fit_the_pool(tmp_path: Path) -> None:
    config = _rule_file_config(
        tmp_path, {}, {"templates": [{"family": "compositional", "weight": 1, "arity": 61}]}
    )
    with pytest.raises(ConfigError, match="at most 60 features") as error:
        make(config)
    assert error.value.field == "templates[0].arity"


def test_every_output_type_needs_a_template_or_an_explicit_rule(tmp_path: Path) -> None:
    config = _rule_file_config(
        tmp_path,
        {},
        {"templates": [{"family": "literal", "weight": 1, "applies_to": "property"}]},
    )
    with pytest.raises(ConfigError, match="PART features") as error:
        make(config)
    assert error.value.field == "templates"


def test_missing_rule_file_names_the_file() -> None:
    config = config_from_mapping({"rules": {"source": "file", "file": "/nowhere/rules.yaml"}})
    with pytest.raises(ConfigError, match="cannot read") as error:
        generate_rules(config, Streams(1))
    assert error.value.source == "/nowhere/rules.yaml"
