"""Stage 3 acceptance tests: feature layout, rule sampling, duplicate checks, the variance
bound, and rule files."""

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

CHAINED = {
    "features": {
        "is": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
        "has": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
        "can": {"count": 10},
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
    config = config_from_mapping({})
    features = build_features(config, Streams(1))
    assert len(features) == 100
    assert features.labels[:3] == ("IS.1", "IS.2", "IS.3")
    assert features.labels[40:42] == ("HAS.1", "HAS.2")
    assert features.labels[-1] == "CAN.20"
    assert [f.position for f in features.features] == list(range(100))
    # Free features first, then determined, within each type.
    assert all(features[f"IS.{i}"].free for i in range(1, 31))
    assert all(not features[f"IS.{i}"].free for i in range(31, 41))
    assert all(features[f"IS.{i}"].layer == 1 for i in range(31, 41))
    assert all(features[f"HAS.{i}"].layer == (0 if i <= 30 else 1) for i in range(1, 41))
    assert all(features[f"CAN.{i}"].layer == 2 for i in range(1, 21))
    assert features.is_has_layers == 1
    assert features.can_layer == 2
    assert len(features.free) == 60
    assert features.base_rates.tolist() == [pytest.approx(0.2)] * 60
    assert all(f.base_rate is None for f in features.determined)
    order = [f.label for f in features.determined]
    assert order[:10] == [f"IS.{i}" for i in range(31, 41)]
    assert order[10:20] == [f"HAS.{i}" for i in range(31, 41)]
    assert order[20:] == [f"CAN.{i}" for i in range(1, 21)]
    assert features.warnings == ()
    with pytest.raises(KeyError):
        features["ISA.C1"]


def test_layers_are_split_evenly_lowest_first() -> None:
    config = config_from_mapping(CHAINED)
    sizes_seen = set()
    type_splits = set()
    for seed in range(5):
        features = build_features(config, Streams(seed))
        sizes = tuple(len(features.layer(k)) for k in (1, 2, 3))
        sizes_seen.add(sizes)
        type_splits.add(tuple(sum(f.type == "is" for f in features.layer(k)) for k in (1, 2, 3)))
        # Determined features are numbered in layer order within each type.
        for t in ("IS", "HAS"):
            layers = [features[f"{t}.{i}"].layer for i in range(11, 21)]
            assert layers == sorted(layers)
            assert all(features[f"{t}.{i}"].free for i in range(1, 11))
        assert features.can_layer == 4
        assert features.max_chain_depth == 3
    assert sizes_seen == {(7, 7, 6)}
    assert len(type_splits) > 1  # the assignment of types to layer slots is random


def test_empty_layers_warn() -> None:
    config = config_from_mapping(
        {
            "features": {
                "is": {"count": 4, "proportion_determined": 0.25, "expected_true_free": 1},
                "has": {"count": 4, "proportion_determined": 0.0, "expected_true_free": 1},
            },
            "rules": {"max_chain_depth": 3},
        }
    )
    features = build_features(config, Streams(1))
    assert features["IS.4"].layer == 1
    assert features.is_has_layers == 1
    assert features.can_layer == 2
    assert len(features.warnings) == 1
    assert "layers [2, 3] are empty" in features.warnings[0]


def test_base_rates() -> None:
    override = build_features(
        config_from_mapping({"features": {"base_rate_override": 0.4}}), Streams(1)
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
    # Only the base_rates stream is consumed for heterogeneity; equal rates consume nothing.
    fresh = Streams(1)
    used = Streams(1)
    build_features(config_from_mapping({}), used)
    assert used.base_rates.random() == fresh.base_rates.random()


# ---------------------------------------------------------------------------------------------
# The rule graph
# ---------------------------------------------------------------------------------------------


def _check_graph(rules: RuleSet) -> None:
    """Acyclic, ordered, no ISA or CAN inputs, and the layer constraints."""
    known = {f.label for f in rules.features.free}
    for rule in rules.rules:
        for f in rule.inputs:
            assert f.label in known, f"{rule.output.label} reads {f.label} before it is computed"
            assert f.type in ("is", "has")
            assert not f.label.startswith(("ISA.", "CAN."))
        assert len(set(rule.inputs)) == rule.arity
        if rule.output.type == "can":
            assert rule.output.layer == rules.features.can_layer
        else:
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
    layered = [r for r in rules.rules if r.output.type != "can" and r.output.layer >= 2]
    assert layered
    for rule in layered:
        assert any(not f.free for f in rule.inputs)
        assert len(rules.cone(rule.output)) >= rule.arity
    default = make({}, seed=3)
    assert all(f.free for r in default.rules if r.output.type != "can" for f in r.inputs)


# ---------------------------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------------------------

TINY_POOL = {
    "features": {
        "is": {"count": 3, "proportion_determined": 0, "expected_true_free": 1},
        "has": {"count": 0, "proportion_determined": 0, "expected_true_free": 0},
        "can": {"count": 6},
    },
    "rules": {"arity": {1: 1}, "input_type_weights": {"is": 1, "has": 0}},
}


def test_no_two_rules_share_inputs_and_truth_table() -> None:
    rules = make(TINY_POOL)
    keys = {canonical_key(r.inputs, r.table) for r in rules.rules}
    assert len(keys) == 6  # the only six functions of one of three inputs
    for seed in range(4):
        rules = make({}, seed)
        keys = [canonical_key(r.inputs, r.table) for r in rules.rules]
        assert len(set(keys)) == len(keys)


def test_duplicates_exhaust_the_pool_or_are_allowed() -> None:
    seven = dict(TINY_POOL, features={**TINY_POOL["features"], "can": {"count": 7}})
    with pytest.raises(GenerationError, match="CAN.7.*duplicated"):
        make(seven)
    allowed = dict(seven, rules={**TINY_POOL["rules"], "allow_duplicate_rules": True})
    rules = make(allowed)
    keys = [canonical_key(r.inputs, r.table) for r in rules.rules]
    assert len(rules.rules) == 7
    assert len(set(keys)) < 7


def test_canonical_key_ignores_input_order() -> None:
    features = build_features(config_from_mapping({}), Streams(1))
    a, b = features["IS.1"], features["HAS.2"]
    ab = parse_expression("IS.1 AND NOT HAS.2").truth_table(["IS.1", "HAS.2"])
    ba = parse_expression("IS.1 AND NOT HAS.2").truth_table(["HAS.2", "IS.1"])
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
    assert "output: IS.31" in text


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
    config = config_from_mapping({})
    used = Streams(5)
    fresh = Streams(5)
    generate_rules(config, used)
    for name in ("base_rates", "superordinates", "tree", "instances", "analysis"):
        assert getattr(used, name).random() == getattr(fresh, name).random(), name
    assert used.rules.random() != fresh.rules.random()


# ---------------------------------------------------------------------------------------------
# The variance bound
# ---------------------------------------------------------------------------------------------

AND_OF_FOUR = {
    "features": {
        "is": {"count": 10, "proportion_determined": 0, "expected_true_free": 5},
        "has": {"count": 0, "proportion_determined": 0, "expected_true_free": 0},
        "can": {"count": 20},
        "base_rate_override": 0.5,
    },
    "rules": {
        "arity": {4: 1},
        "operator_mix": {"AND": 1},
        "nesting_depth": {1: 1},
        "negation_probability": 0,
        "input_type_weights": {"is": 1, "has": 0},
    },
}


def test_expected_true_proportion() -> None:
    rules = make(AND_OF_FOUR)
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
    assert rule_file.templates[3].applies_to == ("can",)
    assert rule_file.templates[2].applies_to == ("is", "has", "can")
    assert rule_file.templates[4].operators == {"AND": 1, "OR": 1}
    assert rule_file.templates[4].nesting_depth == {2: 1.0}
    assert len(rule_file.explicit) == 1
    assert rule_file.explicit[0].output == "CAN.3"


@pytest.mark.parametrize("seed", range(3))
def test_rules_from_the_example_rule_file(seed: int) -> None:
    config = load_config(DATA / "rule_file.yaml", seed=seed)
    rules = generate_rules(config, Streams(config.seed))
    _check_graph(rules)
    explicit = rules.rule_for("CAN.3")
    assert explicit.family == "explicit"
    assert str(explicit.expression) == "(HAS.2 AND NOT IS.5) OR IS.7"
    assert [f.label for f in explicit.inputs] == ["HAS.2", "IS.5", "IS.7"]
    assert explicit.nesting_depth == 2
    families = {r.family for r in rules.rules}
    assert families <= {"literal", "fixed", "shj", "compositional", "explicit"}
    for rule in rules.rules:
        if rule.family == "fixed":
            assert rule.arity == 2 and rule.table.bit_string() in ("0110", "1001")
        elif rule.family == "shj":
            assert rule.shj_type in ("IV", "VI")
            if rule.shj_type == "VI":
                assert rule.output.type == "can"
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
        # Rules never read CAN or ISA features.
        ({"output": "CAN.1", "expression": "CAN.2 AND IS.1"}, "no rule may read a CAN feature"),
        ({"output": layer1[0], "expression": "ISA.C1 OR IS.1"}, "no rule may read an ISA feature"),
        ({"output": "CAN.1", "expression": "IS.99"}, "unknown feature IS.99"),
        # Only determined features have rules.
        ({"output": free[0], "expression": "IS.2"}, "free feature"),
        ({"output": "IS.99", "expression": "IS.2"}, "unknown feature IS.99"),
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
    layer1, layer2, free = info["by_layer"][1], info["by_layer"][2], info["by_layer"][0]
    explicit = [
        {"output": layer2[0], "expression": f"{layer1[0]} XOR NOT {free[3]}"},
        {
            "output": "CAN.1",
            "expression": f"({free[0]} AND NOT {layer1[2]}) OR (NOT {layer2[1]} AND NOT {free[0]})",
        },
        {"output": "CAN.2", "expression": f"NOT {free[0]} OR NOT {free[1]} OR {free[2]}"},
    ]
    config = _rule_file_config(tmp_path, CHAINED, {"templates": TEMPLATES, "explicit": explicit})
    rules = make(config)
    _check_graph(rules)
    assert rules.rule_for(layer2[0]).family == "explicit"
    can_1 = rules.rule_for("CAN.1")
    assert [f.label for f in can_1.inputs] == [free[0], layer1[2], layer2[1]]
    assert can_1.shj_type == "III"  # (C AND NOT B) OR (NOT A AND NOT C) is the type III form
    assert can_1.nesting_depth == 2
    assert rules.rule_for("CAN.2").min_dnf_literals == 3


def test_duplicate_explicit_rules_are_rejected(tmp_path: Path) -> None:
    same_output = {
        "templates": TEMPLATES,
        "explicit": [{"output": "CAN.1", "expression": "IS.1"}] * 2,
    }
    with pytest.raises(ConfigError, match="already has an explicit rule") as error:
        rule_file_from_mapping(same_output, source="x.yaml")
    assert error.value.field == "explicit[1].output"
    same_function = {
        "templates": TEMPLATES,
        "explicit": [
            {"output": "CAN.1", "expression": "IS.1 AND HAS.2"},
            {"output": "CAN.2", "expression": "HAS.2 AND IS.1"},
        ],
    }
    config = _rule_file_config(tmp_path, {}, same_function)
    with pytest.raises(ConfigError, match="allow_duplicate_rules") as error:
        make(config)
    assert error.value.field == "explicit[1].expression"
    allowed = dict(config, rules={**config["rules"], "allow_duplicate_rules": True})
    rules = make(allowed)
    assert rules.rule_for("CAN.1").table == rules.rule_for("CAN.2").table


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
            {"templates": TEMPLATES, "explicit": [{"output": "CAN.1", "expression": "IS.1 AND"}]},
            "explicit[0].expression",
        ),
        ({"templates": TEMPLATES, "explicit": [{"output": "CAN.1"}]}, "explicit[0].expression"),
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
        {"templates": [{"family": "literal", "weight": 1, "applies_to": ["is", "has"]}]},
    )
    with pytest.raises(ConfigError, match="CAN features") as error:
        make(config)
    assert error.value.field == "templates"


def test_missing_rule_file_names_the_file() -> None:
    config = config_from_mapping({"rules": {"source": "file", "file": "/nowhere/rules.yaml"}})
    with pytest.raises(ConfigError, match="cannot read") as error:
        generate_rules(config, Streams(1))
    assert error.value.source == "/nowhere/rules.yaml"
