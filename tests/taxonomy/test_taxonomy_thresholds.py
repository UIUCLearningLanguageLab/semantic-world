"""Stage 8 acceptance tests: threshold literals in rules."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
import yaml

from semantic_world.taxonomy import (
    ConfigError,
    GenerationError,
    Streams,
    config_from_mapping,
    generate,
    generate_rules,
    load_config,
)
from semantic_world.taxonomy.expressions import (
    ExpressionError,
    Gt,
    Not,
    Op,
    Var,
    parse_expression,
    random_read_once,
)
from semantic_world.taxonomy.fixed import FIXED_EXACT, FIXED_LOCAL, fixed_by_rule
from semantic_world.taxonomy.rules import Threshold, canonical_key, model_quantile_threshold

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"

TINY_SCALARS = {
    "features": {
        "is": {"count": 8, "proportion_determined": 0.25, "expected_true_free": 2},
        "has": {"count": 8, "proportion_determined": 0.25, "expected_true_free": 2},
    },
    "taxonomy": {"superordinates": 2, "depth": 2, "branching": 2},
    "instances": {"per_leaf": 3},
    "inheritance": {"proportion_defining": 0.4, "proportion_characteristic": 0.3},
    "scalars": {"count": 2},
}


def rules_for(overrides: dict, seed: int = 1):
    config = config_from_mapping(overrides, seed=seed)
    return config, generate_rules(config, Streams(config.seed))


# ---------------------------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------------------------


def test_threshold_literals_parse_and_print() -> None:
    assert parse_expression("SCALARDIM.2 > 0.4127") == Gt("SCALARDIM.2", 0.4127)
    assert parse_expression("SCALARDIM.2>0.4127") == Gt("SCALARDIM.2", 0.4127)
    assert parse_expression("SCALARDIM.2 <= 0.4127") == Not(Gt("SCALARDIM.2", 0.4127))
    assert parse_expression("SCALARDIM.1 > -1.5") == Gt("SCALARDIM.1", -1.5)
    assert parse_expression("SCALARDIM.1 > 2") == Gt("SCALARDIM.1", 2.0)
    assert str(Gt("SCALARDIM.2", 0.4127)) == "SCALARDIM.2 > 0.4127"
    assert str(Not(Gt("SCALARDIM.2", 0.4127))) == "SCALARDIM.2 <= 0.4127"
    assert str(Not(Not(Gt("SCALARDIM.2", 0.4127)))) == "NOT SCALARDIM.2 <= 0.4127"
    expr = parse_expression("(PROPERTY.1 AND SCALARDIM.1 > 0.5) OR NOT SCALARDIM.2 <= -1.25")
    assert expr == Op(
        "OR",
        (
            Op("AND", (Var("PROPERTY.1"), Gt("SCALARDIM.1", 0.5))),
            Not(Not(Gt("SCALARDIM.2", -1.25))),
        ),
    )
    assert str(expr) == "(PROPERTY.1 AND SCALARDIM.1 > 0.5000) OR NOT SCALARDIM.2 <= -1.2500"
    assert expr.variables() == ("PROPERTY.1", "SCALARDIM.1>0.5000", "SCALARDIM.2>-1.2500")
    assert expr.atoms() == (Var("PROPERTY.1"), Gt("SCALARDIM.1", 0.5), Gt("SCALARDIM.2", -1.25))


def test_thresholds_are_rounded_to_four_decimals() -> None:
    rounded = Gt("SCALARDIM.1", 0.41275).threshold
    assert rounded == 0.4128 or rounded == 0.4127
    assert parse_expression("SCALARDIM.1 > 0.123456").threshold == 0.1235
    assert parse_expression(str(Gt("SCALARDIM.1", 0.123456))) == Gt("SCALARDIM.1", 0.123456)


@pytest.mark.parametrize(
    "text",
    [
        "SCALARDIM.1 < 0.5",
        "SCALARDIM.1 >= 0.5",
        "SCALARDIM.1 >",
        "SCALARDIM.1 > PROPERTY.2",
        "> 0.5",
        "0.5",
        "PROPERTY.1 AND 3",
    ],
)
def test_malformed_threshold_literals_are_rejected(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(text)


def test_printed_expressions_with_thresholds_parse_back() -> None:
    rng = np.random.default_rng(3)
    atoms = [
        Var("PROPERTY.1"),
        Var("PART.2"),
        Gt("SCALARDIM.1", 0.25),
        Gt("SCALARDIM.2", -0.75),
        Gt("SCALARDIM.1", 1.5),
    ]
    keys = [a.name if isinstance(a, Var) else a.key for a in atoms]
    for depth in (1, 2, 3):
        for _ in range(20):
            expr = random_read_once(atoms, depth, {"AND": 1, "OR": 1, "XOR": 1}, 0.4, rng)
            table = expr.truth_table(keys)
            assert parse_expression(str(expr)).truth_table(keys) == table
            assert table.relevant_inputs() == tuple(range(5))


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_arity_pool_counts_scalars() -> None:
    # Two free PROPERTY features and three determined ones, whose rules want three inputs.
    small = {
        "features": {
            "is": {"count": 5, "proportion_determined": 0.6, "expected_true_free": 1},
            "has": {"count": 0, "proportion_determined": 0, "expected_true_free": 0},
        },
        "rules": {"arity": {3: 1}},
    }
    with pytest.raises(ConfigError) as error:
        config_from_mapping(small)
    assert error.value.field == "rules.arity"
    config = config_from_mapping({**small, "scalars": {"count": 1}})
    assert config.scalars.count == 1
    with pytest.raises(ConfigError):
        config_from_mapping(
            {
                **small,
                "scalars": {"count": 1},
                "rules": {"arity": {3: 1}, "input_type_weights": {"is": 1, "has": 1, "scalar": 0}},
            }
        )


def test_scalars_alone_can_feed_rules() -> None:
    only_scalars = {
        "rules": {"input_type_weights": {"is": 0, "has": 0, "scalar": 1}, "arity": {1: 1, 2: 1}},
        "scalars": {"count": 3},
    }
    config = config_from_mapping(only_scalars)
    assert config.rules.sampling.input_types == ()
    with pytest.raises(ConfigError) as error:
        config_from_mapping({"rules": {"input_type_weights": {"is": 0, "has": 0, "scalar": 1}}})
    assert error.value.field == "rules.input_type_weights"


def test_model_distribution() -> None:
    config = config_from_mapping(
        {"taxonomy": {"depth": 3}, "scalars": {"count": 1, "drift": 0.5, "instance_drift": 0.2}}
    )
    assert config.scalars.model_std == pytest.approx(math.sqrt(1 + 0.25 + 0.25 + 0.04))
    threshold = model_quantile_threshold(config.scalars, 0.8)
    assert threshold == pytest.approx(
        round(NormalDist(0, config.scalars.model_std).inv_cdf(0.8), 4)
    )
    assert threshold == round(threshold, 4)


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------


def test_rules_read_threshold_literals() -> None:
    config, rules = rules_for({"scalars": {"count": 2}})
    with_thresholds = [r for r in rules.rules if r.thresholds]
    assert with_thresholds
    assert rules.has_thresholds
    sigma = config.scalars.model_std
    low, high = NormalDist(0, sigma).inv_cdf(0.2), NormalDist(0, sigma).inv_cdf(0.8)
    for rule in rules.rules:
        scalars_used = [t.scalar for t in rule.thresholds]
        assert len(set(scalars_used)) == len(scalars_used)  # at most one literal per scalar
        assert all(t.layer == 0 and t.free and t.type == "scalar" for t in rule.thresholds)
        for t in rule.thresholds:
            assert t.threshold == round(t.threshold, 4)
            assert low - 1e-4 <= t.threshold <= high + 1e-4
            assert 0.2 <= t.quantile <= 0.8
            assert t.label == f"SCALARDIM.{t.scalar}"
            assert t.key == f"SCALARDIM.{t.scalar}>{t.threshold:.4f}"
        # Binary inputs first in feature order, then thresholds by scalar.
        kinds = [isinstance(i, Threshold) for i in rule.inputs]
        assert kinds == sorted(kinds)
        keys = [i.key if isinstance(i, Threshold) else i.label for i in rule.inputs]
        assert parse_expression(str(rule.expression)).truth_table(keys) == rule.table
        record = rule.record(include_thresholds=True)
        assert record["inputs"] == [i.label for i in rule.inputs]
        assert record["thresholds"] == {t.label: t.threshold for t in rule.thresholds}
    assert all("thresholds" in r for r in rules.records())
    _, off = rules_for({})
    assert all("thresholds" not in r for r in off.records())


def test_threshold_weight_controls_their_use() -> None:
    _, none = rules_for(
        {"scalars": {"count": 2}, "rules": {"input_type_weights": {"is": 1, "has": 1, "scalar": 0}}}
    )
    assert not none.has_thresholds
    _, many = rules_for(
        {
            "scalars": {"count": 2},
            "rules": {"input_type_weights": {"is": 1, "has": 1, "scalar": 60}},
        }
    )
    # At most one literal per scalar caps a rule at two thresholds.
    fraction = sum(len(r.thresholds) for r in many.rules) / sum(r.arity for r in many.rules)
    assert fraction > 0.35
    assert len(many.rules) == 20
    assert sum(1 for r in many.rules if r.thresholds) >= 15  # was 30 of the 40 rules with CAN
    _, only = rules_for(
        {
            "scalars": {"count": 4},
            "rules": {
                "input_type_weights": {"is": 0, "has": 0, "scalar": 1},
                "arity": {1: 1, 2: 1, 3: 1},
            },
        }
    )
    assert all(all(isinstance(i, Threshold) for i in r.inputs) for r in only.rules)


def test_thresholds_in_layered_rules() -> None:
    chained = {
        "features": {
            "is": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
            "has": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
        },
        "rules": {"max_chain_depth": 3, "input_type_weights": {"is": 1, "has": 1, "scalar": 5}},
        "scalars": {"count": 3},
    }
    for seed in range(3):
        _, rules = rules_for(chained, seed)
        for rule in rules.rules:
            k = rule.output.layer
            assert all(i.layer < k for i in rule.inputs)
            assert any(i.layer == k - 1 for i in rule.inputs)
        layered = [r for r in rules.rules if r.output.layer >= 2 and r.thresholds]
        assert layered  # thresholds join layer-2 rules as layer-0 inputs
        for rule in layered:
            assert rules.thresholds_of(rule.output)
            binary_cone = rules.cone(rule.output)
            assert all(f.free for f in binary_cone)


def test_compute_with_scalar_values() -> None:
    config, rules = rules_for({"scalars": {"count": 2}})
    rng = np.random.default_rng(0)
    free = rng.integers(0, 2, size=(100, len(rules.features.free)))
    scalars = rng.normal(size=(100, 2))
    out = rules.compute(free, scalars)
    for rule in rules.rules:
        columns = []
        for item in rule.inputs:
            if isinstance(item, Threshold):
                columns.append((scalars[:, item.scalar - 1] > item.threshold).astype(np.uint8))
            else:
                columns.append(out[:, item.position])
        assert np.array_equal(
            out[:, rule.output.position], rule.table.evaluate(np.stack(columns, axis=1))
        )
    with pytest.raises(ValueError):
        rules.compute(free)
    with pytest.raises(ValueError):
        rules.compute(free, scalars[:, :1])


def test_generated_instances_and_categories_use_their_scalars() -> None:
    result = generate(config_from_mapping({"scalars": {"count": 2}}))
    rules = result.rules
    assert rules.has_thresholds
    free = result.instances.free_values(rules.features)
    assert np.array_equal(rules.compute(free, result.instances.scalars), result.instances.values)
    for category in result.tree.categories:
        recomputed = rules.compute(category.free_values[None, :], category.scalars[None, :])[0]
        assert np.array_equal(recomputed, category.values)


def test_canonical_key_distinguishes_thresholds() -> None:
    _, rules = rules_for({"scalars": {"count": 2}})
    feature = rules.features["PROPERTY.1"]
    a = Threshold(1, 0.5, 0.6)
    b = Threshold(1, 0.7, 0.7)
    text = "PROPERTY.1 AND SCALARDIM.1 > 0.5"
    table = parse_expression(text).truth_table(["PROPERTY.1", "SCALARDIM.1>0.5000"])
    assert canonical_key((feature, a), table) != canonical_key((feature, b), table)
    swapped = parse_expression(text).truth_table(["SCALARDIM.1>0.5000", "PROPERTY.1"])
    assert canonical_key((a, feature), swapped) == canonical_key((feature, a), table)
    keys = [canonical_key(r.inputs, r.table) for r in rules.rules]
    assert len(set(keys)) == len(keys)


def test_thresholds_depend_on_the_rules_stream_only() -> None:
    a = rules_for({"scalars": {"count": 2}})[1].records()
    b = rules_for({"scalars": {"count": 2}, "instances": {"per_leaf": 20}})[1].records()
    c = rules_for({"scalars": {"count": 2}, "taxonomy": {"superordinates": 2}})[1].records()
    assert a == b == c
    # Changing the drift changes the model distribution, so the thresholds, but not the inputs.
    d = rules_for({"scalars": {"count": 2, "drift": 2.0}})[1].records()
    assert [r["inputs"] for r in a] == [r["inputs"] for r in d]
    assert [r["truth_table"] for r in a] == [r["truth_table"] for r in d]
    assert [r["thresholds"] for r in a] != [r["thresholds"] for r in d]


# ---------------------------------------------------------------------------------------------
# Rule files
# ---------------------------------------------------------------------------------------------


def _rule_file_config(tmp_path: Path, overrides: dict, rule_file: dict) -> dict:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rule_file, sort_keys=False), encoding="utf-8")
    return {
        **overrides,
        "rules": {**overrides.get("rules", {}), "source": "file", "file": str(path)},
    }


def test_explicit_rules_with_threshold_literals(tmp_path: Path) -> None:
    rule_file = {
        "templates": [{"family": "literal", "weight": 1}],
        "explicit": [
            {"output": "PROPERTY.31", "expression": "PROPERTY.1 AND SCALARDIM.1 > 0.3"},
            {"output": "PROPERTY.32", "expression": "SCALARDIM.2 <= -0.25 OR PART.2"},
        ],
    }
    config, rules = rules_for(_rule_file_config(tmp_path, {"scalars": {"count": 2}}, rule_file))
    first = rules.rule_for("PROPERTY.31")
    assert [i.label for i in first.inputs] == ["PROPERTY.1", "SCALARDIM.1"]
    assert first.thresholds[0].threshold == 0.3
    assert first.thresholds[0].quantile == pytest.approx(
        NormalDist(0, config.scalars.model_std).cdf(0.3)
    )
    assert str(first.expression) == "PROPERTY.1 AND SCALARDIM.1 > 0.3000"
    second = rules.rule_for("PROPERTY.32")
    assert second.table == parse_expression("NOT A OR B").truth_table(["A", "B"])
    # The base rate of PROPERTY.1 is Beta-drawn under the default heterogeneity.
    rate = rules.features["PROPERTY.1"].base_rate
    assert rules.expected_true_proportion(first) == pytest.approx(
        rate * (1 - first.thresholds[0].quantile)
    )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("PROPERTY.1 AND SCALARDIM.9 > 0.3", "unknown scalar SCALARDIM.9"),
        ("PROPERTY.1 AND SCALARDIM.1", "without a threshold"),
        (
            "SCALARDIM.1 > 0.1 AND SCALARDIM.1 > 0.5",
            "more than one threshold literal on the same scalar",
        ),
    ],
)
def test_bad_threshold_literals_in_explicit_rules(
    tmp_path: Path, expression: str, message: str
) -> None:
    rule_file = {
        "templates": [{"family": "literal", "weight": 1}],
        "explicit": [{"output": "PROPERTY.31", "expression": expression}],
    }
    with pytest.raises(ConfigError, match=message) as error:
        rules_for(_rule_file_config(tmp_path, {"scalars": {"count": 2}}, rule_file))
    assert error.value.field == "explicit[0].expression"


def test_threshold_literals_are_layer_zero_in_explicit_rules(tmp_path: Path) -> None:
    chained = {
        "features": {
            "is": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
            "has": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
        },
        "rules": {"max_chain_depth": 2},
        "scalars": {"count": 1},
    }
    from semantic_world.taxonomy import build_features

    features = build_features(config_from_mapping(chained), Streams(1))
    layer2 = [f.label for f in features.layer(2)][0]
    rule_file = {
        "templates": [{"family": "literal", "weight": 1}],
        "explicit": [{"output": layer2, "expression": "PROPERTY.1 AND SCALARDIM.1 > 0.2"}],
    }
    with pytest.raises(ConfigError, match="no feature from layer 1"):
        rules_for(_rule_file_config(tmp_path, chained, rule_file))


# ---------------------------------------------------------------------------------------------
# Expected proportions
# ---------------------------------------------------------------------------------------------


def test_expected_proportion_treats_thresholds_as_independent_inputs() -> None:
    _, rules = rules_for(
        {
            "scalars": {"count": 4},
            "rules": {
                "input_type_weights": {"is": 0, "has": 0, "scalar": 1},
                "arity": {1: 1, 2: 1},
                "operator_mix": {"AND": 1},
                "negation_probability": 0,
            },
        }
    )
    for rule in rules.rules:
        expected = 1.0
        for t in rule.thresholds:
            expected *= 1 - t.quantile
        assert rules.expected_true_proportion(rule) == pytest.approx(expected)


def test_expected_proportion_matches_simulation_with_scalars() -> None:
    config, rules = rules_for(
        {
            "scalars": {"count": 2},
            "rules": {"input_type_weights": {"is": 1, "has": 1, "scalar": 10}},
        }
    )
    rng = np.random.default_rng(5)
    n = 200000
    free = (rng.random((n, len(rules.features.free))) < rules.features.base_rates).astype(np.uint8)
    scalars = rng.normal(0, config.scalars.model_std, size=(n, 2))
    out = rules.compute(free, scalars)
    for rule in rules.rules:
        if len({t.scalar for t in rules.thresholds_of(rule.output)}) == len(
            rules.thresholds_of(rule.output)
        ):
            # Literals on distinct scalars really are independent, so the estimate is exact.
            assert rules.expected_true_proportion(rule) == pytest.approx(
                out[:, rule.output.position].mean(), abs=0.01
            )


# ---------------------------------------------------------------------------------------------
# Fixed by rule
# ---------------------------------------------------------------------------------------------


def _brute_force_fixed(rules, category, scalars_config, grid_points: int = 9) -> np.ndarray:
    """Enumerate every free binary setting (defining held) and, when the scalars are not fixed,
    scalar values on a grid that crosses every threshold; check which determined features are
    constant."""
    features = rules.features
    defining = category.defining_mask()
    open_columns = np.flatnonzero(~defining)
    binary = np.array(
        list(itertools.product((0, 1), repeat=len(open_columns))), dtype=np.uint8
    ).reshape(-1, len(open_columns))
    free_values = np.tile(category.free_values, (binary.shape[0], 1))
    free_values[:, open_columns] = binary
    k = features.scalar_count
    if scalars_config.fixed_below(category.level):
        scalar_grid = category.scalars[None, :]
    else:
        thresholds = sorted({t.threshold for r in rules.rules for t in r.thresholds})
        points = (
            [thresholds[0] - 1]
            + [(a + b) / 2 for a, b in zip(thresholds[:-1], thresholds[1:], strict=True)]
            + [thresholds[-1] + 1]
        )
        scalar_grid = np.array(list(itertools.product(points, repeat=k)))
    rows_binary = np.repeat(free_values, scalar_grid.shape[0], axis=0)
    rows_scalar = np.tile(scalar_grid, (free_values.shape[0], 1))
    out = rules.compute(rows_binary, rows_scalar)
    constant = out.min(axis=0) == out.max(axis=0)
    determined = np.array([not f.free for f in features.features])
    return constant & determined


@pytest.mark.parametrize("seed", range(3))
def test_thresholds_are_fixed_everywhere_when_every_drift_is_zero(seed: int) -> None:
    config = config_from_mapping(
        {
            **TINY_SCALARS,
            "scalars": {"count": 2, "drift": 0, "instance_drift": 0},
            "rules": {"input_type_weights": {"is": 1, "has": 1, "scalar": 20}},
        },
        seed=seed,
    )
    result = generate(config)
    rules = result.rules
    threshold_only = [
        r for r in rules.rules if r.inputs and all(isinstance(i, Threshold) for i in r.inputs)
    ]
    assert threshold_only
    for ci, category in enumerate(result.tree.categories):
        fixed, test = fixed_by_rule(rules, category, scalars=config.scalars)
        for rule in threshold_only:
            assert fixed[rule.output.position]
            assert test[rule.output.position] == FIXED_EXACT
            assert (
                result.vectors.defining[ci, result.vectors.isa_count + rule.output.position]
                == category.values[rule.output.position]
            )
        assert np.array_equal(fixed, _brute_force_fixed(rules, category, config.scalars)), (
            category.label
        )


@pytest.mark.parametrize("seed", range(3))
def test_fixed_by_rule_matches_brute_force_with_open_thresholds(seed: int) -> None:
    config = config_from_mapping(TINY_SCALARS, seed=seed)
    result = generate(config)
    rules = result.rules
    assert rules.has_thresholds
    for category in result.tree.categories:
        fixed, test = fixed_by_rule(rules, category, scalars=config.scalars)
        assert np.array_equal(fixed, _brute_force_fixed(rules, category, config.scalars)), (
            category.label
        )
        assert np.all(test[fixed] == FIXED_EXACT)
        local, local_test = fixed_by_rule(rules, category, cone_limit=-1, scalars=config.scalars)
        assert np.all(local <= fixed)
        assert np.all(local_test[local] == FIXED_LOCAL)


def test_fixed_thresholds_below_a_level() -> None:
    schedule = {"schedule": "list", "values": [0.0]}
    config = config_from_mapping(
        {
            **TINY_SCALARS,
            "scalars": {"count": 2, "drift": schedule, "instance_drift": 0},
            "rules": {
                "input_type_weights": {"is": 0, "has": 0, "scalar": 1},
                "arity": {1: 1, 2: 1},
            },
        }
    )
    result = generate(config)
    for category in result.tree.categories:
        fixed, _ = fixed_by_rule(result.rules, category, scalars=config.scalars)
        determined = np.array([not f.free for f in result.rules.features.features])
        # Every rule reads scalars only, and every scalar is fixed below level 1.
        assert np.array_equal(fixed, determined), category.label
        below = result.instances.below(category, result.tree)
        for position in np.flatnonzero(fixed):
            column = result.instances.values[below, position]
            assert column.min() == column.max() == category.values[position]


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------


def test_rules_yaml_lists_thresholds(tmp_path: Path) -> None:
    result = generate(config_from_mapping({"scalars": {"count": 2}}))
    folder = result.write(tmp_path / "run")
    entries = yaml.safe_load((folder / "rules.yaml").read_text())
    assert all("thresholds" in e for e in entries)
    with_thresholds = [e for e in entries if e["thresholds"]]
    assert with_thresholds
    entry = with_thresholds[0]
    for label, value in entry["thresholds"].items():
        assert label in entry["inputs"]
        assert (
            f"{label} > {value:.4f}" in entry["expression"]
            or f"{label} <= {value:.4f}" in entry["expression"]
        )
    keys = [
        f"{i}>{entry['thresholds'][i]:.4f}" if i in entry["thresholds"] else i
        for i in entry["inputs"]
    ]
    assert (
        parse_expression(entry["expression"]).truth_table(keys).bit_string() == entry["truth_table"]
    )
    reloaded = load_config(folder / "config.yaml")
    assert reloaded.scalars.count == 2


def test_infeasible_threshold_pool_is_reported() -> None:
    with pytest.raises((ConfigError, GenerationError)):
        config_from_mapping(
            {
                "scalars": {"count": 1},
                "rules": {"input_type_weights": {"is": 0, "has": 0, "scalar": 1}, "arity": {2: 1}},
            }
        )
