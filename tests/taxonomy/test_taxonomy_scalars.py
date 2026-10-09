"""Stage 7 acceptance tests: scalar dimensions in the tree and the instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import (
    ConfigError,
    Streams,
    config_from_mapping,
    generate,
    load_config,
)
from semantic_world.taxonomy.analysis import eta_squared

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"

LARGE = {
    "taxonomy": {"superordinates": 10, "depth": 3, "branching": 10},
    "superordinates": {"similarity_bound": None},
    "instances": {"per_leaf": 5},
    "scalars": {
        "count": 4,
        "drift": {"schedule": "list", "values": [0.5, 1.5]},
        "instance_drift": 0.3,
    },
}


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_scalar_defaults_keep_scalars_off() -> None:
    config = config_from_mapping({})
    assert config.scalars.count == 0
    assert config.scalars.drift == (0.5, 0.5)
    assert config.scalars.instance_drift == 0.2
    assert config.scalars.threshold_quantiles == (0.2, 0.8)
    assert config.scalars.thermometer_bins == 0
    assert config.scalars.labels == ()
    assert config.rules.sampling.input_type_weights == {"property": 1, "part": 1, "scalar": 1}
    assert config.rules.sampling.input_types == ("property", "part")
    assert config.rules.sampling.scalar_weight == 1
    resolved = config.resolved()
    assert resolved["scalars"] == {
        "count": 0,
        "drift": {"schedule": "list", "values": [0.5, 0.5]},
        "instance_drift": 0.2,
        "threshold_quantiles": [0.2, 0.8],
        "thermometer_bins": 0,
    }
    assert config_from_mapping(resolved).resolved() == resolved


def test_scalar_settings_load_and_resolve() -> None:
    config = config_from_mapping(
        {
            "taxonomy": {"depth": 4},
            "scalars": {
                "count": 3,
                "drift": {"schedule": "linear", "start": 1.0, "end": 0.0},
                "instance_drift": 0,
                "threshold_quantiles": [0.1, 0.9],
                "thermometer_bins": 5,
            },
        }
    )
    assert config.scalars.labels == ("SCALARDIM.1", "SCALARDIM.2", "SCALARDIM.3")
    assert config.scalars.drift == pytest.approx((1.0, 0.5, 0.0))
    assert config.scalars.fixed_below(4)  # a leaf: only the instance drift, which is 0
    assert config.scalars.fixed_below(3)  # level-3 parents have drift 0
    assert not config.scalars.fixed_below(2)
    depth_one = config_from_mapping({"taxonomy": {"depth": 1}, "scalars": {"count": 1}})
    assert depth_one.scalars.drift == ()
    assert not depth_one.scalars.fixed_below(1)  # the default instance drift is 0.2


def test_scalar_input_type_weight_defaults_to_one() -> None:
    config = config_from_mapping({"rules": {"input_type_weights": {"property": 1, "part": 0}}})
    assert config.rules.sampling.input_type_weights == {"property": 1, "part": 0, "scalar": 1}
    config = config_from_mapping(
        {"rules": {"input_type_weights": {"property": 2, "part": 1, "scalar": 0}}}
    )
    assert config.rules.sampling.scalar_weight == 0


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"scalars": {"count": -1}}, "scalars.count"),
        ({"scalars": {"count": 1.5}}, "scalars.count"),
        ({"scalars": {"drift": -0.1}}, "scalars.drift"),
        ({"scalars": {"drift": {"schedule": "list", "values": [0.1]}}}, "scalars.drift.values"),
        ({"scalars": {"instance_drift": -1}}, "scalars.instance_drift"),
        ({"scalars": {"threshold_quantiles": [0.8, 0.2]}}, "scalars.threshold_quantiles"),
        ({"scalars": {"threshold_quantiles": [0.0, 0.5]}}, "scalars.threshold_quantiles"),
        ({"scalars": {"threshold_quantiles": [0.2, 1.0]}}, "scalars.threshold_quantiles"),
        ({"scalars": {"threshold_quantiles": [0.2]}}, "scalars.threshold_quantiles"),
        ({"scalars": {"thermometer_bins": -2}}, "scalars.thermometer_bins"),
        ({"scalars": {"colour": 1}}, "scalars.colour"),
        (
            {"rules": {"input_type_weights": {"property": 0, "part": 0, "scalar": 1}}},
            "rules.input_type_weights",
        ),
        (
            {"rules": {"input_type_weights": {"property": 1, "isa": 1}}},
            "rules.input_type_weights.isa",
        ),
    ],
)
def test_broken_scalar_settings_name_the_field(overrides: dict, field: str) -> None:
    with pytest.raises(ConfigError) as error:
        config_from_mapping(overrides)
    assert error.value.field == field


# ---------------------------------------------------------------------------------------------
# Values and drift
# ---------------------------------------------------------------------------------------------


def test_scalar_values_have_the_right_shapes() -> None:
    result = generate(config_from_mapping({"scalars": {"count": 2}}))
    n_cat = len(result.tree.categories)
    assert result.tree.scalar_matrix().shape == (n_cat, 2)
    assert result.instances.scalars.shape == (len(result.instances), 2)
    assert result.features.scalar_count == 2
    assert result.features.scalar_labels == ("SCALARDIM.1", "SCALARDIM.2")
    assert result.vectors.scalar_labels == ("SCALARDIM.1", "SCALARDIM.2")
    assert result.vectors.scalar_generative.shape == (n_cat, 2)
    for category in result.tree.categories:
        assert category.scalars.shape == (2,)
    off = generate(config_from_mapping({}))
    assert off.tree.scalar_matrix().shape == (len(off.tree.categories), 0)
    assert off.instances.scalars.shape == (len(off.instances), 0)


def test_drift_matches_the_configured_standard_deviations() -> None:
    result = generate(config_from_mapping(LARGE))
    tree = result.tree
    superordinate_values = np.stack([c.scalars for c in tree.superordinates])
    assert superordinate_values.std() == pytest.approx(1.0, abs=0.25)
    assert abs(superordinate_values.mean()) < 0.35
    for level, drift in ((1, 0.5), (2, 1.5)):
        differences = np.concatenate(
            [
                child.scalars - parent.scalars
                for parent in tree.at_level(level)
                for child in parent.children
            ]
        )
        assert len(differences) >= 400
        assert differences.std() == pytest.approx(drift, rel=0.1)
        assert abs(differences.mean()) < 0.15 * drift
    instance_differences = np.concatenate(
        [
            result.instances.scalars[i] - tree.categories[leaf].scalars
            for i, leaf in enumerate(result.instances.leaf_index)
        ]
    )
    assert len(instance_differences) >= 10000
    assert instance_differences.std() == pytest.approx(0.3, rel=0.05)


def test_zero_drift_fixes_the_scalars() -> None:
    config = config_from_mapping({"scalars": {"count": 2, "drift": 0, "instance_drift": 0}})
    result = generate(config)
    tree = result.tree
    for category in tree.categories:
        for descendant in category.descendants():
            assert np.array_equal(descendant.scalars, category.scalars)
    for i, leaf in enumerate(result.instances.leaf_index):
        assert np.array_equal(result.instances.scalars[i], tree.categories[leaf].scalars)
    assert not np.isnan(result.vectors.scalar_defining).any()
    assert np.array_equal(result.vectors.scalar_defining, result.vectors.scalar_generative)


def test_defining_scalars_follow_the_drift_below() -> None:
    schedule = {"schedule": "list", "values": [0.5, 0.0]}
    config = config_from_mapping({"scalars": {"count": 1, "drift": schedule, "instance_drift": 0}})
    result = generate(config)
    defining = result.vectors.scalar_defining
    for ci, category in enumerate(result.tree.categories):
        if category.level >= 2:
            assert defining[ci, 0] == category.scalars[0]
        else:
            assert np.isnan(defining[ci, 0])
    with_instance_drift = generate(
        config_from_mapping({"scalars": {"count": 1, "drift": schedule}})
    )
    assert np.isnan(with_instance_drift.vectors.scalar_defining).all()


def test_scalar_means_match_the_instances() -> None:
    result = generate(config_from_mapping({"scalars": {"count": 2}}))
    for ci, category in enumerate(result.tree.categories):
        below = result.instances.below(category, result.tree)
        assert np.allclose(
            result.vectors.scalar_mean[ci], result.instances.scalars[below].mean(axis=0)
        )


# ---------------------------------------------------------------------------------------------
# Independence from the binary features
# ---------------------------------------------------------------------------------------------


def _free_snapshot(result) -> tuple:
    """The parts of a run that scalars never touch: the tree, the roles, and the free binary
    features. Determined features may read threshold literals, so they can change (stage 8)."""
    features = result.rules.features
    return (
        [c.label for c in result.tree.categories],
        result.tree.free_matrix().tobytes(),
        np.stack([c.roles for c in result.tree.categories]).tobytes(),
        result.instances.labels,
        result.instances.free_values(features).tobytes(),
    )


def _full_snapshot(result) -> tuple:
    return (
        [{k: v for k, v in r.items() if k != "thresholds"} for r in result.rules.records()],
        result.tree.generative_matrix().tobytes(),
        result.instances.values.tobytes(),
        *_free_snapshot(result),
    )


NO_THRESHOLDS = {"input_type_weights": {"property": 1, "part": 1, "scalar": 0}}


def test_free_binary_features_are_unchanged_when_only_scalar_settings_change() -> None:
    off = generate(config_from_mapping({}))
    variants = [
        {"scalars": {"count": 3}},
        {"scalars": {"count": 1, "drift": 2.0, "instance_drift": 0}},
        {"scalars": {"count": 5, "thermometer_bins": 4}},
    ]
    for overrides in variants:
        assert _free_snapshot(generate(config_from_mapping(overrides))) == _free_snapshot(off)


def test_every_binary_feature_is_unchanged_when_rules_read_no_thresholds() -> None:
    off = generate(config_from_mapping({"rules": NO_THRESHOLDS}))
    for overrides in ({"scalars": {"count": 3}}, {"scalars": {"count": 2, "drift": 1.5}}):
        on = generate(config_from_mapping({**overrides, "rules": NO_THRESHOLDS}))
        assert _full_snapshot(on) == _full_snapshot(off)
    # The rules records without the thresholds key equal the scalars-off records.
    plain = generate(config_from_mapping({}))
    assert [
        {k: v for k, v in r.items() if k != "thresholds"} for r in off.rules.records()
    ] == plain.rules.records()


def test_scalars_use_only_the_scalar_streams() -> None:
    from semantic_world.taxonomy import generate_instances, generate_rules, generate_tree

    def run(config, streams):
        rules = generate_rules(config, streams)
        tree = generate_tree(config, rules, streams)
        generate_instances(config, rules, tree, streams)

    on = Streams(1)
    run(config_from_mapping({"scalars": {"count": 2}, "rules": NO_THRESHOLDS}), on)
    off = Streams(1)
    run(config_from_mapping({"rules": NO_THRESHOLDS}), off)
    fresh = Streams(1)
    # With scalars on, both scalar streams advance; with scalars off, neither does.
    assert on.scalars.random() != Streams(1).scalars.random()
    assert on.scalar_instances.random() != Streams(1).scalar_instances.random()
    assert off.scalars.random() == fresh.scalars.random()
    assert off.scalar_instances.random() == fresh.scalar_instances.random()
    # The other streams are in the same state whether scalars are on or off.
    for name in ("base_rates", "rules", "superordinates", "tree", "instances", "analysis"):
        assert getattr(on, name).random() == getattr(off, name).random(), name
    # With threshold literals allowed, only the rules stream draws differently (the thresholds).
    # The end state of the rules stream is not a safe witness: at seed 1 the two runs happen to
    # draw the same number of values. The rules themselves show the difference.
    thresholds = Streams(1)
    with_thresholds = generate_rules(config_from_mapping({"scalars": {"count": 2}}), thresholds)
    run(config_from_mapping({"scalars": {"count": 2}}), thresholds)
    baseline = Streams(1)
    without = generate_rules(config_from_mapping({"rules": NO_THRESHOLDS}), baseline)
    run(config_from_mapping({"rules": NO_THRESHOLDS}), baseline)
    for name in ("base_rates", "superordinates", "tree", "instances", "analysis"):
        assert getattr(thresholds, name).random() == getattr(baseline, name).random(), name
    assert with_thresholds.has_thresholds and not without.has_thresholds
    assert with_thresholds.records() != without.records()


def test_changing_the_instance_count_leaves_category_scalars_unchanged() -> None:
    a = generate(config_from_mapping({"scalars": {"count": 2}, "instances": {"per_leaf": 3}}))
    b = generate(config_from_mapping({"scalars": {"count": 2}, "instances": {"per_leaf": [6, 9]}}))
    assert np.array_equal(a.tree.scalar_matrix(), b.tree.scalar_matrix())


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scalar_folder(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    config = config_from_mapping(
        {"scalars": {"count": 2, "thermometer_bins": 3}}, source="scalar_run.yaml"
    )
    result = generate(config)
    return result.write(tmp_path_factory.mktemp("scalars") / "run"), result


def test_features_csv_lists_scalars(scalar_folder) -> None:
    folder, result = scalar_folder
    features = pl.read_csv(folder / "features.csv")
    assert features["label"].to_list()[-2:] == ["SCALARDIM.1", "SCALARDIM.2"]
    scalars = features.filter(pl.col("type") == "scalar")
    assert scalars.height == 2
    assert scalars["kind"].to_list() == ["free", "free"]
    assert scalars["layer"].to_list() == [0.0, 0.0]
    assert scalars["base_rate"].is_nan().all()


def test_scalar_columns_come_last_with_six_decimals(scalar_folder) -> None:
    folder, result = scalar_folder
    base = pl.read_csv(folder / "base.csv")
    # base.csv holds the free features, so the scalars follow the last free PART feature.
    assert base.columns[-3:] == ["PART.30", "SCALARDIM.1", "SCALARDIM.2"]
    text = (folder / "base.csv").read_text().splitlines()
    for line in text[1:4]:
        cells = line.split(",")[-2:]
        assert all(len(cell.split(".")[1]) == 6 for cell in cells), cells
    assert np.allclose(base["SCALARDIM.1"].to_numpy(), result.instances.scalars[:, 0], atol=1e-6)
    static = pl.read_csv(folder / "derived" / "static_features.csv")
    assert not any(c.startswith("SCALARDIM.") for c in static.columns)
    for name in ("categories_generative.csv", "categories_defining.csv", "categories_mean.csv"):
        frame = pl.read_csv(folder / name)
        assert frame.columns[-3:] == ["PART.40", "SCALARDIM.1", "SCALARDIM.2"], name
    generative = pl.read_csv(folder / "categories_generative.csv")
    assert np.allclose(
        generative["SCALARDIM.2"].to_numpy(), result.tree.scalar_matrix()[:, 1], atol=1e-6
    )
    defining = pl.read_csv(folder / "categories_defining.csv")
    assert defining["SCALARDIM.1"].is_nan().all()  # the default drifts are not 0
    mean = pl.read_csv(folder / "categories_mean.csv")
    assert np.allclose(mean["SCALARDIM.1"].to_numpy(), result.vectors.scalar_mean[:, 0], atol=1e-6)


def test_feature_stats_for_scalars(scalar_folder) -> None:
    folder, result = scalar_folder
    stats = pl.read_csv(folder / "feature_stats.csv")
    assert stats.columns[-5:] == ["mean", "std", "eta2_level_1", "eta2_level_2", "eta2_level_3"]
    scalar_rows = stats.filter(pl.col("type") == "scalar")
    assert scalar_rows["feature"].to_list() == ["SCALARDIM.1", "SCALARDIM.2"]
    assert scalar_rows["proportion_true"].is_nan().all()
    assert scalar_rows["entropy"].is_nan().all()
    assert scalar_rows["mi_level_1"].is_nan().all()
    assert scalar_rows["defining_new"].is_null().all()
    binary_rows = stats.filter(pl.col("type") != "scalar")
    assert binary_rows["mean"].is_nan().all()
    assert binary_rows["eta2_level_2"].is_nan().all()
    assert binary_rows["defining_new"].dtype == pl.Int64
    assert binary_rows["defining_new"].null_count() == 0
    values = result.instances.scalars
    assert scalar_rows["mean"].to_numpy() == pytest.approx(values.mean(axis=0), abs=1e-6)
    assert scalar_rows["std"].to_numpy() == pytest.approx(values.std(axis=0), abs=1e-6)
    for row in scalar_rows.iter_rows(named=True):
        etas = [row[f"eta2_level_{level}"] for level in (1, 2, 3)]
        assert all(0 <= e <= 1 for e in etas)
        assert etas == sorted(etas)  # finer partitions explain at least as much variance


def test_eta_squared() -> None:
    x = np.array([1.0, 1.0, 3.0, 3.0])
    assert eta_squared(x, np.array([0, 0, 1, 1]), 2) == pytest.approx(1.0)
    assert eta_squared(x, np.array([0, 1, 0, 1]), 2) == pytest.approx(0.0)
    assert np.isnan(eta_squared(np.ones(4), np.array([0, 0, 1, 1]), 2))


def test_thermometer_codes(scalar_folder) -> None:
    folder, result = scalar_folder
    codes = pl.read_csv(folder / "instances_scalar_codes.csv")
    assert codes.columns == [
        "label",
        "SCALARDIM.1>q1",
        "SCALARDIM.1>q2",
        "SCALARDIM.1>q3",
        "SCALARDIM.2>q1",
        "SCALARDIM.2>q2",
        "SCALARDIM.2>q3",
    ]
    assert codes["label"].to_list() == list(result.instances.labels)
    for scalar in ("SCALARDIM.1", "SCALARDIM.2"):
        q1, q2, q3 = (codes[f"{scalar}>q{b}"].to_numpy() for b in (1, 2, 3))
        assert set(np.unique(q1)) <= {0, 1}
        assert np.all(q3 <= q2) and np.all(q2 <= q1)
        for b, column in enumerate((q1, q2, q3), start=1):
            assert column.mean() == pytest.approx(1 - b / 4, abs=0.02)
    without = generate(config_from_mapping({"scalars": {"count": 2}}))
    assert "instances_scalar_codes.csv" not in without.frames()
    assert "instances_scalar_codes.csv" in result.frames()


def test_similarity_statistics_ignore_scalars() -> None:
    off = generate(config_from_mapping({"rules": NO_THRESHOLDS}))
    on = generate(config_from_mapping({"scalars": {"count": 3}, "rules": NO_THRESHOLDS}))
    assert on.similarity.equals(off.similarity)
    assert on.summary["superordinates"] == off.summary["superordinates"]
    everything = generate(
        config_from_mapping({"scalars": {"count": 3}, "analysis": {"similarity_features": "all"}})
    )
    assert everything.similarity.columns == ["label", "within", "between", "instances"]


def test_config_yaml_with_scalars_reloads(scalar_folder) -> None:
    folder, result = scalar_folder
    data = yaml.safe_load((folder / "config.yaml").read_text())
    assert data["scalars"]["count"] == 2
    assert "taxonomy:scalars" in data["provenance"]["stream_seeds"]
    assert load_config(folder / "config.yaml").resolved() == result.config.resolved()


def test_scalar_runs_are_byte_identical(tmp_path: Path) -> None:
    overrides = {"scalars": {"count": 2, "thermometer_bins": 2}}
    a = generate(config_from_mapping(overrides)).write(tmp_path / "a")
    b = generate(config_from_mapping(overrides)).write(tmp_path / "b")
    for file in a.rglob("*"):
        if file.is_file() and file.name != "config.yaml":
            relative = file.relative_to(a)
            assert file.read_bytes() == (b / relative).read_bytes(), str(relative)


def test_tiny_with_scalars_runs(tmp_path: Path) -> None:
    config = load_config(DATA / "tiny.yaml")
    with_scalars = config_from_mapping({**config.resolved(), "scalars": {"count": 1}})
    result = generate(with_scalars)
    result.write(tmp_path / "tiny_scalars")
    assert result.instances.scalars.shape == (12, 1)
