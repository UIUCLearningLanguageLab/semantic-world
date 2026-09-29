"""Stage 6 acceptance tests: the output folder, the statistics, and the command line."""

from __future__ import annotations

import filecmp
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import TaxonomyResult, config_from_mapping, generate, load_config
from semantic_world.taxonomy.analysis import binary_entropy, mutual_information
from semantic_world.taxonomy.io import CSV_FILES, OUTPUT_FILES, default_output_dir, git_commit
from semantic_world.taxonomy.similarity import cross_similarity, similarity_matrix

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"

CATEGORY_LABEL = re.compile(r"^C[1-9]\d*(\.[1-9]\d*)*$")
INSTANCE_LABEL = re.compile(r"^I[1-9]\d*(\.[1-9]\d*)+$")
FEATURE_LABEL = re.compile(r"^(IS|HAS|CAN)\.[1-9]\d*$|^ISA\.C[1-9]\d*(\.[1-9]\d*)*$")

EXPECTED_COLUMNS = {
    "features.csv": ["label", "type", "kind", "layer", "base_rate"],
    "tree.csv": ["label", "parent", "level", "children", "instances"],
    "roles.csv": ["category", "feature", "role", "fixed_test"],
    "similarity.csv": ["label", "within", "between", "instances"],
}


@pytest.fixture(scope="module")
def default_result() -> TaxonomyResult:
    return generate(load_config(DATA / "default.yaml"))


@pytest.fixture(scope="module")
def default_folder(
    default_result: TaxonomyResult, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return default_result.write(tmp_path_factory.mktemp("run") / "default")


def _load(folder: Path) -> dict[str, pl.DataFrame]:
    return {name: pl.read_csv(folder / name) for name in CSV_FILES}


# ---------------------------------------------------------------------------------------------
# Files and columns
# ---------------------------------------------------------------------------------------------


def test_default_configuration_runs_in_under_a_minute() -> None:
    start = time.perf_counter()
    result = generate(load_config(DATA / "default.yaml"))
    elapsed = time.perf_counter() - start
    assert elapsed < 60, f"{elapsed:.1f} s"
    assert result.summary["instances"] > 0


def test_output_folder_has_every_file(default_folder: Path) -> None:
    assert sorted(p.name for p in default_folder.iterdir()) == sorted(OUTPUT_FILES)
    assert len(OUTPUT_FILES) == 12


def test_every_csv_loads_with_the_expected_columns(
    default_result: TaxonomyResult, default_folder: Path
) -> None:
    frames = _load(default_folder)
    labels = list(default_result.vectors.feature_labels)
    depth = default_result.config.taxonomy.depth
    for name, columns in EXPECTED_COLUMNS.items():
        assert frames[name].columns == columns, name
    for name in ("categories_generative.csv", "categories_defining.csv", "categories_mean.csv"):
        assert frames[name].columns == ["label", *labels], name
        assert frames[name].height == len(default_result.tree.categories)
    assert frames["instances.csv"].columns == ["label", "leaf", *labels]
    assert frames["instances.csv"].height == len(default_result.instances)
    stats = frames["feature_stats.csv"]
    assert stats.columns == [
        "feature",
        "type",
        "kind",
        "layer",
        "proportion_true",
        "entropy",
        "defining_inherited",
        "defining_new",
        "characteristic",
        "undiagnostic",
        "fixed_by_rule",
        *[f"mi_level_{level}" for level in range(1, depth + 1)],
    ]
    assert stats.height == len(labels)
    assert frames["features.csv"].height == len(labels)
    assert frames["roles.csv"].height == len(default_result.tree.categories) * len(
        default_result.features
    )


def test_labels_follow_the_label_table(default_folder: Path) -> None:
    frames = _load(default_folder)
    for label in frames["tree.csv"]["label"]:
        assert CATEGORY_LABEL.match(label), label
    for label, leaf in zip(
        frames["instances.csv"]["label"], frames["instances.csv"]["leaf"], strict=True
    ):
        assert INSTANCE_LABEL.match(label), label
        assert label.startswith("I" + leaf[1:] + "."), (label, leaf)
    for label in frames["features.csv"]["label"]:
        assert FEATURE_LABEL.match(label), label
    labels = frames["features.csv"]["label"].to_list()
    isa = [x for x in labels if x.startswith("ISA.")]
    assert isa == ["ISA." + c for c in frames["tree.csv"]["label"]]
    rest = [x for x in labels if not x.startswith("ISA.")]
    assert rest == [f"IS.{i}" for i in range(1, 41)] + [f"HAS.{i}" for i in range(1, 41)] + [
        f"CAN.{i}" for i in range(1, 21)
    ]
    assert frames["instances.csv"]["label"][0] == "I1.1.1.1"


def test_values_are_written_as_specified(default_folder: Path) -> None:
    text = (default_folder / "categories_defining.csv").read_text().splitlines()
    cells = set(text[1].split(",")[1:])
    assert cells <= {"0", "1", "NaN"}
    assert "NaN" in cells
    means = (default_folder / "categories_mean.csv").read_text().splitlines()[1].split(",")[1:]
    assert all(re.match(r"^(NaN|\d\.\d{6})$", cell) for cell in means)
    generative = (
        (default_folder / "categories_generative.csv").read_text().splitlines()[1].split(",")[1:]
    )
    assert set(generative) <= {"0", "1"}
    instances = (default_folder / "instances.csv").read_text().splitlines()[1].split(",")[2:]
    assert set(instances) <= {"0", "1"}
    tree_text = (default_folder / "tree.csv").read_text().splitlines()
    assert tree_text[1].startswith("C1,,1,")  # a superordinate has no parent


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def _folders_identical(a: Path, b: Path) -> bool:
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    match, mismatch, errors = filecmp.cmpfiles(a, b, names, shallow=False)
    return not mismatch and not errors


def test_same_configuration_and_seed_give_byte_identical_folders(tmp_path: Path) -> None:
    config = load_config(DATA / "tiny.yaml")
    a = generate(config).write(tmp_path / "a")
    b = generate(load_config(DATA / "tiny.yaml")).write(tmp_path / "b")
    assert _folders_identical(a, b)


def test_different_seeds_give_different_outputs(tmp_path: Path) -> None:
    a = generate(load_config(DATA / "tiny.yaml", seed=1)).write(tmp_path / "a")
    b = generate(load_config(DATA / "tiny.yaml", seed=2)).write(tmp_path / "b")
    assert not _folders_identical(a, b)
    assert (a / "instances.csv").read_bytes() != (b / "instances.csv").read_bytes()
    assert (a / "rules.yaml").read_bytes() != (b / "rules.yaml").read_bytes()


def test_changing_only_the_instance_count_leaves_rules_and_tree_unchanged(tmp_path: Path) -> None:
    a = generate(config_from_mapping({"instances": {"per_leaf": 3}})).write(tmp_path / "a")
    b = generate(config_from_mapping({"instances": {"per_leaf": [6, 9]}})).write(tmp_path / "b")
    for name in ("rules.yaml", "categories_generative.csv", "features.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    # tree.csv also lists the number of instances below each category, which changes by
    # design; every other column is unchanged.
    tree_a = pl.read_csv(a / "tree.csv").drop("instances")
    tree_b = pl.read_csv(b / "tree.csv").drop("instances")
    assert tree_a.equals(tree_b)
    assert (a / "instances.csv").read_bytes() != (b / "instances.csv").read_bytes()


def test_tree_csv_instance_counts_differ_with_instance_count(tmp_path: Path) -> None:
    a = generate(config_from_mapping({"instances": {"per_leaf": 3}}))
    b = generate(config_from_mapping({"instances": {"per_leaf": 4}}))
    ta, tb = a.frames()["tree.csv"], b.frames()["tree.csv"]
    assert ta.drop("instances").equals(tb.drop("instances"))
    assert ta["instances"].to_list() != tb["instances"].to_list()


# ---------------------------------------------------------------------------------------------
# config.yaml, rules.yaml, summary.yaml
# ---------------------------------------------------------------------------------------------


def test_config_yaml_is_resolved_and_loads_back(
    default_result: TaxonomyResult, default_folder: Path
) -> None:
    data = yaml.safe_load((default_folder / "config.yaml").read_text())
    assert data["seed"] == 1
    assert data["inheritance"]["proportion_defining"] == {
        "schedule": "list",
        "values": [0.1, 0.1, 0.1],
    }
    assert data["taxonomy"]["branching"] == {"schedule": "list", "values": [[2, 4], [2, 4]]}
    provenance = data["provenance"]
    assert set(provenance) == {"git_commit", "git_dirty", "package_version", "stream_seeds"}
    assert list(provenance["stream_seeds"]) == [
        f"taxonomy:{n}"
        for n in ("base_rates", "rules", "superordinates", "tree", "instances", "analysis")
    ]
    commit, dirty = git_commit()
    assert provenance["git_commit"] == commit
    assert provenance["git_dirty"] == dirty
    assert commit is None or re.match(r"^[0-9a-f]{40}$", commit)
    reloaded = load_config(default_folder / "config.yaml")
    assert reloaded.resolved() == default_result.config.resolved()


def test_provenance_must_be_a_mapping() -> None:
    from semantic_world.taxonomy import ConfigError

    with pytest.raises(ConfigError) as error:
        config_from_mapping({"provenance": 5})
    assert error.value.field == "provenance"


def test_rules_yaml(default_result: TaxonomyResult, default_folder: Path) -> None:
    rules = yaml.safe_load((default_folder / "rules.yaml").read_text())
    assert len(rules) == len(default_result.rules.rules) == 40
    assert [r["output"] for r in rules] == [f.label for f in default_result.features.determined]
    example = rules[0]
    assert list(example) == [
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
    assert re.match(r"^[01]+$", example["truth_table"])


def test_summary_yaml(default_result: TaxonomyResult, default_folder: Path) -> None:
    summary = yaml.safe_load((default_folder / "summary.yaml").read_text())
    tree = default_result.tree
    assert summary["categories"] == len(tree.categories)
    assert summary["leaves"] == len(tree.leaves)
    assert summary["instances"] == len(default_result.instances)
    values = default_result.instances.values
    features = default_result.features
    for t in ("is", "has", "can"):
        positions = [f.position for f in features.of_type(t)]
        assert summary["mean_true_features_per_instance"][t] == pytest.approx(
            values[:, positions].sum(axis=1).mean()
        )
    assert summary["constant_features"] == int((values.min(axis=0) == values.max(axis=0)).sum())
    assert summary["duplicate_leaves"] == 0
    assert summary["duplicate_instances"] >= 0
    assert summary["superordinates"]["tries"] >= 4
    assert (
        -1
        <= summary["superordinates"]["similarity_min"]
        <= summary["superordinates"]["similarity_max"]
        <= 0.3
    )
    assert summary["warnings"] == []


# ---------------------------------------------------------------------------------------------
# Similarity statistics
# ---------------------------------------------------------------------------------------------


def test_similarity_statistics_match_direct_computation(default_result: TaxonomyResult) -> None:
    result = default_result
    table = result.similarity
    tree = result.tree
    isa = result.vectors.isa_count
    instances_full = result.instances.full_matrix(tree)[:, isa:]
    rows = {row["label"]: row for row in table.iter_rows(named=True)}
    # A leaf: within = mean pairwise cosine among its instances.
    leaf = tree.leaves[0]
    below = result.instances.below(leaf, tree)
    sims = similarity_matrix(instances_full[below], "cosine")[np.triu_indices(len(below), k=1)]
    assert rows[leaf.label]["within"] == pytest.approx(np.nanmean(sims))
    assert rows[leaf.label]["instances"] == pytest.approx(np.nanmean(sims))
    # A superordinate: between = mean cosine between its children and the children of the
    # other superordinates.
    top = tree.superordinates[0]
    own = np.stack([c.values for c in top.children])
    others = np.stack([c.values for s in tree.superordinates[1:] for c in s.children])
    assert rows[top.label]["between"] == pytest.approx(
        np.nanmean(cross_similarity(own, others, "cosine"))
    )
    within_top = similarity_matrix(own, "cosine")[np.triu_indices(len(own), k=1)]
    assert rows[top.label]["within"] == pytest.approx(np.nanmean(within_top))
    # A middle category: between uses the parent's other children.
    middle = tree.superordinates[0].children[0]
    own = np.stack([c.values for c in middle.children])
    siblings = [c for c in middle.parent.children if c is not middle]
    others = np.stack([c.values for s in siblings for c in s.children])
    assert rows[middle.label]["between"] == pytest.approx(
        np.nanmean(cross_similarity(own, others, "cosine"))
    )
    within = table["within"].fill_nan(0).to_numpy()
    assert np.all((within >= 0) & (within <= 1))


def test_similarity_with_all_features_and_phi() -> None:
    config = config_from_mapping(
        {"analysis": {"similarity_metric": "phi", "similarity_features": "all"}}
    )
    result = generate(config)
    table = result.similarity
    assert table.height == len(result.tree.categories)
    values = table["within"].drop_nans().to_numpy()
    assert np.all((values >= -1) & (values <= 1))


def test_pair_sampling_is_deterministic_and_close() -> None:
    full = generate(config_from_mapping({"instances": {"per_leaf": 20}}))
    sampled_a = generate(
        config_from_mapping({"instances": {"per_leaf": 20}, "analysis": {"max_pairs": 500}})
    )
    sampled_b = generate(
        config_from_mapping({"instances": {"per_leaf": 20}, "analysis": {"max_pairs": 500}})
    )
    assert sampled_a.similarity.equals(sampled_b.similarity)
    top = full.tree.superordinates[0].label
    exact = full.similarity.filter(pl.col("label") == top)["instances"][0]
    estimate = sampled_a.similarity.filter(pl.col("label") == top)["instances"][0]
    assert estimate == pytest.approx(exact, abs=0.03)
    assert not sampled_a.similarity.equals(full.similarity)


def test_single_superordinate_has_no_between_pairs() -> None:
    result = generate(
        config_from_mapping(
            {
                "taxonomy": {"superordinates": 1, "depth": 2},
                "superordinates": {"similarity_bound": None},
            }
        )
    )
    assert math.isnan(result.similarity["between"][0])
    assert result.summary["superordinates"]["similarity_min"] is None


# ---------------------------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------------------------


def test_entropy_and_mutual_information_helpers() -> None:
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0
    assert binary_entropy(0.25) == pytest.approx(0.8112781)
    x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    c = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    assert mutual_information(x, c, 2) == pytest.approx(1.0)
    assert mutual_information(x, np.array([0, 1, 0, 1, 0, 1, 0, 1]), 2) == pytest.approx(0.0)
    assert mutual_information(x, np.zeros(8, dtype=int), 1) == pytest.approx(0.0)


def test_feature_statistics(default_result: TaxonomyResult, default_folder: Path) -> None:
    stats = pl.read_csv(default_folder / "feature_stats.csv")
    instances = pl.read_csv(default_folder / "instances.csv")
    tree = default_result.tree
    n_cat = len(tree.categories)
    for row in stats.iter_rows(named=True):
        column = instances[row["feature"]].to_numpy()
        assert row["proportion_true"] == pytest.approx(column.mean(), abs=1e-6)
        assert row["entropy"] == pytest.approx(binary_entropy(column.mean()), abs=1e-6)
        for level in (1, 2, 3):
            assert 0 <= row[f"mi_level_{level}"] <= row["entropy"] + 1e-6
        if row["kind"] == "free":
            assert (
                row["defining_inherited"]
                + row["defining_new"]
                + row["characteristic"]
                + row["undiagnostic"]
                == n_cat
            )
            assert row["fixed_by_rule"] == 0
        else:
            assert (
                row["defining_inherited"]
                + row["defining_new"]
                + row["characteristic"]
                + row["undiagnostic"]
                == 0
            )
    # An ISA feature of a superordinate carries information about the level-1 category.
    first = stats.filter(pl.col("feature") == "ISA.C1").row(0, named=True)
    assert first["type"] == "isa" and first["kind"] == "determined"
    assert first["mi_level_1"] > 0.5
    assert first["mi_level_1"] == pytest.approx(
        first["entropy"], abs=1e-6
    )  # determined by the level-1 category
    assert stats.filter(pl.col("feature") == "IS.1").row(0, named=True)["kind"] == "free"
    assert stats.filter(pl.col("feature") == "CAN.1").row(0, named=True)["layer"] == 2.0


def test_roles_csv_is_consistent(default_result: TaxonomyResult, default_folder: Path) -> None:
    roles = pl.read_csv(default_folder / "roles.csv")
    assert set(roles["role"].unique()) <= {
        "defining_inherited",
        "defining_new",
        "characteristic",
        "undiagnostic",
        "determined",
        "fixed_by_rule",
    }
    fixed = roles.filter(pl.col("role") == "fixed_by_rule")
    assert set(fixed["fixed_test"].unique()) <= {"exact", "local"}
    unfixed = roles.filter(pl.col("role") != "fixed_by_rule")["fixed_test"]
    assert all(value in (None, "") for value in unfixed.to_list())
    assert fixed.height == int(default_result.vectors.fixed_by_rule.sum())
    superordinate = roles.filter(pl.col("category") == "C1")
    assert (superordinate["role"] == "defining_inherited").sum() == 0


def test_features_csv(default_folder: Path) -> None:
    features = pl.read_csv(default_folder / "features.csv")
    free = features.filter(pl.col("kind") == "free")
    assert free.height == 60
    assert np.allclose(free["base_rate"].to_numpy(), 0.2)
    assert set(free["layer"].to_list()) == {0.0}
    determined = features.filter((pl.col("kind") == "determined") & (pl.col("type") != "isa"))
    assert determined["base_rate"].is_nan().all()
    assert set(determined["layer"].to_list()) == {1.0, 2.0}


# ---------------------------------------------------------------------------------------------
# Other example configurations and the command line
# ---------------------------------------------------------------------------------------------


def test_tiny_and_rule_file_examples_run(tmp_path: Path) -> None:
    tiny = generate(load_config(DATA / "tiny.yaml"))
    folder = tiny.write(tmp_path / "tiny")
    assert pl.read_csv(folder / "instances.csv").height == 12
    rule_file = generate(load_config(DATA / "rule_file.yaml"))
    assert any(r.family == "explicit" for r in rule_file.rules.rules)
    rule_file.write(tmp_path / "rule_file")


def test_default_output_dir() -> None:
    config = load_config(DATA / "tiny.yaml", seed=9)
    assert default_output_dir(config) == Path("runs/taxonomy/tiny_seed9")


def test_command_line(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_world.taxonomy",
            str(DATA / "tiny.yaml"),
            "--seed",
            "3",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith(f"wrote {out}: 6 categories, 4 leaves, 12 instances, 8 rules")
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUT_FILES)
    assert yaml.safe_load((out / "config.yaml").read_text())["seed"] == 3


def test_command_line_reports_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("taxonomy: {depth: 0}\n", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-m", "semantic_world.taxonomy", str(bad), "--out", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert run.returncode == 1
    assert "taxonomy.depth" in run.stderr
    assert "Traceback" not in run.stderr
