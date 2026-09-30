"""Stage 12 acceptance tests: category proportions, sampled pairs, verb statistics, thematic
relatedness, the new files, determinism, and the command line."""

from __future__ import annotations

import filecmp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import TaxonomyResult, config_from_mapping, generate, load_config
from semantic_world.taxonomy.io import OUTPUT_FILES, RELATION_FILES
from semantic_world.taxonomy.relation_stats import level_assignment
from semantic_world.taxonomy.similarity import pair_similarity

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"

EXPECTED_COLUMNS = {
    "verb_features.csv": ["label", "base_rate", "constraint"],
    "verb_tree.csv": ["label", "parent", "level", "children"],
    "verb_roles.csv": ["category", "feature", "role"],
    "relation_proportions.csv": [
        "verb",
        "level",
        "agent",
        "patient",
        "true_pairs",
        "total_pairs",
        "proportion",
        "estimated",
    ],
    "relation_pairs.csv": ["verb", "agent", "patient", "holds"],
    "verb_stats.csv": [
        "verb",
        "proportion_true",
        "agents",
        "patients",
        "constraints",
        "families",
        "symmetric_proportion",
        "within_leaf_proportion",
        "estimated",
    ],
    "thematic.csv": ["leaf_a", "leaf_b", "thematic", "similarity"],
}


@pytest.fixture(scope="module")
def tiny() -> TaxonomyResult:
    return generate(load_config(DATA / "tiny_relations.yaml"))


@pytest.fixture(scope="module")
def relations_run() -> TaxonomyResult:
    return generate(load_config(DATA / "relations.yaml"))


@pytest.fixture(scope="module")
def relations_folder(
    relations_run: TaxonomyResult, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return relations_run.write(tmp_path_factory.mktemp("relations") / "run")


# ---------------------------------------------------------------------------------------------
# Category proportions by brute force
# ---------------------------------------------------------------------------------------------


def _brute_force_counts(
    result: TaxonomyResult, verb
) -> dict[tuple[int, str, str], tuple[int, int]]:
    """Count true and total pairs per level and category pair by evaluating every pair one at a
    time through the constraints' own array evaluation."""
    instances = result.instances
    constraints = result.relations.relation(verb).constraints
    n = len(instances)
    counts: dict[tuple[int, str, str], list[int]] = {}
    assignments = {
        level: level_assignment(result.tree, instances, level)
        for level in range(1, result.tree.depth + 1)
    }
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            held = True
            for c in constraints:
                held &= bool(
                    c.evaluate(
                        instances.values[i : i + 1],
                        instances.scalars[i : i + 1],
                        instances.values[j : j + 1],
                        instances.scalars[j : j + 1],
                    )[0]
                )
            for level, (labels, assignment) in assignments.items():
                key = (level, labels[assignment[i]], labels[assignment[j]])
                entry = counts.setdefault(key, [0, 0])
                entry[0] += int(held)
                entry[1] += 1
    return {key: (v[0], v[1]) for key, v in counts.items()}


def test_category_proportions_match_brute_force_on_tiny(tiny: TaxonomyResult) -> None:
    stats = tiny.relation_stats
    assert stats is not None and not stats.estimated
    table = stats.proportions
    assert table.columns == EXPECTED_COLUMNS["relation_proportions.csv"]
    for verb in tiny.verbs.verbs:
        expected = _brute_force_counts(tiny, verb)
        rows = table.filter(pl.col("verb") == verb.label)
        seen = set()
        for row in rows.iter_rows(named=True):
            key = (row["level"], row["agent"], row["patient"])
            true_pairs, total_pairs = expected[key]
            assert row["true_pairs"] == true_pairs
            assert row["total_pairs"] == total_pairs
            assert row["proportion"] == pytest.approx(true_pairs / total_pairs)
            assert row["estimated"] is False
            assert true_pairs > 0
            seen.add(key)
        # Every category pair with a positive count is listed, and no zero rows are.
        assert seen == {key for key, (t, _) in expected.items() if t > 0}


def test_proportions_at_the_leaf_level_use_distinct_instances(
    relations_run: TaxonomyResult,
) -> None:
    table = relations_run.relation_stats.proportions
    leaves = table.filter((pl.col("level") == 3) & (pl.col("agent") == pl.col("patient")))
    sizes = {
        leaf.label: len(relations_run.instances.below(leaf, relations_run.tree))
        for leaf in relations_run.tree.leaves
    }
    for row in leaves.iter_rows(named=True):
        size = sizes[row["agent"]]
        assert row["total_pairs"] == size * (size - 1)
    assert (table["proportion"] > 0).all() and (table["proportion"] <= 1).all()


# ---------------------------------------------------------------------------------------------
# Sampled pairs
# ---------------------------------------------------------------------------------------------


def test_every_sampled_pairs_holds_value_is_correct(
    tiny: TaxonomyResult, relations_run: TaxonomyResult
) -> None:
    for result in (tiny, relations_run):
        pairs = result.relation_stats.pairs
        assert pairs.columns == EXPECTED_COLUMNS["relation_pairs.csv"]
        index = {label: i for i, label in enumerate(result.instances.labels)}
        for verb in result.verbs.verbs:
            rows = pairs.filter(pl.col("verb") == verb.label)
            agents = np.array([index[a] for a in rows["agent"]])
            patients = np.array([index[p] for p in rows["patient"]])
            assert np.array_equal(
                result.relations.holds(verb, agents, patients),
                rows["holds"].to_numpy().astype(bool),
            )
            matrix = result.relations.matrix(verb)
            assert np.array_equal(matrix[agents, patients], rows["holds"].to_numpy().astype(bool))
            assert not (agents == patients).any()
            assert rows.select(["agent", "patient"]).unique().height == rows.height  # no repeats
            true_rows = rows.filter(pl.col("holds") == 1).height
            false_rows = rows.filter(pl.col("holds") == 0).height
            assert true_rows == min(1000, int(matrix.sum()))
            assert false_rows == min(1000, int((~matrix).sum()) - len(result.instances))


def test_pair_shortfalls_are_reported(tiny: TaxonomyResult) -> None:
    short = tiny.relation_stats.pairs_short
    assert short  # 12 instances give at most 132 pairs, fewer than the 1000 requested
    for verb in tiny.verbs.verbs:
        matrix = tiny.relations.matrix(verb)
        entry = short.get(verb.label, {})
        assert entry.get("true", 1000) == min(1000, int(matrix.sum())) or "true" not in entry
        assert entry.get("false") == int((~matrix).sum()) - 12
    assert tiny.summary["verbs"]["pairs_short"] == short


# ---------------------------------------------------------------------------------------------
# Verb statistics and thematic relatedness
# ---------------------------------------------------------------------------------------------


def test_verb_statistics(relations_run: TaxonomyResult) -> None:
    stats = relations_run.relation_stats.verb_stats
    assert stats.columns == EXPECTED_COLUMNS["verb_stats.csv"]
    n = len(relations_run.instances)
    leaf_index = relations_run.instances.leaf_index
    for row in stats.iter_rows(named=True):
        verb = relations_run.verbs.tree[row["verb"]]
        matrix = relations_run.relations.matrix(verb)
        assert row["proportion_true"] == pytest.approx(matrix.sum() / (n * (n - 1)))
        assert row["agents"] == int(
            relations_run.projections.actual_agent[
                :, list(relations_run.projections.verb_labels).index(verb.label)
            ].sum()
        )
        assert row["patients"] == int(matrix.any(axis=0).sum())
        relation = relations_run.relations.relation(verb)
        assert row["constraints"] == len(relation.constraints)
        assert row["families"] == ";".join(c.family for c in relation.constraints)
        i, j = np.nonzero(matrix)
        if len(i):
            assert row["symmetric_proportion"] == pytest.approx(matrix[j, i].mean())
            assert row["within_leaf_proportion"] == pytest.approx(
                (leaf_index[i] == leaf_index[j]).mean()
            )
        assert row["estimated"] is False


def test_thematic_relatedness(relations_run: TaxonomyResult) -> None:
    thematic = relations_run.relation_stats.thematic
    assert thematic.columns == EXPECTED_COLUMNS["thematic.csv"]
    proportions = relations_run.relation_stats.proportions.filter(
        pl.col("level") == relations_run.tree.depth
    )
    expected: dict[tuple[str, str], float] = {}
    order = {leaf.label: k for k, leaf in enumerate(relations_run.tree.leaves)}
    for row in proportions.iter_rows(named=True):
        a, b = sorted((row["agent"], row["patient"]), key=order.__getitem__)
        expected[(a, b)] = expected.get((a, b), 0.0) + row["proportion"]
    assert thematic.height == len(expected) > 0
    for row in thematic.iter_rows(named=True):
        assert row["thematic"] == pytest.approx(expected[(row["leaf_a"], row["leaf_b"])])
        assert row["thematic"] > 0
        a = relations_run.tree[row["leaf_a"]].values
        b = relations_run.tree[row["leaf_b"]].values
        assert row["similarity"] == pytest.approx(pair_similarity(a, b, "cosine"))
        assert order[row["leaf_a"]] <= order[row["leaf_b"]]


# ---------------------------------------------------------------------------------------------
# Estimation above max_exact_pairs
# ---------------------------------------------------------------------------------------------


def test_estimated_proportions_are_close_to_exact() -> None:
    exact = generate(load_config(DATA / "relations.yaml"))
    estimated = generate(
        config_from_mapping(
            {
                **{k: v for k, v in exact.config.resolved().items() if k != "provenance"},
                "verbs": {
                    **exact.config.resolved()["verbs"],
                    "pairs": {"sampled_true": 50, "sampled_false": 50, "max_exact_pairs": 20000},
                },
            }
        )
    )
    assert estimated.relation_stats.estimated and not exact.relation_stats.estimated
    assert estimated.summary["verbs"]["proportions_estimated"] is True
    exact_stats = exact.relation_stats.verb_stats
    for row in estimated.relation_stats.verb_stats.iter_rows(named=True):
        reference = exact_stats.filter(pl.col("verb") == row["verb"]).row(0, named=True)
        assert row["proportion_true"] == pytest.approx(reference["proportion_true"], abs=0.02)
        assert row["estimated"] is True
    assert estimated.relation_stats.proportions["estimated"].all()
    # Sampled pairs are still exactly labelled.
    index = {label: i for i, label in enumerate(estimated.instances.labels)}
    for verb in estimated.verbs.verbs:
        rows = estimated.relation_stats.pairs.filter(pl.col("verb") == verb.label)
        agents = np.array([index[a] for a in rows["agent"]])
        patients = np.array([index[p] for p in rows["patient"]])
        assert np.array_equal(
            estimated.relations.holds(verb, agents, patients), rows["holds"].to_numpy().astype(bool)
        )
        assert rows.filter(pl.col("holds") == 1).height <= 50
    # The constraints and the verb tree do not depend on the pair settings.
    assert estimated.relations.records() == exact.relations.records()


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------


def test_new_files_load_with_the_expected_columns(
    relations_run: TaxonomyResult, relations_folder: Path
) -> None:
    names = sorted(p.name for p in relations_folder.iterdir())
    assert names == sorted(set(OUTPUT_FILES) | set(RELATION_FILES))
    for name, columns in EXPECTED_COLUMNS.items():
        frame = pl.read_csv(relations_folder / name)
        assert frame.columns == columns, name
    verbs = relations_run.verbs
    labels = list(verbs.features.labels)
    for name in ("verbs_generative.csv", "verbs_defining.csv"):
        frame = pl.read_csv(relations_folder / name)
        assert frame.columns == ["label", *labels]
        assert frame["label"].to_list() == [c.label for c in verbs.categories]
    features = pl.read_csv(relations_folder / "verb_features.csv")
    assert features["label"].to_list() == labels
    assert features["constraint"].to_list() == [f"K.{label}" for label in labels]
    assert np.allclose(features["base_rate"].to_numpy(), 0.25)
    tree = pl.read_csv(relations_folder / "verb_tree.csv")
    assert tree["label"].to_list() == [c.label for c in verbs.categories]
    assert tree.filter(pl.col("level") == 1)["parent"].null_count() == 3
    roles = pl.read_csv(relations_folder / "verb_roles.csv")
    assert roles.height == len(verbs.categories) * len(labels)
    assert set(roles["role"].unique()) <= {
        "defining_inherited",
        "defining_new",
        "characteristic",
        "undiagnostic",
    }
    generative = pl.read_csv(relations_folder / "verbs_generative.csv")
    assert np.array_equal(generative.drop("label").to_numpy(), verbs.tree.generative_matrix())
    defining = pl.read_csv(relations_folder / "verbs_defining.csv")
    matrix = defining.drop("label").to_numpy().astype(float)
    for i, category in enumerate(verbs.categories):
        mask = category.defining_mask()
        assert np.array_equal(matrix[i, mask], category.free_values[mask])
        assert np.isnan(matrix[i, ~mask]).all()
    constraints = yaml.safe_load((relations_folder / "constraints.yaml").read_text())
    assert constraints == relations_run.relations.records()
    assert [c["label"] for c in constraints][:12] == [f"K.VF.{i}" for i in range(1, 13)]
    relations = yaml.safe_load((relations_folder / "relations.yaml").read_text())
    assert relations == relations_run.relations.relation_records()
    assert [r["label"] for r in relations] == [c.label for c in verbs.categories]
    assert all(set(r) == {"label", "base", "constraints", "expression"} for r in relations)
    summary = yaml.safe_load((relations_folder / "summary.yaml").read_text())
    block = summary["verbs"]
    assert block["verbs"] == len(verbs.verbs) and block["constraints"] == len(
        relations_run.relations.constraints
    )
    assert sum(block["constraint_families"].values()) == block["constraints"]
    assert block["approximate_projections"] == 0.0
    assert block["proportions_estimated"] is False


def test_no_relation_files_without_verbs(tmp_path: Path) -> None:
    folder = generate(config_from_mapping({"scalars": {"count": 1}})).write(tmp_path / "run")
    assert not any((folder / name).exists() for name in RELATION_FILES)
    assert "verbs" not in yaml.safe_load((folder / "summary.yaml").read_text())


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def _identical(a: Path, b: Path, skip: set[str]) -> None:
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for name in names:
        if name in skip:
            continue
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_same_configuration_and_seed_give_byte_identical_relation_folders(tmp_path: Path) -> None:
    a = generate(load_config(DATA / "tiny_relations.yaml")).write(tmp_path / "a")
    b = generate(load_config(DATA / "tiny_relations.yaml")).write(tmp_path / "b")
    _identical(a, b, skip={"config.yaml"})
    match, mismatch, errors = filecmp.cmpfiles(
        a, b, [p.name for p in a.iterdir() if p.name != "config.yaml"], shallow=False
    )
    assert not mismatch and not errors


def test_instance_count_never_changes_rules_thresholds_verb_tree_or_constraints(
    tmp_path: Path,
) -> None:
    base = {
        k: v
        for k, v in load_config(DATA / "relations.yaml").resolved().items()
        if k != "provenance"
    }
    a = generate(config_from_mapping({**base, "instances": {"per_leaf": 3}})).write(tmp_path / "a")
    b = generate(config_from_mapping({**base, "instances": {"per_leaf": [6, 9]}})).write(
        tmp_path / "b"
    )
    for name in (
        "rules.yaml",
        "verb_tree.csv",
        "verb_roles.csv",
        "verbs_generative.csv",
        "verbs_defining.csv",
        "verb_features.csv",
        "constraints.yaml",
        "relations.yaml",
        "categories_generative.csv",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert (a / "instances.csv").read_bytes() != (b / "instances.csv").read_bytes()
    assert (a / "relation_pairs.csv").read_bytes() != (b / "relation_pairs.csv").read_bytes()


def test_verb_settings_never_change_noun_outputs(tmp_path: Path) -> None:
    base = {
        k: v
        for k, v in load_config(DATA / "relations.yaml").resolved().items()
        if k != "provenance"
    }
    a = generate(config_from_mapping(base)).write(tmp_path / "a")
    b = generate(
        config_from_mapping(
            {
                **base,
                "verbs": {
                    **base["verbs"],
                    "features": {"count": 6, "expected_true": 2},
                    "own_constraint": False,
                },
            }
        )
    ).write(tmp_path / "b")
    for name in OUTPUT_FILES:
        if name in ("config.yaml", "instances.csv", "summary.yaml"):
            continue
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    # The summary differs only in its verb block.
    summary_a = yaml.safe_load((a / "summary.yaml").read_text())
    summary_b = yaml.safe_load((b / "summary.yaml").read_text())
    assert summary_a["verbs"] != summary_b["verbs"]
    summary_a.pop("verbs")
    summary_b.pop("verbs")
    assert summary_a == summary_b
    noun_columns = [
        c for c in pl.read_csv(a / "instances.csv").columns if not c.startswith(("CAN.V", "CANBE."))
    ]
    assert (
        pl.read_csv(a / "instances.csv")
        .select(noun_columns)
        .equals(pl.read_csv(b / "instances.csv").select(noun_columns))
    )


def test_relations_example_runs_in_under_five_minutes(tmp_path: Path) -> None:
    start = time.perf_counter()
    result = generate(load_config(DATA / "relations.yaml"))
    result.write(tmp_path / "run")
    elapsed = time.perf_counter() - start
    assert elapsed < 300, f"{elapsed:.1f} s"


def test_command_line_with_relations(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_world.taxonomy",
            str(DATA / "tiny_relations.yaml"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert run.returncode == 0, run.stderr
    assert "4 verbs" in run.stdout and "constraints" in run.stdout
    assert all((out / name).exists() for name in RELATION_FILES)
    assert hasattr(generate(load_config(DATA / "tiny_relations.yaml")).relations, "holds")
