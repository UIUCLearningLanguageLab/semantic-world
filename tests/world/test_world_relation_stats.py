"""The relation statistics of the world package (stage a5b of
``docs/specs/WORLD_AND_LANGUAGE.md``): category proportions, sampled pairs, event-type
statistics, thematic relatedness, their files in a world run's ``derived/`` folder, determinism,
and the command line. These tests moved from ``tests/taxonomy/test_taxonomy_relation_outputs.py``
(stage 12 of the taxonomy generator). The column ``verb`` is ``event_type`` now, and the verb
files of the taxonomy run (``verb_features.csv``, ``verb_tree.csv``, ``verb_roles.csv``,
``verbs_generative.csv``, ``verbs_defining.csv``, ``constraints.yaml``, ``relations.yaml``) are no
longer written: the event tree and the constraints live in ``definition.json``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.taxonomy.io import OUTPUT_FILES
from semantic_world.taxonomy.similarity import pair_similarity
from semantic_world.world.capacities import CAPACITIES_FILE, CAPACITY_ROLES_FILE
from semantic_world.world.config import Config, config_from_mapping, load_config
from semantic_world.world.derived import DERIVED_DIR, MANIFEST_FILE
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.relation_stats import RELATION_FILES, level_assignment
from semantic_world.world.statics import StaticWorld, build_statics
from semantic_world.world.streams import WorldStreams

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"
RELATIONS = str(TAXONOMY_DATA / "relations.yaml")

EXPECTED_COLUMNS = {
    "relation_proportions.csv": [
        "event_type",
        "level",
        "agent",
        "patient",
        "true_pairs",
        "total_pairs",
        "proportion",
        "estimated",
    ],
    "relation_pairs.csv": ["event_type", "agent", "patient", "holds"],
    "event_type_stats.csv": [
        "event_type",
        "proportion_true",
        "agents",
        "patients",
        "constraints",
        "families",
        "symmetric_proportion",
        "within_leaf_proportion",
        "estimated",
        "leaf_pair_density",
        "tries",
    ],
    "thematic.csv": ["leaf_a", "leaf_b", "thematic", "similarity"],
}
RETIRED_FILES = (
    "verb_features.csv",
    "verb_tree.csv",
    "verb_roles.csv",
    "verbs_generative.csv",
    "verbs_defining.csv",
    "constraints.yaml",
    "relations.yaml",
    "projections.csv",
    "verb_stats.csv",
    "instances.csv",
)


def world_config(
    taxonomy: str | dict = RELATIONS,
    seed: int = 1,
    tmp_path: Path | None = None,
    **blocks,
) -> Config:
    if isinstance(taxonomy, dict):
        assert tmp_path is not None
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(taxonomy, sort_keys=False), encoding="utf-8")
        taxonomy = str(path)
    data = {"name": "test", "seed": seed, "taxonomy": {"config": taxonomy}, **blocks}
    return config_from_mapping(data, source="<test>")


def statics_of(config: Config) -> StaticWorld:
    taxonomy = generate_taxonomy(config.taxonomy_config())
    return build_statics(taxonomy, config, WorldStreams(config.seed), None)


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def tiny_folder(tiny: WorldResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.write(tmp_path_factory.mktemp("world") / "tiny")


@pytest.fixture(scope="module")
def default() -> StaticWorld:
    return statics_of(load_config(DATA / "default.yaml"))


# ---------------------------------------------------------------------------------------------
# Category proportions by brute force
# ---------------------------------------------------------------------------------------------


def _brute_force_counts(
    statics: StaticWorld, event_type
) -> dict[tuple[int, str, str], tuple[int, int]]:
    """Count true and total pairs per level and category pair by evaluating every pair one at a
    time through the constraints' own array evaluation."""
    instances = statics.instances
    tree = statics.taxonomy.tree
    constraints = statics.relations.relation(event_type).constraints
    n = len(instances)
    counts: dict[tuple[int, str, str], list[int]] = {}
    assignments = {
        level: level_assignment(tree, instances, level) for level in range(1, tree.depth + 1)
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


def test_level_assignment_names_each_instances_ancestor(tiny: WorldResult) -> None:
    statics = tiny.statics
    tree = statics.taxonomy.tree
    labels, assignment = level_assignment(tree, statics.instances, 1)
    assert labels == ["CATEGORY.1", "CATEGORY.2"]
    for k, leaf in enumerate(statics.instances.leaf_labels):
        assert labels[assignment[k]] == "CATEGORY." + leaf[len("CATEGORY.") :].split(".")[0]
    leaves, leaf_assignment = level_assignment(tree, statics.instances, 2)
    assert leaves == [leaf.label for leaf in tree.leaves]
    assert [leaves[i] for i in leaf_assignment] == list(statics.instances.leaf_labels)


def test_category_proportions_match_brute_force_on_tiny(tiny: WorldResult) -> None:
    statics = tiny.statics
    stats = statics.relation_stats
    assert stats is not None and not stats.estimated
    table = stats.proportions
    assert table.columns == EXPECTED_COLUMNS["relation_proportions.csv"]
    for event_type in statics.event_tree.event_types:
        expected = _brute_force_counts(statics, event_type)
        rows = table.filter(pl.col("event_type") == event_type.label)
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


def test_proportions_at_the_leaf_level_use_distinct_instances(default: StaticWorld) -> None:
    table = default.relation_stats.proportions
    tree = default.taxonomy.tree
    leaves = table.filter((pl.col("level") == tree.depth) & (pl.col("agent") == pl.col("patient")))
    sizes = {leaf.label: len(default.instances.below(leaf, tree)) for leaf in tree.leaves}
    for row in leaves.iter_rows(named=True):
        size = sizes[row["agent"]]
        assert row["total_pairs"] == size * (size - 1)
    assert (table["proportion"] > 0).all() and (table["proportion"] <= 1).all()
    assert set(table["event_type"].unique()) <= {v.label for v in default.event_tree.event_types}


# ---------------------------------------------------------------------------------------------
# Sampled pairs
# ---------------------------------------------------------------------------------------------


def test_every_sampled_pairs_holds_value_is_correct(
    tiny: WorldResult, default: StaticWorld
) -> None:
    for statics in (tiny.statics, default):
        pairs = statics.relation_stats.pairs
        assert pairs.columns == EXPECTED_COLUMNS["relation_pairs.csv"]
        index = {label: i for i, label in enumerate(statics.instances.labels)}
        for event_type in statics.event_tree.event_types:
            rows = pairs.filter(pl.col("event_type") == event_type.label)
            agents = np.array([index[a] for a in rows["agent"]])
            patients = np.array([index[p] for p in rows["patient"]])
            assert np.array_equal(
                statics.relations.holds(event_type, agents, patients),
                rows["holds"].to_numpy().astype(bool),
            )
            matrix = statics.relations.matrix(event_type)
            assert np.array_equal(matrix[agents, patients], rows["holds"].to_numpy().astype(bool))
            assert not (agents == patients).any()
            assert rows.select(["agent", "patient"]).unique().height == rows.height  # no repeats
            true_rows = rows.filter(pl.col("holds") == 1).height
            false_rows = rows.filter(pl.col("holds") == 0).height
            assert true_rows == min(1000, int(matrix.sum()))
            assert false_rows == min(1000, int((~matrix).sum()) - len(statics.instances))


def test_pair_shortfalls_are_reported(tiny: WorldResult) -> None:
    statics = tiny.statics
    short = statics.relation_stats.pairs_short
    assert short  # 12 instances give at most 132 pairs, fewer than the 1000 requested
    for event_type in statics.event_tree.event_types:
        matrix = statics.relations.matrix(event_type)
        entry = short.get(event_type.label, {})
        assert entry.get("true", 1000) == min(1000, int(matrix.sum())) or "true" not in entry
        assert entry.get("false") == int((~matrix).sum()) - 12


# ---------------------------------------------------------------------------------------------
# Event-type statistics and thematic relatedness
# ---------------------------------------------------------------------------------------------


def test_event_type_statistics(default: StaticWorld) -> None:
    stats = default.relation_stats.event_type_stats
    assert stats.columns == EXPECTED_COLUMNS["event_type_stats.csv"]
    n = len(default.instances)
    leaf_index = default.instances.leaf_index
    projections = default.projections
    assert stats["event_type"].to_list() == list(projections.event_type_labels)
    for row in stats.iter_rows(named=True):
        event_type = default.event_tree.tree[row["event_type"]]
        matrix = default.relations.matrix(event_type)
        assert row["proportion_true"] == pytest.approx(matrix.sum() / (n * (n - 1)))
        column = list(projections.event_type_labels).index(event_type.label)
        assert row["agents"] == int(projections.actual_agent[:, column].sum())
        assert row["patients"] == int(matrix.any(axis=0).sum())
        relation = default.relations.relation(event_type)
        assert row["constraints"] == len(relation.constraints)
        assert row["families"] == ";".join(c.family for c in relation.constraints)
        i, j = np.nonzero(matrix)
        if len(i):
            assert row["symmetric_proportion"] == pytest.approx(matrix[j, i].mean())
            assert row["within_leaf_proportion"] == pytest.approx(
                (leaf_index[i] == leaf_index[j]).mean()
            )
        assert row["estimated"] is False


def test_thematic_relatedness(default: StaticWorld) -> None:
    thematic = default.relation_stats.thematic
    assert thematic.columns == EXPECTED_COLUMNS["thematic.csv"]
    tree = default.taxonomy.tree
    proportions = default.relation_stats.proportions.filter(pl.col("level") == tree.depth)
    expected: dict[tuple[str, str], float] = {}
    order = {leaf.label: k for k, leaf in enumerate(tree.leaves)}
    for row in proportions.iter_rows(named=True):
        a, b = sorted((row["agent"], row["patient"]), key=order.__getitem__)
        expected[(a, b)] = expected.get((a, b), 0.0) + row["proportion"]
    assert thematic.height == len(expected) > 0
    for row in thematic.iter_rows(named=True):
        assert row["thematic"] == pytest.approx(expected[(row["leaf_a"], row["leaf_b"])])
        assert row["thematic"] > 0
        # The similarity is the taxonomy's, over its generative vectors (no CAN columns).
        a = tree[row["leaf_a"]].values
        b = tree[row["leaf_b"]].values
        assert row["similarity"] == pytest.approx(pair_similarity(a, b, "cosine"))
        assert order[row["leaf_a"]] <= order[row["leaf_b"]]


# ---------------------------------------------------------------------------------------------
# Estimation above max_exact_pairs
# ---------------------------------------------------------------------------------------------


def test_estimated_proportions_are_close_to_exact(default: StaticWorld) -> None:
    # The same world as the default fixture (the default file's event-type tree), with the pair
    # settings changed.
    binary = load_config(DATA / "default.yaml").resolved()["event_types"]["binary"]
    estimated = statics_of(
        world_config(
            event_types={
                "binary": {
                    "taxonomy": binary["taxonomy"],
                    "pairs": {"sampled_true": 50, "sampled_false": 50, "max_exact_pairs": 20000},
                }
            }
        )
    )
    assert estimated.relation_stats.estimated and not default.relation_stats.estimated
    exact_stats = default.relation_stats.event_type_stats
    for row in estimated.relation_stats.event_type_stats.iter_rows(named=True):
        reference = exact_stats.filter(pl.col("event_type") == row["event_type"]).row(0, named=True)
        assert row["proportion_true"] == pytest.approx(reference["proportion_true"], abs=0.02)
        assert row["estimated"] is True
    assert estimated.relation_stats.proportions["estimated"].all()
    # Sampled pairs are still exactly labelled.
    index = {label: i for i, label in enumerate(estimated.instances.labels)}
    for event_type in estimated.event_tree.event_types:
        rows = estimated.relation_stats.pairs.filter(pl.col("event_type") == event_type.label)
        agents = np.array([index[a] for a in rows["agent"]])
        patients = np.array([index[p] for p in rows["patient"]])
        assert np.array_equal(
            estimated.relations.holds(event_type, agents, patients),
            rows["holds"].to_numpy().astype(bool),
        )
        assert rows.filter(pl.col("holds") == 1).height <= 50
    # The constraints, the event tree, and the capacities do not depend on the pair settings.
    assert estimated.relations.records() == default.relations.records()
    assert np.array_equal(estimated.projections.agent, default.projections.agent)
    assert np.array_equal(estimated.unary.values, default.unary.values)


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------


def test_relation_files_load_with_the_expected_columns(
    tiny: WorldResult, tiny_folder: Path
) -> None:
    derived = tiny_folder / DERIVED_DIR
    names = sorted(p.name for p in derived.iterdir())
    assert names == sorted(
        {
            MANIFEST_FILE,
            "static_features.csv",
            CAPACITIES_FILE,
            CAPACITY_ROLES_FILE,
            *RELATION_FILES,
        }
    )
    assert set(RELATION_FILES) == set(EXPECTED_COLUMNS)
    frames = tiny.statics.relation_stats.frames()
    assert list(frames) == list(RELATION_FILES)
    for name, columns in EXPECTED_COLUMNS.items():
        frame = pl.read_csv(derived / name)
        assert frame.columns == columns, name
        assert frames[name].columns == columns
        assert frame.height == frames[name].height
    manifest = yaml.safe_load((derived / MANIFEST_FILE).read_text())
    for name in RELATION_FILES:
        assert manifest["files"][name]["rule_set_id"] == tiny.rule_set_id
    # Nothing of the taxonomy's old relation output survives in either folder.
    taxonomy_folder = tiny_folder / "taxonomy"
    assert sorted(p.name for p in taxonomy_folder.iterdir()) == sorted(
        set(OUTPUT_FILES) | {"rule_matrices.json", DERIVED_DIR}
    )
    for name in RETIRED_FILES:
        assert not (taxonomy_folder / name).exists(), name
        assert not (derived / name).exists(), name
    stats = pl.read_csv(derived / "event_type_stats.csv")
    assert stats["event_type"].to_list() == [v.label for v in tiny.statics.event_tree.event_types]
    assert "verb" not in stats.columns


def test_no_relation_files_without_binary(tmp_path: Path) -> None:
    config = config_from_mapping(
        {
            "name": "tiny",
            "seed": 1,
            "taxonomy": {"config": str(TAXONOMY_DATA / "tiny_relations.yaml")},
            "event_types": {"unary": {"count": 4}, "binary": None},
        },
        source="<test>",
    )
    world = define(config)
    assert world.statics.relation_stats is None
    frames = world.derived_frames()
    assert not any(name in frames for name in RELATION_FILES)
    folder = world.write(tmp_path / "run")
    assert not any((folder / DERIVED_DIR / name).exists() for name in RELATION_FILES)
    assert (folder / DERIVED_DIR / CAPACITIES_FILE).is_file()


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def test_same_configuration_and_seed_give_byte_identical_relation_files(
    tiny_folder: Path, tmp_path: Path
) -> None:
    again = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "again")
    names = _files(tiny_folder)
    assert names == _files(again)
    for name in names:
        if Path(name).name == "config.yaml":
            continue
        assert (tiny_folder / name).read_bytes() == (again / name).read_bytes(), name
    assert any(name.startswith(f"{DERIVED_DIR}/relation_") for name in names)


def test_instance_count_never_changes_rules_thresholds_event_tree_or_constraints(
    tmp_path: Path,
) -> None:
    base = yaml.safe_load(Path(RELATIONS).read_text(encoding="utf-8"))
    a = statics_of(world_config({**base, "instances": {"per_leaf": 3}}, tmp_path=tmp_path / "a"))
    b = statics_of(
        world_config({**base, "instances": {"per_leaf": [6, 9]}}, tmp_path=tmp_path / "b")
    )
    assert a.taxonomy.rules.records() == b.taxonomy.rules.records()
    assert a.unary.rules.records() == b.unary.rules.records()
    assert np.array_equal(a.taxonomy.tree.generative_matrix(), b.taxonomy.tree.generative_matrix())
    assert a.relations.records() == b.relations.records()
    assert a.relations.relation_records() == b.relations.relation_records()
    assert [c.label for c in a.event_tree.categories] == [c.label for c in b.event_tree.categories]
    assert np.array_equal(
        a.event_tree.tree.generative_matrix(), b.event_tree.tree.generative_matrix()
    )
    assert len(a.instances) < len(b.instances)
    assert not a.relation_stats.pairs.equals(b.relation_stats.pairs)


def test_binary_settings_never_change_the_taxonomy_or_the_one_place_side(tmp_path: Path) -> None:
    a = define(
        world_config(
            str(TAXONOMY_DATA / "tiny_relations.yaml"), event_types={"unary": {"count": 4}}
        )
    )
    b = define(
        world_config(
            str(TAXONOMY_DATA / "tiny_relations.yaml"),
            event_types={
                "unary": {"count": 4},
                "binary": {"features": {"count": 6, "expected_true": 2}, "own_constraint": False},
            },
        )
    )
    folder_a = a.write(tmp_path / "a")
    folder_b = b.write(tmp_path / "b")
    for name in _files(folder_a / "taxonomy"):
        if Path(name).name != "config.yaml":
            assert (folder_a / "taxonomy" / name).read_bytes() == (
                folder_b / "taxonomy" / name
            ).read_bytes(), name
    assert a.statics.unary.rules.records() == b.statics.unary.rules.records()
    assert np.array_equal(a.statics.unary.values, b.statics.unary.values)
    one_place = [f"CAN.{label}" for label in a.statics.unary.labels]
    capacities_a = a.derived_frames()[CAPACITIES_FILE].select(["label", *one_place])
    capacities_b = b.derived_frames()[CAPACITIES_FILE].select(["label", *one_place])
    assert capacities_a.equals(capacities_b)
    assert a.statics.relations.records() != b.statics.relations.records()


def test_default_world_statics_build_in_under_five_minutes() -> None:
    start = time.perf_counter()
    statics = statics_of(load_config(DATA / "default.yaml"))
    elapsed = time.perf_counter() - start
    assert elapsed < 300, f"{elapsed:.1f} s"
    assert statics.relation_stats is not None


def test_command_line_writes_the_relation_files(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_world.world",
            "define",
            str(DATA / "tiny.yaml"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert run.returncode == 0, run.stderr
    assert "4 two-place event types" in run.stdout and "constraints" in run.stdout
    assert all((out / DERIVED_DIR / name).exists() for name in RELATION_FILES)
    stats = pl.read_csv(out / DERIVED_DIR / "event_type_stats.csv")
    assert stats.columns == EXPECTED_COLUMNS["event_type_stats.csv"]
