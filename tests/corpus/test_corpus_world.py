"""Stage a5a: loading the world that a corpus is about.

A world configuration file is defined in memory, without its statistics episodes. A world run
folder is regenerated in memory from its ``config.yaml``, and a folder whose rule-set identity,
``entities.csv``, or derived manifest differs from the regenerated world is an error. The
corpus's view of the world carries the labels of the specification, and agrees with the world
package's tables.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from corpus_support import PLAIN_WORLD, REPO, TINY_WORLD, corpus_config, write_world

from semantic_world.corpus import CorpusError, config_from_mapping, load_world, world_identity
from semantic_world.corpus.world import check_run_folder, world_hash
from semantic_world.world.config import load_config as load_world_config
from semantic_world.world.generate import define
from semantic_world.world.labels import translate


def run_folder(tmp_path: Path, seed: int = 1) -> Path:
    return define(load_world_config(TINY_WORLD, seed=seed), episode_stats=False).write(
        tmp_path / "run"
    )


def same_world(a, b) -> bool:
    return (
        a.rule_set_id == b.rule_set_id
        and a.instances == b.instances
        and np.array_equal(a.values, b.values)
        and np.array_equal(a.scalar_values, b.scalar_values)
    )


def test_a_configuration_file_is_defined_in_memory(tiny_world) -> None:
    direct = define(load_world_config(TINY_WORLD), episode_stats=False)
    world = load_world(corpus_config())
    assert world.rule_set_id == direct.rule_set_id == tiny_world.rule_set_id
    assert same_world(world, tiny_world)
    # the statistics episodes are skipped in memory
    assert world.result.stats["episodes"] is None
    # the live objects that the truth tests need are there
    assert world.definition.entity_count == 12 and world.scalars == ("SCALARDIM.1",)
    assert world.binary and world.patient_capacities and world.unary


def test_the_world_seed_chooses_the_world() -> None:
    def world(seed: int):
        return load_world(config_from_mapping({"world": {"config": TINY_WORLD, "seed": seed}}))

    assert same_world(world(1), load_world(corpus_config()))
    assert not same_world(world(2), world(1))
    # the corpus seed is independent of the world's seed
    assert same_world(load_world(corpus_config(seed=99)), world(1))


def test_a_run_folder_is_regenerated_and_checked(tmp_path: Path) -> None:
    folder = run_folder(tmp_path, seed=4)
    config = config_from_mapping({"world": {"run": str(folder)}})
    world = load_world(config)
    direct = define(load_world_config(TINY_WORLD, seed=4), episode_stats=False)
    assert world.rule_set_id == direct.rule_set_id
    # a world source works as well as a whole corpus configuration
    assert same_world(load_world(config.world), world)
    # a written run always has its statistics episodes
    assert "count: 1000" in (folder / "world_stats.yaml").read_text(encoding="utf-8")


def test_a_changed_run_folder_is_an_error(tmp_path: Path) -> None:
    folder = run_folder(tmp_path)
    path = folder / "entities.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    column = lines[0].split(",").index("PROPERTY.1")
    cells[column] = "1" if cells[column] == "0" else "0"
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="entities.csv differs"):
        load_world(config_from_mapping({"world": {"run": str(folder)}}))


def test_a_run_folder_of_another_rule_set_is_an_error(tmp_path: Path, tiny_world) -> None:
    folder = run_folder(tmp_path, seed=2)
    with pytest.raises(CorpusError, match="rule set"):
        check_run_folder(tiny_world.result, folder)


def test_a_changed_manifest_is_an_error(tmp_path: Path, tiny_world) -> None:
    folder = run_folder(tmp_path)
    check_run_folder(tiny_world.result, folder)
    manifest = folder / "derived" / "manifest.yaml"
    saved = manifest.read_text(encoding="utf-8")
    manifest.write_text(saved.replace("capacities.csv", "capacity.csv"), encoding="utf-8")
    with pytest.raises(CorpusError, match="manifest"):
        check_run_folder(tiny_world.result, folder)
    manifest.write_text(saved, encoding="utf-8")
    # config.yaml records the git commit, so a change there is not a difference
    config = folder / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "\n# a note\n", encoding="utf-8")
    (folder / "notes.txt").write_text("an extra file is ignored\n", encoding="utf-8")
    check_run_folder(tiny_world.result, folder)


def test_world_identity(tmp_path: Path) -> None:
    config = corpus_config()
    world = load_world(config)
    identity = world_identity(config.world, world.rule_set_id)
    assert identity == {
        "source": "config",
        "path": TINY_WORLD,
        "name": "tiny",
        "seed": 1,
        "config_hash": world_hash(load_world_config(TINY_WORLD)),
        "rule_set_id": world.rule_set_id,
    }
    assert len(identity["config_hash"]) == 64 and len(identity["rule_set_id"]) == 64
    assert world.identity() == identity
    # the same world has the same hash, whether it comes from a file or from a folder
    folder = run_folder(tmp_path)
    source = config_from_mapping({"world": {"run": str(folder)}}).world
    from_folder = world_identity(source, world.rule_set_id)
    assert from_folder["config_hash"] == identity["config_hash"]
    assert (from_folder["source"], from_folder["path"]) == ("run", str(folder))
    # another seed is another world
    other = config_from_mapping({"world": {"config": TINY_WORLD, "seed": 2}})
    assert world_identity(other.world, "x")["config_hash"] != identity["config_hash"]


# ---------------------------------------------------------------------------------------------
# The corpus's view of the world
# ---------------------------------------------------------------------------------------------


def test_the_view_carries_the_new_labels(tiny_world) -> None:
    world = tiny_world
    assert all(c.startswith("CATEGORY.") for c in world.categories)
    assert all(i.startswith("INSTANCE.") for i in world.instances)
    assert all(f.startswith("PROPERTY.") for f in world.features["is"])
    assert all(f.startswith("PART.") for f in world.features["has"])
    assert all(f.startswith("EVENTTYPE1.") for f in world.features["can"])
    assert all(v.startswith("EVENTTYPE2.") for v in world.binary)
    assert all(c.startswith("CANBE.EVENTTYPE2.") for c in world.patient_capacities)
    assert world.poles == ("SCALARDIM.1.HIGH", "SCALARDIM.1.LOW")
    assert world.leaves == ("CATEGORY.1.1", "CATEGORY.1.2", "CATEGORY.2.1", "CATEGORY.2.2")
    assert world.category["CATEGORY.1.2"].path == ("CATEGORY.1", "CATEGORY.1.2")
    assert world.category["CATEGORY.1"].children == ("CATEGORY.1.1", "CATEGORY.1.2")
    assert world.event_types["EVENTTYPE2.1.1"].path == ("EVENTTYPE2.1", "EVENTTYPE2.1.1")
    assert world.event_names("EVENTTYPE2.1.1") == ("EVENTTYPE2.1.1", "EVENTTYPE2.1")
    assert world.event_types_below("EVENTTYPE2.1") == ("EVENTTYPE2.1.1", "EVENTTYPE2.1.2")
    assert world.instances == tuple(world.definition.entity_labels)


def test_the_view_agrees_with_the_world_packages_tables(tiny_world, tmp_path: Path) -> None:
    world = tiny_world
    folder = world.result.write(tmp_path / "run")
    entities = pl.read_csv(folder / "entities.csv", infer_schema_length=None)
    assert entities["label"].to_list() == list(world.instances)
    for feature in world.features["is"] + world.features["has"]:
        if feature in world.free:
            assert entities[feature].to_list() == world.column(feature).tolist()
    static = pl.read_csv(folder / "derived" / "static_features.csv", infer_schema_length=None)
    for feature in world.features["is"] + world.features["has"]:
        if feature not in world.free:
            assert static[feature].to_list() == world.column(feature).tolist()
    capacities = pl.read_csv(folder / "derived" / "capacities.csv", infer_schema_length=None)
    for event_type in world.unary:
        assert capacities[f"CAN.{event_type}"].to_list() == world.column(event_type).tolist()
        assert world.able(event_type).astype(int).tolist() == world.column(event_type).tolist()
    for label in world.patient_capacities:
        assert capacities[label].to_list() == world.capacity(label).astype(int).tolist()
    # a two-place event type's able matrix is the taxonomy's relation, a category's its base
    # relation
    taxonomy = world.result.taxonomy
    for label in world.binary:
        old = next(c.label for c in taxonomy.verbs.categories if translate(c.label) == label)
        assert np.array_equal(world.able(label), taxonomy.relations.matrix(old))
    thematic = pl.read_csv(folder / "derived" / "thematic.csv", infer_schema_length=None)
    number = {leaf: i for i, leaf in enumerate(world.leaves)}
    for row in thematic.iter_rows(named=True):
        assert world.thematic[number[row["leaf_a"]], number[row["leaf_b"]]] == pytest.approx(
            row["thematic"], abs=1e-6
        )


def test_rule_terms_and_the_fixed_test(tiny_world) -> None:
    world = tiny_world
    terms = world.rule_terms()
    assert terms and world.rule_count == len({t.output for t in terms})
    assert all(t.kind in ("is", "has", "can") for t in terms)
    for term in terms:
        for label, value in term.literals:
            assert label.split(".")[0] in ("PROPERTY", "PART") and isinstance(value, bool)
        # a term is a sufficient condition: the output is fixed at 1 for things with its literals
        if not term.threshold:
            assert world.fixed("THING", term.literals, term.output) == (1, "exact")
    # the fixed test at a category agrees with the taxonomy's defining vectors
    vectors = world.result.taxonomy.vectors
    labels = [translate(x) for x in vectors.feature_labels[vectors.isa_count :]]
    for row, category in enumerate(world.categories):
        defining = vectors.defining[row, vectors.isa_count :]
        for label, value in zip(labels, defining, strict=True):
            fixed, test = world.fixed(category, (), label)
            assert test == "exact"
            assert fixed == (None if np.isnan(value) else int(value)), (category, label)


def test_the_meanings_table_carries_the_new_labels(tiny_world) -> None:
    text = tiny_world.meanings_csv()
    header, first = text.splitlines()[:2]
    columns = header.split(",")
    assert columns[0] == "label" and first.split(",")[0] == "CATEGORY.1"
    assert all(
        c.split(".")[0] in ("ISA", "PROPERTY", "PART", "CAN", "SCALARDIM") for c in columns[1:]
    ), columns
    assert "ISA.CATEGORY.1" in columns and "CAN.EVENTTYPE1.1" in columns


def test_a_world_without_two_place_event_types_loads(tmp_path: Path) -> None:
    world = load_world(corpus_config(write_world(tmp_path, "plain", PLAIN_WORLD)))
    assert world.binary == () and world.patient_capacities == ()
    assert world.scalars == () and world.event_depth is None
    assert not world.thematic.any()
    assert world.definition.base_fluents == ()


def test_the_corpus_package_runs_without_the_rust_core() -> None:
    script = (
        "import sys\n"
        "sys.modules['semantic_world._core'] = None  # makes the import raise ImportError\n"
        "from semantic_world.corpus import Streams, build_lexicon, load_config, load_world\n"
        "config = load_config('data/corpus/tiny.yaml')\n"
        "lexicon = build_lexicon(config, load_world(config), Streams(config.seed))\n"
        "print('lexemes', len(lexicon.lexemes))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "python")}
    run = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO, env=env
    )
    assert run.returncode == 0, run.stderr
    assert "lexemes " in run.stdout
