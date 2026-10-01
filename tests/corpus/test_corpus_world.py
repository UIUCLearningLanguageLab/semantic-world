"""Stage 1: loading the taxonomy that a corpus is about.

A taxonomy configuration file is generated in memory. A taxonomy output folder is regenerated in
memory from its ``config.yaml``, and a folder that differs from the regenerated result is an
error.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from corpus_support import PLAIN_TAXONOMY, REPO, TINY_TAXONOMY, corpus_config

from semantic_world.corpus import CorpusError, config_from_mapping, load_taxonomy, taxonomy_identity
from semantic_world.corpus.world import check_run_folder, taxonomy_hash
from semantic_world.taxonomy import generate
from semantic_world.taxonomy import load_config as load_taxonomy_config
from semantic_world.taxonomy.io import RELATION_FILES


def run_folder(tmp_path: Path, seed: int = 1) -> Path:
    return generate(load_taxonomy_config(TINY_TAXONOMY, seed=seed)).write(tmp_path / "run")


def same_world(a, b) -> bool:
    return (
        a.instances.labels == b.instances.labels
        and np.array_equal(a.instances.values, b.instances.values)
        and np.array_equal(a.instances.scalars, b.instances.scalars)
        and a.relations.relation_records() == b.relations.relation_records()
    )


def test_a_configuration_file_is_generated_in_memory(tiny_world) -> None:
    direct = generate(load_taxonomy_config(TINY_TAXONOMY))
    assert same_world(load_taxonomy(corpus_config()), direct)
    assert same_world(tiny_world, direct)
    # the live objects that the truth tests need are there
    assert tiny_world.relations is not None and tiny_world.projections is not None
    assert tiny_world.verbs is not None and tiny_world.features.scalar_count == 1


def test_the_taxonomy_seed_chooses_the_world() -> None:
    def world(seed: int):
        return load_taxonomy(
            config_from_mapping({"taxonomy": {"config": TINY_TAXONOMY, "seed": seed}})
        )

    assert same_world(world(1), load_taxonomy(corpus_config()))
    assert not same_world(world(2), world(1))
    # the corpus seed is independent of the taxonomy seed
    assert same_world(load_taxonomy(corpus_config(seed=99)), world(1))


def test_an_output_folder_is_regenerated_and_checked(tmp_path: Path) -> None:
    folder = run_folder(tmp_path, seed=4)
    config = config_from_mapping({"taxonomy": {"run": str(folder)}})
    result = load_taxonomy(config)
    assert same_world(result, generate(load_taxonomy_config(TINY_TAXONOMY, seed=4)))
    # a taxonomy source works as well as a whole corpus configuration
    assert same_world(load_taxonomy(config.taxonomy), result)


@pytest.mark.parametrize("name", ["instances.csv", "rules.yaml", "relation_proportions.csv"])
def test_a_changed_output_folder_is_an_error(tmp_path: Path, name: str) -> None:
    folder = run_folder(tmp_path)
    path = folder / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CorpusError) as info:
        load_taxonomy(config_from_mapping({"taxonomy": {"run": str(folder)}}))
    assert name in str(info.value) and "differs" in str(info.value)
    assert str(folder) in str(info.value)


def test_a_changed_value_in_an_output_folder_is_an_error(tmp_path: Path) -> None:
    folder = run_folder(tmp_path)
    path = folder / "instances.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    column = lines[0].split(",").index("IS.1")
    cells[column] = "1" if cells[column] == "0" else "0"
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="instances.csv differs"):
        load_taxonomy(config_from_mapping({"taxonomy": {"run": str(folder)}}))


def test_a_missing_file_in_an_output_folder_is_an_error(tmp_path: Path) -> None:
    folder = run_folder(tmp_path)
    (folder / "thematic.csv").unlink()
    with pytest.raises(CorpusError, match="thematic.csv is missing"):
        load_taxonomy(config_from_mapping({"taxonomy": {"run": str(folder)}}))


def test_the_check_covers_every_file_but_the_configuration(tmp_path: Path, tiny_world) -> None:
    folder = run_folder(tmp_path)
    written = {p.name for p in folder.iterdir()}
    assert set(RELATION_FILES) <= written and "instances.csv" in written
    # config.yaml records the git commit, so a change there is not a difference
    config = folder / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "\n# a note\n", encoding="utf-8")
    (folder / "notes.txt").write_text("an extra file is ignored\n", encoding="utf-8")
    check_run_folder(tiny_world, folder)
    for name in sorted(written - {"config.yaml"}):
        path = folder / name
        saved = path.read_bytes()
        path.write_bytes(saved + b"x")
        with pytest.raises(CorpusError, match=name):
            check_run_folder(tiny_world, folder)
        path.write_bytes(saved)
    check_run_folder(tiny_world, folder)


def test_a_folder_of_another_seed_does_not_match(tmp_path: Path, tiny_world) -> None:
    with pytest.raises(CorpusError):
        check_run_folder(tiny_world, run_folder(tmp_path, seed=2))


def test_taxonomy_identity(tmp_path: Path) -> None:
    config = corpus_config()
    identity = taxonomy_identity(config.taxonomy)
    assert identity == {
        "source": "config",
        "path": TINY_TAXONOMY,
        "name": "tiny_relations",
        "seed": 1,
        "config_hash": taxonomy_hash(load_taxonomy_config(TINY_TAXONOMY)),
    }
    assert len(identity["config_hash"]) == 64
    # the same taxonomy has the same hash, whether it comes from a file or from a folder
    folder = run_folder(tmp_path)
    from_folder = taxonomy_identity(
        config_from_mapping({"taxonomy": {"run": str(folder)}}).taxonomy
    )
    assert from_folder["config_hash"] == identity["config_hash"]
    assert (from_folder["source"], from_folder["path"]) == ("run", str(folder))
    # another seed, or another setting, is another taxonomy
    other_seed = config_from_mapping({"taxonomy": {"config": TINY_TAXONOMY, "seed": 2}})
    assert taxonomy_identity(other_seed.taxonomy)["config_hash"] != identity["config_hash"]
    plain = corpus_config(PLAIN_TAXONOMY)
    assert taxonomy_identity(plain.taxonomy)["config_hash"] != identity["config_hash"]


def test_a_taxonomy_without_verbs_or_scalars_loads() -> None:
    result = load_taxonomy(corpus_config(PLAIN_TAXONOMY))
    assert result.verbs is None and result.relations is None and result.projections is None
    assert result.features.scalar_count == 0


def test_the_corpus_package_runs_without_the_rust_core() -> None:
    script = (
        "import sys\n"
        "sys.modules['semantic_world._core'] = None  # makes the import raise ImportError\n"
        "from semantic_world.corpus import Streams, build_lexicon, load_config, load_taxonomy\n"
        "config = load_config('data/corpus/tiny.yaml')\n"
        "lexicon = build_lexicon(config, load_taxonomy(config), Streams(config.seed))\n"
        "print('lexemes', len(lexicon.lexemes))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "python")}
    run = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO, env=env
    )
    assert run.returncode == 0, run.stderr
    assert "lexemes 50" in run.stdout
