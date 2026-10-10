"""The snapshot of the runs' figures (``tests/figures/snapshot.yaml``).

One test compares the figures of the tiny world and corpus, regenerated in memory into a
temporary folder, with the snapshot (the fast tier), and one the figures of the default world and
corpus (the slow tier). The snapshot is rewritten on purpose by
``python -m semantic_world.figures snapshot``, which reads ``runs/`` and prints what moved; the
tests never read ``runs/``. The baselines of the off settings in the same file are checked by
``tests/world/test_world_off_settings.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_world.corpus import generate
from semantic_world.corpus import load_config as load_corpus_config
from semantic_world.figures import (
    CORPUS_CONFIGS,
    REPO,
    SNAPSHOT_FILE,
    WORLD_CONFIGS,
    corpus_figures,
    flatten,
    load_snapshot,
    moves,
    world_figures,
)
from semantic_world.world.config import load_config as load_world_config
from semantic_world.world.generate import define

SLOW = pytest.mark.slow
NAMES = ["tiny", pytest.param("default", marks=SLOW)]


@pytest.fixture(autouse=True)
def _run_from_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)


def _differences(expected: dict, found: dict) -> str:
    return "\n".join(
        f"{key}: snapshot {old!r}, run {new!r}" for key, old, new in moves(expected, found)
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_world_figures_are_the_snapshots(name: str, tmp_path: Path) -> None:
    snapshot = load_snapshot(SNAPSHOT_FILE)
    folder = define(load_world_config(REPO / WORLD_CONFIGS[name])).write(tmp_path / "world")
    found = world_figures(folder)
    assert found == snapshot[name]["world"], _differences(snapshot[name]["world"], found)


@pytest.mark.parametrize("name", NAMES)
def test_the_corpus_figures_are_the_snapshots(name: str, tmp_path: Path) -> None:
    snapshot = load_snapshot(SNAPSHOT_FILE)
    corpus = generate(load_corpus_config(REPO / CORPUS_CONFIGS[name]))
    found = corpus_figures(corpus.write(tmp_path / "corpus"))
    assert found == snapshot[name]["corpus"], _differences(snapshot[name]["corpus"], found)
    # the corpus is about the snapshot's world
    assert found["world"] == snapshot[name]["world"]["rule_set_short"]


def test_the_snapshot_holds_both_runs_and_the_baselines() -> None:
    snapshot = load_snapshot(SNAPSHOT_FILE)
    assert set(snapshot) == {"tiny", "default", "baselines"}
    for name in ("tiny", "default"):
        assert set(snapshot[name]) == {"world", "corpus"}
    keys = flatten(snapshot)
    assert all(not isinstance(v, float) or v == round(v, 6) for v in keys.values())
