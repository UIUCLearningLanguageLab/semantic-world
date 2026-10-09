"""Fixtures for the corpus generator's tests.

The tests build their own small worlds in memory, from the example world configurations and
from taxonomies written for the tests. No test depends on a saved run. Paths inside a corpus
configuration are read from the current folder, so every test runs from the root of the
repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus_support import (
    DEFAULT_WORLD,
    EXAMPLE_WORLDS,
    PLAIN_WORLD,
    REPO,
    STATIC_WORLD,
    TINY_WORLD,
    WRITTEN_EVENT_TYPES,
    WRITTEN_SEEDS,
    WRITTEN_TAXONOMIES,
    Case,
    world_over,
    write_world,
)

from semantic_world.corpus import config_from_mapping, load_world


@pytest.fixture(autouse=True)
def _run_from_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)


def _world(path: str):
    # Session fixtures are made before the autouse fixture changes the folder.
    return load_world(config_from_mapping({"world": {"config": str(REPO / path)}}))


@pytest.fixture(scope="session")
def tiny_world():
    """The tiny world: 6 categories, 12 instances, 1 scalar, 4 one-place and 4 two-place event
    types, and 4 fluents."""
    return _world(TINY_WORLD)


@pytest.fixture(scope="session")
def default_world():
    """The default world: 56 categories, 2 scalars, 7 two-place event types, and 8 fluents."""
    return _world(DEFAULT_WORLD)


@pytest.fixture(scope="session")
def world_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """World configuration files written for the tests: ``plain`` (one-place event types only,
    no fluents) and ``static`` (the tiny relations taxonomy with no fluents)."""
    root = tmp_path_factory.mktemp("world_files")
    return {
        "plain": write_world(root, "plain", PLAIN_WORLD),
        "static": write_world(root, "static", STATIC_WORLD),
    }


@pytest.fixture(scope="session")
def cases(tmp_path_factory: pytest.TempPathFactory):
    """The worlds of the proposition tests, by name, each made once: the ``tiny`` and
    ``default`` worlds, ``deep`` (chained rules and two scalars), and ``still`` (a scalar
    without drift, and a flat event-type tree), the last two as world configurations over
    taxonomies written for the tests. Each has its world run folder, so truth can be
    recomputed from the files."""
    root = tmp_path_factory.mktemp("worlds")
    made: dict[str, Case] = {}

    def case(name: str) -> Case:
        if name not in made:
            if name in WRITTEN_TAXONOMIES:
                taxonomy = root / f"{name}_taxonomy.yaml"
                taxonomy.write_text(WRITTEN_TAXONOMIES[name], encoding="utf-8")
                world = write_world(
                    root,
                    name,
                    world_over(name, str(taxonomy), WRITTEN_SEEDS[name], WRITTEN_EVENT_TYPES[name]),
                )
            else:
                world = str(REPO / EXAMPLE_WORLDS[name])
            made[name] = Case(name, world, root / name)
        return made[name]

    return case


RUN_COUNTS = {"tiny": 60, "default": 200, "deep": 80, "still": 60}
"""The number of documents of a corpus run in the tests, by world."""
RUN_SETTINGS = {"test_sets": {"size": 30}}
"""Small test sets, so that a run is fast."""


@pytest.fixture(scope="session")
def runs(cases):
    """Whole corpus runs of a world under given settings, made once: the documents, the test
    sets, and the statistics (``semantic_world.corpus.generate.Corpus``)."""
    from semantic_world.corpus.generate import generate

    made: dict = {}

    def run(name: str, count: int | None = None, **sections):
        key = (name, count, json.dumps(sections, sort_keys=True))
        if key not in made:
            settings = {**RUN_SETTINGS, **sections}
            documents = dict(settings.get("documents", {}))
            documents["count"] = count or RUN_COUNTS[name]
            settings["documents"] = documents
            case = cases(name)
            made[key] = generate(case.config(**settings), case.world)
        return made[key]

    return run


__all__ = ["DEFAULT_WORLD", "Path"]
