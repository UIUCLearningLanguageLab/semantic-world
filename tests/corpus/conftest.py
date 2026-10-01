"""Fixtures for the corpus generator's tests.

The tests build their own small taxonomies in memory, from the example taxonomy configurations.
No test depends on a saved taxonomy run. Paths inside a corpus configuration are read from the
current folder, so every test runs from the root of the repository.
"""

from __future__ import annotations

import pytest
from corpus_support import (
    DEFAULT_TAXONOMY,
    EXAMPLE_TAXONOMIES,
    REPO,
    TINY_TAXONOMY,
    WRITTEN_TAXONOMIES,
    Case,
)

from semantic_world.corpus import config_from_mapping, load_taxonomy


@pytest.fixture(autouse=True)
def _run_from_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)


def _world(taxonomy: str):
    # Session fixtures are made before the autouse fixture changes the folder.
    return load_taxonomy(config_from_mapping({"taxonomy": {"config": str(REPO / taxonomy)}}))


@pytest.fixture(scope="session")
def tiny_world():
    """The tiny relations taxonomy: 6 categories, 12 instances, 1 scalar, and 4 verbs."""
    return _world(TINY_TAXONOMY)


@pytest.fixture(scope="session")
def default_world():
    """The default relations taxonomy: 56 categories, 2 scalars, and 7 verbs."""
    return _world(DEFAULT_TAXONOMY)


@pytest.fixture(scope="session")
def cases(tmp_path_factory: pytest.TempPathFactory):
    """The worlds of the proposition tests, by name, each made once: the ``tiny`` and ``default``
    relations taxonomies, ``deep`` (chained rules and two scalars), and ``still`` (a scalar
    without drift, and a flat verb tree). Each has its taxonomy output folder, so truth can be
    recomputed from the files."""
    root = tmp_path_factory.mktemp("worlds")
    made: dict[str, Case] = {}

    def case(name: str) -> Case:
        if name not in made:
            if name in WRITTEN_TAXONOMIES:
                path = root / f"{name}.yaml"
                path.write_text(WRITTEN_TAXONOMIES[name], encoding="utf-8")
                taxonomy = str(path)
            else:
                taxonomy = str(REPO / EXAMPLE_TAXONOMIES[name])
            made[name] = Case(name, taxonomy, root / name)
        return made[name]

    return case
