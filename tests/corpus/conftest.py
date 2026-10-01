"""Fixtures for the corpus generator's tests.

The tests build their own small taxonomies in memory, from the example taxonomy configurations.
No test depends on a saved taxonomy run. Paths inside a corpus configuration are read from the
current folder, so every test runs from the root of the repository.
"""

from __future__ import annotations

import pytest
from corpus_support import DEFAULT_TAXONOMY, REPO, TINY_TAXONOMY

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
