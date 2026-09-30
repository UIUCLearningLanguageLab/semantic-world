"""Fixtures for the word-form pipeline's tests."""

from __future__ import annotations

import pytest
from wordforms_support import DATA


@pytest.fixture(scope="session")
def english():
    pytest.importorskip("cmudict", reason="the cmudict package is not installed")
    from semantic_world.wordforms.english import load_english

    return load_english()


@pytest.fixture
def default_config():
    from semantic_world.wordforms.config import load_config

    return load_config(DATA / "default.yaml")


@pytest.fixture
def tiny_config():
    from semantic_world.wordforms.config import load_config

    return load_config(DATA / "tiny.yaml")
