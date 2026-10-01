"""Shared helpers for the corpus generator's tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from semantic_world.corpus import Config, config_from_mapping

REPO = Path(__file__).resolve().parents[2]
TINY_TAXONOMY = "data/taxonomy/tiny_relations.yaml"
DEFAULT_TAXONOMY = "data/taxonomy/relations.yaml"
PLAIN_TAXONOMY = "data/taxonomy/tiny.yaml"
"""The tiny taxonomy without scalar dimensions and without verbs."""


def corpus_config(taxonomy: str = TINY_TAXONOMY, seed: int = 1, **sections: Any) -> Config:
    """A corpus configuration over an example taxonomy, with the given sections changed."""
    data = {"name": "test", "seed": seed, "taxonomy": {"config": taxonomy}}
    return config_from_mapping({**data, **copy.deepcopy(sections)})
