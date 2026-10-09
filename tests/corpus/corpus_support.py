"""Shared helpers for the corpus generator's tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from semantic_world.corpus import Config, config_from_mapping

REPO = Path(__file__).resolve().parents[2]
TINY_WORLD = "data/world/tiny.yaml"
DEFAULT_WORLD = "data/world/default.yaml"
TINY_TAXONOMY = "data/taxonomy/tiny_relations.yaml"
DEFAULT_TAXONOMY = "data/taxonomy/relations.yaml"
PLAIN_TAXONOMY = "data/taxonomy/tiny.yaml"
"""The tiny taxonomy without scalar dimensions and without verbs."""

PLAIN_WORLD = """\
# The tiny taxonomy without scalars, with no two-place event types and no fluents: a world of
# one-place event types only.
name: plain
seed: 1
taxonomy: {config: data/taxonomy/tiny.yaml}
fluents: {count: 0}
event_types:
  unary: {count: 4}
  binary: null
"""

STATIC_WORLD = """\
# The tiny relations taxonomy with no fluents: every able binding is legal at every time point,
# as in the corpus's old scene generator.
name: static
seed: 1
taxonomy: {config: data/taxonomy/tiny_relations.yaml}
fluents: {count: 0}
event_types:
  unary: {count: 4}
  binary:
    features: {count: 4, expected_true: 2}
    taxonomy: {superordinates: 2, depth: 2, branching: 2}
"""


def corpus_config(world: str = TINY_WORLD, seed: int = 1, **sections: Any) -> Config:
    """A corpus configuration over a world configuration file, with the given sections
    changed."""
    data = {"name": "test", "seed": seed, "world": {"config": world}}
    return config_from_mapping({**data, **copy.deepcopy(sections)})


def write_world(folder: Path, name: str, text: str) -> str:
    """Write a world configuration file into a folder and return its path."""
    path = folder / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------------------------
# Worlds for the proposition tests
# ---------------------------------------------------------------------------------------------

DEEP_TAXONOMY = """\
# A small taxonomy whose rules chain: some determined features are computed from other
# determined features, and some rules read scalar thresholds. Equal base rates, as before the
# heterogeneous default.
name: deep
seed: 3
features:
  property: {count: 12, proportion_determined: 0.5, expected_true_free: 2.5}
  part: {count: 12, proportion_determined: 0.5, expected_true_free: 2.5}
  base_rate_heterogeneity: null
rules: {max_chain_depth: 2}
taxonomy: {superordinates: 3, depth: 2, branching: [2, 3]}
instances: {per_leaf: [4, 6]}
scalars: {count: 2}
"""

DEEP_EVENT_TYPES = """\
event_types:
  unary: {count: 6}
  binary:
    features: {count: 6, expected_true: 2}
    taxonomy: {superordinates: 2, depth: 2, branching: 2}
"""

STILL_TAXONOMY = """\
# A small taxonomy whose scalar never drifts, so a threshold literal is fixed at every category.
# Equal base rates, as before the heterogeneous default.
name: still
seed: 2
features:
  property: {count: 8, proportion_determined: 0.5, expected_true_free: 2}
  part: {count: 8, proportion_determined: 0.5, expected_true_free: 2}
  base_rate_heterogeneity: null
rules: {input_type_weights: {property: 1, part: 1, scalar: 3}}
taxonomy: {superordinates: 3, depth: 2, branching: 2}
instances: {per_leaf: 4}
scalars: {count: 1, drift: 0, instance_drift: 0}
"""

STILL_EVENT_TYPES = """\
event_types:
  unary: {count: 6}
  binary:
    features: {count: 4, expected_true: 2}
    taxonomy: {superordinates: 3, depth: 1, branching: 2}
"""
"""The still world's event-type tree is flat: three event types and no category. (Two event
types, as before stage a5b, hold for 1% and 7% of the pairs under the new draws, which leaves
the scenes almost without two-place events.)"""

WRITTEN_TAXONOMIES = {"deep": DEEP_TAXONOMY, "still": STILL_TAXONOMY}
WRITTEN_EVENT_TYPES = {"deep": DEEP_EVENT_TYPES, "still": STILL_EVENT_TYPES}
WRITTEN_SEEDS = {"deep": 3, "still": 2}
EXAMPLE_WORLDS = {"tiny": TINY_WORLD, "default": DEFAULT_WORLD}


def world_over(name: str, taxonomy_path: str, seed: int, event_types: str = "") -> str:
    """A world configuration over a written taxonomy, with the default fluents, and the given
    ``event_types`` block (the defaults when none is given)."""
    return (
        f"name: {name}\nseed: {seed}\ntaxonomy: {{config: {taxonomy_path}, seed: {seed}}}\n"
        + event_types
    )


class Case:
    """One world for the proposition tests: its configuration file, the world run folder, and
    the facts and the independent oracle under any corpus settings."""

    def __init__(self, name: str, world_path: str, folder: Path) -> None:
        from semantic_world.corpus import load_world

        self.name = name
        self.world_path = world_path
        self.world = load_world(self.config())
        self.result = self.world
        """The world, under the name the earlier tests used."""
        self.folder = self.world.result.write(folder)
        self._facts: dict[str, Any] = {}

    def config(self, **sections: Any) -> Config:
        return corpus_config(self.world_path, **sections)

    def facts(self, **sections: Any):
        """The facts of the world, with the lexicon of the same settings. Cached by settings."""
        from semantic_world.corpus import Streams, build_lexicon
        from semantic_world.corpus.facts import Facts

        key = repr(sorted(sections.items()))
        if key not in self._facts:
            config = self.config(**sections)
            lexicon = build_lexicon(config, self.world, Streams(config.seed))
            self._facts[key] = Facts(config, self.world, lexicon)
        return self._facts[key]

    def oracle(self, **sections: Any):
        """The independent oracle over the world run folder, with the same truth settings."""
        config = self.config(**sections)
        return oracle_over(self.folder, z=config.scalar_z)

    def scenes(self, **sections: Any):
        """The scene generator of the same settings."""
        from semantic_world.corpus.histories import SceneGenerator

        return SceneGenerator(self.config(**sections), self.world)

    def lexicon(self, **sections: Any):
        from semantic_world.corpus import Streams, build_lexicon

        config = self.config(**sections)
        return build_lexicon(config, self.world, Streams(config.seed))

    def realizer(self, **sections: Any):
        """The grammar of the same settings: a realizer over the lexicon of those settings."""
        from semantic_world.corpus import Streams
        from semantic_world.corpus.realize import Realizer

        config = self.config(**sections)
        return Realizer(config, self.lexicon(**sections), Streams(config.seed))


# ---------------------------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------------------------


def oracle_over(folder: Path, *, z: float):
    """The independent oracle over a world run folder."""
    from truth_oracle import Oracle

    return Oracle(folder, z=z)


# ---------------------------------------------------------------------------------------------
# The old scene generator's participant draw
# ---------------------------------------------------------------------------------------------


def old_scene_participants(world, settings):
    """A reference of the participant draw of the corpus's scene generator before stage a5a
    (``corpus/scenes.py``), over the world's leaves: the weights over the instances, and the
    draw. Thematic relatedness and taxonomic similarity are the world's, undefined and
    negative similarities counting 0, as the old generator computed them from the taxonomy."""
    leaf = world.entity_leaf
    thematic = world.thematic
    similarity = np.clip(np.nan_to_num(world.leaf_similarity, nan=0.0), 0.0, None)
    weights = settings.participant_weights

    def participant_weights(seed: int) -> np.ndarray:
        values = (
            weights["thematic"] * thematic[leaf, leaf[seed]]
            + weights["taxonomic"] * similarity[leaf, leaf[seed]]
            + weights["constant"]
        )
        values[seed] = 0.0
        return values

    def draw(rng: np.random.Generator, seed: int) -> tuple[int, ...]:
        low, high = settings.size
        size = int(rng.integers(low, high + 1))
        values = participant_weights(seed)
        size = min(size, int(np.count_nonzero(values)))
        if size == 0:
            return (seed,)
        drawn = rng.choice(len(values), size=size, replace=False, p=values / values.sum())
        return (seed,) + tuple(int(i) for i in drawn)

    return participant_weights, draw
