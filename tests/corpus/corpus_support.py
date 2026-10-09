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
event_types: {binary: null}
"""

STATIC_WORLD = """\
# The tiny relations taxonomy with no fluents: every able binding is legal at every time point,
# as in the corpus's old scene generator.
name: static
seed: 1
taxonomy: {config: data/taxonomy/tiny_relations.yaml}
fluents: {count: 0}
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
# A small world whose rules chain: some determined features are computed from other determined
# features, and some rules read scalar thresholds.
name: deep
seed: 3
features:
  is: {count: 12, proportion_determined: 0.5, expected_true_free: 2.5}
  has: {count: 12, proportion_determined: 0.5, expected_true_free: 2.5}
  can: {count: 6}
rules: {max_chain_depth: 2}
taxonomy: {superordinates: 3, depth: 2, branching: [2, 3]}
instances: {per_leaf: [4, 6]}
scalars: {count: 2}
verbs:
  features: {count: 6, expected_true: 2}
  taxonomy: {superordinates: 2, depth: 2, branching: 2}
"""

STILL_TAXONOMY = """\
# A small world whose scalar never drifts, so a threshold literal is fixed at every category,
# and whose verb tree is flat.
name: still
seed: 2
features:
  is: {count: 8, proportion_determined: 0.5, expected_true_free: 2}
  has: {count: 8, proportion_determined: 0.5, expected_true_free: 2}
  can: {count: 6}
rules: {input_type_weights: {is: 1, has: 1, scalar: 3}}
taxonomy: {superordinates: 3, depth: 2, branching: 2}
instances: {per_leaf: 4}
scalars: {count: 1, drift: 0, instance_drift: 0}
verbs:
  features: {count: 4, expected_true: 2}
  taxonomy: {superordinates: 2, depth: 1, branching: 2}
"""

WRITTEN_TAXONOMIES = {"deep": DEEP_TAXONOMY, "still": STILL_TAXONOMY}
EXAMPLE_WORLDS = {"tiny": TINY_WORLD, "default": DEFAULT_WORLD}


def world_over(name: str, taxonomy_path: str, seed: int) -> str:
    """A world configuration over a written taxonomy, with the default fluents and event
    types."""
    return f"name: {name}\nseed: {seed}\ntaxonomy: {{config: {taxonomy_path}, seed: {seed}}}\n"


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
        from truth_oracle import Oracle

        config = self.config(**sections)
        return Oracle(self.folder, z=config.scalar_z)

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
