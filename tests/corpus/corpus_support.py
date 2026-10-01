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
EXAMPLE_TAXONOMIES = {"tiny": TINY_TAXONOMY, "default": DEFAULT_TAXONOMY}


class Case:
    """One world for the proposition tests: its taxonomy, the taxonomy's output folder, and the
    facts and the independent oracle under any corpus settings."""

    def __init__(self, name: str, taxonomy: str, folder: Path) -> None:
        from semantic_world.corpus import load_taxonomy

        self.name = name
        self.taxonomy = taxonomy
        self.result = load_taxonomy(self.config())
        self.folder = self.result.write(folder)
        self._facts: dict[str, Any] = {}

    def config(self, **sections: Any) -> Config:
        return corpus_config(self.taxonomy, **sections)

    def facts(self, **sections: Any):
        """The facts of the world, with the lexicon of the same settings. Cached by settings."""
        from semantic_world.corpus import Streams, build_lexicon
        from semantic_world.corpus.facts import Facts

        key = repr(sorted(sections.items()))
        if key not in self._facts:
            config = self.config(**sections)
            lexicon = build_lexicon(config, self.result, Streams(config.seed))
            self._facts[key] = Facts(config, self.result, lexicon)
        return self._facts[key]

    def oracle(self, **sections: Any):
        """The independent oracle over the output folder, with the same truth settings."""
        from truth_oracle import Oracle

        config = self.config(**sections)
        return Oracle(
            self.folder,
            z=config.scalar_z,
            most=config.quantifiers.most_min_proportion,
            all_grounding=config.quantifiers.all_grounding,
            generic=config.quantifiers.generic_means,
        )

    def scenes(self, **sections: Any):
        """The scene generator of the same settings, sharing the truth tests of the facts."""
        from semantic_world.corpus.scenes import SceneGenerator

        return SceneGenerator(self.config(**sections), self.result, self.facts(**sections).truth)

    def lexicon(self, **sections: Any):
        from semantic_world.corpus import Streams, build_lexicon

        config = self.config(**sections)
        return build_lexicon(config, self.result, Streams(config.seed))

    def realizer(self, **sections: Any):
        """The grammar of the same settings: a realizer over the lexicon of those settings."""
        from semantic_world.corpus import Streams
        from semantic_world.corpus.realize import Realizer

        config = self.config(**sections)
        return Realizer(config, self.lexicon(**sections), Streams(config.seed))
