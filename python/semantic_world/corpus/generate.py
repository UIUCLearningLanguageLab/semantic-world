"""A corpus run: the documents, the test sets, and the statistics.

:func:`generate` makes the corpus of a configuration in memory, in the formal, the conceptual,
and the propositional renderings. The word forms are attached afterwards, by the ``render``
command. ``Corpus.write`` writes the output folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_world.corpus.config import Config
from semantic_world.corpus.planner import Document, Planner
from semantic_world.corpus.stats import corpus_stats
from semantic_world.corpus.testsets import ItemSet, build_test_sets
from semantic_world.corpus.world import World


@dataclass(frozen=True)
class Corpus:
    """Everything a corpus run makes."""

    config: Config
    planner: Planner
    """The planner that made the documents. It holds the world, the lexicon, the scenes, and
    the streams."""
    documents: tuple[Document, ...]
    test_sets: tuple[ItemSet, ...]
    stats: dict[str, Any]
    """``stats.yaml``, as a plain mapping."""

    def write(self, path: str | Path | None = None) -> Path:
        """Write the output folder and return its path. The default is
        ``runs/corpus/<name>_seed<seed>/`` under the current directory."""
        from semantic_world.corpus.io import write_corpus

        return write_corpus(self, path)


def generate(config: Config, world: World | None = None) -> Corpus:
    """Run the generator: the lexicon, the documents with their scenes, the test sets, and the
    statistics. ``world`` is the world, when it is already made."""
    planner = Planner(config, world)
    documents = planner.generate()
    test_sets = build_test_sets(planner, documents)
    stats = corpus_stats(planner, documents, test_sets)
    return Corpus(config, planner, tuple(documents), tuple(test_sets), stats)
