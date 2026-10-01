"""Seeded random streams for the corpus generator.

The streams follow the scheme of ``semantic_world.taxonomy.streams``: a stream's seed is SHA-256
of the master seed and the stream's full name. The corpus seed is its own master seed,
independent of the taxonomy's seed. Separate streams mean that changing the grammar settings
never changes the propositions, and changing the test-set settings never changes the documents.
"""

from __future__ import annotations

import numpy as np

from semantic_world.taxonomy.streams import stream_seed

STREAM_PREFIX = "corpus:"

STREAM_NAMES = (
    "lexicon",
    "scenes",
    "documents",
    "propositions",
    "mentions",
    "grammar",
    "tests",
)
"""The short names of the generator's streams. The full name is ``corpus:<short name>``."""


class Streams:
    """The named streams of one run, each a fresh ``numpy.random.Generator``."""

    lexicon: np.random.Generator
    scenes: np.random.Generator
    documents: np.random.Generator
    propositions: np.random.Generator
    mentions: np.random.Generator
    grammar: np.random.Generator
    tests: np.random.Generator

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed
        for name in STREAM_NAMES:
            setattr(self, name, np.random.default_rng(self.seed(name)))

    def seed(self, name: str) -> int:
        """The seed of the stream with the given short name."""
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown stream {name!r}; the streams are {', '.join(STREAM_NAMES)}")
        return stream_seed(self.master_seed, STREAM_PREFIX + name)

    def substream(self, name: str, key: str) -> np.random.Generator:
        """A fresh generator for one part of a stream, named ``corpus:<name>:<key>``. Parts are
        independent of each other, so drawing for one concept type, or for one document, never
        changes the draws for another."""
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown stream {name!r}; the streams are {', '.join(STREAM_NAMES)}")
        return np.random.default_rng(stream_seed(self.master_seed, f"{STREAM_PREFIX}{name}:{key}"))

    def seeds(self) -> dict[str, int]:
        """Every stream's full name and seed, in stream order."""
        return {STREAM_PREFIX + name: self.seed(name) for name in STREAM_NAMES}
