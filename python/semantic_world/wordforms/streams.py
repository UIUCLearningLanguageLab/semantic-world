"""Seeded random streams for the word-form pipeline.

The streams follow the scheme of ``semantic_world.taxonomy.streams``: a stream's seed is SHA-256
of the master seed and the stream's full name, so changing the speakers never changes the word
forms, and changing the embeddings never changes the audio.
"""

from __future__ import annotations

import numpy as np

from semantic_world.taxonomy.streams import stream_seed

STREAM_PREFIX = "wordforms:"

STREAM_NAMES = ("generate", "speakers", "synthesis", "augment", "pca", "train", "assign")
"""The short names of the pipeline's streams. The full name is ``wordforms:<short name>``."""


class Streams:
    """The named streams of one run, each a fresh ``numpy.random.Generator``."""

    generate: np.random.Generator
    speakers: np.random.Generator
    synthesis: np.random.Generator
    augment: np.random.Generator
    pca: np.random.Generator
    train: np.random.Generator
    assign: np.random.Generator

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
        """A fresh generator for one part of a stream, named ``wordforms:<name>:<key>``. Parts
        are independent of each other, so drawing for one engine, or for one token, never changes
        the draws for another."""
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown stream {name!r}; the streams are {', '.join(STREAM_NAMES)}")
        return np.random.default_rng(stream_seed(self.master_seed, f"{STREAM_PREFIX}{name}:{key}"))

    def seeds(self) -> dict[str, int]:
        """Every stream's full name and seed, in stream order."""
        return {STREAM_PREFIX + name: self.seed(name) for name in STREAM_NAMES}
