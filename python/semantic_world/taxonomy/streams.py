"""Seeded random streams for the taxonomy generator.

The scheme follows the determinism section of ``docs/specs/MILESTONE_1.md``, implemented in
Python. A stream's seed is SHA-256 of the master seed (8 bytes, little-endian) followed by the
stream name (UTF-8). The 32-byte digest, read as a little-endian integer, seeds
``numpy.random.default_rng``. Adding a new stream never changes an existing stream, and each
stream is independent of the others, so changing the instance count never changes the rules
or the tree.
"""

from __future__ import annotations

import hashlib

import numpy as np

from semantic_world.taxonomy.config import SEED_MAX

STREAM_PREFIX = "taxonomy:"

STREAM_NAMES = (
    "base_rates",
    "rules",
    "superordinates",
    "tree",
    "instances",
    "analysis",
    "scalars",
    "scalar_instances",
    "verb_tree",
    "constraints",
    "pairs",
)
"""The short names of the generator's streams. The full name is ``taxonomy:<short name>``."""


def stream_seed(master_seed: int, name: str) -> int:
    """The seed of the stream ``name`` (a full name such as ``taxonomy:rules``) as an integer."""
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError("the master seed must be an integer")
    if not 0 <= master_seed <= SEED_MAX:
        raise ValueError(f"the master seed must be a 64-bit unsigned integer, got {master_seed}")
    digest = hashlib.sha256(master_seed.to_bytes(8, "little") + name.encode("utf-8")).digest()
    return int.from_bytes(digest, "little")


def stream_generator(master_seed: int, name: str) -> np.random.Generator:
    """A fresh NumPy generator for the stream ``name`` (a full name such as ``taxonomy:rules``)."""
    return np.random.default_rng(stream_seed(master_seed, name))


class Streams:
    """The six named streams of one run, each a fresh ``numpy.random.Generator``.

    Every attribute is created once, when the object is made, so a stage that draws from
    ``streams.tree`` continues where the previous stage left off.
    """

    base_rates: np.random.Generator
    rules: np.random.Generator
    superordinates: np.random.Generator
    tree: np.random.Generator
    instances: np.random.Generator
    analysis: np.random.Generator
    scalars: np.random.Generator
    scalar_instances: np.random.Generator
    verb_tree: np.random.Generator
    constraints: np.random.Generator
    pairs: np.random.Generator

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed
        for name in STREAM_NAMES:
            setattr(self, name, stream_generator(master_seed, STREAM_PREFIX + name))

    def seed(self, name: str) -> int:
        """The seed of the stream with the given short name."""
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown stream {name!r}; the streams are {', '.join(STREAM_NAMES)}")
        return stream_seed(self.master_seed, STREAM_PREFIX + name)

    def seeds(self) -> dict[str, int]:
        """Every stream's full name and seed, in stream order."""
        return {STREAM_PREFIX + name: self.seed(name) for name in STREAM_NAMES}
