"""Seeded random streams of the world generator.

The scheme is the taxonomy's (``semantic_world.taxonomy.streams``): a stream's seed is SHA-256
of the master seed and the stream's full name. The world's streams are ``world:fluents``
(fluent rules and initial rates), ``world:initial`` (initial values), ``world:preconditions``,
``world:effects``, and ``world:stats``. Event types draw their requirements from the taxonomy's
streams, through the embedded taxonomy run, so a world with fluents off reproduces the
taxonomy's CAN rules and verbs for the same seed.
"""

from __future__ import annotations

import numpy as np

from semantic_world.taxonomy.streams import stream_generator, stream_seed

STREAM_PREFIX = "world:"
STREAM_NAMES = ("fluents", "initial", "preconditions", "effects", "stats")


class WorldStreams:
    """The named streams of one world run, each a fresh ``numpy.random.Generator``."""

    fluents: np.random.Generator
    initial: np.random.Generator
    preconditions: np.random.Generator
    effects: np.random.Generator
    stats: np.random.Generator

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed
        for name in STREAM_NAMES:
            setattr(self, name, stream_generator(master_seed, STREAM_PREFIX + name))

    def seed(self, name: str) -> int:
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown stream {name!r}; the streams are {', '.join(STREAM_NAMES)}")
        return stream_seed(self.master_seed, STREAM_PREFIX + name)

    def seeds(self) -> dict[str, int]:
        return {STREAM_PREFIX + name: self.seed(name) for name in STREAM_NAMES}
