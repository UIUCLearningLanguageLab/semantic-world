"""Seeded random streams of the world generator.

The scheme is the taxonomy's (``semantic_world.taxonomy.streams``): a stream's seed is SHA-256
of the master seed and the stream's full name. The world's streams are ``world:requirements``
(the one-place requirements), ``world:event_tree`` (the event-type features and tree),
``world:constraints`` (the constraints of two-place event types, and the samples of the
capacities' local test), ``world:pairs`` (the sampled pairs and estimates of the relation
statistics), ``world:fluents`` (fluent rules and initial rates), ``world:initial`` (initial
values), ``world:preconditions``, ``world:effects``, ``world:stats`` (the statistics episodes),
and ``world:episodes`` (``simulate``). An event type whose preconditions are redrawn draws from
its own part, ``world:preconditions:<label>``.
"""

from __future__ import annotations

import numpy as np

from semantic_world.taxonomy.streams import stream_generator, stream_seed

STREAM_PREFIX = "world:"
STREAM_NAMES = (
    "requirements",
    "event_tree",
    "constraints",
    "pairs",
    "fluents",
    "initial",
    "preconditions",
    "effects",
    "stats",
    "episodes",
)


class WorldStreams:
    """The named streams of one world run, each a fresh ``numpy.random.Generator``."""

    requirements: np.random.Generator
    event_tree: np.random.Generator
    constraints: np.random.Generator
    pairs: np.random.Generator
    fluents: np.random.Generator
    initial: np.random.Generator
    preconditions: np.random.Generator
    effects: np.random.Generator
    stats: np.random.Generator
    episodes: np.random.Generator

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

    def part(self, name: str, label: str) -> np.random.Generator:
        """A fresh generator for a named part of a stream, ``world:<name>:<label>``."""
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown stream {name!r}; the streams are {', '.join(STREAM_NAMES)}")
        return stream_generator(self.master_seed, f"{STREAM_PREFIX}{name}:{label}")
