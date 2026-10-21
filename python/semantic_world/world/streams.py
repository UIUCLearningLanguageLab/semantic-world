"""Seeded random streams of the world generator.

The scheme is the taxonomy's (``semantic_world.taxonomy.streams``): a stream's seed is SHA-256
of the master seed and the stream's full name. The world's streams are ``world:requirements``
(the one-place requirements), ``world:event_tree`` (the event-type features and tree),
``world:constraints`` (the constraints of two-place event types, and the samples of the
capacities' local test), ``world:pairs`` (the sampled pairs and estimates of the relation
statistics), ``world:fluents`` (fluent rules and initial rates), ``world:initial`` (initial
values), ``world:preconditions``, ``world:effects``, ``world:conditions`` (the conditions of
effects, stage b1: a stream of its own, so that a world with ``conditional_share: 0`` is the same
world as before the stage), the numeric streams of stage b2 (``world:numeric`` for the drift of
the numeric fluents, ``world:numeric_initial`` for their initial values, ``world:numeric_effects``,
``world:numeric_preconditions``, and ``world:numeric_conditions`` for the numeric effects,
precondition literals, and condition literals: each a stream of its own, so that with
``numeric_fluents.count: 0``, or a numeric share of 0, every other draw is unchanged),
``world:stats`` (the statistics episodes), and ``world:episodes`` (``simulate``). An event type
whose preconditions are redrawn draws from its own parts, ``world:preconditions:<label>`` and
``world:numeric_preconditions:<label>``. Stage c1 adds ``world:places`` (the seed of the place
taxonomy run, when the configuration gives none) and ``world:place_event_types`` (the
requirements of the place event types, and the samples of their capacities' local test). Stage
c2a adds ``world:pair_fluents`` (the profiles and initial rates of the pair fluents, drawn in
stage c2b; nothing is drawn from it in c2a, where an event file gives them) and
``world:pair_initial`` (the initial pairs of each pair fluent between entities). Stage c2b adds
``world:pair_preconditions`` (the pair literals that replace precondition literals at
``event_types.preconditions.pair_share``; a never-legal event type's redraw uses its own part,
``world:pair_preconditions:<label>``), ``world:pair_effects`` (the pair effects that replace
effects of two-place event types at ``event_types.effects.pair_share``), and
``world:pair_conditions`` (the pair condition literals of the kind ``pair``).
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
    "conditions",
    "numeric",
    "numeric_initial",
    "numeric_effects",
    "numeric_preconditions",
    "numeric_conditions",
    "stats",
    "episodes",
    "places",
    "place_event_types",
    "pair_fluents",
    "pair_initial",
    "pair_preconditions",
    "pair_effects",
    "pair_conditions",
)
"""The order is the order of creation: the two place streams of stage c1 (``world:places``, the
place taxonomy's seed, and ``world:place_event_types``, the requirements of the place event types
and the samples of their capacities' local test) come after every earlier stream, so that with
``places: null`` every earlier draw is unchanged; the two pair streams of stage c2a
(``world:pair_fluents``, ``world:pair_initial``) come after them, so that a world without pair
fluents is byte-identical to the world of stage c1; and the three pair streams of stage c2b
(``world:pair_preconditions``, ``world:pair_effects``, ``world:pair_conditions``) come last, so
that a world whose configuration draws no pair fluent is byte-identical to the world of stage
c2a. Stage c3 adds ``world:place_dynamics`` and ``world:habitats`` after them."""


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
    conditions: np.random.Generator
    numeric: np.random.Generator
    numeric_initial: np.random.Generator
    numeric_effects: np.random.Generator
    numeric_preconditions: np.random.Generator
    numeric_conditions: np.random.Generator
    stats: np.random.Generator
    episodes: np.random.Generator
    places: np.random.Generator
    place_event_types: np.random.Generator
    pair_fluents: np.random.Generator
    pair_initial: np.random.Generator
    pair_preconditions: np.random.Generator
    pair_effects: np.random.Generator
    pair_conditions: np.random.Generator

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
