"""Scenes: the corpus's view of the world's episodes.

Scenes are episodes of the world package (``semantic_world.world.episodes``), recorded as
histories (``semantic_world.world.history``): participants drawn around a seed instance, a
drawn number of steps, and at each step the events the selection policy chose among those that
were legal in the state, with the base fluents each event changed. Events change state, so
what is legal differs from step to step, and an episode ends early when nothing is legal.

Each scene draws from its own part of the ``corpus:scenes`` stream, named by its label
(``SCENE.<n>``), with the seed instance the planner chose, so a scene depends only on the
corpus seed, its number, its seed instance, and the scene settings. Scenes are a fact about the
world, not about the language: the generator uses every event type, with a word or without one.

This module gives the planner and the truth tests what they read of a history: its events in
time order, each with its step and its scene, what happened, and what involves an entity. Since
stage c4a a history may hold place events (``SceneEvent.place``) and pair changes; the readers
that name an event's event type or read a change's fluent skip them, because the corpus says
nothing about pairs and places until stages c4b and c4c.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_world.corpus.config import Config, ConfigError
from semantic_world.corpus.streams import Streams
from semantic_world.corpus.world import PLACE_EVENT_PREFIX, World
from semantic_world.world.episodes import EpisodeGenerator, EpisodeSettings
from semantic_world.world.history import History, HistoryEvent

SCENE_PREFIX = "SCENE."
EVENT_INFIX = ".EVENTINSTANCE."
TIME_PREFIX = "TIME."
TIME_INFIX = ".TIME."


def scene_label(number: int) -> str:
    return f"{SCENE_PREFIX}{number}"


def scene_of(label: str) -> str:
    """The scene that an event label (``SCENE.8.EVENTINSTANCE.5``) or a scene label
    (``SCENE.8``) names."""
    return label.split(EVENT_INFIX)[0]


def event_of(label: str) -> str | None:
    """The event that a label names: the label itself for an event label, and None for a scene
    label, which stands for some event of the scene."""
    return label if EVENT_INFIX in label else None


def is_event_label(label: str) -> bool:
    return label.startswith(SCENE_PREFIX) and EVENT_INFIX in label


def is_scene_label(label: str) -> bool:
    return label.startswith(SCENE_PREFIX) and EVENT_INFIX not in label and TIME_INFIX not in label


def time_label(index: int) -> str:
    """``TIME.<k>``: the time point before step k (``TIME.1`` is the start of a scene)."""
    return f"{TIME_PREFIX}{index}"


def time_index(label: str) -> int:
    """The number of a time point: 2 for ``TIME.2``."""
    if not label.startswith(TIME_PREFIX) or not label[len(TIME_PREFIX) :].isdecimal():
        raise ValueError(f"{label!r} is not a time point (TIME.<k>)")
    return int(label[len(TIME_PREFIX) :])


def time_key(scene: str, time: str) -> str:
    """A time point qualified by its scene, ``SCENE.8.TIME.2``: what the record of a sentence
    holds for a verb phrase about a state, a change, or what was possible at that time."""
    return f"{scene}.{time}"


def split_time_key(key: str) -> tuple[str, str]:
    """The scene and the time point of a qualified time point."""
    scene, infix, number = key.partition(TIME_INFIX)
    if not infix or not scene.startswith(SCENE_PREFIX) or not number.isdecimal():
        raise ValueError(f"{key!r} is not a qualified time point (SCENE.<n>.TIME.<k>)")
    return scene, f"{TIME_PREFIX}{number}"


def is_time_key(label: str) -> bool:
    return label.startswith(SCENE_PREFIX) and TIME_INFIX in label


@dataclass(frozen=True)
class SceneEvent:
    """One event of a history, as the corpus reads it."""

    label: str
    """``SCENE.<n>.EVENTINSTANCE.<k>``: event k of its scene, in time order."""
    scene: str
    step: int
    """The step the event belongs to, from 1: it takes ``TIME.<step>`` to ``TIME.<step + 1>``."""
    type: str
    """The event type: a one-place event type, a two-place event type (never a category), or,
    since stage c3b, a place event type, whose patient is a place."""
    agent: str
    patient: str | None = None

    @property
    def transitive(self) -> bool:
        return self.patient is not None

    @property
    def place(self) -> bool:
        """Whether the event is of a place event type (a move or an act at a place, stage c3b),
        whose patient is a place: never narrated in stage c4a."""
        return self.type.startswith(PLACE_EVENT_PREFIX)

    @property
    def key(self) -> tuple[str, str, str | None]:
        """What happened, apart from when: the event type, the agent, and the patient."""
        return (self.type, self.agent, self.patient)

    def involves(self, instance: str) -> bool:
        return instance in (self.agent, self.patient)


def scene_events(history: History) -> tuple[SceneEvent, ...]:
    """Every event of a history, in time order, and within a step in the order it was drawn."""
    return tuple(
        _scene_event(history.label, step.step, event)
        for step in history.steps
        for event in step.events
    )


def _scene_event(scene: str, step: int, event: HistoryEvent) -> SceneEvent:
    return SceneEvent(event.label, scene, step, event.type, event.agent, event.patient)


def events_at(history: History, step: int) -> tuple[SceneEvent, ...]:
    return tuple(
        _scene_event(history.label, s.step, e)
        for s in history.steps
        if s.step == step
        for e in s.events
    )


def involving(history: History, instance: str) -> tuple[SceneEvent, ...]:
    """The events an instance takes part in, as agent or patient, in time order."""
    return tuple(e for e in scene_events(history) if e.involves(instance))


def happened_in(history: History, event_type: str, agent: str, patient: str | None) -> bool:
    return any(e.key == (event_type, agent, patient) for e in scene_events(history))


# ---------------------------------------------------------------------------------------------
# The episode generator of a corpus
# ---------------------------------------------------------------------------------------------


def episode_settings(config: Config, world: World) -> EpisodeSettings:
    """The world package's episode settings from the corpus's ``scene`` block. The event-type
    weights are checked against the world: a label that is not an event type with events (a
    category of event types names no event of its own) is an error that names the field."""
    scene = config.scene
    weights = dict(scene.event_type_weights or {})
    for label in weights:
        if label not in world.event_types or world.event_types[label].category:
            raise ConfigError(
                config.source,
                f"scene.event_type_weights.{label}",
                "is not an event type of the world (a category of event types has no events)",
            )
    return EpisodeSettings(
        size=(scene.size.min, scene.size.max),
        steps=(scene.steps.min, scene.steps.max),
        events_per_step=scene.events_per_step,
        transitive_share=scene.transitive_share,
        participant_weights=dict(scene.participant_weights),
        event_type_weights=weights,
        policy=scene.policy,
        initial=scene.initial,
        extra_places=scene.extra_places,
        places=scene.places,
        initial_location=scene.initial_location,
    )


class SceneGenerator:
    """The scenes of one world under one corpus configuration: the world package's episode
    generator, driven from the corpus's ``corpus:scenes`` stream."""

    def __init__(self, config: Config, world: World) -> None:
        self.world = world
        self.settings = episode_settings(config, world)
        places = world.result.config.places
        self.generator = EpisodeGenerator(
            world.definition,
            world.relatedness,
            self.settings,
            world.initial_rates,
            record_legal=False,
            habitat_strength=1.0 if places is None else places.habitat_strength,
        )

    def participant_weights(self, seed: str) -> np.ndarray:
        """The weight of every instance as a participant of a scene seeded at ``seed``."""
        return self.generator.participant_weights(self.world.instance_index[seed])

    def generate(self, rng: np.random.Generator, seed: str, label: str) -> History:
        """One scene seeded at an instance, drawn from ``rng``: its participants, its number of
        steps, its initial state, and its steps."""
        if seed not in self.world.instance_index:
            raise KeyError(f"unknown instance {seed!r}")
        return self.generator.episode(rng, label, self.world.instance_index[seed])

    def scene(self, streams: Streams, number: int, seed: str, attempt: int = 0) -> History:
        """Scene ``SCENE.<number>``, seeded at an instance, from its own part of the
        ``corpus:scenes`` stream; a scene drawn again for a document that had nothing to say
        (``attempt`` 1 and up, stage c4a) comes from a further part named by the attempt, so
        that the redraw does not repeat the scene."""
        label = scene_label(number)
        key = label if attempt == 0 else f"{label}.REDRAW.{attempt}"
        return self.generate(streams.substream("scenes", key), seed, label)

    def relatedness(self, history: History) -> tuple[float, float]:
        """The mean thematic relatedness, and the mean taxonomic similarity, of the seed's leaf
        and the leaf of every other participant. NaN for a scene with no other participant."""
        world = self.world
        seed_leaf = world.entity_leaf[world.instance_index[history.seed]]
        others = [world.entity_leaf[world.instance_index[p]] for p in history.participants[1:]]
        if not others:
            return float("nan"), float("nan")
        return (
            float(world.thematic[others, seed_leaf].mean()),
            float(world.relatedness.similarity[others, seed_leaf].mean()),
        )


__all__ = [
    "EVENT_INFIX",
    "SCENE_PREFIX",
    "SceneEvent",
    "SceneGenerator",
    "episode_settings",
    "event_of",
    "events_at",
    "happened_in",
    "involving",
    "is_event_label",
    "is_scene_label",
    "is_time_key",
    "scene_events",
    "scene_label",
    "scene_of",
    "split_time_key",
    "time_index",
    "time_key",
    "time_label",
]
