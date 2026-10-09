"""Selection policies: how the events of one step are chosen (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Episodes and selection").

A policy receives what it may see of a step, a :class:`StepContext` (the definition, the state,
the participants, the legal bindings of every event type among them, the event-type weights,
and the transitive share), its random generator, and the number of events to draw. It returns
the events of the step, drawn one at a time among the events that are legal in the state and do
not interfere with the events already chosen. When no such event remains, the step has fewer
events. A policy never changes the state: the episode runner applies the step afterwards.

Policies are registered by name. ``uniform_event``, the default, draws uniformly among all
available events, each weighted by its event type's weight, so an event type that is legal for
many bindings is more frequent than one legal for few. ``uniform_event_type`` is the balanced
policy of the old corpus: the kind of event first, by the transitive share, then the event type
among those with an available event, by weight, then a binding uniformly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from semantic_world.world.definition import RuntimeDefinition
from semantic_world.world.errors import WorldError
from semantic_world.world.runtime import Binding, Event, State, interferes

DEFAULT_POLICY = "uniform_event"


@dataclass(frozen=True)
class StepContext:
    """What a policy sees of one step."""

    definition: RuntimeDefinition
    state: State
    participants: tuple[int, ...]
    legal: Mapping[str, Sequence[Binding]]
    """The legal bindings of every event type that has events (never a category) among the
    participants, in definition order, each list in the runtime's fixed order."""
    weights: Mapping[str, float]
    """Each event type's weight (``scene.event_type_weights``); 1 by default."""
    transitive_share: float

    def events(self) -> list[Event]:
        """Every legal event, in definition order, event types with the weight 0 left out."""
        return [
            Event(label, binding)
            for label, bindings in self.legal.items()
            if self.weights.get(label, 1.0) > 0
            for binding in bindings
        ]

    def available(self, chosen: Sequence[Event]) -> list[Event]:
        """The legal events that do not interfere with any event already chosen."""
        return [
            event
            for event in self.events()
            if all(not interferes(self.definition, event, other) for other in chosen)
        ]


Policy = Callable[[StepContext, np.random.Generator, int], list[Event]]
"""A policy: the context, the generator, and the number of events to draw, to the events."""

_POLICIES: dict[str, Policy] = {}


def register(name: str) -> Callable[[Policy], Policy]:
    """Register a policy under a name, for ``scene.policy``."""

    def wrap(function: Policy) -> Policy:
        if name in _POLICIES:
            raise WorldError(f"a policy named {name!r} is already registered")
        _POLICIES[name] = function
        return function

    return wrap


def policy(name: str) -> Policy:
    try:
        return _POLICIES[name]
    except KeyError:
        raise WorldError(
            f"unknown policy {name!r}; the policies are {', '.join(policy_names())}"
        ) from None


def policy_names() -> tuple[str, ...]:
    return tuple(_POLICIES)


def _weighted_choice(rng: np.random.Generator, weights: Sequence[float]) -> int:
    array = np.asarray(weights, dtype=float)
    return int(rng.choice(len(array), p=array / array.sum()))


@register("uniform_event")
def uniform_event(context: StepContext, rng: np.random.Generator, count: int) -> list[Event]:
    """Draw uniformly among all available events, each weighted by its event type's weight."""
    chosen: list[Event] = []
    for _ in range(count):
        available = context.available(chosen)
        if not available:
            break
        weights = [context.weights.get(event.event_type, 1.0) for event in available]
        chosen.append(available[_weighted_choice(rng, weights)])
    return chosen


@register("uniform_event_type")
def uniform_event_type(context: StepContext, rng: np.random.Generator, count: int) -> list[Event]:
    """The balanced policy: the kind of event by the transitive share (when both kinds have an
    available event), then the event type by weight among those with an available event, then a
    binding uniformly among that event type's available events."""
    share = context.transitive_share
    chosen: list[Event] = []
    for _ in range(count):
        available = context.available(chosen)
        by_kind: dict[int, dict[str, list[Event]]] = {1: {}, 2: {}}
        for event in available:
            by_kind[len(event.binding)].setdefault(event.event_type, []).append(event)
        one_place = bool(by_kind[1]) and share < 1
        two_place = bool(by_kind[2]) and share > 0
        if not (one_place or two_place):
            break
        kind = 2 if two_place and (not one_place or rng.random() < share) else 1
        labels = list(by_kind[kind])
        label = labels[_weighted_choice(rng, [context.weights.get(v, 1.0) for v in labels])]
        options = by_kind[kind][label]
        chosen.append(options[int(rng.integers(len(options)))])
    return chosen


__all__ = [
    "DEFAULT_POLICY",
    "Policy",
    "StepContext",
    "policy",
    "policy_names",
    "register",
    "uniform_event",
    "uniform_event_type",
]
