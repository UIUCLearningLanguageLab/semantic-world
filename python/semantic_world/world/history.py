"""Histories: the record of an episode (``docs/specs/WORLD_AND_LANGUAGE.md``, "Histories").

A history is one JSON object per episode: its label, seed instance, participants, policy, and
rule-set identity; the base fluents true at ``TIME.1``; each step's events with the base fluents
each event changed; the last time point; and whether the episode ended in quiescence. The
corpus writes histories to ``scenes.jsonl`` and ``simulate`` to ``episodes.jsonl``, with the same
schema, which the 3D engine will write too. Phase (a) records Boolean changes (``to: true`` or
``false``); phase (b) adds numeric changes with their causes, in the same ``changes`` list.

``replay`` runs a history's events through the runtime from its initial state and checks that
every step is legal and free of interference, and that the recorded changes are exactly the
base fluents that changed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_world.world.definition import RuntimeDefinition
from semantic_world.world.errors import WorldError
from semantic_world.world.runtime import Event, State, apply

EPISODES_FILE = "episodes.jsonl"


class HistoryError(WorldError):
    """A history does not replay: an event is illegal or interferes, or the recorded changes are
    not what the runtime computes."""


@dataclass(frozen=True)
class Change:
    """One base fluent of one entity that an event changed."""

    entity: str
    fluent: str
    to: bool

    def to_json(self) -> dict[str, Any]:
        return {"entity": self.entity, "fluent": self.fluent, "to": self.to}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Change:
        return cls(str(data["entity"]), str(data["fluent"]), bool(data["to"]))


@dataclass(frozen=True)
class HistoryEvent:
    label: str
    """``SCENE.<n>.EVENTINSTANCE.<k>``: event k of its episode, in time order."""
    type: str
    agent: str
    patient: str | None = None
    changes: tuple[Change, ...] = ()

    @property
    def binding(self) -> dict[str, str]:
        roles = {"agent": self.agent}
        if self.patient is not None:
            roles["patient"] = self.patient
        return roles

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"label": self.label, "type": self.type, "agent": self.agent}
        if self.patient is not None:
            data["patient"] = self.patient
        data["changes"] = [change.to_json() for change in self.changes]
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> HistoryEvent:
        return cls(
            str(data["label"]),
            str(data["type"]),
            str(data["agent"]),
            None if data.get("patient") is None else str(data["patient"]),
            tuple(Change.from_json(c) for c in data.get("changes", [])),
        )


@dataclass(frozen=True)
class Step:
    step: int
    """Counted from 1: step k takes ``TIME.k`` to ``TIME.k+1``."""
    events: tuple[HistoryEvent, ...]
    legal: dict[str, int] | None = None
    """With ``--legal``: the number of legal bindings of each event type among the participants
    in the state before the step."""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"step": self.step, "events": [e.to_json() for e in self.events]}
        if self.legal is not None:
            data["legal"] = dict(self.legal)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Step:
        legal = data.get("legal")
        return cls(
            int(data["step"]),
            tuple(HistoryEvent.from_json(e) for e in data["events"]),
            None if legal is None else {str(k): int(v) for k, v in legal.items()},
        )


@dataclass(frozen=True)
class History:
    label: str
    """``SCENE.<n>``."""
    seed: str
    participants: tuple[str, ...]
    """The seed instance, then the other participants in the order they were drawn."""
    policy: str
    rule_set_id: str
    initial: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """For each participant, the base fluents true at ``TIME.1``."""
    steps: tuple[Step, ...] = ()
    final: str = "TIME.1"
    """The last time point."""
    quiescent: bool = False

    @property
    def events(self) -> tuple[HistoryEvent, ...]:
        return tuple(e for step in self.steps for e in step.events)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seed": self.seed,
            "participants": list(self.participants),
            "policy": self.policy,
            "rule_set_id": self.rule_set_id,
            "initial": {p: list(self.initial[p]) for p in self.participants},
            "steps": [step.to_json() for step in self.steps],
            "final": self.final,
            "quiescent": self.quiescent,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> History:
        return cls(
            str(data["label"]),
            str(data["seed"]),
            tuple(str(p) for p in data["participants"]),
            str(data["policy"]),
            str(data["rule_set_id"]),
            {str(p): tuple(str(f) for f in fluents) for p, fluents in data["initial"].items()},
            tuple(Step.from_json(s) for s in data["steps"]),
            str(data["final"]),
            bool(data["quiescent"]),
        )


def write_histories(path: str | Path, histories: Iterable[History]) -> Path:
    """Write histories as JSON lines, one episode per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for history in histories:
            handle.write(json.dumps(history.to_json(), ensure_ascii=True) + "\n")
    return path


def read_histories(path: str | Path) -> list[History]:
    with Path(path).open(encoding="utf-8") as handle:
        return [History.from_json(json.loads(line)) for line in handle if line.strip()]


# ---------------------------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------------------------


def initial_state_of(definition: RuntimeDefinition, history: History) -> State:
    """The state at ``TIME.1`` of a history over every entity of the definition: the history's
    initial fluents for its participants, and the definition's initial values for the rest (all
    false when the definition carries none)."""
    from semantic_world.world.runtime import initial_state

    if definition.initial_values is None:
        true: dict[str, list[str]] = {label: [] for label in definition.entity_labels}
    else:
        true = initial_state(definition).true_fluents(definition)
    for participant, fluents in history.initial.items():
        true[participant] = list(fluents)
    return State.from_true(definition, true)


def changes_of(
    definition: RuntimeDefinition, before: State, after: State, entities: Sequence[int]
) -> list[Change]:
    """The base fluents of the given entities whose value differs between two states, in entity
    and then fluent order."""
    found = []
    for entity in entities:
        for j, fluent in enumerate(definition.base_fluents):
            if before.values[entity, j] != after.values[entity, j]:
                found.append(
                    Change(definition.entity_labels[entity], fluent, bool(after.values[entity, j]))
                )
    return found


def replay(definition: RuntimeDefinition, history: History) -> list[State]:
    """Run the history's events through ``apply`` from its initial state. Returns the state at
    every time point. Raises :class:`HistoryError` when a step does not apply, or when an
    event's recorded changes are not exactly what its effects changed."""
    if history.rule_set_id != definition.rule_set_id:
        raise HistoryError(
            f"{history.label} was made with rule set {history.rule_set_id}, but the definition "
            f"is {definition.rule_set_id}"
        )
    state = initial_state_of(definition, history)
    states = [state]
    for step in history.steps:
        events = [Event.from_labels(definition, e.type, e.binding) for e in step.events]
        try:
            after = apply(definition, state, events)
        except WorldError as error:
            raise HistoryError(f"{history.label}, step {step.step}: {error}") from None
        recorded = [c for e in step.events for c in e.changes]
        found = changes_of(
            definition, state, after, sorted(set(i for e in events for i in e.binding))
        )
        if sorted(recorded, key=_change_key) != sorted(found, key=_change_key):
            raise HistoryError(
                f"{history.label}, step {step.step}: the recorded changes are {recorded}, but "
                f"the events change {found}"
            )
        for recorded_event, event in zip(step.events, events, strict=True):
            et = definition.event_type(event.event_type)
            written = {
                (definition.entity_labels[event.binding[et.roles.index(e.role)]], e.fluent)
                for e in et.effects
            }
            for change in recorded_event.changes:
                if (change.entity, change.fluent) not in written:
                    raise HistoryError(
                        f"{history.label}, step {step.step}: {recorded_event.label} records a "
                        f"change to {change.fluent} of {change.entity}, which it does not write"
                    )
        state = after
        states.append(state)
    if history.final != f"TIME.{len(history.steps) + 1}":
        raise HistoryError(
            f"{history.label} has {len(history.steps)} steps but ends at {history.final}"
        )
    return states


def _change_key(change: Change) -> tuple[str, str, bool]:
    return (change.entity, change.fluent, change.to)


__all__ = [
    "EPISODES_FILE",
    "Change",
    "History",
    "HistoryError",
    "HistoryEvent",
    "Step",
    "changes_of",
    "initial_state_of",
    "read_histories",
    "replay",
    "write_histories",
]
