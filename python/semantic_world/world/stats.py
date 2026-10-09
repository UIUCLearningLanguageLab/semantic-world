"""``world_stats.yaml`` (``docs/specs/WORLD_AND_LANGUAGE.md``, "World statistics").

This stage reports the structural statistics: counts of fluents, event types, precondition
literals, and effects by kind and role; the effects and literals dropped or redrawn by the
fix-ups; the enabling graph (an edge from A to B when an effect of A produces a value that a
precondition literal of B requires), its edge count and longest chain; and the absorbing
fluents. The episode statistics come from 1,000 episodes of the default policy on the
``world:stats`` stream: for each event type, the share of steps at which it had a legal binding
among the participants and the share at which it occurred; the mean number of changes per event;
the share of quiescent episodes; the number of times each event type's preconditions were
redrawn because it was never legal (``precondition_redraws``); and, for each event type still
never legal, why: ``never_able`` (no entity, or no ordered pair, meets its requirement) or
``preconditions`` (its preconditions were never met in the episodes, after the redraws).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from semantic_world.world.event_types import EventTypes
from semantic_world.world.fluents import Fluents
from semantic_world.world.history import History


def enabling_edges(event_types: EventTypes) -> list[tuple[str, str]]:
    """Every edge (A, B) of the enabling graph over the event types that can occur."""
    leaves = event_types.leaves
    produced = {et.label: {(e.fluent, e.value) for e in et.effects} for et in leaves}
    edges = []
    for a in leaves:
        for b in leaves:
            if any((lit.fluent, lit.value) in produced[a.label] for lit in b.precondition):
                edges.append((a.label, b.label))
    return edges


def longest_chain(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    """The longest path (in edges) through the graph's strongly connected components, with a
    component of several nodes counting as one step; a cyclic graph therefore reports a finite
    length."""
    successors: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        successors[a].add(b)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    component: dict[str, int] = {}
    counter = [0]
    components = [0]

    def connect(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(successors[v]):
            if w not in index:
                connect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component[w] = components[0]
                if w == v:
                    break
            components[0] += 1

    for node in nodes:
        if node not in index:
            connect(node)
    dag: dict[int, set[int]] = {c: set() for c in range(components[0])}
    for a, b in edges:
        if component[a] != component[b]:
            dag[component[a]].add(component[b])
    memo: dict[int, int] = {}

    def depth(c: int) -> int:
        if c not in memo:
            memo[c] = max((1 + depth(d) for d in dag[c]), default=0)
        return memo[c]

    return max((depth(c) for c in dag), default=0)


def absorbing_fluents(event_types: EventTypes, fluents: Fluents) -> list[str]:
    """Base fluents that some effect sets to one value and no effect sets back."""
    set_to: dict[str, set[bool]] = {f.label: set() for f in fluents.base}
    for et in event_types.leaves:
        for effect in et.effects:
            set_to[effect.fluent].add(effect.value)
    return [label for label, values in set_to.items() if len(values) == 1]


def episode_stats(
    histories: Sequence[History],
    event_types: Sequence[str],
    redraws: Mapping[str, int] | None = None,
    never_able: Sequence[str] = (),
) -> dict[str, Any]:
    """The statistics of a set of histories with legal counts recorded: per event type, the share
    of steps with a legal binding and the share at which it occurred; the mean number of changes
    per event; the share of quiescent episodes; the precondition redraws; and the event types
    never legal, each with its reason."""
    steps = [step for history in histories for step in history.steps]
    total = len(steps)
    legal_steps = dict.fromkeys(event_types, 0)
    occurred_steps = dict.fromkeys(event_types, 0)
    for step in steps:
        for label, count in (step.legal or {}).items():
            if count:
                legal_steps[label] += 1
        for label in {e.type for e in step.events}:
            occurred_steps[label] += 1
    events = [e for step in steps for e in step.events]
    changes = sum(len(e.changes) for e in events)
    never = [label for label in event_types if legal_steps[label] == 0]
    quiescent = sum(1 for h in histories if h.quiescent)
    reasons = {label: "never_able" if label in never_able else "preconditions" for label in never}
    return {
        "count": len(histories),
        "steps": total,
        "events": len(events),
        "legal_share": {
            label: round(legal_steps[label] / total, 6) if total else 0.0 for label in event_types
        },
        "occurrence_share": {
            label: round(occurred_steps[label] / total, 6) if total else 0.0
            for label in event_types
        },
        "mean_changes_per_event": round(changes / len(events), 6) if events else 0.0,
        "quiescent_share": round(quiescent / len(histories), 6) if histories else 0.0,
        "two_place_share": round(sum(1 for e in events if e.patient is not None) / len(events), 6)
        if events
        else 0.0,
        "precondition_redraws": dict(redraws or {}),
        "never_legal": reasons,
        "warnings": [
            f"{label} was never legal in any episode ({reason})"
            for label, reason in reasons.items()
        ],
    }


def world_stats(
    fluents: Fluents, event_types: EventTypes, episodes: dict[str, Any] | None = None
) -> dict[str, Any]:
    leaves = event_types.leaves
    by_kind = {
        "one_place": sum(1 for et in leaves if et.arity == 1),
        "two_place": sum(1 for et in leaves if et.arity == 2),
        "two_place_categories": len(event_types.categories),
    }
    literals_by_role = {"agent": 0, "patient": 0}
    effects_by_role = {"agent": 0, "patient": 0}
    literals_by_kind = {"one_place": 0, "two_place": 0}
    effects_by_kind = {"one_place": 0, "two_place": 0}
    for et in leaves:
        kind = "one_place" if et.arity == 1 else "two_place"
        for lit in et.precondition:
            literals_by_role[lit.role] += 1
            literals_by_kind[kind] += 1
        for effect in et.effects:
            effects_by_role[effect.role] += 1
            effects_by_kind[kind] += 1
    edges = enabling_edges(event_types)
    return {
        "fluents": {"base": len(fluents.base), "derived": len(fluents.derived)},
        "event_types": by_kind,
        "precondition_literals": {
            "total": sum(literals_by_kind.values()),
            "by_kind": literals_by_kind,
            "by_role": literals_by_role,
            "from_features": sum(
                1 for v in event_types.feature_preconditions.values() if v is not None
            ),
        },
        "effects": {
            "total": sum(effects_by_kind.values()),
            "by_kind": effects_by_kind,
            "by_role": effects_by_role,
            "from_features": sum(1 for v in event_types.feature_effects.values() if v is not None),
        },
        "fix_ups": dict(event_types.stats),
        "enabling_graph": {
            "edges": len(edges),
            "longest_chain": longest_chain([et.label for et in leaves], edges),
        },
        "absorbing_fluents": absorbing_fluents(event_types, fluents),
        "episodes": episodes,
    }
