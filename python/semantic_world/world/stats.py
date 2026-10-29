"""``world_stats.yaml`` (``docs/specs/WORLD_AND_LANGUAGE.md``, "World statistics").

This stage reports the structural statistics: counts of fluents, event types, precondition
literals, and effects by kind and role, with the conditional effects and their literals (stage
b1); the effects, literals, and conditions dropped or redrawn by the fix-ups; the enabling graph
(an edge from A to B when an effect of A produces a value that a precondition literal of B
requires), its edge count and longest chain; and the absorbing fluents. The episode statistics
come from 1,000 episodes of the default policy on the ``world:stats`` stream: for each event
type, the share of steps at which it had a legal binding among the participants and the share at
which it occurred; the mean number of changes per event; the share of quiescent episodes; the
number of times each event type's preconditions were redrawn because it was never legal
(``precondition_redraws``); and, for each event type still never legal, why: ``never_able`` (no
entity, or no ordered pair, meets its requirement) or
``preconditions`` (its preconditions were never met in the episodes, after the redraws).
The ``relations`` block reports what the taxonomy's ``summary.yaml`` reported about the
two-place event types before stage a5b: the counts of event-type features, categories, event
types, and constraints, the constraints by family, the share of approximate projections, whether
the relation proportions were estimated from a sample of pairs, the event types with fewer sampled
pairs than requested, and the event types the generator could not bring inside the density range
(stage a6, Jon's ruling 6 on the a5b questions). It is null in a world without two-place event
types. The ``pair_fluents`` block (stage c2b of ``docs/specs/WORLD_PHASE_C.md``, "World
statistics") reports the pair fluents by profile, their initial rates and initial pairs, the
pair literals of preconditions by kind, role, polarity, and kind of event type, the compiled
literals by origin, the pair effects by value, the pair condition literals, the derived rules
that read a pair literal, the pair draws and redraws, and the episodes' pair changes per event;
it is written only when the world has pair fluents, so that the statistics of every other world
are as before. In a world with location fluents (stage c3b) the ``places`` block gains the place
event types by kind, the counts of the drawn dynamics, and the concentration of habitats; the
``pair_fluents`` block counts the location fluents apart and the place event types' literals
and effects; and the episodes report the move share, the act share, and the two-place share
counting the place event types apart; every such key is written only with location fluents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from semantic_world.world.event_types import EventTypes
from semantic_world.world.fluents import Fluents
from semantic_world.world.history import History
from semantic_world.world.numeric import NumericFluents
from semantic_world.world.pair_fluents import PairFluents
from semantic_world.world.places import Places, places_stats

if TYPE_CHECKING:
    from semantic_world.world.statics import StaticWorld


def _produces(effect) -> tuple[str, Any]:
    """What an effect produces, as a precondition literal would require it: ``(fluent, value)``
    for a Boolean effect, ``(fluent, True)`` for a numeric effect that adds (it can bring a
    value above a threshold) and ``(fluent, False)`` for one that subtracts."""
    if effect.numeric:
        return (effect.fluent, effect.amount > 0)
    return (effect.fluent, effect.value)


def enabling_edges(event_types: EventTypes) -> list[tuple[str, str]]:
    """Every edge (A, B) of the enabling graph over the event types that can occur."""
    leaves = event_types.leaves
    produced = {et.label: {_produces(e) for e in et.effects} for et in leaves}
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
            if not effect.numeric and not effect.pair:
                set_to[effect.fluent].add(effect.value)
    return [label for label, values in set_to.items() if len(values) == 1]


def episode_stats(
    histories: Sequence[History],
    event_types: Sequence[str],
    redraws: Mapping[str, int] | None = None,
    never_able: Sequence[str] = (),
    place_kinds: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """The statistics of a set of histories with legal counts recorded: per event type, the share
    of steps with a legal binding and the share at which it occurred; the mean number of changes
    per event; the share of quiescent episodes; the precondition redraws; and the event types
    never legal, each with its reason. With ``place_kinds`` (the kind, ``move`` or ``act``, of
    every performable place event type; stage c3b, given only in a world with location
    fluents) also the share of events that are moves (``move_share``), the share that are acts
    (``act_share``), and the two-place share counting the place event types apart
    (``two_place_share_entities``: the share of events of two-place event types between
    entities), beside ``two_place_share``, which counts every event with a patient."""
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
    changes = sum(1 for e in events for c in e.changes if not c.numeric and not c.pair)
    numeric_changes = sum(1 for e in events for c in e.changes if c.numeric)
    pair_changes = sum(1 for e in events for c in e.changes if c.pair)
    drift_changes = sum(len(step.drift) for step in steps)
    never = [label for label in event_types if legal_steps[label] == 0]
    quiescent = sum(1 for h in histories if h.quiescent)
    reasons = {label: "never_able" if label in never_able else "preconditions" for label in never}
    # the pair changes per event (stage c2b), only in a world with pair fluents, whose
    # histories list their initial pairs
    pairs = (
        {"pair_changes_per_event": round(pair_changes / len(events), 6) if events else 0.0}
        if any(h.initial_pairs is not None for h in histories)
        else {}
    )
    shares: dict[str, float] = {}
    if place_kinds is not None:
        moves = sum(1 for e in events if place_kinds.get(e.type) == "move")
        acts = sum(1 for e in events if place_kinds.get(e.type) == "act")
        between = sum(1 for e in events if e.patient is not None and e.type not in place_kinds)
        shares = {
            "move_share": _share(moves, len(events)),
            "act_share": _share(acts, len(events)),
            "two_place_share_entities": _share(between, len(events)),
        }
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
        "numeric_changes_per_event": round(numeric_changes / len(events), 6) if events else 0.0,
        "drift_changes_per_step": round(drift_changes / total, 6) if total else 0.0,
        "quiescent_share": round(quiescent / len(histories), 6) if histories else 0.0,
        "two_place_share": round(sum(1 for e in events if e.patient is not None) / len(events), 6)
        if events
        else 0.0,
        **shares,
        **pairs,
        "precondition_redraws": dict(redraws or {}),
        "never_legal": reasons,
        "warnings": [
            f"{label} was never legal in any episode ({reason})"
            for label, reason in reasons.items()
        ],
    }


def relation_stats(statics: StaticWorld) -> dict[str, Any] | None:
    """The ``relations`` block of ``world_stats.yaml``, or None without two-place event types."""
    relations, projections, stats = statics.relations, statics.projections, statics.relation_stats
    if relations is None or projections is None or stats is None or statics.event_tree is None:
        return None
    families: dict[str, int] = {}
    for constraint in relations.constraints:
        families[constraint.family] = families.get(constraint.family, 0) + 1
    approximate = int(projections.agent_approximate.sum() + projections.patient_approximate.sum())
    total = 2 * len(projections.event_type_labels)
    block: dict[str, Any] = {
        "event_type_features": len(statics.event_tree.features),
        "event_type_categories": len(statics.event_tree.categories)
        - len(statics.event_tree.event_types),
        "event_types": len(statics.event_tree.event_types),
        "constraints": len(relations.constraints),
        "constraint_families": families,
        "approximate_projections": round(approximate / total, 6) if total else 0.0,
        "proportions_estimated": stats.estimated,
        "pairs_short": {label: dict(short) for label, short in stats.pairs_short.items()},
    }
    if relations.event_type_density is not None:
        block["outside_density"] = len(relations.outside_range)
        block["outside_density_labels"] = list(relations.outside_range)
    return block


def _share(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def numeric_stats(
    numeric: NumericFluents | None, event_types: EventTypes, fluents: Fluents
) -> dict[str, Any]:
    """The ``numeric_fluents`` block of ``world_stats.yaml`` (stage b2): the fluents with their
    ranges, initial distributions, and drifts; the numeric effects, precondition literals, and
    condition literals by kind of event type and role; and the derived Boolean rules that read a
    numeric threshold."""
    leaves = event_types.leaves
    effects = {
        "total": 0,
        "add": 0,
        "subtract": 0,
        "one_place": 0,
        "two_place": 0,
        "agent": 0,
        "patient": 0,
    }
    literals = {
        "total": 0,
        "above": 0,
        "at_or_below": 0,
        "one_place": 0,
        "two_place": 0,
        "agent": 0,
        "patient": 0,
    }
    conditions = {"total": 0, "above": 0, "at_or_below": 0, "agent": 0, "patient": 0}
    for et in leaves:
        kind = "one_place" if et.arity == 1 else "two_place"
        for effect in et.effects:
            if effect.numeric:
                effects["total"] += 1
                effects["add" if effect.amount > 0 else "subtract"] += 1
                effects[kind] += 1
                effects[effect.role] += 1
            for lit in effect.condition:
                if lit.numeric:
                    conditions["total"] += 1
                    conditions["above" if lit.value else "at_or_below"] += 1
                    conditions[lit.role] += 1
        for lit in et.precondition:
            if lit.numeric:
                literals["total"] += 1
                literals["above" if lit.value else "at_or_below"] += 1
                literals[kind] += 1
                literals[lit.role] += 1
    rules_reading = sum(1 for rule in fluents.rules if rule.numeric_thresholds)
    fluent_rows = {}
    if numeric is not None:
        for f in numeric.fluents:
            fluent_rows[f.label] = {
                "range": [f.low, f.high],
                "initial": f.initial.record(),
                "drift": f.drift,
            }
    return {
        "count": len(fluent_rows),
        "drifting": sum(1 for row in fluent_rows.values() if row["drift"] != 0.0),
        "fluents": fluent_rows,
        "effects": {
            "total": effects["total"],
            "by_operator": {"add": effects["add"], "subtract": effects["subtract"]},
            "by_kind": {"one_place": effects["one_place"], "two_place": effects["two_place"]},
            "by_role": {"agent": effects["agent"], "patient": effects["patient"]},
            "from_features": sum(
                1 for v in event_types.feature_effects.values() if v is not None and v.numeric
            ),
        },
        "precondition_literals": {
            "total": literals["total"],
            "by_side": {"above": literals["above"], "at_or_below": literals["at_or_below"]},
            "by_kind": {"one_place": literals["one_place"], "two_place": literals["two_place"]},
            "by_role": {"agent": literals["agent"], "patient": literals["patient"]},
            "from_features": sum(
                1 for v in event_types.feature_preconditions.values() if v is not None and v.numeric
            ),
        },
        "condition_literals": {
            "total": conditions["total"],
            "by_side": {"above": conditions["above"], "at_or_below": conditions["at_or_below"]},
            "by_role": {"agent": conditions["agent"], "patient": conditions["patient"]},
        },
        "rules_reading_a_threshold": rules_reading,
    }


def _place_dynamics_stats(places: Places, pair_fluents: PairFluents) -> dict[str, Any]:
    """The literals and effects of the place event types with dynamics (stage c3b): the
    precondition literals by kind, role, and polarity (the pair literals, and the Boolean and
    numeric literals on the agent), the compiled literals by origin (``move`` and ``act``),
    the effects by kind (the moves' adds on a location fluent, the Boolean and numeric effects
    on the agent), and the condition literals by kind."""
    kinds = ("pair", "pair_some", "pair_someof", "pair_same")
    literals: dict[str, Any] = {
        "total": 0,
        "by_kind": {"boolean": 0, "numeric": 0, **dict.fromkeys(kinds, 0)},
        "by_role": {"agent": 0, "patient": 0},
        "by_polarity": {"positive": 0, "negative": 0},
    }
    compiled: dict[str, Any] = {"total": 0, "by_origin": {}}
    effects: dict[str, Any] = {
        "total": 0,
        "by_kind": {"location": 0, "boolean": 0, "numeric": 0},
        "conditional": 0,
    }
    conditions: dict[str, Any] = {
        "total": 0,
        "by_kind": {"feature": 0, "fluent": 0, "numeric": 0, **dict.fromkeys(kinds, 0)},
        "by_role": {"agent": 0, "patient": 0},
    }
    for et in places.performable:
        for lit in et.precondition:
            if lit.pair and lit.compiled:
                compiled["total"] += 1
                compiled["by_origin"][lit.origin] = compiled["by_origin"].get(lit.origin, 0) + 1
                continue
            literals["total"] += 1
            literals["by_kind"][
                lit.kind if lit.pair else "numeric" if lit.numeric else "boolean"
            ] += 1
            literals["by_role"][lit.role] += 1
            literals["by_polarity"]["positive" if lit.value else "negative"] += 1
        for effect in et.effects:
            effects["total"] += 1
            effects["by_kind"][
                "location" if effect.pair else "numeric" if effect.numeric else "boolean"
            ] += 1
            if effect.condition:
                effects["conditional"] += 1
            for lit in effect.condition:
                conditions["total"] += 1
                conditions["by_kind"][lit.kind] += 1
                conditions["by_role"][lit.role] += 1
    return {
        "precondition_literals": literals,
        "compiled_literals": compiled,
        "effects": effects,
        "condition_literals": conditions,
    }


def pair_fluent_stats(
    pair_fluents: PairFluents | None,
    event_types: EventTypes,
    fluents: Fluents,
    episodes: dict[str, Any] | None = None,
    places: Places | None = None,
) -> dict[str, Any] | None:
    """The ``pair_fluents`` block of ``world_stats.yaml`` (stage c2b), or None without pair
    fluents. In a world with location fluents (stage c3b) the block also counts the location
    fluents apart from the pair fluents between entities (``between_entities``, ``locations``)
    and the place event types' literals and effects (``place_event_types``)."""
    if pair_fluents is None or not len(pair_fluents):
        return None
    leaves = event_types.leaves
    kinds = ("pair", "pair_some", "pair_someof", "pair_same")
    literals: dict[str, Any] = {
        "total": 0,
        "by_kind": dict.fromkeys(kinds, 0),
        "by_role": {"agent": 0, "patient": 0},
        "by_polarity": {"positive": 0, "negative": 0},
        "by_event_kind": {"one_place": 0, "two_place": 0},
        "from_features": sum(
            1 for v in event_types.feature_preconditions.values() if v is not None and v.pair
        ),
    }
    compiled: dict[str, Any] = {"total": 0, "by_origin": {}}
    effects: dict[str, Any] = {
        "total": 0,
        "by_value": {"add": 0, "delete": 0},
        "by_role": {"agent": 0, "patient": 0},
        "by_event_kind": {"one_place": 0, "two_place": 0},
        "by_pair_fluent": {f.label: 0 for f in pair_fluents},
        "from_features": sum(
            1 for v in event_types.feature_effects.values() if v is not None and v.pair
        ),
    }
    conditions: dict[str, Any] = {
        "total": 0,
        "by_kind": dict.fromkeys(kinds, 0),
        "by_role": {"agent": 0, "patient": 0},
    }
    for et in leaves:
        kind = "one_place" if et.arity == 1 else "two_place"
        for lit in et.precondition:
            if not lit.pair:
                continue
            if lit.compiled:
                compiled["total"] += 1
                compiled["by_origin"][lit.origin] = compiled["by_origin"].get(lit.origin, 0) + 1
                continue
            literals["total"] += 1
            literals["by_kind"][lit.kind] += 1
            literals["by_role"][lit.role] += 1
            literals["by_polarity"]["positive" if lit.value else "negative"] += 1
            literals["by_event_kind"][kind] += 1
        for effect in et.effects:
            if effect.pair:
                effects["total"] += 1
                effects["by_value"]["add" if effect.value else "delete"] += 1
                effects["by_role"][effect.role] += 1
                effects["by_event_kind"][kind] += 1
                effects["by_pair_fluent"][effect.fluent] += 1
            for lit in effect.condition:
                if lit.pair:
                    conditions["total"] += 1
                    conditions["by_kind"][lit.kind] += 1
                    conditions["by_role"][lit.role] += 1
    by_symmetry = {"symmetric": 0, "asymmetric": 0, "general": 0}
    rows: dict[str, Any] = {}
    for fluent, matrix in zip(pair_fluents.fluents, pair_fluents.initial_pairs, strict=True):
        by_symmetry[fluent.symmetry] += 1
        rows[fluent.label] = {
            "symmetry": fluent.symmetry,
            "functional": fluent.functional,
            "initial_rate": fluent.initial_rate,
            "initial_pairs": int(matrix.sum()),
        }
    locations = len(pair_fluents.locations)
    return {
        "count": len(pair_fluents),
        **(
            {"between_entities": len(pair_fluents.between_entities), "locations": locations}
            if locations
            else {}
        ),
        "by_symmetry": by_symmetry,
        "functional": sum(1 for f in pair_fluents if f.functional),
        "fluents": rows,
        "initial_pairs": sum(row["initial_pairs"] for row in rows.values()),
        "precondition_literals": literals,
        "compiled_literals": compiled,
        "effects": effects,
        "condition_literals": conditions,
        "rules_reading_a_pair_literal": sum(1 for rule in fluents.rules if rule.pair_inputs),
        "fix_ups": dict(event_types.pair_stats),
        "pair_changes_per_event": (episodes or {}).get("pair_changes_per_event"),
        **(
            {"place_event_types": _place_dynamics_stats(places, pair_fluents)}
            if locations and places is not None
            else {}
        ),
    }


def world_stats(
    fluents: Fluents,
    event_types: EventTypes,
    episodes: dict[str, Any] | None = None,
    statics: StaticWorld | None = None,
    numeric: NumericFluents | None = None,
    places: Places | None = None,
    pair_fluents: PairFluents | None = None,
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
    conditional = {"one_place": 0, "two_place": 0}
    condition_literals = {
        "total": 0,
        "feature": 0,
        "fluent": 0,
        "numeric": 0,
        "agent": 0,
        "patient": 0,
    }
    for et in leaves:
        kind = "one_place" if et.arity == 1 else "two_place"
        for lit in et.precondition:
            literals_by_role[lit.role] += 1
            literals_by_kind[kind] += 1
        for effect in et.effects:
            effects_by_role[effect.role] += 1
            effects_by_kind[kind] += 1
            if effect.condition:
                conditional[kind] += 1
            for lit in effect.condition:
                condition_literals["total"] += 1
                # a pair condition literal (stage c2a) counts under pair, written only when
                # the world has one, so that a world without pair fluents writes its statistics
                # as before
                condition_literals["pair" if lit.pair else lit.kind] = (
                    condition_literals.get("pair" if lit.pair else lit.kind, 0) + 1
                )
                condition_literals[lit.role] += 1
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
            "conditional": {
                "total": sum(conditional.values()),
                "by_kind": conditional,
                # Jon's ruling 2 on stage b1: the share of the effects a condition was drawn for
                # (an inherited effect once) that got one, beside the share of all effects of
                # all event types, which inheritance makes larger
                "share_of_drawn": _share(
                    event_types.condition_draws.get("conditional", 0),
                    event_types.condition_draws.get("drawn", 0),
                ),
                "share_of_all": _share(sum(conditional.values()), sum(effects_by_kind.values())),
            },
            "condition_literals": {
                "total": condition_literals["total"],
                "by_symbol": {
                    "feature": condition_literals["feature"],
                    "fluent": condition_literals["fluent"],
                    "numeric": condition_literals["numeric"],
                    **(
                        {"pair": condition_literals["pair"]}
                        if condition_literals.get("pair")
                        else {}
                    ),
                },
                "by_role": {
                    "agent": condition_literals["agent"],
                    "patient": condition_literals["patient"],
                },
                "from_features": sum(
                    len(v.condition) for v in event_types.feature_effects.values() if v is not None
                ),
            },
        },
        "fix_ups": dict(event_types.stats),
        "enabling_graph": {
            "edges": len(edges),
            "longest_chain": longest_chain([et.label for et in leaves], edges),
        },
        "absorbing_fluents": absorbing_fluents(event_types, fluents),
        "numeric_fluents": numeric_stats(numeric, event_types, fluents),
        "relations": relation_stats(statics) if statics is not None else None,
        "places": places_stats(
            places, pair_fluents, statics.taxonomy if statics is not None else None
        ),
        **(
            {
                "pair_fluents": pair_fluent_stats(
                    pair_fluents, event_types, fluents, episodes, places
                )
            }
            if pair_fluents is not None and len(pair_fluents)
            else {}
        ),
        "episodes": episodes,
    }
