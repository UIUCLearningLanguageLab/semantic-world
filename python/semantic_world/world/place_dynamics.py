"""The dynamics of place event types (stage c3a of ``docs/specs/WORLD_PHASE_C.md``, "Place
event types" and "Explicit event files"): moves and acts at a place as an event file gives them.

A place event type has an entity as its agent and a place as its patient. An event file's
``place_event_types`` entry makes it performable: a **move** (``kind: move``) gets the effect
``PAIRFL.k(agent, patient) := 1`` on its location fluent, which the runtime applies as the
total functional replacement, and the compiled precondition literal ``NOT PAIRFL.k(agent,
patient)`` with the origin ``move`` (WM.53), so that an entity never moves to where it is; an
**act** (``kind: act``) gets the precondition literal ``PAIRFL.k(agent, patient)`` with the
origin ``act``, so that it happens at the agent's place. A literal or an effect the entry
already gives by what it names is kept as given and not added twice (the rule of
``add_compiled_literals``). An entry's further precondition literals read the agent (its
fluents, numeric thresholds, and ``SOME`` and ``SOMEOF`` of its pair fluents between entities)
and, as ``SOMEOF`` on the location fluent, the place; its further effects write the agent's
Boolean and numeric fluents; nothing writes a fact about the place, and nothing deletes a pair
of a location fluent. ``location`` names the location fluent the entry uses and may be left out
when the world has one. Every error names the file and the entry. A place event type without an
entry stays as in stage c1: a requirement only, not performable, until stage c3b draws its
dynamics from ``world:place_dynamics``.

``check_place_dynamics`` holds the same rules as invariants of a built world, which the
generator and the tests check.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.dynamics import (
    ACT_ORIGIN,
    PAIR_KIND,
    AnyLiteral,
    ConditionLiteral,
    Effect,
    PairLiteral,
)
from semantic_world.world.event_file import EventFile, ExplicitPlaceEventType
from semantic_world.world.event_types import (
    compiled_literals,
    pair_effect_key,
    pair_literal_achievable,
    pair_literal_key,
    pair_literal_names,
)
from semantic_world.world.fluents import Fluents
from semantic_world.world.numeric import NumericFluents, empty_numeric
from semantic_world.world.pair_fluents import PairFluents, empty_pair_fluents
from semantic_world.world.places import PlaceEventType, Places

ROLES = ("agent", "patient")


def _location_problem(literal: PairLiteral, total: bool) -> str | None:
    from semantic_world.world.definition import location_literal_problem

    return location_literal_problem(literal, total, True)


class _Checker:
    """The rules of a place event type's dynamics, raising through ``fail``."""

    def __init__(
        self,
        fluents: Fluents,
        numeric: NumericFluents,
        pair_fluents: PairFluents,
        achievable: set[tuple[str, bool]],
        signs: set[tuple[str, int]],
        adds: set[str],
        deletes: set[str],
    ) -> None:
        self.fluents = fluents
        self.numeric = numeric
        self.pair_fluents = pair_fluents
        self.achievable = achievable
        self.signs = signs
        self.adds = adds
        self.deletes = deletes

    def literal_ok(self, lit: AnyLiteral, fail, location: str | None, kind: str | None) -> None:
        """A precondition literal of a place event type."""
        if isinstance(lit, PairLiteral):
            self.pair_literal_ok(lit, fail, location, kind, "precondition")
            return
        if lit.role != "agent":
            fail(f"{lit.text} reads the patient, which is a place and has no fluent")
        if lit.numeric:
            if lit.fluent not in self.numeric:
                fail(f"{lit.text} reads unknown numeric fluent {lit.fluent}")
            low, high = self.numeric[lit.fluent].range
            if lit.threshold is None or not low < lit.threshold < high:
                fail(f"the threshold {lit.threshold!r} on {lit.fluent} is not inside its range")
            if not self.numeric.achievable(lit.fluent, lit.value, lit.threshold, self.signs):
                fail(f"{lit.text} is not achievable")
            return
        if lit.fluent not in self.fluents:
            fail(f"{lit.text} reads unknown fluent {lit.fluent}")
        if (lit.fluent, lit.value) not in self.achievable:
            fail(
                f"{lit.text} is not achievable: no entity starts with that value and no effect "
                f"produces it"
            )

    def pair_literal_ok(
        self, lit: PairLiteral, fail, location: str | None, kind: str | None, where: str
    ) -> None:
        if lit.pair_fluent not in self.pair_fluents:
            fail(
                f"{lit.key} reads unknown pair fluent {lit.pair_fluent}; the world's pair "
                f"fluents are {', '.join(self.pair_fluents.labels) or 'none'}"
            )
        fluent = self.pair_fluents[lit.pair_fluent]
        problem = _location_problem(lit, fluent.total)
        if problem is not None:
            fail(f"{lit.key} {problem}")
        if fluent.total and lit.kind == PAIR_KIND:
            if location is not None and lit.pair_fluent != location:
                fail(
                    f"{lit.key} reads the location fluent {lit.pair_fluent}, but the entry uses "
                    f"{location}"
                )
            if where == "condition":
                fail(
                    f"the condition reads {lit.key}, the agent's place at the bound place, which "
                    f"the {'act' if kind == 'act' else 'move'} already fixes"
                )
            if kind == "act" and not lit.value:
                fail(
                    f"an act needs its location literal {lit.positive}, but the precondition "
                    f"denies it"
                )
            if kind == "move" and lit.value:
                fail(
                    f"a move's agent is never at the bound place (the compiled literal NOT "
                    f"{lit.positive}), but the precondition requires it"
                )
        if not pair_literal_achievable(lit, fluent, self.adds, self.deletes):
            fail(f"{lit.text} is not achievable")

    def effect_ok(
        self, effect: Effect, fail, location: str | None, kind: str | None, seen: dict
    ) -> None:
        if effect.role != "agent":
            fail(f"{effect.text} writes a fact about the patient, which is a place")
        if effect.pair:
            if effect.fluent not in self.pair_fluents:
                fail(f"{effect.text} writes unknown pair fluent {effect.fluent}")
            fluent = self.pair_fluents[effect.fluent]
            if not fluent.total:
                fail(
                    f"{effect.text} writes a pair between entities, but the patient of a place "
                    f"event type is a place"
                )
            if not effect.value:
                fail(f"{effect.text} deletes a pair of the location fluent; nothing does")
            if kind == "act":
                fail(f"{effect.text} moves the agent, but the entry is an act")
            if location is not None and effect.fluent != location:
                fail(f"{effect.text} writes {effect.fluent}, but the entry uses {location}")
            if effect.second != "patient":
                fail(f"a move's effect is PAIRFL.k(agent, patient) := 1, got {effect.text}")
            if effect.condition:
                fail(f"{effect.text} is a conditional move; a move's add has no condition")
            key = pair_effect_key(effect, fluent)
        else:
            if effect.numeric:
                if effect.fluent not in self.numeric:
                    fail(f"{effect.text} adds to unknown numeric fluent {effect.fluent}")
            else:
                if effect.fluent not in self.fluents:
                    fail(f"{effect.text} writes unknown fluent {effect.fluent}")
                if self.fluents[effect.fluent].derived:
                    fail(f"{effect.fluent} is a derived fluent; effects write base fluents only")
            key = (effect.role, effect.fluent)
        if key in seen:
            fail(
                f"contradicts an earlier effect on {effect.key}"
                if seen[key] != effect.value
                else f"writes {effect.key} twice"
            )
        seen[key] = effect.value
        self.condition_ok(effect, fail, location, kind)

    def condition_ok(self, effect: Effect, fail, location: str | None, kind: str | None) -> None:
        keys: set[tuple] = set()
        for lit in effect.condition:
            if isinstance(lit, PairLiteral):
                self.pair_literal_ok(lit, fail, location, kind, "condition")
                key = pair_literal_key(lit, self.pair_fluents[lit.pair_fluent])
            else:
                assert isinstance(lit, ConditionLiteral)
                if lit.role != "agent":
                    fail(
                        f"the condition of {effect.text} reads {lit.key} of the patient, which is "
                        f"a place"
                    )
                if lit.kind == "numeric":
                    if lit.symbol not in self.numeric:
                        fail(f"the condition reads unknown numeric fluent {lit.symbol}")
                elif lit.kind == "fluent":
                    if lit.symbol not in self.fluents:
                        fail(f"the condition reads unknown fluent {lit.symbol}")
                    if (lit.symbol, lit.value) not in self.achievable:
                        fail(f"the condition literal {lit.text} is not achievable")
                    if (lit.role, lit.symbol) == (effect.role, effect.fluent):
                        fail(f"the condition reads {lit.key}, the fluent the effect sets")
                key = (lit.role, lit.symbol)
            if key in keys:
                fail(f"the condition of {effect.text} reads {lit.key} twice")
            keys.add(key)


def _achievable(
    fluents: Fluents,
    initial_present: set[tuple[str, bool]],
    effects: Sequence[Effect],
) -> set[tuple[str, bool]]:
    found = set(initial_present)
    found |= {(e.fluent, e.value) for e in effects if not e.numeric and not e.pair}
    return found


def apply_place_dynamics(
    places: Places | None,
    event_file: EventFile | None,
    fluents: Fluents,
    numeric: NumericFluents | None,
    pair_fluents: PairFluents | None,
    initial_present: set[tuple[str, bool]],
    entity_effects: Sequence[Effect],
    signs: set[tuple[str, int]],
) -> Places | None:
    """The places with the moves and acts of the event file's ``place_event_types`` entries
    applied (stage c3a). ``initial_present`` and ``entity_effects`` give the Boolean values an
    agent literal can take (the values of ``TIME.1`` and those the entity event types' effects
    produce; the entries' own effects count too), ``signs`` the numeric directions."""
    if event_file is None or not event_file.place_event_types:
        return places
    source = event_file.source
    numeric = numeric if numeric is not None else empty_numeric(0)
    pair_fluents = pair_fluents if pair_fluents is not None else empty_pair_fluents()
    if places is None:
        entry = next(iter(event_file.place_event_types.values()))
        raise ConfigError(source, entry.field, "the world has no places (places: null)")
    locations = [f.label for f in pair_fluents.locations]
    all_effects = list(entity_effects) + [
        e for entry in event_file.place_event_types.values() for e in (entry.effects or ())
    ]
    achievable = _achievable(fluents, initial_present, all_effects)
    adds = {e.fluent for e in all_effects if e.pair and e.value} | set(locations)
    deletes = {e.fluent for e in all_effects if e.pair and not e.value}
    checker = _Checker(fluents, numeric, pair_fluents, achievable, signs, adds, deletes)
    by_label = {et.label: et for et in places.event_types}
    updated = dict(by_label)
    for label, entry in event_file.place_event_types.items():
        updated[label] = _apply_entry(entry, by_label, checker, locations, source)
    return dataclasses.replace(
        places, event_types=tuple(updated[et.label] for et in places.event_types)
    )


def _apply_entry(
    entry: ExplicitPlaceEventType,
    by_label: Mapping[str, PlaceEventType],
    checker: _Checker,
    locations: Sequence[str],
    source: str,
) -> PlaceEventType:
    def fail(message: str, where: str | None = None):
        raise ConfigError(source, where or entry.field, message)

    if entry.label not in by_label:
        fail("unknown place event type; the world has none with this label")
    if not locations:
        fail(
            "the world has no location fluent, so no place event type can be a move or an act; "
            "set pair_fluents.locations above 0"
        )
    location = entry.location
    if location is None:
        if len(locations) > 1:
            fail(
                f"the world has {len(locations)} location fluents ({', '.join(locations)}); name "
                f"the one the entry uses under location"
            )
        location = locations[0]
    elif location not in locations:
        fail(
            f"{location} is not a location fluent of the world; the location fluents are "
            f"{', '.join(locations)}",
            f"{entry.field}.location",
        )
    kind = entry.kind
    literals: list[AnyLiteral] = list(entry.precondition or ())
    seen_literals: dict[tuple, bool] = {}
    for lit in literals:
        checker.literal_ok(lit, lambda m: fail(m, f"{entry.field}.precondition"), location, kind)
        key = _literal_slot(lit, checker.pair_fluents)
        if key in seen_literals:
            fail(
                f"holds {lit.key} twice"
                if lit.numeric or seen_literals[key] == lit.value
                else f"holds {lit.key} and its negation",
                f"{entry.field}.precondition",
            )
        seen_literals[key] = lit.value
    effects: list[Effect] = list(entry.effects or ())
    seen_effects: dict[tuple, bool | float] = {}
    for i, effect in enumerate(effects):
        checker.effect_ok(
            effect,
            lambda m, i=i: fail(m, f"{entry.field}.effects[{i}]"),
            location,
            kind,
            seen_effects,
        )
        slot = (effect.role, effect.fluent) if not effect.pair else None
        if slot is not None and not effect.numeric and seen_literals.get(slot) == effect.value:
            fail(
                f"the effect {effect.text} sets a fluent to the value the precondition already "
                f"requires, so it changes nothing"
            )
        for lit in effect.condition:
            if isinstance(lit, PairLiteral):
                names = pair_literal_names(lit, checker.pair_fluents[lit.pair_fluent])
                for other in literals:
                    if isinstance(other, PairLiteral) and names & pair_literal_names(
                        other, checker.pair_fluents[other.pair_fluent]
                    ):
                        fail(
                            f"the condition of {effect.text} reads {lit.key}, which names what "
                            f"the precondition already names ({other.key})",
                            f"{entry.field}.effects[{i}]",
                        )
            elif lit.kind != "feature" and (lit.role, lit.symbol) in seen_literals:
                fail(
                    f"the condition of {effect.text} reads {lit.key}, which the precondition "
                    f"already fixes",
                    f"{entry.field}.effects[{i}]",
                )
    fluent = checker.pair_fluents[location]
    if kind == "move":
        add = Effect("agent", location, True, (), "patient")
        if pair_effect_key(add, fluent) not in seen_effects:
            effects.append(add)
        for compiled in compiled_literals(add, fluent):
            if pair_literal_key(compiled, fluent) not in seen_literals:
                literals.append(compiled)
    else:
        act = PairLiteral(PAIR_KIND, location, "agent", "patient", True, ACT_ORIGIN)
        if pair_literal_key(act, fluent) not in seen_literals:
            literals.append(act)
    return dataclasses.replace(
        by_label[entry.label],
        precondition=tuple(literals),
        effects=tuple(effects),
        explicit=True,
    )


def _literal_slot(literal: AnyLiteral, pair_fluents: PairFluents) -> tuple:
    if isinstance(literal, PairLiteral):
        return pair_literal_key(literal, pair_fluents[literal.pair_fluent])
    return (literal.role, literal.fluent)


def check_place_dynamics(
    places: Places | None,
    fluents: Fluents,
    initial_present: set[tuple[str, bool]],
    entity_effects: Sequence[Effect],
    signs: set[tuple[str, int]],
    numeric: NumericFluents | None = None,
    pair_fluents: PairFluents | None = None,
) -> None:
    """The invariants of the place event types' dynamics (stage c3a), raising
    :class:`ValueError` naming the place event type: a performable place event type is a move
    (exactly one add on a location fluent, of the agent at the patient, unconditional, with
    its compiled ``move`` literal) or an act (its location literal, positive, and no location
    effect); every further literal reads the agent, or ``SOMEOF`` of the place, and is
    achievable; every effect writes the agent's base fluents; no literal twice; and a place
    event type without dynamics has an empty precondition and no effects."""
    if places is None:
        return
    numeric = numeric if numeric is not None else empty_numeric(0)
    pair_fluents = pair_fluents if pair_fluents is not None else empty_pair_fluents()
    locations = [f.label for f in pair_fluents.locations]
    all_effects = list(entity_effects) + [e for et in places.event_types for e in et.effects]
    achievable = _achievable(fluents, initial_present, all_effects)
    adds = {e.fluent for e in all_effects if e.pair and e.value} | set(locations)
    deletes = {e.fluent for e in all_effects if e.pair and not e.value}
    checker = _Checker(fluents, numeric, pair_fluents, achievable, signs, adds, deletes)
    for et in places.event_types:
        if not et.performable:
            continue

        def fail(message: str, label: str = et.label):
            raise ValueError(f"{label}: {message}")

        location = et.location
        if location is None or location not in locations:
            fail("a performable place event type names a location fluent")
        kind = et.dynamics_kind
        seen_literals: dict[tuple, bool] = {}
        location_literals = 0
        for lit in et.precondition:
            checker.literal_ok(lit, fail, location, kind)
            key = _literal_slot(lit, pair_fluents)
            if key in seen_literals:
                fail(f"the precondition reads {lit.key} twice")
            seen_literals[key] = lit.value
            if (
                isinstance(lit, PairLiteral)
                and lit.kind == PAIR_KIND
                and lit.pair_fluent == location
            ):
                location_literals += 1
        if location_literals != 1:
            fail(
                "a move has its compiled literal NOT PAIRFL.k(agent, patient) and an act its "
                "location literal PAIRFL.k(agent, patient), exactly once"
            )
        seen_effects: dict[tuple, bool | float] = {}
        for effect in et.effects:
            checker.effect_ok(effect, fail, location, kind, seen_effects)
            if (
                not effect.pair
                and not effect.numeric
                and seen_literals.get((effect.role, effect.fluent)) == effect.value
            ):
                fail(f"the effect {effect.text} changes nothing")
            for lit in effect.condition:
                if isinstance(lit, PairLiteral):
                    names = pair_literal_names(lit, pair_fluents[lit.pair_fluent])
                    if any(
                        isinstance(o, PairLiteral)
                        and names & pair_literal_names(o, pair_fluents[o.pair_fluent])
                        for o in et.precondition
                    ):
                        fail(f"the condition of {effect.text} reads what the precondition names")
                elif lit.kind != "feature" and (lit.role, lit.symbol) in seen_literals:
                    fail(f"the condition of {effect.text} reads what the precondition fixes")
        moves = sum(1 for e in et.effects if e.pair and e.value)
        if kind == "move" and moves != 1:
            fail("a move has exactly one add on its location fluent")


def place_produced(places: Places | None) -> set[tuple[str, bool]]:
    """The Boolean values the place event types' effects produce (stage c3a)."""
    if places is None:
        return set()
    return {
        (e.fluent, bool(e.value))
        for et in places.event_types
        for e in et.effects
        if not e.numeric and not e.pair
    }


def place_signs(places: Places | None) -> set[tuple[str, int]]:
    """``(fluent, +1 or -1)`` for every numeric effect of the place event types."""
    if places is None:
        return set()
    return {
        (e.fluent, 1 if e.amount > 0 else -1)
        for et in places.event_types
        for e in et.effects
        if e.numeric
    }


__all__ = [
    "apply_place_dynamics",
    "check_place_dynamics",
    "place_produced",
    "place_signs",
]
