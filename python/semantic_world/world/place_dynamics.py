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
entry stays as in stage c1 until stage c3b draws its dynamics from ``world:place_dynamics``
(below, "Drawn dynamics").

Since stage c3b every place event type without an entry draws its dynamics, in label order, from
``world:place_dynamics`` (``draw_place_dynamics``): its kind (a move at
``place_event_types.move_share``, an act otherwise), its location fluent uniformly among the
world's location fluents, and its further effects and precondition literals as the entity event
types draw theirs (``_Dynamics`` in ``event_types.py``): the number of effects from
``event_types.effects.count`` (a move's add on its location fluent counts as its first effect),
each a Boolean effect on the agent replaced by a numeric one at ``effects.numeric_share``; the
number of literals from ``preconditions.literals``, each a Boolean literal on the agent (an
enabling one at ``enabled_share``), replaced by a numeric threshold at ``numeric_share`` and by a
pair literal at ``pair_share``, where the pair literals a place event type can read are ``SOME``
and ``SOMEOF`` of the agent on the pair fluents between entities and ``SOMEOF`` of the place on
its location fluent (``random_place_pair_literal``); a condition on a further effect at
``conditional_share``, its literals on the agent (a static feature, a Boolean fluent, a numeric
threshold) or such a pair literal. An explicit entry's condition whose value the precondition
fixes under the implications of ``fixed_pair_literals`` (an act's location literal fixes
``SOMEOF`` of the place true, so a condition of an act never reads it) is an error that names
the file and the entry, as a drawn condition is drawn again for it (Jon's ruling WM.68 of the
stage c3b follow-up). Every rule the checker holds is a redraw here: a slot taken,
an effect the precondition makes empty, a literal that is not achievable (over every effect of
the world, WM.66), a pair literal implied by or contradicting another or the compiled literal,
a condition that reads what the precondition fixes or names, the fluent the effect sets, or a
feature no able binding has. The compiled ``move`` literal or the ``act`` literal comes last in
the precondition, as the compiled literals of the entity event types do, and a move's add comes
first among its effects. A never-legal place event type draws its further literals again from
its own part, ``world:place_dynamics:<label>`` (``redraw_place_preconditions``), as ``define``
redraws the entity event types.

``check_place_dynamics`` holds the same rules as invariants of a built world, which the
generator and the tests check.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.features import FeatureSet
from semantic_world.taxonomy.rules import MAX_TRIES, _draw
from semantic_world.world.config import Config
from semantic_world.world.dynamics import (
    ACT_ORIGIN,
    PAIR_KIND,
    PAIR_SOME,
    PAIR_SOMEOF,
    AnyConditionLiteral,
    AnyLiteral,
    ConditionLiteral,
    Effect,
    Literal,
    PairLiteral,
)
from semantic_world.world.event_file import EventFile, ExplicitPlaceEventType
from semantic_world.world.event_types import (
    compiled_literals,
    condition_keys,
    fixed_condition_problem,
    fixed_pair_literals,
    implied_conflict,
    pair_effect_key,
    pair_literal_achievable,
    pair_literal_key,
    pair_literal_names,
    precondition_literal_fits_conditions,
)
from semantic_world.world.fluents import Fluents
from semantic_world.world.numeric import NumericFluents, drawn, empty_numeric
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
    # WM.68: no condition whose value the precondition fixes under the implications the
    # generator uses, the location literal included (an act's fixes SOMEOF of the place true)
    for i, effect in enumerate(effects):
        problem = fixed_condition_problem(effect, literals, checker.pair_fluents, ROLES)
        if problem is not None:
            fail(problem, f"{entry.field}.effects[{i}]")
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
    achievable; every effect writes the agent's base fluents; no literal twice; no condition
    whose value the precondition fixes under ``fixed_pair_literals`` (WM.68); and a place event
    type without dynamics has an empty precondition and no effects."""
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
            problem = fixed_condition_problem(effect, et.precondition, pair_fluents, ROLES)
            if problem is not None:
                fail(problem)
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


# ---------------------------------------------------------------------------------------------
# Drawn dynamics (stage c3b)
# ---------------------------------------------------------------------------------------------

PLACE_DYNAMICS_STATS = (
    "drawn",
    "moves",
    "acts",
    "effects_drawn",
    "effects_redrawn",
    "effects_dropped_no_slot",
    "numeric_effect_replacements",
    "literals_drawn",
    "literals_redrawn",
    "literals_dropped",
    "numeric_literal_replacements",
    "pair_replacement_draws",
    "pair_replacements",
    "pair_replacements_failed",
    "pair_literals_redrawn",
    "conditions_drawn",
    "conditions_redrawn",
    "conditions_dropped",
    "pair_conditions_drawn",
)
"""The counts of the drawn place dynamics (``places.dynamics`` of ``world_stats.yaml``): the
place event types drawn, the moves and the acts among them, the effects and literals drawn and
drawn again (a slot taken, an empty effect, an unachievable or implied literal), the Boolean
effects and literals replaced by numeric ones and the literals replaced by pair literals (the
draws against the share, the replacements, and the ones that found no pair literal), the pair
literal candidates drawn again, and the conditions drawn, drawn again, and dropped."""


def random_place_pair_literal(
    rng: np.random.Generator, pair_fluents: PairFluents, location: str
) -> PairLiteral:
    """One drawn pair literal a place event type can read (stage c3b), unchecked: its pair
    fluent uniformly among the pair fluents between entities and the event type's location
    fluent; on a pair fluent between entities ``SOME`` or ``SOMEOF`` of the agent (the form
    uniformly), on the location fluent ``SOMEOF`` of the place; its polarity uniformly. The
    kind of the literal is not drawn: by stage c3a's rules a place event type's ``pair``
    literal on its location fluent is its location literal, which it already has, and ``SAME``
    is never read between an entity and a place, so ``some`` is the one kind it can take."""
    pool = list(pair_fluents.between_entities) + [pair_fluents[location]]
    fluent = pool[int(rng.integers(len(pool)))]
    if fluent.total:
        return PairLiteral(PAIR_SOMEOF, fluent.label, "patient", None, bool(rng.integers(2)))
    form = PAIR_SOME if rng.integers(2) == 0 else PAIR_SOMEOF
    return PairLiteral(form, fluent.label, "agent", None, bool(rng.integers(2)))


class _PlaceDrawer:
    """Draws the dynamics of place event types from a generator, as ``_Dynamics`` draws the
    entity event types': the pools of achievable values hold the initial values and every
    effect of the world known when a literal is drawn (the entity event types', the explicit
    place entries', and the place event types drawn so far; WM.66)."""

    def __init__(
        self,
        config: Config,
        fluents: Fluents,
        numeric: NumericFluents,
        pair_fluents: PairFluents,
        features: FeatureSet,
        entity_values: np.ndarray,
        places: Places,
        initial_present: set[tuple[str, bool]],
        effects: Sequence[Effect],
    ) -> None:
        assert config.place_event_types is not None
        self.effects_config = config.event_types.effects
        self.preconditions_config = config.event_types.preconditions
        self.move_share = config.place_event_types.move_share
        self.pair_drawable = config.pair_fluents.literal_kinds.get("some", 0) > 0
        """Whether a place event type can draw a pair literal at all: ``some`` is the one kind
        it can take, so the kind needs a positive weight in ``pair_fluents.literal_kinds``."""
        self.fluents = fluents
        self.numeric = numeric
        self.pair_fluents = pair_fluents
        self.locations = [f.label for f in pair_fluents.locations]
        self.base_labels = fluents.base_labels
        self.all_labels = fluents.labels
        self.numeric_labels = numeric.labels
        self.feature_labels = tuple(f.label for f in features.features)
        self.feature_position = {f.label: f.position for f in features.features}
        self.entity_values = entity_values
        self.places = places
        self.initial_present = set(initial_present)
        self.effects: list[Effect] = list(effects)
        """Every effect of the world known so far: the pools read them."""
        self.stats: dict[str, int] = dict.fromkeys(PLACE_DYNAMICS_STATS, 0)

    # Pools ----------------------------------------------------------------------------------

    def produced(self) -> list[tuple[str, bool]]:
        return sorted({(e.fluent, e.value) for e in self.effects if not e.numeric and not e.pair})

    def achievable(self) -> set[tuple[str, bool]]:
        return set(self.produced()) | self.initial_present

    def signs(self) -> set[tuple[str, int]]:
        return {(e.fluent, 1 if e.amount > 0 else -1) for e in self.effects if e.numeric}

    def adds_and_deletes(self) -> tuple[set[str], set[str]]:
        adds = {e.fluent for e in self.effects if e.pair and e.value} | set(self.locations)
        deletes = {e.fluent for e in self.effects if e.pair and not e.value}
        return adds, deletes

    # Effects ---------------------------------------------------------------------------------

    def _numeric_effect(self, rng: np.random.Generator) -> Effect:
        cfg = self.effects_config
        fluent = self.numeric_labels[int(rng.integers(len(self.numeric_labels)))]
        sign = 1.0 if rng.integers(2) == 0 else -1.0
        amount = drawn(rng.uniform(cfg.numeric_amounts[0], cfg.numeric_amounts[1]))
        return Effect("agent", fluent, sign * amount)

    def _draw_effect(self, rng: np.random.Generator, forbidden: set[tuple]) -> Effect | None:
        """An effect on the agent whose slot is free, or None after ``MAX_TRIES``."""
        share = self.effects_config.numeric_share
        for _ in range(MAX_TRIES):
            if self.base_labels:
                fluent = self.base_labels[int(rng.integers(len(self.base_labels)))]
                effect = Effect("agent", fluent, bool(rng.integers(2)))
                if self.numeric_labels and share > 0 and rng.random() < share:
                    self.stats["numeric_effect_replacements"] += 1
                    effect = self._numeric_effect(rng)
            else:
                effect = self._numeric_effect(rng)
            if (effect.role, effect.fluent) in forbidden:
                self.stats["effects_redrawn"] += 1
                continue
            return effect
        return None

    # Literals --------------------------------------------------------------------------------

    def _numeric_literal(
        self, rng: np.random.Generator, forbidden: set[tuple], signs: set[tuple[str, int]]
    ) -> Literal | None:
        cfg = self.preconditions_config
        for _ in range(MAX_TRIES):
            fluent = self.numeric_labels[int(rng.integers(len(self.numeric_labels)))]
            above = bool(rng.integers(2) == 0)
            fraction = cfg.numeric_thresholds[int(rng.integers(len(cfg.numeric_thresholds)))]
            threshold = self.numeric[fluent].point(fraction)
            if ("agent", fluent) in forbidden or not self.numeric.achievable(
                fluent, above, threshold, signs
            ):
                self.stats["literals_redrawn"] += 1
                continue
            return Literal("agent", fluent, above, threshold)
        return None

    def _pair_literal(
        self,
        rng: np.random.Generator,
        kept: Mapping[tuple, AnyLiteral],
        effects: Sequence[Effect],
        location: str,
    ) -> PairLiteral | None:
        """A pair literal the place event type can read, drawn again while its slot is taken,
        it is not achievable, it clashes with a pair condition of the effects, or it is
        implied by, contradicts, or implies another pair literal of the precondition (the
        location literal included); None after ``MAX_TRIES``."""
        if not self.pair_drawable:
            return None
        adds, deletes = self.adds_and_deletes()
        others = [lit for lit in kept.values() if isinstance(lit, PairLiteral)]
        for _ in range(MAX_TRIES):
            literal = random_place_pair_literal(rng, self.pair_fluents, location)
            fluent = self.pair_fluents[literal.pair_fluent]
            if (
                pair_literal_key(literal, fluent) in kept
                or not pair_literal_achievable(literal, fluent, adds, deletes)
                or not precondition_literal_fits_conditions(
                    literal, effects, self.pair_fluents, ROLES
                )
                or implied_conflict(literal, others, self.pair_fluents, ROLES)
            ):
                self.stats["pair_literals_redrawn"] += 1
                continue
            return literal
        return None

    def _draw_literal(
        self,
        rng: np.random.Generator,
        kept: Mapping[tuple, AnyLiteral],
        effects: Sequence[Effect],
        location: str,
    ) -> AnyLiteral | None:
        """A precondition literal on the agent: a Boolean literal (an enabling one at
        ``enabled_share``), replaced by a numeric literal at ``numeric_share`` and by a pair
        literal at ``pair_share``; achievable, in a free slot, making no Boolean effect of the
        event type empty, and fixing nothing a condition reads."""
        cfg = self.preconditions_config
        achievable = self.achievable()
        produced = self.produced()
        signs = self.signs()
        if not achievable and not self.numeric_labels:
            return None
        forbidden = set(kept) | condition_keys(effects)
        empty = {(e.role, e.fluent, e.value) for e in effects if not e.numeric and not e.pair}
        for _ in range(MAX_TRIES):
            if not self.all_labels:
                literal = self._numeric_literal(rng, forbidden, signs)
                if literal is None:
                    return None
                return self._maybe_pair_literal(rng, literal, kept, effects, location)
            if produced and rng.random() < cfg.enabled_share:
                fluent, value = produced[int(rng.integers(len(produced)))]
            else:
                fluent = self.all_labels[int(rng.integers(len(self.all_labels)))]
                value = bool(rng.integers(2))
            if (
                (fluent, value) not in achievable
                or ("agent", fluent) in forbidden
                or ("agent", fluent, value) in empty
            ):
                self.stats["literals_redrawn"] += 1
                continue
            literal = Literal("agent", fluent, value)
            if self.numeric_labels and cfg.numeric_share > 0 and rng.random() < cfg.numeric_share:
                numeric = self._numeric_literal(rng, forbidden, signs)
                if numeric is not None:
                    self.stats["numeric_literal_replacements"] += 1
                    literal = numeric
            return self._maybe_pair_literal(rng, literal, kept, effects, location)
        return None

    def _maybe_pair_literal(
        self,
        rng: np.random.Generator,
        literal: Literal,
        kept: Mapping[tuple, AnyLiteral],
        effects: Sequence[Effect],
        location: str,
    ) -> AnyLiteral:
        share = self.preconditions_config.pair_share
        if share <= 0 or not self.pair_drawable:
            return literal
        self.stats["pair_replacement_draws"] += 1
        if rng.random() < share:
            candidate = self._pair_literal(rng, kept, effects, location)
            if candidate is not None:
                self.stats["pair_replacements"] += 1
                return candidate
            self.stats["pair_replacements_failed"] += 1
        return literal

    def draw_literals(
        self,
        rng: np.random.Generator,
        kept: dict[tuple, AnyLiteral],
        effects: Sequence[Effect],
        location: str,
    ) -> None:
        """Draw the further precondition literals into ``kept`` (which holds the location
        literal), their number from ``preconditions.literals``."""
        if not self.all_labels and not self.numeric_labels:
            return
        n = _draw(rng, self.preconditions_config.literals)
        for _ in range(n):
            self.stats["literals_drawn"] += 1
            literal = self._draw_literal(rng, kept, effects, location)
            if literal is None:
                self.stats["literals_dropped"] += 1
                continue
            kept[_literal_slot(literal, self.pair_fluents)] = literal

    # Conditions ------------------------------------------------------------------------------

    def _static_satisfiable(self, et: PlaceEventType, condition: Sequence[Any]) -> bool:
        """Whether some able binding of the place event type (an agent able at some place)
        satisfies the condition's static literals; vacuously true for a never-able one."""
        able = self.places.requirement[et.label].any(axis=1)
        if not able.any():
            return True
        mask = np.ones(self.entity_values.shape[0], dtype=bool)
        for lit in condition:
            if isinstance(lit, ConditionLiteral) and lit.kind == "feature":
                column = self.entity_values[:, self.feature_position[lit.symbol]]
                mask &= column == int(lit.value)
        return bool((able & mask).any())

    def _condition_ok(
        self,
        et: PlaceEventType,
        effect: Effect,
        condition: tuple[AnyConditionLiteral, ...],
        kept: Mapping[tuple, AnyLiteral],
        achievable: set[tuple[str, bool]],
    ) -> bool:
        """The rules of a drawn condition (those of ``_Dynamics._condition_ok``, over the agent):
        no symbol twice; a fluent literal achievable, not on a fluent the precondition fixes,
        and not on the fluent the effect sets; a numeric literal achievable and not on a fluent
        the precondition fixes; a pair literal achievable, naming no pair or row the
        precondition names, and not fixed by the precondition; and some able binding
        satisfying the static literals."""
        keys: set[tuple] = set()
        pair_literals = [lit for lit in kept.values() if isinstance(lit, PairLiteral)]
        names = {
            name
            for lit in pair_literals
            for name in pair_literal_names(lit, self.pair_fluents[lit.pair_fluent])
        }
        fixed = fixed_pair_literals(pair_literals, self.pair_fluents, ROLES)
        adds, deletes = self.adds_and_deletes()
        for lit in condition:
            if isinstance(lit, PairLiteral):
                fluent = self.pair_fluents[lit.pair_fluent]
                key = pair_literal_key(lit, fluent)
                if (
                    not pair_literal_achievable(lit, fluent, adds, deletes)
                    or pair_literal_names(lit, fluent) & names
                    or key in fixed
                ):
                    return False
            else:
                key = (lit.role, lit.symbol)
                if lit.kind == "numeric":
                    assert lit.threshold is not None
                    if key in kept or not self.numeric.achievable(
                        lit.symbol, lit.value, lit.threshold, self.signs()
                    ):
                        return False
                elif lit.kind == "fluent":
                    if (
                        (lit.symbol, lit.value) not in achievable
                        or key in kept
                        or key == (effect.role, effect.fluent)
                    ):
                        return False
            if key in keys:
                return False
            keys.add(key)
        return self._static_satisfiable(et, condition)

    def _draw_condition(
        self,
        rng: np.random.Generator,
        et: PlaceEventType,
        effect: Effect,
        kept: Mapping[tuple, AnyLiteral],
        location: str,
    ) -> tuple[AnyConditionLiteral, ...] | None:
        """A condition for the effect, or None after ``MAX_TRIES`` rejected draws: the number
        of literals from ``condition_literals``, each literal's kind by the weights
        ``condition_kinds`` over the kinds the place event type can take (a static feature, a
        Boolean fluent, or a numeric threshold of the agent, or a pair literal of
        ``random_place_pair_literal``), its symbol uniformly within the kind, and its value
        uniformly."""
        cfg = self.effects_config
        pools: dict[str, Sequence[str]] = {
            "feature": self.feature_labels,
            "fluent": self.all_labels,
            "numeric": self.numeric_labels,
        }
        weights = {
            kind: weight
            for kind, weight in cfg.condition_kinds.items()
            if weight > 0 and (pools.get(kind) or (kind == "pair" and self.pair_drawable))
        }
        if not weights:
            return None
        fractions = self.preconditions_config.numeric_thresholds
        achievable = self.achievable()
        for _ in range(MAX_TRIES):
            count = _draw(rng, cfg.condition_literals)
            literals: list[AnyConditionLiteral] = []
            for _ in range(count):
                kind = next(iter(weights)) if len(weights) == 1 else _draw(rng, weights)
                if kind == "numeric":
                    symbol = self.numeric_labels[int(rng.integers(len(self.numeric_labels)))]
                    above = bool(rng.integers(2) == 0)
                    fraction = fractions[int(rng.integers(len(fractions)))]
                    literals.append(
                        ConditionLiteral(
                            "agent", symbol, above, self.numeric[symbol].point(fraction)
                        )
                    )
                elif kind == "pair":
                    self.stats["pair_conditions_drawn"] += 1
                    literals.append(random_place_pair_literal(rng, self.pair_fluents, location))
                else:
                    pool = pools[kind]
                    symbol = pool[int(rng.integers(len(pool)))]
                    literals.append(ConditionLiteral("agent", symbol, bool(rng.integers(2))))
            condition = tuple(literals)
            if self._condition_ok(et, effect, condition, kept, achievable):
                return condition
            self.stats["conditions_redrawn"] += 1
        return None

    # One place event type --------------------------------------------------------------------

    def draw(self, et: PlaceEventType, rng: np.random.Generator) -> PlaceEventType:
        """The place event type with its dynamics drawn from ``rng``: the kind, the location
        fluent, the effects (a move's add first), the further literals, and the conditions, in
        that order; the compiled ``move`` literal or the ``act`` literal last in the
        precondition."""
        cfg = self.effects_config
        self.stats["drawn"] += 1
        move = bool(rng.random() < self.move_share)
        location = self.locations[int(rng.integers(len(self.locations)))]
        fluent = self.pair_fluents[location]
        self.stats["moves" if move else "acts"] += 1
        effects: dict[tuple, Effect] = {}
        if move:
            add = Effect("agent", location, True, (), "patient")
            effects[pair_effect_key(add, fluent)] = add
            fixed = compiled_literals(add, fluent)[0]
        else:
            fixed = PairLiteral(PAIR_KIND, location, "agent", "patient", True, ACT_ORIGIN)
        if self.base_labels or self.numeric_labels:
            n = _draw(rng, cfg.count) - (1 if move else 0)
            for _ in range(max(n, 0)):
                self.stats["effects_drawn"] += 1
                effect = self._draw_effect(rng, set(effects))
                if effect is None:
                    self.stats["effects_dropped_no_slot"] += 1
                    continue
                effects[(effect.role, effect.fluent)] = effect
        kept: dict[tuple, AnyLiteral] = {pair_literal_key(fixed, fluent): fixed}
        self.draw_literals(rng, kept, tuple(effects.values()), location)
        if cfg.conditional_share > 0 and (self.base_labels or self.numeric_labels):
            for key, effect in list(effects.items()):
                if effect.pair:
                    continue
                self.stats["conditions_drawn"] += 1
                if rng.random() >= cfg.conditional_share:
                    continue
                condition = self._draw_condition(rng, et, effect, kept, location)
                if condition is None:
                    self.stats["conditions_dropped"] += 1
                    continue
                effects[key] = Effect(effect.role, effect.fluent, effect.value, condition)
        self.effects.extend(effects.values())
        return dataclasses.replace(
            et,
            precondition=_ordered_precondition(kept, fixed, fluent),
            effects=tuple(effects.values()),
        )

    def redraw(self, et: PlaceEventType, rng: np.random.Generator) -> PlaceEventType:
        """The place event type with its further precondition literals drawn again from
        ``rng`` (its own part of ``world:place_dynamics``), keeping its kind, its location
        fluent, its location literal, and its effects with their conditions."""
        location = et.location
        assert location is not None
        fluent = self.pair_fluents[location]
        fixed = next(
            lit
            for lit in et.precondition
            if isinstance(lit, PairLiteral)
            and lit.kind == PAIR_KIND
            and lit.pair_fluent == location
        )
        kept: dict[tuple, AnyLiteral] = {pair_literal_key(fixed, fluent): fixed}
        self.draw_literals(rng, kept, et.effects, location)
        return dataclasses.replace(et, precondition=_ordered_precondition(kept, fixed, fluent))


def _ordered_precondition(
    kept: Mapping[tuple, AnyLiteral], fixed: PairLiteral, fluent
) -> tuple[AnyLiteral, ...]:
    """The further literals in draw order, then the location literal last."""
    key = pair_literal_key(fixed, fluent)
    return tuple(lit for k, lit in kept.items() if k != key) + (fixed,)


def _drawer(
    places: Places,
    config: Config,
    fluents: Fluents,
    numeric: NumericFluents | None,
    pair_fluents: PairFluents | None,
    features: FeatureSet,
    entity_values: np.ndarray,
    initial_present: set[tuple[str, bool]],
    effects: Sequence[Effect],
) -> _PlaceDrawer:
    return _PlaceDrawer(
        config,
        fluents,
        numeric if numeric is not None else empty_numeric(0),
        pair_fluents if pair_fluents is not None else empty_pair_fluents(),
        features,
        entity_values,
        places,
        initial_present,
        effects,
    )


def draw_place_dynamics(
    places: Places | None,
    config: Config,
    rng: np.random.Generator,
    fluents: Fluents,
    numeric: NumericFluents | None,
    pair_fluents: PairFluents | None,
    features: FeatureSet,
    entity_values: np.ndarray,
    initial_present: set[tuple[str, bool]],
    entity_effects: Sequence[Effect],
) -> Places | None:
    """The places with the dynamics of every place event type without an event-file entry
    drawn from ``rng`` (``world:place_dynamics``), in label order (stage c3b); nothing is drawn
    in a world without location fluents, or without places. ``entity_effects`` are the effects
    of the entity event types; the explicit entries' effects are read off ``places``. The
    counts of the draws go to ``Places.dynamics_stats``."""
    if places is None or pair_fluents is None or not pair_fluents.locations:
        return places
    explicit = [e for et in places.event_types for e in et.effects]
    drawer = _drawer(
        places,
        config,
        fluents,
        numeric,
        pair_fluents,
        features,
        entity_values,
        initial_present,
        list(entity_effects) + explicit,
    )
    event_types = tuple(et if et.performable else drawer.draw(et, rng) for et in places.event_types)
    return dataclasses.replace(places, event_types=event_types, dynamics_stats=dict(drawer.stats))


def redraw_place_preconditions(
    places: Places,
    labels: Sequence[str],
    config: Config,
    parts: Mapping[str, np.random.Generator],
    fluents: Fluents,
    numeric: NumericFluents | None,
    pair_fluents: PairFluents | None,
    features: FeatureSet,
    entity_values: np.ndarray,
    initial_present: set[tuple[str, bool]],
    entity_effects: Sequence[Effect],
) -> Places:
    """The places with the further precondition literals of the named place event types drawn
    again, each from its own generator (the part ``world:place_dynamics:<label>``), as
    ``define`` redraws a place event type that was never legal in the statistics episodes; the
    kind, the location fluent, the location literal, and the effects with their conditions
    stay, and no other place event type changes."""
    all_effects = list(entity_effects) + [e for et in places.event_types for e in et.effects]
    drawer = _drawer(
        places,
        config,
        fluents,
        numeric,
        pair_fluents,
        features,
        entity_values,
        initial_present,
        all_effects,
    )
    redrawn = {label: drawer.redraw(places.event_type(label), parts[label]) for label in labels}
    return dataclasses.replace(
        places, event_types=tuple(redrawn.get(et.label, et) for et in places.event_types)
    )


__all__ = [
    "PLACE_DYNAMICS_STATS",
    "apply_place_dynamics",
    "check_place_dynamics",
    "draw_place_dynamics",
    "place_produced",
    "place_signs",
    "random_place_pair_literal",
    "redraw_place_preconditions",
]
