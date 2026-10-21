"""Conformance fixtures: the brute-force evaluator, the fixture format, checking, and generating.

A fixture is one JSON file in ``tests/fixtures/world/`` (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Conformance fixtures"). It holds a complete definition inline, the entities' static facts, the
initial state, the steps with what must come out of each, and the error that ``apply`` must
raise, if any. ``tests/fixtures/world/README.md`` is the format's description for whoever builds
the Rust runtime.

The brute-force evaluator here is written apart from the runtime. It reads the definition
record's truth tables, one entity or binding at a time, in plain Python, and never touches the
matrices. The generated fixtures take their expected values from it, and the runtime is tested
against it on random states. Since stage b2 a state has a numeric part (``{entity: {fluent:
value}}``), the evaluator reads numeric thresholds, applies numeric effects and the drift in the
order ``clamp(clamp(v + e) + d)`` with plain Python floats, and judges the comparisons of
"Phase (b): the world" by direct arithmetic; a fixture may give ``initial_numeric``, a
``numeric`` record after each step, and a list of ``comparisons`` with their expected values.
Since stage c2a a state has a pair part (``{pair fluent: frozenset of (first, second) label
pairs}``), which the evaluator reads directly, never through the matrices: the pair literals of
rules, preconditions, and conditions; the pair effects, both directions of a symmetric pair;
the pairs, rows, and columns an event reads for non-interference. A fixture of a world with pair
fluents gives ``initial_pairs`` beside ``initial`` and ``pairs`` after each step, as ``[pair
fluent, first, second]`` triples. Since stage c3a a location fluent's pairs name a place second:
the evaluator reads them directly as the others, applies an add on a location fluent as the
total functional replacement, judges the total profile, and counts a move's row (every pair
of the entity to every place) as its write and a place's column (every pair of every entity to
the place) as what ``SOMEOF`` of the place reads; a fixture of a world with location fluents
gives ``initial_places`` beside ``initial_pairs``, and its ``pairs`` after each step list the
location pairs too.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semantic_world.world.definition import (
    Definition,
    RuntimeDefinition,
    entities_frame,
    runtime_definition,
)
from semantic_world.world.errors import StepError, WorldError
from semantic_world.world.identity import read_json, rule_set_id, to_json, write_json
from semantic_world.world.locations import location_columns
from semantic_world.world.numeric import clamp
from semantic_world.world.pair_fluents import initial_pairs_frame
from semantic_world.world.places import places_frame
from semantic_world.world.runtime import (
    ENTITY,
    RANGE,
    Event,
    Mean,
    State,
    above,
    apply,
    below,
    check_definition_agreement,
    decrease,
    derive,
    greater,
    increase,
    legal_bindings,
)

FIXTURE_VERSION = 1
FIXTURES_DIR = Path("tests/fixtures/world")
ERROR_KINDS = ("illegal", "interference")
COMPARISON_KINDS = ("greater", "above", "below", "increase", "decrease")

# ---------------------------------------------------------------------------------------------
# The brute-force evaluator
# ---------------------------------------------------------------------------------------------

BruteState = dict[str, frozenset[str]]
"""A state for the evaluator: each entity's set of true base fluents."""
BruteNumeric = dict[str, dict[str, float]]
"""The numeric part of a state: each entity's value of each numeric fluent."""
BrutePairs = dict[str, frozenset[tuple[str, str]]]
"""The pair part of a state (stage c2a): for each pair fluent, the ordered pairs that hold, by
labels; both directions of a symmetric pair; a place second for a location fluent (stage c3a)."""
BrutePlaces = dict[str, dict[str, str]]
"""The initial places (stage c3a): for each entity, its place on each location fluent."""
PAIR_KINDS = ("pair", "pair_some", "pair_someof", "pair_same")


@dataclass(frozen=True)
class BruteEvent:
    event_type: str
    binding: dict[str, str]
    """Role to entity label."""

    def key(self) -> tuple:
        return (self.event_type, tuple(sorted(self.binding.items())))


class BruteForce:
    """The reference semantics, by truth-table lookup over the definition record.

    Every rule is evaluated by looking up its truth table at the setting of its inputs, with the
    first input as the most significant bit. Features, thresholds, fluents, comparisons, and
    constraints are evaluated by their definitions, recursively, with no caching across states.
    """

    def __init__(
        self,
        record: Mapping[str, Any],
        entities: Sequence[Mapping[str, Any]],
        places: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.record = record
        self.symbols = {s["label"]: s for s in record["symbols"]}
        self.literals = list(record["literals"])
        self.rules = {r["output"]: r for r in record["rules"]}
        self.event_types = {e["label"]: e for e in record["event_types"]}
        self.entities = {str(row["label"]): row for row in entities}
        self.entity_labels = [str(row["label"]) for row in entities]
        self.places = {str(row["label"]): row for row in places}
        """The places' static facts (stage c1), by label; a place event type's patient is one
        of them."""
        self.place_labels = [str(row["label"]) for row in places]
        self.base_fluents = [
            s["label"] for s in record["symbols"] if s["kind"] == "fluent" and not s["derived"]
        ]
        self.derived_fluents = [
            s["label"] for s in record["symbols"] if s["kind"] == "fluent" and s["derived"]
        ]
        self.derived_features = [
            s["label"]
            for s in record["symbols"]
            if s["kind"] in ("property", "part") and s["derived"]
        ]
        self.numeric_fluents = [
            s["label"] for s in record["symbols"] if s["kind"] == "numeric_fluent"
        ]
        self.numeric_range = {
            s["label"]: (float(s["range"][0]), float(s["range"][1]))
            for s in record["symbols"]
            if s["kind"] == "numeric_fluent"
        }
        self.numeric_drift = {
            s["label"]: float(s.get("drift", 0.0))
            for s in record["symbols"]
            if s["kind"] == "numeric_fluent"
        }
        self.scalars = [s["label"] for s in record["symbols"] if s["kind"] == "scalar"]
        self.pair_fluents = [s["label"] for s in record["symbols"] if s["kind"] == "pair_fluent"]
        self.pair_profile = {s["label"]: s for s in record["symbols"] if s["kind"] == "pair_fluent"}
        self.location_fluents = [f for f in self.pair_fluents if self.pair_profile[f].get("total")]
        self._numeric: BruteNumeric | None = None
        """The numeric part of the state being evaluated (set by the operations)."""
        self._pairs: BrutePairs | None = None
        """The pair part of the state being evaluated (stage c2a; set by the operations)."""

    # Rules and literals ---------------------------------------------------------------------

    def rule(self, output: str, roles: Mapping[str, str], state: BruteState | None) -> int:
        rule = self.rules[output]
        index = 0
        for i in rule["inputs"]:
            index = (index << 1) | self.literal(self.literals[i], roles, state)
        return int(rule["truth_table"][index])

    def _value(self, entity: str, fluent: str) -> float:
        if self._numeric is None:
            raise WorldError(f"the evaluator reads {fluent} of {entity} without numeric values")
        return float(self._numeric[entity][fluent])

    def literal(
        self, literal: Mapping[str, Any], roles: Mapping[str, str], state: BruteState | None
    ) -> int:
        kind = literal["kind"]
        if kind == "comparison":
            agent = float(self._row(roles["agent"])[literal["agent_scalar"]])
            patient = float(self._row(roles["patient"])[literal["patient_scalar"]])
            difference = agent - patient
            holds = difference > literal["low"]
            if literal["high"] is not None:
                holds = holds and difference < literal["high"]
            return int(holds)
        if kind == "constraint":
            return self.rule(literal["constraint"], roles, state)
        entity = roles[literal["role"] if literal["role"] is not None else "entity"]
        if kind == "feature":
            return self.feature(literal["feature"], entity)
        if kind == "threshold":
            return int(float(self._row(entity)[literal["scalar"]]) > literal["threshold"])
        if kind == "fluent":
            assert state is not None
            return self.fluent(literal["fluent"], entity, state)
        if kind == "numeric_threshold":
            return int(self._value(entity, literal["fluent"]) > literal["threshold"])
        if kind in PAIR_KINDS:
            return int(self.pair_literal(literal, roles))
        raise WorldError(f"unknown literal kind {kind!r}")

    # Pairs (stage c2a) ----------------------------------------------------------------------

    def _pair_set(self, fluent: str) -> frozenset[tuple[str, str]]:
        if self._pairs is None:
            raise WorldError(f"the evaluator reads {fluent} without the state's pairs")
        return self._pairs[fluent]

    def symmetric(self, fluent: str) -> bool:
        return self.pair_profile[fluent]["symmetry"] == "symmetric"

    def total(self, fluent: str) -> bool:
        """A location fluent (stage c3a): its pairs go from entities to places."""
        return bool(self.pair_profile[fluent].get("total"))

    def is_place_event_type(self, label: str) -> bool:
        return self.event_types[label]["kind"] == "place_event_type"

    def pair_literal(self, literal: Mapping[str, Any], roles: Mapping[str, str]) -> bool:
        """A pair literal of a rule (``role`` null: the entity itself), a precondition, or a
        condition, read directly from the pair sets: ``pair`` holds when the ordered pair is in
        the set (a symmetric pair fluent holds both directions); ``pair_some`` when some pair
        starts at the entity; ``pair_someof`` when some pair ends at it; ``pair_same`` when the
        two entities have a common partner. The literal's ``value`` (absent in the literal
        table, where it is the positive literal) negates it when false."""
        pairs = self._pair_set(literal["pair_fluent"])
        kind = literal["kind"]
        if "role" in literal and "first" not in literal:
            first = roles[literal["role"] if literal["role"] is not None else "entity"]
        else:
            first = roles[literal["first"]]
        if kind == "pair":
            positive = (first, roles[literal["second"]]) in pairs
        elif kind == "pair_some":
            positive = any(a == first for a, _ in pairs)
        elif kind == "pair_someof":
            positive = any(b == first for _, b in pairs)
        else:
            second = roles[literal["second"]]
            partners_of_first = {b for a, b in pairs if a == first}
            partners_of_second = {b for a, b in pairs if a == second}
            positive = bool(partners_of_first & partners_of_second)
        value = literal.get("value", True)
        return positive if value else not positive

    def _row(self, label: str) -> Mapping[str, Any]:
        """The static facts of an entity, or of a place."""
        row = self.entities.get(label)
        if row is None:
            row = self.places[label]
        return row

    def feature(self, label: str, entity: str) -> int:
        if self.symbols[label]["derived"]:
            return self.rule(label, {"entity": entity}, None)
        return int(self._row(entity)[label])

    def fluent(self, label: str, entity: str, state: BruteState) -> int:
        if self.symbols[label]["derived"]:
            return self.rule(label, {"entity": entity}, state)
        return int(label in state[entity])

    # The operations -------------------------------------------------------------------------

    def with_numeric(self, numeric: BruteNumeric | None) -> BruteForce:
        """Set the numeric part of the state the next operations evaluate (None keeps the one
        set before); a world without numeric fluents needs none, and an operation of a world
        with them fails without one."""
        if numeric is not None:
            self._numeric = numeric
        return self

    def with_pairs(self, pairs: BrutePairs | None) -> BruteForce:
        """Set the pair part of the state the next operations evaluate (None keeps the one set
        before; stage c2a)."""
        if pairs is not None:
            self._pairs = pairs
        return self

    def derived_static(self, entity: str) -> list[str]:
        return [f for f in self.derived_features if self.feature(f, entity)]

    def derived_fluent(
        self,
        entity: str,
        state: BruteState,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> list[str]:
        self.with_numeric(numeric).with_pairs(pairs)
        if self.numeric_fluents and self._numeric is None:
            raise WorldError("the world has numeric fluents, so an operation needs their values")
        if self.pair_fluents and self._pairs is None:
            raise WorldError("the world has pair fluents, so an operation needs their pairs")
        return [f for f in self.derived_fluents if self.fluent(f, entity, state)]

    def able(self, event: BruteEvent) -> bool:
        et = self.event_types[event.event_type]
        return bool(self.rule(et["requirement"]["output"], event.binding, None))

    def _threshold_holds(self, lit: Mapping[str, Any], value: float) -> bool:
        return value > lit["threshold"] if lit["operator"] == ">" else value <= lit["threshold"]

    def precondition_holds(
        self,
        event: BruteEvent,
        state: BruteState,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> bool:
        self.with_numeric(numeric).with_pairs(pairs)
        et = self.event_types[event.event_type]
        for lit in et["precondition"]["literals"]:
            if lit.get("kind") in PAIR_KINDS:
                if not self.pair_literal(lit, event.binding):
                    return False
                continue
            entity = event.binding[lit["role"]]
            if "threshold" in lit:
                if not self._threshold_holds(lit, self._value(entity, lit["fluent"])):
                    return False
                continue
            value = self.fluent(lit["fluent"], entity, state)
            if value != int(lit["value"]):
                return False
        return True

    def legal(
        self,
        event: BruteEvent,
        state: BruteState,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> bool:
        return self.able(event) and self.precondition_holds(event, state, numeric, pairs)

    def bindings(
        self, event_type: str, entities: Sequence[str] | None = None
    ) -> list[dict[str, str]]:
        """Every binding of the event type among the entities, in the fixed order: by agent,
        then patient, in entity order."""
        et = self.event_types[event_type]
        pool = [e for e in self.entity_labels if entities is None or e in set(entities)]
        if et["arity"] == 1:
            return [{"agent": e} for e in pool]
        if et["kind"] == "place_event_type":
            return [{"agent": a, "patient": p} for a in pool for p in self.place_labels]
        return [{"agent": a, "patient": p} for a in pool for p in pool if a != p]

    def legal_bindings(
        self,
        event_type: str,
        state: BruteState,
        entities: Sequence[str] | None = None,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> list[dict[str, str]]:
        self.with_numeric(numeric).with_pairs(pairs)
        return [
            b
            for b in self.bindings(event_type, entities)
            if self.legal(BruteEvent(event_type, b), state)
        ]

    def performable(self) -> list[str]:
        """The event types that have events: every event type that is not a category, and
        (stage c3a) every place event type with a precondition or effects (a move or an act)."""
        return [
            label
            for label, e in self.event_types.items()
            if e["kind"] == "event_type"
            or (e["kind"] == "place_event_type" and (e["precondition"]["literals"] or e["effects"]))
        ]

    def cone(self, fluent: str) -> set[str]:
        """The base fluents, Boolean or numeric, a fluent depends on."""
        if not self.symbols[fluent]["derived"]:
            return {fluent}
        found: set[str] = set()
        for i in self.rules[fluent]["inputs"]:
            literal = self.literals[i]
            if literal["kind"] in ("fluent", "numeric_threshold"):
                found |= self.cone(literal["fluent"])
        return found

    def pair_cone(self, fluent: str) -> set[tuple[str, str]]:
        """The rows and columns of its own entity a fluent reads through the pair literals of
        its rule, directly or through other derived fluents: ``("row", pair fluent)`` for
        ``pair_some`` and ``("column", pair fluent)`` for ``pair_someof``."""
        if not self.symbols[fluent]["derived"]:
            return set()
        found: set[tuple[str, str]] = set()
        for i in self.rules[fluent]["inputs"]:
            literal = self.literals[i]
            if literal["kind"] == "pair_some":
                found.add(("row", literal["pair_fluent"]))
            elif literal["kind"] == "pair_someof":
                found.add(("column", literal["pair_fluent"]))
            elif literal["kind"] in ("fluent", "numeric_threshold"):
                found |= self.pair_cone(literal["fluent"])
        return found

    @staticmethod
    def _effect_value(effect: Mapping[str, Any]) -> bool | float:
        """A Boolean effect's value, or a numeric effect's signed constant."""
        if "amount" in effect:
            amount = float(effect["amount"])
            return amount if effect["operator"] == "+=" else -amount
        return bool(effect["value"])

    def _pair_effect_keys(
        self, effect: Mapping[str, Any], binding: Mapping[str, str]
    ) -> list[tuple[str, str, str]]:
        """The ordered pairs a pair effect writes: the pair between its roles, and its reverse
        for a symmetric pair fluent, as ``(pair fluent, first, second)``."""
        fluent = effect["pair_fluent"]
        first, second = binding[effect["first"]], binding[effect["second"]]
        keys = [(fluent, first, second)]
        if self.symmetric(fluent):
            keys.append((fluent, second, first))
        return keys

    def writes(self, event: BruteEvent) -> dict[tuple, bool | float]:
        """What the event may write: every effect, whatever its condition: ``(entity,
        fluent)`` for a Boolean or numeric fluent, ``(pair fluent, first, second)`` for a pair
        (both directions of a symmetric one; for an add on a location fluent, stage c3a, every
        pair of the agent's row, because the old place goes with the new one)."""
        et = self.event_types[event.event_type]
        found: dict[tuple, bool | float] = {}
        for e in et["effects"]:
            if e.get("kind") == "pair":
                if self.total(e["pair_fluent"]):
                    for key in self._row_pairs(e["pair_fluent"], event.binding[e["first"]]):
                        found[key] = key[2] == event.binding[e["second"]]
                    continue
                for key in self._pair_effect_keys(e, event.binding):
                    found[key] = bool(e["value"])
            else:
                found[(event.binding[e["role"]], e["fluent"])] = self._effect_value(e)
        return found

    def condition_holds(
        self,
        effect: Mapping[str, Any],
        binding: Mapping[str, str],
        state: BruteState,
        numeric: BruteNumeric | None = None,
    ) -> bool:
        """Whether an effect's condition holds for the binding in the state: each literal's
        symbol, a static feature or a fluent of the literal's role, has the literal's value, or
        a numeric fluent is on the literal's side of its threshold. An effect without a
        condition holds."""
        if numeric is not None:
            self.with_numeric(numeric)
        for lit in (effect.get("condition") or {}).get("literals", ()):
            if lit["kind"] in PAIR_KINDS:
                if not self.pair_literal(lit, binding):
                    return False
                continue
            entity = binding[lit["role"]]
            if lit["kind"] == "numeric":
                if not self._threshold_holds(lit, self._value(entity, lit["symbol"])):
                    return False
                continue
            if lit["kind"] == "feature":
                value = self.feature(lit["symbol"], entity)
            else:
                value = self.fluent(lit["symbol"], entity, state)
            if value != int(lit["value"]):
                return False
        return True

    def fired(
        self,
        event: BruteEvent,
        state: BruteState,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> dict[tuple, bool | float]:
        """What the event writes in the state: its effects whose conditions hold there."""
        self.with_numeric(numeric).with_pairs(pairs)
        et = self.event_types[event.event_type]
        found: dict[tuple, bool | float] = {}
        for e in et["effects"]:
            if not self.condition_holds(e, event.binding, state):
                continue
            if e.get("kind") == "pair":
                for key in self._pair_effect_keys(e, event.binding):
                    found[key] = bool(e["value"])
            else:
                found[(event.binding[e["role"]], e["fluent"])] = self._effect_value(e)
        return found

    def _row_pairs(self, fluent: str, entity: str) -> set[tuple[str, str, str]]:
        """Every pair of the entity's row: to every other entity, or (a location fluent) to
        every place."""
        if self.total(fluent):
            return {(fluent, entity, q) for q in self.place_labels}
        return {(fluent, entity, z) for z in self.entity_labels if z != entity}

    def _column_pairs(self, fluent: str, target: str) -> set[tuple[str, str, str]]:
        """Every pair of the column of an entity, or (a location fluent) of a place."""
        if self.total(fluent):
            return {(fluent, z, target) for z in self.entity_labels}
        return {(fluent, z, target) for z in self.entity_labels if z != target}

    def _pair_literal_reads(
        self, lit: Mapping[str, Any], binding: Mapping[str, str]
    ) -> set[tuple[str, str, str]]:
        """The ordered pairs a pair literal reads, every pair of a row or a column counted: the
        pair (both directions of a symmetric one) for ``pair``, the row for ``pair_some``, the
        column for ``pair_someof``, both rows for ``pair_same``."""
        fluent = lit["pair_fluent"]
        if lit["kind"] == "pair":
            first, second = binding[lit["first"]], binding[lit["second"]]
            found = {(fluent, first, second)}
            if self.symmetric(fluent):
                found.add((fluent, second, first))
            return found
        if lit["kind"] == "pair_some":
            return self._row_pairs(fluent, binding[lit["role"]])
        if lit["kind"] == "pair_someof":
            return self._column_pairs(fluent, binding[lit["role"]])
        return self._row_pairs(fluent, binding[lit["first"]]) | self._row_pairs(
            fluent, binding[lit["second"]]
        )

    def _fluent_reads(self, fluent: str, entity: str) -> set[tuple]:
        """What a literal on a fluent of an entity reads: the base fluents of its cone, and the
        rows and columns of the entity its rule reads through pair literals."""
        found: set[tuple] = {(entity, base) for base in self.cone(fluent)}
        for kind, pair_fluent in self.pair_cone(fluent):
            found |= (
                self._row_pairs(pair_fluent, entity)
                if kind == "row"
                else self._column_pairs(pair_fluent, entity)
            )
        return found

    def reads(self, event: BruteEvent) -> set[tuple]:
        """The facts the precondition and the effects' conditions read, through the cones of
        derived fluents: base fluents (Boolean or numeric) of an entity, and ordered pairs (a
        pair, or every pair of a row or a column); a static literal reads nothing an event can
        write."""
        et = self.event_types[event.event_type]
        found: set[tuple] = set()
        for lit in et["precondition"]["literals"]:
            if lit.get("kind") in PAIR_KINDS:
                found |= self._pair_literal_reads(lit, event.binding)
                continue
            found |= self._fluent_reads(lit["fluent"], event.binding[lit["role"]])
        for effect in et["effects"]:
            for lit in (effect.get("condition") or {}).get("literals", ()):
                if lit["kind"] in PAIR_KINDS:
                    found |= self._pair_literal_reads(lit, event.binding)
                    continue
                if lit["kind"] == "feature":
                    continue
                found |= self._fluent_reads(lit["symbol"], event.binding[lit["role"]])
        return found

    def interfere(self, a: BruteEvent, b: BruteEvent) -> bool:
        if a.key() == b.key():
            return True
        wa, wb = set(self.writes(a)), set(self.writes(b))
        return bool(wa & wb) or bool(wa & self.reads(b)) or bool(wb & self.reads(a))

    def initial_numeric(self) -> BruteNumeric:
        """Every entity's initial numeric values, from the entities table (which the fixtures
        give apart, under ``initial_numeric``)."""
        return {
            e: {f: float(self.entities[e][f]) for f in self.numeric_fluents}
            for e in self.entity_labels
        }

    def empty_pairs(self) -> BrutePairs:
        return {f: frozenset() for f in self.pair_fluents}

    def check_profiles(self, pairs: BrutePairs) -> str | None:
        """Why the pair sets break a profile, or None: a pair of an entity with itself, a
        symmetric pair without its reverse, an asymmetric pair with its reverse, two pairs in
        a row of a functional pair fluent, or (a location fluent, stage c3a) an entity at no
        place or at a label that is not a place."""
        for fluent, held in pairs.items():
            profile = self.pair_profile[fluent]
            if self.total(fluent):
                for a, b in held:
                    if a not in self.entities or b not in self.places:
                        return (
                            f"{fluent} holds between {a} and {b}, which are not an entity and "
                            f"a place"
                        )
                for entity in self.entity_labels:
                    count = sum(1 for a, _ in held if a == entity)
                    if count != 1:
                        return (
                            f"{fluent} is total and functional, but {entity} is at {count} places"
                        )
                continue
            for a, b in held:
                if a == b:
                    return f"{fluent} holds between {a} and itself"
                if profile["symmetry"] == "symmetric" and (b, a) not in held:
                    return f"{fluent} is symmetric, but ({a}, {b}) holds without ({b}, {a})"
                if profile["symmetry"] == "asymmetric" and (b, a) in held:
                    return f"{fluent} is asymmetric, but ({a}, {b}) and ({b}, {a}) both hold"
            if profile["functional"]:
                for entity in self.entity_labels:
                    if sum(1 for a, _ in held if a == entity) > 1:
                        return f"{fluent} is functional, but the row of {entity} holds two pairs"
        return None

    def step(
        self,
        state: BruteState,
        numeric: BruteNumeric | None,
        events: Sequence[BruteEvent],
        pairs: BrutePairs | None = None,
    ) -> tuple[BruteState, BruteNumeric, BruteNumeric]:
        """The next state and numeric values, and the numeric values after the effects and
        before the drift. The caller has checked legality and non-interference. Every condition
        is judged in ``state``, before any effect of the step; each numeric value becomes
        ``clamp(clamp(v + e) + d)``, in plain Python floats. The pairs after the step are read
        back with ``pairs_after``."""
        numeric = numeric if numeric is not None else self.initial_numeric()
        pairs = pairs if pairs is not None else (self._pairs or self.empty_pairs())
        self.with_numeric(numeric).with_pairs(pairs)
        new = {e: set(v) for e, v in state.items()}
        new_pairs = {f: set(v) for f, v in pairs.items()}
        after_effects = {e: dict(v) for e, v in numeric.items()}
        for event in events:
            for key, value in self.fired(event, state).items():
                if len(key) == 3:
                    fluent, first, second = key
                    if value:
                        if self.total(fluent):
                            # the total functional replacement (stage c3a): the old place goes
                            new_pairs[fluent] = {p for p in new_pairs[fluent] if p[0] != first}
                        new_pairs[fluent].add((first, second))
                    else:
                        new_pairs[fluent].discard((first, second))
                    continue
                entity, fluent = key
                if fluent in self.numeric_range:
                    low, high = self.numeric_range[fluent]
                    after_effects[entity][fluent] = clamp(
                        after_effects[entity][fluent] + float(value), low, high
                    )
                elif value:
                    new[entity].add(fluent)
                else:
                    new[entity].discard(fluent)
        final = {
            e: {
                f: clamp(v + self.numeric_drift[f], *self.numeric_range[f])
                for f, v in values.items()
            }
            for e, values in after_effects.items()
        }
        self._pairs_after: BrutePairs = {f: frozenset(v) for f, v in new_pairs.items()}
        violation = self.check_profiles(self._pairs_after)
        if violation is not None:
            raise WorldError(f"after the step a profile does not hold: {violation}")
        return {e: frozenset(v) for e, v in new.items()}, final, after_effects

    @property
    def pairs_after(self) -> BrutePairs:
        """The pairs after the last ``step``."""
        return self._pairs_after

    def apply(
        self, state: BruteState, events: Sequence[BruteEvent], numeric: BruteNumeric | None = None
    ) -> BruteState:
        """The next Boolean state (``step`` gives the numeric values too)."""
        return self.step(state, numeric, events)[0]

    def pairs_record(self, pairs: BrutePairs) -> list[list[str]]:
        """The pairs as a fixture writes them: ``[pair fluent, first, second]`` triples, by
        pair fluent in symbol order, then by first and second in entity order (the second of a
        location fluent's pair in place order)."""
        order = {label: i for i, label in enumerate(self.entity_labels)}
        place_order = {label: i for i, label in enumerate(self.place_labels)}
        return [
            [fluent, a, b]
            for fluent in self.pair_fluents
            for a, b in sorted(
                pairs[fluent],
                key=lambda p, f=fluent: (
                    order[p[0]],
                    place_order[p[1]] if self.total(f) else order[p[1]],
                ),
            )
        ]

    def pairs_from_record(
        self, rows: Sequence[Sequence[str]], places: BrutePlaces | None = None
    ) -> BrutePairs:
        """The pair sets from ``[pair fluent, first, second]`` triples and, for the location
        fluents (stage c3a), the places given as ``{entity: {location fluent: place}}``."""
        found: dict[str, set[tuple[str, str]]] = {f: set() for f in self.pair_fluents}
        for fluent, a, b in rows:
            found[str(fluent)].add((str(a), str(b)))
        for entity, at in (places or {}).items():
            for fluent, place in at.items():
                found[str(fluent)] = {p for p in found[str(fluent)] if p[0] != entity}
                found[str(fluent)].add((str(entity), str(place)))
        return {f: frozenset(v) for f, v in found.items()}

    def places_record(self, pairs: BrutePairs) -> BrutePlaces:
        """Every entity's place on every location fluent, as a fixture's ``initial_places``."""
        found: BrutePlaces = {}
        for fluent in self.location_fluents:
            for a, b in pairs[fluent]:
                found.setdefault(a, {})[fluent] = b
        return {e: found[e] for e in self.entity_labels if e in found}

    def state_record(self, state: BruteState) -> dict[str, list[str]]:
        return {e: [f for f in self.base_fluents if f in state[e]] for e in self.entity_labels}

    def numeric_record(self, numeric: BruteNumeric) -> dict[str, dict[str, float]]:
        return {
            e: {f: float(numeric[e][f]) for f in self.numeric_fluents} for e in self.entity_labels
        }

    def derived_record(
        self,
        state: BruteState,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> dict[str, list[str]]:
        self.with_numeric(numeric).with_pairs(pairs)
        return {e: self.derived_fluent(e, state) for e in self.entity_labels}

    def legal_record(
        self,
        state: BruteState,
        numeric: BruteNumeric | None = None,
        pairs: BrutePairs | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        self.with_numeric(numeric).with_pairs(pairs)
        return {label: self.legal_bindings(label, state) for label in self.performable()}

    # Comparisons ----------------------------------------------------------------------------

    def members(self, category: str) -> list[str]:
        if category == ENTITY:
            return list(self.entity_labels)
        return [
            e
            for e in self.entity_labels
            if self.entities[e]["leaf"] == category
            or str(self.entities[e]["leaf"]).startswith(category + ".")
        ]

    def dimension_value(
        self, dimension: str, x: str | Mapping[str, str], numeric: BruteNumeric | None
    ) -> float:
        """An argument's value on a scalar or a numeric fluent: an entity's, or the mean over
        a category (``{"mean": category}``)."""

        def of(entity: str) -> float:
            if dimension in self.numeric_range:
                assert numeric is not None
                return float(numeric[entity][dimension])
            return float(self.entities[entity][dimension])

        if isinstance(x, Mapping):
            rows = self.members(x["mean"])
            return sum(of(e) for e in rows) / len(rows)
        return of(x)

    def class_mean_sd(self, category: str, scalar: str) -> tuple[float, float]:
        values = [float(self.entities[e][scalar]) for e in self.members(category)]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return mean, sd

    def compare(
        self,
        entry: Mapping[str, Any],
        numeric: BruteNumeric | None = None,
        before: BruteNumeric | None = None,
    ) -> bool:
        """Judge a comparison entry of a fixture: ``greater`` (strict), ``above`` and ``below``
        (against a class's mean and standard deviation, or the point at the fraction ``z`` of
        a numeric fluent's range with ``RANGE``: ``above`` strictly greater than the point,
        ``below`` at most it, WM.35), ``increase`` and ``decrease`` (from ``before`` to
        ``numeric``)."""
        kind = entry["kind"]
        dimension = entry["dimension"]
        if kind == "greater":
            return self.dimension_value(dimension, entry["x"], numeric) > self.dimension_value(
                dimension, entry["y"], numeric
            )
        if kind in ("increase", "decrease"):
            assert before is not None and numeric is not None
            was, now = float(before[entry["x"]][dimension]), float(numeric[entry["x"]][dimension])
            return now > was if kind == "increase" else now < was
        value = self.dimension_value(dimension, entry["x"], numeric)
        z = float(entry["z"])
        if entry["class"] == RANGE:
            low, high = self.numeric_range[dimension]
            point = low + z * (high - low)
            return value > point if kind == "above" else value <= point
        mean, sd = self.class_mean_sd(entry["class"], dimension)
        if sd == 0:
            return False
        return value >= mean + z * sd if kind == "above" else value <= mean - z * sd


# ---------------------------------------------------------------------------------------------
# The fixture format
# ---------------------------------------------------------------------------------------------


def fixture_record(
    name: str,
    description: str,
    record: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    initial: Mapping[str, Sequence[str]],
    steps: Sequence[Mapping[str, Any]],
    error: Mapping[str, Any] | None,
    initial_numeric: Mapping[str, Mapping[str, float]] | None = None,
    comparisons: Sequence[Mapping[str, Any]] = (),
    places: Sequence[Mapping[str, Any]] | None = None,
    initial_pairs: Sequence[Sequence[str]] | None = None,
    initial_places: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Assemble a fixture, with the rule-set identity recomputed from the inline definition.
    ``initial_numeric`` and ``comparisons`` (stage b2), ``places`` (stage c1),
    ``initial_pairs`` (stage c2a), and ``initial_places`` (stage c3a) are written only when
    given."""
    identity = rule_set_id(
        record["symbols"], record["literals"], record["rules"], record["event_types"]
    )
    fixture: dict[str, Any] = {
        "version": FIXTURE_VERSION,
        "name": name,
        "description": description,
        "rule_set_id": identity,
        "definition": {**dict(record), "rule_set_id": identity},
        "entities": list(entities),
    }
    if places is not None:
        fixture["places"] = list(places)
    fixture["initial"] = dict(initial)
    if initial_numeric is not None:
        fixture["initial_numeric"] = {e: dict(v) for e, v in initial_numeric.items()}
    if initial_pairs is not None:
        fixture["initial_pairs"] = [list(triple) for triple in initial_pairs]
    if initial_places is not None:
        fixture["initial_places"] = {e: dict(at) for e, at in initial_places.items()}
    fixture["steps"] = list(steps)
    fixture["error"] = error
    if comparisons:
        fixture["comparisons"] = list(comparisons)
    return fixture


def write_fixture(path: str | Path, fixture: Mapping[str, Any]) -> Path:
    return write_json(path, fixture)


def read_fixture(path: str | Path) -> dict[str, Any]:
    return read_json(path)


# ---------------------------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------------------------


class FixtureError(WorldError):
    """A fixture does not pass: the runtime's result differs from the fixture's expectation,
    or the fixture itself is malformed."""


@dataclass(frozen=True)
class FixtureReport:
    name: str
    steps: int
    error: str | None
    """The kind of error the fixture expects, if any."""


def _fail(name: str, message: str) -> FixtureError:
    return FixtureError(f"fixture {name}: {message}")


def _binding_record(
    definition: RuntimeDefinition, roles: Sequence[str], binding: Sequence[int], place: bool = False
) -> dict[str, str]:
    """A binding by labels; the patient of a place event type (``place``) is a place."""
    return {
        role: definition.place_labels[i]
        if place and role == "patient"
        else definition.entity_labels[i]
        for role, i in zip(roles, binding, strict=True)
    }


def check_fixture(fixture: Mapping[str, Any]) -> FixtureReport:
    """Run the Python runtime on one fixture. Raises :class:`FixtureError` on the first
    disagreement, naming the step and what differed."""
    name = str(fixture.get("name", "?"))
    if fixture.get("version") != FIXTURE_VERSION:
        raise _fail(name, f"has version {fixture.get('version')!r}; expected {FIXTURE_VERSION}")
    for key in ("rule_set_id", "definition", "entities", "initial", "steps", "error"):
        if key not in fixture:
            raise _fail(name, f"has no {key}")
    record = fixture["definition"]
    identity = rule_set_id(
        record["symbols"], record["literals"], record["rules"], record["event_types"]
    )
    if identity != fixture["rule_set_id"]:
        raise _fail(
            name,
            f"records the rule set {fixture['rule_set_id']}, but its definition hashes to "
            f"{identity}",
        )
    try:
        definition = runtime_definition(record, fixture["entities"], fixture.get("places"))
        check_definition_agreement(definition)
    except WorldError as error:
        raise _fail(name, f"cannot load its definition: {error}") from None
    error = fixture["error"]
    if error is not None:
        if error.get("kind") not in ERROR_KINDS or not isinstance(error.get("step"), int):
            raise _fail(name, f"has a malformed error entry {error!r}")
        if error["step"] != len(fixture["steps"]):
            raise _fail(
                name,
                f"expects an error at step {error['step']} but has {len(fixture['steps'])} "
                f"steps; the error step is the last step",
            )
    try:
        state = State.from_true(
            definition,
            fixture["initial"],
            _initial_numeric(definition, fixture),
            _initial_pairs(definition, fixture),
            fixture.get("initial_places"),
        )
    except WorldError as err:
        raise _fail(name, f"has a malformed initial state: {err}") from None
    performable = [et.label for et in definition.event_types if et.performable]
    states = [state]
    for k, step in enumerate(fixture["steps"], start=1):
        is_error_step = error is not None and error["step"] == k
        # Legal bindings before the step.
        expected_legal = step.get("legal")
        if not isinstance(expected_legal, dict) or sorted(expected_legal) != sorted(performable):
            raise _fail(name, f"step {k} must list the legal bindings of every event type")
        for label in performable:
            et = definition.event_type(label)
            got = [
                _binding_record(definition, et.roles, b, et.place)
                for b in legal_bindings(definition, state, label)
            ]
            if got != list(expected_legal[label]):
                raise _fail(
                    name,
                    f"step {k}: the legal bindings of {label} are {got}, but the fixture "
                    f"expects {expected_legal[label]}",
                )
        # The events.
        events = []
        for entry in step.get("events", []):
            try:
                events.append(Event.from_labels(definition, entry["event_type"], entry["binding"]))
            except WorldError as err:
                if is_error_step:
                    raise _fail(
                        name,
                        f"step {k}: the event {entry} is malformed at the loader, not at "
                        f"apply: {err}",
                    ) from None
                raise _fail(name, f"step {k}: the event {entry} is malformed: {err}") from None
        if is_error_step:
            try:
                apply(definition, state, events)
            except StepError as err:
                if err.kind != error["kind"]:
                    raise _fail(
                        name,
                        f"step {k}: apply raised {type(err).__name__} ({err}); the fixture "
                        f"expects {error['kind']}",
                    ) from None
                return FixtureReport(name, len(fixture["steps"]), error["kind"])
            raise _fail(
                name, f"step {k}: apply raised no error; the fixture expects {error['kind']}"
            )
        try:
            state = apply(definition, state, events)
        except StepError as err:
            raise _fail(name, f"step {k}: apply raised {type(err).__name__}: {err}") from None
        got_state = state.true_fluents(definition)
        if got_state != {e: list(v) for e, v in step["state"].items()}:
            raise _fail(
                name,
                f"step {k}: the state after the step is {got_state}, but the fixture "
                f"expects {step['state']}",
            )
        got_derived = derive(definition, state).true_fluents(definition)
        if got_derived != {e: list(v) for e, v in step["derived"].items()}:
            raise _fail(
                name,
                f"step {k}: the derived fluents after the step are {got_derived}, but the "
                f"fixture expects {step['derived']}",
            )
        if definition.numeric_fluents or "numeric" in step:
            got_numeric = state.numeric_values(definition)
            expected_numeric = {
                e: {f: float(x) for f, x in v.items()} for e, v in step.get("numeric", {}).items()
            }
            if got_numeric != expected_numeric:
                raise _fail(
                    name,
                    f"step {k}: the numeric values after the step are {got_numeric}, but the "
                    f"fixture expects {step.get('numeric')}",
                )
        if definition.pair_fluents or "pairs" in step:
            got_pairs = state.pairs_record(definition)
            expected_pairs = [list(t) for t in step.get("pairs", [])]
            if got_pairs != expected_pairs:
                raise _fail(
                    name,
                    f"step {k}: the pairs after the step are {got_pairs}, but the fixture "
                    f"expects {expected_pairs}",
                )
        states.append(state)
    for entry in fixture.get("comparisons", ()):
        got = _runtime_comparison(definition, entry, states)
        if got != bool(entry["expected"]):
            raise _fail(
                name,
                f"the comparison {entry} comes out {got}, but the fixture expects "
                f"{entry['expected']}",
            )
    return FixtureReport(name, len(fixture["steps"]), None)


def _initial_numeric(
    definition: RuntimeDefinition, fixture: Mapping[str, Any]
) -> dict[str, dict[str, float]] | None:
    """The fixture's ``initial_numeric``, or None (the entities table's values) when it has
    none and the world has no numeric fluents."""
    given = fixture.get("initial_numeric")
    if given is None:
        return {} if definition.numeric_fluents and definition.initial_numeric is None else None
    return {e: {f: float(x) for f, x in v.items()} for e, v in given.items()}


def _initial_pairs(
    definition: RuntimeDefinition, fixture: Mapping[str, Any]
) -> list[list[str]] | None:
    """The fixture's ``initial_pairs``, or None (the definition's initial pairs) when it has
    none and the world has no pair fluents; a world with pair fluents and no ``initial_pairs``
    starts with none (the location fluents' pairs come from ``initial_places``)."""
    given = fixture.get("initial_pairs")
    if given is None:
        return [] if definition.pair_fluents else None
    return [list(triple) for triple in given]


def _argument(definition: RuntimeDefinition, x: Any) -> Any:
    if isinstance(x, Mapping):
        return Mean(str(x["mean"]))
    return definition.entity_index(str(x))


def _runtime_comparison(
    definition: RuntimeDefinition, entry: Mapping[str, Any], states: Sequence[State]
) -> bool:
    """Judge a fixture's comparison entry with the runtime: a static comparison on the scalars,
    or a comparison at ``TIME.<time>`` (``states[time - 1]``) on a numeric fluent."""
    kind = entry["kind"]
    if kind not in COMPARISON_KINDS:
        raise FixtureError(f"unknown comparison kind {kind!r}")
    dimension = str(entry["dimension"])
    time = entry.get("time")
    state = None if time is None else states[int(time) - 1]
    if kind == "greater":
        return greater(
            definition,
            dimension,
            _argument(definition, entry["x"]),
            _argument(definition, entry["y"]),
            state,
        )
    if kind in ("increase", "decrease"):
        judge = increase if kind == "increase" else decrease
        return judge(
            definition,
            dimension,
            definition.entity_index(str(entry["x"])),
            states[int(time) - 1],
            states[int(time)],
        )
    judge = above if kind == "above" else below
    return judge(
        definition,
        dimension,
        _argument(definition, entry["x"]),
        str(entry["class"]),
        float(entry["z"]),
        state,
    )


def _brute_event_kind(
    brute: BruteForce,
    event: BruteEvent,
    state: BruteState,
    numeric: BruteNumeric | None = None,
    pairs: BrutePairs | None = None,
) -> str | None:
    """``illegal`` when the brute-force evaluator finds the event malformed or not legal."""
    if event.event_type not in brute.performable():
        return "illegal"
    roles = ("agent", "patient")[: brute.event_types[event.event_type]["arity"]]
    if set(event.binding) != set(roles):
        return "illegal"
    if brute.is_place_event_type(event.event_type):
        if (
            event.binding["agent"] not in brute.entities
            or event.binding["patient"] not in brute.places
        ):
            return "illegal"
        return None if brute.legal(event, state, numeric, pairs) else "illegal"
    if len(set(event.binding.values())) != len(roles):
        return "illegal"
    if any(label not in brute.entities for label in event.binding.values()):
        return "illegal"
    return None if brute.legal(event, state, numeric, pairs) else "illegal"


def _brute_initial_pairs(brute: BruteForce, fixture: Mapping[str, Any]) -> BrutePairs:
    given = fixture.get("initial_pairs")
    return brute.pairs_from_record(given or [], fixture.get("initial_places"))


def _brute_initial_numeric(brute: BruteForce, fixture: Mapping[str, Any]) -> BruteNumeric | None:
    given = fixture.get("initial_numeric")
    if given is not None:
        return {e: {f: float(x) for f, x in given[e].items()} for e in brute.entity_labels}
    if brute.numeric_fluents:
        return brute.initial_numeric()
    return None


def check_fixture_brute(fixture: Mapping[str, Any]) -> FixtureReport:
    """Run the brute-force evaluator on one fixture, apart from the runtime: the legal bindings
    before each step, the error step's illegality or interference, the state, the derived
    fluents, and the numeric values after each step, and the comparisons. Raises
    :class:`FixtureError` on the first disagreement."""
    name = str(fixture.get("name", "?"))
    brute = BruteForce(fixture["definition"], fixture["entities"], fixture.get("places", ()))
    state: BruteState = {e: frozenset(fixture["initial"].get(e, ())) for e in brute.entity_labels}
    numeric = _brute_initial_numeric(brute, fixture)
    pairs = _brute_initial_pairs(brute, fixture)
    numerics: list[BruteNumeric | None] = [numeric]
    error = fixture["error"]
    for k, step in enumerate(fixture["steps"], start=1):
        legal = brute.legal_record(state, numeric, pairs)
        if legal != {label: list(b) for label, b in step["legal"].items()}:
            raise _fail(
                name,
                f"step {k}: the brute-force legal bindings are {legal}, but the fixture expects "
                f"{step['legal']}",
            )
        events = [BruteEvent(e["event_type"], dict(e["binding"])) for e in step.get("events", [])]
        kinds = [
            k for k in (_brute_event_kind(brute, e, state, numeric, pairs) for e in events) if k
        ]
        if not kinds and any(brute.interfere(a, b) for a, b in itertools.combinations(events, 2)):
            kinds.append("interference")
        found = kinds[0] if kinds else None
        if error is not None and error["step"] == k:
            if found != error["kind"]:
                raise _fail(
                    name,
                    f"step {k}: the brute-force evaluator finds {found or 'no error'}; the "
                    f"fixture expects {error['kind']}",
                )
            return FixtureReport(name, len(fixture["steps"]), error["kind"])
        if found is not None:
            raise _fail(name, f"step {k}: the brute-force evaluator finds the step {found}")
        state, numeric, _ = brute.step(state, numeric, events, pairs)
        pairs = brute.pairs_after
        if brute.state_record(state) != {e: list(v) for e, v in step["state"].items()}:
            raise _fail(
                name,
                f"step {k}: the brute-force state is {brute.state_record(state)}, but the "
                f"fixture expects {step['state']}",
            )
        if brute.derived_record(state, numeric, pairs) != {
            e: list(v) for e, v in step["derived"].items()
        }:
            raise _fail(
                name,
                f"step {k}: the brute-force derived fluents are "
                f"{brute.derived_record(state, numeric, pairs)}, "
                f"but the fixture expects {step['derived']}",
            )
        if brute.pair_fluents or "pairs" in step:
            expected_pairs = [list(t) for t in step.get("pairs", [])]
            if brute.pairs_record(pairs) != expected_pairs:
                raise _fail(
                    name,
                    f"step {k}: the brute-force pairs are {brute.pairs_record(pairs)}, but the "
                    f"fixture expects {expected_pairs}",
                )
        if brute.numeric_fluents or "numeric" in step:
            expected = {
                e: {f: float(x) for f, x in v.items()} for e, v in step.get("numeric", {}).items()
            }
            if brute.numeric_record(numeric) != expected:
                raise _fail(
                    name,
                    f"step {k}: the brute-force numeric values are "
                    f"{brute.numeric_record(numeric)}, "
                    f"but the fixture expects {step.get('numeric')}",
                )
        numerics.append(numeric)
    for entry in fixture.get("comparisons", ()):
        time = entry.get("time")
        at = None if time is None else numerics[int(time) - 1]
        before = None
        if entry["kind"] in ("increase", "decrease"):
            before, at = numerics[int(time) - 1], numerics[int(time)]
        got = brute.compare(entry, at, before)
        if got != bool(entry["expected"]):
            raise _fail(
                name,
                f"the brute-force evaluator finds the comparison {entry} {got}, but the fixture "
                f"expects {entry['expected']}",
            )
    return FixtureReport(name, len(fixture["steps"]), None)


def fixture_paths(folder: str | Path = FIXTURES_DIR) -> list[Path]:
    return sorted(Path(folder).glob("*.json"))


def check_fixtures(folder: str | Path = FIXTURES_DIR) -> list[FixtureReport]:
    """Check every fixture in the folder, in name order, with the runtime and then with the
    brute-force evaluator. Raises on the first failure."""
    paths = fixture_paths(folder)
    if not paths:
        raise FixtureError(f"no fixture found in {folder}")
    reports = []
    for path in paths:
        fixture = read_fixture(path)
        reports.append(check_fixture(fixture))
        check_fixture_brute(fixture)
    return reports


# ---------------------------------------------------------------------------------------------
# Generating fixtures from a world
# ---------------------------------------------------------------------------------------------


def _rng_choice(rng: np.random.Generator, items: Sequence[Any]) -> Any:
    return items[int(rng.integers(len(items)))]


def _independent_set(
    brute: BruteForce, candidates: Sequence[BruteEvent], rng: np.random.Generator, size: int
) -> list[BruteEvent]:
    """Up to ``size`` mutually non-interfering events, chosen greedily in a random order."""
    order = rng.permutation(len(candidates))
    chosen: list[BruteEvent] = []
    for i in order:
        event = candidates[int(i)]
        if len(chosen) >= size:
            break
        if all(not brute.interfere(event, other) for other in chosen):
            chosen.append(event)
    return chosen


def _event_record(event: BruteEvent) -> dict[str, Any]:
    return {"event_type": event.event_type, "binding": dict(event.binding)}


def generate_fixture(
    name: str,
    description: str,
    record: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    initial: Mapping[str, Sequence[str]],
    seed: int,
    steps: int,
    max_events: int = 3,
    error: str | None = None,
    initial_numeric: Mapping[str, Mapping[str, float]] | None = None,
    places: Sequence[Mapping[str, Any]] | None = None,
    initial_pairs: Sequence[Sequence[str]] | None = None,
    initial_places: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """A fixture whose expected values come from the brute-force evaluator. Each step draws a
    random set of up to ``max_events`` non-interfering legal events. With ``error``, the last
    step instead holds an illegal event (``illegal``) or two interfering legal events
    (``interference``), found among the events of that state; the search fails with
    :class:`WorldError` when the state offers none. ``places`` (stage c1) is the places table
    of a world with places; ``initial_pairs`` (stage c2a) the pairs at ``TIME.1`` of a world
    with pair fluents; ``initial_places`` (stage c3a) every entity's initial place on each
    location fluent."""
    brute = BruteForce(record, entities, places or ())
    rng = np.random.default_rng(seed)
    state: BruteState = {e: frozenset(initial.get(e, ())) for e in brute.entity_labels}
    numeric: BruteNumeric | None = (
        None
        if initial_numeric is None
        else {e: {f: float(x) for f, x in initial_numeric[e].items()} for e in brute.entity_labels}
    )
    if numeric is None and brute.numeric_fluents:
        raise WorldError("the world has numeric fluents, so the fixture needs their initial values")
    pairs = brute.pairs_from_record(initial_pairs or [], initial_places)
    if initial_pairs is None and brute.pair_fluents:
        raise WorldError("the world has pair fluents, so the fixture needs its initial pairs")
    if initial_places is None and brute.location_fluents:
        raise WorldError("the world has location fluents, so the fixture needs its initial places")
    out_steps: list[dict[str, Any]] = []
    error_entry = None
    for k in range(1, steps + 1):
        legal_record = brute.legal_record(state, numeric, pairs)
        candidates = [
            BruteEvent(label, b) for label, bindings in legal_record.items() for b in bindings
        ]
        if error is not None and k == steps:
            events = _error_events(brute, candidates, state, rng, error, numeric, pairs)
            out_steps.append({"events": [_event_record(e) for e in events], "legal": legal_record})
            error_entry = {"step": k, "kind": error}
            break
        size = int(rng.integers(0, max_events + 1))
        events = _independent_set(brute, candidates, rng, size)
        state, numeric, _ = brute.step(state, numeric, events, pairs)
        pairs = brute.pairs_after
        entry = {
            "events": [_event_record(e) for e in events],
            "legal": legal_record,
            "derived": brute.derived_record(state, numeric, pairs),
            "state": brute.state_record(state),
        }
        if brute.numeric_fluents:
            entry["numeric"] = brute.numeric_record(numeric)
        if brute.pair_fluents:
            entry["pairs"] = brute.pairs_record(pairs)
        out_steps.append(entry)
    return fixture_record(
        name,
        description,
        record,
        entities,
        initial,
        out_steps,
        error_entry,
        initial_numeric,
        places=places,
        initial_pairs=initial_pairs,
        initial_places=initial_places,
    )


def _error_events(
    brute: BruteForce,
    legal: Sequence[BruteEvent],
    state: BruteState,
    rng: np.random.Generator,
    kind: str,
    numeric: BruteNumeric | None = None,
    pairs: BrutePairs | None = None,
) -> list[BruteEvent]:
    if kind == "illegal":
        illegal = [
            BruteEvent(label, b)
            for label in brute.performable()
            for b in brute.bindings(label)
            if not brute.legal(BruteEvent(label, b), state, numeric, pairs)
        ]
        # the illegal event names an entity pair, as before stage c3a, when the world has one
        between = [e for e in illegal if not brute.is_place_event_type(e.event_type)]
        illegal = between or illegal
        if not illegal:
            raise WorldError("every event is legal in the state; no illegal event to draw")
        return [_rng_choice(rng, illegal)]
    if kind == "interference":
        pairs = [(a, b) for a, b in itertools.combinations(legal, 2) if brute.interfere(a, b)]
        if not pairs:
            raise WorldError("no two legal events interfere in the state")
        return list(_rng_choice(rng, pairs))
    raise WorldError(f"unknown error kind {kind!r}; expected one of {', '.join(ERROR_KINDS)}")


def fixture_inputs(
    definition: Definition,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]]:
    """A built definition's record, its entities table without the fluent columns (Boolean and
    numeric), and its initial Boolean state, all through their JSON form, as the fixture
    generator needs them; ``fixture_initial_numeric`` gives the initial numeric values."""
    record = json.loads(to_json(definition.record(), indent=1, sort_keys=False))
    rows = json.loads(to_json(entities_frame(definition).to_dicts(), indent=1, sort_keys=False))
    base = [f.label for f in definition.fluents.base]
    numeric = list(definition.numeric.labels)
    initial = {row["label"]: [f for f in base if row[f]] for row in rows}
    entities = [
        {k: v for k, v in row.items() if k not in base and k not in numeric} for row in rows
    ]
    return record, entities, initial


def fixture_places(definition: Definition) -> list[dict[str, Any]] | None:
    """A built definition's places table as a fixture gives it (the rows of ``places.csv``,
    through their JSON form), or None without places (stage c1)."""
    if definition.places is None:
        return None
    return json.loads(
        to_json(places_frame(definition.places).to_dicts(), indent=1, sort_keys=False)
    )


def fixture_initial_pairs(definition: Definition) -> list[list[str]] | None:
    """A built definition's initial pairs between entities as a fixture gives them (the rows
    of ``initial_pairs.csv`` as triples, through their JSON form), or None without pair
    fluents (stage c2a)."""
    if not len(definition.pair_fluents):
        return None
    frame = initial_pairs_frame(definition.pair_fluents, definition.taxonomy.instances.labels)
    rows = json.loads(to_json(frame.to_dicts(), indent=1, sort_keys=False))
    return [[row["pair_fluent"], row["first"], row["second"]] for row in rows]


def fixture_initial_places(definition: Definition) -> dict[str, dict[str, str]] | None:
    """A built definition's initial places as a fixture gives them (the location columns of
    ``entities.csv``), or None without location fluents (stage c3a)."""
    if definition.places is None or not definition.pair_fluents.locations:
        return None
    columns = location_columns(definition.pair_fluents, definition.places.labels)
    labels = definition.taxonomy.instances.labels
    return {
        label: {fluent: column[i] for fluent, column in columns.items()}
        for i, label in enumerate(labels)
    }


def fixture_initial_numeric(definition: Definition) -> dict[str, dict[str, float]] | None:
    """A built definition's initial numeric values as a fixture gives them, or None without
    numeric fluents."""
    if not len(definition.numeric):
        return None
    rows = json.loads(to_json(entities_frame(definition).to_dicts(), indent=1, sort_keys=False))
    return {row["label"]: {f: float(row[f]) for f in definition.numeric.labels} for row in rows}


def world_fixtures(
    definition: Definition, name: str, count: int = 4, steps: int = 6
) -> list[dict[str, Any]]:
    """The generated fixtures of a world: ``count`` fixtures of ``steps`` random steps each
    (fixture seeds 1 to ``count``), one that ends in an illegal event, and one that ends in
    interfering events. Every expected value comes from the brute-force evaluator."""
    record, entities, initial = fixture_inputs(definition)
    initial_numeric = fixture_initial_numeric(definition)
    places = fixture_places(definition)
    initial_pairs = fixture_initial_pairs(definition)
    initial_places = fixture_initial_places(definition)
    fixtures = []
    for seed in range(1, count + 1):
        fixtures.append(
            generate_fixture(
                f"{name}_seed{seed}",
                f"{steps} random steps of the {name} world from its initial state, fixture "
                f"seed {seed}: at each step, up to three non-interfering legal events.",
                record,
                entities,
                initial,
                seed,
                steps,
                initial_numeric=initial_numeric,
                places=places,
                initial_pairs=initial_pairs,
                initial_places=initial_places,
            )
        )
    for kind, text in (
        ("illegal", "an event that is not legal in the state"),
        ("interference", "two legal events that interfere"),
    ):
        fixtures.append(
            generate_fixture(
                f"{name}_{kind}",
                f"Random steps of the {name} world, then a last step with {text}: apply must "
                f"raise the {kind} error.",
                record,
                entities,
                initial,
                count + 1,
                steps,
                error=kind,
                initial_numeric=initial_numeric,
                places=places,
                initial_pairs=initial_pairs,
                initial_places=initial_places,
            )
        )
    return fixtures


def write_world_fixtures(
    definition: Definition,
    name: str,
    folder: str | Path = FIXTURES_DIR,
    count: int = 4,
    steps: int = 6,
) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for fixture in world_fixtures(definition, name, count, steps):
        paths.append(write_fixture(folder / f"{fixture['name']}.json", fixture))
    return paths


__all__ = [
    "COMPARISON_KINDS",
    "ERROR_KINDS",
    "FIXTURES_DIR",
    "FIXTURE_VERSION",
    "BruteEvent",
    "BruteForce",
    "BruteNumeric",
    "BrutePairs",
    "BruteState",
    "FixtureError",
    "FixtureReport",
    "check_fixture",
    "check_fixture_brute",
    "check_fixtures",
    "fixture_initial_numeric",
    "fixture_initial_pairs",
    "fixture_initial_places",
    "fixture_inputs",
    "fixture_paths",
    "fixture_record",
    "generate_fixture",
    "read_fixture",
    "world_fixtures",
    "write_fixture",
    "write_world_fixtures",
]
