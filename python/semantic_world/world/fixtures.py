"""Conformance fixtures: the brute-force evaluator, the fixture format, checking, and generating.

A fixture is one JSON file in ``tests/fixtures/world/`` (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Conformance fixtures"). It holds a complete definition inline, the entities' static facts, the
initial state, the steps with what must come out of each, and the error that ``apply`` must
raise, if any. ``tests/fixtures/world/README.md`` is the format's description for whoever builds
the Rust runtime.

The brute-force evaluator here is written apart from the runtime. It reads the definition
record's truth tables, one entity or binding at a time, in plain Python, and never touches the
matrices. The generated fixtures take their expected values from it, and the runtime is tested
against it on random states.
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
from semantic_world.world.runtime import (
    Event,
    State,
    apply,
    check_definition_agreement,
    derive,
    legal_bindings,
)

FIXTURE_VERSION = 1
FIXTURES_DIR = Path("tests/fixtures/world")
ERROR_KINDS = ("illegal", "interference")

# ---------------------------------------------------------------------------------------------
# The brute-force evaluator
# ---------------------------------------------------------------------------------------------

BruteState = dict[str, frozenset[str]]
"""A state for the evaluator: each entity's set of true base fluents."""


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

    def __init__(self, record: Mapping[str, Any], entities: Sequence[Mapping[str, Any]]) -> None:
        self.record = record
        self.symbols = {s["label"]: s for s in record["symbols"]}
        self.literals = list(record["literals"])
        self.rules = {r["output"]: r for r in record["rules"]}
        self.event_types = {e["label"]: e for e in record["event_types"]}
        self.entities = {str(row["label"]): row for row in entities}
        self.entity_labels = [str(row["label"]) for row in entities]
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

    # Rules and literals ---------------------------------------------------------------------

    def rule(self, output: str, roles: Mapping[str, str], state: BruteState | None) -> int:
        rule = self.rules[output]
        index = 0
        for i in rule["inputs"]:
            index = (index << 1) | self.literal(self.literals[i], roles, state)
        return int(rule["truth_table"][index])

    def literal(
        self, literal: Mapping[str, Any], roles: Mapping[str, str], state: BruteState | None
    ) -> int:
        kind = literal["kind"]
        if kind == "comparison":
            agent = float(self.entities[roles["agent"]][literal["agent_scalar"]])
            patient = float(self.entities[roles["patient"]][literal["patient_scalar"]])
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
            return int(float(self.entities[entity][literal["scalar"]]) > literal["threshold"])
        if kind == "fluent":
            assert state is not None
            return self.fluent(literal["fluent"], entity, state)
        raise WorldError(f"unknown literal kind {kind!r}")

    def feature(self, label: str, entity: str) -> int:
        if self.symbols[label]["derived"]:
            return self.rule(label, {"entity": entity}, None)
        return int(self.entities[entity][label])

    def fluent(self, label: str, entity: str, state: BruteState) -> int:
        if self.symbols[label]["derived"]:
            return self.rule(label, {"entity": entity}, state)
        return int(label in state[entity])

    # The operations -------------------------------------------------------------------------

    def derived_static(self, entity: str) -> list[str]:
        return [f for f in self.derived_features if self.feature(f, entity)]

    def derived_fluent(self, entity: str, state: BruteState) -> list[str]:
        return [f for f in self.derived_fluents if self.fluent(f, entity, state)]

    def able(self, event: BruteEvent) -> bool:
        et = self.event_types[event.event_type]
        return bool(self.rule(et["requirement"]["output"], event.binding, None))

    def precondition_holds(self, event: BruteEvent, state: BruteState) -> bool:
        et = self.event_types[event.event_type]
        for lit in et["precondition"]["literals"]:
            value = self.fluent(lit["fluent"], event.binding[lit["role"]], state)
            if value != int(lit["value"]):
                return False
        return True

    def legal(self, event: BruteEvent, state: BruteState) -> bool:
        return self.able(event) and self.precondition_holds(event, state)

    def bindings(
        self, event_type: str, entities: Sequence[str] | None = None
    ) -> list[dict[str, str]]:
        """Every binding of the event type among the entities, in the fixed order: by agent,
        then patient, in entity order."""
        et = self.event_types[event_type]
        pool = [e for e in self.entity_labels if entities is None or e in set(entities)]
        if et["arity"] == 1:
            return [{"agent": e} for e in pool]
        return [{"agent": a, "patient": p} for a in pool for p in pool if a != p]

    def legal_bindings(
        self, event_type: str, state: BruteState, entities: Sequence[str] | None = None
    ) -> list[dict[str, str]]:
        return [
            b
            for b in self.bindings(event_type, entities)
            if self.legal(BruteEvent(event_type, b), state)
        ]

    def performable(self) -> list[str]:
        """The event types that have events: every event type that is not a category."""
        return [label for label, e in self.event_types.items() if e["kind"] == "event_type"]

    def cone(self, fluent: str) -> set[str]:
        if not self.symbols[fluent]["derived"]:
            return {fluent}
        found: set[str] = set()
        for i in self.rules[fluent]["inputs"]:
            literal = self.literals[i]
            if literal["kind"] == "fluent":
                found |= self.cone(literal["fluent"])
        return found

    def writes(self, event: BruteEvent) -> dict[tuple[str, str], bool]:
        et = self.event_types[event.event_type]
        return {(event.binding[e["role"]], e["fluent"]): bool(e["value"]) for e in et["effects"]}

    def reads(self, event: BruteEvent) -> set[tuple[str, str]]:
        et = self.event_types[event.event_type]
        found: set[tuple[str, str]] = set()
        for lit in et["precondition"]["literals"]:
            for base in self.cone(lit["fluent"]):
                found.add((event.binding[lit["role"]], base))
        return found

    def interfere(self, a: BruteEvent, b: BruteEvent) -> bool:
        if a.key() == b.key():
            return True
        wa, wb = set(self.writes(a)), set(self.writes(b))
        return bool(wa & wb) or bool(wa & self.reads(b)) or bool(wb & self.reads(a))

    def apply(self, state: BruteState, events: Sequence[BruteEvent]) -> BruteState:
        """The next state. The caller has checked legality and non-interference."""
        new = {e: set(v) for e, v in state.items()}
        for event in events:
            for (entity, fluent), value in self.writes(event).items():
                if value:
                    new[entity].add(fluent)
                else:
                    new[entity].discard(fluent)
        return {e: frozenset(v) for e, v in new.items()}

    def state_record(self, state: BruteState) -> dict[str, list[str]]:
        return {e: [f for f in self.base_fluents if f in state[e]] for e in self.entity_labels}

    def derived_record(self, state: BruteState) -> dict[str, list[str]]:
        return {e: self.derived_fluent(e, state) for e in self.entity_labels}

    def legal_record(self, state: BruteState) -> dict[str, list[dict[str, str]]]:
        return {label: self.legal_bindings(label, state) for label in self.performable()}


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
) -> dict[str, Any]:
    """Assemble a fixture, with the rule-set identity recomputed from the inline definition."""
    identity = rule_set_id(
        record["symbols"], record["literals"], record["rules"], record["event_types"]
    )
    return {
        "version": FIXTURE_VERSION,
        "name": name,
        "description": description,
        "rule_set_id": identity,
        "definition": {**dict(record), "rule_set_id": identity},
        "entities": list(entities),
        "initial": dict(initial),
        "steps": list(steps),
        "error": error,
    }


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
    definition: RuntimeDefinition, roles: Sequence[str], binding: Sequence[int]
) -> dict[str, str]:
    return {role: definition.entity_labels[i] for role, i in zip(roles, binding, strict=True)}


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
        definition = runtime_definition(record, fixture["entities"])
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
        state = State.from_true(definition, fixture["initial"])
    except WorldError as err:
        raise _fail(name, f"has a malformed initial state: {err}") from None
    performable = [et.label for et in definition.event_types if not et.is_category]
    for k, step in enumerate(fixture["steps"], start=1):
        is_error_step = error is not None and error["step"] == k
        # Legal bindings before the step.
        expected_legal = step.get("legal")
        if not isinstance(expected_legal, dict) or sorted(expected_legal) != sorted(performable):
            raise _fail(name, f"step {k} must list the legal bindings of every event type")
        for label in performable:
            et = definition.event_type(label)
            got = [
                _binding_record(definition, et.roles, b)
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
    return FixtureReport(name, len(fixture["steps"]), None)


def fixture_paths(folder: str | Path = FIXTURES_DIR) -> list[Path]:
    return sorted(Path(folder).glob("*.json"))


def check_fixtures(folder: str | Path = FIXTURES_DIR) -> list[FixtureReport]:
    """Check every fixture in the folder, in name order. Raises on the first failure."""
    paths = fixture_paths(folder)
    if not paths:
        raise FixtureError(f"no fixture found in {folder}")
    return [check_fixture(read_fixture(path)) for path in paths]


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
) -> dict[str, Any]:
    """A fixture whose expected values come from the brute-force evaluator. Each step draws a
    random set of up to ``max_events`` non-interfering legal events. With ``error``, the last
    step instead holds an illegal event (``illegal``) or two interfering legal events
    (``interference``), found among the events of that state; the search fails with
    :class:`WorldError` when the state offers none."""
    brute = BruteForce(record, entities)
    rng = np.random.default_rng(seed)
    state: BruteState = {e: frozenset(initial.get(e, ())) for e in brute.entity_labels}
    out_steps: list[dict[str, Any]] = []
    error_entry = None
    for k in range(1, steps + 1):
        legal_record = brute.legal_record(state)
        candidates = [
            BruteEvent(label, b) for label, bindings in legal_record.items() for b in bindings
        ]
        if error is not None and k == steps:
            events = _error_events(brute, candidates, state, rng, error)
            out_steps.append({"events": [_event_record(e) for e in events], "legal": legal_record})
            error_entry = {"step": k, "kind": error}
            break
        size = int(rng.integers(0, max_events + 1))
        events = _independent_set(brute, candidates, rng, size)
        state = brute.apply(state, events)
        out_steps.append(
            {
                "events": [_event_record(e) for e in events],
                "legal": legal_record,
                "derived": brute.derived_record(state),
                "state": brute.state_record(state),
            }
        )
    return fixture_record(name, description, record, entities, initial, out_steps, error_entry)


def _error_events(
    brute: BruteForce,
    legal: Sequence[BruteEvent],
    state: BruteState,
    rng: np.random.Generator,
    kind: str,
) -> list[BruteEvent]:
    if kind == "illegal":
        illegal = [
            BruteEvent(label, b)
            for label in brute.performable()
            for b in brute.bindings(label)
            if not brute.legal(BruteEvent(label, b), state)
        ]
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
    """A built definition's record, its entities table without the fluent columns, and its
    initial state, all through their JSON form, as the fixture generator needs them."""
    record = json.loads(to_json(definition.record(), indent=1, sort_keys=False))
    rows = json.loads(to_json(entities_frame(definition).to_dicts(), indent=1, sort_keys=False))
    base = [f.label for f in definition.fluents.base]
    initial = {row["label"]: [f for f in base if row[f]] for row in rows}
    entities = [{k: v for k, v in row.items() if k not in base} for row in rows]
    return record, entities, initial


def world_fixtures(
    definition: Definition, name: str, count: int = 4, steps: int = 6
) -> list[dict[str, Any]]:
    """The generated fixtures of a world: ``count`` fixtures of ``steps`` random steps each
    (fixture seeds 1 to ``count``), one that ends in an illegal event, and one that ends in
    interfering events. Every expected value comes from the brute-force evaluator."""
    record, entities, initial = fixture_inputs(definition)
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
    "ERROR_KINDS",
    "FIXTURES_DIR",
    "FIXTURE_VERSION",
    "BruteEvent",
    "BruteForce",
    "BruteState",
    "FixtureError",
    "FixtureReport",
    "check_fixture",
    "check_fixtures",
    "fixture_inputs",
    "fixture_paths",
    "fixture_record",
    "generate_fixture",
    "read_fixture",
    "world_fixtures",
    "write_fixture",
    "write_world_fixtures",
]
