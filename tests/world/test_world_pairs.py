"""Stage c2a: pair fluents in the definition, the runtime, and the fixtures
(``docs/specs/WORLD_PHASE_C.md``, "Pair fluents", "Literals over pairs", "Preconditions",
"Effects", "Conditions (F5)", "The runtime", "The spatial block (F6)", "Episodes", "Histories").

The acceptance tests of the stage, on the world of ``data/world/tiny_pairs.yaml`` (the tiny world
with the event file ``events/pair_example.yaml``): ``legal`` and ``derive`` agree with the
brute-force evaluator in 1,000 random states; applying a step's events together equals applying
them one at a time in every order, on 1,000 random non-interfering sets; no profile is broken in
1,000 episodes, and the initial pairs keep every profile; replay reproduces every pair change;
every broken event file fails with an error that names the file and the entry, one test per
rule; a world without pair fluents writes none of the new keys (its identity is the snapshot's,
checked in ``test_world_off_settings.py`` and ``tests/figures``); and a corpus run on a world
with pair fluents refuses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from world_support import config_before_phase_c

from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.definition import (
    RuntimeDefinition,
    load_definition,
    runtime_definition,
)
from semantic_world.world.dynamics import Effect, PairLiteral, parse_condition
from semantic_world.world.episodes import (
    STATS_STREAM,
    EpisodeGenerator,
    EpisodeSettings,
    Relatedness,
)
from semantic_world.world.errors import (
    DefinitionError,
    IllegalEventError,
    InterferenceError,
    ProfileViolationError,
    WorldError,
)
from semantic_world.world.event_file import load_event_file
from semantic_world.world.fixtures import (
    FIXTURES_DIR,
    BruteEvent,
    BruteForce,
    fixture_initial_numeric,
    fixture_initial_pairs,
    fixture_inputs,
    fixture_places,
    generate_fixture,
    read_fixture,
)
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.history import History, HistoryError, PairChange, read_histories, replay
from semantic_world.world.identity import canonical_json, read_json, to_json
from semantic_world.world.pair_fluents import (
    INITIAL_PAIRS_COLUMNS,
    INITIAL_PAIRS_FILE,
    PairFluent,
    draw_initial_pairs,
    profile_violations,
)
from semantic_world.world.runtime import (
    Event,
    State,
    able,
    apply,
    derive,
    initial_state,
    interferes,
    legal_bindings,
)
from semantic_world.world.streams import STREAM_NAMES

DATA = Path("data/world")
EVENT_FILE = DATA / "events" / "pair_example.yaml"
RANDOM_STATES = 1000
RANDOM_SETS = 1000
EPISODES = 1000


@pytest.fixture(scope="module")
def pairs_world() -> WorldResult:
    return define(load_config(DATA / "tiny_pairs.yaml"))


@pytest.fixture(scope="module")
def runtime(pairs_world: WorldResult) -> RuntimeDefinition:
    return pairs_world.definition.runtime()


@pytest.fixture(scope="module")
def brute(pairs_world: WorldResult) -> BruteForce:
    record, entities, _ = fixture_inputs(pairs_world.definition)
    return BruteForce(record, entities, fixture_places(pairs_world.definition) or ())


@pytest.fixture(scope="module")
def pairs_folder(pairs_world: WorldResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return pairs_world.write(tmp_path_factory.mktemp("world") / "tiny_pairs")


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    """The tiny world without pair fluents (since stage c4a ``tiny.yaml`` carries them, WM.63):
    the world of stage c3b, for the checks of a world without them."""
    return define(config_before_phase_c())


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if et.performable]


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
    """A random state: Boolean fluents at 0 or 1, numeric fluents uniform in their ranges, and
    the pairs of each pair fluent drawn by the initial-pair procedure at a rate drawn from
    ``[0, 0.1, 0.3, 0.6]``, so that every profile holds and rows are empty, single, or (where
    the profile allows) full."""
    shape = (definition.entity_count, len(definition.base_fluents))
    values = rng.integers(0, 2, size=shape, dtype=np.uint8)
    low, high = definition.range_bounds
    numeric = rng.uniform(
        low, high, size=(definition.entity_count, len(definition.numeric_fluents))
    )
    pairs = []
    for fluent in definition.pair_fluents:
        rate = float(rng.choice([0.0, 0.1, 0.3, 0.6]))
        pairs.append(
            draw_initial_pairs(dataclasses.replace(fluent, initial_rate=rate), shape[0], rng)
        )
    return State(values, numeric, tuple(pairs))


def _brute_state(definition: RuntimeDefinition, state: State) -> dict[str, frozenset[str]]:
    return {
        label: frozenset(f for j, f in enumerate(definition.base_fluents) if state.values[i, j])
        for i, label in enumerate(definition.entity_labels)
    }


def _brute_numeric(definition: RuntimeDefinition, state: State) -> dict[str, dict[str, float]]:
    return {
        label: {f: float(state.numeric[i, j]) for j, f in enumerate(definition.numeric_fluents)}
        for i, label in enumerate(definition.entity_labels)
    }


def _brute_pairs(definition: RuntimeDefinition, state: State) -> dict[str, frozenset]:
    labels = definition.entity_labels
    return {
        fluent.label: frozenset(
            (labels[int(i)], labels[int(j)]) for i, j in zip(*np.nonzero(matrix), strict=True)
        )
        for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True)
    }


def _labels(definition: RuntimeDefinition, label: str, binding: tuple[int, ...]) -> dict:
    roles = definition.event_type(label).roles
    return {r: definition.entity_labels[i] for r, i in zip(roles, binding, strict=True)}


def _brute_event(definition: RuntimeDefinition, event: Event) -> BruteEvent:
    return BruteEvent(event.event_type, _labels(definition, event.event_type, event.binding))


def _legal_events(definition: RuntimeDefinition, state: State) -> list[Event]:
    return [
        Event(label, b)
        for label in _performable(definition)
        for b in legal_bindings(definition, state, label)
    ]


def _generator(result: WorldResult, **settings: Any) -> EpisodeGenerator:
    definition = result.definition.runtime()
    return EpisodeGenerator(
        definition,
        Relatedness.from_statics(result.statics, definition),
        EpisodeSettings(**settings),
        record_legal=True,
    )


# ---------------------------------------------------------------------------------------------
# The event file, the streams, and the definition
# ---------------------------------------------------------------------------------------------


def test_the_event_file_defines_pair_fluents_and_the_world_has_them(
    pairs_world: WorldResult,
) -> None:
    event_file = load_event_file(EVENT_FILE)
    assert list(event_file.pair_fluents) == ["PAIRFL.1", "PAIRFL.2", "PAIRFL.3", "PAIRFL.4"]
    fluents = pairs_world.pair_fluents
    assert fluents.labels == ("PAIRFL.1", "PAIRFL.2", "PAIRFL.3", "PAIRFL.4")
    assert [f.symmetry for f in fluents] == ["symmetric", "asymmetric", "general", "asymmetric"]
    assert [f.functional for f in fluents] == [False, False, True, True]
    assert all(not f.total for f in fluents)
    # every kind of pair literal, both kinds of effect, and a pair condition, as the file gives
    kinds = set()
    conditions = set()
    adds = deletes = 0
    for et in pairs_world.event_types.event_types:
        for lit in et.precondition:
            if lit.pair:
                kinds.add(lit.kind)
        for effect in et.effects:
            if effect.pair:
                adds += effect.value
                deletes += not effect.value
            for lit in effect.condition:
                if lit.pair:
                    conditions.add(lit.kind)
    assert kinds == {"pair", "pair_some", "pair_someof", "pair_same"}
    assert conditions == {"pair", "pair_some"}
    assert adds >= 3 and deletes >= 1


def test_the_pair_streams_come_after_every_earlier_stream(pairs_world: WorldResult) -> None:
    # stage c2b's three streams follow, then stage c3a's and c3b's (test_world_pair_generation.py)
    assert STREAM_NAMES[-9:-5] == ("places", "place_event_types", "pair_fluents", "pair_initial")
    assert "world:pair_fluents" in pairs_world.stream_seeds
    assert "world:pair_initial" in pairs_world.stream_seeds


def test_the_symbols_carry_the_profiles_and_are_hashed(pairs_world: WorldResult) -> None:
    record = pairs_world.definition.record()
    symbols = [s for s in record["symbols"] if s["kind"] == "pair_fluent"]
    assert [s["label"] for s in symbols] == list(pairs_world.pair_fluents.labels)
    for symbol, fluent in zip(symbols, pairs_world.pair_fluents, strict=True):
        assert symbol == {
            "label": fluent.label,
            "kind": "pair_fluent",
            "derived": False,
            "fluent": True,
            "arity": 2,
            "symmetry": fluent.symmetry,
            "functional": fluent.functional,
            "total": False,
            "initial_rate": fluent.initial_rate,
        }
    # the pair fluent symbols follow the numeric fluents and precede the event types
    kinds = [s["kind"] for s in record["symbols"]]
    assert kinds.index("pair_fluent") > kinds.index("numeric_fluent")
    assert kinds.index("pair_fluent") < kinds.index("event_type")
    # a changed profile changes the identity: the symbols are hashed
    changed = json.loads(json.dumps(record))
    next(s for s in changed["symbols"] if s["label"] == "PAIRFL.1")["functional"] = True
    from semantic_world.world.identity import rule_set_id

    assert (
        rule_set_id(
            changed["symbols"], changed["literals"], changed["rules"], changed["event_types"]
        )
        != pairs_world.rule_set_id
    )


def test_the_records_hold_pair_literals_effects_and_compiled_literals(
    pairs_world: WorldResult,
) -> None:
    record = pairs_world.definition.record()
    by_label = {e["label"]: e for e in record["event_types"]}
    follow = by_label["EVENTTYPE2.1.2"]
    assert follow["precondition"]["literals"] == [
        {
            "kind": "pair",
            "pair_fluent": "PAIRFL.1",
            "first": "agent",
            "second": "patient",
            "value": True,
        },
        {
            "kind": "pair_some",
            "pair_fluent": "PAIRFL.3",
            "role": "agent",
            "value": False,
            "origin": "profile",
        },
    ]
    assert follow["precondition"]["expression"] == (
        "PAIRFL.1(agent, patient) AND NOT SOME.PAIRFL.3(agent)"
    )
    assert follow["effects"][0] == {
        "kind": "pair",
        "pair_fluent": "PAIRFL.3",
        "first": "agent",
        "second": "patient",
        "value": True,
        "expression": "PAIRFL.3(agent, patient) := 1",
    }
    assert follow["effects"][1]["condition"]["literals"] == [
        {"kind": "pair_some", "pair_fluent": "PAIRFL.2", "role": "patient", "value": True}
    ]
    mount = by_label["EVENTTYPE2.2.1"]
    assert mount["precondition"]["expression"] == (
        "SAME.PAIRFL.3(agent, patient) AND agent.BOOLFL.1 AND NOT PAIRFL.4(patient, agent) AND "
        "NOT SOME.PAIRFL.4(agent)"
    )
    compiled = [lit for lit in mount["precondition"]["literals"] if "origin" in lit]
    assert [lit["kind"] for lit in compiled] == ["pair", "pair_some"]
    assert all(lit["origin"] == "profile" for lit in compiled)
    # an add on a symmetric pair fluent that is not functional, and a delete, compile nothing
    assert all(
        "origin" not in lit for lit in by_label["EVENTTYPE2.1.1"]["precondition"]["literals"]
    )
    assert all(
        "origin" not in lit for lit in by_label["EVENTTYPE2.2.2"]["precondition"]["literals"]
    )
    # a compiled literal is a read: the runtime's event type knows it as compiled
    runtime = pairs_world.definition.runtime()
    literals = runtime.event_type("EVENTTYPE2.2.1").precondition
    assert [lit.compiled for lit in literals] == [False, False, True, True]


def test_initial_pairs_are_written_keep_every_profile_and_load_back(
    pairs_world: WorldResult, pairs_folder: Path
) -> None:
    fluents = pairs_world.pair_fluents
    for fluent, matrix in zip(fluents, fluents.initial_pairs, strict=True):
        assert profile_violations(fluent, matrix) == []
        if fluent.initial_rate == 0.0:
            assert not matrix.any()
        else:
            assert matrix.any()
    frame = pl.read_csv(pairs_folder / INITIAL_PAIRS_FILE)
    assert frame.columns == list(INITIAL_PAIRS_COLUMNS)
    labels = pairs_world.taxonomy.instances.labels
    index = {label: i for i, label in enumerate(labels)}
    rebuilt = [np.zeros_like(m) for m in fluents.initial_pairs]
    for row in frame.iter_rows(named=True):
        rebuilt[fluents.index(row["pair_fluent"])][index[row["first"]], index[row["second"]]] = True
    for a, b in zip(rebuilt, fluents.initial_pairs, strict=True):
        assert np.array_equal(a, b)
    # a symmetric pair fluent lists both directions
    symmetric = frame.filter(pl.col("pair_fluent") == "PAIRFL.1")
    held = {(r["first"], r["second"]) for r in symmetric.iter_rows(named=True)}
    assert held and all((b, a) in held for a, b in held)
    # the loader reads the file, and refuses a run without it
    loaded = load_definition(pairs_folder)
    assert loaded.initial_pairs is not None
    for a, b in zip(loaded.initial_pairs, fluents.initial_pairs, strict=True):
        assert np.array_equal(a, b)
    assert initial_state(loaded) == initial_state(pairs_world.definition.runtime())
    (pairs_folder / INITIAL_PAIRS_FILE).rename(pairs_folder / "moved.csv")
    try:
        with pytest.raises(DefinitionError, match="initial_pairs.csv is missing"):
            load_definition(pairs_folder)
    finally:
        (pairs_folder / "moved.csv").rename(pairs_folder / INITIAL_PAIRS_FILE)


def test_the_initial_pair_procedure_respects_every_profile() -> None:
    rng = np.random.default_rng(20261021)
    for symmetry in ("symmetric", "asymmetric", "general"):
        for functional in (False, True):
            for rate in (0.0, 0.3, 1.0):
                fluent = PairFluent("PAIRFL.1", 1, symmetry, functional, rate)
                for n in (2, 5, 9):
                    matrix = draw_initial_pairs(fluent, n, rng)
                    assert profile_violations(fluent, matrix) == [], (symmetry, functional, rate)
                    if rate == 0.0:
                        assert not matrix.any()
                    if rate == 1.0 and not functional and symmetry == "general":
                        assert matrix.sum() == n * (n - 1)
                    if rate == 1.0 and not functional and symmetry == "symmetric":
                        assert matrix.sum() == n * (n - 1)
                    if rate == 1.0 and not functional and symmetry == "asymmetric":
                        assert matrix.sum() == n * (n - 1) // 2
    # the same seed gives the same pairs
    a = draw_initial_pairs(
        PairFluent("PAIRFL.1", 1, "general", True, 0.5), 7, np.random.default_rng(3)
    )
    b = draw_initial_pairs(
        PairFluent("PAIRFL.1", 1, "general", True, 0.5), 7, np.random.default_rng(3)
    )
    assert np.array_equal(a, b)


def test_a_world_without_pair_fluents_writes_none_of_the_new_keys(
    tiny: WorldResult, tmp_path: Path
) -> None:
    record = tiny.definition.record()
    assert not any(s["kind"] == "pair_fluent" for s in record["symbols"])
    assert not any(lit["kind"].startswith("pair") for lit in record["literals"])
    assert "spatial" not in record and "spatial_id" not in record
    for entry in record["event_types"]:
        assert all("kind" not in lit for lit in entry["precondition"]["literals"])
        assert all("kind" not in e for e in entry["effects"])
    assert len(tiny.pair_fluents) == 0
    folder = tiny.write(tmp_path / "tiny")
    assert not (folder / INITIAL_PAIRS_FILE).exists()
    runtime = tiny.definition.runtime()
    assert runtime.pair_fluents == () and runtime.initial_pairs is None
    assert initial_state(runtime).pairs == ()
    # the histories of such a world carry no pair key
    for history in _generator(tiny).run(1, 20, STATS_STREAM):
        data = history.to_json()
        assert "initial_pairs" not in data
        assert not any(
            "pair" in c for s in data["steps"] for e in s["events"] for c in e["changes"]
        )
    # the stream seeds are recorded for every world, so config.yaml names the new streams
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    assert "world:pair_initial" in config["provenance"]["stream_seeds"]


# ---------------------------------------------------------------------------------------------
# Acceptance: the runtime against the brute-force evaluator
# ---------------------------------------------------------------------------------------------


def test_legal_and_derive_agree_with_brute_force_in_random_states(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261021)
    seen_legal = {label: 0 for label in _performable(runtime)}
    for _ in range(RANDOM_STATES):
        state = _random_state(runtime, rng)
        brute_state = _brute_state(runtime, state)
        numeric = _brute_numeric(runtime, state)
        pairs = _brute_pairs(runtime, state)
        assert derive(runtime, state).true_fluents(runtime) == brute.derived_record(
            brute_state, numeric, pairs
        )
        for label in _performable(runtime):
            got = [_labels(runtime, label, b) for b in legal_bindings(runtime, state, label)]
            assert got == brute.legal_bindings(label, brute_state, None, numeric, pairs), label
            seen_legal[label] += bool(got)
    # every explicit event type was legal somewhere, so every kind of pair literal was exercised
    for label in (
        "EVENTTYPE1.1",
        "EVENTTYPE2.1.1",
        "EVENTTYPE2.1.2",
        "EVENTTYPE2.2.1",
        "EVENTTYPE2.2.2",
    ):
        assert seen_legal[label] > 0, label


def test_applying_together_equals_one_at_a_time_in_every_order(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261022)
    sets = 0
    pair_writers = 0
    while sets < RANDOM_SETS:
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        if not events:
            continue
        order = rng.permutation(len(events))
        chosen: list[Event] = []
        size = int(rng.integers(1, 5))
        for i in order:
            event = events[int(i)]
            if len(chosen) >= size:
                break
            if all(
                not brute.interfere(_brute_event(runtime, event), _brute_event(runtime, other))
                for other in chosen
            ):
                chosen.append(event)
        together = apply(runtime, state, chosen)
        for permutation in itertools.permutations(chosen):
            step = state
            for event in permutation:
                step = apply(runtime, step, [event], drift=False)
            assert step == apply(runtime, state, chosen, drift=False)
            assert all(
                np.array_equal(a, b) for a, b in zip(step.pairs, together.pairs, strict=True)
            )
        pair_writers += sum(
            1 for e in chosen if any(x.pair for x in runtime.event_type(e.event_type).effects)
        )
        sets += 1
    assert sets == RANDOM_SETS and pair_writers > 0


def test_interfering_events_over_pairs_raise_the_documented_error(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(11)
    found = {"pair": 0, "row or column": 0, "same": 0}
    for _ in range(300):
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        for a, b in itertools.combinations(events, 2):
            brute_says = brute.interfere(_brute_event(runtime, a), _brute_event(runtime, b))
            assert interferes(runtime, a, b) == brute_says
            if not brute_says:
                continue
            with pytest.raises(InterferenceError) as info:
                apply(runtime, state, [a, b])
            message = str(info.value)
            assert a.describe(runtime) in message and b.describe(runtime) in message
            if "between" in message:
                if "both write" in message:
                    found["same"] += 1
                else:
                    found["pair"] += 1
            else:
                found["row or column"] += 1
    assert found["pair"] > 0 and found["same"] > 0


def test_interference_through_a_row_and_a_column() -> None:
    """From the hand world: two functional adds to one row interfere through the compiled
    SOME literal (a row read), and an event whose precondition reads a column interferes with
    one that writes a pair of it."""
    fixture = read_fixture(FIXTURES_DIR / "hand_35_interference_functional_row.json")
    definition = runtime_definition(
        fixture["definition"], fixture["entities"], fixture.get("places")
    )
    state = State.from_true(
        definition, fixture["initial"], fixture["initial_numeric"], [], fixture["initial_places"]
    )
    first = Event.from_labels(
        definition, "EVENTTYPE2.5", {"agent": "INSTANCE.1.1.1", "patient": "INSTANCE.1.1.2"}
    )
    second = Event.from_labels(
        definition, "EVENTTYPE2.5", {"agent": "INSTANCE.1.1.1", "patient": "INSTANCE.1.2.1"}
    )
    with pytest.raises(InterferenceError, match="PAIRFL.3 between") as info:
        apply(definition, state, [first, second])
    assert "the precondition of" in str(info.value)
    # the derived BOOLFL.4 = SOME.PAIRFL.1 of an entity reads its row, so an event whose
    # condition reads BOOLFL.4 of the patient interferes with an add into that row
    assert definition.pair_cone("BOOLFL.4") == (("row", "PAIRFL.1"),)
    assert definition.pair_cone("BOOLFL.3") == ()


# ---------------------------------------------------------------------------------------------
# apply: the symmetric writes and the invariant
# ---------------------------------------------------------------------------------------------


def test_a_symmetric_add_and_delete_write_both_directions(runtime: RuntimeDefinition) -> None:
    state = _random_state(runtime, np.random.default_rng(5))
    k = runtime.pair_index("PAIRFL.1")
    approach = [e for e in _legal_events(runtime, state) if e.event_type == "EVENTTYPE2.1.1"]
    assert approach, "EVENTTYPE2.1.1 (a symmetric add) is legal somewhere"
    event = approach[0]
    a, p = event.binding
    assert not state.pairs[k][a, p] and not state.pairs[k][p, a]
    after = apply(runtime, state, [event])
    assert after.pairs[k][a, p] and after.pairs[k][p, a]
    assert not after.pairs.__len__() == 0
    # every other pair of every pair fluent is as it was (inertia)
    for j, (before, now) in enumerate(zip(state.pairs, after.pairs, strict=True)):
        mask = np.ones_like(before)
        if j == k:
            mask[a, p] = mask[p, a] = False
        assert np.array_equal(before[mask], now[mask])
    assert not after.pairs[k].flags.writeable


def test_the_profile_invariant_is_checked_after_the_step() -> None:
    """A definition without the compiled literal of an asymmetric add can break the asymmetric
    profile; the runtime reports it as a profile error, which no correct definition raises."""
    fixture = read_fixture(FIXTURES_DIR / "hand_31_asymmetric_add_blocked_by_compiled_literal.json")
    record = json.loads(json.dumps(fixture["definition"]))
    entry = next(e for e in record["event_types"] if e["label"] == "EVENTTYPE2.4")
    entry["precondition"] = {"literals": [], "expression": "TRUE"}
    from semantic_world.world.identity import rule_set_id

    record["rule_set_id"] = rule_set_id(
        record["symbols"], record["literals"], record["rules"], record["event_types"]
    )
    definition = runtime_definition(record, fixture["entities"], fixture.get("places"))
    state = State.from_true(
        definition,
        fixture["initial"],
        fixture["initial_numeric"],
        fixture["initial_pairs"],
        fixture.get("initial_places"),
    )
    event = Event.from_labels(
        definition, "EVENTTYPE2.4", {"agent": "INSTANCE.1.1.1", "patient": "INSTANCE.1.1.2"}
    )
    with pytest.raises(ProfileViolationError) as info:
        apply(definition, state, [event])
    message = str(info.value)
    assert "PAIRFL.2" in message and "asymmetric" in message
    assert "INSTANCE.1.1.1" in message and "EVENTTYPE2.4" in message
    assert ProfileViolationError.kind == "profile"
    # the state refuses a pair of an entity with itself, and a mismatched pair part
    with pytest.raises(WorldError, match="itself"):
        State.from_true(
            definition, fixture["initial"], None, [["PAIRFL.1", "INSTANCE.1.1.1", "INSTANCE.1.1.1"]]
        )
    with pytest.raises(WorldError, match="pair fluents"):
        derive(definition, State(state.values, state.numeric, state.pairs[:1]))


def test_illegal_pair_events_name_the_literal(runtime: RuntimeDefinition) -> None:
    rng = np.random.default_rng(9)
    seen = 0
    for _ in range(50):
        state = _random_state(runtime, rng)
        for label in ("EVENTTYPE2.1.2", "EVENTTYPE2.2.1"):
            et = runtime.event_type(label)
            legal = set(legal_bindings(runtime, state, label))
            n = runtime.entity_count
            for binding in ((a, p) for a in range(n) for p in range(n) if a != p):
                if binding in legal or not able(runtime, label, [binding])[0]:
                    continue
                with pytest.raises(IllegalEventError) as info:
                    apply(runtime, state, [Event(label, binding)])
                message = str(info.value)
                assert "precondition literal" in message
                if "compiled literal" in message:
                    assert any(lit.compiled for lit in et.precondition)
                    seen += 1
                break
    assert seen > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: episodes, histories, and replay
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("initial", ["keep", "redraw"])
def test_no_profile_is_broken_in_episodes_and_replay_reproduces_every_pair_change(
    pairs_world: WorldResult, initial: str
) -> None:
    generator = _generator(pairs_world, initial=initial)
    definition = generator.definition
    labels = definition.entity_labels
    changes = 0
    with_pairs = 0
    for history in generator.run(3, EPISODES, STATS_STREAM):
        assert history.initial_pairs is not None
        participants = set(history.participants)
        for _fluent, first, second in history.initial_pairs:
            assert first in participants and second in participants
        states = replay(definition, history)
        for state in states:
            for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True):
                assert profile_violations(fluent, matrix) == []
                rows, columns = np.nonzero(matrix)
                assert all(labels[i] in participants for i in rows)
                assert all(labels[j] in participants for j in columns)
        with_pairs += bool(history.initial_pairs)
        for step, before, after in zip(history.steps, states[:-1], states[1:], strict=True):
            changed = {
                (f.label, labels[int(i)], labels[int(j)])
                for f, a, b in zip(definition.pair_fluents, before.pairs, after.pairs, strict=True)
                for i, j in zip(*np.nonzero(a != b), strict=True)
            }
            recorded = set()
            for event in step.events:
                for change in event.changes:
                    if not change.pair:
                        continue
                    assert isinstance(change, PairChange)
                    k = definition.pair_index(change.pair_fluent)
                    i, j = (
                        definition.entity_index(change.first),
                        definition.entity_index(change.second),
                    )
                    assert bool(after.pairs[k][i, j]) == change.to
                    recorded.add((change.pair_fluent, change.first, change.second))
                    if definition.pair_fluent(change.pair_fluent).symmetric:
                        recorded.add((change.pair_fluent, change.second, change.first))
                    changes += 1
            assert changed == recorded
        # the JSON form round-trips
        assert History.from_json(json.loads(json.dumps(history.to_json()))) == history
    assert changes > 0 and with_pairs > 0


def test_a_history_lists_the_pairs_among_its_participants_and_tampering_fails(
    pairs_world: WorldResult,
) -> None:
    generator = _generator(pairs_world)
    definition = generator.definition
    start = initial_state(definition)
    history = next(h for h in generator.run(4, 50, STATS_STREAM) if h.initial_pairs)
    rows = sorted(definition.entity_index(p) for p in history.participants)
    for fluent, matrix in zip(definition.pair_fluents, start.pairs, strict=True):
        expected = {
            (fluent.label, definition.entity_labels[i], definition.entity_labels[j])
            for i in rows
            for j in rows
            if matrix[i, j]
        }
        assert {t for t in history.initial_pairs if t[0] == fluent.label} == expected
    data = history.to_json()
    assert list(data)[:7] == [
        "label",
        "seed",
        "participants",
        "policy",
        "rule_set_id",
        "initial",
        "initial_numeric",
    ]
    assert list(data)[7] == "initial_pairs"
    # a pair change that the event does not make
    tampered = json.loads(json.dumps(data))
    step = next(s for s in tampered["steps"] if s["events"])
    step["events"][0]["changes"].append(
        {
            "pair": "PAIRFL.1",
            "first": history.participants[0],
            "second": history.participants[1],
            "to": True,
        }
    )
    with pytest.raises(HistoryError, match="pair changes"):
        replay(definition, History.from_json(tampered))
    # a history without initial pairs on a world with pair fluents
    missing = json.loads(json.dumps(data))
    del missing["initial_pairs"]
    with pytest.raises(HistoryError, match="initial pairs"):
        replay(definition, History.from_json(missing))


def test_the_redraw_comes_after_every_other_draw_of_the_part(
    pairs_world: WorldResult, tmp_path: Path
) -> None:
    """The pairs are redrawn after the Boolean and numeric redraws, so the participants and
    the Boolean and numeric initial values of an episode under ``redraw`` are those of the same
    world without pair fluents: the event file with the same requirements and fluents but no
    pair fluents, pair literals, or pair effects (the participants depend on the requirements,
    and the redraws come before any event is drawn)."""
    data = yaml.safe_load(EVENT_FILE.read_text(encoding="utf-8"))
    stripped = {
        "fluents": data["fluents"],
        "event_types": {label: {"requirement": "TRUE"} for label in data["event_types"]},
    }
    without = _generator(
        _event_file_run(tmp_path, stripped, "stripped.yaml", pair_count=0), initial="redraw"
    )
    with_pairs = _generator(pairs_world, initial="redraw")
    for a, b in zip(
        with_pairs.run(2, 30, STATS_STREAM), without.run(2, 30, STATS_STREAM), strict=True
    ):
        assert a.participants == b.participants
        assert a.initial == b.initial
        assert a.initial_numeric == b.initial_numeric
        assert a.initial_pairs is not None and b.initial_pairs is None


def test_simulate_writes_pairs_and_reads_them_back(pairs_folder: Path, tmp_path: Path) -> None:
    from semantic_world.world.episodes import simulate

    path, histories = simulate(pairs_folder, 20, out=tmp_path / "episodes.jsonl")
    assert read_histories(path) == histories
    definition = load_definition(pairs_folder)
    for history in histories:
        replay(definition, history)


# ---------------------------------------------------------------------------------------------
# Broken event files: one test per rule, each naming the file and the entry
# ---------------------------------------------------------------------------------------------


def _event_file_run(
    tmp_path: Path, data: dict[str, Any], name: str = "events.yaml", pair_count: int | None = None
) -> WorldResult:
    """The tiny_pairs configuration over another event file; ``pair_count`` replaces
    ``pair_fluents.count`` (stage c2b: the configuration must allow the file's pair fluents,
    and a file without them needs a count of 0 for a world without pair fluents)."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    world = yaml.safe_load((DATA / "tiny_pairs.yaml").read_text(encoding="utf-8"))
    world["event_types"]["event_file"] = str(path)
    if pair_count is not None:
        world["pair_fluents"]["count"] = pair_count
    return define(config_from_mapping(world, source=str(DATA / "tiny_pairs.yaml")))


BASE_PAIRS = {
    "PAIRFL.1": {"symmetry": "symmetric", "functional": False, "initial_rate": 0.2},
    "PAIRFL.2": {"symmetry": "asymmetric", "functional": False, "initial_rate": 0.1},
    "PAIRFL.3": {"symmetry": "general", "functional": True, "initial_rate": 0.2},
    "PAIRFL.4": {"symmetry": "symmetric", "functional": True, "initial_rate": 0.0},
}


def _entry(precondition: str | None = None, effects: list[str] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"requirement": "TRUE"}
    if precondition is not None:
        entry["precondition"] = precondition
    if effects is not None:
        entry["effects"] = effects
    return entry


BROKEN_ENTRIES = [
    ("same_on_not_functional", _entry("SAME.PAIRFL.1(agent, patient)"), "SAME needs a functional"),
    (
        "same_on_symmetric_functional",
        _entry("SAME.PAIRFL.4(agent, patient)", ["PAIRFL.4(agent, patient) := 1"]),
        "not achievable",
    ),
    ("positive_not_achievable", _entry("PAIRFL.4(agent, patient)"), "not achievable"),
    (
        # stage c2b's follow-up: one event cannot add both directions of an asymmetric pair
        "adds_both_directions_asymmetric",
        _entry(None, ["PAIRFL.2(agent, patient) := 1", "PAIRFL.2(patient, agent) := 1"]),
        "both directions",
    ),
    ("some_not_achievable", _entry("SOMEOF.PAIRFL.4(agent)"), "not achievable"),
    (
        "contradicting_effects",
        _entry(None, ["PAIRFL.1(agent, patient) := 1", "PAIRFL.1(patient, agent) := 0"]),
        "contradicts",
    ),
    (
        "duplicate_effect",
        _entry(None, ["PAIRFL.1(agent, patient) := 1", "PAIRFL.1(patient, agent) := 1"]),
        "twice",
    ),
    (
        "add_already_required",
        _entry("PAIRFL.1(patient, agent)", ["PAIRFL.1(agent, patient) := 1"]),
        "already requires",
    ),
    (
        "delete_already_required_absent",
        _entry("NOT PAIRFL.2(agent, patient)", ["PAIRFL.2(agent, patient) := 0"]),
        "already requires",
    ),
    (
        "conditional_add_asymmetric",
        _entry(None, ["when agent.BOOLFL.1 then PAIRFL.2(agent, patient) := 1"]),
        "conditional add",
    ),
    (
        "conditional_add_functional",
        _entry(None, ["when agent.BOOLFL.1 then PAIRFL.3(agent, patient) := 1"]),
        "conditional add",
    ),
    (
        "condition_names_the_precondition_pair",
        _entry(
            "PAIRFL.1(agent, patient)", ["when PAIRFL.1(patient, agent) then agent.BOOLFL.1 := 1"]
        ),
        "names a pair",
    ),
    (
        "condition_names_the_precondition_row",
        _entry(
            "SOME.PAIRFL.3(agent)", ["when SAME.PAIRFL.3(agent, patient) then agent.BOOLFL.1 := 1"]
        ),
        "names a row",
    ),
    (
        "condition_names_the_compiled_row",
        _entry(
            None,
            [
                "PAIRFL.3(agent, patient) := 1",
                "when SOME.PAIRFL.3(agent) then agent.BOOLFL.1 := 1",
            ],
        ),
        "names a row the precondition or its compiled literals names",
    ),
    (
        "condition_on_the_written_pair",
        _entry(None, ["when PAIRFL.1(patient, agent) then PAIRFL.1(agent, patient) := 0"]),
        "the pair the effect writes",
    ),
    (
        "compiled_contradicts_precondition",
        _entry("PAIRFL.2(patient, agent)", ["PAIRFL.2(agent, patient) := 1"]),
        "compiled literal",
    ),
    (
        "compiled_contradicts_some",
        _entry("SOME.PAIRFL.3(agent)", ["PAIRFL.3(agent, patient) := 1"]),
        "compiled literal",
    ),
    ("unknown_pair_fluent", _entry("PAIRFL.9(agent, patient)"), "unknown pair fluent"),
    (
        "unknown_pair_fluent_effect",
        _entry(None, ["PAIRFL.9(agent, patient) := 1"]),
        "unknown pair fluent",
    ),
    (
        "precondition_twice",
        _entry("PAIRFL.1(agent, patient) AND PAIRFL.1(patient, agent)"),
        "twice",
    ),
    (
        "precondition_contradiction",
        _entry("SOME.PAIRFL.1(agent) AND NOT SOMEOF.PAIRFL.1(agent)"),
        "negation",
    ),
]


@pytest.mark.parametrize("name, entry, message", BROKEN_ENTRIES, ids=[b[0] for b in BROKEN_ENTRIES])
def test_a_broken_two_place_entry_names_the_file_and_the_entry(
    tmp_path: Path, name: str, entry: dict[str, Any], message: str
) -> None:
    data = {"pair_fluents": BASE_PAIRS, "event_types": {"EVENTTYPE2.1.1": entry}}
    with pytest.raises(ConfigError, match=message) as info:
        _event_file_run(tmp_path, data, f"{name}.yaml")
    text = str(info.value)
    assert f"{name}.yaml" in text and "EVENTTYPE2.1.1" in text


def test_a_one_place_entry_cannot_read_two_roles(tmp_path: Path) -> None:
    data = {
        "pair_fluents": BASE_PAIRS,
        "event_types": {"EVENTTYPE1.1": _entry("PAIRFL.1(agent, patient)")},
    }
    with pytest.raises(ConfigError, match="roles agent") as info:
        _event_file_run(tmp_path, data)
    assert "EVENTTYPE1.1" in str(info.value)
    data["event_types"] = {"EVENTTYPE1.1": _entry("SOME.PAIRFL.1(agent)", ["agent.BOOLFL.1 := 1"])}
    result = _event_file_run(tmp_path, data)
    assert result.event_types.event_type("EVENTTYPE1.1").precondition[0].kind == "pair_some"


def test_the_pair_fluents_block_is_validated(tmp_path: Path) -> None:
    for pair_fluents, message in (
        # stage c2b: the file's entries replace the draws of the pair fluents they name, among
        # PAIRFL.1 to PAIRFL.<count> (a gap is allowed; a label above the count is not)
        ({"PAIRFL.5": BASE_PAIRS["PAIRFL.2"]}, "pair_fluents.count is 4"),
        ({"PAIRFL.1": {**BASE_PAIRS["PAIRFL.1"], "total": True}}, "location fluent"),
        ({"PAIRFL.1": {**BASE_PAIRS["PAIRFL.1"], "symmetry": "circular"}}, "symmetry"),
        ({"PAIRFL.1": {"symmetry": "symmetric", "functional": False}}, "initial_rate"),
        ({"PAIRFL.1": {**BASE_PAIRS["PAIRFL.1"], "initial_rate": 1.5}}, "initial_rate"),
        ({"BOOLFL.1": BASE_PAIRS["PAIRFL.1"]}, "PAIRFL"),
    ):
        with pytest.raises(ConfigError, match=message) as info:
            _event_file_run(tmp_path, {"pair_fluents": pair_fluents, "event_types": {}})
        assert "events.yaml" in str(info.value)
    # stage c3a: location fluents are never defined in an event file, and a place_event_types
    # entry needs a location fluent of the configuration
    with pytest.raises(ConfigError, match="not defined in an event file"):
        _event_file_run(tmp_path, {"pair_fluents": BASE_PAIRS, "event_types": {}, "locations": {}})
    with pytest.raises(ConfigError, match="no location fluent"):
        _event_file_run(
            tmp_path,
            {
                "pair_fluents": BASE_PAIRS,
                "event_types": {},
                "place_event_types": {"PLACEEVENTTYPE.1": {"kind": "move"}},
            },
        )
    # a pair literal in a world without pair fluents (no entries, and a count of 0) is an error
    with pytest.raises(ConfigError, match="pair fluents are none"):
        _event_file_run(
            tmp_path,
            {"event_types": {"EVENTTYPE2.1.1": _entry("PAIRFL.1(agent, patient)")}},
            pair_count=0,
        )
    # a file with one entry under a count of 4 gets the other three drawn (stage c2b)
    drawn = _event_file_run(
        tmp_path, {"pair_fluents": {"PAIRFL.2": BASE_PAIRS["PAIRFL.2"]}, "event_types": {}}
    )
    assert drawn.pair_fluents.labels == ("PAIRFL.1", "PAIRFL.2", "PAIRFL.3", "PAIRFL.4")
    assert drawn.pair_fluents["PAIRFL.2"].asymmetric
    # the parser's own errors
    with pytest.raises(ValueError, match="two roles"):
        PairLiteral.parse("SAME.PAIRFL.1(agent)")
    with pytest.raises(ValueError, match="one role"):
        PairLiteral.parse("SOME.PAIRFL.1(agent, patient)")
    with pytest.raises(ValueError, match="different roles"):
        PairLiteral.parse("PAIRFL.1(agent, agent)")
    with pytest.raises(ValueError, match="different roles"):
        Effect.parse("PAIRFL.1(agent, agent) := 1")
    assert parse_condition("NOT SOMEOF.PAIRFL.2(patient)")[0].text == "NOT SOMEOF.PAIRFL.2(patient)"


# ---------------------------------------------------------------------------------------------
# The spatial block (F6)
# ---------------------------------------------------------------------------------------------


def test_the_spatial_block_has_its_own_identity_outside_the_rule_set(
    tmp_path: Path, pairs_world: WorldResult
) -> None:
    data = yaml.safe_load(EVENT_FILE.read_text(encoding="utf-8"))
    data["spatial"] = {
        "PAIRFL.1": {"detector": "within", "parameters": {"distance": 1.0}},
        "PAIRFL.2": {"detector": "above", "parameters": {}},
    }
    result = _event_file_run(tmp_path, data)
    record = result.definition.record()
    assert record["spatial"] == data["spatial"]
    assert (
        record["spatial_id"]
        == hashlib.sha256(canonical_json(data["spatial"]).encode("utf-8")).hexdigest()
    )
    assert list(record)[-2:] == ["spatial", "spatial_id"]
    # neither is part of the rule-set identity
    assert result.rule_set_id == pairs_world.rule_set_id
    # the loader keeps both and checks the form and the identity
    folder = result.write(tmp_path / "spatial")
    loaded = load_definition(folder)
    assert loaded.spatial == data["spatial"] and loaded.spatial_id == record["spatial_id"]
    written = read_json(folder / "definition.json")
    assert written["spatial_id"] == record["spatial_id"]
    entities = pl.read_csv(folder / "entities.csv")
    places = pl.read_csv(folder / "places.csv")
    pairs = pl.read_csv(folder / INITIAL_PAIRS_FILE)
    broken = json.loads(json.dumps(written))
    broken["spatial_id"] = "0" * 64
    with pytest.raises(DefinitionError, match="spatial block hashes"):
        runtime_definition(broken, entities, places, pairs)
    broken = json.loads(json.dumps(written))
    broken["spatial"]["PAIRFL.9"] = {"detector": "within", "parameters": {}}
    with pytest.raises(DefinitionError, match="not a pair fluent"):
        runtime_definition(broken, entities, places, pairs)
    broken = json.loads(json.dumps(written))
    broken["spatial"]["PAIRFL.1"] = {"detector": 3}
    with pytest.raises(DefinitionError, match="detector"):
        runtime_definition(broken, entities, places, pairs)
    broken = json.loads(json.dumps(written))
    del broken["spatial"]
    with pytest.raises(DefinitionError, match="both spatial and spatial_id"):
        runtime_definition(broken, entities, places, pairs)
    # the event file checks the block's form
    data["spatial"] = {"PAIRFL.9": {"detector": "within"}}
    with pytest.raises(ConfigError, match="pair fluent"):
        _event_file_run(tmp_path, data)
    data["spatial"] = {"PAIRFL.1": {"parameters": {}}}
    with pytest.raises(ConfigError, match="detector"):
        _event_file_run(tmp_path, data)


# ---------------------------------------------------------------------------------------------
# Fixtures, the loader, and the corpus
# ---------------------------------------------------------------------------------------------


def test_generated_fixtures_of_the_pair_world_pass_both_checkers(pairs_world: WorldResult) -> None:
    from semantic_world.world.fixtures import check_fixture, check_fixture_brute

    record, entities, initial = fixture_inputs(pairs_world.definition)
    places = fixture_places(pairs_world.definition)
    numeric = fixture_initial_numeric(pairs_world.definition)
    initial_pairs = fixture_initial_pairs(pairs_world.definition)
    assert initial_pairs and all(len(t) == 3 for t in initial_pairs)
    for seed, error in ((1, None), (2, None), (3, "illegal"), (4, "interference")):
        fixture = generate_fixture(
            f"probe{seed}",
            "a probe",
            record,
            entities,
            initial,
            seed,
            6,
            error=error,
            initial_numeric=numeric,
            places=places,
            initial_pairs=initial_pairs,
        )
        assert fixture["initial_pairs"] == initial_pairs
        assert list(fixture)[: list(fixture).index("steps")][-2:] == [
            "initial_numeric",
            "initial_pairs",
        ]
        for step in fixture["steps"]:
            if "state" in step:
                assert "pairs" in step
        check_fixture(fixture)
        check_fixture_brute(fixture)
        if error is None:
            assert any(step["pairs"] != initial_pairs for step in fixture["steps"]), "pairs changed"
    # a wrong pairs record is caught by both checkers
    fixture = generate_fixture(
        "probe",
        "a probe",
        record,
        entities,
        initial,
        1,
        3,
        initial_numeric=numeric,
        places=places,
        initial_pairs=initial_pairs,
    )
    broken = json.loads(json.dumps(fixture))
    broken["steps"][0]["pairs"] = []
    with pytest.raises(WorldError, match="pairs after the step"):
        check_fixture(broken)
    with pytest.raises(WorldError, match="brute-force pairs"):
        check_fixture_brute(broken)
    # the generator needs the initial pairs of a world with pair fluents
    with pytest.raises(WorldError, match="needs its initial pairs"):
        generate_fixture(
            "probe",
            "a probe",
            record,
            entities,
            initial,
            1,
            3,
            initial_numeric=numeric,
            places=places,
        )


def test_the_loader_validates_pair_records(pairs_folder: Path) -> None:
    from semantic_world.world.identity import rule_set_id

    def rehash(data: dict) -> dict:
        data["rule_set_id"] = rule_set_id(
            data["symbols"], data["literals"], data["rules"], data["event_types"]
        )
        return data

    record = read_json(pairs_folder / "definition.json")
    entities = pl.read_csv(pairs_folder / "entities.csv")
    places = pl.read_csv(pairs_folder / "places.csv")
    pairs = pl.read_csv(pairs_folder / INITIAL_PAIRS_FILE)
    runtime_definition(record, entities, places, pairs)
    follow = next(e for e in record["event_types"] if e["label"] == "EVENTTYPE2.1.2")
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == "EVENTTYPE2.1.2")
    entry["precondition"]["literals"][0]["pair_fluent"] = "PAIRFL.9"
    with pytest.raises(DefinitionError, match="not a pair fluent"):
        runtime_definition(rehash(broken), entities, places, pairs)
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == "EVENTTYPE2.2.1")
    entry["precondition"]["literals"][0]["pair_fluent"] = "PAIRFL.1"  # SAME on a non-functional
    with pytest.raises(DefinitionError, match="not functional"):
        runtime_definition(rehash(broken), entities, places, pairs)
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == "EVENTTYPE1.1")
    entry["precondition"]["literals"][0]["role"] = "patient"
    with pytest.raises(DefinitionError, match="role 'patient'"):
        runtime_definition(rehash(broken), entities, places, pairs)
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == "EVENTTYPE2.1.2")
    entry["effects"][0]["second"] = "agent"
    with pytest.raises(DefinitionError, match="names the roles"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # stage c3a: a total pair fluent is a location fluent, general and functional, without an
    # initial rate, after the pair fluents between entities
    broken = json.loads(json.dumps(record))
    symbol = next(s for s in broken["symbols"] if s["label"] == "PAIRFL.1")
    symbol["total"] = True
    with pytest.raises(DefinitionError, match="general and functional"):
        runtime_definition(rehash(broken), entities, places, pairs)
    broken = json.loads(json.dumps(record))
    symbol = next(s for s in broken["symbols"] if s["label"] == "PAIRFL.3")
    symbol.update({"total": True, "symmetry": "general"})
    with pytest.raises(DefinitionError, match="initial rate"):
        runtime_definition(rehash(broken), entities, places, pairs)
    symbol["initial_rate"] = None
    with pytest.raises(DefinitionError, match="after the pair fluents between entities"):
        runtime_definition(rehash(broken), entities, places, pairs)
    broken = json.loads(json.dumps(record))
    symbol = next(s for s in broken["symbols"] if s["label"] == "PAIRFL.4")
    symbol["label"] = "PAIRFL.5"
    with pytest.raises(DefinitionError, match="PAIRFL.1 to"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # the initial pairs are checked against the profiles and the labels
    with pytest.raises(DefinitionError, match="profile"):
        runtime_definition(
            record,
            entities,
            places,
            [
                ["PAIRFL.2", "INSTANCE.1.1.1", "INSTANCE.1.1.2"],
                ["PAIRFL.2", "INSTANCE.1.1.2", "INSTANCE.1.1.1"],
            ],
        )
    with pytest.raises(DefinitionError, match="not an entity"):
        runtime_definition(
            record, entities, places, [["PAIRFL.2", "INSTANCE.1.1.1", "INSTANCE.9.9.9"]]
        )
    assert follow["precondition"]["literals"][0]["kind"] == "pair"


def test_a_corpus_run_on_a_world_with_pair_fluents_loads(tmp_path: Path) -> None:
    """Stage c2a's refusal of a world with pair fluents ended in stage c4a (WM.63): the corpus
    loads the world, lists its pair fluents, and names no event of a place event type."""
    from semantic_world.corpus.config import load_config as load_corpus_config
    from semantic_world.corpus.world import load_world

    corpus = yaml.safe_load((Path("data/corpus/tiny.yaml")).read_text(encoding="utf-8"))
    corpus["world"] = {"config": str((DATA / "tiny_pairs.yaml").resolve())}
    path = tmp_path / "corpus.yaml"
    path.write_text(yaml.safe_dump(corpus), encoding="utf-8")
    world = load_world(load_corpus_config(path))
    assert list(world.pair_fluents) == ["PAIRFL.1", "PAIRFL.2", "PAIRFL.3", "PAIRFL.4"]
    for label in world.place_event_types:
        assert world.is_place_event_type(label) and world.event_names(label) == ()


def test_world_stats_count_pair_condition_literals_only_when_present(
    pairs_world: WorldResult, tiny: WorldResult
) -> None:
    by_symbol = pairs_world.stats["effects"]["condition_literals"]["by_symbol"]
    assert by_symbol["pair"] == 2
    assert "pair" not in tiny.stats["effects"]["condition_literals"]["by_symbol"]
    # the define line names the pair fluents only when there are any
    from semantic_world.world.__main__ import main

    assert (
        main(
            ["define", str(DATA / "tiny_pairs.yaml"), "--out", str(Path(tmp_path_of(pairs_world)))]
        )
        == 0
    )


def tmp_path_of(result: WorldResult) -> Path:
    import tempfile

    return Path(tempfile.mkdtemp()) / result.config.name


def test_the_hand_fixtures_cover_the_specified_cases() -> None:
    names = [p.stem for p in FIXTURES_DIR.glob("hand_*.json")]
    for case in (
        "symmetric_add_writes_both_directions",
        "symmetric_delete_clears_both_directions",
        "asymmetric_add_blocked_by_compiled_literal",
        "functional_add_blocked_by_full_row",
        "derived_over_some_changes",
        "interference_same_pair",
        "interference_functional_row",
        "interference_asymmetric_both_directions",
        "pair_condition",
    ):
        assert any(name.endswith(case) for name in names), case
    fixture = read_fixture(FIXTURES_DIR / "hand_29_symmetric_add_writes_both_directions.json")
    assert fixture["initial_pairs"] == []
    # the location pairs of stage c3a follow the pairs between entities
    assert [p for p in fixture["steps"][0]["pairs"] if p[0] != "PAIRFL.4"] == [
        ["PAIRFL.1", "INSTANCE.1.1.1", "INSTANCE.1.1.2"],
        ["PAIRFL.1", "INSTANCE.1.1.2", "INSTANCE.1.1.1"],
    ]
    assert fixture["version"] == 1
    assert to_json(fixture, indent=1, sort_keys=False) + "\n" == (
        FIXTURES_DIR / "hand_29_symmetric_add_writes_both_directions.json"
    ).read_text(encoding="utf-8")
