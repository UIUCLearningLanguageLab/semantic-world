"""Stage a3: the runtime (``runtime.py``) and the loader (``RuntimeDefinition``)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from semantic_world.world.config import load_config
from semantic_world.world.definition import (
    RuntimeDefinition,
    load_definition,
    runtime_definition,
)
from semantic_world.world.errors import (
    DefinitionError,
    IllegalEventError,
    InterferenceError,
    WorldError,
)
from semantic_world.world.fixtures import BruteEvent, BruteForce, fixture_inputs
from semantic_world.world.fluents import derived_initial_values
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.identity import read_json, to_json
from semantic_world.world.labels import translate
from semantic_world.world.runtime import (
    Event,
    State,
    able,
    apply,
    check_definition_agreement,
    derive,
    initial_state,
    legal,
    legal_bindings,
)

DATA = Path("data/world")
RANDOM_STATES = 1000
RANDOM_SETS = 1000


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def runtime(tiny: WorldResult) -> RuntimeDefinition:
    return tiny.definition.runtime()


@pytest.fixture(scope="module")
def brute(tiny: WorldResult) -> BruteForce:
    record, entities, _ = fixture_inputs(tiny.definition)
    return BruteForce(record, entities)


@pytest.fixture(scope="module")
def tiny_folder(tiny: WorldResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.write(tmp_path_factory.mktemp("world") / "tiny")


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
    shape = (definition.entity_count, len(definition.base_fluents))
    return State(rng.integers(0, 2, size=shape, dtype=np.uint8))


def _brute_state(definition: RuntimeDefinition, state: State) -> dict[str, frozenset[str]]:
    return {
        label: frozenset(f for j, f in enumerate(definition.base_fluents) if state.values[i, j])
        for i, label in enumerate(definition.entity_labels)
    }


def _labels(definition: RuntimeDefinition, label: str, binding: tuple[int, ...]) -> dict:
    roles = definition.event_type(label).roles
    return {r: definition.entity_labels[i] for r, i in zip(roles, binding, strict=True)}


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if not et.is_category]


# ---------------------------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------------------------


def test_loader_reads_a_run_folder_and_matches_the_built_definition(
    tiny: WorldResult, tiny_folder: Path, runtime: RuntimeDefinition
) -> None:
    loaded = load_definition(tiny_folder)
    assert loaded.rule_set_id == tiny.rule_set_id == runtime.rule_set_id
    assert loaded.record == runtime.record == read_json(tiny_folder / "definition.json")
    assert loaded.entity_labels == runtime.entity_labels
    assert np.array_equal(loaded.free_values, runtime.free_values)
    assert np.array_equal(loaded.scalar_values, runtime.scalar_values)
    assert np.array_equal(loaded.initial_values, tiny.fluents.initial_values)
    assert loaded.base_fluents == tuple(f.label for f in tiny.fluents.base)
    assert loaded.derived_fluents == tuple(f.label for f in tiny.fluents.derived)
    assert len(loaded.event_types) == len(tiny.event_types.event_types)
    check_definition_agreement(loaded)


def test_loader_refuses_a_wrong_identity(tiny_folder: Path) -> None:
    record = read_json(tiny_folder / "definition.json")
    entities = pl.read_csv(tiny_folder / "entities.csv")
    record["rule_set_id"] = "0" * 64
    with pytest.raises(DefinitionError, match="hash to"):
        runtime_definition(record, entities)
    record = read_json(tiny_folder / "definition.json")
    record["rules"][0]["truth_table"] = record["rules"][0]["truth_table"][::-1]
    with pytest.raises(DefinitionError, match="hash to"):
        runtime_definition(record, entities)


def test_loader_names_malformed_tables(tiny_folder: Path) -> None:
    from semantic_world.world.identity import rule_set_id

    def rehash(record: dict) -> dict:
        record["rule_set_id"] = rule_set_id(
            record["symbols"], record["literals"], record["rules"], record["event_types"]
        )
        return record

    entities = pl.read_csv(tiny_folder / "entities.csv")
    record = rehash({**read_json(tiny_folder / "definition.json"), "version": 2})
    with pytest.raises(DefinitionError, match="version"):
        runtime_definition(record, entities)
    record = read_json(tiny_folder / "definition.json")
    effect = next(e for e in record["event_types"] if e["effects"])["effects"][0]
    effect["fluent"] = "BOOLFL.4"  # the derived fluent
    with pytest.raises(DefinitionError, match="not a base fluent"):
        runtime_definition(rehash(record), entities)
    record = read_json(tiny_folder / "definition.json")
    with pytest.raises(DefinitionError, match="no column PROPERTY.1"):
        runtime_definition(record, entities.drop("PROPERTY.1"))
    with pytest.raises(DefinitionError, match="column PART.7"):
        runtime_definition(record, entities.with_columns(pl.lit(0).alias("PART.7")))


def test_cones_follow_derived_fluent_rules(tiny: WorldResult, runtime: RuntimeDefinition) -> None:
    for fluent in runtime.base_fluents:
        assert runtime.cone(fluent) == (fluent,)
    for rule in tiny.fluents.rules:
        expected = {k for k in rule.inputs if k in runtime.base_fluents}
        assert set(runtime.cone(rule.output)) >= expected


# ---------------------------------------------------------------------------------------------
# derive and able against the generator's own values
# ---------------------------------------------------------------------------------------------


def test_derive_matches_the_taxonomy_and_the_reference_evaluation(
    tiny: WorldResult, runtime: RuntimeDefinition
) -> None:
    facts = derive(runtime, initial_state(runtime))
    features = tiny.taxonomy.features
    values = tiny.taxonomy.instances.values
    for j, label in enumerate(facts.static_labels):
        feature = features[next(f.label for f in features.features if translate(f.label) == label)]
        assert np.array_equal(facts.static[:, j], values[:, feature.position]), label
    reference = derived_initial_values(tiny.fluents, tiny.taxonomy)
    for j, label in enumerate(facts.fluent_labels):
        assert np.array_equal(facts.fluent[:, j], reference[label]), label
    subset = derive(runtime, initial_state(runtime), [3, 1])
    assert subset.entities == (3, 1)
    assert np.array_equal(subset.fluent, facts.fluent[[3, 1]])


def test_able_matches_the_capacities(tiny: WorldResult, runtime: RuntimeDefinition) -> None:
    from semantic_world.world.capacities import capacities_frame

    capacities = capacities_frame(tiny.taxonomy)
    n = runtime.entity_count
    for et in runtime.event_types:
        if et.is_category:
            continue
        if et.arity == 1:
            got = able(runtime, et.label, np.arange(n).reshape(-1, 1))
            assert np.array_equal(got, capacities[f"CAN.{et.label}"].to_numpy().astype(bool))
        else:
            agents = np.repeat(np.arange(n), n)
            patients = np.tile(np.arange(n), n)
            keep = agents != patients
            got = able(runtime, et.label, np.stack([agents[keep], patients[keep]], axis=1))
            matrix = np.zeros((n, n), dtype=bool)
            matrix[agents[keep], patients[keep]] = got
            # The extensional capacities: over the actual entities (the intensional ones treat
            # the other side as any possible object).
            assert np.array_equal(
                matrix.any(axis=1), capacities[f"ACTUAL_CAN.{et.label}"].to_numpy().astype(bool)
            )
            assert np.array_equal(
                matrix.any(axis=0),
                capacities[f"ACTUAL_CANBE.{et.label}"].to_numpy().astype(bool),
            )


# ---------------------------------------------------------------------------------------------
# Acceptance: legal and derive agree with the brute-force evaluator in 1,000 random states
# ---------------------------------------------------------------------------------------------


def test_legal_and_derive_agree_with_brute_force_in_random_states(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261008)
    for _ in range(RANDOM_STATES):
        state = _random_state(runtime, rng)
        brute_state = _brute_state(runtime, state)
        assert derive(runtime, state).true_fluents(runtime) == brute.derived_record(brute_state)
        for label in _performable(runtime):
            got = [_labels(runtime, label, b) for b in legal_bindings(runtime, state, label)]
            assert got == brute.legal_bindings(label, brute_state), label


def test_legal_bindings_order_and_subsets(runtime: RuntimeDefinition) -> None:
    state = initial_state(runtime)
    for label in _performable(runtime):
        bindings = legal_bindings(runtime, state, label)
        assert bindings == sorted(bindings)
        assert all(legal(runtime, state, label, bindings)) if bindings else True
        subset = legal_bindings(runtime, state, label, [5, 2, 7])
        assert subset == [b for b in bindings if set(b) <= {2, 5, 7}]
        assert legal_bindings(runtime, state, label, []) == []


# ---------------------------------------------------------------------------------------------
# apply: errors, inertia, and order independence
# ---------------------------------------------------------------------------------------------


def _legal_events(definition: RuntimeDefinition, state: State) -> list[Event]:
    return [
        Event(label, b)
        for label in _performable(definition)
        for b in legal_bindings(definition, state, label)
    ]


def _brute_event(definition: RuntimeDefinition, event: Event) -> BruteEvent:
    return BruteEvent(event.event_type, _labels(definition, event.event_type, event.binding))


def test_apply_writes_effects_and_keeps_the_rest(runtime: RuntimeDefinition) -> None:
    state = initial_state(runtime)
    event = next(
        e for e in _legal_events(runtime, state) if runtime.event_type(e.event_type).effects
    )
    et = runtime.event_type(event.event_type)
    after = apply(runtime, state, [event])
    written = {
        (event.binding[et.roles.index(e.role)], runtime.base_fluent_index(e.fluent)): e.value
        for e in et.effects
    }
    for (entity, j), value in written.items():
        assert after.values[entity, j] == int(value)
    mask = np.ones_like(state.values, dtype=bool)
    for entity, j in written:
        mask[entity, j] = False
    assert np.array_equal(after.values[mask], state.values[mask])
    assert not after.values.flags.writeable
    assert apply(runtime, state, []) == State(state.values)


def test_illegal_events_name_the_event_and_the_reason(runtime: RuntimeDefinition) -> None:
    state = initial_state(runtime)
    for label in _performable(runtime):
        et = runtime.event_type(label)
        n = runtime.entity_count
        candidates = (
            [(i,) for i in range(n)]
            if et.arity == 1
            else [(a, p) for a in range(n) for p in range(n) if a != p]
        )
        for binding in candidates:
            if legal(runtime, state, label, [binding])[0]:
                continue
            event = Event(label, binding)
            with pytest.raises(IllegalEventError) as info:
                apply(runtime, state, [event])
            message = str(info.value)
            assert event.describe(runtime) in message
            if not able(runtime, label, [binding])[0]:
                assert f"requirement {et.requirement} does not hold" in message
            else:
                assert "precondition literal" in message and "does not hold" in message
            break


def test_malformed_events_are_illegal(runtime: RuntimeDefinition) -> None:
    state = initial_state(runtime)
    category = next(et.label for et in runtime.event_types if et.is_category)
    for event, text in (
        (Event("EVENTTYPE9.9", (0,)), "not an event type"),
        (Event(category, (0, 1)), "category"),
        (Event("EVENTTYPE1.1", (0, 1)), "names 2 entities"),
        (Event("EVENTTYPE2.1.1", (0, 0)), "one entity twice"),
        (Event("EVENTTYPE1.1", (99,)), "outside the entities table"),
    ):
        with pytest.raises(IllegalEventError, match=text):
            apply(runtime, state, [event])
    with pytest.raises(WorldError, match="category"):
        legal_bindings(runtime, state, category)
    with pytest.raises(WorldError, match="twice"):
        able(runtime, "EVENTTYPE2.1.1", [(0, 0)])


def test_interfering_events_raise_the_documented_error(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(7)
    found = {"same fluent": 0, "reads": 0, "twice": 0}
    for _ in range(200):
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        for a, b in itertools.combinations(events, 2):
            if not brute.interfere(_brute_event(runtime, a), _brute_event(runtime, b)):
                assert apply(runtime, state, [a, b]) == apply(runtime, state, [b, a])
                continue
            with pytest.raises(InterferenceError) as info:
                apply(runtime, state, [a, b])
            message = str(info.value)
            assert a.describe(runtime) in message and b.describe(runtime) in message
            if "both write" in message:
                found["same fluent"] += 1
            else:
                assert "which the precondition of" in message
                found["reads"] += 1
        if events:
            with pytest.raises(InterferenceError, match="occurs twice"):
                apply(runtime, state, [events[0], events[0]])
            found["twice"] += 1
    assert all(found.values()), found


def test_interference_through_a_derived_fluent_cone() -> None:
    """A precondition reading a derived fluent interferes with an event that writes a base
    fluent of its cone, even when the two write different fluents. The tiny world has no such
    precondition, so the hand world of the fixtures is used."""
    from semantic_world.world.fixtures import read_fixture

    fixture = read_fixture(Path("tests/fixtures/world/hand_10_interfere_through_cone.json"))
    definition = runtime_definition(fixture["definition"], fixture["entities"])
    assert definition.cone("BOOLFL.3") == ("BOOLFL.1", "BOOLFL.2")
    state = State.from_true(definition, fixture["initial"])
    reader = Event.from_labels(definition, "EVENTTYPE1.2", {"agent": "INSTANCE.1.1.2"})
    writer = Event.from_labels(definition, "EVENTTYPE1.3", {"agent": "INSTANCE.1.1.2"})
    assert legal(definition, state, reader.event_type, [reader.binding])[0]
    assert legal(definition, state, writer.event_type, [writer.binding])[0]
    for order in ([reader, writer], [writer, reader]):
        with pytest.raises(InterferenceError) as info:
            apply(definition, state, order)
        message = str(info.value)
        assert "EVENTTYPE1.3(agent=INSTANCE.1.1.2) writes BOOLFL.1 of INSTANCE.1.1.2" in message
        assert "which the precondition of EVENTTYPE1.2(agent=INSTANCE.1.1.2) reads" in message
    # The same writer on another entity does not interfere.
    other = Event.from_labels(definition, "EVENTTYPE1.3", {"agent": "INSTANCE.1.1.1"})
    after = apply(definition, state, [reader, other])
    assert after.true_fluents(definition)["INSTANCE.1.1.2"] == ["BOOLFL.1"]


def test_applying_together_equals_one_at_a_time_in_every_order(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261009)
    sets = 0
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
                step = apply(runtime, step, [event])
            assert step == together
        sets += 1
    assert sets == RANDOM_SETS


def test_state_helpers(runtime: RuntimeDefinition) -> None:
    state = initial_state(runtime)
    true = state.true_fluents(runtime)
    assert State.from_true(runtime, true) == state
    with pytest.raises(WorldError, match="lists no fluents"):
        State.from_true(runtime, {})
    with pytest.raises(WorldError, match="not a base fluent"):
        State.from_true(runtime, {**true, runtime.entity_labels[0]: ["BOOLFL.4"]})
    with pytest.raises(WorldError, match="shape"):
        derive(runtime, State(np.zeros((1, 1), dtype=np.uint8)))
    event = Event.from_labels(runtime, "EVENTTYPE1.1", {"agent": runtime.entity_labels[2]})
    assert event.binding == (2,)
    assert event.describe(runtime) == f"EVENTTYPE1.1(agent={runtime.entity_labels[2]})"
    with pytest.raises(WorldError, match="roles"):
        Event.from_labels(runtime, "EVENTTYPE1.1", {"patient": runtime.entity_labels[2]})


def test_runtime_is_pure(runtime: RuntimeDefinition) -> None:
    rng = np.random.default_rng(3)
    state = _random_state(runtime, rng)
    before = json.loads(to_json(runtime.record))
    first = [legal_bindings(runtime, state, label) for label in _performable(runtime)]
    events = _legal_events(runtime, state)[:2]
    after_first = apply(runtime, state, events[:1]) if events else state
    second = [legal_bindings(runtime, state, label) for label in _performable(runtime)]
    assert first == second
    assert (apply(runtime, state, events[:1]) if events else state) == after_first
    assert json.loads(to_json(runtime.record)) == before
