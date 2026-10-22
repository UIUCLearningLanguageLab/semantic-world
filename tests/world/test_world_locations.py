"""Stage c3a: locations as given by hand (``docs/specs/WORLD_PHASE_C.md``, "Pair fluents",
"Place event types", "Literals over pairs", "The runtime", "Episodes", "Histories", "Explicit
event files", "Conformance fixtures"; Jon's decision WM.64).

The acceptance tests of the stage, on the world of ``data/world/tiny_place_chain.yaml`` (the
tiny world with one location fluent and the event file ``events/place_chain_example.yaml``):
every entity is at exactly one place at every time point of every episode; every move changes
the agent's place to the bound place and nothing else of the location fluent; no move goes to
the agent's own place; every act happens at the agent's place; the chain's events occur in the
order its preconditions force; ``legal`` and ``derive`` agree with the brute-force evaluator in
1,000 random states; applying a step's events together equals applying them one at a time in
every order, on 1,000 random non-interfering sets; no profile is broken in 1,000 episodes;
replay reproduces every change; the habitats are uniform with ``habitat_strength: 0`` and follow
the weights with the default, within a binomial margin stated from the configuration; with
``scene.places: all`` every place is in every episode and with ``scene.initial_location: seed``
every participant starts at the seed entity's place; every broken event file fails with an error
that names the file and the entry, one test per rule; and a world without location fluents
writes none of the new keys (its identity is the snapshot's, ``test_world_off_settings.py``).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.corpus.config import config_from_mapping as corpus_config_from_mapping
from semantic_world.corpus.errors import CorpusError
from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.definition import RuntimeDefinition, load_definition, runtime_definition
from semantic_world.world.episodes import (
    STATS_STREAM,
    EpisodeGenerator,
    EpisodeSettings,
    Relatedness,
    read_scene_settings,
    simulate,
)
from semantic_world.world.errors import DefinitionError, IllegalEventError, WorldError
from semantic_world.world.event_file import load_event_file
from semantic_world.world.fixtures import (
    FIXTURES_DIR,
    BruteEvent,
    BruteForce,
    check_fixture,
    check_fixture_brute,
    fixture_inputs,
    fixture_places,
    read_fixture,
    world_fixtures,
)
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.history import History, HistoryError, read_histories, replay
from semantic_world.world.identity import to_json
from semantic_world.world.locations import draw_places, habitat_weights, with_location_fluents
from semantic_world.world.pair_fluents import profile_violations
from semantic_world.world.runtime import (
    Event,
    State,
    apply,
    derive,
    initial_state,
    interferes,
    legal_bindings,
    place_able_table,
)
from semantic_world.world.streams import STREAM_NAMES, WorldStreams

DATA = Path("data/world")
EVENT_FILE = DATA / "events" / "place_chain_example.yaml"
RANDOM_STATES = 1000
RANDOM_SETS = 1000
EPISODES = 1000
HABITAT_SEEDS = 400
"""Seeds of ``world:habitats`` drawn for the habitat checks: 400 seeds of 12 entities, 4,800
draws of a place, so that the binomial margin of a share near a quarter is about 2 points."""
LOCATION = "PAIRFL.2"
NEAR = "PAIRFL.1"
GO, REST, APPROACH, EAT = "PLACEEVENTTYPE.1", "PLACEEVENTTYPE.2", "EVENTTYPE2.1.1", "EVENTTYPE2.1.2"


@pytest.fixture(scope="module")
def chain() -> WorldResult:
    return define(load_config(DATA / "tiny_place_chain.yaml"))


@pytest.fixture(scope="module")
def runtime(chain: WorldResult) -> RuntimeDefinition:
    return chain.definition.runtime()


@pytest.fixture(scope="module")
def brute(chain: WorldResult) -> BruteForce:
    record, entities, _ = fixture_inputs(chain.definition)
    return BruteForce(record, entities, fixture_places(chain.definition) or ())


@pytest.fixture(scope="module")
def chain_folder(chain: WorldResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return chain.write(tmp_path_factory.mktemp("world") / "tiny_place_chain")


@pytest.fixture(scope="module")
def tiny_full() -> WorldResult:
    """The tiny full world without its location fluent (since stage c3b ``tiny_full.yaml`` has
    one): the world of stage c2b, for the checks of a world without location fluents."""
    data = yaml.safe_load((DATA / "tiny_full.yaml").read_text(encoding="utf-8"))
    data["pair_fluents"]["locations"] = 0
    return define(config_from_mapping(data, source=str(DATA / "tiny_full.yaml")))


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if et.performable]


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
    """A random state: Boolean fluents at 0 or 1, numeric fluents uniform in their ranges, the
    pairs of the pair fluents between entities drawn at a rate from ``[0, 0.3, 0.6]``, and
    every entity at a place drawn uniformly on each location fluent."""
    from semantic_world.world.pair_fluents import draw_initial_pairs

    n = definition.entity_count
    values = rng.integers(0, 2, size=(n, len(definition.base_fluents)), dtype=np.uint8)
    low, high = definition.range_bounds
    numeric = rng.uniform(low, high, size=(n, len(definition.numeric_fluents)))
    pairs = []
    for fluent in definition.pair_fluents:
        if fluent.total:
            matrix = np.zeros((n, definition.place_count), dtype=bool)
            matrix[np.arange(n), rng.integers(0, definition.place_count, size=n)] = True
            pairs.append(matrix)
            continue
        rate = float(rng.choice([0.0, 0.3, 0.6]))
        pairs.append(draw_initial_pairs(dataclasses.replace(fluent, initial_rate=rate), n, rng))
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
            (labels[int(i)], (definition.place_labels if fluent.total else labels)[int(j)])
            for i, j in zip(*np.nonzero(matrix), strict=True)
        )
        for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True)
    }


def _brute_event(definition: RuntimeDefinition, event: Event) -> BruteEvent:
    return BruteEvent(event.event_type, event.binding_labels(definition))


def _legal_events(definition: RuntimeDefinition, state: State) -> list[Event]:
    return [
        Event(label, b)
        for label in _performable(definition)
        for b in legal_bindings(definition, state, label)
    ]


def _generator(result: WorldResult, **settings: Any) -> EpisodeGenerator:
    definition = result.definition.runtime()
    places = result.config.places
    return EpisodeGenerator(
        definition,
        Relatedness.from_statics(result.statics, definition),
        EpisodeSettings(**settings),
        record_legal=True,
        habitat_strength=1.0 if places is None else places.habitat_strength,
    )


def _place_of(
    definition: RuntimeDefinition, state: State, entity: int, fluent: str = LOCATION
) -> int:
    row = state.pairs[definition.pair_index(fluent)][entity]
    where = np.flatnonzero(row)
    assert where.size == 1, "every entity is at exactly one place"
    return int(where[0])


# ---------------------------------------------------------------------------------------------
# The configuration, the definition, and the stream
# ---------------------------------------------------------------------------------------------


def test_location_fluents_need_places_and_default_to_none() -> None:
    assert config_from_mapping({}, source="<test>").pair_fluents.locations == 0
    with pytest.raises(ConfigError, match="locations"):
        config_from_mapping({"places": None, "pair_fluents": {"locations": 1}}, source="<test>")
    config = config_from_mapping({"pair_fluents": {"count": 2, "locations": 2}}, source="<test>")
    assert config.pair_fluents.labels == ("PAIRFL.1", "PAIRFL.2")
    assert config.pair_fluents.location_labels == ("PAIRFL.3", "PAIRFL.4")
    assert config.resolved()["pair_fluents"]["locations"] == 2
    # the chain example's world: one pair fluent between entities, then the location fluent
    chain = load_config(DATA / "tiny_place_chain.yaml")
    assert chain.pair_fluents.count == 1 and chain.pair_fluents.locations == 1
    assert chain.pair_fluents.location_labels == (LOCATION,)


def test_the_habitat_stream_comes_after_every_earlier_stream(chain: WorldResult) -> None:
    # stage c3b's world:place_dynamics follows it (test_world_location_generation.py)
    assert STREAM_NAMES[-2] == "habitats" and STREAM_NAMES[-3] == "pair_conditions"
    assert list(WorldStreams(1).seeds())[-2] == "world:habitats"
    assert "world:habitats" in chain.stream_seeds


def test_the_location_fluent_has_the_fixed_profile_and_the_label_after_the_others(
    chain: WorldResult, runtime: RuntimeDefinition
) -> None:
    fluents = chain.pair_fluents
    assert fluents.labels == (NEAR, LOCATION)
    near, location = fluents.fluents
    assert not near.total and location.total
    assert (location.symmetry, location.functional, location.initial_rate) == (
        "general",
        True,
        None,
    )
    assert fluents.locations == (location,) and fluents.between_entities == (near,)
    record = chain.definition.record()
    symbol = next(s for s in record["symbols"] if s["label"] == LOCATION)
    assert symbol == {
        "label": LOCATION,
        "kind": "pair_fluent",
        "derived": False,
        "fluent": True,
        "arity": 2,
        "symmetry": "general",
        "functional": True,
        "total": True,
        "initial_rate": None,
    }
    # the profile is hashed: a different profile is a different identity
    changed = json.loads(json.dumps(record))
    next(s for s in changed["symbols"] if s["label"] == LOCATION)["total"] = False
    from semantic_world.world.identity import rule_set_id

    assert (
        rule_set_id(
            changed["symbols"], changed["literals"], changed["rules"], changed["event_types"]
        )
        != chain.rule_set_id
    )
    assert runtime.location_fluents == (location,)
    assert runtime.pair_targets(LOCATION) == runtime.place_count == 4
    assert runtime.pair_targets(NEAR) == runtime.entity_count
    assert runtime.initial_pairs is not None
    assert runtime.initial_pairs[1].shape == (runtime.entity_count, runtime.place_count)
    assert profile_violations(location, runtime.initial_pairs[1]) == []


def test_entities_csv_holds_the_initial_places_and_loads_back(
    chain: WorldResult, chain_folder: Path, runtime: RuntimeDefinition
) -> None:
    entities = pl.read_csv(chain_folder / "entities.csv", schema_overrides={"label": pl.Utf8})
    assert LOCATION in entities.columns and NEAR not in entities.columns
    assert entities.columns[-1] == LOCATION
    assert set(entities[LOCATION].to_list()) <= set(runtime.place_labels)
    pairs = pl.read_csv(
        chain_folder / "initial_pairs.csv", schema_overrides={"pair_fluent": pl.Utf8}
    )
    assert LOCATION not in pairs["pair_fluent"].to_list()  # the places are columns, not rows
    loaded = load_definition(chain_folder)
    assert loaded.rule_set_id == chain.rule_set_id
    assert loaded.initial_pairs is not None
    for a, b in zip(loaded.initial_pairs, runtime.initial_pairs, strict=True):  # type: ignore[arg-type]
        assert np.array_equal(a, b)
    assert initial_state(loaded).places_record(loaded) == {
        row["label"]: {LOCATION: row[LOCATION]} for row in entities.iter_rows(named=True)
    }
    # the places are the first draws of world:habitats with the configuration's seed
    assert chain.places is not None
    weights = habitat_weights(
        chain.places.requirement.values(),
        (12, 4),
        chain.config.places.habitat_strength,  # type: ignore[union-attr]
    )
    drawn = draw_places(weights, WorldStreams(chain.config.seed).habitats)
    assert [runtime.place_labels[q] for q in drawn] == entities[LOCATION].to_list()


def test_a_world_without_location_fluents_writes_none_of_the_new_keys(
    tiny_full: WorldResult, tmp_path: Path
) -> None:
    assert not tiny_full.pair_fluents.locations
    assert all(not s.get("total") for s in tiny_full.definition.record()["symbols"])
    folder = tiny_full.write(tmp_path / "tiny_full")
    entities = pl.read_csv(folder / "entities.csv")
    assert not any(c.startswith("PAIRFL.") for c in entities.columns)
    definition = tiny_full.definition.runtime()
    assert not definition.has_location_fluents
    for et in definition.event_types:
        if et.place:
            assert not et.performable and not et.precondition and not et.effects
    history = _generator(tiny_full).run(1, 3, STATS_STREAM)[0]
    assert history.places is None and history.initial_places is None
    data = history.to_json()
    assert "places" not in data and "initial_places" not in data


def test_the_loader_validates_location_records(chain: WorldResult, chain_folder: Path) -> None:
    from semantic_world.world.identity import rule_set_id

    record = json.loads((chain_folder / "definition.json").read_text(encoding="utf-8"))
    entities = pl.read_csv(chain_folder / "entities.csv", schema_overrides={"label": pl.Utf8})
    places = pl.read_csv(chain_folder / "places.csv", schema_overrides={"label": pl.Utf8})
    pairs = pl.read_csv(
        chain_folder / "initial_pairs.csv",
        schema_overrides={c: pl.Utf8 for c in ("pair_fluent", "first", "second")},
    )

    def rehash(data: dict) -> dict:
        data["rule_set_id"] = rule_set_id(
            data["symbols"], data["literals"], data["rules"], data["event_types"]
        )
        return data

    runtime_definition(record, entities, places, pairs)
    # SOME of an entity on a location fluent in the literal table
    broken = json.loads(json.dumps(record))
    broken["literals"].append(
        {
            "index": len(broken["literals"]),
            "key": f"SOME.{LOCATION}",
            "kind": "pair_some",
            "role": None,
            "pair_fluent": LOCATION,
        }
    )
    with pytest.raises(DefinitionError, match="always true"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # a location pair read by an event type between entities
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == APPROACH)
    entry["precondition"]["literals"][0] = {
        "kind": "pair",
        "pair_fluent": LOCATION,
        "first": "agent",
        "second": "patient",
        "value": True,
    }
    with pytest.raises(DefinitionError, match="between two entities"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # SAME on a location fluent in a place event type
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == GO)
    entry["precondition"]["literals"].append(
        {
            "kind": "pair_same",
            "pair_fluent": LOCATION,
            "first": "agent",
            "second": "patient",
            "value": True,
        }
    )
    with pytest.raises(DefinitionError, match="SAME on a location fluent"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # a delete on a location fluent
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == GO)
    entry["effects"][0]["value"] = False
    with pytest.raises(DefinitionError, match="deletes a pair of the location fluent"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # an effect on the place
    broken = json.loads(json.dumps(record))
    entry = next(e for e in broken["event_types"] if e["label"] == REST)
    entry["effects"][0]["role"] = "patient"
    with pytest.raises(DefinitionError, match="which is a place"):
        runtime_definition(rehash(broken), entities, places, pairs)
    # a total pair fluent before a pair fluent between entities, or without places
    broken = json.loads(json.dumps(record))
    symbols = {s["label"]: s for s in broken["symbols"]}
    symbols[NEAR]["total"], symbols[LOCATION]["total"] = True, False
    symbols[NEAR]["initial_rate"], symbols[LOCATION]["initial_rate"] = None, 0.0
    symbols[NEAR]["functional"] = True
    with pytest.raises(DefinitionError):
        runtime_definition(rehash(broken), entities, places, pairs)
    # the location column missing, or naming no place
    with pytest.raises(DefinitionError, match="not a place"):
        runtime_definition(
            record, entities.with_columns(pl.lit("PLACE.9.9").alias(LOCATION)), places, pairs
        )
    without = runtime_definition(record, entities.drop(LOCATION), places, pairs)
    assert without.initial_pairs is None  # a fixture gives the places apart


# ---------------------------------------------------------------------------------------------
# Habitats
# ---------------------------------------------------------------------------------------------


def _habitat_check(chain: WorldResult, strength: float) -> None:
    """Over ``HABITAT_SEEDS`` seeds of ``world:habitats``, each entity's share of each place is
    within three standard deviations of a binomial share, plus one draw, of its weight."""
    assert chain.places is not None
    n, m = 12, chain.places.count
    weights = habitat_weights(chain.places.requirement.values(), (n, m), strength)
    counts = np.zeros((n, m))
    for seed in range(1, HABITAT_SEEDS + 1):
        fluents = with_location_fluents(
            chain.pair_fluents,
            0,
            chain.places.requirement.values(),
            n,
            m,
            strength,
            WorldStreams(seed).habitats,
        )
        assert fluents is chain.pair_fluents  # with a count of 0 nothing is drawn
        drawn = with_location_fluents(
            dataclasses.replace(
                chain.pair_fluents,
                fluents=chain.pair_fluents.between_entities,
                initial_pairs=chain.pair_fluents.initial_pairs[:1],
            ),
            1,
            chain.places.requirement.values(),
            n,
            m,
            strength,
            WorldStreams(seed).habitats,
        )
        assert drawn.locations[0].label == LOCATION
        counts += drawn.initial_pairs[-1]
    expected = weights / weights.sum(axis=1, keepdims=True)
    margin = 3 * np.sqrt(expected * (1 - expected) / HABITAT_SEEDS) + 1 / HABITAT_SEEDS
    assert np.all(np.abs(counts / HABITAT_SEEDS - expected) <= margin)
    return expected


def test_habitats_are_uniform_with_strength_0(chain: WorldResult) -> None:
    expected = _habitat_check(chain, 0.0)
    assert np.allclose(expected, 1 / chain.places.count)  # type: ignore[union-attr]


def test_habitats_follow_the_weights_with_the_default_strength(chain: WorldResult) -> None:
    assert chain.config.places is not None and chain.config.places.habitat_strength == 1.0
    assert chain.places is not None
    able = sum(np.asarray(m, dtype=int) for m in chain.places.requirement.values())
    # the check is not vacuous: some entity is able to do more at one place than at another
    assert any((row.max() > row.min()) for row in able)
    expected = _habitat_check(chain, 1.0)
    for x in range(12):
        for q in range(chain.places.count):
            for r in range(chain.places.count):
                if able[x, q] > able[x, r]:
                    assert expected[x, q] > expected[x, r]


def test_the_habitat_draws_leave_every_other_draw_unchanged(chain: WorldResult) -> None:
    """A world with location fluents differs from the same world with ``locations: 0`` only in
    the location fluent, the moves and acts of the file, and what depends on them: the
    taxonomy, the places, the pair fluents between entities and their initial pairs, and the
    fluents are the same."""
    data = yaml.safe_load((DATA / "tiny_place_chain.yaml").read_text(encoding="utf-8"))
    data["pair_fluents"]["locations"] = 0
    text = yaml.safe_load(EVENT_FILE.read_text(encoding="utf-8"))
    del text["place_event_types"]
    text["event_types"][APPROACH]["precondition"] = "NOT PAIRFL.1(agent, patient)"
    folder = Path(chain.config.source).parent
    stripped = folder / "events" / "_stripped_place_chain.yaml"
    stripped.write_text(yaml.safe_dump(text), encoding="utf-8")
    try:
        data["event_types"]["event_file"] = "events/_stripped_place_chain.yaml"
        without = define(config_from_mapping(data, source=str(DATA / "tiny_place_chain.yaml")))
    finally:
        stripped.unlink()
    assert without.taxonomy.instances.labels == chain.taxonomy.instances.labels
    assert without.places is not None and chain.places is not None
    assert np.array_equal(
        without.places.taxonomy.instances.values, chain.places.taxonomy.instances.values
    )
    assert without.pair_fluents.labels == (NEAR,)
    assert np.array_equal(
        without.pair_fluents.initial_pairs[0], chain.pair_fluents.initial_pairs[0]
    )
    assert np.array_equal(without.fluents.initial_values, chain.fluents.initial_values)


# ---------------------------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------------------------


def test_the_records_hold_the_moves_and_the_acts(
    chain: WorldResult, runtime: RuntimeDefinition
) -> None:
    go, rest = runtime.event_type(GO), runtime.event_type(REST)
    assert go.performable and go.move and not go.act and go.location == LOCATION
    assert rest.performable and rest.act and not rest.move and rest.location == LOCATION
    assert [lit.text for lit in go.precondition] == [f"NOT {LOCATION}(agent, patient)"]
    assert go.precondition[0].origin == "move"
    assert [e.text for e in go.effects] == [f"{LOCATION}(agent, patient) := 1"]
    assert [lit.text for lit in rest.precondition] == [
        "agent.BOOLFL.1",
        f"{LOCATION}(agent, patient)",
    ]
    assert rest.precondition[1].origin == "act"
    assert [e.text for e in rest.effects] == [
        f"when SOMEOF.{LOCATION}(patient) then agent.BOOLFL.1 := 0"
    ]
    approach = runtime.event_type(APPROACH)
    assert approach.precondition[0].text == f"SAME.{LOCATION}(agent, patient)"
    assert chain.places is not None
    for et in chain.places.event_types:
        assert et.explicit and et.performable
    assert {et.label for et in chain.places.performable} == {GO, REST}
    record = chain.definition.record()
    entry = next(e for e in record["event_types"] if e["label"] == GO)
    assert entry["explicit"] and entry["precondition"]["literals"][0]["origin"] == "move"
    assert entry["effects"][0] == {
        "kind": "pair", "pair_fluent": LOCATION, "first": "agent", "second": "patient",
        "value": True, "expression": f"{LOCATION}(agent, patient) := 1",
    }  # fmt: skip
    # the explicit requirement of rest reads the place
    assert chain.places.event_type(REST).expression == "patient.PLACEPROPERTY.1"
    assert chain.places.event_type(GO).expression == "TRUE"


def test_legal_and_derive_agree_with_brute_force_in_random_states(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261023)
    seen = {label: 0 for label in _performable(runtime)}
    for _ in range(RANDOM_STATES):
        state = _random_state(runtime, rng)
        brute_state = _brute_state(runtime, state)
        numeric = _brute_numeric(runtime, state)
        pairs = _brute_pairs(runtime, state)
        assert derive(runtime, state).true_fluents(runtime) == brute.derived_record(
            brute_state, numeric, pairs
        )
        for label in _performable(runtime):
            got = [
                Event(label, b).binding_labels(runtime)
                for b in legal_bindings(runtime, state, label)
            ]
            assert got == brute.legal_bindings(label, brute_state, None, numeric, pairs), label
            seen[label] += bool(got)
    for label in (GO, REST, APPROACH, EAT, "EVENTTYPE1.1"):
        assert seen[label] > 0, label


def test_applying_together_equals_one_at_a_time_in_every_order(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261024)
    sets = moves = 0
    while sets < RANDOM_SETS:
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        if not events:
            continue
        chosen: list[Event] = []
        size = int(rng.integers(1, 5))
        for i in rng.permutation(len(events)):
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
        for fluent, matrix in zip(runtime.pair_fluents, together.pairs, strict=True):
            assert profile_violations(fluent, matrix) == []
        moves += sum(1 for e in chosen if runtime.event_type(e.event_type).move)
        sets += 1
    assert moves > 0


def test_interference_verdicts_agree_with_brute_force_over_places(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(12)
    found = {"move_vs_same": 0, "move_vs_someof": 0, "two_moves": 0, "move_vs_act": 0}
    for _ in range(200):
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        for a, b in itertools.combinations(events, 2):
            verdict = brute.interfere(_brute_event(runtime, a), _brute_event(runtime, b))
            assert interferes(runtime, a, b) == verdict
            kinds = {runtime.event_type(e.event_type).label for e in (a, b)}
            if verdict and GO in kinds:
                if APPROACH in kinds:
                    found["move_vs_same"] += 1
                elif REST in kinds:
                    found["move_vs_act"] += 1
                elif kinds == {GO} and a.binding[0] == b.binding[0]:
                    found["two_moves"] += 1
    assert found["move_vs_same"] > 0 and found["move_vs_act"] > 0 and found["two_moves"] > 0


def test_a_move_replaces_the_old_place_and_an_act_happens_at_the_agents_place(
    runtime: RuntimeDefinition,
) -> None:
    state = initial_state(runtime)
    agent = 0
    here = _place_of(runtime, state, agent)
    there = next(q for q in range(runtime.place_count) if q != here)
    assert (agent, there) in legal_bindings(runtime, state, GO, [agent])
    assert (agent, here) not in legal_bindings(runtime, state, GO, [agent])
    with pytest.raises(IllegalEventError, match="move|compiled"):
        apply(runtime, state, [Event(GO, (agent, here))])
    after = apply(runtime, state, [Event(GO, (agent, there))])
    k = runtime.pair_index(LOCATION)
    assert _place_of(runtime, after, agent) == there
    assert np.array_equal(
        np.delete(after.pairs[k], agent, axis=0), np.delete(state.pairs[k], agent, axis=0)
    )
    # an act: legal where the agent is (and the place is able), nowhere else
    for binding in legal_bindings(runtime, after, REST):
        assert binding[1] == _place_of(runtime, after, binding[0])
    text = Event(GO, (agent, there)).describe(runtime)
    assert runtime.place_labels[there] in text and runtime.entity_labels[agent] in text


# ---------------------------------------------------------------------------------------------
# Episodes, histories, and replay
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("initial", ["keep", "redraw"])
def test_episodes_keep_every_entity_at_one_place_and_the_chain_in_order(
    chain: WorldResult, initial: str
) -> None:
    generator = _generator(chain, initial=initial, size=(2, 5), steps=(4, 8))
    definition = generator.definition
    k = definition.pair_index(LOCATION)
    counts = {"moves": 0, "acts": 0, "approaches": 0, "eats": 0, "changes": 0}
    for history in generator.run(5, EPISODES, STATS_STREAM):
        assert history.places is not None and history.initial_places is not None
        states = replay(definition, history)
        for state in states:
            assert profile_violations(definition.pair_fluents[k], state.pairs[k]) == []
        participants = [definition.entity_index(p) for p in history.participants]
        episode_places = {definition.place_index(q) for q in history.places}
        for p in participants:
            assert _place_of(definition, states[0], p) in episode_places
            assert history.initial_places[definition.entity_labels[p]] == {
                LOCATION: definition.place_labels[_place_of(definition, states[0], p)]
            }
        assert (
            len(episode_places)
            <= len({_place_of(definition, states[0], p) for p in participants}) + 1
        )
        for step, before, after in zip(history.steps, states[:-1], states[1:], strict=True):
            for event in step.events:
                et = definition.event_type(event.type)
                agent = definition.entity_index(event.agent)
                if et.move:
                    counts["moves"] += 1
                    target = definition.place_index(event.patient)  # type: ignore[arg-type]
                    assert target in episode_places
                    assert _place_of(definition, before, agent) != target
                    assert _place_of(definition, after, agent) == target
                    movers = {
                        definition.entity_index(e.agent)
                        for e in step.events
                        if definition.event_type(e.type).move
                    }
                    others = [p for p in range(definition.entity_count) if p not in movers]
                    assert np.array_equal(before.pairs[k][others], after.pairs[k][others])
                    changes = [c for c in event.changes if c.pair]
                    assert [(c.second, c.to) for c in changes] == [  # type: ignore[union-attr]
                        (definition.place_labels[_place_of(definition, before, agent)], False),
                        (event.patient, True),
                    ]
                    counts["changes"] += len(changes)
                elif et.act:
                    counts["acts"] += 1
                    assert definition.place_index(event.patient) == _place_of(
                        definition, before, agent
                    )  # type: ignore[arg-type]
                    assert np.array_equal(before.pairs[k], after.pairs[k])
                elif event.type == APPROACH:
                    counts["approaches"] += 1
                    patient = definition.entity_index(event.patient)  # type: ignore[arg-type]
                    assert _place_of(definition, before, agent) == _place_of(
                        definition, before, patient
                    )
                elif event.type == EAT:
                    counts["eats"] += 1
                    patient = definition.entity_index(event.patient)  # type: ignore[arg-type]
                    n = definition.pair_index(NEAR)
                    assert before.pairs[n][agent, patient]  # an approach came before
        assert History.from_json(json.loads(json.dumps(history.to_json()))) == history
    assert all(v > 0 for v in counts.values()), counts


def test_scene_places_all_and_initial_location_seed(chain: WorldResult) -> None:
    every = _generator(chain, places="all")
    definition = every.definition
    for history in every.run(6, 50, STATS_STREAM):
        assert history.places == definition.place_labels
    seeded = _generator(chain, initial_location="seed", extra_places=0)
    for history in seeded.run(7, 50, STATS_STREAM):
        places = {at[LOCATION] for at in history.initial_places.values()}  # type: ignore[union-attr]
        assert places == {history.initial_places[history.seed][LOCATION]}  # type: ignore[index]
        assert history.places == tuple(places)
    redrawn = _generator(chain, initial="redraw", initial_location="seed", extra_places=2)
    for history in redrawn.run(8, 50, STATS_STREAM):
        places = {at[LOCATION] for at in history.initial_places.values()}  # type: ignore[union-attr]
        assert len(places) == 1 and len(history.places) <= 3  # type: ignore[arg-type]
        assert places <= set(history.places)  # type: ignore[arg-type]
    none = _generator(chain, extra_places=0)
    for history in none.run(9, 50, STATS_STREAM):
        assert set(history.places) == {at[LOCATION] for at in history.initial_places.values()}  # type: ignore[union-attr]


def test_the_place_draws_come_after_every_other_draw_of_the_part(chain: WorldResult) -> None:
    """The participants, the Boolean and numeric initial values, and the pairs between entities
    of an episode are the same whatever the place settings, because the place draws come after
    them."""
    a = _generator(chain, initial="redraw").run(2, 30, STATS_STREAM)
    b = _generator(chain, initial="redraw", places="all", initial_location="seed").run(
        2, 30, STATS_STREAM
    )
    differ = 0
    for x, y in zip(a, b, strict=True):
        assert x.participants == y.participants
        assert x.initial == y.initial and x.initial_numeric == y.initial_numeric
        assert x.initial_pairs == y.initial_pairs
        differ += x.initial_places != y.initial_places
    assert differ > 0


def test_uniform_event_type_treats_place_events_as_two_place(chain: WorldResult) -> None:
    both = _generator(chain, policy="uniform_event_type", transitive_share=1.0, steps=(3, 5))
    kinds = {"place": 0, "two_place": 0, "one_place": 0}
    for history in both.run(3, 100, STATS_STREAM):
        for event in history.events:
            et = both.definition.event_type(event.type)
            kinds["place" if et.place else "two_place" if et.arity == 2 else "one_place"] += 1
    assert kinds["place"] > 0 and kinds["two_place"] > 0 and kinds["one_place"] == 0
    none = _generator(chain, policy="uniform_event_type", transitive_share=0.0, steps=(3, 5))
    for history in none.run(3, 50, STATS_STREAM):
        assert all(event.patient is None for event in history.events)


def test_histories_write_places_and_tampering_fails(chain: WorldResult) -> None:
    generator = _generator(chain)
    definition = generator.definition
    history = next(
        h
        for h in generator.run(4, 100, STATS_STREAM)
        if any(definition.event_type(e.type).move for e in h.events)
    )
    data = history.to_json()
    assert list(data)[:4] == ["label", "seed", "participants", "places"]
    assert "initial_places" in data and list(data).index("initial_places") < list(data).index(
        "initial_pairs"
    )
    move = next(e for e in history.events if definition.event_type(e.type).move)
    assert move.patient in history.places  # type: ignore[operator]
    tampered = json.loads(json.dumps(data))
    for step in tampered["steps"]:
        for event in step["events"]:
            if event["label"] == move.label:
                event["changes"] = event["changes"][1:]  # drop the deletion of the old place
    with pytest.raises(HistoryError, match="pair changes"):
        replay(definition, History.from_json(tampered))
    missing = json.loads(json.dumps(data))
    del missing["initial_places"]
    with pytest.raises(HistoryError, match="places"):
        replay(definition, History.from_json(missing))
    wrong = json.loads(json.dumps(data))
    wrong["initial_places"][history.participants[0]][LOCATION] = "PLACE.9.9"
    with pytest.raises(HistoryError, match="not a place"):
        replay(definition, History.from_json(wrong))


def test_simulate_reads_the_scene_keys_and_writes_places(
    chain_folder: Path, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        yaml.safe_dump({"scene": {"places": "all", "extra_places": 3, "initial_location": "seed"}}),
        encoding="utf-8",
    )
    settings = read_scene_settings(yaml.safe_load(corpus.read_text()), str(corpus))
    assert (settings.places, settings.extra_places, settings.initial_location) == ("all", 3, "seed")
    assert settings.resolved()["places"] == "all"
    with pytest.raises(ConfigError, match="initial_location"):
        read_scene_settings({"scene": {"initial_location": "nowhere"}}, "<test>")
    with pytest.raises(ConfigError, match="extra_places"):
        read_scene_settings({"scene": {"extra_places": -1}}, "<test>")
    path, histories = simulate(chain_folder, 10, settings=settings, out=tmp_path / "episodes.jsonl")
    definition = load_definition(chain_folder)
    for history in read_histories(path):
        assert history.places == definition.place_labels
        replay(definition, history)
    assert read_histories(path) == histories


def test_the_corpus_reads_the_scene_keys_and_refuses_the_chain_world(tmp_path: Path) -> None:
    data = yaml.safe_load((Path("data/corpus/tiny.yaml")).read_text(encoding="utf-8"))
    data["scene"] = {**(data.get("scene") or {}), "places": "all", "extra_places": 2}
    config = corpus_config_from_mapping(data, source=str(tmp_path / "corpus.yaml"))
    assert config.scene.places == "all" and config.scene.extra_places == 2
    assert config.scene.initial_location == "world"
    assert config.resolved()["scene"]["initial_location"] == "world"
    data["scene"]["initial_location"] = "elsewhere"
    with pytest.raises(ConfigError, match="initial_location"):
        corpus_config_from_mapping(data, source=str(tmp_path / "corpus.yaml"))
    del data["scene"]["initial_location"]
    data["world"] = {"config": str(DATA / "tiny_place_chain.yaml"), "seed": 1}
    from semantic_world.corpus import load_world

    with pytest.raises(CorpusError, match="pair fluents"):
        load_world(corpus_config_from_mapping(data, source=str(tmp_path / "corpus.yaml")))


# ---------------------------------------------------------------------------------------------
# Broken event files: one test per rule, each naming the file and the entry
# ---------------------------------------------------------------------------------------------


def _event_file_run(
    tmp_path: Path, data: dict[str, Any], name: str = "events.yaml", **world_changes: Any
) -> WorldResult:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    world = yaml.safe_load((DATA / "tiny_place_chain.yaml").read_text(encoding="utf-8"))
    world["event_types"]["event_file"] = str(path)
    for key, value in world_changes.items():
        world[key] = {**world.get(key, {}), **value}
    return define(config_from_mapping(world, source=str(DATA / "tiny_place_chain.yaml")))


BASE = yaml.safe_load(EVENT_FILE.read_text(encoding="utf-8"))


def _with_place(entry: dict[str, Any], label: str = GO) -> dict[str, Any]:
    data = json.loads(json.dumps(BASE))
    data["place_event_types"][label] = entry
    return data


def _with_entity(precondition: str, label: str = APPROACH) -> dict[str, Any]:
    data = json.loads(json.dumps(BASE))
    data["event_types"][label] = {"requirement": "TRUE", "precondition": precondition}
    return data


BROKEN = [
    (
        "act_without_its_location_literal",
        _with_place({"kind": "act", "precondition": f"NOT {LOCATION}(agent, patient)"}, REST),
        REST,
        "act needs its location literal",
    ),
    (
        "move_requires_the_place",
        _with_place({"kind": "move", "precondition": f"{LOCATION}(agent, patient)"}),
        GO,
        "never at the bound place",
    ),
    (
        "delete_on_a_location_fluent",
        _with_place({"kind": "move", "effects": [f"{LOCATION}(agent, patient) := 0"]}),
        GO,
        "deletes a pair of the location fluent",
    ),
    (
        "act_moves",
        _with_place({"kind": "act", "effects": [f"{LOCATION}(agent, patient) := 1"]}, REST),
        REST,
        "but the entry is an act",
    ),
    (
        "some_of_an_entity_on_a_location_fluent",
        _with_place({"kind": "move", "precondition": f"SOME.{LOCATION}(agent)"}),
        GO,
        "always true",
    ),
    (
        "someof_of_the_agent_on_a_location_fluent",
        _with_place({"kind": "move", "precondition": f"SOMEOF.{LOCATION}(agent)"}),
        GO,
        "SOMEOF reads the place",
    ),
    (
        "same_in_a_place_event_type",
        _with_place({"kind": "move", "precondition": f"SAME.{LOCATION}(agent, patient)"}),
        GO,
        "between an entity and a place",
    ),
    ("unknown_kind", _with_place({"kind": "fly"}), GO, "kind"),
    (
        "unknown_location_fluent",
        _with_place({"kind": "move", "location": "PAIRFL.7"}),
        GO,
        "not a location fluent",
    ),
    (
        "a_pair_fluent_between_entities_as_location",
        _with_place({"kind": "move", "location": NEAR}),
        GO,
        "not a location fluent",
    ),
    (
        "unknown_place_event_type",
        _with_place({"kind": "move"}, "PLACEEVENTTYPE.9"),
        "PLACEEVENTTYPE.9",
        "unknown place event type",
    ),
    (
        "effect_on_the_place",
        _with_place({"kind": "move", "effects": ["patient.BOOLFL.1 := 1"]}),
        GO,
        "which is a place",
    ),
    (
        "literal_of_the_place",
        _with_place({"kind": "move", "precondition": "patient.BOOLFL.1"}),
        GO,
        "which is a place",
    ),
    (
        "pair_between_entities_in_a_place_event_type",
        _with_place({"kind": "move", "precondition": f"{NEAR}(agent, patient)"}),
        GO,
        "only SOME and SOMEOF of the agent",
    ),
    (
        "pair_effect_between_entities_in_a_place_event_type",
        _with_place({"kind": "move", "effects": [f"{NEAR}(agent, patient) := 1"]}),
        GO,
        "pair between entities",
    ),
    (
        "condition_on_the_place",
        _with_place(
            {"kind": "move", "effects": ["when patient.PROPERTY.1 then agent.BOOLFL.2 := 1"]}
        ),
        GO,
        "which is a place",
    ),
    (
        "requirement_reads_a_fluent",
        _with_place({"kind": "move", "requirement": "agent.BOOLFL.1"}),
        GO,
        "reads a fluent",
    ),
    (
        "requirement_reads_an_entity_feature_of_the_place",
        _with_place({"kind": "move", "requirement": "patient.PROPERTY.1"}),
        GO,
        "unknown place feature",
    ),
    (
        "entity_event_type_reads_a_location_pair",
        _with_entity(f"{LOCATION}(agent, patient)"),
        APPROACH,
        "between two entities",
    ),
    (
        "entity_event_type_reads_some_on_a_location_fluent",
        _with_entity(f"SOME.{LOCATION}(agent)"),
        APPROACH,
        "always true",
    ),
    (
        "entity_event_type_writes_a_location_fluent",
        {
            **json.loads(json.dumps(BASE)),
            "event_types": {
                **BASE["event_types"],
                APPROACH: {"requirement": "TRUE", "effects": [f"{LOCATION}(agent, patient) := 1"]},
            },
        },
        APPROACH,
        "only a move",
    ),
]


@pytest.mark.parametrize("name, data, label, message", BROKEN, ids=[b[0] for b in BROKEN])
def test_a_broken_place_entry_names_the_file_and_the_entry(
    tmp_path: Path, name: str, data: dict[str, Any], label: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message) as info:
        _event_file_run(tmp_path, data, f"{name}.yaml")
    text = str(info.value)
    assert f"{name}.yaml" in text and label in text


def test_place_entries_need_a_location_fluent_and_name_one_among_several(tmp_path: Path) -> None:
    data = json.loads(json.dumps(BASE))
    data["event_types"][APPROACH]["precondition"] = "NOT PAIRFL.1(agent, patient)"
    with pytest.raises(ConfigError, match="no location fluent") as info:
        _event_file_run(tmp_path, data, "none.yaml", pair_fluents={"locations": 0})
    assert "none.yaml" in str(info.value) and GO in str(info.value)
    with pytest.raises(ConfigError, match="name the one") as info:
        _event_file_run(tmp_path, BASE, "two.yaml", pair_fluents={"locations": 2})
    assert "two.yaml" in str(info.value)
    data = json.loads(json.dumps(BASE))
    data["place_event_types"][GO]["location"] = "PAIRFL.3"
    data["place_event_types"][REST]["location"] = "PAIRFL.2"
    data["event_types"][APPROACH]["precondition"] = (
        "SAME.PAIRFL.3(agent, patient) AND NOT PAIRFL.1(agent, patient)"
    )
    result = _event_file_run(tmp_path, data, "two_named.yaml", pair_fluents={"locations": 2})
    assert result.pair_fluents.labels == (NEAR, "PAIRFL.2", "PAIRFL.3")
    runtime = result.definition.runtime()
    assert (
        runtime.event_type(GO).location == "PAIRFL.3"
        and runtime.event_type(REST).location == "PAIRFL.2"
    )
    history = _generator(result).run(1, 20, STATS_STREAM)[0]
    assert set(history.initial_places[history.seed]) == {"PAIRFL.2", "PAIRFL.3"}  # type: ignore[index]
    replay(runtime, history)


def test_an_event_file_never_defines_a_location_fluent(tmp_path: Path) -> None:
    data = json.loads(json.dumps(BASE))
    data["pair_fluents"][NEAR]["total"] = True
    with pytest.raises(ConfigError, match="location fluent"):
        _event_file_run(tmp_path, data)
    data = json.loads(json.dumps(BASE))
    data["locations"] = 1
    with pytest.raises(ConfigError, match="not defined in an event file"):
        _event_file_run(tmp_path, data)
    loaded = load_event_file(EVENT_FILE)
    assert set(loaded.place_event_types) == {GO, REST}
    assert (
        loaded.place_event_types[GO].kind == "move" and loaded.place_event_types[REST].kind == "act"
    )


def test_the_given_literal_and_effect_are_not_added_twice(tmp_path: Path) -> None:
    data = json.loads(json.dumps(BASE))
    data["place_event_types"][GO]["precondition"] = f"NOT {LOCATION}(agent, patient)"
    data["place_event_types"][GO]["effects"] = [f"{LOCATION}(agent, patient) := 1"]
    data["place_event_types"][REST]["precondition"] = (
        f"{LOCATION}(agent, patient) AND agent.BOOLFL.1"
    )
    result = _event_file_run(tmp_path, data)
    go, rest = result.places.event_type(GO), result.places.event_type(REST)  # type: ignore[union-attr]
    assert len(go.precondition) == 1 and len(go.effects) == 1 and go.precondition[0].origin is None
    assert [lit.text for lit in rest.precondition] == [
        f"{LOCATION}(agent, patient)",
        "agent.BOOLFL.1",
    ]
    assert rest.precondition[0].origin is None


# ---------------------------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------------------------


def test_the_hand_fixtures_cover_the_location_cases() -> None:
    names = [p.stem for p in FIXTURES_DIR.glob("hand_*.json")]
    for case in (
        "move_replaces_old_place",
        "move_to_own_place_blocked",
        "act_at_the_place_and_not_elsewhere",
        "same_holds_event_legal",
        "same_fails_after_move",
        "someof_of_a_place_in_a_precondition",
        "interference_move_and_same",
    ):
        assert any(name.endswith(case) for name in names), case
    fixture = read_fixture(FIXTURES_DIR / "hand_38_move_replaces_old_place.json")
    assert "initial_places" in fixture and fixture["version"] == 1
    assert list(fixture)[:9] == [
        "version", "name", "description", "rule_set_id", "definition", "entities", "places",
        "initial", "initial_numeric",
    ]  # fmt: skip
    assert list(fixture)[9:11] == ["initial_pairs", "initial_places"]
    for name in names:
        fixture = read_fixture(FIXTURES_DIR / f"{name}.json")
        assert set(fixture["initial_places"]) == {e["label"] for e in fixture["entities"]}


def test_generated_fixtures_of_the_chain_world_pass_both_checkers(chain: WorldResult) -> None:
    fixtures = world_fixtures(chain.definition, "tiny_place_chain")
    assert len(fixtures) == 6
    moves = 0
    for fixture in fixtures:
        assert fixture["initial_places"] and fixture["rule_set_id"] == chain.rule_set_id
        for step in fixture["steps"]:
            assert GO in step["legal"] and REST in step["legal"]
            moves += sum(1 for e in step["events"] if e["event_type"] == GO)
            if "pairs" in step:
                assert sum(1 for p in step["pairs"] if p[0] == LOCATION) == 12
        check_fixture(fixture)
        check_fixture_brute(fixture)
        round_trip = json.loads(to_json(fixture, indent=1, sort_keys=False))
        check_fixture(round_trip)
    assert moves > 0


def test_a_state_needs_every_entity_at_a_place(runtime: RuntimeDefinition) -> None:
    true = {label: [] for label in runtime.entity_labels}
    state = State.from_true(
        runtime, true, None, [], {runtime.entity_labels[0]: {LOCATION: runtime.place_labels[3]}}
    )
    assert _place_of(runtime, state, 0) == 3
    assert state.places_record(runtime)[runtime.entity_labels[1]] == {
        LOCATION: runtime.place_labels[_place_of(runtime, initial_state(runtime), 1)]
    }
    with pytest.raises(WorldError, match="not a place"):
        State.from_true(
            runtime, true, None, [], {runtime.entity_labels[0]: {LOCATION: "PLACE.9.9"}}
        )
    with pytest.raises(WorldError, match="not a location fluent"):
        State.from_true(
            runtime, true, None, [], {runtime.entity_labels[0]: {NEAR: runtime.place_labels[0]}}
        )
    bare = dataclasses.replace(runtime, initial_pairs=None)
    with pytest.raises(WorldError, match="no place"):
        State.from_true(
            bare, true, None, [], {runtime.entity_labels[0]: {LOCATION: runtime.place_labels[0]}}
        )
    # the habitat weights the runtime computes for a redraw are the generator's
    assert place_able_table(runtime)[GO].all()
