"""Stage b1: conditional effects (``docs/specs/WORLD_AND_LANGUAGE.md``, "Conditional effects").

The acceptance tests of the stage: every conditional effect fires exactly when its condition
held in the step's starting state, by the brute-force evaluator, on 1,000 random states of the
tiny world with every effect conditional; applying a step's events together equals applying them
one at a time in every order, on 1,000 random non-interfering sets that include conditional
effects, and events whose conditions read what another writes are reported as interfering; every
condition is satisfiable with its event type's requirement and precondition, and no event type
has contradictory or empty effects; with ``conditional_share: 0`` the tiny and default
definitions are the definitions of stage a8; replaying every history reproduces every recorded
change and nothing else; the loader and the event file validate conditions; and the statistics
count them.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from world_support import mapping_before_phase_c

from semantic_world.figures import SNAPSHOT_FILE, load_snapshot
from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.config import Config, config_from_mapping, load_config
from semantic_world.world.definition import RuntimeDefinition, runtime_definition
from semantic_world.world.dynamics import ConditionLiteral, Effect, PairLiteral, parse_condition
from semantic_world.world.episodes import EpisodeGenerator, Relatedness
from semantic_world.world.errors import DefinitionError, InterferenceError
from semantic_world.world.event_types import check_dynamics
from semantic_world.world.fixtures import (
    BruteEvent,
    BruteForce,
    fixture_inputs,
    fixture_places,
    read_fixture,
)
from semantic_world.world.fluents import derived_initial_values
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.history import replay
from semantic_world.world.identity import rule_set_id
from semantic_world.world.runtime import (
    Event,
    State,
    able_table,
    apply,
    fired_effects,
    legal_bindings,
    static_facts,
)
from semantic_world.world.stats import world_stats

DATA = Path("data/world")
RANDOM_STATES = 1000
RANDOM_SETS = 1000


def a8_identity(name: str) -> str:
    """The rule set of a world before the stage (the a8 reference run), kept as the baseline
    of ``conditional_share: 0`` in the snapshot of figures."""
    return load_snapshot(SNAPSHOT_FILE)["baselines"]["conditional_effects_off"][name]["rule_set_id"]


def tiny_config(share: float, **effects: Any) -> Config:
    # the tiny world of stage c3b: the phase (c) settings that tiny.yaml carries since stage
    # c4a are off, so that these tests keep the conditional effects as their subject
    data = mapping_before_phase_c()
    data.setdefault("event_types", {})["effects"] = {"conditional_share": share, **effects}
    return config_from_mapping(data, source=str(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def conditional() -> WorldResult:
    """The tiny world with every effect conditional."""
    return define(tiny_config(1.0))


@pytest.fixture(scope="module")
def runtime(conditional: WorldResult) -> RuntimeDefinition:
    return conditional.definition.runtime()


@pytest.fixture(scope="module")
def brute(conditional: WorldResult) -> BruteForce:
    record, entities, _ = fixture_inputs(conditional.definition)
    return BruteForce(record, entities, fixture_places(conditional.definition) or ())


@pytest.fixture(scope="module")
def default() -> WorldResult:
    return define(load_config(DATA / "default.yaml"))


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
    """A random state: every Boolean fluent at 0 or 1, every numeric fluent uniform in its
    range, and one value in four exactly at a quarter point of the range, so that thresholds
    are hit."""
    shape = (definition.entity_count, len(definition.base_fluents))
    values = rng.integers(0, 2, size=shape, dtype=np.uint8)
    low, high = definition.range_bounds
    numeric = rng.uniform(
        low, high, size=(definition.entity_count, len(definition.numeric_fluents))
    )
    at_point = rng.random(numeric.shape) < 0.25
    points = low + rng.choice([0.25, 0.5, 0.75], size=numeric.shape) * (high - low)
    numeric = np.where(at_point, points, numeric)
    return State(values, numeric)


def _brute_numeric(definition: RuntimeDefinition, state: State) -> dict[str, dict[str, float]]:
    return {
        label: {f: float(state.numeric[i, j]) for j, f in enumerate(definition.numeric_fluents)}
        for i, label in enumerate(definition.entity_labels)
    }


def _brute_state(definition: RuntimeDefinition, state: State) -> dict[str, frozenset[str]]:
    return {
        label: frozenset(f for j, f in enumerate(definition.base_fluents) if state.values[i, j])
        for i, label in enumerate(definition.entity_labels)
    }


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if et.performable]


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


# ---------------------------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------------------------


def test_every_effect_of_the_conditional_tiny_world_has_a_condition(
    conditional: WorldResult,
) -> None:
    leaves = conditional.event_types.leaves
    effects = [e for et in leaves for e in et.effects]
    assert effects
    dropped = conditional.event_types.stats["conditions_dropped"]
    assert sum(1 for e in effects if e.condition) == len(effects) - dropped
    assert dropped == 0
    kinds = {lit.kind for e in effects for lit in e.condition}
    assert {"feature", "fluent"} <= kinds <= {"feature", "fluent", "numeric"}, kinds
    for effect in effects:
        assert 1 <= len(effect.condition) <= 2
        assert effect.text.startswith("when ") and " then " in effect.text
        record = effect.record()
        assert record["condition"]["expression"] == " AND ".join(
            lit.text for lit in effect.condition
        )
        assert [
            ConditionLiteral(r["role"], r["symbol"], r["operator"] == ">", r["threshold"])
            if r["kind"] == "numeric"
            else ConditionLiteral(r["role"], r["symbol"], r["value"])
            for r in record["condition"]["literals"]
        ] == list(effect.condition)


def test_the_record_and_the_identity_change_only_with_a_condition() -> None:
    # the baseline of conditional_share 0 is kept on tiny.yaml as it is (with the phase (c)
    # settings since stage c4a), so the plain world here is tiny.yaml with the share at 0
    data = yaml.safe_load((DATA / "tiny.yaml").read_text(encoding="utf-8"))
    data.setdefault("event_types", {})["effects"] = {"conditional_share": 0.0}
    plain = define(config_from_mapping(data, source=str(DATA / "tiny.yaml")))
    assert plain.rule_set_id == a8_identity("tiny")
    for et in plain.event_types.event_types:
        for effect in et.effects:
            assert not effect.condition
            assert "condition" not in effect.record()
            assert effect.record()["expression"] == effect.assignment
    record = plain.definition.record()
    assert all("condition" not in e for et in record["event_types"] for e in et["effects"])
    # the condition is part of the hashed event_types table
    changed = json.loads(json.dumps(record))
    entry = next(e for e in changed["event_types"] if e["effects"])
    entry["effects"][0]["condition"] = {
        "literals": [{"role": "agent", "kind": "feature", "symbol": "PROPERTY.1", "value": True}],
        "expression": "agent.PROPERTY.1",
    }
    assert rule_set_id(
        changed["symbols"], changed["literals"], changed["rules"], changed["event_types"]
    ) != rule_set_id(record["symbols"], record["literals"], record["rules"], record["event_types"])


def test_the_default_file_lists_the_condition_settings() -> None:
    """The default world with the share at 0 is the a8 world:
    ``tests/world/test_world_off_settings.py`` checks it against the snapshot's baseline."""
    data = yaml.safe_load((DATA / "default.yaml").read_text(encoding="utf-8"))
    assert data["event_types"]["effects"]["conditional_share"] == 0.2
    assert data["event_types"]["effects"]["condition_literals"] == {1: 0.7, 2: 0.3}


def test_the_two_settings_are_read_and_validated() -> None:
    config = load_config(DATA / "default.yaml")
    assert config.event_types.effects.conditional_share == 0.2
    assert config.event_types.effects.condition_literals == {1: 0.7, 2: 0.3}
    defaults = config_from_mapping({}).event_types.effects
    assert defaults.conditional_share == 0.2 and defaults.condition_literals == {1: 0.7, 2: 0.3}
    with pytest.raises(ConfigError, match="conditional_share"):
        tiny_config(1.5)
    with pytest.raises(ConfigError, match="condition_literals"):
        tiny_config(0.5, condition_literals={0: 1})
    custom = tiny_config(0.5, condition_literals={2: 1})
    assert custom.event_types.effects.condition_literals == {2: 1.0}
    world = define(custom)
    assert all(len(e.condition) in (0, 2) for et in world.event_types.leaves for e in et.effects)
    assert any(e.condition for et in world.event_types.leaves for e in et.effects)


def test_the_loader_reads_and_validates_conditions(
    conditional: WorldResult, runtime: RuntimeDefinition
) -> None:
    record = json.loads(json.dumps(runtime.record))
    entities = [
        dict(zip(["label", "leaf", *runtime.free_features, *runtime.scalars], row, strict=True))
        for row in zip(
            runtime.entity_labels,
            runtime.leaves,
            *[runtime.free_values[:, j].tolist() for j in range(len(runtime.free_features))],
            *[runtime.scalar_values[:, j].tolist() for j in range(len(runtime.scalars))],
            strict=True,
        )
    ]

    def rehash(data: dict) -> dict:
        data["rule_set_id"] = rule_set_id(
            data["symbols"], data["literals"], data["rules"], data["event_types"]
        )
        return data

    places = fixture_places(conditional.definition)
    loaded = runtime_definition(rehash(json.loads(json.dumps(record))), entities, places)
    assert loaded.rule_set_id == runtime.rule_set_id
    for a, b in zip(loaded.event_types, runtime.event_types, strict=True):
        assert a.effects == b.effects
    entry = next(e for e in record["event_types"] if e["effects"])
    literal = {"role": "agent", "kind": "feature", "symbol": "PROPERTY.1", "value": True}
    for change, message in (
        ({"symbol": "BOOLFL.1"}, "is a 'feature' literal"),
        ({"kind": "fluent"}, "is a 'fluent' literal"),
        ({"symbol": "NOTHING.1"}, "not a symbol"),
        ({"role": "patient"} if entry["arity"] == 1 else {"role": "nobody"}, "reads the role"),
    ):
        broken = json.loads(json.dumps(record))
        target = next(e for e in broken["event_types"] if e["label"] == entry["label"])
        target["effects"][0]["condition"] = {
            "literals": [{**literal, **change}],
            "expression": "x",
        }
        with pytest.raises(DefinitionError, match=message):
            runtime_definition(rehash(broken), entities, places)
    broken = json.loads(json.dumps(record))
    target = next(e for e in broken["event_types"] if e["label"] == entry["label"])
    target["effects"][0]["condition"] = {"literals": [literal, literal], "expression": "x"}
    with pytest.raises(DefinitionError, match="twice"):
        runtime_definition(rehash(broken), entities, places)


def test_check_dynamics_enforces_the_condition_rules(conditional: WorldResult) -> None:
    import dataclasses

    fluents = conditional.fluents
    present = {
        (f.label, bool(v))
        for i, f in enumerate(fluents.base)
        for v in np.unique(fluents.initial_values[:, i])
    }
    numeric = conditional.numeric
    for label, column in derived_initial_values(fluents, conditional.taxonomy, numeric).items():
        present |= {(label, bool(v)) for v in np.unique(column)}
    features = tuple(f.label for f in conditional.taxonomy.features.features)
    check_dynamics(conditional.event_types, fluents, present, features, numeric)
    event_types = conditional.event_types
    et = next(e for e in event_types.leaves if e.effects)
    effect = et.effects[0]

    def with_effect(new: Effect):
        changed = dataclasses.replace(et, effects=(new,) + et.effects[1:])
        return dataclasses.replace(
            event_types,
            event_types=tuple(
                changed if e.label == et.label else e for e in event_types.event_types
            ),
        )

    def literal(symbol: str, value: bool = True, role: str = "agent") -> ConditionLiteral:
        return ConditionLiteral(role, symbol, value)

    effect = next(e for e in et.effects if not e.numeric)
    own = literal(effect.fluent, not effect.value, effect.role)
    with pytest.raises(ValueError, match="reads the fluent the effect sets"):
        check_dynamics(
            with_effect(dataclasses.replace(effect, condition=(own,))),
            fluents,
            present,
            numeric=numeric,
        )
    twice = (literal("PROPERTY.1"), literal("PROPERTY.1", False))
    with pytest.raises(ValueError, match="twice"):
        check_dynamics(
            with_effect(dataclasses.replace(effect, condition=twice)),
            fluents,
            present,
            numeric=numeric,
        )
    with pytest.raises(ValueError, match="unknown feature"):
        check_dynamics(
            with_effect(dataclasses.replace(effect, condition=(literal("PROPERTY.99"),))),
            fluents,
            present,
            features,
            numeric,
        )
    if et.precondition:
        pre = et.precondition[0]
        fixed = ConditionLiteral(pre.role, pre.fluent, pre.value)
        if (pre.role, pre.fluent) != (effect.role, effect.fluent):
            with pytest.raises(ValueError, match="precondition fixes"):
                check_dynamics(
                    with_effect(dataclasses.replace(effect, condition=(fixed,))), fluents, present
                )
    with pytest.raises(ValueError, match="uses a role it has not"):
        check_dynamics(
            with_effect(
                dataclasses.replace(effect, condition=(literal("PROPERTY.1", role="nobody"),))
            ),
            fluents,
            present,
        )


# ---------------------------------------------------------------------------------------------
# Acceptance: every conditional effect fires exactly when its condition held
# ---------------------------------------------------------------------------------------------


def test_conditional_effects_fire_exactly_when_their_condition_held(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261009)
    fired = skipped = 0
    for _ in range(RANDOM_STATES):
        state = _random_state(runtime, rng)
        brute_state = _brute_state(runtime, state)
        numeric = _brute_numeric(runtime, state)
        for event in _legal_events(runtime, state):
            et = runtime.event_type(event.event_type)
            binding = _labels(runtime, event.event_type, event.binding)
            after = apply(runtime, state, [event])
            # the brute-force evaluator's independent reading of the step
            assert after.true_fluents(runtime) == brute.state_record(
                brute.apply(brute_state, [BruteEvent(event.event_type, binding)], numeric)
            )
            expected = []
            for effect in et.effects:
                record = next(
                    e
                    for e in brute.event_types[et.label]["effects"]
                    if e["fluent"] == effect.fluent and e["role"] == effect.role
                )
                holds = brute.condition_holds(record, binding, brute_state, numeric)
                entity = event.binding[et.roles.index(effect.role)]
                if effect.numeric:
                    # a numeric effect's firing is checked value by value in test_world_numeric
                    if holds:
                        expected.append(effect)
                    continue
                column = runtime.base_fluent_index(effect.fluent)
                if holds:
                    expected.append(effect)
                    fired += 1
                    assert after.values[entity, column] == int(effect.value)
                else:
                    skipped += 1
                    assert after.values[entity, column] == state.values[entity, column]
            assert fired_effects(runtime, state, event) == tuple(expected)
    assert fired > 0 and skipped > 0, (fired, skipped)


# ---------------------------------------------------------------------------------------------
# Acceptance: order independence and interference through conditions
# ---------------------------------------------------------------------------------------------


def test_applying_together_equals_one_at_a_time_with_conditional_effects(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261010)
    sets = with_conditions = 0
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
        if any(
            e.condition for event in chosen for e in runtime.event_type(event.event_type).effects
        ):
            with_conditions += 1
        together = apply(runtime, state, chosen)
        for permutation in itertools.permutations(chosen):
            step = state
            for event in permutation:
                # the effects alone: the drift is added once per step, after them
                step = apply(runtime, step, [event], drift=False)
            assert step == apply(runtime, state, chosen, drift=False)
            assert np.array_equal(step.values, together.values)
        sets += 1
    assert sets == RANDOM_SETS and with_conditions > RANDOM_SETS // 2


def test_events_whose_conditions_read_what_another_writes_interfere(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(11)
    found = 0
    for _ in range(300):
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        for a, b in itertools.combinations(events, 2):
            et_a, et_b = runtime.event_type(a.event_type), runtime.event_type(b.event_type)
            # a condition read of one that the other writes, apart from the precondition reads
            condition_reads = set()
            for event, et in ((a, et_a), (b, et_b)):
                for effect in et.effects:
                    for lit in effect.condition:
                        if lit.kind == "fluent":
                            entity = event.binding[et.roles.index(lit.role)]
                            condition_reads |= {
                                (event, entity, base) for base in runtime.cone(lit.symbol)
                            }
            writes = {
                (event, event.binding[et.roles.index(e.role)], e.fluent)
                for event, et in ((a, et_a), (b, et_b))
                for e in et.effects
            }
            crossing = any(
                (reader, entity, fluent) in condition_reads and (writer, entity, fluent) in writes
                for reader, writer in ((a, b), (b, a))
                for (_, entity, fluent) in condition_reads
            )
            interfere = brute.interfere(_brute_event(runtime, a), _brute_event(runtime, b))
            if crossing:
                assert interfere
                found += 1
                with pytest.raises(InterferenceError):
                    apply(runtime, state, [a, b])
            elif not interfere:
                apply(runtime, state, [a, b])
    assert found > 0


def test_the_hand_world_interference_through_a_condition_names_the_condition() -> None:
    fixture = read_fixture(
        Path("tests/fixtures/world/hand_17_condition_read_by_another_event.json")
    )
    definition = runtime_definition(
        fixture["definition"], fixture["entities"], fixture.get("places")
    )
    state = State.from_true(
        definition,
        fixture["initial"],
        fixture.get("initial_numeric"),
        fixture.get("initial_pairs"),
        fixture.get("initial_places"),
    )
    reader = Event.from_labels(definition, "EVENTTYPE1.5", {"agent": "INSTANCE.1.1.1"})
    writer = Event.from_labels(definition, "EVENTTYPE1.3", {"agent": "INSTANCE.1.1.1"})
    for order in ([reader, writer], [writer, reader]):
        with pytest.raises(InterferenceError) as info:
            apply(definition, state, order)
        assert "which a condition of EVENTTYPE1.5(agent=INSTANCE.1.1.1) reads" in str(info.value)
    # the static condition of EVENTTYPE1.4 reads nothing an event writes
    static = Event.from_labels(definition, "EVENTTYPE1.4", {"agent": "INSTANCE.1.1.1"})
    after = apply(definition, state, [static, writer])
    assert after.true_fluents(definition)["INSTANCE.1.1.1"] == ["BOOLFL.2"]


# ---------------------------------------------------------------------------------------------
# Acceptance: satisfiability, no contradictions, no empty effects
# ---------------------------------------------------------------------------------------------


def _check_satisfiable(result: WorldResult) -> int:
    """Every condition satisfiable by the runtime's able table and static facts, independently
    of the generator's check; returns the number of conditional effects."""
    definition = result.definition.runtime()
    able = able_table(definition)
    statics = static_facts(definition)
    produced = {(e.fluent, e.value) for et in result.event_types.leaves for e in et.effects}
    fluents = result.fluents
    present = {
        (f.label, bool(v))
        for i, f in enumerate(fluents.base)
        for v in np.unique(fluents.initial_values[:, i])
    }
    for label, column in derived_initial_values(
        fluents, result.taxonomy, result.numeric, result.pair_fluents
    ).items():
        present |= {(label, bool(v)) for v in np.unique(column)}
    from semantic_world.world.event_types import produced_numeric

    signs = produced_numeric(result.event_types)
    count = 0
    for et in result.event_types.leaves:
        keys = [(e.role, e.fluent) for e in et.effects]
        assert len(keys) == len(set(keys)), f"{et.label}: two effects write one fluent"
        pre = {(lit.role, lit.fluent): lit.value for lit in et.precondition}
        for effect in et.effects:
            if not effect.numeric:
                assert pre.get((effect.role, effect.fluent)) != effect.value, "an empty effect"
            if not effect.condition:
                continue
            count += 1
            assert len({(lit.role, lit.symbol) for lit in effect.condition}) == len(
                effect.condition
            )
            masks = {role: np.ones(definition.entity_count, dtype=bool) for role in et.roles}
            for lit in effect.condition:
                assert lit.role in et.roles
                if lit.kind == "numeric":
                    # stage b2: achievable by an initial value, the drift, or an effect
                    assert lit.threshold is not None
                    assert result.numeric.achievable(lit.symbol, lit.value, lit.threshold, signs)
                    assert (lit.role, lit.symbol) not in pre
                elif lit.kind == "fluent":
                    assert (lit.symbol, lit.value) in produced | present
                    assert (lit.role, lit.symbol) not in pre
                    assert (lit.role, lit.symbol) != (effect.role, effect.fluent)
                elif isinstance(lit, PairLiteral):
                    # stage c2b: a pair condition literal is satisfiable when its pair fluent
                    # can hold; tests/world/test_world_pair_generation.py checks the pair
                    # conditions, and the default world has them since stage c4a
                    assert lit.pair_fluent in result.pair_fluents.labels
                else:
                    masks[lit.role] &= statics.values[lit.symbol] == int(lit.value)
            if not able[et.label].any():
                continue  # a never-able event type has no event: its condition is vacuous
            if et.arity == 1:
                assert (able[et.label] & masks["agent"]).any(), (et.label, effect.text)
            else:
                assert able[et.label][np.ix_(masks["agent"], masks["patient"])].any(), (
                    et.label,
                    effect.text,
                )
    return count


@pytest.mark.slow
def test_every_condition_is_satisfiable_and_no_effect_is_contradictory_or_empty(
    conditional: WorldResult, default: WorldResult
) -> None:
    assert _check_satisfiable(conditional) > 0
    assert _check_satisfiable(default) > 0
    for seed in (2, 3):
        data = mapping_before_phase_c()
        data["event_types"]["effects"] = {"conditional_share": 1.0}
        assert _check_satisfiable(define(config_from_mapping(data, seed=seed))) > 0


@pytest.mark.slow
def test_an_inherited_effect_brings_its_condition_to_every_event_type(default: WorldResult) -> None:
    event_types = default.event_types
    conditional_features = [
        f for f, e in event_types.feature_effects.items() if e is not None and e.condition
    ]
    assert conditional_features, "the default world has a conditional feature effect"
    for feature in conditional_features:
        effect = event_types.feature_effects[feature]
        holders = [et for et in event_types.event_types if feature in et.features]
        assert holders
        for et in holders:
            same = [e for e in et.effects if (e.role, e.fluent) == (effect.role, effect.fluent)]
            if same:
                assert same[0] == effect, (et.label, same[0].text, effect.text)
    stats = world_stats(default.fluents, event_types)
    assert stats["effects"]["conditional"]["total"] == sum(
        1 for et in event_types.leaves for e in et.effects if e.condition
    )
    assert stats["effects"]["condition_literals"]["from_features"] == sum(
        len(e.condition) for e in event_types.feature_effects.values() if e is not None
    )
    assert set(stats["fix_ups"]) >= {"conditions_redrawn", "conditions_dropped"}


def test_the_conditions_have_a_stream_of_their_own() -> None:
    """Changing the condition settings changes no other draw: the effects without their
    conditions, the preconditions, and the static side are those of the a8 world."""
    plain = define(tiny_config(0.0))
    conditional = define(tiny_config(1.0))
    assert plain.rule_set_id != conditional.rule_set_id
    for a, b in zip(
        plain.event_types.event_types, conditional.event_types.event_types, strict=True
    ):
        assert a.precondition == b.precondition
        assert tuple(e.unconditional for e in a.effects) == tuple(
            e.unconditional for e in b.effects
        )
        assert a.constraints == b.constraints
    assert plain.definition.rules_record() == conditional.definition.rules_record()
    assert np.array_equal(plain.fluents.initial_values, conditional.fluents.initial_values)


# ---------------------------------------------------------------------------------------------
# Histories and the event file
# ---------------------------------------------------------------------------------------------


def test_replay_reproduces_every_recorded_change_under_conditions(
    conditional: WorldResult, runtime: RuntimeDefinition
) -> None:
    generator = EpisodeGenerator(runtime, Relatedness.from_statics(conditional.statics, runtime))
    histories = generator.run(5, 200)
    not_fired = fired = 0
    for history in histories:
        states = replay(runtime, history)
        for step, before in zip(history.steps, states, strict=False):
            for event in step.events:
                runtime_event = Event.from_labels(runtime, event.type, event.binding)
                et = runtime.event_type(event.type)
                live = fired_effects(runtime, before, runtime_event)
                for effect in et.effects:
                    if effect.numeric:
                        continue  # test_world_numeric checks the numeric changes
                    entity = runtime_event.binding[et.roles.index(effect.role)]
                    label = runtime.entity_labels[entity]
                    recorded = any(
                        c.entity == label and c.fluent == effect.fluent for c in event.changes
                    )
                    had = before.values[entity, runtime.base_fluent_index(effect.fluent)]
                    if effect in live:
                        fired += 1
                        assert recorded == (int(had) != int(effect.value))
                    else:
                        not_fired += 1
                        assert not recorded
    assert fired > 0 and not_fired > 0


def test_an_event_file_gives_conditional_effects(tmp_path: Path) -> None:
    base = define(tiny_config(0.0))
    world = mapping_before_phase_c()
    world["event_types"]["effects"] = {"conditional_share": 0}
    entities = base.statics.values
    features = base.taxonomy.features
    # a pair of features that no able agent of EVENTTYPE1.1 has together, for the error case
    able = base.statics.values[:, base.statics.features["EVENTTYPE1.1"].position].astype(bool)
    labels = [f.label for f in features.features]
    unsatisfiable = next(
        (a, b)
        for a in labels
        for b in labels
        if a < b
        and not (
            able
            & (entities[:, features[a].position] == 1)
            & (entities[:, features[b].position] == 1)
        ).any()
    )

    def run(effects: list[str], precondition: str | None = None) -> WorldResult:
        entry: dict[str, Any] = {"effects": effects}
        if precondition is not None:
            entry["precondition"] = precondition
        path = tmp_path / "events.yaml"
        path.write_text(yaml.safe_dump({"event_types": {"EVENTTYPE1.1": entry}}))
        data = {**world, "event_types": {**world["event_types"], "event_file": str(path)}}
        return define(config_from_mapping(data, source=str(DATA / "tiny.yaml")))

    result = run(
        ["when agent.PROPERTY.1 AND NOT agent.BOOLFL.2 then agent.BOOLFL.1 := 1"],
        "NOT agent.BOOLFL.3",
    )
    et = result.event_types.event_type("EVENTTYPE1.1")
    assert et.explicit and et.effects == (
        Effect(
            "agent", "BOOLFL.1", True, parse_condition("agent.PROPERTY.1 AND NOT agent.BOOLFL.2")
        ),
    )
    record = result.definition.record()
    entry = next(e for e in record["event_types"] if e["label"] == "EVENTTYPE1.1")
    assert entry["effects"][0]["condition"]["literals"] == [
        {"role": "agent", "kind": "feature", "symbol": "PROPERTY.1", "value": True},
        {"role": "agent", "kind": "fluent", "symbol": "BOOLFL.2", "value": False},
    ]
    for effects, precondition, message in (
        (["when agent.BOOLFL.3 then agent.BOOLFL.1 := 1"], "agent.BOOLFL.3", "already fixes"),
        (["when NOT agent.BOOLFL.1 then agent.BOOLFL.1 := 1"], None, "the fluent the effect sets"),
        (["when agent.PROPERTY.99 then agent.BOOLFL.1 := 1"], None, "unknown feature"),
        (["when agent.BOOLFL.9 then agent.BOOLFL.1 := 1"], None, "unknown fluent"),
        (["when patient.PROPERTY.1 then agent.BOOLFL.1 := 1"], None, "names the role patient"),
        (
            ["when agent.PROPERTY.1 AND agent.PROPERTY.1 then agent.BOOLFL.1 := 1"],
            None,
            "twice",
        ),
        (
            [
                f"when agent.{unsatisfiable[0]} AND agent.{unsatisfiable[1]} then "
                f"agent.BOOLFL.1 := 1"
            ],
            None,
            "no able binding",
        ),
    ):
        with pytest.raises(ConfigError, match=message):
            run(effects, precondition)
    with pytest.raises(ValueError, match="is not an effect"):
        Effect.parse("when then agent.BOOLFL.1 := 1")
    with pytest.raises(ValueError, match="not a condition literal"):
        Effect.parse("when agent.SCALARDIM.1 then agent.BOOLFL.1 := 1")


def test_fixtures_hold_the_condition_record() -> None:
    fixture = read_fixture(Path("tests/fixtures/world/hand_13_condition_holds.json"))
    entry = next(e for e in fixture["definition"]["event_types"] if e["label"] == "EVENTTYPE1.4")
    assert entry["effects"][0] == {
        "role": "agent",
        "fluent": "BOOLFL.2",
        "value": True,
        "expression": "when agent.PROPERTY.1 then agent.BOOLFL.2 := 1",
        "condition": {
            "literals": [
                {"role": "agent", "kind": "feature", "symbol": "PROPERTY.1", "value": True}
            ],
            "expression": "agent.PROPERTY.1",
        },
    }
    assert fixture["version"] == 1
    tiny = read_fixture(Path("tests/fixtures/world/tiny_seed1.json"))
    assert any("condition" in e for et in tiny["definition"]["event_types"] for e in et["effects"])


# ---------------------------------------------------------------------------------------------
# The follow-ups of the stage (Jon's rulings 1 and 2, built in the process-cleanup session)
# ---------------------------------------------------------------------------------------------


def test_condition_kinds_weigh_the_kinds() -> None:
    """``event_types.effects.condition_kinds`` weighs a static feature, a fluent, and (stage b2)
    a numeric threshold in the draw of a condition literal: all the weight on one kind gives
    literals of that kind alone, the default split in any spelling is the default draw, which
    gives the tiny world, and the even split between features and fluents in any spelling is
    the stage b1 draw."""
    features = define(tiny_config(1.0, condition_kinds={"feature": 1, "fluent": 0}))
    kinds = {
        lit.kind for et in features.event_types.leaves for e in et.effects for lit in e.condition
    }
    assert kinds == {"feature"}
    fluents = define(tiny_config(1.0, condition_kinds={"feature": 0, "fluent": 1}))
    kinds = {
        lit.kind for et in fluents.event_types.leaves for e in et.effects for lit in e.condition
    }
    assert kinds == {"fluent"}
    assert features.rule_set_id != fluents.rule_set_id
    numeric = define(tiny_config(1.0, condition_kinds={"numeric": 1}))
    kinds = {
        lit.kind for et in numeric.event_types.leaves for e in et.effects for lit in e.condition
    }
    assert kinds == {"numeric"}
    spelled = define(tiny_config(0.2, condition_kinds={"feature": 4, "fluent": 4, "numeric": 2}))
    assert spelled.rule_set_id == define(tiny_config(0.2)).rule_set_id
    assert spelled.config.resolved()["event_types"]["effects"]["condition_kinds"] == {
        "feature": 4.0,
        "fluent": 4.0,
        "numeric": 2.0,
    }
    even = define(tiny_config(0.2, condition_kinds={"feature": 2, "fluent": 2}))
    assert (
        even.rule_set_id
        == define(
            tiny_config(0.2, condition_kinds={"feature": 0.5, "fluent": 0.5, "numeric": 0})
        ).rule_set_id
    )
    assert even.rule_set_id != spelled.rule_set_id
    defaults = config_from_mapping({}).event_types.effects
    assert defaults.condition_kinds == {
        "feature": 0.4,
        "fluent": 0.4,
        "numeric": 0.2,
        "pair": 0.2,  # stage c2b: used only when the world has pair fluents
    }
    with pytest.raises(ConfigError, match="condition_kinds"):
        tiny_config(0.5, condition_kinds={"feature": 0, "fluent": 0})
    with pytest.raises(ConfigError, match="condition_kinds"):
        tiny_config(0.5, condition_kinds={"scalar": 1})


def test_the_statistics_report_both_shares(conditional: WorldResult) -> None:
    """``world_stats.yaml`` reports the share of the drawn effects that got a condition (the
    effects of event-type features once each, and the own effects of the event types) and the
    share of all effects of all event types that have one."""
    for world in (conditional, define(tiny_config(0.2)), define(tiny_config(0.0))):
        stats = world.stats["effects"]["conditional"]
        event_types = world.event_types
        drawn = event_types.condition_draws
        feature_effects = [e for e in event_types.feature_effects.values() if e is not None]
        own = [
            e
            for et in event_types.leaves
            for e in et.effects
            if not any(e.unconditional == f.unconditional for f in feature_effects)
        ]
        assert drawn["drawn"] == len(feature_effects) + len(own)
        got = sum(1 for e in feature_effects if e.condition) + sum(1 for e in own if e.condition)
        assert drawn["conditional"] == got
        effects = [e for et in event_types.leaves for e in et.effects]
        assert stats["share_of_drawn"] == round(got / drawn["drawn"], 6)
        assert stats["share_of_all"] == round(
            sum(1 for e in effects if e.condition) / len(effects), 6
        )
        assert stats["total"] == sum(1 for e in effects if e.condition)
    assert conditional.stats["effects"]["conditional"]["share_of_all"] == 1.0
    assert define(tiny_config(0.0)).stats["effects"]["conditional"]["share_of_drawn"] == 0.0
