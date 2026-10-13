"""Stage b2: numeric fluents (``docs/specs/WORLD_AND_LANGUAGE.md``, "Numeric fluents"), and
the numeric condition literals deferred from stage b1.

The acceptance tests of the stage: on 1,000 random states of a tiny world with numeric fluents,
numeric effects, numeric precondition literals, numeric conditions, and derived fluents that read
numeric thresholds, the runtime and the brute-force evaluator agree on ``derive``, ``legal``, and
``apply``, including the order within a step; applying a step's events together equals applying
them one at a time in every order, on 1,000 random non-interfering sets; events that read a
numeric fluent another event writes are reported as interfering; every numeric precondition and
condition literal is achievable; replaying every history reproduces every recorded numeric value
exactly and every recorded Boolean change and nothing else; every numeric value stays inside its
range at every time point; with ``numeric_fluents.count: 0`` the world is the world of stage b1
(``test_world_off_settings.py`` holds the identity check), and with the numeric shares at 0 the
Boolean side is drawn as before; the loader and the event file validate the numeric records; and
the statistics count them.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.__main__ import main
from semantic_world.world.config import Config, config_from_mapping, load_config
from semantic_world.world.definition import (
    DefinitionError,
    RuntimeDefinition,
    runtime_definition,
)
from semantic_world.world.dynamics import ConditionLiteral, Effect, Literal, parse_condition
from semantic_world.world.episodes import EpisodeGenerator, EpisodeSettings, Relatedness
from semantic_world.world.errors import InterferenceError, WorldError
from semantic_world.world.event_file import event_file_from_mapping
from semantic_world.world.event_types import check_dynamics, produced_numeric
from semantic_world.world.fixtures import (
    BruteEvent,
    BruteForce,
    fixture_initial_numeric,
    fixture_inputs,
)
from semantic_world.world.fluents import derived_initial_values
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.history import DRIFT, Change, History, replay
from semantic_world.world.numeric import (
    InitialDistribution,
    NumericFluents,
    clamp,
    step_value,
)
from semantic_world.world.runtime import (
    Event,
    State,
    apply,
    apply_step,
    derive,
    initial_state,
    interferes,
    legal_bindings,
)

DATA = Path("data/world")
RANDOM_STATES = 1000
RANDOM_SETS = 1000
HEAVY: dict[str, Any] = {
    # a tiny world that uses every numeric feature: derived rules that read a numeric
    # threshold (a heavy weight in the input pool), numeric effects and literals at a high
    # share, and every effect conditional with numeric conditions in the mix
    "numeric_fluents": {"count": 2, "rule_weight": 50},
    "event_types": {
        "effects": {
            "numeric_share": 0.6,
            "conditional_share": 1.0,
            "condition_kinds": {"feature": 1, "fluent": 1, "numeric": 2},
        },
        "preconditions": {"numeric_share": 0.6},
    },
}


def tiny_mapping(**overrides: Any) -> dict:
    data = yaml.safe_load((DATA / "tiny.yaml").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            merged = dict(data[key])
            for inner, setting in value.items():
                if isinstance(setting, dict) and isinstance(merged.get(inner), dict):
                    merged[inner] = {**merged[inner], **setting}
                else:
                    merged[inner] = setting
            data[key] = merged
        else:
            data[key] = value
    return data


def tiny_config(**overrides: Any) -> Config:
    return config_from_mapping(tiny_mapping(**overrides), source=str(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def heavy() -> WorldResult:
    return define(tiny_config(**HEAVY))


@pytest.fixture(scope="module")
def runtime(heavy: WorldResult) -> RuntimeDefinition:
    return heavy.definition.runtime()


@pytest.fixture(scope="module")
def brute(heavy: WorldResult) -> BruteForce:
    record, entities, _ = fixture_inputs(heavy.definition)
    return BruteForce(record, entities)


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
    """A random state: Boolean fluents at 0 or 1, numeric fluents uniform in their ranges, and
    one value in four exactly at a quarter point of the range, so that thresholds are hit."""
    shape = (definition.entity_count, len(definition.base_fluents))
    values = rng.integers(0, 2, size=shape, dtype=np.uint8)
    low, high = definition.range_bounds
    numeric = rng.uniform(
        low, high, size=(definition.entity_count, len(definition.numeric_fluents))
    )
    at_point = rng.random(numeric.shape) < 0.25
    points = low + rng.choice([0.25, 0.5, 0.75], size=numeric.shape) * (high - low)
    return State(values, np.where(at_point, points, numeric))


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


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if not et.is_category]


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


def _independent(definition: RuntimeDefinition, events: list[Event], rng) -> list[Event]:
    chosen: list[Event] = []
    for i in rng.permutation(len(events)):
        event = events[int(i)]
        if all(not interferes(definition, event, other) for other in chosen):
            chosen.append(event)
        if len(chosen) == 3:
            break
    return chosen


# ---------------------------------------------------------------------------------------------
# The heavy world has every numeric feature
# ---------------------------------------------------------------------------------------------


def test_the_heavy_world_uses_every_numeric_feature(heavy: WorldResult) -> None:
    stats = heavy.stats["numeric_fluents"]
    assert stats["count"] == 2 and len(heavy.numeric) == 2
    assert stats["effects"]["total"] > 0
    assert stats["precondition_literals"]["total"] > 0
    assert stats["condition_literals"]["total"] > 0
    assert stats["rules_reading_a_threshold"] > 0
    assert (
        stats["effects"]["by_operator"]["add"] + stats["effects"]["by_operator"]["subtract"]
        == (stats["effects"]["total"])
    )
    leaves = heavy.event_types.leaves
    assert any(e.numeric for et in leaves for e in et.effects)
    assert any(lit.numeric for et in leaves for lit in et.precondition)
    assert any(lit.numeric for et in leaves for e in et.effects for lit in e.condition)
    assert any(rule.numeric_thresholds for rule in heavy.fluents.rules)


def test_the_settings_are_read_and_validated() -> None:
    config = tiny_config(
        numeric_fluents={
            "count": 3,
            "range": [0, 10],
            "initial": {"kind": "beta", "mean": 5, "concentration": 4},
            "drift": {"share": 1.0, "amounts": [0.5, 1.5]},
            "fluents": {
                "NUMFL.2": {"range": [-1, 1], "initial": {"kind": "fixed", "value": 0.5}},
                "NUMFL.3": {"drift": None},
            },
        }
    )
    settings = config.numeric_fluents
    assert settings.count == 3 and settings.range == (0.0, 10.0)
    assert settings.initial == {"kind": "beta", "mean": 5.0, "concentration": 4.0}
    assert settings.drift_share == 1.0 and settings.drift_amounts == (0.5, 1.5)
    assert settings.fluents["NUMFL.2"].range == (-1.0, 1.0)
    assert settings.fluents["NUMFL.3"].drift_set and settings.fluents["NUMFL.3"].drift is None
    world = define(config)
    fluents = {f.label: f for f in world.numeric}
    assert fluents["NUMFL.1"].range == (0.0, 10.0) and fluents["NUMFL.1"].drift >= 0.5
    assert fluents["NUMFL.2"].range == (-1.0, 1.0) and fluents["NUMFL.2"].initial.kind == "fixed"
    assert (world.numeric.initial_values[:, 1] == 0.5).all()
    assert fluents["NUMFL.3"].drift == 0.0
    assert (
        (world.numeric.initial_values[:, 0] >= 0) & (world.numeric.initial_values[:, 0] <= 10)
    ).all()
    resolved = config.resolved()["numeric_fluents"]
    assert resolved["fluents"]["NUMFL.3"] == {"drift": None}
    assert (
        config_from_mapping(yaml.safe_load(yaml.safe_dump(config.resolved()))).numeric_fluents
        == settings
    )
    for bad, message in (
        ({"count": -1}, "count"),
        ({"range": [1, 0]}, "range"),
        ({"initial": {"kind": "fixed", "value": 2}}, "inside the range"),
        ({"initial": {"kind": "beta", "mean": 0.0}}, "strictly inside"),
        ({"initial": {"kind": "gamma"}}, "kind"),
        ({"fluents": {"NUMFL.9": {}}}, "NUMFL.9"),
        ({"fluents": {"BOOLFL.1": {}}}, "BOOLFL.1"),
        ({"drift": {"share": 2}}, "share"),
    ):
        with pytest.raises(ConfigError, match=message):
            tiny_config(numeric_fluents=bad)
    for section, bad, message in (
        ("effects", {"numeric_share": 1.5}, "numeric_share"),
        ("effects", {"numeric_amounts": [-0.1, 0.5]}, "numeric_amounts"),
        ("preconditions", {"numeric_thresholds": [0, 0.5]}, "numeric_thresholds"),
        ("preconditions", {"numeric_thresholds": []}, "numeric_thresholds"),
        ("effects", {"condition_kinds": {"feature": 1, "scalar": 1}}, "condition_kinds"),
    ):
        with pytest.raises(ConfigError, match=message):
            tiny_config(event_types={section: bad})


def test_initial_distributions_draw_inside_the_range() -> None:
    rng = np.random.default_rng(3)
    for kind, extra in (
        ("uniform", {}),
        ("fixed", {"value": 0.3}),
        ("beta", {"mean": 0.7, "concentration": 3.0}),
    ):
        distribution = InitialDistribution.from_record({"kind": kind, **extra})
        drawn = (
            distribution.draw(rng, -1.0, 2.0, 500)
            if kind != "fixed"
            else distribution.draw(rng, 0, 1, 5)
        )
        assert ((drawn >= -1.0) & (drawn <= 2.0)).all()
        if kind == "fixed":
            assert (drawn == 0.3).all()
        if kind == "beta":
            assert abs(drawn.mean() - 0.7) < 0.15
        assert InitialDistribution.from_record(distribution.record()) == distribution
    with pytest.raises(ValueError):
        InitialDistribution.from_record({"kind": "gamma"})


# ---------------------------------------------------------------------------------------------
# The world's promise: streams of their own
# ---------------------------------------------------------------------------------------------


def test_with_the_shares_at_zero_the_boolean_side_is_drawn_as_before(tiny: WorldResult) -> None:
    """With numeric fluents present but no numeric effect, literal, condition, or rule input,
    every Boolean draw is the draw of the world without numeric fluents: the rules, the
    effects, the preconditions, and the conditions are the same, and only the numeric symbols
    and the entities' numeric columns differ."""
    off = define(tiny_config(numeric_fluents={"count": 0}))
    shares = define(
        tiny_config(
            numeric_fluents={"rule_weight": 0},
            event_types={
                "effects": {"numeric_share": 0, "condition_kinds": {"feature": 1, "fluent": 1}},
                "preconditions": {"numeric_share": 0},
            },
        )
    )
    assert len(shares.numeric) == 2 and len(off.numeric) == 0
    assert off.rule_set_id != shares.rule_set_id
    assert [r.record() for r in off.fluents.rules] == [r.record() for r in shares.fluents.rules]
    for a, b in zip(off.event_types.event_types, shares.event_types.event_types, strict=True):
        assert a.record() == b.record(), a.label
    assert off.definition.literals_record() == shares.definition.literals_record()
    assert off.definition.rules_record() == shares.definition.rules_record()
    assert off.definition.event_types_record() == shares.definition.event_types_record()
    booleans = [s for s in shares.definition.symbols if s["kind"] != "numeric_fluent"]
    assert booleans == list(off.definition.symbols)
    # the numeric fluents themselves are drawn from their own streams: the same as in the
    # default tiny world, whose shares are above 0
    assert [f.symbol_record() for f in shares.numeric] == [f.symbol_record() for f in tiny.numeric]
    assert np.array_equal(shares.numeric.initial_values, tiny.numeric.initial_values)


def test_numeric_settings_never_change_the_static_side_or_the_boolean_fluents(
    tiny: WorldResult, heavy: WorldResult
) -> None:
    """The numeric draws come after every static draw and the Boolean fluents' rates and
    initial values, so the heavy settings leave those as they are."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_world_define", Path("tests/world/test_world_define.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    static_records = module.static_records

    assert static_records(tiny.statics) == static_records(heavy.statics)
    assert tiny.fluents.initial_rates.tolist() == heavy.fluents.initial_rates.tolist()
    assert np.array_equal(tiny.fluents.initial_values, heavy.fluents.initial_values)
    assert [f.symbol_record() for f in tiny.numeric] == [f.symbol_record() for f in heavy.numeric]
    assert np.array_equal(tiny.numeric.initial_values, heavy.numeric.initial_values)


# ---------------------------------------------------------------------------------------------
# Achievability
# ---------------------------------------------------------------------------------------------


def _achievable(world: WorldResult) -> int:
    """Every numeric literal, in a precondition or a condition, is achievable by the initial
    values, the drift, or some effect; returns how many were checked."""
    numeric: NumericFluents = world.numeric
    signs = produced_numeric(world.event_types)
    checked = 0
    for et in world.event_types.leaves:
        literals: list[Literal | ConditionLiteral] = [lit for lit in et.precondition if lit.numeric]
        literals += [lit for e in et.effects for lit in e.condition if lit.numeric]
        for lit in literals:
            fluent = lit.fluent if isinstance(lit, Literal) else lit.symbol
            assert lit.threshold is not None
            column = numeric.initial_values[:, numeric.index(fluent)]
            drift = numeric[fluent].drift
            if lit.value:
                ways = [(column > lit.threshold).any(), drift > 0, (fluent, 1) in signs]
            else:
                ways = [(column <= lit.threshold).any(), drift < 0, (fluent, -1) in signs]
            assert any(ways), (et.label, lit.text)
            assert numeric[fluent].low < lit.threshold < numeric[fluent].high
            checked += 1
    return checked


def test_every_numeric_literal_is_achievable(heavy: WorldResult) -> None:
    assert _achievable(heavy) > 0
    for seed in (2, 3):
        assert _achievable(define(config_from_mapping(tiny_mapping(**HEAVY), seed=seed))) > 0


def test_achievability_by_drift_or_effect_alone() -> None:
    """A literal that no initial value satisfies is achievable through the drift or an
    effect, and otherwise is refused."""
    fluents = NumericFluents.__new__(NumericFluents)
    from semantic_world.world.numeric import NumericFluent

    object.__setattr__(
        fluents,
        "fluents",
        (
            NumericFluent("NUMFL.1", 1, 0.0, 1.0, InitialDistribution("fixed", value=0.5), 0.0),
            NumericFluent("NUMFL.2", 2, 0.0, 1.0, InitialDistribution("fixed", value=0.5), 0.1),
        ),
    )
    object.__setattr__(fluents, "initial_values", np.full((3, 2), 0.5))
    fluents.__post_init__()
    assert fluents.achievable("NUMFL.1", True, 0.25, set())
    assert fluents.achievable("NUMFL.1", False, 0.75, set())
    assert not fluents.achievable("NUMFL.1", True, 0.75, set())
    assert fluents.achievable("NUMFL.1", True, 0.75, {("NUMFL.1", 1)})
    assert not fluents.achievable("NUMFL.1", False, 0.25, {("NUMFL.1", 1)})
    assert fluents.achievable("NUMFL.1", False, 0.25, {("NUMFL.1", -1)})
    assert fluents.achievable("NUMFL.2", True, 0.75, set())  # the drift rises
    assert not fluents.achievable("NUMFL.2", False, 0.25, set())


def test_check_dynamics_enforces_the_numeric_rules(heavy: WorldResult) -> None:
    import dataclasses

    fluents = heavy.fluents
    present = {
        (f.label, bool(v))
        for i, f in enumerate(fluents.base)
        for v in np.unique(fluents.initial_values[:, i])
    }
    for label, column in derived_initial_values(fluents, heavy.taxonomy, heavy.numeric).items():
        present |= {(label, bool(v)) for v in np.unique(column)}
    event_types = heavy.event_types
    check_dynamics(event_types, fluents, present, None, heavy.numeric)
    with pytest.raises(ValueError, match="unknown numeric fluent|unknown or not achievable"):
        check_dynamics(event_types, fluents, present, None, None)
    et = next(e for e in event_types.leaves if e.effects)

    def with_event_type(changed):
        return dataclasses.replace(
            event_types,
            event_types=tuple(
                changed if e.label == et.label else e for e in event_types.event_types
            ),
        )

    unreachable = Literal("agent", "NUMFL.1", True, heavy.numeric["NUMFL.1"].high + 1.0)
    with pytest.raises(ValueError, match="not achievable"):
        check_dynamics(
            with_event_type(dataclasses.replace(et, precondition=(unreachable,), effects=())),
            fluents,
            present,
            None,
            heavy.numeric,
        )
    fixed = Literal("agent", "NUMFL.1", True, heavy.numeric["NUMFL.1"].point(0.25))
    reads = ConditionLiteral("agent", "NUMFL.1", False, heavy.numeric["NUMFL.1"].point(0.5))
    effect = Effect("agent", "BOOLFL.1", True, (reads,))
    with pytest.raises(ValueError, match="which the precondition fixes"):
        check_dynamics(
            with_event_type(dataclasses.replace(et, precondition=(fixed,), effects=(effect,))),
            fluents,
            present,
            None,
            heavy.numeric,
        )
    zero = Effect("agent", "NUMFL.1", 0.0)
    with pytest.raises(ValueError, match="adds nothing"):
        check_dynamics(
            with_event_type(dataclasses.replace(et, precondition=(), effects=(zero,))),
            fluents,
            present,
            None,
            heavy.numeric,
        )
    # a numeric condition on the effect's own numeric fluent is a real condition, and allowed
    own = Effect("agent", "NUMFL.1", 0.3, (reads,))
    check_dynamics(
        with_event_type(dataclasses.replace(et, precondition=(), effects=(own,))),
        fluents,
        present,
        None,
        heavy.numeric,
    )


# ---------------------------------------------------------------------------------------------
# The runtime against the brute-force evaluator
# ---------------------------------------------------------------------------------------------


def test_runtime_and_brute_force_agree_on_random_states(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    """On 1,000 random states: ``derive`` (derived fluents that read numeric thresholds among
    them), ``legal`` (numeric literals among the preconditions), and ``apply`` of a random
    non-interfering set (numeric effects and conditions, then the drift, in that order) give
    the same Boolean state and exactly the same numeric values."""
    rng = np.random.default_rng(5)
    with_numeric_effect = steps_with_change = 0
    for _ in range(RANDOM_STATES):
        state = _random_state(runtime, rng)
        brute_state, numeric = _brute_state(runtime, state), _brute_numeric(runtime, state)
        assert derive(runtime, state).true_fluents(runtime) == brute.derived_record(
            brute_state, numeric
        )
        events = []
        for label in _performable(runtime):
            got = [_labels(runtime, label, b) for b in legal_bindings(runtime, state, label)]
            assert got == brute.legal_bindings(label, brute_state, None, numeric), label
            events += [Event(label, b) for b in legal_bindings(runtime, state, label)]
        chosen = _independent(runtime, events, rng)
        result = apply_step(runtime, state, chosen)
        expected_state, expected_numeric, expected_after = brute.step(
            brute_state, numeric, [_brute_event(runtime, e) for e in chosen]
        )
        assert result.state.true_fluents(runtime) == brute.state_record(expected_state)
        assert result.state.numeric_values(runtime) == brute.numeric_record(expected_numeric)
        after = State(result.state.values, result.after_effects).numeric_values(runtime)
        assert after == brute.numeric_record(expected_after)
        if any(e.numeric for event in chosen for e in runtime.event_type(event.event_type).effects):
            with_numeric_effect += 1
        if result.state != state:
            steps_with_change += 1
        low, high = runtime.range_bounds
        assert ((result.state.numeric >= low) & (result.state.numeric <= high)).all()
    assert with_numeric_effect > RANDOM_STATES // 4 and steps_with_change > RANDOM_STATES // 2


def test_the_order_within_a_step_is_effect_then_drift(runtime: RuntimeDefinition) -> None:
    """Each numeric value after a step is ``clamp(clamp(v + e) + d)``, checked against the
    reference arithmetic value by value, on states at the ends of the ranges too."""
    rng = np.random.default_rng(8)
    low, high = runtime.range_bounds
    clamped = 0
    for _ in range(200):
        state = _random_state(runtime, rng)
        edges = rng.random(state.numeric.shape) < 0.3
        state = State(
            state.values,
            np.where(
                edges, np.where(rng.random(state.numeric.shape) < 0.5, low, high), state.numeric
            ),
        )
        chosen = _independent(runtime, _legal_events(runtime, state), rng)
        result = apply_step(runtime, state, chosen)
        effects: dict[tuple[int, int], float] = {}
        for event in chosen:
            et = runtime.event_type(event.event_type)
            from semantic_world.world.runtime import fired_effects

            for effect in fired_effects(runtime, state, event):
                if effect.numeric:
                    entity = event.binding[et.roles.index(effect.role)]
                    effects[(entity, runtime.numeric_index(effect.fluent))] = effect.amount
        for i in range(runtime.entity_count):
            for j, fluent in enumerate(runtime.numeric_fluents):
                v = float(state.numeric[i, j])
                e = effects.get((i, j))
                expected = step_value(v, e, runtime.numeric_drift[fluent], low[j], high[j])
                assert result.state.numeric[i, j] == expected
                assert result.after_effects[i, j] == (
                    clamp(v + e, low[j], high[j]) if e is not None else v
                )
                if e is not None and (v + e < low[j] or v + e > high[j]):
                    clamped += 1
    assert clamped > 0


def test_applying_together_equals_one_at_a_time_in_every_order(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    """On 1,000 random non-interfering sets with numeric effects, conditions, and drift: the
    Boolean state and the numeric values after the effects are the same when the events are
    applied one at a time in any order (the effects alone, ``drift=False``), and the step's
    result is those values after one drift."""
    rng = np.random.default_rng(13)
    low, high = runtime.range_bounds
    sets = with_numeric = with_numeric_condition = 0
    while sets < RANDOM_SETS:
        state = _random_state(runtime, rng)
        chosen = _independent(runtime, _legal_events(runtime, state), rng)
        if len(chosen) < 2:
            continue
        for a, b in itertools.combinations(chosen, 2):
            assert not brute.interfere(_brute_event(runtime, a), _brute_event(runtime, b))
        together = apply_step(runtime, state, chosen)
        for order in itertools.permutations(chosen):
            step = state
            for event in order:
                step = apply(runtime, step, [event], drift=False)
            assert np.array_equal(step.values, together.state.values)
            assert np.array_equal(step.numeric, together.after_effects)
        drifted = np.minimum(np.maximum(together.after_effects + runtime.drift_vector, low), high)
        assert np.array_equal(drifted, together.state.numeric)
        kinds = [runtime.event_type(e.event_type) for e in chosen]
        if any(e.numeric for et in kinds for e in et.effects):
            with_numeric += 1
        if any(lit.numeric for et in kinds for e in et.effects for lit in e.condition):
            with_numeric_condition += 1
        sets += 1
    assert with_numeric > RANDOM_SETS // 4 and with_numeric_condition > RANDOM_SETS // 10


def test_events_that_read_a_numeric_fluent_another_writes_interfere(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    """The runtime and the brute-force evaluator agree on interference over random pairs of
    legal events, and a numeric read (a precondition literal, a condition literal, or a derived
    fluent's rule through its cone) of a fluent another event writes is reported."""
    rng = np.random.default_rng(21)
    numeric_reads = same_numeric = 0
    for _ in range(150):
        state = _random_state(runtime, rng)
        events = _legal_events(runtime, state)
        for a, b in itertools.combinations(events, 2):
            expected = brute.interfere(_brute_event(runtime, a), _brute_event(runtime, b))
            assert interferes(runtime, a, b) == expected
            if not expected:
                continue
            with pytest.raises(InterferenceError) as info:
                apply(runtime, state, [a, b])
            message = str(info.value)
            et_a, et_b = runtime.event_type(a.event_type), runtime.event_type(b.event_type)
            writes_a = {(a.binding[et_a.roles.index(e.role)], e.fluent) for e in et_a.effects}
            writes_b = {(b.binding[et_b.roles.index(e.role)], e.fluent) for e in et_b.effects}
            shared = {w for w in writes_a & writes_b if runtime.is_numeric_fluent(w[1])}
            if shared:
                same_numeric += 1
                assert "both write" in message
            elif any(runtime.is_numeric_fluent(f) for _, f in writes_a | writes_b):
                numeric_reads += 1
                assert " reads" in message
    assert numeric_reads > 0 and same_numeric > 0


def test_a_numeric_read_through_a_derived_fluent_interferes(runtime: RuntimeDefinition) -> None:
    """A derived fluent that reads a numeric threshold has the numeric fluent in its cone, so a
    precondition on it reads the numeric fluent, which a numeric effect writes."""
    numeric_cones = {
        f: runtime.cone(f)
        for f in runtime.derived_fluents
        if any(runtime.is_numeric_fluent(b) for b in runtime.cone(f))
    }
    assert numeric_cones, "the heavy world has a derived fluent that reads a numeric threshold"
    for cone in numeric_cones.values():
        assert any(b.startswith("NUMFL.") for b in cone) and all(
            runtime.is_base_fluent(b) or runtime.is_numeric_fluent(b) for b in cone
        )


# ---------------------------------------------------------------------------------------------
# Histories and replay
# ---------------------------------------------------------------------------------------------


def test_replay_reproduces_every_recorded_numeric_value_exactly(
    heavy: WorldResult, runtime: RuntimeDefinition
) -> None:
    """Every history replays: the Boolean changes are exactly what the effects changed, every
    recorded numeric change (the effects' and the drift's) is reproduced to the bit, every
    numeric change of a step is recorded with its cause, and every value stays inside its
    range at every time point."""
    generator = EpisodeGenerator(runtime, Relatedness.from_statics(heavy.statics, runtime))
    histories = generator.run(5, 200)
    low, high = runtime.range_bounds
    effect_changes = drift_changes = clamped = 0
    for history in histories:
        states = replay(runtime, history)
        participants = [runtime.entity_index(p) for p in history.participants]
        for state in states:
            assert ((state.numeric >= low) & (state.numeric <= high)).all()
        for p in history.participants:
            row = runtime.entity_index(p)
            assert history.initial_numeric[p] == {
                f: float(states[0].numeric[row, j]) for j, f in enumerate(runtime.numeric_fluents)
            }
        for step, before, after in zip(history.steps, states, states[1:], strict=False):
            recorded = {}
            for event in step.events:
                for change in event.changes:
                    if change.numeric:
                        assert change.cause == event.label
                        effect_changes += 1
                        recorded[(change.entity, change.fluent)] = change
            for change in step.drift:
                assert change.cause == DRIFT
                drift_changes += 1
                start = (
                    recorded[(change.entity, change.fluent)].to
                    if (change.entity, change.fluent) in recorded
                    else before.numeric[
                        runtime.entity_index(change.entity), runtime.numeric_index(change.fluent)
                    ]
                )
                assert change.from_ == start
                assert (
                    change.to
                    == after.numeric[
                        runtime.entity_index(change.entity), runtime.numeric_index(change.fluent)
                    ]
                )
            # every numeric change of the step is recorded: a participant's value differs from
            # the step's start exactly when an effect or the drift recorded a change for it
            for row in participants:
                for j, fluent in enumerate(runtime.numeric_fluents):
                    label = runtime.entity_labels[row]
                    moved = before.numeric[row, j] != after.numeric[row, j]
                    noted = (label, fluent) in recorded or any(
                        c.entity == label and c.fluent == fluent for c in step.drift
                    )
                    assert moved == noted or (not moved and noted), (history.label, step.step)
            for change in [c for e in step.events for c in e.changes if c.numeric]:
                if change.to in (
                    low[runtime.numeric_index(change.fluent)],
                    high[runtime.numeric_index(change.fluent)],
                ):
                    clamped += 1
    assert effect_changes > 0 and drift_changes > 0 and clamped > 0


def test_replay_refuses_a_wrong_numeric_value(
    heavy: WorldResult, runtime: RuntimeDefinition
) -> None:
    from semantic_world.world.history import HistoryError

    generator = EpisodeGenerator(runtime, Relatedness.from_statics(heavy.statics, runtime))
    history = next(
        h
        for h in generator.run(5, 50)
        if any(c.numeric for e in h.events for c in e.changes) and any(s.drift for s in h.steps)
    )
    data = history.to_json()
    broken = History.from_json(json.loads(json.dumps(data)))
    assert broken == history
    # an effect's recorded value off by one unit in the last place
    for step in data["steps"]:
        for event in step["events"]:
            for change in event["changes"]:
                if "from" in change:
                    change["to"] = np.nextafter(change["to"], 2.0)
                    break
            else:
                continue
            break
        else:
            continue
        break
    with pytest.raises(HistoryError, match="numeric changes"):
        replay(runtime, History.from_json(data))
    data = history.to_json()
    first = next(s for s in data["steps"] if s.get("drift"))
    first["drift"] = first["drift"][1:]
    with pytest.raises(HistoryError, match="drift"):
        replay(runtime, History.from_json(data))


def test_histories_of_a_world_without_numeric_fluents_are_unchanged(tmp_path: Path) -> None:
    """No ``initial_numeric``, no ``drift``, and Boolean changes as before."""
    world = define(tiny_config(numeric_fluents={"count": 0}))
    runtime = world.definition.runtime()
    generator = EpisodeGenerator(runtime, Relatedness.from_statics(world.statics, runtime))
    for history in generator.run(1, 20):
        data = history.to_json()
        assert "initial_numeric" not in data
        for step in data["steps"]:
            assert "drift" not in step
            for event in step["events"]:
                for change in event["changes"]:
                    assert set(change) == {"entity", "fluent", "to"}
        replay(runtime, history)


def test_initial_redraw_draws_the_numeric_fluents_from_their_distributions() -> None:
    world = define(
        tiny_config(
            numeric_fluents={
                "count": 3,
                "fluents": {
                    "NUMFL.1": {"initial": {"kind": "fixed", "value": 0.25}},
                    "NUMFL.2": {"initial": {"kind": "beta", "mean": 0.8, "concentration": 2}},
                    "NUMFL.3": {"range": [2, 4]},
                },
            }
        )
    )
    runtime = world.definition.runtime()
    kept = EpisodeGenerator(runtime, Relatedness.from_statics(world.statics, runtime))
    redrawn = EpisodeGenerator(
        runtime,
        Relatedness.from_statics(world.statics, runtime),
        EpisodeSettings(initial="redraw"),
    )
    start = initial_state(runtime)
    seen_other = 0
    for a, b in zip(kept.run(3, 40), redrawn.run(3, 40), strict=True):
        for p in a.participants:
            row = runtime.entity_index(p)
            assert a.initial_numeric[p] == {
                f: float(start.numeric[row, j]) for j, f in enumerate(runtime.numeric_fluents)
            }
        for p in b.participants:
            values = b.initial_numeric[p]
            assert values["NUMFL.1"] == 0.25
            assert 0.0 <= values["NUMFL.2"] <= 1.0 and 2.0 <= values["NUMFL.3"] <= 4.0
            row = runtime.entity_index(p)
            if values["NUMFL.3"] != start.numeric[row, 2]:
                seen_other += 1
        replay(runtime, b)
    assert seen_other > 0


def test_a_history_round_trips_through_json(heavy: WorldResult, runtime: RuntimeDefinition) -> None:
    generator = EpisodeGenerator(runtime, Relatedness.from_statics(heavy.statics, runtime))
    for history in generator.run(2, 10):
        text = json.dumps(history.to_json())
        again = History.from_json(json.loads(text))
        assert again == history
        for change in history.numeric_changes:
            assert isinstance(change, Change) and change.numeric and change.cause


# ---------------------------------------------------------------------------------------------
# The definition, the loader, and the event file
# ---------------------------------------------------------------------------------------------


def test_the_records_of_numeric_effects_literals_and_conditions(heavy: WorldResult) -> None:
    record = heavy.definition.record()
    symbols = [s for s in record["symbols"] if s["kind"] == "numeric_fluent"]
    assert [s["label"] for s in symbols] == ["NUMFL.1", "NUMFL.2"]
    for symbol in symbols:
        assert set(symbol) == {
            "label",
            "kind",
            "derived",
            "fluent",
            "arity",
            "range",
            "initial",
            "drift",
        }
        assert symbol["fluent"] is True and symbol["derived"] is False
    literals = [lit for lit in record["literals"] if lit["kind"] == "numeric_threshold"]
    assert literals and all(lit["role"] is None and lit["operator"] == ">" for lit in literals)
    seen_effect = seen_literal = seen_condition = False
    for entry in record["event_types"]:
        for effect in entry["effects"]:
            if "amount" in effect:
                seen_effect = True
                assert "value" not in effect and effect["operator"] in ("+=", "-=")
                assert effect["amount"] > 0
                assert effect["expression"].endswith(f"{effect['operator']} {effect['amount']!r}")
            else:
                assert isinstance(effect["value"], bool)
            for lit in (effect.get("condition") or {}).get("literals", ()):
                if lit["kind"] == "numeric":
                    seen_condition = True
                    assert set(lit) == {"role", "kind", "symbol", "operator", "threshold"}
                    assert lit["operator"] in (">", "<=")
        for lit in entry["precondition"]["literals"]:
            if "threshold" in lit:
                seen_literal = True
                assert set(lit) == {"role", "fluent", "operator", "threshold"}
    assert seen_effect and seen_literal and seen_condition
    # the texts parse back
    for et in heavy.event_types.leaves:
        for effect in et.effects:
            assert Effect.parse(effect.text) == effect
            if effect.condition:
                assert (
                    parse_condition(effect.record()["condition"]["expression"]) == effect.condition
                )
        for lit in et.precondition:
            if lit.numeric:
                assert Literal.parse_numeric(lit.text) == lit


def test_the_loader_validates_the_numeric_records(heavy: WorldResult) -> None:
    record, entities, _ = fixture_inputs(heavy.definition)
    rows = json.loads(json.dumps(entities))
    numeric = fixture_initial_numeric(heavy.definition)
    assert numeric is not None
    full = [{**row, **numeric[row["label"]]} for row in rows]
    loaded = runtime_definition(record, full)
    assert loaded.numeric_fluents == ("NUMFL.1", "NUMFL.2")
    assert loaded.initial_numeric is not None and loaded.initial_numeric.shape == (12, 2)
    without = runtime_definition(record, rows)
    assert without.initial_numeric is None
    with pytest.raises(WorldError, match="no initial"):
        initial_state(without)

    def broken(change) -> dict:
        copy = json.loads(json.dumps(record))
        change(copy)
        from semantic_world.world.identity import rule_set_id

        copy["rule_set_id"] = rule_set_id(
            copy["symbols"], copy["literals"], copy["rules"], copy["event_types"]
        )
        return copy

    def numeric_symbol(copy):
        return next(s for s in copy["symbols"] if s["kind"] == "numeric_fluent")

    def numeric_effect(copy):
        return next(e for et in copy["event_types"] for e in et["effects"] if "amount" in e)

    def numeric_literal(copy):
        return next(
            lit
            for et in copy["event_types"]
            for lit in et["precondition"]["literals"]
            if "threshold" in lit
        )

    cases = [
        (lambda c: numeric_symbol(c).__setitem__("range", [1.0, 0.0]), "low < high"),
        (
            lambda c: numeric_symbol(c).__setitem__("initial", {"kind": "gamma"}),
            "initial distribution",
        ),
        (lambda c: numeric_symbol(c).__setitem__("drift", "fast"), "numeric drift"),
        (lambda c: numeric_effect(c).__setitem__("operator", "*="), "operator"),
        (lambda c: numeric_effect(c).__setitem__("amount", -0.5), "not above 0"),
        (lambda c: numeric_effect(c).__setitem__("fluent", "BOOLFL.1"), "not a numeric fluent"),
        (lambda c: numeric_literal(c).__setitem__("operator", "<"), "operator"),
        (lambda c: numeric_literal(c).__setitem__("fluent", "BOOLFL.1"), "not a numeric fluent"),
    ]
    for change, message in cases:
        with pytest.raises(DefinitionError, match=message):
            runtime_definition(broken(change), full)
    outside = json.loads(json.dumps(full))
    outside[0]["NUMFL.1"] = 7.0
    with pytest.raises(DefinitionError, match="outside its range"):
        runtime_definition(record, outside)
    partial = [{k: v for k, v in row.items() if k != "NUMFL.2"} for row in full]
    with pytest.raises(DefinitionError, match="some numeric fluent columns"):
        runtime_definition(record, partial)


def test_an_event_file_gives_numeric_dynamics(tmp_path: Path) -> None:
    base = define(tiny_config())
    numeric = base.numeric
    high_point = numeric["NUMFL.1"].point(0.75)
    world = tiny_mapping()
    world["event_types"]["event_file"] = "events.yaml"
    events = {
        "fluents": {
            "NUMFL.1": {"initial": {"kind": "fixed", "value": 0.5}, "drift": 0.1},
            "NUMFL.2": {"drift": None},
        },
        "event_types": {
            "EVENTTYPE1.1": {
                "precondition": f"agent.NUMFL.1 > {high_point} AND NOT agent.BOOLFL.1",
                "effects": [
                    "agent.NUMFL.2 += 0.3",
                    "when agent.NUMFL.2 <= 0.5 then agent.BOOLFL.1 := 1",
                ],
            },
            "EVENTTYPE1.2": {
                "precondition": "agent.NUMFL.2 <= 0.25",
                "effects": ["agent.NUMFL.1 -= 0.2"],
            },
        },
    }
    (tmp_path / "events.yaml").write_text(yaml.safe_dump(events), encoding="utf-8")
    (tmp_path / "world.yaml").write_text(yaml.safe_dump(world), encoding="utf-8")
    result = define(load_config(tmp_path / "world.yaml"))
    assert result.numeric["NUMFL.1"].drift == 0.1 and result.numeric["NUMFL.2"].drift == 0.0
    assert (result.numeric.initial_values[:, 0] == 0.5).all()
    first = result.event_types.event_type("EVENTTYPE1.1")
    assert first.explicit
    assert first.precondition == (
        Literal("agent", "NUMFL.1", True, high_point),
        Literal("agent", "BOOLFL.1", False),
    )
    assert first.effects[0] == Effect("agent", "NUMFL.2", 0.3)
    assert first.effects[1].condition == (ConditionLiteral("agent", "NUMFL.2", False, 0.5),)
    second = result.event_types.event_type("EVENTTYPE1.2")
    assert second.precondition == (Literal("agent", "NUMFL.2", False, 0.25),)
    assert second.effects == (Effect("agent", "NUMFL.1", -0.2),)
    record = result.definition.record()
    entry = next(e for e in record["event_types"] if e["label"] == "EVENTTYPE1.1")
    assert (
        entry["precondition"]["expression"]
        == f"agent.NUMFL.1 > {high_point!r} AND NOT agent.BOOLFL.1"
    )
    # the file's errors name the entry
    for bad, message in (
        ({"EVENTTYPE1.1": {"effects": ["agent.NUMFL.9 += 0.3"]}}, "unknown numeric fluent"),
        ({"EVENTTYPE1.1": {"effects": ["agent.NUMFL.1 += 0"]}}, "above 0"),
        ({"EVENTTYPE1.1": {"effects": ["agent.NUMFL.1 *= 0.3"]}}, "not an effect"),
        ({"EVENTTYPE1.1": {"precondition": "agent.NUMFL.1 > 5.0"}}, "inside its range"),
        (
            {"EVENTTYPE1.1": {"precondition": "agent.NUMFL.1 > 0.5 AND agent.NUMFL.1 <= 0.75"}},
            "twice",
        ),
        (
            {
                "EVENTTYPE1.1": {
                    "precondition": "agent.NUMFL.1 > 0.5",
                    "effects": ["when agent.NUMFL.1 <= 0.25 then agent.BOOLFL.1 := 1"],
                }
            },
            "already fixes",
        ),
        (
            {"EVENTTYPE1.1": {"effects": ["when patient.NUMFL.1 > 0.5 then agent.BOOLFL.1 := 1"]}},
            "roles",
        ),
    ):
        data = {"fluents": {}, "event_types": bad}
        with pytest.raises((ConfigError, ValueError), match=message):
            file = event_file_from_mapping(data, source="events.yaml")
            from semantic_world.world.event_types import generate_event_types
            from semantic_world.world.streams import WorldStreams

            config = tiny_config()
            streams = WorldStreams(config.seed)
            derived_initial = derived_initial_values(base.fluents, base.taxonomy, numeric)
            generate_event_types(
                base.statics, base.fluents, config, streams, file, derived_initial, numeric
            )
    with pytest.raises(ConfigError, match="NUMFL.9"):
        data = {"fluents": {"NUMFL.9": {"drift": 0.1}}, "event_types": {}}
        from semantic_world.world.numeric import generate_numeric_fluents
        from semantic_world.world.streams import WorldStreams

        generate_numeric_fluents(
            base.config, base.taxonomy, WorldStreams(1), event_file_from_mapping(data).numeric, "x"
        )


# ---------------------------------------------------------------------------------------------
# Statistics, the command line, and views
# ---------------------------------------------------------------------------------------------


def test_the_statistics_count_the_numeric_fluents(heavy: WorldResult, tiny: WorldResult) -> None:
    for world in (heavy, tiny):
        stats = world.stats["numeric_fluents"]
        leaves = world.event_types.leaves
        assert stats["count"] == len(world.numeric)
        assert stats["drifting"] == sum(1 for f in world.numeric if f.drifts)
        assert set(stats["fluents"]) == set(world.numeric.labels)
        for f in world.numeric:
            assert stats["fluents"][f.label] == {
                "range": [f.low, f.high],
                "initial": f.initial.record(),
                "drift": f.drift,
            }
        effects = [e for et in leaves for e in et.effects if e.numeric]
        assert stats["effects"]["total"] == len(effects)
        assert stats["effects"]["by_operator"]["add"] == sum(1 for e in effects if e.amount > 0)
        assert stats["effects"]["by_role"]["agent"] == sum(1 for e in effects if e.role == "agent")
        literals = [lit for et in leaves for lit in et.precondition if lit.numeric]
        assert stats["precondition_literals"]["total"] == len(literals)
        assert stats["precondition_literals"]["by_side"]["above"] == sum(
            1 for lit in literals if lit.value
        )
        conditions = [
            lit for et in leaves for e in et.effects for lit in e.condition if lit.numeric
        ]
        assert stats["condition_literals"]["total"] == len(conditions)
        assert stats["rules_reading_a_threshold"] == sum(
            1 for rule in world.fluents.rules if rule.numeric_thresholds
        )
        assert world.stats["effects"]["condition_literals"]["by_symbol"]["numeric"] == len(
            conditions
        )
        episodes = world.stats["episodes"]
        assert (
            episodes["numeric_changes_per_event"] >= 0 and episodes["drift_changes_per_step"] >= 0
        )
        assert "numeric_literals_redrawn" in world.stats["fix_ups"]
    assert heavy.stats["episodes"]["numeric_changes_per_event"] > 0
    assert heavy.stats["episodes"]["drift_changes_per_step"] > 0


def test_define_and_view_report_the_numeric_fluents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    assert main(["define", str(DATA / "tiny.yaml"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2 numeric fluents" in text
    assert main(["view", str(out), "--preset", "base", "--include", "NUMFL"]) == 0
    import polars as pl

    view = pl.read_csv(out / "views" / "base.csv")
    assert view.columns == ["label", "leaf", "NUMFL.1", "NUMFL.2"]
    entities = pl.read_csv(out / "entities.csv")
    assert view["NUMFL.1"].to_list() == entities["NUMFL.1"].to_list()
    stats = yaml.safe_load((out / "world_stats.yaml").read_text(encoding="utf-8"))
    assert stats["numeric_fluents"]["count"] == 2
