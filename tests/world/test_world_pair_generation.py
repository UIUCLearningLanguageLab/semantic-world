"""Stage c2b: the generation of pair fluents (``docs/specs/WORLD_PHASE_C.md``, "Pair fluents",
"Preconditions", "Effects", "Conditions (F5)", "Derived fluents", "World statistics",
"Configuration", and "Determinism", read under Jon's ruling WM.63).

The worlds of the stage are ``data/world/tiny_full.yaml`` (the fast tier) and
``data/world/default_full.yaml`` (the slow tier): the tiny and default settings plus the pair
settings of the specification, until stage c4 moves them into ``tiny.yaml`` and ``default.yaml``.
Every test checks a property: the rules of the specification on the drawn literals, effects,
and conditions; a rate against a bound stated from the configured share and a sampling margin;
the off settings against the worlds without pair fluents; and the runtime against the
brute-force evaluator.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from semantic_world.figures import baseline_identity, world_figures
from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.config import PairFluentsConfig, config_from_mapping, load_config
from semantic_world.world.definition import RuntimeDefinition
from semantic_world.world.dynamics import (
    PAIR_KIND,
    PAIR_SAME,
    PAIR_SOME,
    PAIR_SOMEOF,
    PairLiteral,
)
from semantic_world.world.episodes import (
    STATS_EPISODES,
    STATS_STREAM,
    EpisodeGenerator,
    EpisodeSettings,
    Relatedness,
)
from semantic_world.world.event_types import (
    check_dynamics,
    compiled_literals,
    fixed_pair_literals,
    implied_conflict,
    keeps_no_condition,
    pair_effect_key,
    pair_literal_key,
    pair_literal_names,
    redraw_preconditions,
)
from semantic_world.world.fixtures import (
    FIXTURES_DIR,
    BruteForce,
    fixture_inputs,
    fixture_paths,
    fixture_places,
    read_fixture,
    world_fixtures,
)
from semantic_world.world.fluents import derived_initial_values
from semantic_world.world.generate import WorldResult, _initial_present, define
from semantic_world.world.history import History, replay
from semantic_world.world.identity import to_json
from semantic_world.world.pair_fluents import PairFluent, draw_initial_pairs, profile_violations
from semantic_world.world.runtime import (
    COLUMN,
    ROW,
    Event,
    InterferenceError,
    State,
    apply,
    covered,
    derive,
    initial_state,
    interference_sets,
    interferes,
    legal_bindings,
    overlaps,
)
from semantic_world.world.streams import STREAM_NAMES, WorldStreams

DATA = Path("data/world")
SLOW = pytest.mark.slow
C2A_TINY_PAIRS_IDENTITY = "7585db2f22a8e41b0990d73bad462150d282e2da49a2cf4352b52acb19dbbba2"
"""The rule-set identity of ``data/world/tiny_pairs.yaml`` in stage c2a, which generation must
leave as it is (WM.63)."""
RANDOM_STATES = 1000
SEEDS = (1, 2, 3, 4, 5)
TINY_FULL_SEEDS = (1, 2, 3, 4, 5, 8, 10, 13, 15, 17, 18, 19, 20)
"""Seeds on which the tiny world defines: on the others the tiny place taxonomy cannot place a
superordinate within its similarity bound (the tiny world's own limit, the same with and without
pair fluents)."""


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _mapping(name: str) -> dict[str, Any]:
    return yaml.safe_load((DATA / f"{name}.yaml").read_text(encoding="utf-8"))


def _deep_update(data: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            _deep_update(data[key], value)
        else:
            data[key] = value
    return data


def _world(
    name: str, seed: int | None = None, *settings: dict[str, Any], **changes: Any
) -> WorldResult:
    """The world of ``name`` with ``settings`` and ``changes`` merged into its mapping."""
    data = _mapping(name)
    for setting in settings:
        _deep_update(data, json.loads(json.dumps(setting)))
    _deep_update(data, changes)
    return define(config_from_mapping(data, source=str(DATA / f"{name}.yaml"), seed=seed))


FORCED_PROFILES = {
    "pair_fluents": {
        "locations": 0,
        "fluents": {
            "PAIRFL.1": {"symmetry": "asymmetric", "functional": True, "initial_rate": 0.2},
            "PAIRFL.2": {"symmetry": "symmetric", "functional": True, "initial_rate": 0.05},
        },
    }
}
"""Overrides that give the tiny_full world an asymmetric functional and a symmetric functional
pair fluent, so that the compiled literals, the ``SAME`` rule, and the adds that keep no
condition are exercised in the fast tier whatever seed 1 draws; and no location fluent (since
stage c3b ``tiny_full.yaml`` has one), so that the forced worlds test the pair fluents between
entities alone, as this file does, and the location dynamics are tested in
``test_world_location_generation.py``."""
FULL_SHARES = {
    "event_types": {
        "preconditions": {"pair_share": 1.0},
        "effects": {"pair_share": 1.0, "conditional_share": 1.0},
    }
}


@pytest.fixture(scope="module")
def tiny_full() -> WorldResult:
    return define(load_config(DATA / "tiny_full.yaml"))


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def forced() -> WorldResult:
    """tiny_full with forced profiles and every share at 1."""
    return _world("tiny_full", None, FORCED_PROFILES, FULL_SHARES)


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if et.performable]


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
    shape = (definition.entity_count, len(definition.base_fluents))
    values = rng.integers(0, 2, size=shape, dtype=np.uint8)
    low, high = definition.range_bounds
    numeric = rng.uniform(
        low, high, size=(definition.entity_count, len(definition.numeric_fluents))
    )
    pairs = []
    for fluent in definition.pair_fluents:
        if fluent.total:  # a location fluent (stage c3b): every entity at a uniform place
            matrix = np.zeros((shape[0], definition.place_count), dtype=bool)
            matrix[np.arange(shape[0]), rng.integers(0, definition.place_count, size=shape[0])] = (
                True
            )
            pairs.append(matrix)
            continue
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
            # a location fluent's second argument is a place (stage c3b)
            (labels[int(i)], (definition.place_labels if fluent.total else labels)[int(j)])
            for i, j in zip(*np.nonzero(matrix), strict=True)
        )
        for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True)
    }


def _labels(definition: RuntimeDefinition, label: str, binding: tuple[int, ...]) -> dict:
    # a place event's patient is a place (stage c3b: tiny_full has a location fluent)
    return Event(label, binding).binding_labels(definition)


def _records(result: WorldResult) -> list[dict[str, Any]]:
    return [et.record() for et in result.event_types.event_types]


def _check_acceptance(result: WorldResult) -> None:
    """The acceptance properties of stage c2b on one world: every pair literal achievable, no
    event type with contradicting effects, every add on an asymmetric or functional pair fluent
    with its compiled literals (``check_dynamics``), every initial pair keeping its profile, no
    ``SAME`` on a symmetric pair fluent, no drawn pair condition naming what the precondition
    names or fixed by it under the implications of item 6, no conditional add on an asymmetric
    or functional pair fluent, and at least ``derived_rules`` derived fluents reading a ``SOME``
    or ``SOMEOF`` literal. Jon's rulings of October 21, 2026: no delete on a pair fluent that
    can never hold (an initial rate of 0 and no add), and no pair literal of a precondition
    implied by, contradicted by, or implying another literal of it, compiled literals
    included."""
    pair_fluents = result.pair_fluents
    # since stage c3b tiny_full has a location fluent beside the pair fluents between entities
    assert len(pair_fluents.between_entities) == result.config.pair_fluents.count
    assert len(pair_fluents.locations) == result.config.pair_fluents.locations
    adds = {e.fluent for et in result.event_types.leaves for e in et.effects if e.pair and e.value}
    dead = {f.label for f in pair_fluents if f.initial_rate == 0.0 and f.label not in adds}
    for et in result.event_types.leaves:
        for effect in et.effects:
            if effect.pair and not effect.value:
                assert effect.fluent not in dead, (et.label, effect.text)
        pair_literals = [lit for lit in et.precondition if isinstance(lit, PairLiteral)]
        for lit in pair_literals:
            others = [o for o in pair_literals if o is not lit]
            assert not implied_conflict(lit, others, pair_fluents, et.roles), (et.label, lit.text)
    derived_initial = derived_initial_values(
        result.fluents, result.taxonomy, result.numeric, pair_fluents
    )
    check_dynamics(
        result.event_types,
        result.fluents,
        _initial_present(result.fluents, derived_initial),
        tuple(f.label for f in result.taxonomy.features.features),
        result.numeric,
        pair_fluents,
    )
    for fluent, matrix in zip(pair_fluents.fluents, pair_fluents.initial_pairs, strict=True):
        assert profile_violations(fluent, matrix) == []
        assert fluent.total == (fluent in pair_fluents.locations)
    for et in result.event_types.leaves:
        fixed = fixed_pair_literals(et.precondition, pair_fluents, et.roles)
        names = {
            name
            for lit in et.precondition
            if isinstance(lit, PairLiteral)
            for name in pair_literal_names(lit, pair_fluents[lit.pair_fluent])
        }
        for lit in et.precondition:
            if not isinstance(lit, PairLiteral):
                continue
            fluent = pair_fluents[lit.pair_fluent]
            if lit.kind == PAIR_SAME:
                assert fluent.functional and not fluent.symmetric, (et.label, lit.text)
            if et.arity == 1:
                assert lit.kind in (PAIR_SOME, PAIR_SOMEOF) and lit.first == "agent"
        for effect in et.effects:
            if keeps_no_condition(effect, pair_fluents):
                assert not effect.condition, (et.label, effect.text)
            if effect.pair and effect.value and not effect.condition:
                fluent = pair_fluents[effect.fluent]
                present = {
                    pair_literal_key(lit, pair_fluents[lit.pair_fluent]): lit.value
                    for lit in et.precondition
                    if isinstance(lit, PairLiteral)
                }
                for literal in compiled_literals(effect, fluent):
                    assert present.get(pair_literal_key(literal, fluent)) is False
            for lit in effect.condition:
                if not isinstance(lit, PairLiteral):
                    continue
                fluent = pair_fluents[lit.pair_fluent]
                key = pair_literal_key(lit, fluent)
                assert not (pair_literal_names(lit, fluent) & names), (et.label, lit.text)
                assert key not in fixed, (et.label, lit.text)
                if effect.pair:
                    assert key != pair_effect_key(effect, pair_fluents[effect.fluent])
    reading = sum(1 for rule in result.fluents.rules if rule.pair_inputs)
    assert reading >= result.config.pair_fluents.derived_rules
    # the statistics block is written, with the figures the guides read
    block = result.stats["pair_fluents"]
    assert block["count"] == len(pair_fluents)
    assert block["rules_reading_a_pair_literal"] == reading
    assert "pair_changes_per_event" in result.stats["episodes"]


# ---------------------------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------------------------


def test_the_code_defaults_follow_default_yaml_and_the_specification() -> None:
    """WM.63: ``count`` 0 and ``locations`` 0 until c4 and c3; every other key, and the pair
    shares and the pair condition weight, take the specification's defaults; ``default.yaml``
    lists them all."""
    defaults = config_from_mapping({}, source="<test>")
    block = defaults.pair_fluents
    assert isinstance(block, PairFluentsConfig)
    assert block.resolved() == {
        "count": 0,
        "locations": 0,
        "symmetry": {"symmetric": 1.0, "asymmetric": 1.0, "general": 1.0},
        "functional_rate": 0.25,
        "initial_rates": [0.0, 0.05, 0.2],
        "literal_kinds": {"pair": 0.6, "same": 0.1, "some": 0.3},
        "same_rules": 2,
        "derived_rules": 1,
        "rule_weight": 1.0,
        "fluents": {},
    }
    assert defaults.event_types.preconditions.pair_share == 0.2
    assert defaults.event_types.effects.pair_share == 0.2
    assert defaults.event_types.effects.condition_kinds["pair"] == 0.2
    listed = load_config(DATA / "default.yaml")
    assert listed.pair_fluents == block
    assert listed.event_types.preconditions.pair_share == 0.2
    assert listed.event_types.effects.pair_share == 0.2
    assert (
        listed.event_types.effects.condition_kinds == defaults.event_types.effects.condition_kinds
    )
    # default_full.yaml is default.yaml with the specification's count (and tiny_full.yaml is
    # tiny.yaml with 2 pair fluents)
    full = load_config(DATA / "default_full.yaml")
    assert full.pair_fluents.count == 4 and full.pair_fluents.locations == 1  # stage c3b
    assert dataclasses.replace(full.pair_fluents, count=0, locations=0) == block
    assert {**full.resolved(), "name": "default", "pair_fluents": block.resolved()} == (
        listed.resolved()
    )
    tiny_full = load_config(DATA / "tiny_full.yaml")
    assert tiny_full.pair_fluents == dataclasses.replace(block, count=2, locations=1)
    assert {**tiny_full.resolved(), "name": "tiny", "pair_fluents": block.resolved()} == (
        load_config(DATA / "tiny.yaml").resolved()
    )


def test_the_pair_fluents_block_is_validated() -> None:
    # stage c3a: location fluents need places
    with pytest.raises(ConfigError, match="locations"):
        config_from_mapping({"places": None, "pair_fluents": {"locations": 1}}, source="<test>")
    with pytest.raises(ConfigError, match="PAIRFL.1 to PAIRFL.2"):
        config_from_mapping(
            {"pair_fluents": {"count": 2, "fluents": {"PAIRFL.3": {}}}}, source="<test>"
        )
    # the quota applies to the first derived_rules derived fluents, all of them when the world
    # has fewer (fluents.count 0 turns the fluents off and leaves the quota empty)
    config = config_from_mapping({"pair_fluents": {"count": 1, "derived_rules": 5}}, source="<t>")
    assert config.pair_fluents.derived_rules == 5
    no_fluents = _world("tiny_full", fluents={"count": 0})
    assert no_fluents.fluents.rules == () and len(no_fluents.pair_fluents.between_entities) == 2
    with pytest.raises(ConfigError, match="literal_kinds"):
        config_from_mapping({"pair_fluents": {"literal_kinds": {"pairs": 1}}}, source="<test>")
    with pytest.raises(ConfigError, match="symmetry"):
        config_from_mapping({"pair_fluents": {"symmetry": {"reflexive": 1}}}, source="<test>")
    with pytest.raises(ConfigError, match="initial_rates"):
        config_from_mapping({"pair_fluents": {"initial_rates": [1.5]}}, source="<test>")
    with pytest.raises(ConfigError, match="condition_kinds"):
        config_from_mapping(
            {"event_types": {"effects": {"condition_kinds": {"pairs": 1}}}}, source="<test>"
        )
    with pytest.raises(ConfigError, match="pair_share"):
        config_from_mapping({"event_types": {"preconditions": {"pair_share": 2}}}, source="<test>")
    # same_rules is read and validated, and does nothing without location fluents (c3)
    config = config_from_mapping({"pair_fluents": {"count": 1, "same_rules": 0}}, source="<test>")
    assert config.pair_fluents.same_rules == 0
    # an override may give any subset of the keys; the resolved configuration loads back
    config = config_from_mapping(
        {
            "pair_fluents": {
                "count": 2,
                "fluents": {"PAIRFL.2": {"functional": True}, "PAIRFL.1": {"symmetry": "general"}},
            }
        },
        source="<test>",
    )
    assert list(config.pair_fluents.fluents) == ["PAIRFL.1", "PAIRFL.2"]
    again = config_from_mapping(config.resolved(), source="<test>")
    assert again.resolved() == config.resolved()


def test_the_three_new_streams_come_after_every_existing_stream() -> None:
    # stage c3a's world:habitats and stage c3b's world:place_dynamics follow them
    # (test_world_locations.py, test_world_location_generation.py)
    assert STREAM_NAMES[-7:-2] == (
        "pair_fluents",
        "pair_initial",
        "pair_preconditions",
        "pair_effects",
        "pair_conditions",
    )
    seeds = WorldStreams(1).seeds()
    assert list(seeds)[-5:-2] == [
        "world:pair_preconditions",
        "world:pair_effects",
        "world:pair_conditions",
    ]


# ---------------------------------------------------------------------------------------------
# Drawn profiles
# ---------------------------------------------------------------------------------------------


def test_profiles_are_drawn_in_label_order_and_an_override_shifts_no_other_draw(
    tiny_full: WorldResult,
) -> None:
    between = tiny_full.pair_fluents.between_entities
    profiles = [(f.symmetry, f.functional, f.initial_rate) for f in between]
    assert tuple(f.label for f in between) == ("PAIRFL.1", "PAIRFL.2")
    assert tiny_full.pair_fluents.labels == ("PAIRFL.1", "PAIRFL.2", "PAIRFL.3")  # the location
    assert all(rate in (0.0, 0.05, 0.2) for _, _, rate in profiles)
    # a third pair fluent keeps the first two (the draws are made in label order), and shifts
    # the location fluent's label (WM.54)
    three = _world("tiny_full", pair_fluents={"count": 3})
    assert [
        (f.symmetry, f.functional, f.initial_rate) for f in three.pair_fluents.between_entities
    ][:2] == profiles
    assert three.pair_fluents.labels == ("PAIRFL.1", "PAIRFL.2", "PAIRFL.3", "PAIRFL.4")
    # an override of PAIRFL.1 replaces its values and leaves PAIRFL.2's draws as they are
    forced = _world(
        "tiny_full",
        pair_fluents={
            "fluents": {
                "PAIRFL.1": {"symmetry": "general", "functional": True, "initial_rate": 0.05}
            }
        },
    )
    first, second = forced.pair_fluents.between_entities
    assert (first.symmetry, first.functional, first.initial_rate) == ("general", True, 0.05)
    assert (second.symmetry, second.functional, second.initial_rate) == profiles[1]
    # a partial override replaces that key alone
    partial = _world("tiny_full", pair_fluents={"fluents": {"PAIRFL.1": {"initial_rate": 0.05}}})
    first = partial.pair_fluents.fluents[0]
    assert (first.symmetry, first.functional) == profiles[0][:2] and first.initial_rate == 0.05
    # the initial pairs come from world:pair_initial by the c2a procedure (the location fluent's
    # from world:habitats, stage c3a)
    between = forced.pair_fluents.between_entities
    for fluent, matrix in zip(between, forced.pair_fluents.initial_pairs, strict=False):
        rng = WorldStreams(forced.config.seed).pair_initial
        expected = [draw_initial_pairs(f, matrix.shape[0], rng) for f in between]
        assert np.array_equal(matrix, expected[forced.pair_fluents.index(fluent.label)])


def test_an_event_file_entry_must_be_among_the_configured_pair_fluents(tmp_path: Path) -> None:
    """``tiny_pairs.yaml`` keeps stage c2a's world: its four pair fluents come from the event
    file, which the configuration must allow with ``count`` 4; with fewer the error names the
    file."""
    result = define(load_config(DATA / "tiny_pairs.yaml"))
    assert result.rule_set_id == C2A_TINY_PAIRS_IDENTITY
    assert result.event_types.pair_stats["precondition_replacements"] == 0
    assert result.event_types.pair_stats["effect_replacements"] == 0
    assert result.event_types.pair_stats["pair_conditions_drawn"] == 0
    assert not any(rule.pair_inputs for rule in result.fluents.rules)
    with pytest.raises(ConfigError, match="pair_example.yaml.*pair_fluents.count is 3") as info:
        _world("tiny_pairs", pair_fluents={"count": 3})
    assert "PAIRFL.1 to PAIRFL.3" in str(info.value)
    with pytest.raises(ConfigError, match="pair_fluents.count is 0"):
        _world("tiny_pairs", pair_fluents={"count": 0})


def test_every_configuration_without_pair_fluents_is_the_world_of_stage_c2a(
    tiny: WorldResult, tiny_full: WorldResult
) -> None:
    """A world whose configuration draws no pair fluent draws nothing from the new streams;
    the snapshot's identities and baselines hold the rest."""
    assert "pair_fluents" not in tiny.stats
    assert "pair_changes_per_event" not in tiny.stats["episodes"]
    assert tiny.event_types.pair_stats == {}
    assert not any(rule.pair_inputs for rule in tiny.fluents.rules)
    assert define(load_config(DATA / "tiny_chain.yaml")).pair_fluents.labels == ()
    # tiny_full with count 0 is the tiny world
    off = _world("tiny_full", pair_fluents={"count": 0, "locations": 0})
    assert off.rule_set_id == tiny.rule_set_id
    assert baseline_identity("tiny_full", "pair_fluents_off") == tiny.rule_set_id
    assert tiny_full.rule_set_id != tiny.rule_set_id


# ---------------------------------------------------------------------------------------------
# Acceptance on tiny_full and default_full
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", TINY_FULL_SEEDS)
def test_tiny_full_meets_the_acceptance_properties_on_every_seed(seed: int) -> None:
    _check_acceptance(_world("tiny_full", seed=seed))


def test_the_forced_profiles_exercise_the_compiled_literals_and_the_same_rule(
    forced: WorldResult,
) -> None:
    _check_acceptance(forced)
    pair_fluents = forced.pair_fluents
    assert pair_fluents["PAIRFL.1"].asymmetric and pair_fluents["PAIRFL.1"].functional
    assert pair_fluents["PAIRFL.2"].symmetric and pair_fluents["PAIRFL.2"].functional
    stats = forced.stats["pair_fluents"]
    # every effect of a two-place event type is a pair effect at a share of 1
    for et in forced.event_types.binary:
        assert all(e.pair for e in et.effects), et.label
    assert stats["effects"]["total"] > 0
    # the adds carry their compiled literals, and keep no condition, so every two-place
    # conditional effect is a delete
    compiled = stats["compiled_literals"]["by_origin"]
    assert compiled.get("profile", 0) > 0 or stats["effects"]["by_value"]["add"] == 0
    for et in forced.event_types.binary:
        for effect in et.effects:
            if effect.condition:
                assert not effect.value
    # SAME is never drawn on the symmetric functional pair fluent
    for et in forced.event_types.leaves:
        for lit in et.precondition:
            if isinstance(lit, PairLiteral) and lit.kind == PAIR_SAME:
                assert lit.pair_fluent == "PAIRFL.1"
        for effect in et.effects:
            for lit in effect.condition:
                if isinstance(lit, PairLiteral) and lit.kind == PAIR_SAME:
                    assert lit.pair_fluent == "PAIRFL.1"


def test_same_literals_are_drawn_only_on_functional_pair_fluents_that_are_not_symmetric() -> None:
    """With all the weight on ``same``, every drawn pair literal of a two-place event type is
    ``SAME`` on the asymmetric functional pair fluent, and a one-place event type, which draws
    ``some`` literals alone, gets none."""
    result = _world(
        "tiny_full",
        None,
        FORCED_PROFILES,
        FULL_SHARES,
        {"pair_fluents": {"literal_kinds": {"pair": 0, "same": 1, "some": 0}}},
    )
    drawn = [
        lit
        for et in result.event_types.leaves
        for lit in et.precondition
        if isinstance(lit, PairLiteral) and not lit.compiled
    ]
    assert drawn, "the share of 1 draws pair literals"
    for lit in drawn:
        assert lit.kind == PAIR_SAME and lit.pair_fluent == "PAIRFL.1"
    for et in result.event_types.unary:
        assert not any(isinstance(lit, PairLiteral) for lit in et.precondition)
    # with the weight on some alone, one-place event types read SOME and SOMEOF of the agent
    some = _world(
        "tiny_full",
        None,
        FULL_SHARES,
        # the quota same_rules (stage c3b) would add SAME on the location fluent
        {"pair_fluents": {"literal_kinds": {"pair": 0, "same": 0, "some": 1}, "same_rules": 0}},
    )
    kinds = {
        lit.kind
        for et in some.event_types.unary
        for lit in et.precondition
        if isinstance(lit, PairLiteral)
    }
    assert kinds and kinds <= {PAIR_SOME, PAIR_SOMEOF}
    for et in some.event_types.leaves:
        for lit in et.precondition:
            if isinstance(lit, PairLiteral) and not lit.compiled:
                assert lit.kind in (PAIR_SOME, PAIR_SOMEOF)
                if et.arity == 1:
                    assert lit.first == "agent"


@SLOW
@pytest.mark.parametrize("seed", SEEDS)
def test_default_full_meets_the_acceptance_properties_on_every_seed(seed: int) -> None:
    result = _world("default_full", seed=seed)
    _check_acceptance(result)
    assert result.stats["pair_fluents"]["precondition_literals"]["total"] > 0


# ---------------------------------------------------------------------------------------------
# The shares
# ---------------------------------------------------------------------------------------------


def _binomial_bound(share: float, draws: int) -> float:
    """Three standard deviations of a share over ``draws`` Bernoulli draws, plus one draw."""
    return 3 * math.sqrt(share * (1 - share) / draws) + 1 / draws


def _share_check(results: list[WorldResult], share: float) -> None:
    stats = [r.event_types.pair_stats for r in results]
    literal_draws = sum(s["precondition_replacement_draws"] for s in stats)
    literal_hits = sum(s["precondition_replacements"] for s in stats)
    effect_draws = sum(s["effect_replacement_draws"] for s in stats)
    effect_hits = sum(s["effect_replacements"] for s in stats)
    assert literal_draws > 0 and effect_draws > 0
    assert abs(literal_hits / literal_draws - share) <= _binomial_bound(share, literal_draws)
    assert abs(effect_hits / effect_draws - share) <= _binomial_bound(share, effect_draws)
    # the pair literals and effects of the records are the replacements that succeeded
    for result, s in zip(results, stats, strict=True):
        drawn = sum(
            1
            for et in result.event_types.leaves
            for lit in et.precondition
            if isinstance(lit, PairLiteral) and not lit.compiled
        )
        assert drawn <= s["precondition_replacements"] - s["precondition_replacements_failed"] + (
            # an inherited literal counts once per event type that keeps it
            sum(len(et.features) for et in result.event_types.binary)
        )


def test_the_pair_shares_hold_within_a_sampling_margin_on_tiny_full() -> None:
    results = [_world("tiny_full", seed=seed) for seed in TINY_FULL_SEEDS]
    _share_check(results, 0.2)


@SLOW
def test_the_pair_shares_hold_within_a_sampling_margin_on_default_full() -> None:
    _share_check([_world("default_full", seed=seed) for seed in SEEDS], 0.2)


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_with_the_pair_dynamics_off_the_event_types_are_those_of_the_world_without_pairs(
    seed: int,
) -> None:
    """The off setting of "Determinism": with the pair shares, the pair condition weight, the
    rule weight, and the quota at 0, the pair fluents exist with their initial pairs, but the
    preconditions, effects, conditions, and derived rules are those of the world with
    ``pair_fluents.count`` 0, record for record. Both worlds are taken without the location
    fluent of stage c3b, whose dynamics (the drawn moves and acts, the quota ``same_rules``)
    are location dynamics, tested in ``test_world_location_generation.py``."""
    without = _world("tiny_full", seed=seed, pair_fluents={"count": 0, "locations": 0})
    off = _world(
        "tiny_full",
        seed=seed,
        pair_fluents={"rule_weight": 0, "derived_rules": 0, "locations": 0},
        event_types={
            "preconditions": {"pair_share": 0},
            "effects": {
                "pair_share": 0,
                "condition_kinds": {"feature": 0.4, "fluent": 0.4, "numeric": 0.2, "pair": 0},
            },
        },
    )
    assert len(off.pair_fluents) == 2 and len(without.pair_fluents) == 0
    assert _records(off) == _records(without)
    assert [r.record() for r in off.fluents.rules] == [r.record() for r in without.fluents.rules]
    assert off.rule_set_id != without.rule_set_id  # the pair symbols are hashed
    # the snapshot's baseline pair_dynamics_off keeps the location fluent (stage c3b) and is
    # checked by test_world_off_settings.py


# ---------------------------------------------------------------------------------------------
# Redraws, conditions, and the implications of item 6
# ---------------------------------------------------------------------------------------------


def test_fixed_pair_literals_lists_the_implications_of_item_6() -> None:
    symmetric = PairFluent("PAIRFL.1", 1, "symmetric", False, 0.1)
    asymmetric = PairFluent("PAIRFL.2", 2, "asymmetric", True, 0.1)
    general = PairFluent("PAIRFL.3", 3, "general", True, 0.1)
    from semantic_world.world.pair_fluents import PairFluents

    n = 3
    fluents = PairFluents(
        (symmetric, asymmetric, general), tuple(np.zeros((n, n), dtype=bool) for _ in range(3))
    )

    def keys(literals: list[PairLiteral]) -> dict[tuple, bool]:
        return fixed_pair_literals(literals, fluents)

    def key(kind: str, fluent: PairFluent, first: str, second: str | None = None) -> tuple:
        return pair_literal_key(PairLiteral(kind, fluent.label, first, second, True), fluent)

    # PAIRFL.1(agent, patient) on a symmetric pair fluent fixes both rows and both columns
    fixed = keys([PairLiteral(PAIR_KIND, "PAIRFL.1", "agent", "patient", True)])
    assert fixed == {
        key(PAIR_KIND, symmetric, "agent", "patient"): True,
        key(PAIR_SOME, symmetric, "agent"): True,
        key(PAIR_SOME, symmetric, "patient"): True,
    }
    assert key(PAIR_SOMEOF, symmetric, "agent") in fixed  # the column is the row
    # on an asymmetric functional one: the reverse false, SOME true, SOMEOF true, SAME false
    fixed = keys([PairLiteral(PAIR_KIND, "PAIRFL.2", "agent", "patient", True)])
    assert fixed == {
        key(PAIR_KIND, asymmetric, "agent", "patient"): True,
        key(PAIR_KIND, asymmetric, "patient", "agent"): False,
        key(PAIR_SOME, asymmetric, "agent"): True,
        key(PAIR_SOMEOF, asymmetric, "patient"): True,
        key(PAIR_SAME, asymmetric, "agent", "patient"): False,
    }
    # NOT SOME.PAIRFL.3(agent) fixes the pair and SAME false; a positive SOME fixes nothing
    fixed = keys([PairLiteral(PAIR_SOME, "PAIRFL.3", "agent", None, False)])
    assert fixed == {
        key(PAIR_SOME, general, "agent"): False,
        key(PAIR_KIND, general, "agent", "patient"): False,
        key(PAIR_SAME, general, "agent", "patient"): False,
    }
    assert keys([PairLiteral(PAIR_SOME, "PAIRFL.3", "agent", None, True)]) == {
        key(PAIR_SOME, general, "agent"): True
    }
    # NOT SOMEOF.PAIRFL.3(patient) fixes the pair toward the patient
    assert keys([PairLiteral(PAIR_SOMEOF, "PAIRFL.3", "patient", None, False)]) == {
        key(PAIR_SOMEOF, general, "patient"): False,
        key(PAIR_KIND, general, "agent", "patient"): False,
    }
    # SAME fixes both rows true and both directions of the pair false
    assert keys([PairLiteral(PAIR_SAME, "PAIRFL.3", "agent", "patient", True)]) == {
        key(PAIR_SAME, general, "agent", "patient"): True,
        key(PAIR_SOME, general, "agent"): True,
        key(PAIR_SOME, general, "patient"): True,
        key(PAIR_KIND, general, "agent", "patient"): False,
        key(PAIR_KIND, general, "patient", "agent"): False,
    }
    # a negative pair literal and a negative SAME fix nothing beyond themselves
    assert keys([PairLiteral(PAIR_KIND, "PAIRFL.2", "agent", "patient", False)]) == {
        key(PAIR_KIND, asymmetric, "agent", "patient"): False
    }
    assert keys([PairLiteral(PAIR_SAME, "PAIRFL.2", "agent", "patient", False)]) == {
        key(PAIR_SAME, asymmetric, "agent", "patient"): False
    }


def test_implied_conflict_reads_both_directions() -> None:
    from semantic_world.world.pair_fluents import PairFluents

    symmetric = PairFluent("PAIRFL.1", 1, "symmetric", False, 0.1)
    general = PairFluent("PAIRFL.2", 2, "general", False, 0.1)
    fluents = PairFluents(
        (symmetric, general), tuple(np.zeros((3, 3), dtype=bool) for _ in range(2))
    )
    pair = PairLiteral(PAIR_KIND, "PAIRFL.2", "agent", "patient", True)
    some = PairLiteral(PAIR_SOME, "PAIRFL.2", "agent", None, True)
    not_some = PairLiteral(PAIR_SOME, "PAIRFL.2", "agent", None, False)
    not_pair = PairLiteral(PAIR_KIND, "PAIRFL.2", "agent", "patient", False)
    someof_patient = PairLiteral(PAIR_SOMEOF, "PAIRFL.2", "patient", None, True)
    # PAIRFL.2(agent, patient) implies SOME(agent) and SOMEOF(patient), either way round
    assert implied_conflict(some, [pair], fluents, ("agent", "patient"))
    assert implied_conflict(pair, [some], fluents, ("agent", "patient"))
    assert implied_conflict(someof_patient, [pair], fluents, ("agent", "patient"))
    # NOT SOME(agent) implies NOT PAIRFL.2(agent, patient), and contradicts the pair
    assert implied_conflict(not_pair, [not_some], fluents, ("agent", "patient"))
    assert implied_conflict(pair, [not_some], fluents, ("agent", "patient"))
    # a literal on another pair fluent, or on the other row, is free
    other = PairLiteral(PAIR_SOME, "PAIRFL.1", "agent", None, True)
    assert not implied_conflict(other, [pair], fluents, ("agent", "patient"))
    some_patient = PairLiteral(PAIR_SOME, "PAIRFL.2", "patient", None, True)
    assert not implied_conflict(some_patient, [pair], fluents, ("agent", "patient"))
    # on a symmetric pair fluent the pair fixes both rows
    sym_pair = PairLiteral(PAIR_KIND, "PAIRFL.1", "agent", "patient", True)
    sym_some_patient = PairLiteral(PAIR_SOME, "PAIRFL.1", "patient", None, True)
    assert implied_conflict(sym_some_patient, [sym_pair], fluents, ("agent", "patient"))


def test_dead_deletes_are_redrawn_and_implied_literals_too() -> None:
    """With every pair fluent at the rate 0 and the effect share at 1, a world draws many
    deletes on pair fluents that no effect adds; the fix-up removes every one (an own effect
    drawn again, an inherited one dropped, counted), and keeps a pair fluent that never holds
    as it is. With the precondition share at 1 the implied literals are counted too."""
    result = _world(
        "tiny_full",
        None,
        FULL_SHARES,
        {"pair_fluents": {"initial_rates": [0.0]}},
    )
    _check_acceptance(result)
    stats = result.event_types.pair_stats
    assert stats["pair_deletes_redrawn_dead"] + stats["pair_deletes_dropped_dead"] > 0
    adds = {e.fluent for et in result.event_types.leaves for e in et.effects if e.pair and e.value}
    for et in result.event_types.leaves:
        for effect in et.effects:
            if effect.pair and not effect.value:
                assert effect.fluent in adds
    # the check does not touch an explicit event file (tiny_pairs: PAIRFL.4 with the rate 0)
    pairs = define(load_config(DATA / "tiny_pairs.yaml"))
    assert pairs.rule_set_id == C2A_TINY_PAIRS_IDENTITY
    assert pairs.event_types.pair_stats["pair_deletes_redrawn_dead"] == 0


def test_pair_conditions_are_drawn_from_their_stream_and_obey_the_rules(
    forced: WorldResult,
) -> None:
    """With every effect conditional and the pair kind among the kinds, pair conditions are
    drawn; their rules are checked by ``_check_acceptance``; the redraws are counted."""
    stats = forced.stats["pair_fluents"]
    assert stats["condition_literals"]["total"] > 0
    assert stats["fix_ups"]["pair_conditions_drawn"] >= stats["condition_literals"]["total"]
    adds = stats["effects"]["by_value"]["add"]
    assert stats["fix_ups"]["adds_kept_unconditional"] > 0 or adds == 0
    # with the pair weight at 0 no pair condition is drawn, and the other kinds are drawn as
    # before: the records of a world with the pair kind off and no pair effects or literals
    # (the shares at 0) are those of the world without pair fluents (the test above); here,
    # that no condition holds a pair literal
    none = _world(
        "tiny_full",
        None,
        FORCED_PROFILES,
        event_types={
            "effects": {
                "conditional_share": 1.0,
                "condition_kinds": {"feature": 0.4, "fluent": 0.4, "numeric": 0.2, "pair": 0},
            }
        },
    )
    assert none.stats["pair_fluents"]["condition_literals"]["total"] == 0
    assert none.event_types.pair_stats["pair_conditions_drawn"] == 0


def test_a_never_legal_redraw_draws_pair_replacements_from_its_own_part(
    forced: WorldResult,
) -> None:
    """``redraw_preconditions`` with a pair part at a share of 1 draws pair literals for the
    named event types from ``world:pair_preconditions:<label>``, keeps their compiled literals
    keyed by what they name, and leaves every rule of the dynamics intact."""
    streams = WorldStreams(forced.config.seed)
    labels = [et.label for et in forced.event_types.binary] + [
        et.label for et in forced.event_types.unary
    ]
    derived_initial = derived_initial_values(
        forced.fluents, forced.taxonomy, forced.numeric, forced.pair_fluents
    )
    redrawn = redraw_preconditions(
        forced.event_types,
        labels,
        forced.fluents,
        forced.config,
        {label: streams.part("preconditions", label) for label in labels},
        derived_initial,
        forced.numeric,
        {label: streams.part("numeric_preconditions", label) for label in labels},
        forced.pair_fluents,
        {label: streams.part("pair_preconditions", label) for label in labels},
    )
    check_dynamics(
        redrawn,
        forced.fluents,
        _initial_present(forced.fluents, derived_initial),
        tuple(f.label for f in forced.taxonomy.features.features),
        forced.numeric,
        forced.pair_fluents,
    )
    drawn = 0
    for label in labels:
        before = forced.event_types.event_type(label)
        after = redrawn.event_type(label)
        compiled_before = {lit.text for lit in before.precondition if lit.pair and lit.compiled}
        compiled_after = {lit.text for lit in after.precondition if lit.pair and lit.compiled}
        assert compiled_before == compiled_after, label
        drawn += sum(1 for lit in after.precondition if lit.pair and not lit.compiled)
        # the redrawn literals leave the conditions as conditions
        fixed = fixed_pair_literals(after.precondition, forced.pair_fluents, after.roles)
        for effect in after.effects:
            for lit in effect.condition:
                if isinstance(lit, PairLiteral):
                    fluent = forced.pair_fluents[lit.pair_fluent]
                    assert pair_literal_key(lit, fluent) not in fixed
    assert drawn > 0
    # without a pair part nothing is drawn from the pair stream, as in stage c2a
    again = redraw_preconditions(
        forced.event_types,
        labels,
        forced.fluents,
        forced.config,
        {label: streams.part("preconditions", label) for label in labels},
        derived_initial,
        forced.numeric,
        {label: streams.part("numeric_preconditions", label) for label in labels},
        forced.pair_fluents,
        {},
    )
    inherited = {
        lit.text for lit in forced.event_types.feature_preconditions.values() if lit is not None
    }
    for label in labels:
        for lit in again.event_type(label).precondition:
            if lit.pair and not lit.compiled:
                assert lit.text in inherited, (label, lit.text)


# ---------------------------------------------------------------------------------------------
# Derived fluents over pairs
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_the_quota_of_derived_rules_holds_on_every_seed_of_tiny_full(seed: int) -> None:
    result = _world("tiny_full", seed=seed)
    reading = [rule for rule in result.fluents.rules if rule.pair_inputs]
    assert len(reading) >= result.config.pair_fluents.derived_rules == 1
    for rule in reading:
        for key in rule.pair_inputs:
            prefix, _, label = key.partition(".")
            assert prefix in ("SOME", "SOMEOF") and label in result.pair_fluents
        # a quota rule reads a fluent of the layer below too
        assert any(key.startswith("BOOLFL.") for key in rule.inputs)
    # the definition's literal table holds the entity-scope pair literals the rules read
    kinds = {lit["key"] for lit in result.definition.record()["literals"]}
    for rule in reading:
        assert set(rule.pair_inputs) <= kinds


def test_derived_initial_values_read_the_initial_pairs(tiny_full: WorldResult) -> None:
    derived_initial = derived_initial_values(
        tiny_full.fluents, tiny_full.taxonomy, tiny_full.numeric, tiny_full.pair_fluents
    )
    runtime = tiny_full.definition.runtime()
    true_fluents = derive(runtime, initial_state(runtime)).true_fluents(runtime)
    for label, column in derived_initial.items():
        for i, entity in enumerate(runtime.entity_labels):
            assert bool(column[i]) == (label in true_fluents[entity]), (label, entity)


def test_the_pool_leaves_the_pair_literals_out_at_weight_0_without_the_quota() -> None:
    """With ``rule_weight`` 0 and ``derived_rules`` 0 the rules are those of the world without
    pair fluents (the pool reads the stream as before); with the quota alone, the quota rule
    reads a pair literal and the others draw from the old pool; with the weight alone, a rule
    may read one."""
    without = _world("tiny_full", pair_fluents={"count": 0})
    off = _world("tiny_full", pair_fluents={"rule_weight": 0, "derived_rules": 0})
    assert [r.record() for r in off.fluents.rules] == [r.record() for r in without.fluents.rules]
    quota = _world("tiny_full", pair_fluents={"rule_weight": 0, "derived_rules": 1})
    assert quota.fluents.rules[0].pair_inputs
    heavy = _world(
        "tiny_full", fluents={"count": 8}, pair_fluents={"rule_weight": 100, "derived_rules": 0}
    )
    assert any(rule.pair_inputs for rule in heavy.fluents.rules)


# ---------------------------------------------------------------------------------------------
# The runtime against the brute-force evaluator, and the episodes
# ---------------------------------------------------------------------------------------------


def test_legal_and_derive_agree_with_brute_force_in_random_states_of_tiny_full(
    tiny_full: WorldResult,
) -> None:
    runtime = tiny_full.definition.runtime()
    record, entities, _ = fixture_inputs(tiny_full.definition)
    brute = BruteForce(record, entities, fixture_places(tiny_full.definition) or ())
    rng = np.random.default_rng(20261021)
    legal_somewhere = {label: 0 for label in _performable(runtime)}
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
            legal_somewhere[label] += bool(got)
    assert any(legal_somewhere.values())


def test_no_profile_is_broken_in_the_statistics_episodes_and_replay_reproduces_every_change(
    tiny_full: WorldResult,
) -> None:
    definition = tiny_full.definition.runtime()
    generator = EpisodeGenerator(
        definition,
        Relatedness.from_statics(tiny_full.statics, definition),
        EpisodeSettings(),
        record_legal=True,
    )
    labels = definition.entity_labels
    changes = 0
    for history in generator.run(tiny_full.config.seed, STATS_EPISODES, STATS_STREAM):
        assert history.initial_pairs is not None
        states = replay(definition, history)
        for state in states:
            for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True):
                assert profile_violations(fluent, matrix) == []
        for step, before, after in zip(history.steps, states[:-1], states[1:], strict=True):
            changed = {
                (f.label, labels[int(i)], (definition.place_labels if f.total else labels)[int(j)])
                for f, a, b in zip(definition.pair_fluents, before.pairs, after.pairs, strict=True)
                for i, j in zip(*np.nonzero(a != b), strict=True)
            }
            recorded = set()
            for event in step.events:
                for change in event.changes:
                    if change.pair:
                        recorded.add((change.pair_fluent, change.first, change.second))
                        if definition.pair_fluent(change.pair_fluent).symmetric:
                            recorded.add((change.pair_fluent, change.second, change.first))
                        changes += 1
            assert changed == recorded
        assert History.from_json(json.loads(json.dumps(history.to_json()))) == history
    assert changes > 0
    assert tiny_full.stats["episodes"]["pair_changes_per_event"] > 0


# ---------------------------------------------------------------------------------------------
# The statistics, the figures, and the fixtures
# ---------------------------------------------------------------------------------------------


def test_the_statistics_and_figures_name_the_pair_fluents_only_when_present(
    tiny_full: WorldResult, tiny: WorldResult, tmp_path: Path
) -> None:
    block = tiny_full.stats["pair_fluents"]
    assert set(block) == {
        "count",
        "by_symmetry",
        "functional",
        "fluents",
        "initial_pairs",
        "precondition_literals",
        "compiled_literals",
        "effects",
        "condition_literals",
        "rules_reading_a_pair_literal",
        "fix_ups",
        "pair_changes_per_event",
        # the location keys of stage c3b (test_world_location_generation.py)
        "between_entities",
        "locations",
        "place_event_types",
    }
    assert sum(block["by_symmetry"].values()) == block["count"] == 3
    assert block["between_entities"] == 2 and block["locations"] == 1
    assert block["initial_pairs"] == sum(int(m.sum()) for m in tiny_full.pair_fluents.initial_pairs)
    literals = block["precondition_literals"]
    assert (
        literals["total"]
        == sum(literals["by_kind"].values())
        == sum(literals["by_polarity"].values())
    )
    assert "pair_fluents" not in tiny.stats
    full_folder = tiny_full.write(tmp_path / "tiny_full")
    tiny_folder = tiny.write(tmp_path / "tiny")
    full_figures = world_figures(full_folder)
    assert full_figures["pair_fluents"]["count"] == 3
    assert "3 pair fluents (2 between entities, 1 location fluent)" in full_figures["wrote_line"]
    assert "pair_fluents" not in world_figures(tiny_folder)
    assert "pair fluents" not in world_figures(tiny_folder)["wrote_line"]
    written = yaml.safe_load((full_folder / "world_stats.yaml").read_text(encoding="utf-8"))
    assert written["pair_fluents"]["fix_ups"] == tiny_full.event_types.pair_stats


def test_the_tiny_full_fixtures_are_committed_and_reproducible(tiny_full: WorldResult) -> None:
    names = {path.stem for path in fixture_paths(FIXTURES_DIR)}
    assert {
        "tiny_full_seed1",
        "tiny_full_seed2",
        "tiny_full_seed3",
        "tiny_full_seed4",
        "tiny_full_illegal",
        "tiny_full_interference",
    } <= names
    for fixture in world_fixtures(tiny_full.definition, "tiny_full"):
        path = FIXTURES_DIR / f"{fixture['name']}.json"
        assert (
            path.read_text(encoding="utf-8") == to_json(fixture, indent=1, sort_keys=False) + "\n"
        ), path
        assert fixture["rule_set_id"] == tiny_full.rule_set_id
        assert fixture["initial_pairs"] or any(step["pairs"] for step in fixture["steps"])
    for path in fixture_paths(FIXTURES_DIR):
        if path.stem.startswith("tiny_full_"):
            assert read_fixture(path)["rule_set_id"] == tiny_full.rule_set_id


# ---------------------------------------------------------------------------------------------
# The default world of stage c2a, unchanged
# ---------------------------------------------------------------------------------------------


@SLOW
def test_default_full_with_count_0_is_the_default_world_of_stage_c2a() -> None:
    default = define(load_config(DATA / "default.yaml"))
    assert "pair_fluents" not in default.stats
    assert baseline_identity("default_full", "pair_fluents_off") == default.rule_set_id


@SLOW
def test_with_the_pair_dynamics_off_default_full_has_the_default_event_types() -> None:
    """Without the location fluent of stage c3b, whose place dynamics the default world lacks
    (``test_world_location_generation.py`` tests them)."""
    default = define(load_config(DATA / "default.yaml"))
    off = _world(
        "default_full",
        pair_fluents={"rule_weight": 0, "derived_rules": 0, "locations": 0},
        event_types={
            "preconditions": {"pair_share": 0},
            "effects": {
                "pair_share": 0,
                "condition_kinds": {"feature": 0.4, "fluent": 0.4, "numeric": 0.2, "pair": 0},
            },
        },
    )
    assert _records(off) == _records(default)
    assert [r.record() for r in off.fluents.rules] == [r.record() for r in default.fluents.rules]
    # the snapshot's baseline pair_dynamics_off keeps the location fluent (stage c3b) and is
    # checked by test_world_off_settings.py


# ---------------------------------------------------------------------------------------------
# Non-interference with rows and columns as single keys (Jon's ruling 3, the follow-up)
# ---------------------------------------------------------------------------------------------


def _expanded(definition: RuntimeDefinition, keys) -> set[tuple]:
    """A read or write set with every row and column expanded to its pairs over every entity
    of the definition, the form of stage c2a."""
    found: set[tuple] = set()
    n = definition.entity_count
    for key in keys:
        if len(key) == 3 and key[0] == ROW:
            found |= {(key[1], key[2], z) for z in range(n) if z != key[2]}
        elif len(key) == 3 and key[0] == COLUMN:
            found |= {(key[1], z, key[2]) for z in range(n) if z != key[2]}
        else:
            found.add(key)
    return found


def test_overlaps_says_when_two_keys_share_a_fact() -> None:
    pair = ("PAIRFL.1", 2, 5)
    assert overlaps(pair, pair)
    assert overlaps(pair, (ROW, "PAIRFL.1", 2)) and overlaps((ROW, "PAIRFL.1", 2), pair)
    assert overlaps(pair, (COLUMN, "PAIRFL.1", 5))
    assert not overlaps(pair, (ROW, "PAIRFL.1", 5)) and not overlaps(pair, (COLUMN, "PAIRFL.1", 2))
    assert not overlaps(pair, (ROW, "PAIRFL.2", 2))
    assert overlaps((ROW, "PAIRFL.1", 2), (COLUMN, "PAIRFL.1", 5))
    assert not overlaps((ROW, "PAIRFL.1", 2), (COLUMN, "PAIRFL.1", 2))  # no pair (2, 2)
    assert not overlaps((ROW, "PAIRFL.1", 2), (ROW, "PAIRFL.1", 3))
    assert not overlaps((2, "BOOLFL.1"), pair) and overlaps((2, "BOOLFL.1"), (2, "BOOLFL.1"))
    # the covered keys come sorted, base keys before pair keys
    writes = [pair, (1, "BOOLFL.2"), ("PAIRFL.1", 0, 1)]
    reads = [(ROW, "PAIRFL.1", 2), (1, "BOOLFL.2")]
    assert covered(writes, reads) == [(1, "BOOLFL.2"), pair]


def test_the_single_keys_give_the_verdicts_of_the_expanded_sets(forced: WorldResult) -> None:
    """On the forced tiny world (pair literals, pair conditions, and derived fluents over pairs
    everywhere), ``interferes`` by the single keys agrees with the intersection of the sets
    expanded to every pair, for every pair of legal events in random states, and ``apply``
    names the same fact and reader as the expanded form would."""
    runtime = forced.definition.runtime()
    rng = np.random.default_rng(7)
    compared = 0
    for _ in range(60):
        state = _random_state(runtime, rng)
        events = [
            (label, b)
            for label in _performable(runtime)
            for b in legal_bindings(runtime, state, label)
        ]
        from semantic_world.world.runtime import Event

        events = [Event(label, b) for label, b in events]
        sets = {event: interference_sets(runtime, event) for event in events}
        for i, a in enumerate(events):
            for b in events[i + 1 :]:
                writes_a, reads_a = sets[a]
                writes_b, reads_b = sets[b]
                expanded = bool(
                    _expanded(runtime, writes_a) & _expanded(runtime, writes_b)
                    or _expanded(runtime, writes_a) & _expanded(runtime, reads_b)
                    or _expanded(runtime, writes_b) & _expanded(runtime, reads_a)
                )
                assert interferes(runtime, a, b) == expanded, (a, b)
                compared += 1
                if not expanded:
                    continue
                with pytest.raises(InterferenceError) as info:
                    apply(runtime, state, [a, b])
                # the fact named is the smallest written key of the expanded intersection
                shared = sorted(
                    _expanded(runtime, writes_a) & _expanded(runtime, writes_b),
                    key=lambda k: (len(k), k),
                )
                if not shared:
                    shared = sorted(
                        _expanded(runtime, writes_a) & _expanded(runtime, reads_b),
                        key=lambda k: (len(k), k),
                    ) or sorted(
                        _expanded(runtime, writes_b) & _expanded(runtime, reads_a),
                        key=lambda k: (len(k), k),
                    )
                key = shared[0]
                labels = runtime.entity_labels
                fact = (
                    f"{key[0]} between {labels[key[1]]} and {labels[key[2]]}"
                    if len(key) == 3
                    else f"{key[1]} of {labels[key[0]]}"
                )
                assert fact in str(info.value)
    assert compared > 1000


def test_apply_shares_the_pair_matrices_no_effect_wrote(tiny_full: WorldResult) -> None:
    runtime = tiny_full.definition.runtime()
    state = initial_state(runtime)
    after = apply(runtime, state, [])
    for before_matrix, after_matrix in zip(state.pairs, after.pairs, strict=True):
        assert after_matrix is before_matrix
        assert not after_matrix.flags.writeable
