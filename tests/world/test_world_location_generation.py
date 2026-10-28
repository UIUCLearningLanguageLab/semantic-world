"""Stage c3b of ``docs/specs/WORLD_PHASE_C.md``: locations generated.

The drawn dynamics of place event types (moves and acts at ``move_share`` from
``world:place_dynamics``, with their further literals, effects, and conditions on the agent),
``SAME`` on a location fluent in drawn preconditions at the pair share, the quota ``same_rules``,
the ``place_kind`` key (WM.65), achievability over every effect of the world (WM.66), the location
statistics, and the full worlds with ``locations: 1`` (WM.63). The world of the fast tier is
``data/world/tiny_full.yaml``, whose two place event types draw their dynamics; the slow tier
runs ``data/world/default_full.yaml`` and its seeds 1 to 5. Every acceptance check of stage c3a
(``test_world_locations.py``, over the chain example) is run here over the full worlds: every
entity at exactly one place at every time point, every move to the bound place and nothing else,
no move to the agent's own place, every act at the agent's place, ``legal`` and ``derive``
against the brute-force evaluator in random states, applying together against one at a time in
every order, no profile broken in the statistics episodes, and replay. The follow-up of October
28, 2026, adds the density band of the place requirements (``place_event_types.density``, Jon's
ruling WM.67) and the expected-share habitat test on seeds 1 to 5 of ``default_full``. Every test
writes to its own temporary folder, and every bound is stated from the configuration.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.definition import RuntimeDefinition, runtime_definition
from semantic_world.world.dynamics import PAIR_KIND, PAIR_SAME, PAIR_SOME, PAIR_SOMEOF, PairLiteral
from semantic_world.world.episodes import (
    STATS_STREAM,
    EpisodeGenerator,
    EpisodeSettings,
    Relatedness,
)
from semantic_world.world.errors import DefinitionError
from semantic_world.world.event_types import check_dynamics
from semantic_world.world.fixtures import (
    BruteEvent,
    BruteForce,
    check_fixture,
    check_fixture_brute,
    fixture_inputs,
    fixture_places,
    world_fixtures,
)
from semantic_world.world.fluents import derived_initial_values
from semantic_world.world.generate import WorldResult, _initial_present, define
from semantic_world.world.history import replay
from semantic_world.world.identity import rule_set_id
from semantic_world.world.locations import habitat_weights
from semantic_world.world.pair_fluents import draw_initial_pairs, profile_violations
from semantic_world.world.place_dynamics import (
    PLACE_DYNAMICS_STATS,
    check_place_dynamics,
    random_place_pair_literal,
    redraw_place_preconditions,
)
from semantic_world.world.places import MAX_TRIES
from semantic_world.world.runtime import (
    Event,
    State,
    apply,
    derive,
    legal_bindings,
)
from semantic_world.world.streams import STREAM_NAMES, WorldStreams

SLOW = pytest.mark.slow
DATA = Path("data/world")
RANDOM_STATES = 1000
RANDOM_SETS = 1000
EPISODES = 1000
SEEDS = [1, 2, 3, 4, 5]
LOCATION = "PAIRFL.3"
"""The location fluent of ``tiny_full`` (2 pair fluents between entities, then the location)."""


def _world(name: str, seed: int | None = None, **overrides: Any) -> WorldResult:
    """``data/world/<name>.yaml`` with top-level blocks overridden (a block given as a mapping
    is merged into the file's block), defined in memory."""
    path = DATA / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if seed is not None:
        data["seed"] = seed
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            for inner, setting in value.items():
                if isinstance(setting, dict) and isinstance(data[key].get(inner), dict):
                    data[key][inner] = {**data[key][inner], **setting}
                else:
                    data[key][inner] = setting
        else:
            data[key] = value
    return define(config_from_mapping(data, source=str(path)))


@pytest.fixture(scope="module")
def tiny_full() -> WorldResult:
    return define(load_config(DATA / "tiny_full.yaml"))


@pytest.fixture(scope="module")
def runtime(tiny_full: WorldResult) -> RuntimeDefinition:
    return tiny_full.definition.runtime()


@pytest.fixture(scope="module")
def brute(tiny_full: WorldResult) -> BruteForce:
    record, entities, _ = fixture_inputs(tiny_full.definition)
    return BruteForce(record, entities, fixture_places(tiny_full.definition) or ())


@pytest.fixture(scope="module")
def default_full() -> WorldResult:
    return define(load_config(DATA / "default_full.yaml"))


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if et.performable]


def _random_state(definition: RuntimeDefinition, rng: np.random.Generator) -> State:
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


def _place_of(definition: RuntimeDefinition, state: State, entity: int, fluent: str) -> int:
    row = state.pairs[definition.pair_index(fluent)][entity]
    where = np.flatnonzero(row)
    assert where.size == 1, "every entity is at exactly one place"
    return int(where[0])


def _present(result: WorldResult) -> set[tuple[str, bool]]:
    """The Boolean values present at ``TIME.1``, base and derived, as ``define`` computes them."""
    derived = derived_initial_values(
        result.fluents, result.taxonomy, result.numeric, result.pair_fluents
    )
    return _initial_present(result.fluents, derived)


def _location_of(result: WorldResult) -> str:
    (fluent,) = result.pair_fluents.locations
    return fluent.label


def _same_on_location(result: WorldResult) -> list[str]:
    """The two-place event types whose precondition requires ``SAME`` on a location fluent."""
    locations = {f.label for f in result.pair_fluents.locations}
    return [
        et.label
        for et in result.event_types.binary
        if any(
            isinstance(lit, PairLiteral)
            and lit.kind == PAIR_SAME
            and lit.value
            and lit.pair_fluent in locations
            for lit in et.precondition
        )
    ]


# ---------------------------------------------------------------------------------------------
# The stream, the configurations, and the drawn dynamics
# ---------------------------------------------------------------------------------------------


def test_the_place_dynamics_stream_comes_after_every_earlier_stream(tiny_full: WorldResult) -> None:
    assert STREAM_NAMES[-1] == "place_dynamics" and STREAM_NAMES[-2] == "habitats"
    assert list(WorldStreams(1).seeds())[-1] == "world:place_dynamics"
    assert "world:place_dynamics" in tiny_full.stream_seeds


def test_the_full_worlds_have_one_location_fluent_and_the_tiny_requirements_read_the_place() -> (
    None
):
    """WM.63: the location settings live in the full worlds; WM.62: every tiny place
    requirement reads the place, in ``tiny.yaml`` and ``tiny_full.yaml`` alike."""
    for name in ("tiny_full", "default_full"):
        config = load_config(DATA / f"{name}.yaml")
        assert config.pair_fluents.locations == 1
        assert config.pair_fluents.same_rules == 2
        # the follow-up of October 28, 2026 (WM.67): the band and the strength
        assert config.place_event_types is not None and config.places is not None
        assert config.place_event_types.density == (0.05, 0.5)
        assert config.places.habitat_strength == 2
    for name in ("tiny", "default"):
        config = load_config(DATA / f"{name}.yaml")
        assert config.pair_fluents.locations == 0
        assert config.place_event_types is not None and config.places is not None
        assert config.place_event_types.density is None
        assert config.places.habitat_strength == 1
    tiny = load_config(DATA / "tiny.yaml")
    assert tiny.place_event_types is not None
    assert tiny.place_event_types.constraint_families.get("agent", 0) == 0
    for name in ("tiny", "tiny_full"):
        result = define(load_config(DATA / f"{name}.yaml"))
        assert result.places is not None
        for et in result.places.event_types:
            assert any(
                lit.key.startswith("patient.") for c in et.constraints for lit in c.literals
            ) or any(c.family == "comparison" for c in et.constraints), et.label


@pytest.mark.parametrize("name", ["tiny_full", pytest.param("default_full", marks=SLOW)])
def test_every_place_event_type_has_dynamics_with_a_place_kind_that_agrees(name: str) -> None:
    """Every place event type of the full worlds is a move or an act (WM.65): a move with its
    add on the location fluent first among its effects and its compiled literal last in the
    precondition, an act with its location literal last and no location effect; every literal
    is achievable over every effect of the world (WM.66), every further literal, effect, and
    condition reads the agent (or ``SOMEOF`` of the place, in a move), and no drawn condition
    reads a place's features."""
    result = define(load_config(DATA / f"{name}.yaml"))
    assert result.places is not None
    location = _location_of(result)
    record = {e["label"]: e for e in result.definition.record()["event_types"]}
    kinds = {"move": 0, "act": 0}
    for et in result.places.event_types:
        assert et.performable and not et.explicit and et.location == location
        kinds[et.dynamics_kind] += 1  # type: ignore[index]
        entry = record[et.label]
        assert entry["place_kind"] == et.dynamics_kind
        assert list(entry)[:3] == ["label", "kind", "place_kind"]
        last = et.precondition[-1]
        assert isinstance(last, PairLiteral) and last.kind == PAIR_KIND
        assert (last.first, last.second, last.pair_fluent) == ("agent", "patient", location)
        if et.move:
            assert last.value is False and last.origin == "move"
            assert et.effects[0].pair and et.effects[0].value and et.effects[0].fluent == location
            assert sum(1 for e in et.effects if e.pair) == 1
        else:
            assert last.value is True and last.origin == "act"
            assert not any(e.pair for e in et.effects)
        for lit in et.precondition[:-1]:
            if isinstance(lit, PairLiteral):
                fluent = result.pair_fluents[lit.pair_fluent]
                if fluent.total:
                    assert et.move and lit.kind == PAIR_SOMEOF and lit.first == "patient"
                    assert lit.value is True, (
                        "a move to an empty place implies its compiled literal"
                    )
                else:
                    assert lit.kind in (PAIR_SOME, PAIR_SOMEOF) and lit.first == "agent"
            else:
                assert lit.role == "agent"
        for effect in et.effects:
            assert effect.role == "agent"
            for lit in effect.condition:
                if isinstance(lit, PairLiteral):
                    assert lit.first == (
                        "patient" if result.pair_fluents[lit.pair_fluent].total else "agent"
                    )
                else:
                    assert lit.role == "agent"
                    assert not lit.symbol.startswith("PLACE")
    assert kinds["move"] + kinds["act"] == len(result.places.event_types)
    # the invariants, with every effect of the world in the pool
    present = _present(result)
    place_effects = [e for et in result.places.event_types for e in et.effects]
    entity_effects = [e for et in result.event_types.event_types for e in et.effects]
    features = tuple(f.label for f in result.taxonomy.features.features)
    check_dynamics(
        result.event_types,
        result.fluents,
        present,
        features,
        result.numeric,
        result.pair_fluents,
        place_effects,
    )
    check_place_dynamics(
        result.places,
        result.fluents,
        present,
        entity_effects,
        {
            (e.fluent, 1 if e.amount > 0 else -1)
            for e in entity_effects + place_effects
            if e.numeric
        },
        result.numeric,
        result.pair_fluents,
    )
    stats = result.stats["places"]
    assert stats["place_event_types_by_kind"] == {
        "moves": kinds["move"],
        "acts": kinds["act"],
        "without_dynamics": 0,
        "explicit": 0,
    }
    assert set(stats["dynamics"]) == set(PLACE_DYNAMICS_STATS)
    assert stats["dynamics"]["drawn"] == len(result.places.event_types)


def test_the_move_share_is_drawn_from_the_configuration() -> None:
    """With ``move_share`` 1 every place event type is a move, with 0 every one is an act; the
    kinds are a draw against the share, from ``world:place_dynamics``."""
    moves = _world("tiny_full", place_event_types={"count": 4, "move_share": 1.0})
    assert all(et.move for et in moves.places.event_types)  # type: ignore[union-attr]
    acts = _world("tiny_full", place_event_types={"count": 4, "move_share": 0.0})
    assert all(et.act for et in acts.places.event_types)  # type: ignore[union-attr]
    for result in (moves, acts):
        stats = result.stats["episodes"]
        assert (stats["move_share"] == 0.0) == (result is acts)
        assert (stats["act_share"] == 0.0) == (result is moves)


def test_a_move_draws_its_further_effects_by_the_effects_count_with_its_add_first() -> None:
    """``effects.count`` fixed at 2: every move has its add and one further effect on the
    agent, every act two effects on the agent; at 1, a move has its add alone."""
    two = _world(
        "tiny_full", place_event_types={"count": 6}, event_types={"effects": {"count": {2: 1}}}
    )
    for et in two.places.event_types:  # type: ignore[union-attr]
        assert len(et.effects) == 2
        if et.move:
            assert et.effects[0].pair and not et.effects[1].pair
    one = _world(
        "tiny_full", place_event_types={"count": 6}, event_types={"effects": {"count": {1: 1}}}
    )
    for et in one.places.event_types:  # type: ignore[union-attr]
        assert len(et.effects) == 1
        assert et.effects[0].pair == et.move


def test_the_drawn_pair_literals_of_place_event_types_take_the_allowed_forms() -> None:
    """With the pair share at 1 and ``some`` the one kind with weight, every further literal
    that could be replaced is a pair literal of the allowed forms: ``SOME`` or ``SOMEOF`` of
    the agent on a pair fluent between entities, or ``SOMEOF`` of the place (positive, in a
    move); with the ``some`` weight at 0 no place event type draws a pair literal."""
    full = _world(
        "tiny_full",
        place_event_types={"count": 8},
        event_types={"preconditions": {"pair_share": 1.0, "literals": {2: 1}}},
        pair_fluents={"literal_kinds": {"pair": 0, "same": 0, "some": 1}},
    )
    location = _location_of(full)
    forms = {"agent": 0, "place": 0}
    for et in full.places.event_types:  # type: ignore[union-attr]
        for lit in et.precondition[:-1]:
            assert isinstance(lit, PairLiteral), lit.text
            if lit.pair_fluent == location:
                assert lit.kind == PAIR_SOMEOF and lit.first == "patient" and lit.value and et.move
                forms["place"] += 1
            else:
                assert lit.kind in (PAIR_SOME, PAIR_SOMEOF) and lit.first == "agent"
                forms["agent"] += 1
    assert forms["agent"] > 0 and forms["place"] > 0, forms
    stats = full.stats["places"]["dynamics"]
    assert stats["pair_replacements"] == forms["agent"] + forms["place"]
    none = _world(
        "tiny_full",
        place_event_types={"count": 8},
        event_types={"preconditions": {"pair_share": 1.0}},
        pair_fluents={"literal_kinds": {"pair": 1, "same": 1, "some": 0}},
    )
    for et in none.places.event_types:  # type: ignore[union-attr]
        assert not any(isinstance(lit, PairLiteral) for lit in et.precondition[:-1])
    assert none.stats["places"]["dynamics"]["pair_replacement_draws"] == 0
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(200):
        lit = random_place_pair_literal(rng, full.pair_fluents, location)
        seen.add((lit.pair_fluent == location, lit.kind, lit.first, lit.value))
    assert seen == {
        *((True, PAIR_SOMEOF, "patient", v) for v in (True, False)),
        *((False, k, "agent", v) for k in (PAIR_SOME, PAIR_SOMEOF) for v in (True, False)),
    }


def test_conditions_of_place_event_types_read_the_agent_or_the_place_column() -> None:
    """With every further effect conditional, every condition literal of a place event type
    is a feature, a Boolean fluent, or a numeric threshold of the agent, or a pair literal of
    the allowed forms; a pair condition names nothing the precondition names and is not fixed
    by it, so ``SOMEOF`` of the place is a condition of a move alone."""
    result = _world(
        "tiny_full",
        place_event_types={"count": 8},
        event_types={"effects": {"conditional_share": 1.0, "count": {2: 1}}},
    )
    location = _location_of(result)
    kinds = {"feature": 0, "fluent": 0, "numeric": 0, "pair": 0}
    for et in result.places.event_types:  # type: ignore[union-attr]
        for effect in et.effects:
            if effect.pair:
                assert not effect.condition
                continue
            for lit in effect.condition:
                if isinstance(lit, PairLiteral):
                    kinds["pair"] += 1
                    if lit.pair_fluent == location:
                        assert et.move and lit.kind == PAIR_SOMEOF and lit.first == "patient"
                    else:
                        assert lit.kind in (PAIR_SOME, PAIR_SOMEOF) and lit.first == "agent"
                else:
                    kinds[lit.kind] += 1
                    assert lit.role == "agent" and not lit.symbol.startswith("PLACE")
                    if lit.kind == "fluent":
                        assert (lit.role, lit.symbol) != (effect.role, effect.fluent)
                    assert not any(
                        not p.pair and (p.role, p.fluent) == (lit.role, lit.symbol)
                        for p in et.precondition
                    )
    assert sum(kinds.values()) > 0 and kinds["feature"] + kinds["fluent"] > 0, kinds
    stats = result.stats["places"]["dynamics"]
    assert stats["conditions_drawn"] == sum(
        1
        for et in result.places.event_types
        for e in et.effects
        if not e.pair  # type: ignore[union-attr]
    )


def test_the_place_draws_leave_the_entity_event_types_and_every_other_draw_unchanged(
    tiny_full: WorldResult,
) -> None:
    """The place dynamics come from their own stream, after every other draw, so the tiny full
    world with ``move_share`` 1 has the same entity event types, fluents, pair fluents, and
    habitats as with the default share; and the world without location fluents is the world of
    stage c2b, by the snapshot's baseline ``locations_off`` (``test_world_off_settings.py``)."""
    other = _world("tiny_full", place_event_types={"move_share": 1.0})
    assert other.event_types.event_types == tiny_full.event_types.event_types
    assert other.fluents.rules == tiny_full.fluents.rules
    assert other.pair_fluents.fluents == tiny_full.pair_fluents.fluents
    for a, b in zip(
        other.pair_fluents.initial_pairs, tiny_full.pair_fluents.initial_pairs, strict=True
    ):
        assert np.array_equal(a, b)
    assert [et.requirement_label for et in other.places.event_types] == [  # type: ignore[union-attr]
        et.requirement_label
        for et in tiny_full.places.event_types  # type: ignore[union-attr]
    ]
    assert other.rule_set_id != tiny_full.rule_set_id or all(
        et.move
        for et in tiny_full.places.event_types  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------------------------
# The density band of the place requirements (WM.67, the follow-up of October 28, 2026)
# ---------------------------------------------------------------------------------------------


def _densities(result: WorldResult) -> dict[str, float]:
    assert result.places is not None
    return {
        et.label: float(result.places.requirement[et.label].mean())
        for et in result.places.event_types
    }


def test_every_drawn_place_requirement_lies_in_the_band(tiny_full: WorldResult) -> None:
    """With a band every drawn requirement's share of the bindings lies in it (the share is
    recomputed from the matrix, not read off the record), the statistics report the band with
    no place event type outside it, and a narrower band on more place event types still holds
    and records the extra draws under ``tries``."""
    band = tiny_full.config.place_event_types.density  # type: ignore[union-attr]
    assert band == (0.05, 0.5)
    for label, density in _densities(tiny_full).items():
        assert band[0] <= density <= band[1], label
    stats = tiny_full.stats["places"]
    assert stats["density_band"] == [0.05, 0.5]
    assert stats["outside_density"] == 0 and stats["outside_density_labels"] == []
    assert stats["warnings"] == []
    narrow = _world("tiny_full", place_event_types={"count": 8, "density": [0.2, 0.3]})
    for label, density in _densities(narrow).items():
        assert 0.2 <= density <= 0.3, label
    tries = narrow.stats["places"]["tries"]
    assert all(1 <= t <= MAX_TRIES for t in tries.values()) and max(tries.values()) > 1
    assert narrow.stats["places"]["outside_density"] == 0


def test_a_world_without_a_band_draws_its_place_requirements_as_before(
    tiny_full: WorldResult,
) -> None:
    """With ``density: null`` only a degenerate requirement is drawn again, so the place event
    types of ``tiny_full`` without the band are those of ``tiny.yaml``, requirement for
    requirement (the same place settings and the same stream; the pair streams come after),
    and ``tiny.yaml``'s own identity is the snapshot's (``tests/figures``). The band's
    statistics keys are absent, and the code's default is no band."""
    tiny = define(load_config(DATA / "tiny.yaml"))
    without = _world("tiny_full", place_event_types={"density": None})
    assert without.config.place_event_types.density is None  # type: ignore[union-attr]
    assert config_from_mapping({}, source="<test>").place_event_types.density is None
    assert [et.constraints for et in without.places.event_types] == [  # type: ignore[union-attr]
        et.constraints
        for et in tiny.places.event_types  # type: ignore[union-attr]
    ]
    assert _densities(without) == _densities(tiny)
    for key in ("density_band", "outside_density", "outside_density_labels"):
        assert key not in without.stats["places"] and key not in tiny.stats["places"]
    # the band moved the tiny_full place requirements (one of them held for 0.917 of the
    # bindings without it)
    assert _densities(without) != _densities(tiny_full)


def test_an_explicit_requirement_outside_the_band_is_kept(tmp_path: Path) -> None:
    """An event file's requirement replaces the drawn one after the draws and is never drawn
    again, whatever its density; the drawn requirements of the other place event types keep
    the band."""
    file = tmp_path / "wide.yaml"
    file.write_text(
        yaml.safe_dump(
            {"place_event_types": {"PLACEEVENTTYPE.1": {"kind": "move", "requirement": "TRUE"}}}
        ),
        encoding="utf-8",
    )
    result = _world("tiny_full", event_types={"event_file": str(file)})
    assert result.places is not None
    first, second = result.places.event_types
    assert first.explicit and first.density == 1.0 and first.in_band
    assert not second.explicit and 0.05 <= second.density <= 0.5 and second.in_band
    stats = result.stats["places"]
    assert stats["outside_density"] == 0 and stats["warnings"] == []


def test_a_band_no_draw_can_meet_is_reported_with_a_warning() -> None:
    """A band between 0 and the smallest non-zero share of the 48 bindings (1/48) can hold for
    no drawn requirement: each place event type takes ``MAX_TRIES`` draws, keeps the last, and
    is reported under ``outside_density`` with a warning, as the two-place range reports its
    event types; a degenerate last draw gets the degenerate warning and counts as outside."""
    result = _world("tiny_full", place_event_types={"density": [0.001, 0.002]})
    assert result.places is not None
    stats = result.stats["places"]
    labels = [et.label for et in result.places.event_types]
    assert stats["outside_density"] == len(labels) and stats["outside_density_labels"] == labels
    assert all(t == MAX_TRIES for t in stats["tries"].values())
    assert len(stats["warnings"]) == len(labels)
    for et, warning in zip(result.places.event_types, stats["warnings"], strict=True):
        assert not et.in_band and et.label in warning and "200 tries" in warning
        assert ("outside the density band" in warning) != (et.density in (0.0, 1.0))
    # the world is still built and checked: every place event type is a move or an act
    assert all(et.performable for et in result.places.event_types)


def test_the_band_is_validated() -> None:
    with pytest.raises(ConfigError, match="density"):
        config_from_mapping({"place_event_types": {"density": [0.5, 0.2]}}, source="<test>")
    with pytest.raises(ConfigError, match="density"):
        config_from_mapping({"place_event_types": {"density": [0.1]}}, source="<test>")
    with pytest.raises(ConfigError, match="density"):
        config_from_mapping({"place_event_types": {"density": [0.1, 1.5]}}, source="<test>")
    config = config_from_mapping({"place_event_types": {"density": [0.1, 0.4]}}, source="<test>")
    assert config.place_event_types.density == (0.1, 0.4)  # type: ignore[union-attr]
    assert config.place_event_types.resolved()["density"] == [0.1, 0.4]  # type: ignore[union-attr]
    assert (
        config_from_mapping(
            {"place_event_types": {"density": None}}, source="<test>"
        ).place_event_types.density
        is None
    )  # type: ignore[union-attr]


# ---------------------------------------------------------------------------------------------
# SAME on location fluents, and the quota same_rules
# ---------------------------------------------------------------------------------------------


def test_the_quota_gives_same_rules_two_place_event_types_same_on_the_location_fluent(
    tiny_full: WorldResult,
) -> None:
    quota = tiny_full.config.pair_fluents.same_rules
    assert quota == 2
    assert len(_same_on_location(tiny_full)) >= quota
    fix_ups = tiny_full.stats["pair_fluents"]["fix_ups"]
    assert fix_ups["same_rules_added"] == quota and fix_ups["same_rules_short"] == 0
    assert set(tiny_full.event_types.same_quota) <= set(_same_on_location(tiny_full))
    assert len(tiny_full.event_types.same_quota) == quota
    for label, literal in tiny_full.event_types.same_quota.items():
        et = tiny_full.event_types.event_type(label)
        assert literal in et.precondition and literal.pair_fluent == LOCATION
        assert literal.value and (literal.first, literal.second) == ("agent", "patient")
    # the loader reads the literal back as a pair_same literal on a total pair fluent
    runtime = tiny_full.definition.runtime()
    for label in tiny_full.event_types.same_quota:
        assert any(
            lit.pair and lit.kind == PAIR_SAME and lit.pair_fluent == LOCATION
            for lit in runtime.event_type(label).precondition
        )


def test_the_quota_draws_from_its_own_part_and_changes_nothing_else(tiny_full: WorldResult) -> None:
    """With ``same_rules`` 0 the two-place event types are those of the quota world with the
    quota's literals removed, record for record, and every other draw is unchanged; with a
    quota above the number of qualifying event types every qualifying one gets the literal and
    the shortfall is recorded."""
    none = _world("tiny_full", pair_fluents={"same_rules": 0})
    assert none.stats["pair_fluents"]["fix_ups"]["same_rules_added"] == 0
    assert not none.event_types.same_quota
    quota = tiny_full.event_types.same_quota
    for a, b in zip(none.event_types.event_types, tiny_full.event_types.event_types, strict=True):
        assert a.label == b.label and a.effects == b.effects
        expected = tuple(lit for lit in b.precondition if lit != quota.get(b.label))
        assert a.precondition == expected, a.label
    assert none.fluents.rules == tiny_full.fluents.rules
    assert none.places.event_types == tiny_full.places.event_types  # type: ignore[union-attr]
    many = _world("tiny_full", pair_fluents={"same_rules": 10})
    binary = [et for et in many.event_types.binary]
    added = many.stats["pair_fluents"]["fix_ups"]["same_rules_added"]
    assert added == len(many.event_types.same_quota) <= len(binary)
    assert many.stats["pair_fluents"]["fix_ups"]["same_rules_short"] == 10 - added
    qualifying = [
        et.label
        for et in binary
        if many.statics.relations.matrix(et.label).any()  # type: ignore[union-attr]
    ]
    assert set(many.event_types.same_quota) == set(qualifying)


def test_same_on_a_location_fluent_is_drawn_at_the_share_from_its_own_part() -> None:
    """With ``same`` the one kind with weight, the pair share at 1, and the quota at 0, some
    two-place event type draws ``SAME`` on the location fluent at the share, from the part
    ``world:pair_preconditions:locations``; the tiny full world has no functional pair fluent
    between entities that is not symmetric, so without location fluents the ``same`` kind could
    not be drawn at all."""
    result = _world(
        "tiny_full",
        pair_fluents={"same_rules": 0, "literal_kinds": {"pair": 0, "same": 1, "some": 0}},
        event_types={"preconditions": {"pair_share": 1.0, "literals": {2: 1}}},
    )
    location = _location_of(result)
    same = _same_on_location(result)
    fix_ups = result.stats["pair_fluents"]["fix_ups"]
    assert fix_ups["same_location_replacements"] > 0 and fix_ups["same_rules_added"] == 0
    assert same, "the share drew SAME on the location fluent for no two-place event type"
    for et in result.event_types.binary:
        for lit in et.precondition:
            if isinstance(lit, PairLiteral) and lit.pair_fluent == location:
                assert lit.kind == PAIR_SAME and (lit.first, lit.second) == ("agent", "patient")
    for et in result.event_types.unary:
        assert not any(
            isinstance(lit, PairLiteral) and lit.pair_fluent == location for lit in et.precondition
        )
    # no SAME on the location fluent in a condition, and never SOME of an entity on it
    for et in result.event_types.event_types:
        for effect in et.effects:
            for lit in effect.condition:
                if isinstance(lit, PairLiteral):
                    assert lit.pair_fluent != location
    stats = result.stats["pair_fluents"]["precondition_literals"]
    assert stats["by_kind"]["pair_same"] >= len(same)


@SLOW
@pytest.mark.parametrize("seed", SEEDS)
def test_the_quota_holds_on_every_seed_of_default_full(seed: int) -> None:
    """Review point 4: on every seed from 1 to 5 of the default full world, at least
    ``same_rules`` two-place event types require ``SAME`` on the location fluent."""
    result = _world("default_full", seed=seed)
    assert len(_same_on_location(result)) >= result.config.pair_fluents.same_rules == 2
    assert result.stats["pair_fluents"]["fix_ups"]["same_rules_short"] == 0


# ---------------------------------------------------------------------------------------------
# The place_kind key (WM.65) and achievability over every effect (WM.66)
# ---------------------------------------------------------------------------------------------


def test_the_loader_checks_place_kind(tiny_full: WorldResult) -> None:
    record, entities, _ = fixture_inputs(tiny_full.definition)
    places = fixture_places(tiny_full.definition)
    assert runtime_definition(record, entities, places).event_type(
        tiny_full.places.event_types[0].label  # type: ignore[union-attr]
    ).place_kind in ("move", "act")

    def broken(change) -> None:
        data = json.loads(json.dumps(record))
        change(data)
        # the identity is recomputed, so that the key's check is reached
        data["rule_set_id"] = rule_set_id(
            data["symbols"], data["literals"], data["rules"], data["event_types"]
        )
        with pytest.raises(DefinitionError, match="place_kind|written as"):
            runtime_definition(data, entities, places)

    place = next(e for e in record["event_types"] if e["kind"] == "place_event_type")
    entity = next(e for e in record["event_types"] if e["kind"] == "event_type")

    def missing(data):
        del next(e for e in data["event_types"] if e["label"] == place["label"])["place_kind"]

    def wrong(data):
        entry = next(e for e in data["event_types"] if e["label"] == place["label"])
        entry["place_kind"] = "act" if entry["place_kind"] == "move" else "move"

    def unknown(data):
        next(e for e in data["event_types"] if e["label"] == place["label"])["place_kind"] = "go"

    def on_an_entity(data):
        next(e for e in data["event_types"] if e["label"] == entity["label"])["place_kind"] = "move"

    def without_dynamics(data):
        entry = next(e for e in data["event_types"] if e["label"] == place["label"])
        entry["precondition"] = {"literals": [], "expression": "TRUE"}
        entry["effects"] = []

    for change in (missing, wrong, unknown, on_an_entity, without_dynamics):
        broken(change)
    # the key is hashed: the identity moves with it
    changed = json.loads(json.dumps(record))
    wrong(changed)
    assert rule_set_id(
        changed["symbols"], changed["literals"], changed["rules"], changed["event_types"]
    ) != rule_set_id(record["symbols"], record["literals"], record["rules"], record["event_types"])


def test_an_entity_entry_may_need_what_only_an_act_produces(tmp_path: Path) -> None:
    """WM.66: an explicit entity entry whose literal only an act's effect can make true is
    accepted, and refused without the act. Every entity event type's effects are given by the
    file, so that no drawn effect produces the value."""
    base = yaml.safe_load((DATA / "tiny_full.yaml").read_text(encoding="utf-8"))
    base["event_types"]["preconditions"] = {"pair_share": 0}
    base["event_types"]["effects"] = {
        "pair_share": 0,
        "conditional_share": 0,
        "condition_kinds": {"feature": 0.4, "fluent": 0.4, "numeric": 0.2, "pair": 0},
    }
    base["pair_fluents"] = {"count": 2, "locations": 1, "same_rules": 0}
    labels = [f"EVENTTYPE1.{i}" for i in range(1, 5)] + [
        f"EVENTTYPE2.{i}.{j}" for i in (1, 2) for j in (1, 2)
    ]
    event_types: dict[str, Any] = {
        label: {"precondition": "NOT agent.BOOLFL.2", "effects": ["agent.BOOLFL.2 := 1"]}
        for label in labels
    }
    event_types["EVENTTYPE1.1"] = {
        "precondition": "NOT agent.BOOLFL.1",
        "effects": ["agent.BOOLFL.2 := 1"],
    }
    file: dict[str, Any] = {
        "fluents": {"BOOLFL.1": {"initial_rate": 1.0}, "BOOLFL.2": {"initial_rate": 0.5}},
        "event_types": event_types,
        "place_event_types": {
            "PLACEEVENTTYPE.1": {"kind": "act", "effects": ["agent.BOOLFL.1 := 0"]},
        },
    }
    with_act = tmp_path / "with_act.yaml"
    with_act.write_text(yaml.safe_dump(file), encoding="utf-8")
    base["event_types"]["event_file"] = str(with_act)
    result = define(config_from_mapping(base, source=str(tmp_path / "world.yaml")))
    assert any(
        not lit.pair and lit.fluent == "BOOLFL.1" and not lit.value
        for lit in result.event_types.event_type("EVENTTYPE1.1").precondition
    )
    del file["place_event_types"]
    without = tmp_path / "without_act.yaml"
    without.write_text(yaml.safe_dump(file), encoding="utf-8")
    base["event_types"]["event_file"] = str(without)
    with pytest.raises(ConfigError, match="not achievable"):
        define(config_from_mapping(base, source=str(tmp_path / "world.yaml")))


def test_redrawing_a_place_event_types_literals_keeps_its_kind_effects_and_literal(
    tiny_full: WorldResult,
) -> None:
    """The redraw of a never-legal place event type (from ``world:place_dynamics:<label>``)
    changes its further literals alone: the kind, the location fluent, the location literal,
    the effects, and the other place event types stay."""
    places = tiny_full.places
    assert places is not None
    label = places.event_types[0].label
    streams = WorldStreams(tiny_full.config.seed)
    present = _present(tiny_full)
    redrawn = redraw_place_preconditions(
        places,
        [label],
        tiny_full.config,
        {label: streams.part("place_dynamics", label)},
        tiny_full.fluents,
        tiny_full.numeric,
        tiny_full.pair_fluents,
        tiny_full.taxonomy.features,
        tiny_full.taxonomy.instances.values,
        present,
        [e for et in tiny_full.event_types.event_types for e in et.effects],
    )
    before, after = places.event_type(label), redrawn.event_type(label)
    assert after.dynamics_kind == before.dynamics_kind and after.location == before.location
    assert after.effects == before.effects and after.precondition[-1] == before.precondition[-1]
    assert redrawn.event_types[1:] == places.event_types[1:]
    for lit in after.precondition[:-1]:
        assert lit.role == "agent" or (isinstance(lit, PairLiteral) and lit.first == "patient")
    check_place_dynamics(
        redrawn,
        tiny_full.fluents,
        present,
        [e for et in tiny_full.event_types.event_types for e in et.effects],
        set(),
        tiny_full.numeric,
        tiny_full.pair_fluents,
    )


# ---------------------------------------------------------------------------------------------
# The acceptance checks of stage c3a, over the full worlds
# ---------------------------------------------------------------------------------------------


def test_legal_and_derive_agree_with_brute_force_in_random_states(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261121)
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
    for et in runtime.event_types:
        if et.place:
            assert seen[et.label] > 0, et.label


def test_applying_together_equals_one_at_a_time_in_every_order(
    runtime: RuntimeDefinition, brute: BruteForce
) -> None:
    rng = np.random.default_rng(20261122)
    sets = moves = acts = 0
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
        acts += sum(1 for e in chosen if runtime.event_type(e.event_type).act)
        sets += 1
    assert moves + acts > 0


def _check_episodes(result: WorldResult, seed: int, count: int) -> dict[str, int]:
    """Every acceptance property of stage c3a's episodes over ``count`` episodes of the world
    under ``keep`` and under ``redraw``: every entity at exactly one place at every time point,
    every move to the bound place (one of the episode's places, not the agent's own) and
    nothing else of the location fluent, every act at the agent's place, two changes per move,
    replay, and the round trip of the history."""
    location = _location_of(result)
    counts = {"moves": 0, "acts": 0, "changes": 0, "episodes": 0}
    for initial in ("keep", "redraw"):
        generator = _generator(result, initial=initial, size=(2, 5), steps=(4, 8))
        definition = generator.definition
        k = definition.pair_index(location)
        for history in generator.run(seed, count, STATS_STREAM):
            counts["episodes"] += 1
            assert history.places is not None and history.initial_places is not None
            states = replay(definition, history)
            for state in states:
                for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True):
                    assert profile_violations(fluent, matrix) == []
            participants = [definition.entity_index(p) for p in history.participants]
            episode_places = {definition.place_index(q) for q in history.places}
            for p in participants:
                assert _place_of(definition, states[0], p, location) in episode_places
            for step, before, after in zip(history.steps, states[:-1], states[1:], strict=True):
                for event in step.events:
                    et = definition.event_type(event.type)
                    agent = definition.entity_index(event.agent)
                    if et.move:
                        counts["moves"] += 1
                        target = definition.place_index(event.patient)  # type: ignore[arg-type]
                        assert target in episode_places
                        assert _place_of(definition, before, agent, location) != target
                        assert _place_of(definition, after, agent, location) == target
                        movers = {
                            definition.entity_index(e.agent)
                            for e in step.events
                            if definition.event_type(e.type).move
                        }
                        others = [p for p in range(definition.entity_count) if p not in movers]
                        assert np.array_equal(before.pairs[k][others], after.pairs[k][others])
                        changes = [c for c in event.changes if c.pair]
                        assert [(c.second, c.to) for c in changes] == [  # type: ignore[union-attr]
                            (
                                definition.place_labels[
                                    _place_of(definition, before, agent, location)
                                ],
                                False,
                            ),
                            (event.patient, True),
                        ]
                        counts["changes"] += len(changes)
                    elif et.act:
                        counts["acts"] += 1
                        assert definition.place_index(event.patient) == _place_of(  # type: ignore[arg-type]
                            definition, before, agent, location
                        )
                        # an act moves nobody; another event of the step may be a move
                        assert np.array_equal(before.pairs[k][agent], after.pairs[k][agent])
    return counts


def test_episodes_of_tiny_full_keep_every_entity_at_one_place(tiny_full: WorldResult) -> None:
    counts = _check_episodes(tiny_full, 5, EPISODES // 2)
    assert counts["moves"] > 0 and counts["changes"] == 2 * counts["moves"]
    if any(et.act for et in tiny_full.places.event_types):  # type: ignore[union-attr]
        assert counts["acts"] > 0


@SLOW
def test_episodes_of_default_full_keep_every_entity_at_one_place(default_full: WorldResult) -> None:
    counts = _check_episodes(default_full, 5, EPISODES // 2)
    assert counts["moves"] > 0 and counts["acts"] > 0
    assert counts["changes"] == 2 * counts["moves"]


def test_the_statistics_episodes_break_no_profile_and_report_the_shares(
    tiny_full: WorldResult,
) -> None:
    """The 1,000 statistics episodes of ``define`` (which would have raised on a broken
    profile) report the move share, the act share, and the two-place share between entities,
    which sum to the two-place share counting place event types as two-place."""
    episodes = tiny_full.stats["episodes"]
    assert episodes["count"] == EPISODES
    assert math.isclose(
        episodes["move_share"] + episodes["act_share"] + episodes["two_place_share_entities"],
        episodes["two_place_share"],
        abs_tol=3e-6,
    )
    assert episodes["move_share"] > 0
    generator = _generator(tiny_full)
    definition = generator.definition
    for history in generator.run(tiny_full.config.seed, EPISODES, STATS_STREAM):
        for state in replay(definition, history):
            for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True):
                assert profile_violations(fluent, matrix) == []


def test_generated_fixtures_of_tiny_full_pass_both_checkers(tiny_full: WorldResult) -> None:
    fixtures = world_fixtures(tiny_full.definition, "tiny_full")
    assert len(fixtures) == 6
    moves = 0
    for fixture in fixtures:
        assert fixture["initial_places"] and fixture["rule_set_id"] == tiny_full.rule_set_id
        for entry in fixture["definition"]["event_types"]:
            assert (entry["kind"] == "place_event_type") == ("place_kind" in entry)
        for step in fixture["steps"]:
            moves += sum(
                1
                for e in step["events"]
                if tiny_full.definition.runtime().event_type(e["event_type"]).move
            )
        check_fixture(fixture)
        check_fixture_brute(fixture)
    assert moves > 0


# ---------------------------------------------------------------------------------------------
# The location statistics and the habitats
# ---------------------------------------------------------------------------------------------


def test_the_location_statistics_are_written_only_with_location_fluents(
    tiny_full: WorldResult,
) -> None:
    stats = tiny_full.stats
    location = _location_of(tiny_full)
    places, pairs, episodes = stats["places"], stats["pair_fluents"], stats["episodes"]
    assert pairs["between_entities"] == 2 and pairs["locations"] == 1 and pairs["count"] == 3
    place_block = pairs["place_event_types"]
    assert place_block["compiled_literals"]["total"] == len(tiny_full.places.event_types)  # type: ignore[union-attr]
    assert (
        place_block["effects"]["by_kind"]["location"]
        == places["place_event_types_by_kind"]["moves"]
    )
    habitats = places["habitats"][location]
    assert set(habitats) == {
        "by_place_category",
        "largest_deviation",
        "entropy_by_category",
        "mean_entropy_by_level",
        "max_entropy",
    }
    assert math.isclose(habitats["max_entropy"], math.log2(tiny_full.places.count))  # type: ignore[union-attr]
    for row in habitats["by_place_category"].values():
        assert set(row) == {c.label for c in tiny_full.taxonomy.tree.at_level(1)}
        assert all(0.0 <= share <= 1.0 for share in row.values())
    for bits in habitats["entropy_by_category"].values():
        assert 0.0 <= bits <= habitats["max_entropy"] + 1e-9
    # the shares of the places at the first level sum to 1 for each superordinate
    first_level = [c.label for c in tiny_full.places.taxonomy.tree.at_level(1)]  # type: ignore[union-attr]
    for superordinate in tiny_full.taxonomy.tree.at_level(1):
        total = sum(habitats["by_place_category"][p][superordinate.label] for p in first_level)
        assert math.isclose(total, 1.0, abs_tol=1e-5)
    for key in ("move_share", "act_share", "two_place_share_entities"):
        assert key in episodes
    # none of the keys in a world without location fluents
    off = _world("tiny_full", pair_fluents={"locations": 0})
    assert "place_event_types_by_kind" not in off.stats["places"]
    assert "habitats" not in off.stats["places"] and "dynamics" not in off.stats["places"]
    assert "locations" not in off.stats["pair_fluents"]
    assert "place_event_types" not in off.stats["pair_fluents"]
    assert "move_share" not in off.stats["episodes"]


def _expected_habitat_shares(
    result: WorldResult,
) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, float]]:
    """The expected share of each entity superordinate's entities at each place category, from
    the habitat weights: for each entity, its probability of each place is its weight over the
    sum of its row, and a superordinate's expected share at a place category is the mean of its
    entities' probabilities over the category's places. Returns the shares (place category to
    superordinate to share), the superordinates' sizes, and the uniform shares."""
    assert result.places is not None and result.config.places is not None
    places = result.places
    n, m = len(result.taxonomy.instances), places.count
    weights = habitat_weights(
        places.requirement.values(), (n, m), result.config.places.habitat_strength
    )
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    tree, instances = result.taxonomy.tree, result.taxonomy.instances
    place_tree, place_instances = places.taxonomy.tree, places.taxonomy.instances
    superordinate_of = np.array(
        [tree.ancestor_label(tree.categories[k], 1) for k in instances.leaf_index]
    )
    shares: dict[str, dict[str, float]] = {}
    uniform: dict[str, float] = {}
    sizes = {c.label: int((superordinate_of == c.label).sum()) for c in tree.at_level(1)}
    for place_category in place_tree.categories:
        members = np.array(
            [
                place_tree.ancestor_label(place_tree.categories[k], place_category.level)
                == place_category.label
                for k in place_instances.leaf_index
            ]
        )
        uniform[place_category.label] = float(members.sum()) / m
        row: dict[str, float] = {}
        for superordinate in tree.at_level(1):
            entities = superordinate_of == superordinate.label
            row[superordinate.label] = float(probabilities[entities][:, members].sum(axis=1).mean())
        shares[place_category.label] = row
    return shares, sizes, uniform


FLAT_SEEDS = {1: 0.65, 2: 0.52, 4: 0.60}
"""The seeds of ``default_full`` on which the expected-share criterion does not hold at
``habitat_strength`` 2 with the band, with the largest expected deviation as a fraction of its
margin (the follow-up of October 28, 2026): a finding, recorded in the stage's proposal file,
not tuned further (Jon's instruction); each is a strict expected failure, so that the test
turns when the configuration or the bound changes."""


@SLOW
@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            seed,
            marks=pytest.mark.xfail(
                strict=True,
                reason=f"the largest expected deviation on seed {seed} of default_full is "
                f"{FLAT_SEEDS[seed]:.2f} of its margin (stage c3b follow-up, open question 7)",
            ),
        )
        if seed in FLAT_SEEDS
        else seed
        for seed in SEEDS
    ],
)
def test_the_default_habitats_are_not_flat(seed: int) -> None:
    """Review point 2, under Jon's ruling WM.67 (the follow-up of October 28, 2026): on every
    seed from 1 to 5 of ``default_full``, some place category's expected share of some entity
    superordinate's entities, computed from the habitat weights rather than from one sampled
    draw, differs from the uniform share (the category's places over the places) by more than
    three binomial standard deviations of a uniform draw of that many entities; the bound is
    stated from the configuration (12 places, the superordinates' sizes), never from the run."""
    result = _world("default_full", seed=seed)
    assert result.config.places is not None and result.config.places.habitat_strength == 2
    shares, sizes, uniform = _expected_habitat_shares(result)
    beyond = 0
    largest = 0.0
    for place_category, row in shares.items():
        for superordinate, share in row.items():
            n = sizes[superordinate]
            u = uniform[place_category]
            margin = 3 * math.sqrt(u * (1 - u) / n)
            ratio = abs(share - u) / margin
            largest = max(largest, ratio)
            beyond += ratio > 1
    assert beyond > 0, f"seed {seed}: the largest expected deviation is {largest:.2f} of its margin"
    # the sampled shares of the run are reported beside the expected ones
    location = _location_of(result)
    habitats = result.stats["places"]["habitats"][location]
    assert set(habitats["by_place_category"]) == set(shares)


def test_tiny_and_tiny_full_differ_in_the_phase_c_settings_alone() -> None:
    """``tiny_full.yaml`` is ``tiny.yaml`` plus the ``pair_fluents`` block (WM.63) and, since
    the follow-up of October 28, 2026, the density band and the habitat strength (WM.67)."""
    tiny = load_config(DATA / "tiny.yaml")
    full = load_config(DATA / "tiny_full.yaml")
    assert full.pair_fluents != tiny.pair_fluents
    resolved = full.resolved()
    assert {
        **resolved,
        "name": "tiny",
        "pair_fluents": tiny.pair_fluents.resolved(),
        "places": {**resolved["places"], "habitat_strength": 1.0},
        "place_event_types": {**resolved["place_event_types"], "density": None},
    } == tiny.resolved()
