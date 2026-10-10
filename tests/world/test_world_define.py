"""Stage a2 of docs/specs/WORLD_AND_LANGUAGE.md: the world generator. Configuration, fluents,
event types with preconditions and effects, the event file, the definition, capacities, views,
determinism, and the command line. Since stage a5b the static side (the one-place requirements,
the event tree, the relations, the capacities, and the relation statistics) is the world's own,
drawn from the world's streams, and the tests compare the dynamic side with it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import generate as generate_taxonomy
from semantic_world.taxonomy import load_config as load_taxonomy_config
from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.tree import Role
from semantic_world.world import DerivedError, RuleMatrices, read_json, rule_set_id
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.definition import check_definition
from semantic_world.world.event_file import event_file_from_mapping
from semantic_world.world.event_types import check_dynamics, generate_event_types
from semantic_world.world.fluents import derived_initial_values
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.relation_stats import EVENT_TYPE_STAT_COLUMNS, RELATION_FILES
from semantic_world.world.statics import StaticWorld
from semantic_world.world.streams import STREAM_NAMES, WorldStreams
from semantic_world.world.views import view_columns, write_view

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"


def world_mapping(name: str, **overrides) -> dict:
    data = yaml.safe_load((DATA / f"{name}.yaml").read_text())
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


def static_records(statics: StaticWorld) -> dict:
    """Everything the static side decides, as plain records: the requirements (the taxonomy's
    rules and the one-place requirements), the constraints and relations, the capacities, and
    the relation statistics."""
    relations = statics.relations
    projections = statics.projections
    return {
        "rules": statics.rules.records(),
        "constraints": None if relations is None else relations.records(),
        "relations": None if relations is None else relations.relation_records(),
        "capacities": statics.unary.values.tolist(),
        "projections": None
        if projections is None
        else {k: np.asarray(v).tolist() for k, v in projections.all_columns().items()},
        "relation_stats": None
        if statics.relation_stats is None
        else {name: frame.to_dicts() for name, frame in statics.relation_stats.frames().items()},
    }


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def default() -> WorldResult:
    return define(load_config(DATA / "default.yaml"))


@pytest.fixture(scope="module")
def chain() -> WorldResult:
    return define(load_config(DATA / "tiny_chain.yaml"))


@pytest.fixture(scope="module")
def tiny_folder(tiny: WorldResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.write(tmp_path_factory.mktemp("world") / "tiny")


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_default_configuration_lists_every_default() -> None:
    config = load_config(DATA / "default.yaml")
    assert config.fluents.count == 8 and config.fluents.derived_count == 2
    assert config.event_types.preconditions.literals == {0: 0.3, 1: 0.5, 2: 0.2}
    assert config.event_types.effects.roles == {"agent": 0.4, "patient": 0.6}
    assert config.event_types.unary.count == 20
    assert config.event_types.binary is not None
    assert config.event_types.binary.feature_count == 12
    assert config.event_types.binary.density is not None
    assert config.event_types.binary.density.resolved() == {
        "min": 0.0,
        "max": 1.0,
        "max_tries": 200,
    }
    assert config.event_types.binary.constraint_min_density is None
    # The file shows every default but five settings of the event types: the event-type tree
    # (stage a6: the default world has 20 two-place event types, as many as its one-place ones;
    # the code's default is 3 superordinates with 2 to 3 each), and the retune of stage a7a (no
    # own constraint, the key-lock family at half weight, the constraints' operator mix weighing
    # OR 4, and the one-place requirements' operator mix weighing AND 3).
    resolved = config.resolved()
    binary = resolved["event_types"]["binary"]
    assert binary["taxonomy"]["superordinates"] == 5
    assert config.event_types.binary.taxonomy.depth == 2
    assert binary["own_constraint"] is False
    assert binary["constraint_families"]["key_lock"] == 0.5
    assert binary["rules"]["operator_mix"] == {"AND": 1, "OR": 4, "XOR": 1}
    assert resolved["event_types"]["unary"]["rules"]["operator_mix"] == {
        "AND": 3,
        "OR": 1,
        "XOR": 1,
    }
    defaults = config_from_mapping({}).resolved()
    assert defaults["event_types"]["binary"]["taxonomy"]["superordinates"] == 3
    assert defaults["event_types"]["binary"]["own_constraint"] is True
    assert defaults["event_types"]["binary"]["constraint_families"]["key_lock"] == 1
    assert defaults["event_types"]["binary"]["rules"]["operator_mix"] == {
        "AND": 1,
        "OR": 1,
        "XOR": 1,
    }
    assert defaults["event_types"]["unary"]["rules"]["operator_mix"] == {
        "AND": 1,
        "OR": 1,
        "XOR": 1,
    }
    for key in ("taxonomy", "own_constraint", "constraint_families", "rules"):
        binary[key] = defaults["event_types"]["binary"][key]
    resolved["event_types"]["unary"]["rules"] = defaults["event_types"]["unary"]["rules"]
    assert defaults == {**resolved, "taxonomy": {**resolved["taxonomy"]}}


def test_unknown_keys_and_bad_values_name_the_file_and_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("fluents: {count: 4, colour: 1}\n")
    with pytest.raises(ConfigError, match=r"fluents.colour"):
        load_config(path)
    path.write_text("event_types: {preconditions: {enabled_share: 2}}\n")
    with pytest.raises(ConfigError, match=r"preconditions.enabled_share"):
        load_config(path)
    path.write_text("event_types: {binary: {projections: {expose_agent: 1}}}\n")
    with pytest.raises(ConfigError, match=r"binary.projections.*views"):
        load_config(path)
    path.write_text("event_types: {unary: {count: 2, rules: {arity: {99: 1.0}}}}\n")
    with pytest.raises(ConfigError, match=r"unary.rules.arity"):
        load_config(path)


def test_unary_and_binary_blocks_are_the_worlds_own(tmp_path: Path) -> None:
    """The event-type settings are read against the taxonomy configuration but never merged
    into it: the taxonomy the world generates from is the taxonomy file as it stands."""
    config = config_from_mapping(
        world_mapping(
            "tiny",
            event_types={"unary": {"count": 2}, "binary": {"features": {"count": 3}}},
        )
    )
    assert config.event_types.unary.count == 2
    assert config.event_types.unary.labels == ("EVENTTYPE1.1", "EVENTTYPE1.2")
    binary = config.event_types.binary
    assert binary is not None and binary.feature_count == 3
    assert binary.feature_labels == ("EVENTFEAT.1", "EVENTFEAT.2", "EVENTFEAT.3")
    assert binary.taxonomy.superordinates == 3  # the default: tiny.yaml's block was replaced
    standalone = load_taxonomy_config(TAXONOMY_DATA / "tiny_relations.yaml")
    assert config.taxonomy_config().resolved() == standalone.resolved()
    off = config_from_mapping(world_mapping("tiny", event_types={"binary": None}))
    assert off.event_types.binary is None
    # The old taxonomy keys name their new place in the world configuration.
    taxonomy_file = tmp_path / "old.yaml"
    for block, place in (
        ("features: {can: {count: 2}}", "event_types.unary"),
        ("verbs: {}", "event_types.binary"),
    ):
        taxonomy_file.write_text(block + "\n")
        with pytest.raises(ConfigError, match=place):
            config_from_mapping({"taxonomy": {"config": str(taxonomy_file)}})


def test_resolved_configuration_loads_back(tiny: WorldResult) -> None:
    again = config_from_mapping(tiny.config.resolved())
    assert again.resolved() == tiny.config.resolved()


# ---------------------------------------------------------------------------------------------
# Fluents
# ---------------------------------------------------------------------------------------------


def test_fluents_have_rates_values_and_derived_rules(default: WorldResult) -> None:
    fluents = default.fluents
    assert [f.label for f in fluents.fluents] == [f"BOOLFL.{i}" for i in range(1, 9)]
    assert len(fluents.base) == 6 and len(fluents.derived) == 2
    assert set(fluents.initial_rates) <= {0.0, 0.2, 0.8, 1.0}
    assert fluents.initial_values.shape == (len(default.taxonomy.instances), 6)
    for rule in fluents.rules:
        assert any(key.startswith("BOOLFL.") for key in rule.inputs), rule.output
        assert fluents[rule.output].derived
    for fluent in fluents.base:
        rate = fluent.initial_rate
        column = fluents.initial_values[:, fluents.base_labels.index(fluent.label)]
        if rate == 0.0:
            assert not column.any()
        if rate == 1.0:
            assert column.all()


def test_static_rules_never_read_fluents(default: WorldResult) -> None:
    for rule in default.definition.entity_rules:
        if not rule.output.startswith("BOOLFL."):
            assert not any(k.startswith("BOOLFL.") for k in rule.inputs), rule.output


# ---------------------------------------------------------------------------------------------
# Event types, preconditions, and effects
# ---------------------------------------------------------------------------------------------


def test_event_types_come_from_the_requirements_and_the_event_tree(default: WorldResult) -> None:
    statics = default.statics
    event_types = default.event_types
    assert statics.unary.labels == tuple(f"EVENTTYPE1.{i}" for i in range(1, 21))
    assert [e.label for e in event_types.unary] == list(statics.unary.labels)
    assert statics.event_tree is not None and statics.relations is not None
    assert [e.label for e in event_types.binary] == [
        v.label for v in statics.event_tree.event_types
    ]
    for et in event_types.unary:
        (constraint,) = et.constraints
        spec = event_types.constraint(constraint)
        rule = statics.unary.rule_for(et.label)
        assert spec.table == rule.table
        assert all(k.startswith("agent.") for k in spec.inputs)
    for et in event_types.binary:
        relation = statics.relations.relation(et.label)
        assert not relation.base
        assert list(et.constraints) == [c.label for c in relation.constraints]
    for et in event_types.categories:
        relation = statics.relations.relation(et.label)
        assert relation.base and list(et.constraints) == [c.label for c in relation.constraints]


def test_requirements_never_read_fluents(default: WorldResult) -> None:
    for spec in default.event_types.constraints:
        assert not any("BOOLFL" in k for k in spec.inputs), spec.label


def test_every_rule_of_preconditions_and_effects_holds(default: WorldResult) -> None:
    fluents = default.fluents
    initial_present = {
        (f.label, bool(v))
        for i, f in enumerate(fluents.base)
        for v in np.unique(fluents.initial_values[:, i])
    }
    for label, column in derived_initial_values(fluents, default.taxonomy).items():
        initial_present |= {(label, bool(v)) for v in np.unique(column)}
    check_dynamics(default.event_types, fluents, initial_present)
    leaves = default.event_types.leaves
    assert sum(len(e.effects) for e in leaves) > 0
    assert sum(len(e.precondition) for e in leaves) > 0
    for et in default.event_types.unary:
        assert all(x.role == "agent" for x in (*et.precondition, *et.effects))


def test_event_type_features_share_preconditions_and_effects(default: WorldResult) -> None:
    """Every event type inherits the effects and literals of its true features, except those the
    fix-ups dropped, which the statistics count."""
    event_types = default.event_types
    stats = event_types.stats
    inherited_effects = kept_effects = 0
    inherited_literals = kept_literals = 0
    for et in event_types.event_types:
        if et.explicit:
            continue
        for feature in et.features:
            effect = event_types.feature_effects.get(feature)
            if effect is not None:
                inherited_effects += 1
                # stage b1: an inherited effect brings its condition with it
                kept_effects += effect in et.effects
            literal = event_types.feature_preconditions.get(feature)
            if literal is not None:
                inherited_literals += 1
                kept_literals += literal in et.precondition
    assert inherited_effects > 0 and kept_effects > 0
    assert inherited_effects - kept_effects == (
        stats["effects_dropped_contradiction"]
        + stats["effects_dropped_duplicate"]
        + stats["effects_dropped_empty"]
    )
    assert inherited_literals - kept_literals == (
        stats["literals_dropped_contradiction"]
        + stats["literals_dropped_duplicate"]
        + stats["literals_dropped_unachievable"]
    )


def test_fluents_off_gives_no_dynamics_and_the_same_static_side(tiny: WorldResult) -> None:
    """Without fluents, no event type has a precondition or an effect, and the static side is
    the same as with them: the requirements, the capacities, and the relation statistics come
    from their own streams."""
    config = config_from_mapping(world_mapping("tiny", fluents={"count": 0}))
    world = define(config)
    assert len(world.fluents) == 0
    for et in world.event_types.event_types:
        assert et.precondition == () and et.effects == ()
    assert static_records(world.statics) == static_records(tiny.statics)
    assert world.stats["effects"]["total"] == 0 and world.stats["enabling_graph"]["edges"] == 0
    assert world.stats["episodes"]["precondition_redraws"] == {}


def test_never_legal_event_types_have_their_preconditions_redrawn(tiny: WorldResult) -> None:
    """An event type never legal in the statistics episodes has its own precondition literals
    drawn again from its own part of ``world:preconditions``, as many times as the statistics
    count, and no other event type's preconditions change. The tiny world at seed 1 redraws
    three event types, after which every event type is legal somewhere."""
    from semantic_world.world.event_types import redraw_preconditions

    episodes = tiny.stats["episodes"]
    redraws = episodes["precondition_redraws"]
    assert redraws and set(redraws) == {"EVENTTYPE1.4", "EVENTTYPE2.1.2", "EVENTTYPE2.2.1"}
    assert all(1 <= count <= 5 for count in redraws.values())
    assert episodes["never_legal"] == {} and episodes["warnings"] == []
    assert all(episodes["legal_share"][label] > 0 for label in redraws)
    # The first draw, before the statistics episodes.
    streams = WorldStreams(tiny.config.seed)
    derived_initial = derived_initial_values(tiny.fluents, tiny.taxonomy)
    first = generate_event_types(
        tiny.statics, tiny.fluents, tiny.config, streams, tiny.event_file, derived_initial
    )
    assert first.features == tiny.event_types.features
    assert first.feature_preconditions == tiny.event_types.feature_preconditions
    assert first.feature_effects == tiny.event_types.feature_effects
    for et in first.event_types:
        final = tiny.event_types.event_type(et.label)
        assert et.effects == final.effects, et.label
        if et.label not in redraws:
            assert et.precondition == final.precondition, et.label
    # Each redrawn event type is reproduced by its own part alone, applied as many times as
    # counted, whatever the other event types did.
    for label, count in redraws.items():
        parts = {label: streams.part("preconditions", label)}
        again = first
        for _ in range(count):
            again = redraw_preconditions(
                again, [label], tiny.fluents, tiny.config, parts, derived_initial
            )
        assert (
            again.event_type(label).precondition == tiny.event_types.event_type(label).precondition
        )
        inherited = [
            tiny.event_types.feature_preconditions[f]
            for f in tiny.event_types.event_type(label).features
            if tiny.event_types.feature_preconditions.get(f) is not None
        ]
        for literal in inherited:
            assert literal in again.event_type(label).precondition


def test_a_redraw_never_reintroduces_an_unachievable_inherited_literal() -> None:
    """Stage a6: the generator's fix-ups drop an inherited literal from an event type when the
    effect that produced its value goes, and a later redraw of the event type's preconditions
    (``define``, for an event type never legal in the statistics episodes) rebuilds the inherited
    literals from the feature literals, so it must leave such a literal out. On the default world
    of seed 1 the redraw of EVENTTYPE2.3.1 reintroduced ``NOT patient.BOOLFL.3`` before the fix."""
    from semantic_world.world.event_types import (
        check_dynamics,
        generate_event_types,
        redraw_preconditions,
    )
    from semantic_world.world.fluents import derived_initial_values, generate_fluents
    from semantic_world.world.generate import _initial_present
    from semantic_world.world.statics import build_statics
    from semantic_world.world.streams import WorldStreams

    config = load_config(DATA / "default.yaml")
    taxonomy = generate_taxonomy(config.taxonomy_config())
    streams = WorldStreams(config.seed)
    statics = build_statics(taxonomy, config, streams, None)
    fluents = generate_fluents(config, taxonomy, streams)
    derived_initial = derived_initial_values(fluents, taxonomy)
    event_types = generate_event_types(statics, fluents, config, streams, None, derived_initial)
    initial_present = _initial_present(fluents, derived_initial)
    check_dynamics(event_types, fluents, initial_present)
    labels = [et.label for et in event_types.event_types]
    parts = {label: streams.part("preconditions", label) for label in labels}
    redrawn = redraw_preconditions(event_types, labels, fluents, config, parts, derived_initial)
    check_dynamics(redrawn, fluents, initial_present)


def test_never_legal_reports_a_reason(default: WorldResult, chain: WorldResult) -> None:
    for world in (default, chain):
        episodes = world.stats["episodes"]
        assert set(episodes["never_legal"].values()) <= {"never_able", "preconditions"}
        assert len(episodes["warnings"]) == len(episodes["never_legal"])
        for label, reason in episodes["never_legal"].items():
            assert episodes["legal_share"][label] == 0
            assert reason in episodes["warnings"][list(episodes["never_legal"]).index(label)]
        assert all(count >= 1 for count in episodes["precondition_redraws"].values())
        assert not set(episodes["precondition_redraws"]) & {
            label for label, reason in episodes["never_legal"].items() if reason == "never_able"
        }


# ---------------------------------------------------------------------------------------------
# The event file
# ---------------------------------------------------------------------------------------------


def test_chain_example_loads(chain: WorldResult) -> None:
    from semantic_world.world.stats import enabling_edges

    catch = chain.event_types.event_type("EVENTTYPE2.1.1")
    eat = chain.event_types.event_type("EVENTTYPE2.1.2")
    assert catch.explicit and eat.explicit
    assert [lit.text for lit in catch.precondition] == ["agent.BOOLFL.1", "NOT patient.BOOLFL.2"]
    assert [e.text for e in catch.effects] == ["patient.BOOLFL.2 := 1"]
    assert [e.text for e in eat.effects] == ["patient.BOOLFL.1 := 0"]
    assert chain.fluents["BOOLFL.1"].initial_rate == 1.0
    assert chain.fluents["BOOLFL.2"].initial_rate == 0.0
    assert not chain.fluents["BOOLFL.1"].derived and not chain.fluents["BOOLFL.2"].derived
    assert ("EVENTTYPE2.1.1", "EVENTTYPE2.1.2") in enabling_edges(chain.event_types)
    relations = chain.statics.relations
    assert relations is not None
    assert relations.explicit == {"EVENTTYPE2.1.1", "EVENTTYPE2.1.2"}
    for label in ("EVENTTYPE2.1.1", "EVENTTYPE2.1.2"):
        own = relations.own_constraints[label]
        assert own.family == "explicit"
        assert str(own.expression) == "agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.0000"
        spec = chain.event_types.constraint(f"CONSTRAINT.{label}")
        assert spec.expression == "agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.0000"


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            {"EVENTTYPE2.9.9": {"effects": ["patient.BOOLFL.1 := 1"]}},
            r"event_types.EVENTTYPE2.9.9.*unknown",
        ),
        ({"EVENTTYPE2.1.1": {"effects": ["patient.BOOLFL.4 := 1"]}}, r"effects\[0\].*derived"),
        (
            {"EVENTTYPE2.1.1": {"effects": ["patient.BOOLFL.1 := 1", "patient.BOOLFL.1 := 0"]}},
            r"effects\[1\].*contradicts",
        ),
        (
            {"EVENTTYPE2.1.1": {"precondition": "agent.BOOLFL.1 AND NOT agent.BOOLFL.1"}},
            r"precondition.*negation",
        ),
        (
            {"EVENTTYPE2.1.1": {"precondition": "agent.BOOLFL.1 OR agent.BOOLFL.2"}},
            r"precondition.*conjunction",
        ),
        ({"EVENTTYPE1.1": {"effects": ["patient.BOOLFL.1 := 1"]}}, r"effects\[0\].*role"),
        (
            {
                "EVENTTYPE2.1.1": {
                    "precondition": "agent.BOOLFL.1",
                    "effects": ["agent.BOOLFL.1 := 1"],
                }
            },
            r"changes nothing",
        ),
        ({"EVENTTYPE2.1.1": {"requirement": "agent.BOOLFL.1"}}, r"requirement.*fluent"),
        ({"EVENTTYPE2.1.1": {"requirement": "agent.PROPERTY.99"}}, r"requirement.*unknown feature"),
        ({"EVENTTYPE1.1": {"requirement": "agent.EVENTTYPE1.2"}}, r"requirement.*event type"),
        ({"EVENTTYPE1.9": {"requirement": "agent.PROPERTY.1"}}, r"EVENTTYPE1.9.*unknown"),
    ],
)
def test_a_broken_event_file_names_the_file_and_the_entry(
    tmp_path: Path, entry: dict, match: str
) -> None:
    file = tmp_path / "events.yaml"
    file.write_text(
        yaml.safe_dump({"fluents": {"BOOLFL.1": {"initial_rate": 1.0}}, "event_types": entry})
    )
    config = config_from_mapping(
        world_mapping("tiny", event_types={"event_file": str(file)}),
        source=str(tmp_path / "w.yaml"),
    )
    with pytest.raises(ConfigError, match=match) as info:
        define(config)
    assert str(file) in str(info.value)


def test_unknown_fluent_in_the_event_file_is_an_error(tmp_path: Path) -> None:
    file = tmp_path / "events.yaml"
    file.write_text("fluents: {BOOLFL.9: {initial_rate: 1.0}}\n")
    config = config_from_mapping(world_mapping("tiny", event_types={"event_file": str(file)}))
    with pytest.raises(ConfigError, match=r"fluents.BOOLFL.9.*unknown fluent"):
        define(config)


def test_malformed_event_file_entries_are_errors() -> None:
    with pytest.raises(ConfigError, match="EVENTTYPE"):
        event_file_from_mapping({"event_types": {"VERB.1": {}}})
    with pytest.raises(ConfigError, match=r"effects\[0\]"):
        event_file_from_mapping({"event_types": {"EVENTTYPE1.1": {"effects": ["BOOLFL.1 = 1"]}}})
    with pytest.raises(ConfigError, match="initial_rate"):
        event_file_from_mapping({"fluents": {"BOOLFL.1": {}}})


def test_explicit_requirements_replace_the_sampled_ones(tiny: WorldResult, tmp_path: Path) -> None:
    """An explicit requirement replaces a one-place event type's sampled rule, or a two-place
    event type's own constraint, before the capacities are computed; every other event type
    keeps the draw it has without the event file."""
    file = tmp_path / "events.yaml"
    file.write_text(
        yaml.safe_dump(
            {
                "event_types": {
                    "EVENTTYPE1.1": {"requirement": "agent.PROPERTY.1 AND NOT agent.PART.2"},
                    "EVENTTYPE2.1.1": {
                        "requirement": "agent.PROPERTY.1 AND patient.SCALARDIM.1 > 0.5"
                    },
                }
            }
        )
    )
    config = config_from_mapping(world_mapping("tiny", event_types={"event_file": str(file)}))
    world = define(config)
    statics = world.statics
    values, features = statics.values, statics.features
    expected = values[:, features["PROPERTY.1"].position] & (
        1 - values[:, features["PART.2"].position]
    )
    assert np.array_equal(statics.unary.column("EVENTTYPE1.1"), expected)
    rule = statics.unary.rule_for("EVENTTYPE1.1")
    assert rule.family == "explicit" and str(rule.expression) == "PROPERTY.1 AND NOT PART.2"
    capacities = world.derived_frames()["capacities.csv"]
    assert capacities["CAN.EVENTTYPE1.1"].to_list() == expected.astype(int).tolist()
    relations = statics.relations
    assert relations is not None and relations.explicit == {"EVENTTYPE2.1.1"}
    own = relations.own_constraints["EVENTTYPE2.1.1"]
    assert own.family == "explicit"
    assert str(own.expression) == "agent.PROPERTY.1 AND patient.SCALARDIM.1 > 0.5000"
    spec = world.event_types.constraint("CONSTRAINT.EVENTTYPE2.1.1")
    assert spec.expression == "agent.PROPERTY.1 AND patient.SCALARDIM.1 > 0.5000"
    # Every other requirement is the draw of the world without the event file, and the
    # taxonomy run is untouched.
    sampled = tiny.statics
    assert [r.record() for r in statics.unary.requirement_rules[1:]] == [
        r.record() for r in sampled.unary.requirement_rules[1:]
    ]
    assert sampled.relations is not None
    assert relations.feature_constraints == sampled.relations.feature_constraints
    for label, constraint in sampled.relations.own_constraints.items():
        if label != "EVENTTYPE2.1.1":
            assert relations.own_constraints[label] == constraint
    assert world.taxonomy.rules.records() == tiny.taxonomy.rules.records()
    assert world.taxonomy.frames()["base.csv"].equals(tiny.taxonomy.frames()["base.csv"])
    check_definition(world.definition)


def test_explicit_requirement_keeps_the_ancestors_base_relations(
    tiny: WorldResult, chain: WorldResult
) -> None:
    """A two-place event type with an explicit requirement keeps the constraints of the
    event-type features that are defining at its ancestors (REL.10), and loses the constraints
    of its other true features together with its sampled own constraint."""
    sampled, explicit = tiny.statics.relations, chain.statics.relations
    assert sampled is not None and explicit is not None
    assert [c.label for c in sampled.event_tree.categories] == [
        c.label for c in explicit.event_tree.categories
    ]
    assert np.array_equal(
        sampled.event_tree.tree.generative_matrix(), explicit.event_tree.tree.generative_matrix()
    )
    dropped = 0
    for label in sorted(explicit.explicit):
        category = explicit.event_tree.tree[label]
        features = explicit.event_tree.features.labels
        defining = {
            features[i]
            for i in np.flatnonzero(
                (category.values == 1) & (category.roles == Role.DEFINING_INHERITED)
            )
        }
        other_true = {features[i] for i in np.flatnonzero(category.values == 1)} - defining
        assert category.parent is not None
        parent = explicit.relation(category.parent)
        assert parent.base and {c.label for c in parent.constraints} == {
            f"CONSTRAINT.{f}" for f in defining
        }
        relation = explicit.relation(label)
        assert [c.label for c in relation.constraints] == [
            *(c.label for c in parent.constraints),
            f"CONSTRAINT.{label}",
        ]
        before = sampled.relation(label)
        assert [c.label for c in before.constraints] == [
            *(f"CONSTRAINT.{f}" for f in features if f in defining | other_true),
            f"CONSTRAINT.{label}",
        ]
        dropped += len(other_true)
        assert chain.event_types.event_type(label).constraints == tuple(
            c.label for c in relation.constraints
        )
    assert dropped > 0, "no explicit event type of the chain has a true non-defining feature"


# ---------------------------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------------------------


def test_definition_json_has_the_spec_keys_and_a_recomputable_identity(
    tiny: WorldResult, tiny_folder: Path
) -> None:
    data = read_json(tiny_folder / "definition.json")
    assert list(data) == [
        "version",
        "rule_set_id",
        "symbols",
        "literals",
        "rules",
        "layers",
        "event_types",
    ]
    assert data["rule_set_id"] == rule_set_id(
        data["symbols"], data["literals"], data["rules"], data["event_types"]
    )
    kinds = {s["kind"] for s in data["symbols"]}
    assert kinds == {"property", "part", "scalar", "fluent", "event_type", "event_type_category"}
    assert all(lit["index"] == i for i, lit in enumerate(data["literals"]))
    for et in data["event_types"]:
        assert et["roles"] == ["agent"][: et["arity"]] + (["patient"] if et["arity"] == 2 else [])
    assert {layer["scope"] for layer in data["layers"]} == {"entity", "binding"}
    rates = {s["label"]: s["initial_rate"] for s in data["symbols"] if s["fluent"]}
    assert rates == {f.label: None if f.derived else f.initial_rate for f in tiny.fluents.fluents}
    assert rates["BOOLFL.4"] is None and all(
        isinstance(rates[f.label], float) for f in tiny.fluents.base
    )
    assert not any("initial_rate" in s for s in data["symbols"] if not s["fluent"])


def test_binding_layers_rebuild_and_agree(tiny: WorldResult, tiny_folder: Path) -> None:
    data = read_json(tiny_folder / "definition.json")
    binding = [layer for layer in data["layers"] if layer["scope"] == "binding"]
    rebuilt = RuleMatrices.from_record(
        data["literals"], [{k: v for k, v in layer.items() if k != "scope"} for layer in binding]
    )
    assert rebuilt.layers == tiny.definition.binding_matrices.layers
    report = check_definition(tiny.definition)
    n = len(tiny.taxonomy.instances)
    assert report.binding.entities == n * n
    assert report.binding.rules == len(tiny.definition.binding_rules)
    assert report.entity.entities == n


def test_requirement_rules_and_the_constraints_in_the_identity(tiny: WorldResult) -> None:
    rules = {r["output"]: r for r in tiny.definition.rules_record()}
    for et in tiny.event_types.event_types:
        rule = rules[et.requirement_label]
        assert rule["scope"] == "binding" and rule["family"] == "requirement"
        assert rule["truth_table"] == "0" * (2 ** len(et.constraints) - 1) + "1"
        for label in et.constraints:
            assert rules[label]["scope"] == "binding"
    base = tiny.definition.rule_set_id
    changed = json.loads(json.dumps(tiny.definition.event_types_record()))
    changed[0]["effects"] = []
    assert (
        rule_set_id(
            tiny.definition.symbols,
            tiny.definition.literals_record(),
            tiny.definition.rules_record(),
            changed,
        )
        != base
    )


def test_identity_changes_with_a_fluent_setting_but_requirements_do_not() -> None:
    a = define(config_from_mapping(world_mapping("tiny", fluents={"count": 4})))
    b = define(config_from_mapping(world_mapping("tiny", fluents={"count": 5})))
    assert a.rule_set_id != b.rule_set_id
    assert [c.record() for c in a.event_types.constraints] == [
        c.record() for c in b.event_types.constraints
    ]
    assert a.statics.rules.records() == b.statics.rules.records()


def test_an_initial_rate_changes_the_identity(tiny: WorldResult) -> None:
    symbols = json.loads(json.dumps(tiny.definition.symbols))
    fluent = next(s for s in symbols if s["fluent"] and not s["derived"])
    fluent["initial_rate"] = 0.5 if fluent["initial_rate"] != 0.5 else 0.6
    assert (
        rule_set_id(
            symbols,
            tiny.definition.literals_record(),
            tiny.definition.rules_record(),
            tiny.definition.event_types_record(),
        )
        != tiny.rule_set_id
    )


def test_entities_csv_holds_base_facts_and_initial_fluents(
    tiny: WorldResult, tiny_folder: Path
) -> None:
    entities = pl.read_csv(tiny_folder / "entities.csv")
    free = [f.label for f in tiny.taxonomy.features.free]
    assert entities.columns == [
        "label",
        "leaf",
        *free,
        "SCALARDIM.1",
        "BOOLFL.1",
        "BOOLFL.2",
        "BOOLFL.3",
    ]
    assert entities["label"][0] == "INSTANCE.1.1.1" and entities["leaf"][0] == "CATEGORY.1.1"
    assert not any(c.startswith(("PROPERTY.7", "PROPERTY.8")) for c in entities.columns)
    base = pl.read_csv(tiny_folder / "taxonomy" / "base.csv")
    assert base.columns == [*entities.columns[: 2 + len(free)], "SCALARDIM.1"]
    assert entities.select(base.columns[:-1]).equals(base.select(base.columns[:-1]))
    assert np.allclose(
        entities["SCALARDIM.1"].to_numpy(), base["SCALARDIM.1"].to_numpy(), atol=1e-6
    )


# ---------------------------------------------------------------------------------------------
# Derived values, views, and the run folder
# ---------------------------------------------------------------------------------------------


def test_capacities_equal_the_static_sides(tiny: WorldResult, tiny_folder: Path) -> None:
    capacities = pl.read_csv(tiny_folder / "derived" / "capacities.csv")
    statics = tiny.statics
    assert statics.projections is not None
    two_place = statics.projections.all_columns()
    assert capacities.columns == [
        "label",
        *(f"CAN.{label}" for label in statics.unary.labels),
        *two_place,
    ]
    assert capacities["label"].to_list() == list(statics.instances.labels)
    for label in statics.unary.labels:
        assert capacities[f"CAN.{label}"].to_list() == statics.unary.column(label).tolist()
        rule = statics.unary.rule_for(label)
        inputs = np.stack(
            [
                i.values(statics.instances.scalars)
                if hasattr(i, "values")
                else statics.values[:, i.position]
                for i in rule.inputs
            ],
            axis=1,
        ).astype(np.uint8)
        assert np.array_equal(rule.table.evaluate(inputs), statics.unary.column(label))
    for column, values in two_place.items():
        if column == "approximate":
            assert capacities[column].null_count() == len(capacities) or set(
                capacities[column].to_list()
            ) == {values}
        else:
            assert capacities[column].to_list() == np.asarray(values).astype(int).tolist(), column
    roles = pl.read_csv(tiny_folder / "derived" / "capacity_roles.csv")
    assert set(roles["status"].to_list()) <= {"fixed_1", "fixed_0", "free"}
    assert len(roles) == len(tiny.taxonomy.tree.categories) * len(statics.unary.labels)
    assert roles["event_type"].to_list()[: len(statics.unary.labels)] == list(statics.unary.labels)


def test_relation_statistics_are_the_static_sides(tiny: WorldResult, tiny_folder: Path) -> None:
    statics = tiny.statics
    assert statics.relations is not None and statics.relation_stats is not None
    event_types = [v.label for v in statics.relations.event_tree.event_types]
    for name in RELATION_FILES:
        assert (tiny_folder / "derived" / name).is_file(), name
    stats = pl.read_csv(tiny_folder / "derived" / "event_type_stats.csv")
    # The default density check is on, so the density columns follow the fixed ones.
    assert stats.columns == [*EVENT_TYPE_STAT_COLUMNS, "leaf_pair_density", "tries"]
    assert stats["event_type"].to_list() == event_types
    assert statics.relations.event_type_density is not None
    assert np.allclose(
        stats["leaf_pair_density"].to_numpy(),
        [statics.relations.event_type_density[label] for label in event_types],
        atol=1e-6,
    )
    assert all(0.0 < d < 1.0 for d in stats["leaf_pair_density"].to_list())
    in_memory = statics.relation_stats.event_type_stats
    for column in ("agents", "patients", "constraints", "families"):
        assert stats[column].to_list() == in_memory[column].to_list(), column
    assert np.allclose(
        stats["proportion_true"].to_numpy(), in_memory["proportion_true"].to_numpy(), atol=1e-6
    )
    pairs = pl.read_csv(tiny_folder / "derived" / "relation_pairs.csv")
    assert pairs.columns == ["event_type", "agent", "patient", "holds"]
    assert set(pairs["event_type"].to_list()) <= set(event_types)
    index = {label: i for i, label in enumerate(statics.instances.labels)}
    for row in pairs.iter_rows(named=True):
        assert row["agent"].startswith("INSTANCE.") and row["patient"].startswith("INSTANCE.")
        holds = statics.relations.holds(
            row["event_type"], [index[row["agent"]]], [index[row["patient"]]]
        )
        assert bool(holds[0]) == bool(row["holds"]), row
    thematic = pl.read_csv(tiny_folder / "derived" / "thematic.csv")
    assert thematic.columns == ["leaf_a", "leaf_b", "thematic", "similarity"]
    assert all(label.startswith("CATEGORY.") for label in thematic["leaf_a"].to_list())


def test_manifest_covers_every_derived_file_and_refuses_another_identity(
    tiny: WorldResult, tiny_folder: Path
) -> None:
    manifest = yaml.safe_load((tiny_folder / "derived" / "manifest.yaml").read_text())
    files = sorted(p.name for p in (tiny_folder / "derived").iterdir() if p.name != "manifest.yaml")
    assert sorted(manifest["files"]) == files
    assert all(entry["rule_set_id"] == tiny.rule_set_id for entry in manifest["files"].values())
    from semantic_world.world import load_derived_csv

    with pytest.raises(DerivedError, match="capacities.csv"):
        load_derived_csv(tiny_folder / "derived", "capacities.csv", "0" * 64)


def test_classic_view_gives_the_old_instances_columns(tiny: WorldResult, tiny_folder: Path) -> None:
    """The classic preset is the old ``instances.csv`` of the taxonomy: ISA columns, every
    PROPERTY and PART feature, the one-place capacities in the place of CAN, and the scalars."""
    path = write_view(tiny_folder, "classic")
    view = pl.read_csv(path)
    statics = tiny.statics
    tree, instances = tiny.taxonomy.tree, statics.instances
    static = sorted(
        (f.label for f in tiny.taxonomy.features.features),
        key=lambda c: (c.split(".")[0] != "PROPERTY", int(c.split(".")[1])),
    )
    assert view.columns == [
        "label",
        "leaf",
        *(f"ISA.{c.label}" for c in tree.categories),
        *static,
        *(f"CAN.{label}" for label in statics.unary.labels),
        "SCALARDIM.1",
    ]
    assert view["label"].to_list() == list(instances.labels)
    assert view["leaf"].to_list() == list(instances.leaf_labels)
    for category in tree.categories:
        members = {category.label} | {d.label for d in category.descendants()}
        assert view[f"ISA.{category.label}"].to_list() == [
            int(leaf in members) for leaf in instances.leaf_labels
        ], category.label
    for label in static:
        position = statics.features[label].position
        assert view[label].to_list() == statics.values[:, position].tolist(), label
    for label in statics.unary.labels:
        assert view[f"CAN.{label}"].to_list() == statics.unary.column(label).tolist()
    assert np.allclose(view["SCALARDIM.1"].to_numpy(), instances.scalars[:, 0], atol=1e-6)
    sidecar = yaml.safe_load(path.with_suffix(".yaml").read_text())
    assert sidecar["preset"] == "classic" and sidecar["columns"] == view.columns
    assert sidecar["rule_set_id"] == read_json(tiny_folder / "definition.json")["rule_set_id"]


def test_base_and_static_presets_and_include(tiny_folder: Path) -> None:
    base = pl.read_csv(write_view(tiny_folder, "base", out=tiny_folder / "views" / "b.csv"))
    assert all(c.startswith(("PROPERTY.", "PART.", "SCALARDIM.")) for c in base.columns[2:])
    assert "PROPERTY.7" not in base.columns
    static = pl.read_csv(write_view(tiny_folder, "static", out=tiny_folder / "views" / "s.csv"))
    assert "PROPERTY.7" in static.columns and not any(c.startswith("CAN.") for c in static.columns)
    narrowed = pl.read_csv(
        write_view(
            tiny_folder, "classic", ["PROPERTY", "CAN.EVENTTYPE1"], tiny_folder / "views" / "n.csv"
        )
    )
    assert all(c.startswith(("PROPERTY.", "CAN.EVENTTYPE1.")) for c in narrowed.columns[2:])
    added = pl.read_csv(
        write_view(tiny_folder, "base", ["CANBE", "BOOLFL"], tiny_folder / "views" / "a.csv")
    )
    assert any(c.startswith("CANBE.") for c in added.columns) and "BOOLFL.1" in added.columns
    from semantic_world.world.errors import WorldError

    with pytest.raises(WorldError, match="no column"):
        view_columns(
            "base",
            ["NOTHING"],
            {
                "base": ["PROPERTY.1"],
                "scalars": [],
                "fluents": [],
                "derived_static": [],
                "capacities": [],
                "isa": [],
            },
        )


def test_taxonomy_folder_is_byte_identical_to_a_standalone_run(
    tiny_folder: Path, tmp_path: Path
) -> None:
    """The world's draws never touch the taxonomy: its folder in a world run is the taxonomy
    run alone with the same configuration and seed, file for file."""
    standalone = generate_taxonomy(
        load_taxonomy_config(TAXONOMY_DATA / "tiny_relations.yaml")
    ).write(tmp_path / "t")
    files = sorted(
        p.relative_to(standalone).as_posix() for p in standalone.rglob("*") if p.is_file()
    )
    assert files == sorted(
        p.relative_to(tiny_folder / "taxonomy").as_posix()
        for p in (tiny_folder / "taxonomy").rglob("*")
        if p.is_file()
    )
    for name in files:
        if name != "config.yaml":
            assert (standalone / name).read_bytes() == (
                tiny_folder / "taxonomy" / name
            ).read_bytes(), name
    assert not any(
        name.startswith(("verb", "relation", "projections", "constraints", "instances"))
        for name in files
    )


def test_same_configuration_and_seed_give_byte_identical_runs(tmp_path: Path) -> None:
    a = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "a")
    b = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "b")
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    for name in files:
        if not name.endswith("config.yaml"):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_dynamics_settings_never_change_the_static_side(tiny: WorldResult) -> None:
    """Fluent, precondition, and effect settings draw from their own streams: changing them
    changes neither the requirements, nor the capacities, nor the relation statistics."""
    reference = static_records(tiny.statics)
    for overrides in (
        {
            "event_types": {
                "effects": {"count": {3: 1.0}},
                "preconditions": {"feature_rate": 1.0},
            }
        },
        {"fluents": {"count": 6, "initial_rates": [0.5]}},
        {"fluents": {"count": 2, "derived_proportion": 0.0}},
    ):
        world = define(config_from_mapping(world_mapping("tiny", **overrides)))
        assert static_records(world.statics) == reference, overrides
        frames = world.derived_frames()
        for name, frame in tiny.derived_frames().items():
            assert frames[name].equals(frame), name
    # And the initial values do not depend on the preconditions or the effects.
    other = define(
        config_from_mapping(world_mapping("tiny", event_types={"effects": {"count": {3: 1.0}}}))
    )
    assert other.fluents.initial_values.tolist() == tiny.fluents.initial_values.tolist()


def test_world_stats_report_the_structure(tiny_folder: Path) -> None:
    stats = yaml.safe_load((tiny_folder / "world_stats.yaml").read_text())
    assert stats["fluents"] == {"base": 3, "derived": 1}
    assert stats["event_types"]["one_place"] == 4 and stats["event_types"]["two_place"] == 4
    assert set(stats["fix_ups"]) >= {"effects_dropped_contradiction", "effects_dropped_empty"}
    assert "longest_chain" in stats["enabling_graph"]
    episodes = stats["episodes"]
    assert episodes["count"] == 1000  # stage a4: the statistics episodes
    assert 0.0 <= episodes["two_place_share"] <= 1.0
    assert isinstance(episodes["precondition_redraws"], dict)
    assert isinstance(episodes["never_legal"], dict)
    assert stats["warnings"] == []
    # Stage a6 (ruling 6 on the a5b questions): the relations block of the old summary.yaml.
    relations = stats["relations"]
    assert relations["event_type_features"] == 4 and relations["event_type_categories"] == 2
    assert relations["event_types"] == 4
    assert relations["constraints"] == sum(relations["constraint_families"].values()) > 0
    assert set(relations["constraint_families"]) <= {
        "agent",
        "patient",
        "cross",
        "key_lock",
        "comparison",
    }
    assert 0.0 <= relations["approximate_projections"] <= 1.0
    assert relations["proportions_estimated"] is False
    assert isinstance(relations["pairs_short"], dict)
    assert relations["outside_density"] == len(relations["outside_density_labels"])


def test_config_yaml_records_the_taxonomy_and_every_seed(tiny_folder: Path) -> None:
    data = yaml.safe_load((tiny_folder / "config.yaml").read_text())
    assert data["taxonomy"]["resolved"]["name"] == "tiny_relations"
    assert "verbs" not in data["taxonomy"]["resolved"]
    assert "can" not in data["taxonomy"]["resolved"]["features"]
    assert data["event_types"]["unary"]["count"] == 4
    assert data["event_types"]["binary"]["features"]["count"] == 4
    seeds = data["provenance"]["stream_seeds"]
    world_streams = [name for name in seeds if name.startswith("world:")]
    assert world_streams == [f"world:{name}" for name in STREAM_NAMES]
    assert world_streams == [
        "world:requirements",
        "world:event_tree",
        "world:constraints",
        "world:pairs",
        "world:fluents",
        "world:initial",
        "world:preconditions",
        "world:effects",
        "world:conditions",
        "world:stats",
        "world:episodes",
    ]
    assert "taxonomy:rules" in seeds
    assert not any(name in seeds for name in ("taxonomy:verb_tree", "taxonomy:constraints"))


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------


def test_define_and_view_from_the_command_line(tmp_path: Path) -> None:
    out = tmp_path / "run"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_world.world",
            "define",
            str(DATA / "tiny.yaml"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith(f"wrote {out}: 12 entities, 4 fluents")
    assert (out / "definition.json").is_file() and (out / "derived" / "capacities.csv").is_file()
    view = subprocess.run(
        [sys.executable, "-m", "semantic_world.world", "view", str(out), "--preset", "classic"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert view.returncode == 0, view.stderr
    assert (out / "views" / "classic.csv").is_file()


def test_command_line_reports_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("fluents: {count: -1}\n")
    run = subprocess.run(
        [sys.executable, "-m", "semantic_world.world", "define", str(bad)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert run.returncode == 1 and run.stderr.startswith("error:") and "fluents.count" in run.stderr
