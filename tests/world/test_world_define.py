"""Stage a2 of docs/specs/WORLD_AND_LANGUAGE.md: the world generator. Configuration, fluents,
event types with preconditions and effects, the event file, the definition, capacities, views,
determinism, and the command line."""

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
from semantic_world.world import DerivedError, RuleMatrices, read_json, rule_set_id
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.definition import check_definition
from semantic_world.world.event_file import event_file_from_mapping
from semantic_world.world.event_types import check_dynamics
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.labels import capacity_label, translate, translate_column
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
    assert config_from_mapping({}).resolved() == {
        **config.resolved(),
        "taxonomy": {**config.resolved()["taxonomy"]},
    }


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


def test_unary_and_binary_blocks_merge_over_the_taxonomy() -> None:
    config = config_from_mapping(
        world_mapping(
            "tiny",
            event_types={"unary": {"count": 2}, "binary": {"features": {"count": 3}}},
        )
    )
    taxonomy = config.taxonomy_config()
    assert taxonomy.features.can.count == 2
    assert taxonomy.verbs is not None and taxonomy.verbs.feature_count == 3
    assert taxonomy.verbs.taxonomy.superordinates == 2  # kept from tiny_relations.yaml
    off = config_from_mapping(world_mapping("tiny", event_types={"binary": None}))
    assert off.taxonomy_config().verbs is None


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


def test_event_types_come_from_can_rules_and_verbs(default: WorldResult) -> None:
    taxonomy = default.taxonomy
    event_types = default.event_types
    assert [e.label for e in event_types.unary] == [f"EVENTTYPE1.{i}" for i in range(1, 21)]
    assert len(event_types.binary) == len(taxonomy.relations.verbs.verbs)
    for et in event_types.unary:
        (constraint,) = et.constraints
        spec = event_types.constraint(constraint)
        rule = taxonomy.rules.rule_for(et.source_label)
        assert spec.table == rule.table
        assert all(k.startswith("agent.") for k in spec.inputs)
    for et in event_types.binary:
        relation = taxonomy.relations.relation(et.source_label)
        assert list(et.constraints) == [translate(c.label) for c in relation.constraints]
    for et in event_types.categories:
        relation = taxonomy.relations.relation(et.source_label)
        assert relation.base and list(et.constraints) == [
            translate(c.label) for c in relation.constraints
        ]


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
    from semantic_world.world.fluents import derived_initial_values

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


def test_fluents_off_gives_no_dynamics_and_the_taxonomys_requirements() -> None:
    config = config_from_mapping(world_mapping("tiny", fluents={"count": 0}))
    world = define(config)
    assert len(world.fluents) == 0
    for et in world.event_types.event_types:
        assert et.precondition == () and et.effects == ()
    taxonomy = generate_taxonomy(config.taxonomy_config())
    assert taxonomy.rules.records() == world.taxonomy.rules.records()
    assert taxonomy.relations.records() == world.taxonomy.relations.records()
    assert np.array_equal(taxonomy.projections.agent, world.taxonomy.projections.agent)
    assert taxonomy.relation_stats.proportions.equals(world.taxonomy.relation_stats.proportions)
    assert world.stats["effects"]["total"] == 0 and world.stats["enabling_graph"]["edges"] == 0


# ---------------------------------------------------------------------------------------------
# The event file
# ---------------------------------------------------------------------------------------------


def test_chain_example_loads(chain: WorldResult) -> None:
    catch = chain.event_types.event_type("EVENTTYPE2.1.1")
    eat = chain.event_types.event_type("EVENTTYPE2.1.2")
    assert catch.explicit and eat.explicit
    assert [lit.text for lit in catch.precondition] == ["agent.BOOLFL.1", "NOT patient.BOOLFL.2"]
    assert [e.text for e in catch.effects] == ["patient.BOOLFL.2 := 1"]
    assert [e.text for e in eat.effects] == ["patient.BOOLFL.1 := 0"]
    assert chain.fluents["BOOLFL.1"].initial_rate == 1.0
    assert chain.fluents["BOOLFL.2"].initial_rate == 0.0
    assert not chain.fluents["BOOLFL.1"].derived and not chain.fluents["BOOLFL.2"].derived
    assert ("EVENTTYPE2.1.1", "EVENTTYPE2.1.2") in [
        (a, b)
        for a, b in __import__("semantic_world.world.stats", fromlist=["x"]).enabling_edges(
            chain.event_types
        )
    ]


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


def test_explicit_requirements_replace_the_taxonomys(tmp_path: Path) -> None:
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
    values = world.taxonomy.instances.values
    features = world.taxonomy.features
    expected = values[:, features["IS.1"].position] & (1 - values[:, features["HAS.2"].position])
    assert np.array_equal(values[:, features["CAN.1"].position], expected)
    capacities = world.derived_frames()["capacities.csv"]
    assert capacities["CAN.EVENTTYPE1.1"].to_list() == expected.astype(int).tolist()
    own = world.taxonomy.relations.own_constraints["V1.1"]
    assert own.family == "explicit" and str(own.expression) == "a.IS.1 AND p.SC.1 > 0.5000"
    spec = world.event_types.constraint("CONSTRAINT.EVENTTYPE2.1.1")
    assert spec.expression == "agent.PROPERTY.1 AND patient.SCALARDIM.1 > 0.5000"
    # The taxonomy run written to taxonomy/ is the unmodified one.
    assert world.taxonomy_run.relations.own_constraints["V1.1"].family != "explicit"
    check_definition(world.definition)


# ---------------------------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------------------------


def test_definition_json_has_the_spec_keys_and_a_recomputable_identity(tiny_folder: Path) -> None:
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
    assert a.taxonomy.rules.records() == b.taxonomy.rules.records()


def test_entities_csv_holds_base_facts_and_initial_fluents(
    tiny: WorldResult, tiny_folder: Path
) -> None:
    entities = pl.read_csv(tiny_folder / "entities.csv")
    features = tiny.taxonomy.features
    free = [translate(f.label) for f in features.features if f.free and f.type != "can"]
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


# ---------------------------------------------------------------------------------------------
# Derived values, views, and the run folder
# ---------------------------------------------------------------------------------------------


def test_capacities_equal_the_taxonomys_projections(tiny: WorldResult, tiny_folder: Path) -> None:
    capacities = pl.read_csv(tiny_folder / "derived" / "capacities.csv")
    projections = pl.read_csv(tiny_folder / "taxonomy" / "projections.csv")
    instances = pl.read_csv(tiny_folder / "taxonomy" / "instances.csv")
    for column in projections.columns:
        if column in ("label", "approximate"):
            continue
        assert capacities[capacity_label(column)].to_list() == projections[column].to_list(), column
    for column in instances.columns:
        if column.startswith("CAN.") and not column.startswith("CAN.V"):
            assert capacities[capacity_label(column)].to_list() == instances[column].to_list()
    roles = pl.read_csv(tiny_folder / "derived" / "capacity_roles.csv")
    assert set(roles["status"].to_list()) <= {"fixed_1", "fixed_0", "free"}
    assert len(roles) == len(tiny.taxonomy.tree.categories) * 4


def test_relation_statistics_are_the_taxonomys_relabeled(tiny_folder: Path) -> None:
    world_stats = pl.read_csv(tiny_folder / "derived" / "event_type_stats.csv")
    taxonomy_stats = pl.read_csv(tiny_folder / "taxonomy" / "verb_stats.csv")
    assert world_stats.columns == ["event_type", *taxonomy_stats.columns[1:]]
    assert world_stats["event_type"].to_list() == [
        translate(v) for v in taxonomy_stats["verb"].to_list()
    ]
    assert world_stats.drop("event_type").equals(taxonomy_stats.drop("verb"))
    pairs = pl.read_csv(tiny_folder / "derived" / "relation_pairs.csv")
    assert pairs["agent"][0].startswith("INSTANCE.")


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


def test_classic_view_gives_the_old_instances_columns_translated(tiny_folder: Path) -> None:
    path = write_view(tiny_folder, "classic")
    view = pl.read_csv(path)
    old = pl.read_csv(tiny_folder / "taxonomy" / "instances.csv")
    kept = [c for c in old.columns if not c.startswith(("CAN.V", "CANBE.V"))]
    assert view.columns == [translate_column(c) for c in kept]
    for column in kept:
        if column == "SC.1":
            assert np.allclose(view["SCALARDIM.1"].to_numpy(), old["SC.1"].to_numpy(), atol=1e-6)
        elif column in ("label", "leaf"):
            assert view[column].to_list() == [translate(v) for v in old[column].to_list()]
        else:
            assert view[translate_column(column)].to_list() == old[column].to_list(), column
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
    standalone = generate_taxonomy(
        load_taxonomy_config(TAXONOMY_DATA / "tiny_relations.yaml")
    ).write(tmp_path / "t")
    for file in standalone.rglob("*"):
        if file.is_file() and file.name != "config.yaml":
            relative = file.relative_to(standalone)
            assert file.read_bytes() == (tiny_folder / "taxonomy" / relative).read_bytes(), str(
                relative
            )


def test_same_configuration_and_seed_give_byte_identical_runs(tmp_path: Path) -> None:
    a = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "a")
    b = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "b")
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    for name in files:
        if not name.endswith("config.yaml"):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_dynamics_settings_never_change_the_requirements() -> None:
    a = define(config_from_mapping(world_mapping("tiny")))
    b = define(
        config_from_mapping(
            world_mapping(
                "tiny",
                event_types={
                    "effects": {"count": {3: 1.0}},
                    "preconditions": {"feature_rate": 1.0},
                },
            )
        )
    )
    assert a.taxonomy.rules.records() == b.taxonomy.rules.records()
    assert a.taxonomy.relations.records() == b.taxonomy.relations.records()
    assert a.derived_frames()["capacities.csv"].equals(b.derived_frames()["capacities.csv"])
    assert a.derived_frames()["relation_proportions.csv"].equals(
        b.derived_frames()["relation_proportions.csv"]
    )
    assert a.fluents.initial_values.tolist() == b.fluents.initial_values.tolist()


def test_world_stats_report_the_structure(tiny_folder: Path) -> None:
    stats = yaml.safe_load((tiny_folder / "world_stats.yaml").read_text())
    assert stats["fluents"] == {"base": 3, "derived": 1}
    assert stats["event_types"]["one_place"] == 4 and stats["event_types"]["two_place"] == 4
    assert set(stats["fix_ups"]) >= {"effects_dropped_contradiction", "effects_dropped_empty"}
    assert "longest_chain" in stats["enabling_graph"] and stats["episodes"] is None


def test_config_yaml_records_the_taxonomy_and_every_seed(tiny_folder: Path) -> None:
    data = yaml.safe_load((tiny_folder / "config.yaml").read_text())
    assert data["taxonomy"]["resolved"]["name"] == "tiny_relations"
    seeds = data["provenance"]["stream_seeds"]
    assert {
        "world:fluents",
        "world:initial",
        "world:preconditions",
        "world:effects",
        "world:stats",
    } <= set(seeds)
    assert "taxonomy:rules" in seeds


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
