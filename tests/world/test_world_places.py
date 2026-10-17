"""Stage c1 of ``docs/specs/WORLD_PHASE_C.md``: places. The second taxonomy run (``places/``,
byte-identical to the taxonomy run alone with the same configuration and seed), ``places.csv``,
the place symbols and scopes of ``definition.json`` (present only with places), the place event
types' requirements (never degenerate on the tiny and default worlds, in matrix form agreeing
with the truth tables over every entity and place), the capacities ``CAN.PLACEEVENTTYPE`` and
``CANBE.PLACEEVENTTYPE`` against a brute-force check, the place comparison classes against a
recomputation from ``places.csv``, the runtime's refusal of a comparison that mixes an entity
scalar with a place scalar, the relation statistics, the statistics block, the views, and the
promise that no episode names a place. The identity with ``places: null`` is checked against
the snapshot by ``test_world_off_settings.py``."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import generate as generate_taxonomy
from semantic_world.taxonomy import load_config as load_taxonomy_config
from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.__main__ import main
from semantic_world.world.comparisons import COMPARISON_CLASSES_FILE, comparison_classes_frame
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.derived import load_derived_csv
from semantic_world.world.episodes import EpisodeGenerator, Relatedness
from semantic_world.world.errors import WorldError
from semantic_world.world.fixtures import BruteEvent, BruteForce, fixture_inputs, fixture_places
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.places import (
    EVERYPLACE,
    PLACE_CAPACITIES_FILE,
    PLACES_DIR,
    PLACES_FILE,
    requirement_matrix,
)
from semantic_world.world.runtime import (
    ENTITY,
    Mean,
    above,
    below,
    comparison_classes,
    greater,
    initial_state,
    legal_bindings,
    members,
    place_categories,
)
from semantic_world.world.streams import STREAM_NAMES, WorldStreams

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"
SLOW = pytest.mark.slow
NAMES = ["tiny", pytest.param("default", marks=SLOW)]


@pytest.fixture(scope="module")
def worlds() -> dict[str, WorldResult]:
    return {}


def world_named(worlds: dict[str, WorldResult], name: str) -> WorldResult:
    if name not in worlds:
        worlds[name] = define(load_config(DATA / f"{name}.yaml"))
    return worlds[name]


@pytest.fixture(scope="module")
def tiny(worlds) -> WorldResult:
    return world_named(worlds, "tiny")


@pytest.fixture(scope="module")
def tiny_folder(tiny: WorldResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.write(tmp_path_factory.mktemp("world") / "tiny")


def tiny_mapping(**overrides) -> dict:
    data = yaml.safe_load((DATA / "tiny.yaml").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


# ---------------------------------------------------------------------------------------------
# Configuration and the second taxonomy run
# ---------------------------------------------------------------------------------------------


def test_the_places_block_is_read_validated_and_defaulted() -> None:
    config = load_config(DATA / "default.yaml")
    assert config.places is not None and config.place_event_types is not None
    assert config.places.taxonomy.config == "data/taxonomy/places_default.yaml"
    assert config.places.taxonomy.seed is None
    assert config.places.habitat_strength == 1.0
    assert config.places.taxonomy_run.labels.kind == "place"
    # the place taxonomy's seed is the stream world:places (its low 64 bits)
    assert config.places.seed == WorldStreams(config.seed).seed("places") % 2**64
    assert config.place_event_types.count == 4
    assert config.place_event_types.move_share == 0.5
    assert config.place_event_types.constraints == {1: 0.6, 2: 0.4}
    assert config.place_event_types.constraint_families == {
        "agent": 1,
        "patient": 1,
        "cross": 1,
        "key_lock": 1,
    }
    # the code's defaults follow the file (WM.E41)
    defaults = config_from_mapping({}, source="<test>")
    assert defaults.places is not None and defaults.place_event_types is not None
    assert defaults.resolved()["places"] == config.resolved()["places"]
    assert defaults.resolved()["place_event_types"] == config.resolved()["place_event_types"]
    # a given seed replaces the stream's
    seeded = config_from_mapping({"places": {"taxonomy": {"seed": 7}}}, source="<test>")
    assert seeded.places is not None and seeded.places.seed == 7
    # places: null turns places off, and then place_event_types must be absent or null
    off = config_from_mapping({"places": None}, source="<test>")
    assert off.places is None and off.place_event_types is None
    assert off.resolved()["places"] is None and off.resolved()["place_event_types"] is None
    with pytest.raises(ConfigError, match="place_event_types"):
        config_from_mapping({"places": None, "place_event_types": {"count": 2}}, source="<test>")
    # an entity taxonomy cannot be the place taxonomy
    with pytest.raises(ConfigError, match="labels.kind: place"):
        config_from_mapping(
            {"places": {"taxonomy": {"config": str(TAXONOMY_DATA / "tiny.yaml")}}},
            source="<test>",
        )
    with pytest.raises(ConfigError, match="habitat_strength"):
        config_from_mapping({"places": {"habitat_strength": -1}}, source="<test>")
    with pytest.raises(ConfigError, match="move_share"):
        config_from_mapping({"place_event_types": {"move_share": 2}}, source="<test>")
    with pytest.raises(ConfigError, match="constraints"):
        config_from_mapping({"place_event_types": {"constraints": {0: 1}}}, source="<test>")
    with pytest.raises(ConfigError, match="constraint_families"):
        config_from_mapping(
            {"place_event_types": {"constraint_families": {"nowhere": 1}}}, source="<test>"
        )
    # the resolved configuration loads back
    again = config_from_mapping(config.resolved(), source="<test>")
    assert again.resolved() == config.resolved()


def test_the_two_place_streams_come_after_every_earlier_stream() -> None:
    assert STREAM_NAMES[-2:] == ("places", "place_event_types")


def test_the_place_run_is_byte_identical_to_the_taxonomy_run_alone(
    tiny: WorldResult, tiny_folder: Path, tmp_path: Path
) -> None:
    assert tiny.places is not None
    assert tiny.config.places is not None
    config = load_taxonomy_config(TAXONOMY_DATA / "places_tiny.yaml", seed=tiny.config.places.seed)
    alone = generate_taxonomy(config).write(tmp_path / "alone")
    embedded = tiny_folder / PLACES_DIR
    files = sorted(p.relative_to(alone).as_posix() for p in alone.rglob("*") if p.is_file())
    assert files == sorted(
        p.relative_to(embedded).as_posix() for p in embedded.rglob("*") if p.is_file()
    )
    for name in files:
        if name == "config.yaml":
            # the same settings; the provenance differs by the stream seeds' owner only
            a = yaml.safe_load((alone / name).read_text(encoding="utf-8"))
            b = yaml.safe_load((embedded / name).read_text(encoding="utf-8"))
            a.pop("provenance"), b.pop("provenance")
            assert a == b
            continue
        assert (alone / name).read_bytes() == (embedded / name).read_bytes(), name
    assert config.labels.kind == "place"


def test_places_csv_holds_the_places_base_facts(tiny: WorldResult, tiny_folder: Path) -> None:
    assert tiny.places is not None
    places = pl.read_csv(tiny_folder / PLACES_FILE)
    taxonomy = tiny.places.taxonomy
    free = [f.label for f in taxonomy.features.free]
    assert places.columns == ["label", "leaf", *free, *taxonomy.features.scalar_labels]
    assert places["label"].to_list() == list(taxonomy.instances.labels)
    assert all(label.startswith("PLACE.") for label in places["label"].to_list())
    assert all(leaf.startswith("PLACECATEGORY.") for leaf in places["leaf"].to_list())
    for feature in taxonomy.features.free:
        assert places[feature.label].to_list() == (
            taxonomy.instances.values[:, feature.position].tolist()
        )
    for j, label in enumerate(taxonomy.features.scalar_labels):
        assert places[label].to_list() == taxonomy.instances.scalars[:, j].tolist()
    # places are not entities: nothing of the entity side names a place
    entities = pl.read_csv(tiny_folder / "entities.csv")
    assert not any(c.startswith("PLACE") for c in entities.columns)
    assert not set(places["label"].to_list()) & set(entities["label"].to_list())


# ---------------------------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_place_symbols_and_scopes_appear_only_with_places(worlds, name: str) -> None:
    world = world_named(worlds, name)
    assert world.places is not None
    record = world.definition.record()
    kinds = {s["kind"] for s in record["symbols"]}
    assert {"place_property", "place_scalar", "place_event_type"} <= kinds
    place_symbols = [s for s in record["symbols"] if s["kind"].startswith("place")]
    assert [s["label"] for s in place_symbols] == [
        *(f.label for f in world.places.taxonomy.features.features),
        *world.places.taxonomy.features.scalar_labels,
        *world.places.event_type_labels,
    ]
    assert {r["scope"] for r in record["rules"]} == {"entity", "binding", "place", "place_binding"}
    assert {layer["scope"] for layer in record["layers"]} >= {"place", "place_binding"}
    for et in record["event_types"]:
        if et["kind"] == "place_event_type":
            assert et["arity"] == 2 and et["roles"] == ["agent", "patient"]
            assert et["precondition"] == {"literals": [], "expression": "TRUE"}
            assert et["effects"] == [] and et["parent"] is None
            assert et["requirement"]["output"] == f"REQUIREMENT.{et['label']}"
    # every place rule and place requirement is in matrix form, and the agreement test ran
    assert world.report.place is not None and world.report.place_binding is not None
    assert world.report.place.rules == len(world.places.taxonomy.rules.rules)
    assert world.report.place_binding.rules == len(world.places.constraints) + len(
        world.places.event_types
    )
    assert world.report.place_binding.entities == len(world.taxonomy.instances) * (
        world.places.count
    )
    # with places off, nothing of the above is in the definition
    data = tiny_mapping(places=None) if name == "tiny" else None
    if data is not None:
        data.pop("place_event_types", None)
        off = define(config_from_mapping(data, source=str(DATA / "tiny.yaml")))
        off_record = off.definition.record()
        assert not any(s["kind"].startswith("place") for s in off_record["symbols"])
        assert {r["scope"] for r in off_record["rules"]} == {"entity", "binding"}
        assert not any("PLACE" in lit["key"] for lit in off_record["literals"])
        assert off.places is None and off.report.place is None
        assert off.stats["places"] is None


@pytest.mark.parametrize("name", NAMES)
def test_no_place_requirement_is_degenerate_and_the_matrices_agree(worlds, name: str) -> None:
    world = world_named(worlds, name)
    places = world.places
    assert places is not None
    entities = world.taxonomy.instances
    place_instances = places.taxonomy.instances
    for et in places.event_types:
        matrix = places.requirement[et.label]
        assert matrix.shape == (len(entities), len(place_instances))
        assert matrix.any() and not matrix.all(), et.label
        assert 0.0 < et.density < 1.0
        # the requirement is the conjunction of its constraints, by truth-table lookup
        expected = requirement_matrix(
            et.constraints,
            entities.values,
            entities.scalars,
            place_instances.values,
            place_instances.scalars,
        )
        assert np.array_equal(matrix, expected)
        assert len(et.constraints) in (1, 2)
        for k, constraint in enumerate(et.constraints, start=1):
            assert constraint.label == f"CONSTRAINT.{et.label}.{k}"
            for item in constraint.literals:
                role = getattr(item, "role", "both")
                if role == "agent":
                    assert not item.key.split(".", 1)[1].startswith("PLACE")
                elif role == "patient":
                    assert item.key.split(".", 1)[1].startswith("PLACE")
    # the runtime computes the same requirement over every entity and place binding
    definition = world.definition.runtime()
    state = initial_state(definition)
    for et in places.event_types:
        bindings = legal_bindings(definition, state, et.label)
        expected = [
            (int(a), int(p)) for a, p in zip(*np.nonzero(places.requirement[et.label]), strict=True)
        ]
        assert bindings == expected


# ---------------------------------------------------------------------------------------------
# Capacities
# ---------------------------------------------------------------------------------------------


def _brute_force_projection(world: WorldResult, et, unknown: str) -> np.ndarray:
    """The intensional capacity by enumeration: every setting of the unknown side's free
    features with representative scalar values, as ``test_world_projections`` does, with the
    unknown side the place taxonomy (for ``CAN``) or the entity taxonomy (for ``CANBE``)."""
    places = world.places
    assert places is not None
    if unknown == "patient":
        unknown_rules, known = places.taxonomy.rules, world.taxonomy.instances
        known_rules = world.taxonomy.rules
    else:
        unknown_rules, known = world.taxonomy.rules, places.taxonomy.instances
        known_rules = places.taxonomy.rules
    features = unknown_rules.features
    free = features.free
    grid = np.array(list(itertools.product((0, 1), repeat=len(free))), dtype=np.uint8).reshape(
        -1, len(free)
    )
    k = features.scalar_count
    thresholds = sorted(
        {t.threshold for rule in unknown_rules.rules for t in rule.thresholds}
        | {
            item.threshold.threshold
            for c in et.constraints
            for item in c.literals
            if hasattr(item, "threshold") and item.role == unknown
        }
    )
    out = np.zeros(len(known), dtype=bool)
    by_output = {r.output.label: r for r in unknown_rules.rules}
    for x in range(len(known)):
        cuts = set(thresholds)
        for c in et.constraints:
            for item in c.comparisons:
                if unknown == "patient":
                    value = known.scalars[x, item.agent_scalar - 1]
                    cuts.update([value - item.low] + ([value - item.high] if item.high else []))
                else:
                    value = known.scalars[x, item.patient_scalar - 1]
                    cuts.update([value + item.low] + ([value + item.high] if item.high else []))
        cuts = sorted(cuts) or [0.0]
        points = (
            [cuts[0] - 1]
            + cuts
            + [(a + b) / 2 for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
            + [cuts[-1] + 1]
        )
        scalar_grid = np.array(list(itertools.product(points, repeat=k))).reshape(-1, k)
        rows_free = np.repeat(grid, scalar_grid.shape[0], axis=0)
        rows_scalar = np.tile(scalar_grid, (grid.shape[0], 1))
        other_values = unknown_rules.compute(rows_free, rows_scalar if k else None)
        m = other_values.shape[0]
        own_values = np.repeat(known.values[x : x + 1], m, axis=0)
        own_scalars = np.repeat(known.scalars[x : x + 1], m, axis=0)
        held = np.ones(m, dtype=bool)
        for c in et.constraints:
            if unknown == "patient":
                held &= c.evaluate(own_values, own_scalars, other_values, rows_scalar)
            else:
                held &= c.evaluate(other_values, rows_scalar, own_values, own_scalars)
        out[x] = held.any()
    del known_rules, by_output
    return out


def test_place_capacities_agree_with_brute_force_and_are_written(
    tiny: WorldResult, tiny_folder: Path
) -> None:
    places = tiny.places
    assert places is not None
    for i, et in enumerate(places.event_types):
        matrix = places.requirement[et.label]
        # extensional: some actual place, or some actual entity
        assert np.array_equal(places.actual_agent[:, i], matrix.any(axis=1))
        assert np.array_equal(places.actual_patient[:, i], matrix.any(axis=0))
        # intensional, by enumeration of the other side's possible features
        assert np.array_equal(places.agent[:, i], _brute_force_projection(tiny, et, "patient"))
        assert np.array_equal(places.patient[:, i], _brute_force_projection(tiny, et, "agent"))
        # every extensional capacity is intensional
        assert not (places.actual_agent[:, i] & ~places.agent[:, i]).any()
        assert not (places.actual_patient[:, i] & ~places.patient[:, i]).any()
    assert not places.agent_approximate.any() and not places.patient_approximate.any()
    capacities = load_derived_csv(tiny_folder / "derived", "capacities.csv", tiny.rule_set_id)
    for i, label in enumerate(places.event_type_labels):
        assert capacities[f"CAN.{label}"].to_list() == places.agent[:, i].astype(int).tolist()
        assert (
            capacities[f"ACTUAL_CAN.{label}"].to_list()
            == places.actual_agent[:, i].astype(int).tolist()
        )
    place_capacities = load_derived_csv(
        tiny_folder / "derived", PLACE_CAPACITIES_FILE, tiny.rule_set_id
    )
    assert place_capacities["label"].to_list() == list(places.labels)
    assert place_capacities.columns == [
        "label",
        *(f"CANBE.{label}" for label in places.event_type_labels),
        "approximate",
        *(f"ACTUAL_CANBE.{label}" for label in places.event_type_labels),
    ]
    for i, label in enumerate(places.event_type_labels):
        assert place_capacities[f"CANBE.{label}"].to_list() == (
            places.patient[:, i].astype(int).tolist()
        )
        assert place_capacities[f"ACTUAL_CANBE.{label}"].to_list() == (
            places.actual_patient[:, i].astype(int).tolist()
        )
    # the brute-force evaluator judges every binding as the requirement matrix does
    record, entities, _ = fixture_inputs(tiny.definition)
    brute = BruteForce(record, entities, fixture_places(tiny.definition) or ())
    for et in places.event_types:
        for a, entity in enumerate(brute.entity_labels):
            for p, place in enumerate(brute.place_labels):
                event = BruteEvent(et.label, {"agent": entity, "patient": place})
                assert brute.able(event) == bool(places.requirement[et.label][a, p])
    assert et.label not in brute.performable()


# ---------------------------------------------------------------------------------------------
# Comparison classes and the runtime's comparisons
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_place_comparison_classes_agree_with_a_recomputation_from_places_csv(
    worlds, name: str, tmp_path: Path
) -> None:
    world = world_named(worlds, name)
    assert world.places is not None
    folder = world.write(tmp_path / name)
    frame = load_derived_csv(folder / "derived", COMPARISON_CLASSES_FILE, world.rule_set_id)
    with (folder / PLACES_FILE).open(encoding="utf-8", newline="") as handle:
        places = list(csv.DictReader(handle))
    with (folder / PLACES_DIR / "tree.csv").open(encoding="utf-8", newline="") as handle:
        tree = list(csv.DictReader(handle))
    parent = {row["label"]: row["parent"] or None for row in tree}
    scalars = [c for c in places[0] if c.startswith("PLACESCALARDIM.")]
    assert scalars

    def below_class(category: str) -> list[dict]:
        if category == EVERYPLACE:
            return places
        found = []
        for row in places:
            node: str | None = row["leaf"]
            while node is not None:
                if node == category:
                    found.append(row)
                    break
                node = parent[node]
        return found

    rows = frame.to_dicts()
    entity_rows = [r for r in rows if not r["scalar"].startswith("PLACE")]
    place_rows = [r for r in rows if r["scalar"].startswith("PLACE")]
    assert rows == entity_rows + place_rows  # the place rows follow the entity rows
    expected = [EVERYPLACE] + [row["label"] for row in tree]
    assert [r["category"] for r in place_rows] == [c for c in expected for _ in scalars]
    for row in place_rows:
        values = np.array([float(r[row["scalar"]]) for r in below_class(row["category"])])
        assert row["instances"] == len(values)
        assert abs(row["mean"] - values.mean()) < 1e-5
        assert abs(row["sd"] - values.std()) < 1e-5
    exact = comparison_classes_frame(world.taxonomy, world.places.taxonomy).to_dicts()
    assert len(exact) == len(rows)
    # the runtime computes the same classes from the place leaves
    definition = world.definition.runtime()
    table = comparison_classes(definition)
    assert set(place_categories(definition)) == {r["category"] for r in place_rows} - {EVERYPLACE}
    for row in exact:
        if row["scalar"].startswith("PLACE"):
            count, mean, sd = table[(row["category"], row["scalar"])]
            assert count == row["instances"] and mean == row["mean"] and sd == row["sd"]
    assert members(definition, EVERYPLACE).size == definition.place_count


def test_the_runtime_compares_places_and_refuses_mixed_comparisons(tiny: WorldResult) -> None:
    definition = tiny.definition.runtime()
    assert definition.has_places
    scalar = definition.place_scalars[0]
    values = definition.place_scalar_values[:, 0]
    assert definition.place_scalar_values is not None
    for x, y in itertools.permutations(range(definition.place_count), 2):
        assert greater(definition, scalar, x, y) == (values[x] > values[y])
    for category in (EVERYPLACE, *place_categories(definition)):
        rows = members(definition, category)
        mean, sd = values[rows].mean(), values[rows].std()
        for x in range(definition.place_count):
            for z in (0.0, 0.5, 1.0):
                assert above(definition, scalar, x, category, z) == (
                    sd > 0 and values[x] >= mean + z * sd
                )
                assert below(definition, scalar, x, category, z) == (
                    sd > 0 and values[x] <= mean - z * sd
                )
        assert greater(definition, scalar, Mean(category), Mean(EVERYPLACE)) == (
            mean > values.mean()
        )
    entity_scalar = definition.scalars[0]
    mixed = (
        lambda: above(definition, scalar, 0, ENTITY, 1.0),
        lambda: above(definition, scalar, 0, "CATEGORY.1", 1.0),
        lambda: below(definition, entity_scalar, 0, EVERYPLACE, 1.0),
        lambda: below(definition, entity_scalar, 0, "PLACECATEGORY.1", 1.0),
        lambda: greater(definition, scalar, 0, Mean(ENTITY)),
        lambda: greater(definition, entity_scalar, Mean(EVERYPLACE), 0),
        lambda: greater(definition, scalar, definition.place_count, 0),
    )
    for comparison in mixed:
        with pytest.raises(WorldError):
            comparison()
    with pytest.raises(WorldError, match="not a category"):
        members(definition, "PLACECATEGORY.9.9")


# ---------------------------------------------------------------------------------------------
# Statistics, views, episodes
# ---------------------------------------------------------------------------------------------


def test_relation_statistics_and_the_places_block(tiny: WorldResult, tiny_folder: Path) -> None:
    places = tiny.places
    assert places is not None
    proportions = pl.read_csv(tiny_folder / "derived" / "relation_proportions.csv")
    rows = proportions.filter(pl.col("event_type").str.starts_with("PLACEEVENTTYPE."))
    assert rows.height > 0
    entity_tree = tiny.taxonomy.tree
    place_tree = places.taxonomy.tree
    for row in rows.iter_rows(named=True):
        level = row["level"]
        assert entity_tree[row["agent"]].level == level
        assert place_tree[row["patient"]].level == min(level, place_tree.depth)
        agents = tiny.taxonomy.instances.below(entity_tree[row["agent"]], entity_tree)
        patients = places.taxonomy.instances.below(place_tree[row["patient"]], place_tree)
        block = places.requirement[row["event_type"]][np.ix_(agents, patients)]
        assert row["true_pairs"] == int(block.sum()) and row["total_pairs"] == block.size
        assert row["proportion"] == pytest.approx(block.mean(), abs=1e-6)
        assert row["estimated"] is False
    assert sorted(set(rows["level"].to_list())) == list(range(1, entity_tree.depth + 1))
    stats = pl.read_csv(tiny_folder / "derived" / "event_type_stats.csv")
    place_stats = stats.filter(pl.col("event_type").str.starts_with("PLACEEVENTTYPE."))
    assert place_stats["event_type"].to_list() == list(places.event_type_labels)
    for row, et in zip(place_stats.iter_rows(named=True), places.event_types, strict=True):
        matrix = places.requirement[et.label]
        assert row["proportion_true"] == pytest.approx(matrix.mean(), abs=1e-6)
        assert row["agents"] == int(matrix.any(axis=1).sum())
        assert row["patients"] == int(matrix.any(axis=0).sum())
        assert row["constraints"] == len(et.constraints)
        assert row["families"] == ";".join(et.families)
        assert row["leaf_pair_density"] == pytest.approx(et.density, abs=1e-6)
    # thematic relatedness stays a relation between entity leaves
    thematic = pl.read_csv(tiny_folder / "derived" / "thematic.csv")
    assert all(label.startswith("CATEGORY.") for label in thematic["leaf_a"].to_list())
    block = yaml.safe_load((tiny_folder / "world_stats.yaml").read_text())["places"]
    assert block["places"] == places.count == 4
    assert block["categories"] == len(places.taxonomy.tree.categories)
    assert block["features"]["place_property"]["total"] == 4
    assert block["features"]["place_part"]["total"] == 0
    assert block["features"]["place_scalar"] == 1
    assert block["place_event_types"] == 2
    assert set(block["density"]) == set(places.event_type_labels)
    assert all(0 < d < 1 for d in block["density"].values())


def test_the_place_views(
    tiny: WorldResult, tiny_folder: Path, capsys: pytest.CaptureFixture
) -> None:
    places = tiny.places
    assert places is not None
    free = [f.label for f in places.taxonomy.features.free]
    determined = [f.label for f in places.taxonomy.features.determined]
    scalars = list(places.taxonomy.features.scalar_labels)
    assert main(["view", str(tiny_folder), "--places", "--preset", "base"]) == 0
    base = pl.read_csv(tiny_folder / "views" / "places_base.csv")
    assert base.columns == ["label", "leaf", *free, *scalars]
    assert base["label"].to_list() == list(places.labels)
    assert main(["view", str(tiny_folder), "--places", "--preset", "static"]) == 0
    static = pl.read_csv(tiny_folder / "views" / "places_static.csv")
    capacities = [f"CANBE.{label}" for label in places.event_type_labels]
    assert static.columns == [
        "label",
        "leaf",
        *sorted_place(free + determined),
        *capacities,
        *scalars,
    ]
    for i, label in enumerate(places.event_type_labels):
        assert static[f"CANBE.{label}"].to_list() == places.patient[:, i].astype(int).tolist()
    assert main(["view", str(tiny_folder), "--places", "--preset", "classic"]) == 0
    classic = pl.read_csv(tiny_folder / "views" / "places_classic.csv")
    isa = [f"ISA.{c.label}" for c in places.taxonomy.tree.categories]
    assert classic.columns == [
        "label",
        "leaf",
        *isa,
        *sorted_place(free + determined),
        *capacities,
        *scalars,
    ]
    sidecar = yaml.safe_load((tiny_folder / "views" / "places_classic.yaml").read_text())
    assert sidecar["places"] is True and sidecar["rule_set_id"] == tiny.rule_set_id
    assert main(["view", str(tiny_folder), "--places", "--include", "PLACEPROPERTY"]) == 0
    narrowed = pl.read_csv(tiny_folder / "views" / "places_classic.csv")
    assert all(c.startswith("PLACEPROPERTY.") for c in narrowed.columns[2:])
    # the entity views do not see the places
    assert main(["view", str(tiny_folder), "--preset", "classic"]) == 0
    entities = pl.read_csv(tiny_folder / "views" / "classic.csv")
    assert not any("PLACE" in c for c in entities.columns)
    capsys.readouterr()


def sorted_place(labels: list[str]) -> list[str]:
    return sorted(labels, key=lambda c: int(c.rsplit(".", 1)[1]))


def test_no_episode_or_history_names_a_place(tiny: WorldResult) -> None:
    definition = tiny.definition.runtime()
    generator = EpisodeGenerator(
        definition, Relatedness.from_statics(tiny.statics, definition), record_legal=True
    )
    assert not any(label.startswith("PLACEEVENTTYPE.") for label in generator.performable)
    histories = generator.run(tiny.config.seed, 50, "episodes")
    for history in histories:
        text = json.dumps(history.to_json())
        assert "PLACE" not in text, history.label
        for step in history.steps:
            assert not any(label.startswith("PLACE") for label in (step.legal or {}))
    assert "PLACE" not in json.dumps(tiny.stats["episodes"])


def test_apply_refuses_a_place_event_in_this_stage(tiny: WorldResult) -> None:
    from semantic_world.world.runtime import Event, apply

    definition = tiny.definition.runtime()
    state = initial_state(definition)
    label = tiny.places.event_type_labels[0]  # type: ignore[union-attr]
    binding = legal_bindings(definition, state, label)[0]
    with pytest.raises(WorldError, match="no precondition or effects in stage c1"):
        apply(definition, state, [Event(label, binding)])
