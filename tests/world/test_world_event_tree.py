"""The event-type tree of the world package (stage a5b of ``docs/specs/WORLD_AND_LANGUAGE.md``):
the ``event_types.binary`` block, the event-type features, the tree of two-place event types,
and its roles. These tests moved from ``tests/taxonomy/test_taxonomy_verb_tree.py`` (stage 9 of
the taxonomy generator) when the verb tree became the event-type tree.

A world configuration names a taxonomy file, so a test that needs its own taxonomy settings
writes them to a temporary file first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic_world.taxonomy import GenerationError
from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.taxonomy.similarity import similarity_matrix
from semantic_world.taxonomy.streams import Streams as TaxonomyStreams
from semantic_world.taxonomy.tree import Role
from semantic_world.world.config import Config, config_from_mapping
from semantic_world.world.event_tree import EVENT_FEATURE_TYPE, EventTree, generate_event_tree
from semantic_world.world.generate import define
from semantic_world.world.statics import build_statics
from semantic_world.world.streams import STREAM_NAMES, WorldStreams

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"
RELATIONS = str(TAXONOMY_DATA / "relations.yaml")
TINY_RELATIONS = str(TAXONOMY_DATA / "tiny_relations.yaml")

LARGE_BINARY = {
    "features": {"count": 30, "expected_true": 6},
    "taxonomy": {"superordinates": 3, "depth": 4, "branching": 4},
    "inheritance": {"proportion_defining": 0.1, "proportion_characteristic": 0.5},
}


def world_config(
    taxonomy: str | dict = RELATIONS,
    seed: int = 1,
    tmp_path: Path | None = None,
    **blocks,
) -> Config:
    """A world configuration over a taxonomy file, or over a taxonomy mapping written to
    ``tmp_path``. ``blocks`` are the world's top-level blocks (``event_types``, ``fluents``)."""
    if isinstance(taxonomy, dict):
        assert tmp_path is not None
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(taxonomy, sort_keys=False), encoding="utf-8")
        taxonomy = str(path)
    data = {"name": "test", "seed": seed, "taxonomy": {"config": taxonomy}, **blocks}
    return config_from_mapping(data, source="<test>")


def event_tree(binary: dict | None, seed: int = 1, taxonomy: str = RELATIONS) -> EventTree:
    config = world_config(taxonomy, seed, event_types={"binary": binary})
    tree = generate_event_tree(config.event_types.binary, WorldStreams(config.seed).event_tree)
    assert tree is not None
    return tree


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_binary_is_on_by_default_and_null_turns_it_off() -> None:
    on = world_config(TINY_RELATIONS)
    assert on.event_types.binary is not None
    off = world_config(TINY_RELATIONS, event_types={"binary": None})
    assert off.event_types.binary is None
    assert off.resolved()["event_types"]["binary"] is None
    assert generate_event_tree(None, WorldStreams(1).event_tree) is None
    world = define(off)
    statics = world.statics
    assert statics.event_tree is None and statics.relations is None
    assert statics.projections is None and statics.relation_stats is None
    assert world.event_types.binary == ()
    # The world's taxonomy still has no verbs block of its own.
    assert "verbs" not in world.taxonomy.config.resolved()


def test_binary_defaults_match_the_specification() -> None:
    config = world_config()
    binary = config.event_types.binary
    assert binary is not None
    assert (binary.feature_count, binary.expected_true) == (12, 3)
    assert binary.base_rate == pytest.approx(0.25)
    assert binary.feature_labels[:2] == ("EVENTFEAT.1", "EVENTFEAT.2")
    assert (binary.taxonomy.superordinates, binary.taxonomy.depth) == (3, 2)
    assert [r.resolved() for r in binary.taxonomy.branching] == [[2, 3]]
    assert binary.similarity_bound is None
    assert binary.inheritance.proportion_defining == (0.4, 0.4)
    assert binary.inheritance.proportion_characteristic == (0.4, 0.4)
    assert binary.inheritance.characteristic_probability == (0.9, 0.9)
    assert binary.inheritance.require_distinct_leaves is True
    assert binary.inheritance.distinct_max_tries == 1000
    assert binary.own_constraint is True
    assert binary.constraint_families == {
        "agent": 1,
        "patient": 1,
        "cross": 1,
        "key_lock": 1,
        "comparison": 1,
    }
    assert binary.key_lock_pairs == {1: 0.5, 2: 0.3, 3: 0.2}
    assert binary.comparison.window_probability == 0.3
    assert binary.comparison.cross_dimension_probability == 0.2
    assert binary.comparison.margin_quantiles == (0.1, 0.9)
    assert binary.rules == config.taxonomy_config().rules.sampling
    assert (binary.sampled_true, binary.sampled_false, binary.max_exact_pairs) == (
        1000,
        1000,
        50_000_000,
    )
    assert binary.tree_settings().prefix == "EVENTTYPE2."
    assert binary.tree_settings().scalars is None
    resolved = config.resolved()["event_types"]["binary"]
    assert list(resolved) == [
        "features",
        "taxonomy",
        "superordinates",
        "inheritance",
        "own_constraint",
        "constraint_families",
        "key_lock_pairs",
        "comparison",
        "rules",
        "pairs",
        "density",
        "constraint_min_density",
    ]
    assert "projections" not in resolved  # exposure left with stage a5b
    assert config_from_mapping(config.resolved()).resolved() == config.resolved()


def test_binary_rules_override_the_taxonomys_settings(tmp_path: Path) -> None:
    taxonomy = {"rules": {"negation_probability": 0.5}, "scalars": {"count": 2}}
    config = world_config(
        taxonomy, tmp_path=tmp_path, event_types={"binary": {"rules": {"arity": {2: 1}}}}
    )
    binary = config.event_types.binary
    assert binary.rules.arity == {2: 1}
    assert binary.rules.negation_probability == 0.5  # inherited from the taxonomy's rules
    assert config.taxonomy_config().rules.sampling.arity == {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}


@pytest.mark.parametrize(
    ("binary", "field"),
    [
        ({"colour": 1}, "event_types.binary.colour"),
        ({"features": {"count": -1}}, "event_types.binary.features.count"),
        (
            {"features": {"count": 4, "expected_true": 5}},
            "event_types.binary.features.expected_true",
        ),
        ({"taxonomy": {"depth": 0}}, "event_types.binary.taxonomy.depth"),
        (
            {"taxonomy": {"depth": 3, "branching": {"schedule": "list", "values": [2]}}},
            "event_types.binary.taxonomy.branching.values",
        ),
        (
            {"inheritance": {"proportion_defining": 0.7, "proportion_characteristic": 0.6}},
            "event_types.binary.inheritance.proportion_characteristic",
        ),
        (
            {"inheritance": {"proportion_defining": {"schedule": "list", "values": [0.1]}}},
            "event_types.binary.inheritance.proportion_defining.values",
        ),
        (
            {"superordinates": {"similarity_bound": {"metric": "euclid"}}},
            "event_types.binary.superordinates.similarity_bound.metric",
        ),
        ({"own_constraint": "yes"}, "event_types.binary.own_constraint"),
        (
            {"constraint_families": {"similarity": 1}},
            "event_types.binary.constraint_families.similarity",
        ),
        ({"key_lock_pairs": {0: 1}}, "event_types.binary.key_lock_pairs.0"),
        (
            {"comparison": {"margin_quantiles": [0.9, 0.1]}},
            "event_types.binary.comparison.margin_quantiles",
        ),
        (
            {"comparison": {"window_probability": 2}},
            "event_types.binary.comparison.window_probability",
        ),
        ({"rules": {"colour": 1}}, "event_types.binary.rules.colour"),
        ({"pairs": {"sampled_true": -1}}, "event_types.binary.pairs.sampled_true"),
        ({"pairs": {"max_exact_pairs": 0}}, "event_types.binary.pairs.max_exact_pairs"),
        ([], "event_types.binary"),
    ],
)
def test_broken_binary_settings_name_the_field(binary: object, field: str) -> None:
    with pytest.raises(ConfigError) as error:
        world_config(event_types={"binary": binary})
    assert error.value.field == field
    assert str(error.value).startswith(f"<test>: {field}: ")


def test_comparison_family_needs_scalars_in_the_taxonomy(tmp_path: Path) -> None:
    only_comparison = {"agent": 0, "patient": 0, "cross": 0, "key_lock": 0, "comparison": 1}
    with_scalars = world_config(event_types={"binary": {"constraint_families": only_comparison}})
    assert with_scalars.event_types.binary.constraint_families["comparison"] == 1
    with pytest.raises(ConfigError) as error:
        world_config(
            {"scalars": {"count": 0}},
            tmp_path=tmp_path,
            event_types={"binary": {"constraint_families": only_comparison}},
        )
    assert error.value.field == "event_types.binary.constraint_families"
    assert "scalars.count" in str(error.value)


# ---------------------------------------------------------------------------------------------
# Structure: the tree tests of the taxonomy's stage 4, for the event-type tree
# ---------------------------------------------------------------------------------------------


def test_event_features_and_tree_structure() -> None:
    tree_of = event_tree({})
    features = tree_of.features
    assert features.labels == tuple(f"EVENTFEAT.{i}" for i in range(1, 13))
    assert all(f.free and f.type == EVENT_FEATURE_TYPE and f.layer == 0 for f in features.features)
    assert features.base_rates.tolist() == [pytest.approx(0.25)] * 12
    assert features.determined == ()
    assert tree_of.rules.rules == ()
    tree = tree_of.tree
    labels = [c.label for c in tree.categories]
    assert labels[:2] == ["EVENTTYPE2.1", "EVENTTYPE2.1.1"]
    assert all(label.startswith("EVENTTYPE2.") for label in labels)
    assert len(tree.superordinates) == 3
    assert tree.depth == 2
    for category in tree.categories:
        assert category.level == len(category.indices)
        assert category.label == "EVENTTYPE2." + ".".join(str(i) for i in category.indices)
        assert category.values.shape == (12,)
        assert np.array_equal(category.values, category.free_values)
        assert category.scalars.shape == (0,)
        if category.level == 1:
            assert 2 <= len(category.children) <= 3
        else:
            assert category.is_leaf
    assert tree_of.event_types == tree.leaves
    assert tree_of.categories == tree.categories
    assert 6 <= len(tree_of.event_types) <= 9
    assert tree.superordinate_tries == 3
    assert tree.superordinate_similarity is None


def test_defining_matrix_holds_the_defining_features_only() -> None:
    tree_of = event_tree(LARGE_BINARY)
    matrix = tree_of.defining_matrix()
    assert matrix.shape == (len(tree_of.categories), 30)
    for i, category in enumerate(tree_of.categories):
        mask = category.defining_mask()
        assert np.array_equal(matrix[i, mask], category.free_values[mask])
        assert np.isnan(matrix[i, ~mask]).all()


def test_event_type_roles_are_valid_and_defining_features_persist() -> None:
    for seed in range(3):
        tree = event_tree(LARGE_BINARY, seed).tree
        assert len(tree.categories) == 3 + 12 + 48 + 192
        for category in tree.categories:
            roles = category.roles
            assert roles.shape == (30,)
            inherited = roles == Role.DEFINING_INHERITED
            if category.parent is None:
                assert not inherited.any()
            else:
                assert np.array_equal(inherited, category.parent.defining_mask())
            mask = category.defining_mask()
            for descendant in category.descendants():
                assert np.array_equal(descendant.free_values[mask], category.free_values[mask])
                assert np.all(descendant.roles[mask] == Role.DEFINING_INHERITED)


def test_event_type_copy_rate_and_base_rate() -> None:
    tree = event_tree(LARGE_BINARY, 1).tree
    copies = flips = trues = undiagnostic_total = 0
    for parent in tree.categories:
        characteristic = parent.roles == Role.CHARACTERISTIC
        undiagnostic = parent.roles == Role.UNDIAGNOSTIC
        for child in parent.children:
            same = child.free_values[characteristic] == parent.free_values[characteristic]
            copies += same.sum()
            flips += (~same).sum()
            trues += child.free_values[undiagnostic].sum()
            undiagnostic_total += undiagnostic.sum()
    assert copies + flips > 2000
    assert copies / (copies + flips) == pytest.approx(0.9, abs=0.02)
    assert trues / undiagnostic_total == pytest.approx(0.2, abs=0.02)


def test_event_type_superordinate_bound() -> None:
    bound = {
        "metric": "phi",
        "scope": "free",
        "min": -0.3,
        "max": 0.1,
        "max_tries": 2000,
        "local_search": True,
    }
    settings = {
        "superordinates": {"similarity_bound": bound},
        "features": {"count": 20, "expected_true": 6},
    }
    tree = event_tree(settings).tree
    rows = np.stack([c.values for c in tree.superordinates])
    sims = similarity_matrix(rows, "phi")[np.triu_indices(3, k=1)]
    assert np.all((sims >= -0.3 - 1e-12) & (sims <= 0.1 + 1e-12))
    assert tree.superordinate_similarity is not None
    infeasible = dict(bound, min=None, max=-0.6, max_tries=100)
    with pytest.raises(GenerationError, match="superordinate EVENTTYPE2."):
        event_tree({**settings, "superordinates": {"similarity_bound": infeasible}})


def test_distinct_event_types() -> None:
    tree_of = event_tree(LARGE_BINARY, 2)
    event_types = tree_of.event_types
    assert len({v.values.tobytes() for v in event_types}) == len(event_types) == 192
    few = {
        "features": {"count": 2, "expected_true": 1},
        "taxonomy": {"superordinates": 1, "depth": 2, "branching": 8},
        "inheritance": {
            "proportion_defining": 0,
            "proportion_characteristic": 0,
            "distinct_max_tries": 20,
        },
    }
    with pytest.raises(GenerationError, match="duplicates leaf"):
        event_tree(few)
    allowed = {**few, "inheritance": {**few["inheritance"], "require_distinct_leaves": False}}
    assert len(event_tree(allowed).event_types) == 8


# ---------------------------------------------------------------------------------------------
# The flat case
# ---------------------------------------------------------------------------------------------


def test_flat_event_tree_gives_independent_event_types() -> None:
    flat = {
        "taxonomy": {"superordinates": 400, "depth": 1},
        "features": {"count": 10, "expected_true": 3},
        "inheritance": {"require_distinct_leaves": False},
    }
    tree_of = event_tree(flat)
    tree = tree_of.tree
    assert len(tree.categories) == 400
    assert all(v.parent is None and v.is_leaf and v.level == 1 for v in tree_of.event_types)
    assert all(not (v.roles == Role.DEFINING_INHERITED).any() for v in tree_of.event_types)
    # Each event type is an independent draw from the base rates: the mean is the base rate,
    # and the event-type features are uncorrelated across event types.
    matrix = tree.generative_matrix().astype(float)
    assert matrix.mean() == pytest.approx(0.3, abs=0.02)
    correlations = np.corrcoef(matrix, rowvar=False)
    off_diagonal = correlations[~np.eye(10, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.2


# ---------------------------------------------------------------------------------------------
# Determinism and independence from the taxonomy
# ---------------------------------------------------------------------------------------------


def _snapshot(tree_of: EventTree) -> list:
    return [(c.label, c.values.tobytes(), c.roles.tobytes()) for c in tree_of.categories]


def _taxonomy_snapshot(result) -> tuple:
    return (
        result.rules.records(),
        result.tree.generative_matrix().tobytes(),
        np.stack([c.roles for c in result.tree.categories]).tobytes(),
        result.instances.labels,
        result.instances.values.tobytes(),
        result.instances.scalars.tobytes(),
        result.similarity.write_csv(),
        result.feature_stats.write_csv(),
    )


def test_event_tree_is_deterministic_and_uses_only_its_stream() -> None:
    a = event_tree({}, 4)
    b = event_tree({}, 4)
    c = event_tree({}, 5)
    assert _snapshot(a) == _snapshot(b)
    assert _snapshot(a) != _snapshot(c)
    config = world_config(seed=4)
    used = WorldStreams(4)
    generate_event_tree(config.event_types.binary, used.event_tree)
    fresh = WorldStreams(4)
    assert used.event_tree.random() != fresh.event_tree.random()
    for name in STREAM_NAMES:
        if name != "event_tree":
            assert getattr(used, name).random() == getattr(fresh, name).random(), name


def test_event_tree_is_seeded_by_the_world_not_the_taxonomy() -> None:
    """The same binary block over two different taxonomies gives the same event tree, and the
    taxonomy's own streams are never touched."""
    over_tiny = event_tree({}, 3, TINY_RELATIONS)
    over_default = event_tree({}, 3, RELATIONS)
    assert _snapshot(over_tiny) == _snapshot(over_default)
    taxonomy_streams = TaxonomyStreams(3)
    generate_event_tree(world_config(seed=3).event_types.binary, WorldStreams(3).event_tree)
    untouched = TaxonomyStreams(3)
    for name in ("base_rates", "rules", "superordinates", "tree", "instances", "analysis"):
        assert getattr(taxonomy_streams, name).random() == getattr(untouched, name).random()


def test_taxonomy_outputs_are_unchanged_when_only_binary_settings_change(tmp_path: Path) -> None:
    variants = [
        None,
        {},
        {"taxonomy": {"superordinates": 3, "depth": 3, "branching": 3}},
        {"taxonomy": {"depth": 1, "superordinates": 5}},
    ]
    baseline = None
    for i, binary in enumerate(variants):
        config = world_config(TINY_RELATIONS, event_types={"binary": binary})
        world = define(config)
        snapshot = _taxonomy_snapshot(world.taxonomy)
        if baseline is None:
            baseline = snapshot
        assert snapshot == baseline
        folder = world.write(tmp_path / f"world_{i}")
        taxonomy_files = {
            p.relative_to(folder / "taxonomy").as_posix(): p.read_bytes()
            for p in (folder / "taxonomy").rglob("*")
            if p.is_file() and p.name != "config.yaml"
        }
        if i == 0:
            first = taxonomy_files
        assert taxonomy_files == first
    assert generate_taxonomy(config.taxonomy_config()).rules.records() == baseline[0]


def test_build_statics_uses_the_static_streams_only() -> None:
    config = world_config(TINY_RELATIONS, seed=2)
    taxonomy = generate_taxonomy(config.taxonomy_config())
    streams = WorldStreams(2)
    statics = build_statics(taxonomy, config, streams, None)
    assert statics.event_tree is not None
    fresh = WorldStreams(2)
    for name in ("requirements", "event_tree", "constraints", "pairs"):
        assert getattr(streams, name).random() != getattr(fresh, name).random(), name
    for name in ("fluents", "initial", "preconditions", "effects", "stats", "episodes"):
        assert getattr(streams, name).random() == getattr(fresh, name).random(), name
    # The world's seeds for the static streams are the world's, computed as the taxonomy's are.
    assert streams.seeds()["world:event_tree"] == WorldStreams(2).seed("event_tree")
    assert WorldStreams(2).seed("event_tree") != TaxonomyStreams(2).seed("tree")
