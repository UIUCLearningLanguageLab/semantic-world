"""The leaf-pair density of two-place event types (stage a5b of
``docs/specs/WORLD_AND_LANGUAGE.md``; ``docs/proposals/2026-09-29-taxonomy-verb-density.md``).
These tests moved from ``tests/taxonomy/test_taxonomy_density.py`` (stage 12a of the taxonomy
generator).

The default changed with the stage: ``event_types.binary.density`` is ``{min: 0.0, max: 1.0,
max_tries: 200}``, which redraws only an event type whose requirement holds for no leaf pair or
for every leaf pair, and a constraint whose density is exactly 0 or 1 is resampled too;
``constraint_min_density`` (a floor above 0) defaults to null. The old settings (``density:
{min: 0.01, max: 0.3}``, ``constraint_min_density: 0.1``) keep working.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.world.config import Config, DensityConfig, config_from_mapping, load_config
from semantic_world.world.constraints import (
    Relations,
    generate_constraints,
    generate_relations,
    leaf_pair_density,
)
from semantic_world.world.event_tree import generate_event_tree
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.statics import StaticWorld, build_statics
from semantic_world.world.streams import WorldStreams

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"
RELATIONS = str(TAXONOMY_DATA / "relations.yaml")
TINY_RELATIONS = str(TAXONOMY_DATA / "tiny_relations.yaml")

CHECKS_OFF = {"density": None, "constraint_min_density": None}
OLD_CHECKS = {"density": {"min": 0.01, "max": 0.3}, "constraint_min_density": 0.1}
TINY_BINARY = {
    "features": {"count": 4, "expected_true": 2},
    "taxonomy": {"superordinates": 2, "depth": 2, "branching": 2},
}


def world_config(
    taxonomy: str | dict = RELATIONS,
    seed: int = 1,
    tmp_path: Path | None = None,
    **blocks,
) -> Config:
    """A world configuration over a taxonomy file, or over a taxonomy mapping written to
    ``tmp_path``."""
    if isinstance(taxonomy, dict):
        assert tmp_path is not None
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(taxonomy, sort_keys=False), encoding="utf-8")
        taxonomy = str(path)
    data = {"name": "test", "seed": seed, "taxonomy": {"config": taxonomy}, **blocks}
    return config_from_mapping(data, source="<test>")


def relations_of(config: Config) -> tuple[TaxonomyResult, Relations]:
    taxonomy = generate_taxonomy(config.taxonomy_config())
    streams = WorldStreams(config.seed)
    event_tree, relations = generate_relations(
        config.event_types.binary,
        taxonomy.config.scalars,
        taxonomy.rules,
        taxonomy.tree,
        taxonomy.instances,
        streams.constraints,
        streams.event_tree,
    )
    assert event_tree is not None and relations is not None
    return taxonomy, relations


def statics_of(config: Config) -> StaticWorld:
    taxonomy = generate_taxonomy(config.taxonomy_config())
    return build_statics(taxonomy, config, WorldStreams(config.seed), None)


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_density_defaults_and_null_forms() -> None:
    config = world_config()
    binary = config.event_types.binary
    assert binary.density is not None
    assert (binary.density.min, binary.density.max, binary.density.max_tries) == (0.0, 1.0, 200)
    assert binary.constraint_min_density is None
    assert binary.density_tries == 200
    resolved = config.resolved()["event_types"]["binary"]
    assert list(resolved)[-2:] == ["density", "constraint_min_density"]
    assert resolved["density"] == {"min": 0.0, "max": 1.0, "max_tries": 200}
    assert resolved["constraint_min_density"] is None
    off = world_config(event_types={"binary": CHECKS_OFF})
    assert off.event_types.binary.density is None
    assert off.event_types.binary.constraint_min_density is None
    assert off.resolved()["event_types"]["binary"]["density"] is None
    old = world_config(event_types={"binary": OLD_CHECKS})
    assert (old.event_types.binary.density.min, old.event_types.binary.density.max) == (0.01, 0.3)
    assert old.event_types.binary.constraint_min_density == 0.1
    assert config_from_mapping(config.resolved()).resolved() == config.resolved()


def test_a_density_of_zero_or_one_is_always_outside_the_range() -> None:
    default = DensityConfig(0.0, 1.0, 200)
    assert default.inside(0.5) and default.inside(1e-6) and default.inside(1 - 1e-6)
    assert not default.inside(0.0) and not default.inside(1.0)
    assert default.distance(0.5) == 0.0
    assert default.distance(0.0) == pytest.approx(1e-9) and default.distance(1.0) == pytest.approx(
        1e-9
    )
    old = DensityConfig(0.01, 0.3, 200)
    assert old.inside(0.1) and not old.inside(0.005) and not old.inside(0.4)
    assert old.distance(0.005) == pytest.approx(0.005)
    assert old.distance(0.4) == pytest.approx(0.1)
    assert old.distance(0.0) == pytest.approx(0.01)
    assert old.distance(1.0) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("binary", "field"),
    [
        ({"density": {"min": 0.5, "max": 0.2}}, "event_types.binary.density.min"),
        ({"density": {"max_tries": 0}}, "event_types.binary.density.max_tries"),
        ({"density": {"colour": 1}}, "event_types.binary.density.colour"),
        ({"density": {"min": -0.1}}, "event_types.binary.density.min"),
        ({"constraint_min_density": 1.5}, "event_types.binary.constraint_min_density"),
    ],
)
def test_broken_density_settings_name_the_field(binary: dict, field: str) -> None:
    with pytest.raises(ConfigError) as error:
        world_config(event_types={"binary": binary})
    assert error.value.field == field


# ---------------------------------------------------------------------------------------------
# Brute force over leaf pairs
# ---------------------------------------------------------------------------------------------


def _brute_force_density(taxonomy: TaxonomyResult, constraints) -> float:
    leaves = taxonomy.tree.leaves
    held = 0
    for a in leaves:
        for p in leaves:
            ok = True
            for c in constraints:
                ok &= bool(
                    c.evaluate(
                        a.values[None, :], a.scalars[None, :], p.values[None, :], p.scalars[None, :]
                    )[0]
                )
            held += ok
    return held / (len(leaves) ** 2)


def test_leaf_pair_densities_match_brute_force_on_tiny(tiny: WorldResult) -> None:
    statics = tiny.statics
    relations = statics.relations
    assert relations.constraint_density is not None and relations.event_type_density is not None
    for constraint in relations.constraints:
        assert relations.constraint_density[constraint.label] == pytest.approx(
            _brute_force_density(statics.taxonomy, [constraint])
        )
    for event_type in statics.event_tree.event_types:
        relation = relations.relation(event_type)
        assert relations.event_type_density[event_type.label] == pytest.approx(
            _brute_force_density(statics.taxonomy, relation.constraints)
        )
    # The measure uses leaves paired with themselves too, and never the instances.
    leaves = statics.taxonomy.tree.leaves
    values = np.stack([leaf.values for leaf in leaves])
    scalars = np.stack([leaf.scalars for leaf in leaves])
    for event_type in statics.event_tree.event_types:
        assert (
            leaf_pair_density(relations.relation(event_type).constraints, values, scalars)
            == relations.event_type_density[event_type.label]
        )
    assert np.isnan(leaf_pair_density([], values[:0], scalars[:0]))


def test_densities_are_reported_in_the_outputs(tiny: WorldResult, tmp_path: Path) -> None:
    folder = tiny.write(tmp_path / "run")
    stats = pl.read_csv(folder / "derived" / "event_type_stats.csv")
    assert stats.columns[-2:] == ["leaf_pair_density", "tries"]
    relations = tiny.statics.relations
    for row in stats.iter_rows(named=True):
        assert row["leaf_pair_density"] == pytest.approx(
            relations.event_type_density[row["event_type"]], abs=1e-6
        )
        assert 1 <= row["tries"] <= 200
        assert row["tries"] == relations.event_type_tries[row["event_type"]]
    assert all("leaf_pair_density" in c for c in relations.records())
    world_stats = yaml.safe_load((folder / "world_stats.yaml").read_text())
    for label in relations.outside_range:
        assert any(label in w for w in world_stats["warnings"])
    assert all(w in world_stats["warnings"] for w in relations.warnings)


# ---------------------------------------------------------------------------------------------
# The range across seeds
# ---------------------------------------------------------------------------------------------


def _survey(binary: dict, seeds=range(10)) -> dict[str, float]:
    """Over ten seeds of the default world: the share of event types inside the range, the share
    with no true instance pair, and the checks that every event type outside the range and every
    constraint below the floor is warned."""
    inside = total = 0
    proportions = []
    for seed in seeds:
        config = world_config(seed=seed, event_types={"binary": binary})
        taxonomy, relations = relations_of(config)
        density = config.event_types.binary.density
        floor = config.event_types.binary.constraint_min_density
        n = len(taxonomy.instances)
        for event_type in relations.event_tree.event_types:
            total += 1
            value = relations.event_type_density[event_type.label]
            if density.inside(value):
                inside += 1
                assert event_type.label not in relations.outside_range
            else:
                assert event_type.label in relations.outside_range
                assert any(
                    event_type.label in w and f"{value:.4f}" in w for w in relations.warnings
                )
            proportions.append(relations.matrix(event_type).sum() / (n * (n - 1)))
        # The floor and the degeneracy check apply to the event-type features' constraints.
        for constraint in relations.feature_constraints:
            value = relations.constraint_density[constraint.label]
            if value <= 0.0 or value >= 1.0 or (floor is not None and value < floor):
                assert any(constraint.label in w for w in relations.warnings)
    p = np.array(proportions)
    return {
        "event_types": total,
        "inside": inside / total,
        "zero_pairs": float(np.mean(p == 0)),
        "below_half_percent": float(np.mean(p < 0.005)),
        "median": float(np.median(p)),
        "above_five_percent": float(np.mean(p > 0.05)),
    }


def _print(label: str, s: dict[str, float]) -> None:
    print(
        f"\n{label}: event types {s['event_types']}; inside range {s['inside']:.0%}; "
        f"zero pairs {s['zero_pairs']:.0%}; below 0.5% {s['below_half_percent']:.0%}; "
        f"median {s['median']:.2%}; above 5% {s['above_five_percent']:.0%}"
    )


def test_every_event_type_is_inside_the_default_range_or_warned_across_ten_seeds() -> None:
    """Under the default, inside means a leaf-pair density strictly between 0 and 1."""
    survey = _survey({})
    _print("DEFAULT", survey)
    assert survey["inside"] > 0.5


def test_the_old_checks_keep_working_across_ten_seeds() -> None:
    """The old range and floor are still settings, with the old acceptance values."""
    survey = _survey(OLD_CHECKS)
    _print("OLD CHECKS", survey)
    assert survey["inside"] > 0.5
    assert survey["zero_pairs"] < 0.15


def test_degenerate_constraints_and_event_types_are_resampled_or_warned() -> None:
    """A configuration that makes degenerate constraints likely: an AND of four agent features
    over the tiny taxonomy's four leaves. With one try, every degenerate constraint and event
    type is warned; with the default tries, the constraints are brought out of degeneracy."""
    hard = {
        **TINY_BINARY,
        "rules": {"arity": {4: 1}, "operator_mix": {"AND": 1}, "negation_probability": 0},
        "constraint_families": {
            "agent": 1,
            "patient": 0,
            "cross": 0,
            "key_lock": 0,
            "comparison": 0,
        },
    }
    _, one_try = relations_of(
        world_config(TINY_RELATIONS, event_types={"binary": {**hard, "density": {"max_tries": 1}}})
    )
    degenerate = [
        c.label
        for c in one_try.feature_constraints
        if not 0.0 < one_try.constraint_density[c.label] < 1.0
    ]
    assert degenerate
    for label in degenerate:
        assert any(
            label in w and "no leaf pair or for every leaf pair" in w for w in one_try.warnings
        )
    for event_type in one_try.event_tree.event_types:
        value = one_try.event_type_density[event_type.label]
        assert (0.0 < value < 1.0) == (event_type.label not in one_try.outside_range)
    assert one_try.outside_range
    _, many_tries = relations_of(world_config(TINY_RELATIONS, event_types={"binary": hard}))
    for constraint in many_tries.feature_constraints:
        assert 0.0 < many_tries.constraint_density[constraint.label] < 1.0
    assert not any("constraint " in w for w in many_tries.warnings)
    assert set(many_tries.outside_range) < set(one_try.outside_range)


# ---------------------------------------------------------------------------------------------
# Independence and the unchanged build with the checks off
# ---------------------------------------------------------------------------------------------


def _tree_snapshot(relations: Relations) -> list:
    return [
        (c.label, c.values.tobytes(), c.roles.tobytes()) for c in relations.event_tree.categories
    ]


def test_instance_count_leaves_constraints_relations_and_event_type_outputs_unchanged(
    tmp_path: Path,
) -> None:
    base = yaml.safe_load(Path(RELATIONS).read_text(encoding="utf-8"))
    a = statics_of(world_config({**base, "instances": {"per_leaf": 3}}, tmp_path=tmp_path / "a"))
    b = statics_of(
        world_config({**base, "instances": {"per_leaf": [6, 9]}}, tmp_path=tmp_path / "b")
    )
    assert len(a.instances) < len(b.instances)
    assert a.relations.records() == b.relations.records()
    assert a.relations.relation_records() == b.relations.relation_records()
    assert _tree_snapshot(a.relations) == _tree_snapshot(b.relations)
    columns = ["event_type", "constraints", "families", "leaf_pair_density", "tries"]
    stats_a = a.relation_stats.event_type_stats.select(columns)
    stats_b = b.relation_stats.event_type_stats.select(columns)
    assert stats_a.equals(stats_b)
    assert not a.relation_stats.event_type_stats.equals(b.relation_stats.event_type_stats)


def test_with_both_checks_off_the_build_equals_the_plain_one() -> None:
    """``generate_relations`` with the checks off draws exactly what ``generate_event_tree`` and
    ``generate_constraints`` draw, so turning the checks off changes nothing else."""
    for seed in range(3):
        config = world_config(seed=seed, event_types={"binary": CHECKS_OFF})
        taxonomy, full = relations_of(config)
        streams = WorldStreams(seed)
        binary = config.event_types.binary
        event_tree = generate_event_tree(binary, streams.event_tree)
        plain = generate_constraints(
            binary,
            taxonomy.config.scalars,
            taxonomy.rules.features,
            event_tree,
            taxonomy.instances,
            streams.constraints,
        )
        assert plain is not None
        assert full.records() == plain.records()
        assert full.relation_records() == plain.relation_records()
        assert _tree_snapshot(full) == _tree_snapshot(plain)
        assert full.constraint_density is None and full.event_type_density is None
        assert full.event_type_tries is None and full.outside_range == ()
        assert full.warnings == ()


def test_checks_on_change_the_event_types_but_keep_them_distinct() -> None:
    _, on = relations_of(world_config(event_types={"binary": OLD_CHECKS}))
    _, off = relations_of(world_config(event_types={"binary": CHECKS_OFF}))
    assert on.records() != off.records()
    event_types = on.event_tree.event_types
    vectors = [v.values.tobytes() for v in event_types]
    assert len(set(vectors)) == len(vectors)
    # Defining features never change under a redraw.
    for event_type in event_types:
        for ancestor in event_type.ancestors():
            mask = ancestor.defining_mask()
            assert np.array_equal(event_type.free_values[mask], ancestor.free_values[mask])
    assert on.event_type_density is not None and off.event_type_density is None
    assert on.constraint_density is not None and off.constraint_density is None
    # The event-type features and their roles are drawn before any tuning: equal in both.
    for a, b in zip(on.event_tree.categories, off.event_tree.categories, strict=True):
        assert a.label == b.label and np.array_equal(a.roles, b.roles)
        if not a.is_leaf:
            assert np.array_equal(a.values, b.values)


def test_same_seed_gives_the_same_tuned_result(tmp_path: Path) -> None:
    a = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "a")
    b = define(load_config(DATA / "tiny.yaml")).write(tmp_path / "b")
    names = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    for name in names:
        if Path(name).name != "config.yaml":
            assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_constraint_floor_alone() -> None:
    config = world_config(event_types={"binary": {"density": None, "constraint_min_density": 0.2}})
    statics = statics_of(config)
    relations = statics.relations
    assert relations.constraint_density is not None and relations.event_type_density is None
    for constraint in relations.feature_constraints:
        value = relations.constraint_density[constraint.label]
        assert value >= 0.2 or any(constraint.label in w for w in relations.warnings)
    assert relations.outside_range == ()
    assert "leaf_pair_density" not in statics.relation_stats.event_type_stats.columns
    assert "tries" not in statics.relation_stats.event_type_stats.columns
