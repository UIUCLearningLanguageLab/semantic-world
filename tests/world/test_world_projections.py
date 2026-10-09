"""Projections of the world package (stage a5b of ``docs/specs/WORLD_AND_LANGUAGE.md``): the
intensional and extensional capacities ``CAN.<event type>`` and ``CANBE.<event type>`` of every
two-place event type, checked against brute force, the local test, their columns in
``derived/capacities.csv``, and determinism. These tests moved from
``tests/taxonomy/test_taxonomy_projections.py`` (stage 11 of the taxonomy generator). Exposure
(``expose_agent``, ``expose_patient``, the exposed columns of ``instances.csv``) is gone: views
choose the columns a model sees.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.world.capacities import capacities_frame
from semantic_world.world.config import Config, config_from_mapping, load_config
from semantic_world.world.constraints import RoleThreshold, ScalarComparison
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.projections import compute_projections, project
from semantic_world.world.statics import StaticWorld, build_statics
from semantic_world.world.streams import WorldStreams

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"


def statics_of(config: Config) -> StaticWorld:
    """The static side of a world, without the dynamics."""
    taxonomy = generate_taxonomy(config.taxonomy_config())
    return build_statics(taxonomy, config, WorldStreams(config.seed), None)


def tiny_statics(seed: int = 1, **binary) -> StaticWorld:
    data = {
        "name": "tiny",
        "seed": seed,
        "taxonomy": {"config": str(TAXONOMY_DATA / "tiny_relations.yaml")},
        "event_types": {
            "unary": {"count": 4},
            "binary": {
                "features": {"count": 4, "expected_true": 2},
                "taxonomy": {"superordinates": 2, "depth": 2, "branching": 2},
                **binary,
            },
        },
    }
    return statics_of(config_from_mapping(data, source="<test>"))


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def default() -> StaticWorld:
    return statics_of(load_config(DATA / "default.yaml"))


# ---------------------------------------------------------------------------------------------
# Brute force over feature settings
# ---------------------------------------------------------------------------------------------


def _brute_force_projection(statics: StaticWorld, event_type, unknown: str) -> np.ndarray:
    """Enumerate every free binary setting of the unknown entity and a grid of scalar values
    that crosses every threshold and every comparison bound, and ask whether some setting makes
    the relation true with each instance on the known side."""
    rules = statics.taxonomy.rules
    features = rules.features
    instances = statics.instances
    constraints = statics.relations.relation(event_type).constraints
    n_free = len(features.free)
    grid = np.array(list(itertools.product((0, 1), repeat=n_free)), dtype=np.uint8).reshape(
        -1, n_free
    )
    k = features.scalar_count
    out = np.zeros(len(instances), dtype=bool)
    thresholds = [
        item.threshold.threshold
        for c in constraints
        for item in c.literals
        if isinstance(item, RoleThreshold)
    ]
    comparisons = [
        item for c in constraints for item in c.literals if isinstance(item, ScalarComparison)
    ]
    for x in range(len(instances)):
        cuts = set(thresholds)
        for item in comparisons:
            if unknown == "patient":
                known = instances.scalars[x, item.agent_scalar - 1]
                cuts.update(
                    [known - item.low] + ([known - item.high] if item.high is not None else [])
                )
            else:
                known = instances.scalars[x, item.patient_scalar - 1]
                cuts.update(
                    [known + item.low] + ([known + item.high] if item.high is not None else [])
                )
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
        other_values = rules.compute(rows_free, rows_scalar)
        m = other_values.shape[0]
        own_values = np.repeat(instances.values[x : x + 1], m, axis=0)
        own_scalars = np.repeat(instances.scalars[x : x + 1], m, axis=0)
        held = np.ones(m, dtype=bool)
        for c in constraints:
            if unknown == "patient":
                held &= c.evaluate(own_values, own_scalars, other_values, rows_scalar)
            else:
                held &= c.evaluate(other_values, rows_scalar, own_values, own_scalars)
        out[x] = held.any()
    return out


@pytest.mark.parametrize("seed", range(3))
def test_intensional_projections_match_brute_force(seed: int) -> None:
    statics = tiny_statics(seed)
    projections = statics.projections
    assert projections is not None
    event_types = statics.event_tree.event_types
    assert projections.event_type_labels == tuple(v.label for v in event_types)
    for vi, event_type in enumerate(event_types):
        assert np.array_equal(
            projections.agent[:, vi], _brute_force_projection(statics, event_type, "patient")
        ), (seed, event_type.label, "agent")
        assert np.array_equal(
            projections.patient[:, vi], _brute_force_projection(statics, event_type, "agent")
        ), (seed, event_type.label, "patient")
    assert not projections.agent_approximate.any()
    assert not projections.patient_approximate.any()
    assert projections.approximate_labels() == []


def test_projections_are_not_all_trivial(tiny: WorldResult, default: StaticWorld) -> None:
    both = np.concatenate([default.projections.agent.ravel(), default.projections.patient.ravel()])
    assert 0 < both.mean() < 1
    assert tiny.statics.projections.agent.shape == (12, 4)
    assert tiny.statics.projections.agent_columns == tuple(
        f"CAN.{v}" for v in tiny.statics.projections.event_type_labels
    )
    assert tiny.statics.projections.patient_columns == tuple(
        f"CANBE.{v}" for v in tiny.statics.projections.event_type_labels
    )


def test_extensional_projections_imply_intensional_ones(
    tiny: WorldResult, default: StaticWorld
) -> None:
    for statics in (tiny.statics, default):
        p = statics.projections
        assert np.all(p.actual_agent <= p.agent)
        assert np.all(p.actual_patient <= p.patient)
        for vi, event_type in enumerate(statics.event_tree.event_types):
            matrix = statics.relations.matrix(event_type)
            assert np.array_equal(p.actual_agent[:, vi], matrix.any(axis=1))
            assert np.array_equal(p.actual_patient[:, vi], matrix.any(axis=0))
        assert p.actual_agent.any() and p.actual_patient.any()


def test_empty_relation_projects_to_everything() -> None:
    statics = tiny_statics(1)
    for unknown in ("agent", "patient"):
        column, approximate = project(
            (), unknown, statics.rules, statics.instances, 20, np.random.default_rng(0)
        )
        assert column.all() and not approximate


def test_local_test_is_approximate_but_never_says_false_when_exact_is_true() -> None:
    for seed in range(3):
        statics = tiny_statics(seed)
        exact = statics.projections
        local = compute_projections(
            statics.rules,
            statics.relations,
            statics.instances,
            np.random.default_rng(0),
            cone_limit=-1,
        )
        assert local.agent_approximate.all() and local.patient_approximate.all()
        assert np.all(exact.agent <= local.agent)
        assert np.all(exact.patient <= local.patient)
        labels = local.event_type_labels
        assert local.approximate_labels() == [f"CAN.{v}" for v in labels] + [
            f"CANBE.{v}" for v in labels
        ]
        assert not exact.agent_approximate.any() and not exact.patient_approximate.any()
        assert local.all_columns()["approximate"] == ";".join(local.approximate_labels())


# ---------------------------------------------------------------------------------------------
# The capacities file
# ---------------------------------------------------------------------------------------------


def test_capacities_csv_columns(tiny: WorldResult, tmp_path: Path) -> None:
    folder = tiny.write(tmp_path / "run")
    capacities = pl.read_csv(folder / "derived" / "capacities.csv")
    p = tiny.statics.projections
    event_types = list(p.event_type_labels)
    one_place = list(tiny.statics.unary.labels)
    assert one_place == [f"EVENTTYPE1.{i}" for i in range(1, 5)]
    expected = (
        ["label"]
        + [f"CAN.{v}" for v in one_place]
        + [f"CAN.{v}" for v in event_types]
        + [f"CANBE.{v}" for v in event_types]
        + ["approximate"]
        + [f"ACTUAL_CAN.{v}" for v in event_types]
        + [f"ACTUAL_CANBE.{v}" for v in event_types]
    )
    assert capacities.columns == expected
    assert capacities.height == len(tiny.statics.instances)
    assert capacities["label"].to_list() == list(tiny.statics.instances.labels)
    # Nothing is approximate on the tiny world: the flag column is empty in every row.
    assert all(v in (None, "") for v in capacities["approximate"].to_list())
    for vi, v in enumerate(event_types):
        assert capacities[f"CAN.{v}"].to_numpy().astype(bool).tolist() == p.agent[:, vi].tolist()
        assert (
            capacities[f"CANBE.{v}"].to_numpy().astype(bool).tolist() == p.patient[:, vi].tolist()
        )
        assert (
            capacities[f"ACTUAL_CANBE.{v}"].to_numpy().astype(bool).tolist()
            == p.actual_patient[:, vi].tolist()
        )
    for label in one_place:
        assert capacities[f"CAN.{label}"].to_list() == tiny.statics.unary.column(label).tolist()
    assert set(np.unique(capacities.drop(["label", "approximate"]).to_numpy())) <= {0, 1}
    assert capacities_frame(tiny.statics).equals(tiny.derived_frames()["capacities.csv"])
    # No file of the run carries the old exposed columns.
    base = pl.read_csv(folder / "taxonomy" / "base.csv")
    assert not any(c.startswith(("CAN.", "CANBE.")) for c in base.columns)


def test_no_two_place_capacities_without_binary() -> None:
    config = config_from_mapping(
        {
            "name": "tiny",
            "seed": 1,
            "taxonomy": {"config": str(TAXONOMY_DATA / "tiny_relations.yaml")},
            "event_types": {"unary": {"count": 4}, "binary": None},
        },
        source="<test>",
    )
    statics = statics_of(config)
    assert statics.projections is None and statics.relations is None
    frame = capacities_frame(statics)
    assert frame.columns == ["label"] + [f"CAN.EVENTTYPE1.{i}" for i in range(1, 5)] + [
        "approximate"
    ]
    assert all(v == "" for v in frame["approximate"].to_list())


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def test_projections_are_deterministic_and_leave_the_pairs_stream_alone() -> None:
    a = tiny_statics(2)
    b = tiny_statics(2)
    assert np.array_equal(a.projections.agent, b.projections.agent)
    assert np.array_equal(a.projections.patient, b.projections.patient)
    assert np.array_equal(a.projections.actual_agent, b.projections.actual_agent)
    # The pair settings change the statistics only, never the projections or the constraints.
    more = tiny_statics(2, pairs={"sampled_true": 10, "sampled_false": 10})
    assert np.array_equal(more.projections.agent, a.projections.agent)
    assert more.relations.records() == a.relations.records()
    # Projections draw from the constraints stream (the local test's samples), never the pairs.
    streams = WorldStreams(2)
    compute_projections(a.rules, a.relations, a.instances, streams.constraints)
    assert streams.pairs.random() == WorldStreams(2).pairs.random()
    assert streams.constraints.random() == WorldStreams(2).constraints.random()  # exact here
    local = WorldStreams(2)
    compute_projections(a.rules, a.relations, a.instances, local.constraints, cone_limit=-1)
    assert local.pairs.random() == WorldStreams(2).pairs.random()
