"""Stage 11 acceptance tests: intensional and extensional projections, and exposure."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from semantic_world.taxonomy import (
    Streams,
    TaxonomyResult,
    config_from_mapping,
    generate,
    load_config,
)
from semantic_world.taxonomy.constraints import RoleThreshold, ScalarComparison
from semantic_world.taxonomy.projections import compute_projections, project

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"


def tiny_config(seed: int = 1, **overrides):
    base = load_config(DATA / "tiny_relations.yaml").resolved()
    base.pop("provenance", None)
    for key, value in overrides.items():
        base[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    return config_from_mapping(base, seed=seed)


@pytest.fixture(scope="module")
def tiny() -> TaxonomyResult:
    return generate(load_config(DATA / "tiny_relations.yaml"))


@pytest.fixture(scope="module")
def relations_run() -> TaxonomyResult:
    return generate(load_config(DATA / "relations.yaml"))


# ---------------------------------------------------------------------------------------------
# Brute force over feature settings
# ---------------------------------------------------------------------------------------------


def _brute_force_projection(result: TaxonomyResult, verb, unknown: str) -> np.ndarray:
    """Enumerate every free binary setting of the unknown object and a grid of scalar values
    that crosses every threshold and every comparison bound, and ask whether some setting makes
    the relation true with each instance on the known side."""
    rules = result.rules
    features = rules.features
    instances = result.instances
    constraints = result.relations.relation(verb).constraints
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
            if unknown == "p":
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
            if unknown == "p":
                held &= c.evaluate(own_values, own_scalars, other_values, rows_scalar)
            else:
                held &= c.evaluate(other_values, rows_scalar, own_values, own_scalars)
        out[x] = held.any()
    return out


@pytest.mark.parametrize("seed", range(3))
def test_intensional_projections_match_brute_force(seed: int) -> None:
    result = generate(tiny_config(seed))
    projections = result.projections
    assert projections is not None
    for vi, verb in enumerate(result.verbs.verbs):
        assert np.array_equal(
            projections.agent[:, vi], _brute_force_projection(result, verb, "p")
        ), (seed, verb.label, "agent")
        assert np.array_equal(
            projections.patient[:, vi], _brute_force_projection(result, verb, "a")
        ), (seed, verb.label, "patient")
    assert not projections.agent_approximate.any()
    assert not projections.patient_approximate.any()
    assert projections.approximate_labels() == []


def test_projections_are_not_all_trivial(
    tiny: TaxonomyResult, relations_run: TaxonomyResult
) -> None:
    both = np.concatenate(
        [relations_run.projections.agent.ravel(), relations_run.projections.patient.ravel()]
    )
    assert 0 < both.mean() < 1
    assert tiny.projections.agent.shape == (12, 4)


def test_extensional_projections_imply_intensional_ones(
    tiny: TaxonomyResult, relations_run: TaxonomyResult
) -> None:
    for result in (tiny, relations_run):
        p = result.projections
        assert np.all(p.actual_agent <= p.agent)
        assert np.all(p.actual_patient <= p.patient)
        for vi, verb in enumerate(result.verbs.verbs):
            matrix = result.relations.matrix(verb)
            assert np.array_equal(p.actual_agent[:, vi], matrix.any(axis=1))
            assert np.array_equal(p.actual_patient[:, vi], matrix.any(axis=0))
        assert p.actual_agent.any() and p.actual_patient.any()


def test_empty_relation_projects_to_everything() -> None:
    result = generate(tiny_config(1))
    rules = result.rules
    column, approximate = project((), "p", rules, result.instances, 20, np.random.default_rng(0))
    assert column.all() and not approximate


def test_local_test_is_approximate_but_never_says_false_when_exact_is_true() -> None:
    for seed in range(3):
        result = generate(tiny_config(seed))
        exact = result.projections
        local = compute_projections(
            result.config,
            result.rules,
            result.relations,
            result.instances,
            np.random.default_rng(0),
            cone_limit=-1,
        )
        assert local.agent_approximate.all() and local.patient_approximate.all()
        assert np.all(exact.agent <= local.agent)
        assert np.all(exact.patient <= local.patient)
        assert local.approximate_labels() == [f"CAN.{v}" for v in local.verb_labels] + [
            f"CANBE.{v}" for v in local.verb_labels
        ]
        assert not exact.agent_approximate.any() and not exact.patient_approximate.any()


# ---------------------------------------------------------------------------------------------
# Exposure and outputs
# ---------------------------------------------------------------------------------------------


def test_exposure_counts_follow_the_proportions(relations_run: TaxonomyResult) -> None:
    p = relations_run.projections
    n_verbs = len(p.verb_labels)
    assert p.exposed_agent == p.verb_labels  # expose_agent 1.0
    assert len(p.exposed_patient) == int(np.floor(0.25 * n_verbs + 0.5))
    assert list(p.exposed_patient) == sorted(p.exposed_patient, key=p.verb_labels.index)
    none = generate(
        config_from_mapping(
            {
                "scalars": {"count": 2},
                "verbs": {"projections": {"expose_agent": 0, "expose_patient": 0}},
            }
        )
    )
    assert none.projections.exposed_agent == () and none.projections.exposed_patient == ()
    assert none.projections.exposed_columns() == {}


def test_exposed_columns_in_instances_csv_equal_projections_csv(
    relations_run: TaxonomyResult, tmp_path: Path
) -> None:
    folder = relations_run.write(tmp_path / "run")
    instances = pl.read_csv(folder / "instances.csv")
    projections = pl.read_csv(folder / "projections.csv")
    p = relations_run.projections
    exposed = [f"CAN.{v}" for v in p.exposed_agent] + [f"CANBE.{v}" for v in p.exposed_patient]
    assert exposed
    # Exposed columns come after the scalar columns, in agent-then-patient order.
    assert instances.columns[-len(exposed) :] == exposed
    assert instances.columns[-len(exposed) - 1] == "SC.2"
    for name in exposed:
        assert instances[name].to_list() == projections[name].to_list(), name
    assert projections["label"].to_list() == instances["label"].to_list()
    unexposed = [f"CANBE.{v}" for v in p.verb_labels if v not in p.exposed_patient]
    assert unexposed and not any(name in instances.columns for name in unexposed)


def test_projections_csv_columns(relations_run: TaxonomyResult, tmp_path: Path) -> None:
    folder = relations_run.write(tmp_path / "run")
    projections = pl.read_csv(folder / "projections.csv")
    p = relations_run.projections
    verbs = list(p.verb_labels)
    expected = (
        ["label"]
        + [f"CAN.{v}" for v in verbs]
        + [f"CANBE.{v}" for v in verbs]
        + ["approximate"]
        + [f"ACTUAL_CAN.{v}" for v in verbs]
        + [f"ACTUAL_CANBE.{v}" for v in verbs]
    )
    assert projections.columns == expected
    assert projections.height == len(relations_run.instances)
    # Nothing is approximate on this example: the flag column is empty in every row.
    assert all(v in (None, "") for v in projections["approximate"].to_list())
    for vi, v in enumerate(verbs):
        assert projections[f"CAN.{v}"].to_numpy().astype(bool).tolist() == p.agent[:, vi].tolist()
        assert (
            projections[f"ACTUAL_CANBE.{v}"].to_numpy().astype(bool).tolist()
            == p.actual_patient[:, vi].tolist()
        )
    assert set(np.unique(projections.drop(["label", "approximate"]).to_numpy())) <= {0, 1}


def test_no_projection_files_without_verbs(tmp_path: Path) -> None:
    result = generate(config_from_mapping({"scalars": {"count": 2}}))
    assert result.projections is None
    folder = result.write(tmp_path / "run")
    assert not (folder / "projections.csv").exists()
    assert not any(
        c.startswith(("CAN.V", "CANBE.")) for c in pl.read_csv(folder / "instances.csv").columns
    )


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def test_projections_are_deterministic_and_exposure_uses_the_constraints_stream() -> None:
    a = generate(tiny_config(2))
    b = generate(tiny_config(2))
    assert np.array_equal(a.projections.agent, b.projections.agent)
    assert a.projections.exposed_patient == b.projections.exposed_patient
    # Exposure is drawn after the constraints, so changing it leaves the constraints unchanged.
    more = generate(
        tiny_config(2, verbs={"projections": {"expose_agent": 0.5, "expose_patient": 1.0}})
    )
    assert more.relations.records() == a.relations.records()
    assert more.projections.exposed_patient == more.projections.verb_labels
    assert len(more.projections.exposed_agent) == 2
    # Projections do not touch the pairs stream.
    config = tiny_config(2)
    streams = Streams(config.seed)
    generate(config)
    assert streams.pairs.random() == Streams(config.seed).pairs.random()
