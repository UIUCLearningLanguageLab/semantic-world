"""Projections: the one-place features every verb gives every object.

The agent projection ``CAN.<verb>`` is true for an object when some possible patient would make
the relation true with this object as agent. The patient projection ``CANBE.<verb>`` is true
when some possible agent would. "Possible" means any combination of features an object could
have, not only the objects that exist in the run.

The exact computation treats the unknown side as unknown. Once per relation it enumerates the
settings of the free binary features in the cones of the unknown side's binary literals (the
cone machinery of the fixed-by-rule test) together with, scalar by scalar, the intervals cut out
by the thresholds those cones and the constraints read. That gives every pattern the unknown
side's literals can take, with the interval each pattern lives in. For each known object, the
comparisons with its scalars split those intervals further, and the projection is true when some
resulting setting satisfies every constraint. When the enumeration would exceed
``2 ** CONE_ENUMERATION_LIMIT`` rows, the local test treats each unknown literal as an
independent variable and marks the projection approximate. The local test never says false when
the exact answer is true.

Extensional projections report whether the object actually has a partner among the run's
instances. Exposure chooses, from the ``taxonomy:constraints`` stream, which projections appear
as columns in ``instances.csv``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.taxonomy.config import Config
from semantic_world.taxonomy.constraints import (
    Constraint,
    Relations,
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
)
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import CONE_ENUMERATION_LIMIT, RuleSet, evaluate_feature

LOCAL_SAMPLES = 1 << 16


@dataclass(frozen=True)
class Projections:
    verb_labels: tuple[str, ...]
    agent: np.ndarray
    """Intensional ``CAN.<verb>``: shape ``(instances, verbs)``, bool."""
    patient: np.ndarray
    """Intensional ``CANBE.<verb>``: shape ``(instances, verbs)``, bool."""
    agent_approximate: np.ndarray
    """Per verb, whether the agent projection came from the local test."""
    patient_approximate: np.ndarray
    actual_agent: np.ndarray
    """Extensional: the instance has some actual patient among the run's instances."""
    actual_patient: np.ndarray
    exposed_agent: tuple[str, ...]
    """The verbs whose agent projection appears in ``instances.csv``, in verb order."""
    exposed_patient: tuple[str, ...]

    @property
    def agent_columns(self) -> tuple[str, ...]:
        return tuple(f"CAN.{v}" for v in self.verb_labels)

    @property
    def patient_columns(self) -> tuple[str, ...]:
        return tuple(f"CANBE.{v}" for v in self.verb_labels)

    def approximate_labels(self) -> list[str]:
        """The projections computed by the local test, as column names."""
        labels = [
            f"CAN.{v}" for v, a in zip(self.verb_labels, self.agent_approximate, strict=True) if a
        ]
        labels += [
            f"CANBE.{v}"
            for v, a in zip(self.verb_labels, self.patient_approximate, strict=True)
            if a
        ]
        return labels

    def exposed_columns(self) -> dict[str, np.ndarray]:
        """The exposed projections as ``instances.csv`` columns, agents then patients."""
        index = {v: i for i, v in enumerate(self.verb_labels)}
        columns: dict[str, np.ndarray] = {}
        for v in self.exposed_agent:
            columns[f"CAN.{v}"] = self.agent[:, index[v]]
        for v in self.exposed_patient:
            columns[f"CANBE.{v}"] = self.patient[:, index[v]]
        return columns

    def all_columns(self) -> dict[str, Any]:
        """Every column of ``projections.csv`` after the label: intensional projections, the
        approximate flag, then the extensional projections."""
        columns: dict[str, Any] = {}
        for i, v in enumerate(self.verb_labels):
            columns[f"CAN.{v}"] = self.agent[:, i]
        for i, v in enumerate(self.verb_labels):
            columns[f"CANBE.{v}"] = self.patient[:, i]
        columns["approximate"] = ";".join(self.approximate_labels())
        for i, v in enumerate(self.verb_labels):
            columns[f"ACTUAL_CAN.{v}"] = self.actual_agent[:, i]
        for i, v in enumerate(self.verb_labels):
            columns[f"ACTUAL_CANBE.{v}"] = self.actual_patient[:, i]
        return columns


# ---------------------------------------------------------------------------------------------
# Computing projections
# ---------------------------------------------------------------------------------------------


def compute_projections(
    config: Config,
    rules: RuleSet,
    relations: Relations,
    instances: Instances,
    rng: np.random.Generator,
    cone_limit: int = CONE_ENUMERATION_LIMIT,
) -> Projections:
    """Every intensional and extensional projection, and the exposure choice."""
    verbs = relations.verbs.verbs
    n = len(instances)
    agent = np.zeros((n, len(verbs)), dtype=bool)
    patient = np.zeros((n, len(verbs)), dtype=bool)
    agent_approximate = np.zeros(len(verbs), dtype=bool)
    patient_approximate = np.zeros(len(verbs), dtype=bool)
    actual_agent = np.zeros((n, len(verbs)), dtype=bool)
    actual_patient = np.zeros((n, len(verbs)), dtype=bool)
    for vi, verb in enumerate(verbs):
        constraints = relations.relation(verb).constraints
        agent[:, vi], agent_approximate[vi] = project(
            constraints, "p", rules, instances, cone_limit, rng
        )
        patient[:, vi], patient_approximate[vi] = project(
            constraints, "a", rules, instances, cone_limit, rng
        )
        matrix = relations.matrix(verb)
        actual_agent[:, vi] = matrix.any(axis=1)
        actual_patient[:, vi] = matrix.any(axis=0)
    labels = tuple(v.label for v in verbs)
    assert config.verbs is not None
    exposed_agent = _expose(labels, config.verbs.expose_agent, rng)
    exposed_patient = _expose(labels, config.verbs.expose_patient, rng)
    return Projections(
        labels,
        agent,
        patient,
        agent_approximate,
        patient_approximate,
        actual_agent,
        actual_patient,
        exposed_agent,
        exposed_patient,
    )


def _expose(
    labels: tuple[str, ...], proportion: float, rng: np.random.Generator
) -> tuple[str, ...]:
    count = int(np.floor(proportion * len(labels) + 0.5))
    if count == 0:
        return ()
    chosen = sorted(int(i) for i in rng.choice(len(labels), size=count, replace=False))
    return tuple(labels[i] for i in chosen)


# ---------------------------------------------------------------------------------------------
# One projection
# ---------------------------------------------------------------------------------------------


def _rows(iterable, width: int, dtype) -> np.ndarray:
    """A 2-D array from an iterable of equal-length tuples, correct when ``width`` is 0."""
    rows = list(iterable)
    if width == 0:
        return np.zeros((len(rows), 0), dtype=dtype)
    return np.array(rows, dtype=dtype).reshape(len(rows), width)


def _unique(items) -> list:
    seen: dict[str, Any] = {}
    for item in items:
        seen.setdefault(item.key, item)
    return list(seen.values())


def _interval_representatives(cuts: Sequence[float]) -> np.ndarray:
    """One value inside each interval between consecutive cuts (and beyond the ends). A value
    equal to a ``>`` threshold behaves like the interval below it, so open midpoints suffice."""
    if not cuts:
        return np.array([0.0])
    points = [cuts[0] - 1.0]
    points.extend((a + b) / 2.0 for a, b in zip(cuts[:-1], cuts[1:], strict=True))
    points.append(cuts[-1] + 1.0)
    return np.array(points)


class _UnknownSide:
    """What the unknown role's literals can look like, enumerated once per relation."""

    def __init__(
        self,
        constraints: tuple[Constraint, ...],
        unknown: str,
        rules: RuleSet,
        cone_limit: int,
        rng: np.random.Generator,
    ) -> None:
        self.unknown = unknown
        self.known = "a" if unknown == "p" else "p"
        self.binary = _unique(
            item
            for c in constraints
            for item in c.literals
            if isinstance(item, RoleFeature) and item.role == unknown
        )
        self.thresholds = _unique(
            item
            for c in constraints
            for item in c.literals
            if isinstance(item, RoleThreshold) and item.role == unknown
        )
        self.comparisons = _unique(
            item for c in constraints for item in c.literals if isinstance(item, ScalarComparison)
        )
        # Fixed cuts per unknown scalar: thresholds read by the cones and by the constraints.
        cuts: dict[int, set[float]] = {}
        for item in self.binary:
            for t in rules.thresholds_of(item.feature):
                cuts.setdefault(t.scalar, set()).add(t.threshold)
        for item in self.thresholds:
            cuts.setdefault(item.threshold.scalar, set()).add(item.threshold.threshold)
        for item in self.comparisons:
            cuts.setdefault(self.unknown_scalar(item), set())
        self.scalars = sorted(cuts)
        self.cuts = {s: sorted(cuts[s]) for s in self.scalars}
        self.patterns, self.intervals, self.approximate = self._enumerate(rules, cone_limit, rng)

    def unknown_scalar(self, item: ScalarComparison) -> int:
        return item.patient_scalar if self.unknown == "p" else item.agent_scalar

    def known_scalar(self, item: ScalarComparison) -> int:
        return item.agent_scalar if self.unknown == "p" else item.patient_scalar

    @property
    def literals(self) -> list:
        return self.binary + self.thresholds

    def _enumerate(
        self, rules: RuleSet, cone_limit: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Rows of (binary literal values, threshold literal values) with the interval index of
        every unknown scalar: exact when small enough, otherwise independent combinations."""
        features = rules.features
        cone: dict[int, Any] = {}
        for item in self.binary:
            for f in rules.cone(item.feature):
                cone[f.position] = f
        cone_features = [cone[p] for p in sorted(cone)]
        interval_counts = [len(self.cuts[s]) + 1 for s in self.scalars]
        combos = _rows(
            itertools.product(*[range(c) for c in interval_counts]), len(self.scalars), np.intp
        )
        rows = 2 ** len(cone_features) * combos.shape[0]
        n_literals = len(self.literals)
        if rows <= 2**cone_limit:
            grid = _rows(
                itertools.product((0, 1), repeat=len(cone_features)), len(cone_features), np.uint8
            )
            g_index = np.repeat(np.arange(grid.shape[0]), combos.shape[0])
            c_index = np.tile(np.arange(combos.shape[0]), grid.shape[0])
            free_values = np.zeros((rows, len(features.free)), dtype=np.uint8)
            free_index = {f.label: i for i, f in enumerate(features.free)}
            free_values[:, [free_index[f.label] for f in cone_features]] = grid[g_index]
            intervals = combos[c_index]
            scalar_values = np.zeros((rows, features.scalar_count), dtype=float)
            for k, s in enumerate(self.scalars):
                representatives = _interval_representatives(self.cuts[s])
                scalar_values[:, s - 1] = representatives[intervals[:, k]]
            by_output = {r.output.label: r for r in rules.rules}
            cache: dict[str, np.ndarray] = {}
            columns = [
                evaluate_feature(
                    item.feature, free_values, features, by_output, cache, None, scalar_values
                )
                for item in self.binary
            ]
            columns += [
                item.instance_values(free_values, scalar_values) for item in self.thresholds
            ]
            literal_rows = (
                np.stack(columns, axis=1).astype(np.intp)
                if columns
                else np.zeros((rows, 0), dtype=np.intp)
            )
            unique = np.unique(np.hstack([literal_rows, intervals]), axis=0)
            return unique[:, :n_literals].astype(np.uint8), unique[:, n_literals:], False
        # The local test: every literal independent, every interval combination possible.
        if n_literals <= cone_limit:
            literal_rows = _rows(itertools.product((0, 1), repeat=n_literals), n_literals, np.uint8)
        else:
            literal_rows = (rng.random((LOCAL_SAMPLES, n_literals)) < 0.5).astype(np.uint8)
        l_index = np.repeat(np.arange(literal_rows.shape[0]), combos.shape[0])
        c_index = np.tile(np.arange(combos.shape[0]), literal_rows.shape[0])
        return literal_rows[l_index], combos[c_index], True


TARGET_CELLS = 1 << 22
"""How many (instance, pattern, candidate) cells a chunk of the projection grid may hold."""


def project(
    constraints: tuple[Constraint, ...],
    unknown: str,
    rules: RuleSet,
    instances: Instances,
    cone_limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """The projection of every instance for a relation, with the role ``unknown`` (``p`` for the
    agent projection, ``a`` for the patient projection) treated as any possible object. Returns
    the bool column and whether the local test was used.

    The grid has one axis for the instances, one for the unknown side's literal patterns, and one
    per unknown scalar that a comparison reads, holding candidate values of that scalar for each
    instance. A cell counts only when the candidate value lies in the interval that the pattern
    was enumerated in. The projection is true when some cell satisfies every constraint.
    """
    n = len(instances)
    if not constraints:
        return np.ones(n, dtype=bool), False
    side = _UnknownSide(constraints, unknown, rules, cone_limit, rng)
    values, scalars = instances.values, instances.scalars
    known_literals = _unique(
        item
        for c in constraints
        for item in c.literals
        if isinstance(item, RoleFeature | RoleThreshold) and item.role == side.known
    )
    comparison_scalars = sorted({side.unknown_scalar(item) for item in side.comparisons})
    by_scalar = {
        s: [item for item in side.comparisons if side.unknown_scalar(item) == s]
        for s in comparison_scalars
    }
    axis_of_scalar = {s: 2 + j for j, s in enumerate(comparison_scalars)}
    n_axes = 2 + len(comparison_scalars)

    def shaped(array: np.ndarray, axis: int) -> np.ndarray:
        shape = [1] * n_axes
        shape[axis] = array.shape[-1]
        if array.ndim == 2:  # per instance and per candidate
            shape[0] = array.shape[0]
        return array.reshape(shape)

    n_patterns = side.patterns.shape[0]
    fixed_columns: dict[str, np.ndarray] = {}
    for k, item in enumerate(side.literals):
        fixed_columns[item.key] = shaped(side.patterns[:, k], 1)
    # Candidate values per instance for every unknown scalar a comparison reads.
    candidate_points: dict[int, np.ndarray] = {}
    for s in comparison_scalars:
        cuts = np.array(side.cuts[s], dtype=float)
        columns = [np.broadcast_to(cuts, (n, len(cuts)))]
        for item in by_scalar[s]:
            known_value = scalars[:, side.known_scalar(item) - 1]
            columns.append(_comparison_cuts_array(item, known_value, unknown))
        all_cuts = np.sort(np.concatenate(columns, axis=1), axis=1)
        candidate_points[s] = _candidates_array(all_cuts)
    cells_per_instance = n_patterns
    for s in comparison_scalars:
        cells_per_instance *= candidate_points[s].shape[1]
    chunk = max(1, TARGET_CELLS // max(cells_per_instance, 1))
    result = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk):
        rows = slice(start, min(start + chunk, n))
        m = rows.stop - rows.start
        columns = dict(fixed_columns)
        for item in known_literals:
            columns[item.key] = shaped(item.instance_values(values[rows], scalars[rows]), 0)
        consistent = np.ones((m,) + (1,) * (n_axes - 1), dtype=bool)
        for s in comparison_scalars:
            points = candidate_points[s][rows]
            axis = axis_of_scalar[s]
            interval_of_point = np.searchsorted(
                np.array(side.cuts[s], dtype=float), points, side="left"
            )
            pattern_interval = shaped(side.intervals[:, side.scalars.index(s)], 1)
            consistent = consistent & (shaped(interval_of_point, axis) == pattern_interval)
            for item in by_scalar[s]:
                columns[item.key] = shaped(
                    _comparison_values(item, points, scalars[rows], unknown).astype(np.uint8), axis
                )
        satisfied = consistent
        for c in constraints:
            satisfied = satisfied & _evaluate_broadcast(c, columns)
        result[rows] = satisfied.reshape(m, -1).any(axis=1)
    return result, side.approximate


def _evaluate_broadcast(constraint: Constraint, columns: dict[str, np.ndarray]) -> np.ndarray:
    """The constraint's skeleton over broadcastable literal columns."""
    if not constraint.literals:
        return np.array(bool(constraint.table.bits[0]))
    index = np.zeros((), dtype=np.int64)
    for k, key in enumerate(constraint.keys):
        bit = 1 << (constraint.arity - 1 - k)
        index = index + bit * columns[key].astype(np.int64)
    return np.asarray(constraint.table.bits, dtype=np.uint8)[index].astype(bool)


def _comparison_cuts_array(
    item: ScalarComparison, known_values: np.ndarray, unknown: str
) -> np.ndarray:
    """The cut values of a comparison for every instance: shape ``(instances, 1 or 2)``."""
    sign = -1.0 if unknown == "p" else 1.0
    cuts = [known_values + sign * item.low]
    if item.high is not None:
        cuts.append(known_values + sign * item.high)
    return np.stack(cuts, axis=1)


def _candidates_array(sorted_cuts: np.ndarray) -> np.ndarray:
    """For every row of sorted cuts: a value below the first, every cut, a value between each
    consecutive pair, and a value above the last."""
    below = sorted_cuts[:, :1] - 1.0
    above = sorted_cuts[:, -1:] + 1.0
    between = (sorted_cuts[:, :-1] + sorted_cuts[:, 1:]) / 2.0
    return np.concatenate([below, sorted_cuts, between, above], axis=1)


def _comparison_cuts(item: ScalarComparison, known_value: float, unknown: str) -> list[float]:
    """The values of the unknown scalar at which a comparison changes truth value."""
    if unknown == "p":
        # a - p > low  <=>  p < a - low;  window: a - high < p < a - low
        cuts = [known_value - item.low]
        if item.high is not None:
            cuts.append(known_value - item.high)
    else:
        # a - p > low  <=>  a > p + low;  window: p + low < a < p + high
        cuts = [known_value + item.low]
        if item.high is not None:
            cuts.append(known_value + item.high)
    return cuts


def _comparison_values(
    item: ScalarComparison, candidates: np.ndarray, known_scalars: np.ndarray, unknown: str
) -> np.ndarray:
    """The comparison at candidate values of the unknown scalar: ``candidates`` has one row per
    instance and ``known_scalars`` one row per instance (all scalars)."""
    if unknown == "p":
        difference = known_scalars[:, item.agent_scalar - 1][:, None] - candidates
    else:
        difference = candidates - known_scalars[:, item.patient_scalar - 1][:, None]
    return item.of_difference(difference)


__all__ = ["Projections", "compute_projections", "project"]
