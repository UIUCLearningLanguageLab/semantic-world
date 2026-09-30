"""Constraints and relations: two-argument relations between instances.

A verb ``v`` is a relation ``R_v(a, p)`` between an agent instance ``a`` and a patient instance
``p``: a conjunction of constraints. Each constraint is a rule over the features of one or both
arguments. Its inputs are binary IS and HAS features (free or determined) and scalar threshold
literals from either argument, named with the prefixes ``a.`` and ``p.``, and scalar comparisons
between the two arguments. No constraint reads an ISA feature, a CAN feature, or a projection.
An instance is never related to itself.

Every constraint is stored as its Boolean skeleton (a truth table over its literals) plus the
definitions of its literals, and printed expressions parse back to the same constraint. Every
verb feature ``VF.<n>`` gets one constraint ``K.VF.<n>``, and, with ``verbs.own_constraint`` on,
every verb gets one of its own, ``K.<verb label>``. A verb's relation is the conjunction of the
constraints of its true verb features plus its own constraint. Every verb category has a base
relation: the constraints of the verb features that are defining with value 1 at the category.
Constraint sampling draws from the ``taxonomy:constraints`` stream.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from semantic_world.taxonomy.boolean import TruthTable, dnf_literal_count
from semantic_world.taxonomy.config import Config, RuleSampling, ScalarsConfig, VerbsConfig
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.expressions import Cmp, Expr, Gt, Op, Var, literal
from semantic_world.taxonomy.features import Feature, FeatureSet
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import (
    MAX_TRIES,
    Threshold,
    _draw,
    build_function,
    model_quantile_threshold,
)
from semantic_world.taxonomy.streams import Streams
from semantic_world.taxonomy.tree import Category
from semantic_world.taxonomy.verbs import VerbTaxonomy

ROLES = ("a", "p")
FAMILIES = ("agent", "patient", "cross", "key_lock", "comparison")
CHUNK_ROWS = 2048


# ---------------------------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleFeature:
    """A binary feature of the agent (``a.IS.7``) or the patient (``p.HAS.4``)."""

    role: str
    feature: Feature

    @property
    def key(self) -> str:
        return f"{self.role}.{self.feature.label}"

    def atom(self) -> Expr:
        return Var(self.key)

    def instance_values(self, values: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        return values[:, self.feature.position]


@dataclass(frozen=True)
class RoleThreshold:
    """A threshold literal on a scalar of the agent or the patient: ``p.SC.1 > 0.2031``."""

    role: str
    threshold: Threshold

    @property
    def key(self) -> str:
        return f"{self.role}.{self.threshold.key}"

    def atom(self) -> Expr:
        return Gt(f"{self.role}.{self.threshold.label}", self.threshold.threshold)

    def instance_values(self, values: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        return self.threshold.values(scalars)


@dataclass(frozen=True)
class ScalarComparison:
    """An order ``a.SC.i - p.SC.j > low`` (``high`` None) or a window
    ``low < a.SC.i - p.SC.j < high`` between the arguments' scalars."""

    agent_scalar: int
    patient_scalar: int
    low: float
    high: float | None

    def atom(self) -> Cmp:
        return Cmp(f"a.SC.{self.agent_scalar}", f"p.SC.{self.patient_scalar}", self.low, self.high)

    @property
    def key(self) -> str:
        return self.atom().key

    def of_difference(self, difference: np.ndarray) -> np.ndarray:
        if self.high is None:
            return difference > self.low
        return (difference > self.low) & (difference < self.high)

    def pair_values(self, agent_scalars: np.ndarray, patient_scalars: np.ndarray) -> np.ndarray:
        difference = (
            agent_scalars[:, self.agent_scalar - 1] - patient_scalars[:, self.patient_scalar - 1]
        )
        return self.of_difference(difference)


ConstraintLiteral = RoleFeature | RoleThreshold | ScalarComparison


def literal_role(item: ConstraintLiteral) -> str:
    """``a``, ``p``, or ``both`` for a comparison."""
    return "both" if isinstance(item, ScalarComparison) else item.role


# ---------------------------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    label: str
    family: str
    literals: tuple[ConstraintLiteral, ...]
    table: TruthTable
    expression: Expr
    min_dnf_literals: int

    @property
    def arity(self) -> int:
        return len(self.literals)

    @property
    def keys(self) -> list[str]:
        return [item.key for item in self.literals]

    @property
    def agent_literals(self) -> tuple[ConstraintLiteral, ...]:
        return tuple(item for item in self.literals if literal_role(item) == "a")

    @property
    def patient_literals(self) -> tuple[ConstraintLiteral, ...]:
        return tuple(item for item in self.literals if literal_role(item) == "p")

    @property
    def comparisons(self) -> tuple[ScalarComparison, ...]:
        return tuple(item for item in self.literals if isinstance(item, ScalarComparison))

    def depends_on_both_roles(self) -> bool:
        """Whether the skeleton depends on at least one agent input and one patient input (a
        comparison counts for both)."""
        roles = {literal_role(self.literals[i]) for i in self.table.relevant_inputs()}
        return ("a" in roles or "both" in roles) and ("p" in roles or "both" in roles)

    def pair_values(
        self, values: np.ndarray, scalars: np.ndarray, agents: np.ndarray, patients: np.ndarray
    ) -> np.ndarray:
        """The constraint for aligned pairs ``(agents[k], patients[k])``, as a bool array."""
        agents = np.asarray(agents, dtype=np.intp)
        patients = np.asarray(patients, dtype=np.intp)
        columns = []
        for item in self.literals:
            if isinstance(item, ScalarComparison):
                columns.append(
                    item.pair_values(scalars[agents], scalars[patients]).astype(np.uint8)
                )
            else:
                column = item.instance_values(values, scalars)
                columns.append(column[agents if item.role == "a" else patients].astype(np.uint8))
        if not columns:
            return np.full(len(agents), bool(self.table.bits[0]))
        return self.table.evaluate(np.stack(columns, axis=1)).astype(bool)

    def matrix(self, values: np.ndarray, scalars: np.ndarray, agents: np.ndarray) -> np.ndarray:
        """The constraint for every pair of an agent in ``agents`` and any instance as patient:
        shape ``(len(agents), instances)``. Each literal is evaluated once per instance and the
        agent and patient columns are combined by broadcasting."""
        agents = np.asarray(agents, dtype=np.intp)
        n = values.shape[0]
        index = np.zeros((len(agents), n), dtype=np.int64)
        for k, item in enumerate(self.literals):
            bit = 1 << (self.arity - 1 - k)
            if isinstance(item, ScalarComparison):
                difference = (
                    scalars[agents, item.agent_scalar - 1][:, None]
                    - scalars[:, item.patient_scalar - 1][None, :]
                )
                index += bit * item.of_difference(difference).astype(np.int64)
            else:
                column = item.instance_values(values, scalars).astype(np.int64)
                if item.role == "a":
                    index += bit * column[agents][:, None]
                else:
                    index += bit * column[None, :]
        return np.asarray(self.table.bits, dtype=np.uint8)[index].astype(bool)

    def record(self) -> dict[str, Any]:
        """The entry written to ``constraints.yaml``."""
        return {
            "label": self.label,
            "family": self.family,
            "agent_literals": [_literal_record(item) for item in self.agent_literals],
            "patient_literals": [_literal_record(item) for item in self.patient_literals],
            "comparisons": [
                {
                    "agent": f"SC.{c.agent_scalar}",
                    "patient": f"SC.{c.patient_scalar}",
                    "low": c.low,
                    "high": c.high,
                }
                for c in self.comparisons
            ],
            "expression": str(self.expression),
            "truth_table": self.table.bit_string(),
            "min_dnf_literals": self.min_dnf_literals,
        }


def _literal_record(item: ConstraintLiteral) -> dict[str, Any]:
    if isinstance(item, RoleThreshold):
        return {"scalar": item.threshold.label, "threshold": item.threshold.threshold}
    assert isinstance(item, RoleFeature)
    return {"feature": item.feature.label}


# ---------------------------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Relation:
    """The relation of a verb (``base`` False) or the base relation of a verb category."""

    label: str
    base: bool
    constraints: tuple[Constraint, ...]

    @property
    def expression(self) -> str:
        if not self.constraints:
            return "TRUE"
        if len(self.constraints) == 1:
            return str(self.constraints[0].expression)
        return " AND ".join(f"({c.expression})" for c in self.constraints)

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "base": self.base,
            "constraints": [c.label for c in self.constraints],
            "expression": self.expression,
        }


class Relations:
    """Every constraint and every relation of a run, with evaluation over the run's instances."""

    def __init__(
        self,
        verbs: VerbTaxonomy,
        feature_constraints: Sequence[Constraint],
        own_constraints: dict[str, Constraint],
        instances: Instances,
    ) -> None:
        self.verbs = verbs
        self.feature_constraints = tuple(feature_constraints)
        self.own_constraints = dict(own_constraints)
        self.instances = instances
        self.constraints: tuple[Constraint, ...] = self.feature_constraints + tuple(
            self.own_constraints[v.label] for v in verbs.verbs if v.label in self.own_constraints
        )
        self._by_label = {c.label: c for c in self.constraints}

    def constraint(self, label: str) -> Constraint:
        try:
            return self._by_label[label]
        except KeyError:
            raise KeyError(f"unknown constraint {label!r}") from None

    def category(self, verb: Category | str) -> Category:
        return verb if isinstance(verb, Category) else self.verbs.tree[verb]

    def relation(self, verb: Category | str) -> Relation:
        """A verb's relation, or the base relation of an internal verb category."""
        category = self.category(verb)
        if category.is_leaf:
            mask = category.values == 1
            constraints = [self.feature_constraints[i] for i in np.flatnonzero(mask)]
            own = self.own_constraints.get(category.label)
            if own is not None:
                constraints.append(own)
            return Relation(category.label, False, tuple(constraints))
        mask = category.defining_mask() & (category.free_values == 1)
        return Relation(
            category.label, True, tuple(self.feature_constraints[i] for i in np.flatnonzero(mask))
        )

    def relations(self) -> list[Relation]:
        return [self.relation(c) for c in self.verbs.categories]

    def holds(self, verb: Category | str, agents: np.ndarray, patients: np.ndarray) -> np.ndarray:
        """Whether the relation holds for the aligned pairs ``(agents[k], patients[k])`` of
        instance indices. An instance is never related to itself."""
        agents = np.asarray(agents, dtype=np.intp)
        patients = np.asarray(patients, dtype=np.intp)
        result = agents != patients
        values, scalars = self.instances.values, self.instances.scalars
        for constraint in self.relation(verb).constraints:
            result &= constraint.pair_values(values, scalars, agents, patients)
        return result

    def matrix(self, verb: Category | str, chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
        """The relation over every ordered pair of instances, as an ``(n, n)`` bool array with a
        False diagonal, computed in chunks of agents."""
        n = len(self.instances)
        values, scalars = self.instances.values, self.instances.scalars
        constraints = self.relation(verb).constraints
        out = np.zeros((n, n), dtype=bool)
        for start in range(0, n, chunk_rows):
            agents = np.arange(start, min(start + chunk_rows, n))
            block = np.ones((len(agents), n), dtype=bool)
            for constraint in constraints:
                block &= constraint.matrix(values, scalars, agents)
            block[np.arange(len(agents)), agents] = False
            out[start : start + len(agents)] = block
        return out

    def records(self) -> list[dict[str, Any]]:
        return [c.record() for c in self.constraints]

    def relation_records(self) -> list[dict[str, Any]]:
        return [r.record() for r in self.relations()]


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------


def generate_constraints(
    config: Config,
    features: FeatureSet,
    verbs: VerbTaxonomy | None,
    instances: Instances,
    streams: Streams,
) -> Relations | None:
    """Sample one constraint per verb feature and, when on, one per verb, from the
    ``constraints`` stream, and assemble the relations."""
    if verbs is None or config.verbs is None:
        return None
    sampler = _ConstraintSampler(config.verbs, config.scalars, features, streams.constraints)
    feature_constraints = [sampler.sample(f"K.{label}") for label in verbs.features.labels]
    own: dict[str, Constraint] = {}
    if config.verbs.own_constraint:
        for verb in verbs.verbs:
            own[verb.label] = sampler.sample(f"K.{verb.label}")
    return Relations(verbs, feature_constraints, own, instances)


class _ConstraintSampler:
    def __init__(
        self,
        verbs: VerbsConfig,
        scalars: ScalarsConfig,
        features: FeatureSet,
        rng: np.random.Generator,
    ) -> None:
        self.verbs = verbs
        self.scalars = scalars
        self.features = features
        self.rng = rng
        self.sampling: RuleSampling = verbs.rules
        self.families = {
            f: w
            for f, w in verbs.constraint_families.items()
            if w > 0 and (f != "comparison" or scalars.count > 0)
        }
        types = self.sampling.input_types
        self.pool: list[Feature | int] = [f for f in features.features if f.type in types]
        if scalars.count and self.sampling.scalar_weight > 0:
            self.pool.extend(range(1, scalars.count + 1))
        self.difference = NormalDist(0.0, math.sqrt(2.0) * scalars.model_std)

    # Entry point -----------------------------------------------------------------------------

    def sample(self, label: str) -> Constraint:
        for _ in range(MAX_TRIES):
            family = _draw(self.rng, self.families)
            builder = {
                "agent": lambda: self._role_rule("a", "agent"),
                "patient": lambda: self._role_rule("p", "patient"),
                "cross": self._cross,
                "key_lock": self._key_lock,
                "comparison": self._comparison,
            }[family]
            constraint = builder()
            if family in ("cross", "key_lock") and not constraint.depends_on_both_roles():
                continue
            return Constraint(
                label,
                constraint.family,
                constraint.literals,
                constraint.table,
                constraint.expression,
                constraint.min_dnf_literals,
            )
        raise GenerationError(
            f"could not sample constraint {label} in {MAX_TRIES} tries: every cross-role candidate "
            f"depended on one argument only; raise the arity or change the operator mix"
        )

    # Literal pools -----------------------------------------------------------------------------

    def _weights(self, entries: Sequence[Feature | int]) -> np.ndarray:
        return np.array(
            [
                self.sampling.scalar_weight
                if isinstance(e, int)
                else self.sampling.input_type_weights[e.type]
                for e in entries
            ],
            dtype=float,
        )

    def _pick(self, entries: Sequence[Any], count: int) -> list[Any]:
        if count == 0:
            return []
        if len(entries) < count:
            raise GenerationError(
                f"a constraint needs {count} literals but only {len(entries)} inputs are eligible; "
                f"lower the arity in verbs.rules or loosen verbs.rules.input_type_weights"
            )
        p = self._weights([e[1] if isinstance(e, tuple) else e for e in entries])
        picks = self.rng.choice(len(entries), size=count, replace=False, p=p / p.sum())
        return [entries[int(i)] for i in picks]

    def _literals(self, role: str, entries: Sequence[Feature | int]) -> list[ConstraintLiteral]:
        """Literals of one role from pool entries: features by position, then thresholds by
        scalar, with the thresholds drawn in that order."""
        binary = sorted((e for e in entries if isinstance(e, Feature)), key=lambda f: f.position)
        scalars = sorted(e for e in entries if isinstance(e, int))
        result: list[ConstraintLiteral] = [RoleFeature(role, f) for f in binary]
        for scalar in scalars:
            result.append(RoleThreshold(role, self._draw_threshold(scalar)))
        return result

    def _draw_threshold(self, scalar: int) -> Threshold:
        low, high = self.scalars.threshold_quantiles
        quantile = float(self.rng.uniform(low, high))
        return Threshold(scalar, model_quantile_threshold(self.scalars, quantile), quantile)

    def _function(self, family: str, literals: Sequence[ConstraintLiteral]) -> Constraint:
        atoms = [item.atom() for item in literals]
        keys = [item.key for item in literals]
        draw = build_function(self.rng, self.sampling, atoms, keys)
        return Constraint(
            "", family, tuple(literals), draw.table, draw.expression, dnf_literal_count(draw.table)
        )

    # Families ------------------------------------------------------------------------------------

    def _role_rule(self, role: str, family: str) -> Constraint:
        arity = _draw(self.rng, self.sampling.arity)
        literals = self._literals(role, self._pick(self.pool, arity))
        return self._function(family, literals)

    def _cross(self) -> Constraint:
        weights = {a: w for a, w in self.sampling.arity.items() if a >= 2 and w > 0}
        if not weights:
            raise GenerationError(
                "cross-role constraints need an arity of at least 2 with positive weight in "
                "verbs.rules.arity"
            )
        arity = _draw(self.rng, weights)
        agent_entry = self._pick(self.pool, 1)[0]
        patient_entry = self._pick(self.pool, 1)[0]
        rest = [("a", e) for e in self.pool if e != agent_entry] + [
            ("p", e) for e in self.pool if e != patient_entry
        ]
        more = self._pick(rest, arity - 2)
        agent_entries = [agent_entry] + [e for role, e in more if role == "a"]
        patient_entries = [patient_entry] + [e for role, e in more if role == "p"]
        literals = self._literals("a", agent_entries) + self._literals("p", patient_entries)
        return self._function("cross", literals)

    def _key_lock(self) -> Constraint:
        pairs = _draw(self.rng, self.verbs.key_lock_pairs)
        agent_entries = self._pick(self.pool, pairs)
        patient_entries = self._pick(self.pool, pairs)
        agent_literals = self._literals("a", agent_entries)
        patient_literals = self._literals("p", patient_entries)
        negated = [bool(v) for v in self.rng.random(2 * pairs) < self.sampling.negation_probability]
        literals: list[ConstraintLiteral] = []
        terms: list[Expr] = []
        for k in range(pairs):
            a, p = agent_literals[k], patient_literals[k]
            literals.extend([a, p])
            terms.append(
                Op(
                    "AND",
                    (literal(a.atom(), negated[2 * k]), literal(p.atom(), negated[2 * k + 1])),
                )
            )
        expression = terms[0] if pairs == 1 else Op("OR", tuple(terms))
        table = expression.truth_table([item.key for item in literals])
        return Constraint(
            "", "key_lock", tuple(literals), table, expression, dnf_literal_count(table)
        )

    def _comparison(self) -> Constraint:
        count = self.scalars.count
        i = int(self.rng.integers(1, count + 1))
        j = i
        if count >= 2 and self.rng.random() < self.verbs.comparison.cross_dimension_probability:
            others = [s for s in range(1, count + 1) if s != i]
            j = others[int(self.rng.integers(len(others)))]
        window = bool(self.rng.random() < self.verbs.comparison.window_probability)
        low_q, high_q = self.verbs.comparison.margin_quantiles
        if window:
            for _ in range(MAX_TRIES):
                q1, q2 = sorted(float(q) for q in self.rng.uniform(low_q, high_q, size=2))
                m1 = round(self.difference.inv_cdf(q1), 4)
                m2 = round(self.difference.inv_cdf(q2), 4)
                if m1 < m2:
                    break
            else:
                raise GenerationError(
                    "could not draw two distinct window margins; widen margin_quantiles"
                )
            comparison = ScalarComparison(i, j, m1, m2)
        else:
            margin = round(self.difference.inv_cdf(float(self.rng.uniform(low_q, high_q))), 4)
            comparison = ScalarComparison(i, j, margin, None)
        table = TruthTable(1, (0, 1))
        return Constraint("", "comparison", (comparison,), table, comparison.atom(), 1)
