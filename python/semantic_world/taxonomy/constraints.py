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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from semantic_world.common.boolean import TruthTable, dnf_literal_count
from semantic_world.taxonomy.config import Config, RuleSampling, ScalarsConfig, VerbsConfig
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.expressions import Cmp, Expr, Gt, Op, Var, literal
from semantic_world.taxonomy.features import Feature, FeatureSet
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import (
    MAX_TRIES,
    RuleSet,
    Threshold,
    _draw,
    build_function,
    model_quantile_threshold,
)
from semantic_world.taxonomy.streams import Streams
from semantic_world.taxonomy.tree import Category, Tree
from semantic_world.taxonomy.verbs import VerbTaxonomy, generate_verb_tree, redraw_verb_features

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
        return self.evaluate(values[agents], scalars[agents], values[patients], scalars[patients])

    def evaluate(
        self,
        agent_values: np.ndarray,
        agent_scalars: np.ndarray,
        patient_values: np.ndarray,
        patient_scalars: np.ndarray,
    ) -> np.ndarray:
        """The constraint for aligned rows of agent and patient feature and scalar matrices."""
        columns = {}
        for item in self.literals:
            if isinstance(item, ScalarComparison):
                columns[item.key] = item.pair_values(agent_scalars, patient_scalars)
            elif item.role == "a":
                columns[item.key] = item.instance_values(agent_values, agent_scalars)
            else:
                columns[item.key] = item.instance_values(patient_values, patient_scalars)
        return self.evaluate_columns(columns, agent_values.shape[0])

    def evaluate_columns(self, columns: Mapping[str, np.ndarray], rows: int) -> np.ndarray:
        """The skeleton over given literal columns, keyed by literal key."""
        if not self.literals:
            return np.full(rows, bool(self.table.bits[0]))
        stacked = np.stack(
            [
                np.broadcast_to(np.asarray(columns[key], dtype=np.uint8), (rows,))
                for key in self.keys
            ],
            axis=1,
        )
        return self.table.evaluate(stacked).astype(bool)

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


def relation_pair_matrix(
    constraints: Sequence[Constraint], values: np.ndarray, scalars: np.ndarray
) -> np.ndarray:
    """A conjunction of constraints over every ordered pair of rows, including a row paired with
    itself: shape ``(rows, rows)``."""
    n = values.shape[0]
    out = np.ones((n, n), dtype=bool)
    agents = np.arange(n)
    for constraint in constraints:
        out &= constraint.matrix(values, scalars, agents)
    return out


def leaf_pair_density(
    constraints: Sequence[Constraint], leaf_values: np.ndarray, leaf_scalars: np.ndarray
) -> float:
    """The proportion of ordered leaf pairs (a leaf paired with itself included) for which the
    conjunction holds between the leaves' generative vectors, with their own scalars."""
    if leaf_values.shape[0] == 0:
        return math.nan
    return float(relation_pair_matrix(constraints, leaf_values, leaf_scalars).mean())


class Relations:
    """Every constraint and every relation of a run, with evaluation over the run's instances."""

    def __init__(
        self,
        verbs: VerbTaxonomy,
        feature_constraints: Sequence[Constraint],
        own_constraints: dict[str, Constraint],
        instances: Instances,
        constraint_density: dict[str, float] | None = None,
        verb_density: dict[str, float] | None = None,
        verb_tries: dict[str, int] | None = None,
        outside_range: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> None:
        self.verbs = verbs
        self.feature_constraints = tuple(feature_constraints)
        self.own_constraints = dict(own_constraints)
        self.instances = instances
        self.constraint_density = constraint_density
        """Leaf-pair density of every constraint alone, when a density check is on."""
        self.verb_density = verb_density
        """Leaf-pair density of every verb's relation, when the verb density check is on."""
        self.verb_tries = verb_tries
        self.outside_range = outside_range
        """The verbs the generator could not bring inside the density range."""
        self.warnings = warnings
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
        """The ``constraints.yaml`` entries, with each constraint's leaf-pair density when a
        density check is on."""
        entries = []
        for c in self.constraints:
            entry = c.record()
            if self.constraint_density is not None:
                entry["leaf_pair_density"] = self.constraint_density[c.label]
            entries.append(entry)
        return entries

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
    ``constraints`` stream, and assemble the relations. This is the build without the density
    checks; :func:`generate_relations` is the full build."""
    if verbs is None or config.verbs is None:
        return None
    sampler = _ConstraintSampler(config.verbs, config.scalars, features, streams.constraints)
    feature_constraints = [sampler.sample(f"K.{label}") for label in verbs.features.labels]
    own: dict[str, Constraint] = {}
    if config.verbs.own_constraint:
        for verb in verbs.verbs:
            own[verb.label] = sampler.sample(f"K.{verb.label}")
    return Relations(verbs, feature_constraints, own, instances)


def generate_relations(
    config: Config, rules: RuleSet, tree: Tree, instances: Instances, streams: Streams
) -> tuple[VerbTaxonomy | None, Relations | None]:
    """The full verb build: verb-feature constraints (with the constraint density floor), the
    verb tree, and each verb's own constraint with the verb density range
    (``docs/proposals/2026-09-29-taxonomy-verb-density.md``).

    Densities are measured over the leaves of the noun tree, never the instances. Constraints
    draw from the ``constraints`` stream and verb features from the ``verb_tree`` stream, so with
    both checks off the build equals the one without them.
    """
    if config.verbs is None:
        return None, None
    settings = config.verbs
    leaf_values = np.stack([leaf.values for leaf in tree.leaves])
    leaf_scalars = np.stack([leaf.scalars for leaf in tree.leaves])
    checks_on = settings.density is not None or settings.constraint_min_density is not None
    sampler = _ConstraintSampler(settings, config.scalars, rules.features, streams.constraints)
    warnings: list[str] = []
    constraint_density: dict[str, float] = {}

    def density(constraints: Sequence[Constraint]) -> float:
        return leaf_pair_density(constraints, leaf_values, leaf_scalars)

    feature_constraints: list[Constraint] = []
    for label in settings.feature_labels:
        name = f"K.{label}"
        if settings.constraint_min_density is None:
            constraint = sampler.sample(name)
        else:
            constraint, warning = _sample_dense_constraint(
                sampler, name, density, settings.constraint_min_density, settings.density_tries
            )
            if warning:
                warnings.append(warning)
        feature_constraints.append(constraint)
        if checks_on:
            constraint_density[name] = density([constraint])

    verbs = generate_verb_tree(config, streams)
    assert verbs is not None
    own: dict[str, Constraint] = {}
    verb_density: dict[str, float] = {}
    verb_tries: dict[str, int] = {}
    outside: list[str] = []
    if settings.density is None:
        for verb in verbs.verbs:
            if settings.own_constraint:
                own[verb.label] = sampler.sample(f"K.{verb.label}")
    else:
        tuner = _DensityTuner(settings, verbs, feature_constraints, sampler, density, streams)
        for verb in verbs.verbs:
            result = tuner.tune(verb)
            if result.own is not None:
                own[verb.label] = result.own
            verb_density[verb.label] = result.density
            verb_tries[verb.label] = result.tries
            if not result.in_range:
                outside.append(verb.label)
                warnings.append(
                    f"verb {verb.label} has leaf-pair density {result.density:.4f}, outside "
                    f"[{settings.density.min}, {settings.density.max}] after {result.tries} tries"
                )
    if checks_on:
        for constraint in own.values():
            constraint_density[constraint.label] = density([constraint])
    relations = Relations(
        verbs,
        feature_constraints,
        own,
        instances,
        constraint_density=constraint_density if checks_on else None,
        verb_density=verb_density if settings.density is not None else None,
        verb_tries=verb_tries if settings.density is not None else None,
        outside_range=tuple(outside),
        warnings=tuple(warnings),
    )
    return verbs, relations


def _sample_dense_constraint(
    sampler: _ConstraintSampler, label: str, density, minimum: float, max_tries: int
) -> tuple[Constraint, str | None]:
    """Resample a constraint until its leaf-pair density reaches ``minimum``; otherwise keep the
    densest candidate and return a warning."""
    best: tuple[float, Constraint] | None = None
    for _ in range(max_tries):
        constraint = sampler.sample(label)
        value = density([constraint])
        if value >= minimum:
            return constraint, None
        if best is None or value > best[0]:
            best = (value, constraint)
    assert best is not None
    return best[1], (
        f"constraint {label} has leaf-pair density {best[0]:.4f}, below "
        f"verbs.constraint_min_density {minimum} after {max_tries} tries"
    )


@dataclass(frozen=True)
class _TuneResult:
    own: Constraint | None
    density: float
    tries: int
    in_range: bool


class _DensityTuner:
    """Brings each verb's leaf-pair density into the configured range, in category order."""

    def __init__(
        self,
        settings: VerbsConfig,
        verbs: VerbTaxonomy,
        feature_constraints: Sequence[Constraint],
        sampler: _ConstraintSampler,
        density,
        streams: Streams,
    ) -> None:
        assert settings.density is not None
        self.settings = settings
        self.range = settings.density
        self.verbs = verbs
        self.feature_constraints = tuple(feature_constraints)
        self.sampler = sampler
        self.density = density
        self.rng = streams.verb_tree
        self.base_rates = verbs.features.base_rates
        self.vectors: dict[str, bytes] = {v.label: v.values.tobytes() for v in verbs.verbs}

    def distance(self, value: float) -> float:
        if self.range.min <= value <= self.range.max:
            return 0.0
        return self.range.min - value if value < self.range.min else value - self.range.max

    def base_constraints(self, verb: Category) -> list[Constraint]:
        return [self.feature_constraints[i] for i in np.flatnonzero(verb.values == 1)]

    def redraw(self, verb: Category) -> bool:
        """Redraw the verb's non-defining features; False when the draw duplicates another verb
        and distinct leaves are required, or when nothing can change."""
        if verb.parent is not None and verb.parent.defining_mask().all():
            return False
        values = redraw_verb_features(verb, self.settings, self.base_rates, self.rng)
        key = values.tobytes()
        if self.settings.inheritance.require_distinct_leaves and any(
            other == key for label, other in self.vectors.items() if label != verb.label
        ):
            return False
        verb.free_values = values
        verb.values = values.copy()
        self.vectors[verb.label] = key
        return True

    def tune(self, verb: Category) -> _TuneResult:
        label = f"K.{verb.label}"
        best: tuple[float, np.ndarray, Constraint | None, float] | None = None
        tries = 0
        stuck = False
        while tries < self.range.max_tries and not stuck:
            tries += 1
            base_density = self.density(self.base_constraints(verb))
            too_sparse = base_density < self.range.min
            too_dense_without_own = (
                not self.settings.own_constraint and base_density > self.range.max
            )
            if too_sparse or too_dense_without_own:
                candidate = (self.distance(base_density), verb.values.copy(), None, base_density)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                if not self.redraw(verb):
                    stuck = True
                continue
            own = self.sampler.sample(label) if self.settings.own_constraint else None
            value = self.density(self.base_constraints(verb) + ([own] if own else []))
            candidate = (self.distance(value), verb.values.copy(), own, value)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if candidate[0] == 0.0:
                break
        assert best is not None
        _, values, own, value = best
        verb.free_values = values.copy()
        verb.values = values.copy()
        self.vectors[verb.label] = values.tobytes()
        if own is None and self.settings.own_constraint:
            # Every candidate was too sparse before its own constraint; the verb still gets one.
            own = self.sampler.sample(label)
            value = self.density(self.base_constraints(verb) + [own])
        return _TuneResult(own, value, tries, self.distance(value) == 0.0)


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
