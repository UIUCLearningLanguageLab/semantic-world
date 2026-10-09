"""Constraints and relations: the requirements of two-place event types.

A two-place event type is a relation ``R(agent, patient)`` between two entities: a conjunction
of constraints. Each constraint is a rule over the static facts of one or both roles. Its inputs
are binary PROPERTY and PART features (free or determined) and scalar threshold literals of either
role, named with the prefixes ``agent.`` and ``patient.``, and scalar comparisons between the two
roles. No constraint reads an ISA feature, a fluent, or a capacity. An entity is never related to
itself.

Every constraint is stored as its Boolean skeleton (a truth table over its literals) plus the
definitions of its literals, and printed expressions parse back to the same constraint. Every
event-type feature ``EVENTFEAT.<n>`` gets one constraint ``CONSTRAINT.EVENTFEAT.<n>``, and, with
``event_types.binary.own_constraint`` on, every event type gets one of its own,
``CONSTRAINT.<event type>``. An event type's relation is the conjunction of the constraints of its
true event-type features plus its own constraint. Every category of event types has a base
relation: the constraints of the event-type features that are defining with value 1 at the
category. Constraint sampling draws from the ``world:constraints`` stream (Part B of
``docs/specs/TAXONOMY_RELATIONS.md``, with "verb" read as "event type").
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from semantic_world.common.boolean import TruthTable, dnf_literal_count
from semantic_world.taxonomy.config import RuleSampling, ScalarsConfig
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
from semantic_world.taxonomy.tree import Category, Role, Tree
from semantic_world.world.config import BinaryConfig
from semantic_world.world.event_tree import EventTree, generate_event_tree, redraw_event_features

ROLES = ("agent", "patient")
FAMILIES = ("agent", "patient", "cross", "key_lock", "comparison")
CONSTRAINT_PREFIX = "CONSTRAINT"
CHUNK_ROWS = 2048


def constraint_label(owner: str) -> str:
    """The label of the constraint of an event-type feature or an event type's own."""
    return f"{CONSTRAINT_PREFIX}.{owner}"


# ---------------------------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleFeature:
    """A binary feature of the agent (``agent.PROPERTY.7``) or the patient (``patient.PART.4``)."""

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
    """A threshold literal on a scalar of the agent or the patient: ``patient.SCALARDIM.1 >
    0.2031``."""

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
    """An order ``agent.SCALARDIM.i - patient.SCALARDIM.j > low`` (``high`` None) or a window
    ``low < agent.SCALARDIM.i - patient.SCALARDIM.j < high`` between the roles' scalars."""

    agent_scalar: int
    patient_scalar: int
    low: float
    high: float | None

    def atom(self) -> Cmp:
        return Cmp(
            f"agent.SCALARDIM.{self.agent_scalar}",
            f"patient.SCALARDIM.{self.patient_scalar}",
            self.low,
            self.high,
        )

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
    """``agent``, ``patient``, or ``both`` for a comparison."""
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
        return tuple(item for item in self.literals if literal_role(item) == "agent")

    @property
    def patient_literals(self) -> tuple[ConstraintLiteral, ...]:
        return tuple(item for item in self.literals if literal_role(item) == "patient")

    @property
    def comparisons(self) -> tuple[ScalarComparison, ...]:
        return tuple(item for item in self.literals if isinstance(item, ScalarComparison))

    def depends_on_both_roles(self) -> bool:
        """Whether the skeleton depends on at least one agent input and one patient input (a
        comparison counts for both)."""
        roles = {literal_role(self.literals[i]) for i in self.table.relevant_inputs()}
        return ("agent" in roles or "both" in roles) and ("patient" in roles or "both" in roles)

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
            elif item.role == "agent":
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
                if item.role == "agent":
                    index += bit * column[agents][:, None]
                else:
                    index += bit * column[None, :]
        return np.asarray(self.table.bits, dtype=np.uint8)[index].astype(bool)

    def record(self) -> dict[str, Any]:
        """The constraint as a record (the form the old ``constraints.yaml`` held)."""
        return {
            "label": self.label,
            "family": self.family,
            "agent_literals": [_literal_record(item) for item in self.agent_literals],
            "patient_literals": [_literal_record(item) for item in self.patient_literals],
            "comparisons": [
                {
                    "agent": f"SCALARDIM.{c.agent_scalar}",
                    "patient": f"SCALARDIM.{c.patient_scalar}",
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
    """The relation of an event type (``base`` False) or the base relation of a category of
    event types."""

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
    """Every constraint and every relation of a world, with evaluation over the world's
    entities."""

    def __init__(
        self,
        event_tree: EventTree,
        feature_constraints: Sequence[Constraint],
        own_constraints: dict[str, Constraint],
        instances: Instances,
        constraint_density: dict[str, float] | None = None,
        event_type_density: dict[str, float] | None = None,
        event_type_tries: dict[str, int] | None = None,
        outside_range: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
        explicit: frozenset[str] = frozenset(),
    ) -> None:
        self.event_tree = event_tree
        self.feature_constraints = tuple(feature_constraints)
        self.own_constraints = dict(own_constraints)
        self.instances = instances
        self.explicit = explicit
        """The event types whose own constraint is an explicit requirement from an event file.
        Such an event type keeps the constraints of the features that are defining at its
        ancestors (their base relations, REL.10) and loses those of its other true features."""
        self.constraint_density = constraint_density
        """Leaf-pair density of every constraint alone, when a density check is on."""
        self.event_type_density = event_type_density
        """Leaf-pair density of every event type's relation, when the density check is on."""
        self.event_type_tries = event_type_tries
        self.outside_range = outside_range
        """The event types the generator could not bring inside the density range."""
        self.warnings = warnings
        self.constraints: tuple[Constraint, ...] = self.feature_constraints + tuple(
            self.own_constraints[v.label]
            for v in event_tree.event_types
            if v.label in self.own_constraints
        )
        self._by_label = {c.label: c for c in self.constraints}

    def constraint(self, label: str) -> Constraint:
        try:
            return self._by_label[label]
        except KeyError:
            raise KeyError(f"unknown constraint {label!r}") from None

    def category(self, event_type: Category | str) -> Category:
        return event_type if isinstance(event_type, Category) else self.event_tree.tree[event_type]

    def relation(self, event_type: Category | str) -> Relation:
        """An event type's relation, or the base relation of a category of event types."""
        category = self.category(event_type)
        if category.is_leaf:
            mask = category.values == 1
            if category.label in self.explicit:
                mask &= category.roles == Role.DEFINING_INHERITED
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
        return [self.relation(c) for c in self.event_tree.categories]

    def holds(
        self, event_type: Category | str, agents: np.ndarray, patients: np.ndarray
    ) -> np.ndarray:
        """Whether the relation holds for the aligned pairs ``(agents[k], patients[k])`` of
        instance indices. An instance is never related to itself."""
        agents = np.asarray(agents, dtype=np.intp)
        patients = np.asarray(patients, dtype=np.intp)
        result = agents != patients
        values, scalars = self.instances.values, self.instances.scalars
        for constraint in self.relation(event_type).constraints:
            result &= constraint.pair_values(values, scalars, agents, patients)
        return result

    def matrix(self, event_type: Category | str, chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
        """The relation over every ordered pair of instances, as an ``(n, n)`` bool array with a
        False diagonal, computed in chunks of agents."""
        n = len(self.instances)
        values, scalars = self.instances.values, self.instances.scalars
        constraints = self.relation(event_type).constraints
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
        """Every constraint's record, with its leaf-pair density when a density check is on."""
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
    binary: BinaryConfig | None,
    scalars: ScalarsConfig,
    features: FeatureSet,
    event_tree: EventTree | None,
    instances: Instances,
    rng: np.random.Generator,
) -> Relations | None:
    """Sample one constraint per event-type feature and, when on, one per event type, from the
    ``world:constraints`` stream, and assemble the relations. This is the build without the
    density checks; :func:`generate_relations` is the full build."""
    if event_tree is None or binary is None:
        return None
    sampler = _ConstraintSampler(binary, scalars, features, rng)
    feature_constraints = [
        sampler.sample(constraint_label(label)) for label in event_tree.features.labels
    ]
    own: dict[str, Constraint] = {}
    if binary.own_constraint:
        for event_type in event_tree.event_types:
            own[event_type.label] = sampler.sample(constraint_label(event_type.label))
    return Relations(event_tree, feature_constraints, own, instances)


def generate_relations(
    binary: BinaryConfig | None,
    scalars: ScalarsConfig,
    rules: RuleSet,
    tree: Tree,
    instances: Instances,
    rng_constraints: np.random.Generator,
    rng_event_tree: np.random.Generator,
) -> tuple[EventTree | None, Relations | None]:
    """The full build of the two-place event types: the constraints of the event-type features
    (with the constraint density floor), the event-type tree, and each event type's own
    constraint with the density range (``docs/proposals/2026-09-29-taxonomy-verb-density.md``).

    Densities are measured over the leaves of the category tree, never the entities. Constraints
    draw from ``world:constraints`` and the event tree from ``world:event_tree``, so with both
    checks off the build equals the one without them.
    """
    if binary is None:
        return None, None
    settings = binary
    leaf_values = np.stack([leaf.values for leaf in tree.leaves])
    leaf_scalars = np.stack([leaf.scalars for leaf in tree.leaves])
    checks_on = settings.density is not None or settings.constraint_min_density is not None
    sampler = _ConstraintSampler(settings, scalars, rules.features, rng_constraints)
    warnings: list[str] = []
    constraint_density: dict[str, float] = {}

    def density(constraints: Sequence[Constraint]) -> float:
        return leaf_pair_density(constraints, leaf_values, leaf_scalars)

    feature_constraints: list[Constraint] = []
    for label in settings.feature_labels:
        name = constraint_label(label)
        if settings.density is None and settings.constraint_min_density is None:
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

    event_tree = generate_event_tree(binary, rng_event_tree)
    assert event_tree is not None
    own: dict[str, Constraint] = {}
    event_type_density: dict[str, float] = {}
    event_type_tries: dict[str, int] = {}
    outside: list[str] = []
    if settings.density is None:
        for event_type in event_tree.event_types:
            if settings.own_constraint:
                own[event_type.label] = sampler.sample(constraint_label(event_type.label))
    else:
        tuner = _DensityTuner(
            settings, event_tree, feature_constraints, sampler, density, rng_event_tree
        )
        for event_type in event_tree.event_types:
            result = tuner.tune(event_type)
            if result.own is not None:
                own[event_type.label] = result.own
            event_type_density[event_type.label] = result.density
            event_type_tries[event_type.label] = result.tries
            if not result.in_range:
                outside.append(event_type.label)
                warnings.append(
                    f"event type {event_type.label} has leaf-pair density {result.density:.4f}, "
                    f"outside ({settings.density.min}, {settings.density.max}) after "
                    f"{result.tries} tries"
                )
    if checks_on:
        for constraint in own.values():
            constraint_density[constraint.label] = density([constraint])
    relations = Relations(
        event_tree,
        feature_constraints,
        own,
        instances,
        constraint_density=constraint_density if checks_on else None,
        event_type_density=event_type_density if settings.density is not None else None,
        event_type_tries=event_type_tries if settings.density is not None else None,
        outside_range=tuple(outside),
        warnings=tuple(warnings),
    )
    return event_tree, relations


def _sample_dense_constraint(
    sampler: _ConstraintSampler, label: str, density, minimum: float | None, max_tries: int
) -> tuple[Constraint, str | None]:
    """Resample a constraint until its leaf-pair density is not degenerate (neither 0 nor 1: a
    constraint that no leaf pair satisfies makes every event type that carries its feature
    impossible, and one that every pair satisfies constrains nothing) and, with ``minimum``
    given, reaches it; otherwise keep the best candidate and return a warning."""
    floor = minimum or 0.0
    best: tuple[float, Constraint] | None = None
    for _ in range(max_tries):
        constraint = sampler.sample(label)
        value = density([constraint])
        if value >= floor and 0.0 < value < 1.0:
            return constraint, None
        score = value if value < 1.0 else -1.0
        if best is None or score > best[0]:
            best = (score, constraint)
    assert best is not None
    value = density([best[1]])
    if minimum is None:
        return best[1], (
            f"constraint {label} has leaf-pair density {value:.4f} after {max_tries} tries: "
            f"it holds for no leaf pair or for every leaf pair"
        )
    return best[1], (
        f"constraint {label} has leaf-pair density {value:.4f}, below "
        f"event_types.binary.constraint_min_density {minimum} after {max_tries} tries"
    )


@dataclass(frozen=True)
class _TuneResult:
    own: Constraint | None
    density: float
    tries: int
    in_range: bool


class _DensityTuner:
    """Brings each event type's leaf-pair density into the configured range, in category
    order. A density of 0 or 1 is always outside the range."""

    def __init__(
        self,
        settings: BinaryConfig,
        event_tree: EventTree,
        feature_constraints: Sequence[Constraint],
        sampler: _ConstraintSampler,
        density,
        rng: np.random.Generator,
    ) -> None:
        assert settings.density is not None
        self.settings = settings
        self.range = settings.density
        self.event_tree = event_tree
        self.feature_constraints = tuple(feature_constraints)
        self.sampler = sampler
        self.density = density
        self.rng = rng
        self.base_rates = event_tree.features.base_rates
        self.vectors: dict[str, bytes] = {
            v.label: v.values.tobytes() for v in event_tree.event_types
        }

    def distance(self, value: float) -> float:
        return self.range.distance(value)

    def base_constraints(self, event_type: Category) -> list[Constraint]:
        return [self.feature_constraints[i] for i in np.flatnonzero(event_type.values == 1)]

    def redraw(self, event_type: Category) -> bool | None:
        """Redraw the event type's non-defining features. None when nothing can change (every
        feature is defining at the parent); False when the draw duplicates another event type
        and distinct leaves are required, which counts as a failed try; True otherwise."""
        if event_type.parent is not None and event_type.parent.defining_mask().all():
            return None
        values = redraw_event_features(event_type, self.settings, self.base_rates, self.rng)
        key = values.tobytes()
        if self.settings.inheritance.require_distinct_leaves and any(
            other == key for label, other in self.vectors.items() if label != event_type.label
        ):
            return False
        event_type.free_values = values
        event_type.values = values.copy()
        self.vectors[event_type.label] = key
        return True

    def tune(self, event_type: Category) -> _TuneResult:
        label = constraint_label(event_type.label)
        best: tuple[float, np.ndarray, Constraint | None, float] | None = None
        tries = 0
        stuck = False
        while tries < self.range.max_tries and not stuck:
            tries += 1
            base_density = self.density(self.base_constraints(event_type))
            too_sparse = base_density <= max(self.range.min, 0.0)
            too_dense_without_own = not self.settings.own_constraint and (
                base_density >= min(self.range.max, 1.0)
            )
            if too_sparse or too_dense_without_own:
                candidate = (
                    self.distance(base_density),
                    event_type.values.copy(),
                    None,
                    base_density,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
                if self.redraw(event_type) is None:
                    stuck = True
                continue
            own = self.sampler.sample(label) if self.settings.own_constraint else None
            value = self.density(self.base_constraints(event_type) + ([own] if own else []))
            candidate = (self.distance(value), event_type.values.copy(), own, value)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if candidate[0] == 0.0:
                break
        assert best is not None
        _, values, own, value = best
        event_type.free_values = values.copy()
        event_type.values = values.copy()
        self.vectors[event_type.label] = values.tobytes()
        if own is None and self.settings.own_constraint:
            # Every candidate was too sparse before its own constraint; it still gets one.
            own = self.sampler.sample(label)
            value = self.density(self.base_constraints(event_type) + [own])
        return _TuneResult(own, value, tries, self.distance(value) == 0.0)


class _ConstraintSampler:
    def __init__(
        self,
        binary: BinaryConfig,
        scalars: ScalarsConfig,
        features: FeatureSet,
        rng: np.random.Generator,
    ) -> None:
        self.binary = binary
        self.scalars = scalars
        self.features = features
        self.rng = rng
        self.sampling: RuleSampling = binary.rules
        self.families = {
            f: w
            for f, w in binary.constraint_families.items()
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
                "agent": lambda: self._role_rule("agent", "agent"),
                "patient": lambda: self._role_rule("patient", "patient"),
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
                f"lower the arity in event_types.binary.rules or loosen its input_type_weights"
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
                "event_types.binary.rules.arity"
            )
        arity = _draw(self.rng, weights)
        agent_entry = self._pick(self.pool, 1)[0]
        patient_entry = self._pick(self.pool, 1)[0]
        rest = [("agent", e) for e in self.pool if e != agent_entry] + [
            ("patient", e) for e in self.pool if e != patient_entry
        ]
        more = self._pick(rest, arity - 2)
        agent_entries = [agent_entry] + [e for role, e in more if role == "agent"]
        patient_entries = [patient_entry] + [e for role, e in more if role == "patient"]
        literals = self._literals("agent", agent_entries) + self._literals(
            "patient", patient_entries
        )
        return self._function("cross", literals)

    def _key_lock(self) -> Constraint:
        pairs = _draw(self.rng, self.binary.key_lock_pairs)
        agent_entries = self._pick(self.pool, pairs)
        patient_entries = self._pick(self.pool, pairs)
        agent_literals = self._literals("agent", agent_entries)
        patient_literals = self._literals("patient", patient_entries)
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
        if count >= 2 and self.rng.random() < self.binary.comparison.cross_dimension_probability:
            others = [s for s in range(1, count + 1) if s != i]
            j = others[int(self.rng.integers(len(others)))]
        window = bool(self.rng.random() < self.binary.comparison.window_probability)
        low_q, high_q = self.binary.comparison.margin_quantiles
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
