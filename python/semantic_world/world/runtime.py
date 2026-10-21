"""The runtime: ``derive``, ``able``, ``legal``, ``legal_bindings``, and ``apply``.

The runtime is a small interpreter of a :class:`RuntimeDefinition` (``docs/specs/
WORLD_AND_LANGUAGE.md``, "The runtime's operations"). Every operation evaluates the matrix layers
of the definition (REL.18), vectorized over all the given entities or bindings at once. Nothing
here reads a truth table. The runtime draws no random number and chooses no event: the same
definition, state, and events always give the same result.

Entities are referred to by index into the definition's entities table, and bindings are tuples
of entity indices in role order (``(agent,)`` or ``(agent, patient)``). A :class:`State` holds
every entity's base fluents at one time point. ``apply`` takes the state at ``TIME.k`` and the
events of step k to the state at ``TIME.k+1``: every event must be legal, the events must not
interfere, every effect whose condition holds in the state at ``TIME.k`` is applied (an effect
without a condition always is), unwritten base fluents keep their values (inertia), and derived
fluents are never carried over, only recomputed. Non-interference counts every effect as a write
of its fluent, whether or not its condition holds, and counts what a condition reads beside what
a precondition reads (stage b1).

Since stage b2 a state also holds every entity's numeric fluents (float64), a precondition or a
condition may read a numeric threshold, a derived fluent may read one through its rule, and an
effect may add a constant to a numeric fluent or subtract one. Within a step each numeric value
becomes ``clamp(clamp(v + e) + d)``: the one effect that wrote it (non-interference allows at
most one), then the drift, each clamped to the range, in float64. ``apply_step`` also returns
the values after the effects and before the drift, which histories record. The comparisons of
"Phase (b): the world" (``greater``, ``above``, ``below``, ``increase``, ``decrease``) are judged
on demand by direct arithmetic on the values, for entities or category means.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from semantic_world.world.definition import EventTypeRecord, RuntimeDefinition
from semantic_world.world.dynamics import Effect
from semantic_world.world.errors import IllegalEventError, InterferenceError, WorldError
from semantic_world.world.matrices import AgreementReport, check_agreement
from semantic_world.world.numeric import clamp
from semantic_world.world.places import EVERYPLACE

Binding = tuple[int, ...]
ABLE_CHUNK_ROWS = 32768
"""Ordered pairs are evaluated in chunks of this many rows when the requirement table is built."""
ENTITY = "ENTITY"
"""The world symbol for all entities: the comparison class of a top-level category."""
RANGE = "RANGE"
"""The standard of an absolute comparison of a numeric fluent: the ends of its range."""
PLACE_CATEGORY_PREFIX = "PLACECATEGORY."


# ---------------------------------------------------------------------------------------------
# States, events, and derived facts
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """The base fluents of every entity at one time point: ``values``, shape ``(entities, base
    fluents)`` with 0 and 1, and ``numeric``, shape ``(entities, numeric fluents)`` in float64
    (stage b2; a world without numeric fluents has an empty second axis). The arrays are
    read-only; ``apply`` returns a new state."""

    values: np.ndarray
    numeric: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.uint8, copy=True)
        if values.ndim != 2:
            raise WorldError(f"a state is a 2-D array (entities, base fluents), got {values.shape}")
        if values.size and values.max() > 1:
            raise WorldError("a state holds 0 and 1 only")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        numeric = (
            np.zeros((values.shape[0], 0), dtype=np.float64)
            if self.numeric is None
            else np.array(self.numeric, dtype=np.float64, copy=True)
        )
        if numeric.ndim != 2 or numeric.shape[0] != values.shape[0]:
            raise WorldError(
                f"the numeric part of a state is a 2-D array (entities, numeric fluents) with "
                f"the entities of the Boolean part, got {numeric.shape} beside {values.shape}"
            )
        if numeric.size and not np.isfinite(numeric).all():
            raise WorldError("a state holds finite numeric values only")
        numeric.setflags(write=False)
        object.__setattr__(self, "numeric", numeric)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.values.shape == other.values.shape
            and bool(np.array_equal(self.values, other.values))
            and self.numeric.shape == other.numeric.shape
            and bool(np.array_equal(self.numeric, other.numeric))
        )

    def __hash__(self) -> int:
        return hash(
            (self.values.shape, self.values.tobytes(), self.numeric.shape, self.numeric.tobytes())
        )

    @classmethod
    def from_true(
        cls,
        definition: RuntimeDefinition,
        true: Mapping[str, Iterable[str]],
        numeric: Mapping[str, Mapping[str, float]] | None = None,
    ) -> State:
        """A state from the base fluents that are true, as ``{entity label: [fluent labels]}``,
        and the numeric values as ``{entity label: {fluent label: value}}`` (the entities
        table's initial values when None, and every entity and fluent must be given otherwise).
        Every entity of the definition must be listed in ``true``; a fluent not listed is
        false."""
        values = np.zeros((definition.entity_count, len(definition.base_fluents)), dtype=np.uint8)
        missing = [label for label in definition.entity_labels if label not in true]
        if missing:
            raise WorldError(f"the state lists no fluents for {missing[0]}")
        for label, fluents in true.items():
            row = definition.entity_index(label)
            for fluent in fluents:
                if not definition.is_base_fluent(fluent):
                    raise WorldError(
                        f"{fluent} is not a base fluent; a state holds base fluents only"
                    )
                values[row, definition.base_fluent_index(fluent)] = 1
        if numeric is None:
            return cls(values, _initial_numeric(definition))
        matrix = np.zeros((definition.entity_count, len(definition.numeric_fluents)))
        if not numeric:
            return cls(values, matrix)  # every numeric fluent at 0, for a caller that fills them
        for i, label in enumerate(definition.entity_labels):
            entry = numeric.get(label)
            if entry is None:
                raise WorldError(f"the state lists no numeric values for {label}")
            for fluent in definition.numeric_fluents:
                if fluent not in entry:
                    raise WorldError(f"the state lists no value of {fluent} for {label}")
                matrix[i, definition.numeric_index(fluent)] = float(entry[fluent])
            for fluent in entry:
                if not definition.is_numeric_fluent(fluent):
                    raise WorldError(f"{fluent} is not a numeric fluent of the definition")
        return cls(values, matrix)

    def true_fluents(self, definition: RuntimeDefinition) -> dict[str, list[str]]:
        """The base fluents that are true, as ``{entity label: [fluent labels]}``, every entity
        listed, fluents in definition order."""
        _check_state(definition, self)
        return {
            label: [f for j, f in enumerate(definition.base_fluents) if self.values[i, j]]
            for i, label in enumerate(definition.entity_labels)
        }

    def numeric_values(self, definition: RuntimeDefinition) -> dict[str, dict[str, float]]:
        """Every entity's numeric fluents, as ``{entity label: {fluent label: value}}``."""
        _check_state(definition, self)
        return {
            label: {f: float(self.numeric[i, j]) for j, f in enumerate(definition.numeric_fluents)}
            for i, label in enumerate(definition.entity_labels)
        }


def _initial_numeric(definition: RuntimeDefinition) -> np.ndarray:
    if definition.initial_numeric is None:
        raise WorldError("the definition's entities table carries no initial numeric values")
    return definition.initial_numeric


def initial_state(definition: RuntimeDefinition) -> State:
    """The state at ``TIME.1`` of every episode: the initial base fluents and numeric values of
    the entities table."""
    if definition.initial_values is None:
        raise WorldError("the definition's entities table carries no initial fluents")
    return State(definition.initial_values, _initial_numeric(definition))


@dataclass(frozen=True)
class Event:
    """One occurrence of an event type with a binding of entity indices in role order."""

    event_type: str
    binding: Binding

    @classmethod
    def from_labels(
        cls, definition: RuntimeDefinition, event_type: str, binding: Mapping[str, str]
    ) -> Event:
        roles = definition.event_type(event_type).roles
        if set(binding) != set(roles):
            raise WorldError(
                f"{event_type} has the roles {', '.join(roles)}, but the binding names "
                f"{', '.join(binding) or 'none'}"
            )
        return cls(event_type, tuple(definition.entity_index(binding[role]) for role in roles))

    def binding_labels(self, definition: RuntimeDefinition) -> dict[str, str]:
        roles = definition.event_type(self.event_type).roles
        return {
            role: definition.entity_labels[i] for role, i in zip(roles, self.binding, strict=True)
        }

    def describe(self, definition: RuntimeDefinition) -> str:
        """``EVENTTYPE2.1.2(agent=INSTANCE.1.1.1, patient=INSTANCE.1.2.1)``."""

        def name(i: int) -> str:
            return definition.entity_labels[i] if 0 <= i < definition.entity_count else str(i)

        try:
            roles = definition.event_type(self.event_type).roles
        except WorldError:
            roles = ()
        if len(roles) == len(self.binding):
            inside = ", ".join(f"{r}={name(i)}" for r, i in zip(roles, self.binding, strict=True))
        else:
            inside = ", ".join(name(i) for i in self.binding)
        return f"{self.event_type}({inside})"


@dataclass(frozen=True)
class DerivedFacts:
    """The derived facts of some entities: the derived static features (the same in every
    state) and the derived fluents in one state. Rows follow ``entities``; columns follow the
    label tuples."""

    entities: tuple[int, ...]
    static_labels: tuple[str, ...]
    static: np.ndarray
    fluent_labels: tuple[str, ...]
    fluent: np.ndarray

    def true_fluents(self, definition: RuntimeDefinition) -> dict[str, list[str]]:
        """The derived fluents that are true, as ``{entity label: [fluent labels]}``."""
        return {
            definition.entity_labels[e]: [
                f for j, f in enumerate(self.fluent_labels) if self.fluent[k, j]
            ]
            for k, e in enumerate(self.entities)
        }


# ---------------------------------------------------------------------------------------------
# Static facts (cached once per definition)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticFacts:
    """Every static fact of every entity, computed once from the free features, the scalars, and
    the entity layers: the entity literal matrix with its static columns filled (fluent columns
    0), and each static feature's column, free and derived."""

    literals: np.ndarray
    """Shape ``(entities, literals)``, uint8."""
    values: dict[str, np.ndarray]
    """Each static feature's column over every entity, free and derived."""
    derived: np.ndarray
    """Shape ``(entities, derived static features)``, in ``definition.derived_features`` order."""


def static_facts(definition: RuntimeDefinition) -> StaticFacts:
    """The static facts, computed on the first call and cached on the definition."""
    cached = definition.cache.get("static")
    if cached is not None:
        return cached
    n = definition.entity_count
    free_index = {label: j for j, label in enumerate(definition.free_features)}
    scalar_index = {label: j for j, label in enumerate(definition.scalars)}
    literals = definition.entity_matrices.literal_matrix(n)
    for i, spec in enumerate(definition.literals):
        reads = spec.reads
        if reads["role"] is not None:
            continue
        if reads["kind"] == "feature" and reads["feature"] in free_index:
            literals[:, i] = definition.free_values[:, free_index[reads["feature"]]]
        elif reads["kind"] == "threshold" and reads["scalar"] in scalar_index:
            column = definition.scalar_values[:, scalar_index[reads["scalar"]]]
            literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    outputs = definition.entity_matrices.evaluate(literals)
    values = {label: definition.free_values[:, j] for label, j in free_index.items()}
    for label in definition.derived_features:
        values[label] = outputs[label]
        column = _literal_column(definition, label)
        if column is not None:
            literals[:, column] = outputs[label]
    derived = (
        np.stack([values[label] for label in definition.derived_features], axis=1)
        if definition.derived_features
        else np.zeros((n, 0), dtype=np.uint8)
    )
    facts = StaticFacts(literals, values, derived)
    definition.cache["static"] = facts
    return facts


def _literal_column(definition: RuntimeDefinition, key: str) -> int | None:
    try:
        return definition.entity_matrices.literal_index(key)
    except KeyError:
        return None


@dataclass(frozen=True)
class PlaceFacts:
    """Every static fact of every place (stage c1): the place literal matrix with its columns
    filled, and each place feature's column, free and derived."""

    literals: np.ndarray
    values: dict[str, np.ndarray]


def place_facts(definition: RuntimeDefinition) -> PlaceFacts:
    """The places' static facts, computed on the first call and cached on the definition."""
    cached = definition.cache.get("place_static")
    if cached is not None:
        return cached
    if not definition.has_places:
        raise WorldError("the definition has no places")
    assert definition.place_matrices is not None
    assert definition.place_free_values is not None and definition.place_scalar_values is not None
    n = definition.place_count
    free_index = {label: j for j, label in enumerate(definition.place_free_features)}
    scalar_index = {label: j for j, label in enumerate(definition.place_scalars)}
    literals = definition.place_matrices.literal_matrix(n)
    for i, spec in enumerate(definition.literals):
        reads = spec.reads
        if reads["role"] is not None:
            continue
        if reads["kind"] == "feature" and reads["feature"] in free_index:
            literals[:, i] = definition.place_free_values[:, free_index[reads["feature"]]]
        elif reads["kind"] == "threshold" and reads["scalar"] in scalar_index:
            column = definition.place_scalar_values[:, scalar_index[reads["scalar"]]]
            literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    outputs = definition.place_matrices.evaluate(literals)
    values = {label: definition.place_free_values[:, j] for label, j in free_index.items()}
    for label in definition.place_derived_features:
        values[label] = outputs[label]
        try:
            literals[:, definition.place_matrices.literal_index(label)] = outputs[label]
        except KeyError:
            pass
    facts = PlaceFacts(literals, values)
    definition.cache["place_static"] = facts
    return facts


# ---------------------------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------------------------


def _check_state(definition: RuntimeDefinition, state: State) -> None:
    expected = (definition.entity_count, len(definition.base_fluents))
    if state.values.shape != expected:
        raise WorldError(
            f"the state has shape {state.values.shape}; the definition needs {expected}"
        )
    numeric = (definition.entity_count, len(definition.numeric_fluents))
    if state.numeric.shape != numeric:
        raise WorldError(
            f"the state's numeric part has shape {state.numeric.shape}; the definition needs "
            f"{numeric}"
        )


def _entity_rows(definition: RuntimeDefinition, entities: Sequence[int] | None) -> np.ndarray:
    if entities is None:
        return np.arange(definition.entity_count, dtype=np.intp)
    rows = np.asarray(list(entities), dtype=np.intp).reshape(-1)
    if rows.size and (rows.min() < 0 or rows.max() >= definition.entity_count):
        raise WorldError(f"an entity index is outside 0..{definition.entity_count - 1}")
    return rows


def derive(
    definition: RuntimeDefinition, state: State, entities: Sequence[int] | None = None
) -> DerivedFacts:
    """The derived static facts (cached once per definition) and the derived fluents of the
    given entities (every entity when None) in ``state``."""
    _check_state(definition, state)
    rows = _entity_rows(definition, entities)
    facts = static_facts(definition)
    literals = np.array(facts.literals[rows], dtype=np.uint8, copy=True)
    for i, spec in enumerate(definition.literals):
        reads = spec.reads
        if reads["role"] is not None:
            continue
        if reads["kind"] == "fluent":
            fluent = reads["fluent"]
            if definition.is_base_fluent(fluent):
                literals[:, i] = state.values[rows, definition.base_fluent_index(fluent)]
        elif reads["kind"] == "numeric_threshold":
            column = state.numeric[rows, definition.numeric_index(reads["fluent"])]
            literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    outputs = definition.entity_matrices.evaluate(literals)
    fluent = (
        np.stack([outputs[label] for label in definition.derived_fluents], axis=1)
        if definition.derived_fluents
        else np.zeros((rows.size, 0), dtype=np.uint8)
    )
    return DerivedFacts(
        tuple(int(r) for r in rows),
        definition.derived_features,
        facts.derived[rows],
        definition.derived_fluents,
        fluent,
    )


# ---------------------------------------------------------------------------------------------
# able and legal
# ---------------------------------------------------------------------------------------------


def _event_type(definition: RuntimeDefinition, label: str) -> EventTypeRecord:
    et = definition.event_type(label)
    if et.is_category:
        raise WorldError(f"{label} is a category of event types, not an event type")
    return et


def _bindings_array(
    definition: RuntimeDefinition, et: EventTypeRecord, bindings: Sequence[Binding] | np.ndarray
) -> np.ndarray:
    array = np.asarray(bindings, dtype=np.intp)
    if array.size == 0:
        return np.zeros((0, et.arity), dtype=np.intp)
    if array.ndim != 2 or array.shape[1] != et.arity:
        what = "an entity and a place" if et.place else f"{et.arity} entities"
        raise WorldError(
            f"a binding of {et.label} names {what} ({', '.join(et.roles)}); "
            f"got an array of shape {array.shape}"
        )
    if et.place:
        if array[:, 0].min() < 0 or array[:, 0].max() >= definition.entity_count:
            raise WorldError(
                f"a binding names an entity index outside 0..{definition.entity_count - 1}"
            )
        if array[:, 1].min() < 0 or array[:, 1].max() >= definition.place_count:
            raise WorldError(
                f"a binding names a place index outside 0..{definition.place_count - 1}"
            )
        return array
    if array.min() < 0 or array.max() >= definition.entity_count:
        raise WorldError(
            f"a binding names an entity index outside 0..{definition.entity_count - 1}"
        )
    if et.arity == 2 and np.any(array[:, 0] == array[:, 1]):
        raise WorldError("a binding never uses one entity twice")
    return array


def _place_binding_literals(definition: RuntimeDefinition, array: np.ndarray) -> np.ndarray:
    """The base literal matrix of bindings of an entity (``array[:, 0]``) and a place
    (``array[:, 1]``): the agent's features and thresholds from the entities' static facts, the
    place's from the places', comparisons of an entity scalar with a place scalar, constraint
    outputs 0."""
    assert definition.place_binding_matrices is not None
    facts = static_facts(definition)
    places = place_facts(definition)
    scalar_index = {label: j for j, label in enumerate(definition.scalars)}
    place_scalar_index = {label: j for j, label in enumerate(definition.place_scalars)}
    agents, patients = array[:, 0], array[:, 1]
    literals = definition.place_binding_matrices.literal_matrix(array.shape[0])
    assert definition.place_scalar_values is not None
    for i, spec in enumerate(definition.literals):
        reads = spec.reads
        role = reads["role"]
        if role is None or reads["kind"] == "constraint":
            continue
        if reads["kind"] == "comparison":
            if reads["patient_scalar"] not in place_scalar_index:
                continue
            a = definition.scalar_values[agents, scalar_index[reads["agent_scalar"]]]
            p = definition.place_scalar_values[
                patients, place_scalar_index[reads["patient_scalar"]]
            ]
            difference = a - p
            holds = difference > reads["low"]
            if reads["high"] is not None:
                holds &= difference < reads["high"]
            literals[:, i] = holds.astype(np.uint8)
        elif role == "agent":
            if reads["kind"] == "feature":
                literals[:, i] = facts.values[reads["feature"]][agents]
            else:
                column = definition.scalar_values[agents, scalar_index[reads["scalar"]]]
                literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
        elif reads["kind"] == "feature":
            if reads["feature"] in places.values:
                literals[:, i] = places.values[reads["feature"]][patients]
        elif reads["scalar"] in place_scalar_index:
            column = definition.place_scalar_values[patients, place_scalar_index[reads["scalar"]]]
            literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    return literals


def place_able_table(definition: RuntimeDefinition) -> dict[str, np.ndarray]:
    """Every place event type's requirement over every binding of an entity and a place, as a
    bool matrix of shape ``(entities, places)``, computed once and cached on the definition."""
    cached = definition.cache.get("place_able")
    if cached is not None:
        return cached
    table: dict[str, np.ndarray] = {}
    if definition.has_places:
        n, m = definition.entity_count, definition.place_count
        agents = np.repeat(np.arange(n, dtype=np.intp), m)
        patients = np.tile(np.arange(m, dtype=np.intp), n)
        pairs = np.stack([agents, patients], axis=1)
        outputs: dict[str, list[np.ndarray]] = {}
        for start in range(0, pairs.shape[0], ABLE_CHUNK_ROWS):
            chunk = pairs[start : start + ABLE_CHUNK_ROWS]
            part = definition.place_binding_matrices.evaluate(  # type: ignore[union-attr]
                _place_binding_literals(definition, chunk)
            )
            for key, column in part.items():
                outputs.setdefault(key, []).append(column)
        for et in definition.event_types:
            if et.place:
                matrix = np.zeros((n, m), dtype=bool)
                matrix[agents, patients] = np.concatenate(outputs[et.requirement]).astype(bool)
                table[et.label] = matrix
    definition.cache["place_able"] = table
    return table


def _binding_literals(
    definition: RuntimeDefinition, roles: Sequence[str], array: np.ndarray
) -> np.ndarray:
    """The base literal matrix of the bindings (one column of ``array`` per role): features and
    thresholds of each role from the static facts, comparisons from the scalars, constraint
    outputs 0 (the layers compute them). A role the bindings lack leaves its literals 0; a
    one-place requirement never reads them."""
    facts = static_facts(definition)
    scalar_index = {label: j for j, label in enumerate(definition.scalars)}
    rows = {role: array[:, k] for k, role in enumerate(roles)}
    literals = definition.binding_matrices.literal_matrix(array.shape[0])
    for i, spec in enumerate(definition.literals):
        reads = spec.reads
        role = reads["role"]
        if role is None or reads["kind"] == "constraint":
            continue
        if reads["kind"] == "comparison":
            if "patient" not in rows or reads["patient_scalar"] not in scalar_index:
                continue
            a = definition.scalar_values[rows["agent"], scalar_index[reads["agent_scalar"]]]
            p = definition.scalar_values[rows["patient"], scalar_index[reads["patient_scalar"]]]
            difference = a - p
            holds = difference > reads["low"]
            if reads["high"] is not None:
                holds &= difference < reads["high"]
            literals[:, i] = holds.astype(np.uint8)
        elif role in rows:
            if reads["kind"] == "feature":
                if reads["feature"] in facts.values:
                    literals[:, i] = facts.values[reads["feature"]][rows[role]]
            elif reads["scalar"] in scalar_index:
                column = definition.scalar_values[rows[role], scalar_index[reads["scalar"]]]
                literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    return literals


def able_table(definition: RuntimeDefinition) -> dict[str, np.ndarray]:
    """Every event type's requirement over every binding of the definition's entities, computed
    once on first use and cached on the definition: for a one-place event type a bool array over
    the entities, for a two-place one a bool matrix over ordered pairs (agent, patient), with the
    diagonal false. Requirements are static, so the table holds in every state."""
    cached = definition.cache.get("able")
    if cached is not None:
        return cached
    n = definition.entity_count
    table: dict[str, np.ndarray] = {}
    singles = np.arange(n, dtype=np.intp).reshape(-1, 1)
    outputs = definition.binding_matrices.evaluate(
        _binding_literals(definition, ("agent",), singles)
    )
    for et in definition.event_types:
        if et.arity == 1:
            table[et.label] = outputs[et.requirement].astype(bool)
    if any(et.arity == 2 and not et.place for et in definition.event_types):
        agents = np.repeat(np.arange(n, dtype=np.intp), n)
        patients = np.tile(np.arange(n, dtype=np.intp), n)
        keep = agents != patients
        pairs = np.stack([agents[keep], patients[keep]], axis=1)
        outputs = {}
        for start in range(0, pairs.shape[0], ABLE_CHUNK_ROWS):
            chunk = pairs[start : start + ABLE_CHUNK_ROWS]
            part = definition.binding_matrices.evaluate(
                _binding_literals(definition, ("agent", "patient"), chunk)
            )
            for key, column in part.items():
                outputs.setdefault(key, []).append(column)
        for et in definition.event_types:
            if et.arity == 2 and not et.place:
                matrix = np.zeros((n, n), dtype=bool)
                matrix[pairs[:, 0], pairs[:, 1]] = np.concatenate(outputs[et.requirement]).astype(
                    bool
                )
                table[et.label] = matrix
    definition.cache["able"] = table
    return table


def able(
    definition: RuntimeDefinition, event_type: str, bindings: Sequence[Binding] | np.ndarray
) -> np.ndarray:
    """Whether each binding's requirement holds (a bool array, one entry per binding). Static:
    the answer is the same in every state, and comes from the cached table."""
    et = _event_type(definition, event_type)
    array = _bindings_array(definition, et, bindings)
    if array.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if et.place:
        return place_able_table(definition)[et.label][array[:, 0], array[:, 1]]
    table = able_table(definition)[et.label]
    if et.arity == 1:
        return table[array[:, 0]]
    return table[array[:, 0], array[:, 1]]


def check_definition_agreement(
    definition: RuntimeDefinition,
) -> tuple[AgreementReport, AgreementReport]:
    """The agreement test (REL.18) over a loaded definition: every entity rule on every entity
    (with the initial base fluents, or none when the entities table has none), and every binding
    rule on every ordered pair of entities. Returns the entity and binding reports, and raises
    :class:`AgreementError` on a disagreement."""
    facts = static_facts(definition)
    literals = np.array(facts.literals, dtype=np.uint8, copy=True)
    if definition.initial_values is not None:
        state = State(definition.initial_values, definition.initial_numeric)
        for i, spec in enumerate(definition.literals):
            reads = spec.reads
            if reads["role"] is not None:
                continue
            if reads["kind"] == "fluent":
                if definition.is_base_fluent(reads["fluent"]):
                    literals[:, i] = state.values[:, definition.base_fluent_index(reads["fluent"])]
            elif reads["kind"] == "numeric_threshold" and definition.initial_numeric is not None:
                column = state.numeric[:, definition.numeric_index(reads["fluent"])]
                literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    entity_report = check_agreement(definition.entity_matrices, definition.entity_rules, literals)
    n = definition.entity_count
    agents = np.repeat(np.arange(n), n)
    patients = np.tile(np.arange(n), n)
    keep = agents != patients
    array = np.stack([agents[keep], patients[keep]], axis=1)
    binding_literals = _binding_literals(definition, ("agent", "patient"), array)
    binding_report = check_agreement(
        definition.binding_matrices, definition.binding_rules, binding_literals
    )
    if definition.has_places:
        assert definition.place_matrices is not None
        assert definition.place_binding_matrices is not None
        check_agreement(
            definition.place_matrices, definition.place_rules, place_facts(definition).literals
        )
        m = definition.place_count
        place_array = np.stack(
            [np.repeat(np.arange(n), m), np.tile(np.arange(m), n)], axis=1
        ).astype(np.intp)
        check_agreement(
            definition.place_binding_matrices,
            definition.place_binding_rules,
            _place_binding_literals(definition, place_array),
        )
    return entity_report, binding_report


def _precondition_holds(
    definition: RuntimeDefinition, state: State, et: EventTypeRecord, array: np.ndarray
) -> np.ndarray:
    """Whether each binding's precondition holds in ``state``: a bool array. Derived fluents
    are computed over the bound entities only."""
    holds = np.ones(array.shape[0], dtype=bool)
    if not et.precondition or array.shape[0] == 0:
        return holds
    needed = np.unique(array)
    facts = (
        derive(definition, state, needed)
        if any(
            not lit.numeric and not definition.is_base_fluent(lit.fluent) for lit in et.precondition
        )
        else None
    )
    for lit in et.precondition:
        rows = array[:, et.roles.index(lit.role)]
        if lit.numeric:
            column = state.numeric[rows, definition.numeric_index(lit.fluent)]
            holds &= (column > lit.threshold) if lit.value else (column <= lit.threshold)
            continue
        if definition.is_base_fluent(lit.fluent):
            values = state.values[rows, definition.base_fluent_index(lit.fluent)]
        else:
            assert facts is not None
            positions = np.searchsorted(needed, rows)
            values = facts.fluent[positions, definition.derived_fluent_index(lit.fluent)]
        holds &= values == int(lit.value)
    return holds


def legal(
    definition: RuntimeDefinition,
    state: State,
    event_type: str,
    bindings: Sequence[Binding] | np.ndarray,
) -> np.ndarray:
    """Whether each binding is legal in ``state``: its requirement holds and its precondition
    holds."""
    _check_state(definition, state)
    et = _event_type(definition, event_type)
    array = _bindings_array(definition, et, bindings)
    return able(definition, event_type, array) & _precondition_holds(definition, state, et, array)


def legal_bindings(
    definition: RuntimeDefinition,
    state: State,
    event_type: str,
    entities: Sequence[int] | None = None,
    places: Sequence[int] | None = None,
) -> list[Binding]:
    """Every legal binding of the event type among the given entities (every entity when None),
    in a fixed order: by agent, then patient, in entity order (the order of the entities table,
    whatever order the entities are given in). For a place event type (stage c1), the patients
    are the given places (every place when None), in place order, after the agent."""
    _check_state(definition, state)
    et = _event_type(definition, event_type)
    rows = np.unique(_entity_rows(definition, entities))
    if et.place:
        place_rows = (
            np.arange(definition.place_count, dtype=np.intp)
            if places is None
            else np.unique(np.asarray(places, dtype=np.intp))
        )
        if place_rows.size and (place_rows.min() < 0 or place_rows.max() >= definition.place_count):
            raise WorldError(f"a place index is outside 0..{definition.place_count - 1}")
        array = np.stack(
            [np.repeat(rows, place_rows.size), np.tile(place_rows, rows.size)], axis=1
        ).reshape(-1, 2)
    elif et.arity == 1:
        array = rows.reshape(-1, 1)
    else:
        agents = np.repeat(rows, rows.size)
        patients = np.tile(rows, rows.size)
        keep = agents != patients
        array = np.stack([agents[keep], patients[keep]], axis=1)
    mask = legal(definition, state, event_type, array)
    return [tuple(int(i) for i in binding) for binding in array[mask]]


# ---------------------------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------------------------


def _check_event(definition: RuntimeDefinition, event: Event) -> EventTypeRecord:
    """The event's event type, after checking that the event is well formed. A malformed event
    is an illegal event."""
    text = event.describe(definition)
    try:
        et = definition.event_type(event.event_type)
    except WorldError:
        raise IllegalEventError(
            f"{text} is illegal: {event.event_type} is not an event type of the definition"
        ) from None
    if et.is_category:
        raise IllegalEventError(
            f"{text} is illegal: {et.label} is a category of event types, not an event type"
        )
    if et.place:
        raise WorldError(
            f"{et.label} is a place event type, which has no precondition or effects in stage "
            f"c1; no event of it can be applied"
        )
    if len(event.binding) != et.arity:
        raise IllegalEventError(
            f"{text} is illegal: {et.label} has the roles {', '.join(et.roles)}, but the "
            f"binding names {len(event.binding)} entities"
        )
    for i in event.binding:
        if not 0 <= int(i) < definition.entity_count:
            raise IllegalEventError(
                f"{text} is illegal: entity index {i} is outside the entities table"
            )
    if len(set(event.binding)) != len(event.binding):
        raise IllegalEventError(f"{text} is illegal: a binding never uses one entity twice")
    return et


def _check_legal(
    definition: RuntimeDefinition, state: State, event: Event, et: EventTypeRecord
) -> None:
    array = np.asarray([event.binding], dtype=np.intp)
    text = event.describe(definition)
    if not able(definition, et.label, array)[0]:
        raise IllegalEventError(
            f"{text} is illegal: its requirement {et.requirement} does not hold"
        )
    if not _precondition_holds(definition, state, et, array)[0]:
        facts = derive(definition, state, list(event.binding))
        for lit in et.precondition:
            row = et.roles.index(lit.role)
            entity = event.binding[row]
            if lit.numeric:
                number = float(state.numeric[entity, definition.numeric_index(lit.fluent)])
                if not lit.holds_for(number):
                    raise IllegalEventError(
                        f"{text} is illegal: the precondition literal {lit.text} does not hold "
                        f"({lit.fluent} of {definition.entity_labels[entity]} is {number!r})"
                    )
                continue
            if definition.is_base_fluent(lit.fluent):
                value = state.values[entity, definition.base_fluent_index(lit.fluent)]
            else:
                value = facts.fluent[row, definition.derived_fluent_index(lit.fluent)]
            if int(value) != int(lit.value):
                raise IllegalEventError(
                    f"{text} is illegal: the precondition literal {lit.text} does not hold "
                    f"({lit.fluent} of {definition.entity_labels[entity]} is {int(value)})"
                )
        raise AssertionError("the precondition fails but every literal holds")


def _writes(event: Event, et: EventTypeRecord) -> dict[tuple[int, str], bool | float]:
    """The base fluents the event may write, ``{(entity, fluent): value}`` (a numeric effect's
    signed constant for a numeric fluent): every effect counts, whether or not its condition
    holds in the state."""
    return {
        (event.binding[et.roles.index(effect.role)], effect.fluent): effect.value
        for effect in et.effects
    }


def _reads_by_reader(
    definition: RuntimeDefinition, event: Event, et: EventTypeRecord
) -> dict[tuple[int, str], str]:
    """The base fluents the event's precondition and its effects' conditions read, through the
    cone of every derived fluent: ``{(entity, base fluent): reader}``, the reader being ``"the
    precondition"`` or ``"a condition"`` (the precondition when both read). A static literal of
    a condition reads nothing that an event can write."""
    found: dict[tuple[int, str], str] = {}
    for effect in et.effects:
        for lit in effect.condition:
            if lit.kind == "feature":
                continue
            entity = event.binding[et.roles.index(lit.role)]
            for base in definition.cone(lit.symbol):
                found[(entity, base)] = "a condition"
    for lit in et.precondition:
        entity = event.binding[et.roles.index(lit.role)]
        for base in definition.cone(lit.fluent):
            found[(entity, base)] = "the precondition"
    return found


def _reads(
    definition: RuntimeDefinition, event: Event, et: EventTypeRecord
) -> set[tuple[int, str]]:
    """The base fluents the event's precondition and its effects' conditions read, through the
    cone of every derived fluent: ``{(entity, base fluent)}``."""
    return set(_reads_by_reader(definition, event, et))


def _condition_holds(
    definition: RuntimeDefinition,
    state: State,
    event: Event,
    et: EventTypeRecord,
    effect: Effect,
    facts: DerivedFacts | None,
) -> bool:
    """Whether the effect's condition holds for the event's binding in ``state``: every literal
    has its value, a static feature from the static facts, a base fluent from the state, a
    derived fluent from ``facts`` (the derived facts of the bound entities in the state). An
    effect without a condition always holds."""
    statics = static_facts(definition)
    for lit in effect.condition:
        row = et.roles.index(lit.role)
        entity = event.binding[row]
        if lit.numeric:
            if not lit.holds_for(
                float(state.numeric[entity, definition.numeric_index(lit.symbol)])
            ):
                return False
            continue
        if lit.kind == "feature":
            value = statics.values[lit.symbol][entity]
        elif definition.is_base_fluent(lit.symbol):
            value = state.values[entity, definition.base_fluent_index(lit.symbol)]
        else:
            assert facts is not None
            value = facts.fluent[row, definition.derived_fluent_index(lit.symbol)]
        if int(value) != int(lit.value):
            return False
    return True


def _fired(
    definition: RuntimeDefinition, state: State, event: Event, et: EventTypeRecord
) -> dict[tuple[int, str], bool | float]:
    """The base fluents the event writes in ``state``: its effects whose conditions hold there,
    judged before any effect of the step (a numeric effect's value is its signed constant)."""
    facts = None
    if any(
        lit.kind == "fluent" and not definition.is_base_fluent(lit.symbol)
        for effect in et.effects
        for lit in effect.condition
    ):
        facts = derive(definition, state, list(event.binding))
    return {
        (event.binding[et.roles.index(effect.role)], effect.fluent): effect.value
        for effect in et.effects
        if _condition_holds(definition, state, event, et, effect, facts)
    }


def _fact(definition: RuntimeDefinition, key: tuple[int, str]) -> str:
    return f"{key[1]} of {definition.entity_labels[key[0]]}"


def interferes(definition: RuntimeDefinition, a: Event, b: Event) -> bool:
    """Whether two well-formed events interfere: both write the same fluent of the same entity
    (an effect counts whatever its condition), one writes a base fluent that the other's
    precondition or a condition of one of its effects reads (through the cones of derived
    fluents), or the two are the same event."""
    if a == b:
        return True
    et_a, et_b = definition.event_type(a.event_type), definition.event_type(b.event_type)
    writes_a, writes_b = set(_writes(a, et_a)), set(_writes(b, et_b))
    if writes_a & writes_b:
        return True
    return bool(writes_a & _reads(definition, b, et_b)) or bool(
        writes_b & _reads(definition, a, et_a)
    )


@dataclass(frozen=True)
class StepResult:
    """What one step computes: the state after it, and the numeric values after the effects
    and before the drift (shape ``(entities, numeric fluents)``), which histories record as the
    effects' changes."""

    state: State
    after_effects: np.ndarray


def apply(
    definition: RuntimeDefinition, state: State, events: Iterable[Event], drift: bool = True
) -> State:
    """The state after one step. Every event must be legal in ``state``, and the events must
    not interfere; otherwise :class:`IllegalEventError` or :class:`InterferenceError` names the
    event and the reason. The effects whose conditions hold in ``state`` are applied (every
    condition is judged in the starting state, before any effect of the step); unwritten base
    fluents keep their values; each numeric value becomes ``clamp(clamp(v + e) + d)``. With
    ``drift`` false the effects alone are applied, each clamped, and the drift is left out: the
    device by which a step's events can be applied one at a time (the drift is added once per
    step, not once per event)."""
    return apply_step(definition, state, events, drift).state


def apply_step(
    definition: RuntimeDefinition, state: State, events: Iterable[Event], drift: bool = True
) -> StepResult:
    """``apply``, with the numeric values after the effects and before the drift beside the
    state after the step."""
    _check_state(definition, state)
    events = tuple(events)
    types = [_check_event(definition, event) for event in events]
    for event, et in zip(events, types, strict=True):
        _check_legal(definition, state, event, et)
    writes = [_writes(event, et) for event, et in zip(events, types, strict=True)]
    reads = [
        _reads_by_reader(definition, event, et) for event, et in zip(events, types, strict=True)
    ]
    for a in range(len(events)):
        for b in range(a + 1, len(events)):
            first, second = events[a].describe(definition), events[b].describe(definition)
            if events[a] == events[b]:
                raise InterferenceError(f"{first} occurs twice in the step")
            shared = sorted(set(writes[a]) & set(writes[b]))
            if shared:
                raise InterferenceError(
                    f"{first} and {second} interfere: both write {_fact(definition, shared[0])}"
                )
            for writer, reader, w, r in (
                (first, second, writes[a], reads[b]),
                (second, first, writes[b], reads[a]),
            ):
                shared = sorted(set(w) & set(r))
                if shared:
                    raise InterferenceError(
                        f"{writer} and {reader} interfere: {writer} writes "
                        f"{_fact(definition, shared[0])}, which {r[shared[0]]} of {reader} reads"
                    )
    values = np.array(state.values, dtype=np.uint8, copy=True)
    numeric = np.array(state.numeric, dtype=np.float64, copy=True)
    low, high = definition.range_bounds
    for event, et in zip(events, types, strict=True):
        for (entity, fluent), value in _fired(definition, state, event, et).items():
            if definition.is_numeric_fluent(fluent):
                j = definition.numeric_index(fluent)
                numeric[entity, j] = clamp(
                    float(numeric[entity, j]) + float(value), low[j], high[j]
                )
            else:
                values[entity, definition.base_fluent_index(fluent)] = int(value)
    after_effects = np.array(numeric, copy=True)
    if numeric.shape[1] and drift:
        numeric = np.minimum(np.maximum(numeric + definition.drift_vector[None, :], low), high)
    return StepResult(State(values, numeric), after_effects)


def fired_effects(definition: RuntimeDefinition, state: State, event: Event) -> tuple[Effect, ...]:
    """The effects of a well-formed event whose conditions hold in ``state`` (every
    unconditional effect among them): what the event writes when it is applied in that state."""
    et = _check_event(definition, event)
    facts = None
    if any(
        lit.kind == "fluent" and not definition.is_base_fluent(lit.symbol)
        for effect in et.effects
        for lit in effect.condition
    ):
        facts = derive(definition, state, list(event.binding))
    return tuple(
        effect
        for effect in et.effects
        if _condition_holds(definition, state, event, et, effect, facts)
    )


# ---------------------------------------------------------------------------------------------
# Comparisons (phase (b): the world)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mean:
    """A category's mean on a dimension, as an argument of a comparison: ``MEAN(CATEGORY.1.4)``,
    or ``MEAN(ENTITY)`` for all entities."""

    category: str


Argument = int | Mean
"""An argument of a comparison: an entity (by index) or a category mean."""


def is_place_class(category: str) -> bool:
    """Whether a comparison class is a class of places: ``EVERYPLACE`` or a place category."""
    return category == EVERYPLACE or category.startswith(PLACE_CATEGORY_PREFIX)


def _leaf_members(leaves: Sequence[str], category: str) -> list[int]:
    prefix = category + "."
    return [i for i, leaf in enumerate(leaves) if leaf == category or leaf.startswith(prefix)]


def members(definition: RuntimeDefinition, category: str) -> np.ndarray:
    """The indices of the entities below a category: those whose leaf is the category or below
    it, read off the leaf labels (``CATEGORY.1.3.2`` is below ``CATEGORY.1.3``); every entity
    for ``ENTITY``. For a class of places (stage c1), ``EVERYPLACE`` or a place category, the
    indices of the places below it."""
    if category == ENTITY:
        return np.arange(definition.entity_count, dtype=np.intp)
    if category == EVERYPLACE:
        return np.arange(definition.place_count, dtype=np.intp)
    leaves = definition.place_leaves if is_place_class(category) else definition.leaves
    found = _leaf_members(leaves, category)
    if not found:
        what = "places" if is_place_class(category) else "entities"
        raise WorldError(f"{category} is not a category of the definition's {what}")
    return np.array(found, dtype=np.intp)


def _ancestor_categories(leaves: Sequence[str]) -> tuple[str, ...]:
    found: dict[str, None] = {}
    for leaf in leaves:
        parts = leaf.split(".")
        for k in range(2, len(parts) + 1):
            found.setdefault(".".join(parts[:k]), None)
    return tuple(
        sorted(found, key=lambda c: tuple(int(p) if p.isdigit() else p for p in c.split(".")))
    )


def categories(definition: RuntimeDefinition) -> tuple[str, ...]:
    """Every category the leaves name, from the top down, with every ancestor: the comparison
    classes beside ``ENTITY``."""
    return _ancestor_categories(definition.leaves)


def place_categories(definition: RuntimeDefinition) -> tuple[str, ...]:
    """Every category of places the place leaves name (stage c1): the comparison classes beside
    ``EVERYPLACE``."""
    return _ancestor_categories(definition.place_leaves)


def comparison_classes(
    definition: RuntimeDefinition,
) -> dict[tuple[str, str], tuple[int, float, float]]:
    """For every category and ``ENTITY``, and every scalar: the number of instances, the mean,
    and the population standard deviation of the scalar over the instances below the category
    (``derived/comparison_classes.csv`` holds the same figures); with places, ``EVERYPLACE``
    and every place category on every place scalar too. Computed once and cached."""
    cached = definition.cache.get("comparison_classes")
    if cached is not None:
        return cached
    table: dict[tuple[str, str], tuple[int, float, float]] = {}
    for category in (ENTITY, *categories(definition)):
        rows = members(definition, category)
        for j, scalar in enumerate(definition.scalars):
            column = definition.scalar_values[rows, j]
            table[(category, scalar)] = (int(rows.size), float(column.mean()), float(column.std()))
    if definition.has_places:
        assert definition.place_scalar_values is not None
        for category in (EVERYPLACE, *place_categories(definition)):
            rows = members(definition, category)
            for j, scalar in enumerate(definition.place_scalars):
                column = definition.place_scalar_values[rows, j]
                table[(category, scalar)] = (
                    int(rows.size),
                    float(column.mean()),
                    float(column.std()),
                )
    definition.cache["comparison_classes"] = table
    return table


def _dimension_column(
    definition: RuntimeDefinition, dimension: str, state: State | None
) -> np.ndarray:
    if dimension in definition.scalars:
        return definition.scalar_values[:, definition.scalars.index(dimension)]
    if definition.is_place_scalar(dimension):
        assert definition.place_scalar_values is not None
        return definition.place_scalar_values[:, definition.place_scalars.index(dimension)]
    if definition.is_numeric_fluent(dimension):
        if state is None:
            raise WorldError(f"a comparison of {dimension} needs a state: it is a numeric fluent")
        _check_state(definition, state)
        return state.numeric[:, definition.numeric_index(dimension)]
    raise WorldError(f"{dimension} is not a scalar dimension or a numeric fluent")


def _check_class_of_dimension(definition: RuntimeDefinition, dimension: str, category: str) -> None:
    """A comparison never mixes an entity scalar with a place scalar: a class of places goes
    with a place scalar, and a class of entities with an entity scalar or a numeric fluent."""
    place_dimension = definition.is_place_scalar(dimension)
    if place_dimension != is_place_class(category):
        raise WorldError(
            f"a comparison of {dimension} cannot take {category}: a place scalar takes a class "
            f"of places (EVERYPLACE or a PLACECATEGORY), and an entity dimension a class of "
            f"entities"
        )


def value_of(
    definition: RuntimeDefinition, dimension: str, x: Argument, state: State | None = None
) -> float:
    """An argument's value on a dimension: the entity's value, or the mean over a category's
    members, on the static scalars, or on a numeric fluent in ``state``; on a place scalar
    (stage c1), a place's value by its index, or the mean of a class of places."""
    column = _dimension_column(definition, dimension, state)
    if isinstance(x, Mean):
        _check_class_of_dimension(definition, dimension, x.category)
        return float(column[members(definition, x.category)].mean())
    count = (
        definition.place_count if definition.is_place_scalar(dimension) else definition.entity_count
    )
    if not 0 <= int(x) < count:
        what = "a place" if definition.is_place_scalar(dimension) else "an entity"
        raise WorldError(f"{what} index is outside 0..{count - 1}")
    return float(column[int(x)])


def greater(
    definition: RuntimeDefinition,
    dimension: str,
    x: Argument,
    y: Argument,
    state: State | None = None,
) -> bool:
    """``GREATER(dimension, x, y)``: ``x`` exceeds ``y`` on the dimension, strictly, so two
    equal values give false both ways."""
    return value_of(definition, dimension, x, state) > value_of(definition, dimension, y, state)


def above(
    definition: RuntimeDefinition,
    dimension: str,
    x: Argument,
    standard: str,
    z: float,
    state: State | None = None,
) -> bool:
    """``ABOVE(dimension, x, K, z)``: on a scalar, the value is at least ``z`` standard
    deviations above the mean of the comparison class ``K`` (a category, or ``ENTITY``); a class
    with no spread has nothing above it. On a numeric fluent, with ``RANGE`` as the standard,
    the value is strictly greater than the point at the fraction ``z`` of the way through the
    fluent's range: ``above`` at ``z`` is the numeric literal ``> point`` and ``below`` at ``z``
    its negation ``<= point`` (Jon's ruling WM.35, October 15, 2026), so a value exactly at the
    point is not above and is below."""
    value = value_of(definition, dimension, x, state)
    if definition.is_numeric_fluent(dimension):
        if standard != RANGE:
            raise WorldError(f"a comparison of {dimension} takes RANGE as its standard")
        low, high = definition.numeric_ranges[dimension]
        return value > low + z * (high - low)
    if standard == RANGE:
        raise WorldError(f"a comparison of the scalar {dimension} takes a comparison class")
    _check_class_of_dimension(definition, dimension, standard)
    _, mean, sd = comparison_classes(definition)[(standard, dimension)]
    return sd > 0 and value >= mean + z * sd


def below(
    definition: RuntimeDefinition,
    dimension: str,
    x: Argument,
    standard: str,
    z: float,
    state: State | None = None,
) -> bool:
    """``BELOW(dimension, x, K, z)``: at least ``z`` standard deviations below the mean of the
    comparison class, or, on a numeric fluent with ``RANGE``, at most the point at the fraction
    ``z`` of the way through the range (inclusive: the negation of ``above`` at ``z``)."""
    value = value_of(definition, dimension, x, state)
    if definition.is_numeric_fluent(dimension):
        if standard != RANGE:
            raise WorldError(f"a comparison of {dimension} takes RANGE as its standard")
        low, high = definition.numeric_ranges[dimension]
        return value <= low + z * (high - low)
    if standard == RANGE:
        raise WorldError(f"a comparison of the scalar {dimension} takes a comparison class")
    _check_class_of_dimension(definition, dimension, standard)
    _, mean, sd = comparison_classes(definition)[(standard, dimension)]
    return sd > 0 and value <= mean - z * sd


def increase(
    definition: RuntimeDefinition, fluent: str, x: int, before: State, after: State
) -> bool:
    """``INCREASE``: the entity's value of the numeric fluent is greater after the step than
    before it."""
    return value_of(definition, fluent, x, after) > value_of(definition, fluent, x, before)


def decrease(
    definition: RuntimeDefinition, fluent: str, x: int, before: State, after: State
) -> bool:
    """``DECREASE``: the value is less after the step than before it."""
    return value_of(definition, fluent, x, after) < value_of(definition, fluent, x, before)


__all__ = [
    "ENTITY",
    "EVERYPLACE",
    "RANGE",
    "Argument",
    "Binding",
    "DerivedFacts",
    "Event",
    "Mean",
    "State",
    "StaticFacts",
    "StepResult",
    "able",
    "able_table",
    "above",
    "apply",
    "apply_step",
    "below",
    "categories",
    "check_definition_agreement",
    "comparison_classes",
    "decrease",
    "derive",
    "fired_effects",
    "greater",
    "increase",
    "initial_state",
    "interferes",
    "legal",
    "legal_bindings",
    "is_place_class",
    "members",
    "place_able_table",
    "place_categories",
    "place_facts",
    "static_facts",
    "value_of",
]
