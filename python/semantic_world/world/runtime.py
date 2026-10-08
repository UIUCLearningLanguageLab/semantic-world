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
interfere, every effect is applied, unwritten base fluents keep their values (inertia), and
derived fluents are never carried over, only recomputed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from semantic_world.world.definition import EventTypeRecord, RuntimeDefinition
from semantic_world.world.errors import IllegalEventError, InterferenceError, WorldError
from semantic_world.world.matrices import AgreementReport, check_agreement

Binding = tuple[int, ...]


# ---------------------------------------------------------------------------------------------
# States, events, and derived facts
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """The base fluents of every entity at one time point: shape ``(entities, base fluents)``,
    values 0 and 1. The array is read-only; ``apply`` returns a new state."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.uint8, copy=True)
        if values.ndim != 2:
            raise WorldError(f"a state is a 2-D array (entities, base fluents), got {values.shape}")
        if values.size and values.max() > 1:
            raise WorldError("a state holds 0 and 1 only")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.values.shape == other.values.shape and bool(
            np.array_equal(self.values, other.values)
        )

    def __hash__(self) -> int:
        return hash((self.values.shape, self.values.tobytes()))

    @classmethod
    def from_true(cls, definition: RuntimeDefinition, true: Mapping[str, Iterable[str]]) -> State:
        """A state from the base fluents that are true, as ``{entity label: [fluent labels]}``.
        Every entity of the definition must be listed; a fluent not listed is false."""
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
        return cls(values)

    def true_fluents(self, definition: RuntimeDefinition) -> dict[str, list[str]]:
        """The base fluents that are true, as ``{entity label: [fluent labels]}``, every entity
        listed, fluents in definition order."""
        _check_state(definition, self)
        return {
            label: [f for j, f in enumerate(definition.base_fluents) if self.values[i, j]]
            for i, label in enumerate(definition.entity_labels)
        }


def initial_state(definition: RuntimeDefinition) -> State:
    """The state at ``TIME.1`` of every episode: the initial base fluents of the entities table."""
    if definition.initial_values is None:
        raise WorldError("the definition's entities table carries no initial fluents")
    return State(definition.initial_values)


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
        elif reads["kind"] == "threshold":
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


# ---------------------------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------------------------


def _check_state(definition: RuntimeDefinition, state: State) -> None:
    expected = (definition.entity_count, len(definition.base_fluents))
    if state.values.shape != expected:
        raise WorldError(
            f"the state has shape {state.values.shape}; the definition needs {expected}"
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
        if reads["role"] is None and reads["kind"] == "fluent":
            fluent = reads["fluent"]
            if definition.is_base_fluent(fluent):
                literals[:, i] = state.values[rows, definition.base_fluent_index(fluent)]
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
        raise WorldError(
            f"a binding of {et.label} names {et.arity} entities ({', '.join(et.roles)}); "
            f"got an array of shape {array.shape}"
        )
    if array.min() < 0 or array.max() >= definition.entity_count:
        raise WorldError(
            f"a binding names an entity index outside 0..{definition.entity_count - 1}"
        )
    if et.arity == 2 and np.any(array[:, 0] == array[:, 1]):
        raise WorldError("a binding never uses one entity twice")
    return array


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
            if "patient" not in rows:
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
                literals[:, i] = facts.values[reads["feature"]][rows[role]]
            else:
                column = definition.scalar_values[rows[role], scalar_index[reads["scalar"]]]
                literals[:, i] = (column > reads["threshold"]).astype(np.uint8)
    return literals


def able(
    definition: RuntimeDefinition, event_type: str, bindings: Sequence[Binding] | np.ndarray
) -> np.ndarray:
    """Whether each binding's requirement holds (a bool array, one entry per binding). Static:
    the answer is the same in every state."""
    et = _event_type(definition, event_type)
    array = _bindings_array(definition, et, bindings)
    if array.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    outputs = definition.binding_matrices.evaluate(_binding_literals(definition, et.roles, array))
    return outputs[et.requirement].astype(bool)


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
        state = State(definition.initial_values)
        for i, spec in enumerate(definition.literals):
            reads = spec.reads
            if reads["role"] is None and reads["kind"] == "fluent":
                if definition.is_base_fluent(reads["fluent"]):
                    literals[:, i] = state.values[:, definition.base_fluent_index(reads["fluent"])]
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
    return entity_report, binding_report


def _precondition_holds(
    definition: RuntimeDefinition, state: State, et: EventTypeRecord, array: np.ndarray
) -> np.ndarray:
    """Whether each binding's precondition holds in ``state``: a bool array, and for the failing
    bindings nothing more (``_first_failing_literal`` names the literal)."""
    holds = np.ones(array.shape[0], dtype=bool)
    if not et.precondition or array.shape[0] == 0:
        return holds
    facts = (
        derive(definition, state)
        if any(not definition.is_base_fluent(lit.fluent) for lit in et.precondition)
        else None
    )
    for lit in et.precondition:
        rows = array[:, et.roles.index(lit.role)]
        holds &= _fluent_values(definition, state, facts, lit.fluent, rows) == int(lit.value)
    return holds


def _fluent_values(
    definition: RuntimeDefinition,
    state: State,
    facts: DerivedFacts | None,
    fluent: str,
    rows: np.ndarray,
) -> np.ndarray:
    if definition.is_base_fluent(fluent):
        return state.values[rows, definition.base_fluent_index(fluent)]
    assert facts is not None
    return facts.fluent[rows, definition.derived_fluent_index(fluent)]


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
) -> list[Binding]:
    """Every legal binding of the event type among the given entities (every entity when None),
    in a fixed order: by agent, then patient, in entity order (the order of the entities table,
    whatever order the entities are given in)."""
    _check_state(definition, state)
    et = _event_type(definition, event_type)
    rows = np.unique(_entity_rows(definition, entities))
    if et.arity == 1:
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


def _writes(event: Event, et: EventTypeRecord) -> dict[tuple[int, str], bool]:
    """The base fluents the event writes, ``{(entity, fluent): value}``."""
    return {
        (event.binding[et.roles.index(effect.role)], effect.fluent): effect.value
        for effect in et.effects
    }


def _reads(
    definition: RuntimeDefinition, event: Event, et: EventTypeRecord
) -> set[tuple[int, str]]:
    """The base fluents the event's precondition reads, through the cone of every derived
    fluent: ``{(entity, base fluent)}``."""
    found: set[tuple[int, str]] = set()
    for lit in et.precondition:
        entity = event.binding[et.roles.index(lit.role)]
        for base in definition.cone(lit.fluent):
            found.add((entity, base))
    return found


def _fact(definition: RuntimeDefinition, key: tuple[int, str]) -> str:
    return f"{key[1]} of {definition.entity_labels[key[0]]}"


def apply(definition: RuntimeDefinition, state: State, events: Iterable[Event]) -> State:
    """The state after one step. Every event must be legal in ``state``, and the events must
    not interfere; otherwise :class:`IllegalEventError` or :class:`InterferenceError` names the
    event and the reason. Effects are applied; unwritten base fluents keep their values."""
    _check_state(definition, state)
    events = tuple(events)
    types = [_check_event(definition, event) for event in events]
    for event, et in zip(events, types, strict=True):
        _check_legal(definition, state, event, et)
    writes = [_writes(event, et) for event, et in zip(events, types, strict=True)]
    reads = [_reads(definition, event, et) for event, et in zip(events, types, strict=True)]
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
                shared = sorted(set(w) & r)
                if shared:
                    raise InterferenceError(
                        f"{writer} and {reader} interfere: {writer} writes "
                        f"{_fact(definition, shared[0])}, which the precondition of {reader} reads"
                    )
    values = np.array(state.values, dtype=np.uint8, copy=True)
    for written in writes:
        for (entity, fluent), value in written.items():
            values[entity, definition.base_fluent_index(fluent)] = int(value)
    return State(values)


__all__ = [
    "Binding",
    "DerivedFacts",
    "Event",
    "State",
    "StaticFacts",
    "able",
    "apply",
    "check_definition_agreement",
    "derive",
    "initial_state",
    "legal",
    "legal_bindings",
    "static_facts",
]
