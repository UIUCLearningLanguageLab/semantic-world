"""The world definition: ``definition.json`` and ``entities.csv``.

The definition is the language-neutral description of a world (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Files"): every symbol, one literal table, every rule in Boolean form and in matrix form, and
every event type. Rules come in two scopes. Entity rules (derived static features, derived
fluents) read literals of one entity, with ``role: null``. Binding rules (constraints and the
requirement that ANDs an event type's constraints) read literals of a binding: a feature or a
threshold of the agent or the patient, a comparison between the two, or a constraint's output.
Both scopes share the literal table; each has its own layers, written with a ``scope`` tag.
Since stage c1 a world with places has two more scopes: ``place`` rules (the determined place
features) read literals of one place, and ``place_binding`` rules (the constraints and
requirements of the place event types) read literals of a binding of an entity and a place: a
feature or a threshold of the agent (shared with the binding scope), a place feature or a place
threshold of the patient, a comparison of an entity scalar with a place scalar, or a constraint's
output. Place symbols, literals, rules, layers, and event types appear only when the world has
places, so with ``places: null`` the definition and its identity are those of the world before
places.

The rule-set identity is the SHA-256 of the canonical JSON of ``symbols``, ``literals``,
``rules``, and ``event_types``. The agreement test checks every entity rule on every entity and
every binding rule on every ordered pair of entities, and every rule with at most 12 inputs on
every input setting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semantic_world.common.boolean import TruthTable
from semantic_world.taxonomy.rules import Rule, Threshold, input_key
from semantic_world.world.constraints import (
    Constraint,
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
)
from semantic_world.world.dynamics import ROLES, ConditionLiteral, Effect, Literal
from semantic_world.world.errors import DefinitionError, WorldError
from semantic_world.world.event_types import ConstraintSpec, EventTypes
from semantic_world.world.fluents import Fluents, ThresholdLiteral
from semantic_world.world.identity import (
    DEFINITION_VERSION,
    read_json,
    rule_set_id,
    to_json,
    write_json,
)
from semantic_world.world.matrices import (
    AgreementReport,
    LiteralSpec,
    RuleMatrices,
    RuleSpec,
    build_matrices,
    check_agreement,
    evaluate_by_tables,
)
from semantic_world.world.numeric import (
    NUMERIC_KIND,
    InitialDistribution,
    NumericFluents,
    NumericThreshold,
    empty_numeric,
)
from semantic_world.world.places import (
    PLACE_EVENT_TYPE_KIND,
    PLACE_KIND_OF_TYPE,
    PLACE_SCALAR_KIND,
    PLACES_FILE,
    Places,
    places_frame,
)
from semantic_world.world.statics import StaticWorld

DEFINITION_FILE = "definition.json"
ENTITIES_FILE = "entities.csv"
BINDING_CHUNK_ROWS = 32768
KIND_OF_TYPE = {"property": "property", "part": "part"}


def _natural(label: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part for part in label.replace(">", ".").split(".")
    )


# ---------------------------------------------------------------------------------------------
# Literal specifications
# ---------------------------------------------------------------------------------------------


def feature_literal(label: str, role: str | None) -> LiteralSpec:
    key = label if role is None else f"{role}.{label}"
    return LiteralSpec(key, {"kind": "feature", "role": role, "feature": label})


def threshold_literal(scalar_label: str, threshold: float, role: str | None) -> LiteralSpec:
    key = f"{scalar_label}>{threshold:.4f}"
    if role is not None:
        key = f"{role}.{key}"
    return LiteralSpec(
        key,
        {
            "kind": "threshold",
            "role": role,
            "scalar": scalar_label,
            "operator": ">",
            "threshold": float(threshold),
        },
    )


def fluent_literal(label: str) -> LiteralSpec:
    return LiteralSpec(label, {"kind": "fluent", "role": None, "fluent": label})


def numeric_threshold_literal(label: str, threshold: float) -> LiteralSpec:
    """A numeric threshold literal of the entity scope (stage b2): ``NUMFL.1 > 0.75``."""
    return LiteralSpec(
        f"{label}>{threshold:.4f}",
        {
            "kind": "numeric_threshold",
            "role": None,
            "fluent": label,
            "operator": ">",
            "threshold": float(threshold),
        },
    )


def comparison_literal(item: ScalarComparison) -> LiteralSpec:
    key = item.key
    return LiteralSpec(
        key,
        {
            "kind": "comparison",
            "role": "binding",
            "agent_scalar": item.agent_label,
            "patient_scalar": item.patient_label,
            "operator": ">",
            "low": float(item.low),
            "high": None if item.high is None else float(item.high),
        },
    )


def constraint_literal(label: str) -> LiteralSpec:
    return LiteralSpec(label, {"kind": "constraint", "role": "binding", "constraint": label})


def _rule_input_literals(rule: Rule, role: str | None) -> list[LiteralSpec]:
    specs = []
    for item in rule.inputs:
        if isinstance(item, Threshold):
            specs.append(threshold_literal(item.label, item.threshold, role))
        else:
            specs.append(feature_literal(item.label, role))
    return specs


def _constraint_input_literals(constraint: Constraint) -> list[LiteralSpec]:
    specs = []
    for item in constraint.literals:
        if isinstance(item, RoleFeature):
            specs.append(feature_literal(item.feature.label, item.role))
        elif isinstance(item, RoleThreshold):
            specs.append(
                threshold_literal(item.threshold.label, item.threshold.threshold, item.role)
            )
        else:
            specs.append(comparison_literal(item))
    return specs


def constraint_literals(spec: ConstraintSpec) -> list[LiteralSpec]:
    """The literal specifications of a constraint's inputs, in input order."""
    if isinstance(spec.source, Rule):
        return _rule_input_literals(spec.source, "agent")
    return _constraint_input_literals(spec.source)


PLACE_CONSTRAINT_PREFIX = "CONSTRAINT.PLACEEVENTTYPE."


def _is_place_literal(reads: Mapping[str, Any], place_features: Mapping[str, int]) -> bool:
    """Whether a literal belongs to the place or the place-binding scope alone: it reads a
    place feature, a place scalar, or a place event type's constraint. A literal of the agent
    (``agent.PROPERTY.3``) is shared with the binding scope and is not a place literal."""
    kind = reads["kind"]
    if kind == "feature":
        return reads["feature"] in place_features
    if kind == "threshold":
        return reads["scalar"].startswith("PLACE")
    if kind == "comparison":
        return reads["patient_scalar"].startswith("PLACE")
    if kind == "constraint":
        return reads["constraint"].startswith(PLACE_CONSTRAINT_PREFIX)
    return False


_GROUP_ORDER = {
    (None, "feature"): 0,
    (None, "threshold"): 1,
    (None, "fluent"): 2,
    (None, "numeric_threshold"): 3,
    ("agent", "feature"): 4,
    ("agent", "threshold"): 5,
    ("patient", "feature"): 6,
    ("patient", "threshold"): 7,
    ("binding", "comparison"): 8,
    ("binding", "constraint"): 9,
}


def _literal_sort_key(
    spec: LiteralSpec, position_of: dict[str, int], place_position_of: dict[str, int] | None = None
) -> tuple:
    """The order of the literal table: the entity groups of ``_GROUP_ORDER``, then the same
    groups for the place literals (stage c1), so that a world without places orders its
    literals as before."""
    reads = spec.reads
    place_position_of = place_position_of or {}
    group = _GROUP_ORDER[(reads["role"], reads["kind"])]
    if _is_place_literal(reads, place_position_of):
        group += len(_GROUP_ORDER)
        if reads["kind"] == "feature":
            return (group, place_position_of[reads["feature"]], 0.0)
    if reads["kind"] == "feature":
        return (group, position_of[reads["feature"]], 0.0)
    if reads["kind"] == "threshold":
        return (group, _natural(reads["scalar"]), reads["threshold"])
    if reads["kind"] == "fluent":
        return (group, _natural(reads["fluent"]), 0.0)
    if reads["kind"] == "numeric_threshold":
        return (group, _natural(reads["fluent"]), reads["threshold"])
    if reads["kind"] == "comparison":
        return (
            group,
            _natural(reads["agent_scalar"]) + _natural(reads["patient_scalar"]),
            reads["low"],
            reads["high"] or 0.0,
        )
    return (group, _natural(reads["constraint"]), 0.0)


# ---------------------------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    statics: StaticWorld
    fluents: Fluents
    event_types: EventTypes
    symbols: tuple[dict[str, Any], ...]
    literals: tuple[LiteralSpec, ...]
    entity_rules: tuple[RuleSpec, ...]
    binding_rules: tuple[RuleSpec, ...]
    families: dict[str, str]
    """Each rule output's family (``shj``, ``cross``, ``requirement``, and so on)."""
    entity_matrices: RuleMatrices
    binding_matrices: RuleMatrices
    rule_set_id: str
    numeric: NumericFluents = None  # type: ignore[assignment]
    """The numeric fluents (stage b2); none when the world has none."""
    places: Places | None = None
    """The places (stage c1); None without them."""
    place_rules: tuple[RuleSpec, ...] = ()
    """The rules of the determined place features (scope ``place``)."""
    place_binding_rules: tuple[RuleSpec, ...] = ()
    """The constraints and requirements of the place event types (scope ``place_binding``)."""
    place_matrices: RuleMatrices | None = None
    place_binding_matrices: RuleMatrices | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {lit.key: i for i, lit in enumerate(self.literals)})
        if self.numeric is None:
            object.__setattr__(self, "numeric", empty_numeric(len(self.statics.instances)))
        if self.places is not None and (
            self.place_matrices is None or self.place_binding_matrices is None
        ):
            raise WorldError("a definition with places needs the place matrices")

    @property
    def literal_count(self) -> int:
        return len(self.literals)

    def literal_index(self, key: str) -> int:
        return self._index[key]  # type: ignore[attr-defined]

    def literals_record(self) -> list[dict[str, Any]]:
        return [lit.record(i) for i, lit in enumerate(self.literals)]

    def rules_record(self) -> list[dict[str, Any]]:
        index = self._index  # type: ignore[attr-defined]
        records = []
        for scope, rules in (
            ("entity", self.entity_rules),
            ("binding", self.binding_rules),
            ("place", self.place_rules),
            ("place_binding", self.place_binding_rules),
        ):
            for rule in rules:
                entry = rule.record(index)
                records.append(
                    {
                        "output": entry["output"],
                        "scope": scope,
                        "family": self.families[rule.output],
                        "inputs": entry["inputs"],
                        "truth_table": entry["truth_table"],
                        "expression": entry["expression"],
                    }
                )
        return records

    def layers_record(self) -> list[dict[str, Any]]:
        records = []
        scopes: list[tuple[str, RuleMatrices | None]] = [
            ("entity", self.entity_matrices),
            ("binding", self.binding_matrices),
            ("place", self.place_matrices),
            ("place_binding", self.place_binding_matrices),
        ]
        for scope, matrices in scopes:
            if matrices is None:
                continue
            for layer in matrices.layers_record():
                records.append({"scope": scope, **layer})
        return records

    def event_types_record(self) -> list[dict[str, Any]]:
        records = [et.record() for et in self.event_types.event_types]
        if self.places is not None:
            records += [et.record() for et in self.places.event_types]
        return records

    def record(self) -> dict[str, Any]:
        """The contents of ``definition.json``."""
        return {
            "version": DEFINITION_VERSION,
            "rule_set_id": self.rule_set_id,
            "symbols": list(self.symbols),
            "literals": self.literals_record(),
            "rules": self.rules_record(),
            "layers": self.layers_record(),
            "event_types": self.event_types_record(),
        }

    def runtime(self) -> RuntimeDefinition:
        """The definition as the runtime reads it, built from this definition's own record and
        entities table through their JSON form, exactly as a reader of the files sees them."""
        record = json.loads(to_json(self.record(), indent=1, sort_keys=False))
        entities = json.loads(to_json(entities_frame(self).to_dicts(), indent=1, sort_keys=False))
        places = None
        if self.places is not None:
            places = json.loads(
                to_json(places_frame(self.places).to_dicts(), indent=1, sort_keys=False)
            )
        return runtime_definition(record, entities, places)

    # Literal values ---------------------------------------------------------------------------

    @property
    def taxonomy(self):
        return self.statics.taxonomy

    def _position(self, label: str) -> int:
        return self.statics.features[label].position

    def entity_literal_values(
        self,
        values: np.ndarray,
        scalars: np.ndarray,
        base_fluents: np.ndarray,
        numeric: np.ndarray | None = None,
    ) -> np.ndarray:
        """The base literal matrix of entities: free features from ``values`` (the static
        feature matrix, by position), thresholds from ``scalars``, base fluents from
        ``base_fluents`` (``(n, base fluents)``), numeric thresholds from ``numeric`` (``(n,
        numeric fluents)``, the initial values when None), and 0 for every literal that a rule
        computes or that a binding reads."""
        n = values.shape[0]
        matrix = self.entity_matrices.literal_matrix(n)
        base_index = {label: i for i, label in enumerate(self.fluents.base_labels)}
        if numeric is None:
            numeric = self.numeric.initial_values
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            if reads["role"] is not None or _is_place_literal(reads, self._place_positions()):
                continue
            if reads["kind"] == "feature":
                feature = self.statics.features[reads["feature"]]
                if feature.free:
                    matrix[:, i] = values[:, feature.position]
            elif reads["kind"] == "threshold":
                scalar = int(reads["scalar"].split(".")[1])
                matrix[:, i] = (scalars[:, scalar - 1] > reads["threshold"]).astype(np.uint8)
            elif reads["kind"] == "fluent" and reads["fluent"] in base_index:
                matrix[:, i] = base_fluents[:, base_index[reads["fluent"]]]
            elif reads["kind"] == "numeric_threshold":
                column = numeric[:, self.numeric.index(reads["fluent"])]
                matrix[:, i] = (column > reads["threshold"]).astype(np.uint8)
        return matrix

    def binding_literal_values(
        self, values: np.ndarray, scalars: np.ndarray, agents: np.ndarray, patients: np.ndarray
    ) -> np.ndarray:
        """The base literal matrix of aligned bindings ``(agents[k], patients[k])``, from the
        entities' full static values (free and derived) and scalars. Constraint outputs are 0."""
        agents = np.asarray(agents, dtype=np.intp)
        patients = np.asarray(patients, dtype=np.intp)
        matrix = self.binding_matrices.literal_matrix(agents.shape[0])
        rows = {"agent": agents, "patient": patients}
        place_positions = self._place_positions()
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            role = reads["role"]
            if role is None or reads["kind"] == "constraint":
                continue
            if _is_place_literal(reads, place_positions):
                continue
            if reads["kind"] == "feature":
                matrix[:, i] = values[rows[role], self._position(reads["feature"])]
            elif reads["kind"] == "threshold":
                scalar = int(reads["scalar"].split(".")[1])
                matrix[:, i] = (scalars[rows[role], scalar - 1] > reads["threshold"]).astype(
                    np.uint8
                )
            else:
                a = int(reads["agent_scalar"].split(".")[1])
                p = int(reads["patient_scalar"].split(".")[1])
                difference = scalars[agents, a - 1] - scalars[patients, p - 1]
                holds = difference > reads["low"]
                if reads["high"] is not None:
                    holds &= difference < reads["high"]
                matrix[:, i] = holds.astype(np.uint8)
        return matrix

    # Places (stage c1) ----------------------------------------------------------------------

    def _place_positions(self) -> dict[str, int]:
        if self.places is None:
            return {}
        return {f.label: f.position for f in self.places.taxonomy.features.features}

    def place_literal_values(self, values: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """The base literal matrix of places: free place features from ``values`` (the place
        feature matrix, by position) and place thresholds from ``scalars``; 0 elsewhere."""
        assert self.places is not None and self.place_matrices is not None
        matrix = self.place_matrices.literal_matrix(values.shape[0])
        features = self.places.taxonomy.features
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            if reads["role"] is not None or reads["kind"] not in ("feature", "threshold"):
                continue
            if reads["kind"] == "feature":
                if reads["feature"] in features:
                    feature = features[reads["feature"]]
                    if feature.free:
                        matrix[:, i] = values[:, feature.position]
            elif reads["scalar"].startswith("PLACE"):
                scalar = int(reads["scalar"].split(".")[1])
                matrix[:, i] = (scalars[:, scalar - 1] > reads["threshold"]).astype(np.uint8)
        return matrix

    def place_binding_literal_values(
        self,
        values: np.ndarray,
        scalars: np.ndarray,
        place_values: np.ndarray,
        place_scalars: np.ndarray,
        agents: np.ndarray,
        places: np.ndarray,
    ) -> np.ndarray:
        """The base literal matrix of aligned bindings ``(agents[k], places[k])`` of an entity
        and a place, from the entities' full static values and scalars and the places' full
        static values and scalars. Constraint outputs are 0."""
        assert self.places is not None and self.place_binding_matrices is not None
        agents = np.asarray(agents, dtype=np.intp)
        places = np.asarray(places, dtype=np.intp)
        matrix = self.place_binding_matrices.literal_matrix(agents.shape[0])
        place_positions = self._place_positions()
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            role = reads["role"]
            if role is None or reads["kind"] == "constraint":
                continue
            place = _is_place_literal(reads, place_positions)
            if reads["kind"] == "feature":
                if role == "agent":
                    matrix[:, i] = values[agents, self._position(reads["feature"])]
                elif place:
                    matrix[:, i] = place_values[places, place_positions[reads["feature"]]]
            elif reads["kind"] == "threshold":
                scalar = int(reads["scalar"].split(".")[1])
                if role == "agent":
                    column = scalars[agents, scalar - 1]
                elif place:
                    column = place_scalars[places, scalar - 1]
                else:
                    continue
                matrix[:, i] = (column > reads["threshold"]).astype(np.uint8)
            elif place:
                a = int(reads["agent_scalar"].split(".")[1])
                p = int(reads["patient_scalar"].split(".")[1])
                difference = scalars[agents, a - 1] - place_scalars[places, p - 1]
                holds = difference > reads["low"]
                if reads["high"] is not None:
                    holds &= difference < reads["high"]
                matrix[:, i] = holds.astype(np.uint8)
        return matrix


# ---------------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------------


def _place_symbols(places: Places) -> list[dict[str, Any]]:
    """The place symbols (stage c1): the place features, the place scalars, and the place event
    types, after every entity symbol."""
    symbols: list[dict[str, Any]] = []
    for feature in places.taxonomy.features.features:
        symbols.append(
            {
                "label": feature.label,
                "kind": PLACE_KIND_OF_TYPE[feature.type],
                "derived": not feature.free,
                "fluent": False,
                "arity": 1,
            }
        )
    for label in places.taxonomy.features.scalar_labels:
        symbols.append(
            {
                "label": label,
                "kind": PLACE_SCALAR_KIND,
                "derived": False,
                "fluent": False,
                "arity": 1,
            }
        )
    for et in places.event_types:
        symbols.append(
            {
                "label": et.label,
                "kind": PLACE_EVENT_TYPE_KIND,
                "derived": False,
                "fluent": False,
                "arity": 2,
            }
        )
    return symbols


def _symbols(
    statics: StaticWorld, fluents: Fluents, event_types: EventTypes, numeric: NumericFluents
) -> list[dict[str, Any]]:
    taxonomy = statics.taxonomy
    symbols: list[dict[str, Any]] = []
    for feature in taxonomy.features.features:
        symbols.append(
            {
                "label": feature.label,
                "kind": KIND_OF_TYPE[feature.type],
                "derived": not feature.free,
                "fluent": False,
                "arity": 1,
            }
        )
    for label in taxonomy.features.scalar_labels:
        symbols.append(
            {
                "label": label,
                "kind": "scalar",
                "derived": False,
                "fluent": False,
                "arity": 1,
            }
        )
    for fluent in fluents.fluents:
        symbols.append(
            {
                "label": fluent.label,
                "kind": "fluent",
                "derived": fluent.derived,
                "fluent": True,
                "arity": 1,
                "initial_rate": None if fluent.derived else float(fluent.initial_rate),
            }
        )
    for numeric_fluent in numeric.fluents:
        symbols.append(numeric_fluent.symbol_record())
    for et in event_types.event_types:
        symbols.append(
            {
                "label": et.label,
                "kind": et.kind,
                "derived": False,
                "fluent": False,
                "arity": et.arity,
            }
        )
    return symbols


def build_definition(
    statics: StaticWorld,
    fluents: Fluents,
    event_types: EventTypes,
    numeric: NumericFluents | None = None,
    places: Places | None = None,
) -> Definition:
    """Assemble the definition: the literal table, the rule scopes (entity and binding, and
    with places the place and place-binding scopes), the matrices, and the identity."""
    taxonomy = statics.taxonomy
    if numeric is None:
        numeric = empty_numeric(len(taxonomy.instances))
    position_of = {f.label: f.position for f in statics.features.features}
    place_position_of = (
        {f.label: f.position for f in places.taxonomy.features.features}
        if places is not None
        else {}
    )
    literal_specs: dict[str, LiteralSpec] = {}

    def add(spec: LiteralSpec) -> None:
        literal_specs.setdefault(spec.key, spec)

    entity_rules: list[RuleSpec] = []
    families: dict[str, str] = {}
    for rule in taxonomy.rules.rules:
        for spec in _rule_input_literals(rule, None):
            add(spec)
        output = rule.output.label
        inputs = tuple(input_key(item) for item in rule.inputs)
        entity_rules.append(RuleSpec(output, inputs, rule.table, str(rule.expression)))
        families[output] = rule.family
    for rule in fluents.rules:
        for key in rule.inputs:
            if key in fluents:
                add(fluent_literal(key))
            elif key.startswith("NUMFL."):
                numeric_literal: NumericThreshold = next(
                    t for t in rule.numeric_thresholds if t.key == key
                )
                add(numeric_threshold_literal(numeric_literal.fluent, numeric_literal.threshold))
            elif ">" in key:
                literal: ThresholdLiteral = next(t for t in rule.thresholds if t.key == key)
                add(threshold_literal(literal.label, literal.threshold, None))
            else:
                add(feature_literal(key, None))
        entity_rules.append(RuleSpec(rule.output, rule.inputs, rule.table, str(rule.expression)))
        families[rule.output] = rule.family
    binding_rules: list[RuleSpec] = []
    for constraint in event_types.constraints:
        for spec in constraint_literals(constraint):
            add(spec)
        binding_rules.append(
            RuleSpec(constraint.label, constraint.inputs, constraint.table, constraint.expression)
        )
        families[constraint.label] = constraint.family
    for et in event_types.event_types:
        for label in et.constraints:
            add(constraint_literal(label))
        k = len(et.constraints)
        table = TruthTable.from_function(k, lambda x: all(x))
        binding_rules.append(
            RuleSpec(et.requirement_label, et.constraints, table, et.requirement_expression)
        )
        families[et.requirement_label] = "requirement"
    place_rules: list[RuleSpec] = []
    place_binding_rules: list[RuleSpec] = []
    if places is not None:
        for rule in places.taxonomy.rules.rules:
            for spec in _rule_input_literals(rule, None):
                add(spec)
            inputs = tuple(input_key(item) for item in rule.inputs)
            place_rules.append(
                RuleSpec(rule.output.label, inputs, rule.table, str(rule.expression))
            )
            families[rule.output.label] = rule.family
        for constraint in places.constraints:
            for spec in _constraint_input_literals(constraint):
                add(spec)
            place_binding_rules.append(
                RuleSpec(
                    constraint.label,
                    tuple(item.key for item in constraint.literals),
                    constraint.table,
                    str(constraint.expression),
                )
            )
            families[constraint.label] = constraint.family
        for pet in places.event_types:
            labels = tuple(c.label for c in pet.constraints)
            for label in labels:
                add(constraint_literal(label))
            table = TruthTable.from_function(len(labels), lambda x: all(x))
            place_binding_rules.append(
                RuleSpec(pet.requirement_label, labels, table, " AND ".join(labels) or "TRUE")
            )
            families[pet.requirement_label] = "requirement"
    literals = tuple(
        sorted(
            literal_specs.values(),
            key=lambda s: _literal_sort_key(s, position_of, place_position_of),
        )
    )
    entity_matrices = build_matrices(literals, entity_rules)
    binding_matrices = build_matrices(literals, binding_rules)
    place_matrices = build_matrices(literals, place_rules) if places is not None else None
    place_binding_matrices = (
        build_matrices(literals, place_binding_rules) if places is not None else None
    )
    symbols = tuple(
        _symbols(statics, fluents, event_types, numeric)
        + (_place_symbols(places) if places is not None else [])
    )

    def assemble(identity: str) -> Definition:
        return Definition(
            statics,
            fluents,
            event_types,
            symbols,
            literals,
            tuple(entity_rules),
            tuple(binding_rules),
            families,
            entity_matrices,
            binding_matrices,
            identity,
            numeric,
            places,
            tuple(place_rules),
            tuple(place_binding_rules),
            place_matrices,
            place_binding_matrices,
        )

    draft = assemble("")
    identity = rule_set_id(
        symbols, draft.literals_record(), draft.rules_record(), draft.event_types_record()
    )
    return assemble(identity)


# ---------------------------------------------------------------------------------------------
# The agreement test
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DefinitionReport:
    entity: AgreementReport
    binding: AgreementReport
    place: AgreementReport | None = None
    """The place rules (stage c1); None without places."""
    place_binding: AgreementReport | None = None
    """The place event types' constraints and requirements over every entity and place."""


def check_definition(definition: Definition) -> DefinitionReport:
    """The agreement test over the whole definition. Entity rules are checked on every entity
    against the taxonomy's values (static features) and the truth tables (derived fluents).
    Binding rules are checked on every ordered pair of entities against the static world's own
    evaluation of each constraint, and the requirement against the conjunction."""
    statics = definition.statics
    taxonomy = statics.taxonomy
    instances = taxonomy.instances
    values, scalars = statics.values, instances.scalars
    base = definition.entity_literal_values(values, scalars, definition.fluents.initial_values)
    expected = evaluate_by_tables(definition.entity_matrices, definition.entity_rules, base)
    for rule in taxonomy.rules.rules:
        column = values[:, rule.output.position]
        label = rule.output.label
        if not np.array_equal(expected[label], column):
            raise WorldError(f"the truth tables of {label} disagree with the taxonomy's instances")
        expected[label] = column
    entity_report = check_agreement(
        definition.entity_matrices, definition.entity_rules, base, expected
    )

    n = len(instances)
    chunk = max(1, BINDING_CHUNK_ROWS // max(n, 1))
    rules = definition.binding_rules
    total = AgreementReport(len(rules), 0, 0, 0)
    first = True
    for start in range(0, n, chunk):
        agent_rows = np.arange(start, min(start + chunk, n))
        agents = np.repeat(agent_rows, n)
        patients = np.tile(np.arange(n), len(agent_rows))
        literal_values = definition.binding_literal_values(values, scalars, agents, patients)
        expected_binding: dict[str, np.ndarray] = {}
        for spec in definition.event_types.constraints:
            if isinstance(spec.source, Rule):
                expected_binding[spec.label] = values[agents, spec.source.output.position]
                # the one-place capacity column of the static world, by the extended position
            else:
                expected_binding[spec.label] = (
                    spec.source.matrix(values, scalars, agent_rows).reshape(-1).astype(np.uint8)
                )
        for et in definition.event_types.event_types:
            out = np.ones(len(agents), dtype=np.uint8)
            for label in et.constraints:
                out &= expected_binding[label]
            expected_binding[et.requirement_label] = out
        report = check_agreement(
            definition.binding_matrices,
            rules,
            literal_values,
            expected_binding,
            max_exhaustive_inputs=12 if first else -1,
        )
        total = AgreementReport(
            len(rules),
            total.entities + report.entities,
            max(total.exhaustive_rules, report.exhaustive_rules),
            max(total.settings, report.settings),
        )
        first = False
    if definition.places is None:
        return DefinitionReport(entity_report, total)
    return DefinitionReport(entity_report, total, *_check_places(definition))


def _check_places(definition: Definition) -> tuple[AgreementReport, AgreementReport]:
    """The agreement test of the place scopes: every place rule on every place against the
    place taxonomy's instances, and every place constraint and requirement on every binding of
    an entity and a place against the constraints' own evaluation."""
    places = definition.places
    assert places is not None
    assert definition.place_matrices is not None
    assert definition.place_binding_matrices is not None
    place_instances = places.taxonomy.instances
    place_values, place_scalars = place_instances.values, place_instances.scalars
    base = definition.place_literal_values(place_values, place_scalars)
    expected: dict[str, np.ndarray] = {}
    for rule in places.taxonomy.rules.rules:
        expected[rule.output.label] = place_values[:, rule.output.position]
    place_report = check_agreement(
        definition.place_matrices, definition.place_rules, base, expected
    )
    instances = definition.statics.taxonomy.instances
    values, scalars = definition.statics.values, instances.scalars
    n, m = len(instances), len(place_instances)
    chunk = max(1, BINDING_CHUNK_ROWS // max(m, 1))
    rules = definition.place_binding_rules
    total = AgreementReport(len(rules), 0, 0, 0)
    first = True
    for start in range(0, n, chunk):
        agent_rows = np.arange(start, min(start + chunk, n))
        agents = np.repeat(agent_rows, m)
        patients = np.tile(np.arange(m), len(agent_rows))
        literal_values = definition.place_binding_literal_values(
            values, scalars, place_values, place_scalars, agents, patients
        )
        expected_binding: dict[str, np.ndarray] = {}
        for constraint in places.constraints:
            expected_binding[constraint.label] = (
                constraint.matrix(values, scalars, agent_rows, place_values, place_scalars)
                .reshape(-1)
                .astype(np.uint8)
            )
        for pet in places.event_types:
            out = np.ones(len(agents), dtype=np.uint8)
            for constraint in pet.constraints:
                out &= expected_binding[constraint.label]
            expected_binding[pet.requirement_label] = out
        report = check_agreement(
            definition.place_binding_matrices,
            rules,
            literal_values,
            expected_binding,
            max_exhaustive_inputs=12 if first else -1,
        )
        total = AgreementReport(
            len(rules),
            total.entities + report.entities,
            max(total.exhaustive_rules, report.exhaustive_rules),
            max(total.settings, report.settings),
        )
        first = False
    return place_report, total


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def entities_frame(definition: Definition) -> pl.DataFrame:
    """``entities.csv``: label, leaf, free PROPERTY and PART features, scalars, and the initial
    base fluents."""
    taxonomy = definition.taxonomy
    instances = taxonomy.instances
    data: dict[str, Any] = {
        "label": list(instances.labels),
        "leaf": list(instances.leaf_labels),
    }
    for feature in taxonomy.features.free:
        data[feature.label] = instances.values[:, feature.position].astype(np.int64).tolist()
    for j, label in enumerate(taxonomy.features.scalar_labels):
        data[label] = instances.scalars[:, j].astype(float).tolist()
    for i, fluent in enumerate(definition.fluents.base):
        data[fluent.label] = definition.fluents.initial_values[:, i].astype(np.int64).tolist()
    for i, numeric_fluent in enumerate(definition.numeric.fluents):
        data[numeric_fluent.label] = definition.numeric.initial_values[:, i].astype(float).tolist()
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})


def write_definition(definition: Definition, folder: str | Path) -> None:
    folder = Path(folder)
    write_json(folder / DEFINITION_FILE, definition.record())
    entities_frame(definition).write_csv(folder / ENTITIES_FILE)
    if definition.places is not None:
        places_frame(definition.places).write_csv(folder / PLACES_FILE)


# ---------------------------------------------------------------------------------------------
# Loading: the definition as the runtime reads it
# ---------------------------------------------------------------------------------------------

STATIC_KINDS = ("property", "part")
PLACE_STATIC_KINDS = ("place_property", "place_part")
SYMBOL_KINDS = STATIC_KINDS + (
    "scalar",
    "fluent",
    NUMERIC_KIND,
    "event_type",
    "event_type_category",
    *PLACE_STATIC_KINDS,
    PLACE_SCALAR_KIND,
    PLACE_EVENT_TYPE_KIND,
)
EVENT_TYPE_KINDS = ("event_type", "event_type_category", PLACE_EVENT_TYPE_KIND)
LITERAL_KINDS = ("feature", "threshold", "fluent", "numeric_threshold", "comparison", "constraint")
SCOPES = ("entity", "binding", "place", "place_binding")
ENTITY_COLUMNS = ("label", "leaf")


@dataclass(frozen=True)
class EventTypeRecord:
    """One event type as the runtime reads it: its roles, the output label of its requirement,
    its precondition literals, and its effects, each with its condition when it has one. A
    category of event types has no events."""

    label: str
    kind: str
    arity: int
    requirement: str
    precondition: tuple[Literal, ...]
    effects: tuple[Effect, ...]

    @property
    def roles(self) -> tuple[str, ...]:
        return ROLES[: self.arity]

    @property
    def is_category(self) -> bool:
        return self.kind == "event_type_category"

    @property
    def place(self) -> bool:
        """Whether the patient is a place (a place event type, stage c1)."""
        return self.kind == PLACE_EVENT_TYPE_KIND

    @property
    def performable(self) -> bool:
        """Whether the event type has events in an episode: an event type proper, never a
        category, and in stage c1 never a place event type (they get their dynamics in c3)."""
        return self.kind == "event_type"


@dataclass(frozen=True)
class RuntimeDefinition:
    """A world definition loaded from its JSON record and its entities table, never from a
    taxonomy run. The runtime (``runtime.py``) and the fixture checker read nothing else.

    The rule-set identity is recomputed from the four hashed tables when the definition is
    loaded, and a record whose identity differs is refused.
    """

    record: dict[str, Any]
    """The definition record as read."""
    rule_set_id: str
    literals: tuple[LiteralSpec, ...]
    entity_rules: tuple[RuleSpec, ...]
    binding_rules: tuple[RuleSpec, ...]
    entity_matrices: RuleMatrices
    binding_matrices: RuleMatrices
    event_types: tuple[EventTypeRecord, ...]
    free_features: tuple[str, ...]
    """The base static features (free PROPERTY and PART), in symbol order."""
    derived_features: tuple[str, ...]
    """The derived static features, in symbol order."""
    scalars: tuple[str, ...]
    base_fluents: tuple[str, ...]
    derived_fluents: tuple[str, ...]
    entity_labels: tuple[str, ...]
    leaves: tuple[str, ...]
    initial_rates: dict[str, float]
    """Each base fluent's initial rate, from its symbol (``scene.initial: redraw`` draws at
    these rates)."""
    free_values: np.ndarray
    """Shape ``(entities, free features)``, uint8."""
    scalar_values: np.ndarray
    """Shape ``(entities, scalars)``, float64."""
    initial_values: np.ndarray | None
    """Shape ``(entities, base fluents)``, uint8: the initial base fluents of ``entities.csv``;
    None when the entities table carries no fluent columns (a fixture gives them apart)."""
    numeric_fluents: tuple[str, ...] = ()
    """The numeric fluents (stage b2), in symbol order."""
    numeric_ranges: dict[str, tuple[float, float]] = None  # type: ignore[assignment]
    numeric_initial: dict[str, InitialDistribution] = None  # type: ignore[assignment]
    """Each numeric fluent's initial distribution (``scene.initial: redraw`` draws from it)."""
    numeric_drift: dict[str, float] = None  # type: ignore[assignment]
    initial_numeric: np.ndarray | None = None
    """Shape ``(entities, numeric fluents)``, float64: the initial values of ``entities.csv``;
    None when the table carries no numeric columns (a fixture gives them apart)."""
    place_labels: tuple[str, ...] = ()
    """The places (stage c1), in the order of ``places.csv``; empty without places."""
    place_leaves: tuple[str, ...] = ()
    place_free_features: tuple[str, ...] = ()
    place_derived_features: tuple[str, ...] = ()
    place_scalars: tuple[str, ...] = ()
    place_free_values: np.ndarray | None = None
    """Shape ``(places, free place features)``, uint8."""
    place_scalar_values: np.ndarray | None = None
    """Shape ``(places, place scalars)``, float64."""
    place_rules: tuple[RuleSpec, ...] = ()
    place_binding_rules: tuple[RuleSpec, ...] = ()
    place_matrices: RuleMatrices | None = None
    place_binding_matrices: RuleMatrices | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_entity_index", {v: i for i, v in enumerate(self.entity_labels)})
        object.__setattr__(self, "_event_types", {et.label: et for et in self.event_types})
        object.__setattr__(self, "_base_index", {v: i for i, v in enumerate(self.base_fluents)})
        object.__setattr__(
            self, "_derived_index", {v: i for i, v in enumerate(self.derived_fluents)}
        )
        object.__setattr__(
            self, "_numeric_index", {v: i for i, v in enumerate(self.numeric_fluents)}
        )
        for name in ("numeric_ranges", "numeric_initial", "numeric_drift"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})
        object.__setattr__(self, "_place_index", {v: i for i, v in enumerate(self.place_labels)})
        object.__setattr__(self, "_cones", _fluent_cones(self))
        object.__setattr__(self, "cache", {})
        for arr in (
            self.free_values,
            self.scalar_values,
            self.initial_values,
            self.initial_numeric,
            self.place_free_values,
            self.place_scalar_values,
        ):
            if arr is not None:
                arr.setflags(write=False)

    @property
    def entity_count(self) -> int:
        return len(self.entity_labels)

    @property
    def has_places(self) -> bool:
        return self.place_matrices is not None

    @property
    def place_count(self) -> int:
        return len(self.place_labels)

    def place_index(self, label: str) -> int:
        try:
            return self._place_index[label]  # type: ignore[attr-defined]
        except KeyError:
            raise WorldError(f"{label} is not a place of the definition") from None

    def is_place_scalar(self, label: str) -> bool:
        return label in self.place_scalars

    def entity_index(self, label: str) -> int:
        try:
            return self._entity_index[label]  # type: ignore[attr-defined]
        except KeyError:
            raise WorldError(f"{label} is not an entity of the definition") from None

    def event_type(self, label: str) -> EventTypeRecord:
        try:
            return self._event_types[label]  # type: ignore[attr-defined]
        except KeyError:
            raise WorldError(f"{label} is not an event type of the definition") from None

    def is_base_fluent(self, label: str) -> bool:
        return label in self._base_index  # type: ignore[attr-defined]

    def base_fluent_index(self, label: str) -> int:
        return self._base_index[label]  # type: ignore[attr-defined]

    def derived_fluent_index(self, label: str) -> int:
        return self._derived_index[label]  # type: ignore[attr-defined]

    def is_numeric_fluent(self, label: str) -> bool:
        return label in self._numeric_index  # type: ignore[attr-defined]

    def numeric_index(self, label: str) -> int:
        return self._numeric_index[label]  # type: ignore[attr-defined]

    @property
    def drift_vector(self) -> np.ndarray:
        """Each numeric fluent's drift per step, in symbol order."""
        return np.array([self.numeric_drift[f] for f in self.numeric_fluents], dtype=np.float64)

    @property
    def range_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The low and the high end of every numeric fluent's range, in symbol order."""
        low = np.array([self.numeric_ranges[f][0] for f in self.numeric_fluents], dtype=np.float64)
        high = np.array([self.numeric_ranges[f][1] for f in self.numeric_fluents], dtype=np.float64)
        return low, high

    def cone(self, fluent: str) -> tuple[str, ...]:
        """The base fluents a fluent depends on: itself for a base fluent (Boolean or numeric),
        and for a derived fluent every base fluent its rule reads, directly or through other
        derived fluents, numeric fluents read through threshold literals included."""
        return self._cones[fluent]  # type: ignore[attr-defined]


def _fluent_cones(definition: RuntimeDefinition) -> dict[str, tuple[str, ...]]:
    literal_of = {spec.key: spec.reads for spec in definition.literals}
    rules = {rule.output: rule for rule in definition.entity_rules}
    cones: dict[str, tuple[str, ...]] = {f: (f,) for f in definition.base_fluents}
    cones.update({f: (f,) for f in definition.numeric_fluents})
    base_order = {f: i for i, f in enumerate(definition.base_fluents + definition.numeric_fluents)}

    def cone_of(label: str) -> tuple[str, ...]:
        if label in cones:
            return cones[label]
        found: set[str] = set()
        for key in rules[label].inputs:
            reads = literal_of[key]
            if reads["kind"] in ("fluent", "numeric_threshold"):
                found.update(cone_of(reads["fluent"]))
        cones[label] = tuple(sorted(found, key=base_order.__getitem__))
        return cones[label]

    for label in definition.derived_fluents:
        cone_of(label)
    return cones


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionError(message)


def _rule_spec(entry: Mapping[str, Any], literals: Sequence[LiteralSpec]) -> RuleSpec:
    inputs = []
    for i in entry["inputs"]:
        _expect(
            isinstance(i, int) and 0 <= i < len(literals),
            f"the rule for {entry['output']} reads literal {i}, which is not in the literal table",
        )
        inputs.append(literals[i].key)
    try:
        table = TruthTable.from_bit_string(str(entry["truth_table"]))
        return RuleSpec(str(entry["output"]), tuple(inputs), table, entry.get("expression"))
    except (ValueError, WorldError) as error:
        raise DefinitionError(f"the rule for {entry['output']} is malformed: {error}") from None


def runtime_definition(
    record: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]] | pl.DataFrame,
    places: Sequence[Mapping[str, Any]] | pl.DataFrame | None = None,
) -> RuntimeDefinition:
    """Build the runtime's definition from a definition record (the contents of
    ``definition.json``), an entities table (the rows of ``entities.csv``, or the same rows
    without the fluent columns), and, for a world with places, the places table (the rows of
    ``places.csv``). Checks the identity and every reference between the tables."""
    rows = entities.to_dicts() if isinstance(entities, pl.DataFrame) else list(entities)
    place_rows = places.to_dicts() if isinstance(places, pl.DataFrame) else list(places or [])
    try:
        return _runtime_definition(record, rows, place_rows)
    except (KeyError, TypeError) as error:
        raise DefinitionError(f"the definition record is malformed: {error!r}") from None


def _runtime_definition(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], place_rows: Sequence
) -> RuntimeDefinition:
    _expect(
        record["version"] == DEFINITION_VERSION,
        f"the definition has version {record['version']!r}; this runtime reads version "
        f"{DEFINITION_VERSION}",
    )
    identity = rule_set_id(
        record["symbols"], record["literals"], record["rules"], record["event_types"]
    )
    _expect(
        record["rule_set_id"] == identity,
        f"the definition says its rule set is {record['rule_set_id']}, but its tables hash to "
        f"{identity}",
    )

    # Symbols.
    symbols = {}
    free_features: list[str] = []
    derived_features: list[str] = []
    scalars: list[str] = []
    base_fluents: list[str] = []
    derived_fluents: list[str] = []
    initial_rates: dict[str, float] = {}
    numeric_fluents: list[str] = []
    numeric_ranges: dict[str, tuple[float, float]] = {}
    numeric_initial: dict[str, InitialDistribution] = {}
    numeric_drift: dict[str, float] = {}
    place_free: list[str] = []
    place_derived: list[str] = []
    place_scalars: list[str] = []
    place_event_types: list[str] = []
    for symbol in record["symbols"]:
        label, kind = str(symbol["label"]), str(symbol["kind"])
        _expect(kind in SYMBOL_KINDS, f"the symbol {label} has the unknown kind {kind!r}")
        _expect(label not in symbols, f"the symbol {label} is listed twice")
        symbols[label] = symbol
        if kind in STATIC_KINDS:
            (derived_features if symbol["derived"] else free_features).append(label)
        elif kind == "scalar":
            scalars.append(label)
        elif kind in PLACE_STATIC_KINDS:
            (place_derived if symbol["derived"] else place_free).append(label)
        elif kind == PLACE_SCALAR_KIND:
            place_scalars.append(label)
        elif kind == PLACE_EVENT_TYPE_KIND:
            place_event_types.append(label)
        elif kind == "fluent":
            rate = symbol.get("initial_rate")
            if symbol["derived"]:
                _expect(rate is None, f"the derived fluent {label} carries an initial rate")
                derived_fluents.append(label)
            else:
                _expect(
                    isinstance(rate, (int, float))
                    and not isinstance(rate, bool)
                    and 0.0 <= float(rate) <= 1.0,
                    f"the base fluent {label} has no initial rate between 0 and 1",
                )
                base_fluents.append(label)
                initial_rates[label] = float(rate)
        elif kind == NUMERIC_KIND:
            bounds = symbol.get("range")
            _expect(
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds)
                and float(bounds[0]) < float(bounds[1]),
                f"the numeric fluent {label} has no range [low, high] with low < high",
            )
            try:
                initial = InitialDistribution.from_record(symbol["initial"])
            except (KeyError, TypeError, ValueError):
                raise DefinitionError(
                    f"the numeric fluent {label} has no initial distribution"
                ) from None
            drift = symbol.get("drift", 0.0)
            _expect(
                isinstance(drift, (int, float)) and not isinstance(drift, bool),
                f"the numeric fluent {label} has no numeric drift",
            )
            numeric_fluents.append(label)
            numeric_ranges[label] = (float(bounds[0]), float(bounds[1]))
            numeric_initial[label] = initial
            numeric_drift[label] = float(drift)

    has_places = bool(place_free or place_derived or place_scalars or place_event_types)
    _expect(
        not has_places or place_rows is not None,
        "the definition has places, but no places table was given",
    )

    # Literals.
    literals: list[LiteralSpec] = []
    for i, entry in enumerate(record["literals"]):
        _expect(entry["index"] == i, f"literal {i} is written with index {entry['index']}")
        kind, role = str(entry["kind"]), entry["role"]
        key = str(entry["key"])
        _expect(kind in LITERAL_KINDS, f"the literal {key} has the unknown kind {kind!r}")
        if kind == "feature":
            _expect(role in (None, *ROLES), f"the literal {key} has the role {role!r}")
            feature = entry["feature"]
            feature_kind = symbols.get(feature, {}).get("kind")
            _expect(
                feature_kind in STATIC_KINDS + PLACE_STATIC_KINDS,
                f"the literal {key} reads {feature}, which is not a static feature",
            )
            _expect(
                role != "agent" or feature_kind in STATIC_KINDS,
                f"the literal {key} reads a place feature of the agent, which is an entity",
            )
        elif kind == "threshold":
            _expect(role in (None, *ROLES), f"the literal {key} has the role {role!r}")
            scalar_kind = symbols.get(entry["scalar"], {}).get("kind")
            _expect(
                scalar_kind in ("scalar", PLACE_SCALAR_KIND),
                f"the literal {key} reads {entry['scalar']}, which is not a scalar",
            )
            _expect(
                role != "agent" or scalar_kind == "scalar",
                f"the literal {key} reads a place scalar of the agent, which is an entity",
            )
            _expect(
                entry["operator"] == ">",
                f"the literal {key} has the operator {entry['operator']!r}; only > is defined",
            )
            float(entry["threshold"])
        elif kind == "fluent":
            _expect(
                role is None,
                f"the literal {key} is a fluent literal with the role {role!r}; fluent literals "
                f"belong to the entity scope",
            )
            _expect(
                symbols.get(entry["fluent"], {}).get("kind") == "fluent",
                f"the literal {key} reads {entry['fluent']}, which is not a fluent",
            )
        elif kind == "numeric_threshold":
            _expect(
                role is None,
                f"the literal {key} is a numeric threshold with the role {role!r}; it belongs "
                f"to the entity scope",
            )
            _expect(
                entry["fluent"] in numeric_fluents,
                f"the literal {key} reads {entry['fluent']}, which is not a numeric fluent",
            )
            _expect(
                entry["operator"] == ">",
                f"the literal {key} has the operator {entry['operator']!r}; only > is defined",
            )
            float(entry["threshold"])
        elif kind == "comparison":
            _expect(role == "binding", f"the literal {key} is a comparison with the role {role!r}")
            _expect(
                symbols.get(entry["agent_scalar"], {}).get("kind") == "scalar",
                f"the literal {key} compares {entry['agent_scalar']}, which is not a scalar",
            )
            _expect(
                symbols.get(entry["patient_scalar"], {}).get("kind")
                in ("scalar", PLACE_SCALAR_KIND),
                f"the literal {key} compares {entry['patient_scalar']}, which is not a scalar",
            )
            _expect(
                entry["operator"] == ">",
                f"the literal {key} has the operator {entry['operator']!r}; only > is defined",
            )
            float(entry["low"])
            if entry["high"] is not None:
                float(entry["high"])
        else:
            _expect(
                role == "binding", f"the literal {key} reads a constraint with the role {role!r}"
            )
        literals.append(
            LiteralSpec(key, {k: v for k, v in entry.items() if k not in ("index", "key")})
        )
    literal_keys = {spec.key for spec in literals}
    _expect(len(literal_keys) == len(literals), "the literal table has a duplicate key")

    # Rules.
    rules_by_scope: dict[str, list[RuleSpec]] = {scope: [] for scope in SCOPES}
    outputs: set[str] = set()
    for entry in record["rules"]:
        scope = entry["scope"]
        _expect(scope in SCOPES, f"the rule for {entry['output']} has the scope {scope!r}")
        rule = _rule_spec(entry, literals)
        _expect(rule.output not in outputs, f"{rule.output} has two rules")
        outputs.add(rule.output)
        if scope == "entity":
            _expect(
                rule.output in derived_features or rule.output in derived_fluents,
                f"the entity rule for {rule.output} does not compute a derived symbol",
            )
        elif scope == "place":
            _expect(
                rule.output in place_derived,
                f"the place rule for {rule.output} does not compute a derived place feature",
            )
        else:
            _expect(
                rule.output not in symbols,
                f"the {scope} rule for {rule.output} computes a symbol; constraints and "
                f"requirements are not symbols",
            )
        rules_by_scope[scope].append(rule)
    entity_rules = rules_by_scope["entity"]
    binding_rules = rules_by_scope["binding"]
    place_rules = rules_by_scope["place"]
    place_binding_rules = rules_by_scope["place_binding"]
    for label in (*derived_features, *derived_fluents, *place_derived):
        _expect(label in outputs, f"the derived symbol {label} has no rule")
    constraint_outputs = {rule.output for rule in binding_rules}
    place_constraint_outputs = {rule.output for rule in place_binding_rules}
    for spec in literals:
        if spec.reads["kind"] == "constraint":
            _expect(
                spec.reads["constraint"] in constraint_outputs | place_constraint_outputs,
                f"the literal {spec.key} reads {spec.reads['constraint']}, which no binding rule "
                f"computes",
            )

    # Layers.
    matrices = {}
    for scope in SCOPES:
        layers = [layer for layer in record["layers"] if layer.get("scope") == scope]
        if scope in ("place", "place_binding") and not has_places and not layers:
            matrices[scope] = None
            continue
        try:
            built = RuleMatrices.from_record(record["literals"], layers)
        except WorldError as error:
            raise DefinitionError(f"the {scope} layers are malformed: {error}") from None
        rules = rules_by_scope[scope]
        _expect(
            sorted(built.outputs) == sorted(rule.output for rule in rules),
            f"the {scope} layers and the {scope} rules do not compute the same outputs",
        )
        matrices[scope] = built
    _expect(
        all(layer.get("scope") in SCOPES for layer in record["layers"]),
        "a layer has no scope, or an unknown one",
    )

    # Event types.
    event_types: list[EventTypeRecord] = []
    seen: set[str] = set()
    for entry in record["event_types"]:
        label = str(entry["label"])
        _expect(
            symbols.get(label, {}).get("kind") in EVENT_TYPE_KINDS,
            f"the event type {label} is not an event-type symbol",
        )
        _expect(label not in seen, f"the event type {label} is listed twice")
        seen.add(label)
        arity = int(entry["arity"])
        _expect(
            arity in (1, 2) and list(entry["roles"]) == list(ROLES[:arity]),
            f"the event type {label} has arity {arity} and roles {entry['roles']}",
        )
        requirement = str(entry["requirement"]["output"])
        if entry["kind"] == PLACE_EVENT_TYPE_KIND:
            _expect(
                arity == 2 and requirement in place_constraint_outputs,
                f"the requirement {requirement} of the place event type {label} has no "
                f"place-binding rule",
            )
            _expect(
                not entry["precondition"]["literals"] and not entry["effects"],
                f"the place event type {label} has a precondition or effects, which stage c1 "
                f"does not define",
            )
        else:
            _expect(
                requirement in constraint_outputs,
                f"the requirement {requirement} of {label} has no binding rule",
            )
        precondition = []
        for lit in entry["precondition"]["literals"]:
            _expect(
                lit["role"] in ROLES[:arity],
                f"the precondition of {label} reads the role {lit['role']!r}",
            )
            if "threshold" in lit:
                _expect(
                    lit["fluent"] in numeric_fluents,
                    f"the precondition of {label} reads {lit['fluent']} at a threshold, but it "
                    f"is not a numeric fluent",
                )
                _expect(
                    lit.get("operator") in (">", "<="),
                    f"the numeric literal of {label} on {lit['fluent']} has the operator "
                    f"{lit.get('operator')!r}; > and <= are defined",
                )
                precondition.append(
                    Literal(
                        str(lit["role"]),
                        str(lit["fluent"]),
                        lit["operator"] == ">",
                        float(lit["threshold"]),
                    )
                )
                continue
            _expect(
                symbols.get(lit["fluent"], {}).get("kind") == "fluent",
                f"the precondition of {label} reads {lit['fluent']}, which is not a fluent",
            )
            precondition.append(Literal(str(lit["role"]), str(lit["fluent"]), bool(lit["value"])))
        effects = []
        for effect in entry["effects"]:
            _expect(
                effect["role"] in ROLES[:arity],
                f"an effect of {label} writes the role {effect['role']!r}",
            )
            if "amount" in effect:
                _expect(
                    effect["fluent"] in numeric_fluents,
                    f"an effect of {label} adds to {effect['fluent']}, which is not a numeric "
                    f"fluent",
                )
                _expect(
                    effect.get("operator") in ("+=", "-="),
                    f"the numeric effect of {label} on {effect['fluent']} has the operator "
                    f"{effect.get('operator')!r}; += and -= are defined",
                )
                amount = float(effect["amount"])
                _expect(
                    amount > 0, f"a numeric effect of {label} has a constant that is not above 0"
                )
                value: bool | float = amount if effect["operator"] == "+=" else -amount
            else:
                _expect(
                    effect["fluent"] in base_fluents,
                    f"an effect of {label} writes {effect['fluent']}, which is not a base fluent",
                )
                value = bool(effect["value"])
            condition = []
            for lit in (effect.get("condition") or {}).get("literals", ()):
                key = f"{lit.get('role')}.{lit.get('symbol')}"
                _expect(
                    lit["role"] in ROLES[:arity],
                    f"the condition of an effect of {label} reads the role {lit['role']!r}",
                )
                kind = symbols.get(lit["symbol"], {}).get("kind")
                _expect(
                    (lit["kind"] == "fluent" and kind == "fluent")
                    or (lit["kind"] == "feature" and kind in STATIC_KINDS)
                    or (lit["kind"] == "numeric" and kind == NUMERIC_KIND),
                    f"the condition literal {key} of {label} is a {lit['kind']!r} literal, but "
                    f"{lit['symbol']} is {'not a symbol' if kind is None else 'a ' + kind}",
                )
                if lit["kind"] == "numeric":
                    _expect(
                        lit.get("operator") in (">", "<="),
                        f"the numeric condition literal {key} of {label} has the operator "
                        f"{lit.get('operator')!r}; > and <= are defined",
                    )
                    condition.append(
                        ConditionLiteral(
                            str(lit["role"]),
                            str(lit["symbol"]),
                            lit["operator"] == ">",
                            float(lit["threshold"]),
                        )
                    )
                else:
                    condition.append(
                        ConditionLiteral(str(lit["role"]), str(lit["symbol"]), bool(lit["value"]))
                    )
            _expect(
                len({(c.role, c.symbol) for c in condition}) == len(condition),
                f"the condition of an effect of {label} reads a symbol twice",
            )
            effects.append(
                Effect(str(effect["role"]), str(effect["fluent"]), value, tuple(condition))
            )
        event_types.append(
            EventTypeRecord(
                label, str(entry["kind"]), arity, requirement, tuple(precondition), tuple(effects)
            )
        )

    # Entities.
    _expect(len(rows) > 0, "the entities table is empty")
    columns = list(rows[0].keys())
    allowed = (
        set(ENTITY_COLUMNS)
        | set(free_features)
        | set(scalars)
        | set(base_fluents)
        | set(numeric_fluents)
    )
    for column in columns:
        _expect(
            column in allowed,
            f"the entities table has the column {column}, which is not a label, a leaf, a free "
            f"feature, a scalar, or a base fluent",
        )
    for needed in ("label", *free_features, *scalars):
        _expect(needed in columns, f"the entities table has no column {needed}")
    fluent_columns = [f for f in base_fluents if f in columns]
    _expect(
        not fluent_columns or len(fluent_columns) == len(base_fluents),
        "the entities table has some base fluent columns but not all of them",
    )
    numeric_columns = [f for f in numeric_fluents if f in columns]
    _expect(
        not numeric_columns or len(numeric_columns) == len(numeric_fluents),
        "the entities table has some numeric fluent columns but not all of them",
    )
    labels = []
    for row in rows:
        _expect(
            set(row.keys()) == set(columns), "the entities table has rows with different columns"
        )
        labels.append(str(row["label"]))
    _expect(len(set(labels)) == len(labels), "the entities table has a duplicate label")
    leaves = tuple(str(row.get("leaf", "")) for row in rows)
    free_values = _bit_matrix(rows, free_features)
    scalar_values = np.array(
        [[float(row[s]) for s in scalars] for row in rows], dtype=np.float64
    ).reshape(len(rows), len(scalars))
    initial = _bit_matrix(rows, base_fluents) if fluent_columns or not base_fluents else None
    initial_numeric = None
    if numeric_columns or not numeric_fluents:
        initial_numeric = np.array(
            [[float(row[f]) for f in numeric_fluents] for row in rows], dtype=np.float64
        ).reshape(len(rows), len(numeric_fluents))
        for j, f in enumerate(numeric_fluents):
            low, high = numeric_ranges[f]
            column = initial_numeric[:, j]
            _expect(
                bool(((column >= low) & (column <= high)).all()),
                f"the entities table has a value of {f} outside its range [{low}, {high}]",
            )

    # Places.
    place_free_values = None
    place_scalar_values = None
    place_labels: list[str] = []
    place_leaves: list[str] = []
    if has_places:
        _expect(len(place_rows) > 0, "the definition has places, but the places table is empty")
        place_columns = list(place_rows[0].keys())
        allowed_place = set(ENTITY_COLUMNS) | set(place_free) | set(place_scalars)
        for column in place_columns:
            _expect(
                column in allowed_place,
                f"the places table has the column {column}, which is not a label, a leaf, a "
                f"free place feature, or a place scalar",
            )
        for needed in ("label", *place_free, *place_scalars):
            _expect(needed in place_columns, f"the places table has no column {needed}")
        for row in place_rows:
            _expect(
                set(row.keys()) == set(place_columns),
                "the places table has rows with different columns",
            )
            place_labels.append(str(row["label"]))
        _expect(
            len(set(place_labels)) == len(place_labels), "the places table has a duplicate label"
        )
        _expect(
            not set(place_labels) & set(labels_so_far := [str(r["label"]) for r in rows]),
            "a place has the label of an entity",
        )
        del labels_so_far
        place_leaves = [str(row.get("leaf", "")) for row in place_rows]
        place_free_values = _bit_matrix(place_rows, place_free)
        place_scalar_values = np.array(
            [[float(row[s]) for s in place_scalars] for row in place_rows], dtype=np.float64
        ).reshape(len(place_rows), len(place_scalars))
    else:
        _expect(not place_rows, "a places table was given, but the definition has no places")

    definition = RuntimeDefinition(
        record=dict(record),
        rule_set_id=identity,
        literals=tuple(literals),
        entity_rules=tuple(entity_rules),
        binding_rules=tuple(binding_rules),
        entity_matrices=matrices["entity"],
        binding_matrices=matrices["binding"],
        event_types=tuple(event_types),
        free_features=tuple(free_features),
        derived_features=tuple(derived_features),
        scalars=tuple(scalars),
        base_fluents=tuple(base_fluents),
        derived_fluents=tuple(derived_fluents),
        entity_labels=tuple(labels),
        leaves=leaves,
        initial_rates=initial_rates,
        free_values=free_values,
        scalar_values=scalar_values,
        initial_values=initial,
        numeric_fluents=tuple(numeric_fluents),
        numeric_ranges=numeric_ranges,
        numeric_initial=numeric_initial,
        numeric_drift=numeric_drift,
        initial_numeric=initial_numeric,
        place_labels=tuple(place_labels),
        place_leaves=tuple(place_leaves),
        place_free_features=tuple(place_free),
        place_derived_features=tuple(place_derived),
        place_scalars=tuple(place_scalars),
        place_free_values=place_free_values,
        place_scalar_values=place_scalar_values,
        place_rules=tuple(place_rules),
        place_binding_rules=tuple(place_binding_rules),
        place_matrices=matrices["place"],
        place_binding_matrices=matrices["place_binding"],
    )
    return definition


def _bit_matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(columns)), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            value = row[column]
            _expect(
                value in (0, 1, True, False),
                f"the entities table has the value {value!r} for {column} of {row['label']}; "
                f"expected 0 or 1",
            )
            matrix[i, j] = int(value)
    return matrix


def load_definition(folder: str | Path) -> RuntimeDefinition:
    """Read ``definition.json``, ``entities.csv``, and, in a world with places, ``places.csv``
    from a world run folder."""
    folder = Path(folder)
    path = folder / DEFINITION_FILE
    if not path.is_file():
        raise DefinitionError(f"{path} is missing")
    record = read_json(path)
    entities = pl.read_csv(
        folder / ENTITIES_FILE, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8}
    )
    places = None
    if any(
        s.get("kind") == PLACE_EVENT_TYPE_KIND
        or s.get("kind") in PLACE_STATIC_KINDS
        or s.get("kind") == PLACE_SCALAR_KIND
        for s in record.get("symbols", ())
    ):
        places_path = folder / PLACES_FILE
        if not places_path.is_file():
            raise DefinitionError(f"{places_path} is missing: the definition has places")
        places = pl.read_csv(places_path, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})
    return runtime_definition(record, entities, places)


__all__ = [
    "DEFINITION_FILE",
    "ENTITIES_FILE",
    "Definition",
    "DefinitionReport",
    "EventTypeRecord",
    "RuntimeDefinition",
    "build_definition",
    "check_definition",
    "constraint_literals",
    "entities_frame",
    "load_definition",
    "runtime_definition",
    "write_definition",
]
