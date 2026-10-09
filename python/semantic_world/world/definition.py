"""The world definition: ``definition.json`` and ``entities.csv``.

The definition is the language-neutral description of a world (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Files"): every symbol, one literal table, every rule in Boolean form and in matrix form, and
every event type. Rules come in two scopes. Entity rules (derived static features, derived
fluents) read literals of one entity, with ``role: null``. Binding rules (constraints and the
requirement that ANDs an event type's constraints) read literals of a binding: a feature or a
threshold of the agent or the patient, a comparison between the two, or a constraint's output.
Both scopes share the literal table; each has its own layers, written with a ``scope`` tag.

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
from semantic_world.world.dynamics import ROLES, Effect, Literal
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


def comparison_literal(item: ScalarComparison) -> LiteralSpec:
    key = item.key
    return LiteralSpec(
        key,
        {
            "kind": "comparison",
            "role": "binding",
            "agent_scalar": f"SCALARDIM.{item.agent_scalar}",
            "patient_scalar": f"SCALARDIM.{item.patient_scalar}",
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


_GROUP_ORDER = {
    (None, "feature"): 0,
    (None, "threshold"): 1,
    (None, "fluent"): 2,
    ("agent", "feature"): 3,
    ("agent", "threshold"): 4,
    ("patient", "feature"): 5,
    ("patient", "threshold"): 6,
    ("binding", "comparison"): 7,
    ("binding", "constraint"): 8,
}


def _literal_sort_key(spec: LiteralSpec, position_of: dict[str, int]) -> tuple:
    reads = spec.reads
    group = _GROUP_ORDER[(reads["role"], reads["kind"])]
    if reads["kind"] == "feature":
        return (group, position_of[reads["feature"]], 0.0)
    if reads["kind"] == "threshold":
        return (group, _natural(reads["scalar"]), reads["threshold"])
    if reads["kind"] == "fluent":
        return (group, _natural(reads["fluent"]), 0.0)
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {lit.key: i for i, lit in enumerate(self.literals)})

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
        for scope, rules in (("entity", self.entity_rules), ("binding", self.binding_rules)):
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
        for scope, matrices in (
            ("entity", self.entity_matrices),
            ("binding", self.binding_matrices),
        ):
            for layer in matrices.layers_record():
                records.append({"scope": scope, **layer})
        return records

    def event_types_record(self) -> list[dict[str, Any]]:
        return [et.record() for et in self.event_types.event_types]

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
        return runtime_definition(record, entities)

    # Literal values ---------------------------------------------------------------------------

    @property
    def taxonomy(self):
        return self.statics.taxonomy

    def _position(self, label: str) -> int:
        return self.statics.features[label].position

    def entity_literal_values(
        self, values: np.ndarray, scalars: np.ndarray, base_fluents: np.ndarray
    ) -> np.ndarray:
        """The base literal matrix of entities: free features from ``values`` (the static
        feature matrix, by position), thresholds from ``scalars``, base fluents from
        ``base_fluents`` (``(n, base fluents)``), and 0 for every literal that a rule computes or
        that a binding reads."""
        n = values.shape[0]
        matrix = self.entity_matrices.literal_matrix(n)
        base_index = {label: i for i, label in enumerate(self.fluents.base_labels)}
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            if reads["role"] is not None:
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
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            role = reads["role"]
            if role is None or reads["kind"] == "constraint":
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


# ---------------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------------


def _symbols(
    statics: StaticWorld, fluents: Fluents, event_types: EventTypes
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


def build_definition(statics: StaticWorld, fluents: Fluents, event_types: EventTypes) -> Definition:
    """Assemble the definition: the literal table, both rule scopes, the matrices, and the
    identity."""
    taxonomy = statics.taxonomy
    position_of = {f.label: f.position for f in statics.features.features}
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
    literals = tuple(
        sorted(literal_specs.values(), key=lambda s: _literal_sort_key(s, position_of))
    )
    entity_matrices = build_matrices(literals, entity_rules)
    binding_matrices = build_matrices(literals, binding_rules)
    symbols = tuple(_symbols(statics, fluents, event_types))
    draft = Definition(
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
        "",
    )
    identity = rule_set_id(
        symbols, draft.literals_record(), draft.rules_record(), draft.event_types_record()
    )
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
    )


# ---------------------------------------------------------------------------------------------
# The agreement test
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DefinitionReport:
    entity: AgreementReport
    binding: AgreementReport


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
    return DefinitionReport(entity_report, total)


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
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})


def write_definition(definition: Definition, folder: str | Path) -> None:
    folder = Path(folder)
    write_json(folder / DEFINITION_FILE, definition.record())
    entities_frame(definition).write_csv(folder / ENTITIES_FILE)


# ---------------------------------------------------------------------------------------------
# Loading: the definition as the runtime reads it
# ---------------------------------------------------------------------------------------------

STATIC_KINDS = ("property", "part")
SYMBOL_KINDS = STATIC_KINDS + ("scalar", "fluent", "event_type", "event_type_category")
LITERAL_KINDS = ("feature", "threshold", "fluent", "comparison", "constraint")
SCOPES = ("entity", "binding")
ENTITY_COLUMNS = ("label", "leaf")


@dataclass(frozen=True)
class EventTypeRecord:
    """One event type as the runtime reads it: its roles, the output label of its requirement,
    its precondition literals, and its effects. A category of event types has no events."""

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "_entity_index", {v: i for i, v in enumerate(self.entity_labels)})
        object.__setattr__(self, "_event_types", {et.label: et for et in self.event_types})
        object.__setattr__(self, "_base_index", {v: i for i, v in enumerate(self.base_fluents)})
        object.__setattr__(
            self, "_derived_index", {v: i for i, v in enumerate(self.derived_fluents)}
        )
        object.__setattr__(self, "_cones", _fluent_cones(self))
        object.__setattr__(self, "cache", {})
        for arr in (self.free_values, self.scalar_values, self.initial_values):
            if arr is not None:
                arr.setflags(write=False)

    @property
    def entity_count(self) -> int:
        return len(self.entity_labels)

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

    def cone(self, fluent: str) -> tuple[str, ...]:
        """The base fluents a fluent depends on: itself for a base fluent, and for a derived
        fluent every base fluent its rule reads, directly or through other derived fluents."""
        return self._cones[fluent]  # type: ignore[attr-defined]


def _fluent_cones(definition: RuntimeDefinition) -> dict[str, tuple[str, ...]]:
    literal_of = {spec.key: spec.reads for spec in definition.literals}
    rules = {rule.output: rule for rule in definition.entity_rules}
    cones: dict[str, tuple[str, ...]] = {f: (f,) for f in definition.base_fluents}
    base_order = {f: i for i, f in enumerate(definition.base_fluents)}

    def cone_of(label: str) -> tuple[str, ...]:
        if label in cones:
            return cones[label]
        found: set[str] = set()
        for key in rules[label].inputs:
            reads = literal_of[key]
            if reads["kind"] == "fluent":
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
    record: Mapping[str, Any], entities: Sequence[Mapping[str, Any]] | pl.DataFrame
) -> RuntimeDefinition:
    """Build the runtime's definition from a definition record (the contents of
    ``definition.json``) and an entities table (the rows of ``entities.csv``, or the same rows
    without the fluent columns). Checks the identity and every reference between the tables."""
    rows = entities.to_dicts() if isinstance(entities, pl.DataFrame) else list(entities)
    try:
        return _runtime_definition(record, rows)
    except (KeyError, TypeError) as error:
        raise DefinitionError(f"the definition record is malformed: {error!r}") from None


def _runtime_definition(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
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
    for symbol in record["symbols"]:
        label, kind = str(symbol["label"]), str(symbol["kind"])
        _expect(kind in SYMBOL_KINDS, f"the symbol {label} has the unknown kind {kind!r}")
        _expect(label not in symbols, f"the symbol {label} is listed twice")
        symbols[label] = symbol
        if kind in STATIC_KINDS:
            (derived_features if symbol["derived"] else free_features).append(label)
        elif kind == "scalar":
            scalars.append(label)
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
            _expect(
                symbols.get(feature, {}).get("kind") in STATIC_KINDS,
                f"the literal {key} reads {feature}, which is not a static feature",
            )
        elif kind == "threshold":
            _expect(role in (None, *ROLES), f"the literal {key} has the role {role!r}")
            _expect(
                symbols.get(entry["scalar"], {}).get("kind") == "scalar",
                f"the literal {key} reads {entry['scalar']}, which is not a scalar",
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
        elif kind == "comparison":
            _expect(role == "binding", f"the literal {key} is a comparison with the role {role!r}")
            for side in ("agent_scalar", "patient_scalar"):
                _expect(
                    symbols.get(entry[side], {}).get("kind") == "scalar",
                    f"the literal {key} compares {entry[side]}, which is not a scalar",
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
    entity_rules: list[RuleSpec] = []
    binding_rules: list[RuleSpec] = []
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
            entity_rules.append(rule)
        else:
            _expect(
                rule.output not in symbols,
                f"the binding rule for {rule.output} computes a symbol; constraints and "
                f"requirements are not symbols",
            )
            binding_rules.append(rule)
    for label in (*derived_features, *derived_fluents):
        _expect(label in outputs, f"the derived symbol {label} has no rule")
    constraint_outputs = {rule.output for rule in binding_rules}
    for spec in literals:
        if spec.reads["kind"] == "constraint":
            _expect(
                spec.reads["constraint"] in constraint_outputs,
                f"the literal {spec.key} reads {spec.reads['constraint']}, which no binding rule "
                f"computes",
            )

    # Layers.
    matrices = {}
    for scope in SCOPES:
        layers = [layer for layer in record["layers"] if layer.get("scope") == scope]
        try:
            built = RuleMatrices.from_record(record["literals"], layers)
        except WorldError as error:
            raise DefinitionError(f"the {scope} layers are malformed: {error}") from None
        rules = entity_rules if scope == "entity" else binding_rules
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
            symbols.get(label, {}).get("kind") in ("event_type", "event_type_category"),
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
            _expect(
                effect["fluent"] in base_fluents,
                f"an effect of {label} writes {effect['fluent']}, which is not a base fluent",
            )
            effects.append(
                Effect(str(effect["role"]), str(effect["fluent"]), bool(effect["value"]))
            )
        event_types.append(
            EventTypeRecord(
                label, str(entry["kind"]), arity, requirement, tuple(precondition), tuple(effects)
            )
        )

    # Entities.
    _expect(len(rows) > 0, "the entities table is empty")
    columns = list(rows[0].keys())
    allowed = set(ENTITY_COLUMNS) | set(free_features) | set(scalars) | set(base_fluents)
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
    """Read ``definition.json`` and ``entities.csv`` from a world run folder."""
    folder = Path(folder)
    path = folder / DEFINITION_FILE
    if not path.is_file():
        raise DefinitionError(f"{path} is missing")
    record = read_json(path)
    entities = pl.read_csv(
        folder / ENTITIES_FILE, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8}
    )
    return runtime_definition(record, entities)


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
