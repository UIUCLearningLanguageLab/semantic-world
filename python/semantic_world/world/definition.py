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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semantic_world.common.boolean import TruthTable
from semantic_world.taxonomy.constraints import (
    Constraint,
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
)
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.rules import Rule, Threshold, input_key
from semantic_world.world.errors import WorldError
from semantic_world.world.event_types import ConstraintSpec, EventTypes
from semantic_world.world.fluents import Fluents, ThresholdLiteral
from semantic_world.world.identity import DEFINITION_VERSION, rule_set_id, write_json
from semantic_world.world.labels import ROLE_PREFIXES, translate, translate_expression, untranslate
from semantic_world.world.matrices import (
    AgreementReport,
    LiteralSpec,
    RuleMatrices,
    RuleSpec,
    build_matrices,
    check_agreement,
    evaluate_by_tables,
)

DEFINITION_FILE = "definition.json"
ENTITIES_FILE = "entities.csv"
BINDING_CHUNK_ROWS = 32768
KIND_OF_TYPE = {"is": "property", "has": "part"}


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
    key = translate_expression(item.key)
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
            specs.append(threshold_literal(translate(item.label), item.threshold, role))
        else:
            specs.append(feature_literal(translate(item.label), role))
    return specs


def _constraint_input_literals(constraint: Constraint) -> list[LiteralSpec]:
    specs = []
    for item in constraint.literals:
        if isinstance(item, RoleFeature):
            specs.append(feature_literal(translate(item.feature.label), ROLE_PREFIXES[item.role]))
        elif isinstance(item, RoleThreshold):
            specs.append(
                threshold_literal(
                    translate(item.threshold.label),
                    item.threshold.threshold,
                    ROLE_PREFIXES[item.role],
                )
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
    taxonomy: TaxonomyResult
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

    # Literal values ---------------------------------------------------------------------------

    def _position(self, label: str) -> int:
        return self.taxonomy.features[untranslate(label)].position

    def entity_literal_values(
        self, values: np.ndarray, scalars: np.ndarray, base_fluents: np.ndarray
    ) -> np.ndarray:
        """The base literal matrix of entities: free features from ``values`` (the taxonomy's
        non-ISA matrix), thresholds from ``scalars``, base fluents from ``base_fluents``
        (``(n, base fluents)``), and 0 for every literal that a rule computes or that a binding
        reads."""
        n = values.shape[0]
        matrix = self.entity_matrices.literal_matrix(n)
        base_index = {label: i for i, label in enumerate(self.fluents.base_labels)}
        for i, spec in enumerate(self.literals):
            reads = spec.reads
            if reads["role"] is not None:
                continue
            if reads["kind"] == "feature":
                feature = self.taxonomy.features[untranslate(reads["feature"])]
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
    taxonomy: TaxonomyResult, fluents: Fluents, event_types: EventTypes
) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for feature in taxonomy.features.features:
        if feature.type == "can":
            continue
        symbols.append(
            {
                "label": translate(feature.label),
                "kind": KIND_OF_TYPE[feature.type],
                "derived": not feature.free,
                "fluent": False,
                "arity": 1,
            }
        )
    for label in taxonomy.features.scalar_labels:
        symbols.append(
            {
                "label": translate(label),
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


def build_definition(
    taxonomy: TaxonomyResult, fluents: Fluents, event_types: EventTypes
) -> Definition:
    """Assemble the definition: the literal table, both rule scopes, the matrices, and the
    identity."""
    features = taxonomy.features
    position_of = {translate(f.label): f.position for f in features.features}
    literal_specs: dict[str, LiteralSpec] = {}

    def add(spec: LiteralSpec) -> None:
        literal_specs.setdefault(spec.key, spec)

    entity_rules: list[RuleSpec] = []
    families: dict[str, str] = {}
    for rule in taxonomy.rules.rules:
        if rule.output.type == "can":
            continue
        for spec in _rule_input_literals(rule, None):
            add(spec)
        output = translate(rule.output.label)
        inputs = tuple(translate_expression(input_key(item)) for item in rule.inputs)
        entity_rules.append(
            RuleSpec(output, inputs, rule.table, translate_expression(str(rule.expression)))
        )
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
    symbols = tuple(_symbols(taxonomy, fluents, event_types))
    draft = Definition(
        taxonomy,
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
        taxonomy,
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
    Binding rules are checked on every ordered pair of entities against the taxonomy's own
    evaluation of each constraint, and the requirement against the conjunction."""
    taxonomy = definition.taxonomy
    instances = taxonomy.instances
    values, scalars = instances.values, instances.scalars
    base = definition.entity_literal_values(values, scalars, definition.fluents.initial_values)
    expected = evaluate_by_tables(definition.entity_matrices, definition.entity_rules, base)
    for rule in taxonomy.rules.rules:
        if rule.output.type != "can":
            column = values[:, rule.output.position]
            label = translate(rule.output.label)
            if not np.array_equal(expected[label], column):
                raise WorldError(
                    f"the truth tables of {label} disagree with the taxonomy's instances"
                )
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
        "label": [translate(label) for label in instances.labels],
        "leaf": [translate(label) for label in instances.leaf_labels],
    }
    for feature in taxonomy.features.features:
        if feature.free and feature.type != "can":
            data[translate(feature.label)] = (
                instances.values[:, feature.position].astype(np.int64).tolist()
            )
    for j, label in enumerate(taxonomy.features.scalar_labels):
        data[translate(label)] = instances.scalars[:, j].astype(float).tolist()
    for i, fluent in enumerate(definition.fluents.base):
        data[fluent.label] = definition.fluents.initial_values[:, i].astype(np.int64).tolist()
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})


def write_definition(definition: Definition, folder: str | Path) -> None:
    folder = Path(folder)
    write_json(folder / DEFINITION_FILE, definition.record())
    entities_frame(definition).write_csv(folder / ENTITIES_FILE)


__all__ = [
    "DEFINITION_FILE",
    "ENTITIES_FILE",
    "Definition",
    "DefinitionReport",
    "build_definition",
    "check_definition",
    "constraint_literals",
    "entities_frame",
    "write_definition",
]
