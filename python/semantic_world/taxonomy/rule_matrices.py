"""The taxonomy's rules in the forms of the world package: symbols, the literal table, the
Boolean form, the threshold matrices (REL.18), the rule-set identity (REL.16), and the
agreement test over the instances.

Stage a1 of ``docs/specs/WORLD_AND_LANGUAGE.md``. The taxonomy keeps computing its determined
features by truth-table lookup; the matrices are a second form of the same rules, written to
``rule_matrices.json`` and checked against the truth tables at the end of every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.taxonomy.features import Feature
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import Input, RuleSet, Threshold, input_key, input_sort_key
from semantic_world.world.identity import DEFINITION_VERSION, rule_set_id
from semantic_world.world.matrices import (
    AgreementReport,
    LiteralSpec,
    RuleMatrices,
    RuleSpec,
    build_matrices,
    check_agreement,
)


@dataclass(frozen=True)
class TaxonomyMatrices:
    """The rule set of one taxonomy in the definition's forms."""

    symbols: tuple[dict[str, Any], ...]
    """One entry per non-ISA feature and per scalar: label, kind, derived, fluent, arity."""
    literals: tuple[LiteralSpec, ...]
    """The literal table: every input that some rule reads, features by matrix position, then
    threshold literals by scalar and threshold."""
    rules: tuple[RuleSpec, ...]
    """The Boolean form of every rule, in the taxonomy's evaluation order."""
    matrices: RuleMatrices
    rule_set_id: str

    def record(self) -> dict[str, Any]:
        """The contents of ``rule_matrices.json``: the ``symbols``, ``literals``, ``rules``, and
        ``layers`` tables of ``definition.json``, with the identity they hash to."""
        index = {literal.key: i for i, literal in enumerate(self.literals)}
        return {
            "version": DEFINITION_VERSION,
            "rule_set_id": self.rule_set_id,
            "symbols": list(self.symbols),
            "literals": self.matrices.literals_record(),
            "rules": [rule.record(index) for rule in self.rules],
            "layers": self.matrices.layers_record(),
        }


def _symbol(label: str, kind: str, derived: bool) -> dict[str, Any]:
    return {"label": label, "kind": kind, "derived": derived, "fluent": False, "arity": 1}


def _literal(item: Input) -> LiteralSpec:
    if isinstance(item, Threshold):
        reads = {
            "kind": "threshold",
            "role": None,
            "scalar": item.label,
            "operator": ">",
            "threshold": float(item.threshold),
        }
    else:
        reads = {"kind": "feature", "role": None, "feature": item.label}
    return LiteralSpec(input_key(item), reads)


def build_rule_matrices(rules: RuleSet) -> TaxonomyMatrices:
    """Every rule of a rule set as literals, Boolean form, matrices, and identity."""
    features = rules.features
    symbols = tuple(
        [_symbol(f.label, f.type, not f.free) for f in features.features]
        + [_symbol(label, "scalar", False) for label in features.scalar_labels]
    )
    read: dict[str, Input] = {}
    for rule in rules.rules:
        for item in rule.inputs:
            read.setdefault(input_key(item), item)
    ordered = sorted(read.values(), key=input_sort_key)
    literals = tuple(_literal(item) for item in ordered)
    specs = tuple(
        RuleSpec(
            rule.output.label,
            tuple(input_key(item) for item in rule.inputs),
            rule.table,
            str(rule.expression),
        )
        for rule in rules.rules
    )
    matrices = build_matrices(literals, specs)
    index = {literal.key: i for i, literal in enumerate(literals)}
    identity = rule_set_id(
        symbols, matrices.literals_record(), [spec.record(index) for spec in specs], []
    )
    return TaxonomyMatrices(symbols, literals, specs, matrices, identity)


def literal_values(
    taxonomy_matrices: TaxonomyMatrices,
    rules: RuleSet,
    values: np.ndarray,
    scalars: np.ndarray | None = None,
) -> np.ndarray:
    """The base literal matrix of a set of objects: free features from ``values`` (the non-ISA
    feature matrix, by position), threshold literals from ``scalars``, and 0 for every literal
    that a rule computes."""
    features = rules.features
    matrix = taxonomy_matrices.matrices.literal_matrix(values.shape[0])
    for i, literal in enumerate(taxonomy_matrices.literals):
        reads = literal.reads
        if reads["kind"] == "threshold":
            if scalars is None:
                raise ValueError(f"the literal {literal.key} needs scalar values")
            scalar = int(str(reads["scalar"]).split(".")[1])
            matrix[:, i] = (np.asarray(scalars)[:, scalar - 1] > reads["threshold"]).astype(
                np.uint8
            )
        else:
            feature: Feature = features[reads["feature"]]
            if feature.free:
                matrix[:, i] = values[:, feature.position]
    return matrix


def check_rule_agreement(
    taxonomy_matrices: TaxonomyMatrices, rules: RuleSet, instances: Instances
) -> AgreementReport:
    """The agreement test over a run's instances: the matrices must reproduce every determined
    feature that the truth tables gave the instances, and every rule with few inputs must agree
    on every input setting. Raises :class:`AgreementError` otherwise."""
    scalars = instances.scalars if rules.features.scalar_count else None
    base = literal_values(taxonomy_matrices, rules, instances.values, scalars)
    expected = {
        rule.output.label: instances.values[:, rule.output.position] for rule in rules.rules
    }
    return check_agreement(taxonomy_matrices.matrices, taxonomy_matrices.rules, base, expected)
