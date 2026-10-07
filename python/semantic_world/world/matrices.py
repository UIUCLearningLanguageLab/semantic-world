"""Rules as two-layer threshold matrices (REL.18), and the agreement test.

A rule is an output, an ordered list of inputs (the keys of literals), and a truth table. Its
Boolean form is readable; its matrix form is what the runtime computes:

- **Literal vector.** For an entity or a binding, ``ℓ`` holds the value of every literal in the
  literal table, followed by every complement (1 minus the value).
- **Term layer.** Each row of ``W1`` marks the literals of one term of the rule's minimal
  disjunctive normal form, with threshold equal to the number of literals in the term. A term
  is true when ``W1_j · ℓ ≥ θ_j``.
- **Output layer.** Each row of ``W2`` marks the terms of one output, with threshold 1.

A rule that is always true has one term with no literals and threshold 0. A rule that is always
false has no terms. Rules are grouped into dependency layers: a rule's layer is one more than the
highest layer of any rule whose output it reads (base literals are layer 0). Threshold literals
on scalars, and comparisons, are computed before the first layer, as literals. The layers of a
definition are written sparsely: for each term, the indices of its literals and whether each is
complemented; for each output, the indices of its terms.

**The agreement test.** For every rule, the matrix form and the truth table must give the same
output for every entity of a run. For every rule with at most ``MAX_EXHAUSTIVE_INPUTS`` inputs,
they must also agree on every setting of the inputs. A disagreement is an
:class:`AgreementError`, and the run that finds one fails.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.common.boolean import TruthTable, minimal_dnf, settings_array
from semantic_world.world.errors import AgreementError, WorldError

MAX_EXHAUSTIVE_INPUTS = 12
"""Rules with at most this many inputs are checked on every input setting."""


# ---------------------------------------------------------------------------------------------
# The Boolean form: literals and rules
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteralSpec:
    """One literal of the literal table: its key (the variable name rules use for it) and what
    it reads, as the mapping written to the definition (for example ``{"kind": "feature",
    "role": None, "feature": "IS.3"}`` or ``{"kind": "threshold", "role": None,
    "scalar": "SC.2", "operator": ">", "threshold": 0.4127}``)."""

    key: str
    reads: Mapping[str, Any]

    def record(self, index: int) -> dict[str, Any]:
        return {"index": index, "key": self.key, **dict(self.reads)}


@dataclass(frozen=True)
class RuleSpec:
    """One rule in Boolean form: its output, its inputs (literal keys, in truth-table order),
    its truth table, and its printed expression."""

    output: str
    inputs: tuple[str, ...]
    table: TruthTable
    expression: str | None = None

    def __post_init__(self) -> None:
        if self.table.arity != len(self.inputs):
            raise WorldError(
                f"the rule for {self.output} has {len(self.inputs)} inputs but a truth table "
                f"of arity {self.table.arity}"
            )
        if len(set(self.inputs)) != len(self.inputs):
            raise WorldError(f"the rule for {self.output} reads a literal twice")

    def record(self, literal_index: Mapping[str, int]) -> dict[str, Any]:
        """The rule's entry in a definition: inputs as literal indices."""
        return {
            "output": self.output,
            "inputs": [literal_index[key] for key in self.inputs],
            "truth_table": self.table.bit_string(),
            "expression": self.expression,
        }


# ---------------------------------------------------------------------------------------------
# The matrix form
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One row of a term layer: the literal indices it reads, whether each is complemented, and
    the threshold (the number of literals)."""

    literals: tuple[int, ...]
    complemented: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.literals) != len(self.complemented):
            raise WorldError("a term needs one complement flag per literal")

    @property
    def threshold(self) -> int:
        return len(self.literals)

    def record(self) -> dict[str, Any]:
        return {
            "literals": list(self.literals),
            "complemented": list(self.complemented),
            "threshold": self.threshold,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> Term:
        term = cls(
            tuple(int(i) for i in data["literals"]), tuple(bool(c) for c in data["complemented"])
        )
        if int(data.get("threshold", term.threshold)) != term.threshold:
            raise WorldError(
                f"a term lists {term.threshold} literals but threshold {data['threshold']}"
            )
        return term


@dataclass(frozen=True)
class RuleMatrix:
    """One rule in matrix form: its terms (the rows of ``W1`` that belong to it) and its row of
    ``W2``, which marks every one of its terms with threshold 1."""

    output: str
    terms: tuple[Term, ...]

    @property
    def is_constant_true(self) -> bool:
        return len(self.terms) == 1 and not self.terms[0].literals

    @property
    def is_constant_false(self) -> bool:
        return not self.terms


@dataclass(frozen=True)
class Layer:
    """One dependency layer: a term layer and an output layer over every rule in it.

    The dense matrices are built once: ``w1`` has one row per term over the doubled literal
    vector (``2 × literal_count`` columns, values then complements), ``theta1`` the term
    thresholds, and ``w2`` one row per rule over the terms of the layer.
    """

    index: int
    rules: tuple[RuleMatrix, ...]
    literal_count: int
    w1: np.ndarray = field(init=False, repr=False, compare=False)
    theta1: np.ndarray = field(init=False, repr=False, compare=False)
    w2: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        terms = [term for rule in self.rules for term in rule.terms]
        w1 = np.zeros((len(terms), 2 * self.literal_count), dtype=np.int32)
        theta1 = np.zeros(len(terms), dtype=np.int32)
        for j, term in enumerate(terms):
            for literal, complemented in zip(term.literals, term.complemented, strict=True):
                if not 0 <= literal < self.literal_count:
                    raise WorldError(f"literal index {literal} is outside the literal table")
                w1[j, literal + (self.literal_count if complemented else 0)] = 1
            theta1[j] = term.threshold
        w2 = np.zeros((len(self.rules), len(terms)), dtype=np.int32)
        offset = 0
        for i, rule in enumerate(self.rules):
            w2[i, offset : offset + len(rule.terms)] = 1
            offset += len(rule.terms)
        object.__setattr__(self, "w1", w1)
        object.__setattr__(self, "theta1", theta1)
        object.__setattr__(self, "w2", w2)

    @property
    def term_count(self) -> int:
        return int(self.w1.shape[0])

    def evaluate(self, literal_values: np.ndarray) -> np.ndarray:
        """The output of every rule of the layer for every row of ``literal_values`` (shape
        ``(n, literal_count)``, 0 and 1). Returns shape ``(n, rules)``, dtype ``uint8``."""
        values = np.asarray(literal_values, dtype=np.int32)
        if values.ndim != 2 or values.shape[1] != self.literal_count:
            raise WorldError(
                f"expected literal values of shape (n, {self.literal_count}), got {values.shape}"
            )
        doubled = np.concatenate([values, 1 - values], axis=1)
        terms = (doubled @ self.w1.T >= self.theta1).astype(np.int32)
        return (terms @ self.w2.T >= 1).astype(np.uint8)

    def record(self) -> dict[str, Any]:
        """The sparse form: every term of the layer, then each output with its term indices."""
        terms: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for rule in self.rules:
            start = len(terms)
            terms.extend(term.record() for term in rule.terms)
            outputs.append(
                {
                    "output": rule.output,
                    "terms": list(range(start, start + len(rule.terms))),
                    "threshold": 1,
                }
            )
        return {"layer": self.index, "terms": terms, "outputs": outputs}


@dataclass(frozen=True)
class RuleMatrices:
    """Every rule of a definition in matrix form, over one literal table, in dependency layers."""

    literals: tuple[LiteralSpec, ...]
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        index = {literal.key: i for i, literal in enumerate(self.literals)}
        if len(index) != len(self.literals):
            raise WorldError("the literal table has a duplicate key")
        rules = {rule.output: rule for layer in self.layers for rule in layer.rules}
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_rules", rules)

    @property
    def literal_count(self) -> int:
        return len(self.literals)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Every output, in evaluation order."""
        return tuple(rule.output for layer in self.layers for rule in layer.rules)

    def literal_index(self, key: str) -> int:
        try:
            return self._index[key]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"{key} is not in the literal table") from None

    def rule(self, output: str) -> RuleMatrix:
        try:
            return self._rules[output]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"{output} has no rule") from None

    def literal_matrix(self, rows: int) -> np.ndarray:
        """A zero literal matrix with one row per entity, for the caller to fill."""
        return np.zeros((rows, self.literal_count), dtype=np.uint8)

    def evaluate(self, literal_values: np.ndarray) -> dict[str, np.ndarray]:
        """Compute every output, layer by layer. ``literal_values`` (shape ``(n, literals)``)
        holds the base literals; the columns of derived literals are overwritten as their rules
        are computed. Returns each output's column, dtype ``uint8``, in evaluation order."""
        values = np.array(literal_values, dtype=np.uint8, copy=True)
        if values.ndim != 2 or values.shape[1] != self.literal_count:
            raise WorldError(
                f"expected literal values of shape (n, {self.literal_count}), got {values.shape}"
            )
        outputs: dict[str, np.ndarray] = {}
        for layer in self.layers:
            result = layer.evaluate(values)
            for j, rule in enumerate(layer.rules):
                outputs[rule.output] = result[:, j]
                column = self._index.get(rule.output)  # type: ignore[attr-defined]
                if column is not None:
                    values[:, column] = result[:, j]
        return outputs

    def evaluate_rule(
        self, output: str, inputs: Sequence[str], input_values: np.ndarray
    ) -> np.ndarray:
        """One rule's two layers alone, over its own inputs: ``input_values`` has one column per
        key of ``inputs`` (shape ``(n, len(inputs))``). Literals the rule does not read are 0."""
        values = np.asarray(input_values, dtype=np.uint8)
        if values.ndim != 2 or values.shape[1] != len(inputs):
            raise WorldError(f"expected values of shape (n, {len(inputs)}), got {values.shape}")
        full = self.literal_matrix(values.shape[0])
        for k, key in enumerate(inputs):
            full[:, self.literal_index(key)] = values[:, k]
        layer = Layer(0, (self.rule(output),), self.literal_count)
        return layer.evaluate(full)[:, 0]

    def literals_record(self) -> list[dict[str, Any]]:
        return [literal.record(i) for i, literal in enumerate(self.literals)]

    def layers_record(self) -> list[dict[str, Any]]:
        return [layer.record() for layer in self.layers]

    @classmethod
    def from_record(
        cls, literals: Sequence[Mapping[str, Any]], layers: Sequence[Mapping[str, Any]]
    ) -> RuleMatrices:
        """Rebuild the matrices from the ``literals`` and ``layers`` tables of a definition."""
        specs = []
        for i, entry in enumerate(literals):
            if int(entry["index"]) != i:
                raise WorldError(f"literal {i} is written with index {entry['index']}")
            reads = {k: v for k, v in entry.items() if k not in ("index", "key")}
            specs.append(LiteralSpec(str(entry["key"]), reads))
        built: list[Layer] = []
        for entry in layers:
            terms = [Term.from_record(t) for t in entry["terms"]]
            rules = []
            used: list[int] = []
            for out in entry["outputs"]:
                indices = [int(t) for t in out["terms"]]
                if int(out.get("threshold", 1)) != 1:
                    raise WorldError(f"the output {out['output']} has a threshold other than 1")
                used.extend(indices)
                rules.append(RuleMatrix(str(out["output"]), tuple(terms[t] for t in indices)))
            if sorted(used) != list(range(len(terms))):
                raise WorldError(f"layer {entry['layer']} has terms that no output uses, or shares")
            built.append(Layer(int(entry["layer"]), tuple(rules), len(specs)))
        return cls(tuple(specs), tuple(built))


# ---------------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------------


def dependency_layers(rules: Sequence[RuleSpec]) -> dict[str, int]:
    """Each rule's layer: 1 for a rule that reads base literals only, and otherwise one more
    than the highest layer among the rules whose outputs it reads. A cycle is an error."""
    by_output = {rule.output: rule for rule in rules}
    if len(by_output) != len(rules):
        raise WorldError("two rules have the same output")
    layers: dict[str, int] = {}
    visiting: set[str] = set()

    def layer_of(output: str) -> int:
        if output in layers:
            return layers[output]
        if output in visiting:
            raise WorldError(f"the rules form a cycle through {output}")
        visiting.add(output)
        rule = by_output[output]
        depth = 1 + max((layer_of(key) for key in rule.inputs if key in by_output), default=0)
        visiting.discard(output)
        layers[output] = depth
        return depth

    for rule in rules:
        layer_of(rule.output)
    return layers


def rule_terms(rule: RuleSpec, literal_index: Mapping[str, int]) -> tuple[Term, ...]:
    """The terms of a rule's minimal DNF over the literal table."""
    terms = []
    for implicant in minimal_dnf(rule.table):
        literals = []
        complemented = []
        for k, value in enumerate(implicant):
            if value is None:
                continue
            literals.append(literal_index[rule.inputs[k]])
            complemented.append(value == 0)
        terms.append(Term(tuple(literals), tuple(complemented)))
    return tuple(terms)


def build_matrices(literals: Sequence[LiteralSpec], rules: Sequence[RuleSpec]) -> RuleMatrices:
    """Turn rules in Boolean form into matrices over a literal table. Every input of every rule
    must be a key of the table. Rules are grouped by dependency layer, in the given order within
    a layer."""
    literal_index = {literal.key: i for i, literal in enumerate(literals)}
    if len(literal_index) != len(literals):
        raise WorldError("the literal table has a duplicate key")
    for rule in rules:
        for key in rule.inputs:
            if key not in literal_index:
                raise WorldError(f"the rule for {rule.output} reads {key}, which is not a literal")
    layer_of = dependency_layers(rules)
    grouped: dict[int, list[RuleMatrix]] = {}
    for rule in rules:
        grouped.setdefault(layer_of[rule.output], []).append(
            RuleMatrix(rule.output, rule_terms(rule, literal_index))
        )
    layers = tuple(Layer(index, tuple(grouped[index]), len(literals)) for index in sorted(grouped))
    return RuleMatrices(tuple(literals), layers)


# ---------------------------------------------------------------------------------------------
# The agreement test
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport:
    """What the agreement test checked: the rules, the entities each was checked on, the rules
    checked exhaustively, and the input settings checked in all."""

    rules: int
    entities: int
    exhaustive_rules: int
    settings: int


def evaluate_by_tables(
    matrices: RuleMatrices, rules: Sequence[RuleSpec], literal_values: np.ndarray
) -> dict[str, np.ndarray]:
    """The reference evaluation: every rule by truth-table lookup, in the matrices' layer order,
    over the same literal matrix. Used when the caller has no values of its own to compare."""
    values = np.array(literal_values, dtype=np.uint8, copy=True)
    by_output = {rule.output: rule for rule in rules}
    outputs: dict[str, np.ndarray] = {}
    for output in matrices.outputs:
        if output not in by_output:
            raise WorldError(f"{output} has a matrix form but no rule to compare it with")
        rule = by_output[output]
        columns = [values[:, matrices.literal_index(key)] for key in rule.inputs]
        stacked = (
            np.stack(columns, axis=1) if columns else np.zeros((values.shape[0], 0), dtype=np.uint8)
        )
        outputs[output] = rule.table.evaluate(stacked)
        column = matrices._index.get(output)  # type: ignore[attr-defined]
        if column is not None:
            values[:, column] = outputs[output]
    return outputs


def check_agreement(
    matrices: RuleMatrices,
    rules: Sequence[RuleSpec],
    literal_values: np.ndarray,
    expected: Mapping[str, np.ndarray] | None = None,
    max_exhaustive_inputs: int = MAX_EXHAUSTIVE_INPUTS,
) -> AgreementReport:
    """The agreement test. ``literal_values`` holds the base literals of every entity;
    ``expected`` gives each output's truth-table values for those entities (computed here by
    table lookup when not given). Raises :class:`AgreementError` on the first disagreement."""
    computed = matrices.evaluate(literal_values)
    if expected is None:
        expected = evaluate_by_tables(matrices, rules, literal_values)
    entities = int(np.asarray(literal_values).shape[0])
    exhaustive = 0
    settings = 0
    for rule in rules:
        want = np.asarray(expected[rule.output], dtype=np.uint8)
        got = computed[rule.output]
        if want.shape != got.shape:
            raise AgreementError(
                f"the rule for {rule.output} was checked on {got.shape[0]} entities but has "
                f"{want.shape[0]} expected values"
            )
        wrong = np.flatnonzero(want != got)
        if wrong.size:
            raise AgreementError(
                f"the matrix form of the rule for {rule.output} disagrees with its truth table "
                f"on {wrong.size} of {entities} entities (first at row {int(wrong[0])}): "
                f"the run cannot be trusted"
            )
        if len(rule.inputs) <= max_exhaustive_inputs:
            grid = settings_array(len(rule.inputs))
            got_all = matrices.evaluate_rule(rule.output, rule.inputs, grid)
            want_all = np.asarray(rule.table.bits, dtype=np.uint8)
            wrong = np.flatnonzero(want_all != got_all)
            if wrong.size:
                bits = "".join(str(b) for b in grid[int(wrong[0])])
                raise AgreementError(
                    f"the matrix form of the rule for {rule.output} disagrees with its truth "
                    f"table on {wrong.size} of {grid.shape[0]} input settings (first: {bits}): "
                    f"the run cannot be trusted"
                )
            exhaustive += 1
            settings += int(grid.shape[0])
    return AgreementReport(len(rules), entities, exhaustive, settings)
