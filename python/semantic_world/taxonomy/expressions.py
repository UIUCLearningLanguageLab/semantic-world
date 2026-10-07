"""Expressions: the printed and parsed form of rules, and random read-once formulas.

Literals are feature labels, or threshold literals on scalar dimensions such as
``SC.2 > 0.4127`` (whose negation prints as ``SC.2 <= 0.4127``). The operators are ``NOT``,
``AND``, ``OR``, and ``XOR``, with parentheses. Precedence, from tightest to loosest, is ``NOT``,
``AND``, ``XOR``, ``OR``. Printed expressions put parentheses around every nested operator, so no
reader has to rely on precedence. The constants ``TRUE`` and ``FALSE`` are accepted so that a
constant function has a printable minimal DNF.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from semantic_world.common.boolean import OPERATORS, Implicant, TruthTable, settings_array

KEYWORDS = ("NOT", "AND", "OR", "XOR", "TRUE", "FALSE")


class ExpressionError(ValueError):
    """A malformed expression, or one evaluated over the wrong inputs."""


# ---------------------------------------------------------------------------------------------
# The syntax tree
# ---------------------------------------------------------------------------------------------


class Expr:
    """A Boolean expression. Subclasses are :class:`Var`, :class:`Gt`, :class:`Const`,
    :class:`Not`, and :class:`Op`."""

    def variables(self) -> tuple[str, ...]:
        """The distinct variables, in order of first appearance. A threshold literal counts as
        the variable named by its :attr:`Gt.key`, such as ``SC.2>0.4127``."""
        seen: dict[str, None] = {}
        for atom in self._atom_occurrences():
            seen.setdefault(atom_key(atom), None)
        return tuple(seen)

    def atoms(self) -> tuple[Var | Gt, ...]:
        """The distinct variables and threshold literals, in order of first appearance."""
        seen: dict[str, Var | Gt] = {}
        for atom in self._atom_occurrences():
            seen.setdefault(atom_key(atom), atom)
        return tuple(seen.values())

    def literal_count(self) -> int:
        """The number of variable occurrences."""
        return len(self._atom_occurrences())

    def _atom_occurrences(self) -> list[Var | Gt]:
        raise NotImplementedError

    def depth(self) -> int:
        """The number of nested operator levels. A literal has depth 0; ``NOT`` adds nothing."""
        raise NotImplementedError

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate over arrays of values, one per variable, with NumPy broadcasting."""
        raise NotImplementedError

    def truth_table(self, inputs: Sequence[str]) -> TruthTable:
        """Tabulate the expression over the given inputs, in that order."""
        inputs = tuple(inputs)
        if len(set(inputs)) != len(inputs):
            raise ExpressionError(f"the inputs {list(inputs)} contain a duplicate")
        missing = [v for v in self.variables() if v not in inputs]
        if missing:
            raise ExpressionError(f"the expression uses {missing}, which are not among the inputs")
        grid = settings_array(len(inputs)).astype(bool)
        env = {name: grid[:, k] for k, name in enumerate(inputs)}
        out = np.broadcast_to(self.evaluate(env), (grid.shape[0],))
        return TruthTable(len(inputs), tuple(int(bit) for bit in out))

    def __str__(self) -> str:
        return format_expression(self)


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def _atom_occurrences(self) -> list[Var | Gt]:
        return [self]

    def depth(self) -> int:
        return 0

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        try:
            return np.asarray(env[self.name], dtype=bool)
        except KeyError:
            raise ExpressionError(f"no value for {self.name}") from None


@dataclass(frozen=True)
class Gt(Expr):
    """A threshold literal on a scalar dimension: ``SC.2 > 0.4127``. Inside a rule it acts
    like a binary input. Its negation prints as ``SC.2 <= 0.4127``. Thresholds are rounded to
    4 decimal places, so printed expressions parse back exactly."""

    scalar: str
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", round(float(self.threshold), 4))

    @property
    def key(self) -> str:
        """The literal as a variable name, ``SC.2>0.4127``."""
        return f"{self.scalar}>{self.threshold:.4f}"

    def _atom_occurrences(self) -> list[Var | Gt]:
        return [self]

    def depth(self) -> int:
        return 0

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        try:
            return np.asarray(env[self.key], dtype=bool)
        except KeyError:
            raise ExpressionError(f"no value for {self.key}") from None


@dataclass(frozen=True)
class Cmp(Expr):
    """A scalar comparison between two arguments of a relation: the order ``a.SC.1 - p.SC.2 >
    low`` when ``high`` is None, and the window ``low < a.SC.1 - p.SC.2 < high`` otherwise.
    Margins are rounded to 4 decimal places."""

    agent: str
    patient: str
    low: float
    high: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", round(float(self.low), 4))
        if self.high is not None:
            object.__setattr__(self, "high", round(float(self.high), 4))
            if self.high <= self.low:
                raise ExpressionError(f"a window needs low < high, got {self.low} and {self.high}")

    @property
    def key(self) -> str:
        if self.high is None:
            return f"{self.agent}-{self.patient}>{self.low:.4f}"
        return f"{self.low:.4f}<{self.agent}-{self.patient}<{self.high:.4f}"

    def _atom_occurrences(self) -> list[Var | Gt]:
        return [self]  # type: ignore[list-item]

    def depth(self) -> int:
        return 0

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        try:
            return np.asarray(env[self.key], dtype=bool)
        except KeyError:
            raise ExpressionError(f"no value for {self.key}") from None


def atom_key(atom: str | Expr) -> str:
    """The variable name of an atom: a label, a :class:`Var` name, or the key of a :class:`Gt`
    or :class:`Cmp`."""
    if isinstance(atom, str):
        return atom
    if isinstance(atom, Var):
        return atom.name
    if isinstance(atom, Gt | Cmp):
        return atom.key
    raise TypeError(f"not an atom: {atom!r}")


def as_atom(atom: str | Expr) -> Expr:
    return Var(atom) if isinstance(atom, str) else atom


@dataclass(frozen=True)
class Const(Expr):
    value: bool

    def _atom_occurrences(self) -> list[Var | Gt]:
        return []

    def depth(self) -> int:
        return 0

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.bool_(self.value)


@dataclass(frozen=True)
class Not(Expr):
    operand: Expr

    def _atom_occurrences(self) -> list[Var | Gt]:
        return self.operand._atom_occurrences()

    def depth(self) -> int:
        return self.operand.depth()

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.logical_not(self.operand.evaluate(env))


_REDUCERS = {"AND": np.logical_and, "OR": np.logical_or, "XOR": np.logical_xor}


@dataclass(frozen=True)
class Op(Expr):
    """An n-ary operator over two or more operands. ``XOR`` of many operands is their parity."""

    operator: str
    operands: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ExpressionError(f"unknown operator {self.operator!r}")
        if len(self.operands) < 2:
            raise ExpressionError(f"{self.operator} needs at least two operands")

    def _atom_occurrences(self) -> list[Var | Gt]:
        return [atom for operand in self.operands for atom in operand._atom_occurrences()]

    def depth(self) -> int:
        return 1 + max(operand.depth() for operand in self.operands)

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        values = [operand.evaluate(env) for operand in self.operands]
        return functools.reduce(_REDUCERS[self.operator], values)


# ---------------------------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------------------------


def format_expression(expr: Expr, nested: bool = False) -> str:
    """Print an expression. Every nested operator is parenthesized; the top level is not."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Gt):
        return f"{expr.scalar} > {expr.threshold:.4f}"
    if isinstance(expr, Cmp):
        if expr.high is None:
            text = f"{expr.agent} - {expr.patient} > {expr.low:.4f}"
        else:
            text = f"{expr.low:.4f} < {expr.agent} - {expr.patient} < {expr.high:.4f}"
        return f"({text})" if nested else text
    if isinstance(expr, Const):
        return "TRUE" if expr.value else "FALSE"
    if isinstance(expr, Not):
        if isinstance(expr.operand, Gt):
            return f"{expr.operand.scalar} <= {expr.operand.threshold:.4f}"
        if isinstance(expr.operand, Op | Cmp):
            return f"NOT ({format_expression(expr.operand)})"
        return f"NOT {format_expression(expr.operand)}"
    if isinstance(expr, Op):
        text = f" {expr.operator} ".join(format_expression(o, nested=True) for o in expr.operands)
        return f"({text})" if nested else text
    raise TypeError(f"not an expression: {expr!r}")


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<lparen>\()|(?P<rparen>\))|(?P<word>[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<number>-?\d+(?:\.\d+)?)|(?P<op><=|>=|<|>|-)|(?P<bad>\S))"
)
COMPARISONS = (">", "<=")


@dataclass(frozen=True)
class _Token:
    kind: str  # "(", ")", "keyword", "name", "op", "number", or "end"
    text: str
    position: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            break
        start = match.start(match.lastgroup)
        if match.group("lparen"):
            tokens.append(_Token("(", "(", start))
        elif match.group("rparen"):
            tokens.append(_Token(")", ")", start))
        elif match.group("word"):
            word = match.group("word")
            kind = "keyword" if word in KEYWORDS else "name"
            tokens.append(_Token(kind, word, start))
        elif match.group("op"):
            tokens.append(_Token("op", match.group("op"), start))
        elif match.group("number"):
            tokens.append(_Token("number", match.group("number"), start))
        else:
            raise ExpressionError(
                f"unexpected character {match.group('bad')!r} at position {start} in {text!r}"
            )
        position = match.end()
    tokens.append(_Token("end", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def advance(self) -> _Token:
        token = self.current
        self.index += 1
        return token

    def error(self, message: str) -> ExpressionError:
        token = self.current
        found = "the end of the expression" if token.kind == "end" else repr(token.text)
        return ExpressionError(
            f"{message} at position {token.position}, found {found}, in {self.text!r}"
        )

    def parse(self) -> Expr:
        if self.current.kind == "end":
            raise ExpressionError("the expression is empty")
        expr = self.parse_or()
        if self.current.kind != "end":
            raise self.error("expected an operator or the end of the expression")
        return expr

    def parse_chain(self, operator: str, parse_operand) -> Expr:
        operands = [parse_operand()]
        while self.current.kind == "keyword" and self.current.text == operator:
            self.advance()
            operands.append(parse_operand())
        return operands[0] if len(operands) == 1 else Op(operator, tuple(operands))

    def parse_or(self) -> Expr:
        return self.parse_chain("OR", self.parse_xor)

    def parse_xor(self) -> Expr:
        return self.parse_chain("XOR", self.parse_and)

    def parse_and(self) -> Expr:
        return self.parse_chain("AND", self.parse_unary)

    def parse_unary(self) -> Expr:
        if self.current.kind == "keyword" and self.current.text == "NOT":
            self.advance()
            return Not(self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        token = self.current
        if token.kind == "name":
            self.advance()
            if self.current.kind == "op" and self.current.text == "-":
                return self.parse_order(token.text)
            if self.current.kind == "op":
                return self.parse_threshold(token.text)
            return Var(token.text)
        if token.kind == "number":
            return self.parse_window()
        if token.kind == "keyword" and token.text in ("TRUE", "FALSE"):
            self.advance()
            return Const(token.text == "TRUE")
        if token.kind == "(":
            self.advance()
            expr = self.parse_or()
            if self.current.kind != ")":
                raise self.error("expected a closing parenthesis")
            self.advance()
            return expr
        raise self.error("expected a feature label, a constant, NOT, or an opening parenthesis")

    def expect(self, kind: str, text: str | None, message: str) -> _Token:
        if self.current.kind != kind or (text is not None and self.current.text != text):
            raise self.error(message)
        return self.advance()

    def parse_order(self, agent: str) -> Expr:
        """``a.SC.i - p.SC.j > low``."""
        self.expect("op", "-", "expected -")
        patient = self.expect("name", None, "expected the patient's scalar").text
        self.expect("op", ">", "a comparison uses > after the difference")
        low = self.expect("number", None, "expected a margin number").text
        return Cmp(agent, patient, float(low))

    def parse_window(self) -> Expr:
        """``low < a.SC.i - p.SC.j < high``."""
        low = self.advance().text
        self.expect("op", "<", "a window uses < after the lower margin")
        agent = self.expect("name", None, "expected the agent's scalar").text
        self.expect("op", "-", "expected - between the two scalars")
        patient = self.expect("name", None, "expected the patient's scalar").text
        self.expect("op", "<", "a window uses < before the upper margin")
        high = self.expect("number", None, "expected the upper margin").text
        try:
            return Cmp(agent, patient, float(low), float(high))
        except ExpressionError as error:
            raise ExpressionError(f"{error} in {self.text!r}") from None

    def parse_threshold(self, scalar: str) -> Expr:
        """``SC.n > x`` is a threshold literal; ``SC.n <= x`` is its negation."""
        operator = self.advance()
        if operator.text not in COMPARISONS:
            self.index -= 1
            raise self.error("a threshold literal uses > or <= (its negation)")
        if self.current.kind != "number":
            raise self.error("expected a threshold number")
        number = self.advance()
        literal_node = Gt(scalar, float(number.text))
        return literal_node if operator.text == ">" else Not(literal_node)


def parse_expression(text: str) -> Expr:
    """Parse an expression. Raises :class:`ExpressionError` with the position of a problem."""
    if not isinstance(text, str):
        raise ExpressionError(f"expected an expression string, got {type(text).__name__}")
    return _Parser(text).parse()


# ---------------------------------------------------------------------------------------------
# From a DNF, and random read-once formulas
# ---------------------------------------------------------------------------------------------


def literal(atom: str | Expr, negated: bool) -> Expr:
    """An atom (a label, a :class:`Var`, or a :class:`Gt`), negated or not."""
    node = as_atom(atom)
    return Not(node) if negated else node


def from_dnf(terms: Iterable[Implicant], inputs: Sequence[str | Expr]) -> Expr:
    """The expression of a disjunctive normal form over the given input atoms (labels or
    threshold literals).

    Terms are printed in the order given, and the literals of a term in input order. No terms
    is ``FALSE``, and a single empty term is ``TRUE``.
    """
    inputs = tuple(inputs)
    term_exprs: list[Expr] = []
    for term in terms:
        if len(term) != len(inputs):
            raise ExpressionError(
                f"a term has {len(term)} entries but there are {len(inputs)} inputs"
            )
        literals = [
            literal(name, value == 0)
            for name, value in zip(inputs, term, strict=True)
            if value is not None
        ]
        if not literals:
            return Const(True)
        term_exprs.append(literals[0] if len(literals) == 1 else Op("AND", tuple(literals)))
    if not term_exprs:
        return Const(False)
    return term_exprs[0] if len(term_exprs) == 1 else Op("OR", tuple(term_exprs))


def random_read_once(
    inputs: Sequence[str | Expr],
    max_depth: int,
    operator_weights: Mapping[str, float],
    negation_probability: float,
    rng: np.random.Generator,
) -> Expr:
    """A random read-once formula: an expression tree whose leaves are the inputs, each used
    exactly once.

    ``max_depth`` 1 is a single operator over all inputs. A larger depth allows up to that many
    levels of operators; at each level the inputs are split at random among two or more
    children, and a child with one input is a literal. Each operator is drawn from
    ``operator_weights``. Each leaf is negated with ``negation_probability``. A read-once
    formula over AND, OR, and XOR depends on every input.
    """
    names = list(inputs)
    if not names:
        raise ValueError("a read-once formula needs at least one input")
    keys = [atom_key(a) for a in names]
    if len(set(keys)) != len(keys):
        raise ValueError(f"the inputs {keys} contain a duplicate")
    if max_depth < 1:
        raise ValueError(f"the nesting depth must be at least 1, got {max_depth}")
    operators = [op for op in OPERATORS if operator_weights.get(op, 0) > 0]
    if not operators:
        raise ValueError("at least one operator must have a positive weight")
    weights = np.array([operator_weights[op] for op in operators], dtype=float)
    probabilities = weights / weights.sum()

    def leaf(name: str | Expr) -> Expr:
        return literal(name, bool(rng.random() < negation_probability))

    def build(group: list, budget: int) -> Expr:
        if len(group) == 1:
            return leaf(group[0])
        operator = operators[int(rng.choice(len(operators), p=probabilities))]
        if budget == 1:
            return Op(operator, tuple(leaf(name) for name in group))
        children = int(rng.integers(2, len(group) + 1))
        order = [group[i] for i in rng.permutation(len(group))]
        cuts = sorted(
            int(c) for c in rng.choice(np.arange(1, len(group)), size=children - 1, replace=False)
        )
        bounds = [0, *cuts, len(group)]
        parts = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        return Op(operator, tuple(build(part, budget - 1) for part in parts))

    return build(names, max_depth)
