"""Expressions: the printed and parsed form of rules, and random read-once formulas.

Literals are feature labels. The operators are ``NOT``, ``AND``, ``OR``, and ``XOR``, with
parentheses. Precedence, from tightest to loosest, is ``NOT``, ``AND``, ``XOR``, ``OR``.
Printed expressions put parentheses around every nested operator, so no reader has to rely on
precedence. The constants ``TRUE`` and ``FALSE`` are accepted so that a constant function has a
printable minimal DNF.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from semantic_world.taxonomy.boolean import OPERATORS, Implicant, TruthTable, settings_array

KEYWORDS = ("NOT", "AND", "OR", "XOR", "TRUE", "FALSE")


class ExpressionError(ValueError):
    """A malformed expression, or one evaluated over the wrong inputs."""


# ---------------------------------------------------------------------------------------------
# The syntax tree
# ---------------------------------------------------------------------------------------------


class Expr:
    """A Boolean expression. Subclasses are :class:`Var`, :class:`Const`, :class:`Not`, and
    :class:`Op`."""

    def variables(self) -> tuple[str, ...]:
        """The distinct variables, in order of first appearance."""
        seen: dict[str, None] = {}
        for name in self._variable_occurrences():
            seen.setdefault(name, None)
        return tuple(seen)

    def literal_count(self) -> int:
        """The number of variable occurrences."""
        return len(self._variable_occurrences())

    def _variable_occurrences(self) -> list[str]:
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

    def _variable_occurrences(self) -> list[str]:
        return [self.name]

    def depth(self) -> int:
        return 0

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        try:
            return np.asarray(env[self.name], dtype=bool)
        except KeyError:
            raise ExpressionError(f"no value for {self.name}") from None


@dataclass(frozen=True)
class Const(Expr):
    value: bool

    def _variable_occurrences(self) -> list[str]:
        return []

    def depth(self) -> int:
        return 0

    def evaluate(self, env: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.bool_(self.value)


@dataclass(frozen=True)
class Not(Expr):
    operand: Expr

    def _variable_occurrences(self) -> list[str]:
        return self.operand._variable_occurrences()

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

    def _variable_occurrences(self) -> list[str]:
        return [name for operand in self.operands for name in operand._variable_occurrences()]

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
    if isinstance(expr, Const):
        return "TRUE" if expr.value else "FALSE"
    if isinstance(expr, Not):
        if isinstance(expr.operand, Op):
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
    r"\s*(?:(?P<lparen>\()|(?P<rparen>\))|(?P<word>[A-Za-z_][A-Za-z0-9_.]*)|(?P<bad>\S))"
)


@dataclass(frozen=True)
class _Token:
    kind: str  # "(", ")", "keyword", "name", or "end"
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
            return Var(token.text)
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


def parse_expression(text: str) -> Expr:
    """Parse an expression. Raises :class:`ExpressionError` with the position of a problem."""
    if not isinstance(text, str):
        raise ExpressionError(f"expected an expression string, got {type(text).__name__}")
    return _Parser(text).parse()


# ---------------------------------------------------------------------------------------------
# From a DNF, and random read-once formulas
# ---------------------------------------------------------------------------------------------


def literal(name: str, negated: bool) -> Expr:
    return Not(Var(name)) if negated else Var(name)


def from_dnf(terms: Iterable[Implicant], inputs: Sequence[str]) -> Expr:
    """The expression of a disjunctive normal form over the given input labels.

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
    inputs: Sequence[str],
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
    if len(set(names)) != len(names):
        raise ValueError(f"the inputs {names} contain a duplicate")
    if max_depth < 1:
        raise ValueError(f"the nesting depth must be at least 1, got {max_depth}")
    operators = [op for op in OPERATORS if operator_weights.get(op, 0) > 0]
    if not operators:
        raise ValueError("at least one operator must have a positive weight")
    weights = np.array([operator_weights[op] for op in operators], dtype=float)
    probabilities = weights / weights.sum()

    def leaf(name: str) -> Expr:
        return literal(name, bool(rng.random() < negation_probability))

    def build(group: list[str], budget: int) -> Expr:
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
