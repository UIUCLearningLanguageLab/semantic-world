"""Stage 2 acceptance tests: the expression parser and printer, and read-once formulas."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from semantic_world.taxonomy.boolean import SHJ_CANONICAL, TruthTable, minimal_dnf
from semantic_world.taxonomy.expressions import (
    Const,
    Expr,
    ExpressionError,
    Not,
    Op,
    Var,
    format_expression,
    from_dnf,
    parse_expression,
    random_read_once,
)

ABC = ("A", "B", "C")


def table_of(text: str, inputs=ABC) -> TruthTable:
    return parse_expression(text).truth_table(inputs)


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


def test_precedence_not_and_xor_or() -> None:
    assert table_of("A OR B AND C") == table_of("A OR (B AND C)")
    assert table_of("A OR B AND C") != table_of("(A OR B) AND C")
    assert table_of("A XOR B AND C") == table_of("A XOR (B AND C)")
    assert table_of("A OR B XOR C") == table_of("A OR (B XOR C)")
    assert table_of("A OR B XOR C") != table_of("(A OR B) XOR C")
    assert table_of("NOT A AND B") == table_of("(NOT A) AND B")
    assert table_of("NOT A AND B") != table_of("NOT (A AND B)")
    assert table_of("NOT NOT A") == table_of("A")


def test_parsed_structure() -> None:
    expr = parse_expression("(HAS.2 AND NOT IS.5) OR IS.7")
    assert expr == Op("OR", (Op("AND", (Var("HAS.2"), Not(Var("IS.5")))), Var("IS.7")))
    assert expr.variables() == ("HAS.2", "IS.5", "IS.7")
    assert expr.literal_count() == 3
    assert expr.depth() == 2
    # Chains of one operator are n-ary; parenthesized chains keep their structure.
    assert parse_expression("A AND B AND C") == Op("AND", (Var("A"), Var("B"), Var("C")))
    assert parse_expression("(A AND B) AND C") == Op(
        "AND", (Op("AND", (Var("A"), Var("B"))), Var("C"))
    )
    assert parse_expression("A XOR B XOR C").truth_table(ABC) == TruthTable.from_function(
        3, lambda x: sum(x) & 1
    )


def test_constants() -> None:
    assert table_of("TRUE") == TruthTable.constant(3, True)
    assert table_of("FALSE AND A") == TruthTable.constant(3, False)
    assert table_of("TRUE AND A") == table_of("A")
    assert parse_expression("TRUE") == Const(True)
    assert str(Const(False)) == "FALSE"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "A AND",
        "AND A",
        "(A OR B",
        "A OR B)",
        "A B",
        "A && B",
        "A OR )",
        "NOT",
        "()",
        "a and b",
        "A OR B AND",
    ],
)
def test_malformed_expressions_are_rejected(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(text)


def test_error_messages_give_the_position() -> None:
    with pytest.raises(ExpressionError, match="position 6"):
        parse_expression("A AND && B")
    with pytest.raises(ExpressionError, match="position 7"):
        parse_expression("(A OR B")
    with pytest.raises(ExpressionError):
        parse_expression(42)  # type: ignore[arg-type]


def test_truth_table_checks_the_inputs() -> None:
    expr = parse_expression("A AND D")
    with pytest.raises(ExpressionError, match="D"):
        expr.truth_table(ABC)
    with pytest.raises(ExpressionError, match="duplicate"):
        expr.truth_table(("A", "D", "A"))
    with pytest.raises(ExpressionError):
        expr.evaluate({"A": np.array([True])})
    with pytest.raises(ExpressionError):
        Op("NAND", (Var("A"), Var("B")))
    with pytest.raises(ExpressionError):
        Op("AND", (Var("A"),))


def test_evaluate_broadcasts_over_arrays() -> None:
    expr = parse_expression("(A AND NOT B) OR C")
    a = np.array([0, 1, 1, 0], dtype=np.uint8)
    b = np.array([0, 0, 1, 1], dtype=np.uint8)
    c = np.array([0, 0, 0, 1], dtype=np.uint8)
    out = expr.evaluate({"A": a, "B": b, "C": c})
    assert out.tolist() == [False, True, False, True]


# ---------------------------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------------------------


def test_printer_parenthesizes_every_nested_operator() -> None:
    expr = Op("OR", (Op("AND", (Var("HAS.2"), Not(Var("IS.5")))), Var("IS.7")))
    assert str(expr) == "(HAS.2 AND NOT IS.5) OR IS.7"
    nested = Op("AND", (Op("OR", (Var("A"), Op("XOR", (Var("B"), Var("C"))))), Not(Var("D"))))
    assert str(nested) == "(A OR (B XOR C)) AND NOT D"
    assert str(Not(Op("AND", (Var("A"), Var("B"))))) == "NOT (A AND B)"
    assert str(Not(Not(Var("A")))) == "NOT NOT A"
    assert format_expression(Var("IS.4")) == "IS.4"


def test_shj_rules_print_as_their_minimal_dnf_over_the_input_labels() -> None:
    inputs = ("HAS.2", "IS.5", "IS.7")
    expr = from_dnf(minimal_dnf(SHJ_CANONICAL["IV"]), inputs)
    assert (
        str(expr)
        == "(NOT HAS.2 AND NOT IS.5) OR (NOT HAS.2 AND NOT IS.7) OR (NOT IS.5 AND NOT IS.7)"
    )
    assert expr.truth_table(inputs).bit_string() == "11101000"
    # The specification's forms, checked as functions.
    spec_forms = {
        "I": "NOT A",
        "II": "(A AND B) OR (NOT A AND NOT B)",
        "III": "(C AND NOT B) OR (NOT A AND NOT C)",
        "IV": "(NOT A AND NOT B) OR (NOT A AND NOT C) OR (NOT B AND NOT C)",
        "V": "(A AND B AND C) OR (NOT A AND NOT B) OR (NOT A AND NOT C)",
        "VI": "A XOR B XOR C XOR TRUE",
    }
    for shj_type, text in spec_forms.items():
        printed = from_dnf(minimal_dnf(SHJ_CANONICAL[shj_type]), ABC)
        assert printed.truth_table(ABC) == table_of(text) == SHJ_CANONICAL[shj_type]
        assert parse_expression(str(printed)).truth_table(ABC) == SHJ_CANONICAL[shj_type]
    assert str(from_dnf(minimal_dnf(SHJ_CANONICAL["I"]), ABC)) == "NOT A"
    assert (
        str(from_dnf(minimal_dnf(SHJ_CANONICAL["III"]), ABC))
        == "(NOT A AND NOT C) OR (NOT B AND C)"
    )


def test_from_dnf_constants_and_errors() -> None:
    assert from_dnf((), ABC) == Const(False)
    assert from_dnf(((None, None, None),), ABC) == Const(True)
    assert from_dnf(((1, None, None),), ABC) == Var("A")
    with pytest.raises(ExpressionError):
        from_dnf(((1, 0),), ABC)


# ---------------------------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------------------------


def _random_expression(rng: np.random.Generator, names: list[str], budget: int) -> Expr:
    """A random expression with repeated variables, nested NOTs, and constants."""
    kind = rng.random()
    if budget == 0 or kind < 0.3:
        if rng.random() < 0.08:
            return Const(bool(rng.random() < 0.5))
        return Var(names[int(rng.integers(len(names)))])
    if kind < 0.45:
        return Not(_random_expression(rng, names, budget - 1))
    operator = ("AND", "OR", "XOR")[int(rng.integers(3))]
    count = int(rng.integers(2, 5))
    return Op(operator, tuple(_random_expression(rng, names, budget - 1) for _ in range(count)))


def test_every_printed_expression_parses_back_to_the_same_truth_table() -> None:
    rng = np.random.default_rng(7)
    names = ["IS.1", "IS.2", "HAS.3", "HAS.4", "CAN.5"]
    for _ in range(300):
        expr = _random_expression(rng, names, budget=4)
        text = str(expr)
        reparsed = parse_expression(text)
        assert reparsed.truth_table(names) == expr.truth_table(names), text
        assert str(reparsed) == text  # printing is stable


# ---------------------------------------------------------------------------------------------
# Read-once formulas
# ---------------------------------------------------------------------------------------------

MIXES = ({"AND": 1, "OR": 1, "XOR": 1}, {"XOR": 1}, {"AND": 1}, {"AND": 3, "OR": 1, "XOR": 0})


@pytest.mark.parametrize("mix", MIXES, ids=lambda m: "+".join(k for k, v in m.items() if v))
def test_every_read_once_formula_depends_on_every_input(mix: dict[str, float]) -> None:
    for arity, depth, seed in itertools.product(range(1, 8), range(1, 5), range(6)):
        inputs = [f"IS.{i}" for i in range(1, arity + 1)]
        rng = np.random.default_rng(seed)
        expr = random_read_once(inputs, depth, mix, 0.3, rng)
        # Read once: every input appears exactly once.
        assert expr.literal_count() == arity
        assert sorted(expr.variables()) == sorted(inputs)
        table = expr.truth_table(inputs)
        assert table.relevant_inputs() == tuple(range(arity)), str(expr)
        assert expr.depth() <= depth
        if arity >= 2:
            assert expr.depth() >= 1
        # The printed form parses back to the same function.
        assert parse_expression(str(expr)).truth_table(inputs) == table
        # Only operators with positive weight appear.
        allowed = {k for k, v in mix.items() if v > 0}
        assert _operators_used(expr) <= allowed


def _operators_used(expr: Expr) -> set[str]:
    if isinstance(expr, Op):
        return {expr.operator}.union(*(_operators_used(o) for o in expr.operands))
    if isinstance(expr, Not):
        return _operators_used(expr.operand)
    return set()


def test_depth_one_is_a_single_operator_over_all_inputs() -> None:
    inputs = ["A", "B", "C", "D", "E"]
    for seed in range(20):
        expr = random_read_once(
            inputs, 1, {"AND": 1, "OR": 1, "XOR": 1}, 0.5, np.random.default_rng(seed)
        )
        assert isinstance(expr, Op)
        assert len(expr.operands) == 5
        assert all(isinstance(o, Var | Not) for o in expr.operands)
        assert expr.depth() == 1


def test_deeper_formulas_do_nest() -> None:
    inputs = [f"X{i}" for i in range(6)]
    depths = {
        random_read_once(inputs, 3, {"AND": 1, "OR": 1}, 0.2, np.random.default_rng(seed)).depth()
        for seed in range(40)
    }
    assert depths & {2, 3}
    assert max(depths) <= 3


def test_negation_probability_extremes() -> None:
    inputs = ["A", "B", "C", "D"]
    never = random_read_once(inputs, 2, {"AND": 1}, 0.0, np.random.default_rng(1))
    always = random_read_once(inputs, 2, {"AND": 1}, 1.0, np.random.default_rng(1))
    assert "NOT" not in str(never)
    assert str(always).count("NOT") == 4


def test_read_once_is_deterministic_for_a_seed() -> None:
    inputs = ["IS.1", "HAS.2", "IS.3", "HAS.4", "IS.5"]
    mix = {"AND": 1, "OR": 2, "XOR": 1}
    a = random_read_once(inputs, 3, mix, 0.2, np.random.default_rng(99))
    b = random_read_once(inputs, 3, mix, 0.2, np.random.default_rng(99))
    c = random_read_once(inputs, 3, mix, 0.2, np.random.default_rng(100))
    assert a == b
    assert a != c


def test_read_once_single_input_is_a_literal() -> None:
    expr = random_read_once(["A"], 2, {"AND": 1}, 0.0, np.random.default_rng(0))
    assert expr == Var("A")


def test_read_once_rejects_bad_arguments() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_read_once([], 1, {"AND": 1}, 0.0, rng)
    with pytest.raises(ValueError):
        random_read_once(["A", "A"], 1, {"AND": 1}, 0.0, rng)
    with pytest.raises(ValueError):
        random_read_once(["A", "B"], 0, {"AND": 1}, 0.0, rng)
    with pytest.raises(ValueError):
        random_read_once(["A", "B"], 1, {"AND": 0}, 0.0, rng)
