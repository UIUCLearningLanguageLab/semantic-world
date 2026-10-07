"""Stage 2 acceptance tests: truth tables, the arity-2 enumeration, the SHJ types, and minimal
DNF."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

from semantic_world.common.boolean import (
    ARITY_2_FUNCTIONS,
    SHJ_CANONICAL,
    SHJ_CLASS_SIZES,
    SHJ_CLASSES,
    SHJ_TRUE_SETTINGS,
    SHJ_TYPES,
    TruthTable,
    all_functions,
    apply_operator,
    dnf_literal_count,
    dnf_truth_table,
    implicant_key,
    implicant_literals,
    minimal_dnf,
    prime_implicants,
    shj_function,
    shj_type_of,
)

# ---------------------------------------------------------------------------------------------
# Truth tables
# ---------------------------------------------------------------------------------------------


def test_bit_string_round_trip() -> None:
    table = TruthTable.from_bit_string("11101000")
    assert table.arity == 3
    assert table.bits == (1, 1, 1, 0, 1, 0, 0, 0)
    assert table.bit_string() == "11101000"
    assert table.minterms() == (0, 1, 2, 4)
    assert TruthTable.from_minterms(3, (0, 1, 2, 4)) == table


def test_first_input_is_the_most_significant_bit() -> None:
    a_only = TruthTable.from_function(3, lambda x: x[0])
    assert a_only.bit_string() == "00001111"
    c_only = TruthTable.from_function(3, lambda x: x[2])
    assert c_only.bit_string() == "01010101"
    assert a_only(1, 0, 0) == 1
    assert a_only(0, 1, 1) == 0


@pytest.mark.parametrize("arity", [1, 2, 3, 5])
def test_vectorized_evaluation_matches_the_table(arity: int) -> None:
    rng = np.random.default_rng(arity)
    table = TruthTable(arity, tuple(int(b) for b in rng.integers(0, 2, size=2**arity)))
    rows = rng.integers(0, 2, size=(50, arity))
    out = table.evaluate(rows)
    assert out.dtype == np.uint8
    assert out.shape == (50,)
    for row, value in zip(rows, out, strict=True):
        assert table(*row) == value


def test_evaluation_rejects_the_wrong_width() -> None:
    table = TruthTable.from_bit_string("0001")
    with pytest.raises(ValueError):
        table.evaluate(np.zeros((3, 3), dtype=int))
    with pytest.raises(ValueError):
        table(1, 1, 1)


def test_invalid_tables_are_rejected() -> None:
    with pytest.raises(ValueError):
        TruthTable(2, (0, 1, 1))
    with pytest.raises(ValueError):
        TruthTable(1, (0, 2))
    with pytest.raises(ValueError):
        TruthTable.from_bit_string("011")
    with pytest.raises(ValueError):
        TruthTable.from_bit_string("01x1")


def test_relevant_inputs() -> None:
    both = TruthTable.from_function(2, lambda x: x[0] & x[1])
    assert both.relevant_inputs() == (0, 1)
    not_a = TruthTable.from_function(3, lambda x: 1 - x[0])
    assert not_a.relevant_inputs() == (0,)
    assert not_a.depends_on(0) and not not_a.depends_on(1)
    assert TruthTable.constant(3, True).relevant_inputs() == ()
    assert TruthTable.constant(3, True).is_constant
    assert not both.is_constant
    assert both.is_balanced is False
    assert not_a.is_balanced


def test_permute_and_negate_inputs() -> None:
    f = TruthTable.from_function(3, lambda x: x[0] & (1 - x[1]) & x[2])  # A AND NOT B AND C
    g = f.permute_inputs((1, 2, 0))  # g(x) = f(x[1], x[2], x[0]) = B AND NOT C AND A
    assert g == TruthTable.from_function(3, lambda x: x[1] & (1 - x[2]) & x[0])
    h = f.negate_inputs((True, False, False))  # h(x) = f(NOT A, B, C)
    assert h == TruthTable.from_function(3, lambda x: (1 - x[0]) & (1 - x[1]) & x[2])
    assert f.negate_output() == TruthTable(3, tuple(1 - b for b in f.bits))
    with pytest.raises(ValueError):
        f.permute_inputs((0, 0, 1))
    with pytest.raises(ValueError):
        f.negate_inputs((True,))


def test_apply_operator() -> None:
    assert apply_operator("AND", (1, 1, 1)) == 1 and apply_operator("AND", (1, 0, 1)) == 0
    assert apply_operator("OR", (0, 0, 1)) == 1 and apply_operator("OR", (0, 0)) == 0
    assert apply_operator("XOR", (1, 1, 1)) == 1 and apply_operator("XOR", (1, 1)) == 0
    with pytest.raises(ValueError):
        apply_operator("NAND", (1, 1))


# ---------------------------------------------------------------------------------------------
# The arity-2 enumeration
# ---------------------------------------------------------------------------------------------


def test_arity_2_enumeration_gives_exactly_ten_functions_depending_on_both_inputs() -> None:
    tables = [f.table for f in ARITY_2_FUNCTIONS]
    assert len(tables) == 10
    assert len(set(tables)) == 10
    for table in tables:
        assert table.relevant_inputs() == (0, 1)
    expected = {t for t in all_functions(2) if t.relevant_inputs() == (0, 1)}
    assert set(tables) == expected
    assert len(expected) == 10


def test_arity_2_enumeration_is_in_operator_order() -> None:
    operators = [f.operator for f in ARITY_2_FUNCTIONS]
    assert operators == ["AND"] * 4 + ["OR"] * 4 + ["XOR"] * 2
    first = ARITY_2_FUNCTIONS[0]
    assert (first.negate_first, first.negate_second) == (False, False)
    assert first.table.bit_string() == "0001"  # AND
    assert ARITY_2_FUNCTIONS[3].table.bit_string() == "1000"  # NOR
    assert ARITY_2_FUNCTIONS[8].table.bit_string() == "0110"  # XOR
    assert ARITY_2_FUNCTIONS[9].table.bit_string() == "1001"  # XNOR


# ---------------------------------------------------------------------------------------------
# The SHJ types
# ---------------------------------------------------------------------------------------------


def test_canonical_shj_functions_match_the_specification() -> None:
    for shj_type in SHJ_TYPES:
        table = SHJ_CANONICAL[shj_type]
        true_settings = {format(m, "03b") for m in table.minterms()}
        assert true_settings == set(SHJ_TRUE_SETTINGS[shj_type])
    assert SHJ_CANONICAL["IV"].bit_string() == "11101000"
    # Minimal DNF forms from the specification.
    assert SHJ_CANONICAL["I"] == TruthTable.from_function(3, lambda x: 1 - x[0])
    assert SHJ_CANONICAL["II"] == TruthTable.from_function(3, lambda x: int(x[0] == x[1]))
    assert SHJ_CANONICAL["VI"] == TruthTable.from_function(3, lambda x: 1 - (sum(x) & 1))


def test_shj_classes_have_the_specified_sizes() -> None:
    sizes = {shj_type: len(SHJ_CLASSES[shj_type]) for shj_type in SHJ_TYPES}
    assert sizes == SHJ_CLASS_SIZES == {"I": 6, "II": 6, "III": 24, "IV": 8, "V": 24, "VI": 2}


def test_shj_classes_cover_all_seventy_balanced_functions() -> None:
    balanced = {t for t in all_functions(3) if t.is_balanced}
    assert len(balanced) == comb(8, 4) == 70
    union: set[TruthTable] = set()
    for shj_type in SHJ_TYPES:
        members = SHJ_CLASSES[shj_type]
        assert not (union & members), f"type {shj_type} overlaps another class"
        union |= members
    assert union == balanced


def test_shj_classes_are_closed_under_negating_the_output() -> None:
    for shj_type in SHJ_TYPES:
        for table in SHJ_CLASSES[shj_type]:
            assert shj_type_of(table.negate_output()) == shj_type


def test_shj_type_of() -> None:
    for shj_type in SHJ_TYPES:
        assert shj_type_of(SHJ_CANONICAL[shj_type]) == shj_type
        for permutation in itertools.permutations(range(3)):
            for negations in itertools.product((False, True), repeat=3):
                assert shj_type_of(shj_function(shj_type, permutation, negations)) == shj_type
    assert shj_type_of(TruthTable.constant(3, True)) is None
    assert shj_type_of(TruthTable.from_function(3, lambda x: x[0] & x[1] & x[2])) is None
    with pytest.raises(ValueError):
        shj_function("VII")


def test_shj_relevant_inputs() -> None:
    assert len(SHJ_CANONICAL["I"].relevant_inputs()) == 1
    assert len(SHJ_CANONICAL["II"].relevant_inputs()) == 2
    for shj_type in ("III", "IV", "V", "VI"):
        assert SHJ_CANONICAL[shj_type].relevant_inputs() == (0, 1, 2)


# ---------------------------------------------------------------------------------------------
# Minimal DNF
# ---------------------------------------------------------------------------------------------


def _brute_force_minimum_literals(table: TruthTable) -> int:
    """The fewest literals over every subset of the prime implicants that covers the function."""
    primes = prime_implicants(table)
    minterms = set(table.minterms())
    if not minterms:
        return 0
    best = None
    for size in range(1, len(primes) + 1):
        for subset in itertools.combinations(primes, size):
            if set(dnf_truth_table(table.arity, subset).minterms()) == minterms:
                cost = sum(implicant_literals(p) for p in subset)
                if best is None or cost < best:
                    best = cost
    assert best is not None
    return best


def test_shj_complexity_scores() -> None:
    scores = {shj_type: dnf_literal_count(SHJ_CANONICAL[shj_type]) for shj_type in SHJ_TYPES}
    assert scores == {"I": 1, "II": 4, "III": 4, "IV": 6, "V": 7, "VI": 12}
    # Every member of a class has the same score.
    for shj_type in SHJ_TYPES:
        assert {dnf_literal_count(t) for t in SHJ_CLASSES[shj_type]} == {scores[shj_type]}


def test_minimal_dnf_of_the_canonical_forms() -> None:
    assert minimal_dnf(SHJ_CANONICAL["I"]) == ((0, None, None),)
    assert minimal_dnf(SHJ_CANONICAL["II"]) == ((0, 0, None), (1, 1, None))
    assert minimal_dnf(SHJ_CANONICAL["III"]) == ((0, None, 0), (None, 0, 1))
    assert minimal_dnf(SHJ_CANONICAL["IV"]) == ((0, 0, None), (0, None, 0), (None, 0, 0))
    assert minimal_dnf(SHJ_CANONICAL["V"]) == ((0, 0, None), (0, None, 0), (1, 1, 1))
    assert len(minimal_dnf(SHJ_CANONICAL["VI"])) == 4


def test_minimal_dnf_of_constants() -> None:
    assert minimal_dnf(TruthTable.constant(3, False)) == ()
    assert minimal_dnf(TruthTable.constant(3, True)) == ((None, None, None),)
    assert dnf_literal_count(TruthTable.constant(3, True)) == 0
    assert dnf_literal_count(TruthTable.constant(2, False)) == 0
    assert dnf_truth_table(3, ()) == TruthTable.constant(3, False)
    assert dnf_truth_table(3, ((None, None, None),)) == TruthTable.constant(3, True)


def test_minimal_dnf_reproduces_and_minimizes_every_three_input_function() -> None:
    for table in all_functions(3):
        terms = minimal_dnf(table)
        assert dnf_truth_table(3, terms) == table
        assert dnf_literal_count(table) == _brute_force_minimum_literals(table)
        assert terms == minimal_dnf(table)  # deterministic


@pytest.mark.parametrize("arity", [4, 5])
def test_minimal_dnf_reproduces_random_functions(arity: int) -> None:
    rng = np.random.default_rng(arity)
    for _ in range(150):
        table = TruthTable(arity, tuple(int(b) for b in rng.integers(0, 2, size=2**arity)))
        terms = minimal_dnf(table)
        assert dnf_truth_table(arity, terms) == table
        assert all(len(t) == arity for t in terms)
        assert terms == tuple(sorted(terms, key=implicant_key))
        assert set(terms) <= set(prime_implicants(table))


def test_minimal_dnf_is_minimal_on_random_four_input_functions() -> None:
    rng = np.random.default_rng(2024)
    for _ in range(60):
        table = TruthTable(4, tuple(int(b) for b in rng.integers(0, 2, size=16)))
        assert dnf_literal_count(table) == _brute_force_minimum_literals(table)


def test_minimal_dnf_of_parity_and_and() -> None:
    parity = TruthTable.from_function(6, lambda x: sum(x) & 1)
    assert dnf_literal_count(parity) == 32 * 6
    conjunction = TruthTable.from_function(6, lambda x: int(all(x)))
    assert minimal_dnf(conjunction) == ((1,) * 6,)
