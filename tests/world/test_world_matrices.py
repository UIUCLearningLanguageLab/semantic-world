"""Stage a1 of docs/specs/WORLD_AND_LANGUAGE.md: rules as two-layer threshold matrices
(REL.18), their evaluation in dependency order, constant rules, the sparse record, and the
agreement test."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from semantic_world.common.boolean import SHJ_CANONICAL, TruthTable, all_functions, settings_array
from semantic_world.world import (
    AgreementError,
    Layer,
    LiteralSpec,
    RuleMatrices,
    RuleMatrix,
    RuleSpec,
    Term,
    WorldError,
    build_matrices,
    check_agreement,
    dependency_layers,
    evaluate_by_tables,
)


def feature(label: str) -> LiteralSpec:
    return LiteralSpec(label, {"kind": "feature", "role": None, "feature": label})


LITERALS = tuple(feature(name) for name in ("A", "B", "C", "D", "X", "Y"))


def rule(output: str, inputs: tuple[str, ...], bits: str) -> RuleSpec:
    return RuleSpec(output, inputs, TruthTable.from_bit_string(bits))


def grid(*columns: int) -> np.ndarray:
    """Every setting of the first ``columns`` literals, with every other literal at 0."""
    values = np.zeros((2 ** len(columns), len(LITERALS)), dtype=np.uint8)
    values[:, list(columns)] = settings_array(len(columns))
    return values


# ---------------------------------------------------------------------------------------------
# Terms and layers
# ---------------------------------------------------------------------------------------------


def test_and_rule_is_one_term_with_threshold_two() -> None:
    matrices = build_matrices(LITERALS, [rule("X", ("A", "B"), "0001")])
    (layer,) = matrices.layers
    (x,) = layer.rules
    assert x.terms == (Term((0, 1), (False, False)),)
    assert x.terms[0].threshold == 2
    assert layer.w1.shape == (1, 12) and layer.theta1.tolist() == [2]
    assert layer.w2.tolist() == [[1]]


def test_or_rule_is_one_term_per_literal() -> None:
    matrices = build_matrices(LITERALS, [rule("X", ("A", "B"), "0111")])
    (x,) = matrices.layers[0].rules
    assert x.terms == (Term((0,), (False,)), Term((1,), (False,)))


def test_negated_literals_mark_the_complement_half() -> None:
    matrices = build_matrices(LITERALS, [rule("X", ("A",), "10")])  # NOT A
    layer = matrices.layers[0]
    assert layer.w1.tolist()[0] == [0] * 6 + [1] + [0] * 5
    assert matrices.evaluate(grid(0))["X"].tolist() == [1, 0]


def test_xor_rule_has_two_terms_of_two_literals() -> None:
    matrices = build_matrices(LITERALS, [rule("X", ("A", "B"), "0110")])
    (x,) = matrices.layers[0].rules
    assert len(x.terms) == 2 and all(t.threshold == 2 for t in x.terms)
    assert matrices.evaluate(grid(0, 1))["X"].tolist() == [0, 1, 1, 0]


def test_constant_true_is_one_empty_term_with_threshold_zero() -> None:
    matrices = build_matrices(LITERALS, [rule("X", ("A", "B"), "1111")])
    (x,) = matrices.layers[0].rules
    assert x.is_constant_true and x.terms == (Term((), ()),)
    assert x.terms[0].threshold == 0
    assert matrices.evaluate(grid(0, 1))["X"].tolist() == [1, 1, 1, 1]


def test_constant_false_has_no_terms() -> None:
    matrices = build_matrices(LITERALS, [rule("X", ("A", "B"), "0000")])
    (x,) = matrices.layers[0].rules
    assert x.is_constant_false and x.terms == ()
    assert matrices.evaluate(grid(0, 1))["X"].tolist() == [0, 0, 0, 0]


def test_constant_rules_without_inputs() -> None:
    specs = [
        RuleSpec("X", (), TruthTable.constant(0, True)),
        RuleSpec("Y", (), TruthTable.constant(0, False)),
    ]
    matrices = build_matrices(LITERALS, specs)
    out = matrices.evaluate(matrices.literal_matrix(3))
    assert out["X"].tolist() == [1, 1, 1] and out["Y"].tolist() == [0, 0, 0]
    report = check_agreement(matrices, specs, matrices.literal_matrix(3))
    assert report.exhaustive_rules == 2 and report.settings == 2
    with pytest.raises(WorldError, match="no rule"):
        check_agreement(matrices, specs[:1], matrices.literal_matrix(3))


def test_irrelevant_inputs_do_not_appear_in_terms() -> None:
    # SHJ type I over (A, B, C) depends on A alone: NOT A.
    matrices = build_matrices(LITERALS, [RuleSpec("X", ("A", "B", "C"), SHJ_CANONICAL["I"])])
    (x,) = matrices.layers[0].rules
    assert x.terms == (Term((0,), (True,)),)


@pytest.mark.parametrize("shj_type", sorted(SHJ_CANONICAL))
def test_shj_types_agree_on_every_setting(shj_type: str) -> None:
    table = SHJ_CANONICAL[shj_type]
    spec = RuleSpec("X", ("A", "B", "C"), table)
    matrices = build_matrices(LITERALS, [spec])
    assert matrices.evaluate(grid(0, 1, 2))["X"].tolist() == list(table.bits)
    check_agreement(matrices, [spec], grid(0, 1, 2))


def test_every_three_input_function_agrees_on_every_setting() -> None:
    specs = [RuleSpec(f"F{i}", ("A", "B", "C"), table) for i, table in enumerate(all_functions(3))]
    literals = LITERALS + tuple(feature(f"F{i}") for i in range(256))
    matrices = build_matrices(literals, specs)
    values = np.zeros((8, len(literals)), dtype=np.uint8)
    values[:, :3] = settings_array(3)
    out = matrices.evaluate(values)
    for spec in specs:
        assert out[spec.output].tolist() == list(spec.table.bits), spec.output
    report = check_agreement(matrices, specs, values)
    assert report.rules == 256 and report.exhaustive_rules == 256 and report.settings == 2048


# ---------------------------------------------------------------------------------------------
# Dependency layers
# ---------------------------------------------------------------------------------------------


def chain_rules() -> list[RuleSpec]:
    return [
        rule("X", ("A", "B"), "0001"),  # X = A AND B
        rule("Y", ("X", "C"), "0111"),  # Y = X OR C
        rule("D", ("Y", "A"), "0110"),  # D = Y XOR A
    ]


def test_rules_are_grouped_in_dependency_order() -> None:
    assert dependency_layers(chain_rules()) == {"X": 1, "Y": 2, "D": 3}
    matrices = build_matrices(LITERALS, chain_rules())
    assert [layer.index for layer in matrices.layers] == [1, 2, 3]
    assert matrices.outputs == ("X", "Y", "D")


def test_dependency_layers_ignore_the_given_order() -> None:
    reordered = list(reversed(chain_rules()))
    matrices = build_matrices(LITERALS, reordered)
    assert matrices.outputs == ("X", "Y", "D")


def test_chain_evaluates_derived_literals_before_the_rules_that_read_them() -> None:
    matrices = build_matrices(LITERALS, chain_rules())
    values = grid(0, 1, 2)
    out = matrices.evaluate(values)
    a, b, c = (values[:, k].astype(int) for k in range(3))
    x = a & b
    y = x | c
    assert out["X"].tolist() == x.tolist()
    assert out["Y"].tolist() == y.tolist()
    assert out["D"].tolist() == (y ^ a).tolist()
    by_tables = evaluate_by_tables(matrices, chain_rules(), values)
    assert all(np.array_equal(by_tables[k], out[k]) for k in out)
    check_agreement(matrices, chain_rules(), values)


def test_a_cycle_is_an_error() -> None:
    with pytest.raises(WorldError, match="cycle"):
        build_matrices(LITERALS, [rule("X", ("Y",), "01"), rule("Y", ("X",), "01")])


def test_an_unknown_input_is_an_error() -> None:
    with pytest.raises(WorldError, match="not a literal"):
        build_matrices(LITERALS, [rule("X", ("A", "Q"), "0001")])


def test_two_rules_for_one_output_is_an_error() -> None:
    with pytest.raises(WorldError, match="same output"):
        build_matrices(LITERALS, [rule("X", ("A",), "01"), rule("X", ("B",), "01")])


def test_arity_must_match_the_inputs() -> None:
    with pytest.raises(WorldError, match="arity"):
        RuleSpec("X", ("A",), TruthTable.from_bit_string("0001"))


# ---------------------------------------------------------------------------------------------
# The sparse record
# ---------------------------------------------------------------------------------------------


def test_record_round_trips_through_from_record() -> None:
    specs = chain_rules() + [rule("Y2", ("A", "B"), "1111"), rule("X2", ("C",), "00")]
    literals = LITERALS + (feature("Y2"), feature("X2"))
    matrices = build_matrices(literals, specs)
    rebuilt = RuleMatrices.from_record(matrices.literals_record(), matrices.layers_record())
    assert rebuilt.literals == matrices.literals
    assert rebuilt.layers == matrices.layers
    values = np.zeros((8, len(literals)), dtype=np.uint8)
    values[:, :3] = settings_array(3)
    original = matrices.evaluate(values)
    again = rebuilt.evaluate(values)
    assert all(np.array_equal(original[k], again[k]) for k in original)


def test_record_lists_terms_per_layer_and_outputs_by_term_index() -> None:
    matrices = build_matrices(LITERALS, chain_rules())
    first = matrices.layers_record()[0]
    assert first == {
        "layer": 1,
        "terms": [{"literals": [0, 1], "complemented": [False, False], "threshold": 2}],
        "outputs": [{"output": "X", "terms": [0], "threshold": 1}],
    }
    literals = matrices.literals_record()
    assert literals[0] == {"index": 0, "key": "A", "kind": "feature", "role": None, "feature": "A"}


def test_from_record_checks_indices_and_thresholds() -> None:
    matrices = build_matrices(LITERALS, chain_rules())
    literals = matrices.literals_record()
    layers = matrices.layers_record()
    bad = [dict(layers[0], terms=[dict(layers[0]["terms"][0], threshold=1)])]
    with pytest.raises(WorldError, match="threshold"):
        RuleMatrices.from_record(literals, bad)
    with pytest.raises(WorldError, match="index"):
        RuleMatrices.from_record([dict(literals[0], index=5)] + literals[1:], layers)


# ---------------------------------------------------------------------------------------------
# The agreement test
# ---------------------------------------------------------------------------------------------


def test_agreement_report_counts_exhaustive_rules() -> None:
    specs = chain_rules() + [RuleSpec("Y2", tuple("ABCDXY"), TruthTable.constant(6, True))]
    literals = LITERALS + (feature("Y2"),)
    matrices = build_matrices(literals, specs)
    values = np.zeros((5, len(literals)), dtype=np.uint8)
    report = check_agreement(matrices, specs, values, max_exhaustive_inputs=3)
    assert report.rules == 4 and report.entities == 5
    assert report.exhaustive_rules == 3 and report.settings == 12


def test_a_disagreement_on_an_entity_fails_and_names_the_rule() -> None:
    specs = chain_rules()
    matrices = build_matrices(LITERALS, specs)
    values = grid(0, 1, 2)
    wrong = {k: v.copy() for k, v in matrices.evaluate(values).items()}
    wrong["Y"][3] ^= 1
    with pytest.raises(AgreementError, match=r"rule for Y disagrees .* on 1 of 8 entities"):
        check_agreement(matrices, specs, values, wrong)


def test_a_disagreement_on_an_input_setting_fails_and_names_the_setting() -> None:
    specs = [rule("X", ("A", "B"), "0001")]
    good = build_matrices(LITERALS, specs)
    # A corrupted matrix form: X = A OR B instead of A AND B.
    broken = RuleMatrices(
        LITERALS,
        (
            Layer(
                1, (RuleMatrix("X", (Term((0,), (False,)), Term((1,), (False,)))),), len(LITERALS)
            ),
        ),
    )
    values = np.zeros((2, len(LITERALS)), dtype=np.uint8)  # A = B = 0 on every entity: no clue
    assert np.array_equal(good.evaluate(values)["X"], broken.evaluate(values)["X"])
    with pytest.raises(AgreementError, match=r"2 of 4 input settings \(first: 01\)"):
        check_agreement(broken, specs, values)


def test_evaluate_rule_uses_the_rules_own_inputs_only() -> None:
    matrices = build_matrices(LITERALS, chain_rules())
    settings = settings_array(2)
    # Y = X OR C over (X, C), with every other literal at 0.
    assert matrices.evaluate_rule("Y", ("X", "C"), settings).tolist() == [0, 1, 1, 1]


def test_literal_values_must_have_the_tables_width() -> None:
    matrices = build_matrices(LITERALS, chain_rules())
    with pytest.raises(WorldError, match="shape"):
        matrices.evaluate(np.zeros((2, 3), dtype=np.uint8))


def test_random_rule_sets_agree_with_table_evaluation() -> None:
    rng = np.random.default_rng(1)
    names = [f"L{i}" for i in range(8)]
    literals = tuple(feature(n) for n in names)
    for _ in range(20):
        specs = []
        for k in range(3, 8):  # L3..L7 read earlier literals only: acyclic by construction
            arity = int(rng.integers(0, 4))
            inputs = tuple(rng.choice(names[:k], size=arity, replace=False).tolist())
            bits = tuple(int(b) for b in rng.integers(0, 2, size=2**arity))
            specs.append(RuleSpec(names[k], inputs, TruthTable(arity, bits)))
        matrices = build_matrices(literals, specs)
        values = np.zeros((50, 8), dtype=np.uint8)
        values[:, :3] = rng.integers(0, 2, size=(50, 3))
        check_agreement(matrices, specs, values)
        for a, b in itertools.combinations(matrices.outputs, 2):
            assert a != b
