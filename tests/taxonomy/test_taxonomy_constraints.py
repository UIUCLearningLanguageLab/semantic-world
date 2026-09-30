"""Stage 10 acceptance tests: constraints, relations, base relations, and pair evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from semantic_world.taxonomy import (
    GenerationError,
    Streams,
    TaxonomyResult,
    config_from_mapping,
    generate,
    load_config,
)
from semantic_world.taxonomy.constraints import (
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
    literal_role,
)
from semantic_world.taxonomy.expressions import (
    Cmp,
    ExpressionError,
    Gt,
    Not,
    Op,
    Var,
    parse_expression,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"


@pytest.fixture(scope="module")
def tiny() -> TaxonomyResult:
    return generate(load_config(DATA / "tiny_relations.yaml"))


@pytest.fixture(scope="module")
def relations_run() -> TaxonomyResult:
    return generate(load_config(DATA / "relations.yaml"))


# ---------------------------------------------------------------------------------------------
# Comparison expressions
# ---------------------------------------------------------------------------------------------


def test_comparisons_parse_and_print() -> None:
    order = parse_expression("a.SC.1 - p.SC.2 > 0.15")
    assert order == Cmp("a.SC.1", "p.SC.2", 0.15)
    assert str(order) == "a.SC.1 - p.SC.2 > 0.1500"
    window = parse_expression("0.1500 < a.SC.1 - p.SC.1 < 1.2200")
    assert window == Cmp("a.SC.1", "p.SC.1", 0.15, 1.22)
    assert str(window) == "0.1500 < a.SC.1 - p.SC.1 < 1.2200"
    negative = parse_expression("a.SC.1 - p.SC.1 > -0.5")
    assert negative.low == -0.5
    mixed = parse_expression("(a.HAS.4 AND p.IS.7) OR (a.HAS.9 AND p.IS.2)")
    assert mixed.variables() == ("a.HAS.4", "p.IS.7", "a.HAS.9", "p.IS.2")
    combined = parse_expression("p.SC.1 <= 0.2031 AND 0.15 < a.SC.1 - p.SC.1 < 1.22 AND a.IS.3")
    assert combined == Op(
        "AND", (Not(Gt("p.SC.1", 0.2031)), Cmp("a.SC.1", "p.SC.1", 0.15, 1.22), Var("a.IS.3"))
    )
    assert str(combined) == "p.SC.1 <= 0.2031 AND (0.1500 < a.SC.1 - p.SC.1 < 1.2200) AND a.IS.3"
    assert parse_expression(str(combined)) == combined
    assert str(Not(order)) == "NOT (a.SC.1 - p.SC.1 > 0.1500)".replace("p.SC.1", "p.SC.2")
    assert parse_expression(str(Not(order))) == Not(order)
    assert combined.variables()[1] == "0.1500<a.SC.1-p.SC.1<1.2200"


@pytest.mark.parametrize(
    "text",
    [
        "a.SC.1 - p.SC.1",
        "a.SC.1 - > 0.5",
        "0.5 < a.SC.1 - p.SC.1",
        "0.5 < a.SC.1 > 0.7",
        "1.0 < a.SC.1 - p.SC.1 < 0.5",
        "a.SC.1 - p.SC.1 < 0.5",
    ],
)
def test_malformed_comparisons_are_rejected(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(text)


def test_comparison_truth_table_and_evaluation() -> None:
    expr = parse_expression("a.IS.1 AND (a.SC.1 - p.SC.1 > 0.5)")
    keys = ["a.IS.1", "a.SC.1-p.SC.1>0.5000"]
    assert expr.truth_table(keys).bit_string() == "0001"
    env = {"a.IS.1": np.array([1, 1, 0]), "a.SC.1-p.SC.1>0.5000": np.array([True, False, True])}
    assert expr.evaluate(env).tolist() == [True, False, False]


# ---------------------------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------------------------


def _brute_force_matrix(result: TaxonomyResult, relation) -> np.ndarray:
    """Evaluate a relation's constraint expressions over every ordered pair of instances by
    building the environment of every atom directly from the two instances' features."""
    instances = result.instances
    features = result.features
    n = len(instances)
    agents, patients = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    agents, patients = agents.ravel(), patients.ravel()
    held = agents != patients

    def value_of(role: str, label: str) -> np.ndarray:
        rows = agents if role == "a" else patients
        if label.startswith("SC."):
            return instances.scalars[rows, int(label[3:]) - 1]
        return instances.values[rows, features[label].position]

    for constraint in relation.constraints:
        env = {}
        for atom in constraint.expression.atoms():
            if isinstance(atom, Var):
                role, label = atom.name.split(".", 1)
                env[atom.name] = value_of(role, label).astype(bool)
            elif isinstance(atom, Gt):
                role, label = atom.scalar.split(".", 1)
                env[atom.key] = value_of(role, label) > atom.threshold
            else:
                assert isinstance(atom, Cmp)
                difference = value_of("a", atom.agent[2:]) - value_of("p", atom.patient[2:])
                if atom.high is None:
                    env[atom.key] = difference > atom.low
                else:
                    env[atom.key] = (difference > atom.low) & (difference < atom.high)
        held &= np.broadcast_to(constraint.expression.evaluate(env), (n * n,))
    return held.reshape(n, n)


def test_tiny_relations_configuration() -> None:
    config = load_config(DATA / "tiny_relations.yaml")
    assert config.scalars.count == 1
    assert config.verbs is not None and config.verbs.feature_count == 4
    result = generate(config)
    assert len(result.verbs.verbs) == 4
    assert len(result.instances) == 12
    assert result.relations is not None
    labels = [c.label for c in result.relations.constraints]
    assert labels[:4] == ["K.VF.1", "K.VF.2", "K.VF.3", "K.VF.4"]
    assert labels[4:] == [f"K.{v.label}" for v in result.verbs.verbs]


def test_every_relation_matches_brute_force_over_all_pairs(tiny: TaxonomyResult) -> None:
    relations = tiny.relations
    n = len(tiny.instances)
    for category in tiny.verbs.categories:
        relation = relations.relation(category)
        matrix = relations.matrix(category)
        assert matrix.shape == (n, n)
        assert np.array_equal(matrix, _brute_force_matrix(tiny, relation)), category.label
        agents, patients = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        held = relations.holds(category, agents.ravel(), patients.ravel()).reshape(n, n)
        assert np.array_equal(held, matrix)
    # Chunking does not change the result.
    verb = tiny.verbs.verbs[0]
    assert np.array_equal(relations.matrix(verb, chunk_rows=5), relations.matrix(verb))


def test_brute_force_on_the_relations_example_for_a_few_verbs(
    relations_run: TaxonomyResult,
) -> None:
    relations = relations_run.relations
    for verb in relations_run.verbs.verbs[:3]:
        assert np.array_equal(
            relations.matrix(verb), _brute_force_matrix(relations_run, relations.relation(verb))
        )


def test_no_instance_is_related_to_itself(
    tiny: TaxonomyResult, relations_run: TaxonomyResult
) -> None:
    for result in (tiny, relations_run):
        n = len(result.instances)
        for verb in result.verbs.verbs:
            assert not np.diag(result.relations.matrix(verb)).any()
        idx = np.arange(n)
        assert not result.relations.holds(result.verbs.verbs[0], idx, idx).any()


def test_relations_imply_their_ancestors_base_relations(
    tiny: TaxonomyResult, relations_run: TaxonomyResult
) -> None:
    for result in (tiny, relations_run):
        relations = result.relations
        checked = 0
        for verb in result.verbs.verbs:
            own = relations.matrix(verb)
            own_relation = relations.relation(verb)
            assert not own_relation.base
            for ancestor in verb.ancestors():
                base = relations.relation(ancestor)
                assert base.base
                assert set(c.label for c in base.constraints) <= set(
                    c.label for c in own_relation.constraints
                )
                assert np.all(own <= relations.matrix(ancestor)), (verb.label, ancestor.label)
                checked += 1
        assert checked > 0
    # Base relations are the defining structure: constraints of features defining with value 1.
    for category in relations_run.verbs.categories:
        if not category.is_leaf:
            expected = [
                f"K.VF.{i + 1}"
                for i in np.flatnonzero(category.defining_mask() & (category.free_values == 1))
            ]
            assert [
                c.label for c in relations_run.relations.relation(category).constraints
            ] == expected


def test_verb_relations_include_true_features_and_the_own_constraint(
    relations_run: TaxonomyResult,
) -> None:
    relations = relations_run.relations
    for verb in relations_run.verbs.verbs:
        relation = relations.relation(verb)
        labels = [c.label for c in relation.constraints]
        expected = [f"K.VF.{i + 1}" for i in np.flatnonzero(verb.values == 1)] + [f"K.{verb.label}"]
        assert labels == expected
        assert relation.expression.count(" AND ") >= len(labels) - 1
    without_own = generate(
        config_from_mapping({"scalars": {"count": 2}, "verbs": {"own_constraint": False}})
    )
    for verb in without_own.verbs.verbs:
        assert all(
            c.label.startswith("K.VF.") for c in without_own.relations.relation(verb).constraints
        )
    assert len(without_own.relations.constraints) == 12


# ---------------------------------------------------------------------------------------------
# Constraint families
# ---------------------------------------------------------------------------------------------


_CACHE: dict[int, TaxonomyResult] = {}


def _all_constraints(seeds=range(4)):
    for seed in seeds:
        if seed not in _CACHE:
            _CACHE[seed] = generate(
                config_from_mapping(
                    {
                        "scalars": {"count": 2},
                        "verbs": {"features": {"count": 20, "expected_true": 5}},
                    },
                    seed=seed,
                )
            )
        yield from _CACHE[seed].relations.constraints


def test_constraint_expressions_parse_back_and_records_are_consistent() -> None:
    families = set()
    for constraint in _all_constraints():
        families.add(constraint.family)
        assert (
            parse_expression(str(constraint.expression)).truth_table(constraint.keys)
            == constraint.table
        )
        assert constraint.table.arity == constraint.arity == len(constraint.keys)
        record = constraint.record()
        assert list(record) == [
            "label",
            "family",
            "agent_literals",
            "patient_literals",
            "comparisons",
            "expression",
            "truth_table",
            "min_dnf_literals",
        ]
        assert (
            len(record["agent_literals"])
            + len(record["patient_literals"])
            + len(record["comparisons"])
            == constraint.arity
        )
        for item in constraint.literals:
            if isinstance(item, RoleFeature):
                assert item.feature.type in ("is", "has")
            elif isinstance(item, RoleThreshold):
                assert item.threshold.threshold == round(item.threshold.threshold, 4)
    assert families == {"agent", "patient", "cross", "key_lock", "comparison"}


def test_agent_and_patient_constraints_read_one_role() -> None:
    for constraint in _all_constraints():
        roles = {literal_role(item) for item in constraint.literals}
        if constraint.family == "agent":
            assert roles == {"a"}
        elif constraint.family == "patient":
            assert roles == {"p"}


def test_cross_and_key_lock_constraints_depend_on_both_roles() -> None:
    seen = {"cross": 0, "key_lock": 0}
    for constraint in _all_constraints(range(6)):
        if constraint.family in seen:
            seen[constraint.family] += 1
            assert constraint.depends_on_both_roles(), str(constraint.expression)
            relevant = {
                literal_role(constraint.literals[i]) for i in constraint.table.relevant_inputs()
            }
            assert {"a", "p"} <= relevant
    assert seen["cross"] > 5 and seen["key_lock"] > 5


def test_key_lock_structure() -> None:
    found = False
    for constraint in _all_constraints(range(6)):
        if constraint.family != "key_lock":
            continue
        found = True
        pairs = constraint.arity // 2
        assert pairs in (1, 2, 3) and constraint.arity == 2 * pairs
        roles = [literal_role(item) for item in constraint.literals]
        assert roles == ["a", "p"] * pairs
        expr = constraint.expression
        terms = list(expr.operands) if isinstance(expr, Op) and expr.operator == "OR" else [expr]
        assert len(terms) == pairs
        for term in terms:
            assert isinstance(term, Op) and term.operator == "AND" and len(term.operands) == 2
        # At most one literal per scalar per role.
        for role in ("a", "p"):
            scalars = [
                item.threshold.scalar
                for item in constraint.literals
                if isinstance(item, RoleThreshold) and item.role == role
            ]
            assert len(set(scalars)) == len(scalars)
    assert found


def test_comparisons_agree_with_direct_arithmetic(
    tiny: TaxonomyResult, relations_run: TaxonomyResult
) -> None:
    checked = 0
    for result in (tiny, relations_run):
        scalars = result.instances.scalars
        n = len(result.instances)
        for constraint in result.relations.constraints:
            for comparison in constraint.comparisons:
                checked += 1
                difference = (
                    scalars[:, comparison.agent_scalar - 1][:, None]
                    - scalars[:, comparison.patient_scalar - 1][None, :]
                )
                expected = difference > comparison.low
                if comparison.high is not None:
                    expected &= difference < comparison.high
                    assert comparison.low < comparison.high
                assert np.array_equal(
                    constraint.matrix(result.instances.values, scalars, np.arange(n)), expected
                )
                assert constraint.family == "comparison" and constraint.arity == 1
                assert constraint.table.bit_string() == "01"
    assert checked > 0


def test_comparison_margins_and_dimensions() -> None:
    from statistics import NormalDist

    config = config_from_mapping(
        {
            "scalars": {"count": 3},
            "verbs": {
                "features": {"count": 40, "expected_true": 10},
                "constraint_families": {
                    "agent": 0,
                    "patient": 0,
                    "cross": 0,
                    "key_lock": 0,
                    "comparison": 1,
                },
            },
        }
    )
    result = generate(config)
    sigma = math.sqrt(2) * config.scalars.model_std
    low = NormalDist(0, sigma).inv_cdf(0.1)
    high = NormalDist(0, sigma).inv_cdf(0.9)
    windows = cross = 0
    comparisons = [c.comparisons[0] for c in result.relations.constraints]
    for comparison in comparisons:
        assert low - 1e-4 <= comparison.low <= high + 1e-4
        if comparison.high is not None:
            windows += 1
            assert comparison.low < comparison.high <= high + 1e-4
        if comparison.agent_scalar != comparison.patient_scalar:
            cross += 1
    assert 0 < windows < len(comparisons)
    assert 0 < cross < len(comparisons)
    single = generate(
        config_from_mapping(
            {
                "scalars": {"count": 1},
                "verbs": {
                    "constraint_families": {
                        "agent": 0,
                        "patient": 0,
                        "cross": 0,
                        "key_lock": 0,
                        "comparison": 1,
                    }
                },
            }
        )
    )
    assert all(
        c.comparisons[0].agent_scalar == c.comparisons[0].patient_scalar == 1
        for c in single.relations.constraints
    )


def test_no_comparisons_without_scalars() -> None:
    result = generate(config_from_mapping({"verbs": {}}))
    assert result.relations is not None
    assert all(c.family != "comparison" for c in result.relations.constraints)
    assert all(
        not isinstance(item, RoleThreshold | ScalarComparison)
        for c in result.relations.constraints
        for item in c.literals
    )


def test_constraints_never_read_isa_can_or_projections() -> None:
    for constraint in _all_constraints():
        for key in constraint.keys:
            assert not any(part in key for part in ("ISA.", "CAN.", "CANBE."))


def test_cross_constraints_need_arity_two() -> None:
    config = config_from_mapping(
        {
            "verbs": {
                "rules": {"arity": {1: 1}},
                "constraint_families": {
                    "agent": 0,
                    "patient": 0,
                    "cross": 1,
                    "key_lock": 0,
                    "comparison": 0,
                },
            }
        }
    )
    with pytest.raises(GenerationError, match="arity of at least 2"):
        generate(config)


# ---------------------------------------------------------------------------------------------
# Determinism and independence
# ---------------------------------------------------------------------------------------------


def test_constraints_are_deterministic_and_use_their_stream() -> None:
    overrides = {"scalars": {"count": 2}, "verbs": {}}
    a = generate(config_from_mapping(overrides, seed=3)).relations.records()
    b = generate(config_from_mapping(overrides, seed=3)).relations.records()
    c = generate(config_from_mapping(overrides, seed=4)).relations.records()
    assert a == b and a != c
    more_instances = generate(
        config_from_mapping({**overrides, "instances": {"per_leaf": 12}}, seed=3)
    )
    assert more_instances.relations.records() == a
    assert (
        more_instances.relations.relation_records()
        == generate(config_from_mapping(overrides, seed=3)).relations.relation_records()
    )


def test_constraints_leave_noun_outputs_and_other_streams_alone() -> None:
    from semantic_world.taxonomy import (
        generate_constraints,
        generate_instances,
        generate_rules,
        generate_tree,
        generate_verb_tree,
    )

    config = config_from_mapping({"scalars": {"count": 2}, "verbs": {}})
    used = Streams(1)
    rules = generate_rules(config, used)
    tree = generate_tree(config, rules, used)
    instances = generate_instances(config, rules, tree, used)
    verbs = generate_verb_tree(config, used)
    before = {name: getattr(Streams(1), name) for name in ("pairs",)}
    generate_constraints(config, rules.features, verbs, instances, used)
    assert used.constraints.random() != Streams(1).constraints.random()
    assert used.pairs.random() == before["pairs"].random()
    off = generate(config_from_mapping({"scalars": {"count": 2}}))
    on = generate(config_from_mapping({"scalars": {"count": 2}, "verbs": {}}))
    assert np.array_equal(off.instances.values, on.instances.values)
    assert off.rules.records() == on.rules.records()


def test_example_configurations_run(tmp_path: Path) -> None:
    for name in ("relations", "tiny_relations"):
        result = generate(load_config(DATA / f"{name}.yaml"))
        assert result.relations is not None
        result.write(tmp_path / name)
