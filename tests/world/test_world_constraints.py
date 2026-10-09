"""Constraints and relations of the world package (stage a5b of
``docs/specs/WORLD_AND_LANGUAGE.md``): comparison expressions, constraint families, the relation
of every two-place event type, base relations, pair evaluation against brute force, explicit
requirements, and determinism. These tests moved from
``tests/taxonomy/test_taxonomy_constraints.py`` (stage 10 of the taxonomy generator).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic_world.taxonomy import GenerationError
from semantic_world.taxonomy.expressions import (
    Cmp,
    ExpressionError,
    Gt,
    Not,
    Op,
    Var,
    parse_expression,
)
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.taxonomy.tree import Role
from semantic_world.world.config import Config, config_from_mapping, load_config
from semantic_world.world.constraints import (
    Relations,
    RoleFeature,
    RoleThreshold,
    ScalarComparison,
    constraint_label,
    generate_constraints,
    generate_relations,
    literal_role,
)
from semantic_world.world.event_file import load_event_file
from semantic_world.world.event_tree import generate_event_tree
from semantic_world.world.generate import define
from semantic_world.world.statics import StaticWorld, build_statics
from semantic_world.world.streams import WorldStreams

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "world"
TAXONOMY_DATA = REPO / "data" / "taxonomy"
RELATIONS = str(TAXONOMY_DATA / "relations.yaml")


def world_config(
    taxonomy: str | dict = RELATIONS,
    seed: int = 1,
    tmp_path: Path | None = None,
    **blocks,
) -> Config:
    """A world configuration over a taxonomy file, or over a taxonomy mapping written to
    ``tmp_path``."""
    if isinstance(taxonomy, dict):
        assert tmp_path is not None
        path = tmp_path / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(taxonomy, sort_keys=False), encoding="utf-8")
        taxonomy = str(path)
    data = {"name": "test", "seed": seed, "taxonomy": {"config": taxonomy}, **blocks}
    return config_from_mapping(data, source="<test>")


def statics_of(config: Config) -> StaticWorld:
    """The static side of a world: the taxonomy run and the requirements, without the
    dynamics."""
    taxonomy = generate_taxonomy(config.taxonomy_config())
    event_file = None
    if config.event_file_path() is not None:
        event_file = load_event_file(config.event_file_path())
    return build_statics(taxonomy, config, WorldStreams(config.seed), event_file)


def relations_of(config: Config) -> tuple[TaxonomyResult, Relations]:
    """The taxonomy run and the relations alone (the event tree and the constraints)."""
    taxonomy = generate_taxonomy(config.taxonomy_config())
    streams = WorldStreams(config.seed)
    event_tree, relations = generate_relations(
        config.event_types.binary,
        taxonomy.config.scalars,
        taxonomy.rules,
        taxonomy.tree,
        taxonomy.instances,
        streams.constraints,
        streams.event_tree,
    )
    assert event_tree is not None and relations is not None
    return taxonomy, relations


@pytest.fixture(scope="module")
def tiny() -> StaticWorld:
    return define(load_config(DATA / "tiny.yaml")).statics


@pytest.fixture(scope="module")
def default() -> StaticWorld:
    return statics_of(load_config(DATA / "default.yaml"))


# ---------------------------------------------------------------------------------------------
# Comparison expressions
# ---------------------------------------------------------------------------------------------


def test_comparisons_parse_and_print() -> None:
    order = parse_expression("agent.SCALARDIM.1 - patient.SCALARDIM.2 > 0.15")
    assert order == Cmp("agent.SCALARDIM.1", "patient.SCALARDIM.2", 0.15)
    assert str(order) == "agent.SCALARDIM.1 - patient.SCALARDIM.2 > 0.1500"
    window = parse_expression("0.1500 < agent.SCALARDIM.1 - patient.SCALARDIM.1 < 1.2200")
    assert window == Cmp("agent.SCALARDIM.1", "patient.SCALARDIM.1", 0.15, 1.22)
    assert str(window) == "0.1500 < agent.SCALARDIM.1 - patient.SCALARDIM.1 < 1.2200"
    negative = parse_expression("agent.SCALARDIM.1 - patient.SCALARDIM.1 > -0.5")
    assert negative.low == -0.5
    mixed = parse_expression(
        "(agent.PART.4 AND patient.PROPERTY.7) OR (agent.PART.9 AND patient.PROPERTY.2)"
    )
    assert mixed.variables() == (
        "agent.PART.4",
        "patient.PROPERTY.7",
        "agent.PART.9",
        "patient.PROPERTY.2",
    )
    combined = parse_expression(
        "patient.SCALARDIM.1 <= 0.2031 AND 0.15 < agent.SCALARDIM.1 - patient.SCALARDIM.1 < 1.22 "
        "AND agent.PROPERTY.3"
    )
    assert combined == Op(
        "AND",
        (
            Not(Gt("patient.SCALARDIM.1", 0.2031)),
            Cmp("agent.SCALARDIM.1", "patient.SCALARDIM.1", 0.15, 1.22),
            Var("agent.PROPERTY.3"),
        ),
    )
    assert str(combined) == (
        "patient.SCALARDIM.1 <= 0.2031 AND "
        "(0.1500 < agent.SCALARDIM.1 - patient.SCALARDIM.1 < 1.2200) AND agent.PROPERTY.3"
    )
    assert parse_expression(str(combined)) == combined
    assert str(Not(order)) == "NOT (agent.SCALARDIM.1 - patient.SCALARDIM.2 > 0.1500)"
    assert parse_expression(str(Not(order))) == Not(order)
    assert combined.variables()[1] == "0.1500<agent.SCALARDIM.1-patient.SCALARDIM.1<1.2200"
    assert order.key == "agent.SCALARDIM.1-patient.SCALARDIM.2>0.1500"


@pytest.mark.parametrize(
    "text",
    [
        "agent.SCALARDIM.1 - patient.SCALARDIM.1",
        "agent.SCALARDIM.1 - > 0.5",
        "0.5 < agent.SCALARDIM.1 - patient.SCALARDIM.1",
        "0.5 < agent.SCALARDIM.1 > 0.7",
        "1.0 < agent.SCALARDIM.1 - patient.SCALARDIM.1 < 0.5",
        "agent.SCALARDIM.1 - patient.SCALARDIM.1 < 0.5",
    ],
)
def test_malformed_comparisons_are_rejected(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(text)


def test_comparison_truth_table_and_evaluation() -> None:
    expr = parse_expression("agent.PROPERTY.1 AND (agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.5)")
    keys = ["agent.PROPERTY.1", "agent.SCALARDIM.1-patient.SCALARDIM.1>0.5000"]
    assert expr.truth_table(keys).bit_string() == "0001"
    env = {keys[0]: np.array([1, 1, 0]), keys[1]: np.array([True, False, True])}
    assert expr.evaluate(env).tolist() == [True, False, False]


# ---------------------------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------------------------


def _brute_force_matrix(taxonomy: TaxonomyResult, relation) -> np.ndarray:
    """Evaluate a relation's constraint expressions over every ordered pair of instances by
    building the environment of every atom directly from the two instances' features."""
    instances = taxonomy.instances
    features = taxonomy.features
    n = len(instances)
    agents, patients = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    agents, patients = agents.ravel(), patients.ravel()
    held = agents != patients

    def value_of(role: str, label: str) -> np.ndarray:
        rows = agents if role == "agent" else patients
        if label.startswith("SCALARDIM."):
            return instances.scalars[rows, int(label[len("SCALARDIM.") :]) - 1]
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
                difference = value_of("agent", atom.agent[len("agent.") :]) - value_of(
                    "patient", atom.patient[len("patient.") :]
                )
                if atom.high is None:
                    env[atom.key] = difference > atom.low
                else:
                    env[atom.key] = (difference > atom.low) & (difference < atom.high)
        held &= np.broadcast_to(constraint.expression.evaluate(env), (n * n,))
    return held.reshape(n, n)


def test_tiny_world_configuration(tiny: StaticWorld) -> None:
    config = load_config(DATA / "tiny.yaml")
    assert config.taxonomy_config().scalars.count == 1
    assert config.event_types.binary is not None
    assert config.event_types.binary.feature_count == 4
    assert len(tiny.event_tree.event_types) == 4
    assert len(tiny.instances) == 12
    relations = tiny.relations
    assert relations is not None and relations.event_tree is tiny.event_tree
    labels = [c.label for c in relations.constraints]
    assert labels[:4] == [f"CONSTRAINT.EVENTFEAT.{i}" for i in range(1, 5)]
    assert labels[4:] == [constraint_label(v.label) for v in tiny.event_tree.event_types]
    assert labels[4:] == [
        "CONSTRAINT.EVENTTYPE2.1.1",
        "CONSTRAINT.EVENTTYPE2.1.2",
        "CONSTRAINT.EVENTTYPE2.2.1",
        "CONSTRAINT.EVENTTYPE2.2.2",
    ]
    assert relations.explicit == frozenset()


def test_every_relation_matches_brute_force_over_all_pairs(tiny: StaticWorld) -> None:
    relations = tiny.relations
    n = len(tiny.instances)
    for category in tiny.event_tree.categories:
        relation = relations.relation(category)
        matrix = relations.matrix(category)
        assert matrix.shape == (n, n)
        assert np.array_equal(matrix, _brute_force_matrix(tiny.taxonomy, relation)), category.label
        agents, patients = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        held = relations.holds(category, agents.ravel(), patients.ravel()).reshape(n, n)
        assert np.array_equal(held, matrix)
        assert np.array_equal(relations.matrix(category.label), matrix)
    # Chunking does not change the result.
    event_type = tiny.event_tree.event_types[0]
    assert np.array_equal(relations.matrix(event_type, chunk_rows=5), relations.matrix(event_type))


def test_brute_force_on_the_default_world_for_a_few_event_types(default: StaticWorld) -> None:
    relations = default.relations
    for event_type in default.event_tree.event_types[:3]:
        assert np.array_equal(
            relations.matrix(event_type),
            _brute_force_matrix(default.taxonomy, relations.relation(event_type)),
        )


def test_no_instance_is_related_to_itself(tiny: StaticWorld, default: StaticWorld) -> None:
    for statics in (tiny, default):
        n = len(statics.instances)
        for event_type in statics.event_tree.event_types:
            assert not np.diag(statics.relations.matrix(event_type)).any()
        idx = np.arange(n)
        assert not statics.relations.holds(statics.event_tree.event_types[0], idx, idx).any()


def test_relations_imply_their_ancestors_base_relations(
    tiny: StaticWorld, default: StaticWorld
) -> None:
    for statics in (tiny, default):
        relations = statics.relations
        checked = 0
        for event_type in statics.event_tree.event_types:
            own = relations.matrix(event_type)
            own_relation = relations.relation(event_type)
            assert not own_relation.base
            for ancestor in event_type.ancestors():
                base = relations.relation(ancestor)
                assert base.base
                assert {c.label for c in base.constraints} <= {
                    c.label for c in own_relation.constraints
                }
                assert np.all(own <= relations.matrix(ancestor)), (event_type.label, ancestor.label)
                checked += 1
        assert checked > 0
    # Base relations are the defining structure: constraints of features defining with value 1.
    for category in default.event_tree.categories:
        if not category.is_leaf:
            expected = [
                f"CONSTRAINT.EVENTFEAT.{i + 1}"
                for i in np.flatnonzero(category.defining_mask() & (category.free_values == 1))
            ]
            assert [c.label for c in default.relations.relation(category).constraints] == expected


def test_relations_include_true_features_and_the_own_constraint(default: StaticWorld) -> None:
    relations = default.relations
    for event_type in default.event_tree.event_types:
        relation = relations.relation(event_type)
        labels = [c.label for c in relation.constraints]
        expected = [
            f"CONSTRAINT.EVENTFEAT.{i + 1}" for i in np.flatnonzero(event_type.values == 1)
        ] + [constraint_label(event_type.label)]
        assert labels == expected
        assert relation.expression.count(" AND ") >= len(labels) - 1
    _, without_own = relations_of(world_config(event_types={"binary": {"own_constraint": False}}))
    for event_type in without_own.event_tree.event_types:
        assert all(
            c.label.startswith("CONSTRAINT.EVENTFEAT.")
            for c in without_own.relation(event_type).constraints
        )
    assert len(without_own.constraints) == 12
    assert without_own.own_constraints == {}


def test_relation_records_name_the_constraints(tiny: StaticWorld) -> None:
    relations = tiny.relations
    records = relations.relation_records()
    assert [r["label"] for r in records] == [c.label for c in tiny.event_tree.categories]
    assert all(set(r) == {"label", "base", "constraints", "expression"} for r in records)
    for record in records:
        relation = relations.relation(record["label"])
        assert record["base"] == relation.base
        assert record["constraints"] == [c.label for c in relation.constraints]
        assert record["expression"] == relation.expression
    assert [c["label"] for c in relations.records()] == [c.label for c in relations.constraints]
    assert all("leaf_pair_density" in c for c in relations.records())  # the default check is on


# ---------------------------------------------------------------------------------------------
# Constraint families
# ---------------------------------------------------------------------------------------------


_CACHE: dict[int, Relations] = {}


def _all_constraints(seeds=range(4)):
    for seed in seeds:
        if seed not in _CACHE:
            config = world_config(
                seed=seed, event_types={"binary": {"features": {"count": 20, "expected_true": 5}}}
            )
            _CACHE[seed] = relations_of(config)[1]
        yield from _CACHE[seed].constraints


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
                assert item.feature.type in ("property", "part")
                assert item.key == f"{item.role}.{item.feature.label}"
            elif isinstance(item, RoleThreshold):
                assert item.threshold.threshold == round(item.threshold.threshold, 4)
                assert item.key.startswith(f"{item.role}.SCALARDIM.")
        for comparison in record["comparisons"]:
            assert comparison["agent"].startswith("SCALARDIM.")
            assert comparison["patient"].startswith("SCALARDIM.")
    assert families == {"agent", "patient", "cross", "key_lock", "comparison"}


def test_agent_and_patient_constraints_read_one_role() -> None:
    for constraint in _all_constraints():
        roles = {literal_role(item) for item in constraint.literals}
        if constraint.family == "agent":
            assert roles == {"agent"}
        elif constraint.family == "patient":
            assert roles == {"patient"}


def test_cross_and_key_lock_constraints_depend_on_both_roles() -> None:
    seen = {"cross": 0, "key_lock": 0}
    for constraint in _all_constraints(range(6)):
        if constraint.family in seen:
            seen[constraint.family] += 1
            assert constraint.depends_on_both_roles(), str(constraint.expression)
            relevant = {
                literal_role(constraint.literals[i]) for i in constraint.table.relevant_inputs()
            }
            assert {"agent", "patient"} <= relevant
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
        assert roles == ["agent", "patient"] * pairs
        expr = constraint.expression
        terms = list(expr.operands) if isinstance(expr, Op) and expr.operator == "OR" else [expr]
        assert len(terms) == pairs
        for term in terms:
            assert isinstance(term, Op) and term.operator == "AND" and len(term.operands) == 2
        # At most one literal per scalar per role.
        for role in ("agent", "patient"):
            scalars = [
                item.threshold.scalar
                for item in constraint.literals
                if isinstance(item, RoleThreshold) and item.role == role
            ]
            assert len(set(scalars)) == len(scalars)
    assert found


def test_comparisons_agree_with_direct_arithmetic(tiny: StaticWorld, default: StaticWorld) -> None:
    checked = 0
    for statics in (tiny, default):
        scalars = statics.instances.scalars
        n = len(statics.instances)
        for constraint in statics.relations.constraints:
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
                    constraint.matrix(statics.instances.values, scalars, np.arange(n)), expected
                )
                assert constraint.family == "comparison" and constraint.arity == 1
                assert constraint.table.bit_string() == "01"
    assert checked > 0


def test_comparison_margins_and_dimensions(tmp_path: Path) -> None:
    from statistics import NormalDist

    only_comparison = {"agent": 0, "patient": 0, "cross": 0, "key_lock": 0, "comparison": 1}
    config = world_config(
        {"scalars": {"count": 3}},
        tmp_path=tmp_path,
        event_types={
            "binary": {
                "features": {"count": 40, "expected_true": 10},
                "constraint_families": only_comparison,
            }
        },
    )
    taxonomy, relations = relations_of(config)
    sigma = math.sqrt(2) * taxonomy.config.scalars.model_std
    low = NormalDist(0, sigma).inv_cdf(0.1)
    high = NormalDist(0, sigma).inv_cdf(0.9)
    windows = cross = 0
    comparisons = [c.comparisons[0] for c in relations.constraints]
    for comparison in comparisons:
        assert low - 1e-4 <= comparison.low <= high + 1e-4
        if comparison.high is not None:
            windows += 1
            assert comparison.low < comparison.high <= high + 1e-4
        if comparison.agent_scalar != comparison.patient_scalar:
            cross += 1
    assert 0 < windows < len(comparisons)
    assert 0 < cross < len(comparisons)
    single = world_config(
        str(TAXONOMY_DATA / "tiny_relations.yaml"),
        event_types={"binary": {"constraint_families": only_comparison}},
    )
    _, single_relations = relations_of(single)
    assert all(
        c.comparisons[0].agent_scalar == c.comparisons[0].patient_scalar == 1
        for c in single_relations.constraints
    )


def test_no_comparisons_without_scalars(tmp_path: Path) -> None:
    config = world_config({"scalars": {"count": 0}}, tmp_path=tmp_path)
    _, relations = relations_of(config)
    assert all(c.family != "comparison" for c in relations.constraints)
    assert all(
        not isinstance(item, RoleThreshold | ScalarComparison)
        for c in relations.constraints
        for item in c.literals
    )


def test_constraints_read_static_facts_only() -> None:
    """No constraint reads an ISA feature, a capacity, a one-place event type, or a fluent."""
    for constraint in _all_constraints():
        for key in constraint.keys:
            assert "agent." in key or "patient." in key  # a window key starts with its bound
            assert not any(
                part in key for part in ("ISA.", "CAN.", "CANBE.", "EVENTTYPE1.", "BOOLFL.")
            )


def test_cross_constraints_need_arity_two() -> None:
    config = world_config(
        event_types={
            "binary": {
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
        relations_of(config)


# ---------------------------------------------------------------------------------------------
# Explicit requirements from an event file
# ---------------------------------------------------------------------------------------------


def test_explicit_requirements_replace_the_own_constraint_and_keep_the_base_relations() -> None:
    """An explicit requirement replaces an event type's own constraint and the constraints of its
    non-defining true features; the constraints of the features defining at an ancestor (the
    parent's base relation) stay (REL.10)."""
    chain = statics_of(load_config(DATA / "tiny_chain.yaml"))
    sampled = define(load_config(DATA / "tiny.yaml")).statics
    relations = chain.relations
    assert relations.explicit == frozenset({"EVENTTYPE2.1.1", "EVENTTYPE2.1.2"})
    for label in sorted(relations.explicit):
        own = relations.own_constraints[label]
        assert own.family == "explicit" and own.label == constraint_label(label)
        assert str(own.expression) == "agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.0000"
        event_type = relations.event_tree.tree[label]
        parent = event_type.parent
        assert parent is not None
        base = relations.relation(parent)
        relation = relations.relation(event_type)
        inherited = (event_type.values == 1) & (event_type.roles == Role.DEFINING_INHERITED)
        assert [c.label for c in relation.constraints] == [
            f"CONSTRAINT.EVENTFEAT.{i + 1}" for i in np.flatnonzero(inherited)
        ] + [own.label]
        assert {c.label for c in base.constraints} <= {c.label for c in relation.constraints}
        assert np.all(relations.matrix(event_type) <= relations.matrix(parent))
        # The explicit requirement is the whole requirement apart from the base relation.
        assert relations.constraint_density is not None
        assert relations.constraint_density[own.label] == pytest.approx(
            float(
                own.evaluate(
                    *_leaf_pairs(chain.taxonomy),
                ).mean()
            )
        )
    # Event types without an explicit entry keep their sampled constraints, label for label.
    for label in ("EVENTTYPE2.2.1", "EVENTTYPE2.2.2"):
        assert relations.own_constraints[label].record() == (
            sampled.relations.own_constraints[label].record()
        )
    assert [c.record() for c in relations.feature_constraints] == [
        c.record() for c in sampled.relations.feature_constraints
    ]


def _leaf_pairs(taxonomy: TaxonomyResult) -> tuple[np.ndarray, ...]:
    """Every ordered pair of leaves, as aligned agent and patient rows."""
    leaves = taxonomy.tree.leaves
    values = np.stack([leaf.values for leaf in leaves])
    scalars = np.stack([leaf.scalars for leaf in leaves])
    n = len(leaves)
    agents, patients = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    agents, patients = agents.ravel(), patients.ravel()
    return values[agents], scalars[agents], values[patients], scalars[patients]


def test_an_explicit_requirement_for_an_unknown_event_type_names_the_file(tmp_path: Path) -> None:
    file = tmp_path / "events.yaml"
    file.write_text(
        yaml.safe_dump(
            {
                "event_types": {
                    "EVENTTYPE2.9.9": {
                        "requirement": "agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.0"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = world_config(
        str(TAXONOMY_DATA / "tiny_relations.yaml"), event_types={"event_file": str(file)}
    )
    from semantic_world.taxonomy.config import ConfigError

    with pytest.raises(ConfigError, match="EVENTTYPE2.9.9.*unknown event type") as info:
        statics_of(config)
    assert str(file) in str(info.value)


# ---------------------------------------------------------------------------------------------
# Determinism and independence
# ---------------------------------------------------------------------------------------------


def test_constraints_are_deterministic_and_use_their_stream(tmp_path: Path) -> None:
    a = relations_of(world_config(seed=3))[1]
    b = relations_of(world_config(seed=3))[1]
    c = relations_of(world_config(seed=4))[1]
    assert a.records() == b.records() and a.records() != c.records()
    assert a.relation_records() == b.relation_records()
    # The constraints are drawn over the leaves, never the instances: more instances change
    # nothing in the constraints or the relations.
    taxonomy = yaml.safe_load(Path(RELATIONS).read_text(encoding="utf-8"))
    more = world_config({**taxonomy, "instances": {"per_leaf": 12}}, seed=3, tmp_path=tmp_path)
    more_relations = relations_of(more)[1]
    assert more_relations.records() == a.records()
    assert more_relations.relation_records() == a.relation_records()
    assert len(more_relations.instances) > len(a.instances)


def test_constraints_leave_the_taxonomy_and_the_other_streams_alone() -> None:
    config = world_config(seed=1)
    taxonomy = generate_taxonomy(config.taxonomy_config())
    used = WorldStreams(1)
    event_tree = generate_event_tree(config.event_types.binary, used.event_tree)
    relations = generate_constraints(
        config.event_types.binary,
        taxonomy.config.scalars,
        taxonomy.rules.features,
        event_tree,
        taxonomy.instances,
        used.constraints,
    )
    assert relations is not None and len(relations.constraints) > 12
    fresh = WorldStreams(1)
    assert used.constraints.random() != fresh.constraints.random()
    assert used.event_tree.random() != fresh.event_tree.random()
    for name in ("requirements", "pairs", "fluents", "initial", "preconditions", "effects"):
        assert getattr(used, name).random() == getattr(fresh, name).random(), name
    # The taxonomy run is the same whether or not the world has two-place event types.
    none = world_config(seed=1, event_types={"binary": None})
    assert generate_taxonomy(none.taxonomy_config()).rules.records() == taxonomy.rules.records()
