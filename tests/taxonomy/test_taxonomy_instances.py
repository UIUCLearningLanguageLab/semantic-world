"""Stage 5 acceptance tests: instances, node vectors, and the fixed-by-rule test."""

# Since stage a5b of the world model the labels are CATEGORY.<path> and INSTANCE.<path>.<k>, and
# the taxonomy has no CAN features.

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from semantic_world.taxonomy import (
    Streams,
    config_from_mapping,
    generate_rules,
    generate_tree,
    load_config,
)
from semantic_world.taxonomy.fixed import (
    FIXED_EXACT,
    FIXED_LOCAL,
    FIXED_NONE,
    all_feature_labels,
    compute_node_vectors,
    fixed_by_rule,
)
from semantic_world.taxonomy.instances import generate_instances
from semantic_world.taxonomy.rules import compute_features
from semantic_world.taxonomy.tree import Role

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"

CATEGORY = "CATEGORY."
CHAINED = {
    "features": {
        "property": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
        "part": {"count": 20, "proportion_determined": 0.5, "expected_true_free": 3},
    },
    "rules": {"max_chain_depth": 3},
    "inheritance": {"proportion_defining": 0.3, "proportion_characteristic": 0.4},
}


def build(overrides: dict | None = None, seed: int = 1, path: Path | None = None):
    config = (
        load_config(path, seed=seed) if path else config_from_mapping(overrides or {}, seed=seed)
    )
    streams = Streams(config.seed)
    rules = generate_rules(config, streams)
    tree = generate_tree(config, rules, streams)
    instances = generate_instances(config, rules, tree, streams)
    return config, rules, tree, instances


# ---------------------------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------------------------


def test_instance_labels_and_counts() -> None:
    config, rules, tree, instances = build()
    assert len(instances) == instances.values.shape[0]
    assert instances.values.shape[1] == len(rules.features)
    assert instances.values.dtype == np.uint8
    per_leaf: dict[str, list[str]] = {}
    for label, leaf_label, index in zip(
        instances.labels, instances.leaf_labels, instances.leaf_index, strict=True
    ):
        leaf = tree.categories[index]
        assert leaf.label == leaf_label
        assert leaf.is_leaf
        assert leaf.label.startswith(CATEGORY)
        path = leaf.label[len(CATEGORY) :]
        assert label == f"INSTANCE.{path}.{len(per_leaf.get(leaf_label, [])) + 1}"
        per_leaf.setdefault(leaf_label, []).append(label)
    assert list(per_leaf) == [leaf.label for leaf in tree.leaves]  # leaf order
    assert all(5 <= len(v) <= 10 for v in per_leaf.values())
    assert set(len(v) for v in per_leaf.values()) != {5}
    assert instances.labels[0] == "INSTANCE.1.1.1.1"


def test_tiny_instances() -> None:
    _, _, tree, instances = build(path=DATA / "tiny.yaml")
    assert len(tree.leaves) == 4
    assert len(instances) == 12
    assert instances.labels[:3] == ("INSTANCE.1.1.1", "INSTANCE.1.1.2", "INSTANCE.1.1.3")
    assert instances.leaf_labels[:3] == ("CATEGORY.1.1",) * 3


def test_instances_follow_the_leaf_roles() -> None:
    _, rules, tree, instances = build()
    free = instances.free_values(rules.features)
    for i, leaf_index in enumerate(instances.leaf_index):
        leaf = tree.categories[leaf_index]
        mask = leaf.defining_mask()
        assert np.array_equal(free[i, mask], leaf.free_values[mask])


def test_instance_copy_and_base_rates() -> None:
    # Equal base rates, so the undiagnostic draws have one rate to compare with.
    _, rules, tree, instances = build(
        {
            "features": {"base_rate_heterogeneity": None},
            "instances": {"per_leaf": 40, "characteristic_probability": 0.8},
        }
    )
    free = instances.free_values(rules.features)
    copies = total_characteristic = trues = total_undiagnostic = 0
    for i, leaf_index in enumerate(instances.leaf_index):
        leaf = tree.categories[leaf_index]
        characteristic = leaf.roles == Role.CHARACTERISTIC
        copies += (free[i, characteristic] == leaf.free_values[characteristic]).sum()
        total_characteristic += characteristic.sum()
        undiagnostic = leaf.roles == Role.UNDIAGNOSTIC
        trues += free[i, undiagnostic].sum()
        total_undiagnostic += undiagnostic.sum()
    assert total_characteristic > 5000
    assert copies / total_characteristic == pytest.approx(0.8, abs=0.02)
    assert trues / total_undiagnostic == pytest.approx(0.2, abs=0.02)


@pytest.mark.parametrize("overrides", [{}, CHAINED], ids=["default", "chained"])
def test_recomputing_determined_features_reproduces_the_instances(overrides: dict) -> None:
    _, rules, _, instances = build(overrides, seed=2)
    recomputed = rules.compute(instances.free_values(rules.features))
    assert np.array_equal(recomputed, instances.values)
    assert np.array_equal(
        compute_features(rules.features, rules.rules, instances.free_values(rules.features)),
        instances.values,
    )


def test_isa_and_full_matrix() -> None:
    _, rules, tree, instances = build()
    isa = instances.isa_matrix(tree)
    assert isa.shape == (len(instances), len(tree.categories))
    for i, leaf_index in enumerate(instances.leaf_index):
        leaf = tree.categories[leaf_index]
        on = {leaf.label} | {a.label for a in leaf.ancestors()}
        assert {tree.categories[j].label for j in np.flatnonzero(isa[i])} == on
    full = instances.full_matrix(tree)
    assert full.shape == (len(instances), len(tree.categories) + len(rules.features))
    assert np.array_equal(full[:, len(tree.categories) :], instances.values)


def test_below() -> None:
    _, _, tree, instances = build()
    everything = np.concatenate([instances.below(c, tree) for c in tree.superordinates])
    assert sorted(everything.tolist()) == list(range(len(instances)))
    leaf = tree.leaves[0]
    below = instances.below(leaf, tree)
    assert all(instances.leaf_labels[i] == leaf.label for i in below)


def test_zero_instances_per_leaf() -> None:
    _, rules, tree, instances = build({"instances": {"per_leaf": 0}})
    assert len(instances) == 0
    assert instances.values.shape == (0, len(rules.features))
    vectors = compute_node_vectors(rules, tree, instances)
    assert np.isnan(vectors.mean).all()


# ---------------------------------------------------------------------------------------------
# Node vectors
# ---------------------------------------------------------------------------------------------


def test_feature_label_order() -> None:
    _, rules, tree, instances = build()
    labels = all_feature_labels(rules, tree)
    n_cat = len(tree.categories)
    assert labels[:n_cat] == tuple(f"ISA.{c.label}" for c in tree.categories)
    assert labels[n_cat:] == rules.features.labels
    vectors = compute_node_vectors(rules, tree, instances)
    assert vectors.feature_labels == labels
    assert vectors.isa_count == n_cat
    assert (
        vectors.generative.shape
        == vectors.defining.shape
        == vectors.mean.shape
        == (n_cat, len(labels))
    )


def test_generative_vectors() -> None:
    _, rules, tree, instances = build()
    vectors = compute_node_vectors(rules, tree, instances)
    n_cat = len(tree.categories)
    for ci, category in enumerate(tree.categories):
        assert np.array_equal(vectors.generative[ci, :n_cat], tree.isa_vector(category))
        assert np.array_equal(vectors.generative[ci, n_cat:], category.values)


@pytest.mark.parametrize("overrides", [{}, CHAINED], ids=["default", "chained"])
@pytest.mark.parametrize("seed", range(3))
def test_defining_vectors_match_every_instance_below(overrides: dict, seed: int) -> None:
    _, rules, tree, instances = build(overrides, seed)
    vectors = compute_node_vectors(rules, tree, instances)
    full = instances.full_matrix(tree)
    n_cat = len(tree.categories)
    any_fixed = False
    for ci, category in enumerate(tree.categories):
        row = vectors.defining[ci]
        known = ~np.isnan(row)
        below = instances.below(category, tree)
        assert len(below) > 0
        assert np.array_equal(full[below][:, known], np.tile(row[known], (len(below), 1)))
        # ISA entries: 1 for the category and its ancestors, NaN for descendants, 0 otherwise.
        isa = row[:n_cat]
        ancestors = {category.label} | {a.label for a in category.ancestors()}
        descendants = {d.label for d in category.descendants()}
        for j, other in enumerate(tree.categories):
            if other.label in ancestors:
                assert isa[j] == 1
            elif other.label in descendants:
                assert np.isnan(isa[j])
            else:
                assert isa[j] == 0
        # Free entries: exactly the defining features.
        free_known = known[n_cat + rules.features.free_positions]
        assert np.array_equal(free_known, category.defining_mask())
        # Determined entries: exactly the features fixed by rule.
        determined_positions = np.array([f.position for f in rules.features.determined])
        assert np.array_equal(
            known[n_cat + determined_positions], vectors.fixed_by_rule[ci, determined_positions]
        )
        any_fixed |= bool(vectors.fixed_by_rule[ci].any())
    assert any_fixed


def test_mean_vectors_match_direct_means() -> None:
    _, rules, tree, instances = build(CHAINED, seed=1)
    vectors = compute_node_vectors(rules, tree, instances)
    full = instances.full_matrix(tree).astype(float)
    for ci, category in enumerate(tree.categories):
        leaves = {leaf.label for leaf in category.leaves()}
        rows = [i for i, leaf in enumerate(instances.leaf_labels) if leaf in leaves]
        assert np.allclose(vectors.mean[ci], full[rows].mean(axis=0))
    # Means of ISA features: 1 for the category and its ancestors.
    n_cat = len(tree.categories)
    for ci, category in enumerate(tree.categories):
        assert vectors.mean[ci, ci] == 1.0
        for ancestor in category.ancestors():
            assert vectors.mean[ci, tree.categories.index(ancestor)] == 1.0
    assert np.all((vectors.mean >= 0) & (vectors.mean <= 1))
    assert vectors.mean.shape == (n_cat, n_cat + len(rules.features))


# ---------------------------------------------------------------------------------------------
# The fixed-by-rule test
# ---------------------------------------------------------------------------------------------


def _brute_force_fixed(rules, category) -> np.ndarray:
    """Enumerate every setting of every free feature, holding the defining ones, and check which
    determined features are constant."""
    features = rules.features
    n_free = len(features.free)
    defining = category.defining_mask()
    open_columns = np.flatnonzero(~defining)
    grid = np.array(
        list(itertools.product((0, 1), repeat=len(open_columns))), dtype=np.uint8
    ).reshape(-1, len(open_columns))
    free_values = np.tile(category.free_values, (grid.shape[0], 1))
    free_values[:, open_columns] = grid
    out = rules.compute(free_values)
    constant = out.min(axis=0) == out.max(axis=0)
    determined = np.array([not f.free for f in features.features])
    assert n_free == len(defining)
    return constant & determined


@pytest.mark.parametrize("seed", range(4))
def test_fixed_by_rule_matches_brute_force_on_tiny(seed: int) -> None:
    # The tiny configuration with more defining features, so fixed-by-rule cases appear.
    config = config_from_mapping(
        {
            **{
                "features": {
                    "property": {"count": 8, "expected_true_free": 2},
                    "part": {"count": 8, "expected_true_free": 2},
                }
            },
            "taxonomy": {"superordinates": 2, "depth": 2, "branching": 2},
            "instances": {"per_leaf": 3},
            "inheritance": {"proportion_defining": 0.4, "proportion_characteristic": 0.3},
        },
        seed=seed,
    )
    streams = Streams(config.seed)
    rules = generate_rules(config, streams)
    tree = generate_tree(config, rules, streams)
    found_fixed = False
    for category in tree.categories:
        fixed, test = fixed_by_rule(rules, category)
        expected = _brute_force_fixed(rules, category)
        assert np.array_equal(fixed, expected), category.label
        assert np.all((test != FIXED_NONE) == fixed)
        assert np.all(
            test[fixed] == FIXED_EXACT
        )  # cones of 12 free features are enumerated exactly
        found_fixed |= bool(fixed.any())
        # A fixed determined feature has the category's own value.
        for position in np.flatnonzero(fixed):
            assert rules.features.features[position].free is False
    assert found_fixed


def test_fixed_by_rule_on_the_example_tiny_configuration() -> None:
    _, rules, tree, _ = build(path=DATA / "tiny.yaml")
    for category in tree.categories:
        fixed, _ = fixed_by_rule(rules, category)
        assert np.array_equal(fixed, _brute_force_fixed(rules, category))


def test_local_test_never_marks_a_feature_that_is_not_fixed() -> None:
    for seed in range(3):
        _, rules, tree, _ = build(CHAINED, seed)
        marked_local = False
        for category in tree.categories:
            exact, exact_test = fixed_by_rule(rules, category)
            local, local_test = fixed_by_rule(rules, category, cone_limit=-1)
            assert np.all(exact_test[exact] == FIXED_EXACT)
            assert np.all(local_test[local] == FIXED_LOCAL)
            assert np.all(local <= exact), category.label  # local ⊆ exact
            marked_local |= bool(local.any())
        assert marked_local


def test_fixed_features_are_constant_over_the_instances_below() -> None:
    _, rules, tree, instances = build(CHAINED, seed=2)
    vectors = compute_node_vectors(rules, tree, instances)
    n_cat = len(tree.categories)
    for ci, category in enumerate(tree.categories):
        below = instances.below(category, tree)
        for position in np.flatnonzero(vectors.fixed_by_rule[ci]):
            column = instances.values[below, position]
            assert column.min() == column.max() == category.values[position]
            assert vectors.defining[ci, n_cat + position] == category.values[position]


# ---------------------------------------------------------------------------------------------
# Determinism and stream independence
# ---------------------------------------------------------------------------------------------


def test_instances_are_deterministic_and_use_only_the_instances_stream() -> None:
    _, _, _, a = build({}, 3)
    _, _, _, b = build({}, 3)
    _, _, _, c = build({}, 4)
    assert a.labels == b.labels and np.array_equal(a.values, b.values)
    assert a.labels != c.labels or not np.array_equal(a.values, c.values)
    config = config_from_mapping({})
    used = Streams(3)
    fresh = Streams(3)
    rules = generate_rules(config, used)
    tree = generate_tree(config, rules, used)
    generate_instances(config, rules, tree, used)
    assert used.analysis.random() == fresh.analysis.random()
    assert used.instances.random() != fresh.instances.random()


def test_changing_the_instance_count_leaves_rules_and_tree_unchanged() -> None:
    _, rules_a, tree_a, instances_a = build({"instances": {"per_leaf": 3}})
    _, rules_b, tree_b, instances_b = build({"instances": {"per_leaf": [8, 12]}})
    assert rules_a.records() == rules_b.records()
    assert [c.label for c in tree_a.categories] == [c.label for c in tree_b.categories]
    assert np.array_equal(tree_a.generative_matrix(), tree_b.generative_matrix())
    assert all(
        np.array_equal(x.roles, y.roles)
        for x, y in zip(tree_a.categories, tree_b.categories, strict=True)
    )
    assert len(instances_a) == 3 * len(tree_a.leaves)
    assert len(instances_b) > len(instances_a)
