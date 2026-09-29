"""Stage 4 acceptance tests: superordinates with the similarity bound, roles, children, and
distinct leaves."""

from __future__ import annotations

import math

import numpy as np
import pytest

from semantic_world.taxonomy import GenerationError, Streams, config_from_mapping, generate_rules
from semantic_world.taxonomy.similarity import cross_similarity, pair_similarity, similarity_matrix
from semantic_world.taxonomy.tree import Role, Tree, generate_tree, stochastic_round


def make(overrides: dict | None = None, seed: int = 1) -> tuple[Tree, object]:
    config = config_from_mapping(overrides or {}, seed=seed)
    streams = Streams(config.seed)
    rules = generate_rules(config, streams)
    return generate_tree(config, rules, streams), rules


LARGE = {
    "taxonomy": {"superordinates": 3, "depth": 4, "branching": 4},
    "superordinates": {"similarity_bound": None},
}


# ---------------------------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------------------------


def test_similarity_metrics() -> None:
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    c = np.array([1, 1, 0, 0])
    assert pair_similarity(a, c, "phi") == pytest.approx(1.0)
    assert pair_similarity(a, b, "phi") == pytest.approx(0.0)
    assert pair_similarity(a, 1 - a, "phi") == pytest.approx(-1.0)
    assert pair_similarity(a, b, "cosine") == pytest.approx(0.5)
    assert pair_similarity(a, b, "jaccard") == pytest.approx(1 / 3)
    assert pair_similarity(a, c, "jaccard") == pytest.approx(1.0)
    # Undefined similarities are NaN.
    zeros = np.zeros(4, dtype=int)
    ones = np.ones(4, dtype=int)
    assert math.isnan(pair_similarity(a, ones, "phi"))
    assert math.isnan(pair_similarity(a, zeros, "cosine"))
    assert math.isnan(pair_similarity(zeros, zeros, "jaccard"))
    assert pair_similarity(ones, ones, "cosine") == pytest.approx(1.0)
    matrix = similarity_matrix(np.stack([a, b, c]), "phi")
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 2] == pytest.approx(1.0)
    assert cross_similarity(np.stack([a]), np.stack([b, c]), "jaccard").shape == (1, 2)
    with pytest.raises(ValueError):
        pair_similarity(a, b, "euclidean")


def test_similarity_against_a_direct_computation() -> None:
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2, size=(20, 15))
    phi = similarity_matrix(rows, "phi")
    for i in range(20):
        for j in range(20):
            expected = np.corrcoef(rows[i], rows[j])[0, 1]
            assert phi[i, j] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------------


def test_default_tree_structure() -> None:
    tree, rules = make()
    labels = [c.label for c in tree.categories]
    assert labels[:2] == ["C1", "C1.1"]
    assert labels == sorted(labels, key=lambda s: tuple(int(i) for i in s[1:].split(".")))
    assert len(tree.superordinates) == 4
    assert tree.depth == 3
    for category in tree.categories:
        assert category.level == len(category.indices)
        assert category.label == "C" + ".".join(str(i) for i in category.indices)
        if category.parent is not None:
            assert category.parent.indices == category.indices[:-1]
            assert category in category.parent.children
        if category.level < 3:
            assert 2 <= len(category.children) <= 4
            assert [c.indices[-1] for c in category.children] == list(
                range(1, len(category.children) + 1)
            )
        else:
            assert category.is_leaf
        assert category.values.shape == (len(rules.features),)
        assert category.free_values.shape == (len(rules.features.free),)
        assert np.array_equal(category.values[rules.features.free_positions], category.free_values)
        assert np.array_equal(rules.compute(category.free_values[None, :])[0], category.values)
    assert all(leaf.level == 3 for leaf in tree.leaves)
    assert tree["C1.1"].parent is tree["C1"]
    with pytest.raises(KeyError):
        tree["C9"]


def test_depth_one_tree() -> None:
    tree, _ = make({"taxonomy": {"depth": 1, "superordinates": 3}})
    assert len(tree.categories) == 3
    assert tree.leaves == tree.superordinates
    assert all(c.roles.shape == (60,) for c in tree.categories)


def test_isa_vectors() -> None:
    tree, _ = make()
    matrix = tree.isa_matrix()
    n = len(tree.categories)
    assert matrix.shape == (n, n)
    assert np.array_equal(np.diag(matrix), np.ones(n, dtype=np.uint8))
    leaf = tree["C2.1.1"]
    on = {tree.categories[i].label for i in np.flatnonzero(tree.isa_vector(leaf))}
    assert on == {"C2", "C2.1", "C2.1.1"}
    assert tree.isa_vector(tree["C2"]).sum() == 1


def test_branching_schedule_and_ranges() -> None:
    tree, _ = make(
        {"taxonomy": {"depth": 3, "branching": {"schedule": "list", "values": [2, [3, 5]]}}}
    )
    for category in tree.at_level(1):
        assert len(category.children) == 2
    counts = {len(c.children) for c in tree.at_level(2)}
    assert counts <= {3, 4, 5}


# ---------------------------------------------------------------------------------------------
# Superordinates and the similarity bound
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_all_superordinate_pairs_lie_within_the_bound(seed: int) -> None:
    tree, rules = make({}, seed)
    rows = np.stack([c.values[rules.features.free_positions] for c in tree.superordinates])
    phi = similarity_matrix(rows, "phi")
    upper = phi[np.triu_indices(4, k=1)]
    assert np.all(upper <= 0.3 + 1e-12)
    assert not np.isnan(upper).any()
    assert tree.superordinate_similarity == (pytest.approx(upper.min()), pytest.approx(upper.max()))
    assert tree.superordinate_tries >= 4


@pytest.mark.parametrize("metric", ["phi", "cosine", "jaccard"])
@pytest.mark.parametrize("scope", ["free", "is_has", "all"])
def test_bound_holds_on_every_metric_and_scope(metric: str, scope: str) -> None:
    low, high = {"phi": (-0.2, 0.25), "cosine": (0.2, 0.6), "jaccard": (0.1, 0.4)}[metric]
    bound = {
        "metric": metric,
        "scope": scope,
        "min": low,
        "max": high,
        "max_tries": 2000,
        "local_search": True,
    }
    tree, rules = make(
        {"superordinates": {"similarity_bound": bound}, "taxonomy": {"depth": 1}}, seed=2
    )
    features = rules.features
    columns = {
        "free": features.free_positions,
        "is_has": np.array([f.position for f in features.features if f.type != "can"]),
        "all": np.arange(len(features)),
    }[scope]
    rows = np.stack([c.values[columns] for c in tree.superordinates])
    sims = similarity_matrix(rows, metric)[np.triu_indices(4, k=1)]
    assert np.all((sims >= low - 1e-12) & (sims <= high + 1e-12)), sims


def test_tight_bound_needs_the_local_search() -> None:
    bound = {
        "metric": "phi",
        "scope": "free",
        "min": -0.05,
        "max": 0.05,
        "max_tries": 20,
        "local_search": True,
    }
    tree, rules = make(
        {"superordinates": {"similarity_bound": bound}, "taxonomy": {"depth": 1}}, seed=3
    )
    rows = np.stack([c.values[rules.features.free_positions] for c in tree.superordinates])
    sims = similarity_matrix(rows, "phi")[np.triu_indices(4, k=1)]
    assert np.all(np.abs(sims) <= 0.05 + 1e-12)
    # One try per superordinate and no local search cannot meet a bound this tight.
    without = dict(bound, local_search=False, max_tries=1, min=-0.01, max=0.01)
    with pytest.raises(GenerationError, match="closest phi similarity reached"):
        make({"superordinates": {"similarity_bound": without}, "taxonomy": {"depth": 1}}, seed=3)


def test_infeasible_bound_reports_the_closest_value() -> None:
    # Three vectors cannot all have pairwise phi below -0.5.
    bound = {
        "metric": "phi",
        "scope": "free",
        "min": None,
        "max": -0.6,
        "max_tries": 200,
        "local_search": True,
    }
    with pytest.raises(GenerationError) as error:
        make(
            {
                "superordinates": {"similarity_bound": bound},
                "taxonomy": {"depth": 1, "superordinates": 3},
            }
        )
    message = str(error.value)
    assert "superordinate C3" in message or "superordinate C2" in message
    assert "closest phi similarity reached was" in message
    assert "loosen superordinates.similarity_bound" in message
    closest = float(message.split("reached was ")[1].split(" ")[0])
    assert closest > -0.6


def test_no_bound_takes_one_try_per_superordinate() -> None:
    tree, _ = make({"superordinates": {"similarity_bound": None}})
    assert tree.superordinate_tries == 4
    assert tree.superordinate_similarity is None


# ---------------------------------------------------------------------------------------------
# Roles and inheritance
# ---------------------------------------------------------------------------------------------


def test_stochastic_rounding_is_unbiased() -> None:
    rng = np.random.default_rng(0)
    draws = [stochastic_round(2.3, rng) for _ in range(20000)]
    assert set(draws) == {2, 3}
    assert np.mean(draws) == pytest.approx(2.3, abs=0.02)
    assert stochastic_round(4.0, rng) == 4


def test_roles_are_valid_at_every_category() -> None:
    tree, rules = make()
    n_free = len(rules.features.free)
    for category in tree.categories:
        roles = category.roles
        assert roles.shape == (n_free,)
        assert set(np.unique(roles)) <= {int(r) for r in Role}
        inherited = roles == Role.DEFINING_INHERITED
        if category.parent is None:
            assert not inherited.any()
        else:
            assert np.array_equal(inherited, category.parent.defining_mask())
        available = ~inherited
        new = (roles == Role.DEFINING_NEW).sum()
        characteristic = (roles == Role.CHARACTERISTIC).sum()
        assert new + characteristic <= available.sum()


def test_role_counts_match_the_proportions_on_average() -> None:
    tree, _ = make(LARGE, seed=1)
    ratios_defining = []
    ratios_characteristic = []
    for category in tree.categories:
        available = (category.roles != Role.DEFINING_INHERITED).sum()
        if available:
            ratios_defining.append((category.roles == Role.DEFINING_NEW).sum() / available)
            ratios_characteristic.append((category.roles == Role.CHARACTERISTIC).sum() / available)
    assert np.mean(ratios_defining) == pytest.approx(0.1, abs=0.02)
    assert np.mean(ratios_characteristic) == pytest.approx(0.5, abs=0.02)


def test_defining_features_keep_their_value_in_every_descendant() -> None:
    for seed in range(3):
        tree, _ = make(LARGE, seed)
        for category in tree.categories:
            mask = category.defining_mask()
            for descendant in category.descendants():
                assert np.array_equal(descendant.free_values[mask], category.free_values[mask])
                assert np.all(descendant.roles[mask] == Role.DEFINING_INHERITED)


def test_characteristic_copy_rate_matches_the_configured_probability() -> None:
    tree, _ = make(LARGE, seed=1)
    copies = flips = 0
    for parent in tree.categories:
        mask = parent.roles == Role.CHARACTERISTIC
        for child in parent.children:
            same = child.free_values[mask] == parent.free_values[mask]
            copies += same.sum()
            flips += (~same).sum()
    assert copies + flips > 5000
    assert copies / (copies + flips) == pytest.approx(0.9, abs=0.02)


def test_characteristic_probability_schedule_applies_per_level() -> None:
    schedule = {"schedule": "list", "values": [1.0, 0.6, 0.9, 0.9]}
    tree, _ = make({**LARGE, "inheritance": {"characteristic_probability": schedule}}, seed=2)
    rates = {}
    for level in (1, 2):
        copies = total = 0
        for parent in tree.at_level(level):
            mask = parent.roles == Role.CHARACTERISTIC
            for child in parent.children:
                same = child.free_values[mask] == parent.free_values[mask]
                copies += same.sum()
                total += mask.sum()
        rates[level] = copies / total
    assert rates[1] == 1.0
    assert rates[2] == pytest.approx(0.6, abs=0.05)


def test_undiagnostic_children_draw_from_the_base_rate() -> None:
    tree, rules = make(LARGE, seed=1)
    trues = total = 0
    for parent in tree.categories:
        mask = parent.roles == Role.UNDIAGNOSTIC
        for child in parent.children:
            trues += child.free_values[mask].sum()
            total += mask.sum()
    assert trues / total == pytest.approx(0.2, abs=0.02)


def test_a_defining_value_can_be_zero() -> None:
    tree, _ = make(LARGE, seed=1)
    zeros = sum(
        int(((c.roles == Role.DEFINING_NEW) & (c.free_values == 0)).sum()) for c in tree.categories
    )
    assert zeros > 0


# ---------------------------------------------------------------------------------------------
# Distinct leaves
# ---------------------------------------------------------------------------------------------


def test_no_two_leaves_share_a_vector() -> None:
    for seed in range(3):
        tree, _ = make(LARGE, seed)
        vectors = {leaf.values.tobytes() for leaf in tree.leaves}
        assert len(vectors) == len(tree.leaves) == 192


FEW_FEATURES = {
    "features": {
        "is": {"count": 2, "proportion_determined": 0, "expected_true_free": 1},
        "has": {"count": 0, "proportion_determined": 0, "expected_true_free": 0},
        "can": {"count": 1},
    },
    "rules": {"arity": {1: 1}, "input_type_weights": {"is": 1, "has": 0}},
    "taxonomy": {"superordinates": 1, "depth": 2, "branching": 8},
    "superordinates": {"similarity_bound": None},
    "inheritance": {
        "proportion_defining": 0,
        "proportion_characteristic": 0,
        "distinct_max_tries": 50,
    },
}


def test_distinct_leaves_fail_when_the_vectors_run_out() -> None:
    # Two free features allow four distinct leaves; eight are asked for.
    with pytest.raises(GenerationError, match="duplicates leaf") as error:
        make(FEW_FEATURES)
    assert "50 redraws" in str(error.value)
    assert "require_distinct_leaves" in str(error.value)


def test_distinct_leaves_can_be_turned_off() -> None:
    config = {
        **FEW_FEATURES,
        "inheritance": {**FEW_FEATURES["inheritance"], "require_distinct_leaves": False},
    }
    tree, _ = make(config)
    assert len(tree.leaves) == 8
    assert len({leaf.values.tobytes() for leaf in tree.leaves}) <= 4


def test_distinct_leaves_redraw_until_unique() -> None:
    four = {**FEW_FEATURES, "taxonomy": {**FEW_FEATURES["taxonomy"], "branching": 4}}
    tree, _ = make(four, seed=1)
    assert len({leaf.values.tobytes() for leaf in tree.leaves}) == 4


# ---------------------------------------------------------------------------------------------
# Determinism and stream independence
# ---------------------------------------------------------------------------------------------


def _snapshot(tree: Tree) -> list[tuple[str, bytes, bytes]]:
    return [(c.label, c.values.tobytes(), c.roles.tobytes()) for c in tree.categories]


def test_tree_is_deterministic_and_uses_only_its_streams() -> None:
    a, _ = make({}, 7)
    b, _ = make({}, 7)
    c, _ = make({}, 8)
    assert _snapshot(a) == _snapshot(b)
    assert _snapshot(a) != _snapshot(c)
    config = config_from_mapping({})
    used = Streams(7)
    fresh = Streams(7)
    generate_tree(config, generate_rules(config, used), used)
    for name in ("instances", "analysis"):
        assert getattr(used, name).random() == getattr(fresh, name).random(), name
    assert used.tree.random() != fresh.tree.random()
    assert used.superordinates.random() != fresh.superordinates.random()


def test_changing_the_tree_does_not_change_the_rules() -> None:
    _, rules_a = make({})
    _, rules_b = make({"taxonomy": {"superordinates": 2, "depth": 2}})
    assert rules_a.records() == rules_b.records()
