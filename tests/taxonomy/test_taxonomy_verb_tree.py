"""Stage 9 acceptance tests: verb features, the verb tree, and verb roles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semantic_world.taxonomy import (
    ConfigError,
    GenerationError,
    Streams,
    config_from_mapping,
    generate,
    generate_verb_tree,
)
from semantic_world.taxonomy.similarity import similarity_matrix
from semantic_world.taxonomy.tree import Role

VERBS = {"verbs": {}}
LARGE_VERBS = {
    "verbs": {
        "features": {"count": 30, "expected_true": 6},
        "taxonomy": {"superordinates": 3, "depth": 4, "branching": 4},
        "inheritance": {"proportion_defining": 0.1, "proportion_characteristic": 0.5},
    }
}


def verb_tree(overrides: dict, seed: int = 1):
    config = config_from_mapping(overrides, seed=seed)
    return config, generate_verb_tree(config, Streams(config.seed))


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_verbs_are_off_by_default() -> None:
    config = config_from_mapping({})
    assert config.verbs is None
    assert config.resolved()["verbs"] is None
    assert generate_verb_tree(config, Streams(1)) is None
    assert generate(config).verbs is None


def test_verb_defaults_match_the_specification() -> None:
    config = config_from_mapping(VERBS)
    verbs = config.verbs
    assert verbs is not None
    assert (verbs.feature_count, verbs.expected_true) == (12, 3)
    assert verbs.base_rate == pytest.approx(0.25)
    assert verbs.feature_labels[:2] == ("VF.1", "VF.2")
    assert (verbs.taxonomy.superordinates, verbs.taxonomy.depth) == (3, 2)
    assert [r.resolved() for r in verbs.taxonomy.branching] == [[2, 3]]
    assert verbs.similarity_bound is None
    assert verbs.inheritance.proportion_defining == (0.4, 0.4)
    assert verbs.inheritance.proportion_characteristic == (0.4, 0.4)
    assert verbs.inheritance.characteristic_probability == (0.9, 0.9)
    assert verbs.inheritance.require_distinct_leaves is True
    assert verbs.inheritance.distinct_max_tries == 1000
    assert verbs.own_constraint is True
    assert verbs.constraint_families == {
        "agent": 1,
        "patient": 1,
        "cross": 1,
        "key_lock": 1,
        "comparison": 1,
    }
    assert verbs.key_lock_pairs == {1: 0.5, 2: 0.3, 3: 0.2}
    assert verbs.comparison.window_probability == 0.3
    assert verbs.comparison.cross_dimension_probability == 0.2
    assert verbs.comparison.margin_quantiles == (0.1, 0.9)
    assert verbs.rules == config.rules.sampling
    assert (verbs.expose_agent, verbs.expose_patient) == (1.0, 0.25)
    assert (verbs.sampled_true, verbs.sampled_false, verbs.max_exact_pairs) == (
        1000,
        1000,
        50_000_000,
    )
    resolved = config.resolved()
    assert list(resolved["verbs"]) == [
        "features",
        "taxonomy",
        "superordinates",
        "inheritance",
        "own_constraint",
        "constraint_families",
        "key_lock_pairs",
        "comparison",
        "rules",
        "projections",
        "pairs",
    ]
    assert config_from_mapping(resolved).resolved() == resolved


def test_verb_rules_override_the_top_level_settings() -> None:
    config = config_from_mapping(
        {"rules": {"negation_probability": 0.5}, "verbs": {"rules": {"arity": {2: 1}}}}
    )
    assert config.verbs.rules.arity == {2: 1}
    assert config.verbs.rules.negation_probability == 0.5
    assert config.rules.sampling.arity == {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"verbs": {"colour": 1}}, "verbs.colour"),
        ({"verbs": {"features": {"count": -1}}}, "verbs.features.count"),
        ({"verbs": {"features": {"count": 4, "expected_true": 5}}}, "verbs.features.expected_true"),
        ({"verbs": {"taxonomy": {"depth": 0}}}, "verbs.taxonomy.depth"),
        (
            {"verbs": {"taxonomy": {"depth": 3, "branching": {"schedule": "list", "values": [2]}}}},
            "verbs.taxonomy.branching.values",
        ),
        (
            {
                "verbs": {
                    "inheritance": {"proportion_defining": 0.7, "proportion_characteristic": 0.6}
                }
            },
            "verbs.inheritance.proportion_characteristic",
        ),
        (
            {
                "verbs": {
                    "inheritance": {"proportion_defining": {"schedule": "list", "values": [0.1]}}
                }
            },
            "verbs.inheritance.proportion_defining.values",
        ),
        (
            {"verbs": {"superordinates": {"similarity_bound": {"metric": "euclid"}}}},
            "verbs.superordinates.similarity_bound.metric",
        ),
        ({"verbs": {"own_constraint": "yes"}}, "verbs.own_constraint"),
        (
            {"verbs": {"constraint_families": {"similarity": 1}}},
            "verbs.constraint_families.similarity",
        ),
        (
            {
                "verbs": {
                    "constraint_families": {
                        "agent": 0,
                        "patient": 0,
                        "cross": 0,
                        "key_lock": 0,
                        "comparison": 1,
                    }
                }
            },
            "verbs.constraint_families",
        ),
        ({"verbs": {"key_lock_pairs": {0: 1}}}, "verbs.key_lock_pairs.0"),
        (
            {"verbs": {"comparison": {"margin_quantiles": [0.9, 0.1]}}},
            "verbs.comparison.margin_quantiles",
        ),
        (
            {"verbs": {"comparison": {"window_probability": 2}}},
            "verbs.comparison.window_probability",
        ),
        ({"verbs": {"rules": {"colour": 1}}}, "verbs.rules.colour"),
        ({"verbs": {"projections": {"expose_patient": 1.5}}}, "verbs.projections.expose_patient"),
        ({"verbs": {"pairs": {"sampled_true": -1}}}, "verbs.pairs.sampled_true"),
        ({"verbs": {"pairs": {"max_exact_pairs": 0}}}, "verbs.pairs.max_exact_pairs"),
        ({"verbs": []}, "verbs"),
    ],
)
def test_broken_verb_settings_name_the_field(overrides: dict, field: str) -> None:
    with pytest.raises(ConfigError) as error:
        config_from_mapping(overrides)
    assert error.value.field == field


def test_comparison_family_is_usable_with_scalars() -> None:
    only_comparison = {"agent": 0, "patient": 0, "cross": 0, "key_lock": 0, "comparison": 1}
    config = config_from_mapping(
        {"scalars": {"count": 1}, "verbs": {"constraint_families": only_comparison}}
    )
    assert config.verbs.constraint_families["comparison"] == 1


# ---------------------------------------------------------------------------------------------
# Structure: the tree tests of stage 4, for the verb tree
# ---------------------------------------------------------------------------------------------


def test_verb_features_and_tree_structure() -> None:
    config, verbs = verb_tree(VERBS)
    assert verbs is not None
    features = verbs.features
    assert features.labels == tuple(f"VF.{i}" for i in range(1, 13))
    assert all(f.free and f.type == "vf" and f.layer == 0 for f in features.features)
    assert features.base_rates.tolist() == [pytest.approx(0.25)] * 12
    assert features.determined == ()
    assert verbs.rules.rules == ()
    tree = verbs.tree
    labels = [c.label for c in tree.categories]
    assert labels[:2] == ["V1", "V1.1"]
    assert all(label.startswith("V") for label in labels)
    assert len(tree.superordinates) == 3
    assert tree.depth == 2
    for category in tree.categories:
        assert category.level == len(category.indices)
        assert category.label == "V" + ".".join(str(i) for i in category.indices)
        assert category.values.shape == (12,)
        assert np.array_equal(category.values, category.free_values)
        assert category.scalars.shape == (0,)
        if category.level == 1:
            assert 2 <= len(category.children) <= 3
        else:
            assert category.is_leaf
    assert verbs.verbs == tree.leaves
    assert 6 <= len(verbs.verbs) <= 9
    assert tree.superordinate_tries == 3
    assert tree.superordinate_similarity is None


def test_verb_roles_are_valid_and_defining_features_persist() -> None:
    for seed in range(3):
        _, verbs = verb_tree(LARGE_VERBS, seed)
        tree = verbs.tree
        assert len(tree.categories) == 3 + 12 + 48 + 192
        for category in tree.categories:
            roles = category.roles
            assert roles.shape == (30,)
            inherited = roles == Role.DEFINING_INHERITED
            if category.parent is None:
                assert not inherited.any()
            else:
                assert np.array_equal(inherited, category.parent.defining_mask())
            mask = category.defining_mask()
            for descendant in category.descendants():
                assert np.array_equal(descendant.free_values[mask], category.free_values[mask])
                assert np.all(descendant.roles[mask] == Role.DEFINING_INHERITED)


def test_verb_copy_rate_and_base_rate() -> None:
    _, verbs = verb_tree(LARGE_VERBS, 1)
    tree = verbs.tree
    copies = flips = trues = undiagnostic_total = 0
    for parent in tree.categories:
        characteristic = parent.roles == Role.CHARACTERISTIC
        undiagnostic = parent.roles == Role.UNDIAGNOSTIC
        for child in parent.children:
            same = child.free_values[characteristic] == parent.free_values[characteristic]
            copies += same.sum()
            flips += (~same).sum()
            trues += child.free_values[undiagnostic].sum()
            undiagnostic_total += undiagnostic.sum()
    assert copies + flips > 2000
    assert copies / (copies + flips) == pytest.approx(0.9, abs=0.02)
    assert trues / undiagnostic_total == pytest.approx(0.2, abs=0.02)


def test_verb_superordinate_bound() -> None:
    bound = {
        "metric": "phi",
        "scope": "free",
        "min": -0.3,
        "max": 0.1,
        "max_tries": 2000,
        "local_search": True,
    }
    config, verbs = verb_tree(
        {
            "verbs": {
                "superordinates": {"similarity_bound": bound},
                "features": {"count": 20, "expected_true": 6},
            }
        }
    )
    rows = np.stack([c.values for c in verbs.tree.superordinates])
    sims = similarity_matrix(rows, "phi")[np.triu_indices(3, k=1)]
    assert np.all((sims >= -0.3 - 1e-12) & (sims <= 0.1 + 1e-12))
    assert verbs.tree.superordinate_similarity is not None
    infeasible = dict(bound, min=None, max=-0.6, max_tries=100)
    with pytest.raises(GenerationError, match="superordinate V"):
        verb_tree(
            {
                "verbs": {
                    "superordinates": {"similarity_bound": infeasible},
                    "features": {"count": 20, "expected_true": 6},
                }
            }
        )


def test_distinct_verbs() -> None:
    _, verbs = verb_tree(LARGE_VERBS, 2)
    assert len({v.values.tobytes() for v in verbs.verbs}) == len(verbs.verbs) == 192
    few = {
        "verbs": {
            "features": {"count": 2, "expected_true": 1},
            "taxonomy": {"superordinates": 1, "depth": 2, "branching": 8},
            "inheritance": {
                "proportion_defining": 0,
                "proportion_characteristic": 0,
                "distinct_max_tries": 20,
            },
        }
    }
    with pytest.raises(GenerationError, match="duplicates leaf"):
        verb_tree(few)
    allowed = {
        "verbs": {
            **few["verbs"],
            "inheritance": {**few["verbs"]["inheritance"], "require_distinct_leaves": False},
        }
    }
    _, verbs = verb_tree(allowed)
    assert len(verbs.verbs) == 8


# ---------------------------------------------------------------------------------------------
# The flat case
# ---------------------------------------------------------------------------------------------


def test_flat_verb_tree_gives_independent_verbs() -> None:
    flat = {
        "verbs": {
            "taxonomy": {"superordinates": 400, "depth": 1},
            "features": {"count": 10, "expected_true": 3},
            "inheritance": {"require_distinct_leaves": False},
        }
    }
    _, verbs = verb_tree(flat)
    tree = verbs.tree
    assert len(tree.categories) == 400
    assert all(v.parent is None and v.is_leaf and v.level == 1 for v in verbs.verbs)
    assert all(not (v.roles == Role.DEFINING_INHERITED).any() for v in verbs.verbs)
    # Each verb is an independent draw from the base rates: the mean is the base rate, and the
    # verb features are uncorrelated across verbs.
    matrix = tree.generative_matrix().astype(float)
    assert matrix.mean() == pytest.approx(0.3, abs=0.02)
    correlations = np.corrcoef(matrix, rowvar=False)
    off_diagonal = correlations[~np.eye(10, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.2


# ---------------------------------------------------------------------------------------------
# Determinism and independence from the nouns
# ---------------------------------------------------------------------------------------------


def _verb_snapshot(verbs) -> list:
    return [(c.label, c.values.tobytes(), c.roles.tobytes()) for c in verbs.tree.categories]


def _noun_snapshot(result) -> tuple:
    return (
        result.rules.records(),
        result.tree.generative_matrix().tobytes(),
        np.stack([c.roles for c in result.tree.categories]).tobytes(),
        result.instances.labels,
        result.instances.values.tobytes(),
        result.instances.scalars.tobytes(),
        result.similarity.write_csv(),
        result.feature_stats.write_csv(),
    )


def test_verb_tree_is_deterministic_and_uses_only_its_stream() -> None:
    _, a = verb_tree(VERBS, 4)
    _, b = verb_tree(VERBS, 4)
    _, c = verb_tree(VERBS, 5)
    assert _verb_snapshot(a) == _verb_snapshot(b)
    assert _verb_snapshot(a) != _verb_snapshot(c)
    used = Streams(4)
    generate_verb_tree(config_from_mapping(VERBS), used)
    fresh = Streams(4)
    assert used.verb_tree.random() != fresh.verb_tree.random()
    for name in (
        "base_rates",
        "rules",
        "superordinates",
        "tree",
        "instances",
        "analysis",
        "scalars",
        "scalar_instances",
    ):
        assert getattr(used, name).random() == getattr(fresh, name).random(), name


def test_noun_outputs_are_unchanged_when_only_verb_settings_change(tmp_path: Path) -> None:
    off = generate(config_from_mapping({"scalars": {"count": 2}}))
    baseline = _noun_snapshot(off)
    variants = [
        VERBS,
        {"verbs": {"taxonomy": {"superordinates": 3, "depth": 3, "branching": 3}}},
        {"verbs": {"taxonomy": {"depth": 1, "superordinates": 5}}},
    ]
    for overrides in variants:
        result = generate(config_from_mapping({"scalars": {"count": 2}, **overrides}))
        assert result.verbs is not None
        assert _noun_snapshot(result) == baseline
    folder_off = off.write(tmp_path / "off")
    folder_on = generate(config_from_mapping({"scalars": {"count": 2}, **VERBS})).write(
        tmp_path / "on"
    )
    import polars as pl

    for file in folder_off.iterdir():
        if file.name == "config.yaml":
            continue
        if file.name == "summary.yaml":
            # With verbs on, the summary gains a verb block (stage 12) and nothing else.
            import yaml

            with_verbs = yaml.safe_load((folder_on / file.name).read_text())
            assert with_verbs.pop("verbs")
            assert with_verbs == yaml.safe_load(file.read_text())
            continue
        if file.name == "instances.csv":
            # With verbs on, instances.csv also carries the exposed projections (stage 11).
            on = pl.read_csv(folder_on / file.name)
            projections = [c for c in on.columns if c.startswith(("CAN.V", "CANBE.V"))]
            assert projections
            assert on.drop(projections).equals(pl.read_csv(file))
            continue
        assert file.read_bytes() == (folder_on / file.name).read_bytes(), file.name


def test_verb_settings_do_not_change_the_noun_streams() -> None:
    config_off = config_from_mapping({})
    config_on = config_from_mapping(VERBS)
    off = generate(config_off)
    on = generate(config_on)
    assert off.rules.records() == on.rules.records()
    assert off.instances.labels == on.instances.labels
    assert {k: v for k, v in on.summary.items() if k != "verbs"} == off.summary
    assert "verbs" in on.summary and "verbs" not in off.summary
