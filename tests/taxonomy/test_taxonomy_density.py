"""Stage 12a acceptance tests: the verb density range
(docs/proposals/2026-09-29-taxonomy-verb-density.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import (
    ConfigError,
    TaxonomyResult,
    config_from_mapping,
    generate,
    load_config,
)
from semantic_world.taxonomy.constraints import leaf_pair_density

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
GOLDEN_HASHES = json.loads(
    (Path(__file__).resolve().parent / "golden_relations_hashes.json").read_text()
)

CHECKS_OFF = {"density": None, "constraint_min_density": None}


def resolved_without_provenance(path: Path) -> dict:
    data = load_config(path).resolved()
    data.pop("provenance", None)
    return data


def with_verbs(base: dict, **verbs) -> dict:
    return {**base, "verbs": {**base["verbs"], **verbs}}


@pytest.fixture(scope="module")
def tiny() -> TaxonomyResult:
    return generate(load_config(DATA / "tiny_relations.yaml"))


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_density_defaults_and_null_forms() -> None:
    config = config_from_mapping({"verbs": {}})
    assert config.verbs.density is not None
    assert (config.verbs.density.min, config.verbs.density.max, config.verbs.density.max_tries) == (
        0.01,
        0.3,
        200,
    )
    assert config.verbs.constraint_min_density == 0.1
    resolved = config.resolved()["verbs"]
    assert list(resolved)[-2:] == ["density", "constraint_min_density"]
    assert resolved["density"] == {"min": 0.01, "max": 0.3, "max_tries": 200}
    off = config_from_mapping({"verbs": CHECKS_OFF})
    assert off.verbs.density is None and off.verbs.constraint_min_density is None
    assert off.resolved()["verbs"]["density"] is None
    assert config_from_mapping(config.resolved()).resolved() == config.resolved()


@pytest.mark.parametrize(
    ("verbs", "field"),
    [
        ({"density": {"min": 0.5, "max": 0.2}}, "verbs.density.min"),
        ({"density": {"max_tries": 0}}, "verbs.density.max_tries"),
        ({"density": {"colour": 1}}, "verbs.density.colour"),
        ({"density": {"min": -0.1}}, "verbs.density.min"),
        ({"constraint_min_density": 1.5}, "verbs.constraint_min_density"),
    ],
)
def test_broken_density_settings_name_the_field(verbs: dict, field: str) -> None:
    with pytest.raises(ConfigError) as error:
        config_from_mapping({"verbs": verbs})
    assert error.value.field == field


# ---------------------------------------------------------------------------------------------
# Brute force over leaf pairs
# ---------------------------------------------------------------------------------------------


def _brute_force_density(result: TaxonomyResult, constraints) -> float:
    leaves = result.tree.leaves
    held = 0
    for a in leaves:
        for p in leaves:
            ok = True
            for c in constraints:
                ok &= bool(
                    c.evaluate(
                        a.values[None, :], a.scalars[None, :], p.values[None, :], p.scalars[None, :]
                    )[0]
                )
            held += ok
    return held / (len(leaves) ** 2)


def test_leaf_pair_densities_match_brute_force_on_tiny(tiny: TaxonomyResult) -> None:
    relations = tiny.relations
    assert relations.constraint_density is not None and relations.verb_density is not None
    for constraint in relations.constraints:
        assert relations.constraint_density[constraint.label] == pytest.approx(
            _brute_force_density(tiny, [constraint])
        )
    for verb in tiny.verbs.verbs:
        relation = relations.relation(verb)
        assert relations.verb_density[verb.label] == pytest.approx(
            _brute_force_density(tiny, relation.constraints)
        )
    # The measure uses leaves paired with themselves too, and never the instances.
    leaves = tiny.tree.leaves
    values = np.stack([leaf.values for leaf in leaves])
    scalars = np.stack([leaf.scalars for leaf in leaves])
    for verb in tiny.verbs.verbs:
        assert (
            leaf_pair_density(relations.relation(verb).constraints, values, scalars)
            == relations.verb_density[verb.label]
        )


def test_densities_are_reported_in_the_outputs(tiny: TaxonomyResult, tmp_path: Path) -> None:
    folder = tiny.write(tmp_path / "run")
    stats = pl.read_csv(folder / "verb_stats.csv")
    assert stats.columns[-2:] == ["leaf_pair_density", "tries"]
    for row in stats.iter_rows(named=True):
        assert row["leaf_pair_density"] == pytest.approx(
            tiny.relations.verb_density[row["verb"]], abs=1e-6
        )
        assert 1 <= row["tries"] <= 200
    constraints = yaml.safe_load((folder / "constraints.yaml").read_text())
    assert all("leaf_pair_density" in c for c in constraints)
    summary = yaml.safe_load((folder / "summary.yaml").read_text())
    assert summary["verbs"]["verbs_outside_density"] == len(tiny.relations.outside_range)
    assert summary["verbs"]["verbs_outside_density_labels"] == list(tiny.relations.outside_range)
    for label in tiny.relations.outside_range:
        assert any(label in w for w in summary["warnings"])


# ---------------------------------------------------------------------------------------------
# The range across seeds
# ---------------------------------------------------------------------------------------------


def test_every_verb_is_inside_the_range_or_warned_across_ten_seeds() -> None:
    inside = total = 0
    proportions = []
    for seed in range(10):
        result = generate(load_config(DATA / "relations.yaml", seed=seed))
        relations = result.relations
        density = result.config.verbs.density
        n = len(result.instances)
        for verb in result.verbs.verbs:
            total += 1
            value = relations.verb_density[verb.label]
            if density.min <= value <= density.max:
                inside += 1
                assert verb.label not in relations.outside_range
            else:
                assert verb.label in relations.outside_range
                assert any(verb.label in w and f"{value:.4f}" in w for w in relations.warnings)
            proportions.append(result.relations.matrix(verb).sum() / (n * (n - 1)))
        # The floor applies to the verb-feature constraints, not to a verb's own constraint.
        for constraint in relations.feature_constraints:
            value = relations.constraint_density[constraint.label]
            if value < result.config.verbs.constraint_min_density:
                assert any(constraint.label in w for w in relations.warnings)
    share = inside / total
    p = np.array(proportions)
    print(
        f"\nAFTER: verbs {total}; inside range {share:.0%}; zero pairs {np.mean(p == 0):.0%}; "
        f"below 0.5% {np.mean(p < 0.005):.0%}; median {np.median(p):.2%}; "
        f"above 5% {np.mean(p > 0.05):.0%}"
    )
    assert share > 0.5
    assert np.mean(p == 0) < 0.15


# ---------------------------------------------------------------------------------------------
# Independence and the unchanged build with the checks off
# ---------------------------------------------------------------------------------------------


def test_instance_count_leaves_constraints_relations_and_verb_outputs_unchanged(
    tmp_path: Path,
) -> None:
    base = resolved_without_provenance(DATA / "relations.yaml")
    a = generate(config_from_mapping({**base, "instances": {"per_leaf": 3}})).write(tmp_path / "a")
    b = generate(config_from_mapping({**base, "instances": {"per_leaf": [6, 9]}})).write(
        tmp_path / "b"
    )
    for name in (
        "constraints.yaml",
        "relations.yaml",
        "verb_features.csv",
        "verb_tree.csv",
        "verb_roles.csv",
        "verbs_generative.csv",
        "verbs_defining.csv",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    stats_a = pl.read_csv(a / "verb_stats.csv").select(
        ["verb", "constraints", "families", "leaf_pair_density", "tries"]
    )
    stats_b = pl.read_csv(b / "verb_stats.csv").select(
        ["verb", "constraints", "families", "leaf_pair_density", "tries"]
    )
    assert stats_a.equals(stats_b)


def test_verb_settings_still_leave_noun_outputs_unchanged(tmp_path: Path) -> None:
    base = resolved_without_provenance(DATA / "relations.yaml")
    on = generate(config_from_mapping(base))
    off = generate(config_from_mapping(with_verbs(base, **CHECKS_OFF)))
    none = generate(config_from_mapping({k: v for k, v in base.items() if k != "verbs"}))
    for result in (on, off):
        assert result.rules.records() == none.rules.records()
        assert np.array_equal(result.instances.values, none.instances.values)
        assert np.array_equal(result.instances.scalars, none.instances.scalars)
        assert np.array_equal(result.tree.generative_matrix(), none.tree.generative_matrix())


@pytest.mark.parametrize("name", ["relations", "tiny_relations"])
def test_with_both_checks_off_the_output_equals_the_stage_12_output(
    name: str, tmp_path: Path
) -> None:
    base = resolved_without_provenance(DATA / f"{name}.yaml")
    folder = generate(config_from_mapping(with_verbs(base, **CHECKS_OFF))).write(tmp_path / name)
    expected = GOLDEN_HASHES[name]
    produced = {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file() and p.name != "config.yaml"
    }
    assert produced == expected


def test_checks_on_change_the_verbs_but_keep_them_distinct() -> None:
    base = resolved_without_provenance(DATA / "relations.yaml")
    on = generate(config_from_mapping(base))
    off = generate(config_from_mapping(with_verbs(base, **CHECKS_OFF)))
    assert on.relations.records() != off.relations.records()
    vectors = [v.values.tobytes() for v in on.verbs.verbs]
    assert len(set(vectors)) == len(vectors)
    # Defining features never change under a redraw.
    for verb in on.verbs.verbs:
        for ancestor in verb.ancestors():
            mask = ancestor.defining_mask()
            assert np.array_equal(verb.free_values[mask], ancestor.free_values[mask])
    assert on.relations.verb_density is not None and off.relations.verb_density is None
    assert off.relations.constraint_density is None
    assert "verbs_outside_density" not in off.summary["verbs"]


def test_same_seed_gives_the_same_tuned_result(tmp_path: Path) -> None:
    a = generate(load_config(DATA / "tiny_relations.yaml")).write(tmp_path / "a")
    b = generate(load_config(DATA / "tiny_relations.yaml")).write(tmp_path / "b")
    for file in a.rglob("*"):
        if file.is_file() and file.name != "config.yaml":
            relative = file.relative_to(a)
            assert file.read_bytes() == (b / relative).read_bytes(), str(relative)


def test_constraint_floor_alone() -> None:
    base = resolved_without_provenance(DATA / "relations.yaml")
    result = generate(
        config_from_mapping(with_verbs(base, density=None, constraint_min_density=0.2))
    )
    relations = result.relations
    assert relations.constraint_density is not None and relations.verb_density is None
    for constraint in relations.feature_constraints:
        value = relations.constraint_density[constraint.label]
        assert value >= 0.2 or any(constraint.label in w for w in relations.warnings)
    assert "verbs_outside_density" not in result.summary["verbs"]
    assert "leaf_pair_density" not in result.relation_stats.verb_stats.columns
