"""Statistics over relations: category proportions, sampled pairs, verb statistics, and
thematic relatedness.

A relation is a pure function of two instances' features, so the generator never stores every
pair. Exact evaluation covers every ordered pair of distinct instances (``Relations.matrix``).
When the number of ordered pairs exceeds ``verbs.pairs.max_exact_pairs``, category proportions and
verb statistics are estimated from a uniform sample of that many pairs, drawn from the
``taxonomy:pairs`` stream, and marked as estimates. Sampled true and false pairs also come from the
``taxonomy:pairs`` stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from semantic_world.taxonomy.config import Config
from semantic_world.taxonomy.constraints import Relations
from semantic_world.taxonomy.fixed import NodeVectors
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.similarity import similarity_matrix
from semantic_world.taxonomy.tree import Tree

PROPORTION_COLUMNS = [
    "verb",
    "level",
    "agent",
    "patient",
    "true_pairs",
    "total_pairs",
    "proportion",
    "estimated",
]
PAIR_COLUMNS = ["verb", "agent", "patient", "holds"]
VERB_STAT_COLUMNS = [
    "verb",
    "proportion_true",
    "agents",
    "patients",
    "constraints",
    "families",
    "symmetric_proportion",
    "within_leaf_proportion",
    "estimated",
]
THEMATIC_COLUMNS = ["leaf_a", "leaf_b", "thematic", "similarity"]


@dataclass(frozen=True)
class RelationStats:
    proportions: pl.DataFrame
    """``relation_proportions.csv``: one row per verb, level, and category pair with a
    proportion above 0."""
    pairs: pl.DataFrame
    """``relation_pairs.csv``: the sampled true and false pairs of every verb."""
    verb_stats: pl.DataFrame
    """``verb_stats.csv``: one row per verb."""
    thematic: pl.DataFrame
    """``thematic.csv``: one row per unordered pair of leaves with a score above 0."""
    estimated: bool
    """Whether proportions and verb statistics were estimated from a sample of pairs."""
    pairs_short: dict[str, dict[str, int]]
    """Verbs with fewer true or false pairs than requested, and how many they had."""


def level_assignment(tree: Tree, instances: Instances, level: int) -> tuple[list[str], np.ndarray]:
    """The categories at a level, and the index of each instance's ancestor at that level."""
    labels = [c.label for c in tree.at_level(level)]
    index = {label: i for i, label in enumerate(labels)}
    assignment = np.array(
        [
            index["C" + ".".join(str(i) for i in tree.categories[k].indices[:level])]
            for k in instances.leaf_index
        ],
        dtype=np.intp,
    )
    return labels, assignment


def compute_relation_stats(
    config: Config,
    tree: Tree,
    instances: Instances,
    relations: Relations,
    vectors: NodeVectors,
    rng: np.random.Generator,
) -> RelationStats:
    assert config.verbs is not None
    settings = config.verbs
    n = len(instances)
    total_pairs = n * (n - 1)
    estimated = total_pairs > settings.max_exact_pairs
    levels = {level: level_assignment(tree, instances, level) for level in range(1, tree.depth + 1)}
    proportion_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    pairs_short: dict[str, dict[str, int]] = {}
    thematic_scores: dict[tuple[int, int], float] = {}
    leaf_labels, leaf_assignment = levels[tree.depth]

    for verb in relations.verbs.verbs:
        relation = relations.relation(verb)
        if not estimated:
            matrix = relations.matrix(verb)
            sample = None
        else:
            matrix = None
            sample = _uniform_pairs(n, settings.max_exact_pairs, rng)
            sample = (sample[0], sample[1], relations.holds(verb, sample[0], sample[1]))

        # Category proportions at every level.
        for level, (labels, assignment) in levels.items():
            counts, totals = _category_counts(matrix, sample, assignment, len(labels), n)
            for a, p in zip(*np.nonzero(counts > 0), strict=True):
                proportion = counts[a, p] / totals[a, p]
                proportion_rows.append(
                    {
                        "verb": verb.label,
                        "level": level,
                        "agent": labels[a],
                        "patient": labels[p],
                        "true_pairs": int(round(counts[a, p])),
                        "total_pairs": int(totals[a, p]),
                        "proportion": float(proportion),
                        "estimated": estimated,
                    }
                )
                if level == tree.depth:
                    key = (min(a, p), max(a, p))
                    thematic_scores[key] = thematic_scores.get(key, 0.0) + float(proportion)

        # Sampled pairs.
        true_pairs, false_pairs, short = _sample_pairs(
            matrix, relations, verb, n, settings.sampled_true, settings.sampled_false, rng
        )
        if short:
            pairs_short[verb.label] = short
        for i, j in true_pairs:
            pair_rows.append(
                {
                    "verb": verb.label,
                    "agent": instances.labels[i],
                    "patient": instances.labels[j],
                    "holds": 1,
                }
            )
        for i, j in false_pairs:
            pair_rows.append(
                {
                    "verb": verb.label,
                    "agent": instances.labels[i],
                    "patient": instances.labels[j],
                    "holds": 0,
                }
            )

        # Verb statistics.
        stat_rows.append(
            _verb_statistics(
                verb.label, relation, matrix, sample, relations, verb, leaf_assignment, n, estimated
            )
        )

    # Thematic relatedness with the taxonomic similarity of the two leaves.
    leaf_rows = np.array([tree.categories.index(leaf) for leaf in tree.leaves], dtype=np.intp)
    columns = (
        np.arange(vectors.generative.shape[1])
        if config.analysis.similarity_features == "all"
        else np.arange(vectors.isa_count, vectors.generative.shape[1])
    )
    leaf_similarity = similarity_matrix(
        vectors.generative[leaf_rows][:, columns], config.analysis.similarity_metric
    )
    thematic_rows = [
        {
            "leaf_a": leaf_labels[a],
            "leaf_b": leaf_labels[b],
            "thematic": score,
            "similarity": float(leaf_similarity[a, b]),
        }
        for (a, b), score in sorted(thematic_scores.items())
        if score > 0
    ]

    return RelationStats(
        proportions=pl.DataFrame(proportion_rows, schema=_proportion_schema()),
        pairs=pl.DataFrame(
            pair_rows,
            schema={"verb": pl.Utf8, "agent": pl.Utf8, "patient": pl.Utf8, "holds": pl.Int64},
        ),
        verb_stats=pl.DataFrame(stat_rows, schema=_verb_stat_schema()),
        thematic=pl.DataFrame(
            thematic_rows,
            schema={
                "leaf_a": pl.Utf8,
                "leaf_b": pl.Utf8,
                "thematic": pl.Float64,
                "similarity": pl.Float64,
            },
        ),
        estimated=estimated,
        pairs_short=pairs_short,
    )


def _proportion_schema() -> dict[str, Any]:
    return {
        "verb": pl.Utf8,
        "level": pl.Int64,
        "agent": pl.Utf8,
        "patient": pl.Utf8,
        "true_pairs": pl.Int64,
        "total_pairs": pl.Int64,
        "proportion": pl.Float64,
        "estimated": pl.Boolean,
    }


def _verb_stat_schema() -> dict[str, Any]:
    return {
        "verb": pl.Utf8,
        "proportion_true": pl.Float64,
        "agents": pl.Int64,
        "patients": pl.Int64,
        "constraints": pl.Int64,
        "families": pl.Utf8,
        "symmetric_proportion": pl.Float64,
        "within_leaf_proportion": pl.Float64,
        "estimated": pl.Boolean,
    }


def _uniform_pairs(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``count`` ordered pairs of distinct instances, uniformly with replacement."""
    i = rng.integers(0, n, size=count)
    j = rng.integers(0, n - 1, size=count)
    j[j >= i] += 1
    return i, j


def _category_counts(
    matrix: np.ndarray | None,
    sample: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    assignment: np.ndarray,
    n_categories: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """True pairs and total pairs per (agent category, patient category). Exact from the pair
    matrix, or estimated from a sample (true pairs then scaled to the total)."""
    sizes = np.bincount(assignment, minlength=n_categories).astype(float)
    totals = np.outer(sizes, sizes) - np.diag(sizes)
    if matrix is not None:
        onehot = np.zeros((n, n_categories))
        onehot[np.arange(n), assignment] = 1.0
        counts = onehot.T @ matrix.astype(float) @ onehot
        return counts, totals
    assert sample is not None
    i, j, held = sample
    sampled = np.zeros((n_categories, n_categories))
    sampled_true = np.zeros((n_categories, n_categories))
    np.add.at(sampled, (assignment[i], assignment[j]), 1.0)
    np.add.at(sampled_true, (assignment[i], assignment[j]), held.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        proportion = np.where(sampled > 0, sampled_true / np.maximum(sampled, 1), 0.0)
    return proportion * totals, totals


def _sample_pairs(
    matrix: np.ndarray | None,
    relations: Relations,
    verb,
    n: int,
    want_true: int,
    want_false: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, int]]:
    short: dict[str, int] = {}
    if matrix is not None:
        off_diagonal = ~np.eye(n, dtype=bool)
        true_index = np.argwhere(matrix)
        false_index = np.argwhere(~matrix & off_diagonal)
        chosen = []
        for name, index, want in (
            ("true", true_index, want_true),
            ("false", false_index, want_false),
        ):
            if len(index) < want:
                short[name] = int(len(index))
            take = min(want, len(index))
            picks = (
                rng.choice(len(index), size=take, replace=False)
                if take
                else np.zeros(0, dtype=np.intp)
            )
            chosen.append([(int(index[k, 0]), int(index[k, 1])) for k in picks])
        return chosen[0], chosen[1], short
    # Estimation mode: draw random pairs until each class has enough, within a budget.
    true_pairs: dict[tuple[int, int], None] = {}
    false_pairs: dict[tuple[int, int], None] = {}
    budget = 20 * (want_true + want_false) + 10_000
    drawn = 0
    while (len(true_pairs) < want_true or len(false_pairs) < want_false) and drawn < budget:
        i, j = _uniform_pairs(n, 4096, rng)
        held = relations.holds(verb, i, j)
        drawn += len(i)
        for a, b, h in zip(i.tolist(), j.tolist(), held.tolist(), strict=True):
            target = true_pairs if h else false_pairs
            want = want_true if h else want_false
            if len(target) < want:
                target[(a, b)] = None
    if len(true_pairs) < want_true:
        short["true"] = len(true_pairs)
    if len(false_pairs) < want_false:
        short["false"] = len(false_pairs)
    return list(true_pairs), list(false_pairs), short


def _verb_statistics(
    label: str,
    relation,
    matrix: np.ndarray | None,
    sample,
    relations: Relations,
    verb,
    leaf_assignment: np.ndarray,
    n: int,
    estimated: bool,
) -> dict[str, Any]:
    families = ";".join(c.family for c in relation.constraints)
    if matrix is not None:
        total = n * (n - 1)
        true_count = int(matrix.sum())
        i, j = np.nonzero(matrix)
        reverse = matrix[j, i]
        proportion_true = true_count / total if total else math.nan
        agents = int(matrix.any(axis=1).sum())
        patients = int(matrix.any(axis=0).sum())
    else:
        i_all, j_all, held = sample
        proportion_true = float(held.mean()) if len(held) else math.nan
        i, j = i_all[held], j_all[held]
        reverse = relations.holds(verb, j, i)
        agents = int(len(np.unique(i)))
        patients = int(len(np.unique(j)))
    symmetric = float(reverse.mean()) if len(i) else math.nan
    within_leaf = float((leaf_assignment[i] == leaf_assignment[j]).mean()) if len(i) else math.nan
    return {
        "verb": label,
        "proportion_true": proportion_true,
        "agents": agents,
        "patients": patients,
        "constraints": len(relation.constraints),
        "families": families,
        "symmetric_proportion": symmetric,
        "within_leaf_proportion": within_leaf,
        "estimated": estimated,
    }
