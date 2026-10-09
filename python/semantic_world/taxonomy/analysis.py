"""Similarity statistics, feature statistics, and the run summary.

Similarity statistics use the configured metric over the non-ISA features (or all features). A
statistic with no pairs, or with only undefined similarities, is NaN. When a statistic has more
pairs than ``analysis.max_pairs``, the pairs are sampled from the ``taxonomy:analysis`` stream.
Feature statistics are computed over the instances.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from semantic_world.taxonomy.config import FEATURE_TYPES, Config
from semantic_world.taxonomy.fixed import NodeVectors
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import RuleSet
from semantic_world.taxonomy.similarity import (
    cross_similarity,
    rowwise_similarity,
    similarity_matrix,
)
from semantic_world.taxonomy.tree import Category, Role, Tree

ROLE_COLUMNS = (
    "defining_inherited",
    "defining_new",
    "characteristic",
    "undiagnostic",
    "fixed_by_rule",
)


# ---------------------------------------------------------------------------------------------
# Mean similarities, with pair sampling
# ---------------------------------------------------------------------------------------------


def _nanmean(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else math.nan


def mean_within_similarity(
    rows: np.ndarray, metric: str, max_pairs: int, rng: np.random.Generator
) -> float:
    """The mean similarity over all unordered pairs of rows, sampling when there are too many."""
    n = rows.shape[0]
    total = n * (n - 1) // 2
    if total == 0:
        return math.nan
    if total <= max_pairs:
        return _nanmean(similarity_matrix(rows, metric)[np.triu_indices(n, k=1)])
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n - 1, size=max_pairs)
    j[j >= i] += 1
    return _nanmean(rowwise_similarity(rows[i], rows[j], metric))


def mean_between_similarity(
    a: np.ndarray, b: np.ndarray, metric: str, max_pairs: int, rng: np.random.Generator
) -> float:
    """The mean similarity over all pairs of one row of ``a`` and one row of ``b``."""
    total = a.shape[0] * b.shape[0]
    if total == 0:
        return math.nan
    if total <= max_pairs:
        return _nanmean(cross_similarity(a, b, metric).ravel())
    i = rng.integers(0, a.shape[0], size=max_pairs)
    j = rng.integers(0, b.shape[0], size=max_pairs)
    return _nanmean(rowwise_similarity(a[i], b[j], metric))


# ---------------------------------------------------------------------------------------------
# Similarity statistics
# ---------------------------------------------------------------------------------------------


def similarity_table(
    config: Config,
    tree: Tree,
    instances: Instances,
    vectors: NodeVectors,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """One row per category: ``within``, ``between``, and ``instances`` similarities."""
    metric = config.analysis.similarity_metric
    max_pairs = config.analysis.max_pairs
    columns = _analysis_columns(config, vectors)
    category_rows = vectors.generative[:, columns]
    instance_rows = instances.full_matrix(tree)[:, columns]
    index = {c.label: i for i, c in enumerate(tree.categories)}

    def children_rows(category: Category) -> np.ndarray:
        if category.is_leaf:
            return instance_rows[instances.below(category, tree)]
        return category_rows[[index[c.label] for c in category.children]]

    def siblings(category: Category) -> list[Category]:
        if category.parent is None:
            return [c for c in tree.superordinates if c is not category]
        return [c for c in category.parent.children if c is not category]

    within = []
    between = []
    below_stat = []
    for category in tree.categories:
        own = children_rows(category)
        within.append(mean_within_similarity(own, metric, max_pairs, rng))
        others = [children_rows(s) for s in siblings(category)]
        other_rows = np.vstack(others) if others else np.zeros((0, own.shape[1]), dtype=own.dtype)
        between.append(mean_between_similarity(own, other_rows, metric, max_pairs, rng))
        below = instance_rows[instances.below(category, tree)]
        below_stat.append(mean_within_similarity(below, metric, max_pairs, rng))
    return pl.DataFrame(
        {
            "label": [c.label for c in tree.categories],
            "within": pl.Series(within, dtype=pl.Float64),
            "between": pl.Series(between, dtype=pl.Float64),
            "instances": pl.Series(below_stat, dtype=pl.Float64),
        }
    )


def _analysis_columns(config: Config, vectors: NodeVectors) -> np.ndarray:
    total = len(vectors.feature_labels)
    if config.analysis.similarity_features == "all":
        return np.arange(total)
    return np.arange(vectors.isa_count, total)


# ---------------------------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------------------------


def binary_entropy(p: float) -> float:
    """The entropy in bits of a Bernoulli variable with probability ``p``."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def mutual_information(x: np.ndarray, categories: np.ndarray, n_categories: int) -> float:
    """The mutual information in bits between a binary column and a category assignment."""
    n = len(x)
    if n == 0:
        return math.nan
    joint = np.bincount(categories * 2 + x, minlength=2 * n_categories).reshape(n_categories, 2) / n
    px = joint.sum(axis=0)
    pc = joint.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log2(joint / (pc[:, None] * px[None, :]))
    return float(np.nansum(np.where(joint > 0, terms, 0.0)))


def feature_stats_table(
    config: Config, rules: RuleSet, tree: Tree, instances: Instances, vectors: NodeVectors
) -> pl.DataFrame:
    """One row per feature (ISA first), with proportions, entropy, role counts, and the mutual
    information with the category at every level."""
    features = rules.features
    labels = vectors.feature_labels
    n_isa = vectors.isa_count
    full = instances.full_matrix(tree)
    n_instances = full.shape[0]

    types = ["isa"] * n_isa + [f.type for f in features.features]
    kinds = ["determined"] * n_isa + ["free" if f.free else "determined" for f in features.features]
    layers = [math.nan] * n_isa + [float(f.layer) for f in features.features]
    proportion = full.mean(axis=0) if n_instances else np.full(len(labels), math.nan)
    entropy = [binary_entropy(p) if not math.isnan(p) else math.nan for p in proportion]

    # Role counts over categories.
    role_counts = {name: [0] * len(labels) for name in ROLE_COLUMNS}
    roles = np.stack([c.roles for c in tree.categories]) if tree.categories else np.zeros((0, 0))
    for i, feature in enumerate(features.free):
        column = roles[:, i]
        role_counts["defining_inherited"][n_isa + feature.position] = int(
            (column == Role.DEFINING_INHERITED).sum()
        )
        role_counts["defining_new"][n_isa + feature.position] = int(
            (column == Role.DEFINING_NEW).sum()
        )
        role_counts["characteristic"][n_isa + feature.position] = int(
            (column == Role.CHARACTERISTIC).sum()
        )
        role_counts["undiagnostic"][n_isa + feature.position] = int(
            (column == Role.UNDIAGNOSTIC).sum()
        )
    fixed_counts = vectors.fixed_by_rule.sum(axis=0)
    for feature in features.determined:
        role_counts["fixed_by_rule"][n_isa + feature.position] = int(fixed_counts[feature.position])

    # Mutual information with the category at each level.
    mi_columns: dict[str, list[float]] = {}
    assignments: dict[int, np.ndarray] = {}
    for level in range(1, config.taxonomy.depth + 1):
        level_labels = [c.label for c in tree.at_level(level)]
        level_index = {label: i for i, label in enumerate(level_labels)}
        assignment = np.array(
            [
                level_index[tree.ancestor_label(tree.categories[k], level)]
                for k in instances.leaf_index
            ],
            dtype=np.intp,
        )
        assignments[level] = assignment
        mi_columns[f"mi_level_{level}"] = [
            mutual_information(full[:, j].astype(np.intp), assignment, len(level_labels))
            for j in range(len(labels))
        ]

    data: dict[str, Any] = {
        "feature": list(labels),
        "type": types,
        "kind": kinds,
        "layer": pl.Series(layers, dtype=pl.Float64),
        "proportion_true": pl.Series(list(proportion), dtype=pl.Float64),
        "entropy": pl.Series(entropy, dtype=pl.Float64),
    }
    data.update(role_counts)
    for name, values in mi_columns.items():
        data[name] = pl.Series(values, dtype=pl.Float64)
    if features.scalar_count:
        _add_scalar_stats(data, features.scalar_labels, instances.scalars, assignments)
    return pl.DataFrame(data)


def eta_squared(x: np.ndarray, categories: np.ndarray, n_categories: int) -> float:
    """The proportion of variance of ``x`` explained by a category assignment (η²)."""
    if len(x) == 0:
        return math.nan
    total = float(((x - x.mean()) ** 2).sum())
    if total == 0.0:
        return math.nan
    counts = np.bincount(categories, minlength=n_categories)
    sums = np.bincount(categories, weights=x, minlength=n_categories)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    between = float((counts * (means - x.mean()) ** 2).sum())
    return between / total


def _add_scalar_stats(
    data: dict[str, Any],
    scalar_labels: tuple[str, ...],
    scalar_values: np.ndarray,
    assignments: dict[int, np.ndarray],
) -> None:
    """Append one row per scalar and the scalar-only columns (mean, standard deviation, and
    η² per level). Binary-only columns are NaN for scalar rows (empty for the integer role
    counts), and scalar-only columns are NaN for binary rows."""
    n_binary = len(data["feature"])
    k = len(scalar_labels)
    data["feature"] = list(data["feature"]) + list(scalar_labels)
    data["type"] = list(data["type"]) + ["scalar"] * k
    data["kind"] = list(data["kind"]) + ["free"] * k
    data["layer"] = pl.Series(list(data["layer"]) + [0.0] * k, dtype=pl.Float64)
    data["proportion_true"] = pl.Series(
        list(data["proportion_true"]) + [math.nan] * k, dtype=pl.Float64
    )
    data["entropy"] = pl.Series(list(data["entropy"]) + [math.nan] * k, dtype=pl.Float64)
    for name in ROLE_COLUMNS:
        data[name] = pl.Series(list(data[name]) + [None] * k, dtype=pl.Int64)
    for name in [key for key in data if key.startswith("mi_level_")]:
        data[name] = pl.Series(list(data[name]) + [math.nan] * k, dtype=pl.Float64)
    n = scalar_values.shape[0]
    means = [float(scalar_values[:, j].mean()) if n else math.nan for j in range(k)]
    stds = [float(scalar_values[:, j].std()) if n else math.nan for j in range(k)]
    data["mean"] = pl.Series([math.nan] * n_binary + means, dtype=pl.Float64)
    data["std"] = pl.Series([math.nan] * n_binary + stds, dtype=pl.Float64)
    for level, assignment in assignments.items():
        n_groups = int(assignment.max()) + 1 if len(assignment) else 0
        values = [eta_squared(scalar_values[:, j], assignment, n_groups) for j in range(k)]
        data[f"eta2_level_{level}"] = pl.Series([math.nan] * n_binary + values, dtype=pl.Float64)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------


def _duplicates(rows: np.ndarray) -> int:
    seen: set[bytes] = set()
    count = 0
    for row in rows:
        key = row.tobytes()
        if key in seen:
            count += 1
        seen.add(key)
    return count


def summary_stats(
    rules: RuleSet, tree: Tree, instances: Instances, warnings: tuple[str, ...]
) -> dict[str, Any]:
    features = rules.features
    values = instances.values
    n = values.shape[0]
    mean_true = {}
    for feature_type in FEATURE_TYPES:
        positions = [f.position for f in features.of_type(feature_type)]
        mean_true[feature_type] = (
            float(values[:, positions].sum(axis=1).mean()) if n and positions else math.nan
        )
    constant = int((values.min(axis=0) == values.max(axis=0)).sum()) if n else 0
    leaf_rows = (
        np.stack([leaf.values for leaf in tree.leaves])
        if tree.leaves
        else np.zeros((0, 0), dtype=np.uint8)
    )
    similarity = tree.superordinate_similarity
    return {
        "categories": len(tree.categories),
        "leaves": len(tree.leaves),
        "instances": n,
        "mean_true_features_per_instance": mean_true,
        "constant_features": constant,
        "duplicate_leaves": _duplicates(leaf_rows),
        "duplicate_instances": _duplicates(values),
        "superordinates": {
            "similarity_min": None if similarity is None else similarity[0],
            "similarity_max": None if similarity is None else similarity[1],
            "tries": tree.superordinate_tries,
        },
        "warnings": list(warnings),
    }
