"""Node vectors and the fixed-by-rule test.

Every category gets three vectors over all features, ISA (in category order) then IS, HAS, and
CAN (in index order):

1. the generative vector, which its children were generated from;
2. the defining vector, with the value of every feature fixed for all members and NaN elsewhere;
3. the mean vector over the instances below it.

A determined feature is fixed by rule at a category when its output is the same for every setting
of the free features in its cone that are not defining at the category. When the cone has at most
``CONE_ENUMERATION_LIMIT`` non-defining features, the test enumerates them, which is exact. Above
that, a local test treats each non-fixed input of the rule as independent. The local test never
marks a feature fixed when it is not, but it can miss a fixed feature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from semantic_world.common.boolean import settings_array
from semantic_world.taxonomy.config import ScalarsConfig
from semantic_world.taxonomy.instances import Instances
from semantic_world.taxonomy.rules import (
    CONE_ENUMERATION_LIMIT,
    RuleSet,
    Threshold,
    evaluate_feature,
)
from semantic_world.taxonomy.tree import Category, Tree

FIXED_NONE = 0
FIXED_EXACT = 1
FIXED_LOCAL = 2
FIXED_TEST_NAMES = {FIXED_NONE: "", FIXED_EXACT: "exact", FIXED_LOCAL: "local"}


@dataclass(frozen=True)
class NodeVectors:
    feature_labels: tuple[str, ...]
    """ISA labels in category order, then the non-ISA labels in matrix order."""
    isa_count: int
    generative: np.ndarray
    """Shape ``(categories, features)``, uint8."""
    defining: np.ndarray
    """Shape ``(categories, features)``, float, NaN where the feature is not fixed."""
    mean: np.ndarray
    """Shape ``(categories, features)``, float, NaN for a category with no instances."""
    fixed_by_rule: np.ndarray
    """Shape ``(categories, non-ISA features)``, bool: determined features fixed by rule."""
    fixed_test: np.ndarray
    """Shape ``(categories, non-ISA features)``, int8: which test marked the feature fixed."""
    scalar_labels: tuple[str, ...] = ()
    scalar_generative: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Shape ``(categories, scalars)``: each category's scalar values."""
    scalar_defining: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """The category's value where the scalar is fixed for its members (all drift below is 0),
    NaN otherwise."""
    scalar_mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """The mean scalar values over the instances below each category."""

    def category_defining(self, index: int) -> np.ndarray:
        return self.defining[index]


def all_feature_labels(rules: RuleSet, tree: Tree) -> tuple[str, ...]:
    return tuple(f"ISA.{c.label}" for c in tree.categories) + rules.features.labels


def compute_node_vectors(
    rules: RuleSet, tree: Tree, instances: Instances, scalars: ScalarsConfig | None = None
) -> NodeVectors:
    """The generative, defining, and mean vectors of every category. ``scalars`` is needed
    for the scalar columns when the feature set has scalar dimensions."""
    features = rules.features
    categories = tree.categories
    n_cat = len(categories)
    n_features = len(features)
    isa = tree.isa_matrix()
    generative = np.hstack([isa, tree.generative_matrix()]).astype(np.uint8)

    fixed = np.zeros((n_cat, n_features), dtype=bool)
    fixed_test = np.zeros((n_cat, n_features), dtype=np.int8)
    defining = np.full((n_cat, n_cat + n_features), np.nan)
    for ci, category in enumerate(categories):
        defining[ci, :n_cat] = _isa_defining(category, categories)
        mask = category.defining_mask()
        defining[ci, n_cat + features.free_positions[mask]] = category.free_values[mask]
        fixed[ci], fixed_test[ci] = fixed_by_rule(rules, category, scalars=scalars)
        positions = np.flatnonzero(fixed[ci])
        defining[ci, n_cat + positions] = category.values[positions]

    full = instances.full_matrix(tree).astype(float)
    mean = np.full((n_cat, n_cat + n_features), np.nan)
    for ci, category in enumerate(categories):
        below = instances.below(category, tree)
        if len(below):
            mean[ci] = full[below].mean(axis=0)

    n_scalars = features.scalar_count
    scalar_generative = tree.scalar_matrix() if n_scalars else np.zeros((n_cat, 0))
    scalar_defining = np.full((n_cat, n_scalars), np.nan)
    scalar_mean = np.full((n_cat, n_scalars), np.nan)
    if n_scalars:
        if scalars is None:
            raise ValueError("the scalar configuration is needed when there are scalar dimensions")
        for ci, category in enumerate(categories):
            if scalars.fixed_below(category.level):
                scalar_defining[ci] = category.scalars
            below = instances.below(category, tree)
            if len(below):
                scalar_mean[ci] = instances.scalars[below].mean(axis=0)

    return NodeVectors(
        all_feature_labels(rules, tree),
        n_cat,
        generative,
        defining,
        mean,
        fixed,
        fixed_test,
        features.scalar_labels,
        scalar_generative,
        scalar_defining,
        scalar_mean,
    )


def _isa_defining(category: Category, categories: tuple[Category, ...]) -> np.ndarray:
    """1 for the category and its ancestors, NaN for strict descendants, 0 otherwise."""
    own = category.indices
    row = np.zeros(len(categories))
    for j, other in enumerate(categories):
        theirs = other.indices
        if own[: len(theirs)] == theirs:
            row[j] = 1.0  # the category itself or an ancestor
        elif theirs[: len(own)] == own:
            row[j] = np.nan  # a strict descendant
    return row


def fixed_by_rule(
    rules: RuleSet,
    category: Category,
    cone_limit: int = CONE_ENUMERATION_LIMIT,
    scalars: ScalarsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Which determined features are fixed by rule at a category, and which test decided.

    Returns two arrays over the non-ISA features: a bool array (free features are always False)
    and an int8 array of ``FIXED_EXACT``, ``FIXED_LOCAL``, or ``FIXED_NONE``. Rules are visited in
    evaluation order, so the local test knows which lower-layer inputs are already fixed.

    A threshold literal is fixed at the category when every drift below the category's level,
    including the instance drift, is 0; it then has the category's own value. Otherwise the
    literal is free to vary. The exact test enumerates, for each scalar with open literals, the
    intervals between its thresholds, so two literals on one scalar are never given an impossible
    combination. The enumeration is exact when it has at most ``2 ** cone_limit`` rows.
    """
    features = rules.features
    free_index = {f.label: i for i, f in enumerate(features.free)}
    defining = category.defining_mask()
    by_output = {r.output.label: r for r in rules.rules}
    fixed = np.zeros(len(features), dtype=bool)
    test = np.zeros(len(features), dtype=np.int8)
    scalars_fixed = scalars is not None and scalars.fixed_below(category.level)

    def literal_is_fixed(t: Threshold) -> bool:
        return scalars_fixed

    def literal_value(t: Threshold) -> int:
        return int(category.scalars[t.scalar - 1] > t.threshold)

    for rule in rules.rules:
        output = rule.output
        cone = rules.cone(output)
        non_defining = [free_index[f.label] for f in cone if not defining[free_index[f.label]]]
        held_literals = {
            t.key: literal_value(t) for t in rules.thresholds_of(output) if literal_is_fixed(t)
        }
        open_groups = _open_threshold_groups(
            [t for t in rules.thresholds_of(output) if not literal_is_fixed(t)]
        )
        rows = 2 ** len(non_defining)
        for group in open_groups:
            rows *= len(group) + 1
        if rows <= 2**cone_limit:
            grid = settings_array(len(non_defining))
            if open_groups:
                intervals = np.array(
                    list(itertools.product(*[range(len(g) + 1) for g in open_groups])),
                    dtype=np.intp,
                )
            else:
                intervals = np.zeros((1, 0), dtype=np.intp)
            binary_index = np.repeat(np.arange(grid.shape[0]), intervals.shape[0])
            interval_index = np.tile(np.arange(intervals.shape[0]), grid.shape[0])
            free_values = np.tile(category.free_values, (rows, 1))
            free_values[:, non_defining] = grid[binary_index]
            literal_values: dict[str, np.ndarray] = {
                key: np.full(rows, value, dtype=np.uint8) for key, value in held_literals.items()
            }
            for gi, group in enumerate(open_groups):
                chosen = intervals[interval_index, gi]
                for j, t in enumerate(group):
                    literal_values[t.key] = (chosen > j).astype(np.uint8)
            out = evaluate_feature(output, free_values, features, by_output, {}, literal_values)
            if out.min() == out.max():
                fixed[output.position] = True
                test[output.position] = FIXED_EXACT
            continue
        # The local test: enumerate the inputs that are not fixed, holding the fixed ones.
        held = np.zeros(rule.arity, dtype=np.uint8)
        open_inputs: list[int] = []
        for k, feature in enumerate(rule.inputs):
            if isinstance(feature, Threshold):
                if literal_is_fixed(feature):
                    held[k] = literal_value(feature)
                else:
                    open_inputs.append(k)
            elif feature.free and defining[free_index[feature.label]]:
                held[k] = category.free_values[free_index[feature.label]]
            elif not feature.free and fixed[feature.position]:
                held[k] = category.values[feature.position]
            else:
                open_inputs.append(k)
        grid = settings_array(len(open_inputs))
        inputs = np.tile(held, (grid.shape[0], 1))
        inputs[:, open_inputs] = grid
        out = rule.table.evaluate(inputs)
        if out.min() == out.max():
            fixed[output.position] = True
            test[output.position] = FIXED_LOCAL
    return fixed, test


def _open_threshold_groups(thresholds: list[Threshold]) -> list[list[Threshold]]:
    """The open threshold literals grouped by scalar, each group sorted by threshold. A scalar
    value in interval ``i`` (0 to the group size) makes the first ``i`` literals true."""
    groups: dict[int, list[Threshold]] = {}
    for t in thresholds:
        groups.setdefault(t.scalar, []).append(t)
    return [sorted(groups[s], key=lambda t: t.threshold) for s in sorted(groups)]
