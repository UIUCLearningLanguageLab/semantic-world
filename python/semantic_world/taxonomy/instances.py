"""Instances: the individual objects at the leaves.

Each leaf gets a number of instances drawn from ``instances.per_leaf``. Instances receive no
roles of their own. Each instance follows the roles assigned at its leaf: a defining feature is
copied, a characteristic feature is copied with ``instances.characteristic_probability`` and
flipped otherwise, and an undiagnostic feature is drawn from its base rate. The determined
features are then computed from the rules. Everything here draws from the ``taxonomy:instances``
stream, so changing the instance count never changes the rules or the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_world.taxonomy.config import Config
from semantic_world.taxonomy.features import FeatureSet
from semantic_world.taxonomy.rules import RuleSet
from semantic_world.taxonomy.streams import Streams
from semantic_world.taxonomy.tree import Category, Role, Tree


@dataclass(frozen=True)
class Instances:
    labels: tuple[str, ...]
    """``I<leaf indices>.<k>``, in leaf order then index order."""
    leaf_labels: tuple[str, ...]
    """The leaf of each instance."""
    leaf_index: np.ndarray
    """The index of each instance's leaf in ``tree.categories``."""
    values: np.ndarray
    """The feature matrix over the non-ISA features, shape ``(instances, features)``, uint8."""
    scalars: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """The scalar values, shape ``(instances, scalars)``, float."""

    def __len__(self) -> int:
        return len(self.labels)

    def free_values(self, features: FeatureSet) -> np.ndarray:
        return self.values[:, features.free_positions]

    def below(self, category: Category, tree: Tree) -> np.ndarray:
        """The indices of the instances below a category, in instance order."""
        leaves = {tree.categories.index(leaf) for leaf in category.leaves()}
        return np.flatnonzero(np.isin(self.leaf_index, sorted(leaves)))

    def isa_matrix(self, tree: Tree) -> np.ndarray:
        """The ISA features of every instance: one column per category, in category order."""
        category_isa = tree.isa_matrix()
        return category_isa[self.leaf_index]

    def full_matrix(self, tree: Tree) -> np.ndarray:
        """ISA features (category order) followed by the non-ISA features (matrix order)."""
        return np.hstack([self.isa_matrix(tree), self.values])


def generate_instances(config: Config, rules: RuleSet, tree: Tree, streams: Streams) -> Instances:
    """Generate the instances of every leaf, in category order."""
    rng = streams.instances
    scalar_rng = streams.scalar_instances
    features = rules.features
    n_scalars = features.scalar_count
    instance_drift = config.scalars.instance_drift
    base_rates = features.base_rates
    copy_probability = config.instances.characteristic_probability
    per_leaf = config.instances.per_leaf
    labels: list[str] = []
    leaf_labels: list[str] = []
    leaf_index: list[int] = []
    free_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    for index, category in enumerate(tree.categories):
        if not category.is_leaf:
            continue
        count = int(rng.integers(per_leaf.min, per_leaf.max + 1))
        for k in range(1, count + 1):
            labels.append("I" + ".".join(str(i) for i in category.indices) + f".{k}")
            leaf_labels.append(category.label)
            leaf_index.append(index)
            free_rows.append(instance_free_values(category, copy_probability, base_rates, rng))
            if n_scalars:
                scalar_rows.append(
                    category.scalars + scalar_rng.normal(0.0, instance_drift, size=n_scalars)
                )
    free_matrix = (
        np.stack(free_rows) if free_rows else np.zeros((0, len(base_rates)), dtype=np.uint8)
    )
    scalar_matrix = (
        np.stack(scalar_rows) if scalar_rows else np.zeros((len(labels), n_scalars), dtype=float)
    )
    return Instances(
        tuple(labels),
        tuple(leaf_labels),
        np.array(leaf_index, dtype=np.intp),
        rules.compute(free_matrix),
        scalar_matrix,
    )


def instance_free_values(
    leaf: Category, copy_probability: float, base_rates: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One instance's free features from its leaf's generative vector and roles. One uniform
    draw per free feature decides the flip of a characteristic feature or the value of an
    undiagnostic one."""
    u = rng.random(len(base_rates))
    values = leaf.free_values.copy()
    characteristic = leaf.roles == Role.CHARACTERISTIC
    values[characteristic] ^= (u[characteristic] >= copy_probability).astype(np.uint8)
    undiagnostic = leaf.roles == Role.UNDIAGNOSTIC
    values[undiagnostic] = (u[undiagnostic] < base_rates[undiagnostic]).astype(np.uint8)
    return values
