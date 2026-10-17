"""Comparison classes: ``derived/comparison_classes.csv`` (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Phase (b): the world", stage b2).

For each category of the taxonomy's tree, for ``ENTITY`` (every entity, the comparison class of
a top-level category, which replaces the corpus's ``THING``), and for each scalar dimension, the
table holds the number of instances below the category, the mean of the scalar over them, and
its population standard deviation (NumPy's default), computed as the corpus computes the class of
a pole adjective. The runtime computes the same figures from the leaf labels alone
(``runtime.comparison_classes``); the two agree on every row, and so does the corpus's own
computation, which stage b3 replaces by this file. Since stage c1 a world with places adds, after
the entities' rows, ``EVERYPLACE`` (every place, WM.49) and every category of places, on every
place scalar, computed in the same way.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from semantic_world.taxonomy.generate import TaxonomyResult

COMPARISON_CLASSES_FILE = "comparison_classes.csv"
ENTITY = "ENTITY"
EVERYPLACE = "EVERYPLACE"


def _class_rows(taxonomy: TaxonomyResult, everything: str, rows: dict[str, list]) -> None:
    instances = taxonomy.instances
    scalars = taxonomy.features.scalar_labels
    leaf_labels = np.array(instances.leaf_labels)
    below: dict[str, np.ndarray] = {everything: np.arange(len(instances))}
    for category in taxonomy.tree.categories:
        leaves = {leaf.label for leaf in category.leaves()}
        below[category.label] = np.flatnonzero(np.isin(leaf_labels, list(leaves)))
    for category, members in below.items():
        for j, scalar in enumerate(scalars):
            column = instances.scalars[members, j]
            rows["category"].append(category)
            rows["scalar"].append(scalar)
            rows["instances"].append(int(members.size))
            rows["mean"].append(float(column.mean()) if members.size else 0.0)
            rows["sd"].append(float(column.std()) if members.size else 0.0)


def comparison_classes_frame(
    taxonomy: TaxonomyResult, places: TaxonomyResult | None = None
) -> pl.DataFrame:
    """One row per category (``ENTITY`` first, then the tree's categories in tree order) and
    scalar: ``category``, ``scalar``, ``instances``, ``mean``, ``sd``; then, with a place
    taxonomy, ``EVERYPLACE`` and the place tree's categories on every place scalar."""
    rows: dict[str, list] = {"category": [], "scalar": [], "instances": [], "mean": [], "sd": []}
    _class_rows(taxonomy, ENTITY, rows)
    if places is not None:
        _class_rows(places, EVERYPLACE, rows)
    return pl.DataFrame(rows, schema_overrides={"category": pl.Utf8, "scalar": pl.Utf8})


__all__ = ["COMPARISON_CLASSES_FILE", "ENTITY", "EVERYPLACE", "comparison_classes_frame"]
