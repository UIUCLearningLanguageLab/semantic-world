"""Capacities: the derived one-place facts of being able to take part in an event type
(``docs/specs/WORLD_AND_LANGUAGE.md``, "Rule-set identity and derived values").

``CAN.EVENTTYPE1.k`` is the one-place requirement evaluated on each entity. ``CAN.EVENTTYPE2.
<path>`` and ``CANBE.EVENTTYPE2.<path>`` are the agent and patient projections of a two-place
event type, intensional (some possible partner) with the approximate flag, followed by the
extensional capacities (some actual partner among the world's entities). ``capacity_roles.csv``
says, for each category and one-place event type, whether the capacity is fixed at 1, fixed at
0, or free, and which test decided. A capacity is a requirement only: it never looks at the state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from semantic_world.taxonomy.fixed import FIXED_TEST_NAMES
from semantic_world.world.statics import StaticWorld

CAPACITIES_FILE = "capacities.csv"
CAPACITY_ROLES_FILE = "capacity_roles.csv"
CAPACITY_ROLE_COLUMNS = ["category", "event_type", "status", "test"]


def capacities_frame(statics: StaticWorld) -> pl.DataFrame:
    """``derived/capacities.csv``."""
    instances = statics.instances
    data: dict[str, Any] = {"label": list(instances.labels)}
    for label in statics.unary.labels:
        data[f"CAN.{label}"] = statics.unary.column(label).astype(np.int64).tolist()
    projections = statics.projections
    if projections is not None:
        for name, column in projections.all_columns().items():
            if isinstance(column, str):
                data[name] = [column] * len(instances)
            else:
                data[name] = np.asarray(column).astype(np.int64).tolist()
    else:
        data["approximate"] = [""] * len(instances)
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "approximate": pl.Utf8})


def capacity_roles_frame(statics: StaticWorld) -> pl.DataFrame:
    """``derived/capacity_roles.csv``: one row per category and one-place event type."""
    unary = statics.unary
    rows: dict[str, list] = {name: [] for name in CAPACITY_ROLE_COLUMNS}
    for ci, category in enumerate(statics.taxonomy.tree.categories):
        for k, feature in enumerate(unary.event_features):
            if unary.fixed[ci, k]:
                value = unary.category_values[ci, feature.position]
                status = "fixed_1" if value else "fixed_0"
                test = FIXED_TEST_NAMES[int(unary.fixed_test[ci, k])]
            else:
                status, test = "free", ""
            rows["category"].append(category.label)
            rows["event_type"].append(feature.label)
            rows["status"].append(status)
            rows["test"].append(test)
    return pl.DataFrame(rows, schema={name: pl.Utf8 for name in CAPACITY_ROLE_COLUMNS})


__all__ = [
    "CAPACITIES_FILE",
    "CAPACITY_ROLES_FILE",
    "CAPACITY_ROLE_COLUMNS",
    "capacities_frame",
    "capacity_roles_frame",
]
