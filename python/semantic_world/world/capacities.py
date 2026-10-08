"""Capacities: the derived one-place facts of being able to take part in an event type
(``docs/specs/WORLD_AND_LANGUAGE.md``, "Rule-set identity and derived values").

``CAN.EVENTTYPE1.k`` is the taxonomy's CAN feature. ``CAN.EVENTTYPE2.<path>`` and
``CANBE.EVENTTYPE2.<path>`` are the taxonomy's agent and patient projections, intensional
(some possible partner) with the approximate flag, followed by the extensional capacities (some
actual partner among the run's entities). ``capacity_roles.csv`` says, for each category and
one-place event type, whether the capacity is fixed at 1, fixed at 0, or free, and which test
decided. A capacity is a requirement only: it never looks at the state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from semantic_world.taxonomy.fixed import FIXED_TEST_NAMES
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.world.labels import capacity_label, translate

CAPACITIES_FILE = "capacities.csv"
CAPACITY_ROLES_FILE = "capacity_roles.csv"
CAPACITY_ROLE_COLUMNS = ["category", "event_type", "status", "test"]


def capacities_frame(taxonomy: TaxonomyResult) -> pl.DataFrame:
    """``derived/capacities.csv``."""
    instances = taxonomy.instances
    data: dict[str, Any] = {"label": [translate(label) for label in instances.labels]}
    can_features = [f for f in taxonomy.features.features if f.type == "can"]
    for feature in can_features:
        data[capacity_label(feature.label)] = (
            instances.values[:, feature.position].astype(np.int64).tolist()
        )
    projections = taxonomy.projections
    if projections is not None:
        intensional = {}
        extensional = {}
        for i, verb in enumerate(projections.verb_labels):
            intensional[capacity_label(f"CAN.{verb}")] = projections.agent[:, i]
            intensional[capacity_label(f"CANBE.{verb}")] = projections.patient[:, i]
            extensional[capacity_label(f"ACTUAL_CAN.{verb}")] = projections.actual_agent[:, i]
            extensional[capacity_label(f"ACTUAL_CANBE.{verb}")] = projections.actual_patient[:, i]
        for name, column in intensional.items():
            data[name] = np.asarray(column).astype(np.int64).tolist()
        approximate = ";".join(capacity_label(label) for label in projections.approximate_labels())
        data["approximate"] = [approximate] * len(instances)
        for name, column in extensional.items():
            data[name] = np.asarray(column).astype(np.int64).tolist()
    else:
        data["approximate"] = [""] * len(instances)
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "approximate": pl.Utf8})


def capacity_roles_frame(taxonomy: TaxonomyResult) -> pl.DataFrame:
    """``derived/capacity_roles.csv``: one row per category and one-place event type."""
    vectors = taxonomy.vectors
    rows: dict[str, list] = {name: [] for name in CAPACITY_ROLE_COLUMNS}
    can_features = [f for f in taxonomy.features.features if f.type == "can"]
    for ci, category in enumerate(taxonomy.tree.categories):
        for feature in can_features:
            value = vectors.defining[ci, vectors.isa_count + feature.position]
            if np.isnan(value):
                status, test = "free", ""
            else:
                status = "fixed_1" if value > 0.5 else "fixed_0"
                test = FIXED_TEST_NAMES[int(vectors.fixed_test[ci, feature.position])]
            rows["category"].append(translate(category.label))
            rows["event_type"].append(translate(feature.label))
            rows["status"].append(status)
            rows["test"].append(test)
    return pl.DataFrame(rows, schema={name: pl.Utf8 for name in CAPACITY_ROLE_COLUMNS})
