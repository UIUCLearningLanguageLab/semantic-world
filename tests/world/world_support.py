"""Shared helpers for the world package's tests.

Since stage c4a ``data/world/tiny.yaml`` carries the phase (c) settings (Jon's ruling WM.63):
two pair fluents between entities, a location fluent, the density band of the place
requirements, and the habitat strength 2. The tests of the earlier stages (the Boolean and
numeric dynamics, the conditional effects, the episodes, the places of stage c1) are about the
world without them, so they define the tiny world with the phase (c) settings put back to the
values they had before stage c4a (``PHASE_C_OFF``), which is the tiny world of stage c3b, and
the tests of stages c2b and c3b run on ``tiny.yaml`` as it is.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from semantic_world.world.config import Config, config_from_mapping

DATA = Path("data/world")
PHASE_C_OFF: dict[str, Any] = {
    "pair_fluents": {"count": 0, "locations": 0},
    "places": {"habitat_strength": 1},
    "place_event_types": {"density": None},
}
"""The phase (c) settings of ``tiny.yaml`` and ``default.yaml`` put back to their values before
stage c4a: no pair fluent and no location fluent, the habitat strength 1, and no density band of
the place requirements."""


def deep_update(data: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """``data`` with ``changes`` merged in, mapping by mapping; returns ``data``."""
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            deep_update(data[key], value)
        else:
            data[key] = copy.deepcopy(value)
    return data


def mapping_before_phase_c(name: str = "tiny", **overrides: Any) -> dict[str, Any]:
    """The mapping of ``data/world/<name>.yaml`` with the phase (c) settings off
    (``PHASE_C_OFF``) and ``overrides`` merged in after them, block by block."""
    data = yaml.safe_load((DATA / f"{name}.yaml").read_text(encoding="utf-8"))
    deep_update(data, PHASE_C_OFF)
    return deep_update(data, overrides)


def config_before_phase_c(name: str = "tiny", seed: int | None = None, **overrides: Any) -> Config:
    """The configuration of ``data/world/<name>.yaml`` with the phase (c) settings off: for
    ``tiny``, the tiny world of stage c3b."""
    return config_from_mapping(
        mapping_before_phase_c(name, **overrides), source=str(DATA / f"{name}.yaml"), seed=seed
    )
