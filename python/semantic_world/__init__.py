"""Semantic World: an artificial world for comparing cognitive models.

The simulation core is written in Rust and compiled into the extension module
``semantic_world._core``. ``make()`` loads an experiment configuration into a ``World``, which
follows the agent-interface contract: observations, events, and info, and never a reward.
See ``docs/specs/MILESTONE_1.md``.

The Rust core loads lazily, on first use of ``World``, ``make``, ``Action``, ``agents``, or
``__version__``. The taxonomy generator (``semantic_world.taxonomy``) is pure Python and runs
without the core, with only NumPy, polars, and PyYAML installed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["Action", "Observation", "World", "__version__", "agents", "make"]

_LAZY_ATTRIBUTES = {
    "Action": "semantic_world.world",
    "World": "semantic_world.world",
    "make": "semantic_world.world",
    "Observation": "semantic_world.observation",
}
_LAZY_MODULES = {"agents": "semantic_world.agents", "_core": "semantic_world._core"}


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return importlib.import_module("semantic_world._core").__version__
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    if name in _LAZY_ATTRIBUTES:
        return getattr(importlib.import_module(_LAZY_ATTRIBUTES[name]), name)
    raise AttributeError(f"module 'semantic_world' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
