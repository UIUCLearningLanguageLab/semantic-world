"""Semantic World: an artificial world for comparing cognitive models.

The simulation core is written in Rust and compiled into the extension module
``semantic_world._core``. Later build stages add ``make()``, ``World``, ``VecWorld``,
``Observation``, the reward library, the agents, the adapters, and the batch runner.
See ``docs/specs/MILESTONE_1.md``.
"""

from semantic_world import _core

__version__: str = _core.__version__

__all__ = ["__version__"]
