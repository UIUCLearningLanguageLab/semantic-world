"""Semantic World: an artificial world for comparing cognitive models.

The simulation core is written in Rust and compiled into the extension module
``semantic_world._core``. ``make()`` loads an experiment configuration into a ``World``, which
follows the agent-interface contract: observations, events, and info, and never a reward.
See ``docs/specs/MILESTONE_1.md``.
"""

from semantic_world import _core, agents
from semantic_world.observation import Observation
from semantic_world.world import Action, World, make

__version__: str = _core.__version__

__all__ = ["Action", "Observation", "World", "__version__", "agents", "make"]
