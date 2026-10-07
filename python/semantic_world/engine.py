"""The core API (contract 3): multi-agent, parallel, keyed by agent ID, no reward.

This module wraps the Rust engine (``semantic_world._core``). The world model of
``docs/specs/WORLD_AND_LANGUAGE.md`` lives in the package ``semantic_world.world``.
"""

from __future__ import annotations

import os
from typing import Any

from semantic_world import _core
from semantic_world.observation import Observation

Action = dict[str, Any]
"""An action: ``{"type": "move", "direction": 0.1, "speed": 0.5}``, with an optional
``"target"`` entity ID for agents that may target by ID."""


class World:
    """One experiment configuration, reset to a seed and stepped one decision at a time.

    ``step`` returns observations, events, and info, and never a reward: reward is computed on
    the agent's side (contract 4).
    """

    def __init__(self, config: str | os.PathLike[str], overrides: dict[str, Any] | None = None):
        self._core = _core.World(os.fspath(config), dict(overrides or {}))
        self.config_path = os.fspath(config)

    def reset(self, seed: int | None = None) -> tuple[dict[str, Observation], dict]:
        """Start a run. Returns ``(observations, info)``, observations keyed by agent ID."""
        raw, info = self._core.reset(seed)
        return _observations(raw), info

    def step(self, actions: dict[str, Action]) -> tuple[dict[str, Observation], list[dict], dict]:
        """Apply one action per awaiting agent and advance to the next decision point.

        An awaiting agent that is left out continues with ``noop``. Returns
        ``(observations, events, info)``.
        """
        raw, events, info = self._core.step(dict(actions))
        return _observations(raw), events, info

    @property
    def agents(self) -> list[str]:
        """Every agent's ID, in creation order."""
        return self._core.agents

    @property
    def agents_awaiting_action(self) -> list[str]:
        """The agents that must act before the next step."""
        return self._core.agents_awaiting_action

    @property
    def done(self) -> bool:
        return self._core.done

    @property
    def tick(self) -> int:
        return self._core.tick

    @property
    def time_s(self) -> float:
        return self._core.time_s

    @property
    def seed(self) -> int | None:
        """The master seed of the current run."""
        return self._core.seed

    def manifests(self) -> dict[str, dict]:
        """The sensor and action manifests of every agent: ``{agent: {"sensors", "actions"}}``."""
        return self._core.manifests()

    def agent_seed(self, agent_id: str) -> int:
        """The seed of the agent's ``agent:<id>`` random stream, as an integer.

        Python agents seed ``numpy.random.default_rng`` with it.
        """
        return int.from_bytes(self._core.agent_seed(agent_id), "little")

    def debug_state(self) -> dict:
        """The full canonical state.

        PRIVILEGED: only scripted agents, tests, and logs may call this. A learning agent
        must never read it.
        """
        return self._core.debug_state()

    def state_hash(self) -> str:
        """The state hash at this tick, as hex."""
        return self._core.state_hash()

    def config(self) -> dict:
        """The fully resolved experiment configuration."""
        return self._core.config()


def _observations(raw: dict[str, dict]) -> dict[str, Observation]:
    return {agent: Observation.from_core(obs) for agent, obs in raw.items()}


def make(config: str | os.PathLike[str], overrides: dict[str, Any] | None = None) -> World:
    """Load an experiment configuration into a world. ``overrides`` maps dotted paths to values."""
    return World(config, overrides)
