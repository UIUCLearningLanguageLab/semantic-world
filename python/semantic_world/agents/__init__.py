"""Nervous systems: the cognitive models placed in agents, registered by name."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from semantic_world.observation import Observation
from semantic_world.world import Action


class NervousSystem(ABC):
    """The interface every cognitive model meets.

    ``manifests`` holds the agent's sensor and action manifests, as ``World.manifests()``
    gives them for one agent. ``seed`` is the agent's random seed from ``World.agent_seed``.
    """

    def __init__(self, manifests: dict, seed: int, **params: Any):
        self.manifests = manifests
        self.seed = seed
        self.params = params

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Choose an action from an observation."""

    def learn(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
    ) -> None:
        """Update from experience. Optional."""

    def report_internals(self) -> dict[str, Any]:
        """Named arrays for logging and the viewer. Optional."""
        return {}

    def save(self, path: str) -> None:
        """Write a checkpoint. Optional."""

    def load(self, path: str) -> None:
        """Read a checkpoint. Optional."""


_REGISTRY: dict[str, type[NervousSystem]] = {}


def register(name: str) -> Callable[[type[NervousSystem]], type[NervousSystem]]:
    """Register a nervous system under the name experiment configurations use."""

    def decorator(cls: type[NervousSystem]) -> type[NervousSystem]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def names() -> list[str]:
    return sorted(_REGISTRY)


def create(name: str, manifests: dict, seed: int, **params: Any) -> NervousSystem:
    """Build the nervous system registered under ``name``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no nervous system named {name!r}; known: {names()}") from None
    return cls(manifests, seed, **params)


from semantic_world.agents import random_agent as _random_agent  # noqa: E402  (registers)

__all__ = ["NervousSystem", "create", "names", "register"]
_ = _random_agent
