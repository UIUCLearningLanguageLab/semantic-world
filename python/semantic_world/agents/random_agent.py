"""The random agent: the floor. Chooses uniformly among action types, then draws uniform
arguments. With masking on, chooses only among unmasked action types."""

from __future__ import annotations

import numpy as np

from semantic_world.agents import Mind, register
from semantic_world.engine import Action
from semantic_world.observation import Observation


@register("random")
class RandomAgent(Mind):
    def __init__(self, manifests: dict, seed: int, **params):
        super().__init__(manifests, seed, **params)
        self.rng = np.random.default_rng(seed)
        self.actions = manifests["actions"]["actions"]

    def act(self, observation: Observation) -> Action:
        candidates = list(range(len(self.actions)))
        if observation.mask is not None:
            candidates = [i for i in candidates if observation.mask[i]]
        info = self.actions[int(self.rng.choice(candidates))]
        action: Action = {"type": info["name"]}
        for arg in info["args"]:
            if arg["kind"] == "continuous":
                lo, hi = arg["range"]
                action[arg["name"]] = float(self.rng.uniform(lo, hi))
            # Target slots are left egocentric: the nearest qualifying object in front.
        return action
