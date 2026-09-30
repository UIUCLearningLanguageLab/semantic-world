"""Observations: named blocks, one per sensor, plus the flat vector (contract 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """What one agent sees at one decision point.

    ``blocks`` holds one NumPy array per sensor. ``vector`` concatenates every non-image block
    in manifest order; images are never in the vector. ``props`` holds the propositional
    sensor's facts when that sensor is on, ``mask`` the action mask when impossible actions are
    masked, and ``text`` heard speech as symbols, which milestone 1 does not produce.
    """

    blocks: dict[str, np.ndarray]
    vector: np.ndarray
    props: list[str] | None = None
    text: str | None = None
    mask: np.ndarray | None = None
    _extra: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_core(cls, raw: dict) -> Observation:
        return cls(
            blocks=dict(raw["blocks"]),
            vector=raw["vector"],
            props=raw["props"],
            text=raw["text"],
            mask=raw["mask"],
        )
