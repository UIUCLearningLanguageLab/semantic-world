"""Precondition literals and effects: the two kinds of dynamic term an event type carries.

A precondition is a conjunction of fluent literals (``agent.BOOLFL.1 AND NOT patient.BOOLFL.2``),
each reading a base or derived fluent of one role. An effect sets one base fluent of one role
(``patient.BOOLFL.3 := 1``). This module holds the two dataclasses and their printed forms; their
generation is in ``preconditions.py`` and ``effects.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ROLES = ("agent", "patient")

_LITERAL = re.compile(r"^(agent|patient)\.(BOOLFL\.\d+)$")
_EFFECT = re.compile(r"^\s*(agent|patient)\.(BOOLFL\.\d+)\s*:=\s*([01])\s*$")


@dataclass(frozen=True, order=True)
class Literal:
    """One fluent literal of a precondition."""

    role: str
    fluent: str
    value: bool

    @property
    def key(self) -> str:
        return f"{self.role}.{self.fluent}"

    @property
    def text(self) -> str:
        return self.key if self.value else f"NOT {self.key}"

    def record(self) -> dict[str, Any]:
        return {"role": self.role, "fluent": self.fluent, "value": self.value}

    @classmethod
    def parse_key(cls, key: str, value: bool) -> Literal:
        match = _LITERAL.match(key)
        if match is None:
            raise ValueError(
                f"{key!r} is not a fluent literal (agent.BOOLFL.<n> or patient.BOOLFL.<n>)"
            )
        return cls(match.group(1), match.group(2), value)


@dataclass(frozen=True, order=True)
class Effect:
    """One effect: a base fluent of one role is set to a value."""

    role: str
    fluent: str
    value: bool

    @property
    def key(self) -> str:
        return f"{self.role}.{self.fluent}"

    @property
    def text(self) -> str:
        return f"{self.key} := {int(self.value)}"

    def record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fluent": self.fluent,
            "value": self.value,
            "expression": self.text,
        }

    @classmethod
    def parse(cls, text: str) -> Effect:
        match = _EFFECT.match(text)
        if match is None:
            raise ValueError(f"{text!r} is not an effect (write agent.BOOLFL.<n> := 0 or := 1)")
        return cls(match.group(1), match.group(2), match.group(3) == "1")


def precondition_text(literals: tuple[Literal, ...]) -> str:
    return " AND ".join(lit.text for lit in literals) if literals else "TRUE"
