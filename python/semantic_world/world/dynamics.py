"""Precondition literals, condition literals, and effects: the dynamic terms an event type carries.

A precondition is a conjunction of fluent literals (``agent.BOOLFL.1 AND NOT patient.BOOLFL.2``),
each reading a base or derived fluent of one role. An effect sets one base fluent of one role
(``patient.BOOLFL.3 := 1``). Since stage b1 an effect may carry a condition
(``docs/specs/WORLD_AND_LANGUAGE.md``, "Conditional effects"): a conjunction of one or two
condition literals over the binding, each a static feature (base or derived) or a Boolean fluent
(base or derived) of one role, positive or negated, written ``when patient.PROPERTY.4 AND
patient.BOOLFL.2 then patient.BOOLFL.3 := 1``. The condition is judged in the step's starting
state; when it is false, the effect does nothing. This module holds the dataclasses and their
printed forms; their generation is in ``event_types.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ROLES = ("agent", "patient")

_LITERAL = re.compile(r"^(agent|patient)\.(BOOLFL\.\d+)$")
_CONDITION_LITERAL = re.compile(r"^(agent|patient)\.((?:PROPERTY|PART|BOOLFL)\.\d+)$")
_EFFECT = re.compile(r"^\s*(agent|patient)\.(BOOLFL\.\d+)\s*:=\s*([01])\s*$")
_CONDITIONAL = re.compile(r"^\s*when\s+(.+?)\s+then\s+(.+)$", re.DOTALL)
FEATURE_KIND = "feature"
FLUENT_KIND = "fluent"


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
class ConditionLiteral:
    """One literal of an effect's condition: a static feature or a fluent of one role, with the
    value it must have in the step's starting state."""

    role: str
    symbol: str
    value: bool

    @property
    def kind(self) -> str:
        """``fluent`` for a Boolean fluent, ``feature`` for a static feature."""
        return FLUENT_KIND if self.symbol.startswith("BOOLFL.") else FEATURE_KIND

    @property
    def key(self) -> str:
        return f"{self.role}.{self.symbol}"

    @property
    def text(self) -> str:
        return self.key if self.value else f"NOT {self.key}"

    def record(self) -> dict[str, Any]:
        return {"role": self.role, "kind": self.kind, "symbol": self.symbol, "value": self.value}

    @classmethod
    def parse_key(cls, key: str, value: bool) -> ConditionLiteral:
        match = _CONDITION_LITERAL.match(key)
        if match is None:
            raise ValueError(
                f"{key!r} is not a condition literal (agent.PROPERTY.<n>, patient.PART.<n>, or "
                f"agent.BOOLFL.<n>, with or without NOT)"
            )
        return cls(match.group(1), match.group(2), value)


def condition_text(literals: tuple[ConditionLiteral, ...]) -> str:
    return " AND ".join(lit.text for lit in literals)


def parse_condition(text: str) -> tuple[ConditionLiteral, ...]:
    """A conjunction of condition literals, ``agent.PROPERTY.4 AND NOT patient.BOOLFL.2``."""
    literals = []
    for part in text.split(" AND "):
        part = part.strip()
        value = True
        if part.upper().startswith("NOT "):
            value = False
            part = part[4:].strip()
        if not part:
            raise ValueError(f"{text!r} is not a condition: an empty literal")
        literals.append(ConditionLiteral.parse_key(part, value))
    return tuple(literals)


@dataclass(frozen=True, order=True)
class Effect:
    """One effect: a base fluent of one role is set to a value, under a condition when it has
    one (an empty tuple is no condition: the effect always applies)."""

    role: str
    fluent: str
    value: bool
    condition: tuple[ConditionLiteral, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.role}.{self.fluent}"

    @property
    def conditional(self) -> bool:
        return bool(self.condition)

    @property
    def unconditional(self) -> Effect:
        """The same effect without its condition."""
        return Effect(self.role, self.fluent, self.value)

    @property
    def assignment(self) -> str:
        return f"{self.key} := {int(self.value)}"

    @property
    def text(self) -> str:
        if not self.condition:
            return self.assignment
        return f"when {condition_text(self.condition)} then {self.assignment}"

    def record(self) -> dict[str, Any]:
        """The effect's record. An unconditional effect is written exactly as before stage b1,
        with no ``condition`` key, so that the identity of a world without conditions is
        unchanged."""
        entry: dict[str, Any] = {
            "role": self.role,
            "fluent": self.fluent,
            "value": self.value,
            "expression": self.text,
        }
        if self.condition:
            entry["condition"] = {
                "literals": [lit.record() for lit in self.condition],
                "expression": condition_text(self.condition),
            }
        return entry

    @classmethod
    def parse(cls, text: str) -> Effect:
        """``agent.BOOLFL.<n> := 1``, or ``when <condition> then <effect>``."""
        condition: tuple[ConditionLiteral, ...] = ()
        match = _CONDITIONAL.match(text)
        if match is not None:
            condition = parse_condition(match.group(1))
            text = match.group(2)
        plain = _EFFECT.match(text)
        if plain is None:
            raise ValueError(
                f"{text!r} is not an effect (write agent.BOOLFL.<n> := 0 or := 1, or when "
                f"<condition> then <effect>)"
            )
        return cls(plain.group(1), plain.group(2), plain.group(3) == "1", condition)


def precondition_text(literals: tuple[Literal, ...]) -> str:
    return " AND ".join(lit.text for lit in literals) if literals else "TRUE"
