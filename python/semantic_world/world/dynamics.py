"""Precondition literals, condition literals, and effects: the dynamic terms an event type carries.

A precondition is a conjunction of fluent literals (``agent.BOOLFL.1 AND NOT patient.BOOLFL.2``),
each reading a base or derived fluent of one role, or, since stage b2, a numeric threshold literal
(``agent.NUMFL.1 > 0.75``, or its negation ``agent.NUMFL.1 <= 0.75``). An effect sets one base
fluent of one role (``patient.BOOLFL.3 := 1``) or, since stage b2, adds a constant to a numeric
fluent of one role or subtracts one (``patient.NUMFL.1 += 0.3``). Since stage b1 an effect may
carry a condition (``docs/specs/WORLD_AND_LANGUAGE.md``, "Conditional effects"): a conjunction of
one or two condition literals over the binding, each a static feature (base or derived), a
Boolean fluent (base or derived), or a numeric threshold of one role, positive or negated,
written ``when patient.PROPERTY.4 AND patient.BOOLFL.2 then patient.BOOLFL.3 := 1``. The
condition is judged in the step's starting state; when it is false, the effect does nothing.
This module holds the dataclasses and their printed forms; their generation is in
``event_types.py``.

A numeric threshold is written in a text with ``repr``, the shortest form that reads back to the
same float64; the records of ``definition.json`` write every float with 17 significant digits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ROLES = ("agent", "patient")

_LITERAL = re.compile(r"^(agent|patient)\.(BOOLFL\.\d+)$")
_NUMERIC_LITERAL = re.compile(
    r"^\s*(agent|patient)\.(NUMFL\.\d+)\s*(>|<=)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_CONDITION_LITERAL = re.compile(r"^(agent|patient)\.((?:PROPERTY|PART|BOOLFL)\.\d+)$")
_EFFECT = re.compile(r"^\s*(agent|patient)\.(BOOLFL\.\d+)\s*:=\s*([01])\s*$")
_NUMERIC_EFFECT = re.compile(
    r"^\s*(agent|patient)\.(NUMFL\.\d+)\s*(\+=|-=)\s*((?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_CONDITIONAL = re.compile(r"^\s*when\s+(.+?)\s+then\s+(.+)$", re.DOTALL)
FEATURE_KIND = "feature"
FLUENT_KIND = "fluent"
NUMERIC_KIND = "numeric"
NUMERIC_PREFIX = "NUMFL."
ABOVE = ">"
AT_OR_BELOW = "<="


def _operator(above: bool) -> str:
    return ABOVE if above else AT_OR_BELOW


@dataclass(frozen=True, order=True)
class Literal:
    """One literal of a precondition: a Boolean fluent with the value it must have, or a numeric
    threshold (``threshold`` is not None): ``value`` True for ``> threshold``, False for
    ``<= threshold``."""

    role: str
    fluent: str
    value: bool
    threshold: float | None = None

    @property
    def numeric(self) -> bool:
        return self.threshold is not None

    @property
    def operator(self) -> str:
        return _operator(self.value)

    @property
    def key(self) -> str:
        return f"{self.role}.{self.fluent}"

    @property
    def text(self) -> str:
        if self.numeric:
            return f"{self.key} {self.operator} {self.threshold!r}"
        return self.key if self.value else f"NOT {self.key}"

    def holds_for(self, value: float) -> bool:
        """Whether a numeric literal holds for a value of its fluent."""
        assert self.threshold is not None
        return value > self.threshold if self.value else value <= self.threshold

    def record(self) -> dict[str, Any]:
        if self.numeric:
            return {
                "role": self.role,
                "fluent": self.fluent,
                "operator": self.operator,
                "threshold": float(self.threshold),  # type: ignore[arg-type]
            }
        return {"role": self.role, "fluent": self.fluent, "value": self.value}

    @classmethod
    def parse_key(cls, key: str, value: bool) -> Literal:
        match = _LITERAL.match(key)
        if match is None:
            raise ValueError(
                f"{key!r} is not a fluent literal (agent.BOOLFL.<n> or patient.BOOLFL.<n>)"
            )
        return cls(match.group(1), match.group(2), value)

    @classmethod
    def parse_numeric(cls, text: str) -> Literal:
        """``agent.NUMFL.1 > 0.75`` or ``agent.NUMFL.1 <= 0.75``."""
        match = _NUMERIC_LITERAL.match(text)
        if match is None:
            raise ValueError(
                f"{text!r} is not a numeric threshold literal (agent.NUMFL.<n> > 0.75, or <=)"
            )
        return cls(match.group(1), match.group(2), match.group(3) == ABOVE, float(match.group(4)))


@dataclass(frozen=True, order=True)
class ConditionLiteral:
    """One literal of an effect's condition: a static feature, a Boolean fluent, or a numeric
    threshold (``threshold`` is not None) of one role, with the value it must have in the step's
    starting state."""

    role: str
    symbol: str
    value: bool
    threshold: float | None = None

    @property
    def kind(self) -> str:
        """``numeric`` for a numeric threshold, ``fluent`` for a Boolean fluent, ``feature`` for
        a static feature."""
        if self.symbol.startswith(NUMERIC_PREFIX):
            return NUMERIC_KIND
        return FLUENT_KIND if self.symbol.startswith("BOOLFL.") else FEATURE_KIND

    @property
    def numeric(self) -> bool:
        return self.kind == NUMERIC_KIND

    @property
    def operator(self) -> str:
        return _operator(self.value)

    @property
    def key(self) -> str:
        return f"{self.role}.{self.symbol}"

    @property
    def text(self) -> str:
        if self.numeric:
            return f"{self.key} {self.operator} {self.threshold!r}"
        return self.key if self.value else f"NOT {self.key}"

    def holds_for(self, value: float) -> bool:
        assert self.threshold is not None
        return value > self.threshold if self.value else value <= self.threshold

    def record(self) -> dict[str, Any]:
        if self.numeric:
            return {
                "role": self.role,
                "kind": NUMERIC_KIND,
                "symbol": self.symbol,
                "operator": self.operator,
                "threshold": float(self.threshold),  # type: ignore[arg-type]
            }
        return {"role": self.role, "kind": self.kind, "symbol": self.symbol, "value": self.value}

    @classmethod
    def parse_key(cls, key: str, value: bool) -> ConditionLiteral:
        match = _CONDITION_LITERAL.match(key)
        if match is None:
            raise ValueError(
                f"{key!r} is not a condition literal (agent.PROPERTY.<n>, patient.PART.<n>, "
                f"agent.BOOLFL.<n>, with or without NOT, or agent.NUMFL.<n> > 0.75)"
            )
        return cls(match.group(1), match.group(2), value)

    @classmethod
    def parse(cls, text: str) -> ConditionLiteral:
        """One literal: ``agent.PROPERTY.4``, ``NOT patient.BOOLFL.2``, ``agent.NUMFL.1 > 0.75``,
        ``agent.NUMFL.1 <= 0.75``, or ``NOT agent.NUMFL.1 > 0.75``."""
        part = text.strip()
        value = True
        if part.upper().startswith("NOT "):
            value = False
            part = part[4:].strip()
        if not part:
            raise ValueError(f"{text!r} is not a condition literal: an empty literal")
        match = _NUMERIC_LITERAL.match(part)
        if match is not None:
            above = (match.group(3) == ABOVE) == value
            return cls(match.group(1), match.group(2), above, float(match.group(4)))
        return cls.parse_key(part, value)


def condition_text(literals: tuple[ConditionLiteral, ...]) -> str:
    return " AND ".join(lit.text for lit in literals)


def parse_condition(text: str) -> tuple[ConditionLiteral, ...]:
    """A conjunction of condition literals, ``agent.PROPERTY.4 AND NOT patient.BOOLFL.2 AND
    agent.NUMFL.1 > 0.75``."""
    return tuple(ConditionLiteral.parse(part) for part in text.split(" AND "))


@dataclass(frozen=True, order=True)
class Effect:
    """One effect: a base Boolean fluent of one role is set to a value, or (a numeric effect,
    on a ``NUMFL`` fluent) a constant is added to a numeric fluent of one role or subtracted
    from it, ``value`` being the signed constant; under a condition when the effect has one (an
    empty tuple is no condition: the effect always applies)."""

    role: str
    fluent: str
    value: bool | float
    condition: tuple[ConditionLiteral, ...] = ()

    @property
    def numeric(self) -> bool:
        return self.fluent.startswith(NUMERIC_PREFIX)

    @property
    def amount(self) -> float:
        """A numeric effect's signed constant."""
        return float(self.value)

    @property
    def operator(self) -> str:
        return "+=" if self.amount >= 0 else "-="

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
        if self.numeric:
            return f"{self.key} {self.operator} {abs(self.amount)!r}"
        return f"{self.key} := {int(self.value)}"

    @property
    def text(self) -> str:
        if not self.condition:
            return self.assignment
        return f"when {condition_text(self.condition)} then {self.assignment}"

    def record(self) -> dict[str, Any]:
        """The effect's record. A Boolean effect without a condition is written exactly as
        before stage b1, with no ``condition`` key, so that the identity of a world without
        conditions is unchanged; a numeric effect writes its ``operator`` and its ``amount`` in
        place of a ``value``."""
        entry: dict[str, Any]
        if self.numeric:
            entry = {
                "role": self.role,
                "fluent": self.fluent,
                "operator": self.operator,
                "amount": abs(self.amount),
                "expression": self.text,
            }
        else:
            entry = {
                "role": self.role,
                "fluent": self.fluent,
                "value": bool(self.value),
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
        """``agent.BOOLFL.<n> := 1``, ``patient.NUMFL.<n> += 0.3`` (or ``-=``), or ``when
        <condition> then <effect>``."""
        condition: tuple[ConditionLiteral, ...] = ()
        match = _CONDITIONAL.match(text)
        if match is not None:
            condition = parse_condition(match.group(1))
            text = match.group(2)
        plain = _EFFECT.match(text)
        if plain is not None:
            return cls(plain.group(1), plain.group(2), plain.group(3) == "1", condition)
        numeric = _NUMERIC_EFFECT.match(text)
        if numeric is not None:
            amount = float(numeric.group(4))
            if amount <= 0:
                raise ValueError(f"{text!r}: a numeric effect's constant is above 0")
            signed = amount if numeric.group(3) == "+=" else -amount
            return cls(numeric.group(1), numeric.group(2), signed, condition)
        raise ValueError(
            f"{text!r} is not an effect (write agent.BOOLFL.<n> := 0 or := 1, agent.NUMFL.<n> "
            f"+= 0.3 or -= 0.3, or when <condition> then <effect>)"
        )


def precondition_text(literals: tuple[Literal, ...]) -> str:
    return " AND ".join(lit.text for lit in literals) if literals else "TRUE"
