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

Since stage c2a (``docs/specs/WORLD_PHASE_C.md``, "Literals over pairs") a precondition or a
condition may hold a pair literal, :class:`PairLiteral`: ``PAIRFL.1(agent, patient)`` (the pair
holds), ``SOME.PAIRFL.1(agent)`` (the row of the role holds a pair), ``SOMEOF.PAIRFL.1(patient)``
(the column does), or ``SAME.PAIRFL.3(agent, patient)`` (the two roles share a partner, on a
functional pair fluent), each positive or negated; a compiled literal (F4) carries its origin,
``profile``. An effect may add or delete a pair between the roles: ``PAIRFL.1(agent, patient)
:= 1`` or ``:= 0`` (an :class:`Effect` whose ``second`` names the second role).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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
_PAIR_LITERAL = re.compile(
    r"^\s*(?:(SOME|SOMEOF|SAME)\.)?(PAIRFL\.\d+)\s*\(\s*(agent|patient)\s*"
    r"(?:,\s*(agent|patient)\s*)?\)\s*$"
)
_PAIR_EFFECT = re.compile(
    r"^\s*(PAIRFL\.\d+)\s*\(\s*(agent|patient)\s*,\s*(agent|patient)\s*\)\s*:=\s*([01])\s*$"
)
FEATURE_KIND = "feature"
FLUENT_KIND = "fluent"
NUMERIC_KIND = "numeric"
NUMERIC_PREFIX = "NUMFL."
PAIR_PREFIX = "PAIRFL."
ABOVE = ">"
AT_OR_BELOW = "<="
PAIR_KIND = "pair"
PAIR_SOME = "pair_some"
PAIR_SOMEOF = "pair_someof"
PAIR_SAME = "pair_same"
PAIR_KINDS = (PAIR_KIND, PAIR_SOME, PAIR_SOMEOF, PAIR_SAME)
"""The kinds of pair literal (stage c2a), as the literal table, the precondition records, and
the condition records name them."""
PAIR_KIND_OF_PREFIX = {None: PAIR_KIND, "SOME": PAIR_SOME, "SOMEOF": PAIR_SOMEOF, "SAME": PAIR_SAME}
PAIR_PREFIX_OF_KIND = {kind: prefix for prefix, kind in PAIR_KIND_OF_PREFIX.items()}
TWO_ARGUMENT_KINDS = (PAIR_KIND, PAIR_SAME)
PROFILE_ORIGIN = "profile"


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
    def pair(self) -> bool:
        return False

    @property
    def compiled(self) -> bool:
        return False

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
    def pair(self) -> bool:
        return False

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


@dataclass(frozen=True, order=True)
class PairLiteral:
    """One pair literal (stage c2a) of a precondition or a condition: ``kind`` is ``pair``,
    ``pair_some``, ``pair_someof``, or ``pair_same``; ``first`` is the role of the first argument
    (the only one of ``SOME`` and ``SOMEOF``) and ``second`` the role of the second argument, or
    None; ``value`` is True for the positive literal and False for its negation; ``origin`` is
    None for a literal the file or the generator gave, and ``profile`` for a compiled literal
    (F4). ``role`` and ``fluent`` name the first role and the pair fluent, so that code that
    counts literals by role reads a pair literal as the other kinds."""

    kind: str
    pair_fluent: str
    first: str
    second: str | None
    value: bool
    origin: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in PAIR_KINDS:
            raise ValueError(f"unknown pair literal kind {self.kind!r}")
        two = self.kind in TWO_ARGUMENT_KINDS
        if two and self.second is None:
            raise ValueError(f"a {self.kind} literal names two roles")
        if not two and self.second is not None:
            raise ValueError(f"a {self.kind} literal names one role")
        if two and self.first == self.second:
            raise ValueError(f"a {self.kind} literal names two different roles")

    @property
    def role(self) -> str:
        return self.first

    @property
    def fluent(self) -> str:
        return self.pair_fluent

    @property
    def symbol(self) -> str:
        return self.pair_fluent

    @property
    def numeric(self) -> bool:
        return False

    @property
    def pair(self) -> bool:
        return True

    @property
    def compiled(self) -> bool:
        return self.origin is not None

    @property
    def roles(self) -> tuple[str, ...]:
        return (self.first,) if self.second is None else (self.first, self.second)

    @property
    def positive(self) -> str:
        """The literal's text without a negation: ``SAME.PAIRFL.3(agent, patient)``."""
        prefix = PAIR_PREFIX_OF_KIND[self.kind]
        head = f"{prefix}.{self.pair_fluent}" if prefix else self.pair_fluent
        return f"{head}({', '.join(self.roles)})"

    @property
    def key(self) -> str:
        return self.positive

    @property
    def text(self) -> str:
        return self.positive if self.value else f"NOT {self.positive}"

    def negated(self) -> PairLiteral:
        return PairLiteral(self.kind, self.pair_fluent, self.first, self.second, not self.value)

    def plain(self) -> PairLiteral:
        """The same literal without its origin."""
        return PairLiteral(self.kind, self.pair_fluent, self.first, self.second, self.value)

    def record(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"kind": self.kind, "pair_fluent": self.pair_fluent}
        if self.second is None:
            entry["role"] = self.first
        else:
            entry["first"] = self.first
            entry["second"] = self.second
        entry["value"] = bool(self.value)
        if self.origin is not None:
            entry["origin"] = self.origin
        return entry

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> PairLiteral:
        kind = str(data["kind"])
        if "role" in data and "first" not in data:
            first, second = str(data["role"]), None
        else:
            first, second = str(data["first"]), str(data["second"])
        origin = data.get("origin")
        return cls(
            kind,
            str(data["pair_fluent"]),
            first,
            second,
            bool(data["value"]),
            None if origin is None else str(origin),
        )

    @classmethod
    def match(cls, text: str, value: bool = True) -> PairLiteral | None:
        """The pair literal a text names, ``SOME.PAIRFL.1(agent)`` or ``PAIRFL.2(patient,
        agent)``, with the given value, or None when the text is not one."""
        match = _PAIR_LITERAL.match(text)
        if match is None:
            return None
        kind = PAIR_KIND_OF_PREFIX[match.group(1)]
        first, second = match.group(3), match.group(4)
        two = kind in TWO_ARGUMENT_KINDS
        if two and second is None:
            raise ValueError(f"{text.strip()!r}: a {kind} literal names two roles")
        if not two and second is not None:
            raise ValueError(f"{text.strip()!r}: a {kind} literal names one role")
        if two and first == second:
            raise ValueError(f"{text.strip()!r}: a {kind} literal names two different roles")
        return cls(kind, match.group(2), first, second, value)

    @classmethod
    def parse(cls, text: str) -> PairLiteral:
        """A pair literal with an optional ``NOT``."""
        part = text.strip()
        value = True
        if part.upper().startswith("NOT "):
            value = False
            part = part[4:].strip()
        literal = cls.match(part, value)
        if literal is None:
            raise ValueError(
                f"{text!r} is not a pair literal (PAIRFL.<n>(agent, patient), "
                f"SOME.PAIRFL.<n>(agent), SOMEOF.PAIRFL.<n>(patient), or "
                f"SAME.PAIRFL.<n>(agent, patient), with or without NOT)"
            )
        return literal


AnyLiteral = Literal | PairLiteral
"""A precondition literal: a Boolean or numeric literal, or, since stage c2a, a pair literal."""
AnyConditionLiteral = ConditionLiteral | PairLiteral


def is_pair_literal(literal: Any) -> bool:
    return isinstance(literal, PairLiteral)


def condition_text(literals: tuple[AnyConditionLiteral, ...]) -> str:
    return " AND ".join(lit.text for lit in literals)


def parse_condition(text: str) -> tuple[AnyConditionLiteral, ...]:
    """A conjunction of condition literals, ``agent.PROPERTY.4 AND NOT patient.BOOLFL.2 AND
    agent.NUMFL.1 > 0.75``, or, since stage c2a, pair literals (``PAIRFL.1(agent, patient)``,
    ``NOT SOME.PAIRFL.2(agent)``)."""
    literals: list[AnyConditionLiteral] = []
    for part in text.split(" AND "):
        stripped = part.strip()
        bare = stripped[4:].strip() if stripped.upper().startswith("NOT ") else stripped
        if _PAIR_LITERAL.match(bare):
            literals.append(PairLiteral.parse(stripped))
        else:
            literals.append(ConditionLiteral.parse(part))
    return tuple(literals)


@dataclass(frozen=True, order=True)
class Effect:
    """One effect: a base Boolean fluent of one role is set to a value, or (a numeric effect,
    on a ``NUMFL`` fluent) a constant is added to a numeric fluent of one role or subtracted
    from it, ``value`` being the signed constant; or, since stage c2a (a pair effect, on a
    ``PAIRFL`` fluent, with ``second`` the role of the second argument), the pair from the role
    to the second role is added (``value`` True) or deleted (False); under a condition when the
    effect has one (an empty tuple is no condition: the effect always applies)."""

    role: str
    fluent: str
    value: bool | float
    condition: tuple[AnyConditionLiteral, ...] = ()
    second: str | None = None

    def __post_init__(self) -> None:
        if self.pair and self.second is None:
            raise ValueError(f"a pair effect on {self.fluent} names a second role")
        if self.pair and self.second == self.role:
            raise ValueError(f"a pair effect on {self.fluent} names two different roles")
        if not self.pair and self.second is not None:
            raise ValueError(f"an effect on {self.fluent} names one role")

    @property
    def numeric(self) -> bool:
        return self.fluent.startswith(NUMERIC_PREFIX)

    @property
    def pair(self) -> bool:
        """A pair effect (stage c2a): an add or a delete of a pair between the roles."""
        return self.fluent.startswith(PAIR_PREFIX)

    @property
    def roles(self) -> tuple[str, ...]:
        return (self.role,) if self.second is None else (self.role, self.second)

    @property
    def literal(self) -> PairLiteral:
        """The pair literal a pair effect makes true: ``PAIRFL.1(agent, patient)`` for an add,
        its negation for a delete."""
        assert self.second is not None
        return PairLiteral(PAIR_KIND, self.fluent, self.role, self.second, bool(self.value))

    @property
    def amount(self) -> float:
        """A numeric effect's signed constant."""
        return float(self.value)

    @property
    def operator(self) -> str:
        return "+=" if self.amount >= 0 else "-="

    @property
    def key(self) -> str:
        if self.pair:
            return f"{self.fluent}({self.role}, {self.second})"
        return f"{self.role}.{self.fluent}"

    @property
    def conditional(self) -> bool:
        return bool(self.condition)

    @property
    def unconditional(self) -> Effect:
        """The same effect without its condition."""
        return Effect(self.role, self.fluent, self.value, (), self.second)

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
        if self.pair:
            entry = {
                "kind": PAIR_KIND,
                "pair_fluent": self.fluent,
                "first": self.role,
                "second": self.second,
                "value": bool(self.value),
                "expression": self.text,
            }
        elif self.numeric:
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
        pair = _PAIR_EFFECT.match(text)
        if pair is not None:
            if pair.group(2) == pair.group(3):
                raise ValueError(f"{text!r}: a pair effect names two different roles")
            return cls(pair.group(2), pair.group(1), pair.group(4) == "1", condition, pair.group(3))
        numeric = _NUMERIC_EFFECT.match(text)
        if numeric is not None:
            amount = float(numeric.group(4))
            if amount <= 0:
                raise ValueError(f"{text!r}: a numeric effect's constant is above 0")
            signed = amount if numeric.group(3) == "+=" else -amount
            return cls(numeric.group(1), numeric.group(2), signed, condition)
        raise ValueError(
            f"{text!r} is not an effect (write agent.BOOLFL.<n> := 0 or := 1, agent.NUMFL.<n> "
            f"+= 0.3 or -= 0.3, PAIRFL.<n>(agent, patient) := 0 or := 1, or when <condition> "
            f"then <effect>)"
        )


def precondition_text(literals: tuple[AnyLiteral, ...]) -> str:
    return " AND ".join(lit.text for lit in literals) if literals else "TRUE"
