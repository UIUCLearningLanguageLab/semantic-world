"""Event files: explicit event types (``docs/specs/WORLD_AND_LANGUAGE.md``, "Explicit event
types").

An event file names fluents with their initial rates, and gives event types a requirement, a
precondition, and effects by hand. Event types without an entry are sampled. Checks that need the
generated world (does the event type exist, is a fluent derived, is a literal achievable) happen
when the file is applied, in ``event_types.py``; every error names the file and the entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from semantic_world.taxonomy.config import _MISSING, ConfigError, _describe, _Node
from semantic_world.taxonomy.expressions import (
    Const,
    Expr,
    ExpressionError,
    Gt,
    Not,
    Op,
    Var,
    parse_expression,
)
from semantic_world.world.dynamics import Effect, Literal

_EVENT_TYPE = re.compile(r"^EVENTTYPE1\.\d+$|^EVENTTYPE2\.\d+(\.\d+)*$")
_FLUENT = re.compile(r"^BOOLFL\.\d+$")
_NUMERIC = re.compile(r"^NUMFL\.\d+$")


@dataclass(frozen=True)
class ExplicitEventType:
    label: str
    requirement: Expr | None
    """A requirement over static literals of the roles (``agent.PROPERTY.3``), or None."""
    requirement_text: str | None
    precondition: tuple[Literal, ...] | None
    """None when the file gives no precondition (it is sampled); an empty tuple is TRUE."""
    effects: tuple[Effect, ...] | None
    field: str


@dataclass(frozen=True)
class EventFile:
    fluents: dict[str, float]
    """Named Boolean fluents and their initial rates."""
    event_types: dict[str, ExplicitEventType]
    source: str
    numeric: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Named numeric fluents (stage b2): each with an ``initial`` distribution and a ``drift``,
    either optional."""


def load_event_file(path: str | Path) -> EventFile:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(str(path), "<file>", f"cannot read the file: {error.strerror}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from error
    return event_file_from_mapping(data, source=str(path))


def event_file_from_mapping(data: Any, *, source: str = "<mapping>") -> EventFile:
    if data is None:
        data = {}
    root = _Node(source, "", data)
    fluents: dict[str, float] = {}
    numeric: dict[str, dict[str, Any]] = {}
    fluents_node = root.mapping("fluents")
    for label in list(fluents_node.data):
        if isinstance(label, str) and _NUMERIC.match(label):
            entry = fluents_node.mapping(label)
            setting: dict[str, Any] = {}
            if "initial" in entry.data:
                from semantic_world.world.config import _read_initial

                setting["initial"] = _read_initial(entry, "initial", None)
            if "drift" in entry.data:
                setting["drift"] = entry.number("drift", None, nullable=True)
            entry.finish()
            numeric[label] = setting
            continue
        if not isinstance(label, str) or not _FLUENT.match(label):
            raise fluents_node.error(label, "expected a fluent label BOOLFL.<n> or NUMFL.<n>")
        entry = fluents_node.mapping(label)
        fluents[label] = float(entry.probability("initial_rate", _MISSING))
        entry.finish()
    fluents_node.finish()
    event_types: dict[str, ExplicitEventType] = {}
    types_node = root.mapping("event_types")
    for label in list(types_node.data):
        if not isinstance(label, str) or not _EVENT_TYPE.match(label):
            raise types_node.error(
                label, "expected an event type label EVENTTYPE1.<n> or EVENTTYPE2.<path>"
            )
        entry = types_node.mapping(label)
        event_types[label] = _read_event_type(label, entry)
        entry.finish()
    types_node.finish()
    root.finish()
    return EventFile(fluents, event_types, source, numeric)


def _read_event_type(label: str, node: _Node) -> ExplicitEventType:
    requirement_text = node.string("requirement", None, nullable=True)
    requirement = None
    if requirement_text is not None:
        try:
            requirement = parse_expression(requirement_text)
        except ExpressionError as error:
            raise node.error("requirement", str(error)) from None
    precondition = None
    if "precondition" in node.data:
        text = node.string("precondition", _MISSING)
        precondition = _parse_precondition(text, node)
    effects = None
    if "effects" in node.data:
        items = node.get("effects")
        if not isinstance(items, list):
            raise node.error("effects", f"expected a list of effects, found {_describe(items)}")
        parsed = []
        for i, item in enumerate(items):
            if not isinstance(item, str):
                raise node.error(f"effects[{i}]", f"expected a string, found {_describe(item)}")
            try:
                parsed.append(Effect.parse(item))
            except ValueError as error:
                raise node.error(f"effects[{i}]", str(error)) from None
        effects = tuple(parsed)
    return ExplicitEventType(label, requirement, requirement_text, precondition, effects, node.path)


def _parse_precondition(text: str, node: _Node) -> tuple[Literal, ...]:
    """A conjunction of fluent literals and numeric threshold literals (``agent.NUMFL.1 >
    0.75``, ``agent.NUMFL.1 <= 0.75``), or TRUE."""
    try:
        expression = parse_expression(text)
    except ExpressionError as error:
        raise node.error("precondition", str(error)) from None
    literals: list[Literal] = []

    def visit(expr: Expr) -> None:
        if isinstance(expr, Const) and expr.value:
            return
        if isinstance(expr, Var):
            literals.append(_literal(expr.name, True, node))
        elif isinstance(expr, Not) and isinstance(expr.operand, Var):
            literals.append(_literal(expr.operand.name, False, node))
        elif isinstance(expr, Gt):
            literals.append(_numeric_literal(expr.scalar, True, expr.threshold, node))
        elif isinstance(expr, Not) and isinstance(expr.operand, Gt):
            literals.append(
                _numeric_literal(expr.operand.scalar, False, expr.operand.threshold, node)
            )
        elif isinstance(expr, Op) and expr.operator == "AND":
            for operand in expr.operands:
                visit(operand)
        else:
            raise node.error(
                "precondition",
                f"a precondition is a conjunction of fluent literals (agent.BOOLFL.<n>, "
                f"NOT patient.BOOLFL.<n>, agent.NUMFL.<n> > 0.75, joined by AND), got {text!r}",
            )

    visit(expression)
    return tuple(literals)


def _literal(key: str, value: bool, node: _Node) -> Literal:
    try:
        return Literal.parse_key(key, value)
    except ValueError as error:
        raise node.error("precondition", str(error)) from None


def _numeric_literal(key: str, above: bool, threshold: float, node: _Node) -> Literal:
    role, _, fluent = key.partition(".")
    if role not in ("agent", "patient") or not _NUMERIC.match(fluent):
        raise node.error(
            "precondition",
            f"{key!r} is not a numeric fluent of a role (agent.NUMFL.<n> or patient.NUMFL.<n>)",
        )
    return Literal(role, fluent, above, float(threshold))
