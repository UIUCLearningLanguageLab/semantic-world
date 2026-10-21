"""Event files: explicit event types (``docs/specs/WORLD_AND_LANGUAGE.md``, "Explicit event
types").

An event file names fluents with their initial rates, and gives event types a requirement, a
precondition, and effects by hand. Event types without an entry are sampled. Checks that need the
generated world (does the event type exist, is a fluent derived, is a literal achievable) happen
when the file is applied, in ``event_types.py``; every error names the file and the entry.

Since stage c2a (``docs/specs/WORLD_PHASE_C.md``, "Explicit event files") an event file may
define pair fluents between entities (``pair_fluents: {PAIRFL.1: {symmetry, functional,
initial_rate}}``), read pair literals in preconditions and conditions, add or delete pairs in
effects, and give the ``spatial`` block (a detector and its parameters for a pair fluent). A
location fluent (``total: true``), a ``place_event_types`` entry, and a ``locations`` key come in
stage c3 and are errors here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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
from semantic_world.world.dynamics import AnyLiteral, Effect, Literal, PairLiteral
from semantic_world.world.pair_fluents import SYMMETRIES

_EVENT_TYPE = re.compile(r"^EVENTTYPE1\.\d+$|^EVENTTYPE2\.\d+(\.\d+)*$")
_FLUENT = re.compile(r"^BOOLFL\.\d+$")
_NUMERIC = re.compile(r"^NUMFL\.\d+$")
_PAIR = re.compile(r"^PAIRFL\.\d+$")
_PAIR_TEXT = re.compile(
    r"(?:(?:SOME|SOMEOF|SAME)\.)?PAIRFL\.\d+\s*\(\s*(?:agent|patient)\s*"
    r"(?:,\s*(?:agent|patient)\s*)?\)"
)
_PLACEHOLDER = "PAIRLITERAL_"


@dataclass(frozen=True)
class ExplicitEventType:
    label: str
    requirement: Expr | None
    """A requirement over static literals of the roles (``agent.PROPERTY.3``), or None."""
    requirement_text: str | None
    precondition: tuple[AnyLiteral, ...] | None
    """None when the file gives no precondition (it is sampled); an empty tuple is TRUE. A
    literal may be a pair literal (stage c2a)."""
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
    pair_fluents: dict[str, dict[str, Any]] = field(default_factory=dict)
    """The pair fluents between entities (stage c2a): each ``{symmetry, functional,
    initial_rate}``, in label order."""
    spatial: dict[str, dict[str, Any]] | None = None
    """The ``spatial`` block (F6): for a pair fluent, ``{detector, parameters}``; None when the
    file gives none."""


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
    for key in ("place_event_types", "locations"):
        if key in root.data:
            raise root.error(
                key,
                f"{key} in an event file comes in stage c3; stage c2a defines pair fluents "
                f"between entities only",
            )
    fluents: dict[str, float] = {}
    numeric: dict[str, dict[str, Any]] = {}
    pair_fluents = _read_pair_fluents(root)
    spatial = _read_spatial(root, pair_fluents)
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
    return EventFile(fluents, event_types, source, numeric, pair_fluents, spatial)


def _read_pair_fluents(root: _Node) -> dict[str, dict[str, Any]]:
    """The ``pair_fluents`` block: ``PAIRFL.<n>: {symmetry: symmetric | asymmetric | general,
    functional: bool, initial_rate}``, in label order. ``total: true`` (a location fluent) is an
    error until stage c3."""
    node = root.mapping("pair_fluents")
    found: dict[str, dict[str, Any]] = {}
    for label in list(node.data):
        if not isinstance(label, str) or not _PAIR.match(label):
            raise node.error(label, "expected a pair fluent label PAIRFL.<n>")
        entry = node.mapping(label)
        if "total" in entry.data:
            if entry.get("total") is not False:
                raise entry.error(
                    "total",
                    "a total pair fluent is a location fluent, which comes in stage c3; a pair "
                    "fluent between entities is never total",
                )
            entry.get("total")
        found[label] = {
            "symmetry": entry.choice("symmetry", _MISSING, SYMMETRIES),
            "functional": entry.bool("functional", _MISSING),
            "initial_rate": float(entry.probability("initial_rate", _MISSING)),
        }
        entry.finish()
    node.finish()
    return dict(sorted(found.items(), key=lambda item: int(item[0].split(".")[1])))


def _read_spatial(root: _Node, pair_fluents: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    """The ``spatial`` block (F6): for a pair fluent of the file, a detector name and a mapping
    of parameters. The runtime checks the form only."""
    if "spatial" not in root.data:
        return None
    value = root.get("spatial", None, nullable=True)
    if value is None:
        return None
    node = _Node(root.source, root.field("spatial"), value)
    if not isinstance(value, dict):
        raise root.error("spatial", f"expected a mapping, found {_describe(value)}")
    found: dict[str, dict[str, Any]] = {}
    for label in list(value):
        if not isinstance(label, str) or label not in pair_fluents:
            raise node.error(label, "expected a pair fluent of the file's pair_fluents block")
        entry = node.mapping(label)
        detector = entry.string("detector", _MISSING)
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict):
            raise entry.error("parameters", f"expected a mapping, found {_describe(parameters)}")
        for key, number in parameters.items():
            if not isinstance(key, str):
                raise entry.error("parameters", f"expected string keys, found {key!r}")
            if isinstance(number, bool) or not isinstance(number, (int, float, str)):
                raise entry.error(
                    f"parameters.{key}", f"expected a number or a string, found {_describe(number)}"
                )
        entry.finish()
        found[label] = {"detector": detector, "parameters": dict(parameters)}
    return found


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


def _parse_precondition(text: str, node: _Node) -> tuple[AnyLiteral, ...]:
    """A conjunction of fluent literals, numeric threshold literals (``agent.NUMFL.1 > 0.75``,
    ``agent.NUMFL.1 <= 0.75``), and, since stage c2a, pair literals (``PAIRFL.1(agent,
    patient)``, ``NOT SOME.PAIRFL.2(agent)``), or TRUE. A pair literal's text is replaced by a
    placeholder name before the expression is parsed, because its parentheses and comma are not
    part of the expression grammar."""
    pairs: list[PairLiteral] = []

    def placeholder(match: re.Match) -> str:
        try:
            literal = PairLiteral.match(match.group(0))
        except ValueError as error:
            raise node.error("precondition", str(error)) from None
        assert literal is not None
        pairs.append(literal)
        return f"{_PLACEHOLDER}{len(pairs) - 1}"

    substituted = _PAIR_TEXT.sub(placeholder, text)
    try:
        expression = parse_expression(substituted)
    except ExpressionError as error:
        raise node.error("precondition", str(error)) from None
    literals: list[AnyLiteral] = []

    def pair_literal(name: str, value: bool) -> PairLiteral:
        literal = pairs[int(name[len(_PLACEHOLDER) :])]
        return literal if value else literal.negated()

    def visit(expr: Expr) -> None:
        if isinstance(expr, Const) and expr.value:
            return
        if isinstance(expr, Var) and expr.name.startswith(_PLACEHOLDER):
            literals.append(pair_literal(expr.name, True))
        elif (
            isinstance(expr, Not)
            and isinstance(expr.operand, Var)
            and expr.operand.name.startswith(_PLACEHOLDER)
        ):
            literals.append(pair_literal(expr.operand.name, False))
        elif isinstance(expr, Var):
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
                f"NOT patient.BOOLFL.<n>, agent.NUMFL.<n> > 0.75, PAIRFL.<n>(agent, patient), "
                f"SOME.PAIRFL.<n>(agent), joined by AND), got {text!r}",
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
