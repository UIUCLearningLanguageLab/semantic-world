"""Rule files: weighted rule templates and explicit rules, loaded from YAML.

A rule file lists ``templates``, each with a sampling weight, and optionally ``explicit`` rules
that fix a named output feature to a named expression. Determined features without an explicit
rule are sampled from the templates that apply to their type. Checks that need the feature
layout (labels, layers, the input pool) happen in ``rules.py`` when the rules are built.

Example::

    templates:
      - {family: literal, weight: 0.1}
      - {family: fixed, arity: 2, operator: XOR, weight: 0.1}
      - {family: shj, type: IV, weight: 0.2}
      - {family: shj, type: VI, weight: 0.05, applies_to: can}
      - {family: compositional, arity: 4, operators: {AND: 1, OR: 1}, nesting_depth: 2, weight: 0.3}
    explicit:
      - {output: CAN.3, expression: "(HAS.2 AND NOT IS.5) OR IS.7"}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_world.taxonomy.config import (
    _MISSING,
    FEATURE_TYPES,
    OPERATORS,
    SHJ_TYPES,
    ConfigError,
    _is_int,
    _Node,
)
from semantic_world.taxonomy.expressions import Expr, ExpressionError, parse_expression

TEMPLATE_FAMILIES = ("literal", "fixed", "shj", "compositional")


@dataclass(frozen=True)
class Template:
    """One weighted rule template."""

    family: str
    weight: float
    applies_to: tuple[str, ...]
    """The output types the template applies to, in ``(is, has, can)`` order."""
    arity: int | None = None
    operator: str | None = None
    """The operator of a ``fixed`` template."""
    shj_type: str | None = None
    operators: dict[str, float] | None = None
    """The operator mix of a ``compositional`` template; None means the configuration's."""
    nesting_depth: dict[int, float] | None = None
    """The nesting-depth weights of a ``compositional`` template; None means the
    configuration's."""
    negation_probability: float | None = None
    """None means the configuration's negation probability."""

    def applies(self, feature_type: str) -> bool:
        return feature_type in self.applies_to

    @property
    def max_arity(self) -> int:
        return (
            1 if self.family == "literal" else 3 if self.family == "shj" else int(self.arity or 0)
        )


@dataclass(frozen=True)
class ExplicitRule:
    output: str
    expression: Expr
    text: str
    """The expression as written in the file."""
    field: str
    """The field path in the file, for error messages."""


@dataclass(frozen=True)
class RuleFile:
    templates: tuple[Template, ...]
    explicit: tuple[ExplicitRule, ...]
    source: str

    def templates_for(self, feature_type: str) -> tuple[Template, ...]:
        return tuple(t for t in self.templates if t.applies(feature_type))


def load_rule_file(path: str | Path) -> RuleFile:
    """Load and validate a rule file. Errors name the file and the field."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(str(path), "<file>", f"cannot read the file: {error.strerror}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from error
    return rule_file_from_mapping(data, source=str(path))


def rule_file_from_mapping(data: Any, *, source: str = "<mapping>") -> RuleFile:
    if data is None:
        data = {}
    root = _Node(source, "", data)
    templates = _read_list(root, "templates", _read_template)
    explicit = _read_list(root, "explicit", _read_explicit)
    root.finish()
    seen: dict[str, str] = {}
    for rule in explicit:
        if rule.output in seen:
            raise ConfigError(
                source,
                f"{rule.field}.output",
                f"{rule.output} already has an explicit rule at {seen[rule.output]}",
            )
        seen[rule.output] = rule.field
    return RuleFile(tuple(templates), tuple(explicit), source)


def _read_list(root: _Node, key: str, read_item) -> list:
    items = root.get(key, [])
    if not isinstance(items, list):
        raise root.error(key, f"expected a list, found {_describe_short(items)}")
    result = []
    for i, item in enumerate(items):
        node = _Node(root.source, f"{key}[{i}]", item)
        result.append(read_item(node))
        node.finish()
    return result


def _describe_short(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _read_applies_to(node: _Node) -> tuple[str, ...]:
    value = node.get("applies_to", None, nullable=True)
    if value is None:
        return FEATURE_TYPES
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise node.error("applies_to", "expected one of is, has, can, or a list of them")
    for v in values:
        if v not in FEATURE_TYPES:
            raise node.error(
                "applies_to", f"expected one of is, has, can, or a list of them, found {v!r}"
            )
    return tuple(t for t in FEATURE_TYPES if t in values)


def _read_template(node: _Node) -> Template:
    family = node.choice("family", _MISSING, TEMPLATE_FAMILIES)
    weight = node.number("weight", _MISSING, min=0)
    applies_to = _read_applies_to(node)
    negation = node.probability("negation_probability", None, nullable=True)
    if family == "literal":
        return Template(family, weight, applies_to, arity=1, negation_probability=negation)
    if family == "fixed":
        arity = node.int("arity", _MISSING, min=2)
        operator = node.choice("operator", _MISSING, OPERATORS)
        return Template(
            family,
            weight,
            applies_to,
            arity=arity,
            operator=operator,
            negation_probability=negation,
        )
    if family == "shj":
        shj_type = node.choice("type", _MISSING, SHJ_TYPES)
        return Template(
            family, weight, applies_to, arity=3, shj_type=shj_type, negation_probability=negation
        )
    arity = node.int("arity", _MISSING, min=2)
    operators = (
        node.weights("operators", _MISSING, allowed=OPERATORS) if "operators" in node.data else None
    )
    node.seen.add("operators")
    nesting_depth = _read_nesting_depth(node)
    return Template(
        family,
        weight,
        applies_to,
        arity=arity,
        operators=operators,
        nesting_depth=nesting_depth,
        negation_probability=negation,
    )


def _read_nesting_depth(node: _Node) -> dict[int, float] | None:
    """A single depth (a positive integer) or a mapping of depth weights."""
    value = node.get("nesting_depth", None, nullable=True)
    if value is None:
        return None
    if _is_int(value):
        node.check_int("nesting_depth", value, min=1)
        return {value: 1.0}
    if isinstance(value, dict):
        return node.weights("nesting_depth", _MISSING, int_keys=True)
    raise node.error(
        "nesting_depth", f"expected a positive integer or a mapping of weights, found {value!r}"
    )


def _read_explicit(node: _Node) -> ExplicitRule:
    output = node.string("output", _MISSING)
    text = node.string("expression", _MISSING)
    try:
        expression = parse_expression(text)
    except ExpressionError as error:
        raise node.error("expression", str(error)) from None
    return ExplicitRule(output, expression, text, node.path)


__all__ = [
    "TEMPLATE_FAMILIES",
    "ExplicitRule",
    "RuleFile",
    "Template",
    "load_rule_file",
    "rule_file_from_mapping",
]
