"""Canonical JSON, and the rule-set identity (REL.16).

The rule-set identity is the SHA-256 hash, in hexadecimal, of the canonical JSON of a
definition's ``symbols``, ``literals``, ``rules``, and ``event_types``. Canonical JSON has
sorted keys, no whitespace, ASCII escapes, and floating-point numbers written with 17
significant digits, so both runtimes read the same values and the same bytes hash the same.
The pretty form written to files uses the same number format with indentation, and keeps the
order of keys as given.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFINITION_VERSION = 1
"""The version of the definition format, written as ``version`` in every definition file."""


def format_float(value: float) -> str:
    """A floating-point number with 17 significant digits, always with a decimal point or an
    exponent, so a reader sees a float. NaN and infinities are not allowed in a definition."""
    if not math.isfinite(value):
        raise ValueError(f"a definition may not contain {value!r}")
    text = format(value, ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def to_json(value: Any, indent: int | None = None, sort_keys: bool = True) -> str:
    """Serialize ``value`` (mappings, sequences, strings, numbers, booleans, None) as JSON.

    ``indent=None`` gives the compact canonical form. With an indent, every element of a mapping
    or a sequence goes on its own line, except that a container whose elements are all scalars,
    or all scalars and sequences of scalars, is written on one line.
    """
    parts: list[str] = []
    _dump(value, indent, sort_keys, 0, parts)
    return "".join(parts)


def _dump(value: Any, indent: int | None, sort_keys: bool, level: int, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        out.append(str(int(value)))
    elif isinstance(value, float):
        out.append(format_float(value))
    elif isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=True))
    elif isinstance(value, Mapping):
        keys = list(value)
        if sort_keys:
            keys.sort()
        if not keys:
            out.append("{}")
            return
        flat = indent is not None and all(_is_simple(value[key]) for key in keys)
        out.append("{")
        for i, key in enumerate(keys):
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {key!r}")
            if i:
                out.append(", " if flat else ",")
            if not flat:
                _newline(indent, level + 1, out)
            out.append(json.dumps(key, ensure_ascii=True))
            out.append(": " if indent is not None else ":")
            _dump(value[key], indent, sort_keys, level + 1, out)
        if not flat:
            _newline(indent, level, out)
        out.append("}")
    elif isinstance(value, Sequence) or hasattr(value, "tolist"):
        items = value.tolist() if hasattr(value, "tolist") else list(value)
        if not items:
            out.append("[]")
            return
        flat = indent is not None and all(_is_simple(item) for item in items)
        out.append("[")
        for i, item in enumerate(items):
            if i:
                out.append(", " if flat else ",")
            if not flat:
                _newline(indent, level + 1, out)
            _dump(item, indent, sort_keys, level + 1, out)
        if not flat:
            _newline(indent, level, out)
        out.append("]")
    else:
        raise TypeError(f"cannot write {type(value).__name__} as JSON")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _is_simple(value: Any) -> bool:
    """A scalar, or a sequence of scalars: written on one line in the indented form."""
    if _is_scalar(value):
        return True
    if isinstance(value, Mapping):
        return False
    if isinstance(value, Sequence) or hasattr(value, "tolist"):
        items = value.tolist() if hasattr(value, "tolist") else list(value)
        return all(_is_scalar(item) for item in items)
    return False


def _newline(indent: int | None, level: int, out: list[str]) -> None:
    if indent is not None:
        out.append("\n" + " " * (indent * level))


def canonical_json(value: Any) -> str:
    """The canonical form: sorted keys, no whitespace, floats with 17 significant digits."""
    return to_json(value, indent=None, sort_keys=True)


def rule_set_id(
    symbols: Sequence[Mapping[str, Any]],
    literals: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    event_types: Sequence[Mapping[str, Any]] = (),
) -> str:
    """The rule-set identity: SHA-256 of the canonical JSON of the four tables, in hexadecimal.
    It changes exactly when a symbol, a literal's numbers, a rule, or an event type changes."""
    document = {
        "symbols": list(symbols),
        "literals": list(literals),
        "rules": list(rules),
        "event_types": list(event_types),
    }
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    """Write a definition file: indented, keys in the given order, a newline at the end."""
    path = Path(path)
    path.write_text(to_json(value, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
