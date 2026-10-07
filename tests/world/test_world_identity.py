"""Stage a1: canonical JSON and the rule-set identity (REL.16)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from semantic_world.world import canonical_json, read_json, rule_set_id, to_json, write_json
from semantic_world.world.identity import format_float

SYMBOLS = [{"label": "IS.1", "kind": "is", "derived": False, "fluent": False, "arity": 1}]
LITERALS = [
    {"index": 0, "key": "IS.1", "kind": "feature", "role": None, "feature": "IS.1"},
    {
        "index": 1,
        "key": "SC.1>0.4127",
        "kind": "threshold",
        "role": None,
        "scalar": "SC.1",
        "operator": ">",
        "threshold": 0.4127,
    },
]
RULES = [
    {
        "output": "IS.2",
        "inputs": [0, 1],
        "truth_table": "0001",
        "expression": "IS.1 AND SC.1 > 0.4127",
    }
]


def test_canonical_json_sorts_keys_and_drops_whitespace() -> None:
    assert canonical_json({"b": [1, 2.5, None], "a": {"d": True, "c": "x"}}) == (
        '{"a":{"c":"x","d":true},"b":[1,2.5,null]}'
    )


def test_floats_are_written_with_17_significant_digits() -> None:
    assert format_float(0.4127) == "0.41270000000000001"
    assert format_float(1.0) == "1.0"
    assert format_float(0.5) == "0.5"
    assert format_float(1e-7) == "9.9999999999999995e-08"
    for value in (0.4127, 0.1, 1 / 3, 2.0**-40, 123456.789):
        assert float(format_float(value)) == value
    with pytest.raises(ValueError):
        format_float(math.nan)


def test_integers_booleans_and_none_are_distinguished() -> None:
    assert canonical_json([1, 1.0, True, None]) == "[1,1.0,true,null]"


def test_indented_form_keeps_key_order_and_flattens_simple_containers(tmp_path: Path) -> None:
    value = {"version": 1, "rows": [{"a": [1, 2], "b": "x"}], "empty": [], "map": {}}
    text = to_json(value, indent=1, sort_keys=False)
    assert text == (
        '{\n "version": 1,\n "rows": [\n  {"a": [1, 2], "b": "x"}\n ],'
        '\n "empty": [],\n "map": {}\n}'
    )
    assert json.loads(text) == value
    path = write_json(tmp_path / "x.json", value)
    assert path.read_text().endswith("}\n")
    assert read_json(path) == value


def test_identity_is_a_sha256_hex_digest_and_is_stable() -> None:
    first = rule_set_id(SYMBOLS, LITERALS, RULES)
    assert len(first) == 64 and int(first, 16) >= 0
    assert rule_set_id(SYMBOLS, LITERALS, RULES, []) == first
    assert rule_set_id(list(reversed(SYMBOLS)), LITERALS, RULES) == first  # one symbol only


def test_identity_changes_with_a_rule_a_literal_number_a_symbol_or_an_event_type() -> None:
    base = rule_set_id(SYMBOLS, LITERALS, RULES)
    changed_rule = [dict(RULES[0], truth_table="0111")]
    assert rule_set_id(SYMBOLS, LITERALS, changed_rule) != base
    changed_literal = [LITERALS[0], dict(LITERALS[1], threshold=0.4128)]
    assert rule_set_id(SYMBOLS, changed_literal, RULES) != base
    changed_symbol = [dict(SYMBOLS[0], derived=True)]
    assert rule_set_id(changed_symbol, LITERALS, RULES) != base
    assert rule_set_id(SYMBOLS, LITERALS, RULES, [{"label": "EVENTTYPE1.1"}]) != base


def test_key_order_in_the_input_does_not_matter() -> None:
    reordered = [{k: RULES[0][k] for k in reversed(list(RULES[0]))}]
    assert rule_set_id(SYMBOLS, LITERALS, reordered) == rule_set_id(SYMBOLS, LITERALS, RULES)
