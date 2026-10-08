"""Stage a3: the conformance fixtures, the brute-force evaluator, and ``check-fixtures``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_world.world.__main__ import main
from semantic_world.world.config import load_config
from semantic_world.world.fixtures import (
    FIXTURES_DIR,
    BruteEvent,
    BruteForce,
    FixtureError,
    check_fixture,
    check_fixtures,
    fixture_inputs,
    fixture_paths,
    generate_fixture,
    read_fixture,
    world_fixtures,
)
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.identity import rule_set_id, to_json

DATA = Path("data/world")
HAND_CASES = (
    "no_event",
    "effect_sets",
    "effect_clears",
    "effect_no_change",
    "derived_changes",
    "precondition_blocks",
    "two_events_independent",
    "two_events_interfere",
)


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


def test_the_fixture_folder_holds_the_spec_cases_and_the_tiny_fixtures() -> None:
    names = [path.stem for path in fixture_paths(FIXTURES_DIR)]
    assert len(names) >= 8 + 2
    for case in HAND_CASES:
        assert any(name.startswith("hand_") and name.endswith(case) for name in names), case
    assert {"tiny_seed1", "tiny_illegal", "tiny_interference"} <= set(names)


@pytest.mark.parametrize("path", fixture_paths(FIXTURES_DIR), ids=lambda p: p.stem)
def test_every_fixture_passes(path: Path) -> None:
    fixture = read_fixture(path)
    report = check_fixture(fixture)
    assert report.name == path.stem == fixture["name"]
    assert report.steps == len(fixture["steps"])
    assert report.error == (fixture["error"] or {}).get("kind")
    # The identity recorded in the file is the identity of the inline definition.
    record = fixture["definition"]
    assert fixture["rule_set_id"] == record["rule_set_id"]
    assert fixture["rule_set_id"] == rule_set_id(
        record["symbols"], record["literals"], record["rules"], record["event_types"]
    )
    # The file is in the indented JSON form, so a reader can diff it.
    assert path.read_text(encoding="utf-8") == to_json(fixture, indent=1, sort_keys=False) + "\n"


def test_generated_fixtures_are_reproducible(tiny: WorldResult) -> None:
    """Regenerating the tiny world's fixtures gives the committed files byte for byte."""
    for fixture in world_fixtures(tiny.definition, "tiny"):
        path = FIXTURES_DIR / f"{fixture['name']}.json"
        assert (
            path.read_text(encoding="utf-8") == to_json(fixture, indent=1, sort_keys=False) + "\n"
        ), path


def test_generated_fixtures_record_the_tiny_world_identity(tiny: WorldResult) -> None:
    for path in fixture_paths(FIXTURES_DIR):
        if path.stem.startswith("tiny_"):
            assert read_fixture(path)["rule_set_id"] == tiny.rule_set_id


def test_brute_force_reads_truth_tables_only(tiny: WorldResult) -> None:
    record, entities, initial = fixture_inputs(tiny.definition)
    record = json.loads(json.dumps(record))
    record["layers"] = []
    brute = BruteForce(record, entities)
    state = {e: frozenset(initial[e]) for e in brute.entity_labels}
    derived = brute.derived_record(state)
    assert set(derived) == set(brute.entity_labels)
    legal = brute.legal_record(state)
    assert set(legal) == set(brute.performable())
    for label, bindings in legal.items():
        for binding in bindings:
            assert brute.legal(BruteEvent(label, binding), state)


def test_check_fixture_reports_disagreements(tiny: WorldResult) -> None:
    record, entities, initial = fixture_inputs(tiny.definition)
    fixture = generate_fixture("probe", "a probe", record, entities, initial, 1, 3)
    check_fixture(fixture)
    broken = json.loads(json.dumps(fixture))
    broken["rule_set_id"] = "0" * 64
    with pytest.raises(FixtureError, match="hashes to"):
        check_fixture(broken)
    broken = json.loads(json.dumps(fixture))
    first = next(k for k, v in broken["steps"][0]["legal"].items() if v)
    broken["steps"][0]["legal"][first] = []
    with pytest.raises(FixtureError, match="legal bindings"):
        check_fixture(broken)
    broken = json.loads(json.dumps(fixture))
    entity = broken["entities"][0]["label"]
    fluents = broken["steps"][0]["state"][entity]
    broken["steps"][0]["state"][entity] = [] if fluents else ["BOOLFL.1"]
    with pytest.raises(FixtureError, match="state after the step"):
        check_fixture(broken)
    broken = json.loads(json.dumps(fixture))
    broken["error"] = {"step": 3, "kind": "illegal"}
    del broken["steps"][2]["state"]
    with pytest.raises(FixtureError, match="raised no error"):
        check_fixture(broken)


def test_error_fixtures_need_the_right_kind(tiny: WorldResult) -> None:
    record, entities, initial = fixture_inputs(tiny.definition)
    fixture = generate_fixture("probe", "a probe", record, entities, initial, 2, 4, error="illegal")
    assert fixture["error"] == {"step": 4, "kind": "illegal"}
    assert "state" not in fixture["steps"][-1]
    check_fixture(fixture)
    wrong = json.loads(json.dumps(fixture))
    wrong["error"]["kind"] = "interference"
    with pytest.raises(FixtureError, match="expects interference"):
        check_fixture(wrong)


def test_check_fixtures_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-fixtures"]) == 0
    out = capsys.readouterr().out
    assert f"{len(fixture_paths(FIXTURES_DIR))} fixtures passed" in out
    assert main(["check-fixtures", str(tmp_path)]) == 1
    assert "no fixture found" in capsys.readouterr().err
    paths = check_fixtures(FIXTURES_DIR)
    assert len(paths) == len(fixture_paths(FIXTURES_DIR))


def test_make_fixtures_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "fixtures"
    assert (
        main(
            [
                "make-fixtures",
                "data/world/tiny.yaml",
                "--out",
                str(out),
                "--count",
                "1",
                "--steps",
                "2",
            ]
        )
        == 0
    )
    names = sorted(p.name for p in out.glob("*.json"))
    assert names == ["tiny_illegal.json", "tiny_interference.json", "tiny_seed1.json"]
    assert main(["check-fixtures", str(out)]) == 0
    assert "3 fixtures passed" in capsys.readouterr().out
