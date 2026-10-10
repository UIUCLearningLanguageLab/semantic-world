"""The off settings of the world's optional features reproduce the earlier outputs.

A feature with an off setting draws from a stream of its own, so that turning it off leaves
every other draw as it was: the rule-set identity of the world with the feature off is a baseline
kept in ``tests/figures/snapshot.yaml`` (taken from a run with the feature off, and for the
conditional effects of stage b1 equal to the identity of the a8 world), and the tiny corpus over
the tiny world without conditional effects is the a8 corpus, by the hash of its
``documents.jsonl``. One test per feature and world: the tiny worlds in the fast tier, the
default world in the slow tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_world.figures import (
    OFF_SETTINGS,
    REPO,
    SNAPSHOT_FILE,
    baseline_identity,
    load_snapshot,
    tiny_corpus_hash_off,
)

SLOW = pytest.mark.slow
NAMES = ["tiny", pytest.param("default", marks=SLOW)]


@pytest.fixture(autouse=True)
def _run_from_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)


@pytest.mark.parametrize("feature", sorted(OFF_SETTINGS))
@pytest.mark.parametrize("name", NAMES)
def test_the_world_with_the_feature_off_has_the_baseline_identity(name: str, feature: str) -> None:
    baseline = load_snapshot(SNAPSHOT_FILE)["baselines"][feature][name]
    assert baseline_identity(name, feature) == baseline["rule_set_id"]


def test_the_tiny_corpus_without_conditional_effects_is_the_a8_corpus(tmp_path: Path) -> None:
    baseline = load_snapshot(SNAPSHOT_FILE)["baselines"]["conditional_effects_off"]["tiny"]
    assert tiny_corpus_hash_off("conditional_effects_off", tmp_path) == baseline["documents_sha256"]
