"""Stage b2: the corpus over a world with numeric fluents says nothing about them, and stays
correct about everything else.

No sentence of any document, and no test item, names a numeric fluent; no causal statement is
true by a numeric effect or a numeric precondition literal (the oracle re-reads the definition
apart); no result sentence states a numeric change; and a derived Boolean fluent that reads a
numeric threshold is judged on the replayed states with their numeric values, in state and
change sentences, by the corpus and by the oracle alike. The world here is the tiny world with
every numeric feature turned up (a derived fluent that reads a numeric threshold among them),
written to a run folder so that the oracle can read its files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from corpus_support import corpus_config, oracle_over

from semantic_world.corpus import load_world
from semantic_world.corpus.generate import generate
from semantic_world.corpus.propositions import CHANGE, EFFECT, STATE

HEAVY_WORLD = {
    "numeric_fluents": {"count": 2, "rule_weight": 50},
    "event_types": {
        "effects": {
            "numeric_share": 0.6,
            "conditional_share": 0.5,
            "condition_kinds": {"feature": 1, "fluent": 1, "numeric": 2},
        },
        "preconditions": {"numeric_share": 0.6},
    },
}


@pytest.fixture(scope="module")
def heavy(tmp_path_factory: pytest.TempPathFactory):
    """The heavy tiny world, its corpus of 60 documents with small test sets, the run folder,
    and the oracle."""
    root = tmp_path_factory.mktemp("numeric")
    data = yaml.safe_load(Path("data/world/tiny.yaml").read_text(encoding="utf-8"))
    for key, value in HEAVY_WORLD.items():
        for inner, setting in value.items():
            if isinstance(setting, dict) and isinstance(data.setdefault(key, {}).get(inner), dict):
                data[key][inner] = {**data[key][inner], **setting}
            else:
                data.setdefault(key, {})[inner] = setting
    world_path = root / "heavy.yaml"
    world_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = corpus_config(str(world_path), documents={"count": 60}, test_sets={"size": 30})
    world = load_world(config)
    folder = world.result.write(root / "world")
    corpus = generate(config, world)
    corpus.write(root / "corpus")
    stats = world.result.stats["numeric_fluents"]
    assert stats["rules_reading_a_threshold"] > 0, "the heavy world reads a numeric threshold"
    assert stats["effects"]["total"] > 0 and stats["precondition_literals"]["total"] > 0
    return {
        "config": config,
        "world": world,
        "corpus": corpus,
        "folder": folder,
        "oracle": oracle_over(folder, z=config.scalar_z),
    }


def _mentions_numeric(value: Any) -> bool:
    return "NUMFL." in json.dumps(value)


def test_no_sentence_or_item_names_a_numeric_fluent(heavy) -> None:
    corpus = heavy["corpus"]
    sentences = 0
    for document in corpus.documents:
        for sentence in document.to_json()["sentences"]:
            sentences += 1
            for key in ("logical_form", "propositional", "conceptual", "formal", "tokens"):
                assert not _mentions_numeric(sentence[key]), (document.label, sentence["label"])
    assert sentences > 0
    items = 0
    for item_set in corpus.test_sets:
        for item in item_set.items:
            items += 1
            assert not _mentions_numeric(item.to_json()["input"])
            assert not _mentions_numeric(item.to_json()["meta"].get("causal"))
    assert items > 0
    lexicon = corpus.planner.lexicon
    assert not any(concept.label.startswith("NUMFL.") for concept in lexicon.concepts)
    world = heavy["world"]
    assert not any(f.startswith("NUMFL.") for f in world.fluents)


def test_no_causal_statement_is_true_by_a_numeric_effect_or_literal(heavy) -> None:
    """Every true causal statement names a Boolean entry, and the oracle, which leaves out the
    numeric effects and literals of the definition, judges each true; a statement about a
    numeric fluent cannot be formed, because the fluents of the corpus are Boolean."""
    corpus, oracle = heavy["corpus"], heavy["oracle"]
    facts = corpus.planner.facts
    statements = list(facts.causal_statements())
    assert statements
    for proposition in statements:
        assert not proposition.predicate.label.startswith("NUMFL.")
        form = proposition.to_json()
        assert oracle.causal(form) is True, form
    # the definition has numeric effects the statements never state
    record = json.loads((heavy["folder"] / "definition.json").read_text(encoding="utf-8"))
    numeric_effects = [
        (et["label"], e) for et in record["event_types"] for e in et["effects"] if "amount" in e
    ]
    assert numeric_effects
    stated = {
        (p.subject.event_type, p.subject.role, p.predicate.label)
        for p in statements
        if p.predicate.kind == EFFECT
    }
    for label, effect in numeric_effects:
        assert (label, effect["role"], effect["fluent"]) not in stated


def test_result_sentences_never_state_a_numeric_change(heavy) -> None:
    corpus, oracle = heavy["corpus"], heavy["oracle"]
    scenes = {scene.label: scene.to_json() for scene in corpus.planner.scenes}
    results = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            if proposition.level != CHANGE:
                continue
            results += 1
            assert not proposition.predicate.label.startswith("NUMFL.")
            assert oracle.change(sentence.logical_form, scenes[proposition.scene])
            caused_by = proposition.grounding["caused_by"]
            scene = scenes[proposition.scene]
            step = int(proposition.time.split(".")[1])
            assert oracle.made_by(
                scene, step, proposition.subject, proposition.predicate.label, caused_by
            )
    assert results > 0


def test_states_and_changes_of_derived_fluents_over_numeric_thresholds_are_judged_right(
    heavy,
) -> None:
    """The derived fluents that read a numeric threshold are stated in state and change
    sentences and in the state test sets, and the oracle, replaying the histories with the
    numeric values (the brute-force evaluator's effects and drift), agrees with every one."""
    corpus, oracle, world = heavy["corpus"], heavy["oracle"], heavy["world"]
    definition = world.definition
    numeric_readers = {
        f
        for f in definition.derived_fluents
        if any(definition.is_numeric_fluent(b) for b in definition.cone(f))
    }
    assert numeric_readers
    scenes = {scene.label: scene.to_json() for scene in corpus.planner.scenes}
    judged = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            if proposition.level not in (STATE, CHANGE):
                continue
            form = sentence.logical_form
            assert oracle.truth(form, scenes[proposition.scene]) is True, form
            if proposition.predicate.label in numeric_readers:
                judged += 1
    for item_set in corpus.test_sets:
        if not item_set.name.startswith("state_"):
            continue
        for item in item_set.items:
            form = item.to_json()["input"]["logical_form"]
            scene = scenes[form["scene"]]
            assert oracle.truth(form, scene) == item.meta["truth"], item.to_json()["meta"]
            if form["predicate"]["label"] in numeric_readers:
                judged += 1
    assert judged > 0, (
        "some state or change sentence or item is about a derived fluent over a threshold"
    )


def test_the_corpus_statistics_count_numeric_and_drift_changes(heavy) -> None:
    stats = heavy["corpus"].stats["scenes"]
    assert stats["numeric_changes_per_event"] > 0
    assert stats["drift_changes_per_step"] > 0
    histories = heavy["corpus"].planner.scenes
    booleans = sum(1 for h in histories for e in h.events for c in e.changes if not c.numeric)
    events = sum(len(h.events) for h in histories)
    assert stats["mean_changes_per_event"] == pytest.approx(booleans / events, abs=1e-6)
