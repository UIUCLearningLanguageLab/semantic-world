"""Stage b3b acceptance tests: the numeric fluents in the corpus, the numeric effect statements
``RAISES`` and ``LOWERS``, and the conditional causal statements.

Every numeric state, change, and comparison sentence is true by the oracle's replay of the
scene; every numeric result sentence states a change the event's own effect made, and none a
change by drift alone; every ``RAISES`` and ``LOWERS`` statement is true by the oracle's
re-reading of ``definition.json``, which says which effects are unconditional and numeric; no
causal statement is true by a conditional effect without its condition, and every conditional
statement is true by the re-reading; every false causal item is false under ``NEC`` and its
``observed`` mark agrees with the scenes; no sentence states a numeric threshold literal; every
numeric pole proposition records its cut; ``interpret(tree)`` recovers every logical form, and
every rendering parses back; the state sets hold numeric poles with ``changed`` marks, and
``state_comparison_role`` swaps the two participants; the comparative items draw their word half
and half (Jon's ruling 3 on stage b3a); numeric fluents are topics of feature documents; and the
statistics count the new sentences. The world here is the tiny world with every numeric feature
turned up (numeric effects, numeric and Boolean conditions, a derived fluent over a numeric
threshold), written to a run folder so that the oracle can read its files.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from corpus_support import corpus_config, oracle_over

from semantic_world.corpus import interpret, load_world, parse_propositional, proposition_of
from semantic_world.corpus.generate import generate
from semantic_world.corpus.planner import CAUSAL, COMPARATIVE_SECTION, INITIAL_STATE, RESULT
from semantic_world.corpus.propositions import (
    CHANGE,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    EFFECT,
    INCREASE,
    LOWERS,
    PRECONDITION,
    RAISES,
    STATE,
    STATE_KIND,
    EventTerm,
    Proposition,
)
from semantic_world.corpus.testsets import CAUSAL_LEVELS, LAWLIKE, ROLE, STATE_COMPARISON

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
MANY = {
    "documents": {
        "count": 80,
        "comparative_rate": 0.6,
        "initial_state_rate": 0.5,
        "result_rate": 0.8,
        "instance_description_rate": 0.1,
    },
    "propositions": {"causal_statement_rate": 0.8},
    "test_sets": {"size": 30},
}
"""Many numeric sentences and causal statements, so that the tests see every kind."""


@pytest.fixture(scope="module")
def heavy(tmp_path_factory: pytest.TempPathFactory):
    """The heavy tiny world, its corpus with small test sets, the run folder, and the oracle."""
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
    config = corpus_config(str(world_path), **MANY)
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
        "record": json.loads((folder / "definition.json").read_text(encoding="utf-8")),
    }


def scenes_json(corpus) -> dict[str, dict]:
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


def sentences_of(corpus):
    return [(d, s) for d in corpus.documents for s in d.sentences]


def read_back(corpus, document, sentence) -> None:
    """The rendering parses back to the proposition, the form reads back, and the tree reads
    back as the plan."""
    proposition = sentence.proposition
    names = document.referents if corpus.config.propositional_referents == "local" else None
    assert proposition_of(parse_propositional(sentence.propositional), names) == proposition
    assert Proposition.from_json(json.loads(json.dumps(sentence.logical_form))) == proposition
    record, plan = sentence.sentence, sentence.plan
    read = interpret(
        record.tree,
        corpus.planner.lexicon,
        record.referents,
        record.events,
        plan.quantifier,
        plan.predication.standard,
    )
    assert read == plan


# ---------------------------------------------------------------------------------------------
# Numeric states, changes, and comparisons at a time point
# ---------------------------------------------------------------------------------------------


def test_numeric_sentences_are_true_by_the_oracle(heavy) -> None:
    corpus, oracle, config = heavy["corpus"], heavy["oracle"], heavy["config"]
    scenes = scenes_json(corpus)
    cuts = {"HIGH": config.cut_high, "LOW": config.cut_low}
    counts: Counter = Counter()
    for document, sentence in sentences_of(corpus):
        proposition = sentence.proposition
        predicate = proposition.predicate
        if not predicate.numeric or proposition.causal:
            continue
        form = json.loads(json.dumps(sentence.logical_form))
        scene = scenes[proposition.scene]
        assert proposition.level in (STATE, CHANGE) and proposition.timed
        assert oracle.truth(form, scene) is True, form
        assert corpus.planner.truth.is_true(proposition)
        assert sentence.plan.proposition() == proposition
        if predicate.kind == COMPARISON:
            # a numeric pole: against RANGE, at the language's cut for its adjective, recorded
            assert proposition.level == STATE and sentence.section == INITIAL_STATE
            assert form["predicate"]["class"] == "RANGE"
            assert form["predicate"]["cut"] == cuts[predicate.label.rsplit(".", 1)[1]]
            assert (
                list(form["predicate"]).index("cut") == list(form["predicate"]).index("class") + 1
            )
            assert "z" not in form or any(
                literal.startswith("SCALARDIM.") for literal in form["subject"]["restriction"]
            )
            assert "HOLDS(" in sentence.propositional and ", RANGE, " in sentence.propositional
            counts["pole", proposition.polarity] += 1
        elif predicate.kind == COMPARATIVE:
            # a comparative at the event's time point, with the agent as the subject
            assert proposition.level == STATE and sentence.section == COMPARATIVE_SECTION
            assert sentence.propositional.split("}")[-1].strip().startswith("HOLDS(")
            assert "GREATER(" in sentence.propositional and "cut" not in form["predicate"]
            assert form["predicate"]["word"] == sentence.plan.predication.label
            before = next(
                s
                for s in reversed(document.sentences[: document.sentences.index(sentence)])
                if s.section == "event"
            )
            event = before.proposition
            assert form["subject"]["instance"] == event.subject
            assert form["predicate"]["patient"]["instance"] == event.predicate.patient
            counts["comparative"] += 1
        else:
            # a numeric change: INCREASE or DECREASE, never negated
            assert predicate.kind == STATE_KIND and proposition.level == CHANGE
            assert sentence.section == RESULT and proposition.polarity is True
            operator = "INCREASE" if predicate.direction == INCREASE else "DECREASE"
            assert sentence.propositional.split("}")[-1].strip().startswith(f"{operator}(")
            assert form["predicate"] == {
                "kind": STATE_KIND,
                "label": predicate.label,
                "direction": predicate.direction,
            }
            counts["change", predicate.direction] += 1
        # no sentence states a threshold literal
        assert ">" not in sentence.propositional and "<=" not in sentence.propositional
        read_back(corpus, document, sentence)
    assert counts["pole", True] > 0 and counts["comparative"] > 0, counts
    assert counts["change", "increase"] > 0 and counts["change", "decrease"] > 0, counts


def test_numeric_result_sentences_are_made_by_the_events_own_effect(heavy) -> None:
    """A result sentence states a numeric change only when the event's own effect moved the
    value and the step's net change has the effect's direction; a change by drift alone is
    never a result sentence."""
    corpus, oracle = heavy["corpus"], heavy["oracle"]
    scenes = scenes_json(corpus)
    results = 0
    for document in corpus.documents:
        events = {
            e.label: e
            for scene in document.scenes
            for e in corpus.planner.truth.events_of(scene.label)
        }
        for index, sentence in enumerate(document.sentences):
            proposition = sentence.proposition
            if sentence.section != RESULT or not proposition.predicate.numeric:
                continue
            results += 1
            before = next(s for s in reversed(document.sentences[:index]) if s.section == "event")
            event = events[before.proposition.event]
            scene = scenes[event.scene]
            fluent, direction = proposition.predicate.label, proposition.predicate.direction
            assert proposition.time == f"TIME.{event.step}" and event.involves(proposition.subject)
            # the event's own recorded change has the direction, and so has the net change
            assert oracle.made_by(
                scene, event.step, proposition.subject, fluent, event.label, direction
            )
            assert oracle.change(sentence.logical_form, scene)
            caused_by = proposition.grounding["caused_by"]
            assert caused_by is not None
            assert oracle.made_by(
                scene, event.step, proposition.subject, fluent, caused_by, direction
            )
            # a change by drift alone is in the history, and is never a result: the drift's
            # record is apart from the events' records
            step = next(s for s in scene["steps"] if s["step"] == event.step)
            own = [
                c
                for e in step["events"]
                for c in e["changes"]
                if c["entity"] == proposition.subject and c["fluent"] == fluent
            ]
            assert own, "the change is an event's, not the drift's alone"
    assert results > 5
    # the heavy world drifts, so some step moves a value by drift alone; such a step has no
    # numeric result sentence about that value unless an event moved it too
    drifted = sum(
        len(step.get("drift", [])) for scene in scenes.values() for step in scene["steps"]
    )
    assert drifted > 0


def test_every_numeric_pole_records_its_cut_and_no_class_level_numeric_sentence(heavy) -> None:
    corpus, config = heavy["corpus"], heavy["config"]
    cuts = {"HIGH": config.cut_high, "LOW": config.cut_low}
    forms = [s.logical_form for _, s in sentences_of(corpus)]
    forms += [item.input["logical_form"] for t in corpus.test_sets for item in t.items]
    poles = 0
    for form in forms:
        predicate = form["predicate"]
        if predicate["kind"] == COMPARISON and predicate["label"].startswith("NUMFL."):
            poles += 1
            assert form["level"] == STATE
            assert predicate["cut"] == cuts[predicate["label"].rsplit(".", 1)[1]]
        elif predicate["label"].startswith("NUMFL."):
            assert "cut" not in predicate
            assert form["level"] in (STATE, CHANGE) or "head" in form["subject"]
        else:
            assert "cut" not in predicate
    assert poles > 0


# ---------------------------------------------------------------------------------------------
# RAISES, LOWERS, and the conditional causal statements
# ---------------------------------------------------------------------------------------------


def numeric_effects(record: dict) -> list[tuple[str, dict]]:
    return [
        (et["label"], e) for et in record["event_types"] for e in et["effects"] if "amount" in e
    ]


def test_raises_and_lowers_are_true_by_a_rereading_of_the_definition(heavy) -> None:
    corpus, oracle, record = heavy["corpus"], heavy["oracle"], heavy["record"]
    facts = corpus.planner.facts
    statements = [p for p in facts.causal_statements() if p.predicate.numeric]
    assert statements
    event_types = {e["label"]: e for e in record["event_types"]}
    for proposition in statements:
        assert proposition.predicate.kind == EFFECT
        assert proposition.predicate.direction in (RAISES, LOWERS)
        assert proposition.predicate.value is None
        form = proposition.to_json()
        assert oracle.causal(form) is True, form
        for leaf in oracle.leaves_below(proposition.subject.event_type):
            matching = [
                e
                for e in event_types[leaf]["effects"]
                if "amount" in e
                and e["role"] == proposition.subject.role
                and e["fluent"] == proposition.predicate.label
                and (e["operator"] == "+=") == (proposition.predicate.direction == RAISES)
            ]
            assert matching, (leaf, form)
            if not proposition.subject.condition:
                assert not matching[0].get("condition")
    # every unconditional numeric effect of an event type with a word on a fluent with words is
    # stated, and no precondition statement is about a numeric fluent
    stated = {
        (p.subject.event_type, p.subject.role, p.predicate.label, p.predicate.direction)
        for p in statements
        if not p.subject.condition
    }
    for label, effect in numeric_effects(record):
        if label not in facts.named or effect["fluent"] not in facts.numeric_fluents:
            continue
        if effect.get("condition") or not oracle.able_somewhere(label):
            continue
        direction = RAISES if effect["operator"] == "+=" else LOWERS
        assert (label, effect["role"], effect["fluent"], direction) in stated
    assert not any(p.predicate.numeric for p in facts.causal_statements(kind=PRECONDITION))
    # the statements are said in feature documents, "become" with the comparative form
    said = Counter()
    for document, sentence in sentences_of(corpus):
        proposition = sentence.proposition
        if proposition.causal and proposition.predicate.numeric:
            assert document.type == "encyclopedic_feature" and sentence.section == CAUSAL
            assert oracle.causal(sentence.logical_form) is True
            operator = "RAISES" if proposition.predicate.direction == RAISES else "LOWERS"
            assert (
                f"{operator}(EVENTVAR.1, {proposition.predicate.label}(" in sentence.propositional
            )
            assert "BECOME" in sentence.sentence.conceptual
            assert "-COMPARATIVE" in sentence.sentence.conceptual
            read_back(corpus, document, sentence)
            said[proposition.predicate.direction] += 1
    assert said[RAISES] > 0 or said[LOWERS] > 0, said


def test_conditional_statements_are_true_and_never_without_their_condition(heavy) -> None:
    """A conditional effect makes a conditional statement, true by the re-reading, whose
    unconditional version is false; no numeric threshold condition is stated."""
    corpus, oracle = heavy["corpus"], heavy["oracle"]
    facts, truth = corpus.planner.facts, corpus.planner.truth
    conditional = [p for p in facts.causal_statements() if p.subject.condition]
    assert conditional, "the heavy world makes conditional statements"
    for proposition in conditional:
        form = proposition.to_json()
        assert oracle.causal(form) is True, form
        assert all(not c.label.startswith("NUMFL.") for c in proposition.subject.condition)
        assert form["causal"]["condition"] == form["subject"]["condition"]
        # the statement without its condition is false, by Truth and by the oracle
        bare = Proposition(
            CLASS,
            proposition.subject.unconditional,
            proposition.predicate,
            True,
            proposition.quantifier,
        )
        evaluation = truth.evaluate(bare)
        assert evaluation.valid and evaluation.true is False
        assert oracle.causal(bare.to_json()) is False
        assert bare not in facts.causal_statements()
    # every conditional effect with a word and a realizable condition is stated; the numeric
    # conditions are counted among the unstated
    report = facts.causal_report
    assert report["conditional_effects"] > 0
    assert report["unstated"]["numeric_condition"] > 0
    assert report["stated"] + sum(report["unstated"].values()) == report["conditional_effects"]
    assert corpus.stats["causal"]["conditional_effects"] == report
    # the sentences: the condition in the restrictor, after the bound event
    said = 0
    for document, sentence in sentences_of(corpus):
        proposition = sentence.proposition
        if not proposition.causal or not proposition.subject.condition:
            continue
        said += 1
        text = sentence.propositional
        assert text.startswith("NEC(ALL(EVENT(EVENTVAR.1, ")
        scope = text.index(
            ", AFTER(" if "AFTER(" in text else (", RAISES(" if "RAISES(" in text else ", LOWERS(")
        )
        restrictor = text[:scope]
        for literal in proposition.subject.condition:
            variable = "VAR.1" if literal.role == "agent" else "VAR.2"
            atom = f"{literal.label}({variable})"
            expected = atom if literal.value else f"NOT {atom}"
            if literal.fluent:
                expected = f"BEFORE(EVENTVAR.1, {expected})"
            assert f" AND {expected}" in restrictor, text
        assert oracle.causal(sentence.logical_form) is True
        read_back(corpus, document, sentence)
    assert said > 0
    assert corpus.stats["causal"]["conditional_sentences"] == said


def test_false_causal_items_are_false_and_observed_marks_agree(heavy) -> None:
    corpus, oracle = heavy["corpus"], heavy["oracle"]
    scenes = list(scenes_json(corpus).values())
    kinds: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS:
            continue
        for true, false in test_set.pairs:
            for item in (true, false):
                form = item.input["logical_form"]
                assert oracle.causal(form) is item.meta["truth"], form
            observed = oracle.observed(false.input["logical_form"], scenes)
            assert false.meta["observed"] is observed
            assert (test_set.kind == LAWLIKE) == observed
            a, b = true.proposition, false.proposition
            if test_set.change == "condition":
                assert a.subject.condition and not b.subject.condition
                assert (a.subject.event_type, a.subject.role) == (
                    b.subject.event_type,
                    b.subject.role,
                )
                assert a.predicate == b.predicate
            elif test_set.change == "polarity" and a.predicate.numeric:
                assert {a.predicate.direction, b.predicate.direction} == {RAISES, LOWERS}
                assert a.subject == b.subject
            kinds[(test_set.change, a.predicate.numeric, bool(a.subject.condition))] += 1
    assert kinds[("condition", False, True)] + kinds[("condition", True, True)] > 0, kinds
    assert kinds[("polarity", True, False)] > 0, kinds


# ---------------------------------------------------------------------------------------------
# The test sets of the states
# ---------------------------------------------------------------------------------------------


def test_state_sets_hold_numeric_poles_and_the_comparison_set_swaps_the_participants(heavy):
    corpus, oracle = heavy["corpus"], heavy["oracle"]
    scenes = scenes_json(corpus)
    poles = comparisons = 0
    for test_set in corpus.test_sets:
        if test_set.level == STATE:
            for true, false in test_set.pairs:
                for item in (true, false):
                    form = item.input["logical_form"]
                    scene = scenes[form["scene"]]
                    assert oracle.state(form, scene) is item.meta["truth"]
                    predicate = form["predicate"]
                    if predicate["kind"] != COMPARISON:
                        continue
                    poles += 1
                    changed = oracle.changed(
                        scene, form["subject"]["instance"], predicate["label"], predicate["cut"]
                    )
                    assert item.meta["changed"] == changed
                assert true.proposition.predicate.kind == false.proposition.predicate.kind
        elif test_set.level == STATE_COMPARISON:
            assert test_set.name == "state_comparison_role" and test_set.change == ROLE
            for true, false in test_set.pairs:
                comparisons += 1
                a, b = true.proposition, false.proposition
                assert a.level == b.level == STATE and a.predicate.kind == COMPARATIVE
                assert (b.subject, b.predicate.patient) == (a.predicate.patient, a.subject)
                words = set()
                for item in (true, false):
                    form = item.input["logical_form"]
                    scene = scenes[form["scene"]]
                    assert form["time"] == scene["final"]
                    assert oracle.state(form, scene) is item.meta["truth"]
                    words.add(form["predicate"]["word"])
                    document = next(
                        d for d in corpus.documents if d.label == item.input["document"]
                    )
                    assert document.type == "situational"
                assert len(words) == 1
    assert poles > 0 and comparisons > 0


def test_comparative_items_draw_their_word_half_and_half(heavy) -> None:
    """Jon's ruling 3 on stage b3a: a comparative test item takes the "bigger" or the "smaller"
    word with equal chances, from the test sets' stream, both items of a pair alike."""
    corpus = heavy["corpus"]
    low = total = 0
    for test_set in corpus.test_sets:
        for true, false in test_set.pairs:
            if true.proposition.predicate.kind != COMPARATIVE:
                continue
            words = {item.input["logical_form"]["predicate"]["word"] for item in (true, false)}
            assert len(words) == 1
            total += 1
            low += words.pop().endswith(".LOW")
    assert total >= 40
    # a binomial share at 0.5: within 4 standard errors of a half
    margin = 4 * 0.5 / math.sqrt(total)
    assert abs(low / total - 0.5) <= margin, (low, total)


# ---------------------------------------------------------------------------------------------
# Topics, statistics, and the derived fluents of stage b2
# ---------------------------------------------------------------------------------------------


def test_numeric_fluents_are_topics_of_feature_documents(heavy) -> None:
    corpus = heavy["corpus"]
    facts = corpus.planner.facts
    assert facts.numeric_fluents
    about = [d for d in corpus.documents if d.topic in facts.numeric_fluents]
    assert about, "some feature document is about a numeric fluent"
    for document in about:
        assert document.type == "encyclopedic_feature"
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert proposition.causal and proposition.predicate.label == document.topic
            assert proposition.predicate.direction in (RAISES, LOWERS)


def test_the_statistics_count_the_numeric_and_conditional_sentences(heavy) -> None:
    corpus = heavy["corpus"]
    sentences = [s for _, s in sentences_of(corpus)]
    states, causal, comparisons = (
        corpus.stats["states"],
        corpus.stats["causal"],
        corpus.stats["comparisons"],
    )
    assert states["numeric_state_sentences"] == sum(
        s.proposition.level == STATE and s.proposition.predicate.kind == COMPARISON
        for s in sentences
    )
    assert states["numeric_change_sentences"] == sum(
        s.proposition.level == CHANGE and s.proposition.predicate.numeric for s in sentences
    )
    assert comparisons["numeric_comparative_sentences"] == sum(
        s.proposition.level == STATE and s.proposition.predicate.kind == COMPARATIVE
        for s in sentences
    )
    assert causal["raises_sentences"] == sum(
        s.proposition.predicate.direction == RAISES for s in sentences if s.proposition.causal
    )
    assert causal["lowers_sentences"] == sum(
        s.proposition.predicate.direction == LOWERS for s in sentences if s.proposition.causal
    )
    assert causal["conditional_sentences"] == sum(
        isinstance(s.proposition.subject, EventTerm) and bool(s.proposition.subject.condition)
        for s in sentences
    )
    assert states["numeric_state_sentences"] > 0 and states["numeric_change_sentences"] > 0
    assert comparisons["numeric_comparative_sentences"] > 0
    assert causal["raises_sentences"] + causal["lowers_sentences"] > 0
    assert corpus.stats["scenes"]["numeric_changes_per_event"] > 0
    assert corpus.stats["scenes"]["drift_changes_per_step"] > 0


def test_states_and_changes_of_derived_fluents_over_numeric_thresholds_are_judged_right(
    heavy,
) -> None:
    """Stage b2: the derived fluents that read a numeric threshold are stated in state and
    change sentences and in the state test sets, and the oracle, replaying the histories with
    the numeric values, agrees with every one."""
    corpus, oracle, world = heavy["corpus"], heavy["oracle"], heavy["world"]
    definition = world.definition
    numeric_readers = {
        f
        for f in definition.derived_fluents
        if any(definition.is_numeric_fluent(b) for b in definition.cone(f))
    }
    assert numeric_readers
    scenes = scenes_json(corpus)
    judged = 0
    for _, sentence in sentences_of(corpus):
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


def _mentions_numeric(value: Any) -> bool:
    return "NUMFL." in json.dumps(value)


def test_a_world_without_numeric_fluents_has_no_numeric_sentence(cases, runs) -> None:
    """The still world has no numeric fluents, so nothing of stage b3b appears in its corpus."""
    corpus = runs("still")
    lexicon = corpus.planner.lexicon
    assert not any(concept.label.startswith("NUMFL.") for concept in lexicon.concepts)
    for document in corpus.documents:
        for sentence in document.to_json()["sentences"]:
            assert not _mentions_numeric(sentence["logical_form"])
    assert corpus.stats["states"]["numeric_state_sentences"] == 0
    assert corpus.stats["causal"]["raises_sentences"] == 0
    assert not any(
        test_set.pairs for test_set in corpus.test_sets if test_set.level == STATE_COMPARISON
    )
