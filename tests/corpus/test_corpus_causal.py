# ruff: noqa: E501
"""Stage a7b acceptance tests: causal statements, descriptions, and the causal test sets.

Every causal statement is true by an independent re-reading of ``definition.json`` (the oracle,
never the corpus's ``Truth``), a statement about a category only when every event type below the
category has the effect or the literal, and no causal statement names a derived fluent. Every
false causal item is false under ``NEC``, and every ``observed`` mark agrees with the corpus's
scenes, by the oracle's replay. The propositional rendering parses back in both the ``marked``
and the ``omitted`` forms, and ``interpret(tree)`` recovers every logical form, causal
statements and descriptive marks included. Every state pair's record of which item changed
agrees with the history. The settings of the stage are settings.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from semantic_world.corpus import (
    Streams,
    build_lexicon,
    interpret,
    parse_propositional,
    proposition_of,
    propositional,
)
from semantic_world.corpus.config import ConfigError
from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_NP,
    GrammarError,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
    check_plan,
    event_phrase_of,
)
from semantic_world.corpus.lexicon import BEFORE_WORD, FUNCTION_WORDS
from semantic_world.corpus.logical import is_descriptive, logical_form
from semantic_world.corpus.planner import CAUSAL, TEMPLATE
from semantic_world.corpus.propositions import (
    AGENT,
    CAN,
    CAUSAL_KINDS,
    CLASS,
    EFFECT,
    EVENT_VARIABLE,
    NEC_ALL,
    PATIENT,
    PRECONDITION,
    PRESENT,
    VERB,
    EventTerm,
    Predicate,
    Proposition,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, leaves
from semantic_world.corpus.renderings import (
    BoundEvent,
    Described,
    Quantified,
    Temporal,
    assertion,
    formula,
    write,
)
from semantic_world.corpus.testsets import (
    BOTH_ITEMS,
    CAUSAL_LEVELS,
    EVENT_SWAP,
    FALSE_ITEM,
    LAWLIKE,
    POLARITY,
    PREDICATE,
    ROLE,
    TRUE_ITEM,
    TestSetBuilder,
    false_items,
    stated_propositions,
)

WORLDS = ("tiny", "default", "deep", "still")
MANY = {"propositions": {"causal_statement_rate": 0.8}}
"""Many causal statements, so that the tests see every kind."""


def scenes_json(corpus) -> dict[str, dict]:
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


def causal_sentences(corpus):
    return [
        (document, sentence)
        for document in corpus.documents
        for sentence in document.sentences
        if sentence.proposition.causal
    ]


# ---------------------------------------------------------------------------------------------
# Acceptance: every causal statement is true by the definition
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_causal_statements_are_true_by_a_rereading_of_the_definition(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    world = corpus.planner.world
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    event_types = {e["label"]: e for e in record["event_types"]}
    kinds: Counter = Counter()
    about_categories = 0
    for document, sentence in causal_sentences(corpus):
        proposition = sentence.proposition
        assert document.type == "encyclopedic_feature" and sentence.section == CAUSAL
        assert proposition.level == CLASS and proposition.quantifier == NEC_ALL
        assert proposition.polarity is True and isinstance(proposition.subject, EventTerm)
        assert proposition.predicate.kind in CAUSAL_KINDS
        assert sentence.plan.proposition() == proposition
        assert corpus.planner.truth.is_true(proposition), sentence.label
        form = json.loads(json.dumps(sentence.logical_form))
        # by the independent re-reading of definition.json
        assert oracle.causal(form) is True, form
        # no causal statement names a derived fluent
        fluent = proposition.predicate.label
        assert fluent in world.base_fluents and fluent not in world.derived_fluents
        # the JSON form: the event term, the predicate with its value, and the causal record
        assert form["subject"] == {
            "head": "THING",
            "event": proposition.subject.event_type,
            "role": proposition.subject.role,
        }
        assert form["predicate"] == {
            "kind": proposition.predicate.kind,
            "label": fluent,
            "value": proposition.predicate.value,
        }
        assert form["causal"] == {
            "event_type": proposition.subject.event_type,
            "role": proposition.subject.role,
            "fluent": fluent,
            "value": proposition.predicate.value,
        }
        assert form["grounding"]["test"] == "definition"
        # a statement about a category holds for every event type below it, by the record
        entry = event_types[proposition.subject.event_type]
        below = oracle.leaves_below(proposition.subject.event_type)
        if entry["kind"] == "event_type_category":
            about_categories += 1
            assert len(below) > 1
        for leaf in below:
            entries = (
                event_types[leaf]["effects"]
                if proposition.predicate.kind == EFFECT
                else event_types[leaf]["precondition"]["literals"]
            )
            # an effect with a condition is no entry (stage b1)
            assert any(
                e["role"] == proposition.subject.role
                and e["fluent"] == fluent
                and e["value"] == proposition.predicate.value
                and not e.get("condition")
                for e in entries
            ), (leaf, form)
        kinds[proposition.predicate.kind] += 1
    assert kinds[EFFECT] > 5 and kinds[PRECONDITION] > 2, kinds
    if name == "default":
        assert about_categories > 0


@pytest.mark.parametrize("name", WORLDS)
def test_every_true_causal_statement_of_the_world_is_enumerated(cases, runs, name) -> None:
    """``Facts.causal_statements`` lists exactly the definition's entries about event types and
    base fluents with a word, categories included when every leaf below shares the entry."""
    case = cases(name)
    corpus = runs(name)
    facts = corpus.planner.facts
    oracle = case.oracle()
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    expected = set()
    for entry in record["event_types"]:
        label = entry["label"]
        if label not in facts.named:
            continue
        for kind in CAUSAL_KINDS:
            found = None
            for leaf in oracle.leaves_below(label):
                entries = (
                    record_entries(record, leaf, "effects")
                    if kind == EFFECT
                    else record_entries(record, leaf, "literals")
                )
                keys = {(e["role"], e["fluent"], e["value"]) for e in entries}
                found = keys if found is None else found & keys
            for role, fluent, value in found or ():
                if fluent in facts.base_fluents and oracle.able_somewhere(label):
                    expected.add((label, role, fluent, value, kind))
    listed = {
        (
            p.subject.event_type,
            p.subject.role,
            p.predicate.label,
            p.predicate.value,
            p.predicate.kind,
        )
        for p in facts.causal_statements()
    }
    assert listed == expected
    assert len(facts.causal_statements(kind=EFFECT)) + len(
        facts.causal_statements(kind=PRECONDITION)
    ) == len(listed)
    for statement in facts.causal_statements():
        assert oracle.causal(statement.to_json()) is True


def record_entries(record: dict, label: str, key: str) -> list[dict]:
    """The unconditional effects, or the precondition literals, of an event type's record."""
    entry = next(e for e in record["event_types"] if e["label"] == label)
    if key == "effects":
        return [e for e in entry["effects"] if not e.get("condition")]
    return entry["precondition"]["literals"]


def test_no_causal_statement_is_true_by_a_conditional_effect(cases, runs) -> None:
    """Stage b1: an effect with a condition guarantees nothing after every event, so the
    statement about it is not enumerated, is judged false by ``Truth``, and is judged false by
    the oracle's independent re-reading of ``definition.json``. The default world has
    conditional effects at the default share."""
    case = cases("default")
    corpus = runs("default")
    facts, truth = corpus.planner.facts, corpus.planner.truth
    oracle = case.oracle()
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    conditional = [
        (entry["label"], effect)
        for entry in record["event_types"]
        for effect in entry["effects"]
        if effect.get("condition")
    ]
    assert conditional, "the default world has conditional effects"
    listed = {
        (p.subject.event_type, p.subject.role, p.predicate.label, p.predicate.value)
        for p in facts.causal_statements(kind=EFFECT)
    }
    checked = 0
    for label, effect in conditional:
        if label not in facts.named or effect["fluent"] not in facts.base_fluents:
            continue
        key = (label, effect["role"], effect["fluent"], effect["value"])
        assert key not in listed, key
        statement = Proposition(
            CLASS,
            EventTerm(label, effect["role"]),
            Predicate(EFFECT, effect["fluent"], value=effect["value"]),
            True,
            NEC_ALL,
        )
        evaluation = truth.evaluate(statement)
        # the statement is well formed, and false: the entry is conditional
        if not evaluation.valid:
            assert "no binding is able" in evaluation.reason
            continue
        assert evaluation.true is False
        assert evaluation.grounding["with_entry"] < evaluation.grounding["event_types"]
        assert oracle.causal(statement.to_json()) is False
        checked += 1
    assert checked > 0
    # no sentence of the corpus and no true test item states one
    for _, sentence in causal_sentences(corpus):
        form = sentence.logical_form
        assert oracle.causal(form) is True
    for test_set in corpus.test_sets:
        if test_set.level in CAUSAL_LEVELS:
            for true, _ in test_set.pairs:
                assert oracle.causal(true.input["logical_form"]) is True


def test_capped_causal_sets_keep_every_statement(runs) -> None:
    """Stage b1 (Jon's ruling 1 on stage a8): a causal set with more pairs than
    ``test_sets.size`` draws its pairs round-robin over its true statements, so every statement
    appears before any appears twice; both items carry the structured statement record."""
    corpus = runs("default")
    facts, truth = corpus.planner.facts, corpus.planner.truth
    size = corpus.config.test_sets.size
    capped = 0
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS:
            continue
        kind = EFFECT if test_set.level == "causal_effect" else PRECONDITION
        for true, false in test_set.pairs:
            record = {"kind": kind, **true.meta["causal"]}
            assert true.meta["statement_record"] == record
            assert false.meta["statement_record"] == record
            assert record["event_type"] == true.proposition.subject.event_type
        every: dict = {}
        for true in facts.causal_statements(kind=kind):
            for false in false_items(facts, true, test_set.change):
                group = LAWLIKE if truth.observed(false) else ""
                if group == test_set.kind:
                    every.setdefault(true, []).append(false)
        if sum(len(v) for v in every.values()) <= size:
            continue
        capped += 1
        counts = Counter(true.proposition for true, _ in test_set.pairs)
        assert len(test_set.pairs) == size
        if len(every) <= size:
            assert set(counts) == set(every), test_set.name
        else:
            assert len(counts) == size
        assert max(counts.values()) - min(counts.values()) <= 1, test_set.name
        # a statement with more pairs than the rounds give is not exhausted before the others
        rounds = max(counts.values())
        for statement, pairs in every.items():
            if statement in counts and counts[statement] < rounds:
                assert counts[statement] == len(pairs) or counts[statement] == rounds - 1
    assert capped > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: the causal test sets
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_false_causal_items_are_false_under_nec_and_observed_marks_agree(cases, runs, name):
    case = cases(name)
    corpus = runs(name)
    oracle = case.oracle()
    scenes = list(scenes_json(corpus).values())
    world = corpus.planner.world
    kinds: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS:
            continue
        kind = EFFECT if test_set.level == "causal_effect" else PRECONDITION
        for true, false in test_set.pairs:
            for item in (true, false):
                form = item.input["logical_form"]
                assert form["level"] == CLASS and "head" in form["subject"]
                assert form["predicate"]["kind"] == kind
                assert "causal" not in form and item.meta["causal"] == {
                    "event_type": form["subject"]["event"],
                    "role": form["subject"]["role"],
                    "fluent": form["predicate"]["label"],
                    "value": form["predicate"]["value"],
                }
                assert item.input["document"] is None
                assert form["predicate"]["label"] in world.base_fluents
                # true by the definition, or false under NEC: the definition lacks the entry
                assert oracle.causal(form) is item.meta["truth"], form
            assert true.meta["observed"] is False
            # the observed mark, by the oracle's replay of every scene of the corpus
            observed = oracle.observed(false.input["logical_form"], scenes)
            assert false.meta["observed"] is observed
            assert (test_set.kind == LAWLIKE) == observed
            a, b = true.proposition, false.proposition
            if test_set.change == ROLE:
                assert not a.subject.event_type.startswith("EVENTTYPE1.")
            elif test_set.change == EVENT_SWAP:
                info_a = world.event_types[a.subject.event_type]
                info_b = world.event_types[b.subject.event_type]
                assert (info_a.arity, info_a.level) == (info_b.arity, info_b.level)
            kinds[(test_set.level, test_set.change, test_set.kind)] += 1
    for level in CAUSAL_LEVELS:
        for change in (PREDICATE, POLARITY, EVENT_SWAP):
            assert kinds[(level, change, "")] > 0, (level, change, kinds)
    if name == "default":
        assert kinds[("causal_effect", ROLE, "")] > 0
        assert sum(count for (_, _, kind), count in kinds.items() if kind == LAWLIKE) > 0


@pytest.mark.parametrize("name", WORLDS)
def test_the_causal_sets_pair_every_statement_with_every_valid_false_item(runs, name) -> None:
    """Stage a8 (Jon's ruling 3 on stage a7b): each true causal statement is paired with every
    valid false item of the set's change, the pairs are capped at ``test_sets.size``, both items
    record the statement they test, and the polarity change has no law-like twin."""
    corpus = runs(name)
    facts, truth = corpus.planner.facts, corpus.planner.truth
    size = corpus.config.test_sets.size
    by_group: dict = {}
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS:
            continue
        assert len(test_set.pairs) <= size
        if test_set.change == POLARITY:
            assert test_set.kind != LAWLIKE
        kind = EFFECT if test_set.level == "causal_effect" else PRECONDITION
        for true, false in test_set.pairs:
            assert true.proposition.predicate.kind == kind
            assert true.proposition in facts.causal_statements(kind=kind)
            # the statement record is the true item's rendering, on both items
            assert true.meta["statement"] == true.input["propositional"]
            assert false.meta["statement"] == true.input["propositional"]
            assert true.meta["statement"].startswith("NEC(ALL(EVENT(EVENTVAR.1, ")
            by_group.setdefault((test_set.level, test_set.change, test_set.kind), []).append(
                (true.proposition, false.proposition)
            )
        stats = test_set.stats()
        assert stats["true_statements"] == len({t.meta["statement"] for t, _ in test_set.pairs})
    # every pair of a true statement and a valid false item is in a set, when the set is not
    # capped; a capped set holds a subset, in enumeration order
    for level in CAUSAL_LEVELS:
        kind = EFFECT if level == "causal_effect" else PRECONDITION
        for change in (PREDICATE, POLARITY, EVENT_SWAP, ROLE):
            expected: dict = {}
            for true in facts.causal_statements(kind=kind):
                for false in false_items(facts, true, change):
                    group = LAWLIKE if truth.observed(false) else ""
                    expected.setdefault(group, []).append((true, false))
            for group, pairs in expected.items():
                made = by_group.get((level, change, group), [])
                if len(pairs) <= size:
                    assert made == pairs, (level, change, group)
                else:
                    assert len(made) == size
                    positions = [pairs.index(pair) for pair in made]
                    assert positions == sorted(positions) and len(set(positions)) == size
            for group in ("", LAWLIKE):
                if group not in expected:
                    assert by_group.get((level, change, group), []) == []
    # the same configuration and seed give the same pairs
    again = TestSetBuilder(corpus.planner, corpus.documents).build()
    for a, b in zip(corpus.test_sets, again, strict=True):
        if a.level in CAUSAL_LEVELS:
            assert [(t.proposition, f.proposition) for t, f in a.pairs] == [
                (t.proposition, f.proposition) for t, f in b.pairs
            ]
            assert [(t.meta, f.meta) for t, f in a.pairs] == [(t.meta, f.meta) for t, f in b.pairs]


def test_the_causal_sets_are_capped_at_the_size(runs) -> None:
    corpus = runs("default", test_sets={"size": 3})
    facts = corpus.planner.facts
    for test_set in corpus.test_sets:
        if test_set.level in CAUSAL_LEVELS:
            assert len(test_set.pairs) <= 3
    # the default world has more effect statements than the test runs' size of 30, so the
    # predicate set is full and, under the stratified draw of stage b1, every kept pair tests
    # another statement; a size above the number of statements lets a statement recur
    full = runs("default")
    predicate = next(s for s in full.test_sets if s.name == "causal_effect_predicate")
    assert len(predicate.pairs) == full.config.test_sets.size == 30
    assert len({t.meta["statement"] for t, _ in predicate.pairs}) == len(predicate.pairs)
    statements = len(facts.causal_statements(kind=EFFECT))
    assert statements > 30
    wide = runs("default", test_sets={"size": statements + 20})
    predicate = next(s for s in wide.test_sets if s.name == "causal_effect_predicate")
    counts = Counter(t.meta["statement"] for t, _ in predicate.pairs)
    assert len(predicate.pairs) == statements + 20 or len(counts) == len(
        {p for p in facts.causal_statements(kind=EFFECT)}
    )
    assert max(counts.values()) > 1 and max(counts.values()) - min(counts.values()) <= 1


def test_an_observed_false_item_held_of_every_event_in_the_scenes(runs) -> None:
    corpus = runs("default")
    truth = corpus.planner.truth
    found = 0
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS or test_set.kind != LAWLIKE:
            continue
        for _, false in test_set.pairs:
            subject, predicate = false.proposition.subject, false.proposition.predicate
            below = set(truth.world.event_types_below(subject.event_type))
            offset = 1 if predicate.kind == EFFECT else 0
            events = 0
            for scene in corpus.planner.scenes:
                for event in truth.events_of(scene.label):
                    if event.type not in below:
                        continue
                    entity = event.agent if subject.role == AGENT else event.patient
                    events += 1
                    value = truth.fluent_value(
                        scene.label, event.step + offset, entity, predicate.label
                    )
                    assert value == predicate.value
            assert events > 0  # the mark is never vacuous
            found += 1
    assert found > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: descriptions, and the renderings parse back
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_descriptions_in_the_logical_form_and_the_rendering(runs, name) -> None:
    corpus = runs(name)
    mode = corpus.config.propositional_referents
    described = plain = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            form = json.loads(json.dumps(sentence.logical_form))
            plan = sentence.plan
            if sentence.proposition.level == CLASS:
                assert "descriptive" not in json.dumps(form)
                continue
            # every mention says whether it describes: a definite mention does, an indefinite
            # one and a pronoun do not
            mentions = mentions_of(form)
            phrases = plan.noun_phrases()
            assert len(mentions) == len(phrases)
            for mention, phrase in zip(mentions, phrases, strict=True):
                assert mention["descriptive"] is (phrase.determiner == "the")
                assert mention["descriptive"] is is_descriptive(phrase)
                if phrase.pronoun:
                    assert mention["descriptive"] is False
            # the marked rendering parses back to the full logical form, and the omitted one
            # to the logical form with its descriptions removed
            marked = sentence.propositional
            assert marked == propositional(form, mode, "marked")
            parsed = parse_propositional(marked)
            assert parsed == formula(form, mode) and write(parsed) == marked
            omitted = propositional(form, mode, "omitted")
            parsed_omitted = parse_propositional(omitted)
            assert parsed_omitted == assertion(form, mode) == formula(form, mode, "omitted")
            assert write(parsed_omitted) == omitted
            names = document.referents if mode == "local" else None
            assert proposition_of(parsed, names) == sentence.proposition
            assert proposition_of(parsed_omitted, names) == sentence.proposition
            if isinstance(parsed[0], Described):
                described += 1
                assert marked.startswith("{") and "} " in marked
                assert omitted == marked.split("} ", 1)[1]
                assert parsed[1:] == parsed_omitted
                # the braces hold what the descriptive mentions contribute
                assert all(m["descriptive"] for m in mentions if m["noun"] is not None) or any(
                    not m["descriptive"] for m in mentions
                )
            else:
                plain += 1
                assert "{" not in marked and omitted == marked
                assert parsed == parsed_omitted
            # the tree reads back as the plan, descriptions and all
            record = sentence.sentence
            assert (
                interpret(record.tree, corpus.planner.lexicon, record.referents, record.events)
                == plan
            )
    assert described > 20 and plain > 20


def mentions_of(form: dict) -> list[dict]:
    found: list[dict] = []

    def visit(mention: dict) -> None:
        found.append(mention)
        for clause in mention.get("clauses", ()):
            other = clause.get("agent") or clause.get("patient")
            if other is not None:
                visit(other)

    visit(form["subject"])
    if "patient" in form["predicate"]:
        visit(form["predicate"]["patient"])
    return found


def test_the_descriptions_setting_changes_the_propositional_rendering_only(runs) -> None:
    marked = runs("default", 150)
    omitted = runs("default", 150, renderings={"propositional": {"descriptions": "omitted"}})
    changed = 0
    for a, b in zip(marked.documents, omitted.documents, strict=True):
        assert a.to_json().keys() == b.to_json().keys()
        for x, y in zip(a.sentences, b.sentences, strict=True):
            left, right = x.to_json(), y.to_json()
            assert {k: v for k, v in left.items() if k != "propositional"} == {
                k: v for k, v in right.items() if k != "propositional"
            }
            if left["propositional"] != right["propositional"]:
                changed += 1
                assert left["propositional"].startswith("{")
                assert "{" not in right["propositional"]
                assert right["propositional"] == left["propositional"].split("} ", 1)[1]
    assert changed > 50
    for s, t in zip(marked.test_sets, omitted.test_sets, strict=True):
        assert s.name == t.name and len(s.pairs) == len(t.pairs)
        for x, y in zip(s.items, t.items, strict=True):
            assert x.meta == y.meta
            assert {k: v for k, v in x.input.items() if k != "propositional"} == {
                k: v for k, v in y.input.items() if k != "propositional"
            }
            assert "{" not in y.input["propositional"]
    # test items are definite mentions, so their renderings carry braces when marked
    assert any("{" in item.input["propositional"] for s in marked.test_sets for item in s.items)
    with pytest.raises(ConfigError, match="renderings.propositional.descriptions"):
        runs("tiny", 5, renderings={"propositional": {"descriptions": "braces"}})


def test_seen_counts_descriptions_by_default_and_not_with_the_setting(runs) -> None:
    # 600 documents: in the default world of stage b1 (with its conditional effects) the
    # 200-document run has no item seen through a description alone
    counted = runs("default", 600)
    uncounted = runs("default", 600, test_sets={"size": 30, "seen": {"descriptions": False}})
    # the documents are the same; only the seen marks can differ, and only downward
    assert [d.to_json() for d in counted.documents] == [d.to_json() for d in uncounted.documents]
    stated_all = stated_propositions(counted.documents)
    stated_asserted = stated_propositions(counted.documents, descriptions=False)
    assert stated_asserted < stated_all
    fewer = 0
    for s, t in zip(counted.test_sets, uncounted.test_sets, strict=True):
        assert s.name == t.name
        for x, y in zip(s.items, t.items, strict=True):
            assert x.input == y.input and x.proposition == y.proposition
            assert {k: v for k, v in x.meta.items() if k != "seen"} == {
                k: v for k, v in y.meta.items() if k != "seen"
            }
            assert y.meta["seen"] <= x.meta["seen"]
            fewer += x.meta["seen"] and not y.meta["seen"]
            # an item is seen when its proposition is among what the documents state
            assert y.meta["seen"] == (y.truth and stated_form(y.proposition) in stated_asserted)
    assert fewer > 0
    # what a description says of its referent is not stated without the setting
    for document in counted.documents:
        for sentence in document.sentences:
            if sentence.proposition.level == CLASS:
                continue
            for phrase in sentence.plan.noun_phrases():
                if phrase.noun is None:
                    continue
                member = Proposition("instance", phrase.referent, Predicate("member", phrase.noun))
                assert member in stated_all
                if not is_descriptive(phrase):
                    assert member in stated_asserted


def stated_form(proposition: Proposition) -> Proposition:
    import dataclasses

    if proposition.level == "event":
        return dataclasses.replace(proposition, event=None, aspect="simple")
    return proposition


# ---------------------------------------------------------------------------------------------
# Acceptance: which item of a state pair changed
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_record_of_which_state_item_changed_agrees_with_the_history(cases, runs, name):
    corpus = runs(name)
    oracle = cases(name).oracle()
    scenes = scenes_json(corpus)
    seen: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level != "state":
            continue
        counts: Counter = Counter()
        for true, false in test_set.pairs:
            marks = []
            for item in (true, false):
                form = item.input["logical_form"]
                changed = oracle.changed(
                    scenes[form["scene"]], form["subject"]["instance"], form["predicate"]["label"]
                )
                assert item.meta["changed"] == changed
                marks.append(changed)
            expected = {
                (True, True): BOTH_ITEMS,
                (True, False): TRUE_ITEM,
                (False, True): FALSE_ITEM,
                (False, False): None,
            }[tuple(marks)]
            assert true.meta["changed_item"] == false.meta["changed_item"] == expected
            assert (expected is None) == (test_set.kind == "unchanged")
            counts[expected or "neither"] += 1
            seen[expected] += 1
        stats = corpus.stats["test_sets"][test_set.name]
        assert stats["pairs_by_changed_item"] == {
            kind: counts[kind] for kind in ("true_item", "false_item", "both", "neither")
        }
        assert sum(stats["pairs_by_changed_item"].values()) == len(test_set.pairs)
    assert seen[TRUE_ITEM] > 0 and seen[FALSE_ITEM] > 0 and seen[None] > 0, seen
    for test_set in corpus.test_sets:
        if test_set.level != "state":
            assert all("changed_item" not in item.meta for item in test_set.items)


# ---------------------------------------------------------------------------------------------
# The documents, the statistics, and the settings
# ---------------------------------------------------------------------------------------------


def test_feature_documents_about_event_types_and_fluents(runs) -> None:
    corpus = runs("default", 300, **MANY)
    facts = corpus.planner.facts
    topics: Counter = Counter()
    for document in corpus.documents:
        if document.type != "encyclopedic_feature":
            continue
        topic = document.topic
        causal = [s for s in document.sentences if s.proposition.causal]
        others = [s for s in document.sentences if not s.proposition.causal]
        if topic in facts.base_fluents:
            # a document about a fluent: the event types that set it, clear it, and need it,
            # one causal statement per sentence, up to its drawn length or the statements
            # about its fluent (stage a8, Jon's ruling 1 on stage a7b)
            topics["fluent"] += 1
            assert causal and not others
            assert all(s.proposition.predicate.label == topic for s in causal)
            statements = facts.causal_statements(fluent=topic)
            assert len(document.sentences) == min(document.drawn_length, len(statements))
        elif topic in facts.verbs or topic in facts.features[CAN]:
            topics["event_type"] += 1
            assert all(s.proposition.subject.event_type == topic for s in causal)
        else:
            topics["feature"] += 1
            assert not causal
        # a document says each causal statement once
        said = [s.proposition for s in causal]
        assert len(set(said)) == len(said)
    assert topics["fluent"] >= 2 and topics["event_type"] > 10 and topics["feature"] > 10, topics


def test_the_causal_statement_rate_is_a_setting(runs) -> None:
    def counts(rate: float) -> Counter:
        corpus = runs("default", 200, propositions={"causal_statement_rate": rate})
        found: Counter = Counter()
        for document in corpus.documents:
            if document.type != "encyclopedic_feature":
                continue
            topic = document.topic
            if topic in corpus.planner.facts.base_fluents:
                found["fluent_documents"] += 1
                found["fluent_sentences"] += len(document.sentences)
            else:
                found["sentences"] += len(document.sentences)
                found["causal"] += sum(s.proposition.causal for s in document.sentences)
        return found

    none = counts(0.0)
    assert none["causal"] == 0
    low, high = counts(0.2), counts(0.9)
    assert 0 < low["causal"] < high["causal"]
    assert low["causal"] / low["sentences"] < high["causal"] / high["sentences"]
    # the rate applies to documents about event types and categories only: a document about a
    # fluent holds causal statements alone at any rate, and says the same at every rate
    assert none["fluent_documents"] == low["fluent_documents"] == high["fluent_documents"] > 0
    assert none["fluent_sentences"] == low["fluent_sentences"] == high["fluent_sentences"] > 0
    with pytest.raises(ConfigError, match="propositions.causal_statement_rate"):
        runs("tiny", 5, propositions={"causal_statement_rate": 2})


def test_the_statistics_count_causal_sentences(runs) -> None:
    corpus = runs("default", 200, **MANY)
    block = corpus.stats["causal"]
    sentences = [s for d in corpus.documents for s in d.sentences if s.proposition.causal]
    assert block["causal_sentences"] == len(sentences) > 0
    assert block["effect_sentences"] == sum(
        s.proposition.predicate.kind == EFFECT for s in sentences
    )
    assert block["precondition_sentences"] == sum(
        s.proposition.predicate.kind == PRECONDITION for s in sentences
    )
    assert block["effect_sentences"] + block["precondition_sentences"] == block["causal_sentences"]
    assert all(s.section == CAUSAL for s in sentences)
    assert corpus.stats["sentences"]["by_section"][CAUSAL] == len(sentences)
    # a causal sentence is a class-level nec_all sentence in the quantifier mix
    assert corpus.stats["quantifiers"]["stated"]["nec_all"]["count"] >= len(sentences)
    assert corpus.stats["ambiguity"]["readings"]["causal"] == len(sentences)


def test_readings_and_renderings_of_causal_sentences(runs) -> None:
    corpus = runs("default", 200, **MANY)
    lexicon = corpus.planner.lexicon
    operators: Counter = Counter()
    for _, sentence in causal_sentences(corpus):
        assert sentence.readings[0] == "causal" and "generic" not in sentence.readings
        # the quantifier readings of the words follow: "all" allows nec_all, the bare plural
        # nec_all and most
        if sentence.plan.bare_plural:
            assert sentence.readings[1:] == ("nec_all", "most")
        else:
            assert sentence.readings[1:] == ("nec_all",)
        text = sentence.propositional
        assert text.startswith("NEC(ALL(EVENT(EVENTVAR.1, ")
        parsed = parse_propositional(text)
        (main,) = parsed
        assert isinstance(main, Quantified) and main.quantifier == NEC_ALL
        assert isinstance(main.restrictor[0], BoundEvent) and isinstance(main.scope, Temporal)
        assert main.scope.operator == (
            "AFTER" if sentence.proposition.predicate.kind == EFFECT else "BEFORE"
        )
        assert proposition_of(parsed) == sentence.proposition
        assert Proposition.from_json(sentence.logical_form) == sentence.proposition
        operators[main.scope.operator] += 1
        record = sentence.sentence
        assert record.events == (None, None)  # no verb phrase reports an event
        read = interpret(record.tree, lexicon, record.referents, record.events, NEC_ALL)
        assert read == sentence.plan
        assert tuple(leaves(record.tree)) == record.tokens
        # the words: the generic noun, the event's verb, become or the copula with before
        glosses = [lexicon.lexeme(t).gloss for t in record.tokens]
        assert glosses.count("THING") >= 1 and "that" in glosses
        assert "can" not in glosses
        if sentence.proposition.predicate.kind == EFFECT:
            assert "become" in glosses and BEFORE_WORD not in glosses
        else:
            assert glosses[-1] == BEFORE_WORD and "is" in glosses and "become" not in glosses
    assert operators["AFTER"] > 5 and operators["BEFORE"] > 2


def test_the_grammar_of_causal_statements(cases) -> None:
    tiny = cases("tiny")
    world = tiny.world
    fluent = world.base_fluents[0]
    unary, binary = world.unary[0], world.binary_leaves[0]

    def say(plan: SentencePlan, **settings) -> tuple[str, list]:
        config = tiny.config(**settings)
        lexicon = build_lexicon(config, world, Streams(config.seed))
        realizer = Realizer(config, lexicon, Streams(config.seed))
        check_plan(plan)
        sentence = realizer.realize(plan, Streams(7).grammar)
        assert (
            interpret(sentence.tree, lexicon, sentence.referents, sentence.events, NEC_ALL) == plan
        )
        assert sentence.events == (None, None)
        return realizer.conceptual(sentence.tokens), list(readings(sentence.tree, lexicon, config))

    agent = event_phrase_of(EventTerm(binary, AGENT), "all")
    patient = event_phrase_of(EventTerm(binary, PATIENT))
    sleeper = event_phrase_of(EventTerm(unary, AGENT))
    effect = Predication(EFFECT, fluent, True)
    cleared = Predication(EFFECT, fluent, False)
    needs = Predication(PRECONDITION, fluent, True)
    lacks = Predication(PRECONDITION, fluent, False)
    assert say(SentencePlan(agent, effect, NEC_ALL)) == (
        f"ALL THING THAT {binary} THING BECOME {fluent}",
        ["causal", "nec_all"],
    )
    assert say(SentencePlan(patient, cleared, NEC_ALL)) == (
        f"THING THAT THING {binary} BECOME NOT {fluent}",
        ["causal", "nec_all", "most"],
    )
    assert say(SentencePlan(sleeper, needs, NEC_ALL)) == (
        f"THING THAT {unary} IS {fluent} BEFORE",
        ["causal", "nec_all", "most"],
    )
    assert say(SentencePlan(patient, lacks, NEC_ALL)) == (
        f"THING THAT THING {binary} IS NOT {fluent} BEFORE",
        ["causal", "nec_all", "most"],
    )
    # where "before" stands is a word-order setting
    first = {"grammar": {"word_order": {"before": "before_predicate"}}}
    assert say(SentencePlan(sleeper, lacks, NEC_ALL), **first)[0] == (
        f"THING THAT {unary} BEFORE IS NOT {fluent}"
    )
    # with number on, the generic noun is plural and the clause's verb agrees; the tense is the
    # present, so nothing marks it
    number = {"grammar": {"morphology": {"number": {"enabled": True}, "tense": {"enabled": True}}}}
    assert say(SentencePlan(agent, effect, NEC_ALL), **number)[0] == (
        f"ALL THING-PLURAL THAT {binary}-PLURAL THING-PLURAL BECOME {fluent}"
    )
    assert say(SentencePlan(sleeper, needs, NEC_ALL), **number)[0] == (
        f"THING-PLURAL THAT {unary}-PLURAL ARE {fluent} BEFORE"
    )
    # the plan says the proposition
    proposition = SentencePlan(patient, cleared, NEC_ALL).proposition()
    assert proposition == Proposition(
        CLASS, EventTerm(binary, PATIENT), Predicate(EFFECT, fluent, value=False), True, NEC_ALL
    )
    assert proposition.causal and proposition.causal_record() == {
        "event_type": binary,
        "role": PATIENT,
        "fluent": fluent,
        "value": False,
    }
    assert logical_form(SentencePlan(patient, cleared, NEC_ALL), proposition)["subject"] == {
        "head": "THING",
        "event": binary,
        "role": PATIENT,
    }


def test_causal_plans_the_grammar_rejects(cases) -> None:
    world = cases("tiny").world
    fluent, unary, binary = world.base_fluents[0], world.unary[0], world.binary_leaves[0]
    things = NounPhrase(CLASS_NP, "THING", "THING")
    bound = Predication(VERB, binary, True, things, EVENT_VARIABLE, PRESENT)
    subject = NounPhrase(CLASS_NP, "THING", "THING", None, (), RelativeClause((bound,)))
    good = SentencePlan(subject, Predication(EFFECT, fluent), NEC_ALL)
    check_plan(good)
    bad = [
        SentencePlan(subject, Predication(EFFECT, fluent)),  # nec_all
        SentencePlan(subject, Predication(EFFECT, fluent), "most"),
        SentencePlan(things, Predication(EFFECT, fluent), NEC_ALL),  # no event clause
        SentencePlan(  # a capacity clause, not an event
            NounPhrase(
                CLASS_NP,
                "THING",
                "THING",
                None,
                (),
                RelativeClause((Predication(VERB, binary, True, things),)),
            ),
            Predication(EFFECT, fluent),
            NEC_ALL,
        ),
        SentencePlan(  # the other participant is the bare generic noun
            NounPhrase(
                CLASS_NP,
                "THING",
                "THING",
                None,
                (),
                RelativeClause(
                    (
                        Predication(
                            VERB,
                            binary,
                            True,
                            NounPhrase(CLASS_NP, "CATEGORY.1", "CATEGORY.1"),
                            EVENT_VARIABLE,
                            PRESENT,
                        ),
                    )
                ),
            ),
            Predication(EFFECT, fluent),
            NEC_ALL,
        ),
        SentencePlan(subject, Predication(EFFECT, fluent, object=things), NEC_ALL),
        SentencePlan(subject, Predication(EFFECT, fluent, tense="past"), NEC_ALL),
        SentencePlan(  # the clause's verb is in the present tense
            NounPhrase(
                CLASS_NP,
                "THING",
                "THING",
                None,
                (),
                RelativeClause((Predication(VERB, binary, True, things, EVENT_VARIABLE, "past"),)),
            ),
            Predication(EFFECT, fluent),
            NEC_ALL,
        ),
        SentencePlan(  # an effect is the predicate of a causal statement, never of a clause
            NounPhrase(INSTANCE_NP, world.instances[0], world.instance_leaf[0], "the"),
            Predication(EFFECT, fluent),
            None,
        ),
    ]
    for plan in bad:
        with pytest.raises(GrammarError):
            check_plan(plan)
    # a bound verb outside a causal statement
    with pytest.raises(GrammarError):
        check_plan(SentencePlan(subject, Predication(CAN, unary), NEC_ALL))


def test_causal_propositions_the_truth_tests_reject(cases) -> None:
    tiny = cases("tiny")
    truth = tiny.facts().truth
    world = tiny.world
    unary, binary = world.unary[0], world.binary_leaves[0]
    fluent = world.base_fluents[0]

    def judge(**fields):
        base = {
            "level": CLASS,
            "subject": EventTerm(binary, AGENT),
            "predicate": Predicate(EFFECT, fluent, value=True),
            "polarity": True,
            "quantifier": NEC_ALL,
        }
        return truth.evaluate(Proposition(**{**base, **fields}))

    assert judge().valid
    assert judge().grounding["test"] == "definition"
    if world.derived_fluents:
        derived = judge(predicate=Predicate(EFFECT, world.derived_fluents[0], value=True))
        assert not derived.valid and "derived" in derived.reason
    assert not judge(subject=EventTerm(unary, PATIENT)).valid  # a one-place event type
    assert not judge(subject=EventTerm("EVENTTYPE2.9.9", AGENT)).valid
    assert not judge(quantifier="all").valid
    assert not judge(quantifier="most").valid
    assert not judge(polarity=False).valid
    assert not judge(predicate=Predicate(EFFECT, fluent)).valid  # a value is needed
    assert not judge(predicate=Predicate(EFFECT, fluent, patient="CATEGORY.1", value=True)).valid
    assert not judge(
        predicate=Predicate("property", world.features["property"][0], value=True)
    ).valid
    # the truth: the definition has the entry, or not
    record = world.definition.event_type(binary)
    for effect in record.effects:
        found = judge(
            subject=EventTerm(binary, effect.role),
            predicate=Predicate(EFFECT, effect.fluent, value=effect.value),
        )
        assert found.valid and found.true
        flipped = judge(
            subject=EventTerm(binary, effect.role),
            predicate=Predicate(EFFECT, effect.fluent, value=not effect.value),
        )
        assert flipped.valid and not flipped.true


def test_the_function_word_before(cases, runs) -> None:
    corpus = runs("tiny")
    lexicon = corpus.planner.lexicon
    assert FUNCTION_WORDS[-2:] == ("become", BEFORE_WORD)
    before = lexicon.function_word(BEFORE_WORD)
    assert before.concept == "BEFORE" and not before.content
    # appended after become: the words before it keep their labels
    glosses = [x.gloss for x in lexicon.function_lexemes]
    assert glosses.index(BEFORE_WORD) == glosses.index("become") + 1
    from semantic_world.corpus.request import wordform_request

    request = wordform_request(corpus)
    assert BEFORE_WORD in request["function_words"]
    assert len(request["function_words"]) == len(lexicon.function_lexemes)


def test_the_renderings_of_the_specification_examples() -> None:
    catch = "NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), AFTER(EVENTVAR.1, BOOLFL.3(VAR.2))))"
    parsed = parse_propositional(catch)
    assert write(parsed) == catch
    assert proposition_of(parsed) == Proposition(
        CLASS,
        EventTerm("EVENTTYPE2.1.2", PATIENT),
        Predicate(EFFECT, "BOOLFL.3", value=True),
        True,
        NEC_ALL,
    )
    awake = "NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), BEFORE(EVENTVAR.1, BOOLFL.1(VAR.1))))"
    assert proposition_of(parse_propositional(awake)) == Proposition(
        CLASS,
        EventTerm("EVENTTYPE2.1.2", AGENT),
        Predicate(PRECONDITION, "BOOLFL.1", value=True),
        True,
        NEC_ALL,
    )
    cleared = (
        "NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE1.3(VAR.1)), AFTER(EVENTVAR.1, NOT BOOLFL.3(VAR.1))))"
    )
    proposition = proposition_of(parse_propositional(cleared))
    assert proposition.subject == EventTerm("EVENTTYPE1.3", AGENT)
    assert proposition.predicate == Predicate(EFFECT, "BOOLFL.3", value=False)
    assert propositional(proposition.to_json()) == cleared
    # the descriptions of the specification's table
    furry = "{CATEGORY.1.3.2(REF.1) AND PROPERTY.12(REF.1)} PART.4(REF.1)"
    parsed = parse_propositional(furry)
    assert isinstance(parsed[0], Described) and len(parsed) == 2 and write(parsed) == furry
    assert proposition_of(parsed, {"REF.1": "INSTANCE.1.3.2.5"}) == Proposition(
        "instance", "INSTANCE.1.3.2.5", Predicate("part", "PART.4")
    )
    legs = parse_propositional("PART.4(REF.1)")
    assert len(legs) == 1 and proposition_of(legs) == proposition_of(parsed)


def test_the_tiny_corpus_holds_every_new_thing(runs) -> None:
    corpus = runs("tiny", 20)
    assert any(s.proposition.causal for d in corpus.documents for s in d.sentences)
    assert any("{" in s.propositional for d in corpus.documents for s in d.sentences)
    names = [s.name for s in corpus.test_sets]
    assert len(names) == 41 and sum(n.startswith("causal") for n in names) == 14
    assert TEMPLATE[-1] == CAUSAL
