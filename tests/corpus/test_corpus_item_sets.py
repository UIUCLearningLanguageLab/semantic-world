"""Stage 6 acceptance tests: the test sets.

Every true item is true and every false item is false, by the engine's truth tests and by the
independent oracle. A false item differs from its true item by exactly one change. The two items
of a pair never differ in the format of their model-facing fields. An event item names only its
scene. An instance-level or event-level item continues a document. The law-like items, the
possible false events, and the impossible false events have test sets of their own.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter

import numpy as np
import pytest

from semantic_world.corpus import Streams, parse_propositional, proposition_of, propositional
from semantic_world.corpus.config import CONCEPT_TYPES, ConfigError
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.histories import happened_in, scene_events, scene_of
from semantic_world.corpus.planner import Planner
from semantic_world.corpus.propositions import (
    CAN,
    CLASS,
    EVENT,
    FEATURE_KINDS,
    INSTANCE,
    NEC_ALL,
    NEC_NO,
    SIMPLE,
    VERB,
    Predicate,
    Proposition,
)
from semantic_world.corpus.realize import leaves, preorder
from semantic_world.corpus.testsets import (
    CHANGES,
    IMPOSSIBLE,
    INPUT_FIELDS,
    LAWLIKE,
    POSSIBLE,
    PREDICATE,
    QUANTIFIER,
    ROLE,
    SUBJECT,
    TestSetBuilder,
    build_test_sets,
    candidates,
    falsify,
    format_differences,
    happened,
    input_problems,
    law_like,
    notation,
    set_names,
    stated_propositions,
)

WORLDS = ("tiny", "default", "deep", "still")
ALL_SETS = (
    "class_predicate",
    "class_predicate_lawlike",
    "class_subject",
    "class_subject_lawlike",
    "class_quantifier",
    "class_quantifier_lawlike",
    "class_role",
    "instance_predicate",
    "instance_subject",
    "instance_role",
    "event_predicate_possible",
    "event_predicate_impossible",
    "event_subject_possible",
    "event_subject_impossible",
    "event_role_possible",
    "event_role_impossible",
)
GRAMMARS = {
    "default": {},
    "marked": {
        "grammar": {
            "word_order": {"clause": "SOV", "adjective": "after", "auxiliary": "after"},
            "morphology": {
                "number": {"enabled": True},
                "tense": {"enabled": True},
                "aspect": {"enabled": True},
            },
        },
    },
    "ambiguous": {
        "grammar": {"can_rate": {"class": 0.2, "instance": 0.5}},
        "renderings": {"propositional": {"referents": "instance"}},
        "lexicon": {"synonym_rate": 0.3},
    },
}
"""Languages for the format tests: the default, one with every inflection marked and another
word order, and one whose surface is ambiguous, with synonyms and with instance labels in the
propositional rendering."""
RICH = {
    "mention": {"relative_clauses": {"rate": 0.3, "max_depth": 2, "object_share": 0.4}},
    "propositions": {"restriction_rate": 0.3},
}


def pairs_of(corpus, *names: str):
    return [
        (test_set, true, false)
        for test_set in corpus.test_sets
        if not names or test_set.name in names
        for true, false in test_set.pairs
    ]


def mentions_of(form: dict) -> list[dict]:
    """The mentions of a test item about instances: its subject, and the patient of its verb."""
    patient = form["predicate"].get("patient")
    return [form["subject"]] if patient is None else [form["subject"], patient]


def scenes_json(corpus) -> dict[str, dict]:
    """The scenes as ``scenes.jsonl`` holds them, by label."""
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


def documents_json(corpus) -> dict[str, dict]:
    return {d.label: json.loads(json.dumps(d.to_json())) for d in corpus.documents}


# ---------------------------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------------------------


def test_the_layout_of_the_test_sets(runs) -> None:
    corpus = runs("default")
    assert [s.name for s in corpus.test_sets] == list(ALL_SETS)
    assert [name for name, *_ in set_names(CHANGES)] == list(ALL_SETS)
    size = corpus.config.test_sets.size
    for test_set in corpus.test_sets:
        level, change, *kind = test_set.name.split("_")
        assert (test_set.level, test_set.change, test_set.kind) == (level, change, "".join(kind))
        assert len(test_set.pairs) <= size
        assert len(test_set.items) == 2 * len(test_set.pairs)
        for number, (true, false) in enumerate(test_set.pairs, start=1):
            for item, truth in ((true, True), (false, False)):
                meta = item.meta
                assert (meta["set"], meta["pair"], meta["truth"]) == (test_set.name, number, truth)
                assert (meta["level"], meta["change"]) == (level, change)
                assert list(item.to_json()) == ["input", "meta"]
                assert list(item.input) == list(INPUT_FIELDS)
                assert item.proposition.level == level
    # the default world fills every set but some of the law-like ones and the possible role
    # swaps of events: its two-place event types hold for few pairs (since stage a5b), so few
    # swapped bindings are able (1 pair in 200 documents)
    sizes = {s.name: len(s.pairs) for s in corpus.test_sets}
    unfilled = {name for name in ALL_SETS if LAWLIKE in name} | {"event_role_possible"}
    assert all(sizes[name] == size for name in ALL_SETS if name not in unfilled), sizes
    assert all(sizes[name] > 0 for name in ALL_SETS), sizes
    # a true item is used once in the sets of one level and change
    for level, change in {(s.level, s.change) for s in corpus.test_sets}:
        used = [
            (true.input["document"], true.proposition)
            for test_set, true, _ in pairs_of(corpus)
            if (test_set.level, test_set.change) == (level, change)
        ]
        assert len(used) == len(set(used))


def test_the_changes_and_the_size_are_settings(cases, runs) -> None:
    case = cases("tiny")
    corpus = runs("tiny", test_sets={"size": 5, "changes": ["subject", "quantifier"]})
    assert [s.name for s in corpus.test_sets] == [
        "class_subject",
        "class_subject_lawlike",
        "class_quantifier",
        "class_quantifier_lawlike",
        "instance_subject",
        "event_subject_possible",
        "event_subject_impossible",
    ]
    assert all(len(s.pairs) <= 5 for s in corpus.test_sets)
    empty = runs("tiny", test_sets={"size": 0})
    assert [s.name for s in empty.test_sets] == list(ALL_SETS)
    assert all(not s.pairs and s.stats()["true_items_seen_share"] is None for s in empty.test_sets)
    with pytest.raises(ConfigError, match="test_sets.changes"):
        case.config(test_sets={"changes": ["tense"]})


def test_a_corpus_without_narratives_has_class_level_sets_only(runs) -> None:
    mix = {"encyclopedic_category": 1, "encyclopedic_feature": 1, "entity": 0, "situational": 0}
    corpus = runs("tiny", 20, documents={"mix": mix})
    sizes = {s.name: len(s.pairs) for s in corpus.test_sets}
    assert list(sizes) == list(ALL_SETS)
    assert all((count > 0) == name.startswith("class") for name, count in sizes.items()), sizes


# ---------------------------------------------------------------------------------------------
# Acceptance: truth, and exactly one change
# ---------------------------------------------------------------------------------------------


def differences(true: Proposition, false: Proposition) -> set[str]:
    a, b = true.to_json(), false.to_json()
    keys = ("level", "quantifier", "polarity", "subject", "predicate", "scene", "tense", "aspect")
    return {key for key in keys if a.get(key) != b.get(key)}


@pytest.mark.parametrize("name", WORLDS)
def test_true_items_are_true_and_false_items_differ_by_one_change(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **RICH)
    truth, oracle = corpus.planner.truth, case.oracle()
    scenes, documents = scenes_json(corpus), documents_json(corpus)
    made = Counter()
    for test_set, true, false in pairs_of(corpus):
        made[test_set.name] += 1
        for item in (true, false):
            form = item.input["logical_form"]
            evaluation = truth.evaluate(item.proposition)
            assert evaluation.valid and evaluation.true == item.truth
            assert item.meta["grounding"] == evaluation.grounding
            if test_set.level == EVENT:
                assert oracle.event(form, scenes[form["scene"]]) is item.truth
            else:
                assert oracle.truth(form) is item.truth, form
            assert corpus.planner.facts.expressible(item.proposition)
        changed = differences(true.proposition, false.proposition)
        a, b = true.proposition, false.proposition
        if test_set.change == PREDICATE:
            assert changed == {"predicate"}
            assert dataclasses.replace(b.predicate, label=a.predicate.label) == a.predicate
        elif test_set.change == SUBJECT:
            assert changed == {"subject"}
            if test_set.level == CLASS:
                facts = corpus.planner.facts
                assert b.subject.restriction == a.subject.restriction
                assert b.subject.clauses == a.subject.clauses
                assert facts.level[b.subject.category] == facts.level[a.subject.category]
            else:
                # the new subject is another referent of the item's document
                referents = documents[true.input["document"]]["referents"].values()
                assert b.subject in referents and b.subject != a.subject
        elif test_set.change == QUANTIFIER:
            assert changed == {"quantifier"} and test_set.level == CLASS
        else:
            assert changed == {"subject", "predicate"} and a.predicate.kind == VERB
            assert (b.subject, b.predicate.patient) == (a.predicate.patient, a.subject)
            assert b.predicate.label == a.predicate.label
    assert sum(made.values()) > 150, made
    if name == "default":
        assert all(made[s] > 0 for s in ALL_SETS)


# ---------------------------------------------------------------------------------------------
# Acceptance: the two items of a pair have the same format
# ---------------------------------------------------------------------------------------------

_FREE = ("formal", "conceptual", "propositional", "text", "tree")


def shape(value, key: str = ""):
    """The format of a model-facing value, worked out here and not by the package: the fields
    of every mapping, the type of every value, and the notation of every label. The indices of
    a label are taken out, except those of a scene or an event, which are counted."""
    if isinstance(value, dict):
        return {k: shape(v, k) for k, v in value.items() if k != "clauses"}
    if isinstance(value, list):
        if key in ("tokens", "restriction", "readings", "distinguished") or key in _FREE:
            return "list"
        return ["list", sorted({json.dumps(shape(v, key), sort_keys=True) for v in value})]
    if value is None or isinstance(value, bool):
        return type(value).__name__
    assert isinstance(value, str)
    if key in _FREE:
        return "text"
    if not re.search(r"\d", value):
        return "word"
    if value.startswith("SCENE."):
        return re.sub(r"\d+", "#", value)
    return re.sub(r"\.(HIGH|LOW)$", "", re.sub(r"\d+(\.\d+)*", "#", value))


@pytest.mark.parametrize("language", list(GRAMMARS))
@pytest.mark.parametrize("name", WORLDS)
def test_true_and_false_items_never_differ_in_format(runs, name, language) -> None:
    corpus = runs(name, **GRAMMARS[language], **RICH)
    mode = corpus.config.propositional_referents
    documents = documents_json(corpus)
    checked = Counter()
    for test_set in corpus.test_sets:
        for true, false in test_set.pairs:
            checked[test_set.name] += 1
            # the package's own check, and the same check worked out here
            assert format_differences(true.input, false.input) == []
            assert shape(true.input) == shape(false.input), test_set.name
            assert list(true.input) == list(false.input) == list(INPUT_FIELDS)
            for item in (true, false):
                record = item.input
                assert input_problems(record) == []
                form = record["logical_form"]
                assert "id" not in form and "grounding" not in form and "rule" not in form
                # no item names an event: an event label is never written
                assert not re.search(r"SCENE\.\d+\.EVENTINSTANCE\.\d+", json.dumps(record))
                assert record["words"] is None and record["text"] is None
                assert leaves(record["tree"]) == record["tokens"]
                # the renderings are made from the tokens and from the logical form alone
                assert record["formal"] == corpus.planner.realizer.formal(record["tokens"])
                assert record["conceptual"] == corpus.planner.realizer.conceptual(record["tokens"])
                assert record["propositional"] == propositional(form, mode)
                referents = {}
                if record["document"] is not None:
                    referents = documents[record["document"]]["referents"]
                parsed = proposition_of(parse_propositional(record["propositional"]), referents)
                assert parsed == item.proposition
                if test_set.level == EVENT:
                    assert form["event"] is None and record["events"] == [form["scene"]]
                    assert f"EVENT({form['scene']}, " in record["propositional"]
                    assert item.proposition.event is None
                else:
                    assert all(event is None for event in record["events"])
    assert sum(checked.values()) > 150
    # every set of the default world is checked
    if name == "default":
        assert all(checked[s] > 0 for s in ALL_SETS), checked


def test_the_format_check_finds_planted_differences(runs) -> None:
    corpus = runs("default")
    _, true, false = pairs_of(corpus, "event_role_possible")[0]
    a, b = json.loads(json.dumps(true.input)), json.loads(json.dumps(false.input))
    assert format_differences(a, b) == [] and input_problems(a) == []

    def planted(change) -> tuple[list[str], list[str]]:
        other = json.loads(json.dumps(a))
        change(other)
        return format_differences(other, b), input_problems(other)

    # an event label where the other item has a scene label
    label = a["logical_form"]["scene"] + ".EVENTINSTANCE.3"
    found, problems = planted(lambda x: x["logical_form"].update(event=label))
    assert any("logical_form.event" in f for f in found) and "names an event" in " ".join(problems)
    found, problems = planted(lambda x: x.update(events=[label]))
    assert "not a scene" in " ".join(problems)
    found, problems = planted(
        lambda x: x.update(
            propositional=x["propositional"].replace("EVENT(SCENE.", "EVENT(SCENE.1.EVENTINSTANCE.")
        )
    )
    assert "names an event" in " ".join(problems)
    # a field of the answer in the model-facing part
    found, problems = planted(lambda x: x["logical_form"].update(grounding={"step": 2}))
    assert found and "'grounding', which is metadata" in " ".join(problems)
    found, problems = planted(lambda x: x["logical_form"].update(id="PROP.7"))
    assert found and "'id', which is metadata" in " ".join(problems)
    # a missing field, a field that is null in one item only, and another notation
    found, _ = planted(lambda x: x["logical_form"]["subject"].pop("noun"))
    assert any("logical_form.subject" in f and "fields" in f for f in found)
    found, _ = planted(lambda x: x["logical_form"]["subject"].update(noun=None))
    assert any("not of one type" in f for f in found)
    found, _ = planted(lambda x: x.update(text="the dog ran"))
    assert any("input.text" in f for f in found)
    found, _ = planted(lambda x: x["logical_form"]["subject"].update(referent="INSTANCE.1.2.3"))
    assert any("not in one notation" in f for f in found)
    found, problems = planted(lambda x: x.pop("readings"))
    assert found and problems
    found, problems = planted(lambda x: x.update(document=None))
    assert found and "names a document" in " ".join(problems)
    # one more relative clause than the other item has
    clause = {"kind": "event_type1", "label": "EVENTTYPE1.1"}
    found, _ = planted(lambda x: x["logical_form"]["subject"].update(clauses=[clause]))
    assert any("relative clauses" in f for f in found)
    assert notation("CATEGORY.1.3") == notation("CATEGORY.1.3.2") == "CATEGORY.#"
    assert notation("most") is None
    assert notation("SCENE.8") != notation("SCENE.8.EVENTINSTANCE.5") and notation(
        "EVENTTYPE2.1"
    ) == notation("EVENTTYPE2.1.2")
    assert notation("SCALARDIM.1.HIGH") == notation("SCALARDIM.2.LOW")


def test_a_pair_that_differs_in_format_stops_the_run(runs, monkeypatch) -> None:
    corpus = runs("tiny")
    builder = TestSetBuilder(corpus.planner, corpus.documents)
    real = builder._input

    def leaky(name, number, document, proposition, bare):
        record = real(name, number, document, proposition, bare)
        if record is not None and proposition.grounding.get("step") is not None:
            record["logical_form"]["event"] = f"{proposition.scene}.EVENTINSTANCE.1"  # a true item
        return record

    monkeypatch.setattr(builder, "_input", leaky)
    with pytest.raises(CorpusError, match="differ in format, or give away their answer"):
        builder._group(EVENT, PREDICATE)


# ---------------------------------------------------------------------------------------------
# Context: an item continues a document
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_instance_and_event_items_continue_a_document(runs, name) -> None:
    corpus = runs(name, **RICH)
    documents = documents_json(corpus)
    lexicon = corpus.planner.lexicon
    checked = 0
    for test_set, true, false in pairs_of(corpus):
        if test_set.level == CLASS:
            assert true.input["document"] is None and false.input["document"] is None
            assert all(label is None for label in true.input["coreference"])
            continue
        assert true.input["document"] == false.input["document"]
        document = documents[true.input["document"]]
        assert document["type"] in ("entity", "situational")
        labels = {instance: label for label, instance in document["referents"].items()}
        for item in (true, false):
            checked += 1
            record = item.input
            form = record["logical_form"]
            mentions = mentions_of(form)
            for mention in mentions:
                # a referent that the document has already mentioned, under its own label
                assert labels[mention["instance"]] == mention["referent"]
                assert mention["noun"] is not None and "clauses" not in mention
            assert record["coreference"] == [m["referent"] for m in mentions]
            assert [r["referent"] for r in record["referents"]] == [m["instance"] for m in mentions]
            # every noun phrase is a definite mention
            phrases = [n for n in preorder(record["tree"]) if n[0] in ("NP-SBJ", "NP-OBJ")]
            assert len(phrases) == len(mentions)
            for phrase in phrases:
                words = {child[0]: child[1] for child in phrase[1:] if len(child) == 2}
                assert "Pro" not in words and lexicon.lexeme(words["Det"]).gloss == "the"
            assert len(record["distinguished"]) == len(mentions)
            assert all(mark in (True, False) for mark in record["distinguished"])
            if test_set.level == EVENT:
                assert form["scene"] in document["scenes"]
        # the two items mention an instance that they share in the same way
        if test_set.level == EVENT or test_set.change != PREDICATE:
            shared = {m["instance"]: m for m in mentions_of(true.input["logical_form"])}
            for mention in mentions_of(false.input["logical_form"]):
                assert shared.get(mention["instance"], mention) == mention
    assert checked > 100


def test_an_item_without_a_noun_for_its_referent_is_left_out(cases) -> None:
    # with no word for any category, a referent can only be "it", which picks out no one
    case = cases("tiny")
    proportions = dict.fromkeys(CONCEPT_TYPES, 1.0)
    proportions["category"] = 0.0
    config = case.config(
        lexicon={"named_proportion": proportions}, test_sets={"size": 5}, documents={"count": 30}
    )
    planner = Planner(config, case.result)
    sets = build_test_sets(planner, planner.generate())
    assert all(not s.pairs for s in sets if s.level != CLASS)


# ---------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_event_items(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **RICH)
    oracle = case.oracle()
    scenes, documents = scenes_json(corpus), documents_json(corpus)
    kinds = Counter()
    category_names = 0
    for test_set, true, false in pairs_of(corpus):
        if test_set.level != EVENT:
            continue
        document = documents[true.input["document"]]
        document_scenes = [scenes[label] for label in document["scenes"]]
        reported = {
            label
            for sentence in document["sentences"]
            for label in sentence["events"]
            if label is not None
        }
        # the true item is an event that the document reports, in a main clause or a relative
        # clause, with the same aspect
        form = true.input["logical_form"]
        matching = oracle.matching(form, scenes[form["scene"]])
        assert any(event["label"] in reported for event in matching)
        assert true.meta["possible"] is True and "step" in true.meta["grounding"]
        assert true.meta["grounding"]["able"] and true.meta["grounding"]["legal"]
        # the false item did not happen in any scene of the document, under any verb that its
        # label names (the aspect is the report's choice, and does not count)
        other = false.input["logical_form"]
        assert other["scene"] == form["scene"]
        assert (other["tense"], other["aspect"]) == (form["tense"], form["aspect"])
        assert not oracle.happened(other, document_scenes)
        predicate = other["predicate"]
        label = predicate["label"]
        patient = predicate["patient"]["instance"] if "patient" in predicate else None
        able = oracle.able(label, other["subject"]["instance"], patient)
        assert false.meta["possible"] is able is false.meta["grounding"]["able"]
        assert test_set.kind == (POSSIBLE if able else IMPOSSIBLE)
        assert "step" not in false.meta["grounding"]
        assert false.meta["grounding"]["legal"] is oracle.legal(
            scenes[other["scene"]], label, other["subject"]["instance"], patient
        )
        kinds[test_set.kind] += 1
        event_types = corpus.planner.world.event_types
        category_names += label in event_types and event_types[label].category
        # the agent and the patient take part in the scene
        participants = scenes[other["scene"]]["participants"]
        assert other["subject"]["instance"] in participants
        assert patient is None or patient in participants
    assert kinds[POSSIBLE] > 10 and kinds[IMPOSSIBLE] > 10, kinds
    if name == "default":
        assert category_names > 0  # some false items name their verb with a verb category


def test_a_false_event_did_not_happen_in_any_scene_of_its_document(cases) -> None:
    case = cases("default")
    facts = case.facts(scene={"events_per_step": 4.0})
    generator = case.scenes(scene={"events_per_step": 4.0})
    truth = facts.truth
    streams = Streams(3)
    seed = case.world.instances[0]
    first, second = (generator.scene(streams, number, seed) for number in (900, 901))
    truth.add_scene(first)
    truth.add_scene(second)
    # an event of the second scene that did not happen in the first, with both participants in
    # the first: the truth test calls it false of the first scene, and the test sets do not
    # take it, because the words would be true of the document
    moved = None
    for event in scene_events(second):
        report = facts.event_fact(event)
        if report is None:
            continue
        claim = dataclasses.replace(report, scene=first.label, event=None, grounding=None)
        evaluation = truth.evaluate(claim)
        if evaluation.valid and not evaluation.true and not happened_in(first, *event.key):
            moved = claim
            break
    assert moved is not None
    assert not happened(truth, [first], moved) and happened(truth, [first, second], moved)
    # an event carries no aspect: a report of it is true in either aspect, and the test sets
    # take neither as a false item
    event = next(e for e in scene_events(first) if facts.event_fact(e) is not None)
    for aspect in ("simple", "progressive"):
        twin = dataclasses.replace(facts.event_fact(event, aspect=aspect), event=None)
        assert truth.is_true(twin) and happened(truth, [first], twin)
    # an event-type category names every event type below it
    transitive = next(e for e in scene_events(first) if e.transitive)
    names = truth.verb_names(transitive.type)
    assert len(names) == 2
    for label in names:
        claim = Proposition(
            EVENT,
            transitive.agent,
            Predicate(VERB, label, transitive.patient),
            scene=first.label,
            tense="past",
            aspect="simple",
        )
        assert happened(truth, [first], claim)


def test_candidates_for_events(cases) -> None:
    case = cases("default")
    facts = case.facts()
    generator = case.scenes()
    instances = case.world.instances
    for n in range(1, 60):
        scene = generator.scene(Streams(1), n, instances[n])
        facts.truth.add_scene(scene)
        if len(scene.participants) > 3 and any(
            e.transitive and facts.event_fact(e) for e in scene_events(scene)
        ):
            break
    report = next(
        p for e in scene_events(scene) if e.transitive and (p := facts.event_fact(e)) is not None
    )
    by_predicate = candidates(facts, report, PREDICATE)
    assert {c.predicate.label for c in by_predicate} == set(facts.verbs) - {report.predicate.label}
    assert all(c.predicate.patient == report.predicate.patient for c in by_predicate)
    by_subject = candidates(facts, report, SUBJECT)
    assert {c.subject for c in by_subject} == set(scene.participants) - {
        report.subject,
        report.predicate.patient,
    }
    limited = candidates(facts, report, SUBJECT, instances=scene.participants[:2])
    assert {c.subject for c in limited} <= set(scene.participants[:2])
    (swapped,) = candidates(facts, report, ROLE)
    assert (swapped.subject, swapped.predicate.patient) == (
        report.predicate.patient,
        report.subject,
    )
    assert candidates(facts, report, QUANTIFIER) == []
    intransitive = next(
        p
        for e in scene_events(scene)
        if not e.transitive and (p := facts.event_fact(e)) is not None
    )
    assert candidates(facts, intransitive, ROLE) == []
    labels = {c.predicate.label for c in candidates(facts, intransitive, PREDICATE)}
    assert labels == set(facts.features[CAN]) - {intransitive.predicate.label}
    # a false event names no event, and an accepted candidate passes the extra condition
    rng = np.random.default_rng(0)
    false = falsify(facts, report, SUBJECT, rng)
    if false is not None:
        assert false.event is None and false.scene == report.scene
        assert not facts.truth.is_true(false)
    assert falsify(facts, report, PREDICATE, rng, accept=lambda candidate: False) is None


# ---------------------------------------------------------------------------------------------
# Law-like items
# ---------------------------------------------------------------------------------------------


def claimed_value(form: dict) -> int | None:
    """The value that a form claims for every member of its subject set, when its quantifier is
    ``nec_all`` or ``nec_no``; None for any other form."""
    quantifier = form["quantifier"]
    if quantifier not in (NEC_ALL, NEC_NO):
        return None
    return 0 if (quantifier == NEC_NO) != (not form["polarity"]) else 1


@pytest.mark.parametrize("name", WORLDS)
def test_law_like_items_have_sets_of_their_own(cases, runs, name) -> None:
    """A law-like item is a false nec item whose extensional twin is true: every instance of the
    subject set has the claimed value, and nothing fixes it."""
    case = cases(name)
    corpus = runs(name, **RICH)
    oracle = case.oracle()
    law_like_pairs = decidable = 0
    for test_set, true, false in pairs_of(corpus):
        if test_set.level != CLASS:
            assert "law_like" not in false.meta
            continue
        assert true.meta["law_like"] is False
        form = false.input["logical_form"]
        kind = form["predicate"]["kind"]
        claimed = claimed_value(form)
        uncontradicted = False
        if claimed is not None and kind in FEATURE_KINDS:
            count, total = oracle._counts(form)
            uncontradicted = count == (total if claimed else 0)
        assert false.meta["law_like"] is uncontradicted
        assert (test_set.kind == LAWLIKE) == uncontradicted
        assert law_like(corpus.planner.truth, false.proposition) is uncontradicted
        if uncontradicted:
            law_like_pairs += 1
            # false by the fixed test alone: nothing fixes the value that every instance has
            assert oracle.fixed(form["subject"], form["predicate"]["label"]) != claimed
            assert false.meta["grounding"]["test"] in ("exact", "local")
            # the extensional twin is true, by the engine and by the oracle
            twin = {**form, "quantifier": "all" if form["quantifier"] == NEC_ALL else "no"}
            assert oracle.truth(twin) is True
        elif claimed is not None and kind in FEATURE_KINDS:
            decidable += 1  # some instance contradicts the item
    assert decidable > 0
    if name in ("default", "deep"):
        assert law_like_pairs > 10


def test_law_like_items_are_nec_items(runs) -> None:
    # relation facts and patient capacities are never nec, so they are never law-like, and with
    # the extensional words, "all" and "no" make no nec item apart from membership and rules
    corpus = runs("default", **RICH)
    for test_set, _, false in pairs_of(corpus):
        if test_set.kind == LAWLIKE:
            assert false.proposition.quantifier in (NEC_ALL, NEC_NO)
            assert false.proposition.predicate.kind in FEATURE_KINDS
    extensional = runs("default", quantifiers={"universal_words": "extensional"})
    for test_set, _, false in pairs_of(extensional):
        if test_set.level == CLASS and false.proposition.quantifier in (NEC_ALL, NEC_NO):
            assert (
                false.proposition.rule is not None or false.proposition.predicate.kind == "member"
            )
    sizes = {s.name: len(s.pairs) for s in extensional.test_sets}
    assert sizes["class_quantifier"] > 0 and sizes["class_predicate"] > 0


# ---------------------------------------------------------------------------------------------
# Seen and unseen items
# ---------------------------------------------------------------------------------------------


def bare(form: dict) -> dict:
    """A logical form of ``documents.jsonl`` without its label, its grounding, its event label,
    and the aspect of its report: what a test item's logical form holds, up to the aspect."""
    kept = {k: v for k, v in form.items() if k not in ("id", "grounding", "rule")}
    if kept["level"] == EVENT:
        kept["event"] = None
        kept["aspect"] = SIMPLE
    return kept


def stated_form(proposition: Proposition) -> Proposition:
    """A proposition as :func:`stated_propositions` holds it."""
    if proposition.level == EVENT:
        return dataclasses.replace(proposition, event=None, aspect=SIMPLE)
    return proposition


def plain(form: dict) -> str:
    """What the main clause of a form says, without the way its referents are mentioned."""
    kept = json.loads(json.dumps(bare(form)))
    if kept["level"] != CLASS:
        kept["subject"] = kept["subject"]["instance"]
        if "patient" in kept["predicate"]:
            kept["predicate"]["patient"] = kept["predicate"]["patient"]["instance"]
    return json.dumps(kept, sort_keys=True)


@pytest.mark.parametrize("name", WORLDS)
def test_seen_says_whether_a_training_document_states_the_item(runs, name) -> None:
    corpus = runs(name, **RICH)
    main_clauses = {
        plain(sentence["logical_form"])
        for document in documents_json(corpus).values()
        for sentence in document["sentences"]
    }
    stated = stated_propositions(corpus.documents)
    seen = Counter()
    for test_set, true, false in pairs_of(corpus):
        # a false item is never seen: the documents state true propositions only
        assert false.meta["seen"] is False
        assert stated_form(false.proposition) not in stated
        assert plain(false.input["logical_form"]) not in main_clauses
        in_main_clause = plain(true.input["logical_form"]) in main_clauses
        if test_set.level == CLASS:
            assert true.meta["seen"] is in_main_clause
        elif in_main_clause:
            assert true.meta["seen"] is True
        seen[(test_set.level, true.meta["seen"])] += 1
    for test_set in corpus.test_sets:
        stats = test_set.stats()
        count = sum(true.meta["seen"] for true, _ in test_set.pairs)
        assert stats == corpus.stats["test_sets"][test_set.name]
        assert stats["pairs"] == len(test_set.pairs) and stats["true_items_seen"] == count
        if test_set.pairs:
            assert stats["true_items_seen_share"] == round(count / len(test_set.pairs), 6)
    # both kinds occur at every level. An event item is unseen when it names its verb at
    # another level of the verb tree than its document does
    for level in (CLASS, INSTANCE):
        assert seen[(level, True)] > 0 and seen[(level, False)] > 0, seen
    assert seen[(EVENT, True)] > 0
    if name in ("tiny", "default"):
        assert seen[(EVENT, False)] > 0, seen


def test_what_the_documents_state(runs) -> None:
    corpus = runs("default", **RICH)
    stated = stated_propositions(corpus.documents)
    world = corpus.planner.world
    assert all(p.event is None for p in stated)
    assert all(p.aspect == SIMPLE for p in stated if p.level == EVENT)
    relative = modifiers = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert stated_form(proposition) in stated
            if proposition.level == CLASS:
                continue
            form = sentence.logical_form
            mentions = [form["subject"]]
            if "patient" in form["predicate"]:
                mentions.append(form["predicate"]["patient"])
            for mention in mentions:
                instance = mention["instance"]
                if mention["noun"] is not None:
                    # the noun says what the referent is
                    member = Proposition(INSTANCE, instance, Predicate("member", mention["noun"]))
                    assert member in stated
                for literal in mention["restriction"]:
                    modifiers += 1
                    positive = not literal.startswith("not ")
                    feature = literal.removeprefix("not ")
                    if feature.startswith("SCALARDIM."):
                        predicate = Predicate("scalar", feature, comparison=mention["noun"])
                    else:
                        predicate = Predicate(world.feature_kind[feature], feature)
                    assert Proposition(INSTANCE, instance, predicate, positive) in stated
                for clause in mention.get("clauses", ()):
                    relative += 1
                    other = clause.get("patient") or clause.get("agent")
                    label = clause["label"]
                    agent, patient = instance, None if other is None else other["instance"]
                    if "agent" in clause:
                        agent, patient = patient, instance
                    predicate = Predicate(clause["kind"], label, patient)
                    if "event" in clause:
                        expected = Proposition(
                            EVENT,
                            agent,
                            predicate,
                            scene=scene_of(clause["event"]),
                            tense=clause["tense"],
                            aspect=SIMPLE,
                        )
                    else:
                        expected = Proposition(
                            INSTANCE, agent, predicate, clause.get("polarity", True)
                        )
                    assert expected in stated
    assert relative > 20 and modifiers > 100
    # every stated proposition is true
    truth = corpus.planner.truth
    assert all(truth.is_true(p) for p in stated)


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def test_the_test_sets_are_reproducible(cases) -> None:
    case = cases("default")

    def made(seed: int = 1, **sections) -> list:
        settings = {"documents": {"count": 60}, "test_sets": {"size": 8}, **sections}
        planner = Planner(case.config(**settings, seed=seed), case.result)
        sets = build_test_sets(planner, planner.generate())
        return [item.to_json() for test_set in sets for item in test_set.items]

    base = made()
    assert made() == base
    assert made(seed=2) != base


def test_grammar_settings_change_no_test_item(runs) -> None:
    base = runs("default", **RICH)
    other = runs("default", **GRAMMARS["marked"], **RICH)
    assert [s.name for s in other.test_sets] == [s.name for s in base.test_sets]
    different = 0
    for (_, a_true, a_false), (_, b_true, b_false) in zip(
        pairs_of(base), pairs_of(other), strict=True
    ):
        for a, b in ((a_true, b_true), (a_false, b_false)):
            assert a.proposition == b.proposition and a.meta == b.meta
            assert a.input["document"] == b.input["document"]
            assert a.input["logical_form"] == b.input["logical_form"]
            assert a.input["propositional"] == b.input["propositional"]
            different += a.input["tokens"] != b.input["tokens"]
    assert different > 100  # the words do change
