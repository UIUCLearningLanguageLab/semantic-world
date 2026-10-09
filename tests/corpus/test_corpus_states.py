"""Stage a7a acceptance tests: states, changes, blocked sentences, and their test sets.

Every state and change sentence is true of its scene's history, by the engine and by the
independent oracle. Every result sentence's change happened in its event's step and was made by
that event's own effect. Every ``ABLE_NOW`` sentence is true, and a negative one is blocked (the
requirement holds, the binding is not legal then). Every ``changed`` mark agrees with the
history, every false event is of the kind its set says, and every false ``able_now`` item is
blocked or impossible as its set says, all by the oracle's replay. The new sentences take the
new readings, parse back, and read back through ``interpret``. The rates, the negation rates,
and ``lexicon.can_words`` are settings.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from semantic_world.corpus import Streams, build_lexicon, interpret, parse_propositional
from semantic_world.corpus.config import ConfigError
from semantic_world.corpus.grammar import (
    INSTANCE_NP,
    NounPhrase,
    Predication,
    SentencePlan,
    check_plan,
)
from semantic_world.corpus.histories import scene_events, split_time_key, time_index
from semantic_world.corpus.lexicon import ABLE_NOW_WORD, FUNCTION_WORDS
from semantic_world.corpus.planner import (
    BLOCKED,
    EVENT_SECTION,
    INITIAL_STATE,
    NARRATIVE_SECTIONS,
    RESULT,
)
from semantic_world.corpus.propositions import (
    ABLE_NOW,
    CAN,
    CHANGE,
    CLASS,
    EVENT,
    INSTANCE,
    STATE,
    STATE_KIND,
    VERB,
    Predicate,
    Proposition,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, leaves
from semantic_world.corpus.renderings import proposition_of, propositional
from semantic_world.corpus.testsets import BLOCKED as BLOCKED_KIND
from semantic_world.corpus.testsets import CHANGED, IMPOSSIBLE, POSSIBLE, UNCHANGED

WORLDS = ("tiny", "default", "deep", "still")
MANY = {
    "documents": {"initial_state_rate": 0.6, "result_rate": 0.9, "blocked_rate": 0.5},
    "propositions": {"negation_rate": {"able_now": 0.6}},
}
"""Many state, result, and blocked sentences, so that the tests see every kind."""
TIMED = (STATE, CHANGE, ABLE_NOW)


def scenes_json(corpus) -> dict[str, dict]:
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


def timed_sentences(corpus, *sections: str):
    return [
        (document, sentence)
        for document in corpus.documents
        for sentence in document.sentences
        if sentence.section in (sections or (INITIAL_STATE, RESULT, BLOCKED))
    ]


def preceding_event(document, index: int):
    """The event sentence that a follow-up sentence belongs to: the nearest one before it."""
    for before in reversed(document.sentences[:index]):
        if before.section == EVENT_SECTION:
            return before
    raise AssertionError("a follow-up sentence comes after an event sentence")


# ---------------------------------------------------------------------------------------------
# Acceptance: every state, change, and blocked sentence is true of its history
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_state_and_change_sentences_are_true_of_their_history(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    scenes = scenes_json(corpus)
    counts: Counter = Counter()
    for _, sentence in timed_sentences(corpus):
        proposition = sentence.proposition
        assert proposition.level in TIMED and proposition.timed
        assert sentence.plan.proposition() == proposition
        assert corpus.planner.truth.is_true(proposition), sentence.label
        form = json.loads(json.dumps(sentence.logical_form))
        assert oracle.truth(form, scenes[proposition.scene]) is True, form
        # the JSON form: the level, the scene, the time point, the tense, and no aspect
        assert form["level"] == proposition.level and form["time"] == proposition.time
        assert form["tense"] == "past" and "aspect" not in form and "event" not in form
        if proposition.level in (STATE, CHANGE):
            assert form["predicate"] == {"kind": STATE_KIND, "label": proposition.predicate.label}
            assert proposition.predicate.label.startswith("BOOLFL.")
        else:
            assert form["predicate"]["kind"] in (CAN, VERB)
        counts[proposition.level] += 1
    assert counts[STATE] > 10 and counts[CHANGE] > 10 and counts[ABLE_NOW] > 5, counts


@pytest.mark.parametrize("name", WORLDS)
def test_result_sentences_state_a_change_the_events_own_effect_made(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    scenes = scenes_json(corpus)
    results = derived = 0
    for document in corpus.documents:
        events = {e.label: e for scene in document.scenes for e in scene_events(scene)}
        for index, sentence in enumerate(document.sentences):
            if sentence.section != RESULT:
                continue
            proposition = sentence.proposition
            assert proposition.level == CHANGE
            event = events[preceding_event(document, index).proposition.event]
            # the change is at the event's step, about a participant of the event
            assert proposition.scene == event.scene and proposition.time == f"TIME.{event.step}"
            assert event.involves(proposition.subject)
            # and the event's own effect made it: by the history alone
            scene = scenes[event.scene]
            fluent = proposition.predicate.label
            assert oracle.made_by(scene, event.step, proposition.subject, fluent, event.label)
            assert oracle.change(sentence.logical_form, scene)
            # the grounding names an event of the step whose own effect made it
            caused_by = proposition.grounding["caused_by"]
            assert caused_by is not None and caused_by.startswith(event.scene + ".EVENTINSTANCE.")
            assert oracle.made_by(scene, event.step, proposition.subject, fluent, caused_by)
            results += 1
            derived += fluent in corpus.planner.world.derived_fluents
    assert results > 20
    if name == "default":
        assert derived > 0  # derived fluents change too, and are stated when the event made it


@pytest.mark.parametrize("name", WORLDS)
def test_initial_state_sentences_follow_first_mentions(cases, runs, name) -> None:
    corpus = runs(name, **MANY)
    oracle = cases(name).oracle(**MANY)
    scenes = scenes_json(corpus)
    found = 0
    for document in corpus.documents:
        events = {e.label: e for scene in document.scenes for e in scene_events(scene)}
        mentioned: set[str] = set()
        for index, sentence in enumerate(document.sentences):
            if sentence.section == INITIAL_STATE:
                proposition = sentence.proposition
                assert proposition.level == STATE
                before = preceding_event(document, index)
                # the subject was first mentioned in the event sentence this one follows
                mentions = {phrase.referent for phrase in before.plan.noun_phrases()}
                assert proposition.subject in mentions
                earlier = {
                    phrase.referent
                    for s in document.sentences[: document.sentences.index(before)]
                    for phrase in s.plan.noun_phrases()
                }
                assert proposition.subject not in earlier
                # at the time point before the earliest event the sentence reports about it
                reported = [before.plan.predication.event] + [
                    p.event
                    for phrase in before.plan.noun_phrases()
                    if phrase.clause is not None
                    for p in phrase.clause.predications
                ]
                steps = [
                    events[label].step
                    for label in reported
                    if label is not None and events[label].involves(proposition.subject)
                ]
                assert proposition.time == f"TIME.{min(steps)}"
                assert oracle.state(sentence.logical_form, scenes[proposition.scene])
                found += 1
            mentioned |= {phrase.referent for phrase in sentence.plan.noun_phrases()}
    assert found > 10


@pytest.mark.parametrize("name", WORLDS)
def test_blocked_sentences_are_true_and_a_negative_one_is_blocked(cases, runs, name) -> None:
    corpus = runs(name, **MANY)
    oracle = cases(name).oracle(**MANY)
    scenes = scenes_json(corpus)
    polarities: Counter = Counter()
    for document in corpus.documents:
        events = {e.label: e for scene in document.scenes for e in scene_events(scene)}
        for index, sentence in enumerate(document.sentences):
            if sentence.section != BLOCKED:
                continue
            proposition = sentence.proposition
            assert proposition.level == ABLE_NOW
            event = events[preceding_event(document, index).proposition.event]
            assert proposition.scene == event.scene and proposition.time == f"TIME.{event.step}"
            assert event.involves(proposition.subject)  # a participant of the event
            scene = scenes[event.scene]
            form = sentence.logical_form
            predicate = proposition.predicate
            patient = predicate.patient if isinstance(predicate.patient, str) else None
            assert oracle.able(predicate.label, proposition.subject, patient)
            legal_now = oracle.legal_at(
                scene, event.step, predicate.label, proposition.subject, patient
            )
            assert legal_now == proposition.polarity
            assert oracle.able_now(form, scene)
            assert form["grounding"]["able"] is True and form["grounding"]["legal"] == legal_now
            if proposition.polarity:
                # could do, and did not do in that step
                assert not any(
                    e.scene == event.scene
                    and e.step == event.step
                    and e.agent == proposition.subject
                    and e.patient == patient
                    and predicate.label in corpus.planner.truth.verb_names(e.type)
                    for e in events.values()
                )
            polarities[proposition.polarity] += 1
    assert polarities[False] > 5 and polarities[True] > 2, polarities


# ---------------------------------------------------------------------------------------------
# Acceptance: the test sets
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_changed_marks_agree_with_the_history(cases, runs, name) -> None:
    corpus = runs(name)
    oracle = cases(name).oracle()
    scenes = scenes_json(corpus)
    kinds: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level != STATE:
            continue
        assert test_set.kind in (CHANGED, UNCHANGED)
        for true, false in test_set.pairs:
            marks = []
            for item in (true, false):
                form = item.input["logical_form"]
                assert form["level"] == STATE and item.proposition.level == STATE
                scene = scenes[form["scene"]]
                # at the scene's final time point, of a participant of a situational narrative
                assert form["time"] == scene["final"]
                document = next(d for d in corpus.documents if d.label == item.input["document"])
                assert document.type == "situational"
                assert item.proposition.subject in document.referents.values()
                changed = oracle.changed(
                    scene, form["subject"]["instance"], form["predicate"]["label"]
                )
                assert item.meta["changed"] == changed
                assert oracle.state(form, scene) == item.meta["truth"]
                marks.append(changed)
            assert (test_set.kind == CHANGED) == any(marks)
            kinds[test_set.kind] += 1
        stats = test_set.stats()
        assert stats["items_changed"] == sum(item.meta["changed"] for item in test_set.items)
    assert kinds[CHANGED] > 5 and kinds[UNCHANGED] > 5, kinds


@pytest.mark.parametrize("name", WORLDS)
def test_every_false_event_is_of_the_kind_its_set_says(cases, runs, name) -> None:
    corpus = runs(name)
    oracle = cases(name).oracle()
    scenes = scenes_json(corpus)
    kinds: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level != EVENT:
            continue
        assert test_set.kind in (POSSIBLE, BLOCKED_KIND, IMPOSSIBLE)
        for _, false in test_set.pairs:
            form = false.input["logical_form"]
            predicate = form["predicate"]
            agent = form["subject"]["instance"]
            patient = predicate["patient"]["instance"] if "patient" in predicate else None
            scene = scenes[form["scene"]]
            able = oracle.able(predicate["label"], agent, patient)
            legal_ever = oracle.legal(scene, predicate["label"], agent, patient)
            document = next(d for d in corpus.documents if d.label == false.input["document"])
            assert not oracle.happened(form, [scenes[s.label] for s in document.scenes])
            if test_set.kind == POSSIBLE:
                assert able and legal_ever
            elif test_set.kind == BLOCKED_KIND:
                assert able and not legal_ever
            else:
                assert not able and not legal_ever
            assert false.meta["possible"] == able
            assert (false.meta["grounding"]["able"], false.meta["grounding"]["legal"]) == (
                able,
                legal_ever,
            )
            kinds[test_set.kind] += 1
    assert kinds[POSSIBLE] > 5 and kinds[IMPOSSIBLE] > 5, kinds
    if name == "default":
        assert kinds[BLOCKED_KIND] > 5, kinds


@pytest.mark.parametrize("name", WORLDS)
def test_able_now_items(cases, runs, name) -> None:
    corpus = runs(name)
    oracle = cases(name).oracle()
    scenes = scenes_json(corpus)
    kinds: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level != ABLE_NOW:
            continue
        assert test_set.kind in (BLOCKED_KIND, IMPOSSIBLE)
        for true, false in test_set.pairs:
            for item in (true, false):
                form = item.input["logical_form"]
                scene = scenes[form["scene"]]
                assert form["level"] == ABLE_NOW and form["time"] == scene["final"]
                assert form["polarity"] is True  # an item says that something was possible
                assert oracle.able_now(form, scene) == item.meta["truth"]
                document = next(d for d in corpus.documents if d.label == item.input["document"])
                assert document.type == "situational"
            form = false.input["logical_form"]
            predicate = form["predicate"]
            patient = predicate["patient"]["instance"] if "patient" in predicate else None
            able = oracle.able(predicate["label"], form["subject"]["instance"], patient)
            assert (test_set.kind == BLOCKED_KIND) == able
            assert false.meta["possible"] == able
            kinds[test_set.kind] += 1
    assert kinds[BLOCKED_KIND] > 5 and kinds[IMPOSSIBLE] > 5, kinds


def test_the_timed_sets_continue_situational_narratives_only(runs) -> None:
    mix = {
        "encyclopedic_category": 0.2,
        "encyclopedic_feature": 0.2,
        "entity": 0.6,
        "situational": 0,
    }
    corpus = runs("tiny", 40, documents={"mix": mix})
    for test_set in corpus.test_sets:
        if test_set.level in (STATE, ABLE_NOW):
            assert not test_set.pairs
        elif test_set.level == EVENT:
            assert test_set.pairs or test_set.kind == BLOCKED_KIND


# ---------------------------------------------------------------------------------------------
# The sentences in the documents
# ---------------------------------------------------------------------------------------------


def test_follow_up_sentences_come_after_event_sentences_in_order(runs) -> None:
    corpus = runs("default", 200, **MANY)
    order = {section: k for k, section in enumerate(NARRATIVE_SECTIONS)}
    for document in corpus.documents:
        if not document.scenes:
            continue
        group: list[str] = []
        for sentence in document.sentences:
            assert sentence.section in NARRATIVE_SECTIONS
            if sentence.section == EVENT_SECTION:
                group = [EVENT_SECTION]
                continue
            assert group, "a follow-up sentence comes after an event sentence"
            assert order[sentence.section] >= order[group[-1]], (document.label, sentence.label)
            # one result, one blocked sentence, and one description per event sentence
            if sentence.section != INITIAL_STATE:
                assert sentence.section not in group
            group.append(sentence.section)


def test_modifiers_stay_static(runs) -> None:
    corpus = runs("default", 200, **MANY)
    for document in corpus.documents:
        for sentence in document.sentences:
            for phrase in sentence.plan.noun_phrases():
                assert not any(x.feature.startswith("BOOLFL.") for x in phrase.restriction)
                if phrase.clause is not None:
                    for predication in phrase.clause.predications:
                        assert predication.kind != STATE_KIND and predication.time is None


def test_the_rates_are_settings(runs) -> None:
    def sections(**documents) -> Counter:
        corpus = runs("default", 200, documents=documents)
        return Counter(s.section for d in corpus.documents for s in d.sentences)

    none = sections(initial_state_rate=0, result_rate=0, blocked_rate=0)
    assert none[INITIAL_STATE] == none[RESULT] == none[BLOCKED] == 0 and none[EVENT_SECTION] > 100
    low = sections(initial_state_rate=0.2, result_rate=0.3, blocked_rate=0.1)
    high = sections(initial_state_rate=0.8, result_rate=0.9, blocked_rate=0.6)
    for section in (INITIAL_STATE, RESULT, BLOCKED):
        assert 0 < low[section] < high[section]
    # the share of event sentences with a result sentence tracks the rate: not every event
    # changes a fluent that has a word
    assert 0.15 < low[RESULT] / low[EVENT_SECTION] < 0.35
    assert 0.4 < high[RESULT] / high[EVENT_SECTION] < 0.95


def test_the_negation_rates_are_settings(runs) -> None:
    def polarities(section: str, **negation) -> Counter:
        corpus = runs(
            "default",
            200,
            documents={"initial_state_rate": 0.8, "blocked_rate": 0.6},
            propositions={"negation_rate": negation},
        )
        return Counter(
            s.proposition.polarity
            for d in corpus.documents
            for s in d.sentences
            if s.section == section
        )

    states = polarities(INITIAL_STATE, state=0.0)
    assert states[False] == 0 and states[True] > 50
    states = polarities(INITIAL_STATE, state=0.5)
    assert 0.3 < states[False] / sum(states.values()) < 0.7
    blocked = polarities(BLOCKED, able_now=1.0)
    assert blocked[True] == 0 and blocked[False] > 20
    blocked = polarities(BLOCKED, able_now=0.0)
    assert blocked[False] == 0 and blocked[True] > 10


def test_the_statistics_count_the_new_sentences(runs) -> None:
    corpus = runs("default", 200, **MANY)
    stats = corpus.stats
    sections = Counter(s.section for d in corpus.documents for s in d.sentences)
    states = stats["states"]
    assert states["initial_state_sentences"] == sections[INITIAL_STATE]
    assert states["result_sentences"] == sections[RESULT]
    assert states["blocked_sentences"] == sections[BLOCKED]
    assert states["event_sentences"] == sections[EVENT_SECTION]
    assert states["events_with_result_share"] == round(
        sections[RESULT] / sections[EVENT_SECTION], 6
    )
    negative = sum(
        not s.proposition.polarity
        for d in corpus.documents
        for s in d.sentences
        if s.section == BLOCKED
    )
    assert states["blocked_negative_share"] == round(negative / sections[BLOCKED], 6)
    levels = stats["sentences"]["by_level"]
    assert set(levels) == {CLASS, INSTANCE, EVENT, STATE, CHANGE, ABLE_NOW}
    assert levels[STATE] == sections[INITIAL_STATE] and levels[CHANGE] == sections[RESULT]
    assert levels[ABLE_NOW] == sections[BLOCKED]
    assert set(stats["sentences"]["negative_share"]) == {CLASS, INSTANCE, STATE, ABLE_NOW}
    for test_set in corpus.test_sets:
        record = stats["test_sets"][test_set.name]
        assert ("items_changed_share" in record) == (test_set.level == STATE)


# ---------------------------------------------------------------------------------------------
# Readings, renderings, and reading back
# ---------------------------------------------------------------------------------------------


def test_readings_of_the_new_sentences(runs) -> None:
    corpus = runs("default", 200, **MANY)
    seen: Counter = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            level = sentence.proposition.level
            if level in (STATE, CHANGE):
                assert sentence.readings == ("state",), sentence.readings
            elif level == ABLE_NOW:
                # with can shared, "can" allows a capacity and what was possible then
                assert sentence.readings == ("capacity", "able_now"), sentence.readings
            elif level == INSTANCE and sentence.proposition.predicate.kind in (CAN, VERB):
                assert sentence.readings == ("capacity", "able_now"), sentence.readings
            elif level == INSTANCE:
                assert sentence.readings == ("capacity",)
            seen[level] += 1
    assert seen[STATE] and seen[CHANGE] and seen[ABLE_NOW]
    counts = corpus.stats["ambiguity"]
    assert counts["readings"]["capacity+able_now"] == seen[ABLE_NOW] + sum(
        1
        for d in corpus.documents
        for s in d.sentences
        if s.proposition.level == INSTANCE and s.proposition.predicate.kind in (CAN, VERB)
    )
    assert counts["readings"]["state"] == seen[STATE] + seen[CHANGE]


def test_can_words_distinct(cases, runs) -> None:
    corpus = runs("default", 200, **MANY, lexicon={"can_words": "distinct"})
    lexicon = corpus.planner.lexicon
    can_now = lexicon.function_word(ABLE_NOW_WORD)
    assert can_now.concept == "CAN_NOW" and can_now.gloss == ABLE_NOW_WORD
    assert lexicon.function_word("become").concept == "BECOME"
    assert "become" in FUNCTION_WORDS and ABLE_NOW_WORD not in FUNCTION_WORDS
    for document in corpus.documents:
        for sentence in document.sentences:
            level = sentence.proposition.level
            tokens = sentence.sentence.tokens
            if level == ABLE_NOW:
                assert can_now.label in tokens and lexicon.function_word("can").label not in tokens
                assert sentence.readings == ("able_now",)
            elif level == INSTANCE:
                assert can_now.label not in tokens
                assert sentence.readings == ("capacity",)
    # the shared language has no such word
    shared = runs("default", 200, **MANY)
    with pytest.raises(KeyError, match="no function word"):
        shared.planner.lexicon.function_word(ABLE_NOW_WORD)
    # the same propositions in the same order: the words change, the logical forms do not
    assert [s.propositional for d in corpus.documents for s in d.sentences] == [
        s.propositional for d in shared.documents for s in d.sentences
    ]
    with pytest.raises(ConfigError, match="lexicon.can_words"):
        cases("tiny").config(lexicon={"can_words": "both"})


def test_the_renderings_parse_back_and_the_trees_read_back(runs) -> None:
    corpus = runs("default", 200, **MANY)
    operators: Counter = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            if sentence.proposition.level not in TIMED:
                continue
            text = sentence.propositional
            assert propositional(sentence.logical_form) == text
            parsed = parse_propositional(text)
            assert proposition_of(parsed, document.referents) == sentence.proposition
            assert Proposition.from_json(sentence.logical_form) == sentence.proposition
            operators[
                text.split("(")[0] if " AND " not in text else text.split(" AND ")[-1].split("(")[0]
            ] += 1
            record = sentence.sentence
            read = interpret(record.tree, corpus.planner.lexicon, record.referents, record.events)
            assert read == sentence.plan
            assert tuple(leaves(record.tree)) == record.tokens
            # the record of the verb phrase: the time point, qualified by its scene
            (event,) = [e for e in record.events if e is not None]
            assert split_time_key(event[0]) == (
                sentence.proposition.scene,
                sentence.proposition.time,
            )
            assert event[1] == "past" and event[2] is None
    assert operators["HOLDS"] and operators["BECOME"] and operators["ABLE_NOW"], operators
    assert operators["NOT HOLDS"] == 0  # a negative state negates its atom, inside


def test_the_grammar_of_timed_sentences(cases) -> None:
    tiny = cases("tiny")
    world = tiny.world
    fluent = world.fluents[0]
    verb = world.unary[0]
    mouse = world.instances[0]
    subject = NounPhrase(INSTANCE_NP, mouse, world.instance_leaf[0], "the")

    def say(predication: Predication, **settings) -> tuple[str, list]:
        config = tiny.config(**settings)
        lexicon = build_lexicon(config, world, Streams(config.seed))
        realizer = Realizer(config, lexicon, Streams(config.seed))
        plan = SentencePlan(subject, predication)
        check_plan(plan)
        sentence = realizer.realize(plan, Streams(7).grammar)
        assert interpret(sentence.tree, lexicon, sentence.referents, sentence.events) == plan
        return realizer.conceptual(sentence.tokens), list(readings(sentence.tree, lexicon, config))

    at = "SCENE.1.TIME.2"
    noun = world.instance_leaf[0]
    held = Predication(STATE_KIND, fluent, True, tense="past", time=at)
    assert say(held) == (f"THE {noun} IS {fluent}", ["state"])
    assert say(Predication(STATE_KIND, fluent, False, tense="past", time=at)) == (
        f"THE {noun} IS NOT {fluent}",
        ["state"],
    )
    became = Predication(STATE_KIND, fluent, True, tense="past", time=at, become=True)
    assert say(became) == (f"THE {noun} BECOME {fluent}", ["state"])
    assert say(Predication(STATE_KIND, fluent, False, tense="past", time=at, become=True)) == (
        f"THE {noun} BECOME NOT {fluent}",
        ["state"],
    )
    could = Predication(CAN, verb, True, tense="past", time=at)
    assert say(could) == (f"THE {noun} CAN {verb}", ["capacity", "able_now"])
    assert say(Predication(CAN, verb, False, tense="past", time=at)) == (
        f"THE {noun} CAN NOT {verb}",
        ["capacity", "able_now"],
    )
    assert say(could, lexicon={"can_words": "distinct"}) == (
        f"THE {noun} CAN_NOW {verb}",
        ["able_now"],
    )
    # the tense is marked on the auxiliary when it is a word, and never as an affix
    word = {"grammar": {"morphology": {"tense": {"enabled": True, "realization": "word"}}}}
    assert say(became, **word) == (f"THE {noun} BECOME PAST {fluent}", ["state"])
    assert say(held, **word) == (f"THE {noun} IS PAST {fluent}", ["state"])
    assert say(could, **word) == (f"THE {noun} CAN PAST {verb}", ["able_now"])
    affix = {"grammar": {"morphology": {"tense": {"enabled": True}}}}
    assert say(became, **affix) == (f"THE {noun} BECOME {fluent}", ["state"])
    assert say(could, **affix) == (f"THE {noun} CAN {verb}", ["capacity", "able_now"])
    # the plan's level, and the proposition
    assert SentencePlan(subject, became).level == CHANGE
    assert SentencePlan(subject, held).level == STATE
    assert SentencePlan(subject, could).level == ABLE_NOW
    proposition = SentencePlan(subject, became).proposition()
    assert proposition == Proposition(
        CHANGE,
        mouse,
        Predicate(STATE_KIND, fluent),
        True,
        scene="SCENE.1",
        time="TIME.2",
        tense="past",
    )
    assert time_index(proposition.time) == 2


def test_plans_about_time_points_the_grammar_rejects(cases) -> None:
    from semantic_world.corpus.grammar import GrammarError

    world = cases("tiny").world
    fluent, verb, mouse = world.fluents[0], world.unary[0], world.instances[0]
    subject = NounPhrase(INSTANCE_NP, mouse, world.instance_leaf[0], "the")
    bad = [
        Predication(STATE_KIND, fluent),  # a fluent needs a time point
        Predication(STATE_KIND, fluent, time="SCENE.1.TIME.2"),  # and a tense
        Predication(STATE_KIND, fluent, tense="past", time="TIME.2"),  # qualified by its scene
        Predication(CAN, verb, tense="past", time="SCENE.1.TIME.2", become=True),  # become: fluents
        Predication(
            CAN,
            verb,
            tense="past",
            time="SCENE.1.TIME.2",
            event="SCENE.1.EVENTINSTANCE.1",
            aspect="simple",
        ),
    ]
    for predication in bad:
        with pytest.raises(GrammarError):
            check_plan(SentencePlan(subject, predication))
    # a relative clause is never about a time point
    from semantic_world.corpus.grammar import RelativeClause

    clause = RelativeClause((Predication(STATE_KIND, fluent, tense="past", time="SCENE.1.TIME.2"),))
    with pytest.raises(GrammarError):
        check_plan(
            SentencePlan(
                NounPhrase(INSTANCE_NP, mouse, world.instance_leaf[0], "the", clause=clause),
                Predication(CAN, verb),
            )
        )


def test_timed_propositions_the_truth_tests_reject(cases) -> None:
    tiny = cases("tiny")
    corpus_planner = __import__("semantic_world.corpus.planner", fromlist=["Planner"]).Planner(
        tiny.config(), tiny.world
    )
    documents = corpus_planner.generate(10)
    truth = corpus_planner.truth
    scene = next(d.scenes[0] for d in documents if d.scenes)
    participant = scene.participants[0]
    fluent = tiny.world.fluents[0]
    final = time_index(scene.final)

    def judge(**fields):
        base = {
            "level": STATE,
            "subject": participant,
            "predicate": Predicate(STATE_KIND, fluent),
            "scene": scene.label,
            "time": "TIME.1",
            "tense": "past",
        }
        return truth.evaluate(Proposition(**{**base, **fields}))

    assert judge().valid
    assert not judge(time=f"TIME.{final + 1}").valid  # past the scene's end
    assert not judge(level=CHANGE, time=scene.final).valid  # a change needs a next time point
    assert not judge(tense="present").valid
    assert not judge(level=INSTANCE, scene=None, time=None, tense=None).valid  # timeless
    assert not judge(aspect="simple").valid
    assert not judge(
        subject=tiny.world.instances[-1]
        if tiny.world.instances[-1] not in scene.participants
        else "INSTANCE.9.9.9"
    ).valid
    assert not judge(level=ABLE_NOW).valid  # a fluent is not an event type
    assert not judge(level=STATE, predicate=Predicate(CAN, tiny.world.unary[0])).valid
    assert judge(level=ABLE_NOW, predicate=Predicate(CAN, tiny.world.unary[0])).valid
