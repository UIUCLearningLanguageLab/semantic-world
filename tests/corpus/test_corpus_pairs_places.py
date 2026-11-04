"""Stages c4a to c4c of ``docs/specs/WORLD_PHASE_C.md``: the corpus on worlds with pair fluents
and places, and its sentences and test items about them.

Since stage c4a ``data/world/tiny.yaml`` and ``data/world/default.yaml`` carry the phase (c)
settings (Jon's ruling WM.63): pair fluents between entities, a location fluent, moves and acts
at a place, ``SAME`` on the location fluent, and derived fluents over ``SOME`` and ``SOMEOF``.
Since stage c4b the corpus names them at the instance level: the pair fluents and the ``SAME``
literals that some precondition or condition reads get prepositions, the places' categories,
features, and event types get words, a narrative reports a move or an act as a transitive verb
with the place as its object, a place is a referent introduced with ``a`` and mentioned later
with ``the``, and a result sentence states a pair change of the event's own effects, a move's
new place, or the loss of the old place, with ``become`` and the preposition (predicate kind
``pair_state``; Jon's decision WM.72). Since stage c4c (Jon's rulings WM.73) an initial-state
sentence states a participant's place, its pair with a referent already mentioned, or ``SAME``
with one; place poles tell places apart; a place is described by its static features and two
places are compared on a place scalar; a blocked sentence can be about a move or an act, never a
move to the agent's own place (WM.53); and the state, event, and ``able_now`` sets hold pair,
location, ``SAME``, and place event items, with the change ``object`` (WM.56). The causal
statements with pair literals are stage c5a's (a conditional effect whose condition holds a pair
literal is stated in neither form, WM.59).
Every sentence must be true by the independent oracle, whose replay carries the pairs and the
places, and whose pairs and places must agree with ``history.replay`` at every time point of
every scene. The fast tier runs the tiny corpus, the slow tier the default one.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from corpus_support import REPO, TINY_WORLD, corpus_config

from semantic_world.corpus import Facts, Streams, build_lexicon, load_world
from semantic_world.corpus.grammar import (
    INSTANCE_NP,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
    check_plan,
)
from semantic_world.corpus.histories import scene_events
from semantic_world.corpus.interpret import interpret
from semantic_world.corpus.lexicon import PREPOSITION
from semantic_world.corpus.logical import logical_form, with_cut
from semantic_world.corpus.planner import (
    BLOCKED,
    EVENT_SECTION,
    INITIAL_STATE,
    PLACE_COMPARATIVE,
    PLACE_DESCRIPTION,
    RESULT,
)
from semantic_world.corpus.propositions import (
    ABLE_NOW,
    CHANGE,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    EFFECT,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    PAIR_STATE,
    PLACE_EVENT,
    STATE,
    VERB,
    EventTerm,
    Literal,
    condition_of,
    effect_matches,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, leaves
from semantic_world.corpus.renderings import (
    formula,
    parse_propositional,
    proposition_of,
    propositional,
)
from semantic_world.world.history import replay

WORLDS = ("tiny", pytest.param("default", marks=pytest.mark.slow))
UNNAMED_LABELS = ("SOME.", "SOMEOF.", "EVERYPLACE", "SOMEPLACE")
"""The labels of phase (c) that no word of the corpus carries before stage c5a: ``SOME`` and
``SOMEOF`` (WM.55), and the bare symbols for places."""
PHASE_C_LABELS = ("PAIRFL", "PLACE", "SAME.") + UNNAMED_LABELS
"""Every label of phase (c)."""
MANY = {
    "documents": {
        "initial_state_rate": 0.6,
        "result_rate": 0.9,
        "blocked_rate": 0.5,
        "comparative_rate": 0.5,
        "place_description_rate": 0.5,
    },
    "propositions": {"negation_rate": {"able_now": 0.6}},
}
"""Many state, result, blocked, comparative, and place description sentences, so that the tests
see every kind."""
CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
ADPOSITIONS = ("preposition", "postposition")


def scenes_json(corpus) -> dict[str, dict]:
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


def _carries(value, labels: tuple[str, ...] = PHASE_C_LABELS) -> bool:
    text = json.dumps(value)
    return any(label in text for label in labels)


def glosses(sentence) -> str:
    """The sentence in glosses: the formal rendering without the lexeme labels."""
    words = []
    for word in sentence.formal.split():
        gloss, _, rest = word.partition("/")
        _, hyphen, affix = rest.partition("-")
        words.append(gloss + hyphen + affix)
    return " ".join(words)


# ---------------------------------------------------------------------------------------------
# Acceptance: the oracle's replay agrees with history.replay at every time point
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_oracles_pairs_and_places_agree_with_history_replay(cases, runs, name) -> None:
    """At every time point of every scene of the corpus, the pairs the oracle replays (both
    directions of a symmetric pair, a move's two changes, the total functional replacement) are
    the pairs of the runtime's replayed state, and every entity's place is the runtime's."""
    case = cases(name)
    corpus = runs(name)
    oracle = case.oracle()
    definition = case.world.definition
    labels = definition.entity_labels
    assert definition.has_location_fluents and len(definition.pair_fluents) > 1
    moves = pair_changes = 0
    for scene in corpus.planner.scenes:
        states = replay(definition, scene)
        oracle_pairs = oracle.pairs_at(json.loads(json.dumps(scene.to_json())))
        assert len(oracle_pairs) == len(states) == len(scene.steps) + 1
        for state, pairs in zip(states, oracle_pairs, strict=True):
            for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True):
                targets = definition.place_labels if fluent.total else labels
                expected = {
                    (labels[int(i)], targets[int(j)])
                    for i, j in zip(*np.nonzero(matrix), strict=True)
                }
                assert pairs[fluent.label] == expected, (scene.label, fluent.label)
            places = state.places_record(definition)
            for entity, at in places.items():
                for fluent, place in at.items():
                    assert (
                        oracle.place_at(scene.to_json(), states.index(state) + 1, entity, fluent)
                        == place
                    )
        for step in scene.steps:
            for event in step.events:
                if definition.event_type(event.type).move:
                    moves += 1
                pair_changes += sum(1 for c in event.changes if c.pair)
    assert moves > 0 and pair_changes > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: every sentence true, every false item false, by the oracle
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_sentence_is_true_and_every_false_item_false_by_the_oracle(cases, runs, name) -> None:
    """Every sentence of the corpus is true by the oracle (the pair changes and the moves and
    acts of stage c4b judged on the replayed pairs and places, and the timed sentences with the
    pairs and the places deciding what is legal), and every false test item is false; the corpus
    holds state, change, and ``able_now`` sentences, pair changes, and narrated place events."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    scenes = scenes_json(corpus)
    cuts = corpus.planner.truth.cuts
    levels: Counter = Counter()
    kinds: Counter = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            levels[proposition.level] += 1
            kinds[proposition.predicate.kind] += 1
            if proposition.level == EVENT:
                assert oracle.event(sentence.logical_form, scenes[proposition.scene])
                continue
            scene = None if proposition.scene is None else scenes[proposition.scene]
            form = with_cut(proposition.to_json(), cuts)
            assert oracle.truth(form, scene) is True, sentence.label
    assert levels[STATE] > 0 and levels[CHANGE] > 0 and levels[ABLE_NOW] > 0
    assert kinds[PAIR_STATE] > 0 and kinds[PLACE_EVENT] > 0
    items = 0
    for test_set in corpus.test_sets:
        for true, false in test_set.pairs:
            for item, expected in ((true, True), (false, False)):
                form = item.input["logical_form"]
                scene = scenes.get(form.get("scene")) if form.get("scene") else None
                if form["level"] == EVENT:
                    judged = oracle.event(form, scene)
                else:
                    judged = oracle.truth(form, scene)
                assert judged is expected, (test_set.name, item.input["propositional"])
                items += 1
    assert items > 100


@pytest.mark.parametrize("name", WORLDS)
def test_blocked_sentences_are_judged_with_the_pairs_and_the_places(cases, runs, name) -> None:
    """A negative ``ABLE_NOW`` sentence is blocked by a precondition that the replayed state
    fails, and since stage c4a that precondition may be a pair literal: ``SAME`` on the location
    fluent, a pair between the two participants, ``SOME`` or ``SOMEOF``, or a compiled literal.
    The oracle, replaying the pairs, agrees with the corpus on every such sentence, the blocked
    moves and acts of stage c4c included."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    scenes = scenes_json(corpus)
    definition = case.world.definition
    blocked = pair_blocked = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            if proposition.level != ABLE_NOW:
                continue
            if proposition.polarity:
                continue
            blocked += 1
            scene = scenes[proposition.scene]
            assert oracle.able_now(sentence.logical_form, scene)
            # which literals of the event type's precondition read a pair
            for leaf in case.world.event_types_below(proposition.predicate.label):
                if any(lit.pair for lit in definition.event_type(leaf).precondition):
                    pair_blocked += 1
                    break
    assert blocked > 0
    # the tiny world's two-place event types carry SAME on the location fluent (the quota), so
    # some blocked sentence is about an event type whose precondition reads a pair
    assert pair_blocked > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: the lexicon of pairs and places (stage c4b)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_lexicon_names_the_pair_fluents_and_the_places(cases, runs, name) -> None:
    """The lexicon's phase (c) concepts (``WORLD_PHASE_C.md``, "Lexicon"): a preposition for
    every pair fluent and every ``SAME`` literal that some precondition or condition reads, the
    location fluent's included; a noun for every category of places; an adjective or a part noun
    for every place feature; two pole adjectives per place scalar; a transitive verb for every
    place event type with dynamics; and a quarter of the place patient capacities. No word for
    ``SOME``, ``SOMEOF``, ``SOMEPLACE``, or ``them``. The concepts come after the generic noun,
    so every earlier content lexeme keeps its label, and the request lists the new lexemes with
    their parts of speech, the prepositions taking no affix."""
    from semantic_world.corpus import wordform_request

    corpus = runs(name)
    lexicon = corpus.planner.lexicon
    world = cases(name).world
    definition = world.definition
    of = lexicon.of_type
    read = {
        lit.pair_fluent
        for et in definition.event_types
        for lit in list(et.precondition) + [c for e in et.effects for c in e.condition]
        if lit.pair and lit.kind == "pair"
    }
    same = {
        lit.pair_fluent
        for et in definition.event_types
        for lit in list(et.precondition) + [c for e in et.effects for c in e.condition]
        if lit.pair and lit.kind == "pair_same"
    }
    assert [c.label for c in of("pair")] == [f for f in world.pair_fluents if f in read]
    assert world.location_fluents[0] in read  # a move's compiled literal reads it
    assert [c.label for c in of("same")] == [f"SAME.{f}" for f in world.pair_fluents if f in same]
    assert all(c.pos == PREPOSITION for c in of("pair") + of("same"))
    assert [c.label for c in of("place_category")] == list(world.place_categories)
    assert [c.label for c in of("place_is")] == list(world.place_features["property"])
    assert [c.label for c in of("place_has")] == list(world.place_features["part"])
    assert [c.label for c in of("place_scalar")] == [
        f"{s}.{pole}" for s in world.place_scalars for pole in ("HIGH", "LOW")
    ]
    assert [c.label for c in of("place_event")] == [
        label
        for label in world.place_event_types
        if definition.event_type(label).performable
        and label not in lexicon.event_types_without_word
    ]
    assert [c.label for c in of("place_patient_projection")] == list(world.place_patient_capacities)
    proportion = corpus.config.lexicon.named_proportion["place_patient_projection"]
    named = sum(lexicon.is_named(c.label) for c in of("place_patient_projection"))
    assert named == int(np.floor(proportion * len(of("place_patient_projection")) + 0.5))
    for concept_type in ("pair", "same", "place_category", "place_is", "place_event"):
        assert all(lexicon.is_named(c.label) for c in of(concept_type))
    assert not any(_carries(c.label, UNNAMED_LABELS) for c in lexicon.concepts)
    assert not any(x.gloss == "them" for x in lexicon.lexemes)
    # the earlier content lexemes keep their labels: every lexeme before the first phase (c)
    # lexeme names no phase (c) concept, and the generic noun stands right before them
    first = next(i for i, x in enumerate(lexicon.content_lexemes) if _carries(x.concept))
    assert all(not _carries(x.concept) for x in lexicon.content_lexemes[:first])
    assert lexicon.content_lexemes[first - 1].concept == "THING"
    request = wordform_request(corpus)
    by_label = {x["label"]: x for x in request["lexemes"]}
    for lexeme in lexicon.content_lexemes:
        assert by_label[lexeme.label]["pos"] == lexeme.pos
    assert PREPOSITION not in request["takes"]
    assert any(x["pos"] == PREPOSITION for x in request["lexemes"])
    assert not any(_carries(x, UNNAMED_LABELS) for x in request["lexemes"])


# ---------------------------------------------------------------------------------------------
# Acceptance: moves and acts in narratives, places as referents
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_narratives_report_the_moves_and_the_acts(cases, runs, name) -> None:
    """An event of a place event type is narrated as a transitive verb with the place as its
    object (``WORLD_PHASE_C.md``, "Grammar"), in a main clause or in a relative clause; a
    situational narrative reports every event of its scene in time order, the moves and the acts
    among them, as far as its length goes; and the oracle finds every reported place event in
    its scene."""
    corpus = runs(name)
    world = cases(name).world
    oracle = cases(name).oracle()
    scenes = scenes_json(corpus)
    narrated: Counter = Counter()
    in_clauses = 0
    for document in corpus.documents:
        if not document.scenes:
            continue
        events = {e.label: e for scene in document.scenes for e in scene_events(scene)}
        reported = [s.proposition.event for s in document.sentences if s.section == EVENT_SECTION]
        if document.type == "situational":
            (scene,) = document.scenes
            assert reported == [e.label for e in scene_events(scene)][: len(reported)]
        for sentence in document.sentences:
            plan = sentence.plan
            for phrase in plan.noun_phrases():
                if phrase.clause is None:
                    continue
                for predication in phrase.clause.predications:
                    if predication.kind == PLACE_EVENT:
                        in_clauses += 1
                        assert predication.object is not None and predication.object.place
            if sentence.section != EVENT_SECTION:
                continue
            proposition = sentence.proposition
            event = events[proposition.event]
            if not event.place:
                assert proposition.predicate.kind != PLACE_EVENT
                continue
            assert proposition.predicate.kind == PLACE_EVENT
            assert proposition.predicate.label == event.type
            assert proposition.predicate.patient == event.patient
            assert event.patient in (document.scenes[0].places or ()) or any(
                event.patient in (scene.places or ()) for scene in document.scenes
            )
            kind = world.place_event_kind(event.type)
            assert kind in ("move", "act")
            narrated[kind] += 1
            target = plan.predication.object
            assert target is not None and target.place and not target.pronoun
            assert target.noun in world.place_path(event.patient)
            assert oracle.event(sentence.logical_form, scenes[proposition.scene])
            assert sentence.readings == ("event",)
    assert narrated["move"] > 0
    if name == "default":
        assert narrated["act"] > 0 and in_clauses > 0


@pytest.mark.parametrize("name", WORLDS)
def test_result_sentences_state_pair_changes(cases, runs, name) -> None:
    """A result sentence states a pair change of the event's own effects with ``become`` and
    the preposition (WM.72): a move's new place (``BECOME(..., PAIRFL.k(REF.a, REF.q))``), the
    loss of the old place (``NOT``), or a pair between two participants; the change is of the
    event's step, the event recorded it, and the oracle judges it true and made by the event."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    world = case.world
    scenes = scenes_json(corpus)
    found: Counter = Counter()
    for document in corpus.documents:
        events = {e.label: e for scene in document.scenes for e in scene_events(scene)}
        recorded = {
            e.label: e for scene in document.scenes for step in scene.steps for e in step.events
        }
        for index, sentence in enumerate(document.sentences):
            proposition = sentence.proposition
            if sentence.section != RESULT or proposition.predicate.kind != PAIR_STATE:
                continue
            assert proposition.level == CHANGE
            before = next(
                s for s in reversed(document.sentences[:index]) if s.section == EVENT_SECTION
            )
            event = events[before.proposition.event]
            assert proposition.scene == event.scene and proposition.time == f"TIME.{event.step}"
            fluent, second = proposition.predicate.label, proposition.predicate.patient
            assert proposition.subject == event.agent or event.involves(proposition.subject)
            changes = [
                c
                for c in recorded[event.label].changes
                if c.pair and c.pair_fluent == fluent and c.first == proposition.subject
            ]
            assert any(c.second == second and c.to == proposition.polarity for c in changes)
            scene = scenes[event.scene]
            assert oracle.change(sentence.logical_form, scene)
            assert oracle.made_by(
                scene, event.step, proposition.subject, fluent, event.label, second=second
            )
            assert proposition.grounding["caused_by"] == event.label
            if world.is_location_fluent(fluent):
                assert world.is_place(second) and world.place_event_kind(event.type) == "move"
                found["new place" if proposition.polarity else "old place"] += 1
            else:
                assert not world.is_place(second)
                found["pair"] += 1
            # the plan says it with become and the preposition's object
            predication = sentence.plan.predication
            assert predication.become and predication.object is not None
            assert predication.object.place == world.is_place(second)
            assert sentence.readings == ("state",)
    assert found["new place"] > 0 and found["old place"] > 0
    # a pair change is a result sentence, and a pair state at a time point an initial state or
    # (stage c4c) a test item
    for document in corpus.documents:
        for sentence in document.sentences:
            if sentence.proposition.predicate.kind == PAIR_STATE:
                assert (sentence.proposition.level == CHANGE) == (sentence.section == RESULT)


@pytest.mark.parametrize("name", WORLDS)
def test_places_are_referents(cases, runs, name) -> None:
    """A place is introduced with ``a`` and mentioned later with ``the``, numbered among the
    referents of its document with the entities, named by a category of places above it, told
    apart from the other places of the document's scenes by place properties, parts, and (stage
    c4c) poles (and kept, counted as not distinguished, when they cannot), never a pronoun,
    never with a relative clause, and the subject of a description of a place or a comparison
    of places alone; no mention of anything carries a pair fluent or a location as a
    modifier."""
    corpus = runs(name, **MANY)
    world = cases(name).world
    rules = corpus.planner.rules
    introduced = later = 0
    for document in corpus.documents:
        seen: set[str] = set()
        places = tuple(dict.fromkeys(q for scene in document.scenes for q in (scene.places or ())))
        for sentence in document.sentences:
            if sentence.plan.subject.place:
                assert sentence.section in (PLACE_DESCRIPTION, PLACE_COMPARATIVE)
                assert sentence.proposition.level == INSTANCE
            for phrase in sentence.plan.noun_phrases():
                for literal in phrase.restriction:
                    assert not literal.feature.startswith(("PAIRFL.", "SAME."))
                if phrase.kind != INSTANCE_NP or not world.is_place(phrase.referent):
                    continue
                place = phrase.referent
                assert phrase.place and not phrase.pronoun and phrase.clause is None
                assert phrase.noun in world.place_path(place)
                assert all(literal.feature.startswith("PLACE") for literal in phrase.restriction)
                assert all(
                    rules.fits(world.place_index[place], x, phrase.noun) for x in phrase.restriction
                )
                assert place in places
                if place not in seen:
                    assert phrase.determiner == "a" and not phrase.restriction
                    seen.add(place)
                    introduced += 1
                else:
                    assert phrase.determiner == "the"
                    later += 1
                    fitting = rules.matches(phrase, places)
                    # the place is among them; when the modifiers cannot tell it apart the
                    # mention is kept as it is, and counted (CG.43)
                    assert place in fitting
            for phrase, flag in zip(sentence.sentence.phrases, sentence.distinguished, strict=True):
                if phrase.place and phrase.determiner == "the":
                    assert flag == (rules.matches(phrase, places) == [phrase.referent])
        # the referent labels number entities and places together, by first mention
        labels = list(document.referents)
        assert labels == [f"REF.{i}" for i in range(1, len(labels) + 1)]
        for instance in document.referents.values():
            if world.is_place(instance):
                assert instance in places
    assert introduced > 0 and later > 0
    stats = corpus.stats["mentions"]
    assert stats["place_mentions"] == introduced + later
    assert stats["place_indefinite"] == introduced and stats["place_definite"] == later


# ---------------------------------------------------------------------------------------------
# Acceptance: the grammar of prepositional predicates and place event types
# ---------------------------------------------------------------------------------------------


def the(instance: str, noun: str, *restriction: Literal, clause=None, determiner="the"):
    return NounPhrase(INSTANCE_NP, instance, noun, determiner, restriction, clause)


def _tiny_plans() -> dict[str, SentencePlan]:
    """Fixed plans on the tiny world: a location state and change, a move, and a move reported
    in a relative clause. ``PAIRFL.3`` is the tiny world's location fluent and
    ``PLACEEVENTTYPE.1`` a move (both with words)."""
    time = "SCENE.1.TIME.2"
    owl = the("INSTANCE.1.1.1", "CATEGORY.1.1")
    place = the("PLACE.1.1", "PLACECATEGORY.1")
    other_place = the("PLACE.2.1", "PLACECATEGORY.2", Literal("PLACEPROPERTY.1"))
    return {
        "location": SentencePlan(
            owl, Predication(PAIR_STATE, "PAIRFL.3", True, place, tense="past", time=time)
        ),
        "left": SentencePlan(
            owl,
            Predication(
                PAIR_STATE, "PAIRFL.3", False, other_place, tense="past", time=time, become=True
            ),
        ),
        "went": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1", determiner="a"),
            Predication(
                PLACE_EVENT,
                "PLACEEVENTTYPE.1",
                True,
                the("PLACE.2.1", "PLACECATEGORY.2", determiner="a"),
                "SCENE.1.EVENTINSTANCE.2",
                "past",
                "simple",
            ),
        ),
        "clause": SentencePlan(
            the(
                "INSTANCE.1.1.1",
                "CATEGORY.1.1",
                clause=RelativeClause(
                    (
                        Predication(
                            PLACE_EVENT,
                            "PLACEEVENTTYPE.1",
                            True,
                            place,
                            "SCENE.1.EVENTINSTANCE.1",
                            "past",
                            "progressive",
                        ),
                    )
                ),
            ),
            Predication(
                VERB,
                "EVENTTYPE2.1.1",
                True,
                the("INSTANCE.2.1.1", "CATEGORY.2.1"),
                "SCENE.1.EVENTINSTANCE.3",
                "past",
                "simple",
            ),
        ),
        # stage c4c: a SAME state, a description of a place, a comparison of two places, and a
        # blocked move
        "same": SentencePlan(
            owl,
            Predication(
                PAIR_STATE,
                "SAME.PAIRFL.3",
                True,
                the("INSTANCE.2.1.1", "CATEGORY.2.1"),
                tense="past",
                time=time,
            ),
        ),
        "described": SentencePlan(
            the("PLACE.1.1", "PLACECATEGORY.1", Literal("PLACESCALARDIM.1.HIGH")),
            Predication("property", "PLACEPROPERTY.1", False),
        ),
        "wider": SentencePlan(
            place,
            Predication("comparative", "PLACESCALARDIM.1.HIGH", True, other_place),
        ),
        "could_not": SentencePlan(
            owl,
            Predication(PLACE_EVENT, "PLACEEVENTTYPE.1", False, place, tense="past", time=time),
        ),
    }


ENGLISH = {
    "location": "the CATEGORY.1.1 is PAIRFL.3 the PLACECATEGORY.1",
    "left": "the CATEGORY.1.1 become not PAIRFL.3 the PLACEPROPERTY.1 PLACECATEGORY.2",
    "went": "a CATEGORY.1.1 PLACEEVENTTYPE.1 a PLACECATEGORY.2",
    "clause": (
        "the CATEGORY.1.1 that PLACEEVENTTYPE.1 the PLACECATEGORY.1 EVENTTYPE2.1.1 the CATEGORY.2.1"
    ),
    "same": "the CATEGORY.1.1 is SAME.PAIRFL.3 the CATEGORY.2.1",
    "described": "the PLACESCALARDIM.1.HIGH PLACECATEGORY.1 is not PLACEPROPERTY.1",
    "wider": (
        "the PLACECATEGORY.1 is PLACESCALARDIM.1.HIGH-COMPARATIVE than the PLACEPROPERTY.1 "
        "PLACECATEGORY.2"
    ),
    "could_not": "the CATEGORY.1.1 can not PLACEEVENTTYPE.1 the PLACECATEGORY.1",
}
TIMELESS = ("described", "wider")
"""The fixed plans about no time point: the instance-level sentences about places."""
LOCATION_ORDERS = {
    ("SVO", "preposition"): "the CATEGORY.1.1 is PAIRFL.3 the PLACECATEGORY.1",
    ("SVO", "postposition"): "the CATEGORY.1.1 is the PLACECATEGORY.1 PAIRFL.3",
    ("SOV", "preposition"): "the CATEGORY.1.1 PAIRFL.3 the PLACECATEGORY.1 is",
    ("VSO", "preposition"): "is the CATEGORY.1.1 PAIRFL.3 the PLACECATEGORY.1",
    ("VOS", "postposition"): "is the PLACECATEGORY.1 PAIRFL.3 the CATEGORY.1.1",
    ("OVS", "preposition"): "PAIRFL.3 the PLACECATEGORY.1 is the CATEGORY.1.1",
    ("OSV", "postposition"): "the PLACECATEGORY.1 PAIRFL.3 the CATEGORY.1.1 is",
}
"""The location sentence in some clause orders, with the preposition before or after its noun
phrase: the prepositional phrase stands where an object stands."""


def _realize(case, plan: SentencePlan, **grammar):
    settings = {"grammar": {"can_rate": {"class": 1.0}, **grammar}}
    realizer = case.realizer(**settings)
    sentence = realizer.realize(plan, np.random.default_rng(0))
    lexicon = case.lexicon(**settings)
    assert tuple(leaves(sentence.tree)) == sentence.tokens
    read = interpret(
        sentence.tree,
        lexicon,
        sentence.referents,
        sentence.events,
        sentence.plan.quantifier,
        sentence.plan.predication.standard,
    )
    assert read == plan
    return sentence


def test_prepositional_predicates_in_the_six_clause_orders(cases) -> None:
    """With the preposition before and after its noun phrase (``word_order.adposition``), the
    trees of a pair state, a pair change, and a place event type are correct in all six clause
    orders: the words are the plan's, the leaves are the tokens, ``interpret(tree)`` recovers the
    plan, and the propositional rendering of the logical form parses back to it."""
    tiny = cases("tiny")
    plans = _tiny_plans()
    for name, plan in plans.items():
        check_plan(plan)
        assert glosses(_realize(tiny, plan)) == ENGLISH[name]
    for order, adposition in itertools.product(CLAUSE_ORDERS, ADPOSITIONS):
        grammar = {"word_order": {"clause": order, "adposition": adposition}}
        for name, plan in plans.items():
            sentence = _realize(tiny, plan, **grammar)
            assert sorted(glosses(sentence).split()) == sorted(ENGLISH[name].split())
            proposition = plan.proposition()
            form = logical_form(plan, proposition, {}, 1.0, {"HIGH": 0.75, "LOW": 0.25})
            text = propositional(form)
            parsed = parse_propositional(text)
            assert parsed == formula(form)
            assert proposition_of(parsed) == proposition
            assert (sentence.events[0] is not None) == (name not in TIMELESS)
        if (order, adposition) in LOCATION_ORDERS:
            assert (
                glosses(_realize(tiny, plans["location"], **grammar))
                == LOCATION_ORDERS[order, adposition]
            )
    # the readings: a pair state, a pair change, and a SAME state read as a state, a move as an
    # event, a description or a comparison of a place as an instance-level sentence does, and
    # a blocked move as every "can" sentence about an instance
    for name, reading in (
        ("location", ("state",)),
        ("left", ("state",)),
        ("same", ("state",)),
        ("went", ("event",)),
        ("described", ("capacity",)),
        ("wider", ("capacity",)),
        ("could_not", ("capacity", "able_now")),
    ):
        sentence = _realize(tiny, plans[name])
        assert readings(sentence.tree, tiny.lexicon(), tiny.config()) == reading
    # the renderings of the fixed plans
    forms = {
        name: propositional(logical_form(plan, plan.proposition(), {}, 1.0))
        for name, plan in plans.items()
    }
    assert forms["location"] == (
        "{CATEGORY.1.1(INSTANCE.1.1.1) AND PLACECATEGORY.1(PLACE.1.1)} "
        "HOLDS(SCENE.1, TIME.2, PAST, PAIRFL.3(INSTANCE.1.1.1, PLACE.1.1))"
    )
    assert forms["left"] == (
        "{CATEGORY.1.1(INSTANCE.1.1.1) AND PLACECATEGORY.2(PLACE.2.1) AND "
        "PLACEPROPERTY.1(PLACE.2.1)} "
        "BECOME(SCENE.1, TIME.2, PAST, NOT PAIRFL.3(INSTANCE.1.1.1, PLACE.2.1))"
    )
    assert forms["went"] == (
        "CATEGORY.1.1(INSTANCE.1.1.1) AND PLACECATEGORY.2(PLACE.2.1) AND "
        "EVENT(SCENE.1.EVENTINSTANCE.2, PAST, SIMPLE, PLACEEVENTTYPE.1(INSTANCE.1.1.1, PLACE.2.1))"
    )
    assert forms["same"] == (
        "{CATEGORY.1.1(INSTANCE.1.1.1) AND CATEGORY.2.1(INSTANCE.2.1.1)} "
        "HOLDS(SCENE.1, TIME.2, PAST, SAME.PAIRFL.3(INSTANCE.1.1.1, INSTANCE.2.1.1))"
    )
    assert forms["described"] == (
        "{PLACECATEGORY.1(PLACE.1.1) AND ABOVE(PLACESCALARDIM.1, PLACE.1.1, PLACECATEGORY.1, 1.0)} "
        "NOT PLACEPROPERTY.1(PLACE.1.1)"
    )
    assert forms["wider"] == (
        "{PLACECATEGORY.1(PLACE.1.1) AND PLACECATEGORY.2(PLACE.2.1) AND "
        "PLACEPROPERTY.1(PLACE.2.1)} "
        "GREATER(PLACESCALARDIM.1, PLACE.1.1, PLACE.2.1)"
    )
    assert forms["could_not"] == (
        "{CATEGORY.1.1(INSTANCE.1.1.1) AND PLACECATEGORY.1(PLACE.1.1)} "
        "NOT ABLE_NOW(SCENE.1, TIME.2, PAST, PLACEEVENTTYPE.1(INSTANCE.1.1.1, PLACE.1.1))"
    )


def test_pairs_between_entities_in_the_six_clause_orders(tmp_path: Path) -> None:
    """On the world of ``tiny_pairs.yaml``, whose event file reads pair fluents between entities
    in its preconditions, a pair state and a pair change between two entities, with the
    preposition before and after, in all six clause orders."""
    config = corpus_config("data/world/tiny_pairs.yaml")
    world = load_world(config)
    streams = Streams(config.seed)
    lexicon = build_lexicon(config, world, streams)
    prepositions = [
        c.label for c in lexicon.of_type("pair") if not world.is_location_fluent(c.label)
    ]
    assert prepositions, "the event file reads a pair fluent between entities"
    fluent = prepositions[0]
    owl, mouse = world.instances[0], world.instances[1]
    time = "SCENE.1.TIME.3"
    plans = {
        "near": SentencePlan(
            the(owl, world.instance_leaf[0]),
            Predication(
                PAIR_STATE,
                fluent,
                True,
                the(mouse, world.instance_leaf[1]),
                tense="past",
                time=time,
            ),
        ),
        "became": SentencePlan(
            NounPhrase(INSTANCE_NP, owl),
            Predication(
                PAIR_STATE,
                fluent,
                True,
                the(mouse, world.instance_leaf[1]),
                tense="past",
                time=time,
                become=True,
            ),
        ),
    }
    for order, adposition in itertools.product(CLAUSE_ORDERS, ADPOSITIONS):
        settings = {"grammar": {"word_order": {"clause": order, "adposition": adposition}}}
        config = corpus_config("data/world/tiny_pairs.yaml", **settings)
        realizer = Realizer(config, lexicon, streams)
        for name, plan in plans.items():
            check_plan(plan)
            sentence = realizer.realize(plan, np.random.default_rng(0))
            read = interpret(
                sentence.tree, lexicon, sentence.referents, sentence.events, None, None
            )
            assert read == plan
            assert fluent in sentence.conceptual
            assert readings(sentence.tree, lexicon, config) == ("state",)
            form = logical_form(plan, plan.proposition(), {owl: "REF.1", mouse: "REF.2"}, 1.0)
            text = propositional(form)
            assert (
                proposition_of(parse_propositional(text), {"REF.1": owl, "REF.2": mouse})
                == plan.proposition()
            )
            if name == "near":
                assert text.endswith(f"HOLDS(SCENE.1, TIME.3, PAST, {fluent}(REF.1, REF.2))")
            else:
                assert text.endswith(f"BECOME(SCENE.1, TIME.3, PAST, {fluent}(REF.1, REF.2))")
    del tmp_path


def test_plans_the_grammar_refuses(cases) -> None:
    """The limits of "Grammar" (``WORLD_PHASE_C.md``): a pair fluent or a location is never a
    modifier of a mention, a pair state is never in a relative clause, a place is never a
    pronoun, never takes a relative clause, and is the subject of a description of its own
    features or a comparison of places alone (stage c4c), a place feature is said of a place,
    the object of a place event type is a place, and the object of ``SAME`` an entity."""
    from semantic_world.corpus.grammar import GrammarError

    tiny = cases("tiny")
    plans = _tiny_plans()
    realizer = tiny.realizer()
    rng = np.random.default_rng(0)
    bad = {
        "a pair fluent as a modifier": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1", Literal("PAIRFL.3")),
            plans["location"].predication,
        ),
        "a pair state in a relative clause": SentencePlan(
            the(
                "INSTANCE.1.1.1",
                "CATEGORY.1.1",
                clause=RelativeClause((plans["location"].predication,)),
            ),
            Predication(
                "event_type1",
                "EVENTTYPE1.1",
                event="SCENE.1.EVENTINSTANCE.1",
                tense="past",
                aspect="simple",
            ),
        ),
        "a place as a pronoun": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1"),
            Predication(
                PAIR_STATE,
                "PAIRFL.3",
                True,
                NounPhrase(INSTANCE_NP, "PLACE.1.1"),
                tense="past",
                time="SCENE.1.TIME.2",
            ),
        ),
        "a place with a relative clause": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1"),
            Predication(
                PAIR_STATE,
                "PAIRFL.3",
                True,
                the(
                    "PLACE.1.1",
                    "PLACECATEGORY.1",
                    clause=RelativeClause((Predication("event_type1", "EVENTTYPE1.1"),)),
                ),
                tense="past",
                time="SCENE.1.TIME.2",
            ),
        ),
        "a place as the subject of a pair state": SentencePlan(
            the("PLACE.1.1", "PLACECATEGORY.1"), plans["location"].predication
        ),
        "a place as the subject of an event": SentencePlan(
            the("PLACE.1.1", "PLACECATEGORY.1"),
            Predication(
                "event_type1",
                "EVENTTYPE1.1",
                event="SCENE.1.EVENTINSTANCE.1",
                tense="past",
                aspect="simple",
            ),
        ),
        "a place as the subject of a capacity": SentencePlan(
            the("PLACE.1.1", "PLACECATEGORY.1"), Predication("event_type1", "EVENTTYPE1.1")
        ),
        "a place feature said of an entity": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1"), Predication("property", "PLACEPROPERTY.1")
        ),
        "a place compared with an entity": SentencePlan(
            the("PLACE.1.1", "PLACECATEGORY.1"),
            Predication(
                "comparative", "PLACESCALARDIM.1.HIGH", True, the("INSTANCE.1.1.1", "CATEGORY.1.1")
            ),
        ),
        "a place as the object of SAME": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1"),
            Predication(
                PAIR_STATE,
                "SAME.PAIRFL.3",
                True,
                the("PLACE.1.1", "PLACECATEGORY.1"),
                tense="past",
                time="SCENE.1.TIME.2",
            ),
        ),
        "an entity as the object of a move": SentencePlan(
            plans["went"].subject,
            Predication(
                PLACE_EVENT,
                "PLACEEVENTTYPE.1",
                True,
                the("INSTANCE.2.1.1", "CATEGORY.2.1"),
                "SCENE.1.EVENTINSTANCE.2",
                "past",
                "simple",
            ),
        ),
        "a pair state without a time point": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1"),
            Predication(PAIR_STATE, "PAIRFL.3", True, the("PLACE.1.1", "PLACECATEGORY.1")),
        ),
    }
    for what, plan in bad.items():
        with pytest.raises(GrammarError):
            check_plan(plan)
        with pytest.raises(GrammarError):
            realizer.realize(plan, rng)
        print(what)


# ---------------------------------------------------------------------------------------------
# Acceptance: the test sets take none of this stage's sentences; the statistics
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Acceptance (stage c4c): the initial states, the place poles, the descriptions and comparisons
# of places, and the blocked place events
# ---------------------------------------------------------------------------------------------


def _preceding(document, index: int, section: str | None = None):
    """The sentence before ``index`` of a section (the event sentence by default)."""
    wanted = EVENT_SECTION if section is None else section
    return next(s for s in reversed(document.sentences[:index]) if s.section == wanted)


def _mentioned_before(document, index: int) -> set[str]:
    return {p.referent for s in document.sentences[:index] for p in s.plan.noun_phrases()}


@pytest.mark.parametrize("name", WORLDS)
def test_initial_states_state_pairs_places_and_same(cases, runs, name) -> None:
    """An initial-state sentence can state the participant's place on a location fluent, its
    pair with an entity already mentioned, or ``SAME`` with one (``WORLD_PHASE_C.md``, "Initial
    states"; Jon's ruling 1 of WM.73): a pair state at the time point before the participant's
    first reported event, true by the oracle from the replayed pairs and places; a positive
    location names the participant's own place, a negative one a place already mentioned; the
    second argument of a pair or of ``SAME`` is an entity already mentioned."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    world = case.world
    scenes = scenes_json(corpus)
    found: Counter = Counter()
    for document in corpus.documents:
        for index, sentence in enumerate(document.sentences):
            proposition = sentence.proposition
            if sentence.section != INITIAL_STATE or proposition.predicate.kind != PAIR_STATE:
                continue
            assert proposition.level == STATE
            scene = scenes[proposition.scene]
            assert oracle.state(sentence.logical_form, scene) is True
            label, second = proposition.predicate.label, proposition.predicate.patient
            time = int(proposition.time.split(".")[1])
            before = _mentioned_before(document, index)
            if world.is_location_fluent(label):
                assert world.is_place(second) and second in scene["places"]
                if proposition.polarity:
                    assert oracle.place_at(scene, time, proposition.subject, label) == second
                    found["place"] += 1
                else:
                    assert oracle.place_at(scene, time, proposition.subject, label) != second
                    assert second in before
                    found["not at a place"] += 1
            else:
                assert not world.is_place(second) and second in before
                assert second in scene["participants"]
                found["same" if label.startswith("SAME.") else "pair"] += 1
            assert sentence.readings == ("state",)
            # the plan: the copula, the preposition, and the second argument as the object
            predication = sentence.plan.predication
            assert not predication.become and predication.object is not None
            assert predication.object.referent == second
    assert found["place"] > 0 and found["same"] > 0, found
    if world.pair_fluents_read() != tuple(world.location_fluents):
        assert found["pair"] > 0, found  # a pair fluent between entities has a preposition
    # a document states them at the rate, and none with the rate at 0
    silent = runs(name, documents={"initial_state_rate": 0.0})
    assert not any(s.section == INITIAL_STATE for d in silent.documents for s in d.sentences)


@pytest.mark.parametrize("name", WORLDS)
def test_place_poles_tell_places_apart(cases, runs, name) -> None:
    """A place scalar's poles tell a place from the other places of the document's scenes, as
    entity poles do, relative to the noun's category of places (Jon's ruling 2 of WM.73): every
    definite place mention is distinguished where the place features and poles allow, and the
    tiny corpus, whose two places of a category share every place property, has distinguished
    definite place mentions by their place scalar (ruling 4)."""
    corpus = runs(name, **MANY)
    world = cases(name).world
    rules = corpus.planner.rules
    truth = corpus.planner.truth
    poles = distinguished = undistinguished = 0
    for document in corpus.documents:
        places = tuple(dict.fromkeys(q for scene in document.scenes for q in (scene.places or ())))
        for sentence in document.sentences:
            for phrase, flag in zip(sentence.sentence.phrases, sentence.distinguished, strict=True):
                if not phrase.place or phrase.determiner != "the":
                    continue
                place, noun = phrase.referent, phrase.noun
                row = world.place_index[place]
                for literal in phrase.restriction:
                    if literal.pole:
                        poles += 1
                        assert world.is_place_pole(literal.feature)
                        # the pole holds of the place against the noun's category, by the
                        # runtime's comparison classes
                        assert truth.place_pole_mask(literal.feature, noun)[0][row]
                if flag:
                    distinguished += 1
                    continue
                undistinguished += 1
                # not distinguished: no attribute of the preference order, over the limits of
                # the mention settings, tells the place from the other places the noun fits,
                # apart from what the sentence itself states (a description's feature, or the
                # poles of the scalar a comparison of places is on)
                predicate = sentence.proposition.predicate
                avoided = {literal.feature for literal in phrase.restriction} | {predicate.label}
                if predicate.kind == COMPARATIVE:
                    avoided |= {f"{predicate.label}.HIGH", f"{predicate.label}.LOW"}
                others = [q for q in rules.matches(phrase, places) if q != place]
                assert others, "an undistinguished mention has another fitting place"
                for attribute in rules.place_preference:
                    literal = rules.value(place, attribute, noun)
                    if literal is None or literal.feature in avoided:
                        continue
                    assert all(rules.fits(world.place_index[q], literal, noun) for q in others), (
                        phrase,
                        attribute,
                    )
    assert distinguished > 0 and poles > 0, (distinguished, poles)
    stats = corpus.stats["mentions"]
    assert stats["place_definite_not_distinguished"] == undistinguished
    assert stats["place_definite"] == distinguished + undistinguished


@pytest.mark.parametrize("name", WORLDS)
def test_places_are_described(cases, runs, name) -> None:
    """A narrative describes a place that an event sentence mentioned, at
    ``documents.place_description_rate`` (Jon's ruling 3 of WM.73): an instance-level sentence
    with the place as its subject and a place property or part, membership in a category of
    places, or a place pole as its predicate, true by the oracle from the places' table; the
    predicate's feature is no modifier of the subject; and none with the rate at 0."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    world = case.world
    kinds: Counter = Counter()
    for document in corpus.documents:
        for index, sentence in enumerate(document.sentences):
            if sentence.section != PLACE_DESCRIPTION:
                continue
            proposition = sentence.proposition
            assert proposition.level == INSTANCE and world.is_place(proposition.subject)
            kind, label = proposition.predicate.kind, proposition.predicate.label
            assert kind in (IS, HAS, MEMBER, COMPARISON)
            assert label.startswith(
                ("PLACEPROPERTY.", "PLACEPART.", "PLACECATEGORY.", "PLACESCALARDIM.")
            )
            assert oracle.truth(sentence.logical_form) is True
            assert corpus.planner.truth.is_true(proposition)
            subject = sentence.plan.subject
            assert subject.place and subject.determiner == "the" and subject.clause is None
            assert label not in {literal.feature for literal in subject.restriction}
            if kind == MEMBER:
                assert subject.noun != label
            if kind == COMPARISON:
                assert proposition.predicate.comparison == subject.noun
            # after the event sentence that mentioned the place
            before = _preceding(document, index)
            assert proposition.subject in {p.referent for p in before.plan.noun_phrases()}
            kinds[kind] += 1
    assert kinds[IS] > 0 and kinds[COMPARISON] > 0, kinds
    silent = runs(name, documents={"place_description_rate": 0.0})
    assert not any(s.section == PLACE_DESCRIPTION for d in silent.documents for s in d.sentences)


@pytest.mark.parametrize("name", WORLDS)
def test_places_are_compared(cases, runs, name) -> None:
    """A sentence that introduces a second place of the scene can be followed, at
    ``documents.comparative_rate``, by a comparison of the two places on a place scalar
    (``WORLD_PHASE_C.md``, "Comparative descriptions"): the new place is the subject, the other
    was mentioned before, both are places, the comparative is true by the oracle, and none is
    made with the rate at 0."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    world = case.world
    found = 0
    for document in corpus.documents:
        for index, sentence in enumerate(document.sentences):
            if sentence.section != PLACE_COMPARATIVE:
                continue
            proposition = sentence.proposition
            assert proposition.level == INSTANCE and proposition.predicate.kind == COMPARATIVE
            assert proposition.predicate.label in world.place_scalars
            plan = sentence.plan
            subject, target = plan.subject, plan.predication.object
            assert subject.place and target is not None and target.place
            assert oracle.truth(sentence.logical_form) is True
            # the subject was first mentioned by the sentence before, the object earlier
            previous = document.sentences[index - 1]
            assert subject.referent in {p.referent for p in previous.plan.noun_phrases()}
            assert subject.referent not in _mentioned_before(document, index - 1)
            assert target.referent in _mentioned_before(document, index)
            assert previous.section in (EVENT_SECTION, INITIAL_STATE, RESULT)
            found += 1
    assert found > 0
    silent = runs(name, documents={"comparative_rate": 0.0})
    assert not any(s.section == PLACE_COMPARATIVE for d in silent.documents for s in d.sentences)


@pytest.mark.parametrize("name", WORLDS)
def test_blocked_sentences_cover_place_events_and_skip_a_move_to_the_agents_own_place(
    cases, runs, name
) -> None:
    """A blocked sentence can say what a participant could not do, or could do and did not do,
    at a place (``WORLD_PHASE_C.md``, "Blocked events"): a move or an act with a place of the
    scene as its object, true by the oracle; and no blocked sentence says that an agent could
    not go to the place it was at (WM.53, review point 5)."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    world = case.world
    definition = world.definition
    scenes = scenes_json(corpus)
    kinds: Counter = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            if sentence.section != BLOCKED or proposition.predicate.kind != PLACE_EVENT:
                continue
            label, place = proposition.predicate.label, proposition.predicate.patient
            scene = scenes[proposition.scene]
            assert world.is_place(place) and place in scene["places"]
            assert oracle.able_now(sentence.logical_form, scene) is True
            time = int(proposition.time.split(".")[1])
            kind = world.place_event_kind(label)
            kinds[kind, proposition.polarity] += 1
            if kind == "move":
                location = world.move_location(label)
                if not proposition.polarity:
                    assert oracle.place_at(scene, time, proposition.subject, location) != place
            assert sentence.plan.predication.object is not None
            assert sentence.plan.predication.object.place
    assert sum(kinds.values()) > 0, kinds
    # a negative sentence about a move needs a move with a literal beyond its compiled one
    # (the compiled literal alone blocks a move to the agent's own place, which is skipped);
    # the default run, with its many blocked sentences, holds every kind
    blockable_moves = [
        label
        for label in world.place_event_types
        if world.place_event_kind(label) == "move"
        and any(
            getattr(lit, "origin", None) != "move"
            for lit in definition.event_type(label).precondition
        )
    ]
    if name == "default":
        assert blockable_moves and kinds["move", False] > 0, kinds
        assert kinds["move", True] > 0 and kinds["act", False] > 0, kinds


# ---------------------------------------------------------------------------------------------
# Acceptance (stage c4c): the test sets
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_test_sets_hold_pair_location_same_and_place_event_items(cases, runs, name) -> None:
    """The state sets hold pair, location, and ``SAME`` items at the scene's final time point,
    with the changes predicate (another pair fluent of the same kind), subject (another
    participant), role (the arguments exchanged, for a pair fluent between entities that is not
    symmetric alone), and object (another referent, or another place, of the same kind); the
    event and ``able_now`` sets hold events of place event types, with the object change in place
    of the role change (WM.56); an item mentions places the document mentioned; and the
    statistics count the items by kind."""
    corpus = runs(name)
    world = cases(name).world
    documents = {d.label: d for d in corpus.documents}
    state_kinds: Counter = Counter()
    event_kinds: Counter = Counter()
    for test_set in corpus.test_sets:
        for true, false in test_set.pairs:
            a, b = true.proposition, false.proposition
            document = documents[true.input["document"]] if true.input["document"] else None
            for item in (true, false):
                for mention in item.input["referents"]:
                    referent = mention["referent"]
                    if world.is_place(referent):
                        assert document is not None and referent in document.referents.values()
            if test_set.level == STATE and a.predicate.kind == PAIR_STATE:
                assert document is not None
                label = a.predicate.label
                kind = (
                    "same"
                    if label.startswith("SAME.")
                    else "location"
                    if world.is_location_fluent(label)
                    else "pair"
                )
                state_kinds[kind, test_set.change] += 1
                assert world.is_place(a.predicate.patient) == (kind == "location")
                if test_set.change == "predicate":
                    assert b.predicate.patient == a.predicate.patient
                    other = b.predicate.label
                    assert other.startswith("SAME.") == (kind == "same")
                    assert world.is_location_fluent(world.pair_fluent_of(other)) == (
                        kind == "location"
                    )
                elif test_set.change == "role":
                    assert kind == "pair" and not world.pair_fluent_info[label].symmetric
                    assert (b.subject, b.predicate.patient) == (a.predicate.patient, a.subject)
                elif test_set.change == "object":
                    assert b.subject == a.subject and b.predicate.label == label
                    assert world.is_place(b.predicate.patient) == world.is_place(
                        a.predicate.patient
                    )
                    assert b.predicate.patient != a.predicate.patient
                else:
                    assert test_set.change == "subject" and b.predicate == a.predicate
            if test_set.level in (EVENT, ABLE_NOW) and a.predicate.kind == PLACE_EVENT:
                event_kinds[test_set.level, test_set.change, test_set.kind] += 1
                assert test_set.change != "role"
                assert world.is_place(a.predicate.patient) and world.is_place(b.predicate.patient)
                if test_set.change == "object":
                    assert b.predicate.patient != a.predicate.patient
                    assert b.predicate.label == a.predicate.label and b.subject == a.subject
                elif test_set.change == "predicate":
                    assert (
                        b.predicate.kind == PLACE_EVENT and b.predicate.label != a.predicate.label
                    )
        if test_set.level == STATE:
            stats = test_set.stats()["pairs_by_item_kind"]
            assert sum(stats.values()) == len(test_set.pairs)
            assert stats["location"] + stats["pair"] + stats["same"] == sum(
                true.proposition.predicate.kind == PAIR_STATE for true, _ in test_set.pairs
            )
    assert state_kinds["location", "object"] > 0 and state_kinds["same", "subject"] > 0
    assert state_kinds["location", "subject"] > 0 and state_kinds["same", "object"] > 0
    # a world with one location fluent has no predicate swap of a location
    if len(world.location_fluents) == 1:
        assert state_kinds["location", "predicate"] == 0
    assert event_kinds[EVENT, "object", "possible"] > 0
    assert any(level == ABLE_NOW for level, _, _ in event_kinds), event_kinds
    assert any(change == "subject" for level, change, _ in event_kinds if level == EVENT)
    # the role swap of a pair state: a pair fluent between entities that is not symmetric
    swappable = [
        f
        for f in world.pair_fluents_read()
        if not world.is_location_fluent(f) and not world.pair_fluent_info[f].symmetric
    ]
    if not swappable:
        assert state_kinds["pair", "role"] == 0
    if name == "default":
        assert state_kinds["pair", "object"] > 0 and state_kinds["pair", "subject"] > 0
        assert event_kinds[EVENT, "object", "impossible"] > 0
        assert event_kinds[EVENT, "object", "blocked"] > 0


@pytest.mark.parametrize("name", WORLDS)
def test_the_statistics_count_the_new_sentences_and_the_place_events(cases, runs, name) -> None:
    """``stats.yaml`` counts the pair-state sentences, the change sentences with ``become``, the
    location sentences, and the narrated moves and acts (Jon's ruling 3 of WM.72), and the
    scenes' place events and pair changes, a move's two changes counted as one move; the counts
    agree with a recount from the documents and the scenes."""
    corpus = runs(name)
    world = cases(name).world
    states = corpus.stats["states"]
    sentences = [s for d in corpus.documents for s in d.sentences]
    pairs = [s for s in sentences if s.proposition.predicate.kind == PAIR_STATE]
    assert states["pair_state_sentences"] == sum(s.proposition.level == STATE for s in pairs)
    assert states["pair_change_sentences"] == sum(s.proposition.level == CHANGE for s in pairs)
    assert states["location_sentences"] == sum(
        world.is_location_fluent(s.proposition.predicate.label) for s in pairs
    )
    # stage c4c: the SAME states, the blocked place events, the place descriptions, and the
    # comparisons of places
    assert states["same_state_sentences"] == sum(
        s.proposition.level == STATE and s.proposition.predicate.label.startswith("SAME.")
        for s in pairs
    )
    blocked_place = [
        s for s in sentences if s.section == BLOCKED and s.proposition.predicate.kind == PLACE_EVENT
    ]
    assert states["blocked_place_sentences"] == len(blocked_place)
    assert states["blocked_move_sentences"] == sum(
        world.place_event_kind(s.proposition.predicate.label) == "move" for s in blocked_place
    )
    assert states["place_description_sentences"] == sum(
        s.section == PLACE_DESCRIPTION for s in sentences
    )
    assert corpus.stats["comparisons"]["place_comparative_sentences"] == sum(
        s.section == PLACE_COMPARATIVE for s in sentences
    )
    place_events = [
        s
        for s in sentences
        if s.section == EVENT_SECTION and s.proposition.predicate.kind == PLACE_EVENT
    ]
    kinds = Counter(world.place_event_kind(s.proposition.predicate.label) for s in place_events)
    assert states["move_sentences"] == kinds["move"]
    assert states["act_sentences"] == kinds["act"]
    scenes = corpus.stats["scenes"]
    events = [e for scene in corpus.planner.scenes for step in scene.steps for e in step.events]
    moves = sum(
        world.place_event_kind(e.type) == "move"
        for e in events
        if world.is_place_event_type(e.type)
    )
    acts = sum(
        world.place_event_kind(e.type) == "act" for e in events if world.is_place_event_type(e.type)
    )
    between = sum(
        1
        for e in events
        for c in e.changes
        if c.pair and not world.is_location_fluent(c.pair_fluent)
    )
    assert scenes["place_events"] == moves + acts
    assert scenes["moves"] == moves and scenes["acts"] == acts
    assert scenes["pair_changes"] == between + moves
    assert scenes["pair_changes_between_entities"] == between
    assert scenes["move_share"] == round(moves / len(events), 6)
    assert scenes["pair_changes_per_event"] == round((between + moves) / len(events), 6)


# ---------------------------------------------------------------------------------------------
# Acceptance: no causal statement states a pair literal or a pair condition
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_no_causal_statement_states_a_pair_literal(cases, runs, name) -> None:
    """Every causal statement is about a base or a numeric fluent, none about a pair fluent;
    no event term's condition holds a pair literal; and the conditional effects with a pair
    condition are counted among the unstated."""
    corpus = runs(name)
    facts = corpus.planner.facts
    statements = facts.causal_statements()
    assert statements
    for statement in statements:
        assert isinstance(statement.subject, EventTerm)
        assert not _carries(statement.predicate.label)
        assert not _carries(statement.subject.event_type)
        assert not any(_carries(c.label) for c in statement.subject.condition)
    report = facts.causal_report
    assert {"pair_condition", "pair_effect"} <= set(report["unstated"])
    definition = cases(name).world.definition
    pair_conditions = sum(
        1
        for label in corpus.planner.world.binary_leaves + corpus.planner.world.unary
        for effect in definition.event_type(label).effects
        if effect.condition and not effect.pair and any(lit.pair for lit in effect.condition)
    )
    assert report["unstated"]["pair_condition"] == pair_conditions


def test_a_conditional_effect_with_a_pair_condition_is_stated_in_neither_form(
    tmp_path: Path,
) -> None:
    """WM.59 (review point 11): on a tiny world where every effect is conditional and every
    condition literal is a pair literal, no conditional statement is made for such an effect,
    and its unconditional form is false by the corpus's truth test and by the oracle's
    re-reading of the definition."""
    data = yaml.safe_load((REPO / TINY_WORLD).read_text(encoding="utf-8"))
    data["event_types"] = {
        **data.get("event_types", {}),
        "effects": {
            "conditional_share": 1.0,
            "condition_kinds": {"feature": 0, "fluent": 0, "numeric": 0, "pair": 1},
        },
    }
    world_path = tmp_path / "pair_conditions.yaml"
    world_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = corpus_config(str(world_path))
    world = load_world(config)
    facts = Facts(config, world)
    truth = facts.truth
    definition = world.definition
    folder = world.result.write(tmp_path / "run")
    from truth_oracle import Oracle

    oracle = Oracle(folder, z=config.scalar_z)
    with_pair_condition = 0
    for label in world.binary_leaves + world.unary:
        record = definition.event_type(label)
        for effect in record.effects:
            if effect.pair or not any(lit.pair for lit in effect.condition):
                continue
            with_pair_condition += 1
            assert condition_of(effect, truth.cut_points) is None
            value = None if effect.numeric else bool(effect.value)
            direction = (
                None if not effect.numeric else ("raises" if effect.amount >= 0 else "lowers")
            )
            # the unconditional form is no entry of this effect
            assert not effect_matches(
                effect, effect.role, effect.fluent, value, direction, (), truth.cut_points
            )
            # and the statement holds only if another, unconditional, effect carries it
            others = [
                e
                for e in record.effects
                if e is not effect
                and not e.pair
                and e.role == effect.role
                and e.fluent == effect.fluent
                and not e.condition
            ]
            unconditional = truth.has_entry(
                label, EFFECT, effect.role, effect.fluent, value, direction, ()
            )
            assert unconditional == bool(others), (label, effect.text)
            form = {
                "level": CLASS,
                "quantifier": "nec_all",
                "polarity": True,
                "subject": {"head": "THING", "event": label, "role": effect.role},
                "predicate": {
                    "kind": EFFECT,
                    "label": effect.fluent,
                    **({"value": value} if direction is None else {"direction": direction}),
                },
            }
            if effect.fluent in world.base_fluents or effect.fluent in world.numeric_fluents:
                assert oracle.causal(form) is unconditional, (label, effect.text)
    assert with_pair_condition > 0
    # no statement of the world carries a condition: every condition holds a pair literal
    for statement in facts.causal_statements():
        assert not statement.subject.condition
    assert facts.causal_report["unstated"]["pair_condition"] == with_pair_condition
