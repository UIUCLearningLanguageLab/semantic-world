"""Jon's rulings WM.76 (November 5, 2026) on the open questions of stage c5b
(``docs/proposals/2026-11-04-world-stage-c5b-decisions.md``), the two that change the corpus,
numbered by the question they answer:

- **Ruling 1 (question 1).** A document about a category of entities states capacity facts
  about places with its topic as the agent ("owls can go to deserts"). The facts join the pool
  of the topic's relation facts on the agent side, with the facts about the two-place event
  types, and are drawn the same way. Each is true by the oracle, takes the extensional
  quantifiers alone, and every false item made from one is false. ``stats.yaml`` counts the
  relation facts of the documents about categories of entities by kind.
- **Ruling 3 (question 3).** No initial-state sentence and no state item, true or false, names
  a pair fluent that can never hold anywhere in the world: one with no initial pair that no
  effect adds, read from ``definition.json``. The pair fluent keeps its preposition and its
  causal statements, and a pair fluent that can hold keeps its sentences and items, the
  negative ones among them.

The tests enumerate what the world gives wherever they can, and read a run only for what must
hold of every sentence.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
from corpus_support import REPO, Case

from semantic_world.corpus.planner import INITIAL_STATE, RELATION, Planner
from semantic_world.corpus.propositions import (
    NEC_QUANTIFIERS,
    PAIR_STATE,
    PLACE_EVENT,
    STATE,
    VERB,
    CategoryTerm,
    Predicate,
    Proposition,
)
from semantic_world.corpus.testsets import (
    PREDICATE,
    QUANTIFIER,
    ROLE,
    SUBJECT,
    candidates,
    changes_for,
    false_items,
)

WORLDS = ("tiny", pytest.param("default", marks=pytest.mark.slow))
HAND_FILES = {
    "tiny_pairs": "data/world/tiny_pairs.yaml",
    "tiny_place_chain": "data/world/tiny_place_chain.yaml",
}
EVERY_WORLD = ("tiny", pytest.param("default", marks=pytest.mark.slow), *HAND_FILES)
ENCYCLOPEDIC = {
    "documents": {
        "mix": {
            "encyclopedic_category": 0.45,
            "encyclopedic_feature": 0.45,
            "entity": 0.05,
            "situational": 0.05,
        },
        "shuffle": 0.0,
    },
}
"""Mostly encyclopedic documents, as the tests of stage c5b use them (the run is shared)."""
ENCYCLOPEDIC_COUNT = {"tiny": 200, "default": 600}
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
"""Many state and result sentences, as the tests of stage c4c use them (the run is shared)."""
NEGATIVE_STATES = {
    "documents": {"initial_state_rate": 1.0},
    "propositions": {"negation_rate": {"state": 0.9}},
}
"""An initial state after every first mention, nine in ten of them negative."""


@pytest.fixture(scope="module")
def worlds(cases, tmp_path_factory: pytest.TempPathFactory):
    """The worlds of this file's tests, by name: the ``tiny`` and ``default`` cases, and the two
    worlds of the hand-written event files, each made once with its world run folder."""
    root = tmp_path_factory.mktemp("wm76")
    made: dict[str, Case] = {}

    def world(name: str) -> Case:
        if name not in HAND_FILES:
            return cases(name)
        if name not in made:
            made[name] = Case(name, str(REPO / HAND_FILES[name]), root / name)
        return made[name]

    return world


def form_of(proposition: Proposition) -> dict:
    """A proposition's JSON form, as the files hold it."""
    return json.loads(json.dumps(proposition.to_json()))


def scenes_json(corpus) -> dict[str, dict]:
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


def never_holding(case: Case) -> set[str]:
    """The pair fluents of a world that can never hold, by a re-reading of ``definition.json``:
    not total (a location fluent always has its pairs), no initial pair (an initial rate of 0),
    and no effect of any event type adds a pair."""
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    symbols = {s["label"]: s for s in record["symbols"] if s["kind"] == "pair_fluent"}
    added = {
        effect["pair_fluent"]
        for entry in record["event_types"]
        for effect in entry["effects"]
        if effect.get("kind") == "pair" and effect["value"]
    }
    return {
        label
        for label, symbol in symbols.items()
        if not symbol.get("total") and not symbol["initial_rate"] and label not in added
    }


# ---------------------------------------------------------------------------------------------
# Ruling 1: capacity facts about places in documents about a category of entities
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_entity_topics_draw_capacity_facts_about_places_as_agents(cases, name) -> None:
    """The relation facts of a category of entities on the agent side are one pool: the facts
    about the two-place event types, as before, and the capacity facts about every category of
    places with the topic as the agent. The patient side is unchanged, and a category of places
    is still the agent of nothing. Every capacity fact of the pool, in both polarities, is true
    by the oracle and takes an extensional quantifier; with the facts about the two-place
    event types already stated, the draw of a relation fact gives a capacity fact about a
    place; and every false item made from one is false by the oracle and stays a statement
    about an entity category and a category of places."""
    case = cases(name)
    planner = Planner(case.config(), case.world)
    facts, truth, oracle = planner.facts, planner.truth, case.oracle()
    assert facts.place_verbs and facts.place_categories
    topics = facts.categories if name == "tiny" else facts.categories[:10]
    for topic in topics:
        term = CategoryTerm(topic)
        assert planner._has_relations(topic)
        agent_side = planner._relation_candidates(term, True)
        events = [
            (term, Predicate(VERB, verb, CategoryTerm(other)))
            for verb in facts.verbs
            for other in facts.categories
        ]
        places = [
            (term, Predicate(PLACE_EVENT, verb, CategoryTerm(place)))
            for verb in facts.place_verbs
            for place in facts.place_categories
        ]
        assert agent_side == events + places
        patient_side = planner._relation_candidates(term, False)
        assert all(predicate.kind == VERB for _, predicate in patient_side)
        assert all(predicate.patient == term for _, predicate in patient_side)
        available = set(planner._available_facts(topic)[RELATION])
        for negative in (False, True):
            stated_events = set()
            for subject, predicate in events:
                fact = facts.class_fact(subject, predicate, negative)
                if fact is not None:
                    stated_events.add(planner._key(fact))
            place_facts = []
            for subject, predicate in places:
                fact = facts.class_fact(subject, predicate, negative)
                if fact is None:
                    continue
                assert fact.subject == term and fact.negative == negative
                assert fact.quantifier not in NEC_QUANTIFIERS
                assert truth.is_place_term(fact.predicate.patient)
                assert oracle.truth(form_of(fact)) is True, fact
                if planner._fits(fact):
                    assert planner._key(fact) in available
                    place_facts.append(fact)
                # every false item made from the fact is false, and keeps the kinds of thing
                assert ROLE not in changes_for(fact, facts)
                assert candidates(facts, fact, ROLE) == []
                for change in (PREDICATE, SUBJECT, QUANTIFIER):
                    for false in false_items(facts, fact, change):
                        assert oracle.truth(form_of(false)) is False, false
                        assert truth.is_true(false) is False
                        assert false.predicate.kind == PLACE_EVENT
                        assert false.subject.category in facts.categories
                        assert truth.is_place_term(false.predicate.patient)
            # the pool is drawn as one: when the facts about the two-place event types are
            # all stated, the draw on the agent side gives a capacity fact about a place
            if place_facts:
                rng = np.random.default_rng(0)
                drawn = planner._first_fact(rng, agent_side, negative, stated_events)
                assert drawn is not None and drawn.predicate.kind == PLACE_EVENT
                assert drawn.subject == term
    # a category of places is still the agent of nothing, and the patient of the same facts
    for place in facts.place_categories:
        assert planner._relation_candidates(CategoryTerm(place), True) == []
        assert all(
            predicate.kind == PLACE_EVENT and predicate.patient == CategoryTerm(place)
            for _, predicate in planner._relation_candidates(CategoryTerm(place), False)
        )


@pytest.mark.parametrize("name", WORLDS)
def test_the_statistics_count_the_relation_facts_of_entity_documents_by_kind(
    cases, runs, name
) -> None:
    """``documents.entity_category_relation_facts`` counts the relation facts of the documents
    about a category of entities by kind, a two-place event type or a place event type, as a
    tally of the documents' logical forms gives them, with the share of the place facts."""
    corpus = runs(name, ENCYCLOPEDIC_COUNT[name], **ENCYCLOPEDIC)
    tally: Counter = Counter()
    for document in corpus.documents:
        if document.type != "encyclopedic_category" or not document.topic.startswith("CATEGORY."):
            continue
        for sentence in document.sentences:
            form = sentence.logical_form
            if form["level"] == "class":
                tally[form["predicate"]["kind"]] += 1
    block = corpus.stats["documents"]["entity_category_relation_facts"]
    assert block["event_type"] == tally["event_type2"]
    assert block["place_event_type"] == tally["place_event_type"]
    total = tally["event_type2"] + tally["place_event_type"]
    if total:
        assert block["place_share"] == pytest.approx(tally["place_event_type"] / total, abs=1e-6)
    else:
        assert block["place_share"] is None


# ---------------------------------------------------------------------------------------------
# Ruling 3: no pair state about a pair fluent that can never hold
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", EVERY_WORLD)
def test_pair_states_leave_out_the_pair_fluents_that_can_never_hold(worlds, name) -> None:
    """The pair fluents a pair state can be about are the pair fluents between entities with a
    preposition of which a pair can ever hold, by a re-reading of ``definition.json``. The
    predicates of an initial state and of a state item, and the predicate swap of a pair state,
    name no other. The lexicon is unchanged: a pair fluent that can never hold keeps its
    preposition, and its causal statements are still made and true by the oracle. In a world
    whose pair fluents can all hold, nothing changes."""
    case = worlds(name)
    facts, world, oracle = case.facts(), case.world, case.oracle()
    never = never_holding(case)
    silent = [f for f in facts.entity_pair_prepositions if f in never]
    stated = tuple(f for f in facts.entity_pair_prepositions if f not in never)
    if name in ("tiny", "default"):
        # the worlds the ruling was made on: each has such a pair fluent with a preposition
        assert silent, "the world has no pair fluent with a preposition that can never hold"
    assert facts.state_pair_prepositions == stated
    for fluent in facts.entity_pair_prepositions:
        assert facts.truth.can_hold(fluent) == (fluent not in never)
    # the predicates of a pair state: the pair fluents that can hold, the location fluents,
    # and SAME, in that order
    others = world.instances[:3]
    places = tuple(world.places[:2])
    expected = [Predicate(PAIR_STATE, fluent, other) for fluent in stated for other in others]
    expected += [
        Predicate(PAIR_STATE, fluent, place)
        for fluent in facts.location_prepositions
        for place in places
    ]
    expected += [
        Predicate(PAIR_STATE, same, other) for same in facts.same_prepositions for other in others
    ]
    assert facts.pair_predicates(others, places) == expected
    # the predicate swap of a pair state between entities brings in the other pair fluents
    # that can hold, and no other
    first, second = world.instances[:2]
    for fluent in stated:
        state = Proposition(STATE, first, Predicate(PAIR_STATE, fluent, second))
        swapped = [c.predicate.label for c in candidates(facts, state, PREDICATE)]
        assert swapped == [other for other in stated if other != fluent]
    # the lexicon and the causal statements are unchanged
    for fluent in silent:
        assert fluent in facts.pair_prepositions and fluent in facts.named
        statements = facts.causal_statements(pair_fluent=fluent)
        assert statements, f"{fluent} has a preposition and no causal statement"
        for statement in statements:
            assert oracle.truth(form_of(statement)) is True


@pytest.mark.parametrize("name", WORLDS)
def test_no_sentence_and_no_state_item_names_a_pair_fluent_that_never_holds(
    cases, runs, name
) -> None:
    """No sentence of a document states a pair of a pair fluent that can never hold, at a time
    point or as a change, and no state item, true or false, names one. A pair fluent between
    entities that can hold keeps its initial-state sentences, the negative ones among them,
    each true by the oracle from the replayed pairs."""
    case = cases(name)
    world = case.world
    never = never_holding(case)
    for settings in (MANY, {}, NEGATIVE_STATES):
        corpus = runs(name, **settings)
        for document in corpus.documents:
            for sentence in document.sentences:
                predicate = sentence.proposition.predicate
                if predicate.kind == PAIR_STATE:
                    assert world.pair_fluent_of(predicate.label) not in never, sentence.label
        for test_set in corpus.test_sets:
            for pair in test_set.pairs:
                for item in pair:
                    predicate = item.proposition.predicate
                    if predicate.kind == PAIR_STATE:
                        assert world.pair_fluent_of(predicate.label) not in never, test_set.name
    # the pair fluents that can hold keep their negative sentences
    corpus = runs(name, **NEGATIVE_STATES)
    oracle = case.oracle(**NEGATIVE_STATES)
    facts = corpus.planner.facts
    scenes = scenes_json(corpus)
    found: Counter = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            label = proposition.predicate.label
            if sentence.section != INITIAL_STATE or proposition.predicate.kind != PAIR_STATE:
                continue
            if label not in facts.state_pair_prepositions:
                continue  # a location or SAME
            assert oracle.state(sentence.logical_form, scenes[proposition.scene]) is True
            found[proposition.polarity] += 1
    if facts.state_pair_prepositions:
        assert found[False] > 0, found


def test_the_corpus_of_a_world_whose_pair_fluents_all_hold_is_unchanged(worlds) -> None:
    """In the hand world ``tiny_pairs`` every pair fluent between entities can hold, so the
    ruling leaves nothing out: the pair fluents of the pair states are the pair fluents between
    entities with a preposition."""
    case = worlds("tiny_pairs")
    facts = case.facts()
    assert not never_holding(case) & set(facts.entity_pair_prepositions)
    assert facts.state_pair_prepositions == facts.entity_pair_prepositions
