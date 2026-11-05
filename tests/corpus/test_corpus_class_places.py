"""Stage c5b of the world-and-language refactor (``docs/specs/WORLD_PHASE_C.md``, "c5b.
Documents, test sets, and the end of the phase"): the class level's documents and test sets
about pair fluents and places, under Jon's rulings WM.75.

- **Places at the class level.** A category of places is a category term the truth tests
  accept: its place properties and parts, the place patient capacities, its place poles, its
  comparisons with a sibling, and its membership, with ``NEC`` by the fixed test of the place
  taxonomy. A relation fact about a place event type has a category of entities as its agent
  and a category of places as its patient. Every such proposition, under every quantifier and
  polarity, is judged alike by the corpus's truth test and by the independent oracle, which
  re-reads ``definition.json`` and the place taxonomy's run.
- **Documents.** The categories of places are topics of category documents; the place
  properties and parts, the place event types, and the pair fluents are topics of feature
  documents. A document about a pair fluent holds its profile statement, then its causal
  statements.
- **Test sets.** The class sets hold facts about categories of places; the profile sets pair
  every profile statement with every valid false item, and the ``observed`` mark of each
  agrees with the scenes; the causal sets of the hand worlds hold the statements with pair
  literals, with marks that agree with the replayed pairs.
- **WM.75.** Every pair fluent the dynamics use has a preposition; a comparison that nothing
  tells apart tries the other dimensions before it is skipped.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pytest
from corpus_support import REPO, Case

from semantic_world.corpus import interpret, parse_propositional, proposition_of
from semantic_world.corpus.generate import generate
from semantic_world.corpus.grammar import INSTANCE_NP, NounPhrase
from semantic_world.corpus.planner import (
    CAUSAL,
    PARTICIPANTS,
    PROFILE_SECTION,
    RELATION,
    Planner,
)
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    HAS,
    IS,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NEC_QUANTIFIERS,
    NO,
    PLACE_EVENT,
    PROJECTION,
    QUANTIFIERS,
    SOME,
    VERB,
    CategoryTerm,
    Clause,
    Predicate,
    Proposition,
    pair_fluent_of_label,
)
from semantic_world.corpus.testsets import (
    CAUSAL_LEVELS,
    FORM,
    LAWLIKE,
    PREDICATE,
    PROFILE_LEVEL,
    ROLE,
    SUBJECT,
    false_items,
)

WORLDS = ("tiny", pytest.param("default", marks=pytest.mark.slow))
HAND_WORLDS = ("tiny_pairs", "tiny_place_chain")
EVERY_WORLD = ("tiny", *HAND_WORLDS, pytest.param("default", marks=pytest.mark.slow))
HAND_FILES = {
    "tiny_pairs": "data/world/tiny_pairs.yaml",
    "tiny_place_chain": "data/world/tiny_place_chain.yaml",
}
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
"""Mostly encyclopedic documents, in the template order, so that the tests see every kind of
topic and the order of a document about a pair fluent."""
ENCYCLOPEDIC_COUNT = {"tiny": 200, "default": 600}


@pytest.fixture(scope="module")
def worlds(cases, tmp_path_factory: pytest.TempPathFactory):
    """The worlds of this file's tests, by name: the ``tiny`` and ``default`` cases, and the two
    worlds of the hand-written event files, each made once with its world run folder."""
    root = tmp_path_factory.mktemp("class_places")
    made: dict[str, Case] = {}

    def world(name: str) -> Case:
        if name not in HAND_FILES:
            return cases(name)
        if name not in made:
            made[name] = Case(name, str(REPO / HAND_FILES[name]), root / name)
        return made[name]

    return world


def form_of(proposition: Proposition) -> dict:
    """A class-level proposition's JSON form, as the files hold it."""
    return json.loads(json.dumps(proposition.to_json()))


def agree(truth, oracle, proposition: Proposition) -> bool | None:
    """The truth test's verdict on a proposition, checked against the oracle's: None when the
    truth test finds the proposition invalid (the oracle then never calls it true)."""
    evaluation = truth.evaluate(proposition)
    verdict = oracle.truth(form_of(proposition))
    if not evaluation.valid:
        assert verdict is not True, proposition
        return None
    assert verdict is evaluation.true, (proposition, evaluation.grounding)
    return evaluation.true


def place_terms(facts, category: str) -> list[CategoryTerm]:
    """A category of places, plain and with each of its restrictions."""
    plain = CategoryTerm(category)
    return [plain] + [CategoryTerm(category, (x,)) for x in facts.restriction_options(plain)]


# ---------------------------------------------------------------------------------------------
# Acceptance: the class-level facts about places are true by a re-reading of the files
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_class_facts_about_places_agree_with_the_oracle(cases, name) -> None:
    """Every class-level proposition about a category of places that the language could
    state, plain or restricted, under every quantifier and both polarities, has the verdict of
    the oracle: the place features and capacities from the places' own table, membership from
    the place tree, the poles and the comparisons from the place scalars, and ``NEC`` from the
    oracle's own fixed test over the place taxonomy's run. Some fact is fixed, and some is
    not."""
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    truth = facts.truth
    assert facts.place_categories and facts.place_projections
    counts: Counter = Counter()
    for category in facts.place_categories:
        assert facts.is_place_category(category) and truth.is_place_term(CategoryTerm(category))
        for term in place_terms(facts, category):
            members = truth.members(term)
            mask = oracle.subject_set(json.loads(json.dumps(term.to_json())))
            assert sorted(np.flatnonzero(mask)) == sorted(int(i) for i in members)
            for predicate in facts.class_predicates(category):
                assert predicate.kind in (IS, HAS, PROJECTION, COMPARISON, MEMBER)
                if predicate.kind == COMPARISON:
                    for polarity in (True, False):
                        verdict = agree(
                            truth, oracle, Proposition(CLASS, term, predicate, polarity)
                        )
                        # a pole is about the category, never about a restricted set
                        assert (verdict is None) == bool(term.restriction)
                        counts["pole", verdict] += 1
                    continue
                for quantifier in QUANTIFIERS:
                    for polarity in (True, False):
                        proposition = Proposition(CLASS, term, predicate, polarity, quantifier)
                        verdict = agree(truth, oracle, proposition)
                        counts[predicate.kind, quantifier, verdict] += 1
                        if verdict and quantifier in NEC_QUANTIFIERS:
                            # the fixed test of the place taxonomy, and the oracle's
                            if predicate.kind != MEMBER:
                                fixed, _ = truth.fixed(term, predicate.label)
                                wanted = 1 if quantifier == NEC_ALL else 0
                                assert fixed == wanted
                                form = json.loads(json.dumps(term.to_json()))
                                assert oracle.fixed(form, predicate.label) == wanted
        # the strongest true facts the documents state are among the verdicts above
        for negative in (False, True):
            for fact in facts.class_facts(category, negative):
                assert truth.is_true(fact) and oracle.truth(form_of(fact)) is True
                counts["stated", fact.predicate.kind] += 1
    # the comparison facts between sibling categories of places
    for category in facts.place_categories:
        for partner in facts.comparison_partners(category):
            assert facts.is_place_category(partner)
            assert facts.world.place_category[partner].parent == (
                facts.world.place_category[category].parent
            )
            for scalar in facts.place_scalars:
                fact = facts.comparison_fact(category, partner, scalar)
                if fact is None:
                    continue
                assert agree(truth, oracle, fact) is True
                swapped = Proposition(
                    CLASS,
                    fact.predicate.patient,
                    dataclasses.replace(fact.predicate, patient=fact.subject),
                )
                assert agree(truth, oracle, swapped) is False
                counts["comparison fact"] += 1
    # a patient capacity of a place event type takes the extensional quantifiers alone
    assert not any(counts[PROJECTION, q, True] for q in NEC_QUANTIFIERS)
    assert any(counts[PROJECTION, q, True] for q in (ALL, MOST, SOME, NO))
    # membership in a category of places is fixed by the place tree
    assert counts[MEMBER, NEC_NO, True] > 0
    assert counts["stated", PROJECTION] > 0 and counts["stated", MEMBER] > 0
    if name == "default":
        # the default place taxonomy has two levels, defining features, and two scalars
        assert counts[MEMBER, NEC_ALL, True] > 0 and counts["comparison fact"] > 0
        assert counts[IS, NEC_ALL, True] + counts[IS, NEC_NO, True] > 0
        assert counts[IS, MOST, True] > 0 and counts["pole", True] > 0


@pytest.mark.parametrize("name", WORLDS)
def test_relation_facts_about_place_event_types_agree_with_the_oracle(cases, name) -> None:
    """A relation fact about a place event type ("owls can go to deserts") has a category of
    entities as its agent and a category of places as its patient, counts the bindings of an
    entity and a place, and takes the extensional quantifiers alone. Every one, plain or with a
    restricted place, has the oracle's verdict, which reads the requirement by brute force."""
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    truth, world = facts.truth, case.world
    assert facts.place_verbs
    true = Counter()
    categories = facts.categories if name == "tiny" else facts.categories[:12]
    for verb in facts.place_verbs:
        for agent in categories:
            for place in facts.place_categories:
                for patient in place_terms(facts, place)[:3]:
                    predicate = Predicate(PLACE_EVENT, verb, patient)
                    subject = CategoryTerm(agent)
                    for quantifier in QUANTIFIERS:
                        for polarity in (True, False):
                            proposition = Proposition(
                                CLASS, subject, predicate, polarity, quantifier
                            )
                            verdict = agree(truth, oracle, proposition)
                            if quantifier in NEC_QUANTIFIERS:
                                assert verdict is None  # no fixed test for a relation
                            true[quantifier] += bool(verdict)
                    # the count is over the bindings of a member and a place
                    grounded = truth.evaluate(Proposition(CLASS, subject, predicate, True, SOME))
                    members, places = truth.members(subject), truth.members(patient)
                    if len(places):
                        assert grounded.grounding["pairs"] == len(members) * len(places)
                        block = world.able(verb)[np.ix_(members, places)]
                        assert grounded.grounding["proportion"] == pytest.approx(
                            block.mean(), abs=1e-6
                        )
                    # a place does nothing: the roles cannot be exchanged
                    swapped = Proposition(
                        CLASS,
                        CategoryTerm(place),
                        Predicate(PLACE_EVENT, verb, subject),
                        True,
                        SOME,
                    )
                    assert not truth.evaluate(swapped).valid
    assert true[SOME] > 0 and true[MOST] > 0
    # the strongest true relation facts are the capacity facts the documents state
    stated = [
        fact
        for verb in facts.place_verbs
        for agent in categories
        for place in facts.place_categories
        if (
            fact := facts.class_fact(
                CategoryTerm(agent), Predicate(PLACE_EVENT, verb, CategoryTerm(place))
            )
        )
        is not None
    ]
    assert stated
    for fact in stated:
        assert oracle.truth(form_of(fact)) is True
        assert proposition_of(parse_propositional(_rendering(facts, fact))) == dataclasses.replace(
            fact, grounding=None
        )


def _rendering(facts, proposition: Proposition) -> str:
    from semantic_world.corpus.logical import logical_form
    from semantic_world.corpus.mentions import plan_for
    from semantic_world.corpus.renderings import propositional

    plan = plan_for(facts, proposition, bare=True)
    form = logical_form(plan, proposition, {}, facts.config.scalar_z, facts.truth.cuts)
    return propositional(json.loads(json.dumps(form)))


def test_places_and_entities_are_kept_apart(cases) -> None:
    """A predicate is said of the kind of thing it is about: a place feature, a place patient
    capacity, a place pole, and a category of places of a category of places alone, and the
    entities' of a category of entities alone; a category of places is the agent of nothing,
    takes no relative clause, and is compared with a category of places; the patient of a
    two-place event type is a category of entities. The truth test finds every crossing
    invalid, and the oracle never calls one true."""
    case = cases("tiny")
    facts, oracle = case.facts(), case.oracle()
    truth = facts.truth
    place, entity = facts.place_categories[0], facts.categories[0]
    place_term, entity_term = CategoryTerm(place), CategoryTerm(entity)
    place_predicates = [p for p in facts.class_predicates(place) if p.kind != MEMBER]
    entity_predicates = [
        p for p in facts.class_predicates(entity, patients=()) if p.kind not in (MEMBER, COMPARISON)
    ]
    crossings = [(entity_term, p) for p in place_predicates]
    crossings += [(place_term, p) for p in entity_predicates]
    crossings += [
        (entity_term, Predicate(MEMBER, place)),
        (place_term, Predicate(MEMBER, entity)),
        (place_term, Predicate(CAN, facts.features[CAN][0])),
        (place_term, Predicate(VERB, facts.verbs[0], entity_term)),
        (entity_term, Predicate(VERB, facts.verbs[0], place_term)),
        (entity_term, Predicate(PLACE_EVENT, facts.place_verbs[0], entity_term)),
        (place_term, Predicate(PLACE_EVENT, facts.place_verbs[0], place_term)),
    ]
    for subject, predicate in crossings:
        for quantifier in (ALL, MOST, SOME, NO):
            proposition = Proposition(CLASS, subject, predicate, True, quantifier)
            assert not truth.evaluate(proposition).valid, proposition
            assert oracle.truth(form_of(proposition)) is not True
    # the comparatives: a place scalar between two categories of places, and never across
    for scalar, first, second in (
        (facts.place_scalars[0], entity_term, place_term),
        (facts.place_scalars[0], place_term, entity_term),
        (facts.scalars[0], place_term, CategoryTerm(facts.place_categories[1])),
        (facts.place_scalars[0], entity_term, CategoryTerm(facts.categories[1])),
    ):
        proposition = Proposition(CLASS, first, Predicate(COMPARATIVE, scalar, second))
        assert not truth.evaluate(proposition).valid
        assert oracle.truth(form_of(proposition)) is not True
    # a category of places takes no relative clause, and no entity feature as a restriction
    clause = Clause(CAN, facts.features[CAN][0])
    predicate = place_predicates[0]
    with_clause = Proposition(CLASS, CategoryTerm(place, (), (clause,)), predicate, True, SOME)
    assert "relative clause" in truth.evaluate(with_clause).reason
    assert facts.draw_clause(np.random.default_rng(1), place_term) is None
    for literal in facts.restriction_options(entity_term)[:2]:
        crossed = Proposition(CLASS, CategoryTerm(place, (literal,)), predicate, True, SOME)
        assert not truth.evaluate(crossed).valid
    # the subject swap and the predicate swap of a test item stay within the kind of thing
    fact = facts.class_fact(place_term, predicate)
    if fact is not None:
        for change in (SUBJECT, PREDICATE):
            for false in false_items(facts, fact, change):
                assert truth.is_place_term(false.subject)
                assert facts.truth._about_a_place(false.predicate)


# ---------------------------------------------------------------------------------------------
# The documents about places and about pair fluents
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_documents_about_places_and_pair_fluents(cases, runs, name) -> None:
    """The categories of places are topics of category documents, with membership, facts about
    the topic and its subcategories, capacity facts with the category of places as the patient,
    and comparison facts with a sibling (a document about a category of entities states the
    capacity facts with its topic as the agent, Jon's ruling 1 of WM.76, and no other document
    states one); the place properties and parts and the place event
    types are topics of feature documents; and a pair fluent with a preposition is a topic
    whose document holds its profile statement, if it has one, and then causal statements
    alone, about the event types that add it, delete it, and need it. Every sentence is true
    by the oracle, reads back from its tree, and parses back from its rendering."""
    case = cases(name)
    corpus = runs(name, ENCYCLOPEDIC_COUNT[name], **ENCYCLOPEDIC)
    oracle = case.oracle(**ENCYCLOPEDIC)
    planner = corpus.planner
    facts, truth, lexicon = planner.facts, planner.truth, planner.lexicon
    topics: Counter = Counter()
    sections: Counter = Counter()
    kinds: Counter = Counter()
    for document in corpus.documents:
        topic = document.topic
        place_category = facts.is_place_category(topic)
        place_feature = topic in facts.place_features[IS] + facts.place_features[HAS]
        place_verb = topic in facts.place_verbs
        pair_fluent = topic in facts.pair_prepositions
        if not (place_category or place_feature or place_verb or pair_fluent):
            for sentence in document.sentences:
                proposition = sentence.proposition
                # no other document is about a place: a place feature and a category of places
                # as a subject stay in the documents above. A relation fact with a place stands
                # in a document about a category of entities too, with the topic as its agent
                # (Jon's ruling 1 of WM.76), and in no other document.
                if proposition.level == CLASS and isinstance(proposition.subject, CategoryTerm):
                    assert not truth.is_place_term(proposition.subject)
                    if proposition.predicate.kind == PLACE_EVENT:
                        assert document.type == "encyclopedic_category"
                        assert topic in facts.categories
                        assert sentence.section in (RELATION, "contrast")
                        if sentence.section == RELATION:
                            assert proposition.subject.category == topic
                        else:
                            assert proposition.subject.category in planner._siblings[topic]
                        assert truth.is_place_term(proposition.predicate.patient)
                        assert proposition.quantifier not in NEC_QUANTIFIERS
                        form = json.loads(json.dumps(sentence.logical_form))
                        assert oracle.truth(form) is True, form
                        assert (
                            proposition_of(parse_propositional(sentence.propositional))
                            == proposition
                        )
                assert not proposition.profile
            continue
        kind = (
            "place_category"
            if place_category
            else "place_feature"
            if place_feature
            else "place_event"
            if place_verb
            else "pair_fluent"
        )
        topics[kind] += 1
        assert document.type == (
            "encyclopedic_category" if place_category else "encyclopedic_feature"
        )
        seen: list[Proposition] = []
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert proposition.level == CLASS and truth.is_true(proposition)
            form = json.loads(json.dumps(sentence.logical_form))
            assert oracle.truth(form) is True, form
            record = sentence.sentence
            read = interpret(
                record.tree,
                lexicon,
                record.referents,
                record.events,
                sentence.plan.quantifier,
                sentence.plan.predication.standard,
            )
            assert read == sentence.plan and read.proposition() == proposition
            assert proposition_of(parse_propositional(sentence.propositional)) == proposition
            sections[kind, sentence.section] += 1
            subject, predicate = proposition.subject, proposition.predicate
            if place_category:
                assert isinstance(subject, CategoryTerm)
                if predicate.kind == PLACE_EVENT:
                    # a capacity fact with the topic as its patient, and an entity as its agent;
                    # its contrast says the same of a sibling of the topic
                    assert sentence.section in (RELATION, "contrast")
                    if sentence.section == RELATION:
                        assert predicate.patient.category == topic
                    else:
                        assert predicate.patient.category in planner._siblings[topic]
                    assert subject.category in facts.categories
                else:
                    # about the topic, a subcategory, or a sibling (a contrast, a comparison)
                    assert truth.is_place_term(subject)
                    family = {topic, *planner._children[topic], *planner._siblings.get(topic, [])}
                    assert subject.category in family
                    if subject.category != topic and predicate.kind == COMPARATIVE:
                        assert predicate.patient.category == topic
                kinds[predicate.kind] += 1
                if proposition.quantifier in NEC_QUANTIFIERS:
                    kinds["nec"] += 1
            elif place_feature:
                # which categories of places have the feature, and which lack it
                assert truth.is_place_term(subject) and predicate.label == topic
            elif place_verb:
                if proposition.causal:
                    assert subject.event_type == topic and sentence.section == CAUSAL
                else:
                    assert predicate.kind == PLACE_EVENT and predicate.label == topic
                    assert sentence.section in (RELATION, "contrast")
                    assert truth.is_place_term(predicate.patient)
            else:
                # a document about a pair fluent: statements alone
                if proposition.profile:
                    assert sentence.section == PROFILE_SECTION and predicate.label == topic
                    assert proposition in facts.profile_statements(topic)
                    assert proposition.grounding["can_hold"] is True
                    assert not seen, "the profile statement comes first"
                    assert "profile" in sentence.readings
                else:
                    assert proposition.causal and sentence.section == CAUSAL
                    assert pair_fluent_of_label(predicate.label) == topic
                    assert proposition in facts.causal_statements(pair_fluent=topic)
                seen.append(proposition)
        if pair_fluent:
            # the profile statement, if the pair fluent has one, and no statement twice
            profile = facts.profile_statements(topic)
            assert [p for p in seen if p.profile] == list(profile)
            assert len(seen) == len(set(seen))
            causal = facts.causal_statements(pair_fluent=topic)
            assert len(seen) == min(document.drawn_length, len(profile) + len(causal))
    # every kind of topic appears in a run of this size
    assert all(topics[kind] > 0 for kind in ("place_category", "place_event", "pair_fluent"))
    assert topics["place_feature"] > 0
    assert kinds[PROJECTION] > 0, "the place patient capacities are stated"
    assert sections["pair_fluent", CAUSAL] > 0
    assert sections["place_event", RELATION] > 0 and sections["place_event", CAUSAL] > 0
    if name == "default":
        assert kinds[COMPARATIVE] > 0 and kinds[COMPARISON] > 0 and kinds["nec"] > 0
        assert sections["pair_fluent", PROFILE_SECTION] > 0
    # the statements about the location fluent are in documents since this stage
    stated_location = any(
        d.topic in facts.location_prepositions and d.sentences for d in corpus.documents
    )
    assert stated_location
    # the statistics count them
    stats = corpus.stats
    by_topic = stats["documents"]["by_topic"]
    assert by_topic["place_category"] == topics["place_category"]
    assert by_topic["place_property"] + by_topic["place_part"] == topics["place_feature"]
    assert by_topic["place_event"] == topics["place_event"]
    assert by_topic["pair_fluent"] == by_topic["about_pair_fluents"] == topics["pair_fluent"]
    assert by_topic["about_places"] == (
        topics["place_category"] + topics["place_feature"] + topics["place_event"]
    )
    places = stats["places"]
    about = [
        s.proposition
        for d in corpus.documents
        for s in d.sentences
        if s.proposition.level == CLASS and truth.is_place_term(s.proposition.subject)
    ]
    assert places["class_sentences"] == len(about)
    assert places["capacity_sentences"] == sum(p.predicate.kind == PROJECTION for p in about)
    assert places["nec_sentences"] == sum(p.quantifier in NEC_QUANTIFIERS for p in about)
    assert places["relation_sentences"] == sum(
        s.proposition.predicate.kind == PLACE_EVENT and s.proposition.level == CLASS
        for d in corpus.documents
        for s in d.sentences
    )
    assert stats["causal"]["profile_sentences"] == sum(
        s.section == PROFILE_SECTION for d in corpus.documents for s in d.sentences
    )


# ---------------------------------------------------------------------------------------------
# Acceptance: the test sets
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_class_sets_hold_facts_about_categories_of_places(cases, runs, name) -> None:
    """The class sets draw their topics among the categories of places too. Each such item is
    true or false by the oracle as its set says, a subject swap brings in another category of
    places of the same level, a predicate swap another predicate about places of its kind, and
    a relation fact about a place event type takes no role swap."""
    case = cases(name)
    corpus = runs(name, test_sets={"size": 60})
    oracle = case.oracle()
    facts, truth = corpus.planner.facts, corpus.planner.truth
    found: Counter = Counter()
    for test_set in corpus.test_sets:
        if test_set.level != CLASS:
            continue
        for true, false in test_set.pairs:
            a, b = true.proposition, false.proposition
            if a.causal or a.profile:
                continue
            place_subject = truth.is_place_term(a.subject)
            place_relation = a.predicate.kind == PLACE_EVENT
            if not (place_subject or place_relation):
                assert not truth.is_place_term(b.subject) and b.predicate.kind != PLACE_EVENT
                continue
            found["subject" if place_subject else "relation"] += 1
            for item in (true, false):
                form = item.input["logical_form"]
                assert oracle.truth(form) is item.meta["truth"], form
                assert truth.is_true(item.proposition) is item.meta["truth"]
            assert truth.is_place_term(b.subject) == place_subject
            assert (b.predicate.kind == PLACE_EVENT) == place_relation
            assert test_set.change != ROLE or not place_relation
            if test_set.change == SUBJECT:
                assert facts.is_place_category(b.subject.category) == place_subject
                assert facts.level_of(b.subject.category) == facts.level_of(a.subject.category)
            if test_set.change == PREDICATE and place_subject:
                assert truth._about_a_place(b.predicate) and b.predicate.kind == a.predicate.kind
            if test_set.change == PREDICATE and place_relation:
                assert b.predicate.label in facts.place_verbs
            if test_set.kind == LAWLIKE:
                assert b.quantifier in NEC_QUANTIFIERS
    assert found["subject"] > 0 and found["relation"] > 0


@pytest.mark.parametrize("name", EVERY_WORLD)
def test_the_profile_sets_and_their_observed_marks(worlds, name) -> None:
    """The profile sets pair every profile statement of the world with every valid false
    item: ``profile_predicate`` the same form about another pair fluent between entities with
    a preposition, which lacks the profile, and ``profile_form`` the other form about the same
    pair fluent. Every false item is false by the truth test and by the oracle, which reads
    the symbol's symmetry; a false item is marked ``observed``, and stands in the ``_lawlike``
    twin, exactly when its form held of every pair at every time point of every scene of the
    corpus and some time point held a pair, by the oracle's replayed pairs. No statement is
    about a pair fluent that can never hold (Jon's ruling 2 of WM.75), and the other form is
    never observed."""
    case = worlds(name)
    settings = {"documents": {"count": 60 if name != "default" else 200}, "test_sets": {"size": 40}}
    corpus = generate(case.config(**settings), case.world)
    oracle = case.oracle(**settings)
    facts, truth = corpus.planner.facts, corpus.planner.truth
    scenes = [json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes]
    sets = {s.name: s for s in corpus.test_sets if s.level == PROFILE_LEVEL}
    assert list(sets) == ["profile_predicate", "profile_predicate_lawlike", "profile_form"]
    statements = facts.profile_statements()
    for statement in statements:
        assert truth.can_hold(statement.predicate.label)
    expected: dict[str, list] = {name: [] for name in sets}
    for change in (PREDICATE, FORM):
        for true in statements:
            for false in false_items(facts, true, change):
                observed = oracle.profile_observed(form_of(false), scenes)
                assert truth.profile_observed(false) is observed
                if change == FORM:
                    assert not observed, "a pair never holds both ways and one way only"
                key = f"profile_{change}" + ("_lawlike" if observed else "")
                expected[key].append((true, false))
    for set_name, test_set in sets.items():
        made = [(t.proposition, f.proposition) for t, f in test_set.pairs]
        assert made == expected[set_name], set_name
        assert test_set.stats()["true_statements"] == len({t for t, _ in made})
        for true, false in test_set.pairs:
            assert true.input["document"] is None and false.input["document"] is None
            assert oracle.truth(true.input["logical_form"]) is True
            assert oracle.truth(false.input["logical_form"]) is False
            assert truth.is_true(true.proposition) and not truth.is_true(false.proposition)
            assert false.meta["observed"] is (test_set.kind == LAWLIKE)
            assert true.meta["observed"] is False
            assert true.meta["statement"] == false.meta["statement"]
            assert "profile" in true.input["readings"] and "profile" in false.input["readings"]
            a, b = true.proposition, false.proposition
            if test_set.change == FORM:
                assert a.predicate.label == b.predicate.label
                assert a.predicate.form != b.predicate.form
            else:
                assert a.predicate.form == b.predicate.form
                symbol = case.world.pair_fluent_info[b.predicate.label]
                assert symbol.symmetry != b.predicate.form and not symbol.total
    if name == "tiny_pairs":
        # the hand world has a symmetric and two asymmetric pair fluents with pairs, and a
        # general one: both sets have pairs
        assert sets["profile_form"].pairs and sets["profile_predicate"].pairs


@pytest.mark.parametrize("name", HAND_WORLDS)
def test_causal_items_with_pair_literals_on_the_hand_worlds(worlds, name) -> None:
    """On a corpus of each hand world, whose event types read and write pairs by hand: every
    causal item is true or false by the oracle's re-reading of ``definition.json``, every
    ``observed`` mark agrees with the oracle's replay of the pairs of every scene, the role
    swap gives no item for a symmetric pair fluent or ``SAME``, and no mirror statement is the
    true item of any pair (Jon's ruling 4 of WM.75)."""
    case = worlds(name)
    settings = {"documents": {"count": 60}, "test_sets": {"size": 60}}
    corpus = generate(case.config(**settings), case.world)
    oracle = case.oracle(**settings)
    facts, world = corpus.planner.facts, case.world
    scenes = [json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes]
    with_pair = marked = 0
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS:
            continue
        for true, false in test_set.pairs:
            assert oracle.causal(true.input["logical_form"]) is True
            assert oracle.causal(false.input["logical_form"]) is False
            observed = oracle.observed(false.input["logical_form"], scenes)
            assert false.meta["observed"] is observed
            assert (test_set.kind == LAWLIKE) == observed
            a = true.proposition
            assert not facts.mirror(a.subject.role, a.predicate.label)
            if test_set.change == ROLE and a.predicate.pair_literal:
                assert not facts.mirror("patient", a.predicate.label)
            with_pair += a.predicate.pair_literal
            marked += observed and a.predicate.pair_literal
    assert with_pair > 0
    # the mirror of each statement on a symmetric pair fluent, and of each SAME statement, is
    # true by the truth test and by the oracle, and is no statement of the world
    mirrors = 0
    for statement in facts.causal_statements():
        label = statement.predicate.label
        if not facts.mirror("patient", label) or statement.subject.condition:
            continue
        assert statement.subject.role == "agent"
        mirror = dataclasses.replace(
            statement,
            subject=dataclasses.replace(statement.subject, role="patient"),
            grounding=None,
        )
        assert facts.truth.is_true(mirror)
        form = form_of(mirror)
        form["subject"] = {"head": "THING", "event": mirror.subject.event_type, "role": "patient"}
        assert oracle.causal(form) is True
        assert all(
            (p.subject, p.predicate) != (mirror.subject, mirror.predicate)
            for p in facts.causal_statements()
        )
        assert false_items(facts, statement, ROLE) == []
        mirrors += 1
    assert mirrors > 0
    assert world.pair_fluents_used()


# ---------------------------------------------------------------------------------------------
# WM.75: the prepositions, and the comparisons that try another dimension
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", EVERY_WORLD)
def test_every_used_pair_fluent_has_a_preposition(worlds, name) -> None:
    """Jon's ruling 1 of WM.75: a pair fluent that some precondition or condition literal
    reads in any form, compiled literals included, or that some effect writes, has a
    preposition, and a pair fluent that nothing reads or writes has none, by a re-reading of
    ``definition.json``. The prepositions of the pair fluents that no literal reads as a pair
    are the last content concepts, after the generic noun for places, so the content lexemes
    before them are those of a lexicon without them, label for label."""
    case = worlds(name)
    lexicon = case.lexicon()
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    used: set[str] = set()
    read_as_pair: set[str] = set()
    for entry in record["event_types"]:
        literals = list(entry["precondition"]["literals"])
        for effect in entry["effects"]:
            literals += (effect.get("condition") or {}).get("literals", [])
            if effect.get("kind") == "pair":
                used.add(effect["pair_fluent"])
        for literal in literals:
            if "pair_fluent" in literal:
                used.add(literal["pair_fluent"])
                if literal["kind"] == "pair":
                    read_as_pair.add(literal["pair_fluent"])
    pair_fluents = [s["label"] for s in record["symbols"] if s["kind"] == "pair_fluent"]
    named = [c.label for c in lexicon.of_type("pair") if lexicon.is_named(c.label)]
    assert sorted(named) == sorted(used)
    assert all(not lexicon.is_named(f) for f in pair_fluents if f not in used)
    later = [f for f in pair_fluents if f in used and f not in read_as_pair]
    content = [x.concept for x in lexicon.content_lexemes]
    assert content[len(content) - len(later) :] == later
    if case.world.place_categories:
        assert content[len(content) - len(later) - 1] == "SOMEPLACE"
    # every content lexeme before them keeps the label its position gives it
    for position, lexeme in enumerate(lexicon.content_lexemes, start=1):
        assert lexeme.label == f"LEXEME.{position}"
    # the facts can state every one: a causal statement or a profile statement names it
    facts = case.facts()
    assert sorted(facts.pair_prepositions) == sorted(used)


class _StubMentions:
    """Mentions that tell two referents apart on chosen dimensions alone."""

    def __init__(self, apart: set[str | None]) -> None:
        self.apart = apart
        self.referents: dict[str, str] = {"x": "REF.1"}
        self.skipped_comparisons: list[str] = []
        self.other_scalar_comparisons: list[str] = []
        self.asked: list[str | None] = []
        self.dropped = 0

    def comparison_phrases(self, rng, first, second, dimension=None):
        self.asked.append(dimension)
        self.referents[second] = "REF.2"
        phrases = (
            NounPhrase(INSTANCE_NP, first, "N", "the"),
            NounPhrase(INSTANCE_NP, second, "N", "the"),
        )
        return phrases[0], phrases[1], dimension in self.apart

    def drop_sentence(self, referents) -> None:
        self.referents = dict(referents)
        self.dropped += 1


def test_a_comparison_tries_the_other_dimensions_before_it_is_skipped(cases) -> None:
    """Jon's ruling 3 of WM.75: when nothing tells the two mentions of the drawn comparison
    apart, the other dimensions of the same two arguments are tried, each once; the comparison
    is made on the first that tells them apart, and counted as made on another scalar; it is
    skipped, and counted, only when no dimension allows it. A comparison whose first draw
    tells the mentions apart tries nothing else, and a dimension of another pair of arguments
    is never tried."""
    case = cases("tiny")
    planner = Planner(case.config(), case.result)
    facts = planner.facts
    scalar = facts.scalars[0]
    numeric = list(facts.numeric_fluents)
    assert len(numeric) >= 2
    fact = Proposition("instance", "x", Predicate(COMPARATIVE, scalar, "y"))
    options = [("x", "y", d, fact) for d in (scalar, *numeric)]
    other_pair = ("x", "z", scalar, fact)
    seen: Counter = Counter()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        # told apart on the first draw: nothing else is asked
        mentions = _StubMentions({None, scalar})
        made = planner._told_apart_comparison(
            rng, rng, mentions, [*options, other_pair], PARTICIPANTS
        )
        assert made is not None and len(mentions.asked) == 1 and not mentions.dropped
        assert not mentions.other_scalar_comparisons and not mentions.skipped_comparisons
        # told apart on the scalar alone (a numeric fluent passes no dimension)
        mentions = _StubMentions({scalar})
        made = planner._told_apart_comparison(
            np.random.default_rng(seed), rng, mentions, options, PARTICIPANTS
        )
        assert made is not None and made[0] == scalar
        first_was_scalar = len(mentions.asked) == 1
        assert mentions.other_scalar_comparisons == ([] if first_was_scalar else [PARTICIPANTS])
        assert mentions.dropped == len(mentions.asked) - 1
        assert not mentions.skipped_comparisons
        seen["first" if first_was_scalar else "another"] += 1
        # told apart on nothing: every dimension of the pair is tried once, and it is skipped
        mentions = _StubMentions(set())
        made = planner._told_apart_comparison(
            np.random.default_rng(seed), rng, mentions, [*options, other_pair], PARTICIPANTS
        )
        assert made is None and mentions.skipped_comparisons == [PARTICIPANTS]
        assert not mentions.other_scalar_comparisons
        assert mentions.referents == {"x": "REF.1"}, "the dropped mentions never happened"
        drawn_other_pair = len(mentions.asked) == 1 and mentions.asked[0] == scalar
        if not drawn_other_pair:
            assert len(mentions.asked) == len(options)
    assert seen["first"] > 0 and seen["another"] > 0


@pytest.mark.parametrize("name", WORLDS)
def test_the_comparisons_counts_of_the_corpus(cases, runs, name) -> None:
    """``stats.yaml`` counts the comparisons made on another dimension than the one first
    drawn and the comparisons still skipped, for participants and for places, as the planner
    counted them; every comparison a document makes is true by the oracle."""
    case = cases(name)
    settings = {"documents": {"comparative_rate": 0.9}}
    corpus = runs(name, **settings)
    oracle = case.oracle(**settings)
    comparisons = corpus.stats["comparisons"]
    assert comparisons["made_on_another_scalar"] == corpus.planner.other_scalar_comparisons
    assert comparisons["skipped_not_told_apart"] == corpus.planner.skipped_comparisons
    assert set(comparisons["made_on_another_scalar"]) == {"participants", "places"}
    scenes = {s.label: json.loads(json.dumps(s.to_json())) for s in corpus.planner.scenes}
    made = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            if sentence.section not in ("comparative", "place_comparative"):
                continue
            form = json.loads(json.dumps(sentence.logical_form))
            assert oracle.truth(form, scenes.get(form.get("scene"))) is True
            made += 1
    assert made > 0
