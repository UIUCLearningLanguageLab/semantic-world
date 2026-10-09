"""Stage 4: the sentence plan of a proposition, and relative clauses."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from semantic_world.corpus import Streams
from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_NP,
    NounPhrase,
    Predication,
    SentencePlan,
    check_plan,
)
from semantic_world.corpus.histories import scene_events, scene_of
from semantic_world.corpus.mentions import (
    MentionRules,
    Mentions,
    RelativeClauses,
    clause_propositions,
    leaf_of,
    mention,
    plan_for,
)
from semantic_world.corpus.propositions import (
    CLASS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NEC_ALL,
    SCALAR,
    VERB,
    CategoryTerm,
    Literal,
    Predicate,
    Proposition,
)

WORLDS = ("tiny", "default", "deep")


def rate_settings(**clauses) -> dict:
    return {"mention": {"relative_clauses": clauses}}


def scenes_of(case, facts, count: int = 40):
    generator = case.scenes()
    labels = case.world.instances
    streams = Streams(1)
    made = [generator.scene(streams, n, labels[(n * 5) % len(labels)]) for n in range(1, count + 1)]
    for scene in made:
        facts.truth.add_scene(scene)
    return made


# ---------------------------------------------------------------------------------------------
# The plan of a proposition
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_plan_of_a_proposition_says_the_proposition(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    propositions = []
    for category in facts.categories[:6]:
        for negative in (False, True):
            propositions += facts.class_facts(category, negative, patients=facts.categories[:4])
    propositions += list(facts.rule_statements())
    labels = case.world.instances
    for instance in labels[:4]:
        for negative in (False, True):
            propositions += facts.instance_facts(instance, negative, patients=labels[:8])
    for scene in scenes_of(case, facts, 10):
        propositions += [facts.event_fact(e) for e in scene_events(scene)]
    propositions = [p for p in propositions if p is not None]
    assert len(propositions) > 500
    for proposition in propositions:
        for bare in (False, True):
            plan = plan_for(facts, proposition, bare=bare)
            check_plan(plan)
            assert plan.proposition() == proposition
            assert plan.level == proposition.level
            if proposition.level == CLASS:
                assert plan.bare_plural == (bare or proposition.quantifier is None)
    assert {p.level for p in propositions} == {CLASS, INSTANCE, EVENT}


def test_class_level_plans(cases) -> None:
    facts = cases("tiny").facts()
    subject = CategoryTerm("CATEGORY.1.1", (Literal("PROPERTY.2"), Literal("PART.3", False)))
    patient = CategoryTerm("CATEGORY.2", (Literal("PROPERTY.4"),))
    proposition = Proposition(
        CLASS, subject, Predicate(VERB, "EVENTTYPE2.1.1", patient), False, MOST
    )
    plan = plan_for(facts, proposition)
    assert plan == SentencePlan(
        NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1.1", "most", subject.restriction),
        Predication(
            VERB,
            "EVENTTYPE2.1.1",
            False,
            NounPhrase(CLASS_NP, "CATEGORY.2", "CATEGORY.2", None, patient.restriction),
        ),
        MOST,
    )
    # the subject's determiner is the quantifier's word, or none for a bare plural; the plan
    # keeps the quantifier either way
    bare = plan_for(facts, proposition, bare=True)
    assert bare.subject.determiner is None and bare.quantifier == MOST
    assert bare.proposition() == proposition and bare.bare_plural
    rule = facts.rule_statements()[0]
    assert plan_for(facts, rule).subject.noun == "THING"
    assert plan_for(facts, rule).subject.determiner == "all"
    assert plan_for(facts, rule).quantifier == NEC_ALL
    nec_no = Proposition(CLASS, subject, Predicate(MEMBER, "CATEGORY.2"), True, "nec_no")
    assert plan_for(facts, nec_no).subject.determiner == "no"
    # a class-level scalar pole has no quantifier word
    pole = Proposition(CLASS, CategoryTerm("CATEGORY.1.1"), Predicate(SCALAR, "SCALARDIM.1.HIGH"))
    assert plan_for(facts, pole).subject.determiner is None
    assert plan_for(facts, pole).quantifier is None and plan_for(facts, pole).bare_plural


def test_instances_are_mentioned_by_their_leaf_unless_told_otherwise(cases) -> None:
    facts = cases("default").facts()
    instance, other = "INSTANCE.1.2.1.3", "INSTANCE.2.1.1.1"
    assert leaf_of(facts, instance) == "CATEGORY.1.2.1"
    assert mention(facts, instance) == NounPhrase(INSTANCE_NP, instance, "CATEGORY.1.2.1", "the")
    capacity = Proposition(INSTANCE, instance, Predicate(VERB, "EVENTTYPE2.1.1", other))
    plan = plan_for(facts, capacity)
    assert plan.subject == mention(facts, instance)
    assert plan.predication.object == mention(facts, other)
    # a caller can give any other mention: a higher noun, "a", modifiers, or a pronoun
    given = {
        instance: NounPhrase(INSTANCE_NP, instance, "CATEGORY.1", "a", (Literal("PROPERTY.3"),)),
        other: NounPhrase(INSTANCE_NP, other),
    }
    plan = plan_for(facts, capacity, given)
    assert plan.subject == given[instance] and plan.predication.object == given[other]
    assert plan.proposition() == capacity
    # a scalar pole's subject is named by the comparison class
    for comparison in ("CATEGORY.1", "CATEGORY.1.2", "CATEGORY.1.2.1"):
        pole = Proposition(
            INSTANCE, instance, Predicate(SCALAR, "SCALARDIM.1.HIGH", comparison=comparison)
        )
        plan = plan_for(facts, pole)
        assert plan.subject.noun == comparison and plan.proposition() == pole


def test_a_membership_sentence_never_names_the_subject_by_the_predicate(cases) -> None:
    facts = cases("default").facts()
    instance = "INSTANCE.1.2.1.3"
    plans = {
        category: plan_for(facts, Proposition(INSTANCE, instance, Predicate(MEMBER, category)))
        for category in ("CATEGORY.1", "CATEGORY.1.2", "CATEGORY.1.2.1")
    }
    assert plans["CATEGORY.1"].subject.noun == "CATEGORY.1.2.1"  # "the penguin is an animal"
    assert plans["CATEGORY.1.2"].subject.noun == "CATEGORY.1.2.1"
    assert plans["CATEGORY.1.2.1"].subject.noun == "CATEGORY.1.2"  # "the bird is a penguin"
    for plan in plans.values():
        check_plan(plan)
    # a world of one level has no category above the leaf: the subject is a pronoun
    flat = cases("tiny").facts()
    still = cases("still").facts()
    assert (
        plan_for(
            flat, Proposition(INSTANCE, "INSTANCE.1.1.1", Predicate(MEMBER, "CATEGORY.1.1"))
        ).subject.noun
        == "CATEGORY.1"
    )
    for facts_of in (flat, still):
        top = Proposition(INSTANCE, facts_of.world.instances[0], Predicate(MEMBER, "CATEGORY.1"))
        assert plan_for(facts_of, top).subject.noun is not None


# ---------------------------------------------------------------------------------------------
# Relative clauses
# ---------------------------------------------------------------------------------------------


def attach_all(case, settings: dict, level: str, seed: int = 0, count: int = 400, scenes: int = 40):
    """Plans of one level with relative clauses drawn, and the facts they were drawn from."""
    facts = case.facts(**settings)
    clauses = RelativeClauses(case.config(**settings), facts)
    rng = Streams(seed).mentions
    labels = case.world.instances
    others = labels[:: max(1, len(labels) // 7)][:7]
    plans = []
    if level == INSTANCE:
        pool = [f for i in others[:4] for f in facts.instance_facts(i, patients=others)]
        for index in rng.choice(len(pool), size=count):
            plans.append(clauses.attach(rng, plan_for(facts, pool[int(index)]), others))
    else:
        events = [e for scene in scenes_of(case, facts, scenes) for e in scene_events(scene)]
        for event in events[:count]:
            report = facts.event_fact(event)
            if report is not None:
                plans.append(clauses.attach(rng, plan_for(facts, report)))
    return plans, facts


@pytest.mark.parametrize("name", WORLDS)
@pytest.mark.parametrize("level", [INSTANCE, EVENT])
def test_a_relative_clause_expresses_a_true_proposition_about_its_head(cases, name, level) -> None:
    case = cases(name)
    plans, facts = attach_all(case, rate_settings(rate=0.7, max_depth=2), level)
    clauses = 0
    for plan in plans:
        check_plan(plan)
        main = plan.proposition()
        assert facts.truth.is_true(main)
        said = [main]
        for proposition in clause_propositions(plan):
            # of the same level as its sentence, and true
            assert proposition.level == level
            assert facts.truth.is_true(proposition), proposition
            # and it does not say again what the sentence already says
            assert proposition not in said
            said.append(proposition)
            clauses += 1
    assert clauses > 100


def test_event_clauses_report_events_of_the_same_scene(cases) -> None:
    case = cases("default")
    # 120 scenes: two-place events, which an object relative needs, are few in the default
    # world (since stage a5b, about 6% of its events)
    plans, facts = attach_all(
        case, rate_settings(rate=0.8, max_depth=2), EVENT, count=1000, scenes=120
    )
    subject_relatives = object_relatives = 0
    for plan in plans:
        scene = scene_of(plan.predication.event)
        events = {e.label: e for e in facts.truth.events_of(scene)}
        # no clause reports an event with the verb, the agent, and the patient of the sentence's
        # own event, or of another clause's, even one that happened at another step
        reported = [events[plan.predication.event].key] + [
            events[phrase.clause.predications[0].event].key
            for phrase in plan.noun_phrases()
            if phrase.clause is not None
        ]
        assert len(set(reported)) == len(reported)
        for phrase in plan.noun_phrases():
            if phrase.clause is None:
                continue
            (predication,) = phrase.clause.predications
            event = events[predication.event]  # an event of the sentence's own scene
            assert event.label != plan.predication.event
            if phrase.clause.agent is not None:
                # "the cat that the dog chased": the head is the patient
                assert event.patient == phrase.referent
                assert phrase.clause.agent.referent == event.agent
                object_relatives += 1
            else:
                assert event.agent == phrase.referent
                if event.patient is not None:
                    assert predication.object.referent == event.patient
                subject_relatives += 1
    assert subject_relatives > 50 and object_relatives > 20


def test_the_rate_and_the_share_of_object_relatives(cases) -> None:
    case = cases("default")

    def shares(**clauses) -> tuple[float, float]:
        plans, _ = attach_all(case, rate_settings(max_depth=1, **clauses), EVENT)
        phrases = [p for plan in plans for p in (plan.subject, plan.predication.object) if p]
        with_clause = [p for p in phrases if p.clause is not None]
        objects = [p for p in with_clause if p.clause.agent is not None]
        return len(with_clause) / len(phrases), len(objects) / max(1, len(with_clause))

    assert shares(rate=0.0) == (0.0, 0.0)
    low, _ = shares(rate=0.1)
    high, _ = shares(rate=0.6)
    assert 0.03 < low < 0.15 and 0.4 < high <= 0.6
    # when the drawn kind has no true proposition, the other kind is used, so the share of
    # object relatives follows the setting only as far as the events allow
    _, none = shares(rate=0.6, object_share=0.0)
    _, most = shares(rate=0.6, object_share=1.0)
    _, default = shares(rate=0.6)
    assert none < default < most


def test_what_takes_no_drawn_relative_clause(cases) -> None:
    case = cases("default")
    settings = rate_settings(rate=1.0, max_depth=3)
    facts = case.facts(**settings)
    clauses = RelativeClauses(case.config(**settings), facts)
    rng = np.random.default_rng(0)
    # a class-level sentence: its relative clauses restrict the subject, so they are drawn with
    # the proposition (facts.draw_clause), and not for a finished plan
    for fact in facts.class_facts("CATEGORY.1.1", patients=("CATEGORY.2",))[:40]:
        plan = plan_for(facts, fact)
        assert clauses.attach(rng, plan) == plan
    # a pronoun, and a noun phrase that already has a clause
    instance = case.world.instances[0]
    fact = facts.instance_facts(instance)[0]
    pronoun = plan_for(facts, fact, {instance: NounPhrase(INSTANCE_NP, instance)})
    assert clauses.attach(rng, pronoun) == pronoun
    drawn = clauses.attach(rng, plan_for(facts, fact))
    assert drawn.subject.clause is not None  # the rate is 1, and the instance has a CAN feature
    assert clauses.attach(rng, drawn) == drawn
    # with the depth limit or the rate at 0, nothing is drawn
    for off in (rate_settings(rate=1.0, max_depth=0), rate_settings(rate=0.0, max_depth=2)):
        none = RelativeClauses(case.config(**off), case.facts(**off))
        assert none.attach(rng, plan_for(facts, fact)) == plan_for(facts, fact)


def test_negated_literals_keep_the_clause_a_subject_relative(cases) -> None:
    case = cases("default")
    settings = rate_settings(rate=1.0, max_depth=1, object_share=1.0)
    facts = case.facts(**settings)
    clauses = RelativeClauses(case.config(**settings), facts)
    rng = np.random.default_rng(0)
    labels = case.world.instances
    instance = labels[0]
    row = facts.truth.instance_index[instance]
    lacking = next(f for f in facts.features[IS] if not facts.world.column(f)[row])
    phrase = dataclasses.replace(mention(facts, instance), restriction=(Literal(lacking, False),))
    fact = facts.instance_facts(instance)[0]
    for _ in range(30):
        plan = clauses.attach(rng, plan_for(facts, fact, {instance: phrase}), labels[:8])
        check_plan(plan)
        # the negated literal is a verb phrase of the clause, so the drawn clause joins it
        assert plan.subject.clause is None or plan.subject.clause.agent is None
    assert plan.subject.clause is not None


def test_clauses_are_drawn_from_the_mentions_stream_alone(cases) -> None:
    case = cases("default")
    base, _ = attach_all(case, rate_settings(rate=0.6, max_depth=2), EVENT, seed=3)
    again, _ = attach_all(case, rate_settings(rate=0.6, max_depth=2), EVENT, seed=3)
    assert base == again
    other, _ = attach_all(case, rate_settings(rate=0.6, max_depth=2), EVENT, seed=4)
    assert other != base
    # the grammar settings change no plan: what a sentence says is settled before the grammar
    grammar = {
        "grammar": {
            "word_order": {"clause": "OSV", "relative_clause": "before"},
            "morphology": {"number": {"enabled": True}, "aspect": {"enabled": True}},
            "can_rate": {"class": 0.9, "instance": 0.4},
            "adjective_order": {"fixed": False},
        }
    }
    changed, _ = attach_all(
        case, {**rate_settings(rate=0.6, max_depth=2), **grammar}, EVENT, seed=3
    )
    assert changed == base


# ---------------------------------------------------------------------------------------------
# Class-level plans with relative clauses
# ---------------------------------------------------------------------------------------------


def test_the_plan_of_a_proposition_with_a_restrictive_clause(cases) -> None:
    case = cases("default")
    settings = rate_settings(rate=0.6, max_depth=2)
    facts = case.facts(**settings)
    rng = Streams(2).propositions
    made = 0
    for category in facts.categories[::3]:
        for _ in range(4):
            term = facts.draw_clause(rng, CategoryTerm(category))
            if term is None:
                continue
            for fact in facts.class_facts(term, patients=facts.categories[:3])[::5]:
                plan = plan_for(facts, fact)
                check_plan(plan)
                # the clause of the noun phrase is the clause of the category term
                assert plan.proposition() == fact
                assert plan.subject.clause is not None and plan.depth() >= 1
                clause = term.clauses[0]
                assert plan.subject.clause.object_relative == (clause.agent is not None)
                made += 1
    assert made > 100


# ---------------------------------------------------------------------------------------------
# Mentions in a document
# ---------------------------------------------------------------------------------------------


def rules_of(case, **mention_settings) -> MentionRules:
    settings = {"mention": mention_settings}
    config = case.config(**settings)
    return MentionRules(config, case.facts(**settings), Streams(config.seed))


def test_the_preference_order_is_fixed_for_the_language(cases) -> None:
    case = cases("default")
    rules = rules_of(case)
    facts = case.facts()
    scalars = sorted({pole.rsplit(".", 1)[0] for pole in facts.poles})
    assert sorted(rules.preference) == sorted(facts.features[IS] + facts.features[HAS]) + scalars
    assert rules_of(case).preference == rules.preference
    other = MentionRules(case.config(seed=2), facts, Streams(2))
    assert other.preference != rules.preference


def test_the_incremental_algorithm(cases) -> None:
    case = cases("default")
    rules = rules_of(case)
    labels = case.world.instances
    rank = {attribute: n for n, attribute in enumerate(rules.preference)}
    told_apart = 0
    for start in range(0, len(labels) - 6, 5):
        cast = labels[start : start + 6]
        for instance in cast[:2]:
            for noun in rules.path(instance):
                literals, alone = rules.distinguish(instance, noun, cast)
                phrase = NounPhrase(INSTANCE_NP, instance, noun, "the", literals)
                fitting = rules.matches(phrase, cast)
                # what the noun phrase says is true of the instance
                assert instance in fitting and alone == (fitting == [instance])
                others = [c for c in cast if c != instance and noun in rules.path(c)]
                if not others:
                    assert literals == () and alone
                    continue
                # the attributes are tried in the preference order, and each one that is added
                # rules out a participant that was still left
                order = [
                    rank[x.feature.rsplit(".", 1)[0] if x.pole else x.feature] for x in literals
                ]
                assert order == sorted(order)
                left = list(others)
                for literal in literals:
                    row = [rules.truth.instance_index[c] for c in left]
                    kept = [
                        c for c, r in zip(left, row, strict=True) if rules.fits(r, literal, noun)
                    ]
                    assert len(kept) < len(left)
                    left = kept
                assert alone == (not left)
                # no negated PROPERTY literal: adjectives and with-phrases only
                assert all(x.positive or x.feature.startswith("PART.") for x in literals)
                assert sum(not x.feature.startswith("PART.") for x in literals) <= 3
                assert sum(x.feature.startswith("PART.") for x in literals) <= 2
                told_apart += alone
    assert told_apart > 50
    # the features that the sentence states are not said again
    instance, noun = labels[0], rules.path(labels[0])[0]
    literals, _ = rules.distinguish(instance, noun, labels[:30])
    assert literals
    avoided, _ = rules.distinguish(instance, noun, labels[:30], avoid=(literals[0].feature,))
    assert literals[0].feature not in [x.feature for x in avoided]


def test_a_document_s_mentions(cases) -> None:
    case = cases("default")
    labels = case.world.instances
    first, second, third = "INSTANCE.1.1.1.1", "INSTANCE.1.1.1.2", "INSTANCE.2.1.1.1"
    assert {first, second, third} <= set(labels)
    cast = (first, second, third)
    rng = np.random.default_rng(0)
    # with the pronoun rate at 0, a later mention is always a noun phrase
    mentions = Mentions(rules_of(case, pronoun_rate=0.0), cast)
    opening = mentions.noun_phrase(rng, first)
    assert opening.determiner == "a" and opening.restriction == ()
    assert opening.noun in ("CATEGORY.1", "CATEGORY.1.1", "CATEGORY.1.1.1")
    assert mentions.referents == {first: "REF.1"} and mentions.label(first) == "REF.1"
    assert mentions.distinguished(opening) is None  # an indefinite mention picks out no one
    mentions.end_sentence(first)
    later = mentions.noun_phrase(rng, first)
    assert later.determiner == "the"
    # the noun fits the other instance of the leaf, so modifiers tell the two apart
    assert later.restriction and mentions.distinguished(later) is True
    assert mentions.rules.matches(later, cast) == [first]
    other = mentions.noun_phrase(rng, third)
    assert other.determiner == "a" and mentions.referents == {first: "REF.1", third: "REF.2"}
    mentions.end_sentence(first)
    # a noun that is given is used, and a noun that is excluded is not
    for _ in range(20):
        assert mentions.noun_phrase(rng, first, noun="CATEGORY.1").noun == "CATEGORY.1"
        assert mentions.noun_phrase(rng, first, exclude="CATEGORY.1.1.1").noun != "CATEGORY.1.1.1"
    mentions.end_sentence(first)
    # a sentence that is dropped leaves no mention behind
    known = dict(mentions.referents)
    mentions.noun_phrase(rng, second)
    assert second in mentions.referents
    mentions.drop_sentence(known)
    assert mentions.referents == known
    assert mentions.noun_phrase(rng, second).determiner == "a"


def test_when_a_mention_can_be_a_pronoun(cases) -> None:
    case = cases("default")
    first, second = "INSTANCE.1.1.1.1", "INSTANCE.2.1.1.1"
    rng = np.random.default_rng(0)
    mentions = Mentions(rules_of(case, pronoun_rate=1.0), (first, second))
    # a first mention is never a pronoun
    assert not mentions.noun_phrase(rng, first).pronoun
    assert not mentions.pronoun_allowed(first)  # not in the same sentence
    mentions.end_sentence(first)
    # the only referent of the sentence before
    assert mentions.pronoun_allowed(first) and not mentions.pronoun_allowed(second)
    assert mentions.noun_phrase(rng, first).pronoun
    assert not mentions.noun_phrase(rng, first).pronoun  # said once in a sentence
    assert not mentions.noun_phrase(rng, second).pronoun
    mentions.end_sentence(first)
    # two referents: the subject can be a pronoun, and the other one cannot
    assert mentions.pronoun_allowed(first) and not mentions.pronoun_allowed(second)
    assert not mentions.noun_phrase(rng, second).pronoun
    mentions.end_sentence(second)
    # the referent was not mentioned in the sentence before
    assert not mentions.pronoun_allowed(first) and mentions.pronoun_allowed(second)
    # a mention whose noun is given, or that must not be a pronoun, is a noun phrase
    assert not mentions.noun_phrase(rng, second, noun="CATEGORY.2").pronoun
    mentions.end_sentence(second)
    assert not mentions.noun_phrase(rng, second, pronoun=False).pronoun
    mentions.end_sentence(second)
    assert mentions.noun_phrase(rng, second).pronoun
    assert mentions.distinguished(mentions.noun_phrase(rng, first)) is not None


def test_a_referent_that_cannot_be_told_apart(cases) -> None:
    """With no adjective and no with-phrase allowed, two instances of one leaf cannot be told
    apart. The mention falls back to the noun of the leaf, is kept, and is marked."""
    case = cases("default")
    first, second, third = "INSTANCE.1.1.1.1", "INSTANCE.1.1.1.2", "INSTANCE.1.1.2.1"
    rules = rules_of(case, max_adjectives=0, max_with_phrases=0, pronoun_rate=0.0)
    assert rules.distinguish(first, "CATEGORY.1.1.1", (first, second)) == ((), False)
    rng = np.random.default_rng(3)
    mentions = Mentions(rules, (first, second, third))
    mentions.noun_phrase(rng, first)
    mentions.end_sentence(first)
    for _ in range(20):
        later = mentions.noun_phrase(rng, first)
        mentions.end_sentence(first)
        # the noun of the leaf fits the fewest participants, and still fits two
        assert later.noun == "CATEGORY.1.1.1" and later.restriction == ()
        assert mentions.distinguished(later) is False
    # the other leaf's instance is alone under its leaf's noun
    mentions.noun_phrase(rng, third)
    mentions.end_sentence(third)
    for _ in range(20):
        later = mentions.noun_phrase(rng, third)
        mentions.end_sentence(third)
        assert later.noun == "CATEGORY.1.1.2" and mentions.distinguished(later) is True
    # a noun that is given is kept, whatever it fits
    fixed = mentions.noun_phrase(rng, third, noun="CATEGORY.1")
    assert fixed.noun == "CATEGORY.1" and mentions.distinguished(fixed) is False


def test_modifiers_at_the_modifier_rate(cases) -> None:
    case = cases("default")
    instance = "INSTANCE.1.1.1.1"
    rng = np.random.default_rng(1)

    def first_mentions(modifiers: bool, **settings):
        rules = rules_of(case, **settings)
        return [
            Mentions(rules, (instance,), modifiers=modifiers).noun_phrase(
                rng, instance, avoid=("PROPERTY.1",)
            )
            for _ in range(300)
        ]

    # only in the documents that ask for them
    assert all(not phrase.restriction for phrase in first_mentions(False))
    drawn = first_mentions(True)
    share = np.mean([bool(phrase.restriction) for phrase in drawn])
    assert abs(share - 0.3) < 0.08
    rules = rules_of(case)
    row = rules.truth.instance_index[instance]
    kinds = set()
    for phrase in drawn:
        for literal in phrase.restriction:
            # one modifier, true of the referent, positive, and not what the sentence says
            assert len(phrase.restriction) == 1 and literal.positive
            assert rules.fits(row, literal, phrase.noun) and literal.feature != "PROPERTY.1"
            kinds.add(literal.feature.split(".")[0])
    assert {"PROPERTY", "PART"} <= kinds
    assert all(phrase.restriction for phrase in first_mentions(True, modifier_rate=1.0))
    assert all(
        not phrase.restriction
        for phrase in first_mentions(True, max_adjectives=0, max_with_phrases=0)
    )
