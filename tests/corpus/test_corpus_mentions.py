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
from semantic_world.corpus.mentions import (
    RelativeClauses,
    clause_propositions,
    leaf_of,
    mention,
    plan_for,
)
from semantic_world.corpus.propositions import (
    ALL,
    CLASS,
    EVENT,
    GENERIC,
    INSTANCE,
    MEMBER,
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
    labels = case.result.instances.labels
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
            stated = facts.class_facts(category, negative, patients=facts.categories[:4])
            propositions += list(stated) + [g for g in map(facts.generic, stated) if g]
    propositions += list(facts.rule_statements())
    labels = case.result.instances.labels
    for instance in labels[:4]:
        for negative in (False, True):
            propositions += facts.instance_facts(instance, negative, patients=labels[:8])
    for scene in scenes_of(case, facts, 10):
        propositions += [facts.event_fact(e) for e in scene.events]
    assert len(propositions) > 500
    for proposition in propositions:
        plan = plan_for(facts, proposition)
        check_plan(plan)
        assert plan.proposition() == proposition
        assert plan.level == proposition.level
    assert {p.level for p in propositions} == {CLASS, INSTANCE, EVENT}


def test_class_level_plans(cases) -> None:
    facts = cases("tiny").facts()
    subject = CategoryTerm("C1.1", (Literal("IS.2"), Literal("HAS.3", False)))
    patient = CategoryTerm("C2", (Literal("IS.4"),))
    proposition = Proposition(CLASS, subject, Predicate(VERB, "V1.1", patient), False, "most")
    plan = plan_for(facts, proposition)
    assert plan == SentencePlan(
        NounPhrase(CLASS_NP, "C1.1", "C1.1", "most", subject.restriction),
        Predication(
            VERB, "V1.1", False, NounPhrase(CLASS_NP, "C2", "C2", None, patient.restriction)
        ),
    )
    # the quantifier is the subject's determiner, and the generic is a bare noun
    generic = dataclasses.replace(proposition, quantifier=GENERIC)
    assert plan_for(facts, generic).subject.determiner is None
    assert plan_for(facts, generic).proposition().quantifier == GENERIC
    rule = facts.rule_statements()[0]
    assert plan_for(facts, rule).subject.noun == "THING"
    assert plan_for(facts, rule).subject.determiner == ALL


def test_instances_are_mentioned_by_their_leaf_unless_told_otherwise(cases) -> None:
    facts = cases("default").facts()
    instance, other = "I1.2.1.3", "I2.1.1.1"
    assert leaf_of(facts, instance) == "C1.2.1"
    assert mention(facts, instance) == NounPhrase(INSTANCE_NP, instance, "C1.2.1", "the")
    capacity = Proposition(INSTANCE, instance, Predicate(VERB, "V1.1", other))
    plan = plan_for(facts, capacity)
    assert plan.subject == mention(facts, instance)
    assert plan.predication.object == mention(facts, other)
    # a caller can give any other mention: a higher noun, "a", modifiers, or a pronoun
    given = {
        instance: NounPhrase(INSTANCE_NP, instance, "C1", "a", (Literal("IS.3"),)),
        other: NounPhrase(INSTANCE_NP, other),
    }
    plan = plan_for(facts, capacity, given)
    assert plan.subject == given[instance] and plan.predication.object == given[other]
    assert plan.proposition() == capacity
    # a scalar pole's subject is named by the comparison class
    for comparison in ("C1", "C1.2", "C1.2.1"):
        pole = Proposition(
            INSTANCE, instance, Predicate(SCALAR, "SC.1.HIGH", comparison=comparison)
        )
        plan = plan_for(facts, pole)
        assert plan.subject.noun == comparison and plan.proposition() == pole


def test_a_membership_sentence_never_names_the_subject_by_the_predicate(cases) -> None:
    facts = cases("default").facts()
    instance = "I1.2.1.3"
    plans = {
        category: plan_for(facts, Proposition(INSTANCE, instance, Predicate(MEMBER, category)))
        for category in ("C1", "C1.2", "C1.2.1")
    }
    assert plans["C1"].subject.noun == "C1.2.1"  # "the penguin is an animal"
    assert plans["C1.2"].subject.noun == "C1.2.1"
    assert plans["C1.2.1"].subject.noun == "C1.2"  # "the bird is a penguin"
    for plan in plans.values():
        check_plan(plan)
    # a world of one level has no category above the leaf: the subject is a pronoun
    flat = cases("tiny").facts()
    still = cases("still").facts()
    assert (
        plan_for(flat, Proposition(INSTANCE, "I1.1.1", Predicate(MEMBER, "C1.1"))).subject.noun
        == "C1"
    )
    for facts_of in (flat, still):
        top = Proposition(INSTANCE, facts_of.result.instances.labels[0], Predicate(MEMBER, "C1"))
        assert plan_for(facts_of, top).subject.noun is not None


# ---------------------------------------------------------------------------------------------
# Relative clauses
# ---------------------------------------------------------------------------------------------


def attach_all(case, settings: dict, level: str, seed: int = 0, count: int = 400):
    """Plans of one level with relative clauses drawn, and the facts they were drawn from."""
    facts = case.facts(**settings)
    clauses = RelativeClauses(case.config(**settings), facts)
    rng = Streams(seed).mentions
    labels = case.result.instances.labels
    others = labels[:: max(1, len(labels) // 7)][:7]
    plans = []
    if level == INSTANCE:
        pool = [f for i in others[:4] for f in facts.instance_facts(i, patients=others)]
        for index in rng.choice(len(pool), size=count):
            plans.append(clauses.attach(rng, plan_for(facts, pool[int(index)]), others))
    else:
        events = [e for scene in scenes_of(case, facts) for e in scene.events]
        for event in events[:count]:
            plans.append(clauses.attach(rng, plan_for(facts, facts.event_fact(event))))
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
    plans, facts = attach_all(case, rate_settings(rate=0.8, max_depth=2), EVENT)
    subject_relatives = object_relatives = 0
    for plan in plans:
        scene = plan.predication.event.rsplit(".", 1)[0]
        events = {e.label: e for e in facts.truth.scenes[scene].events}
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
    # a class-level sentence: what a clause there would mean is not settled
    for fact in facts.class_facts("C1.1", patients=("C2",))[:40]:
        plan = plan_for(facts, fact)
        assert clauses.attach(rng, plan) == plan
    # a pronoun, and a noun phrase that already has a clause
    instance = case.result.instances.labels[0]
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
    labels = case.result.instances.labels
    instance = labels[0]
    row = facts.truth.instance_index[instance]
    lacking = next(
        f
        for f in facts.features["is"]
        if not facts.truth.values[row, facts.truth.features[f].position]
    )
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
            "class_can_rate": 0.9,
            "adjective_order": {"fixed": False},
        }
    }
    changed, _ = attach_all(
        case, {**rate_settings(rate=0.6, max_depth=2), **grammar}, EVENT, seed=3
    )
    assert changed == base
