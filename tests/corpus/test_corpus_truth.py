"""Stage 2 and stage a5a: the truth tests, against an independent recomputation from the world
run's files.

``truth_oracle.Oracle`` reads a world run folder and judges a logical form without any of the
corpus generator's code, with the brute-force evaluator for the requirements. Here every logical
form of a whole grid (every subject, predicate, quantifier, and polarity of a small world) is
judged both ways. The other tests pin down single rules of "Quantifiers" in the specification:
``nec_all`` and ``nec_no`` by the fixed test, ``all`` and ``no`` over the members, ``most`` as
more than half, and the language's words as a matter of felicity, never of truth.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from semantic_world.corpus.lexicon import THING
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    EXACT,
    HAS,
    INSTANCE,
    IS,
    LOCAL,
    MEAN,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NO,
    OBSERVED,
    PROJECTION,
    QUANTIFIERS,
    SCALAR,
    SOME,
    TREE,
    VALUE,
    VERB,
    CategoryTerm,
    Clause,
    Literal,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.world.labels import translate

SMALL = ("tiny", "deep", "still")
SETTINGS = {
    "default": {},
    "extensional": {"quantifiers": {"universal_words": "extensional"}},
    "wide_poles": {"scalar_adjectives": {"z": 0.3}},
}
KIND_OF = {"PROPERTY": IS, "PART": HAS, "EVENTTYPE1": CAN}


def agree(case, settings: dict, propositions) -> int:
    """Check that the engine and the oracle judge every logical form alike. Returns how many
    were valid."""
    facts = case.facts(**settings)
    oracle = case.oracle(**settings)
    valid = 0
    for proposition in propositions:
        evaluation = facts.truth.evaluate(proposition)
        expected = oracle.truth(proposition.to_json())
        form = proposition.to_json()
        if expected is None:
            assert not evaluation.valid, form
        else:
            assert evaluation.valid, (form, evaluation.reason)
            assert evaluation.true == expected, form
            valid += 1
    return valid


def quantifiers_of(predicate: Predicate) -> tuple:
    return (None,) if predicate.kind == SCALAR else QUANTIFIERS


def class_grid(facts, restrictions=((),)):
    subjects = [CategoryTerm(c, r) for c in facts.categories + (THING,) for r in restrictions]
    for subject in subjects:
        for predicate in facts.class_predicates(subject.category):
            for quantifier, polarity in itertools.product(quantifiers_of(predicate), (True, False)):
                yield Proposition(CLASS, subject, predicate, polarity, quantifier)


def instance_grid(facts):
    for instance in facts.world.instances:
        for predicate in facts.instance_predicates(instance):
            for polarity in (True, False):
                yield Proposition(INSTANCE, instance, predicate, polarity)


# ---------------------------------------------------------------------------------------------
# The whole grid
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("setting", list(SETTINGS))
def test_every_class_level_form_is_judged_as_the_output_files_say(cases, name, setting) -> None:
    case = cases(name)
    valid = agree(case, SETTINGS[setting], class_grid(case.facts(**SETTINGS[setting])))
    assert valid > 1000


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("setting", ["default", "wide_poles"])
def test_every_instance_level_form_is_judged_as_the_output_files_say(cases, name, setting) -> None:
    case = cases(name)
    valid = agree(case, SETTINGS[setting], instance_grid(case.facts(**SETTINGS[setting])))
    assert valid > 1000


def restrictions_of(facts, rng: np.random.Generator, count: int) -> list[tuple[Literal, ...]]:
    """Random restrictions of one to three literals: PROPERTY and PART features of both
    polarities, free and determined, and scalar poles."""
    features = facts.features[IS] + facts.features[HAS]
    drawn = []
    for _ in range(count):
        size = int(rng.integers(1, 4))
        literals = [
            Literal(features[int(i)], bool(rng.integers(2)))
            for i in rng.choice(len(features), size=size, replace=False)
        ]
        if facts.poles and rng.random() < 0.3:
            literals[0] = Literal(facts.poles[int(rng.integers(len(facts.poles)))])
        drawn.append(tuple(literals))
    return drawn


@pytest.mark.parametrize("name", SMALL)
def test_restricted_subjects_are_judged_as_the_output_files_say(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    restrictions = restrictions_of(facts, np.random.default_rng(11), 12)
    valid = agree(case, {}, class_grid(facts, restrictions))
    assert valid > 1000  # many restricted subject sets are empty, and so vacuous


def test_restricted_patients_are_judged_as_the_output_files_say(cases) -> None:
    case = cases("deep")
    facts = case.facts()
    rng = np.random.default_rng(5)
    forms = []
    for restriction in restrictions_of(facts, rng, 10):
        for agent, patient, verb in itertools.product(
            facts.categories[:4], facts.categories, facts.verbs
        ):
            predicate = Predicate(VERB, verb, CategoryTerm(patient, restriction))
            for quantifier in QUANTIFIERS:
                forms.append(Proposition(CLASS, CategoryTerm(agent), predicate, True, quantifier))
    assert agree(case, {}, forms) > 500


# ---------------------------------------------------------------------------------------------
# Relative clauses
# ---------------------------------------------------------------------------------------------


def clauses_of(facts, rng: np.random.Generator, count: int) -> list[Clause]:
    """Random relative clauses of the three kinds: a one-place event type, a two-place event
    type with a patient category (a subject relative), and one with an agent category (an object
    relative). The other category sometimes has a restriction, or a relative clause of its
    own."""
    cans, verbs, categories = facts.features[CAN], facts.verbs, facts.categories
    features = facts.features[IS] + facts.features[HAS]

    def other() -> CategoryTerm:
        category = categories[int(rng.integers(len(categories)))]
        kind = rng.random()
        if kind < 0.25:
            feature = features[int(rng.integers(len(features)))]
            return CategoryTerm(category, (Literal(feature, bool(rng.integers(2))),))
        if kind < 0.5:
            return CategoryTerm(category, (), (clause(nested=False),))
        return CategoryTerm(category)

    def clause(nested: bool = True) -> Clause:
        kind = int(rng.integers(3))
        if kind == 0 or not verbs:
            return Clause(CAN, cans[int(rng.integers(len(cans)))])
        verb = verbs[int(rng.integers(len(verbs)))]
        term = other() if nested else CategoryTerm(categories[int(rng.integers(len(categories)))])
        return Clause(VERB, verb, patient=term) if kind == 1 else Clause(VERB, verb, agent=term)

    return [clause() for _ in range(count)]


@pytest.mark.parametrize("name", SMALL)
def test_subjects_with_relative_clauses_are_judged_as_the_output_files_say(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    rng = np.random.default_rng(21)
    clauses = clauses_of(facts, rng, 10)
    assert {(c.kind, c.patient is None, c.agent is None) for c in clauses} == {
        (CAN, True, True),
        (VERB, False, True),
        (VERB, True, False),
    }
    restrictions = [(), *restrictions_of(facts, rng, 2)]
    forms = []
    for category in facts.categories:
        for clause in clauses:
            restriction = restrictions[int(rng.integers(len(restrictions)))]
            subject = CategoryTerm(category, restriction, (clause,))
            for predicate in facts.class_predicates(category, patients=facts.categories[:3]):
                for quantifier, polarity in itertools.product(
                    quantifiers_of(predicate), (True, False)
                ):
                    forms.append(Proposition(CLASS, subject, predicate, polarity, quantifier))
    assert agree(case, {}, forms) > 1000


def test_patients_with_relative_clauses_are_judged_as_the_output_files_say(cases) -> None:
    case = cases("deep")
    facts = case.facts()
    rng = np.random.default_rng(6)
    forms = []
    for clause in clauses_of(facts, rng, 10):
        for agent, patient, verb in itertools.product(
            facts.categories[:4], facts.categories, facts.verbs
        ):
            predicate = Predicate(VERB, verb, CategoryTerm(patient, (), (clause,)))
            for quantifier in QUANTIFIERS:
                forms.append(Proposition(CLASS, CategoryTerm(agent), predicate, True, quantifier))
    assert agree(case, {}, forms) > 500


def test_a_relative_clause_is_restrictive_and_means_at_least_one(cases) -> None:
    case = cases("default")
    facts = case.facts()
    truth = facts.truth
    labels = case.world.instances
    checked = 0
    for verb in facts.verbs:
        holds = truth.matrix(verb)
        for agent, patient in itertools.product(facts.categories[::7], facts.categories[::5]):
            agents = truth.members(CategoryTerm(agent))
            patients = truth.members(CategoryTerm(patient))
            # "owls that eat mice": the owls that can eat at least one mouse
            eaters = truth.members(
                CategoryTerm(agent, (), (Clause(VERB, verb, patient=CategoryTerm(patient)),))
            )
            expected = [a for a in agents if any(holds[a, p] for p in patients if p != a)]
            assert list(eaters) == expected
            # "mice that owls eat": the mice that at least one owl can eat
            eaten = truth.members(
                CategoryTerm(patient, (), (Clause(VERB, verb, agent=CategoryTerm(agent)),))
            )
            assert list(eaten) == [
                p for p in patients if any(holds[a, p] for a in agents if a != p)
            ]
            checked += bool(expected)
    assert checked > 10
    # "penguins that can swim": the members able to be the agent of the one-place event type
    category, feature = facts.categories[0], facts.features[CAN][0]
    swimmers = truth.members(CategoryTerm(category, (), (Clause(CAN, feature),)))
    column = case.world.column(feature)
    assert [labels[i] for i in swimmers] == [
        labels[i] for i in truth.members(CategoryTerm(category)) if column[i]
    ]


def restricted_subject(facts, minimum: int = 3):
    """A category term with a relative clause that some of its members satisfy, and not all,
    and a PROPERTY, PART, or one-place feature that every member of the restricted set has."""
    truth, world = facts.truth, facts.world
    for category in facts.categories:
        for object_relative in (False, True):
            for clause in facts.clause_options(CategoryTerm(category), object_relative):
                term = CategoryTerm(category, (), (clause,))
                members = truth.members(term)
                if len(members) < minimum:
                    continue
                for kind in (IS, HAS, CAN):
                    for feature in facts.features[kind]:
                        column = world.column(feature)[members]
                        if column.all() and feature != clause.label:
                            return term, Predicate(kind, feature)
    raise AssertionError("no restricted subject with a feature that every member has")


def test_a_subject_with_a_relative_clause_takes_the_extensional_quantifiers(cases) -> None:
    case = cases("default")
    facts = case.facts()
    subject, predicate = restricted_subject(facts)
    truth = facts.truth

    def judge(quantifier: str, polarity: bool = True, truth=truth):
        return truth.evaluate(Proposition(CLASS, subject, predicate, polarity, quantifier))

    # most, some, all, and no are judged by the share of the subject set
    most = judge(MOST)
    assert most.valid and most.true
    assert most.grounding["proportion"] == 1.0 and most.grounding["test"] == OBSERVED
    assert most.grounding["instances"] == len(truth.members(subject))
    assert "fixed" not in most.grounding
    assert judge(ALL).true and not judge(NO).true and not judge(MOST, False).true
    # nec_all and nec_no need a fixed test, which a subject with a relative clause has not
    for quantifier in (NEC_ALL, NEC_NO):
        evaluation = judge(quantifier)
        assert not evaluation.valid and "fixed test" in evaluation.reason
    # under the default words, "all" expresses nec_all only, so a document states "most"
    stated = facts.class_fact(subject, predicate)
    assert stated.quantifier == MOST and stated.subject == subject
    # with extensional words, "all" can be said
    extensional = case.facts(quantifiers={"universal_words": "extensional"})
    assert extensional.class_fact(subject, predicate).quantifier == ALL
    # every member has the feature, so "some" is true; under the default words it is still
    # usable, because the language's "all" (nec_all) is false, and under extensional words it
    # is left out by the implicature
    assert judge(SOME).true and judge(SOME).felicitous
    assert not judge(SOME, truth=extensional.truth).felicitous
    # a clause that no member satisfies makes a vacuous subject
    nobody = CategoryTerm(
        subject.category, (), (Clause(VERB, facts.verbs[0], patient=subject), subject.clauses[0])
    )
    if len(truth.members(nobody)) == 0:
        assert not truth.evaluate(Proposition(CLASS, nobody, predicate, True, MOST)).valid


def test_relative_clauses_in_the_logical_form() -> None:
    mice = CategoryTerm("CATEGORY.1.5", (Literal("PROPERTY.4"),))
    owls = CategoryTerm(
        "CATEGORY.1.2",
        (),
        (Clause(VERB, "EVENTTYPE2.2.1", patient=mice), Clause(CAN, "EVENTTYPE1.3")),
    )
    assert owls.to_json() == {
        "category": "CATEGORY.1.2",
        "restriction": [],
        "clauses": [
            {
                "kind": "verb",
                "verb": "EVENTTYPE2.2.1",
                "patient": {"category": "CATEGORY.1.5", "restriction": ["PROPERTY.4"]},
            },
            {"kind": "can", "feature": "EVENTTYPE1.3"},
        ],
    }
    assert CategoryTerm.from_json(owls.to_json()) == owls
    eaten = CategoryTerm(
        "CATEGORY.1.5", (), (Clause(VERB, "EVENTTYPE2.2.1", agent=CategoryTerm("CATEGORY.1.2")),)
    )
    assert eaten.to_json()["clauses"] == [
        {
            "kind": "verb",
            "verb": "EVENTTYPE2.2.1",
            "agent": {"category": "CATEGORY.1.2", "restriction": []},
        }
    ]
    assert CategoryTerm.from_json(eaten.to_json()) == eaten
    assert eaten != CategoryTerm("CATEGORY.1.5") and eaten.plain == CategoryTerm("CATEGORY.1.5")
    # a term without clauses is written as before
    assert "clauses" not in mice.to_json()
    proposition = Proposition(CLASS, owls, Predicate(HAS, "PART.2"), True, MOST)
    assert Proposition.from_json(proposition.to_json()) == proposition
    # the words a sentence needs: the clauses' event types, features, and categories too
    assert proposition.concepts() == (
        "CATEGORY.1.2",
        "EVENTTYPE2.2.1",
        "CATEGORY.1.5",
        "PROPERTY.4",
        "EVENTTYPE1.3",
        "PART.2",
    )


def test_relative_clauses_that_cannot_be_judged(tiny) -> None:
    truth = tiny.facts().truth
    bad = {
        "is not a one-place event type": Clause(CAN, "PROPERTY.1"),
        "has no other category": Clause(CAN, "EVENTTYPE1.1", patient=CategoryTerm("CATEGORY.1")),
        "unknown two-place event type": Clause(
            VERB, "EVENTTYPE2.9", patient=CategoryTerm("CATEGORY.1")
        ),
        "a patient category or an agent category": Clause(VERB, "EVENTTYPE2.1.1"),
        "a patient category or an agent category ": Clause(
            VERB, "EVENTTYPE2.1.1", CategoryTerm("CATEGORY.1"), CategoryTerm("CATEGORY.2")
        ),
        "unknown category": Clause(VERB, "EVENTTYPE2.1.1", patient=CategoryTerm("CATEGORY.9")),
        "not about the generic noun": Clause(VERB, "EVENTTYPE2.1.1", agent=CategoryTerm(THING)),
        "holds a one-place or a two-place event type": Clause(IS, "PROPERTY.1"),
    }
    for reason, clause in bad.items():
        subject = CategoryTerm("CATEGORY.1.1", (), (clause,))
        evaluation = truth.evaluate(
            Proposition(CLASS, subject, Predicate(HAS, "PART.1"), True, MOST)
        )
        assert not evaluation.valid and reason.strip() in evaluation.reason, reason


# ---------------------------------------------------------------------------------------------
# The fixed test
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "default", "deep", "still"])
def test_the_fixed_test_matches_the_defining_vectors(cases, name) -> None:
    # For a category without a restriction, the fixed test is the taxonomy generator's own:
    # categories_defining.csv holds the value of every fixed feature, and NaN elsewhere.
    case = cases(name)
    truth = case.facts().truth
    vectors = case.world.result.taxonomy.vectors
    labels = [translate(x) for x in vectors.feature_labels[vectors.isa_count :]]
    for row, category in enumerate(case.world.categories):
        defining = vectors.defining[row, vectors.isa_count :]
        for label, value in zip(labels, defining, strict=True):
            fixed, test = truth.fixed(CategoryTerm(category), label)
            assert test == EXACT
            assert fixed == (None if np.isnan(value) else int(value)), (category, label)


@pytest.mark.parametrize("name", ["tiny", "deep", "still"])
def test_the_local_test_never_marks_a_feature_fixed_when_it_is_not(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    exact = facts.truth
    local = Truth(case.config(), case.world, cone_limit=-1)  # nothing is small enough to enumerate
    restrictions = [()] + restrictions_of(facts, np.random.default_rng(3), 15)
    features = facts.features[IS] + facts.features[HAS] + facts.features[CAN]
    found = missed = 0
    for category in facts.categories + (THING,):
        for restriction in restrictions:
            term = CategoryTerm(category, restriction)
            if len(exact.members(term)) == 0:
                continue
            for feature in features:
                value, test = local.fixed(term, feature)
                assert test == LOCAL
                if value is not None:
                    assert exact.fixed(term, feature) == (value, EXACT), (term, feature)
                    found += 1
                elif exact.fixed(term, feature)[0] is not None:
                    missed += 1
    assert found > 50  # the local test does find fixed features
    assert missed > 0  # and it misses some: that is the price of not enumerating


def test_a_threshold_literal_is_held_when_the_scalar_never_drifts(cases) -> None:
    # In the still world every member of a category has the category's scalar value, so a rule
    # that reads a threshold can be fixed at a category and not for things in general.
    case = cases("still")
    truth = case.facts().truth
    taxonomy = case.world.result.taxonomy
    reading = [translate(r.output.label) for r in taxonomy.rules.rules if r.thresholds]
    assert reading
    fixed_somewhere = 0
    for feature in reading:
        for category in case.world.categories:
            value, _ = truth.fixed(CategoryTerm(category), feature)
            fixed_somewhere += value is not None
    assert fixed_somewhere > 0
    oracle = case.oracle()
    for feature in reading:
        for category in [*case.world.categories, THING]:
            term = CategoryTerm(category)
            assert truth.fixed(term, feature)[0] == oracle.fixed(term.to_json(), feature)


# ---------------------------------------------------------------------------------------------
# Single rules of the truth grounding, on the tiny world
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def tiny(cases):
    return cases("tiny")


def proportion(case, category: str, feature: str) -> float:
    truth = case.facts().truth
    members = truth.members(CategoryTerm(category))
    return float(case.world.column(feature)[members].mean())


def find(case, low: float, high: float, fixed: bool | None = None) -> tuple[str, str]:
    """A category and a PROPERTY or PART feature whose proportion lies in a range."""
    facts = case.facts()
    for category in facts.categories:
        for feature in facts.features[IS] + facts.features[HAS]:
            is_fixed = facts.truth.fixed(CategoryTerm(category), feature)[0] is not None
            if low <= proportion(case, category, feature) <= high and fixed in (None, is_fixed):
                return category, feature
    raise AssertionError("the world has no such fact")


def judge(case, category, feature, quantifier, polarity=True, **settings):
    kind = KIND_OF[feature.split(".")[0]]
    proposition = Proposition(
        CLASS, CategoryTerm(category), Predicate(kind, feature), polarity, quantifier
    )
    return case.facts(**settings).truth.evaluate(proposition)


def test_nec_all_is_a_law_and_all_an_observation(tiny) -> None:
    # every instance has the feature, but nothing fixes it: true as all, false as nec_all
    category, feature = find(tiny, 1.0, 1.0, fixed=False)
    law = judge(tiny, category, feature, NEC_ALL)
    assert law.valid and not law.true
    assert law.grounding["proportion"] == 1.0 and law.grounding["fixed"] is False
    assert law.grounding["test"] == EXACT
    seen = judge(tiny, category, feature, ALL)
    assert seen.true and seen.grounding["test"] == OBSERVED
    # a fixed feature is true both ways: nec_all implies all
    category, feature = find(tiny, 1.0, 1.0, fixed=True)
    assert judge(tiny, category, feature, NEC_ALL).true
    assert judge(tiny, category, feature, ALL).true
    # and nec_no is the same test at the value 0
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    assert judge(tiny, category, feature, NEC_NO).true and judge(tiny, category, feature, NO).true
    assert not judge(tiny, category, feature, NEC_ALL).true
    category, feature = find(tiny, 0.0, 0.0, fixed=False)
    assert not judge(tiny, category, feature, NEC_NO).true
    assert judge(tiny, category, feature, NO).true


def test_the_words_change_felicity_and_never_truth(tiny) -> None:
    category, feature = find(tiny, 1.0, 1.0, fixed=False)
    for setting in ("nec", "extensional", "either"):
        words = {"quantifiers": {"universal_words": setting}}
        assert judge(tiny, category, feature, ALL, **words).true
        assert not judge(tiny, category, feature, NEC_ALL, **words).true
    # "all" expresses all under extensional and either words, and never under nec
    assert not judge(tiny, category, feature, ALL).felicitous
    extensional = {"quantifiers": {"universal_words": "extensional"}}
    either = {"quantifiers": {"universal_words": "either"}}
    assert judge(tiny, category, feature, ALL, **extensional).felicitous
    assert judge(tiny, category, feature, ALL, **either).felicitous
    category, feature = find(tiny, 1.0, 1.0, fixed=True)
    assert judge(tiny, category, feature, NEC_ALL).felicitous
    assert not judge(tiny, category, feature, NEC_ALL, **extensional).felicitous
    assert judge(tiny, category, feature, NEC_ALL, **either).felicitous
    # the bare plural can say what the words cannot
    truth = tiny.facts(quantifiers={"bare_plural": {"expresses": ["all"]}}).truth
    kind = KIND_OF[feature.split(".")[0]]
    proposition = Proposition(CLASS, CategoryTerm(category), Predicate(kind, feature), True, ALL)
    assert truth.evaluate(proposition).felicitous and truth.bare_plural_expresses(proposition)
    assert ALL not in truth.sayable and truth.statable(proposition)


def test_most_means_more_than_half_and_the_usage_rule_is_felicity(tiny) -> None:
    category, feature = find(tiny, 0.6, 0.69)  # 2 of 3 instances
    most = judge(tiny, category, feature, MOST)
    assert most.true and not most.felicitous  # true above one half, said at or above 0.7
    lower = judge(tiny, category, feature, MOST, quantifiers={"most": {"usage_min": 0.6}})
    assert lower.true and lower.felicitous
    category, feature = find(tiny, 0.4, 0.5)
    assert not judge(tiny, category, feature, MOST).true  # exactly half is not most
    # most of the others lack it: the negative proposition is about the share without it
    category, feature = find(tiny, 0.1, 0.3)
    assert judge(tiny, category, feature, MOST, polarity=False).true
    assert not judge(tiny, category, feature, MOST).true


def test_some_is_true_above_zero_and_used_only_when_the_languages_all_is_false(tiny) -> None:
    category, feature = find(tiny, 0.1, 0.9)
    some = judge(tiny, category, feature, SOME)
    assert some.true and some.felicitous
    category, feature = find(tiny, 0.0, 0.0)
    assert not judge(tiny, category, feature, SOME).true
    # every member has the fixed feature: "some" is true, and left out by the implicature
    category, feature = find(tiny, 1.0, 1.0, fixed=True)
    some = judge(tiny, category, feature, SOME)
    assert some.true and not some.felicitous
    kept = judge(tiny, category, feature, SOME, quantifiers={"some": {"exclude_all": False}})
    assert kept.true and kept.felicitous
    # the implicature follows the language's "all": seen in all, but not fixed, "some" is
    # usable under nec words and left out under extensional words
    category, feature = find(tiny, 1.0, 1.0, fixed=False)
    assert judge(tiny, category, feature, SOME).felicitous
    extensional = {"quantifiers": {"universal_words": "extensional"}}
    assert not judge(tiny, category, feature, SOME, **extensional).felicitous
    # "some ... not" is left out when none has the feature by law
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    lacking = judge(tiny, category, feature, SOME, polarity=False)
    assert lacking.true and not lacking.felicitous


def test_the_bare_plural_expresses_what_the_setting_says(tiny) -> None:
    truth = tiny.facts().truth
    category, feature = find(tiny, 0.1, 0.5)
    kind = KIND_OF[feature.split(".")[0]]

    def proposition(quantifier, polarity=True, **extra):
        return Proposition(
            CLASS, CategoryTerm(category), Predicate(kind, feature), polarity, quantifier, **extra
        )

    assert truth.bare_plural_expresses(proposition(MOST))
    assert truth.bare_plural_expresses(proposition(MOST, False))
    assert not truth.bare_plural_expresses(proposition(SOME))
    assert not truth.bare_plural_expresses(proposition(NEC_ALL))
    assert not truth.bare_plural_expresses(proposition(NO))
    wider = tiny.facts(quantifiers={"bare_plural": {"expresses": ["nec_all", "some"]}}).truth
    assert wider.bare_plural_expresses(proposition(NEC_ALL))
    assert wider.bare_plural_expresses(proposition(NEC_NO))
    assert wider.bare_plural_expresses(proposition(SOME, False))
    assert not wider.bare_plural_expresses(proposition(MOST))
    # membership and rule statements may always use the bare plural, for nec_all
    membership = Proposition(
        CLASS, CategoryTerm("CATEGORY.1.1"), Predicate(MEMBER, "CATEGORY.1"), True, NEC_ALL
    )
    assert truth.bare_plural_expresses(membership)
    assert truth.bare_plural_expresses(proposition(NEC_ALL, rule=(feature, 1)))
    # a class-level scalar pole has no quantifier word
    pole = Proposition(CLASS, CategoryTerm(category), Predicate(SCALAR, "SCALARDIM.1.HIGH"))
    assert truth.bare_plural_expresses(pole)


def test_no_and_nec_no_replace_sentence_negation(tiny) -> None:
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    for quantifier in (ALL, NEC_ALL, NO, NEC_NO):
        denied = judge(tiny, category, feature, quantifier, polarity=False)
        assert not denied.valid and "negative polarity" in denied.reason
    proposition = Proposition(CLASS, CategoryTerm(category), Predicate(IS, "PROPERTY.1"), True, NO)
    assert proposition.negative
    assert dataclasses.replace(proposition, quantifier=NEC_NO).negative
    assert not dataclasses.replace(proposition, quantifier=ALL).negative
    assert dataclasses.replace(proposition, quantifier=MOST, polarity=False).negative


def test_a_vacuous_proposition_is_never_valid(tiny) -> None:
    facts = tiny.facts()
    truth = facts.truth
    category, feature = find(tiny, 0.0, 0.0)
    empty = CategoryTerm(category, (Literal(feature),))
    assert len(truth.members(empty)) == 0
    for quantifier, polarity in itertools.product(QUANTIFIERS, (True, False)):
        for predicate in facts.class_predicates(category):
            if predicate.kind == SCALAR:
                continue
            vacuous = Proposition(CLASS, empty, predicate, polarity, quantifier)
            evaluation = truth.evaluate(vacuous)
            assert not evaluation.valid and truth.grounded(vacuous) is None
    # the same holds for an empty set of pairs: one instance cannot be related to itself
    verb = facts.verbs[0]
    single = None
    for candidate in facts.features[IS] + facts.features[HAS]:
        term = CategoryTerm(THING, (Literal(candidate),))
        if len(truth.members(term)) == 1:
            single = term
            break
    if single is not None:
        alone = Proposition(CLASS, single, Predicate(VERB, verb, single), True, SOME)
        assert truth.evaluate(alone).reason == "there is no pair of distinct instances"


def test_membership(tiny) -> None:
    truth = tiny.facts().truth

    def member(subject, category, quantifier=NEC_ALL, polarity=True):
        return truth.evaluate(
            Proposition(
                CLASS, CategoryTerm(subject), Predicate(MEMBER, category), polarity, quantifier
            )
        )

    assert member("CATEGORY.1.1", "CATEGORY.1").true
    assert member("CATEGORY.1.1", "CATEGORY.1").grounding["test"] == TREE
    assert member("CATEGORY.1.1", "CATEGORY.1", ALL).true  # nec_all implies all
    assert not member("CATEGORY.1.1", "CATEGORY.2").true
    assert not member("CATEGORY.1", "CATEGORY.1.1").true
    assert member("CATEGORY.1.1", "CATEGORY.2", NEC_NO).true
    assert member("CATEGORY.1.1", "CATEGORY.2", NO).true
    assert member("CATEGORY.1.1", "CATEGORY.1.2", NEC_NO).true  # siblings share no instance
    assert not member("CATEGORY.1.1", "CATEGORY.1", NEC_NO).true
    assert not member("CATEGORY.1", "CATEGORY.1.1", NO).true
    for quantifier in (MOST, SOME):
        assert not member("CATEGORY.1.1", "CATEGORY.1", quantifier).valid
    assert not member("CATEGORY.1", "CATEGORY.1").valid
    assert not member(THING, "CATEGORY.1").valid
    # a document states membership as nec_all; with extensional words, the bare plural still can
    facts = tiny.facts()
    fact = facts.class_fact(CategoryTerm("CATEGORY.1.1"), Predicate(MEMBER, "CATEGORY.1"))
    assert fact.quantifier == NEC_ALL
    extensional = tiny.facts(quantifiers={"universal_words": "extensional"})
    stated = extensional.class_fact(CategoryTerm("CATEGORY.1.1"), Predicate(MEMBER, "CATEGORY.1"))
    assert stated.quantifier == NEC_ALL and extensional.truth.bare_plural_expresses(stated)
    # a restriction does not change membership
    restricted = Proposition(
        CLASS,
        CategoryTerm("CATEGORY.1.1", (Literal("PROPERTY.2"),)),
        Predicate(MEMBER, "CATEGORY.1"),
        True,
        NEC_ALL,
    )
    assert truth.evaluate(restricted).true == (len(truth.members(restricted.subject)) > 0)


def test_class_level_scalar_poles(cases) -> None:
    case = cases("default")
    facts = case.facts()
    truth = facts.truth
    world = case.world
    scalars = world.scalar_values

    def pole(category, label, quantifier=None, polarity=True, restriction=(), **settings):
        proposition = Proposition(
            CLASS,
            CategoryTerm(category, restriction),
            Predicate(SCALAR, label),
            polarity,
            quantifier,
        )
        return case.facts(**settings).truth.evaluate(proposition)

    # no quantifier, and a statement about the whole category
    for quantifier in QUANTIFIERS:
        assert not pole("CATEGORY.1.1", "SCALARDIM.1.HIGH", quantifier).valid
    assert not pole(THING, "SCALARDIM.1.HIGH").valid
    restricted = pole("CATEGORY.1.1", "SCALARDIM.1.HIGH", restriction=(Literal("PROPERTY.1"),))
    assert not restricted.valid
    for category in facts.categories:
        parent = world.category[category].parent
        members = truth.members(CategoryTerm(category))
        # a top-level category is compared with all instances in the world
        comparison = (
            np.arange(len(scalars)) if parent is None else truth.members(CategoryTerm(parent))
        )
        for index, scalar in enumerate(("SCALARDIM.1", "SCALARDIM.2")):
            mean, sd = scalars[comparison, index].mean(), scalars[comparison, index].std()
            value = scalars[members, index].mean()
            high = pole(category, f"{scalar}.HIGH")
            low = pole(category, f"{scalar}.LOW")
            assert high.true == (value >= mean + sd) and low.true == (value <= mean - sd)
            assert not (high.true and low.true)
            assert high.grounding["test"] == MEAN
            assert high.grounding["comparison"] == (THING if parent is None else parent)
            assert pole(category, f"{scalar}.HIGH", polarity=False).true == (not high.true)

    # a smaller z makes more categories big or small
    def count(z: float) -> int:
        return sum(
            pole(c, f"SCALARDIM.1.{side}", scalar_adjectives={"z": z}).true
            for c in facts.categories
            for side in ("HIGH", "LOW")
        )

    assert count(2.0) <= count(1.0) < count(0.3)
    assert count(1.0) > 0


def test_instance_level_scalar_poles_compare_with_the_nouns_category(cases) -> None:
    case = cases("default")
    truth = case.facts(scalar_adjectives={"z": 0.5}).truth
    scalars = case.world.scalar_values
    differ = 0
    for instance in case.world.instances[::7]:
        row = truth.instance_index[instance]
        path = truth.paths[row]
        verdicts = []
        for comparison in path:
            proposition = Proposition(
                INSTANCE, instance, Predicate(SCALAR, "SCALARDIM.1.HIGH", comparison=comparison)
            )
            evaluation = truth.evaluate(proposition)
            members = truth.members(CategoryTerm(comparison))
            mean, sd = scalars[members, 0].mean(), scalars[members, 0].std()
            assert evaluation.true == (scalars[row, 0] >= mean + 0.5 * sd)
            assert evaluation.grounding["test"] == VALUE
            verdicts.append(evaluation.true)
        differ += len(set(verdicts)) > 1
        # the comparison class must be a category the instance is below
        other = next(c for c in truth.categories if c not in path)
        outside = Proposition(
            INSTANCE, instance, Predicate(SCALAR, "SCALARDIM.1.HIGH", comparison=other)
        )
        assert not truth.evaluate(outside).valid
        assert not truth.evaluate(
            Proposition(INSTANCE, instance, Predicate(SCALAR, "SCALARDIM.1.HIGH"))
        ).valid
    assert differ > 0  # "the big mouse" can be "the small animal"


def test_a_pole_in_a_restriction_is_relative_to_the_category(cases) -> None:
    case = cases("default")
    truth = case.facts().truth
    scalars = case.world.scalar_values[:, 0]
    for category in ("CATEGORY.1", "CATEGORY.2.1", "CATEGORY.3.2.1"):
        members = truth.members(CategoryTerm(category))
        big = truth.members(CategoryTerm(category, (Literal("SCALARDIM.1.HIGH"),)))
        cut = scalars[members].mean() + scalars[members].std()
        assert set(big) == {int(i) for i in members if scalars[i] >= cut}
    negated = Proposition(
        CLASS,
        CategoryTerm("CATEGORY.1", (Literal("SCALARDIM.1.HIGH", False),)),
        Predicate(IS, "PROPERTY.1"),
        True,
        SOME,
    )
    assert "never negated" in truth.evaluate(negated).reason


def test_patient_capacities_at_the_class_level(tiny) -> None:
    facts = tiny.facts(lexicon={"named_proportion": {"patient_projection": 1.0}})
    world = tiny.world
    assert len(facts.projections) == 4
    projection = facts.projections[-1]
    values = world.capacity(projection)

    def judge_projection(category, quantifier, polarity=True):
        proposition = Proposition(
            CLASS, CategoryTerm(category), Predicate(PROJECTION, projection), polarity, quantifier
        )
        return facts.truth.evaluate(proposition)

    for category in facts.categories:
        members = facts.truth.members(CategoryTerm(category))
        share = values[members].mean()
        # most, some, all, and no come from the share of instances with the capacity: the same
        # values that the instance-level sentences read
        assert judge_projection(category, MOST).true == (share > 0.5)
        assert judge_projection(category, SOME).true == (share > 0)
        assert judge_projection(category, ALL).true == (share == 1)
        assert judge_projection(category, NO).true == (share == 0)
        assert judge_projection(category, MOST, False).true == (1 - share > 0.5)
        assert judge_projection(category, MOST).grounding == {
            "proportion": round(float(share), 6),
            "instances": len(members),
            "test": OBSERVED,
        }
        for index in members:
            instance = Proposition(
                INSTANCE, world.instances[index], Predicate(PROJECTION, projection)
            )
            assert facts.truth.evaluate(instance).true == bool(values[index])
        # a patient capacity is never nec: no fixed test exists for it
        for quantifier in (NEC_ALL, NEC_NO):
            assert not judge_projection(category, quantifier).valid
            assert "extensional" in judge_projection(category, quantifier).reason


def test_relation_facts_use_the_able_tables_for_any_pair_of_categories(cases) -> None:
    case = cases("default")
    facts = case.facts()
    truth = facts.truth
    world = case.world
    proportions = world.result.derived_frames()["relation_proportions.csv"]
    # where relation_proportions.csv has a row, the proportion is the same
    checked = 0
    for row in proportions.sample(300, seed=1).iter_rows(named=True):
        if row["event_type"] not in facts.verbs:
            continue
        proposition = Proposition(
            CLASS,
            CategoryTerm(row["agent"]),
            Predicate(VERB, row["event_type"], CategoryTerm(row["patient"])),
            True,
            SOME,
        )
        evaluation = truth.evaluate(proposition)
        assert evaluation.true
        assert evaluation.grounding == {
            "proportion": round(row["true_pairs"] / row["total_pairs"], 6),
            "pairs": row["total_pairs"],
            "test": OBSERVED,
        }
        checked += 1
    assert checked > 100
    # mixed levels and categories of event types are not in the file, and are computed all the
    # same, from the category's base relation
    category = next(v for v in facts.verbs if world.event_types[v].category)
    agents = truth.members(CategoryTerm("CATEGORY.1"))
    patients = truth.members(CategoryTerm("CATEGORY.2.1.1"))
    holds = world.able(category)[np.ix_(agents, patients)]
    predicate = Predicate(VERB, category, CategoryTerm("CATEGORY.2.1.1"))
    evaluation = truth.evaluate(
        Proposition(CLASS, CategoryTerm("CATEGORY.1"), predicate, True, MOST)
    )
    assert evaluation.grounding["proportion"] == round(float(holds.mean()), 6)
    assert evaluation.grounding["pairs"] == len(agents) * len(patients)
    assert evaluation.true == (holds.mean() > 0.5)
    # within one category, an instance is not paired with itself
    same = Predicate(VERB, category, CategoryTerm("CATEGORY.1"))
    within = truth.evaluate(Proposition(CLASS, CategoryTerm("CATEGORY.1"), same, True, SOME))
    assert within.grounding["pairs"] == len(agents) * (len(agents) - 1)
    # "all" for a relation fact is over the existing pairs, and never nec
    everything = truth.evaluate(
        Proposition(CLASS, CategoryTerm("CATEGORY.1"), predicate, True, ALL)
    )
    assert everything.true == bool(holds.all())
    nec = truth.evaluate(Proposition(CLASS, CategoryTerm("CATEGORY.1"), predicate, True, NEC_ALL))
    assert not nec.valid and "extensional" in nec.reason


def test_instance_level_truth_is_read_from_the_values(tiny) -> None:
    facts = tiny.facts()
    truth = facts.truth
    world = tiny.world
    for row, instance in enumerate(world.instances):
        for kind in (IS, HAS, CAN):
            for feature in facts.features[kind]:
                value = bool(world.column(feature)[row])
                for polarity in (True, False):
                    evaluation = truth.evaluate(
                        Proposition(INSTANCE, instance, Predicate(kind, feature), polarity)
                    )
                    assert evaluation.true == (value == polarity)
                    assert evaluation.grounding == {"value": int(value), "test": VALUE}
        for verb in facts.verbs:
            for other in range(world.count):
                if other == row:
                    continue
                predicate = Predicate(VERB, verb, world.instances[other])
                assert truth.evaluate(Proposition(INSTANCE, instance, predicate)).true == bool(
                    world.able(verb)[row, other]
                )
        own = Predicate(VERB, facts.verbs[0], instance)
        assert not truth.evaluate(Proposition(INSTANCE, instance, own)).valid
        # membership: every category on the instance's path, the leaf included
        leaf = world.instance_leaf[row]
        for category in truth.categories:
            on_path = category == leaf or category in truth.ancestors(leaf)
            member = Proposition(INSTANCE, instance, Predicate(MEMBER, category))
            assert truth.evaluate(member).true == on_path
            assert truth.evaluate(dataclasses.replace(member, polarity=False)).true == (not on_path)


def test_logical_forms_that_cannot_be_judged(tiny) -> None:
    truth = tiny.facts().truth
    subject = CategoryTerm("CATEGORY.1")
    one = CategoryTerm("CATEGORY.1", (Literal("EVENTTYPE1.1"),))
    unknown = CategoryTerm("CATEGORY.1", (Literal("PROPERTY.77"),))
    bad = [
        Proposition(CLASS, CategoryTerm("CATEGORY.9"), Predicate(IS, "PROPERTY.1"), True, NEC_ALL),
        Proposition(CLASS, subject, Predicate(IS, "PROPERTY.99"), True, NEC_ALL),
        Proposition(CLASS, subject, Predicate(IS, "PART.1"), True, NEC_ALL),
        Proposition(CLASS, subject, Predicate(CAN, "EVENTTYPE2.1.1"), True, NEC_ALL),
        Proposition(CLASS, subject, Predicate(SCALAR, "SCALARDIM.2.HIGH")),
        Proposition(CLASS, subject, Predicate(SCALAR, "SCALARDIM.1.MIDDLE")),
        Proposition(CLASS, subject, Predicate(SCALAR, "SCALARDIM.1.HIGH"), True, MOST),
        Proposition(CLASS, subject, Predicate(MEMBER, "CATEGORY.7"), True, NEC_ALL),
        Proposition(CLASS, subject, Predicate(PROJECTION, "CANBE.EVENTTYPE2.9"), True, MOST),
        Proposition(
            CLASS, subject, Predicate(VERB, "EVENTTYPE2.9", CategoryTerm("CATEGORY.2")), True, MOST
        ),
        Proposition(CLASS, subject, Predicate(VERB, "EVENTTYPE2.1.1"), True, MOST),
        Proposition(
            CLASS, subject, Predicate(IS, "PROPERTY.1", CategoryTerm("CATEGORY.2")), True, MOST
        ),
        Proposition(
            CLASS, subject, Predicate(VERB, "EVENTTYPE2.1.1", "INSTANCE.1.1.1"), True, MOST
        ),
        Proposition(CLASS, subject, Predicate("eats", "EVENTTYPE2.1.1"), True, MOST),
        Proposition(CLASS, subject, Predicate(IS, "PROPERTY.1"), True, "few"),
        Proposition(CLASS, subject, Predicate(IS, "PROPERTY.1"), True, "generic"),
        Proposition(CLASS, subject, Predicate(IS, "PROPERTY.1"), True, None),
        Proposition(CLASS, "INSTANCE.1.1.1", Predicate(IS, "PROPERTY.1"), True, NEC_ALL),
        Proposition(CLASS, one, Predicate(IS, "PROPERTY.1"), True, NEC_ALL),
        Proposition(CLASS, unknown, Predicate(IS, "PROPERTY.1"), True, NEC_ALL),
        Proposition(INSTANCE, "INSTANCE.9.9.9", Predicate(IS, "PROPERTY.1")),
        Proposition(INSTANCE, "INSTANCE.1.1.1", Predicate(IS, "PROPERTY.1"), True, NEC_ALL),
        Proposition(
            INSTANCE, "INSTANCE.1.1.1", Predicate(VERB, "EVENTTYPE2.1.1", "INSTANCE.9.9.9")
        ),
        Proposition(
            INSTANCE,
            "INSTANCE.1.1.1",
            Predicate(VERB, "EVENTTYPE2.1.1", CategoryTerm("CATEGORY.1")),
        ),
        Proposition("event", "INSTANCE.1.1.1", Predicate(IS, "PROPERTY.1")),
    ]
    for proposition in bad:
        evaluation = truth.evaluate(proposition)
        assert not evaluation.valid and evaluation.reason, proposition
        assert not truth.is_true(proposition) and truth.grounded(proposition) is None


# ---------------------------------------------------------------------------------------------
# Logical forms
# ---------------------------------------------------------------------------------------------


def test_the_logical_form_of_the_specification() -> None:
    proposition = Proposition(
        CLASS,
        CategoryTerm("CATEGORY.1.3", (Literal("PART.2", False), Literal("PROPERTY.4"))),
        Predicate(CAN, "EVENTTYPE1.3"),
        True,
        MOST,
        {"proportion": 0.93, "fixed": False, "test": "observed"},
        "PROP.310",
    )
    assert proposition.to_json() == {
        "id": "PROP.310",
        "level": "class",
        "quantifier": "most",
        "polarity": True,
        "subject": {"category": "CATEGORY.1.3", "restriction": ["PROPERTY.4", "not PART.2"]},
        "predicate": {"kind": "can", "feature": "EVENTTYPE1.3"},
        "grounding": {"proportion": 0.93, "fixed": False, "test": "observed"},
    }


def test_json_forms_of_every_predicate_kind() -> None:
    subject = CategoryTerm("CATEGORY.1")
    forms = {
        Predicate(IS, "PROPERTY.1"): {"kind": "is", "feature": "PROPERTY.1"},
        Predicate(HAS, "PART.1"): {"kind": "has", "feature": "PART.1"},
        Predicate(CAN, "EVENTTYPE1.2"): {"kind": "can", "feature": "EVENTTYPE1.2"},
        Predicate(MEMBER, "CATEGORY.2"): {"kind": "member", "category": "CATEGORY.2"},
        Predicate(PROJECTION, "CANBE.EVENTTYPE2.1.1"): {
            "kind": "projection",
            "projection": "CANBE.EVENTTYPE2.1.1",
        },
        Predicate(VERB, "EVENTTYPE2.1", CategoryTerm("CATEGORY.2", (Literal("PROPERTY.3"),))): {
            "kind": "verb",
            "verb": "EVENTTYPE2.1",
            "patient": {"category": "CATEGORY.2", "restriction": ["PROPERTY.3"]},
        },
    }
    for predicate, form in forms.items():
        proposition = Proposition(CLASS, subject, predicate, False, SOME)
        assert proposition.to_json()["predicate"] == form
        assert Proposition.from_json(proposition.to_json()) == proposition
    pole = Proposition(CLASS, subject, Predicate(SCALAR, "SCALARDIM.1.LOW"))
    assert pole.to_json()["predicate"] == {"kind": "scalar", "pole": "SCALARDIM.1.LOW"}
    assert pole.to_json()["quantifier"] is None
    assert Proposition.from_json(pole.to_json()) == pole
    instance = Proposition(
        INSTANCE, "INSTANCE.1.1.1", Predicate(VERB, "EVENTTYPE2.1.1", "INSTANCE.2.1.2"), False
    )
    assert instance.to_json() == {
        "id": None,
        "level": "instance",
        "polarity": False,
        "subject": {"instance": "INSTANCE.1.1.1"},
        "predicate": {
            "kind": "verb",
            "verb": "EVENTTYPE2.1.1",
            "patient": {"instance": "INSTANCE.2.1.2"},
        },
        "grounding": None,
    }
    assert Proposition.from_json(instance.to_json()) == instance
    scalar = Proposition(
        INSTANCE, "INSTANCE.1.1.1", Predicate(SCALAR, "SCALARDIM.1.HIGH", comparison="CATEGORY.1")
    )
    assert scalar.to_json()["predicate"] == {
        "kind": "scalar",
        "pole": "SCALARDIM.1.HIGH",
        "class": "CATEGORY.1",
    }
    assert Proposition.from_json(scalar.to_json()) == scalar


def test_a_restriction_is_a_set_in_one_order() -> None:
    a = CategoryTerm(
        "CATEGORY.1",
        (
            Literal("PART.3"),
            Literal("SCALARDIM.1.HIGH"),
            Literal("PROPERTY.10"),
            Literal("PROPERTY.2", False),
        ),
    )
    b = CategoryTerm(
        "CATEGORY.1",
        (
            Literal("PROPERTY.2", False),
            Literal("PROPERTY.10"),
            Literal("PART.3"),
            Literal("SCALARDIM.1.HIGH"),
            Literal("PART.3"),
        ),
    )
    assert a == b and hash(a) == hash(b)
    assert [str(x) for x in a.restriction] == [
        "not PROPERTY.2",
        "PROPERTY.10",
        "PART.3",
        "SCALARDIM.1.HIGH",
    ]
    assert Literal.parse("not PART.3") == Literal("PART.3", False)
    assert CategoryTerm.from_json(a.to_json()) == a


def test_equality_ignores_the_label_the_grounding_and_the_rule() -> None:
    plain = Proposition(
        CLASS,
        CategoryTerm(THING, (Literal("PART.1"),)),
        Predicate(CAN, "EVENTTYPE1.1"),
        True,
        NEC_ALL,
    )
    dressed = dataclasses.replace(
        plain, id="PROP.4", grounding={"test": "exact"}, rule=("EVENTTYPE1.1", 2)
    )
    assert plain == dressed and hash(plain) == hash(dressed)
    assert dressed.to_json()["rule"] == {"feature": "EVENTTYPE1.1", "term": 2}
    restored = Proposition.from_json(dressed.to_json())
    assert (restored.id, restored.grounding, restored.rule) == (
        "PROP.4",
        {"test": "exact"},
        ("EVENTTYPE1.1", 2),
    )
    assert plain != dataclasses.replace(plain, quantifier=ALL)


def test_concepts_a_sentence_needs_words_for() -> None:
    proposition = Proposition(
        CLASS,
        CategoryTerm("CATEGORY.1", (Literal("PROPERTY.2", False), Literal("SCALARDIM.1.HIGH"))),
        Predicate(VERB, "EVENTTYPE2.1.2", CategoryTerm("CATEGORY.2", (Literal("PART.4"),))),
        True,
        MOST,
    )
    assert proposition.concepts() == (
        "CATEGORY.1",
        "PROPERTY.2",
        "SCALARDIM.1.HIGH",
        "CATEGORY.2",
        "PART.4",
        "EVENTTYPE2.1.2",
    )
    rule = Proposition(
        CLASS,
        CategoryTerm(THING, (Literal("PART.1"),)),
        Predicate(CAN, "EVENTTYPE1.1"),
        True,
        NEC_ALL,
    )
    assert rule.concepts() == (THING, "PART.1", "EVENTTYPE1.1")
    # an instance's noun is chosen when it is mentioned, except for a scalar pole's class
    assert Proposition(INSTANCE, "INSTANCE.1.1.1", Predicate(IS, "PROPERTY.3")).concepts() == (
        "PROPERTY.3",
    )
    scalar = Proposition(
        INSTANCE, "INSTANCE.1.1.1", Predicate(SCALAR, "SCALARDIM.1.LOW", comparison="CATEGORY.1")
    )
    assert scalar.concepts() == ("SCALARDIM.1.LOW", "CATEGORY.1")
