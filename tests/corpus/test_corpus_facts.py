"""Stage 2 acceptance tests: the generated propositions.

Every generated proposition passes an independent recomputation of its truth from the taxonomy's
output files (``truth_oracle.Oracle``). Rule statements are ``all``-true.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pytest
import yaml

from semantic_world.corpus import Streams
from semantic_world.corpus.facts import (
    SKIP_MAX_LITERALS,
    SKIP_NO_INSTANCE,
    SKIP_NO_WORD,
    SKIP_REASONS,
    SKIP_THRESHOLD,
    SKIP_UNCONFIRMED,
)
from semantic_world.corpus.lexicon import THING
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    EXACT,
    GENERIC,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NO,
    PROJECTION,
    SCALAR,
    SOME,
    VERB,
    CategoryTerm,
    Clause,
    Literal,
    Predicate,
    Proposition,
)
from semantic_world.taxonomy.boolean import TruthTable, minimal_dnf

WORLDS = ("tiny", "default", "deep", "still")
OBSERVED = {"quantifiers": {"all_grounding": "observed"}}


def confirm(oracle, propositions) -> None:
    """Every proposition is true by the independent recomputation, and usable: a "some" is
    stated only when "all" (for a negative proposition, "no") is false."""
    for proposition in propositions:
        form = proposition.to_json()
        assert oracle.truth(form) is True, form
        if proposition.quantifier == SOME:
            assert not oracle.all_holds(form, 1 if proposition.polarity else 0), form
        assert proposition.grounding is not None


def sample(items, step: int):
    return items[::step] if len(items) > 400 else items


# ---------------------------------------------------------------------------------------------
# Every generated proposition is true
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
@pytest.mark.parametrize("settings", [{}, OBSERVED], ids=["fixed", "observed"])
def test_every_class_fact_passes_the_independent_recomputation(cases, name, settings) -> None:
    case = cases(name)
    facts, oracle = case.facts(**settings), case.oracle(**settings)
    total = 0
    for category in facts.categories:
        for negative in (False, True):
            stated = facts.class_facts(category, negative)
            confirm(oracle, stated)
            assert all(p.negative == negative for p in stated)
            generics = [g for g in map(facts.generic, stated) if g is not None]
            confirm(oracle, generics)
            assert all(g.quantifier == GENERIC and g.negative == negative for g in generics)
            total += len(stated)
    assert total > 300


@pytest.mark.parametrize("name", WORLDS)
def test_every_instance_fact_passes_the_independent_recomputation(cases, name) -> None:
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    total = 0
    for instance in sample(case.result.instances.labels, 12):
        for negative in (False, True):
            stated = facts.instance_facts(instance, negative)
            confirm(oracle, stated)
            assert all(p.polarity != negative and p.quantifier is None for p in stated)
            total += len(stated)
    assert total > 300


@pytest.mark.parametrize("name", WORLDS)
def test_every_fact_about_a_restricted_subject_passes(cases, name) -> None:
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    rng = np.random.default_rng(7)
    features = facts.features[IS] + facts.features[HAS]
    total = 0
    for category in facts.categories[:12]:
        members = facts.truth.members(CategoryTerm(category))
        for _ in range(4):
            # a restriction that one member satisfies, so the subject set is not empty
            row = int(rng.choice(members))
            chosen = [features[int(i)] for i in rng.choice(len(features), size=2, replace=False)]
            literals = tuple(
                Literal(f, bool(facts.truth.values[row, facts.truth.features[f].position]))
                for f in chosen
            )
            term = CategoryTerm(category, literals)
            for negative in (False, True):
                stated = facts.class_facts(term, negative, patients=facts.categories[:3])
                confirm(oracle, stated)
                total += len(stated)
    assert total > 300


@pytest.mark.parametrize("name", WORLDS)
def test_every_fact_with_the_category_as_patient_passes(cases, name) -> None:
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    for category in facts.categories[:6]:
        for negative in (False, True):
            stated = facts.patient_facts(category, negative, agents=facts.categories[:8])
            confirm(oracle, stated)
            assert all(
                p.predicate.kind == VERB and p.predicate.patient == CategoryTerm(category)
                for p in stated
            )


@pytest.mark.parametrize("name", WORLDS)
def test_every_drawn_proposition_passes(cases, name) -> None:
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    rng = Streams(3).propositions
    patients = facts.categories[:5]
    drawn = [
        facts.draw_class(rng, facts.categories[int(rng.integers(len(facts.categories)))], patients)
        for _ in range(400)
    ]
    others = case.result.instances.labels[:6]
    drawn += [
        facts.draw_instance(
            rng, case.result.instances.labels[int(rng.integers(len(others)))], others
        )
        for _ in range(400)
    ]
    assert all(p is not None for p in drawn)
    confirm(oracle, drawn)


# ---------------------------------------------------------------------------------------------
# The shape of the facts
# ---------------------------------------------------------------------------------------------


def test_the_strongest_true_quantifier_is_stated(cases) -> None:
    case = cases("default")
    facts, truth = case.facts(), case.facts().truth
    seen: Counter = Counter()
    for category in facts.categories[::5]:
        subject = CategoryTerm(category)
        for negative in (False, True):
            order = (
                [(NO, True), (MOST, False), (SOME, False)]
                if negative
                else [(ALL, True), (MOST, True), (SOME, True)]
            )
            for fact in facts.class_facts(category, negative):
                kind = fact.predicate.kind
                if kind in (SCALAR, MEMBER):
                    continue
                stated = order.index((fact.quantifier, fact.polarity))
                # every stronger quantifier is false
                for quantifier, polarity in order[:stated]:
                    stronger = Proposition(CLASS, subject, fact.predicate, polarity, quantifier)
                    assert not truth.is_true(stronger), fact
                seen[(kind, fact.quantifier, fact.polarity)] += 1
    for kind in (IS, HAS, CAN, VERB):
        for key in (
            (ALL, True),
            (MOST, True),
            (SOME, True),
            (NO, True),
            (MOST, False),
            (SOME, False),
        ):
            assert seen[(kind, *key)] > 0, (kind, key)
    # a patient projection takes no law-like quantifier
    assert seen[(PROJECTION, MOST, True)] and not seen[(PROJECTION, ALL, True)]
    assert not seen[(PROJECTION, NO, True)]


def test_facts_come_in_the_order_of_the_predicates(cases) -> None:
    facts = cases("tiny").facts()
    predicates = facts.class_predicates("C1.1")
    kinds = [p.kind for p in predicates]
    assert [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k] == [
        IS,
        HAS,
        CAN,
        PROJECTION,
        SCALAR,
        MEMBER,
        VERB,
    ]
    assert len(predicates) == 8 + 8 + 4 + 1 + 2 + 5 + 5 * 6
    # membership in a category at the same level or above, never in itself or in a subcategory
    assert [p.label for p in predicates if p.kind == MEMBER] == ["C1", "C1.2", "C2", "C2.1", "C2.2"]
    assert [p.label for p in facts.class_predicates("C1") if p.kind == MEMBER] == ["C2"]
    assert not [p for p in facts.class_predicates(THING) if p.kind == MEMBER]
    order = {p: i for i, p in enumerate(predicates)}
    stated = [order[f.predicate] for f in facts.class_facts("C1.1")]
    assert stated == sorted(stated) and len(set(stated)) == len(stated)
    limited = facts.class_predicates("C1.1", patients=("C2",))
    assert [p.patient for p in limited if p.kind == VERB] == [CategoryTerm("C2")] * 5


def test_membership_facts(cases) -> None:
    facts = cases("default").facts()
    members = [f for f in facts.class_facts("C1.2.1") if f.predicate.kind == MEMBER]
    assert [(f.predicate.label, f.quantifier) for f in members] == [("C1", ALL), ("C1.2", ALL)]
    apart = [f for f in facts.class_facts("C1.2.1", negative=True) if f.predicate.kind == MEMBER]
    assert all(f.quantifier == NO and f.polarity for f in apart)
    assert {f.predicate.label for f in apart} == {
        c for c in facts.categories if c not in ("C1", "C1.2", "C1.2.1")
    }
    generic = facts.generic(apart[0])
    assert generic.quantifier == GENERIC and generic.polarity is False


def test_verb_categories_are_stated_in_capacity_sentences(cases) -> None:
    case = cases("default")
    facts = case.facts()
    tree = case.result.verbs.tree
    general = [v for v in facts.verbs if not tree[v].is_leaf]
    assert len(general) == 3 and len(facts.verbs) == 10
    for category in ("C1", "C2.1"):
        stated = {
            f.predicate.label for f in facts.class_facts(category) if f.predicate.kind == VERB
        }
        assert stated & set(general)
    # a verb entails the base relation of every verb category above it, so the general verb
    # holds for at least as large a share of the pairs
    truth = facts.truth
    for verb in facts.verbs:
        if tree[verb].is_leaf and tree[verb].parent.label in general:
            assert not (truth.matrix(verb) & ~truth.matrix(tree[verb].parent.label)).any()
    instance = case.result.instances.labels[0]
    capacity = {
        f.predicate.label for f in facts.instance_facts(instance) if f.predicate.kind == VERB
    }
    assert capacity & set(general)


def test_instance_predicates(cases) -> None:
    case = cases("tiny")
    facts = case.facts()
    predicates = facts.instance_predicates("I1.1.1")
    by_kind = Counter(p.kind for p in predicates)
    assert by_kind == {IS: 8, HAS: 8, CAN: 4, PROJECTION: 1, SCALAR: 4, MEMBER: 6, VERB: 5 * 11}
    # a scalar pole against every category the instance is below
    assert {p.comparison for p in predicates if p.kind == SCALAR} == {"C1", "C1.1"}
    assert all(p.patient != "I1.1.1" for p in predicates if p.kind == VERB)
    near = facts.instance_predicates("I1.1.1", patients=("I1.1.1", "I1.1.2", "I2.1.1"))
    assert {p.patient for p in near if p.kind == VERB} == {"I1.1.2", "I2.1.1"}
    positive = facts.instance_facts("I1.1.1")
    negative = facts.instance_facts("I1.1.1", negative=True)
    # each predicate is true or false of the instance, so it is asserted or denied, never both
    assert len(positive) + len(negative) == len(predicates)
    assert not {f.predicate for f in positive} & {f.predicate for f in negative}


def test_only_named_concepts_take_part(cases) -> None:
    case = cases("default")
    proportions = dict.fromkeys(
        ("category", "is", "has", "can", "verb", "verb_category", "patient_projection", "scalar"),
        0.5,
    )
    facts = case.facts(lexicon={"named_proportion": proportions})
    everything = case.facts()
    assert len(facts.named) < len(everything.named)
    assert len(facts.categories) == 28 and len(facts.features[IS]) == 20
    for category in facts.categories[:6]:
        for negative in (False, True):
            for fact in facts.class_facts(category, negative):
                assert set(fact.concepts()) <= facts.named
    instance = case.result.instances.labels[3]
    for fact in facts.instance_facts(instance):
        assert set(fact.concepts()) <= facts.named
    for statement in facts.rule_statements():
        assert set(statement.concepts()) <= facts.named
    assert facts.rule_report["skipped"][SKIP_NO_WORD] > 0
    unnamed = next(c for c in everything.categories if c not in facts.named)
    assert facts.class_fact(CategoryTerm(unnamed), Predicate(HAS, facts.features[HAS][0])) is None
    # the truth tests still judge a proposition about a concept without a word
    assert facts.truth.evaluate(
        Proposition(
            CLASS, CategoryTerm(unnamed), Predicate(HAS, facts.features[HAS][0]), True, SOME
        )
    ).valid


def test_verbs_without_a_word_are_never_stated(cases) -> None:
    facts = cases("tiny").facts()
    assert "V1" not in facts.verbs and facts.verbs == ("V1.1", "V1.2", "V2.1", "V2.2", "V2")
    for category in facts.categories:
        assert all(f.predicate.label != "V1" for f in facts.class_facts(category))


# ---------------------------------------------------------------------------------------------
# Rule statements
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_rule_statements_are_all_true(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    statements = facts.rule_statements()
    assert statements
    fixed, observed = case.oracle(), case.oracle(**OBSERVED)
    seen = case.facts(**OBSERVED)
    for statement in statements:
        form = statement.to_json()
        assert statement.quantifier == ALL and statement.polarity and statement.level == CLASS
        assert statement.subject.category == THING
        # true as a law, by the engine and by the independent recomputation
        assert fixed.truth(form) is True, form
        assert statement.grounding["fixed"] is True and statement.grounding["test"] == EXACT
        assert statement.grounding["proportion"] == 1.0 and statement.grounding["instances"] >= 1
        # and true of every instance that satisfies the term
        assert observed.truth(form) is True and seen.truth.is_true(statement)
        assert fixed.fixed(form["subject"], statement.predicate.label) == 1
        # the bare generic says the same, and is true too
        assert facts.generic(statement).quantifier == GENERIC
    assert seen.rule_statements() == statements


@pytest.mark.parametrize("name", WORLDS)
def test_rule_statements_are_the_terms_of_the_minimal_dnf(cases, name) -> None:
    # Recount the terms from rules.yaml: each term of each rule is stated or skipped, once.
    case = cases(name)
    facts = case.facts()
    statements = {s.rule: s for s in facts.rule_statements()}
    rules = yaml.safe_load((case.folder / "rules.yaml").read_text(encoding="utf-8"))
    truth = facts.truth
    expected = dict.fromkeys(SKIP_REASONS, 0)
    terms = 0
    for rule in rules:
        table = TruthTable.from_bit_string(rule["truth_table"])
        for number, term in enumerate(minimal_dnf(table), start=1):
            terms += 1
            used = [
                (name, bit)
                for name, bit in zip(rule["inputs"], term, strict=True)
                if bit is not None
            ]
            key = (rule["output"], number)
            if any(name.startswith("SC.") for name, _ in used):
                expected[SKIP_THRESHOLD] += 1
            else:
                subject = CategoryTerm(THING, tuple(Literal(name, bool(bit)) for name, bit in used))
                if len(truth.members(subject)) == 0:
                    expected[SKIP_NO_INSTANCE] += 1
                else:
                    statement = statements.pop(key)
                    assert statement.subject == subject
                    assert statement.predicate.label == rule["output"]
                    assert statement.predicate.kind == rule["output"].split(".")[0].lower()
                    continue
            assert key not in statements
    assert not statements
    report = facts.rule_report
    assert report == {
        "rules": len(rules),
        "terms": terms,
        "stated": terms - sum(expected.values()),
        "skipped": expected,
    }
    assert report["skipped"][SKIP_UNCONFIRMED] == 0 and report["skipped"][SKIP_MAX_LITERALS] == 0


def test_rule_statement_counts_of_the_default_world(cases) -> None:
    facts = cases("default").facts()
    facts.rule_statements()
    report = facts.rule_report
    assert report == {
        "rules": 40,
        "terms": 103,
        "stated": 86,
        "skipped": {
            SKIP_THRESHOLD: 13,
            SKIP_MAX_LITERALS: 0,
            SKIP_NO_WORD: 0,
            SKIP_NO_INSTANCE: 4,
            SKIP_UNCONFIRMED: 0,
        },
    }


def test_long_terms_are_stated_unless_a_cap_is_set(cases) -> None:
    case = cases("default")
    uncapped = case.facts()
    lengths = Counter(len(s.subject.restriction) for s in uncapped.rule_statements())
    # rule statements are exempt from the limits on adjectives and with-phrases: one of the
    # default world's statements has four with-phrases, and the limit is two
    assert lengths == {1: 14, 2: 31, 3: 10, 4: 31}
    assert case.config().mention.max_with_phrases == 2
    with_phrases = [
        sum(x.feature.startswith("HAS.") for x in s.subject.restriction)
        for s in uncapped.rule_statements()
    ]
    assert max(with_phrases) == 4
    for cap in (1, 2, 3):
        capped = case.facts(propositions={"rule_statements": {"max_literals": cap}})
        statements = capped.rule_statements()
        assert all(len(s.subject.restriction) <= cap for s in statements)
        assert set(statements) == {
            s for s in uncapped.rule_statements() if len(s.subject.restriction) <= cap
        }
        over = sum(count for length, count in lengths.items() if length > cap)
        skipped = capped.rule_report["skipped"]
        # a term that is skipped for another reason is not counted against the cap
        assert skipped[SKIP_MAX_LITERALS] >= over
        assert capped.rule_report["stated"] + sum(skipped.values()) == 103


def test_negated_is_literals_share_one_relative_clause(cases) -> None:
    # "things with wings that are not red and not big can fly": a relative clause joins several
    # verb phrases with "and", so a term with any number of negated IS literals is stated.
    facts = cases("default").facts()
    negated = [
        sum(not x.positive and x.feature.startswith("IS.") for x in s.subject.restriction)
        for s in facts.rule_statements()
    ]
    assert Counter(negated)[0] > 0 and sum(count > 1 for count in negated) == 27
    assert max(negated) == 4
    assert "relative_clauses" not in facts.rule_report["skipped"]
    # only a term that reads a scalar threshold, or that no instance satisfies, stays unstated
    skipped = facts.rule_report["skipped"]
    assert {reason for reason, count in skipped.items() if count} == {
        SKIP_THRESHOLD,
        SKIP_NO_INSTANCE,
    }


def test_the_quantifier_of_a_rule_statement_is_drawn_like_any_other(cases) -> None:
    case = cases("default")
    facts = case.facts()
    oracle = case.oracle()
    rng = Streams(2).propositions
    drawn = [facts.draw_rule_statement(rng) for _ in range(1000)]
    # "all things with wings ..." or the bare plural, at the generic rate; true under both
    assert {p.quantifier for p in drawn} == {ALL, GENERIC}
    assert abs(sum(p.quantifier == GENERIC for p in drawn) / len(drawn) - 0.5) < 0.05
    for statement in drawn[:200]:
        assert oracle.truth(statement.to_json()) is True
        assert statement.rule is not None and statement.polarity
        assert statement.subject.category == THING
    assert len({p.rule for p in drawn}) == 86  # every rule statement is drawn
    for means in ("all", "most", "some"):
        other = case.facts(quantifiers={"generic": {"means": means}, "generic_rate": 1.0})
        generic = [other.draw_rule_statement(rng) for _ in range(100)]
        assert all(p.quantifier == GENERIC for p in generic)
    never = case.facts(quantifiers={"generic_rate": 0.0})
    assert all(never.draw_rule_statement(rng).quantifier == ALL for _ in range(100))
    # a pool limits the draw: the sufficient conditions of one feature
    feature = facts.rule_statements()[0].predicate.label
    pool = facts.rule_statements(feature)
    assert all(facts.draw_rule_statement(rng, pool).predicate.label == feature for _ in range(50))
    assert facts.draw_rule_statement(rng, ()) is None


def test_rule_statements_by_feature(cases) -> None:
    facts = cases("default").facts()
    statements = facts.rule_statements()
    outputs = list(dict.fromkeys(s.predicate.label for s in statements))
    assert sum(len(facts.rule_statements(f)) for f in outputs) == len(statements)
    for feature in outputs[:5]:
        assert all(s.predicate.label == feature for s in facts.rule_statements(feature))
    # what a feature makes possible: the statements that read it
    inputs = Counter(x.feature for s in statements for x in s.subject.restriction)
    feature, count = inputs.most_common(1)[0]
    reading = facts.rule_statements_reading(feature)
    assert len(reading) == count
    assert all(any(x.feature == feature for x in s.subject.restriction) for s in reading)
    assert facts.rule_statements("IS.1") == ()  # a free feature has no rule


def test_a_chained_rule_is_stated_with_determined_literals(cases) -> None:
    # In the deep world some rules read determined features, so a term's literal can be a
    # determined feature. The fixed test keeps only the settings that satisfy it.
    case = cases("deep")
    facts = case.facts()
    features = case.result.features
    chained = [
        s
        for s in facts.rule_statements()
        if any(not features[x.feature].free for x in s.subject.restriction)
    ]
    assert chained
    oracle = case.oracle()
    for statement in chained:
        assert oracle.truth(statement.to_json()) is True


def test_rule_statements_round_trip_through_json(cases) -> None:
    for statement in cases("deep").facts().rule_statements():
        text = json.dumps(statement.to_json())
        restored = Proposition.from_json(json.loads(text))
        assert restored == statement
        assert (restored.rule, restored.grounding) == (statement.rule, statement.grounding)


# ---------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------


def draws(facts, count: int, seed: int = 1, level: str = CLASS):
    rng = Streams(seed).propositions
    labels = facts.categories if level == CLASS else facts.result.instances.labels[:8]
    drawn = []
    for _ in range(count):
        label = labels[int(rng.integers(len(labels)))]
        if level == CLASS:
            drawn.append(facts.draw_class(rng, label, patients=facts.categories[:4]))
        else:
            drawn.append(
                facts.draw_instance(rng, label, patients=facts.result.instances.labels[:8])
            )
    return drawn


@pytest.mark.parametrize("level", [CLASS, INSTANCE])
def test_the_negation_rate(cases, level) -> None:
    case = cases("default")
    for rate in (0.0, 0.1, 0.5, 1.0):
        facts = case.facts(propositions={"negation_rate": {level: rate}})
        drawn = draws(facts, 1500, level=level)
        share = sum(p.negative for p in drawn) / len(drawn)
        assert abs(share - rate) < 0.04, (rate, share)
    # the default rate is 0.1 at each level
    assert case.config().propositions.negation_rate == {CLASS: 0.1, INSTANCE: 0.1}


def test_the_generic_rate(cases) -> None:
    case = cases("default")
    never = draws(case.facts(quantifiers={"generic_rate": 0.0}), 800)
    assert not any(p.quantifier == GENERIC and p.predicate.kind != SCALAR for p in never)
    always = case.facts(quantifiers={"generic_rate": 1.0})
    for proposition in draws(always, 800):
        if proposition.quantifier != GENERIC:
            # the generic is used whenever it is true
            assert always.generic(proposition) is None
    half = draws(case.facts(), 3000)
    # among the draws that have a true generic, half use it
    facts = case.facts()
    could = [
        p
        for p in half
        if p.predicate.kind != SCALAR and (p.quantifier == GENERIC or facts.generic(p))
    ]
    used = sum(p.quantifier == GENERIC for p in could)
    assert abs(used / len(could) - 0.5) < 0.05


def test_draws_are_reproducible(cases) -> None:
    facts = cases("default").facts()
    assert draws(facts, 200, seed=4) == draws(facts, 200, seed=4)
    assert draws(facts, 200, seed=4) != draws(facts, 200, seed=5)
    first = draws(facts, 50, seed=4, level=INSTANCE)
    assert first == draws(facts, 50, seed=4, level=INSTANCE)
    assert all(dataclasses.replace(p, grounding=None) == p for p in first)


# ---------------------------------------------------------------------------------------------
# Restricted subjects
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_a_restriction_is_drawn_among_those_that_do_work(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    truth = facts.truth
    rng = Streams(4).propositions
    polarities: Counter = Counter()
    kinds: Counter = Counter()
    for category in facts.categories:
        term = CategoryTerm(category)
        total = len(truth.members(term))
        options = facts.restriction_options(term)
        for literal in options:
            # some members satisfy the literal, and not all
            inside = len(truth.members(CategoryTerm(category, (literal,))))
            assert 0 < inside < total
        # every literal that does work is an option
        for feature in facts.features[IS] + facts.features[HAS]:
            count = int(truth.values[truth.members(term), truth.features[feature].position].sum())
            assert (Literal(feature) in options) == (0 < count < total)
            assert (Literal(feature, False) in options) == (0 < count < total)
        for _ in range(20):
            drawn = facts.draw_restriction(rng, term)
            if not options:
                assert drawn is None
                continue
            (literal,) = drawn.restriction
            assert drawn.category == category and literal in options
            polarities[literal.positive] += 1
            kinds[literal.feature.split(".")[0]] += 1
            # a second literal narrows the subject again
            again = facts.draw_restriction(rng, drawn)
            if again is not None:
                assert len(again.restriction) == 2
                assert 0 < len(truth.members(again)) < len(truth.members(drawn))
    # negative at the class-level negation rate (0.1)
    assert 0.03 < polarities[False] / sum(polarities.values()) < 0.2
    assert {"IS", "HAS"} <= set(kinds)


def test_a_relative_clause_is_drawn_among_those_that_do_work(cases) -> None:
    case = cases("default")
    settings = {"mention": {"relative_clauses": {"rate": 0.5, "max_depth": 2, "object_share": 0.3}}}
    facts = case.facts(**settings)
    truth = facts.truth
    rng = Streams(4).propositions
    kinds: Counter = Counter()
    nested = 0
    for category in facts.categories:
        term = CategoryTerm(category)
        total = len(truth.members(term))
        for object_relative in (False, True):
            for clause in facts.clause_options(term, object_relative):
                inside = len(truth.members(CategoryTerm(category, (), (clause,))))
                assert 0 < inside < total
                assert (clause.agent is not None) == object_relative
                assert clause.kind in (CAN, VERB) and (clause.kind == CAN) == (clause.other is None)
        for _ in range(12):
            drawn = facts.draw_clause(rng, term)
            if drawn is None:
                assert not facts.clause_options(term, False)
                assert not facts.clause_options(term, True)
                continue
            (clause,) = drawn.clauses
            assert 0 < len(truth.members(drawn)) < total
            kinds[(clause.kind, clause.agent is not None)] += 1
            if clause.other is not None and clause.other.clauses:
                # a clause inside the clause, down to the depth limit and no further
                (inner,) = clause.other.clauses
                assert inner.other is None or not inner.other.clauses
                assert 0 < len(truth.members(clause.other)) < len(truth.members(clause.other.plain))
                nested += 1
            # a term takes one relative clause
            assert facts.draw_clause(rng, drawn) is None
    drawn = sum(kinds.values())
    # an object relative with probability object_share, and a subject relative otherwise: a CAN
    # feature or a verb, each with the same chance
    assert abs(kinds[(VERB, True)] / drawn - 0.3) < 0.07
    assert abs(kinds[(CAN, False)] / drawn - 0.35) < 0.07
    assert abs(kinds[(VERB, False)] / drawn - 0.35) < 0.07
    assert nested > 20


def test_the_settings_of_a_drawn_relative_clause(cases) -> None:
    case = cases("default")

    def drawn(**clauses):
        facts = case.facts(mention={"relative_clauses": clauses})
        rng = Streams(4).propositions
        terms = [
            facts.draw_clause(rng, CategoryTerm(c)) for c in facts.categories for _ in range(4)
        ]
        return [term for term in terms if term is not None]

    # the depth limit: none at 0, and no clause inside a clause at 1
    assert drawn(rate=1.0, max_depth=0) == []
    assert all(not t.clauses[0].other or not t.clauses[0].other.clauses for t in drawn(max_depth=1))
    deep = drawn(rate=1.0, max_depth=3)
    assert any(
        t.clauses[0].other and t.clauses[0].other.clauses and t.clauses[0].other.clauses[0].other
        for t in deep
    )
    # the share of object relatives
    assert all(t.clauses[0].agent is None for t in drawn(object_share=0.0, max_depth=1))
    most = drawn(object_share=1.0, max_depth=1)
    assert np.mean([t.clauses[0].agent is not None for t in most]) > 0.9
    # the negated IS literals need a subject relative to join, so no object relative is drawn
    facts = case.facts(mention={"relative_clauses": {"object_share": 1.0}})
    rng = Streams(4).propositions
    checked = 0
    for category in facts.categories:
        negated = [x for x in facts.restriction_options(CategoryTerm(category)) if not x.positive]
        negated = [x for x in negated if x.feature.startswith("IS.")]
        for literal in negated[:2]:
            term = facts.draw_clause(rng, CategoryTerm(category, (literal,)))
            if term is not None:
                assert term.clauses[0].agent is None and term.restriction == (literal,)
                checked += 1
    assert checked > 10
    assert isinstance(drawn(max_depth=1)[0].clauses[0], Clause)
