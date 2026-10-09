"""Stage 2 and stage a5a acceptance tests: the generated propositions.

Every generated proposition passes an independent recomputation of its truth from the world
run's files (``truth_oracle.Oracle``). Rule statements are ``nec_all`` by construction.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pytest
import yaml

from semantic_world.common.boolean import TruthTable, minimal_dnf
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
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
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

WORLDS = ("tiny", "default", "deep", "still")
EXTENSIONAL = {"quantifiers": {"universal_words": "extensional"}}
KIND_OF = {"PROPERTY": IS, "PART": HAS, "EVENTTYPE1": CAN}


def one_place_requirements(folder) -> list[tuple[str, list[str], str]]:
    """The one-place requirements of a world run, from ``definition.json``: each event type's
    label, the input labels of its one constraint (a scalar's label for a threshold literal),
    and the constraint's truth table."""
    record = json.loads((folder / "definition.json").read_text(encoding="utf-8"))
    literals = record["literals"]
    rules = {r["output"]: r for r in record["rules"]}
    found = []
    for event_type in record["event_types"]:
        if event_type["arity"] != 1:
            continue
        (constraint,) = event_type["requirement"]["constraints"]
        assert event_type["requirement"]["expression"] == constraint
        rule = rules[constraint]
        inputs = []
        for index in rule["inputs"]:
            literal = literals[index]
            assert literal["kind"] in ("feature", "threshold"), literal
            inputs.append(literal["feature"] if literal["kind"] == "feature" else literal["scalar"])
        found.append((event_type["label"], inputs, rule["truth_table"]))
    return found


def confirm(oracle, propositions, universal_words: str = "nec") -> None:
    """Every proposition is true by the independent recomputation, and usable: a "some" is
    stated only when the language's "all" (for a negative proposition, its "no") is false."""
    for proposition in propositions:
        form = proposition.to_json()
        assert oracle.truth(form) is True, form
        if proposition.quantifier == SOME:
            nec = universal_words == "nec" and proposition.predicate.kind in (IS, HAS, CAN)
            universal = (
                (NEC_ALL if nec else ALL) if proposition.polarity else (NEC_NO if nec else NO)
            )
            twin = {**form, "quantifier": universal, "polarity": True}
            assert oracle.truth(twin) is not True, form
        assert proposition.grounding is not None


def sample(items, step: int):
    return items[::step] if len(items) > 400 else items


# ---------------------------------------------------------------------------------------------
# Every generated proposition is true
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
@pytest.mark.parametrize("settings", [{}, EXTENSIONAL], ids=["nec", "extensional"])
def test_every_class_fact_passes_the_independent_recomputation(cases, name, settings) -> None:
    case = cases(name)
    facts, oracle = case.facts(**settings), case.oracle(**settings)
    words = "extensional" if settings else "nec"
    total = 0
    for category in facts.categories:
        for negative in (False, True):
            stated = facts.class_facts(category, negative)
            confirm(oracle, stated, words)
            assert all(p.negative == negative for p in stated)
            # every stated quantifier is one the language can say, by a word or a bare plural
            assert all(facts.truth.statable(p) for p in stated)
            total += len(stated)
    assert total > 300


@pytest.mark.parametrize("name", WORLDS)
def test_every_instance_fact_passes_the_independent_recomputation(cases, name) -> None:
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    total = 0
    for instance in sample(case.world.instances, 12):
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
            literals = tuple(Literal(f, bool(case.world.column(f)[row])) for f in chosen)
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
    others = case.world.instances[:6]
    drawn += [
        facts.draw_instance(rng, case.world.instances[int(rng.integers(len(others)))], others)
        for _ in range(400)
    ]
    assert all(p is not None for p in drawn)
    confirm(oracle, drawn)


# ---------------------------------------------------------------------------------------------
# The shape of the facts
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("settings", [{}, EXTENSIONAL], ids=["nec", "extensional"])
def test_the_strongest_statable_true_quantifier_is_stated(cases, settings) -> None:
    case = cases("default")
    facts = case.facts(**settings)
    truth = facts.truth
    seen: Counter = Counter()
    for category in facts.categories[::5]:
        subject = CategoryTerm(category)
        for negative in (False, True):
            order = (
                [(NEC_NO, True), (NO, True), (MOST, False), (SOME, False)]
                if negative
                else [(NEC_ALL, True), (ALL, True), (MOST, True), (SOME, True)]
            )
            for fact in facts.class_facts(category, negative):
                kind = fact.predicate.kind
                if kind in (SCALAR, MEMBER):
                    continue
                stated = order.index((fact.quantifier, fact.polarity))
                # every stronger quantifier is false, not statable, or barred by a usage rule
                for quantifier, polarity in order[:stated]:
                    stronger = Proposition(CLASS, subject, fact.predicate, polarity, quantifier)
                    assert not truth.evaluate(stronger).felicitous, fact
                seen[(kind, fact.quantifier, fact.polarity)] += 1
    nec = not settings
    universal, negative_universal = (NEC_ALL, NEC_NO) if nec else (ALL, NO)
    for kind in (IS, HAS, CAN):
        for key in (
            (universal, True),
            (MOST, True),
            (SOME, True),
            (negative_universal, True),
            (MOST, False),
            (SOME, False),
        ):
            assert seen[(kind, *key)] > 0, (kind, key)
    # a relation fact and a patient capacity are never nec; under nec words their universal
    # cannot be said, so most is the strongest
    for kind in (VERB, PROJECTION):
        assert seen[(kind, MOST, True)] and seen[(kind, SOME, True)]
        assert not seen[(kind, NEC_ALL, True)] and not seen[(kind, NEC_NO, True)]
        assert bool(seen[(kind, ALL, True)] or seen[(kind, NO, True)]) == (not nec), kind


def test_facts_come_in_the_order_of_the_predicates(cases) -> None:
    facts = cases("tiny").facts()
    predicates = facts.class_predicates("CATEGORY.1.1")
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
    # 6 two-place event types (4 leaves and 2 categories, all with a word) by 6 patient categories
    assert len(predicates) == 8 + 8 + 4 + 1 + 2 + 5 + 6 * 6
    # membership in a category at the same level or above, never in itself or in a subcategory
    assert [p.label for p in predicates if p.kind == MEMBER] == [
        "CATEGORY.1",
        "CATEGORY.1.2",
        "CATEGORY.2",
        "CATEGORY.2.1",
        "CATEGORY.2.2",
    ]
    assert [p.label for p in facts.class_predicates("CATEGORY.1") if p.kind == MEMBER] == [
        "CATEGORY.2"
    ]
    assert not [p for p in facts.class_predicates(THING) if p.kind == MEMBER]
    order = {p: i for i, p in enumerate(predicates)}
    stated = [order[f.predicate] for f in facts.class_facts("CATEGORY.1.1")]
    assert stated == sorted(stated) and len(set(stated)) == len(stated)
    limited = facts.class_predicates("CATEGORY.1.1", patients=("CATEGORY.2",))
    assert [p.patient for p in limited if p.kind == VERB] == [CategoryTerm("CATEGORY.2")] * 6


def test_membership_facts(cases) -> None:
    facts = cases("default").facts()
    members = [f for f in facts.class_facts("CATEGORY.1.2.1") if f.predicate.kind == MEMBER]
    assert [(f.predicate.label, f.quantifier) for f in members] == [
        ("CATEGORY.1", NEC_ALL),
        ("CATEGORY.1.2", NEC_ALL),
    ]
    apart = [
        f for f in facts.class_facts("CATEGORY.1.2.1", negative=True) if f.predicate.kind == MEMBER
    ]
    assert all(f.quantifier == NEC_NO and f.polarity for f in apart)
    assert {f.predicate.label for f in apart} == {
        c for c in facts.categories if c not in ("CATEGORY.1", "CATEGORY.1.2", "CATEGORY.1.2.1")
    }
    # the bare plural can state membership, for nec_all and its counterpart
    assert facts.truth.bare_plural_expresses(members[0])
    assert facts.truth.bare_plural_expresses(apart[0])


def test_event_type_categories_are_stated_in_capacity_sentences(cases) -> None:
    case = cases("default")
    facts = case.facts()
    event_types = case.world.event_types
    general = [v for v in facts.verbs if event_types[v].category]
    assert len(general) == 5 and len(facts.verbs) == 25
    for category in ("CATEGORY.1", "CATEGORY.2.1"):
        stated = {
            f.predicate.label for f in facts.class_facts(category) if f.predicate.kind == VERB
        }
        assert stated & set(general)
    # an event type entails the base relation of every category above it, so the category
    # holds for at least as large a share of the pairs
    truth = facts.truth
    for verb in facts.verbs:
        parent = event_types[verb].parent
        if not event_types[verb].category and parent in general:
            assert not (truth.matrix(verb) & ~truth.matrix(parent)).any()
    instance = case.world.instances[0]
    capacity = {
        f.predicate.label for f in facts.instance_facts(instance) if f.predicate.kind == VERB
    }
    assert capacity & set(general)


def test_instance_predicates(cases) -> None:
    case = cases("tiny")
    facts = case.facts()
    predicates = facts.instance_predicates("INSTANCE.1.1.1")
    by_kind = Counter(p.kind for p in predicates)
    assert by_kind == {IS: 8, HAS: 8, CAN: 4, PROJECTION: 1, SCALAR: 4, MEMBER: 6, VERB: 6 * 11}
    # a scalar pole against every category the instance is below
    assert {p.comparison for p in predicates if p.kind == SCALAR} == {"CATEGORY.1", "CATEGORY.1.1"}
    assert all(p.patient != "INSTANCE.1.1.1" for p in predicates if p.kind == VERB)
    near = facts.instance_predicates(
        "INSTANCE.1.1.1", patients=("INSTANCE.1.1.1", "INSTANCE.1.1.2", "INSTANCE.2.1.1")
    )
    assert {p.patient for p in near if p.kind == VERB} == {"INSTANCE.1.1.2", "INSTANCE.2.1.1"}
    positive = facts.instance_facts("INSTANCE.1.1.1")
    negative = facts.instance_facts("INSTANCE.1.1.1", negative=True)
    # each predicate is true or false of the instance, so it is asserted or denied, never both
    assert len(positive) + len(negative) == len(predicates)
    assert not {f.predicate for f in positive} & {f.predicate for f in negative}


def test_only_named_concepts_take_part(cases) -> None:
    case = cases("default")
    proportions = dict.fromkeys(
        ("category", "property", "part", "event_unary", "event", "event_category", "scalar"), 0.5
    )
    facts = case.facts(lexicon={"named_proportion": proportions})
    everything = case.facts()
    assert len(facts.named) < len(everything.named)
    assert len(facts.categories) == 28 and len(facts.features[IS]) == 20
    for category in facts.categories[:6]:
        for negative in (False, True):
            for fact in facts.class_facts(category, negative):
                assert set(fact.concepts()) <= facts.named
    instance = case.world.instances[3]
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


def test_event_types_without_a_word_are_never_stated(cases) -> None:
    # EVENTTYPE2.2 of the deep world holds for every pair, so it has no word
    facts = cases("deep").facts()
    assert "EVENTTYPE2.2" not in facts.verbs
    assert facts.verbs == (
        "EVENTTYPE2.1.1", "EVENTTYPE2.1.2", "EVENTTYPE2.2.1", "EVENTTYPE2.2.2", "EVENTTYPE2.1"
    )  # fmt: skip
    for category in facts.categories:
        assert all(f.predicate.label != "EVENTTYPE2.2" for f in facts.class_facts(category))


# ---------------------------------------------------------------------------------------------
# Rule statements
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_rule_statements_are_nec_all(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    statements = facts.rule_statements()
    assert statements
    oracle = case.oracle()
    extensional = case.facts(**EXTENSIONAL)
    for statement in statements:
        form = statement.to_json()
        assert statement.quantifier == NEC_ALL and statement.polarity
        assert statement.level == CLASS and statement.subject.category == THING
        # true as a law, by the engine and by the independent recomputation
        assert oracle.truth(form) is True, form
        assert statement.grounding["fixed"] is True and statement.grounding["test"] == EXACT
        assert statement.grounding["proportion"] == 1.0 and statement.grounding["instances"] >= 1
        # and true of every instance that satisfies the term
        assert oracle.truth({**form, "quantifier": ALL}) is True
        assert oracle.fixed(form["subject"], statement.predicate.label) == 1
        # the bare plural can state it, whatever "all" expresses
        assert facts.truth.bare_plural_expresses(statement)
        assert extensional.truth.bare_plural_expresses(statement)
        assert extensional.truth.evaluate(statement).felicitous
    assert extensional.rule_statements() == statements


@pytest.mark.parametrize("name", WORLDS)
def test_rule_statements_are_the_terms_of_the_minimal_dnf(cases, name) -> None:
    # Recount the terms from the files: the taxonomy's rules from its rules.yaml, and the
    # one-place requirements from definition.json. Each term of each rule is stated or
    # skipped, once.
    case = cases(name)
    facts = case.facts()
    statements = {s.rule: s for s in facts.rule_statements()}
    rules = yaml.safe_load((case.folder / "taxonomy" / "rules.yaml").read_text(encoding="utf-8"))
    rules = [(r["output"], r["inputs"], r["truth_table"]) for r in rules]
    rules += one_place_requirements(case.folder)
    truth = facts.truth
    expected = dict.fromkeys(SKIP_REASONS, 0)
    terms = 0
    for output, inputs, bits in rules:
        table = TruthTable.from_bit_string(bits)
        for number, term in enumerate(minimal_dnf(table), start=1):
            terms += 1
            used = [(name, bit) for name, bit in zip(inputs, term, strict=True) if bit is not None]
            key = (output, number)
            if any(name.startswith("SCALARDIM.") for name, _ in used):
                expected[SKIP_THRESHOLD] += 1
            else:
                subject = CategoryTerm(THING, tuple(Literal(name, bool(bit)) for name, bit in used))
                if len(truth.members(subject)) == 0:
                    expected[SKIP_NO_INSTANCE] += 1
                else:
                    statement = statements.pop(key)
                    assert statement.subject == subject
                    assert statement.predicate.label == output
                    assert statement.predicate.kind == KIND_OF[output.split(".")[0]]
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
    # 20 rules of the taxonomy and 20 one-place requirements
    assert report == {
        "rules": 40,
        "terms": 111,
        "stated": 94,
        "skipped": {
            SKIP_THRESHOLD: 6,
            SKIP_MAX_LITERALS: 0,
            SKIP_NO_WORD: 0,
            SKIP_NO_INSTANCE: 11,
            SKIP_UNCONFIRMED: 0,
        },
    }


def test_long_terms_are_stated_unless_a_cap_is_set(cases) -> None:
    case = cases("default")
    uncapped = case.facts()
    lengths = Counter(len(s.subject.restriction) for s in uncapped.rule_statements())
    # rule statements are exempt from the limits on adjectives and with-phrases: some of the
    # default world's statements have four adjectives, and the limit is three (before stage
    # a5b, one statement had four with-phrases, and the limit is two)
    assert lengths == {1: 24, 2: 30, 3: 9, 4: 31}
    mention = case.config().mention
    assert (mention.max_adjectives, mention.max_with_phrases) == (3, 2)
    adjectives = [
        sum(x.feature.startswith("PROPERTY.") for x in s.subject.restriction)
        for s in uncapped.rule_statements()
    ]
    with_phrases = [
        sum(x.feature.startswith("PART.") for x in s.subject.restriction)
        for s in uncapped.rule_statements()
    ]
    assert max(adjectives) == 4 and max(with_phrases) == 2
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
        assert capped.rule_report["stated"] + sum(skipped.values()) == 111


def test_negated_property_literals_share_one_relative_clause(cases) -> None:
    # "things with wings that are not red and not big can fly": a relative clause joins several
    # verb phrases with "and", so a term with any number of negated PROPERTY literals is stated.
    facts = cases("default").facts()
    negated = [
        sum(not x.positive and x.feature.startswith("PROPERTY.") for x in s.subject.restriction)
        for s in facts.rule_statements()
    ]
    assert Counter(negated)[0] > 0 and sum(count > 1 for count in negated) == 25
    assert max(negated) == 4
    assert "relative_clauses" not in facts.rule_report["skipped"]
    # only a term that reads a scalar threshold, or that no instance satisfies, stays unstated
    skipped = facts.rule_report["skipped"]
    assert {reason for reason, count in skipped.items() if count} == {
        SKIP_THRESHOLD,
        SKIP_NO_INSTANCE,
    }


def test_a_drawn_rule_statement_is_always_nec_all(cases) -> None:
    case = cases("default")
    facts = case.facts()
    oracle = case.oracle()
    rng = Streams(2).propositions
    drawn = [facts.draw_rule_statement(rng) for _ in range(1000)]
    # "all things with wings ..." and the bare plural say the same proposition (CG.61); whether
    # the word or the bare plural is used is the planner's choice, not the proposition's
    assert {p.quantifier for p in drawn} == {NEC_ALL}
    for statement in drawn[:200]:
        assert oracle.truth(statement.to_json()) is True
        assert statement.rule is not None and statement.polarity
        assert statement.subject.category == THING
    assert len({p.rule for p in drawn}) == 94  # every rule statement is drawn
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
    assert facts.rule_statements("PROPERTY.1") == ()  # a free feature has no rule


def test_a_chained_rule_is_stated_with_determined_literals(cases) -> None:
    # In the deep world some rules read determined features, so a term's literal can be a
    # determined feature. The fixed test keeps only the settings that satisfy it.
    case = cases("deep")
    facts = case.facts()
    chained = [
        s
        for s in facts.rule_statements()
        if any(x.feature not in case.world.free for x in s.subject.restriction)
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
    labels = facts.categories if level == CLASS else facts.world.instances[:8]
    drawn = []
    for _ in range(count):
        label = labels[int(rng.integers(len(labels)))]
        if level == CLASS:
            drawn.append(facts.draw_class(rng, label, patients=facts.categories[:4]))
        else:
            drawn.append(facts.draw_instance(rng, label, patients=facts.world.instances[:8]))
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
            count = int(case.world.column(feature)[truth.members(term)].sum())
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
    assert {"PROPERTY", "PART"} <= set(kinds)


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
    # an object relative with probability object_share, and a subject relative otherwise: a
    # one-place or a two-place event type, each with the same chance
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
    # the negated PROPERTY literals need a subject relative to join, so no object relative is
    # drawn
    facts = case.facts(mention={"relative_clauses": {"object_share": 1.0}})
    rng = Streams(4).propositions
    checked = 0
    for category in facts.categories:
        negated = [x for x in facts.restriction_options(CategoryTerm(category)) if not x.positive]
        negated = [x for x in negated if x.feature.startswith("PROPERTY.")]
        for literal in negated[:2]:
            term = facts.draw_clause(rng, CategoryTerm(category, (literal,)))
            if term is not None:
                assert term.clauses[0].agent is None and term.restriction == (literal,)
                checked += 1
    assert checked > 10
    assert isinstance(drawn(max_depth=1)[0].clauses[0], Clause)
