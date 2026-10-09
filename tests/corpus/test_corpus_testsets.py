"""Stage 2 acceptance tests: false items.

Every false item fails the independent recomputation of its truth, and differs from its matched
true item by exactly one change.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_world.corpus import Streams
from semantic_world.corpus.histories import scene_events
from semantic_world.corpus.lexicon import THING
from semantic_world.corpus.propositions import (
    ALL,
    CLASS,
    HAS,
    INSTANCE,
    IS,
    LABEL_KEY,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NO,
    QUANTIFIERS,
    SCALAR,
    SOME,
    VERB,
    CategoryTerm,
    Literal,
)
from semantic_world.corpus.testsets import (
    CHANGES,
    PREDICATE,
    QUANTIFIER,
    ROLE,
    SUBJECT,
    candidates,
    changes_for,
    falsify,
)

WORLDS = ("tiny", "default", "deep", "still")


def true_items(case, per_kind: int = 12) -> list:
    """True propositions of every kind, level, and polarity, and some rule statements."""
    facts = case.facts()
    rng = np.random.default_rng(2)
    pool = []
    categories = facts.categories[:: max(1, len(facts.categories) // 8)]
    for category in categories:
        for negative in (False, True):
            pool += facts.class_facts(category, negative, patients=facts.categories[:6])
    labels = case.world.instances
    for instance in labels[:: max(1, len(labels) // 6)]:
        for negative in (False, True):
            pool += facts.instance_facts(instance, negative, patients=labels[:10])
    by_kind: dict[tuple, list] = {}
    for item in pool:
        key = (item.level, item.predicate.kind, item.quantifier, item.polarity)
        by_kind.setdefault(key, []).append(item)
    chosen = []
    for items in by_kind.values():
        picks = rng.choice(len(items), size=min(per_kind, len(items)), replace=False)
        chosen += [items[int(i)] for i in sorted(picks)]
    return chosen + list(facts.rule_statements()[:20])


def differences(true_form: dict, false_form: dict) -> set[str]:
    return {
        key
        for key in ("level", "quantifier", "polarity", "subject", "predicate")
        if true_form.get(key) != false_form.get(key)
    }


@pytest.mark.parametrize("name", WORLDS)
def test_every_false_item_fails_and_differs_by_exactly_one_change(cases, name) -> None:
    case = cases(name)
    facts, oracle = case.facts(), case.oracle()
    rng = Streams(1).tests
    made = dict.fromkeys(CHANGES, 0)
    for item in true_items(case):
        true_form = item.to_json()
        assert oracle.truth(true_form) is True
        for change in CHANGES:
            false = falsify(facts, item, change, rng)
            if false is None:
                # no change of this kind gives a false proposition
                for candidate in candidates(facts, item, change):
                    evaluation = facts.truth.evaluate(candidate)
                    assert (
                        not evaluation.valid
                        or evaluation.true
                        or not facts.expressible(candidate)
                        or (candidate.level == CLASS and not facts.truth.statable(candidate))
                    )
                continue
            made[change] += 1
            false_form = false.to_json()
            # false by the independent recomputation, not vacuous, and statable
            assert oracle.truth(false_form) is False, (true_form, false_form)
            assert false.grounding is not None and false.id is None and false.rule is None
            assert facts.expressible(false)
            if false.level == CLASS:
                assert facts.truth.statable(false)
            changed = differences(true_form, false_form)
            if change == PREDICATE:
                assert changed == {"predicate"}
                kept = {k: v for k, v in true_form["predicate"].items() if k != LABEL_KEY}
                new = {k: v for k, v in false_form["predicate"].items() if k != LABEL_KEY}
                assert kept == new  # the kind, the patient, and the comparison class stay
            elif change == SUBJECT:
                assert changed == {"subject"}
                if item.level == CLASS:
                    assert false.subject.restriction == item.subject.restriction
                    assert facts.level[false.subject.category] == facts.level[item.subject.category]
            elif change == QUANTIFIER:
                assert changed == {"quantifier"} and item.level == CLASS
            else:
                assert changed == {"subject", "predicate"} and item.predicate.kind == VERB
                assert false.subject == item.predicate.patient
                assert false.predicate.patient == item.subject
                assert false.predicate.label == item.predicate.label
    assert all(count > 10 for count in made.values()), made


def test_changes_that_can_apply(cases) -> None:
    facts = cases("tiny").facts()
    verb = next(f for f in facts.class_facts("CATEGORY.1") if f.predicate.kind == VERB)
    feature = facts.class_facts("CATEGORY.1")[0]
    instance = facts.instance_facts("INSTANCE.1.1.1")[0]
    capacity = next(f for f in facts.instance_facts("INSTANCE.1.1.1") if f.predicate.kind == VERB)
    assert changes_for(feature) == (PREDICATE, SUBJECT, QUANTIFIER)
    assert changes_for(verb) == (PREDICATE, SUBJECT, QUANTIFIER, ROLE)
    assert changes_for(instance) == (PREDICATE, SUBJECT)
    assert changes_for(capacity) == (PREDICATE, SUBJECT, ROLE)
    rng = np.random.default_rng(0)
    assert falsify(facts, feature, ROLE, rng) is None  # only verbs have roles to exchange
    assert falsify(facts, instance, QUANTIFIER, rng) is None  # an instance has no quantifier
    with pytest.raises(ValueError, match="unknown change"):
        falsify(facts, feature, "tense", rng)
    assert candidates(facts, instance, QUANTIFIER) == [] and candidates(facts, feature, ROLE) == []


def test_false_events_did_not_happen_in_their_scene(cases) -> None:
    # A false event is an event that did not happen in its scene. It names no event.
    case = cases("default")
    facts = case.facts()
    generator = case.scenes()
    made = dict.fromkeys(CHANGES, 0)
    rng = np.random.default_rng(0)
    instances = case.world.instances
    for number in range(1, 151):
        scene = generator.scene(Streams(1), number, instances[(number * 7) % len(instances)])
        facts.truth.add_scene(scene)
        for event in scene_events(scene)[:6]:
            report = facts.event_fact(event)
            if report is None:
                continue
            assert facts.truth.is_true(report)
            assert changes_for(report) == (
                (PREDICATE, SUBJECT, ROLE) if event.transitive else (PREDICATE, SUBJECT)
            )
            assert falsify(facts, report, QUANTIFIER, rng) is None
            for change in changes_for(report):
                false = falsify(facts, report, change, rng)
                if false is None:
                    continue
                made[change] += 1
                assert false.event is None and false.scene == scene.label
                assert (false.tense, false.aspect) == (report.tense, report.aspect)
                evaluation = facts.truth.evaluate(false)
                assert evaluation.valid and not evaluation.true
                assert false.grounding == evaluation.grounding
                assert {"able", "legal"} <= set(false.grounding)
                # no event of the scene has its event type, its agent, and its patient
                assert not any(
                    e.agent == false.subject
                    and e.patient == false.predicate.patient
                    and false.predicate.label in facts.truth.verb_names(e.type)
                    for e in scene_events(scene)
                )
    assert all(made[change] > 20 for change in (PREDICATE, SUBJECT, ROLE)), made


def test_quantifier_swaps(cases) -> None:
    case = cases("default")
    facts, truth = case.facts(), case.facts().truth
    rng = np.random.default_rng(3)
    seen = set()
    for category in facts.categories[::4]:
        for negative in (False, True):
            for fact in facts.class_facts(category, negative, patients=facts.categories[:3]):
                options = candidates(facts, fact, QUANTIFIER)
                quantifiers = {c.quantifier for c in options}
                assert fact.quantifier not in quantifiers
                assert all(c.polarity == fact.polarity for c in options)
                if fact.predicate.kind == SCALAR:
                    assert options == []  # a scalar pole has no quantifier
                    continue
                if fact.polarity:
                    assert quantifiers == set(QUANTIFIERS) - {fact.quantifier}
                else:
                    # "no" replaces sentence negation, so a negative proposition never takes it
                    assert quantifiers == {MOST, SOME} - {fact.quantifier}
                false = falsify(facts, fact, QUANTIFIER, rng)
                if false is not None:
                    assert not truth.is_true(false) and truth.statable(false)
                    seen.add((fact.predicate.kind, fact.quantifier, false.quantifier))
    # "all" (nec_all) for a "most" fact, as in the specification, and "no" for a "some" fact
    assert (HAS, MOST, NEC_ALL) in seen and (IS, SOME, NEC_NO) in seen
    # under the default words, the extensional quantifiers cannot be said
    assert not any(false in (ALL, NO) for _, _, false in seen)
    # a scalar pole has no quantifier, and membership is true or false of the whole category,
    # so neither has a false quantifier swap
    assert not any(kind == SCALAR for kind, _, _ in seen)


def test_a_false_item_about_a_scalar_pole_keeps_its_comparison_class(cases) -> None:
    case = cases("default")
    facts = case.facts(scalar_adjectives={"z": 0.5})
    rng = np.random.default_rng(1)
    made = 0
    for instance in case.world.instances[::9]:
        for fact in facts.instance_facts(instance):
            if fact.predicate.kind != SCALAR:
                continue
            for change in (PREDICATE, SUBJECT):
                false = falsify(facts, fact, change, rng)
                if false is None:
                    continue
                made += 1
                assert false.predicate.comparison == fact.predicate.comparison
                # the new subject is an instance of the comparison class: the noun still fits
                row = facts.truth.instance_index[false.subject]
                assert false.predicate.comparison in facts.truth.paths[row]
    assert made > 10


def test_rule_statements_have_false_versions(cases) -> None:
    case = cases("default")
    facts, oracle = case.facts(), case.oracle()
    rng = np.random.default_rng(4)
    for statement in facts.rule_statements():
        # the generic noun has no other category of its level to swap in
        assert falsify(facts, statement, SUBJECT, rng) is None
        swapped = falsify(facts, statement, PREDICATE, rng)
        assert swapped is not None and swapped.subject == statement.subject
        assert swapped.subject.category == THING and swapped.rule is None
        assert oracle.truth(swapped.to_json()) is False
        denied = falsify(facts, statement, QUANTIFIER, rng)
        assert denied is not None and denied.quantifier == NEC_NO  # the only statable false one


def test_false_items_are_never_vacuous(cases) -> None:
    case = cases("tiny")
    facts = case.facts()
    rng = np.random.default_rng(6)
    restricted = []
    for category in facts.categories:
        members = facts.truth.members(CategoryTerm(category))
        for feature in facts.features[HAS]:
            term = CategoryTerm(category, (Literal(feature),))
            if 0 < len(facts.truth.members(term)) < len(members):
                restricted += facts.class_facts(term, patients=facts.categories[:2])
    assert len(restricted) > 50
    made = 0
    for fact in restricted:
        false = falsify(facts, fact, SUBJECT, rng)
        if false is not None:
            made += 1
            assert len(facts.truth.members(false.subject)) > 0
            assert false.subject.restriction == fact.subject.restriction
    assert made > 10


def test_false_items_are_reproducible(cases) -> None:
    case = cases("default")
    facts = case.facts()
    items = true_items(case, per_kind=3)

    def run(seed: int):
        rng = Streams(seed).tests
        return [falsify(facts, item, change, rng) for item in items for change in CHANGES]

    assert run(1) == run(1)
    assert run(1) != run(2)
    kinds = {item.predicate.kind for item in items}
    assert {MEMBER, VERB, SCALAR} <= kinds and {INSTANCE, CLASS} == {item.level for item in items}
    assert any(item.quantifier == SOME for item in items) and any(
        item.quantifier == MOST for item in items
    )
