"""Stage 2: the truth tests, against an independent recomputation from the taxonomy's files.

``truth_oracle.Oracle`` reads a taxonomy output folder and judges a logical form without any of
the corpus generator's code. Here every logical form of a whole grid (every subject, predicate,
quantifier, and polarity of a small world) is judged both ways, under several truth settings.
The other tests pin down single rules of "Truth grounding" in the specification.
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
    GENERIC,
    HAS,
    INSTANCE,
    IS,
    LOCAL,
    MEAN,
    MEMBER,
    MOST,
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
    Literal,
    Predicate,
    Proposition,
    Truth,
)

SMALL = ("tiny", "deep", "still")
SETTINGS = {
    "default": {},
    "observed": {"quantifiers": {"all_grounding": "observed"}},
    "generic_all": {"quantifiers": {"generic": {"means": "all"}}},
    "generic_some": {
        "quantifiers": {"generic": {"means": "some"}, "most": {"min_proportion": 0.5}}
    },
    "wide_poles": {"scalar_adjectives": {"z": 0.3}},
}


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


def class_grid(facts, restrictions=((),)):
    subjects = [CategoryTerm(c, r) for c in facts.categories + (THING,) for r in restrictions]
    for subject in subjects:
        for predicate in facts.class_predicates(subject.category):
            for quantifier, polarity in itertools.product(QUANTIFIERS, (True, False)):
                yield Proposition(CLASS, subject, predicate, polarity, quantifier)


def instance_grid(facts):
    for instance in facts.result.instances.labels:
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
    """Random restrictions of one to three literals: IS and HAS features of both polarities,
    free and determined, and scalar poles."""
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
@pytest.mark.parametrize("setting", ["default", "observed"])
def test_restricted_subjects_are_judged_as_the_output_files_say(cases, name, setting) -> None:
    case = cases(name)
    facts = case.facts(**SETTINGS[setting])
    restrictions = restrictions_of(facts, np.random.default_rng(11), 12)
    valid = agree(case, SETTINGS[setting], class_grid(facts, restrictions))
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
# The fixed test
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "default", "deep", "still"])
def test_the_fixed_test_matches_the_defining_vectors(cases, name) -> None:
    # For a category without a restriction, the fixed test is the taxonomy generator's own:
    # categories_defining.csv holds the value of every fixed feature, and NaN elsewhere.
    case = cases(name)
    truth = case.facts().truth
    vectors = case.result.vectors
    labels = vectors.feature_labels[vectors.isa_count :]
    for row, category in enumerate(case.result.tree.categories):
        defining = vectors.defining[row, vectors.isa_count :]
        for label, value in zip(labels, defining, strict=True):
            fixed, test = truth.fixed(CategoryTerm(category.label), label)
            assert test == EXACT
            assert fixed == (None if np.isnan(value) else int(value)), (category.label, label)


@pytest.mark.parametrize("name", ["tiny", "deep", "still"])
def test_the_local_test_never_marks_a_feature_fixed_when_it_is_not(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    exact = facts.truth
    local = Truth(case.config(), case.result, cone_limit=-1)  # nothing is small enough to enumerate
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
    reading = [r.output.label for r in case.result.rules.rules if r.thresholds]
    assert reading
    fixed_somewhere = 0
    for feature in reading:
        for category in case.result.tree.categories:
            value, _ = truth.fixed(CategoryTerm(category.label), feature)
            fixed_somewhere += value is not None
    assert fixed_somewhere > 0
    oracle = case.oracle()
    for feature in reading:
        for category in [c.label for c in case.result.tree.categories] + [THING]:
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
    return float(truth.values[members, truth.features[feature].position].mean())


def find(case, low: float, high: float, fixed: bool | None = None) -> tuple[str, str]:
    """A category and an IS or HAS feature whose proportion lies in a range."""
    facts = case.facts()
    for category in facts.categories:
        for feature in facts.features[IS] + facts.features[HAS]:
            is_fixed = facts.truth.fixed(CategoryTerm(category), feature)[0] is not None
            if low <= proportion(case, category, feature) <= high and fixed in (None, is_fixed):
                return category, feature
    raise AssertionError("the world has no such fact")


def judge(case, category, feature, quantifier, polarity=True, **settings):
    kind = feature.split(".")[0].lower()
    proposition = Proposition(
        CLASS, CategoryTerm(category), Predicate(kind, feature), polarity, quantifier
    )
    return case.facts(**settings).truth.evaluate(proposition)


def test_all_and_no_are_laws_and_not_observations(tiny) -> None:
    # every instance has the feature, but nothing fixes it: true as observed, not as a law
    category, feature = find(tiny, 1.0, 1.0, fixed=False)
    law = judge(tiny, category, feature, ALL)
    assert law.valid and not law.true
    assert law.grounding["proportion"] == 1.0 and law.grounding["fixed"] is False
    assert law.grounding["test"] == EXACT
    seen = judge(tiny, category, feature, ALL, quantifiers={"all_grounding": "observed"})
    assert seen.true and seen.grounding["test"] == OBSERVED
    # a fixed feature is true both ways
    category, feature = find(tiny, 1.0, 1.0, fixed=True)
    assert judge(tiny, category, feature, ALL).true
    assert judge(tiny, category, feature, ALL, quantifiers={"all_grounding": "observed"}).true
    # and no is the same test at the value 0
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    assert judge(tiny, category, feature, NO).true
    assert not judge(tiny, category, feature, ALL).true
    category, feature = find(tiny, 0.0, 0.0, fixed=False)
    assert not judge(tiny, category, feature, NO).true
    assert judge(tiny, category, feature, NO, quantifiers={"all_grounding": "observed"}).true


def test_most_uses_the_minimum_proportion(tiny) -> None:
    category, feature = find(tiny, 0.6, 0.69)  # 2 of 3 instances
    assert not judge(tiny, category, feature, MOST).true
    assert judge(tiny, category, feature, MOST, quantifiers={"most": {"min_proportion": 0.6}}).true
    # most of the others lack it: the negative proposition is about the share without it
    category, feature = find(tiny, 0.1, 0.3)
    assert judge(tiny, category, feature, MOST, polarity=False).true
    assert not judge(tiny, category, feature, MOST).true


def test_some_is_true_above_zero_and_used_only_when_all_is_false(tiny) -> None:
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
    # the implicature follows the law-like reading of "all": seen in all, but not fixed
    category, feature = find(tiny, 1.0, 1.0, fixed=False)
    assert judge(tiny, category, feature, SOME).felicitous
    observed = {"quantifiers": {"all_grounding": "observed"}}
    assert not judge(tiny, category, feature, SOME, **observed).felicitous
    # "some ... not" is left out when none has the feature by law
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    lacking = judge(tiny, category, feature, SOME, polarity=False)
    assert lacking.true and not lacking.felicitous


def test_the_generic_means_what_the_setting_says(tiny) -> None:
    category, feature = find(tiny, 0.1, 0.5)
    assert not judge(tiny, category, feature, GENERIC).true  # most, by default
    assert judge(tiny, category, feature, GENERIC, quantifiers={"generic": {"means": "some"}}).true
    assert not judge(
        tiny, category, feature, GENERIC, quantifiers={"generic": {"means": "all"}}
    ).true
    assert judge(tiny, category, feature, GENERIC, polarity=False).true
    category, feature = find(tiny, 1.0, 1.0, fixed=True)
    for means in ("all", "most", "some"):
        assert judge(
            tiny, category, feature, GENERIC, quantifiers={"generic": {"means": means}}
        ).true
    # a negative generic that means all is the "no" test
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    means_all = {"quantifiers": {"generic": {"means": "all"}}}
    assert judge(tiny, category, feature, GENERIC, polarity=False, **means_all).true
    category, feature = find(tiny, 0.0, 0.0, fixed=False)
    assert not judge(tiny, category, feature, GENERIC, polarity=False, **means_all).true


def test_no_replaces_sentence_negation(tiny) -> None:
    category, feature = find(tiny, 0.0, 0.0, fixed=True)
    for quantifier in (ALL, NO):
        denied = judge(tiny, category, feature, quantifier, polarity=False)
        assert not denied.valid and "negative polarity" in denied.reason
    proposition = Proposition(CLASS, CategoryTerm(category), Predicate(IS, "IS.1"), True, NO)
    assert proposition.negative
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

    def member(subject, category, quantifier=ALL, polarity=True):
        return truth.evaluate(
            Proposition(
                CLASS, CategoryTerm(subject), Predicate(MEMBER, category), polarity, quantifier
            )
        )

    assert member("C1.1", "C1").true and member("C1.1", "C1").grounding["test"] == TREE
    assert member("C1.1", "C1", GENERIC).true  # the generic of a membership sentence means all
    assert not member("C1.1", "C2").true and not member("C1", "C1.1").true
    assert member("C1.1", "C2", NO).true and member("C1.1", "C2", GENERIC, False).true
    assert member("C1.1", "C1.2", NO).true  # siblings share no instance
    assert not member("C1.1", "C1", NO).true and not member("C1", "C1.1", NO).true
    for quantifier in (MOST, SOME):
        assert not member("C1.1", "C1", quantifier).valid
    assert not member("C1", "C1").valid and not member(THING, "C1").valid
    # a restriction does not change membership
    restricted = Proposition(
        CLASS, CategoryTerm("C1.1", (Literal("IS.2"),)), Predicate(MEMBER, "C1"), True, ALL
    )
    assert truth.evaluate(restricted).true == (len(truth.members(restricted.subject)) > 0)


def test_class_level_scalar_poles(cases) -> None:
    case = cases("default")
    facts = case.facts()
    truth = facts.truth
    scalars = case.result.instances.scalars

    def pole(category, label, quantifier=GENERIC, polarity=True, **settings):
        proposition = Proposition(
            CLASS, CategoryTerm(category), Predicate(SCALAR, label), polarity, quantifier
        )
        return case.facts(**settings).truth.evaluate(proposition)

    # only the generic
    for quantifier in (ALL, MOST, SOME, NO):
        assert not pole("C1.1", "SC.1.HIGH", quantifier).valid
    assert not pole(THING, "SC.1.HIGH").valid
    for category in facts.categories:
        node = case.result.tree[category]
        members = truth.members(CategoryTerm(category))
        # a top-level category is compared with all instances in the world
        comparison = (
            np.arange(len(scalars))
            if node.parent is None
            else truth.members(CategoryTerm(node.parent.label))
        )
        for index, scalar in enumerate(("SC.1", "SC.2")):
            mean, sd = scalars[comparison, index].mean(), scalars[comparison, index].std()
            value = scalars[members, index].mean()
            high = pole(category, f"{scalar}.HIGH")
            low = pole(category, f"{scalar}.LOW")
            assert high.true == (value >= mean + sd) and low.true == (value <= mean - sd)
            assert not (high.true and low.true)
            assert high.grounding["test"] == MEAN
            assert high.grounding["comparison"] == (
                THING if node.parent is None else node.parent.label
            )
            assert pole(category, f"{scalar}.HIGH", polarity=False).true == (not high.true)

    # a smaller z makes more categories big or small
    def count(z: float) -> int:
        return sum(
            pole(c, f"SC.1.{side}", scalar_adjectives={"z": z}).true
            for c in facts.categories
            for side in ("HIGH", "LOW")
        )

    assert count(2.0) <= count(1.0) < count(0.3)
    assert count(1.0) > 0


def test_instance_level_scalar_poles_compare_with_the_nouns_category(cases) -> None:
    case = cases("default")
    truth = case.facts(scalar_adjectives={"z": 0.5}).truth
    scalars = case.result.instances.scalars
    differ = 0
    for instance in case.result.instances.labels[::7]:
        row = truth.instance_index[instance]
        path = truth.paths[row]
        verdicts = []
        for comparison in path:
            proposition = Proposition(
                INSTANCE, instance, Predicate(SCALAR, "SC.1.HIGH", comparison=comparison)
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
        outside = Proposition(INSTANCE, instance, Predicate(SCALAR, "SC.1.HIGH", comparison=other))
        assert not truth.evaluate(outside).valid
        assert not truth.evaluate(
            Proposition(INSTANCE, instance, Predicate(SCALAR, "SC.1.HIGH"))
        ).valid
    assert differ > 0  # "the big mouse" can be "the small animal"


def test_a_pole_in_a_restriction_is_relative_to_the_category(cases) -> None:
    case = cases("default")
    truth = case.facts().truth
    scalars = case.result.instances.scalars[:, 0]
    for category in ("C1", "C2.1", "C3.2.1"):
        members = truth.members(CategoryTerm(category))
        big = truth.members(CategoryTerm(category, (Literal("SC.1.HIGH"),)))
        cut = scalars[members].mean() + scalars[members].std()
        assert set(big) == {int(i) for i in members if scalars[i] >= cut}
    negated = Proposition(
        CLASS, CategoryTerm("C1", (Literal("SC.1.HIGH", False),)), Predicate(IS, "IS.1"), True, SOME
    )
    assert "never negated" in truth.evaluate(negated).reason


def test_patient_projections_at_the_class_level(tiny) -> None:
    facts = tiny.facts()
    (projection,) = facts.projections
    values = tiny.result.projections.patient[:, tiny.result.projections.verb_labels.index("V2.2")]

    def judge_projection(category, quantifier, polarity=True, **settings):
        proposition = Proposition(
            CLASS, CategoryTerm(category), Predicate(PROJECTION, projection), polarity, quantifier
        )
        return tiny.facts(**settings).truth.evaluate(proposition)

    observed = {"quantifiers": {"all_grounding": "observed"}}
    for category in facts.categories:
        members = facts.truth.members(CategoryTerm(category))
        share = values[members].mean()
        # most, some, and the generic come from the share of instances with the projection:
        # the same values that the instance-level sentences read
        assert judge_projection(category, MOST).true == (share >= 0.7)
        assert judge_projection(category, SOME).true == (share > 0)
        assert judge_projection(category, GENERIC).true == (share >= 0.7)
        assert judge_projection(category, MOST, False).true == (1 - share >= 0.7)
        assert judge_projection(category, MOST).grounding == {
            "proportion": round(float(share), 6),
            "instances": len(members),
            "test": OBSERVED,
        }
        for index in members:
            instance = Proposition(
                INSTANCE, tiny.result.instances.labels[index], Predicate(PROJECTION, projection)
            )
            assert facts.truth.evaluate(instance).true == bool(values[index])
        # all and no are allowed only under the observed reading
        for quantifier in (ALL, NO):
            assert not judge_projection(category, quantifier).valid
            assert "observed reading" in judge_projection(category, quantifier).reason
        assert judge_projection(category, ALL, **observed).true == (share == 1)
        assert judge_projection(category, NO, **observed).true == (share == 0)


def test_verbs_use_the_relation_matrices_for_any_pair_of_categories(cases) -> None:
    case = cases("default")
    facts = case.facts()
    truth = facts.truth
    relations = case.result.relations
    proportions = case.result.relation_stats.proportions
    # where relation_proportions.csv has a row, the proportion is the same
    checked = 0
    for row in proportions.sample(300, seed=1).iter_rows(named=True):
        if row["verb"] not in facts.verbs:
            continue
        proposition = Proposition(
            CLASS,
            CategoryTerm(row["agent"]),
            Predicate(VERB, row["verb"], CategoryTerm(row["patient"])),
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
    # mixed levels and verb categories are not in the file, and are computed all the same
    verb_category = next(v for v in facts.verbs if not case.result.verbs.tree[v].is_leaf)
    agents = truth.members(CategoryTerm("C1"))
    patients = truth.members(CategoryTerm("C2.1.1"))
    grid_a, grid_p = np.meshgrid(agents, patients, indexing="ij")
    holds = relations.holds(verb_category, grid_a.ravel(), grid_p.ravel())
    predicate = Predicate(VERB, verb_category, CategoryTerm("C2.1.1"))
    evaluation = truth.evaluate(Proposition(CLASS, CategoryTerm("C1"), predicate, True, MOST))
    assert evaluation.grounding["proportion"] == round(float(holds.mean()), 6)
    assert evaluation.grounding["pairs"] == len(agents) * len(patients)
    assert evaluation.true == (holds.mean() >= 0.7)
    # within one category, an instance is not paired with itself
    same = Predicate(VERB, verb_category, CategoryTerm("C1"))
    within = truth.evaluate(Proposition(CLASS, CategoryTerm("C1"), same, True, SOME))
    assert within.grounding["pairs"] == len(agents) * (len(agents) - 1)
    # "all" for a verb is over the existing pairs, whatever the grounding of "all"
    everything = truth.evaluate(Proposition(CLASS, CategoryTerm("C1"), predicate, True, ALL))
    assert everything.true == bool(holds.all())


def test_instance_level_truth_is_read_from_the_values(tiny) -> None:
    facts = tiny.facts()
    truth = facts.truth
    instances = tiny.result.instances
    for row, instance in enumerate(instances.labels):
        for kind in (IS, HAS, CAN):
            for feature in facts.features[kind]:
                value = bool(instances.values[row, truth.features[feature].position])
                for polarity in (True, False):
                    evaluation = truth.evaluate(
                        Proposition(INSTANCE, instance, Predicate(kind, feature), polarity)
                    )
                    assert evaluation.true == (value == polarity)
                    assert evaluation.grounding == {"value": int(value), "test": VALUE}
        for verb in facts.verbs:
            others = [i for i in range(len(instances)) if i != row]
            holds = tiny.result.relations.holds(verb, np.full(len(others), row), np.array(others))
            for other, value in zip(others, holds, strict=True):
                predicate = Predicate(VERB, verb, instances.labels[other])
                assert truth.evaluate(Proposition(INSTANCE, instance, predicate)).true == bool(
                    value
                )
        own = Predicate(VERB, facts.verbs[0], instance)
        assert not truth.evaluate(Proposition(INSTANCE, instance, own)).valid
        # membership: every category on the instance's path, the leaf included
        leaf = instances.leaf_labels[row]
        for category in truth.categories:
            on_path = category == leaf or category in truth.ancestors(leaf)
            member = Proposition(INSTANCE, instance, Predicate(MEMBER, category))
            assert truth.evaluate(member).true == on_path
            assert truth.evaluate(dataclasses.replace(member, polarity=False)).true == (not on_path)


def test_logical_forms_that_cannot_be_judged(tiny) -> None:
    truth = tiny.facts().truth
    subject = CategoryTerm("C1")
    bad = [
        Proposition(CLASS, CategoryTerm("C9"), Predicate(IS, "IS.1"), True, ALL),
        Proposition(CLASS, subject, Predicate(IS, "IS.99"), True, ALL),
        Proposition(CLASS, subject, Predicate(IS, "HAS.1"), True, ALL),
        Proposition(CLASS, subject, Predicate(CAN, "CAN.V1.1"), True, ALL),
        Proposition(CLASS, subject, Predicate(SCALAR, "SC.2.HIGH"), True, GENERIC),
        Proposition(CLASS, subject, Predicate(SCALAR, "SC.1.MIDDLE"), True, GENERIC),
        Proposition(CLASS, subject, Predicate(MEMBER, "C7"), True, ALL),
        Proposition(CLASS, subject, Predicate(PROJECTION, "CANBE.V9"), True, MOST),
        Proposition(CLASS, subject, Predicate(VERB, "V9", CategoryTerm("C2")), True, MOST),
        Proposition(CLASS, subject, Predicate(VERB, "V1.1"), True, MOST),
        Proposition(CLASS, subject, Predicate(IS, "IS.1", CategoryTerm("C2")), True, MOST),
        Proposition(CLASS, subject, Predicate(VERB, "V1.1", "I1.1.1"), True, MOST),
        Proposition(CLASS, subject, Predicate("eats", "V1.1"), True, MOST),
        Proposition(CLASS, subject, Predicate(IS, "IS.1"), True, "few"),
        Proposition(CLASS, subject, Predicate(IS, "IS.1"), True, None),
        Proposition(CLASS, "I1.1.1", Predicate(IS, "IS.1"), True, ALL),
        Proposition(
            CLASS, CategoryTerm("C1", (Literal("CAN.1"),)), Predicate(IS, "IS.1"), True, ALL
        ),
        Proposition(
            CLASS, CategoryTerm("C1", (Literal("IS.77"),)), Predicate(IS, "IS.1"), True, ALL
        ),
        Proposition(INSTANCE, "I9.9.9", Predicate(IS, "IS.1")),
        Proposition(INSTANCE, "I1.1.1", Predicate(IS, "IS.1"), True, ALL),
        Proposition(INSTANCE, "I1.1.1", Predicate(VERB, "V1.1", "I9.9.9")),
        Proposition(INSTANCE, "I1.1.1", Predicate(VERB, "V1.1", CategoryTerm("C1"))),
        Proposition("event", "I1.1.1", Predicate(IS, "IS.1")),
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
        CategoryTerm("C1.3", (Literal("HAS.2", False), Literal("IS.4"))),
        Predicate(CAN, "CAN.3"),
        True,
        MOST,
        {"proportion": 0.93, "fixed": False, "test": "observed"},
        "PR.310",
    )
    assert proposition.to_json() == {
        "id": "PR.310",
        "level": "class",
        "quantifier": "most",
        "polarity": True,
        "subject": {"category": "C1.3", "restriction": ["IS.4", "not HAS.2"]},
        "predicate": {"kind": "can", "feature": "CAN.3"},
        "grounding": {"proportion": 0.93, "fixed": False, "test": "observed"},
    }


def test_json_forms_of_every_predicate_kind() -> None:
    subject = CategoryTerm("C1")
    forms = {
        Predicate(IS, "IS.1"): {"kind": "is", "feature": "IS.1"},
        Predicate(HAS, "HAS.1"): {"kind": "has", "feature": "HAS.1"},
        Predicate(SCALAR, "SC.1.LOW"): {"kind": "scalar", "pole": "SC.1.LOW"},
        Predicate(MEMBER, "C2"): {"kind": "member", "category": "C2"},
        Predicate(PROJECTION, "CANBE.V1.1"): {"kind": "projection", "projection": "CANBE.V1.1"},
        Predicate(VERB, "V1", CategoryTerm("C2", (Literal("IS.3"),))): {
            "kind": "verb",
            "verb": "V1",
            "patient": {"category": "C2", "restriction": ["IS.3"]},
        },
    }
    for predicate, form in forms.items():
        proposition = Proposition(CLASS, subject, predicate, False, SOME)
        assert proposition.to_json()["predicate"] == form
        assert Proposition.from_json(proposition.to_json()) == proposition
    instance = Proposition(INSTANCE, "I1.1.1", Predicate(VERB, "V1.1", "I2.1.2"), False)
    assert instance.to_json() == {
        "id": None,
        "level": "instance",
        "polarity": False,
        "subject": {"instance": "I1.1.1"},
        "predicate": {"kind": "verb", "verb": "V1.1", "patient": {"instance": "I2.1.2"}},
        "grounding": None,
    }
    assert Proposition.from_json(instance.to_json()) == instance
    scalar = Proposition(INSTANCE, "I1.1.1", Predicate(SCALAR, "SC.1.HIGH", comparison="C1"))
    assert scalar.to_json()["predicate"] == {"kind": "scalar", "pole": "SC.1.HIGH", "class": "C1"}
    assert Proposition.from_json(scalar.to_json()) == scalar


def test_a_restriction_is_a_set_in_one_order() -> None:
    a = CategoryTerm(
        "C1", (Literal("HAS.3"), Literal("SC.1.HIGH"), Literal("IS.10"), Literal("IS.2", False))
    )
    b = CategoryTerm(
        "C1",
        (
            Literal("IS.2", False),
            Literal("IS.10"),
            Literal("HAS.3"),
            Literal("SC.1.HIGH"),
            Literal("HAS.3"),
        ),
    )
    assert a == b and hash(a) == hash(b)
    assert [str(x) for x in a.restriction] == ["not IS.2", "IS.10", "HAS.3", "SC.1.HIGH"]
    assert Literal.parse("not HAS.3") == Literal("HAS.3", False)
    assert CategoryTerm.from_json(a.to_json()) == a


def test_equality_ignores_the_label_the_grounding_and_the_rule() -> None:
    plain = Proposition(
        CLASS, CategoryTerm(THING, (Literal("HAS.1"),)), Predicate(CAN, "CAN.1"), True, ALL
    )
    dressed = dataclasses.replace(plain, id="PR.4", grounding={"test": "exact"}, rule=("CAN.1", 2))
    assert plain == dressed and hash(plain) == hash(dressed)
    assert dressed.to_json()["rule"] == {"feature": "CAN.1", "term": 2}
    restored = Proposition.from_json(dressed.to_json())
    assert (restored.id, restored.grounding, restored.rule) == (
        "PR.4",
        {"test": "exact"},
        ("CAN.1", 2),
    )
    assert plain != dataclasses.replace(plain, quantifier=GENERIC)


def test_concepts_a_sentence_needs_words_for() -> None:
    proposition = Proposition(
        CLASS,
        CategoryTerm("C1", (Literal("IS.2", False), Literal("SC.1.HIGH"))),
        Predicate(VERB, "V1.2", CategoryTerm("C2", (Literal("HAS.4"),))),
        True,
        MOST,
    )
    assert proposition.concepts() == ("C1", "IS.2", "SC.1.HIGH", "C2", "HAS.4", "V1.2")
    rule = Proposition(
        CLASS, CategoryTerm(THING, (Literal("HAS.1"),)), Predicate(CAN, "CAN.1"), True, ALL
    )
    assert rule.concepts() == (THING, "HAS.1", "CAN.1")
    # an instance's noun is chosen when it is mentioned, except for a scalar pole's class
    assert Proposition(INSTANCE, "I1.1.1", Predicate(IS, "IS.3")).concepts() == ("IS.3",)
    scalar = Proposition(INSTANCE, "I1.1.1", Predicate(SCALAR, "SC.1.LOW", comparison="C1"))
    assert scalar.concepts() == ("SC.1.LOW", "C1")
