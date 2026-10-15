"""Stage b3a acceptance tests: the corpus's comparisons of scalars (CG.65 and CG.66).

Every comparison sentence, a pole adjective or a comparative at either level, is true by an
independent recomputation from the world run's files (the oracle, not the corpus's ``Truth``),
and every pole modifier of a mention holds of its referent. Every false comparison item is false
by the same recomputation and differs from its true item by its set's change. Every logical
form with a pole records the language's z, every class-level pole names the comparison class of
CG.3 at ``scalar_adjectives.class_levels_up``, and no output names ``THING`` as a comparison
class. The renderings parse back, ``interpret(tree)`` recovers every plan, and the comparative is
realized correctly in all six clause orders, as an affix and as a word, before and after the
adjective. The settings are read, and the statistics count the new sentences.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter

import numpy as np
import pytest

from semantic_world.corpus import interpret, parse_propositional, proposition_of
from semantic_world.corpus.config import ConfigError
from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_NP,
    NounPhrase,
    Predication,
    SentencePlan,
)
from semantic_world.corpus.logical import has_pole
from semantic_world.corpus.mentions import comparative_plan
from semantic_world.corpus.planner import COMPARATIVE_SECTION, COMPARISON_KIND
from semantic_world.corpus.propositions import (
    ABOVE,
    BELOW,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    GREATER,
    INSTANCE,
    CategoryTerm,
    Predicate,
    Proposition,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import leaves
from semantic_world.corpus.testsets import INSTANCE_COMPARISON, PREDICATE, ROLE
from semantic_world.corpus.world import ENTITY, THING

WORLDS = ("tiny", "deep", pytest.param("default", marks=pytest.mark.slow))
MANY = {"documents": {"comparative_rate": 0.6, "instance_description_rate": 0.4}}
"""Many comparative descriptions, so that the tests see every form."""
CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")


def forms_of(corpus) -> list[tuple]:
    """Every sentence of a corpus with its document, as JSON holds it."""
    return [
        (document, sentence, json.loads(json.dumps(sentence.logical_form)))
        for document in corpus.documents
        for sentence in document.sentences
    ]


def pole_modifiers(form: dict) -> list[tuple[str, str, str]]:
    """The pole modifiers of the mentions of an instance-level form: the instance, the noun,
    and the pole."""
    found = []

    def visit(mention: dict) -> None:
        for literal in mention.get("restriction", ()):
            if literal.startswith("SCALARDIM."):
                found.append((mention["instance"], mention["noun"], literal))
        for clause in mention.get("clauses", ()):
            for key in ("agent", "patient"):
                if key in clause:
                    visit(clause[key])

    if form["level"] != CLASS:
        visit(form["subject"])
        if "patient" in form["predicate"]:
            visit(form["predicate"]["patient"])
    return found


# ---------------------------------------------------------------------------------------------
# Acceptance: every comparison sentence is true by the independent recomputation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_comparison_sentence_is_true_by_the_oracle(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    facts, config = corpus.planner.facts, corpus.config
    counts: Counter = Counter()
    for document, sentence, form in forms_of(corpus):
        proposition = sentence.proposition
        kind = proposition.predicate.kind
        # the z is recorded exactly when the form holds a pole, and it is the language's
        if has_pole(form):
            assert form["z"] == config.scalar_z
            assert list(form).index("z") == list(form).index("predicate") + 1
        else:
            assert "z" not in form
        assert '"THING"' not in json.dumps(form["predicate"])  # no THING as a class
        # every pole modifier holds of its referent against the noun's category, by the oracle
        for instance, noun, pole in pole_modifiers(form):
            assert oracle.pole(pole, oracle.below(noun), form["z"])[oracle.row[instance]]
            counts["modifier"] += 1
        if kind not in (COMPARISON, COMPARATIVE):
            continue
        # the main clause is true by the oracle, which reads the scalar columns alone
        assert oracle.truth(form) is True, form
        assert corpus.planner.truth.is_true(proposition)
        predicate = form["predicate"]
        assert predicate["operator"] == proposition.predicate.operator
        if kind == COMPARISON:
            assert predicate["operator"] == (
                ABOVE if predicate["label"].endswith("HIGH") else BELOW
            )
            if proposition.level == CLASS:
                category = proposition.subject.category
                assert predicate["class"] == facts.comparison_class(category)
                if facts.world.category[category].parent is None:
                    assert predicate["class"] == ENTITY
                else:
                    assert predicate["class"] == facts.world.category[category].parent
                counts["class pole"] += 1
            else:
                assert predicate["class"] == form["subject"]["noun"]
                counts["instance pole"] += 1
        else:
            assert predicate["operator"] == GREATER and "class" not in predicate
            assert predicate["word"] == sentence.plan.predication.label
            if proposition.level == CLASS:
                assert sentence.section == "characteristic"
                counts["class comparative"] += 1
            else:
                assert sentence.section == COMPARATIVE_SECTION
                # the two participants of the event sentence before it, the agent as the subject
                before = next(
                    s
                    for s in reversed(document.sentences[: document.sentences.index(sentence)])
                    if s.section == "event"
                )
                event = before.proposition
                assert form["subject"]["instance"] == event.subject
                assert form["predicate"]["patient"]["instance"] == event.predicate.patient
                low = predicate["word"].endswith(".LOW")
                assert (proposition.subject == event.subject) is not low
                counts["instance comparative"] += 1
        # the rendering parses back to the proposition, and the tree reads back as the plan
        text = sentence.propositional
        names = document.referents if config.propositional_referents == "local" else None
        assert proposition_of(parse_propositional(text), names) == proposition
        assert Proposition.from_json(form) == proposition
        record = sentence.sentence
        plan = sentence.plan
        read = interpret(
            record.tree,
            corpus.planner.lexicon,
            record.referents,
            record.events,
            plan.quantifier,
            plan.predication.standard,
        )
        assert read == plan
    assert counts["instance comparative"] > 10 and counts["class comparative"] > 10, counts
    assert counts["class pole"] > 0 and counts["modifier"] > 10, counts
    if name == "default":
        assert counts["instance pole"] > 0, counts


@pytest.mark.parametrize("name", WORLDS)
def test_every_false_comparison_item_is_false_by_the_oracle(cases, runs, name) -> None:
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    seen: Counter = Counter()
    for test_set in corpus.test_sets:
        for true, false in test_set.pairs:
            if true.proposition.predicate.kind not in (COMPARISON, COMPARATIVE):
                continue
            a, b = true.proposition, false.proposition
            seen[test_set.name] += 1
            for item, truth in ((true, True), (false, False)):
                form = item.input["logical_form"]
                assert oracle.truth(form) is truth, form
                assert item.meta["truth"] is truth
                if has_pole(form):
                    assert form["z"] == corpus.config.scalar_z
            assert b.predicate.kind == a.predicate.kind
            if test_set.change == PREDICATE:
                # another pole or another dimension; the arguments and the class stay
                assert b.predicate.label != a.predicate.label
                assert (b.subject, b.predicate.patient) == (a.subject, a.predicate.patient)
                assert b.predicate.comparison == a.predicate.comparison
            elif test_set.change == ROLE:
                assert a.predicate.kind == COMPARATIVE
                assert (b.subject, b.predicate.patient) == (a.predicate.patient, a.subject)
                assert b.predicate.label == a.predicate.label
            else:
                assert test_set.change == "subject" and a.predicate.kind == COMPARISON
                assert b.subject != a.subject and b.predicate == a.predicate
            if test_set.level == INSTANCE_COMPARISON:
                assert a.level == INSTANCE and true.input["document"] is not None
                assert true.meta["level"] == INSTANCE_COMPARISON
    assert seen["instance_comparison_role"] > 0
    # the comparison facts join the class-level sets: a role swap of a comparative, and the
    # predicate and subject swaps of a pole (the tiny world, with one scalar and six categories,
    # has few true poles, and a small run may draw none for its class sets)
    assert seen["class_role"] > 0
    if name != "tiny":
        assert seen["class_predicate"] > 0

    scalars = corpus.planner.facts.scalars
    # a predicate swap needs another dimension on which the comparison is false
    if len(scalars) > 1:
        assert seen["instance_comparison_predicate"] > 0
    else:
        assert seen["instance_comparison_predicate"] == 0


def test_the_comparison_sets_are_in_the_layout(runs) -> None:
    corpus = runs("tiny")
    names = [s.name for s in corpus.test_sets]
    index = names.index("instance_role")
    assert names[index + 1 : index + 3] == [
        "instance_comparison_predicate",
        "instance_comparison_role",
    ]
    for test_set in corpus.test_sets:
        if test_set.level == INSTANCE_COMPARISON:
            assert test_set.kind == ""
            for true, false in test_set.pairs:
                assert true.input["document"] == false.input["document"]
                # the two items mention the shared instances alike
                for item in (true, false):
                    assert item.input["logical_form"]["predicate"]["word"].endswith(".HIGH")


# ---------------------------------------------------------------------------------------------
# The comparison class (CG.3) and the setting class_levels_up
# ---------------------------------------------------------------------------------------------


def test_the_comparison_class_follows_cg3_and_class_levels_up(cases) -> None:
    case = cases("deep")
    world = case.world
    for up in (1, 2, 3):
        facts = case.facts(scalar_adjectives={"class_levels_up": up})
        oracle = case.oracle(scalar_adjectives={"class_levels_up": up})
        checked = 0
        for category in facts.categories:
            path = world.category[category].path
            expected = path[-1 - up] if len(path) > up else ENTITY
            assert facts.comparison_class(category) == expected
            for predicate in facts.class_predicates(category):
                if predicate.kind != COMPARISON:
                    continue
                assert predicate.comparison == expected
                for polarity in (True, False):
                    proposition = Proposition(CLASS, CategoryTerm(category), predicate, polarity)
                    evaluation = facts.truth.evaluate(proposition)
                    assert evaluation.valid
                    form = {**proposition.to_json(), "z": facts.truth.z}
                    assert oracle.truth(form) is evaluation.true
                    checked += 1
        assert checked > 20
        if up >= 2:
            assert all(facts.comparison_class(c) == ENTITY for c in facts.categories)
    # the generic noun has no class-level pole, and a class that is no ancestor is invalid
    facts = case.facts()
    assert not [p for p in facts.class_predicates(THING) if p.kind == COMPARISON]
    stray = Proposition(
        CLASS,
        CategoryTerm(facts.categories[-1]),
        Predicate(COMPARISON, facts.poles[0], comparison=facts.categories[-2]),
    )
    evaluation = facts.truth.evaluate(stray)
    assert not evaluation.valid and "ancestor" in evaluation.reason
    with pytest.raises(ConfigError, match="class_levels_up"):
        case.config(scalar_adjectives={"class_levels_up": 0})


def test_the_truth_tests_judge_with_the_runtime(cases) -> None:
    """A comparison is judged by the world's runtime functions: the pole mask equals ``above``
    or ``below`` for every entity, an instance-level pole and a comparative agree with direct
    arithmetic on the entities' values, and a comparative is never negated."""
    from semantic_world.world.runtime import Mean, above, below, greater

    case = cases("deep")
    facts = case.facts()
    truth, world = facts.truth, case.world
    definition = world.definition
    rng = np.random.default_rng(5)
    for pole in facts.poles:
        scalar, side = world.pole_parts(pole)
        for category in (THING, *facts.categories):
            mask, mean, sd = truth.pole_mask(pole, category)
            judge = above if side == "HIGH" else below
            standard = ENTITY if category == THING else category
            for i in range(world.count):
                assert mask[i] == judge(definition, scalar, i, standard, truth.z)
    instances = world.instances
    for _ in range(100):
        x, y = (instances[int(i)] for i in rng.choice(len(instances), 2, replace=False))
        scalar = facts.scalars[int(rng.integers(len(facts.scalars)))]
        column = world.scalar_column(scalar)
        i, j = world.instance_index[x], world.instance_index[y]
        proposition = Proposition(INSTANCE, x, Predicate(COMPARATIVE, scalar, y))
        evaluation = truth.evaluate(proposition)
        assert evaluation.valid and evaluation.true == (column[i] > column[j])
        assert evaluation.true == greater(definition, scalar, i, j)
        assert evaluation.grounding == {
            "value": round(float(column[i]), 6),
            "other": round(float(column[j]), 6),
            "test": "value",
        }
        negated = Proposition(INSTANCE, x, Predicate(COMPARATIVE, scalar, y), False)
        assert not truth.evaluate(negated).valid
    for a, b in itertools.combinations(facts.categories[:6], 2):
        for scalar in facts.scalars:
            proposition = Proposition(
                CLASS, CategoryTerm(a), Predicate(COMPARATIVE, scalar, CategoryTerm(b))
            )
            evaluation = truth.evaluate(proposition)
            assert evaluation.valid
            assert evaluation.true == greater(definition, scalar, Mean(a), Mean(b))
    # the oracle's own recomputation agrees on every class-level comparative of the facts
    oracle = case.oracle()
    for a in facts.categories:
        for b in facts.comparison_partners(a):
            for scalar in facts.scalars:
                fact = facts.comparison_fact(a, b, scalar)
                if fact is not None:
                    assert oracle.truth(fact.to_json()) is True


# ---------------------------------------------------------------------------------------------
# The grammar: the comparative in every clause order, as an affix and as a word
# ---------------------------------------------------------------------------------------------

SUBJECT = "the CATEGORY.1.1"
OBJECT = "the CATEGORY.2.1"


def glosses(sentence) -> str:
    words = []
    for word in sentence.formal.split():
        gloss, _, rest = word.partition("/")
        _, hyphen, affix = rest.partition("-")
        words.append(gloss + hyphen + affix)
    return " ".join(words)


def comparative_words(realization: str, position: str) -> str:
    """The verb phrase of "is bigger" under the comparative settings."""
    if realization == "affix":
        return "is SCALARDIM.1.HIGH-COMPARATIVE"
    if position == "before":
        return "is COMPARATIVE SCALARDIM.1.HIGH"
    return "is SCALARDIM.1.HIGH COMPARATIVE"


def ordered(order: str, subject: str, verb: str, target: str) -> str:
    parts = {"S": subject, "V": verb, "O": target}
    return " ".join(parts[letter] for letter in order)


@pytest.mark.parametrize("order", CLAUSE_ORDERS)
@pytest.mark.parametrize(
    ("realization", "position"),
    [("affix", "after"), ("affix", "before"), ("word", "after"), ("word", "before")],
)
def test_the_comparative_in_every_clause_order(cases, order, realization, position) -> None:
    case = cases("tiny")
    grammar = {
        "word_order": {"clause": order},
        "morphology": {"comparative": {"realization": realization, "position": position}},
    }
    settings = {"grammar": grammar}
    realizer, lexicon = case.realizer(**settings), case.lexicon(**settings)
    owl = NounPhrase(INSTANCE_NP, "INSTANCE.1.1.1", "CATEGORY.1.1", "the")
    mouse = NounPhrase(INSTANCE_NP, "INSTANCE.2.1.1", "CATEGORY.2.1", "the")
    plan = comparative_plan(owl, mouse, "SCALARDIM.1")
    sentence = realizer.realize(plan, np.random.default_rng(0))
    verb = comparative_words(realization, position)
    assert glosses(sentence) == ordered(order, SUBJECT, verb, f"than {OBJECT}")
    assert tuple(leaves(sentence.tree)) == sentence.tokens
    # the standard is a PP inside the verb phrase when it is the verb's neighbor, and a
    # daughter of the sentence when the subject stands between them
    tree = sentence.tree
    labels = [child[0] for child in tree[1:]]
    verb_phrase = next(child for child in tree[1:] if child[0] == "VP")
    if abs(order.index("V") - order.index("O")) == 1:
        assert (
            labels == ["NP-SBJ", "VP"] if order.index("S") < order.index("V") else ["VP", "NP-SBJ"]
        )
        assert "PP" in [child[0] for child in verb_phrase[1:]]
    else:
        assert "PP" in labels and "PP" not in [child[0] for child in verb_phrase[1:]]
    # the comparative mark is on the adjective: an affix joined to its token, or a word node
    adjective = next(node for node in _nodes(tree) if node[0] == "A")
    if realization == "affix":
        assert sentence.tokens[[t.endswith("-COMPARATIVE") for t in sentence.tokens].index(True)]
        assert any(t.endswith("-COMPARATIVE") for t in sentence.tokens)
    else:
        inner = [c for c in adjective[1:] if not isinstance(c, str)]
        assert [c[0] for c in inner] == (
            ["COMPARATIVE", "A"] if position == "before" else ["A", "COMPARATIVE"]
        )
    # the tree reads back as the plan, the reading is a capacity (the predicate word is an
    # adjective), and the smaller-than plan says the same proposition
    read = interpret(sentence.tree, lexicon, sentence.referents, sentence.events)
    assert read == plan
    assert readings(sentence.tree, lexicon, case.config(**settings)) == ("capacity",)
    low = comparative_plan(owl, mouse, "SCALARDIM.1", low=True)
    assert low.subject == mouse and low.predication.object == owl
    assert low.proposition() == plan.proposition()
    said = realizer.realize(low, np.random.default_rng(0))
    assert glosses(said) == ordered(order, OBJECT, verb.replace("HIGH", "LOW"), f"than {SUBJECT}")
    assert interpret(said.tree, lexicon, said.referents, said.events) == low
    # a class-level comparative: bare plurals, no quantifier reading
    owls = NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1.1")
    mice = NounPhrase(CLASS_NP, "CATEGORY.2.1", "CATEGORY.2.1")
    generic = comparative_plan(owls, mice, "SCALARDIM.1")
    sentence = realizer.realize(generic, np.random.default_rng(0))
    assert glosses(sentence) == ordered(order, "CATEGORY.1.1", verb, "than CATEGORY.2.1")
    assert interpret(sentence.tree, lexicon, sentence.referents, sentence.events) == generic
    assert readings(sentence.tree, lexicon, case.config(**settings)) == ("generic",)


def _nodes(tree) -> list:
    found = [tree]
    for child in tree[1:]:
        if not isinstance(child, str):
            found += _nodes(child)
    return found


def test_than_follows_the_adposition_setting_and_a_class_pole_needs_its_standard(cases) -> None:
    case = cases("tiny")
    settings = {"grammar": {"word_order": {"adposition": "postposition"}}}
    realizer, lexicon = case.realizer(**settings), case.lexicon(**settings)
    owl = NounPhrase(INSTANCE_NP, "INSTANCE.1.1.1", "CATEGORY.1.1", "the")
    mouse = NounPhrase(INSTANCE_NP, "INSTANCE.2.1.1", "CATEGORY.2.1", "the")
    plan = comparative_plan(owl, mouse, "SCALARDIM.1")
    sentence = realizer.realize(plan, np.random.default_rng(0))
    assert glosses(sentence) == f"{SUBJECT} is SCALARDIM.1.HIGH-COMPARATIVE {OBJECT} than"
    assert interpret(sentence.tree, lexicon, sentence.referents, sentence.events) == plan
    # a class-level pole: the record gives the comparison class, as it gives the quantifier
    owls = NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1.1")
    pole = SentencePlan(owls, Predication(COMPARISON, "SCALARDIM.1.HIGH", standard="CATEGORY.1"))
    sentence = realizer.realize(pole, np.random.default_rng(0))
    assert glosses(sentence) == "CATEGORY.1.1 is SCALARDIM.1.HIGH"
    read = interpret(sentence.tree, lexicon, sentence.referents, sentence.events)
    assert read.predication.standard is None and read != pole
    assert interpret(sentence.tree, lexicon, sentence.referents, (), None, "CATEGORY.1") == pole
    assert pole.proposition() == Proposition(
        CLASS,
        CategoryTerm("CATEGORY.1.1"),
        Predicate(COMPARISON, "SCALARDIM.1.HIGH", comparison="CATEGORY.1"),
        True,
        None,
    )


def test_plans_the_grammar_rejects(cases) -> None:
    from semantic_world.corpus.grammar import GrammarError, check_plan

    owl = NounPhrase(INSTANCE_NP, "INSTANCE.1.1.1", "CATEGORY.1.1", "the")
    mouse = NounPhrase(INSTANCE_NP, "INSTANCE.2.1.1", "CATEGORY.2.1", "the")
    owls = NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1.1")
    mice = NounPhrase(CLASS_NP, "CATEGORY.2.1", "CATEGORY.2.1")
    bad = {
        "never negated": SentencePlan(
            owl, Predication(COMPARATIVE, "SCALARDIM.1.HIGH", False, mouse)
        ),
        "about categories, and the other levels about instances": SentencePlan(
            owl, Predication(COMPARATIVE, "SCALARDIM.1.HIGH", True, mice)
        ),
        "has an object": SentencePlan(owl, Predication(COMPARATIVE, "SCALARDIM.1.HIGH")),
        "compared with itself": SentencePlan(
            owl, Predication(COMPARATIVE, "SCALARDIM.1.HIGH", True, owl)
        ),
        "comparative form of a pole": SentencePlan(
            owl, Predication(COMPARATIVE, "PROPERTY.1", True, mouse)
        ),
        "names its comparison class": SentencePlan(
            owls, Predication(COMPARISON, "SCALARDIM.1.HIGH")
        ),
        "from the noun": SentencePlan(
            owl, Predication(COMPARISON, "SCALARDIM.1.HIGH", standard="CATEGORY.1")
        ),
        "plain categories": SentencePlan(
            owls,
            Predication(
                COMPARATIVE,
                "SCALARDIM.1.HIGH",
                True,
                NounPhrase(CLASS_NP, "CATEGORY.2.1", "CATEGORY.2.1", None, ()),
            ),
            None,
        ),
    }
    bad["plain categories"] = SentencePlan(
        NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1.1", None, ()),
        Predication(
            COMPARATIVE,
            "SCALARDIM.1.HIGH",
            True,
            NounPhrase(CLASS_NP, "CATEGORY.2.1", "CATEGORY.2.1", None, ()),
        ),
    )
    from semantic_world.corpus.propositions import Literal

    bad["plain categories"] = SentencePlan(
        NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1.1", None, (Literal("PROPERTY.1"),)),
        Predication(COMPARATIVE, "SCALARDIM.1.HIGH", True, mice),
    )
    for reason, plan in bad.items():
        with pytest.raises(GrammarError, match=reason):
            check_plan(plan)
    check_plan(comparative_plan(owls, mice, "SCALARDIM.1"))
    check_plan(comparative_plan(owl, mouse, "SCALARDIM.1", low=True))


# ---------------------------------------------------------------------------------------------
# Settings, statistics, and the planner
# ---------------------------------------------------------------------------------------------


def test_the_comparative_rate_and_the_statistics(cases, runs) -> None:
    case = cases("tiny")
    corpus = runs("tiny", **MANY)
    stats = corpus.stats["comparisons"]
    sentences = [s for d in corpus.documents for s in d.sentences]
    by_kind = Counter((s.proposition.predicate.kind, s.proposition.level) for s in sentences)
    assert stats["comparison_sentences"] == {
        "class": by_kind[(COMPARISON, CLASS)],
        "instance": by_kind[(COMPARISON, INSTANCE)],
    }
    assert stats["comparative_sentences"] == {
        "class": by_kind[(COMPARATIVE, CLASS)],
        "instance": by_kind[(COMPARATIVE, INSTANCE)],
    }
    assert stats["comparative_descriptions"] == sum(
        s.section == COMPARATIVE_SECTION for s in sentences
    )
    assert stats["comparative_descriptions"] == by_kind[(COMPARATIVE, INSTANCE)] > 0
    assert stats["pole_modifiers"] == sum(
        1
        for s in sentences
        for phrase in s.plan.noun_phrases()
        if phrase.kind == INSTANCE_NP and any(x.pole for x in phrase.restriction)
    )
    assert (
        corpus.stats["quantifiers"]["stated"]["comparative"]["count"]
        == by_kind[(COMPARATIVE, CLASS)]
    )
    # the rate is a setting: 0 leaves the narratives without comparative descriptions
    silent = runs("tiny", documents={"comparative_rate": 0.0})
    assert not any(s.section == COMPARATIVE_SECTION for d in silent.documents for s in d.sentences)
    assert silent.stats["comparisons"]["comparative_descriptions"] == 0
    # a comparison fact of a category document compares the topic with one of its partners
    facts = corpus.planner.facts
    for document in corpus.documents:
        if document.type != "encyclopedic_category":
            continue
        for sentence in document.sentences:
            if sentence.proposition.predicate.kind != COMPARATIVE:
                continue
            subject = sentence.plan.subject.referent
            other = sentence.plan.predication.object.referent
            assert subject == document.topic
            assert other in facts.comparison_partners(document.topic)
    assert COMPARISON_KIND in corpus.planner._available_facts(facts.categories[0])
    # the settings are read, with their defaults
    config = case.config()
    assert config.documents.comparative_rate == 0.1 and config.class_levels_up == 1
    assert config.grammar.morphology.comparative.realization == "affix"
    assert config.grammar.morphology.comparative.position == "after"
    resolved = config.resolved()
    assert resolved["grammar"]["morphology"]["comparative"] == {
        "realization": "affix",
        "position": "after",
    }
    assert resolved["scalar_adjectives"] == {"z": 1.0, "class_levels_up": 1}
    assert resolved["documents"]["comparative_rate"] == 0.1
    for bad, message in (
        ({"grammar": {"morphology": {"comparative": {"enabled": True}}}}, "enabled"),
        ({"grammar": {"morphology": {"comparative": {"realization": "clitic"}}}}, "realization"),
        ({"documents": {"comparative_rate": 2}}, "comparative_rate"),
    ):
        with pytest.raises(ConfigError, match=message):
            case.config(**bad)


def test_comparison_partners(cases) -> None:
    case = cases("deep")
    facts = case.facts()
    world = case.world
    for topic in facts.categories:
        partners = facts.comparison_partners(topic)
        info = world.category[topic]
        siblings = [
            c
            for c in facts.categories
            if c != topic and info.parent is not None and world.category[c].parent == info.parent
        ]
        assert partners[: len(siblings)] == tuple(siblings)
        others = partners[len(siblings) :]
        assert len(others) == max(1, len(siblings)) or not [
            c
            for c in facts.categories
            if c != topic and facts.level[c] == info.level and c not in siblings
        ]
        assert all(facts.level[c] == info.level and c != topic for c in partners)
        assert len(set(partners)) == len(partners)
