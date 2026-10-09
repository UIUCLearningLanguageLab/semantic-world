"""Stage 4 acceptance tests: grammar and realization, on a fixed set of sentence plans.

All six clause orders and all two-way settings produce the expected words and trees. Every
tree's leaves equal its tokens, and ``interpret(tree)`` recovers the plan exactly. The expected
strings are written in glosses: a content word is its concept label, and an affix follows a
hyphen.
"""

# ruff: noqa: E501

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from semantic_world.corpus import ConfigError
from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_NP,
    GrammarError,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
    check_plan,
    phrase_of,
    term_of,
)
from semantic_world.corpus.interpret import interpret
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    PROJECTION,
    SCALAR,
    SOME,
    VERB,
    CategoryTerm,
    Clause,
    Literal,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import leaves, tree_depth

CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
ALWAYS_CAN = {"can_rate": {"class": 1.0}}
NEVER_CAN = {"can_rate": {"class": 0.0}}
BARE = {"can_rate": {"class": 0.0, "instance": 0.0}}


def report(label: str, aspect: str = "simple") -> tuple[str, str, str]:
    """An event as a sentence records it: its label, its tense, and its aspect."""
    return (label, "past", aspect)


def the(instance: str, noun: str, *restriction: Literal, clause=None, determiner="the"):
    return NounPhrase(INSTANCE_NP, instance, noun, determiner, restriction, clause)


def kind_of(category: str, *restriction: Literal, quantifier=None, clause=None):
    return NounPhrase(CLASS_NP, category, category, quantifier, restriction, clause)


IT = NounPhrase(INSTANCE_NP, "INSTANCE.1.1.1")

# The fixed set, on the tiny world. The instances INSTANCE.1.1.1 (a CATEGORY.1.1), INSTANCE.1.2.1 (a CATEGORY.1.2), and INSTANCE.2.1.1
# (a CATEGORY.2.1) stand in every instance-level and event-level plan. A class-level plan carries
# its quantifier; a bare plural is a subject with no determiner.
PLANS = {
    # class level
    "generic": SentencePlan(kind_of("CATEGORY.1.1"), Predication(CAN, "EVENTTYPE1.1"), NEC_ALL),
    "most": SentencePlan(
        kind_of(
            "CATEGORY.1.1",
            Literal("PROPERTY.2"),
            Literal("PART.1"),
            Literal("PART.3", False),
            quantifier="most",
        ),
        Predication(HAS, "PART.2"),
        MOST,
    ),
    "no": SentencePlan(
        kind_of("CATEGORY.1.1", quantifier="no"), Predication(HAS, "PART.2"), NEC_NO
    ),
    "negative": SentencePlan(
        kind_of("CATEGORY.1"),
        Predication(VERB, "EVENTTYPE2.1.1", False, kind_of("CATEGORY.2", Literal("PROPERTY.4"))),
        ALL,
    ),
    "member": SentencePlan(
        kind_of("CATEGORY.1.1", quantifier="all"), Predication(MEMBER, "CATEGORY.1"), NEC_ALL
    ),
    "pole": SentencePlan(kind_of("CATEGORY.1.1"), Predication(SCALAR, "SCALARDIM.1.HIGH", False)),
    "rule": SentencePlan(
        kind_of(
            "THING",
            Literal("PART.1"),
            Literal("PROPERTY.3", False),
            Literal("PROPERTY.5", False),
            quantifier="all",
        ),
        Predication(CAN, "EVENTTYPE1.2"),
        NEC_ALL,
    ),
    # instance level
    "capacity": SentencePlan(
        the("INSTANCE.1.1.1", "CATEGORY.1.1", Literal("PROPERTY.2")),
        Predication(
            VERB,
            "EVENTTYPE2.1.1",
            False,
            the("INSTANCE.2.1.1", "CATEGORY.2.1", Literal("PART.1"), determiner="a"),
        ),
    ),
    "lacks": SentencePlan(the("INSTANCE.1.1.1", "CATEGORY.1.1"), Predication(HAS, "PART.4", False)),
    "pronoun": SentencePlan(IT, Predication(MEMBER, "CATEGORY.2", False)),
    "big": SentencePlan(
        the("INSTANCE.1.1.1", "CATEGORY.1"), Predication(SCALAR, "SCALARDIM.1.HIGH")
    ),
    "edible": SentencePlan(
        the("INSTANCE.1.1.1", "CATEGORY.1.1"),
        Predication(PROJECTION, "CANBE.EVENTTYPE2.2.1", False),
    ),
    "swims": SentencePlan(the("INSTANCE.1.1.1", "CATEGORY.1.1"), Predication(CAN, "EVENTTYPE1.3")),
    # event level
    "event": SentencePlan(
        the(
            "INSTANCE.1.1.1",
            "CATEGORY.1.1",
            clause=RelativeClause(
                (
                    Predication(
                        CAN,
                        "EVENTTYPE1.3",
                        event="SCENE.1.EVENTINSTANCE.1",
                        tense="past",
                        aspect="simple",
                    ),
                )
            ),
        ),
        Predication(
            VERB,
            "EVENTTYPE2.1.1",
            True,
            the(
                "INSTANCE.2.1.1",
                "CATEGORY.2.1",
                clause=RelativeClause(
                    (
                        Predication(
                            VERB,
                            "EVENTTYPE2.2.1",
                            event="SCENE.1.EVENTINSTANCE.2",
                            tense="past",
                            aspect="simple",
                        ),
                    ),
                    the("INSTANCE.1.2.1", "CATEGORY.1.2"),
                ),
            ),
            "SCENE.1.EVENTINSTANCE.3",
            "past",
            "simple",
        ),
    ),
    "ran": SentencePlan(
        the("INSTANCE.1.2.1", "CATEGORY.1.2", determiner="a"),
        Predication(
            CAN, "EVENTTYPE1.1", event="SCENE.1.EVENTINSTANCE.4", tense="past", aspect="simple"
        ),
    ),
}

ENGLISH = {
    "generic": "CATEGORY.1.1 can EVENTTYPE1.1",
    "most": "most PROPERTY.2 CATEGORY.1.1 with PART.1 and without PART.3 has PART.2",
    "no": "no CATEGORY.1.1 has PART.2",
    "negative": "CATEGORY.1 can not EVENTTYPE2.1.1 PROPERTY.4 CATEGORY.2",
    "member": "all CATEGORY.1.1 is a CATEGORY.1",
    "pole": "CATEGORY.1.1 is not SCALARDIM.1.HIGH",
    "rule": "all THING with PART.1 that is not PROPERTY.3 and not PROPERTY.5 can EVENTTYPE1.2",
    "capacity": "the PROPERTY.2 CATEGORY.1.1 can not EVENTTYPE2.1.1 a CATEGORY.2.1 with PART.1",
    "lacks": "the CATEGORY.1.1 has no PART.4",
    "pronoun": "it is not a CATEGORY.2",
    "big": "the CATEGORY.1 is SCALARDIM.1.HIGH",
    "edible": "the CATEGORY.1.1 is not CANBE.EVENTTYPE2.2.1",
    "swims": "the CATEGORY.1.1 can EVENTTYPE1.3",
    "event": "the CATEGORY.1.1 that EVENTTYPE1.3 EVENTTYPE2.1.1 the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1",
    "ran": "a CATEGORY.1.2 EVENTTYPE1.1",
}


def settings(**grammar) -> dict:
    return {"grammar": {**ALWAYS_CAN, **grammar}}


def glosses(sentence) -> str:
    """The sentence in glosses: the formal rendering without the lexeme labels."""
    words = []
    for word in sentence.formal.split():
        gloss, _, rest = word.partition("/")
        _, hyphen, affix = rest.partition("-")
        words.append(gloss + hyphen + affix)
    return " ".join(words)


def say(case, name: str, **grammar) -> str:
    return glosses(realize(case, name, **grammar))


def realize(case, name: str, **grammar):
    realizer = case.realizer(**settings(**grammar))
    return realizer.realize(PLANS[name], np.random.default_rng(0))


def shape(tree) -> list:
    """A tree down to its phrases: the labels of the sentence's and the verb phrase's parts."""
    parts = []
    for child in tree[1:]:
        if child[0] == "VP":
            parts.append(["VP", *(c[0] for c in child[1:])])
        else:
            parts.append(child[0])
    return parts


def check(case, sentence, **grammar) -> None:
    """The checks that hold for every sentence: the leaves are the tokens, the renderings line
    up with the tokens, and the tree reads back as the plan."""
    lexicon = case.lexicon(**settings(**grammar))
    assert tuple(leaves(sentence.tree)) == sentence.tokens
    assert len(sentence.formal.split()) == len(sentence.conceptual.split()) == len(sentence.tokens)
    quantifier = sentence.plan.quantifier
    assert (
        interpret(sentence.tree, lexicon, sentence.referents, sentence.events, quantifier)
        == sentence.plan
    )
    # the referents alone, without the nouns, are enough
    referents = [referent for referent, _ in sentence.referents]
    assert (
        interpret(sentence.tree, lexicon, referents, sentence.events, quantifier) == sentence.plan
    )


@pytest.fixture
def tiny(cases):
    return cases("tiny")


# ---------------------------------------------------------------------------------------------
# The default, English order
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PLANS))
def test_the_fixed_set_in_the_default_order(tiny, name) -> None:
    sentence = realize(tiny, name)
    assert glosses(sentence) == ENGLISH[name]
    check(tiny, sentence)
    assert sentence.plan == PLANS[name]


def test_the_levels_of_the_fixed_set() -> None:
    levels = {name: plan.level for name, plan in PLANS.items()}
    assert [levels[n] for n in ("generic", "rule", "capacity", "pronoun", "event", "ran")] == [
        CLASS,
        CLASS,
        INSTANCE,
        INSTANCE,
        EVENT,
        EVENT,
    ]
    for plan in PLANS.values():
        check_plan(plan)


def test_trees_of_the_default_order(tiny) -> None:
    the_lexeme = tiny.lexicon().function_word("the").label
    sentence = realize(tiny, "capacity")
    lexicon = tiny.lexicon()

    def lexeme(concept: str) -> str:
        return lexicon.lexemes_of(concept)[0].label

    def word(gloss: str) -> str:
        return lexicon.function_word(gloss).label

    assert sentence.tree == [
        "S",
        [
            "NP-SBJ",
            ["Det", the_lexeme],
            ["AP", ["A", lexeme("PROPERTY.2")]],
            ["N", lexeme("CATEGORY.1.1")],
        ],
        [
            "VP",
            ["AUX", word("can")],
            ["Neg", word("not")],
            ["V", lexeme("EVENTTYPE2.1.1")],
            [
                "NP-OBJ",
                ["Det", word("a")],
                ["N", lexeme("CATEGORY.2.1")],
                ["PP", ["P", word("with")], ["N", lexeme("PART.1")]],
            ],
        ],
    ]
    assert sentence.referents == (
        ("INSTANCE.1.1.1", "CATEGORY.1.1"),
        ("INSTANCE.2.1.1", "CATEGORY.2.1"),
    )
    assert sentence.events == (None,)
    rule = realize(tiny, "rule")
    assert rule.tree == [
        "S",
        [
            "NP-SBJ",
            ["Det", word("all")],
            ["N", lexeme("THING")],
            ["PP", ["P", word("with")], ["N", lexeme("PART.1")]],
            [
                "RC",
                ["Rel", word("that")],
                ["VP", ["AUX", word("is")], ["Neg", word("not")], ["A", lexeme("PROPERTY.3")]],
                ["Conj", word("and")],
                ["VP", ["Neg", word("not")], ["A", lexeme("PROPERTY.5")]],
            ],
        ],
        ["VP", ["AUX", word("can")], ["V", lexeme("EVENTTYPE1.2")]],
    ]
    assert rule.referents == (("THING", "THING"),) and rule.events == (None, None, None)
    event = realize(tiny, "event")
    assert event.tree == [
        "S",
        [
            "NP-SBJ",
            ["Det", word("the")],
            ["N", lexeme("CATEGORY.1.1")],
            ["RC", ["Rel", word("that")], ["VP", ["V", lexeme("EVENTTYPE1.3")]]],
        ],
        [
            "VP",
            ["V", lexeme("EVENTTYPE2.1.1")],
            [
                "NP-OBJ",
                ["Det", word("the")],
                ["N", lexeme("CATEGORY.2.1")],
                [
                    "RC",
                    ["Rel", word("that")],
                    ["NP-SBJ", ["Det", word("the")], ["N", lexeme("CATEGORY.1.2")]],
                    ["VP", ["V", lexeme("EVENTTYPE2.2.1")]],
                ],
            ],
        ],
    ]
    # the referents and the events follow the tree, a node before its children
    assert event.referents == (
        ("INSTANCE.1.1.1", "CATEGORY.1.1"),
        ("INSTANCE.2.1.1", "CATEGORY.2.1"),
        ("INSTANCE.1.2.1", "CATEGORY.1.2"),
    )
    assert event.events == (
        report("SCENE.1.EVENTINSTANCE.1"),
        report("SCENE.1.EVENTINSTANCE.3"),
        report("SCENE.1.EVENTINSTANCE.2"),
    )
    assert event.event_labels == (
        "SCENE.1.EVENTINSTANCE.1",
        "SCENE.1.EVENTINSTANCE.3",
        "SCENE.1.EVENTINSTANCE.2",
    )
    assert [phrase.referent for phrase in event.phrases] == [
        "INSTANCE.1.1.1",
        "INSTANCE.2.1.1",
        "INSTANCE.1.2.1",
    ]
    pronoun = realize(tiny, "pronoun")
    assert pronoun.tree[1] == ["NP-SBJ", ["Pro", word("it")]]
    assert pronoun.referents == (("INSTANCE.1.1.1", None),)
    assert pronoun.tree[2] == [
        "VP",
        ["AUX", word("is")],
        ["Neg", word("not")],
        ["NP-PRD", ["Det", word("a")], ["N", lexeme("CATEGORY.2")]],
    ]


def test_the_renderings(tiny) -> None:
    sentence = realize(tiny, "capacity")
    lexicon = tiny.lexicon()
    assert (
        sentence.conceptual
        == "THE PROPERTY.2 CATEGORY.1.1 CAN NOT EVENTTYPE2.1.1 A CATEGORY.2.1 WITH PART.1"
    )
    assert sentence.formal.split()[:3] == [
        f"the/{lexicon.function_word('the').label}",
        f"PROPERTY.2/{lexicon.lexemes_of('PROPERTY.2')[0].label}",
        f"CATEGORY.1.1/{lexicon.lexemes_of('CATEGORY.1.1')[0].label}",
    ]
    assert [w.split("/")[1] for w in sentence.formal.split()] == list(sentence.tokens)
    assert realize(tiny, "rule").conceptual == (
        "ALL THING WITH PART.1 THAT IS NOT PROPERTY.3 AND NOT PROPERTY.5 CAN EVENTTYPE1.2"
    )


# ---------------------------------------------------------------------------------------------
# The six clause orders
# ---------------------------------------------------------------------------------------------

SUBJECT = "the PROPERTY.2 CATEGORY.1.1"
VERB_COMPLEX = "can not EVENTTYPE2.1.1"
OBJECT = "a CATEGORY.2.1 with PART.1"
CLAUSES = {
    "SVO": (f"{SUBJECT} {VERB_COMPLEX} {OBJECT}", ["NP-SBJ", ["VP", "AUX", "Neg", "V", "NP-OBJ"]]),
    "SOV": (f"{SUBJECT} {OBJECT} {VERB_COMPLEX}", ["NP-SBJ", ["VP", "NP-OBJ", "AUX", "Neg", "V"]]),
    "VSO": (f"{VERB_COMPLEX} {SUBJECT} {OBJECT}", [["VP", "AUX", "Neg", "V"], "NP-SBJ", "NP-OBJ"]),
    "VOS": (f"{VERB_COMPLEX} {OBJECT} {SUBJECT}", [["VP", "AUX", "Neg", "V", "NP-OBJ"], "NP-SBJ"]),
    "OVS": (f"{OBJECT} {VERB_COMPLEX} {SUBJECT}", [["VP", "NP-OBJ", "AUX", "Neg", "V"], "NP-SBJ"]),
    "OSV": (f"{OBJECT} {SUBJECT} {VERB_COMPLEX}", ["NP-OBJ", "NP-SBJ", ["VP", "AUX", "Neg", "V"]]),
}
# The event sentence has two relative clauses. An object relative is the clause's subject and
# its verb in the clause order, so it turns around when the verb comes before the subject.
EVENTS = {
    "SVO": "the CATEGORY.1.1 that EVENTTYPE1.3 EVENTTYPE2.1.1 the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1",
    "SOV": "the CATEGORY.1.1 that EVENTTYPE1.3 the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1 EVENTTYPE2.1.1",
    "VSO": "EVENTTYPE2.1.1 the CATEGORY.1.1 that EVENTTYPE1.3 the CATEGORY.2.1 that EVENTTYPE2.2.1 the CATEGORY.1.2",
    "VOS": "EVENTTYPE2.1.1 the CATEGORY.2.1 that EVENTTYPE2.2.1 the CATEGORY.1.2 the CATEGORY.1.1 that EVENTTYPE1.3",
    "OVS": "the CATEGORY.2.1 that EVENTTYPE2.2.1 the CATEGORY.1.2 EVENTTYPE2.1.1 the CATEGORY.1.1 that EVENTTYPE1.3",
    "OSV": "the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1 the CATEGORY.1.1 that EVENTTYPE1.3 EVENTTYPE2.1.1",
}


@pytest.mark.parametrize("order", CLAUSE_ORDERS)
def test_the_six_clause_orders(tiny, order) -> None:
    grammar = {"word_order": {"clause": order}}
    sentence = realize(tiny, "capacity", **grammar)
    words, expected_shape = CLAUSES[order]
    assert glosses(sentence) == words
    # the object is inside the verb phrase when it is the verb's neighbor, and a daughter of the
    # sentence when the subject stands between them
    assert shape(sentence.tree) == expected_shape
    assert say(tiny, "event", **grammar) == EVENTS[order]
    # an intransitive verb phrase stands after the subject, or before it
    subject_first = order.index("S") < order.index("V")
    assert say(tiny, "generic", **grammar) == (
        "CATEGORY.1.1 can EVENTTYPE1.1" if subject_first else "can EVENTTYPE1.1 CATEGORY.1.1"
    )
    assert say(tiny, "lacks", **grammar) == (
        "the CATEGORY.1.1 has no PART.4" if subject_first else "has no PART.4 the CATEGORY.1.1"
    )
    for name in PLANS:
        sentence = realize(tiny, name, **grammar)
        check(tiny, sentence, **grammar)
        assert sorted(glosses(sentence).split()) == sorted(ENGLISH[name].split())


# ---------------------------------------------------------------------------------------------
# The two-way settings
# ---------------------------------------------------------------------------------------------

TWO_WAY = {
    ("determiner", "after"): {
        "capacity": "PROPERTY.2 CATEGORY.1.1 the can not EVENTTYPE2.1.1 CATEGORY.2.1 a with PART.1",
        "lacks": "CATEGORY.1.1 the has PART.4 no",
        "pronoun": "it is not CATEGORY.2 a",
        "most": "PROPERTY.2 CATEGORY.1.1 most with PART.1 and without PART.3 has PART.2",
    },
    ("adjective", "after"): {
        "capacity": "the CATEGORY.1.1 PROPERTY.2 can not EVENTTYPE2.1.1 a CATEGORY.2.1 with PART.1",
        "negative": "CATEGORY.1 can not EVENTTYPE2.1.1 CATEGORY.2 PROPERTY.4",
    },
    ("with_phrase", "before"): {
        "capacity": "the PROPERTY.2 CATEGORY.1.1 can not EVENTTYPE2.1.1 with PART.1 a CATEGORY.2.1",
        "most": "with PART.1 and without PART.3 most PROPERTY.2 CATEGORY.1.1 has PART.2",
    },
    ("relative_clause", "before"): {
        "event": "that EVENTTYPE1.3 the CATEGORY.1.1 EVENTTYPE2.1.1 that the CATEGORY.1.2 EVENTTYPE2.2.1 the CATEGORY.2.1",
        "rule": "that is not PROPERTY.3 and not PROPERTY.5 all THING with PART.1 can EVENTTYPE1.2",
    },
    ("adposition", "postposition"): {
        "capacity": "the PROPERTY.2 CATEGORY.1.1 can not EVENTTYPE2.1.1 a CATEGORY.2.1 PART.1 with",
        "most": "most PROPERTY.2 CATEGORY.1.1 PART.1 with and PART.3 without has PART.2",
    },
    ("auxiliary", "after"): {
        "capacity": "the PROPERTY.2 CATEGORY.1.1 EVENTTYPE2.1.1 can not a CATEGORY.2.1 with PART.1",
        "lacks": "the CATEGORY.1.1 no PART.4 has",
        "pronoun": "it a CATEGORY.2 is not",
        "big": "the CATEGORY.1 SCALARDIM.1.HIGH is",
        "rule": "all THING with PART.1 that PROPERTY.3 is not and PROPERTY.5 not EVENTTYPE1.2 can",
    },
    ("negation", "before_auxiliary"): {
        "capacity": "the PROPERTY.2 CATEGORY.1.1 not can EVENTTYPE2.1.1 a CATEGORY.2.1 with PART.1",
        "pole": "CATEGORY.1.1 not is SCALARDIM.1.HIGH",
        "rule": "all THING with PART.1 that not is PROPERTY.3 and not PROPERTY.5 can EVENTTYPE1.2",
    },
}


@pytest.mark.parametrize(("setting", "value"), list(TWO_WAY))
def test_every_two_way_setting(tiny, setting, value) -> None:
    grammar = {"word_order": {setting: value}}
    for name, words in TWO_WAY[setting, value].items():
        assert say(tiny, name, **grammar) == words
    changed = 0
    for name in PLANS:
        sentence = realize(tiny, name, **grammar)
        check(tiny, sentence, **grammar)
        assert sorted(glosses(sentence).split()) == sorted(ENGLISH[name].split())
        changed += glosses(sentence) != ENGLISH[name]
    assert changed >= len(TWO_WAY[setting, value])


def test_the_settings_combine(tiny) -> None:
    # every setting at its other value, with the verb first and the object last
    grammar = {
        "word_order": {
            "clause": "VSO",
            "determiner": "after",
            "adjective": "after",
            "with_phrase": "before",
            "relative_clause": "before",
            "adposition": "postposition",
            "auxiliary": "after",
            "negation": "before_auxiliary",
        }
    }
    assert (
        say(tiny, "capacity", **grammar)
        == "EVENTTYPE2.1.1 not can CATEGORY.1.1 PROPERTY.2 the PART.1 with CATEGORY.2.1 a"
    )
    assert say(tiny, "rule", **grammar) == (
        "EVENTTYPE1.2 can that PROPERTY.3 not is and PROPERTY.5 not PART.1 with THING all"
    )
    assert (
        say(tiny, "event", **grammar)
        == "EVENTTYPE2.1.1 that EVENTTYPE1.3 CATEGORY.1.1 the that EVENTTYPE2.2.1 CATEGORY.1.2 the CATEGORY.2.1 the"
    )
    for name in PLANS:
        check(tiny, realize(tiny, name, **grammar), **grammar)


@pytest.mark.parametrize("order", CLAUSE_ORDERS)
def test_every_combination_of_settings_reads_back(tiny, order) -> None:
    names = ("determiner", "adjective", "with_phrase", "relative_clause", "auxiliary")
    for values in itertools.product(("before", "after"), repeat=len(names)):
        for adposition, negation in itertools.product(
            ("preposition", "postposition"), ("after_auxiliary", "before_auxiliary")
        ):
            word_order = dict(zip(names, values, strict=True))
            word_order.update(clause=order, adposition=adposition, negation=negation)
            grammar = {"word_order": word_order}
            for name in ("capacity", "rule", "event", "most", "pronoun"):
                sentence = realize(tiny, name, **grammar)
                check(tiny, sentence, **grammar)
                assert sorted(glosses(sentence).split()) == sorted(ENGLISH[name].split())


# ---------------------------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------------------------

NUMBER = {"number": {"enabled": True}}


def test_with_every_inflection_off_words_have_one_form(tiny) -> None:
    # "all penguin swim"
    assert say(tiny, "member") == "all CATEGORY.1.1 is a CATEGORY.1"
    assert "-" not in " ".join(say(tiny, name) for name in PLANS)
    plain = SentencePlan(
        kind_of("CATEGORY.1.1", quantifier="all"), Predication(CAN, "EVENTTYPE1.1"), NEC_ALL
    )
    realizer = tiny.realizer(grammar=NEVER_CAN)
    assert (
        glosses(realizer.realize(plain, np.random.default_rng(0)))
        == "all CATEGORY.1.1 EVENTTYPE1.1"
    )


def test_number_and_agreement(tiny) -> None:
    grammar = {"morphology": NUMBER}
    # a class-level noun is plural, and the auxiliaries agree: is and has become are and have
    assert say(tiny, "most", **grammar) == (
        "most PROPERTY.2 CATEGORY.1.1-PLURAL with PART.1 and without PART.3 have PART.2"
    )
    assert say(tiny, "no", **grammar) == "no CATEGORY.1.1-PLURAL have PART.2"
    assert say(tiny, "pole", **grammar) == "CATEGORY.1.1-PLURAL are not SCALARDIM.1.HIGH"
    # a plural predicate noun takes no "a"
    assert say(tiny, "member", **grammar) == "all CATEGORY.1.1-PLURAL are CATEGORY.1-PLURAL"
    # can does not agree, and the verb after it is not marked
    assert say(tiny, "generic", **grammar) == "CATEGORY.1.1-PLURAL can EVENTTYPE1.1"
    assert (
        say(tiny, "negative", **grammar)
        == "CATEGORY.1-PLURAL can not EVENTTYPE2.1.1 PROPERTY.4 CATEGORY.2-PLURAL"
    )
    # an instance is singular
    assert say(tiny, "capacity", **grammar) == ENGLISH["capacity"]
    assert say(tiny, "lacks", **grammar) == ENGLISH["lacks"]
    assert say(tiny, "pronoun", **grammar) == ENGLISH["pronoun"]
    assert say(tiny, "event", **grammar) == ENGLISH["event"]
    # agreement holds across a relative clause: "are" in the clause, and "can" after it
    assert say(tiny, "rule", **grammar) == (
        "all THING-PLURAL with PART.1 that are not PROPERTY.3 and not PROPERTY.5 can EVENTTYPE1.2"
    )
    for name in PLANS:
        check(tiny, realize(tiny, name, **grammar), **grammar)


def bare(case, name: str, **morphology) -> str:
    """A sentence without the optional "can", so that its verb is the finite word."""
    realizer = case.realizer(grammar={**NEVER_CAN, "morphology": morphology})
    return glosses(realizer.realize(PLANS[name], np.random.default_rng(0)))


def test_the_verb_takes_the_plural_marker_of_its_subject(tiny) -> None:
    # verb_marks: plural (the default): a verb with a plural subject is marked
    assert bare(tiny, "generic", **NUMBER) == "CATEGORY.1.1-PLURAL EVENTTYPE1.1-PLURAL"
    assert bare(tiny, "event", **NUMBER) == ENGLISH["event"]
    assert bare(tiny, "ran", **NUMBER) == "a CATEGORY.1.2 EVENTTYPE1.1"
    # verb_marks: singular, as in English: "the penguin swims", "penguins swim"
    english = {"number": {"enabled": True, "verb_marks": "singular"}}
    assert bare(tiny, "generic", **english) == "CATEGORY.1.1-PLURAL EVENTTYPE1.1"
    assert bare(tiny, "ran", **english) == "a CATEGORY.1.2 EVENTTYPE1.1-PLURAL"
    assert bare(tiny, "event", **english) == (
        "the CATEGORY.1.1 that EVENTTYPE1.3-PLURAL EVENTTYPE2.1.1-PLURAL the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1-PLURAL"
    )
    # the verb after "can" is never marked, and the auxiliaries follow the subject's number
    assert bare(tiny, "swims", **english) == "the CATEGORY.1.1 can EVENTTYPE1.3"
    assert bare(tiny, "lacks", **english) == "the CATEGORY.1.1 has no PART.4"
    # with agreement off, only nouns are marked
    off = {"number": {"enabled": True, "agreement": False}}
    assert bare(tiny, "generic", **off) == "CATEGORY.1.1-PLURAL EVENTTYPE1.1"
    assert (
        bare(tiny, "most", **off)
        == "most PROPERTY.2 CATEGORY.1.1-PLURAL with PART.1 and without PART.3 has PART.2"
    )
    assert bare(tiny, "member", **off) == "all CATEGORY.1.1-PLURAL is CATEGORY.1-PLURAL"


def test_agreement_across_a_relative_clause(tiny) -> None:
    # The subject's noun and its verb are apart: the clause between them has its own verbs.
    plan = SentencePlan(
        kind_of("CATEGORY.1.1", Literal("PROPERTY.3", False), Literal("PROPERTY.5", False)),
        Predication(VERB, "EVENTTYPE2.1.1", True, kind_of("CATEGORY.2")),
        ALL,
    )
    realizer = tiny.realizer(grammar={**NEVER_CAN, "morphology": NUMBER})
    sentence = realizer.realize(plan, np.random.default_rng(0))
    assert glosses(sentence) == (
        "CATEGORY.1.1-PLURAL that are not PROPERTY.3 and not PROPERTY.5 EVENTTYPE2.1.1-PLURAL CATEGORY.2-PLURAL"
    )
    # in an object relative, the verb agrees with the clause's own subject
    nested = SentencePlan(
        the(
            "INSTANCE.2.1.1",
            "CATEGORY.2.1",
            clause=RelativeClause(
                (
                    Predication(
                        VERB,
                        "EVENTTYPE2.2.1",
                        event="SCENE.1.EVENTINSTANCE.2",
                        tense="past",
                        aspect="simple",
                    ),
                ),
                the("INSTANCE.1.2.1", "CATEGORY.1.2"),
            ),
        ),
        Predication(
            CAN, "EVENTTYPE1.1", event="SCENE.1.EVENTINSTANCE.4", tense="past", aspect="simple"
        ),
    )
    english = {"number": {"enabled": True, "verb_marks": "singular"}}
    realizer = tiny.realizer(grammar={"morphology": english})
    assert glosses(realizer.realize(nested, np.random.default_rng(0))) == (
        "the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1-PLURAL EVENTTYPE1.1-PLURAL"
    )


def test_number_as_a_separate_word(tiny) -> None:
    after = {"number": {"enabled": True, "realization": "word"}}
    assert bare(tiny, "generic", **after) == "CATEGORY.1.1 PLURAL EVENTTYPE1.1 PLURAL"
    assert bare(tiny, "member", **after) == "all CATEGORY.1.1 PLURAL are CATEGORY.1 PLURAL"
    before = {"number": {"enabled": True, "realization": "word", "position": "before"}}
    assert bare(tiny, "generic", **before) == "PLURAL CATEGORY.1.1 PLURAL EVENTTYPE1.1"
    realizer = tiny.realizer(grammar={**NEVER_CAN, "morphology": after})
    sentence = realizer.realize(PLANS["generic"], np.random.default_rng(0))
    lexicon = tiny.lexicon(grammar={"morphology": after})
    plural = lexicon.function_word("PLURAL").label
    noun, verb = (
        lexicon.lexemes_of("CATEGORY.1.1")[0].label,
        lexicon.lexemes_of("EVENTTYPE1.1")[0].label,
    )
    # the marked word is a node with two children, and the marker is a token of its own
    assert sentence.tree == [
        "S",
        ["NP-SBJ", ["N", ["N", noun], ["PLURAL", plural]]],
        ["VP", ["V", ["V", verb], ["PLURAL", plural]]],
    ]
    assert sentence.tokens == (noun, plural, verb, plural)
    assert sentence.conceptual == "CATEGORY.1.1 PLURAL EVENTTYPE1.1 PLURAL"
    assert interpret(sentence.tree, lexicon, sentence.referents, (), NEC_ALL) == PLANS["generic"]
    # as an affix, the marker joins the token
    affixed = tiny.realizer(grammar={**NEVER_CAN, "morphology": NUMBER})
    sentence = affixed.realize(PLANS["generic"], np.random.default_rng(0))
    assert sentence.tokens == (f"{noun}-PLURAL", f"{verb}-PLURAL")
    assert sentence.formal == f"CATEGORY.1.1/{noun}-PLURAL EVENTTYPE1.1/{verb}-PLURAL"
    assert sentence.conceptual == "CATEGORY.1.1-PLURAL EVENTTYPE1.1-PLURAL"


def test_tense_marks_events_only(tiny) -> None:
    past = {"tense": {"enabled": True}}
    assert bare(tiny, "event", **past) == (
        "the CATEGORY.1.1 that EVENTTYPE1.3-PAST EVENTTYPE2.1.1-PAST the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1-PAST"
    )
    assert bare(tiny, "ran", **past) == "a CATEGORY.1.2 EVENTTYPE1.1-PAST"
    # class-level and instance-level propositions are present, and the present is not marked
    for name in ("generic", "capacity", "swims", "rule"):
        assert "PAST" not in bare(tiny, name, **past)
    # the tense is part of the plan, and the present is not marked
    ran = PLANS["ran"]
    present = SentencePlan(ran.subject, dataclasses.replace(ran.predication, tense="present"))
    realizer = tiny.realizer(grammar={"morphology": past})
    sentence = realizer.realize(present, np.random.default_rng(0))
    assert glosses(sentence) == "a CATEGORY.1.2 EVENTTYPE1.1"
    lexicon = tiny.lexicon(grammar={"morphology": past})
    assert interpret(sentence.tree, lexicon, sentence.referents, sentence.events) == present
    # with tense marking off, a past event is not marked either, and still reads back as past
    plain = tiny.realizer().realize(ran, np.random.default_rng(0))
    assert glosses(plain) == "a CATEGORY.1.2 EVENTTYPE1.1"
    assert plain.events == (report("SCENE.1.EVENTINSTANCE.4"),)
    assert interpret(plain.tree, tiny.lexicon(), plain.referents, plain.events) == ran
    word = {"tense": {"enabled": True, "realization": "word", "position": "before"}}
    assert bare(tiny, "ran", **word) == "a CATEGORY.1.2 PAST EVENTTYPE1.1"
    # a verb with a tense marker takes no agreement marker, as in English "chased"
    both = {"tense": {"enabled": True}, "number": {"enabled": True, "verb_marks": "singular"}}
    assert bare(tiny, "ran", **both) == "a CATEGORY.1.2 EVENTTYPE1.1-PAST"
    assert bare(tiny, "generic", **both) == "CATEGORY.1.1-PLURAL EVENTTYPE1.1"


def progressive(name: str) -> SentencePlan:
    """A plan of the fixed set with every event progressive."""

    def phrase(noun_phrase):
        if noun_phrase is None or noun_phrase.clause is None:
            return noun_phrase
        clause = noun_phrase.clause
        return dataclasses.replace(
            noun_phrase,
            clause=RelativeClause(
                tuple(predication(p) for p in clause.predications), phrase(clause.agent)
            ),
        )

    def predication(p):
        aspect = None if p.event is None else "progressive"
        return dataclasses.replace(p, object=phrase(p.object), aspect=aspect)

    plan = PLANS[name]
    return SentencePlan(phrase(plan.subject), predication(plan.predication), plan.quantifier)


def test_aspect_is_part_of_the_plan(tiny) -> None:
    def said(plan, **morphology) -> str:
        realizer = tiny.realizer(grammar={**NEVER_CAN, "morphology": morphology})
        return glosses(realizer.realize(plan, np.random.default_rng(0)))

    aspect = {"aspect": {"enabled": True}}
    assert (
        said(progressive("ran"), **aspect) == "a CATEGORY.1.2 EVENTTYPE1.1 PROGRESSIVE"
    )  # a word, by default
    assert said(progressive("event"), **aspect) == (
        "the CATEGORY.1.1 that EVENTTYPE1.3 PROGRESSIVE EVENTTYPE2.1.1 PROGRESSIVE the CATEGORY.2.1 that the CATEGORY.1.2 EVENTTYPE2.2.1 PROGRESSIVE"
    )
    # a simple event is not marked, and neither is a capacity
    assert said(PLANS["ran"], **aspect) == "a CATEGORY.1.2 EVENTTYPE1.1"
    assert "PROGRESSIVE" not in said(PLANS["capacity"], **aspect)
    affix = {"aspect": {"enabled": True, "realization": "affix"}}
    assert said(progressive("ran"), **affix) == "a CATEGORY.1.2 EVENTTYPE1.1-PROGRESSIVE"
    # past and progressive together: an affix and a word
    both = {"tense": {"enabled": True}, **aspect}
    assert said(progressive("ran"), **both) == "a CATEGORY.1.2 EVENTTYPE1.1-PAST PROGRESSIVE"
    # the grammar never draws the aspect: the same plan gives the same marker every time
    realizer = tiny.realizer(grammar={"morphology": aspect})
    rng = np.random.default_rng(5)
    assert {glosses(realizer.realize(PLANS["ran"], rng)) for _ in range(50)} == {
        "a CATEGORY.1.2 EVENTTYPE1.1"
    }
    # with aspect marking off, a progressive event has no marker, and still reads back
    plain = tiny.realizer().realize(progressive("ran"), np.random.default_rng(0))
    assert glosses(plain) == "a CATEGORY.1.2 EVENTTYPE1.1"
    assert plain.events == (report("SCENE.1.EVENTINSTANCE.4", "progressive"),)
    assert interpret(plain.tree, tiny.lexicon(), plain.referents, plain.events) == progressive(
        "ran"
    )
    for name in PLANS:
        grammar = {"morphology": {**both, **NUMBER}}
        check(tiny, realize(tiny, name, **grammar), **grammar)
        realizer = tiny.realizer(**settings(**grammar))
        sentence = realizer.realize(progressive(name), np.random.default_rng(0))
        check(tiny, sentence, **grammar)


def test_a_word_takes_one_affix(tiny) -> None:
    with pytest.raises(ConfigError) as info:
        tiny.config(
            grammar={
                "morphology": {
                    "tense": {"enabled": True},
                    "aspect": {"enabled": True, "realization": "affix"},
                }
            }
        )
    assert info.value.field == "grammar.morphology.aspect.realization"
    assert "one affix" in info.value.message
    # no clash when the tense is the present, which has no marker, or when one is a word
    tiny.config(
        grammar={
            "morphology": {
                "tense": {"enabled": True},
                "aspect": {"enabled": True, "realization": "affix"},
            }
        },
        propositions={"events": {"tense": "present"}},
    )
    tiny.config(
        grammar={
            "morphology": {
                "tense": {"enabled": True, "realization": "word"},
                "aspect": {"enabled": True, "realization": "affix"},
            }
        }
    )


# ---------------------------------------------------------------------------------------------
# The grammar's own choices
# ---------------------------------------------------------------------------------------------


def test_the_optional_can_of_a_capacity(tiny) -> None:
    assert say(tiny, "generic", can_rate={"class": 1.0}) == "CATEGORY.1.1 can EVENTTYPE1.1"
    assert say(tiny, "generic", can_rate={"class": 0.0}) == "CATEGORY.1.1 EVENTTYPE1.1"
    realizer = tiny.realizer()
    rng = np.random.default_rng(2)
    drawn = [glosses(realizer.realize(PLANS["generic"], rng)) for _ in range(800)]
    assert set(drawn) == {"CATEGORY.1.1 can EVENTTYPE1.1", "CATEGORY.1.1 EVENTTYPE1.1"}
    assert abs(np.mean([words == "CATEGORY.1.1 can EVENTTYPE1.1" for words in drawn]) - 0.5) < 0.05
    # both say the same: the tree reads back as the same plan
    for rate in (0.0, 1.0):
        sentence = realize(tiny, "generic", can_rate={"class": rate})
        read = interpret(sentence.tree, tiny.lexicon(), sentence.referents, (), NEC_ALL)
        assert read == PLANS["generic"]
    # a negative capacity needs "can" to carry "not", at both levels and at any rate
    never = tiny.realizer(grammar=BARE)
    rng = np.random.default_rng(0)
    assert glosses(never.realize(PLANS["negative"], rng)) == ENGLISH["negative"]
    assert glosses(never.realize(PLANS["capacity"], rng)) == ENGLISH["capacity"]
    # "no" replaces the negation, so the rate applies
    none = SentencePlan(
        kind_of("CATEGORY.1.1", quantifier="no"), Predication(CAN, "EVENTTYPE1.1"), NEC_NO
    )
    assert glosses(never.realize(none, rng)) == "no CATEGORY.1.1 EVENTTYPE1.1"
    # an instance's capacity says "can" by default, and drops it at the instance rate
    assert glosses(tiny.realizer(grammar=NEVER_CAN).realize(PLANS["swims"], rng)) == (
        "the CATEGORY.1.1 can EVENTTYPE1.3"
    )
    assert glosses(never.realize(PLANS["swims"], rng)) == "the CATEGORY.1.1 EVENTTYPE1.3"
    half = tiny.realizer(grammar={"can_rate": {"instance": 0.5}})
    drawn = [glosses(half.realize(PLANS["swims"], rng)) for _ in range(800)]
    assert set(drawn) == {"the CATEGORY.1.1 can EVENTTYPE1.3", "the CATEGORY.1.1 EVENTTYPE1.3"}
    assert (
        abs(np.mean([words == "the CATEGORY.1.1 can EVENTTYPE1.3" for words in drawn]) - 0.5) < 0.05
    )
    # the rate applies inside a relative clause too
    clause = SentencePlan(
        the(
            "INSTANCE.1.1.1",
            "CATEGORY.1.1",
            clause=RelativeClause((Predication(CAN, "EVENTTYPE1.1"),)),
        ),
        Predication(IS, "PROPERTY.2"),
    )
    sentence = never.realize(clause, rng)
    assert glosses(sentence) == "the CATEGORY.1.1 that EVENTTYPE1.1 is PROPERTY.2"
    assert interpret(sentence.tree, tiny.lexicon(), sentence.referents, sentence.events) == clause


def test_a_capacity_and_an_event_can_have_the_same_words(tiny) -> None:
    """With "can" dropped and the tense and the aspect unmarked, "the penguin swim" is a
    capacity or an event. The two sentences have the same tokens and the same tree, so the tree
    carries no information about the reading, and both get the same readings. Only the
    sentence's record tells them apart."""
    subject = the("INSTANCE.1.1.1", "CATEGORY.1.1")
    target = the("INSTANCE.2.1.1", "CATEGORY.2.1")
    pairs = [
        (
            Predication(CAN, "EVENTTYPE1.3"),
            Predication(CAN, "EVENTTYPE1.3", True, None, *report("SCENE.1.EVENTINSTANCE.1")),
        ),
        (
            Predication(VERB, "EVENTTYPE2.1.1", True, target),
            Predication(
                VERB,
                "EVENTTYPE2.1.1",
                True,
                target,
                *report("SCENE.1.EVENTINSTANCE.2", "progressive"),
            ),
        ),
    ]
    config = tiny.config(grammar=BARE)
    lexicon = tiny.lexicon(grammar=BARE)
    realizer = tiny.realizer(grammar=BARE)
    for order in CLAUSE_ORDERS:
        grammar = {**BARE, "word_order": {"clause": order}}
        config, lexicon = tiny.config(grammar=grammar), tiny.lexicon(grammar=grammar)
        realizer = tiny.realizer(grammar=grammar)
        for able, happened in pairs:
            capacity = realizer.realize(SentencePlan(subject, able), np.random.default_rng(0))
            event = realizer.realize(SentencePlan(subject, happened), np.random.default_rng(0))
            assert capacity.tokens == event.tokens
            assert capacity.tree == event.tree
            assert capacity.formal == event.formal and capacity.conceptual == event.conceptual
            assert readings(capacity.tree, lexicon, config) == ("capacity", "event")
            assert readings(event.tree, lexicon, config) == ("capacity", "event")
            # the logical forms differ, and the record of the sentence says which one it is
            assert capacity.plan != event.plan
            assert capacity.events == (None,) and event.events == (
                (happened.event, happened.tense, happened.aspect),
            )
            for sentence in (capacity, event):
                read = interpret(sentence.tree, lexicon, sentence.referents, sentence.events)
                assert read == sentence.plan
    # with "can" kept, or with the tense marked, each sentence has one reading
    # ("can" itself expresses ABLE and ABLE_NOW alike, so a capacity with "can" keeps the
    # reading able_now beside capacity)
    for grammar, expected in (
        ({}, (("capacity", "able_now"), ("event",))),
        ({**BARE, "morphology": {"tense": {"enabled": True}}}, (("capacity",), ("event",))),
        (
            {"can_rate": {"instance": 1.0}, "morphology": {"aspect": {"enabled": True}}},
            (("capacity", "able_now"), ("event",)),
        ),
    ):
        config, lexicon = tiny.config(grammar=grammar), tiny.lexicon(grammar=grammar)
        realizer = tiny.realizer(grammar=grammar)
        for (able, happened), _ in zip(pairs, range(2), strict=True):
            capacity = realizer.realize(SentencePlan(subject, able), np.random.default_rng(0))
            event = realizer.realize(SentencePlan(subject, happened), np.random.default_rng(0))
            assert (
                readings(capacity.tree, lexicon, config),
                readings(event.tree, lexicon, config),
            ) == expected


# The quantifier readings of the class-level plans in the default language: "all" and "no" state
# the nec quantifiers, a bare plural states nec_all and what quantifiers.bare_plural.expresses
# lists (most, by default), and a scalar pole has no quantifier reading.
BARE_PLURAL = ("generic", NEC_ALL, MOST)
QUANTIFIER_READINGS = {
    "generic": BARE_PLURAL,
    "most": ("generic", MOST),
    "no": ("generic", NEC_NO),
    "negative": ("generic", MOST),  # a negated bare plural allows the counterparts only
    "member": ("generic", NEC_ALL),
    "pole": ("generic",),
    "rule": ("generic", NEC_ALL),
}


def test_readings_of_the_fixed_set(tiny) -> None:
    config, lexicon = tiny.config(), tiny.lexicon()
    for name, plan in PLANS.items():
        sentence = realize(tiny, name)
        found = readings(sentence.tree, lexicon, config)
        if plan.level == "instance":
            # a verb with "can" expresses ABLE and ABLE_NOW alike (lexicon.can_words: shared)
            expected = (
                ("capacity", "able_now") if plan.predication.kind in (CAN, VERB) else ("capacity",)
            )
        else:
            expected = ("event",)
        assert found == QUANTIFIER_READINGS.get(name, expected), name
    # which quantifiers "all" and "no" state is a setting of the language
    for words, universal in (("extensional", (ALL,)), ("either", (NEC_ALL, ALL))):
        config = tiny.config(quantifiers={"universal_words": words})
        rule = realize(tiny, "rule")
        assert readings(rule.tree, lexicon, config) == ("generic", *universal)
        assert readings(realize(tiny, "generic").tree, lexicon, config) == BARE_PLURAL
    # and so is what a bare plural states
    config = tiny.config(quantifiers={"bare_plural": {"expresses": ["all", "some"]}})
    generic = realize(tiny, "generic")
    assert readings(generic.tree, lexicon, config) == ("generic", NEC_ALL, ALL, SOME)
    config, lexicon = tiny.config(), tiny.lexicon()
    # with "can" dropped, a bare verb after an instance has two readings, and the sentences
    # that say "is", "has", "not", or name a category keep one
    config, lexicon = tiny.config(grammar=BARE), tiny.lexicon(grammar=BARE)
    realizer = tiny.realizer(grammar=BARE)
    found = {}
    for name, plan in PLANS.items():
        sentence = realizer.realize(plan, np.random.default_rng(0))
        found[name] = readings(sentence.tree, lexicon, config)
        level = {"class": "generic", "instance": "capacity", "event": "event"}[plan.level]
        assert level in found[name]
    both = ("capacity", "event")
    assert {name for name, kinds in found.items() if kinds == both} == {"swims", "event", "ran"}
    assert found["capacity"] == ("capacity", "able_now")  # a negative capacity keeps "can"
    assert found["lacks"] == found["pronoun"] == ("capacity",)
    assert found["generic"] == BARE_PLURAL and found["rule"] == ("generic", NEC_ALL)
    # a language whose events are always marked has no ambiguous sentence
    marked = {**BARE, "morphology": {"aspect": {"enabled": True}}}
    config = tiny.config(grammar=marked, documents={"progressive_rate": 1.0})
    lexicon = tiny.lexicon(grammar=marked)
    sentence = tiny.realizer(grammar=marked).realize(PLANS["swims"], np.random.default_rng(0))
    assert readings(sentence.tree, lexicon, config) == ("capacity",)


def test_adjective_order(cases) -> None:
    case = cases("default")
    adjectives = ("PROPERTY.3", "PROPERTY.17", "SCALARDIM.1.HIGH", "PROPERTY.30", "SCALARDIM.2.LOW")
    realizer = case.realizer()
    rank = realizer.adjective_rank
    # one ordering of every adjective concept, drawn once per language
    lexicon = case.lexicon()
    assert sorted(rank) == sorted(c.label for c in lexicon.concepts if c.pos == "adjective")
    assert sorted(rank.values()) == list(range(len(rank)))
    assert case.realizer().adjective_rank == rank
    assert case.realizer(seed=2).adjective_rank != rank
    rng = np.random.default_rng(0)
    for chosen in itertools.combinations(adjectives, 3):
        plan = SentencePlan(
            kind_of("CATEGORY.1.1", *(Literal(a) for a in chosen)),
            Predication(HAS, "PART.1"),
            NEC_ALL,
        )
        words = glosses(realizer.realize(plan, rng)).split()
        said = words[:3]
        assert said == sorted(chosen, key=rank.get)
    # fixed: false makes the order random for each phrase
    free = case.realizer(grammar={"adjective_order": {"fixed": False}})
    plan = SentencePlan(
        kind_of("CATEGORY.1.1", *(Literal(a) for a in adjectives[:3])),
        Predication(HAS, "PART.1"),
        NEC_ALL,
    )
    orders = {tuple(glosses(free.realize(plan, rng)).split()[:3]) for _ in range(200)}
    assert len(orders) == 6
    lexicon = case.lexicon()
    for _ in range(20):
        sentence = free.realize(plan, rng)
        assert interpret(sentence.tree, lexicon, sentence.referents, (), NEC_ALL) == plan


def test_a_mention_picks_one_of_a_concepts_lexemes(tiny) -> None:
    knobs = {"lexicon": {"synonym_rate": 1.0}}
    realizer = tiny.realizer(**knobs)
    lexicon = tiny.lexicon(**knobs)
    rng = np.random.default_rng(1)
    sentences = [realizer.realize(PLANS["capacity"], rng) for _ in range(200)]
    # the formal rendering tells the synonyms apart, and the conceptual rendering does not
    assert len({s.formal for s in sentences}) > 20
    assert {s.conceptual for s in sentences} == {
        "THE PROPERTY.2 CATEGORY.1.1 CAN NOT EVENTTYPE2.1.1 A CATEGORY.2.1 WITH PART.1"
    }
    first, second = lexicon.lexemes_of("CATEGORY.1.1")
    used = {s.tokens[2] for s in sentences}
    assert used == {first.label, second.label}
    for sentence in sentences[:30]:
        assert interpret(sentence.tree, lexicon, sentence.referents) == PLANS["capacity"]


def test_word_order_never_changes_the_grammars_draws(tiny) -> None:
    knobs = {"lexicon": {"synonym_rate": 1.0}}
    morphology = {"aspect": {"enabled": True}}
    base = None
    for order in CLAUSE_ORDERS:
        for side in ("before", "after"):
            grammar = {
                "morphology": morphology,
                "word_order": {"clause": order, "adjective": side, "relative_clause": side},
            }
            realizer = tiny.realizer(grammar=grammar, **knobs)
            rng = np.random.default_rng(9)
            tokens = [sorted(realizer.realize(PLANS[name], rng).tokens) for name in PLANS]
            base = base or tokens
            assert tokens == base  # the same lexemes and the same markers, in another order


def test_the_same_generator_gives_the_same_sentence(tiny) -> None:
    realizer = tiny.realizer(lexicon={"synonym_rate": 1.0})
    for name in PLANS:
        first = realizer.realize(PLANS[name], np.random.default_rng(4))
        assert realizer.realize(PLANS[name], np.random.default_rng(4)) == first


# ---------------------------------------------------------------------------------------------
# Relative clauses
# ---------------------------------------------------------------------------------------------


def test_a_subject_relative_joins_verb_phrases(tiny) -> None:
    def clause_of(*predications: Predication, restriction=()) -> str:
        plan = SentencePlan(
            the(
                "INSTANCE.1.1.1", "CATEGORY.1.1", *restriction, clause=RelativeClause(predications)
            ),
            Predication(IS, "PROPERTY.2"),
        )
        sentence = realize_plan(tiny, plan)
        return glosses(sentence).removeprefix("the CATEGORY.1.1 ").removesuffix(" is PROPERTY.2")

    target = the("INSTANCE.2.1.1", "CATEGORY.2.1")
    can_swim = Predication(CAN, "EVENTTYPE1.1")
    can_not_fly = Predication(CAN, "EVENTTYPE1.2", False)
    chases = Predication(VERB, "EVENTTYPE2.1.1", True, target)
    edible = Predication(PROJECTION, "CANBE.EVENTTYPE2.2.1")
    a_bird = Predication(MEMBER, "CATEGORY.1", False)
    not_red = (Literal("PROPERTY.3", False),)
    assert clause_of(can_swim) == "that can EVENTTYPE1.1"
    # the same auxiliary is not said again
    assert (
        clause_of(can_swim, chases) == "that can EVENTTYPE1.1 and EVENTTYPE2.1.1 the CATEGORY.2.1"
    )
    assert clause_of(can_swim, can_not_fly) == "that can EVENTTYPE1.1 and not EVENTTYPE1.2"
    assert clause_of(edible, a_bird) == "that is CANBE.EVENTTYPE2.2.1 and not a CATEGORY.1"
    # another auxiliary is said
    assert clause_of(can_swim, edible) == "that can EVENTTYPE1.1 and is CANBE.EVENTTYPE2.2.1"
    assert (
        clause_of(edible, can_swim, chases)
        == "that is CANBE.EVENTTYPE2.2.1 and can EVENTTYPE1.1 and EVENTTYPE2.1.1 the CATEGORY.2.1"
    )
    # the negated IS literals come first, and the other verb phrases join them
    assert clause_of(can_swim, restriction=not_red) == "that is not PROPERTY.3 and can EVENTTYPE1.1"
    assert (
        clause_of(edible, restriction=not_red) == "that is not PROPERTY.3 and CANBE.EVENTTYPE2.2.1"
    )


def realize_plan(case, plan: SentencePlan, **grammar):
    realizer = case.realizer(**settings(**grammar))
    sentence = realizer.realize(plan, np.random.default_rng(0))
    check(case, sentence, **grammar)
    return sentence


def test_negated_literals_alone_make_a_relative_clause(tiny) -> None:
    many = (
        Literal("PROPERTY.1", False),
        Literal("PROPERTY.3", False),
        Literal("PROPERTY.5", False),
    )
    plan = SentencePlan(the("INSTANCE.1.1.1", "CATEGORY.1.1", *many), Predication(IS, "PROPERTY.2"))
    assert glosses(realize_plan(tiny, plan)) == (
        "the CATEGORY.1.1 that is not PROPERTY.1 and not PROPERTY.3 and not PROPERTY.5 is PROPERTY.2"
    )
    assert plan.depth() == 1 and tree_depth(realize_plan(tiny, plan).tree) == 1


def test_relative_clauses_nest(tiny) -> None:
    # "the CATEGORY.1.1 that chased the CATEGORY.2.1 that the CATEGORY.1.2 that ran saw ran": depth 3
    inner = the(
        "INSTANCE.1.2.1",
        "CATEGORY.1.2",
        clause=RelativeClause(
            (
                Predication(
                    CAN,
                    "EVENTTYPE1.1",
                    event="SCENE.1.EVENTINSTANCE.1",
                    tense="past",
                    aspect="simple",
                ),
            )
        ),
    )
    middle = the(
        "INSTANCE.2.1.1",
        "CATEGORY.2.1",
        clause=RelativeClause(
            (
                Predication(
                    VERB,
                    "EVENTTYPE2.2.1",
                    event="SCENE.1.EVENTINSTANCE.2",
                    tense="past",
                    aspect="simple",
                ),
            ),
            inner,
        ),
    )
    outer = the(
        "INSTANCE.1.1.1",
        "CATEGORY.1.1",
        clause=RelativeClause(
            (
                Predication(
                    VERB,
                    "EVENTTYPE2.1.1",
                    True,
                    middle,
                    "SCENE.1.EVENTINSTANCE.3",
                    "past",
                    "simple",
                ),
            )
        ),
    )
    plan = SentencePlan(
        outer,
        Predication(
            CAN, "EVENTTYPE1.3", event="SCENE.1.EVENTINSTANCE.4", tense="past", aspect="simple"
        ),
    )
    sentence = realize_plan(tiny, plan)
    assert glosses(sentence) == (
        "the CATEGORY.1.1 that EVENTTYPE2.1.1 the CATEGORY.2.1 that the CATEGORY.1.2 that EVENTTYPE1.1 EVENTTYPE2.2.1 EVENTTYPE1.3"
    )
    assert plan.depth() == 3 and tree_depth(sentence.tree) == 3
    # a node before its children: the innermost clause comes before the verb that follows it
    assert sentence.event_labels == (
        "SCENE.1.EVENTINSTANCE.3",
        "SCENE.1.EVENTINSTANCE.1",
        "SCENE.1.EVENTINSTANCE.2",
        "SCENE.1.EVENTINSTANCE.4",
    )
    assert [r for r, _ in sentence.referents] == [
        "INSTANCE.1.1.1",
        "INSTANCE.2.1.1",
        "INSTANCE.1.2.1",
    ]
    for name, plan in PLANS.items():
        expected = {"rule": 1, "event": 1}.get(name, 0)
        assert plan.depth() == expected == tree_depth(realize(tiny, name).tree)
    for order in CLAUSE_ORDERS:
        realize_plan(
            tiny,
            SentencePlan(
                outer,
                Predication(
                    CAN,
                    "EVENTTYPE1.3",
                    event="SCENE.1.EVENTINSTANCE.4",
                    tense="past",
                    aspect="simple",
                ),
            ),
            word_order={"clause": order},
        )


def test_class_level_relative_clauses(tiny) -> None:
    swim = RelativeClause((Predication(CAN, "EVENTTYPE1.1"),))
    chase = RelativeClause((Predication(VERB, "EVENTTYPE2.1.1", True, kind_of("CATEGORY.2")),))
    chased = RelativeClause((Predication(VERB, "EVENTTYPE2.1.1"),), kind_of("CATEGORY.2"))
    swimmers = kind_of("CATEGORY.2", clause=swim)
    plans = {
        # "penguins that can swim", "owls that eat mice", and "mice that owls eat"
        "can": SentencePlan(kind_of("CATEGORY.1.1", clause=swim), Predication(HAS, "PART.2"), ALL),
        "subject": SentencePlan(
            kind_of("CATEGORY.1.1", clause=chase, quantifier="most"),
            Predication(IS, "PROPERTY.2"),
            MOST,
        ),
        "object": SentencePlan(
            kind_of("CATEGORY.1.1", clause=chased), Predication(SCALAR, "SCALARDIM.1.HIGH", False)
        ),
        # a clause inside a clause, and a clause on the patient
        "nested": SentencePlan(
            kind_of(
                "CATEGORY.1.1",
                clause=RelativeClause((Predication(VERB, "EVENTTYPE2.1.1", True, swimmers),)),
                quantifier="some",
            ),
            Predication(HAS, "PART.2"),
            SOME,
        ),
        "patient": SentencePlan(
            kind_of("CATEGORY.1"), Predication(VERB, "EVENTTYPE2.1.1", True, swimmers), ALL
        ),
        # the negated IS literals join the clause
        "joined": SentencePlan(
            kind_of("CATEGORY.1.1", Literal("PROPERTY.3", False), clause=swim),
            Predication(HAS, "PART.2"),
            ALL,
        ),
    }
    english = {
        "can": "CATEGORY.1.1 that can EVENTTYPE1.1 has PART.2",
        "subject": "most CATEGORY.1.1 that can EVENTTYPE2.1.1 CATEGORY.2 is PROPERTY.2",
        "object": "CATEGORY.1.1 that CATEGORY.2 can EVENTTYPE2.1.1 is not SCALARDIM.1.HIGH",
        "nested": "some CATEGORY.1.1 that can EVENTTYPE2.1.1 CATEGORY.2 that can EVENTTYPE1.1 has PART.2",
        "patient": "CATEGORY.1 can EVENTTYPE2.1.1 CATEGORY.2 that can EVENTTYPE1.1",
        "joined": "CATEGORY.1.1 that is not PROPERTY.3 and can EVENTTYPE1.1 has PART.2",
    }
    bare = {
        "can": "CATEGORY.1.1 that EVENTTYPE1.1 has PART.2",
        "subject": "most CATEGORY.1.1 that EVENTTYPE2.1.1 CATEGORY.2 is PROPERTY.2",
        "object": "CATEGORY.1.1 that CATEGORY.2 EVENTTYPE2.1.1 is not SCALARDIM.1.HIGH",
        "nested": "some CATEGORY.1.1 that EVENTTYPE2.1.1 CATEGORY.2 that EVENTTYPE1.1 has PART.2",
        "patient": "CATEGORY.1 EVENTTYPE2.1.1 CATEGORY.2 that EVENTTYPE1.1",
        "joined": "CATEGORY.1.1 that is not PROPERTY.3 and EVENTTYPE1.1 has PART.2",
    }
    plural = {
        # the verbs of a subject relative agree with the head, and those of an object relative
        # with the clause's own subject
        "can": "CATEGORY.1.1-PLURAL that EVENTTYPE1.1-PLURAL have PART.2",
        "subject": "most CATEGORY.1.1-PLURAL that EVENTTYPE2.1.1-PLURAL CATEGORY.2-PLURAL are PROPERTY.2",
        "object": "CATEGORY.1.1-PLURAL that CATEGORY.2-PLURAL EVENTTYPE2.1.1-PLURAL are not SCALARDIM.1.HIGH",
    }
    for name, plan in plans.items():
        assert glosses(realize_plan(tiny, plan)) == english[name]
        assert glosses(realize_plan(tiny, plan, **NEVER_CAN)) == bare[name]
        for order in CLAUSE_ORDERS:
            realize_plan(tiny, plan, word_order={"clause": order, "relative_clause": "before"})
        if name in plural:
            grammar = {**NEVER_CAN, "morphology": NUMBER}
            assert glosses(realize_plan(tiny, plan, **grammar)) == plural[name]
        # the clause is part of the category term that the truth tests judge
        proposition = plan.proposition()
        assert phrase_of(term_of(plan.subject), plan.subject.determiner) == plan.subject
        assert proposition.subject == term_of(plan.subject)
    assert term_of(plans["can"].subject) == CategoryTerm(
        "CATEGORY.1.1", (), (Clause(CAN, "EVENTTYPE1.1"),)
    )
    assert term_of(plans["subject"].subject).clauses == (
        Clause(VERB, "EVENTTYPE2.1.1", patient=CategoryTerm("CATEGORY.2")),
    )
    assert term_of(plans["object"].subject).clauses == (
        Clause(VERB, "EVENTTYPE2.1.1", agent=CategoryTerm("CATEGORY.2")),
    )
    assert term_of(plans["nested"].subject).clauses == (
        Clause(
            VERB,
            "EVENTTYPE2.1.1",
            patient=CategoryTerm("CATEGORY.2", (), (Clause(CAN, "EVENTTYPE1.1"),)),
        ),
    )
    assert plans["nested"].depth() == 2 and plans["joined"].depth() == 1
    assert plans["patient"].proposition().predicate.patient.clauses == (
        Clause(CAN, "EVENTTYPE1.1"),
    )
    # a term with an object relative has that one clause
    both = CategoryTerm(
        "CATEGORY.1.1",
        (),
        (
            Clause(CAN, "EVENTTYPE1.1"),
            Clause(VERB, "EVENTTYPE2.1.1", agent=CategoryTerm("CATEGORY.2")),
        ),
    )
    with pytest.raises(GrammarError, match="the only relative clause"):
        phrase_of(both)
    # a term with several subject relatives joins them: "that can swim and chase fish"
    joined = CategoryTerm(
        "CATEGORY.1.1",
        (),
        (
            Clause(CAN, "EVENTTYPE1.1"),
            Clause(VERB, "EVENTTYPE2.1.1", patient=CategoryTerm("CATEGORY.2")),
        ),
    )
    plan = SentencePlan(phrase_of(joined), Predication(HAS, "PART.2"), ALL)
    assert (
        glosses(realize_plan(tiny, plan))
        == "CATEGORY.1.1 that can EVENTTYPE1.1 and EVENTTYPE2.1.1 CATEGORY.2 has PART.2"
    )
    assert plan.proposition().subject == joined


# ---------------------------------------------------------------------------------------------
# Plans the grammar does not realize, and trees it does not read
# ---------------------------------------------------------------------------------------------


def test_plans_the_grammar_does_not_realize(tiny) -> None:
    subject = the("INSTANCE.1.1.1", "CATEGORY.1.1")
    target = the("INSTANCE.2.1.1", "CATEGORY.2.1")
    swims = Predication(CAN, "EVENTTYPE1.1")
    bad = {
        "takes no determiner": SentencePlan(
            NounPhrase(INSTANCE_NP, "INSTANCE.1.1.1", None, "the"), swims
        ),
        "with a or the": SentencePlan(
            NounPhrase(INSTANCE_NP, "INSTANCE.1.1.1", "CATEGORY.1.1", "most"), swims
        ),
        "names its own category": SentencePlan(
            NounPhrase(CLASS_NP, "CATEGORY.1.1", "CATEGORY.1"), swims, NEC_ALL
        ),
        "only the subject takes a quantifier": SentencePlan(
            kind_of("CATEGORY.1"),
            Predication(VERB, "EVENTTYPE2.1.1", True, kind_of("CATEGORY.2", quantifier="all")),
            ALL,
        ),
        "about categories": SentencePlan(
            kind_of("CATEGORY.1"), Predication(VERB, "EVENTTYPE2.1.1", True, target), ALL
        ),
        # the quantifier of a class-level sentence, and the word that states it
        "a class-level sentence has a quantifier": SentencePlan(kind_of("CATEGORY.1.1"), swims),
        "does not state the quantifier": SentencePlan(
            kind_of("CATEGORY.1.1", quantifier="most"), swims, NEC_ALL
        ),
        "takes no quantifier": SentencePlan(
            kind_of("CATEGORY.1.1"), Predication(SCALAR, "SCALARDIM.1.HIGH"), NEC_ALL
        ),
        "takes no quantifier ": SentencePlan(
            kind_of("CATEGORY.1.1", quantifier="all"), Predication(SCALAR, "SCALARDIM.1.HIGH")
        ),
        "only a class-level sentence has a quantifier": SentencePlan(subject, swims, NEC_ALL),
        "a verb, and only a verb": SentencePlan(subject, Predication(VERB, "EVENTTYPE2.1.1")),
        "a verb, and only a verb ": SentencePlan(
            subject, Predication(CAN, "EVENTTYPE1.1", True, target)
        ),
        "never negated": SentencePlan(
            subject,
            Predication(
                CAN,
                "EVENTTYPE1.1",
                False,
                event="SCENE.1.EVENTINSTANCE.1",
                tense="past",
                aspect="simple",
            ),
        ),
        "an event is a one-place event type": SentencePlan(
            subject,
            Predication(
                IS, "PROPERTY.1", event="SCENE.1.EVENTINSTANCE.1", tense="past", aspect="simple"
            ),
        ),
        "every verb phrase of an event-level sentence": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1", clause=RelativeClause((swims,))),
            Predication(
                CAN, "EVENTTYPE1.3", event="SCENE.1.EVENTINSTANCE.1", tense="past", aspect="simple"
            ),
        ),
        "about an instance": SentencePlan(
            kind_of("CATEGORY.1"),
            Predication(
                CAN, "EVENTTYPE1.1", event="SCENE.1.EVENTINSTANCE.1", tense="past", aspect="simple"
            ),
        ),
        "no comparison class": SentencePlan(IT, Predication(SCALAR, "SCALARDIM.1.HIGH")),
        "already named": SentencePlan(subject, Predication(MEMBER, "CATEGORY.1.1")),
        "says something about its head": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1", clause=RelativeClause(())), swims
        ),
        "belong in the restriction": SentencePlan(
            the(
                "INSTANCE.1.1.1",
                "CATEGORY.1.1",
                clause=RelativeClause((Predication(IS, "PROPERTY.3", False),)),
            ),
            swims,
        ),
        "belong in the restriction ": SentencePlan(
            the(
                "INSTANCE.1.1.1",
                "CATEGORY.1.1",
                clause=RelativeClause((Predication(HAS, "PART.3"),)),
            ),
            swims,
        ),
        "one verb, whose patient is the head": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1", clause=RelativeClause((swims,), target)), swims
        ),
        "one relative clause": SentencePlan(
            the(
                "INSTANCE.1.1.1",
                "CATEGORY.1.1",
                Literal("PROPERTY.3", False),
                clause=RelativeClause((Predication(VERB, "EVENTTYPE2.1.1"),), target),
            ),
            swims,
        ),
        "never negated ": SentencePlan(
            the("INSTANCE.1.1.1", "CATEGORY.1.1", Literal("SCALARDIM.1.HIGH", False)), swims
        ),
        "a report has a tense": SentencePlan(
            subject, Predication(CAN, "EVENTTYPE1.1", event="SCENE.1.EVENTINSTANCE.9")
        ),
        "only a report of an event, or a sentence about a time point, has a tense": SentencePlan(
            subject, Predication(CAN, "EVENTTYPE1.1", tense="past")
        ),
        "a class-level relative clause holds": SentencePlan(
            kind_of(
                "CATEGORY.1.1",
                clause=RelativeClause((Predication(PROJECTION, "CANBE.EVENTTYPE2.2.1"),)),
            ),
            Predication(HAS, "PART.2"),
            ALL,
        ),
        "a class-level relative clause holds ": SentencePlan(
            kind_of(
                "CATEGORY.1.1", clause=RelativeClause((Predication(CAN, "EVENTTYPE1.1", False),))
            ),
            Predication(HAS, "PART.2"),
            ALL,
        ),
    }
    realizer = tiny.realizer()
    for reason, plan in bad.items():
        with pytest.raises(GrammarError, match=reason.strip()):
            realizer.realize(plan, np.random.default_rng(0))


def test_a_concept_without_a_word_cannot_be_said(tiny) -> None:
    realizer = tiny.realizer(lexicon={"named_proportion": {"event_unary": 0.0}})
    with pytest.raises(GrammarError, match="EVENTTYPE1.1 has no word"):
        realizer.realize(PLANS["generic"], np.random.default_rng(0))
    # a category of event types without a word cannot name a relation either
    general = SentencePlan(
        kind_of("CATEGORY.1"), Predication(VERB, "EVENTTYPE2.1", True, kind_of("CATEGORY.2")), ALL
    )
    unnamed = tiny.realizer(lexicon={"named_proportion": {"event_category": 0.0}})
    with pytest.raises(GrammarError, match="EVENTTYPE2.1 has no word"):
        unnamed.realize(general, np.random.default_rng(0))


def test_trees_that_cannot_be_read(tiny) -> None:
    sentence = realize(tiny, "event")
    lexicon = tiny.lexicon()
    with pytest.raises(GrammarError, match="3 noun phrases, and 2 referents"):
        interpret(sentence.tree, lexicon, sentence.referents[:2], sentence.events)
    with pytest.raises(GrammarError, match="3 verb phrases, and 1 events"):
        interpret(sentence.tree, lexicon, sentence.referents, sentence.events[:1])
    # a verb with a tense marker reports an event, and needs its record
    past = {"morphology": {"tense": {"enabled": True}}}
    marked = realize(tiny, "ran", **past)
    with pytest.raises(GrammarError, match="no event was given"):
        interpret(marked.tree, tiny.lexicon(**settings(**past)), marked.referents, (None,))
    # a verb with "can" states a capacity, and reports no event
    able = realize(tiny, "swims")
    with pytest.raises(GrammarError, match="an event is reported by a verb with no auxiliary"):
        interpret(able.tree, lexicon, able.referents, (report("SCENE.1.EVENTINSTANCE.1"),))
    with pytest.raises(GrammarError, match="an event is reported by a verb with no auxiliary"):
        generic = realize(tiny, "generic")
        interpret(generic.tree, lexicon, generic.referents, (report("SCENE.1.EVENTINSTANCE.1"),))
    with pytest.raises(GrammarError, match="a sentence is an S"):
        interpret(sentence.tree[1], lexicon, sentence.referents[:1])
    generic = realize(tiny, "generic")
    with pytest.raises(GrammarError, match="the referent given is CATEGORY.2"):
        interpret(generic.tree, lexicon, ["CATEGORY.2"])
