"""Stage 4 acceptance tests: grammar and realization, on a fixed set of sentence plans.

All six clause orders and all two-way settings produce the expected words and trees. Every
tree's leaves equal its tokens, and ``interpret(tree)`` recovers the plan exactly. The expected
strings are written in glosses: a content word is its concept label, and an affix follows a
hyphen.
"""

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
    CAN,
    CLASS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    PROJECTION,
    SCALAR,
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


IT = NounPhrase(INSTANCE_NP, "I1.1.1")

# The fixed set, on the tiny world. The instances I1.1.1 (a C1.1), I1.2.1 (a C1.2), and I2.1.1
# (a C2.1) stand in every instance-level and event-level plan.
PLANS = {
    # class level
    "generic": SentencePlan(kind_of("C1.1"), Predication(CAN, "CAN.1")),
    "most": SentencePlan(
        kind_of(
            "C1.1", Literal("IS.2"), Literal("HAS.1"), Literal("HAS.3", False), quantifier="most"
        ),
        Predication(HAS, "HAS.2"),
    ),
    "no": SentencePlan(kind_of("C1.1", quantifier="no"), Predication(HAS, "HAS.2")),
    "negative": SentencePlan(
        kind_of("C1"), Predication(VERB, "V1.1", False, kind_of("C2", Literal("IS.4")))
    ),
    "member": SentencePlan(kind_of("C1.1", quantifier="all"), Predication(MEMBER, "C1")),
    "pole": SentencePlan(kind_of("C1.1"), Predication(SCALAR, "SC.1.HIGH", False)),
    "rule": SentencePlan(
        kind_of(
            "THING",
            Literal("HAS.1"),
            Literal("IS.3", False),
            Literal("IS.5", False),
            quantifier="all",
        ),
        Predication(CAN, "CAN.2"),
    ),
    # instance level
    "capacity": SentencePlan(
        the("I1.1.1", "C1.1", Literal("IS.2")),
        Predication(VERB, "V1.1", False, the("I2.1.1", "C2.1", Literal("HAS.1"), determiner="a")),
    ),
    "lacks": SentencePlan(the("I1.1.1", "C1.1"), Predication(HAS, "HAS.4", False)),
    "pronoun": SentencePlan(IT, Predication(MEMBER, "C2", False)),
    "big": SentencePlan(the("I1.1.1", "C1"), Predication(SCALAR, "SC.1.HIGH")),
    "edible": SentencePlan(the("I1.1.1", "C1.1"), Predication(PROJECTION, "CANBE.V2.2", False)),
    "swims": SentencePlan(the("I1.1.1", "C1.1"), Predication(CAN, "CAN.3")),
    # event level
    "event": SentencePlan(
        the(
            "I1.1.1",
            "C1.1",
            clause=RelativeClause(
                (Predication(CAN, "CAN.3", event="SN.1.1", tense="past", aspect="simple"),)
            ),
        ),
        Predication(
            VERB,
            "V1.1",
            True,
            the(
                "I2.1.1",
                "C2.1",
                clause=RelativeClause(
                    (Predication(VERB, "V2.1", event="SN.1.2", tense="past", aspect="simple"),),
                    the("I1.2.1", "C1.2"),
                ),
            ),
            "SN.1.3",
            "past",
            "simple",
        ),
    ),
    "ran": SentencePlan(
        the("I1.2.1", "C1.2", determiner="a"),
        Predication(CAN, "CAN.1", event="SN.1.4", tense="past", aspect="simple"),
    ),
}

ENGLISH = {
    "generic": "C1.1 can CAN.1",
    "most": "most IS.2 C1.1 with HAS.1 and without HAS.3 has HAS.2",
    "no": "no C1.1 has HAS.2",
    "negative": "C1 can not V1.1 IS.4 C2",
    "member": "all C1.1 is a C1",
    "pole": "C1.1 is not SC.1.HIGH",
    "rule": "all THING with HAS.1 that is not IS.3 and not IS.5 can CAN.2",
    "capacity": "the IS.2 C1.1 can not V1.1 a C2.1 with HAS.1",
    "lacks": "the C1.1 has no HAS.4",
    "pronoun": "it is not a C2",
    "big": "the C1 is SC.1.HIGH",
    "edible": "the C1.1 is not CANBE.V2.2",
    "swims": "the C1.1 can CAN.3",
    "event": "the C1.1 that CAN.3 V1.1 the C2.1 that the C1.2 V2.1",
    "ran": "a C1.2 CAN.1",
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
    assert interpret(sentence.tree, lexicon, sentence.referents, sentence.events) == sentence.plan
    # the referents alone, without the nouns, are enough
    referents = [referent for referent, _ in sentence.referents]
    assert interpret(sentence.tree, lexicon, referents, sentence.events) == sentence.plan


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
        ["NP-SBJ", ["Det", the_lexeme], ["AP", ["A", lexeme("IS.2")]], ["N", lexeme("C1.1")]],
        [
            "VP",
            ["AUX", word("can")],
            ["Neg", word("not")],
            ["V", lexeme("V1.1")],
            [
                "NP-OBJ",
                ["Det", word("a")],
                ["N", lexeme("C2.1")],
                ["PP", ["P", word("with")], ["N", lexeme("HAS.1")]],
            ],
        ],
    ]
    assert sentence.referents == (("I1.1.1", "C1.1"), ("I2.1.1", "C2.1"))
    assert sentence.events == (None,)
    rule = realize(tiny, "rule")
    assert rule.tree == [
        "S",
        [
            "NP-SBJ",
            ["Det", word("all")],
            ["N", lexeme("THING")],
            ["PP", ["P", word("with")], ["N", lexeme("HAS.1")]],
            [
                "RC",
                ["Rel", word("that")],
                ["VP", ["AUX", word("is")], ["Neg", word("not")], ["A", lexeme("IS.3")]],
                ["Conj", word("and")],
                ["VP", ["Neg", word("not")], ["A", lexeme("IS.5")]],
            ],
        ],
        ["VP", ["AUX", word("can")], ["V", lexeme("CAN.2")]],
    ]
    assert rule.referents == (("THING", "THING"),) and rule.events == (None, None, None)
    event = realize(tiny, "event")
    assert event.tree == [
        "S",
        [
            "NP-SBJ",
            ["Det", word("the")],
            ["N", lexeme("C1.1")],
            ["RC", ["Rel", word("that")], ["VP", ["V", lexeme("CAN.3")]]],
        ],
        [
            "VP",
            ["V", lexeme("V1.1")],
            [
                "NP-OBJ",
                ["Det", word("the")],
                ["N", lexeme("C2.1")],
                [
                    "RC",
                    ["Rel", word("that")],
                    ["NP-SBJ", ["Det", word("the")], ["N", lexeme("C1.2")]],
                    ["VP", ["V", lexeme("V2.1")]],
                ],
            ],
        ],
    ]
    # the referents and the events follow the tree, a node before its children
    assert event.referents == (("I1.1.1", "C1.1"), ("I2.1.1", "C2.1"), ("I1.2.1", "C1.2"))
    assert event.events == (report("SN.1.1"), report("SN.1.3"), report("SN.1.2"))
    assert event.event_labels == ("SN.1.1", "SN.1.3", "SN.1.2")
    assert [phrase.referent for phrase in event.phrases] == ["I1.1.1", "I2.1.1", "I1.2.1"]
    pronoun = realize(tiny, "pronoun")
    assert pronoun.tree[1] == ["NP-SBJ", ["Pro", word("it")]]
    assert pronoun.referents == (("I1.1.1", None),)
    assert pronoun.tree[2] == [
        "VP",
        ["AUX", word("is")],
        ["Neg", word("not")],
        ["NP-PRD", ["Det", word("a")], ["N", lexeme("C2")]],
    ]


def test_the_renderings(tiny) -> None:
    sentence = realize(tiny, "capacity")
    lexicon = tiny.lexicon()
    assert sentence.conceptual == "THE IS.2 C1.1 CAN NOT V1.1 A C2.1 WITH HAS.1"
    assert sentence.formal.split()[:3] == [
        f"the/{lexicon.function_word('the').label}",
        f"IS.2/{lexicon.lexemes_of('IS.2')[0].label}",
        f"C1.1/{lexicon.lexemes_of('C1.1')[0].label}",
    ]
    assert [w.split("/")[1] for w in sentence.formal.split()] == list(sentence.tokens)
    assert realize(tiny, "rule").conceptual == (
        "ALL THING WITH HAS.1 THAT IS NOT IS.3 AND NOT IS.5 CAN CAN.2"
    )


# ---------------------------------------------------------------------------------------------
# The six clause orders
# ---------------------------------------------------------------------------------------------

SUBJECT = "the IS.2 C1.1"
VERB_COMPLEX = "can not V1.1"
OBJECT = "a C2.1 with HAS.1"
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
    "SVO": "the C1.1 that CAN.3 V1.1 the C2.1 that the C1.2 V2.1",
    "SOV": "the C1.1 that CAN.3 the C2.1 that the C1.2 V2.1 V1.1",
    "VSO": "V1.1 the C1.1 that CAN.3 the C2.1 that V2.1 the C1.2",
    "VOS": "V1.1 the C2.1 that V2.1 the C1.2 the C1.1 that CAN.3",
    "OVS": "the C2.1 that V2.1 the C1.2 V1.1 the C1.1 that CAN.3",
    "OSV": "the C2.1 that the C1.2 V2.1 the C1.1 that CAN.3 V1.1",
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
        "C1.1 can CAN.1" if subject_first else "can CAN.1 C1.1"
    )
    assert say(tiny, "lacks", **grammar) == (
        "the C1.1 has no HAS.4" if subject_first else "has no HAS.4 the C1.1"
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
        "capacity": "IS.2 C1.1 the can not V1.1 C2.1 a with HAS.1",
        "lacks": "C1.1 the has HAS.4 no",
        "pronoun": "it is not C2 a",
        "most": "IS.2 C1.1 most with HAS.1 and without HAS.3 has HAS.2",
    },
    ("adjective", "after"): {
        "capacity": "the C1.1 IS.2 can not V1.1 a C2.1 with HAS.1",
        "negative": "C1 can not V1.1 C2 IS.4",
    },
    ("with_phrase", "before"): {
        "capacity": "the IS.2 C1.1 can not V1.1 with HAS.1 a C2.1",
        "most": "with HAS.1 and without HAS.3 most IS.2 C1.1 has HAS.2",
    },
    ("relative_clause", "before"): {
        "event": "that CAN.3 the C1.1 V1.1 that the C1.2 V2.1 the C2.1",
        "rule": "that is not IS.3 and not IS.5 all THING with HAS.1 can CAN.2",
    },
    ("adposition", "postposition"): {
        "capacity": "the IS.2 C1.1 can not V1.1 a C2.1 HAS.1 with",
        "most": "most IS.2 C1.1 HAS.1 with and HAS.3 without has HAS.2",
    },
    ("auxiliary", "after"): {
        "capacity": "the IS.2 C1.1 V1.1 can not a C2.1 with HAS.1",
        "lacks": "the C1.1 no HAS.4 has",
        "pronoun": "it a C2 is not",
        "big": "the C1 SC.1.HIGH is",
        "rule": "all THING with HAS.1 that IS.3 is not and IS.5 not CAN.2 can",
    },
    ("negation", "before_auxiliary"): {
        "capacity": "the IS.2 C1.1 not can V1.1 a C2.1 with HAS.1",
        "pole": "C1.1 not is SC.1.HIGH",
        "rule": "all THING with HAS.1 that not is IS.3 and not IS.5 can CAN.2",
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
    assert say(tiny, "capacity", **grammar) == "V1.1 not can C1.1 IS.2 the HAS.1 with C2.1 a"
    assert say(tiny, "rule", **grammar) == (
        "CAN.2 can that IS.3 not is and IS.5 not HAS.1 with THING all"
    )
    assert say(tiny, "event", **grammar) == "V1.1 that CAN.3 C1.1 the that V2.1 C1.2 the C2.1 the"
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
    assert say(tiny, "member") == "all C1.1 is a C1"
    assert "-" not in " ".join(say(tiny, name) for name in PLANS)
    plain = SentencePlan(kind_of("C1.1", quantifier="all"), Predication(CAN, "CAN.1"))
    realizer = tiny.realizer(grammar=NEVER_CAN)
    assert glosses(realizer.realize(plain, np.random.default_rng(0))) == "all C1.1 CAN.1"


def test_number_and_agreement(tiny) -> None:
    grammar = {"morphology": NUMBER}
    # a class-level noun is plural, and the auxiliaries agree: is and has become are and have
    assert say(tiny, "most", **grammar) == (
        "most IS.2 C1.1-PLURAL with HAS.1 and without HAS.3 have HAS.2"
    )
    assert say(tiny, "no", **grammar) == "no C1.1-PLURAL have HAS.2"
    assert say(tiny, "pole", **grammar) == "C1.1-PLURAL are not SC.1.HIGH"
    # a plural predicate noun takes no "a"
    assert say(tiny, "member", **grammar) == "all C1.1-PLURAL are C1-PLURAL"
    # can does not agree, and the verb after it is not marked
    assert say(tiny, "generic", **grammar) == "C1.1-PLURAL can CAN.1"
    assert say(tiny, "negative", **grammar) == "C1-PLURAL can not V1.1 IS.4 C2-PLURAL"
    # an instance is singular
    assert say(tiny, "capacity", **grammar) == ENGLISH["capacity"]
    assert say(tiny, "lacks", **grammar) == ENGLISH["lacks"]
    assert say(tiny, "pronoun", **grammar) == ENGLISH["pronoun"]
    assert say(tiny, "event", **grammar) == ENGLISH["event"]
    # agreement holds across a relative clause: "are" in the clause, and "can" after it
    assert say(tiny, "rule", **grammar) == (
        "all THING-PLURAL with HAS.1 that are not IS.3 and not IS.5 can CAN.2"
    )
    for name in PLANS:
        check(tiny, realize(tiny, name, **grammar), **grammar)


def bare(case, name: str, **morphology) -> str:
    """A sentence without the optional "can", so that its verb is the finite word."""
    realizer = case.realizer(grammar={**NEVER_CAN, "morphology": morphology})
    return glosses(realizer.realize(PLANS[name], np.random.default_rng(0)))


def test_the_verb_takes_the_plural_marker_of_its_subject(tiny) -> None:
    # verb_marks: plural (the default): a verb with a plural subject is marked
    assert bare(tiny, "generic", **NUMBER) == "C1.1-PLURAL CAN.1-PLURAL"
    assert bare(tiny, "event", **NUMBER) == ENGLISH["event"]
    assert bare(tiny, "ran", **NUMBER) == "a C1.2 CAN.1"
    # verb_marks: singular, as in English: "the penguin swims", "penguins swim"
    english = {"number": {"enabled": True, "verb_marks": "singular"}}
    assert bare(tiny, "generic", **english) == "C1.1-PLURAL CAN.1"
    assert bare(tiny, "ran", **english) == "a C1.2 CAN.1-PLURAL"
    assert bare(tiny, "event", **english) == (
        "the C1.1 that CAN.3-PLURAL V1.1-PLURAL the C2.1 that the C1.2 V2.1-PLURAL"
    )
    # the verb after "can" is never marked, and the auxiliaries follow the subject's number
    assert bare(tiny, "swims", **english) == "the C1.1 can CAN.3"
    assert bare(tiny, "lacks", **english) == "the C1.1 has no HAS.4"
    # with agreement off, only nouns are marked
    off = {"number": {"enabled": True, "agreement": False}}
    assert bare(tiny, "generic", **off) == "C1.1-PLURAL CAN.1"
    assert (
        bare(tiny, "most", **off) == "most IS.2 C1.1-PLURAL with HAS.1 and without HAS.3 has HAS.2"
    )
    assert bare(tiny, "member", **off) == "all C1.1-PLURAL is C1-PLURAL"


def test_agreement_across_a_relative_clause(tiny) -> None:
    # The subject's noun and its verb are apart: the clause between them has its own verbs.
    plan = SentencePlan(
        kind_of("C1.1", Literal("IS.3", False), Literal("IS.5", False)),
        Predication(VERB, "V1.1", True, kind_of("C2")),
    )
    realizer = tiny.realizer(grammar={**NEVER_CAN, "morphology": NUMBER})
    sentence = realizer.realize(plan, np.random.default_rng(0))
    assert glosses(sentence) == ("C1.1-PLURAL that are not IS.3 and not IS.5 V1.1-PLURAL C2-PLURAL")
    # in an object relative, the verb agrees with the clause's own subject
    nested = SentencePlan(
        the(
            "I2.1.1",
            "C2.1",
            clause=RelativeClause(
                (Predication(VERB, "V2.1", event="SN.1.2", tense="past", aspect="simple"),),
                the("I1.2.1", "C1.2"),
            ),
        ),
        Predication(CAN, "CAN.1", event="SN.1.4", tense="past", aspect="simple"),
    )
    english = {"number": {"enabled": True, "verb_marks": "singular"}}
    realizer = tiny.realizer(grammar={"morphology": english})
    assert glosses(realizer.realize(nested, np.random.default_rng(0))) == (
        "the C2.1 that the C1.2 V2.1-PLURAL CAN.1-PLURAL"
    )


def test_number_as_a_separate_word(tiny) -> None:
    after = {"number": {"enabled": True, "realization": "word"}}
    assert bare(tiny, "generic", **after) == "C1.1 PLURAL CAN.1 PLURAL"
    assert bare(tiny, "member", **after) == "all C1.1 PLURAL are C1 PLURAL"
    before = {"number": {"enabled": True, "realization": "word", "position": "before"}}
    assert bare(tiny, "generic", **before) == "PLURAL C1.1 PLURAL CAN.1"
    realizer = tiny.realizer(grammar={**NEVER_CAN, "morphology": after})
    sentence = realizer.realize(PLANS["generic"], np.random.default_rng(0))
    lexicon = tiny.lexicon(grammar={"morphology": after})
    plural = lexicon.function_word("PLURAL").label
    noun, verb = lexicon.lexemes_of("C1.1")[0].label, lexicon.lexemes_of("CAN.1")[0].label
    # the marked word is a node with two children, and the marker is a token of its own
    assert sentence.tree == [
        "S",
        ["NP-SBJ", ["N", ["N", noun], ["PLURAL", plural]]],
        ["VP", ["V", ["V", verb], ["PLURAL", plural]]],
    ]
    assert sentence.tokens == (noun, plural, verb, plural)
    assert sentence.conceptual == "C1.1 PLURAL CAN.1 PLURAL"
    assert interpret(sentence.tree, lexicon, sentence.referents) == PLANS["generic"]
    # as an affix, the marker joins the token
    affixed = tiny.realizer(grammar={**NEVER_CAN, "morphology": NUMBER})
    sentence = affixed.realize(PLANS["generic"], np.random.default_rng(0))
    assert sentence.tokens == (f"{noun}-PLURAL", f"{verb}-PLURAL")
    assert sentence.formal == f"C1.1/{noun}-PLURAL CAN.1/{verb}-PLURAL"
    assert sentence.conceptual == "C1.1-PLURAL CAN.1-PLURAL"


def test_tense_marks_events_only(tiny) -> None:
    past = {"tense": {"enabled": True}}
    assert bare(tiny, "event", **past) == (
        "the C1.1 that CAN.3-PAST V1.1-PAST the C2.1 that the C1.2 V2.1-PAST"
    )
    assert bare(tiny, "ran", **past) == "a C1.2 CAN.1-PAST"
    # class-level and instance-level propositions are present, and the present is not marked
    for name in ("generic", "capacity", "swims", "rule"):
        assert "PAST" not in bare(tiny, name, **past)
    # the tense is part of the plan, and the present is not marked
    ran = PLANS["ran"]
    present = SentencePlan(ran.subject, dataclasses.replace(ran.predication, tense="present"))
    realizer = tiny.realizer(grammar={"morphology": past})
    sentence = realizer.realize(present, np.random.default_rng(0))
    assert glosses(sentence) == "a C1.2 CAN.1"
    lexicon = tiny.lexicon(grammar={"morphology": past})
    assert interpret(sentence.tree, lexicon, sentence.referents, sentence.events) == present
    # with tense marking off, a past event is not marked either, and still reads back as past
    plain = tiny.realizer().realize(ran, np.random.default_rng(0))
    assert glosses(plain) == "a C1.2 CAN.1"
    assert plain.events == (report("SN.1.4"),)
    assert interpret(plain.tree, tiny.lexicon(), plain.referents, plain.events) == ran
    word = {"tense": {"enabled": True, "realization": "word", "position": "before"}}
    assert bare(tiny, "ran", **word) == "a C1.2 PAST CAN.1"
    # a verb with a tense marker takes no agreement marker, as in English "chased"
    both = {"tense": {"enabled": True}, "number": {"enabled": True, "verb_marks": "singular"}}
    assert bare(tiny, "ran", **both) == "a C1.2 CAN.1-PAST"
    assert bare(tiny, "generic", **both) == "C1.1-PLURAL CAN.1"


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
    return SentencePlan(phrase(plan.subject), predication(plan.predication))


def test_aspect_is_part_of_the_plan(tiny) -> None:
    def said(plan, **morphology) -> str:
        realizer = tiny.realizer(grammar={**NEVER_CAN, "morphology": morphology})
        return glosses(realizer.realize(plan, np.random.default_rng(0)))

    aspect = {"aspect": {"enabled": True}}
    assert said(progressive("ran"), **aspect) == "a C1.2 CAN.1 PROGRESSIVE"  # a word, by default
    assert said(progressive("event"), **aspect) == (
        "the C1.1 that CAN.3 PROGRESSIVE V1.1 PROGRESSIVE the C2.1 that the C1.2 V2.1 PROGRESSIVE"
    )
    # a simple event is not marked, and neither is a capacity
    assert said(PLANS["ran"], **aspect) == "a C1.2 CAN.1"
    assert "PROGRESSIVE" not in said(PLANS["capacity"], **aspect)
    affix = {"aspect": {"enabled": True, "realization": "affix"}}
    assert said(progressive("ran"), **affix) == "a C1.2 CAN.1-PROGRESSIVE"
    # past and progressive together: an affix and a word
    both = {"tense": {"enabled": True}, **aspect}
    assert said(progressive("ran"), **both) == "a C1.2 CAN.1-PAST PROGRESSIVE"
    # the grammar never draws the aspect: the same plan gives the same marker every time
    realizer = tiny.realizer(grammar={"morphology": aspect})
    rng = np.random.default_rng(5)
    assert {glosses(realizer.realize(PLANS["ran"], rng)) for _ in range(50)} == {"a C1.2 CAN.1"}
    # with aspect marking off, a progressive event has no marker, and still reads back
    plain = tiny.realizer().realize(progressive("ran"), np.random.default_rng(0))
    assert glosses(plain) == "a C1.2 CAN.1"
    assert plain.events == (report("SN.1.4", "progressive"),)
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
    assert say(tiny, "generic", can_rate={"class": 1.0}) == "C1.1 can CAN.1"
    assert say(tiny, "generic", can_rate={"class": 0.0}) == "C1.1 CAN.1"
    realizer = tiny.realizer()
    rng = np.random.default_rng(2)
    drawn = [glosses(realizer.realize(PLANS["generic"], rng)) for _ in range(800)]
    assert set(drawn) == {"C1.1 can CAN.1", "C1.1 CAN.1"}
    assert abs(np.mean([words == "C1.1 can CAN.1" for words in drawn]) - 0.5) < 0.05
    # both say the same: the tree reads back as the same plan
    for rate in (0.0, 1.0):
        sentence = realize(tiny, "generic", can_rate={"class": rate})
        assert interpret(sentence.tree, tiny.lexicon(), sentence.referents) == PLANS["generic"]
    # a negative capacity needs "can" to carry "not", at both levels and at any rate
    never = tiny.realizer(grammar=BARE)
    rng = np.random.default_rng(0)
    assert glosses(never.realize(PLANS["negative"], rng)) == ENGLISH["negative"]
    assert glosses(never.realize(PLANS["capacity"], rng)) == ENGLISH["capacity"]
    # "no" replaces the negation, so the rate applies
    none = SentencePlan(kind_of("C1.1", quantifier="no"), Predication(CAN, "CAN.1"))
    assert glosses(never.realize(none, rng)) == "no C1.1 CAN.1"
    # an instance's capacity says "can" by default, and drops it at the instance rate
    assert glosses(tiny.realizer(grammar=NEVER_CAN).realize(PLANS["swims"], rng)) == (
        "the C1.1 can CAN.3"
    )
    assert glosses(never.realize(PLANS["swims"], rng)) == "the C1.1 CAN.3"
    half = tiny.realizer(grammar={"can_rate": {"instance": 0.5}})
    drawn = [glosses(half.realize(PLANS["swims"], rng)) for _ in range(800)]
    assert set(drawn) == {"the C1.1 can CAN.3", "the C1.1 CAN.3"}
    assert abs(np.mean([words == "the C1.1 can CAN.3" for words in drawn]) - 0.5) < 0.05
    # the rate applies inside a relative clause too
    clause = SentencePlan(
        the("I1.1.1", "C1.1", clause=RelativeClause((Predication(CAN, "CAN.1"),))),
        Predication(IS, "IS.2"),
    )
    sentence = never.realize(clause, rng)
    assert glosses(sentence) == "the C1.1 that CAN.1 is IS.2"
    assert interpret(sentence.tree, tiny.lexicon(), sentence.referents, sentence.events) == clause


def test_a_capacity_and_an_event_can_have_the_same_words(tiny) -> None:
    """With "can" dropped and the tense and the aspect unmarked, "the penguin swim" is a
    capacity or an event. The two sentences have the same tokens and the same tree, so the tree
    carries no information about the reading, and both get the same readings. Only the
    sentence's record tells them apart."""
    subject = the("I1.1.1", "C1.1")
    target = the("I2.1.1", "C2.1")
    pairs = [
        (Predication(CAN, "CAN.3"), Predication(CAN, "CAN.3", True, None, *report("SN.1.1"))),
        (
            Predication(VERB, "V1.1", True, target),
            Predication(VERB, "V1.1", True, target, *report("SN.1.2", "progressive")),
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
    for grammar, expected in (
        ({}, (("capacity",), ("event",))),
        ({**BARE, "morphology": {"tense": {"enabled": True}}}, (("capacity",), ("event",))),
        (
            {"can_rate": {"instance": 1.0}, "morphology": {"aspect": {"enabled": True}}},
            (("capacity",), ("event",)),
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


def test_readings_of_the_fixed_set(tiny) -> None:
    config, lexicon = tiny.config(), tiny.lexicon()
    expected = {"class": ("generic",), "instance": ("capacity",), "event": ("event",)}
    for name, plan in PLANS.items():
        sentence = realize(tiny, name)
        assert readings(sentence.tree, lexicon, config) == expected[plan.level], name
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
    assert found["capacity"] == found["lacks"] == found["pronoun"] == ("capacity",)
    assert found["generic"] == found["rule"] == ("generic",)
    # a language whose events are always marked has no ambiguous sentence
    marked = {**BARE, "morphology": {"aspect": {"enabled": True}}}
    always = {"events": {"progressive_rate": 1.0}}
    config = tiny.config(grammar=marked, propositions=always)
    lexicon = tiny.lexicon(grammar=marked)
    sentence = tiny.realizer(grammar=marked).realize(PLANS["swims"], np.random.default_rng(0))
    assert readings(sentence.tree, lexicon, config) == ("capacity",)


def test_adjective_order(cases) -> None:
    case = cases("default")
    adjectives = ("IS.3", "IS.17", "SC.1.HIGH", "IS.30", "SC.2.LOW")
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
            kind_of("C1.1", *(Literal(a) for a in chosen)), Predication(HAS, "HAS.1")
        )
        words = glosses(realizer.realize(plan, rng)).split()
        said = words[:3]
        assert said == sorted(chosen, key=rank.get)
    # fixed: false makes the order random for each phrase
    free = case.realizer(grammar={"adjective_order": {"fixed": False}})
    plan = SentencePlan(
        kind_of("C1.1", *(Literal(a) for a in adjectives[:3])), Predication(HAS, "HAS.1")
    )
    orders = {tuple(glosses(free.realize(plan, rng)).split()[:3]) for _ in range(200)}
    assert len(orders) == 6
    lexicon = case.lexicon()
    for _ in range(20):
        sentence = free.realize(plan, rng)
        assert interpret(sentence.tree, lexicon, sentence.referents) == plan


def test_a_mention_picks_one_of_a_concepts_lexemes(tiny) -> None:
    knobs = {"lexicon": {"synonym_rate": 1.0}}
    realizer = tiny.realizer(**knobs)
    lexicon = tiny.lexicon(**knobs)
    rng = np.random.default_rng(1)
    sentences = [realizer.realize(PLANS["capacity"], rng) for _ in range(200)]
    # the formal rendering tells the synonyms apart, and the conceptual rendering does not
    assert len({s.formal for s in sentences}) > 20
    assert {s.conceptual for s in sentences} == {"THE IS.2 C1.1 CAN NOT V1.1 A C2.1 WITH HAS.1"}
    first, second = lexicon.lexemes_of("C1.1")
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
            the("I1.1.1", "C1.1", *restriction, clause=RelativeClause(predications)),
            Predication(IS, "IS.2"),
        )
        sentence = realize_plan(tiny, plan)
        return glosses(sentence).removeprefix("the C1.1 ").removesuffix(" is IS.2")

    target = the("I2.1.1", "C2.1")
    can_swim = Predication(CAN, "CAN.1")
    can_not_fly = Predication(CAN, "CAN.2", False)
    chases = Predication(VERB, "V1.1", True, target)
    edible = Predication(PROJECTION, "CANBE.V2.2")
    a_bird = Predication(MEMBER, "C1", False)
    not_red = (Literal("IS.3", False),)
    assert clause_of(can_swim) == "that can CAN.1"
    # the same auxiliary is not said again
    assert clause_of(can_swim, chases) == "that can CAN.1 and V1.1 the C2.1"
    assert clause_of(can_swim, can_not_fly) == "that can CAN.1 and not CAN.2"
    assert clause_of(edible, a_bird) == "that is CANBE.V2.2 and not a C1"
    # another auxiliary is said
    assert clause_of(can_swim, edible) == "that can CAN.1 and is CANBE.V2.2"
    assert (
        clause_of(edible, can_swim, chases) == "that is CANBE.V2.2 and can CAN.1 and V1.1 the C2.1"
    )
    # the negated IS literals come first, and the other verb phrases join them
    assert clause_of(can_swim, restriction=not_red) == "that is not IS.3 and can CAN.1"
    assert clause_of(edible, restriction=not_red) == "that is not IS.3 and CANBE.V2.2"


def realize_plan(case, plan: SentencePlan, **grammar):
    realizer = case.realizer(**settings(**grammar))
    sentence = realizer.realize(plan, np.random.default_rng(0))
    check(case, sentence, **grammar)
    return sentence


def test_negated_literals_alone_make_a_relative_clause(tiny) -> None:
    many = (Literal("IS.1", False), Literal("IS.3", False), Literal("IS.5", False))
    plan = SentencePlan(the("I1.1.1", "C1.1", *many), Predication(IS, "IS.2"))
    assert glosses(realize_plan(tiny, plan)) == (
        "the C1.1 that is not IS.1 and not IS.3 and not IS.5 is IS.2"
    )
    assert plan.depth() == 1 and tree_depth(realize_plan(tiny, plan).tree) == 1


def test_relative_clauses_nest(tiny) -> None:
    # "the C1.1 that chased the C2.1 that the C1.2 that ran saw ran": depth 3
    inner = the(
        "I1.2.1",
        "C1.2",
        clause=RelativeClause(
            (Predication(CAN, "CAN.1", event="SN.1.1", tense="past", aspect="simple"),)
        ),
    )
    middle = the(
        "I2.1.1",
        "C2.1",
        clause=RelativeClause(
            (Predication(VERB, "V2.1", event="SN.1.2", tense="past", aspect="simple"),), inner
        ),
    )
    outer = the(
        "I1.1.1",
        "C1.1",
        clause=RelativeClause(
            (Predication(VERB, "V1.1", True, middle, "SN.1.3", "past", "simple"),)
        ),
    )
    plan = SentencePlan(
        outer, Predication(CAN, "CAN.3", event="SN.1.4", tense="past", aspect="simple")
    )
    sentence = realize_plan(tiny, plan)
    assert glosses(sentence) == ("the C1.1 that V1.1 the C2.1 that the C1.2 that CAN.1 V2.1 CAN.3")
    assert plan.depth() == 3 and tree_depth(sentence.tree) == 3
    # a node before its children: the innermost clause comes before the verb that follows it
    assert sentence.event_labels == ("SN.1.3", "SN.1.1", "SN.1.2", "SN.1.4")
    assert [r for r, _ in sentence.referents] == ["I1.1.1", "I2.1.1", "I1.2.1"]
    for name, plan in PLANS.items():
        expected = {"rule": 1, "event": 1}.get(name, 0)
        assert plan.depth() == expected == tree_depth(realize(tiny, name).tree)
    for order in CLAUSE_ORDERS:
        realize_plan(
            tiny,
            SentencePlan(
                outer, Predication(CAN, "CAN.3", event="SN.1.4", tense="past", aspect="simple")
            ),
            word_order={"clause": order},
        )


def test_class_level_relative_clauses(tiny) -> None:
    swim = RelativeClause((Predication(CAN, "CAN.1"),))
    chase = RelativeClause((Predication(VERB, "V1.1", True, kind_of("C2")),))
    chased = RelativeClause((Predication(VERB, "V1.1"),), kind_of("C2"))
    swimmers = kind_of("C2", clause=swim)
    plans = {
        # "penguins that can swim", "owls that eat mice", and "mice that owls eat"
        "can": SentencePlan(kind_of("C1.1", clause=swim), Predication(HAS, "HAS.2")),
        "subject": SentencePlan(
            kind_of("C1.1", clause=chase, quantifier="most"), Predication(IS, "IS.2")
        ),
        "object": SentencePlan(
            kind_of("C1.1", clause=chased), Predication(SCALAR, "SC.1.HIGH", False)
        ),
        # a clause inside a clause, and a clause on the patient
        "nested": SentencePlan(
            kind_of(
                "C1.1",
                clause=RelativeClause((Predication(VERB, "V1.1", True, swimmers),)),
                quantifier="some",
            ),
            Predication(HAS, "HAS.2"),
        ),
        "patient": SentencePlan(kind_of("C1"), Predication(VERB, "V1.1", True, swimmers)),
        # the negated IS literals join the clause
        "joined": SentencePlan(
            kind_of("C1.1", Literal("IS.3", False), clause=swim), Predication(HAS, "HAS.2")
        ),
    }
    english = {
        "can": "C1.1 that can CAN.1 has HAS.2",
        "subject": "most C1.1 that can V1.1 C2 is IS.2",
        "object": "C1.1 that C2 can V1.1 is not SC.1.HIGH",
        "nested": "some C1.1 that can V1.1 C2 that can CAN.1 has HAS.2",
        "patient": "C1 can V1.1 C2 that can CAN.1",
        "joined": "C1.1 that is not IS.3 and can CAN.1 has HAS.2",
    }
    bare = {
        "can": "C1.1 that CAN.1 has HAS.2",
        "subject": "most C1.1 that V1.1 C2 is IS.2",
        "object": "C1.1 that C2 V1.1 is not SC.1.HIGH",
        "nested": "some C1.1 that V1.1 C2 that CAN.1 has HAS.2",
        "patient": "C1 V1.1 C2 that CAN.1",
        "joined": "C1.1 that is not IS.3 and CAN.1 has HAS.2",
    }
    plural = {
        # the verbs of a subject relative agree with the head, and those of an object relative
        # with the clause's own subject
        "can": "C1.1-PLURAL that CAN.1-PLURAL have HAS.2",
        "subject": "most C1.1-PLURAL that V1.1-PLURAL C2-PLURAL are IS.2",
        "object": "C1.1-PLURAL that C2-PLURAL V1.1-PLURAL are not SC.1.HIGH",
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
    assert term_of(plans["can"].subject) == CategoryTerm("C1.1", (), (Clause(CAN, "CAN.1"),))
    assert term_of(plans["subject"].subject).clauses == (
        Clause(VERB, "V1.1", patient=CategoryTerm("C2")),
    )
    assert term_of(plans["object"].subject).clauses == (
        Clause(VERB, "V1.1", agent=CategoryTerm("C2")),
    )
    assert term_of(plans["nested"].subject).clauses == (
        Clause(VERB, "V1.1", patient=CategoryTerm("C2", (), (Clause(CAN, "CAN.1"),))),
    )
    assert plans["nested"].depth() == 2 and plans["joined"].depth() == 1
    assert plans["patient"].proposition().predicate.patient.clauses == (Clause(CAN, "CAN.1"),)
    # a term with an object relative has that one clause
    both = CategoryTerm(
        "C1.1", (), (Clause(CAN, "CAN.1"), Clause(VERB, "V1.1", agent=CategoryTerm("C2")))
    )
    with pytest.raises(GrammarError, match="the only relative clause"):
        phrase_of(both)
    # a term with several subject relatives joins them: "that can swim and chase fish"
    joined = CategoryTerm(
        "C1.1", (), (Clause(CAN, "CAN.1"), Clause(VERB, "V1.1", patient=CategoryTerm("C2")))
    )
    plan = SentencePlan(phrase_of(joined), Predication(HAS, "HAS.2"))
    assert glosses(realize_plan(tiny, plan)) == "C1.1 that can CAN.1 and V1.1 C2 has HAS.2"
    assert plan.proposition().subject == joined


# ---------------------------------------------------------------------------------------------
# Plans the grammar does not realize, and trees it does not read
# ---------------------------------------------------------------------------------------------


def test_plans_the_grammar_does_not_realize(tiny) -> None:
    subject = the("I1.1.1", "C1.1")
    target = the("I2.1.1", "C2.1")
    swims = Predication(CAN, "CAN.1")
    bad = {
        "takes no determiner": SentencePlan(NounPhrase(INSTANCE_NP, "I1.1.1", None, "the"), swims),
        "with a or the": SentencePlan(NounPhrase(INSTANCE_NP, "I1.1.1", "C1.1", "most"), swims),
        "names its own category": SentencePlan(NounPhrase(CLASS_NP, "C1.1", "C1"), swims),
        "only the subject takes a quantifier": SentencePlan(
            kind_of("C1"), Predication(VERB, "V1.1", True, kind_of("C2", quantifier="all"))
        ),
        "about categories": SentencePlan(kind_of("C1"), Predication(VERB, "V1.1", True, target)),
        "a verb, and only a verb": SentencePlan(subject, Predication(VERB, "V1.1")),
        "a verb, and only a verb ": SentencePlan(subject, Predication(CAN, "CAN.1", True, target)),
        "never negated": SentencePlan(
            subject, Predication(CAN, "CAN.1", False, event="SN.1.1", tense="past", aspect="simple")
        ),
        "an event is a CAN feature": SentencePlan(
            subject, Predication(IS, "IS.1", event="SN.1.1", tense="past", aspect="simple")
        ),
        "every verb phrase of an event-level sentence": SentencePlan(
            the("I1.1.1", "C1.1", clause=RelativeClause((swims,))),
            Predication(CAN, "CAN.3", event="SN.1.1", tense="past", aspect="simple"),
        ),
        "about an instance": SentencePlan(
            kind_of("C1"), Predication(CAN, "CAN.1", event="SN.1.1", tense="past", aspect="simple")
        ),
        "no comparison class": SentencePlan(IT, Predication(SCALAR, "SC.1.HIGH")),
        "already named": SentencePlan(subject, Predication(MEMBER, "C1.1")),
        "says something about its head": SentencePlan(
            the("I1.1.1", "C1.1", clause=RelativeClause(())), swims
        ),
        "belong in the restriction": SentencePlan(
            the("I1.1.1", "C1.1", clause=RelativeClause((Predication(IS, "IS.3", False),))), swims
        ),
        "belong in the restriction ": SentencePlan(
            the("I1.1.1", "C1.1", clause=RelativeClause((Predication(HAS, "HAS.3"),))), swims
        ),
        "one verb, whose patient is the head": SentencePlan(
            the("I1.1.1", "C1.1", clause=RelativeClause((swims,), target)), swims
        ),
        "one relative clause": SentencePlan(
            the(
                "I1.1.1",
                "C1.1",
                Literal("IS.3", False),
                clause=RelativeClause((Predication(VERB, "V1.1"),), target),
            ),
            swims,
        ),
        "never negated ": SentencePlan(the("I1.1.1", "C1.1", Literal("SC.1.HIGH", False)), swims),
        "an event has a tense": SentencePlan(subject, Predication(CAN, "CAN.1", event="SN.1.9")),
        "only an event has a tense": SentencePlan(subject, Predication(CAN, "CAN.1", tense="past")),
        "a class-level relative clause holds": SentencePlan(
            kind_of("C1.1", clause=RelativeClause((Predication(PROJECTION, "CANBE.V2.2"),))),
            Predication(HAS, "HAS.2"),
        ),
        "a class-level relative clause holds ": SentencePlan(
            kind_of("C1.1", clause=RelativeClause((Predication(CAN, "CAN.1", False),))),
            Predication(HAS, "HAS.2"),
        ),
    }
    realizer = tiny.realizer()
    for reason, plan in bad.items():
        with pytest.raises(GrammarError, match=reason.strip()):
            realizer.realize(plan, np.random.default_rng(0))


def test_a_concept_without_a_word_cannot_be_said(tiny) -> None:
    realizer = tiny.realizer(lexicon={"named_proportion": {"can": 0.0}})
    with pytest.raises(GrammarError, match="CAN.1 has no word"):
        realizer.realize(PLANS["generic"], np.random.default_rng(0))
    # V1 holds for every pair of instances, so it never has a word
    general = SentencePlan(kind_of("C1"), Predication(VERB, "V1", True, kind_of("C2")))
    with pytest.raises(GrammarError, match="V1 has no word"):
        tiny.realizer().realize(general, np.random.default_rng(0))


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
        interpret(able.tree, lexicon, able.referents, (report("SN.1.1"),))
    with pytest.raises(GrammarError, match="an event is reported by a verb with no auxiliary"):
        generic = realize(tiny, "generic")
        interpret(generic.tree, lexicon, generic.referents, (report("SN.1.1"),))
    with pytest.raises(GrammarError, match="a sentence is an S"):
        interpret(sentence.tree[1], lexicon, sentence.referents[:1])
    generic = realize(tiny, "generic")
    with pytest.raises(GrammarError, match="the referent given is C2"):
        interpret(generic.tree, lexicon, ["C2"])
