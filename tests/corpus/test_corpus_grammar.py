"""Stage 4 acceptance tests: grammar and realization, on a fixed set of sentence plans.

All six clause orders and all two-way settings produce the expected words and trees. Every
tree's leaves equal its tokens, and ``interpret(tree)`` recovers the plan exactly. The expected
strings are written in glosses: a content word is its concept label, and an affix follows a
hyphen.
"""

from __future__ import annotations

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
    Literal,
)
from semantic_world.corpus.realize import leaves, tree_depth

CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
ALWAYS_CAN = {"class_can_rate": 1.0}
NEVER_CAN = {"class_can_rate": 0.0}


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
            clause=RelativeClause((Predication(CAN, "CAN.3", event="SN.1.1"),)),
        ),
        Predication(
            VERB,
            "V1.1",
            True,
            the(
                "I2.1.1",
                "C2.1",
                clause=RelativeClause(
                    (Predication(VERB, "V2.1", event="SN.1.2"),), the("I1.2.1", "C1.2")
                ),
            ),
            "SN.1.3",
        ),
    ),
    "ran": SentencePlan(
        the("I1.2.1", "C1.2", determiner="a"), Predication(CAN, "CAN.1", event="SN.1.4")
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
    assert event.events == ("SN.1.1", "SN.1.3", "SN.1.2")
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
                (Predication(VERB, "V2.1", event="SN.1.2"),), the("I1.2.1", "C1.2")
            ),
        ),
        Predication(CAN, "CAN.1", event="SN.1.4"),
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
    present = {"tense": {"enabled": True, "event_tense": "present"}}
    assert bare(tiny, "event", **present) == ENGLISH["event"]
    word = {"tense": {"enabled": True, "realization": "word", "position": "before"}}
    assert bare(tiny, "ran", **word) == "a C1.2 PAST CAN.1"
    # a verb with a tense marker takes no agreement marker, as in English "chased"
    both = {"tense": {"enabled": True}, "number": {"enabled": True, "verb_marks": "singular"}}
    assert bare(tiny, "ran", **both) == "a C1.2 CAN.1-PAST"
    assert bare(tiny, "generic", **both) == "C1.1-PLURAL CAN.1"


def test_aspect_is_drawn_for_events(tiny) -> None:
    always = {"aspect": {"enabled": True, "progressive_rate": 1.0}}
    assert bare(tiny, "ran", **always) == "a C1.2 CAN.1 PROGRESSIVE"  # a word, by default
    assert bare(tiny, "event", **always) == (
        "the C1.1 that CAN.3 PROGRESSIVE V1.1 PROGRESSIVE the C2.1 that the C1.2 V2.1 PROGRESSIVE"
    )
    assert "PROGRESSIVE" not in bare(tiny, "capacity", **always)
    never = {"aspect": {"enabled": True, "progressive_rate": 0.0}}
    assert bare(tiny, "ran", **never) == "a C1.2 CAN.1"
    affix = {"aspect": {"enabled": True, "progressive_rate": 1.0, "realization": "affix"}}
    assert bare(tiny, "ran", **affix) == "a C1.2 CAN.1-PROGRESSIVE"
    # past and progressive together: an affix and a word
    both = {"tense": {"enabled": True}, **always}
    assert bare(tiny, "ran", **both) == "a C1.2 CAN.1-PAST PROGRESSIVE"
    realizer = tiny.realizer(grammar={"morphology": {"aspect": {"enabled": True}}})
    rng = np.random.default_rng(5)
    drawn = [glosses(realizer.realize(PLANS["ran"], rng)) for _ in range(600)]
    share = np.mean(["PROGRESSIVE" in words for words in drawn])
    assert abs(share - 0.3) < 0.05  # the default rate
    for name in PLANS:
        grammar = {"morphology": {**both, **NUMBER}}
        check(tiny, realize(tiny, name, **grammar), **grammar)


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
                "tense": {"enabled": True, "event_tense": "present"},
                "aspect": {"enabled": True, "realization": "affix"},
            }
        }
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


def test_the_optional_can_of_a_class_level_capacity(tiny) -> None:
    assert say(tiny, "generic", class_can_rate=1.0) == "C1.1 can CAN.1"
    assert say(tiny, "generic", class_can_rate=0.0) == "C1.1 CAN.1"
    realizer = tiny.realizer()
    rng = np.random.default_rng(2)
    drawn = [glosses(realizer.realize(PLANS["generic"], rng)) for _ in range(800)]
    assert set(drawn) == {"C1.1 can CAN.1", "C1.1 CAN.1"}
    assert abs(np.mean([words == "C1.1 can CAN.1" for words in drawn]) - 0.5) < 0.05
    # both say the same: the tree reads back as the same plan
    for sentence in (realize(tiny, "generic", class_can_rate=rate) for rate in (0.0, 1.0)):
        assert interpret(sentence.tree, tiny.lexicon(), sentence.referents) == PLANS["generic"]
    # a negative capacity needs "can" to carry "not", and an instance's capacity always has it,
    # because a bare verb after an instance reports an event
    never = tiny.realizer(grammar=NEVER_CAN)
    rng = np.random.default_rng(0)
    assert glosses(never.realize(PLANS["negative"], rng)) == ENGLISH["negative"]
    assert glosses(never.realize(PLANS["swims"], rng)) == "the C1.1 can CAN.3"
    assert glosses(never.realize(PLANS["capacity"], rng)) == ENGLISH["capacity"]
    # "no" replaces the negation, so the rate applies
    none = SentencePlan(kind_of("C1.1", quantifier="no"), Predication(CAN, "CAN.1"))
    assert glosses(never.realize(none, rng)) == "no C1.1 CAN.1"


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
    morphology = {"aspect": {"enabled": True, "progressive_rate": 0.5}}
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
        "I1.2.1", "C1.2", clause=RelativeClause((Predication(CAN, "CAN.1", event="SN.1.1"),))
    )
    middle = the(
        "I2.1.1",
        "C2.1",
        clause=RelativeClause((Predication(VERB, "V2.1", event="SN.1.2"),), inner),
    )
    outer = the(
        "I1.1.1",
        "C1.1",
        clause=RelativeClause((Predication(VERB, "V1.1", True, middle, "SN.1.3"),)),
    )
    plan = SentencePlan(outer, Predication(CAN, "CAN.3", event="SN.1.4"))
    sentence = realize_plan(tiny, plan)
    assert glosses(sentence) == ("the C1.1 that V1.1 the C2.1 that the C1.2 that CAN.1 V2.1 CAN.3")
    assert plan.depth() == 3 and tree_depth(sentence.tree) == 3
    # a node before its children: the innermost clause comes before the verb that follows it
    assert sentence.events == ("SN.1.3", "SN.1.1", "SN.1.2", "SN.1.4")
    assert [r for r, _ in sentence.referents] == ["I1.1.1", "I2.1.1", "I1.2.1"]
    for name, plan in PLANS.items():
        expected = {"rule": 1, "event": 1}.get(name, 0)
        assert plan.depth() == expected == tree_depth(realize(tiny, name).tree)
    for order in CLAUSE_ORDERS:
        realize_plan(
            tiny,
            SentencePlan(outer, Predication(CAN, "CAN.3", event="SN.1.4")),
            word_order={"clause": order},
        )


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
        "never negated": SentencePlan(subject, Predication(CAN, "CAN.1", False, event="SN.1.1")),
        "an event is a CAN feature": SentencePlan(subject, Predication(IS, "IS.1", event="SN.1.1")),
        "every verb phrase of an event-level sentence": SentencePlan(
            the("I1.1.1", "C1.1", clause=RelativeClause((swims,))),
            Predication(CAN, "CAN.3", event="SN.1.1"),
        ),
        "about an instance": SentencePlan(kind_of("C1"), Predication(CAN, "CAN.1", event="SN.1.1")),
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
    # a verb with no auxiliary after an instance reports an event, and needs its label
    with pytest.raises(GrammarError, match="no event was given"):
        interpret(sentence.tree, lexicon, sentence.referents, (None, None, None))
    with pytest.raises(GrammarError, match="a sentence is an S"):
        interpret(sentence.tree[1], lexicon, sentence.referents[:1])
    generic = realize(tiny, "generic")
    with pytest.raises(GrammarError, match="the referent given is C2"):
        interpret(generic.tree, lexicon, ["C2"])
