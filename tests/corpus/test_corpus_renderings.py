"""The renderings: the formal rendering of a sequence of lexemes (stage 1), and the
propositional rendering of a logical form, in the notation of decision 33 (stage 5)."""

from __future__ import annotations

import dataclasses
import json

import pytest
from corpus_support import TINY_TAXONOMY, corpus_config

from semantic_world.corpus import Streams, build_lexicon, formal
from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_NP,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
)
from semantic_world.corpus.logical import logical_form
from semantic_world.corpus.propositions import (
    CAN,
    HAS,
    IS,
    MEMBER,
    PROJECTION,
    SCALAR,
    VERB,
    Literal,
    Proposition,
    event_of,
    scene_of,
)
from semantic_world.corpus.renderings import (
    Able,
    Atom,
    Exists,
    Not,
    Quantified,
    RenderingError,
    Report,
    formal_word,
    formula,
    parse_propositional,
    proposition_of,
    propositional,
    write,
)


def lexicon_of(world, **knobs):
    config = corpus_config(TINY_TAXONOMY, lexicon=knobs)
    return build_lexicon(config, world, Streams(config.seed))


def test_formal_rendering(tiny_world) -> None:
    lexicon = lexicon_of(tiny_world)
    the = lexicon.function_word("the")
    (noun,) = lexicon.lexemes_of("C1.1")
    (verb,) = lexicon.lexemes_of("CAN.2")
    # a content lexeme's gloss is its concept's label, and a function word's is its English gloss
    assert formal([the, noun, verb]) == "the/L.37 C1.1/L.2 CAN.2/L.24"
    assert formal_word(lexicon.function_word("no")) == "no/L.41"
    assert formal([]) == ""
    for lexeme in lexicon.lexemes:
        gloss, label = formal_word(lexeme).split("/")
        assert label == lexeme.label and lexicon.lexeme(label) is lexeme
        assert gloss == (lexeme.concept if lexeme.content else lexeme.concept.lower())


def test_the_formal_rendering_tells_synonyms_and_homonyms_apart(tiny_world) -> None:
    lexicon = lexicon_of(tiny_world, synonym_rate=1.0, homonym_rate=1.0)
    first, second = lexicon.lexemes_of("C1.1")
    # synonyms share the gloss, and differ in the lexeme label
    assert formal([first]) == "C1.1/L.2" and formal([second]).startswith("C1.1/L.")
    assert formal([first]) != formal([second])
    # a homonym's two lexemes differ in both
    for earlier, later in lexicon.homonym_pairs():
        a, b = formal_word(earlier).split("/"), formal_word(later).split("/")
        assert a[0] != b[0] and a[1] != b[1]


# ---------------------------------------------------------------------------------------------
# The propositional rendering
# ---------------------------------------------------------------------------------------------

REFERENTS = {"I1.3.2.1": "R.1", "I1.4.1.1": "R.2", "I1.2.1.1": "R.3"}
INSTANCES = {label: instance for instance, label in REFERENTS.items()}
DOG, CAT, OWL = "I1.3.2.1", "I1.4.1.1", "I1.2.1.1"


def the(instance: str, noun: str | None, *restriction: Literal, clause=None) -> NounPhrase:
    if noun is None:
        return NounPhrase(INSTANCE_NP, instance)
    return NounPhrase(INSTANCE_NP, instance, noun, "the", restriction, clause)


def kind_of(category: str, *restriction: Literal, quantifier=None, clause=None) -> NounPhrase:
    return NounPhrase(CLASS_NP, category, category, quantifier, restriction, clause)


def happened(kind: str, label: str, event: str, target=None, aspect="simple") -> Predication:
    return Predication(kind, label, True, target, event, "past", aspect)


def render(plan: SentencePlan, grounding=None, mode: str = "local") -> str:
    """The propositional rendering of a plan, by way of its JSON logical form."""
    proposition = plan.proposition()
    if grounding is not None:
        proposition = Proposition(
            proposition.level,
            proposition.subject,
            proposition.predicate,
            proposition.polarity,
            proposition.quantifier,
            grounding=grounding,
        )
    form = json.loads(json.dumps(logical_form(plan, proposition, REFERENTS)))
    text = propositional(form, mode)
    # the rendering reads back as the formula it writes, and as the proposition it says
    parsed = parse_propositional(text)
    assert parsed == formula(form, mode) and write(parsed) == text
    names = INSTANCES if mode == "local" else None
    assert proposition_of(parsed, names) == plan.proposition()
    assert Proposition.from_json(form) == plan.proposition()
    return text


EATS_MICE = RelativeClause((Predication(VERB, "V2.1", True, kind_of("C1.5")),))
OWLS_EAT = RelativeClause((Predication(VERB, "V2.1"),), kind_of("C1.2"))
PARENT = {"comparison": "C1"}

SENTENCES = {
    # the furry dog has legs
    "furry": (
        SentencePlan(the(DOG, "C1.3.2", Literal("IS.12")), Predication(HAS, "HAS.4")),
        "C1.3.2(R.1) AND IS.12(R.1) AND HAS.4(R.1)",
    ),
    # the dog that chased the cat ran
    "chased": (
        SentencePlan(
            the(
                DOG,
                "C1.3.2",
                clause=RelativeClause((happened(VERB, "V1.2", "SN.3.2", the(CAT, "C1.4.1")),)),
            ),
            happened(CAN, "CAN.7", "SN.3.4"),
        ),
        "C1.3.2(R.1) AND C1.4.1(R.2) AND EVENT(SN.3.2, PAST, SIMPLE, V1.2(R.1, R.2)) AND "
        "EVENT(SN.3.4, PAST, SIMPLE, CAN.7(R.1))",
    ),
    # the cat that the dog was chasing ran: an object relative, and a progressive event
    "was chased": (
        SentencePlan(
            the(
                CAT,
                "C1.4.1",
                clause=RelativeClause(
                    (happened(VERB, "V1.2", "SN.3.2", aspect="progressive"),), the(DOG, "C1.3")
                ),
            ),
            happened(CAN, "CAN.7", "SN.3.5"),
        ),
        "C1.4.1(R.2) AND C1.3(R.1) AND EVENT(SN.3.2, PAST, PROGRESSIVE, V1.2(R.1, R.2)) AND "
        "EVENT(SN.3.5, PAST, SIMPLE, CAN.7(R.2))",
    ),
    # the penguin can swim, and it can not swim
    "able": (
        SentencePlan(the(DOG, "C1.3"), Predication(CAN, "CAN.3")),
        "C1.3(R.1) AND ABLE(CAN.3(R.1))",
    ),
    "not able": (
        SentencePlan(the(DOG, None), Predication(CAN, "CAN.3", False)),
        "NOT ABLE(CAN.3(R.1))",
    ),
    # the owl can eat the mouse, with a pronoun as the object
    "can eat": (
        SentencePlan(the(OWL, "C1.2"), Predication(VERB, "V2.1", True, the(CAT, None))),
        "C1.2(R.3) AND ABLE(V2.1(R.3, R.2))",
    ),
    # the big mouse without fins is small for an animal: a pole names its comparison class
    "poles": (
        SentencePlan(
            the(CAT, "C1", Literal("SC.1.HIGH"), Literal("HAS.2", False)),
            Predication(SCALAR, "SC.2.LOW", False),
        ),
        "C1(R.2) AND NOT HAS.2(R.2) AND SC.1.HIGH(R.2, C1) AND NOT SC.2.LOW(R.2, C1)",
    ),
    # the bird that is not red is a penguin, and it is edible
    "member": (
        SentencePlan(the(DOG, "C1.3", Literal("IS.4", False)), Predication(MEMBER, "C1.3.2")),
        "C1.3(R.1) AND NOT IS.4(R.1) AND C1.3.2(R.1)",
    ),
    "edible": (
        SentencePlan(the(DOG, None), Predication(PROJECTION, "CANBE.V1.1")),
        "CANBE.V1.1(R.1)",
    ),
    # the owl that can eat the mouse that can swim is big
    "nested": (
        SentencePlan(
            the(
                OWL,
                "C1.2",
                clause=RelativeClause(
                    (
                        Predication(
                            VERB,
                            "V2.1",
                            True,
                            the(
                                CAT,
                                "C1.5",
                                clause=RelativeClause((Predication(CAN, "CAN.3"),)),
                            ),
                        ),
                    )
                ),
            ),
            Predication(SCALAR, "SC.1.HIGH"),
        ),
        "C1.2(R.3) AND C1.5(R.2) AND ABLE(CAN.3(R.2)) AND ABLE(V2.1(R.3, R.2)) AND "
        "SC.1.HIGH(R.3, C1.2)",
    ),
    # most red penguins can swim, and no penguins have fur
    "most": (
        SentencePlan(
            kind_of("C1.3", Literal("IS.4"), quantifier="most"), Predication(CAN, "CAN.3")
        ),
        "MOST(C1.3(X.1) AND IS.4(X.1), ABLE(CAN.3(X.1)))",
    ),
    "no": (
        SentencePlan(kind_of("C1.3", quantifier="no"), Predication(HAS, "HAS.7")),
        "NO(C1.3(X.1), HAS.7(X.1))",
    ),
    "most not": (
        SentencePlan(kind_of("C1.3", quantifier="most"), Predication(CAN, "CAN.2", False)),
        "MOST(C1.3(X.1), NOT ABLE(CAN.2(X.1)))",
    ),
    # all things with wings and with feathers can fly, and the bare generic
    "rule": (
        SentencePlan(
            kind_of("THING", Literal("HAS.2"), Literal("HAS.5"), quantifier="all"),
            Predication(CAN, "CAN.1"),
        ),
        "ALL(HAS.2(X.1) AND HAS.5(X.1), ABLE(CAN.1(X.1)))",
    ),
    "generic rule": (
        SentencePlan(
            kind_of("THING", Literal("HAS.2"), Literal("IS.3", False)), Predication(IS, "IS.9")
        ),
        "GEN(NOT IS.3(X.1) AND HAS.2(X.1), IS.9(X.1))",
    ),
    # owls eat mice: the quantifier ranges over pairs
    "owls eat mice": (
        SentencePlan(kind_of("C1.2"), Predication(VERB, "V2.1", True, kind_of("C1.5"))),
        "GEN(C1.2(X.1) AND C1.5(X.2), ABLE(V2.1(X.1, X.2)))",
    ),
    # penguins are birds, and penguins are not fish
    "are birds": (
        SentencePlan(kind_of("C1.3.2", quantifier="all"), Predication(MEMBER, "C1.3")),
        "ALL(C1.3.2(X.1), C1.3(X.1))",
    ),
    "are not fish": (
        SentencePlan(kind_of("C1.3.2"), Predication(MEMBER, "C2", False)),
        "GEN(C1.3.2(X.1), NOT C2(X.1))",
    ),
    # owls that eat mice are big: at least one mouse, and the comparison class of decision 22
    "owls that eat mice": (
        SentencePlan(kind_of("C1.2", clause=EATS_MICE), Predication(SCALAR, "SC.1.HIGH")),
        "GEN(C1.2(X.1) AND EXISTS(X.2, C1.5(X.2) AND ABLE(V2.1(X.1, X.2))), SC.1.HIGH(X.1, C1))",
    ),
    # mice that owls eat are small
    "mice that owls eat": (
        SentencePlan(kind_of("C1.5", clause=OWLS_EAT, quantifier="most"), Predication(IS, "IS.4")),
        "MOST(C1.5(X.1) AND EXISTS(X.2, C1.2(X.2) AND ABLE(V2.1(X.2, X.1))), IS.4(X.1))",
    ),
    # big penguins that can swim chase fish that owls eat
    "both clauses": (
        SentencePlan(
            kind_of(
                "C1.3",
                Literal("SC.1.HIGH"),
                clause=RelativeClause((Predication(CAN, "CAN.3"),)),
            ),
            Predication(VERB, "V1.2", True, kind_of("C2.1", clause=OWLS_EAT)),
        ),
        "GEN(C1.3(X.1) AND SC.1.HIGH(X.1, C1.3) AND ABLE(CAN.3(X.1)) AND C2.1(X.2) AND "
        "EXISTS(X.3, C1.2(X.3) AND ABLE(V2.1(X.3, X.2))), ABLE(V1.2(X.1, X.2)))",
    ),
    # owls that eat mice that can swim have wings: a clause inside a clause
    "deep": (
        SentencePlan(
            kind_of(
                "C1.2",
                clause=RelativeClause(
                    (
                        Predication(
                            VERB,
                            "V2.1",
                            True,
                            kind_of("C1.5", clause=RelativeClause((Predication(CAN, "CAN.3"),))),
                        ),
                    )
                ),
                quantifier="some",
            ),
            Predication(HAS, "HAS.2"),
        ),
        "SOME(C1.2(X.1) AND EXISTS(X.2, C1.5(X.2) AND ABLE(CAN.3(X.2)) AND "
        "ABLE(V2.1(X.1, X.2))), HAS.2(X.1))",
    ),
}


@pytest.mark.parametrize("name", list(SENTENCES))
def test_the_propositional_rendering(name) -> None:
    plan, expected = SENTENCES[name]
    grounding = PARENT if plan.predication.kind == SCALAR else None
    assert render(plan, grounding) == expected


def test_referents_can_be_named_by_their_instances() -> None:
    plan, _ = SENTENCES["chased"]
    assert render(plan, mode="instance") == (
        "C1.3.2(I1.3.2.1) AND C1.4.1(I1.4.1.1) AND "
        "EVENT(SN.3.2, PAST, SIMPLE, V1.2(I1.3.2.1, I1.4.1.1)) AND "
        "EVENT(SN.3.4, PAST, SIMPLE, CAN.7(I1.3.2.1))"
    )
    # a class-level form has variables, and no referents
    plan, expected = SENTENCES["owls eat mice"]
    assert render(plan, mode="instance") == expected
    with pytest.raises(ValueError, match="unknown referent labels"):
        propositional(logical_form(plan, plan.proposition()), "global")


def test_the_formula_of_a_logical_form() -> None:
    plan, _ = SENTENCES["owls that eat mice"]
    proposition = plan.proposition()
    grounded = Proposition(
        proposition.level,
        proposition.subject,
        proposition.predicate,
        True,
        "generic",
        grounding=PARENT,
    )
    form = logical_form(plan, grounded)
    assert form["subject"]["clauses"] == [
        {"kind": "verb", "verb": "V2.1", "patient": {"category": "C1.5", "restriction": []}}
    ]
    assert form["predicate"] == {"kind": "scalar", "pole": "SC.1.HIGH", "class": "C1"}
    assert formula(form) == (
        Quantified(
            "generic",
            (
                Atom("C1.2", ("X.1",)),
                Exists("X.2", (Atom("C1.5", ("X.2",)), Able(Atom("V2.1", ("X.1", "X.2"))))),
            ),
            Atom("SC.1.HIGH", ("X.1", "C1")),
        ),
    )
    plan, _ = SENTENCES["was chased"]
    form = logical_form(plan, plan.proposition(), REFERENTS)
    assert form["subject"] == {
        "instance": CAT,
        "referent": "R.2",
        "noun": "C1.4.1",
        "restriction": [],
        "clauses": [
            {
                "kind": "verb",
                "verb": "V1.2",
                "agent": {"instance": DOG, "referent": "R.1", "noun": "C1.3", "restriction": []},
                "event": "SN.3.2",
                "tense": "past",
                "aspect": "progressive",
            }
        ],
    }
    assert (form["event"], form["tense"], form["aspect"]) == ("SN.3.5", "past", "simple")
    assert formula(form)[-1] == Report("SN.3.5", "past", "simple", Atom("CAN.7", ("R.2",)))
    # a pronoun gives no noun, and a negative capacity is NOT ABLE
    plan, _ = SENTENCES["not able"]
    form = logical_form(plan, plan.proposition(), REFERENTS)
    assert form["subject"] == {
        "instance": DOG,
        "referent": "R.1",
        "noun": None,
        "restriction": [],
    }
    assert formula(form) == (Not(Able(Atom("CAN.3", ("R.1",)))),)


def test_a_noun_phrase_said_twice_is_written_once() -> None:
    # "the owl that can eat the mouse can chase the mouse": the mouse's noun is written once
    plan = SentencePlan(
        the(
            OWL,
            "C1.2",
            clause=RelativeClause((Predication(VERB, "V2.1", True, the(CAT, "C1.5")),)),
        ),
        Predication(VERB, "V1.2", True, the(CAT, "C1.5")),
    )
    assert render(plan) == (
        "C1.2(R.3) AND C1.5(R.2) AND ABLE(V2.1(R.3, R.2)) AND ABLE(V1.2(R.3, R.2))"
    )


def test_renderings_that_cannot_be_read() -> None:
    for text in (
        "C1.3(R.1) AND",
        "C1.3(R.1",
        "ABLE(NOT CAN.3(R.1))",
        "EVENT(SN.1.1, PAST, SIMPLE, ABLE(CAN.3(R.1)))",
        "MOST(C1.3(X.1))",
        "C1.3(R.1) C1.4(R.2)",
        "C1.3[R.1]",
    ):
        with pytest.raises(RenderingError):
            parse_propositional(text)
    with pytest.raises(RenderingError, match="unknown kind of concept"):
        proposition_of(parse_propositional("L.5(R.1)"))
    # a pole without its comparison class has no rendering
    plan, _ = SENTENCES["owls that eat mice"]
    with pytest.raises(RenderingError, match="no comparison class"):
        propositional(plan.proposition().to_json())


def test_a_test_item_names_only_its_scene() -> None:
    # a report that names no event is written with the label of its scene: some event of the
    # scene was this one. Documents keep their event labels.
    plan, rendering = SENTENCES["chased"]
    form = logical_form(plan, plan.proposition(), REFERENTS)
    label = form["event"]
    assert f"EVENT({label}, " in rendering and propositional(form) == rendering
    scene = label.rsplit(".", 1)[0]
    form["event"] = None
    unnamed = propositional(form)
    assert unnamed == rendering.replace(f"EVENT({label}, ", f"EVENT({scene}, ")
    proposition = proposition_of(parse_propositional(unnamed), {v: k for k, v in REFERENTS.items()})
    assert proposition.level == "event" and proposition.event is None
    assert proposition.scene == scene
    assert proposition == dataclasses.replace(plan.proposition(), event=None)
    # the plan of a test item holds the scene's label where a document's holds the event's
    said = dataclasses.replace(plan.predication, event=scene)
    assert SentencePlan(plan.subject, said).proposition() == proposition
    assert (scene_of(label), event_of(label)) == (scene, label)
    assert (scene_of(scene), event_of(scene)) == (scene, None)
