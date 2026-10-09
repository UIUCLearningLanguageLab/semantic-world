"""The renderings: the formal rendering of a sequence of lexemes (stage 1), and the
propositional rendering of a logical form, in the notation of decision 33 (stage 5)."""

# ruff: noqa: E501

from __future__ import annotations

import dataclasses
import json

import pytest
from corpus_support import TINY_WORLD, corpus_config

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
    ALL,
    CAN,
    HAS,
    IS,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    PROJECTION,
    SCALAR,
    SOME,
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
    config = corpus_config(TINY_WORLD, lexicon=knobs)
    return build_lexicon(config, world, Streams(config.seed))


def test_formal_rendering(tiny_world) -> None:
    lexicon = lexicon_of(tiny_world)
    the = lexicon.function_word("the")
    (noun,) = lexicon.lexemes_of("CATEGORY.1.1")
    (verb,) = lexicon.lexemes_of("EVENTTYPE1.2")
    # a content lexeme's gloss is its concept's label, and a function word's is its English gloss
    assert formal([the, noun, verb]) == "the/LEXEME.42 CATEGORY.1.1/LEXEME.2 EVENTTYPE1.2/LEXEME.28"
    assert formal_word(lexicon.function_word("no")) == "no/LEXEME.46"
    assert formal([]) == ""
    for lexeme in lexicon.lexemes:
        gloss, label = formal_word(lexeme).split("/")
        assert label == lexeme.label and lexicon.lexeme(label) is lexeme
        assert gloss == (lexeme.concept if lexeme.content else lexeme.concept.lower())


def test_the_formal_rendering_tells_synonyms_and_homonyms_apart(tiny_world) -> None:
    lexicon = lexicon_of(tiny_world, synonym_rate=1.0, homonym_rate=1.0)
    first, second = lexicon.lexemes_of("CATEGORY.1.1")
    # synonyms share the gloss, and differ in the lexeme label
    assert formal([first]) == "CATEGORY.1.1/LEXEME.2" and formal([second]).startswith(
        "CATEGORY.1.1/LEXEME."
    )
    assert formal([first]) != formal([second])
    # a homonym's two lexemes differ in both
    for earlier, later in lexicon.homonym_pairs():
        a, b = formal_word(earlier).split("/"), formal_word(later).split("/")
        assert a[0] != b[0] and a[1] != b[1]


# ---------------------------------------------------------------------------------------------
# The propositional rendering
# ---------------------------------------------------------------------------------------------

REFERENTS = {"INSTANCE.1.3.2.1": "REF.1", "INSTANCE.1.4.1.1": "REF.2", "INSTANCE.1.2.1.1": "REF.3"}
INSTANCES = {label: instance for instance, label in REFERENTS.items()}
DOG, CAT, OWL = "INSTANCE.1.3.2.1", "INSTANCE.1.4.1.1", "INSTANCE.1.2.1.1"


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


EATS_MICE = RelativeClause((Predication(VERB, "EVENTTYPE2.2.1", True, kind_of("CATEGORY.1.5")),))
OWLS_EAT = RelativeClause((Predication(VERB, "EVENTTYPE2.2.1"),), kind_of("CATEGORY.1.2"))
PARENT = {"comparison": "CATEGORY.1"}

SENTENCES = {
    # the furry dog has legs
    "furry": (
        SentencePlan(
            the(DOG, "CATEGORY.1.3.2", Literal("PROPERTY.12")), Predication(HAS, "PART.4")
        ),
        "CATEGORY.1.3.2(REF.1) AND PROPERTY.12(REF.1) AND PART.4(REF.1)",
    ),
    # the dog that chased the cat ran
    "chased": (
        SentencePlan(
            the(
                DOG,
                "CATEGORY.1.3.2",
                clause=RelativeClause(
                    (
                        happened(
                            VERB,
                            "EVENTTYPE2.1.2",
                            "SCENE.3.EVENTINSTANCE.2",
                            the(CAT, "CATEGORY.1.4.1"),
                        ),
                    )
                ),
            ),
            happened(CAN, "EVENTTYPE1.7", "SCENE.3.EVENTINSTANCE.4"),
        ),
        "CATEGORY.1.3.2(REF.1) AND CATEGORY.1.4.1(REF.2) AND EVENT(SCENE.3.EVENTINSTANCE.2, PAST, SIMPLE, EVENTTYPE2.1.2(REF.1, REF.2)) AND "
        "EVENT(SCENE.3.EVENTINSTANCE.4, PAST, SIMPLE, EVENTTYPE1.7(REF.1))",
    ),
    # the cat that the dog was chasing ran: an object relative, and a progressive event
    "was chased": (
        SentencePlan(
            the(
                CAT,
                "CATEGORY.1.4.1",
                clause=RelativeClause(
                    (
                        happened(
                            VERB, "EVENTTYPE2.1.2", "SCENE.3.EVENTINSTANCE.2", aspect="progressive"
                        ),
                    ),
                    the(DOG, "CATEGORY.1.3"),
                ),
            ),
            happened(CAN, "EVENTTYPE1.7", "SCENE.3.EVENTINSTANCE.5"),
        ),
        "CATEGORY.1.4.1(REF.2) AND CATEGORY.1.3(REF.1) AND EVENT(SCENE.3.EVENTINSTANCE.2, PAST, PROGRESSIVE, EVENTTYPE2.1.2(REF.1, REF.2)) AND "
        "EVENT(SCENE.3.EVENTINSTANCE.5, PAST, SIMPLE, EVENTTYPE1.7(REF.2))",
    ),
    # the penguin can swim, and it can not swim
    "able": (
        SentencePlan(the(DOG, "CATEGORY.1.3"), Predication(CAN, "EVENTTYPE1.3")),
        "CATEGORY.1.3(REF.1) AND ABLE(EVENTTYPE1.3(REF.1))",
    ),
    "not able": (
        SentencePlan(the(DOG, None), Predication(CAN, "EVENTTYPE1.3", False)),
        "NOT ABLE(EVENTTYPE1.3(REF.1))",
    ),
    # the owl can eat the mouse, with a pronoun as the object
    "can eat": (
        SentencePlan(
            the(OWL, "CATEGORY.1.2"), Predication(VERB, "EVENTTYPE2.2.1", True, the(CAT, None))
        ),
        "CATEGORY.1.2(REF.3) AND ABLE(EVENTTYPE2.2.1(REF.3, REF.2))",
    ),
    # the big mouse without fins is small for an animal: a pole names its comparison class
    "poles": (
        SentencePlan(
            the(CAT, "CATEGORY.1", Literal("SCALARDIM.1.HIGH"), Literal("PART.2", False)),
            Predication(SCALAR, "SCALARDIM.2.LOW", False),
        ),
        "CATEGORY.1(REF.2) AND NOT PART.2(REF.2) AND SCALARDIM.1.HIGH(REF.2, CATEGORY.1) AND NOT SCALARDIM.2.LOW(REF.2, CATEGORY.1)",
    ),
    # the bird that is not red is a penguin, and it is edible
    "member": (
        SentencePlan(
            the(DOG, "CATEGORY.1.3", Literal("PROPERTY.4", False)),
            Predication(MEMBER, "CATEGORY.1.3.2"),
        ),
        "CATEGORY.1.3(REF.1) AND NOT PROPERTY.4(REF.1) AND CATEGORY.1.3.2(REF.1)",
    ),
    "edible": (
        SentencePlan(the(DOG, None), Predication(PROJECTION, "CANBE.EVENTTYPE2.1.1")),
        "CANBE.EVENTTYPE2.1.1(REF.1)",
    ),
    # the owl that can eat the mouse that can swim is big
    "nested": (
        SentencePlan(
            the(
                OWL,
                "CATEGORY.1.2",
                clause=RelativeClause(
                    (
                        Predication(
                            VERB,
                            "EVENTTYPE2.2.1",
                            True,
                            the(
                                CAT,
                                "CATEGORY.1.5",
                                clause=RelativeClause((Predication(CAN, "EVENTTYPE1.3"),)),
                            ),
                        ),
                    )
                ),
            ),
            Predication(SCALAR, "SCALARDIM.1.HIGH"),
        ),
        "CATEGORY.1.2(REF.3) AND CATEGORY.1.5(REF.2) AND ABLE(EVENTTYPE1.3(REF.2)) AND ABLE(EVENTTYPE2.2.1(REF.3, REF.2)) AND "
        "SCALARDIM.1.HIGH(REF.3, CATEGORY.1.2)",
    ),
    # most red penguins can swim, and no penguins have fur: the quantifier of the plan is
    # written, whether or not a word states it
    "most": (
        SentencePlan(
            kind_of("CATEGORY.1.3", Literal("PROPERTY.4"), quantifier="most"),
            Predication(CAN, "EVENTTYPE1.3"),
            MOST,
        ),
        "MOST(CATEGORY.1.3(VAR.1) AND PROPERTY.4(VAR.1), ABLE(EVENTTYPE1.3(VAR.1)))",
    ),
    "no": (
        SentencePlan(kind_of("CATEGORY.1.3", quantifier="no"), Predication(HAS, "PART.7"), NEC_NO),
        "NEC(NO(CATEGORY.1.3(VAR.1), PART.7(VAR.1)))",
    ),
    "extensional no": (
        SentencePlan(kind_of("CATEGORY.1.3"), Predication(HAS, "PART.7"), "no"),
        "NO(CATEGORY.1.3(VAR.1), PART.7(VAR.1))",
    ),
    "most not": (
        SentencePlan(
            kind_of("CATEGORY.1.3", quantifier="most"),
            Predication(CAN, "EVENTTYPE1.2", False),
            MOST,
        ),
        "MOST(CATEGORY.1.3(VAR.1), NOT ABLE(EVENTTYPE1.2(VAR.1)))",
    ),
    # all things with wings and with feathers can fly: a rule statement, and the bare plural
    "rule": (
        SentencePlan(
            kind_of("THING", Literal("PART.2"), Literal("PART.5"), quantifier="all"),
            Predication(CAN, "EVENTTYPE1.1"),
            NEC_ALL,
        ),
        "NEC(ALL(PART.2(VAR.1) AND PART.5(VAR.1), ABLE(EVENTTYPE1.1(VAR.1))))",
    ),
    "bare rule": (
        SentencePlan(
            kind_of("THING", Literal("PART.2"), Literal("PROPERTY.3", False)),
            Predication(IS, "PROPERTY.9"),
            NEC_ALL,
        ),
        "NEC(ALL(NOT PROPERTY.3(VAR.1) AND PART.2(VAR.1), PROPERTY.9(VAR.1)))",
    ),
    # owls eat mice: the quantifier ranges over pairs, and a relation fact is never nec
    "owls eat mice": (
        SentencePlan(
            kind_of("CATEGORY.1.2"),
            Predication(VERB, "EVENTTYPE2.2.1", True, kind_of("CATEGORY.1.5")),
            ALL,
        ),
        "ALL(CATEGORY.1.2(VAR.1) AND CATEGORY.1.5(VAR.2), ABLE(EVENTTYPE2.2.1(VAR.1, VAR.2)))",
    ),
    # penguins are birds, and penguins are not fish
    "are birds": (
        SentencePlan(
            kind_of("CATEGORY.1.3.2", quantifier="all"),
            Predication(MEMBER, "CATEGORY.1.3"),
            NEC_ALL,
        ),
        "NEC(ALL(CATEGORY.1.3.2(VAR.1), CATEGORY.1.3(VAR.1)))",
    ),
    "are not fish": (
        SentencePlan(kind_of("CATEGORY.1.3.2"), Predication(MEMBER, "CATEGORY.2", False), NEC_ALL),
        "NEC(ALL(CATEGORY.1.3.2(VAR.1), NOT CATEGORY.2(VAR.1)))",
    ),
    # owls are big: a class-level scalar pole has no quantifier, and names the comparison class
    "owls are big": (
        SentencePlan(kind_of("CATEGORY.1.2"), Predication(SCALAR, "SCALARDIM.1.HIGH")),
        "SCALARDIM.1.HIGH(CATEGORY.1.2, CATEGORY.1)",
    ),
    "owls are not big": (
        SentencePlan(kind_of("CATEGORY.1.2"), Predication(SCALAR, "SCALARDIM.1.HIGH", False)),
        "NOT SCALARDIM.1.HIGH(CATEGORY.1.2, CATEGORY.1)",
    ),
    # owls that eat mice have wings: at least one mouse, and an extensional quantifier
    "owls that eat mice": (
        SentencePlan(kind_of("CATEGORY.1.2", clause=EATS_MICE), Predication(HAS, "PART.2"), ALL),
        "ALL(CATEGORY.1.2(VAR.1) AND EXISTS(VAR.2, CATEGORY.1.5(VAR.2) AND ABLE(EVENTTYPE2.2.1(VAR.1, VAR.2))), PART.2(VAR.1))",
    ),
    # mice that owls eat are small
    "mice that owls eat": (
        SentencePlan(
            kind_of("CATEGORY.1.5", clause=OWLS_EAT, quantifier="most"),
            Predication(IS, "PROPERTY.4"),
            MOST,
        ),
        "MOST(CATEGORY.1.5(VAR.1) AND EXISTS(VAR.2, CATEGORY.1.2(VAR.2) AND ABLE(EVENTTYPE2.2.1(VAR.2, VAR.1))), PROPERTY.4(VAR.1))",
    ),
    # big penguins that can swim chase fish that owls eat
    "both clauses": (
        SentencePlan(
            kind_of(
                "CATEGORY.1.3",
                Literal("SCALARDIM.1.HIGH"),
                clause=RelativeClause((Predication(CAN, "EVENTTYPE1.3"),)),
            ),
            Predication(VERB, "EVENTTYPE2.1.2", True, kind_of("CATEGORY.2.1", clause=OWLS_EAT)),
            ALL,
        ),
        "ALL(CATEGORY.1.3(VAR.1) AND SCALARDIM.1.HIGH(VAR.1, CATEGORY.1.3) AND ABLE(EVENTTYPE1.3(VAR.1)) AND CATEGORY.2.1(VAR.2) AND "
        "EXISTS(VAR.3, CATEGORY.1.2(VAR.3) AND ABLE(EVENTTYPE2.2.1(VAR.3, VAR.2))), ABLE(EVENTTYPE2.1.2(VAR.1, VAR.2)))",
    ),
    # owls that eat mice that can swim have wings: a clause inside a clause
    "deep": (
        SentencePlan(
            kind_of(
                "CATEGORY.1.2",
                clause=RelativeClause(
                    (
                        Predication(
                            VERB,
                            "EVENTTYPE2.2.1",
                            True,
                            kind_of(
                                "CATEGORY.1.5",
                                clause=RelativeClause((Predication(CAN, "EVENTTYPE1.3"),)),
                            ),
                        ),
                    )
                ),
                quantifier="some",
            ),
            Predication(HAS, "PART.2"),
            SOME,
        ),
        "SOME(CATEGORY.1.2(VAR.1) AND EXISTS(VAR.2, CATEGORY.1.5(VAR.2) AND ABLE(EVENTTYPE1.3(VAR.2)) AND "
        "ABLE(EVENTTYPE2.2.1(VAR.1, VAR.2))), PART.2(VAR.1))",
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
        "CATEGORY.1.3.2(INSTANCE.1.3.2.1) AND CATEGORY.1.4.1(INSTANCE.1.4.1.1) AND "
        "EVENT(SCENE.3.EVENTINSTANCE.2, PAST, SIMPLE, EVENTTYPE2.1.2(INSTANCE.1.3.2.1, INSTANCE.1.4.1.1)) AND "
        "EVENT(SCENE.3.EVENTINSTANCE.4, PAST, SIMPLE, EVENTTYPE1.7(INSTANCE.1.3.2.1))"
    )
    # a class-level form has variables, and no referents
    plan, expected = SENTENCES["owls eat mice"]
    assert render(plan, mode="instance") == expected
    with pytest.raises(ValueError, match="unknown referent labels"):
        propositional(logical_form(plan, plan.proposition()), "global")


def test_the_formula_of_a_logical_form() -> None:
    plan, _ = SENTENCES["owls that eat mice"]
    form = logical_form(plan, plan.proposition())
    assert form["subject"]["clauses"] == [
        {
            "kind": "event_type2",
            "label": "EVENTTYPE2.2.1",
            "patient": {"category": "CATEGORY.1.5", "restriction": []},
        }
    ]
    assert form["predicate"] == {"kind": "part", "label": "PART.2"}
    assert form["quantifier"] == ALL
    assert formula(form) == (
        Quantified(
            ALL,
            (
                Atom("CATEGORY.1.2", ("VAR.1",)),
                Exists(
                    "VAR.2",
                    (
                        Atom("CATEGORY.1.5", ("VAR.2",)),
                        Able(Atom("EVENTTYPE2.2.1", ("VAR.1", "VAR.2"))),
                    ),
                ),
            ),
            Atom("PART.2", ("VAR.1",)),
        ),
    )
    # a nec quantifier wraps the extensional one
    plan, _ = SENTENCES["are not fish"]
    form = logical_form(plan, plan.proposition())
    assert form["quantifier"] == NEC_ALL and form["polarity"] is False
    assert formula(form) == (
        Quantified(
            NEC_ALL, (Atom("CATEGORY.1.3.2", ("VAR.1",)),), Not(Atom("CATEGORY.2", ("VAR.1",)))
        ),
    )
    # a class-level scalar pole is one atom of the category and its comparison class
    plan, _ = SENTENCES["owls are big"]
    proposition = plan.proposition()
    grounded = Proposition(
        proposition.level, proposition.subject, proposition.predicate, True, None, grounding=PARENT
    )
    form = logical_form(plan, grounded)
    assert form["quantifier"] is None
    assert form["predicate"] == {
        "kind": "scalar",
        "label": "SCALARDIM.1.HIGH",
        "class": "CATEGORY.1",
    }
    assert formula(form) == (Atom("SCALARDIM.1.HIGH", ("CATEGORY.1.2", "CATEGORY.1")),)
    plan, _ = SENTENCES["was chased"]
    form = logical_form(plan, plan.proposition(), REFERENTS)
    assert form["subject"] == {
        "instance": CAT,
        "referent": "REF.2",
        "noun": "CATEGORY.1.4.1",
        "restriction": [],
        "clauses": [
            {
                "kind": "event_type2",
                "label": "EVENTTYPE2.1.2",
                "agent": {
                    "instance": DOG,
                    "referent": "REF.1",
                    "noun": "CATEGORY.1.3",
                    "restriction": [],
                },
                "event": "SCENE.3.EVENTINSTANCE.2",
                "tense": "past",
                "aspect": "progressive",
            }
        ],
    }
    assert (form["event"], form["tense"], form["aspect"]) == (
        "SCENE.3.EVENTINSTANCE.5",
        "past",
        "simple",
    )
    assert formula(form)[-1] == Report(
        "SCENE.3.EVENTINSTANCE.5", "past", "simple", Atom("EVENTTYPE1.7", ("REF.2",))
    )
    # a pronoun gives no noun, and a negative capacity is NOT ABLE
    plan, _ = SENTENCES["not able"]
    form = logical_form(plan, plan.proposition(), REFERENTS)
    assert form["subject"] == {
        "instance": DOG,
        "referent": "REF.1",
        "noun": None,
        "restriction": [],
    }
    assert formula(form) == (Not(Able(Atom("EVENTTYPE1.3", ("REF.1",)))),)


def test_a_noun_phrase_said_twice_is_written_once() -> None:
    # "the owl that can eat the mouse can chase the mouse": the mouse's noun is written once
    plan = SentencePlan(
        the(
            OWL,
            "CATEGORY.1.2",
            clause=RelativeClause(
                (Predication(VERB, "EVENTTYPE2.2.1", True, the(CAT, "CATEGORY.1.5")),)
            ),
        ),
        Predication(VERB, "EVENTTYPE2.1.2", True, the(CAT, "CATEGORY.1.5")),
    )
    assert render(plan) == (
        "CATEGORY.1.2(REF.3) AND CATEGORY.1.5(REF.2) AND ABLE(EVENTTYPE2.2.1(REF.3, REF.2)) AND ABLE(EVENTTYPE2.1.2(REF.3, REF.2))"
    )


def test_renderings_that_cannot_be_read() -> None:
    for text in (
        "CATEGORY.1.3(REF.1) AND",
        "CATEGORY.1.3(REF.1",
        "ABLE(NOT EVENTTYPE1.3(REF.1))",
        "EVENT(SCENE.1.EVENTINSTANCE.1, PAST, SIMPLE, ABLE(EVENTTYPE1.3(REF.1)))",
        "MOST(CATEGORY.1.3(VAR.1))",
        "CATEGORY.1.3(REF.1) CATEGORY.1.4(REF.2)",
        "CATEGORY.1.3[REF.1]",
        "NEC(MOST(CATEGORY.1.3(VAR.1), PART.7(VAR.1)))",
        "NEC(CATEGORY.1.3(REF.1))",
    ):
        with pytest.raises(RenderingError):
            parse_propositional(text)
    with pytest.raises(RenderingError, match="unknown kind of concept"):
        proposition_of(parse_propositional("LEXEME.5(REF.1)"))
    # a pole without its comparison class has no rendering
    plan, _ = SENTENCES["owls are big"]
    with pytest.raises(RenderingError, match="no comparison class"):
        propositional(plan.proposition().to_json())


def test_a_test_item_names_only_its_scene() -> None:
    # a report that names no event is written with the label of its scene: some event of the
    # scene was this one. Documents keep their event labels.
    plan, rendering = SENTENCES["chased"]
    form = logical_form(plan, plan.proposition(), REFERENTS)
    label = form["event"]
    assert f"EVENT({label}, " in rendering and propositional(form) == rendering
    scene = scene_of(label)
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
