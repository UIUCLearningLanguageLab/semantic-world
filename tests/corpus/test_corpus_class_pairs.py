"""Stage c5a of ``docs/specs/WORLD_PHASE_C.md``: the corpus at the class level, grammar and
statements.

The stage adds the bound pronoun ``them`` and the generic noun ``SOMEPLACE``; relative clauses
with a prepositional predicate, with ``grammar.word_order.stranding``; quantified objects of
prepositions; causal statements with pair literals (a pair fluent, ``SAME``, and ``SOME``),
about the event types between entities and about the place event types; profile statements; and
Jon's ruling WM.74 on comparisons. A conditional effect whose condition holds a pair literal is
stated in neither form (WM.59; its test is in ``test_corpus_pairs_places.py``).

Every causal and profile statement must be true by the oracle's re-reading of
``definition.json``, read back by ``interpret`` in every word order, and parse back from its
propositional rendering. The fast tier runs the tiny world and the two worlds whose event files
give pair fluents by hand (``tiny_pairs.yaml`` and ``tiny_place_chain.yaml``, which between
them hold a symmetric and an asymmetric pair fluent, ``SAME``, ``SOME`` in both polarities, a
functional add with its compiled literals, moves, and acts); the slow tier the default world.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections import Counter

import numpy as np
import pytest
from corpus_support import REPO, Case, corpus_config

from semantic_world.corpus import build_lexicon, load_world
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.grammar import (
    CLASS_NP,
    GrammarError,
    NounPhrase,
    Predication,
    SentencePlan,
    bare_things,
    check_plan,
    event_phrase_of,
    pair_phrase_of,
    quantified_things,
    them,
)
from semantic_world.corpus.interpret import interpret
from semantic_world.corpus.lexicon import FUNCTION_WORDS, GENERIC, NOUN, THEM_WORD
from semantic_world.corpus.logical import logical_form, with_cut
from semantic_world.corpus.mentions import Mentions, plan_for
from semantic_world.corpus.planner import (
    CAUSAL,
    COMPARATIVE_SECTION,
    PARTICIPANTS,
    PLACE_COMPARATIVE,
    PLACES,
)
from semantic_world.corpus.propositions import (
    AGENT,
    ASYMMETRIC,
    CAUSAL_KINDS,
    CLASS,
    COMPARATIVE,
    EFFECT,
    NEC_ALL,
    PAIR_STATE,
    PATIENT,
    PRECONDITION,
    PROFILE,
    SYMMETRIC,
    EventTerm,
    PairTerm,
    Predicate,
    Proposition,
    is_pair_label,
    is_same_label,
    is_some_label,
    pole_dimension,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, leaves
from semantic_world.corpus.renderings import (
    RenderingError,
    parse_propositional,
    proposition_of,
    propositional,
)
from semantic_world.corpus.request import wordform_request
from semantic_world.corpus.streams import Streams
from semantic_world.corpus.testsets import CAUSAL_LEVELS

WORLDS = ("tiny", pytest.param("default", marks=pytest.mark.slow))
HAND_WORLDS = ("tiny_pairs", "tiny_place_chain")
EVERY_WORLD = ("tiny", *HAND_WORLDS, pytest.param("default", marks=pytest.mark.slow))
HAND_FILES = {
    "tiny_pairs": "data/world/tiny_pairs.yaml",
    "tiny_place_chain": "data/world/tiny_place_chain.yaml",
}
CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
ADPOSITIONS = ("preposition", "postposition")
STRANDINGS = ("strand", "pied_pipe")
MANY = {
    "documents": {
        "mix": {
            "encyclopedic_category": 0.1,
            "encyclopedic_feature": 0.4,
            "entity": 0.25,
            "situational": 0.25,
        },
        "comparative_rate": 0.9,
    },
    "propositions": {"causal_statement_rate": 0.9},
}
"""Many feature documents with causal statements, and many comparisons."""


@pytest.fixture(scope="module")
def worlds(cases, tmp_path_factory: pytest.TempPathFactory):
    """The worlds of this file's tests, by name: the ``tiny`` and ``default`` cases, and the two
    worlds of the hand-written event files, each made once with its world run folder."""
    root = tmp_path_factory.mktemp("class_pairs")
    made: dict[str, Case] = {}

    def world(name: str) -> Case:
        if name not in HAND_FILES:
            return cases(name)
        if name not in made:
            made[name] = Case(name, str(REPO / HAND_FILES[name]), root / name)
        return made[name]

    return world


def new_statements(facts) -> list[Proposition]:
    """The causal statements of stage c5a: with a pair literal, or about a place event type."""
    return [p for p in facts.causal_statements() if p.predicate.pair_literal or p.subject.place]


def glosses(sentence) -> str:
    """The sentence in glosses: the formal rendering without the lexeme labels."""
    words = []
    for word in sentence.formal.split():
        gloss, _, rest = word.partition("/")
        _, hyphen, affix = rest.partition("-")
        words.append(gloss + hyphen + affix)
    return " ".join(words)


def form_of(case: Case, proposition: Proposition, **sections) -> dict:
    """A statement's JSON logical form, as the documents write it."""
    facts = case.facts(**sections)
    plan = plan_for(facts, proposition, bare=True)
    form = logical_form(plan, proposition, {}, facts.config.scalar_z, facts.truth.cuts)
    return json.loads(json.dumps(form))


# ---------------------------------------------------------------------------------------------
# The words
# ---------------------------------------------------------------------------------------------


def test_them_and_someplace_are_the_new_words(runs, tmp_path) -> None:
    """``them`` is a base function word, appended last, and ``SOMEPLACE`` the generic head
    noun for places: a content noun, the last content lexeme of a world with places, so that
    every earlier content lexeme keeps its label. A world without places has no ``SOMEPLACE``.
    The word-form request lists both."""
    assert FUNCTION_WORDS[-1] == THEM_WORD
    corpus = runs("tiny")
    lexicon = corpus.planner.lexicon
    them_word = lexicon.function_word(THEM_WORD)
    assert them_word.concept == "THEM" and not them_word.content
    glosses_in_order = [x.gloss for x in lexicon.function_lexemes]
    assert glosses_in_order[: len(FUNCTION_WORDS)] == list(FUNCTION_WORDS)
    assert [c.label for c in lexicon.of_type(GENERIC)] == ["THING", "SOMEPLACE"]
    last = lexicon.content_lexemes[-1]
    assert last.concept == "SOMEPLACE" and last.pos == NOUN and last.content
    assert last.label == f"LEXEME.{len(lexicon.content_lexemes)}"
    assert lexicon.is_named("SOMEPLACE")
    request = wordform_request(corpus)
    assert THEM_WORD in request["function_words"]
    assert request["lexemes"][-1] == {"label": last.label, "concept": "SOMEPLACE", "pos": NOUN}
    # a world without places has no generic noun for them, and every language has the pronoun
    text = (REPO / "data/world/tiny.yaml").read_text(encoding="utf-8")
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith(("places:", "pair_fluents:", "place_event_types:"))
    ]
    lines += ["places: null", "pair_fluents: {count: 0, locations: 0}"]
    path = tmp_path / "no_places.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = corpus_config(str(path))
    plain = build_lexicon(config, load_world(config), Streams(config.seed))
    assert [c.label for c in plain.of_type(GENERIC)] == ["THING"]
    assert plain.function_word(THEM_WORD).gloss == THEM_WORD


# ---------------------------------------------------------------------------------------------
# Acceptance: every causal and profile statement is true by a re-reading of definition.json
# ---------------------------------------------------------------------------------------------


def _record_pair_entries(record: dict, leaf: str) -> set[tuple]:
    """The pair entries of one event type of ``definition.json`` that a statement can state,
    as ``(kind, role, label, value)``: an unconditional pair effect and a pair literal about
    the role of the first argument, and on a symmetric pair fluent about either role; ``SAME``
    about either role; ``SOME`` about its role. ``SOMEOF``, and an effect under a condition,
    give none here (the conditional statements are judged by the oracle alone)."""
    symmetric = {s["label"] for s in record["symbols"] if s.get("symmetry") == "symmetric"}
    entry = next(e for e in record["event_types"] if e["label"] == leaf)
    found: set[tuple] = set()
    for effect in entry["effects"]:
        if effect.get("kind") != "pair" or effect.get("condition"):
            continue
        fluent = effect["pair_fluent"]
        roles = [effect["first"]] + ([effect["second"]] if fluent in symmetric else [])
        found |= {(EFFECT, role, fluent, bool(effect["value"])) for role in roles}
    for literal in entry["precondition"]["literals"]:
        kind, value = literal.get("kind"), bool(literal.get("value"))
        if kind == "pair":
            fluent = literal["pair_fluent"]
            roles = [literal["first"]] + ([literal["second"]] if fluent in symmetric else [])
            found |= {(PRECONDITION, role, fluent, value) for role in roles}
        elif kind == "pair_same":
            label = f"SAME.{literal['pair_fluent']}"
            found |= {
                (PRECONDITION, role, label, value) for role in (literal["first"], literal["second"])
            }
        elif kind == "pair_some":
            found.add((PRECONDITION, literal["role"], f"SOME.{literal['pair_fluent']}", value))
    return found


@pytest.mark.parametrize("name", EVERY_WORLD)
def test_statements_with_pair_literals_are_the_definitions_entries(worlds, name) -> None:
    """``Facts.causal_statements`` lists, beside the statements of the earlier stages, exactly
    the pair entries of ``definition.json`` whose event type and pair fluent have words: for
    an event type, its own entries, and for a category of event types, the entries that every
    event type below it has. Each statement is true by the corpus's truth test and by the
    oracle, and the statements about place event types are about their agent."""
    case = worlds(name)
    facts, oracle = case.facts(), case.oracle()
    world = case.world
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    expected: set[tuple] = set()
    for entry in record["event_types"]:
        label = entry["label"]
        if label not in facts.named or not oracle.able_somewhere(label):
            continue
        shared: set[tuple] | None = None
        for leaf in oracle.leaves_below(label):
            entries = _record_pair_entries(record, leaf)
            shared = entries if shared is None else shared & entries
        for kind, role, fluent, value in shared or ():
            word = fluent.removeprefix("SOME.")  # SOME is said with the pair fluent's preposition
            if word not in facts.named:
                continue
            if entry["kind"] == "place_event_type" and role != AGENT:
                continue
            expected.add((label, kind, role, fluent, value))
    listed = {
        (p.subject.event_type, p.predicate.kind, p.subject.role, p.predicate.label, value)
        for p in facts.causal_statements()
        for value in (p.predicate.value,)
        if p.predicate.pair_literal and not p.subject.condition
    }
    assert listed == expected
    statements = new_statements(facts)
    assert statements, "every world of this test has a pair literal with words"
    for statement in statements:
        assert isinstance(statement.subject, EventTerm)
        assert statement.level == CLASS and statement.quantifier == NEC_ALL and statement.polarity
        assert statement.predicate.kind in CAUSAL_KINDS
        assert facts.truth.is_true(statement)
        assert statement.grounding["test"] == "definition"
        form = with_cut(statement.to_json(), facts.truth.cuts)
        assert oracle.truth(json.loads(json.dumps(form))) is True, form
        if statement.subject.place:
            assert statement.subject.role == AGENT
            assert world.is_place_event_type(statement.subject.event_type)
        label = statement.predicate.label
        if statement.predicate.pair_literal:
            # the causal record names the entry: the fluent, and the second role of a pair
            causal = statement.causal_record()
            assert causal["fluent"] == label
            two = is_pair_label(label) or is_same_label(label)
            assert ("second" in causal) == two
            if two:
                assert {causal["role"], causal["second"]} == {AGENT, PATIENT}
        # the opposite value is false, by both
        if statement.predicate.value is not None:
            flipped = dataclasses.replace(
                statement,
                predicate=dataclasses.replace(
                    statement.predicate, value=not statement.predicate.value
                ),
            )
            assert facts.truth.evaluate(flipped).true is False
            assert oracle.causal(flipped.to_json()) is False
    # no statement is made for SOMEOF (WM.55): no label of a statement names it
    assert not any("SOMEOF" in p.predicate.label for p in facts.causal_statements())
    # the report counts every pair entry once, stated or not
    report = facts.causal_report["pair_entries"]
    assert report["stated"] + sum(report["unstated"].values()) == (
        report["effects"] + report["precondition_literals"]
    )
    assert report["stated"] > 0


@pytest.mark.parametrize("name", EVERY_WORLD)
def test_profile_statements_are_the_symmetries_of_the_definition(worlds, name) -> None:
    """A profile statement is made for each symmetric or asymmetric pair fluent between
    entities with a preposition, and for no other: its form is the symbol's symmetry in
    ``definition.json``, it is true by the truth test and by the oracle, and the other form is
    false by both. A general pair fluent, a location fluent, and a functional or total profile
    make no statement."""
    case = worlds(name)
    facts, oracle = case.facts(), case.oracle()
    record = json.loads((case.folder / "definition.json").read_text(encoding="utf-8"))
    symbols = {s["label"]: s for s in record["symbols"] if s["kind"] == "pair_fluent"}
    expected = {
        (label, symbol["symmetry"])
        for label, symbol in symbols.items()
        if symbol["symmetry"] in (SYMMETRIC, ASYMMETRIC)
        and not symbol.get("total")
        and label in facts.named
    }
    statements = facts.profile_statements()
    assert {(p.predicate.label, p.predicate.form) for p in statements} == expected
    for statement in statements:
        assert isinstance(statement.subject, PairTerm)
        assert statement.quantifier == NEC_ALL and statement.polarity
        assert statement.predicate.kind == PROFILE
        assert facts.truth.is_true(statement)
        assert facts.profile_statements(statement.predicate.label) == (statement,)
        grounding = statement.grounding
        assert grounding["symmetry"] == statement.predicate.form
        assert grounding["test"] == "definition" and isinstance(grounding["can_hold"], bool)
        form = form_of(case, statement)
        assert form["predicate"] == {
            "kind": "profile",
            "label": statement.predicate.label,
            "form": statement.predicate.form,
        }
        assert form["subject"] == {
            "head": "THING",
            "pair": statement.predicate.label,
            "role": "second",
        }
        assert oracle.truth(form) is True
        assert Proposition.from_json(form) == statement
        other = SYMMETRIC if statement.predicate.form == ASYMMETRIC else ASYMMETRIC
        false = dataclasses.replace(
            statement, predicate=dataclasses.replace(statement.predicate, form=other)
        )
        assert facts.truth.evaluate(false).valid and not facts.truth.is_true(false)
        assert oracle.profile(false.to_json()) is False
    # a location fluent, an unknown pair fluent, and a statement about two pair fluents are
    # not valid
    truth = facts.truth
    for label in case.world.location_fluents:
        about = Proposition(
            CLASS, PairTerm(label), Predicate(PROFILE, label, form=SYMMETRIC), True, NEC_ALL
        )
        assert not truth.evaluate(about).valid and oracle.profile(about.to_json()) is None
    if len(statements) > 1:
        first, second = statements[0], statements[1]
        mixed = dataclasses.replace(first, subject=second.subject)
        assert not truth.evaluate(mixed).valid


@pytest.mark.parametrize("name", HAND_WORLDS)
def test_the_hand_worlds_hold_every_kind_of_statement(worlds, name) -> None:
    """The two worlds of the hand-written event files give the kinds of statement that the
    grammar tests below run over: between them a pair effect in both values, a pair
    precondition in both values, ``SAME``, ``SOME`` in both polarities, a symmetric and an
    asymmetric profile, and the statements about an act at a place (the tiny world has the
    moves)."""
    facts = worlds(name).facts()
    kinds = Counter()
    for p in new_statements(facts):
        label = p.predicate.label
        what = "same" if is_same_label(label) else "some" if is_some_label(label) else "pair"
        if p.predicate.pair_literal:
            kinds[(p.predicate.kind, what, p.predicate.value)] += 1
        if p.subject.place:
            kinds["place"] += 1
    forms = {p.predicate.form for p in facts.profile_statements()}
    if name == "tiny_pairs":
        assert kinds[(EFFECT, "pair", True)] and kinds[(EFFECT, "pair", False)]
        assert kinds[(PRECONDITION, "pair", True)] and kinds[(PRECONDITION, "pair", False)]
        assert kinds[(PRECONDITION, "same", True)]
        assert kinds[(PRECONDITION, "some", False)]  # the compiled literal of a functional add
        assert forms == {SYMMETRIC, ASYMMETRIC}
    else:
        assert kinds[(PRECONDITION, "some", True)]
        assert kinds["place"] > 0
        assert kinds[(EFFECT, "pair", True)] and kinds[(PRECONDITION, "same", True)]
        assert forms == {SYMMETRIC}


@pytest.mark.parametrize("name", HAND_WORLDS)
def test_profile_statements_hold_in_the_replayed_scenes(worlds, name) -> None:
    """On a corpus of each hand world, whose scenes hold pairs of the pair fluents, every
    profile statement holds of every pair at every time point of every scene, by the oracle's
    replayed pairs, and the other form does not; and every causal sentence of the corpus with
    a pair literal is true by the oracle and reads back."""
    from semantic_world.corpus.generate import generate

    case = worlds(name)
    settings = {"documents": {"count": 40}, "test_sets": {"size": 5}}
    corpus = generate(case.config(**settings), case.world)
    oracle = case.oracle(**settings)
    scenes = [json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes]
    statements = corpus.planner.facts.profile_statements()
    assert statements
    for statement in statements:
        label = statement.predicate.label
        form = statement.to_json()
        seen = sum(len(pairs[label]) for scene in scenes for pairs in oracle.pairs_at(scene))
        assert seen > 0, "the scenes hold pairs of the pair fluent"
        assert statement.grounding["can_hold"] is True
        assert oracle.profile_held(form, scenes)
        other = SYMMETRIC if statement.predicate.form == ASYMMETRIC else ASYMMETRIC
        form["predicate"]["form"] = other
        assert not oracle.profile_held(form, scenes)
    lexicon = corpus.planner.lexicon
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert not proposition.profile
            if not (proposition.causal and proposition.predicate.pair_literal):
                continue
            assert oracle.truth(json.loads(json.dumps(sentence.logical_form))) is True
            read = interpret(
                sentence.sentence.tree,
                lexicon,
                sentence.sentence.referents,
                sentence.sentence.events,
                proposition.quantifier,
            )
            assert read == sentence.plan and read.proposition() == proposition


# ---------------------------------------------------------------------------------------------
# Acceptance: trees for all six clause orders, interpret, and the rendering
# ---------------------------------------------------------------------------------------------


def _first_token(tree, node) -> int:
    """The position of the first token below a node, in the order of the sentence."""
    tokens: list[int] = []

    def walk(current, offset: int) -> int:
        for child in current[1:]:
            if isinstance(child, str):
                if current is node or inside[0]:
                    tokens.append(offset)
                offset += 1
            else:
                entered = child is node
                if entered:
                    inside[0] += 1
                offset = walk(child, offset)
                if entered:
                    inside[0] -= 1
        return offset

    inside = [1 if tree is node else 0]
    walk(tree, 0)
    return min(tokens)


def _children(node, *labels: str) -> list:
    return [c for c in node[1:] if not isinstance(c, str) and c[0] in labels]


@pytest.mark.parametrize("name", HAND_WORLDS)
def test_statements_in_the_six_clause_orders_with_stranding_and_pied_piping(worlds, name) -> None:
    """With stranding and pied-piping, each combined with the preposition before and after
    its noun phrase, and with ``before`` at either end of the verb phrase, every causal
    statement with a pair literal or about a place event type, and every profile statement,
    has a correct tree in all six clause orders; ``interpret(tree)`` recovers its plan and its
    proposition; its propositional rendering parses back; and its reading is ``causal`` or
    ``profile``."""
    case = worlds(name)
    facts = case.facts()
    statements = new_statements(facts) + list(facts.profile_statements())
    assert statements
    orders = itertools.product(
        CLAUSE_ORDERS, ADPOSITIONS, STRANDINGS, ("after_predicate", "before_predicate")
    )
    for clause, adposition, stranding, before in orders:
        word_order = {
            "clause": clause,
            "adposition": adposition,
            "stranding": stranding,
            "before": before,
        }
        settings = {
            "grammar": {"word_order": word_order, "morphology": {"number": {"enabled": True}}}
        }
        config = case.config(**settings)
        streams = Streams(config.seed)
        realizer = Realizer(config, case.lexicon(**settings), streams)
        lexicon = realizer.lexicon
        for proposition, bare in itertools.product(statements, (True, False)):
            plan = plan_for(facts, proposition, bare=bare)
            check_plan(plan)
            assert plan.proposition() == proposition
            sentence = realizer.realize(plan, np.random.default_rng(0))
            tree = sentence.tree
            assert tuple(leaves(tree)) == sentence.tokens
            read = interpret(
                tree, lexicon, sentence.referents, sentence.events, proposition.quantifier
            )
            assert read == plan, (word_order, glosses(sentence))
            assert read.proposition() == proposition
            form = logical_form(plan, proposition, {}, config.scalar_z, facts.truth.cuts)
            text = propositional(json.loads(json.dumps(form)))
            assert proposition_of(parse_propositional(text)) == proposition, text
            wanted = "profile" if proposition.profile else "causal"
            assert readings(tree, lexicon, config)[0] == wanted
            _check_tree(sentence, plan, word_order)


def _check_tree(sentence, plan: SentencePlan, word_order: dict[str, str]) -> None:
    """The properties of a statement's tree under a word order: the main clause's subject,
    verb phrase, and prepositional phrase stand in the clause order, the preposition before or
    after its object, ``before`` at its end of the verb phrase, and, in the relative clause of
    a profile statement, the preposition stranded where the object would stand, or at the front
    of the clause with the relative word."""
    tree = sentence.tree
    clause = word_order["clause"]
    preposition = word_order["adposition"] == "preposition"
    (subject,) = _children(tree, "NP-SBJ")
    (verb_phrase,) = _children(tree, "VP")
    predication = plan.predication
    if predication.prepositional:
        neighbors = abs(clause.index("V") - clause.index("O")) == 1
        holder = verb_phrase if neighbors else tree
        (phrase,) = _children(holder, "PP")
        (adposition,) = _children(phrase, "P")
        (target,) = _children(phrase, "NP-OBJ")
        assert [c[0] for c in phrase[1:]] == (["P", "NP-OBJ"] if preposition else ["NP-OBJ", "P"])
        # the object: the bound pronoun, or the quantified generic noun
        if predication.object.bound_pronoun:
            assert [c[0] for c in target[1:]] == ["Pro"]
        else:
            assert {c[0] for c in target[1:]} == {"Det", "N"}
            assert not _children(verb_phrase, "Neg")
        # the clause order of the subject, the verb (its auxiliary), and the object
        (auxiliary,) = _children(verb_phrase, "AUX")
        position = {
            "S": _first_token(tree, subject),
            "V": _first_token(tree, auxiliary),
            "O": _first_token(tree, phrase),
        }
        assert sorted(position, key=position.get) == list(clause), (clause, position)
        adverbs = _children(verb_phrase, "Adv")
        assert len(adverbs) == (predication.kind == PRECONDITION)
        if adverbs:
            edge = verb_phrase[-1] if word_order["before"] == "after_predicate" else verb_phrase[1]
            assert edge is adverbs[0]
    if not plan.profile:
        return
    # the relative clause of a profile statement's subject
    (relative,) = _children(subject, "RC")
    (inner_subject,) = _children(relative, "NP-SBJ")
    (inner_verb,) = _children(relative, "VP")
    if word_order["stranding"] == "pied_pipe":
        front = relative[1]
        assert front[0] == "PP"
        assert [c[0] for c in front[1:]] == (["P", "Rel"] if preposition else ["Rel", "P"])
        assert not _children(inner_verb, "PP")
        order = [c[0] for c in relative[2:]]
        subject_first = clause.index("S") < clause.index("V")
        assert order == (["NP-SBJ", "VP"] if subject_first else ["VP", "NP-SBJ"])
    else:
        assert relative[1][0] == "Rel"
        neighbors = abs(clause.index("V") - clause.index("O")) == 1
        (stranded,) = _children(inner_verb if neighbors else relative, "PP")
        assert [c[0] for c in stranded[1:]] == ["P"], "the gap has no word"
        (auxiliary,) = _children(inner_verb, "AUX")
        position = {
            "S": _first_token(tree, inner_subject),
            "V": _first_token(tree, auxiliary),
            "O": _first_token(tree, stranded),
        }
        assert sorted(position, key=position.get) == list(clause)


def test_the_words_of_the_statements_in_the_english_like_order(worlds) -> None:
    """The specification's examples, in glosses, in the default word order, on the two hand
    worlds: the bound pronoun, the generic noun for places, the quantified object, and the
    preposition stranded or moved to the front of the clause."""
    pairs = worlds("tiny_pairs")
    facts = pairs.facts()
    world = pairs.world
    symmetric = next(
        f for f in facts.entity_pair_prepositions if world.pair_fluent_info[f].symmetric
    )
    asymmetric = next(
        f
        for f in facts.entity_pair_prepositions
        if world.pair_fluent_info[f].symmetry == ASYMMETRIC
    )

    def said(case, proposition, **word_order) -> str:
        settings = {"grammar": {"word_order": word_order}} if word_order else {}
        realizer = Realizer(case.config(**settings), case.lexicon(**settings), Streams(1))
        plan = plan_for(case.facts(), proposition, bare=True)
        return glosses(realizer.realize(plan, np.random.default_rng(0)))

    (near,) = facts.profile_statements(symmetric)
    (above,) = facts.profile_statements(asymmetric)
    assert said(pairs, near) == f"THING that THING is {symmetric} is {symmetric} them"
    assert said(pairs, above) == f"THING that THING is {asymmetric} is not {asymmetric} them"
    assert (
        said(pairs, near, stranding="pied_pipe")
        == f"THING {symmetric} that THING is is {symmetric} them"
    )
    assert (
        said(pairs, near, stranding="pied_pipe", adposition="postposition")
        == f"THING that {symmetric} THING is is them {symmetric}"
    )
    assert propositional(form_of(pairs, near)) == (
        f"NEC(ALL({symmetric}(VAR.2, VAR.1), {symmetric}(VAR.1, VAR.2)))"
    )
    assert propositional(form_of(pairs, above)) == (
        f"NEC(ALL({asymmetric}(VAR.2, VAR.1), NOT {asymmetric}(VAR.1, VAR.2)))"
    )
    # "things that approach things become near them", and the compiled literal of a functional
    # add, "things that mount things are <carrying> no things before"
    approach = next(
        p
        for p in facts.causal_statements(kind=EFFECT)
        if p.predicate.label == symmetric and p.predicate.value and p.subject.role == AGENT
    )
    verb = approach.subject.event_type
    assert said(pairs, approach) == f"THING that {verb} THING become {symmetric} them"
    assert propositional(form_of(pairs, approach)) == (
        f"NEC(ALL(EVENT(EVENTVAR.1, {verb}(VAR.1, VAR.2)), "
        f"AFTER(EVENTVAR.1, {symmetric}(VAR.1, VAR.2))))"
    )
    # the same effect about the patient, because the pair fluent is symmetric
    mirrored = dataclasses.replace(approach, subject=EventTerm(verb, PATIENT))
    assert mirrored in facts.causal_statements()
    assert said(pairs, mirrored) == f"THING that THING {verb} become {symmetric} them"
    assert f"AFTER(EVENTVAR.1, {symmetric}(VAR.2, VAR.1))" in propositional(
        form_of(pairs, mirrored)
    )
    none = next(p for p in facts.causal_statements() if is_some_label(p.predicate.label))
    fluent = none.predicate.label.removeprefix("SOME.")
    verb = none.subject.event_type
    assert not none.predicate.value
    assert said(pairs, none) == f"THING that {verb} THING is {fluent} no THING before"
    assert propositional(form_of(pairs, none)) == (
        f"NEC(ALL(EVENT(EVENTVAR.1, {verb}(VAR.1, VAR.2)), "
        f"BEFORE(EVENTVAR.1, NOT SOME.{fluent}(VAR.1))))"
    )
    # a move of the tiny world: "things that go to places become at them", and its compiled
    # literal, "are not at them before"
    tiny = worlds("tiny")
    facts = tiny.facts()
    (location,) = facts.location_prepositions
    move = next(v for v in facts.place_verbs if tiny.world.place_event_kind(v) == "move")
    become = next(
        p for p in facts.causal_statements(event_type=move, kind=EFFECT) if p.predicate.pair_literal
    )
    before = next(
        p
        for p in facts.causal_statements(event_type=move, kind=PRECONDITION)
        if p.predicate.pair_literal
    )
    assert said(tiny, become) == f"THING that {move} SOMEPLACE become {location} them"
    assert said(tiny, before) == f"THING that {move} SOMEPLACE is not {location} them before"
    assert propositional(form_of(tiny, before)) == (
        f"NEC(ALL(EVENT(EVENTVAR.1, {move}(VAR.1, VAR.2)), "
        f"BEFORE(EVENTVAR.1, NOT {location}(VAR.1, VAR.2))))"
    )
    # the bound pronoun of a statement about a place event type refers to a place
    plan = plan_for(facts, become, bare=True)
    assert plan.predication.object == them("SOMEPLACE")
    assert plan.subject == event_phrase_of(become.subject)
    assert plan.subject.clause.predications[0].object == NounPhrase(
        CLASS_NP, "SOMEPLACE", "SOMEPLACE"
    )
    # the place chain: an act at a place, "things that rest at places are at them before", and
    # SOME in the positive, "things that wake are near some things before"
    chain = worlds("tiny_place_chain")
    facts = chain.facts()
    (location,) = facts.location_prepositions
    act = next(v for v in facts.place_verbs if chain.world.place_event_kind(v) == "act")
    at = next(
        p
        for p in facts.causal_statements(event_type=act, kind=PRECONDITION)
        if p.predicate.pair_literal
    )
    assert said(chain, at) == f"THING that {act} SOMEPLACE is {location} them before"
    some = next(
        p
        for p in facts.causal_statements()
        if is_some_label(p.predicate.label) and p.predicate.value
    )
    fluent = some.predicate.label.removeprefix("SOME.")
    assert (
        said(chain, some) == f"THING that {some.subject.event_type} is {fluent} some THING before"
    )


def test_plans_and_forms_the_grammar_and_the_truth_tests_refuse(worlds) -> None:
    """The limits of the new grammar: the bound pronoun is the object of a causal or a profile
    statement's preposition and of nothing else; a quantified object states ``SOME`` in a
    precondition statement, never negated; ``SAME`` and ``SOME`` are never effects; a profile
    statement is about one pair fluent, and its subject's relative clause has the bare generic
    noun as its subject; and the limits of note 4 of stage b3b stay (one relative clause per
    noun phrase, an object relative with one verb)."""
    case = worlds("tiny_pairs")
    facts = case.facts()
    truth = facts.truth
    world = case.world
    symmetric = next(
        f for f in facts.entity_pair_prepositions if world.pair_fluent_info[f].symmetric
    )
    (same,) = facts.same_prepositions
    (profile,) = facts.profile_statements(symmetric)
    good = plan_for(facts, profile, bare=True)
    check_plan(good)
    verb = facts.verbs[-1]
    subject = event_phrase_of(EventTerm(verb, AGENT))
    refused = {
        "the bound pronoun as a subject": SentencePlan(
            them(), Predication(PROFILE, symmetric, True, them()), NEC_ALL
        ),
        "a profile about two pair fluents": SentencePlan(
            good.subject, Predication(PROFILE, same, True, them()), NEC_ALL
        ),
        "a profile with a noun as its object": SentencePlan(
            good.subject, Predication(PROFILE, symmetric, True, bare_things()), NEC_ALL
        ),
        "a profile with a quantified object": SentencePlan(
            good.subject, Predication(PROFILE, symmetric, True, quantified_things(True)), NEC_ALL
        ),
        "a profile that is not nec_all": SentencePlan(good.subject, good.predication, "all"),
        "a profile whose relative clause has a category as its subject": SentencePlan(
            dataclasses.replace(
                good.subject,
                clause=dataclasses.replace(
                    good.subject.clause,
                    agent=NounPhrase(CLASS_NP, facts.categories[0], facts.categories[0]),
                ),
            ),
            good.predication,
            NEC_ALL,
        ),
        "a negated SOME": SentencePlan(
            subject, Predication(PRECONDITION, symmetric, False, quantified_things(False)), NEC_ALL
        ),
        "SOME as an effect": SentencePlan(
            subject, Predication(EFFECT, symmetric, True, quantified_things(True)), NEC_ALL
        ),
        "SAME as an effect": SentencePlan(
            subject, Predication(EFFECT, same, True, them()), NEC_ALL
        ),
        "SAME with a quantified object": SentencePlan(
            subject, Predication(PRECONDITION, same, True, quantified_things(True)), NEC_ALL
        ),
        "a pair literal without an object": SentencePlan(
            subject, Predication(EFFECT, symmetric, True), NEC_ALL
        ),
        "a fluent with an object": SentencePlan(
            subject, Predication(EFFECT, facts.base_fluents[0], True, them()), NEC_ALL
        ),
        "the bound pronoun of a place in an event between entities": SentencePlan(
            subject, Predication(EFFECT, symmetric, True, them("SOMEPLACE")), NEC_ALL
        ),
        "the bound pronoun in a one-place event type": SentencePlan(
            event_phrase_of(EventTerm(facts.features["event_type1"][0], AGENT)),
            Predication(PRECONDITION, symmetric, True, them()),
            NEC_ALL,
        ),
        "a prepositional relative clause outside a profile statement": SentencePlan(
            good.subject, Predication("property", facts.features["property"][0]), "all"
        ),
        "the bound pronoun as the object of a verb": SentencePlan(
            NounPhrase(CLASS_NP, facts.categories[0], facts.categories[0]),
            Predication("event_type2", verb, True, them()),
            "all",
        ),
    }
    for what, plan in refused.items():
        with pytest.raises(GrammarError):
            check_plan(plan)
        assert what
    # a prepositional relative clause holds one predicate, as an object relative holds one verb
    assert len(pair_phrase_of(PairTerm(symmetric)).clause.predications) == 1
    assert pair_phrase_of(PairTerm(symmetric)).clause.predications[0].kind == PAIR_STATE
    # the truth tests: forms that no statement has are not valid
    invalid = [
        # SAME and SOME are read, never written
        Proposition(
            CLASS, EventTerm(verb, AGENT), Predicate(EFFECT, same, value=True), True, NEC_ALL
        ),
        Proposition(
            CLASS,
            EventTerm(verb, AGENT),
            Predicate(EFFECT, f"SOME.{symmetric}", value=True),
            True,
            NEC_ALL,
        ),
        # a pair relates two participants
        Proposition(
            CLASS,
            EventTerm(facts.features["event_type1"][0], AGENT),
            Predicate(PRECONDITION, symmetric, value=True),
            True,
            NEC_ALL,
        ),
        # a pair literal has a value, and no direction
        Proposition(CLASS, EventTerm(verb, AGENT), Predicate(EFFECT, symmetric), True, NEC_ALL),
        # a profile statement has a form
        Proposition(CLASS, PairTerm(symmetric), Predicate(PROFILE, symmetric), True, NEC_ALL),
        Proposition(
            CLASS,
            PairTerm(symmetric),
            Predicate(PROFILE, symmetric, form=SYMMETRIC),
            False,
            NEC_ALL,
        ),
        Proposition(
            CLASS, PairTerm(symmetric), Predicate(PROFILE, symmetric, form=SYMMETRIC), True, "all"
        ),
    ]
    for proposition in invalid:
        assert not truth.evaluate(proposition).valid, proposition
    # the rendering refuses a profile whose scope is not its restrictor's pair the other way
    with pytest.raises(RenderingError):
        proposition_of(
            parse_propositional(f"NEC(ALL({symmetric}(VAR.2, VAR.1), {symmetric}(VAR.2, VAR.1)))")
        )
    with pytest.raises(RenderingError):
        proposition_of(
            parse_propositional(
                f"NEC(ALL(EVENT(EVENTVAR.1, {verb}(VAR.1, VAR.2)), "
                f"AFTER(EVENTVAR.1, SOME.{symmetric}(VAR.1))))"
            )
        )
    assert issubclass(GrammarError, CorpusError)


# ---------------------------------------------------------------------------------------------
# Where the statements appear
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_documents_state_pair_literals_where_they_call_for_causal_statements(
    cases, runs, name
) -> None:
    """The causal statements with pair literals, and those about place event types, appear
    where documents already call for causal statements: in feature documents, about an event
    type between entities (its own statements) or about a fluent (the event types, place event
    types included, that set it or need it). Each is true by the oracle, reads back, and has
    the reading ``causal``. No document states a profile statement, and no causal test item is
    one of this stage's statements (stage c5b)."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    facts = corpus.planner.facts
    lexicon = corpus.planner.lexicon
    with_pair = about_place = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert not proposition.profile and proposition.predicate.kind != PROFILE
            assert "profile" not in sentence.readings
            if not proposition.causal:
                continue
            new = proposition.predicate.pair_literal or proposition.subject.place
            if not new:
                continue
            assert document.type == "encyclopedic_feature" and sentence.section == CAUSAL
            assert proposition in facts.causal_statements()
            if proposition.subject.place:
                # no document of this stage is about a place event type or a pair fluent
                assert document.topic == proposition.predicate.label
                assert document.topic in facts.base_fluents + facts.numeric_fluents
                about_place += 1
            else:
                assert document.topic == proposition.subject.event_type
            with_pair += proposition.predicate.pair_literal
            form = json.loads(json.dumps(sentence.logical_form))
            assert oracle.truth(form) is True, form
            assert sentence.readings[0] == "causal"
            read = interpret(
                sentence.sentence.tree,
                lexicon,
                sentence.sentence.referents,
                sentence.sentence.events,
                proposition.quantifier,
            )
            assert read == sentence.plan and read.proposition() == proposition
            assert proposition_of(parse_propositional(sentence.propositional)) == proposition
            if proposition.predicate.pair_literal:
                assert form["causal"]["fluent"] == proposition.predicate.label
    assert with_pair > 0 and about_place > 0
    stats = corpus.stats["causal"]
    assert stats["pair_literal_sentences"] == with_pair
    assert stats["place_event_sentences"] == about_place
    assert stats["statements"] == len(facts.causal_statements())
    assert sum(stats["pair_literal_statements"].values()) == sum(
        p.predicate.pair_literal for p in facts.causal_statements()
    )
    assert sum(stats["profile_statements"].values()) == len(facts.profile_statements())
    # the causal sets keep to the statements of stage c4c
    items = 0
    for test_set in corpus.test_sets:
        if test_set.level not in CAUSAL_LEVELS:
            continue
        for item in test_set.items:
            proposition = item.proposition
            assert not proposition.predicate.pair_literal and not proposition.subject.place
            items += 1
    assert items > 0


# ---------------------------------------------------------------------------------------------
# Jon's ruling WM.74: comparisons and the compared scalar's poles
# ---------------------------------------------------------------------------------------------


def _fits(oracle, phrase: NounPhrase, referent: str, z: float) -> bool:
    """Whether the noun and the modifiers of a mention fit a referent, by the oracle: its
    category, its PROPERTY and PART literals, and its poles against the noun's category."""

    def judge(predicate: dict, polarity: bool = True) -> bool:
        form = {
            "level": "instance",
            "polarity": polarity,
            "subject": {"instance": referent},
            "predicate": predicate,
            "z": z,
        }
        return oracle.truth(form) is True

    if not judge({"kind": "member", "label": phrase.noun}):
        return False
    for literal in phrase.restriction:
        feature = literal.feature
        if literal.pole:
            operator = "ABOVE" if feature.endswith(".HIGH") else "BELOW"
            said = {
                "kind": "comparison",
                "label": feature,
                "operator": operator,
                "class": phrase.noun,
            }
        else:
            kind = "part" if "PART." in feature else "property"
            said = {"kind": kind, "label": feature}
        if not judge(said, literal.positive):
            return False
    return True


@pytest.mark.parametrize("name", WORLDS)
def test_comparisons_use_the_compared_poles_across_comparison_classes_alone(
    cases, runs, name
) -> None:
    """WM.74: a comparison of two participants, or of two places, has the compared scalar's
    poles among its modifiers only when its two nouns name different categories (the poles
    are then relative to different comparison classes); no comparison is made whose two
    mentions nothing tells apart (the noun and the modifiers of one of them do not fit the
    other's referent, by the oracle); every comparison, and every pole of its mentions, is
    true by the oracle; and ``stats.yaml`` counts the comparisons with the compared poles and
    the comparisons that were not made."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    z = corpus.config.scalar_z
    across = Counter()
    found = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            if sentence.section not in (COMPARATIVE_SECTION, PLACE_COMPARATIVE):
                continue
            kind = PLACES if sentence.section == PLACE_COMPARATIVE else PARTICIPANTS
            found[kind] += 1
            plan = sentence.plan
            predication = plan.predication
            assert predication.kind == COMPARATIVE
            first, second = plan.subject, predication.object
            dimension = pole_dimension(predication.label)
            compared = [
                literal
                for phrase in (first, second)
                for literal in phrase.restriction
                if literal.pole and pole_dimension(literal.feature) == dimension
            ]
            if compared:
                # the poles are relative to different comparison classes
                assert first.noun is not None and second.noun is not None
                assert first.noun != second.noun, sentence.label
                # and the comparison is true by the oracle (a scalar is static)
                assert not sentence.proposition.timed
                assert oracle.truth(json.loads(json.dumps(sentence.logical_form))) is True
                across[kind] += 1
            if first.noun is None or second.noun is None:
                continue  # a pronoun is told apart by the rule that allowed it
            # each mention fits its own referent, poles included, and something tells the two
            # apart: one of the two descriptions does not fit the other's referent
            assert _fits(oracle, first, first.referent, z), sentence.label
            assert _fits(oracle, second, second.referent, z), sentence.label
            assert not (
                _fits(oracle, first, second.referent, z)
                and _fits(oracle, second, first.referent, z)
            ), sentence.label
    assert found[PARTICIPANTS] > 0
    stats = corpus.stats["comparisons"]
    assert stats["compared_poles_across_classes"] == {
        PARTICIPANTS: across[PARTICIPANTS],
        PLACES: across[PLACES],
    }
    skipped = stats["skipped_not_told_apart"]
    assert set(skipped) == {PARTICIPANTS, PLACES}
    assert skipped == corpus.planner.skipped_comparisons
    assert all(isinstance(count, int) and count >= 0 for count in skipped.values())


def test_the_two_mentions_of_a_comparison_under_wm74(cases) -> None:
    """The rule itself, on every ordered pair of the tiny world's places and of its entities,
    over many draws of the nouns: within one comparison class (the same noun) no mention has a
    pole of the compared scalar; the two are told apart exactly when one description does not
    fit the other's referent; and the tiny world's two places of a category, which differ by
    their one place scalar alone, are never told apart under one noun, while two places named
    by different nouns can carry the compared poles ("the cold desert is hotter than the hot
    meadow")."""
    case = cases("tiny")
    facts = case.facts()
    world = case.world
    from semantic_world.corpus.mentions import MentionRules

    rules = MentionRules(case.config(), facts, Streams(1))
    (scalar,) = facts.place_scalars
    rng = np.random.default_rng(5)
    same_noun = across = poles_across = apart_within = 0
    pools = (
        (world.places, {"places": tuple(world.places)}, scalar),
        (world.instances, {"cast": tuple(world.instances)}, facts.scalars[0]),
    )
    for labels, pool, dimension in pools:
        compared = (f"{dimension}.HIGH", f"{dimension}.LOW")
        for first, second in itertools.permutations(labels, 2):
            for _ in range(6):
                mentions = Mentions(rules, **pool)
                # both were mentioned before, as the two arguments of a comparison are
                mentions.noun_phrase(rng, first)
                mentions.noun_phrase(rng, second)
                mentions.end_sentence(None)
                a, b, apart = mentions.comparison_phrases(rng, first, second, dimension)
                assert a.referent == first and b.referent == second
                assert a.determiner == "the" and b.determiner == "the"
                used = [
                    literal.feature
                    for phrase in (a, b)
                    for literal in phrase.restriction
                    if literal.feature in compared
                ]
                fits_both = bool(rules.matches(a, [second])) and bool(rules.matches(b, [first]))
                assert apart == (not fits_both)
                assert rules.matches(a, [first]) and rules.matches(b, [second])
                if a.noun == b.noun:
                    same_noun += 1
                    assert not used, "within one comparison class the compared poles stay out"
                    apart_within += apart
                    if world.is_place(first) and (
                        world.place_leaf_of(first) == world.place_leaf_of(second)
                    ):
                        # the tiny world's two places of a category differ by their scalar
                        # alone: "the place is hotter than the place" is not made
                        assert not apart
                else:
                    across += 1
                    assert apart, "different nouns tell the two apart"
                    poles_across += bool(used)
    assert same_noun > 0 and across > 0
    assert poles_across > 0, "across comparison classes the compared poles are allowed"
    assert apart_within > 0, "another feature can tell two mentions of one noun apart"
