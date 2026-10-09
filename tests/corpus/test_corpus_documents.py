"""Stage 5 acceptance tests: documents.

The propositional rendering of every sentence parses back to its logical form, and is made from
the JSON logical form. Every distinguishing definite mention picks out exactly one participant
of its scene. Every pronoun's chain matches the referent recorded in its logical form. With
shuffle 0, encyclopedic documents follow the template order exactly. Every sentence is true, by
the engine's truth tests and by the independent oracle, and every tree reads back as its plan.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pytest
from corpus_support import corpus_config

from semantic_world.corpus import (
    ConfigError,
    CorpusError,
    Planner,
    Proposition,
    interpret,
    load_world,
    parse_propositional,
    proposition_of,
    propositional,
)
from semantic_world.corpus.config import CONCEPT_TYPES
from semantic_world.corpus.grammar import CLASS_NP, INSTANCE_NP, check_plan
from semantic_world.corpus.histories import scene_events
from semantic_world.corpus.mentions import clause_propositions
from semantic_world.corpus.planner import (
    CHARACTERISTIC,
    CONTRAST,
    DEFINING,
    DESCRIPTION,
    EVENT_SECTION,
    MEMBERSHIP,
    RARER,
    RELATION,
    RULE,
    TEMPLATE,
    content_words,
    reading_counts,
)
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    EVENT,
    INSTANCE,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NO,
    SCALAR,
    SOME,
    VERB,
    CategoryTerm,
)
from semantic_world.corpus.realize import leaves, token_parts
from semantic_world.corpus.renderings import formula

WORLDS = ("tiny", "default", "deep", "still")
COUNTS = {"tiny": 120, "default": 300, "deep": 150, "still": 120}
RICH = {
    "mention": {"relative_clauses": {"rate": 0.3, "max_depth": 2, "object_share": 0.4}},
    "propositions": {"restriction_rate": 0.3},
    "documents": {"sibling_contrast_rate": 0.6},
}
"""More relative clauses, restrictions, and contrasts than the defaults, so that the tests see
many of each."""
ENCYCLOPEDIC = ("encyclopedic_category", "encyclopedic_feature")
NARRATIVES = ("entity", "situational")


def merged(*sections: dict) -> dict:
    """Corpus settings laid over each other, section by section."""
    result: dict = {}
    for layer in sections:
        for key, value in layer.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merged(result[key], value)
            else:
                result[key] = value
    return result


@pytest.fixture(scope="module")
def corpora(cases):
    """The documents of a world under given settings, made once: the planner and its
    documents."""
    made: dict = {}

    def of(name: str, count: int | None = None, **sections):
        key = (name, count, json.dumps(sections, sort_keys=True))
        if key not in made:
            case = cases(name)
            planner = Planner(case.config(**sections), case.result)
            made[key] = (planner, planner.generate(count or COUNTS[name]))
        return made[key]

    return of


def sentences_of(documents, *types: str):
    return [(d, s) for d in documents if not types or d.type in types for s in d.sentences]


def mentions_of(form: dict) -> list[dict]:
    """The mentions of a JSON logical form about instances, in the order of the plan: the
    subject, the noun phrases of its relative clause, then the patient and its own."""
    found: list[dict] = []

    def visit(mention: dict) -> None:
        found.append(mention)
        for clause in mention.get("clauses", ()):
            other = clause.get("agent") or clause.get("patient")
            if other is not None:
                visit(other)

    visit(form["subject"])
    if "patient" in form["predicate"]:
        visit(form["predicate"]["patient"])
    return found


# ---------------------------------------------------------------------------------------------
# Acceptance: every sentence is true, and reads back
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_sentence_is_true_and_reads_back(cases, corpora, name) -> None:
    case = cases(name)
    planner, documents = corpora(name, **RICH)
    oracle = case.oracle(**RICH)
    levels: Counter = Counter()
    clauses = 0
    for document in documents:
        assert document.sentences
        for sentence in document.sentences:
            plan, proposition = sentence.plan, sentence.proposition
            check_plan(plan)
            # the plan says the proposition, and the proposition is true
            assert plan.proposition() == proposition
            assert planner.truth.is_true(proposition), sentence.label
            assert proposition.grounding is not None and proposition.id.startswith("PROP.")
            if proposition.level != EVENT:
                # by the independent recomputation from the world run's files too
                assert oracle.truth(proposition.to_json()) is True, proposition.to_json()
            else:
                assert proposition.grounding["able"] and proposition.grounding["legal"]
            if proposition.level != CLASS:
                for said in clause_propositions(plan):
                    assert planner.truth.is_true(said), (sentence.label, said)
                    clauses += 1
            # the tree's leaves are the tokens, and the tree reads back as the plan
            tree = sentence.sentence.tree
            assert tuple(leaves(tree)) == sentence.sentence.tokens
            record = sentence.sentence
            read = interpret(
                tree, planner.lexicon, record.referents, record.events, plan.quantifier
            )
            assert read == plan
            levels[proposition.level] += 1
    assert set(levels) == {CLASS, INSTANCE, EVENT} and min(levels.values()) > 30
    assert clauses > 20


@pytest.mark.parametrize("name", WORLDS)
@pytest.mark.parametrize("mode", ["local", "instance"])
def test_the_propositional_rendering_parses_back_to_the_logical_form(corpora, name, mode) -> None:
    planner, documents = corpora(
        name, **merged(RICH, {"renderings": {"propositional": {"referents": mode}}})
    )
    quantified = existential = events = capacities = 0
    for document, sentence in sentences_of(documents):
        text = sentence.propositional
        # the rendering is made from the JSON logical form alone, as the files hold it
        form = json.loads(json.dumps(sentence.logical_form))
        assert propositional(form, mode) == text
        # it parses back to the formula of the logical form
        parsed = parse_propositional(text)
        assert parsed == formula(form, mode)
        # and to the proposition, which the JSON logical form gives too: the two agree
        names = document.referents if mode == "local" else None
        assert proposition_of(parsed, names) == sentence.proposition
        assert Proposition.from_json(form) == sentence.proposition
        assert form["id"] == sentence.proposition.id
        assert form["grounding"] == sentence.proposition.grounding
        # nothing of the surface is in it: no determiner, no pronoun, no word
        assert " THE(" not in text and "LEXEME." not in text
        if sentence.proposition.level == CLASS:
            if sentence.proposition.predicate.kind == SCALAR:
                assert text.removeprefix("NOT ").startswith("SCALARDIM.")
            else:
                assert text.split("(")[0] in ("NEC", "ALL", "MOST", "SOME", "NO")
            assert not re.search(r"\bREF\.\d|\bINSTANCE\.\d", text)  # no referent, no instance
            quantified += 1
            existential += "EXISTS(" in text
        else:
            assert "VAR." not in text
            label = "REF." if mode == "local" else "INSTANCE."
            assert f"({label}" in text
            events += "EVENT(" in text
            capacities += "ABLE(" in text
            # the tense and the aspect of every event are written
            for part in parsed:
                if type(part).__name__ == "Report":
                    assert part.tense == "past" and part.aspect in ("simple", "progressive")
    assert min(quantified, events) > 50 and capacities > 20 and existential > 5


def test_the_propositional_rendering_never_depends_on_the_grammar(corpora) -> None:
    grammars = {
        "plain": {},
        "flipped": {
            "word_order": {
                "clause": "OSV",
                "determiner": "after",
                "adjective": "after",
                "relative_clause": "before",
                "adposition": "postposition",
                "auxiliary": "after",
                "negation": "before_auxiliary",
            },
            "adjective_order": {"fixed": False},
        },
        "marked": {
            "morphology": {
                "number": {"enabled": True, "verb_marks": "singular"},
                "tense": {"enabled": True},
                "aspect": {"enabled": True},
            },
            "can_rate": {"class": 0.1, "instance": 0.2},
        },
    }
    made = {
        name: corpora("default", 150, **merged(RICH, {"grammar": grammar}))[1]
        for name, grammar in grammars.items()
    }
    base = made["plain"]

    def said(documents):
        return [
            (d.label, d.type, d.topic, s.label, s.section, s.propositional, s.logical_form)
            for d in documents
            for s in d.sentences
        ]

    for name in ("flipped", "marked"):
        # the same documents, the same sentences in the same order, and the same logical forms
        assert said(made[name]) == said(base)
        assert [s.plan for _, s in sentences_of(made[name])] == [
            s.plan for _, s in sentences_of(base)
        ]
        # and other words
        assert [s.sentence.tokens for _, s in sentences_of(made[name])] != [
            s.sentence.tokens for _, s in sentences_of(base)
        ]


# ---------------------------------------------------------------------------------------------
# Acceptance: distinguishing mentions
# ---------------------------------------------------------------------------------------------


class Described:
    """Which instances a noun phrase describes, worked out from the world's own tables."""

    def __init__(self, world, z: float = 1.0) -> None:
        self.world = world
        self.z = z
        self.labels = list(world.instances)
        self.row = {label: i for i, label in enumerate(self.labels)}

    @staticmethod
    def path(instance: str) -> list[str]:
        prefix, *numbers = instance.rsplit(".", 1)[0].replace("INSTANCE", "CATEGORY", 1).split(".")
        return [".".join([prefix, *numbers[: k + 1]]) for k in range(len(numbers))]

    def fits(self, instance: str, phrase) -> bool:
        if phrase.noun not in self.path(instance):
            return False
        world = self.world
        row = self.row[instance]
        for literal in phrase.restriction:
            if literal.feature.startswith("SCALARDIM."):
                scalar, side = literal.feature.rsplit(".", 1)
                column = world.scalar_values[:, world.scalars.index(scalar)]
                below = [self.row[i] for i in self.labels if phrase.noun in self.path(i)]
                mean, sd = column[below].mean(), column[below].std()
                high = sd > 0 and column[row] >= mean + self.z * sd
                low = sd > 0 and column[row] <= mean - self.z * sd
                if not (high if side == "HIGH" else low):
                    return False
            else:
                value = bool(world.column(literal.feature)[row])
                if value != literal.positive:
                    return False
        return True


@pytest.mark.parametrize("name", WORLDS)
def test_every_distinguishing_mention_picks_out_one_participant(cases, corpora, name) -> None:
    case = cases(name)
    _, documents = corpora(name, **RICH)
    described = Described(case.world)
    flags: Counter = Counter()
    needed = 0
    for document in documents:
        cast = list(dict.fromkeys(p for scene in document.scenes for p in scene.participants))
        if document.type in ENCYCLOPEDIC:
            assert not cast
        for sentence in document.sentences:
            phrases = sentence.sentence.phrases
            assert len(phrases) == len(sentence.distinguished)
            for phrase, flag in zip(phrases, sentence.distinguished, strict=True):
                definite = (
                    phrase.kind == INSTANCE_NP and not phrase.pronoun and phrase.determiner == "the"
                )
                if not definite:
                    assert flag is None
                    continue
                # what the noun phrase says is true of its referent
                assert described.fits(phrase.referent, phrase)
                fitting = [p for p in cast if described.fits(p, phrase)]
                assert flag == (fitting == [phrase.referent])
                flags[flag] += 1
                if flag:
                    # exactly one participant of each of the document's scenes that it is in
                    for scene in document.scenes:
                        if phrase.referent in scene.participants:
                            inside = [p for p in scene.participants if described.fits(p, phrase)]
                            assert inside == [phrase.referent]
                # in a situational document, modifiers do referential work only: a mention has
                # them when its noun alone fits another participant
                bare = [p for p in cast if phrase.noun in described.path(p)]
                if document.type == "situational":
                    assert bool(phrase.restriction) == (len(bare) > 1) or not flag
                needed += len(bare) > 1
    assert flags[True] > 100 and needed > 30
    # nearly every definite mention can be told apart. An entity narrative has 2 to 5 scenes,
    # so its cast is large, and the smallest worlds have few features to tell instances apart
    # (the still world: 20 of 189 definite mentions)
    assert flags[False] <= 0.15 * flags[True]


def test_first_mentions_in_situational_documents_take_no_modifier(corpora) -> None:
    _, documents = corpora("default", **RICH)
    plain = with_modifier = 0
    for document in documents:
        if document.type not in NARRATIVES:
            continue
        for sentence in document.sentences:
            for phrase in sentence.plan.noun_phrases():
                if phrase.determiner != "a":
                    continue
                if document.type == "situational":
                    assert not phrase.restriction
                else:
                    with_modifier += bool(phrase.restriction)
                    plain += not phrase.restriction
                    assert len(phrase.restriction) <= 1
    # in an entity document, a mention takes a modifier at mention.modifier_rate (0.3)
    assert 0.15 < with_modifier / (with_modifier + plain) < 0.45


# ---------------------------------------------------------------------------------------------
# Acceptance: pronouns, first and later mentions, and referent labels
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_pronouns_and_chains(corpora, name) -> None:
    _, documents = corpora(name, **RICH)
    pronouns = eligible = taken = later = 0
    for document in documents:
        if document.type in ENCYCLOPEDIC:
            assert document.referents == {}
            for sentence in document.sentences:
                assert set(sentence.coreference) == {None}
            continue
        labels = {instance: label for label, instance in document.referents.items()}
        # the referents are numbered in the order of first mention
        assert list(document.referents) == [f"REF.{n}" for n in range(1, len(labels) + 1)]
        seen: list[str] = []
        previous: tuple[str, set[str]] | None = None
        for sentence in document.sentences:
            plan = sentence.plan
            phrases = plan.noun_phrases()
            mentions = mentions_of(sentence.logical_form)
            assert len(mentions) == len(phrases)
            main = [plan.subject] + ([plan.predication.object] if plan.predication.object else [])
            for phrase, mention in zip(phrases, mentions, strict=True):
                instance = phrase.referent
                first = instance not in seen
                if first:
                    seen.append(instance)
                # the logical form records the referent, and the noun, or none for a pronoun
                assert mention["instance"] == instance
                assert mention["referent"] == labels[instance] == f"REF.{seen.index(instance) + 1}"
                assert mention["noun"] == phrase.noun
                if phrase.pronoun:
                    pronouns += 1
                    # a pronoun stands in the main clause, for a referent of the sentence before
                    assert any(phrase is m for m in main)
                    if phrase.noun is None and not first:
                        assert previous is not None
                        subject, mentioned = previous
                        assert instance in mentioned
                        assert mentioned == {instance} or subject == instance
                elif first:
                    assert phrase.determiner == "a"
                else:
                    assert phrase.determiner == "the"
                    later += 1
            # how often a later mention that could be a pronoun is one
            for phrase in main:
                could = (
                    previous is not None
                    and phrase.referent in previous[1]
                    and (previous[1] == {phrase.referent} or previous[0] == phrase.referent)
                    and not (phrase is plan.subject and plan.predication.kind == SCALAR)
                )
                if could and [p.referent for p in phrases].index(phrase.referent) == phrases.index(
                    phrase
                ):
                    eligible += 1
                    taken += phrase.pronoun
            previous = (plan.subject.referent, {p.referent for p in phrases})
            # every pronoun's chain matches the referent recorded in its logical form
            record = sentence.sentence
            assert len(sentence.coreference) == len(record.referents) == len(record.phrases)
            for chain, (referent, noun), phrase in zip(
                sentence.coreference, record.referents, record.phrases, strict=True
            ):
                assert chain == labels[referent] and document.referents[chain] == referent
                assert phrase.referent == referent and phrase.noun == noun
                if noun is None:
                    recorded = [m for m in mentions if m["noun"] is None]
                    assert chain in [m["referent"] for m in recorded]
                    assert {m["instance"] for m in recorded if m["referent"] == chain} == {referent}
        assert seen == list(labels)
    assert pronouns > 30 and later > 100
    # a later mention becomes "it" at mention.pronoun_rate (0.5), when it can
    assert eligible > 60 and 0.35 < taken / eligible < 0.65


def test_the_pronoun_rate(corpora) -> None:
    def pronouns(rate: float) -> int:
        _, documents = corpora("default", 150, mention={"pronoun_rate": rate})
        return sum(
            phrase.pronoun for _, s in sentences_of(documents) for phrase in s.plan.noun_phrases()
        )

    assert pronouns(0.0) == 0
    assert pronouns(1.0) > 2 * pronouns(0.5) * 0.7 > 0


@pytest.mark.parametrize("name", ["default", "deep"])
def test_noun_levels_apply_to_instances_only(cases, corpora, name) -> None:
    case = cases(name)
    planner, documents = corpora(name, **RICH)
    depth = max(len(c.path) for c in case.world.category.values())
    levels: Counter = Counter()
    for _, sentence in sentences_of(documents):
        for phrase in sentence.plan.noun_phrases():
            if phrase.kind == CLASS_NP:
                # a class-level noun phrase is always named by its own noun
                assert phrase.noun == phrase.referent
            elif not phrase.pronoun:
                # an instance is named by its leaf, or by a category above it
                path = Described.path(phrase.referent)
                assert phrase.noun in path
                levels[path.index(phrase.noun) + 1] += 1
    # every level is used, and the leaf level most
    assert set(levels) == set(range(1, depth + 1))
    assert max(levels, key=levels.get) == depth
    # with all the weight on the top level, instances are named by their top category, apart
    # from the mentions that need another noun: to tell a referent apart, to name the comparison
    # class of a pole, or to avoid "the animal is an animal"
    top = {"mention": {"level_weights": {"schedule": "list", "values": [1] + [0] * (depth - 1)}}}
    _, documents = corpora(name, 100, **top)
    named: Counter = Counter()
    for _, sentence in sentences_of(documents, *NARRATIVES):
        for phrase in sentence.plan.noun_phrases():
            if phrase.pronoun:
                continue
            level = Described.path(phrase.referent).index(phrase.noun) + 1
            named[level] += 1
            if phrase.determiner == "a" and sentence.plan.predication.kind not in (SCALAR, MEMBER):
                assert level == 1
    assert named[1] > 0.6 * sum(named.values())


# ---------------------------------------------------------------------------------------------
# Acceptance: the template order
# ---------------------------------------------------------------------------------------------


def inversions(document) -> int:
    """How many pairs of sentences of an encyclopedic document stand against the template."""
    ranks = [TEMPLATE.index(s.section) for s in document.sentences if s.section != CONTRAST]
    return sum(a > b for i, a in enumerate(ranks) for b in ranks[i + 1 :])


def test_with_shuffle_0_encyclopedic_documents_follow_the_template(corpora) -> None:
    settings = merged(RICH, {"documents": {"shuffle": 0.0, "sibling_contrast_rate": 1.0}})
    _, documents = corpora("default", **settings)
    sections: Counter = Counter()
    checked = 0
    for document in documents:
        if document.type not in ENCYCLOPEDIC:
            continue
        checked += 1
        assert inversions(document) == 0
        ranks = [TEMPLATE.index(s.section) for s in document.sentences if s.section != CONTRAST]
        assert ranks == sorted(ranks)
        for index, sentence in enumerate(document.sentences):
            sections[sentence.section] += 1
            proposition = sentence.proposition
            kind = proposition.predicate.kind
            # what each section holds
            if sentence.section == MEMBERSHIP:
                assert kind == MEMBER
            elif sentence.section == RELATION:
                assert kind == VERB
            elif sentence.section == RULE:
                assert proposition.rule is not None and proposition.subject.category == "THING"
            elif sentence.section == CONTRAST:
                assert index > 0 and document.sentences[index - 1].section != CONTRAST
            else:
                assert sentence.section in (DEFINING, CHARACTERISTIC, RARER)
                assert kind not in (MEMBER, VERB) and proposition.rule is None
                if proposition.quantifier in (ALL, NO):
                    assert sentence.section == DEFINING
                if proposition.quantifier == MOST or kind == SCALAR:
                    assert sentence.section == CHARACTERISTIC
                if proposition.quantifier == SOME:
                    assert sentence.section == RARER
    assert checked > 100
    assert set(sections) == {*TEMPLATE, CONTRAST}


def test_shuffle_moves_from_the_template_to_a_random_order(corpora) -> None:
    def disorder(shuffle: float) -> float:
        _, documents = corpora("default", 200, documents={"shuffle": shuffle})
        counts = [inversions(d) for d in documents if d.type in ENCYCLOPEDIC]
        return float(np.mean(counts))

    strict, loose, free = disorder(0.0), disorder(0.3), disorder(1.0)
    assert strict == 0 < loose < free
    # shuffling changes the order of a document, and not what it says
    _, ordered = corpora("default", 200, documents={"shuffle": 0.0})
    _, shuffled = corpora("default", 200, documents={"shuffle": 1.0})
    for a, b in zip(ordered, shuffled, strict=True):
        assert (a.type, a.topic) == (b.type, b.topic)
        if a.type in ENCYCLOPEDIC:
            assert sorted(s.propositional for s in a.sentences) == sorted(
                s.propositional for s in b.sentences
            )


# ---------------------------------------------------------------------------------------------
# The four document types
# ---------------------------------------------------------------------------------------------


def test_the_document_mix_and_lengths(cases, corpora) -> None:
    planner, documents = corpora("default", 600)
    config = planner.config
    assert [d.label for d in documents] == [f"DOC.{n}" for n in range(1, 601)]
    types = Counter(d.type for d in documents)
    for kind, weight in config.documents.mix.items():
        assert abs(types[kind] / 600 - weight) < 0.07
    for document in documents:
        limits = config.documents.sentences[document.type]
        assert 1 <= len(document.sentences) <= limits.max
        assert [s.label for s in document.sentences] == [
            f"{document.label}.SENT.{k}" for k in range(1, len(document.sentences) + 1)
        ]
        if document.type in ENCYCLOPEDIC:
            # the default world has enough to say about every topic
            assert len(document.sentences) >= limits.min
    # one type alone
    only = {"documents": {"mix": {"situational": 1.0}}}
    _, documents = corpora("tiny", 30, **only)
    assert {d.type for d in documents} == {"situational"}
    case = cases("tiny")
    with pytest.raises(ConfigError, match="at least one weight must be positive"):
        nothing = dict.fromkeys((*ENCYCLOPEDIC, *NARRATIVES), 0)
        case.config(documents={"mix": nothing})
    assert Planner(case.config(), case.result).generate(0) == []


def test_category_documents(corpora) -> None:
    planner, documents = corpora("default", **RICH)
    facts, truth = planner.facts, planner.truth
    topics: Counter = Counter()
    kinds: Counter = Counter()
    for document in documents:
        if document.type != "encyclopedic_category":
            continue
        topic = document.topic
        topics[facts.level[topic]] += 1
        children = list(planner.world.category[topic].children)
        stated = set()
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert proposition.level == CLASS
            subject, predicate = proposition.subject, proposition.predicate
            key = (subject, predicate, proposition.negative)
            assert key not in stated  # a document says a thing once
            stated.add(key)
            if sentence.section == CONTRAST:
                continue
            if predicate.kind == MEMBER:
                # the topic's ancestors, its children, or a category it shares nothing with
                if subject.category == topic:
                    assert predicate.label in truth.ancestors(topic) or proposition.negative
                    kinds["above" if not proposition.negative else "apart"] += 1
                else:
                    assert subject.category in children and predicate.label == topic
                    kinds["below"] += 1
            elif predicate.kind == VERB:
                assert topic in (subject.category, predicate.patient.category)
                kinds["agent" if subject.category == topic else "patient"] += 1
            else:
                assert subject.category == topic or subject.category in children
                kinds["topic" if subject.category == topic else "subcategory"] += 1
    assert set(kinds) == {"above", "apart", "below", "agent", "patient", "topic", "subcategory"}
    # topics come from every level, by documents.topic_level_weights (1 to 2 over three levels)
    assert set(topics) == {1, 2, 3}


def test_feature_documents(corpora) -> None:
    planner, documents = corpora("default", **RICH)
    facts = planner.facts
    topics: Counter = Counter()
    rules = sentences = 0
    with_rules = set()
    for document in documents:
        if document.type != "encyclopedic_feature":
            continue
        topic = document.topic
        kind = "verb" if topic in facts.verbs else planner.world.feature_kind[topic]
        topics[kind] += 1
        for sentence in document.sentences:
            proposition = sentence.proposition
            assert proposition.level == CLASS
            if proposition.rule is not None:
                # a sufficient condition for the topic, or something the topic makes possible
                restriction = [x.feature for x in proposition.subject.restriction]
                assert topic == proposition.predicate.label or topic in restriction
                rules += 1
                with_rules.add(document.label)
            else:
                # which categories have the feature, and which lack it
                assert proposition.predicate.label == topic
                assert proposition.subject.category in facts.categories
            sentences += 1
    assert set(topics) == {"is", "has", "can", "verb"}
    assert rules > 20 and 0.05 < rules / sentences < 0.4
    # with the rate at 0, no document states a rule, and at 1 the rules come first
    _, none = corpora("default", 150, propositions={"rule_statement_rate": 0.0})
    assert not [s for _, s in sentences_of(none) if s.proposition.rule is not None]


def test_narratives(cases, corpora) -> None:
    planner, documents = corpora("default", **RICH)
    config = planner.config
    scene_numbers = []
    described = recurring = with_clauses = 0
    for document in documents:
        if document.type in ENCYCLOPEDIC:
            assert document.scenes == ()
            continue
        scene_numbers += [int(scene.label.split(".")[1]) for scene in document.scenes]
        reported = [s.proposition.event for s in document.sentences if s.section == EVENT_SECTION]
        assert reported and len(set(reported)) == len(reported)
        order = [(int(label.split(".")[1]), int(label.rsplit(".", 1)[1])) for label in reported]
        assert order == sorted(order)  # scene by scene, and in time order
        events = {e.label: e for scene in document.scenes for e in scene_events(scene)}
        if document.type == "entity":
            limits = config.entity_scenes
            assert limits.min <= len(document.scenes) <= limits.max
            assert {scene.seed for scene in document.scenes} == {document.topic}
            # the events that involve the instance
            assert all(events[label].involves(document.topic) for label in reported)
        else:
            (scene,) = document.scenes
            assert document.topic == scene.label
            # every event of the scene, as far as the document's length goes
            assert reported == [e.label for e in scene_events(scene)][: len(reported)]
        for index, sentence in enumerate(document.sentences):
            proposition = sentence.proposition
            if sentence.section == EVENT_SECTION:
                event = events[proposition.event]
                assert proposition.level == EVENT and proposition.subject == event.agent
                assert proposition.predicate.patient == event.patient
                # the event carries no aspect: the report chooses one
                assert proposition.tense == "past" and proposition.aspect in (
                    "simple",
                    "progressive",
                )
                # the verb is the event's own, or a verb category above it
                assert proposition.predicate.label in planner.truth.verb_names(event.type)
                # a relative clause reports an earlier event of the same scene
                clauses = clause_propositions(sentence.plan)
                for said in clauses:
                    assert said.scene == proposition.scene
                    assert int(said.event.rsplit(".", 1)[1]) < int(
                        proposition.event.rsplit(".", 1)[1]
                    )
                # and never an event with the verb, the agent, and the patient of the sentence's
                # own event or of another clause's: "the dog chased the cat that the dog chased"
                happened = [event.key] + [events[said.event].key for said in clauses]
                assert len(set(happened)) == len(happened), sentence.label
                recurring += sum(e.key == event.key and e is not event for e in events.values())
                with_clauses += bool(clauses)
            else:
                assert sentence.section == DESCRIPTION and proposition.level == INSTANCE
                # a description follows an event sentence, and is about one of its participants
                before = document.sentences[index - 1]
                assert index > 0 and before.section == EVENT_SECTION
                event = events[before.proposition.event]
                if document.type == "entity":
                    assert proposition.subject == document.topic
                else:
                    assert event.involves(proposition.subject)
                if isinstance(proposition.predicate.patient, str):
                    cast = {p for scene in document.scenes for p in scene.participants}
                    assert proposition.predicate.patient in cast
                described += 1
    # the scenes are numbered across the corpus, in the order of the documents
    assert scene_numbers == list(range(1, len(scene_numbers) + 1))
    assert [scene.label for scene in planner.scenes] == [f"SCENE.{n}" for n in scene_numbers]
    assert described > 50
    # the rule about repeated events has work to do: events recur, and sentences have clauses
    assert recurring > 20 and with_clauses > 20
    # events are named at every level of the verb tree
    names = Counter(
        len(s.proposition.predicate.label.split(".")) - 1  # EVENTTYPE2.1 and EVENTTYPE2.1.2
        for _, s in sentences_of(documents, *NARRATIVES)
        if s.section == EVENT_SECTION and s.proposition.predicate.kind == VERB
    )
    assert set(names) == {1, 2}


def test_the_description_rate(corpora) -> None:
    def share(rate: float) -> float:
        _, documents = corpora("default", 200, documents={"instance_description_rate": rate})
        sections = Counter(s.section for _, s in sentences_of(documents, *NARRATIVES))
        return sections[DESCRIPTION] / max(1, sections[EVENT_SECTION])

    assert share(0.0) == 0.0
    assert 0.1 < share(0.2) < 0.3 < share(0.8) < 0.9


# ---------------------------------------------------------------------------------------------
# The share of relation facts in category documents
# ---------------------------------------------------------------------------------------------


def test_the_relation_fact_share(corpora, world_files) -> None:
    """``documents.relation_fact_share`` sets how often a sentence of a category document draws
    a relation fact. The default, null, keeps the chance that ``content_kind_weights`` gives
    the kinds of content: here equal. The setting changes the category documents alone."""

    def made(**documents):
        documents = {"content_kind_weights": "equal", **documents}
        planner, found = corpora("default", 300, documents=documents)
        drawn = [
            s.section
            for d in found
            if d.type == "encyclopedic_category"
            for s in d.sentences
            if s.section != CONTRAST
        ]
        others = [
            [s.sentence.tokens for s in d.sentences]
            for d in found
            if d.type != "encyclopedic_category"
        ]
        return sum(section == RELATION for section in drawn) / len(drawn), others, found

    default, others, documents = made()
    # one kind among four for a topic with subcategories, and among three for a leaf; a little
    # more in the documents, because the membership facts of a topic run out first
    assert 0.25 < default < 0.40
    explicit = made(relation_fact_share=None)[2]
    assert [d.to_json() for d in explicit] == [d.to_json() for d in documents]
    # 0 turns relation facts off in category documents, and only there
    none, same_others, without = made(relation_fact_share=0.0)
    assert none == 0.0 and same_others == others
    feature = [s.section for d in without if d.type == "encyclopedic_feature" for s in d.sentences]
    assert feature.count(RELATION) > 40
    assert [(d.label, d.type, d.topic) for d in without] == [
        (d.label, d.type, d.topic) for d in documents
    ]
    # a share is the probability that a sentence draws a relation fact
    assert 0.52 < made(relation_fact_share=0.6)[0] < 0.68
    assert made(relation_fact_share=1.0)[0] == 1.0
    assert made(relation_fact_share=0.6)[1] == others
    # without verbs, a category document has no relation fact to draw
    planner = Planner(corpus_config(world_files["plain"], documents={"relation_fact_share": 0.9}))
    plain = planner.generate(40)
    assert all(s.section != RELATION for d in plain for s in d.sentences)
    assert sum(d.type == "encyclopedic_category" for d in plain) > 5


# ---------------------------------------------------------------------------------------------
# Sibling contrasts, restrictions, and class-level relative clauses
# ---------------------------------------------------------------------------------------------


def test_sibling_contrasts(corpora) -> None:
    def contrasts(rate: float):
        planner, documents = corpora("default", 300, documents={"sibling_contrast_rate": rate})
        found = []
        for document in documents:
            for index, sentence in enumerate(document.sentences):
                if sentence.section == CONTRAST:
                    found.append((planner, document, document.sentences[index - 1], sentence))
        return found

    assert contrasts(0.0) == []
    every = contrasts(1.0)
    # a contrast needs a strong fact about a sibling, and in the default language the strong
    # facts are the nec universals and most: 17 contrasts in 300 documents
    assert len(contrasts(0.2)) < len(every) and len(every) > 10
    for planner, document, fact, contrast in every:
        assert document.type == "encyclopedic_category"
        first, second = fact.proposition, contrast.proposition
        # the matching fact: the same predicate, about a sibling of the topic, and it differs
        assert first.predicate.kind == second.predicate.kind
        assert first.predicate.label == second.predicate.label
        assert first.negative != second.negative
        if first.subject == second.subject:
            ours, theirs = first.predicate.patient, second.predicate.patient
        else:
            ours, theirs = first.subject, second.subject
            assert first.predicate.patient == second.predicate.patient
        assert ours == CategoryTerm(document.topic) and theirs != ours
        assert theirs.restriction == () and theirs.clauses == ()
        category = planner.world.category
        assert category[ours.category].parent == category[theirs.category].parent
        # both are strong facts: a universal, most, or a scalar pole
        for proposition in (first, second):
            strong = planner.facts.class_fact(
                proposition.subject, proposition.predicate, proposition.negative
            )
            assert strong.predicate.kind == SCALAR or strong.quantifier in (
                NEC_ALL, ALL, NEC_NO, NO, MOST
            )  # fmt: skip


def test_restricted_subjects_come_from_the_proposition_layer(cases, corpora) -> None:
    def restricted(**sections):
        planner, documents = corpora("default", 200, **sections)
        found = []
        for _, sentence in sentences_of(documents, *ENCYCLOPEDIC):
            proposition = sentence.proposition
            if proposition.rule is not None:
                continue
            for term in (proposition.subject, proposition.predicate.patient):
                if term is not None and term.restriction:
                    found.append((planner, proposition, term))
        return found

    off = {"propositions": {"restriction_rate": 0.0}}
    assert restricted(**off) == []
    some = restricted()
    many = restricted(propositions={"restriction_rate": 1.0})
    assert 0 < len(some) < len(many) / 4
    polarities: Counter = Counter()
    for planner, proposition, term in many:
        # one literal, which some members of the category satisfy, and not all
        (literal,) = term.restriction
        inside = len(planner.truth.members(CategoryTerm(term.category, term.restriction)))
        assert 0 < inside < len(planner.truth.members(CategoryTerm(term.category)))
        # the sentence does not say what its subject already says
        assert proposition.predicate.label != literal.feature
        polarities[literal.positive] += 1
    assert polarities[True] > polarities[False] > 0
    # instance mentions take their modifiers from the mention layer, and class-level noun
    # phrases never do: with the restriction rate at 0, no class-level noun phrase has one
    _, documents = corpora("default", 200, **merged(off, {"mention": {"modifier_rate": 1.0}}))
    for _, sentence in sentences_of(documents):
        for phrase in sentence.plan.noun_phrases():
            if phrase.kind == CLASS_NP and sentence.proposition.rule is None:
                assert not phrase.restriction


def test_class_level_relative_clauses(cases, corpora) -> None:
    case = cases("default")
    settings = {"mention": {"relative_clauses": {"rate": 0.5, "max_depth": 2}}}
    planner, documents = corpora("default", 250, **settings)
    oracle = case.oracle(**settings)
    kinds: Counter = Counter()
    deep = on_patient = 0
    for _, sentence in sentences_of(documents, *ENCYCLOPEDIC):
        proposition = sentence.proposition
        subject, patient = proposition.subject, proposition.predicate.patient
        for term in (subject, patient):
            if term is None or not term.clauses:
                continue
            (clause,) = term.clauses
            kinds[
                (clause.kind, "agent" if clause.agent else "patient" if clause.patient else "")
            ] += 1
            # the clause is restrictive: some members satisfy it, and not all
            inside = len(planner.truth.members(term))
            assert 0 < inside < len(planner.truth.members(term.plain))
            deep += clause.other is not None and bool(clause.other.clauses)
            on_patient += term is patient
            assert sentence.plan.depth() >= 1
            # a relative clause is a subject relative or an object relative in the tree
            assert "RC" in json.dumps(sentence.sentence.tree)
        if subject.clauses:
            # a subject with a relative clause takes the extensional universals, never nec
            assert proposition.quantifier not in (NEC_ALL, NEC_NO)
            assert oracle.truth(proposition.to_json()) is True
            assert "EXISTS(" in sentence.propositional or "ABLE(" in sentence.propositional
    assert set(kinds) == {(CAN, ""), (VERB, "patient"), (VERB, "agent")}
    assert min(kinds.values()) > 15 and deep > 5 and on_patient > 10
    # in the default language, "all" and "no" state the nec quantifiers, so such a subject is
    # said with most or some; when the words state the extensional ones too, it takes them
    stated = Counter(
        s.proposition.quantifier
        for _, s in sentences_of(documents, *ENCYCLOPEDIC)
        if s.proposition.subject.clauses
    )
    assert set(stated) <= {MOST, SOME}
    either = merged(settings, {"quantifiers": {"universal_words": "either"}})
    _, documents = corpora("default", 250, **either)
    quantifiers = Counter(
        s.proposition.quantifier
        for _, s in sentences_of(documents, *ENCYCLOPEDIC)
        if s.proposition.subject.clauses
    )
    assert quantifiers[ALL] > 0 and quantifiers[NO] > 0
    # with the rate at 0, or the depth limit at 0, no class-level noun phrase has a drawn clause
    for off in ({"rate": 0.0}, {"rate": 0.5, "max_depth": 0}):
        _, documents = corpora("default", 100, mention={"relative_clauses": off})
        for _, sentence in sentences_of(documents, *ENCYCLOPEDIC):
            proposition = sentence.proposition
            assert not proposition.subject.clauses
            assert (
                proposition.predicate.patient is None or not proposition.predicate.patient.clauses
            )


# ---------------------------------------------------------------------------------------------
# Limits, readings, and the conceptual rendering
# ---------------------------------------------------------------------------------------------


def test_the_limit_on_content_words(corpora) -> None:
    limit = 6
    settings = {
        "mention": {
            "max_content_words": limit,
            "relative_clauses": {"rate": 1.0, "max_depth": 2},
        }
    }
    _, documents = corpora("default", 200, **settings)
    over = within = 0
    for _, sentence in sentences_of(documents):
        plan, proposition = sentence.plan, sentence.proposition
        count = content_words(plan)
        if proposition.rule is not None:
            continue  # rule statements are exempt
        if count <= limit:
            within += 1
            continue
        # a sentence passes the limit only for the modifiers that tell its referents apart
        over += 1
        assert proposition.level != CLASS
        assert all(p.clause is None for p in plan.noun_phrases())
    assert within > 500 and over < 0.1 * within
    # with the default limit, such sentences keep their relative clauses
    _, documents = corpora(
        "default", 200, mention={"relative_clauses": {"rate": 1.0, "max_depth": 2}}
    )
    depths = Counter(s.plan.depth() for _, s in sentences_of(documents))
    assert depths[2] > 50 and max(depths) == 2


def test_content_words_count_nouns_adjectives_and_verbs(corpora) -> None:
    planner, documents = corpora("default", **RICH)
    content = {x.label for x in planner.lexicon.content_lexemes}
    for _, sentence in sentences_of(documents):
        words = [t for t in sentence.sentence.tokens if token_parts(t)[0] in content]
        assert content_words(sentence.plan) == len(words)


def test_readings_in_documents(corpora) -> None:
    kinds = {CLASS: "generic", INSTANCE: "capacity", EVENT: "event"}
    levels = set(kinds.values())
    # the default language is not ambiguous: every sentence has the one reading of its level,
    # and a class-level sentence has its quantifier readings after it
    _, documents = corpora("default", 200)
    quantified: Counter = Counter()
    for _, sentence in sentences_of(documents):
        proposition = sentence.proposition
        assert sentence.readings[0] == kinds[proposition.level]
        rest = sentence.readings[1:]
        assert not (set(rest) & levels)
        if proposition.level != CLASS:
            assert not rest
        elif proposition.predicate.kind == SCALAR:
            assert not rest
        else:
            # the true quantifier is among the readings of the words
            assert proposition.quantifier in rest, (sentence.readings, proposition.quantifier)
            quantified["+".join(rest)] += 1
    counts = reading_counts(documents)
    assert counts["ambiguous"] == 0 and counts["ambiguous_share"] == 0.0
    assert set(counts["readings"]) == levels
    assert sum(counts["readings"].values()) == counts["sentences"]
    assert {k: v for k, v in counts["quantifier_readings"].items() if k != "none"} == quantified
    assert {NEC_ALL, f"{NEC_ALL}+{MOST}", MOST, SOME, NEC_NO} <= set(quantified)
    # with "can" dropped for instances, and the tense and the aspect not marked, a bare verb
    # after an instance is a capacity or an event
    bare = {"grammar": {"can_rate": {"instance": 0.0}}}
    _, documents = corpora("default", 200, **bare)
    ambiguous: Counter = Counter()
    for _, sentence in sentences_of(documents):
        level = sentence.proposition.level
        # the true reading is always among the readings
        assert kinds[level] in sentence.readings
        if len(sentence.readings) > 1 and level != CLASS:
            assert sentence.readings == ("capacity", "event")
            ambiguous[level] += 1
        predicate = sentence.proposition.predicate
        if level == EVENT:
            assert len(sentence.readings) == 2
        if level == INSTANCE and predicate.kind in (CAN, VERB) and sentence.proposition.polarity:
            verbal = [
                p for p in clause_propositions(sentence.plan) if p.predicate.kind not in (CAN, VERB)
            ]
            assert len(sentence.readings) == (1 if verbal else 2)
    assert ambiguous[EVENT] > 100 and ambiguous[INSTANCE] > 20 and ambiguous[CLASS] == 0
    counts = reading_counts(documents)
    assert counts["ambiguous"] == sum(ambiguous.values())
    assert counts["readings"]["capacity+event"] == counts["ambiguous"]
    # the logical forms are the same as in the unambiguous language
    _, plain = corpora("default", 200)
    assert [s.propositional for _, s in sentences_of(documents)] == [
        s.propositional for _, s in sentences_of(plain)
    ]
    # with the tense marked, the language is not ambiguous again
    marked = merged(bare, {"grammar": {"morphology": {"tense": {"enabled": True}}}})
    _, documents = corpora("default", 200, **marked)
    assert reading_counts(documents)["ambiguous"] == 0


def test_the_conceptual_rendering_is_the_same_sentence(corpora) -> None:
    """The conceptual rendering has the same words in the same order as the tokens, with the
    same affixes and the same marker words. Only the lexical forms differ: a concept label in
    place of a lexeme."""
    grammar = {
        "word_order": {"clause": "SOV", "adjective": "after"},
        "morphology": {
            "number": {"enabled": True},
            "tense": {"enabled": True, "realization": "word"},
            "aspect": {"enabled": True, "realization": "affix"},
        },
        "can_rate": {"class": 0.5, "instance": 0.5},
    }
    settings = merged(RICH, {"grammar": grammar, "lexicon": {"synonym_rate": 0.5}})
    planner, documents = corpora("default", 150, **settings)
    lexicon = planner.lexicon
    affixes: Counter = Counter()
    synonyms = 0
    for _, sentence in sentences_of(documents):
        record = sentence.sentence
        concepts = record.conceptual.split()
        formal = record.formal.split()
        assert len(concepts) == len(formal) == len(record.tokens)
        for token, concept, word in zip(record.tokens, concepts, formal, strict=True):
            label, affix = token_parts(token)
            lexeme = lexicon.lexeme(label)
            expected = lexeme.concept if affix is None else f"{lexeme.concept}-{affix}"
            assert concept == expected
            assert word == (
                f"{lexeme.gloss}/{label}" if affix is None else f"{lexeme.gloss}/{label}-{affix}"
            )
            affixes[affix] += 1
            synonyms += len(lexicon.lexemes_of(lexeme.concept)) > 1
        # a function word's concept is its gloss in capitals, and a marker word its own gloss
        for concept in concepts:
            assert concept == concept.upper()
    assert affixes["PLURAL"] > 100 and affixes["PROGRESSIVE"] > 20 and synonyms > 100
    assert any("PAST" in s.sentence.conceptual.split() for _, s in sentences_of(documents))


# ---------------------------------------------------------------------------------------------
# The record, determinism, and other worlds
# ---------------------------------------------------------------------------------------------


def test_the_document_record(corpora) -> None:
    planner, documents = corpora("default", **RICH)
    for document in documents[:120]:
        form = document.to_json()
        assert list(form) == ["label", "type", "topic", "scenes", "referents", "sentences"]
        assert json.loads(json.dumps(form)) == form
        assert form["scenes"] == [scene.label for scene in document.scenes]
        for sentence, record in zip(document.sentences, form["sentences"], strict=True):
            assert list(record) == [
                "label",
                "tokens",
                "words",
                "text",
                "formal",
                "conceptual",
                "propositional",
                "tree",
                "logical_form",
                "referents",
                "events",
                "coreference",
                "distinguished",
                "readings",
            ]
            # the word forms come later, with the render command
            assert record["words"] is None and record["text"] is None
            assert leaves(record["tree"]) == record["tokens"]
            assert propositional(record["logical_form"]) == record["propositional"]
            assert len(record["referents"]) == len(record["coreference"])
            assert len(record["referents"]) == len(record["distinguished"])
            assert len(record["events"]) == json.dumps(record["tree"]).count('["VP"')
            assert [e for e in record["events"] if e] == [
                e[0] for e in sentence.sentence.events if e
            ]
            if sentence.proposition.level == EVENT:
                assert record["logical_form"]["event"] in record["events"]
                assert record["logical_form"]["tense"] == "past"


def test_proposition_labels(corpora) -> None:
    _, documents = corpora("default", **RICH)
    labels: dict = {}
    order = []
    for _, sentence in sentences_of(documents):
        proposition = sentence.proposition
        # one label for one proposition, wherever it is said
        assert labels.setdefault(proposition, proposition.id) == proposition.id
        if proposition.id not in order:
            order.append(proposition.id)
    assert order == [f"PROP.{n}" for n in range(1, len(order) + 1)]
    assert len(set(labels.values())) == len(labels)
    assert len(labels) < len(sentences_of(documents))  # some propositions are said twice


def test_documents_are_deterministic(cases) -> None:
    case = cases("default")

    def made(seed: int = 1, **sections) -> list:
        config = corpus_config(case.world_path, seed=seed, **sections)
        return [d.to_json() for d in Planner(config, case.world).generate(40)]

    base = made()
    assert made() == base
    assert made(seed=2) != base
    # a document depends on the documents before it only for its scene and proposition labels:
    # its type, its topic, and its propositions come from its own parts of the streams
    planner = Planner(corpus_config(case.world_path), case.world)
    assert [next(iter(planner)).to_json() for _ in range(3)] == base[:3]
    # the test-set settings change no document
    assert made(test_sets={"size": 3, "changes": ["subject"]}) == base
    # the world can be loaded by the planner itself
    config = corpus_config(case.world_path)
    assert [d.to_json() for d in Planner(config).generate(5)] == base[:5]
    assert load_world(config).instances == case.world.instances


def test_quantifier_weights_rebalance_the_choice_of_facts(cases) -> None:
    case = cases("default")

    def made(**weights):
        # equal kinds of content, so that the category documents state scalar poles too
        sections = {"documents": {"content_kind_weights": "equal"}}
        if weights:
            sections["quantifiers"] = {"weights": weights}
        planner = Planner(case.config(**sections), case.world)
        documents = planner.generate(150)
        strengths = Counter(
            s.strength for _, s in sentences_of(documents, *ENCYCLOPEDIC) if s.section != RULE
        )
        return planner, documents, strengths

    def share(strengths: Counter, quantifier: str) -> float:
        return strengths[quantifier] / sum(strengths.values())

    planner, base, strengths = made()
    # in the default language, "all" and "no" state the nec quantifiers
    assert set(strengths) == {NEC_ALL, MOST, SOME, NEC_NO, "pole"}
    for _, sentence in sentences_of(base):
        proposition = sentence.proposition
        if proposition.level != CLASS:
            assert sentence.strength is None
        elif proposition.predicate.kind == SCALAR:
            assert sentence.strength == "pole"
        else:
            # a fact is stated with its strongest true quantifier that the language can state,
            # with a word or with a bare plural
            fact = planner.facts.class_fact(
                proposition.subject, proposition.predicate, proposition.negative
            )
            assert fact.quantifier == sentence.strength == proposition.quantifier
    # equal weights, of any size, change nothing
    _, same, _ = made(nec_all=3, all=3, most=3, some=3, none=3, nec_none=3)
    assert [d.to_json() for d in same] == [d.to_json() for d in base]
    # a lighter weight makes the documents state fewer facts of that quantifier
    planner, lighter, fewer = made(some=0.2)
    assert share(fewer, SOME) < share(strengths, SOME) - 0.15
    assert share(fewer, NEC_ALL) > share(strengths, NEC_ALL)
    assert share(fewer, MOST) > share(strengths, MOST)
    for _, sentence in sentences_of(lighter):
        assert planner.truth.is_true(sentence.proposition)
        proposition = sentence.proposition
        if proposition.level == CLASS and proposition.predicate.kind != SCALAR:
            fact = planner.facts.class_fact(
                proposition.subject, proposition.predicate, proposition.negative
            )
            assert fact.quantifier == sentence.strength  # still the strongest true quantifier
    # the weights change the facts of encyclopedic documents only: every narrative says the
    # same, apart from the labels of its propositions, which are numbered across the corpus
    assert [d.type for d in lighter] == [d.type for d in base]
    for a, b in zip(lighter, base, strict=True):
        if a.type in NARRATIVES:
            assert (a.topic, a.scenes, a.referents) == (b.topic, b.scenes, b.referents)
            assert [(s.propositional, s.sentence.tokens) for s in a.sentences] == [
                (s.propositional, s.sentence.tokens) for s in b.sentences
            ]
    # a weight of 0 leaves the quantifier out, and a heavier one states it more often
    _, _, without = made(some=0)
    assert without[SOME] == 0 and without[NEC_ALL] > strengths[NEC_ALL]
    # the polarity of a fact is drawn first, at the negation rate, so the weight of "no" moves
    # the mix among the negative facts: "no" against "most ... not" and "some ... not"
    _, _, heavier = made(nec_none=4)
    assert share(heavier, NEC_NO) > 1.5 * share(strengths, NEC_NO)
    with pytest.raises(ConfigError, match="quantifiers.weights.no"):
        case.config(quantifiers={"weights": {"no": 4}})


def test_an_entity_narrative_has_two_to_five_scenes(cases) -> None:
    case = cases("default")
    planner = Planner(case.config(documents={"mix": {"entity": 1}}), case.result)
    documents = planner.generate(80)
    counts = Counter(len(d.scenes) for d in documents)
    assert set(counts) == {2, 3, 4, 5}
    for document in documents:
        assert all(scene.seed == document.topic for scene in document.scenes)
        assert len(document.sentences) <= document.drawn_length
    one = Planner(case.config(entity={"scenes": 1}, documents={"mix": {"entity": 1}}), case.result)
    shorter = one.generate(80)
    assert {len(d.scenes) for d in shorter} == {1}
    # more scenes give a narrative more events to report, so it comes closer to its drawn length
    assert sum(len(d.sentences) for d in documents) > 1.3 * sum(len(d.sentences) for d in shorter)


def test_a_lexicon_with_unnamed_concepts(cases) -> None:
    case = cases("default")
    proportions = dict.fromkeys(CONCEPT_TYPES, 0.5)
    config = case.config(lexicon={"named_proportion": proportions}, **RICH)
    planner = Planner(config, case.result)
    documents = planner.generate(150)
    named = planner.facts.named
    assert len(named) < 0.6 * len(planner.lexicon.concepts)
    for _, sentence in sentences_of(documents):
        # every concept of every sentence has a word
        for token in sentence.sentence.tokens:
            assert planner.lexicon.lexeme(token_parts(token)[0])
        for concept in sentence.proposition.concepts():
            assert concept in named
        assert planner.truth.is_true(sentence.proposition)
    assert {d.type for d in documents} == {*ENCYCLOPEDIC, *NARRATIVES}


def test_a_language_with_nothing_to_say(cases) -> None:
    case = cases("tiny")
    proportions = dict.fromkeys(CONCEPT_TYPES, 0.0)
    planner = Planner(case.config(lexicon={"named_proportion": proportions}), case.result)
    with pytest.raises(CorpusError, match="no document with a sentence could be made for DOC.1"):
        planner.generate(1)
    # with words for the CAN features alone, the narratives still report events: "it swam"
    proportions["event_unary"] = 1.0
    planner = Planner(case.config(lexicon={"named_proportion": proportions}), case.result)
    documents = planner.generate(10)
    assert {d.type for d in documents} <= {*ENCYCLOPEDIC, *NARRATIVES}
    for document, sentence in sentences_of(documents, *NARRATIVES):
        assert all(phrase.pronoun for phrase in sentence.plan.noun_phrases())
        assert (
            proposition_of(parse_propositional(sentence.propositional), document.referents)
            == sentence.proposition
        )


def test_a_world_without_verbs_and_scalars(world_files) -> None:
    config = corpus_config(world_files["plain"], **RICH)
    planner = Planner(config)
    documents = planner.generate(80)
    sections = Counter(s.section for _, s in sentences_of(documents))
    assert sections[RELATION] == 0 and sections[EVENT_SECTION] > 50
    for document, sentence in sentences_of(documents):
        proposition = sentence.proposition
        assert proposition.predicate.kind not in (VERB, SCALAR)
        assert planner.truth.is_true(proposition)
        parsed = parse_propositional(sentence.propositional)
        assert proposition_of(parsed, document.referents) == proposition
    with pytest.raises(ConfigError):
        corpus_config(world_files["plain"], documents={"sibling_contrast_rate": 2})
