"""Layer 4: realization. A sentence plan becomes words and a parse tree.

The tree is a nested list of the form ``[label, child, ...]``, and its leaves are the sentence's
tokens. A token is a lexeme label, joined to the gloss of its affix when it has one
(``L.57-PLURAL``). The node labels are:

- ``S``, the sentence; ``NP-SBJ`` and ``NP-OBJ``, the subject and the object; ``VP``, the verb
  phrase; ``NP-PRD``, the noun of a "has" or "is a" predicate; ``AP``, the adjectives; ``PP``, a
  with-phrase; ``RC``, a relative clause;
- ``N``, ``V``, ``A``, ``Det``, ``Pro``, ``P``, ``Conj``, ``Rel``, ``AUX``, and ``Neg``, the words;
- ``PLURAL``, ``PAST``, and ``PROGRESSIVE``, an inflection realized as a separate word. The word
  it marks is then a node with two children: ``["N", ["N", "L.57"], ["PLURAL", "L.190"]]``.

**The clause.** The subject, the verb phrase, and the object are placed by ``word_order.clause``.
When the verb and the object are neighbors (SVO, SOV, VOS, OVS), the object is inside the verb
phrase. When the subject stands between them (VSO, OSV), the object is a daughter of ``S``.

**The verb phrase.** The auxiliary (``can``, ``is``, ``has``) stands before or after the
predicate word by ``word_order.auxiliary``, and ``not`` stands right after or right before the
auxiliary by ``word_order.negation``.

**The noun phrase.** The noun is built outward: the adjectives, then the determiner, then the
with-phrases, then the relative clause, each placed before or after what is already there.

**The relative clause.** ``that`` comes first. A subject relative is one or more verb phrases
joined by ``and``. A verb phrase with the same auxiliary as the one before it leaves the
auxiliary out ("that are not red and not big"). An object relative is the clause's subject and
its verb, in the order of ``word_order.clause``.

**Number and agreement.** With number on, a class-level noun is plural. A verb with no auxiliary
agrees with its subject: it takes the plural marker when ``number.verb_marks`` names its
subject's number. A verb that carries a tense or aspect marker takes no agreement marker, and a
verb after ``can`` takes none. ``is`` and ``has`` agree by becoming ``are`` and ``have``.

**Tense and aspect.** An event's tense and aspect are in the plan. The morphology marks them or
not: a past event takes the past marker when tense is on, and a progressive event takes the
progressive marker when aspect is on. The present and the simple aspect are never marked.

**``can``.** A negative capacity always says ``can``. A positive capacity says ``can`` at
``grammar.can_rate``, for a category and for an instance. Without ``can``, and without a tense
or aspect marker, an instance's capacity and an event have the same words and the same tree.

The grammar's own choices are drawn from the generator it is given: a synonym for a concept
with two lexemes, the adjective order when it is not fixed, and the ``can`` of a positive
capacity. The draws are made in one order, whatever the word order, so a word-order setting
never changes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.grammar import (
    CLASS_NP,
    GrammarError,
    NounPhrase,
    Predication,
    SentencePlan,
    check_plan,
)
from semantic_world.corpus.lexicon import ADJECTIVE, Lexeme, Lexicon
from semantic_world.corpus.propositions import CAN, HAS, IS, MEMBER, NEC_NO, NO, VERB
from semantic_world.corpus.propositions import PAST as PAST_TENSE
from semantic_world.corpus.propositions import PROGRESSIVE as PROGRESSIVE_ASPECT
from semantic_world.corpus.streams import Streams

PLURAL = "PLURAL"
PAST = "PAST"
PROGRESSIVE = "PROGRESSIVE"
MARKERS = (PLURAL, PAST, PROGRESSIVE)
NP_LABELS = ("NP-SBJ", "NP-OBJ")
WORD_LABELS = ("N", "V", "A", "Det", "Pro", "P", "Conj", "Rel", "AUX", "Neg")

Tree = list


def leaves(tree: Tree) -> list[str]:
    """The tokens of a tree, in order."""
    found: list[str] = []
    for child in tree[1:]:
        if isinstance(child, str):
            found.append(child)
        else:
            found += leaves(child)
    return found


def preorder(tree: Tree) -> list[Tree]:
    """Every node of a tree, a node before its children."""
    nodes = [tree]
    for child in tree[1:]:
        if not isinstance(child, str):
            nodes += preorder(child)
    return nodes


def token_parts(token: str) -> tuple[str, str | None]:
    """A token's lexeme label, and the gloss of its affix or None."""
    label, _, affix = token.partition("-")
    return label, affix or None


@dataclass(frozen=True)
class Sentence:
    """One realized sentence."""

    plan: SentencePlan
    tokens: tuple[str, ...]
    """Lexeme labels, each joined to the gloss of its affix when it has one."""
    tree: Tree
    formal: str
    conceptual: str
    referents: tuple[tuple[str, str | None], ...]
    """For each noun phrase (``NP-SBJ`` and ``NP-OBJ``), a node before its children: the
    instance or category it refers to, and the category its noun names (None for a pronoun)."""
    phrases: tuple[NounPhrase, ...]
    """For each noun phrase, in the same order: its part of the plan."""
    events: tuple[tuple[str, str, str] | None, ...]
    """For each verb phrase (``VP``), a node before its children: the event it reports, with
    its tense and its aspect, or None."""

    @property
    def event_labels(self) -> tuple[str | None, ...]:
        """For each verb phrase, the label of the event it reports, or None."""
        return tuple(None if e is None else e[0] for e in self.events)


class Realizer:
    """The grammar of one language: a lexicon and the grammar settings."""

    def __init__(self, config: Config, lexicon: Lexicon, streams: Streams) -> None:
        self.settings = config.grammar
        self.order = config.grammar.word_order
        self.morphology = config.grammar.morphology
        self.lexicon = lexicon
        self._marker = {
            PLURAL: self.morphology.number,
            PAST: self.morphology.tense,
            PROGRESSIVE: self.morphology.aspect,
        }
        # The fixed adjective order: a random ordering of the adjective concepts, drawn once.
        adjectives = [c.label for c in lexicon.concepts if c.pos == ADJECTIVE]
        order = streams.substream("grammar", "adjective_order").permutation(len(adjectives))
        self.adjective_rank = {adjectives[int(i)]: rank for rank, i in enumerate(order)}

    def realize(self, plan: SentencePlan, rng: np.random.Generator) -> Sentence:
        """The words and the tree of a sentence plan. ``rng`` supplies the grammar's own
        choices."""
        check_plan(plan)
        return _Builder(self, rng).sentence(plan)

    # Renderings ------------------------------------------------------------------------------

    def formal(self, tokens: tuple[str, ...] | list[str]) -> str:
        """The formal rendering: each word as its gloss and its lexeme label, with the gloss of
        its affix (``C1.3/L.5-PLURAL``)."""
        words = []
        for token in tokens:
            label, affix = token_parts(token)
            word = f"{self.lexicon.lexeme(label).gloss}/{label}"
            words.append(word if affix is None else f"{word}-{affix}")
        return " ".join(words)

    def conceptual(self, tokens: tuple[str, ...] | list[str]) -> str:
        """The conceptual rendering: each word replaced by its concept label, joined to the
        gloss of its affix (``C1.3-PLURAL``)."""
        words = []
        for token in tokens:
            label, affix = token_parts(token)
            concept = self.lexicon.lexeme(label).concept
            words.append(concept if affix is None else f"{concept}-{affix}")
        return " ".join(words)


class _Builder:
    """The build of one sentence. The parts are built in one order (the subject, the verb phrase,
    then the object), so the draws do not depend on the word order."""

    def __init__(self, realizer: Realizer, rng: np.random.Generator) -> None:
        self.realizer = realizer
        self.lexicon = realizer.lexicon
        self.order = realizer.order
        self.morphology = realizer.morphology
        self.rng = rng
        self.referents: dict[int, tuple[str, str | None]] = {}
        self.phrases: dict[int, NounPhrase] = {}
        self.events: dict[int, tuple[str, str, str] | None] = {}
        clause = self.order.clause
        self.subject_first = clause.index("S") < clause.index("V")
        self.verb_first = clause.index("V") < clause.index("O")

    # Words -----------------------------------------------------------------------------------

    def lexeme(self, concept: str) -> Lexeme:
        """A lexeme of a concept: one of the two at random for a concept with a synonym."""
        lexemes = self.lexicon.lexemes_of(concept)
        if not lexemes:
            raise GrammarError(f"the concept {concept} has no word")
        if len(lexemes) == 1:
            return lexemes[0]
        return lexemes[int(self.rng.integers(len(lexemes)))]

    def function(self, label: str, gloss: str) -> Tree:
        try:
            return [label, self.lexicon.function_word(gloss).label]
        except KeyError as error:
            raise GrammarError(str(error.args[0])) from None

    def word(self, label: str, concept: str, marks: tuple[str, ...] = ()) -> Tree:
        """A content word with its inflections: an affix joins the token, and an inflection
        realized as a word stands right before or right after it."""
        lexeme = self.lexeme(concept)
        affixes = [m for m in marks if self.realizer._marker[m].realization == "affix"]
        if len(affixes) > 1:
            raise GrammarError(f"a word takes one affix, and {concept} would take {affixes}")
        token = lexeme.label if not affixes else f"{lexeme.label}-{affixes[0]}"
        words = [m for m in marks if self.realizer._marker[m].realization == "word"]
        if not words:
            return [label, token]
        before = [m for m in words if self.realizer._marker[m].position == "before"]
        after = [m for m in words if self.realizer._marker[m].position == "after"]
        return [
            label,
            *(self.function(m, m) for m in before),
            [label, token],
            *(self.function(m, m) for m in after),
        ]

    # Noun phrases ----------------------------------------------------------------------------

    def plural(self, phrase: NounPhrase) -> bool:
        return self.morphology.number.enabled and phrase.kind == CLASS_NP

    def noun_phrase(self, phrase: NounPhrase, label: str) -> Tree:
        if phrase.pronoun:
            node = [label, self.function("Pro", "it")]
            self.referents[id(node)] = (phrase.referent, None)
            self.phrases[id(node)] = phrase
            return node
        assert phrase.noun is not None
        plural = self.plural(phrase)
        parts: list[Tree] = [self.word("N", phrase.noun, (PLURAL,) if plural else ())]
        adjectives = [x.feature for x in phrase.adjectives]
        if adjectives:
            if self.realizer.settings.adjective_order_fixed:
                adjectives.sort(key=lambda a: self.realizer.adjective_rank[a])
            else:
                adjectives = [adjectives[int(i)] for i in self.rng.permutation(len(adjectives))]
            group = ["AP", *(self.word("A", a) for a in adjectives)]
            parts = self.place(parts, [group], self.order.adjective)
        if phrase.determiner is not None:
            determiner = self.function("Det", phrase.determiner)
            parts = self.place(parts, [determiner], self.order.determiner)
        phrases: list[Tree] = []
        for literal in phrase.with_phrases:
            if phrases:
                phrases.append(self.function("Conj", "and"))
            adposition = self.function("P", "with" if literal.positive else "without")
            part = self.word("N", literal.feature)
            preposition = self.order.adposition == "preposition"
            phrases.append(["PP", adposition, part] if preposition else ["PP", part, adposition])
        if phrases:
            parts = self.place(parts, phrases, self.order.with_phrase)
        clause = self.relative_clause(phrase)
        if clause is not None:
            parts = self.place(parts, [clause], self.order.relative_clause)
        node = [label, *parts]
        self.referents[id(node)] = (phrase.referent, phrase.noun)
        self.phrases[id(node)] = phrase
        return node

    @staticmethod
    def place(parts: list[Tree], added: list[Tree], side: str) -> list[Tree]:
        return added + parts if side == "before" else parts + added

    def relative_clause(self, phrase: NounPhrase) -> Tree | None:
        clause = phrase.clause
        negated = [Predication(IS, x.feature, False) for x in phrase.negated]
        if clause is None and not negated:
            return None
        relativizer = self.function("Rel", "that")
        if clause is not None and clause.agent is not None:
            # An object relative: the clause's subject and its verb, with the head as the object.
            agent = self.noun_phrase(clause.agent, "NP-SBJ")
            parts, _, _ = self.verb_phrase(clause.predications[0], clause.agent)
            verb_phrase = self.verb_node(parts, clause.predications[0])
            ordered = [agent, verb_phrase] if self.subject_first else [verb_phrase, agent]
            return ["RC", relativizer, *ordered]
        # A subject relative: verb phrases about the head, joined by "and".
        predications = negated + list(clause.predications if clause is not None else ())
        children: list[Tree] = [relativizer]
        previous: str | None = None
        for predication in predications:
            if len(children) > 1:
                children.append(self.function("Conj", "and"))
            parts, target, previous = self.verb_phrase(predication, phrase, elide=previous)
            if target is not None:
                parts = parts + [target] if self.verb_first else [target] + parts
            children.append(self.verb_node(parts, predication))
        return ["RC", *children]

    # Verb phrases ----------------------------------------------------------------------------

    def verb_node(self, parts: list[Tree], predication: Predication) -> Tree:
        node = ["VP", *parts]
        if predication.event is None:
            self.events[id(node)] = None
        else:
            assert predication.tense is not None and predication.aspect is not None
            self.events[id(node)] = (predication.event, predication.tense, predication.aspect)
        return node

    def verb_phrase(
        self,
        predication: Predication,
        subject: NounPhrase,
        elide: str | None = None,
        negate: bool = False,
    ) -> tuple[list[Tree], Tree | None, str | None]:
        """The words of a verb phrase without its object, the object's noun phrase, and the
        gloss of the auxiliary. ``elide`` is the auxiliary of the verb phrase before it in a
        joined relative clause: the same auxiliary is not said again. ``negate`` says the phrase
        is negated although its predication is positive: a bare plural that states ``no`` or
        ``nec_no`` ("penguins are not fish")."""
        morphology = self.morphology
        plural = self.plural(subject)
        agree = morphology.agreement
        kind = predication.kind
        polarity = predication.polarity and not negate
        auxiliary: str | None = None
        negated = False
        target: Tree | None = None
        if kind in (CAN, VERB):
            marks: tuple[str, ...] = ()
            if predication.event is not None:
                if morphology.tense.enabled and predication.tense == PAST_TENSE:
                    marks += (PAST,)
                if morphology.aspect.enabled and predication.aspect == PROGRESSIVE_ASPECT:
                    marks += (PROGRESSIVE,)
            elif not polarity:
                auxiliary, negated = "can", True
            else:
                level = "class" if subject.kind == CLASS_NP else "instance"
                if self.rng.random() < self.realizer.settings.can_rate[level]:
                    auxiliary = "can"
            marked_number = "plural" if plural else "singular"
            if (
                auxiliary is None
                and not marks
                and agree
                and morphology.number.verb_marks == marked_number
            ):
                marks = (PLURAL,)
            head = self.word("V", predication.label, marks)
            if predication.object is not None:
                target = self.noun_phrase(predication.object, "NP-OBJ")
        elif kind == HAS:
            auxiliary = "have" if plural and agree else "has"
            part = [self.word("N", predication.label)]
            if not polarity:
                part = self.place(part, [self.function("Det", "no")], self.order.determiner)
            head = ["NP-PRD", *part]
        elif kind == MEMBER:
            auxiliary = "are" if plural and agree else "is"
            negated = not polarity
            noun = [self.word("N", predication.label, (PLURAL,) if plural else ())]
            if not plural:
                noun = self.place(noun, [self.function("Det", "a")], self.order.determiner)
            head = ["NP-PRD", *noun]
        else:  # a property, a scalar pole, or an exposed patient projection
            auxiliary = "are" if plural and agree else "is"
            negated = not polarity
            head = self.word("A", predication.label)

        negation = self.function("Neg", "not") if negated else None
        before = self.order.auxiliary == "before"
        if auxiliary is not None and auxiliary != elide:
            group = [self.function("AUX", auxiliary)]
            if negation is not None:
                after = self.order.negation == "after_auxiliary"
                group = group + [negation] if after else [negation] + group
            parts = group + [head] if before else [head] + group
        elif negation is not None:
            # the auxiliary is left out: "not" keeps its side of the predicate word
            parts = [negation, head] if before else [head, negation]
        else:
            parts = [head]
        return parts, target, auxiliary

    # The sentence ----------------------------------------------------------------------------

    def sentence(self, plan: SentencePlan) -> Sentence:
        subject = self.noun_phrase(plan.subject, "NP-SBJ")
        negate = plan.bare_plural and plan.quantifier in (NO, NEC_NO)
        parts, target, _ = self.verb_phrase(plan.predication, plan.subject, negate=negate)
        clause = self.order.clause
        if target is None:
            verb_phrase = self.verb_node(parts, plan.predication)
            children = [subject, verb_phrase] if self.subject_first else [verb_phrase, subject]
        elif abs(clause.index("V") - clause.index("O")) == 1:
            # the verb and the object are neighbors: the object is inside the verb phrase
            inside = parts + [target] if self.verb_first else [target] + parts
            verb_phrase = self.verb_node(inside, plan.predication)
            children = [subject, verb_phrase] if self.subject_first else [verb_phrase, subject]
        else:
            # the subject stands between them: the object is a daughter of the sentence
            nodes = {"S": subject, "V": self.verb_node(parts, plan.predication), "O": target}
            children = [nodes[letter] for letter in clause]
        tree: Tree = ["S", *children]
        tokens = tuple(leaves(tree))
        nodes_in_order = preorder(tree)
        return Sentence(
            plan=plan,
            tokens=tokens,
            tree=tree,
            formal=self.realizer.formal(tokens),
            conceptual=self.realizer.conceptual(tokens),
            referents=tuple(self.referents[id(n)] for n in nodes_in_order if n[0] in NP_LABELS),
            phrases=tuple(self.phrases[id(n)] for n in nodes_in_order if n[0] in NP_LABELS),
            events=tuple(self.events[id(n)] for n in nodes_in_order if n[0] == "VP"),
        )


def tree_depth(tree: Tree) -> int:
    """The deepest nesting of relative clauses in a tree."""
    inner = max((tree_depth(c) for c in tree[1:] if not isinstance(c, str)), default=0)
    return inner + (1 if tree[0] == "RC" else 0)


def as_json(tree: Any) -> Any:
    """A tree as plain nested lists, for ``documents.jsonl``."""
    return [tree[0], *(c if isinstance(c, str) else as_json(c) for c in tree[1:])]
