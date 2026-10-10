"""Readings: the kinds of logical form that a sentence's words allow.

The propositional rendering is never ambiguous, and the surface language can be. A sentence has
one of three kinds of logical form, one for each level:

- ``generic``: a class-level sentence, about a category ("penguins swim");
- ``capacity``: an instance-level sentence, about what an instance is, has, or can do ("the
  penguin can swim");
- ``event``: an event-level sentence, about what happened ("the penguin swam");
- ``state``: a sentence about a fluent at a time point ("the mouse was asleep", "the mouse
  became asleep"): the predicate word is a state adjective, or ``become`` stands before it;
- ``able_now``: what a participant could do at a time point ("the owl could not eat the
  mouse"). With ``lexicon.can_words: shared``, "can" expresses ``ABLE`` and ``ABLE_NOW`` alike,
  so a sentence about an instance with "can" has both the ``capacity`` and the ``able_now``
  readings; with ``distinct``, "can" allows ``capacity`` only and the word ``can_now`` allows
  ``able_now`` only. A tense marker beside the auxiliary leaves ``able_now`` alone, because a
  capacity has no tense;
- ``causal``: a class-level sentence with ``become`` or with ``before`` ("things that things
  catch become caught"), which states what events do or need. It takes the place of
  ``generic``, and the quantifier readings of its words follow it as for any class-level
  sentence.

A noun phrase shows whether it names a category or an instance, so a class-level sentence has the
one level reading ``generic``, followed by its quantifier readings: the quantifiers that its
words allow. A bare plural allows every quantifier in ``quantifiers.bare_plural.expresses``,
and ``nec_all``, because membership and rule statements use the bare plural for ``nec_all``.
"All" allows ``nec_all``, ``all``, or both, and "no" allows ``nec_no``, ``no``, or both, by
``quantifiers.universal_words``; "most" and "some" allow ``most`` and ``some``. A class-level
scalar pole ("penguins are big") is a bare plural with no quantifier reading. A sentence about
instances can be ambiguous between ``capacity`` and ``event``: with ``grammar.can_rate.instance``
below 1 a capacity can drop ``can``, and with the tense and the aspect unmarked an event has a
bare verb too. "The penguin swim" then has both readings.

The readings are worked out from the tree and the lexemes of its leaves alone. Nothing of the
sentence's record is used: not the events its verb phrases report, and not its logical form. The
tree itself carries no information about the reading, because a capacity without ``can`` and an
unmarked event have the same words and the same tree. Lexemes, not word forms, are read, so a
homonym adds no reading.

Every verb phrase of a sentence has the sentence's level, so the readings of a sentence are those
that every one of its verb phrases allows:

- a verb phrase with ``can`` or ``not``, and one whose predicate word is not a verb ("is red",
  "has fins", "is a bird"), states a capacity. A relative clause "that is not red" belongs to its
  noun phrase's restriction, and allows both readings;
- a verb marked for tense or aspect reports an event;
- a bare verb states a capacity when the language lets a capacity drop ``can``, and reports an
  event when the language has events that no marker shows: the tense is the present or is not
  marked, and the simple aspect occurs or the aspect is not marked.
"""

from __future__ import annotations

from semantic_world.corpus.config import Config
from semantic_world.corpus.lexicon import ABLE_NOW_WORD, BEFORE_WORD, Lexicon
from semantic_world.corpus.propositions import ALL, COUNTERPART, MOST, NEC_ALL, NEC_NO, NO, SOME
from semantic_world.corpus.realize import Tree, token_parts
from semantic_world.corpus.world import (
    CATEGORY_PREFIX,
    FLUENT_PREFIX,
    PROPERTY_PREFIX,
    SCALAR_PREFIX,
)

GENERIC = "generic"
CAPACITY = "capacity"
EVENT = "event"
STATE = "state"
ABLE_NOW = "able_now"
CAUSAL = "causal"
READINGS = (GENERIC, CAPACITY, EVENT, STATE, ABLE_NOW, CAUSAL)
LEVEL_READINGS = READINGS
"""The readings that name a level, in the order they are listed."""
QUANTIFIER_READINGS = (NEC_ALL, ALL, MOST, SOME, NO, NEC_NO)
"""The quantifier readings of a class-level sentence, in the order they are listed."""

_INSTANCE_DETERMINERS = ("a", "the")
_EVENT_MARKS = {"PAST", "PROGRESSIVE"}


def readings(tree: Tree, lexicon: Lexicon, config: Config) -> tuple[str, ...]:
    """The readings that a sentence's tree and lexemes allow, in the order ``generic``,
    ``capacity``, ``event``, then the quantifier readings of a class-level sentence."""
    return _Reader(lexicon, config).sentence(tree)


class _Reader:
    def __init__(self, lexicon: Lexicon, config: Config) -> None:
        self.lexicon = lexicon
        morphology = config.grammar.morphology
        events = config.propositions
        self.bare_capacity = config.grammar.can_rate["instance"] < 1
        """Whether an instance's capacity can be said without ``can``."""
        shared = config.lexicon.can_words == "shared"
        self.auxiliary_readings: dict[str, set[str]] = {
            "can": {CAPACITY, ABLE_NOW} if shared else {CAPACITY},
            ABLE_NOW_WORD: {ABLE_NOW},
        }
        """The readings that "can" and ``can_now`` allow before a verb."""
        self.bare_event = (not morphology.tense.enabled or events.event_tense == "present") and (
            not morphology.aspect.enabled or config.documents.progressive_rate < 1
        )
        """Whether an event can be said with no marker on its verb."""
        quantifiers = config.quantifiers
        universal = quantifiers.universal_words
        self.word_quantifiers: dict[str, tuple[str, ...]] = {
            "all": {"nec": (NEC_ALL,), "extensional": (ALL,), "either": (NEC_ALL, ALL)}[universal],
            "no": {"nec": (NEC_NO,), "extensional": (NO,), "either": (NEC_NO, NO)}[universal],
            "most": (MOST,),
            "some": (SOME,),
        }
        """The quantifiers that each determiner word allows."""
        self.bare_quantifiers: tuple[str, ...] = tuple(
            q for q in QUANTIFIER_READINGS if q == NEC_ALL or q in quantifiers.bare_plural_expresses
        )
        """The quantifiers that a bare plural allows, and, when the sentence is negated, their
        counterparts (``nec_no`` for a negative membership fact, and the negative counterpart of
        each quantifier the bare plural expresses)."""
        self.negated_bare_quantifiers: tuple[str, ...] = tuple(
            q
            for q in QUANTIFIER_READINGS
            if q in {COUNTERPART[b] for b in self.bare_quantifiers if b != NEC_ALL}
        )
        """The quantifiers that a negated bare plural allows; a membership fact adds ``nec_no``
        (a bare plural states ``nec_no`` for membership facts only)."""

    def gloss(self, node: Tree) -> str:
        return self.lexicon.lexeme(token_parts(self.token(node))[0]).gloss

    def token(self, node: Tree) -> str:
        for child in node[1:]:
            if isinstance(child, str):
                return child
            if child[0] == node[0]:
                return self.token(child)
        raise ValueError(f"the node {node[0]} holds no word")

    def marks(self, node: Tree) -> set[str]:
        """The inflections of a word node: its affix, and the markers that stand beside it."""
        found: set[str] = set()
        for child in node[1:]:
            if isinstance(child, str):
                affix = token_parts(child)[1]
                if affix is not None:
                    found.add(affix)
            elif child[0] == node[0]:
                found |= self.marks(child)
            else:
                found.add(child[0])
        return found

    @staticmethod
    def child(node: Tree, label: str) -> Tree | None:
        return next((c for c in node[1:] if not isinstance(c, str) and c[0] == label), None)

    def names_an_instance(self, phrase: Tree) -> bool:
        if self.child(phrase, "Pro") is not None:
            return True
        determiner = self.child(phrase, "Det")
        return determiner is not None and self.gloss(determiner) in _INSTANCE_DETERMINERS

    def sentence(self, tree: Tree) -> tuple[str, ...]:
        subject = self.child(tree, "NP-SBJ")
        if subject is None:
            raise ValueError("a sentence has a subject")
        if not self.names_an_instance(subject):
            level = CAUSAL if self.causal(tree) else GENERIC
            return (level, *self.quantifier_readings(tree, subject))
        allowed = self.walk(tree)
        return tuple(reading for reading in READINGS if reading in allowed)

    def quantifier_readings(self, tree: Tree, subject: Tree) -> tuple[str, ...]:
        """The quantifiers that a class-level sentence's words allow: those of its determiner
        word, or of the bare plural. A scalar pole as the predicate word has none."""
        verb_phrase = self.child(tree, "VP")
        adjective = None if verb_phrase is None else self.child(verb_phrase, "A")
        if adjective is not None:
            concept = self.lexicon.lexeme(token_parts(self.token(adjective))[0]).concept
            if concept.startswith(SCALAR_PREFIX):
                return ()
        determiner = self.child(subject, "Det")
        if determiner is None:
            # the "not" of a causal statement gives the fluent's value, and negates nothing
            if verb_phrase is None or not self.negated(verb_phrase) or self.causal(tree):
                return self.bare_quantifiers
            readings = self.negated_bare_quantifiers
            if self.names_a_category(verb_phrase) and NEC_ALL in self.bare_quantifiers:
                readings = tuple(q for q in QUANTIFIER_READINGS if q == NEC_NO or q in readings)
            return readings
        return self.word_quantifiers.get(self.gloss(determiner), ())

    def causal(self, tree: Tree) -> bool:
        """Whether a class-level sentence is a causal statement: its verb phrase has ``become``
        or ``before``."""
        verb_phrase = self.child(tree, "VP")
        if verb_phrase is None:
            return False
        auxiliary = self.child(verb_phrase, "AUX")
        adverb = self.child(verb_phrase, "Adv")
        return (auxiliary is not None and self.gloss(auxiliary) == "become") or (
            adverb is not None and self.gloss(adverb) == BEFORE_WORD
        )

    def negated(self, verb_phrase: Tree) -> bool:
        """Whether a verb phrase is negated: by ``not``, or by ``no`` on its predicate noun."""
        if self.child(verb_phrase, "Neg") is not None:
            return True
        nominal = self.child(verb_phrase, "NP-PRD")
        determiner = None if nominal is None else self.child(nominal, "Det")
        return determiner is not None and self.gloss(determiner) == "no"

    def names_a_category(self, verb_phrase: Tree) -> bool:
        """Whether the verb phrase's predicate is a membership predicate: a predicate noun whose
        concept is a category."""
        nominal = self.child(verb_phrase, "NP-PRD")
        noun = None if nominal is None else self.child(nominal, "N")
        if noun is None:
            return False
        concept = self.lexicon.lexeme(token_parts(self.token(noun))[0]).concept
        return concept.startswith(CATEGORY_PREFIX)

    def walk(self, node: Tree) -> set[str]:
        """The readings that every verb phrase below a node allows."""
        allowed = {CAPACITY, EVENT, STATE, ABLE_NOW}
        previous: str | None = None
        for child in node[1:]:
            if isinstance(child, str):
                continue
            if child[0] == "VP":
                allowed &= self.verb_phrase(child, node[0] == "RC", previous)
                auxiliary = self.child(child, "AUX")
                previous = previous if auxiliary is None else self.gloss(auxiliary)
            allowed &= self.walk(child)
        return allowed

    def verb_phrase(self, node: Tree, in_clause: bool, previous: str | None) -> set[str]:
        """The readings that one verb phrase of a sentence about instances allows. ``previous``
        is the auxiliary of the verb phrases before it in a joined relative clause, which a verb
        phrase can leave out."""
        verb = self.child(node, "V")
        negated = self.child(node, "Neg") is not None
        auxiliary = self.child(node, "AUX")
        auxiliary_gloss = None if auxiliary is None else self.gloss(auxiliary)
        if auxiliary_gloss is None and previous is not None:
            auxiliary_gloss = previous
        if verb is None:
            adjective = self.child(node, "A")
            if adjective is not None:
                concept = self.lexicon.lexeme(token_parts(self.token(adjective))[0]).concept
                if concept.startswith(FLUENT_PREFIX) or auxiliary_gloss == "become":
                    return {STATE}
                if in_clause and negated and concept.startswith(PROPERTY_PREFIX):
                    # "that is not red" restricts its noun phrase, at any level
                    return {CAPACITY, EVENT, STATE, ABLE_NOW}
            return {CAPACITY}
        if auxiliary_gloss in self.auxiliary_readings:
            allowed = set(self.auxiliary_readings[auxiliary_gloss])
            if auxiliary is not None and self.marks(auxiliary) & _EVENT_MARKS:
                allowed.discard(CAPACITY)  # a capacity has no tense
            return allowed
        if negated or auxiliary is not None:
            return {CAPACITY}
        if self.marks(verb) & _EVENT_MARKS:
            return {EVENT}
        allowed = set()
        if self.bare_capacity or previous == "can":
            allowed.add(CAPACITY)
        if self.bare_event:
            allowed.add(EVENT)
        return allowed
