"""Readings: the kinds of logical form that a sentence's words allow.

The propositional rendering is never ambiguous, and the surface language can be. A sentence has
one of three kinds of logical form, one for each level:

- ``generic``: a class-level sentence, about a category ("penguins swim");
- ``capacity``: an instance-level sentence, about what an instance is, has, or can do ("the
  penguin can swim");
- ``event``: an event-level sentence, about what happened ("the penguin swam").

A noun phrase shows whether it names a category or an instance, so a class-level sentence has the
one reading ``generic``. A sentence about instances can be ambiguous between ``capacity`` and
``event``: with ``grammar.can_rate.instance`` below 1 a capacity can drop ``can``, and with the
tense and the aspect unmarked an event has a bare verb too. "The penguin swim" then has both
readings.

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
from semantic_world.corpus.lexicon import Lexicon
from semantic_world.corpus.realize import Tree, token_parts

GENERIC = "generic"
CAPACITY = "capacity"
EVENT = "event"
READINGS = (GENERIC, CAPACITY, EVENT)

_INSTANCE_DETERMINERS = ("a", "the")
_EVENT_MARKS = {"PAST", "PROGRESSIVE"}


def readings(tree: Tree, lexicon: Lexicon, config: Config) -> tuple[str, ...]:
    """The readings that a sentence's tree and lexemes allow, in the order ``generic``,
    ``capacity``, ``event``."""
    return _Reader(lexicon, config).sentence(tree)


class _Reader:
    def __init__(self, lexicon: Lexicon, config: Config) -> None:
        self.lexicon = lexicon
        morphology = config.grammar.morphology
        events = config.propositions
        self.bare_capacity = config.grammar.can_rate["instance"] < 1
        """Whether an instance's capacity can be said without ``can``."""
        self.bare_event = (not morphology.tense.enabled or events.event_tense == "present") and (
            not morphology.aspect.enabled or events.progressive_rate < 1
        )
        """Whether an event can be said with no marker on its verb."""

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
            return (GENERIC,)
        allowed = self.walk(tree)
        return tuple(reading for reading in READINGS if reading in allowed)

    def walk(self, node: Tree) -> set[str]:
        """The readings that every verb phrase below a node allows."""
        allowed = {CAPACITY, EVENT}
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
        if verb is None:
            adjective = self.child(node, "A")
            if in_clause and negated and adjective is not None:
                concept = self.lexicon.lexeme(token_parts(self.token(adjective))[0]).concept
                if concept.startswith("IS."):
                    # "that is not red" restricts its noun phrase, at any level
                    return {CAPACITY, EVENT}
            return {CAPACITY}
        if negated or self.child(node, "AUX") is not None:
            return {CAPACITY}
        if self.marks(verb) & _EVENT_MARKS:
            return {EVENT}
        allowed = set()
        if self.bare_capacity or previous == "can":
            allowed.add(CAPACITY)
        if self.bare_event:
            allowed.add(EVENT)
        return allowed
