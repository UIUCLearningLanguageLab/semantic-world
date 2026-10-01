"""Layer 4: what a sentence says, in the form the grammar realizes.

A :class:`SentencePlan` is the logical form of one sentence: a subject noun phrase and a
predication. It holds everything that the planner decides, and nothing that the grammar decides:

- a :class:`NounPhrase` names a referent. A class-level noun phrase names a category, with a
  quantifier or bare. An instance-level noun phrase names an instance, with the category its
  noun names, the determiner ``a`` or ``the``, or as a pronoun. Either can carry a restriction
  (adjectives, with-phrases, and negated IS literals) and one relative clause;
- a :class:`Predication` is the content of one verb phrase: a predicate, a polarity, an object
  noun phrase for a verb, and, at the event level, the event it reports with its tense and
  aspect;
- a :class:`RelativeClause` is a subject relative, with one or more predications about the head
  joined by "and", or an object relative, with the clause's own subject (``agent``) and one verb.

The grammar turns a plan into words and a tree (``realize``), and a tree back into the plan
(``interpret``). Word order, morphology, adjective order, the choice between synonyms, and the
optional ``can`` of a positive capacity are the grammar's own, and are not in the plan. The
tense and the aspect of an event are part of the plan: the morphology only decides whether and
how they are marked.

A relative clause on a class-level noun phrase is restrictive, and holds CAN features and verbs
only: "penguins that can swim", "owls that eat mice", and "mice that owls eat". It is part of
the category term that the truth tests judge (:func:`term_of`).

How a restriction is realized is fixed, so that a tree has one reading:

- a positive IS literal and a scalar pole are adjectives;
- a HAS literal is a with-phrase or a without-phrase;
- the negated IS literals share one relative clause, joined with "and" ("that are not red and
  not big"), and the predications of a subject relative join the same clause.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.propositions import (
    ALL,
    ASPECTS,
    CAN,
    CLASS,
    EVENT,
    GENERIC,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NO,
    PROJECTION,
    SCALAR,
    SOME,
    TENSES,
    VERB,
    CategoryTerm,
    Clause,
    Literal,
    Predicate,
    Proposition,
)

CLASS_NP = "class"
INSTANCE_NP = "instance"
INSTANCE_DETERMINERS = ("a", "the")
QUANTIFIER_WORDS = (ALL, MOST, SOME, NO)
CLAUSE_KINDS = (CAN, VERB, PROJECTION, MEMBER)
"""The predicates a relative clause can hold, apart from the negated IS literals of the
restriction. A positive IS literal, a scalar pole, and a HAS literal are never in a relative
clause: they are adjectives and with-phrases."""


class GrammarError(CorpusError):
    """A sentence plan that the grammar cannot realize, or a tree that it cannot read."""


@dataclass(frozen=True)
class NounPhrase:
    kind: str
    """``class`` or ``instance``."""
    referent: str
    """A category label or ``THING`` (class), or an instance label (instance)."""
    noun: str | None = None
    """The category whose noun heads the phrase. For a class-level phrase it is the referent.
    None for a pronoun."""
    determiner: str | None = None
    """``a`` or ``the`` (instance), a quantifier word (class), or None: a bare noun (the
    generic), or a pronoun."""
    restriction: tuple[Literal, ...] = ()
    clause: RelativeClause | None = None

    def __post_init__(self) -> None:
        ordered = tuple(sorted(set(self.restriction), key=lambda x: x.sort_key))
        object.__setattr__(self, "restriction", ordered)

    @property
    def pronoun(self) -> bool:
        return self.kind == INSTANCE_NP and self.noun is None

    @property
    def negated(self) -> tuple[Literal, ...]:
        """The negated IS literals, which go into the relative clause."""
        return tuple(x for x in self.restriction if not x.positive and x.feature.startswith("IS."))

    @property
    def adjectives(self) -> tuple[Literal, ...]:
        """The positive IS literals and the scalar poles."""
        return tuple(x for x in self.restriction if x.positive and not x.feature.startswith("HAS."))

    @property
    def with_phrases(self) -> tuple[Literal, ...]:
        return tuple(x for x in self.restriction if x.feature.startswith("HAS."))


@dataclass(frozen=True)
class Predication:
    kind: str
    """``is``, ``has``, ``can``, ``scalar``, ``member``, ``projection``, or ``verb``."""
    label: str
    polarity: bool = True
    object: NounPhrase | None = None
    """Verbs only: the patient. None in an object relative, where the head noun is the
    patient."""
    event: str | None = None
    """Event level only: the label of the event that the verb phrase reports."""
    tense: str | None = None
    """Event level only: ``past`` or ``present``."""
    aspect: str | None = None
    """Event level only: ``simple`` or ``progressive``."""


@dataclass(frozen=True)
class RelativeClause:
    predications: tuple[Predication, ...]
    agent: NounPhrase | None = None
    """An object relative ("the mouse that the owl chased"): the clause's own subject. The
    clause then has one predication, a verb whose patient is the head noun."""

    @property
    def object_relative(self) -> bool:
        return self.agent is not None


@dataclass(frozen=True)
class SentencePlan:
    subject: NounPhrase
    predication: Predication

    @property
    def level(self) -> str:
        if self.subject.kind == CLASS_NP:
            return CLASS
        return EVENT if self.predication.event is not None else INSTANCE

    def proposition(self) -> Proposition:
        """The proposition of the main clause, as the truth tests judge it: the subject, the
        predicate, the quantifier, and the polarity, without the mentions and the relative
        clauses."""
        predication = self.predication
        target = predication.object
        if self.subject.kind == CLASS_NP:
            patient = None if target is None else term_of(target)
            return Proposition(
                CLASS,
                term_of(self.subject),
                Predicate(predication.kind, predication.label, patient),
                predication.polarity,
                self.subject.determiner or GENERIC,
            )
        patient = None if target is None else target.referent
        if predication.event is not None:
            return Proposition(
                EVENT,
                self.subject.referent,
                Predicate(predication.kind, predication.label, patient),
                scene=predication.event.rsplit(".", 1)[0],
                event=predication.event,
                tense=predication.tense,
                aspect=predication.aspect,
            )
        comparison = self.subject.noun if predication.kind == SCALAR else None
        return Proposition(
            INSTANCE,
            self.subject.referent,
            Predicate(predication.kind, predication.label, patient, comparison),
            predication.polarity,
        )

    def noun_phrases(self) -> list[NounPhrase]:
        """Every noun phrase of the sentence, relative clauses included."""
        found: list[NounPhrase] = []

        def visit(phrase: NounPhrase | None) -> None:
            if phrase is None:
                return
            found.append(phrase)
            if phrase.clause is not None:
                visit(phrase.clause.agent)
                for predication in phrase.clause.predications:
                    visit(predication.object)

        visit(self.subject)
        visit(self.predication.object)
        return found

    def depth(self) -> int:
        """The deepest nesting of relative clauses: 0 without one, 1 for a relative clause on a
        noun phrase of the main clause, 2 when a noun phrase inside it has one, and so on. The
        negated IS literals of a restriction make a relative clause too."""
        return max(_depth(self.subject), _depth(self.predication.object))


def term_of(phrase: NounPhrase) -> CategoryTerm:
    """The category term of a class-level noun phrase: its category, its restriction, and its
    relative clause as restrictive clauses."""
    clauses: tuple[Clause, ...] = ()
    clause = phrase.clause
    if clause is not None and clause.agent is not None:
        clauses = (Clause(VERB, clause.predications[0].label, agent=term_of(clause.agent)),)
    elif clause is not None:
        clauses = tuple(
            Clause(p.kind, p.label, patient=None if p.object is None else term_of(p.object))
            for p in clause.predications
        )
    return CategoryTerm(phrase.referent, phrase.restriction, clauses)


def phrase_of(term: CategoryTerm, determiner: str | None = None) -> NounPhrase:
    """The class-level noun phrase of a category term. Its clauses make one relative clause: an
    object relative for a clause with an agent category, and a subject relative otherwise."""
    clause: RelativeClause | None = None
    if any(c.agent is not None for c in term.clauses):
        if len(term.clauses) != 1:
            raise GrammarError("an object relative is the only relative clause of its noun phrase")
        only = term.clauses[0]
        assert only.agent is not None
        clause = RelativeClause((Predication(VERB, only.label),), phrase_of(only.agent))
    elif term.clauses:
        clause = RelativeClause(
            tuple(
                Predication(
                    c.kind, c.label, True, None if c.patient is None else phrase_of(c.patient)
                )
                for c in term.clauses
            )
        )
    return NounPhrase(CLASS_NP, term.category, term.category, determiner, term.restriction, clause)


def _depth(phrase: NounPhrase | None) -> int:
    if phrase is None:
        return 0
    if phrase.clause is None:
        return 1 if phrase.negated else 0
    inner = [_depth(phrase.clause.agent)] + [_depth(p.object) for p in phrase.clause.predications]
    return 1 + max(inner)


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def check_plan(plan: SentencePlan) -> None:
    """Stop with a :class:`GrammarError` when a plan is not one the grammar realizes. The
    planner makes only plans that pass."""
    subject, predication = plan.subject, plan.predication
    if subject.kind == CLASS_NP and predication.event is not None:
        raise GrammarError("an event is about an instance, not about a category")
    _check_phrase(subject, plan.level, top=True)
    _check_predication(predication, subject, plan.level, in_clause=False)
    if predication.kind == MEMBER and predication.label == subject.noun:
        raise GrammarError(
            f"the subject is already named as a {subject.noun}: name it by another category"
        )
    if predication.kind == SCALAR and subject.pronoun:
        raise GrammarError(
            "a pronoun names no category, so a scalar pole has no comparison class: mention the "
            "subject with a noun"
        )


def _check_phrase(phrase: NounPhrase, level: str, top: bool = False) -> None:
    if phrase.kind not in (CLASS_NP, INSTANCE_NP):
        raise GrammarError(f"unknown kind of noun phrase {phrase.kind!r}")
    if (phrase.kind == CLASS_NP) != (level == CLASS):
        raise GrammarError(
            "a class-level sentence is about categories, and the other levels about instances"
        )
    if phrase.kind == CLASS_NP:
        if phrase.noun != phrase.referent:
            raise GrammarError("the noun of a class-level noun phrase names its own category")
        allowed: tuple = QUANTIFIER_WORDS + (None,) if top else (None,)
        if phrase.determiner not in allowed:
            raise GrammarError(
                f"{phrase.determiner!r} is not a determiner for this class-level noun phrase: "
                "only the subject takes a quantifier"
            )
    elif phrase.pronoun:
        if phrase.determiner is not None or phrase.restriction or phrase.clause is not None:
            raise GrammarError("a pronoun takes no determiner, modifier, or relative clause")
    elif phrase.determiner not in INSTANCE_DETERMINERS:
        raise GrammarError(f"an instance is mentioned with a or the, not {phrase.determiner!r}")
    for literal in phrase.restriction:
        if literal.pole and not literal.positive:
            raise GrammarError("a scalar pole in a restriction is never negated")
    clause = phrase.clause
    if clause is None:
        return
    if not clause.predications:
        raise GrammarError("a relative clause says something about its head")
    if clause.object_relative:
        assert clause.agent is not None
        if phrase.negated:
            raise GrammarError(
                "an object relative cannot join the negated IS literals: a noun phrase has one "
                "relative clause"
            )
        predication = clause.predications[0]
        if (
            len(clause.predications) != 1
            or predication.kind != VERB
            or predication.object is not None
        ):
            raise GrammarError("an object relative has one verb, whose patient is the head noun")
        _check_phrase(clause.agent, level)
        _check_predication(predication, clause.agent, level, in_clause=True, gap=True)
        return
    for predication in clause.predications:
        _check_predication(predication, phrase, level, in_clause=True)


def _check_predication(
    predication: Predication,
    subject: NounPhrase,
    level: str,
    in_clause: bool,
    gap: bool = False,
) -> None:
    kind = predication.kind
    if in_clause and kind not in CLAUSE_KINDS:
        raise GrammarError(
            f"a relative clause does not hold a predicate of kind {kind!r}: an IS literal, a "
            "scalar pole, and a HAS literal belong in the restriction"
        )
    if kind not in (IS, HAS, CAN, SCALAR, MEMBER, PROJECTION, VERB):
        raise GrammarError(f"unknown predicate kind {kind!r}")
    if in_clause and level == CLASS and (kind not in (CAN, VERB) or not predication.polarity):
        raise GrammarError(
            "a class-level relative clause holds a CAN feature or a verb, and is never negated"
        )
    if (kind == VERB) != (predication.object is not None or gap):
        raise GrammarError("a verb, and only a verb, has an object")
    if (predication.event is not None) != (level == EVENT):
        raise GrammarError(
            "every verb phrase of an event-level sentence reports an event, and no other does"
        )
    if predication.event is not None:
        if kind not in (CAN, VERB):
            raise GrammarError("an event is a CAN feature, or a verb with a patient")
        if not predication.polarity:
            raise GrammarError("an event-level proposition is never negated")
        if predication.tense not in TENSES or predication.aspect not in ASPECTS:
            raise GrammarError(
                "an event has a tense (past or present) and an aspect (simple or progressive)"
            )
    elif predication.tense is not None or predication.aspect is not None:
        raise GrammarError("only an event has a tense and an aspect")
    if predication.object is not None:
        _check_phrase(predication.object, level)
