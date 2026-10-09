"""Layer 4: what a sentence says, in the form the grammar realizes.

A :class:`SentencePlan` is the logical form of one sentence: a subject noun phrase and a
predication. It holds everything that the planner decides, and nothing that the grammar decides:

- a :class:`NounPhrase` names a referent. A class-level noun phrase names a category, with a
  quantifier word or bare. An instance-level noun phrase names an instance, with the category
  its noun names, the determiner ``a`` or ``the``, or as a pronoun. Either can carry a
  restriction (adjectives, with-phrases, and negated PROPERTY literals) and one relative clause;
- a :class:`Predication` is the content of one verb phrase: a predicate, a polarity, an object
  noun phrase for a verb, and, at the event level, the event it reports with its tense and
  aspect. At the state, change, and able_now levels it holds the time point it is about
  (``SCENE.8.TIME.2``) and the tense, and a change says ``become``;
- a :class:`RelativeClause` is a subject relative, with one or more predications about the head
  joined by "and", or an object relative, with the clause's own subject (``agent``) and one verb.

The grammar turns a plan into words and a tree (``realize``), and a tree back into the plan
(``interpret``). Word order, morphology, adjective order, the choice between synonyms, and the
optional ``can`` of a positive capacity are the grammar's own, and are not in the plan. The
tense and the aspect of a report are part of the plan: the morphology only decides whether and
how they are marked. So is the quantifier of a class-level sentence: the plan holds the
proposition's quantifier (``nec_all``, ``all``, ``most``, ``some``, ``no``, ``nec_no``, or none
for a scalar pole) apart from the subject's determiner word ("all", "most", "some", "no", or
none for a bare plural), because what a word expresses is a setting of the language.

A relative clause on a class-level noun phrase is restrictive, and holds one-place and two-place
event types only: "penguins that can swim", "owls that eat mice", and "mice that owls eat". It is
part of the category term that the truth tests judge (:func:`term_of`).

How a restriction is realized is fixed, so that a tree has one reading:

- a positive IS literal and a scalar pole are adjectives;
- a HAS literal is a with-phrase or a without-phrase;
- the negated PROPERTY literals share one relative clause, joined with "and" ("that are not
  red and not big"), and the predications of a subject relative join the same clause.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.histories import split_time_key
from semantic_world.corpus.propositions import (
    ABLE_NOW,
    ALL,
    ASPECTS,
    CAN,
    CHANGE,
    CLASS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NO,
    PROJECTION,
    QUANTIFIERS,
    SCALAR,
    SOME,
    STATE,
    STATE_KIND,
    TENSES,
    VERB,
    CategoryTerm,
    Clause,
    Literal,
    Predicate,
    Proposition,
    event_of,
    scene_of,
)
from semantic_world.corpus.world import PART_PREFIX, PROPERTY_PREFIX

CLASS_NP = "class"
INSTANCE_NP = "instance"
INSTANCE_DETERMINERS = ("a", "the")
QUANTIFIER_WORDS = ("all", "most", "some", "no")
"""The determiner words of a class-level subject. Which quantifiers "all" and "no" express is
a language setting; "most" and "some" express ``most`` and ``some``."""
WORD_OF = {NEC_ALL: "all", ALL: "all", MOST: "most", SOME: "some", NO: "no", NEC_NO: "no"}
"""The determiner word that states each quantifier."""
CLAUSE_KINDS = (CAN, VERB, PROJECTION, MEMBER)
"""The predicates a relative clause can hold, apart from the negated PROPERTY literals of the
restriction. A positive PROPERTY literal, a scalar pole, and a PART literal are never in a
relative clause: they are adjectives and with-phrases."""


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
    """``a`` or ``the`` (instance), a quantifier word (class), or None: a bare noun (a bare
    plural), or a pronoun."""
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
        """The negated PROPERTY literals, which go into the relative clause."""
        return tuple(
            x for x in self.restriction if not x.positive and x.feature.startswith(PROPERTY_PREFIX)
        )

    @property
    def adjectives(self) -> tuple[Literal, ...]:
        """The positive PROPERTY literals and the scalar poles."""
        return tuple(
            x for x in self.restriction if x.positive and not x.feature.startswith(PART_PREFIX)
        )

    @property
    def with_phrases(self) -> tuple[Literal, ...]:
        return tuple(x for x in self.restriction if x.feature.startswith(PART_PREFIX))


@dataclass(frozen=True)
class Predication:
    kind: str
    """``property``, ``part``, ``event_type1``, ``scalar``, ``member``, ``patient_capacity``,
    ``event_type2``, or ``state``."""
    label: str
    polarity: bool = True
    object: NounPhrase | None = None
    """Verbs only: the patient. None in an object relative, where the head noun is the
    patient."""
    event: str | None = None
    """Event level only: the label of the event that the verb phrase reports
    (``SCENE.8.EVENTINSTANCE.5``). In a test item it is the label of the scene (``SCENE.8``):
    some event of the scene."""
    tense: str | None = None
    """Event, state, change, and able_now levels: ``past`` or ``present``."""
    aspect: str | None = None
    """Event level only: ``simple`` or ``progressive``."""
    time: str | None = None
    """State, change, and able_now levels: the time point the verb phrase is about, qualified
    by its scene (``SCENE.8.TIME.2``)."""
    become: bool = False
    """A change: the fluent came to have the value at the time point (``become``), rather than
    held it (the copula)."""

    @property
    def timed(self) -> bool:
        return self.time is not None

    @property
    def timed_level(self) -> str | None:
        """``able_now``, ``change``, or ``state`` for a predication about a time point."""
        if self.time is None:
            return None
        if self.kind in (CAN, VERB):
            return ABLE_NOW
        return CHANGE if self.become else STATE


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
    quantifier: str | None = None
    """Class level only: the proposition's quantifier. None for a class-level scalar pole, and
    for a sentence about instances. The subject's determiner is the word that states it, or
    None for a bare plural."""

    @property
    def level(self) -> str:
        if self.subject.kind == CLASS_NP:
            return CLASS
        if self.predication.event is not None:
            return EVENT
        timed = self.predication.timed_level
        return INSTANCE if timed is None else timed

    @property
    def bare_plural(self) -> bool:
        return self.subject.kind == CLASS_NP and self.subject.determiner is None

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
                self.quantifier,
            )
        patient = None if target is None else target.referent
        if predication.event is not None:
            return Proposition(
                EVENT,
                self.subject.referent,
                Predicate(predication.kind, predication.label, patient),
                scene=scene_of(predication.event),
                event=event_of(predication.event),
                tense=predication.tense,
                aspect=predication.aspect,
            )
        if predication.time is not None:
            scene, time = split_time_key(predication.time)
            return Proposition(
                self.level,
                self.subject.referent,
                Predicate(predication.kind, predication.label, patient),
                predication.polarity,
                scene=scene,
                time=time,
                tense=predication.tense,
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
    if subject.kind == CLASS_NP:
        quantifier = plan.quantifier
        if predication.kind == SCALAR:
            if quantifier is not None or subject.determiner is not None:
                raise GrammarError("a class-level scalar pole takes no quantifier")
        elif quantifier not in QUANTIFIERS:
            raise GrammarError(f"a class-level sentence has a quantifier, not {quantifier!r}")
        elif subject.determiner is not None and subject.determiner != WORD_OF[quantifier]:
            raise GrammarError(
                f"the word {subject.determiner!r} does not state the quantifier {quantifier}"
            )
    elif plan.quantifier is not None:
        raise GrammarError("only a class-level sentence has a quantifier")
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
    if kind not in (IS, HAS, CAN, SCALAR, MEMBER, PROJECTION, VERB, STATE_KIND):
        raise GrammarError(f"unknown predicate kind {kind!r}")
    if in_clause and predication.time is not None:
        raise GrammarError("a relative clause is never about a time point")
    if (kind == STATE_KIND) != (predication.time is not None and kind not in (CAN, VERB)):
        raise GrammarError(
            "a fluent is stated at a time point of a scene (a state or a change), and a time "
            "point belongs to a fluent or to what was possible then (able_now)"
        )
    if predication.become and kind != STATE_KIND:
        raise GrammarError("only a change of a fluent says become")
    if predication.time is not None:
        if level not in (STATE, CHANGE, ABLE_NOW) or subject.kind != INSTANCE_NP:
            raise GrammarError("a sentence about a time point is about an instance")
        if predication.event is not None or predication.aspect is not None:
            raise GrammarError("a sentence about a time point reports no event and has no aspect")
        if predication.tense not in TENSES:
            raise GrammarError("a sentence about a time point has a tense (past or present)")
        try:
            split_time_key(predication.time)
        except ValueError as error:
            raise GrammarError(str(error)) from None
    if in_clause and level == CLASS and (kind not in (CAN, VERB) or not predication.polarity):
        raise GrammarError(
            "a class-level relative clause holds a one-place or a two-place event type, and is "
            "never negated"
        )
    if (kind == VERB) != (predication.object is not None or gap):
        raise GrammarError("a verb, and only a verb, has an object")
    if (predication.event is not None) != (level == EVENT):
        raise GrammarError(
            "every verb phrase of an event-level sentence reports an event, and no other does"
        )
    if predication.event is not None:
        if kind not in (CAN, VERB):
            raise GrammarError(
                "an event is a one-place event type, or a two-place event type with a patient"
            )
        if not predication.polarity:
            raise GrammarError("an event-level proposition is never negated")
        if predication.tense not in TENSES or predication.aspect not in ASPECTS:
            raise GrammarError(
                "a report has a tense (past or present) and an aspect (simple or progressive)"
            )
    elif predication.time is None and (
        predication.tense is not None or predication.aspect is not None
    ):
        raise GrammarError(
            "only a report of an event, or a sentence about a time point, has a tense"
        )
    if predication.object is not None:
        _check_phrase(predication.object, level)
