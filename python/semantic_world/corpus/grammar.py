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
  (``SCENE.8.TIME.2``) and the tense, and a change says ``become``. A comparative ("bigger
  than") is a predication whose label is the pole adjective the sentence uses and whose object
  is the standard of comparison; a class-level pole holds its comparison class (``standard``),
  which the words do not say (stage b3a);
- a :class:`RelativeClause` is a subject relative, with one or more predications about the head
  joined by "and", or an object relative, with the clause's own subject (``agent``) and one verb.

A **causal statement** ("things that things catch become caught") is a class-level plan whose
subject is the generic noun with a relative clause that names an event: its one predication is a
verb whose ``event`` is the event variable ``EVENTVAR.1``, in the present tense, with the bare
generic noun as its other participant. Its predication is an ``effect`` (``become`` with the
state adjective) or a ``precondition`` (the copula with the state adjective, and the function
word ``before``), whose polarity is the fluent's value, and its quantifier is ``nec_all``.

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

**A comparative** ("the owl is bigger than the mouse") is the copula with the comparative form of
a pole adjective and the function word ``than`` before (or after) the object. Its proposition is
``GREATER`` over the dimension, with the subject first for the high pole's adjective ("bigger")
and the object first for the low pole's ("smaller"): the language chooses the word, and the
proposition stays one form. A comparative is never negated and never in a relative clause.

**Numeric fluents** (stage b3b). A numeric pole adjective (``NUMFL.1.HIGH``) is stated at a
time point like a Boolean fluent: a predication of kind ``comparison`` with a ``time`` ("the owl
was hungry"), whose comparison class is the fluent's range. Its comparative form with ``become``
is a change (kind ``state``, the pole as its label, ``become``: "the owl became hungrier", an
``increase`` for the high pole and a ``decrease`` for the low one), with the copula and ``than``
a comparative at a time point (kind ``comparative`` with a ``time``: "the owl was hungrier than
the mouse"), and with a category as the subject an effect statement ("things that eat things
become fuller", kind ``effect`` with the pole as its label: ``raises`` for the high pole and
``lowers`` for the low one).

**Conditional causal statements** (stage b3b). The condition of a conditional effect is
realized on the subject noun phrase: a static literal of the subject's role as a modifier ("red
things that things strike", "things with wings that strike things"), a negated PROPERTY literal
in the subject relative ("things that are not red and strike things"), a fluent literal of the
subject's role as a predication of kind ``precondition`` joined to the subject relative ("things
that strike things and are awake before"), and a literal of the other role on the bare "things"
inside the relative clause, as a modifier or as a relative clause of its own ("things that strike
red things", "things that strike things that are asleep before"). An object relative holds one
verb, so a negated PROPERTY literal or a fluent literal of the patient's role has no place in a
statement about the patient, and such a statement is not made.

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
    AGENT,
    ALL,
    ASPECTS,
    CAN,
    CAUSAL_KINDS,
    CHANGE,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    COMPARISON_KINDS,
    EFFECT,
    EVENT,
    EVENT_VARIABLE,
    HAS,
    HIGH,
    INSTANCE,
    IS,
    LOW,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NO,
    PATIENT,
    PRECONDITION,
    PRESENT,
    PROJECTION,
    QUANTIFIERS,
    SOME,
    STATE,
    STATE_KIND,
    TENSES,
    VERB,
    CategoryTerm,
    Clause,
    Condition,
    EventTerm,
    Literal,
    Predicate,
    Proposition,
    event_of,
    is_event_variable,
    is_numeric,
    pole_dimension,
    pole_direction,
    scene_of,
)
from semantic_world.corpus.world import (
    FLUENT_PREFIX,
    PART_PREFIX,
    PROPERTY_PREFIX,
    SCALAR_PREFIX,
    THING,
)
from semantic_world.world.runtime import RANGE

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
    """``property``, ``part``, ``event_type1``, ``comparison``, ``comparative``, ``member``,
    ``patient_capacity``, ``event_type2``, ``state``, ``effect``, or ``precondition``."""
    label: str
    """The predicate word's concept. For a comparative, the pole adjective the sentence uses
    (``SCALARDIM.1.HIGH`` for "bigger than", ``SCALARDIM.1.LOW`` for "smaller than"); for a
    numeric change or a numeric effect statement, the numeric pole whose comparative form says
    the direction (``NUMFL.1.HIGH`` for "became hungrier" and "become hungrier")."""
    polarity: bool = True
    """For an effect or a precondition, the fluent's value."""
    object: NounPhrase | None = None
    """Verbs and comparatives only: the patient, or the standard of a comparison ("than the
    mouse"). None in an object relative, where the head noun is the patient."""
    event: str | None = None
    """Event level only: the label of the event that the verb phrase reports
    (``SCENE.8.EVENTINSTANCE.5``). In a test item it is the label of the scene (``SCENE.8``):
    some event of the scene. In the relative clause of a causal statement it is the event
    variable (``EVENTVAR.1``): the events of the event type."""
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
    standard: str | None = None
    """A class-level pole only: its comparison class, an ancestor of the subject's category or
    ``ENTITY``, which the words do not say ("owls are big" compares owls with their parent
    category, or with every entity)."""

    @property
    def timed(self) -> bool:
        return self.time is not None

    @property
    def low(self) -> bool:
        """For a comparative: whether the sentence uses the low pole's adjective ("smaller
        than"), so that the object is the greater argument."""
        return self.kind == COMPARATIVE and self.label.rsplit(".", 1)[1] == LOW

    @property
    def numeric(self) -> bool:
        """Whether the predicate word is about a numeric fluent (stage b3b)."""
        return is_numeric(self.label)

    @property
    def comparative_form(self) -> bool:
        """Whether the predicate word takes its comparative form: a comparative, a numeric
        change ("became hungrier"), or a numeric effect statement ("become hungrier")."""
        return self.kind == COMPARATIVE or (
            self.numeric and (self.kind == EFFECT or (self.kind == STATE_KIND and self.become))
        )

    @property
    def direction(self) -> str | None:
        """The direction of a numeric change or effect statement, from the pole adjective."""
        if self.numeric and self.kind == EFFECT:
            return pole_direction(self.label, EFFECT)
        if self.numeric and self.kind == STATE_KIND and self.become:
            return pole_direction(self.label, STATE_KIND)
        return None

    @property
    def bound(self) -> bool:
        """Whether the verb phrase names the events of an event type, bound to an event
        variable: the relative clause of a causal statement."""
        return is_event_variable(self.event)

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
    """Class level only: the proposition's quantifier. None for a class-level comparison or
    comparative, and for a sentence about instances. The subject's determiner is the word that
    states it, or None for a bare plural."""

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

    @property
    def causal(self) -> bool:
        """Whether the plan is a causal statement: an effect or a precondition."""
        return self.predication.kind in CAUSAL_KINDS

    def proposition(self) -> Proposition:
        """The proposition of the main clause, as the truth tests judge it: the subject, the
        predicate, the quantifier, and the polarity, without the mentions and the relative
        clauses."""
        predication = self.predication
        target = predication.object
        if self.causal:
            if predication.numeric:
                predicate = Predicate(
                    EFFECT, pole_dimension(predication.label), direction=predication.direction
                )
            else:
                predicate = Predicate(
                    predication.kind, predication.label, value=predication.polarity
                )
            return Proposition(CLASS, event_term_of(self.subject), predicate, True, NEC_ALL)
        if predication.kind == COMPARATIVE:
            return self._comparative()
        if self.subject.kind == CLASS_NP:
            patient = None if target is None else term_of(target)
            comparison = predication.standard if predication.kind == COMPARISON else None
            return Proposition(
                CLASS,
                term_of(self.subject),
                Predicate(predication.kind, predication.label, patient, comparison),
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
            if predication.kind == COMPARISON:
                # a numeric pole at a time point: its class is the fluent's range
                predicate = Predicate(COMPARISON, predication.label, comparison=RANGE)
            elif predication.direction is not None:
                predicate = Predicate(
                    STATE_KIND, pole_dimension(predication.label), direction=predication.direction
                )
            else:
                predicate = Predicate(predication.kind, predication.label, patient)
            return Proposition(
                self.level,
                self.subject.referent,
                predicate,
                predication.polarity,
                scene=scene,
                time=time,
                tense=predication.tense,
            )
        comparison = self.subject.noun if predication.kind == COMPARISON else None
        return Proposition(
            INSTANCE,
            self.subject.referent,
            Predicate(predication.kind, predication.label, patient, comparison),
            predication.polarity,
        )

    def _comparative(self) -> Proposition:
        """The proposition of a comparative: ``GREATER`` over the dimension, with the subject
        first for the high pole's word and the object first for the low pole's."""
        predication = self.predication
        target = predication.object
        if target is None:
            raise GrammarError("a comparative has an object: the standard of comparison")
        dimension = pole_dimension(predication.label)
        if self.subject.kind == CLASS_NP:
            first: CategoryTerm | str = term_of(self.subject)
            second: CategoryTerm | str = term_of(target)
        else:
            first, second = self.subject.referent, target.referent
        if predication.low:
            first, second = second, first
        if predication.time is not None:
            # a numeric comparative at a time point (stage b3b): a state of the scene
            scene, time = split_time_key(predication.time)
            return Proposition(
                STATE,
                first,
                Predicate(COMPARATIVE, dimension, second),
                True,
                scene=scene,
                time=time,
                tense=predication.tense,
            )
        level = CLASS if self.subject.kind == CLASS_NP else INSTANCE
        return Proposition(level, first, Predicate(COMPARATIVE, dimension, second), True, None)

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


def bound_predication(clause: RelativeClause) -> Predication:
    """The one predication of the relative clause of a causal statement's subject that is bound
    to the event variable."""
    bound = [p for p in clause.predications if p.bound]
    if len(bound) != 1:
        raise GrammarError(
            "the relative clause of a causal statement has one verb, bound to the event variable"
        )
    return bound[0]


def _conditions_of(phrase: NounPhrase, role: str) -> list[Condition]:
    """The condition literals that a noun phrase of a causal statement's subject says of one
    role (stage b3b): its static modifiers and negated PROPERTY literals, and the ``before``
    predications of its relative clause."""
    found = [Condition(role, x.feature, x.positive) for x in phrase.restriction]
    if phrase.clause is not None:
        for predication in phrase.clause.predications:
            if predication.kind == PRECONDITION:
                found.append(Condition(role, predication.label, predication.polarity))
    return found


def event_term_of(phrase: NounPhrase) -> EventTerm:
    """The event term of the subject of a causal statement: the event type that its relative
    clause names, the head's role, the patient in an object relative and the agent otherwise,
    and the condition of a conditional statement (stage b3b): the literals said of the head
    and of the other participant."""
    clause = phrase.clause
    if phrase.kind != CLASS_NP or phrase.referent != THING or clause is None:
        raise GrammarError(
            "the subject of a causal statement is the generic noun with a relative clause that "
            "names the event"
        )
    bound = bound_predication(clause)
    role = PATIENT if clause.object_relative else AGENT
    other_role = AGENT if clause.object_relative else PATIENT
    other = clause.agent if clause.object_relative else bound.object
    condition = _conditions_of(phrase, role)
    if other is not None:
        condition += _conditions_of(other, other_role)
    return EventTerm(bound.label, role, tuple(condition))


def bare_things() -> NounPhrase:
    """The bare generic noun, "things": the other participant in the relative clause of a
    causal statement."""
    return NounPhrase(CLASS_NP, THING, THING)


def _things_under(condition: tuple[Condition, ...], role: str) -> NounPhrase:
    """The bare generic noun with the condition literals of one role on it (stage b3b): the
    static literals as its restriction, and the fluent literals as ``before`` predications of
    a relative clause."""
    literals = tuple(
        Literal(c.label, c.value) for c in condition if c.role == role and not c.fluent
    )
    before = tuple(
        Predication(PRECONDITION, c.label, c.value)
        for c in condition
        if c.role == role and c.fluent
    )
    clause = RelativeClause(before) if before else None
    return NounPhrase(CLASS_NP, THING, THING, None, literals, clause)


def event_phrase_of(term: EventTerm, determiner: str | None = None) -> NounPhrase:
    """The subject noun phrase of a causal statement: the generic noun with "all" or bare, and
    a relative clause naming the events of the term's event type, in which the head is the
    agent (a subject relative: "things that catch things", "things that sleep") or the patient
    (an object relative: "things that things catch"). The condition of a conditional statement
    (stage b3b) goes on the head (its static literals as modifiers, its fluent literals as
    ``before`` predications joined to the subject relative) and on the other participant."""
    kind = CAN if term.event_type.startswith("EVENTTYPE1.") else VERB
    head = _things_under(term.condition, term.role)
    other_role = AGENT if term.role == PATIENT else PATIENT
    if term.role == PATIENT:
        bound = Predication(VERB, term.event_type, True, None, EVENT_VARIABLE, PRESENT)
        if head.clause is not None:
            raise GrammarError(
                "an object relative holds one verb: a fluent condition of the patient has no "
                "place in a statement about the patient"
            )
        clause = RelativeClause((bound,), _things_under(term.condition, other_role))
    else:
        target = _things_under(term.condition, other_role) if kind == VERB else None
        bound = Predication(kind, term.event_type, True, target, EVENT_VARIABLE, PRESENT)
        before = head.clause.predications if head.clause is not None else ()
        clause = RelativeClause((bound, *before))
    return NounPhrase(CLASS_NP, THING, THING, determiner, head.restriction, clause)


def realizable_condition(term: EventTerm) -> bool:
    """Whether the grammar can realize a causal statement's condition (stage b3b): a statement
    about the patient is an object relative, which holds one verb and joins no negated
    PROPERTY literal, so a fluent literal or a negated PROPERTY literal of the patient's role
    has no place in it. Every other condition has a place."""
    if term.role != PATIENT:
        return True
    return not any(
        c.role == PATIENT and (c.fluent or (not c.value and c.label.startswith(PROPERTY_PREFIX)))
        for c in term.condition
    )


def phrase_of(term: CategoryTerm | EventTerm, determiner: str | None = None) -> NounPhrase:
    """The class-level noun phrase of a category term. Its clauses make one relative clause: an
    object relative for a clause with an agent category, and a subject relative otherwise. An
    event term gives the subject of a causal statement."""
    if isinstance(term, EventTerm):
        return event_phrase_of(term, determiner)
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
    if plan.causal:
        _check_causal(plan)
        _check_phrase(subject, CLASS, top=True, causal=True)
        return
    if subject.kind == CLASS_NP:
        quantifier = plan.quantifier
        if predication.kind in COMPARISON_KINDS:
            if predication.numeric:
                raise GrammarError(
                    "a numeric fluent is compared at a time point of a scene, never of a category"
                )
            if quantifier is not None or subject.determiner is not None:
                raise GrammarError(f"a class-level {predication.kind} takes no quantifier")
            if predication.kind == COMPARISON and predication.standard is None:
                raise GrammarError("a class-level pole names its comparison class (standard)")
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
    if predication.kind == COMPARISON and subject.pronoun and not predication.numeric:
        raise GrammarError(
            "a pronoun names no category, so a scalar pole has no comparison class: mention the "
            "subject with a noun"
        )
    if predication.kind == COMPARISON and subject.kind == INSTANCE_NP and predication.standard:
        raise GrammarError("an instance-level pole takes its comparison class from the noun")
    if predication.kind == COMPARISON and predication.numeric and predication.time is None:
        raise GrammarError("a numeric pole is stated at a time point of a scene")
    if predication.kind == COMPARATIVE:
        _check_comparative(plan)


def _check_comparative(plan: SentencePlan) -> None:
    """A comparative: the comparative form of a pole adjective, a positive polarity, an object
    of the subject's kind (an instance with an instance, a plain category with a plain
    category), and no time point; a numeric comparative (stage b3b) compares two instances at
    a time point."""
    subject, predication = plan.subject, plan.predication
    target = predication.object
    label = predication.label
    scalar = label.startswith(SCALAR_PREFIX) or is_numeric(label)
    if not scalar or label.rsplit(".", 1)[1] not in (HIGH, LOW):
        raise GrammarError("a comparative is the comparative form of a pole adjective")
    if not predication.polarity:
        raise GrammarError("a comparative is never negated: say it the other way round")
    if target is None or target.kind != subject.kind:
        raise GrammarError(
            "a comparative compares an instance with an instance, or a category with a category"
        )
    if predication.numeric:
        if predication.time is None or subject.kind != INSTANCE_NP:
            raise GrammarError(
                "a numeric comparative compares two instances at a time point of a scene"
            )
        if predication.event is not None or predication.standard:
            raise GrammarError("a numeric comparative reports no event and has no class")
    elif predication.time is not None or predication.event is not None or predication.standard:
        raise GrammarError("a comparative is timeless and has no comparison class")
    if target.kind == CLASS_NP and (
        target.restriction
        or target.clause is not None
        or subject.restriction
        or subject.clause is not None
    ):
        raise GrammarError("a class-level comparative compares the means of plain categories")

    if target.referent == subject.referent:
        raise GrammarError("a thing is not compared with itself")


def _check_causal(plan: SentencePlan) -> None:
    """A causal statement: the generic noun with one event-bound relative clause, ``nec_all``,
    said with "all" or bare, and an effect or a precondition with no object, no event, and no
    time point. The subject's modifiers and ``before`` predications, and those of the other
    participant, are the condition of a conditional effect statement (stage b3b)."""
    subject, predication = plan.subject, plan.predication
    if plan.quantifier != NEC_ALL:
        raise GrammarError("a causal statement is nec_all: the definition guarantees it")
    if subject.determiner not in (WORD_OF[NEC_ALL], None):
        raise GrammarError("a causal statement says all, or uses the bare plural")
    if (
        predication.object is not None
        or predication.time is not None
        or predication.tense is not None
        or predication.aspect is not None
        or predication.become
    ):
        raise GrammarError(
            "an effect or a precondition has no object, no event, no time point, and no tense: "
            "a causal statement is timeless"
        )
    if predication.numeric and predication.kind != EFFECT:
        raise GrammarError(
            "a precondition on a numeric fluent is a threshold, which no word states: only an "
            "effect statement is about a numeric fluent"
        )
    if predication.numeric and predication.label.rsplit(".", 1)[1] not in (HIGH, LOW):
        raise GrammarError("a numeric effect statement is said with a numeric pole adjective")
    term = event_term_of(subject)  # the shape of the subject
    clause = subject.clause
    assert clause is not None
    bound = bound_predication(clause)
    kind = CAN if term.event_type.startswith("EVENTTYPE1.") else VERB
    if bound.kind != kind or bound.event != EVENT_VARIABLE:
        raise GrammarError(
            "the relative clause of a causal statement names the events of a one-place or a "
            f"two-place event type, bound to {EVENT_VARIABLE}"
        )
    if bound.tense != PRESENT or bound.aspect is not None or not bound.polarity:
        raise GrammarError(
            "the relative clause of a causal statement is in the present tense, has no aspect, "
            "and is never negated"
        )
    other = clause.agent if clause.object_relative else bound.object
    if kind == CAN:
        if other is not None or clause.object_relative:
            raise GrammarError("a one-place event type has an agent only")
    elif other is None or other.kind != CLASS_NP or other.referent != THING:
        raise GrammarError(
            "the other participant of the event of a causal statement is the generic noun"
        )
    if predication.kind == PRECONDITION and term.condition:
        raise GrammarError("a precondition statement has no condition: only an effect does")
    for phrase in (subject, other):
        if phrase is None:
            continue
        for literal in phrase.restriction:
            if literal.pole or is_numeric(literal.feature):
                raise GrammarError(
                    "a condition literal is a PROPERTY or PART feature or a Boolean fluent, not "
                    f"{literal.feature}"
                )
        if phrase.clause is not None:
            for part in phrase.clause.predications:
                if part is bound or (
                    part.kind == PRECONDITION and part.label.startswith(FLUENT_PREFIX)
                ):
                    continue
                raise GrammarError(
                    "the relative clause of a causal statement holds the bound verb and the "
                    "before predications of its fluent condition"
                )
    if subject != event_phrase_of(term, subject.determiner):
        raise GrammarError(
            "the condition of a causal statement is said in one way: static literals as "
            "modifiers, fluent literals with before, each on its role's noun phrase"
        )


def _check_phrase(phrase: NounPhrase, level: str, top: bool = False, causal: bool = False) -> None:
    """``causal`` says the phrase is in a causal statement, whose class-level relative clauses
    may hold ``before`` predications (a fluent condition, stage b3b)."""
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
        _check_phrase(clause.agent, level, causal=causal)
        _check_predication(
            predication, clause.agent, level, in_clause=True, gap=True, causal=causal
        )
        return
    for predication in clause.predications:
        _check_predication(predication, phrase, level, in_clause=True, causal=causal)


def _check_predication(
    predication: Predication,
    subject: NounPhrase,
    level: str,
    in_clause: bool,
    gap: bool = False,
    causal: bool = False,
) -> None:
    kind = predication.kind
    if in_clause and causal and predication.bound:
        # the verb of a causal statement's subject, checked by _check_causal; its object is the
        # other participant, which may carry a condition (stage b3b)
        if predication.object is not None:
            _check_phrase(predication.object, level, causal=True)
        return
    if in_clause and causal and kind == PRECONDITION:
        # a fluent condition of a conditional causal statement, "that are asleep before"
        if (
            predication.object is not None
            or predication.time is not None
            or predication.event is not None
            or predication.become
            or not predication.label.startswith(FLUENT_PREFIX)
        ):
            raise GrammarError("a before predication of a condition is a bare state adjective")
        return
    if in_clause and kind not in CLAUSE_KINDS:
        raise GrammarError(
            f"a relative clause does not hold a predicate of kind {kind!r}: an IS literal, a "
            "scalar pole, and a HAS literal belong in the restriction"
        )
    if kind not in (
        IS,
        HAS,
        CAN,
        COMPARISON,
        COMPARATIVE,
        MEMBER,
        PROJECTION,
        VERB,
        STATE_KIND,
        *CAUSAL_KINDS,
    ):
        raise GrammarError(f"unknown predicate kind {kind!r}")
    if kind in CAUSAL_KINDS:
        raise GrammarError("an effect or a precondition is the predicate of a causal statement")
    if predication.bound:
        raise GrammarError(
            "a verb bound to an event variable names the event of a causal statement, in the "
            "relative clause of its subject"
        )
    if in_clause and predication.time is not None:
        raise GrammarError("a relative clause is never about a time point")
    timed_kinds = (STATE_KIND,) + ((COMPARISON, COMPARATIVE) if predication.numeric else ())
    if (kind in timed_kinds) != (predication.time is not None and kind not in (CAN, VERB)):
        raise GrammarError(
            "a fluent is stated at a time point of a scene (a state or a change), and a time "
            "point belongs to a fluent, a numeric pole or comparative, or to what was possible "
            "then (able_now)"
        )
    if predication.become and kind != STATE_KIND:
        raise GrammarError("only a change of a fluent says become")
    if kind == STATE_KIND and is_numeric(predication.label):
        if not predication.become:
            raise GrammarError(
                "a numeric fluent is stated through its pole adjectives (a comparison), and "
                "changes with become and the comparative form"
            )
        if predication.label.rsplit(".", 1)[1] not in (HIGH, LOW) or not predication.polarity:
            raise GrammarError(
                "a numeric change is said with a numeric pole adjective's comparative form, and "
                "is never negated"
            )
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
    if (kind in (VERB, COMPARATIVE)) != (predication.object is not None or gap):
        raise GrammarError("a verb or a comparative, and only one, has an object")
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
        _check_phrase(predication.object, level, causal=causal)
    if kind != COMPARISON and predication.standard is not None:
        raise GrammarError("only a class-level pole names a comparison class")
