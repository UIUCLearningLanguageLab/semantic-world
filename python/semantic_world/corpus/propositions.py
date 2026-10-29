"""Layer 2: propositions. Logical forms, and the tests of their truth.

A proposition is a logical form with a level, a polarity, and a truth grounding. This module
holds the logical forms of the three levels, and :class:`Truth`, which checks any logical form
against the world and, for the event level, against the scenes.

**Class level.** The subject is a :class:`CategoryTerm`: a category, or the generic ``THING``,
with a restriction of literals. Its *subject set* is the set of instances below the category that
satisfy the restriction. A proposition with an empty subject set is vacuous, and is never valid.
The quantifier is one of ``nec_all``, ``all``, ``most``, ``some``, ``no``, and ``nec_no``
(``docs/specs/WORLD_AND_LANGUAGE.md``, "Quantifiers"):

- ``all``: every member of the subject set has the predicate, and ``no``: none does. Both are
  extensional, and both are available for every predicate;
- ``nec_all``: the rules guarantee the predicate for the subject (the fixed test), and
  ``nec_no`` the same with the predicate fixed at 0. ``nec_all`` implies ``all``, because a
  subject set is never empty. The ``nec`` quantifiers apply to one-place predicates only:
  PROPERTY and PART features, one-place capacities, and membership. Relation facts and patient
  capacities take the extensional quantifiers only, because no fixed test exists for them;
- ``most``: more than half of the subject set. ``some``: at least one member.

Membership ("penguins are birds") is ``nec_all`` by construction, and a rule statement is always
``nec_all``. A class-level scalar pole is a statement about the category's mean, not a
quantification over its members, and has no quantifier.

**Comparisons of scalars** (``docs/specs/WORLD_AND_LANGUAGE.md``, "Phase (b): the corpus", stage
b3a). A relative gradable adjective ("big") is a comparison: a predicate of kind ``comparison``
whose label is the pole concept (``SCALARDIM.1.HIGH``) and whose operator is ``ABOVE`` (the high
pole) or ``BELOW`` (the low pole), with its comparison class. At the instance level the referent
is compared with the category its noun names, ``ABOVE(SCALARDIM.1, REF.1, CATEGORY.1.5, z)``; at
the class level the category's mean is compared with an ancestor, or with ``ENTITY`` (every
entity) for a top-level category, ``ABOVE(SCALARDIM.1, MEAN(CATEGORY.1.4), CATEGORY.1, z)``, a
statement about the category with no quantifier. The threshold ``z`` is the language's
(``scalar_adjectives.z``), in standard deviations of the comparison class; every logical form
with a pole records it, and the truth tests judge every comparison with the world's runtime
functions (``above``, ``below``) over the world's comparison classes. A comparative ("bigger
than") is a predicate of kind ``comparative`` whose label is the dimension and whose patient is
the other argument: ``GREATER(SCALARDIM.1, REF.1, REF.2)`` says that the subject's value exceeds
the patient's, strictly, and at the class level ``GREATER(SCALARDIM.1, MEAN(CATEGORY.1.2),
MEAN(CATEGORY.1.5))`` compares two categories' means. ``GREATER`` is the only comparative
operator: "smaller than" is the language's word for ``GREATER`` with its arguments in the other
order. A comparative is never negated.

**Numeric fluents** (stage b3b). A numeric fluent's two words are absolute adjectives, judged
against the ends of its range at the language's cuts (``numeric_adjectives.cut_high`` and
``cut_low``): a numeric pole is a predicate of kind ``comparison`` whose label is the pole
(``NUMFL.1.HIGH``), whose comparison class is ``RANGE``, and whose operator is ``ABOVE`` or
``BELOW``, stated at a time point of a scene, ``HOLDS(SCENE.8, TIME.2, PAST, ABOVE(NUMFL.1,
REF.1, RANGE, 0.75))``. A numeric comparative compares two participants at a time point,
``HOLDS(SCENE.8, TIME.2, PAST, GREATER(NUMFL.1, REF.1, REF.2))``. A numeric change is a
predicate of kind ``state`` whose label is the fluent and whose ``direction`` is ``increase`` or
``decrease``: ``INCREASE(SCENE.8, TIME.2, PAST, NUMFL.1(REF.1))`` says that the value at the
next time point exceeds the value at the time point, whatever moved it. A numeric effect
statement is a predicate of kind ``effect`` whose label is the fluent and whose ``direction`` is
``raises`` or ``lowers``: ``NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)),
LOWERS(EVENTVAR.1, NUMFL.1(VAR.1))))`` says that the event type's effect subtracts from the
fluent (decision 28). A numeric threshold is never stated (decision 23), except as the
condition of a conditional statement when it is exactly the language's cut, ``> cut_high`` or
``<= cut_low``, which the numeric adjective says (WM.35): ``BEFORE(EVENTVAR.1, ABOVE(NUMFL.1,
VAR.1, RANGE, 0.75))`` in the restrictor.

**Conditional causal statements** (stage b3b). The condition of a conditional effect joins the
event term: ``EventTerm(event_type, role, condition)``, with the condition a tuple of
:class:`Condition` literals, each a static feature or a Boolean fluent of a role with a value.
The restrictor of the rendering holds them beside the bound event: ``NEC(ALL(EVENT(EVENTVAR.1,
EVENTTYPE2.1.2(VAR.1, VAR.2)) AND PROPERTY.4(VAR.2), AFTER(EVENTVAR.1, BOOLFL.3(VAR.2))))``, a
static literal as an atom and a fluent literal as ``BEFORE(EVENTVAR.1, BOOLFL.2(VAR.2))``. The
statement is true exactly when every event type below its event type has the effect with that
condition; the statement without its condition is false, because the definition guarantees the
effect only when the condition held.

**Polarity.** A negative proposition denies its predicate: ``most`` with a negative polarity
says that most members of the subject set lack the predicate. ``no`` and ``nec_no`` replace
sentence negation, so they always have a positive polarity in the logical form, and ``all`` and
``nec_all`` never have a negative one.

**Instance level.** The subject is an instance, and truth is read from the instance's values.

**Relative clauses.** A category term can also be restricted by relative clauses, which are
restrictive: "penguins that can swim" are the penguins able to swim, "owls that eat mice" are
the owls that can eat at least one mouse, and "mice that owls eat" are the mice that at least
one owl can eat. No fixed test exists for such a subject, so it takes the extensional
quantifiers only.

**Event level.** The proposition says that something happened in a scene: the subject is the
agent, and the predicate is a one-place event type, or a two-place event type with a patient
instance. The event type can be the event's own or a category above it, as a noun can name a
category above a leaf. The proposition is true when such an event occurred in the scene. An
event-level proposition is never negated. Its tense is the corpus's
(``propositions.events.tense``). Its aspect belongs to the report, not to the event: both
aspects are true of any event that occurred (CG.64). Its grounding says whether the binding is
``able`` (the requirement holds) and ``legal`` (the binding was legal at some time point of the
scene), which is what tells an impossible false test item from one that merely did not happen.

**State, change, and ``able_now`` levels** (``docs/specs/WORLD_AND_LANGUAGE.md``, "States and
changes" and "'Can': what is held fixed"). All three are about one participant of a scene at a
time point, and take the scene's tense and no aspect:

- a *state* says that a fluent, base or derived, held of the subject at the time point:
  ``HOLDS(SCENE.8, TIME.2, PAST, BOOLFL.3(REF.1))``, or with ``NOT`` the negative state;
- a *change* says that the fluent was false at the time point and true at the next, or the
  reverse: ``BECOME(SCENE.8, TIME.2, PAST, BOOLFL.3(REF.1))``. A change says nothing of its
  cause; the grounding records the event of the step whose own effect made it, when one did;
- an *able_now* proposition says that the binding was legal at the time point, the state of
  the scene held fixed: ``ABLE_NOW(SCENE.8, TIME.2, PAST, EVENTTYPE2.1.2(REF.1, REF.2))``. The
  negative one with the requirement true says that a precondition blocked the event.

Their truth is read from the scene's history, replayed through the runtime.

**Causal statements** (``docs/specs/WORLD_AND_LANGUAGE.md``, "Causal statements") are class-level
and timeless: what events of an event type do and what they need. The subject is an
:class:`EventTerm`, the things that take a role in events of an event type ("things that things
catch"), the predicate is an *effect* or a *precondition* of a base fluent with a value, and the
quantifier is ``nec_all``, because the definition guarantees the statement:
``NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), AFTER(EVENTVAR.1, BOOLFL.3(VAR.2))))``.
An effect statement is true exactly when the event type has that effect, and a precondition
statement exactly when its precondition holds that literal; a statement about a category of
event types is true when every event type below the category has the entry. The polarity is
always positive: the value is in the predicate.

**The language's words** decide what a proposition's words mean, never what the proposition
means: which quantifiers "all" and "no" express (``quantifiers.universal_words``), the least
share at which a speaker says "most" (``quantifiers.most.usage_min``), and the implicature of
"some" (``quantifiers.some.exclude_all``). :class:`Truth` answers both questions: whether a
proposition is true, and whether a document can state it (``felicitous``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.histories import (
    SceneEvent,
    event_of,
    scene_events,
    scene_of,
    time_index,
    time_label,
)
from semantic_world.corpus.world import (
    ENTITY,
    FLUENT_PREFIX,
    NUMERIC_PREFIX,
    PART_PREFIX,
    PLACE_PART_PREFIX,
    PLACE_PROPERTY_PREFIX,
    PLACE_SCALAR_PREFIX,
    PROPERTY_PREFIX,
    SCALAR_PREFIX,
    THING,
    World,
)
from semantic_world.taxonomy.rules import CONE_ENUMERATION_LIMIT
from semantic_world.world.history import History, replay
from semantic_world.world.runtime import (
    RANGE,
    Mean,
    State,
    above,
    below,
    comparison_classes,
    derive,
    greater,
    legal,
    legal_bindings,
)

CLASS = "class"
INSTANCE = "instance"
EVENT = "event"
STATE = "state"
CHANGE = "change"
ABLE_NOW = "able_now"
LEVELS = (CLASS, INSTANCE, EVENT, STATE, CHANGE, ABLE_NOW)
TIMED_LEVELS = (STATE, CHANGE, ABLE_NOW)
"""The levels whose proposition is about one time point of a scene."""

NEC_ALL = "nec_all"
ALL = "all"
MOST = "most"
SOME = "some"
NO = "no"
NEC_NO = "nec_no"
QUANTIFIERS = (NEC_ALL, ALL, MOST, SOME, NO, NEC_NO)
POSITIVE_ORDER = (NEC_ALL, ALL, MOST, SOME)
"""The quantifiers of a positive fact, strongest first."""
NEGATIVE_ORDER = (NEC_NO, NO, MOST, SOME)
"""The quantifiers of a negative fact, strongest first: ``no`` and ``nec_no`` with a positive
polarity, ``most`` and ``some`` with a negative one."""
NEC_QUANTIFIERS = (NEC_ALL, NEC_NO)
UNIVERSALS = (NEC_ALL, ALL, NO, NEC_NO)
COUNTERPART = {NEC_ALL: NEC_NO, NEC_NO: NEC_ALL, ALL: NO, NO: ALL, MOST: MOST, SOME: SOME}
"""Each quantifier's counterpart for a fact of the other polarity."""

IS = "property"
"""The kind of a PROPERTY predicate (the constant keeps the name of the old ``is`` kind)."""
HAS = "part"
"""The kind of a PART predicate (the old ``has`` kind)."""
CAN = "event_type1"
"""The kind of a one-place event-type predicate (the old ``can`` kind)."""
COMPARISON = "comparison"
"""The kind of a scalar pole's predicate: a relative gradable adjective, a comparison of the
subject (or its category's mean) with a comparison class (``ABOVE`` or ``BELOW``)."""
COMPARATIVE = "comparative"
"""The kind of a comparative's predicate: ``GREATER``, the subject's value against the
patient's on one dimension."""
ABOVE = "ABOVE"
BELOW = "BELOW"
GREATER = "GREATER"
OPERATORS = (ABOVE, BELOW, GREATER)
"""The operators of the comparisons, as the propositional rendering writes them."""
HIGH = "HIGH"
LOW = "LOW"
INCREASE = "increase"
DECREASE = "decrease"
CHANGE_DIRECTIONS = (INCREASE, DECREASE)
"""The directions of a numeric change (stage b3b): the value at the next time point is greater
than, or less than, the value at the time point."""
RAISES = "raises"
LOWERS = "lowers"
EFFECT_DIRECTIONS = (RAISES, LOWERS)
"""The directions of a numeric effect statement (stage b3b): the event type's effect adds to the
fluent, or subtracts from it."""
DIRECTIONS = CHANGE_DIRECTIONS + EFFECT_DIRECTIONS
MEMBER = "member"
PROJECTION = "patient_capacity"
"""The kind of a patient-capacity predicate (the old ``projection`` kind)."""
VERB = "event_type2"
"""The kind of a two-place event-type predicate (the old ``verb`` kind)."""
STATE_KIND = "state"
"""The kind of a fluent predicate: a state (``HOLDS``) or a change (``BECOME``)."""
EFFECT = "effect"
"""The kind of the predicate of an effect statement: after every event of the subject's event
type, the participant in the subject's role has the fluent's value (``AFTER``)."""
PRECONDITION = "precondition"
"""The kind of the predicate of a precondition statement: before every event of the subject's
event type, the participant in the subject's role has the fluent's value (``BEFORE``)."""
PAIR_STATE = "pair_state"
"""The kind of a pair fluent's predicate (stage c4b, ``docs/specs/WORLD_PHASE_C.md``, "Instance
level"): a pair state between two entities, or from an entity to a place, at a time point of a
scene, with the pair fluent as its ``label`` and the second argument as its ``patient``, as the
timed comparative carries its patient: ``HOLDS(SCENE.8, TIME.2, PAST, PAIRFL.2(REF.1, REF.2))``;
with ``become`` a change, ``BECOME(SCENE.8, TIME.2, PAST, NOT PAIRFL.5(REF.1, REF.3))``."""
PLACE_EVENT = "place_event_type"
"""The kind of a place event type's predicate (stage c4b): an event of a move or an act at a
place, a transitive verb whose patient is a place, ``EVENT(SCENE.8.EVENTINSTANCE.4, PAST,
SIMPLE, PLACEEVENTTYPE.2(REF.1, REF.3))``. The key is the world's symbol kind."""
CAUSAL_KINDS = (EFFECT, PRECONDITION)
COMPARISON_KINDS = (COMPARISON, COMPARATIVE)
"""The kinds judged by the world's comparisons of scalars."""
EVENT_KINDS = (CAN, VERB, PLACE_EVENT)
"""The kinds whose predicate is an event type with events: a one-place event type, a two-place
event type or a category of them, and (stage c4b) a place event type."""
KINDS = (
    IS,
    HAS,
    CAN,
    COMPARISON,
    COMPARATIVE,
    MEMBER,
    PROJECTION,
    VERB,
    STATE_KIND,
    EFFECT,
    PRECONDITION,
    PAIR_STATE,
    PLACE_EVENT,
)
FEATURE_KINDS = (IS, HAS, CAN)
"""The kinds whose predicate is a feature of the world's feature table: a PROPERTY feature, a
PART feature, or a one-place event type (the capacity to be its agent)."""
NEC_KINDS = FEATURE_KINDS + (MEMBER,)
"""The kinds that the ``nec`` quantifiers apply to."""

PAST = "past"
PRESENT = "present"
TENSES = (PAST, PRESENT)
SIMPLE = "simple"
PROGRESSIVE = "progressive"
ASPECTS = (SIMPLE, PROGRESSIVE)

# How a proposition's truth was decided: the ``test`` of its grounding.
EXACT = "exact"
"""The fixed test, by enumerating the cone: exact."""
LOCAL = "local"
"""The fixed test, by the local test of the taxonomy generator: never wrongly fixed, but it can
miss a fixed feature."""
OBSERVED = "observed"
"""A proportion over the instances of the subject set, or over pairs of instances."""
TREE = "tree"
"""Membership, read from the tree."""
MEAN = "mean"
"""A class-level comparison: the subject category's mean against its comparison class, or
against another category's mean."""
VALUE = "value"
"""An instance's own value."""
OCCURRED = "event"
"""An event-level proposition: whether such an event occurred in the scene."""
DEFINITION = "definition"
"""A causal statement: judged by the definition's entries for the event types."""

LABEL_KEY = "label"
"""The key under which every predicate and clause of the JSON logical form names its symbol,
the state predicate's fluent included (Jon's ruling of October 9, 2026, on the stage a7a
question 4: the specification's ``fluent`` key becomes ``label``, like every other kind)."""
_JSON_KEY = dict.fromkeys(KINDS, LABEL_KEY)
HISTORY = "history"
"""A state, a change, or an ``able_now`` proposition: judged by replaying the scene."""
AGENT = "agent"
PATIENT = "patient"
ROLES = (AGENT, PATIENT)
EVENT_VARIABLE = "EVENTVAR.1"
"""The event variable of a causal statement, which binds the events of the subject's event
type. A statement has one event, so the variable is always the first."""
_LITERAL_ORDER = {
    PROPERTY_PREFIX: 0,
    PART_PREFIX: 1,
    SCALAR_PREFIX: 2,
    FLUENT_PREFIX: 3,
    NUMERIC_PREFIX: 4,
    # the place features (stage c4b), in the same order: a place's restriction holds place
    # properties, place parts, then place scalar poles
    PLACE_PROPERTY_PREFIX: 0,
    PLACE_PART_PREFIX: 1,
    PLACE_SCALAR_PREFIX: 2,
}


def is_part_feature(label: str) -> bool:
    """Whether a label is a PART feature of entities or (stage c4b) a place part: the literals
    said with a with-phrase or a without-phrase."""
    return label.startswith(PART_PREFIX) or label.startswith(PLACE_PART_PREFIX)


def is_property_feature(label: str) -> bool:
    """Whether a label is a PROPERTY feature of entities or (stage c4b) a place property."""
    return label.startswith(PROPERTY_PREFIX) or label.startswith(PLACE_PROPERTY_PREFIX)


def is_scalar_pole(label: str) -> bool:
    """Whether a label is a pole of a scalar dimension of entities or (stage c4b) of places:
    a relative adjective, a comparison with a comparison class."""
    return label.startswith(SCALAR_PREFIX) or label.startswith(PLACE_SCALAR_PREFIX)


def _number(value: float) -> float:
    """A real number as it is written to the output files: 6 decimal places."""
    return round(float(value), 6)


# ---------------------------------------------------------------------------------------------
# Logical forms
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Literal:
    """One literal of a restriction: a PROPERTY or PART feature, positive or negative, or a
    scalar pole (``SCALARDIM.1.HIGH``), which is always positive."""

    feature: str
    positive: bool = True

    @property
    def pole(self) -> bool:
        """Whether the literal is a scalar pole, of entities or (stage c4b) of places."""
        return is_scalar_pole(self.feature)

    @property
    def sort_key(self) -> tuple:
        prefix, _, rest = self.feature.partition(".")
        number, _, pole = rest.partition(".")
        return (
            _LITERAL_ORDER.get(prefix + ".", 3),
            int(number) if number.isdecimal() else 0,
            pole,
        )

    def __str__(self) -> str:
        return self.feature if self.positive else f"not {self.feature}"

    @classmethod
    def parse(cls, text: str) -> Literal:
        if text.startswith("not "):
            return cls(text[4:], False)
        return cls(text)


@dataclass(frozen=True)
class Clause:
    """A relative clause that restricts a category term. It keeps the members that are able to
    be the agent of a one-place event type ("penguins that can swim"), the members able to be
    the agent of a two-place event type with at least one member of another category as its
    patient ("owls that eat mice", with ``patient``), or the patient with at least one member
    of another category as its agent ("mice that owls eat", with ``agent``)."""

    kind: str
    """``event_type1`` or ``event_type2``."""
    label: str
    patient: CategoryTerm | None = None
    """A two-place event type in a subject relative: the other category, which the head acts
    on."""
    agent: CategoryTerm | None = None
    """A two-place event type in an object relative: the other category, which acts on the
    head."""

    @property
    def other(self) -> CategoryTerm | None:
        return self.patient if self.patient is not None else self.agent

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, _JSON_KEY[self.kind]: self.label}
        if self.patient is not None:
            data["patient"] = self.patient.to_json()
        if self.agent is not None:
            data["agent"] = self.agent.to_json()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Clause:
        kind = data["kind"]
        patient, agent = data.get("patient"), data.get("agent")
        return cls(
            kind,
            data[_JSON_KEY[kind]],
            None if patient is None else CategoryTerm.from_json(patient),
            None if agent is None else CategoryTerm.from_json(agent),
        )


@dataclass(frozen=True)
class CategoryTerm:
    """A category with a restriction: the subject of a class-level proposition, or the patient
    of a class-level relation. The category is a category label or ``THING``. The restriction is
    a set, kept in one order: PROPERTY literals, PART literals, then scalar poles, each by index.
    The relative clauses restrict the category further, and are kept in the order given."""

    category: str
    restriction: tuple[Literal, ...] = ()
    clauses: tuple[Clause, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(set(self.restriction), key=lambda x: x.sort_key))
        object.__setattr__(self, "restriction", ordered)
        object.__setattr__(self, "clauses", tuple(self.clauses))

    @property
    def plain(self) -> CategoryTerm:
        """The term without its relative clauses."""
        return CategoryTerm(self.category, self.restriction) if self.clauses else self

    def concepts(self) -> list[str]:
        """The concepts that the term needs words for, its clauses' included."""
        labels = [self.category] + [x.feature for x in self.restriction]
        for clause in self.clauses:
            labels.append(clause.label)
            if clause.other is not None:
                labels += clause.other.concepts()
        return labels

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "category": self.category,
            "restriction": [str(x) for x in self.restriction],
        }
        if self.clauses:
            data["clauses"] = [clause.to_json() for clause in self.clauses]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CategoryTerm:
        return cls(
            data["category"],
            tuple(Literal.parse(x) for x in data.get("restriction", ())),
            tuple(Clause.from_json(x) for x in data.get("clauses", ())),
        )


@dataclass(frozen=True)
class Condition:
    """One literal of the condition of a conditional causal statement (stage b3b): a static
    feature (PROPERTY or PART) or a Boolean fluent (base or derived) of one role of the event,
    with the value it must have at the event's time point, or (WM.35, October 15, 2026) a
    numeric pole, ``NUMFL.1.HIGH``, always positive: the world's numeric threshold literal
    whose threshold is the language's cut for that adjective (``> cut_high`` for the high
    pole, ``<= cut_low`` for the low one). In the JSON logical form it is ``{"role":
    "patient", "label": "PROPERTY.4", "value": true}``; a numeric literal records its ``cut``
    after its value, as a numeric pole predicate does."""

    role: str
    label: str
    value: bool = True

    @property
    def fluent(self) -> bool:
        """Whether the literal names a Boolean fluent, which the sentence says with ``before``
        ("that are asleep before"); a static feature is a modifier."""
        return self.label.startswith(FLUENT_PREFIX)

    @property
    def numeric(self) -> bool:
        """Whether the literal is a numeric pole (``NUMFL.1.HIGH``), said with ``before`` as a
        Boolean fluent is ("that are hungry before"; WM.35)."""
        return self.label.startswith(NUMERIC_PREFIX)

    @property
    def timed(self) -> bool:
        """Whether the literal is about the state at the event's time point, a Boolean fluent
        or a numeric pole, so that the sentence says it with ``before``; a static feature is a
        modifier."""
        return self.fluent or self.numeric

    @property
    def sort_key(self) -> tuple:
        prefix, _, rest = self.label.partition(".")
        number = rest.split(".")[0]
        return (
            ROLES.index(self.role),
            _LITERAL_ORDER.get(prefix + ".", 5),
            int(number) if number.isdecimal() else 0,
            self.label,
            not self.value,
        )

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "label": self.label, "value": self.value}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(data["role"], data["label"], bool(data["value"]))


@dataclass(frozen=True)
class EventTerm:
    """The subject of a causal statement: the things that take a role in the events of an
    event type, "things that catch things" (the agent) or "things that things catch" (the
    patient). The head is the generic noun, and the event type is a one-place event type, a
    two-place event type, or a category of two-place event types. In the JSON logical form it
    is ``{"head": "THING", "event": "EVENTTYPE2.1.2", "role": "patient"}``. A conditional
    statement (stage b3b) adds the condition of its effect, ``"condition": [{"role":
    "patient", "label": "PROPERTY.4", "value": true}]``: the events of the event type whose
    participants satisfied the condition at the event's time point."""

    event_type: str
    role: str = AGENT
    """``agent`` or ``patient``."""
    condition: tuple[Condition, ...] = ()
    """The condition of a conditional effect statement, in one order; empty otherwise."""

    def __post_init__(self) -> None:
        ordered = tuple(sorted(set(self.condition), key=lambda c: c.sort_key))
        object.__setattr__(self, "condition", ordered)

    @property
    def category(self) -> str:
        """The head noun's category: the generic noun."""
        return THING

    @property
    def conditional(self) -> bool:
        return bool(self.condition)

    @property
    def unconditional(self) -> EventTerm:
        """The same term without its condition."""
        return EventTerm(self.event_type, self.role)

    def concepts(self) -> list[str]:
        """The concepts that the term needs words for: the generic noun, the event type, and
        the features and fluents of its condition."""
        return [THING, self.event_type] + [c.label for c in self.condition]

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"head": THING, "event": self.event_type, "role": self.role}
        if self.condition:
            data["condition"] = [c.to_json() for c in self.condition]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EventTerm:
        return cls(
            data["event"],
            data["role"],
            tuple(Condition.from_json(c) for c in data.get("condition", ())),
        )


def is_event_variable(label: str | None) -> bool:
    """Whether a label is an event variable (``EVENTVAR.<n>``)."""
    return label is not None and label.startswith("EVENTVAR.")


def pole_operator(pole: str) -> str:
    """The comparison operator of a pole concept: ``ABOVE`` for ``SCALARDIM.1.HIGH`` and
    ``BELOW`` for ``SCALARDIM.1.LOW``."""
    return ABOVE if pole.rsplit(".", 1)[1] == HIGH else BELOW


def pole_dimension(pole: str) -> str:
    """The scalar dimension, or the numeric fluent, of a pole concept."""
    return pole.rsplit(".", 1)[0]


def is_numeric(label: str) -> bool:
    """Whether a label is a numeric fluent, or a pole of one (``NUMFL.1``, ``NUMFL.1.HIGH``)."""
    return label.startswith(NUMERIC_PREFIX)


def pole_direction(pole: str, kind: str) -> str:
    """The direction that a numeric pole's comparative form states (stage b3b): the high
    pole's ("hungrier") an ``increase`` of a change (kind ``state``) or a ``raises`` of an
    effect statement (kind ``effect``), the low pole's a ``decrease`` or a ``lowers``."""
    high = pole.rsplit(".", 1)[1] == HIGH
    if kind == EFFECT:
        return RAISES if high else LOWERS
    return INCREASE if high else DECREASE


def direction_pole(fluent: str, direction: str) -> str:
    """The numeric pole whose comparative form says a direction: the high pole for
    ``increase`` and ``raises``, the low pole for ``decrease`` and ``lowers``."""
    return f"{fluent}.{HIGH if direction in (INCREASE, RAISES) else LOW}"


def low_word(predicate: dict[str, Any]) -> bool:
    """Whether a comparative's JSON predicate says the sentence used the low pole's word
    (``"word": "SCALARDIM.1.LOW"``), so that its subject is the lesser argument."""
    word = predicate.get("word")
    return isinstance(word, str) and word.rsplit(".", 1)[-1] == LOW


def comparison_class_of(category: str) -> str:
    """The comparison class a term's category gives its poles: the category itself, or
    ``ENTITY`` for the generic noun."""
    return ENTITY if category == THING else category


@dataclass(frozen=True)
class Predicate:
    kind: str
    """``property``, ``part``, ``event_type1``, ``comparison``, ``comparative``, ``member``,
    ``patient_capacity``, ``event_type2``, ``state``, ``effect``, or ``precondition``."""
    label: str
    """The predicate's concept: a PROPERTY or PART feature, a one-place event type, a scalar
    pole (a comparison), a scalar dimension (a comparative), a category, a patient capacity
    (``CANBE.<event type>``), a two-place event type or a category of them, or a fluent
    (``BOOLFL.3``)."""
    patient: CategoryTerm | str | None = None
    """Two-place event types and comparatives only: the patient category (class level) or the
    patient instance (instance level); for a comparative, the argument the subject exceeds; for
    a pair state (stage c4b), the second argument of the pair, an instance or a place; for a
    place event type, the place."""
    comparison: str | None = None
    """Comparisons only: the comparison class. At the instance level it is the category that
    the subject's noun names ("big for a mouse"); at the class level an ancestor of the
    subject's category, or ``ENTITY`` for a top-level category (CG.3,
    ``scalar_adjectives.class_levels_up``)."""
    value: bool | None = None
    """Effects and preconditions only: the value the fluent is set to (an effect) or must have
    (a precondition)."""
    direction: str | None = None
    """Numeric fluents only (stage b3b): ``increase`` or ``decrease`` for a change of kind
    ``state`` whose label is a numeric fluent, ``raises`` or ``lowers`` for an effect statement
    about one."""

    @property
    def causal(self) -> bool:
        return self.kind in CAUSAL_KINDS

    @property
    def numeric(self) -> bool:
        """Whether the predicate is about a numeric fluent: a numeric pole, a numeric
        comparative, a numeric change, or a numeric effect statement."""
        return is_numeric(self.label)

    @property
    def operator(self) -> str | None:
        """The comparison operator: ``ABOVE`` or ``BELOW`` for a pole, ``GREATER`` for a
        comparative, and None for any other predicate."""
        if self.kind == COMPARISON:
            return pole_operator(self.label)
        if self.kind == COMPARATIVE:
            return GREATER
        return None

    @property
    def dimension(self) -> str | None:
        """The scalar dimension of a comparison or a comparative."""
        if self.kind == COMPARISON:
            return pole_dimension(self.label)
        if self.kind == COMPARATIVE:
            return self.label
        return None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, _JSON_KEY[self.kind]: self.label}
        if self.operator is not None:
            data["operator"] = self.operator
        if isinstance(self.patient, CategoryTerm):
            data["patient"] = self.patient.to_json()
        elif self.patient is not None:
            data["patient"] = {"instance": self.patient}
        if self.comparison is not None:
            data["class"] = self.comparison
        if self.value is not None:
            data["value"] = self.value
        if self.direction is not None:
            data["direction"] = self.direction
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Predicate:
        kind = data["kind"]
        patient = data.get("patient")
        if patient is not None:
            patient = (
                patient["instance"] if "instance" in patient else CategoryTerm.from_json(patient)
            )
        return cls(
            kind,
            data[_JSON_KEY[kind]],
            patient,
            data.get("class"),
            data.get("value"),
            data.get("direction"),
        )


@dataclass(frozen=True)
class Proposition:
    """A logical form. Two propositions are equal when they say the same thing: the label, the
    grounding, and the rule a statement comes from take no part in comparisons."""

    level: str
    subject: CategoryTerm | EventTerm | str
    """A category term (class level), an event term (a causal statement, class level), or an
    instance label (the other levels)."""
    predicate: Predicate
    polarity: bool = True
    quantifier: str | None = None
    """Class level only. None for a class-level comparison or comparative, which have no
    quantifier."""
    grounding: dict[str, Any] | None = field(default=None, compare=False)
    id: str | None = field(default=None, compare=False)
    rule: tuple[str, int] | None = field(default=None, compare=False)
    """For a rule statement: the determined feature, and the number of the term of its rule's
    minimal DNF, from 1."""
    scene: str | None = None
    """Event, state, change, and able_now levels: the scene."""
    event: str | None = None
    """Event level only: the event that the proposition reports
    (``SCENE.8.EVENTINSTANCE.5``). A test item, true or false, names only its scene, and has
    None: it says that some event of the scene was this one."""
    tense: str | None = None
    """Event, state, change, and able_now levels: ``past`` or ``present``."""
    aspect: str | None = None
    """Event level only: ``simple`` or ``progressive``, the report's choice."""
    time: str | None = None
    """State, change, and able_now levels: the time point of the scene (``TIME.2``). A change
    is from this time point to the next."""

    @property
    def timed(self) -> bool:
        """Whether the proposition is about one time point of a scene."""
        return self.level in TIMED_LEVELS

    @property
    def causal(self) -> bool:
        """Whether the proposition is a causal statement: an effect or a precondition of the
        events of an event type."""
        return isinstance(self.subject, EventTerm)

    def causal_record(self) -> dict[str, Any] | None:
        """The definition entry that a causal statement states (the ``causal`` record of
        "Causal statements"): the event type, the role, the fluent, and the value. None for any
        other proposition."""
        if not isinstance(self.subject, EventTerm):
            return None
        record: dict[str, Any] = {
            "event_type": self.subject.event_type,
            "role": self.subject.role,
            "fluent": self.predicate.label,
        }
        if self.predicate.direction is not None:
            record["direction"] = self.predicate.direction
        else:
            record["value"] = self.predicate.value
        if self.subject.condition:
            record["condition"] = [c.to_json() for c in self.subject.condition]
        return record

    @property
    def negative(self) -> bool:
        """Whether the proposition denies its predicate: a negative polarity, or ``no``."""
        return not self.polarity or self.quantifier in (NO, NEC_NO)

    @property
    def nec(self) -> bool:
        return self.quantifier in NEC_QUANTIFIERS

    def concepts(self) -> tuple[str, ...]:
        """The concepts that a sentence needs words for, apart from function words: the subject
        category, the restrictions, the predicate, and the patient category. An instance's noun
        is chosen when the instance is mentioned, so an instance adds no concept here, apart
        from the comparison class of a scalar pole (a category; ``ENTITY`` and ``RANGE`` need no
        word). A comparative needs the two pole adjectives of its dimension, because the
        language says it with either ("bigger than", or "smaller than" with the arguments the
        other way); a numeric change or effect statement needs the pole adjective whose
        comparative form says its direction ("became hungrier", "become fuller")."""
        labels: list[str] = []
        for term in (self.subject, self.predicate.patient):
            if isinstance(term, CategoryTerm | EventTerm):
                labels += term.concepts()
        predicate = self.predicate
        if predicate.kind == COMPARATIVE:
            labels += [f"{predicate.label}.{HIGH}", f"{predicate.label}.{LOW}"]
        elif predicate.direction is not None:
            labels.append(direction_pole(predicate.label, predicate.direction))
        else:
            labels.append(predicate.label)
        if predicate.comparison is not None and predicate.comparison not in (ENTITY, RANGE):
            labels.append(predicate.comparison)
        return tuple(dict.fromkeys(labels))

    def to_json(self) -> dict[str, Any]:
        """The logical form as it is written to ``documents.jsonl``."""
        data: dict[str, Any] = {"id": self.id, "level": self.level}
        if self.level == CLASS:
            assert isinstance(self.subject, CategoryTerm | EventTerm)
            data["quantifier"] = self.quantifier
            data["polarity"] = self.polarity
            data["subject"] = self.subject.to_json()
        else:
            if self.level == EVENT:
                data["scene"] = self.scene
                data["event"] = self.event
                data["tense"] = self.tense
                data["aspect"] = self.aspect
            elif self.level in TIMED_LEVELS:
                data["scene"] = self.scene
                data["time"] = self.time
                data["tense"] = self.tense
            data["polarity"] = self.polarity
            data["subject"] = {"instance": self.subject}
        data["predicate"] = self.predicate.to_json()
        if self.rule is not None:
            data["rule"] = {"feature": self.rule[0], "term": self.rule[1]}
        causal = self.causal_record()
        if causal is not None:
            data["causal"] = causal
        data["grounding"] = self.grounding
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Proposition:
        subject = data["subject"]
        rule = data.get("rule")
        predicate = Predicate.from_json(data["predicate"])
        if "instance" in subject:
            read_subject: CategoryTerm | EventTerm | str = subject["instance"]
        elif "head" in subject:
            read_subject = EventTerm.from_json(subject)
        else:
            read_subject = CategoryTerm.from_json(subject)
        if predicate.kind == COMPARATIVE and low_word(data["predicate"]):
            # a sentence's form holds its subject and object in the sentence's order: said
            # with the low pole's word ("smaller than"), the greater argument is the patient
            assert predicate.patient is not None
            read_subject, predicate = (
                predicate.patient,
                dataclasses.replace(predicate, patient=read_subject),
            )
        return cls(
            level=data["level"],
            subject=read_subject,
            predicate=predicate,
            polarity=data["polarity"],
            quantifier=data.get("quantifier"),
            grounding=data.get("grounding"),
            id=data.get("id"),
            rule=None if rule is None else (rule["feature"], rule["term"]),
            scene=data.get("scene"),
            event=data.get("event"),
            tense=data.get("tense"),
            aspect=data.get("aspect"),
            time=data.get("time"),
        )


@dataclass(frozen=True)
class Evaluation:
    """What the truth tests say about a logical form."""

    valid: bool
    """Whether the logical form can be judged at all: its labels exist, its quantifier is
    allowed for its predicate, and it is not vacuous."""
    true: bool = False
    felicitous: bool = False
    """True, and usable in a document: its quantifier has a word in the language, "most" is said
    only at or above ``quantifiers.most.usage_min``, and with ``quantifiers.some.exclude_all``
    on, "some" is used only when the language's "all" (for a negative proposition, its "no") is
    false."""
    grounding: dict[str, Any] | None = None
    reason: str = ""
    """Why the logical form is not valid."""


def _invalid(reason: str) -> Evaluation:
    return Evaluation(False, reason=reason)


# ---------------------------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------------------------


def unconditional_effects(record: Any) -> tuple:
    """The Boolean effects of an event-type record without a condition: the entries of the
    unconditional Boolean effect statements (stage b1). A conditional effect guarantees its
    fluent only when its condition held, which only a conditional statement says (stage b3b),
    and a numeric effect is the entry of a ``RAISES`` or ``LOWERS`` statement."""
    return tuple(e for e in record.effects if not e.condition and not e.numeric)


CutPoints = Mapping[str, tuple[float, float]]
"""For each numeric fluent, the points of its range at the language's two cuts, ``(the point
at cut_high, the point at cut_low)``, computed as the runtime computes them (``low + cut *
(high - low)``)."""


def numeric_pole_of(literal: Any, cut_points: CutPoints | None) -> str | None:
    """The numeric pole that says a numeric threshold literal of the world (a condition
    literal), when its threshold is exactly the language's cut for that adjective: ``> t``
    with ``t`` the point at ``cut_high`` is the high pole, and ``<= t`` with ``t`` the point at
    ``cut_low`` the low pole (Jon's ruling WM.35, October 15, 2026). None for every other
    threshold, which no word of the language states (decision 23), and when no cut points are
    given."""
    if cut_points is None or literal.symbol not in cut_points:
        return None
    high_point, low_point = cut_points[literal.symbol]
    if literal.value and literal.threshold == high_point:
        return f"{literal.symbol}.{HIGH}"
    if not literal.value and literal.threshold == low_point:
        return f"{literal.symbol}.{LOW}"
    return None


def condition_of(effect: Any, cut_points: CutPoints | None = None) -> tuple[Condition, ...] | None:
    """The condition of an effect record of the world, as the corpus states it: its literals
    as :class:`Condition` literals, in one order. A numeric threshold literal is the positive
    numeric pole whose cut it is at (:func:`numeric_pole_of`, WM.35); any other numeric
    threshold literal has no :class:`Condition` (decision 23), so an effect with one gives
    None; so does an effect whose condition holds a pair literal (WM.59, stage c4a)."""
    found = []
    for literal in effect.condition:
        if literal.pair:
            # a pair condition literal (stage c2b) makes no statement in phase (c), and its
            # unconditional form is never stated as true (Jon's ruling WM.59, review point 11)
            return None
        if literal.numeric:
            pole = numeric_pole_of(literal, cut_points)
            if pole is None:
                return None
            found.append(Condition(literal.role, pole, True))
        else:
            found.append(Condition(literal.role, literal.symbol, bool(literal.value)))
    return tuple(sorted(set(found), key=lambda c: c.sort_key))


def effect_matches(
    effect: Any,
    role: str,
    fluent: str,
    value: bool | None,
    direction: str | None,
    condition: tuple[Condition, ...],
    cut_points: CutPoints | None = None,
) -> bool:
    """Whether an effect record of the world is the entry of an effect statement: the same role
    and fluent, the value (a Boolean effect) or the sign (a numeric effect: ``raises`` for an
    addition, ``lowers`` for a subtraction), and the same condition (none for an unconditional
    statement; a numeric threshold literal at the language's cut is its numeric pole, WM.35).
    An effect under any other numeric threshold condition is no entry of any statement, and
    neither is a pair effect or an effect under a pair condition (stage c4a; stage c5a states
    pair effects)."""
    if effect.pair or effect.role != role or effect.fluent != fluent:
        return False
    if effect.numeric:
        if direction is None or (effect.amount >= 0) != (direction == RAISES):
            return False
    elif direction is not None or effect.value != value:
        return False
    return condition_of(effect, cut_points) == condition


class Truth:
    """The truth tests of one world, under one corpus configuration."""

    def __init__(
        self, config: Config, world: World, cone_limit: int = CONE_ENUMERATION_LIMIT
    ) -> None:
        self.world = world
        self.quantifiers = config.quantifiers
        self.event_tense = config.propositions.event_tense
        self.z = config.scalar_z
        """The language's threshold for its pole adjectives, in standard deviations of the
        comparison class (``scalar_adjectives.z``)."""
        self.cuts = {HIGH: config.cut_high, LOW: config.cut_low}
        """The language's cuts for the two absolute adjectives of a numeric fluent, as
        fractions of its range (``numeric_adjectives``, stage b3b)."""
        self.cut_points: CutPoints = {
            fluent: (low + config.cut_high * (high - low), low + config.cut_low * (high - low))
            for fluent, (low, high) in world.definition.numeric_ranges.items()
        }
        """For each numeric fluent, the points of its range at the two cuts, in the runtime's
        arithmetic: a numeric condition literal of the world whose threshold is exactly one of
        them is stated with the numeric adjective (WM.35)."""
        self.cone_limit = cone_limit
        self._masks: dict[tuple[str, str], tuple[np.ndarray, float, float]] = {}
        self.count = world.count
        self.instance_index = world.instance_index
        self.paths = world.paths
        self.categories = world.category
        self.scenes: dict[str, History] = {}
        """The scenes that event-level propositions are judged against, by label."""
        self._events: dict[str, tuple[SceneEvent, ...]] = {}
        self._events_by_type: dict[str, tuple[tuple[str, SceneEvent], ...]] | None = None
        """Every known event by its event type, with its scene, built on first use; the causal
        sets ask ``observed`` about thousands of candidates, and each needs the events of a few
        event types only."""
        self._states: dict[str, list[State]] = {}
        self._derived: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._legal: dict[tuple[str, int], dict[str, set[tuple[int, ...]]]] = {}
        self._members: dict[CategoryTerm, np.ndarray] = {}
        self._fixed: dict[tuple[CategoryTerm, str], tuple[int | None, str]] = {}
        universal = self.quantifiers.universal_words
        self.sayable: frozenset[str] = frozenset(
            (MOST, SOME)
            + ((NEC_ALL, NEC_NO) if universal in ("nec", "either") else ())
            + ((ALL, NO) if universal in ("extensional", "either") else ())
        )
        """The quantifiers that the language has words for."""

    # The language's words --------------------------------------------------------------------

    def universal(self, negative: bool) -> str:
        """The strongest quantifier that the language's "all" (or "no") can express."""
        if self.quantifiers.universal_words == "extensional":
            return NO if negative else ALL
        return NEC_NO if negative else NEC_ALL

    def statable(self, proposition: Proposition) -> bool:
        """Whether the language can state a class-level proposition's quantifier: with a word,
        or with a bare plural."""
        return proposition.quantifier in self.sayable or self.bare_plural_expresses(proposition)

    def bare_plural_expresses(self, proposition: Proposition) -> bool:
        """Whether a bare plural can state a class-level proposition: its quantifier (or, for a
        negative fact, its positive counterpart) is in ``quantifiers.bare_plural.expresses``, or
        the proposition is a membership fact or a rule statement with a ``nec`` quantifier."""
        quantifier = proposition.quantifier
        if quantifier is None:
            return True  # a class-level comparison or comparative takes no quantifier word
        if quantifier in NEC_QUANTIFIERS and (
            proposition.predicate.kind == MEMBER
            or proposition.rule is not None
            or proposition.causal
        ):
            return True
        positive = COUNTERPART[quantifier] if quantifier in (NO, NEC_NO) else quantifier
        return positive in self.quantifiers.bare_plural_expresses

    # Sets of instances -----------------------------------------------------------------------

    def members(self, term: CategoryTerm) -> np.ndarray:
        """The subject set of a category term: the indices of the instances below its category
        that satisfy its restriction and its relative clauses. A scalar pole in the restriction
        is relative to the category: "big penguins" are big for a penguin. A relative clause
        about another category keeps the members related to at least one member of it."""
        if term not in self._members:
            world = self.world
            below = world.below(term.category)
            keep = np.ones(len(below), dtype=bool)
            for literal in term.restriction:
                if literal.pole:
                    keep &= self.pole_mask(literal.feature, term.category)[0][below]
                else:
                    keep &= world.column(literal.feature)[below] == int(literal.positive)
            for clause in term.clauses:
                if clause.kind == CAN:
                    keep &= world.column(clause.label)[below] == 1
                elif clause.patient is not None:
                    others = self.members(clause.patient)
                    keep &= self.matrix(clause.label)[np.ix_(below, others)].any(axis=1)
                else:
                    assert clause.agent is not None
                    others = self.members(clause.agent)
                    keep &= self.matrix(clause.label)[np.ix_(others, below)].any(axis=0)
            self._members[term] = below[keep]
        return self._members[term]

    def comparison_class(self, category: str, scalar: str) -> tuple[int, float, float]:
        """A comparison class as the world holds it: the number of instances below the category
        (every entity for ``ENTITY``, and for the generic noun), the mean of the scalar over
        them, and its population standard deviation, from the runtime's ``comparison_classes``
        (the figures of ``derived/comparison_classes.csv``). The corpus computes none of its
        own (stage b3a)."""
        return comparison_classes(self.world.definition)[(comparison_class_of(category), scalar)]

    def pole_mask(self, pole: str, comparison: str) -> tuple[np.ndarray, float, float]:
        """Which instances count as a scalar pole against a comparison class (a category, or
        ``ENTITY``), with the class's mean and standard deviation, at the language's ``z``. The
        mask is the runtime's ``above`` (``HIGH``) or ``below`` (``LOW``) judged for every
        instance at once, from the same class statistics: an instance is ``HIGH`` when its value
        is at least ``z`` standard deviations above the class's mean, ``LOW`` when at least that
        far below, and a class with no spread has no poles. Cached per pole and class."""
        key = (pole, comparison)
        if key not in self._masks:
            scalar, side = self.world.pole_parts(pole)
            column = self.world.scalar_column(scalar)
            _, mean, sd = self.comparison_class(comparison, scalar)
            if sd == 0:
                mask = np.zeros(self.count, dtype=bool)
            elif side == HIGH:
                mask = column >= mean + self.z * sd
            else:
                mask = column <= mean - self.z * sd
            self._masks[key] = (mask, mean, sd)
        return self._masks[key]

    def cut(self, pole: str) -> float:
        """The language's cut for a numeric pole: ``cut_high`` for the high adjective and
        ``cut_low`` for the low one."""
        return self.cuts[pole.rsplit(".", 1)[1]]

    def condition_of(self, effect: Any) -> tuple[Condition, ...] | None:
        """The condition of an effect record as the corpus states it (:func:`condition_of`
        with this language's cut points): None when a numeric threshold literal of the
        condition is not at a cut."""
        return condition_of(effect, self.cut_points)

    def compare(
        self,
        operator: str,
        dimension: str,
        x: int | Mean,
        y: int | Mean | None = None,
        standard: str | None = None,
        state: State | None = None,
        threshold: float | None = None,
    ) -> bool:
        """One comparison, judged by the world's runtime: ``GREATER`` of ``x`` over ``y``, or
        ``ABOVE`` or ``BELOW`` of ``x`` against the comparison class ``standard`` at the
        language's ``z``. The arguments are entity indices or category means. A comparison of a
        numeric fluent (stage b3b) takes the state at the time point, ``RANGE`` as its
        standard, and the language's cut as its ``threshold``."""
        definition = self.world.definition
        if operator == GREATER:
            assert y is not None
            return greater(definition, dimension, x, y, state)
        assert standard is not None
        judge = above if operator == ABOVE else below
        if standard == RANGE:
            assert threshold is not None
            return judge(definition, dimension, x, RANGE, threshold, state)
        return judge(definition, dimension, x, comparison_class_of(standard), self.z)

    def is_pole(self, label: str) -> bool:
        return self.world.is_pole(label)

    def matrix(self, event_type: str) -> np.ndarray:
        """A two-place event type's requirement, or a category's base relation, over every
        ordered pair of instances, with a false diagonal."""
        return self.world.able(event_type)

    def projection(self, label: str) -> np.ndarray:
        """A patient capacity (``CANBE.<event type>``) for every instance."""
        return self.world.capacity(label)

    def ancestors(self, category: str) -> tuple[str, ...]:
        """The labels of a category's strict ancestors, from the top."""
        return self.categories[category].ancestors

    def disjoint(self, a: str, b: str) -> bool:
        """Whether two categories share no instance: neither is the other or above it."""
        return a != b and a not in self.ancestors(b) and b not in self.ancestors(a)

    # The fixed test --------------------------------------------------------------------------

    def fixed(self, term: CategoryTerm, feature: str) -> tuple[int | None, str]:
        """Whether a PROPERTY feature, a PART feature, or a one-place capacity is fixed for a
        category term, and by which test: ``(1, test)`` or ``(0, test)`` when every possible
        member has that value, and ``(None, test)`` otherwise. A scalar pole in the restriction
        holds nothing, because no rule reads a pole."""
        key = (term.plain, feature)
        if key not in self._fixed:
            restriction = tuple(
                (literal.feature, literal.positive)
                for literal in term.restriction
                if not literal.pole
            )
            self._fixed[key] = self.world.fixed(
                term.category, restriction, feature, self.cone_limit
            )
        return self._fixed[key]

    # Evaluation ------------------------------------------------------------------------------

    def evaluate(self, proposition: Proposition) -> Evaluation:
        """Judge a logical form against the world."""
        if proposition.level == CLASS:
            if proposition.causal or proposition.predicate.causal:
                return self._evaluate_causal(proposition)
            return self._evaluate_class(proposition)
        if proposition.level == INSTANCE:
            return self._evaluate_instance(proposition)
        if proposition.level == EVENT:
            return self._evaluate_event(proposition)
        if proposition.level in TIMED_LEVELS:
            return self._evaluate_timed(proposition)
        return _invalid(f"unknown level {proposition.level!r}")

    def is_true(self, proposition: Proposition) -> bool:
        evaluation = self.evaluate(proposition)
        return evaluation.valid and evaluation.true

    def grounded(self, proposition: Proposition) -> Proposition | None:
        """The proposition with its grounding, when it is true and usable in a document."""
        evaluation = self.evaluate(proposition)
        if not (evaluation.valid and evaluation.felicitous):
            return None
        return dataclasses.replace(proposition, grounding=evaluation.grounding)

    def _term_problem(self, term: Any, what: str) -> str:
        world = self.world
        if not isinstance(term, CategoryTerm):
            return f"the {what} of a class-level proposition is a category term"
        if term.category != THING and term.category not in self.categories:
            return f"unknown category {term.category!r}"
        for literal in term.restriction:
            if literal.pole:
                if not self.is_pole(literal.feature):
                    return f"unknown scalar pole {literal.feature!r}"
                if not literal.positive:
                    return "a scalar pole in a restriction is never negated"
            elif world.feature_kind.get(literal.feature) not in (IS, HAS):
                return f"{literal.feature!r} is not a PROPERTY or PART feature"
        for clause in term.clauses:
            if clause.kind == CAN:
                if world.feature_kind.get(clause.label) != CAN:
                    return f"{clause.label!r} is not a one-place event type"
                if clause.other is not None:
                    return "a one-place event type in a relative clause has no other category"
            elif clause.kind == VERB:
                if clause.label not in world.binary:
                    return f"unknown two-place event type {clause.label!r}"
                if (clause.patient is None) == (clause.agent is None):
                    return (
                        "a two-place event type in a relative clause has a patient category or "
                        "an agent category"
                    )
                problem = self._term_problem(clause.other, "category of a relative clause")
                if problem:
                    return problem
                if clause.other.category == THING:
                    return "a relative clause is about a category, not about the generic noun"
            else:
                return (
                    "a class-level relative clause holds a one-place or a two-place event type, "
                    f"not {clause.kind!r}"
                )
        return ""

    def _predicate_problem(self, predicate: Predicate) -> str:
        world = self.world
        kind, label = predicate.kind, predicate.label
        if kind in FEATURE_KINDS:
            if world.feature_kind.get(label) != kind:
                what = {
                    IS: "a PROPERTY feature",
                    HAS: "a PART feature",
                    CAN: "a one-place event type",
                }
                return f"{label!r} is not {what[kind]}"
        elif kind == COMPARISON:
            comparison = predicate.comparison
            if world.is_numeric_pole(label):
                if comparison != RANGE:
                    return f"a numeric pole ({label}) compares with RANGE, the ends of its range"
            elif not self.is_pole(label):
                return f"unknown scalar pole {label!r}"
            elif comparison is None:
                return "a comparison names its comparison class"
            elif comparison != ENTITY and comparison not in self.categories:
                return f"unknown comparison class {comparison!r}: a category, or ENTITY"
        elif kind == COMPARATIVE:
            if label not in world.scalars and label not in world.numeric_fluents:
                return (
                    f"{label!r} is not a scalar dimension or a numeric fluent: a comparative "
                    "is about one"
                )
            if predicate.patient is None:
                return "a comparative has a patient: the argument the subject exceeds"
            if predicate.comparison is not None:
                return "a comparative has no comparison class: it compares two values"
        elif kind == MEMBER:
            if label not in self.categories:
                return f"unknown category {label!r}"
        elif kind == PROJECTION:
            if label not in world.patient_capacities:
                return f"unknown patient capacity {label!r}"
        elif kind == VERB:
            if label not in world.binary:
                return f"unknown two-place event type {label!r}"
        elif kind == STATE_KIND:
            if label in world.numeric_fluents:
                if predicate.direction not in CHANGE_DIRECTIONS:
                    return (
                        f"a change of the numeric fluent {label} has a direction, increase or "
                        "decrease; a numeric fluent is never a state, its poles are"
                    )
            elif label not in world.fluents:
                return f"unknown fluent {label!r}"
            elif predicate.direction is not None:
                return f"the Boolean fluent {label} has no direction"
        elif kind == PAIR_STATE:
            # a pair state (stage c4b): a pair fluent, with the second argument as its patient
            if not world.is_pair_fluent(label):
                return f"unknown pair fluent {label!r}"
            if predicate.patient is None:
                return "a pair state has a patient: the second argument of the pair"
        elif kind == PLACE_EVENT:
            if not world.is_place_event_type(label):
                return f"unknown place event type {label!r}"
            if not world.event_names(label):
                return f"{label} has no dynamics: no event is a move or an act of it"
            if predicate.patient is None:
                return "an event of a place event type has a patient: the place"
        elif kind in CAUSAL_KINDS:
            if label in world.numeric_fluents:
                if kind == PRECONDITION:
                    return (
                        f"a precondition on the numeric fluent {label} is a threshold, which no "
                        "word of the language states (decision 23)"
                    )
                if predicate.direction not in EFFECT_DIRECTIONS:
                    return f"an effect statement about {label} has a direction, raises or lowers"
                if predicate.value is not None:
                    return "a numeric effect statement has a direction, not a value"
            elif label not in world.base_fluents:
                return (
                    f"{label!r} is not a base fluent: a causal statement is about a base "
                    "fluent, because the definition guarantees nothing about a derived one"
                )
            elif predicate.direction is not None:
                return f"the Boolean fluent {label} has a value, not a direction"
            elif not isinstance(predicate.value, bool):
                return f"an {kind} statement has a value, true or false"
        else:
            return f"unknown predicate kind {kind!r}"
        if kind not in CAUSAL_KINDS and predicate.value is not None:
            return "only an effect or a precondition has a value"
        if predicate.direction is not None and kind not in (STATE_KIND, EFFECT):
            return "only a numeric change or a numeric effect statement has a direction"
        if predicate.direction is not None and not predicate.numeric:
            return "only a predicate about a numeric fluent has a direction"
        if (kind in (VERB, COMPARATIVE, PAIR_STATE, PLACE_EVENT)) != (
            predicate.patient is not None
        ):
            return (
                "a two-place event type, a comparative, a pair state, or a place event type, "
                "and only one, has a patient"
            )
        if kind != COMPARISON and predicate.comparison is not None:
            return "only a comparison (a scalar pole) has a comparison class"
        return ""

    def _evaluate_class(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        quantifier, polarity = proposition.quantifier, proposition.polarity
        kind = predicate.kind
        problem = self._term_problem(subject, "subject") or self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        assert isinstance(subject, CategoryTerm)
        if kind in COMPARISON_KINDS:
            if quantifier is not None:
                return _invalid(f"a class-level {kind} has no quantifier")
        elif quantifier not in QUANTIFIERS:
            return _invalid(f"unknown quantifier {quantifier!r}")
        if quantifier in UNIVERSALS and not polarity:
            return _invalid(f"{quantifier} never has a negative polarity")
        if quantifier in NEC_QUANTIFIERS:
            if kind not in NEC_KINDS:
                return _invalid(
                    f"{quantifier} applies to one-place predicates only: relation facts and "
                    f"patient capacities take the extensional quantifiers"
                )
            if subject.clauses:
                return _invalid(
                    f"{quantifier} needs a fixed test, and a subject with a relative clause has "
                    f"none: it takes the extensional quantifiers"
                )
        members = self.members(subject)
        if len(members) == 0:
            return _invalid("the subject set is empty")

        if kind in COMPARISON_KINDS:
            if predicate.numeric:
                return _invalid(
                    "a numeric fluent is compared at a time point of a scene, never at the "
                    "class level"
                )
            if subject.restriction or subject.clauses:
                return _invalid(
                    f"a class-level {kind} is a statement about the category's mean, not about "
                    "a restricted set"
                )
            if kind == COMPARISON:
                return self._class_comparison(subject, members, predicate, polarity)
            return self._class_comparative(subject, members, predicate, polarity)
        if kind == MEMBER:
            if quantifier not in UNIVERSALS:
                return _invalid("a membership sentence takes all, nec_all, no, or nec_no")
            return self._class_member(subject, members, predicate.label, quantifier, polarity)

        grounding: dict[str, Any]
        fixed_value: int | None = None
        fixed_test = EXACT
        if kind == VERB:
            patient = predicate.patient
            problem = self._term_problem(patient, "patient")
            if problem:
                return _invalid(problem)
            assert isinstance(patient, CategoryTerm)
            patients = self.members(patient)
            pairs = len(members) * len(patients) - len(np.intersect1d(members, patients))
            if pairs == 0:
                return _invalid("there is no pair of distinct instances")
            count = int(self.matrix(predicate.label)[np.ix_(members, patients)].sum())
            total = pairs
            grounding = {"proportion": _number(count / total), "pairs": total, "test": OBSERVED}
        else:
            if kind == PROJECTION:
                column = self.projection(predicate.label)[members]
            else:
                column = self.world.column(predicate.label)[members]
            count, total = int(column.sum()), len(members)
            grounding = {"proportion": _number(count / total), "instances": total}
            if kind in FEATURE_KINDS and not subject.clauses:
                fixed_value, fixed_test = self.fixed(subject, predicate.label)
                grounding["fixed"] = fixed_value is not None
            grounding["test"] = OBSERVED

        asserted = count if polarity else total - count
        if quantifier in NEC_QUANTIFIERS:
            true = fixed_value == (1 if quantifier == NEC_ALL else 0)
            grounding["test"] = fixed_test
        elif quantifier in (ALL, NO):
            true = asserted == total if quantifier == ALL else count == 0
        elif quantifier == MOST:
            true = asserted > total / 2
        else:
            true = asserted > 0
        felicitous = true and self.statable(proposition)
        if quantifier == MOST:
            felicitous = felicitous and asserted >= self.quantifiers.most_usage_min * total - 1e-9
        if quantifier == SOME and self.quantifiers.some_exclude_all and felicitous:
            # the implicature: "some" is left out when the language's "all" (or "no") is true
            universal = self.universal(not polarity)
            if universal in NEC_QUANTIFIERS:
                universal_true = kind in FEATURE_KINDS and fixed_value == (1 if polarity else 0)
            else:
                universal_true = asserted == total
            felicitous = not universal_true
        return Evaluation(True, true, felicitous, grounding)

    def _class_comparison(
        self, subject: CategoryTerm, members: np.ndarray, predicate: Predicate, polarity: bool
    ) -> Evaluation:
        """A class-level pole: the category's mean against its comparison class, an ancestor
        of the category or ``ENTITY``, judged by the runtime's ``above`` or ``below``."""
        category = subject.category
        if category == THING:
            return _invalid("the generic noun has no comparison class")
        comparison = predicate.comparison
        assert comparison is not None and predicate.operator is not None
        if comparison != ENTITY and comparison not in self.ancestors(category):
            return _invalid(
                f"the comparison class of {category} is one of its ancestors, or ENTITY, not "
                f"{comparison!r}"
            )
        scalar = pole_dimension(predicate.label)
        has_pole = self.compare(predicate.operator, scalar, Mean(category), standard=comparison)
        _, mean, sd = self.comparison_class(comparison, scalar)
        grounding = {
            "value": _number(self.world.scalar_column(scalar)[members].mean()),
            "comparison": comparison,
            "mean": _number(mean),
            "sd": _number(sd),
            "instances": len(members),
            "z": self.z,
            "test": MEAN,
        }
        true = has_pole == polarity
        return Evaluation(True, true, true, grounding)

    def _class_comparative(
        self, subject: CategoryTerm, members: np.ndarray, predicate: Predicate, polarity: bool
    ) -> Evaluation:
        """A class-level comparative: the subject category's mean exceeds the patient
        category's, by the runtime's ``greater``. Never negated."""
        if not polarity:
            return _invalid("a comparative is never negated: say it the other way round")
        patient = predicate.patient
        problem = self._term_problem(patient, "patient")
        if problem:
            return _invalid(problem)
        assert isinstance(patient, CategoryTerm)
        if THING in (subject.category, patient.category):
            return _invalid("a comparative compares the means of two categories, not of things")
        if patient.restriction or patient.clauses:
            return _invalid("a class-level comparative compares the means of plain categories")
        if patient.category == subject.category:
            return _invalid("a category is not compared with itself")
        others = self.members(patient)
        scalar = predicate.label
        true = self.compare(GREATER, scalar, Mean(subject.category), Mean(patient.category))
        column = self.world.scalar_column(scalar)
        grounding = {
            "value": _number(column[members].mean()),
            "other": _number(column[others].mean()),
            "instances": len(members),
            "other_instances": len(others),
            "test": MEAN,
        }
        return Evaluation(True, true, true, grounding)

    def _class_member(
        self,
        subject: CategoryTerm,
        members: np.ndarray,
        category: str,
        quantifier: str,
        polarity: bool,
    ) -> Evaluation:
        if subject.category == THING:
            return _invalid("the generic noun is not below any category")
        if subject.category == category:
            return _invalid("a category is not said to be a member of itself")
        above = category in self.ancestors(subject.category)
        apart = self.disjoint(subject.category, category)
        true = apart if quantifier in (NO, NEC_NO) else above
        share = np.isin(members, self.world.below(category)).mean()
        grounding = {"proportion": _number(share), "instances": len(members), "test": TREE}
        felicitous = true and self.statable(
            Proposition(CLASS, subject, Predicate(MEMBER, category), polarity, quantifier)
        )
        return Evaluation(True, true, felicitous, grounding)

    def _evaluate_instance(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        world = self.world
        if proposition.quantifier is not None:
            return _invalid("an instance-level proposition has no quantifier")
        if proposition.scene is not None or proposition.event is not None:
            return _invalid("only an event-level proposition has a scene and an event")
        if proposition.time is not None:
            return _invalid("only a state, a change, or an able_now proposition has a time point")
        if not isinstance(subject, str) or subject not in self.instance_index:
            return _invalid(f"unknown instance {subject!r}")
        problem = self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        if predicate.kind in (STATE_KIND, PAIR_STATE) or predicate.numeric:
            return _invalid("a fluent is stated at a time point of a scene, never timelessly")
        if predicate.kind == PLACE_EVENT:
            return _invalid("an event of a place event type is reported in a scene")
        index = self.instance_index[subject]
        kind, label = predicate.kind, predicate.label
        grounding: dict[str, Any]
        if kind in FEATURE_KINDS:
            value = bool(world.column(label)[index])
            grounding = {"value": int(value), "test": VALUE}
        elif kind == PROJECTION:
            value = bool(self.projection(label)[index])
            grounding = {"value": int(value), "test": VALUE}
        elif kind == MEMBER:
            value = label in self.paths[index]
            grounding = {"value": int(value), "test": TREE}
        elif kind == COMPARISON:
            comparison = predicate.comparison
            assert comparison is not None and predicate.operator is not None
            if comparison not in self.paths[index]:
                return _invalid(f"{subject} is not below the comparison class {comparison!r}")
            scalar = pole_dimension(label)
            value = self.compare(predicate.operator, scalar, index, standard=comparison)
            _, mean, sd = self.comparison_class(comparison, scalar)
            grounding = {
                "value": _number(world.scalar_column(scalar)[index]),
                "mean": _number(mean),
                "sd": _number(sd),
                "z": self.z,
                "test": VALUE,
            }
        elif kind == COMPARATIVE:
            if not proposition.polarity:
                return _invalid("a comparative is never negated: say it the other way round")
            patient = predicate.patient
            if not isinstance(patient, str) or patient not in self.instance_index:
                return _invalid(f"unknown patient instance {patient!r}")
            if patient == subject:
                return _invalid("an instance is never compared with itself")
            other = self.instance_index[patient]
            value = self.compare(GREATER, label, index, other)
            column = world.scalar_column(label)
            grounding = {
                "value": _number(column[index]),
                "other": _number(column[other]),
                "test": VALUE,
            }
        else:
            patient = predicate.patient
            if not isinstance(patient, str) or patient not in self.instance_index:
                return _invalid(f"unknown patient instance {patient!r}")
            if patient == subject:
                return _invalid("an instance is never related to itself")
            value = bool(self.matrix(label)[index, self.instance_index[patient]])
            grounding = {"value": int(value), "test": VALUE}
        true = value == proposition.polarity
        return Evaluation(True, true, true, grounding)

    # Causal statements -----------------------------------------------------------------------

    def has_entry(
        self,
        event_type: str,
        kind: str,
        role: str,
        fluent: str,
        value: bool | None,
        direction: str | None = None,
        condition: tuple[Condition, ...] = (),
    ) -> bool:
        """Whether an event type's definition has the effect (``kind`` ``effect``) or the
        precondition literal (``precondition``) that sets, or requires, the fluent of the role
        to the value; for a numeric effect (stage b3b), the effect of the ``direction``'s sign;
        for a conditional effect statement, the effect under exactly that ``condition`` (a
        numeric threshold literal at the language's cut being its numeric pole, WM.35). A
        category of event types has it when every event type below it does. An unconditional
        statement needs an unconditional effect (stage b1): an effect with a condition
        guarantees nothing after every event."""
        definition = self.world.definition
        for label in self.world.event_types_below(event_type):
            record = definition.event_type(label)
            if kind == EFFECT:
                found = any(
                    effect_matches(e, role, fluent, value, direction, condition, self.cut_points)
                    for e in record.effects
                )
            else:
                # a pair literal is no entry (stage c4a; the statements with pair literals are
                # stage c5a's)
                found = any(
                    not lit.numeric
                    and not lit.pair
                    and lit.role == role
                    and lit.fluent == fluent
                    and lit.value == value
                    for lit in record.precondition
                )
            if not found:
                return False
        return True

    def _condition_problem(self, subject: EventTerm, roles: tuple[str, ...]) -> str:
        """Why a conditional statement's condition cannot be judged, or an empty string."""
        world = self.world
        if subject.condition and subject.event_type not in world.event_types:
            return ""
        for literal in subject.condition:
            if literal.role not in roles:
                return f"{subject.event_type} has no {literal.role} for the condition to be about"
            if literal.label in world.fluents:
                continue
            if world.feature_kind.get(literal.label) in (IS, HAS):
                continue
            if world.is_numeric_pole(literal.label) and literal.value:
                continue  # the numeric adjective, at the language's cut (WM.35)
            if is_numeric(literal.label):
                return (
                    f"a condition on the numeric fluent {literal.label} is a threshold, which the "
                    "language states only at its cut, as the positive numeric adjective (decision "
                    "23 and WM.35)"
                )
            return f"{literal.label!r} is not a PROPERTY or PART feature or a Boolean fluent"
        return ""

    def _evaluate_causal(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        world = self.world
        if not isinstance(subject, EventTerm):
            return _invalid("the subject of a causal statement is an event term")
        if not predicate.causal:
            return _invalid("a causal statement states an effect or a precondition")
        if proposition.quantifier != NEC_ALL:
            return _invalid("a causal statement is nec_all: the definition guarantees it")
        if not proposition.polarity:
            return _invalid("a causal statement is never negated: its value is in its predicate")
        if subject.event_type not in world.event_types:
            return _invalid(f"unknown event type {subject.event_type!r}")
        roles = ROLES[: world.event_types[subject.event_type].arity]
        if subject.role not in roles:
            return _invalid(
                f"{subject.event_type} has no {subject.role}: its roles are {', '.join(roles)}"
            )
        problem = self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        if predicate.patient is not None:
            return _invalid("a causal statement has no patient: the roles are in its subject")
        problem = self._condition_problem(subject, roles)
        if problem:
            return _invalid(problem)
        if subject.condition and predicate.kind != EFFECT:
            return _invalid("only an effect statement has a condition: a precondition has none")
        below = world.event_types_below(subject.event_type)
        able = int(world.able(subject.event_type).sum())
        if able == 0:
            return _invalid(
                f"no binding is able for {subject.event_type}: the statement is about no event"
            )
        holding = sum(
            self.has_entry(
                label,
                predicate.kind,
                subject.role,
                predicate.label,
                predicate.value,
                predicate.direction,
                subject.condition,
            )
            for label in below
        )
        true = holding == len(below)
        grounding = {
            "event_types": len(below),
            "with_entry": holding,
            "able_bindings": able,
            "test": DEFINITION,
        }
        felicitous = true and self.statable(proposition)
        return Evaluation(True, true, felicitous, grounding)

    def observed(self, proposition: Proposition, scenes: list[str] | None = None) -> bool:
        """Whether a causal statement held of every event of its type in the known scenes (or
        in ``scenes``): the participant in the statement's role had the fluent's value at the
        time point after the event's step (an effect) or at the event's time point (a
        precondition), and at least one such event occurred. A false statement that is
        observed is law-like: no observation contradicts it."""
        subject, predicate = proposition.subject, proposition.predicate
        assert isinstance(subject, EventTerm)
        offset = 1 if predicate.kind == EFFECT else 0
        wanted = None if scenes is None else set(scenes)
        seen = False
        for event_type in self.world.event_types_below(subject.event_type):
            for scene, event in self._events_of_type(event_type):
                if wanted is not None and scene not in wanted:
                    continue
                entity = event.agent if subject.role == AGENT else event.patient
                if entity is None:
                    continue
                if not self.condition_held(scene, event, subject.condition):
                    continue  # a conditional statement is about the events under its condition
                seen = True
                if predicate.numeric:
                    # a numeric statement is about the effect: the event's own recorded change
                    # of the fluent, in the direction's sign (stage b3b)
                    if not self.recorded_direction(scene, event, entity, predicate):
                        return False
                    continue
                value = self.fluent_value(scene, event.step + offset, entity, predicate.label)
                if value != predicate.value:
                    return False
        return seen

    def condition_held(
        self, scene: str, event: SceneEvent, condition: tuple[Condition, ...]
    ) -> bool:
        """Whether the condition of a conditional statement held of an event's participants at
        the event's time point: a static feature by the entity's value, a Boolean fluent by the
        replayed state, a numeric pole by the replayed value against the range at the
        language's cut (WM.35)."""
        for literal in condition:
            entity = event.agent if literal.role == AGENT else event.patient
            if entity is None:
                return False
            if literal.numeric:
                held = self.compare(
                    pole_operator(literal.label),
                    pole_dimension(literal.label),
                    self.instance_index[entity],
                    standard=RANGE,
                    state=self.states_of(scene)[event.step - 1],
                    threshold=self.cut(literal.label),
                )
            elif literal.fluent:
                held = self.fluent_value(scene, event.step, entity, literal.label)
            else:
                held = bool(self.world.column(literal.label)[self.instance_index[entity]])
            if held != literal.value:
                return False
        return True

    def recorded_direction(
        self, scene: str, event: SceneEvent, entity: str, predicate: Predicate
    ) -> bool:
        """Whether an event's own recorded changes move a participant's numeric fluent in the
        direction of a numeric effect statement (``raises``: the value after the effect exceeds
        the value before it)."""
        recorded = self._history_event(scene, event.label)
        if recorded is None:
            return False
        return any(
            c.numeric
            and c.entity == entity
            and c.fluent == predicate.label
            and (float(c.to) > float(c.from_)) == (predicate.direction == RAISES)
            for c in recorded.changes
        )

    def _history_event(self, scene: str, label: str):
        history = self.scenes[scene]
        for step in history.steps:
            for recorded in step.events:
                if recorded.label == label:
                    return recorded
        return None

    def _events_of_type(self, event_type: str) -> tuple[tuple[str, SceneEvent], ...]:
        """The known events of one event type, each with its scene, in scene order."""
        if self._events_by_type is None:
            by_type: dict[str, list[tuple[str, SceneEvent]]] = {}
            for scene, events in self._events.items():
                for event in events:
                    by_type.setdefault(event.type, []).append((scene, event))
            self._events_by_type = {label: tuple(found) for label, found in by_type.items()}
        return self._events_by_type.get(event_type, ())

    # Events ----------------------------------------------------------------------------------

    def add_scene(self, history: History) -> None:
        """Make a scene known, so that event-level propositions about it can be judged."""
        self.scenes[history.label] = history
        self._events[history.label] = scene_events(history)
        self._events_by_type = None
        self._states.pop(history.label, None)
        for key in [k for k in self._derived if k[0] == history.label]:
            del self._derived[key]
        for key in [k for k in self._legal if k[0] == history.label]:
            del self._legal[key]

    def events_of(self, scene: str) -> tuple[SceneEvent, ...]:
        return self._events[scene]

    def verb_names(self, event_type: str) -> tuple[str, ...]:
        """The labels that can name an event of an event type: the event type, then the
        categories above it, from the nearest. A one-place event type has only its own label."""
        return self.world.event_names(event_type)

    def allows(self, label: str, agent: str, patient: str | None) -> bool:
        """Whether the binding is able: the requirement of the event type (or the base relation
        of the category) holds for the agent, or for the agent and the patient, or (stage c4b)
        for the agent and the place of a place event type."""
        row = self.instance_index[agent]
        if patient is None:
            return bool(self.world.able(label)[row])
        return bool(self.world.able(label)[row, self._patient_index(label, patient)])

    def _patient_index(self, label: str, patient: str) -> int:
        """The column of a patient in the able table of an event type: a place's index for a
        place event type, an entity's otherwise."""
        if self.world.is_place_event_type(label):
            return self.world.place_index[patient]
        return self.instance_index[patient]

    def _binding(self, label: str, agent: str, patient: str | None) -> tuple[int, ...]:
        if patient is None:
            return (self.instance_index[agent],)
        return (self.instance_index[agent], self._patient_index(label, patient))

    def states_of(self, scene: str) -> list[State]:
        """The state at every time point of a scene, by replaying its history."""
        if scene not in self._states:
            self._states[scene] = replay(self.world.definition, self.scenes[scene])
        return self._states[scene]

    def legal_in(self, scene: str, label: str, agent: str, patient: str | None) -> bool:
        """Whether the binding was legal at some time point of the scene for the event type, or
        for some event type below the category."""
        world = self.world
        binding = self._binding(label, agent, patient)
        if not self.allows(label, agent, patient):
            return False
        for event_type in world.event_types_below(label):
            if not self.allows(event_type, agent, patient):
                continue
            for state in self.states_of(scene):
                if legal(world.definition, state, event_type, [binding])[0]:
                    return True
        return False

    def _evaluate_event(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        if proposition.quantifier is not None:
            return _invalid("an event-level proposition has no quantifier")
        if proposition.time is not None:
            return _invalid("an event-level proposition has no time point")
        if not proposition.polarity:
            return _invalid("an event-level proposition is never negated")
        if proposition.tense != self.event_tense:
            return _invalid(
                f"events are in the {self.event_tense} tense (propositions.events.tense), and "
                f"the proposition has {proposition.tense!r}"
            )
        if proposition.aspect not in ASPECTS:
            return _invalid(f"a report is simple or progressive, not {proposition.aspect!r}")
        scene = self.scenes.get(proposition.scene)
        if scene is None:
            return _invalid(f"unknown scene {proposition.scene!r}")
        if predicate.kind not in EVENT_KINDS:
            return _invalid(
                "an event is a one-place event type, a two-place event type with a patient "
                "instance, or a place event type with a place"
            )
        problem = self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        patient = predicate.patient
        if predicate.kind in (VERB, PLACE_EVENT) and not isinstance(patient, str):
            return _invalid("the patient of an event is an instance, or a place")
        if subject not in scene.participants:
            return _invalid(f"{subject!r} takes no part in the scene {scene.label}")
        if predicate.kind == PLACE_EVENT:
            # the patient of a move or an act (stage c4b) is a place of the scene
            if patient not in (scene.places or ()):
                return _invalid(f"{patient!r} is not a place of the scene {scene.label}")
        elif patient is not None and patient not in scene.participants:
            return _invalid(f"{patient!r} takes no part in the scene {scene.label}")
        if patient == subject:
            return _invalid("an instance is never related to itself")
        assert isinstance(subject, str)
        if proposition.event is not None and scene_of(proposition.event) != scene.label:
            return _invalid(f"the event {proposition.event} is not in the scene {scene.label}")
        matching = [
            event
            for event in self.events_of(scene.label)
            if event.agent == subject
            and event.patient == patient
            and predicate.label in self.verb_names(event.type)
            and proposition.event in (None, event.label)
        ]
        grounding: dict[str, Any] = {"scene": scene.label}
        if matching:
            grounding["step"] = matching[0].step
        grounding["able"] = self.allows(predicate.label, subject, patient)
        grounding["legal"] = bool(matching) or self.legal_in(
            scene.label, predicate.label, subject, patient
        )
        grounding["test"] = OCCURRED
        return Evaluation(True, bool(matching), bool(matching), grounding)

    # States, changes, and what was possible at a time point -----------------------------------

    def fluent_value(self, scene: str, time: int, entity: str, fluent: str) -> bool:
        """The value of a fluent, base or derived, of a participant at ``TIME.<time>`` of a
        scene, from the replayed history."""
        world = self.world
        row = self.instance_index[entity]
        state = self.states_of(scene)[time - 1]
        definition = world.definition
        if definition.is_base_fluent(fluent):
            return bool(state.values[row, definition.base_fluent_index(fluent)])
        key = (scene, time)
        if key not in self._derived:
            participants = [self.instance_index[p] for p in self.scenes[scene].participants]
            facts = derive(definition, state, participants)
            self._derived[key] = {
                label: facts.fluent[:, j] for j, label in enumerate(facts.fluent_labels)
            }
            self._derived[key]["__rows__"] = np.asarray(participants)
        rows = self._derived[key]["__rows__"]
        position = int(np.flatnonzero(rows == row)[0])
        return bool(self._derived[key][fluent][position])

    def predicate_holds(self, scene: str, time: int, entity: str, predicate: Predicate) -> bool:
        """Whether a state predicate holds of a participant at ``TIME.<time>``: a Boolean
        fluent's value, or, for a numeric pole (stage b3b), the comparison with the range at the
        language's cut."""
        if predicate.kind == COMPARISON:
            assert predicate.operator is not None
            return self.compare(
                predicate.operator,
                pole_dimension(predicate.label),
                self.instance_index[entity],
                standard=RANGE,
                state=self.states_of(scene)[time - 1],
                threshold=self.cut(predicate.label),
            )
        return self.fluent_value(scene, time, entity, predicate.label)

    def numeric_value(self, scene: str, time: int, entity: str, fluent: str) -> float:
        """The value of a numeric fluent of a participant at ``TIME.<time>`` of a scene, from
        the replayed history (stage b3b)."""
        state = self.states_of(scene)[time - 1]
        return float(self.world.numeric_column(state, fluent)[self.instance_index[entity]])

    def numeric_caused_by(
        self, scene: str, time: int, entity: str, fluent: str, direction: str
    ) -> str | None:
        """The event of step ``time`` whose own effect moved a participant's numeric fluent in
        the direction (``increase``: its recorded change raised the value), or None when no
        event's effect did: the drift alone moved it."""
        history = self.scenes[scene]
        step = next((s for s in history.steps if s.step == time), None)
        if step is None:
            return None
        for event in step.events:
            for change in event.changes:
                if change.numeric and change.entity == entity and change.fluent == fluent:
                    if (float(change.to) > float(change.from_)) == (direction == INCREASE):
                        return event.label
        return None

    def legal_at(self, scene: str, time: int, label: str, agent: str, patient: str | None) -> bool:
        """Whether a binding was legal at ``TIME.<time>`` of a scene, for the event type or for
        some event type below the category; for a place event type (stage c4b), the binding of
        the agent and a place of the scene."""
        world = self.world
        binding = self._binding(label, agent, patient)
        table = self.legal_table(scene, time)
        return any(binding in table[event_type] for event_type in world.event_types_below(label))

    def legal_table(self, scene: str, time: int) -> dict[str, set[tuple[int, ...]]]:
        """The legal bindings of every event type among the participants of a scene at
        ``TIME.<time>``, as sets of entity-index tuples (the agent and a place index for a place
        event type, over the scene's places, stage c4b), computed once per time point."""
        key = (scene, time)
        if key not in self._legal:
            world = self.world
            history = self.scenes[scene]
            state = self.states_of(scene)[time - 1]
            participants = [self.instance_index[p] for p in history.participants]
            table = {
                label: set(legal_bindings(world.definition, state, label, participants))
                for label in world.binary_leaves + world.unary
            }
            if history.places is not None:
                places = [world.place_index[q] for q in history.places]
                for label in world.place_event_types:
                    if world.event_names(label):
                        table[label] = set(
                            legal_bindings(world.definition, state, label, participants, places)
                        )
            self._legal[key] = table
        return self._legal[key]

    def pair_value(self, scene: str, time: int, fluent: str, first: str, second: str) -> bool:
        """Whether a pair holds at ``TIME.<time>`` of a scene (stage c4b): the pair of a pair
        fluent between two participants, or of a location fluent from a participant to a place,
        from the replayed history."""
        definition = self.world.definition
        state = self.states_of(scene)[time - 1]
        k = definition.pair_index(fluent)
        j = self._pair_target(fluent, second)
        return bool(state.pairs[k][self.instance_index[first], j])

    def _pair_target(self, fluent: str, second: str) -> int:
        if self.world.is_location_fluent(fluent):
            return self.world.place_index[second]
        return self.instance_index[second]

    def pair_caused_by(
        self, scene: str, time: int, fluent: str, first: str, second: str
    ) -> str | None:
        """The event of step ``time`` whose own effect changed the pair from ``TIME.<time>`` to
        the next time point (stage c4b): the event that records the pair change, in either
        direction for a symmetric pair fluent; a move records the old place's deletion and the
        new place's add. None when no event of the step records it."""
        history = self.scenes[scene]
        step = next((s for s in history.steps if s.step == time), None)
        if step is None:
            return None
        symmetric = self.world.pair_fluent_info[fluent].symmetric
        for event in step.events:
            for change in event.changes:
                if not change.pair or change.pair_fluent != fluent:
                    continue
                if (change.first, change.second) == (first, second) or (
                    symmetric and (change.first, change.second) == (second, first)
                ):
                    return event.label
        return None

    def made_by(self, scene: str, time: int, entity: str, fluent: str, event: str) -> bool:
        """Whether the event's own effect made the change of a participant's fluent from
        ``TIME.<time>`` to the next time point: for a base fluent, the event records the
        change; for a derived fluent, the event's own changes, applied alone to the state at
        ``TIME.<time>``, already change it."""
        definition = self.world.definition
        history = self.scenes[scene]
        step = next((s for s in history.steps if s.step == time), None)
        if step is None:
            return False
        recorded = next((e for e in step.events if e.label == event), None)
        if recorded is None:
            return False
        if definition.is_base_fluent(fluent):
            return any(
                not c.pair and c.entity == entity and c.fluent == fluent for c in recorded.changes
            )
        if not recorded.changes:
            return False
        before = self.states_of(scene)[time - 1]
        row = self.instance_index[entity]
        j = definition.derived_fluent_index(fluent)
        was = bool(derive(definition, before, [row]).fluent[0, j])
        values = np.array(before.values, dtype=np.uint8, copy=True)
        numeric = np.array(before.numeric, dtype=np.float64, copy=True)
        pairs = [np.array(matrix, dtype=bool, copy=True) for matrix in before.pairs]
        for change in recorded.changes:
            if change.pair:
                # a pair change of the event (stage c4a) is applied as recorded, both
                # directions of a symmetric pair; a move records the old place's deletion and
                # the new place's add, so the row of the location fluent follows; a derived
                # fluent whose rule reads SOME or SOMEOF then reads the pairs after the event
                k = definition.pair_index(change.pair_fluent)
                pair_fluent = definition.pair_fluent(change.pair_fluent)
                i = self.instance_index[change.first]
                second = (
                    definition.place_index(change.second)
                    if pair_fluent.total
                    else self.instance_index[change.second]
                )
                pairs[k][i, second] = bool(change.to)
                if pair_fluent.symmetric:
                    pairs[k][second, i] = bool(change.to)
                continue
            # a numeric change of the event (stage b2) is applied as recorded: the value after
            # the effect and before the drift
            if change.numeric:
                numeric[
                    self.instance_index[change.entity], definition.numeric_index(change.fluent)
                ] = float(change.to)
                continue
            values[
                self.instance_index[change.entity], definition.base_fluent_index(change.fluent)
            ] = int(change.to)
        after = State(values, numeric, tuple(pairs))
        return bool(derive(definition, after, [row]).fluent[0, j]) != was

    def caused_by(self, scene: str, time: int, entity: str, fluent: str) -> str | None:
        """The event of step ``time`` of a scene whose own effect changed a participant's
        fluent from ``TIME.<time>`` to the next time point: for a base fluent, the event that
        records the change; for a derived fluent, the first event of the step whose own
        changes, applied alone to the state at ``TIME.<time>``, already change it. None when
        no event of the step made the change on its own."""
        history = self.scenes[scene]
        step = next((s for s in history.steps if s.step == time), None)
        if step is None:
            return None
        for event in step.events:
            if self.made_by(scene, time, entity, fluent, event.label):
                return event.label
        return None

    def _evaluate_timed(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        level = proposition.level
        if proposition.quantifier is not None:
            return _invalid(f"a {level} proposition has no quantifier")
        if proposition.event is not None or proposition.aspect is not None:
            return _invalid(f"a {level} proposition reports no event and has no aspect")
        if proposition.tense != self.event_tense:
            return _invalid(
                f"states and changes take the scene's tense, the {self.event_tense} "
                f"(propositions.events.tense), and the proposition has {proposition.tense!r}"
            )
        scene = self.scenes.get(proposition.scene)
        if scene is None:
            return _invalid(f"unknown scene {proposition.scene!r}")
        try:
            time = time_index(proposition.time or "")
        except ValueError:
            return _invalid(f"{proposition.time!r} is not a time point (TIME.<k>)")
        final = time_index(scene.final)
        last = final - 1 if level == CHANGE else final
        if not 1 <= time <= last:
            what = "a change is from a time point before the last" if level == CHANGE else ""
            return _invalid(
                f"{proposition.time} is not a time point of {scene.label}, which runs from "
                f"{time_label(1)} to {scene.final}" + (f": {what}" if what else "")
            )
        problem = self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        patient = predicate.patient
        if level == ABLE_NOW:
            if predicate.kind not in EVENT_KINDS:
                return _invalid(
                    "an able_now proposition is about a one-place event type, a two-place "
                    "event type with a patient instance, or a place event type with a place"
                )
            if predicate.kind in (VERB, PLACE_EVENT) and not isinstance(patient, str):
                return _invalid("the patient of an able_now proposition is an instance or a place")
        elif predicate.kind == PAIR_STATE:
            # a pair state or a pair change (stage c4b)
            if not isinstance(patient, str):
                return _invalid("the second argument of a pair state is an instance or a place")
            if level == CHANGE and predicate.numeric:
                return _invalid("a pair change has a value, not a direction")
        elif level == STATE and predicate.kind in COMPARISON_KINDS:
            # a numeric pole, or a numeric comparative, at a time point (stage b3b)
            if not predicate.numeric:
                return _invalid(
                    "a comparison of a scalar is timeless: only a numeric fluent is compared at "
                    "a time point"
                )
            if predicate.kind == COMPARATIVE and not isinstance(patient, str):
                return _invalid("the patient of a numeric comparative is an instance")
            if predicate.kind == COMPARATIVE and not proposition.polarity:
                return _invalid("a comparative is never negated: say it the other way round")
        elif predicate.kind != STATE_KIND:
            return _invalid(f"a {level} proposition is about a fluent")
        elif level == STATE and predicate.numeric:
            return _invalid("a numeric fluent is stated through its poles: ABOVE or BELOW")
        elif level == CHANGE and predicate.numeric and not proposition.polarity:
            return _invalid("a numeric change is never negated: its direction says it")
        if subject not in scene.participants:
            return _invalid(f"{subject!r} takes no part in the scene {scene.label}")
        toward_place = predicate.kind == PLACE_EVENT or (
            predicate.kind == PAIR_STATE and self.world.is_location_fluent(predicate.label)
        )
        if patient is not None and toward_place:
            # the second argument of a location, or the patient of a move or an act (stage
            # c4b), is a place of the scene
            if patient not in (scene.places or ()):
                return _invalid(f"{patient!r} is not a place of the scene {scene.label}")
        elif patient is not None and patient not in scene.participants:
            return _invalid(f"{patient!r} takes no part in the scene {scene.label}")
        if patient == subject:
            return _invalid("an instance is never related to itself")
        assert isinstance(subject, str)
        grounding: dict[str, Any] = {"scene": scene.label, "time": proposition.time}
        if level == ABLE_NOW:
            able = self.allows(predicate.label, subject, patient)
            legal_now = self.legal_at(scene.label, time, predicate.label, subject, patient)
            grounding.update({"able": able, "legal": legal_now, "test": HISTORY})
            return Evaluation(True, legal_now == proposition.polarity, True, grounding)
        if predicate.kind == PAIR_STATE:
            return self._evaluate_pair_state(proposition, scene, time, grounding)
        fluent = predicate.label
        if predicate.kind in COMPARISON_KINDS:
            return self._evaluate_numeric_comparison(proposition, scene, time, grounding)
        if predicate.numeric:
            assert isinstance(predicate.direction, str)
            before = self.numeric_value(scene.label, time, subject, fluent)
            after = self.numeric_value(scene.label, time + 1, subject, fluent)
            true = after > before if predicate.direction == INCREASE else after < before
            grounding.update(
                {
                    "from": _number(before),
                    "to": _number(after),
                    "caused_by": self.numeric_caused_by(
                        scene.label, time, subject, fluent, predicate.direction
                    )
                    if true
                    else None,
                    "test": HISTORY,
                }
            )
            return Evaluation(True, true, true, grounding)
        value = self.fluent_value(scene.label, time, subject, fluent)
        initial = self.fluent_value(scene.label, 1, subject, fluent)
        if level == STATE:
            grounding.update(
                {
                    "value": int(value),
                    "initial": int(initial),
                    "changed": value != initial,
                    "test": HISTORY,
                }
            )
            return Evaluation(True, value == proposition.polarity, True, grounding)
        after = self.fluent_value(scene.label, time + 1, subject, fluent)
        true = value != proposition.polarity and after == proposition.polarity
        grounding.update(
            {
                "from": int(value),
                "to": int(after),
                "caused_by": self.caused_by(scene.label, time, subject, fluent)
                if value != after
                else None,
                "test": HISTORY,
            }
        )
        return Evaluation(True, true, True, grounding)

    def _evaluate_pair_state(
        self, proposition: Proposition, scene: History, time: int, grounding: dict[str, Any]
    ) -> Evaluation:
        """A pair state (``HOLDS`` around ``PAIRFL.k(REF.a, REF.b)``) or a pair change
        (``BECOME``) at a time point of a scene (stage c4b), judged on the replayed pairs as a
        Boolean fluent is judged on the replayed values: a state holds when the pair holds at
        the time point; a change when the pair lacks the stated value at the time point and has
        it at the next, its grounding naming the event of the step that recorded the change."""
        subject, predicate = proposition.subject, proposition.predicate
        assert isinstance(subject, str) and isinstance(predicate.patient, str)
        fluent, patient = predicate.label, predicate.patient
        value = self.pair_value(scene.label, time, fluent, subject, patient)
        if proposition.level == STATE:
            initial = self.pair_value(scene.label, 1, fluent, subject, patient)
            grounding.update(
                {
                    "value": int(value),
                    "initial": int(initial),
                    "changed": value != initial,
                    "test": HISTORY,
                }
            )
            return Evaluation(True, value == proposition.polarity, True, grounding)
        after = self.pair_value(scene.label, time + 1, fluent, subject, patient)
        true = value != proposition.polarity and after == proposition.polarity
        grounding.update(
            {
                "from": int(value),
                "to": int(after),
                "caused_by": self.pair_caused_by(scene.label, time, fluent, subject, patient)
                if value != after
                else None,
                "test": HISTORY,
            }
        )
        return Evaluation(True, true, True, grounding)

    def _evaluate_numeric_comparison(
        self, proposition: Proposition, scene: History, time: int, grounding: dict[str, Any]
    ) -> Evaluation:
        """A numeric pole (``HOLDS`` around ``ABOVE`` or ``BELOW`` against ``RANGE`` at the
        language's cut) or a numeric comparative (``HOLDS`` around ``GREATER``) at a time point
        of a scene, judged on the replayed numeric values (stage b3b)."""
        subject, predicate = proposition.subject, proposition.predicate
        assert isinstance(subject, str) and predicate.operator is not None
        row = self.instance_index[subject]
        state = self.states_of(scene.label)[time - 1]
        if predicate.kind == COMPARISON:
            fluent = pole_dimension(predicate.label)
            cut = self.cut(predicate.label)
            has_pole = self.compare(
                predicate.operator, fluent, row, standard=RANGE, state=state, threshold=cut
            )
            initial_pole = self.compare(
                predicate.operator,
                fluent,
                row,
                standard=RANGE,
                state=self.states_of(scene.label)[0],
                threshold=cut,
            )
            grounding.update(
                {
                    "value": _number(self.numeric_value(scene.label, time, subject, fluent)),
                    "cut": cut,
                    "initial": _number(self.numeric_value(scene.label, 1, subject, fluent)),
                    "changed": has_pole != initial_pole,
                    "test": HISTORY,
                }
            )
            true = has_pole == proposition.polarity
            return Evaluation(True, true, true, grounding)
        patient = predicate.patient
        assert isinstance(patient, str)
        other = self.instance_index[patient]
        true = self.compare(GREATER, predicate.label, row, other, state=state)
        grounding.update(
            {
                "value": _number(self.numeric_value(scene.label, time, subject, predicate.label)),
                "other": _number(self.numeric_value(scene.label, time, patient, predicate.label)),
                "test": HISTORY,
            }
        )
        return Evaluation(True, true, true, grounding)


__all__ = [
    "ABLE_NOW",
    "ABOVE",
    "AGENT",
    "ALL",
    "ASPECTS",
    "BELOW",
    "CAN",
    "CAUSAL_KINDS",
    "CHANGE",
    "CHANGE_DIRECTIONS",
    "CLASS",
    "COMPARATIVE",
    "COMPARISON",
    "COMPARISON_KINDS",
    "COUNTERPART",
    "DECREASE",
    "DEFINITION",
    "DIRECTIONS",
    "EFFECT",
    "EFFECT_DIRECTIONS",
    "EVENT",
    "EVENT_KINDS",
    "EVENT_VARIABLE",
    "EXACT",
    "FEATURE_KINDS",
    "GREATER",
    "HAS",
    "HIGH",
    "HISTORY",
    "INCREASE",
    "INSTANCE",
    "IS",
    "KINDS",
    "LEVELS",
    "LOCAL",
    "LOW",
    "LOWERS",
    "MEAN",
    "MEMBER",
    "MOST",
    "NEC_ALL",
    "NEC_KINDS",
    "NEC_NO",
    "NEC_QUANTIFIERS",
    "NEGATIVE_ORDER",
    "NO",
    "OBSERVED",
    "OCCURRED",
    "OPERATORS",
    "PAIR_STATE",
    "PAST",
    "PATIENT",
    "PLACE_EVENT",
    "POSITIVE_ORDER",
    "PRECONDITION",
    "PRESENT",
    "PROGRESSIVE",
    "PROJECTION",
    "QUANTIFIERS",
    "RAISES",
    "ROLES",
    "SIMPLE",
    "SOME",
    "STATE",
    "STATE_KIND",
    "TENSES",
    "TIMED_LEVELS",
    "TREE",
    "UNIVERSALS",
    "VALUE",
    "VERB",
    "CategoryTerm",
    "Clause",
    "Condition",
    "Evaluation",
    "EventTerm",
    "Literal",
    "Predicate",
    "Proposition",
    "Truth",
    "comparison_class_of",
    "condition_of",
    "direction_pole",
    "effect_matches",
    "event_of",
    "is_event_variable",
    "is_numeric",
    "is_part_feature",
    "is_property_feature",
    "is_scalar_pole",
    "low_word",
    "pole_dimension",
    "pole_direction",
    "pole_operator",
    "scene_of",
    "unconditional_effects",
]
