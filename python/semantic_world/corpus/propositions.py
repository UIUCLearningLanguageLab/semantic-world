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

**The language's words** decide what a proposition's words mean, never what the proposition
means: which quantifiers "all" and "no" express (``quantifiers.universal_words``), the least
share at which a speaker says "most" (``quantifiers.most.usage_min``), and the implicature of
"some" (``quantifiers.some.exclude_all``). :class:`Truth` answers both questions: whether a
proposition is true, and whether a document can state it (``felicitous``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.histories import (
    SceneEvent,
    event_of,
    scene_events,
    scene_of,
)
from semantic_world.corpus.world import (
    PART_PREFIX,
    PROPERTY_PREFIX,
    SCALAR_PREFIX,
    THING,
    World,
)
from semantic_world.taxonomy.rules import CONE_ENUMERATION_LIMIT
from semantic_world.world.history import History, replay
from semantic_world.world.runtime import State, legal

CLASS = "class"
INSTANCE = "instance"
EVENT = "event"
LEVELS = (CLASS, INSTANCE, EVENT)

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
SCALAR = "scalar"
MEMBER = "member"
PROJECTION = "patient_capacity"
"""The kind of a patient-capacity predicate (the old ``projection`` kind)."""
VERB = "event_type2"
"""The kind of a two-place event-type predicate (the old ``verb`` kind)."""
KINDS = (IS, HAS, CAN, SCALAR, MEMBER, PROJECTION, VERB)
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
"""A class-level scalar pole: the subject set's mean against its comparison class."""
VALUE = "value"
"""An instance's own value."""
OCCURRED = "event"
"""An event-level proposition: whether such an event occurred in the scene."""

LABEL_KEY = "label"
"""The key under which every predicate and clause of the JSON logical form names its symbol."""
_JSON_KEY = dict.fromkeys(KINDS, LABEL_KEY)
_LITERAL_ORDER = {PROPERTY_PREFIX: 0, PART_PREFIX: 1, SCALAR_PREFIX: 2}


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
        return self.feature.startswith(SCALAR_PREFIX)

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
class Predicate:
    kind: str
    """``property``, ``part``, ``event_type1``, ``scalar``, ``member``, ``patient_capacity``,
    or ``event_type2``."""
    label: str
    """The predicate's concept: a PROPERTY or PART feature, a one-place event type, a scalar
    pole, a category, a patient capacity (``CANBE.<event type>``), or a two-place event type or
    a category of them."""
    patient: CategoryTerm | str | None = None
    """Two-place event types only: the patient category (class level) or the patient instance
    (instance level)."""
    comparison: str | None = None
    """Instance-level scalar poles only: the comparison class, which is the category that the
    subject's noun names ("big for a mouse")."""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, _JSON_KEY[self.kind]: self.label}
        if isinstance(self.patient, CategoryTerm):
            data["patient"] = self.patient.to_json()
        elif self.patient is not None:
            data["patient"] = {"instance": self.patient}
        if self.comparison is not None:
            data["class"] = self.comparison
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Predicate:
        kind = data["kind"]
        patient = data.get("patient")
        if patient is not None:
            patient = (
                patient["instance"] if "instance" in patient else CategoryTerm.from_json(patient)
            )
        return cls(kind, data[_JSON_KEY[kind]], patient, data.get("class"))


@dataclass(frozen=True)
class Proposition:
    """A logical form. Two propositions are equal when they say the same thing: the label, the
    grounding, and the rule a statement comes from take no part in comparisons."""

    level: str
    subject: CategoryTerm | str
    """A category term (class level) or an instance label (instance level)."""
    predicate: Predicate
    polarity: bool = True
    quantifier: str | None = None
    """Class level only. None for a class-level scalar pole, which has no quantifier."""
    grounding: dict[str, Any] | None = field(default=None, compare=False)
    id: str | None = field(default=None, compare=False)
    rule: tuple[str, int] | None = field(default=None, compare=False)
    """For a rule statement: the determined feature, and the number of the term of its rule's
    minimal DNF, from 1."""
    scene: str | None = None
    """Event level only: the scene the event belongs to."""
    event: str | None = None
    """Event level only: the event that the proposition reports
    (``SCENE.8.EVENTINSTANCE.5``). A test item, true or false, names only its scene, and has
    None: it says that some event of the scene was this one."""
    tense: str | None = None
    """Event level only: ``past`` or ``present``."""
    aspect: str | None = None
    """Event level only: ``simple`` or ``progressive``, the report's choice."""

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
        from the comparison class of a scalar pole."""
        labels: list[str] = []
        for term in (self.subject, self.predicate.patient):
            if isinstance(term, CategoryTerm):
                labels += term.concepts()
        labels.append(self.predicate.label)
        if self.predicate.comparison is not None:
            labels.append(self.predicate.comparison)
        return tuple(dict.fromkeys(labels))

    def to_json(self) -> dict[str, Any]:
        """The logical form as it is written to ``documents.jsonl``."""
        data: dict[str, Any] = {"id": self.id, "level": self.level}
        if self.level == CLASS:
            assert isinstance(self.subject, CategoryTerm)
            data["quantifier"] = self.quantifier
            data["polarity"] = self.polarity
            data["subject"] = self.subject.to_json()
        else:
            if self.level == EVENT:
                data["scene"] = self.scene
                data["event"] = self.event
                data["tense"] = self.tense
                data["aspect"] = self.aspect
            data["polarity"] = self.polarity
            data["subject"] = {"instance": self.subject}
        data["predicate"] = self.predicate.to_json()
        if self.rule is not None:
            data["rule"] = {"feature": self.rule[0], "term": self.rule[1]}
        data["grounding"] = self.grounding
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Proposition:
        subject = data["subject"]
        rule = data.get("rule")
        predicate = Predicate.from_json(data["predicate"])
        if data["level"] == CLASS and predicate.comparison is not None:
            # a sentence's logical form writes the comparison class of a class-level pole for
            # the reader. The class comes from the tree, so it is no part of the proposition.
            predicate = dataclasses.replace(predicate, comparison=None)
        return cls(
            level=data["level"],
            subject=subject["instance"]
            if "instance" in subject
            else CategoryTerm.from_json(subject),
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


class Truth:
    """The truth tests of one world, under one corpus configuration."""

    def __init__(
        self, config: Config, world: World, cone_limit: int = CONE_ENUMERATION_LIMIT
    ) -> None:
        self.world = world
        self.quantifiers = config.quantifiers
        self.event_tense = config.propositions.event_tense
        self.z = config.scalar_z
        self.cone_limit = cone_limit
        self.count = world.count
        self.instance_index = world.instance_index
        self.paths = world.paths
        self.categories = world.category
        self.scenes: dict[str, History] = {}
        """The scenes that event-level propositions are judged against, by label."""
        self._events: dict[str, tuple[SceneEvent, ...]] = {}
        self._states: dict[str, list[State]] = {}
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
            return True  # a class-level scalar pole takes no quantifier word
        if quantifier in NEC_QUANTIFIERS and (
            proposition.predicate.kind == MEMBER or proposition.rule is not None
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
                    keep &= self.pole_mask(literal.feature, below)[0][below]
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

    def pole_mask(self, pole: str, comparison: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Which instances count as a scalar pole against a comparison class, with the class's
        mean and standard deviation. An instance is ``HIGH`` when its value is at least ``z``
        standard deviations above the class's mean, and ``LOW`` when at least that far below.
        A class with no spread has no poles."""
        scalar, side = self.world.pole_parts(pole)
        column = self.world.scalar_column(scalar)
        mean = float(column[comparison].mean())
        sd = float(column[comparison].std())
        if sd == 0:
            return np.zeros(self.count, dtype=bool), mean, sd
        if side == "HIGH":
            return column >= mean + self.z * sd, mean, sd
        return column <= mean - self.z * sd, mean, sd

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
            return self._evaluate_class(proposition)
        if proposition.level == INSTANCE:
            return self._evaluate_instance(proposition)
        if proposition.level == EVENT:
            return self._evaluate_event(proposition)
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
        elif kind == SCALAR:
            if not self.is_pole(label):
                return f"unknown scalar pole {label!r}"
        elif kind == MEMBER:
            if label not in self.categories:
                return f"unknown category {label!r}"
        elif kind == PROJECTION:
            if label not in world.patient_capacities:
                return f"unknown patient capacity {label!r}"
        elif kind == VERB:
            if label not in world.binary:
                return f"unknown two-place event type {label!r}"
        else:
            return f"unknown predicate kind {kind!r}"
        if (kind == VERB) != (predicate.patient is not None):
            return "a two-place event type, and only one, has a patient"
        return ""

    def _evaluate_class(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        quantifier, polarity = proposition.quantifier, proposition.polarity
        kind = predicate.kind
        problem = self._term_problem(subject, "subject") or self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        assert isinstance(subject, CategoryTerm)
        if predicate.comparison is not None:
            return _invalid("a class-level scalar pole takes its comparison class from the tree")
        if kind == SCALAR:
            if quantifier is not None:
                return _invalid("a class-level scalar pole has no quantifier")
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

        if kind == SCALAR:
            if subject.restriction or subject.clauses:
                return _invalid(
                    "a class-level scalar pole is a statement about the category, not about a "
                    "restricted set"
                )
            return self._class_scalar(subject, members, predicate.label, polarity)
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

    def _class_scalar(
        self, subject: CategoryTerm, members: np.ndarray, pole: str, polarity: bool
    ) -> Evaluation:
        if subject.category == THING:
            return _invalid("the generic noun has no comparison class")
        parent = self.categories[subject.category].parent
        comparison = self.world.below(THING if parent is None else parent)
        scalar, side = self.world.pole_parts(pole)
        column = self.world.scalar_column(scalar)
        value = float(column[members].mean())
        mean, sd = float(column[comparison].mean()), float(column[comparison].std())
        if sd == 0:
            has_pole = False
        elif side == "HIGH":
            has_pole = value >= mean + self.z * sd
        else:
            has_pole = value <= mean - self.z * sd
        true = has_pole == polarity
        grounding = {
            "value": _number(value),
            "comparison": THING if parent is None else parent,
            "mean": _number(mean),
            "sd": _number(sd),
            "instances": len(members),
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
        if not isinstance(subject, str) or subject not in self.instance_index:
            return _invalid(f"unknown instance {subject!r}")
        problem = self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        index = self.instance_index[subject]
        kind, label = predicate.kind, predicate.label
        if (kind == SCALAR) != (predicate.comparison is not None):
            return _invalid("a scalar pole, and only a scalar pole, has a comparison class")
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
        elif kind == SCALAR:
            comparison = predicate.comparison
            if comparison not in self.paths[index]:
                return _invalid(f"{subject} is not below the comparison class {comparison!r}")
            mask, mean, sd = self.pole_mask(label, world.below(comparison))
            value = bool(mask[index])
            grounding = {
                "value": _number(world.scalar_column(world.pole_parts(label)[0])[index]),
                "mean": _number(mean),
                "sd": _number(sd),
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

    # Events ----------------------------------------------------------------------------------

    def add_scene(self, history: History) -> None:
        """Make a scene known, so that event-level propositions about it can be judged."""
        self.scenes[history.label] = history
        self._events[history.label] = scene_events(history)
        self._states.pop(history.label, None)

    def events_of(self, scene: str) -> tuple[SceneEvent, ...]:
        return self._events[scene]

    def verb_names(self, event_type: str) -> tuple[str, ...]:
        """The labels that can name an event of an event type: the event type, then the
        categories above it, from the nearest. A one-place event type has only its own label."""
        return self.world.event_names(event_type)

    def allows(self, label: str, agent: str, patient: str | None) -> bool:
        """Whether the binding is able: the requirement of the event type (or the base relation
        of the category) holds for the agent, or for the agent and the patient."""
        row = self.instance_index[agent]
        if patient is None:
            return bool(self.world.able(label)[row])
        return bool(self.world.able(label)[row, self.instance_index[patient]])

    def states_of(self, scene: str) -> list[State]:
        """The state at every time point of a scene, by replaying its history."""
        if scene not in self._states:
            self._states[scene] = replay(self.world.definition, self.scenes[scene])
        return self._states[scene]

    def legal_in(self, scene: str, label: str, agent: str, patient: str | None) -> bool:
        """Whether the binding was legal at some time point of the scene for the event type, or
        for some event type below the category."""
        world = self.world
        binding = tuple(world.instance_index[x] for x in (agent, patient) if x is not None)
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
        if predicate.kind not in (CAN, VERB):
            return _invalid(
                "an event is a one-place event type, or a two-place event type with a patient "
                "instance"
            )
        problem = self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        patient = predicate.patient
        if predicate.kind == VERB and not isinstance(patient, str):
            return _invalid("the patient of an event is an instance")
        for instance in (subject, patient):
            if instance is not None and instance not in scene.participants:
                return _invalid(f"{instance!r} takes no part in the scene {scene.label}")
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


__all__ = [
    "ALL",
    "ASPECTS",
    "CAN",
    "CLASS",
    "COUNTERPART",
    "EVENT",
    "EXACT",
    "FEATURE_KINDS",
    "HAS",
    "INSTANCE",
    "IS",
    "KINDS",
    "LEVELS",
    "LOCAL",
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
    "PAST",
    "POSITIVE_ORDER",
    "PRESENT",
    "PROGRESSIVE",
    "PROJECTION",
    "QUANTIFIERS",
    "SCALAR",
    "SIMPLE",
    "SOME",
    "TENSES",
    "TREE",
    "UNIVERSALS",
    "VALUE",
    "VERB",
    "CategoryTerm",
    "Clause",
    "Evaluation",
    "Literal",
    "Predicate",
    "Proposition",
    "Truth",
    "event_of",
    "scene_of",
]
