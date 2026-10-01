"""Layer 2: propositions. Logical forms, and the tests of their truth.

A proposition is a logical form with a level, a polarity, and a truth grounding. This module
holds the logical forms of the three levels, and :class:`Truth`, which checks any logical form
against the world and, for the event level, against the scenes.

**Class level.** The subject is a :class:`CategoryTerm`: a category, or the generic ``THING``,
with a restriction of literals. Its *subject set* is the set of instances below the category that
satisfy the restriction. A proposition with an empty subject set is vacuous, and is never valid.
The quantifier is ``all``, ``most``, ``some``, ``no``, or ``generic``.

**Polarity.** A negative proposition denies its predicate: ``most`` with a negative polarity
says that most members of the subject set lack the predicate. The quantifier ``no`` replaces
sentence negation, so ``no`` always has a positive polarity in the logical form, and ``all``
never has a negative one ("no fish have fur", never "all fish do not have fur").

**Instance level.** The subject is an instance, and truth is read from the instance's values.

**Relative clauses.** A category term can also be restricted by relative clauses, which are
restrictive: "penguins that can swim" are the penguins with the CAN feature, "owls that eat mice"
are the owls that can eat at least one mouse, and "mice that owls eat" are the mice that at least
one owl can eat. With such a subject, ``most``, ``some``, and the generic are judged by the share
of the subject set, and ``all`` and ``no`` are allowed only under the observed reading.

**Event level.** The proposition says that something happened in a scene: the subject is the
agent, and the predicate is a CAN feature, or a verb with a patient instance. The verb can be the
event's own verb or a verb category above it, as a noun can name a category above a leaf. The
proposition is true when such an event occurred in the scene. An event-level proposition is
never negated. Its tense and aspect are part of the logical form: the tense is the corpus's
(``propositions.events.tense``), and the aspect is the event's own. Its grounding says whether
the world allows the event (``possible``), which is what tells an impossible false test item from
one that merely did not happen.

The truth tests follow "Truth grounding" in ``docs/specs/CORPUS_GENERATOR.md``.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.lexicon import SCALAR_POLES, THING
from semantic_world.taxonomy.boolean import settings_array
from semantic_world.taxonomy.features import Feature
from semantic_world.taxonomy.fixed import fixed_by_rule
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.rules import CONE_ENUMERATION_LIMIT, Threshold, evaluate_feature
from semantic_world.taxonomy.tree import Category, Role

CLASS = "class"
INSTANCE = "instance"
EVENT = "event"
LEVELS = (CLASS, INSTANCE, EVENT)

ALL = "all"
MOST = "most"
SOME = "some"
NO = "no"
GENERIC = "generic"
QUANTIFIERS = (ALL, MOST, SOME, NO, GENERIC)

IS = "is"
HAS = "has"
CAN = "can"
SCALAR = "scalar"
MEMBER = "member"
PROJECTION = "projection"
VERB = "verb"
KINDS = (IS, HAS, CAN, SCALAR, MEMBER, PROJECTION, VERB)
FEATURE_KINDS = (IS, HAS, CAN)

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

_JSON_KEY = {
    IS: "feature",
    HAS: "feature",
    CAN: "feature",
    SCALAR: "pole",
    MEMBER: "category",
    PROJECTION: "projection",
    VERB: "verb",
}
_LITERAL_ORDER = {"IS": 0, "HAS": 1, "SC": 2}


def _number(value: float) -> float:
    """A real number as it is written to the output files: 6 decimal places."""
    return round(float(value), 6)


# ---------------------------------------------------------------------------------------------
# Logical forms
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Literal:
    """One literal of a restriction: an IS or HAS feature, positive or negative, or a scalar
    pole (``SC.1.HIGH``), which is always positive."""

    feature: str
    positive: bool = True

    @property
    def pole(self) -> bool:
        return self.feature.startswith("SC.")

    @property
    def sort_key(self) -> tuple:
        prefix, _, rest = self.feature.partition(".")
        number, _, pole = rest.partition(".")
        return (_LITERAL_ORDER.get(prefix, 3), int(number) if number.isdecimal() else 0, pole)

    def __str__(self) -> str:
        return self.feature if self.positive else f"not {self.feature}"

    @classmethod
    def parse(cls, text: str) -> Literal:
        if text.startswith("not "):
            return cls(text[4:], False)
        return cls(text)


@dataclass(frozen=True)
class Clause:
    """A relative clause that restricts a category term. It keeps the members that have a CAN
    feature ("penguins that can swim"), the members that have a verb's relation with at least
    one member of another category as its agent ("owls that eat mice", with ``patient``), or as
    its patient ("mice that owls eat", with ``agent``)."""

    kind: str
    """``can`` or ``verb``."""
    label: str
    patient: CategoryTerm | None = None
    """A verb in a subject relative: the other category, which the head acts on."""
    agent: CategoryTerm | None = None
    """A verb in an object relative: the other category, which acts on the head."""

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
    of a class-level verb. The category is a category label or ``THING``. The restriction is a
    set, kept in one order: IS literals, HAS literals, then scalar poles, each by index. The
    relative clauses restrict the category further, and are kept in the order given."""

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
    """``is``, ``has``, ``can``, ``scalar``, ``member``, ``projection``, or ``verb``."""
    label: str
    """The predicate's concept: a feature, a scalar pole, a category, an exposed patient
    projection, or a verb or verb category."""
    patient: CategoryTerm | str | None = None
    """Verbs only: the patient category (class level) or the patient instance (instance
    level)."""
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
    """Class level only."""
    grounding: dict[str, Any] | None = field(default=None, compare=False)
    id: str | None = field(default=None, compare=False)
    rule: tuple[str, int] | None = field(default=None, compare=False)
    """For a rule statement: the determined feature, and the number of the term of its rule's
    minimal DNF, from 1."""
    scene: str | None = None
    """Event level only: the scene the event belongs to."""
    event: str | None = None
    """Event level only: the event that the proposition reports (``SN.8.5``). A false test item
    reports no event, and has None."""
    tense: str | None = None
    """Event level only: ``past`` or ``present``."""
    aspect: str | None = None
    """Event level only: ``simple`` or ``progressive``."""

    @property
    def negative(self) -> bool:
        """Whether the proposition denies its predicate: a negative polarity, or ``no``."""
        return not self.polarity or self.quantifier == NO

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
    """True, and usable in a document: with ``quantifiers.some.exclude_all`` on, ``some`` is
    used only when ``all`` (for a negative proposition, ``no``) is false."""
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
        self, config: Config, result: TaxonomyResult, cone_limit: int = CONE_ENUMERATION_LIMIT
    ) -> None:
        self.result = result
        self.quantifiers = config.quantifiers
        self.event_tense = config.propositions.event_tense
        self.z = config.scalar_z
        self.cone_limit = cone_limit
        self.features = result.features
        self.rules = result.rules
        self.tree = result.tree
        self.instances = result.instances
        self.values = result.instances.values
        self.scalars = result.instances.scalars
        self.scalar_config = result.config.scalars
        self.count = len(result.instances)
        self.instance_index = {label: i for i, label in enumerate(result.instances.labels)}
        self.categories = {c.label: c for c in result.tree.categories}
        self._rules_by_output = {r.output.label: r for r in result.rules.rules}
        self._free_index = {f.label: i for i, f in enumerate(result.features.free)}
        # the instances below every category, and the categories on every instance's path
        every = np.arange(self.count)
        self._below: dict[str, np.ndarray] = {THING: every}
        for category in result.tree.categories:
            self._below[category.label] = result.instances.below(category, result.tree)
        self.paths: list[tuple[str, ...]] = []
        for leaf_index in result.instances.leaf_index:
            leaf = result.tree.categories[int(leaf_index)]
            self.paths.append(tuple(a.label for a in reversed(leaf.ancestors())) + (leaf.label,))
        # verbs and verb categories, and the patient projections of every verb
        self.verbs: tuple[str, ...] = ()
        self._projection: dict[str, np.ndarray] = {}
        if result.verbs is not None and result.projections is not None:
            self.verbs = tuple(c.label for c in result.verbs.categories)
            for i, verb in enumerate(result.projections.verb_labels):
                self._projection[f"CANBE.{verb}"] = result.projections.patient[:, i].astype(bool)
        self._verb_ancestors: dict[str, tuple[str, ...]] = {}
        if result.verbs is not None:
            for category in result.verbs.categories:
                self._verb_ancestors[category.label] = tuple(a.label for a in category.ancestors())
        self.scenes: dict[str, Any] = {}
        """The scenes that event-level propositions are judged against, by label."""
        self._matrices: dict[str, np.ndarray] = {}
        self._members: dict[CategoryTerm, np.ndarray] = {}
        self._fixed: dict[tuple[CategoryTerm, str], tuple[int | None, str]] = {}

    # Sets of instances -----------------------------------------------------------------------

    def members(self, term: CategoryTerm) -> np.ndarray:
        """The subject set of a category term: the indices of the instances below its category
        that satisfy its restriction and its relative clauses. A scalar pole in the restriction
        is relative to the category: "big penguins" are big for a penguin. A relative clause
        about another category keeps the members related to at least one member of it."""
        if term not in self._members:
            below = self._below[term.category]
            keep = np.ones(len(below), dtype=bool)
            for literal in term.restriction:
                if literal.pole:
                    keep &= self.pole_mask(literal.feature, below)[0][below]
                else:
                    column = self.values[below, self.features[literal.feature].position]
                    keep &= column == int(literal.positive)
            for clause in term.clauses:
                if clause.kind == CAN:
                    keep &= self.values[below, self.features[clause.label].position] == 1
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
        scalar, side = self._pole(pole)
        column = self.scalars[:, scalar]
        mean = float(column[comparison].mean())
        sd = float(column[comparison].std())
        if sd == 0:
            return np.zeros(self.count, dtype=bool), mean, sd
        if side == "HIGH":
            return column >= mean + self.z * sd, mean, sd
        return column <= mean - self.z * sd, mean, sd

    def _pole(self, pole: str) -> tuple[int, str]:
        scalar, _, side = pole.rpartition(".")
        return self.features.scalar_labels.index(scalar), side

    def is_pole(self, label: str) -> bool:
        scalar, _, side = label.rpartition(".")
        return side in SCALAR_POLES and scalar in self.features.scalar_labels

    def matrix(self, verb: str) -> np.ndarray:
        """A verb's relation, or a verb category's base relation, over every ordered pair of
        instances, with a false diagonal."""
        if verb not in self._matrices:
            assert self.result.relations is not None
            self._matrices[verb] = self.result.relations.matrix(verb)
        return self._matrices[verb]

    def projection(self, label: str) -> np.ndarray:
        """A patient projection (``CANBE.<verb>``) for every instance."""
        return self._projection[label]

    def ancestors(self, category: str) -> tuple[str, ...]:
        """The labels of a category's strict ancestors, from the top."""
        return tuple(a.label for a in reversed(self.categories[category].ancestors()))

    def disjoint(self, a: str, b: str) -> bool:
        """Whether two categories share no instance: neither is the other or above it."""
        return a != b and a not in self.ancestors(b) and b not in self.ancestors(a)

    # The fixed test --------------------------------------------------------------------------

    def fixed(self, term: CategoryTerm, feature: str) -> tuple[int | None, str]:
        """Whether an IS, HAS, or CAN feature is fixed for a category term, and by which test:
        ``(1, test)`` or ``(0, test)`` when every possible member has that value, and ``(None,
        test)`` otherwise.

        The test is the fixed-by-rule test of the taxonomy generator, with the restriction's
        literals added as fixed. The free features that are defining at the category, and the
        restriction's literals on free features, are held. The other free features in the cone
        are enumerated, with the intervals between the thresholds of every scalar that is free
        to vary. A restriction's literal on a determined feature keeps only the settings that
        satisfy it. A scalar pole in the restriction holds nothing, because no rule reads a
        pole. Above ``2 ** cone_limit`` settings, the local test is used instead.
        """
        key = (term.plain, feature)
        if key not in self._fixed:
            self._fixed[key] = self._fixed_test(term.plain, self.features[feature])
        return self._fixed[key]

    def _fixed_test(self, term: CategoryTerm, feature: Feature) -> tuple[int | None, str]:
        category = None if term.category == THING else self.categories[term.category]
        n_free = len(self.features.free)
        if category is None:
            base = np.zeros(n_free, dtype=np.uint8)
            held = np.zeros(n_free, dtype=bool)
        else:
            base = category.free_values.astype(np.uint8).copy()
            held = category.defining_mask().copy()
        filters: list[tuple[Feature, int]] = []
        for literal in term.restriction:
            if literal.pole:
                continue
            restricted = self.features[literal.feature]
            if restricted.free:
                index = self._free_index[restricted.label]
                if held[index] and base[index] != int(literal.positive):
                    return None, EXACT  # no member can satisfy the restriction
                base[index] = int(literal.positive)
                held[index] = True
            else:
                filters.append((restricted, int(literal.positive)))

        targets = [feature] + [f for f, _ in filters]
        cone = sorted({self._free_index[f.label] for t in targets for f in self.rules.cone(t)})
        open_features = [i for i in cone if not held[i]]
        thresholds = {t.key: t for target in targets for t in self.rules.thresholds_of(target)}
        scalars_fixed = category is not None and self.scalar_config.fixed_below(category.level)
        literal_values: dict[str, np.ndarray] = {}
        groups: dict[int, list[Threshold]] = {}
        for threshold in thresholds.values():
            if not scalars_fixed:
                groups.setdefault(threshold.scalar, []).append(threshold)
        open_groups = [sorted(groups[s], key=lambda t: t.threshold) for s in sorted(groups)]
        rows = 2 ** len(open_features)
        for group in open_groups:
            rows *= len(group) + 1
        if rows > 2**self.cone_limit:
            return self._fixed_local(category, base, held, feature)

        grid = settings_array(len(open_features))
        if open_groups:
            intervals = np.array(
                list(itertools.product(*[range(len(g) + 1) for g in open_groups])), dtype=np.intp
            )
        else:
            intervals = np.zeros((1, 0), dtype=np.intp)
        binary_index = np.repeat(np.arange(grid.shape[0]), intervals.shape[0])
        interval_index = np.tile(np.arange(intervals.shape[0]), grid.shape[0])
        free_values = np.tile(base, (rows, 1))
        free_values[:, open_features] = grid[binary_index]
        if scalars_fixed:
            assert category is not None
            for key, threshold in thresholds.items():
                value = int(category.scalars[threshold.scalar - 1] > threshold.threshold)
                literal_values[key] = np.full(rows, value, dtype=np.uint8)
        for g, group in enumerate(open_groups):
            chosen = intervals[interval_index, g]
            for j, threshold in enumerate(group):
                # a scalar value in interval i makes the first i literals true
                literal_values[threshold.key] = (chosen > j).astype(np.uint8)

        cache: dict[str, np.ndarray] = {}

        def column(target: Feature) -> np.ndarray:
            return evaluate_feature(
                target, free_values, self.features, self._rules_by_output, cache, literal_values
            )

        keep = np.ones(rows, dtype=bool)
        for restricted, value in filters:
            keep &= column(restricted) == value
        outputs = column(feature)[keep]
        if len(outputs) and outputs.min() == outputs.max():
            return int(outputs[0]), EXACT
        return None, EXACT

    def _fixed_local(
        self, category: Category | None, base: np.ndarray, held: np.ndarray, feature: Feature
    ) -> tuple[int | None, str]:
        """The local test, for a cone too large to enumerate: the taxonomy generator's test on a
        copy of the category in which the held features are defining. Literals on determined
        features are left out, which can only miss a fixed feature."""
        if feature.free:
            index = self._free_index[feature.label]
            return (int(base[index]) if held[index] else None), LOCAL
        roles = np.where(held, Role.DEFINING_NEW, Role.UNDIAGNOSTIC).astype(np.int8)
        scalars = np.zeros(self.features.scalar_count) if category is None else category.scalars
        values = self.rules.compute(base[None, :], scalars[None, :])[0]
        if category is None:
            stand_in = Category(THING, (), 0, None, base, values, roles, scalars=scalars)
            fixed, _ = fixed_by_rule(self.rules, stand_in, self.cone_limit, scalars=None)
        else:
            stand_in = dataclasses.replace(category, free_values=base, values=values, roles=roles)
            fixed, _ = fixed_by_rule(self.rules, stand_in, self.cone_limit, self.scalar_config)
        return (int(values[feature.position]) if fixed[feature.position] else None), LOCAL

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
            elif literal.feature not in self.features or self.features[
                literal.feature
            ].type not in (
                IS,
                HAS,
            ):
                return f"{literal.feature!r} is not an IS or HAS feature"
        for clause in term.clauses:
            if clause.kind == CAN:
                if clause.label not in self.features or self.features[clause.label].type != CAN:
                    return f"{clause.label!r} is not a CAN feature"
                if clause.other is not None:
                    return "a CAN feature in a relative clause has no other category"
            elif clause.kind == VERB:
                if clause.label not in self.verbs:
                    return f"unknown verb {clause.label!r}"
                if (clause.patient is None) == (clause.agent is None):
                    return "a verb in a relative clause has a patient category or an agent category"
                problem = self._term_problem(clause.other, "category of a relative clause")
                if problem:
                    return problem
                if clause.other.category == THING:
                    return "a relative clause is about a category, not about the generic noun"
            else:
                return (
                    "a class-level relative clause holds a CAN feature or a verb, not "
                    f"{clause.kind!r}"
                )
        return ""

    def _predicate_problem(self, predicate: Predicate) -> str:
        kind, label = predicate.kind, predicate.label
        if kind in FEATURE_KINDS:
            if label not in self.features or self.features[label].type != kind:
                return f"{label!r} is not {'an' if kind == IS else 'a'} {kind.upper()} feature"
        elif kind == SCALAR:
            if not self.is_pole(label):
                return f"unknown scalar pole {label!r}"
        elif kind == MEMBER:
            if label not in self.categories:
                return f"unknown category {label!r}"
        elif kind == PROJECTION:
            if label not in self._projection:
                return f"unknown patient projection {label!r}"
        elif kind == VERB:
            if label not in self.verbs:
                return f"unknown verb {label!r}"
        else:
            return f"unknown predicate kind {kind!r}"
        if (kind == VERB) != (predicate.patient is not None):
            return "a verb, and only a verb, has a patient"
        return ""

    def _evaluate_class(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
        quantifier, polarity = proposition.quantifier, proposition.polarity
        kind = predicate.kind
        problem = self._term_problem(subject, "subject") or self._predicate_problem(predicate)
        if problem:
            return _invalid(problem)
        assert isinstance(subject, CategoryTerm)
        if quantifier not in QUANTIFIERS:
            return _invalid(f"unknown quantifier {quantifier!r}")
        if quantifier in (ALL, NO) and not polarity:
            return _invalid(f"{quantifier} never has a negative polarity")
        if predicate.comparison is not None:
            return _invalid("a class-level scalar pole takes its comparison class from the tree")
        members = self.members(subject)
        if len(members) == 0:
            return _invalid("the subject set is empty")
        observed = self.quantifiers.all_grounding == "observed"
        if subject.clauses and quantifier in (ALL, NO) and not observed:
            return _invalid(
                "a subject with a relative clause takes all and no only under the observed reading"
            )

        if kind == SCALAR:
            if quantifier != GENERIC:
                return _invalid("a class-level scalar pole takes the generic only")
            return self._class_scalar(subject, members, predicate.label, polarity)
        if kind == MEMBER:
            if quantifier not in (ALL, NO, GENERIC):
                return _invalid("a membership sentence takes all, no, or the generic")
            return self._class_member(subject, members, predicate.label, quantifier, polarity)

        fixed_value: int | None = None
        grounding: dict[str, Any]
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
            by_law = False
        else:
            if kind == PROJECTION:
                if quantifier in (ALL, NO) and not observed:
                    return _invalid(
                        "a patient projection takes all and no only under the observed reading"
                    )
                column = self.projection(predicate.label)[members]
                by_law = False
            else:
                column = self.values[members, self.features[predicate.label].position]
                by_law = not observed and not subject.clauses
            count, total = int(column.sum()), len(members)
            grounding = {"proportion": _number(count / total), "instances": total}
            if kind in FEATURE_KINDS:
                fixed_value, fixed_test = self.fixed(subject, predicate.label)
                grounding["fixed"] = fixed_value is not None
            grounding["test"] = OBSERVED

        def holds_for_all(value: int) -> bool:
            """``all`` (value 1) or ``no`` (value 0): by the fixed test, or over the instances."""
            if by_law:
                return fixed_value == value
            return count == (total if value else 0)

        asserted = count if polarity else total - count
        means = self.quantifiers.generic_means if quantifier == GENERIC else quantifier
        if means == NO:
            true = holds_for_all(0)
        elif means == ALL:
            true = holds_for_all(1 if polarity else 0)
        elif means == MOST:
            true = asserted >= self.quantifiers.most_min_proportion * total - 1e-9
        else:
            true = asserted > 0
        if by_law and means in (ALL, NO):
            grounding["test"] = fixed_test
        felicitous = true
        if quantifier == SOME and self.quantifiers.some_exclude_all:
            felicitous = true and not holds_for_all(1 if polarity else 0)
        return Evaluation(True, true, felicitous, grounding)

    def _class_scalar(
        self, subject: CategoryTerm, members: np.ndarray, pole: str, polarity: bool
    ) -> Evaluation:
        if subject.category == THING:
            return _invalid("the generic noun has no comparison class")
        parent = self.categories[subject.category].parent
        comparison = self._below[THING if parent is None else parent.label]
        scalar, side = self._pole(pole)
        column = self.scalars[:, scalar]
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
            "comparison": THING if parent is None else parent.label,
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
        if quantifier == NO or not polarity:
            true = apart
        else:
            true = above
        share = np.isin(members, self._below[category]).mean()
        grounding = {"proportion": _number(share), "instances": len(members), "test": TREE}
        return Evaluation(True, true, true, grounding)

    def _evaluate_instance(self, proposition: Proposition) -> Evaluation:
        subject, predicate = proposition.subject, proposition.predicate
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
            value = bool(self.values[index, self.features[label].position])
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
            mask, mean, sd = self.pole_mask(label, self._below[comparison])
            value = bool(mask[index])
            grounding = {
                "value": _number(self.scalars[index, self._pole(label)[0]]),
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

    def add_scene(self, scene: Any) -> None:
        """Make a scene known, so that event-level propositions about it can be judged."""
        self.scenes[scene.label] = scene

    def verb_names(self, verb: str) -> tuple[str, ...]:
        """The labels that can name an event of a verb: the verb, then the verb categories above
        it, from the nearest. A CAN feature has only its own label."""
        return (verb,) + self._verb_ancestors.get(verb, ())

    def allows(self, label: str, agent: str, patient: str | None) -> bool:
        """Whether the world allows an event: the agent has the CAN feature, or the verb's
        relation (a verb category's base relation) holds for the agent and the patient."""
        row = self.instance_index[agent]
        if patient is None:
            return bool(self.values[row, self.features[label].position])
        return bool(self.matrix(label)[row, self.instance_index[patient]])

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
            return _invalid(f"an event is simple or progressive, not {proposition.aspect!r}")
        scene = self.scenes.get(proposition.scene)
        if scene is None:
            return _invalid(f"unknown scene {proposition.scene!r}")
        if predicate.kind not in (CAN, VERB):
            return _invalid("an event is a CAN feature, or a verb with a patient instance")
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
        matching = [
            event
            for event in scene.events
            if event.agent == subject
            and event.patient == patient
            and predicate.label in self.verb_names(event.verb)
            and proposition.event in (None, event.label)
            and proposition.aspect == event.aspect
        ]
        grounding: dict[str, Any] = {"scene": scene.label}
        if matching:
            grounding["step"] = matching[0].step
        grounding["possible"] = self.allows(predicate.label, subject, patient)
        grounding["test"] = OCCURRED
        return Evaluation(True, bool(matching), bool(matching), grounding)
