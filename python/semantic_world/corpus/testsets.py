"""False propositions, and the test sets.

A false proposition is made from a true one by one minimal change:

- **predicate:** another predicate of the same kind that makes the proposition false;
- **subject:** another category or instance of the same level for which it is false;
- **quantifier:** another quantifier that makes it false ("all" for a "most" fact);
- **role** (verbs only): agent and patient exchanged, when the reversed relation does not hold.

Every false item is checked false by the same truth tests that ground the true propositions. A
false item is never vacuous: its subject set has at least one instance.

**Test sets.** A test set holds matched pairs: a true item, and the false item made from it. There
is one test set for each proposition level and change, with ``test_sets.size`` pairs, and three
kinds of item have test sets of their own:

- ``class_<change>`` and ``class_<change>_lawlike``. A false ``all`` or ``no`` item that no
  instance contradicts is *law-like*: every penguin in the world swims, but nothing fixes it. The
  ordinary sets hold only items that observing the instances could decide;
- ``instance_<change>``;
- ``event_<change>_possible`` and ``event_<change>_impossible``. A false event is one that the
  world allows, or one that the world rules out.

**Context.** An instance-level or event-level item names a narrative document, and is a
continuation of it: its noun phrases are definite mentions of referents that the document has
already mentioned. A subject swap brings in another referent of the document.

**Events.** A true event item is an event that its document reports, in a main clause or in a
relative clause. An event item, true or false, names only its scene: it says that some event of
the scene was this one. A false event item is stricter than the truth test asks: no event with
its verb, its agent, and its patient happened in any scene of its document, in either aspect.
When the item names its verb with a verb category, no event of any verb below the category
matches either. The test sentence names neither its scene nor, in most languages, its aspect, so
a weaker rule would give false items whose words are true of the document.

**Model-facing fields and metadata.** An item has two parts. ``input`` holds what a model may be
given: the document, the sentence, its renderings, its tree, and its logical form. ``meta`` holds
the answer and the bookkeeping: the truth label, the grounding, whether the item's logical form
appears in a training document (``seen``), and the marks ``law_like`` and ``possible``. The two
items of a pair never differ in the format of their ``input``: every field that the true item
has, the false item has too, in the same notation (:func:`format_differences`). The metadata
differ by design.

**Streams.** The items are chosen from ``corpus:tests``, each level and change from its own
part. The mentions and the grammar's choices of an item come from parts of ``corpus:mentions``
and ``corpus:grammar`` named by its set and its pair, so a grammar setting never changes what an
item says, and the two items of a pair mention an instance in the same way.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.grammar import NounPhrase, Predication, SentencePlan
from semantic_world.corpus.lexicon import THING
from semantic_world.corpus.logical import logical_form
from semantic_world.corpus.mentions import Mentions, clause_propositions, plan_for
from semantic_world.corpus.planner import CONTENT_KINDS, RELATION, Document, Planner
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    EVENT,
    FEATURE_KINDS,
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
    VERB,
    CategoryTerm,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import as_json
from semantic_world.corpus.renderings import propositional
from semantic_world.corpus.scenes import Event, Scene

PREDICATE = "predicate"
SUBJECT = "subject"
QUANTIFIER = "quantifier"
ROLE = "role"
CHANGES = (PREDICATE, SUBJECT, QUANTIFIER, ROLE)

LAWLIKE = "lawlike"
POSSIBLE = "possible"
IMPOSSIBLE = "impossible"
_ORDINARY = ""

_DRAWS_PER_PAIR = 60
_DRAWS = 300
_PATIENCE_PER_PAIR = 6
"""The draws of true items for the test sets of one level and change: at most 60 for each pair
of one set and 300 more, and no more than 6 for each pair and 300 more without a new pair."""

LEVEL_CHANGES = {
    CLASS: (PREDICATE, SUBJECT, QUANTIFIER, ROLE),
    INSTANCE: (PREDICATE, SUBJECT, ROLE),
    EVENT: (PREDICATE, SUBJECT, ROLE),
}
"""The changes that can apply at each level: only a class-level proposition has a quantifier."""

INPUT_FIELDS = (
    "document",
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
)
"""The model-facing fields of a test item, in order."""


# ---------------------------------------------------------------------------------------------
# False propositions
# ---------------------------------------------------------------------------------------------


def candidates(
    facts: Facts,
    proposition: Proposition,
    change: str,
    instances: Sequence[str] | None = None,
) -> list[Proposition]:
    """Every proposition that differs from ``proposition`` by one change of the given kind,
    true or false, in a fixed order. ``instances`` are the instances that a subject swap can
    bring in: the referents of a document. Without them, a subject swap brings in any instance
    of the world, or, for an event, any participant of its scene."""
    if change not in CHANGES:
        raise ValueError(f"unknown change {change!r}; the changes are {', '.join(CHANGES)}")
    predicate = proposition.predicate
    subject = proposition.subject
    truth = facts.truth

    def with_predicate(labels: Sequence[str]) -> list[Proposition]:
        return [
            dataclasses.replace(proposition, predicate=dataclasses.replace(predicate, label=label))
            for label in labels
            if label != predicate.label
        ]

    if change == PREDICATE:
        if proposition.level == EVENT:
            # an event is a verb with a patient, or a CAN feature: the kind is kept
            return with_predicate(facts.verbs if predicate.kind == VERB else facts.features[CAN])
        if predicate.kind in FEATURE_KINDS:
            return with_predicate(facts.features[predicate.kind])
        if predicate.kind == PROJECTION:
            return with_predicate(facts.projections)
        if predicate.kind == SCALAR:
            return with_predicate(facts.poles)
        if predicate.kind == VERB:
            return with_predicate(facts.verbs)
        level = facts.level[predicate.label]
        return with_predicate([c for c in facts.categories if facts.level[c] == level])

    if change == SUBJECT:
        if isinstance(subject, CategoryTerm):
            if subject.category == THING:
                return []
            level = facts.level[subject.category]
            return [
                dataclasses.replace(proposition, subject=dataclasses.replace(subject, category=c))
                for c in facts.categories
                if c != subject.category and facts.level[c] == level
            ]
        pool: Sequence[str]
        if instances is not None:
            pool = instances
        elif proposition.level == EVENT:
            scene = truth.scenes.get(proposition.scene)
            pool = () if scene is None else scene.participants
        else:
            pool = truth.result.instances.labels
        return [
            dataclasses.replace(proposition, subject=other)
            for other in pool
            if other != subject and other != predicate.patient
        ]

    if change == QUANTIFIER:
        if proposition.level != CLASS:
            return []
        options = (ALL, MOST, SOME, NO, GENERIC) if proposition.polarity else (MOST, SOME, GENERIC)
        return [
            dataclasses.replace(proposition, quantifier=quantifier)
            for quantifier in options
            if quantifier != proposition.quantifier
        ]

    if predicate.kind != VERB:
        return []
    swapped = dataclasses.replace(predicate, patient=subject)
    return [dataclasses.replace(proposition, subject=predicate.patient, predicate=swapped)]


def falsify(
    facts: Facts,
    proposition: Proposition,
    change: str,
    rng: np.random.Generator,
    *,
    instances: Sequence[str] | None = None,
    accept: Callable[[Proposition], bool] | None = None,
) -> Proposition | None:
    """A false proposition made from a true one by one minimal change, drawn at random from the
    changes that give a false, expressible, non-vacuous proposition. None when there is none.
    The false item carries the grounding that shows it false. A false event names no event.
    ``instances`` limits a subject swap (see :func:`candidates`), and ``accept`` is one more
    condition on the false item."""
    options = candidates(facts, proposition, change, instances)
    for index in rng.permutation(len(options)):
        candidate = dataclasses.replace(
            options[int(index)], grounding=None, id=None, rule=None, event=None
        )
        if not facts.expressible(candidate):
            continue
        evaluation = facts.truth.evaluate(candidate)
        if evaluation.valid and not evaluation.true:
            candidate = dataclasses.replace(candidate, grounding=evaluation.grounding)
            if accept is None or accept(candidate):
                return candidate
    return None


def changes_for(proposition: Proposition) -> tuple[str, ...]:
    """The kinds of change that can apply to a proposition at all."""
    changes = [PREDICATE, SUBJECT]
    if proposition.level == CLASS:
        changes.append(QUANTIFIER)
    if proposition.predicate.kind == VERB:
        changes.append(ROLE)
    return tuple(changes)


def law_like(truth: Truth, proposition: Proposition) -> bool:
    """Whether a false class-level proposition is law-like: it says ``all`` or ``no``, by its
    quantifier or as a bare generic that means all, and no instance contradicts it. Every member
    of its subject set has the value it claims, but nothing fixes that value. Only IS, HAS, and
    CAN features are judged by the fixed test, so only they can be law-like."""
    predicate = proposition.predicate
    subject = proposition.subject
    if proposition.level != CLASS or predicate.kind not in FEATURE_KINDS:
        return False
    assert isinstance(subject, CategoryTerm)
    quantifier = proposition.quantifier
    means = truth.quantifiers.generic_means if quantifier == GENERIC else quantifier
    if means not in (ALL, NO):
        return False
    claimed = 0 if means == NO or not proposition.polarity else 1
    column = truth.values[truth.members(subject), truth.features[predicate.label].position]
    return bool((column == claimed).all())


def _can_be_law_like(truth: Truth, true: Proposition, change: str) -> bool:
    """Whether a change of a true proposition could give a law-like false one: the false item
    has a feature for its predicate, and says ``all`` or ``no``. Only a quantifier swap changes
    the quantifier."""
    if true.predicate.kind not in FEATURE_KINDS or change == ROLE:
        return False
    if change == QUANTIFIER:
        return True
    quantifier = true.quantifier
    means = truth.quantifiers.generic_means if quantifier == GENERIC else quantifier
    return means in (ALL, NO)


def happened(truth: Truth, scenes: Sequence[Scene], proposition: Proposition) -> bool:
    """Whether some event of the given scenes has an event-level proposition's verb, agent, and
    patient, in either aspect. A verb category names every verb below it."""
    predicate = proposition.predicate
    return any(
        event.agent == proposition.subject
        and event.patient == predicate.patient
        and predicate.label in truth.verb_names(event.verb)
        for scene in scenes
        for event in scene.events
    )


# ---------------------------------------------------------------------------------------------
# What the documents state
# ---------------------------------------------------------------------------------------------


def stated_propositions(documents: Sequence[Document]) -> set[Proposition]:
    """Every proposition that the documents state, for the ``seen`` mark of a test item:

    - the proposition of every main clause;
    - in a sentence about instances, the proposition of every relative clause, and what every
      noun phrase says of its referent: its noun (membership), and its modifiers.

    An event-level proposition is kept without its event label, as a test item writes it. The
    restriction of a class-level subject asserts nothing, and adds nothing here."""
    stated: set[Proposition] = set()

    def add(proposition: Proposition) -> None:
        if proposition.level == EVENT:
            proposition = dataclasses.replace(proposition, event=None)
        stated.add(proposition)

    for document in documents:
        for sentence in document.sentences:
            add(sentence.proposition)
            if sentence.proposition.level == CLASS:
                continue
            for proposition in clause_propositions(sentence.plan):
                add(proposition)
            for phrase in sentence.plan.noun_phrases():
                if phrase.noun is None:
                    continue
                instance = phrase.referent
                add(Proposition(INSTANCE, instance, Predicate(MEMBER, phrase.noun)))
                for literal in phrase.restriction:
                    if literal.pole:
                        predicate = Predicate(SCALAR, literal.feature, comparison=phrase.noun)
                        add(Proposition(INSTANCE, instance, predicate))
                    else:
                        kind = IS if literal.feature.startswith("IS.") else HAS
                        predicate = Predicate(kind, literal.feature)
                        add(Proposition(INSTANCE, instance, predicate, literal.positive))
    return stated


# ---------------------------------------------------------------------------------------------
# The format of an item
# ---------------------------------------------------------------------------------------------

_FREE_TEXT = ("formal", "conceptual", "propositional", "text")
"""Fields whose text is made from other fields of the item (the tokens, the logical form), so
their notation follows from those fields."""
_OPTIONAL = "clauses"
"""The one field that the logical form leaves out when it is empty."""
_INDICES = re.compile(r"\d+(\.\d+)*")
_EVENT_LABEL = re.compile(r"\bSN\.\d+\.\d+\b")
_SCENE_LABEL = re.compile(r"SN\.\d+")


def notation(value: str) -> str | None:
    """The notation of a label: the label with its indices taken out, so that ``C1.3`` and
    ``C1.3.2`` are written alike. A scene (``SN.8``) and an event (``SN.8.5``) are not written
    alike, so their indices are counted. None for a word without an index, which is a value and
    not a notation."""
    if not any(character.isdigit() for character in value):
        return None
    if value.startswith("SN."):
        return re.sub(r"\d+", "#", value)
    return re.sub(r"\.(HIGH|LOW)$", ".POLE", _INDICES.sub("#", value))


def _count(value: Any, key: str) -> int:
    """How many times a field appears in a JSON value, at any depth."""
    if isinstance(value, dict):
        return sum((k == key) + _count(v, key) for k, v in value.items())
    if isinstance(value, list):
        return sum(_count(v, key) for v in value)
    return 0


def format_differences(a: Any, b: Any, path: str = "input") -> list[str]:
    """The differences of format between the model-facing fields of two test items: a field that
    one has and the other lacks, a field that is null in one only, a value of another type, or a
    label in another notation. Lists can differ in length, and the words of a rendering can
    differ: only their fields and types are compared. An empty list means the same format."""
    if isinstance(a, dict) and isinstance(b, dict):
        # a category term leaves ``clauses`` out when it has none, and a role swap moves the
        # clause with its category: the two forms must hold as many, wherever they stand
        fields_a = [key for key in a if key != _OPTIONAL]
        fields_b = [key for key in b if key != _OPTIONAL]
        if fields_a != fields_b:
            return [f"{path}: the fields {list(a)} and {list(b)} differ"]
        found: list[str] = []
        if path.endswith(".logical_form") and _count(a, _OPTIONAL) != _count(b, _OPTIONAL):
            found.append(f"{path}: the forms do not hold as many relative clauses")
        for key in a:
            if key == "tree" or (key in _FREE_TEXT and type(a[key]) is type(b[key])):
                continue
            if key == _OPTIONAL and key not in b:
                continue
            found += format_differences(a[key], b[key], f"{path}.{key}")
        return found
    if type(a) is not type(b):
        return [f"{path}: {a!r} and {b!r} are not of one type"]
    if isinstance(a, list):
        found = []
        entries = [x for x in a + b if isinstance(x, dict)]
        for entry in entries[1:]:
            found += format_differences(entries[0], entry, f"{path}[]")
        if len(a) == len(b) and not entries:
            for x, y in zip(a, b, strict=True):
                if (x is None) != (y is None):
                    found.append(f"{path}: {x!r} and {y!r} are not of one type")
        return found
    if isinstance(a, str) and notation(a) != notation(b):
        return [f"{path}: {a!r} and {b!r} are not in one notation"]
    return []


def input_problems(item: dict[str, Any]) -> list[str]:
    """What is wrong with the model-facing fields of one test item, true or false: a missing or
    extra field, an event named by its label, or a part of the answer in the logical form. An
    empty list means nothing is wrong."""
    problems: list[str] = []
    if list(item) != list(INPUT_FIELDS):
        problems.append(f"the fields are {list(item)}, not {list(INPUT_FIELDS)}")
        return problems
    form = item["logical_form"]
    for key in ("id", "grounding", "rule"):
        if key in form:
            problems.append(f"the logical form holds {key!r}, which is metadata")
    if _EVENT_LABEL.search(item["propositional"]) or _EVENT_LABEL.search(str(form)):
        problems.append("the item names an event: a test item names only its scene")
    if form["level"] == EVENT:
        if form["event"] is not None:
            problems.append("the logical form names an event")
        if f"EVENT({form['scene']}, " not in item["propositional"]:
            problems.append("the propositional rendering does not name the scene")
    for label in item["events"]:
        if label is not None and not _SCENE_LABEL.fullmatch(label):
            problems.append(f"the verb phrase names {label!r}, which is not a scene")
    if (item["document"] is None) != (form["level"] == CLASS):
        problems.append("an instance-level or event-level item, and no other, names a document")
    return problems


# ---------------------------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One test item."""

    input: dict[str, Any]
    """The model-facing fields: the document that the item continues (null at the class level),
    and the sentence as ``documents.jsonl`` holds one, with its renderings, its tree, and its
    logical form, without its label."""
    meta: dict[str, Any]
    """The answer and the bookkeeping: the set, the pair, the truth label, the grounding, and
    the marks."""
    proposition: Proposition

    @property
    def truth(self) -> bool:
        return self.meta["truth"]

    def to_json(self) -> dict[str, Any]:
        return {"input": self.input, "meta": self.meta}


@dataclass(frozen=True)
class ItemSet:
    """One test set: matched pairs, each a true item and the false item made from it."""

    name: str
    level: str
    change: str
    kind: str
    """``lawlike``, ``possible``, ``impossible``, or empty for an ordinary set."""
    pairs: tuple[tuple[Item, Item], ...]

    @property
    def items(self) -> tuple[Item, ...]:
        """The items in file order: each true item, then its false item."""
        return tuple(item for pair in self.pairs for item in pair)

    def stats(self) -> dict[str, Any]:
        seen = sum(true.meta["seen"] for true, _ in self.pairs)
        return {
            "pairs": len(self.pairs),
            "true_items_seen": seen,
            "true_items_seen_share": round(seen / len(self.pairs), 6) if self.pairs else None,
        }


def set_names(changes: Sequence[str]) -> tuple[tuple[str, str, str, str], ...]:
    """The test sets of a run, as ``(name, level, change, kind)``, in file order."""
    names: list[tuple[str, str, str, str]] = []
    for level in (CLASS, INSTANCE, EVENT):
        for change in changes:
            if change not in LEVEL_CHANGES[level]:
                continue
            for kind in _kinds(level, change):
                names.append((_set_name(level, change, kind), level, change, kind))
    return tuple(names)


def _kinds(level: str, change: str) -> tuple[str, ...]:
    if level == CLASS:
        # a role swap is about a verb, which is never judged by the fixed test
        return (_ORDINARY,) if change == ROLE else (_ORDINARY, LAWLIKE)
    return (POSSIBLE, IMPOSSIBLE) if level == EVENT else (_ORDINARY,)


def _set_name(level: str, change: str, kind: str) -> str:
    return f"{level}_{change}_{kind}" if kind else f"{level}_{change}"


class TestSetBuilder:
    """The test sets of one corpus, made after its documents."""

    __test__ = False  # not a pytest class, whatever its name

    def __init__(self, planner: Planner, documents: Sequence[Document]) -> None:
        self.planner = planner
        self.config = planner.config
        self.facts = planner.facts
        self.truth = planner.truth
        self.streams = planner.streams
        self.size = planner.config.test_sets.size
        self.stated = stated_propositions(documents)
        self.narratives = [d for d in documents if d.scenes and d.referents]
        self._events: dict[str, Event] = {
            event.label: event for d in documents for scene in d.scenes for event in scene.events
        }
        self._reports: dict[str, tuple[str, ...]] = {}
        for document in self.narratives:
            labels = dict.fromkeys(
                label
                for sentence in document.sentences
                for label in sentence.sentence.event_labels
                if label is not None
            )
            self._reports[document.label] = tuple(labels)
        self.reporting = [d for d in self.narratives if self._reports[d.label]]

    # The sets --------------------------------------------------------------------------------

    def build(self) -> list[ItemSet]:
        """Every test set of the run, in file order."""
        sets: list[ItemSet] = []
        for level in (CLASS, INSTANCE, EVENT):
            for change in self.config.test_sets.changes:
                if change in LEVEL_CHANGES[level]:
                    sets += self._group(level, change)
        return sets

    def _group(self, level: str, change: str) -> list[ItemSet]:
        """The test sets of one level and one change. True items are drawn until every set has
        ``test_sets.size`` pairs, or the draws run out. A true item is used once."""
        rng = self.streams.substream("tests", f"{level}_{change}")
        kinds = _kinds(level, change)
        pairs: dict[str, list[tuple[Item, Item]]] = {kind: [] for kind in kinds}
        used: set = set()
        draw = {CLASS: self._draw_class, INSTANCE: self._draw_instance, EVENT: self._draw_event}
        idle = 0
        for _ in range(_DRAWS_PER_PAIR * self.size + _DRAWS if self.size else 0):
            open_kinds = {kind for kind in kinds if len(pairs[kind]) < self.size}
            if not open_kinds or idle > _PATIENCE_PER_PAIR * self.size + _DRAWS:
                break  # every set is full, or the true items have run out
            idle += 1
            drawn = draw[level](rng, change)
            if drawn is None:
                continue
            document, true = drawn
            key = (None if document is None else document.label, true)
            if key in used:
                continue
            if open_kinds == {LAWLIKE} and not _can_be_law_like(self.truth, true, change):
                continue

            def accept(
                candidate: Proposition,
                document: Document | None = document,
                open_kinds: set[str] = open_kinds,
            ) -> bool:
                return self._kind(level, document, candidate) in open_kinds

            referents = None if document is None else tuple(document.referents.values())
            false = falsify(self.facts, true, change, rng, instances=referents, accept=accept)
            if false is None:
                continue
            kind = self._kind(level, document, false)
            assert kind is not None
            name = _set_name(level, change, kind)
            pair = self._pair(name, level, change, len(pairs[kind]) + 1, document, true, false)
            if pair is None:
                continue
            used.add(key)
            pairs[kind].append(pair)
            idle = 0
        return [
            ItemSet(_set_name(level, change, kind), level, change, kind, tuple(pairs[kind]))
            for kind in kinds
        ]

    def _kind(self, level: str, document: Document | None, false: Proposition) -> str | None:
        """The kind of test set that a false item belongs to, or None for a false item that no
        set takes."""
        if level == CLASS:
            return LAWLIKE if law_like(self.truth, false) else _ORDINARY
        if level == INSTANCE:
            return _ORDINARY
        assert document is not None and false.grounding is not None
        if happened(self.truth, document.scenes, false):
            return None  # the words would be true of another scene, or of another aspect
        return POSSIBLE if false.grounding["possible"] else IMPOSSIBLE

    # True items ------------------------------------------------------------------------------

    def _draw_class(self, rng: np.random.Generator, change: str) -> tuple[None, Proposition] | None:
        # only a relation fact has roles to exchange
        kinds = (RELATION,) if change == ROLE else CONTENT_KINDS
        fact = self.planner.draw_class_fact(rng, kinds)
        return None if fact is None else (None, fact)

    def _draw_instance(
        self, rng: np.random.Generator, change: str
    ) -> tuple[Document, Proposition] | None:
        """A true fact about a referent of a narrative document, drawn as a description is: the
        polarity at the instance-level negation rate, and the patient of a verb among the
        document's other referents."""
        if not self.narratives:
            return None
        document = self.narratives[int(rng.integers(len(self.narratives)))]
        referents = tuple(document.referents.values())
        instance = referents[int(rng.integers(len(referents)))]
        others = tuple(r for r in referents if r != instance)
        negative = bool(rng.random() < self.config.propositions.negation_rate[INSTANCE])
        for polarity in (negative, not negative):
            pool = self.facts.instance_facts(instance, polarity, others)
            if change == ROLE:
                pool = tuple(fact for fact in pool if fact.predicate.kind == VERB)
            if pool:
                return document, pool[int(rng.integers(len(pool)))]
        return None

    def _draw_event(
        self, rng: np.random.Generator, change: str
    ) -> tuple[Document, Proposition] | None:
        """An event that a narrative document reports, in a main clause or a relative clause,
        with its verb named at a level drawn by ``mention.verb_level_weights``. The item names
        only the scene."""
        if not self.reporting:
            return None
        document = self.reporting[int(rng.integers(len(self.reporting)))]
        labels = self._reports[document.label]
        event = self._events[labels[int(rng.integers(len(labels)))]]
        if change == ROLE and not event.transitive:
            return None
        report = self.facts.draw_event(rng, event)
        if report is None:
            return None
        unnamed = self.truth.grounded(dataclasses.replace(report, event=None, grounding=None))
        return None if unnamed is None else (document, unnamed)

    # Items -----------------------------------------------------------------------------------

    def _pair(
        self,
        name: str,
        level: str,
        change: str,
        number: int,
        document: Document | None,
        true: Proposition,
        false: Proposition,
    ) -> tuple[Item, Item] | None:
        """The two items of a pair, or None when one of them cannot be said."""
        items = []
        for proposition, truth in ((true, True), (false, False)):
            record = self._input(name, number, document, proposition)
            if record is None:
                return None
            assert proposition.grounding is not None
            meta: dict[str, Any] = {
                "set": name,
                "pair": number,
                "truth": truth,
                "level": level,
                "change": change,
                "seen": truth and dataclasses.replace(proposition, event=None) in self.stated,
            }
            if level == CLASS:
                meta["law_like"] = not truth and law_like(self.truth, proposition)
                rule = proposition.rule
                meta["rule"] = None if rule is None else {"feature": rule[0], "term": rule[1]}
            elif level == EVENT:
                meta["possible"] = proposition.grounding["possible"]
            meta["grounding"] = proposition.grounding
            items.append(Item(record, meta, proposition))
        problems = format_differences(items[0].input, items[1].input)
        for item in items:
            problems += input_problems(item.input)
        if problems:
            raise CorpusError(
                f"the test items of pair {number} of {name} differ in format, or give away "
                f"their answer: {'; '.join(problems)}"
            )
        return items[0], items[1]

    def _input(
        self, name: str, number: int, document: Document | None, proposition: Proposition
    ) -> dict[str, Any] | None:
        """The model-facing fields of an item: its sentence, said as a continuation of its
        document. None when a referent has no noun."""
        planner = self.planner
        predicate = proposition.predicate
        referents: dict[str, str] = {}
        mentions: Mentions | None = None
        if document is None:
            plan = plan_for(self.facts, proposition)
        else:
            referents = {instance: label for label, instance in document.referents.items()}
            cast = tuple(dict.fromkeys(p for scene in document.scenes for p in scene.participants))
            mentions = Mentions(planner.rules, cast)
            mentions.referents = dict(referents)

            def phrase(instance: str, **options: Any) -> NounPhrase:
                # every instance of a pair draws from its own part of the stream, so the two
                # items mention an instance that they share in the same way
                rng = self.streams.substream("mentions", f"tests:{name}.{number}:{instance}")
                assert mentions is not None
                return mentions.noun_phrase(rng, instance, pronoun=False, **options)

            assert isinstance(proposition.subject, str)
            if proposition.level == EVENT:
                subject = phrase(proposition.subject)
            else:
                subject = phrase(
                    proposition.subject,
                    noun=predicate.comparison,
                    exclude=predicate.label if predicate.kind == MEMBER else None,
                    avoid=(predicate.label,),
                )
            target = None if predicate.patient is None else phrase(str(predicate.patient))
            if subject.pronoun or (target is not None and target.pronoun):
                return None  # no noun can name the referent, and "it" would not pick it out
            if proposition.level == EVENT:
                assert proposition.scene is not None
                said = Predication(
                    predicate.kind,
                    predicate.label,
                    True,
                    target,
                    proposition.scene,
                    proposition.tense,
                    proposition.aspect,
                )
            else:
                said = Predication(predicate.kind, predicate.label, proposition.polarity, target)
            plan = SentencePlan(subject, said)
        if plan.proposition() != proposition:
            raise CorpusError(f"the plan of a test item of {name} says another proposition")
        grammar_rng = self.streams.substream("grammar", f"tests:{name}.{number}")
        sentence = planner.realizer.realize(plan, grammar_rng)
        form = logical_form(plan, proposition, referents)
        for key in ("id", "grounding", "rule"):
            form.pop(key, None)
        return {
            "document": None if document is None else document.label,
            "tokens": list(sentence.tokens),
            "words": None,
            "text": None,
            "formal": sentence.formal,
            "conceptual": sentence.conceptual,
            "propositional": propositional(form, self.config.propositional_referents),
            "tree": as_json(sentence.tree),
            "logical_form": form,
            "referents": [
                {"referent": referent, "noun": noun} for referent, noun in sentence.referents
            ],
            "events": list(sentence.event_labels),
            "coreference": [referents.get(r) for r, _ in sentence.referents],
            "distinguished": [
                None if mentions is None else mentions.distinguished(phrase)
                for phrase in sentence.phrases
            ],
            "readings": list(readings(sentence.tree, planner.lexicon, self.config)),
        }


def build_test_sets(planner: Planner, documents: Sequence[Document]) -> list[ItemSet]:
    """The test sets of a corpus: one for each level and change, with sets of their own for the
    law-like items and for the possible and the impossible false events."""
    return TestSetBuilder(planner, documents).build()
