"""Renderings of a sentence: the formal rendering of a sequence of lexemes, and the propositional
rendering of a logical form. The realizer makes the formal and the conceptual renderings of a
sentence from its tokens (``realize.py``), and the spelled rendering comes with the word forms.

**The formal rendering** writes every word as its gloss and its lexeme label: ``the/LEXEME.176
CATEGORY.1.3/LEXEME.5 EVENTTYPE1.2/LEXEME.138``. The gloss of a content lexeme is its concept's
label, and the gloss of a function word is its English gloss.

**The propositional rendering** is the logical form of a sentence, written as a formula. It is
never ambiguous: capacity and event, tense, aspect, number, and the quantifier are explicit in
it, whatever the grammar settings. It is made from the JSON logical form alone
(:func:`propositional`), and it parses back (:func:`parse_propositional`,
:func:`proposition_of`). The notation:

- an atom is a concept label with its arguments: ``CATEGORY.1.3.2(REF.1)`` (membership),
  ``PROPERTY.12(REF.1)``, ``PART.4(REF.1)``, ``CANBE.EVENTTYPE2.1.1(REF.1)``, and
  ``SCALARDIM.1.HIGH(REF.1, CATEGORY.1.3.2)``, a scalar pole with its comparison class. ``NOT``
  before an atom negates it;
- a capacity is wrapped in ``ABLE``: ``ABLE(EVENTTYPE1.3(REF.1))`` and
  ``ABLE(EVENTTYPE2.1.2(REF.1, REF.2))``, with the agent first. ``ABLE`` wraps only event types.
  A negative capacity is ``NOT ABLE(...)``;
- an event is ``EVENT(<event label>, <tense>, <aspect>, <atom>)``:
  ``EVENT(SCENE.8.EVENTINSTANCE.5, PAST, PROGRESSIVE, EVENTTYPE2.1.2(REF.1, REF.2))``. The tense
  and the aspect are always written. A test item names only the scene, ``EVENT(SCENE.8, PAST,
  SIMPLE, EVENTTYPE2.1.2(REF.1, REF.2))``: some event of the scene was this one;
- a state is ``HOLDS(<scene>, <time point>, <tense>, <atom>)``: ``HOLDS(SCENE.8, TIME.2, PAST,
  BOOLFL.3(REF.1))``, and a negative state has ``NOT`` before the atom. A change is ``BECOME``
  with the same arguments: the fluent came to have the atom's value from that time point to the
  next, ``BECOME(SCENE.8, TIME.2, PAST, NOT BOOLFL.3(REF.1))`` for the change to false. What was
  possible at a time point is ``ABLE_NOW(SCENE.8, TIME.2, PAST, EVENTTYPE2.1.2(REF.1,
  REF.2))``, and ``NOT ABLE_NOW(...)`` says that the binding was not legal then. The tense is
  written in all three, as in ``EVENT``, so that the rendering parses back to the logical form;
- ``REF.n`` is one individual, a referent of the document, and ``VAR.n`` is a variable bound by
  a quantifier;
- a class-level form is a quantifier with a restrictor and a scope:
  ``MOST(CATEGORY.1.3(VAR.1) AND PROPERTY.4(VAR.1), ABLE(EVENTTYPE1.3(VAR.1)))``, with
  ``NEC(ALL(...))``, ``ALL``, ``MOST``, ``SOME``, ``NO``, and ``NEC(NO(...))``. A two-place event
  type's patient category is in the restrictor with a variable of its own, so the quantifier
  ranges over pairs: ``MOST(CATEGORY.1.2(VAR.1) AND CATEGORY.1.5(VAR.2),
  ABLE(EVENTTYPE2.2.1(VAR.1, VAR.2)))``;
- a class-level scalar pole is a statement about the category's mean, with no quantifier:
  ``SCALARDIM.1.HIGH(CATEGORY.1.3, CATEGORY.1)``, the category and its comparison class;
- a relative clause about another category means at least one member of it, written ``EXISTS``
  inside the restrictor: ``MOST(CATEGORY.1.2(VAR.1) AND EXISTS(VAR.2, CATEGORY.1.5(VAR.2) AND
  ABLE(EVENTTYPE2.2.1(VAR.1, VAR.2))), PART.3(VAR.1))`` for "owls that eat mice have claws";
- a causal statement binds an event variable in its restrictor, ``EVENT(EVENTVAR.1, atom)``, the
  events of the atom's event type with its participants, and its scope says what holds at the
  time point after the event (``AFTER``) or before it (``BEFORE``):
  ``NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), AFTER(EVENTVAR.1, BOOLFL.3(VAR.2))))``
  for "things that things catch become caught". The agent is ``VAR.1`` and the patient
  ``VAR.2``, whichever the statement is about.

A sentence about instances is a conjunction. Each noun phrase gives its noun and its modifiers,
then the propositions of its relative clause, and the proposition of the main clause comes last.
Variables are numbered in the order of the logical form: the subject, its relative clauses, then
the patient. Word order never changes the rendering.

**Descriptions (CG.63).** The parts that a descriptive mention contributes (its noun, its
modifiers, and the propositions of its relative clause) are descriptions, which identify the
referent; the rest is the assertion. With ``renderings.propositional.descriptions: marked``, the
descriptions come first, inside braces, and the assertion follows: ``{CATEGORY.1.3.2(REF.1) AND
PROPERTY.12(REF.1)} PART.4(REF.1)`` for "the furry dog has legs". A part that an asserted mention
contributes too is asserted. With ``omitted``, the braces and their contents are left out, and
the rendering parses back to the assertion alone (:func:`assertion`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.lexicon import THING, Lexeme
from semantic_world.corpus.propositions import (
    _JSON_KEY,
    ABLE_NOW,
    AGENT,
    ALL,
    CAN,
    CHANGE,
    CLASS,
    EFFECT,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    LABEL_KEY,
    MEMBER,
    MOST,
    NEC_ALL,
    NEC_NO,
    NO,
    PATIENT,
    PRECONDITION,
    PROJECTION,
    SCALAR,
    SOME,
    STATE,
    STATE_KIND,
    TIMED_LEVELS,
    VERB,
    CategoryTerm,
    Clause,
    EventTerm,
    Literal,
    Predicate,
    Proposition,
    event_of,
    is_event_variable,
    scene_of,
)
from semantic_world.corpus.world import (
    CATEGORY_PREFIX,
    FLUENT_PREFIX,
    ONE_PLACE_PREFIX,
    PART_PREFIX,
    PATIENT_CAPACITY_PREFIX,
    PROPERTY_PREFIX,
    SCALAR_PREFIX,
    TWO_PLACE_PREFIX,
)

REFERENT_MODES = ("local", "instance")
DESCRIPTION_MODES = ("marked", "omitted")
MARKED, OMITTED = DESCRIPTION_MODES
QUANTIFIER_NAMES = {ALL: "ALL", MOST: "MOST", SOME: "SOME", NO: "NO"}
"""The operators of the extensional quantifiers. ``nec_all`` and ``nec_no`` are ``NEC(ALL(...))``
and ``NEC(NO(...))``."""
NEC = "NEC"
_QUANTIFIERS = {name: quantifier for quantifier, name in QUANTIFIER_NAMES.items()}
_NEC_OF = {NEC_ALL: ALL, NEC_NO: NO}
_NEC_BACK = {ALL: NEC_ALL, NO: NEC_NO}
VARIABLE_PREFIX = "VAR."


class RenderingError(CorpusError):
    """A logical form that has no propositional rendering, or a rendering that cannot be read."""


def formal_word(lexeme: Lexeme) -> str:
    """One word of the formal rendering: ``<gloss>/<lexeme label>``."""
    return f"{lexeme.gloss}/{lexeme.label}"


def formal(lexemes: Iterable[Lexeme]) -> str:
    """The formal rendering of a sequence of lexemes, with one space between words."""
    return " ".join(formal_word(lexeme) for lexeme in lexemes)


# ---------------------------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    label: str
    arguments: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.label}({', '.join(self.arguments)})"


@dataclass(frozen=True)
class Not:
    body: Any

    def __str__(self) -> str:
        return f"NOT {self.body}"


@dataclass(frozen=True)
class Able:
    body: Atom

    def __str__(self) -> str:
        return f"ABLE({self.body})"


@dataclass(frozen=True)
class Report:
    """An event: its label, its tense, its aspect, and what happened. The label is the scene's
    in a test item, which names no event."""

    event: str
    tense: str
    aspect: str
    body: Atom

    def __str__(self) -> str:
        return f"EVENT({self.event}, {self.tense.upper()}, {self.aspect.upper()}, {self.body})"


@dataclass(frozen=True)
class Timed:
    """A state (``HOLDS``), a change (``BECOME``), or what was possible (``ABLE_NOW``) at a
    time point of a scene: the operator, the scene, the time point, the tense, and the atom,
    negated inside for a state or a change (the fluent's value), and outside for ``ABLE_NOW``
    (``Not(Timed(...))``)."""

    operator: str
    scene: str
    time: str
    tense: str
    body: Any

    def __str__(self) -> str:
        return f"{self.operator}({self.scene}, {self.time}, {self.tense.upper()}, {self.body})"


TIMED_OPERATORS = {STATE: "HOLDS", CHANGE: "BECOME", ABLE_NOW: "ABLE_NOW"}
_LEVEL_OF_OPERATOR = {operator: level for level, operator in TIMED_OPERATORS.items()}


@dataclass(frozen=True)
class BoundEvent:
    """The events of an event type, bound to an event variable in the restrictor of a causal
    statement: ``EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2))``."""

    variable: str
    body: Atom

    def __str__(self) -> str:
        return f"EVENT({self.variable}, {self.body})"


@dataclass(frozen=True)
class Temporal:
    """What holds at the time point after (``AFTER``) or before (``BEFORE``) a bound event: the
    scope of a causal statement, an atom of a fluent, negated for the value false."""

    operator: str
    variable: str
    body: Any

    def __str__(self) -> str:
        return f"{self.operator}({self.variable}, {self.body})"


CAUSAL_OPERATORS = {EFFECT: "AFTER", PRECONDITION: "BEFORE"}
_KIND_OF_CAUSAL_OPERATOR = {operator: kind for kind, operator in CAUSAL_OPERATORS.items()}


@dataclass(frozen=True)
class Described:
    """The descriptions of a sentence about instances (CG.63): the parts in braces, which come
    before the assertion."""

    body: tuple[Any, ...]

    def __str__(self) -> str:
        return "{" + _conjunction(self.body) + "}"


@dataclass(frozen=True)
class Exists:
    variable: str
    body: tuple[Any, ...]

    def __str__(self) -> str:
        return f"EXISTS({self.variable}, {_conjunction(self.body)})"


@dataclass(frozen=True)
class Quantified:
    quantifier: str
    restrictor: tuple[Any, ...]
    scope: Any

    def __str__(self) -> str:
        if self.quantifier in _NEC_OF:
            inner = QUANTIFIER_NAMES[_NEC_OF[self.quantifier]]
            return f"{NEC}({inner}({_conjunction(self.restrictor)}, {self.scope}))"
        name = QUANTIFIER_NAMES[self.quantifier]
        return f"{name}({_conjunction(self.restrictor)}, {self.scope})"


Formula = tuple[Any, ...]
"""A conjunction of formulas, with the descriptions first (:class:`Described`) when the
rendering marks them. A class-level form is one :class:`Quantified`."""


def _conjunction(parts: tuple[Any, ...]) -> str:
    return " AND ".join(str(part) for part in parts)


def write(formula: Formula) -> str:
    """A formula as text: the descriptions in braces, a space, then the assertion."""
    if formula and isinstance(formula[0], Described):
        return f"{formula[0]} {_conjunction(formula[1:])}"
    return _conjunction(formula)


# ---------------------------------------------------------------------------------------------
# From the JSON logical form
# ---------------------------------------------------------------------------------------------


def _literal(text: str, argument: str, comparison: str | None) -> Any:
    literal = Literal.parse(text)
    if literal.pole:
        if comparison is None:
            raise RenderingError(f"the pole {literal.feature} has no comparison class")
        atom = Atom(literal.feature, (argument, comparison))
    else:
        atom = Atom(literal.feature, (argument,))
    return atom if literal.positive else Not(atom)


class _Variables:
    def __init__(self) -> None:
        self.count = 0

    def new(self) -> str:
        self.count += 1
        return f"{VARIABLE_PREFIX}{self.count}"


def _term(term: dict[str, Any], variable: str, variables: _Variables) -> list[Any]:
    """The restrictor of a category term: its category, its literals, and its relative
    clauses."""
    category = term["category"]
    parts: list[Any] = [] if category == THING else [Atom(category, (variable,))]
    parts += [_literal(text, variable, category) for text in term.get("restriction", ())]
    for clause in term.get("clauses", ()):
        label = clause[LABEL_KEY]
        if clause["kind"] == CAN:
            parts.append(Able(Atom(label, (variable,))))
            continue
        other = variables.new()
        if "patient" in clause:
            body = _term(clause["patient"], other, variables)
            body.append(Able(Atom(label, (variable, other))))
        else:
            body = _term(clause["agent"], other, variables)
            body.append(Able(Atom(label, (other, variable))))
        parts.append(Exists(other, tuple(body)))
    if not parts:
        parts.append(Atom(THING, (variable,)))
    return parts


def _predication(
    predicate: dict[str, Any],
    subject: str,
    patient: str | None,
    polarity: bool = True,
    report: tuple[str, str, str] | None = None,
    timed: tuple[str, str, str, str] | None = None,
) -> Any:
    """The proposition of one predicate: an atom, a capacity, an event, or, with ``timed`` (the
    level, the scene, the time point, and the tense), a state, a change, or what was
    possible."""
    kind = predicate["kind"]
    label = predicate[_JSON_KEY[kind]]
    if timed is not None:
        level, scene, time, tense = timed
        if kind == STATE_KIND:
            atom = Atom(label, (subject,))
            return Timed(
                TIMED_OPERATORS[level], scene, time, tense, atom if polarity else Not(atom)
            )
        if kind == VERB:
            if patient is None:
                raise RenderingError(f"the verb {label} has no patient")
            atom = Atom(label, (subject, patient))
        else:
            atom = Atom(label, (subject,))
        body = Timed(TIMED_OPERATORS[level], scene, time, tense, atom)
        return body if polarity else Not(body)
    if kind == STATE_KIND:
        raise RenderingError(f"the fluent {label} is stated at a time point of a scene")
    if kind == VERB:
        if patient is None:
            raise RenderingError(f"the verb {label} has no patient")
        atom = Atom(label, (subject, patient))
    elif kind == SCALAR:
        comparison = predicate.get("class")
        if comparison is None:
            raise RenderingError(f"the pole {label} has no comparison class")
        atom = Atom(label, (subject, comparison))
    else:
        atom = Atom(label, (subject,))
    body: Any = atom
    if report is not None:
        body = Report(report[0], report[1], report[2], atom)
    elif kind in (CAN, VERB):
        body = Able(atom)
    return body if polarity else Not(body)


def _causal(form: dict[str, Any]) -> Formula:
    """The formula of a causal statement: ``NEC(ALL(EVENT(EVENTVAR.1, <event type>(VAR.1[,
    VAR.2])), AFTER|BEFORE(EVENTVAR.1, [NOT] <fluent>(<the role's variable>))))``."""
    subject, predicate = form["subject"], form["predicate"]
    event_type, role = subject["event"], subject["role"]
    variables = {AGENT: "VAR.1"}
    if not event_type.startswith("EVENTTYPE1."):
        variables[PATIENT] = "VAR.2"
    if role not in variables:
        raise RenderingError(f"{event_type} has no {role}")
    bound = BoundEvent("EVENTVAR.1", Atom(event_type, tuple(variables.values())))
    atom = Atom(predicate[LABEL_KEY], (variables[role],))
    operator = CAUSAL_OPERATORS[predicate["kind"]]
    scope = Temporal(operator, "EVENTVAR.1", atom if predicate["value"] else Not(atom))
    return (Quantified(form["quantifier"], (bound,), scope),)


def formula(form: dict[str, Any], referents: str = "local", descriptions: str = MARKED) -> Formula:
    """The formula of a JSON logical form. ``referents`` is ``local``, for the referent labels
    of the document (``R.1``), or ``instance``, for the taxonomy's instance labels.
    ``descriptions`` is ``marked``, for the descriptions in braces before the assertion, or
    ``omitted``, for the assertion alone."""
    if referents not in REFERENT_MODES:
        raise ValueError(f"unknown referent labels {referents!r}")
    if descriptions not in DESCRIPTION_MODES:
        raise ValueError(f"unknown descriptions mode {descriptions!r}")
    predicate = form["predicate"]
    if form["level"] == CLASS:
        if "head" in form["subject"] or predicate["kind"] in CAUSAL_OPERATORS:
            return _causal(form)
        if predicate["kind"] == SCALAR:
            # a statement about the category's mean, with no quantifier
            return (_predication(predicate, form["subject"]["category"], None, form["polarity"]),)
        variables = _Variables()
        subject = variables.new()
        restrictor = _term(form["subject"], subject, variables)
        patient = None
        if "patient" in predicate:
            patient = variables.new()
            restrictor += _term(predicate["patient"], patient, variables)
        scope = _predication(predicate, subject, patient, form["polarity"])
        return (Quantified(form["quantifier"], tuple(restrictor), scope),)

    def name(mention: dict[str, Any]) -> str:
        if referents == "local" and mention.get("referent") is not None:
            return mention["referent"]
        return mention["instance"]

    def mention_parts(mention: dict[str, Any]) -> list[tuple[Any, bool]]:
        """A noun phrase: its noun, its modifiers, and the propositions of its relative
        clause, each after the noun phrases it brings in, and each with whether it is a
        description (the mention is descriptive) or asserted."""
        argument, noun = name(mention), mention.get("noun")
        descriptive = bool(mention.get("descriptive", False))
        parts: list[Any] = [] if noun is None else [Atom(noun, (argument,))]
        parts += [_literal(text, argument, noun) for text in mention.get("restriction", ())]
        found = [(part, descriptive) for part in parts]
        for clause in mention.get("clauses", ()):
            report = None
            if clause.get("event") is not None:
                report = (clause["event"], clause["tense"], clause["aspect"])
            polarity = clause.get("polarity", True)
            if "agent" in clause:
                found += mention_parts(clause["agent"])
                said = _predication(clause, name(clause["agent"]), argument, polarity, report)
                found.append((said, descriptive))
                continue
            other = None
            if "patient" in clause:
                found += mention_parts(clause["patient"])
                other = name(clause["patient"])
            found.append((_predication(clause, argument, other, polarity, report), descriptive))
        return found

    found = mention_parts(form["subject"])
    patient = None
    if "patient" in predicate:
        found += mention_parts(predicate["patient"])
        patient = name(predicate["patient"])
    report = timed = None
    if form["level"] == EVENT:
        # a test item names no event: it is written with the label of its scene
        report = (form["event"] or form["scene"], form["tense"], form["aspect"])
    elif form["level"] in TIMED_LEVELS:
        timed = (form["level"], form["scene"], form["time"], form["tense"])
    main = _predication(predicate, name(form["subject"]), patient, form["polarity"], report, timed)
    # a noun phrase said twice gives the same proposition twice: it is written once, and a
    # part that an asserted mention contributes is asserted
    asserted = tuple(dict.fromkeys(part for part, d in found if not d and part != main))
    described = tuple(
        dict.fromkeys(part for part, d in found if d and part != main and part not in asserted)
    )
    if described and descriptions == MARKED:
        return (Described(described), *asserted, main)
    return (*asserted, main)


def assertion(form: dict[str, Any], referents: str = "local") -> Formula:
    """The formula of a logical form with its descriptions removed: what the ``omitted``
    rendering parses back to."""
    return formula(form, referents, OMITTED)


def propositional(
    form: dict[str, Any], referents: str = "local", descriptions: str = MARKED
) -> str:
    """The propositional rendering of a JSON logical form."""
    return write(formula(form, referents, descriptions))


# ---------------------------------------------------------------------------------------------
# Reading a rendering back
# ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*([(),{}]|[^\s(),{}]+)")


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _TOKEN.findall(text)
        if "".join(self.tokens) != re.sub(r"\s+", "", text):
            raise RenderingError(f"cannot read the rendering {text!r}")
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            wanted = "more" if expected is None else repr(expected)
            raise RenderingError(
                f"cannot read the rendering {self.text!r}: expected {wanted} and found "
                f"{'the end' if token is None else repr(token)}"
            )
        self.position += 1
        return token

    def conjunction(self) -> tuple[Any, ...]:
        parts = [self.item()]
        while self.peek() == "AND":
            self.take()
            parts.append(self.item())
        return tuple(parts)

    def item(self) -> Any:
        word = self.take()
        if word == "NOT":
            return Not(self.item())
        self.take("(")
        if word == "ABLE":
            body = self.item()
            self.take(")")
            if not isinstance(body, Atom):
                raise RenderingError(f"ABLE wraps an atom in {self.text!r}")
            return Able(body)
        if word == "EVENT" and is_event_variable(self.peek()):
            variable = self.take()
            self.take(",")
            body = self.item()
            self.take(")")
            if not isinstance(body, Atom):
                raise RenderingError(f"EVENT binds an atom in {self.text!r}")
            return BoundEvent(variable, body)
        if word in _KIND_OF_CAUSAL_OPERATOR:
            variable = self.take()
            self.take(",")
            body = self.item()
            self.take(")")
            inner = body.body if isinstance(body, Not) else body
            if not is_event_variable(variable) or not isinstance(inner, Atom):
                raise RenderingError(
                    f"{word} holds an atom about an event variable in {self.text!r}"
                )
            return Temporal(word, variable, body)
        if word == "EVENT":
            event = self.take()
            self.take(",")
            tense = self.take().lower()
            self.take(",")
            aspect = self.take().lower()
            self.take(",")
            body = self.item()
            self.take(")")
            if not isinstance(body, Atom):
                raise RenderingError(f"EVENT reports an atom in {self.text!r}")
            return Report(event, tense, aspect, body)
        if word == "EXISTS":
            variable = self.take()
            self.take(",")
            body = self.conjunction()
            self.take(")")
            return Exists(variable, body)
        if word in _LEVEL_OF_OPERATOR:
            scene = self.take()
            self.take(",")
            time = self.take()
            self.take(",")
            tense = self.take().lower()
            self.take(",")
            body = self.item()
            self.take(")")
            inner = body.body if isinstance(body, Not) else body
            if not isinstance(inner, Atom) or (
                isinstance(body, Not) and word == TIMED_OPERATORS[ABLE_NOW]
            ):
                raise RenderingError(f"{word} holds an atom in {self.text!r}")
            return Timed(word, scene, time, tense, body)
        if word == NEC:
            inner = self.item()
            self.take(")")
            if not isinstance(inner, Quantified) or inner.quantifier not in _NEC_BACK:
                raise RenderingError(f"NEC wraps ALL or NO in {self.text!r}")
            return Quantified(_NEC_BACK[inner.quantifier], inner.restrictor, inner.scope)
        if word in _QUANTIFIERS:
            restrictor = self.conjunction()
            self.take(",")
            scope = self.item()
            self.take(")")
            return Quantified(_QUANTIFIERS[word], restrictor, scope)
        arguments = [self.take()]
        while self.peek() == ",":
            self.take()
            arguments.append(self.take())
        self.take(")")
        return Atom(word, tuple(arguments))


def parse_propositional(text: str) -> Formula:
    """The formula that a propositional rendering writes: the descriptions in braces, when
    there are any, then the assertion."""
    parser = _Parser(text)
    described: tuple[Any, ...] = ()
    if parser.peek() == "{":
        parser.take()
        described = (Described(parser.conjunction()),)
        parser.take("}")
    parsed = parser.conjunction()
    if parser.peek() is not None:
        raise RenderingError(f"cannot read the rendering {text!r}: {parser.peek()!r} is left over")
    return described + parsed


def _kind(label: str) -> str:
    """The predicate kind that a concept label belongs to."""
    for prefix, kind in (
        (PROPERTY_PREFIX, IS),
        (PART_PREFIX, HAS),
        (FLUENT_PREFIX, STATE_KIND),
        (PATIENT_CAPACITY_PREFIX, PROJECTION),
        (ONE_PLACE_PREFIX, CAN),
        (SCALAR_PREFIX, SCALAR),
        (TWO_PLACE_PREFIX, VERB),
        (CATEGORY_PREFIX, MEMBER),
    ):
        if label.startswith(prefix):
            return kind
    if label == THING:
        return MEMBER
    raise RenderingError(f"unknown kind of concept {label!r}")


def _is_category(label: str) -> bool:
    return label == THING or label.startswith(CATEGORY_PREFIX)


def _read_term(variable: str, parts: tuple[Any, ...]) -> CategoryTerm:
    """The category term that the restrictor's parts about one variable describe."""
    category = THING
    literals: list[Literal] = []
    clauses: list[Clause] = []
    for part in parts:
        if isinstance(part, Exists):
            link = part.body[-1]
            if not isinstance(link, Able):
                raise RenderingError("a relative clause ends with the relation to its head")
            agent, patient = link.body.arguments
            other = _read_term(part.variable, part.body[:-1])
            if agent == variable:
                clauses.append(Clause(VERB, link.body.label, patient=other))
            else:
                clauses.append(Clause(VERB, link.body.label, agent=other))
        elif isinstance(part, Able):
            clauses.append(Clause(CAN, part.body.label))
        else:
            positive = not isinstance(part, Not)
            atom = part if positive else part.body
            if _kind(atom.label) == MEMBER:
                category = atom.label
            else:
                literals.append(Literal(atom.label, positive))
    return CategoryTerm(category, tuple(literals), tuple(clauses))


def _mentions(part: Any, variable: str) -> bool:
    """Whether a part of a restrictor is about a variable."""
    if isinstance(part, Exists):
        return variable in part.body[-1].body.arguments
    while isinstance(part, Not | Able):
        part = part.body
    return part.arguments[0] == variable


def _causal_proposition(main: Quantified) -> Proposition:
    """The causal statement that a quantified formula with a temporal scope says."""
    scope = main.scope
    assert isinstance(scope, Temporal)
    bound = [part for part in main.restrictor if isinstance(part, BoundEvent)]
    if len(bound) != 1 or len(main.restrictor) != 1 or bound[0].variable != scope.variable:
        raise RenderingError("a causal statement binds one event, which its scope is about")
    value = not isinstance(scope.body, Not)
    atom = scope.body if value else scope.body.body
    event = bound[0].body
    if atom.arguments[0] not in event.arguments:
        raise RenderingError(f"{scope.operator} is about a participant of the event")
    role = AGENT if event.arguments.index(atom.arguments[0]) == 0 else PATIENT
    kind = _KIND_OF_CAUSAL_OPERATOR[scope.operator]
    return Proposition(
        CLASS,
        EventTerm(event.label, role),
        Predicate(kind, atom.label, value=value),
        True,
        main.quantifier,
    )


def proposition_of(formula: Formula, referents: Mapping[str, str] | None = None) -> Proposition:
    """The proposition of a formula: the whole logical form of a class-level sentence, and the
    proposition of the main clause of a sentence about instances, which is the last part of the
    conjunction (the descriptions in braces take no part). ``referents`` gives the instance of
    every referent label, for the ``local`` labels."""
    names = referents or {}
    main = formula[-1]
    if isinstance(main, Quantified) and isinstance(main.scope, Temporal):
        return _causal_proposition(main)
    if isinstance(main, Quantified):
        scope = main.scope
        polarity = not isinstance(scope, Not)
        scope = scope if polarity else scope.body
        atom = scope.body if isinstance(scope, Able) else scope
        subject_variable = atom.arguments[0]
        kind = _kind(atom.label)
        patient = None
        if kind == VERB:
            variable = atom.arguments[1]
            patient = _read_term(
                variable, tuple(p for p in main.restrictor if _mentions(p, variable))
            )
        subject = _read_term(
            subject_variable, tuple(p for p in main.restrictor if _mentions(p, subject_variable))
        )
        return Proposition(
            CLASS, subject, Predicate(kind, atom.label, patient), polarity, main.quantifier
        )
    polarity = not isinstance(main, Not)
    body = main if polarity else main.body
    if isinstance(body, Timed):
        level = _LEVEL_OF_OPERATOR[body.operator]
        inner = body.body
        if isinstance(inner, Not):
            polarity, inner = False, inner.body
        kind = _kind(inner.label)
        subject = names.get(inner.arguments[0], inner.arguments[0])
        patient = None
        if kind == VERB:
            patient = names.get(inner.arguments[1], inner.arguments[1])
        return Proposition(
            level,
            subject,
            Predicate(kind, inner.label, patient),
            polarity,
            scene=body.scene,
            time=body.time,
            tense=body.tense,
        )
    atom = body.body if isinstance(body, Able | Report) else body
    kind = _kind(atom.label)
    if kind == SCALAR and _is_category(atom.arguments[0]):
        # a class-level scalar pole: the category's mean against its comparison class
        return Proposition(
            CLASS, CategoryTerm(atom.arguments[0]), Predicate(SCALAR, atom.label), polarity, None
        )
    subject = names.get(atom.arguments[0], atom.arguments[0])
    patient = comparison = None
    if kind == VERB:
        patient = names.get(atom.arguments[1], atom.arguments[1])
    elif kind == SCALAR:
        comparison = atom.arguments[1]
    predicate = Predicate(kind, atom.label, patient, comparison)
    if isinstance(body, Report):
        return Proposition(
            EVENT,
            subject,
            predicate,
            scene=scene_of(body.event),
            event=event_of(body.event),
            tense=body.tense,
            aspect=body.aspect,
        )
    return Proposition(INSTANCE, subject, predicate, polarity)
