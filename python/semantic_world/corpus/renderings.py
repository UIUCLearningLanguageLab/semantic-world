"""Renderings of a sentence: the formal rendering of a sequence of lexemes, and the propositional
rendering of a logical form. The realizer makes the formal and the conceptual renderings of a
sentence from its tokens (``realize.py``), and the spelled rendering comes with the word forms.

**The formal rendering** writes every word as its gloss and its lexeme label: ``the/L.176
C1.3/L.5 CAN.2/L.138``. The gloss of a content lexeme is its concept's label, and the gloss of a
function word is its English gloss.

**The propositional rendering** is the logical form of a sentence, written as a formula. It is
never ambiguous: capacity and event, tense, aspect, and number are explicit in it, whatever the
grammar settings. It is made from the JSON logical form alone (:func:`propositional`), and it
parses back (:func:`parse_propositional`, :func:`proposition_of`). The notation:

- an atom is a concept label with its arguments: ``C1.3.2(R.1)`` (membership), ``IS.12(R.1)``,
  ``HAS.4(R.1)``, ``CANBE.V1.1(R.1)``, and ``SC.1.HIGH(R.1, C1.3.2)``, a scalar pole with its
  comparison class. ``NOT`` before an atom negates it;
- a capacity is wrapped in ``ABLE``: ``ABLE(CAN.3(R.1))`` and ``ABLE(V1.2(R.1, R.2))``, with the
  agent first. ``ABLE`` wraps only CAN features and verbs. A negative capacity is ``NOT
  ABLE(...)``;
- an event is ``EVENT(<event label>, <tense>, <aspect>, <atom>)``: ``EVENT(SN.8.5, PAST,
  PROGRESSIVE, V1.2(R.1, R.2))``. The tense and the aspect are always written;
- ``R.n`` is one individual, a referent of the document, and ``X.n`` is a variable bound by a
  quantifier;
- a class-level form is a quantifier with a restrictor and a scope: ``MOST(C1.3(X.1) AND
  IS.4(X.1), ABLE(CAN.3(X.1)))``, with ``ALL``, ``MOST``, ``SOME``, ``NO``, and ``GEN``. A verb's
  patient category is in the restrictor with a variable of its own, so the quantifier ranges
  over pairs: ``GEN(C1.2(X.1) AND C1.5(X.2), ABLE(V2.1(X.1, X.2)))``;
- a relative clause about another category means at least one member of it, written ``EXISTS``
  inside the restrictor: ``GEN(C1.2(X.1) AND EXISTS(X.2, C1.5(X.2) AND ABLE(V2.1(X.1, X.2))),
  SC.1.HIGH(X.1, C1))`` for "owls that eat mice are big".

A sentence about instances is a conjunction. Each noun phrase gives its noun and its modifiers,
then the propositions of its relative clause, and the proposition of the main clause comes last.
Variables are numbered in the order of the logical form: the subject, its relative clauses, then
the patient. Word order never changes the rendering.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.lexicon import THING, Lexeme
from semantic_world.corpus.propositions import (
    ALL,
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
    VERB,
    CategoryTerm,
    Clause,
    Literal,
    Predicate,
    Proposition,
)

REFERENT_MODES = ("local", "instance")
QUANTIFIER_NAMES = {ALL: "ALL", MOST: "MOST", SOME: "SOME", NO: "NO", GENERIC: "GEN"}
_QUANTIFIERS = {name: quantifier for quantifier, name in QUANTIFIER_NAMES.items()}


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
    """An event: its label, its tense, its aspect, and what happened."""

    event: str
    tense: str
    aspect: str
    body: Atom

    def __str__(self) -> str:
        return f"EVENT({self.event}, {self.tense.upper()}, {self.aspect.upper()}, {self.body})"


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
        name = QUANTIFIER_NAMES[self.quantifier]
        return f"{name}({_conjunction(self.restrictor)}, {self.scope})"


Formula = tuple[Any, ...]
"""A conjunction of formulas. A class-level form is one :class:`Quantified`."""


def _conjunction(parts: tuple[Any, ...]) -> str:
    return " AND ".join(str(part) for part in parts)


def write(formula: Formula) -> str:
    """A formula as text."""
    return _conjunction(formula)


# ---------------------------------------------------------------------------------------------
# From the JSON logical form
# ---------------------------------------------------------------------------------------------

_LABEL_KEY = {
    IS: "feature",
    HAS: "feature",
    CAN: "feature",
    SCALAR: "pole",
    MEMBER: "category",
    PROJECTION: "projection",
    VERB: "verb",
}


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
        return f"X.{self.count}"


def _term(term: dict[str, Any], variable: str, variables: _Variables) -> list[Any]:
    """The restrictor of a category term: its category, its literals, and its relative
    clauses."""
    category = term["category"]
    parts: list[Any] = [] if category == THING else [Atom(category, (variable,))]
    parts += [_literal(text, variable, category) for text in term.get("restriction", ())]
    for clause in term.get("clauses", ()):
        label = clause[_LABEL_KEY[clause["kind"]]]
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
) -> Any:
    """The proposition of one predicate: an atom, a capacity, or an event."""
    kind = predicate["kind"]
    label = predicate[_LABEL_KEY[kind]]
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
        if report[0] is None:
            raise RenderingError("a report that names no event has no rendering yet")
        body = Report(report[0], report[1], report[2], atom)
    elif kind in (CAN, VERB):
        body = Able(atom)
    return body if polarity else Not(body)


def formula(form: dict[str, Any], referents: str = "local") -> Formula:
    """The formula of a JSON logical form. ``referents`` is ``local``, for the referent labels
    of the document (``R.1``), or ``instance``, for the taxonomy's instance labels."""
    if referents not in REFERENT_MODES:
        raise ValueError(f"unknown referent labels {referents!r}")
    predicate = form["predicate"]
    if form["level"] == CLASS:
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

    def mention_parts(mention: dict[str, Any]) -> list[Any]:
        """A noun phrase: its noun, its modifiers, and the propositions of its relative
        clause, each after the noun phrases it brings in."""
        argument, noun = name(mention), mention.get("noun")
        parts: list[Any] = [] if noun is None else [Atom(noun, (argument,))]
        parts += [_literal(text, argument, noun) for text in mention.get("restriction", ())]
        for clause in mention.get("clauses", ()):
            report = None
            if clause.get("event") is not None:
                report = (clause["event"], clause["tense"], clause["aspect"])
            polarity = clause.get("polarity", True)
            if "agent" in clause:
                parts += mention_parts(clause["agent"])
                parts.append(
                    _predication(clause, name(clause["agent"]), argument, polarity, report)
                )
                continue
            other = None
            if "patient" in clause:
                parts += mention_parts(clause["patient"])
                other = name(clause["patient"])
            parts.append(_predication(clause, argument, other, polarity, report))
        return parts

    parts = mention_parts(form["subject"])
    patient = None
    if "patient" in predicate:
        parts += mention_parts(predicate["patient"])
        patient = name(predicate["patient"])
    report = None
    if form["level"] == EVENT:
        report = (form["event"], form["tense"], form["aspect"])
    main = _predication(predicate, name(form["subject"]), patient, form["polarity"], report)
    # a noun phrase said twice gives the same proposition twice: it is written once
    return tuple(dict.fromkeys(part for part in parts if part != main)) + (main,)


def propositional(form: dict[str, Any], referents: str = "local") -> str:
    """The propositional rendering of a JSON logical form."""
    return write(formula(form, referents))


# ---------------------------------------------------------------------------------------------
# Reading a rendering back
# ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*([(),]|[^\s(),]+)")


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
    """The formula that a propositional rendering writes."""
    parser = _Parser(text)
    parsed = parser.conjunction()
    if parser.peek() is not None:
        raise RenderingError(f"cannot read the rendering {text!r}: {parser.peek()!r} is left over")
    return parsed


def _kind(label: str) -> str:
    """The predicate kind that a concept label belongs to."""
    for prefix, kind in (
        ("IS.", IS),
        ("HAS.", HAS),
        ("CANBE.", PROJECTION),
        ("CAN.", CAN),
        ("SC.", SCALAR),
    ):
        if label.startswith(prefix):
            return kind
    if label == THING or label.startswith("C"):
        return MEMBER
    if label.startswith("V"):
        return VERB
    raise RenderingError(f"unknown kind of concept {label!r}")


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


def proposition_of(formula: Formula, referents: Mapping[str, str] | None = None) -> Proposition:
    """The proposition of a formula: the whole logical form of a class-level sentence, and the
    proposition of the main clause of a sentence about instances, which is the last part of the
    conjunction. ``referents`` gives the instance of every referent label, for the ``local``
    labels."""
    names = referents or {}
    main = formula[-1]
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
    atom = body.body if isinstance(body, Able | Report) else body
    kind = _kind(atom.label)
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
            scene=body.event.rsplit(".", 1)[0],
            event=body.event,
            tense=body.tense,
            aspect=body.aspect,
        )
    return Proposition(INSTANCE, subject, predicate, polarity)
