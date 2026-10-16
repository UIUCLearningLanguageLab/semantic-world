"""The JSON logical form of a sentence.

A sentence's logical form is the proposition of its main clause, as the truth tests judge it,
together with everything its noun phrases say. The form is what ``documents.jsonl`` holds, and
the propositional rendering is made from it alone (``renderings.propositional``).

**Class level.** The form is the proposition's own: the subject and the patient are category
terms, each with its restriction and, when it has them, its relative clauses (``clauses``). A
clause is written like a predicate: ``{"kind": "event_type1", "label": "EVENTTYPE1.3"}``, or a
two-place event type with the other category, as ``patient`` when the head is the agent ("owls
that eat mice") and as ``agent`` when the head is the patient ("mice that owls eat"). The
predicate of a scalar pole (kind ``comparison``) holds its operator (``ABOVE`` or ``BELOW``) and
its comparison class (``class``): an ancestor of the subject category, or ``ENTITY`` for a
top-level category (stage b3a). A comparative (kind ``comparative``, operator ``GREATER``) holds
the other category as its ``patient``.

**Comparatives.** The subject and the patient of a comparative are the sentence's subject and
object, in the sentence's order, and the predicate records the pole adjective the sentence uses
under ``word`` (``SCALARDIM.1.HIGH`` for "bigger than", ``SCALARDIM.1.LOW`` for "smaller
than"), as it records whether a mention is descriptive: the word is no part of the proposition,
which is ``GREATER`` with the greater argument first, so a form whose word is the low pole's
names the greater argument as its patient. ``Proposition.from_json`` and the propositional
rendering read the word and put the arguments in the proposition's order.

**The threshold of the poles.** A logical form that holds a pole adjective anywhere, in its
predicate, in the restriction of a category term, or among the modifiers of a mention, records
the language's threshold under ``z`` (``scalar_adjectives.z``, in standard deviations of the
comparison class), so that every proposition with a pole records the z it used (CG.65). A form
without a pole has no ``z``.

**Instance, event, state, change, and able_now levels.** The subject, and the patient of a
verb, are mentions:

```json
{"instance": "INSTANCE.1.3.2.5", "referent": "REF.1", "noun": "CATEGORY.1.3.2",
 "restriction": ["PROPERTY.12"],
 "clauses": [{"kind": "event_type2", "label": "EVENTTYPE2.1.2", "patient": {...},
              "event": "SCENE.3.EVENTINSTANCE.2", "tense": "past", "aspect": "simple"}]}
```

``referent`` is the referent's label in its document. ``noun`` is the category that the noun
names, or null for a pronoun. ``restriction`` holds the modifiers. ``clauses`` holds the
propositions of the relative clause, each with its other mention as ``patient`` or ``agent``,
and, at the event level, with the event it reports. ``clauses`` is left out when there is none.
A determiner is no part of the logical form, but ``descriptive`` says what it does (CG.63): in
a definite mention ("the furry dog") the noun, the modifiers, and the relative clause are
descriptions, which identify the referent, and ``descriptive`` is true; in an indefinite mention
("a penguin") they are asserted, the sentence introduces the referent, and it is false; a
pronoun has no description, and it is false too.

**Causal statements.** The form is the proposition's own: the subject is the event term
(``{"head": "THING", "event": "EVENTTYPE2.1.2", "role": "patient"}``, with the ``condition`` of
a conditional statement, stage b3b), the predicate an effect or a precondition with its value
(or, for a numeric effect statement, its ``direction``), and ``causal`` the definition entry
that the statement states.

**Numeric fluents** (stage b3b). A numeric pole at a time point (kind ``comparison``, class
``RANGE``) records the language's cut for its adjective under ``cut`` in the predicate
(``numeric_adjectives.cut_high`` or ``cut_low``), as every proposition with a numeric pole
records the cut it used; a numeric comparative at a time point records the adjective the
sentence uses under ``word``, as a comparative of a scalar does; a numeric change (kind
``state``, a numeric fluent as its label) carries its ``direction``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from semantic_world.corpus.grammar import NounPhrase, Predication, SentencePlan, term_of
from semantic_world.corpus.propositions import (
    _JSON_KEY,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    HIGH,
    LOW,
    Proposition,
    is_numeric,
)
from semantic_world.corpus.world import SCALAR_PREFIX


def is_descriptive(phrase: NounPhrase) -> bool:
    """Whether a mention's noun, modifiers, and relative clause are descriptions (CG.63): a
    definite mention's are; an indefinite mention's are asserted, and a pronoun has none."""
    return phrase.determiner == "the"


def has_pole(data: Any) -> bool:
    """Whether a JSON logical form (or a part of one) holds a pole adjective of a scalar: a
    predicate of kind ``comparison`` whose label is a scalar pole, or a pole literal in a
    restriction. A numeric pole (``NUMFL.1.HIGH``, stage b3b) is judged at its own cut, which
    its predicate records, and is no scalar pole."""
    if isinstance(data, dict):
        if data.get("kind") == COMPARISON and not is_numeric(str(data.get("label", ""))):
            return True
        return any(has_pole(value) for key, value in data.items() if key != "grounding")
    if isinstance(data, list):
        return any(
            (isinstance(item, str) and item.removeprefix("not ").startswith(SCALAR_PREFIX))
            or has_pole(item)
            for item in data
        )
    return False


def with_z(form: dict[str, Any], z: float | None) -> dict[str, Any]:
    """The form with the language's threshold recorded under ``z``, after the predicate, when
    the form holds a pole and ``z`` is given; otherwise the form as it is."""
    if z is None or not has_pole(form):
        return form
    ordered: dict[str, Any] = {}
    for key, value in form.items():
        ordered[key] = value
        if key == "predicate":
            ordered["z"] = float(z)
    if "z" not in ordered:
        ordered["z"] = float(z)
    return ordered


def _literals_with_cuts(
    literals: list[dict[str, Any]], cuts: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Condition literals with the language's cut recorded under ``cut``, after the value, on
    each literal that is a numeric pole (``NUMFL.1.HIGH``; WM.35)."""
    found = []
    for literal in literals:
        label = str(literal.get("label", ""))
        if is_numeric(label) and "cut" not in literal:
            literal = {**literal, "cut": float(cuts[label.rsplit(".", 1)[1]])}
        found.append(literal)
    return found


def with_cut(form: dict[str, Any], cuts: Mapping[str, float] | None) -> dict[str, Any]:
    """The form with the language's cut for its numeric pole recorded under ``cut`` in the
    predicate, after the comparison class, when the predicate is a numeric pole and the cuts
    are given (``{"HIGH": 0.75, "LOW": 0.25}``); otherwise the form as it is. A conditional
    causal statement's numeric condition literals record their cut too, in the subject and
    in the ``causal`` record (WM.35)."""
    if cuts is not None:
        for key in ("subject", "causal"):
            block = form.get(key)
            if isinstance(block, dict) and block.get("condition"):
                block["condition"] = _literals_with_cuts(block["condition"], cuts)
    predicate = form.get("predicate", {})
    label = str(predicate.get("label", ""))
    if cuts is None or predicate.get("kind") != COMPARISON or not is_numeric(label):
        return form
    side = label.rsplit(".", 1)[1]
    ordered: dict[str, Any] = {}
    for key, value in predicate.items():
        ordered[key] = value
        if key == "class":
            ordered["cut"] = float(cuts[side])
    if "cut" not in ordered:
        ordered["cut"] = float(cuts[side])
    form["predicate"] = ordered
    return form


def logical_form(
    plan: SentencePlan,
    proposition: Proposition,
    referents: Mapping[str, str] | None = None,
    z: float | None = None,
    cuts: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The JSON logical form of a sentence: its main proposition, with its grounding, and the
    mentions of its plan. ``referents`` gives the referent label of every instance mentioned in
    the document. ``z`` is the language's threshold for its pole adjectives, recorded when the
    form holds one, and ``cuts`` the language's cuts for its numeric adjectives, by pole
    (``HIGH``, ``LOW``), recorded when the predicate is a numeric pole."""
    form = with_cut(proposition.to_json(), cuts)
    if proposition.level == CLASS:
        if plan.predication.kind == COMPARATIVE:
            # the sentence's subject and object, and the adjective it uses
            assert isinstance(plan.subject.referent, str) and plan.predication.object is not None
            form["subject"] = term_of(plan.subject).to_json()
            form["predicate"]["patient"] = term_of(plan.predication.object).to_json()
            form["predicate"]["word"] = plan.predication.label
        return with_z(form, z)
    labels = referents or {}

    def mention(phrase: NounPhrase) -> dict[str, Any]:
        data: dict[str, Any] = {
            "instance": phrase.referent,
            "referent": labels.get(phrase.referent),
            "noun": phrase.noun,
            "restriction": [str(literal) for literal in phrase.restriction],
            "descriptive": is_descriptive(phrase),
        }
        clause = phrase.clause
        if clause is not None:
            data["clauses"] = [predication(p, clause.agent) for p in clause.predications]
        return data

    def predication(part: Predication, agent: NounPhrase | None) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": part.kind, _JSON_KEY[part.kind]: part.label}
        if agent is not None:
            data["agent"] = mention(agent)
        elif part.object is not None:
            data["patient"] = mention(part.object)
        if not part.polarity:
            data["polarity"] = False
        if part.event is not None:
            data.update(event=part.event, tense=part.tense, aspect=part.aspect)
        return data

    form["subject"] = mention(plan.subject)
    if plan.predication.object is not None:
        form["predicate"]["patient"] = mention(plan.predication.object)
    if plan.predication.kind == COMPARATIVE:
        form["predicate"]["word"] = plan.predication.label
    return with_z(form, z)


CUT_SIDES = (HIGH, LOW)
