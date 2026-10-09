"""The JSON logical form of a sentence.

A sentence's logical form is the proposition of its main clause, as the truth tests judge it,
together with everything its noun phrases say. The form is what ``documents.jsonl`` holds, and
the propositional rendering is made from it alone (``renderings.propositional``).

**Class level.** The form is the proposition's own: the subject and the patient are category
terms, each with its restriction and, when it has them, its relative clauses (``clauses``). A
clause is written like a predicate: ``{"kind": "event_type1", "label": "EVENTTYPE1.3"}``, or a
two-place event type with the other category, as ``patient`` when the head is the agent ("owls
that eat mice") and as ``agent`` when the head is the patient ("mice that owls eat"). The
predicate of a scalar pole also holds its comparison class (``class``): the subject category's
parent, or ``THING`` for a top-level category.

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
A determiner is no part of the logical form.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from semantic_world.corpus.grammar import NounPhrase, Predication, SentencePlan
from semantic_world.corpus.propositions import _JSON_KEY, CLASS, SCALAR, Proposition


def logical_form(
    plan: SentencePlan, proposition: Proposition, referents: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The JSON logical form of a sentence: its main proposition, with its grounding, and the
    mentions of its plan. ``referents`` gives the referent label of every instance mentioned in
    the document."""
    form = proposition.to_json()
    if proposition.level == CLASS:
        if proposition.predicate.kind == SCALAR and proposition.grounding is not None:
            form["predicate"]["class"] = proposition.grounding["comparison"]
        return form
    labels = referents or {}

    def mention(phrase: NounPhrase) -> dict[str, Any]:
        data: dict[str, Any] = {
            "instance": phrase.referent,
            "referent": labels.get(phrase.referent),
            "noun": phrase.noun,
            "restriction": [str(literal) for literal in phrase.restriction],
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
    return form
