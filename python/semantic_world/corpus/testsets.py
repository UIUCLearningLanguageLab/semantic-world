"""False propositions for the test sets.

A false proposition is made from a true one by one minimal change:

- **predicate:** another predicate of the same kind that makes the proposition false;
- **subject:** another category or instance of the same level for which it is false;
- **quantifier:** another quantifier that makes it false ("all" for a "most" fact);
- **role** (verbs only): agent and patient exchanged, when the reversed relation does not hold.

Every false item is checked false by the same truth tests that ground the true propositions. A
false item is never vacuous: its subject set has at least one instance. Stage 6 builds the test
sets from these items.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from semantic_world.corpus.facts import Facts
from semantic_world.corpus.lexicon import THING
from semantic_world.corpus.propositions import (
    ALL,
    CLASS,
    FEATURE_KINDS,
    GENERIC,
    MOST,
    NO,
    PROJECTION,
    SCALAR,
    SOME,
    VERB,
    CategoryTerm,
    Proposition,
)

PREDICATE = "predicate"
SUBJECT = "subject"
QUANTIFIER = "quantifier"
ROLE = "role"
CHANGES = (PREDICATE, SUBJECT, QUANTIFIER, ROLE)


def candidates(facts: Facts, proposition: Proposition, change: str) -> list[Proposition]:
    """Every proposition that differs from ``proposition`` by one change of the given kind,
    true or false, in a fixed order."""
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
        return [
            dataclasses.replace(proposition, subject=other)
            for other in truth.result.instances.labels
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

    if change == ROLE:
        if predicate.kind != VERB:
            return []
        swapped = dataclasses.replace(predicate, patient=subject)
        return [dataclasses.replace(proposition, subject=predicate.patient, predicate=swapped)]

    raise ValueError(f"unknown change {change!r}; the changes are {', '.join(CHANGES)}")


def falsify(
    facts: Facts, proposition: Proposition, change: str, rng: np.random.Generator
) -> Proposition | None:
    """A false proposition made from a true one by one minimal change, drawn at random from the
    changes that give a false, expressible, non-vacuous proposition. None when there is none.
    The false item carries the grounding that shows it false."""
    options = candidates(facts, proposition, change)
    for index in rng.permutation(len(options)):
        candidate = dataclasses.replace(options[int(index)], grounding=None, id=None, rule=None)
        if not facts.expressible(candidate):
            continue
        evaluation = facts.truth.evaluate(candidate)
        if evaluation.valid and not evaluation.true:
            return dataclasses.replace(candidate, grounding=evaluation.grounding)
    return None


def changes_for(proposition: Proposition) -> tuple[str, ...]:
    """The kinds of change that can apply to a proposition at all."""
    changes = [PREDICATE, SUBJECT]
    if proposition.level == CLASS:
        changes.append(QUANTIFIER)
    if proposition.predicate.kind == VERB:
        changes.append(ROLE)
    return tuple(changes)
