"""Mentions: how a proposition's referents become noun phrases.

The planner decides everything that changes what a sentence says. Stage 4 builds two parts of
that work, which the grammar needs: the sentence plan of a proposition, and relative clauses.
The other mention decisions (the noun's level, first and later mentions, pronouns, and
distinguishing modifiers) come with the documents.

**The plan of a proposition.** A class-level proposition names its categories with their own
nouns, and its quantifier is the subject's determiner. An instance is mentioned by default with
the noun of its leaf and the determiner ``the``, and a caller can give any other mention. The
subject of an instance-level scalar pole is named by its comparison class, because "the mouse is
big" means big for a mouse. The subject of "is a penguin" is named by the category above the
leaf ("the bird is a penguin"), or by a pronoun when the leaf has none above it.

**Relative clauses.** A noun phrase takes a relative clause at ``mention.relative_clauses.rate``.
The clause is an object relative with probability ``object_share``, and a subject relative
otherwise. It expresses a true proposition of the same level as its sentence, about the same
referent:

- in an event-level sentence, another event of the same scene that the referent takes part in:
  as its agent (a subject relative, "the dog that chased the cat") or as its patient (an object
  relative, "the cat that the dog chased");
- in an instance-level sentence, a capacity of the referent: a CAN feature it has, or a verb's
  relation with another instance of the document, as agent ("the owl that can eat the mouse") or
  as patient ("the mouse that the owl can eat").

When the drawn kind has no true proposition, the other kind is used, and when neither has one
the noun phrase takes no relative clause. The noun phrases inside a relative clause can take
relative clauses of their own, up to ``max_depth``. Class-level noun phrases take no drawn
relative clause yet: what such a clause means for the truth of its sentence is not settled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_NP,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
)
from semantic_world.corpus.propositions import (
    CAN,
    CLASS,
    EVENT,
    GENERIC,
    MEMBER,
    SCALAR,
    VERB,
    CategoryTerm,
    Predicate,
    Proposition,
)


def leaf_of(facts: Facts, instance: str) -> str:
    """The leaf category of an instance."""
    return facts.truth.paths[facts.truth.instance_index[instance]][-1]


def mention(
    facts: Facts, instance: str, noun: str | None = None, determiner: str = "the"
) -> NounPhrase:
    """A plain mention of an instance: the noun of its leaf, or of a given category, and a
    determiner."""
    return NounPhrase(INSTANCE_NP, instance, noun or leaf_of(facts, instance), determiner)


def class_phrase(term: CategoryTerm, determiner: str | None = None) -> NounPhrase:
    return NounPhrase(CLASS_NP, term.category, term.category, determiner, term.restriction)


def plan_for(
    facts: Facts, proposition: Proposition, mentions: Mapping[str, NounPhrase] | None = None
) -> SentencePlan:
    """The sentence plan of a proposition. ``mentions`` gives the noun phrase of an instance;
    an instance without one is mentioned by the noun of its leaf, with ``the``."""
    mentions = mentions or {}
    predicate = proposition.predicate

    def phrase(instance: str, noun: str | None = None) -> NounPhrase:
        return mentions.get(instance) or mention(facts, instance, noun)

    if proposition.level == CLASS:
        assert isinstance(proposition.subject, CategoryTerm)
        quantifier = None if proposition.quantifier == GENERIC else proposition.quantifier
        subject = class_phrase(proposition.subject, quantifier)
        target = None
        if isinstance(predicate.patient, CategoryTerm):
            target = class_phrase(predicate.patient)
        return SentencePlan(
            subject, Predication(predicate.kind, predicate.label, proposition.polarity, target)
        )
    assert isinstance(proposition.subject, str)
    noun = predicate.comparison
    if predicate.kind == MEMBER and predicate.label == leaf_of(facts, proposition.subject):
        # "the penguin is a penguin" says nothing: name the instance by the category above
        path = facts.truth.paths[facts.truth.instance_index[proposition.subject]]
        noun = path[-2] if len(path) > 1 else None
        if noun is None and proposition.subject not in mentions:
            mentions = {
                **mentions,
                proposition.subject: NounPhrase(INSTANCE_NP, proposition.subject),
            }
    subject = phrase(proposition.subject, noun)
    target = None if predicate.patient is None else phrase(str(predicate.patient))
    event = proposition.event if proposition.level == EVENT else None
    return SentencePlan(
        subject, Predication(predicate.kind, predicate.label, proposition.polarity, target, event)
    )


class RelativeClauses:
    """Drawing relative clauses for the noun phrases of a sentence plan."""

    def __init__(self, config: Config, facts: Facts) -> None:
        self.settings = config.mention.relative_clauses
        self.facts = facts

    def attach(
        self, rng: np.random.Generator, plan: SentencePlan, others: Sequence[str] = ()
    ) -> SentencePlan:
        """The plan with relative clauses drawn for its noun phrases. ``others`` are the
        instances that an instance-level relative clause can relate the referent to: the other
        referents of the document. An event-level clause uses the events of the sentence's own
        scene."""
        if plan.level == CLASS or self.settings.max_depth == 0 or self.settings.rate == 0:
            return plan
        used = {self._key(plan.predication, plan.subject.referent)}
        subject = self._phrase(rng, plan.subject, plan, others, 1, used)
        predication = plan.predication
        if predication.object is not None:
            target = self._phrase(rng, predication.object, plan, others, 1, used)
            predication = dataclasses.replace(predication, object=target)
        return SentencePlan(subject, predication)

    @staticmethod
    def _key(predication: Predication, subject: str, patient: str | None = None) -> tuple:
        """What a predication says, to keep a sentence from saying the same thing twice."""
        if predication.event is not None:
            return ("event", predication.event)
        target = patient if predication.object is None else predication.object.referent
        return (predication.label, subject, target)

    def _phrase(
        self,
        rng: np.random.Generator,
        phrase: NounPhrase,
        plan: SentencePlan,
        others: Sequence[str],
        depth: int,
        used: set,
    ) -> NounPhrase:
        """A noun phrase with a relative clause drawn for it, at the rate. A pronoun, a noun
        phrase that already has a clause, and one past the depth limit are left as they are."""
        if (
            depth > self.settings.max_depth
            or phrase.kind != INSTANCE_NP
            or phrase.pronoun
            or phrase.clause is not None
            or rng.random() >= self.settings.rate
        ):
            return phrase
        referent = phrase.referent
        as_object = rng.random() < self.settings.object_share
        if phrase.negated:
            as_object = False  # the negated IS literals need a subject relative to join
        for object_relative in (as_object, not as_object):
            if object_relative and phrase.negated:
                continue
            options = self._options(referent, plan, others, object_relative, used)
            if not options:
                continue
            predication, other = options[int(rng.integers(len(options)))]
            used.add(self._key(predication, other or referent, referent if other else None))
            if object_relative:
                assert other is not None
                agent = self._phrase(rng, mention(self.facts, other), plan, others, depth + 1, used)
                return dataclasses.replace(phrase, clause=RelativeClause((predication,), agent))
            if predication.object is not None:
                target = self._phrase(rng, predication.object, plan, others, depth + 1, used)
                predication = dataclasses.replace(predication, object=target)
            return dataclasses.replace(phrase, clause=RelativeClause((predication,)))
        return phrase

    def _options(
        self,
        referent: str,
        plan: SentencePlan,
        others: Sequence[str],
        object_relative: bool,
        used: set,
    ) -> list[tuple[Predication, str | None]]:
        """The true propositions that a relative clause about a referent can express, as a
        predication and, for an object relative, the clause's own subject."""
        facts = self.facts
        options: list[tuple[Predication, str | None]] = []
        if plan.level == EVENT:
            assert plan.predication.event is not None
            scene = facts.truth.scenes[plan.predication.event.rsplit(".", 1)[0]]
            for event in scene.events:
                if ("event", event.label) in used:
                    continue
                names = facts.event_names(event)
                if not names:
                    continue
                kind = VERB if event.transitive else CAN
                if object_relative and event.patient == referent:
                    options.append((Predication(kind, names[0], event=event.label), event.agent))
                elif not object_relative and event.agent == referent:
                    target = None if event.patient is None else mention(facts, event.patient)
                    options.append((Predication(kind, names[0], True, target, event.label), None))
            return options
        truth = facts.truth
        if object_relative:
            for verb in facts.verbs:
                for other in others:
                    if other != referent and (verb, other, referent) not in used:
                        if truth.allows(verb, other, referent):
                            options.append((Predication(VERB, verb), other))
            return options
        for feature in facts.features[CAN]:
            if (feature, referent, None) not in used and truth.allows(feature, referent, None):
                options.append((Predication(CAN, feature), None))
        for verb in facts.verbs:
            for other in others:
                if other != referent and (verb, referent, other) not in used:
                    if truth.allows(verb, referent, other):
                        target = mention(facts, other)
                        options.append((Predication(VERB, verb, True, target), None))
        return options


def clause_propositions(plan: SentencePlan) -> list[Proposition]:
    """The proposition that every relative clause of a plan expresses about its head, for the
    truth tests: each is of the same level as the sentence."""
    found: list[Proposition] = []
    level = plan.level

    def proposition(predication: Predication, subject: str, patient: str | None) -> Proposition:
        predicate = Predicate(predication.kind, predication.label, patient)
        if predication.event is not None:
            scene = predication.event.rsplit(".", 1)[0]
            return Proposition(EVENT, subject, predicate, scene=scene, event=predication.event)
        return Proposition(level, subject, predicate, predication.polarity)

    def visit(phrase: NounPhrase | None) -> None:
        if phrase is None or phrase.clause is None:
            return
        clause = phrase.clause
        if clause.agent is not None:
            found.append(
                proposition(clause.predications[0], clause.agent.referent, phrase.referent)
            )
            visit(clause.agent)
            return
        for predication in clause.predications:
            target = predication.object
            if predication.kind == SCALAR:
                continue
            found.append(
                proposition(
                    predication, phrase.referent, None if target is None else target.referent
                )
            )
            visit(target)

    visit(plan.subject)
    visit(plan.predication.object)
    return found
