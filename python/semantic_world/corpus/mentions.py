"""Mentions: how a proposition's referents become noun phrases.

The planner decides everything that changes what a sentence says. This module holds the mention
decisions: the sentence plan of a proposition, relative clauses, and, for the instances of a
document, the noun's level, first and later mentions, pronouns, and modifiers.

**The plan of a proposition.** A class-level proposition names its categories with their own
nouns, and its subject's determiner is the word that states its quantifier ("all", "most",
"some", "no"), or none for a bare plural; the plan keeps the quantifier itself. An instance is
mentioned by default with the noun of its leaf and the determiner ``the``, and a caller can give
any other mention. The subject of an instance-level scalar pole is named by its comparison
class, because "the mouse is big" means big for a mouse. The subject of "is a penguin" is named
by the category above the leaf ("the bird is a penguin"), or by a pronoun when the leaf has none
above it.

**Relative clauses on instances.** A noun phrase takes a relative clause at
``mention.relative_clauses.rate``. The clause is an object relative with probability
``object_share``, and a subject relative otherwise. It expresses a true proposition of the same
level as its sentence, about the same referent:

- in an event-level sentence, another event of the same scene that the referent takes part in:
  as its agent (a subject relative, "the dog that chased the cat") or as its patient (an object
  relative, "the cat that the dog chased"), with the aspect the document gives the report;
- in an instance-level sentence, a capacity of the referent: a one-place event type it is able
  to be the agent of, or a two-place event type with another instance of the document, as agent
  ("the owl that can eat the mouse") or as patient ("the mouse that the owl can eat").

When the drawn kind has no true proposition, the other kind is used, and when neither has one
the noun phrase takes no relative clause. The noun phrases inside a relative clause can take
relative clauses of their own, up to ``max_depth``. A class-level relative clause restricts its
category, so it changes the truth of the sentence: it is drawn with the proposition
(``facts.draw_clause``), and not here.

**Mentions in a document** (:class:`Mentions`). Each mention of an instance draws the level of
its noun by ``mention.level_weights``, among the instance's leaf and the categories above it. A
first mention takes ``a``, and a later mention takes ``the``, or becomes ``it`` at
``mention.pronoun_rate`` when the referent was mentioned in the sentence before and was the only
referent there, or its subject. When the noun of a definite mention fits other participants of
the document's scenes, the mention adds adjectives and with-phrases until it picks out its
referent, by the incremental algorithm of Dale and Reiter (1995): the attributes are tried in
one preference order, fixed for the language, and an attribute is added when it rules out a
participant that is still left. When the modifiers cannot tell the referent apart, the noun of
the leaf is tried, and then the mention is kept as it is. Other modifiers are added at
``mention.modifier_rate``, in the documents that ask for them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.grammar import (
    INSTANCE_NP,
    WORD_OF,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
    phrase_of,
)
from semantic_world.corpus.histories import SceneEvent, event_of, scene_of, time_key
from semantic_world.corpus.propositions import (
    CAN,
    CHANGE,
    CLASS,
    EVENT,
    HAS,
    IS,
    MEMBER,
    SCALAR,
    SIMPLE,
    VERB,
    CategoryTerm,
    Literal,
    Predicate,
    Proposition,
)
from semantic_world.corpus.streams import Streams
from semantic_world.corpus.world import PART_PREFIX, PROPERTY_PREFIX, SCALAR_PREFIX

REFERENT_PREFIX = "REF."


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
    """The noun phrase of a category term, with its relative clauses."""
    return phrase_of(term, determiner)


def plan_for(
    facts: Facts,
    proposition: Proposition,
    mentions: Mapping[str, NounPhrase] | None = None,
    bare: bool = False,
) -> SentencePlan:
    """The sentence plan of a proposition. ``mentions`` gives the noun phrase of an instance;
    an instance without one is mentioned by the noun of its leaf, with ``the``. ``bare`` says a
    class-level subject with a bare plural in place of its quantifier word."""
    mentions = mentions or {}
    predicate = proposition.predicate

    def phrase(instance: str, noun: str | None = None) -> NounPhrase:
        return mentions.get(instance) or mention(facts, instance, noun)

    if proposition.level == CLASS:
        assert isinstance(proposition.subject, CategoryTerm)
        quantifier = proposition.quantifier
        determiner = None if bare or quantifier is None else WORD_OF[quantifier]
        subject = class_phrase(proposition.subject, determiner)
        target = None
        if isinstance(predicate.patient, CategoryTerm):
            target = class_phrase(predicate.patient)
        return SentencePlan(
            subject,
            Predication(predicate.kind, predicate.label, proposition.polarity, target),
            quantifier,
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
    if proposition.timed:
        assert proposition.scene is not None and proposition.time is not None
        return SentencePlan(
            subject,
            Predication(
                predicate.kind,
                predicate.label,
                proposition.polarity,
                target,
                tense=proposition.tense,
                time=time_key(proposition.scene, proposition.time),
                become=proposition.level == CHANGE,
            ),
        )
    if proposition.level != EVENT:
        return SentencePlan(
            subject, Predication(predicate.kind, predicate.label, proposition.polarity, target)
        )
    return SentencePlan(
        subject,
        Predication(
            predicate.kind,
            predicate.label,
            True,
            target,
            proposition.event,
            proposition.tense,
            proposition.aspect,
        ),
    )


@dataclass
class ClauseContext:
    """What the relative clauses of one sentence are drawn from."""

    level: str
    others: Sequence[str] = ()
    """Instance level: the instances that a clause can relate the referent to."""
    events: Sequence[SceneEvent] = ()
    """Event level: the events that a clause can report."""
    mention: Callable[[str], NounPhrase] | None = None
    """The noun phrase of an instance that a clause brings in."""
    aspect: Callable[[SceneEvent], str] | None = None
    """Event level: the aspect of a report of an event (``simple`` by default)."""
    used: set = field(default_factory=set)
    """What the sentence already says, so that no clause says it again. An event is held twice:
    by its label, and by what happened (:func:`happened`), because the same event can occur
    again at a later step."""


def happened(event: SceneEvent) -> tuple:
    """What an event reports, apart from when: its event type, its agent, and its patient. A
    clause never reports an event with the same three as its sentence's own event, or as the
    event of another clause of the sentence: "the dog chased the cat that the dog chased"."""
    return ("happened", *event.key)


class RelativeClauses:
    """Drawing relative clauses for the instance noun phrases of a sentence plan."""

    def __init__(self, config: Config, facts: Facts) -> None:
        self.settings = config.mention.relative_clauses
        self.tense = config.propositions.event_tense
        self.facts = facts

    @property
    def enabled(self) -> bool:
        return self.settings.max_depth > 0 and self.settings.rate > 0

    def context(
        self,
        plan: SentencePlan,
        others: Sequence[str] = (),
        mention: Callable[[str], NounPhrase] | None = None,
        events: Sequence[SceneEvent] | None = None,
        aspect: Callable[[SceneEvent], str] | None = None,
    ) -> ClauseContext:
        """The context of a sentence's relative clauses. ``others`` are the instances that an
        instance-level clause can relate the referent to: the other referents of the document.
        ``events`` are the events that an event-level clause can report: by default, every event
        of the sentence's own scene. ``aspect`` gives the aspect of a report."""
        if plan.level == EVENT and events is None:
            assert plan.predication.event is not None
            events = self.facts.truth.events_of(scene_of(plan.predication.event))
        used = {self._key(plan.predication, plan.subject.referent)}
        own = plan.predication.event
        used |= {happened(event) for event in events or () if event.label == own}
        return ClauseContext(plan.level, others, events or (), mention, aspect, used)

    def attach(
        self,
        rng: np.random.Generator,
        plan: SentencePlan,
        others: Sequence[str] = (),
        mention: Callable[[str], NounPhrase] | None = None,
        events: Sequence[SceneEvent] | None = None,
        aspect: Callable[[SceneEvent], str] | None = None,
    ) -> SentencePlan:
        """The plan with relative clauses drawn for its noun phrases: the subject first, then
        the object."""
        if plan.level == CLASS or not self.enabled:
            return plan
        context = self.context(plan, others, mention, events, aspect)
        subject = self.extend(rng, plan.subject, context)
        predication = plan.predication
        if predication.object is not None:
            target = self.extend(rng, predication.object, context)
            predication = dataclasses.replace(predication, object=target)
        return SentencePlan(subject, predication)

    @staticmethod
    def _key(predication: Predication, subject: str, patient: str | None = None) -> tuple:
        """What a predication says, to keep a sentence from saying the same thing twice."""
        if predication.event is not None:
            return ("event", predication.event)
        target = patient if predication.object is None else predication.object.referent
        return (predication.label, subject, target)

    def extend(
        self, rng: np.random.Generator, phrase: NounPhrase, context: ClauseContext, depth: int = 1
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
        used = context.used
        mention_of = context.mention or (lambda instance: mention(self.facts, instance))
        as_object = rng.random() < self.settings.object_share
        if phrase.negated:
            as_object = False  # the negated PROPERTY literals need a subject relative to join
        for object_relative in (as_object, not as_object):
            if object_relative and phrase.negated:
                continue
            options = self._options(referent, context, object_relative)
            if not options:
                continue
            predication, other = options[int(rng.integers(len(options)))]
            if context.aspect is not None and predication.event is not None:
                event = next(e for e in context.events if e.label == predication.event)
                predication = dataclasses.replace(predication, aspect=context.aspect(event))
            used.add(self._key(predication, other or referent, referent if other else None))
            used |= {happened(e) for e in context.events if e.label == predication.event}
            if object_relative:
                assert other is not None
                agent = self.extend(rng, mention_of(other), context, depth + 1)
                return dataclasses.replace(phrase, clause=RelativeClause((predication,), agent))
            if predication.object is not None:
                target = mention_of(predication.object.referent)
                target = self.extend(rng, target, context, depth + 1)
                predication = dataclasses.replace(predication, object=target)
            return dataclasses.replace(phrase, clause=RelativeClause((predication,)))
        return phrase

    def _options(
        self, referent: str, context: ClauseContext, object_relative: bool
    ) -> list[tuple[Predication, str | None]]:
        """The true propositions that a relative clause about a referent can express, as a
        predication and, for an object relative, the clause's own subject. The object of a
        subject relative is a plain mention, which the caller replaces."""
        facts = self.facts
        used = context.used
        options: list[tuple[Predication, str | None]] = []
        if context.level == EVENT:
            for event in context.events:
                if ("event", event.label) in used or happened(event) in used:
                    continue
                names = facts.event_names(event)
                if not names:
                    continue
                kind = VERB if event.transitive else CAN
                report = (event.label, self.tense, SIMPLE)  # the aspect is drawn once chosen
                if object_relative and event.patient == referent:
                    options.append((Predication(kind, names[0], True, None, *report), event.agent))
                elif not object_relative and event.agent == referent:
                    target = None if event.patient is None else mention(facts, event.patient)
                    options.append((Predication(kind, names[0], True, target, *report), None))
            return options
        truth = facts.truth
        others = context.others
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
            return Proposition(
                EVENT,
                subject,
                predicate,
                scene=scene_of(predication.event),
                event=event_of(predication.event),
                tense=predication.tense,
                aspect=predication.aspect,
            )
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


# ---------------------------------------------------------------------------------------------
# Mentions in a document
# ---------------------------------------------------------------------------------------------


class MentionRules:
    """The mention rules of one language: the settings, and the preference order of the
    attributes that tell referents apart, drawn once."""

    def __init__(self, config: Config, facts: Facts, streams: Streams) -> None:
        self.settings = config.mention
        self.facts = facts
        self.truth = facts.truth
        self.world = facts.world
        scalars = tuple(dict.fromkeys(pole.rsplit(".", 1)[0] for pole in facts.poles))
        attributes = facts.features[IS] + facts.features[HAS] + scalars
        order = streams.substream("mentions", "preference").permutation(len(attributes))
        self.preference: tuple[str, ...] = tuple(attributes[int(i)] for i in order)
        """PROPERTY features, PART features, and scalar dimensions, in the order they are
        tried."""

    def path(self, instance: str) -> tuple[str, ...]:
        """The categories an instance is below, from the top to its leaf."""
        return self.truth.paths[self.truth.instance_index[instance]]

    def nouns(self, instance: str, exclude: str | None = None) -> tuple[list[str], np.ndarray]:
        """The categories whose noun can name an instance, and their weights: its leaf and the
        categories above it that have a word, by ``mention.level_weights``."""
        named = [c for c in self.path(instance) if c in self.facts.named and c != exclude]
        weights = np.array(
            [self.settings.level_weights[self.facts.level[c] - 1] for c in named], dtype=float
        )
        return named, weights

    def value(self, instance: str, attribute: str, noun: str) -> Literal | None:
        """The literal that states an instance's value on an attribute, in a noun phrase headed
        by ``noun``: an adjective for a PROPERTY feature it has, a with-phrase or a
        without-phrase for a PART feature, and a pole adjective for a scalar on which it is at a
        pole of the noun's category. None when no adjective or with-phrase states the value."""
        row = self.truth.instance_index[instance]
        if attribute.startswith(SCALAR_PREFIX):
            for side in ("HIGH", "LOW"):
                pole = f"{attribute}.{side}"
                if pole in self.facts.named and self.fits(row, Literal(pole), noun):
                    return Literal(pole)
            return None
        value = bool(self.world.column(attribute)[row])
        if attribute.startswith(PROPERTY_PREFIX):
            return Literal(attribute) if value else None
        return Literal(attribute, value)

    def fits(self, row: int, literal: Literal, noun: str) -> bool:
        """Whether an instance satisfies a literal of a noun phrase headed by ``noun``. A pole
        is relative to the noun's category."""
        if literal.pole:
            return bool(self.truth.pole_mask(literal.feature, self.world.below(noun))[0][row])
        value = bool(self.world.column(literal.feature)[row])
        return value == literal.positive

    def matches(self, phrase: NounPhrase, cast: Sequence[str]) -> list[str]:
        """The instances among ``cast`` that the noun and the modifiers of a noun phrase fit."""
        assert phrase.noun is not None
        found = []
        for instance in cast:
            if phrase.noun not in self.path(instance):
                continue
            row = self.truth.instance_index[instance]
            if all(self.fits(row, literal, phrase.noun) for literal in phrase.restriction):
                found.append(instance)
        return found

    def distinguish(
        self, instance: str, noun: str, cast: Sequence[str], avoid: Sequence[str] = ()
    ) -> tuple[tuple[Literal, ...], bool]:
        """The modifiers that tell an instance apart from the other instances among ``cast``
        that the noun fits, by the incremental algorithm, and whether they pick out the
        instance alone. The limits on adjectives and with-phrases hold. ``avoid`` are the
        features that the sentence itself states, which no modifier says again."""
        truth = self.truth
        left = [
            truth.instance_index[other]
            for other in cast
            if other != instance and noun in self.path(other)
        ]
        chosen: list[Literal] = []
        adjectives = with_phrases = 0
        for attribute in self.preference:
            if not left:
                break
            literal = self.value(instance, attribute, noun)
            if literal is None or literal.feature in avoid:
                continue
            is_with = literal.feature.startswith(PART_PREFIX)
            if is_with and with_phrases >= self.settings.max_with_phrases:
                continue
            if not is_with and adjectives >= self.settings.max_adjectives:
                continue
            kept = [row for row in left if self.fits(row, literal, noun)]
            if len(kept) == len(left):
                continue  # the attribute rules out no one
            left = kept
            chosen.append(literal)
            with_phrases += is_with
            adjectives += not is_with
        return tuple(chosen), not left


class Mentions:
    """The mentions of one document: which instances were mentioned, and how each next mention
    is made."""

    def __init__(
        self, rules: MentionRules, cast: Sequence[str] = (), modifiers: bool = False
    ) -> None:
        self.rules = rules
        self.cast = tuple(cast)
        """The participants of the document's scenes: the instances a mention could be taken
        for."""
        self.modifiers = modifiers
        """Whether mentions take modifiers at ``mention.modifier_rate``, beside the modifiers
        that tell referents apart."""
        self.referents: dict[str, str] = {}
        """The referent label (``REF.<n>``) of every instance mentioned so far, in the order of
        first mention."""
        self._previous: tuple[str | None, frozenset[str]] | None = None
        self._current: list[str] = []

    def label(self, instance: str) -> str:
        return self.referents[instance]

    def pronoun_allowed(self, instance: str) -> bool:
        """Whether a mention can be ``it``: the referent was mentioned in the sentence before,
        and was the only referent there, or its subject."""
        if self._previous is None or instance in self._current:
            return False
        subject, mentioned = self._previous
        return instance in mentioned and (mentioned == {instance} or subject == instance)

    def noun_phrase(
        self,
        rng: np.random.Generator,
        instance: str,
        *,
        noun: str | None = None,
        exclude: str | None = None,
        pronoun: bool = True,
        avoid: Sequence[str] = (),
    ) -> NounPhrase:
        """The next mention of an instance. ``noun`` fixes the category that the noun names (the
        comparison class of a scalar pole). ``exclude`` is a category the noun must not name
        (the predicate of "is a penguin"). ``pronoun`` says whether the mention can be ``it``.
        ``avoid`` are the features that a modifier must not state: what the sentence says."""
        rules, settings = self.rules, self.rules.settings
        first = instance not in self.referents
        if first:
            self.referents[instance] = f"{REFERENT_PREFIX}{len(self.referents) + 1}"
        can_be_pronoun = pronoun and noun is None and not first and self.pronoun_allowed(instance)
        self._current.append(instance)
        if can_be_pronoun and rng.random() < settings.pronoun_rate:
            return NounPhrase(INSTANCE_NP, instance)
        fixed = noun is not None
        if noun is None:
            named, weights = rules.nouns(instance, exclude)
            if not named:
                return NounPhrase(INSTANCE_NP, instance)  # no noun can name it here
            if weights.sum() == 0:
                weights = np.ones(len(named))
            noun = named[int(rng.choice(len(named), p=weights / weights.sum()))]
        if first:
            literals: tuple[Literal, ...] = ()
        else:
            literals, alone = rules.distinguish(instance, noun, self.cast, avoid)
            leaf = rules.path(instance)[-1]
            if not alone and not fixed and noun != leaf and leaf != exclude:
                if leaf in rules.facts.named:
                    # a more specific noun fits fewer of the participants
                    noun = leaf
                    literals, alone = rules.distinguish(instance, noun, self.cast, avoid)
        if self.modifiers and rng.random() < settings.modifier_rate:
            literals = self._modifier(rng, instance, noun, literals, avoid)
        return NounPhrase(INSTANCE_NP, instance, noun, "a" if first else "the", literals)

    def _modifier(
        self,
        rng: np.random.Generator,
        instance: str,
        noun: str,
        literals: tuple[Literal, ...],
        avoid: Sequence[str],
    ) -> tuple[Literal, ...]:
        """The literals with one more modifier, chosen from the features true of the referent:
        an adjective for a PROPERTY feature or a pole, or a with-phrase for a PART feature."""
        rules, settings = self.rules, self.rules.settings
        stated = {literal.feature for literal in literals} | set(avoid)
        adjectives = sum(not x.feature.startswith(PART_PREFIX) for x in literals)
        with_phrases = len(literals) - adjectives
        options = []
        for attribute in rules.preference:
            literal = rules.value(instance, attribute, noun)
            if literal is None or not literal.positive or literal.feature in stated:
                continue
            is_with = literal.feature.startswith(PART_PREFIX)
            if (with_phrases if is_with else adjectives) >= (
                settings.max_with_phrases if is_with else settings.max_adjectives
            ):
                continue
            options.append(literal)
        if not options:
            return literals
        options.sort(key=lambda x: x.sort_key)
        return literals + (options[int(rng.integers(len(options)))],)

    def distinguished(self, phrase: NounPhrase) -> bool | None:
        """For a definite mention with a noun: whether the noun and the modifiers pick out its
        referent alone among the participants of the document's scenes. None for any other
        noun phrase."""
        if phrase.kind != INSTANCE_NP or phrase.pronoun or phrase.determiner != "the":
            return None
        cast = self.cast if phrase.referent in self.cast else self.cast + (phrase.referent,)
        return self.rules.matches(phrase, cast) == [phrase.referent]

    def end_sentence(self, subject: str | None) -> None:
        """Close a sentence: what it mentioned decides the pronouns of the next one."""
        self._previous = (subject, frozenset(self._current))
        self._current = []

    def drop_sentence(self, referents: Mapping[str, str]) -> None:
        """Forget a sentence that was planned and not kept: its mentions never happened."""
        self.referents = dict(referents)
        self._current = []
