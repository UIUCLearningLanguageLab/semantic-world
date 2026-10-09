"""Layer 3: documents. The discourse planner chooses and orders propositions by document type,
and decides how each referent is mentioned.

There are four document types (``documents.mix``):

- **encyclopedic, about a category.** The topic is a category. Each sentence draws a kind of
  content: a membership fact, a fact about the topic, a fact about one of its subcategories, or
  a relation fact with the topic as agent or as patient. By default
  (``documents.content_kind_weights: proportional``) a kind is drawn in proportion to the
  number of facts of that kind the document can still state, so the mix follows the world;
  with ``equal``, each kind has the same chance. With ``documents.relation_fact_share``, a
  sentence draws a relation fact with that probability instead, and the other kinds share the
  rest. A fact about the topic can be followed by the matching fact about a sibling category, at
  ``documents.sibling_contrast_rate``, when the sibling differs;
- **encyclopedic, about a feature.** The topic is a PROPERTY or PART feature, a one-place event
  type, or a two-place event type or category. A sentence states a rule, at
  ``propositions.rule_statement_rate``, or says which categories have the feature and which lack
  it;
- **entity narrative.** The topic is one instance. The document narrates the events of
  ``entity.scenes`` scenes that involve the instance, in time order, scene by scene;
- **situational narrative.** One scene. The document narrates its events in time order.

In a narrative, an event sentence is followed by an instance-level description at
``documents.instance_description_rate``: of the topic in an entity narrative, and of a
participant of the event in a situational narrative, a newly introduced one first. Each report
of an event chooses its aspect at ``documents.progressive_rate`` (CG.64); with
``documents.one_aspect_per_event`` every report of one event in one document uses the aspect of
its first report.

**Restricted subjects.** The subject of a class-level fact takes a restriction ("red penguins")
at ``propositions.restriction_rate``, and a restrictive relative clause ("penguins that can
swim") at ``mention.relative_clauses.rate``. Both are drawn before the fact, so the fact is true
of the restricted subject.

**Bare plurals.** A class-level fact is stated with its quantifier word, or with a bare plural
at ``quantifiers.generic_rate`` when the bare plural can express its quantifier
(``quantifiers.bare_plural.expresses``, and always ``nec_all`` for membership and rule
statements). A fact whose quantifier the language has no word for is stated with the bare
plural. The choice never changes the proposition.

**Ordering.** An encyclopedic document follows a template: membership, defining facts
(``nec_all``, ``all``, ``no``, and ``nec_no``), characteristic facts (``most``, and scalar
poles), rarer facts (``some``), relation facts, and rule statements. ``documents.shuffle`` moves
from the template (0) to a random order (1). A sibling contrast stays right after its fact. A
narrative follows time.

**Quantifier weights.** A fact is stated with its strongest true quantifier. With
``quantifiers.weights``, a document chooses a fact with a probability proportional to the weight
of that quantifier, so a study can rebalance the mix. Equal weights, the default, change nothing.

**Streams.** Each document draws from its own parts of four streams, named by its label: the
document's type, topic, length, and order from ``corpus:documents``; its facts from
``corpus:propositions``; its mentions and the aspects of its reports from ``corpus:mentions``;
and the grammar's choices from ``corpus:grammar``. Each scene draws from its own part of
``corpus:scenes``. So a grammar setting never changes what a document says, or in what order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.corpus.config import DOCUMENT_TYPES, Config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.grammar import NounPhrase, Predication, SentencePlan
from semantic_world.corpus.histories import SceneEvent, SceneGenerator, involving, scene_events
from semantic_world.corpus.lexicon import Lexicon, build_lexicon
from semantic_world.corpus.logical import logical_form
from semantic_world.corpus.mentions import (
    ClauseContext,
    MentionRules,
    Mentions,
    RelativeClauses,
    happened,
    plan_for,
)
from semantic_world.corpus.propositions import (
    CAN,
    CLASS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    PROGRESSIVE,
    SCALAR,
    SIMPLE,
    UNIVERSALS,
    VERB,
    CategoryTerm,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, Sentence, as_json
from semantic_world.corpus.renderings import propositional
from semantic_world.corpus.streams import Streams
from semantic_world.corpus.world import World, load_world
from semantic_world.world.history import History

CATEGORY, FEATURE, ENTITY, SITUATIONAL = DOCUMENT_TYPES

DOCUMENT_PREFIX = "DOC."
SENTENCE_INFIX = ".SENT."
PROPOSITION_PREFIX = "PROP."

MEMBERSHIP = "membership"
DEFINING = "defining"
CHARACTERISTIC = "characteristic"
RARER = "rarer"
RELATION = "relation"
RULE = "rule"
TEMPLATE = (MEMBERSHIP, DEFINING, CHARACTERISTIC, RARER, RELATION, RULE)
"""The sections of an encyclopedic document, in the order of the template."""
TOPIC = "topic"
SUBCATEGORY = "subcategory"
CONTENT_KINDS = (MEMBERSHIP, TOPIC, SUBCATEGORY, RELATION, RULE)
"""The kinds of content of a class-level sentence: a membership fact, a fact about the topic, a
fact about one of its subcategories, a relation fact, and a rule statement."""
CONTRAST = "contrast"
"""A sibling contrast: it stays right after the fact it matches."""
EVENT_SECTION = "event"
DESCRIPTION = "description"
POLE = "pole"
"""The strength of a scalar-pole fact, which takes no quantifier word."""

_STRONG = UNIVERSALS + (MOST,)
_MAX_REDRAWS = 100


@dataclass(frozen=True)
class DocumentSentence:
    """One sentence of a document, with everything that is kept about it."""

    label: str
    """``DOC.<n>.SENT.<k>``."""
    section: str
    """What the sentence does in its document: a section of the template, ``contrast``,
    ``event``, or ``description``."""
    proposition: Proposition
    """The proposition of the main clause, with its label and its grounding."""
    sentence: Sentence
    logical_form: dict[str, Any]
    propositional: str
    readings: tuple[str, ...]
    """The kinds of logical form that the sentence's words allow."""
    coreference: tuple[str | None, ...]
    """For each noun phrase, in the order of the tree: the referent it mentions (``REF.<n>``),
    or None for a noun phrase that names a category."""
    distinguished: tuple[bool | None, ...]
    """For each noun phrase, in the same order: for a definite mention with a noun, whether it
    picks out its referent alone among the participants of the document's scenes."""
    strength: str | None = None
    """For a class-level sentence: the quantifier of its fact (``nec_all``, ``all``, ``most``,
    ``some``, ``no``, or ``nec_no``), or ``pole`` for a scalar pole, which takes no quantifier
    word. It is kept for the statistics, and is not written to ``documents.jsonl``."""

    @property
    def plan(self) -> SentencePlan:
        return self.sentence.plan

    @property
    def bare_plural(self) -> bool:
        return self.plan.bare_plural

    def to_json(self) -> dict[str, Any]:
        """The sentence as ``documents.jsonl`` holds it. The word forms come with ``render``."""
        sentence = self.sentence
        return {
            "label": self.label,
            "tokens": list(sentence.tokens),
            "words": None,
            "text": None,
            "formal": sentence.formal,
            "conceptual": sentence.conceptual,
            "propositional": self.propositional,
            "tree": as_json(sentence.tree),
            "logical_form": self.logical_form,
            "referents": [
                {"referent": referent, "noun": noun} for referent, noun in sentence.referents
            ],
            "events": list(sentence.event_labels),
            "coreference": list(self.coreference),
            "distinguished": list(self.distinguished),
            "readings": list(self.readings),
        }


@dataclass(frozen=True)
class Document:
    label: str
    """``DOC.<n>``."""
    type: str
    topic: str
    """A category, a feature or an event type, an instance, or, for a situational narrative,
    its scene."""
    scenes: tuple[History, ...]
    referents: dict[str, str]
    """The instance of every referent, by referent label, in the order of first mention."""
    sentences: tuple[DocumentSentence, ...]
    drawn_length: int = 0
    """The length that was drawn for the document, in sentences. The document can be shorter. It
    is kept for the statistics, and is not written to ``documents.jsonl``."""

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "type": self.type,
            "topic": self.topic,
            "scenes": [scene.label for scene in self.scenes],
            "referents": dict(self.referents),
            "sentences": [sentence.to_json() for sentence in self.sentences],
        }


@dataclass
class _Item:
    """One planned sentence, before it is realized."""

    section: str
    proposition: Proposition
    plan: SentencePlan | None = None
    follows: bool = False
    """Whether the sentence stays right after the one before it (a sibling contrast)."""
    strength: str | None = None
    """The quantifier of a class-level fact, or ``pole``."""
    bare: bool = False
    """Whether a class-level fact is said with a bare plural."""


@dataclass
class _Draft:
    type: str
    topic: str
    items: list[_Item]
    scenes: tuple[History, ...] = ()
    mentions: Mentions | None = None
    length: int = 0
    """The drawn length."""


class Aspects:
    """The aspects of the reports of one document: each report draws its aspect at
    ``documents.progressive_rate`` from the document's part of ``corpus:mentions``, and with
    ``documents.one_aspect_per_event`` every report of one event uses the first report's."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self.rate = config.documents.progressive_rate
        self.one_per_event = config.documents.one_aspect_per_event
        self.rng = rng
        self.chosen: dict[str, str] = {}

    def of(self, event: SceneEvent) -> str:
        if self.one_per_event and event.label in self.chosen:
            return self.chosen[event.label]
        aspect = PROGRESSIVE if self.rng.random() < self.rate else SIMPLE
        self.chosen.setdefault(event.label, aspect)
        return aspect


def content_words(plan: SentencePlan) -> int:
    """The nouns, adjectives, and verbs of a sentence plan: what ``mention.max_content_words``
    limits. Function words are not counted, so the count does not depend on the grammar."""

    def phrase(noun_phrase: NounPhrase | None) -> int:
        if noun_phrase is None:
            return 0
        count = (0 if noun_phrase.pronoun else 1) + len(noun_phrase.restriction)
        if noun_phrase.clause is not None:
            count += phrase(noun_phrase.clause.agent)
            count += sum(predication(p) for p in noun_phrase.clause.predications)
        return count

    def predication(part: Predication) -> int:
        return 1 + phrase(part.object)

    return phrase(plan.subject) + predication(plan.predication)


def reading_counts(documents: Sequence[Document]) -> dict[str, Any]:
    """How often sentences are ambiguous between the levels: the number of sentences, the
    number with more than one level reading, and the count of every set of level readings. The
    quantifier readings of class-level sentences are counted apart, under
    ``quantifier_readings``."""
    sets: dict[str, int] = {}
    quantifier_sets: dict[str, int] = {}
    total = ambiguous = 0
    for document in documents:
        for sentence in document.sentences:
            total += 1
            levels = tuple(r for r in sentence.readings if r in ("generic", "capacity", "event"))
            ambiguous += len(levels) > 1
            key = "+".join(levels)
            sets[key] = sets.get(key, 0) + 1
            if sentence.proposition.level == CLASS:
                quantifiers = "+".join(r for r in sentence.readings if r not in levels) or "none"
                quantifier_sets[quantifiers] = quantifier_sets.get(quantifiers, 0) + 1
    return {
        "sentences": total,
        "ambiguous": ambiguous,
        "ambiguous_share": ambiguous / total if total else 0.0,
        "readings": dict(sorted(sets.items())),
        "quantifier_readings": dict(sorted(quantifier_sets.items())),
    }


class Planner:
    """The documents of one corpus. Documents are made in order, because scenes and propositions
    are numbered across the corpus."""

    def __init__(
        self,
        config: Config,
        world: World | None = None,
        *,
        lexicon: Lexicon | None = None,
    ) -> None:
        self.config = config
        self.world = load_world(config) if world is None else world
        self.streams = Streams(config.seed)
        self.lexicon = lexicon or build_lexicon(config, self.world, self.streams)
        self.truth = Truth(config, self.world)
        self.facts = Facts(config, self.world, self.lexicon, self.truth)
        self.scene_generator = SceneGenerator(config, self.world)
        self.realizer = Realizer(config, self.lexicon, self.streams)
        self.clauses = RelativeClauses(config, self.facts)
        self.rules = MentionRules(config, self.facts, self.streams)
        self.scenes: list[History] = []
        """Every scene of the documents made so far, in order."""
        self._proposition_ids: dict[Proposition, str] = {}
        self._made = 0
        facts = self.facts
        self._instances = self.world.instances
        self._by_level: dict[int, list[str]] = {}
        for category in facts.categories:
            self._by_level.setdefault(facts.level[category], []).append(category)
        self._children: dict[str, list[str]] = {c: [] for c in facts.categories}
        self._siblings: dict[str, list[str]] = {}
        parents: dict[str | None, list[str]] = {}
        for label, info in self.world.category.items():
            if label not in facts.named:
                continue
            parents.setdefault(info.parent, []).append(label)
            if info.parent in self._children:
                self._children[info.parent].append(label)
        for group in parents.values():
            for category in group:
                self._siblings[category] = [c for c in group if c != category]
        self._feature_topics = (
            facts.features[IS] + facts.features[HAS] + facts.features[CAN] + facts.verbs
        )
        self._available: dict[str, dict[str, tuple[tuple, ...]]] = {}

    # Documents -------------------------------------------------------------------------------

    def generate(self, count: int | None = None) -> list[Document]:
        """The next ``count`` documents: by default, ``documents.count`` of them."""
        count = self.config.documents.count if count is None else count
        return [self.next_document() for _ in range(count)]

    def __iter__(self) -> Iterator[Document]:
        while True:
            yield self.next_document()

    def next_document(self) -> Document:
        number = self._made + 1
        label = f"{DOCUMENT_PREFIX}{number}"
        rngs = {
            name: self.streams.substream(name, label)
            for name in ("documents", "propositions", "mentions", "grammar")
        }
        mix = self.config.documents.mix
        weights = np.array([mix[kind] for kind in DOCUMENT_TYPES], dtype=float)
        makers = {
            CATEGORY: self._category_document,
            FEATURE: self._feature_document,
            ENTITY: self._narrative,
            SITUATIONAL: self._narrative,
        }
        for _ in range(_MAX_REDRAWS):
            kind = DOCUMENT_TYPES[int(rngs["documents"].choice(4, p=weights / weights.sum()))]
            draft = makers[kind](kind, rngs)
            if draft is not None and draft.items:
                break
        else:
            raise CorpusError(
                f"no document with a sentence could be made for {label}: the world, the "
                "lexicon, or the document settings leave nothing to say"
            )
        self._made = number
        self.scenes += draft.scenes
        return self._realize(label, draft, rngs["grammar"])

    def _realize(self, label: str, draft: _Draft, rng: np.random.Generator) -> Document:
        mentions = draft.mentions
        referents = {} if mentions is None else mentions.referents
        mode = self.config.propositional_referents
        sentences = []
        for number, item in enumerate(draft.items, start=1):
            plan = item.plan or plan_for(self.facts, item.proposition, bare=item.bare)
            proposition = item.proposition
            identifier = self._proposition_ids.setdefault(
                proposition, f"{PROPOSITION_PREFIX}{len(self._proposition_ids) + 1}"
            )
            proposition = dataclasses.replace(proposition, id=identifier)
            sentence = self.realizer.realize(plan, rng)
            form = logical_form(plan, proposition, referents)
            sentences.append(
                DocumentSentence(
                    label=f"{label}{SENTENCE_INFIX}{number}",
                    section=item.section,
                    proposition=proposition,
                    sentence=sentence,
                    logical_form=form,
                    propositional=propositional(form, mode),
                    readings=readings(sentence.tree, self.lexicon, self.config),
                    coreference=tuple(referents.get(r) for r, _ in sentence.referents),
                    distinguished=tuple(
                        None if mentions is None else mentions.distinguished(phrase)
                        for phrase in sentence.phrases
                    ),
                    strength=item.strength,
                )
            )
        return Document(
            label=label,
            type=draft.type,
            topic=draft.topic,
            scenes=draft.scenes,
            referents={r: instance for instance, r in referents.items()},
            sentences=tuple(sentences),
            drawn_length=draft.length,
        )

    def _length(self, kind: str, rng: np.random.Generator) -> int:
        limits = self.config.documents.sentences[kind]
        return int(rng.integers(limits.min, limits.max + 1))

    # Class-level facts -----------------------------------------------------------------------

    @staticmethod
    def _key(proposition: Proposition) -> tuple:
        """What a proposition says, apart from how strongly: a document says it once."""
        return (proposition.subject, proposition.predicate, proposition.negative)

    def _section(self, fact: Proposition) -> str:
        kind = fact.predicate.kind
        if fact.rule is not None:
            return RULE
        if kind == MEMBER:
            return MEMBERSHIP
        if kind == VERB:
            return RELATION
        if kind == SCALAR or fact.quantifier == MOST:
            return CHARACTERISTIC
        return DEFINING if fact.quantifier in UNIVERSALS else RARER

    def _restricted(self, rng: np.random.Generator, term: CategoryTerm) -> CategoryTerm:
        """A category term with a restriction, at ``propositions.restriction_rate``, and a
        relative clause, at ``mention.relative_clauses.rate``."""
        if rng.random() < self.config.propositions.restriction_rate:
            term = self.facts.draw_restriction(rng, term) or term
        if self.clauses.enabled and rng.random() < self.clauses.settings.rate:
            term = self.facts.draw_clause(rng, term) or term
        return term

    def _fits(self, proposition: Proposition) -> bool:
        """Whether a class-level sentence stays within ``mention.max_content_words``, and says
        nothing twice: its predicate is no part of its subject's restriction."""
        subject = proposition.subject
        assert isinstance(subject, CategoryTerm)
        said = [x.feature for x in subject.restriction] + [c.label for c in subject.clauses]
        if proposition.predicate.label in said:
            return False
        limit = self.config.mention.max_content_words
        return content_words(plan_for(self.facts, proposition)) <= limit

    def _bare(self, rng: np.random.Generator, fact: Proposition) -> bool:
        """Whether a class-level fact is said with a bare plural: always when the language has
        no word for its quantifier, and otherwise at the generic rate when the bare plural can
        express it."""
        if fact.quantifier is None or fact.quantifier not in self.truth.sayable:
            return True
        if not self.truth.bare_plural_expresses(fact):
            return False
        return bool(rng.random() < self.config.quantifiers.generic_rate)

    def _stated(
        self,
        rng: np.random.Generator,
        fact: Proposition,
        section: str | None = None,
        follows: bool = False,
    ) -> _Item:
        """The planned sentence of a class-level fact, with its quantifier word or bare."""
        strength = POLE if fact.predicate.kind == SCALAR else fact.quantifier
        return _Item(
            section or self._section(fact), fact, None, follows, strength, self._bare(rng, fact)
        )

    def _first_fact(
        self,
        rng: np.random.Generator,
        candidates: Sequence[tuple[CategoryTerm, Predicate]],
        negative: bool,
        stated: set,
    ) -> Proposition | None:
        """The fact of the first candidate, in a random order, that has one: the strongest true
        proposition of the wanted polarity that the document has not stated yet. The polarity
        is kept, so that the share of negative sentences stays at the negation rate: when the
        facts of one polarity run out, the document says something else, or ends."""
        facts = self.facts

        def fact_of(index: int) -> Proposition | None:
            subject, predicate = candidates[index]
            fact = facts.class_fact(subject, predicate, negative)
            if fact is not None and self._key(fact) not in stated and self._fits(fact):
                return fact
            return None

        if not self.config.quantifiers.weighted:
            for index in rng.permutation(len(candidates)):
                fact = fact_of(int(index))
                if fact is not None:
                    return fact
            return None
        # With weights, a fact is chosen with a probability proportional to the weight of its
        # quantifier: a candidate is drawn uniformly, and kept at its weight over the largest.
        top = max(self.config.quantifiers.weights.values())
        left = list(range(len(candidates)))
        found: dict[int, Proposition | None] = {}
        while left:
            position = int(rng.integers(len(left)))
            index = left[position]
            if index not in found:
                found[index] = fact_of(index)
            fact = found[index]
            weight = 0.0 if fact is None else self._weight(fact)
            if weight == 0:
                left[position] = left[-1]
                left.pop()
            elif rng.random() < weight / top:
                return fact
        return None

    def _weight(self, fact: Proposition) -> float:
        """The weight of a fact's quantifier (``quantifiers.weights``). A scalar pole takes no
        quantifier word, and has the weight 1."""
        if fact.quantifier is None:
            return 1.0
        return self.config.quantifiers.weights.get(fact.quantifier, 1.0)

    def _weight_allows(
        self, rng: np.random.Generator, fact: Proposition, other: Proposition
    ) -> bool:
        """Whether a chosen fact can give way to another one, which says the same of a narrower
        subject. With quantifier weights, a fact of another quantifier takes its place with a
        probability of the ratio of the two weights, so the weights still hold."""
        if not self.config.quantifiers.weighted or fact.quantifier == other.quantifier:
            return True
        return bool(rng.random() * self._weight(fact) < self._weight(other))

    def _negative(self, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.config.propositions.negation_rate[CLASS])

    def _feature_predicates(self, term: CategoryTerm) -> list[Predicate]:
        """The one-place predicates that a fact about a category can have, apart from
        membership. A scalar pole is a statement about the category itself, so a restricted
        term takes none."""
        return [
            predicate
            for predicate in self.facts.class_predicates(term.category, patients=())
            if predicate.kind != MEMBER
            and not (predicate.kind == SCALAR and (term.restriction or term.clauses))
        ]

    def _membership_candidates(
        self, topic: str, negative: bool
    ) -> list[tuple[CategoryTerm, Predicate]]:
        facts = self.facts
        if negative:
            return [
                (CategoryTerm(topic), Predicate(MEMBER, c))
                for c in facts.categories
                if facts.level[c] <= facts.level[topic] and self.truth.disjoint(topic, c)
            ]
        above = [
            (CategoryTerm(topic), Predicate(MEMBER, a))
            for a in self.truth.ancestors(topic)
            if a in facts.named
        ]
        below = [(CategoryTerm(c), Predicate(MEMBER, topic)) for c in self._children[topic]]
        return above + below

    def _membership_fact(
        self, rng: np.random.Generator, topic: str, negative: bool, stated: set
    ) -> Proposition | None:
        # a positive fact names a category above or below, and a negative one a category apart
        return self._first_fact(rng, self._membership_candidates(topic, negative), negative, stated)

    def _relation_candidates(
        self, topic: CategoryTerm, as_agent: bool
    ) -> list[tuple[CategoryTerm, Predicate]]:
        facts = self.facts
        pairs = [(verb, other) for verb in facts.verbs for other in facts.categories]
        if as_agent:
            return [(topic, Predicate(VERB, verb, CategoryTerm(other))) for verb, other in pairs]
        return [(CategoryTerm(other), Predicate(VERB, verb, topic)) for verb, other in pairs]

    def _relation_fact(
        self, rng: np.random.Generator, topic: CategoryTerm, negative: bool, stated: set
    ) -> Proposition | None:
        """A relation fact with the topic as agent or as patient, each with the same chance."""
        as_agent = bool(rng.random() < 0.5)
        for agent_role in (as_agent, not as_agent):
            candidates = self._relation_candidates(topic, agent_role)
            fact = self._first_fact(rng, candidates, negative, stated)
            if fact is not None:
                return fact
        return None

    def _contrast(
        self, rng: np.random.Generator, topic: str, fact: Proposition, stated: set
    ) -> Proposition | None:
        """The matching fact about a sibling of the topic: the same predicate, with the opposite
        polarity. A fact has a contrast when it is a strong one (a universal, ``most``, or a
        scalar pole) about the plain topic, and the sibling's fact is strong too."""
        subject, predicate = fact.subject, fact.predicate
        plain = CategoryTerm(topic)
        strong = predicate.kind == SCALAR or fact.quantifier in _STRONG
        if not strong or predicate.kind == MEMBER:
            return None
        as_subject = subject == plain
        as_patient = predicate.patient == plain
        if not (as_subject or as_patient):
            return None
        siblings = self._siblings.get(topic, [])
        for index in rng.permutation(len(siblings)):
            sibling = CategoryTerm(siblings[int(index)])
            if as_subject:
                other = self.facts.class_fact(sibling, predicate, not fact.negative)
            else:
                other = self.facts.class_fact(
                    subject, dataclasses.replace(predicate, patient=sibling), not fact.negative
                )
            if other is None or self._key(other) in stated or self._weight(other) == 0:
                continue
            if predicate.kind == SCALAR or other.quantifier in _STRONG:
                return other
        return None

    def _order(self, rng: np.random.Generator, items: list[_Item]) -> list[_Item]:
        """The sentences of an encyclopedic document in the template order, moved toward a
        random order by ``documents.shuffle``. A sibling contrast stays after its fact."""
        units: list[list[_Item]] = []
        for item in items:
            if item.follows and units:
                units[-1].append(item)
            else:
                units.append([item])
        units.sort(key=lambda unit: TEMPLATE.index(unit[0].section))
        shuffle = self.config.documents.shuffle
        draws = rng.random(len(units))
        keys = [
            (1 - shuffle) * index / len(units) + shuffle * float(draws[index])
            for index in range(len(units))
        ]
        ordered = sorted(range(len(units)), key=lambda index: (keys[index], index))
        return [item for index in ordered for item in units[index]]

    # The kinds of content of a category document ----------------------------------------------

    def _available_facts(self, topic: str) -> dict[str, tuple[tuple, ...]]:
        """The keys of the facts of each kind that a document about a category could state, for
        the plain topic: membership facts, facts about the topic, facts about its subcategories,
        and relation facts in both roles. Computed once per category."""
        if topic not in self._available:
            facts = self.facts
            found: dict[str, list[tuple]] = {kind: [] for kind in CONTENT_KINDS}

            def keys(candidates, negative: bool) -> list[tuple]:
                return [
                    self._key(fact)
                    for subject, predicate in candidates
                    if (fact := facts.class_fact(subject, predicate, negative)) is not None
                    and self._fits(fact)
                ]

            for negative in (False, True):
                found[MEMBERSHIP] += keys(self._membership_candidates(topic, negative), negative)
                term = CategoryTerm(topic)
                found[TOPIC] += keys([(term, p) for p in self._feature_predicates(term)], negative)
                for child in self._children[topic]:
                    child_term = CategoryTerm(child)
                    found[SUBCATEGORY] += keys(
                        [(child_term, p) for p in self._feature_predicates(child_term)], negative
                    )
                for as_agent in (True, False):
                    found[RELATION] += keys(self._relation_candidates(term, as_agent), negative)
            self._available[topic] = {kind: tuple(found[kind]) for kind in CONTENT_KINDS}
        return self._available[topic]

    def _draw_content(
        self, rng: np.random.Generator, topic: str, kinds: list[str], stated: set
    ) -> str | None:
        """The kind of content of the next sentence of a category document: with
        ``documents.relation_fact_share``, a relation fact at that share and the other kinds
        among the rest; otherwise every kind by its weight, proportional to the facts of that
        kind still left to state, or equal. None when no fact is left."""
        settings = self.config.documents
        share = settings.relation_fact_share
        others = [k for k in kinds if k != RELATION]
        if share is not None and RELATION in kinds:
            if rng.random() < share:
                return RELATION
            kinds = others
        if settings.content_kind_weights == "equal":
            return kinds[int(rng.integers(len(kinds)))] if kinds else None
        available = self._available_facts(topic)
        counts = np.array(
            [sum(key not in stated for key in available[kind]) for kind in kinds], dtype=float
        )
        if counts.sum() == 0:
            return None
        return kinds[int(rng.choice(len(kinds), p=counts / counts.sum()))]

    def _category_document(self, kind: str, rngs: dict[str, np.random.Generator]) -> _Draft:
        rng, facts_rng = rngs["documents"], rngs["propositions"]
        weights = np.array(
            [
                weight if level in self._by_level else 0.0
                for level, weight in enumerate(self.config.documents.topic_level_weights, start=1)
            ]
        )
        if weights.sum() == 0:
            return _Draft(kind, "", [])
        level = int(rng.choice(len(weights), p=weights / weights.sum())) + 1
        topic = self._by_level[level][int(rng.integers(len(self._by_level[level])))]
        length = self._length(kind, rng)
        children = self._children[topic]
        kinds = [MEMBERSHIP, TOPIC]
        if children:
            kinds.append(SUBCATEGORY)
        if self.facts.verbs:
            kinds.append(RELATION)
        rate = self.config.documents.sibling_contrast_rate
        items: list[_Item] = []
        stated: set = set()
        for _ in range(4 * length + 20):
            if len(items) >= length:
                break
            content = self._draw_content(facts_rng, topic, kinds, stated)
            if content is None:
                break
            negative = self._negative(facts_rng)
            if content == MEMBERSHIP:
                fact = self._membership_fact(facts_rng, topic, negative, stated)
            elif content == RELATION:
                term = self._restricted(facts_rng, CategoryTerm(topic))
                fact = self._relation_fact(facts_rng, term, negative, stated)
            else:
                category = topic
                if content == SUBCATEGORY:
                    category = children[int(facts_rng.integers(len(children)))]
                term = self._restricted(facts_rng, CategoryTerm(category))
                candidates = [(term, p) for p in self._feature_predicates(term)]
                fact = self._first_fact(facts_rng, candidates, negative, stated)
            if fact is None:
                continue
            stated.add(self._key(fact))
            items.append(self._stated(facts_rng, fact))
            if len(items) < length and facts_rng.random() < rate:
                contrast = self._contrast(facts_rng, topic, fact, stated)
                if contrast is not None:
                    stated.add(self._key(contrast))
                    items.append(self._stated(facts_rng, contrast, CONTRAST, True))
        return _Draft(kind, topic, self._order(rng, items), length=length)

    def _feature_document(self, kind: str, rngs: dict[str, np.random.Generator]) -> _Draft:
        rng, facts_rng = rngs["documents"], rngs["propositions"]
        facts = self.facts
        if not self._feature_topics:
            return _Draft(kind, "", [])
        topic = self._feature_topics[int(rng.integers(len(self._feature_topics)))]
        length = self._length(kind, rng)
        is_verb = topic in facts.verbs
        rules: list[Proposition] = []
        topic_kind = VERB
        if not is_verb:
            topic_kind = next(k for k in (IS, HAS, CAN) if topic in facts.features[k])
            rules = list(
                dict.fromkeys(facts.rule_statements(topic) + facts.rule_statements_reading(topic))
            )
        categories = facts.categories
        if is_verb:
            candidates = [
                (CategoryTerm(agent), Predicate(VERB, topic, CategoryTerm(patient)))
                for agent in categories
                for patient in categories
            ]
        else:
            candidates = [(CategoryTerm(c), Predicate(topic_kind, topic)) for c in categories]
        rule_rate = self.config.propositions.rule_statement_rate
        items: list[_Item] = []
        stated: set = set()
        for _ in range(4 * length + 20):
            if len(items) >= length:
                break
            left = [r for r in rules if self._key(r) not in stated]
            if left and facts_rng.random() < rule_rate:
                fact: Proposition | None = left[int(facts_rng.integers(len(left)))]
            else:
                negative = self._negative(facts_rng)
                fact = self._first_fact(facts_rng, candidates, negative, stated)
                if fact is not None:
                    # the same fact about a restricted subject, when there is one
                    assert isinstance(fact.subject, CategoryTerm)
                    term = self._restricted(facts_rng, fact.subject)
                    if term != fact.subject:
                        narrowed = facts.class_fact(term, fact.predicate, fact.negative)
                        if (
                            narrowed is not None
                            and self._key(narrowed) not in stated
                            and self._fits(narrowed)
                            and self._weight_allows(facts_rng, fact, narrowed)
                        ):
                            fact = narrowed
            if fact is None:
                continue
            stated.add(self._key(fact))
            items.append(self._stated(facts_rng, fact))
        return _Draft(kind, topic, self._order(rng, items), length=length)

    # Facts for the test sets -----------------------------------------------------------------

    def draw_class_fact(
        self, rng: np.random.Generator, kinds: Sequence[str] = CONTENT_KINDS
    ) -> tuple[Proposition, bool] | None:
        """One true class-level proposition, drawn the way an encyclopedic document draws a
        sentence, for the test sets, with whether it is said with a bare plural. The topic is a
        category, drawn by ``documents.topic_level_weights``. The kind of content is drawn among
        ``kinds``, each with the same chance: a membership fact, a fact about the topic, a fact
        about a subcategory, a relation fact, or a rule statement. The polarity, the
        restriction, and the bare plural are drawn at their rates. None when the draw gives no
        fact."""
        weights = np.array(
            [
                weight if level in self._by_level else 0.0
                for level, weight in enumerate(self.config.documents.topic_level_weights, start=1)
            ]
        )
        if weights.sum() == 0:
            return None
        level = int(rng.choice(len(weights), p=weights / weights.sum())) + 1
        topic = self._by_level[level][int(rng.integers(len(self._by_level[level])))]
        content = kinds[int(rng.integers(len(kinds)))]
        negative = self._negative(rng)
        stated: set = set()
        fact: Proposition | None
        if content == RULE:
            rules = self.facts.rule_statements()
            fact = rules[int(rng.integers(len(rules)))] if rules else None
        elif content == MEMBERSHIP:
            fact = self._membership_fact(rng, topic, negative, stated)
        elif content == RELATION:
            if not self.facts.verbs:
                return None
            term = self._restricted(rng, CategoryTerm(topic))
            fact = self._relation_fact(rng, term, negative, stated)
        else:
            category = topic
            if content == SUBCATEGORY:
                children = self._children[topic]
                if not children:
                    return None
                category = children[int(rng.integers(len(children)))]
            term = self._restricted(rng, CategoryTerm(category))
            candidates = [(term, p) for p in self._feature_predicates(term)]
            fact = self._first_fact(rng, candidates, negative, stated)
        return None if fact is None else (fact, self._bare(rng, fact))

    # Narratives ------------------------------------------------------------------------------

    def _narrative(self, kind: str, rngs: dict[str, np.random.Generator]) -> _Draft:
        rng, facts_rng, mention_rng = rngs["documents"], rngs["propositions"], rngs["mentions"]
        seed = self._instances[int(rng.integers(len(self._instances)))]
        count = 1
        if kind == ENTITY:
            limits = self.config.entity_scenes
            count = int(rng.integers(limits.min, limits.max + 1))
        first = len(self.scenes) + 1
        scenes = tuple(
            self.scene_generator.scene(self.streams, first + k, seed) for k in range(count)
        )
        for scene in scenes:
            self.truth.add_scene(scene)
        topic = seed if kind == ENTITY else scenes[0].label
        cast = tuple(dict.fromkeys(p for scene in scenes for p in scene.participants))
        mentions = Mentions(self.rules, cast, modifiers=kind == ENTITY)
        aspects = Aspects(self.config, mention_rng)
        length = self._length(kind, rng)
        rate = self.config.documents.instance_description_rate
        items: list[_Item] = []
        stated: set = set()
        for scene in scenes:
            events = involving(scene, seed) if kind == ENTITY else scene_events(scene)
            for event in events:
                if len(items) >= length:
                    break
                report = self.facts.draw_event(mention_rng, event, aspects.of(event))
                if report is None:
                    continue  # no word can report the event
                known = set(mentions.referents)
                items.append(
                    self._event_sentence(mention_rng, report, event, scene, mentions, aspects)
                )
                if len(items) < length and facts_rng.random() < rate:
                    if kind == ENTITY:
                        about = seed
                    else:
                        taking_part = [event.agent] + ([event.patient] if event.patient else [])
                        new = [p for p in taking_part if p not in known]
                        pool = new or taking_part
                        about = pool[int(facts_rng.integers(len(pool)))]
                    described = self._description(facts_rng, mention_rng, about, mentions, stated)
                    if described is not None:
                        items.append(described)
        return _Draft(kind, topic, items, scenes, mentions, length)

    def _within_limit(self, plan: SentencePlan) -> bool:
        return content_words(plan) <= self.config.mention.max_content_words

    def _event_sentence(
        self,
        rng: np.random.Generator,
        report: Proposition,
        event: SceneEvent,
        scene: History,
        mentions: Mentions,
        aspects: Aspects,
    ) -> _Item:
        """The sentence that reports an event. A relative clause reports an earlier event of
        the scene."""
        events = self.truth.events_of(scene.label)
        earlier = events[: events.index(event)]
        known = dict(mentions.referents)
        plan: SentencePlan | None = None
        for with_clauses in (True, False):
            context = ClauseContext(
                EVENT,
                events=earlier,
                mention=lambda instance: mentions.noun_phrase(rng, instance, pronoun=False),
                used={("event", event.label), happened(event)},
                aspect=aspects.of,
            )
            subject = mentions.noun_phrase(rng, event.agent)
            if with_clauses:
                subject = self.clauses.extend(rng, subject, context)
            target = None
            if event.patient is not None:
                target = mentions.noun_phrase(rng, event.patient)
                if with_clauses:
                    target = self.clauses.extend(rng, target, context)
            predicate = report.predicate
            plan = SentencePlan(
                subject,
                Predication(
                    predicate.kind,
                    predicate.label,
                    True,
                    target,
                    event.label,
                    report.tense,
                    report.aspect,
                ),
            )
            if self._within_limit(plan):
                break
            mentions.drop_sentence(known)  # too long: say it again without relative clauses
        assert plan is not None
        mentions.end_sentence(event.agent)
        return _Item(EVENT_SECTION, report, plan)

    def _description(
        self,
        facts_rng: np.random.Generator,
        rng: np.random.Generator,
        instance: str,
        mentions: Mentions,
        stated: set,
    ) -> _Item | None:
        """An instance-level sentence about an instance: a fact drawn among its true facts,
        with the other participants of the document's scenes as the patients of its two-place
        event types."""
        others = tuple(p for p in mentions.cast if p != instance)
        fact = self.facts.draw_instance(facts_rng, instance, others)
        if fact is None or self._key(fact) in stated:
            return None
        predicate = fact.predicate
        patient = predicate.patient if isinstance(predicate.patient, str) else None
        known = dict(mentions.referents)
        plan: SentencePlan | None = None
        for with_clauses in (True, False):
            context = ClauseContext(
                INSTANCE,
                others=tuple(known),
                mention=lambda other: mentions.noun_phrase(rng, other, pronoun=False),
                used={(predicate.label, instance, patient)},
            )
            subject = mentions.noun_phrase(
                rng,
                instance,
                noun=predicate.comparison,
                exclude=predicate.label if predicate.kind == MEMBER else None,
                avoid=(predicate.label,),
            )
            if with_clauses:
                subject = self.clauses.extend(rng, subject, context)
            target = None
            if patient is not None:
                target = mentions.noun_phrase(rng, patient)
                if with_clauses:
                    target = self.clauses.extend(rng, target, context)
            plan = SentencePlan(
                subject, Predication(predicate.kind, predicate.label, fact.polarity, target)
            )
            if self._within_limit(plan):
                break
            mentions.drop_sentence(known)
        assert plan is not None
        stated.add(self._key(fact))
        mentions.end_sentence(instance)
        return _Item(DESCRIPTION, fact, plan)
