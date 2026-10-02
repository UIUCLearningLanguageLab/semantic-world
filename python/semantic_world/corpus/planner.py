"""Layer 3: documents. The discourse planner chooses and orders propositions by document type,
and decides how each referent is mentioned.

There are four document types (``documents.mix``):

- **encyclopedic, about a category.** The topic is a category. Each sentence draws a kind of
  content, each kind with the same chance: a membership fact, a fact about the topic, a fact
  about one of its subcategories, or a relation fact with the topic as agent or as patient.
  With ``documents.relation_fact_share``, a sentence draws a relation fact with that
  probability instead, and the other kinds share the rest. A fact about the topic can be
  followed by the matching fact about a sibling category, at
  ``documents.sibling_contrast_rate``, when the sibling differs;
- **encyclopedic, about a feature.** The topic is an IS, HAS, or CAN feature, or a verb. A
  sentence states a rule, at ``propositions.rule_statement_rate``, or says which categories have
  the feature and which lack it;
- **entity narrative.** The topic is one instance. The document narrates the events of
  ``entity.scenes`` scenes that involve the instance, in time order, scene by scene;
- **situational narrative.** One scene. The document narrates its events in time order.

In a narrative, an event sentence is followed by an instance-level description at
``documents.instance_description_rate``: of the topic in an entity narrative, and of a
participant of the event in a situational narrative, a newly introduced one first.

**Restricted subjects.** The subject of a class-level fact takes a restriction ("red penguins")
at ``propositions.restriction_rate``, and a restrictive relative clause ("penguins that can
swim") at ``mention.relative_clauses.rate``. Both are drawn before the fact, so the fact is true
of the restricted subject.

**Ordering.** An encyclopedic document follows a template: membership, defining facts (``all``
and ``no``), characteristic facts (``most``, and scalar poles), rarer facts (``some``), relation
facts, and rule statements. ``documents.shuffle`` moves from the template (0) to a random order
(1). A sibling contrast stays right after its fact. A narrative follows time.

**Quantifier weights.** A fact is stated with its strongest true quantifier. With
``quantifiers.weights``, a document chooses a fact with a probability proportional to the weight
of that quantifier, so a study can rebalance the mix. Equal weights, the default, change nothing.

**Streams.** Each document draws from its own parts of four streams, named by its label: the
document's type, topic, length, and order from ``corpus:documents``; its facts from
``corpus:propositions``; its mentions from ``corpus:mentions``; and the grammar's choices from
``corpus:grammar``. So a grammar setting never changes what a document says, or in what order.
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
    ALL,
    CAN,
    CLASS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NO,
    SCALAR,
    VERB,
    CategoryTerm,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, Sentence, as_json
from semantic_world.corpus.renderings import propositional
from semantic_world.corpus.scenes import Event, Scene, SceneGenerator
from semantic_world.corpus.streams import Streams
from semantic_world.corpus.world import load_taxonomy
from semantic_world.taxonomy.generate import TaxonomyResult

CATEGORY, FEATURE, ENTITY, SITUATIONAL = DOCUMENT_TYPES

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

_STRONG = (ALL, NO, MOST)
_MAX_REDRAWS = 100


@dataclass(frozen=True)
class DocumentSentence:
    """One sentence of a document, with everything that is kept about it."""

    label: str
    """``D.<n>.<k>``."""
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
    """For each noun phrase, in the order of the tree: the referent it mentions (``R.<n>``), or
    None for a noun phrase that names a category."""
    distinguished: tuple[bool | None, ...]
    """For each noun phrase, in the same order: for a definite mention with a noun, whether it
    picks out its referent alone among the participants of the document's scenes."""
    strength: str | None = None
    """For a class-level sentence: the strongest true quantifier of its fact, which a bare
    generic can stand for (``all``, ``most``, ``some``, or ``no``), or ``pole`` for a scalar
    pole, which takes no quantifier word. It is kept for the statistics, and is not written to
    ``documents.jsonl``."""

    @property
    def plan(self) -> SentencePlan:
        return self.sentence.plan

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
    """``D.<n>``."""
    type: str
    topic: str
    """A category, a feature or a verb, an instance, or, for a situational narrative, its
    scene."""
    scenes: tuple[Scene, ...]
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
    """The strongest true quantifier of a class-level fact, before the bare generic replaces
    it."""


@dataclass
class _Draft:
    type: str
    topic: str
    items: list[_Item]
    scenes: tuple[Scene, ...] = ()
    mentions: Mentions | None = None
    length: int = 0
    """The drawn length."""


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
    """How often sentences are ambiguous: the number of sentences, the number with more than one
    reading, and the count of every set of readings."""
    sets: dict[str, int] = {}
    total = ambiguous = 0
    for document in documents:
        for sentence in document.sentences:
            total += 1
            ambiguous += len(sentence.readings) > 1
            key = "+".join(sentence.readings)
            sets[key] = sets.get(key, 0) + 1
    return {
        "sentences": total,
        "ambiguous": ambiguous,
        "ambiguous_share": ambiguous / total if total else 0.0,
        "readings": dict(sorted(sets.items())),
    }


class Planner:
    """The documents of one corpus. Documents are made in order, because scenes and propositions
    are numbered across the corpus."""

    def __init__(
        self,
        config: Config,
        result: TaxonomyResult | None = None,
        *,
        lexicon: Lexicon | None = None,
    ) -> None:
        self.config = config
        self.result = load_taxonomy(config) if result is None else result
        self.streams = Streams(config.seed)
        self.lexicon = lexicon or build_lexicon(config, self.result, self.streams)
        self.truth = Truth(config, self.result)
        self.facts = Facts(config, self.result, self.lexicon, self.truth)
        self.scene_generator = SceneGenerator(config, self.result, self.truth)
        self.realizer = Realizer(config, self.lexicon, self.streams)
        self.clauses = RelativeClauses(config, self.facts)
        self.rules = MentionRules(config, self.facts, self.streams)
        self.scenes: list[Scene] = []
        """Every scene of the documents made so far, in order."""
        self._proposition_ids: dict[Proposition, str] = {}
        self._made = 0
        facts = self.facts
        self._instances = self.result.instances.labels
        self._by_level: dict[int, list[str]] = {}
        for category in facts.categories:
            self._by_level.setdefault(facts.level[category], []).append(category)
        self._children: dict[str, list[str]] = {c: [] for c in facts.categories}
        self._siblings: dict[str, list[str]] = {}
        parents: dict[str | None, list[str]] = {}
        for category in self.result.tree.categories:
            if category.label not in facts.named:
                continue
            parent = None if category.parent is None else category.parent.label
            parents.setdefault(parent, []).append(category.label)
            if parent in self._children:
                self._children[parent].append(category.label)
        for group in parents.values():
            for category in group:
                self._siblings[category] = [c for c in group if c != category]
        self._feature_topics = (
            facts.features[IS] + facts.features[HAS] + facts.features[CAN] + facts.verbs
        )

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
        label = f"D.{number}"
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
            plan = item.plan or plan_for(self.facts, item.proposition)
            proposition = item.proposition
            identifier = self._proposition_ids.setdefault(
                proposition, f"PR.{len(self._proposition_ids) + 1}"
            )
            proposition = dataclasses.replace(proposition, id=identifier)
            sentence = self.realizer.realize(plan, rng)
            form = logical_form(plan, proposition, referents)
            sentences.append(
                DocumentSentence(
                    label=f"{label}.{number}",
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
        return DEFINING if fact.quantifier in (ALL, NO) else RARER

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

    def _as_generic(self, rng: np.random.Generator, fact: Proposition) -> Proposition:
        """The fact, or the bare generic that says the same, at the generic rate."""
        if rng.random() < self.config.quantifiers.generic_rate:
            return self.facts.generic(fact) or fact
        return fact

    def _stated(
        self,
        rng: np.random.Generator,
        fact: Proposition,
        section: str | None = None,
        follows: bool = False,
    ) -> _Item:
        """The planned sentence of a class-level fact: the fact, or its bare generic."""
        strength = POLE if fact.predicate.kind == SCALAR else fact.quantifier
        return _Item(
            section or self._section(fact), self._as_generic(rng, fact), None, follows, strength
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
        return self.config.quantifiers.weights.get(str(fact.quantifier), 1.0)

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
        membership."""
        return [
            predicate
            for predicate in self.facts.class_predicates(term.category, patients=())
            if predicate.kind != MEMBER
        ]

    def _membership_fact(
        self, rng: np.random.Generator, topic: str, negative: bool, stated: set
    ) -> Proposition | None:
        facts = self.facts
        above = [
            (CategoryTerm(topic), Predicate(MEMBER, a))
            for a in self.truth.ancestors(topic)
            if a in facts.named
        ]
        below = [(CategoryTerm(c), Predicate(MEMBER, topic)) for c in self._children[topic]]
        apart = [
            (CategoryTerm(topic), Predicate(MEMBER, c))
            for c in facts.categories
            if facts.level[c] <= facts.level[topic] and self.truth.disjoint(topic, c)
        ]
        # a positive fact names a category above, and a negative one a category apart
        return self._first_fact(rng, apart if negative else above + below, negative, stated)

    def _relation_fact(
        self, rng: np.random.Generator, topic: CategoryTerm, negative: bool, stated: set
    ) -> Proposition | None:
        """A relation fact with the topic as agent or as patient, each with the same chance."""
        facts = self.facts
        as_agent = bool(rng.random() < 0.5)
        pairs = [(verb, other) for verb in facts.verbs for other in facts.categories]
        for agent_role in (as_agent, not as_agent):
            if agent_role:
                candidates = [
                    (topic, Predicate(VERB, verb, CategoryTerm(other))) for verb, other in pairs
                ]
            else:
                candidates = [
                    (CategoryTerm(other), Predicate(VERB, verb, topic)) for verb, other in pairs
                ]
            fact = self._first_fact(rng, candidates, negative, stated)
            if fact is not None:
                return fact
        return None

    def _contrast(
        self, rng: np.random.Generator, topic: str, fact: Proposition, stated: set
    ) -> Proposition | None:
        """The matching fact about a sibling of the topic: the same predicate, with the opposite
        polarity. A fact has a contrast when it is a strong one (``all``, ``no``, ``most``, or a
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
        share = self.config.documents.relation_fact_share
        others = [k for k in kinds if k != RELATION]
        items: list[_Item] = []
        stated: set = set()
        for _ in range(4 * length + 20):
            if len(items) >= length:
                break
            if share is None:
                content = kinds[int(facts_rng.integers(len(kinds)))]
            elif RELATION in kinds and facts_rng.random() < share:
                content = RELATION
            else:
                # the other kinds share the rest, each with the same chance
                content = others[int(facts_rng.integers(len(others)))]
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
    ) -> Proposition | None:
        """One true class-level proposition, drawn the way an encyclopedic document draws a
        sentence, for the test sets. The topic is a category, drawn by
        ``documents.topic_level_weights``. The kind of content is drawn among ``kinds``, each
        with the same chance: a membership fact, a fact about the topic, a fact about a
        subcategory, a relation fact, or a rule statement. The polarity, the restriction, and
        the bare generic are drawn at their rates. None when the draw gives no fact."""
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
        return None if fact is None else self._as_generic(rng, fact)

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
        topic = seed if kind == ENTITY else scenes[0].label
        cast = tuple(dict.fromkeys(p for scene in scenes for p in scene.participants))
        mentions = Mentions(self.rules, cast, modifiers=kind == ENTITY)
        length = self._length(kind, rng)
        rate = self.config.documents.instance_description_rate
        items: list[_Item] = []
        stated: set = set()
        for scene in scenes:
            events = scene.involving(seed) if kind == ENTITY else scene.events
            for event in events:
                if len(items) >= length:
                    break
                report = self.facts.draw_event(mention_rng, event)
                if report is None:
                    continue  # no word can report the event
                known = set(mentions.referents)
                items.append(self._event_sentence(mention_rng, report, event, scene, mentions))
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
        event: Event,
        scene: Scene,
        mentions: Mentions,
    ) -> _Item:
        """The sentence that reports an event. A relative clause reports an earlier event of
        the scene."""
        earlier = scene.events[: scene.events.index(event)]
        known = dict(mentions.referents)
        plan: SentencePlan | None = None
        for with_clauses in (True, False):
            context = ClauseContext(
                EVENT,
                events=earlier,
                mention=lambda instance: mentions.noun_phrase(rng, instance, pronoun=False),
                used={("event", event.label), happened(event)},
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
        with the other participants of the document's scenes as the patients of its verbs."""
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
