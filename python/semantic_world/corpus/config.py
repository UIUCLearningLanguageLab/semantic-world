"""Configuration for the corpus generator.

This module loads the YAML configuration described in ``docs/specs/CORPUS_GENERATOR.md`` and
``docs/specs/WORLD_AND_LANGUAGE.md`` ("Phase (a): the corpus"), fills in every default,
validates every field, resolves the level schedules to per-level values, and writes the fully
resolved configuration back out as YAML.

Validation follows the taxonomy generator's conventions, and uses its reader. Unknown keys are
errors. Every error is a :class:`ConfigError` whose message names the source file and the dotted
field path, for example ``data/corpus/x.yaml: documents.count: ...``. A key that the refactor
renamed gets an error that names its new key.

The ``world`` setting names the world the corpus is about: a world configuration file with a
seed, or a world run folder. Loading a corpus configuration also loads the world's configuration,
because the level schedules need the taxonomy's depth and the event-type tree's depth. The world
itself is generated later, by :func:`semantic_world.corpus.world.load_world`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_world.taxonomy.config import (
    SEED_MAX,
    ConfigError,
    Range,
    _describe,
    _is_number,
    _Node,  # the taxonomy generator's field-by-field reader, shared so the conventions match
    resolve_schedule,
)
from semantic_world.world.config import Config as WorldConfig
from semantic_world.world.config import load_config as load_world_config
from semantic_world.world.policies import DEFAULT_POLICY, policy_names

CONCEPT_TYPES = (
    "category",
    "property",
    "part",
    "state",
    "event_unary",
    "event",
    "event_category",
    "patient_projection",
    "scalar",
)
"""The concept types whose share of named concepts is a parameter."""
NAMED_PROPORTION_DEFAULTS = {**dict.fromkeys(CONCEPT_TYPES, 1.0), "patient_projection": 0.25}
RENAMED_CONCEPT_TYPES = {
    "is": "property",
    "has": "part",
    "can": "event_unary",
    "verb": "event",
    "verb_category": "event_category",
}
DOCUMENT_TYPES = ("encyclopedic_category", "encyclopedic_feature", "entity", "situational")
NEGATION_LEVELS = ("class", "instance", "state", "able_now")
"""The keys of ``propositions.negation_rate``: the class level, the instance level, the state
sentences of narratives (initial states), and the blocked sentences (``ABLE_NOW``)."""
NEGATION_DEFAULTS = {"class": 0.1, "instance": 0.1, "state": 0.1, "able_now": 0.8}
CAN_WORDS = ("shared", "distinct")
"""``lexicon.can_words``: with ``shared``, "can" expresses both ``ABLE`` and ``ABLE_NOW``; with
``distinct``, ``ABLE_NOW`` gets a function word of its own (``can_now``)."""
UNIVERSAL_WORDS = ("nec", "extensional", "either")
BARE_PLURAL_QUANTIFIERS = ("nec_all", "all", "most", "some")
"""The quantifiers a bare plural can be set to express. A negative bare plural expresses the
negative counterpart of each (``nec_no``, ``no``, ``most ... not``, ``some ... not``)."""
QUANTIFIER_WEIGHTS = {
    "nec_all": "nec_all",
    "all": "all",
    "most": "most",
    "some": "some",
    "none": "no",
    "nec_none": "nec_no",
}
"""The keys of ``quantifiers.weights``, and the quantifier that each one weighs. The key for
``no`` is ``none``, because YAML reads a bare ``no`` as the boolean false."""
PARTICIPANT_WEIGHTS = ("thematic", "taxonomic", "constant")
INITIAL_MODES = ("keep", "redraw")
CONTENT_KIND_WEIGHTS = ("proportional", "equal")
CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
SIDES = ("before", "after")
ADPOSITIONS = ("preposition", "postposition")
NEGATION_POSITIONS = ("after_auxiliary", "before_auxiliary")
BEFORE_POSITIONS = ("after_predicate", "before_predicate")
"""``grammar.word_order.before``: where the function word ``before`` of a precondition
statement stands, at the end of the verb phrase or at its start."""
DESCRIPTION_MODES = ("marked", "omitted")
"""``renderings.propositional.descriptions``: whether the propositional rendering writes the
descriptions of a sentence about instances, in braces before the assertion, or leaves them
out."""
REALIZATIONS = ("affix", "word")
VERB_MARKS = ("plural", "singular")
EVENT_TENSES = ("past", "present")
REFERENT_LABELS = ("local", "instance")
TEST_CHANGES = ("predicate", "subject", "quantifier", "polarity", "event", "role")
"""The changes a false test item can be made by. ``predicate``, ``subject``, ``quantifier``, and
``role`` are those of ``CORPUS_GENERATOR.md``; ``polarity`` and ``event`` are the changes of the
causal sets (``WORLD_AND_LANGUAGE.md``, "Test sets"), beside ``predicate`` and ``role``."""
WORLD_KINDS = ("config", "run")


# ---------------------------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldSource:
    """Where the world comes from."""

    kind: str
    """``config``: a world configuration file, generated in memory. ``run``: a world run
    folder, regenerated in memory and checked against its files."""
    path: str
    config: WorldConfig
    """The world's configuration, with the seed applied."""

    @property
    def seed(self) -> int:
        return self.config.seed

    @property
    def depth(self) -> int:
        """The depth of the taxonomy's category tree."""
        return self.config.taxonomy_config().taxonomy.depth

    @property
    def event_depth(self) -> int | None:
        """The depth of the event-type tree, or None when the world has no two-place event
        types."""
        binary = self.config.event_types.binary
        return None if binary is None else binary.taxonomy.depth

    def resolved(self) -> dict[str, Any]:
        if self.kind == "run":
            return {"run": self.path}
        return {"config": self.path, "seed": self.seed}


@dataclass(frozen=True)
class LexiconConfig:
    named_proportion: dict[str, float]
    """For each concept type, the proportion of concepts that get a word."""
    synonym_rate: float
    homonym_rate: float
    homonym_same_pos: float
    can_words: str
    """``shared``: "can" expresses ``ABLE`` and ``ABLE_NOW`` alike. ``distinct``: ``ABLE_NOW``
    has a function word of its own."""

    def resolved(self) -> dict[str, Any]:
        return {
            "named_proportion": dict(self.named_proportion),
            "synonym_rate": self.synonym_rate,
            "homonym_rate": self.homonym_rate,
            "homonym_same_pos": self.homonym_same_pos,
            "can_words": self.can_words,
        }


@dataclass(frozen=True)
class DocumentsConfig:
    count: int
    mix: dict[str, float]
    """The weight of every document type."""
    sentences: dict[str, Range]
    """The range of a document's length in sentences, for every document type."""
    topic_level_weights: tuple[float, ...]
    """The weight of every category level, 1 to the taxonomy's depth, for category topics."""
    shuffle: float
    instance_description_rate: float
    sibling_contrast_rate: float
    """The probability that a class-level fact in a category-topic document is followed by the
    matching fact about a sibling category."""
    relation_fact_share: float | None
    """The share of the sentences of a category-topic document that draw a relation fact. None:
    relation facts are one kind of content among the others, weighted like them. 0: a category
    document states no relation fact."""
    content_kind_weights: str
    """How a category document draws the kind of its next sentence: ``proportional`` to the
    number of facts of each kind it can still state, or ``equal``."""
    progressive_rate: float
    """The probability that a report of an event is progressive. Each report chooses its
    aspect; both aspects are true of any event that occurred."""
    one_aspect_per_event: bool
    """Whether every report of one event in one document uses the aspect of its first report."""
    initial_state_rate: float
    """The probability that the event sentence that first mentions a participant is followed by
    a sentence stating one of the participant's fluents at the time point before the event."""
    result_rate: float
    """The probability that an event sentence is followed by a sentence stating a change that
    the event's own effects made."""
    blocked_rate: float
    """The probability that an event sentence is followed by a sentence saying what a
    participant could, or could not, do at that time point (``ABLE_NOW``)."""

    def resolved(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mix": dict(self.mix),
            "sentences": {t: [r.min, r.max] for t, r in self.sentences.items()},
            "topic_level_weights": _list_schedule(self.topic_level_weights),
            "shuffle": self.shuffle,
            "instance_description_rate": self.instance_description_rate,
            "sibling_contrast_rate": self.sibling_contrast_rate,
            "relation_fact_share": self.relation_fact_share,
            "content_kind_weights": self.content_kind_weights,
            "progressive_rate": self.progressive_rate,
            "one_aspect_per_event": self.one_aspect_per_event,
            "initial_state_rate": self.initial_state_rate,
            "result_rate": self.result_rate,
            "blocked_rate": self.blocked_rate,
        }


@dataclass(frozen=True)
class PropositionsConfig:
    negation_rate: dict[str, float]
    """The rate of negative propositions at the class level, at the instance level, among the
    initial-state sentences (``state``), and among the blocked sentences (``able_now``)."""
    rule_statement_rate: float
    causal_statement_rate: float
    """The probability that a sentence of a feature document about an event type or a fluent is
    a causal statement."""
    rule_max_literals: int | None
    """The most literals in the term of a rule statement; a longer term is skipped and counted.
    None: no cap."""
    restriction_rate: float
    """The probability that the subject of a class-level fact in an encyclopedic document takes
    a restriction ("red penguins")."""
    event_tense: str
    """The tense of every event: ``past`` or ``present``. It is part of an event's logical
    form."""

    def resolved(self) -> dict[str, Any]:
        return {
            "negation_rate": dict(self.negation_rate),
            "rule_statement_rate": self.rule_statement_rate,
            "causal_statement_rate": self.causal_statement_rate,
            "rule_statements": {"max_literals": self.rule_max_literals},
            "restriction_rate": self.restriction_rate,
            "events": {"tense": self.event_tense},
        }


@dataclass(frozen=True)
class QuantifiersConfig:
    """The language's quantifier words. These settings decide what the words mean, never what a
    proposition means."""

    universal_words: str
    """What "all" and "no" express: ``nec`` (``NEC_ALL`` and ``NEC_NO``), ``extensional``
    (``ALL`` and ``NO``), or ``either``."""
    most_usage_min: float
    """A speaker says "most" only when at least this share has the predicate. ``MOST`` itself is
    true above one half."""
    some_exclude_all: bool
    bare_plural_expresses: tuple[str, ...]
    """The quantifiers a bare plural can express. Membership and rule statements may always use
    the bare plural, for ``NEC_ALL``."""
    generic_rate: float
    weights: dict[str, float]
    """How often a document states the facts of each quantifier, by quantifier. A fact is still
    stated with its strongest true quantifier. The weights change only which facts a document
    chooses. Equal weights change nothing. The configuration writes the key of ``no`` as ``none``
    and of ``nec_no`` as ``nec_none``."""

    @property
    def weighted(self) -> bool:
        """Whether the weights differ, so that the choice of facts is reweighted."""
        return len(set(self.weights.values())) > 1

    def resolved(self) -> dict[str, Any]:
        return {
            "universal_words": self.universal_words,
            "most": {"usage_min": self.most_usage_min},
            "some": {"exclude_all": self.some_exclude_all},
            "bare_plural": {"expresses": list(self.bare_plural_expresses)},
            "generic_rate": self.generic_rate,
            "weights": {key: self.weights[q] for key, q in QUANTIFIER_WEIGHTS.items()},
        }


@dataclass(frozen=True)
class SceneConfig:
    size: Range
    """The number of participants drawn beside the seed instance."""
    steps: Range
    events_per_step: float
    transitive_share: float
    event_type_weights: dict[str, float] | None
    """None: uniform. Otherwise a weight for each named event type; an event type that is left
    out has the weight 1."""
    participant_weights: dict[str, float]
    policy: str
    """The selection policy (``semantic_world.world.policies``)."""
    initial: str
    """``keep``: every scene starts from the entities' initial values. ``redraw``: each scene
    draws its participants' initial values afresh, at each fluent's initial rate."""

    def resolved(self) -> dict[str, Any]:
        return {
            "size": [self.size.min, self.size.max],
            "steps": [self.steps.min, self.steps.max],
            "events_per_step": self.events_per_step,
            "transitive_share": self.transitive_share,
            "event_type_weights": (
                "uniform" if self.event_type_weights is None else dict(self.event_type_weights)
            ),
            "participant_weights": dict(self.participant_weights),
            "policy": self.policy,
            "initial": self.initial,
        }


@dataclass(frozen=True)
class RelativeClausesConfig:
    rate: float
    max_depth: int
    object_share: float

    def resolved(self) -> dict[str, Any]:
        return {"rate": self.rate, "max_depth": self.max_depth, "object_share": self.object_share}


@dataclass(frozen=True)
class MentionConfig:
    """How referents are mentioned. Everything here can change what a sentence says, so it
    belongs to the planner and not to the grammar."""

    level_weights: tuple[float, ...]
    """The weight of every category level, 1 to the taxonomy's depth, for the noun of a
    mention."""
    event_level_weights: tuple[float, ...]
    """The weight of every level of the event-type tree, 1 to its depth, for the verb that names
    an event. Empty when the world has no two-place event types."""
    pronoun_rate: float
    modifier_rate: float
    max_adjectives: int
    max_with_phrases: int
    max_content_words: int
    """The most nouns, adjectives, and verbs in one sentence."""
    relative_clauses: RelativeClausesConfig

    def resolved(self) -> dict[str, Any]:
        return {
            "level_weights": _list_schedule(self.level_weights),
            "event_level_weights": _list_schedule(self.event_level_weights),
            "pronoun_rate": self.pronoun_rate,
            "modifier_rate": self.modifier_rate,
            "max_adjectives": self.max_adjectives,
            "max_with_phrases": self.max_with_phrases,
            "max_content_words": self.max_content_words,
            "relative_clauses": self.relative_clauses.resolved(),
        }


@dataclass(frozen=True)
class WordOrderConfig:
    clause: str
    determiner: str
    adjective: str
    with_phrase: str
    relative_clause: str
    adposition: str
    auxiliary: str
    negation: str
    before: str
    """Where the function word ``before`` of a precondition statement stands:
    ``after_predicate`` or ``before_predicate``."""

    def resolved(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class NumberConfig:
    enabled: bool
    realization: str
    position: str
    agreement: bool
    verb_marks: str

    def resolved(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TenseConfig:
    enabled: bool
    realization: str
    position: str

    def resolved(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AspectConfig:
    enabled: bool
    realization: str
    position: str

    def resolved(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MorphologyConfig:
    number: NumberConfig
    tense: TenseConfig
    aspect: AspectConfig

    @property
    def agreement(self) -> bool:
        """Whether verbs agree with their subjects: number is on, and agreement is not off."""
        return self.number.enabled and self.number.agreement

    def inflection_words(self) -> tuple[str, ...]:
        """The glosses of the inflections that are realized as separate function words."""
        marked = (
            ("PLURAL", self.number),
            ("PAST", self.tense),
            ("PROGRESSIVE", self.aspect),
        )
        return tuple(g for g, part in marked if part.enabled and part.realization == "word")

    def resolved(self) -> dict[str, Any]:
        return {
            "number": self.number.resolved(),
            "tense": self.tense.resolved(),
            "aspect": self.aspect.resolved(),
        }


@dataclass(frozen=True)
class GrammarConfig:
    """How a logical form is realized. Nothing here changes what a sentence says."""

    adjective_order_fixed: bool
    can_rate: dict[str, float]
    """The share of positive capacities that say ``can``, at the class level ("penguins can
    swim" and not "penguins swim") and at the instance level ("the penguin can swim" and not
    "the penguin swim"). Both forms say the same. A negative capacity always says ``can``."""
    word_order: WordOrderConfig
    morphology: MorphologyConfig

    def resolved(self) -> dict[str, Any]:
        return {
            "adjective_order": {"fixed": self.adjective_order_fixed},
            "can_rate": dict(self.can_rate),
            "word_order": self.word_order.resolved(),
            "morphology": self.morphology.resolved(),
        }


@dataclass(frozen=True)
class TestSetsConfig:
    __test__ = False  # not a pytest class, whatever its name

    size: int
    changes: tuple[str, ...]
    seen_descriptions: bool
    """Whether a description counts as stated, for the ``seen`` mark of a test item
    (``test_sets.seen.descriptions``)."""

    def resolved(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "changes": list(self.changes),
            "seen": {"descriptions": self.seen_descriptions},
        }


@dataclass(frozen=True)
class Config:
    """A fully validated and resolved corpus configuration."""

    name: str
    seed: int
    world: WorldSource
    lexicon: LexiconConfig
    documents: DocumentsConfig
    propositions: PropositionsConfig
    quantifiers: QuantifiersConfig
    scene: SceneConfig
    entity_scenes: Range
    mention: MentionConfig
    grammar: GrammarConfig
    scalar_z: float
    propositional_referents: str
    propositional_descriptions: str
    """``marked``: the propositional rendering writes the descriptions in braces before the
    assertion. ``omitted``: it holds the assertion only."""
    test_sets: TestSetsConfig
    source: str
    """Where the configuration came from: the file path, or a label for an in-memory mapping."""

    def resolved(self) -> dict[str, Any]:
        """The fully resolved configuration as a plain mapping, in the order of the
        specification. Every default is filled in, every schedule is written in its ``list``
        form with one value per level, and the world's seed is written out. The mapping loads
        back through :func:`config_from_mapping` to an equal configuration."""
        return {
            "name": self.name,
            "seed": self.seed,
            "world": self.world.resolved(),
            "lexicon": self.lexicon.resolved(),
            "documents": self.documents.resolved(),
            "propositions": self.propositions.resolved(),
            "quantifiers": self.quantifiers.resolved(),
            "scene": self.scene.resolved(),
            "entity": {"scenes": [self.entity_scenes.min, self.entity_scenes.max]},
            "mention": self.mention.resolved(),
            "grammar": self.grammar.resolved(),
            "scalar_adjectives": {"z": self.scalar_z},
            "renderings": {
                "propositional": {
                    "referents": self.propositional_referents,
                    "descriptions": self.propositional_descriptions,
                }
            },
            "test_sets": self.test_sets.resolved(),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.resolved(), sort_keys=False, default_flow_style=None)


def _list_schedule(values) -> dict[str, Any]:
    return {"schedule": "list", "values": list(values)}


def _renamed(node: _Node, old: str, new: str, why: str = "") -> None:
    """An error for a key that the world-and-language refactor renamed."""
    if old in node.data:
        raise node.error(old, f"is now {new}" + (f": {why}" if why else ""))


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def _read_world(root: _Node) -> WorldSource:
    if "taxonomy" in root.data:
        raise root.error(
            "taxonomy",
            "is now world: the corpus is about a world run, {config: <world configuration "
            "file>, seed: <seed>} or {run: <world run folder>}",
        )
    if "world" not in root.data:
        raise root.error(
            "world",
            "is required: {config: <world configuration file>, seed: <seed>} or "
            "{run: <world run folder>}",
        )
    node = root.mapping("world")
    config_path = node.string("config", None, nullable=True)
    run_path = node.string("run", None, nullable=True)
    seed = node.get("seed", None, nullable=True)
    node.finish()
    if (config_path is None) == (run_path is None):
        raise root.error("world", "give exactly one of config and run")
    if seed is not None:
        if run_path is not None:
            raise node.error(
                "seed", "goes with config only; a run folder is regenerated with its own seed"
            )
        node.check_int("seed", seed, min=0, max=SEED_MAX)
    if config_path is not None:
        if not Path(config_path).is_file():
            raise node.error("config", f"cannot read the world configuration {config_path}")
        return WorldSource("config", config_path, load_world_config(config_path, seed=seed))
    run_config = Path(run_path) / "config.yaml"
    if not run_config.is_file() or not (Path(run_path) / "definition.json").is_file():
        raise node.error(
            "run",
            f"{run_path} is not a world run folder: it has no config.yaml and definition.json",
        )
    return WorldSource("run", run_path, load_world_config(run_config))


def _read_lexicon(node: _Node) -> LexiconConfig:
    proportions = node.mapping("named_proportion")
    for old, new in RENAMED_CONCEPT_TYPES.items():
        _renamed(proportions, old, new)
    named = {
        key: proportions.probability(key, NAMED_PROPORTION_DEFAULTS[key]) for key in CONCEPT_TYPES
    }
    proportions.finish()
    config = LexiconConfig(
        named_proportion=named,
        synonym_rate=node.probability("synonym_rate", 0.0),
        homonym_rate=node.probability("homonym_rate", 0.0),
        homonym_same_pos=node.probability("homonym_same_pos", 0.5),
        can_words=node.choice("can_words", "shared", CAN_WORDS),
    )
    node.finish()
    return config


def _level_weights(node: _Node, key: str, default: Any, depth: int) -> tuple[float, ...]:
    """Non-negative weights over the category levels 1 to ``depth``, in any schedule form of the
    taxonomy generator, with at least one positive weight."""
    weights = resolve_schedule(node.get(key, default), depth, node.source, node.field(key))
    if any(w < 0 for w in weights):
        raise node.error(key, f"the weights must not be negative, found {list(weights)}")
    if not any(w > 0 for w in weights):
        raise node.error(key, "at least one weight must be positive")
    return weights


def _read_documents(node: _Node, depth: int) -> DocumentsConfig:
    mix = node.weights(
        "mix",
        {"encyclopedic_category": 0.3, "encyclopedic_feature": 0.2, "entity": 0.2,
         "situational": 0.3},
        allowed=DOCUMENT_TYPES,
    )  # fmt: skip
    sentences_node = node.mapping("sentences")
    defaults = {
        "encyclopedic_category": [5, 15],
        "encyclopedic_feature": [5, 15],
        "entity": [5, 20],
        "situational": [5, 20],
    }
    sentences = {t: sentences_node.range(t, defaults[t], min=1) for t in DOCUMENT_TYPES}
    sentences_node.finish()
    relation_fact_share = node.get("relation_fact_share", None, nullable=True)
    if relation_fact_share is not None:
        relation_fact_share = float(node.probability("relation_fact_share", None))
    config = DocumentsConfig(
        count=node.int("count", 10000, min=0),
        mix={t: float(mix.get(t, 0.0)) for t in DOCUMENT_TYPES},
        sentences=sentences,
        topic_level_weights=_level_weights(
            node, "topic_level_weights", {"schedule": "linear", "start": 1, "end": 2}, depth
        ),
        shuffle=node.probability("shuffle", 0.3),
        instance_description_rate=node.probability("instance_description_rate", 0.2),
        sibling_contrast_rate=node.probability("sibling_contrast_rate", 0.2),
        relation_fact_share=relation_fact_share,
        content_kind_weights=node.choice("content_kind_weights", "equal", CONTENT_KIND_WEIGHTS),
        progressive_rate=node.probability("progressive_rate", 0.3),
        one_aspect_per_event=node.bool("one_aspect_per_event", True),
        initial_state_rate=node.probability("initial_state_rate", 0.2),
        result_rate=node.probability("result_rate", 0.5),
        blocked_rate=node.probability("blocked_rate", 0.1),
    )
    node.finish()
    return config


def _read_propositions(node: _Node) -> PropositionsConfig:
    statements = node.mapping("rule_statements")
    max_literals = statements.get("max_literals", None, nullable=True)
    if max_literals is not None:
        statements.check_int("max_literals", max_literals, min=1)
    statements.finish()
    events = node.mapping("events")
    _renamed(
        events,
        "progressive_rate",
        "documents.progressive_rate",
        "events have no aspect; each report chooses its aspect (CG.64)",
    )
    event_tense = events.choice("tense", "past", EVENT_TENSES)
    events.finish()
    config = PropositionsConfig(
        negation_rate=_probabilities(node.mapping("negation_rate"), NEGATION_DEFAULTS),
        rule_statement_rate=node.probability("rule_statement_rate", 0.3),
        causal_statement_rate=node.probability("causal_statement_rate", 0.5),
        rule_max_literals=max_literals,
        restriction_rate=node.probability("restriction_rate", 0.1),
        event_tense=event_tense,
    )
    node.finish()
    return config


def _probabilities(node: _Node, defaults: dict[str, float]) -> dict[str, float]:
    values = {key: node.probability(key, default) for key, default in defaults.items()}
    node.finish()
    return values


def _read_quantifiers(node: _Node) -> QuantifiersConfig:
    _renamed(node, "all_grounding", "quantifiers.universal_words")
    universal = node.choice("universal_words", "nec", UNIVERSAL_WORDS)
    most = node.mapping("most")
    _renamed(most, "min_proportion", "quantifiers.most.usage_min")
    usage_min = most.number("usage_min", 0.7, min=0, max=1)
    if usage_min == 0:
        raise most.error("usage_min", "must be more than 0")
    most.finish()
    some = node.mapping("some")
    exclude_all = some.bool("exclude_all", True)
    some.finish()
    if "generic" in node.data:
        raise node.error("generic", "is now quantifiers.bare_plural.expresses")
    bare = node.mapping("bare_plural")
    expresses = bare.get("expresses", ["most"])
    if (
        not isinstance(expresses, list)
        or not expresses
        or any(q not in BARE_PLURAL_QUANTIFIERS for q in expresses)
        or len(set(expresses)) != len(expresses)
    ):
        raise bare.error(
            "expresses",
            f"expected a non-empty list of distinct quantifiers among "
            f"{', '.join(BARE_PLURAL_QUANTIFIERS)}, found {_describe(expresses)}",
        )
    bare.finish()
    weights_node = node.mapping("weights")
    for key in weights_node.data:
        if key is False or key == "no":
            raise weights_node.error(
                "no", "is written none: YAML reads a bare no as the boolean false"
            )
        if key == "nec_no":
            raise weights_node.error("nec_no", "is written nec_none, like the key none")
    weights = {
        quantifier: float(weights_node.number(key, 1.0, min=0))
        for key, quantifier in QUANTIFIER_WEIGHTS.items()
    }
    weights_node.finish()
    if not any(weight > 0 for weight in weights.values()):
        raise node.error("weights", "at least one weight must be positive")
    config = QuantifiersConfig(
        universal_words=universal,
        most_usage_min=usage_min,
        some_exclude_all=exclude_all,
        bare_plural_expresses=tuple(expresses),
        generic_rate=node.probability("generic_rate", 0.5),
        weights=weights,
    )
    node.finish()
    return config


def _read_event_type_weights(node: _Node) -> dict[str, float] | None:
    _renamed(node, "verb_weights", "scene.event_type_weights")
    value = node.get("event_type_weights", "uniform")
    if value == "uniform":
        return None
    if not isinstance(value, dict) or not value:
        raise node.error(
            "event_type_weights",
            f"expected uniform or a mapping from event types to weights, found {_describe(value)}",
        )
    inner = _Node(node.source, node.field("event_type_weights"), value)
    weights: dict[str, float] = {}
    for label, weight in value.items():
        if not isinstance(label, str):
            raise inner.error(label, "the keys must be the labels of event types")
        weights[label] = float(inner.check_number(label, weight, min=0))
    return weights


def _read_scene(node: _Node) -> SceneConfig:
    participants = node.weights(
        "participant_weights",
        {"thematic": 1.0, "taxonomic": 0.5, "constant": 0.1},
        allowed=PARTICIPANT_WEIGHTS,
    )
    config = SceneConfig(
        size=node.range("size", [2, 6], min=0),
        steps=node.range("steps", [3, 8], min=1),
        events_per_step=float(node.number("events_per_step", 1.5, min=0)),
        transitive_share=node.probability("transitive_share", 0.5),
        event_type_weights=_read_event_type_weights(node),
        participant_weights={k: float(participants.get(k, 0.0)) for k in PARTICIPANT_WEIGHTS},
        policy=node.choice("policy", DEFAULT_POLICY, policy_names()),
        initial=node.choice("initial", "keep", INITIAL_MODES),
    )
    node.finish()
    return config


def _read_mention(node: _Node, depth: int, event_depth: int | None) -> MentionConfig:
    _renamed(node, "verb_level_weights", "mention.event_level_weights")
    default_weights = {"schedule": "linear", "start": 1, "end": 4}
    if event_depth is None:
        node.get("event_level_weights", default_weights)  # no event tree, so no levels to weigh
        event_level_weights: tuple[float, ...] = ()
    else:
        event_level_weights = _level_weights(
            node, "event_level_weights", default_weights, event_depth
        )
    clauses = node.mapping("relative_clauses")
    relative_clauses = RelativeClausesConfig(
        rate=clauses.probability("rate", 0.1),
        max_depth=clauses.int("max_depth", 1, min=0),
        object_share=clauses.probability("object_share", 0.3),
    )
    clauses.finish()
    config = MentionConfig(
        level_weights=_level_weights(node, "level_weights", default_weights, depth),
        event_level_weights=event_level_weights,
        pronoun_rate=node.probability("pronoun_rate", 0.5),
        modifier_rate=node.probability("modifier_rate", 0.3),
        max_adjectives=node.int("max_adjectives", 3, min=0),
        max_with_phrases=node.int("max_with_phrases", 2, min=0),
        max_content_words=node.int("max_content_words", 20, min=3),
        relative_clauses=relative_clauses,
    )
    node.finish()
    return config


def _inflection(node: _Node, key: str, moved: dict[str, str] | None = None) -> _Node:
    """The settings of one inflection. The switch is ``enabled``. It was ``on`` in stage 1,
    which YAML reads as the boolean true when the key is written bare, so the error for the old
    key covers both readings. ``moved`` names the keys that now live elsewhere."""
    part = node.mapping(key)
    if any(k is True or k == "on" for k in part.data):
        raise part.error("on", "is now enabled")
    for old, new in (moved or {}).items():
        if old in part.data:
            raise part.error(
                old,
                f"is now {new}: an event's tense is part of its logical form, and a report's "
                f"aspect is drawn by the planner; the morphology only says whether and how they "
                f"are marked",
            )
    return part


def _read_morphology(node: _Node, event_tense: str) -> MorphologyConfig:
    number_node = _inflection(node, "number")
    number = NumberConfig(
        enabled=number_node.bool("enabled", False),
        realization=number_node.choice("realization", "affix", REALIZATIONS),
        position=number_node.choice("position", "after", SIDES),
        agreement=number_node.bool("agreement", True),
        verb_marks=number_node.choice("verb_marks", "plural", VERB_MARKS),
    )
    number_node.finish()
    tense_node = _inflection(node, "tense", {"event_tense": "propositions.events.tense"})
    tense = TenseConfig(
        enabled=tense_node.bool("enabled", False),
        realization=tense_node.choice("realization", "affix", REALIZATIONS),
        position=tense_node.choice("position", "after", SIDES),
    )
    tense_node.finish()
    aspect_node = _inflection(node, "aspect", {"progressive_rate": "documents.progressive_rate"})
    aspect = AspectConfig(
        enabled=aspect_node.bool("enabled", False),
        realization=aspect_node.choice("realization", "word", REALIZATIONS),
        position=aspect_node.choice("position", "after", SIDES),
    )
    aspect_node.finish()
    node.finish()
    if (
        tense.enabled
        and aspect.enabled
        and event_tense == "past"
        and tense.realization == aspect.realization == "affix"
    ):
        raise aspect_node.error(
            "realization",
            "a word takes one affix, and a past progressive verb would need two: make tense or "
            "aspect a word",
        )
    return MorphologyConfig(number, tense, aspect)


def _read_grammar(node: _Node, event_tense: str) -> GrammarConfig:
    if "class_can_rate" in node.data:
        raise node.error("class_can_rate", "is now grammar.can_rate.class")
    moved = {
        "max_adjectives": "mention.max_adjectives",
        "max_with_phrases": "mention.max_with_phrases",
        "max_sentence_tokens": "mention.max_content_words",
        "relative_clauses": "mention.relative_clauses",
    }
    for key, new in moved.items():
        if key in node.data:
            raise node.error(
                key,
                f"is now {new}: the planner decides what a sentence says, and a grammar setting "
                f"never changes a logical form",
            )
    order_node = node.mapping("adjective_order")
    fixed = order_node.bool("fixed", True)
    order_node.finish()
    words = node.mapping("word_order")
    word_order = WordOrderConfig(
        clause=words.choice("clause", "SVO", CLAUSE_ORDERS),
        determiner=words.choice("determiner", "before", SIDES),
        adjective=words.choice("adjective", "before", SIDES),
        with_phrase=words.choice("with_phrase", "after", SIDES),
        relative_clause=words.choice("relative_clause", "after", SIDES),
        adposition=words.choice("adposition", "preposition", ADPOSITIONS),
        auxiliary=words.choice("auxiliary", "before", SIDES),
        negation=words.choice("negation", "after_auxiliary", NEGATION_POSITIONS),
        before=words.choice("before", "after_predicate", BEFORE_POSITIONS),
    )
    words.finish()
    can_node = node.mapping("can_rate")
    can_rate = {
        "class": can_node.probability("class", 0.5),
        "instance": can_node.probability("instance", 1.0),
    }
    can_node.finish()
    config = GrammarConfig(
        adjective_order_fixed=fixed,
        can_rate=can_rate,
        word_order=word_order,
        morphology=_read_morphology(node.mapping("morphology"), event_tense),
    )
    node.finish()
    return config


def _read_test_sets(node: _Node) -> TestSetsConfig:
    changes = node.get("changes", list(TEST_CHANGES))
    if not isinstance(changes, list):
        raise node.error("changes", f"expected a list, found {_describe(changes)}")
    for change in changes:
        if change not in TEST_CHANGES:
            raise node.error(
                "changes",
                f"expected a list of {', '.join(TEST_CHANGES)}, found {_describe(change)}",
            )
    if len(set(changes)) != len(changes):
        raise node.error("changes", "the changes must be distinct")
    seen = node.mapping("seen")
    seen_descriptions = seen.bool("descriptions", True)
    seen.finish()
    config = TestSetsConfig(
        size=node.int("size", 500, min=0),
        changes=tuple(changes),
        seen_descriptions=seen_descriptions,
    )
    node.finish()
    return config


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def config_from_mapping(data: Any, *, source: str = "<mapping>", seed: int | None = None) -> Config:
    """Build a configuration from an already parsed mapping.

    ``source`` names the origin in error messages. ``seed`` overrides the corpus's master seed
    (the command line's ``--seed``). The corpus seed is independent of the world's seed.
    """
    if data is None:
        data = {}
    root = _Node(source, "", data)
    name = root.string("name", "default")
    master_seed = root.int("seed", 1, min=0, max=SEED_MAX)
    if seed is not None:
        master_seed = root.check_int("seed", seed, min=0, max=SEED_MAX)
    # A run's config.yaml records its provenance under this key, so that a run folder's
    # configuration loads back unchanged.
    provenance = root.get("provenance", None, nullable=True)
    if provenance is not None and not isinstance(provenance, dict):
        raise root.error("provenance", f"expected a mapping, found {_describe(provenance)}")
    world = _read_world(root)
    depth = world.depth
    lexicon = _read_lexicon(root.mapping("lexicon"))
    documents = _read_documents(root.mapping("documents"), depth)
    propositions = _read_propositions(root.mapping("propositions"))
    quantifiers = _read_quantifiers(root.mapping("quantifiers"))
    scene = _read_scene(root.mapping("scene"))
    entity = root.mapping("entity")
    entity_scenes = entity.range("scenes", [2, 5], min=1)
    entity.finish()
    mention = _read_mention(root.mapping("mention"), depth, world.event_depth)
    grammar = _read_grammar(root.mapping("grammar"), propositions.event_tense)
    scalar = root.mapping("scalar_adjectives")
    z = scalar.get("z", 1.0)
    if not _is_number(z) or z <= 0:
        raise scalar.error("z", f"expected a number above 0, found {_describe(z)}")
    scalar.finish()
    renderings = root.mapping("renderings")
    propositional = renderings.mapping("propositional")
    referents = propositional.choice("referents", "local", REFERENT_LABELS)
    descriptions = propositional.choice("descriptions", "marked", DESCRIPTION_MODES)
    propositional.finish()
    renderings.finish()
    test_sets = _read_test_sets(root.mapping("test_sets"))
    root.finish()
    return Config(
        name=name,
        seed=master_seed,
        world=world,
        lexicon=lexicon,
        documents=documents,
        propositions=propositions,
        quantifiers=quantifiers,
        scene=scene,
        entity_scenes=entity_scenes,
        mention=mention,
        grammar=grammar,
        scalar_z=float(z),
        propositional_referents=referents,
        propositional_descriptions=descriptions,
        test_sets=test_sets,
        source=source,
    )


def load_config(path: str | Path, *, seed: int | None = None) -> Config:
    """Load, validate, and resolve a YAML configuration file. Relative paths inside the file
    are read from the current folder."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(str(path), "<file>", f"cannot read the file: {error.strerror}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from error
    return config_from_mapping(data, source=str(path), seed=seed)
