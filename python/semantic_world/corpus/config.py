"""Configuration for the corpus generator.

This module loads the YAML configuration described in ``docs/specs/CORPUS_GENERATOR.md``, fills
in every default, validates every field, resolves the level schedules to per-level values, and
writes the fully resolved configuration back out as YAML.

Validation follows the taxonomy generator's conventions, and uses its reader. Unknown keys are
errors. Every error is a :class:`ConfigError` whose message names the source file and the dotted
field path, for example ``data/corpus/x.yaml: documents.count: ...``.

The ``taxonomy`` setting names the world the corpus is about: a taxonomy configuration file with
a taxonomy seed, or a taxonomy output folder. Loading a corpus configuration also loads the
taxonomy's configuration, because the level schedules need the taxonomy's depth. The taxonomy
itself is generated later, by :func:`semantic_world.corpus.world.load_taxonomy`.
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
from semantic_world.taxonomy.config import Config as TaxonomyConfig
from semantic_world.taxonomy.config import load_config as load_taxonomy_config

CONCEPT_TYPES = (
    "category",
    "is",
    "has",
    "can",
    "verb",
    "verb_category",
    "patient_projection",
    "scalar",
)
"""The concept types whose share of named concepts is a parameter."""
DOCUMENT_TYPES = ("encyclopedic_category", "encyclopedic_feature", "entity", "situational")
NEGATION_LEVELS = ("class", "instance")
ALL_GROUNDINGS = ("fixed", "observed")
GENERIC_MEANINGS = ("all", "most", "some")
PARTICIPANT_WEIGHTS = ("thematic", "taxonomic", "constant")
CLAUSE_ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
SIDES = ("before", "after")
ADPOSITIONS = ("preposition", "postposition")
NEGATION_POSITIONS = ("after_auxiliary", "before_auxiliary")
REALIZATIONS = ("affix", "word")
VERB_MARKS = ("plural", "singular")
EVENT_TENSES = ("past", "present")
REFERENT_LABELS = ("local", "instance")
TEST_CHANGES = ("predicate", "subject", "quantifier", "role")
TAXONOMY_KINDS = ("config", "run")


# ---------------------------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxonomySource:
    """Where the taxonomy comes from."""

    kind: str
    """``config``: a taxonomy configuration file, generated in memory. ``run``: a taxonomy
    output folder, regenerated in memory and checked against its files."""
    path: str
    config: TaxonomyConfig
    """The taxonomy's configuration, with the taxonomy seed applied."""

    @property
    def seed(self) -> int:
        return self.config.seed

    @property
    def depth(self) -> int:
        return self.config.taxonomy.depth

    @property
    def verb_depth(self) -> int | None:
        """The depth of the verb tree, or None when the taxonomy has no verbs."""
        verbs = self.config.verbs
        return None if verbs is None else verbs.taxonomy.depth

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

    def resolved(self) -> dict[str, Any]:
        return {
            "named_proportion": dict(self.named_proportion),
            "synonym_rate": self.synonym_rate,
            "homonym_rate": self.homonym_rate,
            "homonym_same_pos": self.homonym_same_pos,
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

    def resolved(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mix": dict(self.mix),
            "sentences": {t: [r.min, r.max] for t, r in self.sentences.items()},
            "topic_level_weights": _list_schedule(self.topic_level_weights),
            "shuffle": self.shuffle,
            "instance_description_rate": self.instance_description_rate,
        }


@dataclass(frozen=True)
class PropositionsConfig:
    negation_rate: dict[str, float]
    """The rate of negative propositions at the class level and at the instance level."""
    rule_statement_rate: float
    rule_max_literals: int | None
    """The most literals in the term of a rule statement; a longer term is skipped and counted.
    None: no cap."""

    def resolved(self) -> dict[str, Any]:
        return {
            "negation_rate": dict(self.negation_rate),
            "rule_statement_rate": self.rule_statement_rate,
            "rule_statements": {"max_literals": self.rule_max_literals},
        }


@dataclass(frozen=True)
class QuantifiersConfig:
    all_grounding: str
    most_min_proportion: float
    some_exclude_all: bool
    generic_means: str
    generic_rate: float

    def resolved(self) -> dict[str, Any]:
        return {
            "all_grounding": self.all_grounding,
            "most": {"min_proportion": self.most_min_proportion},
            "some": {"exclude_all": self.some_exclude_all},
            "generic": {"means": self.generic_means},
            "generic_rate": self.generic_rate,
        }


@dataclass(frozen=True)
class SceneConfig:
    size: Range
    """The number of participants drawn beside the seed instance."""
    steps: Range
    events_per_step: float
    transitive_share: float
    verb_weights: dict[str, float] | None
    """None: uniform. Otherwise a weight for each named CAN feature or verb; a label that is
    left out has the weight 1."""
    participant_weights: dict[str, float]

    def resolved(self) -> dict[str, Any]:
        return {
            "size": [self.size.min, self.size.max],
            "steps": [self.steps.min, self.steps.max],
            "events_per_step": self.events_per_step,
            "transitive_share": self.transitive_share,
            "verb_weights": "uniform" if self.verb_weights is None else dict(self.verb_weights),
            "participant_weights": dict(self.participant_weights),
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
    verb_level_weights: tuple[float, ...]
    """The weight of every level of the verb tree, 1 to its depth, for the verb that names an
    event. Empty when the taxonomy has no verbs."""
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
            "verb_level_weights": _list_schedule(self.verb_level_weights),
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
    event_tense: str

    def resolved(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AspectConfig:
    enabled: bool
    realization: str
    position: str
    progressive_rate: float

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
    word_order: WordOrderConfig
    morphology: MorphologyConfig

    def resolved(self) -> dict[str, Any]:
        return {
            "adjective_order": {"fixed": self.adjective_order_fixed},
            "word_order": self.word_order.resolved(),
            "morphology": self.morphology.resolved(),
        }


@dataclass(frozen=True)
class TestSetsConfig:
    __test__ = False  # not a pytest class, whatever its name

    size: int
    changes: tuple[str, ...]

    def resolved(self) -> dict[str, Any]:
        return {"size": self.size, "changes": list(self.changes)}


@dataclass(frozen=True)
class Config:
    """A fully validated and resolved corpus configuration."""

    name: str
    seed: int
    taxonomy: TaxonomySource
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
    test_sets: TestSetsConfig
    source: str
    """Where the configuration came from: the file path, or a label for an in-memory mapping."""

    def resolved(self) -> dict[str, Any]:
        """The fully resolved configuration as a plain mapping, in the order of the
        specification. Every default is filled in, every schedule is written in its ``list``
        form with one value per level, and the taxonomy seed is written out. The mapping loads
        back through :func:`config_from_mapping` to an equal configuration."""
        return {
            "name": self.name,
            "seed": self.seed,
            "taxonomy": self.taxonomy.resolved(),
            "lexicon": self.lexicon.resolved(),
            "documents": self.documents.resolved(),
            "propositions": self.propositions.resolved(),
            "quantifiers": self.quantifiers.resolved(),
            "scene": self.scene.resolved(),
            "entity": {"scenes": [self.entity_scenes.min, self.entity_scenes.max]},
            "mention": self.mention.resolved(),
            "grammar": self.grammar.resolved(),
            "scalar_adjectives": {"z": self.scalar_z},
            "renderings": {"propositional": {"referents": self.propositional_referents}},
            "test_sets": self.test_sets.resolved(),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.resolved(), sort_keys=False, default_flow_style=None)


def _list_schedule(values) -> dict[str, Any]:
    return {"schedule": "list", "values": list(values)}


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def _read_taxonomy(root: _Node) -> TaxonomySource:
    if "taxonomy" not in root.data:
        raise root.error(
            "taxonomy",
            "is required: {config: <taxonomy configuration file>, seed: <taxonomy seed>} or "
            "{run: <taxonomy output folder>}",
        )
    node = root.mapping("taxonomy")
    config_path = node.string("config", None, nullable=True)
    run_path = node.string("run", None, nullable=True)
    seed = node.get("seed", None, nullable=True)
    node.finish()
    if (config_path is None) == (run_path is None):
        raise root.error("taxonomy", "give exactly one of config and run")
    if seed is not None:
        if run_path is not None:
            raise node.error(
                "seed", "goes with config only; an output folder is regenerated with its own seed"
            )
        node.check_int("seed", seed, min=0, max=SEED_MAX)
    if config_path is not None:
        if not Path(config_path).is_file():
            raise node.error("config", f"cannot read the taxonomy configuration {config_path}")
        return TaxonomySource("config", config_path, load_taxonomy_config(config_path, seed=seed))
    run_config = Path(run_path) / "config.yaml"
    if not run_config.is_file():
        raise node.error(
            "run", f"{run_path} is not a taxonomy output folder: it has no config.yaml"
        )
    return TaxonomySource("run", run_path, load_taxonomy_config(run_config))


def _probabilities(node: _Node, keys: tuple[str, ...], default: float) -> dict[str, float]:
    values = {key: node.probability(key, default) for key in keys}
    node.finish()
    return values


def _read_lexicon(node: _Node) -> LexiconConfig:
    config = LexiconConfig(
        named_proportion=_probabilities(node.mapping("named_proportion"), CONCEPT_TYPES, 1.0),
        synonym_rate=node.probability("synonym_rate", 0.0),
        homonym_rate=node.probability("homonym_rate", 0.0),
        homonym_same_pos=node.probability("homonym_same_pos", 0.5),
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
    config = DocumentsConfig(
        count=node.int("count", 10000, min=0),
        mix={t: float(mix.get(t, 0.0)) for t in DOCUMENT_TYPES},
        sentences=sentences,
        topic_level_weights=_level_weights(
            node, "topic_level_weights", {"schedule": "linear", "start": 1, "end": 2}, depth
        ),
        shuffle=node.probability("shuffle", 0.3),
        instance_description_rate=node.probability("instance_description_rate", 0.2),
    )
    node.finish()
    return config


def _read_propositions(node: _Node) -> PropositionsConfig:
    statements = node.mapping("rule_statements")
    max_literals = statements.get("max_literals", None, nullable=True)
    if max_literals is not None:
        statements.check_int("max_literals", max_literals, min=1)
    statements.finish()
    config = PropositionsConfig(
        negation_rate=_probabilities(node.mapping("negation_rate"), NEGATION_LEVELS, 0.1),
        rule_statement_rate=node.probability("rule_statement_rate", 0.3),
        rule_max_literals=max_literals,
    )
    node.finish()
    return config


def _read_quantifiers(node: _Node) -> QuantifiersConfig:
    most = node.mapping("most")
    min_proportion = most.number("min_proportion", 0.7, min=0, max=1)
    if min_proportion == 0:
        raise most.error("min_proportion", "must be more than 0")
    most.finish()
    some = node.mapping("some")
    exclude_all = some.bool("exclude_all", True)
    some.finish()
    generic = node.mapping("generic")
    means = generic.choice("means", "most", GENERIC_MEANINGS)
    generic.finish()
    config = QuantifiersConfig(
        all_grounding=node.choice("all_grounding", "fixed", ALL_GROUNDINGS),
        most_min_proportion=min_proportion,
        some_exclude_all=exclude_all,
        generic_means=means,
        generic_rate=node.probability("generic_rate", 0.5),
    )
    node.finish()
    return config


def _read_verb_weights(node: _Node) -> dict[str, float] | None:
    value = node.get("verb_weights", "uniform")
    if value == "uniform":
        return None
    if not isinstance(value, dict) or not value:
        raise node.error(
            "verb_weights",
            f"expected uniform or a mapping from CAN features and verbs to weights, found "
            f"{_describe(value)}",
        )
    inner = _Node(node.source, node.field("verb_weights"), value)
    weights: dict[str, float] = {}
    for label, weight in value.items():
        if not isinstance(label, str):
            raise inner.error(label, "the keys must be the labels of CAN features and verbs")
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
        verb_weights=_read_verb_weights(node),
        participant_weights={k: float(participants.get(k, 0.0)) for k in PARTICIPANT_WEIGHTS},
    )
    node.finish()
    return config


def _read_mention(node: _Node, depth: int, verb_depth: int | None) -> MentionConfig:
    default_weights = {"schedule": "linear", "start": 1, "end": 4}
    if verb_depth is None:
        node.get("verb_level_weights", default_weights)  # no verbs, so no levels to weigh
        verb_level_weights: tuple[float, ...] = ()
    else:
        verb_level_weights = _level_weights(node, "verb_level_weights", default_weights, verb_depth)
    clauses = node.mapping("relative_clauses")
    relative_clauses = RelativeClausesConfig(
        rate=clauses.probability("rate", 0.1),
        max_depth=clauses.int("max_depth", 1, min=0),
        object_share=clauses.probability("object_share", 0.3),
    )
    clauses.finish()
    config = MentionConfig(
        level_weights=_level_weights(node, "level_weights", default_weights, depth),
        verb_level_weights=verb_level_weights,
        pronoun_rate=node.probability("pronoun_rate", 0.5),
        modifier_rate=node.probability("modifier_rate", 0.3),
        max_adjectives=node.int("max_adjectives", 3, min=0),
        max_with_phrases=node.int("max_with_phrases", 2, min=0),
        max_content_words=node.int("max_content_words", 20, min=3),
        relative_clauses=relative_clauses,
    )
    node.finish()
    return config


def _inflection(node: _Node, key: str) -> _Node:
    """The settings of one inflection. The switch is ``enabled``. It was ``on`` in stage 1,
    which YAML reads as the boolean true when the key is written bare, so the error for the old
    key covers both readings."""
    part = node.mapping(key)
    if any(k is True or k == "on" for k in part.data):
        raise part.error("on", "is now enabled")
    return part


def _read_morphology(node: _Node) -> MorphologyConfig:
    number_node = _inflection(node, "number")
    number = NumberConfig(
        enabled=number_node.bool("enabled", False),
        realization=number_node.choice("realization", "affix", REALIZATIONS),
        position=number_node.choice("position", "after", SIDES),
        agreement=number_node.bool("agreement", True),
        verb_marks=number_node.choice("verb_marks", "plural", VERB_MARKS),
    )
    number_node.finish()
    tense_node = _inflection(node, "tense")
    tense = TenseConfig(
        enabled=tense_node.bool("enabled", False),
        realization=tense_node.choice("realization", "affix", REALIZATIONS),
        position=tense_node.choice("position", "after", SIDES),
        event_tense=tense_node.choice("event_tense", "past", EVENT_TENSES),
    )
    tense_node.finish()
    aspect_node = _inflection(node, "aspect")
    aspect = AspectConfig(
        enabled=aspect_node.bool("enabled", False),
        realization=aspect_node.choice("realization", "word", REALIZATIONS),
        position=aspect_node.choice("position", "after", SIDES),
        progressive_rate=aspect_node.probability("progressive_rate", 0.3),
    )
    aspect_node.finish()
    node.finish()
    return MorphologyConfig(number, tense, aspect)


def _read_grammar(node: _Node) -> GrammarConfig:
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
    )
    words.finish()
    config = GrammarConfig(fixed, word_order, _read_morphology(node.mapping("morphology")))
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
    config = TestSetsConfig(size=node.int("size", 500, min=0), changes=tuple(changes))
    node.finish()
    return config


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def config_from_mapping(data: Any, *, source: str = "<mapping>", seed: int | None = None) -> Config:
    """Build a configuration from an already parsed mapping.

    ``source`` names the origin in error messages. ``seed`` overrides the corpus's master seed
    (the command line's ``--seed``). The corpus seed is independent of the taxonomy's seed.
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
    taxonomy = _read_taxonomy(root)
    depth = taxonomy.depth
    lexicon = _read_lexicon(root.mapping("lexicon"))
    documents = _read_documents(root.mapping("documents"), depth)
    propositions = _read_propositions(root.mapping("propositions"))
    quantifiers = _read_quantifiers(root.mapping("quantifiers"))
    scene = _read_scene(root.mapping("scene"))
    entity = root.mapping("entity")
    entity_scenes = entity.range("scenes", [1, 3], min=1)
    entity.finish()
    mention = _read_mention(root.mapping("mention"), depth, taxonomy.verb_depth)
    grammar = _read_grammar(root.mapping("grammar"))
    scalar = root.mapping("scalar_adjectives")
    z = scalar.get("z", 1.0)
    if not _is_number(z) or z <= 0:
        raise scalar.error("z", f"expected a number above 0, found {_describe(z)}")
    scalar.finish()
    renderings = root.mapping("renderings")
    propositional = renderings.mapping("propositional")
    referents = propositional.choice("referents", "local", REFERENT_LABELS)
    propositional.finish()
    renderings.finish()
    test_sets = _read_test_sets(root.mapping("test_sets"))
    root.finish()
    return Config(
        name=name,
        seed=master_seed,
        taxonomy=taxonomy,
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
