"""The corpus generator.

The generator turns the taxonomy generator's world into a corpus of documents written in an
artificial language. The corpus is built in four layers: a lexicon, propositions that are checked
against the world, documents, and sentences. The specification is
``docs/specs/CORPUS_GENERATOR.md``.

Stage 1 builds the configuration, the loading of the taxonomy, the lexicon, and the formal
rendering. Stage 2 builds the class-level and instance-level propositions, their truth tests,
the facts and rule statements of a world, and the false items of the test sets. Stage 3 builds
scenes, events, and event-level propositions. Stage 4 builds the grammar: sentence plans, their
realization as words and trees, the reading of a tree back into its plan, and relative clauses.
The generator is pure Python and does not use the Rust engine.
"""

from semantic_world.corpus.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.grammar import (
    GrammarError,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
)
from semantic_world.corpus.interpret import interpret
from semantic_world.corpus.lexicon import Concept, Lexeme, Lexicon, build_lexicon
from semantic_world.corpus.mentions import RelativeClauses, plan_for
from semantic_world.corpus.propositions import (
    CategoryTerm,
    Evaluation,
    Literal,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.realize import Realizer, Sentence
from semantic_world.corpus.renderings import formal
from semantic_world.corpus.scenes import Event, Scene, SceneGenerator
from semantic_world.corpus.streams import STREAM_NAMES, Streams
from semantic_world.corpus.testsets import falsify
from semantic_world.corpus.world import load_taxonomy, taxonomy_identity

__all__ = [
    "STREAM_NAMES",
    "CategoryTerm",
    "Concept",
    "Config",
    "ConfigError",
    "CorpusError",
    "Evaluation",
    "Event",
    "Facts",
    "GrammarError",
    "Lexeme",
    "Lexicon",
    "Literal",
    "NounPhrase",
    "Predicate",
    "Predication",
    "Proposition",
    "Realizer",
    "RelativeClause",
    "RelativeClauses",
    "Scene",
    "SceneGenerator",
    "Sentence",
    "SentencePlan",
    "Streams",
    "Truth",
    "build_lexicon",
    "config_from_mapping",
    "falsify",
    "formal",
    "interpret",
    "load_config",
    "load_taxonomy",
    "plan_for",
    "taxonomy_identity",
]
