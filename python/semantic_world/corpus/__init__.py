"""The corpus generator.

The generator turns the taxonomy generator's world into a corpus of documents written in an
artificial language. The corpus is built in four layers: a lexicon, propositions that are checked
against the world, documents, and sentences. The specification is
``docs/specs/CORPUS_GENERATOR.md``.

Stage 1 builds the configuration, the loading of the taxonomy, the lexicon, and the formal
rendering. Stage 2 builds the class-level and instance-level propositions, their truth tests,
the facts and rule statements of a world, and the false items of the test sets. The generator is
pure Python and does not use the Rust engine.
"""

from semantic_world.corpus.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.lexicon import Concept, Lexeme, Lexicon, build_lexicon
from semantic_world.corpus.propositions import (
    CategoryTerm,
    Evaluation,
    Literal,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.renderings import formal
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
    "Facts",
    "Lexeme",
    "Lexicon",
    "Literal",
    "Predicate",
    "Proposition",
    "Streams",
    "Truth",
    "build_lexicon",
    "config_from_mapping",
    "falsify",
    "formal",
    "load_config",
    "load_taxonomy",
    "taxonomy_identity",
]
