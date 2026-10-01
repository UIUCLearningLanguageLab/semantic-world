"""The corpus generator.

The generator turns the taxonomy generator's world into a corpus of documents written in an
artificial language. The corpus is built in four layers: a lexicon, propositions that are checked
against the world, documents, and sentences. The specification is
``docs/specs/CORPUS_GENERATOR.md``.

Stage 1 builds the configuration, the loading of the taxonomy, the lexicon, and the formal
rendering. The generator is pure Python and does not use the Rust engine.
"""

from semantic_world.corpus.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.lexicon import Concept, Lexeme, Lexicon, build_lexicon
from semantic_world.corpus.renderings import formal
from semantic_world.corpus.streams import STREAM_NAMES, Streams
from semantic_world.corpus.world import load_taxonomy, taxonomy_identity

__all__ = [
    "STREAM_NAMES",
    "Concept",
    "Config",
    "ConfigError",
    "CorpusError",
    "Lexeme",
    "Lexicon",
    "Streams",
    "build_lexicon",
    "config_from_mapping",
    "formal",
    "load_config",
    "load_taxonomy",
    "taxonomy_identity",
]
