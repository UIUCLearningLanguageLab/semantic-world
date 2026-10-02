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
Stage 5 builds the documents: the planner of the four document types, the mentions of referents,
the readings of a sentence, the JSON logical form, and the propositional rendering. Stage 6
builds a whole run: the test sets, the statistics, the output folder, and the ``generate``
command. Stage 7 builds the request for the word-form pipeline and the ``render`` command, which
attaches the word forms.
The generator is pure Python and does not use the Rust engine.
"""

from semantic_world.corpus.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.facts import Facts
from semantic_world.corpus.generate import Corpus, generate
from semantic_world.corpus.grammar import (
    GrammarError,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
)
from semantic_world.corpus.interpret import interpret
from semantic_world.corpus.lexicon import Concept, Lexeme, Lexicon, build_lexicon
from semantic_world.corpus.logical import logical_form
from semantic_world.corpus.mentions import MentionRules, Mentions, RelativeClauses, plan_for
from semantic_world.corpus.planner import Document, DocumentSentence, Planner
from semantic_world.corpus.propositions import (
    CategoryTerm,
    Clause,
    Evaluation,
    Literal,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.readings import readings
from semantic_world.corpus.realize import Realizer, Sentence
from semantic_world.corpus.render import render
from semantic_world.corpus.renderings import (
    formal,
    parse_propositional,
    proposition_of,
    propositional,
)
from semantic_world.corpus.request import wordform_request
from semantic_world.corpus.scenes import Event, Scene, SceneGenerator
from semantic_world.corpus.stats import corpus_stats
from semantic_world.corpus.streams import STREAM_NAMES, Streams
from semantic_world.corpus.testsets import Item, ItemSet, build_test_sets, falsify
from semantic_world.corpus.world import load_taxonomy, taxonomy_identity

__all__ = [
    "STREAM_NAMES",
    "CategoryTerm",
    "Clause",
    "Concept",
    "Config",
    "ConfigError",
    "Corpus",
    "CorpusError",
    "Document",
    "DocumentSentence",
    "Evaluation",
    "Event",
    "Facts",
    "GrammarError",
    "Item",
    "ItemSet",
    "Lexeme",
    "Lexicon",
    "Literal",
    "MentionRules",
    "Mentions",
    "NounPhrase",
    "Planner",
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
    "build_test_sets",
    "config_from_mapping",
    "corpus_stats",
    "falsify",
    "formal",
    "generate",
    "interpret",
    "load_config",
    "load_taxonomy",
    "logical_form",
    "parse_propositional",
    "plan_for",
    "proposition_of",
    "propositional",
    "readings",
    "render",
    "taxonomy_identity",
    "wordform_request",
]
