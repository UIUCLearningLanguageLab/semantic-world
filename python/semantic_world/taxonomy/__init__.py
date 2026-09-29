"""The taxonomy feature generator.

The generator builds a tree of categories and a set of instances at the leaves, each with a
binary feature vector that follows controlled rules of inheritance and controlled logical rules
between features. The specification is ``docs/specs/TAXONOMY_GENERATOR.md``.

The generator is a standalone Python program. It does not use the Rust engine.
"""

from semantic_world.taxonomy.boolean import TruthTable
from semantic_world.taxonomy.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.expressions import Expr, ExpressionError, parse_expression
from semantic_world.taxonomy.features import Feature, FeatureSet, build_features
from semantic_world.taxonomy.rules import Rule, RuleSet, generate_rules
from semantic_world.taxonomy.streams import STREAM_NAMES, Streams, stream_seed

__all__ = [
    "STREAM_NAMES",
    "Config",
    "ConfigError",
    "Expr",
    "ExpressionError",
    "Feature",
    "FeatureSet",
    "GenerationError",
    "Rule",
    "RuleSet",
    "Streams",
    "TruthTable",
    "build_features",
    "config_from_mapping",
    "generate_rules",
    "load_config",
    "parse_expression",
    "stream_seed",
]
