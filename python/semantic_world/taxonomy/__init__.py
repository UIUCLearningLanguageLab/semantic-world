"""The taxonomy feature generator.

The generator builds a tree of categories and a set of instances at the leaves, each with a
binary feature vector that follows controlled rules of inheritance and controlled logical rules
between features. The specification is ``docs/specs/TAXONOMY_GENERATOR.md``.

The generator is a standalone Python program. It does not use the Rust engine.
"""

from semantic_world.taxonomy.boolean import TruthTable
from semantic_world.taxonomy.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.taxonomy.expressions import Expr, ExpressionError, parse_expression
from semantic_world.taxonomy.streams import STREAM_NAMES, Streams, stream_seed

__all__ = [
    "STREAM_NAMES",
    "Config",
    "ConfigError",
    "Expr",
    "ExpressionError",
    "Streams",
    "TruthTable",
    "config_from_mapping",
    "load_config",
    "parse_expression",
    "stream_seed",
]
