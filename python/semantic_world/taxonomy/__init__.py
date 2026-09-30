"""The taxonomy feature generator.

The generator builds a tree of categories and a set of instances at the leaves, each with a
binary feature vector that follows controlled rules of inheritance and controlled logical rules
between features. The specification is ``docs/specs/TAXONOMY_GENERATOR.md``.

The generator is a standalone Python program. It does not use the Rust engine.
"""

from semantic_world.taxonomy.boolean import TruthTable
from semantic_world.taxonomy.config import Config, ConfigError, config_from_mapping, load_config
from semantic_world.taxonomy.constraints import (
    Constraint,
    Relation,
    Relations,
    generate_constraints,
)
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.expressions import Expr, ExpressionError, parse_expression
from semantic_world.taxonomy.features import Feature, FeatureSet, build_features
from semantic_world.taxonomy.fixed import NodeVectors, compute_node_vectors
from semantic_world.taxonomy.generate import TaxonomyResult, generate
from semantic_world.taxonomy.instances import Instances, generate_instances
from semantic_world.taxonomy.projections import Projections, compute_projections
from semantic_world.taxonomy.relation_stats import RelationStats, compute_relation_stats
from semantic_world.taxonomy.rules import Rule, RuleSet, generate_rules
from semantic_world.taxonomy.streams import STREAM_NAMES, Streams, stream_seed
from semantic_world.taxonomy.tree import Category, Role, Tree, generate_tree
from semantic_world.taxonomy.verbs import VerbTaxonomy, generate_verb_tree

__all__ = [
    "STREAM_NAMES",
    "Category",
    "Config",
    "ConfigError",
    "Constraint",
    "Expr",
    "ExpressionError",
    "Feature",
    "FeatureSet",
    "GenerationError",
    "Instances",
    "NodeVectors",
    "Projections",
    "Relation",
    "RelationStats",
    "Relations",
    "Role",
    "Rule",
    "RuleSet",
    "Streams",
    "TaxonomyResult",
    "Tree",
    "TruthTable",
    "VerbTaxonomy",
    "build_features",
    "compute_node_vectors",
    "compute_projections",
    "compute_relation_stats",
    "config_from_mapping",
    "generate",
    "generate_constraints",
    "generate_instances",
    "generate_rules",
    "generate_tree",
    "generate_verb_tree",
    "load_config",
    "parse_expression",
    "stream_seed",
]
