"""The whole run: ``generate(config)`` and the ``TaxonomyResult`` it returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from semantic_world.taxonomy.analysis import feature_stats_table, similarity_table, summary_stats
from semantic_world.taxonomy.config import Config
from semantic_world.taxonomy.features import FeatureSet
from semantic_world.taxonomy.fixed import NodeVectors, compute_node_vectors
from semantic_world.taxonomy.instances import Instances, generate_instances
from semantic_world.taxonomy.rules import RuleSet, generate_rules
from semantic_world.taxonomy.streams import Streams
from semantic_world.taxonomy.tree import Tree, generate_tree
from semantic_world.taxonomy.verbs import VerbTaxonomy, generate_verb_tree


@dataclass(frozen=True)
class TaxonomyResult:
    """Everything one run produced. ``write`` writes the output folder."""

    config: Config
    stream_seeds: dict[str, int]
    rules: RuleSet
    tree: Tree
    instances: Instances
    vectors: NodeVectors
    similarity: pl.DataFrame
    feature_stats: pl.DataFrame
    summary: dict[str, Any]
    warnings: tuple[str, ...]
    verbs: VerbTaxonomy | None = None
    """The verb taxonomy, or None when ``verbs`` is null in the configuration."""

    @property
    def features(self) -> FeatureSet:
        return self.rules.features

    def frames(self) -> dict[str, pl.DataFrame]:
        """The CSV tables, keyed by file name."""
        from semantic_world.taxonomy.io import result_frames

        return result_frames(self)

    def write(self, path: str | Path | None = None) -> Path:
        """Write the output folder and return its path. The default is
        ``runs/taxonomy/<name>_seed<seed>/`` under the current directory."""
        from semantic_world.taxonomy.io import write_result

        return write_result(self, path)


def generate(config: Config) -> TaxonomyResult:
    """Run the generator: rules, tree, instances, node vectors, and statistics."""
    streams = Streams(config.seed)
    rules = generate_rules(config, streams)
    tree = generate_tree(config, rules, streams)
    instances = generate_instances(config, rules, tree, streams)
    vectors = compute_node_vectors(rules, tree, instances, config.scalars)
    similarity = similarity_table(config, tree, instances, vectors, streams.analysis)
    feature_stats = feature_stats_table(config, rules, tree, instances, vectors)
    verbs = generate_verb_tree(config, streams)
    warnings = tuple(rules.warnings) + tuple(tree.warnings)
    if verbs is not None:
        warnings += tuple(verbs.tree.warnings)
    summary = summary_stats(rules, tree, instances, warnings)
    return TaxonomyResult(
        config=config,
        stream_seeds=streams.seeds(),
        rules=rules,
        tree=tree,
        instances=instances,
        vectors=vectors,
        similarity=similarity,
        feature_stats=feature_stats,
        summary=summary,
        warnings=warnings,
        verbs=verbs,
    )
