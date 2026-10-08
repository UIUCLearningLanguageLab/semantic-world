"""A whole world run: ``define(config)`` and the ``WorldResult`` it returns, with ``write``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.taxonomy.io import STATIC_FEATURES_FILE, provenance
from semantic_world.taxonomy.io import static_features_frame as taxonomy_static_features
from semantic_world.taxonomy.streams import Streams as TaxonomyStreams
from semantic_world.world.capacities import (
    CAPACITIES_FILE,
    CAPACITY_ROLES_FILE,
    capacities_frame,
    capacity_roles_frame,
)
from semantic_world.world.config import Config
from semantic_world.world.definition import (
    Definition,
    DefinitionReport,
    build_definition,
    check_definition,
    write_definition,
)
from semantic_world.world.derived import DERIVED_DIR, write_manifest
from semantic_world.world.errors import WorldError
from semantic_world.world.event_file import EventFile, load_event_file
from semantic_world.world.event_types import EventTypes, check_dynamics, generate_event_types
from semantic_world.world.fluents import Fluents, derived_initial_values, generate_fluents
from semantic_world.world.labels import translate, translate_column
from semantic_world.world.relation_stats import relation_frames
from semantic_world.world.requirements import apply_requirements
from semantic_world.world.stats import world_stats
from semantic_world.world.streams import WorldStreams

TAXONOMY_DIR = "taxonomy"
STATS_FILE = "world_stats.yaml"


@dataclass(frozen=True)
class WorldResult:
    config: Config
    taxonomy: TaxonomyResult
    """The taxonomy run the world is built on, with explicit requirements applied."""
    taxonomy_run: TaxonomyResult
    """The unmodified taxonomy run, written to ``taxonomy/``."""
    event_file: EventFile | None
    fluents: Fluents
    event_types: EventTypes
    definition: Definition
    report: DefinitionReport
    stream_seeds: dict[str, int]
    stats: dict[str, Any]
    warnings: tuple[str, ...]

    @property
    def rule_set_id(self) -> str:
        return self.definition.rule_set_id

    def derived_frames(self) -> dict[str, pl.DataFrame]:
        """Every table of ``derived/``, keyed by file name."""
        static = taxonomy_static_features(self.taxonomy)
        static.columns = [translate_column(c) for c in static.columns]
        static = static.with_columns(
            pl.Series("label", [translate(v) for v in static["label"].to_list()], dtype=pl.Utf8)
        )
        frames = {
            STATIC_FEATURES_FILE: static,
            CAPACITIES_FILE: capacities_frame(self.taxonomy),
            CAPACITY_ROLES_FILE: capacity_roles_frame(self.taxonomy),
        }
        frames.update(relation_frames(self.taxonomy))
        return frames

    def write(self, path: str | Path | None = None) -> Path:
        return write_result(self, path)


def default_output_dir(config: Config, base: str | Path = "runs/world") -> Path:
    return Path(base) / f"{config.name}_seed{config.seed}"


def define(config: Config) -> WorldResult:
    """Generate a world: the taxonomy run, the fluents, the event types, the definition, and the
    agreement test. A disagreement, or a broken invariant of the dynamics, raises
    :class:`WorldError` and fails the run."""
    taxonomy_run = generate_taxonomy(config.taxonomy_config())
    event_file = None
    path = config.event_file_path()
    if path is not None:
        event_file = load_event_file(path)
    taxonomy = (
        apply_requirements(taxonomy_run, event_file) if event_file is not None else taxonomy_run
    )
    streams = WorldStreams(config.seed)
    fluents = generate_fluents(
        config,
        taxonomy,
        streams,
        explicit_rates=event_file.fluents if event_file is not None else None,
        explicit_source=event_file.source if event_file is not None else None,
    )
    derived_initial = derived_initial_values(fluents, taxonomy)
    event_types = generate_event_types(
        taxonomy, fluents, config, streams, event_file, derived_initial
    )
    initial_present: set[tuple[str, bool]] = set()
    for i, fluent in enumerate(fluents.base):
        for value in np.unique(fluents.initial_values[:, i]):
            initial_present.add((fluent.label, bool(value)))
    for label, column in derived_initial.items():
        for value in np.unique(column):
            initial_present.add((label, bool(value)))
    try:
        check_dynamics(event_types, fluents, initial_present)
    except ValueError as error:
        raise WorldError(f"the event types break a rule of the dynamics: {error}") from None
    definition = build_definition(taxonomy, fluents, event_types)
    report = check_definition(definition)
    stats = world_stats(fluents, event_types)
    seeds = {**TaxonomyStreams(taxonomy.config.seed).seeds(), **streams.seeds()}
    return WorldResult(
        config=config,
        taxonomy=taxonomy,
        taxonomy_run=taxonomy_run,
        event_file=event_file,
        fluents=fluents,
        event_types=event_types,
        definition=definition,
        report=report,
        stream_seeds=seeds,
        stats=stats,
        warnings=tuple(taxonomy.warnings),
    )


def _yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True)


def write_result(result: WorldResult, path: str | Path | None = None) -> Path:
    folder = Path(path) if path is not None else default_output_dir(result.config)
    folder.mkdir(parents=True, exist_ok=True)
    config_data = result.config.resolved()
    config_data["provenance"] = provenance(result.stream_seeds)
    (folder / "config.yaml").write_text(_yaml(config_data), encoding="utf-8")
    result.taxonomy_run.write(folder / TAXONOMY_DIR)
    write_definition(result.definition, folder)
    derived = folder / DERIVED_DIR
    derived.mkdir(exist_ok=True)
    frames = result.derived_frames()
    for name, frame in frames.items():
        frame.write_csv(derived / name, float_precision=6, null_value="")
    write_manifest(derived, {name: result.rule_set_id for name in frames})
    stats = dict(result.stats)
    stats["warnings"] = list(result.warnings)
    (folder / STATS_FILE).write_text(_yaml(stats), encoding="utf-8")
    return folder
