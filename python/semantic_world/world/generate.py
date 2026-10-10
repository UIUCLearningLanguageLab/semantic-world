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
from semantic_world.taxonomy.io import STATIC_FEATURES_FILE, provenance, static_features_frame
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
from semantic_world.world.event_types import (
    EventTypes,
    check_dynamics,
    generate_event_types,
    redraw_preconditions,
)
from semantic_world.world.fluents import Fluents, derived_initial_values, generate_fluents
from semantic_world.world.statics import StaticWorld, build_statics
from semantic_world.world.stats import episode_stats, world_stats
from semantic_world.world.streams import WorldStreams

TAXONOMY_DIR = "taxonomy"
STATS_FILE = "world_stats.yaml"
MAX_REDRAW_ROUNDS = 5
"""How many times the preconditions of an event type that was never legal in the statistics
episodes are drawn again before the world keeps it as it is."""


@dataclass(frozen=True)
class WorldResult:
    config: Config
    taxonomy: TaxonomyResult
    """The taxonomy run the world is built on, written to ``taxonomy/``."""
    statics: StaticWorld
    """The static side: the one-place requirements, the event tree, the relations, the
    capacities, and the relation statistics."""
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
        frames = {
            STATIC_FEATURES_FILE: static_features_frame(self.taxonomy),
            CAPACITIES_FILE: capacities_frame(self.statics),
            CAPACITY_ROLES_FILE: capacity_roles_frame(self.statics),
        }
        if self.statics.relation_stats is not None:
            frames.update(self.statics.relation_stats.frames())
        return frames

    def write(self, path: str | Path | None = None) -> Path:
        return write_result(self, path)


def default_output_dir(config: Config, base: str | Path = "runs/world") -> Path:
    return Path(base) / f"{config.name}_seed{config.seed}"


def _initial_present(fluents: Fluents, derived_initial: dict[str, np.ndarray]) -> set:
    present: set[tuple[str, bool]] = set()
    for i, fluent in enumerate(fluents.base):
        for value in np.unique(fluents.initial_values[:, i]):
            present.add((fluent.label, bool(value)))
    for label, column in derived_initial.items():
        for value in np.unique(column):
            present.add((label, bool(value)))
    return present


def define(config: Config) -> WorldResult:
    """Generate a world: the taxonomy run, the static side, the fluents, the event types, the
    definition, the agreement test, and the statistics episodes. An event type that is never
    legal in the statistics episodes has its own precondition literals drawn again, from its own
    part of ``world:preconditions``, up to ``MAX_REDRAW_ROUNDS`` times; the redraws and whatever
    is still never legal are reported in ``world_stats.yaml``. A disagreement, or a broken
    invariant of the dynamics, raises :class:`WorldError` and fails the run."""
    taxonomy = generate_taxonomy(config.taxonomy_config())
    event_file = None
    path = config.event_file_path()
    if path is not None:
        event_file = load_event_file(path)
    streams = WorldStreams(config.seed)
    statics = build_statics(taxonomy, config, streams, event_file)
    fluents = generate_fluents(
        config,
        taxonomy,
        streams,
        explicit_rates=event_file.fluents if event_file is not None else None,
        explicit_source=event_file.source if event_file is not None else None,
    )
    derived_initial = derived_initial_values(fluents, taxonomy)
    event_types = generate_event_types(
        statics, fluents, config, streams, event_file, derived_initial
    )
    initial_present = _initial_present(fluents, derived_initial)
    features = tuple(f.label for f in taxonomy.features.features)
    _check(event_types, fluents, initial_present, features)
    definition = build_definition(statics, fluents, event_types)
    fixed = set()
    if event_file is not None:
        fixed = {
            label
            for label, entry in event_file.event_types.items()
            if entry.precondition is not None
        }
    redraws: dict[str, int] = {}
    parts: dict[str, np.random.Generator] = {}
    episodes = statistics_episodes(definition, statics, config, redraws)
    for _ in range(MAX_REDRAW_ROUNDS):
        never = [
            label
            for label, reason in episodes["never_legal"].items()
            if reason == "preconditions" and label not in fixed
        ]
        if not never:
            break
        for label in never:
            parts.setdefault(label, streams.part("preconditions", label))
            redraws[label] = redraws.get(label, 0) + 1
        event_types = redraw_preconditions(
            event_types, never, fluents, config, parts, derived_initial
        )
        _check(event_types, fluents, initial_present, features)
        definition = build_definition(statics, fluents, event_types)
        episodes = statistics_episodes(definition, statics, config, redraws)
    report = check_definition(definition)
    stats = world_stats(fluents, event_types, episodes, statics)
    seeds = {**TaxonomyStreams(taxonomy.config.seed).seeds(), **streams.seeds()}
    return WorldResult(
        config=config,
        taxonomy=taxonomy,
        statics=statics,
        event_file=event_file,
        fluents=fluents,
        event_types=event_types,
        definition=definition,
        report=report,
        stream_seeds=seeds,
        stats=stats,
        warnings=tuple(taxonomy.warnings) + tuple(statics.warnings),
    )


def _check(
    event_types: EventTypes, fluents: Fluents, initial_present: set, features: tuple[str, ...]
) -> None:
    try:
        check_dynamics(event_types, fluents, initial_present, features)
    except ValueError as error:
        raise WorldError(f"the event types break a rule of the dynamics: {error}") from None


def statistics_episodes(
    definition: Definition,
    statics: StaticWorld,
    config: Config,
    redraws: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The episode statistics of ``world_stats.yaml``: 1,000 episodes of the default policy with
    the default scene settings, on the ``world:stats`` stream, each from its own part."""
    from semantic_world.world.episodes import (
        STATS_EPISODES,
        STATS_STREAM,
        EpisodeGenerator,
        Relatedness,
    )
    from semantic_world.world.runtime import able_table

    runtime = definition.runtime()
    generator = EpisodeGenerator(
        runtime, Relatedness.from_statics(statics, runtime), record_legal=True
    )
    histories = generator.run(config.seed, STATS_EPISODES, STATS_STREAM)
    able = able_table(runtime)
    never_able = [label for label in generator.performable if not able[label].any()]
    return episode_stats(histories, generator.performable, redraws, never_able)


def _yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True)


def write_result(result: WorldResult, path: str | Path | None = None) -> Path:
    folder = Path(path) if path is not None else default_output_dir(result.config)
    folder.mkdir(parents=True, exist_ok=True)
    config_data = result.config.resolved()
    config_data["provenance"] = provenance(result.stream_seeds)
    (folder / "config.yaml").write_text(_yaml(config_data), encoding="utf-8")
    result.taxonomy.write(folder / TAXONOMY_DIR)
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


__all__ = [
    "MAX_REDRAW_ROUNDS",
    "STATS_FILE",
    "TAXONOMY_DIR",
    "WorldResult",
    "default_output_dir",
    "define",
    "statistics_episodes",
    "write_result",
]
