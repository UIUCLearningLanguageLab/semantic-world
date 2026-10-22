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
from semantic_world.world.comparisons import COMPARISON_CLASSES_FILE, comparison_classes_frame
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
    produced_numeric,
    redraw_preconditions,
)
from semantic_world.world.fluents import Fluents, derived_initial_values, generate_fluents
from semantic_world.world.locations import with_location_fluents
from semantic_world.world.numeric import NumericFluents, generate_numeric_fluents
from semantic_world.world.pair_fluents import PairFluents, generate_pair_fluents
from semantic_world.world.place_dynamics import (
    apply_place_dynamics,
    check_place_dynamics,
    draw_place_dynamics,
    place_signs,
    redraw_place_preconditions,
)
from semantic_world.world.places import (
    PLACE_CAPACITIES_FILE,
    PLACES_DIR,
    Places,
    generate_places,
    place_capacities_frame,
    place_proportion_rows,
    place_stat_rows,
)
from semantic_world.world.relation_stats import (
    event_type_stat_schema,
    proportion_schema,
)
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
    numeric: NumericFluents
    report: DefinitionReport
    stream_seeds: dict[str, int]
    stats: dict[str, Any]
    warnings: tuple[str, ...]
    places: Places | None = None
    """The places (stage c1): the place taxonomy run, written to ``places/``, the place event
    types, and their capacities; None with ``places: null``."""

    @property
    def rule_set_id(self) -> str:
        return self.definition.rule_set_id

    @property
    def pair_fluents(self) -> PairFluents:
        """The pair fluents and their initial pairs (stage c2a): the pair fluents between
        entities, drawn (stage c2b) or from the event file, then the location fluents (stage
        c3a) with the initial places."""
        return self.definition.pair_fluents

    def derived_frames(self) -> dict[str, pl.DataFrame]:
        """Every table of ``derived/``, keyed by file name. With places, ``capacities.csv``
        holds the entities' ``CAN.PLACEEVENTTYPE`` columns, ``place_capacities.csv`` the
        places' ``CANBE.PLACEEVENTTYPE`` columns, ``comparison_classes.csv`` the place classes,
        and ``relation_proportions.csv`` and ``event_type_stats.csv`` the rows of the place
        event types after the two-place ones (written even without two-place event types)."""
        places = self.places
        place_taxonomy = places.taxonomy if places is not None else None
        frames = {
            STATIC_FEATURES_FILE: static_features_frame(self.taxonomy),
            CAPACITIES_FILE: capacities_frame(self.statics, places),
            CAPACITY_ROLES_FILE: capacity_roles_frame(self.statics),
            COMPARISON_CLASSES_FILE: comparison_classes_frame(self.taxonomy, place_taxonomy),
        }
        relation_stats = self.statics.relation_stats
        if relation_stats is not None:
            frames.update(relation_stats.frames())
        if places is not None:
            frames[PLACE_CAPACITIES_FILE] = place_capacities_frame(places)
            with_density = (
                self.statics.relations is not None
                and self.statics.relations.event_type_density is not None
            ) or relation_stats is None
            proportions = pl.DataFrame(
                place_proportion_rows(places, self.taxonomy), schema=proportion_schema()
            )
            stats = pl.DataFrame(
                place_stat_rows(places, with_density),
                schema=event_type_stat_schema(with_density),
            )
            for name, rows in (
                ("relation_proportions.csv", proportions),
                ("event_type_stats.csv", stats),
            ):
                frames[name] = (
                    pl.concat([frames[name], rows], how="vertical") if name in frames else rows
                )
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
    places = generate_places(config, taxonomy, streams, event_file)
    pair_fluents = generate_pair_fluents(
        len(taxonomy.instances),
        streams,
        explicit=event_file.pair_fluents if event_file is not None else None,
        explicit_source=event_file.source if event_file is not None else None,
        config=config.pair_fluents,
    )
    if config.pair_fluents.locations:
        assert places is not None and config.places is not None  # the configuration checks it
        pair_fluents = with_location_fluents(
            pair_fluents,
            config.pair_fluents.locations,
            places.requirement.values(),
            len(taxonomy.instances),
            places.count,
            config.places.habitat_strength,
            streams.habitats,
        )
    spatial = event_file.spatial if event_file is not None else None
    numeric = generate_numeric_fluents(
        config,
        taxonomy,
        streams,
        explicit=event_file.numeric if event_file is not None else None,
        explicit_source=event_file.source if event_file is not None else None,
    )
    fluents = generate_fluents(
        config,
        taxonomy,
        streams,
        explicit_rates=event_file.fluents if event_file is not None else None,
        explicit_source=event_file.source if event_file is not None else None,
        numeric=numeric,
        pair_fluents=pair_fluents,
    )
    derived_initial = derived_initial_values(fluents, taxonomy, numeric, pair_fluents)
    # the effects of the event file's place entries are fixed before any draw, so the entity
    # event types' pools of achievable values count them (stage c3b, WM.66)
    explicit_place_effects = (
        [e for entry in event_file.place_event_types.values() for e in (entry.effects or ())]
        if event_file is not None
        else []
    )
    event_types = generate_event_types(
        statics,
        fluents,
        config,
        streams,
        event_file,
        derived_initial,
        numeric,
        pair_fluents,
        explicit_place_effects,
    )
    initial_present = _initial_present(fluents, derived_initial)
    features = tuple(f.label for f in taxonomy.features.features)
    _check(
        event_types,
        fluents,
        initial_present,
        features,
        numeric,
        pair_fluents,
        explicit_place_effects,
    )
    # the moves and acts of the event file's place_event_types entries (stage c3a)
    places = apply_place_dynamics(
        places,
        event_file,
        fluents,
        numeric,
        pair_fluents,
        initial_present,
        [e for et in event_types.event_types for e in et.effects],
        produced_numeric(event_types),
    )
    # the drawn dynamics of the other place event types (stage c3b), from world:place_dynamics
    places = draw_place_dynamics(
        places,
        config,
        streams.place_dynamics,
        fluents,
        numeric,
        pair_fluents,
        taxonomy.features,
        taxonomy.instances.values,
        initial_present,
        [e for et in event_types.event_types for e in et.effects],
    )
    _check(
        event_types,
        fluents,
        initial_present,
        features,
        numeric,
        pair_fluents,
        _place_effects(places),
    )
    _check_places(places, event_types, fluents, initial_present, numeric, pair_fluents)
    definition = build_definition(
        statics, fluents, event_types, numeric, places, pair_fluents, spatial
    )
    fixed = set()
    if event_file is not None:
        fixed = {
            label
            for label, entry in event_file.event_types.items()
            if entry.precondition is not None
        }
        # an explicit place entry's further literals are the file's: never drawn again
        fixed |= set(event_file.place_event_types)
    place_labels = set(places.event_type_labels) if places is not None else set()
    redraws: dict[str, int] = {}
    parts: dict[str, np.random.Generator] = {}
    numeric_parts: dict[str, np.random.Generator] = {}
    pair_parts: dict[str, np.random.Generator] = {}
    place_parts: dict[str, np.random.Generator] = {}
    episodes = statistics_episodes(definition, statics, config, redraws)
    for _ in range(MAX_REDRAW_ROUNDS):
        never = [
            label
            for label, reason in episodes["never_legal"].items()
            if reason == "preconditions" and label not in fixed
        ]
        if not never:
            break
        never_entities = [label for label in never if label not in place_labels]
        never_places = [label for label in never if label in place_labels]
        for label in never_entities:
            parts.setdefault(label, streams.part("preconditions", label))
            numeric_parts.setdefault(label, streams.part("numeric_preconditions", label))
            if len(pair_fluents):
                pair_parts.setdefault(label, streams.part("pair_preconditions", label))
        for label in never_places:
            place_parts.setdefault(label, streams.part("place_dynamics", label))
        for label in never:
            redraws[label] = redraws.get(label, 0) + 1
        if never_entities:
            event_types = redraw_preconditions(
                event_types,
                never_entities,
                fluents,
                config,
                parts,
                derived_initial,
                numeric,
                numeric_parts,
                pair_fluents,
                pair_parts,
                _place_effects(places),
            )
        if never_places:
            assert places is not None
            places = redraw_place_preconditions(
                places,
                never_places,
                config,
                place_parts,
                fluents,
                numeric,
                pair_fluents,
                taxonomy.features,
                taxonomy.instances.values,
                initial_present,
                [e for et in event_types.event_types for e in et.effects],
            )
        _check(
            event_types,
            fluents,
            initial_present,
            features,
            numeric,
            pair_fluents,
            _place_effects(places),
        )
        _check_places(places, event_types, fluents, initial_present, numeric, pair_fluents)
        definition = build_definition(
            statics, fluents, event_types, numeric, places, pair_fluents, spatial
        )
        episodes = statistics_episodes(definition, statics, config, redraws)
    report = check_definition(definition)
    stats = world_stats(fluents, event_types, episodes, statics, numeric, places, pair_fluents)
    seeds = {**TaxonomyStreams(taxonomy.config.seed).seeds(), **streams.seeds()}
    warnings = tuple(taxonomy.warnings) + tuple(statics.warnings)
    if places is not None:
        warnings += tuple(places.taxonomy.warnings) + tuple(places.warnings)
    return WorldResult(
        config=config,
        taxonomy=taxonomy,
        statics=statics,
        event_file=event_file,
        fluents=fluents,
        event_types=event_types,
        definition=definition,
        numeric=numeric,
        report=report,
        stream_seeds=seeds,
        stats=stats,
        warnings=warnings,
        places=places,
    )


def _place_effects(places: Places | None) -> list:
    """Every effect of the place event types (stage c3b, WM.66: the pools of achievable values
    count them)."""
    if places is None:
        return []
    return [e for et in places.event_types for e in et.effects]


def _check(
    event_types: EventTypes,
    fluents: Fluents,
    initial_present: set,
    features: tuple[str, ...],
    numeric: NumericFluents | None = None,
    pair_fluents: PairFluents | None = None,
    extra_effects=(),
) -> None:
    try:
        check_dynamics(
            event_types, fluents, initial_present, features, numeric, pair_fluents, extra_effects
        )
    except ValueError as error:
        raise WorldError(f"the event types break a rule of the dynamics: {error}") from None


def _check_places(
    places: Places | None,
    event_types: EventTypes,
    fluents: Fluents,
    initial_present: set,
    numeric: NumericFluents | None,
    pair_fluents: PairFluents | None,
) -> None:
    try:
        check_place_dynamics(
            places,
            fluents,
            initial_present,
            [e for et in event_types.event_types for e in et.effects],
            produced_numeric(event_types) | place_signs(places),
            numeric,
            pair_fluents,
        )
    except ValueError as error:
        raise WorldError(f"the place event types break a rule of the dynamics: {error}") from None


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
    from semantic_world.world.runtime import able_table, place_able_table

    runtime = definition.runtime()
    strength = config.places.habitat_strength if config.places is not None else 1.0
    generator = EpisodeGenerator(
        runtime,
        Relatedness.from_statics(statics, runtime),
        record_legal=True,
        habitat_strength=strength,
    )
    histories = generator.run(config.seed, STATS_EPISODES, STATS_STREAM)
    able = {**place_able_table(runtime), **able_table(runtime)}
    never_able = [label for label in generator.performable if not able[label].any()]
    # the move share, the act share, and the two-place share between entities (stage c3b),
    # only in a world with location fluents
    place_kinds = (
        {et.label: et.place_kind for et in runtime.event_types if et.place and et.performable}
        if runtime.has_location_fluents
        else None
    )
    return episode_stats(histories, generator.performable, redraws, never_able, place_kinds)


def _yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True)


def write_result(result: WorldResult, path: str | Path | None = None) -> Path:
    folder = Path(path) if path is not None else default_output_dir(result.config)
    folder.mkdir(parents=True, exist_ok=True)
    config_data = result.config.resolved()
    config_data["provenance"] = provenance(result.stream_seeds)
    (folder / "config.yaml").write_text(_yaml(config_data), encoding="utf-8")
    result.taxonomy.write(folder / TAXONOMY_DIR)
    if result.places is not None:
        result.places.taxonomy.write(folder / PLACES_DIR)
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
