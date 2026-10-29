"""Places (stage c1 of ``docs/specs/WORLD_PHASE_C.md``): the second taxonomy run, the place
event types with their requirements, their capacities, and their statistics.

Places come from a second run of the taxonomy generator (WM.40), on the configuration that
``places.taxonomy`` names (``data/taxonomy/places_default.yaml`` by default), with
``labels.kind: place``, so that the run writes ``PLACECATEGORY``, ``PLACE``, ``PLACEPROPERTY``,
``PLACEPART``, and ``PLACESCALARDIM``. The run takes the seed of the stream ``world:places`` when
the configuration gives none, so the place taxonomy never repeats the entities' draws, and is
written unmodified to ``places/`` in the world's folder, as the entity run is written to
``taxonomy/``. Places are not entities: they have place features and no fluents, and nothing
that reads ``entities.csv`` sees a place.

A place event type ``PLACEEVENTTYPE.<n>`` (WM.45) has an entity as its agent and a place as its
patient. Its requirement is a conjunction of one or two constraints, drawn from the stream
``world:place_event_types`` with the constraint sampler of the two-place event types over two
pools: the agent's literals read the entity taxonomy's features and scalars, the patient's the
place taxonomy's (place properties and parts, free or determined, and threshold literals on
place scalars); a comparison, when its family has weight, compares an entity scalar with a
place scalar. A requirement that holds for no binding or for every binding of the world (the
entities paired with the places) is drawn again, as a degenerate two-place event type is by
default; and with the band ``place_event_types.density: [low, high]`` (Jon's ruling WM.67 of the
stage c3b follow-up) so is one whose share of the bindings lies outside the band, up to the same
number of tries, from the same stream, because requirements that hold for most bindings make
the habitat weights nearly flat. A place event type has no precondition and no effects until it
is made a move or an act at a place: since stage c3a an event file's ``place_event_types``
entry gives it its dynamics (``place_dynamics.py``), and stage c3b draws them.

The capacities: ``CAN.PLACEEVENTTYPE.<n>``, an entity is able to be the agent with some possible
place, is the agent projection of the requirement, with the place taxonomy's rules as the
unknown side; ``CANBE.PLACEEVENTTYPE.<n>``, a place is able to be the patient with some possible
entity, is the patient projection with the entity taxonomy's rules as the unknown side. Both are
intensional, flagged when the local test decided, and followed by the extensional capacities
over the world's bindings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from semantic_world.taxonomy.config import ConfigError, ScalarsConfig
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.generate import generate as generate_taxonomy
from semantic_world.taxonomy.rules import CONE_ENUMERATION_LIMIT, _draw
from semantic_world.world.config import Config
from semantic_world.world.constraints import Constraint, _ConstraintSampler
from semantic_world.world.dynamics import AnyLiteral, Effect, precondition_text
from semantic_world.world.projections import project
from semantic_world.world.streams import WorldStreams

PLACE_EVENT_TYPE = "PLACEEVENTTYPE"
PLACE_EVENT_TYPE_KIND = "place_event_type"
"""The symbol kind of a place event type in ``definition.json``."""
PLACE_KIND_OF_TYPE = {"property": "place_property", "part": "place_part"}
"""The symbol kinds of the place features, by the taxonomy's feature type."""
PLACE_SCALAR_KIND = "place_scalar"
PLACE_KINDS = ("move", "act")
"""The values of a place event type's ``place_kind`` (stage c3b, WM.65): a move adds the
location pair, an act requires it."""
EVERYPLACE = "EVERYPLACE"
"""The comparison class of every place (WM.49), as ``ENTITY`` is of every entity."""
PLACES_DIR = "places"
PLACES_FILE = "places.csv"
PLACE_CAPACITIES_FILE = "place_capacities.csv"
MAX_TRIES = 200
"""How many times a degenerate requirement (holding for no binding, or for every binding), or
one outside the density band, is drawn again before the world keeps the last draw with a
warning."""


def place_event_type_label(index: int) -> str:
    return f"{PLACE_EVENT_TYPE}.{index}"


def place_constraint_label(index: int, position: int) -> str:
    """The label of the ``position``-th constraint of the ``index``-th place event type:
    ``CONSTRAINT.PLACEEVENTTYPE.2.1``."""
    return f"CONSTRAINT.{PLACE_EVENT_TYPE}.{index}.{position}"


@dataclass(frozen=True)
class PlaceEventType:
    label: str
    index: int
    constraints: tuple[Constraint, ...]
    density: float
    """The share of the world's bindings (an entity and a place) for which the requirement
    holds."""
    tries: int
    """How many draws the requirement took."""
    in_band: bool = True
    """Whether the drawn requirement's density is neither degenerate nor outside
    ``place_event_types.density``: false only for a drawn requirement kept after ``MAX_TRIES``
    draws (an explicit requirement is never drawn again, so it is always true of one)."""
    precondition: tuple[AnyLiteral, ...] = ()
    """The precondition (stage c3a): empty for a place event type without dynamics; an act's
    location literal ``PAIRFL.k(agent, patient)`` (origin ``act``) or a move's compiled
    ``NOT PAIRFL.k(agent, patient)`` (origin ``move``), with the entry's further literals."""
    effects: tuple[Effect, ...] = ()
    """The effects (stage c3a): a move's add on its location fluent, and further effects on
    the agent's fluents."""
    explicit: bool = False
    """Whether an event file gave the entry (the requirement, the dynamics, or both)."""

    @property
    def requirement_label(self) -> str:
        return f"REQUIREMENT.{self.label}"

    @property
    def performable(self) -> bool:
        """Whether the place event type has events: a move or an act (stage c3a)."""
        return bool(self.precondition or self.effects)

    @property
    def move(self) -> bool:
        return any(e.pair and e.value for e in self.effects)

    @property
    def act(self) -> bool:
        return self.performable and not self.move

    @property
    def dynamics_kind(self) -> str | None:
        """``move``, ``act``, or None without dynamics."""
        if not self.performable:
            return None
        return "move" if self.move else "act"

    @property
    def location(self) -> str | None:
        """The location fluent the move or the act uses, or None."""
        for effect in self.effects:
            if effect.pair and effect.value:
                return effect.fluent
        for lit in self.precondition:
            if lit.pair and lit.kind == "pair" and lit.value:
                return lit.fluent
        return None

    @property
    def roles(self) -> tuple[str, str]:
        return ("agent", "patient")

    @property
    def arity(self) -> int:
        return 2

    @property
    def expression(self) -> str:
        if not self.constraints:
            return "TRUE"
        if len(self.constraints) == 1:
            return str(self.constraints[0].expression)
        return " AND ".join(f"({c.expression})" for c in self.constraints)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(c.family for c in self.constraints)

    def record(self) -> dict[str, Any]:
        """The event-type record of ``definition.json``: a two-place event type whose patient is
        a place, flat (no parent); with an empty precondition and no effects without dynamics
        (stage c1), and with its precondition and effects when an event file made it a move or
        an act (stage c3a, ``explicit`` true) or the generator drew its dynamics (stage c3b),
        written as an entity event type's are, plus the key ``place_kind`` (``move`` or
        ``act``, WM.65), which is part of the rule-set identity and absent from a place event
        type without dynamics."""
        record: dict[str, Any] = {"label": self.label, "kind": PLACE_EVENT_TYPE_KIND}
        if self.performable:
            record["place_kind"] = self.dynamics_kind
        record.update(
            arity=2,
            roles=["agent", "patient"],
            parent=None,
            level=1,
            features=[],
            explicit=self.explicit,
            requirement={
                "constraints": [c.label for c in self.constraints],
                "output": self.requirement_label,
                "expression": " AND ".join(c.label for c in self.constraints)
                if self.constraints
                else "TRUE",
            },
            precondition={
                "literals": [lit.record() for lit in self.precondition],
                "expression": precondition_text(self.precondition),
            },
            effects=[effect.record() for effect in self.effects],
        )
        return record


@dataclass(frozen=True)
class Places:
    """The places of a world: the place taxonomy run, the place event types, their requirements
    over every binding, and their capacities."""

    taxonomy: TaxonomyResult
    """The place taxonomy run, written to ``places/``."""
    event_types: tuple[PlaceEventType, ...]
    requirement: dict[str, np.ndarray]
    """Each place event type's requirement over every binding: shape ``(entities, places)``,
    bool."""
    agent: np.ndarray
    """Intensional ``CAN.PLACEEVENTTYPE.<n>``: shape ``(entities, place event types)``, bool."""
    patient: np.ndarray
    """Intensional ``CANBE.PLACEEVENTTYPE.<n>``: shape ``(places, place event types)``, bool."""
    agent_approximate: np.ndarray
    patient_approximate: np.ndarray
    actual_agent: np.ndarray
    """Extensional: the entity has some place among the world's places."""
    actual_patient: np.ndarray
    """Extensional: the place has some entity among the world's entities."""
    warnings: tuple[str, ...]
    dynamics_stats: dict[str, int] = field(default_factory=dict)
    """The counts of the drawn place dynamics (stage c3b, ``PLACE_DYNAMICS_STATS`` of
    ``place_dynamics.py``); empty in a world without location fluents."""
    density_band: tuple[float, float] | None = None
    """``place_event_types.density``, or None without a band."""
    habitat_strength: float = 1.0
    """``places.habitat_strength``: the exponent of the habitat weights (stage c3a), kept here
    so that the statistics can compute the expected habitats from the requirements (WM.69)."""

    @property
    def outside_band(self) -> tuple[str, ...]:
        """The place event types whose drawn requirement is still degenerate or outside the
        band after ``MAX_TRIES`` draws (reported in the statistics only with a band, as the
        two-place range is)."""
        return tuple(et.label for et in self.event_types if not et.in_band)

    @property
    def labels(self) -> tuple[str, ...]:
        """The places, in place order."""
        return self.taxonomy.instances.labels

    @property
    def count(self) -> int:
        return len(self.taxonomy.instances)

    @property
    def event_type_labels(self) -> tuple[str, ...]:
        return tuple(et.label for et in self.event_types)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        return tuple(c for et in self.event_types for c in et.constraints)

    def event_type(self, label: str) -> PlaceEventType:
        for et in self.event_types:
            if et.label == label:
                return et
        raise KeyError(f"unknown place event type {label!r}")

    @property
    def performable(self) -> tuple[PlaceEventType, ...]:
        """The place event types with dynamics (stage c3a): the moves and the acts."""
        return tuple(et for et in self.event_types if et.performable)

    @property
    def moves(self) -> tuple[PlaceEventType, ...]:
        return tuple(et for et in self.event_types if et.move)

    @property
    def acts(self) -> tuple[PlaceEventType, ...]:
        return tuple(et for et in self.event_types if et.act)

    def holds(self, label: str, agents: np.ndarray, places: np.ndarray) -> np.ndarray:
        """Whether the requirement holds for the aligned bindings ``(agents[k], places[k])`` of
        entity and place indices."""
        matrix = self.requirement[label]
        return matrix[np.asarray(agents, dtype=np.intp), np.asarray(places, dtype=np.intp)]

    def agent_columns(self) -> dict[str, Any]:
        """The place columns of ``derived/capacities.csv``: the intensional agent capacities,
        the approximate flags, and the extensional ones, keyed as the two-place columns are."""
        columns: dict[str, Any] = {}
        for i, label in enumerate(self.event_type_labels):
            columns[f"CAN.{label}"] = self.agent[:, i]
        for i, label in enumerate(self.event_type_labels):
            columns[f"ACTUAL_CAN.{label}"] = self.actual_agent[:, i]
        return columns

    def approximate_agent_labels(self) -> list[str]:
        return [
            f"CAN.{label}"
            for label, a in zip(self.event_type_labels, self.agent_approximate, strict=True)
            if a
        ]

    def approximate_patient_labels(self) -> list[str]:
        return [
            f"CANBE.{label}"
            for label, a in zip(self.event_type_labels, self.patient_approximate, strict=True)
            if a
        ]


# ---------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------


def _sampler_settings(config: Config) -> SimpleNamespace:
    """What the constraint sampler reads: the place event types' rules and families, and the
    key-lock and comparison settings of the two-place event types (the code's defaults when the
    world has none)."""
    settings = config.place_event_types
    assert settings is not None
    binary = config.event_types.binary
    if binary is not None:
        key_lock_pairs, comparison = binary.key_lock_pairs, binary.comparison
    else:
        from semantic_world.world.config import ComparisonConfig

        key_lock_pairs = {1: 0.5, 2: 0.3, 3: 0.2}
        comparison = ComparisonConfig(0.3, 0.2, (0.1, 0.9))
    return SimpleNamespace(
        rules=settings.rules,
        constraint_families=settings.constraint_families,
        key_lock_pairs=key_lock_pairs,
        comparison=comparison,
    )


def requirement_matrix(
    constraints: tuple[Constraint, ...],
    values: np.ndarray,
    scalars: np.ndarray,
    place_values: np.ndarray,
    place_scalars: np.ndarray,
) -> np.ndarray:
    """A conjunction of constraints over every binding of an entity (a row of ``values``) and
    a place (a row of ``place_values``): shape ``(entities, places)``, bool."""
    n = values.shape[0]
    out = np.ones((n, place_values.shape[0]), dtype=bool)
    agents = np.arange(n)
    for constraint in constraints:
        out &= constraint.matrix(values, scalars, agents, place_values, place_scalars)
    return out


def generate_places(
    config: Config, taxonomy: TaxonomyResult, streams: WorldStreams, event_file=None
) -> Places | None:
    """The places of a world, or None with ``places: null``: the place taxonomy run (in memory,
    on the resolved place configuration), then the place event types from
    ``world:place_event_types``, each requirement drawn again while it holds for no binding or
    for every binding of the world, or for a share of the bindings outside the band
    ``place_event_types.density`` when one is set (WM.67), up to ``MAX_TRIES`` draws, then the
    capacities. An event file's ``place_event_types``
    entry with a requirement (stage c3a) replaces the drawn requirement of its place event
    type after every draw is made, before the capacities, as an explicit two-place requirement
    replaces an own constraint; the entry's dynamics come after the fluents
    (``place_dynamics.py``)."""
    if config.places is None:
        return None
    place_run = generate_taxonomy(config.places.taxonomy_run)
    explicit: dict[str, Constraint] = {}
    if event_file is not None:
        from semantic_world.world.requirements import place_requirement

        for label, entry in event_file.place_event_types.items():
            if entry.requirement is None:
                continue
            if label not in config.place_event_types.labels:  # type: ignore[union-attr]
                raise ConfigError(
                    event_file.source,
                    entry.field,
                    "unknown place event type; the world has none with this label",
                )
            explicit[label] = place_requirement(
                entry,
                taxonomy.features,
                taxonomy.config.scalars,
                place_run.features,
                place_run.config.scalars,
                event_file.source,
            )
    settings = config.place_event_types
    assert settings is not None
    rng = streams.place_event_types
    sampler = _ConstraintSampler(
        _sampler_settings(config),
        taxonomy.config.scalars,
        taxonomy.features,
        rng,
        patient_scalars=place_run.config.scalars,
        patient_features=place_run.features,
    )
    values, scalars = taxonomy.instances.values, taxonomy.instances.scalars
    place_values, place_scalars = place_run.instances.values, place_run.instances.scalars
    event_types: list[PlaceEventType] = []
    requirement: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    bindings = values.shape[0] * place_values.shape[0]
    band = settings.density
    for index in range(1, settings.count + 1):
        label = place_event_type_label(index)
        tries = 0
        while True:
            tries += 1
            count = _draw(rng, settings.constraints)
            constraints = tuple(
                sampler.sample(place_constraint_label(index, position))
                for position in range(1, count + 1)
            )
            matrix = requirement_matrix(constraints, values, scalars, place_values, place_scalars)
            density = float(matrix.mean()) if bindings else math.nan
            if bindings == 0 or settings.density_ok(density) or tries >= MAX_TRIES:
                break
        in_band = label in explicit or bindings == 0 or settings.density_ok(density)
        if label in explicit:
            constraints = (explicit[label],)
            matrix = requirement_matrix(constraints, values, scalars, place_values, place_scalars)
            density = float(matrix.mean()) if bindings else math.nan
        elif bindings and not 0.0 < density < 1.0:
            warnings.append(
                f"place event type {label} holds for "
                f"{'no' if density == 0.0 else 'every'} binding after {tries} tries"
            )
        elif bindings and not settings.density_ok(density):
            assert band is not None
            warnings.append(
                f"place event type {label} holds for {density:.4f} of the bindings, outside the "
                f"density band [{band[0]}, {band[1]}], after {tries} tries"
            )
        event_types.append(
            PlaceEventType(
                label,
                index,
                constraints,
                density,
                tries,
                in_band=in_band,
                explicit=label in explicit,
            )
        )
        requirement[label] = matrix
    n, m, k = values.shape[0], place_values.shape[0], len(event_types)
    agent = np.zeros((n, k), dtype=bool)
    patient = np.zeros((m, k), dtype=bool)
    agent_approximate = np.zeros(k, dtype=bool)
    patient_approximate = np.zeros(k, dtype=bool)
    actual_agent = np.zeros((n, k), dtype=bool)
    actual_patient = np.zeros((m, k), dtype=bool)
    for i, et in enumerate(event_types):
        agent[:, i], agent_approximate[i] = project(
            et.constraints, "patient", place_run.rules, taxonomy.instances,
            CONE_ENUMERATION_LIMIT, rng,
        )  # fmt: skip
        patient[:, i], patient_approximate[i] = project(
            et.constraints, "agent", taxonomy.rules, place_run.instances,
            CONE_ENUMERATION_LIMIT, rng,
        )  # fmt: skip
        actual_agent[:, i] = requirement[et.label].any(axis=1)
        actual_patient[:, i] = requirement[et.label].any(axis=0)
    return Places(
        place_run,
        tuple(event_types),
        requirement,
        agent,
        patient,
        agent_approximate,
        patient_approximate,
        actual_agent,
        actual_patient,
        tuple(warnings),
        density_band=band,
        habitat_strength=config.places.habitat_strength,
    )


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def places_frame(places: Places) -> pl.DataFrame:
    """``places.csv``: one row per place with its label, its leaf, its free place features, and
    its place scalars at full precision, as ``entities.csv`` holds the entities' base facts."""
    instances = places.taxonomy.instances
    data: dict[str, Any] = {
        "label": list(instances.labels),
        "leaf": list(instances.leaf_labels),
    }
    for feature in places.taxonomy.features.free:
        data[feature.label] = instances.values[:, feature.position].astype(np.int64).tolist()
    for j, label in enumerate(places.taxonomy.features.scalar_labels):
        data[label] = instances.scalars[:, j].astype(float).tolist()
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})


def place_capacities_frame(places: Places) -> pl.DataFrame:
    """``derived/place_capacities.csv``: one row per place with ``CANBE.PLACEEVENTTYPE.<n>``
    for every place event type (intensional), the ``approximate`` flag, and the extensional
    ``ACTUAL_CANBE.`` columns."""
    data: dict[str, Any] = {"label": list(places.labels)}
    for i, label in enumerate(places.event_type_labels):
        data[f"CANBE.{label}"] = places.patient[:, i].astype(np.int64).tolist()
    data["approximate"] = [";".join(places.approximate_patient_labels())] * places.count
    for i, label in enumerate(places.event_type_labels):
        data[f"ACTUAL_CANBE.{label}"] = places.actual_patient[:, i].astype(np.int64).tolist()
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "approximate": pl.Utf8})


def _level_assignment(taxonomy: TaxonomyResult, level: int) -> tuple[list[str], np.ndarray]:
    tree, instances = taxonomy.tree, taxonomy.instances
    labels = [c.label for c in tree.at_level(level)]
    index = {label: i for i, label in enumerate(labels)}
    assignment = np.array(
        [index[tree.ancestor_label(tree.categories[k], level)] for k in instances.leaf_index],
        dtype=np.intp,
    )
    return labels, assignment


def place_proportion_rows(places: Places, taxonomy: TaxonomyResult) -> list[dict[str, Any]]:
    """The rows of ``relation_proportions.csv`` for the place event types: for each level of
    the entity tree, every entity category at that level paired with every place category at
    the same level, or at the place tree's last level when the entity tree is deeper, with the
    share of their bindings for which the requirement holds; rows with a share above 0."""
    place_depth = places.taxonomy.tree.depth
    rows: list[dict[str, Any]] = []
    for level in range(1, taxonomy.tree.depth + 1):
        agent_labels, agent_assignment = _level_assignment(taxonomy, level)
        patient_labels, patient_assignment = _level_assignment(
            places.taxonomy, min(level, place_depth)
        )
        agent_onehot = np.zeros((agent_assignment.size, len(agent_labels)))
        agent_onehot[np.arange(agent_assignment.size), agent_assignment] = 1.0
        patient_onehot = np.zeros((patient_assignment.size, len(patient_labels)))
        patient_onehot[np.arange(patient_assignment.size), patient_assignment] = 1.0
        totals = np.outer(agent_onehot.sum(axis=0), patient_onehot.sum(axis=0))
        for et in places.event_types:
            counts = agent_onehot.T @ places.requirement[et.label].astype(float) @ patient_onehot
            for a, p in zip(*np.nonzero(counts > 0), strict=True):
                rows.append(
                    {
                        "event_type": et.label,
                        "level": level,
                        "agent": agent_labels[a],
                        "patient": patient_labels[p],
                        "true_pairs": int(round(counts[a, p])),
                        "total_pairs": int(totals[a, p]),
                        "proportion": float(counts[a, p] / totals[a, p]),
                        "estimated": False,
                    }
                )
    return rows


def place_stat_rows(places: Places, with_density: bool) -> list[dict[str, Any]]:
    """The rows of ``event_type_stats.csv`` for the place event types: the share of bindings
    for which the requirement holds, the entities and the places with a partner, the
    constraints and their families; the symmetric and within-leaf shares are NaN, because a
    binding is never reversed and an entity shares no leaf with a place."""
    rows = []
    for et in places.event_types:
        matrix = places.requirement[et.label]
        row: dict[str, Any] = {
            "event_type": et.label,
            "proportion_true": float(matrix.mean()) if matrix.size else math.nan,
            "agents": int(matrix.any(axis=1).sum()),
            "patients": int(matrix.any(axis=0).sum()),
            "constraints": len(et.constraints),
            "families": ";".join(et.families),
            "symmetric_proportion": math.nan,
            "within_leaf_proportion": math.nan,
            "estimated": False,
        }
        if with_density:
            row["leaf_pair_density"] = et.density
            row["tries"] = et.tries
        rows.append(row)
    return rows


def places_stats(
    places: Places | None, pair_fluents=None, taxonomy: TaxonomyResult | None = None
) -> dict[str, Any] | None:
    """The ``places`` block of ``world_stats.yaml``: the places and their categories, the place
    features by kind, the place event types with their constraints by family, and their
    densities over the world's bindings (with a density band, WM.67, also the band and the
    place event types still outside it, ``outside_density`` and ``outside_density_labels``,
    as the ``relations`` block reports the two-place range); and, only in a world with location
    fluents (stage
    c3b), the place event types by kind (``place_event_types_by_kind``: moves, acts, and
    those without dynamics), the counts of the drawn dynamics (``dynamics``), and the
    concentration of habitats (``habitats``, from ``habitat_stats`` in ``locations.py``, when
    ``pair_fluents`` and the entity ``taxonomy`` are given: the sampled shares and entropies,
    and under WM.69 the expected effect sizes at the leaf level, from the requirements and
    the habitat strength)."""
    if places is None:
        return None
    features = places.taxonomy.features
    by_kind: dict[str, Any] = {}
    for feature_type, kind in PLACE_KIND_OF_TYPE.items():
        of_type = features.of_type(feature_type)
        by_kind[kind] = {
            "total": len(of_type),
            "free": sum(1 for f in of_type if f.free),
            "determined": sum(1 for f in of_type if not f.free),
        }
    by_kind[PLACE_SCALAR_KIND] = features.scalar_count
    families: dict[str, int] = {}
    for constraint in places.constraints:
        families[constraint.family] = families.get(constraint.family, 0) + 1
    total = 2 * len(places.event_types)
    approximate = int(places.agent_approximate.sum() + places.patient_approximate.sum())
    block: dict[str, Any] = {
        "places": places.count,
        "categories": len(places.taxonomy.tree.categories),
        "features": by_kind,
        "place_event_types": len(places.event_types),
        "constraints": len(places.constraints),
        "constraint_families": families,
        "density": {
            et.label: round(et.density, 6) if not math.isnan(et.density) else None
            for et in places.event_types
        },
        "tries": {et.label: et.tries for et in places.event_types},
        "approximate_capacities": round(approximate / total, 6) if total else 0.0,
        "warnings": list(places.warnings),
    }
    if places.density_band is not None:
        # the band (WM.67), as relations.outside_density reports the two-place range
        block["density_band"] = list(places.density_band)
        block["outside_density"] = len(places.outside_band)
        block["outside_density_labels"] = list(places.outside_band)
    locations = [f for f in (pair_fluents.fluents if pair_fluents is not None else ()) if f.total]
    if locations:
        from semantic_world.world.locations import habitat_stats

        block["place_event_types_by_kind"] = {
            "moves": len(places.moves),
            "acts": len(places.acts),
            "without_dynamics": sum(1 for et in places.event_types if not et.performable),
            "explicit": sum(1 for et in places.event_types if et.explicit and et.performable),
        }
        block["dynamics"] = dict(places.dynamics_stats)
        if taxonomy is not None:
            block["habitats"] = habitat_stats(
                pair_fluents,
                taxonomy,
                places.taxonomy,
                places.requirement.values(),
                places.habitat_strength,
            )
    return block


def place_scalars_config(places: Places) -> ScalarsConfig:
    return places.taxonomy.config.scalars


__all__ = [
    "EVERYPLACE",
    "MAX_TRIES",
    "PLACES_DIR",
    "PLACES_FILE",
    "PLACE_CAPACITIES_FILE",
    "PLACE_EVENT_TYPE",
    "PLACE_EVENT_TYPE_KIND",
    "PLACE_KINDS",
    "PLACE_KIND_OF_TYPE",
    "PLACE_SCALAR_KIND",
    "PlaceEventType",
    "Places",
    "generate_places",
    "place_capacities_frame",
    "place_constraint_label",
    "place_event_type_label",
    "place_proportion_rows",
    "place_stat_rows",
    "places_frame",
    "places_stats",
    "requirement_matrix",
]
