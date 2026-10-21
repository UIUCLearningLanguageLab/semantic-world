"""Location fluents and habitats (stage c3a of ``docs/specs/WORLD_PHASE_C.md``, "Pair fluents":
"How many", "Profiles", and "Initial places (habitats)").

A location fluent is a pair fluent from entities to places with the fixed profile ``general``,
functional, and total: each entity is at exactly one place at every time point. The world has
``pair_fluents.locations`` of them (0 by default until stage c4, WM.63; they need places), labeled
after the pair fluents between entities, ``PAIRFL.<count+1>`` and on (WM.54), so that turning
places on or off leaves the labels of the pair fluents between entities unchanged. A location
fluent carries no initial rate: its initial pairs are the entities' initial places, drawn once,
when the world is generated, from the stream ``world:habitats``, with the weight ``(1 + a(x, q))
^ places.habitat_strength`` for place ``q`` and entity ``x``, where ``a(x, q)`` is the number of
place event types whose requirement holds for the binding ``(x, q)`` (WM.39); strength 0 gives
the uniform draw. With several location fluents each draws its own place, in label order. The
places are columns of ``entities.csv``, one per location fluent, named by the fluent's label,
written only when the world has location fluents; the runtime reads them back as the location
fluents' initial pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from semantic_world.world.errors import WorldError
from semantic_world.world.pair_fluents import PairFluent, PairFluents, pair_label

LOCATION_SYMMETRY = "general"
DEFAULT_HABITAT_STRENGTH = 1.0


def location_fluent(index: int) -> PairFluent:
    """The location fluent ``PAIRFL.<index>``: general, functional, total, no initial rate."""
    return PairFluent(pair_label(index), index, LOCATION_SYMMETRY, True, None, total=True)


def habitat_weights(
    requirements: Iterable[np.ndarray], shape: tuple[int, int], strength: float
) -> np.ndarray:
    """The habitat weights of every binding of an entity and a place: ``(1 + a(x, q)) ^
    strength``, where ``a(x, q)`` counts the place event types whose requirement (one bool matrix
    of shape ``(entities, places)`` each) holds for the binding. Strength 0 gives 1 everywhere."""
    able = np.zeros(shape, dtype=float)
    for matrix in requirements:
        able += np.asarray(matrix, dtype=bool)
    return np.power(1.0 + able, float(strength))


def draw_places(
    weights: np.ndarray, rng: np.random.Generator, entities: Sequence[int] | None = None
) -> np.ndarray:
    """One place per entity (every entity when ``entities`` is None, else the given ones in the
    given order), each drawn from ``rng`` by its row of weights: one draw per entity, in order.
    Returns the place index of each drawn entity."""
    rows = np.arange(weights.shape[0]) if entities is None else np.asarray(entities, dtype=np.intp)
    m = weights.shape[1]
    if m == 0:
        raise WorldError("a location fluent needs at least one place")
    found = np.zeros(rows.size, dtype=np.intp)
    for k, x in enumerate(rows):
        row = weights[int(x)]
        found[k] = int(rng.choice(m, p=row / row.sum()))
    return found


def places_matrix(choices: np.ndarray, place_count: int) -> np.ndarray:
    """The location matrix of one place per entity: shape ``(entities, places)``, bool."""
    matrix = np.zeros((choices.size, place_count), dtype=bool)
    matrix[np.arange(choices.size), np.asarray(choices, dtype=np.intp)] = True
    return matrix


def with_location_fluents(
    pair_fluents: PairFluents,
    count: int,
    requirements: Iterable[np.ndarray],
    entity_count: int,
    place_count: int,
    strength: float,
    rng: np.random.Generator,
) -> PairFluents:
    """The pair fluents with ``count`` location fluents appended, labeled after the pair fluents
    between entities, each with its initial places drawn from ``rng`` (``world:habitats``) in
    label order, one draw per entity. With ``count`` 0 nothing is drawn and the pair fluents
    are returned as they are."""
    if count <= 0:
        return pair_fluents
    if place_count <= 0:
        raise WorldError("location fluents need places")
    weights = habitat_weights(requirements, (entity_count, place_count), strength)
    fluents = list(pair_fluents.fluents)
    matrices = list(pair_fluents.initial_pairs)
    first = len(pair_fluents.between_entities) + 1
    for index in range(first, first + count):
        fluents.append(location_fluent(index))
        matrices.append(places_matrix(draw_places(weights, rng), place_count))
    return PairFluents(tuple(fluents), tuple(matrices))


def location_columns(
    pair_fluents: PairFluents, place_labels: Sequence[str]
) -> dict[str, list[str]]:
    """The location columns of ``entities.csv`` (stage c3a): for each location fluent, in label
    order, each entity's initial place as a place label."""
    columns: dict[str, list[str]] = {}
    for fluent, matrix in zip(pair_fluents.fluents, pair_fluents.initial_pairs, strict=True):
        if not fluent.total:
            continue
        places = np.argmax(matrix, axis=1)
        if not matrix.any(axis=1).all():
            raise WorldError(f"{fluent.label}: an entity has no initial place")
        columns[fluent.label] = [place_labels[int(q)] for q in places]
    return columns


def location_matrices_from_columns(
    rows: Sequence[Mapping[str, Any]],
    fluents: Sequence[PairFluent],
    place_index: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """The location fluents' initial matrices from the rows of ``entities.csv``: for each
    location fluent, its column of place labels, one per entity. An unknown place is a
    :class:`WorldError`."""
    found: dict[str, np.ndarray] = {}
    for fluent in fluents:
        if not fluent.total:
            continue
        choices = np.zeros(len(rows), dtype=np.intp)
        for i, row in enumerate(rows):
            place = str(row[fluent.label])
            if place not in place_index:
                raise WorldError(
                    f"the entities table puts {row.get('label')} at {place} on {fluent.label}, "
                    f"which is not a place of the definition"
                )
            choices[i] = place_index[place]
        found[fluent.label] = places_matrix(choices, len(place_index))
    return found


__all__ = [
    "DEFAULT_HABITAT_STRENGTH",
    "LOCATION_SYMMETRY",
    "draw_places",
    "habitat_weights",
    "location_columns",
    "location_fluent",
    "location_matrices_from_columns",
    "places_matrix",
    "with_location_fluents",
]
