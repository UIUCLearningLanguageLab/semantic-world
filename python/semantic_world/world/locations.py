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


def _entropy_bits(counts: np.ndarray) -> float:
    """The entropy, in bits, of a distribution given by counts; 0 for an empty one."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _leaf_members(tree, instances) -> list[tuple[str, np.ndarray]]:
    """Each leaf category of a taxonomy, in tree order, with the bool mask of its instances."""
    leaf_of = np.array([tree.categories[k].label for k in instances.leaf_index])
    return [(leaf.label, leaf_of == leaf.label) for leaf in tree.leaves]


def expected_leaf_habitats(weights: np.ndarray, taxonomy, place_taxonomy) -> dict[str, Any]:
    """The expected habitats at the leaf level, from the habitat weights rather than from one
    draw (Jon's ruling WM.69, the stage c3b follow-up of October 28, 2026): each entity's
    probability of each place is its weight over the sum of its row, and an entity leaf
    category's expected distribution over the place leaf categories is the mean, over its
    entities, of their probabilities summed over each place leaf category's places. Returns,
    for the entity leaf categories, the median and the largest ratio of a category's favorite
    place leaf category (the one with its largest expected share) to that category's uniform
    share (its places over the places), with the pair that gives the largest; the mean of the
    categories' expected entropies over the place leaf categories, in bits; and the entropy of
    the uniform draw grouped by place leaf category (``uniform_entropy``: the mean entropy at
    strength 0, and the largest the mean can be when the place leaves are of one size)."""
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    place_leaves = _leaf_members(place_taxonomy.tree, place_taxonomy.instances)
    members = np.array([mask for _, mask in place_leaves], dtype=float)  # (place leaves, places)
    uniform = members.sum(axis=1) / weights.shape[1]
    ratios: list[tuple[float, str, str]] = []
    entropies: list[float] = []
    for label, entities in _leaf_members(taxonomy.tree, taxonomy.instances):
        if not entities.any():
            continue
        distribution = (probabilities[entities] @ members.T).mean(axis=0)
        favorite = int(np.argmax(distribution))
        ratios.append(
            (float(distribution[favorite] / uniform[favorite]), label, place_leaves[favorite][0])
        )
        entropies.append(_entropy_bits(distribution))
    largest = max(ratios) if ratios else (0.0, "", "")
    return {
        "level": int(max((c.level for c in taxonomy.tree.leaves), default=0)),
        "favorite_ratio": {
            "median": round(float(np.median([r for r, _, _ in ratios])), 6) if ratios else 0.0,
            "largest": round(largest[0], 6),
            "category": largest[1],
            "place_category": largest[2],
        },
        "entropy": {
            "mean": round(float(np.mean(entropies)), 6) if entropies else 0.0,
            "uniform": round(_entropy_bits(uniform), 6),
        },
    }


def habitat_stats(
    pair_fluents: PairFluents,
    taxonomy,
    place_taxonomy,
    requirements: Iterable[np.ndarray] = (),
    strength: float = DEFAULT_HABITAT_STRENGTH,
) -> dict[str, Any]:
    """The concentration of habitats (stage c3b, ``places.habitats`` of ``world_stats.yaml``),
    per location fluent in label order: for each place category (in tree order), the share of
    each entity superordinate's entities that start there (``by_place_category``); the entropy,
    in bits, of the initial places of each entity category's entities over the places
    (``entropy_by_category``, in tree order), its mean over the categories of each level of the
    entity tree (``mean_entropy_by_level``), and the entropy of the uniform draw over the places
    (``max_entropy``), so that a flat habitat reads as the maximum and a concentrated one as
    less; and, as the figure review point 2 asks for, the largest absolute deviation of a share
    from the uniform share (``largest_deviation``: the place category, the superordinate, the
    share, and the uniform share, the place category's number of places over the places). Under
    WM.69 also the expected effect sizes at the leaf level (``expected_by_leaf``, from
    :func:`expected_leaf_habitats` on the weights of the ``requirements`` and the ``strength``),
    the same for every location fluent, because every one draws from the same weights."""
    tree, instances = taxonomy.tree, taxonomy.instances
    place_tree, place_instances = place_taxonomy.tree, place_taxonomy.instances
    superordinates = [c for c in tree.at_level(1)]
    requirements = list(requirements)
    found: dict[str, Any] = {}
    for fluent, matrix in zip(pair_fluents.fluents, pair_fluents.initial_pairs, strict=True):
        if not fluent.total:
            continue
        places = np.argmax(matrix, axis=1)
        place_count = matrix.shape[1]
        by_place: dict[str, dict[str, float]] = {}
        largest: dict[str, Any] | None = None
        for place_category in place_tree.categories:
            members = np.array(
                [
                    place_tree.ancestor_label(place_tree.categories[k], place_category.level)
                    == place_category.label
                    for k in place_instances.leaf_index
                ],
                dtype=bool,
            )
            uniform = float(members.sum()) / place_count if place_count else 0.0
            row: dict[str, float] = {}
            for superordinate in superordinates:
                entities = np.array(
                    [
                        tree.ancestor_label(tree.categories[k], 1) == superordinate.label
                        for k in instances.leaf_index
                    ],
                    dtype=bool,
                )
                here = members[places[entities]]
                share = float(here.mean()) if entities.any() else 0.0
                row[superordinate.label] = round(share, 6)
                deviation = abs(share - uniform)
                if largest is None or deviation > largest["deviation"]:
                    largest = {
                        "place_category": place_category.label,
                        "superordinate": superordinate.label,
                        "share": round(share, 6),
                        "uniform_share": round(uniform, 6),
                        "deviation": round(deviation, 6),
                        "entities": int(entities.sum()),
                    }
            by_place[place_category.label] = row
        entropy: dict[str, float] = {}
        by_level: dict[int, list[float]] = {}
        for category in tree.categories:
            entities = np.array(
                [
                    tree.ancestor_label(tree.categories[k], category.level) == category.label
                    for k in instances.leaf_index
                ],
                dtype=bool,
            )
            counts = np.bincount(places[entities], minlength=place_count)
            bits = _entropy_bits(counts)
            entropy[category.label] = round(bits, 6)
            by_level.setdefault(category.level, []).append(bits)
        found[fluent.label] = {
            "by_place_category": by_place,
            "largest_deviation": largest,
            "entropy_by_category": entropy,
            "mean_entropy_by_level": {
                level: round(float(np.mean(values)), 6) for level, values in by_level.items()
            },
            "max_entropy": round(float(np.log2(place_count)), 6) if place_count else 0.0,
            "expected_by_leaf": expected_leaf_habitats(
                habitat_weights(requirements, matrix.shape, strength), taxonomy, place_taxonomy
            ),
        }
    return found


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
    "expected_leaf_habitats",
    "habitat_stats",
    "habitat_weights",
    "location_columns",
    "location_fluent",
    "location_matrices_from_columns",
    "places_matrix",
    "with_location_fluents",
]
