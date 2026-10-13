"""Stage b2: the world's comparisons of scalars and numeric fluents (``docs/specs/
WORLD_AND_LANGUAGE.md``, "Phase (b): the world", "Comparisons at a time").

The acceptance tests of the stage: every comparison (``GREATER``, ``ABOVE``, ``BELOW``,
``INCREASE``, ``DECREASE``) agrees with direct arithmetic on the values, ties included (false
both ways) and category means; the comparison-class statistics of ``derived/
comparison_classes.csv`` agree with a recomputation from ``base.csv``, with the runtime's own
computation from the leaf labels, and with the corpus's computation of the class of a pole
adjective; and the brute-force evaluator judges the same comparisons the same way.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from semantic_world.world.comparisons import COMPARISON_CLASSES_FILE, comparison_classes_frame
from semantic_world.world.config import load_config
from semantic_world.world.definition import RuntimeDefinition
from semantic_world.world.derived import load_derived_csv
from semantic_world.world.errors import WorldError
from semantic_world.world.fixtures import BruteForce, fixture_inputs
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.runtime import (
    ENTITY,
    RANGE,
    Mean,
    State,
    above,
    apply,
    below,
    categories,
    comparison_classes,
    decrease,
    greater,
    increase,
    initial_state,
    members,
    value_of,
)

DATA = Path("data/world")
SLOW = pytest.mark.slow
NAMES = ["tiny", pytest.param("default", marks=SLOW)]


@pytest.fixture(scope="module")
def worlds() -> dict[str, WorldResult]:
    return {}


def world_named(worlds: dict[str, WorldResult], name: str) -> WorldResult:
    if name not in worlds:
        worlds[name] = define(load_config(DATA / f"{name}.yaml"))
    return worlds[name]


@pytest.fixture(scope="module")
def tiny(worlds) -> WorldResult:
    return world_named(worlds, "tiny")


@pytest.fixture(scope="module")
def runtime(tiny: WorldResult) -> RuntimeDefinition:
    return tiny.definition.runtime()


# ---------------------------------------------------------------------------------------------
# Comparison classes
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_comparison_classes_agree_with_a_recomputation_from_base_csv(
    worlds, name: str, tmp_path: Path
) -> None:
    """``derived/comparison_classes.csv`` holds, for ``ENTITY`` and every category of the tree
    and every scalar, the count, the mean, and the population standard deviation of the scalar
    over the instances below the category, recomputed here from ``taxonomy/base.csv`` and
    ``taxonomy/tree.csv`` alone (to the file's six decimals)."""
    world = world_named(worlds, name)
    folder = world.write(tmp_path / name)
    frame = load_derived_csv(folder / "derived", COMPARISON_CLASSES_FILE, world.rule_set_id)
    assert frame.columns == ["category", "scalar", "instances", "mean", "sd"]
    with (folder / "taxonomy" / "base.csv").open(encoding="utf-8", newline="") as handle:
        base = list(csv.DictReader(handle))
    with (folder / "taxonomy" / "tree.csv").open(encoding="utf-8", newline="") as handle:
        tree = list(csv.DictReader(handle))
    parent = {row["label"]: row["parent"] or None for row in tree}
    scalars = [c for c in base[0] if c.startswith("SCALARDIM.")]

    def below(category: str) -> list[dict]:
        if category == ENTITY:
            return base
        found = []
        for row in base:
            node: str | None = row["leaf"]
            while node is not None:
                if node == category:
                    found.append(row)
                    break
                node = parent[node]
        return found

    rows = frame.to_dicts()
    assert [r["category"] for r in rows][: len(scalars)] == [ENTITY] * len(scalars)
    expected_categories = [ENTITY] + [row["label"] for row in tree]
    assert [r["category"] for r in rows] == [c for c in expected_categories for _ in scalars]
    for row in rows:
        values = np.array([float(r[row["scalar"]]) for r in below(row["category"])])
        assert row["instances"] == len(values)
        assert abs(row["mean"] - values.mean()) < 1e-5
        assert abs(row["sd"] - values.std()) < 1e-5
    # the in-memory frame, at full precision
    exact = comparison_classes_frame(world.taxonomy).to_dicts()
    for row in exact:
        values = np.array([float(r[row["scalar"]]) for r in below(row["category"])])
        assert abs(row["mean"] - values.mean()) < 1e-6 and abs(row["sd"] - values.std()) < 1e-6


@pytest.mark.parametrize("name", NAMES)
def test_the_runtime_computes_the_same_classes_from_the_leaf_labels(worlds, name: str) -> None:
    world = world_named(worlds, name)
    definition = world.definition.runtime()
    table = comparison_classes(definition)
    frame = comparison_classes_frame(world.taxonomy).to_dicts()
    assert set(categories(definition)) == {row["category"] for row in frame} - {ENTITY}
    assert len(table) == len(frame)
    for row in frame:
        count, mean, sd = table[(row["category"], row["scalar"])]
        assert count == row["instances"]
        assert mean == row["mean"] and sd == row["sd"]
    for category in (ENTITY, *categories(definition)):
        rows = members(definition, category)
        assert rows.size == table[(category, definition.scalars[0])][0]
    with pytest.raises(WorldError, match="not a category"):
        members(definition, "CATEGORY.9.9")


@pytest.mark.parametrize("name", NAMES)
def test_the_corpus_computes_the_same_classes(worlds, name: str) -> None:
    """The corpus's own computation of a pole adjective's comparison class (``Truth.pole_mask``
    and ``_class_scalar``, with ``THING`` for every entity) gives the mean and the standard
    deviation of ``comparison_classes.csv`` for every category and scalar; stage b3 moves the
    corpus to the file."""
    from semantic_world.corpus import Streams, build_lexicon
    from semantic_world.corpus import load_config as corpus_config
    from semantic_world.corpus.propositions import Truth
    from semantic_world.corpus.world import THING, World

    world = world_named(worlds, name)
    config = corpus_config(Path("data/corpus") / f"{name}.yaml")
    corpus_world = World(world)
    build_lexicon(config, corpus_world, Streams(config.seed))
    truth = Truth(config, corpus_world)
    table = comparison_classes(world.definition.runtime())
    for (category, scalar), (count, mean, sd) in table.items():
        rows = corpus_world.below(THING if category == ENTITY else category)
        assert rows.size == count
        _, corpus_mean, corpus_sd = truth.pole_mask(f"{scalar}.HIGH", rows)
        assert corpus_mean == mean and corpus_sd == sd


# ---------------------------------------------------------------------------------------------
# The comparisons against direct arithmetic
# ---------------------------------------------------------------------------------------------


def _value(runtime: RuntimeDefinition, dimension: str, x, state: State | None) -> float:
    column = (
        runtime.scalar_values[:, runtime.scalars.index(dimension)]
        if dimension in runtime.scalars
        else state.numeric[:, runtime.numeric_index(dimension)]
    )
    if isinstance(x, Mean):
        return float(column[members(runtime, x.category)].mean())
    return float(column[x])


def test_greater_agrees_with_direct_arithmetic_and_ties_are_false_both_ways(
    runtime: RuntimeDefinition,
) -> None:
    rng = np.random.default_rng(1)
    arguments = list(range(runtime.entity_count)) + [
        Mean(c) for c in (ENTITY, *categories(runtime))
    ]
    checked = ties = 0
    for _ in range(300):
        x, y = (arguments[int(i)] for i in rng.integers(len(arguments), size=2))
        for scalar in runtime.scalars:
            a, b = _value(runtime, scalar, x, None), _value(runtime, scalar, y, None)
            assert greater(runtime, scalar, x, y) == (a > b)
            assert greater(runtime, scalar, y, x) == (b > a)
            assert value_of(runtime, scalar, x) == a
            checked += 1
    # ties: an entity against itself, a leaf category's mean against its members' mean, and
    # two entities given the same numeric value
    for scalar in runtime.scalars:
        for i in range(runtime.entity_count):
            assert not greater(runtime, scalar, i, i)
            ties += 1
        for category in categories(runtime):
            twin = Mean(category)
            assert not greater(runtime, scalar, twin, Mean(category))
    state = initial_state(runtime)
    numeric = np.array(state.numeric, copy=True)
    numeric[0, :] = numeric[1, :]
    tied = State(state.values, numeric)
    for fluent in runtime.numeric_fluents:
        assert not greater(runtime, fluent, 0, 1, tied) and not greater(runtime, fluent, 1, 0, tied)
        assert greater(runtime, fluent, 0, 2, tied) == (
            tied.numeric[0, 0] > tied.numeric[2, 0]
        ) or (runtime.numeric_index(fluent) != 0)
        ties += 1
    assert checked > 0 and ties > 0
    with pytest.raises(WorldError, match="needs a state"):
        greater(runtime, runtime.numeric_fluents[0], 0, 1)
    with pytest.raises(WorldError, match="not a scalar dimension"):
        greater(runtime, "PROPERTY.1", 0, 1)


def test_above_and_below_agree_with_the_class_mean_and_standard_deviation(
    runtime: RuntimeDefinition,
) -> None:
    rng = np.random.default_rng(2)
    classes = comparison_classes(runtime)
    arguments = list(range(runtime.entity_count)) + [
        Mean(c) for c in (ENTITY, *categories(runtime))
    ]
    seen = {True: 0, False: 0}
    for _ in range(400):
        x = arguments[int(rng.integers(len(arguments)))]
        category = (ENTITY, *categories(runtime))[int(rng.integers(len(categories(runtime)) + 1))]
        z = float(rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, 2.0]))
        for scalar in runtime.scalars:
            value = _value(runtime, scalar, x, None)
            _, mean, sd = classes[(category, scalar)]
            expected_above = sd > 0 and value >= mean + z * sd
            expected_below = sd > 0 and value <= mean - z * sd
            assert above(runtime, scalar, x, category, z) == expected_above
            assert below(runtime, scalar, x, category, z) == expected_below
            seen[expected_above] += 1
    assert seen[True] > 0 and seen[False] > 0
    # a class with no spread has nothing above or below it: a leaf with one instance would;
    # here, an entity against the class of its own value alone is checked through z = 0
    for scalar in runtime.scalars:
        for i in range(runtime.entity_count):
            _, mean, sd = classes[(ENTITY, scalar)]
            assert above(runtime, scalar, i, ENTITY, 0.0) == (
                runtime.scalar_values[i, 0] >= mean
            ) or (runtime.scalars.index(scalar) != 0)
    with pytest.raises(WorldError, match="takes a comparison class"):
        above(runtime, runtime.scalars[0], 0, RANGE, 1.0)


def test_numeric_comparisons_at_a_time_agree_with_direct_arithmetic(
    tiny: WorldResult, runtime: RuntimeDefinition
) -> None:
    """``GREATER`` between entities at a time point, ``ABOVE`` and ``BELOW`` against ``RANGE``
    (the fraction of the way through the range), and ``INCREASE`` and ``DECREASE`` over a step,
    on states of the tiny world and the states after random steps."""
    from semantic_world.world.runtime import legal_bindings

    rng = np.random.default_rng(3)
    state = initial_state(runtime)
    performable = [et.label for et in runtime.event_types if not et.is_category]
    seen_increase = seen_decrease = seen_above = seen_below = 0
    for _ in range(60):
        events = [
            (label, b) for label in performable for b in legal_bindings(runtime, state, label)
        ]
        chosen = []
        from semantic_world.world.runtime import Event, interferes

        for i in rng.permutation(len(events))[:3]:
            event = Event(*events[int(i)])
            if all(not interferes(runtime, event, other) for other in chosen):
                chosen.append(event)
        after = apply(runtime, state, chosen)
        for fluent in runtime.numeric_fluents:
            j = runtime.numeric_index(fluent)
            low, high = runtime.numeric_ranges[fluent]
            for x in range(runtime.entity_count):
                y = int(rng.integers(runtime.entity_count))
                assert greater(runtime, fluent, x, y, state) == (
                    state.numeric[x, j] > state.numeric[y, j]
                )
                q = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
                point = low + q * (high - low)
                assert above(runtime, fluent, x, RANGE, q, state) == (state.numeric[x, j] >= point)
                assert below(runtime, fluent, x, RANGE, q, state) == (state.numeric[x, j] <= point)
                seen_above += above(runtime, fluent, x, RANGE, q, state)
                seen_below += below(runtime, fluent, x, RANGE, q, state)
                assert increase(runtime, fluent, x, state, after) == (
                    after.numeric[x, j] > state.numeric[x, j]
                )
                assert decrease(runtime, fluent, x, state, after) == (
                    after.numeric[x, j] < state.numeric[x, j]
                )
                seen_increase += increase(runtime, fluent, x, state, after)
                seen_decrease += decrease(runtime, fluent, x, state, after)
            # a category mean at a time point
            for category in categories(runtime):
                rows = members(runtime, category)
                assert value_of(runtime, fluent, Mean(category), state) == float(
                    state.numeric[rows, j].mean()
                )
        state = after
    assert seen_increase > 0 and seen_above > 0 and seen_below > 0
    assert seen_decrease > 0 or all(f.drift >= 0 for f in tiny.numeric)
    with pytest.raises(WorldError, match="takes RANGE"):
        above(runtime, runtime.numeric_fluents[0], 0, ENTITY, 1.0, state)


def test_the_brute_force_evaluator_judges_the_same_comparisons(
    tiny: WorldResult, runtime: RuntimeDefinition
) -> None:
    record, entities, _ = fixture_inputs(tiny.definition)
    brute = BruteForce(record, entities)
    rng = np.random.default_rng(4)
    state = initial_state(runtime)
    numeric = state.numeric_values(runtime)
    labels = runtime.entity_labels
    arguments: list = list(range(runtime.entity_count)) + [
        Mean(c) for c in (ENTITY, *categories(runtime))
    ]

    def spell(x):
        return {"mean": x.category} if isinstance(x, Mean) else labels[x]

    for _ in range(200):
        x, y = (arguments[int(i)] for i in rng.integers(len(arguments), size=2))
        scalar = runtime.scalars[0]
        entry = {"kind": "greater", "dimension": scalar, "x": spell(x), "y": spell(y)}
        assert brute.compare(entry) == greater(runtime, scalar, x, y)
        category = (ENTITY, *categories(runtime))[int(rng.integers(len(categories(runtime)) + 1))]
        z = float(rng.choice([0.25, 0.5, 1.0]))
        for kind, judge in (("above", above), ("below", below)):
            entry = {"kind": kind, "dimension": scalar, "x": spell(x), "class": category, "z": z}
            assert brute.compare(entry) == judge(runtime, scalar, x, category, z)
        fluent = runtime.numeric_fluents[int(rng.integers(len(runtime.numeric_fluents)))]
        if not isinstance(x, Mean) and not isinstance(y, Mean):
            entry = {"kind": "greater", "dimension": fluent, "x": labels[x], "y": labels[y]}
            assert brute.compare(entry, numeric) == greater(runtime, fluent, x, y, state)
            for kind, judge in (("above", above), ("below", below)):
                entry = {"kind": kind, "dimension": fluent, "x": labels[x], "class": RANGE, "z": z}
                assert brute.compare(entry, numeric) == judge(runtime, fluent, x, RANGE, z, state)
