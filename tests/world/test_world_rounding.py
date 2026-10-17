"""Stage c1, item 1: every real the world draws and stores is rounded to 4 decimals when drawn
(Jon's ruling of October 16, 2026, WM.60), so that the same seed gives the same world on every
machine. The drawn reals are a numeric fluent's drift, each entity's initial value of a numeric
fluent (uniform or beta, when the world is generated and when an episode redraws its initial
state), and a numeric effect's constant; the scalar thresholds and the comparison margins were
rounded already. The tiny world on five seeds is in the fast tier, the default world in the slow
tier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.episodes import EpisodeGenerator, EpisodeSettings, Relatedness
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.numeric import DECIMALS, InitialDistribution, drawn

DATA = Path("data/world")
SLOW = pytest.mark.slow
SEEDS = range(1, 6)
CASES = [("tiny", seed) for seed in SEEDS] + [
    pytest.param("default", seed, marks=SLOW) for seed in SEEDS
]


def _rounded(value: float) -> bool:
    return float(value) == round(float(value), DECIMALS)


def _world(name: str, seed: int) -> WorldResult:
    return define(load_config(DATA / f"{name}.yaml", seed=seed))


@pytest.mark.parametrize(("name", "seed"), CASES)
def test_every_drawn_real_of_the_world_equals_itself_rounded(name: str, seed: int) -> None:
    world = _world(name, seed)
    numeric = world.numeric
    assert len(numeric) > 0
    for fluent in numeric:
        assert _rounded(fluent.drift), (name, seed, fluent.label, fluent.drift)
    assert all(_rounded(v) for v in numeric.initial_values.ravel()), (name, seed)
    amounts = [
        effect.amount
        for et in world.event_types.event_types
        for effect in et.effects
        if effect.numeric
    ]
    assert amounts, (name, seed)
    assert all(_rounded(a) for a in amounts), (name, seed)
    # the same holds of the definition's record, which both runtimes read
    record = world.definition.record()
    for symbol in record["symbols"]:
        if symbol["kind"] == "numeric_fluent":
            assert _rounded(symbol["drift"])
    for literal in record["literals"]:
        for key in ("threshold", "low", "high"):
            if literal.get(key) is not None:
                assert _rounded(literal[key]), literal
    for et in record["event_types"]:
        for effect in et["effects"]:
            if "amount" in effect:
                assert _rounded(effect["amount"]), effect
        for lit in et["precondition"]["literals"]:
            if "threshold" in lit:
                assert _rounded(lit["threshold"]), lit


def test_the_beta_and_uniform_draws_are_rounded_and_inside_the_range() -> None:
    rng = np.random.default_rng(3)
    for distribution in (
        InitialDistribution("uniform"),
        InitialDistribution("beta", mean=0.3, concentration=4.0),
    ):
        for low, high in ((0.0, 1.0), (0.00044, 0.33333), (-2.5, 7.25)):
            values = distribution.draw(rng, low, high, 500)
            assert values.dtype == np.float64
            assert all(_rounded(v) for v in values)
            assert bool(((values >= low) & (values <= high)).all())
    assert drawn(0.39696669888439323) == 0.397
    assert drawn(0.005553129001236879) == 0.0056


def test_a_redrawn_initial_state_draws_rounded_numeric_values() -> None:
    world = define(load_config(DATA / "tiny.yaml"))
    definition = world.definition.runtime()
    generator = EpisodeGenerator(
        definition,
        Relatedness.from_statics(world.statics, definition),
        EpisodeSettings(initial="redraw"),
    )
    histories = generator.run(7, 20, "episodes")
    assert any(h.initial_numeric for h in histories)
    for history in histories:
        for values in history.initial_numeric.values():
            assert all(_rounded(v) for v in values.values()), history.label


def test_a_fixed_initial_value_is_the_configurations_own() -> None:
    config = config_from_mapping(
        {
            "name": "fixed",
            "taxonomy": {"config": "data/taxonomy/tiny_relations.yaml"},
            "numeric_fluents": {
                "count": 1,
                "fluents": {"NUMFL.1": {"initial": {"kind": "fixed", "value": 0.123456789}}},
            },
            "event_types": {"unary": {"count": 2}, "binary": None},
        },
        source="<test>",
    )
    world = define(config)
    assert set(world.numeric.initial_values[:, 0].tolist()) == {0.123456789}
