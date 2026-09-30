"""Stage 1 acceptance tests: the four schedule forms give the right per-level values."""

from __future__ import annotations

import math

import pytest

from semantic_world.taxonomy import ConfigError, config_from_mapping
from semantic_world.taxonomy.config import Range, resolve_branching, resolve_schedule


def test_constant_schedule() -> None:
    assert resolve_schedule(0.3, 4, "x.yaml", "f") == (0.3, 0.3, 0.3, 0.3)
    assert resolve_schedule(1, 2, "x.yaml", "f") == (1.0, 1.0)


def test_linear_schedule() -> None:
    values = resolve_schedule({"schedule": "linear", "start": 0.1, "end": 0.3}, 3, "x.yaml", "f")
    assert values == pytest.approx((0.1, 0.2, 0.3))
    values = resolve_schedule({"schedule": "linear", "start": 1.0, "end": 0.0}, 5, "x.yaml", "f")
    assert values == pytest.approx((1.0, 0.75, 0.5, 0.25, 0.0))


def test_linear_schedule_with_depth_one_is_the_start() -> None:
    assert resolve_schedule({"schedule": "linear", "start": 0.1, "end": 0.3}, 1, "x.yaml", "f") == (
        0.1,
    )


def test_exponential_schedule() -> None:
    start, asymptote, rate = 0.9, 0.5, 0.7
    values = resolve_schedule(
        {"schedule": "exponential", "start": start, "asymptote": asymptote, "rate": rate},
        4,
        "x.yaml",
        "f",
    )
    expected = tuple(
        asymptote + (start - asymptote) * math.exp(-rate * (level - 1)) for level in range(1, 5)
    )
    assert values == pytest.approx(expected)
    assert values[0] == pytest.approx(start)
    assert values[1] < values[0]
    assert values[3] > asymptote


def test_exponential_schedule_with_zero_rate_is_constant() -> None:
    values = resolve_schedule(
        {"schedule": "exponential", "start": 0.4, "asymptote": 0.1, "rate": 0}, 3, "x.yaml", "f"
    )
    assert values == pytest.approx((0.4, 0.4, 0.4))


def test_list_schedule() -> None:
    values = resolve_schedule({"schedule": "list", "values": [0.1, 0.2, 1]}, 3, "x.yaml", "f")
    assert values == (0.1, 0.2, 1.0)
    assert all(isinstance(v, float) for v in values)


@pytest.mark.parametrize(
    ("value", "field"),
    [
        ({"schedule": "list", "values": [0.1, 0.2]}, "f.values"),
        ({"schedule": "list", "values": 0.1}, "f.values"),
        ({"schedule": "list"}, "f.values"),
        ({"schedule": "linear", "start": 0.1}, "f.end"),
        ({"schedule": "linear", "start": 0.1, "end": 0.2, "extra": 1}, "f.extra"),
        ({"schedule": "exponential", "start": 0.1, "asymptote": 0.2}, "f.rate"),
        ({"schedule": "exponential", "start": 0.1, "asymptote": 0.2, "rate": -1}, "f.rate"),
        ({"schedule": "step"}, "f.schedule"),
        ({"start": 0.1, "end": 0.2}, "f.schedule"),
        ("0.1", "f"),
        ([0.1, 0.2, 0.3], "f"),
        (None, "f"),
    ],
)
def test_schedule_errors_name_the_field(value: object, field: str) -> None:
    with pytest.raises(ConfigError) as info:
        resolve_schedule(value, 3, "x.yaml", "f")
    assert info.value.field == field
    assert str(info.value).startswith(f"x.yaml: {field}: ")


def test_schedules_resolve_per_level_in_a_configuration() -> None:
    config = config_from_mapping(
        {
            "taxonomy": {"depth": 4},
            "inheritance": {
                "proportion_defining": {"schedule": "linear", "start": 0.3, "end": 0.0},
                "proportion_characteristic": {"schedule": "list", "values": [0.2, 0.4, 0.6, 0.8]},
                "characteristic_probability": {
                    "schedule": "exponential",
                    "start": 1.0,
                    "asymptote": 0.5,
                    "rate": 1.0,
                },
            },
        }
    )
    inheritance = config.inheritance
    assert inheritance.proportion_defining == pytest.approx((0.3, 0.2, 0.1, 0.0))
    assert inheritance.proportion_characteristic == (0.2, 0.4, 0.6, 0.8)
    assert inheritance.characteristic_probability[0] == pytest.approx(1.0)
    assert inheritance.characteristic_probability[3] == pytest.approx(0.5 + 0.5 * math.exp(-3))
    resolved = config.resolved()["inheritance"]
    assert resolved["proportion_characteristic"] == {
        "schedule": "list",
        "values": [0.2, 0.4, 0.6, 0.8],
    }
    assert resolved["proportion_defining"]["values"] == pytest.approx([0.3, 0.2, 0.1, 0.0])


def test_branching_forms() -> None:
    assert resolve_branching(3, 3, "x.yaml", "taxonomy.branching") == (Range(3, 3), Range(3, 3))
    assert resolve_branching([2, 4], 3, "x.yaml", "taxonomy.branching") == (
        Range(2, 4),
        Range(2, 4),
    )
    assert resolve_branching([2, 4], 1, "x.yaml", "taxonomy.branching") == ()
    per_level = resolve_branching(
        {"schedule": "list", "values": [2, [3, 5], 1]}, 4, "x.yaml", "taxonomy.branching"
    )
    assert per_level == (Range(2, 2), Range(3, 5), Range(1, 1))
    assert [r.resolved() for r in per_level] == [2, [3, 5], 1]


@pytest.mark.parametrize(
    ("value", "depth", "field"),
    [
        (0, 3, "taxonomy.branching"),
        ([0, 2], 3, "taxonomy.branching"),
        ([3, 2], 3, "taxonomy.branching"),
        ([2, 3, 4], 3, "taxonomy.branching"),
        ([2.5, 3], 3, "taxonomy.branching"),
        ("2", 3, "taxonomy.branching"),
        ({"schedule": "list", "values": [2]}, 3, "taxonomy.branching.values"),
        ({"schedule": "list", "values": [2, 0]}, 3, "taxonomy.branching.values.1"),
        ({"schedule": "linear", "start": 2, "end": 4}, 3, "taxonomy.branching.schedule"),
        ({"values": [2, 3]}, 3, "taxonomy.branching.schedule"),
    ],
)
def test_branching_errors_name_the_field(value: object, depth: int, field: str) -> None:
    with pytest.raises(ConfigError) as info:
        resolve_branching(value, depth, "x.yaml", "taxonomy.branching")
    assert info.value.field == field
