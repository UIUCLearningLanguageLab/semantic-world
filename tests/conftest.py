"""The shared fixtures and the two tiers of the tests.

``pytest`` runs the fast tier: every test without the ``slow`` marker. ``pytest --slow`` runs the
slow tier alone: the tests marked ``slow``, which need a default-scale run (the default world of
``data/world/default.yaml``, with its 1,000 statistics episodes, or a corpus on it) or take more
than a few seconds. ``pytest --all`` runs both tiers, as the full check list does. A test of one
tier is deselected in the other, and the summary counts the deselected tests.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SLOW = pytest.mark.slow
"""The marker of the slow tier. ``pytest.param("default", marks=SLOW)`` marks one case of a
parametrized test."""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run the slow tier alone: the tests that need default-scale runs",
    )
    parser.addoption(
        "--all", action="store_true", default=False, help="run both tiers, fast and slow"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: needs a default-scale run, or takes more than a few seconds; deselected by plain "
        "pytest, run by pytest --slow or pytest --all",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--all"):
        return
    want_slow = bool(config.getoption("--slow"))
    kept, dropped = [], []
    for item in items:
        is_slow = item.get_closest_marker("slow") is not None
        (kept if is_slow == want_slow else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


@pytest.fixture
def smoke_config() -> Path:
    return REPO / "data" / "experiments" / "m1_smoke.yaml"
