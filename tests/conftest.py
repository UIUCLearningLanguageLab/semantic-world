from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def smoke_config() -> Path:
    return REPO / "data" / "experiments" / "m1_smoke.yaml"
