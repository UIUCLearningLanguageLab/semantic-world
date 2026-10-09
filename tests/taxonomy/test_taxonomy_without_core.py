"""The taxonomy generator runs without the Rust core.

``semantic_world`` loads its Rust extension lazily, so ``semantic_world.taxonomy`` imports and
runs with only NumPy, polars, and PyYAML. These tests run the generator in a subprocess whose
``semantic_world._core`` import is blocked, and once as the plain command line the guide gives.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

BLOCK_CORE = (
    "import sys\nsys.modules['semantic_world._core'] = None  # makes the import raise ImportError\n"
)


def _run(script: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO / "python")}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO, env=env
    )


def test_taxonomy_imports_and_runs_with_the_core_unavailable(tmp_path: Path) -> None:
    script = BLOCK_CORE + (
        "import semantic_world\n"
        "import semantic_world.taxonomy\n"
        "from semantic_world.taxonomy import generate, load_config\n"
        "result = generate(load_config('data/taxonomy/tiny.yaml'))\n"
        f"result.write({str(tmp_path / 'run')!r})\n"
        "print('instances', len(result.instances))\n"
        "try:\n"
        "    semantic_world.World\n"
        "except ImportError:\n"
        "    print('core unavailable')\n"
    )
    run = _run(script)
    assert run.returncode == 0, run.stderr
    assert "instances 12" in run.stdout
    assert "core unavailable" in run.stdout
    assert (tmp_path / "run" / "base.csv").exists()
    assert not (tmp_path / "run" / "instances.csv").exists()
    assert (tmp_path / "run" / "rule_matrices.json").exists()
    assert (tmp_path / "run" / "derived" / "static_features.csv").exists()


def test_command_line_runs_with_the_core_unavailable(tmp_path: Path) -> None:
    script = BLOCK_CORE + (
        "from semantic_world.taxonomy.__main__ import main\n"
        f"raise SystemExit(main(['data/taxonomy/tiny.yaml', '--out', {str(tmp_path / 'cli')!r}]))\n"
    )
    run = _run(script)
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith(f"wrote {tmp_path / 'cli'}")


def test_command_line_with_pythonpath(tmp_path: Path) -> None:
    """The exact command from the guide, with the repository's ``python`` folder on the path."""
    env = {**os.environ, "PYTHONPATH": str(REPO / "python")}
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_world.taxonomy",
            "data/taxonomy/tiny.yaml",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "out" / "summary.yaml").exists()


def test_core_attributes_still_load_lazily() -> None:
    """With the core available, the lazy attributes resolve as before."""
    import semantic_world

    assert isinstance(semantic_world.__version__, str)
    assert semantic_world.World is semantic_world.engine.World
    assert semantic_world.make is semantic_world.engine.make
    assert "World" in dir(semantic_world)
    with pytest.raises(AttributeError):
        getattr(semantic_world, "no_such_thing")  # noqa: B009
