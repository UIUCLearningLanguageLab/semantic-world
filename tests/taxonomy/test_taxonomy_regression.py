"""The regression test of docs/specs/TAXONOMY_RELATIONS.md.

With scalar dimensions set to 0 and relations turned off, the generator must produce exactly the
output it produced before the relations work began. The folders under ``golden/`` were written by
the stage 6 generator for every example configuration. Every output file except ``config.yaml``
must be byte-identical, and ``config.yaml`` may differ only by added keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from semantic_world.taxonomy import generate, load_config

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
GOLDEN = Path(__file__).resolve().parent / "golden"

# Provenance values that legitimately change from run to run.
VOLATILE = {
    ("provenance", "git_commit"),
    ("provenance", "git_dirty"),
    ("provenance", "package_version"),
}


def golden_configurations() -> list[str]:
    return sorted(p.name for p in GOLDEN.iterdir() if p.is_dir())


def test_every_base_example_configuration_has_a_golden_folder() -> None:
    """Configurations with scalars off and verbs null are the ones the rule protects."""
    examples = []
    for path in sorted(DATA.glob("*.yaml")):
        config = load_config(path)
        if config.scalars.count == 0 and config.verbs is None:
            examples.append(path.stem)
    assert golden_configurations() == examples


def assert_only_added_keys(golden: Any, current: Any, path: tuple[str, ...] = ()) -> None:
    """Every key of ``golden`` is in ``current`` with an equal value; ``current`` may add keys."""
    if isinstance(golden, dict):
        assert isinstance(current, dict), path
        for key, value in golden.items():
            if (*path, key) in VOLATILE:
                continue
            assert key in current, f"{'.'.join(map(str, (*path, key)))} was removed"
            assert_only_added_keys(value, current[key], (*path, str(key)))
    else:
        assert golden == current, ".".join(path)


@pytest.mark.parametrize("name", golden_configurations())
def test_outputs_are_unchanged_with_scalars_off_and_relations_off(
    name: str, tmp_path: Path
) -> None:
    config = load_config(DATA / f"{name}.yaml")
    folder = generate(config).write(tmp_path / name)
    golden = GOLDEN / name
    assert sorted(p.name for p in folder.iterdir()) == sorted(p.name for p in golden.iterdir())
    for file in sorted(golden.iterdir()):
        current = folder / file.name
        if file.name == "config.yaml":
            assert_only_added_keys(
                yaml.safe_load(file.read_text()), yaml.safe_load(current.read_text())
            )
        else:
            assert current.read_bytes() == file.read_bytes(), f"{name}/{file.name} changed"
