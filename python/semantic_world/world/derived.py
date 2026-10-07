"""Derived values and their manifest (REL.16).

Derived values are written in a ``derived/`` folder, and ``derived/manifest.yaml`` lists each
file with the rule-set identity that produced it. A loader that finds a different identity
refuses the file, with an error that names the file and both identities. Nothing in
``derived/`` is read back as an input to a world, and nothing there is edited by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from semantic_world.world.errors import DerivedError

DERIVED_DIR = "derived"
MANIFEST_FILE = "manifest.yaml"
MANIFEST_VERSION = 1


def write_manifest(folder: str | Path, files: Mapping[str, str]) -> Path:
    """Write ``manifest.yaml`` in ``folder``: for each derived file name, the rule-set identity
    that produced it. The folder is created when it does not exist."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "files": {name: {"rule_set_id": identity} for name, identity in files.items()},
    }
    path = folder / MANIFEST_FILE
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def read_manifest(folder: str | Path) -> dict[str, str]:
    """The manifest of a derived folder as ``{file name: rule-set identity}``."""
    path = Path(folder) / MANIFEST_FILE
    if not path.is_file():
        raise DerivedError(f"{path} is missing: the derived folder has no manifest")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise DerivedError(f"{path} is malformed: expected a mapping with a 'files' mapping")
    result: dict[str, str] = {}
    for name, entry in data["files"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("rule_set_id"), str):
            raise DerivedError(f"{path} is malformed: the entry for {name} has no rule_set_id")
        result[str(name)] = entry["rule_set_id"]
    return result


def check_derived_file(folder: str | Path, name: str, rule_set_id: str) -> Path:
    """The path of a derived file, after checking that the manifest lists it with the given
    rule-set identity. Raises :class:`DerivedError` otherwise."""
    folder = Path(folder)
    path = folder / name
    manifest = read_manifest(folder)
    if name not in manifest:
        raise DerivedError(f"{path} is not listed in {folder / MANIFEST_FILE}")
    found = manifest[name]
    if found != rule_set_id:
        raise DerivedError(
            f"refusing {path}: it was made with rule set {found}, "
            f"but the rule set in use is {rule_set_id}"
        )
    if not path.is_file():
        raise DerivedError(f"{path} is listed in the manifest but missing")
    return path


def load_derived_csv(folder: str | Path, name: str, rule_set_id: str) -> pl.DataFrame:
    """Read a derived CSV table after the identity check."""
    return pl.read_csv(check_derived_file(folder, name, rule_set_id))
