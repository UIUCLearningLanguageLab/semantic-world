"""Stage a1: the derived-value manifest and the loader that refuses a different rule-set
identity (REL.16)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from semantic_world.world import (
    DerivedError,
    check_derived_file,
    load_derived_csv,
    read_manifest,
    write_manifest,
)

ID_A = "a" * 64
ID_B = "b" * 64


def derived_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "derived"
    folder.mkdir()
    pl.DataFrame({"label": ["I1.1.1"], "IS.7": [1]}).write_csv(folder / "static_features.csv")
    write_manifest(folder, {"static_features.csv": ID_A})
    return folder


def test_manifest_lists_each_file_with_its_identity(tmp_path: Path) -> None:
    folder = derived_folder(tmp_path)
    text = (folder / "manifest.yaml").read_text()
    assert text == f"version: 1\nfiles:\n  static_features.csv:\n    rule_set_id: {ID_A}\n"
    assert read_manifest(folder) == {"static_features.csv": ID_A}


def test_matching_identity_loads_the_file(tmp_path: Path) -> None:
    folder = derived_folder(tmp_path)
    assert check_derived_file(folder, "static_features.csv", ID_A) == folder / "static_features.csv"
    frame = load_derived_csv(folder, "static_features.csv", ID_A)
    assert frame.columns == ["label", "IS.7"]


def test_different_identity_is_refused_naming_the_file_and_both_identities(tmp_path: Path) -> None:
    folder = derived_folder(tmp_path)
    with pytest.raises(DerivedError) as info:
        load_derived_csv(folder, "static_features.csv", ID_B)
    message = str(info.value)
    assert str(folder / "static_features.csv") in message
    assert ID_A in message and ID_B in message


def test_unlisted_missing_or_malformed_files_are_errors(tmp_path: Path) -> None:
    folder = derived_folder(tmp_path)
    with pytest.raises(DerivedError, match="not listed"):
        check_derived_file(folder, "capacities.csv", ID_A)
    (folder / "static_features.csv").unlink()
    with pytest.raises(DerivedError, match="missing"):
        check_derived_file(folder, "static_features.csv", ID_A)
    (folder / "manifest.yaml").write_text("files: [1, 2]\n")
    with pytest.raises(DerivedError, match="malformed"):
        read_manifest(folder)
    (folder / "manifest.yaml").unlink()
    with pytest.raises(DerivedError, match="no manifest"):
        read_manifest(folder)
