"""The golden tests of the relabeled taxonomy (stage a5b of ``docs/specs/WORLD_AND_LANGUAGE.md``).

Two things are checked. First, the relabel changed nothing but the labels: for every golden
folder (the a5a generator's output for an example configuration, written before the relabel),
the relabeled generator run on the same settings makes the same categories, features,
instances, scalar values, and rules, label for label. The CAN features and rules left the
taxonomy, so they are left out of the comparison, and ``similarity.csv`` is not compared because
its vectors summed over the CAN columns. Second, the relabeled outputs are stable: every file of
every example configuration hashes to the value recorded in ``golden_hashes.json``, and the same
configuration and seed give byte-identical folders.

This file replaces ``test_taxonomy_regression.py``, which required the old outputs byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from semantic_world.taxonomy import config_from_mapping, generate, load_config
from semantic_world.taxonomy.config import RENAMED_TYPES, RENAMED_VALUES

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
GOLDEN = Path(__file__).resolve().parent / "golden"
HASHES = Path(__file__).resolve().parent / "golden_hashes.json"

_PATH = r"\d+(?:\.\d+)*"
_INDEX = r"\d+"
_OLD_TO_NEW = tuple(
    (re.compile(f"^{pattern}$"), replacement)
    for pattern, replacement in (
        (rf"ISA\.C({_PATH})", r"ISA.CATEGORY.\1"),
        (rf"IS\.({_INDEX})", r"PROPERTY.\1"),
        (rf"HAS\.({_INDEX})", r"PART.\1"),
        (rf"SC\.({_INDEX})", r"SCALARDIM.\1"),
        (rf"C({_PATH})", r"CATEGORY.\1"),
        (rf"I({_PATH})", r"INSTANCE.\1"),
    )
)
_TOKEN = re.compile(r"(?<![\w.])([A-Za-z_]+(?:\.[\w]+)*)")


def new_label(old: str) -> str:
    """The new label of an old one; a label that is not a label passes through."""
    for pattern, replacement in _OLD_TO_NEW:
        if pattern.match(old):
            return pattern.sub(replacement, old)
    return old


def relabel_text(text: str) -> str:
    """Every label token of an expression or a key, relabeled."""
    return _TOKEN.sub(lambda m: new_label(m.group(1)), text)


def is_can(label: str) -> bool:
    return label.startswith("CAN.")


def golden_configurations() -> list[str]:
    return sorted(p.name for p in GOLDEN.iterdir() if p.is_dir())


def old_settings(name: str, tmp_path: Path) -> dict[str, Any]:
    """The golden run's resolved configuration without the keys that left the taxonomy. The
    rule file of ``rule_file`` is rewritten without its CAN template and CAN rule, in the new
    labels."""
    data = yaml.safe_load((GOLDEN / name / "config.yaml").read_text(encoding="utf-8"))
    data.pop("provenance", None)
    data["features"].pop("can", None)
    data["rules"].get("overrides", {}).pop("can", None)
    data.pop("verbs", None)
    # The type names followed the labels in stage a6 (is -> property, has -> part).
    data["features"] = {RENAMED_TYPES.get(k, k): v for k, v in data["features"].items()}
    data["rules"]["input_type_weights"] = {
        RENAMED_TYPES.get(k, k): v for k, v in data["rules"]["input_type_weights"].items()
    }
    data["rules"]["overrides"] = {
        RENAMED_TYPES.get(k, k): v for k, v in data["rules"].get("overrides", {}).items()
    }
    bound = data["taxonomy"].get("similarity_bound")
    if isinstance(bound, dict) and bound.get("scope") in RENAMED_VALUES:
        bound["scope"] = RENAMED_VALUES[bound["scope"]]
    if data["rules"]["file"] is not None:
        rules = yaml.safe_load((DATA / "rules" / "example.yaml").read_text(encoding="utf-8"))
        # The committed example changed with the stage; the golden run used the old file.
        old = {
            "templates": [
                {"family": "literal", "weight": 0.1},
                {"family": "fixed", "arity": 2, "operator": "XOR", "weight": 0.1},
                {"family": "shj", "type": "IV", "weight": 0.2},
                {
                    "family": "compositional",
                    "arity": 4,
                    "operators": {"AND": 1, "OR": 1},
                    "nesting_depth": 2,
                    "weight": 0.3,
                },
            ],
            "explicit": [],
        }
        assert rules["templates"][0] == old["templates"][0]
        path = tmp_path / "old_rules.yaml"
        path.write_text(yaml.safe_dump(old), encoding="utf-8")
        data["rules"]["file"] = str(path)
    return data


def _read(folder: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(folder / name, infer_schema_length=None)


def _relabeled(frame: pl.DataFrame, label_columns: tuple[str, ...]) -> pl.DataFrame:
    """A golden frame with its label columns relabeled, its CAN rows dropped (rows whose first
    label column is a CAN feature), and its CAN columns dropped."""
    for column in label_columns:
        if column in frame.columns:
            frame = frame.filter(~pl.col(column).str.starts_with("CAN.").fill_null(False))
            frame = frame.with_columns(
                pl.col(column).map_elements(new_label, return_dtype=pl.Utf8).alias(column)
            )
    keep = [c for c in frame.columns if not is_can(c)]
    frame = frame.select(keep)
    frame.columns = [new_label(c) for c in frame.columns]
    if "type" in frame.columns:  # the type names followed the labels in stage a6
        frame = frame.with_columns(pl.col("type").replace(RENAMED_TYPES))
    return frame


def _assert_frames_equal(expected: pl.DataFrame, got: pl.DataFrame, what: str) -> None:
    assert expected.columns == got.columns, f"{what}: columns differ"
    assert expected.height == got.height, f"{what}: row counts differ"
    for column in expected.columns:
        a, b = expected[column].to_list(), got[column].to_list()
        if expected[column].dtype in (pl.Float64, pl.Float32):
            for x, y in zip(a, b, strict=True):
                both_nan = x != x and y != y
                assert both_nan or x == pytest.approx(y, abs=1e-9), f"{what}: {column} differs"
        else:
            assert a == b, f"{what}: {column} differs"


@pytest.mark.parametrize("name", golden_configurations())
def test_relabeled_outputs_equal_the_a5a_outputs_label_for_label(name: str, tmp_path: Path) -> None:
    golden = GOLDEN / name
    config = config_from_mapping(old_settings(name, tmp_path), source=str(tmp_path / "x.yaml"))
    assert config.features.base_rate_heterogeneity is None
    folder = generate(config).write(tmp_path / name)

    # The tree, the roles, and the features.
    _assert_frames_equal(
        _relabeled(_read(golden, "tree.csv"), ("label", "parent")),
        _read(folder, "tree.csv"),
        "tree.csv",
    )
    roles = _relabeled(_read(golden, "roles.csv"), ("category", "feature"))
    roles = roles.filter(~pl.col("feature").str.starts_with("EVENTTYPE"))
    _assert_frames_equal(roles, _read(folder, "roles.csv"), "roles.csv")
    features = _relabeled(_read(golden, "features.csv"), ("label",))
    _assert_frames_equal(features, _read(folder, "features.csv"), "features.csv")

    # The category vectors, without their CAN columns.
    for file in ("categories_generative.csv", "categories_defining.csv", "categories_mean.csv"):
        _assert_frames_equal(_relabeled(_read(golden, file), ("label",)), _read(folder, file), file)

    # instances.csv becomes base.csv (free features and scalars) and derived/static_features.csv.
    old = _relabeled(_read(golden, "instances.csv"), ("label", "leaf"))
    free = [
        f["label"]
        for f in features.iter_rows(named=True)
        if f["kind"] == "free" and f["type"] != "scalar"
    ]
    scalars = [f["label"] for f in features.iter_rows(named=True) if f["type"] == "scalar"]
    determined = [
        f["label"]
        for f in features.iter_rows(named=True)
        if f["kind"] == "determined" and f["type"] != "isa"
    ]
    _assert_frames_equal(
        old.select(["label", "leaf", *free, *scalars]), _read(folder, "base.csv"), "base.csv"
    )
    _assert_frames_equal(
        old.select(["label", *determined]),
        _read(folder / "derived", "static_features.csv"),
        "static_features.csv",
    )

    # The rules, without the CAN rules.
    golden_rules = [
        r for r in yaml.safe_load((golden / "rules.yaml").read_text()) if not is_can(r["output"])
    ]
    rules = yaml.safe_load((folder / "rules.yaml").read_text())
    assert len(golden_rules) == len(rules)
    for expected, got in zip(golden_rules, rules, strict=True):
        assert new_label(expected["output"]) == got["output"]
        assert [new_label(i) for i in expected["inputs"]] == got["inputs"]
        assert [new_label(i) for i in expected["relevant_inputs"]] == got["relevant_inputs"]
        assert relabel_text(expected["expression"]) == got["expression"]
        for key in (
            "layer",
            "family",
            "shj_type",
            "arity",
            "nesting_depth",
            "truth_table",
            "min_dnf_literals",
        ):
            assert expected[key] == got[key], (got["output"], key)
        if "thresholds" in expected:
            assert {new_label(k): v for k, v in expected["thresholds"].items()} == got["thresholds"]

    # The feature statistics, without the CAN rows.
    stats = _relabeled(_read(golden, "feature_stats.csv"), ("feature",))
    _assert_frames_equal(stats, _read(folder, "feature_stats.csv"), "feature_stats.csv")

    # The summary's counts.
    expected_summary = yaml.safe_load((golden / "summary.yaml").read_text())
    summary = yaml.safe_load((folder / "summary.yaml").read_text())
    for key in (
        "categories",
        "leaves",
        "instances",
        "duplicate_leaves",
        "duplicate_instances",
        "superordinates",
    ):
        assert expected_summary[key] == summary[key], key
    for old, new in RENAMED_TYPES.items():
        assert expected_summary["mean_true_features_per_instance"][old] == pytest.approx(
            summary["mean_true_features_per_instance"][new]
        )


# ---------------------------------------------------------------------------------------------
# Golden hashes of the relabeled outputs
# ---------------------------------------------------------------------------------------------


def _hashes(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "config.yaml"
    }


def example_configurations() -> list[str]:
    return sorted(p.stem for p in DATA.glob("*.yaml"))


def test_every_example_configuration_has_golden_hashes() -> None:
    recorded = json.loads(HASHES.read_text(encoding="utf-8"))
    assert sorted(recorded) == example_configurations()


@pytest.mark.parametrize("name", example_configurations())
def test_outputs_hash_to_the_recorded_values(name: str, tmp_path: Path) -> None:
    """Regenerate with ``python tests/taxonomy/make_golden_hashes.py`` after a deliberate change
    of the generator, and list the change in the stage proposal."""
    recorded = json.loads(HASHES.read_text(encoding="utf-8"))[name]
    folder = generate(load_config(DATA / f"{name}.yaml")).write(tmp_path / name)
    assert _hashes(folder) == recorded, f"the outputs of {name} changed"


def test_same_configuration_and_seed_give_byte_identical_folders(tmp_path: Path) -> None:
    config = load_config(DATA / "tiny.yaml")
    a = generate(config).write(tmp_path / "a")
    b = generate(config).write(tmp_path / "b")
    assert _hashes(a) == _hashes(b)
