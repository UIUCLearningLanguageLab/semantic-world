"""Writing the output folder.

A run writes one folder, by default ``runs/taxonomy/<name>_seed<seed>/``. CSV files have a header
row and one ID column first. Binary values are written as 0 and 1, missing values as ``NaN``, and
means and other real numbers with 6 decimal places. Feature columns are ordered ISA (in category
order), then PROPERTY and PART (in index order). The same configuration and seed give
byte-identical folders: nothing here depends on the time or the machine, apart from the git commit
hash and the package version recorded in ``config.yaml``.

The base vector and the derived values are written apart (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Taxonomy outputs"): ``base.csv`` holds each instance's label, leaf, free PROPERTY and PART
features, and scalars; ``derived/static_features.csv`` holds the determined features, listed in
``derived/manifest.yaml`` with the rule-set identity that produced it; ``rule_matrices.json`` holds
the rules as threshold matrices with that identity. The category-vector files keep both kinds of
feature, because they describe categories, not world inputs.
"""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import yaml

from semantic_world.taxonomy.config import Config
from semantic_world.taxonomy.fixed import FIXED_TEST_NAMES
from semantic_world.taxonomy.tree import Role
from semantic_world.world.derived import DERIVED_DIR, MANIFEST_FILE, write_manifest
from semantic_world.world.identity import write_json

if TYPE_CHECKING:
    from semantic_world.taxonomy.generate import TaxonomyResult

BASE_FILE = "base.csv"
CSV_FILES = (
    "features.csv",
    "tree.csv",
    "roles.csv",
    "categories_generative.csv",
    "categories_defining.csv",
    "categories_mean.csv",
    BASE_FILE,
    "similarity.csv",
    "feature_stats.csv",
)
YAML_FILES = ("config.yaml", "rules.yaml", "summary.yaml")
OUTPUT_FILES = YAML_FILES[:1] + CSV_FILES[:1] + YAML_FILES[1:2] + CSV_FILES[1:] + YAML_FILES[2:]

STATIC_FEATURES_FILE = "static_features.csv"
WORLD_FILES = (
    "rule_matrices.json",
    f"{DERIVED_DIR}/{MANIFEST_FILE}",
    f"{DERIVED_DIR}/{STATIC_FEATURES_FILE}",
)
"""The files of the matrix form and the derived values, as paths relative to the run folder."""

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_output_dir(config: Config, base: str | Path = "runs/taxonomy") -> Path:
    return Path(base) / f"{config.name}_seed{config.seed}"


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------


def git_commit(root: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """The commit hash of the repository at ``root`` and whether the tree has uncommitted
    changes to tracked files, or ``(None, None)`` outside a repository."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def package_version() -> str | None:
    try:
        return metadata.version("semantic_world")
    except metadata.PackageNotFoundError:
        return None


def provenance(stream_seeds: dict[str, int]) -> dict[str, Any]:
    commit, dirty = git_commit()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "package_version": package_version(),
        "stream_seeds": dict(stream_seeds),
    }


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def _binary_or_nan(matrix: np.ndarray) -> list[list[str]]:
    """Columns of ``0``, ``1``, or ``NaN`` for a float matrix with NaN gaps."""
    missing = np.isnan(matrix)
    text = np.where(missing, "NaN", np.where(matrix > 0.5, "1", "0"))
    return [text[:, j].tolist() for j in range(matrix.shape[1])]


def _means(matrix: np.ndarray) -> list[list[str]]:
    missing = np.isnan(matrix)
    text = np.where(missing, "NaN", np.char.mod("%.6f", np.nan_to_num(matrix)))
    return [text[:, j].tolist() for j in range(matrix.shape[1])]


def _matrix_frame(
    id_name: str, ids: list[str], labels: tuple[str, ...], columns: list
) -> pl.DataFrame:
    data: dict[str, Any] = {id_name: ids}
    for label, column in zip(labels, columns, strict=True):
        data[label] = column
    return pl.DataFrame(data)


def result_frames(result: TaxonomyResult) -> dict[str, pl.DataFrame]:
    """Every CSV table of the run, keyed by file name."""
    features = result.features
    tree = result.tree
    instances = result.instances
    vectors = result.vectors
    labels = vectors.feature_labels
    category_labels = [c.label for c in tree.categories]

    k = features.scalar_count
    scalar_labels = list(features.scalar_labels)
    features_frame = pl.DataFrame(
        {
            "label": list(labels) + scalar_labels,
            "type": ["isa"] * vectors.isa_count
            + [f.type for f in features.features]
            + ["scalar"] * k,
            "kind": ["determined"] * vectors.isa_count
            + ["free" if f.free else "determined" for f in features.features]
            + ["free"] * k,
            "layer": pl.Series(
                [float("nan")] * vectors.isa_count
                + [float(f.layer) for f in features.features]
                + [0.0] * k,
                dtype=pl.Float64,
            ),
            "base_rate": pl.Series(
                [float("nan")] * vectors.isa_count
                + [float("nan") if f.base_rate is None else f.base_rate for f in features.features]
                + [float("nan")] * k,
                dtype=pl.Float64,
            ),
        }
    )

    tree_frame = pl.DataFrame(
        {
            "label": category_labels,
            "parent": [None if c.parent is None else c.parent.label for c in tree.categories],
            "level": [c.level for c in tree.categories],
            "children": [len(c.children) for c in tree.categories],
            "instances": [int(len(instances.below(c, tree))) for c in tree.categories],
        },
        schema={
            "label": pl.Utf8,
            "parent": pl.Utf8,
            "level": pl.Int64,
            "children": pl.Int64,
            "instances": pl.Int64,
        },
    )

    role_names = {int(r): r.csv_name for r in Role}
    role_rows: dict[str, list] = {"category": [], "feature": [], "role": [], "fixed_test": []}
    free_index = {f.position: i for i, f in enumerate(features.free)}
    for ci, category in enumerate(tree.categories):
        for feature in features.features:
            if feature.free:
                role = role_names[int(category.roles[free_index[feature.position]])]
                test = ""
            elif vectors.fixed_by_rule[ci, feature.position]:
                role = "fixed_by_rule"
                test = FIXED_TEST_NAMES[int(vectors.fixed_test[ci, feature.position])]
            else:
                role = "determined"
                test = ""
            role_rows["category"].append(category.label)
            role_rows["feature"].append(feature.label)
            role_rows["role"].append(role)
            role_rows["fixed_test"].append(test)
    roles_frame = pl.DataFrame(role_rows)

    all_labels = tuple(labels) + tuple(scalar_labels)
    generative = _matrix_frame(
        "label",
        category_labels,
        all_labels,
        [vectors.generative[:, j].astype(np.int64).tolist() for j in range(len(labels))]
        + _means(vectors.scalar_generative),
    )
    defining = _matrix_frame(
        "label",
        category_labels,
        all_labels,
        _binary_or_nan(vectors.defining) + _means(vectors.scalar_defining),
    )
    mean = _matrix_frame(
        "label", category_labels, all_labels, _means(vectors.mean) + _means(vectors.scalar_mean)
    )

    extra: dict[str, pl.DataFrame] = {}
    bins = result.config.scalars.thermometer_bins
    if k and bins:
        extra["instances_scalar_codes.csv"] = thermometer_frame(
            list(instances.labels), scalar_labels, instances.scalars, bins
        )

    return {
        "features.csv": features_frame,
        "tree.csv": tree_frame,
        "roles.csv": roles_frame,
        "categories_generative.csv": generative,
        "categories_defining.csv": defining,
        "categories_mean.csv": mean,
        BASE_FILE: base_frame(result),
        "similarity.csv": result.similarity,
        "feature_stats.csv": result.feature_stats,
        **extra,
    }


def thermometer_frame(
    instance_labels: list[str], scalar_labels: list[str], values: np.ndarray, bins: int
) -> pl.DataFrame:
    """Thermometer codes: for each scalar, ``bins`` binary columns ``SCALARDIM.<n>>q<j>``, one per
    quantile ``j / (bins + 1)`` of the realized instance values, 1 where the value exceeds it."""
    data: dict[str, Any] = {"label": instance_labels}
    for j, label in enumerate(scalar_labels):
        column = values[:, j]
        for b in range(1, bins + 1):
            cut = np.quantile(column, b / (bins + 1)) if len(column) else np.nan
            data[f"{label}>q{b}"] = (column > cut).astype(np.int64).tolist()
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8})


def base_frame(result: TaxonomyResult) -> pl.DataFrame:
    """``base.csv``: one row per instance, with its label, its leaf, every free PROPERTY and PART
    feature in matrix order, and every scalar with 6 decimals."""
    instances = result.instances
    data: dict[str, Any] = {
        "label": list(instances.labels),
        "leaf": list(instances.leaf_labels),
    }
    for feature in result.features.free:
        data[feature.label] = instances.values[:, feature.position].astype(np.int64).tolist()
    for j, label in enumerate(result.features.scalar_labels):
        data[label] = _means(instances.scalars[:, j : j + 1])[0]
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})


def static_features_frame(result: TaxonomyResult) -> pl.DataFrame:
    """``derived/static_features.csv``: one row per instance, with every derived static feature
    (the determined PROPERTY and PART features, in matrix order)."""
    features = result.features
    data: dict[str, Any] = {"label": list(result.instances.labels)}
    for feature in features.determined:
        data[feature.label] = result.instances.values[:, feature.position].astype(np.int64).tolist()
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8})


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def _yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True)


def write_result(result: TaxonomyResult, path: str | Path | None = None) -> Path:
    folder = Path(path) if path is not None else default_output_dir(result.config)
    folder.mkdir(parents=True, exist_ok=True)

    config_data = result.config.resolved()
    config_data["provenance"] = provenance(result.stream_seeds)
    (folder / "config.yaml").write_text(_yaml(config_data), encoding="utf-8")
    (folder / "rules.yaml").write_text(_yaml(result.rules.records()), encoding="utf-8")
    (folder / "summary.yaml").write_text(_yaml(result.summary), encoding="utf-8")
    for name, frame in result_frames(result).items():
        frame.write_csv(folder / name, float_precision=6, null_value="")
    if result.matrices is not None:
        write_json(folder / "rule_matrices.json", result.matrices.record())
        derived = folder / DERIVED_DIR
        derived.mkdir(exist_ok=True)
        static_features_frame(result).write_csv(derived / STATIC_FEATURES_FILE)
        write_manifest(derived, {STATIC_FEATURES_FILE: result.matrices.rule_set_id})
    return folder
