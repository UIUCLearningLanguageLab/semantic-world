"""Views: tables for models, chosen from the base facts and the derived values (REL.16).

A view reads a world run: ``entities.csv`` for the base facts, ``derived/`` for the derived
static features and the capacities (each checked against the run's rule-set identity), and
``taxonomy/tree.csv`` for the ISA columns. Presets: ``base`` (free PROPERTY and PART features and
scalars), ``static`` (the base vector and the derived static features), and ``classic`` (the
columns of the old ``instances.csv``: ISA, PROPERTY, PART, the one-place capacities in the place
of CAN, and the scalars). ``--include`` takes label prefixes: a prefix that matches columns of the
preset narrows the view to them, and a prefix that matches none of the preset's columns adds every
available column with that prefix. A sidecar YAML file records the preset, the prefixes, the
columns, and the rule-set identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from semantic_world.taxonomy.io import STATIC_FEATURES_FILE
from semantic_world.world.capacities import CAPACITIES_FILE
from semantic_world.world.definition import DEFINITION_FILE, ENTITIES_FILE
from semantic_world.world.derived import DERIVED_DIR, load_derived_csv
from semantic_world.world.errors import WorldError
from semantic_world.world.identity import read_json
from semantic_world.world.labels import translate

PRESETS = ("base", "static", "classic")
VIEWS_DIR = "views"
_ORDER = (
    "ISA.",
    "PROPERTY.",
    "PART.",
    "CAN.EVENTTYPE1.",
    "CAN.EVENTTYPE2.",
    "CANBE.",
    "SCALARDIM.",
    "BOOLFL.",
)


def _natural(label: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in label.split("."))


def _group(column: str) -> int:
    for i, prefix in enumerate(_ORDER):
        if column.startswith(prefix):
            return i
    return len(_ORDER)


def available_columns(run: Path) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Every column a view can hold, as one frame keyed by entity, and the columns by kind."""
    definition = read_json(run / DEFINITION_FILE)
    identity = definition["rule_set_id"]
    entities = pl.read_csv(run / ENTITIES_FILE)
    derived = run / DERIVED_DIR
    static = load_derived_csv(derived, STATIC_FEATURES_FILE, identity)
    capacities = load_derived_csv(derived, CAPACITIES_FILE, identity)
    frame = entities.join(static, on="label", how="left").join(
        capacities.drop("approximate"), on="label", how="left"
    )
    isa = _isa_frame(run, entities)
    frame = frame.join(isa, on="label", how="left")
    kinds = {
        "base": [c for c in entities.columns if c.startswith(("PROPERTY.", "PART."))],
        "scalars": [c for c in entities.columns if c.startswith("SCALARDIM.")],
        "fluents": [c for c in entities.columns if c.startswith("BOOLFL.")],
        "derived_static": [c for c in static.columns if c != "label"],
        "capacities": [c for c in capacities.columns if c not in ("label", "approximate")],
        "isa": [c for c in isa.columns if c != "label"],
    }
    return frame, kinds


def _isa_frame(run: Path, entities: pl.DataFrame) -> pl.DataFrame:
    tree = pl.read_csv(run / "taxonomy" / "tree.csv")
    labels = [translate(label) for label in tree["label"].to_list()]
    parents = [None if p is None else translate(p) for p in tree["parent"].to_list()]
    parent_of = dict(zip(labels, parents, strict=True))
    data: dict[str, Any] = {"label": entities["label"].to_list()}
    ancestors_of: dict[str, set[str]] = {}
    for label in labels:
        chain = set()
        node: str | None = label
        while node is not None:
            chain.add(node)
            node = parent_of[node]
        ancestors_of[label] = chain
    leaves = entities["leaf"].to_list()
    for label in labels:
        data[f"ISA.{label}"] = [1 if label in ancestors_of[leaf] else 0 for leaf in leaves]
    return pl.DataFrame(data, schema_overrides={"label": pl.Utf8})


def preset_columns(preset: str, kinds: dict[str, list[str]]) -> list[str]:
    if preset == "base":
        return kinds["base"] + kinds["scalars"]
    if preset == "static":
        return _sorted(kinds["base"] + kinds["derived_static"]) + kinds["scalars"]
    if preset == "classic":
        one_place = [c for c in kinds["capacities"] if c.startswith("CAN.EVENTTYPE1.")]
        return (
            kinds["isa"]
            + _sorted(kinds["base"] + kinds["derived_static"])
            + one_place
            + kinds["scalars"]
        )
    raise WorldError(f"unknown preset {preset!r}; the presets are {', '.join(PRESETS)}")


def _sorted(columns: Sequence[str]) -> list[str]:
    return sorted(columns, key=lambda c: (_group(c), _natural(c)))


def view_columns(preset: str, include: Sequence[str], kinds: dict[str, list[str]]) -> list[str]:
    columns = preset_columns(preset, kinds)
    if not include:
        return columns
    everything = _sorted([c for group in kinds.values() for c in group])
    chosen: list[str] = []
    for prefix in include:
        in_preset = [c for c in columns if c.startswith(prefix)]
        chosen.extend(in_preset if in_preset else [c for c in everything if c.startswith(prefix)])
    if not chosen:
        raise WorldError(f"no column matches the prefixes {', '.join(include)}")
    seen: dict[str, None] = {}
    for c in chosen:
        seen.setdefault(c, None)
    return _sorted(seen)


def write_view(
    run: str | Path,
    preset: str = "classic",
    include: Sequence[str] = (),
    out: str | Path | None = None,
) -> Path:
    """Write a view of a run and its sidecar, and return the table's path."""
    run = Path(run)
    frame, kinds = available_columns(run)
    columns = view_columns(preset, list(include), kinds)
    table = frame.select(["label", "leaf", *columns])
    path = Path(out) if out is not None else run / VIEWS_DIR / f"{preset}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path)
    sidecar = {
        "preset": preset,
        "include": list(include),
        "columns": ["label", "leaf", *columns],
        "rule_set_id": read_json(run / DEFINITION_FILE)["rule_set_id"],
    }
    path.with_suffix(".yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    return path
