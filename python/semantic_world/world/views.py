"""Views: tables for models, chosen from the base facts and the derived values (REL.16).

A view reads a world run: ``entities.csv`` for the base facts, ``derived/`` for the derived
static features and the capacities (each checked against the run's rule-set identity), and
``taxonomy/tree.csv`` for the ISA columns. Presets: ``base`` (free PROPERTY and PART features and
scalars), ``static`` (the base vector and the derived static features), and ``classic`` (the
columns of the old ``instances.csv``: ISA, PROPERTY, PART, the one-place capacities in the place
of CAN, and the scalars). ``--include`` takes label prefixes: a prefix that matches columns of the
preset narrows the view to them, and a prefix that matches none of the preset's columns adds every
available column with that prefix (the initial Boolean and numeric fluents among them). A
sidecar YAML file records the preset, the prefixes, the columns, and the rule-set identity.

Since stage c1, ``--places`` writes a table of the places instead: ``label``, ``leaf``, the free
place features, and the place scalars (``base``); with the derived place features and the place
capacities ``CANBE.PLACEEVENTTYPE.<n>`` (``static``); or with the ISA columns of the place tree
too (``classic``). ``--include`` works as for entities, over the place columns.
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
from semantic_world.world.places import PLACE_CAPACITIES_FILE, PLACES_DIR, PLACES_FILE

PRESETS = ("base", "static", "classic")
VIEWS_DIR = "views"
_ORDER = (
    "ISA.",
    "PROPERTY.",
    "PART.",
    "CAN.EVENTTYPE1.",
    "CAN.EVENTTYPE2.",
    "CAN.PLACEEVENTTYPE.",
    "CANBE.",
    "SCALARDIM.",
    "BOOLFL.",
    "NUMFL.",
    "PLACEPROPERTY.",
    "PLACEPART.",
    "PLACESCALARDIM.",
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
        "numeric": [c for c in entities.columns if c.startswith("NUMFL.")],
        "derived_static": [c for c in static.columns if c != "label"],
        "capacities": [c for c in capacities.columns if c not in ("label", "approximate")],
        "isa": [c for c in isa.columns if c != "label"],
    }
    return frame, kinds


def available_place_columns(run: Path) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Every column a view of the places can hold (stage c1), as one frame keyed by place, and
    the columns by kind: the free place features, the place scalars, the derived place features
    (from ``derived/static_features.csv`` of the place run, under ``places/``), the place
    capacities, and the ISA columns of the place tree."""
    definition = read_json(run / DEFINITION_FILE)
    identity = definition["rule_set_id"]
    path = run / PLACES_FILE
    if not path.is_file():
        raise WorldError(f"{run} has no {PLACES_FILE}: the world has no places")
    places = pl.read_csv(path, schema_overrides={"label": pl.Utf8, "leaf": pl.Utf8})
    static = pl.read_csv(
        run / PLACES_DIR / DERIVED_DIR / STATIC_FEATURES_FILE, schema_overrides={"label": pl.Utf8}
    )
    capacities = load_derived_csv(run / DERIVED_DIR, PLACE_CAPACITIES_FILE, identity)
    frame = places.join(static, on="label", how="left").join(
        capacities.drop("approximate"), on="label", how="left"
    )
    isa = _isa_frame(run, places, PLACES_DIR)
    frame = frame.join(isa, on="label", how="left")
    kinds = {
        "base": [c for c in places.columns if c.startswith(("PLACEPROPERTY.", "PLACEPART."))],
        "scalars": [c for c in places.columns if c.startswith("PLACESCALARDIM.")],
        "fluents": [],
        "numeric": [],
        "derived_static": [c for c in static.columns if c != "label"],
        "capacities": [c for c in capacities.columns if c not in ("label", "approximate")],
        "static_extra": [c for c in capacities.columns if c.startswith("CANBE.")],
        "isa": [c for c in isa.columns if c != "label"],
    }
    return frame, kinds


def _isa_frame(run: Path, entities: pl.DataFrame, taxonomy_dir: str = "taxonomy") -> pl.DataFrame:
    tree = pl.read_csv(run / taxonomy_dir / "tree.csv", schema_overrides={"parent": pl.Utf8})
    labels = tree["label"].to_list()
    parents = tree["parent"].to_list()
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
        return (
            _sorted(kinds["base"] + kinds["derived_static"])
            + kinds.get("static_extra", [])
            + kinds["scalars"]
        )
    if preset == "classic":
        one_place = [
            c
            for c in kinds["capacities"]
            if c.startswith("CAN.EVENTTYPE1.") or c.startswith("CANBE.PLACEEVENTTYPE.")
        ]
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
    places: bool = False,
) -> Path:
    """Write a view of a run and its sidecar, and return the table's path. With ``places``, the
    table is of the places (stage c1): the static preset adds the determined place features and
    the place capacities, and the classic preset the ISA columns of the place tree."""
    run = Path(run)
    frame, kinds = available_place_columns(run) if places else available_columns(run)
    columns = view_columns(preset, list(include), kinds)
    table = frame.select(["label", "leaf", *columns])
    name = f"places_{preset}.csv" if places else f"{preset}.csv"
    path = Path(out) if out is not None else run / VIEWS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path)
    sidecar = {
        "preset": preset,
        "places": places,
        "include": list(include),
        "columns": ["label", "leaf", *columns],
        "rule_set_id": read_json(run / DEFINITION_FILE)["rule_set_id"],
    }
    path.with_suffix(".yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    return path
