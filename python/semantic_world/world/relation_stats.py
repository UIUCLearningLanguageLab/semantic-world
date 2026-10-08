"""Relation statistics of a world: the taxonomy's relation statistics with "verb" read as
"event type" and every label translated. Requirements only: they describe what entities are able
to do, never what is legal in a state."""

from __future__ import annotations

import polars as pl

from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.world.labels import translate

RELATION_FILES = (
    "relation_proportions.csv",
    "relation_pairs.csv",
    "event_type_stats.csv",
    "thematic.csv",
)
LABEL_COLUMNS = {"verb", "agent", "patient", "leaf_a", "leaf_b"}


def _translate_frame(frame: pl.DataFrame) -> pl.DataFrame:
    columns = []
    for name in frame.columns:
        if name in LABEL_COLUMNS:
            columns.append(
                pl.Series(
                    "event_type" if name == "verb" else name,
                    [translate(v) for v in frame[name].to_list()],
                    dtype=pl.Utf8,
                )
            )
        else:
            columns.append(frame[name])
    return pl.DataFrame(columns)


def relation_frames(taxonomy: TaxonomyResult) -> dict[str, pl.DataFrame]:
    """The four relation files of ``derived/``, keyed by file name, or an empty mapping for a
    world without two-place event types."""
    stats = taxonomy.relation_stats
    if stats is None:
        return {}
    return {
        "relation_proportions.csv": _translate_frame(stats.proportions),
        "relation_pairs.csv": _translate_frame(stats.pairs),
        "event_type_stats.csv": _translate_frame(stats.verb_stats),
        "thematic.csv": _translate_frame(stats.thematic),
    }
