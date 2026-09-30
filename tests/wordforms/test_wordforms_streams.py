"""The named random streams."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_world.taxonomy.streams import stream_seed
from semantic_world.wordforms.streams import STREAM_NAMES, Streams


def test_stream_names_match_the_spec():
    assert STREAM_NAMES == (
        "generate",
        "speakers",
        "synthesis",
        "augment",
        "train",
        "assign",
        "eval",
        "closed_class",
    )
    assert list(Streams(1).seeds()) == [f"wordforms:{n}" for n in STREAM_NAMES]


def test_seeds_use_the_taxonomy_stream_seed_function():
    streams = Streams(42)
    assert streams.seed("generate") == stream_seed(42, "wordforms:generate")
    assert streams.seeds()["wordforms:eval"] == stream_seed(42, "wordforms:eval")
    assert "wordforms:pca" not in streams.seeds()  # the projection is exact and needs no stream


def test_streams_are_independent_and_reproducible():
    a, b = Streams(5), Streams(5)
    assert a.generate.random(4).tolist() == b.generate.random(4).tolist()
    assert a.generate.random(4).tolist() != a.speakers.random(4).tolist()
    assert not np.array_equal(Streams(5).generate.random(4), Streams(6).generate.random(4))


def test_unknown_stream():
    with pytest.raises(ValueError, match="unknown stream"):
        Streams(1).seed("nope")
