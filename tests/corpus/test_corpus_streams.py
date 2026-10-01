"""The named random streams of the corpus generator."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_world.corpus.streams import STREAM_NAMES, Streams
from semantic_world.taxonomy.streams import stream_seed


def test_stream_names_match_the_spec() -> None:
    assert STREAM_NAMES == (
        "lexicon",
        "scenes",
        "documents",
        "propositions",
        "mentions",
        "grammar",
        "tests",
    )
    assert list(Streams(1).seeds()) == [f"corpus:{n}" for n in STREAM_NAMES]


def test_seeds_use_the_taxonomy_stream_seed_function() -> None:
    streams = Streams(42)
    assert streams.seed("lexicon") == stream_seed(42, "corpus:lexicon")
    assert streams.seeds()["corpus:tests"] == stream_seed(42, "corpus:tests")


def test_the_corpus_streams_differ_from_the_taxonomy_streams() -> None:
    # The corpus seed is its own master seed: the same number seeds different streams.
    assert stream_seed(1, "corpus:lexicon") != stream_seed(1, "taxonomy:rules")


def test_streams_are_independent_and_reproducible() -> None:
    a, b = Streams(5), Streams(5)
    assert a.lexicon.random(4).tolist() == b.lexicon.random(4).tolist()
    assert a.lexicon.random(4).tolist() != a.scenes.random(4).tolist()
    assert not np.array_equal(Streams(5).grammar.random(4), Streams(6).grammar.random(4))


def test_substreams_are_independent_of_the_stream_and_of_each_other() -> None:
    streams = Streams(7)
    first = streams.substream("lexicon", "synonyms").random(4).tolist()
    streams.lexicon.random(100)  # drawing from the stream does not move a substream
    assert streams.substream("lexicon", "synonyms").random(4).tolist() == first
    assert streams.substream("lexicon", "homonyms").random(4).tolist() != first


def test_unknown_stream() -> None:
    with pytest.raises(ValueError, match="unknown stream"):
        Streams(1).seed("nope")
    with pytest.raises(ValueError, match="unknown stream"):
        Streams(1).substream("nope", "key")
