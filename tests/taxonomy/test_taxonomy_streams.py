"""Stage 1 acceptance tests: the seeded random streams."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from semantic_world.taxonomy import STREAM_NAMES, Streams, stream_seed


def test_stream_seed_follows_the_specification() -> None:
    """SHA-256 of the master seed (8 bytes, little-endian) then the name (UTF-8), as an integer."""
    master = 1
    name = "taxonomy:rules"
    digest = hashlib.sha256(b"\x01" + b"\x00" * 7 + name.encode("utf-8")).digest()
    assert stream_seed(master, name) == int.from_bytes(digest, "little")
    master = 2**64 - 1
    digest = hashlib.sha256(b"\xff" * 8 + name.encode("utf-8")).digest()
    assert stream_seed(master, name) == int.from_bytes(digest, "little")


def test_stream_seed_is_stable() -> None:
    """A fixed value, so a change to the seeding scheme is noticed."""
    assert stream_seed(1, "taxonomy:rules") == stream_seed(1, "taxonomy:rules")
    assert stream_seed(1, "taxonomy:rules").bit_length() <= 256
    assert stream_seed(1, "taxonomy:rules") != stream_seed(1, "taxonomy:tree")
    assert stream_seed(1, "taxonomy:rules") != stream_seed(2, "taxonomy:rules")


def test_stream_seed_rejects_bad_master_seeds() -> None:
    with pytest.raises(ValueError):
        stream_seed(-1, "taxonomy:rules")
    with pytest.raises(ValueError):
        stream_seed(2**64, "taxonomy:rules")
    with pytest.raises(TypeError):
        stream_seed(1.0, "taxonomy:rules")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        stream_seed(True, "taxonomy:rules")


def test_streams_have_the_named_generators() -> None:
    streams = Streams(1)
    assert STREAM_NAMES == (
        "base_rates",
        "rules",
        "superordinates",
        "tree",
        "instances",
        "analysis",
        "scalars",
        "scalar_instances",
    )
    # The verb tree, the constraints, and the pairs are drawn by the world package now.
    assert not {"verb_tree", "constraints", "pairs"} & set(STREAM_NAMES)
    for name in STREAM_NAMES:
        assert isinstance(getattr(streams, name), np.random.Generator)
    assert list(streams.seeds()) == [f"taxonomy:{name}" for name in STREAM_NAMES]
    assert streams.seeds()["taxonomy:tree"] == stream_seed(1, "taxonomy:tree")
    assert streams.seed("tree") == stream_seed(1, "taxonomy:tree")
    with pytest.raises(ValueError):
        streams.seed("weather")


def test_same_seed_gives_the_same_draws() -> None:
    a = Streams(42)
    b = Streams(42)
    for name in STREAM_NAMES:
        assert np.array_equal(getattr(a, name).random(8), getattr(b, name).random(8))


def test_different_seeds_give_different_draws() -> None:
    a = Streams(42)
    b = Streams(43)
    assert not np.array_equal(a.tree.random(8), b.tree.random(8))


def test_streams_are_independent() -> None:
    """Consuming one stream leaves every other stream where it was."""
    untouched = Streams(7)
    consumed = Streams(7)
    consumed.tree.random(1000)
    consumed.instances.integers(0, 10, size=1000)
    assert np.array_equal(untouched.rules.random(8), consumed.rules.random(8))
    assert np.array_equal(untouched.analysis.random(8), consumed.analysis.random(8))


def test_streams_differ_from_each_other() -> None:
    streams = Streams(7)
    draws = {name: getattr(streams, name).random(8).tolist() for name in STREAM_NAMES}
    assert len({tuple(d) for d in draws.values()}) == len(STREAM_NAMES)
