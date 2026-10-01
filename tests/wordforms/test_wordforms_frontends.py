"""Layer 3: the waveform, log-mel, and cochleagram front ends, and their storage."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_synth import ToneEngine, espeak_only
from wordforms_support import (
    DATA,
    FIXTURES,
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_piper,
    needs_wordfreq,
)

from semantic_world.wordforms import run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.config import parse_config
from semantic_world.wordforms.frontends import (
    LOG_FLOOR,
    FrontendStore,
    compute_frontends,
    hz_to_mel,
    make_frontends,
    mel_filterbank,
    mel_to_hz,
    stored_frontends,
    write_frontend,
)
from semantic_world.wordforms.synth import synthesize_lexicon

RATE = 16000


def tone(frequency, seconds=0.61, amplitude=0.1):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def frontends_for(**frontends):
    return make_frontends(parse_config({"frontends": frontends}, "x"))


def nearest(centers, frequency):
    return int(np.argmin(np.abs(np.asarray(centers) - frequency)))


# ---------------------------------------------------------------------------------------------
# Waveform and log-mel
# ---------------------------------------------------------------------------------------------


def test_waveform_is_the_clip_itself():
    waveform = frontends_for()["waveform"]
    clip = tone(440)
    frames = waveform.compute(clip)
    assert frames.shape == (len(clip), 1) and frames.dtype == np.float32
    assert np.array_equal(frames[:, 0], clip)
    assert waveform.frame_rate == RATE and waveform.channels == 1
    assert waveform.frame_count(1234) == 1234


def test_logmel_shape_and_frame_rate_match_the_settings():
    logmel = frontends_for()["logmel"]
    assert (logmel.channels, logmel.frame_rate, logmel.window, logmel.hop) == (80, 100.0, 400, 160)
    for samples in (1, 159, 160, 161, 9600, 9777, 16000):
        clip = np.random.default_rng(samples).normal(0, 0.05, samples).astype(np.float32)
        frames = logmel.compute(clip)
        assert frames.shape == (math.ceil(samples / 160), 80) and frames.dtype == np.float32
        assert frames.shape[0] == logmel.frame_count(samples)
        assert np.isfinite(frames).all()
    other = frontends_for(logmel={"n_mels": 40, "window_ms": 50, "hop_ms": 20})["logmel"]
    assert (other.channels, other.frame_rate, other.window, other.hop) == (40, 50.0, 800, 320)
    assert other.n_fft == 1024
    assert other.compute(tone(440, 1.0)).shape == (50, 40)


def test_mel_filterbank():
    filters, centers = mel_filterbank(80, 512, RATE, 20.0, 8000.0)
    assert filters.shape == (80, 257) and len(centers) == 80
    assert (filters >= 0).all() and (filters.max(axis=1) > 0).all()
    assert np.all(np.diff(centers) > 0) and 20 < centers[0] < centers[-1] < 8000
    # equal steps on the mel scale
    assert np.allclose(np.diff(hz_to_mel(centers)), np.diff(hz_to_mel(centers))[0])
    assert mel_to_hz(hz_to_mel(1000.0)) == pytest.approx(1000.0)


def test_logmel_tone_peaks_in_the_band_nearest_the_tone():
    logmel = frontends_for()["logmel"]
    for frequency in (300, 1000, 3000):
        frames = logmel.compute(tone(frequency))
        assert int(frames[5:-5].mean(axis=0).argmax()) == nearest(logmel.centers, frequency)
    silence = logmel.compute(np.zeros(1600, dtype=np.float32))
    assert np.allclose(silence, np.log(LOG_FLOOR))


def test_logmel_frames_are_centered_in_their_hop():
    logmel = frontends_for()["logmel"]
    click = np.zeros(9600, dtype=np.float32)
    click[4800:4816] = 0.5  # in the hop of frame 30
    assert int(logmel.compute(click).sum(axis=1).argmax()) == 30


# ---------------------------------------------------------------------------------------------
# Cochleagram
# ---------------------------------------------------------------------------------------------


def test_cochleagram_shape_and_frame_rate_match_the_settings():
    cochleagram = frontends_for()["cochleagram"]
    assert (cochleagram.channels, cochleagram.frame_rate) == (64, 100.0)
    centers = cochleagram.centers
    assert len(centers) == 64 and centers[0] == 50.0 and centers[-1] == 8000.0
    assert np.all(np.diff(centers) > 0)
    # the channels are spaced evenly on the ERB scale
    erb = 9.265 * np.log(1 + np.asarray(centers) / (24.7 * 9.265))
    assert np.allclose(np.diff(erb), np.diff(erb)[0], rtol=1e-3)
    # clips shorter than, at, and beyond the half-second padding steps
    for samples in (1, 800, 7999, 8000, 8001, 9777, 16000, 16001, 40000):
        clip = np.random.default_rng(samples).normal(0, 0.05, samples).astype(np.float32)
        frames = cochleagram.compute(clip)
        assert frames.shape == (math.ceil(samples / 160), 64) and frames.dtype == np.float32
        assert frames.shape[0] == cochleagram.frame_count(samples)
        assert np.isfinite(frames).all() and (frames >= 0).all()
    other = frontends_for(
        cochleagram={"channels": 32, "low_hz": 100, "high_hz": 4000, "frame_rate": 50}
    )["cochleagram"]
    assert (other.channels, other.frame_rate) == (32, 50.0)
    assert other.centers[0] == 100.0 and other.centers[-1] == 4000.0
    assert other.compute(tone(440, 1.0)).shape == (50, 32)


def test_a_1_khz_tone_peaks_in_the_channel_nearest_1_khz():
    cochleagram = frontends_for()["cochleagram"]
    frames = cochleagram.compute(tone(1000))
    peak = int(frames[5:-5].mean(axis=0).argmax())
    assert peak == nearest(cochleagram.centers, 1000)
    assert abs(cochleagram.centers[peak] - 1000) < 60
    for frequency in (200, 3000, 6000):
        frames = cochleagram.compute(tone(frequency))
        assert int(frames[5:-5].mean(axis=0).argmax()) == nearest(cochleagram.centers, frequency)


def test_cochleagram_matches_the_chcochleagram_reference():
    """The NumPy cochleagram against saved outputs of the McDermott lab's chcochleagram package
    on tones, noise, a click, silence, and two synthesized words."""
    import json

    reference = np.load(FIXTURES / "cochleagram_reference.npz")
    cases = json.loads(str(reference["index"]))
    assert len(cases) == 9
    assert {c["name"] for c in cases} >= {"tone_1000hz", "white_noise", "word_1", "word_2"}
    worst = 0.0
    for case in cases:
        name = case["name"]
        cochleagram = frontends_for(cochleagram=case["settings"])["cochleagram"]
        expected = reference[f"{name}_output"]
        frames = cochleagram.compute(reference[f"{name}_input"])
        assert frames.shape == expected.shape and frames.dtype == expected.dtype == np.float32
        worst = max(worst, float(np.abs(frames - expected).max()))
        assert np.allclose(frames, expected, atol=2e-4, rtol=0), name
        assert np.allclose(cochleagram.centers, reference[f"{name}_centers"], atol=1e-3)
    # the largest difference measured when the reference was made was 4.5e-5
    assert worst < 1e-4


def test_erb_filterbank_squared_responses_add_to_one():
    from semantic_world.wordforms.frontends import erb_filterbank

    for size in (8000, 8001, 16000):
        filters, centers = erb_filterbank(size, RATE, 64, 50.0, 8000.0)
        assert filters.shape == (64, size // 2 + 1) and len(centers) == 64
        assert (filters >= 0).all()
        assert np.allclose((filters**2).sum(axis=0), 1.0)
        assert centers[0] == pytest.approx(50.0) and centers[-1] == pytest.approx(8000.0)


def test_cochleagram_compression_and_timing():
    cochleagram = frontends_for()["cochleagram"]
    channel = nearest(cochleagram.centers, 1000)
    quiet = cochleagram.compute(tone(1000, amplitude=0.01))[20:40, channel].mean()
    loud = cochleagram.compute(tone(1000, amplitude=0.1))[20:40, channel].mean()
    # ten times the amplitude gives 10 ** 0.3 times the output
    assert loud / quiet == pytest.approx(10**0.3, rel=0.02)
    linear = frontends_for(cochleagram={"compression": 1.0})["cochleagram"]
    ratio = (
        linear.compute(tone(1000, amplitude=0.1))[20:40, channel].mean()
        / linear.compute(tone(1000, amplitude=0.01))[20:40, channel].mean()
    )
    assert ratio == pytest.approx(10, rel=0.02)
    click = np.zeros(9600, dtype=np.float32)
    click[4800:4816] = 0.5
    assert int(cochleagram.compute(click).sum(axis=1).argmax()) == 30
    # silence gives the small floor of the package's envelope extraction
    silence = cochleagram.compute(np.zeros(1600, dtype=np.float32))
    assert silence.max() < 0.01 and silence.max() < 0.05 * loud


def test_front_ends_are_deterministic_and_depend_only_on_the_clip():
    clip = np.random.default_rng(0).normal(0, 0.05, 9000).astype(np.float32)
    for name in ("logmel", "cochleagram"):
        first = frontends_for()[name]
        expected = first.compute(clip)
        assert np.array_equal(first.compute(clip), expected)
        # a new front end that has seen other clips gives the same frames
        second = frontends_for()[name]
        second.compute(tone(500, 1.7))
        second.compute(tone(900, 0.2))
        assert np.array_equal(second.compute(clip), expected)


# ---------------------------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def synthesis(tmp_path):
    pytest.importorskip("cmudict")
    pytest.importorskip("wordfreq")
    pytest.importorskip("soundfile")
    config = espeak_only(tmp_path)
    run = run_forms(config)
    result = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": ToneEngine()}, check=False
    )
    return config, result


def test_the_index_recovers_every_token_exactly(synthesis, tmp_path):
    config, result = synthesis
    stores = compute_frontends(config, result, tmp_path / "run")
    assert list(stores) == ["logmel", "cochleagram"] == stored_frontends(config)
    frontends = make_frontends(config)
    for name, store in stores.items():
        folder = tmp_path / "run" / "frontends" / name
        assert sorted(p.name for p in folder.iterdir()) == ["frames.npy", "index.csv", "meta.yaml"]
        frames = np.load(folder / "frames.npy")
        assert frames.dtype == np.float32 and frames.ndim == 2
        assert frames.shape == (int(store.counts.sum()), frontends[name].channels)
        index = pl.read_csv(folder / "index.csv")
        assert index.columns == ["token", "first_frame", "frames"]
        assert index["token"].to_list() == [t.label for t in result.tokens] == store.labels
        assert index["first_frame"][0] == 0
        assert np.array_equal(
            index["first_frame"].to_numpy()[1:], np.cumsum(index["frames"].to_numpy())[:-1]
        )
        loaded = FrontendStore.load(folder)
        assert isinstance(loaded.frames, np.memmap)
        for token in result.tokens:
            clip = result.audio(token)
            expected = frontends[name].compute(clip)
            assert expected.shape[0] == math.ceil(len(clip) / 160)
            assert np.array_equal(store.token_frames(token.label), expected)
            assert np.array_equal(loaded.token_frames(token.label), expected)
        meta = yaml.safe_load((folder / "meta.yaml").read_text())
        assert meta["name"] == name and meta["frame_rate"] == 100.0
        assert meta["channels"] == frontends[name].channels == frames.shape[1]
        assert meta["tokens"] == 36 and meta["frames"] == frames.shape[0]
        assert len(meta["settings"]["center_hz"]) == meta["channels"]
        assert store.summary() == {
            "tokens": 36,
            "frames": frames.shape[0],
            "channels": frames.shape[1],
            "frame_rate": 100.0,
            "reused": False,
        }


def test_a_stored_front_end_is_reused_until_something_changes(synthesis, tmp_path):
    config, result = synthesis
    logmel = make_frontends(config)["logmel"]
    folder = tmp_path / "run" / "frontends" / "logmel"
    first = write_frontend(logmel, result, folder)
    assert not first.reused
    written = (folder / "frames.npy").stat().st_mtime_ns
    data = (folder / "frames.npy").read_bytes()
    second = write_frontend(logmel, result, folder)
    assert second.reused and (folder / "frames.npy").stat().st_mtime_ns == written
    assert np.array_equal(second.frames, first.frames)
    # other settings: computed again
    smaller = make_frontends(parse_config({"frontends": {"logmel": {"n_mels": 40}}}, "x"))["logmel"]
    third = write_frontend(smaller, result, folder)
    assert not third.reused and third.frames.shape[1] == 40
    # the first settings again: computed again, with the same bytes as before
    fourth = write_frontend(logmel, result, folder)
    assert not fourth.reused and (folder / "frames.npy").read_bytes() == data
    # other audio: computed again
    other_config = espeak_only(tmp_path / "other", count=7)
    other_run = run_forms(other_config)
    other = synthesize_lexicon(
        other_config,
        other_run.streams,
        other_run.lexicon.words,
        engines={"espeak": ToneEngine()},
        check=False,
    )
    fifth = write_frontend(logmel, other, folder)
    assert not fifth.reused and len(fifth) == 42
    # an unfinished write is not taken for a stored front end
    (folder / "meta.yaml").unlink()
    assert not write_frontend(logmel, other, folder).reused


def test_waveforms_are_stored_only_on_request(synthesis, tmp_path):
    config, result = synthesis
    assert "waveform" not in stored_frontends(config)
    data = config.resolved()
    data["frontends"] = {"waveform": {"store": True}, "logmel": None, "cochleagram": None}
    stored = parse_config(data, "x")
    assert stored_frontends(stored) == ["waveform"]
    stores = compute_frontends(stored, result, tmp_path / "run")
    store = stores["waveform"]
    assert store.meta["frame_rate"] == 16000.0 and store.meta["channels"] == 1
    for token in result.tokens:
        assert np.array_equal(store.token_frames(token.label)[:, 0], result.audio(token))
    assert store.frames.shape == (sum(len(result.audio(t)) for t in result.tokens), 1)


def test_progress_is_reported(synthesis, tmp_path):
    config, result = synthesis
    data = config.resolved()
    data["frontends"]["cochleagram"] = None
    seen = []
    compute_frontends(
        parse_config(data, "x"),
        result,
        tmp_path / "run",
        lambda name, i, n: seen.append((name, i, n)),
    )
    assert seen[0] == ("logmel", 1, 36) and seen[-1] == ("logmel", 36, 36) and len(seen) == 36


# ---------------------------------------------------------------------------------------------
# The command line, with the real engines
# ---------------------------------------------------------------------------------------------


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_espeak
@needs_piper
def test_frontends_command_on_the_tiny_configuration(tmp_path, capsys):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(tmp_path / "cache")
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["frontends", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "front end logmel:" in text and "80 channels at 100 per second, computed" in text
    assert "front end cochleagram:" in text and "64 channels" in text
    assert sorted(p.name for p in (out / "frontends").iterdir()) == [
        "cochleagram",
        "logmel",
        "modulation",
    ]
    tokens = pl.read_csv(out / "tokens.csv")
    for name, channels in (("logmel", 80), ("cochleagram", 64), ("modulation", 16)):
        store = FrontendStore.load(out / "frontends" / name)
        assert store.labels == tokens["label"].to_list()
        expected = [math.ceil(round(d * RATE) / 160) for d in tokens["duration"]]
        assert store.counts.tolist() == expected
        assert store.frames.shape == (sum(expected), channels)
        assert np.isfinite(store.frames).all()
    summary = yaml.safe_load((out / "summary.yaml").read_text())
    assert summary["frontends"]["logmel"]["tokens"] == tokens.height
    assert summary["frontends"]["cochleagram"]["reused"] is False
    # a second run reads the audio cache and finds the front ends already stored
    assert main(["all", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"0 synthesized, {summary['synthesis']['synthesized']} read from the cache" in text
    assert text.count("already stored") == 3
