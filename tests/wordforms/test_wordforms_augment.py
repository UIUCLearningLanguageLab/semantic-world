"""Stage 5: augmentation, Praat manipulation, and the modulation front end."""

# ruff: noqa: E501

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_embeddings import WordEngine
from wordforms_support import (
    DATA,
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_parselmouth,
    needs_piper,
    needs_pyroomacoustics,
    needs_wordfreq,
)

from semantic_world.wordforms import Run, run_forms, run_synthesis
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.augment import (
    Augmenter,
    add_noise,
    augment_synthesis,
    babble,
    draw_settings,
    long_term_spectrum,
    measured_snr_db,
    pink_magnitude,
    reverberate,
    room_impulse_response,
    shaped_noise,
    white_noise,
)
from semantic_world.wordforms.config import ConfigError, parse_config
from semantic_world.wordforms.embeddings import SoundEmbeddings, compute_embeddings
from semantic_world.wordforms.frontends import Modulation, compute_frontends, gabor, make_frontends
from semantic_world.wordforms.synth import synthesize_lexicon

RATE = 16000


def voice(f0: float = 150.0, seconds: float = 0.8, glide: float = 1.0, rate: int = RATE):
    """A synthetic voiced sound: harmonics of a pitch that glides from ``f0`` to ``f0 * glide``,
    with a formant-like spectral tilt, under a raised-cosine envelope."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pitch = f0 * glide ** (t / seconds)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    signal = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    return (0.3 * signal * envelope).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_modulation_configuration():
    assert parse_config({}, "x").frontends.modulation is None
    config = parse_config({"frontends": {"modulation": {}}}, "x")
    modulation = config.frontends.modulation
    assert modulation.rates == (2.0, 4.0, 8.0, 16.0, 32.0)
    assert modulation.scales == (0.25, 0.5, 1.0, 2.0, 4.0) and modulation.bands == 8
    assert config.frontends.names == ("waveform", "logmel", "cochleagram", "modulation")
    assert parse_config(config.resolved(), "x") == config
    for data, field, message in (
        ({"frontends": {"modulation": {"scales": [1, 8]}}}, "frontends.modulation.scales", "4.30"),
        ({"frontends": {"modulation": {"rates": [64]}}}, "frontends.modulation.rates", "half"),
        ({"frontends": {"modulation": {"rates": [4, 2]}}}, "frontends.modulation.rates", "ascending"),
        ({"frontends": {"modulation": {"bands": 65}}}, "frontends.modulation.bands", "at most 64"),
        ({"frontends": {"cochleagram": None, "modulation": {}}}, "frontends.modulation", "needs"),
    ):  # fmt: skip
        with pytest.raises(ConfigError) as info:
            parse_config(data, "x")
        assert info.value.field == field and message in info.value.message, info.value
    fine = parse_config(
        {"frontends": {"cochleagram": {"channels": 128}, "modulation": {"scales": [1, 8]}}}, "x"
    )
    assert fine.frontends.modulation.scales == (1.0, 8.0)


def test_augmentation_configuration():
    assert parse_config({}, "x").augmentation is None
    assert parse_config({"augmentation": None}, "x").augmentation is None
    config = parse_config(
        {
            "augmentation": {
                "recipes": [
                    {"name": "noisy", "noise": {"kinds": ["babble"], "snr_db": 10}},
                    {"name": "room", "reverberation": {"rt60": [0.3, 0.6]}},
                    {"name": "shifted", "speed_pitch": {}},
                    {"name": "voice", "manipulation": {"pitch_median_hz": [100, 250]}},
                ],
                "proportion": 0.5,
                "speakers": "train",
            }
        },
        "x",
    )
    augmentation = config.augmentation
    assert [r.name for r in augmentation.recipes] == ["noisy", "room", "shifted", "voice"]
    noisy, room, shifted, voice = augmentation.recipes
    assert noisy.noise.kinds == ("babble",) and noisy.noise.snr_db == (10.0, 10.0)
    assert noisy.noise.babble_voices == 6 and noisy.reverberation is None
    assert room.reverberation.rt60 == (0.3, 0.6) and room.reverberation.room_m == (3.0, 6.0)
    assert shifted.speed_pitch.speed == (0.9, 1.1) and shifted.speed_pitch.pitch_semitones == (-2.0, 2.0)  # fmt: skip
    assert voice.manipulation.pitch_median_hz == (100.0, 250.0)
    assert voice.manipulation.formant_shift_ratio is None
    assert augmentation.proportion == 0.5 and augmentation.speakers == "train"
    assert parse_config(config.resolved(), "x") == config
    for section, field, message in (
        ({}, "augmentation.recipes", "required"),
        ({"recipes": []}, "augmentation.recipes", "non-empty"),
        ({"recipes": [{"name": "a"}]}, "augmentation.recipes[0].name", "no transformation"),
        ({"recipes": [{"noise": {}}]}, "augmentation.recipes[0].name", "required"),
        ({"recipes": [{"name": "a", "noise": {}}, {"name": "a", "noise": {}}]}, "augmentation.recipes", "distinct"),
        ({"recipes": [{"name": "a", "noise": {"kinds": ["brown"]}}]}, "augmentation.recipes[0].noise.kinds", "unknown noise"),
        ({"recipes": [{"name": "a", "noise": {"snr_db": [20, 0]}}]}, "augmentation.recipes[0].noise.snr_db", "exceeds"),
        ({"recipes": [{"name": "a", "manipulation": {}}]}, "augmentation.recipes[0].manipulation", "sets nothing"),
        ({"recipes": [{"name": "a", "speed_pitch": {"speed": 0}}]}, "augmentation.recipes[0].speed_pitch.speed", "more than 0"),
        ({"recipes": [{"name": "a", "noise": {}}], "speakers": "some"}, "augmentation.speakers", "all, train, held_out"),
        ({"recipes": [{"name": "a", "noise": {}, "extra": 1}]}, "augmentation.recipes[0].extra", "unknown key"),
    ):  # fmt: skip
        with pytest.raises(ConfigError) as info:
            parse_config({"augmentation": section}, "x")
        assert info.value.field == field and message in info.value.message, info.value


# ---------------------------------------------------------------------------------------------
# Noise and reverberation
# ---------------------------------------------------------------------------------------------


def test_noise_kinds_and_exact_snr():
    rng = np.random.default_rng(0)
    clip = voice()
    n = len(clip)
    spectrum = long_term_spectrum([clip, voice(200.0)])
    assert spectrum.shape == (513,) and np.all(spectrum >= 0)
    kinds = {
        "white": white_noise(rng, n),
        "pink": shaped_noise(rng, n, pink_magnitude()),
        "speech": shaped_noise(rng, n, spectrum),
        "babble": babble(rng, n, [voice(120.0, 0.3), voice(180.0, 1.2)]),
    }
    for kind, noise in kinds.items():
        assert noise.shape == (n,) and np.isfinite(noise).all(), kind
        for snr in (0.0, 5.5, 20.0):
            mixture = add_noise(clip, noise, snr)
            assert measured_snr_db(mixture, clip) == pytest.approx(snr, abs=1e-6), kind

    # pink noise has more low-frequency power than white noise; speech-shaped noise follows the
    # long-term spectrum of the clips it was shaped by
    def band_power(noise, low, high):
        power = np.abs(np.fft.rfft(noise)) ** 2
        freqs = np.fft.rfftfreq(len(noise), 1 / RATE)
        return power[(freqs >= low) & (freqs < high)].mean()

    assert band_power(kinds["pink"], 100, 500) / band_power(kinds["pink"], 4000, 8000) > 5
    assert band_power(kinds["white"], 100, 500) / band_power(kinds["white"], 4000, 8000) < 2
    assert band_power(kinds["speech"], 100, 1000) > band_power(kinds["speech"], 5000, 8000)
    # babble at a random offset is deterministic for a seeded generator
    again = babble(np.random.default_rng(0), n, [voice(120.0, 0.3)])
    assert np.array_equal(again, babble(np.random.default_rng(0), n, [voice(120.0, 0.3)]))
    with pytest.raises(Exception, match="silent"):
        add_noise(clip, np.zeros(n), 10.0)


@needs_pyroomacoustics
def test_room_impulse_response_decays_at_the_reverberation_time():
    rir, absorption = room_impulse_response(
        RATE, (5.0, 4.0, 3.0), 0.5, [1, 1, 1.5], [3.5, 2.5, 1.5]
    )
    assert rir.ndim == 1 and len(rir) > RATE * 0.4 and 0 < absorption < 1
    with pytest.raises(Exception, match="no room"):
        room_impulse_response(RATE, (12.0, 12.0, 12.0), 0.1, [1, 1, 1.5], [3.5, 2.5, 1.5])
    # the Schroeder decay curve falls by 60 dB in about rt60 seconds
    energy = np.cumsum(rir[::-1] ** 2)[::-1]
    decay = 10 * np.log10(energy / energy[0])
    start = int(np.argmax(rir**2))
    rt60 = (np.argmax(decay[start:] <= -60) if (decay[start:] <= -60).any() else len(decay) - start) / RATE  # fmt: skip
    assert 0.35 < rt60 < 0.75
    again, _ = room_impulse_response(RATE, (5.0, 4.0, 3.0), 0.5, [1, 1, 1.5], [3.5, 2.5, 1.5])
    assert np.array_equal(rir, again)
    clip = voice()
    wet = reverberate(clip, rir)
    assert len(wet) == len(clip) + len(rir) - 1
    assert np.abs(wet[len(clip) :]).max() > 0  # the tail rings on after the clip


# ---------------------------------------------------------------------------------------------
# Praat tools
# ---------------------------------------------------------------------------------------------


@needs_parselmouth
def test_praat_tools_on_a_synthetic_voice():
    from semantic_world.wordforms import praat

    clip = voice(150.0, 1.0, glide=1.5)  # the pitch glides from 150 to 225 Hz
    measured = praat.measure_pitch(clip, RATE)
    assert 150 < measured.median_hz < 225 and measured.voiced_frames > 100
    assert 5 < measured.range_semitones < 8  # 150 to 225 Hz is 7 semitones
    assert measured.as_dict()["frames"] == measured.frames
    # a span
    first_half = praat.measure_pitch(clip, RATE, (0.0, 0.5))
    second_half = praat.measure_pitch(clip, RATE, (0.5, 1.0))
    assert first_half.median_hz < measured.median_hz < second_half.median_hz
    # the median pitch moves to within 5% of its target, on the whole clip and on a span
    for target in (110.0, 200.0, 300.0):
        moved = praat.change_pitch(clip, RATE, median_hz=target)
        assert len(moved) == len(clip)
        assert praat.measure_pitch(moved, RATE).median_hz == pytest.approx(target, rel=0.05)
        gendered = praat.manipulate(clip, RATE, pitch_median_hz=target)
        assert praat.measure_pitch(gendered, RATE).median_hz == pytest.approx(target, rel=0.05)
    spanned = praat.change_pitch(clip, RATE, factor=1.4, time_range=(0.0, 0.5))
    assert praat.measure_pitch(spanned, RATE, (0.0, 0.5)).median_hz == pytest.approx(
        1.4 * first_half.median_hz, rel=0.05
    )
    assert praat.measure_pitch(spanned, RATE, (0.5, 1.0)).median_hz == pytest.approx(
        second_half.median_hz, rel=0.05
    )
    # the pitch range scales around the median
    flat = praat.change_pitch(clip, RATE, range_factor=0.25)
    assert praat.measure_pitch(flat, RATE).range_semitones < 0.5 * measured.range_semitones
    assert praat.measure_pitch(flat, RATE).median_hz == pytest.approx(measured.median_hz, rel=0.05)  # fmt: skip
    # the duration changes on the whole clip or on a span, keeping the pitch
    longer = praat.change_duration(clip, RATE, 1.5)
    assert len(longer) == pytest.approx(1.5 * len(clip), rel=0.01)
    assert praat.measure_pitch(longer, RATE).median_hz == pytest.approx(measured.median_hz, rel=0.05)  # fmt: skip
    part = praat.change_duration(clip, RATE, 1.5, (0.25, 0.75))
    assert len(part) == pytest.approx(1.25 * len(clip), rel=0.01)
    assert len(praat.change_duration(clip, RATE, 0.5, (0.0, 0.5))) == pytest.approx(0.75 * len(clip), rel=0.01)  # fmt: skip
    # "Change gender" on a span is spliced back: the samples outside the span are untouched
    shifted = praat.manipulate(clip, RATE, formant_shift_ratio=1.2, time_range=(0.25, 0.5))
    assert len(shifted) == pytest.approx(len(clip), abs=2)
    assert np.array_equal(shifted[: int(0.25 * RATE)], clip[: int(0.25 * RATE)])
    assert np.array_equal(shifted[-int(0.5 * RATE) :], clip[-int(0.5 * RATE) :])
    assert not np.allclose(shifted[int(0.3 * RATE) : int(0.45 * RATE)], clip[int(0.3 * RATE) : int(0.45 * RATE)])  # fmt: skip
    stretched = praat.manipulate(clip, RATE, duration_factor=1.25, time_range=(0.0, 0.4))
    assert len(stretched) == pytest.approx(1.1 * len(clip), rel=0.01)
    # nothing to do gives the clip back; errors name the problem
    assert np.array_equal(praat.manipulate(clip, RATE), clip)
    assert np.array_equal(praat.change_pitch(clip, RATE), clip)
    with pytest.raises(praat.PraatError, match="no voiced frame"):
        praat.change_pitch(np.zeros(RATE, dtype=np.float32) + 1e-4, RATE, median_hz=200.0)
    with pytest.raises(ValueError):
        praat.manipulate(clip, RATE, duration_factor=2.0, time_range=(0.5, 0.2))
    with pytest.raises(ValueError, match="not both"):
        praat.change_pitch(clip, RATE, median_hz=100.0, factor=2.0)
    assert "Praat" in praat.praat_version()


# ---------------------------------------------------------------------------------------------
# The modulation front end
# ---------------------------------------------------------------------------------------------


def test_gabor_filters_have_zero_mean_and_unit_gain():
    for sigma, cycles, complex_valued in ((10.0, 0.05, True), (4.3, 0.116, False), (25.0, 0.02, True)):  # fmt: skip
        kernel = gabor(sigma, cycles, complex_valued)
        assert abs(kernel.sum()) < 1e-9
        n = np.arange(len(kernel)) - len(kernel) // 2
        gain = np.abs(np.sum(kernel * np.exp(-2j * np.pi * cycles * n)))
        assert gain == pytest.approx(1.0)


def test_modulation_responds_most_to_its_own_rate_and_scale():
    config = parse_config({"frontends": {"modulation": {}}}, "x")
    modulation = make_frontends(config)["modulation"]
    assert isinstance(modulation, Modulation)
    assert modulation.channels == 5 * 5 * 8 and modulation.frame_rate == 100.0
    frames, channels = 300, 64
    t = np.arange(frames)[:, None] / modulation.frame_rate
    k = np.arange(channels)[None, :] * modulation.octaves_per_channel
    for rate in modulation.rates:
        for scale in modulation.scales:
            for direction in (1, -1):
                ripple = 1.0 + np.cos(2 * np.pi * (rate * t + direction * scale * k))
                responses = modulation.responses(ripple)
                assert responses.shape == (frames, 5, 5, channels)
                strength = responses[50:250].mean(axis=(0, 3))
                i, j = np.unravel_index(np.argmax(strength), strength.shape)
                assert (modulation.rates[i], modulation.scales[j]) == (rate, scale)


def test_modulation_frames_from_a_clip():
    config = parse_config(
        {"frontends": {"modulation": {"rates": [4, 16], "scales": [0.5, 2], "bands": 4}}}, "x"
    )
    modulation = make_frontends(config)["modulation"]
    clip = voice(120.0, 0.73)
    frames = modulation.compute(clip)
    assert frames.shape == (modulation.frame_count(len(clip)), 16)
    assert frames.dtype == np.float32 and np.isfinite(frames).all() and (frames >= 0).all()
    assert modulation.frame_count(len(clip)) == make_frontends(config)["cochleagram"].frame_count(len(clip))  # fmt: skip
    assert np.array_equal(frames, modulation.compute(clip))  # deterministic
    settings = modulation.settings()
    assert settings["channel_order"] == "rate, then scale, then band"
    assert settings["cochleagram"]["channels"] == 64


# ---------------------------------------------------------------------------------------------
# Augmentation in a run
# ---------------------------------------------------------------------------------------------


def stand_in_config(tmp_path, recipes, **augmentation):
    return parse_config(
        {
            "name": "augment_test",
            "wordforms": {"count": 6},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 2,
                "held_out_speaker_proportion": 0.34,
                "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
            },
            "frontends": {"modulation": {"rates": [4, 16], "scales": [0.5, 2], "bands": 4}},
            "embeddings": [
                {"name": "cochleagram_fixed", "encoder": "fixed", "frontend": "cochleagram", "pca_dims": 8},
                {"name": "modulation_fixed", "encoder": "fixed", "frontend": "modulation", "pca_dims": 8},
            ],
            "closed_class": None,
            "augmentation": {"recipes": recipes, **augmentation},
        },
        "augment_test",
    )  # fmt: skip


def stand_in_run(tmp_path, config) -> Run:
    run = run_forms(config)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": WordEngine()}, check=False
    )
    augment_synthesis(config, run.streams, run.synthesis)
    return run


NOISE_AND_SPEED = [
    {"name": "noisy", "noise": {"kinds": ["white", "pink", "speech", "babble"], "snr_db": [0, 20], "babble_voices": 3}},
    {"name": "faster", "speed_pitch": {"speed": [1.1, 1.3], "pitch_semitones": 0}},
]  # fmt: skip


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_augmented_tokens_are_new_tokens_with_their_own_records(tmp_path):
    config = stand_in_config(tmp_path, NOISE_AND_SPEED)
    run = stand_in_run(tmp_path, config)
    synthesis = run.synthesis
    originals = [t for t in synthesis.tokens if not t.augmentation]
    augmented = [t for t in synthesis.tokens if t.augmentation]
    assert len(originals) == 6 * 3 * 2 and len(augmented) == 2 * len(originals)
    assert synthesis.tokens[: len(originals)] == originals  # the originals come first
    by_source = {t.label: t for t in originals}
    for token in augmented:
        record = json.loads(token.augmentation)
        source = by_source[record["source"]]
        number = 1 if record["recipe"] == "noisy" else 2
        assert token.label == f"{source.label}.A.{number}"
        assert token.word == source.word and token.speaker == source.speaker
        assert token.engine == source.engine and token.phonemes == source.phonemes
        assert token.cache_path.startswith("v2/augment/") and token.sha256 != source.sha256
        assert (synthesis.cache_dir / token.cache_path).exists()
        clip = synthesis.audio(token)
        assert token.duration == pytest.approx(len(clip) / RATE, abs=1e-6)
        assert abs(token.rms_db - config.synthesis.level.rms_db) < 0.05 or token.peak == pytest.approx(0.9, abs=1e-3)  # fmt: skip
        if record["recipe"] == "noisy":
            assert record["noise"]["kind"] in ("white", "pink", "speech", "babble")
            assert 0 <= record["noise"]["snr_db"] <= 20
            if record["noise"]["kind"] == "babble":
                assert len(record["noise"]["voices"]) == 3
            assert token.duration == source.duration
        else:
            speed = record["speed_pitch"]["speed"]
            assert 1.1 <= speed <= 1.3 and record["speed_pitch"]["pitch_semitones"] == 0
            assert token.duration == pytest.approx(source.duration / speed, abs=0.002)
    summary = synthesis.summary()
    assert summary["augmented_tokens"] == len(augmented)
    assert summary["augmentation"]["by_recipe"] == {"noisy": 36, "faster": 36}
    assert summary["augmentation"]["computed"] == 72 and summary["augmentation"]["skipped"] == []
    # the run folder records them
    run.write(tmp_path / "run")
    table = pl.read_csv(tmp_path / "run" / "tokens.csv")
    assert table.height == 108
    assert (table["augmentation"].fill_null("") == "").sum() == 36
    assert json.loads(table["augmentation"][-1])["recipe"] == "faster"


def snr_errors(run: Run, recipe_name: str) -> tuple[list[float], list[float]]:
    """For every token of a noise recipe: the error of the stored mixture's signal-to-noise
    ratio against its target, with the noise rebuilt from the token's substream (exact up to
    16-bit rounding), and the error of a least-squares estimate from the clean clip alone (which
    the correlation between speech and babble or speech-shaped noise biases)."""
    config, synthesis = run.config, run.synthesis
    originals = [t for t in synthesis.tokens if not t.augmentation]
    augmenter = Augmenter(config, synthesis, originals)
    by_source = {t.label: t for t in originals}
    recipe = next(r for r in config.augmentation.recipes if r.name == recipe_name)
    exact, estimated = [], []
    for token in synthesis.tokens:
        if not token.augmentation:
            continue
        record = json.loads(token.augmentation)
        if record["recipe"] != recipe_name:
            continue
        source = by_source[record["source"]]
        rng = run.streams.substream("augment", f"{source.label}|{recipe_name}")
        drawn = draw_settings(rng, recipe, len(originals))
        assert drawn["noise"] == record["noise"]
        clean = synthesis.audio(source).astype(np.float64)
        noise = augmenter.noise_for(rng, len(clean), record["noise"])
        mixture = add_noise(clean, noise, record["noise"]["snr_db"])
        # the stored clip is the leveled mixture: the same up to a gain and 16-bit rounding
        stored = synthesis.audio(token).astype(np.float64)
        gain = float(np.dot(stored, mixture) / np.dot(mixture, mixture))
        assert np.allclose(stored, gain * mixture, atol=2e-4)
        exact.append(measured_snr_db(stored, gain * clean) - record["noise"]["snr_db"])
        fit = float(np.dot(stored, clean) / np.dot(clean, clean))
        estimated.append(measured_snr_db(stored, fit * clean) - record["noise"]["snr_db"])
    return exact, estimated


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_achieved_snr_is_the_target(tmp_path):
    run = stand_in_run(tmp_path, stand_in_config(tmp_path, NOISE_AND_SPEED[:1]))
    exact, estimated = snr_errors(run, "noisy")
    assert len(exact) == 36
    assert max(abs(e) for e in exact) < 0.05
    # the stand-in engine's clips are chords, which babble made of other chords correlates
    # with, so the least-squares estimate is only roughly right here
    assert np.median(np.abs(estimated)) < 0.5


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_augmentation_is_cached_seeded_and_independent(tmp_path):
    config = stand_in_config(tmp_path, NOISE_AND_SPEED)
    first = stand_in_run(tmp_path, config)
    assert first.synthesis.augmentation["computed"] == 72
    second = stand_in_run(tmp_path, config)
    assert second.synthesis.augmentation["computed"] == 0
    assert second.synthesis.augmentation["read_from_cache"] == 72
    assert second.synthesis.tokens == first.synthesis.tokens
    # a fresh cache gives the same clips (the stand-in engine is deterministic)
    other = stand_in_run(tmp_path / "other", stand_in_config(tmp_path / "other", NOISE_AND_SPEED))
    assert [t.sha256 for t in other.synthesis.tokens] == [t.sha256 for t in first.synthesis.tokens]
    # adding a recipe changes nothing about the earlier recipes' tokens
    more = stand_in_run(tmp_path, stand_in_config(tmp_path, NOISE_AND_SPEED + [{"name": "third", "speed_pitch": {"speed": 0.8, "pitch_semitones": 0}}]))  # fmt: skip
    kept = [t for t in more.synthesis.tokens if not t.label.endswith(".A.3")]
    assert kept == first.synthesis.tokens
    # another seed gives other draws
    seeded = stand_in_run(tmp_path, config.with_seed(2))
    assert [t.sha256 for t in seeded.synthesis.tokens[36:]] != [t.sha256 for t in first.synthesis.tokens[36:]]  # fmt: skip


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_proportion_and_speakers_select_the_tokens(tmp_path):
    held_out = stand_in_run(tmp_path, stand_in_config(tmp_path, NOISE_AND_SPEED[:1], speakers="held_out"))  # fmt: skip
    speakers = {s.label: s for s in held_out.synthesis.speakers}
    augmented = [t for t in held_out.synthesis.tokens if t.augmentation]
    assert augmented and all(speakers[t.speaker].held_out for t in augmented)
    train = stand_in_run(tmp_path, stand_in_config(tmp_path, NOISE_AND_SPEED[:1], speakers="train"))  # fmt: skip
    augmented = [t for t in train.synthesis.tokens if t.augmentation]
    assert len(augmented) == 24 and not any(speakers[t.speaker].held_out for t in augmented)
    half = stand_in_run(tmp_path, stand_in_config(tmp_path, NOISE_AND_SPEED[:1], proportion=0.5))
    count = sum(bool(t.augmentation) for t in half.synthesis.tokens)
    assert 8 <= count <= 28  # about half of 36
    # a token's draw does not depend on the proportion: the chosen tokens are the same clips
    full = {t.label: t.sha256 for t in stand_in_run(tmp_path, stand_in_config(tmp_path, NOISE_AND_SPEED[:1])).synthesis.tokens}  # fmt: skip
    assert all(full[t.label] == t.sha256 for t in half.synthesis.tokens)


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_augmented_tokens_go_through_frontends_embeddings_and_the_interface(tmp_path):
    config = stand_in_config(tmp_path, NOISE_AND_SPEED)
    run = stand_in_run(tmp_path, config)
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / "run")
    assert set(run.frontends) == {"logmel", "cochleagram", "modulation"}
    assert len(run.frontends["modulation"]) == 108
    assert run.frontends["modulation"].meta["channels"] == 16
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run"
    )
    for name in ("cochleagram_fixed", "modulation_fixed"):
        assert run.embeddings[name].tokens.shape == (108, 8)
        assert run.embeddings[name].types.shape == (6, 8)
    run.write(tmp_path / "run")
    sounds = SoundEmbeddings.load(tmp_path / "run", "modulation_fixed")
    assert sounds.token_augmented.sum() == 72 and not sounds.token_augmented[:36].any()
    assert sounds.to_torch()["token_augmented"].sum() == 72 if hasattr(sounds, "to_torch") and _torch() else True  # fmt: skip
    # embed returns the synthesized token of a lexicon form, never an augmented one
    sounds.engines = {}
    labels = sounds.speakers["label"].to_list()
    embedded = sounds.embed(sounds.words["arpabet"].to_list(), labels)
    assert embedded.shape == (6, 3, 8)
    for w in range(6):
        for s in range(3):
            row = sounds._stored[(f"W.{w + 1}", labels[s], 1)]
            assert not sounds.token_augmented[row]
            assert np.allclose(embedded[w, s], sounds.tokens[row], atol=1e-5)


def _torch() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_parselmouth
@needs_pyroomacoustics
def test_reverberation_and_manipulation_recipes_with_the_stand_in_engine(tmp_path):
    recipes = [
        {"name": "room", "reverberation": {"rt60": [0.3, 0.6], "room_m": [4, 6]}},
        {
            "name": "voice",
            "manipulation": {"formant_shift_ratio": [0.9, 1.1], "duration_factor": [1.2, 1.4]},
        },  # fmt: skip
        {"name": "higher", "speed_pitch": {"speed": 1.0, "pitch_semitones": [3, 5]}},
    ]
    run = stand_in_run(tmp_path, stand_in_config(tmp_path, recipes, proportion=0.3))
    synthesis = run.synthesis
    by_source = {t.label: t for t in synthesis.tokens if not t.augmentation}
    seen = set()
    for token in synthesis.tokens:
        if not token.augmentation:
            continue
        record = json.loads(token.augmentation)
        source = by_source[record["source"]]
        seen.add(record["recipe"])
        if record["recipe"] == "room":
            assert 0.3 <= record["reverberation"]["rt60"] <= 0.6
            assert all(4 <= d <= 6 for d in record["reverberation"]["room_m"])
            assert token.duration >= source.duration  # the room's tail, less the trimming
        elif record["recipe"] == "voice":
            factor = record["manipulation"]["duration_factor"]
            assert token.duration == pytest.approx(source.duration * factor, rel=0.05)
        else:
            assert token.duration == pytest.approx(source.duration, rel=0.02)
    assert seen == {"room", "voice", "higher"}
    assert synthesis.augmentation["skipped"] == []


@pytest.fixture(scope="module")
def augmented_tiny(tmp_path_factory):
    """The tiny configuration with every kind of recipe, on the real engines."""
    folder = tmp_path_factory.mktemp("augmented")
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["name"] = "tiny_augmented"
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(folder / "cache")
    data["closed_class"] = None
    data["augmentation"] = {
        "recipes": [
            {"name": "noisy", "noise": {"snr_db": [0, 20]}},
            {"name": "room", "reverberation": {"rt60": [0.3, 0.8]}},
            {"name": "pitched", "manipulation": {"pitch_median_hz": [100, 250]}},
        ],
        "proportion": 0.5,
    }
    config = parse_config(data, "tiny_augmented")
    run = run_forms(config)
    run_synthesis(run)
    return run


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_espeak
@needs_piper
@needs_parselmouth
@needs_pyroomacoustics
def test_praat_manipulation_moves_the_median_pitch_of_real_clips(augmented_tiny):
    from semantic_world.wordforms import praat

    synthesis = augmented_tiny.synthesis
    errors = []
    for token in synthesis.tokens:
        if not token.augmentation:
            continue
        record = json.loads(token.augmentation)
        if record["recipe"] != "pitched":
            continue
        target = record["manipulation"]["pitch_median_hz"]
        measured = praat.measure_pitch(synthesis.audio(token), synthesis.sample_rate)
        errors.append(abs(measured.median_hz / target - 1))
    errors = np.array(errors)
    assert len(errors) > 20
    # the median clip lands within 5% of its target; short clips with few voiced frames drift
    # more, and the share within 5% is reported
    assert np.median(errors) < 0.05
    assert np.mean(errors <= 0.05) > 0.8, f"{np.mean(errors <= 0.05):.2f} within 5%"


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_espeak
@needs_piper
@needs_parselmouth
@needs_pyroomacoustics
def test_augmented_snr_on_real_clips(augmented_tiny):
    exact, estimated = snr_errors(augmented_tiny, "noisy")
    assert len(exact) > 20
    assert max(abs(e) for e in exact) < 0.05
    # a least-squares estimate from the clean clip alone is within about half a decibel
    assert np.median(np.abs(estimated)) < 0.2 and max(abs(e) for e in estimated) < 1.0


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_espeak
@needs_piper
def test_all_stores_the_modulation_front_end_of_the_tiny_configuration(tmp_path, capsys):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(tmp_path / "cache")
    data["wordforms"]["count"] = 4
    data["closed_class"] = None
    data["embeddings"].append(
        {"name": "modulation_fixed", "encoder": "fixed", "frontend": "modulation", "pca_dims": 8}
    )
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["embed", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "front end modulation: " in text and "16 channels" in text
    assert "embedding modulation_fixed (fixed): 8 dimensions, computed" in text


# ---------------------------------------------------------------------------------------------
# Word embeddings, achieved values, and the evaluation with augmentation
# ---------------------------------------------------------------------------------------------


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_word_embeddings_leave_augmented_tokens_out_unless_asked(tmp_path):
    from semantic_world.wordforms.embeddings import token_layout, word_embedding_tokens, word_means

    config = stand_in_config(tmp_path, NOISE_AND_SPEED)
    assert config.word_embeddings.tokens == "clean"
    run = stand_in_run(tmp_path, config)
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / "run")
    stores = compute_embeddings(config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run")  # fmt: skip
    token_words, _, train = token_layout(run.lexicon.words, run.synthesis)
    clean = np.array([not t.augmentation for t in run.synthesis.tokens])
    chosen = word_embedding_tokens(config, run.synthesis, train)
    assert np.array_equal(chosen, train & clean) and chosen.sum() == 24
    store = stores["cochleagram_fixed"]
    assert np.allclose(store.types, word_means(store.tokens, token_words, train & clean, 6), atol=1e-6)  # fmt: skip
    assert store.meta["word_embedding_tokens"] == "clean"
    assert "without augmented tokens" in store.meta["word_embedding"]
    assert store.meta["projection"]["fitted_tokens"] == 24
    # with every token, the means change and the fingerprint with them
    data = yaml.safe_load(config.to_yaml())
    data["word_embeddings"] = {"tokens": "all"}
    every = parse_config(data, "all")
    stores = compute_embeddings(every, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run_all")  # fmt: skip
    store_all = stores["cochleagram_fixed"]
    assert np.array_equal(store_all.tokens, store.tokens)  # the projection is still fitted clean
    assert np.allclose(store_all.types, word_means(store.tokens, token_words, train, 6), atol=1e-6)  # fmt: skip
    assert not np.allclose(store_all.types, store.types)
    assert store_all.meta["word_embedding_tokens"] == "all"
    with pytest.raises(ConfigError) as info:
        parse_config({"word_embeddings": {"tokens": "some"}}, "x")
    assert info.value.field == "word_embeddings.tokens"


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_parselmouth
@needs_pyroomacoustics
def test_achieved_values_are_recorded_beside_their_targets(tmp_path):
    recipes = [
        {"name": "noisy", "noise": {"snr_db": [5, 15]}},
        {"name": "room", "reverberation": {"rt60": [0.3, 0.6], "room_m": [4, 6]}},
        {"name": "faster", "speed_pitch": {"speed": [1.1, 1.3], "pitch_semitones": [1, 3]}},
        {"name": "longer", "manipulation": {"duration_factor": [1.2, 1.4], "pitch_range_factor": [0.5, 0.8]}},
    ]  # fmt: skip
    run = stand_in_run(tmp_path, stand_in_config(tmp_path, recipes, proportion=0.4))
    synthesis = run.synthesis
    seen = set()
    for token in synthesis.tokens:
        if not token.augmentation:
            assert token.achieved == ""
            continue
        record = json.loads(token.augmentation)
        achieved = json.loads(token.achieved)
        seen.add(record["recipe"])
        for entry in achieved.values():
            assert set(entry) == {"target", "measured", "miss"}
            if entry["measured"] is not None and abs(entry["target"]) > 1e-3:
                # the values are rounded, so a small target gives a rough miss
                assert entry["miss"] == pytest.approx(abs(entry["measured"] / entry["target"] - 1), rel=0.01, abs=1e-5)  # fmt: skip
        if record["recipe"] == "noisy":
            assert set(achieved) == {"snr_db"}
            assert achieved["snr_db"]["target"] == record["noise"]["snr_db"]
            assert achieved["snr_db"]["miss"] < 1e-6  # exact by construction
        elif record["recipe"] == "room":
            assert set(achieved) == {"rt60_s"}
            assert achieved["rt60_s"]["target"] == record["reverberation"]["rt60"]
            assert achieved["rt60_s"]["measured"] > 0.1
        elif record["recipe"] == "faster":
            assert set(achieved) == {"speed_duration_s", "speed_pitch_median_hz"}
            assert achieved["speed_duration_s"]["miss"] < 0.01
            assert achieved["speed_duration_s"]["measured"] == pytest.approx(token.duration, abs=1e-5)  # fmt: skip
        else:
            assert set(achieved) == {"duration_s", "pitch_range_semitones"}
            assert achieved["duration_s"]["miss"] < 0.01
    assert seen == {"noisy", "room", "faster", "longer"}
    summary = synthesis.augmentation["achieved"]
    assert set(summary) == {"snr_db", "rt60_s", "speed_duration_s", "speed_pitch_median_hz", "duration_s", "pitch_range_semitones"}  # fmt: skip
    for counts in summary.values():
        assert set(counts) == {"tokens", "unmeasured", "over_5_percent", "over_10_percent"}
        assert counts["over_10_percent"] <= counts["over_5_percent"] <= counts["tokens"]
    assert summary["snr_db"]["over_5_percent"] == 0
    assert summary["duration_s"]["over_5_percent"] == 0
    # the run folder has the column, and a second run reads the values from the cache
    run.write(tmp_path / "run")
    table = pl.read_csv(tmp_path / "run" / "tokens.csv")
    assert "achieved" in table.columns
    assert json.loads(table.filter(pl.col("augmentation").is_not_null())["achieved"][0])
    again = stand_in_run(tmp_path, stand_in_config(tmp_path, recipes, proportion=0.4))
    assert again.synthesis.augmentation["read_from_cache"] == len(
        [t for t in synthesis.tokens if t.augmentation]
    )
    assert [t.achieved for t in again.synthesis.tokens] == [t.achieved for t in synthesis.tokens]


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_evaluation_by_token_set_and_robustness(tmp_path):
    from semantic_world.wordforms.evaluate import evaluate_embeddings, robustness

    # the measure itself: a token nearest its own word retrieves it
    types = np.eye(3)
    tokens = np.array([[0.9, 0.1, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 0.9], [0.5, 0.6, 0.0]])
    ap, top1, chance = robustness(tokens, np.array([0, 1, 2, 0]), types, np.ones(3, dtype=bool))
    assert top1 == 0.75 and chance == pytest.approx(1 / 3) and 0.5 < ap < 1.0
    assert np.isnan(robustness(tokens, np.array([0, 1, 2, 0]), types, np.zeros(3, dtype=bool))[0])
    config = stand_in_config(tmp_path, NOISE_AND_SPEED)
    run = stand_in_run(tmp_path, config)
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / "run")
    stores = compute_embeddings(config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run")  # fmt: skip
    table = evaluate_embeddings(stores, run.lexicon.words, run.synthesis, run.streams.eval, config=config)  # fmt: skip
    sets = ["clean", "augmented", "all", "recipe:noisy", "recipe:faster"]
    assert table["tokens"].to_list() == sets * 2
    assert table["kind"].to_list() == ["content"] * 10
    by = {(r["embedding"], r["tokens"]): r for r in table.iter_rows(named=True)}
    for name in stores:
        assert by[name, "clean"]["tokens_evaluated"] == 36
        assert by[name, "augmented"]["tokens_evaluated"] == 72
        assert by[name, "all"]["tokens_evaluated"] == 108
        assert by[name, "recipe:noisy"]["tokens_evaluated"] == 36
        for tokens in sets:
            row = by[name, tokens]
            assert row["robustness_chance"] == pytest.approx(1 / 6)
            assert 0 <= row["robustness_top1"] <= 1 and row["robustness_ap"] > row["robustness_chance"]  # fmt: skip
        # clean tokens retrieve their own word best: they are part of its mean
        assert by[name, "clean"]["robustness_ap"] >= by[name, "augmented"]["robustness_ap"]
    # a run without augmentation has the clean rows only, and every column
    plain = stand_in_config(tmp_path / "plain", NOISE_AND_SPEED)
    plain = parse_config({**yaml.safe_load(plain.to_yaml()), "augmentation": None}, "plain")
    run = stand_in_run(tmp_path / "plain", plain)
    run.frontends = compute_frontends(plain, run.synthesis, tmp_path / "plain" / "run")
    stores = compute_embeddings(plain, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "plain" / "run")  # fmt: skip
    table = evaluate_embeddings(stores, run.lexicon.words, run.synthesis, run.streams.eval, config=plain)  # fmt: skip
    assert table["tokens"].to_list() == ["clean", "clean"]
    assert table["robustness_ap"].drop_nulls().len() == 2
