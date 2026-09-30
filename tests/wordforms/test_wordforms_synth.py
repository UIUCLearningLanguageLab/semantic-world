"""Layer 2: speakers, tokens, audio processing, the cache, and the two engines."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
import yaml
from wordforms_support import (
    DATA,
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_piper,
    needs_wordfreq,
)

from semantic_world.wordforms import Run, run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.config import parse_config
from semantic_world.wordforms.io import SPEAKER_COLUMNS, TOKEN_COLUMNS
from semantic_world.wordforms.streams import Streams
from semantic_world.wordforms.synth import (
    Synthesis,
    cache_key,
    synthesize_lexicon,
    token_perturbation,
)
from semantic_world.wordforms.synth import audio as audio_tools
from semantic_world.wordforms.synth.speakers import Speaker, draw_speakers, held_out_count

RATE = 16000


class ToneEngine:
    """A stand-in engine: a tone between two silences, at 22,050 Hz. The tone's frequency comes
    from the phonemes and the voice, and its length from the duration scale."""

    name = "espeak"

    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def phonemes(self, word) -> str:
        return word.espeak

    def settings_for(self, speaker, duration_scale):
        return {"voice": speaker.identity, "scale": round(duration_scale, 6)}

    def synthesize(self, phonemes, settings):
        self.calls += 1
        digest = hashlib.sha256(f"{phonemes}|{settings['voice']}".encode()).digest()
        frequency = 200 + digest[0] * 4
        n = int(22050 * 0.3 * settings["scale"])
        tone = 0.5 * np.sin(2 * np.pi * frequency * np.arange(n) / 22050)
        silence = np.zeros(2205)
        return np.concatenate([silence, tone, silence, silence]).astype(np.float32), 22050

    def reset(self, seed) -> None:
        self.resets += 1

    def provenance(self):
        return {"program": "tone"}


def espeak_only(tmp_path, **wordforms):
    """A small configuration with the espeak-ng engine only and a cache in ``tmp_path``."""
    return parse_config(
        {
            "name": "synth_test",
            "wordforms": {"count": 6, **wordforms},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 2,
                "held_out_speaker_proportion": 0.34,
                "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
            },
            "embeddings": [],
        },
        "synth_test",
    )


def check_clips(synthesis: Synthesis, config) -> None:
    """Every clip is mono, 16 kHz, unclipped, and trimmed."""
    import soundfile

    trim = config.synthesis.trim
    level = config.synthesis.level
    margin = int(round(trim.margin_ms * RATE / 1000))
    assert synthesis.tokens
    for token in synthesis.tokens:
        path = synthesis.cache_dir / token.cache_path
        info = soundfile.info(path)
        assert info.channels == 1 and info.samplerate == RATE and info.format == "FLAC"
        clip = synthesis.audio(token)
        assert clip.ndim == 1 and clip.dtype == np.float32
        assert token.duration == pytest.approx(len(clip) / RATE, abs=1e-6)
        assert 0.05 < token.duration < 3.0
        # unclipped: the peak is within the limit, and the RMS level is the target unless the
        # peak limit held the clip down
        peak = float(np.abs(clip).max())
        rms = audio_tools.rms_db(clip)
        assert peak <= level.max_peak + 1e-4
        assert rms == pytest.approx(level.rms_db, abs=0.05) or (
            rms < level.rms_db and peak == pytest.approx(level.max_peak, abs=1e-3)
        )
        assert token.peak == pytest.approx(peak, abs=1e-5)
        assert token.rms_db == pytest.approx(rms, abs=1e-3)
        assert 1 <= token.tries <= 5
        before, after = audio_tools.silence_margins(clip, trim.threshold_db)
        assert before <= margin + 1 and after <= margin + 1
        assert token.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------------------------


def tone(frequency, seconds, rate, amplitude=0.5):
    return (
        amplitude * np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate)
    ).astype(np.float32)


def peak_frequency(clip, rate):
    spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip))))
    return float(np.fft.rfftfreq(len(clip), 1 / rate)[int(np.argmax(spectrum))])


@needs_audio
def test_resample_keeps_frequency_and_scales_length():
    clip = tone(1000, 0.5, 22050)
    out = audio_tools.resample(clip, 22050, RATE)
    assert out.dtype == np.float32
    assert abs(len(out) - 8000) <= 1
    assert peak_frequency(out, RATE) == pytest.approx(1000, abs=5)
    assert np.array_equal(audio_tools.resample(out, RATE, RATE), out)


@needs_audio
def test_reading_at_a_higher_rate_raises_pitch_and_shortens():
    clip = tone(1000, 0.5, 22050)
    factor = 2 ** (1 / 12)  # one semitone
    out = audio_tools.resample(clip, 22050 * factor, RATE)
    assert len(out) == pytest.approx(8000 / factor, abs=2)
    assert peak_frequency(out, RATE) == pytest.approx(1000 * factor, abs=5)


def test_trim_removes_silence_and_keeps_the_margin():
    silence = np.zeros(1600, dtype=np.float32)
    clip = np.concatenate([silence, tone(440, 0.25, RATE), silence, silence])
    trimmed = audio_tools.trim(clip, RATE, -40, 20)
    assert len(trimmed) == pytest.approx(4000 + 2 * 320, abs=4)
    before, after = audio_tools.silence_margins(trimmed, -40)
    assert before <= 321 and after <= 321
    # quiet sound below the threshold is silence
    quiet = np.concatenate([0.001 * tone(440, 0.1, RATE), tone(440, 0.25, RATE)])
    assert len(audio_tools.trim(quiet, RATE, -40, 0)) == pytest.approx(4000, abs=4)
    assert len(audio_tools.trim(quiet, RATE, -70, 0)) > 5000
    with pytest.raises(audio_tools.AudioError, match="silent"):
        audio_tools.trim(silence, RATE, -40, 20)


def test_set_level_reaches_the_target_rms():
    clip = audio_tools.set_level(tone(440, 0.1, RATE, amplitude=0.2), -24, 0.9)
    assert clip.dtype == np.float32
    assert audio_tools.rms_db(clip) == pytest.approx(-24, abs=1e-3)
    assert audio_tools.peak(clip) == pytest.approx(10 ** (-24 / 20) * np.sqrt(2), rel=1e-3)
    assert audio_tools.rms_db(tone(440, 0.1, RATE, amplitude=1.0)) == pytest.approx(-3.01, abs=0.01)
    with pytest.raises(audio_tools.AudioError):
        audio_tools.set_level(np.zeros(10, dtype=np.float32), -24, 0.9)


def test_set_level_peak_guard():
    # a click in quiet noise has a high peak for its RMS level
    rng = np.random.default_rng(0)
    clip = (0.001 * rng.normal(size=8000)).astype(np.float32)
    clip[4000] = 0.5
    out = audio_tools.set_level(clip, -24, 0.9)
    assert audio_tools.peak(out) == pytest.approx(0.9, abs=1e-6)
    assert audio_tools.rms_db(out) < -24
    # a louder target than the peak allows is held down too
    loud = audio_tools.set_level(tone(440, 0.1, RATE), -1, 0.9)
    assert audio_tools.peak(loud) == pytest.approx(0.9, abs=1e-6)


@needs_audio
def test_flac_round_trip(tmp_path):
    import soundfile

    clip = audio_tools.set_level(tone(440, 0.2, RATE), -24, 0.9)
    path = tmp_path / "a" / "clip.flac"
    audio_tools.write_flac(path, clip, RATE)
    back, rate = audio_tools.read_flac(path)
    assert rate == RATE and back.dtype == np.float32 and back.shape == clip.shape
    assert np.abs(back - clip).max() < 1e-4  # 16-bit quantization
    info = soundfile.info(path)
    assert info.channels == 1 and info.subtype == "PCM_16"
    assert audio_tools.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(path.parent.glob("*.tmp.flac"))


# ---------------------------------------------------------------------------------------------
# Speakers and tokens
# ---------------------------------------------------------------------------------------------


def test_default_speakers(default_config):
    speakers = draw_speakers(default_config, Streams(1), piper_voice_speakers=904)
    assert [s.label for s in speakers] == [f"S.{i}" for i in range(1, 46)]
    piper = [s for s in speakers if s.engine == "piper"]
    espeak = [s for s in speakers if s.engine == "espeak"]
    assert speakers == piper + espeak and len(piper) == 40 and len(espeak) == 5
    assert len({s.speaker_id for s in piper}) == 40
    assert all(0 <= s.speaker_id < 904 and s.voice == "en_US-libritts_r-medium" for s in piper)
    assert [s.variant for s in espeak] == ["m1", "m3", "m7", "f2", "f4"]
    assert all(35 <= s.pitch <= 65 and 150 <= s.rate <= 190 for s in espeak)
    # a proportion of each engine's speakers is held out
    assert sum(s.held_out for s in piper) == 8 and sum(s.held_out for s in espeak) == 1
    assert len({s.identity for s in speakers}) == 45
    assert speakers[0].record()["split"] in ("train", "held_out")
    assert list(speakers[0].record()) == list(SPEAKER_COLUMNS)


def test_speakers_are_seeded_and_engines_are_independent(default_config):
    a = draw_speakers(default_config, Streams(1), 904)
    assert a == draw_speakers(default_config, Streams(1), 904)
    assert a != draw_speakers(default_config, Streams(2), 904)
    fewer = parse_config({"synthesis": {"engines": {"piper": {"speakers": 10}, "espeak": {}}}}, "x")
    b = draw_speakers(fewer, Streams(1), 904)
    assert [s.identity for s in b if s.engine == "espeak"] == [
        s.identity for s in a if s.engine == "espeak"
    ]
    no_piper = parse_config({"synthesis": {"engines": {"piper": None, "espeak": {}}}}, "x")
    c = draw_speakers(no_piper, Streams(1))
    assert [s.identity for s in c] == [s.identity for s in a if s.engine == "espeak"]
    assert [s.label for s in c] == ["S.1", "S.2", "S.3", "S.4", "S.5"]


def test_speaker_errors_and_held_out_counts(default_config):
    with pytest.raises(ValueError, match="has 3 speakers, fewer than 40"):
        draw_speakers(default_config, Streams(1), piper_voice_speakers=3)
    with pytest.raises(ValueError, match="needed"):
        draw_speakers(default_config, Streams(1))
    assert held_out_count(40, 0.2) == 8 and held_out_count(5, 0.2) == 1
    assert held_out_count(3, 0.34) == 1 and held_out_count(1, 0.9) == 0
    assert held_out_count(4, 1.0) == 3  # at least one training speaker stays
    more = parse_config({"synthesis": {"engines": {"piper": None, "espeak": {"speakers": 7}}}}, "x")
    speakers = draw_speakers(more, Streams(1))
    assert [s.variant for s in speakers] == ["m1", "m3", "m7", "f2", "f4", "m1", "m3"]


@needs_cmudict
@needs_wordfreq
def test_token_perturbations_are_small_seeded_and_stable(tmp_path):
    config = espeak_only(tmp_path)
    streams = Streams(config.seed)
    words = run_forms(config).lexicon.words
    speakers = draw_speakers(config, streams)
    draws = [
        token_perturbation(config, streams, w, s, k)
        for w in words
        for s in speakers
        for k in (1, 2)
    ]
    assert all(0.95 <= rate <= 1.05 and -0.5 <= pitch <= 0.5 for rate, pitch in draws)
    assert len(set(draws)) == len(draws)
    # a new try draws a new perturbation
    first = token_perturbation(config, streams, words[0], speakers[0], 1)
    assert first == token_perturbation(config, streams, words[0], speakers[0], 1, 1)
    retries = {token_perturbation(config, streams, words[0], speakers[0], 1, t) for t in (2, 3, 4)}
    assert len(retries | {first}) == 4
    again = token_perturbation(config, Streams(config.seed), words[2], speakers[1], 2)
    assert again == token_perturbation(config, streams, words[2], speakers[1], 2)
    assert again != token_perturbation(config, Streams(9), words[2], speakers[1], 2)
    # the label of a speaker does not matter, only the voice
    relabeled = Speaker(**{**speakers[1].__dict__, "label": "S.99"})
    assert token_perturbation(config, streams, words[2], relabeled, 2) == again
    still = parse_config(
        {"synthesis": {"token_perturbation": {"rate": 0, "pitch_semitones": 0}}}, "x"
    )
    assert token_perturbation(still, streams, words[0], speakers[0], 1) == (1.0, 0.0)


def test_cache_key_depends_on_everything_that_shapes_the_clip(default_config):
    settings = {"voice": "en-us+m1", "pitch": 50, "rate": 170}
    key = cache_key(default_config, "espeak", "k'at", settings, 0.1, 1)
    assert key == cache_key(default_config, "espeak", "k'at", dict(settings), 0.1, 1)
    assert len(key) == 64
    others = [
        cache_key(default_config, "piper", "k'at", settings, 0.1, 1),
        cache_key(default_config, "espeak", "b'at", settings, 0.1, 1),
        cache_key(default_config, "espeak", "k'at", {**settings, "rate": 171}, 0.1, 1),
        cache_key(default_config, "espeak", "k'at", settings, 0.2, 1),
        cache_key(default_config, "espeak", "k'at", settings, 0.1, 2),
        cache_key(default_config, "espeak", "k'at", settings, 0.1, 1, 2),
        cache_key(
            parse_config({"synthesis": {"trim": {"margin_ms": 30}}}, "x"),
            "espeak", "k'at", settings, 0.1, 1,
        ),
        cache_key(
            parse_config({"synthesis": {"level": {"rms_db": -20}}}, "x"),
            "espeak", "k'at", settings, 0.1, 1,
        ),
    ]  # fmt: skip
    assert len({key, *others}) == 9
    assert key == cache_key(default_config, "espeak", "k'at", settings, 0.1, 1, 1)


# ---------------------------------------------------------------------------------------------
# The run and the cache, with a stand-in engine
# ---------------------------------------------------------------------------------------------


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_run_with_a_stand_in_engine(tmp_path):
    config = espeak_only(tmp_path)
    run = run_forms(config)
    engine = ToneEngine()
    synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": engine}
    )
    assert len(synthesis.speakers) == 3 and sum(s.held_out for s in synthesis.speakers) == 1
    assert len(synthesis.tokens) == 6 * 3 * 2
    assert synthesis.synthesized == 36 and synthesis.cached == 0
    labels = [t.label for t in synthesis.tokens]
    assert labels[:3] == ["W.1.S.1.1", "W.1.S.1.2", "W.1.S.2.1"] and labels[-1] == "W.6.S.3.2"
    assert len({t.cache_path for t in synthesis.tokens}) == 36
    check_clips(synthesis, config)
    # the rate perturbation changes the duration
    assert len({t.duration for t in synthesis.tokens}) > 10
    # the reproducibility of the engine is tested and recorded
    assert synthesis.reproducibility == {
        "espeak": {
            "same_settings_identical": True,
            "fresh_session_identical": True,
            "clips_tested": 9,
            "varying_speakers": [],
        }
    }
    assert synthesis.summary()["models"] == {"espeak": {"program": "tone"}}


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_reproducibility_check_detects_a_varying_engine(tmp_path):
    from semantic_world.wordforms.synth import check_reproducibility

    class NoisyEngine(ToneEngine):
        """Noise that a fresh session repeats, like Piper's."""

        def reset(self, seed):
            self.rng = np.random.default_rng(seed)

        def synthesize(self, phonemes, settings):
            clip, rate = super().synthesize(phonemes, settings)
            return clip + self.rng.normal(0, 0.01, len(clip)).astype(np.float32), rate

    class UnseededEngine(NoisyEngine):
        def reset(self, seed):
            self.rng = np.random.default_rng()

    config = espeak_only(tmp_path)
    run = run_forms(config)
    speakers = draw_speakers(config, run.streams)
    noisy = check_reproducibility(NoisyEngine(), run.lexicon.words, speakers, 7)
    assert noisy["same_settings_identical"] is False and noisy["fresh_session_identical"] is True
    assert noisy["varying_speakers"] == sorted(s.identity for s in speakers)
    unseeded = check_reproducibility(UnseededEngine(), run.lexicon.words, speakers, 7)
    assert unseeded["same_settings_identical"] is False
    assert unseeded["fresh_session_identical"] is False


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_second_run_reads_the_cache_without_synthesizing(tmp_path):
    config = espeak_only(tmp_path)
    run = run_forms(config)
    engine = ToneEngine()
    first = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": engine}, check=False
    )
    calls = engine.calls
    assert calls == 36
    second = synthesize_lexicon(
        config, Streams(config.seed), run.lexicon.words, engines={"espeak": engine}, check=False
    )
    assert engine.calls == calls  # nothing was synthesized
    assert second.synthesized == 0 and second.cached == 36
    assert second.tokens == first.tokens
    # more words reuse the clips of the words already synthesized
    bigger = espeak_only(tmp_path, count=8)
    bigger_run = run_forms(bigger)
    assert [w.arpabet for w in bigger_run.lexicon.words[:6]] == [
        w.arpabet for w in run.lexicon.words
    ]
    third = synthesize_lexicon(
        bigger,
        bigger_run.streams,
        bigger_run.lexicon.words,
        engines={"espeak": engine},
        check=False,
    )
    assert third.cached == 36 and third.synthesized == 12
    assert third.tokens[:36] == first.tokens


class StretchEngine(ToneEngine):
    """Makes chosen clips three times too long, like a neural engine that now and then stretches
    a word. ``long_scales`` holds the duration scales of the tries to stretch; each try has its
    own perturbation, so its own scale."""

    def __init__(self, phonemes: str, voice: str, long_scales: set[float]) -> None:
        super().__init__()
        self.target = (phonemes, voice)
        self.long_scales = long_scales

    def synthesize(self, phonemes, settings):
        clip, rate = super().synthesize(phonemes, settings)
        if (phonemes, settings["voice"]) == self.target and settings["scale"] in self.long_scales:
            return np.concatenate([clip[:2205], np.tile(clip[2205:-4410], 3), clip[-4410:]]), rate
        return clip, rate


def stretch_run(tmp_path, long_tries, **duration_check):
    data = espeak_only(tmp_path).resolved()
    if duration_check.get("off"):
        data["synthesis"]["duration_check"] = None
    elif duration_check:
        data["synthesis"]["duration_check"] = duration_check
    config = parse_config(data, "stretch")
    run = run_forms(config)
    speakers = draw_speakers(config, run.streams)
    word = run.lexicon.words[1]
    long_scales = set()
    for k in (1, 2):
        for attempt in range(1, min(long_tries, 10) + 1):
            rate, pitch = token_perturbation(config, run.streams, word, speakers[0], k, attempt)
            long_scales.add(round(2.0 ** (pitch / 12.0) / rate, 6))
    engine = StretchEngine(word.espeak, speakers[0].identity, long_scales)
    synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": engine}, check=False
    )
    target = [t for t in synthesis.tokens if t.word == word.label and t.speaker == "S.1"]
    others = [t for t in synthesis.tokens if t not in target]
    return config, run, engine, synthesis, target, others


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_a_long_clip_is_synthesized_again(tmp_path):
    config, run, engine, synthesis, target, others = stretch_run(tmp_path, long_tries=1)
    assert len(target) == 2 and all(t.tries == 2 for t in target)
    assert all(t.tries == 1 for t in others)
    median = float(np.median([t.duration for t in synthesis.tokens if t.word == target[0].word]))
    assert all(t.duration <= 1.8 * median for t in synthesis.tokens if t.word == target[0].word)
    assert engine.calls == 36 + 2 and synthesis.synthesized == 38
    assert synthesis.over_limit == 0
    check = synthesis.summary()["duration_check"]
    assert check == {"retried": 2, "still_over_limit": 0, "tries": {"1": 34, "2": 2}}
    check_clips(synthesis, config)
    # the labels and the order of the tokens are unchanged
    assert [t.label for t in synthesis.tokens][:3] == ["W.1.S.1.1", "W.1.S.1.2", "W.1.S.2.1"]
    # a second run takes every try from the cache and makes the same choices
    again = synthesize_lexicon(
        config, Streams(config.seed), run.lexicon.words, engines={"espeak": engine}, check=False
    )
    assert engine.calls == 38 and again.synthesized == 0 and again.cached == 38
    assert again.tokens == synthesis.tokens


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_the_shortest_try_is_kept_when_none_passes(tmp_path):
    config, run, engine, synthesis, target, others = stretch_run(tmp_path, long_tries=99)
    assert all(t.tries == 5 for t in target)
    assert synthesis.over_limit == 2
    assert synthesis.summary()["duration_check"]["still_over_limit"] == 2
    assert synthesis.summary()["duration_check"]["tries"] == {"1": 34, "5": 2}
    assert engine.calls == 36 + 2 * 4
    # the kept clip is the shortest of the five tries
    speaker = synthesis.speakers[0]
    word = run.lexicon.words[1]
    for k, token in zip((1, 2), target, strict=True):
        rates = [
            token_perturbation(config, run.streams, word, speaker, k, t)[0] for t in range(1, 6)
        ]
        assert token.rate_factor == max(rates)  # the fastest try is the shortest


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_duration_check_settings(tmp_path):
    _, _, engine, synthesis, target, _ = stretch_run(tmp_path / "off", long_tries=99, off=True)
    assert all(t.tries == 1 for t in synthesis.tokens) and engine.calls == 36
    assert synthesis.over_limit == 0
    _, _, engine, synthesis, target, _ = stretch_run(
        tmp_path / "one", long_tries=99, max_ratio=1.8, max_tries=1
    )
    assert all(t.tries == 1 for t in synthesis.tokens) and synthesis.over_limit == 2
    _, _, engine, synthesis, target, _ = stretch_run(
        tmp_path / "loose", long_tries=99, max_ratio=4.0, max_tries=5
    )
    assert all(t.tries == 1 for t in synthesis.tokens) and synthesis.over_limit == 0
    _, _, engine, synthesis, target, _ = stretch_run(
        tmp_path / "three", long_tries=2, max_ratio=1.8, max_tries=5
    )
    assert [t.tries for t in target] == [3, 3] and synthesis.over_limit == 0


@needs_cmudict
@needs_wordfreq
def test_long_synthesis_flags_words_that_piper_stretches(tmp_path):
    from pathlib import Path

    from semantic_world.wordforms.synth import Token, flag_long_synthesis

    config = espeak_only(tmp_path, count=12, syllables={2: 1})
    words = run_forms(config).lexicon.words
    assert all(w.long_synthesis is None for w in words)

    def tokens(engine, long_word):
        result = []
        for i, word in enumerate(words):
            for k, speaker in enumerate(("S.1", "S.2", "S.3")):
                duration = 0.5 + 0.01 * i + 0.02 * k
                if word.label == long_word:
                    duration *= 2 if speaker != "S.3" else 1  # long for most speakers
                result.append(
                    Token(f"{word.label}.{speaker}.1", word.label, speaker, engine, "", {}, 1.0,
                          0.0, duration, "", "")
                )  # fmt: skip
        return result

    synthesis = Synthesis(Path("."), RATE, [], tokens("piper", "W.4"))
    flag_long_synthesis(config, words, synthesis)
    assert [w.label for w in words if w.long_synthesis] == ["W.4"]
    assert all(w.long_synthesis is False for w in words if w.label != "W.4")
    record = synthesis.long_synthesis
    assert record["ratio"] == 1.6 and list(record["flagged"]) == ["W.4"]
    assert record["flagged"]["W.4"]["ratio"] > 1.6
    assert list(record["median_seconds_by_syllables"]) == [2]
    assert synthesis.summary()["long_synthesis"] == record
    # a looser rule flags nothing
    data = config.resolved()
    data["synthesis"]["long_synthesis_ratio"] = 2.5
    for word in words:
        word.long_synthesis = None
    flag_long_synthesis(parse_config(data, "x"), words, synthesis)
    assert not any(w.long_synthesis for w in words) and synthesis.long_synthesis["flagged"] == {}
    # without Piper tokens, or without a ratio, the flag stays unset
    for word in words:
        word.long_synthesis = None
    flag_long_synthesis(config, words, Synthesis(Path("."), RATE, [], tokens("espeak", "W.4")))
    assert all(w.long_synthesis is None for w in words)
    data["synthesis"]["long_synthesis_ratio"] = None
    flag_long_synthesis(parse_config(data, "x"), words, synthesis)
    assert all(w.long_synthesis is None for w in words)


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_a_silent_clip_is_an_error_that_names_the_token(tmp_path):
    class SilentEngine(ToneEngine):
        def synthesize(self, phonemes, settings):
            return np.zeros(2000, dtype=np.float32), 22050

    config = espeak_only(tmp_path)
    run = run_forms(config)
    with pytest.raises(audio_tools.AudioError, match=r"W\.1\.S\.1\.1 .*silent"):
        synthesize_lexicon(
            config, run.streams, run.lexicon.words, engines={"espeak": SilentEngine()}, check=False
        )


@needs_cmudict
@needs_wordfreq
@needs_audio
def test_run_folder_with_synthesis(tmp_path):
    config = espeak_only(tmp_path)
    run: Run = run_forms(config)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": ToneEngine()}
    )
    folder = run.write(tmp_path / "run")
    assert sorted(p.name for p in folder.iterdir()) == [
        "config.yaml",
        "speakers.csv",
        "summary.yaml",
        "tokens.csv",
        "words.csv",
    ]
    speakers = pl.read_csv(folder / "speakers.csv")
    assert speakers.columns == list(SPEAKER_COLUMNS) and speakers.height == 3
    assert speakers["split"].to_list().count("held_out") == 1
    assert speakers["speaker_id"].null_count() == 3  # espeak-ng speakers have no speaker ID
    # no Piper tokens: the long_synthesis flag is empty
    assert pl.read_csv(folder / "words.csv")["long_synthesis"].null_count() == 6
    tokens = pl.read_csv(folder / "tokens.csv")
    assert tokens.columns == list(TOKEN_COLUMNS) and tokens.height == 36
    assert tokens["label"][0] == "W.1.S.1.1"
    assert tokens["tries"].to_list() == [1] * 36
    assert tokens["rms_db"].max() <= -23.9 and tokens["peak"].max() <= 0.9
    assert set(tokens["word"]) == {f"W.{i}" for i in range(1, 7)}
    assert yaml.safe_load(tokens["settings"][0])["voice"].startswith("espeak/en-us+m1")
    assert all((tmp_path / "cache" / p).exists() for p in tokens["cache_path"])
    assert all(len(h) == 64 for h in tokens["sha256"])
    summary = yaml.safe_load((folder / "summary.yaml").read_text())["synthesis"]
    assert summary["tokens"] == 36 and summary["synthesized"] == 36
    assert summary["engines"] == {"espeak": {"speakers": 3, "held_out": 1}}
    assert summary["reproducibility"]["espeak"]["same_settings_identical"] is True
    assert summary["duration_check"] == {"retried": 0, "still_over_limit": 0, "tries": {"1": 36}}
    levels = summary["levels"]["espeak"]
    assert levels["clips"] == 36 and set(levels["peak"]) == {"min", "p01", "median", "p99", "max"}
    assert levels["rms_db"]["median"] == pytest.approx(-24, abs=0.05)
    written = yaml.safe_load((folder / "config.yaml").read_text())
    assert written["provenance"]["models"] == {"espeak": {"program": "tone"}}
    # a second write from the cache gives identical token and speaker tables
    again: Run = run_forms(config)
    again.synthesis = synthesize_lexicon(
        config, again.streams, again.lexicon.words, engines={"espeak": ToneEngine()}
    )
    other = again.write(tmp_path / "run2")
    assert (other / "tokens.csv").read_bytes() == (folder / "tokens.csv").read_bytes()
    assert (other / "speakers.csv").read_bytes() == (folder / "speakers.csv").read_bytes()


# ---------------------------------------------------------------------------------------------
# The real engines
# ---------------------------------------------------------------------------------------------


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_espeak
def test_espeak_engine(tmp_path):
    config = espeak_only(tmp_path)
    run = run_forms(config)
    synthesis = synthesize_lexicon(config, run.streams, run.lexicon.words)
    assert synthesis.synthesized == 36
    check_clips(synthesis, config)
    # the reproducibility is tested and recorded; the breathy variant f2 may vary now and then
    record = synthesis.reproducibility["espeak"]
    assert isinstance(record["same_settings_identical"], bool)
    assert isinstance(record["fresh_session_identical"], bool)
    assert record["clips_tested"] == 9
    assert all("+f2" in identity for identity in record["varying_speakers"])
    assert synthesis.engines["espeak"]["program"] == "espeak-ng"
    assert synthesis.engines["espeak"]["version"][0].isdigit()
    token = synthesis.tokens[0]
    assert token.phonemes == run.lexicon.words[0].espeak
    assert token.settings["voice"] == "en-us+m1" and 140 <= token.settings["rate"] <= 200
    # speakers and tokens differ, and a second run reads the cache
    assert len({t.sha256 for t in synthesis.tokens}) == 36
    again = synthesize_lexicon(config, Streams(config.seed), run.lexicon.words)
    assert again.synthesized == 0 and again.cached == 36 and again.tokens == synthesis.tokens
    # the variants without a breath setting are deterministic: a new cache gives the same files
    resolved = config.resolved()
    resolved["synthesis"]["cache_dir"] = str(tmp_path / "cache2")
    fresh = parse_config(resolved, "x")
    assert fresh.synthesis.piper is None
    rebuilt = synthesize_lexicon(fresh, Streams(config.seed), run.lexicon.words, check=False)
    steady = {s.label for s in synthesis.speakers if s.variant in ("m1", "m3")}
    assert len(steady) == 2
    assert [t.sha256 for t in rebuilt.tokens if t.speaker in steady] == [
        t.sha256 for t in synthesis.tokens if t.speaker in steady
    ]


def piper_only(tmp_path):
    return parse_config(
        {
            "name": "piper_test",
            "wordforms": {"count": 5},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 2,
                "engines": {"piper": {"speakers": 3, "voice_dir": str(VOICE_DIR)}, "espeak": None},
            },
            "embeddings": [],
        },
        "piper_test",
    )


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_piper
def test_piper_engine(tmp_path):
    config = piper_only(tmp_path)
    run = run_forms(config)
    synthesis = synthesize_lexicon(config, run.streams, run.lexicon.words)
    assert len(synthesis.speakers) == 3 and synthesis.synthesized == 30
    check_clips(synthesis, config)
    token = synthesis.tokens[0]
    assert token.phonemes == run.lexicon.words[0].ipa
    assert token.settings["noise_scale"] == 0.667 and token.settings["noise_w"] == 0.8
    assert token.settings["speaker_id"] == synthesis.speakers[0].speaker_id
    assert 0.9 < token.settings["length_scale"] < 1.1
    assert len({t.sha256 for t in synthesis.tokens}) == 30
    # the reproducibility is tested and recorded, whatever it is
    record = synthesis.reproducibility["piper"]
    assert isinstance(record["same_settings_identical"], bool)
    assert isinstance(record["fresh_session_identical"], bool)
    assert record["clips_tested"] == 9
    provenance = synthesis.engines["piper"]
    assert (
        provenance["voice"] == "en_US-libritts_r-medium" and len(provenance["voice_sha256"]) == 64
    )
    # a second run reads the cache without synthesizing
    again = synthesize_lexicon(config, Streams(config.seed), run.lexicon.words)
    assert again.synthesized == 0 and again.cached == 30 and again.tokens == synthesis.tokens


@needs_audio
def test_missing_piper_voice_names_the_download_command(tmp_path, capsys):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(tmp_path / "no_voices")
    data["synthesis"]["cache_dir"] = str(tmp_path / "cache")
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    pytest.importorskip("cmudict")
    pytest.importorskip("wordfreq")
    assert main(["synth", str(path), "--out", str(tmp_path / "run")]) == 1
    err = capsys.readouterr().err
    assert "python -m piper.download_voices en_US-libritts_r-medium" in err


@needs_cmudict
@needs_wordfreq
@needs_audio
@needs_espeak
@needs_piper
def test_synth_command_on_the_tiny_configuration(tmp_path, capsys):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(tmp_path / "cache")
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["synth", str(path), "--out", str(out)]) == 0
    assert "120 synthesized, 0 read from the cache" in capsys.readouterr().out
    words = pl.read_csv(out / "words.csv")
    assert words["long_synthesis"].dtype == pl.Boolean and words["long_synthesis"].null_count() == 0
    speakers = pl.read_csv(out / "speakers.csv")
    assert speakers["engine"].to_list() == ["piper"] * 3 + ["espeak"] * 3
    assert speakers["split"].to_list().count("held_out") == 2
    tokens = pl.read_csv(out / "tokens.csv")
    assert tokens.height == 20 * 6 * 1
    summary = yaml.safe_load((out / "summary.yaml").read_text())["synthesis"]
    assert set(summary["reproducibility"]) == {"piper", "espeak"}
    assert main(["all", str(path), "--out", str(out)]) == 0
    assert "0 synthesized, 120 read from the cache" in capsys.readouterr().out
    assert pl.read_csv(out / "tokens.csv").equals(tokens)
