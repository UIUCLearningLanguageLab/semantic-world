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
        peak = float(np.abs(clip).max())
        assert peak == pytest.approx(audio_tools.PEAK, abs=1e-3) and peak < 1.0
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


def test_normalize_peak():
    clip = audio_tools.normalize_peak(tone(440, 0.1, RATE, amplitude=0.2))
    assert float(np.abs(clip).max()) == pytest.approx(audio_tools.PEAK)
    with pytest.raises(audio_tools.AudioError):
        audio_tools.normalize_peak(np.zeros(10, dtype=np.float32))


@needs_audio
def test_flac_round_trip(tmp_path):
    import soundfile

    clip = audio_tools.normalize_peak(tone(440, 0.2, RATE))
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
        cache_key(
            parse_config({"synthesis": {"trim": {"margin_ms": 30}}}, "x"),
            "espeak", "k'at", settings, 0.1, 1,
        ),
    ]  # fmt: skip
    assert len({key, *others}) == 7


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
    tokens = pl.read_csv(folder / "tokens.csv")
    assert tokens.columns == list(TOKEN_COLUMNS) and tokens.height == 36
    assert tokens["label"][0] == "W.1.S.1.1"
    assert set(tokens["word"]) == {f"W.{i}" for i in range(1, 7)}
    assert yaml.safe_load(tokens["settings"][0])["voice"].startswith("espeak/en-us+m1")
    assert all((tmp_path / "cache" / p).exists() for p in tokens["cache_path"])
    assert all(len(h) == 64 for h in tokens["sha256"])
    summary = yaml.safe_load((folder / "summary.yaml").read_text())["synthesis"]
    assert summary["tokens"] == 36 and summary["synthesized"] == 36
    assert summary["engines"] == {"espeak": {"speakers": 3, "held_out": 1}}
    assert summary["reproducibility"]["espeak"]["same_settings_identical"] is True
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
