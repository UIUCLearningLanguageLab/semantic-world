"""The Whisper check of synthesis quality."""

from __future__ import annotations

import pytest
from test_wordforms_synth import ToneEngine, espeak_only
from wordforms_support import (
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_piper,
    needs_whisper,
    needs_wordfreq,
)

from semantic_world.wordforms.config import parse_config
from semantic_world.wordforms.transcribe import (
    THRESHOLDS,
    common_words,
    normalize_text,
    real_word_forms,
    whisper_check,
)

pytestmark = [needs_cmudict, needs_wordfreq]


def test_normalize_text():
    assert normalize_text(" Hello. ") == "hello"
    assert normalize_text("Rake down.") == "rakedown"
    assert normalize_text("KA-GOS") == "kagos"
    assert normalize_text("don't!") == "don't"
    assert normalize_text("") == ""


def test_common_words_are_a_seeded_sample_of_frequent_words(common_english):
    from wordfreq import zipf_frequency

    words = common_words(common_english, 200, 1)
    assert len(words) == len(set(words)) == 200
    assert words == common_words(common_english, 200, 1)
    assert words != common_words(common_english, 200, 2)
    for word in words:
        assert word.isalpha() and zipf_frequency(word, "en") >= 4.0
        assert len(common_english.words[word]) == 1
        assert len(common_english.words[word][0]) >= 3


def test_real_word_forms(common_english):
    forms = real_word_forms(["hello", "computer"], common_english)
    assert [f.label for f in forms] == ["WORD.1", "WORD.2"]
    assert forms[0].real_word and forms[0].english_word == forms[0].spelling == "hello"
    assert forms[0].ipa == "həlˈoʊ" and forms[0].espeak == "h@l'oU"
    assert forms[1].arpabet == "K AH0 M P Y UW1 T ER0"


class FakeTranscriber:
    """Hears every word of the first speaker exactly, a homophone or the word for the second
    speaker, and nothing right for the third."""

    name = "fake"
    revision = "r1"
    device = "cpu"

    def __init__(self, words):
        self.words = words
        self.clips = 0

    def transcribe(self, clips, sample_rate):
        assert sample_rate == 16000 and len(clips) == len(self.words) * 3
        self.clips = len(clips)
        texts = []
        for word in self.words:
            homophone = {"their": "there", "there": "their"}.get(word, word)
            texts += [f" {word.capitalize()}.", homophone, "thank you"]
        return texts


@needs_audio
def test_whisper_check_report_with_stand_ins(tmp_path, common_english):
    config = espeak_only(tmp_path)
    words = common_words(common_english, 12, config.seed)
    transcriber = FakeTranscriber(words)
    report = whisper_check(
        config, words=12, speakers=3, transcriber=transcriber, engines={"espeak": ToneEngine()}
    )
    assert transcriber.clips == 36
    assert report["model"] == "fake" and report["revision"] == "r1"
    assert report["words"] == 12 and report["speakers_per_engine"] == 3
    assert list(report["engines"]) == ["espeak"]
    espeak = report["engines"]["espeak"]
    assert espeak["clips"] == 36
    assert espeak["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert espeak["exact_accuracy"] <= espeak["accuracy"]
    assert espeak["threshold"] == THRESHOLDS["espeak"] == 0.6
    assert espeak["meets_threshold"] is True
    assert sorted(espeak["accuracy_by_speaker"].values()) == [0.0, 1.0, 1.0]
    assert len(espeak["errors"]) == 12
    assert espeak["errors"][0]["heard"] == "thank you"


@needs_audio
def test_a_missed_threshold_is_reported(tmp_path, common_english):
    class Deaf(FakeTranscriber):
        def transcribe(self, clips, sample_rate):
            return ["you"] * len(clips)

    report = whisper_check(
        espeak_only(tmp_path),
        words=5,
        speakers=1,
        transcriber=Deaf([]),
        engines={"espeak": ToneEngine()},
    )
    assert report["engines"]["espeak"]["accuracy"] == 0.0
    assert report["engines"]["espeak"]["meets_threshold"] is False


@needs_audio
def test_homophones_count_as_correct(tmp_path, common_english):
    # "their" and "there" share a pronunciation in the dictionary
    assert set(common_english.pronunciations[("DH", "EH", "R")]) >= {"their", "there"}


@needs_audio
@needs_espeak
@needs_piper
@needs_whisper
def test_whisper_hears_real_words_from_both_engines(tmp_path):
    """A small version of the stage's check. The full check (200 words, 3 speakers for each
    engine) runs from the command line, and the stage report gives its numbers."""
    config = parse_config(
        {
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "engines": {"piper": {"voice_dir": str(VOICE_DIR)}, "espeak": {}},
            },
            "device": "cpu",
        },
        "whisper_test",
    )
    report = whisper_check(config, words=24, speakers=1, local_only=True)
    assert report["model"] == "openai/whisper-small.en" and report["revision"]
    assert set(report["engines"]) == {"piper", "espeak"}
    for engine, floor in (("piper", 0.5), ("espeak", 0.3)):
        result = report["engines"][engine]
        assert result["clips"] == 24
        assert floor <= result["accuracy"] <= 1.0
        assert result["exact_accuracy"] <= result["accuracy"]
        assert result["threshold"] == THRESHOLDS[engine]
