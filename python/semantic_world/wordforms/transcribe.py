"""The Whisper check of synthesis quality (``docs/specs/WORDFORM_PIPELINE.md``, stage 2).

Common real English words are synthesized from their dictionary phonemes, by each engine, and a
Whisper model transcribes the clips. If the mapping tables and the engines are right, the model
hears the words. The check reports the accuracy for each engine against provisional thresholds.
A missed threshold is reported, not tuned away.

A transcription is *exact* when it is the word itself, and *correct* when it is the word or a
homophone (a dictionary word with the same phonemes), since audio cannot tell homophones apart.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np

from semantic_world.wordforms.config import Config
from semantic_world.wordforms.english import English, load_english, strip_stress, zipf_frequencies
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.phonemes import load_tables
from semantic_world.wordforms.streams import Streams

DEFAULT_MODEL = "openai/whisper-small.en"
THRESHOLDS = {"piper": 0.80, "espeak": 0.60}
"""Provisional accuracy thresholds for each engine."""
CHECK_MIN_ZIPF = 4.0
CHECK_MIN_PHONEMES = 3


def common_words(english: English, count: int, seed: int) -> list[str]:
    """A seeded sample of common real words: plain words with one pronunciation, a Zipf
    frequency of at least 4.0, and at least 3 phonemes."""
    candidates = sorted(w for w in english.words if w.isalpha() and len(english.words[w]) == 1)
    zipf = zipf_frequencies(candidates)
    candidates = [
        w
        for w in candidates
        if zipf[w] >= CHECK_MIN_ZIPF and len(english.words[w][0]) >= CHECK_MIN_PHONEMES
    ]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return [candidates[i] for i in sorted(chosen)]


def real_word_forms(words: list[str], english: English) -> list[WordForm]:
    """Word forms for real words, with their IPA and espeak-ng strings."""
    ipa_table, espeak_table = load_tables()
    forms = []
    for i, word in enumerate(words):
        syllables = english.syllables[english.words[word][0]]
        forms.append(
            WordForm(
                label=f"W.{i + 1}",
                syllables=syllables,
                real_word=True,
                english_word=word,
                ipa=ipa_table.render(syllables),
                espeak=espeak_table.render(syllables),
                spelling=word,
            )
        )
    return forms


def normalize_text(text: str) -> str:
    """A transcription in lower case without punctuation or spaces."""
    return re.sub(r"[^a-z']", "", text.lower()).strip("'")


class WhisperTranscriber:
    """A Whisper model from Hugging Face ``transformers``, in evaluation mode."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", local_only: bool = False):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()

        from semantic_world.wordforms.device import resolve_device

        self.torch = torch
        self.name = model
        self.device = resolve_device(device)
        self.processor = WhisperProcessor.from_pretrained(model, local_files_only=local_only)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, local_files_only=local_only
        )
        self.model.eval().to(self.device)
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def transcribe(self, clips: list[np.ndarray], sample_rate: int, batch: int = 32) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(clips), batch):
            features = self.processor(
                clips[start : start + batch], sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)
            with self.torch.no_grad():
                ids = self.model.generate(features, max_new_tokens=12, do_sample=False)
            texts.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return texts


def model_is_cached(model: str = DEFAULT_MODEL) -> bool:
    """Whether the model is already in the Hugging Face cache, so that no download is needed."""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


def whisper_check(
    config: Config,
    *,
    words: int = 200,
    speakers: int = 3,
    model: str = DEFAULT_MODEL,
    local_only: bool = False,
    transcriber: Any = None,
    engines: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize ``words`` common real words with ``speakers`` speakers of each engine, one
    token each, transcribe the clips, and report the accuracy for each engine."""
    from semantic_world.wordforms.synth import synthesize_lexicon

    synthesis = config.synthesis
    piper = synthesis.piper and dataclasses.replace(synthesis.piper, speakers=speakers)
    espeak = synthesis.espeak and dataclasses.replace(synthesis.espeak, speakers=speakers)
    check_config = dataclasses.replace(
        config,
        synthesis=dataclasses.replace(synthesis, piper=piper, espeak=espeak, tokens_per_speaker=1),
    )
    streams = Streams(config.seed)
    english = load_english(config.wordforms.english_min_zipf)
    chosen = common_words(english, words, config.seed)
    forms = real_word_forms(chosen, english)
    result = synthesize_lexicon(check_config, streams, forms, engines=engines, check=False)
    transcriber = transcriber or WhisperTranscriber(model, config.device, local_only)
    clips = [result.audio(token) for token in result.tokens]
    texts = transcriber.transcribe(clips, result.sample_rate)

    word_of = {form.label: form for form in forms}
    report: dict[str, Any] = {
        "model": getattr(transcriber, "name", model),
        "revision": getattr(transcriber, "revision", None),
        "device": getattr(transcriber, "device", None),
        "words": len(forms),
        "speakers_per_engine": speakers,
        "word_selection": (
            f"a seeded sample of plain words with one pronunciation, Zipf frequency >= "
            f"{CHECK_MIN_ZIPF}, and >= {CHECK_MIN_PHONEMES} phonemes"
        ),
        "engines": {},
    }
    speaker_of = {s.label: s for s in result.speakers}
    for engine in result.engines:
        rows = [(t, x) for t, x in zip(result.tokens, texts, strict=True) if t.engine == engine]
        exact = correct = 0
        by_speaker: dict[str, list[int]] = {}
        errors = []
        for token, text in rows:
            form = word_of[token.word]
            heard = normalize_text(text)
            is_exact = heard == form.english_word
            is_correct = is_exact or heard in english.pronunciations.get(
                strip_stress(form.phones), ()
            )
            exact += is_exact
            correct += is_correct
            tally = by_speaker.setdefault(speaker_of[token.speaker].identity, [0, 0])
            tally[0] += is_correct
            tally[1] += 1
            if not is_correct and len(errors) < 25:
                errors.append(
                    {"word": form.english_word, "heard": text.strip(), "speaker": token.speaker}
                )
        threshold = THRESHOLDS.get(engine)
        accuracy = correct / len(rows) if rows else 0.0
        report["engines"][engine] = {
            "clips": len(rows),
            "accuracy": round(accuracy, 4),
            "exact_accuracy": round(exact / len(rows), 4) if rows else 0.0,
            "threshold": threshold,
            "meets_threshold": bool(threshold is None or accuracy >= threshold),
            "accuracy_by_speaker": {k: round(c / n, 4) for k, (c, n) in by_speaker.items()},
            "errors": errors,
        }
    return report
