"""Layer 2: synthesis. Audio for every word, from several speakers, with several tokens each.

Every word is synthesized by every speaker, ``tokens_per_speaker`` times. Tokens differ through
the engine's own variability and through a small seeded perturbation of rate and pitch. The
perturbation is the same for both engines: the engine speaks slower or faster, and the clip is
then read at a slightly different sample rate, which shifts its pitch and restores the wanted
duration.

A neural engine now and then stretches a word far beyond its usual length. After synthesis, each
clip's duration is compared with the median duration of its word's tokens. A clip longer than
``duration_check.max_ratio`` times that median is synthesized again with a new perturbation seed,
up to ``duration_check.max_tries`` tries in all, and the shortest try is kept when none passes.

Synthesis happens once. Each clip goes into the cache folder as a 16 kHz mono FLAC file, named
by a hash of its phoneme string, engine, speaker, settings, perturbation, token number, and try
number. Every later step reads from the cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from semantic_world.wordforms.config import Config
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.streams import Streams
from semantic_world.wordforms.synth import audio as audio_tools
from semantic_world.wordforms.synth.speakers import Speaker, draw_speakers

CACHE_VERSION = 2
"""Part of every cache key and cache path. Raise it when the processing of clips changes."""


class Engine(Protocol):
    name: str

    def phonemes(self, word: WordForm) -> str: ...

    def settings_for(self, speaker: Speaker, duration_scale: float) -> dict[str, Any]: ...

    def synthesize(self, phonemes: str, settings: dict[str, Any]) -> tuple[np.ndarray, int]: ...

    def reset(self, seed: int) -> None: ...

    def provenance(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Token:
    """One recording: token ``k`` of a word by a speaker."""

    label: str
    word: str
    speaker: str
    engine: str
    phonemes: str
    settings: dict[str, Any]
    rate_factor: float
    pitch_semitones: float
    duration: float
    cache_path: str
    """The clip's path inside the cache folder."""
    sha256: str
    tries: int = 1
    """How many times the token was synthesized before its duration passed the check (or the
    try limit, when none passed)."""
    peak: float = 0.0
    rms_db: float = 0.0

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "word": self.word,
            "speaker": self.speaker,
            "engine": self.engine,
            "phonemes": self.phonemes,
            "settings": json.dumps(self.settings, sort_keys=True),
            "rate_factor": self.rate_factor,
            "pitch_semitones": self.pitch_semitones,
            "augmentation": "",
            "duration": self.duration,
            "tries": self.tries,
            "peak": self.peak,
            "rms_db": self.rms_db,
            "cache_path": self.cache_path,
            "sha256": self.sha256,
        }


@dataclass
class Synthesis:
    """The speakers and tokens of a run, with what the run did."""

    cache_dir: Path
    sample_rate: int
    speakers: list[Speaker]
    tokens: list[Token]
    synthesized: int = 0
    cached: int = 0
    reproducibility: dict[str, dict[str, Any]] = field(default_factory=dict)
    engines: dict[str, dict[str, Any]] = field(default_factory=dict)
    over_limit: int = 0
    """Tokens whose kept clip is still longer than the duration limit."""
    long_synthesis: dict[str, Any] = field(default_factory=dict)
    """The long-word rule and the words that it flags, when Piper is an engine."""

    def audio(self, token: Token) -> np.ndarray:
        """The token's clip: mono, float32, at ``sample_rate``."""
        clip, _ = audio_tools.read_flac(self.cache_dir / token.cache_path)
        return clip

    def summary(self) -> dict[str, Any]:
        durations = np.array([t.duration for t in self.tokens], dtype=np.float64)
        by_engine: dict[str, dict[str, int]] = {}
        for speaker in self.speakers:
            counts = by_engine.setdefault(speaker.engine, {"speakers": 0, "held_out": 0})
            counts["speakers"] += 1
            counts["held_out"] += int(speaker.held_out)
        return {
            "speakers": len(self.speakers),
            "engines": by_engine,
            "tokens": len(self.tokens),
            "synthesized": self.synthesized,
            "read_from_cache": self.cached,
            "duration_seconds": {
                "total": round(float(durations.sum()), 3),
                "mean": round(float(durations.mean()), 3),
                "min": round(float(durations.min()), 3),
                "max": round(float(durations.max()), 3),
            }
            if len(durations)
            else {},
            "duration_check": {
                "retried": sum(t.tries > 1 for t in self.tokens),
                "still_over_limit": self.over_limit,
                "tries": {
                    str(k): sum(t.tries == k for t in self.tokens)
                    for k in sorted({t.tries for t in self.tokens})
                },
            },
            "long_synthesis": self.long_synthesis,
            "levels": {
                name: _level_summary([t for t in self.tokens if t.engine == name])
                for name in by_engine
            },
            "reproducibility": self.reproducibility,
            "models": self.engines,
        }


def _level_summary(tokens: list[Token]) -> dict[str, Any]:
    """The distribution of peaks and RMS levels of an engine's clips."""
    if not tokens:
        return {}
    summary: dict[str, Any] = {"clips": len(tokens)}
    for name, values in (
        ("peak", [t.peak for t in tokens]),
        ("rms_db", [t.rms_db for t in tokens]),
    ):
        array = np.array(values, dtype=np.float64)
        quantiles = np.percentile(array, [0, 1, 50, 99, 100])
        summary[name] = {
            key: round(float(q), 4)
            for key, q in zip(("min", "p01", "median", "p99", "max"), quantiles, strict=True)
        }
    return summary


def token_perturbation(
    config: Config, streams: Streams, word: WordForm, speaker: Speaker, k: int, attempt: int = 1
) -> tuple[float, float]:
    """The rate factor and the pitch shift in semitones of one token. The draw depends only on
    the master seed, the word's phonemes, the speaker's voice, the token number, and the try
    number, so adding words or speakers never changes the other tokens."""
    perturbation = config.synthesis.token_perturbation
    key = f"{word.arpabet}|{speaker.identity}|{k}"
    if attempt > 1:
        key += f"|try{attempt}"
    rng = streams.substream("synthesis", key)
    rate = 1.0 + float(rng.uniform(-perturbation.rate, perturbation.rate))
    pitch = float(rng.uniform(-perturbation.pitch_semitones, perturbation.pitch_semitones))
    return round(rate, 6), round(pitch, 6)


def cache_key(
    config: Config,
    engine: str,
    phonemes: str,
    settings: dict,
    pitch: float,
    k: int,
    attempt: int = 1,
) -> str:
    """The name of a clip in the cache: a hash of everything that shapes the clip."""
    trim = config.synthesis.trim
    level = config.synthesis.level
    description = {
        "version": CACHE_VERSION,
        "engine": engine,
        "phonemes": phonemes,
        "settings": settings,
        "pitch_semitones": pitch,
        "token": k,
        "try": attempt,
        "sample_rate": config.synthesis.sample_rate,
        "trim": [trim.threshold_db, trim.margin_ms],
        "level": [level.rms_db, level.max_peak],
    }
    text = json.dumps(description, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finish_clip(config: Config, raw: np.ndarray, native_rate: int, pitch: float) -> np.ndarray:
    """Turn an engine's raw clip into the stored clip: shift the pitch by reading the clip at a
    scaled rate, resample to the run's sample rate, trim the silence, and set the level."""
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    factor = 2.0 ** (pitch / 12.0)
    trim = config.synthesis.trim
    level = config.synthesis.level
    clip = audio_tools.resample(raw, native_rate * factor, config.synthesis.sample_rate)
    clip = audio_tools.trim(clip, config.synthesis.sample_rate, trim.threshold_db, trim.margin_ms)
    return audio_tools.set_level(clip, level.rms_db, level.max_peak)


def flag_long_synthesis(config: Config, words: list[WordForm], result: Synthesis) -> None:
    """Set each word's ``long_synthesis`` flag: whether the median duration of the word's Piper
    tokens is more than ``synthesis.long_synthesis_ratio`` times the median for words with the
    same number of syllables. Piper stretches some words for most speakers, and the flag lets
    later analyses leave those words out. The median for a number of syllables is taken over the
    content words, and a closed-class form is compared with it. Without Piper tokens, or with a
    null ratio, the flags stay unset."""
    ratio = config.synthesis.long_synthesis_ratio
    durations: dict[str, list[float]] = {}
    for token in result.tokens:
        if token.engine == "piper":
            durations.setdefault(token.word, []).append(token.duration)
    if ratio is None or not durations:
        return
    medians = {word: float(np.median(values)) for word, values in durations.items()}
    # The reference is the content words, so that closed-class forms never change their flags.
    by_syllables: dict[int, list[float]] = {}
    for word in words:
        if word.label in medians and word.kind == "content":
            by_syllables.setdefault(word.syllable_count, []).append(medians[word.label])
    reference = {count: float(np.median(values)) for count, values in by_syllables.items()}
    flagged = {}
    for word in words:
        if word.label not in medians or word.syllable_count not in reference:
            continue
        relative = medians[word.label] / reference[word.syllable_count]
        word.long_synthesis = bool(relative > ratio)
        if word.long_synthesis:
            flagged[word.label] = {
                "median_seconds": round(medians[word.label], 3),
                "ratio": round(relative, 3),
            }
    result.long_synthesis = {
        "ratio": ratio,
        "median_seconds_by_syllables": {k: round(reference[k], 3) for k in sorted(reference)},
        "flagged": flagged,
    }


def make_engines(config: Config, streams: Streams) -> dict[str, Engine]:
    """The configured engines. Raises ``RuntimeError`` naming the missing tool when an engine
    cannot run."""
    engines: dict[str, Engine] = {}
    if config.synthesis.piper is not None:
        from semantic_world.wordforms.synth.piper import PiperEngine

        engines["piper"] = PiperEngine(config.synthesis.piper, streams.seed("synthesis"))
    if config.synthesis.espeak is not None:
        from semantic_world.wordforms.synth.espeak import EspeakEngine

        engines["espeak"] = EspeakEngine(config.synthesis.espeak)
    return engines


def speakers_for(config: Config, streams: Streams) -> list[Speaker]:
    """The speakers of a configuration. Reads the Piper voice's speaker count when Piper is on."""
    count = None
    if config.synthesis.piper is not None:
        from semantic_world.wordforms.synth.piper import voice_speaker_count

        count = voice_speaker_count(config.synthesis.piper)
    return draw_speakers(config, streams, count)


REPRODUCIBILITY_WORDS = 3
REPRODUCIBILITY_SPEAKERS = 8
REPRODUCIBILITY_REPEATS = 4


def check_reproducibility(
    engine: Engine, words: list[WordForm], speakers: list[Speaker], seed: int
) -> dict[str, Any]:
    """Whether the engine gives identical audio for the same settings.

    The check synthesizes a few words by a few speakers several times in one session
    (``same_settings_identical``), and once each in two fresh sessions with the same seed
    (``fresh_session_identical``). ``varying_speakers`` lists the voices whose audio varied. An
    engine can vary only now and then, so a pass here is evidence, not proof; the SHA-256 hashes
    of the cached clips are what a run records.
    """
    clips = [
        (engine.phonemes(word), engine.settings_for(speaker, 1.0), speaker.identity)
        for word in words[:REPRODUCIBILITY_WORDS]
        for speaker in speakers[:REPRODUCIBILITY_SPEAKERS]
    ]
    varying: set[str] = set()
    engine.reset(seed)
    repeatable = True
    for phonemes, settings, identity in clips:
        first, _ = engine.synthesize(phonemes, settings)
        for _ in range(REPRODUCIBILITY_REPEATS - 1):
            again, _ = engine.synthesize(phonemes, settings)
            if not np.array_equal(first, again):
                repeatable = False
                varying.add(identity)
    sessions = []
    for _ in range(2):
        engine.reset(seed)
        sessions.append(
            [engine.synthesize(phonemes, settings)[0] for phonemes, settings, _ in clips]
        )
    fresh = True
    for (_, _, identity), a, b in zip(clips, *sessions, strict=True):
        if not np.array_equal(a, b):
            fresh = False
            varying.add(identity)
    return {
        "same_settings_identical": repeatable,
        "fresh_session_identical": fresh,
        "clips_tested": len(clips),
        "varying_speakers": sorted(varying),
    }


def synthesize_lexicon(
    config: Config,
    streams: Streams,
    words: list[WordForm],
    *,
    engines: dict[str, Engine] | None = None,
    speakers: list[Speaker] | None = None,
    check: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> Synthesis:
    """Synthesize every word by every speaker, reading clips from the cache when they exist.
    The tokens come in the order of ``words``, which holds the content words first."""
    settings = config.synthesis
    cache_dir = Path(settings.cache_dir)
    engines = engines if engines is not None else make_engines(config, streams)
    speakers = speakers if speakers is not None else speakers_for(config, streams)
    result = Synthesis(cache_dir, settings.sample_rate, speakers, [])
    seed = streams.seed("synthesis")
    for name, engine in engines.items():
        result.engines[name] = engine.provenance()
        own = [s for s in speakers if s.engine == name]
        if check and words and own:
            result.reproducibility[name] = check_reproducibility(engine, words, own, seed)
        engine.reset(seed)

    def clip_for(word: WordForm, speaker: Speaker, k: int, attempt: int) -> Token:
        """One try at a token: read from the cache, or synthesized and written to it."""
        engine = engines[speaker.engine]
        phonemes = engine.phonemes(word)
        rate, pitch = token_perturbation(config, streams, word, speaker, k, attempt)
        duration_scale = 2.0 ** (pitch / 12.0) / rate
        clip_settings = engine.settings_for(speaker, duration_scale)
        key = cache_key(config, engine.name, phonemes, clip_settings, pitch, k, attempt)
        relative = Path(f"v{CACHE_VERSION}") / engine.name / key[:2] / f"{key}.flac"
        path = cache_dir / relative
        label = f"{word.label}.{speaker.label}.{k}"
        if path.exists():
            result.cached += 1
        else:
            raw, native_rate = engine.synthesize(phonemes, clip_settings)
            try:
                clip = finish_clip(config, raw, native_rate, pitch)
            except audio_tools.AudioError as error:
                raise audio_tools.AudioError(f"{label} ({phonemes}): {error}") from error
            audio_tools.write_flac(path, clip, settings.sample_rate)
            result.synthesized += 1
        clip, _ = audio_tools.read_flac(path)
        return Token(
            label=label,
            word=word.label,
            speaker=speaker.label,
            engine=engine.name,
            phonemes=phonemes,
            settings=clip_settings,
            rate_factor=rate,
            pitch_semitones=pitch,
            duration=round(len(clip) / settings.sample_rate, 6),
            cache_path=relative.as_posix(),
            sha256=audio_tools.sha256_file(path),
            tries=attempt,
            peak=round(audio_tools.peak(clip), 6),
            rms_db=round(audio_tools.rms_db(clip), 4),
        )

    # The content words come first, through their duration check, and the closed-class forms
    # after them. An engine whose audio depends on what it synthesized before (Piper) therefore
    # gives the content words the same audio with and without closed-class forms.
    groups = [
        [word for word in words if word.kind == "content"],
        [word for word in words if word.kind != "content"],
    ]
    total = len(words) * len(speakers) * settings.tokens_per_speaker
    check_settings = settings.duration_check
    for group in groups:
        # First, one try at every token.
        plan = [
            (word, speaker, k)
            for word in group
            for speaker in speakers
            for k in range(1, settings.tokens_per_speaker + 1)
        ]
        first = len(result.tokens)
        for word, speaker, k in plan:
            result.tokens.append(clip_for(word, speaker, k, 1))
            if progress is not None:
                progress(len(result.tokens), total)
        if check_settings is None:
            continue
        # Then the duration check: a clip much longer than its word's median is tried again.
        durations: dict[str, list[float]] = {}
        for token in result.tokens[first:]:
            durations.setdefault(token.word, []).append(token.duration)
        limits = {
            word: check_settings.max_ratio * float(np.median(values))
            for word, values in durations.items()
        }
        for i, (word, speaker, k) in enumerate(plan, start=first):
            best = result.tokens[i]
            limit = limits[word.label]
            attempt = 1
            latest = best
            while latest.duration > limit and attempt < check_settings.max_tries:
                attempt += 1
                latest = clip_for(word, speaker, k, attempt)
                if latest.duration < best.duration:
                    best = latest
            kept = latest if latest.duration <= limit else best
            result.tokens[i] = dataclasses.replace(kept, tries=attempt)
            result.over_limit += int(kept.duration > limit)
    flag_long_synthesis(config, words, result)
    return result
