"""Layer 2: synthesis. Audio for every word, from several speakers, with several tokens each.

Every word is synthesized by every speaker, ``tokens_per_speaker`` times. Tokens differ through
the engine's own variability and through a small seeded perturbation of rate and pitch. The
perturbation is the same for both engines: the engine speaks slower or faster, and the clip is
then read at a slightly different sample rate, which shifts its pitch and restores the wanted
duration.

Synthesis happens once. Each clip goes into the cache folder as a 16 kHz mono FLAC file, named
by a hash of its phoneme string, engine, speaker, settings, perturbation, and token number.
Every later step reads from the cache.
"""

from __future__ import annotations

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

CACHE_VERSION = 1
"""Part of every cache key. Raise it when the processing of clips changes."""


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
            "reproducibility": self.reproducibility,
            "models": self.engines,
        }


def token_perturbation(
    config: Config, streams: Streams, word: WordForm, speaker: Speaker, k: int
) -> tuple[float, float]:
    """The rate factor and the pitch shift in semitones of one token. The draw depends only on
    the master seed, the word's phonemes, the speaker's voice, and the token number, so adding
    words or speakers never changes the other tokens."""
    perturbation = config.synthesis.token_perturbation
    rng = streams.substream("synthesis", f"{word.arpabet}|{speaker.identity}|{k}")
    rate = 1.0 + float(rng.uniform(-perturbation.rate, perturbation.rate))
    pitch = float(rng.uniform(-perturbation.pitch_semitones, perturbation.pitch_semitones))
    return round(rate, 6), round(pitch, 6)


def cache_key(
    config: Config, engine: str, phonemes: str, settings: dict, pitch: float, k: int
) -> str:
    """The name of a clip in the cache: a hash of everything that shapes the clip."""
    trim = config.synthesis.trim
    description = {
        "version": CACHE_VERSION,
        "engine": engine,
        "phonemes": phonemes,
        "settings": settings,
        "pitch_semitones": pitch,
        "token": k,
        "sample_rate": config.synthesis.sample_rate,
        "trim": [trim.threshold_db, trim.margin_ms],
        "peak": audio_tools.PEAK,
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
    clip = audio_tools.resample(raw, native_rate * factor, config.synthesis.sample_rate)
    clip = audio_tools.trim(clip, config.synthesis.sample_rate, trim.threshold_db, trim.margin_ms)
    return audio_tools.normalize_peak(clip)


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
    """Synthesize every word by every speaker, reading clips from the cache when they exist."""
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
    total = len(words) * len(speakers) * settings.tokens_per_speaker
    done = 0
    for word in words:
        for speaker in speakers:
            engine = engines[speaker.engine]
            phonemes = engine.phonemes(word)
            for k in range(1, settings.tokens_per_speaker + 1):
                rate, pitch = token_perturbation(config, streams, word, speaker, k)
                duration_scale = 2.0 ** (pitch / 12.0) / rate
                clip_settings = engine.settings_for(speaker, duration_scale)
                key = cache_key(config, engine.name, phonemes, clip_settings, pitch, k)
                relative = Path(engine.name) / key[:2] / f"{key}.flac"
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
                result.tokens.append(
                    Token(
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
                    )
                )
                done += 1
                if progress is not None:
                    progress(done, total)
    return result
