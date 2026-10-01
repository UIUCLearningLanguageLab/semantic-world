"""Augmentation: seeded transformations of cached audio, each result a new token.

A recipe names the transformations it applies, in this order: Praat manipulation (pitch median,
pitch range, formant shift, duration, as in Praat's "Change gender"), speed and pitch
perturbation, reverberation, and additive noise. Each recipe is applied to a seeded share of the
eligible tokens. An augmented token keeps its source's word and speaker, gets the label
``<source label>.A.<recipe number>``, and records its recipe with every value that was drawn
(``augmentation`` in ``tokens.csv``). Its clip goes in the audio cache under a hash of the
source clip and the drawn values, so a second run reads it back.

Every draw comes from a substream of ``wordforms:augment`` named by the source token and the
recipe, so augmenting one token never changes another. Every transformation's target is checked:
the value it aimed at and the value measured right after it (median pitch, pitch range, formant
ratio, duration, reverberation time, signal-to-noise ratio) go in the token's ``achieved``
record, and the summary counts the tokens that miss a target by more than 5% and by more than
10%. Noise is generated from the same
substream. Reverberation uses ``pyroomacoustics`` (MIT) and manipulation uses Praat
(:mod:`semantic_world.wordforms.praat`); both are imported only where used.

Noise kinds: ``white``; ``pink`` (power falling as 1/f); ``speech``, white noise shaped by the
long-term average spectrum of the run's own clips; and ``babble``, several other tokens of the
run added together. The signal-to-noise ratio is exact by construction: the noise is scaled to
the target ratio of powers over the whole clip, and the mixture is then leveled like every other
clip, which leaves the ratio unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from semantic_world.wordforms.config import Config, RecipeConfig
from semantic_world.wordforms.streams import Streams
from semantic_world.wordforms.synth import CACHE_VERSION, Synthesis, Token
from semantic_world.wordforms.synth import audio as audio_tools

AUGMENT_VERSION = 2
"""Part of every augmented clip's cache key. Raise it when the transformations or the measures
of their achieved values change. Version 2 measures the formant shift and the pitch shift frame
by frame."""
SPECTRUM_BINS = 513
"""The resolution of the long-term average spectrum for speech-shaped noise."""
SPECTRUM_TOKENS = 200
"""How many clips (the first of the run) the long-term average spectrum is taken from."""
REVERB_MAX_ORDER = 200
"""The largest image-source order of the room simulation; the order that the reverberation
time needs is usually far below it."""


@dataclass
class Augmentation:
    """What a run's augmentation did."""

    tokens: list[Token]
    """The augmented tokens, in the order of the source tokens and then of the recipes."""
    computed: int = 0
    cached: int = 0
    skipped: list[dict[str, str]] | None = None
    """Tokens a recipe could not augment, with the reason (for example, no voiced frame)."""

    def summary(self) -> dict[str, Any]:
        recipes: dict[str, int] = {}
        for token in self.tokens:
            name = json.loads(token.augmentation)["recipe"]
            recipes[name] = recipes.get(name, 0) + 1
        return {
            "tokens": len(self.tokens),
            "computed": self.computed,
            "read_from_cache": self.cached,
            "by_recipe": recipes,
            "skipped": self.skipped or [],
            "achieved": achieved_summary(self.tokens),
        }


MISS_LEVELS = (0.05, 0.10)
"""The relative errors that the achieved-value summary counts misses against."""


def achieved_summary(tokens: list[Token]) -> dict[str, dict[str, int]]:
    """For each measured quantity, over the augmented tokens: how many tokens it was measured on,
    how many could not be measured, and how many miss their target by more than 5% and 10%."""
    summary: dict[str, dict[str, int]] = {}
    for token in tokens:
        if not token.achieved:
            continue
        for name, entry in json.loads(token.achieved).items():
            counts = summary.setdefault(
                name, {"tokens": 0, "unmeasured": 0, "over_5_percent": 0, "over_10_percent": 0}
            )
            counts["tokens"] += 1
            if entry["miss"] is None:
                counts["unmeasured"] += 1
                continue
            if entry["miss"] > MISS_LEVELS[0]:
                counts["over_5_percent"] += 1
            if entry["miss"] > MISS_LEVELS[1]:
                counts["over_10_percent"] += 1
    return summary


def _achieved(target: float, measured: float) -> dict[str, float | None]:
    # a relative miss needs a target that is not (nearly) zero, such as a flat pitch range
    ok = bool(np.isfinite(measured)) and abs(target) > 1e-6
    return {
        "target": round(float(target), 6),
        "measured": round(float(measured), 6) if np.isfinite(measured) else None,
        "miss": round(abs(float(measured) / target - 1.0), 6) if ok else None,
    }


# ---------------------------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------------------------


def white_noise(rng: np.random.Generator, samples: int) -> np.ndarray:
    return rng.standard_normal(samples)


def shaped_noise(rng: np.random.Generator, samples: int, magnitude: np.ndarray) -> np.ndarray:
    """White noise whose spectrum is shaped by ``magnitude``, given on ``SPECTRUM_BINS`` bins
    from 0 Hz to half the sample rate."""
    white = white_noise(rng, samples)
    spectrum = np.fft.rfft(white)
    bins = np.linspace(0.0, 1.0, len(spectrum))
    grid = np.linspace(0.0, 1.0, len(magnitude))
    return np.fft.irfft(spectrum * np.interp(bins, grid, magnitude), n=samples)


def pink_magnitude() -> np.ndarray:
    """The magnitude of pink noise: power falling as 1/f, so magnitude as 1/sqrt(f)."""
    f = np.arange(SPECTRUM_BINS, dtype=np.float64)
    f[0] = 1.0
    return 1.0 / np.sqrt(f)


def long_term_spectrum(clips: list[np.ndarray]) -> np.ndarray:
    """The long-term average magnitude spectrum of some clips, on ``SPECTRUM_BINS`` bins: the
    root of the mean power spectrum of 1024-sample Hann-windowed frames."""
    size = 2 * (SPECTRUM_BINS - 1)
    window = np.hanning(size + 1)[:-1]
    power = np.zeros(SPECTRUM_BINS, dtype=np.float64)
    frames = 0
    for clip in clips:
        data = np.asarray(clip, dtype=np.float64)
        for start in range(0, max(1, len(data) - size + 1), size // 2):
            frame = data[start : start + size]
            if len(frame) < size:
                frame = np.pad(frame, (0, size - len(frame)))
            power += np.abs(np.fft.rfft(frame * window)) ** 2
            frames += 1
    if frames == 0:
        return np.ones(SPECTRUM_BINS)
    return np.sqrt(power / frames)


def babble(rng: np.random.Generator, samples: int, voices: list[np.ndarray]) -> np.ndarray:
    """Other clips added together, each at a random offset (tiled when shorter than the clip)
    and at the same level."""
    total = np.zeros(samples, dtype=np.float64)
    for voice in voices:
        data = np.asarray(voice, dtype=np.float64)
        repeats = int(np.ceil((samples + len(data)) / len(data)))
        tiled = np.tile(data, repeats)
        offset = int(rng.integers(len(data)))
        piece = tiled[offset : offset + samples]
        power = np.mean(piece**2)
        if power > 0:
            total += piece / np.sqrt(power)
    return total


def add_noise(clip: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """The clip plus the noise, scaled so that the ratio of the clip's power to the noise's
    power is ``snr_db`` decibels."""
    signal_power = float(np.mean(np.square(clip, dtype=np.float64)))
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    if noise_power <= 0:
        raise audio_tools.AudioError("the noise is silent")
    gain = np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return np.asarray(clip, dtype=np.float64) + gain * noise


def measured_snr_db(mixture: np.ndarray, clip: np.ndarray) -> float:
    """The signal-to-noise ratio of a mixture, given the clean clip it contains (both at the
    same level), from the residual."""
    residual = np.asarray(mixture, dtype=np.float64) - np.asarray(clip, dtype=np.float64)
    return 10.0 * float(np.log10(np.mean(np.square(clip, dtype=np.float64)) / np.mean(residual**2)))


# ---------------------------------------------------------------------------------------------
# Reverberation
# ---------------------------------------------------------------------------------------------


SOUND_SPEED = 343.0
REVERB_CALIBRATION = (0.1975, 0.1988, -0.0330, -0.0873)
"""The fit of the simulated reverberation time against Eyring's formula, on 300 random rooms
with sides of 3 to 6 m: the measured T30 is ``g`` times ``0.161 V / (S L)``, where ``L`` is
``-ln(1 - absorption)``, and ``log g = c0 + c1 log(longest side / shortest side) + c2
log(V / S) + c3 log L``. Solving for the absorption puts the measured time within 10% of the
target for 99% of such rooms (it was 25 to 40% too long with Sabine's formula)."""
MAX_ABSORPTION = 0.99


def calibrated_absorption(dimensions: tuple[float, float, float], rt60: float) -> float:
    """The wall absorption that gives a shoebox room the reverberation time ``rt60``, from the
    calibration of the image-source simulation (:data:`REVERB_CALIBRATION`)."""
    c0, c1, c2, c3 = REVERB_CALIBRATION
    x, y, z = dimensions
    volume = x * y * z
    surface = 2.0 * (x * y + y * z + x * z)
    aspect = max(dimensions) / min(dimensions)
    ratio = volume / surface
    log_l = (
        c0 + c1 * math.log(aspect) + c2 * math.log(ratio) + math.log(0.161 * ratio) - math.log(rt60)
    ) / (1.0 - c3)
    absorption = 1.0 - math.exp(-math.exp(log_l))
    if not 0.0 < absorption <= MAX_ABSORPTION:
        raise audio_tools.AudioError(
            f"no room of {dimensions} m has a reverberation time of {rt60} s: the walls would "
            f"need an absorption of {absorption:.3f}"
        )
    return absorption


def room_impulse_response(
    rate: int, dimensions: tuple[float, float, float], rt60: float, source, microphone
) -> tuple[np.ndarray, float]:
    """The impulse response of a shoebox room with the given reverberation time, from the
    image-source method of ``pyroomacoustics``, with the wall absorption from the calibration;
    and that absorption."""
    try:
        import pyroomacoustics as pra
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reverberation needs the pyroomacoustics package; install the 'speech' extra"
        ) from error

    absorption = calibrated_absorption(dimensions, rt60)
    order = math.ceil(SOUND_SPEED * 1.5 * rt60 / min(dimensions)) + 2
    room = pra.ShoeBox(
        dimensions,
        fs=rate,
        materials=pra.Material(absorption),
        max_order=min(order, REVERB_MAX_ORDER),
    )
    room.add_source(list(source))
    room.add_microphone(list(microphone))
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=np.float64), float(absorption)


def reverberation_time(rir: np.ndarray, rate: int) -> float:
    """The reverberation time of an impulse response: the time for its Schroeder decay curve to
    fall from -5 dB to -35 dB, doubled (T30)."""
    energy = np.cumsum(rir[::-1] ** 2)[::-1]
    if energy[0] <= 0:
        return float("nan")
    decay = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-30))
    start = int(np.argmax(decay <= -5.0))
    end = int(np.argmax(decay <= -35.0))
    if decay[end] > -35.0 or end <= start:
        return float("nan")
    return 2.0 * (end - start) / rate


def reverberate(clip: np.ndarray, rir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    return fftconvolve(np.asarray(clip, dtype=np.float64), rir)


def _place(rng: np.random.Generator, dimensions, margin: float = 0.5) -> list[float]:
    return [float(rng.uniform(margin, d - margin)) for d in dimensions]


# ---------------------------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------------------------


def _draw(rng: np.random.Generator, low_high: tuple[float, float]) -> float:
    low, high = low_high
    return round(low if low == high else float(rng.uniform(low, high)), 6)


def draw_settings(rng: np.random.Generator, recipe: RecipeConfig, voices: int) -> dict[str, Any]:
    """Every value a recipe needs for one token, drawn in a fixed order from ``rng``: the
    manipulation, the speed and pitch, the room, and the noise."""
    drawn: dict[str, Any] = {"recipe": recipe.name}
    if recipe.manipulation is not None:
        m = recipe.manipulation
        drawn["manipulation"] = {
            "pitch_median_hz": None if m.pitch_median_hz is None else _draw(rng, m.pitch_median_hz),
            "pitch_range_factor": None
            if m.pitch_range_factor is None
            else _draw(rng, m.pitch_range_factor),
            "formant_shift_ratio": None
            if m.formant_shift_ratio is None
            else _draw(rng, m.formant_shift_ratio),
            "duration_factor": None if m.duration_factor is None else _draw(rng, m.duration_factor),
        }
    if recipe.speed_pitch is not None:
        drawn["speed_pitch"] = {
            "speed": _draw(rng, recipe.speed_pitch.speed),
            "pitch_semitones": _draw(rng, recipe.speed_pitch.pitch_semitones),
        }
    if recipe.reverberation is not None:
        r = recipe.reverberation
        dimensions = [_draw(rng, r.room_m) for _ in range(3)]
        drawn["reverberation"] = {
            "rt60": _draw(rng, r.rt60),
            "room_m": dimensions,
            "source": [round(x, 3) for x in _place(rng, dimensions)],
            "microphone": [round(x, 3) for x in _place(rng, dimensions)],
        }
    if recipe.noise is not None:
        n = recipe.noise
        kind = n.kinds[int(rng.integers(len(n.kinds)))]
        drawn["noise"] = {"kind": kind, "snr_db": _draw(rng, n.snr_db)}
        if kind == "babble":
            count = min(n.babble_voices, voices)
            drawn["noise"]["voices"] = sorted(
                int(i) for i in rng.choice(voices, size=count, replace=False)
            )
    return drawn


def augmented_cache_key(config: Config, source_sha256: str, settings: dict[str, Any]) -> str:
    description = {
        "version": [CACHE_VERSION, AUGMENT_VERSION],
        "source": source_sha256,
        "settings": settings,
        "sample_rate": config.synthesis.sample_rate,
        "trim": [config.synthesis.trim.threshold_db, config.synthesis.trim.margin_ms],
        "level": [config.synthesis.level.rms_db, config.synthesis.level.max_peak],
    }
    text = json.dumps(description, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Augmenter:
    """Applies recipes to clips. Holds what the recipes share: the run's long-term spectrum for
    speech-shaped noise, and the clips that babble is made of."""

    def __init__(self, config: Config, synthesis: Synthesis, originals: list[Token]) -> None:
        self.config = config
        self.synthesis = synthesis
        self.rate = config.synthesis.sample_rate
        self.originals = originals
        self._spectrum: np.ndarray | None = None

    @property
    def spectrum(self) -> np.ndarray:
        if self._spectrum is None:
            clips = [self.synthesis.audio(t) for t in self.originals[:SPECTRUM_TOKENS]]
            self._spectrum = long_term_spectrum(clips)
        return self._spectrum

    def noise_for(self, rng: np.random.Generator, samples: int, settings: dict[str, Any]):
        kind = settings["kind"]
        if kind == "white":
            return white_noise(rng, samples)
        if kind == "pink":
            return shaped_noise(rng, samples, pink_magnitude())
        if kind == "speech":
            return shaped_noise(rng, samples, self.spectrum)
        voices = [self.synthesis.audio(self.originals[i]) for i in settings["voices"]]
        return babble(rng, samples, voices)

    def apply(self, clip: np.ndarray, rng: np.random.Generator, settings: dict[str, Any]):
        """The augmented clip: manipulated, perturbed, reverberated, mixed with noise, trimmed,
        and leveled; and the achieved values, each measured right after its own transformation.
        ``rng`` continues the token's substream after the draws."""
        trim = self.config.synthesis.trim
        level = self.config.synthesis.level
        out = np.asarray(clip, dtype=np.float32)
        achieved: dict[str, Any] = {}
        if "manipulation" in settings:
            from semantic_world.wordforms import praat

            m = settings["manipulation"]
            before = praat.measure_pitch(out, self.rate)
            source = out
            duration_before = len(out) / self.rate
            out = praat.manipulate(
                out,
                self.rate,
                formant_shift_ratio=m["formant_shift_ratio"] or 1.0,
                pitch_median_hz=m["pitch_median_hz"],
                pitch_range_factor=m["pitch_range_factor"] or 1.0,
                duration_factor=m["duration_factor"] or 1.0,
            )
            after = praat.measure_pitch(out, self.rate)
            if m["pitch_median_hz"]:
                achieved["pitch_median_hz"] = _achieved(m["pitch_median_hz"], after.median_hz)
            if m["pitch_range_factor"]:
                achieved["pitch_range_semitones"] = _achieved(
                    before.range_semitones * m["pitch_range_factor"], after.range_semitones
                )
            if m["formant_shift_ratio"]:
                # frame by frame, over the frames that are voiced before and after
                achieved["formant_ratio"] = _achieved(
                    m["formant_shift_ratio"], praat.measure_formant_shift(source, out, self.rate)
                )
            if m["duration_factor"]:
                achieved["duration_s"] = _achieved(
                    duration_before * m["duration_factor"], len(out) / self.rate
                )
        if "speed_pitch" in settings:
            from semantic_world.wordforms import praat

            speed = settings["speed_pitch"]["speed"]
            semitones = settings["speed_pitch"]["pitch_semitones"]
            before = praat.measure_pitch(out, self.rate)
            source = out
            duration_before = len(out) / self.rate
            if speed != 1.0:
                out = audio_tools.resample(out, self.rate * speed, self.rate)
            if semitones != 0.0:
                out = praat.change_pitch(out, self.rate, factor=2.0 ** (semitones / 12.0))
            achieved["speed_duration_s"] = _achieved(duration_before / speed, len(out) / self.rate)
            if speed != 1.0 or semitones != 0.0:
                # the median pitch that the measured shift gives: the shift is measured frame
                # by frame, because a changed clip's own median also moves with the frames that
                # Praat finds voiced
                target = before.median_hz * speed * 2.0 ** (semitones / 12.0)
                shift = praat.measure_pitch_shift(source, out, self.rate)
                achieved["speed_pitch_median_hz"] = _achieved(
                    target, before.median_hz * 2.0 ** (shift / 12.0)
                )
        if "reverberation" in settings:
            r = settings["reverberation"]
            rir, _ = room_impulse_response(
                self.rate, tuple(r["room_m"]), r["rt60"], r["source"], r["microphone"]
            )
            achieved["rt60_s"] = _achieved(r["rt60"], reverberation_time(rir, self.rate))
            out = reverberate(out, rir)
            out = audio_tools.trim(out, self.rate, trim.threshold_db, trim.margin_ms)
        if "noise" in settings:
            noise = self.noise_for(rng, len(out), settings["noise"])
            mixed = add_noise(out, noise, settings["noise"]["snr_db"])
            achieved["snr_db"] = _achieved(settings["noise"]["snr_db"], measured_snr_db(mixed, out))
            out = mixed
        leveled = audio_tools.set_level(
            np.asarray(out, dtype=np.float32), level.rms_db, level.max_peak
        )
        return leveled, achieved


def eligible_tokens(config: Config, synthesis: Synthesis) -> list[Token]:
    """The tokens a recipe may be applied to: the clean tokens of the configured speakers. With
    acoustic mapping, a mapped word's clean tokens are its mapped tokens."""
    which = config.augmentation.speakers
    held_out = {s.label: s.held_out for s in synthesis.speakers}
    return [
        t
        for t in synthesis.tokens
        if t.clean and (which == "all" or (which == "held_out") == held_out[t.speaker])
    ]


def augment_synthesis(
    config: Config,
    streams: Streams,
    synthesis: Synthesis,
    progress: Callable[[int, int], None] | None = None,
) -> Augmentation:
    """Apply every recipe of the configuration and append the augmented tokens to the
    synthesis. Clips already in the cache are read back."""
    settings = config.augmentation
    result = Augmentation([], skipped=[])
    if settings is None:
        return result
    originals = [t for t in synthesis.tokens if t.clean]
    candidates = eligible_tokens(config, synthesis)
    augmenter = Augmenter(config, synthesis, originals)
    cache_dir = Path(config.synthesis.cache_dir)
    plan = []
    for number, recipe in enumerate(settings.recipes, start=1):
        for token in candidates:
            # the choice of tokens has its own substream, so the proportion never changes the
            # draws of a chosen token
            chooser = streams.substream("augment", f"{token.label}|{recipe.name}|choose")
            if settings.proportion < 1.0 and chooser.random() >= settings.proportion:
                continue
            rng = streams.substream("augment", f"{token.label}|{recipe.name}")
            plan.append((number, recipe, token, rng))
    for done, (number, recipe, token, rng) in enumerate(plan, start=1):
        drawn = draw_settings(rng, recipe, len(originals))
        drawn["source"] = token.label
        key = augmented_cache_key(config, token.sha256, drawn)
        relative = Path(f"v{CACHE_VERSION}") / "augment" / key[:2] / f"{key}.flac"
        path = cache_dir / relative
        sidecar = path.with_suffix(".json")  # the achieved values, beside the clip
        if path.exists() and sidecar.exists():
            result.cached += 1
            achieved = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            try:
                clip, achieved = augmenter.apply(synthesis.audio(token), rng, drawn)
            except (audio_tools.AudioError, RuntimeError) as error:
                result.skipped.append(
                    {"token": token.label, "recipe": recipe.name, "reason": str(error)}
                )
                continue
            audio_tools.write_flac(path, clip, config.synthesis.sample_rate)
            sidecar.write_text(json.dumps(achieved, sort_keys=True), encoding="utf-8")
            result.computed += 1
        clip, _ = audio_tools.read_flac(path)
        result.tokens.append(
            replace(
                token,
                label=f"{token.label}.A.{number}",
                duration=round(len(clip) / config.synthesis.sample_rate, 6),
                cache_path=relative.as_posix(),
                sha256=audio_tools.sha256_file(path),
                tries=1,
                peak=round(audio_tools.peak(clip), 6),
                rms_db=round(audio_tools.rms_db(clip), 4),
                augmentation=json.dumps(drawn, sort_keys=True),
                achieved=json.dumps(achieved, sort_keys=True),
            )
        )
        if progress is not None:
            progress(done, len(plan))
    synthesis.tokens.extend(result.tokens)
    synthesis.augmentation = result.summary()
    return result
