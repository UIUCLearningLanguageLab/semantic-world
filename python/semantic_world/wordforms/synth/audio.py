"""Audio processing for synthesized clips: resampling, trimming, level, and FLAC files.

Audio is mono, 16 kHz, and 32-bit float in memory, and 16-bit FLAC on disk. Every clip has the
same RMS level, unless its peak would then exceed the configured limit. A clip is always
returned as read back from its file, so a clip is the same whether it was just synthesized or
came from the cache.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np


class AudioError(RuntimeError):
    """A clip could not be processed, for example because the engine returned silence."""


def resample(audio: np.ndarray, source_rate: float, target_rate: int) -> np.ndarray:
    """Resample by polyphase filtering. ``source_rate`` may be fractional: reading a clip at a
    higher rate than the rate at which it was synthesized raises its pitch and shortens it."""
    from scipy.signal import resample_poly

    ratio = Fraction(target_rate / source_rate).limit_denominator(1000)
    if ratio == 1:
        return audio.astype(np.float32)
    return resample_poly(audio.astype(np.float64), ratio.numerator, ratio.denominator).astype(
        np.float32
    )


def trim(audio: np.ndarray, rate: int, threshold_db: float, margin_ms: float) -> np.ndarray:
    """Remove leading and trailing silence. A sample is sound when its absolute value is above
    ``threshold_db`` relative to the clip's peak. ``margin_ms`` of audio is kept on each side."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        raise AudioError("the clip is silent")
    loud = np.flatnonzero(np.abs(audio) > peak * 10.0 ** (threshold_db / 20.0))
    margin = int(round(margin_ms * rate / 1000.0))
    start = max(0, int(loud[0]) - margin)
    end = min(len(audio), int(loud[-1]) + 1 + margin)
    return audio[start:end]


def rms_db(audio: np.ndarray) -> float:
    """The RMS level of a clip in dB relative to full scale."""
    power = float(np.mean(np.square(audio, dtype=np.float64))) if audio.size else 0.0
    if power <= 0.0:
        raise AudioError("the clip is silent")
    return 10.0 * float(np.log10(power))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def set_level(audio: np.ndarray, target_rms_db: float, max_peak: float) -> np.ndarray:
    """Scale a clip to the target RMS level. When the scaled clip's peak would exceed
    ``max_peak``, the clip is scaled to that peak instead, and its RMS level is lower."""
    gain = 10.0 ** ((target_rms_db - rms_db(audio)) / 20.0)
    largest = peak(audio)
    if largest * gain > max_peak:
        gain = max_peak / largest
    return (audio * gain).astype(np.float32)


def silence_margins(audio: np.ndarray, threshold_db: float) -> tuple[int, int]:
    """The number of samples before the first sound and after the last sound, with sound defined
    as in :func:`trim`."""
    peak = float(np.max(np.abs(audio)))
    loud = np.flatnonzero(np.abs(audio) > peak * 10.0 ** (threshold_db / 20.0))
    return int(loud[0]), int(len(audio) - 1 - loud[-1])


def write_flac(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.flac")
    soundfile.write(temporary, audio, rate, format="FLAC", subtype="PCM_16")
    temporary.replace(path)


def read_flac(path: Path) -> tuple[np.ndarray, int]:
    """The clip as a float32 array, and its sample rate."""
    import soundfile

    audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    return audio, int(rate)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
