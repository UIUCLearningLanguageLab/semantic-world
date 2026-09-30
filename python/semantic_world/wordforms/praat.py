"""Acoustic measurement and manipulation with Praat, through ``praat-parselmouth``.

The tools work on any clip (mono float32 or float64 at a sample rate), whole or within a time
range in seconds, so that later work on connected speech can apply them to the aligned span of a
word, a vowel, or a syllable inside an utterance:

- :func:`measure_pitch`: the median pitch and the pitch range of a clip or a span;
- :func:`change_pitch`: move the median pitch and scale the pitch range, through Praat's
  Manipulation object and its pitch tier, within a span;
- :func:`change_duration`: stretch or shrink a span, through the duration tier;
- :func:`manipulate`: Praat's "Change gender" command (formant shift ratio, new pitch median,
  pitch range factor, duration factor), applied to a span and spliced back.

``praat-parselmouth`` is GPL-3.0, so this module imports it only inside the functions that use
it, and nothing else in the package imports it (``docs/specs/WORDFORM_PIPELINE.md``, "Licenses").
Praat is deterministic: the same clip and settings give the same audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PITCH_FLOOR = 75.0
"""The lowest pitch Praat looks for, in Hz."""
PITCH_CEILING = 600.0
"""The highest pitch Praat looks for, in Hz."""
PITCH_TIME_STEP = 0.005
"""The spacing of pitch measurements, in seconds."""
RANGE_PERCENTILES = (5.0, 95.0)
"""The pitch range is the distance between these percentiles of the voiced pitch, in
semitones."""


class PraatError(RuntimeError):
    """A manipulation could not be made, for example because the clip has no voiced frames."""


def _parselmouth():
    try:
        import parselmouth
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "acoustic manipulation needs the praat-parselmouth package; install the 'speech' extra"
        ) from error
    return parselmouth


def sound(clip: np.ndarray, rate: int):
    """A clip as a Praat Sound."""
    parselmouth = _parselmouth()
    return parselmouth.Sound(np.asarray(clip, dtype=np.float64), sampling_frequency=rate)


def _span(clip: np.ndarray, rate: int, time_range: tuple[float, float] | None) -> tuple[int, int]:
    """The sample range of a time range, or of the whole clip."""
    if time_range is None:
        return 0, len(clip)
    start, end = time_range
    if not 0.0 <= start < end:
        raise ValueError(f"the time range must satisfy 0 <= start < end, got {time_range}")
    first = int(round(start * rate))
    last = min(len(clip), int(round(end * rate)))
    if last - first < 2:
        raise ValueError(f"the time range {time_range} holds fewer than two samples")
    return first, last


@dataclass(frozen=True)
class PitchMeasure:
    """The pitch of a clip or a span: the median of the voiced frames in Hz, the range between
    the 5th and 95th percentiles in semitones, and the frame counts."""

    median_hz: float
    range_semitones: float
    voiced_frames: int
    frames: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "median_hz": round(self.median_hz, 3),
            "range_semitones": round(self.range_semitones, 3),
            "voiced_frames": self.voiced_frames,
            "frames": self.frames,
        }


def pitch_track(
    clip: np.ndarray,
    rate: int,
    floor: float = PITCH_FLOOR,
    ceiling: float = PITCH_CEILING,
    time_step: float = PITCH_TIME_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Praat's pitch track: the frame times and the pitch in Hz (0 where unvoiced)."""
    pitch = sound(clip, rate).to_pitch(
        time_step=time_step, pitch_floor=floor, pitch_ceiling=ceiling
    )
    return np.asarray(pitch.xs()), np.asarray(pitch.selected_array["frequency"])


def measure_pitch(
    clip: np.ndarray,
    rate: int,
    time_range: tuple[float, float] | None = None,
    floor: float = PITCH_FLOOR,
    ceiling: float = PITCH_CEILING,
) -> PitchMeasure:
    """The median pitch and the pitch range of a clip, or of the span ``time_range`` (seconds).
    NaN values when the span has no voiced frame."""
    times, hz = pitch_track(clip, rate, floor, ceiling)
    if time_range is not None:
        start, end = time_range
        keep = (times >= start) & (times < end)
        times, hz = times[keep], hz[keep]
    voiced = hz[hz > 0]
    if voiced.size == 0:
        return PitchMeasure(float("nan"), float("nan"), 0, int(hz.size))
    low, high = np.percentile(voiced, RANGE_PERCENTILES)
    return PitchMeasure(
        float(np.median(voiced)),
        float(12.0 * np.log2(high / low)),
        int(voiced.size),
        int(hz.size),
    )


def _manipulation(clip: np.ndarray, rate: int, floor: float, ceiling: float):
    from parselmouth.praat import call

    return call(sound(clip, rate), "To Manipulation", PITCH_TIME_STEP, floor, ceiling)


def _resynthesis(manipulation) -> np.ndarray:
    from parselmouth.praat import call

    result = call(manipulation, "Get resynthesis (overlap-add)")
    return np.asarray(result.values[0], dtype=np.float32)


def change_pitch(
    clip: np.ndarray,
    rate: int,
    *,
    median_hz: float | None = None,
    factor: float | None = None,
    range_factor: float = 1.0,
    time_range: tuple[float, float] | None = None,
    floor: float = PITCH_FLOOR,
    ceiling: float = PITCH_CEILING,
) -> np.ndarray:
    """Move the pitch of a clip, or of a span, to a new median (``median_hz``, or the measured
    median times ``factor``), and scale its range around the new median by ``range_factor``:
    each voiced frame's pitch ``f`` becomes ``new + (f - median) * factor * range_factor``, as
    in Praat's "Change gender". The audio outside the span is unchanged, and the clip keeps its
    length. Raises :class:`PraatError` when the span has no voiced frame."""
    from parselmouth.praat import call

    if median_hz is not None and factor is not None:
        raise ValueError("give median_hz or factor, not both")
    measured = measure_pitch(clip, rate, time_range, floor, ceiling)
    if measured.voiced_frames == 0:
        raise PraatError("the span has no voiced frame to change the pitch of")
    scale = (
        1.0
        if median_hz is None and factor is None
        else (factor if factor is not None else median_hz / measured.median_hz)
    )
    if scale == 1.0 and range_factor == 1.0:
        return np.asarray(clip, dtype=np.float32)
    manipulation = _manipulation(clip, rate, floor, ceiling)
    tier = call(manipulation, "Extract pitch tier")
    start, end = (0.0, len(clip) / rate) if time_range is None else time_range
    old = measured.median_hz
    formula = (
        f"if x >= {start!r} and x <= {end!r} then "
        f"{old * scale!r} + (self - {old!r}) * {scale * range_factor!r} else self fi"
    )
    call(tier, "Formula", formula)
    call([manipulation, tier], "Replace pitch tier")
    return _resynthesis(manipulation)


def change_duration(
    clip: np.ndarray,
    rate: int,
    factor: float,
    time_range: tuple[float, float] | None = None,
    floor: float = PITCH_FLOOR,
    ceiling: float = PITCH_CEILING,
) -> np.ndarray:
    """Stretch (``factor`` above 1) or shrink a clip, or a span of it, keeping the pitch, through
    Praat's duration tier. The audio outside the span keeps its timing."""
    from parselmouth.praat import call

    if factor <= 0:
        raise ValueError("the duration factor must be positive")
    if factor == 1.0:
        return np.asarray(clip, dtype=np.float32)
    manipulation = _manipulation(clip, rate, floor, ceiling)
    tier = call(manipulation, "Extract duration tier")
    total = len(clip) / rate
    step = 1.0 / rate
    start, end = (0.0, total) if time_range is None else time_range
    if start > 0.0:
        call(tier, "Add point", 0.0, 1.0)
        call(tier, "Add point", max(0.0, start - step), 1.0)
    call(tier, "Add point", start, factor)
    call(tier, "Add point", end, factor)
    if end < total:
        call(tier, "Add point", min(total, end + step), 1.0)
        call(tier, "Add point", total, 1.0)
    call([manipulation, tier], "Replace duration tier")
    return _resynthesis(manipulation)


def manipulate(
    clip: np.ndarray,
    rate: int,
    *,
    formant_shift_ratio: float = 1.0,
    pitch_median_hz: float | None = None,
    pitch_range_factor: float = 1.0,
    duration_factor: float = 1.0,
    time_range: tuple[float, float] | None = None,
    floor: float = PITCH_FLOOR,
    ceiling: float = PITCH_CEILING,
) -> np.ndarray:
    """Praat's "Change gender" command on a clip, or on a span of it: the formants are shifted
    by ``formant_shift_ratio`` (above 1 for a shorter vocal tract), the pitch median moves to
    ``pitch_median_hz`` (None keeps it), the pitch range is scaled by ``pitch_range_factor``
    around the new median, and the duration by ``duration_factor``. A span is cut out,
    changed, and spliced back, so the clip's length changes with the span's."""
    from parselmouth.praat import call

    if (
        formant_shift_ratio == 1.0
        and pitch_median_hz is None
        and pitch_range_factor == 1.0
        and duration_factor == 1.0
    ):
        return np.asarray(clip, dtype=np.float32)
    first, last = _span(clip, rate, time_range)
    part = np.asarray(clip[first:last], dtype=np.float64)
    changed = call(
        sound(part, rate),
        "Change gender",
        floor,
        ceiling,
        formant_shift_ratio,
        0.0 if pitch_median_hz is None else pitch_median_hz,
        pitch_range_factor,
        duration_factor,
    )
    middle = np.asarray(changed.values[0], dtype=np.float32)
    return np.concatenate(
        [np.asarray(clip[:first], dtype=np.float32), middle, np.asarray(clip[last:], np.float32)]
    )


def praat_version() -> str:
    """The version of Praat inside parselmouth, for provenance."""
    parselmouth = _parselmouth()
    return f"parselmouth {parselmouth.__version__}, Praat {parselmouth.PRAAT_VERSION}"
