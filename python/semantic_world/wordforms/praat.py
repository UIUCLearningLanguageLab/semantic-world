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
- :func:`measure_formants` and :func:`measure_formant_shift`: the mean formants of a clip or a
  span, and the ratio between the formants of a clip and of a changed copy of it;
- :func:`measure_pitch_shift`: the shift in pitch between a clip and a changed copy of it.

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


def measure_pitch_shift(
    before: np.ndarray,
    after: np.ndarray,
    rate: int,
    time_range: tuple[float, float] | None = None,
    floor: float = PITCH_FLOOR,
    ceiling: float = PITCH_CEILING,
) -> float:
    """The shift in pitch, in semitones, from ``before`` to ``after``, where ``after`` is a
    changed copy of ``before`` (its duration may differ: the frames are matched by their
    relative time). ``time_range`` is a span of ``before``, in seconds.

    The shift is the median, over the frames that are voiced in both clips, of the frame's pitch
    after over its pitch before. A change of pitch can change which frames Praat finds voiced,
    which moves a clip's median pitch by itself, so the difference between the two clips' median
    pitches is a much noisier measure of the same shift. NaN without a frame that is voiced in
    both clips."""
    times, hz = pitch_track(before, rate, floor, ceiling)
    if time_range is not None:
        keep = (times >= time_range[0]) & (times < time_range[1])
        times, hz = times[keep], hz[keep]
    changed_times, changed = pitch_track(after, rate, floor, ceiling)
    if not len(hz) or not len(changed):
        return float("nan")
    matched = times * (len(after) / len(before))
    changed = changed[np.clip(np.searchsorted(changed_times, matched), 0, len(changed) - 1)]
    both = (hz > 0) & (changed > 0)
    if not both.any():
        return float("nan")
    return float(np.median(12.0 * np.log2(changed[both] / hz[both])))


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


FORMANT_CEILING = 5500.0
"""The highest formant frequency Praat looks for, in Hz."""


def measure_formants(
    clip: np.ndarray,
    rate: int,
    time_range: tuple[float, float] | None = None,
    count: int = 3,
    voiced_only: bool = True,
) -> tuple[float, ...]:
    """The mean of the first ``count`` formant frequencies (Hz) over the frames of a clip, or of
    a span, from Praat's Burg formant analysis, by default over the voiced frames only; NaN for
    a formant that no frame has."""
    from parselmouth.praat import call

    formant = call(sound(clip, rate), "To Formant (burg)", 0.0, 5, FORMANT_CEILING, 0.025, 50.0)
    times = np.arange(formant.n_frames) * formant.dt + formant.t1
    if time_range is not None:
        times = times[(times >= time_range[0]) & (times < time_range[1])]
    if voiced_only:
        pitch_times, hz = pitch_track(clip, rate)
        voiced = hz[np.clip(np.searchsorted(pitch_times, times), 0, len(hz) - 1)] > 0
        if voiced.any():
            times = times[voiced]
    means = []
    for k in range(1, count + 1):
        values = np.array(
            [call(formant, "Get value at time", k, t, "Hertz", "Linear") for t in times]
        )
        values = values[np.isfinite(values)]
        means.append(float(values.mean()) if values.size else float("nan"))
    return tuple(means)


FORMANT_SHIFT_SCALES = tuple(round(0.7 + 0.025 * k, 3) for k in range(31))
"""The shifts that :func:`measure_formant_shift` tries: 0.7 to 1.45 in steps of 0.025."""


def measure_formant_shift(
    before: np.ndarray,
    after: np.ndarray,
    rate: int,
    time_range: tuple[float, float] | None = None,
    count: int = 3,
) -> float:
    """The ratio by which the formants of ``after`` lie above those of ``before``, where
    ``after`` is a changed copy of ``before`` (its duration may differ: the frames are matched
    by their relative time). ``time_range`` is a span of ``before``, in seconds.

    The ratio is measured frame by frame, over the frames that are voiced in both clips: the
    median, over the frames and the first ``count`` formants, of the formant after over the
    formant before. A formant analysis looks for five formants below a ceiling, and a shifted
    voice needs a shifted ceiling, so the analysis of ``after`` is repeated with the ceiling
    scaled by each of ``FORMANT_SHIFT_SCALES``, and the scale that the frames agree with most
    closely (the smallest median distance between the frames' ratios and the scale) is used.
    The measure does not use the shift that a manipulation aimed at. NaN without a frame that
    is voiced in both clips."""
    from parselmouth.praat import call

    def formants(clip, ceiling, times):
        analysis = call(sound(clip, rate), "To Formant (burg)", 0.0, 5, ceiling, 0.025, 50.0)
        if times is None:
            times = np.arange(analysis.n_frames) * analysis.dt + analysis.t1
        values = [
            [call(analysis, "Get value at time", k, t, "Hertz", "Linear") for t in times]
            for k in range(1, count + 1)
        ]
        return times, np.array(values, dtype=np.float64)

    def voiced_at(clip, times):
        pitch_times, hz = pitch_track(clip, rate)
        return hz[np.clip(np.searchsorted(pitch_times, times), 0, len(hz) - 1)] > 0

    times, reference = formants(before, FORMANT_CEILING, None)
    if time_range is not None:
        keep = (times >= time_range[0]) & (times < time_range[1])
        times, reference = times[keep], reference[:, keep]
    matched = times * (len(after) / len(before))
    voiced = voiced_at(before, times) & voiced_at(after, matched)
    if not voiced.any():
        return float("nan")
    best = (float("inf"), float("nan"))
    for scale in FORMANT_SHIFT_SCALES:
        ceiling = min(FORMANT_CEILING * scale, rate / 2.0)
        _, shifted = formants(after, ceiling, matched[voiced])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.log(shifted / reference[:, voiced])
        ratios = ratios[np.isfinite(ratios)]
        if not ratios.size:
            continue
        distance = float(np.median(np.abs(ratios - np.log(scale))))
        if distance < best[0]:
            best = (distance, float(np.exp(np.median(ratios))))
    return best[1]


def praat_version() -> str:
    """The version of Praat inside parselmouth, for provenance."""
    parselmouth = _parselmouth()
    return f"parselmouth {parselmouth.__version__}, Praat {parselmouth.PRAAT_VERSION}"
