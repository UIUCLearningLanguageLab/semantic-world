"""Acoustic mapping: semantic features shift acoustic properties of every token of a word.

Under an assignment of words to meanings, each configured mapping names a semantic feature, an
acoustic property, and an amount. Every token of a word whose meaning has the feature is
changed by that amount: the median pitch (``pitch``, in semitones), the formants (``formants``,
a ratio, as a longer or shorter vocal tract), the duration (``duration``, a factor), or the
spectral tilt (``tilt``, in decibels per octave). Acoustic mapping models sound symbolism of the
frequency-code and bouba/kiki kinds.

The changes use the manipulation tools of stage 5 (:mod:`semantic_world.wordforms.praat`, and
the spectral tilt in :mod:`semantic_world.wordforms.synth.audio`). A mapped token is a new token,
labeled ``<source token>.M``, with the same word and speaker, and it records each mapping's
target beside the value measured right after the change (``achieved`` in ``tokens.csv``), like
an augmented token. Its recipe is ``acoustic_mapping``.

The analysis uses the achieved values, never the targets: for each mapping, the mean and spread
of the measured shifts, how many tokens miss the amount by more than 5% and 10%, and the
correlation between the feature and the measured shift over every token of the assigned words
(a token of a word without the feature has the neutral shift).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from semantic_world.wordforms.augment import MISS_LEVELS, _achieved
from semantic_world.wordforms.config import AcousticMappingConfig, Config
from semantic_world.wordforms.synth import CACHE_VERSION, Synthesis, Token
from semantic_world.wordforms.synth import audio as audio_tools

MAPPING_VERSION = 1
"""Part of every mapped clip's cache key. Raise it when the manipulations change."""
RECIPE = "acoustic_mapping"
ACHIEVED_KEYS = {
    "pitch": "pitch_semitones",
    "formants": "formant_ratio",
    "duration": "duration_factor",
    "tilt": "tilt_db_per_octave",
}
NEUTRAL = {"pitch": 0.0, "formants": 1.0, "duration": 1.0, "tilt": 0.0}
"""The shift of a token that a mapping does not change."""


def apply_mappings(
    clip: np.ndarray, rate: int, mappings: list[AcousticMappingConfig]
) -> tuple[np.ndarray, dict[str, Any]]:
    """A clip with each mapping applied in order, and the achieved values: each mapping's amount
    beside the shift measured right after it."""
    from semantic_world.wordforms import praat

    out = np.asarray(clip, dtype=np.float32)
    achieved: dict[str, Any] = {}
    for mapping in mappings:
        key = ACHIEVED_KEYS[mapping.property]
        if mapping.property == "pitch":
            before = praat.measure_pitch(out, rate).median_hz
            out = praat.change_pitch(out, rate, factor=2.0 ** (mapping.amount / 12.0))
            after = praat.measure_pitch(out, rate).median_hz
            measured = 12.0 * np.log2(after / before)
        elif mapping.property == "formants":
            before = out
            out = praat.manipulate(out, rate, formant_shift_ratio=mapping.amount)
            measured = praat.measure_formant_shift(before, out, rate)
        elif mapping.property == "duration":
            before = len(out)
            out = praat.change_duration(out, rate, mapping.amount)
            measured = len(out) / before
        else:
            before = audio_tools.spectral_tilt(out, rate)
            out = audio_tools.change_tilt(out, rate, mapping.amount)
            measured = audio_tools.spectral_tilt(out, rate) - before
        achieved[key] = _achieved(mapping.amount, measured)
    return out, achieved


def mapped_cache_key(config: Config, source_sha256: str, mappings: list[dict[str, Any]]) -> str:
    description = {
        "version": [CACHE_VERSION, MAPPING_VERSION],
        "source": source_sha256,
        "mappings": mappings,
        "level": [config.synthesis.level.rms_db, config.synthesis.level.max_peak],
    }
    text = json.dumps(description, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def map_tokens(
    config: Config,
    synthesis: Synthesis,
    assignment,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Apply the configuration's acoustic mappings to the synthesized tokens of the assigned
    words, append the mapped tokens to the synthesis, and return the report. Clips already in
    the cache are read back."""
    table = assignment.features
    mappings = list(config.assignment.acoustic)
    feature_of = {m.feature: table.feature(m.feature) for m in mappings}
    meaning_of = {w.label: i for i, w in enumerate(assignment.words)}
    level = config.synthesis.level
    cache_dir = Path(config.synthesis.cache_dir)
    rate = config.synthesis.sample_rate
    originals = [t for t in synthesis.tokens if not t.augmentation and t.word in meaning_of]
    mapped: list[Token] = []
    skipped: list[dict[str, str]] = []
    computed = cached = 0
    shifts: dict[str, list[tuple[int, float]]] = {
        m.feature + "|" + m.property: [] for m in mappings
    }
    for done, token in enumerate(originals, start=1):
        row = meaning_of[token.word]
        active = [m for m in mappings if feature_of[m.feature][row] == 1]
        achieved: dict[str, Any] = {}
        if active:
            drawn = {
                "recipe": RECIPE,
                "source": token.label,
                "meaning": assignment.meanings[row],
                "mappings": [m.resolved() for m in active],
            }
            key = mapped_cache_key(config, token.sha256, drawn["mappings"])
            relative = Path(f"v{CACHE_VERSION}") / "mapping" / key[:2] / f"{key}.flac"
            path = cache_dir / relative
            sidecar = path.with_suffix(".json")
            try:
                if path.exists() and sidecar.exists():
                    cached += 1
                    achieved = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    clip, achieved = apply_mappings(synthesis.audio(token), rate, active)
                    clip = audio_tools.set_level(clip, level.rms_db, level.max_peak)
                    audio_tools.write_flac(path, clip, rate)
                    sidecar.write_text(json.dumps(achieved, sort_keys=True), encoding="utf-8")
                    computed += 1
            except (audio_tools.AudioError, RuntimeError) as error:
                skipped.append({"token": token.label, "reason": str(error)})
                achieved = {}
            else:
                clip, _ = audio_tools.read_flac(path)
                mapped.append(
                    replace(
                        token,
                        label=f"{token.label}.M",
                        duration=round(len(clip) / rate, 6),
                        cache_path=relative.as_posix(),
                        sha256=audio_tools.sha256_file(path),
                        tries=1,
                        peak=round(audio_tools.peak(clip), 6),
                        rms_db=round(audio_tools.rms_db(clip), 4),
                        augmentation=json.dumps(drawn, sort_keys=True),
                        achieved=json.dumps(achieved, sort_keys=True),
                    )
                )
        # every token of an assigned word enters the analysis with its achieved shift: the
        # measured value where a mapping applied, and the neutral value where none did
        for m in mappings:
            entry = achieved.get(ACHIEVED_KEYS[m.property]) if m in active else None
            value = NEUTRAL[m.property] if m not in active else (entry or {}).get("measured")
            if value is not None:
                shifts[m.feature + "|" + m.property].append((int(m in active), float(value)))
        if progress is not None:
            progress(done, len(originals))
    synthesis.tokens.extend(mapped)
    report: dict[str, Any] = {
        "tokens_of_assigned_words": len(originals),
        "mapped_tokens": len(mapped),
        "computed": computed,
        "read_from_cache": cached,
        "skipped": skipped,
        "mappings": [],
    }
    for m in mappings:
        pairs = shifts[m.feature + "|" + m.property]
        has = np.array([p[0] for p in pairs], dtype=np.float64)
        value = np.array([p[1] for p in pairs], dtype=np.float64)
        measured = value[has == 1]
        misses = np.abs(measured / m.amount - 1.0) if m.amount != 0 and measured.size else measured
        if has.std() > 0 and value.std() > 0:
            feature_correlation = float(np.corrcoef(has, value)[0, 1])
        else:
            feature_correlation = float("nan")
        report["mappings"].append(
            {
                **m.resolved(),
                "quantity": ACHIEVED_KEYS[m.property],
                "words_with_the_feature": int(feature_of[m.feature][: len(assignment.words)].sum()),
                "tokens": int(measured.size),
                "achieved_mean": None if not measured.size else round(float(measured.mean()), 6),
                "achieved_std": None
                if measured.size < 2
                else round(float(measured.std(ddof=1)), 6),
                "over_5_percent": int((misses > MISS_LEVELS[0]).sum()) if m.amount != 0 else None,
                "over_10_percent": int((misses > MISS_LEVELS[1]).sum()) if m.amount != 0 else None,
                # from the achieved shifts: 1 would be a perfect, noiseless mapping
                "feature_correlation": None
                if not np.isfinite(feature_correlation)
                else round(feature_correlation, 6),
            }
        )
    return report
