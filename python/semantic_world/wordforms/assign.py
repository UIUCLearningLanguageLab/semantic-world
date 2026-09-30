"""Sound-meaning assignment: which word form goes with which meaning.

The meanings are any table of IDs with binary feature vectors, such as the taxonomy generator's
``categories_generative.csv``: a CSV file whose first column is the ID and whose other columns
hold 0 and 1. Stage 4 provides the ``arbitrary`` mode, a seeded random assignment from the
``wordforms:assign`` stream. The other modes arrive in stage 7.

Every assignment reports the sound-meaning correlation: the correlation, across pairs of
meanings, between the phoneme edit distance of their words and the Hamming distance of their
feature vectors. A null distribution comes from random reassignments of the same words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.wordforms.generate import WordForm

NULL_SAMPLES = 1000
"""The number of random reassignments in the null distribution."""


class AssignmentError(ValueError):
    """The assignment cannot be made, for example because there are more meanings than words."""


def load_meanings(path: str | Path) -> tuple[list[str], np.ndarray]:
    """The IDs and the feature matrix (meanings by features, 0 and 1) of a meanings table."""
    table = pl.read_csv(path, infer_schema_length=None)
    if table.width < 2 or table.height == 0:
        raise AssignmentError(f"{path}: expected an ID column and at least one feature column")
    ids = [str(value) for value in table.to_series(0)]
    if len(set(ids)) != len(ids):
        raise AssignmentError(f"{path}: the IDs in the first column are not distinct")
    try:
        features = table.select(table.columns[1:]).cast(pl.Float64).to_numpy()
    except pl.exceptions.PolarsError as error:
        raise AssignmentError(f"{path}: the feature columns must hold numbers") from error
    if not np.isin(features, (0.0, 1.0)).all():
        raise AssignmentError(f"{path}: the feature columns must hold only 0 and 1")
    return ids, features.astype(np.int8)


def hamming_distances(features: np.ndarray) -> np.ndarray:
    """The proportion of differing features for every pair of meanings (``np.triu_indices``
    order)."""
    rows, columns = np.triu_indices(len(features), 1)
    return (features[rows] != features[columns]).mean(axis=1)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Assignment:
    """Meanings with their words, and the sound-meaning correlation."""

    mode: str
    meanings: list[str]
    words: list[WordForm]
    """The word of each meaning, in the order of ``meanings``."""
    summary: dict[str, Any]

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "meaning": self.meanings,
                "word": [w.label for w in self.words],
                "spelling": [w.spelling for w in self.words],
                "arpabet": [w.arpabet for w in self.words],
            }
        )

    def write(self, folder: str | Path) -> Path:
        out = Path(folder) / "assignment"
        out.mkdir(parents=True, exist_ok=True)
        self.frame().write_csv(out / "lexicon.csv")
        (out / "summary.yaml").write_text(
            yaml.safe_dump(self.summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return out


def assign_arbitrary(
    words: list[WordForm],
    meanings: list[str],
    features: np.ndarray,
    rng: np.random.Generator,
    null_samples: int = NULL_SAMPLES,
) -> Assignment:
    """A seeded random assignment: each meaning gets one word, and no word is used twice. With
    more words than meanings, the words that are left have no meaning."""
    if len(meanings) > len(words):
        raise AssignmentError(
            f"there are {len(meanings)} meanings and only {len(words)} words; raise wordforms.count"
        )
    chosen = rng.permutation(len(words))[: len(meanings)]
    assigned = [words[int(i)] for i in chosen]
    from semantic_world.wordforms.evaluate import edit_distance_matrix

    sound = edit_distance_matrix(assigned)
    rows, columns = np.triu_indices(len(meanings), 1)
    meaning = hamming_distances(features)
    observed = correlation(sound[rows, columns], meaning)
    null = np.empty(null_samples, dtype=np.float64)
    for k in range(null_samples):
        order = rng.permutation(len(meanings))
        null[k] = correlation(sound[np.ix_(order, order)][rows, columns], meaning)
    finite = null[np.isfinite(null)]
    summary: dict[str, Any] = {
        "mode": "arbitrary",
        "meanings": len(meanings),
        "features": int(features.shape[1]),
        "words_available": len(words),
        "sound_distance": "phoneme edit distance",
        "meaning_distance": "Hamming",
        "correlation": _rounded(observed),
        "null": {
            "samples": null_samples,
            "mean": _rounded(finite.mean() if finite.size else float("nan")),
            "std": _rounded(finite.std(ddof=1) if finite.size > 1 else float("nan")),
            "p05": _rounded(np.percentile(finite, 5) if finite.size else float("nan")),
            "p95": _rounded(np.percentile(finite, 95) if finite.size else float("nan")),
            "p_value": _rounded(
                (np.sum(np.abs(finite) >= abs(observed)) + 1) / (finite.size + 1)
                if finite.size and np.isfinite(observed)
                else float("nan")
            ),
        },
    }
    return Assignment("arbitrary", list(meanings), assigned, summary)


def _rounded(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)
