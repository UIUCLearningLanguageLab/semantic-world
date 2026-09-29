"""Similarity between binary feature vectors: phi, cosine, and Jaccard.

A similarity that is undefined (phi of a constant vector, cosine or Jaccard of an all-zero
vector) is NaN. Callers decide what NaN means: the superordinate bound rejects a candidate with
an undefined similarity, and the analysis statistics leave undefined pairs out.
"""

from __future__ import annotations

import numpy as np

METRICS = ("phi", "cosine", "jaccard")


def similarity_matrix(rows: np.ndarray, metric: str) -> np.ndarray:
    """Pairwise similarities between the rows of a binary matrix, as a square array with NaN
    where the similarity is undefined."""
    return cross_similarity(rows, rows, metric)


def cross_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Similarities between every row of ``a`` and every row of ``b``, as an array of shape
    ``(len(a), len(b))``."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"expected two matrices with the same number of columns, got {x.shape} and {y.shape}"
        )
    n = x.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "phi":
            if n == 0:
                return np.full((x.shape[0], y.shape[0]), np.nan)
            xc = x - x.mean(axis=1, keepdims=True)
            yc = y - y.mean(axis=1, keepdims=True)
            numerator = xc @ yc.T
            denominator = np.sqrt(np.outer((xc**2).sum(axis=1), (yc**2).sum(axis=1)))
            result = numerator / denominator
        elif metric == "cosine":
            numerator = x @ y.T
            denominator = np.sqrt(np.outer((x**2).sum(axis=1), (y**2).sum(axis=1)))
            result = numerator / denominator
        elif metric == "jaccard":
            intersection = x @ y.T
            union = x.sum(axis=1)[:, None] + y.sum(axis=1)[None, :] - intersection
            result = intersection / union
        else:
            raise ValueError(
                f"unknown similarity metric {metric!r}; the metrics are {', '.join(METRICS)}"
            )
    result = np.asarray(result, dtype=float)
    result[~np.isfinite(result)] = np.nan
    if metric == "phi":
        np.clip(result, -1.0, 1.0, out=result)
    return result


def pair_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """The similarity of two vectors, or NaN."""
    return float(cross_similarity(np.atleast_2d(a), np.atleast_2d(b), metric)[0, 0])
