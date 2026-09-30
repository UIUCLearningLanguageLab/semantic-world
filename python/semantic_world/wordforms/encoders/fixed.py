"""The fixed encoder: time-bin averages of a front end, with an optional PCA projection.

A clip's frames are averaged within ``time_bins`` equal time bins, and the bins are joined into
one vector of ``time_bins * channels`` numbers. The bins divide the clip's duration exactly: a
frame that straddles two bins counts in each by its share, so clips with fewer frames than bins
work too. No learning is involved beyond the projection.

The projection is an exact principal component analysis of the training-speaker tokens. It needs
no random numbers, so the ``wordforms:pca`` stream is not drawn from. Each component's sign is
fixed (its largest entry is positive), so the projection is the same on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from semantic_world.wordforms.frontends import Frontend


@lru_cache(maxsize=4096)
def bin_weights(frames: int, bins: int) -> np.ndarray:
    """The weights that average ``frames`` frames into ``bins`` equal time bins: an array of
    bins by frames whose rows sum to one."""
    edges = np.linspace(0.0, frames, bins + 1)
    starts = np.arange(frames, dtype=np.float64)
    overlap = np.minimum(edges[1:, None], starts[None, :] + 1.0) - np.maximum(
        edges[:-1, None], starts[None, :]
    )
    weights = np.maximum(overlap, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


def time_bin_means(frames: np.ndarray, bins: int) -> np.ndarray:
    """The frames (frames by channels) averaged within equal time bins, as one vector of
    ``bins * channels`` numbers: bin 1's channels, then bin 2's, and so on."""
    if len(frames) == 0:
        raise ValueError("a clip with no frames cannot be encoded")
    means = bin_weights(len(frames), bins) @ np.asarray(frames, dtype=np.float64)
    return means.reshape(-1).astype(np.float32)


@dataclass(frozen=True)
class Projection:
    """A principal component projection: ``(x - mean) @ components.T``."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray

    @property
    def dims(self) -> int:
        return len(self.components)

    def apply(self, features: np.ndarray) -> np.ndarray:
        centered = np.asarray(features, dtype=np.float64) - self.mean
        return (centered @ self.components.T).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            explained_variance=self.explained_variance,
        )

    @classmethod
    def load(cls, path: Path) -> Projection:
        data = np.load(path)
        return cls(data["mean"], data["components"], data["explained_variance"])


def fit_pca(features: np.ndarray, dims: int) -> Projection:
    """The projection onto the first ``dims`` principal components of ``features`` (rows are
    tokens). ``dims`` is lowered to the number of available components when it is larger."""
    data = np.asarray(features, dtype=np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    dims = max(1, min(dims, data.shape[1], data.shape[0] - 1))
    if data.shape[0] >= data.shape[1]:
        values, vectors = np.linalg.eigh(centered.T @ centered / (len(data) - 1))
        order = np.argsort(values)[::-1][:dims]
        components, variance = vectors[:, order].T, values[order]
    else:
        _, singular, rows = np.linalg.svd(centered, full_matrices=False)
        components, variance = rows[:dims], singular[:dims] ** 2 / (len(data) - 1)
    largest = np.abs(components).argmax(axis=1)
    signs = np.sign(components[np.arange(dims), largest])
    signs[signs == 0] = 1.0
    return Projection(mean, components * signs[:, None], np.maximum(variance, 0.0))


class FixedEncoder:
    """Time-bin averages of a front end, then the projection when there is one."""

    kind = "fixed"

    def __init__(self, frontend: Frontend, time_bins: int, projection: Projection | None = None):
        self.frontend = frontend
        self.time_bins = time_bins
        self.projection = projection

    @property
    def feature_dims(self) -> int:
        return self.time_bins * self.frontend.channels

    @property
    def dims(self) -> int:
        return self.projection.dims if self.projection is not None else self.feature_dims

    def features(self, frames: np.ndarray) -> np.ndarray:
        """The time-bin averages of a clip's frames, before the projection."""
        return time_bin_means(frames, self.time_bins)

    def project(self, features: np.ndarray) -> np.ndarray:
        if self.projection is None:
            return np.asarray(features, dtype=np.float32)
        return self.projection.apply(features)

    def encode(self, clip: np.ndarray) -> np.ndarray:
        """The embedding of one clip."""
        return self.project(self.features(self.frontend.compute(clip)))
