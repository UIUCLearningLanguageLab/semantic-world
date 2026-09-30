"""The evaluation of sound embeddings. Every embedding is evaluated in the same way.

- **Same-different average precision.** Over pairs of tokens, rank the pairs by cosine distance,
  and compute the average precision for detecting pairs of the same word. It is reported for
  three sets of pairs: within a speaker, across two training speakers, and across two speakers
  of whom at least one is held out. Chance is the proportion of same-word pairs in the set.
- **Phonological fidelity.** The correlation, across pairs of words, between the cosine distance
  of the word embeddings and the phoneme edit distance.
- **Layer sweep.** For a pretrained model, every layer is evaluated, so that the default layer
  can be chosen from evidence.

A run with many tokens has too many pairs, so the evaluation uses a seeded sample of tokens
(from the ``wordforms:eval`` stream), the same sample for every embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semantic_world.wordforms.embeddings import EmbeddingStore, token_layout, word_means
from semantic_world.wordforms.english import edit_distance
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth import Synthesis

MAX_TOKENS = 5000
"""The evaluation samples this many tokens when a run has more."""
CONDITIONS = ("within_speaker", "across_train", "held_out")
EVAL_COLUMNS = (
    "embedding",
    "encoder",
    "source",
    "layer",
    "configured",
    "dims",
    "ap_within_speaker",
    "chance_within_speaker",
    "ap_across_train",
    "chance_across_train",
    "ap_held_out",
    "chance_held_out",
    "fidelity_pearson",
    "fidelity_spearman",
    "tokens_evaluated",
)


def average_precision(distances: np.ndarray, same: np.ndarray) -> float:
    """The average precision of ranking pairs by distance, smallest first, for finding the pairs
    marked ``same``. NaN when there are no such pairs."""
    positives = int(same.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(distances, kind="stable")
    hits = same[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / positives)


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """The matrix of cosine distances between the rows."""
    data = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    unit = data / np.where(norms > 0, norms, 1.0)
    return 1.0 - unit @ unit.T


class PairSets:
    """The pairs of an evaluation: which pairs of tokens are of the same word, and which pairs
    belong to each condition. Built once and used for every embedding."""

    def __init__(self, words: np.ndarray, speakers: np.ndarray, train: np.ndarray) -> None:
        count = len(words)
        upper = np.triu(np.ones((count, count), dtype=bool), 1)
        same_speaker = speakers[:, None] == speakers[None, :]
        both_train = train[:, None] & train[None, :]
        self.same_word = (words[:, None] == words[None, :])[upper]
        self.upper = upper
        self.masks = {
            "within_speaker": same_speaker[upper],
            "across_train": (~same_speaker & both_train)[upper],
            "held_out": (~same_speaker & ~both_train)[upper],
        }

    def evaluate(self, tokens: np.ndarray) -> dict[str, float]:
        distances = cosine_distances(tokens)[self.upper]
        result: dict[str, float] = {}
        for name, mask in self.masks.items():
            same = self.same_word[mask]
            result[f"ap_{name}"] = average_precision(distances[mask], same)
            result[f"chance_{name}"] = (
                float(same.mean()) if same.size and same.any() else float("nan")
            )
        return result


def edit_distances(words: list[WordForm]) -> np.ndarray:
    """The phoneme edit distance of every pair of words, in the order of ``np.triu_indices``."""
    stripped = [word.stripped for word in words]
    rows, columns = np.triu_indices(len(words), 1)
    return np.array(
        [edit_distance(stripped[i], stripped[j]) for i, j in zip(rows, columns, strict=True)],
        dtype=np.float64,
    )


def phonological_fidelity(types: np.ndarray, distances: np.ndarray) -> tuple[float, float]:
    """The Pearson and Spearman correlations between the cosine distance of word embeddings and
    the phoneme edit distance, across pairs of words."""
    from scipy.stats import rankdata

    rows, columns = np.triu_indices(len(types), 1)
    embedding = cosine_distances(types)[rows, columns]
    if len(embedding) < 3 or embedding.std() == 0 or distances.std() == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(embedding, distances)[0, 1])
    spearman = float(np.corrcoef(rankdata(embedding), rankdata(distances))[0, 1])
    return pearson, spearman


def evaluation_sample(count: int, rng: np.random.Generator, limit: int = MAX_TOKENS) -> np.ndarray:
    """The tokens that the evaluation uses: all of them, or a seeded sample of ``limit``."""
    if count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, size=limit, replace=False))


def evaluate_embeddings(
    stores: dict[str, EmbeddingStore],
    words: list[WordForm],
    synthesis: Synthesis,
    rng: np.random.Generator,
    max_tokens: int = MAX_TOKENS,
) -> pl.DataFrame:
    """The evaluation table: one row per embedding, and one row per layer of each pretrained
    model. ``configured`` marks the layer that the embedding stores."""
    token_words, token_speakers, train = token_layout(words, synthesis)
    sample = evaluation_sample(len(token_words), rng, max_tokens)
    pairs = PairSets(token_words[sample], token_speakers[sample], train[sample])
    phonemes = edit_distances(words)
    rows: list[dict[str, Any]] = []

    def row(store: EmbeddingStore, tokens, types, layer, configured) -> dict[str, Any]:
        meta = store.meta
        pearson, spearman = phonological_fidelity(types, phonemes)
        return {
            "embedding": store.name,
            "encoder": meta["encoder"],
            "source": meta.get("model") or meta.get("frontend"),
            "layer": layer,
            "configured": configured,
            "dims": int(tokens.shape[1]),
            **pairs.evaluate(tokens[sample]),
            "fidelity_pearson": pearson,
            "fidelity_spearman": spearman,
            "tokens_evaluated": len(sample),
        }

    for store in stores.values():
        layers = store.layers
        if layers is None:
            rows.append(row(store, store.tokens, store.types, None, True))
            continue
        for layer in range(layers.shape[1]):
            tokens = np.asarray(layers[:, layer])
            types = word_means(tokens, token_words, train, len(words))
            rows.append(row(store, tokens, types, layer, layer == store.meta["layer"]))
    schema = {
        name: pl.Float64 for name in EVAL_COLUMNS if name.startswith(("ap_", "chance_", "fid"))
    }
    schema.update(
        {
            "embedding": pl.String,
            "encoder": pl.String,
            "source": pl.String,
            "layer": pl.Int64,
            "configured": pl.Boolean,
            "dims": pl.Int64,
            "tokens_evaluated": pl.Int64,
        }
    )
    frame = pl.DataFrame(rows, schema={name: schema[name] for name in EVAL_COLUMNS})
    return frame.with_columns(pl.col(pl.Float64).fill_nan(None))


def write_evaluation(table: pl.DataFrame, folder: str | Path) -> Path:
    path = Path(folder) / "eval" / "embeddings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path, float_precision=6)
    return path
