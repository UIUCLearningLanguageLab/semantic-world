"""The evaluation of sound embeddings. Every embedding is evaluated in the same way.

- **Same-different average precision.** Over pairs of tokens, rank the pairs by cosine distance,
  and compute the average precision for detecting pairs of the same word. It is reported for
  three sets of pairs: within a speaker, across two training speakers, and across two speakers
  of whom at least one is held out. Chance is the proportion of same-word pairs in the set.
- **Phonological fidelity**, two measures on the word embeddings. The first is the correlation,
  across pairs of words, between cosine distance and phoneme edit distance. The second is the
  neighbor AUC: for a word, the probability that a word at edit distance 1 is closer in embedding
  space than a word at edit distance 3 or more, averaged over the words that have a neighbor at
  distance 1.
- **Layer sweep.** For a pretrained model, every layer is evaluated, so that the default layer
  can be chosen from evidence.

A run with many tokens has too many pairs, so the evaluation uses a seeded sample of tokens
(5,000, from the ``wordforms:eval`` stream), the same sample for every embedding.

The table has a row for each stored embedding (``basis`` is ``stored``): the token measures use
the sample, and the word measures use the stored word embeddings. The layer sweep adds a row for
each layer of each pretrained model (``basis`` is ``sweep``). The sweep runs the model on the
sample only, so no run has to store every layer; a sweep row's word embeddings are the means
over the sample's training-speaker tokens.

When some words are flagged ``long_synthesis``, every row is given twice: for all words
(``word_set`` is ``all``) and without the flagged words (``without_long_synthesis``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semantic_world.wordforms.config import Config
from semantic_world.wordforms.embeddings import EmbeddingStore, token_layout
from semantic_world.wordforms.english import edit_distance
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth import Synthesis

MAX_TOKENS = 5000
"""The evaluation samples this many tokens when a run has more."""
NEAR_DISTANCE = 1
FAR_DISTANCE = 3
"""The neighbor AUC compares words at edit distance 1 with words at distance 3 or more."""
CONDITIONS = ("within_speaker", "across_train", "held_out")
WORD_SETS = ("all", "without_long_synthesis")
EVAL_COLUMNS = (
    "embedding",
    "encoder",
    "source",
    "layer",
    "configured",
    "basis",
    "word_set",
    "dims",
    "ap_within_speaker",
    "chance_within_speaker",
    "ap_across_train",
    "chance_across_train",
    "ap_held_out",
    "chance_held_out",
    "fidelity_pearson",
    "fidelity_spearman",
    "fidelity_auc",
    "auc_words",
    "tokens_evaluated",
    "words_evaluated",
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
        self.count = count
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
            defined = same.size > 0 and same.any()
            result[f"ap_{name}"] = average_precision(distances[mask], same)
            result[f"chance_{name}"] = float(same.mean()) if defined else float("nan")
        return result


def edit_distance_matrix(words: list[WordForm]) -> np.ndarray:
    """The phoneme edit distance between every two words, as a square matrix."""
    stripped = [word.stripped for word in words]
    matrix = np.zeros((len(words), len(words)), dtype=np.float64)
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            matrix[i, j] = matrix[j, i] = edit_distance(stripped[i], stripped[j])
    return matrix


def phonological_fidelity(types: np.ndarray, phonemes: np.ndarray) -> tuple[float, float]:
    """The Pearson and Spearman correlations between the cosine distance of word embeddings and
    the phoneme edit distance (a square matrix), across pairs of words."""
    from scipy.stats import rankdata

    rows, columns = np.triu_indices(len(types), 1)
    embedding = cosine_distances(types)[rows, columns]
    distances = phonemes[rows, columns]
    if len(embedding) < 3 or embedding.std() == 0 or distances.std() == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(embedding, distances)[0, 1])
    spearman = float(np.corrcoef(rankdata(embedding), rankdata(distances))[0, 1])
    return pearson, spearman


def neighbor_auc(types: np.ndarray, phonemes: np.ndarray) -> tuple[float, int]:
    """The neighbor AUC, and the number of words it is averaged over.

    For one word, the AUC is the probability that a word at phoneme edit distance 1 is closer in
    embedding space (cosine distance) than a word at edit distance 3 or more; a tie counts as
    one half. The result is the mean over the words that have at least one word at distance 1
    and at least one at distance 3 or more. NaN when there is no such word. 0.5 is chance.
    """
    embedding = cosine_distances(types)
    values = []
    for i in range(len(types)):
        near = embedding[i, phonemes[i] == NEAR_DISTANCE]
        far = embedding[i, phonemes[i] >= FAR_DISTANCE]
        if near.size == 0 or far.size == 0:
            continue
        closer = (near[:, None] < far[None, :]).mean()
        tied = (near[:, None] == far[None, :]).mean()
        values.append(closer + 0.5 * tied)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def evaluation_sample(count: int, rng: np.random.Generator, limit: int = MAX_TOKENS) -> np.ndarray:
    """The tokens that the evaluation uses: all of them, or a seeded sample of ``limit``."""
    if count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, size=limit, replace=False))


def sample_word_means(tokens: np.ndarray, words: np.ndarray, train: np.ndarray, count: int):
    """Word embeddings from a sample of tokens: the mean of each word's training-speaker tokens
    in the sample, and which words have such a token."""
    sums = np.zeros((count, tokens.shape[1]), dtype=np.float64)
    totals = np.zeros(count, dtype=np.int64)
    np.add.at(sums, words[train], np.asarray(tokens, dtype=np.float64)[train])
    np.add.at(totals, words[train], 1)
    present = totals > 0
    sums[present] /= totals[present][:, None]
    return sums, present


def sweep_layers(
    store: EmbeddingStore,
    config: Config,
    synthesis: Synthesis,
    sample: np.ndarray,
    local_only: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """The pooled output of every layer of a pretrained model for the sampled tokens: sample by
    layers by dimensions. It comes from the stored layers when the run kept them, and otherwise
    from running the model on the sample."""
    stored = store.layers
    if stored is not None:
        return np.asarray(stored[sample])
    from semantic_world.wordforms.encoders.pretrained import PretrainedEncoder

    settings = store.meta["settings"]
    encoder = PretrainedEncoder(
        settings["model"],
        settings["layer"],
        settings["pooling"],
        config.device,
        config.synthesis.sample_rate,
        local_only,
    )
    layers = np.zeros((len(sample), encoder.layers, encoder.dims), dtype=np.float32)
    for k, index in enumerate(sample):
        layers[k] = encoder.all_layers(synthesis.audio(synthesis.tokens[int(index)]))
        if progress is not None:
            progress(k + 1, len(sample))
    return layers


def evaluate_embeddings(
    stores: dict[str, EmbeddingStore],
    words: list[WordForm],
    synthesis: Synthesis,
    rng: np.random.Generator,
    *,
    config: Config | None = None,
    max_tokens: int = MAX_TOKENS,
    sweep: bool = True,
    local_only: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> pl.DataFrame:
    """The evaluation table. ``config`` is needed for the layer sweep of a pretrained model that
    did not store its layers; ``sweep=False`` leaves the sweep out."""
    token_words, token_speakers, train = token_layout(words, synthesis)
    sample = evaluation_sample(len(token_words), rng, max_tokens)
    phonemes = edit_distance_matrix(words)
    flagged = np.array([bool(word.long_synthesis) for word in words])
    word_sets = {"all": np.ones(len(words), dtype=bool)}
    if flagged.any():
        word_sets["without_long_synthesis"] = ~flagged
    in_sample = {name: keep[token_words[sample]] for name, keep in word_sets.items()}
    pairs = {
        name: PairSets(token_words[sample][mask], token_speakers[sample][mask], train[sample][mask])
        for name, mask in in_sample.items()
    }
    rows: list[dict[str, Any]] = []

    def add_rows(store, sampled, types, present, layer, configured, basis) -> None:
        """One row for each word set. ``sampled`` holds the sample's token embeddings, and
        ``present`` marks the words that have a word embedding."""
        meta = store.meta
        for name, keep in word_sets.items():
            chosen = keep & present
            distances = phonemes[np.ix_(chosen, chosen)]
            pearson, spearman = phonological_fidelity(types[chosen], distances)
            auc, auc_words = neighbor_auc(types[chosen], distances)
            rows.append(
                {
                    "embedding": store.name,
                    "encoder": meta["encoder"],
                    "source": meta.get("model") or meta.get("frontend"),
                    "layer": layer,
                    "configured": configured,
                    "basis": basis,
                    "word_set": name,
                    "dims": int(sampled.shape[1]),
                    **pairs[name].evaluate(sampled[in_sample[name]]),
                    "fidelity_pearson": pearson,
                    "fidelity_spearman": spearman,
                    "fidelity_auc": auc,
                    "auc_words": auc_words,
                    "tokens_evaluated": int(in_sample[name].sum()),
                    "words_evaluated": int(chosen.sum()),
                }
            )

    everything = np.ones(len(words), dtype=bool)
    for store in stores.values():
        pretrained = bool(store.meta["pretrained"])
        layer = store.meta["layer"] if pretrained else None
        add_rows(store, store.tokens[sample], store.types, everything, layer, True, "stored")
    for store in stores.values():
        if not (sweep and store.meta["pretrained"]):
            continue
        if store.layers is None and config is None:
            raise ValueError("the layer sweep needs the run's configuration to run the model")
        name = store.name
        report = None if progress is None else (lambda i, n, name=name: progress(name, i, n))
        layers = sweep_layers(store, config, synthesis, sample, local_only, report)
        for layer in range(layers.shape[1]):
            sampled = layers[:, layer]
            types, present = sample_word_means(
                sampled, token_words[sample], train[sample], len(words)
            )
            add_rows(store, sampled, types, present, layer, layer == store.meta["layer"], "sweep")

    floats = ("ap_", "chance_", "fidelity_")
    schema: dict[str, Any] = {name: pl.Float64 for name in EVAL_COLUMNS if name.startswith(floats)}
    schema.update(
        {
            "embedding": pl.String,
            "encoder": pl.String,
            "source": pl.String,
            "layer": pl.Int64,
            "configured": pl.Boolean,
            "basis": pl.String,
            "word_set": pl.String,
            "dims": pl.Int64,
            "auc_words": pl.Int64,
            "tokens_evaluated": pl.Int64,
            "words_evaluated": pl.Int64,
        }
    )
    frame = pl.DataFrame(rows, schema={name: schema[name] for name in EVAL_COLUMNS})
    return frame.with_columns(pl.col(pl.Float64).fill_nan(None))


def write_evaluation(table: pl.DataFrame, folder: str | Path) -> Path:
    path = Path(folder) / "eval" / "embeddings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path, float_precision=6)
    return path
