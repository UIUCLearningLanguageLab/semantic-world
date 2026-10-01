"""The evaluation of sound embeddings. Every embedding is evaluated in the same way.

- **Same-different average precision.** Over pairs of tokens, rank the pairs by cosine distance,
  and compute the average precision for detecting pairs of the same word. It is reported for
  three sets of pairs: within a speaker, across two training speakers, and across two speakers
  of whom at least one is held out. Chance is the proportion of same-word pairs in the set.
- **Phonological fidelity**, on the word embeddings. The first measure is the correlation,
  across pairs of words, between cosine distance and phoneme edit distance. The second is the
  neighbor AUC: for a word, the probability that a word at edit distance 1 is closer in embedding
  space than a word at edit distance 3 or more, averaged over the words that have a neighbor at
  distance 1. That measure is near its ceiling for every embedding, so a harder version stands
  beside it: distance 1 against distance 2 exactly, averaged over the words that have neighbors
  at both distances.
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

A run with closed-class forms reports every measure for each kind of form separately (``kind``
is ``content``, ``function``, or ``inflected``) and for all forms together (``all``). Each kind
has its own sample of tokens, and so do all forms together. The sample of the content words is
drawn first, so it is the sample of the same run without closed-class forms. A run with content
words only has the ``content`` rows alone.

- **Stem AUC**, for inflected forms, on the word embeddings: the probability that an inflected
  form is closer to its own stem than to another stem, averaged over the inflected forms. The
  other stems are the stems of the other inflected forms. 0.5 is chance. The measure is given in
  the ``inflected`` rows and in the ``all`` rows.

When some words are held out from the trained encoders, every measure is reported for all words,
for the training words, and for the held-out words (``word_split``: ``all``, ``train``,
``held_out``), on the tokens of those words within each sample. The held-out rows are the fair
test of an encoder on novel words.

Every stored embedding is evaluated twice (``talker_normalized``): as the encoder gives it, and
with each speaker's mean token embedding (over the speaker's clean tokens of training content
words)
subtracted. ``configured`` marks the variant that the run stores. The layer sweep is given
without normalization.

A run with augmented tokens reports every measure separately for the clean (synthesized) tokens,
the augmented tokens, and both together (``tokens``: ``clean``, ``augmented``, ``all``), each
with its own sample, and once more for each recipe (``recipe:<name>``) on the widest kind. A run
without augmentation has the ``clean`` rows alone.

- **Robustness**: how well a token retrieves its own word's clean embedding. Pairs of a sampled
  token and a word embedding (the mean of the word's clean training tokens) are ranked by cosine
  distance, and the average precision for the pairs of a token with its own word is reported
  (``robustness_ap``; chance is one over the number of words), with the share of tokens whose
  nearest word embedding is their own (``robustness_top1``). For clean tokens the measure is a
  baseline, since a training token is part of its own word's mean.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semantic_world.wordforms.config import Config
from semantic_world.wordforms.embeddings import (
    EmbeddingStore,
    normalization_basis,
    speaker_means,
    token_layout,
    word_means,
)
from semantic_world.wordforms.english import edit_distance
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth import Synthesis

MAX_TOKENS = 5000
"""The evaluation samples this many tokens when a run has more."""
NEAR_DISTANCE = 1
FAR_DISTANCE = 3
"""The neighbor AUC compares words at edit distance 1 with words at distance 3 or more."""
HARD_FAR_DISTANCE = 2
"""The harder neighbor AUC compares words at edit distance 1 with words at distance 2."""
CONDITIONS = ("within_speaker", "across_train", "held_out")
WORD_SETS = ("all", "without_long_synthesis")
KINDS = ("content", "function", "inflected")
ALL_KINDS = "all"
EVAL_COLUMNS = (
    "embedding",
    "encoder",
    "source",
    "layer",
    "configured",
    "basis",
    "word_set",
    "word_split",
    "kind",
    "tokens",
    "talker_normalized",
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
    "fidelity_auc_1v2",
    "auc_1v2_words",
    "stem_auc",
    "stem_auc_forms",
    "robustness_ap",
    "robustness_top1",
    "robustness_chance",
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


def neighbor_auc(
    types: np.ndarray, phonemes: np.ndarray, far: int = FAR_DISTANCE, far_exact: bool = False
) -> tuple[float, int]:
    """The neighbor AUC, and the number of words it is averaged over.

    For one word, the AUC is the probability that a word at phoneme edit distance 1 is closer in
    embedding space (cosine distance) than a far word; a tie counts as one half. A far word is
    at edit distance ``far`` or more, or at exactly ``far`` with ``far_exact``. The result is the
    mean over the words that have at least one word at distance 1 and at least one far word. NaN
    when there is no such word. 0.5 is chance.
    """
    embedding = cosine_distances(types)
    values = []
    for i in range(len(types)):
        near = embedding[i, phonemes[i] == NEAR_DISTANCE]
        distant = phonemes[i] == far if far_exact else phonemes[i] >= far
        others = embedding[i, distant]
        if near.size == 0 or others.size == 0:
            continue
        closer = (near[:, None] < others[None, :]).mean()
        tied = (near[:, None] == others[None, :]).mean()
        values.append(closer + 0.5 * tied)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def robustness(tokens: np.ndarray, token_words: np.ndarray, types: np.ndarray, present):
    """How well tokens retrieve their own word's embedding among the ``present`` words: the
    average precision of the pairs of a token with its own word, over all token-word pairs
    ranked by cosine distance; the share of tokens whose nearest word is their own; and chance
    (one over the number of words). NaN when a token's word is absent or nothing is present."""
    words = np.flatnonzero(present)
    keep = present[token_words]
    if len(words) < 2 or not keep.any():
        return float("nan"), float("nan"), float("nan")
    vectors = np.asarray(tokens[keep], dtype=np.float64)
    means = np.asarray(types[words], dtype=np.float64)
    unit_tokens = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    unit_types = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - unit_tokens @ unit_types.T
    same = token_words[keep][:, None] == words[None, :]
    ap = average_precision(distances.ravel(), same.ravel())
    top1 = float(same[np.arange(len(distances)), np.argmin(distances, axis=1)].mean())
    return ap, top1, 1.0 / len(words)


def stem_auc(forms: np.ndarray, stems: np.ndarray, own: np.ndarray) -> tuple[float, int]:
    """The stem AUC, and the number of inflected forms it is averaged over.

    ``forms`` holds the embeddings of inflected forms, ``stems`` the embeddings of the distinct
    stems, and ``own[i]`` the row of form ``i``'s stem. For one form, the AUC is the probability
    that the form is closer (cosine distance) to its own stem than to another stem; a tie counts
    as one half. NaN when there are no forms or only one stem. 0.5 is chance.
    """
    if len(forms) == 0 or len(stems) < 2:
        return float("nan"), 0
    count = len(forms)
    distances = cosine_distances(np.concatenate([forms, stems]))[:count, count:]
    rows = np.arange(count)
    mine = distances[rows, own][:, None]
    other = np.ones_like(distances, dtype=bool)
    other[rows, own] = False
    closer = ((mine < distances) & other).sum(axis=1)
    tied = ((mine == distances) & other).sum(axis=1)
    return float(((closer + 0.5 * tied) / (len(stems) - 1)).mean()), count


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
    word_kinds = np.array([word.kind for word in words])
    clean = np.array([not t.augmentation for t in synthesis.tokens], dtype=bool)
    recipe_of = np.array(
        [json.loads(t.augmentation)["recipe"] if t.augmentation else "" for t in synthesis.tokens]
    )
    # The kinds of words that get rows: each kind, and all forms together when the run has
    # closed-class forms. The token sets: the clean tokens, and with augmentation also the
    # augmented tokens, all tokens, and each recipe's tokens (on the widest kind). Each group,
    # a kind with a token set, has its own sample of tokens, the clean content words' first.
    kinds = {kind: word_kinds == kind for kind in KINDS if (word_kinds == kind).any()}
    if set(kinds) != {"content"}:
        kinds[ALL_KINDS] = np.ones(len(words), dtype=bool)
    widest = ALL_KINDS if ALL_KINDS in kinds else "content"
    token_sets: dict[str, np.ndarray] = {"clean": clean}
    if not clean.all():
        token_sets["augmented"] = ~clean
        token_sets["all"] = np.ones(len(clean), dtype=bool)
        for name in dict.fromkeys(recipe_of[~clean]):
            token_sets[f"recipe:{name}"] = recipe_of == name
    groups: list[tuple[str, str]] = [
        (kind, tokens)
        for kind in kinds
        for tokens in token_sets
        if not tokens.startswith("recipe:") or kind == widest
    ]
    samples: dict[tuple[str, str], np.ndarray] = {}
    for kind, tokens in groups:
        own = np.flatnonzero(kinds[kind][token_words] & token_sets[tokens])
        samples[kind, tokens] = own[evaluation_sample(len(own), rng, max_tokens)]
    phonemes = edit_distance_matrix(words)
    flagged = np.array([bool(word.long_synthesis) for word in words])
    index = {word.label: i for i, word in enumerate(words)}
    stem_of = np.array([index.get(word.stem, -1) if word.stem else -1 for word in words])
    held_out_words = np.array([word.held_out for word in words], dtype=bool)
    splits: dict[str, np.ndarray] = {"all": np.ones(len(words), dtype=bool)}
    if held_out_words.any():
        splits["train"] = ~held_out_words
        splits["held_out"] = held_out_words
    # The sets of words that get a row within each kind: every split of the words, and all
    # words without the long_synthesis ones when the kind has any.
    word_sets: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for kind, members in kinds.items():
        entries = []
        for split, chosen in splits.items():
            if not (members & chosen).any():
                continue
            entries.append((split, "all", members & chosen))
            if split == "all" and (flagged & members).any():
                entries.append((split, "without_long_synthesis", members & ~flagged))
        word_sets[kind] = entries
    in_sample: dict[tuple[str, str], list[np.ndarray]] = {}
    pairs: dict[tuple[str, str], list[PairSets]] = {}
    for group in groups:
        sample = samples[group]
        in_sample[group] = [keep[token_words[sample]] for _, _, keep in word_sets[group[0]]]
        pairs[group] = [
            PairSets(token_words[sample][mask], token_speakers[sample][mask], train[sample][mask])
            for mask in in_sample[group]
        ]
    rows: list[dict[str, Any]] = []

    def stem_measure(kind, name, keep, types, present, stem_types, stem_present):
        """The stem AUC of a kind's inflected forms. A form counts when the form and its stem
        both have a word embedding, and neither is left out of the word set."""
        if kind not in ("inflected", ALL_KINDS):
            return float("nan"), 0
        stem_kept = stem_present & ~flagged if name == "without_long_synthesis" else stem_present
        forms = np.flatnonzero((word_kinds == "inflected") & keep & present & (stem_of >= 0))
        forms = forms[stem_kept[stem_of[forms]]]
        stems, own = np.unique(stem_of[forms], return_inverse=True)
        return stem_auc(types[forms], stem_types[stems], own)

    def add_rows(
        store,
        group,
        sampled,
        types,
        present,
        stems,
        reference,
        layer,
        configured,
        basis,
        normalized,
    ):
        """One row for each word set of a group. ``sampled`` holds the token embeddings of the
        group's sample, ``present`` marks the words that have a word embedding in ``types``,
        ``stems`` gives the word embeddings of the stems with their own ``present``, and
        ``reference`` the clean word embeddings that the robustness measure retrieves."""
        meta = store.meta
        kind, tokens = group
        for k, (split, name, keep) in enumerate(word_sets[kind]):
            chosen = keep & present
            distances = phonemes[np.ix_(chosen, chosen)]
            pearson, spearman = phonological_fidelity(types[chosen], distances)
            auc, auc_words = neighbor_auc(types[chosen], distances)
            hard, hard_words = neighbor_auc(types[chosen], distances, HARD_FAR_DISTANCE, True)
            stem, stem_forms = stem_measure(kind, name, keep, types, present, *stems)
            inflected = kind in ("inflected", ALL_KINDS)
            mask = in_sample[group][k]
            robust_ap, robust_top1, robust_chance = robustness(
                sampled[mask], token_words[samples[group]][mask], reference[0], keep & reference[1]
            )
            rows.append(
                {
                    "embedding": store.name,
                    "encoder": meta["encoder"],
                    "source": meta.get("model") or meta.get("frontend"),
                    "layer": layer,
                    "configured": configured,
                    "basis": basis,
                    "word_set": name,
                    "word_split": split,
                    "kind": kind,
                    "tokens": tokens,
                    "talker_normalized": normalized,
                    "dims": int(sampled.shape[1]),
                    **pairs[group][k].evaluate(sampled[mask]),
                    "fidelity_pearson": pearson,
                    "fidelity_spearman": spearman,
                    "fidelity_auc": auc,
                    "auc_words": auc_words,
                    "fidelity_auc_1v2": hard,
                    "auc_1v2_words": hard_words,
                    "stem_auc": stem,
                    "stem_auc_forms": stem_forms if inflected else None,
                    "robustness_ap": robust_ap,
                    "robustness_top1": robust_top1,
                    "robustness_chance": robust_chance,
                    "tokens_evaluated": int(mask.sum()),
                    "words_evaluated": int(chosen.sum()),
                }
            )

    everything = np.ones(len(words), dtype=bool)
    use_all = config is not None and config.word_embeddings.tokens == "all"
    word_tokens = train if use_all else train & clean
    basis = normalization_basis(words, synthesis, token_words)
    for store in stores.values():
        pretrained = bool(store.meta["pretrained"])
        layer = store.meta["layer"] if pretrained else None
        stored_means = store.speaker_means
        raw = store.tokens if stored_means is None else store.tokens + stored_means[token_speakers]
        for normalized in (False, True):
            tokens = raw
            if normalized:
                means = speaker_means(raw, token_speakers, basis, len(synthesis.speakers))
                tokens = raw - means[token_speakers]
            types = word_means(tokens, token_words, word_tokens, len(words))
            # the robustness reference: each word's clean training tokens, whatever the run's
            # word embeddings are made of
            reference = (word_means(tokens, token_words, train & clean, len(words)), everything)
            configured = normalized == bool(store.meta.get("talker_normalization", False))
            for group in groups:
                add_rows(
                    store,
                    group,
                    tokens[samples[group]],
                    types,
                    everything,
                    (types, everything),
                    reference,
                    layer,
                    configured,
                    "stored",
                    normalized,
                )
    union = np.unique(np.concatenate(list(samples.values())))
    for store in stores.values():
        if not (sweep and store.meta["pretrained"]):
            continue
        if store.layers is None and config is None:
            raise ValueError("the layer sweep needs the run's configuration to run the model")
        name = store.name
        report = None if progress is None else (lambda i, n, name=name: progress(name, i, n))
        # the model runs once on every sampled token, whichever samples the token is in
        layers = sweep_layers(store, config, synthesis, union, local_only, report)
        for layer in range(layers.shape[1]):
            means = {}
            for group, sample in samples.items():
                sampled = layers[np.searchsorted(union, sample), layer]
                # a sweep row's word embeddings are the means over the sample's training tokens
                means[group] = (
                    sampled,
                    *sample_word_means(sampled, token_words[sample], train[sample], len(words)),
                )
            for group in groups:
                kind, tokens = group
                sampled, types, present = means[group]
                # an inflected form's stem is a content word, from the content words' sample;
                # the robustness reference is the kind's clean sample
                stems = means["content", tokens][1:] if kind == "inflected" else (types, present)
                reference = means[kind, "clean"][1:]
                configured = layer == store.meta["layer"]
                add_rows(
                    store,
                    group,
                    sampled,
                    types,
                    present,
                    stems,
                    reference,
                    layer,
                    configured,
                    "sweep",
                    False,
                )

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
            "word_split": pl.String,
            "kind": pl.String,
            "tokens": pl.String,
            "talker_normalized": pl.Boolean,
            "dims": pl.Int64,
            "auc_words": pl.Int64,
            "auc_1v2_words": pl.Int64,
            "stem_auc": pl.Float64,
            "stem_auc_forms": pl.Int64,
            "robustness_ap": pl.Float64,
            "robustness_top1": pl.Float64,
            "robustness_chance": pl.Float64,
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
