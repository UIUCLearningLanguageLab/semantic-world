"""Layer 4: encoders, word embeddings, the evaluation, the assignment, and SoundEmbeddings."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_synth import ToneEngine, espeak_only
from wordforms_support import (
    DATA,
    FIXTURES,
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_hubert,
    needs_piper,
    needs_torch,
    needs_whisper,
    needs_wordfreq,
    plain_rows,
)

from semantic_world.wordforms import Run, run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.assign import (
    AssignmentError,
    assign_arbitrary,
    hamming_distances,
    load_meanings,
)
from semantic_world.wordforms.config import parse_config
from semantic_world.wordforms.embeddings import (
    EmbeddingStore,
    SoundEmbeddings,
    compute_embeddings,
    token_layout,
    word_means,
)
from semantic_world.wordforms.encoders.fixed import (
    FixedEncoder,
    Projection,
    bin_weights,
    fit_pca,
    time_bin_means,
)
from semantic_world.wordforms.evaluate import (
    EVAL_COLUMNS,
    PairSets,
    average_precision,
    cosine_distances,
    edit_distance_matrix,
    evaluate_embeddings,
    evaluation_sample,
    neighbor_auc,
    phonological_fidelity,
    sample_word_means,
)
from semantic_world.wordforms.frontends import compute_frontends, make_frontends
from semantic_world.wordforms.synth import synthesize_lexicon

RATE = 16000


class WordEngine(ToneEngine):
    """A stand-in engine whose clips depend on the word: a chord of three tones chosen by the
    phonemes, a little detuned by the voice, with its level rising or falling by the word."""

    def synthesize(self, phonemes, settings):
        self.calls += 1
        word = hashlib.sha256(phonemes.encode()).digest()
        voice = hashlib.sha256(settings["voice"].encode()).digest()
        n = int(22050 * 0.3 * settings["scale"])
        t = np.arange(n) / 22050
        detune = 1.0 + (voice[0] - 128) / 4000.0
        clip = sum(
            np.sin(2 * np.pi * (200 + 12 * word[k]) * detune * t) for k in range(3)
        ) * np.linspace(0.3 + word[3] / 400, 0.3 + word[4] / 400, n)
        silence = np.zeros(2205)
        return np.concatenate([silence, 0.2 * clip, silence]).astype(np.float32), 22050


def stand_in_run(tmp_path, embeddings=None, **wordforms) -> tuple[Run, WordEngine]:
    """A run with the espeak-ng speakers and the stand-in engine, through the front ends."""
    data = espeak_only(tmp_path, **wordforms).resolved()
    data["embeddings"] = (
        embeddings
        if embeddings is not None
        else [
            {
                "name": "cochleagram_fixed",
                "encoder": "fixed",
                "frontend": "cochleagram",
                "pca_dims": 8,
            },
            {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel", "pca_dims": 8},
        ]
    )
    config = parse_config(data, "stand_in")
    run = run_forms(config)
    engine = WordEngine()
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": engine}, check=False
    )
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / "run")
    return run, engine


def embed_run(tmp_path, embeddings=None, **wordforms):
    run, engine = stand_in_run(tmp_path, embeddings, **wordforms)
    run.embeddings = compute_embeddings(
        run.config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run"
    )
    run.write(tmp_path / "run")
    return run, engine


pytestmark = [needs_cmudict, needs_wordfreq, needs_audio]


# ---------------------------------------------------------------------------------------------
# The fixed encoder
# ---------------------------------------------------------------------------------------------


def test_time_bins_divide_the_clip_equally():
    frames = np.arange(40, dtype=np.float32).reshape(20, 2)
    means = time_bin_means(frames, 10).reshape(10, 2)
    assert np.allclose(means[:, 0], frames[:, 0].reshape(10, 2).mean(axis=1))
    assert means.dtype == np.float32 and time_bin_means(frames, 10).shape == (20,)
    # a frame that straddles two bins counts in each by its share
    weights = bin_weights(3, 2)
    assert np.allclose(weights, [[2 / 3, 1 / 3, 0], [0, 1 / 3, 2 / 3]])
    for count in (1, 3, 7, 10, 23, 100):
        weights = bin_weights(count, 10)
        assert weights.shape == (10, count) and np.allclose(weights.sum(axis=1), 1.0)
        assert np.allclose(weights.sum(axis=0), 10 / count)  # every frame counts equally
    # fewer frames than bins still gives every bin a value
    short = time_bin_means(np.array([[1.0], [3.0]], dtype=np.float32), 4)
    assert np.allclose(short, [1, 1, 3, 3])
    assert np.allclose(time_bin_means(np.full((7, 3), 2.5, dtype=np.float32), 5), 2.5)
    with pytest.raises(ValueError, match="no frames"):
        time_bin_means(np.zeros((0, 3), dtype=np.float32), 5)


def test_pca_projection():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 6)) * np.array([5, 3, 2, 1, 0.5, 0.1]) + 10
    projection = fit_pca(data, 3)
    assert projection.dims == 3 and projection.components.shape == (3, 6)
    assert np.allclose(projection.components @ projection.components.T, np.eye(3), atol=1e-10)
    assert np.all(np.diff(projection.explained_variance) <= 0)
    projected = projection.apply(data)
    assert projected.dtype == np.float32 and projected.shape == (200, 3)
    assert np.allclose(projected.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(projected.var(axis=0, ddof=1), projection.explained_variance, rtol=1e-4)
    # each component's largest entry is positive, so the projection is the same everywhere
    largest = np.abs(projection.components).argmax(axis=1)
    assert (projection.components[np.arange(3), largest] > 0).all()
    # with fewer tokens than dimensions the result is the same subspace, by another route
    few = data[:4]
    wide = fit_pca(few, 10)
    assert wide.dims == 3  # at most one fewer than the number of tokens
    assert np.allclose(wide.apply(few).var(axis=0, ddof=1), wide.explained_variance, rtol=1e-4)
    assert np.allclose(
        np.abs(fit_pca(data.T[:5], 2).components @ fit_pca(data.T[:5], 2).components.T), np.eye(2)
    )


def test_projection_round_trips_through_a_file(tmp_path):
    projection = fit_pca(np.random.default_rng(1).normal(size=(50, 5)), 2)
    projection.save(tmp_path / "projection.npz")
    loaded = Projection.load(tmp_path / "projection.npz")
    x = np.random.default_rng(2).normal(size=(3, 5))
    assert np.array_equal(loaded.apply(x), projection.apply(x))


def test_fixed_encoder_on_a_clip():
    frontend = make_frontends(parse_config({}, "x"))["logmel"]
    encoder = FixedEncoder(frontend, 10)
    clip = (0.1 * np.sin(2 * np.pi * 440 * np.arange(9000) / RATE)).astype(np.float32)
    vector = encoder.encode(clip)
    assert vector.shape == (800,) == (encoder.dims,) and vector.dtype == np.float32
    assert np.array_equal(vector, time_bin_means(frontend.compute(clip), 10))
    projected = FixedEncoder(
        frontend, 10, fit_pca(np.random.default_rng(0).normal(size=(30, 800)), 4)
    )
    assert projected.dims == 4 and projected.encode(clip).shape == (4,)


# ---------------------------------------------------------------------------------------------
# Token and word embeddings
# ---------------------------------------------------------------------------------------------


def test_word_embeddings_are_means_over_training_speakers(tmp_path):
    run, _ = embed_run(tmp_path)
    words, synthesis = run.lexicon.words, run.synthesis
    token_words, token_speakers, train = token_layout(words, synthesis)
    assert len(token_words) == 36 and train.sum() == 24  # one of three speakers is held out
    held_out = [s.label for s in synthesis.speakers if s.held_out]
    assert [t.speaker in held_out for t in synthesis.tokens] == list(~train)
    for name, store in run.embeddings.items():
        folder = tmp_path / "run" / "embeddings" / name
        assert sorted(p.name for p in folder.iterdir()) == [
            "meta.yaml",
            "projection.npz",
            "tokens.npy",
            "types.npy",
        ]
        tokens, types = np.load(folder / "tokens.npy"), np.load(folder / "types.npy")
        assert tokens.shape == (36, 8) and types.shape == (6, 8)
        assert tokens.dtype == types.dtype == np.float32
        for w in range(6):
            rows = (token_words == w) & train
            assert np.allclose(types[w], tokens[rows].mean(axis=0), atol=1e-6)
        assert not np.allclose(types[0], tokens[token_words == 0].mean(axis=0), atol=1e-6)
        meta = yaml.safe_load((folder / "meta.yaml").read_text())
        assert meta["name"] == name and meta["encoder"] == "fixed" and meta["pretrained"] is False
        assert meta["dims"] == 8 and meta["tokens"] == 36 and meta["words"] == 6
        assert meta["projection"]["fitted_tokens"] == 24
        assert 0 < meta["projection"]["explained_variance"] <= 1
        assert store.summary() == {
            "encoder": "fixed",
            "pretrained": False,
            "dims": 8,
            "tokens": 36,
            "words": 6,
            "reused": False,
        }
    with pytest.raises(ValueError, match="no tokens from training speakers"):
        word_means(np.ones((2, 3)), np.array([0, 1]), np.array([True, False]), 2)


def test_the_projection_is_fitted_on_training_speakers_only(tmp_path):
    run, _ = embed_run(tmp_path)
    _, _, train = token_layout(run.lexicon.words, run.synthesis)
    store = run.embeddings["logmel_fixed"]
    frontend = make_frontends(run.config)["logmel"]
    features = np.stack(
        [time_bin_means(frontend.compute(run.synthesis.audio(t)), 10) for t in run.synthesis.tokens]
    )
    expected = fit_pca(features[train], 8)
    stored = Projection.load(store.folder / "projection.npz")
    assert np.allclose(stored.mean, features[train].mean(axis=0), atol=1e-5)
    assert np.allclose(stored.components, expected.components, atol=1e-6)
    assert np.allclose(store.tokens, expected.apply(features), atol=1e-4)


def test_embedding_without_a_projection_and_from_the_waveform(tmp_path):
    run, _ = embed_run(
        tmp_path,
        embeddings=[
            {"name": "raw", "encoder": "fixed", "frontend": "cochleagram", "time_bins": 4}
            | {"pca_dims": None},
            {"name": "wave", "encoder": "fixed", "frontend": "waveform", "time_bins": 5}
            | {"pca_dims": None},
        ],
    )
    raw = run.embeddings["raw"]
    assert raw.tokens.shape == (36, 4 * 64) and "projection" not in raw.meta
    assert not (raw.folder / "projection.npz").exists()
    frontend = make_frontends(run.config)["cochleagram"]
    token = run.synthesis.tokens[5]
    expected = time_bin_means(frontend.compute(run.synthesis.audio(token)), 4)
    assert np.array_equal(raw.tokens[5], expected)
    # the waveform front end is not stored, so its frames are computed from the audio cache
    assert run.embeddings["wave"].tokens.shape == (36, 5)


def test_a_stored_embedding_is_reused_until_something_changes(tmp_path):
    run, _ = embed_run(tmp_path)
    folder = tmp_path / "run"
    written = (folder / "embeddings" / "logmel_fixed" / "tokens.npy").stat().st_mtime_ns
    again = compute_embeddings(run.config, run.lexicon.words, run.synthesis, run.frontends, folder)
    assert all(store.reused for store in again.values())
    assert (folder / "embeddings" / "logmel_fixed" / "tokens.npy").stat().st_mtime_ns == written
    data = run.config.resolved()
    data["embeddings"][1]["time_bins"] = 5
    changed = compute_embeddings(
        parse_config(data, "x"), run.lexicon.words, run.synthesis, run.frontends, folder
    )
    assert changed["cochleagram_fixed"].reused and not changed["logmel_fixed"].reused
    assert (
        EmbeddingStore.load(folder / "embeddings" / "logmel_fixed").meta["settings"]["time_bins"]
        == 5
    )


# ---------------------------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------------------------


def test_average_precision():
    same = np.array([True, False, True, False])
    assert average_precision(np.array([0.1, 0.5, 0.2, 0.9]), same) == 1.0
    assert average_precision(np.array([0.9, 0.1, 0.8, 0.2]), same) == pytest.approx(
        (1 / 3 + 2 / 4) / 2
    )
    assert average_precision(np.array([0.1, 0.2, 0.3]), np.array([False, True, False])) == 0.5
    assert np.isnan(average_precision(np.array([0.1, 0.2]), np.array([False, False])))
    rng = np.random.default_rng(0)
    labels = rng.random(20000) < 0.1
    assert average_precision(rng.random(20000), labels) == pytest.approx(0.1, abs=0.02)


def test_cosine_distances():
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [0.0, 0.0]])
    distances = cosine_distances(vectors)
    assert np.allclose(distances[0, :4], [0, 0, 1, 2]) and np.allclose(distances, distances.T)
    assert np.isfinite(distances).all()


def test_pair_sets_split_the_pairs_into_three_conditions():
    words = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    speakers = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    train = np.array([True, True, True, True, True, True, False, False])
    pairs = PairSets(words, speakers, train)
    total = 8 * 7 // 2
    counts = {name: int(mask.sum()) for name, mask in pairs.masks.items()}
    assert counts == {"within_speaker": 6 + 1 + 1, "across_train": 8, "held_out": 12}
    assert sum(counts.values()) == total and len(pairs.same_word) == total
    # embeddings that are the word give a perfect score, and chance is the share of same pairs
    result = pairs.evaluate(np.eye(2)[words] + 1e-3)
    assert result["ap_within_speaker"] == result["ap_across_train"] == result["ap_held_out"] == 1.0
    assert result["chance_across_train"] == 0.5
    assert result["chance_within_speaker"] == pytest.approx(2 / 8)
    # embeddings that are the speaker cannot tell same-word pairs across speakers
    by_speaker = pairs.evaluate(np.eye(3)[speakers] + 1e-3)
    assert by_speaker["ap_across_train"] < 1.0
    # one token per speaker: no same-word pair within a speaker
    single = PairSets(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([True] * 4))
    result = single.evaluate(np.eye(2)[[0, 1, 0, 1]])
    assert np.isnan(result["ap_within_speaker"]) and np.isnan(result["chance_within_speaker"])
    assert result["ap_across_train"] == 1.0


def test_phonological_fidelity_and_edit_distances(tmp_path):
    words = run_forms(espeak_only(tmp_path, count=12)).lexicon.words
    matrix = edit_distance_matrix(words)
    assert matrix.shape == (12, 12) and np.array_equal(matrix, matrix.T)
    assert (np.diag(matrix) == 0).all() and matrix[np.triu_indices(12, 1)].min() >= 1
    # embeddings built from the distances themselves have a high fidelity
    centered = matrix**2 - (matrix**2).mean(axis=0) - (matrix**2).mean(axis=1)[:, None]
    values, vectors = np.linalg.eigh(-0.5 * (centered + (matrix**2).mean()))
    layout = vectors[:, -6:] * np.sqrt(np.maximum(values[-6:], 0))
    pearson, spearman = phonological_fidelity(layout + 5.0, matrix)
    assert pearson > 0.3 and spearman > 0.3
    random = np.random.default_rng(0).normal(size=(12, 6))
    assert abs(phonological_fidelity(random, matrix)[0]) < 0.4
    assert np.isnan(phonological_fidelity(np.ones((12, 3)), matrix)[0])


def test_neighbor_auc():
    # five words on a line: word 0 has a neighbor at distance 1 (word 1) and far words (3, 4)
    phonemes = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 2, 0, 2, 2],
            [3, 3, 2, 0, 2],
            [4, 4, 2, 2, 0],
        ],
        dtype=float,
    )
    angles = np.radians([0, 10, 45, 80, 90])
    ordered = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    auc, words = neighbor_auc(ordered, phonemes)
    assert auc == 1.0 and words == 2  # only words 0 and 1 have a neighbor at distance 1
    # the neighbor is the farthest word: the opposite of what the phonemes say
    reversed_angles = np.radians([0, 90, 45, 10, 20])
    reversed_layout = np.stack([np.cos(reversed_angles), np.sin(reversed_angles)], axis=1)
    assert neighbor_auc(reversed_layout, phonemes)[0] == 0.0
    # ties count one half, and random embeddings are near one half
    assert neighbor_auc(np.ones((5, 2)), phonemes) == (0.5, 2)
    rng = np.random.default_rng(0)
    big = np.abs(np.subtract.outer(rng.integers(0, 6, 60), rng.integers(0, 6, 60))).astype(float)
    big = np.minimum(np.triu(big, 1) + np.triu(big, 1).T, 5)
    assert abs(neighbor_auc(rng.normal(size=(60, 8)), big)[0] - 0.5) < 0.1
    # no word has a neighbor at distance 1: undefined
    far = np.full((4, 4), 3.0) - 3.0 * np.eye(4)
    result = neighbor_auc(rng.normal(size=(4, 3)), far)
    assert np.isnan(result[0]) and result[1] == 0


def test_harder_neighbor_auc_compares_distance_1_with_distance_2():
    # words 0 and 1 are neighbors at distance 1; word 2 is at distance 2 from both, word 3 at 4
    phonemes = np.array(
        [[0, 1, 2, 4], [1, 0, 2, 4], [2, 2, 0, 4], [4, 4, 4, 0]],
        dtype=float,
    )
    # the distance-2 word lies between the two neighbors, and the distance-4 word is far away
    angles = np.radians([0, 30, 10, 90])
    layout = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert neighbor_auc(layout, phonemes) == (1.0, 2)  # against distance 3 or more: perfect
    assert neighbor_auc(layout, phonemes, 2, True) == (0.0, 2)  # against distance 2: all wrong
    # with the neighbors closer to each other than to the distance-2 word, both are 1
    angles = np.radians([0, 10, 30, 90])
    layout = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert neighbor_auc(layout, phonemes) == (1.0, 2)
    assert neighbor_auc(layout, phonemes, 2, True) == (1.0, 2)
    # only words with neighbors at both distances count: word 1 loses its distance-2 neighbor
    fewer = phonemes.copy()
    fewer[1, 2] = fewer[2, 1] = 3
    assert neighbor_auc(layout, fewer, 2, True) == (1.0, 1)
    assert neighbor_auc(layout, fewer)[1] == 2
    # distance 2 exactly: without any word at distance 2 the measure is undefined
    result = neighbor_auc(layout, np.where(phonemes == 2, 3.0, phonemes), 2, True)
    assert np.isnan(result[0]) and result[1] == 0


def test_sample_word_means():
    tokens = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 9.0]])
    words = np.array([0, 0, 1, 2])
    train = np.array([True, True, True, False])
    means, present = sample_word_means(tokens, words, train, 4)
    assert present.tolist() == [True, True, False, False]
    assert np.allclose(means[0], [2, 0]) and np.allclose(means[1], [0, 5])


def test_evaluation_sample():
    rng = np.random.default_rng(0)
    assert np.array_equal(evaluation_sample(30, rng, 50), np.arange(30))
    sample = evaluation_sample(1000, rng, 50)
    assert len(sample) == len(set(sample)) == 50 and np.all(np.diff(sample) > 0)


def test_evaluation_table_is_above_chance(tmp_path):
    run, _ = embed_run(tmp_path, count=10)
    table = evaluate_embeddings(run.embeddings, run.lexicon.words, run.synthesis, run.streams.eval)
    table = plain_rows(table)
    assert table.columns == list(EVAL_COLUMNS) and table.height == 2
    assert table["embedding"].to_list() == ["cochleagram_fixed", "logmel_fixed"]
    assert table["encoder"].to_list() == ["fixed", "fixed"]
    assert table["source"].to_list() == ["cochleagram", "logmel"]
    assert table["layer"].null_count() == 2 and table["configured"].all()
    assert set(table["basis"]) == {"stored"} and set(table["word_set"]) == {"all"}
    assert table["tokens_evaluated"].to_list() == [60, 60]
    assert table["words_evaluated"].to_list() == [10, 10]
    for row in table.iter_rows(named=True):
        for condition in ("within_speaker", "across_train", "held_out"):
            assert row[f"ap_{condition}"] > 2 * row[f"chance_{condition}"], (row, condition)
        assert -1 <= row["fidelity_pearson"] <= 1 and -1 <= row["fidelity_spearman"] <= 1
    # two tokens per speaker: one same-word pair for each word and speaker
    assert table["chance_within_speaker"][0] == pytest.approx(10 / (3 * 20 * 19 / 2) * 3, rel=1e-6)
    # a sample of tokens still evaluates
    again = evaluate_embeddings(
        run.embeddings, run.lexicon.words, run.synthesis, np.random.default_rng(1), max_tokens=40
    )
    again = plain_rows(again)
    assert again["tokens_evaluated"].to_list() == [40, 40]
    assert (again["ap_across_train"] > again["chance_across_train"]).all()
    run.evaluation = table
    folder = run.write(tmp_path / "run")
    written = pl.read_csv(folder / "eval" / "embeddings.csv")
    written = plain_rows(written)
    assert written.columns == list(EVAL_COLUMNS) and written.height == 2
    assert written["ap_held_out"].to_list() == pytest.approx(
        table["ap_held_out"].to_list(), abs=1e-6
    )


def test_evaluation_with_and_without_the_long_synthesis_words(tmp_path):
    run, _ = embed_run(tmp_path, count=10)
    words = run.lexicon.words
    plain = evaluate_embeddings(run.embeddings, words, run.synthesis, np.random.default_rng(0))
    plain = plain_rows(plain)
    assert set(plain["word_set"]) == {"all"}  # no word is flagged: one set of rows
    for word in words:
        word.long_synthesis = word.label in ("W.2", "W.7")
    table = evaluate_embeddings(run.embeddings, words, run.synthesis, np.random.default_rng(0))
    table = plain_rows(table)
    assert table.height == 4
    assert table["word_set"].to_list() == ["all", "without_long_synthesis"] * 2
    everything = table.filter(pl.col("word_set") == "all")
    without = table.filter(pl.col("word_set") == "without_long_synthesis")
    assert everything["words_evaluated"].to_list() == [10, 10]
    assert without["words_evaluated"].to_list() == [8, 8]
    assert everything["tokens_evaluated"].to_list() == [60, 60]
    assert without["tokens_evaluated"].to_list() == [48, 48]
    # the rows for all words are what they were without the flags
    assert everything["ap_across_train"].to_list() == plain["ap_across_train"].to_list()
    assert everything["fidelity_spearman"].to_list() == plain["fidelity_spearman"].to_list()
    # the rows without the flagged words equal an evaluation of the other words alone
    kept = [w for w in words if not w.long_synthesis]
    _, _, train = token_layout(words, run.synthesis)
    store = run.embeddings["logmel_fixed"]
    keep_rows = np.array([t.word not in ("W.2", "W.7") for t in run.synthesis.tokens])
    token_words, token_speakers, _ = token_layout(words, run.synthesis)
    expected = PairSets(
        token_words[keep_rows], token_speakers[keep_rows], train[keep_rows]
    ).evaluate(store.tokens[keep_rows])
    row = without.filter(pl.col("embedding") == "logmel_fixed").row(0, named=True)
    assert row["ap_across_train"] == pytest.approx(expected["ap_across_train"])
    assert row["chance_held_out"] == pytest.approx(expected["chance_held_out"])
    keep_words = np.array([not w.long_synthesis for w in words])
    pearson, spearman = phonological_fidelity(store.types[keep_words], edit_distance_matrix(kept))
    assert row["fidelity_pearson"] == pytest.approx(pearson)
    assert row["fidelity_spearman"] == pytest.approx(spearman)


def test_neighbor_auc_in_the_table(tmp_path):
    # one-syllable words with minimal pairs, so that some words have a neighbor at distance 1
    run, _ = embed_run(tmp_path, count=40, syllables={1: 1})
    matrix = edit_distance_matrix(run.lexicon.words)
    with_neighbor = int(((matrix == 1).any(axis=1) & (matrix >= 3).any(axis=1)).sum())
    assert with_neighbor > 0
    table = evaluate_embeddings(
        run.embeddings, run.lexicon.words, run.synthesis, np.random.default_rng(0)
    )
    table = plain_rows(table)
    assert table["auc_words"].to_list() == [with_neighbor, with_neighbor]
    for name, value in zip(table["embedding"], table["fidelity_auc"], strict=True):
        expected, _ = neighbor_auc(run.embeddings[name].types, matrix)
        assert 0 <= value <= 1 and value == pytest.approx(expected)
    # the harder measure: distance 1 against distance 2, over the words with both
    with_both = int(((matrix == 1).any(axis=1) & (matrix == 2).any(axis=1)).sum())
    assert 0 < with_both <= with_neighbor
    assert table["auc_1v2_words"].to_list() == [with_both, with_both]
    for name, value in zip(table["embedding"], table["fidelity_auc_1v2"], strict=True):
        expected, _ = neighbor_auc(run.embeddings[name].types, matrix, 2, True)
        assert 0 <= value <= 1 and value == pytest.approx(expected)


# ---------------------------------------------------------------------------------------------
# SoundEmbeddings
# ---------------------------------------------------------------------------------------------


@needs_torch
def test_sound_embeddings_interface(tmp_path):
    run, engine = embed_run(tmp_path)
    assert SoundEmbeddings.names(tmp_path / "run") == ["cochleagram_fixed", "logmel_fixed"]
    sounds = SoundEmbeddings.load(tmp_path / "run", "cochleagram_fixed", engines={"espeak": engine})
    assert sounds.types.shape == (6, 8) and sounds.tokens.shape == (36, 8) and sounds.dims == 8
    assert sounds.pretrained is False
    assert sounds.words["label"].to_list() == [w.label for w in run.lexicon.words]
    assert sounds.speakers["label"].to_list() == ["S.1", "S.2", "S.3"]
    assert sounds.token_words.tolist() == [i for i in range(6) for _ in range(6)]
    assert sounds.token_speakers.tolist() == [0, 0, 1, 1, 2, 2] * 6
    held_out = (sounds.speakers["split"] == "held_out").to_numpy()
    assert np.array_equal(sounds.token_held_out, held_out[sounds.token_speakers])
    for w in range(6):
        rows = (sounds.token_words == w) & ~sounds.token_held_out
        assert np.allclose(sounds.types[w], sounds.tokens[rows].mean(axis=0), atol=1e-6)
    tensors = sounds.to_torch()
    import torch

    assert set(tensors) == {
        "types",
        "tokens",
        "token_words",
        "token_speakers",
        "token_held_out",
        "token_augmented",
    }
    assert all(isinstance(t, torch.Tensor) for t in tensors.values())
    assert tensors["types"].shape == (6, 8) and tensors["types"].dtype == torch.float32
    assert (
        tensors["token_words"].dtype == torch.int64
        and tensors["token_held_out"].dtype == torch.bool
    )
    assert np.array_equal(tensors["tokens"].numpy(), sounds.tokens)
    forms = sounds.lexicon_forms()
    assert [f.label for f in forms] == sounds.words["label"].to_list()
    assert [f.arpabet for f in forms] == sounds.words["arpabet"].to_list()
    assert [f.espeak for f in forms] == sounds.words["espeak"].to_list()


def test_embed_reproduces_the_stored_embeddings(tmp_path):
    run, engine = embed_run(tmp_path)
    calls = engine.calls
    for name in ("cochleagram_fixed", "logmel_fixed"):
        sounds = SoundEmbeddings.load(tmp_path / "run", name, engines={"espeak": engine})
        arpabet = sounds.words["arpabet"].to_list()
        labels = sounds.speakers["label"].to_list()
        for token in (1, 2):
            embedded = sounds.embed(arpabet, labels, token)
            assert embedded.shape == (6, 3, 8) and embedded.dtype == np.float32
            for w in range(6):
                for s in range(3):
                    row = sounds._stored[(f"W.{w + 1}", labels[s], token)]
                    assert np.allclose(embedded[w, s], sounds.tokens[row], atol=1e-5)
        # the default speakers are the training speakers, and the types are their mean
        train = sounds.speakers.filter(pl.col("split") == "train")["label"].to_list()
        assert sounds.embed(arpabet[:2]).shape == (2, len(train), 8)
        assert np.allclose(sounds.embed_types(arpabet), sounds.types, atol=1e-5)
        # WordForm objects work as well as ARPAbet strings
        assert np.array_equal(sounds.embed(sounds.lexicon_forms()[:2]), sounds.embed(arpabet[:2]))
    # a form of the lexicon comes from the token table and the cache: nothing is synthesized
    assert engine.calls == calls


def test_embed_a_novel_word_looks_up_the_cache_before_synthesizing(tmp_path):
    run, engine = embed_run(tmp_path)
    sounds = SoundEmbeddings.load(tmp_path / "run", "logmel_fixed", engines={"espeak": engine})
    novel = "Z AE1 M P IH0 K"
    assert novel not in sounds.words["arpabet"].to_list()
    calls = engine.calls
    first = sounds.embed([novel], ["S.1", "S.3"])
    assert first.shape == (1, 2, 8) and np.isfinite(first).all()
    assert engine.calls == calls + 4  # two speakers, two tokens each
    # the clips are now in the audio cache, so the same call synthesizes nothing
    again = sounds.embed([novel], ["S.1", "S.3"])
    assert engine.calls == calls + 4 and np.array_equal(again, first)
    other = SoundEmbeddings.load(tmp_path / "run", "logmel_fixed", engines={"espeak": engine})
    assert np.array_equal(other.embed([novel], ["S.3"])[0, 0], first[0, 1])
    assert engine.calls == calls + 4
    # the word embedding of a novel word: the mean over the training speakers' tokens
    types = sounds.embed_types([novel])
    assert types.shape == (1, 8)
    train = sounds.speakers.filter(pl.col("split") == "train")["label"].to_list()
    both = np.stack([sounds.embed([novel], train, k) for k in (1, 2)])
    assert np.allclose(types[0], both.mean(axis=(0, 2))[0], atol=1e-6)
    # a novel word is not any stored word
    assert not any(np.allclose(types[0], t, atol=1e-3) for t in sounds.types)


def test_embed_errors(tmp_path):
    run, engine = embed_run(tmp_path)
    sounds = SoundEmbeddings.load(tmp_path / "run", "logmel_fixed", engines={"espeak": engine})
    with pytest.raises(ValueError, match="unknown speaker"):
        sounds.embed(["K AE1 T"], ["S.9"])
    with pytest.raises(ValueError, match="not ARPAbet"):
        sounds.embed(["K XX1 T"])
    with pytest.raises(ValueError, match="stress digit"):
        sounds.embed(["K AE T"])
    with pytest.raises(ValueError, match="beyond tokens_per_speaker"):
        sounds.embed(["Z AE1 M P IH0 K"], ["S.1"], token=3)


# ---------------------------------------------------------------------------------------------
# Pretrained encoders
# ---------------------------------------------------------------------------------------------

HUBERT = {"name": "hubert", "encoder": "pretrained", "model": "facebook/hubert-base-ls960"}


def hubert_run(tmp_path, **settings):
    run, engine = stand_in_run(tmp_path, embeddings=[{**HUBERT, **settings}], count=4)
    data = run.config.resolved()
    data["device"] = "cpu"
    run.config = parse_config(data, "stand_in")
    run.embeddings = compute_embeddings(
        run.config,
        run.lexicon.words,
        run.synthesis,
        run.frontends,
        tmp_path / "run",
        local_only=True,
    )
    return run, engine


@needs_hubert
def test_pretrained_embedding_stores_the_configured_layer(tmp_path):
    run, engine = hubert_run(tmp_path)
    folder = tmp_path / "run"
    store = run.embeddings["hubert"]
    # only the configured layer is stored: layer 8 by default
    assert sorted(p.name for p in store.folder.iterdir()) == [
        "meta.yaml",
        "tokens.npy",
        "types.npy",
    ]
    assert store.layers is None
    assert store.tokens.shape == (24, 768) and store.types.shape == (4, 768)
    assert np.isfinite(store.tokens).all()
    # every output labels the embedding as pretrained
    meta = store.meta
    assert meta["pretrained"] is True and meta["encoder"] == "pretrained"
    assert meta["model"] == "facebook/hubert-base-ls960" and meta["revision"]
    assert meta["layers"] == 13 and meta["layer"] == 8 and meta["layers_stored"] is False
    assert meta["device"] == "cpu" and store.summary()["pretrained"] is True
    run.write(folder)
    written = yaml.safe_load((folder / "config.yaml").read_text())
    assert written["provenance"]["models"]["embedding:hubert"] == {
        "model": "facebook/hubert-base-ls960",
        "revision": meta["revision"],
        "pretrained": True,
    }
    assert (
        written["embeddings"][0]["layer"] == 8 and written["embeddings"][0]["store_layers"] is False
    )
    # on the CPU, a second computation is identical
    other = compute_embeddings(
        run.config, run.lexicon.words, run.synthesis, None, tmp_path / "again", local_only=True
    )["hubert"]
    assert np.array_equal(other.tokens, store.tokens)
    # embed reproduces the stored embeddings, and handles a novel word
    sounds = SoundEmbeddings.load(folder, "hubert", local_only=True, engines={"espeak": engine})
    assert sounds.pretrained and sounds.dims == 768
    embedded = sounds.embed(sounds.words["arpabet"].to_list(), sounds.speakers["label"].to_list())
    for w in range(4):
        for s in range(3):
            row = sounds._stored[(f"W.{w + 1}", f"S.{s + 1}", 1)]
            assert np.allclose(embedded[w, s], sounds.tokens[row], atol=1e-5)
    assert sounds.embed(["Z AE1 M P IH0 K"], ["S.1"]).shape == (1, 1, 768)


@needs_hubert
def test_layer_sweep_runs_on_the_sample_without_stored_layers(tmp_path):
    run, _ = hubert_run(tmp_path / "plain")
    arguments = (run.embeddings, run.lexicon.words, run.synthesis)
    table = evaluate_embeddings(
        *arguments, np.random.default_rng(0), config=run.config, local_only=True
    )
    table = plain_rows(table)
    # one row for the stored embedding, and one sweep row for each of the 13 layers
    assert table.height == 14 and table["basis"].to_list() == ["stored"] + ["sweep"] * 13
    assert table["layer"].to_list() == [8, *range(13)]
    assert table["configured"].to_list() == [True] + [layer == 8 for layer in range(13)]
    assert set(table["encoder"]) == {"pretrained"}
    assert set(table["source"]) == {"facebook/hubert-base-ls960"}
    assert (table["ap_across_train"] > table["chance_across_train"]).all()
    stored = table.row(0, named=True)
    swept = table.filter((pl.col("basis") == "sweep") & pl.col("configured")).row(0, named=True)
    # the sample is every token here, so the sweep's row for layer 8 matches the stored row
    for column in ("ap_within_speaker", "ap_across_train", "ap_held_out", "fidelity_spearman"):
        assert swept[column] == pytest.approx(stored[column], abs=1e-4)
    # the sweep needs the configuration to run the model, and can be left out
    with pytest.raises(ValueError, match="needs the run's configuration"):
        evaluate_embeddings(*arguments, np.random.default_rng(0))
    plain = plain_rows(evaluate_embeddings(*arguments, np.random.default_rng(0), sweep=False))
    assert plain.height == 1
    # a sample of the tokens: the model runs on the sample only
    seen = []
    small = evaluate_embeddings(
        *arguments,
        np.random.default_rng(0),
        config=run.config,
        max_tokens=10,
        local_only=True,
        progress=lambda name, done, total: seen.append((name, done, total)),
    )
    small = plain_rows(small)
    assert seen[-1] == ("hubert", 10, 10) and len(seen) == 10
    assert set(small["tokens_evaluated"]) == {10}
    # with store_layers, every layer is stored, and the sweep reads the stored layers
    kept, _ = hubert_run(tmp_path / "kept", store_layers=True, layer=3)
    store = kept.embeddings["hubert"]
    assert (store.folder / "layers.npy").exists() and store.meta["layers_stored"] is True
    assert store.layers.shape == (24, 13, 768) and store.meta["layer"] == 3
    assert np.array_equal(store.tokens, store.layers[:, 3])
    from_stored = evaluate_embeddings(
        kept.embeddings, kept.lexicon.words, kept.synthesis, np.random.default_rng(0)
    )
    from_stored = plain_rows(from_stored)
    assert from_stored.height == 14 and from_stored["layer"][0] == 3
    both = ["ap_across_train", "ap_held_out", "fidelity_pearson"]
    ours = table.filter(pl.col("basis") == "sweep").select(both).to_numpy()
    theirs = from_stored.filter(pl.col("basis") == "sweep").select(both).to_numpy()
    assert np.allclose(ours, theirs, atol=1e-4)
    # computing the embedding again without store_layers removes the stored layers
    data = kept.config.resolved()
    data["embeddings"][0]["store_layers"] = False
    again = compute_embeddings(
        parse_config(data, "x"),
        kept.lexicon.words,
        kept.synthesis,
        None,
        tmp_path / "kept" / "run",
        local_only=True,
    )
    assert not again["hubert"].reused and not (store.folder / "layers.npy").exists()


@needs_hubert
def test_pretrained_encoder_checks_the_layer_and_short_clips():
    from semantic_world.wordforms.encoders.pretrained import PretrainedEncoder

    with pytest.raises(ValueError, match="layers 0 to 12"):
        PretrainedEncoder("facebook/hubert-base-ls960", 13, device="cpu", local_only=True)
    encoder = PretrainedEncoder("facebook/hubert-base-ls960", 12, device="cpu", local_only=True)
    assert (encoder.layers, encoder.dims, encoder.whisper) == (13, 768, False)
    short = encoder.all_layers(
        np.full(100, 0.01, dtype=np.float32)
    )  # shorter than the model's window
    assert short.shape == (13, 768) and np.isfinite(short).all()
    assert encoder.provenance()["pretrained"] is True


@needs_whisper
def test_whisper_encoder_pools_the_frames_of_the_clip():
    from semantic_world.wordforms.encoders.pretrained import PretrainedEncoder

    encoder = PretrainedEncoder("openai/whisper-small.en", 6, device="cpu", local_only=True)
    assert (encoder.layers, encoder.dims, encoder.whisper) == (13, 768, True)
    reference = np.load(FIXTURES / "cochleagram_reference.npz")
    first = encoder.all_layers(reference["word_1_input"])
    second = encoder.all_layers(reference["word_2_input"])
    assert first.shape == (13, 768) and np.isfinite(first).all()
    assert np.array_equal(first, encoder.all_layers(reference["word_1_input"]))
    assert not np.allclose(first[6], second[6], atol=1e-3)
    assert np.array_equal(encoder.encode(reference["word_1_input"]), first[6])


# ---------------------------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------------------------


def meanings_file(tmp_path, count=5, features=12, seed=0):
    rng = np.random.default_rng(seed)
    table = {"label": [f"C{i + 1}" for i in range(count)]}
    for j in range(features):
        table[f"IS.{j + 1}"] = rng.integers(0, 2, size=count).tolist()
    path = tmp_path / "categories_generative.csv"
    pl.DataFrame(table).write_csv(path)
    return path


def test_load_meanings(tmp_path):
    path = meanings_file(tmp_path)
    ids, features = load_meanings(path)
    assert ids == ["C1", "C2", "C3", "C4", "C5"] and features.shape == (5, 12)
    assert set(np.unique(features)) <= {0, 1}
    assert hamming_distances(features).shape == (10,)
    bad = tmp_path / "bad.csv"
    bad.write_text("label,f1\nA,0\nA,1\n")
    with pytest.raises(AssignmentError, match="not distinct"):
        load_meanings(bad)
    bad.write_text("label,f1\nA,0\nB,2\n")
    with pytest.raises(AssignmentError, match="only 0 and 1"):
        load_meanings(bad)
    bad.write_text("label\nA\n")
    with pytest.raises(AssignmentError, match="feature column"):
        load_meanings(bad)


def test_arbitrary_assignment(tmp_path):
    words = run_forms(espeak_only(tmp_path, count=12)).lexicon.words
    ids, features = load_meanings(meanings_file(tmp_path, count=8))
    first = assign_arbitrary(words, ids, features, np.random.default_rng(3))
    assert first.meanings == ids and len(first.words) == 8
    assert len({w.label for w in first.words}) == 8  # no word is used twice
    again = assign_arbitrary(words, ids, features, np.random.default_rng(3))
    assert [w.label for w in again.words] == [w.label for w in first.words]
    other = assign_arbitrary(words, ids, features, np.random.default_rng(4))
    assert [w.label for w in other.words] != [w.label for w in first.words]
    summary = first.summary
    assert summary["mode"] == "arbitrary" and summary["meanings"] == 8
    assert summary["words_available"] == 12 and summary["features"] == 12
    assert -1 <= summary["correlation"] <= 1
    null = summary["null"]
    assert null["samples"] == 1000 and abs(null["mean"]) < 0.1 and 0 < null["p_value"] <= 1
    assert null["p05"] < null["mean"] < null["p95"]
    frame = first.frame()
    assert frame.columns == ["meaning", "word", "spelling", "arpabet"] and frame.height == 8
    # arbitrary assignments have a mean correlation near 0 over seeds
    values = [
        assign_arbitrary(words, ids, features, np.random.default_rng(s), 10).summary["correlation"]
        for s in range(40)
    ]
    assert abs(np.mean(values)) < 0.1
    with pytest.raises(AssignmentError, match="raise wordforms.count"):
        assign_arbitrary(words[:5], ids, features, np.random.default_rng(0))


def test_assign_command_writes_the_lexicon(tmp_path, capsys):
    data = espeak_only(tmp_path, count=9).resolved()
    data["assignment"] = {"mode": "arbitrary", "meanings": str(meanings_file(tmp_path, count=6))}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["assign", str(path), "--out", str(out)]) == 0
    assert "assignment (arbitrary): 6 meanings" in capsys.readouterr().out
    lexicon = pl.read_csv(out / "assignment" / "lexicon.csv")
    assert lexicon["meaning"].to_list() == [f"C{i}" for i in range(1, 7)]
    words = pl.read_csv(out / "words.csv")
    assert set(lexicon["word"]) <= set(words["label"]) and lexicon["word"].n_unique() == 6
    summary = yaml.safe_load((out / "assignment" / "summary.yaml").read_text())
    assert summary["mode"] == "arbitrary" and summary["null"]["samples"] == 1000
    assert yaml.safe_load((out / "summary.yaml").read_text())["assignment"] == summary
    # the same seed gives the same assignment, and changing the speakers does not change it
    data["synthesis"]["tokens_per_speaker"] = 3
    path.write_text(yaml.safe_dump(data))
    assert main(["assign", str(path), "--out", str(tmp_path / "run2")]) == 0
    assert (tmp_path / "run2" / "assignment" / "lexicon.csv").read_bytes() == (
        out / "assignment" / "lexicon.csv"
    ).read_bytes()
    # too many meanings is an error that says what to change
    data["assignment"]["meanings"] = str(meanings_file(tmp_path, count=30))
    path.write_text(yaml.safe_dump(data))
    assert main(["assign", str(path), "--out", str(out)]) == 1
    assert "raise wordforms.count" in capsys.readouterr().err
    # without a meanings table nothing is assigned
    data["assignment"]["meanings"] = None
    path.write_text(yaml.safe_dump(data))
    assert main(["assign", str(path), "--out", str(tmp_path / "run3")]) == 0
    assert not (tmp_path / "run3" / "assignment").exists()


# ---------------------------------------------------------------------------------------------
# The whole pipeline and the examples, with the real engines
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """The tiny configuration, run once with the real engines."""
    folder = tmp_path_factory.mktemp("tiny")
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(folder / "cache")
    data["device"] = "cpu"
    path = folder / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    assert main(["all", str(path), "--out", str(folder / "run")]) == 0
    return path, folder / "run"


@needs_espeak
@needs_piper
def test_all_on_the_tiny_configuration_is_above_chance(tiny_run, capsys):
    path, run = tiny_run
    names = sorted(p.name for p in run.iterdir())
    assert names == [
        "affixes.csv",
        "config.yaml",
        "embeddings",
        "eval",
        "frontends",
        "speakers.csv",
        "summary.yaml",
        "tokens.csv",
        "words.csv",
    ]
    table = pl.read_csv(run / "eval" / "embeddings.csv")
    table = plain_rows(table)
    assert table["embedding"].to_list() == ["cochleagram_fixed"] * 4 + ["logmel_fixed"] * 4
    assert table["kind"].to_list() == ["content", "function", "inflected", "all"] * 2
    for row in table.iter_rows(named=True):
        # same-different average precision is above chance for every embedding and kind
        assert row["ap_across_train"] > row["chance_across_train"]
        assert row["ap_held_out"] > row["chance_held_out"]
        # one token per speaker: there is no same-word pair within a speaker
        assert row["ap_within_speaker"] is None
        assert row["dims"] == 16 and row["tokens_evaluated"] == 6 * row["words_evaluated"]
        if row["kind"] == "content":
            assert row["chance_across_train"] == pytest.approx(0.05)
            assert row["tokens_evaluated"] == 120
        # the stem AUC is reported for every embedding, and is far above chance
        if row["kind"] in ("inflected", "all"):
            assert row["stem_auc"] > 0.7 and row["stem_auc_forms"] > 40
        else:
            assert row["stem_auc"] is None
    summary = yaml.safe_load((run / "summary.yaml").read_text())
    assert set(summary["embeddings"]) == {"cochleagram_fixed", "logmel_fixed"}
    # a second run finds everything stored, and gives the same table
    assert main(["eval", str(path), "--out", str(run)]) == 0
    text = capsys.readouterr().out
    assert text.count("already stored") == 5 and "0 synthesized" in text
    assert "closed-class forms: 15 function words, 3 affixes" in text
    assert "cochleagram_fixed" in text and "held-out" in text
    assert plain_rows(pl.read_csv(run / "eval" / "embeddings.csv")).equals(table)


@needs_espeak
@needs_piper
def test_embed_with_the_real_engines(tiny_run):
    _, run = tiny_run
    for name in ("cochleagram_fixed", "logmel_fixed"):
        sounds = SoundEmbeddings.load(run, name)
        # stored words: from the cache, never from an engine (Piper cannot repeat a clip)
        sounds.engines = {}
        arpabet = sounds.words["arpabet"].to_list()
        words = sounds.words["label"].to_list()
        labels = sounds.speakers["label"].to_list()
        embedded = sounds.embed(arpabet, labels)
        assert embedded.shape == (len(words), 6, 16) and len(words) > 35
        assert list(sounds.word_kinds[:35]) == ["content"] * 20 + ["function"] * 15
        for w, word in enumerate(words):
            for s in range(6):
                row = sounds._stored[(word, labels[s], 1)]
                assert np.allclose(embedded[w, s], sounds.tokens[row], atol=1e-5)
        assert np.allclose(sounds.embed_types(arpabet), sounds.types, atol=1e-5)
    # a novel word is synthesized by both engines, and then comes from the cache
    from semantic_world.wordforms.streams import Streams
    from semantic_world.wordforms.synth import make_engines

    sounds = SoundEmbeddings.load(run, "cochleagram_fixed")
    engines = make_engines(sounds.config, Streams(sounds.config.seed))
    calls = []
    for engine in engines.values():
        original = engine.synthesize

        def counted(phonemes, settings, original=original):
            calls.append(phonemes)
            return original(phonemes, settings)

        engine.synthesize = counted
    sounds.engines = engines
    novel = sounds.embed(["Z AE1 M P IH0 K"], ["S.1", "S.5"])
    assert novel.shape == (1, 2, 16) and np.isfinite(novel).all()
    assert len(calls) == 2  # one clip from Piper and one from espeak-ng
    assert np.array_equal(sounds.embed(["Z AE1 M P IH0 K"], ["S.1", "S.5"]), novel)
    assert len(calls) == 2  # the second time, both clips came from the audio cache


@needs_espeak
@needs_piper
@needs_torch
def test_examples_run_on_the_tiny_configuration(tiny_run, capsys):
    import importlib.util
    import time

    path, run = tiny_run
    results = {}
    for name in ("wordforms_lm_inputs", "wordforms_contrastive"):
        spec = importlib.util.spec_from_file_location(
            name, DATA.parents[1] / "examples" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        start = time.time()
        results[name] = module.main(["--config", str(path), "--run", str(run)])
        assert time.time() - start < 600  # each example runs in under 10 minutes on a CPU
    text = capsys.readouterr().out
    language = results["wordforms_lm_inputs"]
    assert language["last_loss"] < language["uniform"] < language["first_loss"]
    # the tiny run's 86 forms include inflected forms whose sounds barely differ from their
    # stems, so the model gets less close to the bigram entropy than with content words alone
    assert language["last_loss"] < language["entropy"] + 1.0
    assert np.isfinite(language["novel_log_probability"])
    assert "novel word 'Z AE1 M P IH0 K'" in text
    contrastive = results["wordforms_contrastive"]
    assert contrastive["training tokens"] > 0.9
    assert contrastive["held-out speakers"] > 3 / 12  # chance is 1 in 12
    assert 0 <= contrastive["novel words"] <= 1
    assert "retrieval accuracy, novel words" in text
