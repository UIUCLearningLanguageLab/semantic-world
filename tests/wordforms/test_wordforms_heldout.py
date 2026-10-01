"""Held-out words for trained encoders, talker normalization, the CPC options, and the
reverberation calibration."""

# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from test_wordforms_embeddings import WordEngine
from wordforms_support import (
    needs_audio,
    needs_cmudict,
    needs_pyroomacoustics,
    needs_torch,
    needs_wordfreq,
)

from semantic_world.wordforms import Run, run_forms
from semantic_world.wordforms.config import ConfigError, parse_config
from semantic_world.wordforms.embeddings import (
    SoundEmbeddings,
    compute_embeddings,
    normalization_basis,
    speaker_means,
    token_layout,
    training_word_tokens,
)
from semantic_world.wordforms.evaluate import evaluate_embeddings
from semantic_world.wordforms.frontends import compute_frontends
from semantic_world.wordforms.synth import synthesize_lexicon

pytestmark = [needs_cmudict, needs_wordfreq]

FIXED = {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel", "pca_dims": 8}
NORMALIZED = {**FIXED, "name": "logmel_normalized", "talker_normalization": True}
CPC = {"name": "cpc", "encoder": "learned", "kind": "cpc", "dims": 16, "hidden": 32, "epochs": 2, "batch_size": 8, "steps_ahead": 4, "negatives": 8}  # fmt: skip


def held_config(tmp_path, embeddings, count: int = 20, **extra):
    return parse_config(
        {
            "name": "held_test",
            "wordforms": {"count": count},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 2,
                "held_out_speaker_proportion": 0.34,
                "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
            },
            "embeddings": embeddings,
            "device": "cpu",
            **extra,
        },
        "held_test",
    )


def held_run(tmp_path, config, folder="run") -> Run:
    run = run_forms(config)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": WordEngine()}, check=False
    )
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / folder)
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / folder
    )
    return run


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_training_and_normalization_settings():
    config = parse_config({}, "x")
    assert config.training.held_out_word_proportion == 0.2
    assert config.resolved()["training"] == {"held_out_word_proportion": 0.2}
    # talker normalization is off for every embedding by default
    assert all(e.talker_normalization is False for e in config.embeddings)
    cpc = next(e for e in config.embeddings if e.name == "cpc_logmel")
    assert cpc.negatives_from == "batch" and cpc.embedding_from == "context"
    chosen = parse_config({"training": {"held_out_word_proportion": 0.5}, "embeddings": [NORMALIZED, {**CPC, "embedding_from": "latents", "negatives_from": "clip"}]}, "x")  # fmt: skip
    assert chosen.training.held_out_word_proportion == 0.5
    assert chosen.embeddings[0].talker_normalization is True
    assert chosen.embeddings[1].embedding_from == "latents"
    assert chosen.embeddings[1].negatives_from == "clip"
    assert parse_config(chosen.resolved(), "x") == chosen
    for data, field in (
        ({"training": {"held_out_word_proportion": 1.5}}, "training.held_out_word_proportion"),
        ({"training": {"words": 3}}, "training.words"),
        ({"embeddings": [{**FIXED, "talker_normalization": "yes"}]}, "embeddings[0].talker_normalization"),
        ({"embeddings": [{**CPC, "embedding_from": "frames"}]}, "embeddings[0].embedding_from"),
        ({"embeddings": [{**CPC, "negatives_from": "speaker"}]}, "embeddings[0].negatives_from"),
    ):  # fmt: skip
        with pytest.raises(ConfigError) as info:
            parse_config(data, "x")
        assert info.value.field == field, info.value


# ---------------------------------------------------------------------------------------------
# Held-out words
# ---------------------------------------------------------------------------------------------


def test_a_seeded_share_of_the_content_words_is_held_out():
    config = parse_config({"wordforms": {"count": 40}, "closed_class": None}, "x")
    words = run_forms(config).lexicon.words
    held = [w.label for w in words if w.held_out]
    assert len(held) == 8  # 20% of 40
    assert held == [w.label for w in run_forms(config).lexicon.words if w.held_out]
    assert held != [w.label for w in run_forms(config.with_seed(2)).lexicon.words if w.held_out]
    assert all(w.record()["split"] == ("held_out" if w.held_out else "train") for w in words)
    for proportion, expected in ((0, 0), (0.5, 20), (1.0, 39)):
        data = {"wordforms": {"count": 40}, "closed_class": None, "training": {"held_out_word_proportion": proportion}}  # fmt: skip
        assert sum(w.held_out for w in run_forms(parse_config(data, "x")).lexicon.words) == expected
    # closed-class forms never change which content words are held out; an inflected form
    # follows its stem, and a function word is never held out
    inflect = [{"words": "all", "affixes": ["PLURAL", "PAST", "PROGRESSIVE"]}]
    closed = run_forms(parse_config({"wordforms": {"count": 40}, "closed_class": {"inflect": inflect}}, "x")).lexicon.words  # fmt: skip
    assert [w.label for w in closed if w.held_out and w.kind == "content"] == held
    assert not any(w.held_out for w in closed if w.kind == "function")
    inflected = [w for w in closed if w.kind == "inflected"]
    assert inflected and all(w.held_out == (w.stem in held) for w in inflected)
    assert any(w.held_out for w in inflected)


@needs_audio
@needs_torch
def test_trained_encoders_never_see_the_held_out_words(tmp_path):
    config = held_config(tmp_path, [FIXED, CPC], closed_class=None)
    run = held_run(tmp_path, config)
    words, synthesis = run.lexicon.words, run.synthesis
    assert sum(w.held_out for w in words) == 4
    token_words, _, train = token_layout(words, synthesis)
    training = training_word_tokens(words, token_words)
    assert training.sum() == 16 * 3 * 2
    fixed = run.embeddings["logmel_fixed"]
    assert fixed.meta["projection"]["fitted_tokens"] == int((train & training).sum()) == 16 * 2 * 2
    assert "without held-out words" in fixed.meta["projection"]["fitted_on"]
    assert fixed.meta["held_out_words"] == 4
    assert run.embeddings["cpc"].meta["training"]["tokens"] == 16 * 2 * 2
    # every word still has a word embedding, the held-out ones from the frozen encoders
    assert fixed.types.shape == (20, 8) and run.embeddings["cpc"].types.shape[0] == 20
    # with no held-out words the projection is fitted on all of them, and differs
    everything = held_config(tmp_path, [FIXED], closed_class=None, training={"held_out_word_proportion": 0})  # fmt: skip
    other = held_run(tmp_path, everything, "all_words")
    assert other.embeddings["logmel_fixed"].meta["projection"]["fitted_tokens"] == 20 * 2 * 2
    assert not np.allclose(other.embeddings["logmel_fixed"].tokens, fixed.tokens)
    # the run folder and the interface record the split
    run.write(tmp_path / "run")
    table = pl.read_csv(tmp_path / "run" / "words.csv")
    assert table["split"].to_list().count("held_out") == 4
    sounds = SoundEmbeddings.load(tmp_path / "run", "logmel_fixed")
    assert sounds.word_held_out.sum() == 4
    assert [f.held_out for f in sounds.lexicon_forms()] == [w.held_out for w in words]


@needs_audio
def test_the_evaluation_reports_training_words_and_held_out_words(tmp_path):
    config = held_config(tmp_path, [FIXED], closed_class=None)
    run = held_run(tmp_path, config)
    table = evaluate_embeddings(run.embeddings, run.lexicon.words, run.synthesis, run.streams.eval, config=config)  # fmt: skip
    plain = table.filter(~pl.col("talker_normalized"))
    assert plain["word_split"].to_list() == ["all", "train", "held_out"]
    by = {r["word_split"]: r for r in plain.iter_rows(named=True)}
    assert by["all"]["words_evaluated"] == 20 and by["all"]["tokens_evaluated"] == 120
    assert by["train"]["words_evaluated"] == 16 and by["train"]["tokens_evaluated"] == 96
    assert by["held_out"]["words_evaluated"] == 4 and by["held_out"]["tokens_evaluated"] == 24
    assert by["held_out"]["chance_across_train"] > by["train"]["chance_across_train"]
    assert by["held_out"]["robustness_chance"] == pytest.approx(1 / 4)
    for row in by.values():
        assert row["ap_across_train"] > row["chance_across_train"]
    # without held-out words there is one split
    none = held_config(tmp_path, [FIXED], closed_class=None, training={"held_out_word_proportion": 0})  # fmt: skip
    run = held_run(tmp_path, none, "none")
    table = evaluate_embeddings(run.embeddings, run.lexicon.words, run.synthesis, run.streams.eval, config=none)  # fmt: skip
    assert set(table["word_split"]) == {"all"}


# ---------------------------------------------------------------------------------------------
# Talker normalization
# ---------------------------------------------------------------------------------------------


@needs_audio
def test_talker_normalization_subtracts_each_speakers_mean(tmp_path):
    config = held_config(tmp_path, [FIXED, NORMALIZED])
    run = held_run(tmp_path, config)
    words, synthesis = run.lexicon.words, run.synthesis
    raw, normalized = run.embeddings["logmel_fixed"], run.embeddings["logmel_normalized"]
    assert raw.speaker_means is None and raw.meta["talker_normalization"] is False
    means = normalized.speaker_means
    assert means.shape == (3, 8) and normalized.meta["talker_normalization"] is True
    token_words, token_speakers, _ = token_layout(words, synthesis)
    basis = normalization_basis(words, synthesis, token_words)
    # the basis is the clean tokens of the training content words: no held-out word, and no
    # function word
    kinds = np.array([w.kind for w in words])[token_words]
    held = np.array([w.held_out for w in words])[token_words]
    assert basis.sum() == 16 * 3 * 2 and not (basis & held).any() and (kinds[basis] == "content").all()  # fmt: skip
    expected = speaker_means(raw.tokens, token_speakers, basis, 3)
    assert np.allclose(means, expected, atol=1e-6)
    assert np.allclose(normalized.tokens, raw.tokens - means[token_speakers], atol=1e-6)
    for speaker in range(3):
        own = basis & (token_speakers == speaker)
        assert np.allclose(normalized.tokens[own].mean(axis=0), 0.0, atol=1e-5)
    # embed gives the stored (normalized) embeddings, for training and held-out speakers alike
    run.write(tmp_path / "run")
    sounds = SoundEmbeddings.load(tmp_path / "run", "logmel_normalized")
    sounds.engines = {"espeak": WordEngine()}
    labels = sounds.speakers["label"].to_list()
    embedded = sounds.embed(sounds.words["arpabet"].to_list()[:4], labels)
    for w in range(4):
        for s in range(3):
            row = sounds._stored[(sounds.words["label"][w], labels[s], 1)]
            assert np.allclose(embedded[w, s], sounds.tokens[row], atol=1e-5)


@needs_audio
def test_the_evaluation_reports_every_embedding_with_and_without_normalization(tmp_path):
    config = held_config(tmp_path, [FIXED, NORMALIZED], closed_class=None)
    run = held_run(tmp_path, config)
    table = evaluate_embeddings(run.embeddings, run.lexicon.words, run.synthesis, run.streams.eval, config=config)  # fmt: skip
    rows = table.filter(pl.col("word_split") == "all")
    assert rows["embedding"].to_list() == ["logmel_fixed"] * 2 + ["logmel_normalized"] * 2
    assert rows["talker_normalized"].to_list() == [False, True, False, True]
    # ``configured`` marks the variant that the run stores
    assert rows["configured"].to_list() == [True, False, False, True]
    # both embeddings come from the same encoder, so their variants agree
    for column in ("ap_across_train", "ap_held_out", "fidelity_spearman", "robustness_ap"):
        values = rows[column].to_list()
        assert values[0] == pytest.approx(values[2], abs=1e-6)
        assert values[1] == pytest.approx(values[3], abs=1e-6)
    assert rows["ap_across_train"][0] != rows["ap_across_train"][1]


# ---------------------------------------------------------------------------------------------
# The CPC options
# ---------------------------------------------------------------------------------------------


@needs_audio
@needs_torch
def test_cpc_embedding_from_the_latents_or_the_context(tmp_path):
    latents = {**CPC, "name": "cpc_latents", "embedding_from": "latents"}
    batch = {**CPC, "name": "cpc_batch", "negatives_from": "clip"}
    run = held_run(tmp_path, held_config(tmp_path, [CPC, latents, batch], closed_class=None))
    assert run.embeddings["cpc"].tokens.shape[1] == 32  # the context: hidden
    assert run.embeddings["cpc_latents"].tokens.shape[1] == 16  # the latents: dims
    assert run.embeddings["cpc_batch"].tokens.shape[1] == 32
    for store in run.embeddings.values():
        assert np.isfinite(store.tokens).all()
        assert store.meta["training"]["losses"][-1] < store.meta["training"]["losses"][0]


@needs_torch
def test_cpc_negatives_from_the_clip_never_include_the_target():
    import torch

    from semantic_world.wordforms.encoders.learned import cpc_loss

    heads = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False)])
    with torch.no_grad():
        heads[0].weight.copy_(torch.eye(3))
    # each clip repeats one latent, so every within-clip negative equals the target: the loss
    # is then log(1 + negatives); a negative that could be the target itself would not change
    # that, so a clip of distinct latents checks the offsets instead
    z = torch.nn.functional.one_hot(torch.arange(6) % 3, 3).float().repeat(2, 1, 1)
    mask = torch.ones(2, 6, dtype=torch.bool)
    c = torch.roll(z, shifts=-1, dims=1) * 20.0  # the context points at the next latent
    loss = cpc_loss(z, c, mask, heads, 5, torch.Generator().manual_seed(0), "clip")
    assert torch.isfinite(loss)
    # frames past a short clip's end are never drawn
    short = mask.clone()
    short[1, 3:] = False
    z_nan = z.clone()
    z_nan[1, 3:] = float("nan")
    c_nan = c.clone()
    c_nan[1, 3:] = float("nan")
    assert torch.isfinite(cpc_loss(z_nan, c_nan, short, heads, 5, torch.Generator().manual_seed(0), "clip"))  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Reverberation
# ---------------------------------------------------------------------------------------------


@needs_pyroomacoustics
def test_reverberation_time_is_within_ten_percent_of_its_target():
    from semantic_world.wordforms.augment import (
        calibrated_absorption,
        reverberation_time,
        room_impulse_response,
    )

    rng = np.random.default_rng(3)
    errors = []
    for _ in range(40):
        dims = tuple(float(rng.uniform(3, 6)) for _ in range(3))
        rt60 = float(rng.uniform(0.2, 0.8))
        source = [float(rng.uniform(0.5, d - 0.5)) for d in dims]
        microphone = [float(rng.uniform(0.5, d - 0.5)) for d in dims]
        rir, absorption = room_impulse_response(16000, dims, rt60, source, microphone)
        assert absorption == calibrated_absorption(dims, rt60) and 0 < absorption < 1
        errors.append(abs(reverberation_time(rir, 16000) / rt60 - 1))
    errors = np.array(errors)
    assert np.mean(errors <= 0.10) >= 0.95 and errors.max() < 0.2
    # a longer reverberation time needs less absorbent walls
    assert calibrated_absorption((4.0, 5.0, 3.0), 0.8) < calibrated_absorption((4.0, 5.0, 3.0), 0.3)
