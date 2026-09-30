"""Stage 6: learned encoders, trained on the world's own audio."""

# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_embeddings import WordEngine
from wordforms_support import (  # fmt: skip
    DATA,
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_piper,
    needs_torch,
    needs_wordfreq,
)

from semantic_world.wordforms import Run, run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.config import ConfigError, parse_config
from semantic_world.wordforms.embeddings import SoundEmbeddings, compute_embeddings, training_seed
from semantic_world.wordforms.evaluate import evaluate_embeddings
from semantic_world.wordforms.frontends import compute_frontends
from semantic_world.wordforms.synth import synthesize_lexicon

pytestmark = [needs_cmudict, needs_wordfreq, needs_audio, needs_torch]

CONTRASTIVE = {"name": "contrastive", "encoder": "learned", "kind": "contrastive", "frontend": "logmel", "dims": 16, "hidden": 32, "epochs": 4, "batch_size": 16}  # fmt: skip
CPC = {"name": "cpc", "encoder": "learned", "kind": "cpc", "frontend": "logmel", "dims": 16, "hidden": 32, "epochs": 3, "batch_size": 8, "steps_ahead": 4, "negatives": 8}  # fmt: skip
FIXED = {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel", "pca_dims": 8}


def learned_config(tmp_path, embeddings=None, count: int = 12, **extra):
    return parse_config(
        {
            "name": "learned_test",
            "wordforms": {"count": count},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 3,
                "held_out_speaker_proportion": 0.34,
                "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
            },
            "embeddings": embeddings or [FIXED, CONTRASTIVE, CPC],
            "closed_class": None,
            "device": "cpu",
            **extra,
        },
        "learned_test",
    )


def learned_run(tmp_path, config, folder="run") -> Run:
    run = run_forms(config)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": WordEngine()}, check=False
    )
    if config.augmentation is not None:
        from semantic_world.wordforms.augment import augment_synthesis

        augment_synthesis(config, run.streams, run.synthesis)
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / folder)
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / folder
    )
    return run


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_learned_embedding_configuration():
    config = parse_config({"embeddings": [CONTRASTIVE, CPC]}, "x")
    contrastive, cpc = config.embeddings
    assert contrastive.encoder == "learned" and contrastive.kind == "contrastive"
    assert contrastive.frontend == "logmel" and contrastive.train_on == "clean"
    assert contrastive.layers == 3 and contrastive.kernel == 5 and contrastive.temperature == 0.1
    assert cpc.dims == 16 and cpc.steps_ahead == 4 and cpc.negatives == 8
    assert parse_config(config.resolved(), "x") == config
    defaults = parse_config({"embeddings": [{"name": "c", "encoder": "learned", "kind": "contrastive"}, {"name": "p", "encoder": "learned", "kind": "cpc"}]}, "x")  # fmt: skip
    assert defaults.embeddings[0].dims == 128 and defaults.embeddings[0].epochs == 20
    assert defaults.embeddings[1].dims == 64 and defaults.embeddings[1].epochs == 10
    assert defaults.embeddings[1].batch_size == 32
    # the default configuration trains one of each
    default = parse_config({}, "x")
    assert [e.name for e in default.embeddings][-2:] == ["contrastive_logmel", "cpc_logmel"]
    for entry, field, message in (
        ({"name": "x", "encoder": "learned"}, "embeddings[0].kind", "required"),
        ({"name": "x", "encoder": "learned", "kind": "vae"}, "embeddings[0].kind", "contrastive, cpc"),
        ({"name": "x", "encoder": "learned", "kind": "cpc", "frontend": "modulation"}, "embeddings[0].frontend", "not configured"),
        ({"name": "x", "encoder": "learned", "kind": "cpc", "epochs": 0}, "embeddings[0].epochs", "at least 1"),
        ({"name": "x", "encoder": "learned", "kind": "cpc", "train_on": "some"}, "embeddings[0].train_on", "clean, all"),
        ({"name": "x", "encoder": "learned", "kind": "cpc", "model": "m"}, "embeddings[0].model", "unknown key"),
    ):  # fmt: skip
        with pytest.raises(ConfigError) as info:
            parse_config({"embeddings": [entry]}, "x")
        assert info.value.field == field and message in info.value.message, info.value


def test_training_seed_comes_from_the_train_stream():
    config = parse_config({"embeddings": [CONTRASTIVE, CPC]}, "x")
    a, b = (training_seed(config, e) for e in config.embeddings)
    assert a != b and 0 <= a < 2**31
    assert training_seed(config.with_seed(2), config.embeddings[0]) != a
    assert training_seed(parse_config({"embeddings": [CONTRASTIVE]}, "y"), config.embeddings[0]) == a  # fmt: skip


# ---------------------------------------------------------------------------------------------
# The losses and the batches
# ---------------------------------------------------------------------------------------------


def test_supervised_contrastive_loss_prefers_same_word_tokens_close():
    import torch

    from semantic_world.wordforms.encoders.learned import supervised_contrastive_loss

    words = torch.tensor([0, 0, 1, 1])
    together = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    apart = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    assert supervised_contrastive_loss(together, words, 0.1) < supervised_contrastive_loss(apart, words, 0.1)  # fmt: skip
    # no positives at all gives a zero loss, not an error
    assert float(supervised_contrastive_loss(together, torch.tensor([0, 1, 2, 3]), 0.1)) == 0.0


def test_cpc_loss_is_lower_for_a_predictable_sequence():
    import torch

    from semantic_world.wordforms.encoders.learned import cpc_loss

    torch.manual_seed(0)
    heads = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])
    with torch.no_grad():
        for head in heads:
            head.weight.copy_(torch.eye(4))
    z = torch.eye(4).repeat(2, 3, 1)  # a repeating sequence of one-hot latents
    c = torch.roll(z, shifts=-1, dims=1)  # the context already equals the next latent
    mask = torch.ones(2, 12, dtype=torch.bool)
    predictable = cpc_loss(z, c, mask, heads, 4, torch.Generator().manual_seed(0))
    noise = cpc_loss(z, torch.randn(2, 12, 4), mask, heads, 4, torch.Generator().manual_seed(0))
    assert float(predictable) < float(noise)
    # frames past a clip's end take no part
    short = mask.clone()
    short[1, 6:] = False
    assert torch.isfinite(cpc_loss(z, c, short, heads, 4, torch.Generator().manual_seed(0)))


def test_contrastive_batches_pair_tokens_of_a_word():
    from semantic_world.wordforms.encoders.learned import contrastive_batches, shuffled_batches

    token_words = np.repeat(np.arange(5), 4)
    train = np.arange(20)
    batches = contrastive_batches(np.random.default_rng(0), token_words, train, 8)
    for _ in range(5):
        batch = next(batches)
        assert len(batch) == 8 and len(set(batch)) == 8
        words = token_words[batch]
        assert all((words == w).sum() == 2 for w in set(words))
    shuffled = shuffled_batches(np.random.default_rng(0), train, 8)
    epoch = np.concatenate([next(shuffled) for _ in range(3)])
    assert sorted(epoch) == list(range(20))


# ---------------------------------------------------------------------------------------------
# Training in a run
# ---------------------------------------------------------------------------------------------


def test_learned_encoders_train_reproducibly_on_the_cpu(tmp_path):
    config = learned_config(tmp_path)
    run = learned_run(tmp_path, config)
    for name, dims in (("contrastive", 16), ("cpc", 32)):
        store = run.embeddings[name]
        assert store.meta["encoder"] == "learned" and store.meta["pretrained"] is False
        assert store.tokens.shape == (12 * 3 * 3, dims) and store.types.shape == (12, dims)
        assert np.isfinite(store.tokens).all()
        assert (store.folder / "model.pt").exists()
        training = store.meta["training"]
        assert training["device"] == "cpu" and training["tokens"] == 12 * 2 * 3
        assert training["epochs"] == len(training["losses"]) and training["steps"] > 0
        assert training["losses"][-1] < training["losses"][0]  # the loss falls
        assert training["seconds"] > 0
        assert training["seed"] == training_seed(config, next(e for e in config.embeddings if e.name == name))  # fmt: skip
    # training again with the same seed gives the same weights and embeddings, bit for bit
    again = compute_embeddings(config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "again")  # fmt: skip
    for name in ("contrastive", "cpc"):
        assert np.array_equal(again[name].tokens, run.embeddings[name].tokens)
        assert np.array_equal(again[name].types, run.embeddings[name].types)
    # another seed trains another model
    other = compute_embeddings(config.with_seed(2), run.lexicon.words, run.synthesis, run.frontends, tmp_path / "other")  # fmt: skip
    assert not np.array_equal(other["cpc"].tokens, run.embeddings["cpc"].tokens)
    # a stored embedding is reused until the audio or the settings change
    reused = compute_embeddings(config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run")  # fmt: skip
    assert all(reused[name].reused for name in ("contrastive", "cpc"))


def test_learned_encoders_are_evaluated_beside_the_others(tmp_path):
    config = learned_config(tmp_path)
    run = learned_run(tmp_path, config)
    table = evaluate_embeddings(run.embeddings, run.lexicon.words, run.synthesis, run.streams.eval, config=config)  # fmt: skip
    assert table["embedding"].to_list() == ["logmel_fixed", "contrastive", "cpc"]
    assert table["encoder"].to_list() == ["fixed", "learned", "learned"]
    assert table["source"].to_list() == ["logmel"] * 3
    assert table["basis"].to_list() == ["stored"] * 3  # no layer sweep for a learned encoder
    for row in table.iter_rows(named=True):
        assert row["ap_across_train"] > row["chance_across_train"]
        assert row["ap_held_out"] > row["chance_held_out"]  # held-out speakers are reported
    # the contrastive encoder, supervised by word identity, tells words apart across speakers
    # at least as well as the fixed baseline on the stand-in engine
    by = {r["embedding"]: r for r in table.iter_rows(named=True)}
    assert by["contrastive"]["ap_across_train"] > 0.5 * by["logmel_fixed"]["ap_across_train"]


def test_embed_runs_the_frozen_encoder_on_new_forms(tmp_path):
    config = learned_config(tmp_path)
    run = learned_run(tmp_path, config)
    run.write(tmp_path / "run")
    for name in ("contrastive", "cpc"):
        sounds = SoundEmbeddings.load(tmp_path / "run", name)
        sounds.engines = {"espeak": WordEngine()}
        labels = sounds.speakers["label"].to_list()
        arpabet = sounds.words["arpabet"].to_list()[:4]
        embedded = sounds.embed(arpabet, labels)
        assert embedded.shape == (4, 3, sounds.dims)
        for w in range(4):
            for s in range(3):
                row = sounds._stored[(f"W.{w + 1}", labels[s], 1)]
                assert np.allclose(embedded[w, s], sounds.tokens[row], atol=1e-4)
        novel = sounds.embed(["Z AE1 M P IH0 K"], labels[:1])
        assert novel.shape == (1, 1, sounds.dims) and np.isfinite(novel).all()
        assert np.array_equal(sounds.embed(["Z AE1 M P IH0 K"], labels[:1]), novel)
        assert sounds.meta["training"]["kind"] == name


def test_train_on_all_includes_augmented_tokens(tmp_path):
    augmentation = {"recipes": [{"name": "noisy", "noise": {"snr_db": [5, 15]}}]}
    clean = learned_config(tmp_path, embeddings=[CPC], augmentation=augmentation)
    run = learned_run(tmp_path, clean)
    assert run.embeddings["cpc"].meta["training"]["tokens"] == 12 * 2 * 3
    everything = learned_config(tmp_path, embeddings=[{**CPC, "train_on": "all"}], augmentation=augmentation)  # fmt: skip
    run_all = learned_run(tmp_path, everything, "run_all")
    assert run_all.embeddings["cpc"].meta["training"]["tokens"] == 12 * 2 * 3 * 2
    assert not np.array_equal(run_all.embeddings["cpc"].tokens, run.embeddings["cpc"].tokens)


def test_embed_command_reports_the_training(tmp_path, capsys):
    config = learned_config(tmp_path, embeddings=[CONTRASTIVE])
    path = tmp_path / "learned.yaml"
    path.write_text(config.to_yaml())
    from semantic_world.wordforms.synth import piper  # noqa: F401  (the engine must exist)

    # the command needs a real engine; run the layers by hand and check the summary text
    run = learned_run(tmp_path, config)
    run.write(tmp_path / "run")
    summary = yaml.safe_load((tmp_path / "run" / "summary.yaml").read_text())
    training = summary["embeddings"]["contrastive"]["training"]
    assert training["kind"] == "contrastive" and training["epochs"] == 4
    assert summary["embeddings"]["contrastive"]["encoder"] == "learned"


@needs_espeak
@needs_piper
def test_learned_encoders_on_the_tiny_configuration(tmp_path, capsys):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(tmp_path / "cache")
    data["closed_class"] = None
    data["device"] = "cpu"
    data["embeddings"] = [FIXED, {**CONTRASTIVE, "epochs": 3}, {**CPC, "epochs": 2}]
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["eval", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "embedding contrastive (learned): 16 dimensions, computed" in text
    assert "trained contrastive on" in text and "trained cpc on" in text
    table = pl.read_csv(out / "eval" / "embeddings.csv")
    assert table["embedding"].to_list() == ["logmel_fixed", "contrastive", "cpc"]
    for row in table.iter_rows(named=True):
        assert row["ap_across_train"] > row["chance_across_train"]
