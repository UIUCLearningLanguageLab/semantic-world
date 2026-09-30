"""The run folder and the command line."""

from __future__ import annotations

import polars as pl
import yaml
from wordforms_support import DATA, needs_cmudict, needs_wordfreq

from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.io import WORD_COLUMNS

pytestmark = [needs_cmudict, needs_wordfreq]


def test_forms_writes_the_run_folder(tmp_path):
    out = tmp_path / "run"
    assert main(["forms", str(DATA / "tiny.yaml"), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "summary.yaml", "words.csv"]
    words = pl.read_csv(out / "words.csv")
    assert words.columns == list(WORD_COLUMNS)
    assert words.height == 20
    assert words["label"].to_list() == [f"W.{i}" for i in range(1, 21)]
    assert words["real_word"].dtype == pl.Boolean and not words["real_word"].any()
    assert words["log_probability"].dtype == pl.Float64
    assert words["long_synthesis"].null_count() == 20  # unknown until Piper has synthesized
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["name"] == "tiny" and config["seed"] == 1
    assert config["wordforms"]["count"] == 20
    assert config["synthesis"]["engines"]["piper"]["speakers"] == 3
    assert list(config["provenance"]["stream_seeds"]) == [
        f"wordforms:{n}"
        for n in ("generate", "speakers", "synthesis", "augment", "pca", "train", "assign", "eval")
    ]
    assert "cmudict" in config["provenance"]["packages"]
    assert "wordfreq" in config["provenance"]["packages"]
    assert config["wordforms"]["english_min_zipf"] == 3.0
    assert "git_commit" in config["provenance"]
    summary = yaml.safe_load((out / "summary.yaml").read_text())
    assert summary["words"] == 20 and summary["rejections"]["total"] >= 0
    assert summary["english"]["english_min_zipf"] == 3.0
    assert summary["english"]["pattern_words"] < summary["english"]["words"]


def test_written_config_reloads(tmp_path):
    from semantic_world.wordforms.config import load_config

    out = tmp_path / "run"
    main(["forms", str(DATA / "tiny.yaml"), "--out", str(out)])
    again = load_config(out / "config.yaml")
    assert again.resolved() == load_config(DATA / "tiny.yaml").resolved()


def test_same_seed_gives_identical_files(tmp_path):
    main(["forms", str(DATA / "tiny.yaml"), "--out", str(tmp_path / "a")])
    main(["forms", str(DATA / "tiny.yaml"), "--out", str(tmp_path / "b")])
    main(["forms", str(DATA / "tiny.yaml"), "--seed", "2", "--out", str(tmp_path / "c")])
    assert (tmp_path / "a" / "words.csv").read_bytes() == (
        tmp_path / "b" / "words.csv"
    ).read_bytes()
    assert (tmp_path / "a" / "words.csv").read_bytes() != (
        tmp_path / "c" / "words.csv"
    ).read_bytes()
    assert (tmp_path / "a" / "summary.yaml").read_bytes() == (
        tmp_path / "b" / "summary.yaml"
    ).read_bytes()


def test_all_runs_the_built_layers(tmp_path):
    out = tmp_path / "run"
    assert main(["all", str(DATA / "tiny.yaml"), "--out", str(out)]) == 0
    assert (out / "words.csv").exists()


def test_errors_exit_with_status_1(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("wordforms: {count: 0}\n")
    assert main(["forms", str(bad)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "wordforms.count" in err
