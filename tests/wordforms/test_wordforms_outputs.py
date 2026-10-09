"""The run folder and the command line."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import yaml
from wordforms_support import DATA, needs_cmudict, needs_wordfreq

from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.io import WORD_COLUMNS

pytestmark = [needs_cmudict, needs_wordfreq]


def test_forms_writes_the_run_folder(tmp_path):
    out = tmp_path / "run"
    assert main(["forms", str(DATA / "tiny.yaml"), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "affixes.csv",
        "config.yaml",
        "summary.yaml",
        "words.csv",
    ]
    words = pl.read_csv(out / "words.csv")
    assert words.columns == list(WORD_COLUMNS)
    content = words.filter(pl.col("kind") == "content")
    assert content.height == 20 and words.height > 35
    assert content["label"].to_list() == [f"WORD.{i}" for i in range(1, 21)]
    assert words["real_word"].dtype == pl.Boolean and not words["real_word"].any()
    assert words["log_probability"].dtype == pl.Float64
    assert words["long_synthesis"].null_count() == words.height  # unknown until Piper has run
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["name"] == "tiny" and config["seed"] == 1
    assert config["wordforms"]["count"] == 20
    assert config["synthesis"]["engines"]["piper"]["speakers"] == 3
    assert list(config["provenance"]["stream_seeds"]) == [
        f"wordforms:{n}"
        for n in (
            "generate",
            "speakers",
            "synthesis",
            "augment",
            "train",
            "assign",
            "eval",
            "closed_class",
        )
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
    check_labels(out)


FORM = r"(WORD\.\d+(\.MARKER\.\d+)?|FUNCWORD\.\d+)(\.AFFIX\.\d+)?"
TOKEN = FORM + r"\.SPEAKER\.\d+\.TOKEN\.\d+(\.MAPPED)?(\.AUGMENTED\.\d+)?"
LABEL_GRAMMAR = {
    "words.csv": {"label": FORM, "stem": FORM, "affix": r"AFFIX\.\d+"},
    "affixes.csv": {"label": r"AFFIX\.\d+"},
    "speakers.csv": {"label": r"SPEAKER\.\d+"},
    "tokens.csv": {"label": TOKEN, "word": FORM, "speaker": r"SPEAKER\.\d+"},
    "index.csv": {"token": TOKEN, "label": TOKEN},
    "lexicon.csv": {"word": FORM, "base_word": FORM, "marker": r"MARKER\.\d+"},
    "markers.csv": {"label": r"MARKER\.\d+"},
}
"""The labels of stage a6 of ``docs/specs/WORLD_AND_LANGUAGE.md``, by file and column."""


def check_labels(run: Path) -> int:
    """Every label column of every CSV file of a run matches the grammar of its kind of label
    (the labels of "Labels"); the number of labels checked."""
    checked = 0
    for path in sorted(run.rglob("*.csv")):
        grammar = LABEL_GRAMMAR.get(path.name)
        if grammar is None:
            continue
        table = pl.read_csv(path, infer_schema_length=0)
        for column, pattern in grammar.items():
            if column not in table.columns:
                continue
            for value in table[column].drop_nulls().to_list():
                assert re.fullmatch(pattern, value), f"{path.name} {column}: {value!r}"
                checked += 1
    assert checked, run
    return checked


def test_every_output_file_uses_the_labels_of_the_refactor(tmp_path):
    """Stage a6 of the world-and-language refactor: ``WORD.<n>``, ``SPEAKER.<n>``,
    ``WORD.<n>.SPEAKER.<m>.TOKEN.<k>``, ``FUNCWORD.<n>``, ``AFFIX.<n>``, ``WORD.<n>.AFFIX.<m>``,
    and nothing in the old forms (``W.12``, ``S.3``, ``W.12.S.3.2``, ``F.2``, ``AF.1``)."""
    out = tmp_path / "run"
    assert main(["forms", str(DATA / "tiny.yaml"), "--out", str(out)]) == 0
    words = pl.read_csv(out / "words.csv")
    assert words.filter(pl.col("kind") == "function")["label"][0] == "FUNCWORD.1"
    inflected = words.filter(pl.col("kind") == "inflected")
    assert inflected["label"][0] == "WORD.1.AFFIX.1"
    assert inflected["stem"][0] == "WORD.1" and inflected["affix"][0] == "AFFIX.1"
    assert pl.read_csv(out / "affixes.csv")["label"].to_list() == ["AFFIX.1", "AFFIX.2", "AFFIX.3"]
    assert check_labels(out) == words.height + inflected.height * 2 + 3
    old = re.compile(r"(?<![A-Z])(W|S|F|AF|M)\.\d")
    for path in out.rglob("*.csv"):
        assert not old.search(path.read_text()), path


def test_errors_exit_with_status_1(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("wordforms: {count: 0}\n")
    assert main(["forms", str(bad)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "wordforms.count" in err
