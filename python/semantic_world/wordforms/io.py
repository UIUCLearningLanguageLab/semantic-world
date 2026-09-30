"""Writing the output folder of a run.

A run writes one folder, by default ``runs/wordforms/<name>_seed<seed>/``. Stage 1 writes
``config.yaml`` (the resolved configuration with provenance) and ``words.csv``. Stage 2 adds
``speakers.csv`` and ``tokens.csv``; the audio itself stays in the shared cache. Real numbers are
written with 6 decimal places, and the same configuration, seed, and cache give byte-identical
files, apart from the provenance in ``config.yaml``.
"""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from semantic_world.wordforms.config import Config
from semantic_world.wordforms.generate import Lexicon
from semantic_world.wordforms.streams import Streams
from semantic_world.wordforms.synth import Synthesis

REPO_ROOT = Path(__file__).resolve().parents[3]
WORD_COLUMNS = (
    "label",
    "arpabet",
    "ipa",
    "espeak",
    "spelling",
    "syllables",
    "stress",
    "log_probability",
    "english_neighbors",
    "nearest_english",
    "lexicon_neighbors",
    "real_word",
)
SPEAKER_COLUMNS = ("label", "engine", "voice", "speaker_id", "variant", "pitch", "rate", "split")
TOKEN_COLUMNS = (
    "label",
    "word",
    "speaker",
    "engine",
    "phonemes",
    "settings",
    "rate_factor",
    "pitch_semitones",
    "augmentation",
    "duration",
    "tries",
    "peak",
    "rms_db",
    "cache_path",
    "sha256",
)
PACKAGES = (
    "numpy",
    "polars",
    "pyyaml",
    "cmudict",
    "wordfreq",
    "scipy",
    "soundfile",
    "piper-tts",
    "onnxruntime",
    "torch",
    "chcochleagram",
)


def default_output_dir(config: Config, base: str | Path = "runs/wordforms") -> Path:
    return Path(base) / f"{config.name}_seed{config.seed}"


def git_commit(root: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """The commit hash of the repository at ``root`` and whether tracked files have uncommitted
    changes, or ``(None, None)`` outside a repository."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        versions["semantic_world"] = metadata.version("semantic_world")
    except metadata.PackageNotFoundError:
        versions["semantic_world"] = None
    return versions


def provenance(streams: Streams, models: dict[str, Any] | None = None) -> dict[str, Any]:
    commit, dirty = git_commit()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": package_versions(),
        "models": models or {},
        "stream_seeds": streams.seeds(),
    }


def write_config(
    config: Config, streams: Streams, path: Path, models: dict[str, Any] | None = None
) -> None:
    """``models`` names every model used, with its version or file hash."""
    data = config.resolved()
    data["provenance"] = provenance(streams, models)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def words_frame(lexicon: Lexicon) -> pl.DataFrame:
    records = [w.record() for w in lexicon.words]
    frame = pl.DataFrame(records, schema_overrides={"log_probability": pl.Float64})
    return frame.select(WORD_COLUMNS)


def write_words(lexicon: Lexicon, path: Path) -> None:
    words_frame(lexicon).write_csv(path, float_precision=6)


def write_speakers(synthesis: Synthesis, path: Path) -> None:
    schema = {
        "label": pl.String,
        "engine": pl.String,
        "voice": pl.String,
        "speaker_id": pl.Int64,
        "variant": pl.String,
        "pitch": pl.Int64,
        "rate": pl.Int64,
        "split": pl.String,
    }
    frame = pl.DataFrame([s.record() for s in synthesis.speakers], schema=schema)
    frame.select(SPEAKER_COLUMNS).write_csv(path)


def write_tokens(synthesis: Synthesis, path: Path) -> None:
    schema = {
        "label": pl.String,
        "word": pl.String,
        "speaker": pl.String,
        "engine": pl.String,
        "phonemes": pl.String,
        "settings": pl.String,
        "rate_factor": pl.Float64,
        "pitch_semitones": pl.Float64,
        "augmentation": pl.String,
        "duration": pl.Float64,
        "tries": pl.Int64,
        "peak": pl.Float64,
        "rms_db": pl.Float64,
        "cache_path": pl.String,
        "sha256": pl.String,
    }
    frame = pl.DataFrame([t.record() for t in synthesis.tokens], schema=schema)
    frame.select(TOKEN_COLUMNS).write_csv(path, float_precision=6)


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
