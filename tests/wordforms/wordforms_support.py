"""Shared paths and skip markers for the word-form pipeline's tests. Tests that need a missing
tool are skipped with a message naming the tool."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "wordforms"

needs_cmudict = pytest.mark.skipif(
    importlib.util.find_spec("cmudict") is None,
    reason="the cmudict package is not installed (the 'speech' extra)",
)

needs_wordfreq = pytest.mark.skipif(
    importlib.util.find_spec("wordfreq") is None,
    reason="the wordfreq package is not installed (the 'speech' extra)",
)

needs_espeak = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="espeak-ng is not installed (brew install espeak-ng)"
)

VOICE_DIR = REPO / "runs" / "wordforms" / "voices"
PIPER_VOICE = "en_US-libritts_r-medium"

needs_audio = pytest.mark.skipif(
    importlib.util.find_spec("soundfile") is None or importlib.util.find_spec("scipy") is None,
    reason="the soundfile and scipy packages are not installed (the 'speech' extra)",
)

needs_piper = pytest.mark.skipif(
    importlib.util.find_spec("piper") is None
    or not (VOICE_DIR / f"{PIPER_VOICE}.onnx").exists()
    or not (VOICE_DIR / f"{PIPER_VOICE}.onnx.json").exists(),
    reason=(
        "piper-tts or its voice is missing (the 'speech' extra, then: python -m "
        f"piper.download_voices {PIPER_VOICE} --download-dir runs/wordforms/voices)"
    ),
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _model_cached(model: str) -> bool:
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


def _whisper_cached() -> bool:
    return _model_cached("openai/whisper-small.en")


needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="the torch and transformers packages are not installed (the 'speech' extra)",
)

needs_hubert = pytest.mark.skipif(
    not _model_cached("facebook/hubert-base-ls960"),
    reason=(
        "torch, transformers, or the cached model facebook/hubert-base-ls960 is missing (run: "
        "python -m semantic_world.wordforms embed data/wordforms/default.yaml)"
    ),
)


needs_whisper = pytest.mark.skipif(
    not _whisper_cached(),
    reason=(
        "torch, transformers, or the cached model openai/whisper-small.en is missing (run: "
        "python -m semantic_world.wordforms check-whisper data/wordforms/default.yaml)"
    ),
)
