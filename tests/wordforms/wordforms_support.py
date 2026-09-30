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

needs_espeak = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="espeak-ng is not installed (brew install espeak-ng)"
)
