"""The espeak-ng engine: a rule-based formant synthesizer, called as an external program.

espeak-ng is GPL-3.0. The pipeline only runs the installed program (``brew install espeak-ng``)
and never links or vendors its code. Phonemes are given as espeak-ng mnemonics inside ``[[ ]]``,
so no spelling is involved.

The engine has no seed. Most voice variants give identical audio every time. The variants with
a ``breath`` setting (``f2``, ``f3``, and ``f5`` in espeak-ng 1.52) now and then give a slightly
different clip, so the cached audio, with its SHA-256 hash, is the reproducible artifact.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from typing import Any

import numpy as np

from semantic_world.wordforms.config import EspeakConfig
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth.speakers import Speaker


class EspeakEngine:
    name = "espeak"

    def __init__(self, settings: EspeakConfig) -> None:
        self.settings = settings
        self.program = shutil.which("espeak-ng")
        if self.program is None:
            raise RuntimeError("espeak-ng is not installed (brew install espeak-ng)")

    def version(self) -> str:
        result = subprocess.run([self.program, "--version"], capture_output=True, text=True)
        first = result.stdout.strip().split("  ")[0]
        return first.rsplit(" ", 1)[-1] if first else ""

    def phonemes(self, word: WordForm) -> str:
        return word.espeak

    def settings_for(self, speaker: Speaker, duration_scale: float) -> dict[str, Any]:
        """The engine settings of one clip. ``duration_scale`` stretches the clip in time."""
        return {
            "voice": f"{speaker.voice}+{speaker.variant}",
            "pitch": speaker.pitch,
            "rate": max(1, int(round(speaker.rate / duration_scale))),
        }

    def synthesize(self, phonemes: str, settings: dict[str, Any]) -> tuple[np.ndarray, int]:
        """The raw clip and its sample rate."""
        import soundfile

        result = subprocess.run(
            [
                self.program,
                "-v",
                settings["voice"],
                "-p",
                str(settings["pitch"]),
                "-s",
                str(settings["rate"]),
                "--stdout",
                f"[[{phonemes}]]",
            ],
            capture_output=True,
            check=True,
        )
        audio, rate = soundfile.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
        return audio, int(rate)

    def reset(self, seed: int) -> None:
        """Nothing to reset: the engine has no random state."""

    def provenance(self) -> dict[str, Any]:
        return {"program": "espeak-ng", "version": self.version()}
