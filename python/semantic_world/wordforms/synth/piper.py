"""The Piper engine: a neural synthesizer with a multi-speaker voice.

``piper-tts`` is GPL-3.0 (since version 1.3.0). It is an optional dependency, in the ``speech``
extra, and it is imported only inside this module. Phonemes are given as IPA inside ``[[ ]]``,
one character per Piper phoneme, so no spelling is involved. This module was written against
piper-tts 1.8 (``PiperVoice.load``, ``PiperVoice.synthesize``, and ``SynthesisConfig`` with
``noise_w_scale``).

The noise of a Piper voice is drawn inside the ONNX model. Two calls with the same settings give
different audio. A session created after ``onnxruntime.set_seed`` gives the same sequence of
clips each time, so a run from an empty cache is reproducible, but a single clip depends on the
clips synthesized before it. The cached audio, with its SHA-256 hash, is the reproducible
artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from semantic_world.wordforms.config import PiperConfig
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth.speakers import Speaker


def voice_paths(settings: PiperConfig) -> tuple[Path, Path]:
    """The voice's model file and its configuration file."""
    folder = Path(settings.voice_dir)
    return folder / f"{settings.voice}.onnx", folder / f"{settings.voice}.onnx.json"


def missing_voice_message(settings: PiperConfig) -> str:
    return (
        f"the Piper voice {settings.voice} is not in {settings.voice_dir}; download it with: "
        f"python -m piper.download_voices {settings.voice} --download-dir {settings.voice_dir}"
    )


def voice_speaker_count(settings: PiperConfig) -> int:
    """The number of speakers in the voice, read from the voice's configuration file."""
    _, config_path = voice_paths(settings)
    if not config_path.exists():
        raise RuntimeError(missing_voice_message(settings))
    return int(json.loads(config_path.read_text(encoding="utf-8")).get("num_speakers", 1))


class PiperEngine:
    name = "piper"

    def __init__(self, settings: PiperConfig, seed: int = 0) -> None:
        self.settings = settings
        self.model_path, self.config_path = voice_paths(settings)
        if not self.model_path.exists() or not self.config_path.exists():
            raise RuntimeError(missing_voice_message(settings))
        try:
            import piper  # noqa: F401
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise RuntimeError("piper-tts is not installed; install the 'speech' extra") from error
        self.voice_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.sample_rate = int(self.voice_config["audio"]["sample_rate"])
        self.voice = None
        self.reset(seed)

    def reset(self, seed: int) -> None:
        """Start a fresh session whose noise is seeded by ``seed``."""
        import onnxruntime
        from piper import PiperVoice

        onnxruntime.set_seed(seed % (2**31 - 1))
        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)

    def phonemes(self, word: WordForm) -> str:
        return word.ipa

    def settings_for(self, speaker: Speaker, duration_scale: float) -> dict[str, Any]:
        """The engine settings of one clip. ``duration_scale`` stretches the clip in time."""
        return {
            "voice": speaker.voice,
            "speaker_id": speaker.speaker_id,
            "noise_scale": self.settings.noise_scale,
            "length_scale": round(self.settings.length_scale * duration_scale, 6),
            "noise_w": self.settings.noise_w,
        }

    def synthesize(self, phonemes: str, settings: dict[str, Any]) -> tuple[np.ndarray, int]:
        """The raw clip and its sample rate."""
        from piper import SynthesisConfig

        missing = sorted({c for c in phonemes if c not in self.voice_config["phoneme_id_map"]})
        if missing:
            raise RuntimeError(f"the voice has no phoneme for {' '.join(missing)} in {phonemes}")
        synthesis = SynthesisConfig(
            speaker_id=settings["speaker_id"],
            length_scale=settings["length_scale"],
            noise_scale=settings["noise_scale"],
            noise_w_scale=settings["noise_w"],
            normalize_audio=False,
        )
        chunks = [
            c.audio_float_array for c in self.voice.synthesize(f"[[ {phonemes} ]]", synthesis)
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        return np.concatenate(chunks).astype(np.float32), self.sample_rate

    def provenance(self) -> dict[str, Any]:
        from importlib import metadata

        return {
            "package": "piper-tts",
            "version": metadata.version("piper-tts"),
            "voice": self.settings.voice,
            "voice_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
        }
