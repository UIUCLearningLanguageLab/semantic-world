"""The pretrained encoder: a speech model from Hugging Face ``transformers``.

The supported models are HuBERT, wav2vec 2.0, and WavLM (which take waveforms), and the Whisper
encoder (which takes a 30-second log-mel spectrogram). The output of one layer is mean-pooled
over the clip's frames. Layer 0 is the input to the transformer, and layer ``n`` is the output of
transformer layer ``n``.

Pretrained models were trained on human speech, so their embeddings stand for an adult English
listener, not for a learner of the world's language. Every output labels them as pretrained.

Each clip is run on its own, without padding against other clips, so a clip's embedding never
depends on the other clips of a run. The model runs in evaluation mode. On the CPU the results
are identical across runs, and on a GPU they match within a small tolerance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_SAMPLES = 400
"""The shortest input the waveform models accept. Shorter clips are padded with zeros."""
WHISPER_SAMPLES_PER_FRAME = 320


class PretrainedEncoder:
    kind = "pretrained"

    def __init__(
        self,
        model: str,
        layer: int,
        pooling: str = "mean",
        device: str = "auto",
        sample_rate: int = 16000,
        local_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        from transformers.utils import logging

        from semantic_world.wordforms.device import resolve_device

        logging.set_verbosity_error()
        logging.disable_progress_bar()
        if pooling != "mean":
            raise ValueError(f"unknown pooling {pooling!r}")
        self.torch = torch
        self.name = model
        self.sample_rate = sample_rate
        self.device = resolve_device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model, local_files_only=local_only)
        if getattr(self.extractor, "sampling_rate", sample_rate) != sample_rate:
            raise ValueError(
                f"{model} expects audio at {self.extractor.sampling_rate} Hz, not {sample_rate} Hz"
            )
        loaded = AutoModel.from_pretrained(model, local_files_only=local_only)
        self.revision = getattr(loaded.config, "_commit_hash", None)
        self.whisper = loaded.config.model_type == "whisper"
        self.model = (loaded.get_encoder() if self.whisper else loaded).eval().to(self.device)
        layers_key = "encoder_layers" if self.whisper else "num_hidden_layers"
        self.layers = int(getattr(loaded.config, layers_key)) + 1
        """The number of layer outputs: the transformer's input, then each transformer layer."""
        self.dims = int(getattr(loaded.config, "d_model", None) or loaded.config.hidden_size)
        if not 0 <= layer < self.layers:
            raise ValueError(f"{model} has layers 0 to {self.layers - 1}, not layer {layer}")
        self.layer = layer

    def all_layers(self, clip: np.ndarray) -> np.ndarray:
        """The mean-pooled output of every layer for one clip: layers by dimensions."""
        torch = self.torch
        clip = np.asarray(clip, dtype=np.float32)
        if self.whisper:
            features = self.extractor(clip, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = features.input_features.to(self.device)
            frames = max(1, math.ceil(len(clip) / WHISPER_SAMPLES_PER_FRAME))
        else:
            if len(clip) < MIN_SAMPLES:
                clip = np.concatenate([clip, np.zeros(MIN_SAMPLES - len(clip), dtype=np.float32)])
            features = self.extractor(clip, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = features.input_values.to(self.device)
            frames = None
        with torch.no_grad():
            states = self.model(inputs, output_hidden_states=True).hidden_states
            pooled = torch.stack([s[0, :frames].mean(dim=0) for s in states])
        return pooled.float().cpu().numpy()

    def encode(self, clip: np.ndarray) -> np.ndarray:
        """The embedding of one clip: the configured layer, mean-pooled."""
        return self.all_layers(clip)[self.layer]

    def provenance(self) -> dict[str, Any]:
        return {"model": self.name, "revision": self.revision, "pretrained": True}
