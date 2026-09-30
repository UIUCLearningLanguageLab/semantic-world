"""Make ``cochleagram_reference.npz``: the outputs of the McDermott lab's chcochleagram package
on a few test signals.

The pipeline's cochleagram (``semantic_world.wordforms.frontends.Cochleagram``) is a NumPy
implementation of the same filter bank. ``test_wordforms_frontends.py`` compares it with this
file, so neither the tests nor users need chcochleagram. This script is kept as the record of how
the file was made. It is not run by the tests. To run it again:

    uv pip install "chcochleagram @ git+https://github.com/jenellefeather/chcochleagram@7313877"
    python tests/wordforms/fixtures/make_cochleagram_reference.py CLIP_1.flac CLIP_2.flac

The two clips are synthesized words from the audio cache (mono, 16 kHz). The reference was made
on September 30, 2026, with chcochleagram at commit 731387778e40ee0298fdb13ac72789802f8e3cbb
(MIT license), torch 2.14, and the settings of each case below.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

RATE = 16000
WINDOW = 1001
DEFAULT = {"channels": 64, "low_hz": 50.0, "high_hz": 8000.0, "compression": 0.3, "frame_rate": 100}
OTHER = {"channels": 32, "low_hz": 100.0, "high_hz": 4000.0, "compression": 0.5, "frame_rate": 50}


def reference(clip: np.ndarray, settings: dict) -> np.ndarray:
    """The cochleagram of a clip from chcochleagram: the clip is padded with zeros to the next
    half second, and the frames beyond the clip are dropped."""
    import torch
    from chcochleagram import cochleagram, cochlear_filters, downsampling, envelope_extraction

    step = RATE // settings["frame_rate"]
    bucket = (RATE // 2) // step * step
    size = max(1, math.ceil(len(clip) / bucket)) * bucket
    padded = np.zeros(size, dtype=np.float32)
    padded[: len(clip)] = clip
    filters = cochlear_filters.ERBCosFilters(
        size,
        RATE,
        use_rfft=True,
        pad_factor=None,
        filter_kwargs={
            "n": settings["channels"] - 2,
            "low_lim": settings["low_hz"],
            "high_lim": settings["high_hz"],
            "sample_factor": 1,
            "full_filter": False,
        },
    )
    module = cochleagram.Cochleagram(
        filters,
        envelope_extraction.HilbertEnvelopeExtraction(size, RATE, use_rfft=True, pad_factor=None),
        downsampling.SincWithKaiserWindow(
            RATE,
            settings["frame_rate"],
            window_size=WINDOW,
            padding=downsampling.calculate_same_padding(size, WINDOW, stride=step),
        ),
        compression=None,
    )
    module.eval()
    with torch.no_grad():
        envelopes = module(torch.from_numpy(padded)[None, None, :])
        compressed = torch.clamp(envelopes, min=0.0) ** settings["compression"]
    frames = compressed[0, 0].T.contiguous().numpy()
    centers = np.asarray(filters.filter_extras["cf"], dtype=np.float64)
    return frames[: math.ceil(len(clip) / step)].astype(np.float32), centers


def tone(frequency: float, seconds: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def main(clip_paths: list[str]) -> None:
    import soundfile

    rng = np.random.default_rng(0)
    click = np.zeros(9600, dtype=np.float32)
    click[4800:4816] = 0.5
    cases = [
        ("tone_200hz", tone(200, 0.4), DEFAULT),
        ("tone_1000hz", tone(1000, 0.61), DEFAULT),
        ("tone_3000hz", tone(3000, 0.61), DEFAULT),
        ("white_noise", rng.normal(0, 0.05, 19200).astype(np.float32), DEFAULT),
        ("click", click, DEFAULT),
        ("silence", np.zeros(1600, dtype=np.float32), DEFAULT),
        ("white_noise_other_settings", rng.normal(0, 0.05, 12345).astype(np.float32), OTHER),
    ]
    for i, path in enumerate(clip_paths, start=1):
        clip, rate = soundfile.read(path, dtype="float32")
        assert rate == RATE and clip.ndim == 1
        cases.append((f"word_{i}", clip, DEFAULT))
    arrays = {}
    index = []
    for name, clip, settings in cases:
        frames, centers = reference(clip, settings)
        arrays[f"{name}_input"] = clip
        arrays[f"{name}_output"] = frames
        arrays[f"{name}_centers"] = centers
        index.append({"name": name, "settings": settings})
        print(name, clip.shape, frames.shape, float(frames.max()))
    arrays["index"] = np.array(json.dumps(index))
    np.savez_compressed(Path(__file__).with_name("cochleagram_reference.npz"), **arrays)


if __name__ == "__main__":
    main(sys.argv[1:])
