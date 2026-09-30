"""Layer 3: auditory front ends. Each front end turns a clip into a matrix of frames by channels.

- ``waveform``: the trimmed audio itself, one channel, for encoders that take waveforms.
- ``logmel``: a log-mel spectrogram, the standard input of speech recognition models.
- ``cochleagram``: a model of the cochlea: half-cosine filters spaced on the ERB scale, Hilbert
  envelopes, downsampling, and power-law compression. The lowest and highest channels are the
  low-pass and high-pass filters that complete the filter bank, so the center frequencies run
  from ``low_hz`` to ``high_hz``.

The cochleagram is a NumPy implementation of the filter bank of the McDermott lab's
``pycochleagram`` and ``chcochleagram`` packages (Feather et al.), written here so that the
pipeline needs neither package. A test compares it with saved outputs of ``chcochleagram``
(``tests/wordforms/fixtures/cochleagram_reference.npz``).

A front end's output depends only on the clip and the settings, never on the other clips of a
run, so a novel word gets the same frames as a stored word. Front ends run in NumPy on the CPU,
where the results are identical across runs.

Output is stored as one float32 array per front end (``frames.npy``), with all clips joined along
the frame axis, plus an index of each token's first frame and number of frames (``index.csv``)
and the settings (``meta.yaml``). The array loads with memory mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.wordforms.config import CochleagramConfig, Config, LogmelConfig
from semantic_world.wordforms.synth import Synthesis

LOG_FLOOR = 1e-10
MEL_LOW_HZ = 20.0
DOWNSAMPLING_WINDOW = 1001
ENVELOPE_FLOOR = 1e-16
"""The smallest squared envelope, as in chcochleagram."""


class Frontend:
    """A front end: ``compute`` maps a clip (mono float32) to frames by channels (float32)."""

    name: str
    channels: int
    frame_rate: float
    sample_rate: int

    def frame_count(self, samples: int) -> int:
        """The number of frames of a clip with ``samples`` samples."""
        raise NotImplementedError

    def compute(self, clip: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        """The settings that shape the output, for ``meta.yaml`` and for reuse checks."""
        raise NotImplementedError

    def meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channels": self.channels,
            "frame_rate": self.frame_rate,
            "sample_rate": self.sample_rate,
            "settings": self.settings(),
        }


class Waveform(Frontend):
    name = "waveform"
    channels = 1

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.frame_rate = float(sample_rate)

    def frame_count(self, samples: int) -> int:
        return samples

    def compute(self, clip: np.ndarray) -> np.ndarray:
        return np.asarray(clip, dtype=np.float32).reshape(-1, 1)

    def settings(self) -> dict[str, Any]:
        return {}


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int, low_hz: float, high_hz: float):
    """Triangular filters equally spaced on the mel scale: an array of bands by FFT bins, and
    the center frequency of each band."""
    edges = mel_to_hz(np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins[None, :] - lower) / (center - lower)
    falling = (upper - bins[None, :]) / (upper - center)
    return np.maximum(0.0, np.minimum(rising, falling)), edges[1:-1]


class LogMel(Frontend):
    """A log-mel spectrogram: Hann windows, a power spectrum, triangular mel filters from 20 Hz
    to half the sample rate, and the natural logarithm with a floor. Frame ``t`` is centered on
    sample ``(t + 1/2) * hop``, so a clip of ``n`` samples has ``ceil(n / hop)`` frames."""

    name = "logmel"

    def __init__(self, settings: LogmelConfig, sample_rate: int) -> None:
        self.config = settings
        self.sample_rate = sample_rate
        self.channels = settings.n_mels
        self.window = int(round(settings.window_ms * sample_rate / 1000.0))
        self.hop = int(round(settings.hop_ms * sample_rate / 1000.0))
        if self.hop < 1 or self.window < self.hop:
            raise ValueError("frontends.logmel: the window must be at least as long as the hop")
        self.frame_rate = sample_rate / self.hop
        self.n_fft = 1 << (self.window - 1).bit_length()
        self.taper = np.hanning(self.window + 1)[:-1]
        self.filters, self.centers = mel_filterbank(
            settings.n_mels, self.n_fft, sample_rate, MEL_LOW_HZ, sample_rate / 2.0
        )

    def frame_count(self, samples: int) -> int:
        return math.ceil(samples / self.hop)

    def compute(self, clip: np.ndarray) -> np.ndarray:
        count = self.frame_count(len(clip))
        left = (self.window - self.hop) // 2
        needed = (count - 1) * self.hop + self.window
        padded = np.zeros(needed, dtype=np.float64)
        padded[left : left + len(clip)] = clip[: needed - left]
        frames = np.lib.stride_tricks.sliding_window_view(padded, self.window)[:: self.hop]
        power = np.abs(np.fft.rfft(frames * self.taper, self.n_fft, axis=1)) ** 2
        return np.log(np.maximum(power @ self.filters.T, LOG_FLOOR)).astype(np.float32)

    def settings(self) -> dict[str, Any]:
        return {
            **self.config.resolved(),
            "window_samples": self.window,
            "hop_samples": self.hop,
            "n_fft": self.n_fft,
            "low_hz": MEL_LOW_HZ,
            "high_hz": self.sample_rate / 2.0,
            "log_floor": LOG_FLOOR,
            "center_hz": [round(float(c), 3) for c in self.centers],
        }


def hz_to_erb(hz):
    """Hz to the ERB-rate scale of Glasberg and Moore."""
    return 9.265 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / (24.7 * 9.265))


def erb_to_hz(erb):
    return 24.7 * 9.265 * (np.exp(np.asarray(erb, dtype=np.float64) / 9.265) - 1.0)


def erb_filterbank(size: int, sample_rate: int, channels: int, low_hz: float, high_hz: float):
    """Half-cosine filters for the real FFT of a signal of ``size`` samples: an array of
    channels by FFT bins, and the center frequency of each channel.

    ``channels - 2`` band-pass filters have their centers equally spaced on the ERB scale between
    ``low_hz`` and ``high_hz``, each reaching to its neighbors' centers. A low-pass filter below
    the first center and a high-pass filter above the last one complete the bank, so that the
    squared responses add to one.
    """
    bandpass = channels - 2
    if size % 2 == 0:
        bins, top = size // 2, sample_rate / 2.0
    else:
        bins, top = (size - 1) // 2, sample_rate * (size - 1) / 2.0 / size
    hz = np.linspace(0.0, top, bins + 1)
    erb = hz_to_erb(hz)
    cutoffs, spacing = np.linspace(
        hz_to_erb(low_hz), hz_to_erb(high_hz), bandpass + 2, retstep=True
    )
    centers = cutoffs[1:-1]
    filters = np.zeros((channels, bins + 1), dtype=np.float64)
    for i, center in enumerate(centers):
        inside = (erb > center - spacing) & (erb < center + spacing)
        filters[i + 1, inside] = np.cos((erb[inside] - center) / (2.0 * spacing) * np.pi)
    below = int(np.max(np.flatnonzero(hz < erb_to_hz(centers[0]))))
    filters[0, : below + 1] = np.sqrt(1.0 - filters[1, : below + 1] ** 2)
    above = int(np.min(np.flatnonzero(hz > erb_to_hz(centers[-1]))))
    filters[-1, above:] = np.sqrt(1.0 - filters[-2, above:] ** 2)
    center_erb = np.concatenate([[centers[0] - spacing], centers, [centers[-1] + spacing]])
    return filters, erb_to_hz(center_erb)


def downsampling_filter(step: int, window: int = DOWNSAMPLING_WINDOW) -> np.ndarray:
    """A sinc low-pass filter under a Kaiser window, for downsampling by ``step``."""
    times = np.arange(-window / 2.0, int(window / 2))
    return np.kaiser(window, 5.0) * np.sinc(times / step) / step


class Cochleagram(Frontend):
    """A cochleagram: ERB half-cosine filters applied to the clip's spectrum, the Hilbert
    envelope of each channel, downsampling to the frame rate, and power-law compression.

    The filters are built for one signal length, so a clip is padded with zeros to the next half
    second, and the frames beyond the clip are dropped. A clip of ``n`` samples has
    ``ceil(n * frame_rate / sample_rate)`` frames, and frame ``t`` is centered on sample
    ``(t + 1/2) * step``. The envelope is that of the one-sided spectrum, which is half the
    amplitude of a tone, as in ``chcochleagram``.
    """

    name = "cochleagram"

    def __init__(self, settings: CochleagramConfig, sample_rate: int) -> None:
        if sample_rate % settings.frame_rate != 0:
            raise ValueError(
                f"frontends.cochleagram.frame_rate: {settings.frame_rate} must divide the sample "
                f"rate {sample_rate}"
            )
        if settings.high_hz > sample_rate / 2:
            raise ValueError(
                f"frontends.cochleagram.high_hz: {settings.high_hz} is above half the sample "
                f"rate {sample_rate}"
            )
        if settings.channels < 3:
            raise ValueError("frontends.cochleagram.channels: must be at least 3")
        self.config = settings
        self.sample_rate = sample_rate
        self.channels = settings.channels
        self.frame_rate = float(settings.frame_rate)
        self.step = sample_rate // settings.frame_rate
        self.bucket = max(self.step, (sample_rate // 2) // self.step * self.step)
        self.lowpass = downsampling_filter(self.step)
        self._filters: dict[int, np.ndarray] = {}
        self.centers = [
            round(float(c), 3)
            for c in erb_filterbank(
                self.bucket, sample_rate, self.channels, settings.low_hz, settings.high_hz
            )[1]
        ]
        """The center frequency of each channel, in Hz."""

    def frame_count(self, samples: int) -> int:
        return math.ceil(samples / self.step)

    def filters(self, size: int) -> np.ndarray:
        """The filter bank for signals of ``size`` samples, built once for each size."""
        bank = self._filters.get(size)
        if bank is None:
            bank, _ = erb_filterbank(
                size, self.sample_rate, self.channels, self.config.low_hz, self.config.high_hz
            )
            self._filters[size] = bank
        return bank

    def compute(self, clip: np.ndarray) -> np.ndarray:
        count = self.frame_count(len(clip))
        size = max(1, math.ceil(len(clip) / self.bucket)) * self.bucket
        padded = np.zeros(size, dtype=np.float64)
        padded[: len(clip)] = clip
        # Filter in the frequency domain, and keep only the positive frequencies: the inverse
        # transform is then the analytic signal, whose magnitude is the envelope.
        subbands = np.fft.rfft(padded)[None, :] * self.filters(size)
        spectrum = np.zeros((self.channels, size), dtype=np.complex128)
        spectrum[:, : subbands.shape[1]] = subbands
        analytic = np.fft.ifft(spectrum, axis=1)
        envelopes = np.sqrt(np.maximum(analytic.real**2 + analytic.imag**2, ENVELOPE_FLOOR))
        # Downsample: a low-pass filter centered on each frame.
        window = len(self.lowpass)
        total = (math.ceil(size / self.step) - 1) * self.step + window - size
        envelopes = np.pad(envelopes, ((0, 0), (total // 2, total - total // 2)))
        windows = np.lib.stride_tricks.sliding_window_view(envelopes, window, axis=1)
        windows = np.ascontiguousarray(windows[:, :: self.step][:, :count])
        frames = (windows.reshape(-1, window) @ self.lowpass).reshape(self.channels, count)
        compressed = np.maximum(frames, 0.0) ** self.config.compression
        return np.ascontiguousarray(compressed.T, dtype=np.float32)

    def settings(self) -> dict[str, Any]:
        return {
            **self.config.resolved(),
            "implementation": "numpy",
            "filters": "half-cosine, ERB scale, with low-pass and high-pass end filters",
            "envelope": "Hilbert",
            "downsampling": f"sinc with a Kaiser window, {DOWNSAMPLING_WINDOW} samples",
            "padding": f"zeros to a multiple of {self.bucket} samples",
            "center_hz": self.centers,
        }


def make_frontends(config: Config) -> dict[str, Frontend]:
    """The front ends of a configuration, by name: ``waveform`` always, and the others that are
    configured."""
    rate = config.synthesis.sample_rate
    frontends: dict[str, Frontend] = {"waveform": Waveform(rate)}
    if config.frontends.logmel is not None:
        frontends["logmel"] = LogMel(config.frontends.logmel, rate)
    if config.frontends.cochleagram is not None:
        frontends["cochleagram"] = Cochleagram(config.frontends.cochleagram, rate)
    return frontends


# ---------------------------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------------------------


@dataclass
class FrontendStore:
    """A stored front end: the joined frames of every token, with the index."""

    folder: Path
    meta: dict[str, Any]
    labels: list[str]
    starts: np.ndarray
    counts: np.ndarray
    reused: bool = False
    _frames: np.ndarray | None = None

    @classmethod
    def load(cls, folder: str | Path) -> FrontendStore:
        folder = Path(folder)
        meta = yaml.safe_load((folder / "meta.yaml").read_text(encoding="utf-8"))
        index = pl.read_csv(folder / "index.csv")
        return cls(
            folder=folder,
            meta=meta,
            labels=index["token"].to_list(),
            starts=index["first_frame"].to_numpy().astype(np.int64),
            counts=index["frames"].to_numpy().astype(np.int64),
        )

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def frames(self) -> np.ndarray:
        """All frames, memory-mapped: total frames by channels."""
        if self._frames is None:
            self._frames = np.load(self.folder / "frames.npy", mmap_mode="r")
        return self._frames

    def position(self, label: str) -> int:
        if not hasattr(self, "_positions"):
            self._positions = {name: i for i, name in enumerate(self.labels)}
        return self._positions[label]

    def token_frames(self, label: str) -> np.ndarray:
        """The frames of one token: frames by channels."""
        i = self.position(label)
        return self.frames[self.starts[i] : self.starts[i] + self.counts[i]]

    def __len__(self) -> int:
        return len(self.labels)

    def summary(self) -> dict[str, Any]:
        return {
            "tokens": len(self.labels),
            "frames": int(self.counts.sum()),
            "channels": self.meta["channels"],
            "frame_rate": self.meta["frame_rate"],
            "reused": self.reused,
        }


def fingerprint(frontend: Frontend, synthesis: Synthesis) -> str:
    """A hash of the front end's settings and of every token's label and audio. A stored front
    end with the same fingerprint is up to date."""
    digest = hashlib.sha256()
    digest.update(json.dumps(frontend.meta(), sort_keys=True).encode("utf-8"))
    for token in synthesis.tokens:
        digest.update(f"{token.label}:{token.sha256}\n".encode())
    return digest.hexdigest()


def write_frontend(
    frontend: Frontend,
    synthesis: Synthesis,
    folder: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> FrontendStore:
    """Compute a front end for every token and store it in ``folder``. When the folder already
    holds the same front end for the same audio, nothing is computed."""
    folder = Path(folder)
    mark = fingerprint(frontend, synthesis)
    if all((folder / name).exists() for name in ("frames.npy", "index.csv", "meta.yaml")):
        store = FrontendStore.load(folder)
        if store.meta.get("fingerprint") == mark:
            store.reused = True
            return store
    folder.mkdir(parents=True, exist_ok=True)
    samples = [int(round(t.duration * synthesis.sample_rate)) for t in synthesis.tokens]
    counts = np.array([frontend.frame_count(n) for n in samples], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    total = int(counts.sum())
    (folder / "meta.yaml").unlink(missing_ok=True)  # an unfinished write is never up to date
    frames = np.lib.format.open_memmap(
        folder / "frames.npy", mode="w+", dtype=np.float32, shape=(total, frontend.channels)
    )
    for i, token in enumerate(synthesis.tokens):
        output = frontend.compute(synthesis.audio(token))
        if output.shape != (counts[i], frontend.channels):
            raise RuntimeError(
                f"{frontend.name}: {token.label} gave frames of shape {output.shape}, expected "
                f"{(int(counts[i]), frontend.channels)}"
            )
        frames[starts[i] : starts[i] + counts[i]] = output
        if progress is not None:
            progress(i + 1, len(synthesis.tokens))
    frames.flush()
    del frames
    labels = [t.label for t in synthesis.tokens]
    pl.DataFrame({"token": labels, "first_frame": starts, "frames": counts}).write_csv(
        folder / "index.csv"
    )
    meta = {**frontend.meta(), "tokens": len(labels), "frames": total, "fingerprint": mark}
    (folder / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return FrontendStore(folder, meta, labels, starts, counts)


def stored_frontends(config: Config) -> list[str]:
    """The names of the front ends that a run stores: ``logmel`` and ``cochleagram`` when
    configured, and ``waveform`` only when ``frontends.waveform.store`` is on, because stored
    waveforms repeat the audio cache."""
    names = ["waveform"] if config.frontends.store_waveform else []
    return names + [n for n in config.frontends.names if n != "waveform"]


def compute_frontends(
    config: Config,
    synthesis: Synthesis,
    folder: str | Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, FrontendStore]:
    """Compute and store the run's front ends under ``folder/frontends/<name>/``."""
    frontends = make_frontends(config)
    stores: dict[str, FrontendStore] = {}
    for name in stored_frontends(config):
        report = None if progress is None else (lambda i, n, name=name: progress(name, i, n))
        stores[name] = write_frontend(
            frontends[name], synthesis, Path(folder) / "frontends" / name, report
        )
    return stores
