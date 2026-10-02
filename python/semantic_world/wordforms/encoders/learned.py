"""Learned encoders: trained on the world's own audio, on training speakers only, and frozen
afterwards (``docs/specs/WORDFORM_PIPELINE.md``, layer 4, stage 6).

- ``contrastive``: an acoustic word encoder trained so that tokens of the same word, from
  different speakers, lie close together (in the manner of Kamper, Wang, and Livescu, 2016). A
  stack of one-dimensional convolutions over a front end's frames, pooled over time (mean and
  maximum) and projected to the embedding. The loss is a supervised contrastive loss: for each
  token in a batch, the other tokens of the same word are the positives, and every other token
  is a negative. The encoder uses word identity as supervision.
- ``cpc``: a self-supervised encoder trained by prediction alone, in the manner of contrastive
  predictive coding (van den Oord, Li, and Vinyals, 2018), with no word labels. The same
  convolutions give a latent frame sequence, a GRU summarizes the past into a context, and the
  context predicts the latents ``steps_ahead`` frames ahead against negatives drawn from the
  batch (or, with ``negatives_from: clip``, from the same clip). A token's embedding is the
  mean of its context frames, like a pretrained model's pooled layer, or of its latent frames
  (``embedding_from``). In a comparison on the default run (September 30, 2026), the context
  with batch negatives gave the best word embeddings, 30 epochs were no better than 10, and no
  setting made the encoder good at telling words apart.

Training draws its batches from the ``wordforms:train`` stream and seeds PyTorch from it, and
PyTorch runs its deterministic algorithms, so training on the CPU with a fixed seed gives the
same weights and embeddings. The trained model is stored beside the embedding (``model.pt``)
with the frame statistics it standardizes its input with, so ``SoundEmbeddings.embed`` can run
the frozen encoder on new forms.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from semantic_world.wordforms.frontends import Frontend

MIN_FRAMES = 4
"""Clips with fewer frames than this are padded with zeros before encoding."""


@dataclass
class TrainingReport:
    """What a training run did."""

    kind: str
    device: str
    seed: int
    tokens: int
    epochs: int
    steps: int
    seconds: float
    losses: list[float] = field(default_factory=list)
    """The mean loss of each epoch."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "device": self.device,
            "seed": self.seed,
            "tokens": self.tokens,
            "epochs": self.epochs,
            "steps": self.steps,
            "seconds": round(self.seconds, 3),
            "losses": [round(x, 6) for x in self.losses],
        }


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "learned encoders need the torch package; install the 'speech' extra"
        ) from error
    return torch


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------


def build_model(kind: str, channels: int, settings: dict[str, Any]):
    """The PyTorch module of a learned encoder."""
    torch = _torch()
    nn = torch.nn

    class ConvStack(nn.Module):
        """Convolutions over time: channels by frames in, hidden by frames out."""

        def __init__(self) -> None:
            super().__init__()
            layers = []
            width = channels
            for _ in range(settings["layers"]):
                layers.append(
                    nn.Conv1d(width, settings["hidden"], settings["kernel"], padding="same")
                )
                width = settings["hidden"]
            self.layers = nn.ModuleList(layers)
            self.activation = nn.GELU()

        def forward(self, frames, mask):
            # frames: batch by time by channels; mask: batch by time (True where the clip is).
            # The frames past a clip's end are zeroed after every layer, so a clip gets the same
            # output in a padded batch as on its own.
            hidden = frames.transpose(1, 2)
            keep = mask.unsqueeze(1)
            for layer in self.layers:
                hidden = self.activation(layer(hidden)) * keep
            return hidden.transpose(1, 2)

    class Contrastive(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stack = ConvStack()
            self.projection = nn.Linear(2 * settings["hidden"], settings["dims"])

        def forward(self, frames, mask):
            hidden = self.stack(frames, mask)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean = hidden.sum(dim=1) / lengths
            largest = hidden.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            return self.projection(torch.cat([mean, largest], dim=1))

    class CPC(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stack = ConvStack()
            self.latent = nn.Linear(settings["hidden"], settings["dims"])
            self.context = nn.GRU(settings["dims"], settings["hidden"], batch_first=True)
            self.heads = nn.ModuleList(
                [
                    nn.Linear(settings["hidden"], settings["dims"], bias=False)
                    for _ in range(settings["steps_ahead"])
                ]
            )
            self.embedding_from = settings.get("embedding_from", "context")
            """What the embedding is the mean of: the context frames or the latents."""

        def latents(self, frames, mask):
            z = self.latent(self.stack(frames, mask))
            c, _ = self.context(z)
            return z, c * mask.unsqueeze(-1)

        def forward(self, frames, mask):
            z, c = self.latents(frames, mask)
            pooled = z * mask.unsqueeze(-1) if self.embedding_from == "latents" else c
            return pooled.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    return Contrastive() if kind == "contrastive" else CPC()


# ---------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------


def supervised_contrastive_loss(embeddings, words, temperature: float):
    """For each token with another token of its word in the batch: minus the mean log
    probability of picking one of those tokens, among every other token, by similarity."""
    torch = _torch()
    unit = torch.nn.functional.normalize(embeddings, dim=1)
    similarity = unit @ unit.T / temperature
    n = len(words)
    own = torch.eye(n, dtype=torch.bool, device=similarity.device)
    similarity = similarity.masked_fill(own, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positives = (words.unsqueeze(0) == words.unsqueeze(1)) & ~own
    count = positives.sum(dim=1)
    anchors = count > 0
    if not anchors.any():
        return similarity.new_zeros(())
    per_anchor = -(log_probability.masked_fill(~positives, 0.0).sum(dim=1) / count.clamp(min=1))
    return per_anchor[anchors].mean()


def cpc_loss(z, c, mask, heads, negatives: int, generator, negatives_from: str = "batch"):
    """The InfoNCE loss of predicting the latent ``k`` frames ahead from the context, for each
    prediction head, against ``negatives`` other latent frames. With ``negatives_from``
    ``batch`` they are drawn from every clip of the batch; with ``clip`` they are other frames
    of the same clip, so that the speaker's voice, which every frame of a clip shares, cannot
    tell the target from them."""
    torch = _torch()
    lengths = mask.sum(dim=1)
    valid = mask.reshape(-1).nonzero().squeeze(1)  # the frames of the clips, over the batch
    pool = z.reshape(-1, z.shape[-1])[valid]
    total = z.new_zeros(())
    terms = 0
    for k, head in enumerate(heads, start=1):
        if z.shape[1] <= k:
            continue
        prediction = head(c[:, :-k])  # batch by time-k by dims
        target = z[:, k:]
        keep = mask[:, k:] & mask[:, :-k]
        if not keep.any():
            continue
        clips, times = keep.nonzero(as_tuple=True)
        prediction = prediction[keep]
        target = target[keep]
        if negatives_from == "clip":
            # other frames of the same clip: the target's position plus a random offset, modulo
            # the clip's length, which never lands on the target itself
            span = (lengths[clips] - 1).clamp(min=1).unsqueeze(1)
            draws = torch.rand(len(prediction), negatives, generator=generator).to(span.device)
            offsets = 1 + (draws * span).long().clamp(max=span - 1)
            positions = (times.unsqueeze(1) + k + offsets) % lengths[clips].unsqueeze(1)
            others = z[clips.unsqueeze(1), positions]
        else:
            draws = torch.randint(len(pool), (len(prediction), negatives), generator=generator)
            others = pool[draws]
        candidates = torch.cat([target.unsqueeze(1), others], dim=1)
        scores = (candidates * prediction.unsqueeze(1)).sum(dim=-1)
        labels = torch.zeros(len(prediction), dtype=torch.long, device=scores.device)
        total = total + torch.nn.functional.cross_entropy(scores, labels)
        terms += 1
    return total / max(terms, 1)


# ---------------------------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------------------------


class FrameSource:
    """The frames of tokens, from a stored front end or computed from clips, standardized by
    channel with statistics from the training tokens."""

    def __init__(self, frames_of, mean: np.ndarray, std: np.ndarray) -> None:
        self.frames_of = frames_of
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def frames(self, index: int) -> np.ndarray:
        raw = np.asarray(self.frames_of(index), dtype=np.float32)
        if len(raw) < MIN_FRAMES:
            raw = np.concatenate([raw, np.zeros((MIN_FRAMES - len(raw), raw.shape[1]), np.float32)])
        return (raw - self.mean) / self.std

    def batch(self, indices, torch, device):
        """Padded frames (batch by time by channels) and the mask of real frames."""
        clips = [self.frames(int(i)) for i in indices]
        longest = max(len(c) for c in clips)
        frames = np.zeros((len(clips), longest, clips[0].shape[1]), dtype=np.float32)
        mask = np.zeros((len(clips), longest), dtype=bool)
        for i, clip in enumerate(clips):
            frames[i, : len(clip)] = clip
            mask[i, : len(clip)] = True
        return torch.from_numpy(frames).to(device), torch.from_numpy(mask).to(device)


def frame_statistics(frames_of, indices) -> tuple[np.ndarray, np.ndarray]:
    """The per-channel mean and standard deviation of the frames of some tokens."""
    total = None
    squares = None
    count = 0
    for i in indices:
        frames = np.asarray(frames_of(int(i)), dtype=np.float64)
        total = frames.sum(axis=0) if total is None else total + frames.sum(axis=0)
        squares = (frames**2).sum(axis=0) if squares is None else squares + (frames**2).sum(axis=0)
        count += len(frames)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-8))
    return mean, std


def contrastive_batches(rng: np.random.Generator, token_words, train_indices, batch_size: int):
    """Batches of ``batch_size`` training tokens: half as many words, drawn at random, with two
    tokens each (from different speakers when the word has them), so that every token has a
    positive in its batch."""
    by_word: dict[int, np.ndarray] = {}
    for i in train_indices:
        by_word.setdefault(int(token_words[i]), []).append(int(i))
    words = sorted(w for w, tokens in by_word.items() if len(tokens) >= 2)
    per_batch = max(1, batch_size // 2)
    while True:
        chosen = rng.choice(words, size=min(per_batch, len(words)), replace=False)
        batch = []
        for w in chosen:
            batch.extend(int(i) for i in rng.choice(by_word[w], size=2, replace=False))
        yield np.array(batch)


def shuffled_batches(rng: np.random.Generator, train_indices, batch_size: int):
    """Batches of training tokens in a fresh random order each epoch."""
    while True:
        order = rng.permutation(train_indices)
        for start in range(0, len(order), batch_size):
            yield order[start : start + batch_size]


# ---------------------------------------------------------------------------------------------
# Training and encoding
# ---------------------------------------------------------------------------------------------


def train_encoder(
    kind: str,
    settings: dict[str, Any],
    source: FrameSource,
    token_words: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    device: str = "cpu",
    progress=None,
    checkpoint=None,
) -> tuple[Any, TrainingReport]:
    """Train a learned encoder on the training tokens and return the frozen model with its
    report. ``seed`` seeds PyTorch and the batch draws. ``checkpoint(epoch, model)`` is called
    after every epoch with the model in evaluation mode."""
    torch = _torch()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    channels = source.frames(int(train_indices[0])).shape[1]
    model = build_model(kind, channels, settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    if kind == "contrastive":
        batches = contrastive_batches(rng, token_words, train_indices, settings["batch_size"])
    else:
        batches = shuffled_batches(rng, train_indices, settings["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_indices) / settings["batch_size"]))
    report = TrainingReport(kind, device, seed, len(train_indices), settings["epochs"], 0, 0.0)
    started = time.time()
    model.train()
    for epoch in range(settings["epochs"]):
        total = 0.0
        for _ in range(steps_per_epoch):
            indices = next(batches)
            frames, mask = source.batch(indices, torch, device)
            if kind == "contrastive":
                words = torch.from_numpy(token_words[indices]).to(device)
                loss = supervised_contrastive_loss(
                    model(frames, mask), words, settings["temperature"]
                )
            else:
                z, c = model.latents(frames, mask)
                loss = cpc_loss(
                    z,
                    c,
                    mask,
                    model.heads,
                    settings["negatives"],
                    generator,
                    settings.get("negatives_from", "batch"),
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            report.steps += 1
        report.losses.append(total / steps_per_epoch)
        if progress is not None:
            progress(epoch + 1, settings["epochs"], report.losses[-1])
        if checkpoint is not None:
            model.eval()
            checkpoint(epoch + 1, model)
            model.train()
    report.seconds = time.time() - started
    model.eval()
    return model, report


def encode_tokens(
    model, source: FrameSource, count: int, device: str, batch_size: int = 64, groups=None
):
    """The embedding of every token, in batches, from the frozen model. ``groups`` lists sets
    of token indices that are batched apart, each in its own order: a token's embedding can
    differ in its last bits with the other tokens of its batch, so tokens that must not depend
    on another group (the content words, on the inflected forms) get batches of their own. The
    default is one group of every token."""
    torch = _torch()
    if groups is None:
        groups = [np.arange(count)]
    outputs: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for group in groups:
            for start in range(0, len(group), batch_size):
                indices = [int(i) for i in group[start : start + batch_size]]
                frames, mask = source.batch(indices, torch, device)
                outputs.append(model(frames, mask).float().cpu().numpy())
                rows.append(np.asarray(indices, dtype=np.int64))
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.concatenate(outputs)
    tokens = np.empty_like(stacked)
    tokens[np.concatenate(rows)] = stacked
    return tokens


def save_model(path: Path, model, kind: str, settings: dict[str, Any], mean, std) -> None:
    torch = _torch()
    torch.save(
        {
            "kind": kind,
            "settings": settings,
            "channels": int(len(mean)),
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        },
        path,
    )


def load_model(path: Path, device: str = "cpu"):
    """The frozen model and its frame statistics."""
    torch = _torch()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["kind"], saved["channels"], saved["settings"])
    model.load_state_dict(saved["state_dict"])
    model.eval().to(device)
    return model, saved["mean"], saved["std"]


class LearnedEncoder:
    """A frozen learned encoder: a front end's frames through the trained model."""

    kind = "learned"

    def __init__(self, frontend: Frontend, model_path: Path, device: str = "cpu") -> None:
        self.frontend = frontend
        self.device = device
        self.model, mean, std = load_model(model_path, device)
        self.source = FrameSource(None, mean, std)

    def encode(self, clip: np.ndarray) -> np.ndarray:
        torch = _torch()
        frames = self.frontend.compute(clip)
        self.source.frames_of = lambda _index, frames=frames: frames
        batch, mask = self.source.batch([0], torch, self.device)
        with torch.no_grad():
            return self.model(batch, mask)[0].float().cpu().numpy()
