"""Sound embeddings as the input embeddings of a language model.

A plumbing demonstration, not an experiment. A small transformer language model from
``transformers`` takes the sound embedding of each word through a learned linear projection, in
place of an embedding table, using ``inputs_embeds``. The same projected sound embeddings serve
as the output matrix, with tied weights. With tied weights, the model can score a novel word that
it has never seen, from the word's sound alone.

The training sequences are random walks over a toy bigram model, because grammar comes later.

    python examples/wordforms_lm_inputs.py            # uses data/wordforms/tiny.yaml

The script builds the run (word forms, audio, front ends, embeddings) when the run's folder does
not hold the embedding yet. It needs the ``speech`` extra, and the synthesis engines to build the
run and to synthesize the novel word.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from semantic_world.wordforms import SoundEmbeddings
from semantic_world.wordforms.__main__ import main as wordforms_main


def ensure_run(config: str, run: Path, embedding: str) -> None:
    """Build the run when its folder does not hold the embedding."""
    if not (run / "embeddings" / embedding / "meta.yaml").exists():
        status = wordforms_main(["embed", config, "--out", str(run)])
        if status != 0:
            raise SystemExit(status)


def toy_bigram(words: int, rng: np.random.Generator) -> np.ndarray:
    """A transition matrix in which each word has three likely next words."""
    matrix = np.full((words, words), 0.02 / words)
    for word in range(words):
        successors = rng.choice(words, size=3, replace=False)
        matrix[word, successors] += (0.6, 0.28, 0.1)
    return matrix / matrix.sum(axis=1, keepdims=True)


def random_walks(matrix: np.ndarray, count: int, length: int, rng: np.random.Generator):
    walks = np.empty((count, length), dtype=np.int64)
    walks[:, 0] = rng.integers(len(matrix), size=count)
    for t in range(1, length):
        for i in range(count):
            walks[i, t] = rng.choice(len(matrix), p=matrix[walks[i, t - 1]])
    return torch.from_numpy(walks)


class SoundInputLM(torch.nn.Module):
    """A small GPT-2 whose input and output embeddings are projected sound embeddings."""

    def __init__(self, sound_dims: int, hidden: int = 64) -> None:
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        self.projection = torch.nn.Linear(sound_dims, hidden)
        self.transformer = GPT2Model(
            GPT2Config(vocab_size=1, n_positions=64, n_embd=hidden, n_layer=2, n_head=2)
        )

    def forward(self, ids: torch.Tensor, sounds: torch.Tensor) -> torch.Tensor:
        """Next-word logits over the rows of ``sounds`` at each position of ``ids``."""
        table = self.projection(sounds)  # words by hidden: the input and the output matrix
        hidden = self.transformer(inputs_embeds=table[ids]).last_hidden_state
        return hidden @ table.T


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="data/wordforms/tiny.yaml")
    parser.add_argument("--run", default=None, help="the run's folder")
    parser.add_argument("--embedding", default="cochleagram_fixed")
    parser.add_argument("--novel", default="Z AE1 M P IH0 K", help="a new word form, in ARPAbet")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from semantic_world.wordforms import load_config
    from semantic_world.wordforms.io import default_output_dir

    run = Path(args.run) if args.run else default_output_dir(load_config(args.config))
    ensure_run(args.config, run, args.embedding)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sounds = SoundEmbeddings.load(run, args.embedding)
    types = sounds.to_torch()["types"]
    mean, std = types.mean(dim=0), types.std(dim=0) + 1e-6
    table = (types - mean) / std  # standardized word embeddings: words by dimensions
    words = len(table)
    print(f"{words} words, sound embedding {sounds.name} with {sounds.dims} dimensions")

    matrix = toy_bigram(words, rng)
    entropy = float(-(matrix * np.log(matrix)).sum(axis=1).mean())
    model = SoundInputLM(sounds.dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss_on(walks: torch.Tensor) -> torch.Tensor:
        logits = model(walks[:, :-1], table)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, words), walks[:, 1:].reshape(-1)
        )

    first = last = float("nan")
    for step in range(args.steps):
        loss = loss_on(random_walks(matrix, 32, 17, rng))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"loss per word: {first:.3f} at the start, {last:.3f} after {args.steps} steps")
    print(
        f"(uniform guessing is {math.log(words):.3f}; the bigram model's entropy is {entropy:.3f})"
    )

    # A novel word: synthesize it, embed it, and score it with the tied output matrix.
    novel = torch.from_numpy(sounds.embed_types([args.novel]))
    extended = torch.cat([table, (novel - mean) / std])
    context = random_walks(matrix, 8, 6, rng)
    model.eval()
    with torch.no_grad():
        logits = model(context, extended)[:, -1]  # scores over the lexicon and the novel word
        log_probability = torch.log_softmax(logits, dim=-1)[:, -1].mean()
        as_input = torch.cat([context, torch.full((8, 1), words)], dim=1)
        after = torch.softmax(model(as_input, extended)[:, -1], dim=-1)
    print(
        f"novel word {args.novel!r}: mean log probability as the next word "
        f"{float(log_probability):.3f}; as an input, the model's next-word distribution has "
        f"entropy {float(-(after * torch.log(after)).sum(dim=-1).mean()):.3f}"
    )
    return {
        "first_loss": first,
        "last_loss": last,
        "uniform": math.log(words),
        "entropy": entropy,
        "novel_log_probability": float(log_probability),
    }


if __name__ == "__main__":
    main(sys.argv[1:])
