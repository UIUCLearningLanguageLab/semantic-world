"""Aligning sound embeddings with meaning vectors, in the manner of CLIP.

A plumbing demonstration, not an experiment. Each word form is assigned to a meaning (a category
of the taxonomy generator, with its binary feature vector) at random. Two linear projections map
token sound embeddings and meaning vectors into one space, and a contrastive loss pulls each
token toward its word's meaning. The script reports retrieval accuracy (which meaning is nearest
to a token?) for held-out speakers and for novel words.

Under an arbitrary assignment, the sound of a novel word says nothing about its meaning, so
retrieval for novel words should be near chance. Retrieval for held-out speakers should be well
above chance.

    python examples/wordforms_contrastive.py          # uses data/wordforms/tiny.yaml

The script builds the run when the run's folder does not hold the embedding yet. It needs the
``speech`` extra, and the synthesis engines to build the run.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from semantic_world.wordforms import SoundEmbeddings
from semantic_world.wordforms.__main__ import main as wordforms_main
from semantic_world.wordforms.assign import assign_arbitrary, load_meanings

# A taxonomy with 15 categories: 3 superordinates with 4 basic categories each.
TAXONOMY = {
    "name": "contrastive_example",
    "seed": 1,
    "features": {
        "property": {"count": 12, "proportion_determined": 0.25, "expected_true_free": 3},
        "part": {"count": 12, "proportion_determined": 0.25, "expected_true_free": 3},
    },
    "taxonomy": {"superordinates": 3, "depth": 2, "branching": 4},
    "instances": {"per_leaf": 1},
}


def ensure_run(config: str, run: Path, embedding: str) -> None:
    """Build the run when its folder does not hold the embedding."""
    if not (run / "embeddings" / embedding / "meta.yaml").exists():
        status = wordforms_main(["embed", config, "--out", str(run)])
        if status != 0:
            raise SystemExit(status)


def taxonomy_meanings(path: str | None) -> tuple[list[str], np.ndarray]:
    """The meanings: a ``categories_generative.csv`` file, or a small generated taxonomy."""
    if path is not None:
        return load_meanings(path)
    from semantic_world.taxonomy import generate, load_config

    with tempfile.TemporaryDirectory() as folder:
        config_path = Path(folder) / "taxonomy.yaml"
        config_path.write_text(yaml.safe_dump(TAXONOMY))
        generate(load_config(config_path)).write(Path(folder) / "out")
        return load_meanings(Path(folder) / "out" / "categories_generative.csv")


def retrieval_accuracy(sound: torch.Tensor, meaning: torch.Tensor, targets: torch.Tensor) -> float:
    """The proportion of tokens whose nearest meaning is their own."""
    if len(sound) == 0:
        return float("nan")
    return float(((sound @ meaning.T).argmax(dim=1) == targets).float().mean())


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="data/wordforms/tiny.yaml")
    parser.add_argument("--run", default=None, help="the run's folder")
    parser.add_argument("--embedding", default="cochleagram_fixed")
    parser.add_argument("--meanings", default=None, help="a categories_generative.csv file")
    parser.add_argument("--novel", type=int, default=3, help="meanings kept out of training")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from semantic_world.wordforms import load_config
    from semantic_world.wordforms.io import default_output_dir

    run = Path(args.run) if args.run else default_output_dir(load_config(args.config))
    ensure_run(args.config, run, args.embedding)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sounds = SoundEmbeddings.load(run, args.embedding)
    arrays = sounds.to_torch()
    tokens = arrays["tokens"]
    tokens = (tokens - tokens.mean(dim=0)) / (tokens.std(dim=0) + 1e-6)

    ids, features = taxonomy_meanings(args.meanings)
    # meanings go to content words; function words and inflected forms get none
    content = [form for form in sounds.lexicon_forms() if form.kind == "content"]
    assignment = assign_arbitrary(content, ids, features, rng)
    print(
        f"{len(ids)} meanings with {features.shape[1]} features, assigned at random to "
        f"{len(ids)} of {len(content)} content words (sound-meaning correlation "
        f"{assignment.summary['correlation']})"
    )
    word_rows = {label: i for i, label in enumerate(sounds.words["label"])}
    meaning_of_word = torch.full((len(sounds.words),), -1, dtype=torch.long)
    for m, word in enumerate(assignment.words):
        meaning_of_word[word_rows[word.label]] = m
    token_meaning = meaning_of_word[arrays["token_words"]]
    meanings = torch.from_numpy(features.astype(np.float32))

    novel = torch.arange(len(ids) - args.novel, len(ids))  # the last meanings are novel
    known = torch.arange(len(ids) - args.novel)
    has_meaning = token_meaning >= 0
    is_novel = torch.isin(token_meaning, novel)
    held_out = arrays["token_held_out"]
    train = has_meaning & ~is_novel & ~held_out
    print(
        f"training on {int(train.sum())} tokens of {len(known)} words from training speakers; "
        f"testing on {int((has_meaning & ~is_novel & held_out).sum())} tokens of held-out "
        f"speakers and {int(is_novel.sum())} tokens of {len(novel)} novel words"
    )

    sound_projection = torch.nn.Linear(tokens.shape[1], 32)
    meaning_projection = torch.nn.Linear(meanings.shape[1], 32)
    parameters = [*sound_projection.parameters(), *meaning_projection.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    normalize = torch.nn.functional.normalize
    train_rows = {int(m): torch.nonzero(train & (token_meaning == m)).flatten() for m in known}
    targets = torch.arange(len(known))
    for _ in range(args.steps):
        # one random token of each known meaning, so that a batch holds each meaning once
        batch = torch.stack([rows[rng.integers(len(rows))] for rows in train_rows.values()])
        sound = normalize(sound_projection(tokens[batch]), dim=1)
        meaning = normalize(meaning_projection(meanings[known]), dim=1)
        logits = sound @ meaning.T / 0.1
        loss = (
            torch.nn.functional.cross_entropy(logits, targets)
            + torch.nn.functional.cross_entropy(logits.T, targets)
        ) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        sound = normalize(sound_projection(tokens), dim=1)
        meaning = normalize(meaning_projection(meanings), dim=1)
        results = {}
        for name, rows, candidates in (
            ("training tokens", train, known),
            ("held-out speakers", has_meaning & ~is_novel & held_out, known),
            ("novel words", is_novel, novel),
        ):
            position = {int(m): i for i, m in enumerate(candidates)}
            wanted = torch.tensor([position[int(m)] for m in token_meaning[rows]], dtype=torch.long)
            accuracy = retrieval_accuracy(sound[rows], meaning[candidates], wanted)
            results[name] = accuracy
            print(
                f"retrieval accuracy, {name}: {accuracy:.3f} "
                f"(chance {1 / len(candidates):.3f}, {int(rows.sum())} tokens)"
            )
    return {"loss": loss.item(), **results}


if __name__ == "__main__":
    main(sys.argv[1:])
