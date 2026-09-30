"""The word-form pipeline: pseudowords, synthesis, auditory front ends, and sound embeddings.

See ``docs/specs/WORDFORM_PIPELINE.md``. Stages 1 to 4 provide the word forms, their audio, the
auditory front ends, and the sound embeddings with their evaluation and the ``SoundEmbeddings``
interface; stage 4a adds the closed-class forms (function words, affixes, and inflected forms);
the later stages add augmentation, learned encoders, and sound-meaning assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_world.wordforms.config import Config, ConfigError, load_config
from semantic_world.wordforms.generate import GenerationError, Lexicon, generate_lexicon
from semantic_world.wordforms.streams import Streams

__all__ = [
    "Config",
    "ConfigError",
    "GenerationError",
    "Lexicon",
    "Run",
    "SoundEmbeddings",
    "Streams",
    "generate_lexicon",
    "load_config",
    "run_assignment",
    "run_embeddings",
    "run_evaluation",
    "run_forms",
    "run_frontends",
    "run_synthesis",
]


@dataclass
class Run:
    """The results of a run so far: the configuration, its streams, the word table, and the
    synthesis and the stored front ends when those layers have run."""

    config: Config
    streams: Streams
    lexicon: Lexicon
    synthesis: Any = None
    frontends: dict[str, Any] | None = None
    embeddings: dict[str, Any] | None = None
    evaluation: Any = None
    """The evaluation table, a polars DataFrame."""
    assignment: Any = None

    def summary(self) -> dict[str, Any]:
        summary = {"name": self.config.name, "seed": self.config.seed, **self.lexicon.summary()}
        if self.synthesis is not None:
            summary["synthesis"] = self.synthesis.summary()
        if self.frontends is not None:
            summary["frontends"] = {name: s.summary() for name, s in self.frontends.items()}
        if self.embeddings is not None:
            summary["embeddings"] = {name: s.summary() for name, s in self.embeddings.items()}
        if self.assignment is not None:
            summary["assignment"] = self.assignment.summary
        return summary

    def folder(self, out: str | Path | None = None) -> Path:
        """The run's folder: ``out``, or ``runs/wordforms/<name>_seed<seed>``."""
        from semantic_world.wordforms import io

        return Path(out) if out is not None else io.default_output_dir(self.config)

    def write(self, out: str | Path | None = None) -> Path:
        """Write the run's folder and return its path."""
        from semantic_world.wordforms import io

        folder = self.folder(out)
        folder.mkdir(parents=True, exist_ok=True)
        models = dict(self.synthesis.engines) if self.synthesis is not None else None
        for name, store in (self.embeddings or {}).items():
            if store.meta.get("pretrained"):
                models[f"embedding:{name}"] = {
                    "model": store.meta["model"],
                    "revision": store.meta["revision"],
                    "pretrained": True,
                }
        io.write_config(self.config, self.streams, folder / "config.yaml", models)
        io.write_words(self.lexicon, folder / "words.csv")
        if self.config.closed_class is not None:
            io.write_affixes(self.lexicon, folder / "affixes.csv")
        if self.synthesis is not None:
            io.write_speakers(self.synthesis, folder / "speakers.csv")
            io.write_tokens(self.synthesis, folder / "tokens.csv")
        if self.evaluation is not None:
            from semantic_world.wordforms.evaluate import write_evaluation

            write_evaluation(self.evaluation, folder)
        if self.assignment is not None:
            self.assignment.write(folder)
        io.write_summary(self.summary(), folder / "summary.yaml")
        return folder


def run_forms(config: Config) -> Run:
    """Generate the word forms of a configuration: the content words, and the closed-class forms
    when the configuration has them."""
    from semantic_world.wordforms.closed_class import add_closed_class

    streams = Streams(config.seed)
    lexicon = generate_lexicon(config, streams.generate)
    add_closed_class(config, streams, lexicon)
    return Run(config, streams, lexicon)


def run_synthesis(run: Run, progress=None) -> Run:
    """Synthesize the run's word forms (layer 2). Needs the ``speech`` extra's audio packages,
    and each configured engine's tool."""
    from semantic_world.wordforms.synth import synthesize_lexicon

    run.synthesis = synthesize_lexicon(
        run.config, run.streams, run.lexicon.words, progress=progress
    )
    return run


def run_frontends(run: Run, out: str | Path | None = None, progress=None) -> Run:
    """Compute the run's front ends (layer 3) and store them in the run's folder, under
    ``frontends/<name>/``. Needs the synthesis."""
    from semantic_world.wordforms.frontends import compute_frontends

    if run.synthesis is None:
        run_synthesis(run)
    run.frontends = compute_frontends(run.config, run.synthesis, run.folder(out), progress)
    return run


def run_embeddings(
    run: Run, out: str | Path | None = None, progress=None, local_only: bool = False
) -> Run:
    """Compute the run's sound embeddings (layer 4) and store them in the run's folder, under
    ``embeddings/<name>/``. Needs the synthesis, and uses the stored front ends."""
    from semantic_world.wordforms.embeddings import compute_embeddings

    if run.frontends is None:
        run_frontends(run, out)
    run.embeddings = compute_embeddings(
        run.config,
        run.lexicon.words,
        run.synthesis,
        run.frontends,
        run.folder(out),
        progress,
        local_only,
    )
    return run


def run_evaluation(
    run: Run, out: str | Path | None = None, progress=None, local_only: bool = False
) -> Run:
    """Evaluate every embedding of the run: same-different average precision, phonological
    fidelity, and the layer sweep of pretrained models."""
    from semantic_world.wordforms.evaluate import evaluate_embeddings

    if run.embeddings is None:
        run_embeddings(run, out)
    run.evaluation = evaluate_embeddings(
        run.embeddings,
        run.lexicon.words,
        run.synthesis,
        run.streams.eval,
        config=run.config,
        local_only=local_only,
        progress=progress,
    )
    return run


def run_assignment(run: Run) -> Run:
    """Assign the content words to the meanings of ``assignment.meanings``. Without a meanings
    table, nothing is assigned."""
    from semantic_world.wordforms.assign import assign_arbitrary, load_meanings

    if run.config.assignment.meanings is None:
        return run
    meanings, features = load_meanings(run.config.assignment.meanings)
    run.assignment = assign_arbitrary(run.lexicon.content, meanings, features, run.streams.assign)
    return run


def __getattr__(name: str):
    # SoundEmbeddings is loaded on first use, so that the word forms need no audio packages.
    if name == "SoundEmbeddings":
        from semantic_world.wordforms.embeddings import SoundEmbeddings

        return SoundEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
