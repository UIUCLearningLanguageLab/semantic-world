"""The word-form pipeline: pseudowords, synthesis, auditory front ends, and sound embeddings.

See ``docs/specs/WORDFORM_PIPELINE.md``. Stages 1 to 3 provide the word forms, their audio, and
the auditory front ends; the later layers are added stage by stage.
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
    "Streams",
    "generate_lexicon",
    "load_config",
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

    def summary(self) -> dict[str, Any]:
        summary = {"name": self.config.name, "seed": self.config.seed, **self.lexicon.summary()}
        if self.synthesis is not None:
            summary["synthesis"] = self.synthesis.summary()
        if self.frontends is not None:
            summary["frontends"] = {name: s.summary() for name, s in self.frontends.items()}
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
        models = self.synthesis.engines if self.synthesis is not None else None
        io.write_config(self.config, self.streams, folder / "config.yaml", models)
        io.write_words(self.lexicon, folder / "words.csv")
        if self.synthesis is not None:
            io.write_speakers(self.synthesis, folder / "speakers.csv")
            io.write_tokens(self.synthesis, folder / "tokens.csv")
        io.write_summary(self.summary(), folder / "summary.yaml")
        return folder


def run_forms(config: Config) -> Run:
    """Generate the word forms of a configuration."""
    streams = Streams(config.seed)
    lexicon = generate_lexicon(config, streams.generate)
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
