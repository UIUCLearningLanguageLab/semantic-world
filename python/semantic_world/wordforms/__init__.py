"""The word-form pipeline: pseudowords, synthesis, auditory front ends, and sound embeddings.

See ``docs/specs/WORDFORM_PIPELINE.md``. Stages 1 and 2 provide the word forms and their audio;
the later layers are added stage by stage.
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
    "run_synthesis",
]


@dataclass
class Run:
    """The results of a run so far: the configuration, its streams, the word table, and the
    synthesis when that layer has run."""

    config: Config
    streams: Streams
    lexicon: Lexicon
    synthesis: Any = None

    def summary(self) -> dict[str, Any]:
        summary = {"name": self.config.name, "seed": self.config.seed, **self.lexicon.summary()}
        if self.synthesis is not None:
            summary["synthesis"] = self.synthesis.summary()
        return summary

    def write(self, out: str | Path | None = None) -> Path:
        """Write the run's folder and return its path."""
        from semantic_world.wordforms import io

        folder = Path(out) if out is not None else io.default_output_dir(self.config)
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
