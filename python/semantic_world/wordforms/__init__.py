"""The word-form pipeline: pseudowords, synthesis, auditory front ends, and sound embeddings.

See ``docs/specs/WORDFORM_PIPELINE.md``. Stage 1 provides the word forms; the later layers are
added stage by stage.
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
]


@dataclass
class Run:
    """The results of a run so far: the configuration, its streams, and the word table."""

    config: Config
    streams: Streams
    lexicon: Lexicon

    def summary(self) -> dict[str, Any]:
        return {"name": self.config.name, "seed": self.config.seed, **self.lexicon.summary()}

    def write(self, out: str | Path | None = None) -> Path:
        """Write the run's folder and return its path."""
        from semantic_world.wordforms import io

        folder = Path(out) if out is not None else io.default_output_dir(self.config)
        folder.mkdir(parents=True, exist_ok=True)
        io.write_config(self.config, self.streams, folder / "config.yaml")
        io.write_words(self.lexicon, folder / "words.csv")
        io.write_summary(self.summary(), folder / "summary.yaml")
        return folder


def run_forms(config: Config) -> Run:
    """Generate the word forms of a configuration."""
    streams = Streams(config.seed)
    lexicon = generate_lexicon(config, streams.generate)
    return Run(config, streams, lexicon)
