"""The word-form pipeline: pseudowords, synthesis, auditory front ends, and sound embeddings.

See ``docs/specs/WORDFORM_PIPELINE.md``. Stages 1 to 4 provide the word forms, their audio, the
auditory front ends, and the sound embeddings with their evaluation and the ``SoundEmbeddings``
interface; stage 4a adds the closed-class forms (function words, affixes, and inflected forms);
stage 5 adds augmentation, Praat manipulation, and the modulation front end; stage 6 adds
learned encoders; and stage 7 adds sound-meaning assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_world.wordforms.config import Config, ConfigError, load_config
from semantic_world.wordforms.generate import (
    GenerationError,
    Lexicon,
    assign_word_splits,
    generate_lexicon,
)
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
        io.write_words(self.lexicon, folder / "words.csv", pos=self.config.assigns_lexemes)
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
    when the configuration has them. With the lexemes of a request, the lexemes are assigned to
    the content words, and the inflected forms that the request asks for are made from the forms
    the lexemes got. When the assignment's sound distance is an embedding's, the assignment
    waits for the embeddings (:func:`run_assignment`)."""
    from semantic_world.wordforms.assign import needs_embeddings
    from semantic_world.wordforms.closed_class import add_closed_class

    streams = Streams(config.seed)
    lexicon = generate_lexicon(config, streams.generate)
    add_closed_class(config, streams, lexicon)
    assign_word_splits(config, streams, lexicon)
    run = Run(config, streams, lexicon)
    if config.assigns_lexemes and not needs_embeddings(config):
        run_assignment(run)
    return run


def run_synthesis(run: Run, progress=None) -> Run:
    """Synthesize the run's word forms (layer 2), and augment the tokens when the configuration
    asks for it. Needs the ``speech`` extra's audio packages, and each configured engine's
    tool."""
    from semantic_world.wordforms.synth import synthesize_lexicon

    run.synthesis = synthesize_lexicon(
        run.config, run.streams, run.lexicon.words, progress=progress
    )
    if run.assignment is not None and run.config.assignment.mode == "acoustic_mapping":
        from semantic_world.wordforms.mapping import map_tokens

        # The meaning of a word shifts the sound of its tokens. The mapped tokens are the
        # word's tokens from here on, so the mapping comes before the augmentation.
        run.assignment.summary["acoustic_mapping"] = map_tokens(
            run.config, run.synthesis, run.assignment, progress=progress
        )
    if run.config.augmentation is not None:
        from semantic_world.wordforms.augment import augment_synthesis

        augment_synthesis(run.config, run.streams, run.synthesis, progress=progress)
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


def run_assignment(run: Run, out: str | Path | None = None) -> Run:
    """Assign the content words to the meanings of ``assignment.meanings``, in the configured
    mode. Without a meanings table, nothing is assigned. With branch markers, the marked forms
    join the run's word forms, so the assignment must come before the synthesis. With an
    embedding's distance as the sound distance, the embeddings are computed first.

    With the lexemes of a request, every lexeme gets a form, and the run has two passes: the
    assignment, and then the inflected forms of the lexemes' forms, which are synthesized and
    embedded with the rest. When the assignment needed the embeddings, the layers of the first
    pass are dropped, and the audio cache keeps the second pass cheap."""
    from semantic_world.wordforms.assign import assign, needs_embeddings

    config = run.config
    if run.assignment is not None or not (config.assigns_lexemes or config.meanings is not None):
        return run
    content = run.lexicon.content
    types = None
    if needs_embeddings(config):
        if run.embeddings is None:
            run_embeddings(run, out)
        rows = [i for i, word in enumerate(run.lexicon.words) if word.kind == "content"]
        types = run.embeddings[config.assignment.sound_distance].types[rows]
    if config.assigns_lexemes:
        from semantic_world.wordforms.closed_class import add_function_words
        from semantic_world.wordforms.lexemes import assign_lexemes, forms_to_avoid, inflect_lexemes

        run.assignment = assign_lexemes(config, run.lexicon, run.streams, types=types)
        if run.assignment.marked:
            run.lexicon.words = run.lexicon.words + run.assignment.marked
        if config.closed_class is not None:
            # The function words come after the assignment, so that no lexeme's word depends
            # on them, and before the inflected forms, in the run's usual order of forms.
            add_function_words(
                config, run.streams, run.lexicon, forms_to_avoid(run.lexicon, run.assignment)
            )
        inflect_lexemes(config, run.lexicon, run.assignment)
        # the layers of the first pass do not hold the forms that the assignment added
        run.synthesis = run.frontends = run.embeddings = run.evaluation = None
        return run
    run.assignment = assign(
        config,
        content,
        run.streams.assign,
        types=types,
        spellings={word.spelling for word in run.lexicon.words},
        others=[word for word in run.lexicon.words if word.kind != "content"],
        affixes=run.lexicon.affixes,
    )
    if run.assignment.marked:
        run.lexicon.words = run.lexicon.words + run.assignment.marked
    return run


def __getattr__(name: str):
    # SoundEmbeddings is loaded on first use, so that the word forms need no audio packages.
    if name == "SoundEmbeddings":
        from semantic_world.wordforms.embeddings import SoundEmbeddings

        return SoundEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
