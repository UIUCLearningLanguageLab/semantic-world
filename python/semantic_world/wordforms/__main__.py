"""The command line: ``python -m semantic_world.wordforms <subcommand> ...``.

Subcommands: ``forms CONFIG [--seed N] [--out DIR]`` generates the word forms; ``synth`` also
synthesizes them; ``frontends`` also computes the auditory front ends; ``embed`` also computes
the sound embeddings; ``eval`` also evaluates them; ``assign`` generates the word forms and
assigns them to meanings; ``all`` runs every layer;
``check-ipa [--sample N] [--seed N]`` reports the agreement between the IPA table and espeak-ng;
and ``check-whisper CONFIG`` reports how well a Whisper model recognizes real words synthesized
from their phonemes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from semantic_world.wordforms import (
    load_config,
    run_assignment,
    run_embeddings,
    run_evaluation,
    run_forms,
    run_frontends,
    run_synthesis,
)
from semantic_world.wordforms.assign import AssignmentError
from semantic_world.wordforms.config import ConfigError
from semantic_world.wordforms.generate import GenerationError
from semantic_world.wordforms.phonemes import IPA_TABLE, PhonemeTable, espeak_path, ipa_agreement


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_world.wordforms",
        description="Generate spoken word forms, audio, auditory representations, and embeddings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("forms", "generate the word forms"),
        ("synth", "generate the word forms and synthesize them"),
        ("frontends", "word forms, synthesis, and the auditory front ends"),
        ("embed", "word forms, synthesis, front ends, and the sound embeddings"),
        ("eval", "everything up to the evaluation of the embeddings"),
        ("assign", "word forms and their assignment to meanings"),
        ("all", "run every layer that is built"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("config", help="the YAML configuration file")
        sub.add_argument("--seed", type=int, default=None, help="override the master seed")
        sub.add_argument(
            "--out",
            default=None,
            help="the output folder (default: runs/wordforms/<name>_seed<seed>)",
        )
    check = subparsers.add_parser("check-ipa", help="compare the IPA table with espeak-ng")
    check.add_argument("--sample", type=int, default=500, help="the number of words to compare")
    check.add_argument("--seed", type=int, default=0, help="the seed of the sample")
    check.add_argument("--voice", default="en-us", help="the espeak-ng voice")
    whisper = subparsers.add_parser(
        "check-whisper", help="transcribe real words synthesized from their phonemes"
    )
    whisper.add_argument("config", help="the YAML configuration file")
    whisper.add_argument("--seed", type=int, default=None, help="override the master seed")
    whisper.add_argument("--words", type=int, default=200, help="the number of real words")
    whisper.add_argument("--speakers", type=int, default=3, help="speakers per engine")
    whisper.add_argument("--model", default="openai/whisper-small.en", help="the Whisper model")
    whisper.add_argument("--out", default=None, help="write the report to this YAML file")
    args = parser.parse_args(argv)

    if args.command == "check-ipa":
        if espeak_path() is None:
            print("error: espeak-ng is not installed (brew install espeak-ng)", file=sys.stderr)
            return 1
        report = ipa_agreement(
            PhonemeTable.load(IPA_TABLE), sample_size=args.sample, seed=args.seed, voice=args.voice
        )
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), end="")
        return 0

    if args.command == "check-whisper":
        from semantic_world.wordforms.transcribe import whisper_check

        try:
            config = load_config(args.config, seed=args.seed)
            report = whisper_check(
                config, words=args.words, speakers=args.speakers, model=args.model
            )
        except (ConfigError, GenerationError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        text = yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0

    try:
        config = load_config(args.config, seed=args.seed)
        run = run_forms(config)
        if args.command in ("synth", "frontends", "embed", "eval", "all"):
            run_synthesis(run, progress=_progress)
        if args.command in ("frontends", "embed", "eval", "all"):
            run_frontends(run, args.out, progress=_frontend_progress)
        if args.command in ("embed", "eval", "all"):
            run_embeddings(run, args.out, progress=_frontend_progress)
        if args.command in ("eval", "all"):
            run_evaluation(run, args.out, progress=_sweep_progress)
        if args.command in ("assign", "all"):
            run_assignment(run)
        folder = run.write(args.out)
    except (ConfigError, GenerationError, AssignmentError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = run.summary()
    print(
        f"wrote {folder}: {summary['words']} words ({summary['real_words']} real), "
        f"{summary['minimal_pairs']} minimal pairs, "
        f"{summary['rejections']['total']} candidates rejected"
    )
    if "synthesis" in summary:
        synthesis = summary["synthesis"]
        print(
            f"synthesis: {synthesis['tokens']} tokens from {synthesis['speakers']} speakers, "
            f"{synthesis['synthesized']} synthesized, {synthesis['read_from_cache']} read from "
            f"the cache"
        )
        check = synthesis["duration_check"]
        print(
            f"duration check: {check['retried']} clips tried again, "
            f"{check['still_over_limit']} still over the limit"
        )
    for name, frontend in summary.get("frontends", {}).items():
        state = "already stored" if frontend["reused"] else "computed"
        print(
            f"front end {name}: {frontend['frames']} frames of {frontend['channels']} channels "
            f"at {frontend['frame_rate']:g} per second, {state}"
        )
    for name, embedding in summary.get("embeddings", {}).items():
        state = "already stored" if embedding["reused"] else "computed"
        kind = "pretrained" if embedding["pretrained"] else embedding["encoder"]
        print(f"embedding {name} ({kind}): {embedding['dims']} dimensions, {state}")
    if run.evaluation is not None:
        print(f"evaluation: {folder / 'eval' / 'embeddings.csv'}")
        print(_evaluation_text(run.evaluation))
    if "assignment" in summary:
        assignment = summary["assignment"]
        print(
            f"assignment ({assignment['mode']}): {assignment['meanings']} meanings, "
            f"sound-meaning correlation {assignment['correlation']} "
            f"(null mean {assignment['null']['mean']}, p = {assignment['null']['p_value']})"
        )
    return 0


def _evaluation_text(table) -> str:
    """The stored embeddings' rows of the evaluation table, as aligned text: every embedding for
    all words, and again without the long_synthesis words when there are any."""
    lines = [
        f"  {'embedding':26s} {'words':>22s} {'dims':>5s}  {'within':>7s} {'across':>7s} "
        f"{'held-out':>8s}  {'spearman':>8s} {'auc':>6s}"
    ]

    def shown(value, width: int) -> str:
        return f"{'n/a':>{width}s}" if value is None else f"{value:{width}.3f}"

    for row in table.filter(table["basis"] == "stored").iter_rows(named=True):
        name = row["embedding"] + ("" if row["layer"] is None else f" (layer {row['layer']})")
        lines.append(
            f"  {name:26s} {row['word_set']:>22s} {row['dims']:5d}  "
            f"{shown(row['ap_within_speaker'], 7)} {shown(row['ap_across_train'], 7)} "
            f"{shown(row['ap_held_out'], 8)}  {shown(row['fidelity_spearman'], 8)} "
            f"{shown(row['fidelity_auc'], 6)}"
        )
    sweep = table.filter((table["basis"] == "sweep") & (table["word_set"] == "all"))
    for name in sweep["embedding"].unique(maintain_order=True):
        lines.append(f"  layer sweep of {name} (on the evaluation sample):")
        for row in sweep.filter(sweep["embedding"] == name).iter_rows(named=True):
            mark = "*" if row["configured"] else " "
            lines.append(
                f"   {mark}layer {row['layer']:2d} {'':38s}"
                f"{shown(row['ap_within_speaker'], 7)} {shown(row['ap_across_train'], 7)} "
                f"{shown(row['ap_held_out'], 8)}  {shown(row['fidelity_spearman'], 8)} "
                f"{shown(row['fidelity_auc'], 6)}"
            )
    return "\n".join(lines)


def _sweep_progress(name: str, done: int, total: int) -> None:
    if done % 1000 == 0 or done == total:
        print(f"  layer sweep of {name}: {done} of {total} tokens", file=sys.stderr)


def _frontend_progress(name: str, done: int, total: int) -> None:
    if done % 5000 == 0 or done == total:
        print(f"  {name}: {done} of {total} tokens", file=sys.stderr)


def _progress(done: int, total: int) -> None:
    if done % 1000 == 0 or done == total:
        print(f"  {done} of {total} tokens", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
