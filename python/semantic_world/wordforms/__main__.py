"""The command line: ``python -m semantic_world.wordforms <subcommand> ...``.

Subcommands: ``forms CONFIG [--seed N] [--out DIR]`` generates the word forms, ``all`` runs every
built layer (in stage 1, the same as ``forms``), and ``check-ipa [--sample N] [--seed N]`` reports
the agreement between the IPA table and espeak-ng.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from semantic_world.wordforms import load_config, run_forms
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

    try:
        config = load_config(args.config, seed=args.seed)
        run = run_forms(config)
        folder = run.write(args.out)
    except (ConfigError, GenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = run.summary()
    print(
        f"wrote {folder}: {summary['words']} words ({summary['real_words']} real), "
        f"{summary['minimal_pairs']} minimal pairs, "
        f"{summary['rejections']['total']} candidates rejected"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
