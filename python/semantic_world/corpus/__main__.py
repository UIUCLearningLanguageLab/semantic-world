"""The command line of the corpus generator.

``python -m semantic_world.corpus generate CONFIG [--seed N] [--out DIR]`` writes the corpus in
the formal, the conceptual, and the propositional renderings, with its test sets, its
statistics, and the request for the word-form pipeline.

``python -m semantic_world.corpus render RUN_FOLDER --wordforms WORDFORM_RUN_FOLDER`` attaches
the word forms of a word-form run made from that request: the word labels, the spelled
rendering, and the word-form columns of ``lexicon.csv``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from semantic_world.corpus.config import ConfigError, load_config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.generate import generate
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.world.errors import WorldError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_world.corpus",
        description="Generate a corpus of documents about a world run.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "generate", help="write the documents, the test sets, and the statistics"
    )
    command.add_argument("config", help="the YAML configuration file")
    command.add_argument("--seed", type=int, default=None, help="override the corpus's master seed")
    command.add_argument(
        "--out", default=None, help="the output folder (default: runs/corpus/<name>_seed<seed>)"
    )
    attach = commands.add_parser("render", help="attach the word forms of a word-form run")
    attach.add_argument("run", help="the corpus run folder")
    attach.add_argument(
        "--wordforms", required=True, help="the word-form run folder, made from the request"
    )
    args = parser.parse_args(argv)
    if args.command == "render":
        from semantic_world.corpus.render import render

        try:
            report = render(args.run, args.wordforms)
        except CorpusError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(
            f"rendered {args.run}: {report['sentences']} sentences of {report['documents']} "
            f"documents and {report['test_items']} test items, with the word forms of "
            f"{report['wordforms']['name']} (seed {report['wordforms']['seed']})"
        )
        return 0
    try:
        config = load_config(args.config, seed=args.seed)
        corpus = generate(config)
        folder = corpus.write(args.out)
    except (ConfigError, CorpusError, GenerationError, WorldError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    stats = corpus.stats
    pairs = sum(len(test_set.pairs) for test_set in corpus.test_sets)
    print(
        f"wrote {folder}: {stats['documents']['count']} documents, "
        f"{stats['sentences']['count']} sentences, {stats['tokens']['count']} tokens, "
        f"{stats['scenes']['count']} scenes, {len(corpus.test_sets)} test sets with {pairs} pairs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
