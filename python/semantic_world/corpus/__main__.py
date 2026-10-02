"""The command line: ``python -m semantic_world.corpus generate CONFIG [--seed N] [--out DIR]``.

``generate`` writes the corpus in the formal, the conceptual, and the propositional renderings,
with its test sets and its statistics. The ``render`` command, which attaches word forms, comes
with stage 7.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from semantic_world.corpus.config import ConfigError, load_config
from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.generate import generate
from semantic_world.taxonomy.errors import GenerationError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_world.corpus",
        description="Generate a corpus of documents about a taxonomy's world.",
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
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, seed=args.seed)
        corpus = generate(config)
        folder = corpus.write(args.out)
    except (ConfigError, CorpusError, GenerationError) as error:
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
