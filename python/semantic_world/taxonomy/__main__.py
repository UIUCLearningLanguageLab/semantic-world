"""The command line: ``python -m semantic_world.taxonomy CONFIG [--seed N] [--out DIR]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from semantic_world.taxonomy.config import ConfigError, load_config
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.generate import generate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_world.taxonomy",
        description="Generate a taxonomy of categories and instances with their features.",
    )
    parser.add_argument("config", help="the YAML configuration file")
    parser.add_argument("--seed", type=int, default=None, help="override the master seed")
    parser.add_argument(
        "--out", default=None, help="the output folder (default: runs/taxonomy/<name>_seed<seed>)"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, seed=args.seed)
        result = generate(config)
        folder = result.write(args.out)
    except (ConfigError, GenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = result.summary
    print(
        f"wrote {folder}: {summary['categories']} categories, {summary['leaves']} leaves, "
        f"{summary['instances']} instances, {len(result.rules.rules)} rules"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
