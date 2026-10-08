"""The command line of the world package.

python -m semantic_world.world define CONFIG [--seed N] [--out DIR]
python -m semantic_world.world view RUN_FOLDER --preset classic [--include A,B] [--out FILE]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.world.config import load_config
from semantic_world.world.errors import WorldError
from semantic_world.world.generate import define
from semantic_world.world.views import PRESETS, write_view


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_world.world",
        description="Generate a world definition, or write a view of a world run.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    define_parser = commands.add_parser("define", help="generate a world from a configuration")
    define_parser.add_argument("config", help="the YAML world configuration")
    define_parser.add_argument("--seed", type=int, default=None, help="override the master seed")
    define_parser.add_argument(
        "--out", default=None, help="the output folder (default: runs/world/<name>_seed<seed>)"
    )
    view_parser = commands.add_parser("view", help="write a view of a world run for a model")
    view_parser.add_argument("run", help="a world run folder")
    view_parser.add_argument("--preset", default="classic", choices=PRESETS)
    view_parser.add_argument(
        "--include", default=None, help="comma-separated label prefixes that narrow or add columns"
    )
    view_parser.add_argument(
        "--out", default=None, help="the CSV file (default: RUN/views/<preset>.csv)"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "define":
            return _define(args)
        return _view(args)
    except (ConfigError, GenerationError, WorldError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _define(args: argparse.Namespace) -> int:
    config = load_config(args.config, seed=args.seed)
    result = define(config)
    folder = result.write(args.out)
    event_types = result.event_types
    print(
        f"wrote {folder}: {len(result.taxonomy.instances)} entities, {len(result.fluents)} fluents "
        f"({len(result.fluents.derived)} derived), {len(event_types.unary)} one-place and "
        f"{len(event_types.binary)} two-place event types, "
        f"{len(event_types.constraints)} constraints; rule set {result.rule_set_id[:12]}"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


def _view(args: argparse.Namespace) -> int:
    include = [p for p in (args.include or "").split(",") if p]
    path = write_view(args.run, args.preset, include, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
