"""The command line of the world package.

python -m semantic_world.world define CONFIG [--seed N] [--out DIR]
python -m semantic_world.world view RUN_FOLDER --preset classic [--include A,B] [--out FILE]
python -m semantic_world.world simulate RUN_FOLDER --episodes N [--seed N] [--legal] [--out FILE]
    [--config CORPUS.yaml]
python -m semantic_world.world check-fixtures [FOLDER]
python -m semantic_world.world make-fixtures CONFIG [--out FOLDER] [--count N] [--steps N]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.world.config import load_config
from semantic_world.world.definition import load_definition
from semantic_world.world.episodes import load_scene_settings, simulate
from semantic_world.world.errors import WorldError
from semantic_world.world.fixtures import FIXTURES_DIR, check_fixtures, write_world_fixtures
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
    simulate_parser = commands.add_parser("simulate", help="run episodes of a world run")
    simulate_parser.add_argument("run", help="a world run folder")
    simulate_parser.add_argument("--episodes", type=int, required=True, help="how many episodes")
    simulate_parser.add_argument(
        "--seed", type=int, default=None, help="the master seed (default: the run's seed)"
    )
    simulate_parser.add_argument(
        "--legal", action="store_true", help="record each step's legal binding counts"
    )
    simulate_parser.add_argument(
        "--out", default=None, help="the JSON lines file (default: RUN/episodes.jsonl)"
    )
    simulate_parser.add_argument(
        "--config",
        default=None,
        help="a corpus configuration file whose scene block gives the settings",
    )
    check_parser = commands.add_parser(
        "check-fixtures", help="run the Python runtime on every conformance fixture"
    )
    check_parser.add_argument(
        "folder", nargs="?", default=str(FIXTURES_DIR), help="the folder of fixture JSON files"
    )
    make_parser = commands.add_parser(
        "make-fixtures", help="generate conformance fixtures from a world configuration"
    )
    make_parser.add_argument("config", help="the YAML world configuration (data/world/tiny.yaml)")
    make_parser.add_argument("--out", default=str(FIXTURES_DIR), help="the fixture folder")
    make_parser.add_argument(
        "--count", type=int, default=4, help="the number of fixtures without an error"
    )
    make_parser.add_argument("--steps", type=int, default=6, help="the steps of each fixture")
    args = parser.parse_args(argv)
    try:
        if args.command == "define":
            return _define(args)
        if args.command == "simulate":
            return _simulate(args)
        if args.command == "check-fixtures":
            return _check_fixtures(args)
        if args.command == "make-fixtures":
            return _make_fixtures(args)
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
    episodes = result.stats.get("episodes") or {}
    for label, count in (episodes.get("precondition_redraws") or {}).items():
        print(f"note: the preconditions of {label} were drawn again {count} time(s)")
    for warning in result.warnings:
        print(f"warning: {warning}")
    for warning in episodes.get("warnings") or ():
        print(f"warning: {warning}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    settings = None
    if args.config is not None:
        settings = load_scene_settings(args.config, load_definition(args.run))
    path, histories = simulate(args.run, args.episodes, args.seed, settings, args.legal, args.out)
    events = sum(len(h.events) for h in histories)
    quiescent = sum(1 for h in histories if h.quiescent)
    print(
        f"wrote {path}: {len(histories)} episodes, {events} events, {quiescent} quiescent, "
        f"policy {histories[0].policy if histories else '-'}"
    )
    return 0


def _check_fixtures(args: argparse.Namespace) -> int:
    reports = check_fixtures(args.folder)
    for report in reports:
        outcome = f"{report.error} error at the last step" if report.error else "no error"
        print(f"ok  {report.name}: {report.steps} steps, {outcome}")
    print(f"{len(reports)} fixtures passed the runtime and the brute-force evaluator")
    return 0


def _make_fixtures(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = define(config)
    paths = write_world_fixtures(result.definition, config.name, args.out, args.count, args.steps)
    for path in paths:
        print(f"wrote {path}")
    return 0


def _view(args: argparse.Namespace) -> int:
    include = [p for p in (args.include or "").split(",") if p]
    path = write_view(args.run, args.preset, include, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
