"""The levers of the two-place share: the default world with one setting changed at a time.

For each lever, the script defines the default world (``data/world/default.yaml``) with one
setting changed, on three seeds, and reports the share of two-place events among the events of
the statistics episodes (``episodes.two_place_share`` of ``world_stats.yaml``), the mean
one-place capacity rate (the share of entities that meet a one-place requirement, averaged over
the event types), and the mean two-place pair density (the share of ordered pairs of distinct
entities that meet a two-place requirement, averaged over the event types). The baseline row is
the default world itself. Each row is one ``define`` per seed, with the 1,000 statistics
episodes, so the whole table takes about half an hour on a laptop.

    python examples/two_place_levers.py [--seeds 1,2,3] [--only NAME,NAME] [--out FILE]

The levers (the stage a7a proposal, ``docs/proposals/2026-10-09-world-stage-a7a-decisions.md``,
has the table and a sentence on each):

- the number of event types of each kind (``event_types.unary.count``, the event-type tree);
- the number of constraints per two-place event type (the event-type features' count and
  expected count, the own constraint);
- the constraint-family weights;
- the complexity of the one-place requirements (``event_types.unary.rules``);
- the precondition settings (``literals``, ``feature_rate``);
- the scene size and ``events_per_step`` of the statistics episodes;
- ``event_type_weights`` of the statistics episodes.

The scene settings and the event-type weights belong to the corpus configuration; the script
passes them to the episode generator directly, as ``simulate --config`` would.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from semantic_world.world.config import Config, config_from_mapping
from semantic_world.world.episodes import (
    STATS_EPISODES,
    STATS_STREAM,
    EpisodeGenerator,
    EpisodeSettings,
    Relatedness,
)
from semantic_world.world.generate import define
from semantic_world.world.runtime import able_table

DEFAULT_WORLD = Path("data/world/default.yaml")

Lever = tuple[str, dict[str, Any], dict[str, Any]]
"""A lever: its name, the world-configuration override, and the episode-settings override."""

LEVERS: list[Lever] = [
    ("baseline (the default world)", {}, {}),
    # the number of event types of each kind
    ("unary.count: 10", {"event_types": {"unary": {"count": 10}}}, {}),
    ("unary.count: 40", {"event_types": {"unary": {"count": 40}}}, {}),
    (
        "binary.taxonomy: 3 superordinates, 2 to 3 each (the code's default, 7 event types)",
        {
            "event_types": {
                "binary": {"taxonomy": {"superordinates": 3, "depth": 2, "branching": [2, 3]}}
            }
        },
        {},
    ),
    (
        "binary.taxonomy: 5 superordinates, 8 each (40 event types)",
        {
            "event_types": {
                "binary": {"taxonomy": {"superordinates": 5, "depth": 2, "branching": 8}}
            }
        },
        {},
    ),
    # the number of constraints per two-place event type
    (
        "binary.features.expected_true: 1",
        {"event_types": {"binary": {"features": {"count": 12, "expected_true": 1}}}},
        {},
    ),
    (
        "binary.features.expected_true: 6",
        {"event_types": {"binary": {"features": {"count": 12, "expected_true": 6}}}},
        {},
    ),
    (
        "binary.features.count: 6, expected_true: 3",
        {"event_types": {"binary": {"features": {"count": 6, "expected_true": 3}}}},
        {},
    ),
    (
        "binary.own_constraint: true (the code's default)",
        {"event_types": {"binary": {"own_constraint": True}}},
        {},
    ),
    # the constraint-family weights
    (
        "binary.constraint_families: equal (the code's default)",
        {
            "event_types": {
                "binary": {
                    "constraint_families": {
                        "agent": 1,
                        "patient": 1,
                        "cross": 1,
                        "key_lock": 1,
                        "comparison": 1,
                    }
                }
            }
        },
        {},
    ),
    (
        "binary.constraint_families: no key_lock",
        {
            "event_types": {
                "binary": {
                    "constraint_families": {
                        "agent": 1,
                        "patient": 1,
                        "cross": 1,
                        "key_lock": 0,
                        "comparison": 1,
                    }
                }
            }
        },
        {},
    ),
    (
        "binary.constraint_families: comparison only",
        {
            "event_types": {
                "binary": {
                    "constraint_families": {
                        "agent": 0,
                        "patient": 0,
                        "cross": 0,
                        "key_lock": 0,
                        "comparison": 1,
                    }
                }
            }
        },
        {},
    ),
    (
        "binary.rules.operator_mix: equal (the taxonomy's default)",
        {"event_types": {"binary": {"rules": {"operator_mix": {"AND": 1, "OR": 1, "XOR": 1}}}}},
        {},
    ),
    (
        "binary.rules.arity: {1: 0.5, 2: 0.5}",
        {"event_types": {"binary": {"rules": {"arity": {1: 0.5, 2: 0.5}}}}},
        {},
    ),
    # the complexity of the one-place requirements
    (
        "unary.rules.operator_mix: equal (the taxonomy's default)",
        {"event_types": {"unary": {"rules": {"operator_mix": {"AND": 1, "OR": 1, "XOR": 1}}}}},
        {},
    ),
    (
        "unary.rules.operator_mix: AND only",
        {"event_types": {"unary": {"rules": {"operator_mix": {"AND": 1, "OR": 0, "XOR": 0}}}}},
        {},
    ),
    (
        "unary.rules.arity: {2: 0.2, 3: 0.4, 4: 0.4}",
        {"event_types": {"unary": {"rules": {"arity": {2: 0.2, 3: 0.4, 4: 0.4}}}}},
        {},
    ),
    # the precondition settings
    (
        "preconditions.literals: {0: 1} and feature_rate: 0 (no preconditions)",
        {"event_types": {"preconditions": {"literals": {0: 1.0}, "feature_rate": 0.0}}},
        {},
    ),
    (
        "preconditions.feature_rate: 0",
        {"event_types": {"preconditions": {"feature_rate": 0.0}}},
        {},
    ),
    (
        "preconditions.feature_rate: 0.6",
        {"event_types": {"preconditions": {"feature_rate": 0.6}}},
        {},
    ),
    (
        "preconditions.literals: {0: 0.1, 1: 0.4, 2: 0.5}",
        {"event_types": {"preconditions": {"literals": {0: 0.1, 1: 0.4, 2: 0.5}}}},
        {},
    ),
    # the scene size and events_per_step
    ("scene.size: [1, 3]", {}, {"size": (1, 3)}),
    ("scene.size: [6, 10]", {}, {"size": (6, 10)}),
    ("scene.events_per_step: 0.5", {}, {"events_per_step": 0.5}),
    ("scene.events_per_step: 4", {}, {"events_per_step": 4.0}),
    # event_type_weights
    ("scene.event_type_weights: two-place event types weighed 3", {}, {"two_place_weight": 3.0}),
    ("scene.event_type_weights: one-place event types weighed 3", {}, {"one_place_weight": 3.0}),
]


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def episode_share(result, config: Config, episode_override: dict[str, Any]) -> float:
    """The two-place share of the statistics episodes under other episode settings, run as
    ``define`` runs them (the ``world:stats`` stream, 1,000 episodes)."""
    definition = result.definition.runtime()
    kwargs = {k: v for k, v in episode_override.items() if k in ("size", "events_per_step")}
    weights: dict[str, float] = {}
    performable = [et.label for et in definition.event_types if not et.is_category]
    if "two_place_weight" in episode_override:
        weights = {
            label: episode_override["two_place_weight"]
            for label in performable
            if definition.event_type(label).arity == 2
        }
    if "one_place_weight" in episode_override:
        weights = {
            label: episode_override["one_place_weight"]
            for label in performable
            if definition.event_type(label).arity == 1
        }
    settings = EpisodeSettings(event_type_weights=weights, **kwargs)
    generator = EpisodeGenerator(
        definition, Relatedness.from_statics(result.statics, definition), settings
    )
    histories = generator.run(config.seed, STATS_EPISODES, STATS_STREAM)
    events = [e for h in histories for e in h.events]
    return sum(1 for e in events if e.patient is not None) / len(events) if events else 0.0


def measure(config: Config, episode_override: dict[str, Any]) -> dict[str, float]:
    result = define(config)
    definition = result.definition.runtime()
    table = able_table(definition)
    n = definition.entity_count
    one_place = [float(table[et.label].mean()) for et in definition.event_types if et.arity == 1]
    two_place = [
        float(table[et.label].sum() / (n * (n - 1)))
        for et in definition.event_types
        if et.arity == 2 and not et.is_category
    ]
    if episode_override:
        share = episode_share(result, config, episode_override)
    else:
        share = float(result.stats["episodes"]["two_place_share"])
    return {
        "share": share,
        "one_place": float(np.mean(one_place)) if one_place else 0.0,
        "two_place": float(np.mean(two_place)) if two_place else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", default="1,2,3", help="the world seeds, comma-separated")
    parser.add_argument(
        "--only", default=None, help="run only the levers whose name holds this text"
    )
    parser.add_argument("--out", default=None, help="write the Markdown table to this file too")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    base = yaml.safe_load(DEFAULT_WORLD.read_text(encoding="utf-8"))
    rows: list[str] = []
    header = (
        "| Lever | Two-place share (seeds "
        + ", ".join(str(s) for s in seeds)
        + ") | Mean | One-place capacity rate | Two-place pair density |"
    )
    rows.append(header)
    rows.append("| --- | --- | --- | --- | --- |")
    print(header, flush=True)
    for name, world_override, episode_override in LEVERS:
        if args.only and args.only not in name:
            continue
        shares, ones, twos = [], [], []
        started = time.time()
        for seed in seeds:
            data = merge(base, world_override)
            config = config_from_mapping(data, source=str(DEFAULT_WORLD), seed=seed)
            found = measure(config, episode_override)
            shares.append(found["share"])
            ones.append(found["one_place"])
            twos.append(found["two_place"])
        row = (
            f"| {name} | {', '.join(f'{s:.2f}' for s in shares)} | {np.mean(shares):.2f} | "
            f"{np.mean(ones):.2f} | {np.mean(twos):.3f} |"
        )
        rows.append(row)
        print(row, f"  ({time.time() - started:.0f}s)", flush=True)
    if args.out:
        Path(args.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
