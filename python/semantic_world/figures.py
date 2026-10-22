"""The figures of the default and tiny runs: one snapshot for the tests, and the guides' figures.

python -m semantic_world.figures snapshot [--runs runs] [--snapshot tests/figures/snapshot.yaml]
python -m semantic_world.figures guides [--runs runs] [--snapshot ...] [--check]

A figure is a number, a label, or a short text that a run's files hold: ``config.yaml``,
``world_stats.yaml``, ``definition.json``, and ``derived/`` of a world run, and ``config.yaml``
and ``stats.yaml`` of a corpus run. ``world_figures`` and ``corpus_figures`` read them into one
mapping each. The snapshot (``tests/figures/snapshot.yaml``) holds the figures of the tiny and
default worlds and corpora, of the ``tiny_full`` and ``default_full`` worlds (the phase (c)
settings until stage c4, Jon's ruling WM.63; no corpus runs on them) and of the ``tiny_pairs``
world (stage c2a's world from an event file, whose identity must not move), and the baselines of
the settings that turn a feature off (the rule-set identity of each world without an event file
with the feature off, and the hash of the tiny corpus's ``documents.jsonl`` without conditional
effects). One test compares a regenerated run with the snapshot; ``snapshot`` rewrites the file
on purpose and prints what moved.

The guides quote figures between marker comments, ``<!--figure:KEY|FORMAT-->TEXT<!--/figure-->``,
where ``KEY`` is a dotted path into the figures (``default.corpus.sentences``,
``tiny.world.rule_set_short``, ``baselines.conditional_effects_off.default.rule_set_short``) and
``FORMAT`` an optional Python format specification (``,`` for a thousands separator, ``d`` for a
plain integer, ``.3f``, ``.0%``). A figure that is a whole fenced block (a quick start's output)
has its markers on lines of their own around the block, because a Markdown line that opens an
HTML comment is an HTML block to its end. ``guides`` rewrites every marked figure from the runs and
prints what moved; ``--check`` prints without writing. Prose around a figure stays hand-written,
and a figure from another run, or a run time, is hand-written with the date of its run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_FILE = REPO / "tests" / "figures" / "snapshot.yaml"
RUNS = REPO / "runs"
WORLD_CONFIGS = {
    "tiny": "data/world/tiny.yaml",
    "default": "data/world/default.yaml",
    "tiny_full": "data/world/tiny_full.yaml",
    "default_full": "data/world/default_full.yaml",
    "tiny_pairs": "data/world/tiny_pairs.yaml",
    "tiny_place_chain": "data/world/tiny_place_chain.yaml",
}
"""The worlds of the snapshot. ``tiny_full`` and ``default_full`` carry the phase (c) settings
until stage c4 (stage c2b, WM.63), ``tiny_pairs`` is stage c2a's world from an event file, kept
so that its identity is tested, and ``tiny_place_chain`` is stage c3a's world with a location
fluent and the place chain example's moves and acts, the only world of ``data/`` with location
fluents until stage c3b."""
CORPUS_CONFIGS = {"tiny": "data/corpus/tiny.yaml", "default": "data/corpus/default.yaml"}
"""The corpora of the snapshot: none runs on a world with pair fluents until stage c4."""
BASELINE_NAMES = ("tiny", "default", "tiny_full", "default_full")
"""The worlds whose off-setting baselines the snapshot holds: every world of the snapshot
without an event file (``tiny_pairs`` and ``tiny_place_chain`` define pair fluents and place
event types in a file, which the off settings would refuse)."""
GUIDES = (
    "README.md",
    "docs/guides/README.md",
    "docs/guides/WORLD.md",
    "docs/guides/CORPUS.md",
    "docs/guides/WORDFORMS.md",
)
OFF_SETTINGS: dict[str, dict[str, Any]] = {
    "conditional_effects_off": {"event_types": {"effects": {"conditional_share": 0}}},
    "fluents_off": {"fluents": {"count": 0}},
    "two_place_off": {"event_types": {"binary": None}},
    "numeric_fluents_off": {"numeric_fluents": {"count": 0}},
    # places off turns the location fluents off with them (stage c3b: the full worlds have one)
    "places_off": {"places": None, "place_event_types": None, "pair_fluents": {"locations": 0}},
    "pair_fluents_off": {"pair_fluents": {"count": 0, "locations": 0}},
    "locations_off": {"pair_fluents": {"locations": 0}},
    "pair_dynamics_off": {
        "pair_fluents": {"rule_weight": 0, "derived_rules": 0, "same_rules": 0},
        "event_types": {
            "preconditions": {"pair_share": 0},
            "effects": {
                "pair_share": 0,
                "condition_kinds": {"feature": 0.4, "fluent": 0.4, "numeric": 0.2, "pair": 0},
            },
        },
    },
}
"""The features of the world with an off setting, and the setting: conditional effects (stage
b1, promised byte-identical to the world without them), fluents (the static side is promised
unchanged), two-place event types (the taxonomy and the one-place side are promised unchanged),
numeric fluents (stage b2, promised byte-identical to the b1 world), places (stage c1, promised
byte-identical to the world before places, after the rounding of drawn reals), pair fluents
(stage c2b: ``default_full`` with ``count`` 0 is promised byte-identical to the default world of
stage c2a, and every world without pair fluents is its own baseline), and the pair dynamics
(stage c2b: with the pair shares, the pair condition weight, the rule weight, and the quotas
``derived_rules`` and, since stage c3b, ``same_rules`` at 0, the pair fluents exist with their
initial pairs but no entity event type reads or writes them, and in a world without location
fluents the event types and the derived rules are promised to be those of the world without
pair fluents; with location fluents the place event types keep their drawn dynamics, which
are location dynamics), and the location fluents (stage c3b: ``tiny_full`` and ``default_full`` with
``locations`` 0 are promised byte-identical to the full worlds of stage c2b, because every draw
of stage c3b happens only in a world with location fluents; a world without them is its own
baseline)."""

_MARKER = re.compile(
    r"<!--figure:(?P<key>[A-Za-z0-9_.]+)(?:\|(?P<format>[^>]*?))?-->(?P<text>.*?)<!--/figure-->",
    re.DOTALL,
)


# ---------------------------------------------------------------------------------------------
# Reading a world run
# ---------------------------------------------------------------------------------------------


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rate(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def world_figures(folder: str | Path) -> dict[str, Any]:
    """The figures of a world run folder."""
    folder = Path(folder)
    config = _yaml(folder / "config.yaml")
    stats = _yaml(folder / "world_stats.yaml")
    definition = json.loads((folder / "definition.json").read_text(encoding="utf-8"))
    entities = sum(1 for _ in (folder / "entities.csv").open(encoding="utf-8")) - 1
    categories = (
        sum(1 for _ in (folder / "taxonomy" / "categories_generative.csv").open(encoding="utf-8"))
        - 1
    )
    event_types = {e["label"]: e for e in definition["event_types"]}
    leaves = [e for e in definition["event_types"] if e["kind"] == "event_type"]
    conditional_effects = [
        (e["label"], effect) for e in leaves for effect in e["effects"] if effect.get("condition")
    ]
    # An inherited effect is one conditional effect shared by several event types; its holders
    # are counted. An effect that one event type alone carries is its own.
    by_effect: dict[str, list[str]] = {}
    for label, effect in conditional_effects:
        by_effect.setdefault(effect["expression"], []).append(label)
    inherited = [
        {"expression": expression, "code": f"`{expression}`", "holders": len(labels)}
        for expression, labels in by_effect.items()
        if len(labels) > 1
    ]
    own = [
        {"expression": expression, "code": f"`{expression}`", "event_type": labels[0]}
        for expression, labels in by_effect.items()
        if len(labels) == 1
    ]
    capacities = _csv_rows(folder / "derived" / "capacities.csv")
    one_place_labels = (
        [column for column in capacities[0] if column.startswith("CAN.EVENTTYPE1.")]
        if capacities
        else []
    )
    one_place_rates = [
        sum(int(row[column]) for row in capacities) / len(capacities) for column in one_place_labels
    ]
    relation_rows = (
        _csv_rows(folder / "derived" / "event_type_stats.csv")
        if (folder / "derived" / "event_type_stats.csv").is_file()
        else []
    )
    # the place event types' rows (stage c1) are densities over entity and place bindings,
    # reported under ``places``; the two-place figure stays over ordered pairs of entities
    two_place_rates = [
        float(row["proportion_true"])
        for row in relation_rows
        if not row["event_type"].startswith("PLACEEVENTTYPE.")
    ]
    episodes = stats["episodes"]
    relations = stats.get("relations") or {}
    effects = stats["effects"]
    conditional = effects["conditional"]
    numeric = stats.get("numeric_fluents") or {}
    places = stats.get("places") or {}
    place_densities = list((places.get("density") or {}).values())
    pair_block = stats.get("pair_fluents")
    figures: dict[str, Any] = {
        "name": config["name"],
        "seed": config["seed"],
        "rule_set_id": definition["rule_set_id"],
        "rule_set_short": definition["rule_set_id"][:12],
        "rule_set_code": f"`{definition['rule_set_id'][:12]}...`",
        "entities": entities,
        "categories": categories,
        "fluents": dict(stats["fluents"]),
        "fluents_total": stats["fluents"]["base"] + stats["fluents"]["derived"],
        "event_types": {
            **stats["event_types"],
            "leaves": stats["event_types"]["one_place"] + stats["event_types"]["two_place"],
        },
        "constraints": relations.get("constraints", 0) + stats["event_types"]["one_place"],
        "constraints_two_place": relations.get("constraints", 0),
        "event_type_features": relations.get("event_type_features", 0),
        "constraint_families": dict(relations.get("constraint_families", {})),
        "precondition_literals": json.loads(json.dumps(stats["precondition_literals"])),
        "effects": {
            key: json.loads(json.dumps(effects[key]))
            for key in ("total", "by_kind", "by_role", "from_features")
        },
        "conditional": {
            "total": conditional["total"],
            "by_kind": dict(conditional["by_kind"]),
            "share_of_drawn": conditional.get("share_of_drawn"),
            "share_of_all": conditional.get("share_of_all"),
            "literals": json.loads(json.dumps(effects["condition_literals"])),
            "inherited": inherited,
            "inherited_count": len(inherited),
            "own": own,
            "own_count": len(own),
            "expressions": [
                f"`{effect['expression']}`, on `{label}`" for label, effect in conditional_effects
            ],
        },
        "numeric": {
            "count": numeric.get("count", 0),
            "drifting": numeric.get("drifting", 0),
            "fluents": {
                label: {**row, "drift": round(row["drift"], 6)}
                for label, row in numeric.get("fluents", {}).items()
            },
            # the drifts as a list, so that a guide can name one by index (a label with a
            # period cannot be a step of a dotted key); rounded to six decimals, as a figure
            "drifts": [round(row["drift"], 6) for row in numeric.get("fluents", {}).values()],
            "effects": json.loads(json.dumps(numeric.get("effects", {}))),
            "precondition_literals": json.loads(
                json.dumps(numeric.get("precondition_literals", {}))
            ),
            "condition_literals": json.loads(json.dumps(numeric.get("condition_literals", {}))),
            "rules_reading_a_threshold": numeric.get("rules_reading_a_threshold", 0),
            "changes_per_event": episodes.get("numeric_changes_per_event"),
            "drift_changes_per_step": episodes.get("drift_changes_per_step"),
        },
        "places": {
            "on": bool(places),
            "places": places.get("places", 0),
            "categories": places.get("categories", 0),
            "features": json.loads(json.dumps(places.get("features", {}))),
            "place_event_types": places.get("place_event_types", 0),
            "constraints": places.get("constraints", 0),
            "constraint_families": dict(places.get("constraint_families", {})),
            "density": _rate(place_densities) if place_densities else None,
            "approximate_capacities": places.get("approximate_capacities"),
            # the location figures (stage c3b), only in a world with location fluents, so
            # that the figures of every other world do not move
            **(
                {
                    "by_kind": dict(places["place_event_types_by_kind"]),
                    "dynamics": dict(places.get("dynamics", {})),
                    "habitats": json.loads(json.dumps(places.get("habitats", {}))),
                    # the first location fluent's habitats, reachable by a dotted key (a
                    # label with a period cannot be a step of one)
                    "habitat": json.loads(
                        json.dumps(next(iter(places.get("habitats", {}).values()), {}))
                    ),
                }
                if "place_event_types_by_kind" in places
                else {}
            ),
        },
        "fix_ups": dict(stats["fix_ups"]),
        "enabling_graph": dict(stats["enabling_graph"]),
        "absorbing_fluents": list(stats["absorbing_fluents"]),
        "absorbing_code": ", ".join(f"`{f}`" for f in stats["absorbing_fluents"]) or "none",
        "requirements": {
            "one_place": _rate(one_place_rates) if one_place_rates else None,
            "two_place": _rate(two_place_rates) if two_place_rates else None,
        },
        "episodes": {
            key: json.loads(json.dumps(episodes[key]))
            for key in (
                "count",
                "steps",
                "events",
                "two_place_share",
                "mean_changes_per_event",
                "quiescent_share",
                "precondition_redraws",
                "never_legal",
                # the shares of stage c3b, only with location fluents
                *(
                    ("move_share", "act_share", "two_place_share_entities")
                    if "move_share" in episodes
                    else ()
                ),
            )
        },
    }
    if pair_block:
        # the pair figures (stage c2b), only in a world with pair fluents, so that the figures
        # of every other world do not move
        figures["pair_fluents"] = {
            "count": pair_block["count"],
            "by_symmetry": dict(pair_block["by_symmetry"]),
            "functional": pair_block["functional"],
            "fluents": json.loads(json.dumps(pair_block["fluents"])),
            "initial_pairs": pair_block["initial_pairs"],
            "precondition_literals": json.loads(json.dumps(pair_block["precondition_literals"])),
            "compiled_literals": json.loads(json.dumps(pair_block["compiled_literals"])),
            "effects": json.loads(json.dumps(pair_block["effects"])),
            "condition_literals": json.loads(json.dumps(pair_block["condition_literals"])),
            "rules_reading_a_pair_literal": pair_block["rules_reading_a_pair_literal"],
            "fix_ups": dict(pair_block["fix_ups"]),
            "changes_per_event": pair_block["pair_changes_per_event"],
            # the location counts (stage c3b), only with location fluents
            **(
                {
                    "between_entities": pair_block["between_entities"],
                    "locations": pair_block["locations"],
                    "place_event_types": json.loads(
                        json.dumps(pair_block.get("place_event_types", {}))
                    ),
                }
                if "locations" in pair_block
                else {}
            ),
            "profiles": ", ".join(
                f"`{label}` (the location fluent: general, functional, and total)"
                if row["initial_rate"] is None
                else f"`{label}` ({row['symmetry']}{', functional' if row['functional'] else ''}, "
                f"rate {row['initial_rate']})"
                for label, row in pair_block["fluents"].items()
            ),
        }
    figures["categories_with_word"] = sum(
        1 for e in definition["event_types"] if e["kind"] == "event_type_category"
    )
    place_text = (
        f", {figures['places']['places']} places and "
        f"{figures['places']['place_event_types']} place event types"
        if figures["places"]["on"]
        else ""
    )
    if pair_block:
        place_text += f", {pair_block['count']} pair fluents"
        if pair_block.get("locations"):
            place_text += (
                f" ({pair_block['between_entities']} between entities, "
                f"{pair_block['locations']} location fluent"
                f"{'s' if pair_block['locations'] != 1 else ''})"
            )
    figures["wrote_line"] = (
        f"{entities} entities, {figures['fluents_total']} fluents "
        f"({figures['fluents']['derived']} derived), {figures['numeric']['count']} numeric "
        f"fluents, {figures['event_types']['one_place']} "
        f"one-place and {figures['event_types']['two_place']} two-place event types, "
        f"{figures['constraints']} constraints{place_text}; rule set {figures['rule_set_short']}"
    )
    figures["redraw_notes"] = "\n".join(
        f"note: the preconditions of {label} were drawn again {count} time(s)"
        for label, count in episodes["precondition_redraws"].items()
    )
    output = f"runs/world/{config['name']}_seed{config['seed']}"
    lines = [f"wrote {output}: {figures['wrote_line']}"]
    if figures["redraw_notes"]:
        lines.append(figures["redraw_notes"])
    figures["quick_start_block"] = "\n```\n" + "\n".join(lines) + "\n```\n"
    del event_types
    return figures


# ---------------------------------------------------------------------------------------------
# Reading a corpus run
# ---------------------------------------------------------------------------------------------

_CONTEXT_LEVELS = ("class", "instance", "event")


def corpus_figures(folder: str | Path) -> dict[str, Any]:
    """The figures of a corpus run folder."""
    folder = Path(folder)
    config = _yaml(folder / "config.yaml")
    stats = _yaml(folder / "stats.yaml")
    lexicon = stats["lexicon"]
    function = lexicon["concepts"]["function"]
    sets = stats["test_sets"]
    pairs = sum(s["pairs"] for s in sets.values())
    seen = {
        name: s["true_items_seen_share"]
        for name, s in sets.items()
        if s["true_items_seen_share"] is not None
        and any(name.startswith(level + "_") for level in _CONTEXT_LEVELS)
    }
    by_type = stats["documents"]["by_type"]
    cooccurrence = stats["cooccurrence"]["by_document_type"]
    figures: dict[str, Any] = {
        "name": config["name"],
        "seed": config["seed"],
        "world": config["provenance"]["world"]["rule_set_id"][:12],
        "documents": stats["documents"]["count"],
        "sentences": stats["sentences"]["count"],
        "by_level": dict(stats["sentences"]["by_level"]),
        "tokens": stats["tokens"]["count"],
        "scenes": stats["scenes"]["count"],
        "events": stats["scenes"]["events"],
        "lexemes": {
            "total": lexicon["lexemes"],
            "content": lexicon["lexemes"] - function,
            "function": function,
            "without_word": dict(lexicon["event_types_without_word"]),
        },
        "test_sets": {
            "count": len(sets),
            "pairs": pairs,
            "items": 2 * pairs,
            "by_set": {
                name: {
                    key: value
                    for key, value in s.items()
                    if key in ("pairs", "true_statements", "true_items_seen_share")
                }
                for name, s in sets.items()
            },
        },
        "seen": {
            "min_share": min(seen.values()) if seen else None,
            "min_set": min(seen, key=seen.get) if seen else None,
            "max_share": max(seen.values()) if seen else None,
            "max_set": max(seen, key=seen.get) if seen else None,
        },
        "rule_statements": {
            key: stats["rule_statements"][key] for key in ("rules", "terms", "stated")
        },
        "causal": dict(stats["causal"]),
        "comparisons": json.loads(json.dumps(stats.get("comparisons", {}))),
        "effect_statements": sets.get("causal_effect_polarity", {}).get("true_statements"),
        "precondition_statements": sets.get("causal_precondition_polarity", {}).get(
            "true_statements"
        ),
        "states": {
            "events_with_result_share": stats["states"]["events_with_result_share"],
            "result_sentences": stats["states"]["result_sentences"],
            "numeric_state_sentences": stats["states"].get("numeric_state_sentences"),
            "numeric_change_sentences": stats["states"].get("numeric_change_sentences"),
        },
        "narratives_short": {
            kind: by_type[kind]["shorter_than_drawn_share"] for kind in ("entity", "situational")
        },
        "cooccurrence": {
            "pairs": stats["cooccurrence"]["pairs"],
            "words": {
                kind: {
                    "thematic": cooccurrence[kind]["words"]["thematic"]["pearson"],
                    "taxonomic": cooccurrence[kind]["words"]["taxonomic"]["pearson"],
                }
                for kind in (
                    "entity",
                    "situational",
                    "encyclopedic_category",
                    "encyclopedic_feature",
                )
            },
            "check": json.loads(json.dumps(stats["cooccurrence"]["check"])),
        },
        "ambiguity": {
            "ambiguous": stats["ambiguity"]["ambiguous"],
            "share": stats["ambiguity"]["ambiguous_share"],
        },
        "changed_items_share": {
            name: s["items_changed_share"]
            for name, s in sets.items()
            if "items_changed_share" in s and name.endswith("_changed")
        },
    }
    figures["wrote_line"] = (
        f"{figures['documents']} documents, {figures['sentences']} sentences, "
        f"{figures['tokens']} tokens, {figures['scenes']} scenes, "
        f"{figures['test_sets']['count']} test sets with {pairs} pairs"
    )
    figures["wrote_code"] = f"`{figures['wrote_line']}`"
    output = f"runs/corpus/{config['name']}_seed{config['seed']}"
    figures["quick_start_block"] = f"\n```\nwrote {output}: {figures['wrote_line']}\n```\n"
    figures["seen"]["min_set_code"] = f"`{figures['seen']['min_set']}`"
    figures["seen"]["max_set_code"] = f"`{figures['seen']['max_set']}`"
    return figures


def documents_hash(folder: str | Path) -> str:
    """The SHA-256 of a corpus run's ``documents.jsonl``."""
    return hashlib.sha256((Path(folder) / "documents.jsonl").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# The baselines of the off settings
# ---------------------------------------------------------------------------------------------


def world_config_off(name: str, feature: str):
    """The world configuration of ``name`` with ``feature`` turned off."""
    from semantic_world.world.config import config_from_mapping

    path = REPO / WORLD_CONFIGS[name]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in OFF_SETTINGS[feature].items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            for inner, setting in value.items():
                if isinstance(setting, dict) and isinstance(data[key].get(inner), dict):
                    data[key][inner] = {**data[key][inner], **setting}
                else:
                    data[key][inner] = setting
        else:
            data[key] = value
    return config_from_mapping(data, source=str(path))


def baseline_identity(name: str, feature: str) -> str:
    """The rule-set identity of the world ``name`` with ``feature`` off, defined in memory."""
    from semantic_world.world.generate import define

    return define(world_config_off(name, feature)).rule_set_id


def baselines(names: Sequence[str] = BASELINE_NAMES) -> dict[str, Any]:
    """The baselines of every off setting for the given worlds: the rule-set identities, and
    for conditional effects the hash of the tiny corpus's ``documents.jsonl``."""
    import tempfile

    found: dict[str, Any] = {}
    for feature in OFF_SETTINGS:
        found[feature] = {}
        for name in names:
            identity = baseline_identity(name, feature)
            entry: dict[str, Any] = {
                "rule_set_id": identity,
                "rule_set_short": identity[:12],
                "rule_set_code": f"`{identity[:12]}...`",
            }
            if feature == "conditional_effects_off" and name == "tiny":
                entry["documents_sha256"] = tiny_corpus_hash_off(feature, tempfile.mkdtemp())
            found[feature][name] = entry
    return found


def tiny_corpus_hash_off(feature: str, folder: str | Path) -> str:
    """The hash of ``documents.jsonl`` of the tiny corpus over the tiny world with ``feature``
    off, generated into ``folder``."""
    from semantic_world.corpus import config_from_mapping, generate

    folder = Path(folder)
    world_path = folder / "world.yaml"
    world_path.write_text(
        yaml.safe_dump(world_config_off("tiny", feature).resolved(), sort_keys=False),
        encoding="utf-8",
    )
    data = yaml.safe_load((REPO / CORPUS_CONFIGS["tiny"]).read_text(encoding="utf-8"))
    data["world"] = {"config": str(world_path), "seed": data["world"].get("seed", 1)}
    corpus = generate(config_from_mapping(data))
    return documents_hash(corpus.write(folder / "corpus"))


# ---------------------------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------------------------


def run_folders(runs: str | Path = RUNS) -> dict[str, dict[str, Path]]:
    """The run folders under ``runs``, by the configurations' names and seeds: a world for
    every entry of ``WORLD_CONFIGS``, and a corpus for every entry of ``CORPUS_CONFIGS``."""
    from semantic_world.corpus import load_config as load_corpus_config
    from semantic_world.world.config import load_config as load_world_config

    runs = Path(runs)
    found: dict[str, dict[str, Path]] = {}
    for name in WORLD_CONFIGS:
        world = load_world_config(REPO / WORLD_CONFIGS[name])
        found[name] = {"world": runs / "world" / f"{world.name}_seed{world.seed}"}
        if name in CORPUS_CONFIGS:
            corpus = load_corpus_config(REPO / CORPUS_CONFIGS[name])
            found[name]["corpus"] = runs / "corpus" / f"{corpus.name}_seed{corpus.seed}"
    return found


def figures_of_runs(runs: str | Path = RUNS) -> dict[str, Any]:
    """The figures of the default and tiny runs under ``runs``."""
    found: dict[str, Any] = {}
    for name, folders in run_folders(runs).items():
        found[name] = {}
        if (folders["world"] / "world_stats.yaml").is_file():
            found[name]["world"] = world_figures(folders["world"])
        if "corpus" in folders and (folders["corpus"] / "stats.yaml").is_file():
            found[name]["corpus"] = corpus_figures(folders["corpus"])
    return found


def load_snapshot(path: str | Path = SNAPSHOT_FILE) -> dict[str, Any]:
    return _yaml(Path(path))


def flatten(mapping: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a nested mapping, by its dotted path; lists stay whole."""
    if not isinstance(mapping, Mapping):
        return {prefix: mapping}
    found: dict[str, Any] = {}
    for key, value in mapping.items():
        found.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return found


def moves(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """The figures that differ between two snapshots, as (key, old value, new value)."""
    before, after = flatten(old), flatten(new)
    found = []
    for key in sorted(set(before) | set(after)):
        if before.get(key, "<absent>") != after.get(key, "<absent>"):
            found.append((key, before.get(key, "<absent>"), after.get(key, "<absent>")))
    return found


def write_snapshot(
    runs: str | Path = RUNS, path: str | Path = SNAPSHOT_FILE, with_baselines: bool = True
) -> list[tuple[str, Any, Any]]:
    """Rewrite the snapshot from the runs (and the baselines, defined in memory); return what
    moved."""
    path = Path(path)
    old = load_snapshot(path) if path.is_file() else {}
    new = figures_of_runs(runs)
    new["baselines"] = baselines() if with_baselines else old.get("baselines", {})
    changed = moves(old, new)
    header = (
        "# The figures of the default and tiny runs (and of the default_full, tiny_full, and\n"
        "# tiny_pairs worlds), and the baselines of the off settings.\n"
        "# Written by `python -m semantic_world.figures snapshot` from the runs in runs/;\n"
        "# compared by tests/figures/test_figures.py against runs regenerated in memory. Never\n"
        "# edited by hand: rewrite it on purpose after a change that alters a run, and say what\n"
        "# moved.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        header + yaml.safe_dump(new, sort_keys=False, default_flow_style=None, width=100),
        encoding="utf-8",
    )
    return changed


# ---------------------------------------------------------------------------------------------
# The guides
# ---------------------------------------------------------------------------------------------


def lookup(figures: Mapping[str, Any], key: str) -> Any:
    value: Any = figures
    for part in key.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(key)
    return value


def format_figure(value: Any, spec: str | None) -> str:
    """A figure as the guides write it: an integer with a thousands separator unless the
    specification says otherwise, a float as given, a list joined by commas."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "none"
    if isinstance(value, Mapping):
        return ", ".join(f"`{k}` ({v})" for k, v in value.items()) or "none"
    if value is None:
        return "none"
    if spec:
        return format(value, spec)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def fill_guides(
    figures: Mapping[str, Any], root: str | Path = REPO, check: bool = False
) -> list[tuple[str, str, str, str]]:
    """Rewrite every marked figure of the guides; return (file, key, old text, new text) for
    each figure that moved. With ``check``, write nothing."""
    changed: list[tuple[str, str, str, str]] = []
    for name in GUIDES:
        path = Path(root) / name
        text = path.read_text(encoding="utf-8")
        rewritten = _MARKER.sub(lambda m, name=name: _replace(m, figures, name, changed), text)
        if rewritten != text and not check:
            path.write_text(rewritten, encoding="utf-8")
    return changed


def _replace(
    match: re.Match[str],
    figures: Mapping[str, Any],
    name: str,
    changed: list[tuple[str, str, str, str]],
) -> str:
    key, spec, old = match.group("key"), match.group("format"), match.group("text")
    new = format_figure(lookup(figures, key), spec)
    if new != old:
        changed.append((name, key, old, new))
    return f"<!--figure:{key}{'|' + spec if spec else ''}-->{new}<!--/figure-->"


def guide_keys(root: str | Path = REPO) -> dict[str, list[str]]:
    """Every figure key a guide quotes, by guide."""
    found: dict[str, list[str]] = {}
    for name in GUIDES:
        text = (Path(root) / name).read_text(encoding="utf-8")
        found[name] = [m.group("key") for m in _MARKER.finditer(text)]
    return found


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_world.figures",
        description="Rewrite the snapshot of the runs' figures, or the guides' figures.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser(
        "snapshot", help="rewrite tests/figures/snapshot.yaml from runs/ and print what moved"
    )
    snapshot.add_argument("--runs", default=str(RUNS), help="the runs folder (default: runs)")
    snapshot.add_argument("--snapshot", default=str(SNAPSHOT_FILE), help="the snapshot file")
    snapshot.add_argument(
        "--no-baselines",
        action="store_true",
        help="keep the recorded baselines instead of defining the worlds with each feature off",
    )
    guides = commands.add_parser(
        "guides", help="write the guides' figures from runs/ between their marker comments"
    )
    guides.add_argument("--runs", default=str(RUNS), help="the runs folder (default: runs)")
    guides.add_argument("--snapshot", default=str(SNAPSHOT_FILE), help="the snapshot file")
    guides.add_argument(
        "--check", action="store_true", help="report what would move; write nothing"
    )
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        changed = write_snapshot(args.runs, args.snapshot, with_baselines=not args.no_baselines)
        for key, old, new in changed:
            print(f"{key}: {old} -> {new}")
        print(f"wrote {args.snapshot}: {len(changed)} figure(s) moved")
        return 0
    figures = figures_of_runs(args.runs)
    snapshot_path = Path(args.snapshot)
    figures["baselines"] = (
        load_snapshot(snapshot_path).get("baselines", {}) if snapshot_path.is_file() else {}
    )
    try:
        changed = fill_guides(figures, REPO, check=args.check)
    except KeyError as error:
        print(f"error: no figure {error} in the runs under {args.runs}", file=sys.stderr)
        return 1
    for name, key, old, new in changed:
        print(f"{name}: {key}: {old!r} -> {new!r}")
    verb = "would move" if args.check else "moved"
    print(f"{len(changed)} figure(s) {verb} in {len(GUIDES)} guides")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
