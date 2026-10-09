"""Episodes: runs of the world over participants (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Episodes and selection").

An episode draws its participants as corpus scenes draw them: a seed instance, then
``scene.size`` others without replacement, each weighted by the thematic relatedness and the
taxonomic similarity of its leaf and the seed's leaf, and a constant. It starts from the
participants' initial values in ``entities.csv``, or redraws them at each fluent's initial rate
(``scene.initial: redraw``). It runs a drawn number of steps. At each step a Poisson number of
events is drawn one at a time by the selection policy among the events that are legal in the
state and do not interfere with the events already chosen; then the runtime applies the step.
When no event is legal at a time point, the episode ends there, quiescent.

Every episode draws from its own part of a stream, named by its label, so it depends only on the
master seed, its number, its seed instance, and the scene settings. The scene settings live in
the corpus configuration; ``simulate`` takes their defaults, or reads the ``scene`` block of a
corpus configuration file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.taxonomy.config import ConfigError, _Node
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.similarity import similarity_matrix
from semantic_world.taxonomy.streams import stream_seed
from semantic_world.world.definition import RuntimeDefinition, load_definition
from semantic_world.world.derived import DERIVED_DIR, load_derived_csv
from semantic_world.world.errors import WorldError
from semantic_world.world.history import (
    EPISODES_FILE,
    Change,
    History,
    HistoryEvent,
    Step,
    changes_of,
    write_histories,
)
from semantic_world.world.labels import translate
from semantic_world.world.policies import DEFAULT_POLICY, StepContext, policy
from semantic_world.world.runtime import (
    Event,
    State,
    apply,
    initial_state,
    legal_bindings,
)

EPISODE_STREAM = "world:episodes"
"""The stream of ``simulate``; the world statistics use ``world:stats``. Both are in the world's
stream table, so a run's ``config.yaml`` records their seeds."""
STATS_STREAM = "world:stats"
STATS_EPISODES = 1000
PARTICIPANT_WEIGHTS = ("thematic", "taxonomic", "constant")
INITIAL_MODES = ("keep", "redraw")
THEMATIC_FILE = "thematic.csv"


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSettings:
    """The scene settings of the corpus configuration, as episodes read them. The defaults are
    the corpus's."""

    size: tuple[int, int] = (2, 6)
    """The number of participants drawn beside the seed instance."""
    steps: tuple[int, int] = (3, 8)
    events_per_step: float = 1.5
    transitive_share: float = 0.5
    participant_weights: Mapping[str, float] = None  # type: ignore[assignment]
    event_type_weights: Mapping[str, float] = None  # type: ignore[assignment]
    """A weight per event type; an event type left out has the weight 1."""
    policy: str = DEFAULT_POLICY
    initial: str = "keep"

    def __post_init__(self) -> None:
        if self.participant_weights is None:
            object.__setattr__(
                self, "participant_weights", {"thematic": 1.0, "taxonomic": 0.5, "constant": 0.1}
            )
        if self.event_type_weights is None:
            object.__setattr__(self, "event_type_weights", {})
        if self.initial not in INITIAL_MODES:
            raise WorldError(f"scene.initial must be one of {', '.join(INITIAL_MODES)}")

    def resolved(self) -> dict[str, Any]:
        return {
            "size": list(self.size),
            "steps": list(self.steps),
            "events_per_step": self.events_per_step,
            "transitive_share": self.transitive_share,
            "participant_weights": dict(self.participant_weights),
            "event_type_weights": dict(self.event_type_weights) or "uniform",
            "policy": self.policy,
            "initial": self.initial,
        }


def _range(node: _Node, key: str, default: list[int], minimum: int) -> tuple[int, int]:
    value = node.get(key, default)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        raise node.error(key, f"expected a pair [min, max] of integers, found {value!r}")
    low, high = int(value[0]), int(value[1])
    if low < minimum or high < low:
        raise node.error(key, f"expected {minimum} <= min <= max, found {value!r}")
    return (low, high)


def _weights(node: _Node, key: str, default: dict[str, float], allowed: tuple[str, ...]) -> dict:
    value = node.get(key, default)
    if not isinstance(value, dict):
        raise node.error(key, f"expected a mapping, found {value!r}")
    for name, weight in value.items():
        if name not in allowed:
            raise node.error(f"{key}.{name}", f"is not one of {', '.join(allowed)}")
        node.check_number(f"{key}.{name}", weight, min=0)
    return {name: float(value.get(name, 0.0)) for name in allowed}


def _event_type_weights(node: _Node, definition: RuntimeDefinition | None) -> dict[str, float]:
    """``event_type_weights`` (new labels), or ``verb_weights`` (the corpus's old labels,
    translated). ``uniform`` or ``null`` means 1 for every event type."""
    weights: dict[str, float] = {}
    for key, translated in (("event_type_weights", False), ("verb_weights", True)):
        value = node.get(key, None, nullable=True)
        if value is None or value == "uniform":
            continue
        if not isinstance(value, dict):
            raise node.error(key, f"expected a mapping, 'uniform', or null, found {value!r}")
        for label, weight in value.items():
            name = translate(str(label)) if translated else str(label)
            node.check_number(f"{key}.{label}", weight, min=0)
            if definition is not None and not any(
                et.label == name and not et.is_category for et in definition.event_types
            ):
                raise node.error(f"{key}.{label}", "is not an event type of the world")
            weights[name] = float(weight)
    return weights


def read_scene_settings(
    data: Mapping[str, Any], source: str, definition: RuntimeDefinition | None = None
) -> EpisodeSettings:
    """The episode settings from the ``scene`` block of a corpus configuration mapping. Keys the
    corpus does not know yet (``event_type_weights``, ``policy``, ``initial``) are read here;
    the corpus's ``verb_weights`` is accepted with its labels translated."""
    if not isinstance(data, dict):
        raise ConfigError(source, "<file>", "expected a mapping")
    node = _Node(source, "scene", data.get("scene") or {})
    settings = EpisodeSettings(
        size=_range(node, "size", [2, 6], 0),
        steps=_range(node, "steps", [3, 8], 1),
        events_per_step=float(node.number("events_per_step", 1.5, min=0)),
        transitive_share=node.probability("transitive_share", 0.5),
        participant_weights=_weights(
            node,
            "participant_weights",
            {"thematic": 1.0, "taxonomic": 0.5, "constant": 0.1},
            PARTICIPANT_WEIGHTS,
        ),
        event_type_weights=_event_type_weights(node, definition),
        policy=node.choice("policy", DEFAULT_POLICY, _policy_names()),
        initial=node.choice("initial", "keep", INITIAL_MODES),
    )
    node.finish()
    return settings


def _policy_names() -> tuple[str, ...]:
    from semantic_world.world.policies import policy_names

    return policy_names()


def load_scene_settings(
    path: str | Path, definition: RuntimeDefinition | None = None
) -> EpisodeSettings:
    """Read the ``scene`` block of a corpus configuration file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(str(path), "<file>", f"cannot read the file: {error.strerror}") from None
    except yaml.YAMLError as error:
        raise ConfigError(str(path), "<file>", f"invalid YAML: {error}") from None
    return read_scene_settings(data, str(path), definition)


# ---------------------------------------------------------------------------------------------
# Relatedness: what participant draws weigh
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Relatedness:
    """The thematic relatedness and the taxonomic similarity of every pair of leaves, and each
    entity's leaf. Thematic relatedness comes from ``thematic.csv`` (0 everywhere for a world
    without two-place event types); similarity is computed from the leaves' generative vectors
    with the taxonomy's similarity settings, exactly as the corpus computes it."""

    leaves: tuple[str, ...]
    thematic: np.ndarray
    similarity: np.ndarray
    entity_leaf: np.ndarray
    """Each entity's leaf as an index into ``leaves``."""

    @classmethod
    def from_frames(
        cls,
        definition: RuntimeDefinition,
        thematic: pl.DataFrame | None,
        similarity: np.ndarray | None = None,
    ) -> Relatedness:
        """From the ``thematic.csv`` table of a run (``leaf_a``, ``leaf_b``, ``thematic``) and
        a leaf similarity matrix in the order of the definition's leaves (0 when None). An
        undefined similarity, or a negative one (phi), adds nothing to a weight."""
        leaves = tuple(dict.fromkeys(definition.leaves))
        number = {leaf: i for i, leaf in enumerate(leaves)}
        n = len(leaves)
        thematic_matrix = np.zeros((n, n))
        if thematic is not None:
            for row in thematic.iter_rows(named=True):
                a, b = number.get(row["leaf_a"]), number.get(row["leaf_b"])
                if a is None or b is None:
                    continue
                thematic_matrix[a, b] = thematic_matrix[b, a] = float(row["thematic"])
        matrix = np.zeros((n, n)) if similarity is None else np.asarray(similarity, dtype=float)
        if matrix.shape != (n, n):
            raise WorldError(f"the similarity matrix has shape {matrix.shape}; expected {(n, n)}")
        matrix = np.clip(np.nan_to_num(matrix, nan=0.0), 0.0, None)
        entity_leaf = np.array([number[leaf] for leaf in definition.leaves], dtype=np.intp)
        return cls(leaves, thematic_matrix, matrix, entity_leaf)

    @classmethod
    def from_run(cls, folder: str | Path, definition: RuntimeDefinition) -> Relatedness:
        """From a world run folder: ``derived/thematic.csv`` when the run has it, and the leaf
        similarity from ``taxonomy/categories_generative.csv`` with the settings of
        ``taxonomy/config.yaml``."""
        folder = Path(folder)
        derived = folder / DERIVED_DIR
        thematic = None
        if (derived / THEMATIC_FILE).is_file():
            thematic = load_derived_csv(derived, THEMATIC_FILE, definition.rule_set_id)
        return cls.from_frames(definition, thematic, _leaf_similarity_from_run(folder, definition))

    @classmethod
    def from_taxonomy(cls, taxonomy: TaxonomyResult, definition: RuntimeDefinition) -> Relatedness:
        """In memory, from the taxonomy run a world embeds."""
        from semantic_world.world.relation_stats import relation_frames

        frames = relation_frames(taxonomy)
        leaves = tuple(dict.fromkeys(definition.leaves))
        rows = {translate(c.label): i for i, c in enumerate(taxonomy.tree.categories)}
        leaf_rows = np.array([rows[leaf] for leaf in leaves], dtype=np.intp)
        analysis = taxonomy.config.analysis
        similarity = _leaf_similarity(
            taxonomy.vectors.generative,
            taxonomy.vectors.isa_count,
            leaf_rows,
            analysis.similarity_metric,
            analysis.similarity_features,
        )
        return cls.from_frames(definition, frames.get(THEMATIC_FILE), similarity)


def _leaf_similarity(
    generative: np.ndarray, isa_count: int, leaf_rows: np.ndarray, metric: str, features: str
) -> np.ndarray:
    """The taxonomic similarity of every pair of leaves, as the corpus computes it: the
    configured metric over the leaves' generative vectors, all features or the non-ISA ones."""
    start = 0 if features == "all" else isa_count
    return similarity_matrix(generative[leaf_rows][:, start:], metric)


VECTOR_PREFIXES = ("ISA", "IS", "HAS", "CAN", "PROPERTY", "PART")


def _leaf_similarity_from_run(folder: Path, definition: RuntimeDefinition) -> np.ndarray:
    taxonomy = folder / "taxonomy"
    config = yaml.safe_load((taxonomy / "config.yaml").read_text(encoding="utf-8")) or {}
    analysis = config.get("analysis") or {}
    frame = pl.read_csv(taxonomy / "categories_generative.csv", schema_overrides={"label": pl.Utf8})
    columns = [c for c in frame.columns if c.split(".")[0] in VECTOR_PREFIXES]
    generative = frame.select(columns).to_numpy().astype(np.uint8)
    isa_count = sum(1 for c in columns if c.startswith("ISA."))
    rows = {translate(label): i for i, label in enumerate(frame["label"].to_list())}
    leaves = tuple(dict.fromkeys(definition.leaves))
    leaf_rows = np.array([rows[leaf] for leaf in leaves], dtype=np.intp)
    return _leaf_similarity(
        generative,
        isa_count,
        leaf_rows,
        str(analysis.get("similarity_metric", "cosine")),
        str(analysis.get("similarity_features", "non_isa")),
    )


# ---------------------------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------------------------


class EpisodeGenerator:
    """The episodes of one world under one set of scene settings."""

    def __init__(
        self,
        definition: RuntimeDefinition,
        relatedness: Relatedness,
        settings: EpisodeSettings | None = None,
        initial_rates: Mapping[str, float] | None = None,
        record_legal: bool = False,
    ) -> None:
        self.definition = definition
        self.relatedness = relatedness
        self.settings = settings or EpisodeSettings()
        self.initial_rates = dict(initial_rates or {})
        self.record_legal = record_legal
        self.policy = policy(self.settings.policy)
        self.performable = tuple(et.label for et in definition.event_types if not et.is_category)
        for label in self.settings.event_type_weights:
            if label not in self.performable:
                raise WorldError(
                    f"scene.event_type_weights.{label} is not an event type of the world"
                )
        if self.settings.initial == "redraw":
            missing = [f for f in definition.base_fluents if f not in self.initial_rates]
            if missing:
                raise WorldError(
                    f"scene.initial: redraw needs every base fluent's initial rate; none is "
                    f"known for {missing[0]}"
                )
        if definition.initial_values is not None:
            self.start = initial_state(definition)
        elif self.settings.initial == "redraw":
            self.start = State(
                np.zeros((definition.entity_count, len(definition.base_fluents)), dtype=np.uint8)
            )
        else:
            raise WorldError(
                "the definition carries no initial fluents; episodes need scene.initial: redraw"
            )

    # Participants ----------------------------------------------------------------------------

    def participant_weights(self, seed: int) -> np.ndarray:
        """The weight of every entity as a participant of an episode seeded at ``seed``: a
        weighted sum of the thematic relatedness and the taxonomic similarity of its leaf and
        the seed's leaf, and a constant. The seed itself has the weight 0."""
        weights = self.settings.participant_weights
        leaf = self.relatedness.entity_leaf
        seed_leaf = leaf[seed]
        values = (
            weights["thematic"] * self.relatedness.thematic[leaf, seed_leaf]
            + weights["taxonomic"] * self.relatedness.similarity[leaf, seed_leaf]
            + weights["constant"]
        )
        values[seed] = 0.0
        return values

    def draw_participants(self, rng: np.random.Generator, seed: int) -> tuple[int, ...]:
        """The seed, then ``scene.size`` other entities drawn without replacement by their
        weights. An entity with the weight 0 is never drawn, so an episode can be smaller than
        asked."""
        low, high = self.settings.size
        size = int(rng.integers(low, high + 1))
        weights = self.participant_weights(seed)
        size = min(size, int(np.count_nonzero(weights)))
        if size == 0:
            return (seed,)
        drawn = rng.choice(len(weights), size=size, replace=False, p=weights / weights.sum())
        return (seed,) + tuple(int(i) for i in drawn)

    # Steps -----------------------------------------------------------------------------------

    def _initial(self, rng: np.random.Generator, participants: tuple[int, ...]) -> State:
        if self.settings.initial == "keep":
            return self.start
        values = np.array(self.start.values, dtype=np.uint8, copy=True)
        for entity in participants:
            for j, fluent in enumerate(self.definition.base_fluents):
                values[entity, j] = 1 if rng.random() < self.initial_rates[fluent] else 0
        return State(values)

    def _legal(self, state: State, participants: tuple[int, ...]) -> dict[str, list]:
        return {
            label: legal_bindings(self.definition, state, label, participants)
            for label in self.performable
        }

    def episode(self, rng: np.random.Generator, label: str, seed: int | None = None) -> History:
        """One episode drawn from ``rng``: the seed instance (drawn uniformly first when none is
        given), the participants, the number of steps, the initial state, then each step."""
        definition = self.definition
        if seed is None:
            seed = int(rng.integers(definition.entity_count))
        participants = self.draw_participants(rng, seed)
        low, high = self.settings.steps
        steps = int(rng.integers(low, high + 1))
        state = self._initial(rng, participants)
        labels = definition.entity_labels
        initial = {
            labels[p]: tuple(f for j, f in enumerate(definition.base_fluents) if state.values[p, j])
            for p in participants
        }
        recorded: list[Step] = []
        quiescent = False
        count = 0
        for k in range(1, steps + 1):
            legal = self._legal(state, participants)
            if not any(legal.values()):
                quiescent = True
                break
            context = StepContext(
                definition,
                state,
                participants,
                legal,
                self.settings.event_type_weights,
                self.settings.transitive_share,
            )
            wanted = int(rng.poisson(self.settings.events_per_step))
            events = self.policy(context, rng, wanted) if wanted else []
            after = apply(definition, state, events)
            history_events = []
            for event in events:
                count += 1
                et = definition.event_type(event.event_type)
                binding = {r: labels[i] for r, i in zip(et.roles, event.binding, strict=True)}
                history_events.append(
                    HistoryEvent(
                        f"{label}.EVENTINSTANCE.{count}",
                        event.event_type,
                        binding["agent"],
                        binding.get("patient"),
                        tuple(_event_changes(definition, state, after, event)),
                    )
                )
            recorded.append(
                Step(
                    k,
                    tuple(history_events),
                    {t: len(b) for t, b in legal.items()} if self.record_legal else None,
                )
            )
            state = after
        return History(
            label,
            labels[seed],
            tuple(labels[p] for p in participants),
            self.settings.policy,
            definition.rule_set_id,
            initial,
            tuple(recorded),
            f"TIME.{len(recorded) + 1}",
            quiescent,
        )

    def run(
        self,
        master_seed: int,
        count: int,
        stream: str = EPISODE_STREAM,
        seeds: Sequence[int] | None = None,
        first: int = 1,
    ) -> list[History]:
        """Episodes ``SCENE.<first>`` and on, each from its own part of the stream, named by its
        label (``<stream>:SCENE.<n>``). ``seeds`` gives each episode's seed instance; without
        them, each episode draws its own."""
        histories = []
        for k in range(count):
            label = f"SCENE.{first + k}"
            rng = np.random.default_rng(stream_seed(master_seed, f"{stream}:{label}"))
            histories.append(self.episode(rng, label, None if seeds is None else seeds[k]))
        return histories


def _event_changes(
    definition: RuntimeDefinition, before: State, after: State, event: Event
) -> list[Change]:
    """The base fluents the event changed: of the fluents it writes, those whose value differs
    between the states (non-interfering events write distinct fluents, so each change has one
    event)."""
    et = definition.event_type(event.event_type)
    entities = sorted({event.binding[et.roles.index(e.role)] for e in et.effects})
    written = {(event.binding[et.roles.index(e.role)], e.fluent) for e in et.effects}
    return [
        change
        for change in changes_of(definition, before, after, entities)
        if (definition.entity_index(change.entity), change.fluent) in written
    ]


# ---------------------------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------------------------


def read_initial_rates(folder: str | Path) -> dict[str, float]:
    """Each base fluent's initial rate, from ``world_stats.yaml`` of a run."""
    path = Path(folder) / "world_stats.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rates = (data.get("fluents") or {}).get("initial_rates") or {}
    return {str(k): float(v) for k, v in rates.items()}


def simulate(
    folder: str | Path,
    episodes: int,
    seed: int | None = None,
    settings: EpisodeSettings | None = None,
    record_legal: bool = False,
    out: str | Path | None = None,
) -> tuple[Path, list[History]]:
    """Run episodes of a world run folder and write them to ``episodes.jsonl`` (or ``out``).
    The master seed defaults to the run's seed."""
    folder = Path(folder)
    definition = load_definition(folder)
    if seed is None:
        config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) or {}
        seed = int(config.get("seed", 1))
    generator = EpisodeGenerator(
        definition,
        Relatedness.from_run(folder, definition),
        settings,
        read_initial_rates(folder),
        record_legal,
    )
    histories = generator.run(seed, episodes)
    path = Path(out) if out is not None else folder / EPISODES_FILE
    write_histories(path, histories)
    return path, histories


__all__ = [
    "EPISODE_STREAM",
    "INITIAL_MODES",
    "STATS_EPISODES",
    "STATS_STREAM",
    "EpisodeGenerator",
    "EpisodeSettings",
    "Relatedness",
    "load_scene_settings",
    "read_initial_rates",
    "read_scene_settings",
    "simulate",
]
