"""Episodes: runs of the world over participants (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Episodes and selection").

An episode draws its participants as corpus scenes draw them: a seed instance, then
``scene.size`` others without replacement, each weighted by the thematic relatedness and the
taxonomic similarity of its leaf and the seed's leaf, and a constant. It starts from the
participants' initial values in ``entities.csv``, or redraws them at each fluent's initial rate
(``scene.initial: redraw``); in a world with pair fluents (stage c2a) it keeps the initial pairs
among its participants and no other pair (WM.38), or redraws them at each pair fluent's rate
after every other draw of its part. It runs a drawn number of steps. At each step a Poisson
number of events is drawn one at a time by the selection policy among the events that are legal
in the state and do not interfere with the events already chosen; then the runtime applies the
step.
When no event is legal at a time point, the episode ends there, quiescent.

In a world with location fluents (stage c3a) an episode also has places: the distinct places of
its participants at ``TIME.1`` plus ``scene.extra_places`` (default 1) places drawn uniformly
from the others, or every place with ``scene.places: all``; a place event type's bindings range
over them. Each participant starts at its place of ``entities.csv``; ``scene.initial: redraw``
redraws the participants' places with the habitat weights, after every other redraw; and
``scene.initial_location: seed`` starts every participant at the seed entity's place, the
episode's places being that place and the extra places. The new draws (the redrawn places,
then the extra places) come after every existing draw of the episode's part, and only in a
world with location fluents, so the episodes of every other world are unchanged.

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
from semantic_world.taxonomy.similarity import similarity_matrix
from semantic_world.taxonomy.streams import stream_seed
from semantic_world.world.definition import RuntimeDefinition, load_definition
from semantic_world.world.derived import DERIVED_DIR, load_derived_csv
from semantic_world.world.errors import WorldError
from semantic_world.world.history import (
    DRIFT,
    EPISODES_FILE,
    AnyChange,
    Change,
    History,
    HistoryEvent,
    Step,
    changes_of,
    numeric_changes_of,
    pair_changes_of,
    write_histories,
)
from semantic_world.world.locations import DEFAULT_HABITAT_STRENGTH, draw_places, habitat_weights
from semantic_world.world.pair_fluents import draw_initial_pairs, pair_rows
from semantic_world.world.policies import DEFAULT_POLICY, StepContext, policy
from semantic_world.world.runtime import (
    Event,
    State,
    apply_step,
    initial_state,
    legal_bindings,
    place_able_table,
)
from semantic_world.world.statics import StaticWorld

EPISODE_STREAM = "world:episodes"
"""The stream of ``simulate``; the world statistics use ``world:stats``. Both are in the world's
stream table, so a run's ``config.yaml`` records their seeds."""
STATS_STREAM = "world:stats"
STATS_EPISODES = 1000
PARTICIPANT_WEIGHTS = ("thematic", "taxonomic", "constant")
INITIAL_MODES = ("keep", "redraw")
PLACES_MODES = ("episode", "all")
"""``scene.places`` (stage c3a): the participants' places plus the extra places, or every
place."""
LOCATION_MODES = ("world", "seed")
"""``scene.initial_location`` (stage c3a): each participant's own initial place, or the seed
entity's place for every participant."""
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
    extra_places: int = 1
    """The places drawn beside the participants' places (stage c3a)."""
    places: str = "episode"
    initial_location: str = "world"

    def __post_init__(self) -> None:
        if self.participant_weights is None:
            object.__setattr__(
                self, "participant_weights", {"thematic": 1.0, "taxonomic": 0.5, "constant": 0.1}
            )
        if self.event_type_weights is None:
            object.__setattr__(self, "event_type_weights", {})
        if self.initial not in INITIAL_MODES:
            raise WorldError(f"scene.initial must be one of {', '.join(INITIAL_MODES)}")
        if self.places not in PLACES_MODES:
            raise WorldError(f"scene.places must be one of {', '.join(PLACES_MODES)}")
        if self.initial_location not in LOCATION_MODES:
            raise WorldError(f"scene.initial_location must be one of {', '.join(LOCATION_MODES)}")
        if self.extra_places < 0:
            raise WorldError("scene.extra_places is 0 or more")

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
            "extra_places": self.extra_places,
            "places": self.places,
            "initial_location": self.initial_location,
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
    """``event_type_weights``: a weight per event type; ``uniform`` or ``null`` means 1 for every
    event type. The old key ``verb_weights`` is an error that names the new one."""
    weights: dict[str, float] = {}
    if "verb_weights" in node.data:
        raise node.error("verb_weights", "this key is now scene.event_type_weights")
    key = "event_type_weights"
    value = node.get(key, None, nullable=True)
    if value is None or value == "uniform":
        return weights
    if not isinstance(value, dict):
        raise node.error(key, f"expected a mapping, 'uniform', or null, found {value!r}")
    for label, weight in value.items():
        name = str(label)
        node.check_number(f"{key}.{label}", weight, min=0)
        if definition is not None and not any(
            et.label == name and et.performable for et in definition.event_types
        ):
            raise node.error(f"{key}.{label}", "is not an event type of the world")
        weights[name] = float(weight)
    return weights


def read_scene_settings(
    data: Mapping[str, Any], source: str, definition: RuntimeDefinition | None = None
) -> EpisodeSettings:
    """The episode settings from the ``scene`` block of a corpus configuration mapping."""
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
        extra_places=node.int("extra_places", 1, min=0),
        places=node.choice("places", "episode", PLACES_MODES),
        initial_location=node.choice("initial_location", "world", LOCATION_MODES),
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
    def from_statics(cls, statics: StaticWorld, definition: RuntimeDefinition) -> Relatedness:
        """In memory, from the static side of a world."""
        taxonomy = statics.taxonomy
        thematic = None if statics.relation_stats is None else statics.relation_stats.thematic
        leaves = tuple(dict.fromkeys(definition.leaves))
        rows = {c.label: i for i, c in enumerate(taxonomy.tree.categories)}
        leaf_rows = np.array([rows[leaf] for leaf in leaves], dtype=np.intp)
        analysis = taxonomy.config.analysis
        similarity = _leaf_similarity(
            taxonomy.vectors.generative,
            taxonomy.vectors.isa_count,
            leaf_rows,
            analysis.similarity_metric,
            analysis.similarity_features,
        )
        return cls.from_frames(definition, thematic, similarity)


def _leaf_similarity(
    generative: np.ndarray, isa_count: int, leaf_rows: np.ndarray, metric: str, features: str
) -> np.ndarray:
    """The taxonomic similarity of every pair of leaves, as the corpus computes it: the
    configured metric over the leaves' generative vectors, all features or the non-ISA ones."""
    start = 0 if features == "all" else isa_count
    return similarity_matrix(generative[leaf_rows][:, start:], metric)


VECTOR_PREFIXES = ("ISA", "PROPERTY", "PART")


def _leaf_similarity_from_run(folder: Path, definition: RuntimeDefinition) -> np.ndarray:
    taxonomy = folder / "taxonomy"
    config = yaml.safe_load((taxonomy / "config.yaml").read_text(encoding="utf-8")) or {}
    analysis = config.get("analysis") or {}
    frame = pl.read_csv(taxonomy / "categories_generative.csv", schema_overrides={"label": pl.Utf8})
    columns = [c for c in frame.columns if c.split(".")[0] in VECTOR_PREFIXES]
    generative = frame.select(columns).to_numpy().astype(np.uint8)
    isa_count = sum(1 for c in columns if c.startswith("ISA."))
    rows = {label: i for i, label in enumerate(frame["label"].to_list())}
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
        habitat_strength: float = DEFAULT_HABITAT_STRENGTH,
    ) -> None:
        self.definition = definition
        self.relatedness = relatedness
        self.settings = settings or EpisodeSettings()
        self.habitat_strength = float(habitat_strength)
        """``places.habitat_strength`` of the world's configuration (stage c3a): the weights
        ``scene.initial: redraw`` redraws the participants' places with."""
        self._habitat_weights: np.ndarray | None = None
        self.initial_rates = dict(
            definition.initial_rates if initial_rates is None else initial_rates
        )
        self.record_legal = record_legal
        self.policy = policy(self.settings.policy)
        self.performable = tuple(et.label for et in definition.event_types if et.performable)
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
        if definition.has_location_fluents and definition.initial_pairs is None:
            raise WorldError("the definition carries no initial places; episodes need them")
        if definition.initial_values is not None:
            self.start = initial_state(definition)
        elif self.settings.initial == "redraw":
            n = definition.entity_count
            self.start = State(
                np.zeros((n, len(definition.base_fluents)), dtype=np.uint8),
                np.zeros((n, len(definition.numeric_fluents))),
                tuple(
                    np.zeros((n, definition.pair_targets(f.label)), dtype=bool)
                    for f in definition.pair_fluents
                ),
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

    # Places ----------------------------------------------------------------------------------

    @property
    def habitat_weights(self) -> np.ndarray:
        """The habitat weights of every entity at every place (stage c3a), from the place event
        types' requirements of the definition and the habitat strength."""
        if self._habitat_weights is None:
            definition = self.definition
            self._habitat_weights = habitat_weights(
                place_able_table(definition).values(),
                (definition.entity_count, definition.place_count),
                self.habitat_strength,
            )
        return self._habitat_weights

    def _location_indices(self) -> list[int]:
        return [i for i, f in enumerate(self.definition.pair_fluents) if f.total]

    def _redraw_places(
        self, rng: np.random.Generator, participants: tuple[int, ...], pairs: list[np.ndarray]
    ) -> None:
        """Redraw the participants' places (in entity order) on each location fluent in label
        order, with the habitat weights, after every other redraw."""
        rows = np.array(sorted(participants), dtype=np.intp)
        for k in self._location_indices():
            drawn = draw_places(self.habitat_weights, rng, rows)
            matrix = np.array(pairs[k], dtype=bool, copy=True)
            matrix[rows, :] = False
            matrix[rows, drawn] = True
            pairs[k] = matrix

    def _seed_places(self, participants: tuple[int, ...], pairs: list[np.ndarray]) -> None:
        """``scene.initial_location: seed``: every participant at the seed entity's place."""
        for k in self._location_indices():
            matrix = np.array(pairs[k], dtype=bool, copy=True)
            matrix[list(participants), :] = matrix[participants[0], :]
            pairs[k] = matrix

    def episode_places(
        self, rng: np.random.Generator, participants: tuple[int, ...], state: State
    ) -> tuple[int, ...]:
        """The episode's places (F2): the distinct places of the participants at ``TIME.1``
        on every location fluent, plus ``scene.extra_places`` places drawn uniformly without
        replacement from the others (one draw), in place order; every place with
        ``scene.places: all``; none in a world without location fluents."""
        definition = self.definition
        if not definition.has_location_fluents:
            return ()
        if self.settings.places == "all":
            return tuple(range(definition.place_count))
        found: set[int] = set()
        for k in self._location_indices():
            found |= {
                int(q) for q in np.flatnonzero(state.pairs[k][list(participants)].any(axis=0))
            }
        others = [q for q in range(definition.place_count) if q not in found]
        extra = min(self.settings.extra_places, len(others))
        if extra > 0:
            drawn = rng.choice(len(others), size=extra, replace=False)
            found |= {others[int(i)] for i in drawn}
        return tuple(sorted(found))

    # Steps -----------------------------------------------------------------------------------

    def _initial(self, rng: np.random.Generator, participants: tuple[int, ...]) -> State:
        """The participants' initial values: the definition's, or redrawn (``scene.initial:
        redraw``), the Boolean fluents at their initial rates and then the numeric fluents from
        their initial distributions, participant by participant. The pairs (stage c2a): the
        initial pairs among the participants, every other pair dropped (an episode keeps the
        pairs among its participants, WM.38); under ``redraw``, drawn again among the
        participants (in entity order) at each pair fluent's rate by the procedure of the initial
        pairs, after every other draw of the episode's part."""
        definition = self.definition
        pairs = self._participant_pairs(participants)
        if self.settings.initial == "keep":
            if self.settings.initial_location == "seed":
                pairs = list(pairs)
                self._seed_places(participants, pairs)
            return State(self.start.values, self.start.numeric, tuple(pairs))
        values = np.array(self.start.values, dtype=np.uint8, copy=True)
        numeric = np.array(self.start.numeric, dtype=np.float64, copy=True)
        for entity in participants:
            for j, fluent in enumerate(definition.base_fluents):
                values[entity, j] = 1 if rng.random() < self.initial_rates[fluent] else 0
        for entity in participants:
            for j, fluent in enumerate(definition.numeric_fluents):
                low, high = definition.numeric_ranges[fluent]
                numeric[entity, j] = definition.numeric_initial[fluent].draw(rng, low, high, 1)[0]
        if definition.pair_fluents:
            rows = np.array(sorted(participants), dtype=np.intp)
            redrawn = []
            for fluent, matrix in zip(definition.pair_fluents, pairs, strict=True):
                if fluent.total:
                    redrawn.append(matrix)  # the places are redrawn after every pair (c3a)
                    continue
                local = draw_initial_pairs(fluent, rows.size, rng)
                matrix = np.zeros((definition.entity_count, definition.entity_count), dtype=bool)
                matrix[np.ix_(rows, rows)] = local
                redrawn.append(matrix)
            if definition.has_location_fluents:
                self._redraw_places(rng, participants, redrawn)
                if self.settings.initial_location == "seed":
                    self._seed_places(participants, redrawn)
            pairs = tuple(redrawn)
        return State(values, numeric, pairs)

    def _participant_pairs(self, participants: tuple[int, ...]) -> tuple[np.ndarray, ...]:
        """The definition's initial pairs restricted to the participants; a location fluent's
        matrix (stage c3a) is kept whole, because every entity is somewhere."""
        if not self.definition.pair_fluents:
            return ()
        mask = np.zeros(self.definition.entity_count, dtype=bool)
        mask[list(participants)] = True
        keep = np.outer(mask, mask)
        return tuple(
            matrix if fluent.total else matrix & keep
            for fluent, matrix in zip(self.definition.pair_fluents, self.start.pairs, strict=True)
        )

    def _legal(
        self, state: State, participants: tuple[int, ...], places: tuple[int, ...] = ()
    ) -> dict[str, list]:
        definition = self.definition
        return {
            label: legal_bindings(
                definition,
                state,
                label,
                participants,
                places if definition.event_type(label).place else None,
            )
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
        places = self.episode_places(rng, participants, state)
        labels = definition.entity_labels
        initial = {
            labels[p]: tuple(f for j, f in enumerate(definition.base_fluents) if state.values[p, j])
            for p in participants
        }
        initial_numeric = (
            {
                labels[p]: {
                    f: float(state.numeric[p, j]) for j, f in enumerate(definition.numeric_fluents)
                }
                for p in participants
            }
            if definition.numeric_fluents
            else {}
        )
        between = [
            (f, m) for f, m in zip(definition.pair_fluents, state.pairs, strict=True) if not f.total
        ]
        initial_pairs = (
            tuple(pair_rows([f for f, _ in between], [m for _, m in between], labels))
            if definition.pair_fluents
            else None
        )
        initial_places = None
        if definition.has_location_fluents:
            at = state.places_record(definition)
            initial_places = {labels[p]: at[labels[p]] for p in participants}
        recorded: list[Step] = []
        quiescent = False
        count = 0
        for k in range(1, steps + 1):
            legal = self._legal(state, participants, places)
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
            result = apply_step(definition, state, events)
            after = result.state
            history_events = []
            for event in events:
                count += 1
                binding = event.binding_labels(definition)
                event_label = f"{label}.EVENTINSTANCE.{count}"
                history_events.append(
                    HistoryEvent(
                        event_label,
                        event.event_type,
                        binding["agent"],
                        binding.get("patient"),
                        tuple(
                            _event_changes(definition, state, after, event)
                            + _event_numeric_changes(
                                definition, state, result.after_effects, event, event_label
                            )
                            + pair_changes_of(definition, state, after, event)
                        ),
                    )
                )
            drift = numeric_changes_of(
                definition, result.after_effects, after.numeric, sorted(participants), DRIFT
            )
            recorded.append(
                Step(
                    k,
                    tuple(history_events),
                    {t: len(b) for t, b in legal.items()} if self.record_legal else None,
                    tuple(drift),
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
            initial_numeric,
            initial_pairs,
            tuple(definition.place_labels[q] for q in places)
            if definition.has_location_fluents
            else None,
            initial_places,
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
) -> list[AnyChange]:
    """The base Boolean fluents the event changed: of the fluents it writes, those whose value
    differs between the states (non-interfering events write distinct fluents, so each change
    has one event)."""
    et = definition.event_type(event.event_type)
    entities = sorted({event.binding[et.roles.index(e.role)] for e in et.effects if not e.pair})
    written = {(event.binding[et.roles.index(e.role)], e.fluent) for e in et.effects if not e.pair}
    return [
        change
        for change in changes_of(definition, before, after, entities)
        if (definition.entity_index(change.entity), change.fluent) in written
    ]


def _event_numeric_changes(
    definition: RuntimeDefinition,
    before: State,
    after_effects: np.ndarray,
    event: Event,
    label: str,
) -> list[Change]:
    """The numeric fluents the event's effects changed, judged after the effects of the step and
    before the drift, each with the event as its cause."""
    et = definition.event_type(event.event_type)
    numeric = [e for e in et.effects if e.numeric]
    if not numeric:
        return []
    entities = sorted({event.binding[et.roles.index(e.role)] for e in numeric})
    written = {(event.binding[et.roles.index(e.role)], e.fluent) for e in numeric}
    return [
        change
        for change in numeric_changes_of(
            definition, before.numeric, after_effects, entities, label, [e.fluent for e in numeric]
        )
        if (definition.entity_index(change.entity), change.fluent) in written
    ]


# ---------------------------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------------------------


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
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) or {}
    if seed is None:
        seed = int(config.get("seed", 1))
    strength = float((config.get("places") or {}).get("habitat_strength", DEFAULT_HABITAT_STRENGTH))
    generator = EpisodeGenerator(
        definition,
        Relatedness.from_run(folder, definition),
        settings,
        None,
        record_legal,
        habitat_strength=strength,
    )
    histories = generator.run(seed, episodes)
    path = Path(out) if out is not None else folder / EPISODES_FILE
    write_histories(path, histories)
    return path, histories


__all__ = [
    "EPISODE_STREAM",
    "INITIAL_MODES",
    "LOCATION_MODES",
    "PLACES_MODES",
    "STATS_EPISODES",
    "STATS_STREAM",
    "EpisodeGenerator",
    "EpisodeSettings",
    "Relatedness",
    "load_scene_settings",
    "read_scene_settings",
    "simulate",
]
