"""Scenes and events: things that happen, for event-level sentences and narratives.

The scene generator is a simple stand-in until event schemas or world-simulation logs can supply
events. A scene has participants and a timeline.

**Participants.** A scene starts with a seed instance. The generator then draws ``scene.size``
other instances, without replacement, each with probability proportional to a weighted sum over
its leaf and the seed's leaf: the thematic relatedness of the two leaves, their taxonomic
similarity, and a constant. The weights are ``scene.participant_weights``.

**Timeline.** The number of time steps is drawn from ``scene.steps``. At each step, the number
of events is drawn from a Poisson distribution with mean ``scene.events_per_step``. Each event is
drawn from the pool of possible events among the participants:

- an intransitive event for every participant and every CAN feature the participant has;
- a transitive event for every ordered pair of distinct participants and every verb whose
  relation holds for the pair.

Events are drawn verb first. An event is transitive with probability ``scene.transitive_share``.
Then a CAN feature or a verb is drawn among those with at least one possible event in the scene,
weighted by ``scene.verb_weights``. Then the agent, or the pair of agent and patient, is drawn
uniformly among those for which that verb's event is possible. So a verb that holds for many
pairs is no more frequent than one that holds for few. The same event does not occur twice at
one time step.
Nothing changes state, so the pool is the same at every step, and an event never contradicts a
capacity.

Scenes are a fact about the world and not about the language: they use every CAN feature and
every verb, with a word or without one, so the lexicon never changes a scene. A scene draws from
its own part of the ``corpus:scenes`` stream, named by its label, so one scene never changes
another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.corpus.config import Config, ConfigError
from semantic_world.corpus.propositions import CAN, EVENT, VERB, Predicate, Proposition, Truth
from semantic_world.corpus.streams import Streams
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.similarity import similarity_matrix


@dataclass(frozen=True)
class Event:
    label: str
    """``SN.<scene>.<k>``: event ``k`` of its scene, in time order."""
    scene: str
    step: int
    """The time step, from 1."""
    verb: str
    """A CAN feature (an intransitive event) or a verb (a transitive event)."""
    agent: str
    patient: str | None = None

    @property
    def transitive(self) -> bool:
        return self.patient is not None

    @property
    def key(self) -> tuple[str, str, str | None]:
        """What happened, apart from when: the verb, the agent, and the patient."""
        return (self.verb, self.agent, self.patient)

    def involves(self, instance: str) -> bool:
        return instance in (self.agent, self.patient)

    def proposition(self, verb: str | None = None) -> Proposition:
        """The event-level logical form that reports the event. ``verb`` names it with a verb
        category above its own verb ("hunt" for a chase)."""
        kind = VERB if self.transitive else CAN
        predicate = Predicate(kind, self.verb if verb is None else verb, self.patient)
        return Proposition(EVENT, self.agent, predicate, scene=self.scene, event=self.label)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "verb": self.verb,
            "agent": self.agent,
            "patient": self.patient,
        }


@dataclass(frozen=True)
class Scene:
    label: str
    """``SN.<n>``."""
    seed: str
    participants: tuple[str, ...]
    """The seed instance, then the other participants in the order they were drawn."""
    steps: int
    events: tuple[Event, ...]
    """Every event, in time order, and within a time step in the order it was drawn."""

    def at(self, step: int) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.step == step)

    def involving(self, instance: str) -> tuple[Event, ...]:
        """The events an instance takes part in, as agent or patient, in time order."""
        return tuple(e for e in self.events if e.involves(instance))

    def happened(self, verb: str, agent: str, patient: str | None = None) -> bool:
        return (verb, agent, patient) in {e.key for e in self.events}

    def to_json(self) -> dict[str, Any]:
        """The scene as it is written to ``scenes.jsonl``: the participants, and the events by
        time step."""
        return {
            "label": self.label,
            "seed": self.seed,
            "participants": list(self.participants),
            "steps": [[e.to_json() for e in self.at(step)] for step in range(1, self.steps + 1)],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Scene:
        events = tuple(
            Event(e["label"], data["label"], step, e["verb"], e["agent"], e["patient"])
            for step, listed in enumerate(data["steps"], start=1)
            for e in listed
        )
        return cls(
            data["label"], data["seed"], tuple(data["participants"]), len(data["steps"]), events
        )


class SceneGenerator:
    """The scenes of one world, under one corpus configuration."""

    def __init__(self, config: Config, result: TaxonomyResult, truth: Truth | None = None) -> None:
        self.settings = config.scene
        self.result = result
        self.truth = truth or Truth(config, result)
        instances = result.instances
        self.labels = instances.labels
        self.index = self.truth.instance_index
        leaves = result.tree.leaves
        leaf_rows = np.array([result.tree.categories.index(leaf) for leaf in leaves], dtype=np.intp)
        leaf_number = {int(row): i for i, row in enumerate(leaf_rows)}
        self.leaf = np.array([leaf_number[int(row)] for row in instances.leaf_index], dtype=np.intp)
        """The leaf of every instance, as an index into the leaves."""
        self.thematic = self._thematic([leaf.label for leaf in leaves])
        """The thematic relatedness of every pair of leaves: the sum, over verbs and over both
        directions, of the leaf-pair proportions, as in ``thematic.csv``."""
        self.similarity = self._similarity(leaf_rows)
        """The taxonomic similarity of every pair of leaves, as in ``thematic.csv``."""
        self.can_features = tuple(f.label for f in result.features.of_type("can"))
        self.verbs = () if result.verbs is None else tuple(v.label for v in result.verbs.verbs)
        self.verb_weight = self._verb_weights(config)

    def _thematic(self, leaf_labels: list[str]) -> np.ndarray:
        matrix = np.zeros((len(leaf_labels), len(leaf_labels)))
        if self.result.relation_stats is not None:
            number = {label: i for i, label in enumerate(leaf_labels)}
            for row in self.result.relation_stats.thematic.iter_rows(named=True):
                a, b = number[row["leaf_a"]], number[row["leaf_b"]]
                matrix[a, b] = matrix[b, a] = row["thematic"]
        return matrix

    def _similarity(self, leaf_rows: np.ndarray) -> np.ndarray:
        vectors = self.result.vectors
        analysis = self.result.config.analysis
        start = 0 if analysis.similarity_features == "all" else vectors.isa_count
        matrix = similarity_matrix(
            vectors.generative[leaf_rows][:, start:], analysis.similarity_metric
        )
        # an undefined similarity, or a negative one (phi), adds nothing to a weight
        return np.clip(np.nan_to_num(matrix, nan=0.0), 0.0, None)

    def _verb_weights(self, config: Config) -> dict[str, float]:
        weights = dict.fromkeys(self.can_features + self.verbs, 1.0)
        for label, weight in (self.settings.verb_weights or {}).items():
            if label not in weights:
                raise ConfigError(
                    config.source,
                    f"scene.verb_weights.{label}",
                    "is not a CAN feature or a verb of the taxonomy",
                )
            weights[label] = weight
        return weights

    # Participants ----------------------------------------------------------------------------

    def participant_weights(self, seed: str) -> np.ndarray:
        """The weight of every instance as a participant of a scene seeded at ``seed``: a
        weighted sum of the thematic relatedness and the taxonomic similarity of its leaf and
        the seed's leaf, and a constant. The seed itself has the weight 0."""
        weights = self.settings.participant_weights
        seed_index = self.index[seed]
        seed_leaf = self.leaf[seed_index]
        values = (
            weights["thematic"] * self.thematic[self.leaf, seed_leaf]
            + weights["taxonomic"] * self.similarity[self.leaf, seed_leaf]
            + weights["constant"]
        )
        values[seed_index] = 0.0
        return values

    def draw_participants(self, rng: np.random.Generator, seed: str) -> tuple[str, ...]:
        """The seed, then ``scene.size`` other instances drawn without replacement by their
        weights. An instance with the weight 0 is never drawn, so a scene can be smaller than
        asked."""
        size = int(rng.integers(self.settings.size.min, self.settings.size.max + 1))
        weights = self.participant_weights(seed)
        size = min(size, int(np.count_nonzero(weights)))
        if size == 0:
            return (seed,)
        drawn = rng.choice(len(weights), size=size, replace=False, p=weights / weights.sum())
        return (seed,) + tuple(self.labels[int(i)] for i in drawn)

    # Events ----------------------------------------------------------------------------------

    def possible_events(
        self, participants: tuple[str, ...]
    ) -> tuple[list[tuple[str, str, None]], list[tuple[str, str, str]]]:
        """Every event the world allows among the participants, as ``(verb, agent, patient)``:
        the intransitive events (a participant and a CAN feature it has), then the transitive
        events (an ordered pair of distinct participants and a verb whose relation holds)."""
        rows = [self.index[p] for p in participants]
        features = self.result.features
        intransitive = [
            (feature, participant, None)
            for participant, row in zip(participants, rows, strict=True)
            for feature in self.can_features
            if self.truth.values[row, features[feature].position]
        ]
        transitive = []
        for verb in self.verbs:
            holds = self.truth.matrix(verb)
            transitive += [
                (verb, agent, patient)
                for agent, a in zip(participants, rows, strict=True)
                for patient, p in zip(participants, rows, strict=True)
                if a != p and holds[a, p]
            ]
        return intransitive, transitive

    def _by_verb(self, pool: list) -> dict[str, list]:
        """The possible events of one kind, by their CAN feature or verb, in the pool's order. A
        verb with the weight 0 is left out."""
        grouped: dict[str, list] = {}
        for event in pool:
            if self.verb_weight[event[0]] > 0:
                grouped.setdefault(event[0], []).append(event)
        return grouped

    def _draw_step(
        self, rng: np.random.Generator, pools: tuple[list, list]
    ) -> list[tuple[str, str, str | None]]:
        """The events of one time step: a Poisson number of distinct events, each drawn verb
        first. The kind is drawn by ``scene.transitive_share``. Then a CAN feature or a verb is
        drawn, by its weight, among those with at least one possible event left at this step.
        Then the agent, or the pair of agent and patient, is drawn uniformly among those for
        which that verb's event is possible."""
        count = int(rng.poisson(self.settings.events_per_step))
        share = self.settings.transitive_share
        left = [self._by_verb(pool) for pool in pools]
        drawn = []
        for _ in range(count):
            intransitive = bool(left[0]) and share < 1
            transitive = bool(left[1]) and share > 0
            if not (intransitive or transitive):
                break
            kind = 1 if transitive and (not intransitive or rng.random() < share) else 0
            verbs = list(left[kind])
            weights = np.array([self.verb_weight[verb] for verb in verbs], dtype=float)
            verb = verbs[int(rng.choice(len(verbs), p=weights / weights.sum()))]
            options = left[kind][verb]
            # the same event does not occur twice at one time step
            drawn.append(options.pop(int(rng.integers(len(options)))))
            if not options:
                del left[kind][verb]
        return drawn

    def generate(self, rng: np.random.Generator, seed: str, label: str) -> Scene:
        """One scene seeded at an instance, drawn from ``rng``. The scene is made known to the
        truth tests, so event-level propositions about it can be judged."""
        if seed not in self.index:
            raise KeyError(f"unknown instance {seed!r}")
        participants = self.draw_participants(rng, seed)
        steps = int(rng.integers(self.settings.steps.min, self.settings.steps.max + 1))
        pools = self.possible_events(participants)
        events: list[Event] = []
        for step in range(1, steps + 1):
            for verb, agent, patient in self._draw_step(rng, pools):
                events.append(
                    Event(f"{label}.{len(events) + 1}", label, step, verb, agent, patient)
                )
        scene = Scene(label, seed, participants, steps, tuple(events))
        self.truth.add_scene(scene)
        return scene

    def scene(self, streams: Streams, number: int, seed: str) -> Scene:
        """Scene ``SN.<number>``, seeded at an instance. The scene draws from its own part of the
        ``corpus:scenes`` stream, so it depends only on the corpus seed, its number, its seed
        instance, and the scene settings."""
        label = f"SN.{number}"
        return self.generate(streams.substream("scenes", label), seed, label)

    # Statistics ------------------------------------------------------------------------------

    def relatedness(self, scene: Scene) -> tuple[float, float]:
        """The mean thematic relatedness, and the mean taxonomic similarity, of the seed's leaf
        and the leaf of every other participant. NaN for a scene with no other participant."""
        seed_leaf = self.leaf[self.index[scene.seed]]
        others = [self.leaf[self.index[p]] for p in scene.participants[1:]]
        if not others:
            return float("nan"), float("nan")
        return (
            float(self.thematic[others, seed_leaf].mean()),
            float(self.similarity[others, seed_leaf].mean()),
        )
