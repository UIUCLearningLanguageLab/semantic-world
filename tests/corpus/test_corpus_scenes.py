"""Stage 3 and stage a5a acceptance tests: scenes and events.

Scenes are the world's episodes, recorded as histories. Every event was legal in its step's
starting state, by the brute-force evaluator of the world run's files; no event repeats within a
step; every history replays on the run's definition; with the thematic weight above 0,
participants are more thematically related than with the weight at 0, on average; and on a
world without fluents, under the balanced policy, scenes have the participants the old scene
generator drew for the same seeds.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pytest
from corpus_support import corpus_config, old_scene_participants

from semantic_world.corpus import ConfigError, Streams, load_world
from semantic_world.corpus.histories import (
    SceneGenerator,
    events_at,
    happened_in,
    involving,
    scene_events,
)
from semantic_world.corpus.propositions import (
    CAN,
    EVENT,
    INSTANCE,
    IS,
    PROGRESSIVE,
    SIMPLE,
    VERB,
    Predicate,
    Proposition,
)
from semantic_world.world.history import History, replay
from semantic_world.world.runtime import legal

WORLDS = ("tiny", "default", "deep", "still")
BALANCED = {"policy": "uniform_event_type"}


def make(generator: SceneGenerator, count: int, seed: int = 1, first: int = 1) -> list[History]:
    """Scenes ``SCENE.<first>`` onward, each seeded at an instance drawn from a fixed order."""
    streams = Streams(seed)
    labels = generator.world.instances
    return [
        generator.scene(streams, number, labels[(number * 7) % len(labels)])
        for number in range(first, first + count)
    ]


def scene_settings(**scene) -> dict:
    return {"scene": scene}


def knowing(facts, scenes: list[History]):
    """Facts of other settings have truth tests of their own: make the scenes known to them."""
    for scene in scenes:
        facts.truth.add_scene(scene)
    return facts


def event_keys(scene: History) -> list[tuple]:
    return [e.key for e in scene_events(scene)]


# ---------------------------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_event_was_legal_by_the_output_files(cases, name) -> None:
    case = cases(name)
    oracle = case.oracle()
    scenes = make(case.scenes(), 300)
    events = [e for scene in scenes for e in scene_events(scene)]
    assert len(events) > 1000
    for scene in scenes:
        states = oracle.states(scene.to_json())  # replays with the brute-force evaluator
        assert len(states) == len(scene.steps) + 1
        for event in scene_events(scene):
            assert event.agent != event.patient
            assert oracle.able(event.type, event.agent, event.patient), event
            assert oracle.legal(scene.to_json(), event.type, event.agent, event.patient)
    kinds = Counter(e.transitive for e in events)
    assert kinds[True] > 100 and kinds[False] > 100


@pytest.mark.parametrize("name", WORLDS)
def test_every_scene_replays_on_the_runs_definition(cases, name) -> None:
    case = cases(name)
    definition = case.world.definition
    for scene in make(case.scenes(), 100):
        states = replay(definition, scene)
        assert scene.rule_set_id == definition.rule_set_id
        assert scene.final == f"TIME.{len(states)}"
        participants = tuple(definition.entity_index(p) for p in scene.participants)
        for step, state in zip(scene.steps, states[:-1], strict=True):
            for event in step.events:
                binding = tuple(
                    definition.entity_index(x) for x in (event.agent, event.patient) if x
                )
                assert legal(definition, state, event.type, [binding])[0]
                assert set(binding) <= set(participants)


@pytest.mark.parametrize("name", WORLDS)
def test_no_event_repeats_within_a_step(cases, name) -> None:
    case = cases(name)
    # many events at every step, so that a repeat would show
    scenes = make(case.scenes(**scene_settings(events_per_step=6.0)), 200)
    steps = repeats_across_steps = 0
    for scene in scenes:
        for step in scene.steps:
            keys = [e.key for e in events_at(scene, step.step)]
            assert len(set(keys)) == len(keys), (scene.label, step.step)
            steps += 1
        keys = event_keys(scene)
        repeats_across_steps += len(keys) - len(set(keys))
    assert steps > 500
    assert repeats_across_steps > 0  # the same event can happen again at a later step


def test_the_thematic_weight_draws_thematically_related_participants(cases) -> None:
    case = cases("default")

    def mean_relatedness(**weights) -> tuple[float, float]:
        generator = case.scenes(**scene_settings(participant_weights=weights))
        return tuple(np.nanmean([generator.relatedness(s) for s in make(generator, 600)], axis=0))

    without = mean_relatedness(thematic=0, taxonomic=0.5, constant=0.1)
    default = mean_relatedness(thematic=1.0, taxonomic=0.5, constant=0.1)
    strong = mean_relatedness(thematic=5.0, taxonomic=0.5, constant=0.1)
    only = mean_relatedness(thematic=1.0, taxonomic=0, constant=0)
    assert without[0] < default[0] < strong[0] <= only[0]
    assert default[0] - without[0] > 0.15
    # the taxonomic weight does the same for taxonomic similarity
    plain = mean_relatedness(thematic=0, taxonomic=0, constant=1)
    similar = mean_relatedness(thematic=0, taxonomic=3.0, constant=0.1)
    assert similar[1] > plain[1] + 0.03
    # with every weight but the constant at 0, participants are a uniform draw
    assert abs(plain[0] - case.world.thematic.mean()) < 0.1


def test_participants_are_the_old_scene_generators_on_a_static_world(world_files) -> None:
    """With no fluents and the balanced policy, a scene draws the participants the corpus's
    scene generator drew before stage a5a for the same seed instance and part of the stream.
    The events need not agree: an episode draws its number of steps and its initial state from
    the part before the first step, where the old generator drew the participants and then the
    steps, so the later draws diverge."""
    world = load_world(corpus_config(world_files["static"]))
    assert world.definition.base_fluents == ()
    generator = SceneGenerator(corpus_config(world_files["static"], scene=BALANCED), world)
    weights, draw = old_scene_participants(world, generator.generator.settings)
    streams = Streams(1)
    for number, seed in enumerate(world.instances, start=1):
        label = f"SCENE.{number}"
        assert np.allclose(generator.participant_weights(seed), weights(number - 1))
        rng = streams.substream("scenes", label)
        expected = draw(rng, number - 1)
        scene = generator.scene(streams, number, seed)
        assert scene.participants == tuple(world.instances[i] for i in expected)
        assert scene.policy == "uniform_event_type"


# ---------------------------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_participants(cases, name) -> None:
    case = cases(name)
    scenes = make(case.scenes(), 300)
    sizes = Counter()
    for number, scene in enumerate(scenes, start=1):
        assert scene.label == f"SCENE.{number}"
        assert scene.participants[0] == scene.seed
        assert len(set(scene.participants)) == len(scene.participants)
        assert set(scene.participants) <= set(case.world.instances)
        sizes[len(scene.participants) - 1] += 1
    # scene.size other instances, drawn from the range
    assert set(sizes) == {2, 3, 4, 5, 6}
    assert min(sizes.values()) > 300 / 5 * 0.5


def test_participant_weights_are_the_weighted_sum(cases) -> None:
    case = cases("default")
    weights = {"thematic": 2.0, "taxonomic": 0.7, "constant": 0.05}
    generator = case.scenes(**scene_settings(participant_weights=weights))
    world = case.world
    thematic = world.result.derived_frames()["thematic.csv"]
    # thematic.csv gives both numbers for every pair of leaves with a thematic score above 0
    score = {}
    for row in thematic.iter_rows(named=True):
        score[row["leaf_a"], row["leaf_b"]] = score[row["leaf_b"], row["leaf_a"]] = (
            row["thematic"],
            row["similarity"],
        )
    assert len(score) > 100
    for seed in world.instances[::37]:
        values = generator.participant_weights(seed)
        seed_leaf = world.instance_leaf[world.instance_index[seed]]
        assert values[world.instance_index[seed]] == 0  # the seed is not drawn again
        for row in range(0, world.count, 5):
            if world.instances[row] == seed:
                continue
            pair = (world.instance_leaf[row], seed_leaf)
            if pair in score:
                expected = 2.0 * score[pair][0] + 0.7 * score[pair][1] + 0.05
                assert values[row] == pytest.approx(expected, abs=2e-6)
            else:
                # no thematic score: the similarity and the constant are left
                assert 0.05 <= values[row] <= 0.05 + 0.7 + 1e-9


def test_an_instance_with_no_weight_is_never_drawn(cases) -> None:
    case = cases("default")
    weights = {"thematic": 1.0, "taxonomic": 0, "constant": 0}
    generator = case.scenes(**scene_settings(participant_weights=weights, size=[283, 283]))
    smaller = 0
    for scene in make(generator, 60):
        drawn = generator.participant_weights(scene.seed)
        for participant in scene.participants[1:]:
            assert drawn[case.world.instance_index[participant]] > 0
        related = int(np.count_nonzero(drawn))
        assert len(scene.participants) - 1 == related  # every related instance, and no other
        smaller += related < 283
    assert smaller > 0  # a scene can be smaller than asked


def test_a_scene_never_has_more_participants_than_the_world(cases) -> None:
    case = cases("tiny")
    generator = case.scenes(**scene_settings(size=[30, 30]))
    for scene in make(generator, 20):
        assert sorted(scene.participants) == sorted(case.world.instances)
    alone = case.scenes(**scene_settings(size=0))
    for scene in make(alone, 20):
        assert scene.participants == (scene.seed,)
        assert all(not e.transitive and e.agent == scene.seed for e in scene_events(scene))


# ---------------------------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------------------------


def test_the_timeline_on_a_static_world(world_files) -> None:
    # Without fluents every able binding is legal at every step, so the timeline is the old
    # one: a drawn number of steps, and a Poisson number of events at each step.
    world = load_world(corpus_config(world_files["static"]))
    generator = SceneGenerator(corpus_config(world_files["static"], scene=BALANCED), world)
    scenes = make(generator, 400)
    steps = Counter(len(scene.steps) for scene in scenes)
    assert set(steps) == {3, 4, 5, 6, 7, 8}
    assert not any(scene.quiescent for scene in scenes)
    per_step = []
    for scene in scenes:
        events = scene_events(scene)
        # events are labeled in time order, and a step can be empty
        assert [e.label for e in events] == [
            f"{scene.label}.EVENTINSTANCE.{k}" for k in range(1, len(events) + 1)
        ]
        assert [e.step for e in events] == sorted(e.step for e in events)
        assert all(1 <= e.step <= len(scene.steps) and e.scene == scene.label for e in events)
        assert all(
            e.involves(e.agent)
            and set(filter(None, (e.agent, e.patient))) <= set(scene.participants)
            for e in events
        )
        assert scene.final == f"TIME.{len(scene.steps) + 1}"
        per_step += [len(step.events) for step in scene.steps]
    # a Poisson number of events at each step, with the configured mean
    assert abs(np.mean(per_step) - 1.5) < 0.1
    assert abs(np.var(per_step) - 1.5) < 0.3
    assert 0 in per_step and max(per_step) >= 5


@pytest.mark.parametrize("name", WORLDS)
def test_the_timeline_with_fluents(cases, name) -> None:
    # With fluents, a step holds only events that are legal and do not interfere, and a scene
    # ends early, quiescent, when nothing is legal.
    case = cases(name)
    scenes = make(case.scenes(), 400)
    steps = Counter(len(scene.steps) for scene in scenes)
    assert max(steps) == 8 and min(steps) >= 0
    quiescent = [scene for scene in scenes if scene.quiescent]
    for scene in scenes:
        if not scene.quiescent:
            assert 3 <= len(scene.steps) <= 8
        for event in scene_events(scene):
            assert event.scene == scene.label
    per_step = [len(step.events) for scene in scenes for step in scene.steps]
    assert np.mean(per_step) <= 1.5 + 0.1  # never more than the Poisson draw asks for
    assert 0 in per_step
    assert len(quiescent) < len(scenes)
    # events change state: some event records a change to a base fluent
    changes = [c for scene in scenes for e in scene.events for c in e.changes]
    assert changes and all(c.fluent.startswith("BOOLFL.") for c in changes)


def test_the_able_bindings(cases) -> None:
    case = cases("tiny")
    world = case.world
    participants = ("INSTANCE.1.1.1", "INSTANCE.1.2.2", "INSTANCE.2.1.3", "INSTANCE.2.2.1")
    rows = [world.instance_index[p] for p in participants]
    # every participant and every one-place event type it is able to be the agent of
    for event_type in world.unary:
        able = world.able(event_type)
        for row in rows:
            assert able[row] == bool(world.column(event_type)[row])
    # every ordered pair of distinct participants and every two-place leaf event type whose
    # requirement holds; a category names no event of its own
    taxonomy = world.result.taxonomy
    for label in world.binary_leaves:
        old = label.replace("EVENTTYPE2.", "V")
        for a, row_a in zip(participants, rows, strict=True):
            for p, row_p in zip(participants, rows, strict=True):
                if a != p:
                    expected = taxonomy.relations.holds(old, [row_a], [row_p])[0]
                    assert world.able(label)[row_a, row_p] == expected
    assert set(world.event_types_below("EVENTTYPE2.1")) == {"EVENTTYPE2.1.1", "EVENTTYPE2.1.2"}


def test_the_transitive_share_under_the_balanced_policy(world_files) -> None:
    # The balanced policy draws the kind of event by the share when both kinds have an available
    # event, and draws the other kind otherwise. The share is measured over the scenes whose
    # participants have both an able agent and an able pair.
    world = load_world(corpus_config(world_files["static"]))
    index = world.instance_index

    def both_kinds(scene: History) -> bool:
        rows = [index[p] for p in scene.participants]
        one = any(world.able(u)[r] for u in world.unary for r in rows)
        two = any(
            world.able(v)[a, b] for v in world.binary_leaves for a in rows for b in rows if a != b
        )
        return one and two

    def share(value: float, **extra) -> float:
        settings = {"transitive_share": value, **BALANCED, **extra}
        generator = SceneGenerator(corpus_config(world_files["static"], scene=settings), world)
        scenes = [s for s in make(generator, 300) if extra or both_kinds(s)]
        return float(np.mean([e.transitive for s in scenes for e in scene_events(s)]))

    assert share(0.0) == 0.0 and share(1.0) == 1.0
    assert abs(share(0.5) - 0.5) < 0.05
    assert abs(share(0.8) - 0.8) < 0.05
    # a scene with one participant has no transitive event, whatever the share
    assert share(0.9, size=0) == 0.0
    # the default policy draws uniformly among all legal events, so the share follows the
    # numbers of able agents and able pairs among the participants of each scene
    plain = SceneGenerator(corpus_config(world_files["static"]), world)
    scenes = make(plain, 300)
    uniform = float(np.mean([e.transitive for s in scenes for e in scene_events(s)]))
    agents = pairs = 0
    for scene in scenes:
        rows = [index[p] for p in scene.participants]
        agents += sum(int(world.able(u)[rows].sum()) for u in world.unary)
        pairs += sum(int(world.able(v)[np.ix_(rows, rows)].sum()) for v in world.binary_leaves)
    expected = pairs / (pairs + agents)
    assert abs(uniform - expected) < 0.05 and 0.3 < expected < 0.6


def test_the_balanced_policy_draws_the_event_type_first(world_files) -> None:
    # With every instance of the tiny world in the scene, the able pool is the same in every
    # scene, and the event types differ widely in how many pairs they hold for.
    world = load_world(corpus_config(world_files["static"]))
    settings = {"size": [11, 11], "events_per_step": 1.0, "steps": [4, 4], **BALANCED}
    generator = SceneGenerator(corpus_config(world_files["static"], scene=settings), world)
    off_diagonal = ~np.eye(world.count, dtype=bool)
    pairs = {label: int(world.able(label)[off_diagonal].sum()) for label in world.binary_leaves}
    agents = {label: int(world.able(label).sum()) for label in world.unary}
    agents = {label: count for label, count in agents.items() if count}  # one is able for nobody
    assert max(pairs.values()) > 3 * min(pairs.values())
    events = [e for s in make(generator, 1500) for e in scene_events(s)]
    drawn = Counter(e.type for e in events)
    # each event type with an able event is equally likely, however many pairs it holds for
    for counts in (pairs, agents):
        share = np.array([drawn[label] for label in counts], dtype=float)
        share /= share.sum()
        assert np.abs(share - 1 / len(counts)).max() < 0.03, dict(zip(counts, share, strict=True))
    # and within an event type, each able pair is equally likely
    label = max(pairs, key=pairs.get)
    by_pair = Counter(e.key for e in events if e.type == label)
    assert len(by_pair) == pairs[label]
    expected = drawn[label] / pairs[label]
    assert all(abs(count - expected) < 5 * np.sqrt(expected) for count in by_pair.values())
    # the weights are weights of event types, not of events
    weighted = SceneGenerator(
        corpus_config(
            world_files["static"],
            scene={**settings, "event_type_weights": {"EVENTTYPE2.1.1": 3}},
        ),
        world,
    )
    counts = Counter(e.type for s in make(weighted, 1500) for e in scene_events(s) if e.transitive)
    assert abs(counts["EVENTTYPE2.1.1"] / counts["EVENTTYPE2.2.1"] - 3) < 0.5
    # under the default policy, an event type legal for many bindings is drawn more often
    plain = SceneGenerator(
        corpus_config(
            world_files["static"], scene={k: v for k, v in settings.items() if k != "policy"}
        ),
        world,
    )
    uniform = Counter(e.type for s in make(plain, 1500) for e in scene_events(s))
    assert uniform[max(pairs, key=pairs.get)] > 2 * uniform[min(pairs, key=pairs.get)]


def test_event_type_weights(world_files) -> None:
    # In the static world, every able binding is legal. (In the tiny world, the preconditions
    # of some event types are never met, so those event types never occur.)
    world = load_world(corpus_config(world_files["static"]))

    def scenes(**scene) -> SceneGenerator:
        return SceneGenerator(corpus_config(world_files["static"], scene=scene), world)

    uniform = Counter(e.type for s in make(scenes(), 300) for e in scene_events(s))
    assert {"EVENTTYPE1.1", "EVENTTYPE1.3", "EVENTTYPE2.1.2", "EVENTTYPE2.2.1"} <= set(uniform)
    weights = {"EVENTTYPE1.1": 0, "EVENTTYPE2.1.2": 20, "EVENTTYPE2.2.1": 0}
    weighted = scenes(event_type_weights=weights)
    counts = Counter(e.type for s in make(weighted, 300) for e in scene_events(s))
    # a weight of 0 keeps an event type out, and a large weight brings it in more often
    assert counts["EVENTTYPE1.1"] == 0 and counts["EVENTTYPE2.2.1"] == 0
    assert counts["EVENTTYPE2.1.2"] > 2 * uniform["EVENTTYPE2.1.2"]
    assert counts["EVENTTYPE1.3"] > 0  # a label that is left out keeps the weight 1
    # the labels are checked against the world: a category names no event of its own
    for label in ("EVENTTYPE2.2", "EVENTTYPE1.9", "PROPERTY.1", "EVENTTYPE2.9.1", "CAN.1"):
        with pytest.raises(ConfigError) as info:
            scenes(event_type_weights={label: 1})
        assert info.value.field == f"scene.event_type_weights.{label}"
        assert "not an event type" in info.value.message


def test_events_per_step(cases) -> None:
    case = cases("default")
    assert all(
        not scene_events(s) for s in make(case.scenes(**scene_settings(events_per_step=0)), 30)
    )
    busy = make(case.scenes(**scene_settings(events_per_step=4.0, steps=[5, 5])), 200)
    assert all(len(scene.steps) == 5 or scene.quiescent for scene in busy)
    assert np.mean([len(scene_events(s)) for s in busy]) > 5


def test_initial_states(cases) -> None:
    case = cases("tiny")
    world = case.world
    kept = make(case.scenes(), 60)
    initial = world.definition.initial_values
    for scene in kept:
        for participant, fluents in scene.initial.items():
            row = world.instance_index[participant]
            expected = [f for j, f in enumerate(world.definition.base_fluents) if initial[row, j]]
            assert list(fluents) == expected
    redrawn = make(case.scenes(**scene_settings(initial="redraw")), 60)
    assert [s.participants for s in redrawn] == [s.participants for s in kept]
    assert any(a.initial != b.initial for a, b in zip(kept, redrawn, strict=True))


def test_a_world_without_two_place_event_types_has_one_place_events_only(world_files) -> None:
    config = corpus_config(world_files["plain"])
    world = load_world(config)
    generator = SceneGenerator(config, world)
    scenes = make(generator, 100)
    events = [e for s in scenes for e in scene_events(s)]
    assert events and not any(e.transitive for e in events)
    assert world.binary == () and not world.thematic.any()
    # with two-place event types off, the thematic weight has no effect
    weights = {"thematic": 9.0, "taxonomic": 0.5, "constant": 0.1}
    other = SceneGenerator(
        corpus_config(world_files["plain"], scene={"participant_weights": weights}), world
    )
    assert make(other, 100) == scenes


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_scenes(cases) -> None:
    case = cases("default")
    generator = case.scenes()
    first = make(generator, 50, seed=3)
    assert make(generator, 50, seed=3) == first
    assert make(generator, 50, seed=4) != first
    # a scene draws from its own part of the stream: it does not depend on the scenes before it
    assert make(generator, 5, seed=3, first=30) == first[29:34]
    streams = Streams(3)
    alone = generator.scene(streams, 17, first[16].seed)
    assert alone == first[16]
    # another seed instance gives another scene with the same label
    other = generator.scene(streams, 17, first[0].seed)
    assert other.label == "SCENE.17" and other != first[16]
    with pytest.raises(KeyError, match="unknown instance"):
        generator.scene(streams, 1, "INSTANCE.9.9.9")


def test_only_the_scene_settings_change_a_scene(cases) -> None:
    case = cases("default")
    base = make(case.scenes(), 40)
    # the lexicon, the grammar, and the other sections leave every scene as it is: scenes are a
    # fact about the world, and use every event type, with a word or without one
    half = dict.fromkeys(("category", "event_unary", "event", "event_category"), 0.5)
    others = {
        "lexicon": {"named_proportion": half, "synonym_rate": 0.4},
        "grammar": {"word_order": {"clause": "SOV"}, "morphology": {"tense": {"enabled": True}}},
        "mention": {"pronoun_rate": 0.9, "event_level_weights": 1},
        "propositions": {"negation_rate": {"class": 0.5}},
        "documents": {"progressive_rate": 0.9, "one_aspect_per_event": False},
        "test_sets": {"size": 3},
    }
    assert make(case.scenes(**others), 40) == base
    assert make(case.scenes(**scene_settings(events_per_step=1.6)), 40) != base


# ---------------------------------------------------------------------------------------------
# The scene record
# ---------------------------------------------------------------------------------------------


def test_scene_json_is_a_history(cases) -> None:
    case = cases("tiny")
    for scene in make(case.scenes(), 50):
        form = scene.to_json()
        assert list(form) == [
            "label", "seed", "participants", "policy", "rule_set_id", "initial", "steps",
            "final", "quiescent",
        ]  # fmt: skip
        assert form["policy"] == "uniform_event"
        assert form["rule_set_id"] == case.world.rule_set_id
        assert list(form["initial"]) == list(form["participants"])
        for k, step in enumerate(form["steps"], start=1):
            assert step["step"] == k and "legal" not in step
            for event in step["events"]:
                assert list(event)[:3] == ["label", "type", "agent"]
                assert list(event)[-1] == "changes"
                assert "aspect" not in event
        assert History.from_json(json.loads(json.dumps(form))) == scene


def test_scene_lookups(cases) -> None:
    scene = next(
        s for s in make(cases("tiny").scenes(), 50) if any(e.transitive for e in scene_events(s))
    )
    events = scene_events(scene)
    assert all(e.scene == scene.label for e in events)
    for step in scene.steps:
        assert [e.label for e in events_at(scene, step.step)] == [e.label for e in step.events]
    event = next(e for e in events if e.transitive)
    assert happened_in(scene, event.type, event.agent, event.patient)
    assert not happened_in(scene, event.type, event.agent, "INSTANCE.9.9.9")
    assert event in involving(scene, event.agent) and event in involving(scene, event.patient)
    assert event.key == (event.type, event.agent, event.patient)
    assert all(e.involves(e.agent) for e in events)


# ---------------------------------------------------------------------------------------------
# Event-level propositions
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_event_is_reported_by_a_true_proposition(cases, name) -> None:
    case = cases(name)
    facts = case.facts()
    truth = facts.truth
    for scene in make(case.scenes(), 100):
        truth.add_scene(scene)
        assert truth.scenes[scene.label] is scene
        for event in scene_events(scene):
            for aspect in (SIMPLE, PROGRESSIVE):
                proposition = facts.event_fact(event, aspect=aspect)
                if proposition is None:
                    continue  # no word
                assert proposition.level == EVENT and proposition.polarity
                assert (proposition.scene, proposition.event) == (scene.label, event.label)
                assert proposition.subject == event.agent and proposition.aspect == aspect
                assert proposition.predicate == Predicate(
                    VERB if event.transitive else CAN, event.type, event.patient
                )
                assert proposition.grounding == {
                    "scene": scene.label,
                    "step": event.step,
                    "able": True,
                    "legal": True,
                    "test": "event",
                }


def test_event_logical_forms(cases) -> None:
    case = cases("tiny")
    facts = case.facts()
    truth = facts.truth
    scenes = make(case.scenes(), 50)
    for scene in scenes:
        truth.add_scene(scene)
    scene = next(s for s in scenes if any(e.transitive for e in scene_events(s)))
    event = next(e for e in scene_events(scene) if e.transitive)
    proposition = facts.event_fact(event, aspect=PROGRESSIVE)
    assert proposition.to_json() == {
        "id": None,
        "level": "event",
        "scene": scene.label,
        "event": event.label,
        "tense": "past",
        "aspect": "progressive",
        "polarity": True,
        "subject": {"instance": event.agent},
        "predicate": {"kind": "verb", "verb": event.type, "patient": {"instance": event.patient}},
        "grounding": {
            "scene": scene.label,
            "step": event.step,
            "able": True,
            "legal": True,
            "test": "event",
        },
    }
    restored = Proposition.from_json(json.loads(json.dumps(proposition.to_json())))
    assert restored == proposition and restored.grounding == proposition.grounding
    # the event is part of what the proposition says: two reports of two events differ
    assert proposition != dataclasses.replace(proposition, event=f"{scene.label}.EVENTINSTANCE.999")
    assert proposition.concepts() == (event.type,)
    intransitive = next(e for s in scenes for e in scene_events(s) if not e.transitive)
    assert facts.event_fact(intransitive).to_json()["predicate"] == {
        "kind": "can",
        "feature": intransitive.type,
    }


def test_an_event_is_true_only_if_it_happened_in_its_scene(cases) -> None:
    case = cases("default")
    facts = case.facts()
    truth = facts.truth
    oracle = case.oracle()
    scenes = make(case.scenes(), 60)
    for scene in scenes:
        truth.add_scene(scene)
    not_happened = {True: 0, False: 0}
    for scene in scenes:
        happened = {e.key for e in scene_events(scene)}
        record = scene.to_json()
        for verb in facts.verbs[:3]:
            for agent in scene.participants:
                for patient in scene.participants:
                    if agent == patient:
                        continue
                    key = (verb, agent, patient)
                    claim = Proposition(
                        EVENT,
                        agent,
                        Predicate(VERB, verb, patient),
                        scene=scene.label,
                        tense="past",
                        aspect="simple",
                    )
                    evaluation = truth.evaluate(claim)
                    assert evaluation.valid
                    expected = any(
                        e.agent == agent
                        and e.patient == patient
                        and verb in truth.verb_names(e.type)
                        for e in scene_events(scene)
                    )
                    assert evaluation.true == expected
                    # both aspects are true of any event that occurred
                    ongoing = dataclasses.replace(claim, aspect="progressive")
                    assert truth.evaluate(ongoing).true == expected
                    # the grounding says whether the binding is able, and whether it was legal
                    # at some time point of the scene, by the world run's files too
                    assert evaluation.grounding["able"] == oracle.able(verb, agent, patient)
                    assert evaluation.grounding["legal"] == oracle.legal(
                        record, verb, agent, patient
                    )
                    if evaluation.true:
                        assert evaluation.grounding["able"] and evaluation.grounding["legal"]
                    assert ("step" in evaluation.grounding) == evaluation.true
                    if key not in happened:
                        not_happened[evaluation.grounding["able"]] += 1
        for feature in facts.features[CAN][:4]:
            for agent in scene.participants:
                claim = Proposition(
                    EVENT,
                    agent,
                    Predicate(CAN, feature),
                    scene=scene.label,
                    tense="past",
                    aspect="simple",
                )
                evaluation = truth.evaluate(claim)
                assert evaluation.true == ((feature, agent, None) in happened)
                assert evaluation.grounding["able"] == oracle.able(feature, agent, None)
                assert evaluation.grounding["legal"] == oracle.legal(record, feature, agent, None)
    assert not_happened[True] > 50 and not_happened[False] > 50
    # what happened in one scene did not happen in another
    moved = 0
    for scene in scenes:
        for event in scene_events(scene)[:3]:
            for other in scenes:
                if (
                    other is not scene
                    and event.agent in other.participants
                    and event.patient in (None, *other.participants)
                    and not happened_in(other, *event.key)
                ):
                    report = facts.event_fact(event)
                    if report is None:
                        continue
                    claim = dataclasses.replace(report, scene=other.label, event=None)
                    evaluation = truth.evaluate(claim)
                    assert evaluation.valid and not evaluation.true
                    assert evaluation.grounding["able"] is True
                    moved += 1
    assert moved > 0


def test_a_report_names_one_event(cases) -> None:
    case = cases("default")
    facts = case.facts(scene={"events_per_step": 5.0})
    truth = facts.truth
    generator = case.scenes(**scene_settings(events_per_step=5.0))
    scenes = make(generator, 100)
    for scene in scenes:
        truth.add_scene(scene)
    scene = next(s for s in scenes if len({e.key for e in scene_events(s)}) < len(scene_events(s)))
    keys = Counter(e.key for e in scene_events(scene))
    repeated = next(key for key, count in keys.items() if count > 1)
    again = [e for e in scene_events(scene) if e.key == repeated]
    other = next(e for e in scene_events(scene) if e.key != repeated)
    report = facts.event_fact(again[1])
    assert truth.evaluate(report).grounding["step"] == again[1].step
    # without an event label, the first such event grounds the proposition
    unlabeled = dataclasses.replace(report, event=None)
    assert truth.evaluate(unlabeled).grounding["step"] == again[0].step
    # a label of another event, or of no event, makes the report false
    for label in (other.label, f"{scene.label}.EVENTINSTANCE.999"):
        wrong = dataclasses.replace(report, event=label)
        assert truth.evaluate(wrong).valid and not truth.evaluate(wrong).true


def test_event_forms_that_cannot_be_judged(cases) -> None:
    case = cases("tiny")
    facts = case.facts()
    truth = facts.truth
    scenes = make(case.scenes(), 50)
    for scene in scenes:
        truth.add_scene(scene)
    scene = next(s for s in scenes if any(e.transitive for e in scene_events(s)))
    event = next(e for e in scene_events(scene) if e.transitive)
    report = facts.event_fact(event)
    outsider = next(i for i in case.world.instances if i not in scene.participants)
    bad = {
        "never negated": dataclasses.replace(report, polarity=False),
        "no quantifier": dataclasses.replace(report, quantifier="nec_all"),
        "unknown scene": dataclasses.replace(report, scene="SCENE.999"),
        "unknown scene ": dataclasses.replace(report, scene=None),
        "takes no part": dataclasses.replace(report, subject=outsider),
        "takes no part ": dataclasses.replace(
            report, predicate=Predicate(VERB, event.type, outsider)
        ),
        "never related to itself": dataclasses.replace(
            report, predicate=Predicate(VERB, event.type, event.agent)
        ),
        "one-place event type, or a two-place": dataclasses.replace(
            report, predicate=Predicate(IS, "PROPERTY.1")
        ),
        "unknown two-place event type": dataclasses.replace(
            report, predicate=Predicate(VERB, "EVENTTYPE2.7", event.patient)
        ),
        "and only one, has a patient": dataclasses.replace(
            report, predicate=Predicate(VERB, event.type)
        ),
        "events are in the past tense": dataclasses.replace(report, tense="present"),
        "events are in the past tense ": dataclasses.replace(report, tense=None),
        "simple or progressive": dataclasses.replace(report, aspect=None),
        "simple or progressive ": dataclasses.replace(report, aspect="perfect"),
        "is not in the scene": dataclasses.replace(report, event="SCENE.999.EVENTINSTANCE.1"),
    }
    for reason, proposition in bad.items():
        evaluation = truth.evaluate(proposition)
        assert not evaluation.valid and reason.strip() in evaluation.reason, reason
    # a scene and an event belong to the event level only
    instance = Proposition(INSTANCE, event.agent, Predicate(IS, "PROPERTY.1"), scene=scene.label)
    assert not truth.evaluate(instance).valid


# ---------------------------------------------------------------------------------------------
# Tense and aspect
# ---------------------------------------------------------------------------------------------


def test_the_tense_of_events_is_the_corpus_s(cases) -> None:
    case = cases("tiny")
    settings = {"propositions": {"events": {"tense": "present"}}}
    generator = case.scenes(**settings)
    facts = case.facts(**settings)
    scene = next(s for s in make(generator, 50) if scene_events(s))
    facts.truth.add_scene(scene)
    event = scene_events(scene)[0]
    report = facts.event_fact(event)
    assert report.tense == "present" and report.to_json()["tense"] == "present"
    assert facts.truth.is_true(report)
    past = dataclasses.replace(report, tense="past")
    assert not facts.truth.evaluate(past).valid


def test_a_report_chooses_its_aspect(cases) -> None:
    case = cases("default")
    facts = case.facts()
    scenes = make(case.scenes(), 60)
    for scene in scenes:
        facts.truth.add_scene(scene)
    for scene in scenes[:20]:
        for event in scene_events(scene):
            simple = facts.event_fact(event, aspect=SIMPLE)
            ongoing = facts.event_fact(event, aspect=PROGRESSIVE)
            if simple is None:
                continue
            # both aspects are true of any event that occurred, and the aspect is part of the
            # report's logical form
            assert facts.truth.is_true(simple) and facts.truth.is_true(ongoing)
            assert simple != ongoing and simple.event == ongoing.event
    # a scene records no aspect
    for scene in scenes[:5]:
        for step in scene.to_json()["steps"]:
            assert not any("aspect" in e for e in step["events"])


# ---------------------------------------------------------------------------------------------
# Naming an event's event type
# ---------------------------------------------------------------------------------------------


def test_an_event_can_be_named_by_a_category_above_its_event_type(cases) -> None:
    case = cases("default")
    facts = case.facts()
    world = case.world
    scenes = make(case.scenes(), 200)  # two-place events are rare under the preconditions
    for scene in scenes:
        facts.truth.add_scene(scene)
    checked = 0
    for scene in scenes:
        for event in scene_events(scene):
            names = facts.event_names(event)
            if not event.transitive:
                assert names == (event.type,)
                assert facts.event_fact(event).predicate == Predicate(CAN, event.type)
                continue
            parent = world.event_types[event.type].parent
            assert names == (event.type, parent)
            general = facts.event_fact(event, parent)
            assert general.predicate.label == parent and general.event == event.label
            # an event type entails the base relation of its category
            assert general.grounding["able"] is True and general.grounding["legal"] is True
            # a category that is not above the event type does not name the event
            other = next(c for c in world.binary if world.event_types[c].category and c != parent)
            assert facts.event_fact(event, other) is None
            sibling = next(v for v in world.binary_leaves if v != event.type)
            if not happened_in(scene, sibling, event.agent, event.patient):
                assert facts.event_fact(event, sibling) is None
            checked += 1
    assert checked > 60


def test_the_event_level_is_drawn_by_weight(cases) -> None:
    case = cases("default")
    scenes = make(case.scenes(), 200)
    events = [e for s in scenes for e in scene_events(s)]
    transitive = [e for e in events if e.transitive]

    def category_share(weights) -> float:
        facts = knowing(case.facts(mention={"event_level_weights": weights}), scenes)
        rng = Streams(1).mentions
        named = [facts.draw_event(rng, event) for event in transitive]
        assert all(p is not None and facts.truth.is_true(p) for p in named)
        return float(
            np.mean([p.predicate.label != e.type for p, e in zip(named, transitive, strict=True)])
        )

    # heaviest at the leaf by default: weights 1 and 4 over the two levels of the tree
    assert case.config().mention.event_level_weights == (1.0, 4.0)
    assert abs(category_share({"schedule": "linear", "start": 1, "end": 4}) - 0.2) < 0.03
    assert category_share({"schedule": "list", "values": [0, 1]}) == 0.0
    assert category_share({"schedule": "list", "values": [1, 0]}) == 1.0
    assert abs(category_share(1) - 0.5) < 0.04
    # a one-place event has one name
    facts = knowing(case.facts(), scenes)
    rng = Streams(1).mentions
    for event in [e for e in events if not e.transitive][:50]:
        assert facts.draw_event(rng, event).predicate == Predicate(CAN, event.type)


def test_an_event_without_a_word_at_any_usable_level_is_not_reported(cases) -> None:
    case = cases("tiny")
    scenes = make(case.scenes(), 200)
    events = [e for s in scenes for e in scene_events(s)]
    facts = knowing(case.facts(), scenes)
    rng = Streams(1).mentions
    # EVENTTYPE2.1 holds for every pair, so it has no word: an event of EVENTTYPE2.1.1 has one
    # name
    for event in events:
        if event.type.startswith("EVENTTYPE2.1."):
            assert facts.event_names(event) == (event.type,)
            assert facts.draw_event(rng, event).predicate.label == event.type
        elif event.type.startswith("EVENTTYPE2.2."):
            assert facts.event_names(event) == (event.type, "EVENTTYPE2.2")
    # with only the top level usable, an event of EVENTTYPE2.1.1 cannot be named, and an event
    # of EVENTTYPE2.2.1 is always named by EVENTTYPE2.2
    levels = {"event_level_weights": {"schedule": "list", "values": [1, 0]}}
    top_only = knowing(case.facts(mention=levels), scenes)
    for event in events:
        named = top_only.draw_event(rng, event)
        if event.type.startswith("EVENTTYPE2.1."):
            assert named is None
        elif event.type.startswith("EVENTTYPE2.2."):
            assert named.predicate.label == "EVENTTYPE2.2"
    # an event type without a word names nothing
    no_words = knowing(
        case.facts(lexicon={"named_proportion": {"event": 0.0, "event_unary": 0.0}}), scenes
    )
    for event in events:
        named = no_words.draw_event(rng, event)
        if event.type.startswith("EVENTTYPE2.2."):
            assert named.predicate.label == "EVENTTYPE2.2"
        else:
            assert named is None and no_words.event_fact(event) is None
