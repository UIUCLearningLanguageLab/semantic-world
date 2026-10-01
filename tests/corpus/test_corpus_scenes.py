"""Stage 3 acceptance tests: scenes and events.

Every event is possible: its CAN feature or its relation holds, by the taxonomy's output files.
No event repeats within a time step. With the thematic weight above 0, participants are more
thematically related than with the weight at 0, on average.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import polars as pl
import pytest
from corpus_support import PLAIN_TAXONOMY, corpus_config

from semantic_world.corpus import ConfigError, Streams, load_taxonomy
from semantic_world.corpus.propositions import (
    CAN,
    EVENT,
    INSTANCE,
    IS,
    VERB,
    Predicate,
    Proposition,
)
from semantic_world.corpus.scenes import Event, Scene, SceneGenerator

WORLDS = ("tiny", "default", "deep", "still")


def make(generator: SceneGenerator, count: int, seed: int = 1, first: int = 1) -> list[Scene]:
    """Scenes ``SN.<first>`` onward, each seeded at an instance drawn from a fixed order."""
    streams = Streams(seed)
    labels = generator.labels
    return [
        generator.scene(streams, number, labels[(number * 7) % len(labels)])
        for number in range(first, first + count)
    ]


def scene_settings(**scene) -> dict:
    return {"scene": scene}


def knowing(facts, scenes: list[Scene]):
    """Facts of other settings have truth tests of their own: make the scenes known to them."""
    for scene in scenes:
        facts.truth.add_scene(scene)
    return facts


# ---------------------------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_event_is_possible_by_the_output_files(cases, name) -> None:
    case = cases(name)
    oracle = case.oracle()
    scenes = make(case.scenes(), 300)
    events = [e for scene in scenes for e in scene.events]
    assert len(events) > 1000
    for event in events:
        agent = oracle.row[event.agent]
        if event.patient is None:
            # an intransitive event: the agent has the CAN feature
            assert event.verb.startswith("CAN.") and oracle.column[event.verb][agent] == 1, event
        else:
            # a transitive event: the verb's relation holds for the agent and the patient
            assert event.agent != event.patient
            assert oracle.matrix(event.verb)[agent, oracle.row[event.patient]], event
    kinds = Counter(e.transitive for e in events)
    assert kinds[True] > 100 and kinds[False] > 100


@pytest.mark.parametrize("name", WORLDS)
def test_no_event_repeats_within_a_time_step(cases, name) -> None:
    case = cases(name)
    # many events at every step, so that a repeat would show
    scenes = make(case.scenes(**scene_settings(events_per_step=6.0)), 200)
    steps = repeats_across_steps = 0
    for scene in scenes:
        for step in range(1, scene.steps + 1):
            keys = [e.key for e in scene.at(step)]
            assert len(set(keys)) == len(keys), (scene.label, step)
            steps += 1
        keys = [e.key for e in scene.events]
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
    assert abs(plain[0] - case.scenes().thematic.mean()) < 0.1


# ---------------------------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_participants(cases, name) -> None:
    case = cases(name)
    scenes = make(case.scenes(), 300)
    sizes = Counter()
    for number, scene in enumerate(scenes, start=1):
        assert scene.label == f"SN.{number}"
        assert scene.participants[0] == scene.seed
        assert len(set(scene.participants)) == len(scene.participants)
        assert set(scene.participants) <= set(case.result.instances.labels)
        sizes[len(scene.participants) - 1] += 1
    # scene.size other instances, drawn from the range
    assert set(sizes) == {2, 3, 4, 5, 6}
    assert min(sizes.values()) > 300 / 5 * 0.5


def test_participant_weights_are_the_weighted_sum(cases) -> None:
    case = cases("default")
    weights = {"thematic": 2.0, "taxonomic": 0.7, "constant": 0.05}
    generator = case.scenes(**scene_settings(participant_weights=weights))
    instances = case.result.instances
    thematic = pl.read_csv(case.folder / "thematic.csv")
    # thematic.csv gives both numbers for every pair of leaves with a thematic score above 0
    score = {}
    for row in thematic.iter_rows(named=True):
        score[row["leaf_a"], row["leaf_b"]] = score[row["leaf_b"], row["leaf_a"]] = (
            row["thematic"],
            row["similarity"],
        )
    assert len(score) > 100
    for seed in instances.labels[::37]:
        values = generator.participant_weights(seed)
        seed_leaf = instances.leaf_labels[instances.labels.index(seed)]
        assert values[instances.labels.index(seed)] == 0  # the seed is not drawn again
        for row in range(0, len(instances), 5):
            if instances.labels[row] == seed:
                continue
            pair = (instances.leaf_labels[row], seed_leaf)
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
            assert drawn[generator.index[participant]] > 0
        related = int(np.count_nonzero(drawn))
        assert len(scene.participants) - 1 == related  # every related instance, and no other
        smaller += related < 283
    assert smaller > 0  # a scene can be smaller than asked


def test_a_scene_never_has_more_participants_than_the_world(cases) -> None:
    case = cases("tiny")
    generator = case.scenes(**scene_settings(size=[30, 30]))
    for scene in make(generator, 20):
        assert sorted(scene.participants) == sorted(case.result.instances.labels)
    alone = case.scenes(**scene_settings(size=0))
    for scene in make(alone, 20):
        assert scene.participants == (scene.seed,)
        assert all(not e.transitive and e.agent == scene.seed for e in scene.events)


# ---------------------------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_timeline(cases, name) -> None:
    case = cases(name)
    scenes = make(case.scenes(), 400)
    steps = Counter(scene.steps for scene in scenes)
    assert set(steps) == {3, 4, 5, 6, 7, 8}
    per_step = []
    for scene in scenes:
        # events are labeled in time order, and a step can be empty
        assert [e.label for e in scene.events] == [
            f"{scene.label}.{k}" for k in range(1, len(scene.events) + 1)
        ]
        assert [e.step for e in scene.events] == sorted(e.step for e in scene.events)
        assert all(1 <= e.step <= scene.steps and e.scene == scene.label for e in scene.events)
        assert all(
            e.involves(e.agent)
            and set(filter(None, (e.agent, e.patient))) <= set(scene.participants)
            for e in scene.events
        )
        per_step += [len(scene.at(step)) for step in range(1, scene.steps + 1)]
    # a Poisson number of events at each step, with the configured mean
    assert abs(np.mean(per_step) - 1.5) < 0.1
    assert abs(np.var(per_step) - 1.5) < 0.3
    assert 0 in per_step and max(per_step) >= 5


def test_the_pool_of_possible_events(cases) -> None:
    case = cases("tiny")
    generator = case.scenes()
    result = case.result
    participants = ("I1.1.1", "I1.2.2", "I2.1.3", "I2.2.1")
    intransitive, transitive = generator.possible_events(participants)
    features = result.features
    rows = [result.instances.labels.index(p) for p in participants]
    expected = [
        (f.label, p, None)
        for p, row in zip(participants, rows, strict=True)
        for f in features.of_type("can")
        if result.instances.values[row, f.position]
    ]
    assert intransitive == expected and len(expected) > 0
    # every ordered pair of distinct participants, and every verb whose relation holds: the
    # verbs are the leaves of the verb tree, with a word or without one
    expected_pairs = []
    for verb in ("V1.1", "V1.2", "V2.1", "V2.2"):
        for a, row_a in zip(participants, rows, strict=True):
            for p, row_p in zip(participants, rows, strict=True):
                if a != p and result.relations.holds(verb, [row_a], [row_p])[0]:
                    expected_pairs.append((verb, a, p))
    assert transitive == expected_pairs and len(expected_pairs) > 0
    assert not any(verb in ("V1", "V2") for verb, _, _ in transitive)


def test_the_transitive_share(cases) -> None:
    case = cases("default")

    def share(value: float) -> float:
        scenes = make(case.scenes(**scene_settings(transitive_share=value)), 300)
        return float(np.mean([e.transitive for s in scenes for e in s.events]))

    assert share(0.0) == 0.0 and share(1.0) == 1.0
    assert abs(share(0.5) - 0.5) < 0.04
    assert abs(share(0.8) - 0.8) < 0.04
    # a scene with one participant has no transitive event, whatever the share
    alone = case.scenes(**scene_settings(size=0, transitive_share=1.0))
    assert all(not scene.events for scene in make(alone, 30))
    mostly = case.scenes(**scene_settings(size=0, transitive_share=0.9))
    assert any(scene.events for scene in make(mostly, 30))


def test_events_are_drawn_verb_first(cases) -> None:
    # With every instance of the tiny world in the scene, the pool is the same in every scene,
    # and the verbs differ widely in how many pairs they hold for.
    case = cases("tiny")
    generator = case.scenes(**scene_settings(size=[11, 11], events_per_step=1.0, steps=[4, 4]))
    intransitive, transitive = generator.possible_events(case.result.instances.labels)
    pairs = Counter(verb for verb, _, _ in transitive)
    agents = Counter(verb for verb, _, _ in intransitive)
    assert max(pairs.values()) > 3 * min(pairs.values())
    events = [e for s in make(generator, 1500) for e in s.events]
    drawn = Counter(e.verb for e in events)
    # each verb with a possible event is equally likely, however many pairs it holds for
    for counts in (pairs, agents):
        share = np.array([drawn[verb] for verb in counts], dtype=float)
        share /= share.sum()
        assert np.abs(share - 1 / len(counts)).max() < 0.03, dict(zip(counts, share, strict=True))
    # and within a verb, each possible pair is equally likely
    verb = max(pairs, key=pairs.get)
    by_pair = Counter(e.key for e in events if e.verb == verb)
    assert set(by_pair) == {key for key in transitive if key[0] == verb}
    expected = drawn[verb] / pairs[verb]
    assert all(abs(count - expected) < 5 * np.sqrt(expected) for count in by_pair.values())
    # the weights are weights of verbs, not of events
    weighted = case.scenes(
        **scene_settings(size=[11, 11], events_per_step=1.0, verb_weights={"V1.1": 3})
    )
    counts = Counter(e.verb for s in make(weighted, 1500) for e in s.events if e.transitive)
    assert abs(counts["V1.1"] / counts["V2.1"] - 3) < 0.5


def test_verb_weights(cases) -> None:
    case = cases("tiny")
    uniform = Counter(e.verb for s in make(case.scenes(), 300) for e in s.events)
    assert set(uniform) == {"CAN.1", "CAN.3", "CAN.4", "V1.1", "V1.2", "V2.1", "V2.2"}
    weighted = case.scenes(**scene_settings(verb_weights={"CAN.1": 0, "V1.2": 20, "V2.1": 0}))
    counts = Counter(e.verb for s in make(weighted, 300) for e in s.events)
    # a weight of 0 keeps a verb out, and a large weight brings it in more often
    assert counts["CAN.1"] == 0 and counts["V2.1"] == 0
    assert counts["V1.2"] > 2 * uniform["V1.2"]
    assert counts["CAN.3"] > 0  # a label that is left out keeps the weight 1
    # the labels are checked against the world: a verb category names no event of its own
    for label in ("V2", "CAN.9", "IS.1", "V9.1"):
        with pytest.raises(ConfigError) as info:
            case.scenes(**scene_settings(verb_weights={label: 1}))
        assert info.value.field == f"scene.verb_weights.{label}"
        assert "not a CAN feature or a verb" in info.value.message


def test_events_per_step(cases) -> None:
    case = cases("default")
    assert all(not s.events for s in make(case.scenes(**scene_settings(events_per_step=0)), 30))
    busy = make(case.scenes(**scene_settings(events_per_step=4.0, steps=[5, 5])), 200)
    assert all(scene.steps == 5 for scene in busy)
    assert abs(np.mean([len(s.events) for s in busy]) - 20) < 1.0


def test_a_world_without_verbs_has_intransitive_events_only() -> None:
    config = corpus_config(PLAIN_TAXONOMY)
    result = load_taxonomy(config)
    generator = SceneGenerator(config, result)
    scenes = make(generator, 100)
    events = [e for s in scenes for e in s.events]
    assert events and not any(e.transitive for e in events)
    assert generator.verbs == () and not generator.thematic.any()
    # with verbs off, the thematic weight has no effect
    weights = {"thematic": 9.0, "taxonomic": 0.5, "constant": 0.1}
    other = SceneGenerator(
        corpus_config(PLAIN_TAXONOMY, scene={"participant_weights": weights}), result
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
    assert other.label == "SN.17" and other != first[16]
    with pytest.raises(KeyError, match="unknown instance"):
        generator.scene(streams, 1, "I9.9.9")


def test_only_the_scene_settings_change_a_scene(cases) -> None:
    case = cases("default")
    base = make(case.scenes(), 40)
    # the lexicon, the grammar, and the other sections leave every scene as it is: scenes are a
    # fact about the world, and use every verb, with a word or without one
    half = dict.fromkeys(("category", "can", "verb", "verb_category"), 0.5)
    others = {
        "lexicon": {"named_proportion": half, "synonym_rate": 0.4},
        "grammar": {"word_order": {"clause": "SOV"}, "morphology": {"tense": {"enabled": True}}},
        "mention": {"pronoun_rate": 0.9, "verb_level_weights": 1},
        "propositions": {"negation_rate": {"class": 0.5}},
        "test_sets": {"size": 3},
    }
    assert make(case.scenes(**others), 40) == base
    assert make(case.scenes(**scene_settings(events_per_step=1.6)), 40) != base


# ---------------------------------------------------------------------------------------------
# The scene record
# ---------------------------------------------------------------------------------------------


def test_scene_json_round_trips(cases) -> None:
    case = cases("tiny")
    for scene in make(case.scenes(), 50):
        form = scene.to_json()
        assert list(form) == ["label", "seed", "participants", "steps"]
        # the events, by time step
        assert len(form["steps"]) == scene.steps
        assert sum(len(step) for step in form["steps"]) == len(scene.events)
        for event in (e for step in form["steps"] for e in step):
            assert list(event) == ["label", "verb", "agent", "patient"]
        assert Scene.from_json(json.loads(json.dumps(form))) == scene


def test_scene_lookups() -> None:
    events = (
        Event("SN.2.1", "SN.2", 1, "CAN.1", "I1.1.1"),
        Event("SN.2.2", "SN.2", 1, "V1.1", "I1.1.1", "I2.1.1"),
        Event("SN.2.3", "SN.2", 3, "V1.1", "I2.1.1", "I1.1.2"),
    )
    scene = Scene("SN.2", "I1.1.1", ("I1.1.1", "I2.1.1", "I1.1.2"), 3, events)
    assert scene.at(1) == events[:2] and scene.at(2) == () and scene.at(3) == events[2:]
    assert scene.involving("I2.1.1") == events[1:] and scene.involving("I1.1.2") == events[2:]
    assert scene.happened("V1.1", "I1.1.1", "I2.1.1") and scene.happened("CAN.1", "I1.1.1")
    assert not scene.happened("V1.1", "I2.1.1", "I1.1.1")
    assert [e.transitive for e in events] == [False, True, True]
    assert events[1].key == ("V1.1", "I1.1.1", "I2.1.1")


# ---------------------------------------------------------------------------------------------
# Event-level propositions
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_event_is_reported_by_a_true_proposition(cases, name) -> None:
    case = cases(name)
    generator = case.scenes()
    truth = generator.truth
    for scene in make(generator, 100):
        assert truth.scenes[scene.label] is scene
        for event in scene.events:
            proposition = event.proposition()
            assert proposition.level == EVENT and proposition.polarity
            assert (proposition.scene, proposition.event) == (scene.label, event.label)
            assert proposition.subject == event.agent
            assert proposition.predicate == Predicate(
                VERB if event.transitive else CAN, event.verb, event.patient
            )
            evaluation = truth.evaluate(proposition)
            assert evaluation.valid and evaluation.true
            assert evaluation.grounding == {
                "scene": scene.label,
                "step": event.step,
                "possible": True,
                "test": "event",
            }


def test_event_logical_forms(cases) -> None:
    case = cases("tiny")
    generator = case.scenes()
    scene = next(s for s in make(generator, 50) if any(e.transitive for e in s.events))
    event = next(e for e in scene.events if e.transitive)
    proposition = generator.truth.grounded(event.proposition())
    assert proposition.to_json() == {
        "id": None,
        "level": "event",
        "scene": scene.label,
        "event": event.label,
        "polarity": True,
        "subject": {"instance": event.agent},
        "predicate": {"kind": "verb", "verb": event.verb, "patient": {"instance": event.patient}},
        "grounding": {"scene": scene.label, "step": event.step, "possible": True, "test": "event"},
    }
    restored = Proposition.from_json(json.loads(json.dumps(proposition.to_json())))
    assert restored == proposition and restored.grounding == proposition.grounding
    # the event is part of what the proposition says: two reports of two events differ
    assert proposition != dataclasses.replace(proposition, event=f"{scene.label}.999")
    assert proposition.concepts() == (event.verb,)
    intransitive = next(e for s in make(generator, 50) for e in s.events if not e.transitive)
    assert intransitive.proposition().to_json()["predicate"] == {
        "kind": "can",
        "feature": intransitive.verb,
    }


def test_an_event_is_true_only_if_it_happened_in_its_scene(cases) -> None:
    case = cases("default")
    generator = case.scenes()
    truth = generator.truth
    scenes = make(generator, 60)
    not_happened = {True: 0, False: 0}
    for scene in scenes:
        happened = {e.key for e in scene.events}
        intransitive, transitive = generator.possible_events(scene.participants)
        possible = set(intransitive) | set(transitive)
        for verb in generator.verbs[:3]:
            for agent in scene.participants:
                for patient in scene.participants:
                    if agent == patient:
                        continue
                    claim = Proposition(
                        EVENT, agent, Predicate(VERB, verb, patient), scene=scene.label
                    )
                    evaluation = truth.evaluate(claim)
                    key = (verb, agent, patient)
                    assert evaluation.valid and evaluation.true == (key in happened)
                    # the grounding says whether the world allows the event: what tells an
                    # impossible false item from one that merely did not happen
                    assert evaluation.grounding["possible"] == (key in possible)
                    assert ("step" in evaluation.grounding) == evaluation.true
                    if not evaluation.true:
                        not_happened[key in possible] += 1
        for feature in generator.can_features[:4]:
            for agent in scene.participants:
                claim = Proposition(EVENT, agent, Predicate(CAN, feature), scene=scene.label)
                evaluation = truth.evaluate(claim)
                assert evaluation.true == ((feature, agent, None) in happened)
                assert evaluation.grounding["possible"] == ((feature, agent, None) in possible)
    assert not_happened[True] > 50 and not_happened[False] > 50
    # what happened in one scene did not happen in another
    moved = 0
    for scene in scenes:
        for event in scene.events[:3]:
            for other in scenes:
                if (
                    other is not scene
                    and event.agent in other.participants
                    and event.patient in (None, *other.participants)
                    and not other.happened(*event.key)
                ):
                    claim = dataclasses.replace(event.proposition(), scene=other.label, event=None)
                    evaluation = truth.evaluate(claim)
                    assert evaluation.valid and not evaluation.true
                    assert evaluation.grounding["possible"] is True
                    moved += 1
    assert moved > 0


def test_a_report_names_one_event(cases) -> None:
    case = cases("default")
    generator = case.scenes(**scene_settings(events_per_step=5.0))
    truth = generator.truth
    scene = next(s for s in make(generator, 100) if len({e.key for e in s.events}) < len(s.events))
    keys = Counter(e.key for e in scene.events)
    repeated = next(key for key, count in keys.items() if count > 1)
    again = [e for e in scene.events if e.key == repeated]
    other = next(e for e in scene.events if e.key != repeated)
    report = again[1].proposition()
    assert truth.evaluate(report).grounding["step"] == again[1].step
    # without an event label, the first such event grounds the proposition
    unlabeled = dataclasses.replace(report, event=None)
    assert truth.evaluate(unlabeled).grounding["step"] == again[0].step
    # a label of another event, or of no event, makes the report false
    for label in (other.label, f"{scene.label}.999"):
        wrong = dataclasses.replace(report, event=label)
        assert truth.evaluate(wrong).valid and not truth.evaluate(wrong).true


def test_event_forms_that_cannot_be_judged(cases) -> None:
    case = cases("tiny")
    generator = case.scenes()
    truth = generator.truth
    scene = next(s for s in make(generator, 50) if any(e.transitive for e in s.events))
    event = next(e for e in scene.events if e.transitive)
    report = event.proposition()
    outsider = next(i for i in case.result.instances.labels if i not in scene.participants)
    bad = {
        "never negated": dataclasses.replace(report, polarity=False),
        "no quantifier": dataclasses.replace(report, quantifier="all"),
        "unknown scene": dataclasses.replace(report, scene="SN.999"),
        "unknown scene ": dataclasses.replace(report, scene=None),
        "takes no part": dataclasses.replace(report, subject=outsider),
        "takes no part ": dataclasses.replace(
            report, predicate=Predicate(VERB, event.verb, outsider)
        ),
        "never related to itself": dataclasses.replace(
            report, predicate=Predicate(VERB, event.verb, event.agent)
        ),
        "CAN feature, or a verb": dataclasses.replace(report, predicate=Predicate(IS, "IS.1")),
        "unknown verb": dataclasses.replace(report, predicate=Predicate(VERB, "V7", event.patient)),
        "only a verb, has a patient": dataclasses.replace(
            report, predicate=Predicate(VERB, event.verb)
        ),
    }
    for reason, proposition in bad.items():
        evaluation = truth.evaluate(proposition)
        assert not evaluation.valid and reason.strip() in evaluation.reason, reason
    # a scene and an event belong to the event level only
    instance = Proposition(INSTANCE, event.agent, Predicate(IS, "IS.1"), scene=scene.label)
    assert not truth.evaluate(instance).valid


# ---------------------------------------------------------------------------------------------
# Naming an event's verb
# ---------------------------------------------------------------------------------------------


def test_an_event_can_be_named_by_a_verb_category_above_its_verb(cases) -> None:
    case = cases("default")
    facts = case.facts()
    generator = case.scenes()
    tree = case.result.verbs.tree
    checked = 0
    for scene in make(generator, 60):
        for event in scene.events:
            names = facts.event_names(event)
            if not event.transitive:
                assert names == (event.verb,)
                assert facts.event_fact(event).predicate == Predicate(CAN, event.verb)
                continue
            parent = tree[event.verb].parent.label
            assert names == (event.verb, parent)
            general = facts.event_fact(event, parent)
            assert general.predicate.label == parent and general.event == event.label
            assert general.grounding["possible"] is True  # the verb entails the base relation
            # a verb category that is not above the verb does not name the event
            other = next(c.label for c in tree.categories if not c.is_leaf and c.label != parent)
            assert facts.event_fact(event, other) is None
            sibling = next(v.label for v in tree.leaves if v.label != event.verb)
            if not scene.happened(sibling, event.agent, event.patient):
                assert facts.event_fact(event, sibling) is None
            checked += 1
    assert checked > 100


def test_the_verb_level_is_drawn_by_weight(cases) -> None:
    case = cases("default")
    scenes = make(case.scenes(), 200)
    events = [e for s in scenes for e in s.events]
    transitive = [e for e in events if e.transitive]

    def category_share(weights) -> float:
        facts = knowing(case.facts(mention={"verb_level_weights": weights}), scenes)
        rng = Streams(1).mentions
        named = [facts.draw_event(rng, event) for event in transitive]
        assert all(p is not None and case.facts().truth.is_true(p) for p in named)
        return float(
            np.mean([p.predicate.label != e.verb for p, e in zip(named, transitive, strict=True)])
        )

    # heaviest at the leaf by default: weights 1 and 4 over the two levels of the verb tree
    assert case.config().mention.verb_level_weights == (1.0, 4.0)
    assert abs(category_share({"schedule": "linear", "start": 1, "end": 4}) - 0.2) < 0.03
    assert category_share({"schedule": "list", "values": [0, 1]}) == 0.0
    assert category_share({"schedule": "list", "values": [1, 0]}) == 1.0
    assert abs(category_share(1) - 0.5) < 0.04
    # an intransitive event has one name
    facts = case.facts()
    rng = Streams(1).mentions
    for event in [e for e in events if not e.transitive][:50]:
        assert facts.draw_event(rng, event).predicate == Predicate(CAN, event.verb)


def test_an_event_without_a_word_at_any_usable_level_is_not_reported(cases) -> None:
    case = cases("tiny")
    scenes = make(case.scenes(), 200)
    events = [e for s in scenes for e in s.events]
    facts = case.facts()
    rng = Streams(1).mentions
    # V1 holds for every pair, so it has no word: an event of V1.1 has one name
    for event in events:
        if event.verb.startswith("V1."):
            assert facts.event_names(event) == (event.verb,)
            assert facts.draw_event(rng, event).predicate.label == event.verb
        elif event.verb.startswith("V2."):
            assert facts.event_names(event) == (event.verb, "V2")
    # with only the top level usable, an event of V1.1 cannot be named, and an event of V2.1
    # is always named by V2
    levels = {"verb_level_weights": {"schedule": "list", "values": [1, 0]}}
    top_only = knowing(case.facts(mention=levels), scenes)
    for event in events:
        named = top_only.draw_event(rng, event)
        if event.verb.startswith("V1."):
            assert named is None
        elif event.verb.startswith("V2."):
            assert named.predicate.label == "V2"
    # a verb or a CAN feature without a word names nothing
    no_words = knowing(case.facts(lexicon={"named_proportion": {"verb": 0.0, "can": 0.0}}), scenes)
    for event in events:
        named = no_words.draw_event(rng, event)
        if event.verb.startswith("V2."):
            assert named.predicate.label == "V2"
        else:
            assert named is None and no_words.event_fact(event) is None
