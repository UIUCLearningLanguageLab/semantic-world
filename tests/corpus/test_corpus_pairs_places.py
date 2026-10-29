"""Stage c4a of ``docs/specs/WORLD_PHASE_C.md``: the corpus on worlds with pair fluents and places.

Since stage c4a ``data/world/tiny.yaml`` and ``data/world/default.yaml`` carry the phase (c)
settings (Jon's ruling WM.63): pair fluents between entities, a location fluent, moves and acts
at a place, ``SAME`` on the location fluent, and derived fluents over ``SOME`` and ``SOMEOF``.
The corpus runs on them and says nothing about them yet (the lexicon and the sentences are
stages c4b, c4c, and c5a): no word is made for a pair fluent, a place, a place feature, a place
event type, ``SAME``, ``SOME``, or ``SOMEOF``; no sentence or test item names one; an event of a
place event type is never narrated; a causal statement never states a pair literal; and a
conditional effect whose condition holds a pair literal is stated in neither form (WM.59). Every
sentence the corpus does make must stay true by the independent oracle, whose replay now carries
the pairs and the places, and whose pairs and places must agree with ``history.replay`` at every
time point of every scene. The fast tier runs the tiny corpus, the slow tier the default one.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from corpus_support import REPO, TINY_WORLD, corpus_config

from semantic_world.corpus import Facts, load_world
from semantic_world.corpus.histories import scene_events
from semantic_world.corpus.logical import with_cut
from semantic_world.corpus.planner import EVENT_SECTION
from semantic_world.corpus.propositions import (
    ABLE_NOW,
    CHANGE,
    CLASS,
    EFFECT,
    EVENT,
    STATE,
    EventTerm,
    condition_of,
    effect_matches,
)
from semantic_world.world.history import replay

WORLDS = ("tiny", pytest.param("default", marks=pytest.mark.slow))
PAIR_LABELS = ("PAIRFL", "PLACE", "SAME.", "SOME.", "SOMEOF.", "EVERYPLACE", "SOMEPLACE")
"""The labels of phase (c) that no sentence, test item, or word of the corpus may carry in
stage c4a: the pair fluents, the places and their categories, features, and event types, the
pair operators, and the bare symbols for places."""
MANY = {
    "documents": {"initial_state_rate": 0.6, "result_rate": 0.9, "blocked_rate": 0.5},
    "propositions": {"negation_rate": {"able_now": 0.6}},
}
"""Many state, result, and blocked sentences, so that the tests see every kind."""


def scenes_json(corpus) -> dict[str, dict]:
    return {scene.label: json.loads(json.dumps(scene.to_json())) for scene in corpus.planner.scenes}


# ---------------------------------------------------------------------------------------------
# Acceptance: the oracle's replay agrees with history.replay at every time point
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_the_oracles_pairs_and_places_agree_with_history_replay(cases, runs, name) -> None:
    """At every time point of every scene of the corpus, the pairs the oracle replays (both
    directions of a symmetric pair, a move's two changes, the total functional replacement) are
    the pairs of the runtime's replayed state, and every entity's place is the runtime's."""
    case = cases(name)
    corpus = runs(name)
    oracle = case.oracle()
    definition = case.world.definition
    labels = definition.entity_labels
    assert definition.has_location_fluents and len(definition.pair_fluents) > 1
    moves = pair_changes = 0
    for scene in corpus.planner.scenes:
        states = replay(definition, scene)
        oracle_pairs = oracle.pairs_at(json.loads(json.dumps(scene.to_json())))
        assert len(oracle_pairs) == len(states) == len(scene.steps) + 1
        for state, pairs in zip(states, oracle_pairs, strict=True):
            for fluent, matrix in zip(definition.pair_fluents, state.pairs, strict=True):
                targets = definition.place_labels if fluent.total else labels
                expected = {
                    (labels[int(i)], targets[int(j)])
                    for i, j in zip(*np.nonzero(matrix), strict=True)
                }
                assert pairs[fluent.label] == expected, (scene.label, fluent.label)
            places = state.places_record(definition)
            for entity, at in places.items():
                for fluent, place in at.items():
                    assert (
                        oracle.place_at(scene.to_json(), states.index(state) + 1, entity, fluent)
                        == place
                    )
        for step in scene.steps:
            for event in step.events:
                if definition.event_type(event.type).move:
                    moves += 1
                pair_changes += sum(1 for c in event.changes if c.pair)
    assert moves > 0 and pair_changes > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: every sentence true, every false item false, by the oracle
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_every_sentence_is_true_and_every_false_item_false_by_the_oracle(cases, runs, name) -> None:
    """Every sentence of the corpus (the timed ones judged with the pairs and the places, which
    now decide what is legal) is true by the oracle, and every false test item is false; the
    corpus holds state, change, and ``able_now`` sentences, and blocked sentences whose binding
    the pairs or the places block."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    scenes = scenes_json(corpus)
    cuts = corpus.planner.truth.cuts
    levels: Counter = Counter()
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            levels[proposition.level] += 1
            if proposition.level == EVENT:
                assert oracle.event(sentence.logical_form, scenes[proposition.scene])
                continue
            scene = None if proposition.scene is None else scenes[proposition.scene]
            form = with_cut(proposition.to_json(), cuts)
            assert oracle.truth(form, scene) is True, sentence.label
    assert levels[STATE] > 0 and levels[CHANGE] > 0 and levels[ABLE_NOW] > 0
    items = 0
    for test_set in corpus.test_sets:
        for true, false in test_set.pairs:
            for item, expected in ((true, True), (false, False)):
                form = item.input["logical_form"]
                scene = scenes.get(form.get("scene")) if form.get("scene") else None
                if form["level"] == EVENT:
                    judged = oracle.event(form, scene)
                else:
                    judged = oracle.truth(form, scene)
                assert judged is expected, (test_set.name, item.input["propositional"])
                items += 1
    assert items > 100


@pytest.mark.parametrize("name", WORLDS)
def test_blocked_sentences_are_judged_with_the_pairs_and_the_places(cases, runs, name) -> None:
    """A negative ``ABLE_NOW`` sentence is blocked by a precondition that the replayed state
    fails, and since stage c4a that precondition may be a pair literal: ``SAME`` on the location
    fluent, a pair between the two participants, ``SOME`` or ``SOMEOF``, or a compiled literal.
    The oracle, replaying the pairs, agrees with the corpus on every such sentence."""
    case = cases(name)
    corpus = runs(name, **MANY)
    oracle = case.oracle(**MANY)
    scenes = scenes_json(corpus)
    definition = case.world.definition
    blocked = pair_blocked = 0
    for document in corpus.documents:
        for sentence in document.sentences:
            proposition = sentence.proposition
            if proposition.level != ABLE_NOW or proposition.polarity:
                continue
            blocked += 1
            scene = scenes[proposition.scene]
            assert oracle.able_now(sentence.logical_form, scene)
            # which literals of the event type's precondition read a pair
            for leaf in case.world.event_types_below(proposition.predicate.label):
                if any(lit.pair for lit in definition.event_type(leaf).precondition):
                    pair_blocked += 1
                    break
    assert blocked > 0
    # the tiny world's two-place event types carry SAME on the location fluent (the quota), so
    # some blocked sentence is about an event type whose precondition reads a pair
    assert pair_blocked > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: nothing names a pair fluent, a place, or a place event type
# ---------------------------------------------------------------------------------------------


def _carries(value, labels: tuple[str, ...] = PAIR_LABELS) -> bool:
    text = json.dumps(value)
    return any(label in text for label in labels)


@pytest.mark.parametrize("name", WORLDS)
def test_no_sentence_item_or_word_names_a_pair_fluent_or_a_place(cases, runs, name) -> None:
    """The lexicon has no concept of phase (c), no sentence's tokens, logical form, or
    rendering name one, no test item does, and the word-form request lists the same lexemes as
    the lexicon's content lexemes, none for a pair fluent or a place."""
    from semantic_world.corpus import wordform_request

    corpus = runs(name)
    lexicon = corpus.planner.lexicon
    assert not any(_carries(c.label) for c in lexicon.concepts)
    assert not any(_carries(x.concept) for x in lexicon.lexemes)
    world = case = cases(name).world
    assert world.place_event_types and world.pair_fluents
    for label in world.place_event_types:
        assert world.event_names(label) == ()
        assert not lexicon.is_named(label)
    for document in corpus.documents:
        for sentence in document.sentences:
            assert not _carries(sentence.to_json()), sentence.label
    for test_set in corpus.test_sets:
        for item in test_set.items:
            assert not _carries(item.input), (test_set.name, item.input["propositional"])
    request = wordform_request(corpus)
    assert [x["concept"] for x in request["lexemes"]] == [
        x.concept for x in lexicon.content_lexemes
    ]
    assert not _carries(request)
    del case


@pytest.mark.parametrize("name", WORLDS)
def test_narratives_skip_the_events_of_place_event_types(cases, runs, name) -> None:
    """A move or an act at a place is never narrated, in a main clause or in a relative clause,
    while the scenes hold such events; the narrative's other sentences stay true around a
    skipped event (the test above), because every sentence is judged on the replayed history."""
    corpus = runs(name)
    world = cases(name).world
    place_events = narrated_events = skipped = 0
    for document in corpus.documents:
        if not document.scenes:
            continue
        reported = {
            label
            for sentence in document.sentences
            for label in sentence.sentence.event_labels
            if label is not None
        }
        for scene in document.scenes:
            for event in scene_events(scene):
                if event.place:
                    place_events += 1
                    assert world.is_place_event_type(event.type)
                    assert event.label not in reported
                    skipped += 1
        for sentence in document.sentences:
            if sentence.section == EVENT_SECTION:
                narrated_events += 1
                assert not world.is_place_event_type(sentence.proposition.predicate.label)
    assert place_events > 0 and skipped == place_events and narrated_events > 0


# ---------------------------------------------------------------------------------------------
# Acceptance: no causal statement states a pair literal or a pair condition
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
def test_no_causal_statement_states_a_pair_literal(cases, runs, name) -> None:
    """Every causal statement is about a base or a numeric fluent, none about a pair fluent;
    no event term's condition holds a pair literal; and the conditional effects with a pair
    condition are counted among the unstated."""
    corpus = runs(name)
    facts = corpus.planner.facts
    statements = facts.causal_statements()
    assert statements
    for statement in statements:
        assert isinstance(statement.subject, EventTerm)
        assert not _carries(statement.predicate.label)
        assert not any(_carries(c.label) for c in statement.subject.condition)
    report = facts.causal_report
    assert {"pair_condition", "pair_effect"} <= set(report["unstated"])
    definition = cases(name).world.definition
    pair_conditions = sum(
        1
        for label in corpus.planner.world.binary_leaves + corpus.planner.world.unary
        for effect in definition.event_type(label).effects
        if effect.condition and not effect.pair and any(lit.pair for lit in effect.condition)
    )
    assert report["unstated"]["pair_condition"] == pair_conditions


def test_a_conditional_effect_with_a_pair_condition_is_stated_in_neither_form(
    tmp_path: Path,
) -> None:
    """WM.59 (review point 11): on a tiny world where every effect is conditional and every
    condition literal is a pair literal, no conditional statement is made for such an effect,
    and its unconditional form is false by the corpus's truth test and by the oracle's
    re-reading of the definition."""
    data = yaml.safe_load((REPO / TINY_WORLD).read_text(encoding="utf-8"))
    data["event_types"] = {
        **data.get("event_types", {}),
        "effects": {
            "conditional_share": 1.0,
            "condition_kinds": {"feature": 0, "fluent": 0, "numeric": 0, "pair": 1},
        },
    }
    world_path = tmp_path / "pair_conditions.yaml"
    world_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = corpus_config(str(world_path))
    world = load_world(config)
    facts = Facts(config, world)
    truth = facts.truth
    definition = world.definition
    folder = world.result.write(tmp_path / "run")
    from truth_oracle import Oracle

    oracle = Oracle(folder, z=config.scalar_z)
    with_pair_condition = 0
    for label in world.binary_leaves + world.unary:
        record = definition.event_type(label)
        for effect in record.effects:
            if effect.pair or not any(lit.pair for lit in effect.condition):
                continue
            with_pair_condition += 1
            assert condition_of(effect, truth.cut_points) is None
            value = None if effect.numeric else bool(effect.value)
            direction = (
                None if not effect.numeric else ("raises" if effect.amount >= 0 else "lowers")
            )
            # the unconditional form is no entry of this effect
            assert not effect_matches(
                effect, effect.role, effect.fluent, value, direction, (), truth.cut_points
            )
            # and the statement holds only if another, unconditional, effect carries it
            others = [
                e
                for e in record.effects
                if e is not effect
                and not e.pair
                and e.role == effect.role
                and e.fluent == effect.fluent
                and not e.condition
            ]
            unconditional = truth.has_entry(
                label, EFFECT, effect.role, effect.fluent, value, direction, ()
            )
            assert unconditional == bool(others), (label, effect.text)
            form = {
                "level": CLASS,
                "quantifier": "nec_all",
                "polarity": True,
                "subject": {"head": "THING", "event": label, "role": effect.role},
                "predicate": {
                    "kind": EFFECT,
                    "label": effect.fluent,
                    **({"value": value} if direction is None else {"direction": direction}),
                },
            }
            if effect.fluent in world.base_fluents or effect.fluent in world.numeric_fluents:
                assert oracle.causal(form) is unconditional, (label, effect.text)
    assert with_pair_condition > 0
    # no statement of the world carries a condition: every condition holds a pair literal
    for statement in facts.causal_statements():
        assert not statement.subject.condition
    assert facts.causal_report["unstated"]["pair_condition"] == with_pair_condition
