"""Stage 1 acceptance tests: the lexicon.

Every concept type gets lexemes in the configured proportions. With both knobs at 0, lexemes and
concepts are one to one. With the knobs on, the realized synonym and homonym rates are within
tolerance of the configured rates.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from corpus_support import DEFAULT_WORLD, TINY_WORLD, corpus_config

from semantic_world.corpus import Lexicon, Streams, build_lexicon, load_world
from semantic_world.corpus.config import CONCEPT_TYPES
from semantic_world.corpus.lexicon import (
    ADJECTIVE,
    AGREEMENT_WORDS,
    EVERY_PAIR,
    FUNCTION,
    FUNCTION_WORD,
    FUNCTION_WORDS,
    GENERIC,
    INTRANSITIVE_VERB,
    LEXICON_COLUMNS,
    NO_PAIR,
    NOUN,
    PART_NOUN,
    THING,
    TRANSITIVE_VERB,
    event_types_without_word,
    function_word_glosses,
    relation_extent,
    world_concepts,
)
from semantic_world.corpus.world import EVENT_TYPE1_KIND, PART_KIND, PROPERTY_KIND

SEEDS = range(1, 21)


def lexicon_of(world, path: str = DEFAULT_WORLD, seed: int = 1, **sections: Any) -> Lexicon:
    config = corpus_config(path, seed=seed, **sections)
    return build_lexicon(config, world, Streams(config.seed))


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


# ---------------------------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------------------------


def test_concepts_of_the_default_world(default_world) -> None:
    world = default_world
    lexicon = lexicon_of(world)
    labels = {t: [c.label for c in lexicon.of_type(t)] for t in CONCEPT_TYPES}
    assert labels["category"] == list(world.categories) and len(labels["category"]) == 56
    assert labels["is"] == list(world.features[PROPERTY_KIND]) and len(labels["is"]) == 40
    assert labels["has"] == list(world.features[PART_KIND]) and len(labels["has"]) == 40
    assert labels["state"] == []  # state adjectives come in stage a7
    assert labels["event_unary"] == list(world.features[EVENT_TYPE1_KIND])
    assert len(labels["event_unary"]) == 20
    assert labels["event"] == [
        v for v in world.binary_leaves if v not in lexicon.event_types_without_word
    ]
    assert labels["event_category"] == [
        v
        for v in world.binary
        if world.event_types[v].category and v not in lexicon.event_types_without_word
    ]
    # every two-place leaf event type has a patient capacity; a quarter get words
    assert labels["patient_projection"] == list(world.patient_capacities)
    assert len(labels["patient_projection"]) == 7
    assert sum(lexicon.is_named(c) for c in labels["patient_projection"]) == 2
    assert labels["scalar"] == [
        "SCALARDIM.1.HIGH", "SCALARDIM.1.LOW", "SCALARDIM.2.HIGH", "SCALARDIM.2.LOW"
    ]  # fmt: skip
    assert [c.label for c in lexicon.of_type(GENERIC)] == [THING]
    # no agent capacity is a concept: the event type itself is the word
    assert not any(c.label.startswith("CAN.") for c in lexicon.concepts)


def test_parts_of_speech(default_world) -> None:
    lexicon = lexicon_of(default_world)
    expected = {
        "category": NOUN,
        "is": ADJECTIVE,
        "has": PART_NOUN,
        "state": ADJECTIVE,
        "event_unary": INTRANSITIVE_VERB,
        "event": TRANSITIVE_VERB,
        "event_category": TRANSITIVE_VERB,
        "patient_projection": ADJECTIVE,
        "scalar": ADJECTIVE,
        GENERIC: NOUN,
        FUNCTION: FUNCTION_WORD,
    }
    for concept in lexicon.concepts:
        assert concept.pos == expected[concept.type], concept
    for lexeme in lexicon.lexemes:
        assert lexeme.pos == lexicon.concept(lexeme.concept).pos


def test_concept_order_follows_the_specification_table(default_world) -> None:
    lexicon = lexicon_of(default_world)
    types = [c.type for c in lexicon.concepts]
    order = [t for t in CONCEPT_TYPES if t != "state"] + [GENERIC, FUNCTION]
    assert [t for i, t in enumerate(types) if i == 0 or types[i - 1] != t] == order


def test_a_world_without_two_place_event_types_or_scalars(world_files) -> None:
    world = load_world(corpus_config(world_files["plain"]))
    lexicon = lexicon_of(world, world_files["plain"])
    for concept_type in ("event", "event_category", "patient_projection", "scalar"):
        assert lexicon.of_type(concept_type) == ()
    assert lexicon.event_types_without_word == {}
    assert len(lexicon.content_lexemes) == 6 + 8 + 8 + 4 + 1
    assert lexicon.stats()["lexemes_by_pos"][TRANSITIVE_VERB] == 0


# ---------------------------------------------------------------------------------------------
# Verbs without a word
# ---------------------------------------------------------------------------------------------


def test_an_event_type_category_that_holds_for_every_pair_gets_no_lexeme(cases) -> None:
    # In the deep world, the base relation of EVENTTYPE2.2 holds for every pair (since stage
    # a5b, every category of the tiny world has a defining constraint).
    case = cases("deep")
    world = case.world
    lexicon = case.lexicon()
    n = world.count
    assert world.able("EVENTTYPE2.2").sum() == n * (n - 1)
    assert lexicon.event_types_without_word == {"EVENTTYPE2.2": EVERY_PAIR}
    assert not lexicon.is_named("EVENTTYPE2.2") and lexicon.lexemes_of("EVENTTYPE2.2") == ()
    assert "EVENTTYPE2.2" not in {c.label for c in lexicon.concepts}
    assert "EVENTTYPE2.2" not in lexicon.unnamed  # left out as a concept, not by the proportion
    assert [c.label for c in lexicon.of_type("event_category")] == ["EVENTTYPE2.1"]
    assert [c.label for c in lexicon.of_type("event")] == [
        "EVENTTYPE2.1.1", "EVENTTYPE2.1.2", "EVENTTYPE2.2.1", "EVENTTYPE2.2.2"
    ]  # fmt: skip
    assert lexicon.stats()["event_types_without_word"] == {"EVENTTYPE2.2": EVERY_PAIR}


@pytest.mark.parametrize("world_name", ["tiny_world", "default_world"])
def test_event_types_without_a_word_match_brute_force(world_name: str, request) -> None:
    world = request.getfixturevalue(world_name)
    named = {c.label for c in world_concepts(world)}
    without_word = event_types_without_word(world)
    for label in world.binary:
        holds = world.able(label)[~np.eye(world.count, dtype=bool)]
        if holds.all():
            assert without_word[label] == EVERY_PAIR
        elif not holds.any():
            assert without_word[label] == NO_PAIR
        else:
            assert label not in without_word
        assert (label in named) != (label in without_word)


def test_relation_extent() -> None:
    def world(matrix: np.ndarray):
        return SimpleNamespace(count=len(matrix), able=lambda label: matrix)

    every = ~np.eye(4, dtype=bool)
    assert relation_extent(world(every), "EVENTTYPE2.1") == EVERY_PAIR
    assert relation_extent(world(np.zeros((4, 4), dtype=bool)), "EVENTTYPE2.1") == NO_PAIR
    some = every.copy()
    some[0, 1] = False
    assert relation_extent(world(some), "EVENTTYPE2.1") is None
    one = np.zeros((4, 4), dtype=bool)
    one[2, 3] = True
    assert relation_extent(world(one), "EVENTTYPE2.1") is None
    # a single instance has no pair
    assert relation_extent(world(np.zeros((1, 1), dtype=bool)), "EVENTTYPE2.1") == NO_PAIR


def test_an_event_type_that_holds_for_no_pair_gets_no_lexeme(tiny_world, monkeypatch) -> None:
    n = tiny_world.count
    tiny_world.able("EVENTTYPE2.2.1")  # fill the table, then blank one event type
    monkeypatch.setitem(tiny_world._able, "EVENTTYPE2.2.1", np.zeros((n, n), dtype=bool))
    lexicon = lexicon_of(tiny_world, TINY_WORLD)
    assert lexicon.event_types_without_word == {"EVENTTYPE2.2.1": NO_PAIR}
    assert [c.label for c in lexicon.of_type("event")] == [
        "EVENTTYPE2.1.1", "EVENTTYPE2.1.2", "EVENTTYPE2.2.2"
    ]  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Named proportions
# ---------------------------------------------------------------------------------------------


def units(lexicon: Lexicon, concept_type: str) -> list[str]:
    """The units a named proportion counts: concepts, or scalar dimensions."""
    labels = [c.label for c in lexicon.of_type(concept_type)]
    if concept_type == "scalar":
        return sorted({label.rsplit(".", 1)[0] for label in labels})
    return labels


@pytest.mark.parametrize("proportion", [0.0, 0.25, 0.5, 0.8, 1.0])
def test_every_concept_type_gets_lexemes_in_the_configured_proportion(
    default_world, proportion: float
) -> None:
    proportions = dict.fromkeys(CONCEPT_TYPES, proportion)
    for seed in (1, 2, 3):
        lexicon = lexicon_of(default_world, seed=seed, lexicon={"named_proportion": proportions})
        for concept_type in CONCEPT_TYPES:
            all_units = units(lexicon, concept_type)
            named = [u for u in all_units if lexicon.is_named(u) or lexicon.is_named(u + ".HIGH")]
            assert len(named) == round_half_up(proportion * len(all_units)), concept_type
        # the generic noun and the function words always have words
        assert lexicon.is_named(THING)
        assert all(lexicon.is_named(c.label) for c in lexicon.of_type(FUNCTION))
        content = [c.label for c in lexicon.concepts if c.content]
        assert set(lexicon.unnamed) == {label for label in content if not lexicon.is_named(label)}
        assert list(lexicon.unnamed) == [label for label in content if label in lexicon.unnamed]


def test_each_type_has_its_own_proportion(default_world) -> None:
    proportions = {
        "category": 0.5,
        "is": 0.1,
        "has": 1.0,
        "event_unary": 0.75,
        "event": 0.4,
        "event_category": 0.0,
        "patient_projection": 0.5,
        "scalar": 0.5,
    }
    lexicon = lexicon_of(default_world, lexicon={"named_proportion": proportions})
    counts = lexicon.stats()
    assert counts["concepts"] == {
        "category": 56,
        "is": 40,
        "has": 40,
        "state": 0,
        "event_unary": 20,
        "event": 7,
        "event_category": 3,
        "patient_projection": 7,
        "scalar": 4,
        GENERIC: 1,
        FUNCTION: 15,
    }
    assert counts["named_concepts"] == {
        "category": 28,
        "is": 4,
        "has": 40,
        "state": 0,
        "event_unary": 15,
        "event": 3,  # 2.8 rounds to 3
        "event_category": 0,
        "patient_projection": 4,  # 3.5 rounds to 4
        "scalar": 2,  # one of the two dimensions, with both of its poles
        GENERIC: 1,
        FUNCTION: 15,
    }


def test_the_poles_of_a_scalar_dimension_are_named_together(default_world) -> None:
    seen = set()
    for seed in SEEDS:
        lexicon = lexicon_of(
            default_world, seed=seed, lexicon={"named_proportion": {"scalar": 0.5}}
        )
        named = [c.label for c in lexicon.of_type("scalar") if lexicon.is_named(c.label)]
        assert named in (
            ["SCALARDIM.1.HIGH", "SCALARDIM.1.LOW"],
            ["SCALARDIM.2.HIGH", "SCALARDIM.2.LOW"],
        )
        seen.add(named[0])
    assert len(seen) == 2  # the dimension is drawn at random


def test_one_types_proportion_never_changes_another_types_concepts(default_world) -> None:
    def named(lexicon: Lexicon, concept_type: str) -> list[str]:
        return [c.label for c in lexicon.of_type(concept_type) if lexicon.is_named(c.label)]

    half = dict.fromkeys(CONCEPT_TYPES, 0.5)
    base = lexicon_of(default_world, lexicon={"named_proportion": half})
    changed = lexicon_of(default_world, lexicon={"named_proportion": {**half, "is": 0.2}})
    assert named(changed, "is") != named(base, "is")
    for concept_type in CONCEPT_TYPES:
        if concept_type != "is":
            assert named(changed, concept_type) == named(base, concept_type)


def test_the_named_concepts_are_drawn_at_random(default_world) -> None:
    half = {"named_proportion": {"category": 0.5}}
    draws = {tuple(lexicon_of(default_world, seed=seed, lexicon=half).unnamed) for seed in SEEDS}
    assert len(draws) == len(SEEDS)
    assert lexicon_of(default_world, seed=3, lexicon=half) == lexicon_of(
        default_world, seed=3, lexicon=half
    )


# ---------------------------------------------------------------------------------------------
# One lexeme per concept
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("world_name", ["tiny_world", "default_world"])
def test_with_both_knobs_at_zero_lexemes_and_concepts_are_one_to_one(
    world_name: str, request
) -> None:
    world = request.getfixturevalue(world_name)
    path = TINY_WORLD if world_name == "tiny_world" else DEFAULT_WORLD
    every = {"named_proportion": {"patient_projection": 1.0}}
    lexicon = lexicon_of(world, path, lexicon=every)
    assert lexicon.unnamed == ()
    assert [x.concept for x in lexicon.lexemes] == [c.label for c in lexicon.concepts]
    for concept in lexicon.concepts:
        (lexeme,) = lexicon.lexemes_of(concept.label)
        assert (lexeme.concept, lexeme.pos, lexeme.gloss) == (
            concept.label,
            concept.pos,
            concept.gloss,
        )
    assert all(x.same_form_as is None for x in lexicon.lexemes)
    assert lexicon.homonym_pairs() == ()
    stats = lexicon.stats()
    assert stats["synonym_rate"] == 0 and stats["homonym_rate"] == 0
    assert stats["lexemes"] == len(lexicon.concepts)


def test_lexeme_labels(default_world) -> None:
    lexicon = lexicon_of(default_world)
    assert [x.label for x in lexicon.lexemes] == [f"LEXEME.{i}" for i in range(1, 189)]
    assert len(lexicon.content_lexemes) == 173 and len(lexicon.function_lexemes) == 15
    # content lexemes come first, then the function words
    assert all(x.content for x in lexicon.lexemes[:173])
    assert lexicon.lexeme("LEXEME.1").concept == "CATEGORY.1"
    assert lexicon.lexeme("LEXEME.173").concept == THING
    assert lexicon.lexeme("LEXEME.174").gloss == "a"
    with pytest.raises(KeyError, match="unknown lexeme"):
        lexicon.lexeme("LEXEME.189")
    with pytest.raises(KeyError, match="unknown concept"):
        lexicon.concept("CATEGORY.99")


def test_the_same_seed_gives_the_same_lexicon(default_world) -> None:
    knobs = {
        "synonym_rate": 0.3,
        "homonym_rate": 0.2,
        "named_proportion": dict.fromkeys(CONCEPT_TYPES, 0.7),
    }
    first = lexicon_of(default_world, seed=5, lexicon=knobs)
    assert first == lexicon_of(default_world, seed=5, lexicon=knobs)
    assert first != lexicon_of(default_world, seed=6, lexicon=knobs)
    # with nothing to draw, the corpus seed does not matter
    assert lexicon_of(default_world, seed=5) == lexicon_of(default_world, seed=6)


# ---------------------------------------------------------------------------------------------
# Function words
# ---------------------------------------------------------------------------------------------


def function_glosses(world, **morphology: Any) -> list[str]:
    lexicon = lexicon_of(world, TINY_WORLD, grammar={"morphology": morphology})
    return [x.gloss for x in lexicon.function_lexemes]


def test_function_words(tiny_world) -> None:
    base = [
        "a", "the", "all", "most", "some", "no", "not", "can", "is", "has", "with", "without",
        "and", "that", "it",
    ]  # fmt: skip
    assert list(FUNCTION_WORDS) == base and AGREEMENT_WORDS == ("are", "have")
    assert function_glosses(tiny_world) == base
    # with agreement on, are and have are the plural forms of is and has
    assert function_glosses(tiny_world, number={"enabled": True}) == base + ["are", "have"]
    assert function_glosses(tiny_world, number={"enabled": True, "agreement": False}) == base
    assert function_glosses(tiny_world, number={"enabled": False, "agreement": True}) == base
    # an inflection realized as a separate word adds a function word, glossed like the affix
    assert function_glosses(tiny_world, number={"enabled": True, "realization": "word"}) == base + [
        "are",
        "have",
        "PLURAL",
    ]
    assert function_glosses(tiny_world, tense={"enabled": True}) == base  # an affix
    assert function_glosses(tiny_world, tense={"enabled": True, "realization": "word"}) == base + [
        "PAST"
    ]
    assert function_glosses(tiny_world, aspect={"enabled": True}) == base + ["PROGRESSIVE"]
    everything = {
        "number": {"enabled": True, "realization": "word"},
        "tense": {"enabled": True, "realization": "word"},
        "aspect": {"enabled": True, "realization": "word"},
    }
    assert function_glosses(tiny_world, **everything) == base + [
        "are",
        "have",
        "PLURAL",
        "PAST",
        "PROGRESSIVE",
    ]
    config = corpus_config(grammar={"morphology": everything})
    assert list(function_word_glosses(config)) == function_glosses(tiny_world, **everything)


def test_function_words_are_ordinary_lexemes(tiny_world) -> None:
    lexicon = lexicon_of(
        tiny_world, TINY_WORLD, grammar={"morphology": {"aspect": {"enabled": True}}}
    )
    the = lexicon.function_word("the")
    assert (the.pos, the.concept, the.gloss) == (FUNCTION_WORD, "THE", "the")
    assert lexicon.concept("THE").type == FUNCTION
    assert lexicon.function_word("no").concept == "NO"
    progressive = lexicon.function_word("PROGRESSIVE")
    assert (progressive.concept, progressive.gloss) == ("PROGRESSIVE", "PROGRESSIVE")
    for gloss in ("are", "PLURAL", "thing", "THING"):
        with pytest.raises(KeyError, match="no function word"):
            lexicon.function_word(gloss)


def test_grammar_settings_never_change_a_content_lexeme(default_world) -> None:
    knobs = {"synonym_rate": 0.3, "homonym_rate": 0.2}
    base = lexicon_of(default_world, lexicon=knobs)
    grammar = {
        "word_order": {"clause": "SOV", "adjective": "after"},
        "morphology": {
            "number": {"enabled": True, "realization": "word"},
            "tense": {"enabled": True},
            "aspect": {"enabled": True},
        },
    }
    changed = lexicon_of(default_world, lexicon=knobs, grammar=grammar)
    assert changed.content_lexemes == base.content_lexemes
    assert len(changed.function_lexemes) == len(base.function_lexemes) + 4
    shared = len(base.function_lexemes)
    assert changed.function_lexemes[:shared] == base.function_lexemes


# ---------------------------------------------------------------------------------------------
# Synonyms
# ---------------------------------------------------------------------------------------------


def test_the_realized_synonym_rate_is_within_tolerance(default_world) -> None:
    for rate in (0.1, 0.3, 0.6):
        rates = [
            lexicon_of(default_world, seed=seed, lexicon={"synonym_rate": rate}).stats()[
                "synonym_rate"
            ]
            for seed in SEEDS
        ]
        # 173 content concepts and 20 seeds: the standard error of the mean is under 0.01
        assert abs(np.mean(rates) - rate) < 0.03, (rate, np.mean(rates))
        assert all(abs(r - rate) < 0.15 for r in rates)
    assert lexicon_of(default_world, lexicon={"synonym_rate": 1.0}).stats()["synonym_rate"] == 1


def test_synonyms(default_world) -> None:
    base = lexicon_of(default_world)
    lexicon = lexicon_of(default_world, lexicon={"synonym_rate": 0.3})
    doubled = [c for c in lexicon.concepts if len(lexicon.lexemes_of(c.label)) == 2]
    assert 20 < len(doubled) < 90
    for concept in lexicon.concepts:
        expected = (0,) if concept.label in lexicon.unnamed else (1, 2)
        assert len(lexicon.lexemes_of(concept.label)) in expected
    assert len(lexicon.content_lexemes) == 173 + len(doubled)
    for concept in doubled:
        first, second = lexicon.lexemes_of(concept.label)
        assert concept.content  # function words never get synonyms
        assert (first.concept, first.pos, first.gloss) == (second.concept, second.pos, second.gloss)
        assert first.label != second.label
    # first lexemes keep their labels, second lexemes follow them, then the function words
    assert lexicon.lexemes[:173] == base.lexemes[:173]
    seconds = lexicon.lexemes[173 : 173 + len(doubled)]
    assert [x.concept for x in seconds] == [c.label for c in doubled]
    assert [x.gloss for x in lexicon.lexemes[173 + len(doubled) :]] == [
        x.gloss for x in base.function_lexemes
    ]
    assert [x.label for x in lexicon.lexemes] == [
        f"LEXEME.{i}" for i in range(1, len(lexicon.lexemes) + 1)
    ]


def test_the_synonyms_at_one_rate_are_among_those_at_a_higher_rate(default_world) -> None:
    def doubled(rate: float) -> set[str]:
        lexicon = lexicon_of(default_world, lexicon={"synonym_rate": rate})
        return {c.label for c in lexicon.concepts if len(lexicon.lexemes_of(c.label)) == 2}

    assert doubled(0.0) == set()
    assert doubled(0.1) < doubled(0.3) < doubled(0.6)


def test_synonyms_only_for_named_concepts(default_world) -> None:
    knobs = {"synonym_rate": 1.0, "named_proportion": {"is": 0.5, "scalar": 0.0}}
    lexicon = lexicon_of(default_world, lexicon=knobs)
    for concept in lexicon.concepts:
        expected = 0 if concept.label in lexicon.unnamed else 2 if concept.content else 1
        assert len(lexicon.lexemes_of(concept.label)) == expected, concept
    assert len(lexicon.lexemes_of(THING)) == 2


# ---------------------------------------------------------------------------------------------
# Homonyms
# ---------------------------------------------------------------------------------------------


def test_the_realized_homonym_rate_is_within_tolerance(default_world) -> None:
    for rate in (0.1, 0.2, 0.5):
        for seed in SEEDS:
            lexicon = lexicon_of(default_world, seed=seed, lexicon={"homonym_rate": rate})
            stats = lexicon.stats()
            shared = [x for x in lexicon.content_lexemes if x.same_form_as is not None]
            assert stats["homonym_pairs"] == len(shared)
            assert stats["homonym_rate"] == 2 * len(shared) / 173
            # the number of pairs is rate x lexemes / 2, rounded down or up
            assert abs(stats["homonym_rate"] - rate) <= 2 / 173
    every = lexicon_of(default_world, lexicon={"homonym_rate": 1.0}).stats()
    assert every["homonym_pairs"] == 86  # 173 lexemes: one is left over


def test_the_share_of_same_pos_pairs_is_within_tolerance(default_world) -> None:
    for same_pos in (0.0, 0.5, 0.8, 1.0):
        pairs = same = 0
        for seed in SEEDS:
            knobs = {"homonym_rate": 0.2, "homonym_same_pos": same_pos}
            stats = lexicon_of(default_world, seed=seed, lexicon=knobs).stats()
            pairs += stats["homonym_pairs"]
            same += stats["homonym_pairs_same_pos"]
        assert pairs > 300
        if same_pos in (0.0, 1.0):
            assert same == same_pos * pairs
        else:
            assert abs(same / pairs - same_pos) < 0.08, (same_pos, same / pairs)


def test_homonym_pairs(default_world) -> None:
    knobs = {"homonym_rate": 0.3, "synonym_rate": 0.3}
    lexicon = lexicon_of(default_world, lexicon=knobs)
    pairs = lexicon.homonym_pairs()
    assert len(pairs) > 20
    in_a_pair: list[str] = []
    for earlier, later in pairs:
        # two lexemes, each with its one concept, and the concepts differ
        assert earlier.concept != later.concept
        assert earlier.content and later.content  # function words are never homonyms
        assert earlier.same_form_as is None and later.same_form_as == earlier.label
        assert int(earlier.label[7:]) < int(later.label[7:])
        in_a_pair += [earlier.label, later.label]
    assert len(set(in_a_pair)) == len(in_a_pair)  # no lexeme is in two pairs
    assert not any(x.same_form_as for x in lexicon.function_lexemes)
    # homonyms change no lexeme's label, concept, or part of speech
    plain = lexicon_of(default_world, lexicon={"synonym_rate": 0.3})
    assert [(x.label, x.concept, x.pos, x.gloss) for x in lexicon.lexemes] == [
        (x.label, x.concept, x.pos, x.gloss) for x in plain.lexemes
    ]


def test_a_homonym_never_pairs_a_concept_with_its_own_synonym(tiny_world) -> None:
    for seed in SEEDS:
        knobs = {"homonym_rate": 1.0, "synonym_rate": 1.0}
        lexicon = lexicon_of(tiny_world, TINY_WORLD, seed=seed, lexicon=knobs)
        assert len(lexicon.content_lexemes) == 72  # 36 content concepts, each with a synonym
        pairs = lexicon.homonym_pairs()
        assert len(pairs) >= 33
        assert all(a.concept != b.concept for a, b in pairs)


def test_a_missing_kind_of_partner_falls_back_to_the_other_kind(world_files) -> None:
    # A world of categories only has one part of speech: every pair is within it.
    world = load_world(corpus_config(world_files["plain"]))
    only_nouns = {
        "named_proportion": {"is": 0.0, "has": 0.0, "event_unary": 0.0},
        "homonym_rate": 1.0,
        "homonym_same_pos": 0.0,
    }
    lexicon = lexicon_of(world, world_files["plain"], lexicon=only_nouns)
    assert {x.pos for x in lexicon.content_lexemes} == {NOUN}
    pairs = lexicon.homonym_pairs()
    assert len(pairs) == 3 and all(a.pos == b.pos for a, b in pairs)


# ---------------------------------------------------------------------------------------------
# The lexicon table
# ---------------------------------------------------------------------------------------------


def test_lexicon_frame(default_world) -> None:
    lexicon = lexicon_of(default_world, lexicon={"homonym_rate": 0.2, "synonym_rate": 0.2})
    frame = lexicon.frame()
    assert tuple(frame.columns) == LEXICON_COLUMNS
    assert frame.columns == ["label", "pos", "concept", "word", "spelling", "gloss", "same_form_as"]
    assert frame.height == len(lexicon.lexemes)
    assert frame["label"].to_list() == [x.label for x in lexicon.lexemes]
    # one concept per lexeme
    assert frame["concept"].null_count() == 0
    assert not frame["concept"].str.contains(";").any()
    # word forms are attached later
    assert frame["word"].null_count() == frame.height
    assert frame["spelling"].null_count() == frame.height
    shared = frame.filter(pl.col("same_form_as").is_not_null())
    assert shared.height == len(lexicon.homonym_pairs()) > 0
    row = frame.row(by_predicate=pl.col("gloss") == "the", named=True)
    assert (row["pos"], row["concept"]) == ("function_word", "THE")
