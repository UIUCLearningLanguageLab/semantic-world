"""The generator's acceptance tests: phonotactics, exclusions, distances, distributions, and
determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest
from wordforms_support import needs_cmudict, needs_wordfreq

from semantic_world.wordforms.config import parse_config
from semantic_world.wordforms.english import edit_distance, strip_stress
from semantic_world.wordforms.generate import (
    GenerationError,
    draw_stress,
    draw_syllable_count,
    generate_lexicon,
    lexicon_distance,
)
from semantic_world.wordforms.streams import Streams

pytestmark = [needs_cmudict, needs_wordfreq]


def lexicon_for(data, english=None, seed=1):
    """The lexicon of a configuration. The generator loads the dictionary with the
    configuration's ``english_min_zipf``; ``english`` is accepted for the stage-1 call sites."""
    config = parse_config(data, "test", seed=seed)
    return generate_lexicon(config, Streams(config.seed).generate)


@pytest.fixture(scope="module")
def default_lexicon(english):
    return lexicon_for({}, english)


def test_labels_and_shape(default_lexicon):
    words = default_lexicon.words
    assert len(words) == 500
    assert [w.label for w in words] == [f"WORD.{i}" for i in range(1, 501)]
    for w in words:
        assert not w.real_word
        assert w.syllable_count in (1, 2, 3)
        assert len(w.stress) == w.syllable_count and w.stress.count("1") == 1
        assert set(w.stress) <= {"0", "1"}
        assert w.ipa and w.espeak and w.spelling
        assert math.isfinite(w.log_probability) and w.log_probability < 0
        assert w.nearest_english


def test_every_form_passes_the_phonotactic_check(default_lexicon, english):
    for w in default_lexicon.words:
        assert english.phonotactic(w.phones), w.arpabet


def test_no_form_is_a_real_word_when_real_words_are_excluded(default_lexicon, english):
    for w in default_lexicon.words:
        assert not english.is_pronunciation(w.phones), w.arpabet
        assert english.neighbors(w.phones).distance >= 1


def test_forms_are_distinct_with_min_lexicon_distance_1(default_lexicon):
    stripped = [w.stripped for w in default_lexicon.words]
    assert len(set(stripped)) == len(stripped)
    # minimal pairs are allowed, and the lexicon-neighbor counts are symmetric
    total = sum(w.lexicon_neighbors for w in default_lexicon.words)
    assert total % 2 == 0
    assert default_lexicon.summary()["minimal_pairs"] == total // 2


def test_min_lexicon_distance_2_forbids_minimal_pairs(english):
    lexicon = lexicon_for({"wordforms": {"count": 150, "min_lexicon_distance": 2}}, english)
    stripped = [w.stripped for w in lexicon.words]
    for i in range(len(stripped)):
        for j in range(i + 1, len(stripped)):
            assert edit_distance(stripped[i], stripped[j]) >= 2
    assert all(w.lexicon_neighbors == 0 for w in lexicon.words)
    assert lexicon.summary()["minimal_pairs"] == 0


def test_min_english_distance_2_forbids_english_neighbors(english):
    lexicon = lexicon_for({"wordforms": {"count": 100, "min_english_distance": 2}}, english)
    for w in lexicon.words:
        assert w.english_neighbors == 0
        assert english.neighbors(w.phones).distance >= 2


def test_min_english_distance_3(english):
    lexicon = lexicon_for(
        {"wordforms": {"count": 30, "min_english_distance": 3, "syllables": {3: 1}}}, english
    )
    for w in lexicon.words:
        assert english.neighbors(w.phones).distance >= 3


def test_syllable_counts_match_the_distribution(english):
    lexicon = lexicon_for({"wordforms": {"count": 2000}}, english)
    counts = lexicon.summary()["syllable_counts"]
    for k, p in {1: 0.3, 2: 0.5, 3: 0.2}.items():
        assert abs(counts[k] / 2000 - p) < 0.04, counts


def test_stress_pattern_matches_the_probability(english):
    lexicon = lexicon_for({"wordforms": {"count": 1000, "syllables": {2: 0.5, 3: 0.5}}}, english)
    initial = sum(w.stress[0] == "1" for w in lexicon.words)
    assert abs(initial / 1000 - 0.8) < 0.05
    three = [w for w in lexicon.words if w.syllable_count == 3 and w.stress[0] != "1"]
    assert {w.stress for w in three} == {"010", "001"}


def test_same_seed_gives_identical_word_tables(english):
    a = lexicon_for({"wordforms": {"count": 200}}, english, seed=3)
    b = lexicon_for({"wordforms": {"count": 200}}, english, seed=3)
    assert [w.record() for w in a.words] == [w.record() for w in b.words]
    c = lexicon_for({"wordforms": {"count": 200}}, english, seed=4)
    assert [w.arpabet for w in a.words] != [w.arpabet for w in c.words]


def test_word_forms_do_not_depend_on_other_settings(english):
    a = lexicon_for({"wordforms": {"count": 50}}, english)
    b = lexicon_for(
        {"wordforms": {"count": 50}, "synthesis": {"tokens_per_speaker": 5}, "embeddings": []},
        english,
    )
    assert [w.arpabet for w in a.words] == [w.arpabet for w in b.words]


def test_spellings_are_unique(english):
    lexicon = lexicon_for({"wordforms": {"count": 300, "syllables": {1: 1}}}, english)
    spellings = [w.spelling for w in lexicon.words]
    assert len(set(spellings)) == len(spellings)
    assert all(s.isalpha() or s[:-1].isalpha() for s in spellings)


def test_english_source(english):
    lexicon = lexicon_for({"wordforms": {"source": "english", "count": 100}}, english)
    assert lexicon.summary()["real_words"] == 100
    for w in lexicon.words:
        assert w.real_word and w.english_word
        assert w.spelling == w.english_word
        assert english.words[w.english_word][0] == w.phones
        assert w.nearest_english == english.neighbors(w.phones).nearest
        assert english.neighbors(w.phones).distance == 0
    stripped = [w.stripped for w in lexicon.words]
    assert len(set(stripped)) == len(stripped)
    counts = lexicon.summary()["syllable_counts"]
    assert set(counts) <= {1, 2, 3}


def test_mixed_source(english):
    lexicon = lexicon_for(
        {"wordforms": {"source": "mixed", "count": 40, "mixed_proportion_english": 0.25}}, english
    )
    assert lexicon.summary()["real_words"] == 10
    real = [w for w in lexicon.words if w.real_word]
    pseudo = [w for w in lexicon.words if not w.real_word]
    assert len(real) == 10 and len(pseudo) == 30
    for w in pseudo:
        assert not english.is_pronunciation(w.phones)
    # the same forms come out when only the real-word proportion changes the slots
    assert len({w.stripped for w in lexicon.words}) == 40


def test_real_words_allowed_when_not_excluded(english):
    lexicon = lexicon_for(
        {"wordforms": {"count": 300, "exclude_real_words": False, "syllables": {1: 1}}}, english
    )
    assert lexicon.rejections.english == 0
    assert any(english.is_pronunciation(w.phones) for w in lexicon.words)
    assert not any(w.real_word for w in lexicon.words)


def test_impossible_configuration_reports_what_to_loosen(english):
    with pytest.raises(GenerationError, match="min_lexicon_distance"):
        lexicon_for(
            {"wordforms": {"count": 400, "syllables": {1: 1}, "min_lexicon_distance": 4}}, english
        )


def test_draws():
    rng = np.random.default_rng(0)
    counts = [draw_syllable_count(rng, {1: 0.5, 3: 0.5}) for _ in range(200)]
    assert set(counts) == {1, 3}
    assert all(draw_stress(rng, 1, 0.0) == 0 for _ in range(10))
    assert all(draw_stress(rng, 3, 1.0) == 0 for _ in range(10))
    assert set(draw_stress(rng, 3, 0.0) for _ in range(100)) == {1, 2}


def test_lexicon_distance():
    accepted = [("K", "AE", "T"), ("D", "AO", "G", "Z")]
    assert lexicon_distance(("K", "AE", "T"), accepted, 3) == 0
    assert lexicon_distance(("B", "AE", "T"), accepted, 3) == 1
    assert lexicon_distance(("F", "IY", "SH", "M", "AH", "N"), accepted, 3) == 3
    assert strip_stress(("K", "AE1", "T")) == ("K", "AE", "T")


# ---------------------------------------------------------------------------------------------
# Patterns from common words (stage 1a)
# ---------------------------------------------------------------------------------------------


def uncommon_trigram_words(lexicon, common_english):
    """The number of words with a phoneme trigram that occurs in no common English word."""
    return sum(not common_english.phonotactic(w.phones) for w in lexicon.words)


def test_default_words_use_only_trigrams_of_common_words(default_lexicon, common_english):
    assert default_lexicon.english_summary["english_min_zipf"] == 3.0
    assert uncommon_trigram_words(default_lexicon, common_english) == 0


def test_whole_dictionary_patterns_give_uncommon_trigrams(common_english):
    lexicon = lexicon_for({"wordforms": {"english_min_zipf": None}})
    assert lexicon.english_summary["english_min_zipf"] is None
    # Jon measured 144 of the 500 default words before the change
    assert uncommon_trigram_words(lexicon, common_english) > 100


def test_real_words_are_rejected_against_the_whole_dictionary(default_lexicon, english):
    # english is the whole dictionary, rare words included
    assert default_lexicon.rejections.english > 0
    for w in default_lexicon.words:
        assert not english.is_pronunciation(w.phones)
        assert w.english_neighbors == english.count_at_one(w.stripped)


def test_a_higher_threshold_changes_the_words():
    a = lexicon_for({"wordforms": {"count": 50}})
    b = lexicon_for({"wordforms": {"count": 50, "english_min_zipf": 4.5}})
    assert [w.arpabet for w in a.words] != [w.arpabet for w in b.words]


def test_real_words_are_drawn_from_the_common_words(common_english):
    lexicon = lexicon_for({"wordforms": {"source": "english", "count": 200}})
    assert all(
        w.real_word and w.english_word in common_english.pattern_words for w in lexicon.words
    )
    mixed = lexicon_for({"wordforms": {"source": "mixed", "count": 60}})
    real = [w for w in mixed.words if w.real_word]
    assert len(real) == 30
    assert all(w.english_word in common_english.pattern_words for w in real)


def test_real_words_come_from_the_whole_dictionary_without_a_threshold(common_english):
    lexicon = lexicon_for(
        {"wordforms": {"source": "english", "count": 200, "english_min_zipf": None}}
    )
    assert any(w.english_word not in common_english.pattern_words for w in lexicon.words)


def test_no_pseudoword_is_spelled_like_a_common_word(common_english):
    from semantic_world.wordforms.spelling import Speller

    speller = Speller.load()
    lexicon = lexicon_for({"wordforms": {"count": 1500}}, seed=2)
    assert not any(w.spelling in common_english.pattern_words for w in lexicon.words)
    # some words needed the next-best spelling, and most did not
    moved = [w for w in lexicon.words if speller.spell(w.phones) in common_english.pattern_words]
    assert 0 < len(moved) < 75
    for w in moved:
        assert w.spelling.rstrip("0123456789") in list(speller.candidates(w.phones))[1:]
