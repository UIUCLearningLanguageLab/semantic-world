"""The English source: syllabification, counts, the trigram model, and neighbors."""

from __future__ import annotations

import math

import pytest
from wordforms_support import needs_cmudict

from semantic_world.wordforms.english import (
    MIN_ONSET_WORDS,
    PHONEMES,
    POSITIONS,
    edit_distance,
    position_of,
    rime_position_of,
    syllabify,
)

pytestmark = needs_cmudict

# Thirty words, syllabified by hand under maximal onset with the legal onsets of English. Each
# entry is the CMUdict pronunciation with periods between the syllables.
HAND_CHECKED = {
    "cat": "K AE1 T",
    "hello": "HH AH0 . L OW1",
    "extra": "EH1 K . S T R AH0",
    "atlas": "AE1 T . L AH0 S",
    "happy": "HH AE1 . P IY0",
    "banana": "B AH0 . N AE1 . N AH0",
    "computer": "K AH0 M . P Y UW1 . T ER0",
    "window": "W IH1 N . D OW0",
    "monster": "M AA1 N . S T ER0",
    "construct": "K AH0 N . S T R AH1 K T",
    "apple": "AE1 . P AH0 L",
    "little": "L IH1 . T AH0 L",
    "mister": "M IH1 . S T ER0",
    "actor": "AE1 K . T ER0",
    "picture": "P IH1 K . CH ER0",
    "tumble": "T AH1 M . B AH0 L",
    "english": "IH1 NG . G L IH0 SH",
    "answer": "AE1 N . S ER0",
    "secret": "S IY1 . K R AH0 T",
    "athlete": "AE1 TH . L IY2 T",
    "umbrella": "AH0 M . B R EH1 . L AH0",
    "oxygen": "AA1 K . S AH0 . JH AH0 N",
    "whisper": "W IH1 . S P ER0",
    "pilgrim": "P IH1 L . G R AH0 M",
    "instrument": "IH1 N . S T R AH0 . M AH0 N T",
    "yellow": "Y EH1 . L OW0",
    "strengths": "S T R EH1 NG K TH S",
    "hamster": "HH AE1 M . S T ER0",
    "explain": "IH0 K . S P L EY1 N",
    "abrupt": "AH0 . B R AH1 P T",
}


def test_hand_checked_syllabifications(english):
    assert len(HAND_CHECKED) == 30
    for word, expected in HAND_CHECKED.items():
        pron = english.words[word][0]
        got = " . ".join(str(s) for s in english.syllables[pron])
        assert got == expected, word


def test_legal_onsets(english):
    onsets = {" ".join(o) for o in english.onsets}
    assert {"", "S T R", "S P L", "K W", "TH R", "B", "SH R"} <= onsets
    # onsets that begin only a few proper names are not legal medial onsets
    assert not {"T L", "D M", "M N", "N D", "K T"} & onsets
    assert all(all(p not in ("AA", "IY") for p in o) for o in english.onsets)


def test_syllabify_edge_cases(english):
    with pytest.raises(ValueError, match="no vowel"):
        syllabify(("HH", "M"), english.onsets)
    # a word-initial cluster is the onset whatever it is, and a final cluster is the coda
    syllables = syllabify(("T", "L", "AE1", "K", "S", "TH"), english.onsets)
    assert [str(s) for s in syllables] == ["T L AE1 K S TH"]
    s = syllables[0]
    assert s.onset == ("T", "L") and s.vowel == "AE1" and s.coda == ("K", "S", "TH")
    assert s.stress == 1 and s.stressed
    assert s.with_stress(0).vowel == "AE0" and not s.with_stress(0).stressed


def test_positions():
    assert [position_of(i, 3) for i in range(3)] == list(POSITIONS)
    assert position_of(0, 1) == "initial" and rime_position_of(0, 1) == "final"
    assert rime_position_of(1, 3) == "medial"


def test_dictionary_filtering(english):
    assert "hello" in english.words and "hmm" not in english.words
    assert "'bout" not in english.words
    assert english.words["record"] == (
        ("R", "AH0", "K", "AO1", "R", "D"),
        ("R", "EH1", "K", "ER0", "D"),
        ("R", "IH0", "K", "AO1", "R", "D"),
    )
    assert english.summary()["min_onset_words"] == MIN_ONSET_WORDS


def test_counts_by_position_and_stress(english):
    # every pronunciation contributed one onset and one rime per syllable
    onsets = sum(sum(c.values()) for c in english.onset_counts.values())
    rimes = sum(sum(c.values()) for c in english.rime_counts.values())
    syllables = sum(len(s) for s in english.syllables.values())
    assert onsets == rimes == syllables
    # monosyllables count as initial onsets and final rimes, so no medial or final onset comes
    # from them and every rime starts with a vowel
    for counter in english.rime_counts.values():
        for rime in counter:
            assert rime[0] in PHONEMES and rime[0] not in {p for o in english.onsets for p in o}
    stressed_initial = english.onset_counts["initial", True]
    assert stressed_initial[("K",)] > 1000 and stressed_initial[("S", "T", "R")] > 100
    assert english.rime_counts["final", True][("AE", "T")] > 50  # cat, bat, hat, ...
    assert english.rime_counts["final", False][("AH",)] > 1000  # -a endings


def test_trigram_model(english):
    assert english.phonotactic(english.words["strengths"][0])
    assert not english.phonotactic(("NG", "K", "AE1", "T"))  # no English word begins with NG
    assert english.log_probability(("NG", "K", "AE1", "T")) == -math.inf
    lp = english.log_probability(english.words["cat"][0])
    assert math.isfinite(lp) and lp < 0
    # a real word is at least as probable as an odd but legal sequence of the same length
    assert lp > english.log_probability(("ZH", "AE1", "TH"))
    # every dictionary pronunciation passes the phonotactic check
    assert all(english.phonotactic(p) for p in list(english.syllables)[:5000])


def test_edit_distance():
    assert edit_distance(("K", "AE", "T"), ("K", "AE", "T")) == 0
    assert edit_distance(("K", "AE", "T"), ("B", "AE", "T")) == 1
    assert edit_distance(("K", "AE", "T"), ("K", "AE", "T", "S")) == 1
    assert edit_distance(("K", "AE", "T"), ("AE", "T")) == 1
    assert edit_distance(("K", "AE", "T"), ("D", "AO", "G")) == 3
    assert edit_distance((), ("A", "B")) == 2


def test_neighbors_of_real_words(english):
    n = english.neighbors(english.words["cat"][0])
    assert n.distance == 0 and n.nearest == "cat"
    assert n.at_one > 50  # bat, cot, cut, at, cats, ...
    n = english.neighbors(("K", "AE1", "T", "S"))
    assert n.distance == 0 and n.nearest in ("cat's", "cats") and n.at_one >= 1


def test_neighbors_of_nonwords_match_brute_force(english):
    from semantic_world.wordforms.english import strip_stress

    cases = [("S", "T", "R", "IY", "M", "P", "AH", "K"), ("B", "L", "IY", "K", "T", "AH", "N", "Z")]
    for phones in cases:
        found = english.neighbors(phones)
        assert found.distance == 2 and found.at_one == 0
        best = min(
            (edit_distance(phones, strip_stress(p)), english.pronunciations[strip_stress(p)][0])
            for p in english.syllables
            if abs(len(p) - len(phones)) <= 2
        )
        assert best[0] == 2
        assert edit_distance(phones, strip_stress(english.words[found.nearest][0])) == 2 or any(
            edit_distance(phones, strip_stress(p)) == 2 for p in english.words[found.nearest]
        )
    # count_at_one agrees with brute force for a short nonword
    phones = ("F", "AE", "T", "S", "K")
    brute = {
        w
        for p, words in english.pronunciations.items()
        if abs(len(p) - 5) <= 1 and edit_distance(phones, p) == 1
        for w in words
    }
    assert english.count_at_one(phones) == len(brute)


def test_neighbors_with_max_distance(english):
    far = ("ZH", "AE", "ZH", "AE", "ZH", "AE", "ZH", "AE", "ZH", "AE", "ZH", "AE", "ZH")
    limited = english.neighbors(far, max_distance=2)
    assert limited.distance == 3 and limited.nearest == ""
    assert english.neighbors(far).distance > 2
