"""The readable spellings: common English spellings chosen by position in the word."""

from __future__ import annotations

import pytest
import yaml
from wordforms_support import needs_cmudict, needs_wordfreq

from semantic_world.wordforms.english import CONSONANTS, VOWELS
from semantic_world.wordforms.spelling import SPELLING_TABLE, Speller


@pytest.fixture(scope="module")
def speller():
    return Speller.load()


def spell(speller, arpabet: str) -> str:
    return speller.spell(tuple(arpabet.split()))


def test_table_covers_every_phoneme(speller):
    for phone in VOWELS:
        assert speller.vowel_entry(phone, 0)["default"]
        assert speller.vowel_entry(phone, 1)["default"]
    for phone in CONSONANTS:
        assert speller.consonant(phone)
    assert speller.vowel_entry("AH", 0)["default"] == "a"
    assert speller.vowel_entry("AH", 1)["default"] == "u"
    assert speller.vowel_entry("AH", 0)["final"] == "a"  # the stress entry overrides by context


@pytest.mark.parametrize(
    ("arpabet", "spelling"),
    [
        # the vowel of "my" by position: final, magic e, open syllable, before a vowel, elsewhere
        ("M AY1", "my"),
        ("T AY1 M", "time"),
        ("T AY1 G ER0", "tiger"),
        ("L AY1 AH0 N", "lien"),
        ("P AY1 N T", "pighnt"),
        # a magic-e ending may take a plural or past ending that agrees in voicing
        ("T AY1 M Z", "times"),
        ("L AY1 K T", "liked"),
        ("K AY1 N D", "kined"),
        ("P EY1 S T", "paced"),
        # other long vowels by position
        ("D EY1", "day"),
        ("K EY1 K", "cake"),
        ("P EY1 P ER0", "paiper"),
        ("G OW1", "go"),
        ("HH OW1 M", "home"),
        ("T R IY1", "tree"),
        ("F AH1 N IY0", "funny"),
        ("R EY1 D IY0 OW2", "raidio"),
        ("M UW1 N", "moon"),
        ("B OY1", "boy"),
        ("V OY1 S", "voice"),
        ("K AW1", "cow"),
        ("L AO1", "law"),
        # Y and UW after a consonant
        ("K Y UW1 T", "cute"),
        ("F Y UW1", "few"),
        ("M Y UW1 Z IH0 K", "muzic"),
        ("Y UW1", "yoo"),
        # vowels before R
        ("K AA1 R", "car"),
        ("S T AO1 R IY0", "story"),
        ("K EH1 R", "cair"),
        ("M EH1 R IY0", "merry"),
        ("F IH1 R", "feer"),
        ("B ER1 D", "burd"),
        ("B AH1 T ER0", "butter"),
        # consonants are doubled after a stressed short vowel
        ("HH AE1 P IY0", "happy"),
        ("K IH1 T AH0 N", "kitten"),
        ("M AA1 N AH0 T IH0 N", "monnatin"),
        ("B AE1 K", "back"),
        ("K AE1 CH", "catch"),
        ("B AE1 JH", "badge"),
        ("B EH1 L", "bell"),
        # c and k, qu, x, wh, nk
        ("K AE1 T", "cat"),
        ("K IH1 T", "kit"),
        ("K L AE1 S P", "clasp"),
        ("K W IH1 K", "quick"),
        ("B AA1 K S", "box"),
        ("EH1 K S T R AH0", "extra"),
        ("HH W EH1 N", "when"),
        ("HH AA1 K IH0 NG K", "hockink"),
        ("AH1 NG K AH0 L", "uncle"),
        # endings
        ("AE1 P AH0 L", "apple"),
        ("T AE1 K AH0 L", "tackle"),
        ("P AH1 Z AH0 L", "puzzle"),
        ("AE1 L B AH0 M", "album"),
        ("B AE1 S K AH0 T", "basket"),
        ("V IH1 L AH0 JH", "village"),
        ("AE1 K T IH0 V", "active"),
        ("D AE1 N S", "dance"),
        ("W ER1 N Z", "wurns"),
        ("S L IY1 V", "sleeve"),
        ("TH R IY1 V D", "threeved"),
        ("L IH1 V Z", "lives"),
        ("B R IY1 DH", "breethe"),
        ("T IH1 M B ER0 Z", "timbers"),
        ("K AE1 R IY0 D", "carried"),
        ("S OW1 F AH0", "soafa"),
    ],
)
def test_spellings_by_position(speller, arpabet, spelling):
    assert spell(speller, arpabet) == spelling


def test_stress_digits_are_optional(speller):
    # without stress digits every vowel is unstressed, so nothing is doubled
    assert spell(speller, "K AE T") == "cat"
    assert spell(speller, "HH AE P IY") == "hapy"


def test_spellings_are_lowercase_letters(speller):
    for arpabet in ("S T R EH1 NG K TH S", "ZH AA1 ZH", "Y AO1 DH", "AH0", "IY1"):
        result = spell(speller, arpabet)
        assert result.isalpha() and result.islower()


@needs_cmudict
@needs_wordfreq
def test_spellings_match_many_common_english_words(common_english, speller):
    """A sanity check of the table as a whole: the speller gives the dictionary spelling of a
    fair share of common words, and a nonempty alphabetic string for every word."""
    words = [w for w in sorted(common_english.pattern_words) if w.isalpha()]
    spellings = [speller.spell(common_english.words[w][0]) for w in words]
    assert all(s.isalpha() for s in spellings)
    agreement = sum(s == w for s, w in zip(spellings, words, strict=True)) / len(words)
    assert agreement > 0.2
    for word in ("cat", "happy", "time", "tiger", "cake", "kitten", "quick", "apple", "dance"):
        assert speller.spell(common_english.words[word][0]) == word


@needs_cmudict
@needs_wordfreq
def test_colliding_spellings_get_numeric_suffixes(common_english, speller):
    from semantic_world.wordforms.english import Syllable
    from semantic_world.wordforms.generate import WordForm, _add_statistics
    from semantic_world.wordforms.phonemes import load_tables

    # TH and DH are both written "th", so these three forms collide
    forms = [
        WordForm("W.1", (Syllable(("TH",), ("IH1", "V")),), real_word=False),
        WordForm("W.2", (Syllable(("DH",), ("IH1", "V")),), real_word=False),
        WordForm("W.3", (Syllable(("B",), ("IH1", "V")),), real_word=False),
        WordForm("W.4", (Syllable(("DH",), ("IH1", "V")),), real_word=False),
    ]
    _add_statistics(forms, common_english, *load_tables(), speller)
    assert [w.spelling for w in forms] == ["thiv", "thiv2", "biv", "thiv3"]


@needs_cmudict
@needs_wordfreq
def test_generated_spellings_are_unique():
    from semantic_world.wordforms.config import parse_config
    from semantic_world.wordforms.generate import generate_lexicon
    from semantic_world.wordforms.streams import Streams

    config = parse_config({"wordforms": {"count": 400, "syllables": {1: 1}}}, "x")
    lexicon = generate_lexicon(config, Streams(config.seed).generate)
    spellings = [w.spelling for w in lexicon.words]
    assert len(set(spellings)) == len(spellings)
    assert all(s.rstrip("0123456789").isalpha() for s in spellings)


def test_table_validation(tmp_path):
    good = yaml.safe_load(SPELLING_TABLE.read_text())
    bad = tmp_path / "bad.yaml"

    def load_with(change):
        data = yaml.safe_load(SPELLING_TABLE.read_text())
        change(data)
        bad.write_text(yaml.safe_dump(data, allow_unicode=True))
        return Speller.load(bad)

    assert load_with(lambda d: None).spell(("K", "AE1", "T")) == "cat"
    with pytest.raises(ValueError, match="no default spelling for ZH"):
        load_with(lambda d: d["consonants"].pop("ZH"))
    with pytest.raises(ValueError, match="no default spelling for AE"):
        load_with(lambda d: d["vowels"].pop("AE"))
    with pytest.raises(ValueError, match="unknown context"):
        load_with(lambda d: d["vowels"]["AE"].update({"medial": "a"}))
    with pytest.raises(ValueError, match="not an ARPAbet phoneme"):
        load_with(lambda d: d["consonants"].update({"AE": {"default": "a"}}))
    with pytest.raises(ValueError, match="unknown key"):
        load_with(lambda d: d.update({"extra": 1}))
    with pytest.raises(ValueError, match="expected two consonants"):
        load_with(lambda d: d["pairs"].update({"K": "c"}))
    assert set(good) >= {"vowels", "consonants", "pairs", "final_pairs"}
