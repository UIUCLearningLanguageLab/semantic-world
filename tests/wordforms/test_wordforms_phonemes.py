"""The phoneme mapping tables and the IPA check against espeak-ng."""

from __future__ import annotations

import pytest
import yaml
from wordforms_support import needs_cmudict, needs_espeak

from semantic_world.wordforms.english import PHONEMES, Syllable
from semantic_world.wordforms.phonemes import (
    ESPEAK_TABLE,
    IPA_TABLE,
    SPELLING_TABLE,
    PhonemeTable,
    ipa_agreement,
    load_tables,
    strip_ipa_stress,
)


def test_tables_cover_every_phoneme():
    for table in load_tables():
        assert set(PHONEMES) <= set(table.phonemes)
        for phone in PHONEMES:
            assert table.phone(phone)
            assert table.phone(phone + "1")


def test_stress_marks():
    ipa, espeak, spelling = load_tables()
    assert (ipa.primary, ipa.secondary) == ("ˈ", "ˌ")
    assert (espeak.primary, espeak.secondary) == ("'", ",")
    assert (spelling.primary, spelling.secondary) == ("", "")


def test_stress_specific_entries():
    ipa, espeak, spelling = load_tables()
    assert ipa.phone("AH1") == "ʌ" and ipa.phone("AH0") == "ə" and ipa.phone("AH") == "ʌ"
    assert ipa.phone("ER1") == "ɜː" and ipa.phone("ER0") == "ɚ"
    assert espeak.phone("AH0") == "@" and espeak.phone("AH1") == "V"
    assert spelling.phone("AH0") == "a" and spelling.phone("AH1") == "u"


def test_render_places_stress_before_the_vowel():
    ipa, espeak, spelling = load_tables()
    hello = (Syllable(("HH",), ("AH0",)), Syllable(("L",), ("OW1",)))
    assert ipa.render(hello) == "həlˈoʊ"
    assert espeak.render(hello) == "h@l'oU"
    assert spelling.spell(tuple(p for s in hello for p in s.phones)) == "haloe"
    athlete = (Syllable((), ("AE1", "TH")), Syllable(("L",), ("IY2", "T")))
    assert ipa.render(athlete) == "ˈæθlˌiːt"
    assert strip_ipa_stress(ipa.render(athlete)) == "æθliːt"


def test_table_validation(tmp_path):
    good = yaml.safe_load(IPA_TABLE.read_text())
    bad = tmp_path / "bad.yaml"
    missing = {"phonemes": {k: v for k, v in good["phonemes"].items() if k != "ZH"}}
    bad.write_text(yaml.safe_dump(missing, allow_unicode=True))
    with pytest.raises(ValueError, match="no entry for ZH"):
        PhonemeTable.load(bad)
    bad.write_text(
        yaml.safe_dump({"phonemes": {**good["phonemes"], "XX": "x"}}, allow_unicode=True)
    )
    with pytest.raises(ValueError, match="not an ARPAbet phoneme"):
        PhonemeTable.load(bad)
    bad.write_text(yaml.safe_dump({**good, "extra": 1}, allow_unicode=True))
    with pytest.raises(ValueError, match="unknown key"):
        PhonemeTable.load(bad)
    assert PhonemeTable.load(ESPEAK_TABLE).name == "arpabet_espeak"
    assert PhonemeTable.load(SPELLING_TABLE).name == "spelling"


@needs_cmudict
@needs_espeak
def test_ipa_agreement_with_espeak(english):
    """The acceptance test only asks for the agreement to be reported. The floor here catches a
    broken table; the stage report gives the number."""
    report = ipa_agreement(PhonemeTable.load(IPA_TABLE), english, sample_size=300, seed=0)
    assert report["sample_size"] == 300
    assert 0 <= report["segments"] <= report["lenient"] <= 1
    assert report["lenient"] > 0.3
    assert all({"word", "table", "espeak"} == set(m) for m in report["mismatches"])
