"""The phoneme mapping tables: ARPAbet to IPA, to espeak-ng mnemonics, and to a readable spelling.

The tables live in ``data/wordforms/``, not in code. Each table maps every one of the 39 ARPAbet
phonemes, and may add stress-specific entries (``AH0``) that override the plain entry for a vowel
carrying that stress. The two phonetic tables also give the primary and secondary stress marks,
which the pipeline writes before the stressed vowel, as espeak-ng does.

The IPA sanity check compares the table's IPA for a sample of dictionary words with the IPA that
espeak-ng produces from the spelled words.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from semantic_world.wordforms.english import PHONEMES, English, Syllable, base, load_english

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "wordforms"
IPA_TABLE = DATA_DIR / "arpabet_ipa.yaml"
ESPEAK_TABLE = DATA_DIR / "arpabet_espeak.yaml"
SPELLING_TABLE = DATA_DIR / "spelling.yaml"
IPA_STRESS_MARKS = "ˈˌ"


@dataclass(frozen=True)
class PhonemeTable:
    """A mapping from ARPAbet phonemes to strings in another notation."""

    name: str
    phonemes: dict[str, str]
    primary: str
    secondary: str

    @classmethod
    def load(cls, path: str | Path) -> PhonemeTable:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("phonemes"), dict):
            raise ValueError(f"{path}: expected a mapping with a 'phonemes' mapping")
        phonemes: dict[str, str] = {}
        for key, value in data["phonemes"].items():
            if not isinstance(key, str) or base(key) not in PHONEMES:
                raise ValueError(f"{path}: {key!r} is not an ARPAbet phoneme")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path}: phonemes.{key}: expected a non-empty string")
            phonemes[key] = value
        missing = [p for p in PHONEMES if p not in phonemes]
        if missing:
            raise ValueError(f"{path}: no entry for {', '.join(missing)}")
        stress = data.get("stress") or {}
        for key in data:
            if key not in ("phonemes", "stress"):
                raise ValueError(f"{path}: unknown key {key!r}")
        return cls(
            name=path.stem,
            phonemes=phonemes,
            primary=str(stress.get("primary", "")),
            secondary=str(stress.get("secondary", "")),
        )

    def phone(self, phone: str) -> str:
        """The string for one phoneme, with or without a stress digit."""
        return self.phonemes.get(phone) or self.phonemes[base(phone)]

    def render(self, syllables: tuple[Syllable, ...]) -> str:
        """The string for a syllabified word, with stress marks before stressed vowels."""
        parts = []
        for syllable in syllables:
            parts.extend(self.phone(p) for p in syllable.onset)
            if syllable.stress == 1:
                parts.append(self.primary)
            elif syllable.stress == 2:
                parts.append(self.secondary)
            parts.extend(self.phone(p) for p in syllable.rime)
        return "".join(parts)

    def spell(self, phones: tuple[str, ...]) -> str:
        """The string for a phoneme sequence, without stress marks."""
        return "".join(self.phone(p) for p in phones)


def load_tables() -> tuple[PhonemeTable, PhonemeTable, PhonemeTable]:
    """The IPA, espeak-ng, and spelling tables from ``data/wordforms/``."""
    return (
        PhonemeTable.load(IPA_TABLE),
        PhonemeTable.load(ESPEAK_TABLE),
        PhonemeTable.load(SPELLING_TABLE),
    )


# ---------------------------------------------------------------------------------------------
# The IPA sanity check against espeak-ng
# ---------------------------------------------------------------------------------------------


def espeak_path() -> str | None:
    """The espeak-ng program, or None when it is not installed."""
    return shutil.which("espeak-ng")


def espeak_ipa(words: list[str], voice: str = "en-us") -> list[str]:
    """espeak-ng's IPA for each spelled word, one call for the whole list. Blank lines separate
    the words, so that each word is its own clause and gets word stress rather than sentence
    stress."""
    program = espeak_path()
    if program is None:
        raise RuntimeError("espeak-ng is not installed (brew install espeak-ng)")
    result = subprocess.run(
        [program, "-q", "--ipa", "-v", voice, "--stdin"],
        input="\n\n".join(words) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != len(words):
        raise RuntimeError(f"espeak-ng returned {len(lines)} lines for {len(words)} words")
    return lines


def strip_ipa_stress(ipa: str) -> str:
    return "".join(c for c in ipa if c not in IPA_STRESS_MARKS)


def ipa_agreement(
    table: PhonemeTable,
    english: English | None = None,
    *,
    sample_size: int = 500,
    seed: int = 0,
    voice: str = "en-us",
) -> dict[str, Any]:
    """Compare the table's IPA with espeak-ng's for a sample of plain dictionary words with a
    single pronunciation. Stress marks are ignored in every comparison. ``segments`` is the
    proportion of words whose IPA matches exactly; ``lenient`` also ignores length marks and
    treats espeak-ng's flap (ɾ) and reduced vowels (ɐ, ᵻ) as t, ə, and ɪ. Disagreement has two
    sources: the table's conventions, and espeak-ng pronouncing a spelled word differently from
    the dictionary. ``mismatches`` lists the first differing words."""
    english = english or load_english()
    rng = np.random.default_rng(seed)
    candidates = sorted(w for w, prons in english.words.items() if len(prons) == 1 and w.isalpha())
    indices = rng.choice(len(candidates), size=min(sample_size, len(candidates)), replace=False)
    words = [candidates[i] for i in sorted(indices)]
    ours = [strip_ipa_stress(table.render(english.syllables[english.words[w][0]])) for w in words]
    theirs = [strip_ipa_stress(s) for s in espeak_ipa(words, voice)]
    exact = sum(a == b for a, b in zip(ours, theirs, strict=True))
    lenient = sum(_lenient(a) == _lenient(b) for a, b in zip(ours, theirs, strict=True))
    differing = [(w, a, b) for w, a, b in zip(words, ours, theirs, strict=True) if a != b]
    return {
        "sample_size": len(words),
        "segments": exact / len(words),
        "lenient": lenient / len(words),
        "mismatches": [{"word": w, "table": a, "espeak": b} for w, a, b in differing[:20]],
    }


def _lenient(ipa: str) -> str:
    return ipa.replace("ː", "").replace("ɾ", "t").replace("ɐ", "ə").replace("ᵻ", "ɪ")
