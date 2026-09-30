"""Readable spellings of word forms.

A spelling is built from common English spellings of each phoneme, chosen by position in the
word: the vowel of *my* is written ``y`` at the end of a word and ``i`` with a silent ``e`` before
a final consonant, a consonant is doubled after a stressed short vowel, and so on. The spellings
live in ``data/wordforms/spelling.yaml``. This module defines the contexts that the table's
entries refer to. The spellings are for human readers only; no model is given them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_world.wordforms.english import CONSONANTS, VOWELS, base, stress_of

SPELLING_TABLE = Path(__file__).resolve().parents[3] / "data" / "wordforms" / "spelling.yaml"

VOWEL_CONTEXTS = (
    "final",
    "last",
    "magic_e",
    "before_r_closed",
    "before_r",
    "before_vowel",
    "open",
    "default",
)
CONSONANT_CONTEXTS = (
    "magic_e",
    "final_short",
    "final_long",
    "final_consonant",
    "final",
    "doubled",
    "before_velar",
    "hard",
    "default",
)
Y_UW = "YUW"
TOP_LEVEL_KEYS = (
    "short_vowels",
    "long_vowels",
    "hard_c_before",
    "syllabic_l",
    "magic_e_suffixes",
    "voiceless",
    "vowels",
    "consonants",
    "pairs",
    "final_pairs",
)


@dataclass(frozen=True)
class Speller:
    """The spelling table, with the rules that choose a spelling by position."""

    vowels: dict[str, dict[str, Any]]
    """Vowel (with or without a stress digit, or YUW) to its spellings by context."""
    consonants: dict[str, dict[str, str]]
    short_vowels: frozenset[str]
    long_vowels: frozenset[str]
    hard_c_before: str
    syllabic_l_after: frozenset[str]
    syllabic_l: str
    pairs: dict[tuple[str, str], str]
    final_pairs: dict[tuple[str, str], str]
    magic_e_suffixes: dict[str, str]
    voiceless: frozenset[str]

    @classmethod
    def load(cls, path: str | Path = SPELLING_TABLE) -> Speller:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping")
        for key in data:
            if key not in TOP_LEVEL_KEYS:
                raise ValueError(f"{path}: unknown key {key!r}")
        vowels = _entries(path, data, "vowels", VOWEL_CONTEXTS, VOWELS | {Y_UW})
        consonants = _entries(path, data, "consonants", CONSONANT_CONTEXTS, CONSONANTS)
        for name, table, phonemes in (
            ("vowels", vowels, VOWELS),
            ("consonants", consonants, CONSONANTS),
        ):
            missing = sorted(p for p in phonemes if "default" not in table.get(p, {}))
            if missing:
                raise ValueError(f"{path}: {name}: no default spelling for {', '.join(missing)}")
        syllabic = data.get("syllabic_l") or {}
        return cls(
            vowels=vowels,
            consonants=consonants,
            short_vowels=frozenset(data.get("short_vowels") or ()),
            long_vowels=frozenset(data.get("long_vowels") or ()),
            hard_c_before=str(data.get("hard_c_before") or ""),
            syllabic_l_after=frozenset(syllabic.get("after") or ()),
            syllabic_l=str(syllabic.get("spelling") or ""),
            pairs=_pairs(path, data, "pairs"),
            final_pairs=_pairs(path, data, "final_pairs"),
            magic_e_suffixes={
                str(k): str(v) for k, v in (data.get("magic_e_suffixes") or {}).items()
            },
            voiceless=frozenset(data.get("voiceless") or ()),
        )

    # Lookups ---------------------------------------------------------------------------------

    def vowel_entry(self, key: str, stress: int) -> dict[str, Any]:
        """The contexts of a vowel: the plain entry with the stress-specific entry over it."""
        entry = dict(self.vowels.get(key, {}))
        entry.update(self.vowels.get(f"{key}{stress}", {}))
        return entry

    def consonant(self, phone: str, *contexts: str) -> str:
        """The spelling of a consonant in the first of ``contexts`` that it has."""
        entry = self.consonants[phone]
        for context in contexts:
            if context in entry:
                return entry[context]
        return entry["default"]

    def magic_suffix(self, consonant: str, suffix: str) -> bool:
        """Whether ``suffix`` may follow the magic-e consonant: it is a plural or past ending
        that agrees in voicing with the consonant (times, liked, but not a second t)."""
        if suffix not in self.magic_e_suffixes or suffix == consonant:
            return False
        if "magic_e" not in self.consonants[consonant]:
            return False
        return (consonant in self.voiceless) == (suffix in self.voiceless)

    # Spelling --------------------------------------------------------------------------------

    def spell(self, phones: tuple[str, ...]) -> str:
        """The spelling of a phoneme sequence (ARPAbet, with or without stress digits)."""
        bases = [base(p) for p in phones]
        stresses = [stress_of(p) for p in phones]
        n = len(bases)
        vowel = [b in VOWELS for b in bases]
        chunks: list[str | None] = [None] * n
        magic: set[int] = set()  # consonants written with a silent e
        hard_check: set[int] = set()  # K written c or k by the next letter

        def consonant_at(i: int) -> bool:
            return 0 <= i < n and not vowel[i]

        # Vowels first: their spellings depend only on the phonemes around them.
        for i in range(n):
            if not vowel[i]:
                continue
            key = bases[i]
            if key == "UW" and i >= 2 and bases[i - 1] == "Y" and consonant_at(i - 2):
                key = Y_UW
                chunks[i - 1] = ""  # the Y is not written: cute, music
            entry = self.vowel_entry(key, stresses[i])
            last = i == n - 1
            one_final_consonant = i == n - 2 and consonant_at(i + 1)
            if last and "final" in entry:
                chunks[i] = entry["final"]
            elif (
                one_final_consonant
                and key == "AH"
                and stresses[i] == 0
                and bases[i + 1] == "L"
                and i >= 1
                and bases[i - 1] in self.syllabic_l_after
            ):
                chunks[i], chunks[i + 1] = self.syllabic_l, ""
            elif one_final_consonant and bases[i + 1] in entry.get("last", {}):
                chunks[i], chunks[i + 1] = entry["last"][bases[i + 1]], ""
            elif (
                one_final_consonant
                and "magic_e" in entry
                and "magic_e" in self.consonants[bases[i + 1]]
            ):
                chunks[i] = entry["magic_e"]
                magic.add(i + 1)
            elif (
                i == n - 3
                and consonant_at(i + 1)
                and consonant_at(i + 2)
                and "magic_e" in entry
                and self.magic_suffix(bases[i + 1], bases[i + 2])
            ):
                chunks[i] = entry["magic_e"]
                magic.add(i + 1)
                chunks[i + 2] = self.magic_e_suffixes[bases[i + 2]]
            elif (
                consonant_at(i + 1)
                and bases[i + 1] == "R"
                and (i + 2 == n or consonant_at(i + 2))
                and "before_r_closed" in entry
            ):
                chunks[i] = entry["before_r_closed"]
            elif consonant_at(i + 1) and bases[i + 1] == "R" and "before_r" in entry:
                chunks[i] = entry["before_r"]
            elif i + 1 < n and vowel[i + 1] and "before_vowel" in entry:
                chunks[i] = entry["before_vowel"]
            elif consonant_at(i + 1) and i + 2 < n and vowel[i + 2] and "open" in entry:
                chunks[i] = entry["open"]
            else:
                chunks[i] = entry["default"]

        # Consonants.
        for i in range(n):
            if chunks[i] is not None:
                continue
            phone = bases[i]
            if i in magic:
                chunks[i] = self.consonant(phone, "magic_e")
                continue
            if i + 1 < n and chunks[i + 1] is None and i + 1 not in magic:
                pair = (phone, bases[i + 1])
                if i + 1 == n - 1 and pair in self.final_pairs:
                    chunks[i], chunks[i + 1] = self.final_pairs[pair], ""
                    continue
                if pair in self.pairs and not (pair == ("K", "S") and i == 0):
                    chunks[i], chunks[i + 1] = self.pairs[pair], ""
                    continue
            after_short = (
                i >= 1
                and vowel[i - 1]
                and stresses[i - 1] > 0
                and bases[i - 1] in self.short_vowels
            )
            if i == n - 1:
                if after_short:
                    chunks[i] = self.consonant(phone, "final_short")
                elif i >= 1 and vowel[i - 1] and bases[i - 1] in self.long_vowels:
                    chunks[i] = self.consonant(phone, "final_long")
                elif i >= 1 and not vowel[i - 1]:
                    chunks[i] = self.consonant(phone, "final_consonant")
                else:
                    chunks[i] = self.consonant(phone, "final")
            elif after_short and vowel[i + 1] and "doubled" in self.consonants[phone]:
                chunks[i] = self.consonants[phone]["doubled"]
            elif bases[i + 1] in ("K", "G") and "before_velar" in self.consonants[phone]:
                chunks[i] = self.consonants[phone]["before_velar"]
            else:
                chunks[i] = self.consonants[phone]["default"]
                if "hard" in self.consonants[phone]:
                    hard_check.add(i)

        # K is written c before a, o, u, l, r, and t.
        for i in sorted(hard_check):
            following = "".join(c or "" for c in chunks[i + 1 :])
            if following and following[0] in self.hard_c_before:
                chunks[i] = self.consonants[bases[i]]["hard"]
        return "".join(c or "" for c in chunks)


def _entries(
    path: Path,
    data: dict,
    name: str,
    contexts: tuple[str, ...],
    phonemes: frozenset[str] | set[str],
) -> dict[str, dict[str, Any]]:
    table = data.get(name)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: expected a '{name}' mapping")
    result: dict[str, dict[str, Any]] = {}
    for key, entry in table.items():
        plain = key if key in phonemes else base(str(key))
        if plain not in phonemes:
            raise ValueError(f"{path}: {name}.{key}: not an ARPAbet phoneme of this kind")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {name}.{key}: expected a mapping of contexts")
        for context, spelling in entry.items():
            if context not in contexts:
                raise ValueError(f"{path}: {name}.{key}.{context}: unknown context")
            if context == "last":
                if not isinstance(spelling, dict) or not all(
                    c in CONSONANTS and isinstance(v, str) for c, v in spelling.items()
                ):
                    raise ValueError(f"{path}: {name}.{key}.last: expected consonants to strings")
            elif not isinstance(spelling, str):
                raise ValueError(f"{path}: {name}.{key}.{context}: expected a string")
        result[str(key)] = dict(entry)
    return result


def _pairs(path: Path, data: dict, name: str) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for key, spelling in (data.get(name) or {}).items():
        parts = tuple(str(key).split())
        if len(parts) != 2 or not all(p in CONSONANTS for p in parts):
            raise ValueError(f"{path}: {name}.{key}: expected two consonants")
        result[parts[0], parts[1]] = str(spelling)
    return result
