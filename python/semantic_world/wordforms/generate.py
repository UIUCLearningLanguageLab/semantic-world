"""Generating word forms: pseudowords, real English words, filters, statistics, and spelling.

A pseudoword is built syllable by syllable (``docs/specs/WORDFORM_PIPELINE.md``, layer 1): draw
the syllable count, draw the stress pattern, and draw an onset and a rime for each syllable from
the dictionary counts for that syllable's position and stress. Candidates are rejected by the
phonotactic trigram check, by closeness to real words, and by closeness to accepted forms.

Real words (``source: english`` or ``mixed``) are drawn uniformly, by syllable count, from the
plain alphabetic dictionary words, using each word's first pronunciation.

All draws come from the ``wordforms:generate`` stream, in a fixed order, so the same seed gives
the same word table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.wordforms.config import Config, WordformsConfig
from semantic_world.wordforms.english import (
    English,
    Syllable,
    edit_distance,
    load_english,
    position_of,
    rime_position_of,
    strip_stress,
)
from semantic_world.wordforms.phonemes import PhonemeTable, load_tables

MAX_REJECTIONS = 10_000
"""Consecutive rejected candidates before generation gives up."""


class GenerationError(RuntimeError):
    """The generator could not satisfy the configuration. The message says what was rejected
    and which parameters to loosen."""


@dataclass
class WordForm:
    """One word form with its statistics. Phonemes are ARPAbet with stress digits."""

    label: str
    syllables: tuple[Syllable, ...]
    real_word: bool
    english_word: str | None = None
    """The dictionary spelling of a real word, used as its spelling."""
    ipa: str = ""
    espeak: str = ""
    spelling: str = ""
    log_probability: float = 0.0
    english_neighbors: int = 0
    nearest_english: str = ""
    lexicon_neighbors: int = 0

    @property
    def phones(self) -> tuple[str, ...]:
        return tuple(p for s in self.syllables for p in s.phones)

    @property
    def stripped(self) -> tuple[str, ...]:
        return strip_stress(self.phones)

    @property
    def arpabet(self) -> str:
        return " ".join(self.phones)

    @property
    def syllable_count(self) -> int:
        return len(self.syllables)

    @property
    def stress(self) -> str:
        """The stress digit of each syllable, for example ``100``."""
        return "".join(str(s.stress) for s in self.syllables)

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "arpabet": self.arpabet,
            "ipa": self.ipa,
            "espeak": self.espeak,
            "spelling": self.spelling,
            "syllables": self.syllable_count,
            "stress": self.stress,
            "log_probability": self.log_probability,
            "english_neighbors": self.english_neighbors,
            "nearest_english": self.nearest_english,
            "lexicon_neighbors": self.lexicon_neighbors,
            "real_word": self.real_word,
        }


@dataclass
class Rejections:
    """Why candidates were rejected, for the summary."""

    phonotactic: int = 0
    english: int = 0
    lexicon: int = 0
    duplicate_real: int = 0

    def total(self) -> int:
        return self.phonotactic + self.english + self.lexicon + self.duplicate_real

    def as_dict(self) -> dict[str, int]:
        return {
            "phonotactic": self.phonotactic,
            "english": self.english,
            "lexicon": self.lexicon,
            "duplicate_real": self.duplicate_real,
            "total": self.total(),
        }


@dataclass
class Lexicon:
    """The generated word forms with the summary of their generation."""

    words: list[WordForm]
    rejections: Rejections
    english_summary: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        counts: dict[int, int] = {}
        for w in self.words:
            counts[w.syllable_count] = counts.get(w.syllable_count, 0) + 1
        return {
            "words": len(self.words),
            "real_words": sum(w.real_word for w in self.words),
            "syllable_counts": {k: counts[k] for k in sorted(counts)},
            "minimal_pairs": sum(w.lexicon_neighbors for w in self.words) // 2,
            "rejections": self.rejections.as_dict(),
            "english": self.english_summary,
        }


class _Sampler:
    """Weighted draws of onsets and rimes for each position and stress."""

    def __init__(self, english: English) -> None:
        self.onsets: dict[tuple[str, bool], tuple[list[tuple[str, ...]], np.ndarray]] = {}
        self.rimes: dict[tuple[str, bool], tuple[list[tuple[str, ...]], np.ndarray]] = {}
        for key, counter in english.onset_counts.items():
            self.onsets[key] = _weights(counter)
        for key, counter in english.rime_counts.items():
            self.rimes[key] = _weights(counter)

    def onset(self, rng: np.random.Generator, position: str, stressed: bool) -> tuple[str, ...]:
        items, probabilities = self.onsets[position, stressed]
        return items[int(rng.choice(len(items), p=probabilities))]

    def rime(self, rng: np.random.Generator, position: str, stressed: bool) -> tuple[str, ...]:
        items, probabilities = self.rimes[position, stressed]
        return items[int(rng.choice(len(items), p=probabilities))]


def _weights(counter) -> tuple[list[tuple[str, ...]], np.ndarray]:
    items = sorted(counter)
    if not items:
        raise GenerationError("the dictionary has no counts for a syllable position")
    weights = np.array([counter[i] for i in items], dtype=np.float64)
    return items, weights / weights.sum()


def draw_syllable_count(rng: np.random.Generator, syllables: dict[int, float]) -> int:
    counts = list(syllables)
    probabilities = np.array([syllables[c] for c in counts], dtype=np.float64)
    return counts[int(rng.choice(len(counts), p=probabilities))]


def draw_stress(rng: np.random.Generator, count: int, initial_probability: float) -> int:
    """The index of the stressed syllable."""
    if count == 1 or rng.random() < initial_probability:
        return 0
    return int(rng.integers(1, count))


def draw_pseudoword(
    rng: np.random.Generator, sampler: _Sampler, count: int, stressed_index: int
) -> tuple[Syllable, ...]:
    """One candidate pseudoword of ``count`` syllables with stress on ``stressed_index``:
    syllables with stress digits 1 (stressed) and 0 (unstressed)."""
    syllables = []
    for i in range(count):
        stressed = i == stressed_index
        onset = sampler.onset(rng, position_of(i, count), stressed)
        rime = sampler.rime(rng, rime_position_of(i, count), stressed)
        vowel = rime[0] + ("1" if stressed else "0")
        syllables.append(Syllable(onset, (vowel,) + rime[1:]))
    return tuple(syllables)


def draw_english_word(
    rng: np.random.Generator, english: English, count: int
) -> tuple[str, tuple[Syllable, ...]]:
    """One real word, drawn uniformly among the plain words with ``count`` syllables."""
    candidates = english.by_syllable_count.get(count)
    if not candidates:
        raise GenerationError(f"the dictionary has no plain words with {count} syllables")
    word, pron = candidates[int(rng.integers(len(candidates)))]
    return word, english.syllables[pron]


def lexicon_distance(stripped: tuple[str, ...], accepted: list[tuple[str, ...]], limit: int) -> int:
    """The smallest edit distance from ``stripped`` to any accepted form, or ``limit`` when
    every accepted form is at least ``limit`` away."""
    best = limit
    for other in accepted:
        if abs(len(other) - len(stripped)) >= best:
            continue
        d = edit_distance(stripped, other)
        if d < best:
            best = d
            if best == 0:
                break
    return best


def generate_lexicon(
    config: Config, rng: np.random.Generator, english: English | None = None
) -> Lexicon:
    """Generate the word table of a configuration from the ``wordforms:generate`` stream."""
    english = english or load_english()
    settings = config.wordforms
    sampler = _Sampler(english)
    ipa_table, espeak_table, spelling_table = load_tables()

    real_slots = _real_word_slots(rng, settings)
    words: list[WordForm] = []
    accepted: list[tuple[str, ...]] = []
    rejections = Rejections()
    for index in range(settings.count):
        label = f"W.{index + 1}"
        # The syllable count and the stress pattern are drawn once per word, so that rejections
        # do not skew their distributions.
        count = draw_syllable_count(rng, settings.syllables)
        stressed_index = draw_stress(rng, count, settings.initial_stress_probability)
        consecutive = 0
        while True:
            if consecutive >= MAX_REJECTIONS:
                raise GenerationError(
                    f"gave up on {label} after {MAX_REJECTIONS} consecutive rejections "
                    f"({rejections.as_dict()}) for a word of {count} syllables; lower "
                    f"min_lexicon_distance or min_english_distance, give longer words more "
                    f"weight in syllables, or reduce count"
                )
            if index in real_slots:
                english_word, syllables = draw_english_word(rng, english, count)
                stripped = strip_stress(tuple(p for s in syllables for p in s.phones))
                if lexicon_distance(stripped, accepted, settings.min_lexicon_distance) < (
                    settings.min_lexicon_distance
                ):
                    rejections.duplicate_real += 1
                    consecutive += 1
                    continue
                form = WordForm(label, syllables, real_word=True, english_word=english_word)
            else:
                syllables = draw_pseudoword(rng, sampler, count, stressed_index)
                phones = tuple(p for s in syllables for p in s.phones)
                stripped = strip_stress(phones)
                if not english.phonotactic(phones):
                    rejections.phonotactic += 1
                    consecutive += 1
                    continue
                if settings.exclude_real_words and _too_close_to_english(
                    english, stripped, settings.min_english_distance
                ):
                    rejections.english += 1
                    consecutive += 1
                    continue
                if lexicon_distance(stripped, accepted, settings.min_lexicon_distance) < (
                    settings.min_lexicon_distance
                ):
                    rejections.lexicon += 1
                    consecutive += 1
                    continue
                form = WordForm(label, syllables, real_word=False)
            break
        words.append(form)
        accepted.append(stripped)

    _add_statistics(words, english, ipa_table, espeak_table, spelling_table)
    return Lexicon(words, rejections, english.summary())


def _real_word_slots(rng: np.random.Generator, settings: WordformsConfig) -> set[int]:
    """Which word indices are real English words: a seeded choice of ``english_count`` slots."""
    n = settings.english_count
    if n == 0:
        return set()
    if n >= settings.count:
        return set(range(settings.count))
    return {int(i) for i in rng.choice(settings.count, size=n, replace=False)}


def _too_close_to_english(english: English, stripped: tuple[str, ...], min_distance: int) -> bool:
    if english.is_pronunciation(stripped):
        return True
    if min_distance == 1:
        return False
    if min_distance == 2:
        return english.count_at_one(stripped) > 0
    return english.neighbors(stripped, max_distance=min_distance - 1).distance < min_distance


def _add_statistics(
    words: list[WordForm],
    english: English,
    ipa_table: PhonemeTable,
    espeak_table: PhonemeTable,
    spelling_table: PhonemeTable,
) -> None:
    spellings: dict[str, int] = {}
    stripped = [w.stripped for w in words]
    for i, word in enumerate(words):
        word.ipa = ipa_table.render(word.syllables)
        word.espeak = espeak_table.render(word.syllables)
        spelling = word.english_word or spelling_table.spell(word.phones)
        seen = spellings.get(spelling, 0)
        spellings[spelling] = seen + 1
        word.spelling = spelling if seen == 0 else f"{spelling}{seen + 1}"
        word.log_probability = english.log_probability(word.phones)
        neighbors = english.neighbors(word.phones)
        word.english_neighbors = neighbors.at_one
        word.nearest_english = neighbors.nearest
        word.lexicon_neighbors = sum(
            1
            for j, other in enumerate(stripped)
            if j != i
            and abs(len(other) - len(stripped[i])) <= 1
            and edit_distance(stripped[i], other) == 1
        )
