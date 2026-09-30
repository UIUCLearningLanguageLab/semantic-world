"""The English source: the CMU Pronouncing Dictionary, syllabified and counted.

The dictionary comes from the ``cmudict`` package (in the ``speech`` extra). Every pronunciation
is syllabified by maximal onset: consonants between two vowels go to the following syllable as
far as the result is a legal onset, where a legal onset is a consonant sequence that begins at
least :data:`MIN_ONSET_WORDS` dictionary words. The word count keeps the onsets of proper names
and loanwords (``tl``, ``dm``, ``mn``) from splitting words like *atlas* and *admit* wrongly.

From the syllabified dictionary this module counts, by type frequency, onsets and rimes by
syllable position and stress, and phoneme trigrams with word boundaries. It also answers the
generator's questions: is a sequence a dictionary pronunciation, and which dictionary words are
its nearest neighbors by phoneme edit distance.

Stress is read from the vowel's digit: 1 and 2 count as stressed, 0 as unstressed. Edit distances
and trigrams ignore stress. A monosyllable's onset is counted as word-initial and its rime as
word-final, since that is what they are; the generator draws them the same way.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

VOWELS = frozenset(
    ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
)
CONSONANTS = frozenset(
    (
        "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S",
        "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
    )
)  # fmt: skip
PHONEMES = tuple(sorted(VOWELS | CONSONANTS))
"""The 39 ARPAbet phonemes, sorted."""
BOUNDARY = "#"
POSITIONS = ("initial", "medial", "final")
MIN_ONSET_WORDS = 20
"""A consonant sequence is a legal medial onset when it begins at least this many words."""

WORD_PATTERN = re.compile(r"[a-z][a-z'\-]*")
PLAIN_WORD_PATTERN = re.compile(r"[a-z]+")


def base(phone: str) -> str:
    """The phoneme without its stress digit."""
    return phone.rstrip("012")


def is_vowel(phone: str) -> bool:
    return base(phone) in VOWELS


def stress_of(phone: str) -> int:
    """The stress digit of a vowel (0, 1, or 2)."""
    return int(phone[-1]) if phone[-1].isdigit() else 0


def strip_stress(phones: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(base(p) for p in phones)


@dataclass(frozen=True)
class Syllable:
    onset: tuple[str, ...]
    rime: tuple[str, ...]
    """The vowel, with its stress digit, followed by the coda."""

    @property
    def vowel(self) -> str:
        return self.rime[0]

    @property
    def coda(self) -> tuple[str, ...]:
        return self.rime[1:]

    @property
    def stress(self) -> int:
        return stress_of(self.vowel)

    @property
    def stressed(self) -> bool:
        return self.stress > 0

    @property
    def phones(self) -> tuple[str, ...]:
        return self.onset + self.rime

    def with_stress(self, stress: int) -> Syllable:
        return Syllable(self.onset, (base(self.vowel) + str(stress),) + self.coda)

    def __str__(self) -> str:
        return " ".join(self.phones)


def syllabify(phones: tuple[str, ...], onsets: frozenset[tuple[str, ...]]) -> tuple[Syllable, ...]:
    """Split a pronunciation into syllables by maximal onset, using the legal ``onsets``.

    Consonants before the first vowel form the first onset whatever they are, and consonants after
    the last vowel form the last coda. Raises ``ValueError`` for a pronunciation without a vowel.
    """
    vowel_positions = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not vowel_positions:
        raise ValueError(f"no vowel in {' '.join(phones)}")
    syllables = []
    onset_start = 0
    for k, v in enumerate(vowel_positions):
        onset = phones[onset_start:v]
        if k + 1 < len(vowel_positions):
            cluster = phones[v + 1 : vowel_positions[k + 1]]
            split = 0  # the number of cluster consonants kept in this coda
            for split in range(len(cluster) + 1):
                if cluster[split:] in onsets:
                    break
            rime = phones[v : v + 1 + split]
            onset_start = v + 1 + split
        else:
            rime = phones[v:]
        syllables.append(Syllable(onset, rime))
    return tuple(syllables)


def position_of(index: int, count: int) -> str:
    """The position label of syllable ``index`` in a word of ``count`` syllables. A monosyllable
    is ``initial``; its rime is counted as ``final`` by the caller."""
    if index == 0:
        return "initial"
    if index == count - 1:
        return "final"
    return "medial"


def rime_position_of(index: int, count: int) -> str:
    """The position used for a syllable's rime: as :func:`position_of`, except that a
    monosyllable's rime is ``final``."""
    if count == 1:
        return "final"
    return position_of(index, count)


def edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """The Levenshtein distance between two phoneme sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class Neighbors:
    """The dictionary words nearest a phoneme sequence."""

    distance: int
    """The smallest edit distance to any dictionary pronunciation."""
    nearest: str
    """The alphabetically first word at that distance."""
    at_one: int
    """The number of dictionary words at edit distance exactly 1."""


class English:
    """The syllabified, counted CMU Pronouncing Dictionary. Build it with :func:`load_english`."""

    def __init__(self, entries: dict[str, list[list[str]]]) -> None:
        self.words: dict[str, tuple[tuple[str, ...], ...]] = {}
        for word in sorted(entries):
            if WORD_PATTERN.fullmatch(word) is None:
                continue
            prons = tuple(tuple(p) for p in entries[word] if any(is_vowel(x) for x in p))
            if prons:
                self.words[word] = prons
        self.pronunciations: dict[tuple[str, ...], tuple[str, ...]] = {}
        """Stress-stripped pronunciation to the sorted words that have it."""
        by_pron: dict[tuple[str, ...], set[str]] = {}
        initial: Counter[tuple[str, ...]] = Counter()
        for word, prons in self.words.items():
            for pron in prons:
                by_pron.setdefault(strip_stress(pron), set()).add(word)
                first_vowel = next(i for i, p in enumerate(pron) if is_vowel(p))
                initial[pron[:first_vowel]] += 1
        self.pronunciations = {k: tuple(sorted(v)) for k, v in by_pron.items()}
        self.onsets: frozenset[tuple[str, ...]] = frozenset(
            o for o, n in initial.items() if n >= MIN_ONSET_WORDS
        )
        """The legal onsets: consonant sequences that begin at least MIN_ONSET_WORDS words."""
        self.syllables: dict[tuple[str, ...], tuple[Syllable, ...]] = {}
        """Every pronunciation (with stress), syllabified."""
        self.onset_counts: dict[tuple[str, bool], Counter[tuple[str, ...]]] = {}
        self.rime_counts: dict[tuple[str, bool], Counter[tuple[str, ...]]] = {}
        for position in POSITIONS:
            for stressed in (True, False):
                self.onset_counts[position, stressed] = Counter()
                self.rime_counts[position, stressed] = Counter()
        self.trigrams: Counter[tuple[str, str, str]] = Counter()
        self.contexts: Counter[tuple[str, str]] = Counter()
        self.by_syllable_count: dict[int, list[tuple[str, tuple[str, ...]]]] = {}
        """Plain alphabetic words with their first pronunciation, by syllable count."""
        for word, prons in self.words.items():
            for k, pron in enumerate(prons):
                if pron in self.syllables:
                    continue
                syllables = syllabify(pron, self.onsets)
                self.syllables[pron] = syllables
                n = len(syllables)
                for i, s in enumerate(syllables):
                    self.onset_counts[position_of(i, n), s.stressed][s.onset] += 1
                    self.rime_counts[rime_position_of(i, n), s.stressed][strip_stress(s.rime)] += 1
                padded = (BOUNDARY, BOUNDARY, *strip_stress(pron), BOUNDARY)
                for i in range(2, len(padded)):
                    self.trigrams[padded[i - 2], padded[i - 1], padded[i]] += 1
                    self.contexts[padded[i - 2], padded[i - 1]] += 1
                if k == 0 and PLAIN_WORD_PATTERN.fullmatch(word):
                    self.by_syllable_count.setdefault(n, []).append((word, pron))
        self._index = _DistanceIndex(self.pronunciations)

    # Membership and phonotactics -------------------------------------------------------------

    def is_pronunciation(self, phones: tuple[str, ...]) -> bool:
        """Whether the sequence (stress ignored) is the pronunciation of a dictionary word."""
        return strip_stress(phones) in self.pronunciations

    def words_with_pronunciation(self, phones: tuple[str, ...]) -> tuple[str, ...]:
        return self.pronunciations.get(strip_stress(phones), ())

    def phonotactic(self, phones: tuple[str, ...]) -> bool:
        """Whether every phoneme trigram of the sequence, with word boundaries, occurs in the
        dictionary."""
        padded = (BOUNDARY, BOUNDARY, *strip_stress(phones), BOUNDARY)
        return all(padded[i - 2 : i + 1] in self.trigrams for i in range(2, len(padded)))

    def log_probability(self, phones: tuple[str, ...]) -> float:
        """The log probability of the sequence under the trigram model (natural log, maximum
        likelihood, including the end boundary); ``-inf`` for an unseen trigram."""
        padded = (BOUNDARY, BOUNDARY, *strip_stress(phones), BOUNDARY)
        total = 0.0
        for i in range(2, len(padded)):
            count = self.trigrams.get(padded[i - 2 : i + 1], 0)
            if count == 0:
                return -math.inf
            total += math.log(count / self.contexts[padded[i - 2 : i]])
        return total

    # Neighbors -------------------------------------------------------------------------------

    def neighbors(self, phones: tuple[str, ...], max_distance: int | None = None) -> Neighbors:
        """The nearest dictionary words to the sequence (stress ignored). With ``max_distance``,
        the search stops early once no word can lie within it, and ``distance`` may then exceed
        the true minimum (it is reported as ``max_distance + 1``)."""
        stripped = strip_stress(phones)
        at_one = self.count_at_one(stripped)
        exact = self.pronunciations.get(stripped)
        if exact:
            return Neighbors(0, exact[0], at_one)
        if at_one:
            return Neighbors(1, self._first_at_one(stripped), at_one)
        distance, nearest = self._index.nearest(stripped, max_distance)
        return Neighbors(distance, nearest, at_one)

    def count_at_one(self, stripped: tuple[str, ...]) -> int:
        """The number of dictionary words whose pronunciation is at edit distance exactly 1."""
        words: set[str] = set()
        for variant in _variants_at_one(stripped):
            words.update(self.pronunciations.get(variant, ()))
        return len(words)

    def _first_at_one(self, stripped: tuple[str, ...]) -> str:
        words: set[str] = set()
        for variant in _variants_at_one(stripped):
            words.update(self.pronunciations.get(variant, ()))
        return min(words)

    # Summary -----------------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "words": len(self.words),
            "pronunciations": len(self.pronunciations),
            "legal_onsets": len(self.onsets),
            "trigrams": len(self.trigrams),
            "min_onset_words": MIN_ONSET_WORDS,
        }


def _variants_at_one(stripped: tuple[str, ...]):
    """Every sequence at edit distance exactly 1: substitutions, deletions, and insertions."""
    n = len(stripped)
    seen: set[tuple[str, ...]] = {stripped}
    for i in range(n):
        for p in PHONEMES:
            if p != stripped[i]:
                variant = stripped[:i] + (p,) + stripped[i + 1 :]
                if variant not in seen:
                    seen.add(variant)
                    yield variant
        if n > 1:
            variant = stripped[:i] + stripped[i + 1 :]
            if variant not in seen:
                seen.add(variant)
                yield variant
    for i in range(n + 1):
        for p in PHONEMES:
            variant = stripped[:i] + (p,) + stripped[i:]
            if variant not in seen:
                seen.add(variant)
                yield variant


class _DistanceIndex:
    """Dictionary pronunciations as integer arrays grouped by length, for vectorized edit
    distances."""

    def __init__(self, pronunciations: dict[tuple[str, ...], tuple[str, ...]]) -> None:
        self.codes = {p: i + 1 for i, p in enumerate(PHONEMES)}
        by_length: dict[int, list[tuple[str, ...]]] = {}
        for pron in pronunciations:
            by_length.setdefault(len(pron), []).append(pron)
        self.groups: dict[int, tuple[np.ndarray, list[str]]] = {}
        for length, prons in by_length.items():
            prons.sort()
            array = np.array([[self.codes[p] for p in pron] for pron in prons], dtype=np.int16)
            words = [pronunciations[pron][0] for pron in prons]
            self.groups[length] = (array, words)

    def nearest(self, stripped: tuple[str, ...], max_distance: int | None) -> tuple[int, str]:
        query = np.array([self.codes[p] for p in stripped], dtype=np.int16)
        m = len(query)
        best = math.inf if max_distance is None else max_distance + 1
        best_word = ""
        for length in sorted(self.groups, key=lambda n: (abs(n - m), n)):
            if abs(length - m) >= best:
                break
            array, words = self.groups[length]
            distances = _levenshtein_rows(query, array)
            i = int(np.argmin(distances))
            if distances[i] < best:
                best = int(distances[i])
                best_word = words[i]
        if best_word == "" and max_distance is not None:
            return max_distance + 1, ""
        return int(best), best_word


def _levenshtein_rows(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The Levenshtein distance from ``query`` to every row of ``rows`` (all of one length)."""
    n_rows, n = rows.shape
    m = len(query)
    previous = np.tile(np.arange(n + 1, dtype=np.int32), (n_rows, 1))
    current = np.empty_like(previous)
    for i in range(1, m + 1):
        current[:, 0] = i
        for j in range(1, n + 1):
            cost = (rows[:, j - 1] != query[i - 1]).astype(np.int32)
            current[:, j] = np.minimum(
                np.minimum(previous[:, j] + 1, current[:, j - 1] + 1), previous[:, j - 1] + cost
            )
        previous, current = current, previous
    return previous[:, n]


@lru_cache(maxsize=1)
def load_english() -> English:
    """Load and syllabify the CMU Pronouncing Dictionary once per process."""
    try:
        import cmudict
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the word-form pipeline needs the cmudict package; install the 'speech' extra"
        ) from error
    return English(cmudict.dict())
