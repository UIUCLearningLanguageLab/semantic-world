"""Closed-class forms: function words, affixes, and inflected forms.

Content words are an open class. Function words and affixes are a closed class: a small, fixed
set used constantly, with short, simple forms (``docs/specs/WORDFORM_PIPELINE.md``, "Closed-class
forms").

- A **function word** has one syllable of a simple shape (CV, CVC, VC, or V). Its consonant and
  its rime are drawn from the counts that a stressed monosyllabic content word uses, restricted
  to single consonants. A form is never a dictionary word, never a content word, passes the
  phonotactic check, and lies at least ``min_distance`` from every other function word.
- An **affix** is a bound form of shape C, VC, or V, with an unstressed vowel. A suffix is drawn
  from the ends of the pattern words, and a prefix from their beginnings.
- An **inflected form** is a stem joined to an affix. When the plain join fails the phonotactic
  check, an unstressed schwa goes between the two. A pair that fails even then is skipped and
  reported.

Every draw comes from the ``wordforms:closed_class`` stream, so closed-class forms never change a
content word. A shape with no form left is drawn again from the other shapes, and the summary
records each time that happens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.wordforms.config import ClosedClassConfig, Config
from semantic_world.wordforms.english import (
    English,
    Syllable,
    edit_distance,
    is_vowel,
    load_english,
    strip_stress,
    syllabify,
)
from semantic_world.wordforms.generate import GenerationError, Lexicon, WordForm, lexicon_distance
from semantic_world.wordforms.phonemes import PhonemeTable, load_tables
from semantic_world.wordforms.spelling import Speller
from semantic_world.wordforms.streams import Streams

SCHWA = "AH0"
"""The vowel that joins a stem and an affix when the plain join is not legal."""


@dataclass(frozen=True)
class Affix:
    """A bound form. Phonemes are ARPAbet, and the vowel is unstressed."""

    label: str
    gloss: str
    position: str
    """``suffix`` or ``prefix``."""
    phones: tuple[str, ...]
    ipa: str

    @property
    def arpabet(self) -> str:
        return " ".join(self.phones)

    @property
    def shape(self) -> str:
        return shape_of(self.phones)

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "gloss": self.gloss,
            "position": self.position,
            "arpabet": self.arpabet,
            "ipa": self.ipa,
        }


def shape_of(phones: tuple[str, ...]) -> str:
    """The shape of a phoneme sequence, such as ``CVC``."""
    return "".join("V" if is_vowel(p) else "C" for p in phones)


# ---------------------------------------------------------------------------------------------
# Candidates and draws
# ---------------------------------------------------------------------------------------------


def function_candidates(english: English, shape: str) -> list[tuple[Syllable, float]]:
    """Every syllable of a function-word shape with its weight: the probability of its onset
    among the single-consonant onsets, times the probability of its rime among the rimes of the
    shape. The counts are those of a stressed monosyllable, as for content words."""
    closed = shape.endswith("VC")
    rimes = _probabilities(
        {
            rime: n
            for rime, n in english.rime_counts["final", True].items()
            if len(rime) == (2 if closed else 1)
        }
    )
    if shape.startswith("C"):
        onsets = _probabilities(
            {o: n for o, n in english.onset_counts["initial", True].items() if len(o) == 1}
        )
    else:
        onsets = [((), 1.0)]
    return [
        (Syllable(onset, (rime[0] + "1",) + rime[1:]), p * q)
        for onset, p in onsets
        for rime, q in rimes
    ]


def affix_candidates(
    english: English, shape: str, position: str
) -> list[tuple[tuple[str, ...], float]]:
    """Every affix of a shape with its weight. A suffix comes from the ends of the pattern words:
    the last consonant of a word (C), or an unstressed final rime (VC, V). A prefix comes from
    their beginnings in the same way."""
    where = "final" if position == "suffix" else "initial"
    counts: Counter[tuple[str, ...]] = Counter()
    if shape == "C":
        for stressed in (True, False):
            if position == "suffix":
                for rime, n in english.rime_counts[where, stressed].items():
                    if len(rime) > 1:
                        counts[rime[-1:]] += n
            else:
                for onset, n in english.onset_counts[where, stressed].items():
                    if onset:
                        counts[onset[:1]] += n
    else:
        length = 2 if shape == "VC" else 1
        for rime, n in english.rime_counts[where, False].items():
            if len(rime) == length:
                counts[(rime[0] + "0",) + rime[1:]] += n
    return _probabilities(counts)


def _probabilities(counts) -> list[tuple[Any, float]]:
    total = sum(counts.values())
    return [(item, counts[item] / total) for item in sorted(counts)]


def _draw(rng: np.random.Generator, items: list[tuple[Any, float]]) -> Any:
    weights = np.array([w for _, w in items], dtype=np.float64)
    return items[int(rng.choice(len(items), p=weights / weights.sum()))][0]


def _draw_shape(rng: np.random.Generator, weights: dict[str, float], used_up: set[str]):
    """A shape drawn by weight among the shapes that are not used up, or None when every shape
    with a positive weight is used up."""
    remaining = [(s, w) for s, w in weights.items() if w > 0 and s not in used_up]
    return _draw(rng, remaining) if remaining else None


def _unique(spelling: str, used: set[str]) -> str:
    """The spelling, with a numeric suffix when an earlier form has it already."""
    candidate, n = spelling, 1
    while candidate in used:
        n += 1
        candidate = f"{spelling}{n}"
    used.add(candidate)
    return candidate


def _add_statistics(
    form: WordForm,
    spelling: str,
    english: English,
    tables: tuple[PhonemeTable, PhonemeTable],
    content: list[tuple[str, ...]],
    spellings: set[str],
) -> None:
    """The statistics of a closed-class form. Its lexicon neighbors are the content words at
    edit distance 1."""
    ipa_table, espeak_table = tables
    form.ipa = ipa_table.render(form.syllables)
    form.espeak = espeak_table.render(form.syllables)
    form.spelling = _unique(spelling, spellings)
    form.log_probability = english.log_probability(form.phones)
    neighbors = english.neighbors(form.phones)
    form.english_neighbors = neighbors.at_one
    form.nearest_english = neighbors.nearest
    own = form.stripped
    form.lexicon_neighbors = sum(
        1 for other in content if abs(len(other) - len(own)) <= 1 and edit_distance(own, other) == 1
    )


# ---------------------------------------------------------------------------------------------
# Function words
# ---------------------------------------------------------------------------------------------


def generate_function_words(
    settings: ClosedClassConfig,
    rng: np.random.Generator,
    english: English,
    content: list[WordForm],
) -> tuple[list[WordForm], dict[str, Any]]:
    """One function word for each gloss, without statistics, and a report of the draws."""
    content_forms = {w.stripped for w in content}
    pools: dict[str, list[tuple[Syllable, float]]] = {}
    candidates: dict[str, dict[str, int]] = {}
    for shape in settings.function_shapes:
        possible = function_candidates(english, shape)
        legal = [(s, w) for s, w in possible if english.phonotactic(s.phones)]
        free = [(s, w) for s, w in legal if not english.is_pronunciation(s.phones)]
        pools[shape] = [(s, w) for s, w in free if strip_stress(s.phones) not in content_forms]
        candidates[shape] = {
            "possible": len(possible),
            "pass_the_phonotactic_check": len(legal),
            "not_english_words": len(free),
            "not_content_words": len(pools[shape]),
        }
    words: list[WordForm] = []
    accepted: list[tuple[str, ...]] = []
    shapes: Counter[str] = Counter()
    redraws: Counter[str] = Counter()
    for index, gloss in enumerate(settings.glosses):
        label = f"F.{index + 1}"
        used_up: set[str] = set()
        while True:
            shape = _draw_shape(rng, settings.function_shapes, used_up)
            if shape is None:
                raise GenerationError(
                    f"no form is left for the function word {label} ({gloss}): every shape is "
                    f"used up; lower closed_class.function_words.min_distance, give the shape "
                    f"CVC more weight, or ask for fewer function words"
                )
            live = [
                (s, w)
                for s, w in pools[shape]
                if lexicon_distance(strip_stress(s.phones), accepted, settings.min_distance)
                >= settings.min_distance
            ]
            if live:
                break
            used_up.add(shape)
            redraws[shape] += 1
        syllable = _draw(rng, live)
        words.append(WordForm(label, (syllable,), real_word=False, kind="function", gloss=gloss))
        accepted.append(strip_stress(syllable.phones))
        shapes[shape] += 1
    report = {
        "count": len(words),
        "shapes": {s: shapes[s] for s in settings.function_shapes if shapes[s]},
        "shape_redraws": {s: redraws[s] for s in settings.function_shapes if redraws[s]},
        "candidates": candidates,
    }
    return words, report


# ---------------------------------------------------------------------------------------------
# Affixes and inflected forms
# ---------------------------------------------------------------------------------------------


def generate_affixes(
    settings: ClosedClassConfig, rng: np.random.Generator, english: English, ipa: PhonemeTable
) -> tuple[list[Affix], dict[str, Any]]:
    """One affix for each item. Two affixes never have the same phonemes."""
    affixes: list[Affix] = []
    used: set[tuple[str, ...]] = set()
    shapes: Counter[str] = Counter()
    redraws: Counter[str] = Counter()
    for index, item in enumerate(settings.affixes):
        label = f"AF.{index + 1}"
        used_up: set[str] = set()
        while True:
            shape = _draw_shape(rng, settings.affix_shapes, used_up)
            if shape is None:
                raise GenerationError(
                    f"no form is left for the affix {label} ({item.gloss}); give another shape "
                    f"weight in closed_class.affixes.shapes, or ask for fewer affixes"
                )
            live = [
                (phones, w)
                for phones, w in affix_candidates(english, shape, item.position)
                if phones not in used
            ]
            if live:
                break
            used_up.add(shape)
            redraws[shape] += 1
        phones = _draw(rng, live)
        used.add(phones)
        shapes[shape] += 1
        affixes.append(
            Affix(label, item.gloss, item.position, phones, "".join(ipa.phone(p) for p in phones))
        )
    report = {
        "count": len(affixes),
        "shapes": {s: shapes[s] for s in settings.affix_shapes if shapes[s]},
        "shape_redraws": {s: redraws[s] for s in settings.affix_shapes if redraws[s]},
    }
    return affixes, report


def join(stem: tuple[str, ...], affix: Affix, schwa: bool = False) -> tuple[str, ...]:
    """The phonemes of a stem with an affix, with or without a schwa between the two."""
    middle = (SCHWA,) if schwa else ()
    if affix.position == "prefix":
        return affix.phones + middle + stem
    return stem + middle + affix.phones


def inflection_pairs(
    settings: ClosedClassConfig, content: list[WordForm], affixes: list[Affix]
) -> list[tuple[WordForm, Affix]]:
    """The stem and affix pairs that the inflect entries name, each once, in word order and then
    affix order."""
    by_gloss = {affix.gloss: affix for affix in affixes}
    wanted: set[tuple[str, str]] = set()
    for entry in settings.inflect:
        if entry.words == "none":
            continue
        labels = [w.label for w in content] if entry.words == "all" else entry.words
        for label in labels:
            for gloss in entry.affixes:
                wanted.add((label, by_gloss[gloss].label))
    return [(w, a) for w in content for a in affixes if (w.label, a.label) in wanted]


def inflect(
    settings: ClosedClassConfig,
    english: English,
    speller: Speller,
    pairs: list[tuple[WordForm, Affix]],
) -> tuple[list[tuple[WordForm, str]], list[dict[str, str]]]:
    """The inflected form of each pair, with its spelling, and the pairs that were skipped. The
    spelling is the stem's spelling with the affix's spelling, so that a reader sees the stem."""
    forms: list[tuple[WordForm, str]] = []
    skipped: list[dict[str, str]] = []
    for stem, affix in pairs:
        phones = join(stem.phones, affix)
        epenthesis = False
        if not english.phonotactic(phones):
            if not settings.epenthesis:
                skipped.append(
                    {
                        "stem": stem.label,
                        "affix": affix.label,
                        "reason": "the plain join fails the phonotactic check",
                    }
                )
                continue
            phones = join(stem.phones, affix, schwa=True)
            epenthesis = True
            if not english.phonotactic(phones):
                skipped.append(
                    {
                        "stem": stem.label,
                        "affix": affix.label,
                        "reason": "the join fails the phonotactic check with the schwa too",
                    }
                )
                continue
        form = WordForm(
            f"{stem.label}.{affix.label}",
            syllabify(phones, english.onsets),
            real_word=False,
            kind="inflected",
            stem=stem.label,
            affix=affix.label,
            epenthesis=epenthesis,
        )
        middle = (SCHWA,) if epenthesis else ()
        if affix.position == "prefix":
            spelling = speller.spell(affix.phones + middle) + stem.spelling
        else:
            spelling = stem.spelling + speller.spell(middle + affix.phones)
        forms.append((form, spelling))
    return forms, skipped


# ---------------------------------------------------------------------------------------------
# The whole closed class
# ---------------------------------------------------------------------------------------------


def add_closed_class(
    config: Config, streams: Streams, lexicon: Lexicon, english: English | None = None
) -> Lexicon:
    """Add the closed-class forms of a configuration to a lexicon of content words: the function
    words and the inflected forms join ``lexicon.words``, after the content words, and the
    affixes go in ``lexicon.affixes``. The content words are not changed."""
    settings = config.closed_class
    if settings is None:
        return lexicon
    english = english or load_english(config.wordforms.english_min_zipf)
    tables = load_tables()
    speller = Speller.load()
    content = lexicon.content
    content_forms = [w.stripped for w in content]
    spellings = {w.spelling for w in content}

    function_words, function_report = generate_function_words(
        settings, streams.substream("closed_class", "function_words"), english, content
    )
    for word in function_words:
        spelling = speller.spell_avoiding(word.phones, english.pattern_words)
        _add_statistics(word, spelling, english, tables, content_forms, spellings)

    affixes, affix_report = generate_affixes(
        settings, streams.substream("closed_class", "affixes"), english, tables[0]
    )
    pairs = inflection_pairs(settings, content, affixes)
    inflected, skipped = inflect(settings, english, speller, pairs)
    for form, spelling in inflected:
        _add_statistics(form, spelling, english, tables, content_forms, spellings)

    added = function_words + [form for form, _ in inflected]
    lexicon.words = content + added
    lexicon.affixes = affixes
    by_form: dict[tuple[str, ...], list[str]] = {}
    for word in lexicon.words:
        by_form.setdefault(word.stripped, []).append(word.label)
    lexicon.closed_class = {
        "request": settings.request,
        "function_words": function_report,
        "affixes": affix_report,
        "inflected": {
            "requested": len(pairs),
            "made": len(inflected),
            "with_schwa": sum(bool(form.epenthesis) for form, _ in inflected),
            "skipped": skipped,
        },
        # forms that sound the same as another form of the run, or as a dictionary word
        "identical_forms": [labels for labels in by_form.values() if len(labels) > 1],
        "english_words": {
            word.label: english.words_with_pronunciation(word.phones)[0]
            for word in added
            if english.is_pronunciation(word.phones)
        },
    }
    return lexicon
