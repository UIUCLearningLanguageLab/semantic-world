"""Closed-class forms: function words, affixes, and inflected forms.

Content words are an open class. Function words and affixes are a closed class: a small, fixed
set used constantly, with short, simple forms (``docs/specs/WORDFORM_PIPELINE.md``, "Closed-class
forms").

- A **function word** has one syllable of a simple shape (CV, CVC, VC, or V). Its consonant and
  its rime are drawn from the counts that a stressed monosyllabic content word uses, restricted
  to single consonants. A form is never a common English word, never a content word, passes the
  phonotactic check, and lies at least ``min_distance`` from every other function word. The
  glosses come in order of frequency, and the most frequent half get two-phoneme shapes. With
  ``source: english``, each gloss takes its English citation pronunciation instead, and the
  dictionary's other pronunciations are recorded as its weak forms.
- An **affix** is a bound form of shape C, VC, or V, with an unstressed vowel. A suffix is drawn
  from the ends of the pattern words, and a prefix from their beginnings. An affix that more than
  ``max_skipped`` of the content words cannot take is rejected. With ``source: english``, the
  glosses PLURAL, PAST, and PROGRESSIVE become the English suffixes, with English allomorphy.
- An **inflected form** is a stem joined to an affix. When the plain join fails the phonotactic
  check, a glide (``Y`` after a front vowel, ``W`` after a back or rounded vowel) goes between
  a vowel and a vowel, and otherwise an unstressed schwa goes between the two. A pair that fails
  even then, or whose form is a common English word, is skipped and reported.

Every draw comes from the ``wordforms:closed_class`` stream, so closed-class forms never change a
content word. A shape with no form left is drawn again from the other shapes, and the summary
records each time that happens.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_world.wordforms.config import ClosedClassConfig, Config
from semantic_world.wordforms.english import (
    English,
    Syllable,
    base,
    edit_distance,
    is_vowel,
    load_english,
    strip_stress,
    syllabify,
)
from semantic_world.wordforms.generate import GenerationError, Lexicon, WordForm, lexicon_distance
from semantic_world.wordforms.labels import affix_label, function_word_label, joined_label
from semantic_world.wordforms.phonemes import PhonemeTable, load_tables
from semantic_world.wordforms.spelling import Speller
from semantic_world.wordforms.streams import Streams

SCHWA = "AH0"
"""The vowel that joins a stem and an affix when the plain join is not legal."""
FRONT_VOWELS = frozenset(("IY", "IH", "EY", "EH", "AE"))
"""Vowels after which the glide is ``Y``."""
BACK_VOWELS = frozenset(("UW", "UH", "OW", "AO", "AW"))
"""Vowels after which the glide is ``W``."""
JOINS = ("none", "schwa", "glide")
SHORT_SHARE = 0.5
"""The share of the function words, the most frequent ones, that get two-phoneme shapes."""

SIBILANTS = frozenset(("S", "Z", "SH", "ZH", "CH", "JH"))
VOICELESS = frozenset(("P", "T", "K", "F", "TH", "S", "SH", "CH"))
ENGLISH_AFFIXES: dict[str, tuple[tuple[str, ...], ...]] = {
    "PLURAL": (("Z",), ("S",), ("IH0", "Z")),
    "PAST": (("D",), ("T",), ("IH0", "D")),
    "PROGRESSIVE": (("IH0", "NG"),),
}
"""The English suffixes and their allomorphs, the most general first."""


def english_allomorph(gloss: str, stem: tuple[str, ...]) -> tuple[str, ...]:
    """The English suffix for a stem: *-s* is ``IH0 Z`` after a sibilant, ``S`` after another
    voiceless consonant, and ``Z`` otherwise; *-ed* is ``IH0 D`` after ``T`` or ``D``, ``T`` after
    another voiceless consonant, and ``D`` otherwise."""
    last = base(stem[-1])
    if gloss == "PLURAL":
        if last in SIBILANTS:
            return ("IH0", "Z")
        return ("S",) if last in VOICELESS else ("Z",)
    if gloss == "PAST":
        if last in ("T", "D"):
            return ("IH0", "D")
        return ("T",) if last in VOICELESS else ("D",)
    return ENGLISH_AFFIXES[gloss][0]


@dataclass(frozen=True)
class Affix:
    """A bound form. Phonemes are ARPAbet, and the vowel is unstressed. An English affix has
    several allomorphs, and ``phones`` is the most general one."""

    label: str
    gloss: str
    position: str
    """``suffix`` or ``prefix``."""
    phones: tuple[str, ...]
    ipa: str
    allomorphs: tuple[tuple[str, ...], ...] = ()
    """Every form of an English affix; empty for a generated affix."""

    @property
    def arpabet(self) -> str:
        return " / ".join(" ".join(p) for p in self.forms)

    @property
    def forms(self) -> tuple[tuple[str, ...], ...]:
        return self.allomorphs or (self.phones,)

    @property
    def shape(self) -> str:
        return shape_of(self.phones)

    def select(self, stem: tuple[str, ...]) -> tuple[str, ...]:
        """The affix's phonemes for a stem."""
        if self.allomorphs:
            return english_allomorph(self.gloss, stem)
        return self.phones

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


def _draw_shape(
    rng: np.random.Generator, weights: dict[str, float], used_up: set[str], allowed=None
):
    """A shape drawn by weight among the shapes that are not used up (and are in ``allowed``,
    when given), or None when every such shape with a positive weight is used up."""
    remaining = [
        (s, w)
        for s, w in weights.items()
        if w > 0 and s not in used_up and (allowed is None or s in allowed)
    ]
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
    avoid: set[tuple[str, ...]] | None = None,
) -> tuple[list[WordForm], dict[str, Any]]:
    """One function word for each gloss, without statistics, and a report of the draws. The
    glosses are in order of frequency, and the first ``SHORT_SHARE`` of them (rounded up) take
    two-phoneme shapes when a two-phoneme shape has weight. ``avoid`` holds more forms, without
    stress, that a function word must not be."""
    content_forms = {w.stripped for w in content} | (avoid or set())
    pools: dict[str, list[tuple[Syllable, float]]] = {}
    candidates: dict[str, dict[str, int]] = {}
    for shape in settings.function_shapes:
        possible = function_candidates(english, shape)
        legal = [(s, w) for s, w in possible if english.phonotactic(s.phones)]
        free = [(s, w) for s, w in legal if not english.is_common_pronunciation(s.phones)]
        pools[shape] = [(s, w) for s, w in free if strip_stress(s.phones) not in content_forms]
        candidates[shape] = {
            "possible": len(possible),
            "pass_the_phonotactic_check": len(legal),
            "not_common_english_words": len(free),
            "not_content_words": len(pools[shape]),
        }
    two_phoneme = {s for s in settings.function_shapes if len(s) == 2}
    short = math.ceil(SHORT_SHARE * len(settings.glosses))
    words: list[WordForm] = []
    accepted: list[tuple[str, ...]] = []
    shapes: Counter[str] = Counter()
    redraws: Counter[str] = Counter()
    for index, gloss in enumerate(settings.glosses):
        label = function_word_label(index + 1)
        used_up: set[str] = set()
        allowed = two_phoneme if index < short and two_phoneme else None
        while True:
            shape = _draw_shape(rng, settings.function_shapes, used_up, allowed)
            if shape is None and allowed is not None:
                allowed = None  # the two-phoneme shapes are used up: any shape will do
                continue
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
        "source": "pseudo",
        "count": len(words),
        "two_phoneme_words": min(short, len(words)),
        "shapes": {s: shapes[s] for s in settings.function_shapes if shapes[s]},
        "shape_redraws": {s: redraws[s] for s in settings.function_shapes if redraws[s]},
        "candidates": candidates,
    }
    return words, report


def english_function_words(
    settings: ClosedClassConfig, english: English
) -> tuple[list[WordForm], dict[str, Any]]:
    """Each gloss's English word: its citation pronunciation (the first dictionary pronunciation
    with primary stress, or else the first), with the other pronunciations as weak forms."""
    words: list[WordForm] = []
    for index, gloss in enumerate(settings.glosses):
        prons = english.words.get(gloss)
        if prons is None:
            raise GenerationError(
                f"the function word {gloss!r} has no English pronunciation in the dictionary; "
                f"with closed_class.function_words.source english, every gloss must be an "
                f"English word"
            )
        citation = next((p for p in prons if any(x.endswith("1") for x in p)), prons[0])
        weak = tuple(" ".join(p) for p in prons if p != citation)
        words.append(
            WordForm(
                function_word_label(index + 1),
                english.syllables[citation],
                real_word=True,
                english_word=gloss,
                kind="function",
                gloss=gloss,
                weak_forms=weak,
            )
        )
    return words, {"source": "english", "count": len(words)}


# ---------------------------------------------------------------------------------------------
# Affixes and inflected forms
# ---------------------------------------------------------------------------------------------


def glide_after(vowel: str, default: str) -> str:
    """The glide that follows a vowel: ``Y`` after a front vowel, ``W`` after a back or rounded
    vowel, and ``default`` otherwise."""
    if base(vowel) in FRONT_VOWELS:
        return "Y"
    if base(vowel) in BACK_VOWELS:
        return "W"
    return default


def vowel_collision(stem: tuple[str, ...], affix: Affix) -> str | None:
    """The vowel before the join when a vowel-initial suffix follows a vowel-final stem, or a
    vowel-final prefix precedes a vowel-initial stem; None otherwise."""
    phones = affix.select(stem)
    if not phones:
        return None
    if affix.position == "prefix":
        return phones[-1] if is_vowel(phones[-1]) and is_vowel(stem[0]) else None
    return stem[-1] if is_vowel(stem[-1]) and is_vowel(phones[0]) else None


def join(
    stem: tuple[str, ...], affix: Affix, repair: str = "none", glide: str = "Y"
) -> tuple[str, ...]:
    """The phonemes of a stem with an affix: joined plainly (``none``), with a schwa between the
    two (``schwa``), or with a glide between a vowel and a vowel (``glide``, ``Y`` after a front
    vowel, ``W`` after a back or rounded vowel, and the configured ``glide`` otherwise)."""
    if repair == "schwa":
        middle: tuple[str, ...] = (SCHWA,)
    elif repair == "glide":
        vowel = vowel_collision(stem, affix)
        middle = () if vowel is None else (glide_after(vowel, glide),)
    else:
        middle = ()
    phones = affix.select(stem)
    if affix.position == "prefix":
        return phones + middle + stem
    return stem + middle + phones


def repair_join(
    english: English, stem: tuple[str, ...], affix: Affix, epenthesis: bool, glide: str = "Y"
) -> tuple[tuple[str, ...], str] | None:
    """The joined phonemes of a stem and an affix, with the repair that made them legal: the
    plain join when it passes the phonotactic check; else a glide when a vowel meets a vowel;
    else a schwa, when epenthesis is on. None when no join passes. An English affix takes its
    allomorph and is not checked."""
    phones = join(stem, affix)
    if affix.allomorphs or english.phonotactic(phones):
        return phones, "none"
    if vowel_collision(stem, affix) is not None:
        phones = join(stem, affix, "glide", glide)
        if english.phonotactic(phones):
            return phones, "glide"
    if epenthesis:
        phones = join(stem, affix, "schwa")
        if english.phonotactic(phones):
            return phones, "schwa"
    return None


def skipped_share(
    english: English,
    phones: tuple[str, ...],
    position: str,
    content,
    epenthesis: bool,
    glide: str = "Y",
) -> float:
    """The share of the content words that cannot take an affix with these phonemes."""
    if not content:
        return 0.0
    trial = Affix("", "", position, phones, "")
    failing = sum(
        repair_join(english, stem.phones, trial, epenthesis, glide) is None for stem in content
    )
    return failing / len(content)


def same_after_schwa(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether one affix is the other with the joining schwa, so that a stem plus one, repaired
    with the schwa, would sound like the stem plus the other (``L`` and ``AH0 L``)."""
    return a == (SCHWA,) + b or b == (SCHWA,) + a or a == b + (SCHWA,) or b == a + (SCHWA,)


def generate_affixes(
    settings: ClosedClassConfig,
    rng: np.random.Generator,
    english: English,
    ipa: PhonemeTable,
    content: list[WordForm] | None = None,
) -> tuple[list[Affix], dict[str, Any]]:
    """One affix for each item. Two affixes never have the same phonemes, nor the same phonemes
    once the joining schwa is added to one of them, and an affix that more than ``max_skipped``
    of the content words cannot take is rejected."""
    content = content or []
    affixes: list[Affix] = []
    used: set[tuple[str, ...]] = set()
    shapes: Counter[str] = Counter()
    redraws: Counter[str] = Counter()
    rejected: dict[str, int] = {}
    for index, item in enumerate(settings.affixes):
        label = affix_label(index + 1)
        used_up: set[str] = set()
        while True:
            shape = _draw_shape(rng, settings.affix_shapes, used_up)
            if shape is None:
                raise GenerationError(
                    f"no form is left for the affix {label} ({item.gloss}); give another shape "
                    f"weight in closed_class.affixes.shapes, raise closed_class.affixes."
                    f"max_skipped, or ask for fewer affixes"
                )
            fresh = [
                (phones, w)
                for phones, w in affix_candidates(english, shape, item.position)
                if phones not in used and not any(same_after_schwa(phones, u) for u in used)
            ]
            shares = {
                phones: skipped_share(
                    english, phones, item.position, content, settings.epenthesis, settings.glide
                )
                for phones, _ in fresh
            }
            key = f"{item.position} {shape}"
            if key not in rejected:
                rejected[key] = sum(share > settings.max_skipped for share in shares.values())
            live = [(p, w) for p, w in fresh if shares[p] <= settings.max_skipped]
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
        "source": "pseudo",
        "count": len(affixes),
        "shapes": {s: shapes[s] for s in settings.affix_shapes if shapes[s]},
        "shape_redraws": {s: redraws[s] for s in settings.affix_shapes if redraws[s]},
        "max_skipped": settings.max_skipped,
        "candidates_rejected_for_skipped_stems": rejected,
    }
    return affixes, report


def english_affixes(
    settings: ClosedClassConfig, ipa: PhonemeTable
) -> tuple[list[Affix], dict[str, Any]]:
    """The English suffixes for the glosses PLURAL, PAST, and PROGRESSIVE."""
    affixes = []
    for index, item in enumerate(settings.affixes):
        allomorphs = ENGLISH_AFFIXES.get(item.gloss)
        if allomorphs is None:
            raise GenerationError(
                f"the affix {item.gloss!r} has no English equivalent; with "
                f"closed_class.affixes.source english, the glosses are "
                f"{', '.join(ENGLISH_AFFIXES)}"
            )
        affixes.append(
            Affix(
                affix_label(index + 1),
                item.gloss,
                item.position,
                allomorphs[0],
                " / ".join("".join(ipa.phone(p) for p in form) for form in allomorphs),
                allomorphs,
            )
        )
    return affixes, {"source": "english", "count": len(affixes)}


def inflection_pairs(
    settings: ClosedClassConfig, content: list[WordForm], affixes: list[Affix]
) -> list[tuple[WordForm, Affix]]:
    """The stem and affix pairs that the inflect entries name, each once, in word order and then
    affix order."""
    by_gloss = {affix.gloss: affix for affix in affixes}
    wanted: set[tuple[str, str]] = set()
    for entry in settings.word_inflect:
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
    spelling is the stem's spelling with the affix's spelling, so that a reader sees the stem.
    An English affix takes its allomorph for the stem, no repair, and no phonotactic check: only
    the check that the form is not a common English word."""
    forms: list[tuple[WordForm, str]] = []
    skipped: list[dict[str, str]] = []

    def skip(stem: WordForm, affix: Affix, reason: str) -> None:
        skipped.append({"stem": stem.label, "affix": affix.label, "reason": reason})

    for stem, affix in pairs:
        # An English affix is regular English morphology, so its join is not checked against
        # the trigrams of the uninflected pattern words, which lack the inflectional endings.
        joined = repair_join(english, stem.phones, affix, settings.epenthesis, settings.glide)
        if joined is None:
            repairs = "the glide" if vowel_collision(stem.phones, affix) else ""
            if settings.epenthesis:
                repairs += " and the schwa" if repairs else "the schwa"
            skip(
                stem,
                affix,
                "the join fails the phonotactic check"
                + (f" with {repairs} too" if repairs else ""),
            )
            continue
        phones, repair = joined
        if english.is_common_pronunciation(phones):
            word = english.common_words_with_pronunciation(phones)[0]
            skip(stem, affix, f"the form is the common English word {word!r}")
            continue
        form = WordForm(
            joined_label(stem.label, affix.label),
            syllabify(phones, english.onsets),
            real_word=False,
            kind="inflected",
            stem=stem.label,
            affix=affix.label,
            join=repair,
        )
        # the affix's spelling covers the repair too ("-al", "-ya")
        if affix.position == "prefix":
            spelling = speller.spell(phones[: len(phones) - len(stem.phones)]) + stem.spelling
        else:
            spelling = stem.spelling + speller.spell(phones[len(stem.phones) :])
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
    english = english or load_english(
        config.wordforms.english_min_zipf, config.wordforms.exclude_inflections
    )
    tables = load_tables()
    speller = Speller.load()
    content = lexicon.content
    content_forms = [w.stripped for w in content]
    spellings = {w.spelling for w in content}

    if config.assigns_lexemes:
        # With the lexemes of a request, the function words are made after the assignment
        # (add_function_words), so that the assignment never depends on them: their forms
        # depend on their order of frequency in the corpus.
        function_words, function_report = [], {"count": 0}
    else:
        function_words, function_report = _function_words(
            config, streams, english, content, tables, speller, spellings
        )

    if settings.affix_source == "english":
        affixes, affix_report = english_affixes(settings, tables[0])
    else:
        affixes, affix_report = generate_affixes(
            settings, streams.substream("closed_class", "affixes"), english, tables[0], content
        )
    # The entries that name lexemes wait for the assignment (add_inflected).
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
            "joins": {name: sum(form.join == name for form, _ in inflected) for name in JOINS},
            "skipped": skipped,
            "skipped_as_common_words": sum("common English" in s["reason"] for s in skipped),
        },
        # forms that sound the same as another form of the run, or as a dictionary word
        "identical_forms": [labels for labels in by_form.values() if len(labels) > 1],
        "english_words": {
            word.label: english.words_with_pronunciation(word.phones)[0]
            for word in added
            if not word.real_word and english.is_pronunciation(word.phones)
        },
    }
    return lexicon


def _function_words(
    config: Config,
    streams: Streams,
    english: English,
    content: list[WordForm],
    tables: tuple[PhonemeTable, PhonemeTable],
    speller: Speller,
    spellings: set[str],
    avoid: set[tuple[str, ...]] | None = None,
) -> tuple[list[WordForm], dict[str, Any]]:
    """The function words of a configuration, with their statistics, and the report."""
    settings = config.closed_class
    if settings.function_source == "english":
        function_words, report = english_function_words(settings, english)
    else:
        function_words, report = generate_function_words(
            settings, streams.substream("closed_class", "function_words"), english, content, avoid
        )
    content_forms = [w.stripped for w in content]
    for word in function_words:
        spelling = word.english_word or speller.spell_avoiding(word.phones, english.pattern_words)
        _add_statistics(word, spelling, english, tables, content_forms, spellings)
    return function_words, report


def add_function_words(
    config: Config,
    streams: Streams,
    lexicon: Lexicon,
    avoid: set[tuple[str, ...]],
    english: English | None = None,
) -> list[WordForm]:
    """Add the function words to a lexicon whose lexemes are assigned: the step that a run with
    the lexemes of a request takes after the assignment. A function word is none of the forms in
    ``avoid`` (the marked forms, the markers, and every form that a word could have with an
    affix), beside the content words. The function words stand right after the content words,
    as in every run."""
    english = english or load_english(
        config.wordforms.english_min_zipf, config.wordforms.exclude_inflections
    )
    content = lexicon.content
    function_words, report = _function_words(
        config,
        streams,
        english,
        content,
        load_tables(),
        Speller.load(),
        {w.spelling for w in lexicon.words},
        avoid,
    )
    rest = [w for w in lexicon.words if w.kind != "content"]
    lexicon.words = content + function_words + rest
    lexicon.closed_class["function_words"] = report
    for word in function_words:
        if not word.real_word and english.is_pronunciation(word.phones):
            lexicon.closed_class["english_words"][word.label] = english.words_with_pronunciation(
                word.phones
            )[0]
    return function_words


def add_inflected(
    config: Config,
    lexicon: Lexicon,
    pairs: list[tuple[WordForm, Affix]],
    english: English | None = None,
) -> list[WordForm]:
    """Add the inflected forms of more stem and affix pairs to a lexicon that has its
    closed-class forms: the second pass of a run that assigns the lexemes of a request, whose
    stems (content words and marked forms) are known only after the assignment. Every pair must
    give a form: the assignment gave each lexeme a word that can take its affixes."""
    settings = config.closed_class
    english = english or load_english(
        config.wordforms.english_min_zipf, config.wordforms.exclude_inflections
    )
    tables = load_tables()
    speller = Speller.load()
    content_forms = [w.stripped for w in lexicon.content]
    spellings = {w.spelling for w in lexicon.words}
    inflected, skipped = inflect(settings, english, speller, pairs)
    if skipped:
        first = skipped[0]
        raise GenerationError(
            f"the inflected form of {first['stem']} with {first['affix']} cannot be made "
            f"({first['reason']}), and {len(skipped) - 1} more: the request's takes must list "
            f"every affix that an inflect entry gives a lexeme"
        )
    for form, spelling in inflected:
        _add_statistics(form, spelling, english, tables, content_forms, spellings)
    made = [form for form, _ in inflected]
    lexicon.words = lexicon.words + made
    report = lexicon.closed_class["inflected"]
    report["requested"] += len(pairs)
    report["made"] += len(made)
    for name in JOINS:
        report["joins"][name] += sum(form.join == name for form in made)
    by_form: dict[tuple[str, ...], list[str]] = {}
    for word in lexicon.words:
        by_form.setdefault(word.stripped, []).append(word.label)
    lexicon.closed_class["identical_forms"] = [
        labels for labels in by_form.values() if len(labels) > 1
    ]
    for word in made:
        if english.is_pronunciation(word.phones):
            lexicon.closed_class["english_words"][word.label] = english.words_with_pronunciation(
                word.phones
            )[0]
    return made
