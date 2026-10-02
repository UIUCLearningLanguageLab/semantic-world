"""Sound-meaning assignment: which word form goes with which meaning.

The meanings are any table of IDs with binary feature vectors, such as the taxonomy generator's
``categories_generative.csv``: a CSV file whose first column is the ID and whose other columns
hold 0 and 1. A column that holds other numbers (a scalar dimension) is dropped and reported,
or is an error, as configured. The assignment modes are:

- ``arbitrary``: a seeded random assignment from the ``wordforms:assign`` stream.
- ``target_correlation``: the assignment starts random, and pairs of words are swapped while
  each swap moves the sound-meaning correlation toward a configured target. The result is
  within a tolerance of the target, or the closest value reached, and the summary says which.
- ``branch_markers``: the words of the meanings in one branch of the taxonomy share a marker
  syllable, at a configured depth and position, so the systematicity is morphological. A marked
  form is a new word form (kind ``marked``), joined like an inflected form.
- ``acoustic_mapping``: an arbitrary assignment, whose semantic features then shift acoustic
  properties of every token of a word (:mod:`semantic_world.wordforms.mapping`).

Every assignment reports the sound-meaning correlation: the correlation, across pairs of
meanings, between the sound distance of their words and the distance of their feature vectors.
The sound distance is the phoneme edit distance or the cosine distance of a named embedding, and
the meaning distance is Hamming, cosine, or Jaccard. A null distribution comes from random
reassignments of the same words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.wordforms.generate import WordForm

NULL_SAMPLES = 1000
"""The number of random reassignments in the null distribution."""
MARKER_TRIES = 200
"""How many candidate markers a branch tries before the assignment gives up."""


class AssignmentError(ValueError):
    """The assignment cannot be made, for example because there are more meanings than words."""


# ---------------------------------------------------------------------------------------------
# Meanings
# ---------------------------------------------------------------------------------------------


@dataclass
class Meanings:
    """A meanings table: IDs, binary features (meanings by features), and the feature names."""

    ids: list[str]
    features: np.ndarray
    names: list[str]
    dropped: list[str] = field(default_factory=list)
    """The columns that were dropped because they hold other values than 0 and 1."""

    def feature(self, name: str) -> np.ndarray:
        if name not in self.names:
            raise AssignmentError(f"the meanings table has no feature column {name!r}")
        return self.features[:, self.names.index(name)]

    def select(self, categories) -> Meanings:
        """The meanings that receive words: ``all``, the ``leaves`` (IDs that no other ID
        extends with a period), or the listed IDs."""
        if categories == "all":
            return self
        if categories == "leaves":
            keep = [
                i
                for i, x in enumerate(self.ids)
                if not any(y.startswith(x + ".") for y in self.ids)
            ]
        else:
            missing = [c for c in categories if c not in self.ids]
            if missing:
                raise AssignmentError(f"the meanings table has no ID {missing[0]!r}")
            keep = [self.ids.index(c) for c in categories]
        return Meanings([self.ids[i] for i in keep], self.features[keep], self.names, self.dropped)


def load_meaning_table(path: str | Path, non_binary: str = "error") -> Meanings:
    """A meanings table from a CSV file. ``non_binary`` says what to do with a column that holds
    other values than 0 and 1: ``drop`` it, or raise an error."""
    table = pl.read_csv(path, infer_schema_length=None)
    if table.width < 2 or table.height == 0:
        raise AssignmentError(f"{path}: expected an ID column and at least one feature column")
    ids = [str(value) for value in table.to_series(0)]
    if len(set(ids)) != len(ids):
        raise AssignmentError(f"{path}: the IDs in the first column are not distinct")
    names, columns, dropped = [], [], []
    for name in table.columns[1:]:
        try:
            values = table[name].cast(pl.Float64).to_numpy()
        except pl.exceptions.PolarsError as error:
            raise AssignmentError(f"{path}: the feature columns must hold numbers") from error
        if np.isin(values, (0.0, 1.0)).all():
            names.append(name)
            columns.append(values)
        elif non_binary == "drop":
            dropped.append(name)
        else:
            raise AssignmentError(f"{path}: the feature columns must hold only 0 and 1")
    if not columns:
        raise AssignmentError(f"{path}: no column holds only 0 and 1")
    return Meanings(ids, np.stack(columns, axis=1).astype(np.int8), names, dropped)


def load_meanings(path: str | Path) -> tuple[list[str], np.ndarray]:
    """The IDs and the feature matrix (meanings by features, 0 and 1) of a meanings table whose
    feature columns are all binary."""
    table = load_meaning_table(path)
    return table.ids, table.features


# ---------------------------------------------------------------------------------------------
# Distances and the correlation
# ---------------------------------------------------------------------------------------------


def hamming_distances(features: np.ndarray) -> np.ndarray:
    """The proportion of differing features for every pair of meanings (``np.triu_indices``
    order)."""
    rows, columns = np.triu_indices(len(features), 1)
    return (features[rows] != features[columns]).mean(axis=1)


def meaning_distances(features: np.ndarray, kind: str = "hamming") -> np.ndarray:
    """The distance between every two meanings' feature vectors, in ``np.triu_indices`` order:
    ``hamming`` (the proportion of differing features), ``cosine``, or ``jaccard`` (one minus
    the shared features over the features that either has)."""
    if kind == "hamming":
        return hamming_distances(features)
    rows, columns = np.triu_indices(len(features), 1)
    a = features[rows].astype(np.float64)
    b = features[columns].astype(np.float64)
    both = (a * b).sum(axis=1)
    if kind == "cosine":
        norms = np.sqrt(a.sum(axis=1) * b.sum(axis=1))
        return 1.0 - np.divide(both, norms, out=np.zeros_like(both), where=norms > 0)
    if kind == "jaccard":
        either = np.maximum(a, b).sum(axis=1)
        return 1.0 - np.divide(both, either, out=np.ones_like(both), where=either > 0)
    raise ValueError(f"unknown meaning distance {kind!r}")


def edit_distances(words: list[WordForm]) -> np.ndarray:
    """The phoneme edit distance between every two words, as a square matrix."""
    from semantic_world.wordforms.evaluate import edit_distance_matrix

    return edit_distance_matrix(words)


def embedding_distances(types: np.ndarray) -> np.ndarray:
    """The cosine distance between every two word embeddings, as a square matrix."""
    from semantic_world.wordforms.evaluate import cosine_distances

    return cosine_distances(types)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rounded(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)


def null_distribution(
    sound: np.ndarray, meaning: np.ndarray, observed: float, rng: np.random.Generator, samples: int
) -> dict[str, Any]:
    """The sound-meaning correlation under random reassignments of the same words: ``sound`` is
    the square matrix of the assigned words' sound distances, in the order of the meanings."""
    rows, columns = np.triu_indices(len(sound), 1)
    null = np.empty(samples, dtype=np.float64)
    for k in range(samples):
        order = rng.permutation(len(sound))
        null[k] = correlation(sound[np.ix_(order, order)][rows, columns], meaning)
    finite = null[np.isfinite(null)]
    return {
        "samples": samples,
        "mean": _rounded(finite.mean() if finite.size else float("nan")),
        "std": _rounded(finite.std(ddof=1) if finite.size > 1 else float("nan")),
        "p05": _rounded(np.percentile(finite, 5) if finite.size else float("nan")),
        "p95": _rounded(np.percentile(finite, 95) if finite.size else float("nan")),
        "p_value": _rounded(
            (np.sum(np.abs(finite) >= abs(observed)) + 1) / (finite.size + 1)
            if finite.size and np.isfinite(observed)
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------------------------


@dataclass
class Assignment:
    """Meanings with their words, and the sound-meaning correlation."""

    mode: str
    meanings: list[str]
    words: list[WordForm]
    """The word of each meaning, in the order of ``meanings``."""
    summary: dict[str, Any]
    base_words: list[WordForm] | None = None
    """With branch markers: the unmarked word of each meaning."""
    branches: list[str | None] | None = None
    markers: list[Any] = field(default_factory=list)
    """With branch markers: the marker of each branch (``closed_class.Affix``), by label."""
    marker_branches: dict[str, str] = field(default_factory=dict)
    """With branch markers: each marker's branch."""
    marked: list[WordForm] = field(default_factory=list)
    """With branch markers: the marked word forms, which join the run's lexicon."""
    features: Meanings | None = None
    lexemes: list[dict[str, Any]] | None = None
    """In a run that assigns the lexemes of a request: one record for each lexeme, with its form
    (:mod:`semantic_world.wordforms.lexemes`). ``meanings`` and ``words`` then hold the lexemes
    that the mode assigned, the lexemes of categories."""
    forms: dict[str, WordForm] = field(default_factory=dict)
    """With lexemes: the form of every lexeme, by lexeme label."""
    reserved: set[tuple[str, ...]] = field(default_factory=set)
    """With lexemes: the forms, without stress, that the assignment made or could still make
    (the marked forms, and every form that a word could have with an affix). The function words
    avoid them."""

    def frame(self) -> pl.DataFrame:
        if self.lexemes is not None:
            columns = ["lexeme", "meaning", "pos", "word", "spelling", "arpabet", "assigned"]
            if self.base_words is not None:
                columns += ["base_word", "branch", "marker"]
            return pl.DataFrame(
                {c: [record.get(c) for record in self.lexemes] for c in columns},
                schema={c: pl.String for c in columns},
            )
        data: dict[str, Any] = {
            "meaning": self.meanings,
            "word": [w.label for w in self.words],
            "spelling": [w.spelling for w in self.words],
            "arpabet": [w.arpabet for w in self.words],
        }
        if self.base_words is not None:
            data["base_word"] = [w.label for w in self.base_words]
            data["branch"] = self.branches
            data["marker"] = [w.affix for w in self.words]
        return pl.DataFrame(data, schema_overrides={"branch": pl.String, "marker": pl.String})

    def write(self, folder: str | Path) -> Path:
        out = Path(folder) / "assignment"
        out.mkdir(parents=True, exist_ok=True)
        self.frame().write_csv(out / "lexicon.csv")
        if self.markers:
            records = [
                {**m.record(), "branch": self.marker_branches[m.label]} for m in self.markers
            ]
            pl.DataFrame(records).write_csv(out / "markers.csv")
        (out / "summary.yaml").write_text(
            yaml.safe_dump(self.summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return out


def _summary(
    mode: str,
    assigned_sound: np.ndarray,
    meaning: np.ndarray,
    features: np.ndarray,
    words_available: int,
    rng: np.random.Generator,
    null_samples: int,
    sound_name: str,
    meaning_name: str,
) -> dict[str, Any]:
    rows, columns = np.triu_indices(len(assigned_sound), 1)
    observed = correlation(assigned_sound[rows, columns], meaning)
    return {
        "mode": mode,
        "meanings": len(assigned_sound),
        "features": int(features.shape[1]),
        "words_available": words_available,
        "sound_distance": sound_name,
        "meaning_distance": meaning_name,
        "correlation": _rounded(observed),
        "null": null_distribution(assigned_sound, meaning, observed, rng, null_samples),
    }


SOUND_NAMES = {"edit": "phoneme edit distance"}
MEANING_NAMES = {"hamming": "Hamming", "cosine": "cosine", "jaccard": "Jaccard"}


def _check_counts(words: list[WordForm], meanings: list[str]) -> None:
    if len(meanings) > len(words):
        raise AssignmentError(
            f"there are {len(meanings)} meanings and only {len(words)} words; raise wordforms.count"
        )


def assign_arbitrary(
    words: list[WordForm],
    meanings: list[str],
    features: np.ndarray,
    rng: np.random.Generator,
    null_samples: int = NULL_SAMPLES,
    *,
    sound: np.ndarray | None = None,
    sound_name: str = "edit",
    meaning_distance: str = "hamming",
) -> Assignment:
    """A seeded random assignment: each meaning gets one word, and no word is used twice. With
    more words than meanings, the words that are left have no meaning. ``sound`` is the square
    matrix of sound distances between the words (the phoneme edit distance by default)."""
    _check_counts(words, meanings)
    chosen = rng.permutation(len(words))[: len(meanings)]
    assigned = [words[int(i)] for i in chosen]
    meaning = meaning_distances(features, meaning_distance)
    if sound is None:
        assigned_sound = edit_distances(assigned)
    else:
        assigned_sound = sound[np.ix_(chosen, chosen)]
    summary = _summary(
        "arbitrary",
        assigned_sound,
        meaning,
        features,
        len(words),
        rng,
        null_samples,
        SOUND_NAMES.get(sound_name, f"cosine distance of {sound_name}"),
        MEANING_NAMES[meaning_distance],
    )
    return Assignment("arbitrary", list(meanings), assigned, summary)


def assign_target_correlation(
    words: list[WordForm],
    meanings: list[str],
    features: np.ndarray,
    rng: np.random.Generator,
    target: float,
    tolerance: float = 0.01,
    max_swaps: int = 20000,
    null_samples: int = NULL_SAMPLES,
    *,
    sound: np.ndarray | None = None,
    sound_name: str = "edit",
    meaning_distance: str = "hamming",
    strict: bool = False,
) -> Assignment:
    """An assignment whose sound-meaning correlation approaches ``target``. It starts random.
    Each step proposes to exchange one meaning's word with another word (another meaning's, or
    an unassigned one), and keeps the exchange when the correlation moves toward the target.
    It stops within ``tolerance`` of the target, or after ``max_swaps`` proposals, and the
    summary gives the correlation reached and whether the target was reached. A target that is
    not reached gives a warning in the summary, or an :class:`AssignmentError` when ``strict``.
    """
    _check_counts(words, meanings)
    if sound is None:
        sound = edit_distances(words)
    meaning = meaning_distances(features, meaning_distance)
    n = len(meanings)
    rows, columns = np.triu_indices(n, 1)
    order = rng.permutation(len(words))

    def current_correlation() -> float:
        chosen = order[:n]
        return correlation(sound[np.ix_(chosen, chosen)][rows, columns], meaning)

    start = current = current_correlation()
    proposals = accepted = 0
    while proposals < max_swaps and not abs(current - target) <= tolerance:
        proposals += 1
        a = int(rng.integers(n))
        b = int(rng.integers(len(words)))
        if a == b:
            continue
        order[a], order[b] = order[b], order[a]
        proposed = current_correlation()
        if np.isfinite(proposed) and (
            not np.isfinite(current) or abs(proposed - target) < abs(current - target)
        ):
            current = proposed
            accepted += 1
        else:
            order[a], order[b] = order[b], order[a]
    gap = abs(current - target)
    reached = bool(gap <= tolerance)
    warning = None
    if not reached:
        warning = (
            f"the target correlation {target} was not reached: the closest value is "
            f"{_rounded(current)} after {proposals} proposals, {_rounded(gap)} from the target "
            f"(the tolerance is {tolerance})"
        )
        if strict:
            raise AssignmentError(
                f"{warning}; change the target, raise target_correlation.max_swaps or "
                "wordforms.count, or set assignment.strict to false to keep the closest value"
            )
    chosen = order[:n]
    assigned = [words[int(i)] for i in chosen]
    summary = _summary(
        "target_correlation",
        sound[np.ix_(chosen, chosen)],
        meaning,
        features,
        len(words),
        rng,
        null_samples,
        SOUND_NAMES.get(sound_name, f"cosine distance of {sound_name}"),
        MEANING_NAMES[meaning_distance],
    )
    summary["target_correlation"] = {
        "target": target,
        "tolerance": tolerance,
        "reached": reached,
        "gap": _rounded(gap),
        "start": _rounded(start),
        "proposals": proposals,
        "accepted": accepted,
        "max_swaps": max_swaps,
    }
    if warning is not None:
        summary["target_correlation"]["warning"] = warning
    return Assignment("target_correlation", list(meanings), assigned, summary)


# ---------------------------------------------------------------------------------------------
# Branch markers
# ---------------------------------------------------------------------------------------------


def branch_of(meaning: str, depth: int) -> str | None:
    """The branch of a meaning at a depth: the first ``depth`` parts of its ID, which the
    taxonomy generator separates with periods (``C1.2.3`` is in ``C1`` at depth 1 and in
    ``C1.2`` at depth 2). None for a meaning above that depth."""
    parts = meaning.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else None


def marker_candidates(english, shape: str, position: str) -> list[tuple[tuple[str, ...], float]]:
    """Every marker syllable of a shape with its weight, from the unstressed syllables at the
    beginnings (``initial``) or the ends (``final``) of the pattern words: a single-consonant
    onset for CV and CVC, and an open rime for CV or a rime with one coda consonant for VC and
    CVC."""
    where = "initial" if position == "initial" else "final"
    closed = shape.endswith("VC")
    rimes = {
        (rime[0] + "0",) + rime[1:]: n
        for rime, n in english.rime_counts[where, False].items()
        if len(rime) == (2 if closed else 1)
    }
    onsets = {(): 1}
    if shape.startswith("C"):
        onsets = {o: n for o, n in english.onset_counts[where, False].items() if len(o) == 1}
    total_onsets, total_rimes = sum(onsets.values()), sum(rimes.values())
    return [
        (onset + rime, (n / total_onsets) * (m / total_rimes))
        for onset, n in sorted(onsets.items())
        for rime, m in sorted(rimes.items())
    ]


def assign_branch_markers(
    config,
    words: list[WordForm],
    table: Meanings,
    rng: np.random.Generator,
    english,
    spellings: set[str],
    null_samples: int = NULL_SAMPLES,
    others: list[WordForm] | None = None,
    affixes: list[Any] | None = None,
    requires: list[tuple[Any, ...]] | None = None,
    fits=None,
    taken_forms: set[tuple[str, ...]] | None = None,
) -> Assignment:
    """A random assignment in which the words of one branch share a marker syllable. Each branch
    at ``branch_markers.depth`` draws a marker, and its meanings get words that can take the
    marker: the marker joined to the word, before it or after it, with the joining repairs of
    inflected forms, gives a form that passes the phonotactic check and is neither a common
    English word nor another form of the run. The words are otherwise drawn at random, so every
    word in a branch carries the branch's marker. A meaning above the depth has no branch, and
    its word is unmarked.

    ``others`` are the run's other forms (function words and inflected forms) and ``affixes``
    its affixes: a marked form is none of the other forms, and a marker is neither a function
    word nor an affix, with or without the joining schwa.

    The rows of ``table`` can repeat an ID: the synonyms of one category are assigned one by
    one, and share its marker. ``requires`` gives, for each row, the affixes
    (``closed_class.Affix``) that the row's form must be able to take: a branch's marked forms
    then take every affix that a row of the branch requires, with the closed-class joining
    settings. ``fits(row, word)`` says whether an unmarked word can serve a row above the
    markers' depth. ``taken_forms`` are forms that no marked form, and no inflected marked
    form, may repeat; the forms that the assignment makes or reserves are added to it."""
    from semantic_world.wordforms.closed_class import (
        SCHWA,
        Affix,
        _add_statistics,
        inflect,
        repair_join,
    )
    from semantic_world.wordforms.english import strip_stress
    from semantic_world.wordforms.phonemes import load_tables
    from semantic_world.wordforms.spelling import Speller

    settings = config.assignment
    _check_counts(words, table.ids)
    tables = load_tables()
    speller = Speller.load()
    position = "prefix" if settings.marker_position == "initial" else "suffix"
    branches = [branch_of(m, settings.marker_depth) for m in table.ids]
    candidates = marker_candidates(english, settings.marker_shape, settings.marker_position)
    weights = np.array([w for _, w in candidates], dtype=np.float64)
    weights /= weights.sum()
    joining = _Joining(epenthesis=True, glide="Y")
    pool = [words[int(i)] for i in rng.permutation(len(words))]  # the words, in a random order
    others = others or []
    taken = taken_forms if taken_forms is not None else set()
    taken |= {w.stripped for w in words} | {w.stripped for w in others}
    closed = config.closed_class
    # a marker must not sound like a function word or like an affix
    reserved = [w.stripped for w in others if w.kind == "function"]
    reserved += [strip_stress(form) for affix in affixes or [] for form in affix.forms]
    schwa = strip_stress((SCHWA,))

    def same_after_schwa(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
        # one form is the other with the joining schwa (stress left out)
        return a == schwa + b or b == schwa + a or a == b + schwa or b == a + schwa

    free = {w.label for w in pool}
    base_of: dict[int, WordForm] = {}
    marked_of: dict[int, WordForm] = {}
    markers: list[Affix] = []
    marker_branches: dict[str, str] = {}
    used: list[tuple[str, ...]] = []

    def inflections(phones: tuple[str, ...], wanted: list[Any]) -> list[tuple[str, ...]] | None:
        """The forms of a marked form with each affix it must take, or None when it cannot take
        one of them."""
        made = []
        for affix in wanted:
            joined = repair_join(english, phones, affix, closed.epenthesis, closed.glide)
            if joined is None or english.is_common_pronunciation(joined[0]):
                return None
            made.append(strip_stress(joined[0]))
        return made

    for number, branch in enumerate(sorted({b for b in branches if b is not None}), start=1):
        members = [i for i, b in enumerate(branches) if b == branch]
        wanted: list[Any] = []
        for row in members:
            wanted += [a for a in (requires[row] if requires else ()) if a not in wanted]
        tries = min(MARKER_TRIES, int((weights > 0).sum()))
        for index in rng.choice(len(candidates), size=tries, replace=False, p=weights):
            phones = candidates[int(index)][0]
            plain = strip_stress(phones)
            if phones in used or any(same_after_schwa(plain, strip_stress(u)) for u in used):
                continue
            if any(plain == r or same_after_schwa(plain, r) for r in reserved):
                continue
            ipa = "".join(tables[0].phone(p) for p in phones)
            marker = Affix(f"M.{number}", branch, position, phones, ipa)
            fitting = []
            reserved_now: set[tuple[str, ...]] = set()
            for word in pool:
                if word.label not in free:
                    continue
                joined = repair_join(english, word.phones, marker, True, joining.glide)
                if joined is None or english.is_common_pronunciation(joined[0]):
                    continue
                if strip_stress(joined[0]) in taken:
                    continue
                if wanted:
                    # the marked form takes every affix that a lexeme of the branch requires,
                    # and neither it nor its inflected forms repeat another form
                    made = inflections(joined[0], wanted)
                    forms = None if made is None else [strip_stress(joined[0]), *made]
                    if forms is None or len(set(forms)) != len(forms):
                        continue
                    if any(form in taken or form in reserved_now for form in forms):
                        continue
                    reserved_now.update(forms)
                fitting.append(word)
                if len(fitting) == len(members):
                    break
            if len(fitting) == len(members):
                break
        else:
            raise AssignmentError(
                f"no marker of shape {settings.marker_shape} has enough words that can take it "
                f"for the branch {branch}; try another shape, position, or depth, or more words"
            )
        used.append(phones)
        markers.append(marker)
        marker_branches[marker.label] = branch
        forms, skipped = inflect(joining, english, speller, [(w, marker) for w in fitting])
        assert not skipped, skipped
        taken |= reserved_now
        for row, word, (form, spelling) in zip(members, fitting, forms, strict=True):
            form.kind = "marked"
            form.held_out = word.held_out
            _add_statistics(form, spelling, english, tables, [w.stripped for w in words], spellings)
            taken.add(form.stripped)
            free.discard(word.label)
            base_of[row] = word
            marked_of[row] = form
    for row, branch in enumerate(branches):
        if branch is None:  # above the markers' depth: the next free word, unmarked
            word = next(
                (w for w in pool if w.label in free and (fits is None or fits(row, w))), None
            )
            if word is None:
                raise AssignmentError(
                    f"no word is left for {table.ids[row]} that can take its affixes; raise "
                    f"wordforms.count"
                )
            free.discard(word.label)
            base_of[row] = word
    every_row = range(len(table.ids))
    base_words = [base_of[row] for row in every_row]
    final = [marked_of.get(row, base_of[row]) for row in every_row]
    meaning = meaning_distances(table.features, settings.meaning_distance)
    rows, columns = np.triu_indices(len(final), 1)
    unmarked = correlation(edit_distances(base_words)[rows, columns], meaning)
    summary = _summary(
        "branch_markers",
        edit_distances(final),
        meaning,
        table.features,
        len(words),
        rng,
        null_samples,
        SOUND_NAMES["edit"],
        MEANING_NAMES[settings.meaning_distance],
    )
    summary["branch_markers"] = {
        "depth": settings.marker_depth,
        "position": settings.marker_position,
        "shape": settings.marker_shape,
        "branches": len(markers),
        "marked_words": len(marked_of),
        "unmarked_meanings": sum(b is None for b in branches),
        "unmarked_correlation": _rounded(unmarked),
        "joins": {
            name: sum(form.join == name for form in marked_of.values())
            for name in ("none", "schwa", "glide")
        },
        "markers": {m.label: {"branch": m.gloss, "arpabet": m.arpabet} for m in markers},
    }
    return Assignment(
        "branch_markers",
        list(table.ids),
        final,
        summary,
        base_words=base_words,
        branches=branches,
        markers=markers,
        marker_branches=marker_branches,
        marked=[marked_of[row] for row in every_row if row in marked_of],
    )


@dataclass(frozen=True)
class _Joining:
    """The joining settings that a marker shares with an affix."""

    epenthesis: bool
    glide: str


# ---------------------------------------------------------------------------------------------
# The assignment of a run
# ---------------------------------------------------------------------------------------------


def needs_embeddings(config) -> bool:
    """Whether the assignment needs the run's embeddings: its sound distance is an embedding's."""
    return config.meanings is not None and config.assignment.sound_distance != "edit"


def assign(
    config,
    words: list[WordForm],
    rng: np.random.Generator,
    *,
    english=None,
    types=None,
    spellings: set[str] | None = None,
    others: list[WordForm] | None = None,
    affixes: list[Any] | None = None,
):
    """The assignment of a configuration for the content words ``words``. ``types`` holds the
    word embeddings of ``words`` when the sound distance is an embedding's. ``spellings`` are the
    spellings already used in the run, ``others`` the run's function words and inflected forms,
    and ``affixes`` its affixes: a marked form repeats none of them. Returns None without a
    meanings table."""
    settings = config.assignment
    if settings.meanings is None:
        return None
    table = load_meaning_table(settings.meanings, settings.non_binary).select(settings.categories)
    sound = None
    if settings.sound_distance != "edit":
        if types is None:
            raise AssignmentError(
                f"the sound distance {settings.sound_distance!r} needs the run's embeddings"
            )
        sound = embedding_distances(types)
    common = {
        "sound": sound,
        "sound_name": settings.sound_distance,
        "meaning_distance": settings.meaning_distance,
    }
    if settings.mode == "target_correlation":
        result = assign_target_correlation(
            words,
            table.ids,
            table.features,
            rng,
            settings.target,
            settings.tolerance,
            settings.max_swaps,
            settings.null_samples,
            strict=settings.strict,
            **common,
        )
    elif settings.mode == "branch_markers":
        from semantic_world.wordforms.english import load_english

        english = english or load_english(
            config.wordforms.english_min_zipf, config.wordforms.exclude_inflections
        )
        spellings = set(spellings) if spellings is not None else {w.spelling for w in words}
        result = assign_branch_markers(
            config, words, table, rng, english, spellings, settings.null_samples, others, affixes
        )
    else:
        result = assign_arbitrary(
            words, table.ids, table.features, rng, settings.null_samples, **common
        )
        result.mode = result.summary["mode"] = settings.mode
    for name in settings.acoustic and [m.feature for m in settings.acoustic]:
        table.feature(name)  # an unknown feature is an error before any audio is made
    result.features = table
    if table.dropped:
        result.summary["dropped_columns"] = table.dropped
    return result
