"""Assigning the lexemes of a request to content words, and inflecting the lexemes' forms.

A request (``docs/specs/CORPUS_GENERATOR.md``, "Word forms for the corpus") lists the content
lexemes of a corpus's language. Every lexeme gets a form:

- a lexeme whose concept is in the meanings table (a category) gets its word by the configured
  assignment mode: arbitrary, target correlation, branch markers, or acoustic mapping. The
  synonyms of one category are assigned one by one, each with the category's meaning vector, so
  they share its sound-meaning structure, including its branch marker;
- every other lexeme gets a word at random, from the words that are left;
- a homonym (``same_form_as``) gets the form of the lexeme it names.

**Affixes.** The request's ``takes`` lists, for each part of speech, the affixes that a lexeme's
word must be able to take. A lexeme gets only a word that can take them all, in every mode, so no
inflected form that the corpus needs is ever skipped. A word can take an affix when the joined
form passes the phonotactic check (with the glide and the schwa repairs), is not a common English
word, and is no other form that the run has or could make. The requirement comes from the
request's ``takes`` alone, never from its inflect entries. So the inflected forms that a corpus
happens to use never change which word a lexeme gets.

**Two passes.** The assignment comes first. The inflect entries that name lexemes are then
turned into inflected forms of whatever form each lexeme got, a marked form of branch-marker mode
included (``WORD.12.MARKER.2.AFFIX.1``), and those forms are synthesized and embedded with the rest.

**Function words.** A request lists its function words in order of frequency in the corpus, and
their forms depend on that order. So the function words are made after the assignment, and they
avoid its forms: a function word is no marked form, no marker, and no form that a word could
have with an affix. The words of the lexemes therefore depend on the request's lexemes, its
``takes``, its affixes, and its meanings, and on nothing that the documents of a corpus change.

The mode draws from the ``wordforms:assign`` stream, and the lexemes outside the mode draw from
its part ``wordforms:assign:lexemes``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from semantic_world.wordforms.assign import (
    MEANING_NAMES,
    SOUND_NAMES,
    Assignment,
    AssignmentError,
    Meanings,
    _rounded,
    _summary,
    assign_branch_markers,
    correlation,
    edit_distances,
    embedding_distances,
    load_meaning_table,
    meaning_distances,
)
from semantic_world.wordforms.generate import Lexicon, WordForm

RANDOM = "random"
"""How a lexeme outside the meanings table got its word."""
SAME_FORM = "same_form"
"""How a homonym got its form: it shares the form of another lexeme."""


def takeable(
    english, closed, content: list[WordForm], others: list[WordForm], affixes: list[Any]
) -> tuple[dict[str, frozenset[str]], set[tuple[str, ...]]]:
    """For each content word, the glosses of the affixes that the word can take, and every form
    that some word would have with some affix. A word can take an affix when the joined form
    passes the phonotactic check, with the repairs of an inflected form, is not a common English
    word, and is neither a form of the run (``content`` and ``others``) nor the form of another
    word with an affix. The rule looks at the word forms and the affixes alone, so it does not
    depend on which words end up assigned, or inflected."""
    from semantic_world.wordforms.closed_class import repair_join
    from semantic_world.wordforms.english import strip_stress

    base = {w.stripped for w in content} | {w.stripped for w in others}
    joined: dict[tuple[str, str], tuple[str, ...] | None] = {}
    counts: dict[tuple[str, ...], int] = {}
    for word in content:
        for affix in affixes:
            made = repair_join(english, word.phones, affix, closed.epenthesis, closed.glide)
            form = None
            if made is not None and not english.is_common_pronunciation(made[0]):
                form = strip_stress(made[0])
                counts[form] = counts.get(form, 0) + 1
            joined[word.label, affix.gloss] = form
    can = {
        word.label: frozenset(
            affix.gloss
            for affix in affixes
            if (form := joined[word.label, affix.gloss]) is not None
            and form not in base
            and counts[form] == 1
        )
        for word in content
    }
    return can, set(counts)


def constrained_order(rng: np.random.Generator, allowed: np.ndarray) -> np.ndarray:
    """A random order of the words (the columns of ``allowed``) in which row ``i`` of
    ``allowed`` is true for the word at position ``i``: each meaning gets a word it allows. The
    words come in a seeded random order, and the meanings that allow the fewest words choose
    first, each taking the first free word it allows. The words that are left follow."""
    meanings, words = allowed.shape
    pool = [int(i) for i in rng.permutation(words)]
    chosen = [-1] * meanings
    free = dict.fromkeys(pool)
    for row in sorted(range(meanings), key=lambda r: (int(allowed[r].sum()), r)):
        word = next((w for w in free if allowed[row, w]), None)
        if word is None:
            raise AssignmentError(
                "too few content words can take the affixes that the lexemes need; raise "
                "wordforms.count, or lower closed_class.affixes.max_skipped"
            )
        chosen[row] = word
        del free[word]
    return np.array(chosen + list(free), dtype=np.int64)


def _target_search(
    rng: np.random.Generator,
    allowed: np.ndarray,
    sound: np.ndarray,
    meaning: np.ndarray,
    settings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The target-correlation search of :func:`assign.assign_target_correlation`, among the
    assignments that give every meaning a word it allows."""
    n = allowed.shape[0]
    rows, columns = np.triu_indices(n, 1)
    order = constrained_order(rng, allowed)
    target, tolerance = settings.target, settings.tolerance

    def current_correlation() -> float:
        chosen = order[:n]
        return correlation(sound[np.ix_(chosen, chosen)][rows, columns], meaning)

    start = current = current_correlation()
    proposals = accepted = 0
    while proposals < settings.max_swaps and not abs(current - target) <= tolerance:
        proposals += 1
        a = int(rng.integers(n))
        b = int(rng.integers(len(order)))
        if a == b:
            continue
        if not allowed[a, order[b]] or (b < n and not allowed[b, order[a]]):
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
    report: dict[str, Any] = {
        "target": target,
        "tolerance": tolerance,
        "reached": reached,
        "gap": _rounded(gap),
        "start": _rounded(start),
        "proposals": proposals,
        "accepted": accepted,
        "max_swaps": settings.max_swaps,
    }
    if not reached:
        warning = (
            f"the target correlation {target} was not reached: the closest value is "
            f"{_rounded(current)} after {proposals} proposals, {_rounded(gap)} from the target "
            f"(the tolerance is {tolerance})"
        )
        if settings.strict:
            raise AssignmentError(
                f"{warning}; change the target, raise target_correlation.max_swaps or "
                "wordforms.count, or set assignment.strict to false to keep the closest value"
            )
        report["warning"] = warning
    return order[:n], report


def assign_lexemes(
    config,
    lexicon: Lexicon,
    streams,
    *,
    english=None,
    types: np.ndarray | None = None,
) -> Assignment:
    """Assign the lexemes of the configuration's request to the content words of ``lexicon``
    (see the module's description). ``types`` holds the content words' embeddings when the sound
    distance is an embedding's. The returned assignment holds the lexemes that the mode assigned
    (``meanings``, ``words``, ``features``), the form of every lexeme (``forms``), the records of
    ``assignment/lexicon.csv`` (``lexemes``), and, with branch markers, the marked forms."""
    from semantic_world.wordforms.english import load_english

    request = config.request
    settings = config.assignment
    closed = config.closed_class
    english = english or load_english(
        config.wordforms.english_min_zipf, config.wordforms.exclude_inflections
    )
    content = lexicon.content
    # the run's other forms so far: the inflected forms of entries that name words. The
    # function words are made after the assignment.
    others = [w for w in lexicon.words if w.kind != "content"]
    index = {w.label: i for i, w in enumerate(content)}
    affix_of = {a.gloss: a for a in lexicon.affixes}
    needs = request.requirements()
    if closed is not None and lexicon.affixes:
        can, potential = takeable(english, closed, content, others, list(lexicon.affixes))
    else:
        can, potential = {w.label: frozenset() for w in content}, set()

    leaders = request.leaders
    table = None
    if config.meanings is not None:
        table = load_meaning_table(config.meanings, settings.non_binary).select(settings.categories)
    row_of = {} if table is None else {m: i for i, m in enumerate(table.ids)}
    by_mode = [x for x in leaders if x.concept in row_of]
    at_random = [x for x in leaders if x.concept not in row_of]

    def allows(lexeme, word: WordForm) -> bool:
        return set(needs[lexeme.label]) <= can[word.label]

    # ---- the lexemes of categories, by the mode ----------------------------------------------
    mode = settings.mode
    rng = streams.assign
    form_of: dict[str, WordForm] = {}
    base_of: dict[str, WordForm] = {}
    taken: set[tuple[str, ...]] = set(potential)
    result: Assignment | None = None
    features = None
    if by_mode:
        rows = [row_of[x.concept] for x in by_mode]
        features = Meanings(
            [x.concept for x in by_mode], table.features[rows], table.names, table.dropped
        )
        for name in [m.feature for m in settings.acoustic]:
            features.feature(name)  # an unknown feature is an error before any audio is made
        allowed = np.array([[allows(x, w) for w in content] for x in by_mode], dtype=bool)
        if mode == "branch_markers":
            result = assign_branch_markers(
                config,
                content,
                features,
                rng,
                english,
                {w.spelling for w in lexicon.words},
                settings.null_samples,
                others,
                list(lexicon.affixes),
                requires=[tuple(affix_of[g] for g in needs[x.label]) for x in by_mode],
                fits=lambda row, word: bool(allowed[row, index[word.label]]),
                taken_forms=taken,
            )
            chosen_words, bases = result.words, result.base_words
        else:
            meaning = meaning_distances(features.features, settings.meaning_distance)
            sound_name = SOUND_NAMES.get(
                settings.sound_distance, f"cosine distance of {settings.sound_distance}"
            )
            sound = None
            if settings.sound_distance != "edit":
                if types is None:
                    raise AssignmentError(
                        f"the sound distance {settings.sound_distance!r} needs the run's embeddings"
                    )
                sound = embedding_distances(types)
            report = None
            if mode == "target_correlation":
                sound = edit_distances(content) if sound is None else sound
                chosen, report = _target_search(rng, allowed, sound, meaning, settings)
                assigned_sound = sound[np.ix_(chosen, chosen)]
            else:
                chosen = constrained_order(rng, allowed)[: len(by_mode)]
                if sound is None:
                    assigned_sound = edit_distances([content[int(i)] for i in chosen])
                else:
                    assigned_sound = sound[np.ix_(chosen, chosen)]
            chosen_words = bases = [content[int(i)] for i in chosen]
            summary = _summary(
                mode,
                assigned_sound,
                meaning,
                features.features,
                len(content),
                rng,
                settings.null_samples,
                sound_name,
                MEANING_NAMES[settings.meaning_distance],
            )
            if report is not None:
                summary["target_correlation"] = report
            result = Assignment(mode, list(features.ids), chosen_words, summary)
        for lexeme, word, base in zip(by_mode, chosen_words, bases, strict=True):
            form_of[lexeme.label], base_of[lexeme.label] = word, base
    if result is None:
        # no lexeme has a meaning vector: nothing for the mode to assign
        summary = {
            "mode": mode,
            "meanings": 0,
            "words_available": len(content),
            "correlation": None,
            "null": {"samples": 0, "mean": None, "std": None, "p05": None, "p95": None,
                     "p_value": None},
        }  # fmt: skip
        result = Assignment(mode, [], [], summary)
    result.mode = result.summary["mode"] = mode
    result.features = features
    if table is not None and table.dropped:
        result.summary["dropped_columns"] = table.dropped

    # ---- every other lexeme, at random among the words that are left -------------------------
    used = {w.label for w in base_of.values()}
    lexeme_rng = streams.substream("assign", "lexemes")
    pool = [content[int(i)] for i in lexeme_rng.permutation(len(content))]

    # With branch markers, the marked forms and their inflected forms came after the rule in
    # takeable, so a word's inflected forms are checked against them here.
    later = taken - potential if mode == "branch_markers" else set()

    def free_for(lexeme, word: WordForm) -> bool:
        if word.label in used or not allows(lexeme, word):
            return False
        if not later or not needs[lexeme.label]:
            return True
        from semantic_world.wordforms.closed_class import repair_join
        from semantic_world.wordforms.english import strip_stress

        made = [
            repair_join(english, word.phones, affix_of[g], closed.epenthesis, closed.glide)
            for g in needs[lexeme.label]
        ]
        return all(m is not None and strip_stress(m[0]) not in later for m in made)

    # the lexemes whose words must take the most affixes choose first
    order = sorted(range(len(at_random)), key=lambda i: (-len(needs[at_random[i].label]), i))
    for i in order:
        lexeme = at_random[i]
        word = next((w for w in pool if free_for(lexeme, w)), None)
        if word is None:
            wanted = ", ".join(needs[lexeme.label]) or "no affix"
            raise AssignmentError(
                f"no content word is left for the lexeme {lexeme.label} ({lexeme.concept}), "
                f"whose word must take {wanted}; raise wordforms.count"
            )
        used.add(word.label)
        form_of[lexeme.label] = base_of[lexeme.label] = word

    # ---- homonyms, the records, and the parts of speech --------------------------------------
    for lexeme in request.lexemes:
        if lexeme.same_form_as is not None:
            form_of[lexeme.label] = form_of[lexeme.same_form_as]
            base_of[lexeme.label] = base_of[lexeme.same_form_as]
    mode_labels = {x.label for x in by_mode}
    branch_of = dict(zip((x.label for x in by_mode), result.branches or (), strict=False))
    records = []
    for lexeme in request.lexemes:
        form, base = form_of[lexeme.label], base_of[lexeme.label]
        leader = lexeme.same_form_as or lexeme.label
        if lexeme.same_form_as is not None:
            how = SAME_FORM
        else:
            how = mode if lexeme.label in mode_labels else RANDOM
        records.append(
            {
                "lexeme": lexeme.label,
                "meaning": lexeme.concept,
                "pos": lexeme.pos,
                "word": form.label,
                "spelling": form.spelling,
                "arpabet": form.arpabet,
                "assigned": how,
                "base_word": base.label,
                "branch": branch_of.get(leader),
                "marker": form.affix if form is not base else None,
            }
        )
        parts = form.pos.split("; ") if form.pos else []
        if lexeme.pos not in parts:
            form.pos = "; ".join([*parts, lexeme.pos])
    result.lexemes = records
    result.forms = form_of
    result.reserved = taken | {w.stripped for w in result.marked}
    followers = [x for x in request.lexemes if x.same_form_as is not None]
    result.summary["lexemes"] = {
        "count": len(request.lexemes),
        "assigned_by_the_mode": len(by_mode),
        "assigned_at_random": len(at_random),
        "sharing_a_form": len(followers),
        "of_categories_sharing_a_form": sum(x.concept in row_of for x in followers),
        "words_without_a_lexeme": len(content) - len(used),
        "affixes_required": {
            gloss: {
                "lexemes": sum(gloss in wanted for wanted in needs.values()),
                "words_that_can_take_it": sum(gloss in glosses for glosses in can.values()),
            }
            for gloss in affix_of
            if any(gloss in wanted for wanted in needs.values())
        },
    }
    return result


def forms_to_avoid(lexicon: Lexicon, assignment: Assignment) -> set[tuple[str, ...]]:
    """The forms, without stress, that a function word must not be, beside the content words:
    the marked forms, the markers (with and without the joining schwa), and every form that a
    word could have with an affix."""
    from semantic_world.wordforms.closed_class import SCHWA
    from semantic_world.wordforms.english import strip_stress

    avoid = set(assignment.reserved)
    schwa = strip_stress((SCHWA,))
    for marker in assignment.markers:
        plain = strip_stress(marker.phones)
        avoid |= {plain, schwa + plain, plain + schwa}
    return avoid


def lexeme_inflections(config, lexicon: Lexicon, assignment: Assignment) -> list[tuple[Any, Any]]:
    """The stem and affix pairs that the inflect entries naming lexemes ask for, each once, in
    the order of the run's forms and then of the affixes. The stem is the form that the lexeme
    got: a content word, or a marked form."""
    by_gloss = {affix.gloss: affix for affix in lexicon.affixes}
    wanted: set[tuple[str, str]] = set()
    for entry in config.closed_class.lexeme_inflect if config.closed_class else ():
        for label in entry.lexemes:
            for gloss in entry.affixes:
                wanted.add((assignment.forms[label].label, by_gloss[gloss].label))
    return [(w, a) for w in lexicon.words for a in lexicon.affixes if (w.label, a.label) in wanted]


def inflect_lexemes(config, lexicon: Lexicon, assignment: Assignment, english=None) -> None:
    """Make the inflected forms that the request's lexemes need (the second pass), and add them
    to the lexicon. Each form takes the part of speech and the training split of its stem."""
    from semantic_world.wordforms.closed_class import add_inflected

    pairs = lexeme_inflections(config, lexicon, assignment)
    if not pairs:
        return
    made = add_inflected(config, lexicon, pairs, english)
    stems = {w.label: w for w in lexicon.words}
    for form in made:
        form.pos = stems[form.stem].pos
        form.held_out = stems[form.stem].held_out
