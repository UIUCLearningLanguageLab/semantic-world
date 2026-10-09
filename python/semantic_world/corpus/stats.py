"""The statistics of a corpus: ``stats.yaml``.

The statistics describe the documents, never the test items, apart from the block ``test_sets``.
They hold:

- ``documents``: for each document type, the number of documents, and their achieved length
  beside their drawn length. A document can be shorter than its drawn length: a narrative ends
  when its events run out, and an encyclopedic document when it has nothing more to say;
- ``sentences``: counts by proposition level and by what a sentence does in its document, the
  share of negative sentences, the share of progressive reports among the event sentences, the
  lengths, and the depths of the relative clauses;
- ``quantifiers``: the quantifier mix of the class-level sentences, by the quantifier of each
  fact (the strongest true one the language can state, which ``quantifiers.weights``
  reweights), and how many are said with a bare plural;
- ``tokens``: the number of tokens, by part of speech, and ``lexeme_frequencies``;
- ``ambiguity``: how often sentences are ambiguous, by their readings;
- ``mentions``: the mentions of instances, and how many definite mentions could not be told
  apart;
- ``scenes`` (the histories: steps, events, changes, quiescence), ``propositions``,
  ``lexicon`` (with the event types that get no word), and ``rule_statements`` (with the rule
  terms that were skipped, and why);
- ``cooccurrence``: the co-occurrence check;
- ``test_sets``: for each test set, the number of pairs, and the share of its true items whose
  logical form appears in a document.

**The co-occurrence check.** Over the unordered pairs of leaves, we correlate within-document
co-occurrence with thematic relatedness and with taxonomic similarity, separately for each
document type. A pair's co-occurrence is the number of documents in which both leaves occur. A
leaf occurs in a document in one of two ways, and both are reported:

- ``words``: the leaf's own noun appears in the document;
- ``referents``: a noun phrase of the document refers to an instance of the leaf, whatever its
  noun ("the bird", "it"), or to the leaf category itself.

The correlations are Pearson's and Spearman's. The check confirms that the document mix works as
a lever: situational documents should correlate more with thematic relatedness than encyclopedic
documents do, and less with taxonomic similarity.

**Separating the two signals.** Thematic relatedness and taxonomic similarity are themselves
correlated in a world, so a correlation with one carries some of the other. The check therefore
also reports:

- ``world``: the correlation between thematic relatedness and taxonomic similarity over the
  pairs of leaves, in the world itself;
- ``partial``, for each group of documents and each measure: the correlation of co-occurrence
  with thematic relatedness controlling for taxonomic similarity, and the reverse.

A partial correlation is computed from the three pairwise correlations, over the pairs with a
defined taxonomic similarity. Spearman's partial correlation is the same formula on the ranks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from semantic_world.corpus.config import DOCUMENT_TYPES
from semantic_world.corpus.grammar import INSTANCE_NP
from semantic_world.corpus.planner import POLE, Document, Planner, reading_counts
from semantic_world.corpus.propositions import (
    CLASS,
    EVENT,
    INSTANCE,
    PROGRESSIVE,
    QUANTIFIERS,
)
from semantic_world.corpus.realize import token_parts
from semantic_world.corpus.testsets import ItemSet

ENCYCLOPEDIC = "encyclopedic"
NARRATIVE = "narrative"
EVERY = "all"
GROUPS = {
    **{kind: (kind,) for kind in DOCUMENT_TYPES},
    ENCYCLOPEDIC: DOCUMENT_TYPES[:2],
    NARRATIVE: DOCUMENT_TYPES[2:],
    EVERY: DOCUMENT_TYPES,
}
"""The groups of documents that the co-occurrence check reports: each document type, the two
encyclopedic types together, the two narrative types together, and every document."""
WORDS = "words"
REFERENTS = "referents"
MEASURES = (WORDS, REFERENTS)


def _number(value: float) -> float:
    """A real number as it is written to the output files: 6 decimal places."""
    return round(float(value), 6)


def _share(count: int, total: int) -> float | None:
    return _number(count / total) if total else None


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": _number(np.mean(values)), "min": int(min(values)), "max": int(max(values))}


def _histogram(values: Sequence[int]) -> dict[int, int]:
    return {int(value): count for value, count in sorted(Counter(values).items())}


# ---------------------------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------------------------


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """Pearson's and Spearman's correlations of two variables. Null when one does not vary."""

    def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            return None
        return _number(np.corrcoef(a, b)[0, 1])

    def ranks(a: np.ndarray) -> np.ndarray:
        return pl.Series(a).rank("average").to_numpy()

    return {"pearson": pearson(x, y), "spearman": pearson(ranks(x), ranks(y))}


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float | None]:
    """Pearson's and Spearman's partial correlations of ``x`` and ``y`` controlling for ``z``:
    ``(r_xy - r_xz r_yz) / sqrt((1 - r_xz^2) (1 - r_yz^2))``, on the values and on their ranks.
    Null when a variable does not vary, or when ``z`` determines ``x`` or ``y``."""

    def ranks(a: np.ndarray) -> np.ndarray:
        return pl.Series(a).rank("average").to_numpy()

    def partial(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
        if len(a) < 3 or a.std() == 0 or b.std() == 0 or c.std() == 0:
            return None
        r = np.corrcoef(np.stack([a, b, c]))
        rest = (1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2)
        if rest <= 1e-12:
            return None
        return _number((r[0, 1] - r[0, 2] * r[1, 2]) / np.sqrt(rest))

    return {"pearson": partial(x, y, z), "spearman": partial(ranks(x), ranks(y), ranks(z))}


def occurrences(planner: Planner, documents: Sequence[Document]) -> dict[str, np.ndarray]:
    """For each measure, which leaves occur in which documents: a matrix with one row for each
    document and one column for each leaf."""
    world = planner.world
    leaf_of_label = {leaf: i for i, leaf in enumerate(world.leaves)}
    leaf_of_instance = {
        label: int(world.entity_leaf[row]) for row, label in enumerate(world.instances)
    }
    concept = {lexeme.label: lexeme.concept for lexeme in planner.lexicon.lexemes}
    shape = (len(documents), len(leaf_of_label))
    found = {measure: np.zeros(shape, dtype=np.int64) for measure in MEASURES}
    for row, document in enumerate(documents):
        for sentence in document.sentences:
            for token in sentence.sentence.tokens:
                leaf = leaf_of_label.get(concept[token_parts(token)[0]])
                if leaf is not None:
                    found[WORDS][row, leaf] = 1
            for referent, _ in sentence.sentence.referents:
                leaf = leaf_of_instance.get(referent, leaf_of_label.get(referent))
                if leaf is not None:
                    found[REFERENTS][row, leaf] = 1
    return found


def cooccurrence(planner: Planner, documents: Sequence[Document]) -> dict[str, Any]:
    """The co-occurrence check (see the module's description)."""
    world = planner.world
    leaves = len(world.leaves)
    upper = np.triu_indices(leaves, 1)
    thematic = world.thematic[upper]
    taxonomic = world.leaf_similarity[upper]
    defined = ~np.isnan(taxonomic)
    found = occurrences(planner, documents)
    types = np.array([document.type for document in documents])
    groups: dict[str, Any] = {}
    for group, kinds in GROUPS.items():
        rows = np.isin(types, kinds)
        record: dict[str, Any] = {"documents": int(rows.sum())}
        for measure in MEASURES:
            matrix = found[measure][rows]
            counts = (matrix.T @ matrix)[upper].astype(float)
            record[measure] = {
                "thematic": correlation(counts, thematic),
                "taxonomic": correlation(counts[defined], taxonomic[defined]),
                "partial": {
                    "thematic_given_taxonomic": partial_correlation(
                        counts[defined], thematic[defined], taxonomic[defined]
                    ),
                    "taxonomic_given_thematic": partial_correlation(
                        counts[defined], taxonomic[defined], thematic[defined]
                    ),
                },
            }
        groups[group] = record

    def above(a: float | None, b: float | None) -> bool | None:
        return None if a is None or b is None else bool(a > b)

    check: dict[str, Any] = {}
    situational, encyclopedic = groups[DOCUMENT_TYPES[3]], groups[ENCYCLOPEDIC]
    for measure in MEASURES:
        check[measure] = {
            "thematic_situational_above_encyclopedic": {
                kind: above(
                    situational[measure]["thematic"][kind], encyclopedic[measure]["thematic"][kind]
                )
                for kind in ("pearson", "spearman")
            },
            "taxonomic_situational_below_encyclopedic": {
                kind: above(
                    encyclopedic[measure]["taxonomic"][kind],
                    situational[measure]["taxonomic"][kind],
                )
                for kind in ("pearson", "spearman")
            },
        }
    return {
        "leaves": leaves,
        "pairs": len(thematic),
        "pairs_with_taxonomic_similarity": int(defined.sum()),
        "world": {"thematic_taxonomic": correlation(thematic[defined], taxonomic[defined])},
        "by_document_type": groups,
        "check": check,
    }


# ---------------------------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------------------------


def _documents(documents: Sequence[Document]) -> dict[str, Any]:
    by_type: dict[str, Any] = {}
    for kind in DOCUMENT_TYPES:
        made = [d for d in documents if d.type == kind]
        lengths = [len(d.sentences) for d in made]
        drawn = [d.drawn_length for d in made]
        short = sum(a < b for a, b in zip(lengths, drawn, strict=True))
        by_type[kind] = {
            "count": len(made),
            "sentences": int(sum(lengths)),
            "drawn_length": _summary(drawn),
            "achieved_length": _summary(lengths),
            "shorter_than_drawn": short,
            "shorter_than_drawn_share": _share(short, len(made)),
        }
    return {"count": len(documents), "by_type": by_type}


def _sentences(documents: Sequence[Document]) -> dict[str, Any]:
    sentences = [s for d in documents for s in d.sentences]
    levels = Counter(s.proposition.level for s in sentences)
    negative = Counter(s.proposition.level for s in sentences if s.proposition.negative)
    lengths = [len(s.sentence.tokens) for s in sentences]
    reports = [s for s in sentences if s.proposition.level == EVENT]
    progressive = sum(s.proposition.aspect == PROGRESSIVE for s in reports)
    return {
        "count": len(sentences),
        "by_level": {level: levels[level] for level in (CLASS, INSTANCE, EVENT)},
        "by_section": dict(sorted(Counter(s.section for s in sentences).items())),
        "negative_share": {
            level: _share(negative[level], levels[level]) for level in (CLASS, INSTANCE)
        },
        "progressive_share": _share(progressive, len(reports)),
        "length_in_tokens": {**_summary(lengths), "histogram": _histogram(lengths)},
        "relative_clause_depth": _histogram([s.plan.depth() for s in sentences]),
    }


def _quantifiers(documents: Sequence[Document]) -> dict[str, Any]:
    sentences = [s for d in documents for s in d.sentences if s.proposition.level == CLASS]
    stated = Counter(str(s.strength) for s in sentences)
    bare = sum(s.bare_plural for s in sentences)
    return {
        "class_level_sentences": len(sentences),
        "stated": {
            key: {"count": stated[key], "share": _share(stated[key], len(sentences))}
            for key in (*QUANTIFIERS, POLE)
        },
        "bare_plural": {"count": bare, "share": _share(bare, len(sentences))},
    }


def _tokens(planner: Planner, documents: Sequence[Document]) -> tuple[dict[str, Any], dict]:
    counts: Counter = Counter()
    for document in documents:
        for sentence in document.sentences:
            counts.update(token_parts(token)[0] for token in sentence.sentence.tokens)
    by_pos: Counter = Counter()
    for lexeme in planner.lexicon.lexemes:
        by_pos[lexeme.pos] += counts[lexeme.label]
    frequencies = {lexeme.label: counts[lexeme.label] for lexeme in planner.lexicon.lexemes}
    return {"count": int(sum(counts.values())), "by_pos": dict(by_pos)}, frequencies


def _mentions(documents: Sequence[Document]) -> dict[str, Any]:
    kinds: Counter = Counter()
    undistinguished = 0
    for document in documents:
        for sentence in document.sentences:
            undistinguished += sum(mark is False for mark in sentence.distinguished)
            for phrase in sentence.sentence.phrases:
                if phrase.kind != INSTANCE_NP:
                    continue
                kinds["pronoun" if phrase.pronoun else str(phrase.determiner)] += 1
    definite = kinds["the"]
    return {
        "instance_mentions": int(sum(kinds.values())),
        "indefinite": kinds["a"],
        "definite": definite,
        "pronouns": kinds["pronoun"],
        "definite_not_distinguished": undistinguished,
        "definite_not_distinguished_share": _share(undistinguished, definite),
    }


def _scenes(planner: Planner) -> dict[str, Any]:
    scenes = planner.scenes
    steps = [step for scene in scenes for step in scene.steps]
    events = [event for step in steps for event in step.events]
    changes = sum(len(event.changes) for event in events)
    quiescent = sum(scene.quiescent for scene in scenes)
    return {
        "count": len(scenes),
        "steps": len(steps),
        "events": len(events),
        "mean_changes_per_event": _share(changes, len(events)),
        "quiescent_share": _share(quiescent, len(scenes)),
    }


def corpus_stats(
    planner: Planner, documents: Sequence[Document], test_sets: Sequence[ItemSet] = ()
) -> dict[str, Any]:
    """``stats.yaml``: the statistics of a corpus, as a plain mapping."""
    tokens, frequencies = _tokens(planner, documents)
    ambiguity = reading_counts(documents)
    ambiguity["ambiguous_share"] = _number(ambiguity["ambiguous_share"])
    propositions = {s.proposition for d in documents for s in d.sentences}
    planner.facts.rule_statements()  # the report on the rule terms is made with the statements
    return {
        "documents": _documents(documents),
        "sentences": _sentences(documents),
        "quantifiers": _quantifiers(documents),
        "tokens": tokens,
        "ambiguity": ambiguity,
        "mentions": _mentions(documents),
        "scenes": _scenes(planner),
        "propositions": {"distinct": len(propositions)},
        "lexicon": planner.lexicon.stats(),
        "rule_statements": dict(planner.facts.rule_report),
        "cooccurrence": cooccurrence(planner, documents),
        "test_sets": {test_set.name: test_set.stats() for test_set in test_sets},
        "lexeme_frequencies": frequencies,
    }
