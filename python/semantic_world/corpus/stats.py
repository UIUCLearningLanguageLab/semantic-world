"""The statistics of a corpus: ``stats.yaml``.

The statistics describe the documents, never the test items, apart from the block ``test_sets``.
They hold:

- ``documents``: for each document type, the number of documents, and their achieved length
  beside their drawn length. A document can be shorter than its drawn length: a narrative ends
  when its events run out, and an encyclopedic document when it has nothing more to say;
- ``sentences``: counts by proposition level and by what a sentence does in its document, the
  share of negative sentences, the lengths, and the depths of the relative clauses;
- ``quantifiers``: the quantifier mix of the class-level sentences, as they are stated (a bare
  generic counts as ``generic``) and by the strongest true quantifier of each fact, which
  ``quantifiers.weights`` reweights;
- ``tokens``: the number of tokens, by part of speech, and ``lexeme_frequencies``;
- ``ambiguity``: how often sentences are ambiguous, by their readings;
- ``mentions``: the mentions of instances, and how many definite mentions could not be told
  apart;
- ``scenes``, ``propositions``, ``lexicon`` (with the verbs that get no word), and
  ``rule_statements`` (with the rule terms that were skipped, and why);
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
    ALL,
    CLASS,
    EVENT,
    GENERIC,
    INSTANCE,
    MOST,
    NO,
    SOME,
)
from semantic_world.corpus.realize import token_parts
from semantic_world.corpus.scenes import leaf_similarity
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


def occurrences(planner: Planner, documents: Sequence[Document]) -> dict[str, np.ndarray]:
    """For each measure, which leaves occur in which documents: a matrix with one row for each
    document and one column for each leaf."""
    result = planner.result
    leaf_of_label = {leaf.label: i for i, leaf in enumerate(result.tree.leaves)}
    generator = planner.scene_generator
    leaf_of_instance = {
        label: int(generator.leaf[row]) for row, label in enumerate(result.instances.labels)
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
    generator = planner.scene_generator
    leaves = len(planner.result.tree.leaves)
    upper = np.triu_indices(leaves, 1)
    thematic = generator.thematic[upper]
    taxonomic = leaf_similarity(planner.result, generator.leaf_rows)[upper]
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
    return {
        "count": len(sentences),
        "by_level": {level: levels[level] for level in (CLASS, INSTANCE, EVENT)},
        "by_section": dict(sorted(Counter(s.section for s in sentences).items())),
        "negative_share": {
            level: _share(negative[level], levels[level]) for level in (CLASS, INSTANCE)
        },
        "length_in_tokens": {**_summary(lengths), "histogram": _histogram(lengths)},
        "relative_clause_depth": _histogram([s.plan.depth() for s in sentences]),
    }


def _quantifiers(documents: Sequence[Document]) -> dict[str, Any]:
    sentences = [s for d in documents for s in d.sentences if s.proposition.level == CLASS]
    stated = Counter(str(s.proposition.quantifier) for s in sentences)
    strongest = Counter(str(s.strength) for s in sentences)

    def mix(counts: Counter, keys: Sequence[str]) -> dict[str, Any]:
        return {
            key: {"count": counts[key], "share": _share(counts[key], len(sentences))}
            for key in keys
        }

    return {
        "class_level_sentences": len(sentences),
        "stated": mix(stated, (ALL, MOST, SOME, NO, GENERIC)),
        "strongest_true": mix(strongest, (ALL, MOST, SOME, NO, POLE)),
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
    events = [event for scene in planner.scenes for event in scene.events]
    progressive = sum(event.aspect == "progressive" for event in events)
    return {
        "count": len(planner.scenes),
        "events": len(events),
        "progressive_share": _share(progressive, len(events)),
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
