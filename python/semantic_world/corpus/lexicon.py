"""Layer 1: the lexicon. Concepts from the world get words.

A **concept** is something the language can name: a category, a feature, a verb, a verb category,
an exposed patient projection, a pole of a scalar dimension, the generic head noun, or a function
word. Each concept has a concept label. Categories, features, verbs, and projections keep their
taxonomy labels (``C1.3.2``, ``IS.12``, ``V1.2``, ``CANBE.V1.1``). A scalar pole is its dimension's
label with ``HIGH`` or ``LOW`` (``SC.1.HIGH``). The generic head noun is ``THING``, and a function
word's concept label is its gloss in capitals (``THE``).

A **lexeme** is a word of the language, labeled ``L.<n>``. Every lexeme has exactly one concept.
By default there is one lexeme per named concept. Two knobs add ambiguity:

- a concept with a synonym has two lexemes;
- a homonym is two lexemes, with different concepts, that share one word form. The later lexeme
  of the pair records the earlier one in ``same_form_as``.

Lexeme labels are numbered in this order: the first lexeme of every named content concept, in
concept order; then the second lexemes of concepts with synonyms; then the function words.
Content lexemes therefore keep their labels when the grammar settings change the set of function
words.

Word forms are attached later, by the ``render`` command, so ``word`` and ``spelling`` start
empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

from semantic_world.corpus.config import CONCEPT_TYPES, Config
from semantic_world.corpus.streams import Streams
from semantic_world.taxonomy.generate import TaxonomyResult

NOUN = "noun"
ADJECTIVE = "adjective"
PART_NOUN = "part_noun"
INTRANSITIVE_VERB = "intransitive_verb"
TRANSITIVE_VERB = "transitive_verb"
FUNCTION_WORD = "function_word"
PARTS_OF_SPEECH = (NOUN, ADJECTIVE, PART_NOUN, INTRANSITIVE_VERB, TRANSITIVE_VERB, FUNCTION_WORD)

GENERIC = "generic"
"""The concept type of the generic head noun."""
FUNCTION = "function"
"""The concept type of a function word."""
THING = "THING"
SCALAR_POLES = ("HIGH", "LOW")

FUNCTION_WORDS = (
    "a", "the", "all", "most", "some", "no", "not", "can", "is", "has", "with", "without",
    "and", "that", "it",
)  # fmt: skip
"""The glosses of the function words every language has, in the order of the specification."""
AGREEMENT_WORDS = ("are", "have")
"""The plural forms of ``is`` and ``has``, added when verbs agree with their subjects."""

EVERY_PAIR = "every pair"
NO_PAIR = "no pair"

LEXICON_COLUMNS = ("label", "pos", "concept", "word", "spelling", "gloss", "same_form_as")


@dataclass(frozen=True)
class Concept:
    label: str
    type: str
    """One of the configuration's concept types, or ``generic`` or ``function``."""
    pos: str
    gloss: str
    """What the formal rendering shows for the concept's lexemes: the concept label, or a
    function word's gloss (``the``)."""

    @property
    def content(self) -> bool:
        return self.type != FUNCTION


@dataclass(frozen=True)
class Lexeme:
    label: str
    pos: str
    concept: str
    """The label of the lexeme's one concept."""
    gloss: str
    same_form_as: str | None = None
    """For the later lexeme of a homonym pair: the lexeme whose word form it shares."""
    word: str | None = None
    """The word form's label, once word forms are attached."""
    spelling: str | None = None

    @property
    def content(self) -> bool:
        return self.pos != FUNCTION_WORD


@dataclass(frozen=True)
class Lexicon:
    concepts: tuple[Concept, ...]
    """Every concept that could get a word, named or not, in concept order."""
    lexemes: tuple[Lexeme, ...]
    unnamed: tuple[str, ...]
    """The concepts that the named proportions left without a word."""
    verbs_without_word: dict[str, str]
    """The verbs and verb categories that never get a word, because their relation holds for
    ``every pair`` of distinct instances or for ``no pair``. They are not among ``concepts``."""

    def __post_init__(self) -> None:
        by_concept: dict[str, list[Lexeme]] = {}
        for lexeme in self.lexemes:
            by_concept.setdefault(lexeme.concept, []).append(lexeme)
        object.__setattr__(self, "_concepts", {c.label: c for c in self.concepts})
        object.__setattr__(self, "_lexemes", {x.label: x for x in self.lexemes})
        object.__setattr__(self, "_by_concept", {k: tuple(v) for k, v in by_concept.items()})

    def concept(self, label: str) -> Concept:
        try:
            return self._concepts[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown concept {label!r}") from None

    def lexeme(self, label: str) -> Lexeme:
        try:
            return self._lexemes[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown lexeme {label!r}") from None

    def lexemes_of(self, concept: str) -> tuple[Lexeme, ...]:
        """The lexemes of a concept: none when the concept has no word, two with a synonym."""
        return self._by_concept.get(concept, ())  # type: ignore[attr-defined]

    def is_named(self, concept: str) -> bool:
        return concept in self._by_concept  # type: ignore[attr-defined]

    def function_word(self, gloss: str) -> Lexeme:
        """The lexeme of a function word, by its gloss (``the``, ``PLURAL``)."""
        lexemes = self.lexemes_of(gloss.upper())
        if not lexemes or lexemes[0].pos != FUNCTION_WORD:
            raise KeyError(f"the language has no function word {gloss!r}")
        return lexemes[0]

    @property
    def content_lexemes(self) -> tuple[Lexeme, ...]:
        return tuple(x for x in self.lexemes if x.content)

    @property
    def function_lexemes(self) -> tuple[Lexeme, ...]:
        return tuple(x for x in self.lexemes if not x.content)

    def of_type(self, concept_type: str) -> tuple[Concept, ...]:
        return tuple(c for c in self.concepts if c.type == concept_type)

    def homonym_pairs(self) -> tuple[tuple[Lexeme, Lexeme], ...]:
        """The homonym pairs: the lexeme whose form is shared, then the lexeme that shares it."""
        return tuple(
            (self.lexeme(x.same_form_as), x) for x in self.lexemes if x.same_form_as is not None
        )

    def frame(self) -> pl.DataFrame:
        """``lexicon.csv``: one row per lexeme."""
        data = {column: [getattr(x, column) for x in self.lexemes] for column in LEXICON_COLUMNS}
        return pl.DataFrame(data, schema={column: pl.Utf8 for column in LEXICON_COLUMNS})

    def stats(self) -> dict[str, Any]:
        """The lexicon's block of ``stats.yaml``: counts by concept type and part of speech, the
        realized synonym and homonym rates, and the verbs without a word."""
        content_concepts = [c for c in self.concepts if c.content and self.is_named(c.label)]
        content = self.content_lexemes
        pairs = self.homonym_pairs()
        types = CONCEPT_TYPES + (GENERIC, FUNCTION)
        return {
            "concepts": {t: len(self.of_type(t)) for t in types},
            "named_concepts": {
                t: sum(self.is_named(c.label) for c in self.of_type(t)) for t in types
            },
            "lexemes": len(self.lexemes),
            "lexemes_by_pos": {
                pos: sum(x.pos == pos for x in self.lexemes) for pos in PARTS_OF_SPEECH
            },
            "synonym_rate": (
                sum(len(self.lexemes_of(c.label)) > 1 for c in content_concepts)
                / len(content_concepts)
                if content_concepts
                else 0.0
            ),
            "homonym_rate": 2 * len(pairs) / len(content) if content else 0.0,
            "homonym_pairs": len(pairs),
            "homonym_pairs_same_pos": sum(a.pos == b.pos for a, b in pairs),
            "verbs_without_word": dict(self.verbs_without_word),
        }


# ---------------------------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------------------------


def function_word_glosses(config: Config) -> tuple[str, ...]:
    """The glosses of the language's function words. The set is fixed by the settings: the base
    words, ``are`` and ``have`` when verbs agree with their subjects, and one word for each
    inflection that is realized as a separate word (``PLURAL``, ``PAST``, ``PROGRESSIVE``)."""
    morphology = config.grammar.morphology
    glosses = FUNCTION_WORDS
    if morphology.agreement:
        glosses += AGREEMENT_WORDS
    return glosses + morphology.inflection_words()


def relation_extent(result: TaxonomyResult, verb: str) -> str | None:
    """``every pair`` when a verb's relation (a verb category's base relation) holds for every
    ordered pair of distinct instances, ``no pair`` when it holds for none, and None otherwise.
    With fewer than two instances there is no pair."""
    assert result.relations is not None
    n = len(result.instances)
    true_pairs = int(result.relations.matrix(verb).sum())
    if true_pairs == 0:
        return NO_PAIR
    if true_pairs == n * (n - 1):
        return EVERY_PAIR
    return None


def world_concepts(result: TaxonomyResult) -> tuple[tuple[Concept, ...], dict[str, str]]:
    """The content concepts of a world, in the order of the specification's table, and the verbs
    and verb categories that never get a word."""
    features = result.features
    concepts = [Concept(c.label, "category", NOUN, c.label) for c in result.tree.categories]
    for feature_type, pos in (("is", ADJECTIVE), ("has", PART_NOUN), ("can", INTRANSITIVE_VERB)):
        concepts += [
            Concept(f.label, feature_type, pos, f.label) for f in features.of_type(feature_type)
        ]
    without_word: dict[str, str] = {}
    if result.verbs is not None:
        categories = result.verbs.categories
        for concept_type, leaf in (("verb", True), ("verb_category", False)):
            for category in categories:
                if category.is_leaf != leaf:
                    continue
                extent = relation_extent(result, category.label)
                if extent is not None:
                    without_word[category.label] = extent
                else:
                    concepts.append(
                        Concept(category.label, concept_type, TRANSITIVE_VERB, category.label)
                    )
        # category order, whatever the order of the two passes above
        order = {c.label: i for i, c in enumerate(categories)}
        without_word = dict(sorted(without_word.items(), key=lambda item: order[item[0]]))
    if result.projections is not None:
        for verb in result.projections.exposed_patient:
            label = f"CANBE.{verb}"
            concepts.append(Concept(label, "patient_projection", ADJECTIVE, label))
    for scalar in features.scalar_labels:
        for pole in SCALAR_POLES:
            label = f"{scalar}.{pole}"
            concepts.append(Concept(label, "scalar", ADJECTIVE, label))
    concepts.append(Concept(THING, GENERIC, NOUN, THING))
    return tuple(concepts), without_word


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _stochastic_round(x: float, rng: np.random.Generator) -> int:
    """``x`` rounded down or up at random, so that the expected value is ``x``."""
    low = math.floor(x)
    return low + int(rng.random() < x - low)


def _named(config: Config, streams: Streams, concepts: tuple[Concept, ...]) -> set[str]:
    """The content concepts that get a word. For each concept type, the number of named units is
    the proportion times the number of units, rounded half up, and the units are drawn at random
    from the type's own part of the lexicon stream. A unit is one concept, or the two poles of
    one scalar dimension."""
    named = {c.label for c in concepts if c.type == GENERIC}
    for concept_type in CONCEPT_TYPES:
        units: dict[str, list[str]] = {}
        for concept in concepts:
            if concept.type != concept_type:
                continue
            unit = concept.label.rsplit(".", 1)[0] if concept_type == "scalar" else concept.label
            units.setdefault(unit, []).append(concept.label)
        keys = list(units)
        count = _round_half_up(config.lexicon.named_proportion[concept_type] * len(keys))
        if count >= len(keys):
            chosen = range(len(keys))
        else:
            rng = streams.substream("lexicon", f"named:{concept_type}")
            chosen = rng.choice(len(keys), size=count, replace=False)
        for index in chosen:
            named.update(units[keys[int(index)]])
    return named


# ---------------------------------------------------------------------------------------------
# Lexemes
# ---------------------------------------------------------------------------------------------


def _homonym_pairs(
    lexemes: list[Lexeme], rate: float, same_pos: float, rng: np.random.Generator
) -> dict[int, int]:
    """Homonym pairs among the content lexemes, as a mapping from the later lexeme's index to
    the earlier one's. The number of pairs is ``rate`` times the number of lexemes, over 2,
    rounded at random, so the expected share of lexemes that share a word form is ``rate``. The
    first lexeme of a pair is drawn from the unpaired lexemes, and its partner from the unpaired
    lexemes of other concepts: of the same part of speech with probability ``same_pos``, and of
    another part of speech otherwise. When the wanted kind of partner does not exist, the other
    kind is used."""
    pairs: dict[int, int] = {}
    wanted = _stochastic_round(rate * len(lexemes) / 2, rng)
    unpaired = list(range(len(lexemes)))
    while len(pairs) < wanted and len(unpaired) >= 2:
        first = unpaired.pop(int(rng.integers(len(unpaired))))
        others = [i for i in unpaired if lexemes[i].concept != lexemes[first].concept]
        same = [i for i in others if lexemes[i].pos == lexemes[first].pos]
        different = [i for i in others if lexemes[i].pos != lexemes[first].pos]
        want_same = rng.random() < same_pos
        pool = (same or different) if want_same else (different or same)
        if not pool:
            continue  # only the lexeme's own synonym is left
        second = pool[int(rng.integers(len(pool)))]
        unpaired.remove(second)
        pairs[max(first, second)] = min(first, second)
    return pairs


def build_lexicon(config: Config, result: TaxonomyResult, streams: Streams) -> Lexicon:
    """The lexicon of a world: its concepts, and the lexemes of the named ones."""
    content, without_word = world_concepts(result)
    named = _named(config, streams, content)
    named_concepts = [c for c in content if c.label in named]

    settings = config.lexicon
    # One draw per named concept, so the synonyms at one rate are among those at a higher rate.
    draws = streams.substream("lexicon", "synonyms").random(len(named_concepts))
    with_synonym = [
        c for c, u in zip(named_concepts, draws, strict=True) if u < settings.synonym_rate
    ]

    lexemes = [
        Lexeme(f"L.{i}", c.pos, c.label, c.gloss)
        for i, c in enumerate(named_concepts + with_synonym, start=1)
    ]
    pairs = _homonym_pairs(
        lexemes,
        settings.homonym_rate,
        settings.homonym_same_pos,
        streams.substream("lexicon", "homonyms"),
    )
    for later, earlier in pairs.items():
        lexemes[later] = replace(lexemes[later], same_form_as=lexemes[earlier].label)

    function = tuple(
        Concept(gloss.upper(), FUNCTION, FUNCTION_WORD, gloss)
        for gloss in function_word_glosses(config)
    )
    lexemes += [
        Lexeme(f"L.{i}", c.pos, c.label, c.gloss)
        for i, c in enumerate(function, start=len(lexemes) + 1)
    ]
    return Lexicon(
        concepts=content + function,
        lexemes=tuple(lexemes),
        unnamed=tuple(c.label for c in content if c.label not in named),
        verbs_without_word=without_word,
    )
