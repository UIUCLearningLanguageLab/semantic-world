"""The request for the word-form pipeline: ``wordform_request.yaml``.

The corpus is generated first, and the word forms are made for it. ``generate`` writes a request
that the word-form pipeline reads (``request:`` in its configuration). The request lists:

- ``lexemes``: every content lexeme, with its concept, its part of speech, and, for a homonym,
  the lexeme whose word form it shares (``same_form_as``);
- ``takes``: for each part of speech, the affixes that its lexemes' words must be able to take.
  The list comes from the grammar settings alone: with number realized as an affix, every noun
  takes ``PLURAL``, and so does every verb when verbs agree; with tense as an affix, every verb
  takes ``PAST``; with aspect as an affix, every verb takes ``PROGRESSIVE``. So the word that a
  lexeme gets never depends on which inflected forms the documents or the test items happen to
  use;
- ``function_words``: the glosses of the language's function words, most frequent first, by
  their counts in the documents. Function words with the same count keep the lexicon's order;
- ``affixes``: the gloss and the position of every inflection that the grammar settings realize
  as an affix;
- ``inflect``: the lexemes that appear inflected, in a document or in a test item, with the
  affix each one takes. The list decides only which inflected forms are made, synthesized, and
  embedded;
- ``meanings``: the file of the categories' meaning vectors, ``wordform_meanings.csv``, which
  ``generate`` writes beside the request. The file is the taxonomy's
  ``categories_generative.csv`` with the world's labels. The path is relative to the request
  file, so the output folder can be moved.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from semantic_world.corpus.config import Config
from semantic_world.corpus.lexicon import INTRANSITIVE_VERB, NOUN, TRANSITIVE_VERB
from semantic_world.corpus.realize import PAST, PLURAL, PROGRESSIVE, token_parts
from semantic_world.corpus.world import World

if TYPE_CHECKING:
    from semantic_world.corpus.generate import Corpus

REQUEST_FILE = "wordform_request.yaml"
MEANINGS_FILE = "wordform_meanings.csv"
VERBS = (INTRANSITIVE_VERB, TRANSITIVE_VERB)


def affix_items(config: Config) -> list[dict[str, str]]:
    """The inflections that the grammar settings realize as affixes, with their positions: an
    inflection that stands after its word is a suffix, and one that stands before it a prefix."""
    morphology = config.grammar.morphology
    parts = (
        (PLURAL, morphology.number),
        (PAST, morphology.tense),
        (PROGRESSIVE, morphology.aspect),
    )
    return [
        {"gloss": gloss, "position": "suffix" if part.position == "after" else "prefix"}
        for gloss, part in parts
        if part.enabled and part.realization == "affix"
    ]


def takes(config: Config) -> dict[str, list[str]]:
    """For each part of speech, the affixes that its lexemes' words must be able to take, by the
    grammar settings alone. A part of speech that takes no affix is left out. Part nouns and
    adjectives are never inflected."""
    morphology = config.grammar.morphology
    affixes = {item["gloss"] for item in affix_items(config)}
    noun = [PLURAL] if PLURAL in affixes else []
    verb = [PLURAL] if PLURAL in affixes and morphology.agreement else []
    verb += [gloss for gloss in (PAST, PROGRESSIVE) if gloss in affixes]
    wanted = {NOUN: noun, **{pos: list(verb) for pos in VERBS}}
    return {pos: glosses for pos, glosses in wanted.items() if glosses}


def inflected(token_lists: Iterable[Iterable[str]]) -> dict[str, set[str]]:
    """For each affix gloss, the lexemes that appear with the affix among the tokens."""
    found: dict[str, set[str]] = {}
    for tokens in token_lists:
        for token in tokens:
            label, affix = token_parts(token)
            if affix is not None:
                found.setdefault(affix, set()).add(label)
    return found


def _number(label: str) -> int:
    return int(label.rsplit(".", 1)[1])


def wordform_request(corpus: Corpus) -> dict[str, Any]:
    """The request of a corpus, as a plain mapping in the file's key order."""
    config = corpus.config
    lexicon = corpus.planner.lexicon
    lexemes = []
    for lexeme in lexicon.content_lexemes:
        record = {"label": lexeme.label, "concept": lexeme.concept, "pos": lexeme.pos}
        if lexeme.same_form_as is not None:
            record["same_form_as"] = lexeme.same_form_as
        lexemes.append(record)
    frequencies = corpus.stats["lexeme_frequencies"]
    function = sorted(
        enumerate(lexicon.function_lexemes), key=lambda item: (-frequencies[item[1].label], item[0])
    )
    token_lists = [s.sentence.tokens for d in corpus.documents for s in d.sentences]
    token_lists += [
        item.input["tokens"] for test_set in corpus.test_sets for item in test_set.items
    ]
    found = inflected(token_lists)
    affixes = affix_items(config)
    return {
        "lexemes": lexemes,
        "takes": takes(config),
        "function_words": [lexeme.gloss for _, lexeme in function],
        "affixes": affixes,
        "inflect": [
            {"lexemes": sorted(found[item["gloss"]], key=_number), "affixes": [item["gloss"]]}
            for item in affixes
            if item["gloss"] in found
        ],
        "meanings": MEANINGS_FILE,
    }


def meanings_csv(world: World) -> str:
    """The categories' meaning vectors, as the taxonomy generator writes them to
    ``categories_generative.csv``, with the world's labels."""
    return world.meanings_csv()
