"""Stage 1: the formal rendering of a sequence of lexemes."""

from __future__ import annotations

from corpus_support import TINY_TAXONOMY, corpus_config

from semantic_world.corpus import Streams, build_lexicon, formal
from semantic_world.corpus.renderings import formal_word


def lexicon_of(world, **knobs):
    config = corpus_config(TINY_TAXONOMY, lexicon=knobs)
    return build_lexicon(config, world, Streams(config.seed))


def test_formal_rendering(tiny_world) -> None:
    lexicon = lexicon_of(tiny_world)
    the = lexicon.function_word("the")
    (noun,) = lexicon.lexemes_of("C1.1")
    (verb,) = lexicon.lexemes_of("CAN.2")
    # a content lexeme's gloss is its concept's label, and a function word's is its English gloss
    assert formal([the, noun, verb]) == "the/L.37 C1.1/L.2 CAN.2/L.24"
    assert formal_word(lexicon.function_word("no")) == "no/L.41"
    assert formal([]) == ""
    for lexeme in lexicon.lexemes:
        gloss, label = formal_word(lexeme).split("/")
        assert label == lexeme.label and lexicon.lexeme(label) is lexeme
        assert gloss == (lexeme.concept if lexeme.content else lexeme.concept.lower())


def test_the_formal_rendering_tells_synonyms_and_homonyms_apart(tiny_world) -> None:
    lexicon = lexicon_of(tiny_world, synonym_rate=1.0, homonym_rate=1.0)
    first, second = lexicon.lexemes_of("C1.1")
    # synonyms share the gloss, and differ in the lexeme label
    assert formal([first]) == "C1.1/L.2" and formal([second]).startswith("C1.1/L.")
    assert formal([first]) != formal([second])
    # a homonym's two lexemes differ in both
    for earlier, later in lexicon.homonym_pairs():
        a, b = formal_word(earlier).split("/"), formal_word(later).split("/")
        assert a[0] != b[0] and a[1] != b[1]
