"""Renderings of a sentence. Stage 1 builds the formal rendering of a sequence of lexemes; the
conceptual, propositional, and spelled renderings come with the later stages.

The formal rendering writes every word as its gloss and its lexeme label: ``the/L.176 C1.3/L.5
CAN.2/L.138``. The gloss of a content lexeme is its concept's label, and the gloss of a function
word is its English gloss.
"""

from __future__ import annotations

from collections.abc import Iterable

from semantic_world.corpus.lexicon import Lexeme


def formal_word(lexeme: Lexeme) -> str:
    """One word of the formal rendering: ``<gloss>/<lexeme label>``."""
    return f"{lexeme.gloss}/{lexeme.label}"


def formal(lexemes: Iterable[Lexeme]) -> str:
    """The formal rendering of a sequence of lexemes, with one space between words."""
    return " ".join(formal_word(lexeme) for lexeme in lexemes)
