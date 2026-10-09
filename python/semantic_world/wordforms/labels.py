"""The labels of the word-form pipeline (``docs/specs/WORLD_AND_LANGUAGE.md``, "Labels").

Every label is a capitalized word for the kind of object, a period, and an index, with indices
from 1:

- a content word ``WORD.<n>``, a speaker ``SPEAKER.<n>``, and a token (one recording)
  ``WORD.<n>.SPEAKER.<m>.TOKEN.<k>``;
- a function word ``FUNCWORD.<n>``, an affix ``AFFIX.<n>``, and an inflected form
  ``WORD.<n>.AFFIX.<m>``;
- a branch marker ``MARKER.<k>``, a marked form ``WORD.<n>.MARKER.<k>``, and an inflected marked
  form ``WORD.<n>.MARKER.<k>.AFFIX.<m>``;
- an augmented token ``<source token>.AUGMENTED.<recipe number>``, and a token changed by
  acoustic mapping ``<source token>.MAPPED``.

The tokens of function words, inflected forms, and marked forms extend their labels as a word's
do (``FUNCWORD.2.SPEAKER.3.TOKEN.1``). Stage a6 of the world-and-language refactor replaced the
old forms (``W.12``, ``S.3``, ``W.12.S.3.2``, ``F.2``, ``AF.1``, ``M.2``, ``.A.1``, ``.M``).
"""

from __future__ import annotations

WORD_PREFIX = "WORD"
SPEAKER_PREFIX = "SPEAKER"
TOKEN_PREFIX = "TOKEN"
FUNCTION_WORD_PREFIX = "FUNCWORD"
AFFIX_PREFIX = "AFFIX"
MARKER_PREFIX = "MARKER"
AUGMENTED_PREFIX = "AUGMENTED"
MAPPED_SUFFIX = ".MAPPED"
"""The end of the label of a token changed by acoustic mapping
(``WORD.12.SPEAKER.3.TOKEN.2.MAPPED``)."""


def word_label(number: int) -> str:
    """``WORD.<n>``."""
    return f"{WORD_PREFIX}.{number}"


def speaker_label(number: int) -> str:
    """``SPEAKER.<n>``."""
    return f"{SPEAKER_PREFIX}.{number}"


def token_label(word: str, speaker: str, number: int) -> str:
    """``<word>.<speaker>.TOKEN.<k>``: token ``k`` of a form by a speaker."""
    return f"{word}.{speaker}.{TOKEN_PREFIX}.{number}"


def function_word_label(number: int) -> str:
    """``FUNCWORD.<n>``."""
    return f"{FUNCTION_WORD_PREFIX}.{number}"


def affix_label(number: int) -> str:
    """``AFFIX.<n>``."""
    return f"{AFFIX_PREFIX}.{number}"


def marker_label(number: int) -> str:
    """``MARKER.<k>``: the marker of the ``k``-th branch."""
    return f"{MARKER_PREFIX}.{number}"


def joined_label(stem: str, affix: str) -> str:
    """The label of a stem joined to an affix or a marker: ``WORD.12.AFFIX.1``,
    ``WORD.12.MARKER.2``, ``WORD.12.MARKER.2.AFFIX.1``."""
    return f"{stem}.{affix}"


def augmented_label(token: str, recipe: int) -> str:
    """``<source token>.AUGMENTED.<recipe number>``."""
    return f"{token}.{AUGMENTED_PREFIX}.{recipe}"


def mapped_label(token: str) -> str:
    """``<source token>.MAPPED``."""
    return f"{token}{MAPPED_SUFFIX}"


def word_number(label: str) -> int | None:
    """The index of a content word's label (``WORD.<n>``), or None for any other string."""
    prefix, _, number = label.partition(".")
    if prefix != WORD_PREFIX or not number.isdecimal():
        return None
    return int(number)


def token_number(label: str) -> int:
    """The token number of a synthesized or mapped token's label (the index after ``TOKEN``)."""
    parts = label.removesuffix(MAPPED_SUFFIX).rsplit(".", 2)
    if len(parts) != 3 or parts[1] != TOKEN_PREFIX or not parts[2].isdecimal():
        raise ValueError(f"{label!r} is not the label of a token")
    return int(parts[2])
