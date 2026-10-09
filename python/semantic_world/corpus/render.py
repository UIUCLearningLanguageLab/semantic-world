"""Attaching word forms to a corpus run: the ``render`` command.

The corpus is generated without word forms. The word-form pipeline then makes the forms from the
corpus's request (``wordform_request.yaml``), and ``render`` reads that word-form run and fills
in the corpus run:

- the word labels (``words``) and the spelled rendering (``text``) of every sentence in
  ``documents.jsonl`` and of every item in ``tests/``;
- ``corpus.txt``, which then holds the spelled rendering;
- the word-form columns of ``lexicon.csv`` (``word`` and ``spelling``);
- ``provenance.wordforms`` in ``config.yaml``: the word-form run's identity.

Nothing else in the folder changes, byte for byte. Rendering again, with the same word-form run
or with another one made from the same request, replaces what the earlier rendering wrote.

A content lexeme's form is the one that the word-form run assigned to it
(``assignment/lexicon.csv``). A function word's form is the run's function word with the same
gloss. An inflected token (``L.5-PLURAL``) is the inflected form of its lexeme's form with the
affix of that gloss (``WORD.12.AFFIX.1``, or ``WORD.12.MARKER.2.AFFIX.1`` for a marked
form).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from semantic_world.corpus.errors import CorpusError
from semantic_world.corpus.io import TESTS_FOLDER, _json_lines, _yaml, corpus_text
from semantic_world.corpus.lexicon import FUNCTION_WORD, LEXICON_COLUMNS
from semantic_world.corpus.realize import token_parts

SPELLED_FILE = "corpus.txt"


def _rows(path: Path, what: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise CorpusError(f"{path} is missing: {what}")
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def wordform_identity(folder: str | Path) -> dict[str, Any]:
    """The identity of a word-form run, for the corpus run's ``config.yaml``: its name, its
    seed, and the SHA-256 of its resolved configuration as YAML, without the provenance. The
    folder's path is left out, so a corpus run does not depend on where the word forms lie."""
    path = Path(folder) / "config.yaml"
    if not path.is_file():
        raise CorpusError(f"{path} is missing: {folder} is not a word-form run folder")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("provenance", None)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return {
        "name": data.get("name"),
        "seed": data.get("seed"),
        "config_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class WordForms:
    """The forms of a word-form run that a corpus needs: the form of every lexeme, and the
    inflected forms."""

    def __init__(self, folder: str | Path, lexicon: list[dict[str, str]]) -> None:
        folder = Path(folder)
        words = _rows(folder / "words.csv", "the folder is not a word-form run folder")
        self.spelling = {row["label"]: row["spelling"] for row in words}
        assigned = _rows(
            folder / "assignment" / "lexicon.csv",
            "the word-form run assigned no words. Make it from the corpus's request "
            "(request: in its configuration)",
        )
        if assigned and "lexeme" not in assigned[0]:
            raise CorpusError(
                f"{folder / 'assignment' / 'lexicon.csv'} has no lexeme column: the word-form "
                f"run was not made from a request with lexemes"
            )
        by_lexeme = {row["lexeme"]: row for row in assigned}
        function = {row["gloss"]: row["label"] for row in words if row["kind"] == "function"}
        affixes = folder / "affixes.csv"
        self.affix = (
            {row["gloss"]: row["label"] for row in _rows(affixes, "")}
            if (affixes.is_file())
            else {}
        )
        self.word: dict[str, str] = {}
        content = {row["label"] for row in lexicon if row["pos"] != FUNCTION_WORD}
        other = sorted(set(by_lexeme) - content)
        if other:
            raise CorpusError(
                f"the word-form run {folder} has the lexeme {other[0]}, which the corpus does "
                f"not have: the run was made from another corpus's request"
            )
        for row in lexicon:
            label = row["label"]
            if row["pos"] == FUNCTION_WORD:
                word = function.get(row["gloss"])
                if word is None:
                    raise CorpusError(
                        f"the word-form run {folder} has no function word with the gloss "
                        f"{row['gloss']!r} ({label}): the run was not made from this corpus's "
                        f"request"
                    )
            else:
                found = by_lexeme.get(label)
                if found is None or found["meaning"] != row["concept"]:
                    raise CorpusError(
                        f"the word-form run {folder} has no form for the lexeme {label} "
                        f"({row['concept']}): the run was not made from this corpus's request"
                    )
                word = found["word"]
            if word not in self.spelling:
                raise CorpusError(f"{folder}: words.csv has no word {word} (lexeme {label})")
            self.word[label] = word
        self.folder = folder

    def form(self, token: str) -> str:
        """The word label of a token: its lexeme's form, or the inflected form of that form."""
        label, gloss = token_parts(token)
        word = self.word[label]
        if gloss is None:
            return word
        affix = self.affix.get(gloss)
        form = f"{word}.{affix}"
        if affix is None or form not in self.spelling:
            raise CorpusError(
                f"the word-form run {self.folder} has no inflected form of {word} with {gloss} "
                f"(token {token}): the run was not made from this corpus's request"
            )
        return form

    def render(self, sentence: dict[str, Any]) -> None:
        """Fill in the word labels and the spelled rendering of a sentence's record."""
        words = [self.form(token) for token in sentence["tokens"]]
        sentence["words"] = words
        sentence["text"] = " ".join(self.spelling[word] for word in words)


def render(run: str | Path, wordforms: str | Path) -> dict[str, Any]:
    """Attach the word forms of the word-form run ``wordforms`` to the corpus run ``run`` (see
    the module's description), and return a short report."""
    run = Path(run)
    lexicon_path = run / "lexicon.csv"
    lexicon = _rows(lexicon_path, f"{run} is not a corpus run folder")
    forms = WordForms(wordforms, lexicon)
    identity = wordform_identity(wordforms)

    documents_path = run / "documents.jsonl"
    if not documents_path.is_file():
        raise CorpusError(f"{documents_path} is missing: {run} is not a corpus run folder")
    documents = [
        json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()
    ]
    sentences = 0
    for document in documents:
        for sentence in document["sentences"]:
            forms.render(sentence)
            sentences += 1
    test_sets: dict[Path, list[dict[str, Any]]] = {}
    for path in sorted((run / TESTS_FOLDER).glob("*.jsonl")):
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for item in items:
            forms.render(item["input"])
        test_sets[path] = items

    # Everything is rendered before anything is written, so an error leaves the folder as it was.
    documents_path.write_text(_json_lines(documents), encoding="utf-8")
    (run / SPELLED_FILE).write_text(corpus_text(documents, "text"), encoding="utf-8")
    for path, items in test_sets.items():
        path.write_text(_json_lines(items), encoding="utf-8")
    with lexicon_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LEXICON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in lexicon:
            word = forms.word[row["label"]]
            writer.writerow({**row, "word": word, "spelling": forms.spelling[word]})
    config_path = run / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["provenance"]["wordforms"] = identity
    config_path.write_text(_yaml(config), encoding="utf-8")
    return {
        "documents": len(documents),
        "sentences": sentences,
        "test_items": sum(len(items) for items in test_sets.values()),
        "lexemes": len(lexicon),
        "wordforms": identity,
    }
