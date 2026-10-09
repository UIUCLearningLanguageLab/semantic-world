"""Stage 7 acceptance tests: the request for the word-form pipeline, and the ``render`` command.

``generate`` writes the request. The word-form pipeline makes the forms from it, and ``render``
attaches them. Every lexeme gets a word form, distinct lexemes get distinct forms unless they are
homonyms, and category lexemes follow the assignment mode. Function words are ordered by their
corpus counts. Only the inflections that the corpus uses are made. ``corpus.txt`` matches the
spelled renderings, and rendering leaves the rest of the output folder byte-identical. With
branch markers and plural affixes, marked forms are inflected. The words of the lexemes never
depend on which inflected forms the documents or the test items happen to use.

The word-form pipeline runs up to its word forms here, which needs no audio. The whole chain
with audio runs in the word-form pipeline's own suite and in the stage's report.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl
import pytest
import yaml

from semantic_world.corpus import CorpusError, render, wordform_request
from semantic_world.corpus.__main__ import main
from semantic_world.corpus.generate import generate
from semantic_world.corpus.io import corpus_text
from semantic_world.corpus.lexicon import LEXICON_COLUMNS
from semantic_world.corpus.render import wordform_identity
from semantic_world.corpus.request import MEANINGS_FILE, REQUEST_FILE, affix_items, takes

pytest.importorskip("cmudict", reason="the cmudict package is not installed (the 'speech' extra)")
pytest.importorskip("wordfreq", reason="the wordfreq package is not installed (the 'speech' extra)")

from semantic_world.wordforms import run_forms  # noqa: E402
from semantic_world.wordforms.__main__ import main as wordforms_main  # noqa: E402
from semantic_world.wordforms.config import parse_config  # noqa: E402

NUMBER = {"grammar": {"morphology": {"number": {"enabled": True}}}}
INFLECTED = {
    "grammar": {
        "morphology": {
            "number": {"enabled": True},
            "tense": {"enabled": True},
            "aspect": {"enabled": True},
        }
    }
}
"""Number and tense as affixes, and aspect as a word (the default realizations)."""
AMBIGUOUS = {"lexicon": {"synonym_rate": 0.3, "homonym_rate": 0.3}}
WORD_COUNTS = {"tiny": 80, "default": 320}


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def files_of(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def wordform_config(request: Path, name: str = "tiny", seed: int = 1, **assignment):
    """A word-form configuration for a corpus's request."""
    data = {
        "name": "corpus_forms",
        "request": str(request),
        "wordforms": {"count": WORD_COUNTS[name]},
        "assignment": {"null_samples": 20, **assignment},
    }
    return parse_config(data, "corpus_forms", seed=seed)


@pytest.fixture(scope="module")
def chains(cases, tmp_path_factory):
    """The chain of a corpus run: generate, make the word forms, and render. Each chain is made
    once, and gives the corpus, its folder before rendering (as bytes), the folder, and the
    word-form run's folder."""
    root = tmp_path_factory.mktemp("chains")
    made: dict = {}

    def chain(name: str = "tiny", count: int = 30, mode: str = "arbitrary", **sections):
        key = (name, count, mode, json.dumps(sections, sort_keys=True))
        if key not in made:
            case = cases(name)
            settings = {"test_sets": {"size": 10}, **sections}
            settings["documents"] = {**settings.get("documents", {}), "count": count}
            corpus = generate(case.config(**settings), case.result)
            folder = corpus.write(root / f"corpus{len(made)}")
            before = files_of(folder)
            config = wordform_config(folder / REQUEST_FILE, name, mode=mode)
            forms = run_forms(config).write(root / f"forms{len(made)}")
            report = render(folder, forms)
            made[key] = (corpus, before, folder, forms, report)
        return made[key]

    return chain


# ---------------------------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "default"])
def test_the_request_lists_what_the_corpus_needs(cases, chains, name) -> None:
    corpus, before, folder, _, _ = chains(name, **INFLECTED)
    request = yaml.safe_load(before[REQUEST_FILE])
    assert request == wordform_request(corpus)
    assert list(request) == ["lexemes", "takes", "function_words", "affixes", "inflect", "meanings"]
    lexicon = corpus.planner.lexicon
    # every content lexeme, with its concept and its part of speech
    assert [x["label"] for x in request["lexemes"]] == [x.label for x in lexicon.content_lexemes]
    for entry, lexeme in zip(request["lexemes"], lexicon.content_lexemes, strict=True):
        assert (entry["concept"], entry["pos"]) == (lexeme.concept, lexeme.pos)
        assert "same_form_as" not in entry
    # every function word of the language, most frequent first, by its count in the documents
    function = {x.gloss: x.label for x in lexicon.function_lexemes}
    assert sorted(request["function_words"]) == sorted(function)
    assert {"PROGRESSIVE", "are", "have"} <= set(function)
    counts = Counter(
        token.split("-")[0] for d in corpus.documents for s in d.sentences for token in s.sentence.tokens
    )  # fmt: skip
    frequencies = [counts[function[gloss]] for gloss in request["function_words"]]
    assert frequencies == sorted(frequencies, reverse=True) and frequencies[0] > frequencies[-1]
    glosses = [x.gloss for x in lexicon.function_lexemes]
    for a, b in zip(request["function_words"], request["function_words"][1:], strict=False):
        if counts[function[a]] == counts[function[b]]:  # ties keep the lexicon's order
            assert glosses.index(a) < glosses.index(b)
    # the affixes and the requirements come from the grammar settings
    assert request["affixes"] == [
        {"gloss": "PLURAL", "position": "suffix"},
        {"gloss": "PAST", "position": "suffix"},
    ]
    assert request["takes"] == {
        "noun": ["PLURAL"],
        "intransitive_verb": ["PLURAL", "PAST"],
        "transitive_verb": ["PLURAL", "PAST"],
    }
    # inflect: the lexemes that appear inflected, in a document or in a test item
    found: dict[str, set[str]] = {}
    token_lists = [s["tokens"] for d in lines(folder / "documents.jsonl") for s in d["sentences"]]
    in_documents = sum(len(tokens) for tokens in token_lists)
    for path in sorted((folder / "tests").glob("*.jsonl")):
        token_lists += [item["input"]["tokens"] for item in lines(path)]
    assert sum(len(tokens) for tokens in token_lists) > in_documents
    for tokens in token_lists:
        for token in tokens:
            if "-" in token:
                label, gloss = token.split("-")
                found.setdefault(gloss, set()).add(label)
    assert [entry["affixes"] for entry in request["inflect"]] == [["PLURAL"], ["PAST"]]
    pos = {x.label: x.pos for x in lexicon.lexemes}
    for entry in request["inflect"]:
        (gloss,) = entry["affixes"]
        assert set(entry["lexemes"]) == found[gloss]
        assert entry["lexemes"] == sorted(entry["lexemes"], key=lambda x: int(x.split(".")[1]))
        assert all(gloss in request["takes"][pos[label]] for label in entry["lexemes"])
    # the meanings: the taxonomy's categories_generative.csv with the labels of the world,
    # beside the request
    assert request["meanings"] == MEANINGS_FILE
    assert before[MEANINGS_FILE] == cases(name).world.meanings_csv().encode("utf-8")


def test_the_request_of_a_language_without_inflection(chains) -> None:
    corpus, before, _, _, _ = chains("tiny")
    request = yaml.safe_load(before[REQUEST_FILE])
    assert request["takes"] == {} and request["affixes"] == [] and request["inflect"] == []
    assert len(request["function_words"]) == 15
    assert len(request["lexemes"]) == len(corpus.planner.lexicon.content_lexemes)


@pytest.mark.parametrize(
    ("morphology", "affixes", "expected"),
    [
        ({}, [], {}),
        ({"number": {"enabled": True}}, [("PLURAL", "suffix")], {"noun": ["PLURAL"], "verb": ["PLURAL"]}),
        ({"number": {"enabled": True, "agreement": False}}, [("PLURAL", "suffix")], {"noun": ["PLURAL"]}),
        ({"number": {"enabled": True, "realization": "word"}}, [], {}),
        ({"number": {"enabled": True, "position": "before"}}, [("PLURAL", "prefix")], {"noun": ["PLURAL"], "verb": ["PLURAL"]}),
        ({"tense": {"enabled": True}}, [("PAST", "suffix")], {"verb": ["PAST"]}),
        ({"aspect": {"enabled": True, "realization": "affix", "position": "before"}}, [("PROGRESSIVE", "prefix")], {"verb": ["PROGRESSIVE"]}),
        (
            {"number": {"enabled": True}, "tense": {"enabled": True, "realization": "word"}, "aspect": {"enabled": True, "realization": "affix"}},
            [("PLURAL", "suffix"), ("PROGRESSIVE", "suffix")],
            {"noun": ["PLURAL"], "verb": ["PLURAL", "PROGRESSIVE"]},
        ),
    ],
)  # fmt: skip
def test_takes_and_affixes_come_from_the_grammar_settings(cases, morphology, affixes, expected):
    """Which lexemes must take which affixes comes from the grammar settings alone, by part of
    speech: nouns with number as an affix, and verbs with agreement, tense, or aspect."""
    config = cases("tiny").config(grammar={"morphology": morphology})
    assert affix_items(config) == [{"gloss": g, "position": p} for g, p in affixes]
    wanted = {}
    if "noun" in expected:
        wanted["noun"] = expected["noun"]
    for pos in ("intransitive_verb", "transitive_verb"):
        if "verb" in expected:
            wanted[pos] = expected["verb"]
    assert takes(config) == wanted


def test_homonyms_and_synonyms_in_the_request(chains) -> None:
    corpus, before, _, _, _ = chains("default", 30, **AMBIGUOUS)
    request = yaml.safe_load(before[REQUEST_FILE])
    lexicon = corpus.planner.lexicon
    sharing = {x["label"]: x["same_form_as"] for x in request["lexemes"] if "same_form_as" in x}
    assert sharing == {b.label: a.label for a, b in lexicon.homonym_pairs()} and len(sharing) > 10
    concepts = Counter(x["concept"] for x in request["lexemes"])
    assert sum(n == 2 for n in concepts.values()) > 20


# ---------------------------------------------------------------------------------------------
# Acceptance: every lexeme gets a word form, and the corpus is rendered
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mode", "sections"),
    [
        ("tiny", "arbitrary", {}),
        ("tiny", "branch_markers", NUMBER),
        ("default", "target_correlation", {**INFLECTED, **AMBIGUOUS}),
    ],
)
def test_generate_make_word_forms_and_render(chains, name, mode, sections) -> None:
    corpus, before, folder, forms, report = chains(name, mode=mode, **sections)
    lexicon = corpus.planner.lexicon
    words = {row["label"]: row for row in pl.read_csv(forms / "words.csv").iter_rows(named=True)}
    assigned = pl.read_csv(forms / "assignment" / "lexicon.csv")
    rows = pl.read_csv(folder / "lexicon.csv", infer_schema_length=None)
    assert rows.columns == list(LEXICON_COLUMNS) and rows.height == len(lexicon.lexemes)
    form_of = dict(zip(rows["label"].to_list(), rows["word"].to_list(), strict=True))
    # every lexeme gets a word form
    assert all(form_of[x.label] in words for x in lexicon.lexemes)
    assert rows["spelling"].to_list() == [
        words[form_of[x.label]]["spelling"] for x in lexicon.lexemes
    ]
    # distinct lexemes get distinct forms, unless they are homonyms
    leaders = [x for x in lexicon.lexemes if x.same_form_as is None]
    assert len({form_of[x.label] for x in leaders}) == len(leaders)
    assert len({words[form_of[x.label]]["arpabet"] for x in leaders}) == len(leaders)
    for lexeme in lexicon.lexemes:
        if lexeme.same_form_as is not None:
            assert form_of[lexeme.label] == form_of[lexeme.same_form_as]
    # function words are the run's function words, by gloss, in the order of the request
    request = yaml.safe_load(before[REQUEST_FILE])
    function = [row for row in words.values() if row["kind"] == "function"]
    assert [row["gloss"] for row in function] == request["function_words"]
    for lexeme in lexicon.function_lexemes:
        assert words[form_of[lexeme.label]]["gloss"] == lexeme.gloss
    # category lexemes follow the assignment mode
    by_lexeme = {row["lexeme"]: row for row in assigned.iter_rows(named=True)}
    categories = set(corpus.planner.world.categories)
    for lexeme in lexicon.content_lexemes:
        row = by_lexeme[lexeme.label]
        assert row["meaning"] == lexeme.concept and row["word"] == form_of[lexeme.label]
        if lexeme.same_form_as is not None:
            assert row["assigned"] == "same_form"
        else:
            assert row["assigned"] == (mode if lexeme.concept in categories else "random")
    summary = yaml.safe_load((forms / "assignment" / "summary.yaml").read_text())
    assert summary["mode"] == mode and summary["lexemes"]["count"] == len(by_lexeme)
    if mode == "target_correlation":
        assert summary["target_correlation"]["reached"]
    # only the inflections that the corpus uses are made
    affix = {row["gloss"]: row["label"] for row in pl.read_csv(forms / "affixes.csv").iter_rows(named=True)}  # fmt: skip
    wanted = {
        f"{form_of[label]}.{affix[gloss]}"
        for entry in request["inflect"]
        for label in entry["lexemes"]
        for gloss in entry["affixes"]
    }
    assert {label for label, row in words.items() if row["kind"] == "inflected"} == wanted
    assert bool(wanted) == bool(sections)

    # the sentences: word labels, and the spelled rendering
    documents = lines(folder / "documents.jsonl")
    sentences = [s for d in documents for s in d["sentences"]]
    items = [item for path in sorted((folder / "tests").glob("*.jsonl")) for item in lines(path)]
    assert report["sentences"] == len(sentences) and report["test_items"] == len(items)
    for record in sentences + [item["input"] for item in items]:
        assert len(record["words"]) == len(record["tokens"])
        for token, word in zip(record["tokens"], record["words"], strict=True):
            label, _, gloss = token.partition("-")
            assert word == (f"{form_of[label]}.{affix[gloss]}" if gloss else form_of[label])
        assert record["text"] == " ".join(words[word]["spelling"] for word in record["words"])
    # corpus.txt matches the spelled renderings
    assert (folder / "corpus.txt").read_text(encoding="utf-8") == corpus_text(documents, "text")
    assert (folder / "corpus.txt").read_bytes() != before["corpus.txt"]


@pytest.mark.parametrize(
    ("name", "mode", "sections"),
    [("tiny", "branch_markers", NUMBER), ("default", "target_correlation", {**INFLECTED, **AMBIGUOUS})],
)  # fmt: skip
def test_rendering_leaves_the_rest_of_the_folder_byte_identical(chains, name, mode, sections):
    _, before, folder, forms, report = chains(name, mode=mode, **sections)
    after = files_of(folder)
    assert list(after) == list(before)
    changed = {name for name in before if after[name] != before[name]}
    tests = {name for name in before if name.startswith("tests/") and before[name]}
    assert changed == {"config.yaml", "corpus.txt", "documents.jsonl", "lexicon.csv"} | tests

    # documents.jsonl and the test sets: only the word labels and the spelled rendering
    def cleared(text: bytes, items: bool) -> str:
        out = []
        for line in text.decode("utf-8").splitlines():
            record = json.loads(line)
            for sentence in [record["input"]] if items else record["sentences"]:
                assert sentence["words"] is not None and sentence["text"] is not None
                sentence["words"] = sentence["text"] = None
            out.append(json.dumps(record, ensure_ascii=False) + "\n")
        return "".join(out)

    assert cleared(after["documents.jsonl"], False) == before["documents.jsonl"].decode("utf-8")
    for test_set in tests:
        assert cleared(after[test_set], True) == before[test_set].decode("utf-8")
    # lexicon.csv: only the word-form columns
    old, new = (
        pl.read_csv(data, infer_schema_length=None, schema_overrides={"word": pl.String, "spelling": pl.String})
        for data in (before["lexicon.csv"], after["lexicon.csv"])
    )  # fmt: skip
    assert new.drop("word", "spelling").equals(old.drop("word", "spelling"))
    assert new["word"].null_count() == 0 and old["word"].null_count() == old.height
    blank = "".join(
        ",".join(v if c not in ("word", "spelling") else "" for c, v in zip(LEXICON_COLUMNS, line.split(","), strict=True)) + "\n"
        for line in after["lexicon.csv"].decode("utf-8").splitlines()[1:]
    )  # fmt: skip
    assert ",".join(LEXICON_COLUMNS) + "\n" + blank == before["lexicon.csv"].decode("utf-8")
    # config.yaml: only provenance.wordforms, the word-form run's identity
    config = yaml.safe_load(after["config.yaml"])
    identity = config["provenance"]["wordforms"]
    assert identity == report["wordforms"] == wordform_identity(forms)
    assert identity["name"] == "corpus_forms" and identity["seed"] == 1
    assert len(identity["config_hash"]) == 64
    assert after["config.yaml"].decode("utf-8").replace(
        yaml.safe_dump({"wordforms": identity}, sort_keys=False, default_flow_style=None).replace("\n", "\n  ").rstrip(),
        "wordforms: null",
    ) == before["config.yaml"].decode("utf-8")  # fmt: skip
    # rendering again changes nothing
    assert render(folder, forms) == report and files_of(folder) == after


def test_marked_forms_are_inflected(chains) -> None:
    """With branch markers and plural affixes, the plural of a category's noun is the inflected
    form of its marked form: ``W.12.M.2.AF.1``."""
    corpus, _, folder, forms, _ = chains("tiny", mode="branch_markers", **NUMBER)
    words = {row["label"]: row for row in pl.read_csv(forms / "words.csv").iter_rows(named=True)}
    lexicon = {row["label"]: row for row in pl.read_csv(folder / "lexicon.csv").iter_rows(named=True)}  # fmt: skip
    categories = set(corpus.planner.world.categories)
    markers = pl.read_csv(forms / "assignment" / "markers.csv")
    assert markers.height == 2  # one for each top-level branch
    # the two markers differ in both phonemes, so the branches can be told apart when spoken
    first, second = (arpabet.split() for arpabet in markers["arpabet"].to_list())
    assert len(first) == len(second) == 2
    assert all(a.rstrip("012") != b.rstrip("012") for a, b in zip(first, second, strict=True))
    marker_of = dict(zip(markers["branch"].to_list(), markers["label"].to_list(), strict=True))
    for row in lexicon.values():
        form = words[row["word"]]
        if row["concept"] in categories:
            assert form["kind"] == "marked" and words[form["stem"]]["kind"] == "content"
            assert form["affix"] == marker_of[".".join(row["concept"].split(".")[:2])]
        else:
            assert form["kind"] in ("content", "function")
    plurals = marked_plurals = 0
    for document in lines(folder / "documents.jsonl"):
        for sentence in document["sentences"]:
            for token, word in zip(sentence["tokens"], sentence["words"], strict=True):
                if not token.endswith("-PLURAL"):
                    continue
                plurals += 1
                form = words[word]
                assert (
                    form["kind"] == "inflected"
                    and form["stem"] == lexicon[token.split("-")[0]]["word"]
                )
                if words[form["stem"]]["kind"] == "marked":
                    marked_plurals += 1
                    assert word.count(".M.") == 1 and word.endswith(".AF.1")
                    assert form["spelling"].startswith(words[form["stem"]]["spelling"])
    assert marked_plurals > 20 and plurals > marked_plurals  # verbs agree, and are unmarked


# ---------------------------------------------------------------------------------------------
# The words of the lexemes do not depend on the documents
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["arbitrary", "branch_markers"])
def test_documents_and_test_sets_change_no_lexemes_word(cases, tmp_path, mode) -> None:
    """The request's inflect list and the order of its function words follow the documents and
    the test items. The word of a lexeme must not: its affix requirement comes from the grammar
    settings alone, and the function words are made after the assignment."""
    case = cases("default")
    made = []
    for index, (count, size) in enumerate(((8, 0), (60, 40))):
        settings = {**INFLECTED, "documents": {"count": count}, "test_sets": {"size": size}}
        folder = generate(case.config(**settings), case.result).write(tmp_path / f"corpus{index}")
        request = yaml.safe_load((folder / REQUEST_FILE).read_text(encoding="utf-8"))
        run = run_forms(wordform_config(folder / REQUEST_FILE, "default", mode=mode))
        made.append((request, run, run.write(tmp_path / f"forms{index}")))
    (small, small_run, small_forms), (large, large_run, large_forms) = made
    # the two corpora ask for different inflected forms, and order their function words apart
    assert small["inflect"] != large["inflect"]
    assert small["function_words"] != large["function_words"]
    assert sum(w.kind == "inflected" for w in small_run.lexicon.words) < sum(
        w.kind == "inflected" for w in large_run.lexicon.words
    )
    for key in ("lexemes", "takes", "affixes", "meanings"):
        assert small[key] == large[key]
    # every lexeme has the same word in both runs
    for name in ("assignment/lexicon.csv", "affixes.csv"):
        assert (small_forms / name).read_bytes() == (large_forms / name).read_bytes(), name
    for kind in ("content", "marked"):
        assert [w.record() for w in small_run.lexicon.words if w.kind == kind] == [
            w.record() for w in large_run.lexicon.words if w.kind == kind
        ]
    # and an inflected form that both corpora use is the same form
    forms = [{w.label: w.arpabet for w in run.lexicon.words if w.kind == "inflected"} for _, run, _ in made]  # fmt: skip
    shared = set(forms[0]) & set(forms[1])
    assert shared and all(forms[0][label] == forms[1][label] for label in shared)


# ---------------------------------------------------------------------------------------------
# The render command, and its errors
# ---------------------------------------------------------------------------------------------


def test_the_generate_wordforms_and_render_commands(tmp_path, capsys) -> None:
    """The three steps on the tiny configuration, with the word-form configuration that was made
    for the tiny corpus (word forms only: the audio layers run in the stage's report)."""
    out = tmp_path / "corpus"
    assert main(["generate", "data/corpus/tiny.yaml", "--out", str(out)]) == 0
    assert (out / REQUEST_FILE).exists() and (out / MEANINGS_FILE).exists()
    data = yaml.safe_load(Path("data/wordforms/corpus_tiny.yaml").read_text(encoding="utf-8"))
    assert data["request"] == "runs/corpus/tiny_seed1/wordform_request.yaml"
    data["request"] = str(out / REQUEST_FILE)
    config = tmp_path / "corpus_tiny.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    forms = tmp_path / "forms"
    assert wordforms_main(["forms", str(config), "--out", str(forms)]) == 0
    capsys.readouterr()
    assert main(["render", str(out), "--wordforms", str(forms)]) == 0
    printed = capsys.readouterr().out
    assert "rendered" in printed and "the word forms of corpus_tiny (seed 1)" in printed
    text = (out / "corpus.txt").read_text(encoding="utf-8")
    assert "/LEXEME." not in text and len(text.split()) == 1075
    spellings = set(pl.read_csv(forms / "words.csv")["spelling"].to_list())
    assert set(text.split()) <= spellings
    # the word forms of tiny.yaml are too few for the tiny corpus, as the specification says
    data["wordforms"] = {"count": 20}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert wordforms_main(["forms", str(config), "--out", str(forms)]) == 1
    assert "need distinct forms" in capsys.readouterr().err


def test_the_word_form_configuration_for_the_default_corpus() -> None:
    """``data/wordforms/corpus_default.yaml`` has the default word-form settings, and its 500
    words are enough for the default corpus's lexemes."""
    from semantic_world.corpus import Streams, build_lexicon, load_config, load_world

    data = yaml.safe_load(Path("data/wordforms/corpus_default.yaml").read_text(encoding="utf-8"))
    assert data["request"] == "runs/corpus/default_seed1/wordform_request.yaml"
    assert set(data) == {"name", "seed", "request", "wordforms", "assignment"}
    config = parse_config({**data, "request": None}, "corpus_default")
    defaults = parse_config({}, "defaults")
    assert config.wordforms == defaults.wordforms and config.wordforms.count == 500
    assert config.synthesis == defaults.synthesis and config.embeddings == defaults.embeddings
    assert len(config.embeddings) == 5 and config.assignment == defaults.assignment
    corpus = load_config("data/corpus/default.yaml")
    assert (corpus.name, corpus.seed) == ("default", 1)  # the run folder that the request is in
    lexicon = build_lexicon(corpus, load_world(corpus), Streams(corpus.seed))
    content = lexicon.content_lexemes
    assert len(content) == 173 <= config.wordforms.count
    assert sum(x.same_form_as is None for x in content) == 173
    assert len(lexicon.function_lexemes) == 15


def test_render_errors_leave_the_folder_unchanged(cases, chains, tmp_path, capsys) -> None:
    corpus, _, _, forms, _ = chains("tiny")
    folder = corpus.write(tmp_path / "corpus")
    before = files_of(folder)

    def fails(wordforms: Path, message: str) -> None:
        with pytest.raises(CorpusError, match=message):
            render(folder, wordforms)
        assert files_of(folder) == before

    fails(tmp_path / "nowhere", "is missing")
    # a word-form run without a request has assigned no lexemes
    plain = run_forms(parse_config({"wordforms": {"count": 20}}, "plain")).write(tmp_path / "plain")
    fails(plain, "assigned no words")
    # a run made from another corpus's request: more lexemes, or other function words
    other = chains("tiny", **AMBIGUOUS)[3]
    fails(other, "made from another corpus's request")
    agreeing = chains("tiny", **NUMBER)
    with pytest.raises(CorpusError, match="no function word with the gloss 'are'"):
        render(agreeing[0].write(tmp_path / "agreeing"), forms)
    # a run that lacks an inflected form the corpus uses
    with pytest.raises(CorpusError, match="no inflected form|no function word"):
        render(chains("tiny", mode="branch_markers", **NUMBER)[0].write(tmp_path / "marked"), forms)
    assert main(["render", str(folder), "--wordforms", str(plain)]) == 1
    assert "assigned no words" in capsys.readouterr().err
    with pytest.raises(CorpusError, match="not a corpus run folder"):
        render(tmp_path / "nowhere", forms)
    # the same corpus renders with another word-form run of its request
    again = run_forms(wordform_config(folder / REQUEST_FILE, seed=2)).write(tmp_path / "again")
    first = render(folder, forms)
    spelled = (folder / "corpus.txt").read_bytes()
    second = render(folder, again)
    assert second["wordforms"]["seed"] == 2 and first["wordforms"]["seed"] == 1
    assert (folder / "corpus.txt").read_bytes() != spelled
    assert render(folder, forms) == first and (folder / "corpus.txt").read_bytes() == spelled
