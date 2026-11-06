"""The guides' examples of the tiny corpus (Jon's ruling 5 of WM.76).

``python -m semantic_world.figures guides`` writes the examples of ``docs/guides/CORPUS.md``
from the rendered tiny run, between the figure markers, under the keys ``tiny.examples.<name>``.
Each example is chosen by a stated property (``figures.EXAMPLES``). The tests make the tiny
chain in a temporary folder (generate, word forms, render) and check that:

- every example is found, has the form its property guarantees, by a second reading of the
  run's files that does not use the script's own tests, and is the first with it where the
  second reading can tell;
- where an example shows renderings, they are the renderings of one sentence;
- the guides' markers are filled with the examples, ``--check`` reports them without writing,
  and a missing example fails as a missing figure does;
- every example key the guides quote is one the script writes, and every example is quoted.

The tests never read ``runs/``, and they pin no sentence of the run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from semantic_world.corpus import generate, render
from semantic_world.corpus import load_config as load_corpus_config
from semantic_world.corpus.request import REQUEST_FILE
from semantic_world.figures import (
    CORPUS_CONFIGS,
    EXAMPLE_CORPUS,
    EXAMPLE_SENTENCES,
    EXAMPLES,
    GUIDES,
    REPO,
    corpus_examples,
    fill_guides,
    guide_keys,
    lookup,
    main,
    missing_example,
)

pytest.importorskip("cmudict", reason="the cmudict package is not installed (the 'speech' extra)")
pytest.importorskip("wordfreq", reason="the wordfreq package is not installed (the 'speech' extra)")

from semantic_world.wordforms import run_forms  # noqa: E402
from semantic_world.wordforms.config import parse_config  # noqa: E402

NARRATIVES = ("entity", "situational")


@pytest.fixture(autouse=True)
def _run_from_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)


@pytest.fixture(scope="module")
def unrendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny corpus, generated and not rendered."""
    corpus = generate(load_corpus_config(REPO / CORPUS_CONFIGS[EXAMPLE_CORPUS]))
    return corpus.write(tmp_path_factory.mktemp("unrendered") / "corpus")


@pytest.fixture(scope="module")
def rendered(unrendered: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny corpus, rendered with word forms made from its own request (forms only), by
    the word-form configuration of the tiny corpus."""
    root = tmp_path_factory.mktemp("rendered")
    folder = Path(shutil.copytree(unrendered, root / "corpus"))
    data = yaml.safe_load((REPO / "data/wordforms/corpus_tiny.yaml").read_text(encoding="utf-8"))
    data["request"] = str(folder / REQUEST_FILE)
    data.setdefault("synthesis", {})["cache_dir"] = str(root / "cache")
    forms = run_forms(parse_config(data, "corpus_tiny", seed=1)).write(root / "forms")
    render(folder, forms)
    return folder


def _documents(folder: Path) -> list[dict]:
    text = (folder / "documents.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _items(folder: Path, name: str) -> list[dict]:
    text = (folder / "tests" / f"{name}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _plain(code: str) -> str:
    assert code.startswith("`") and code.endswith("`"), code
    return code[1:-1]


def _lines(block: str) -> list[str]:
    """The lines of a fenced block, as a guide holds it between markers on their own lines."""
    assert block.startswith("\n```\n") and block.endswith("\n```\n"), block
    return block[len("\n```\n") : -len("\n```\n")].split("\n")


def _sentence(folder: Path, label: str) -> dict:
    """The sentence a label names: a sentence of a document, or the true item of a pair of a
    test set (``<set> pair <n>``), with its rule from the item's metadata."""
    if " pair " in label:
        name, _, number = label.partition(" pair ")
        item = next(
            i
            for i in _items(folder, name)
            if i["meta"]["pair"] == int(number) and i["meta"]["truth"]
        )
        return {**item["input"], "rule": item["meta"].get("rule")}
    found = next(s for d in _documents(folder) for s in d["sentences"] if s["label"] == label)
    return {**found, "rule": found["logical_form"].get("rule")}


def _labeled(folder: Path, block: str) -> list[dict]:
    """The sentences of a block of labeled lines, each checked against its line's text."""
    found = []
    for line in _lines(block):
        label, _, text = line.partition("  ")
        sentence = _sentence(folder, label.strip())
        assert sentence["conceptual"] == text.strip()
        found.append({**sentence, "from_documents": " pair " not in label})
    return found


def _glosses(sentence: dict) -> list[str]:
    return [word.partition("/")[0] for word in sentence["formal"].split()]


def test_every_example_is_found_and_has_its_property(rendered: Path) -> None:
    """The rendered tiny run holds every example of ``EXAMPLES``, and each has the form its
    property guarantees, read again from the run's files."""
    examples = corpus_examples(rendered)
    assert set(examples) == set(EXAMPLES)
    documents = _documents(rendered)
    by_label = {d["label"]: d for d in documents}
    sentences = [s for d in documents for s in d["sentences"]]
    types = {s["label"]: d["type"] for d in documents for s in d["sentences"]}

    # one sentence, four renderings: an event report of a narrative whose subject is a definite
    # mention with a relative clause, and the four texts are that sentence's
    example = examples["renderings"]
    sentence = _sentence(rendered, _plain(example["label"]))
    assert types[sentence["label"]] in NARRATIVES
    for key, field in (
        ("conceptual", "conceptual"),
        ("formal", "formal"),
        ("propositional", "propositional"),
        ("spelled", "text"),
    ):
        assert _plain(example[key]) == sentence[field] and sentence[field]

    def relative_subject(s: dict) -> bool:
        words = _glosses(s)
        return s["readings"] == ["event"] and words[0] == "the" and "that" in words

    assert relative_subject(sentence)
    assert "EVENT(" in sentence["propositional"].split("} ")[0]  # the clause reports an event
    # the spelled rendering has a word for every token, and is not the formal rendering
    assert len(sentence["text"].split()) == len(sentence["tokens"])
    assert "LEXEME" not in sentence["text"]
    # it is the first such sentence of a narrative whose subject holds the clause
    earlier = sentences[
        : sentences.index(next(s for s in sentences if s["label"] == sentence["label"]))
    ]
    for other in earlier:
        if types[other["label"]] in NARRATIVES and relative_subject(other):
            subject = other["logical_form"]["subject"]
            assert not subject.get("clauses"), other["label"]

    # rule statements: the first is said with "all", the second with a bare plural
    worded, bare = _labeled(rendered, examples["rule_statements"]["block"])
    for rule in (worded, bare):
        assert rule["rule"] is not None
        assert rule["logical_form"]["quantifier"] == "nec_all"
        assert "THING" in _glosses(rule)  # the generic noun heads the subject
    assert _glosses(worded)[0] == "all" and "most" not in worded["readings"]
    assert _glosses(bare)[0] != "all" and "most" in bare["readings"]
    # a rule statement of a document is preferred to one of a test set
    rules = [s for s in sentences if s["logical_form"].get("rule") is not None]
    assert worded["from_documents"] == any(_glosses(s)[0] == "all" for s in rules)
    assert bare["from_documents"] == any(_glosses(s)[0] != "all" for s in rules)

    # comparisons: two entities of a narrative, and two categories of entities, on a scalar
    instance, category = _labeled(rendered, examples["comparisons"]["block"])
    for comparison, level in ((instance, "instance"), (category, "class")):
        form = comparison["logical_form"]
        assert form["level"] == level and form["predicate"]["operator"] == "GREATER"
        assert form["predicate"]["label"].startswith("SCALARDIM.")
        assert "than" in _glosses(comparison)
    assert "instance" in instance["logical_form"]["subject"]
    assert category["logical_form"]["subject"]["category"].startswith("CATEGORY.")
    assert category["logical_form"]["predicate"]["patient"]["category"].startswith("CATEGORY.")

    # the category document: its first sentences, with a membership fact and a fact with "all"
    example = examples["category_document"]
    document = by_label[_plain(example["label"])]
    assert document["type"] == "encyclopedic_category"
    assert document["topic"] == _plain(example["topic"]) and document["topic"].startswith(
        "CATEGORY."
    )
    first = document["sentences"][:EXAMPLE_SENTENCES]
    assert _lines(example["block"]) == [s["conceptual"] for s in first]
    assert any(s["logical_form"]["predicate"]["kind"] == "member" for s in first)
    assert any(_glosses(s)[0] == "all" for s in first)

    # the situational narrative: its first sentences, with a state, a change, and a definite
    # mention with a modifier
    example = examples["narrative"]
    document = by_label[_plain(example["label"])]
    assert document["type"] == "situational"
    first = document["sentences"][:EXAMPLE_SENTENCES]
    assert _lines(example["block"]) == [s["conceptual"] for s in first]
    levels = {s["logical_form"]["level"] for s in first}
    assert {"state", "change"} <= levels
    assert any(
        words[i] == "the"
        and not words[i + 1].startswith("CATEGORY.")
        or (words[i] == "the" and i + 2 < len(words) and words[i + 2] in ("with", "without"))
        for words in map(_glosses, first)
        for i in range(len(words) - 1)
    )

    # the tree: a report of a two-place event, "a X verb a Y"
    example = examples["tree"]
    sentence = _sentence(rendered, _plain(example["label"]))
    assert types[sentence["label"]] in NARRATIVES
    assert _plain(example["conceptual"]) == sentence["conceptual"]
    assert json.loads(_plain(example["tree"])) == sentence["tree"]
    assert sentence["logical_form"]["predicate"]["kind"] == "event_type2"
    assert sentence["readings"] == ["event"] and len(sentence["tokens"]) == 5
    assert "NP-SBJ" in example["tree"] and "NP-OBJ" in example["tree"]

    # the law-like pair: a pair of class_quantifier_lawlike whose false item says "all"
    example = examples["lawlike_pair"]
    pair = [
        i
        for i in _items(rendered, "class_quantifier_lawlike")
        if i["meta"]["pair"] == example["pair"]
    ]
    true = next(i for i in pair if i["meta"]["truth"])
    false = next(i for i in pair if not i["meta"]["truth"])
    assert _lines(example["block"]) == [
        f"true:   {true['input']['conceptual']}",
        f"false:  {false['input']['conceptual']}",
    ]
    assert false["meta"]["law_like"] and _glosses(false["input"])[0] == "all"
    assert false["input"]["logical_form"]["quantifier"] == "nec_all"
    for item in _items(rendered, "class_quantifier_lawlike"):
        if item["meta"]["pair"] < example["pair"] and not item["meta"]["truth"]:
            assert item["input"]["logical_form"]["quantifier"] != "nec_all"


def test_an_unrendered_run_has_no_example_of_the_renderings(unrendered: Path) -> None:
    """The example of the four renderings needs the spelled rendering: a run that was not
    rendered gives every other example, and ``guides`` would fail on that one."""
    examples = corpus_examples(unrendered)
    assert set(EXAMPLES) - set(examples) == {"renderings"}


def _guide_root(tmp_path: Path, text: str) -> Path:
    """A folder with the guides' files, empty but for ``docs/guides/CORPUS.md``."""
    for name in GUIDES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if name == "docs/guides/CORPUS.md" else "", encoding="utf-8")
    return tmp_path


def test_guides_writes_the_examples_and_check_covers_them(rendered: Path, tmp_path: Path) -> None:
    """Every text of every example is written between its markers, inline or as a block;
    ``check`` reports each one and writes nothing; a second pass moves nothing; and an example
    the run does not hold fails as a missing figure does, with its property to report."""
    examples = corpus_examples(rendered)
    figures = {EXAMPLE_CORPUS: {"examples": examples}}
    parts = []
    keys = []
    for name, example in examples.items():
        for field, value in example.items():
            key = f"{EXAMPLE_CORPUS}.examples.{name}.{field}"
            keys.append(key)
            if field == "block":
                parts.append(f"Prose.\n\n<!--figure:{key}-->\n```\nSTALE\n```\n<!--/figure-->\n")
            else:
                spec = "|d" if isinstance(value, int) else ""
                parts.append(f"Prose (<!--figure:{key}{spec}-->STALE<!--/figure-->) goes on.\n")
    stale = "\n".join(parts)
    root = _guide_root(tmp_path, stale)
    guide = root / "docs/guides/CORPUS.md"

    moved = fill_guides(figures, root, check=True)
    assert [key for _, key, _, _ in moved] == keys
    assert guide.read_text(encoding="utf-8") == stale

    assert len(fill_guides(figures, root)) == len(keys)
    written = guide.read_text(encoding="utf-8")
    assert "STALE" not in written
    for name, example in examples.items():
        for field, value in example.items():
            assert str(value) in written
            if field == "block":
                # the markers stay on lines of their own around the fenced block
                key = f"{EXAMPLE_CORPUS}.examples.{name}.block"
                assert f"\n<!--figure:{key}-->{value}<!--/figure-->\n" in written
    assert written.count("Prose") == len(keys)
    assert fill_guides(figures, root, check=True) == []

    # a run without one of the examples: the key is missing, as a missing figure is
    name = next(iter(examples))
    without = {EXAMPLE_CORPUS: {"examples": {k: v for k, v in examples.items() if k != name}}}
    with pytest.raises(KeyError) as error:
        fill_guides(without, root, check=True)
    assert missing_example(str(error.value.args[0])) == EXAMPLES[name]
    assert missing_example("tiny.corpus.sentences") is None


def test_the_guides_command_writes_checks_and_fails(
    unrendered: Path,
    rendered: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m semantic_world.figures guides`` on a guide that quotes an example: with
    ``--check`` it reports a stale example, returns 1, and writes nothing; without it, it
    writes the example; and ``--check`` then returns 0. On runs whose tiny corpus lacks the
    example (the unrendered run has no spelled rendering) it returns 1 and names the property,
    as it fails on a missing figure."""
    import semantic_world.figures as figures

    key = f"{EXAMPLE_CORPUS}.examples.renderings.spelled"
    guide = tmp_path / "GUIDE.md"
    stale = f"The sentence is <!--figure:{key}-->STALE<!--/figure-->.\n"
    guide.write_text(stale, encoding="utf-8")
    # an absolute path stands for itself under any root
    monkeypatch.setattr(figures, "GUIDES", (str(guide),))
    config = load_corpus_config(REPO / CORPUS_CONFIGS[EXAMPLE_CORPUS])
    name = f"{config.name}_seed{config.seed}"

    runs = tmp_path / "runs"
    shutil.copytree(rendered, runs / "corpus" / name)
    assert main(["guides", "--runs", str(runs), "--check"]) == 1
    assert key in capsys.readouterr().out
    assert guide.read_text(encoding="utf-8") == stale
    assert main(["guides", "--runs", str(runs)]) == 0
    spelled = corpus_examples(rendered)["renderings"]["spelled"]
    assert guide.read_text(encoding="utf-8") == stale.replace("STALE", spelled)
    capsys.readouterr()
    assert main(["guides", "--runs", str(runs), "--check"]) == 0

    bare = tmp_path / "bare"
    shutil.copytree(unrendered, bare / "corpus" / name)
    assert main(["guides", "--runs", str(bare), "--check"]) == 1
    printed = capsys.readouterr().err
    assert "no figure" in printed and key in printed
    assert EXAMPLES["renderings"] in printed


def test_every_example_key_of_the_guides_is_written(rendered: Path) -> None:
    """Every key under ``tiny.examples`` that a guide quotes names a text the script writes
    for the tiny run, and every example of ``EXAMPLES`` is quoted by a guide."""
    figures = {EXAMPLE_CORPUS: {"examples": corpus_examples(rendered)}}
    prefix = f"{EXAMPLE_CORPUS}.examples."
    quoted = [key for keys in guide_keys(REPO).values() for key in keys if key.startswith(prefix)]
    assert quoted
    for key in quoted:
        assert isinstance(lookup(figures, key), (str, int)), key
    assert {key[len(prefix) :].split(".")[0] for key in quoted} == set(EXAMPLES)
