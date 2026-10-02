"""Stage 6 acceptance tests: the output folder, the statistics, the ``generate`` command, and
the determinism properties.

``documents.jsonl`` round-trips through a JSON parser with the documented schema. Each corpus
text file matches its rendering in ``documents.jsonl``. The same configuration and seed give
byte-identical folders. The grammar settings change no logical form, and the test-set settings
change no document. On the default configuration, situational documents' co-occurrence
correlates more with thematic relatedness than encyclopedic documents' does, and less with
taxonomic similarity.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from corpus_support import corpus_config

from semantic_world.corpus import Proposition, config_from_mapping, load_config, propositional
from semantic_world.corpus.__main__ import main
from semantic_world.corpus.generate import generate
from semantic_world.corpus.io import OUTPUT_FILES, default_output_dir
from semantic_world.corpus.lexicon import LEXICON_COLUMNS
from semantic_world.corpus.planner import Planner
from semantic_world.corpus.realize import leaves, preorder
from semantic_world.corpus.scenes import Scene
from semantic_world.corpus.stats import cooccurrence, corpus_stats, correlation
from semantic_world.corpus.testsets import set_names

DOCUMENT_FIELDS = ["label", "type", "topic", "scenes", "referents", "sentences"]
SENTENCE_FIELDS = [
    "label",
    "tokens",
    "words",
    "text",
    "formal",
    "conceptual",
    "propositional",
    "tree",
    "logical_form",
    "referents",
    "events",
    "coreference",
    "distinguished",
    "readings",
]
RICH = {
    "mention": {"relative_clauses": {"rate": 0.3, "max_depth": 2, "object_share": 0.4}},
    "propositions": {"restriction_rate": 0.3},
}
MARKED = {
    "grammar": {
        "word_order": {"clause": "SOV", "adjective": "after", "auxiliary": "after"},
        "adjective_order": {"fixed": False},
        "can_rate": {"class": 0.9, "instance": 0.6},
        "morphology": {
            "number": {"enabled": True},
            "tense": {"enabled": True},
            "aspect": {"enabled": True},
        },
    },
}


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def folders(cases, tmp_path_factory):
    """The output folders of runs of a world, written once."""
    root = tmp_path_factory.mktemp("corpus_runs")
    made: dict = {}

    def folder(name: str = "default", count: int = 60, seed: int = 1, **sections) -> Path:
        key = (name, count, seed, json.dumps(sections, sort_keys=True))
        if key not in made:
            case = cases(name)
            settings = {"test_sets": {"size": 10}, **sections}
            settings["documents"] = {**settings.get("documents", {}), "count": count}
            corpus = generate(case.config(**settings, seed=seed), case.result)
            made[key] = corpus.write(root / f"run{len(made)}")
        return made[key]

    return folder


def files_of(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------------------------
# Acceptance: the files
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "default"])
def test_the_output_folder(cases, runs, tmp_path, name) -> None:
    corpus = runs(name, **RICH)
    folder = corpus.write(tmp_path / "run")
    sets = [set_name for set_name, *_ in set_names(corpus.config.test_sets.changes)]
    expected = set(OUTPUT_FILES) | {f"tests/{set_name}.jsonl" for set_name in sets}
    assert set(files_of(folder)) == expected
    assert "wordform_request.yaml" not in expected  # the request comes with stage 7

    # config.yaml: the resolved configuration loads back, and the provenance names the world
    data = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    provenance = data["provenance"]
    assert list(provenance) == [
        "git_commit",
        "git_dirty",
        "packages",
        "stream_seeds",
        "taxonomy",
        "wordforms",
    ]
    assert provenance["stream_seeds"] == corpus.planner.streams.seeds()
    taxonomy = provenance["taxonomy"]
    assert taxonomy["seed"] == cases(name).result.config.seed and len(taxonomy["config_hash"]) == 64
    assert provenance["wordforms"] is None and "numpy" in provenance["packages"]
    assert load_config(folder / "config.yaml").resolved() == corpus.config.resolved()

    # lexicon.csv: one row per lexeme
    lexicon = pl.read_csv(folder / "lexicon.csv", infer_schema_length=None)
    assert lexicon.columns == list(LEXICON_COLUMNS)
    assert lexicon["label"].to_list() == [x.label for x in corpus.planner.lexicon.lexemes]
    assert lexicon["word"].null_count() == len(lexicon)

    # scenes.jsonl: one scene per line, as the planner made them
    scenes = lines(folder / "scenes.jsonl")
    assert [Scene.from_json(record) for record in scenes] == corpus.planner.scenes
    assert [record["label"] for record in scenes] == [f"SN.{n}" for n in range(1, len(scenes) + 1)]

    # stats.yaml holds the statistics, and the test sets hold their items
    stats = yaml.safe_load((folder / "stats.yaml").read_text(encoding="utf-8"))
    assert stats == json.loads(json.dumps(corpus.stats), object_hook=_int_keys)
    for test_set in corpus.test_sets:
        items = lines(folder / "tests" / f"{test_set.name}.jsonl")
        assert items == [json.loads(json.dumps(item.to_json())) for item in test_set.items]
        assert all(list(item) == ["input", "meta"] for item in items)
        assert [item["meta"]["truth"] for item in items] == [True, False] * len(test_set.pairs)


def _int_keys(mapping: dict) -> dict:
    """JSON writes the integer keys of a histogram as text; YAML keeps them."""
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in mapping.items()}


@pytest.mark.parametrize("name", ["tiny", "default"])
def test_documents_jsonl_round_trips_with_the_documented_schema(runs, tmp_path, name) -> None:
    corpus = runs(name, **RICH)
    folder = corpus.write(tmp_path / "run")
    text = (folder / "documents.jsonl").read_text(encoding="utf-8")
    documents = lines(folder / "documents.jsonl")
    # the file is what a JSON parser reads and writes back
    assert "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents) == text
    assert documents == [json.loads(json.dumps(d.to_json())) for d in corpus.documents]
    assert len(documents) == corpus.config.documents.count
    lexemes = {x.label for x in corpus.planner.lexicon.lexemes}
    instances = set(corpus.planner.result.instances.labels)
    categories = {c.label for c in corpus.planner.result.tree.categories} | {"THING"}
    scene_labels = {scene.label for scene in corpus.planner.scenes}
    event_labels = {e.label for scene in corpus.planner.scenes for e in scene.events}
    mode = corpus.config.propositional_referents
    for number, document in enumerate(documents, start=1):
        assert list(document) == DOCUMENT_FIELDS
        assert document["label"] == f"D.{number}"
        assert document["type"] in corpus.config.documents.mix
        assert isinstance(document["topic"], str) and document["topic"]
        assert set(document["scenes"]) <= scene_labels
        referents = document["referents"]
        assert list(referents) == [f"R.{n}" for n in range(1, len(referents) + 1)]
        assert set(referents.values()) <= instances
        assert document["sentences"]
        for k, sentence in enumerate(document["sentences"], start=1):
            assert list(sentence) == SENTENCE_FIELDS
            assert sentence["label"] == f"{document['label']}.{k}"
            tokens = sentence["tokens"]
            assert tokens and {token.split("-")[0] for token in tokens} <= lexemes
            assert sentence["words"] is None and sentence["text"] is None
            assert len(sentence["formal"].split()) == len(tokens)
            assert len(sentence["conceptual"].split()) == len(tokens)
            tree = sentence["tree"]
            assert tree[0] == "S" and leaves(tree) == tokens
            nodes = preorder(tree)
            phrases = [node for node in nodes if node[0] in ("NP-SBJ", "NP-OBJ")]
            verb_phrases = [node for node in nodes if node[0] == "VP"]
            # the logical form reads back as a proposition, and gives the rendering
            form = sentence["logical_form"]
            proposition = Proposition.from_json(form)
            assert proposition.id == form["id"] and form["id"].startswith("PR.")
            assert form["grounding"]["test"]
            assert sentence["propositional"] == propositional(form, mode)
            assert len(sentence["referents"]) == len(phrases)
            for entry in sentence["referents"]:
                assert list(entry) == ["referent", "noun"]
                assert entry["referent"] in instances | categories
                assert entry["noun"] is None or entry["noun"] in categories
            assert len(sentence["events"]) == len(verb_phrases)
            assert all(e is None or e in event_labels for e in sentence["events"])
            assert (form["level"] == "event") == any(e is not None for e in sentence["events"])
            assert len(sentence["coreference"]) == len(phrases)
            for entry, label in zip(sentence["referents"], sentence["coreference"], strict=True):
                assert label == next(
                    (r for r, instance in referents.items() if instance == entry["referent"]), None
                )
            assert len(sentence["distinguished"]) == len(phrases)
            assert all(mark in (None, True, False) for mark in sentence["distinguished"])
            assert sentence["readings"] and set(sentence["readings"]) <= {
                "generic",
                "capacity",
                "event",
            }


@pytest.mark.parametrize("name", ["tiny", "default"])
def test_each_corpus_text_file_matches_its_rendering(runs, tmp_path, name) -> None:
    corpus = runs(name, **RICH)
    folder = corpus.write(tmp_path / "run")
    documents = lines(folder / "documents.jsonl")
    fields = {
        "corpus.txt": "formal",  # the spelled rendering, once word forms are attached
        "corpus_formal.txt": "formal",
        "corpus_conceptual.txt": "conceptual",
        "corpus_propositional.txt": "propositional",
    }
    for file, field in fields.items():
        text = (folder / file).read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n")
        blocks = text[:-1].split("\n\n")
        assert len(blocks) == len(documents)
        for block, document in zip(blocks, documents, strict=True):
            # one sentence per line, and a blank line between documents
            assert block.split("\n") == [s[field] for s in document["sentences"]]


# ---------------------------------------------------------------------------------------------
# Acceptance: determinism
# ---------------------------------------------------------------------------------------------


def test_the_same_configuration_and_seed_give_byte_identical_folders(cases, tmp_path) -> None:
    case = cases("default")
    settings = {"documents": {"count": 60}, "test_sets": {"size": 10}, **RICH}

    def run(path: str, seed: int = 1) -> Path:
        return generate(case.config(**settings, seed=seed), case.result).write(tmp_path / path)

    a, b = files_of(run("a")), files_of(run("b"))
    assert list(a) == list(b)
    for name in a:
        assert a[name] == b[name], name
    other = files_of(run("c", seed=2))
    assert list(other) == list(a)
    for name in ("documents.jsonl", "scenes.jsonl", "corpus_formal.txt", "stats.yaml"):
        assert other[name] != a[name], name
    assert other["tests/class_predicate.jsonl"] != a["tests/class_predicate.jsonl"]
    # the corpus seed is its own master seed: the taxonomy is the same world
    assert other["lexicon.csv"].count(b"\n") == a["lexicon.csv"].count(b"\n")
    # writing over a folder replaces its test sets
    stale = tmp_path / "a" / "tests" / "class_tense.jsonl"
    stale.write_text("{}\n", encoding="utf-8")
    assert files_of(run("a")) == a and not stale.exists()


def test_grammar_settings_change_no_logical_form_and_no_sentence_order(folders) -> None:
    base, other = folders(**RICH), folders(**RICH, **MARKED)
    a, b = lines(base / "documents.jsonl"), lines(other / "documents.jsonl")
    different = 0
    for x, y in zip(a, b, strict=True):
        for key in ("label", "type", "topic", "scenes", "referents"):
            assert x[key] == y[key]
        assert [s["label"] for s in x["sentences"]] == [s["label"] for s in y["sentences"]]
        for s, t in zip(x["sentences"], y["sentences"], strict=True):
            assert s["logical_form"] == t["logical_form"]
            assert s["propositional"] == t["propositional"]
            assert s["coreference"] == t["coreference"]
            different += s["conceptual"] != t["conceptual"]
    assert different > 200  # the realization does change
    # the scenes, the propositional corpus, and what the test items say are the same
    for name in ("scenes.jsonl", "corpus_propositional.txt"):
        assert (base / name).read_bytes() == (other / name).read_bytes(), name
    assert (base / "corpus_conceptual.txt").read_bytes() != (
        other / "corpus_conceptual.txt"
    ).read_bytes()
    for path in sorted((base / "tests").glob("*.jsonl")):
        ours, theirs = lines(path), lines(other / "tests" / path.name)
        assert [i["meta"] for i in ours] == [i["meta"] for i in theirs]
        for mine, yours in zip(ours, theirs, strict=True):
            assert mine["input"]["logical_form"] == yours["input"]["logical_form"]
            assert mine["input"]["propositional"] == yours["input"]["propositional"]


def test_test_set_settings_change_no_document(folders) -> None:
    base = files_of(folders(**RICH))
    other = files_of(folders(**RICH, test_sets={"size": 3, "changes": ["role", "subject"]}))
    documents = ("documents.jsonl", "scenes.jsonl", "lexicon.csv", *sorted(
        name for name in base if name.startswith("corpus")
    ))  # fmt: skip
    assert len(documents) == 7
    for name in documents:
        assert base[name] == other[name], name
    assert {name for name in other if name.startswith("tests/")} == {
        "tests/class_role.jsonl",
        "tests/class_subject.jsonl",
        "tests/class_subject_lawlike.jsonl",
        "tests/instance_role.jsonl",
        "tests/instance_subject.jsonl",
        "tests/event_role_possible.jsonl",
        "tests/event_role_impossible.jsonl",
        "tests/event_subject_possible.jsonl",
        "tests/event_subject_impossible.jsonl",
    }
    # the statistics differ only in their block on the test sets
    a, b = yaml.safe_load(base["stats.yaml"]), yaml.safe_load(other["stats.yaml"])
    assert a.pop("test_sets") != b.pop("test_sets") and a == b


# ---------------------------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "default", "deep"])
def test_the_statistics_count_what_the_documents_hold(runs, tmp_path, name) -> None:
    corpus = runs(name, **RICH)
    folder = corpus.write(tmp_path / "run")
    documents = lines(folder / "documents.jsonl")
    stats = yaml.safe_load((folder / "stats.yaml").read_text(encoding="utf-8"))
    assert list(stats) == [
        "documents",
        "sentences",
        "quantifiers",
        "tokens",
        "ambiguity",
        "mentions",
        "scenes",
        "propositions",
        "lexicon",
        "rule_statements",
        "cooccurrence",
        "test_sets",
        "lexeme_frequencies",
    ]
    sentences = [s for d in documents for s in d["sentences"]]
    forms = [s["logical_form"] for s in sentences]

    # documents: counts by type, and the achieved length beside the drawn length
    assert stats["documents"]["count"] == len(documents)
    limits = corpus.config.documents.sentences
    for kind, block in stats["documents"]["by_type"].items():
        made = [d for d in documents if d["type"] == kind]
        lengths = [len(d["sentences"]) for d in made]
        assert (block["count"], block["sentences"]) == (len(made), sum(lengths))
        achieved, drawn = block["achieved_length"], block["drawn_length"]
        assert achieved["mean"] == pytest.approx(np.mean(lengths), abs=1e-6)
        assert (achieved["min"], achieved["max"]) == (min(lengths), max(lengths))
        assert limits[kind].min <= drawn["min"] <= drawn["max"] <= limits[kind].max
        assert achieved["mean"] <= drawn["mean"] and achieved["max"] <= drawn["max"]
        short = sum(len(d.sentences) < d.drawn_length for d in corpus.documents if d.type == kind)
        assert block["shorter_than_drawn"] == short
        assert block["shorter_than_drawn_share"] == pytest.approx(short / len(made), abs=1e-6)
    for document in corpus.documents:
        assert 1 <= len(document.sentences) <= document.drawn_length
    by_type = stats["documents"]["by_type"]
    assert by_type["encyclopedic_category"]["shorter_than_drawn"] <= len(documents)
    assert by_type["entity"]["shorter_than_drawn"] > 0  # a narrative ends with its events

    # sentences: counts by level, the share of negative sentences, and the lengths
    block = stats["sentences"]
    levels = Counter(form["level"] for form in forms)
    assert block["count"] == len(sentences) and block["by_level"] == dict(levels)
    assert sum(block["by_section"].values()) == len(sentences)
    for level in ("class", "instance"):
        negative = sum(
            form["level"] == level and (not form["polarity"] or form.get("quantifier") == "no")
            for form in forms
        )
        assert block["negative_share"][level] == pytest.approx(negative / levels[level], abs=1e-6)
    lengths = Counter(len(s["tokens"]) for s in sentences)
    assert block["length_in_tokens"]["histogram"] == dict(sorted(lengths.items()))
    total = sum(length * count for length, count in lengths.items())
    assert block["length_in_tokens"]["mean"] == pytest.approx(total / len(sentences), abs=1e-6)
    assert (block["length_in_tokens"]["min"], block["length_in_tokens"]["max"]) == (
        min(lengths),
        max(lengths),
    )
    depths = Counter(_clause_depth(s["tree"]) for s in sentences)
    assert block["relative_clause_depth"] == dict(sorted(depths.items()))
    assert max(depths) == 2

    # the quantifier mix, as stated and by the strongest true quantifier
    block = stats["quantifiers"]
    stated = Counter(form["quantifier"] for form in forms if form["level"] == "class")
    assert block["class_level_sentences"] == levels["class"]
    assert {q: v["count"] for q, v in block["stated"].items()} == {
        q: stated[q] for q in ("all", "most", "some", "no", "generic")
    }
    for mix in (block["stated"], block["strongest_true"]):
        assert sum(v["count"] for v in mix.values()) == levels["class"]
        assert sum(v["share"] for v in mix.values()) == pytest.approx(1, abs=1e-5)
    strongest = {q: v["count"] for q, v in block["strongest_true"].items()}
    assert list(strongest) == ["all", "most", "some", "no", "pole"]
    # a bare generic stands for a stronger fact, so the strongest quantifiers are never fewer
    for quantifier in ("all", "most", "some", "no"):
        assert strongest[quantifier] >= stated[quantifier]
    poles = sum(f["level"] == "class" and f["predicate"]["kind"] == "scalar" for f in forms)
    assert strongest["pole"] == poles

    # tokens, by part of speech, and the frequency of every lexeme
    frequencies = Counter(token.split("-")[0] for s in sentences for token in s["tokens"])
    assert stats["lexeme_frequencies"] == {
        x.label: frequencies[x.label] for x in corpus.planner.lexicon.lexemes
    }
    assert stats["tokens"]["count"] == sum(frequencies.values())
    by_pos = Counter()
    for lexeme in corpus.planner.lexicon.lexemes:
        by_pos[lexeme.pos] += frequencies[lexeme.label]
    assert stats["tokens"]["by_pos"] == dict(by_pos)

    # ambiguity, mentions, scenes, propositions, the lexicon, and the rule statements
    readings = Counter("+".join(s["readings"]) for s in sentences)
    assert stats["ambiguity"]["readings"] == dict(sorted(readings.items()))
    assert stats["ambiguity"]["sentences"] == len(sentences)
    assert stats["ambiguity"]["ambiguous"] == sum(len(s["readings"]) > 1 for s in sentences)
    block = stats["mentions"]
    undistinguished = sum(mark is False for s in sentences for mark in s["distinguished"])
    definite = sum(mark is not None for s in sentences for mark in s["distinguished"])
    assert block["definite_not_distinguished"] == undistinguished
    assert block["definite"] == definite
    pronouns = sum(entry["noun"] is None for s in sentences for entry in s["referents"])
    assert block["pronouns"] == pronouns
    assert block["instance_mentions"] == block["indefinite"] + definite + pronouns
    scenes = lines(folder / "scenes.jsonl")
    events = [event for scene in scenes for step in scene["steps"] for event in step]
    assert (stats["scenes"]["count"], stats["scenes"]["events"]) == (len(scenes), len(events))
    progressive = sum(event["aspect"] == "progressive" for event in events)
    assert stats["scenes"]["progressive_share"] == pytest.approx(
        progressive / len(events), abs=1e-6
    )
    assert stats["propositions"]["distinct"] == len({form["id"] for form in forms})
    assert stats["lexicon"] == corpus.planner.lexicon.stats()
    assert stats["lexicon"]["verbs_without_word"] == corpus.planner.lexicon.verbs_without_word
    report = stats["rule_statements"]
    assert report["terms"] == report["stated"] + sum(report["skipped"].values())
    assert set(report["skipped"]) == {
        "threshold",
        "max_literals",
        "no_word",
        "no_instance",
        "unconfirmed",
    }
    if name == "tiny":
        assert stats["lexicon"]["verbs_without_word"] == {"V1": "every pair"}


def _clause_depth(tree) -> int:
    inner = max((_clause_depth(c) for c in tree[1:] if not isinstance(c, str)), default=0)
    return inner + (tree[0] == "RC")


def test_ambiguous_sentences_are_counted(runs) -> None:
    plain = runs("default").stats["ambiguity"]
    assert plain["ambiguous"] == 0 and plain["ambiguous_share"] == 0
    assert set(plain["readings"]) == {"capacity", "event", "generic"}
    # with "can" left out at times, and no tense or aspect marked, "the penguin swim" is both a
    # capacity and an event
    corpus = runs("default", grammar={"can_rate": {"class": 0.5, "instance": 0.3}})
    block = corpus.stats["ambiguity"]
    counted = sum(len(s.readings) > 1 for d in corpus.documents for s in d.sentences)
    assert block["ambiguous"] == counted > 50
    assert block["readings"]["capacity+event"] == counted
    assert block["ambiguous_share"] == round(counted / block["sentences"], 6)


def test_the_cooccurrence_check_by_brute_force(runs) -> None:
    corpus = runs("default", **RICH)
    planner, documents = corpus.planner, corpus.documents
    result = planner.result
    leaves_ = [leaf.label for leaf in result.tree.leaves]
    block = corpus.stats["cooccurrence"]
    assert block == json.loads(json.dumps(cooccurrence(planner, documents)))
    pairs = list(itertools.combinations(range(len(leaves_)), 2))
    assert (block["leaves"], block["pairs"]) == (len(leaves_), len(pairs))
    thematic = {
        frozenset((row["leaf_a"], row["leaf_b"])): row["thematic"]
        for row in result.relation_stats.thematic.iter_rows(named=True)
    }
    leaf_of = {
        label: label.rsplit(".", 1)[0].replace("I", "C", 1) for label in result.instances.labels
    }
    concept = {x.label: x.concept for x in planner.lexicon.lexemes}

    def occurring(document, measure: str) -> set[str]:
        found = set()
        for sentence in document.to_json()["sentences"]:
            if measure == "words":
                found |= {concept[token.split("-")[0]] for token in sentence["tokens"]}
            else:
                for entry in sentence["referents"]:
                    found.add(leaf_of.get(entry["referent"], entry["referent"]))
        return found & set(leaves_)

    groups = {
        "encyclopedic_category": ("encyclopedic_category",),
        "encyclopedic_feature": ("encyclopedic_feature",),
        "entity": ("entity",),
        "situational": ("situational",),
        "encyclopedic": ("encyclopedic_category", "encyclopedic_feature"),
        "narrative": ("entity", "situational"),
        "all": ("encyclopedic_category", "encyclopedic_feature", "entity", "situational"),
    }
    assert list(block["by_document_type"]) == list(groups)
    for group, kinds in groups.items():
        made = [d for d in documents if d.type in kinds]
        record = block["by_document_type"][group]
        assert record["documents"] == len(made)
        for measure in ("words", "referents"):
            sets = [occurring(d, measure) for d in made]
            counts = np.array(
                [sum(leaves_[a] in s and leaves_[b] in s for s in sets) for a, b in pairs], float
            )
            related = np.array(
                [thematic.get(frozenset((leaves_[a], leaves_[b])), 0.0) for a, b in pairs]
            )
            expected = np.corrcoef(counts, related)[0, 1]
            assert record[measure]["thematic"]["pearson"] == pytest.approx(expected, abs=1e-6)
            assert -1 <= record[measure]["thematic"]["spearman"] <= 1
            assert -1 <= record[measure]["taxonomic"]["pearson"] <= 1
    # in a narrative, "the bird" and "it" count for the penguin's leaf by referents, and not by
    # words: the two measures differ there
    narrative = block["by_document_type"]["narrative"]
    assert narrative["words"] != narrative["referents"]
    assert set(block["check"]) == {"words", "referents"}


def test_correlations() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 50.0])
    assert correlation(x, 2 * x + 1) == {"pearson": 1.0, "spearman": 1.0}
    assert correlation(x, -(x**3)) == {"pearson": pytest.approx(-0.9, abs=0.1), "spearman": -1.0}
    # ties take their average rank
    tied = correlation(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 5.0]))
    assert tied["spearman"] == pytest.approx(0.833333, abs=1e-6)
    assert correlation(x, np.ones(5)) == {"pearson": None, "spearman": None}


def test_cooccurrence_on_the_default_configuration() -> None:
    # The acceptance check of stage 6, on the default configuration with fewer documents:
    # situational documents' co-occurrence correlates more with thematic relatedness than
    # encyclopedic documents' does, and less with taxonomic similarity.
    data = yaml.safe_load(Path("data/corpus/default.yaml").read_text(encoding="utf-8"))
    data["documents"]["count"] = 2500
    data["test_sets"]["size"] = 0
    planner = Planner(config_from_mapping(data))
    block = cooccurrence(planner, planner.generate())
    groups = block["by_document_type"]
    for measure in ("words", "referents"):
        for kind in ("pearson", "spearman"):
            situational = groups["situational"][measure]
            encyclopedic = groups["encyclopedic"][measure]
            assert situational["thematic"][kind] > encyclopedic["thematic"][kind] + 0.1
            assert situational["taxonomic"][kind] < encyclopedic["taxonomic"][kind]
        assert block["check"][measure] == {
            "thematic_situational_above_encyclopedic": {"pearson": True, "spearman": True},
            "taxonomic_situational_below_encyclopedic": {"pearson": True, "spearman": True},
        }


def test_statistics_without_test_sets_and_without_documents(cases) -> None:
    case = cases("tiny")
    planner = Planner(case.config(), case.result)
    stats = corpus_stats(planner, planner.generate(15))
    assert stats["test_sets"] == {} and stats["documents"]["count"] == 15
    empty = generate(case.config(documents={"count": 0}, test_sets={"size": 5}), case.result)
    assert empty.documents == () and empty.stats["sentences"]["count"] == 0
    assert all(not test_set.pairs for test_set in empty.test_sets if test_set.level != "class")
    assert empty.stats["cooccurrence"]["by_document_type"]["all"]["words"]["thematic"] == {
        "pearson": None,
        "spearman": None,
    }


# ---------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------


def test_the_generate_command(tmp_path, capsys) -> None:
    out = tmp_path / "tiny"
    assert main(["generate", "data/corpus/tiny.yaml", "--out", str(out)]) == 0
    message = capsys.readouterr().out
    assert message.startswith(f"wrote {out}: 20 documents, ")
    assert "16 test sets" in message
    config = load_config(out / "config.yaml")
    assert (config.name, config.seed, config.documents.count) == ("tiny", 1, 20)
    assert len(lines(out / "documents.jsonl")) == 20
    assert all(len(lines(path)) <= 40 for path in (out / "tests").glob("*.jsonl"))
    # the seed can be given on the command line, and it changes the corpus
    other = tmp_path / "other"
    assert main(["generate", "data/corpus/tiny.yaml", "--seed", "5", "--out", str(other)]) == 0
    assert load_config(other / "config.yaml").seed == 5
    assert (other / "documents.jsonl").read_bytes() != (out / "documents.jsonl").read_bytes()
    assert default_output_dir(config) == Path("runs/corpus/tiny_seed1")
    capsys.readouterr()
    # an error names the file and the field
    bad = tmp_path / "bad.yaml"
    bad.write_text("taxonomy: {config: data/taxonomy/tiny.yaml}\nentity: {scenes: 0}\n")
    assert main(["generate", str(bad)]) == 1
    assert "entity.scenes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["render"])  # the render command comes with stage 7


def test_a_corpus_can_be_made_in_python() -> None:
    config = corpus_config(documents={"count": 12}, test_sets={"size": 4})
    corpus = generate(config)
    assert len(corpus.documents) == 12 and corpus.config is config
    assert corpus.stats["documents"]["count"] == 12
    assert corpus.planner.scenes and corpus.test_sets
