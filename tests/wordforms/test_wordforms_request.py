"""Corpus stage 7: the request with lexemes. The lexemes of a request are assigned to content
words, and the inflected forms are made from the forms that the lexemes got."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_assign import taxonomy_file
from test_wordforms_embeddings import WordEngine
from wordforms_support import DATA, needs_audio, needs_cmudict, needs_torch, needs_wordfreq

from semantic_world.wordforms import Run, run_assignment, run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.assign import AssignmentError, branch_of
from semantic_world.wordforms.closed_class import repair_join
from semantic_world.wordforms.config import ConfigError, load_config, parse_config
from semantic_world.wordforms.embeddings import compute_embeddings
from semantic_world.wordforms.frontends import compute_frontends
from semantic_world.wordforms.io import WORD_COLUMNS, words_frame
from semantic_world.wordforms.lexemes import RANDOM, SAME_FORM, constrained_order, takeable
from semantic_world.wordforms.synth import synthesize_lexicon

pytestmark = [needs_cmudict, needs_wordfreq]

CATEGORIES = [f"CATEGORY.{b}" + (f".{k}" if k else "") for b in (1, 2, 3) for k in range(4)]
"""The 12 categories of the test meanings table: 3 branches, each a parent and three leaves."""
NOUN, ADJECTIVE, PART, VERB = "noun", "adjective", "part_noun", "transitive_verb"
MODES = ("arbitrary", "target_correlation", "branch_markers")


def request_data(inflect="all") -> dict:
    """A request like the corpus generator's: 13 lexemes of categories (CATEGORY.1.1 has a
    synonym), the generic noun, 4 adjectives, 3 part nouns, 5 verbs, and 2 homonyms (an
    adjective that shares a noun's form, and a verb that shares an adjective's). Nouns take
    PLURAL, and verbs take PLURAL and PAST. ``inflect`` is ``all`` (every noun with PLURAL and
    every verb with both affixes), ``some``, or ``none``."""
    lexemes = [{"concept": c, "pos": NOUN} for c in CATEGORIES]
    lexemes.append({"concept": "THING", "pos": NOUN})
    lexemes += [{"concept": f"PROPERTY.{k}", "pos": ADJECTIVE} for k in range(1, 5)]
    lexemes += [{"concept": f"PART.{k}", "pos": PART} for k in range(1, 4)]
    lexemes += [{"concept": f"EVENTTYPE1.{k}", "pos": "intransitive_verb"} for k in range(1, 3)]
    lexemes += [{"concept": f"EVENTTYPE2.1.{k}", "pos": VERB} for k in range(1, 4)]
    lexemes.append({"concept": "CATEGORY.1.1", "pos": NOUN})  # L.26, a synonym
    lexemes.append({"concept": "PROPERTY.9", "pos": ADJECTIVE, "same_form_as": "L.3"})  # L.27
    lexemes.append({"concept": "EVENTTYPE2.1.9", "pos": VERB, "same_form_as": "L.15"})  # L.28
    for number, lexeme in enumerate(lexemes, start=1):
        lexeme["label"] = f"L.{number}"
    nouns = [x["label"] for x in lexemes if x["pos"] == NOUN]
    verbs = [x["label"] for x in lexemes if x["pos"].endswith("verb")]
    entries = {
        "all": [
            {"lexemes": nouns, "affixes": ["PLURAL"]},
            {"lexemes": verbs, "affixes": ["PLURAL", "PAST"]},
        ],
        "some": [{"lexemes": nouns[:3], "affixes": ["PLURAL"]}],
        "none": [],
    }[inflect]
    return {
        "lexemes": lexemes,
        "takes": {NOUN: ["PLURAL"], "intransitive_verb": ["PLURAL", "PAST"], VERB: ["PLURAL", "PAST"]},
        "function_words": ["the", "a", "is", "not", "can"],
        "affixes": [{"gloss": "PLURAL", "position": "suffix"}, {"gloss": "PAST", "position": "suffix"}],
        "inflect": entries,
        "meanings": "meanings.csv",
    }  # fmt: skip


def write_request(tmp_path, data: dict | None = None, name: str = "request.yaml"):
    """A request file in ``tmp_path/corpus``, beside its meanings table."""
    folder = tmp_path / "corpus"
    folder.mkdir(exist_ok=True)
    taxonomy_file(folder)  # writes meanings.csv
    path = folder / name
    path.write_text(yaml.safe_dump(request_data() if data is None else data, sort_keys=False))
    return path


def request_config(tmp_path, request, count: int = 90, seed: int = 1, **sections):
    data = {
        "name": "request_test",
        "request": str(request) if not isinstance(request, dict) else request,
        "wordforms": {"count": count},
        "synthesis": {
            "cache_dir": str(tmp_path / "cache"),
            "tokens_per_speaker": 2,
            "held_out_speaker_proportion": 0.34,
            "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
        },
        "embeddings": [
            {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel", "pca_dims": 8}
        ],
        "device": "cpu",
        **sections,
    }
    return parse_config(data, "request_test", seed=seed)


def mode_settings(mode: str) -> dict:
    settings: dict = {"mode": mode, "null_samples": 50}
    if mode == "target_correlation":
        settings["target_correlation"] = {"target": 0.3, "tolerance": 0.05}
    return settings


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_the_request_is_a_top_level_setting(tmp_path):
    path = write_request(tmp_path)
    config = request_config(tmp_path, path)
    request, closed = config.request, config.closed_class
    assert config.assigns_lexemes and request.file == str(path)
    assert len(request.lexemes) == 28 and len(request.leaders) == 26
    assert request.lexemes[26].same_form_as == "L.3" and request.lexemes[0].concept == "CATEGORY.1"
    assert request.takes == {
        NOUN: ("PLURAL",),
        "intransitive_verb": ("PLURAL", "PAST"),
        VERB: ("PLURAL", "PAST"),
    }
    # the meanings path is relative to the request file
    assert request.meanings == str(path.parent / "meanings.csv") == config.meanings
    assert config.assignment.meanings is None
    assert closed.glosses == ("the", "a", "is", "not", "can") and closed.request == str(path)
    assert [a.gloss for a in closed.affixes] == ["PLURAL", "PAST"]
    assert not closed.word_inflect and len(closed.lexeme_inflect) == 2
    assert closed.lexeme_inflect[1].affixes == ("PLURAL", "PAST")
    # a word takes the affixes of its lexeme's part of speech, and of the lexemes sharing it
    needs = request.requirements()
    assert needs["L.1"] == ("PLURAL",) and needs["L.14"] == () == needs["L.20"]
    assert needs["L.3"] == ("PLURAL",) and needs["L.15"] == ("PLURAL", "PAST")  # homonyms
    assert needs["L.23"] == ("PLURAL", "PAST") and "L.27" not in needs
    # the resolved configuration holds the request inline, and reloads equal
    resolved = config.resolved()
    assert resolved["request"]["lexemes"][26] == {
        "label": "L.27", "concept": "PROPERTY.9", "pos": ADJECTIVE, "same_form_as": "L.3",
    }  # fmt: skip
    assert resolved["closed_class"]["inflect"][0]["lexemes"][0] == "L.1"
    again = parse_config(resolved, "request_test")
    assert again == config and again.resolved() == resolved


def test_request_errors_name_the_file_and_the_field(tmp_path):
    def error(change, **sections) -> ConfigError:
        data = request_data()
        change(data)
        with pytest.raises(ConfigError) as info:
            request_config(tmp_path, write_request(tmp_path, data), **sections)
        return info.value

    path = str(tmp_path / "corpus" / "request.yaml")
    found = error(lambda d: d["lexemes"].append(dict(d["lexemes"][0])))
    assert (found.source, found.field) == (path, "lexemes") and "distinct" in found.message
    found = error(lambda d: d["lexemes"][0].update(same_form_as="L.99"))
    assert found.field == "lexemes[0].same_form_as" and "not the label" in found.message
    found = error(lambda d: d["lexemes"][0].update(same_form_as="L.27"))
    assert "shares the form of L.3 itself" in found.message
    found = error(lambda d: d["lexemes"][0].pop("pos"))
    assert found.field == "lexemes[0].pos" and "required" in found.message
    found = error(lambda d: d["takes"].update(noun=["DUAL"]))
    assert (found.source, found.field) == (path, "takes.noun") and "not the gloss" in found.message
    found = error(lambda d: d["inflect"].append({"lexemes": ["L.99"], "affixes": ["PLURAL"]}))
    assert found.field == "inflect[2].lexemes" and "not a lexeme" in found.message
    # an inflected form must be one that the lexeme's word is sure to take
    found = error(lambda d: d["inflect"].append({"lexemes": ["L.14"], "affixes": ["PLURAL"]}))
    assert found.field == "inflect[2].lexemes" and "takes does not list" in found.message
    found = error(lambda d: d["inflect"].append({"lexemes": ["L.1"], "words": "all", "affixes": ["PLURAL"]}))  # fmt: skip
    assert found.field == "inflect[2].lexemes" and "not both" in found.message
    found = error(lambda d: d.update(lexemes=[]))
    assert found.field == "lexemes" and "needs lexemes" in found.message
    # the request's meanings and assignment.meanings cannot both be given
    found = error(lambda d: None, assignment={"meanings": str(tmp_path / "corpus/meanings.csv")})
    assert found.field == "assignment.meanings" and "must be null" in found.message
    # too few content words for the lexemes that need distinct forms
    found = error(lambda d: None, count=25)
    assert found.field == "wordforms.count" and "26 lexemes" in found.message
    # a request file asks for function words, so the closed class cannot be off
    found = error(lambda d: None, closed_class=None)
    assert found.field == "closed_class"
    # lexemes in an inflect entry need a request
    with pytest.raises(ConfigError) as info:
        parse_config(
            {"closed_class": {"inflect": [{"lexemes": ["L.1"], "affixes": ["PLURAL"]}]}}, "x"
        )
    assert info.value.field == "closed_class.inflect" and "no lexemes" in info.value.message
    with pytest.raises(ConfigError) as info:
        parse_config({"request": 3}, "x")
    assert info.value.field == "request"


def test_a_configuration_without_a_request_is_unchanged():
    """The word tables of the two example configurations, as they were before the request gained
    lexemes: the same bytes, and no ``pos`` column."""
    expected = {
        # The digests of the tables with the labels of stage a6. Mapped back to the old labels,
        # the tables hash to b80903c6837a0499225ab203398db400f68386d7ff10fe8b08b6fa699ed835dc
        # and dbfb05963ac3a9b929d0d61ef87aae6f54e4259d30df512e3299e2b10e159fff, as before.
        "tiny.yaml": "e526b061d76b3323f62a9ab62453e2d3dbf899e67359565da86440d8eb22c4e6",
        "default.yaml": "85cb8d4b60489650b0405cb95db28db2b3708ae1b77e438b6f576410cda66de5",
    }
    for name, digest in expected.items():
        config = load_config(DATA / name)
        assert config.request is None and not config.assigns_lexemes
        run = run_forms(config)
        assert run.assignment is None
        frame = words_frame(run.lexicon)
        assert tuple(frame.columns) == WORD_COLUMNS
        text = frame.write_csv(None, float_precision=6)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, name


# ---------------------------------------------------------------------------------------------
# The assignment of lexemes
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Runs of the request through the word forms, by mode and inflect list, each made once."""
    root = tmp_path_factory.mktemp("request_runs")
    made: dict[tuple, Run] = {}

    def run(mode: str = "arbitrary", inflect: str = "all", **sections) -> Run:
        key = (mode, inflect, yaml.safe_dump(sections))
        if key not in made:
            path = write_request(root, request_data(inflect), f"request_{inflect}.yaml")
            config = request_config(root, path, assignment=mode_settings(mode), **sections)
            made[key] = run_forms(config)
        return made[key]

    return run


@pytest.mark.parametrize("mode", MODES)
def test_every_lexeme_gets_a_form(runs, mode):
    run = runs(mode)
    assignment, request = run.assignment, run.config.request
    frame = assignment.frame()
    assert frame["lexeme"].to_list() == [x.label for x in request.lexemes]
    assert frame["meaning"].to_list() == [x.concept for x in request.lexemes]
    assert frame["pos"].to_list() == [x.pos for x in request.lexemes]
    by_label = {w.label: w for w in run.lexicon.words}
    rows = {row["lexeme"]: row for row in frame.iter_rows(named=True)}
    for row in rows.values():
        form = by_label[row["word"]]
        assert assignment.forms[row["lexeme"]] is form
        assert (row["spelling"], row["arpabet"]) == (form.spelling, form.arpabet)
    # distinct lexemes get distinct forms, unless they are homonyms
    leaders = [x.label for x in request.leaders]
    assert len({rows[label]["word"] for label in leaders}) == len(leaders) == 26
    assert rows["L.27"]["word"] == rows["L.3"]["word"]
    assert rows["L.28"]["word"] == rows["L.15"]["word"]
    assert rows["L.27"]["assigned"] == rows["L.28"]["assigned"] == SAME_FORM
    # the lexemes of categories follow the mode (the synonym L.26 too), the others are random
    categories = [x.label for x in request.leaders if x.concept in CATEGORIES]
    assert len(categories) == 13 and "L.26" in categories
    for label in leaders:
        assert rows[label]["assigned"] == (mode if label in categories else RANDOM)
    assert assignment.meanings == [rows[label]["meaning"] for label in categories]
    assert [w.label for w in assignment.words] == [rows[label]["word"] for label in categories]
    summary = assignment.summary
    assert summary["mode"] == mode and summary["meanings"] == 13
    assert summary["lexemes"] == {
        "count": 28,
        "assigned_by_the_mode": 13,
        "assigned_at_random": 13,
        "sharing_a_form": 2,
        "of_categories_sharing_a_form": 0,
        "words_without_a_lexeme": 90 - 26,
        "affixes_required": summary["lexemes"]["affixes_required"],
    }
    required = summary["lexemes"]["affixes_required"]
    assert required["PLURAL"]["lexemes"] == 20 and required["PAST"]["lexemes"] == 6
    # the words that no lexeme got are kept
    assert len(run.lexicon.content) == 90
    if mode == "target_correlation":
        report = summary["target_correlation"]
        assert report["reached"] and abs(summary["correlation"] - 0.3) <= 0.05
    # the part of speech of each assigned word, with both for a form that homonyms share
    table = words_frame(run.lexicon, pos=True)
    assert tuple(table.columns) == (*WORD_COLUMNS, "pos")
    pos = dict(zip(table["label"].to_list(), table["pos"].to_list(), strict=True))
    assert pos[rows["L.1"]["word"]] == NOUN and pos[rows["L.13"]["word"]] == NOUN
    assert pos[rows["L.3"]["word"]] == f"{NOUN}; {ADJECTIVE}"
    assert pos[rows["L.15"]["word"]] == f"{ADJECTIVE}; {VERB}"
    unassigned = [w for w in run.lexicon.content if w.label not in set(frame["base_word" if mode == "branch_markers" else "word"].to_list())]  # fmt: skip
    assert len(unassigned) == 64 and all(w.pos is None for w in unassigned)


def test_category_lexemes_carry_their_branch_marker(runs):
    run = runs("branch_markers")
    assignment = run.assignment
    markers = {m.label: m for m in assignment.markers}
    by_label = {w.label: w for w in run.lexicon.words}
    marker_of_branch: dict[str, str] = {}
    for row in assignment.frame().iter_rows(named=True):
        form = by_label[row["word"]]
        if row["meaning"] not in CATEGORIES:
            # no marker outside the meanings table, unless the form is a category's (L.27)
            assert row["lexeme"] == "L.27" or (form.kind == "content" and row["marker"] is None)
            continue
        branch = branch_of(row["meaning"], 1)
        assert row["branch"] == branch and form.kind == "marked"
        assert form.stem == row["base_word"] and form.label == f"{row['base_word']}.{row['marker']}"
        marker = markers[row["marker"]]
        assert form.phones[: len(marker.phones)] == marker.phones
        assert marker_of_branch.setdefault(branch, marker.label) == marker.label
    assert len(marker_of_branch) == 3 == len(markers)
    # the two lexemes of CATEGORY.1.1 are synonyms: two words, one marker
    rows = {row["lexeme"]: row for row in assignment.frame().iter_rows(named=True)}
    assert rows["L.2"]["marker"] == rows["L.26"]["marker"]
    assert rows["L.2"]["base_word"] != rows["L.26"]["base_word"]
    # the homonym L.27 shares the marked form of L.3
    assert rows["L.27"]["word"] == rows["L.3"]["word"] and rows["L.27"]["marker"] is not None


@pytest.mark.parametrize("mode", MODES)
def test_inflecting_lexemes_get_words_that_take_their_affixes(runs, mode, common_english):
    run = runs(mode)
    request, assignment = run.config.request, run.assignment
    closed = run.config.closed_class
    affixes = {a.gloss: a for a in run.lexicon.affixes}
    by_label = {w.label: w for w in run.lexicon.words}
    needs = request.requirements()
    # every lexeme's form takes every affix of its part of speech
    for lexeme in request.lexemes:
        form = assignment.forms[lexeme.label]
        for gloss in request.takes.get(lexeme.pos, ()):
            joined = repair_join(
                common_english, form.phones, affixes[gloss], closed.epenthesis, closed.glide
            )
            assert joined is not None, (lexeme.label, gloss)
            assert not common_english.is_common_pronunciation(joined[0])
    # so every inflected form that the request asks for is made, and none is skipped
    wanted = set()
    for entry in closed.lexeme_inflect:
        for label in entry.lexemes:
            for gloss in entry.affixes:
                assert gloss in needs[request_leader(request, label)]
                wanted.add((assignment.forms[label].label, affixes[gloss].label))
    inflected = [w for w in run.lexicon.words if w.kind == "inflected"]
    assert {(w.stem, w.affix) for w in inflected} == wanted
    assert len(inflected) == len(wanted) == 14 + 6 * 2  # 14 nouns, and 6 verbs with 2 affixes
    report = run.lexicon.closed_class["inflected"]
    assert report["requested"] == report["made"] == len(wanted) and not report["skipped"]
    assert sum(report["joins"].values()) == len(wanted)
    for form in inflected:
        stem = by_label[form.stem]
        assert form.label == f"{stem.label}.{form.affix}"
        assert form.held_out == stem.held_out and form.pos == stem.pos
        assert common_english.phonotactic(form.phones)
    # no two forms of the run sound the same
    assert len({w.stripped for w in run.lexicon.words}) == len(run.lexicon.words)
    assert not run.lexicon.closed_class["identical_forms"]
    # the forms come in the order: content, function, marked, inflected
    kinds = [w.kind for w in run.lexicon.words]
    order = ["content", "function", "marked", "inflected"]
    assert kinds == sorted(kinds, key=order.index)
    if mode == "branch_markers":
        # marked forms take affixes: WORD.12.MARKER.2.AFFIX.1
        marked = [w for w in inflected if by_label[w.stem].kind == "marked"]
        assert len(marked) == 13 and all(w.label.count(".MARKER.") == 1 for w in marked)
        assert all(w.label.endswith(".AFFIX.1") for w in marked)
        assert all(by_label[w.stem].stem in by_label for w in marked)


def request_leader(request, label: str) -> str:
    lexeme = next(x for x in request.lexemes if x.label == label)
    return lexeme.same_form_as or lexeme.label


def test_words_that_cannot_take_an_affix(common_english):
    """The rule behind the requirement: a word takes an affix when the joined form is legal, is
    not a common English word, and repeats no other form."""
    config = parse_config(
        {
            "wordforms": {"count": 200},
            "closed_class": {
                "affixes": {"max_skipped": 1.0, "shapes": {"C": 1.0}},
                "inflect": [],
            },
        },
        "x",
    )
    run = run_forms(config)
    closed, content = config.closed_class, run.lexicon.content
    others = [w for w in run.lexicon.words if w.kind != "content"]
    can, potential = takeable(common_english, closed, content, others, run.lexicon.affixes)
    base = {w.stripped for w in run.lexicon.words}
    checked = unable = 0
    for word in content:
        for affix in run.lexicon.affixes:
            joined = repair_join(common_english, word.phones, affix, True, "Y")
            legal = joined is not None and not common_english.is_common_pronunciation(joined[0])
            if affix.gloss in can[word.label]:
                assert legal and tuple(p.rstrip("012") for p in joined[0]) not in base
                checked += 1
            else:
                unable += 1
    assert checked > 300 and unable > 0 and len(potential) > 300
    # an assignment gives each meaning a word it allows, or says that the words run out
    rng = np.random.default_rng(0)
    allowed = np.array([[{"PLURAL", "PAST"} <= can[w.label] for w in content]] * 20)
    order = constrained_order(rng, allowed)
    assert sorted(order.tolist()) == list(range(200))
    assert all(allowed[i, order[i]] for i in range(20))
    with pytest.raises(AssignmentError, match="too few content words"):
        constrained_order(rng, np.zeros((2, 5), dtype=bool))


def test_categories_setting_and_a_request_without_meanings(tmp_path):
    # assignment.categories chooses the lexemes that the mode assigns: here the leaves
    path = write_request(tmp_path)
    config = request_config(tmp_path, path, assignment={"categories": "leaves", "null_samples": 20})
    assignment = run_forms(config).assignment
    rows = {row["lexeme"]: row for row in assignment.frame().iter_rows(named=True)}
    assert rows["L.1"]["assigned"] == RANDOM and rows["L.2"]["assigned"] == "arbitrary"
    assert assignment.summary["meanings"] == 10  # 9 leaves and the synonym
    # without meanings, every lexeme gets its word at random
    data = request_data()
    del data["meanings"]
    config = request_config(tmp_path, write_request(tmp_path, data, "plain.yaml"))
    run = run_forms(config)
    summary = run.assignment.summary
    assert summary["meanings"] == 0 and summary["correlation"] is None
    assert summary["lexemes"]["assigned_at_random"] == 26
    assert set(run.assignment.frame()["assigned"].to_list()) == {RANDOM, SAME_FORM}
    assert len([w for w in run.lexicon.words if w.kind == "inflected"]) == 26


@pytest.mark.parametrize("mode", ["arbitrary", "branch_markers"])
def test_the_function_words_never_change_a_lexemes_word(tmp_path, mode, common_english):
    """A request lists its function words by their frequency in the corpus, and their forms
    depend on that order. The lexemes' words must not: the function words are made after the
    assignment, and avoid its forms."""
    made = []
    for name, glosses in (
        ("one", ["the", "a", "is", "not", "can"]),
        ("two", ["can", "not", "is", "a", "the", "all", "some", "most", "that", "it", "with"]),
    ):
        data = request_data()
        data["function_words"] = glosses
        path = write_request(tmp_path, data, f"{name}.yaml")
        made.append(run_forms(request_config(tmp_path, path, assignment=mode_settings(mode))))
    one, two = made
    assert one.assignment.frame().equals(two.assignment.frame())
    for kind in ("content", "marked", "inflected"):
        mine = _records(w for w in one.lexicon.words if w.kind == kind)
        assert mine == _records(w for w in two.lexicon.words if w.kind == kind), kind
    forms = [
        {w.gloss: w.arpabet for w in run.lexicon.words if w.kind == "function"} for run in made
    ]
    assert len(forms[0]) == 5 and len(forms[1]) == 11 and forms[0]["the"] != forms[1]["the"]
    for run in made:
        function = [w for w in run.lexicon.words if w.kind == "function"]
        assert [w.label for w in function] == [f"FUNCWORD.{i + 1}" for i in range(len(function))]
        # no function word is another form of the run, a marker, or a form that a word could
        # have with an affix
        assert len({w.stripped for w in run.lexicon.words}) == len(run.lexicon.words)
        avoid = run.assignment.reserved | {
            tuple(p.rstrip("012") for p in m.phones) for m in run.assignment.markers
        }
        assert not {w.stripped for w in function} & avoid
        assert run.lexicon.closed_class["function_words"]["count"] == len(function)


# ---------------------------------------------------------------------------------------------
# The inflected forms never change the content words
# ---------------------------------------------------------------------------------------------


def _records(words) -> list[dict]:
    return [w.record() for w in words]


@pytest.mark.parametrize("mode", MODES)
def test_the_inflect_list_changes_no_word_and_no_assignment(runs, mode):
    """Which inflected forms a corpus happens to use must not change a lexeme's word: the
    requirement on a word comes from ``takes`` alone."""
    full, some, none = (runs(mode, inflect) for inflect in ("all", "some", "none"))
    for other in (some, none):
        for kind in ("content", "function", "marked"):
            mine = _records(w for w in full.lexicon.words if w.kind == kind)
            assert mine == _records(w for w in other.lexicon.words if w.kind == kind), kind
        assert [a.record() for a in other.lexicon.affixes] == [
            a.record() for a in full.lexicon.affixes
        ]
        assert other.assignment.frame().equals(full.assignment.frame())
        assert other.assignment.summary == full.assignment.summary
    count = lambda run: sum(w.kind == "inflected" for w in run.lexicon.words)  # noqa: E731
    assert (count(full), count(some), count(none)) == (26, 3, 0)


CONTRASTIVE = {"name": "contrastive", "encoder": "learned", "kind": "contrastive", "frontend": "logmel", "dims": 16, "hidden": 32, "epochs": 3, "batch_size": 16}  # fmt: skip


def embedded(tmp_path, run: Run, folder: str) -> Run:
    """The layers of a run with the stand-in engine: synthesis, front ends, and embeddings."""
    config = run.config
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": WordEngine()}, check=False
    )
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / folder)
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / folder
    )
    return run


@needs_audio
@needs_torch
@pytest.mark.parametrize("mode", ["arbitrary", "branch_markers"])
def test_the_inflect_list_changes_no_content_embedding(tmp_path, mode):
    """Two requests that differ only in their inflect lists give byte-identical tokens and
    embeddings for every form but the inflected ones: fixed embeddings (whose projection is
    fitted on content words) and a learned encoder (which never trains on inflected forms)."""
    made = {}
    for inflect in ("all", "some"):
        path = write_request(tmp_path, request_data(inflect), f"request_{inflect}.yaml")
        config = request_config(
            tmp_path,
            path,
            count=40,
            assignment=mode_settings(mode),
            embeddings=[
                {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel", "pca_dims": 8},
                CONTRASTIVE,
            ],
        )
        made[inflect] = embedded(tmp_path, run_forms(config), f"run_{inflect}")
    full, some = made["all"], made["some"]
    assert len(full.lexicon.words) > len(some.lexicon.words)

    def kept(run: Run):
        """The rows of the words and of the tokens that are not of inflected forms."""
        words = [i for i, w in enumerate(run.lexicon.words) if w.kind != "inflected"]
        labels = {run.lexicon.words[i].label for i in words}
        tokens = [i for i, t in enumerate(run.synthesis.tokens) if t.word in labels]
        return words, tokens

    full_words, full_tokens = kept(full)
    some_words, some_tokens = kept(some)
    assert [full.synthesis.tokens[i].record() for i in full_tokens] == [
        some.synthesis.tokens[i].record() for i in some_tokens
    ]
    for name in ("logmel_fixed", "contrastive"):
        a, b = full.embeddings[name], some.embeddings[name]
        assert np.array_equal(np.asarray(a.tokens)[full_tokens], np.asarray(b.tokens)[some_tokens])
        assert np.array_equal(np.asarray(a.types)[full_words], np.asarray(b.types)[some_words])
    # the learned encoder trained on the same tokens in both runs: none of an inflected form
    reports = [run.embeddings["contrastive"].meta["training"] for run in (full, some)]
    assert reports[0]["tokens"] == reports[1]["tokens"]
    training = [
        t
        for t in some.synthesis.tokens
        if not next(s for s in some.synthesis.speakers if s.label == t.speaker).held_out
        and not next(w for w in some.lexicon.words if w.label == t.word).held_out
        and next(w for w in some.lexicon.words if w.label == t.word).kind != "inflected"
    ]
    assert reports[1]["tokens"] == len(training)


@needs_audio
def test_two_passes_with_an_embedding_as_the_sound_distance(tmp_path):
    """With an embedding's distance, the first pass embeds the words, the lexemes are assigned,
    and the layers are made again with the inflected forms."""
    path = write_request(tmp_path)
    settings = {
        "mode": "target_correlation",
        "sound_distance": "logmel_fixed",
        "null_samples": 20,
        "target_correlation": {"target": 0.2, "tolerance": 0.05},
    }
    config = request_config(tmp_path, path, count=40, assignment=settings)
    run = run_forms(config)
    # the assignment waits for the embeddings, so the first pass has the content words alone
    assert run.assignment is None
    assert {w.kind for w in run.lexicon.words} == {"content"}
    with pytest.raises(AssignmentError, match="needs the run's embeddings"):
        from semantic_world.wordforms.lexemes import assign_lexemes

        assign_lexemes(config, run.lexicon, run.streams)
    embedded(tmp_path, run, "run")
    first = np.asarray(run.embeddings["logmel_fixed"].types).copy()
    run_assignment(run, tmp_path / "run")
    summary = run.assignment.summary
    assert summary["sound_distance"] == "cosine distance of logmel_fixed"
    assert summary["target_correlation"]["reached"]
    assert sum(w.kind == "inflected" for w in run.lexicon.words) == 26
    # the layers of the first pass are dropped, and the second pass holds every form
    assert run.synthesis is None and run.frontends is None and run.embeddings is None
    embedded(tmp_path, run, "run")
    second = np.asarray(run.embeddings["logmel_fixed"].types)
    assert [w.kind for w in run.lexicon.words[40:45]] == ["function"] * 5
    assert len(second) == len(first) + 5 + 26 and np.array_equal(second[: len(first)], first)
    assert len(run.synthesis.tokens) == len(run.lexicon.words) * 3 * 2


# ---------------------------------------------------------------------------------------------
# The command line and the output files
# ---------------------------------------------------------------------------------------------


def test_the_forms_command_assigns_the_lexemes(tmp_path, capsys):
    path = write_request(tmp_path)
    config = request_config(tmp_path, path, assignment=mode_settings("branch_markers"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config.to_yaml())
    out = tmp_path / "out"
    assert main(["forms", str(config_path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "28 lexemes have forms: 13 by the mode, 13 at random, 2 sharing a form" in printed
    assert "26 inflected forms" in printed and "3 branch markers" in printed
    words = pl.read_csv(out / "words.csv")
    assert words.columns == [*WORD_COLUMNS, "pos"]
    lexicon = pl.read_csv(out / "assignment" / "lexicon.csv")
    assert lexicon.columns == [
        "lexeme", "meaning", "pos", "word", "spelling", "arpabet", "assigned", "base_word",
        "branch", "marker",
    ]  # fmt: skip
    assert lexicon.height == 28 and set(lexicon["word"]) <= set(words["label"])
    inflected = words.filter(pl.col("kind") == "inflected")
    assert inflected.height == 26 and inflected["pos"].null_count() == 0
    assert (out / "assignment" / "markers.csv").exists()
    summary = yaml.safe_load((out / "summary.yaml").read_text())
    assert summary["assignment"]["lexemes"]["count"] == 28
    assert summary["closed_class"]["inflected"]["made"] == 26
    # the run's config.yaml holds the request inline, and gives the same run again
    again = tmp_path / "again"
    assert main(["forms", str(out / "config.yaml"), "--out", str(again)]) == 0
    for name in ("words.csv", "affixes.csv", "assignment/lexicon.csv", "assignment/summary.yaml"):
        assert (again / name).read_bytes() == (out / name).read_bytes(), name
    # a second seed gives other words
    assert main(["forms", str(config_path), "--out", str(again), "--seed", "2"]) == 0
    assert (again / "words.csv").read_bytes() != (out / "words.csv").read_bytes()
