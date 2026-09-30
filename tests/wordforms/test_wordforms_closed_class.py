"""Stage 4a: function words, affixes, inflected forms, the request file, and the evaluation by
kind."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_embeddings import WordEngine
from wordforms_support import DATA, FIXTURES, needs_audio, needs_cmudict, needs_wordfreq

from semantic_world.wordforms import Run, run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.closed_class import (
    SCHWA,
    Affix,
    add_closed_class,
    affix_candidates,
    function_candidates,
    join,
    shape_of,
)
from semantic_world.wordforms.config import (
    AFFIX_SHAPES,
    FUNCTION_SHAPES,
    ConfigError,
    load_config,
    parse_config,
)
from semantic_world.wordforms.embeddings import compute_embeddings, token_layout
from semantic_world.wordforms.english import edit_distance, is_vowel, stress_of
from semantic_world.wordforms.evaluate import evaluate_embeddings, stem_auc
from semantic_world.wordforms.frontends import compute_frontends
from semantic_world.wordforms.generate import GenerationError
from semantic_world.wordforms.io import AFFIX_COLUMNS, WORD_COLUMNS
from semantic_world.wordforms.streams import Streams
from semantic_world.wordforms.synth import synthesize_lexicon

pytestmark = [needs_cmudict, needs_wordfreq]

INFLECT_ALL = [{"words": "all", "affixes": ["PLURAL", "PAST", "PROGRESSIVE"]}]


def config_with(closed_class, count: int = 20, seed: int = 1, **extra):
    """A configuration of ``count`` content words with the given closed-class section."""
    return parse_config(
        {"wordforms": {"count": count}, "closed_class": closed_class, **extra}, "test", seed=seed
    )


def forms_of(run: Run, kind: str):
    return [w for w in run.lexicon.words if w.kind == kind]


# ---------------------------------------------------------------------------------------------
# Configuration and the request file
# ---------------------------------------------------------------------------------------------


def test_default_closed_class_has_function_words_and_affixes_and_inflects_nothing():
    config = parse_config({}, "x")
    closed = config.closed_class
    assert closed is not None
    assert closed.glosses[:3] == ("a", "the", "all") and len(closed.glosses) == 15
    assert closed.function_shapes == {"CV": 0.4, "CVC": 0.3, "VC": 0.2, "V": 0.1}
    assert closed.min_distance == 2
    assert [(a.gloss, a.position) for a in closed.affixes] == [
        ("PLURAL", "suffix"),
        ("PAST", "suffix"),
        ("PROGRESSIVE", "suffix"),
    ]
    assert closed.affix_shapes == {"C": 0.4, "VC": 0.4, "V": 0.2}
    assert closed.epenthesis is True and closed.inflect == ()
    assert load_config(DATA / "default.yaml").closed_class == closed
    assert parse_config({"closed_class": None}, "x").closed_class is None
    assert parse_config({"closed_class": None}, "x").resolved()["closed_class"] is None


def test_tiny_config_inflects_every_word_with_every_affix(tiny_config):
    entries = tiny_config.closed_class.inflect
    assert len(entries) == 1 and entries[0].words == "all"
    assert entries[0].affixes == ("PLURAL", "PAST", "PROGRESSIVE")


def test_closed_class_settings_and_their_errors():
    config = config_with(
        {
            "function_words": {
                "glosses": ["x", "y"],
                "shapes": {"CVC": 3, "V": 1},
                "min_distance": 1,
            },
            "affixes": {
                "items": [{"gloss": "A"}, {"gloss": "B", "position": "prefix"}],
                "shapes": {"C": 1},
                "epenthesis": False,
            },
            "inflect": [
                {"words": ["W.1", "W.20"], "affixes": ["B"]},
                {"words": "none", "affixes": ["A"]},
            ],
        }
    )
    closed = config.closed_class
    assert closed.glosses == ("x", "y") and closed.function_shapes == {"CVC": 0.75, "V": 0.25}
    assert closed.min_distance == 1 and closed.epenthesis is False
    assert [a.position for a in closed.affixes] == ["suffix", "prefix"]
    assert closed.affix_shapes == {"C": 1.0}
    assert closed.inflect[0].words == ("W.1", "W.20") and closed.inflect[1].words == "none"
    assert parse_config(config.resolved(), "again").closed_class == config.closed_class
    function_field = "closed_class.function_words"
    cases = [
        (
            {"function_words": {"shapes": {"CVCC": 1}}},
            f"{function_field}.shapes.CVCC",
            "unknown shape",
        ),
        ({"function_words": {"shapes": {"CV": 0}}}, f"{function_field}.shapes", "positive"),
        ({"function_words": {"glosses": ["a", "a"]}}, f"{function_field}.glosses", "distinct"),
        (
            {"function_words": {"glosses": ["a", False]}},
            f"{function_field}.glosses[1]",
            'quotes ("no")',
        ),
        ({"function_words": {"min_distance": 0}}, f"{function_field}.min_distance", "at least 1"),
        (
            {"affixes": {"items": [{"gloss": "A", "position": "infix"}]}},
            "closed_class.affixes.items[0].position",
            "suffix, prefix",
        ),
        (
            {"affixes": {"items": [{"position": "prefix"}]}},
            "closed_class.affixes.items[0].gloss",
            "required",
        ),
        (
            {"affixes": {"items": [{"gloss": "A"}, {"gloss": "A"}]}},
            "closed_class.affixes.items",
            "distinct",
        ),
        ({"affixes": {"shapes": {"CV": 1}}}, "closed_class.affixes.shapes.CV", "unknown shape"),
        (
            {"inflect": [{"words": "all", "affixes": ["PLURAL", "DUAL"]}]},
            "closed_class.inflect[0].affixes",
            "'DUAL' is not the gloss",
        ),
        (
            {"inflect": [{"words": ["W.21"], "affixes": ["PLURAL"]}]},
            "closed_class.inflect[0].words",
            "W.1 to W.20",
        ),
        (
            {"inflect": [{"words": "some", "affixes": ["PLURAL"]}]},
            "closed_class.inflect[0].words",
            "all, none",
        ),
        (
            {"inflect": [{"words": "all", "affixes": []}]},
            "closed_class.inflect[0].affixes",
            "non-empty",
        ),
        (
            {"inflect": [{"words": "all", "affixes": ["PLURAL"], "extra": 1}]},
            "closed_class.inflect[0].extra",
            "unknown key",
        ),
        ({"unknown": 1}, "closed_class.unknown", "unknown key"),
    ]  # noqa: E501
    for section, field, message in cases:
        with pytest.raises(ConfigError) as info:
            config_with(section)
        assert info.value.field == field, info.value
        assert message in info.value.message, info.value


def test_request_file_replaces_the_inline_request(tmp_path):
    request = tmp_path / "request.yaml"
    request.write_text(
        yaml.safe_dump(
            {
                "function_words": ["the", "no", "of"],
                "affixes": [{"gloss": "PLURAL"}, {"gloss": "AGENT", "position": "prefix"}],
                "inflect": [{"words": ["W.2"], "affixes": ["PLURAL", "AGENT"]}],
            }
        )
    )
    config = config_with({"request": str(request), "function_words": {"min_distance": 3}})
    closed = config.closed_class
    assert closed.request == str(request)
    assert closed.glosses == ("the", "no", "of") and closed.min_distance == 3
    assert [(a.gloss, a.position) for a in closed.affixes] == [
        ("PLURAL", "suffix"),
        ("AGENT", "prefix"),
    ]
    assert closed.inflect[0].words == ("W.2",) and closed.inflect[0].affixes == ("PLURAL", "AGENT")
    # the resolved configuration holds the request inline, and reloads equal
    resolved = config.resolved()["closed_class"]
    assert resolved["request"] is None and resolved["function_words"]["glosses"] == [
        "the",
        "no",
        "of",
    ]
    assert parse_config(config.resolved(), "again").closed_class == config.closed_class
    # errors name the request file and the field, and the inline request must not be given too
    with pytest.raises(ConfigError) as info:
        config_with({"request": str(request), "inflect": []})
    assert info.value.field == "closed_class.inflect" and "request" in info.value.message
    with pytest.raises(ConfigError) as info:
        config_with({"request": str(tmp_path / "missing.yaml")})
    assert info.value.field == "closed_class.request" and "cannot read" in info.value.message
    request.write_text(
        yaml.safe_dump(
            {
                "function_words": ["the"],
                "affixes": [{"gloss": "A"}],
                "inflect": [{"words": "all", "affixes": ["B"]}],
            }
        )
    )
    with pytest.raises(ConfigError) as info:
        config_with({"request": str(request)})
    assert info.value.source == str(request) and info.value.field == "inflect[0].affixes"
    request.write_text("function_words: [the]\nbad: 1\n")
    with pytest.raises(ConfigError) as info:
        config_with({"request": str(request)})
    assert info.value.source == str(request) and info.value.field == "bad"


# ---------------------------------------------------------------------------------------------
# Function words
# ---------------------------------------------------------------------------------------------


def test_function_candidates_have_the_shape_and_are_weighted(common_english):
    for shape in FUNCTION_SHAPES:
        candidates = function_candidates(common_english, shape)
        assert candidates and all(shape_of(s.phones) == shape for s, _ in candidates)
        assert all(stress_of(s.vowel) == 1 for s, _ in candidates)
        assert sum(w for _, w in candidates) == pytest.approx(1.0)
    # a consonant that begins many words weighs more than a rare one
    weights = {
        s.onset[0]: w for s, w in function_candidates(common_english, "CV") if s.vowel == "IY1"
    }
    assert weights["S"] > weights["ZH"]


def test_function_words_meet_every_rule(common_english):
    for seed in (1, 2, 3):
        run = run_forms(config_with({}, count=30, seed=seed))
        function = forms_of(run, "function")
        content = forms_of(run, "content")
        assert [w.label for w in function] == [f"F.{i}" for i in range(1, 16)]
        assert [w.gloss for w in function] == list(run.config.closed_class.glosses)
        weighted = {s for s, w in run.config.closed_class.function_shapes.items() if w > 0}
        for word in function:
            # one syllable of an allowed shape, with at most one consonant on each side
            assert word.syllable_count == 1 and word.stress == "1"
            assert shape_of(word.phones) in weighted
            assert common_english.phonotactic(word.phones)
            assert not common_english.is_pronunciation(word.phones)
            assert word.stripped not in {w.stripped for w in content}
            assert word.real_word is False and word.kind == "function"
            assert word.ipa and word.espeak and word.spelling
            assert word.spelling not in common_english.pattern_words
            assert word.stem is None and word.affix is None and word.epenthesis is None
        for a in function:
            for b in function:
                if a is not b:
                    assert edit_distance(a.stripped, b.stripped) >= 2
        report = run.lexicon.closed_class["function_words"]
        assert report["count"] == 15 and sum(report["shapes"].values()) == 15
        # every one-vowel form is a dictionary word, so no function word has the shape V
        assert report["candidates"]["V"]["not_english_words"] == 0 and "V" not in report["shapes"]
        assert report["candidates"]["CVC"]["not_content_words"] > 500


def test_function_words_use_the_common_word_counts_only_through_their_shape():
    config = config_with({"function_words": {"shapes": {"CVC": 1}}}, count=10)
    for word in forms_of(run_forms(config), "function"):
        assert shape_of(word.phones) == "CVC"


def test_min_distance_between_function_words():
    config = config_with({"function_words": {"min_distance": 1, "shapes": {"CVC": 1}}})
    run = run_forms(config)
    function = forms_of(run, "function")
    distances = [
        edit_distance(a.stripped, b.stripped)
        for i, a in enumerate(function)
        for b in function[i + 1 :]
    ]
    assert min(distances) >= 1
    with pytest.raises(GenerationError, match="every shape is used up"):
        run_forms(config_with({"function_words": {"min_distance": 4, "shapes": {"CV": 1}}}))


def test_adding_a_gloss_keeps_the_earlier_function_words():
    short = forms_of(
        run_forms(config_with({"function_words": {"glosses": ["a", "the"]}})), "function"
    )
    long = forms_of(
        run_forms(config_with({"function_words": {"glosses": ["a", "the", "of"]}})), "function"
    )
    assert [w.arpabet for w in long[:2]] == [w.arpabet for w in short]
    # the affixes do not change either, and a request without function words is allowed
    none = run_forms(config_with({"function_words": {"glosses": []}}))
    assert forms_of(none, "function") == []
    assert [a.arpabet for a in none.lexicon.affixes] == [
        a.arpabet for a in run_forms(config_with({})).lexicon.affixes
    ]


# ---------------------------------------------------------------------------------------------
# Affixes and inflected forms
# ---------------------------------------------------------------------------------------------


def test_affix_candidates(common_english):
    for shape in AFFIX_SHAPES:
        for position in ("suffix", "prefix"):
            candidates = affix_candidates(common_english, shape, position)
            assert candidates and all(shape_of(p) == shape for p, _ in candidates)
            assert all(stress_of(v) == 0 for p, _ in candidates for v in p if is_vowel(v))
            assert sum(w for _, w in candidates) == pytest.approx(1.0)
    suffixes = dict(affix_candidates(common_english, "C", "suffix"))
    assert suffixes[("Z",)] > suffixes[("B",)]  # the last consonants of common words
    assert dict(affix_candidates(common_english, "VC", "suffix"))[("IH0", "NG")] > 0.1


def test_affixes_have_an_allowed_shape_and_are_distinct():
    config = config_with(
        {
            "affixes": {
                "items": [
                    {"gloss": g, "position": p}
                    for g, p in zip("ABCDE", ["suffix", "prefix"] * 2 + ["suffix"], strict=True)
                ],
                "shapes": {"C": 1, "VC": 1, "V": 1},
            }
        }
    )
    for seed in (1, 2, 3):
        affixes = run_forms(config.with_seed(seed)).lexicon.affixes
        assert [a.label for a in affixes] == [f"AF.{i}" for i in range(1, 6)]
        assert [a.gloss for a in affixes] == ["A", "B", "C", "D", "E"]
        assert [a.position for a in affixes] == ["suffix", "prefix", "suffix", "prefix", "suffix"]
        for affix in affixes:
            assert affix.shape in AFFIX_SHAPES
            assert all(stress_of(p) == 0 for p in affix.phones if is_vowel(p))
            assert affix.ipa
        assert len({a.phones for a in affixes}) == 5


def test_joining_and_epenthesis():
    suffix = Affix("AF.1", "S", "suffix", ("Z",), "z")
    prefix = Affix("AF.2", "P", "prefix", ("AH0", "N"), "ən")
    assert join(("K", "AE1", "T"), suffix) == ("K", "AE1", "T", "Z")
    assert join(("K", "AE1", "T"), suffix, schwa=True) == ("K", "AE1", "T", SCHWA, "Z")
    assert join(("K", "AE1", "T"), prefix) == ("AH0", "N", "K", "AE1", "T")
    assert join(("K", "AE1", "T"), prefix, schwa=True) == ("AH0", "N", SCHWA, "K", "AE1", "T")


def test_inflected_forms_pass_the_check_and_take_a_schwa_exactly_where_the_join_fails(
    common_english,
):
    for seed in (1, 2):
        run = run_forms(config_with({"inflect": INFLECT_ALL}, count=40, seed=seed))
        content = {w.label: w for w in forms_of(run, "content")}
        affixes = {a.label: a for a in run.lexicon.affixes}
        inflected = forms_of(run, "inflected")
        report = run.lexicon.closed_class["inflected"]
        assert report["requested"] == 120 and report["made"] == len(inflected)
        assert report["made"] + len(report["skipped"]) == 120
        assert report["with_schwa"] == sum(w.epenthesis for w in inflected)
        for form in inflected:
            stem, affix = content[form.stem], affixes[form.affix]
            assert form.label == f"{stem.label}.{affix.label}" and form.kind == "inflected"
            assert common_english.phonotactic(form.phones)
            plain = join(stem.phones, affix)
            assert form.epenthesis is (not common_english.phonotactic(plain))
            assert form.phones == join(stem.phones, affix, schwa=form.epenthesis)
            assert form.syllable_count == sum(is_vowel(p) for p in form.phones)
            assert form.ipa.startswith(stem.ipa) and form.ipa.endswith(affix.ipa)
            assert form.spelling.startswith(stem.spelling)
            assert form.real_word is False and form.gloss is None
        for skip in report["skipped"]:
            stem, affix = content[skip["stem"]], affixes[skip["affix"]]
            assert not common_english.phonotactic(join(stem.phones, affix))
            assert not common_english.phonotactic(join(stem.phones, affix, schwa=True))
        # in word order, then affix order
        labels = [w.label for w in inflected]
        assert labels == sorted(labels, key=lambda s: (int(s.split(".")[1]), s.split(".")[3]))


def test_a_prefix_goes_before_the_stem(common_english):
    config = config_with(
        {
            "affixes": {"items": [{"gloss": "UN", "position": "prefix"}]},
            "inflect": [{"words": "all", "affixes": ["UN"]}],
        }
    )
    run = run_forms(config)
    affix = run.lexicon.affixes[0]
    content = {w.label: w for w in forms_of(run, "content")}
    inflected = forms_of(run, "inflected")
    assert inflected
    for form in inflected:
        assert form.phones[: len(affix.phones)] == affix.phones
        assert form.phones[-len(content[form.stem].phones) :] == content[form.stem].phones
        assert form.spelling.endswith(content[form.stem].spelling)


def test_inflect_entries_choose_words_and_affixes():
    config = config_with(
        {
            "inflect": [
                {"words": ["W.3", "W.1"], "affixes": ["PAST"]},
                {"words": ["W.1"], "affixes": ["PLURAL", "PAST"]},
            ]
        }  # fmt: skip
    )
    run = run_forms(config)
    inflected = forms_of(run, "inflected")
    pairs = {(w.stem, w.affix) for w in inflected} | {
        (s["stem"], s["affix"]) for s in run.lexicon.closed_class["inflected"]["skipped"]
    }
    assert pairs == {("W.1", "AF.1"), ("W.1", "AF.2"), ("W.3", "AF.2")}
    assert run.lexicon.closed_class["inflected"]["requested"] == 3
    none = run_forms(config_with({"inflect": [{"words": "none", "affixes": ["PAST"]}]}))
    assert forms_of(none, "inflected") == []


def test_without_epenthesis_a_failing_join_is_skipped():
    on = run_forms(config_with({"inflect": INFLECT_ALL}, count=30))
    off = run_forms(
        config_with({"inflect": INFLECT_ALL, "affixes": {"epenthesis": False}}, count=30)
    )
    schwa = [w for w in forms_of(on, "inflected") if w.epenthesis]
    assert schwa
    assert all(w.epenthesis is False for w in forms_of(off, "inflected"))
    skipped = {(s["stem"], s["affix"]) for s in off.lexicon.closed_class["inflected"]["skipped"]}
    assert {(w.stem, w.affix) for w in schwa} <= skipped
    assert all(
        "plain join" in s["reason"] for s in off.lexicon.closed_class["inflected"]["skipped"]
    )


# ---------------------------------------------------------------------------------------------
# Independence from the content words
# ---------------------------------------------------------------------------------------------


def test_closed_class_forms_change_no_content_word():
    off = run_forms(config_with(None, count=25))
    on = run_forms(config_with({"inflect": INFLECT_ALL}, count=25))
    assert [w.record() for w in off.lexicon.words] == [w.record() for w in forms_of(on, "content")]
    assert on.lexicon.words[:25] == off.lexicon.words
    assert off.lexicon.summary() == {
        k: v for k, v in on.lexicon.summary().items() if k != "closed_class"
    }
    # the closed-class forms come from their own stream
    assert on.streams.seed("closed_class") != on.streams.seed("generate")


def test_content_words_are_the_stage_4_words():
    """The content words of the tiny and default configurations are unchanged from stage 4."""
    from semantic_world.wordforms.io import words_frame

    stored = (FIXTURES / "tiny_words_stage4.csv").read_bytes()
    columns = stored.split(b"\n", 1)[0].decode().split(",")

    def content_csv(config) -> bytes:
        frame = words_frame(run_forms(load_config(config)).lexicon)
        return (
            frame.filter(pl.col("kind") == "content")
            .select(columns)
            .write_csv(float_precision=6)
            .encode()
        )

    assert content_csv(DATA / "tiny.yaml") == stored
    digest = hashlib.sha256(content_csv(DATA / "default.yaml")).hexdigest()
    assert digest == "1702284f9b737ee8e8cf78b3238983176f0d6da145872e18426d07998a66eb68"


def stand_in_run(tmp_path, closed_class, count: int = 8) -> Run:
    """A run with the stand-in engine, the espeak-ng speakers, and a fresh cache in
    ``tmp_path``, through the embeddings and the evaluation."""
    config = parse_config(
        {
            "name": "closed_test",
            "wordforms": {"count": count},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 2,
                "held_out_speaker_proportion": 0.34,
                "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
            },
            "embeddings": [
                {"name": f"{f}_fixed", "encoder": "fixed", "frontend": f, "pca_dims": 8}
                for f in ("cochleagram", "logmel")
            ],
            "closed_class": closed_class,
        },
        "closed_test",
    )
    run = run_forms(config)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": WordEngine()}, check=False
    )
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / "run")
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run"
    )
    run.evaluation = evaluate_embeddings(
        run.embeddings, run.lexicon.words, run.synthesis, run.streams.eval, config=config
    )
    run.write(tmp_path / "run")
    return run


@needs_audio
def test_turning_closed_class_forms_on_changes_no_audio_embedding_or_evaluation(tmp_path):
    off = stand_in_run(tmp_path / "off", None)
    on = stand_in_run(tmp_path / "on", {"inflect": INFLECT_ALL})
    n = len(off.synthesis.tokens)
    assert n == 8 * 3 * 2 and len(on.synthesis.tokens) > n
    # the content words' tokens come first, and are the same clips
    assert [t.label for t in on.synthesis.tokens[:n]] == [t.label for t in off.synthesis.tokens]
    assert [t.sha256 for t in on.synthesis.tokens[:n]] == [t.sha256 for t in off.synthesis.tokens]
    assert [t.cache_path for t in on.synthesis.tokens[:n]] == [
        t.cache_path for t in off.synthesis.tokens
    ]
    assert all(t.word.startswith(("F.", "W.")) for t in on.synthesis.tokens[n:])
    assert any(t.word.startswith("F.") for t in on.synthesis.tokens[n:])
    assert any(".AF." in t.word for t in on.synthesis.tokens[n:])
    # the same front-end frames, and the same embeddings (the projection is fitted on the
    # content words)
    for name in ("logmel", "cochleagram"):
        a, b = off.frontends[name], on.frontends[name]
        assert np.array_equal(a.frames, b.frames[: len(a.frames)])
    for name in ("cochleagram_fixed", "logmel_fixed"):
        a, b = off.embeddings[name], on.embeddings[name]
        assert np.array_equal(a.tokens, b.tokens[:n])
        assert np.array_equal(a.types, b.types[:8])
        assert b.meta["projection"]["fitted_tokens"] == a.meta["projection"]["fitted_tokens"]
        assert b.meta["projection"]["fitted_on"] == "training-speaker tokens of content words"
        assert a.meta["projection"]["fitted_on"] == "training-speaker tokens"
    # the content rows of the evaluation are the rows of the run without closed-class forms
    content = on.evaluation.filter(pl.col("kind") == "content")
    assert off.evaluation["kind"].to_list() == ["content"] * len(off.evaluation)
    assert content.equals(off.evaluation)


@needs_audio
def test_evaluation_by_kind_and_the_stem_auc(tmp_path):
    run = stand_in_run(tmp_path, {"inflect": INFLECT_ALL})
    table = run.evaluation
    kinds = ["content", "function", "inflected", "all"]
    assert table["kind"].to_list() == kinds * 2
    counts = {kind: len(forms_of(run, kind)) for kind in kinds[:3]}
    counts["all"] = len(run.lexicon.words)
    for row in table.iter_rows(named=True):
        assert row["words_evaluated"] == counts[row["kind"]]
        assert row["tokens_evaluated"] == counts[row["kind"]] * 6
        assert row["ap_across_train"] > row["chance_across_train"] > 0
        if row["kind"] in ("inflected", "all"):
            assert 0 <= row["stem_auc"] <= 1 and row["stem_auc_forms"] == counts["inflected"]
        else:
            assert row["stem_auc"] is None and row["stem_auc_forms"] is None
    # the stem AUC is reported for every embedding, in the run's folder
    stored = pl.read_csv(tmp_path / "run" / "eval" / "embeddings.csv")
    assert stored["stem_auc"].drop_nulls().len() == 4
    assert stored.columns == table.columns and "kind" in stored.columns


def test_stem_auc():
    stems = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    forms = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.6], [0.0, 1.0]])
    own = np.array([0, 1, 2, 0])
    # forms 0 to 2 are nearest their own stem; form 3 is nearest stem 1, then stem 2
    value, count = stem_auc(forms, stems, own)
    assert count == 4 and value == pytest.approx((1 + 1 + 1 + 0) / 4)
    tie, _ = stem_auc(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0]))
    assert tie == 0.5
    assert np.isnan(stem_auc(forms[:0], stems, own[:0])[0])
    assert np.isnan(stem_auc(forms[:1], stems[:1], own[:1])[0])


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------


def test_forms_command_writes_closed_class_forms(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["forms", str(DATA / "tiny.yaml"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "closed-class forms: 15 function words, 3 affixes" in text
    words = pl.read_csv(out / "words.csv")
    assert words.columns == list(WORD_COLUMNS)
    kinds = words["kind"].to_list()
    assert kinds[:35] == ["content"] * 20 + ["function"] * 15
    assert set(kinds[35:]) == {"inflected"} and len(kinds) > 35
    function = words.filter(pl.col("kind") == "function")
    assert function["gloss"].to_list() == list(load_config(DATA / "tiny.yaml").closed_class.glosses)
    assert function["label"].to_list() == [f"F.{i}" for i in range(1, 16)]
    inflected = words.filter(pl.col("kind") == "inflected")
    assert inflected["stem"].str.starts_with("W.").all()
    assert inflected["affix"].str.starts_with("AF.").all()
    assert inflected["epenthesis"].dtype == pl.Boolean and inflected["epenthesis"].null_count() == 0
    assert inflected["label"].to_list() == [
        f"{s}.{a}" for s, a in zip(inflected["stem"], inflected["affix"], strict=True)
    ]
    assert words.filter(pl.col("kind") != "inflected")["epenthesis"].null_count() == 35
    affixes = pl.read_csv(out / "affixes.csv")
    assert affixes.columns == list(AFFIX_COLUMNS)
    assert affixes["label"].to_list() == ["AF.1", "AF.2", "AF.3"]
    assert affixes["gloss"].to_list() == ["PLURAL", "PAST", "PROGRESSIVE"]
    assert affixes["position"].to_list() == ["suffix"] * 3
    summary = yaml.safe_load((out / "summary.yaml").read_text())
    assert summary["words"] == 20  # the content words, as before
    closed = summary["closed_class"]
    assert closed["function_words"]["count"] == 15 and closed["affixes"]["count"] == 3
    assert closed["inflected"]["requested"] == 60
    assert closed["inflected"]["made"] == inflected.height
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["closed_class"]["inflect"] == [
        {"words": "all", "affixes": ["PLURAL", "PAST", "PROGRESSIVE"]}
    ]
    assert config["provenance"]["stream_seeds"]["wordforms:closed_class"] > 0


def test_closed_class_null_writes_the_columns_but_no_affix_table(tmp_path):
    config = tmp_path / "null.yaml"
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["closed_class"] = None
    config.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["forms", str(config), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "summary.yaml", "words.csv"]
    words = pl.read_csv(out / "words.csv")
    assert words.columns == list(WORD_COLUMNS) and words.height == 20
    assert words["kind"].to_list() == ["content"] * 20
    for column in ("gloss", "stem", "affix", "epenthesis"):
        assert words[column].null_count() == 20
    assert "closed_class" not in yaml.safe_load((out / "summary.yaml").read_text())


def test_add_closed_class_to_a_lexicon_by_hand():
    config = config_with({"inflect": INFLECT_ALL}, count=5)
    from semantic_world.wordforms.generate import generate_lexicon

    streams = Streams(config.seed)
    lexicon = generate_lexicon(config, streams.generate)
    assert lexicon.closed_class is None and lexicon.affixes == []
    add_closed_class(config, streams, lexicon)
    assert len(lexicon.content) == 5 and len(lexicon.affixes) == 3
    assert len(lexicon.words) == 5 + 15 + lexicon.closed_class["inflected"]["made"]
    token_words, _, _ = token_layout(lexicon.words, type("S", (), {"speakers": [], "tokens": []})())
    assert token_words.size == 0
