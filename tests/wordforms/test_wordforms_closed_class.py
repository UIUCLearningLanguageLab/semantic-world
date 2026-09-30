"""Stage 4a: function words, affixes, inflected forms, the request file, and the evaluation by
kind."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
    vowel_collision,
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
    assert closed.glosses[:3] == ("the", "and", "a") and len(closed.glosses) == 15
    assert closed.function_shapes == {"CV": 0.3, "CVC": 0.4, "VC": 0.3}
    assert closed.min_distance == 2 and closed.function_source == "pseudo"
    assert closed.affix_source == "pseudo" and closed.max_skipped == 0.1
    assert [(a.gloss, a.position) for a in closed.affixes] == [
        ("PLURAL", "suffix"),
        ("PAST", "suffix"),
        ("PROGRESSIVE", "suffix"),
    ]
    assert closed.affix_shapes == {"C": 0.4, "VC": 0.4, "V": 0.2}
    assert closed.epenthesis is True and closed.glide == "Y" and closed.inflect == ()
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
            assert not common_english.is_common_pronunciation(word.phones)
            assert word.stripped not in {w.stripped for w in content}
            assert word.real_word is False and word.kind == "function"
            assert word.ipa and word.espeak and word.spelling
            assert word.spelling not in common_english.pattern_words
            assert word.stem is None and word.affix is None and word.join is None
        for a in function:
            for b in function:
                if a is not b:
                    assert edit_distance(a.stripped, b.stripped) >= 2
        report = run.lexicon.closed_class["function_words"]
        assert report["count"] == 15 and sum(report["shapes"].values()) == 15
        assert report["candidates"]["CVC"]["not_content_words"] > 1000
        # the most frequent half (8 of 15) have two phonemes
        assert report["two_phoneme_words"] == 8
        assert all(len(w.phones) == 2 for w in function[:8])
        assert any(len(w.phones) == 3 for w in function)


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
    assert suffixes[("N",)] > suffixes[("B",)]  # the last consonants of uninflected words
    assert dict(affix_candidates(common_english, "VC", "suffix"))[("AH0", "N")] > 0.1


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


def test_joining_with_a_schwa_or_a_glide():
    from semantic_world.wordforms.closed_class import glide_after, repair_join, vowel_collision

    suffix = Affix("AF.1", "S", "suffix", ("Z",), "z")
    prefix = Affix("AF.2", "P", "prefix", ("AH0", "N"), "ən")
    assert join(("K", "AE1", "T"), suffix) == ("K", "AE1", "T", "Z")
    assert join(("K", "AE1", "T"), suffix, "schwa") == ("K", "AE1", "T", SCHWA, "Z")
    assert join(("K", "AE1", "T"), prefix) == ("AH0", "N", "K", "AE1", "T")
    assert join(("K", "AE1", "T"), prefix, "schwa") == ("AH0", "N", SCHWA, "K", "AE1", "T")
    # a glide goes only where a vowel meets a vowel: Y after a front vowel, W after a back or
    # rounded one, and the configured default otherwise
    vowel = Affix("AF.3", "V", "suffix", ("AH0",), "ə")
    assert vowel_collision(("K", "AE1", "T"), vowel) is None
    assert vowel_collision(("S", "IY1"), vowel) == "IY1"
    assert vowel_collision(("S", "IY1"), suffix) is None
    assert join(("K", "AE1", "T"), vowel, "glide") == ("K", "AE1", "T", "AH0")
    assert join(("S", "IY1"), vowel, "glide") == ("S", "IY1", "Y", "AH0")
    assert join(("S", "UW1"), vowel, "glide") == ("S", "UW1", "W", "AH0")
    assert join(("S", "AA1"), vowel, "glide") == ("S", "AA1", "Y", "AH0")
    assert join(("S", "AA1"), vowel, "glide", "W") == ("S", "AA1", "W", "AH0")
    for front in ("IY", "IH1", "EY0", "EH", "AE"):
        assert glide_after(front, "W") == "Y"
    for back in ("UW", "UH1", "OW0", "AO", "AW"):
        assert glide_after(back, "Y") == "W"
    assert glide_after("AH0", "W") == "W" and glide_after("ER", "Y") == "Y"
    # a vowel-final prefix before a vowel-initial stem
    open_prefix = Affix("AF.4", "P", "prefix", ("AH0",), "ə")
    assert vowel_collision(("AE1", "T"), open_prefix) == "AH0"
    assert join(("AE1", "T"), open_prefix, "glide", "W") == ("AH0", "W", "AE1", "T")
    assert join(("K", "AE1", "T"), open_prefix, "glide") == ("AH0", "K", "AE1", "T")
    # an English affix is not repaired or checked
    english = Affix("AF.5", "PLURAL", "suffix", ("Z",), "z", (("Z",), ("S",), ("IH0", "Z")))
    assert repair_join(object(), ("K", "AE1", "T"), english, True) == (
        ("K", "AE1", "T", "S"),
        "none",
    )


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
        joins = report["joins"]
        assert set(joins) == {"none", "schwa", "glide"} and sum(joins.values()) == len(inflected)
        for name in joins:
            assert joins[name] == sum(w.join == name for w in inflected)
        for form in inflected:
            stem, affix = content[form.stem], affixes[form.affix]
            assert form.label == f"{stem.label}.{affix.label}" and form.kind == "inflected"
            assert common_english.phonotactic(form.phones)
            # the repair applies exactly where the plain join fails: a glide where a vowel meets
            # a vowel and the glide helps, and otherwise a schwa
            plain = join(stem.phones, affix)
            if common_english.phonotactic(plain):
                assert form.join == "none" and form.phones == plain
            elif vowel_collision(stem.phones, affix) and common_english.phonotactic(
                join(stem.phones, affix, "glide")
            ):
                assert form.join == "glide" and form.phones == join(stem.phones, affix, "glide")
            else:
                assert form.join == "schwa" and form.phones == join(stem.phones, affix, "schwa")
            assert form.syllable_count == sum(is_vowel(p) for p in form.phones)
            assert form.ipa.startswith(stem.ipa) and form.ipa.endswith(affix.ipa)
            assert form.spelling.startswith(stem.spelling)
            assert form.real_word is False and form.gloss is None
        for skip in report["skipped"]:
            stem, affix = content[skip["stem"]], affixes[skip["affix"]]
            if "common English word" in skip["reason"]:
                continue
            assert not common_english.phonotactic(join(stem.phones, affix))
            assert not common_english.phonotactic(join(stem.phones, affix, "glide"))
            assert not common_english.phonotactic(join(stem.phones, affix, "schwa"))
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
    # max_skipped 1 keeps the affix draws the same with and without epenthesis
    on = run_forms(config_with({"inflect": INFLECT_ALL, "affixes": {"max_skipped": 1}}, count=30))
    off = run_forms(
        config_with(
            {"inflect": INFLECT_ALL, "affixes": {"epenthesis": False, "max_skipped": 1}}, count=30
        )
    )
    assert [a.phones for a in on.lexicon.affixes] == [a.phones for a in off.lexicon.affixes]
    schwa = [w for w in forms_of(on, "inflected") if w.join == "schwa"]
    assert schwa
    assert all(w.join != "schwa" for w in forms_of(off, "inflected"))
    skipped = {(s["stem"], s["affix"]) for s in off.lexicon.closed_class["inflected"]["skipped"]}
    assert {(w.stem, w.affix) for w in schwa} <= skipped
    for skip in off.lexicon.closed_class["inflected"]["skipped"]:
        assert "phonotactic check" in skip["reason"] or "common English" in skip["reason"]
        assert "schwa" not in skip["reason"]


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
    """With the inflections kept among the pattern words, the content words of the tiny and
    default configurations are the stage 4 words."""
    from semantic_world.wordforms.io import words_frame

    stored = (FIXTURES / "tiny_words_stage4.csv").read_bytes()
    columns = stored.split(b"\n", 1)[0].decode().split(",")

    def content_csv(config) -> bytes:
        data = yaml.safe_load(Path(config).read_text())
        data["wordforms"]["exclude_inflections"] = False
        frame = words_frame(run_forms(parse_config(data, "stage4")).lexicon)
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
    assert set(inflected["join"].to_list()) <= {"none", "schwa", "glide"}
    assert inflected["join"].null_count() == 0
    assert inflected["label"].to_list() == [
        f"{s}.{a}" for s, a in zip(inflected["stem"], inflected["affix"], strict=True)
    ]
    assert words.filter(pl.col("kind") != "inflected")["join"].null_count() == 35
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
    for column in ("gloss", "stem", "affix", "join"):
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


# ---------------------------------------------------------------------------------------------
# Stage 4b: the rules decided on September 30, 2026, and the English options
# ---------------------------------------------------------------------------------------------


def test_function_words_may_sound_like_rare_dictionary_words_but_not_common_ones(common_english):
    run = run_forms(config_with({}, count=30))
    function = forms_of(run, "function")
    report = run.lexicon.closed_class
    # the rules reject common words only, and the summary lists the rare homophones
    assert report["function_words"]["candidates"]["CV"]["not_common_english_words"] > 30
    for word in function:
        assert not common_english.is_common_pronunciation(word.phones)
    rare = {label for label in report["english_words"] if label.startswith("F.")}
    assert rare == {w.label for w in function if common_english.is_pronunciation(w.phones)}


def test_the_frequent_half_gets_two_phonemes_even_with_a_cvc_weight():
    config = config_with(
        {"function_words": {"glosses": ["x", "y", "z"], "shapes": {"CVC": 0.8, "VC": 0.2}}}
    )
    function = forms_of(run_forms(config), "function")
    assert [len(w.phones) for w in function[:2]] == [2, 2]
    # with no two-phoneme shape weighted, every word takes the weighted shape
    config = config_with({"function_words": {"glosses": ["x", "y", "z"], "shapes": {"CVC": 1}}})
    assert [len(w.phones) for w in forms_of(run_forms(config), "function")] == [3, 3, 3]


def test_an_affix_that_too_many_stems_cannot_take_is_rejected(common_english):
    from semantic_world.wordforms.closed_class import skipped_share

    run = run_forms(config_with({"inflect": INFLECT_ALL}, count=60))
    content = forms_of(run, "content")
    report = run.lexicon.closed_class["affixes"]
    assert report["max_skipped"] == 0.1
    assert sum(report["candidates_rejected_for_skipped_stems"].values()) > 0
    for affix in run.lexicon.affixes:
        assert skipped_share(common_english, affix.phones, "suffix", content, True) <= 0.1
    # NG cannot follow most stems, so it is never drawn as a suffix
    assert skipped_share(common_english, ("NG",), "suffix", content, True) > 0.1
    assert all(a.phones != ("NG",) for a in run.lexicon.affixes)
    loose = run_forms(config_with({"affixes": {"max_skipped": 1.0}}, count=60))
    rejected = loose.lexicon.closed_class["affixes"]["candidates_rejected_for_skipped_stems"]
    assert sum(rejected.values()) == 0


def test_an_inflected_form_that_is_a_common_word_is_skipped(common_english):
    found = False
    for seed in (1, 2, 3, 4):
        run = run_forms(config_with({"inflect": INFLECT_ALL}, count=200, seed=seed))
        report = run.lexicon.closed_class["inflected"]
        for form in forms_of(run, "inflected"):
            assert not common_english.is_common_pronunciation(form.phones)
        skipped = [s for s in report["skipped"] if "common English word" in s["reason"]]
        assert report["skipped_as_common_words"] == len(skipped)
        content = {w.label: w for w in forms_of(run, "content")}
        affixes = {a.label: a for a in run.lexicon.affixes}
        for skip in skipped:
            found = True
            from semantic_world.wordforms.closed_class import repair_join

            phones, _ = repair_join(
                common_english, content[skip["stem"]].phones, affixes[skip["affix"]], True
            )
            word = common_english.common_words_with_pronunciation(phones)[0]
            assert skip["reason"].endswith(f"{word!r}")
    assert found


def test_english_function_words(common_english):
    run = run_forms(config_with({"function_words": {"source": "english"}}))
    function = forms_of(run, "function")
    assert [w.gloss for w in function] == list(run.config.closed_class.glosses)
    by_gloss = {w.gloss: w for w in function}
    assert by_gloss["the"].arpabet == "DH AH1"
    assert by_gloss["the"].weak_forms == ("DH AH0", "DH IY0")
    assert by_gloss["and"].arpabet == "AE1 N D" and by_gloss["and"].weak_forms == ("AH0 N D",)
    assert by_gloss["not"].arpabet == "N AA1 T" and by_gloss["not"].weak_forms == ()
    assert by_gloss["without"].arpabet == "W IH0 TH AW1 T"
    for word in function:
        assert word.real_word is True and word.spelling == word.gloss
        assert common_english.is_common_pronunciation(word.phones)
        assert word.ipa and word.espeak and word.kind == "function"
    assert run.lexicon.closed_class["function_words"] == {"source": "english", "count": 15}
    assert not any(k.startswith("F.") for k in run.lexicon.closed_class["english_words"])
    with pytest.raises(GenerationError, match="'blorp' has no English pronunciation"):
        run_forms(config_with({"function_words": {"source": "english", "glosses": ["blorp"]}}))


def test_english_affixes_and_allomorphy(common_english):
    from semantic_world.wordforms.closed_class import english_allomorph

    assert english_allomorph("PLURAL", ("K", "AE1", "T")) == ("S",)
    assert english_allomorph("PLURAL", ("D", "AO1", "G")) == ("Z",)
    assert english_allomorph("PLURAL", ("B", "AH1", "S")) == ("IH0", "Z")
    assert english_allomorph("PLURAL", ("JH", "AH1", "JH")) == ("IH0", "Z")
    assert english_allomorph("PLURAL", ("T", "R", "IY1")) == ("Z",)
    assert english_allomorph("PAST", ("W", "AO1", "K")) == ("T",)
    assert english_allomorph("PAST", ("R", "AH1", "B")) == ("D",)
    assert english_allomorph("PAST", ("W", "EY1", "T")) == ("IH0", "D")
    assert english_allomorph("PAST", ("N", "IY1", "D")) == ("IH0", "D")
    assert english_allomorph("PROGRESSIVE", ("W", "AO1", "K")) == ("IH0", "NG")
    config = config_with({"affixes": {"source": "english"}, "inflect": INFLECT_ALL}, count=40)
    run = run_forms(config)
    affixes = {a.gloss: a for a in run.lexicon.affixes}
    assert affixes["PLURAL"].arpabet == "Z / S / IH0 Z" and affixes["PLURAL"].ipa == "z / s / ɪz"
    assert affixes["PAST"].arpabet == "D / T / IH0 D"
    assert affixes["PROGRESSIVE"].arpabet == "IH0 NG" and affixes["PROGRESSIVE"].ipa == "ɪŋ"
    content = {w.label: w for w in forms_of(run, "content")}
    by_label = {a.label: a for a in run.lexicon.affixes}
    inflected = forms_of(run, "inflected")
    assert len(inflected) > 100
    for form in inflected:
        stem = content[form.stem]
        gloss = by_label[form.affix].gloss
        assert form.phones == stem.phones + english_allomorph(gloss, stem.phones)
        assert form.join == "none"
        assert not common_english.is_common_pronunciation(form.phones)
    report = run.lexicon.closed_class
    assert report["affixes"] == {"source": "english", "count": 3}
    # English joins are not checked against the trigrams: only common words are skipped
    assert all("common English" in s["reason"] for s in report["inflected"]["skipped"])
    with pytest.raises(GenerationError, match="'DUAL' has no English equivalent"):
        run_forms(config_with({"affixes": {"source": "english", "items": [{"gloss": "DUAL"}]}}))
    with pytest.raises(ConfigError) as info:
        config_with(
            {"affixes": {"source": "english", "items": [{"gloss": "PAST", "position": "prefix"}]}}
        )
    assert info.value.field == "closed_class.affixes.items[0].position"
    with pytest.raises(ConfigError) as info:
        config_with({"affixes": {"source": "french"}})
    assert info.value.field == "closed_class.affixes.source"


def test_english_closed_class_in_the_run_folder(tmp_path):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["closed_class"]["function_words"] = {"source": "english"}
    data["closed_class"]["affixes"] = {"source": "english"}
    config = tmp_path / "english.yaml"
    config.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["forms", str(config), "--out", str(out)]) == 0
    words = pl.read_csv(out / "words.csv")
    assert words.columns == list(WORD_COLUMNS)
    function = words.filter(pl.col("kind") == "function")
    assert function["spelling"].to_list() == function["gloss"].to_list()
    assert function["real_word"].all()
    assert function.filter(pl.col("gloss") == "the")["weak_forms"].item() == "DH AH0; DH IY0"
    assert function.filter(pl.col("gloss") == "not")["weak_forms"].item() is None
    others = words.filter(pl.col("kind") != "function")
    assert others["weak_forms"].null_count() == others.height
    affixes = pl.read_csv(out / "affixes.csv")
    assert affixes["arpabet"].to_list() == ["Z / S / IH0 Z", "D / T / IH0 D", "IH0 NG"]
    reloaded = load_config(out / "config.yaml").closed_class
    assert reloaded.function_source == "english" and reloaded.affix_source == "english"


def test_the_glide_repair_in_a_run(common_english):
    """A vowel-final stem with a vowel-initial affix takes a glide where the plain join fails."""
    from semantic_world.wordforms.closed_class import affix_candidates, skipped_share

    config = config_with(
        {
            "affixes": {"items": [{"gloss": "A"}], "shapes": {"V": 1}},
            "inflect": [{"words": "all", "affixes": ["A"]}],
        },
        count=200,
    )
    run = run_forms(config)
    affix = run.lexicon.affixes[0]
    assert affix.shape == "V"
    content = {w.label: w for w in forms_of(run, "content")}
    glided = [w for w in forms_of(run, "inflected") if w.join == "glide"]
    for form in glided:
        stem = content[form.stem]
        vowel = stem.phones[-1]
        assert is_vowel(vowel)
        expected = "W" if vowel[:-1] in ("UW", "UH", "OW", "AO", "AW") else "Y"
        assert form.phones == stem.phones + (expected,) + affix.phones
        assert form.spelling.startswith(stem.spelling)
    assert run.lexicon.closed_class["inflected"]["joins"]["glide"] == len(glided)
    # the glide lets some VC suffix candidates pass the 10% rule
    stems = list(content.values())
    passing = [
        p
        for p, _ in affix_candidates(common_english, "VC", "suffix")
        if skipped_share(common_english, p, "suffix", stems, True) <= 0.1
    ]
    assert passing and all(p[0] == "AH0" for p in passing)
    with pytest.raises(ConfigError) as info:
        config_with({"affixes": {"glide": "L"}})
    assert info.value.field == "closed_class.affixes.glide"
