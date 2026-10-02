"""Stage 7: sound-meaning assignment: target correlation, branch markers, and acoustic mapping."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest
import yaml
from test_wordforms_synth import ToneEngine
from wordforms_support import (
    DATA,
    VOICE_DIR,
    needs_audio,
    needs_cmudict,
    needs_espeak,
    needs_parselmouth,
    needs_piper,
    needs_torch,
    needs_wordfreq,
    plain_rows,
)

from semantic_world.wordforms import Run, run_assignment, run_forms
from semantic_world.wordforms.__main__ import main
from semantic_world.wordforms.assign import (
    AssignmentError,
    assign,
    assign_arbitrary,
    assign_target_correlation,
    branch_of,
    load_meaning_table,
    load_meanings,
    marker_candidates,
    meaning_distances,
)
from semantic_world.wordforms.config import ConfigError, parse_config
from semantic_world.wordforms.english import edit_distance, is_vowel, strip_stress
from semantic_world.wordforms.synth import synthesize_lexicon

pytestmark = [needs_cmudict, needs_wordfreq]


def taxonomy_file(tmp_path, scalar: bool = False):
    """A small taxonomy-like meanings table: 3 branches of 4 categories each (a parent and three
    leaves), with ISA columns and random binary features; optionally a scalar column."""
    rng = np.random.default_rng(0)
    ids = [f"C{b}" + (f".{k}" if k else "") for b in (1, 2, 3) for k in range(4)]
    table: dict[str, list] = {"label": ids}
    for branch in (1, 2, 3):
        table[f"ISA.C{branch}"] = [int(i.split(".")[0] == f"C{branch}") for i in ids]
    for j in range(12):
        # features are mostly shared within a branch, so meaning distance follows the taxonomy
        base = rng.integers(0, 2, size=3)
        table[f"IS.{j + 1}"] = [int(base[int(i[1]) - 1] ^ (rng.random() < 0.15)) for i in ids]
    if scalar:
        table["SIZE"] = [round(float(x), 3) for x in rng.uniform(0, 10, size=len(ids))]
    path = tmp_path / "meanings.csv"
    pl.DataFrame(table).write_csv(path)
    return path


def assign_config(tmp_path, count: int = 60, **assignment):
    return parse_config(
        {
            "name": "assign_test",
            "wordforms": {"count": count},
            "synthesis": {
                "cache_dir": str(tmp_path / "cache"),
                "tokens_per_speaker": 1,
                "held_out_speaker_proportion": 0.34,
                "engines": {"piper": None, "espeak": {"variants": ["m1", "m3", "f2"]}},
            },
            "embeddings": [
                {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel", "pca_dims": 8}
            ],
            "closed_class": None,
            "assignment": {"meanings": str(taxonomy_file(tmp_path)), **assignment},
        },
        "assign_test",
    )


# ---------------------------------------------------------------------------------------------
# Configuration and meanings
# ---------------------------------------------------------------------------------------------


def test_assignment_configuration(tmp_path):
    default = parse_config({}, "x").assignment
    assert default.mode == "arbitrary" and default.meanings is None and default.categories == "all"
    assert default.sound_distance == "edit" and default.meaning_distance == "hamming"
    assert default.target == 0.3 and default.marker_depth == 1 and default.acoustic == ()
    assert default.strict is False
    config = parse_config(
        {
            "embeddings": [{"name": "e", "encoder": "fixed", "frontend": "logmel"}],
            "assignment": {
                "mode": "target_correlation",
                "meanings": "m.csv",
                "categories": "leaves",
                "sound_distance": "e",
                "meaning_distance": "jaccard",
                "strict": True,
                "target_correlation": {"target": -0.2, "tolerance": 0.05, "max_swaps": 100},
                "branch_markers": {"depth": 2, "position": "final", "shape": "VC"},
                "acoustic_mapping": [{"feature": "IS.1", "property": "pitch", "amount": 2}],
            },
        },
        "x",
    )
    a = config.assignment
    assert a.categories == "leaves" and a.sound_distance == "e" and a.meaning_distance == "jaccard"
    assert (a.target, a.tolerance, a.max_swaps) == (-0.2, 0.05, 100) and a.strict is True
    assert (a.marker_depth, a.marker_position, a.marker_shape) == (2, "final", "VC")
    assert a.acoustic[0].feature == "IS.1" and a.acoustic[0].amount == 2.0
    assert parse_config(config.resolved(), "x") == config
    for section, field in (
        ({"mode": "systematic"}, "assignment.mode"),
        ({"strict": "yes"}, "assignment.strict"),
        ({"categories": "some"}, "assignment.categories"),
        ({"sound_distance": "hubert"}, "assignment.sound_distance"),
        ({"meaning_distance": "euclid"}, "assignment.meaning_distance"),
        ({"target_correlation": {"target": 2}}, "assignment.target_correlation.target"),
        ({"branch_markers": {"depth": 0}}, "assignment.branch_markers.depth"),
        ({"branch_markers": {"shape": "V"}}, "assignment.branch_markers.shape"),
        ({"acoustic_mapping": [{"feature": "IS.1", "property": "loudness", "amount": 1}]}, "assignment.acoustic_mapping[0].property"),
        ({"acoustic_mapping": [{"feature": "IS.1", "property": "duration", "amount": 0}]}, "assignment.acoustic_mapping[0].amount"),
        ({"mode": "acoustic_mapping"}, "assignment.acoustic_mapping"),
        ({"mode": "branch_markers", "sound_distance": "e"}, "assignment.sound_distance"),
    ):  # fmt: skip
        with pytest.raises(ConfigError) as info:
            parse_config(
                {"embeddings": config.resolved()["embeddings"], "assignment": section}, "x"
            )
        assert info.value.field == field, info.value


def test_meanings_tables_drop_scalar_columns_and_select_categories(tmp_path):
    path = taxonomy_file(tmp_path, scalar=True)
    with pytest.raises(AssignmentError, match="only 0 and 1"):
        load_meanings(path)
    with pytest.raises(AssignmentError, match="only 0 and 1"):
        load_meaning_table(path, "error")
    table = load_meaning_table(path, "drop")
    assert table.dropped == ["SIZE"] and "SIZE" not in table.names
    assert table.features.shape == (12, 15) and set(np.unique(table.features)) == {0, 1}
    assert table.feature("ISA.C2").tolist() == [0] * 4 + [1] * 4 + [0] * 4
    with pytest.raises(AssignmentError, match="no feature column 'NOPE'"):
        table.feature("NOPE")
    leaves = table.select("leaves")
    assert leaves.ids == [f"C{b}.{k}" for b in (1, 2, 3) for k in (1, 2, 3)]
    assert leaves.features.shape == (9, 15)
    chosen = table.select(("C2", "C1.3"))
    assert chosen.ids == ["C2", "C1.3"]
    with pytest.raises(AssignmentError, match="no ID 'C9'"):
        table.select(("C9",))
    assert table.select("all") is table
    assert branch_of("C1.2.3", 1) == "C1" and branch_of("C1.2.3", 2) == "C1.2"
    assert branch_of("C1", 2) is None


def test_meaning_distances():
    features = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    hamming = meaning_distances(features, "hamming")
    assert hamming.tolist() == [0.25, 1.0, 0.5, 0.75, 0.25, 0.5]
    jaccard = meaning_distances(features, "jaccard")
    assert jaccard[0] == 0.5 and jaccard[1] == 1.0  # one shared of two; none shared
    cosine = meaning_distances(features, "cosine")
    assert cosine[0] == pytest.approx(1 - 1 / np.sqrt(2)) and cosine[1] == 1.0
    assert np.isfinite(cosine).all() and np.isfinite(jaccard).all()  # the empty meaning too
    with pytest.raises(ValueError):
        meaning_distances(features, "euclid")


# ---------------------------------------------------------------------------------------------
# Arbitrary and target correlation
# ---------------------------------------------------------------------------------------------


def test_arbitrary_assignments_have_a_mean_correlation_near_zero(tmp_path):
    config = assign_config(tmp_path)
    words = run_forms(config).lexicon.content
    ids, features = load_meanings(config.assignment.meanings)
    values = [
        assign_arbitrary(words, ids, features, np.random.default_rng(seed), 10).summary[
            "correlation"
        ]
        for seed in range(60)
    ]
    assert abs(np.mean(values)) < 0.05 and np.std(values) > 0.02
    # the other meaning distances work the same way
    for kind in ("cosine", "jaccard"):
        result = assign_arbitrary(
            words, ids, features, np.random.default_rng(0), 10, meaning_distance=kind
        )
        assert result.summary["meaning_distance"] in ("cosine", "Jaccard")
        assert -1 <= result.summary["correlation"] <= 1


def test_target_correlation_reaches_its_target_or_reports_the_closest_value(tmp_path):
    config = assign_config(tmp_path)
    words = run_forms(config).lexicon.content
    ids, features = load_meanings(config.assignment.meanings)
    for target in (0.3, -0.2, 0.0):
        result = assign_target_correlation(
            words, ids, features, np.random.default_rng(1), target, 0.01, 20000, 50
        )
        report = result.summary["target_correlation"]
        assert report["reached"] is True and report["target"] == target
        assert abs(result.summary["correlation"] - target) <= 0.01
        assert len(set(w.label for w in result.words)) == len(ids) == 12  # no word is used twice
        assert report["accepted"] <= report["proposals"] <= 20000
    # a target that the words cannot give: the closest value reached is reported
    result = assign_target_correlation(
        words, ids, features, np.random.default_rng(1), 0.99, 0.01, 3000, 50
    )
    report = result.summary["target_correlation"]
    assert report["reached"] is False and report["proposals"] == 3000
    assert report["start"] < result.summary["correlation"] < 0.99
    # the miss is a reported result, with a warning in the summary
    assert report["gap"] == pytest.approx(0.99 - result.summary["correlation"], abs=2e-6)
    assert report["gap"] > report["tolerance"]
    assert report["warning"].startswith("the target correlation 0.99 was not reached")
    assert str(result.summary["correlation"]) in report["warning"]
    # with strict, the same miss is an error
    with pytest.raises(AssignmentError, match=r"0\.99 was not reached.*assignment\.strict"):
        assign_target_correlation(
            words, ids, features, np.random.default_rng(1), 0.99, 0.01, 3000, 50, strict=True
        )
    # a target that is reached has no warning, strict or not
    reached = assign_target_correlation(
        words, ids, features, np.random.default_rng(1), 0.3, 0.01, 20000, 50, strict=True
    )
    assert reached.summary["target_correlation"]["reached"] is True
    assert "warning" not in reached.summary["target_correlation"]
    assert reached.summary["target_correlation"]["gap"] <= 0.01
    assert result.summary["null"]["p_value"] < 0.05  # far outside the random assignments
    # every accepted exchange moved toward the target, so more proposals never end further away
    short = assign_target_correlation(
        words, ids, features, np.random.default_rng(1), 0.99, 0.01, 300, 10
    )
    assert result.summary["correlation"] >= short.summary["correlation"]
    # seeded
    again = assign_target_correlation(
        words, ids, features, np.random.default_rng(1), 0.3, 0.01, 20000, 50
    )
    first = assign_target_correlation(
        words, ids, features, np.random.default_rng(1), 0.3, 0.01, 20000, 50
    )
    assert [w.label for w in again.words] == [w.label for w in first.words]
    with pytest.raises(AssignmentError, match="only 5 words"):
        assign_target_correlation(words[:5], ids, features, np.random.default_rng(0), 0.3)


def test_an_unreached_target_warns_on_the_command_line_or_stops_when_strict(tmp_path, capsys):
    data = assign_config(
        tmp_path, mode="target_correlation", null_samples=20,
        target_correlation={"target": 0.99, "max_swaps": 300},
    ).resolved()  # fmt: skip
    path = tmp_path / "assign.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["assign", str(path), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert (
        "target 0.99 (not reached: the closest value is given) after 300 proposals" in captured.out
    )
    assert (
        "warning: the target correlation 0.99 was not reached: the closest value is" in captured.err
    )
    report = yaml.safe_load((out / "assignment" / "summary.yaml").read_text())["target_correlation"]
    assert report["reached"] is False and report["gap"] > 0.01 and "warning" in report
    assert (out / "assignment" / "lexicon.csv").exists()
    # strict: the run stops, and nothing is written
    data["assignment"]["strict"] = True
    path.write_text(yaml.safe_dump(data))
    assert main(["assign", str(path), "--out", str(tmp_path / "strict")]) == 1
    captured = capsys.readouterr()
    assert "error: the target correlation 0.99 was not reached" in captured.err
    assert "assignment.strict" in captured.err
    assert not (tmp_path / "strict").exists()
    # strict with a target that is reached: no warning and no error
    data["assignment"]["target_correlation"] = {"target": 0.2, "max_swaps": 20000}
    path.write_text(yaml.safe_dump(data))
    assert main(["assign", str(path), "--out", str(tmp_path / "reached")]) == 0
    captured = capsys.readouterr()
    assert "target 0.2 (reached)" in captured.out and "warning" not in captured.err


@needs_audio
def test_target_correlation_with_an_embeddings_distance(tmp_path):
    from test_wordforms_embeddings import WordEngine

    from semantic_world.wordforms.embeddings import compute_embeddings
    from semantic_world.wordforms.frontends import compute_frontends

    config = assign_config(
        tmp_path, count=30, mode="target_correlation", sound_distance="logmel_fixed",
        target_correlation={"target": 0.25, "tolerance": 0.02},
    )  # fmt: skip
    run = run_forms(config)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": WordEngine()}, check=False
    )
    run.frontends = compute_frontends(config, run.synthesis, tmp_path / "run")
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, run.synthesis, run.frontends, tmp_path / "run"
    )
    run_assignment(run, tmp_path / "run")
    summary = run.assignment.summary
    assert summary["sound_distance"] == "cosine distance of logmel_fixed"
    assert summary["target_correlation"]["reached"] and abs(summary["correlation"] - 0.25) <= 0.02
    # without the embeddings, the assignment says what it needs
    with pytest.raises(AssignmentError, match="needs the run's embeddings"):
        assign(config, run.lexicon.content, np.random.default_rng(0))


# ---------------------------------------------------------------------------------------------
# Branch markers
# ---------------------------------------------------------------------------------------------


def test_marker_candidates(common_english):
    for shape in ("CV", "VC", "CVC"):
        for position in ("initial", "final"):
            candidates = marker_candidates(common_english, shape, position)
            assert len(candidates) > 20
            assert sum(w for _, w in candidates) == pytest.approx(1.0)
            for phones, _ in candidates:
                assert "".join("V" if is_vowel(p) else "C" for p in phones) == shape
                assert all(p.endswith("0") for p in phones if is_vowel(p))  # unstressed


def marker_distances(assignment) -> list[int]:
    """The phoneme edit distance between every two markers of an assignment."""
    plain = [strip_stress(m.phones) for m in assignment.markers]
    return [edit_distance(a, b) for i, a in enumerate(plain) for b in plain[i + 1 :]]


def test_branch_markers_differ_by_at_least_the_minimum_distance(tmp_path):
    """``assignment.branch_markers.min_distance`` (default 2): markers that differ in one
    phoneme alone, such as ``R IY0`` and ``R IH0``, would be nearly impossible to tell apart."""
    config = assign_config(tmp_path, mode="branch_markers")
    assert config.assignment.marker_min_distance == 2
    assert config.resolved()["assignment"]["branch_markers"]["min_distance"] == 2
    assert parse_config(config.resolved(), "assign_test") == config
    with pytest.raises(ConfigError) as info:
        assign_config(tmp_path, mode="branch_markers", branch_markers={"min_distance": 0})
    assert info.value.field == "assignment.branch_markers.min_distance"

    def distances(minimum: int, seed: int) -> list[int]:
        config = assign_config(
            tmp_path, count=150, mode="branch_markers", null_samples=5,
            branch_markers={"depth": 2, "min_distance": minimum},
        ).with_seed(seed)  # fmt: skip
        run = run_forms(config)
        run_assignment(run)
        assert len(run.assignment.markers) == 9
        assert run.assignment.summary["branch_markers"]["min_distance"] == minimum
        return marker_distances(run.assignment)

    seeds = range(1, 7)
    # with a minimum of 1, markers one phoneme apart are drawn
    assert any(min(distances(1, seed)) == 1 for seed in seeds)
    # with the default, never
    assert all(min(distances(2, seed)) >= 2 for seed in seeds)
    # a CV marker has two phonemes, so no two are three phonemes apart: the error says so
    with pytest.raises(AssignmentError) as error:
        distances(3, 1)
    message = str(error.value)
    assert "closer than 3 phonemes" in message and "branch_markers.min_distance" in message
    assert "C1.2" in message  # the first branch got its marker, and the second could not
    # a longer shape has room
    config = assign_config(
        tmp_path, count=150, mode="branch_markers", null_samples=5,
        branch_markers={"shape": "CVC", "min_distance": 3},
    )  # fmt: skip
    run = run_forms(config)
    run_assignment(run)
    assert min(marker_distances(run.assignment)) == 3


@pytest.mark.parametrize(
    ("depth", "position", "shape"), [(1, "initial", "CV"), (1, "final", "CVC"), (2, "final", "VC")]
)
def test_every_word_in_a_branch_carries_the_branch_marker(
    tmp_path, common_english, depth, position, shape
):
    config = assign_config(
        tmp_path, count=120, mode="branch_markers",
        branch_markers={"depth": depth, "position": position, "shape": shape},
    )  # fmt: skip
    run = run_forms(config)
    content = len(run.lexicon.words)
    run_assignment(run)
    assignment = run.assignment
    markers = {m.label: m for m in assignment.markers}
    assert len({m.phones for m in markers.values()}) == len(markers)
    # two markers differ by at least two phonemes, so that they can be told apart when spoken
    assert min(marker_distances(assignment)) >= 2
    assert assignment.summary["branch_markers"]["min_distance"] == 2
    frame = assignment.frame()
    branches = {}
    for row, word, base in zip(
        frame.iter_rows(named=True), assignment.words, assignment.base_words, strict=True
    ):
        branch = branch_of(row["meaning"], depth)
        assert row["branch"] == branch and row["base_word"] == base.label
        if branch is None:
            assert word is base and row["marker"] is None and word.kind == "content"
            continue
        marker = markers[row["marker"]]
        branches.setdefault(branch, set()).add(marker.label)
        assert assignment.marker_branches[marker.label] == branch
        # the word carries the marker: at its start or at its end, around the base word
        n = len(marker.phones)
        if position == "initial":
            assert (
                word.phones[:n] == marker.phones and word.phones[-len(base.phones) :] == base.phones
            )
        else:
            assert (
                word.phones[-n:] == marker.phones and word.phones[: len(base.phones)] == base.phones
            )
        assert word.kind == "marked" and word.stem == base.label and word.affix == marker.label
        assert word.label == f"{base.label}.{marker.label}" and word.held_out == base.held_out
        assert common_english.phonotactic(word.phones)
        assert not common_english.is_common_pronunciation(word.phones)
        assert word.join in ("none", "schwa", "glide") and word.ipa and word.spelling
    assert all(len(found) == 1 for found in branches.values())  # one marker for each branch
    expected = 3 if depth == 1 else 9
    assert len(branches) == len(markers) == expected
    assert len({w.label for w in assignment.base_words}) == 12  # no word is used twice
    # the marked forms join the run's word forms, after the content words
    assert len(run.lexicon.words) == content + len(assignment.marked)
    assert [w.kind for w in run.lexicon.words[content:]] == ["marked"] * len(assignment.marked)
    assert len({w.stripped for w in run.lexicon.words}) == len(run.lexicon.words)
    report = assignment.summary["branch_markers"]
    assert report["branches"] == expected and report["marked_words"] == len(assignment.marked)
    assert report["unmarked_meanings"] == (0 if depth == 1 else 3)
    assert sum(report["joins"].values()) == len(assignment.marked)
    if depth == 1 and shape != "VC":
        # a shared syllable makes the words of a branch sound alike: the correlation rises
        assert assignment.summary["correlation"] > report["unmarked_correlation"] + 0.1


@needs_audio
def test_branch_marker_run_writes_the_marked_forms_and_synthesizes_them(tmp_path):
    config = assign_config(tmp_path, count=60, mode="branch_markers")
    run = run_forms(config)
    run_assignment(run)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": ToneEngine()}, check=False
    )
    marked = [t for t in run.synthesis.tokens if ".M." in t.word]
    assert len(marked) == 12 * 3 and run.synthesis.tokens[0].word == "W.1"
    folder = run.write(tmp_path / "run")
    words = pl.read_csv(folder / "words.csv")
    assert words.filter(pl.col("kind") == "marked").height == 12
    assert words.filter(pl.col("kind") == "marked")["stem"].str.starts_with("W.").all()
    lexicon = pl.read_csv(folder / "assignment" / "lexicon.csv")
    assert lexicon.columns == [
        "meaning",
        "word",
        "spelling",
        "arpabet",
        "base_word",
        "branch",
        "marker",
    ]
    assert lexicon["word"].str.contains(".M.", literal=True).all()
    markers = pl.read_csv(folder / "assignment" / "markers.csv")
    assert markers["label"].to_list() == ["M.1", "M.2", "M.3"]
    assert (
        markers["branch"].to_list() == ["C1", "C2", "C3"]
        and markers["position"].to_list() == ["prefix"] * 3
    )
    summary = yaml.safe_load((folder / "assignment" / "summary.yaml").read_text())
    assert summary["mode"] == "branch_markers" and summary["branch_markers"]["branches"] == 3


def test_markers_avoid_the_closed_class_forms(tmp_path):
    data = assign_config(tmp_path, count=60, mode="branch_markers").resolved()
    data["closed_class"] = {"inflect": [{"words": "all", "affixes": ["PLURAL", "PAST"]}]}
    for position, shape in (("final", "VC"), ("initial", "CV"), ("final", "CVC")):
        data["assignment"]["branch_markers"] = {"depth": 2, "position": position, "shape": shape}
        run = run_forms(parse_config(data, "assign_test"))
        before = list(run.lexicon.words)
        run_assignment(run)
        closed = [w.stripped for w in before if w.kind == "function"]
        closed += [strip_stress(form) for affix in run.lexicon.affixes for form in affix.forms]
        schwa = ("AH",)
        for marker in run.assignment.markers:
            plain = strip_stress(marker.phones)
            assert plain not in closed
            assert schwa + plain not in closed and plain + schwa not in closed
            assert plain[1:] not in closed or plain[0] != "AH"
            assert plain[:-1] not in closed or plain[-1] != "AH"
        # no marked form is a function word, an inflected form, or a content word
        assert len(run.assignment.marked) == 9
        forms = [w.stripped for w in run.lexicon.words]
        assert len(set(forms)) == len(forms)
        assert len({w.spelling for w in run.lexicon.words}) == len(run.lexicon.words)
        # the closed-class forms are the ones the run had before the assignment
        assert run.lexicon.words[: len(before)] == before


# ---------------------------------------------------------------------------------------------
# Acoustic mapping
# ---------------------------------------------------------------------------------------------


class VoiceEngine(ToneEngine):
    """A stand-in engine with a voice that Praat can measure: harmonics of a pitch that the
    speaker sets, through three resonances that the word sets."""

    def synthesize(self, phonemes, settings):
        from scipy.signal import lfilter

        self.calls += 1
        voice = hashlib.sha256(settings["voice"].encode()).digest()
        word = hashlib.sha256(phonemes.encode()).digest()
        rate = 22050
        n = int(rate * 0.45 * settings["scale"])
        t = np.arange(n) / rate
        f0 = (110 + voice[0] / 3) * (1.0 + 0.15 * t / t[-1])
        phase = 2 * np.pi * np.cumsum(f0) / rate
        source = sum(np.sin(k * phase) / k for k in range(1, 30))
        out = np.zeros(n)
        for k, center in enumerate((500 + word[0], 1400 + 3 * word[1], 2500 + 2 * word[2])):
            r = np.exp(-np.pi * 90 / rate)
            theta = 2 * np.pi * center / rate
            out += lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], source) / (k + 1)
        out *= 0.3 / np.abs(out).max() * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))
        silence = np.zeros(2205)
        return np.concatenate([silence, out, silence]).astype(np.float32), rate


def voice_clip(rate: int = 16000) -> np.ndarray:
    from semantic_world.wordforms.synth import audio as audio_tools

    clip, engine_rate = VoiceEngine().synthesize("v OY s", {"voice": "en+m1", "scale": 1.0})
    return audio_tools.resample(clip, engine_rate, rate)


@needs_audio
def test_spectral_tilt_tools():
    from semantic_world.wordforms.synth import audio as audio_tools

    clip = voice_clip()
    before = audio_tools.spectral_tilt(clip, 16000)
    for amount in (-6.0, -3.0, 3.0):
        changed = audio_tools.change_tilt(clip, 16000, amount)
        assert len(changed) == len(clip)
        assert audio_tools.spectral_tilt(changed, 16000) - before == pytest.approx(amount, abs=0.1)
    assert np.allclose(audio_tools.change_tilt(clip, 16000, 0.0), clip, atol=1e-6)


@needs_audio
@needs_parselmouth
def test_the_formant_shift_measure_does_not_use_the_target():
    from semantic_world.wordforms import praat

    clip = voice_clip()
    assert praat.measure_formant_shift(clip, clip, 16000) == pytest.approx(1.0, abs=1e-6)
    for ratio in (0.9, 1.15):
        shifted = praat.manipulate(clip, 16000, formant_shift_ratio=ratio)
        assert praat.measure_formant_shift(clip, shifted, 16000) == pytest.approx(ratio, rel=0.04)
    # a copy with another duration is matched by relative time, and a span can be measured
    longer = praat.manipulate(clip, 16000, formant_shift_ratio=1.15, duration_factor=1.3)
    assert len(longer) == pytest.approx(1.3 * len(clip), rel=0.02)
    assert praat.measure_formant_shift(clip, longer, 16000) == pytest.approx(1.15, rel=0.04)
    span = praat.measure_formant_shift(clip, shifted, 16000, time_range=(0.2, 0.4))
    assert span == pytest.approx(1.15, rel=0.04)
    assert np.isnan(praat.measure_formant_shift(np.zeros(8000), np.zeros(8000), 16000))


@needs_audio
@needs_parselmouth
def test_the_pitch_shift_measure():
    from semantic_world.wordforms import praat

    clip = voice_clip()
    assert praat.measure_pitch_shift(clip, clip, 16000) == 0.0
    for semitones in (-3.0, 2.0):
        shifted = praat.change_pitch(clip, 16000, factor=2.0 ** (semitones / 12.0))
        assert praat.measure_pitch_shift(clip, shifted, 16000) == pytest.approx(semitones, abs=0.05)
    # a copy with another duration is matched by relative time, and a span can be measured
    longer = praat.change_duration(shifted, 16000, 1.3)
    assert praat.measure_pitch_shift(clip, longer, 16000) == pytest.approx(2.0, abs=0.1)
    span = praat.measure_pitch_shift(clip, shifted, 16000, time_range=(0.2, 0.4))
    assert span == pytest.approx(2.0, abs=0.05)
    assert np.isnan(praat.measure_pitch_shift(np.zeros(8000), np.zeros(8000), 16000))


MAPPINGS = [
    {"feature": "ISA.C1", "property": "pitch", "amount": 3.0},
    {"feature": "ISA.C2", "property": "duration", "amount": 1.25},
    {"feature": "ISA.C3", "property": "tilt", "amount": -4.0},
    {"feature": "IS.1", "property": "formants", "amount": 1.15},
]


def mapped_run(tmp_path, mappings=MAPPINGS, **sections) -> Run:
    """A run with acoustic mapping on the stand-in voice, through the synthesis, the mapping,
    and the augmentation (in the order of ``run_synthesis``)."""
    from semantic_world.wordforms.augment import augment_synthesis
    from semantic_world.wordforms.mapping import map_tokens

    config = assign_config(tmp_path, count=30, mode="acoustic_mapping", acoustic_mapping=mappings)
    config = parse_config({**config.resolved(), **sections}, "assign_test")
    run = run_forms(config)
    run_assignment(run)
    run.synthesis = synthesize_lexicon(
        config, run.streams, run.lexicon.words, engines={"espeak": VoiceEngine()}, check=False
    )
    run.assignment.summary["acoustic_mapping"] = map_tokens(config, run.synthesis, run.assignment)
    if config.augmentation is not None:
        augment_synthesis(config, run.streams, run.synthesis)
    return run


@needs_audio
@needs_parselmouth
def test_acoustic_mapping_produces_the_configured_shifts_as_measured(tmp_path):
    run = mapped_run(tmp_path)
    assignment, synthesis = run.assignment, run.synthesis
    table = assignment.features
    meaning_of = {w.label: i for i, w in enumerate(assignment.words)}
    originals = {t.label: t for t in synthesis.tokens if not t.mapping}
    mapped = [t for t in synthesis.tokens if t.mapping]
    assert len(originals) == 30 * 3
    for token in mapped:
        record = json.loads(token.mapping)
        achieved = json.loads(token.achieved)
        source = originals[record["source"]]
        assert token.label == f"{source.label}.M" and set(record) == {
            "source",
            "meaning",
            "mappings",
        }
        assert token.word == source.word and token.speaker == source.speaker
        assert token.cache_path.startswith("v2/mapping/")
        # the mapped token is the word's token, and its unmapped original is a control token
        assert token.clean and not token.control and not token.augmentation
        assert source.control and not source.clean and not source.mapping
        row = meaning_of[token.word]
        assert record["meaning"] == assignment.meanings[row]
        # exactly the mappings whose feature the word's meaning has
        active = [m for m in MAPPINGS if table.feature(m["feature"])[row] == 1]
        assert record["mappings"] == [{**m, "amount": float(m["amount"])} for m in active]
        assert len(achieved) == len(active) and active
        for entry in achieved.values():
            assert set(entry) == {"target", "measured", "miss"}
        if any(m["property"] == "duration" for m in active):
            assert token.duration == pytest.approx(source.duration * 1.25, rel=0.02)
    # every token of a word with a mapped feature is mapped, and no other token
    expected = sum(
        any(table.feature(m["feature"])[meaning_of[t.word]] == 1 for m in MAPPINGS)
        for t in originals.values()
        if t.word in meaning_of
    )
    assert len(mapped) == expected > 0
    # a token of a word without a meaning is neither mapped nor a control token
    assert sum(t.control for t in originals.values()) == len(mapped)
    assert all(t.clean for t in originals.values() if t.word not in meaning_of)
    # the analysis uses the achieved values: the measured shifts are the configured ones
    report = assignment.summary["acoustic_mapping"]
    assert report["tokens_of_assigned_words"] == 12 * 3
    assert report["mapped_tokens"] == report["control_tokens"] == len(mapped)
    assert report["skipped"] == []
    by = {m["property"]: m for m in report["mappings"]}
    assert by["pitch"]["quantity"] == "pitch_semitones"
    assert by["pitch"]["achieved_mean"] == pytest.approx(3.0, rel=0.02)
    assert by["duration"]["achieved_mean"] == pytest.approx(1.25, rel=0.01)
    assert by["tilt"]["achieved_mean"] == pytest.approx(-4.0, rel=0.02)
    assert by["formants"]["achieved_mean"] == pytest.approx(1.15, rel=0.05)
    assert by["formants"]["over_10_percent"] == 0 and by["formants"]["feature_correlation"] > 0.95
    for name in ("pitch", "duration", "tilt"):
        assert by[name]["tokens"] == 12 and by[name]["over_10_percent"] == 0
        assert abs(by[name]["feature_correlation"]) > 0.95
    # the report's numbers are the measured ones, recomputed from the tokens' records
    measured = [
        json.loads(t.achieved)["pitch_semitones"]["measured"]
        for t in mapped
        if "pitch_semitones" in json.loads(t.achieved)
    ]
    assert by["pitch"]["achieved_mean"] == pytest.approx(np.mean(measured), abs=1e-5)
    # a second run reads the mapped clips and their achieved values from the cache
    again = mapped_run(tmp_path)
    assert again.assignment.summary["acoustic_mapping"]["computed"] == 0
    assert again.synthesis.tokens == synthesis.tokens


@needs_audio
@needs_parselmouth
def test_mapped_tokens_are_the_words_tokens_and_the_originals_are_a_control_set(tmp_path):
    from semantic_world.wordforms.embeddings import (
        SoundEmbeddings,
        compute_embeddings,
        token_layout,
        word_embedding_tokens,
        word_means,
    )
    from semantic_world.wordforms.evaluate import evaluate_embeddings
    from semantic_world.wordforms.frontends import compute_frontends

    run = mapped_run(
        tmp_path,
        augmentation={
            "recipes": [{"name": "noisy", "noise": {"kinds": ["white"], "snr_db": [10, 20]}}]
        },
        training={"held_out_word_proportion": 0},
    )
    config, synthesis, tokens = run.config, run.synthesis, run.synthesis.tokens
    assigned = {w.label for w in run.assignment.words}
    control = np.array([t.control for t in tokens])
    augmented = np.array([bool(t.augmentation) for t in tokens])
    mapped = np.array([bool(t.mapping) for t in tokens]) & ~augmented
    clean = np.array([t.clean for t in tokens])
    assert control.sum() == mapped.sum() == 36 and np.array_equal(clean, ~control & ~augmented)
    for token in tokens:
        if token.control:
            assert token.word in assigned and not token.mapping and not token.augmentation
        else:  # every other token of a mapped word is mapped, or is made from a mapped token
            assert bool(token.mapping) == (token.word in assigned)
    # augmentation is applied to the words' own tokens: to mapped tokens, never to a control
    by_label = {t.label: t for t in tokens}
    sources = [by_label[json.loads(t.augmentation)["source"]] for t in tokens if t.augmentation]
    assert len(sources) == 90 and all(source.clean for source in sources)
    assert sum(source.label.endswith(".M") for source in sources) == 36
    assert sum(t.label.endswith(".M.A.1") for t in tokens) == 36

    # word embeddings: the mapped tokens make a mapped word's embedding
    run.frontends = compute_frontends(config, synthesis, tmp_path / "run")
    run.embeddings = compute_embeddings(
        config, run.lexicon.words, synthesis, run.frontends, tmp_path / "run"
    )
    store = run.embeddings["logmel_fixed"]
    token_words, _, train = token_layout(run.lexicon.words, synthesis)
    assert np.array_equal(word_embedding_tokens(config, synthesis, train), train & clean)
    assert np.allclose(
        store.types, word_means(store.tokens, token_words, train & clean, 30), atol=1e-6
    )
    assert store.meta["projection"]["fitted_tokens"] == int((train & clean).sum()) == 60
    labels = [w.label for w in run.lexicon.words]
    for word in run.assignment.words:
        row = labels.index(word.label)
        own = train & mapped & (token_words == row)
        assert own.sum() == 2 and np.allclose(
            store.types[row], store.tokens[own].mean(axis=0), atol=1e-5
        )
    # the control tokens sound different, and are in no word embedding
    row = labels.index(run.assignment.words[0].label)
    held = train & control & (token_words == row)
    assert not np.allclose(store.types[row], store.tokens[held].mean(axis=0), atol=1e-3)
    every = parse_config({**config.resolved(), "word_embeddings": {"tokens": "all"}}, "assign_test")
    assert np.array_equal(word_embedding_tokens(every, synthesis, train), train & ~control)

    # the evaluation: control tokens are in their own set alone, beside the mapped tokens
    run.evaluation = evaluate_embeddings(
        run.embeddings, run.lexicon.words, synthesis, run.streams.eval, config=config
    )
    rows = plain_rows(run.evaluation).filter(pl.col("word_set") == "all")
    counts = dict(zip(rows["tokens"].to_list(), rows["tokens_evaluated"].to_list(), strict=True))
    assert counts == {
        "clean": 90,
        "augmented": 90,
        "all": 180,
        "recipe:noisy": 90,
        "mapped": 36,
        "control": 36,
    }

    # the interface gives a learner the mapped tokens, and holds the control tokens apart
    folder = run.write(tmp_path / "run")
    sounds = SoundEmbeddings.load(folder, "logmel_fixed")
    assert len(sounds.tokens) == len(sounds.token_words) == len(sounds.token_table) == 180
    assert sounds.token_mapped.sum() == 72 and sounds.token_augmented.sum() == 90
    assert np.allclose(sounds.tokens, store.tokens[~control]) and np.allclose(
        sounds.types, store.types
    )
    assert sounds.control["tokens"].shape == (36, 8)
    assert np.allclose(sounds.control["tokens"], store.tokens[control])
    assert sounds.control["token_table"]["label"].to_list() == [
        t.label for t in tokens if t.control
    ]
    assert len(sounds.control["token_words"]) == len(sounds.control["token_held_out"]) == 36
    assert not set(sounds.control["token_table"]["label"]) & set(sounds.token_table["label"])
    # embed gives a mapped word's mapped tokens
    sounds.engines = {}
    speakers = sounds.speakers["label"].to_list()
    word = run.assignment.words[0]
    embedded = sounds.embed([word.arpabet], speakers)
    for j, speaker in enumerate(speakers):
        stored = sounds._stored[(word.label, speaker, 1)]
        assert sounds.token_table["label"][stored] == f"{word.label}.{speaker}.1.M"
        assert np.allclose(embedded[0, j], sounds.tokens[stored], atol=1e-5)
    # the run folder labels both sets
    table = pl.read_csv(folder / "tokens.csv")
    assert (
        table["control"].sum() == 36
        and table.filter(pl.col("mapping").fill_null("") != "").height == 72
    )
    assert table.filter(pl.col("control"))["label"].str.ends_with(".M").sum() == 0
    summary = yaml.safe_load((folder / "summary.yaml").read_text())["synthesis"]
    assert summary["mapped_tokens"] == summary["control_tokens"] == 36


@needs_audio
@needs_parselmouth
@needs_torch
def test_trained_encoders_never_see_a_control_token(tmp_path):
    from semantic_world.wordforms.embeddings import compute_embeddings
    from semantic_world.wordforms.frontends import compute_frontends

    learned = {"encoder": "learned", "kind": "contrastive", "frontend": "logmel", "dims": 8, "hidden": 16, "epochs": 1, "batch_size": 16}  # fmt: skip
    run = mapped_run(
        tmp_path,
        embeddings=[{"name": "clean", **learned}, {"name": "every", **learned, "train_on": "all"}],
        augmentation={"recipes": [{"name": "noisy", "noise": {"kinds": ["white"], "snr_db": 15}}]},
        training={"held_out_word_proportion": 0},
        device="cpu",
    )
    config, synthesis = run.config, run.synthesis
    frontends = compute_frontends(config, synthesis, tmp_path / "run")
    stores = compute_embeddings(config, run.lexicon.words, synthesis, frontends, tmp_path / "run")
    held_out = {s.label for s in synthesis.speakers if s.held_out}
    train = [t for t in synthesis.tokens if t.speaker not in held_out]
    assert sum(t.control for t in train) == 24
    # 2 training speakers by 30 words: the mapped tokens stand in for the 24 control tokens
    assert stores["clean"].meta["training"]["tokens"] == sum(t.clean for t in train) == 60
    assert stores["every"].meta["training"]["tokens"] == sum(not t.control for t in train) == 120


def test_an_unknown_feature_is_an_error_before_any_audio(tmp_path):
    config = assign_config(
        tmp_path, mode="acoustic_mapping",
        acoustic_mapping=[{"feature": "IS.99", "property": "pitch", "amount": 1}],
    )  # fmt: skip
    with pytest.raises(AssignmentError, match="no feature column 'IS.99'"):
        run_assignment(run_forms(config))


@needs_audio
@needs_parselmouth
@needs_espeak
@needs_piper
def test_all_with_acoustic_mapping_on_the_tiny_configuration(tmp_path, capsys):
    data = yaml.safe_load((DATA / "tiny.yaml").read_text())
    data["synthesis"]["engines"]["piper"]["voice_dir"] = str(VOICE_DIR)
    data["synthesis"]["cache_dir"] = str(tmp_path / "cache")
    data["closed_class"] = None
    data["frontends"] = {}
    data["assignment"] = {
        "mode": "acoustic_mapping",
        "meanings": str(taxonomy_file(tmp_path)),
        "acoustic_mapping": [
            {"feature": "ISA.C1", "property": "duration", "amount": 1.2},
            {"feature": "ISA.C2", "property": "pitch", "amount": 2.0},
            {"feature": "ISA.C3", "property": "tilt", "amount": 3.0},
            {"feature": "IS.1", "property": "formants", "amount": 1.1},
        ],
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "run"
    assert main(["all", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "assignment (acoustic_mapping): 12 meanings" in text
    assert "ISA.C1 -> duration 1.2: 24 tokens, achieved mean 1." in text
    summary = yaml.safe_load((out / "assignment" / "summary.yaml").read_text())
    by = {m["property"]: m for m in summary["acoustic_mapping"]["mappings"]}
    assert by["duration"]["achieved_mean"] == pytest.approx(1.2, rel=0.01)
    assert by["tilt"]["achieved_mean"] == pytest.approx(3.0, rel=0.02)
    assert by["formants"]["achieved_mean"] == pytest.approx(1.1, rel=0.02)
    assert by["formants"]["over_10_percent"] == 0 and by["formants"]["feature_correlation"] > 0.95
    # real clips: the measured pitch shift is the amount, and the misses are counted
    assert by["pitch"]["achieved_mean"] == pytest.approx(2.0, rel=0.03)
    assert by["pitch"]["over_10_percent"] <= by["pitch"]["over_5_percent"] <= 2
    assert by["pitch"]["feature_correlation"] > 0.99
    assert "72 mapped tokens are their words' tokens; their 72 unmapped originals" in text
    tokens = pl.read_csv(out / "tokens.csv")
    mapped = tokens.filter(pl.col("label").str.ends_with(".M"))
    assert mapped.height == summary["acoustic_mapping"]["mapped_tokens"] == 72
    assert json.loads(mapped["achieved"][0]) and json.loads(mapped["mapping"][0])["source"]
    assert not mapped["control"].any() and tokens["control"].sum() == 72
    # the mapped tokens and the control tokens are evaluated as two token sets
    table = pl.read_csv(out / "eval" / "embeddings.csv")
    assert set(table["tokens"].unique()) == {"clean", "mapped", "control"}
