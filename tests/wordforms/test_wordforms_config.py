"""Loading and validating the configuration."""

from __future__ import annotations

import pytest
import yaml
from wordforms_support import DATA

from semantic_world.wordforms.config import (
    ConfigError,
    FixedEmbeddingConfig,
    PretrainedEmbeddingConfig,
    load_config,
    parse_config,
)


def test_default_config_matches_the_spec_example(default_config):
    c = default_config
    assert c.name == "default" and c.seed == 1
    assert c.wordforms.source == "pseudowords"
    assert c.wordforms.count == 500
    assert c.wordforms.syllables == {1: 0.3, 2: 0.5, 3: 0.2}
    assert c.wordforms.initial_stress_probability == 0.8
    assert c.wordforms.exclude_real_words is True
    assert c.wordforms.min_english_distance == 1 and c.wordforms.min_lexicon_distance == 1
    assert c.wordforms.english_min_zipf == 3.0
    assert c.synthesis.sample_rate == 16000 and c.synthesis.tokens_per_speaker == 2
    assert c.synthesis.piper.speakers == 40 and c.synthesis.piper.voice == "en_US-libritts_r-medium"
    assert c.synthesis.espeak.variants == ("m1", "m3", "m7", "f2", "f4")
    assert c.synthesis.espeak.speakers == 5
    assert c.synthesis.engines == ("piper", "espeak")
    assert c.frontends.names == ("waveform", "logmel", "cochleagram")
    assert c.frontends.cochleagram.channels == 64
    assert [e.name for e in c.embeddings] == ["cochleagram_fixed", "logmel_fixed", "hubert_base"]
    assert isinstance(c.embeddings[0], FixedEmbeddingConfig) and c.embeddings[0].pca_dims == 256
    # HuBERT base layer 8 is the default, and only the configured layer is stored
    assert isinstance(c.embeddings[2], PretrainedEmbeddingConfig) and c.embeddings[2].layer == 8
    assert c.embeddings[2].store_layers is False
    assert c.augmentation is None
    assert c.assignment.mode == "arbitrary" and c.assignment.meanings is None
    assert c.device == "auto"


def test_empty_config_equals_the_default_config(default_config):
    assert parse_config({}, "empty").resolved() == default_config.resolved()


def test_tiny_config(tiny_config):
    c = tiny_config
    assert c.wordforms.count == 20
    assert c.synthesis.tokens_per_speaker == 1
    assert c.synthesis.piper.speakers == 3 and c.synthesis.espeak.speakers == 3
    assert [e.encoder for e in c.embeddings] == ["fixed", "fixed"]


def test_resolved_config_round_trips(default_config, tmp_path):
    path = tmp_path / "resolved.yaml"
    path.write_text(default_config.to_yaml())
    again = load_config(path)
    assert again.resolved() == default_config.resolved()


def test_an_engine_that_is_off_stays_off_when_the_resolved_config_reloads():
    config = parse_config({"synthesis": {"engines": {"piper": None, "espeak": {}}}}, "x")
    assert config.synthesis.engines == ("espeak",)
    assert config.resolved()["synthesis"]["engines"]["piper"] is None
    again = parse_config(config.resolved(), "x")
    assert again.synthesis.piper is None and again.resolved() == config.resolved()
    assert again.synthesis.espeak.speakers == 5
    piper = parse_config({}, "x").synthesis.piper
    assert piper.voice_dir == "runs/wordforms/voices"


def test_level_and_duration_check_settings(default_config):
    synthesis = default_config.synthesis
    assert (synthesis.level.rms_db, synthesis.level.max_peak) == (-24, 0.9)
    assert (synthesis.duration_check.max_ratio, synthesis.duration_check.max_tries) == (1.8, 5)
    off = parse_config({"synthesis": {"duration_check": None}}, "x")
    assert off.synthesis.duration_check is None
    assert off.resolved()["synthesis"]["duration_check"] is None
    assert parse_config(off.resolved(), "x").synthesis.duration_check is None
    assert synthesis.long_synthesis_ratio == 1.6
    no_flag = parse_config({"synthesis": {"long_synthesis_ratio": None}}, "x")
    assert no_flag.synthesis.long_synthesis_ratio is None
    assert parse_config(no_flag.resolved(), "x").synthesis.long_synthesis_ratio is None
    with pytest.raises(ConfigError) as info:
        parse_config({"synthesis": {"long_synthesis_ratio": 1}}, "x")
    assert info.value.field == "synthesis.long_synthesis_ratio"
    changed = parse_config(
        {"synthesis": {"level": {"rms_db": -20}, "duration_check": {"max_tries": 3}}}, "x"
    )
    assert changed.synthesis.level.rms_db == -20 and changed.synthesis.level.max_peak == 0.9
    assert changed.synthesis.duration_check.max_ratio == 1.8
    for data, field in (
        ({"level": {"rms_db": 3}}, "synthesis.level.rms_db"),
        ({"level": {"max_peak": 1.5}}, "synthesis.level.max_peak"),
        ({"level": {"peak": 0.9}}, "synthesis.level.peak"),
        ({"duration_check": {"max_ratio": 1}}, "synthesis.duration_check.max_ratio"),
        ({"duration_check": {"max_tries": 0}}, "synthesis.duration_check.max_tries"),
    ):
        with pytest.raises(ConfigError) as info:
            parse_config({"synthesis": data}, "x")
        assert info.value.field == field


def test_frontend_settings_are_checked_against_the_sample_rate():
    config = parse_config({}, "x")
    assert config.frontends.store_waveform is False
    assert config.resolved()["frontends"]["waveform"] == {"store": False}
    stored = parse_config({"frontends": {"waveform": {"store": True}}}, "x")
    assert stored.frontends.store_waveform is True
    assert parse_config(stored.resolved(), "x").frontends.store_waveform is True
    for data, field, message in (
        ({"cochleagram": {"frame_rate": 300}}, "frontends.cochleagram.frame_rate", "divide"),
        ({"cochleagram": {"high_hz": 9000}}, "frontends.cochleagram.high_hz", "half the sample"),
        ({"cochleagram": {"channels": 2}}, "frontends.cochleagram.channels", "at least 3"),
        ({"logmel": {"window_ms": 5, "hop_ms": 10}}, "frontends.logmel.window_ms", "hop_ms"),
        ({"waveform": {"keep": True}}, "frontends.waveform.keep", "unknown key"),
    ):
        with pytest.raises(ConfigError) as info:
            parse_config({"frontends": data}, "x")
        assert info.value.field == field and message in str(info.value)
    # a lower sample rate lowers the highest allowed frequency
    with pytest.raises(ConfigError) as info:
        parse_config({"synthesis": {"sample_rate": 8000}}, "x")
    assert info.value.field == "frontends.cochleagram.high_hz"


def test_pretrained_embedding_settings():
    base = {"name": "h", "encoder": "pretrained", "model": "facebook/hubert-base-ls960"}
    config = parse_config({"embeddings": [base]}, "x")
    assert config.embeddings[0].layer == 8 and config.embeddings[0].store_layers is False
    kept = parse_config({"embeddings": [{**base, "layer": 3, "store_layers": True}]}, "x")
    assert kept.embeddings[0].layer == 3 and kept.embeddings[0].store_layers is True
    assert kept.resolved()["embeddings"][0]["store_layers"] is True
    assert parse_config(kept.resolved(), "x").resolved() == kept.resolved()
    with pytest.raises(ConfigError) as info:
        parse_config({"embeddings": [{**base, "store_layers": "yes"}]}, "x")
    assert info.value.field == "embeddings[0].store_layers"


def test_seed_override():
    config = load_config(DATA / "default.yaml", seed=7)
    assert config.seed == 7
    assert config.with_seed(9).seed == 9


def test_syllable_weights_are_normalized():
    config = parse_config({"wordforms": {"syllables": {2: 1, 1: 3}}}, "x")
    assert config.wordforms.syllables == {1: 0.75, 2: 0.25}


def test_english_min_zipf():
    assert parse_config({}, "x").wordforms.english_min_zipf == 3.0
    null = parse_config({"wordforms": {"english_min_zipf": None}}, "x")
    assert null.wordforms.english_min_zipf is None
    assert null.resolved()["wordforms"]["english_min_zipf"] is None
    assert (
        parse_config({"wordforms": {"english_min_zipf": 4}}, "x").wordforms.english_min_zipf == 4.0
    )
    assert parse_config(null.resolved(), "x").resolved() == null.resolved()
    for bad in (-1, 9, "common"):
        with pytest.raises(ConfigError) as info:
            parse_config({"wordforms": {"english_min_zipf": bad}}, "x")
        assert info.value.field == "wordforms.english_min_zipf"


def test_english_count():
    assert parse_config({"wordforms": {"source": "english"}}, "x").wordforms.english_count == 500
    mixed = parse_config(
        {"wordforms": {"source": "mixed", "count": 11, "mixed_proportion_english": 0.5}}, "x"
    )
    assert mixed.wordforms.english_count == 6


@pytest.mark.parametrize(
    ("data", "field"),
    [
        ({"wordform": {}}, "wordform"),
        ({"wordforms": {"counts": 3}}, "wordforms.counts"),
        ({"synthesis": {"engines": {"festival": {}}}}, "synthesis.engines.festival"),
        ({"synthesis": {"engines": {"piper": {"speaker": 3}}}}, "synthesis.engines.piper.speaker"),
        ({"synthesis": {"trim": {"threshold": -40}}}, "synthesis.trim.threshold"),
        ({"frontends": {"logmel": {"mels": 80}}}, "frontends.logmel.mels"),
        (
            {"embeddings": [{"name": "a", "encoder": "fixed", "frontend": "logmel", "bins": 2}]},
            "embeddings[0].bins",
        ),
        ({"assignment": {"modes": "arbitrary"}}, "assignment.modes"),
    ],
)
def test_unknown_keys_name_the_file_and_the_field(data, field, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert info.value.source == str(path)
    assert info.value.field == field
    assert "unknown key" in str(info.value)
    assert str(path) in str(info.value) and field in str(info.value)


@pytest.mark.parametrize(
    ("data", "field", "message"),
    [
        ({"wordforms": {"count": 0}}, "wordforms.count", "at least 1"),
        ({"wordforms": {"source": "latin"}}, "wordforms.source", "expected one of"),
        ({"wordforms": {"syllables": {0: 1}}}, "wordforms.syllables.0", "positive"),
        ({"wordforms": {"syllables": {1: 0, 2: 0}}}, "wordforms.syllables", "positive"),
        (
            {"wordforms": {"initial_stress_probability": 1.5}},
            "wordforms.initial_stress_probability",
            "at most 1",
        ),
        (
            {"wordforms": {"exclude_real_words": "yes"}},
            "wordforms.exclude_real_words",
            "true or false",
        ),
        ({"seed": -1}, "seed", "at least 0"),
        (
            {"synthesis": {"engines": {"piper": None, "espeak": None}}},
            "synthesis.engines",
            "at least one engine",
        ),
        (
            {"synthesis": {"engines": {"espeak": {"pitch": [65, 35]}}}},
            "synthesis.engines.espeak.pitch",
            "exceeds",
        ),
        (
            {"synthesis": {"engines": {"espeak": {"variants": ["m1", "m1"]}}}},
            "synthesis.engines.espeak.variants",
            "distinct",
        ),
        (
            {"frontends": {"cochleagram": {"low_hz": 9000}}},
            "frontends.cochleagram.high_hz",
            "more than",
        ),
        ({"embeddings": {"name": "a"}}, "embeddings", "expected a list"),
        (
            {"embeddings": [{"encoder": "fixed", "frontend": "logmel"}]},
            "embeddings[0].name",
            "required",
        ),
        ({"embeddings": [{"name": "a", "encoder": "learned"}]}, "embeddings[0].encoder", "stage 6"),
        (
            {
                "embeddings": [
                    {"name": "a", "encoder": "fixed", "frontend": "logmel"},
                    {"name": "a", "encoder": "fixed", "frontend": "logmel"},
                ]
            },
            "embeddings[1].name",
            "twice",
        ),
        (
            {
                "frontends": {"logmel": None},
                "embeddings": [{"name": "a", "encoder": "fixed", "frontend": "logmel"}],
            },
            "embeddings[0].frontend",
            "not configured",
        ),
        (
            {"embeddings": [{"name": "a", "encoder": "pretrained"}]},
            "embeddings[0].model",
            "required",
        ),
        ({"augmentation": {"noise": {}}}, "augmentation.recipes", "required"),
        ({"assignment": {"mode": "branch_markers"}}, "assignment.mode", "stage 7"),
        ({"device": "tpu"}, "device", "expected one of"),
        ({"wordforms": 3}, "wordforms", "expected a mapping"),
    ],
)
def test_invalid_values(data, field, message):
    with pytest.raises(ConfigError) as info:
        parse_config(data, "cfg.yaml")
    assert info.value.field == field
    assert message in str(info.value)


def test_missing_file_and_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError) as info:
        load_config(tmp_path / "missing.yaml")
    assert info.value.field == "<file>"
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed")
    with pytest.raises(ConfigError) as info:
        load_config(bad)
    assert "invalid YAML" in str(info.value)


def test_example_configs_load():
    for path in sorted(DATA.glob("*.yaml")):
        if path.name.startswith("arpabet") or path.name == "spelling.yaml":
            continue
        load_config(path)
