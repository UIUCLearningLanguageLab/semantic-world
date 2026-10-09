"""Stage 1 and stage a5a: configuration loading, validation, and the resolved configuration.

The example configurations load. Deliberately broken configurations each fail with an error that
names the file and the field. Every key that the world-and-language refactor renamed fails with
an error that names its new key. The resolved configuration round-trips.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from corpus_support import PLAIN_WORLD, TINY_WORLD, corpus_config, write_world

from semantic_world.corpus import ConfigError, config_from_mapping, load_config
from semantic_world.taxonomy.config import Range
from semantic_world.world.config import load_config as load_world_config
from semantic_world.world.generate import define

DATA = Path("data/corpus")
EXAMPLES = ("default.yaml", "tiny.yaml")


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_configurations_load(name: str) -> None:
    config = load_config(DATA / name)
    assert config.source == str(DATA / name)
    assert config.name == Path(name).stem
    # the examples name world configuration files, so they run with nothing saved on disk
    assert config.world.kind == "config"
    assert Path(config.world.path).is_file()


def test_default_values() -> None:
    config = load_config(DATA / "default.yaml")
    assert config.seed == 1
    assert (config.world.path, config.world.seed) == ("data/world/default.yaml", 1)
    assert config.world.depth == 3 and config.world.event_depth == 2
    lexicon = config.lexicon
    assert list(lexicon.named_proportion) == [
        "category",
        "property",
        "part",
        "state",
        "event_unary",
        "event",
        "event_category",
        "patient_projection",
        "scalar",
    ]
    assert lexicon.named_proportion["patient_projection"] == 0.25
    assert all(v == 1.0 for k, v in lexicon.named_proportion.items() if k != "patient_projection")
    assert (lexicon.synonym_rate, lexicon.homonym_rate, lexicon.homonym_same_pos) == (0, 0, 0.5)
    documents = config.documents
    assert documents.count == 10000
    assert documents.mix == {
        "encyclopedic_category": 0.3,
        "encyclopedic_feature": 0.2,
        "entity": 0.2,
        "situational": 0.3,
    }
    assert documents.sentences["encyclopedic_category"] == Range(5, 15)
    assert documents.sentences["situational"] == Range(5, 20)
    assert documents.topic_level_weights == (1.0, 1.5, 2.0)
    assert (documents.shuffle, documents.instance_description_rate) == (0.3, 0.2)
    assert documents.sibling_contrast_rate == 0.2
    assert documents.relation_fact_share is None
    assert documents.content_kind_weights == "equal"
    assert documents.progressive_rate == 0.3 and documents.one_aspect_per_event is True
    assert config.propositions.negation_rate == {"class": 0.1, "instance": 0.1}
    assert config.propositions.rule_statement_rate == 0.3
    assert config.propositions.rule_max_literals is None
    assert config.propositions.restriction_rate == 0.1
    assert config.propositions.event_tense == "past"
    quantifiers = config.quantifiers
    assert quantifiers.universal_words == "nec"
    assert quantifiers.most_usage_min == 0.7
    assert quantifiers.some_exclude_all is True
    assert quantifiers.bare_plural_expresses == ("most",)
    assert quantifiers.generic_rate == 0.5
    assert quantifiers.weights == {
        "nec_all": 1.0,
        "all": 1.0,
        "most": 1.0,
        "some": 1.0,
        "no": 1.0,
        "nec_no": 1.0,
    }
    assert quantifiers.weighted is False
    scene = config.scene
    assert (scene.size, scene.steps) == (Range(2, 6), Range(3, 8))
    assert (scene.events_per_step, scene.transitive_share) == (1.5, 0.5)
    assert scene.event_type_weights is None
    assert scene.participant_weights == {"thematic": 1.0, "taxonomic": 0.5, "constant": 0.1}
    assert (scene.policy, scene.initial) == ("uniform_event", "keep")
    assert config.entity_scenes == Range(2, 5)
    mention = config.mention
    assert mention.level_weights == (1.0, 2.5, 4.0)
    assert mention.event_level_weights == (1.0, 4.0)  # the default event-type tree has two levels
    assert (mention.pronoun_rate, mention.modifier_rate) == (0.5, 0.3)
    assert (mention.max_adjectives, mention.max_with_phrases) == (3, 2)
    assert mention.max_content_words == 20
    clauses = mention.relative_clauses
    assert (clauses.rate, clauses.max_depth, clauses.object_share) == (0.1, 1, 0.3)
    grammar = config.grammar
    assert grammar.adjective_order_fixed is True
    assert grammar.can_rate == {"class": 0.5, "instance": 1.0}
    assert grammar.word_order.resolved() == {
        "clause": "SVO",
        "determiner": "before",
        "adjective": "before",
        "with_phrase": "after",
        "relative_clause": "after",
        "adposition": "preposition",
        "auxiliary": "before",
        "negation": "after_auxiliary",
    }
    morphology = grammar.morphology
    assert not (morphology.number.enabled or morphology.tense.enabled or morphology.aspect.enabled)
    assert (morphology.number.realization, morphology.number.verb_marks) == ("affix", "plural")
    assert morphology.number.agreement is True and morphology.agreement is False
    assert (morphology.tense.realization, morphology.aspect.realization) == ("affix", "word")
    assert config.scalar_z == 1.0
    assert config.propositional_referents == "local"
    assert config.test_sets.size == 500
    assert config.test_sets.changes == ("predicate", "subject", "quantifier", "role")


def test_default_file_lists_every_default() -> None:
    # default.yaml shows every parameter, and an empty configuration gives the same values.
    listed = load_config(DATA / "default.yaml")
    bare = config_from_mapping({"world": {"config": "data/world/default.yaml"}})
    assert listed.resolved() == bare.resolved()
    written = yaml.safe_load((DATA / "default.yaml").read_text(encoding="utf-8"))
    assert _key_paths(written) == _key_paths(listed.resolved())


def _key_paths(data: Any, prefix: str = "") -> set[str]:
    """Every dotted key of a nested mapping. A schedule is written differently in a file and
    in the resolved form, so it counts as one value."""
    if not isinstance(data, dict) or "schedule" in data:
        return {prefix}
    paths: set[str] = set()
    for key, value in data.items():
        paths |= _key_paths(value, f"{prefix}.{key}" if prefix else str(key))
    return paths


def test_tiny_configuration() -> None:
    config = load_config(DATA / "tiny.yaml")
    assert config.documents.count == 20
    assert config.world.path == TINY_WORLD
    # the level schedules are resolved over the tiny taxonomy's two levels
    assert config.world.depth == 2
    assert config.mention.level_weights == (1.0, 4.0)
    assert config.documents.topic_level_weights == (1.0, 2.0)


def test_rule_statement_cap_and_event_level_weights(tmp_path: Path) -> None:
    config = corpus_config(
        propositions={"rule_statements": {"max_literals": 3}},
        mention={"event_level_weights": {"schedule": "list", "values": [0, 1]}},
    )
    assert config.propositions.rule_max_literals == 3
    assert config.mention.event_level_weights == (0.0, 1.0)
    assert config.resolved()["propositions"]["rule_statements"] == {"max_literals": 3}
    assert config_from_mapping(config.resolved()) == config
    # a world without two-place event types has no event-type levels to weigh
    plain = corpus_config(write_world(tmp_path, "plain", PLAIN_WORLD))
    assert plain.world.event_depth is None and plain.mention.event_level_weights == ()
    assert config_from_mapping(plain.resolved()) == plain


def test_seed_override_is_the_corpus_seed_only() -> None:
    config = load_config(DATA / "tiny.yaml", seed=9)
    assert config.seed == 9
    assert config.world.seed == 1
    with pytest.raises(ConfigError) as info:
        load_config(DATA / "tiny.yaml", seed=-1)
    assert info.value.field == "seed"


@pytest.mark.parametrize("name", EXAMPLES)
def test_resolved_configuration_round_trips(name: str) -> None:
    config = load_config(DATA / name)
    assert config_from_mapping(config.resolved(), source=config.source) == config
    reloaded = config_from_mapping(yaml.safe_load(config.to_yaml()), source=config.source)
    assert reloaded == config
    assert reloaded.to_yaml() == config.to_yaml()


def test_resolved_configuration_keeps_changed_values() -> None:
    config = corpus_config(
        lexicon={"synonym_rate": 0.2, "named_proportion": {"property": 0.5}},
        documents={"mix": {"situational": 1}, "sentences": {"entity": 7}},
        scene={
            "event_type_weights": {"EVENTTYPE1.1": 2, "EVENTTYPE2.1.1": 0.5},
            "policy": "uniform_event_type",
            "initial": "redraw",
        },
        quantifiers={"universal_words": "either", "bare_plural": {"expresses": ["nec_all", "all"]}},
        grammar={"morphology": {"number": {"enabled": True, "realization": "word"}}},
        test_sets={"changes": ["role"]},
    )
    resolved = config.resolved()
    assert resolved["lexicon"]["named_proportion"]["property"] == 0.5
    assert resolved["documents"]["mix"] == {
        "encyclopedic_category": 0.0,
        "encyclopedic_feature": 0.0,
        "entity": 0.0,
        "situational": 1.0,
    }
    assert resolved["documents"]["sentences"]["entity"] == [7, 7]
    assert resolved["scene"]["event_type_weights"] == {"EVENTTYPE1.1": 2.0, "EVENTTYPE2.1.1": 0.5}
    assert resolved["scene"]["policy"] == "uniform_event_type"
    assert resolved["scene"]["initial"] == "redraw"
    assert resolved["quantifiers"]["universal_words"] == "either"
    assert resolved["quantifiers"]["bare_plural"] == {"expresses": ["nec_all", "all"]}
    assert resolved["grammar"]["morphology"]["number"]["enabled"] is True
    assert config_from_mapping(yaml.safe_load(config.to_yaml())) == config


def test_provenance_is_accepted_when_a_run_folder_configuration_is_read_back() -> None:
    data = corpus_config().resolved()
    data["provenance"] = {"git_commit": "abc", "stream_seeds": {}}
    assert config_from_mapping(data) == corpus_config()


# ---------------------------------------------------------------------------------------------
# The world setting
# ---------------------------------------------------------------------------------------------


def test_world_configuration_file_with_a_seed() -> None:
    config = corpus_config()
    assert (config.world.kind, config.world.path) == ("config", TINY_WORLD)
    assert config.world.seed == 1  # seed left out: the seed in the world file
    assert config.world.config == load_world_config(TINY_WORLD)
    assert config.world.resolved() == {"config": TINY_WORLD, "seed": 1}
    for seed in (None, 5):
        data = {"world": {"config": TINY_WORLD, "seed": seed}}
        assert config_from_mapping(data).world.seed == (1 if seed is None else seed)


def test_world_run_folder(tmp_path: Path) -> None:
    world = load_world_config(TINY_WORLD, seed=3)
    folder = define(world).write(tmp_path / "run")
    config = config_from_mapping({"world": {"run": str(folder)}})
    assert (config.world.kind, config.world.path) == ("run", str(folder))
    assert config.world.seed == 3
    assert config.world.config.resolved() == world.resolved()
    assert config.world.resolved() == {"run": str(folder)}
    assert config_from_mapping(config.resolved()) == config


WORLD_ERRORS: list[tuple[Any, str, str]] = [
    (None, "world", "must not be null"),
    ("runs/world/default_seed1", "world", "expected a mapping"),
    ({}, "world", "exactly one of config and run"),
    ({"config": TINY_WORLD, "run": "runs/x"}, "world", "exactly one of config and run"),
    ({"seed": 1}, "world", "exactly one of config and run"),
    ({"config": "data/world/missing.yaml"}, "world.config", "cannot read"),
    ({"run": "data/world"}, "world.run", "not a world run folder"),
    ({"run": "data/world", "seed": 1}, "world.seed", "goes with config only"),
    ({"config": TINY_WORLD, "seed": -1}, "world.seed", "at least 0"),
    ({"config": TINY_WORLD, "seed": "one"}, "world.seed", "expected an integer"),
    ({"config": TINY_WORLD, "folder": "x"}, "world.folder", "unknown key"),
]


@pytest.mark.parametrize(("value", "field", "message"), WORLD_ERRORS)
def test_world_setting_errors(value: Any, field: str, message: str) -> None:
    with pytest.raises(ConfigError) as info:
        config_from_mapping({"world": value}, source="broken.yaml")
    assert info.value.source == "broken.yaml"
    assert info.value.field == field
    assert message in info.value.message


def test_world_setting_is_required() -> None:
    with pytest.raises(ConfigError) as info:
        config_from_mapping({})
    assert info.value.field == "world" and "is required" in info.value.message


def test_errors_in_the_world_file_name_that_file(tmp_path: Path) -> None:
    broken = tmp_path / "world.yaml"
    broken.write_text("fluents: {count: -1}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        config_from_mapping({"world": {"config": str(broken)}})
    assert info.value.source == str(broken) and info.value.field == "fluents.count"


# ---------------------------------------------------------------------------------------------
# The renamed keys
# ---------------------------------------------------------------------------------------------

RENAMED: list[tuple[dict[str, Any], str, str]] = [
    ({"taxonomy": {"config": "data/taxonomy/relations.yaml"}}, "taxonomy", "world"),
    ({"lexicon": {"named_proportion": {"is": 1.0}}}, "lexicon.named_proportion.is", "property"),
    ({"lexicon": {"named_proportion": {"has": 1.0}}}, "lexicon.named_proportion.has", "part"),
    (
        {"lexicon": {"named_proportion": {"can": 1.0}}},
        "lexicon.named_proportion.can",
        "event_unary",
    ),
    ({"lexicon": {"named_proportion": {"verb": 1.0}}}, "lexicon.named_proportion.verb", "event"),
    (
        {"lexicon": {"named_proportion": {"verb_category": 1.0}}},
        "lexicon.named_proportion.verb_category",
        "event_category",
    ),
    ({"quantifiers": {"all_grounding": "fixed"}}, "quantifiers.all_grounding", "universal_words"),
    (
        {"quantifiers": {"most": {"min_proportion": 0.7}}},
        "quantifiers.most.min_proportion",
        "quantifiers.most.usage_min",
    ),
    (
        {"quantifiers": {"generic": {"means": "most"}}},
        "quantifiers.generic",
        "quantifiers.bare_plural.expresses",
    ),
    (
        {"propositions": {"events": {"progressive_rate": 0.3}}},
        "propositions.events.progressive_rate",
        "documents.progressive_rate",
    ),
    ({"scene": {"verb_weights": "uniform"}}, "scene.verb_weights", "scene.event_type_weights"),
    (
        {"mention": {"verb_level_weights": 1}},
        "mention.verb_level_weights",
        "mention.event_level_weights",
    ),
    (
        {"grammar": {"morphology": {"aspect": {"progressive_rate": 0.3}}}},
        "grammar.morphology.aspect.progressive_rate",
        "documents.progressive_rate",
    ),
]


@pytest.mark.parametrize(
    ("change", "field", "new"), RENAMED, ids=[field for _, field, _ in RENAMED]
)
def test_every_old_key_names_its_new_key(change: dict[str, Any], field: str, new: str) -> None:
    data = {"world": {"config": TINY_WORLD}, **copy.deepcopy(change)}
    if field == "taxonomy":
        data.pop("world")
    with pytest.raises(ConfigError) as info:
        config_from_mapping(data, source="old.yaml")
    assert info.value.source == "old.yaml"
    assert info.value.field == field
    assert new in info.value.message


# ---------------------------------------------------------------------------------------------
# Broken configurations
# ---------------------------------------------------------------------------------------------

BROKEN: list[tuple[dict[str, Any], str, str]] = [
    ({"nmae": "x"}, "nmae", "unknown key"),
    ({"seed": -1}, "seed", "at least 0"),
    ({"seed": 2**64}, "seed", "at most"),
    ({"name": ""}, "name", "non-empty string"),
    ({"provenance": 3}, "provenance", "expected a mapping"),
    ({"lexicon": {"synonym_rate": 1.5}}, "lexicon.synonym_rate", "at most 1"),
    ({"lexicon": {"homonym_rate": -0.1}}, "lexicon.homonym_rate", "at least 0"),
    ({"lexicon": {"homonym_same_pos": "half"}}, "lexicon.homonym_same_pos", "expected a number"),
    (
        {"lexicon": {"named_proportion": {"noun": 1.0}}},
        "lexicon.named_proportion.noun",
        "unknown key",
    ),
    (
        {"lexicon": {"named_proportion": {"property": 2}}},
        "lexicon.named_proportion.property",
        "at most 1",
    ),
    ({"lexicon": {"synonyms": 0.1}}, "lexicon.synonyms", "unknown key"),
    ({"documents": {"count": -1}}, "documents.count", "at least 0"),
    ({"documents": {"count": 2.5}}, "documents.count", "expected an integer"),
    ({"documents": {"mix": {"essay": 1}}}, "documents.mix.essay", "unknown key"),
    ({"documents": {"mix": {"entity": 0}}}, "documents.mix", "at least one weight"),
    ({"documents": {"mix": {"entity": -1}}}, "documents.mix.entity", "at least 0"),
    (
        {"documents": {"sentences": {"entity": [9, 3]}}},
        "documents.sentences.entity",
        "exceeds the maximum",
    ),
    ({"documents": {"sentences": {"entity": 0}}}, "documents.sentences.entity", "at least 1"),
    ({"documents": {"sentences": {"essay": 3}}}, "documents.sentences.essay", "unknown key"),
    (
        {"documents": {"topic_level_weights": {"schedule": "list", "values": [1, 2, 3]}}},
        "documents.topic_level_weights.values",
        "exactly 2 values",
    ),
    (
        {"documents": {"topic_level_weights": {"schedule": "linear", "start": 1, "end": -1}}},
        "documents.topic_level_weights",
        "must not be negative",
    ),
    ({"documents": {"topic_level_weights": 0}}, "documents.topic_level_weights", "positive"),
    ({"documents": {"shuffle": 2}}, "documents.shuffle", "at most 1"),
    (
        {"documents": {"content_kind_weights": "random"}},
        "documents.content_kind_weights",
        "proportional, equal",
    ),
    ({"documents": {"progressive_rate": 2}}, "documents.progressive_rate", "at most 1"),
    ({"documents": {"one_aspect_per_event": "yes"}}, "documents.one_aspect_per_event", "true or"),
    (
        {"propositions": {"negation_rate": {"event": 0.1}}},
        "propositions.negation_rate.event",
        "unknown key",
    ),
    (
        {"propositions": {"negation_rate": {"class": 1.1}}},
        "propositions.negation_rate.class",
        "at most 1",
    ),
    ({"propositions": {"rule_statement_rate": None}}, "propositions.rule_statement_rate", "null"),
    (
        {"propositions": {"rule_statements": {"max_literals": 0}}},
        "propositions.rule_statements.max_literals",
        "at least 1",
    ),
    (
        {"propositions": {"rule_statements": {"max_literals": 2.5}}},
        "propositions.rule_statements.max_literals",
        "expected an integer",
    ),
    (
        {"propositions": {"rule_statements": {"cap": 3}}},
        "propositions.rule_statements.cap",
        "unknown key",
    ),
    (
        {"mention": {"event_level_weights": {"schedule": "list", "values": [1, 2, 3]}}},
        "mention.event_level_weights.values",
        "exactly 2 values",
    ),
    (
        {"mention": {"event_level_weights": -1}},
        "mention.event_level_weights",
        "must not be negative",
    ),
    (
        {"quantifiers": {"universal_words": "law"}},
        "quantifiers.universal_words",
        "nec, extensional",
    ),
    ({"quantifiers": {"most": {"usage_min": 0}}}, "quantifiers.most.usage_min", "more"),
    ({"quantifiers": {"most": {"usage_min": 1.2}}}, "quantifiers.most.usage_min", "1"),
    ({"quantifiers": {"some": {"exclude_all": "yes"}}}, "quantifiers.some.exclude_all", "true or"),
    (
        {"quantifiers": {"bare_plural": {"expresses": ["few"]}}},
        "quantifiers.bare_plural.expresses",
        "nec_all, all, most, some",
    ),
    (
        {"quantifiers": {"bare_plural": {"expresses": []}}},
        "quantifiers.bare_plural.expresses",
        "non-empty",
    ),
    (
        {"quantifiers": {"bare_plural": {"expresses": "most"}}},
        "quantifiers.bare_plural.expresses",
        "list",
    ),
    ({"quantifiers": {"generic_rate": -1}}, "quantifiers.generic_rate", "at least 0"),
    ({"quantifiers": {"weights": {"some": -1}}}, "quantifiers.weights.some", "at least 0"),
    ({"quantifiers": {"weights": {"few": 1}}}, "quantifiers.weights.few", "unknown key"),
    (
        {
            "quantifiers": {
                "weights": {"nec_all": 0, "all": 0, "most": 0, "some": 0, "none": 0, "nec_none": 0}
            }
        },
        "quantifiers.weights",
        "at least one weight must be positive",
    ),  # fmt: skip
    # YAML reads a bare no as the boolean false, so the key of the quantifier "no" is none
    ({"quantifiers": {"weights": {False: 2}}}, "quantifiers.weights.no", "is written none"),
    ({"quantifiers": {"weights": {"no": 2}}}, "quantifiers.weights.no", "is written none"),
    ({"quantifiers": {"weights": {"nec_no": 2}}}, "quantifiers.weights.nec_no", "nec_none"),
    ({"scene": {"size": [6, 2]}}, "scene.size", "exceeds the maximum"),
    ({"scene": {"steps": 0}}, "scene.steps", "at least 1"),
    ({"scene": {"events_per_step": -1}}, "scene.events_per_step", "at least 0"),
    ({"scene": {"transitive_share": 3}}, "scene.transitive_share", "at most 1"),
    ({"scene": {"event_type_weights": "equal"}}, "scene.event_type_weights", "expected uniform"),
    (
        {"scene": {"event_type_weights": {"EVENTTYPE2.1.1": -2}}},
        "scene.event_type_weights.EVENTTYPE2.1.1",
        "at least 0",
    ),
    (
        {"scene": {"participant_weights": {"random": 1}}},
        "scene.participant_weights.random",
        "unknown key",
    ),
    (
        {"scene": {"participant_weights": {"thematic": 0}}},
        "scene.participant_weights",
        "at least one weight",
    ),
    ({"scene": {"policy": "random_walk"}}, "scene.policy", "uniform_event"),
    ({"scene": {"initial": "fresh"}}, "scene.initial", "keep, redraw"),
    ({"entity": {"scenes": 0}}, "entity.scenes", "at least 1"),
    ({"mention": {"pronoun_rate": 7}}, "mention.pronoun_rate", "at most 1"),
    ({"mention": {"max_adjectives": -1}}, "mention.max_adjectives", "at least 0"),
    ({"mention": {"max_content_words": 2}}, "mention.max_content_words", "at least 3"),
    ({"mention": {"max_sentence_tokens": 30}}, "mention.max_sentence_tokens", "unknown key"),
    (
        {"mention": {"level_weights": {"schedule": "list", "values": [1]}}},
        "mention.level_weights.values",
        "exactly 2 values",
    ),
    (
        {"mention": {"relative_clauses": {"max_depth": -1}}},
        "mention.relative_clauses.max_depth",
        "at least 0",
    ),
    (
        {"mention": {"relative_clauses": {"object_share": 2}}},
        "mention.relative_clauses.object_share",
        "at most 1",
    ),
    ({"grammar": {"adjective_order": {"fixed": 1}}}, "grammar.adjective_order.fixed", "true or"),
    ({"grammar": {"can_rate": {"class": 1.5}}}, "grammar.can_rate.class", "at most 1"),
    ({"grammar": {"can_rate": {"instance": -1}}}, "grammar.can_rate.instance", "at least 0"),
    ({"grammar": {"can_rate": {"event": 1}}}, "grammar.can_rate.event", "unknown key"),
    ({"grammar": {"class_can_rate": 0.5}}, "grammar.class_can_rate", "grammar.can_rate.class"),
    ({"propositions": {"restriction_rate": 2}}, "propositions.restriction_rate", "at most 1"),
    ({"documents": {"sibling_contrast_rate": 2}}, "documents.sibling_contrast_rate", "at most 1"),
    ({"documents": {"relation_fact_share": 2}}, "documents.relation_fact_share", "at most 1"),
    ({"documents": {"relation_fact_share": "x"}}, "documents.relation_fact_share", "a number"),
    ({"grammar": {"word_order": {"clause": "SVV"}}}, "grammar.word_order.clause", "SVO, SOV"),
    ({"grammar": {"word_order": {"adjective": "left"}}}, "grammar.word_order.adjective", "before"),
    (
        {"grammar": {"word_order": {"adposition": "before"}}},
        "grammar.word_order.adposition",
        "preposition, postposition",
    ),
    (
        {"grammar": {"word_order": {"negation": "after"}}},
        "grammar.word_order.negation",
        "after_auxiliary, before_auxiliary",
    ),
    ({"grammar": {"word_order": {"object": "after"}}}, "grammar.word_order.object", "unknown key"),
    (
        {"grammar": {"morphology": {"number": {"realization": "suffix"}}}},
        "grammar.morphology.number.realization",
        "affix, word",
    ),
    (
        {"grammar": {"morphology": {"number": {"verb_marks": "both"}}}},
        "grammar.morphology.number.verb_marks",
        "plural, singular",
    ),
    (
        {"propositions": {"events": {"tense": "future"}}},
        "propositions.events.tense",
        "past, present",
    ),
    (
        {"grammar": {"morphology": {"tense": {"event_tense": "past"}}}},
        "grammar.morphology.tense.event_tense",
        "is now propositions.events.tense",
    ),
    (
        {"grammar": {"morphology": {"tense": {"enabled": "yes"}}}},
        "grammar.morphology.tense.enabled",
        "true or false",
    ),
    (
        {"grammar": {"morphology": {"case": {"enabled": True}}}},
        "grammar.morphology.case",
        "unknown key",
    ),
    ({"scalar_adjectives": {"z": 0}}, "scalar_adjectives.z", "above 0"),
    ({"scalar_adjectives": {"z": "one"}}, "scalar_adjectives.z", "above 0"),
    (
        {"renderings": {"propositional": {"referents": "global"}}},
        "renderings.propositional.referents",
        "local, instance",
    ),
    ({"renderings": {"formal": {}}}, "renderings.formal", "unknown key"),
    ({"test_sets": {"size": -5}}, "test_sets.size", "at least 0"),
    ({"test_sets": {"changes": "role"}}, "test_sets.changes", "expected a list"),
    ({"test_sets": {"changes": ["tense"]}}, "test_sets.changes", "predicate, subject"),
    ({"test_sets": {"changes": ["role", "role"]}}, "test_sets.changes", "distinct"),
]


@pytest.mark.parametrize(
    ("change", "field", "message"),
    BROKEN,
    ids=[f"{field}:{i}" for i, (_, field, _) in enumerate(BROKEN)],
)
def test_broken_configurations_name_the_file_and_the_field(
    change: dict[str, Any], field: str, message: str
) -> None:
    data = {"world": {"config": TINY_WORLD}, **copy.deepcopy(change)}
    with pytest.raises(ConfigError) as info:
        config_from_mapping(data, source="broken.yaml")
    assert info.value.source == "broken.yaml"
    assert info.value.field == field
    assert message in info.value.message
    assert str(info.value).startswith(f"broken.yaml: {field}: ")


@pytest.mark.parametrize(
    ("key", "new"),
    [
        ("max_adjectives", "mention.max_adjectives"),
        ("max_with_phrases", "mention.max_with_phrases"),
        ("max_sentence_tokens", "mention.max_content_words"),
        ("relative_clauses", "mention.relative_clauses"),
    ],
)
def test_settings_that_moved_to_the_planner_say_where_they_went(key: str, new: str) -> None:
    # The planner decides what a sentence says, so these are not grammar settings.
    with pytest.raises(ConfigError) as info:
        corpus_config(grammar={key: 1})
    assert info.value.field == f"grammar.{key}"
    assert new in info.value.message


def test_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_config(tmp_path / "missing.yaml")
    assert info.value.field == "<file>" and "cannot read" in info.value.message
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(bad)
    assert info.value.field == "<file>" and "invalid YAML" in info.value.message
    bad.write_text("- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(bad)
    assert info.value.field == "<root>"


# ---------------------------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------------------------


def test_the_old_on_key_says_where_it_went(tmp_path: Path) -> None:
    # The switch was `on` in stage 1. YAML reads a bare `on` as the boolean true, so the error
    # covers both readings of the old key.
    path = tmp_path / "corpus.yaml"
    for written in ("on: true", '"on": true'):
        path.write_text(
            f"world: {{config: {TINY_WORLD}}}\n"
            f"grammar: {{morphology: {{number: {{{written}}}}}}}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert info.value.field == "grammar.morphology.number.on"
        assert "is now enabled" in info.value.message
    path.write_text(
        f"world: {{config: {TINY_WORLD}}}\n"
        "grammar: {morphology: {number: {enabled: true}, aspect: {enabled: true}}}\n",
        encoding="utf-8",
    )
    morphology = load_config(path).grammar.morphology
    assert morphology.number.enabled and morphology.aspect.enabled and not morphology.tense.enabled
    assert "enabled: true" in load_config(path).to_yaml()  # no quoted key in the resolved form


def test_agreement_needs_number() -> None:
    def morphology(**number: Any):
        return corpus_config(grammar={"morphology": {"number": number}}).grammar.morphology

    assert morphology().agreement is False
    assert morphology(enabled=True).agreement is True
    assert morphology(enabled=True, agreement=False).agreement is False


def test_inflection_words() -> None:
    def words(**parts: Any) -> tuple[str, ...]:
        return corpus_config(grammar={"morphology": parts}).grammar.morphology.inflection_words()

    assert words() == ()
    assert words(number={"enabled": True}, tense={"enabled": True}) == ()  # affixes, not words
    assert words(aspect={"enabled": True}) == ("PROGRESSIVE",)  # aspect is a word by default
    assert words(
        number={"enabled": True, "realization": "word"},
        tense={"enabled": True, "realization": "word"},
        aspect={"enabled": True},
    ) == ("PLURAL", "PAST", "PROGRESSIVE")
    assert words(number={"enabled": False, "realization": "word"}) == ()
