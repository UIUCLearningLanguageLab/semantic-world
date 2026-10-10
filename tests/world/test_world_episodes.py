"""Stage a4: episodes, policies, histories, and ``simulate``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic_world.taxonomy.config import ConfigError
from semantic_world.world.__main__ import main
from semantic_world.world.config import config_from_mapping, load_config
from semantic_world.world.definition import RuntimeDefinition, load_definition, runtime_definition
from semantic_world.world.episodes import (
    STATS_STREAM,
    EpisodeGenerator,
    EpisodeSettings,
    Relatedness,
    load_scene_settings,
    read_scene_settings,
    simulate,
)
from semantic_world.world.errors import WorldError
from semantic_world.world.fixtures import fixture_record, read_fixture
from semantic_world.world.generate import WorldResult, define, statistics_episodes
from semantic_world.world.history import (
    History,
    HistoryError,
    initial_state_of,
    read_histories,
    replay,
    write_histories,
)
from semantic_world.world.policies import StepContext, policy, policy_names
from semantic_world.world.runtime import (
    Event,
    able,
    apply,
    interferes,
    legal,
    legal_bindings,
)

DATA = Path("data/world")


@pytest.fixture(scope="module")
def tiny() -> WorldResult:
    return define(load_config(DATA / "tiny.yaml"))


@pytest.fixture(scope="module")
def default() -> WorldResult:
    return define(load_config(DATA / "default.yaml"))


@pytest.fixture(scope="module")
def chain() -> WorldResult:
    return define(load_config(DATA / "tiny_chain.yaml"))


def _generator(result: WorldResult, **settings) -> EpisodeGenerator:
    runtime = result.definition.runtime()
    return EpisodeGenerator(
        runtime,
        Relatedness.from_statics(result.statics, runtime),
        EpisodeSettings(**settings),
        record_legal=True,
    )


def _performable(definition: RuntimeDefinition) -> list[str]:
    return [et.label for et in definition.event_types if not et.is_category]


# ---------------------------------------------------------------------------------------------
# Acceptance: legality, non-interference, replay
# ---------------------------------------------------------------------------------------------


def test_every_event_was_legal_and_no_two_events_interfere(tiny: WorldResult) -> None:
    generator = _generator(tiny)
    definition = generator.definition
    histories = generator.run(1, 200, STATS_STREAM)
    assert sum(len(h.events) for h in histories) > 500
    for history in histories:
        state = initial_state_of(definition, history)
        participants = tuple(definition.entity_index(p) for p in history.participants)
        for step in history.steps:
            events = [Event.from_labels(definition, e.type, e.binding) for e in step.events]
            for event in events:
                assert legal(definition, state, event.event_type, [event.binding])[0]
                assert set(event.binding) <= set(participants)
            for a in range(len(events)):
                for b in range(a + 1, len(events)):
                    assert not interferes(definition, events[a], events[b])
            assert step.legal == {
                label: len(legal_bindings(definition, state, label, participants))
                for label in _performable(definition)
            }
            state = apply(definition, state, events)


def test_replay_reproduces_every_recorded_change_and_nothing_else(tiny: WorldResult) -> None:
    generator = _generator(tiny)
    definition = generator.definition
    for history in generator.run(2, 100, STATS_STREAM):
        states = replay(definition, history)
        assert len(states) == len(history.steps) + 1
        assert history.final == f"TIME.{len(states)}"
        for step, before, after in zip(history.steps, states[:-1], states[1:], strict=True):
            changed = {
                (definition.entity_labels[i], f)
                for i in range(definition.entity_count)
                for j, f in enumerate(definition.base_fluents)
                if before.values[i, j] != after.values[i, j]
            }
            recorded = {(c.entity, c.fluent) for e in step.events for c in e.changes}
            assert changed == recorded
            for e in step.events:
                for c in e.changes:
                    assert after.values[
                        definition.entity_index(c.entity), definition.base_fluent_index(c.fluent)
                    ] == int(c.to)
        # A tampered history does not replay.
        if history.events:
            data = history.to_json()
            event = data["steps"][0]["events"] or next(
                s["events"] for s in data["steps"] if s["events"]
            )
            event[0]["changes"].append(
                {"entity": event[0]["agent"], "fluent": "BOOLFL.1", "to": True}
            )
            with pytest.raises(HistoryError, match="recorded changes"):
                replay(definition, History.from_json(data))
            break


def test_event_labels_and_numbering(tiny: WorldResult) -> None:
    generator = _generator(tiny)
    for history in generator.run(3, 20, STATS_STREAM):
        labels = [e.label for e in history.events]
        assert labels == [f"{history.label}.EVENTINSTANCE.{k}" for k in range(1, len(labels) + 1)]
        assert history.participants[0] == history.seed
        assert len(set(history.participants)) == len(history.participants)
        assert set(history.initial) == set(history.participants)
        assert history.rule_set_id == tiny.rule_set_id
        for k, step in enumerate(history.steps, start=1):
            assert step.step == k


# ---------------------------------------------------------------------------------------------
# Acceptance: no fluents, the chain example, the two policies
# ---------------------------------------------------------------------------------------------


def test_without_fluents_every_able_binding_is_available_at_every_step(tmp_path: Path) -> None:
    data = yaml.safe_load((DATA / "tiny.yaml").read_text(encoding="utf-8"))
    data["fluents"] = {"count": 0}
    data["taxonomy"]["config"] = str(Path("data/taxonomy/tiny_relations.yaml"))
    path = tmp_path / "static.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = define(load_config(path))
    generator = _generator(result)
    definition = generator.definition
    assert definition.base_fluents == ()
    for history in generator.run(1, 100, STATS_STREAM):
        participants = tuple(definition.entity_index(p) for p in history.participants)
        for step in history.steps:
            for label in _performable(definition):
                et = definition.event_type(label)
                if et.arity == 1:
                    candidates = [(p,) for p in sorted(participants)]
                else:
                    candidates = [
                        (a, p) for a in sorted(participants) for p in sorted(participants) if a != p
                    ]
                able_count = int(able(definition, label, candidates).sum()) if candidates else 0
                assert step.legal[label] == able_count
            for event in step.events:
                assert event.changes == ()
        assert (
            not history.quiescent or all(count == 0 for count in history.steps[-1].legal)
            if history.steps
            else True
        )


def _catch_before_eat(result: WorldResult, silence: bool) -> int:
    """Run episodes and check the order the chain's preconditions force: an eating of a patient
    (EVENTTYPE2.1.2, which needs BOOLFL.2 of the patient) follows, in an earlier step, an event
    that set BOOLFL.2 of that patient. With ``silence``, every setter but the chain's catch
    (EVENTTYPE2.1.1) is weighted out, so the setter is always a catch. Returns the eatings."""
    definition = result.definition.runtime()
    setters = {
        et.label
        for et in definition.event_types
        if not et.is_category and any(e.fluent == "BOOLFL.2" and e.value for e in et.effects)
    }
    weights = {label: 0.0 for label in setters - {"EVENTTYPE2.1.1"}} if silence else {}
    generator = _generator(result, size=(3, 6), steps=(6, 10), event_type_weights=weights)
    eaten = 0
    for history in generator.run(1, 300, STATS_STREAM):
        caught: dict[str, int] = {}
        for step in history.steps:
            for event in step.events:
                if event.type == "EVENTTYPE2.1.2":
                    assert event.patient in caught and caught[event.patient] < step.step
                    eaten += 1
            for event in step.events:
                for change in event.changes:
                    if change.fluent == "BOOLFL.2" and change.to:
                        assert not silence or event.type == "EVENTTYPE2.1.1"
                        caught[change.entity] = step.step
                    if change.fluent == "BOOLFL.2" and not change.to:
                        caught.pop(change.entity, None)
        states = replay(definition, history)
        for step, before in zip(history.steps, states[:-1], strict=True):
            for event in step.events:
                if event.type == "EVENTTYPE2.1.2":
                    patient = definition.entity_index(event.patient)
                    agent = definition.entity_index(event.agent)
                    assert before.values[patient, definition.base_fluent_index("BOOLFL.2")] == 1
                    assert before.values[agent, definition.base_fluent_index("BOOLFL.1")] == 1
    return eaten


def test_chain_example_events_follow_the_preconditions(chain: WorldResult) -> None:
    """On the chain example, every eating follows the catching that made it legal, never in
    the same step."""
    assert _catch_before_eat(chain, silence=False) > 20
    assert _catch_before_eat(chain, silence=True) > 20


def test_chain_example_completes_on_every_seed() -> None:
    """The acceptance test of stage a5b: with the chain's explicit requirement (the agent is
    bigger than the patient on the first scalar, shared by both event types), whatever is
    caught can be eaten, so on every seed 1 to 10 an eating (EVENTTYPE2.1.2) follows a catching
    (EVENTTYPE2.1.1) of the same patient. The statistics episodes see both event types occur,
    and with every other setter of BOOLFL.2 weighted out the forced order holds."""
    data = yaml.safe_load((DATA / "tiny_chain.yaml").read_text(encoding="utf-8"))
    for seed in range(1, 11):
        config = config_from_mapping(data, source=str(DATA / "tiny_chain.yaml"), seed=seed)
        assert config.seed == seed
        result = define(config)
        occurred = result.stats["episodes"]["occurrence_share"]
        assert occurred["EVENTTYPE2.1.1"] > 0 and occurred["EVENTTYPE2.1.2"] > 0, seed
        assert "EVENTTYPE2.1.2" not in result.stats["episodes"]["never_legal"], seed
        assert _catch_before_eat(result, silence=True) > 0, seed


def _occurrence_and_legal(result: WorldResult, name: str, count: int) -> tuple[dict, dict]:
    generator = _generator(result, policy=name)
    occurred = dict.fromkeys(generator.performable, 0)
    legal_total = dict.fromkeys(generator.performable, 0)
    for history in generator.run(5, count, STATS_STREAM):
        for step in history.steps:
            for label, n in step.legal.items():
                legal_total[label] += n
            for event in step.events:
                occurred[event.type] += 1
    return occurred, legal_total


def test_policies_differ_in_the_expected_direction(default: WorldResult) -> None:
    """Under uniform_event, an event type legal for many bindings occurs more often, so the
    occurrences follow the legal counts more closely than under uniform_event_type."""
    assert set(policy_names()) >= {"uniform_event", "uniform_event_type"}
    uniform_event, legal_event = _occurrence_and_legal(default, "uniform_event", 300)
    uniform_type, legal_type = _occurrence_and_legal(default, "uniform_event_type", 300)
    assert legal_event == legal_type or sum(legal_event.values()) > 0
    labels = [label for label in uniform_event if legal_event[label] > 0]
    assert len(labels) > 5

    def correlation(occurred: dict, legal_total: dict) -> float:
        x = np.array([legal_total[label] for label in labels], dtype=float)
        y = np.array([occurred[label] for label in labels], dtype=float)
        return float(np.corrcoef(x, y)[0, 1])

    def spread(occurred: dict) -> float:
        y = np.array([occurred[label] for label in labels], dtype=float)
        return float(y.std() / y.mean())

    assert sum(uniform_event.values()) > 1000 and sum(uniform_type.values()) > 1000
    assert correlation(uniform_event, legal_event) > correlation(uniform_type, legal_type)
    assert spread(uniform_event) > spread(uniform_type)


def test_policies_never_change_the_state_and_respect_weights(tiny: WorldResult) -> None:
    generator = _generator(tiny)
    definition = generator.definition
    history = generator.run(1, 1, STATS_STREAM)[0]
    state = initial_state_of(definition, history)
    participants = tuple(definition.entity_index(p) for p in history.participants)
    legal_map = {
        label: legal_bindings(definition, state, label, participants)
        for label in _performable(definition)
    }
    only = next(label for label, b in legal_map.items() if b)
    weights = {label: (1.0 if label == only else 0.0) for label in legal_map}
    context = StepContext(definition, state, participants, legal_map, weights, 0.5)
    before = state.values.copy()
    for name in ("uniform_event", "uniform_event_type"):
        events = policy(name)(context, np.random.default_rng(0), 5)
        assert events and all(e.event_type == only for e in events)
        assert len(set(events)) == len(events)
        for a in events:
            for b in events:
                assert a == b or not interferes(definition, a, b)
        assert np.array_equal(state.values, before)
    with pytest.raises(WorldError, match="unknown policy"):
        policy("random_walk")


# ---------------------------------------------------------------------------------------------
# Determinism, participants, initial states, quiescence
# ---------------------------------------------------------------------------------------------


def test_each_episode_depends_only_on_its_own_part(tiny: WorldResult) -> None:
    generator = _generator(tiny)
    five = generator.run(7, 5, STATS_STREAM)
    third = generator.run(7, 1, STATS_STREAM, first=3)[0]
    assert third == five[2]
    again = generator.run(7, 5, STATS_STREAM)
    assert again == five
    other_seed = generator.run(8, 5, STATS_STREAM)
    assert other_seed != five
    other_stream = generator.run(7, 5, "world:episodes")
    assert other_stream != five
    labels = [h.label for h in five]
    assert labels == [f"SCENE.{k}" for k in range(1, 6)]
    given = generator.run(7, 2, STATS_STREAM, seeds=[4, 4])
    assert all(h.seed == generator.definition.entity_labels[4] for h in given)


def old_scene_participants(statics, settings: EpisodeSettings):
    """A reference of the participant draw of the corpus's scene generator before stage a5a
    (``corpus/scenes.py``): the weights over the instances, and the draw. Thematic relatedness
    comes from the world's ``thematic`` table, and similarity from the leaves' generative
    vectors with the taxonomy's similarity settings, undefined and negative values counting 0."""
    from semantic_world.taxonomy.similarity import similarity_matrix

    taxonomy = statics.taxonomy
    instances, tree = taxonomy.instances, taxonomy.tree
    leaves = [leaf.label for leaf in tree.leaves]
    number = {label: i for i, label in enumerate(leaves)}
    leaf_rows = np.array([tree.categories.index(leaf) for leaf in tree.leaves], dtype=np.intp)
    leaf_of_row = {int(row): i for i, row in enumerate(leaf_rows)}
    leaf = np.array([leaf_of_row[int(row)] for row in instances.leaf_index], dtype=np.intp)
    thematic = np.zeros((len(leaves), len(leaves)))
    if statics.relation_stats is not None:
        for row in statics.relation_stats.thematic.iter_rows(named=True):
            a, b = number[row["leaf_a"]], number[row["leaf_b"]]
            thematic[a, b] = thematic[b, a] = row["thematic"]
    analysis = taxonomy.config.analysis
    start = 0 if analysis.similarity_features == "all" else taxonomy.vectors.isa_count
    similarity = similarity_matrix(
        taxonomy.vectors.generative[leaf_rows][:, start:], analysis.similarity_metric
    )
    similarity = np.clip(np.nan_to_num(similarity, nan=0.0), 0.0, None)
    weights = settings.participant_weights

    def participant_weights(seed: int) -> np.ndarray:
        values = (
            weights["thematic"] * thematic[leaf, leaf[seed]]
            + weights["taxonomic"] * similarity[leaf, leaf[seed]]
            + weights["constant"]
        )
        values[seed] = 0.0
        return values

    def draw(rng: np.random.Generator, seed: int) -> tuple[int, ...]:
        low, high = settings.size
        size = int(rng.integers(low, high + 1))
        values = participant_weights(seed)
        size = min(size, int(np.count_nonzero(values)))
        if size == 0:
            return (seed,)
        drawn = rng.choice(len(values), size=size, replace=False, p=values / values.sum())
        return (seed,) + tuple(int(i) for i in drawn)

    return participant_weights, draw


def test_participants_are_drawn_as_corpus_scenes_drew_them(tiny: WorldResult) -> None:
    """The weights are the old corpus scene generator's: thematic relatedness, taxonomic
    similarity, and a constant, over the leaf of each entity and the seed's leaf."""
    generator = _generator(tiny)
    definition = generator.definition
    theirs, draw = old_scene_participants(tiny.statics, generator.settings)
    for seed in range(definition.entity_count):
        assert np.allclose(generator.participant_weights(seed), theirs(seed), atol=1e-6)
        rng_a, rng_b = np.random.default_rng(seed), np.random.default_rng(seed)
        assert generator.draw_participants(rng_a, seed) == draw(rng_b, seed)


def test_relatedness_from_a_run_equals_in_memory(tiny: WorldResult, tmp_path: Path) -> None:
    runtime = tiny.definition.runtime()
    in_memory = Relatedness.from_statics(tiny.statics, runtime)
    from_run = Relatedness.from_run(tiny.write(tmp_path / "tiny"), runtime)
    assert from_run.leaves == in_memory.leaves
    assert np.allclose(from_run.thematic, in_memory.thematic)
    assert np.allclose(from_run.similarity, in_memory.similarity)
    assert in_memory.thematic.any() and in_memory.similarity.any()
    assert np.array_equal(from_run.entity_leaf, in_memory.entity_leaf)


def test_relatedness_without_two_place_event_types(tmp_path: Path) -> None:
    data = yaml.safe_load((DATA / "tiny.yaml").read_text(encoding="utf-8"))
    data["event_types"]["binary"] = None
    path = tmp_path / "unary.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = define(load_config(path))
    assert result.statics.relations is None and result.statics.relation_stats is None
    assert [et.label for et in result.event_types.event_types] == [
        f"EVENTTYPE1.{k}" for k in range(1, 5)
    ]
    runtime = result.definition.runtime()
    in_memory = Relatedness.from_statics(result.statics, runtime)
    assert not in_memory.thematic.any()
    assert in_memory.similarity.max() > 0
    folder = result.write(tmp_path / "unary_run")
    from_run = Relatedness.from_run(folder, runtime)
    assert from_run.leaves == in_memory.leaves
    assert np.allclose(from_run.similarity, in_memory.similarity)
    assert np.array_equal(from_run.entity_leaf, in_memory.entity_leaf)
    generator = EpisodeGenerator(runtime, in_memory, EpisodeSettings(), record_legal=True)
    assert generator.run(1, 3, STATS_STREAM)


def test_initial_state_kept_or_redrawn(tiny: WorldResult) -> None:
    keep = _generator(tiny)
    definition = keep.definition
    entities = {
        label: tuple(
            f for j, f in enumerate(definition.base_fluents) if definition.initial_values[i, j]
        )
        for i, label in enumerate(definition.entity_labels)
    }
    for history in keep.run(1, 10, STATS_STREAM):
        assert history.initial == {p: entities[p] for p in history.participants}
    redraw = _generator(tiny, initial="redraw")
    rates = redraw.initial_rates
    assert rates == definition.initial_rates == {f.label: f.initial_rate for f in tiny.fluents.base}
    differs = 0
    for history in redraw.run(1, 50, STATS_STREAM):
        for participant, fluents in history.initial.items():
            differs += fluents != entities[participant]
            for fluent, rate in rates.items():
                if rate == 0.0:
                    assert fluent not in fluents
                if rate == 1.0:
                    assert fluent in fluents
        replay(definition, history)
    assert differs > 0
    # Rates given apart replace the definition's; a rate missing for a base fluent is an error.
    halves = dict.fromkeys(definition.base_fluents, 0.5)
    given = EpisodeGenerator(
        definition, keep.relatedness, EpisodeSettings(initial="redraw"), halves
    )
    assert given.initial_rates == halves
    with pytest.raises(WorldError, match="initial rate"):
        EpisodeGenerator(
            definition, keep.relatedness, EpisodeSettings(initial="redraw"), {"BOOLFL.1": 1.0}
        )


def test_quiescence_ends_an_episode() -> None:
    """A world where nothing is ever legal ends at TIME.1 with no steps."""
    fixture = read_fixture(Path("tests/fixtures/world/hand_01_no_event.json"))
    record = dict(fixture["definition"])
    # without the event types that are always legal (EVENTTYPE1.4 and EVENTTYPE1.5 since b1)
    always = {"EVENTTYPE1.3", "EVENTTYPE1.4", "EVENTTYPE1.5"}
    record["symbols"] = [s for s in record["symbols"] if s["label"] not in always]
    record["event_types"] = [e for e in record["event_types"] if e["label"] not in always]
    entities = [{**row, "PROPERTY.1": 1} for row in fixture["entities"]]
    rehashed = fixture_record("quiet", "", record, entities, {}, [], None)["definition"]
    definition = runtime_definition(rehashed, entities)
    initial = {label: ["BOOLFL.1"] for label in definition.entity_labels}
    from semantic_world.world.runtime import State

    state = State.from_true(definition, initial)
    assert all(not legal_bindings(definition, state, label) for label in _performable(definition))
    relatedness = Relatedness.from_frames(definition, None)
    generator = EpisodeGenerator(
        definition,
        relatedness,
        EpisodeSettings(initial="redraw"),
        {"BOOLFL.1": 1.0, "BOOLFL.2": 0.0},
    )
    history = generator.run(1, 1, STATS_STREAM)[0]
    assert history.quiescent and history.steps == () and history.final == "TIME.1"
    replay(definition, history)


# ---------------------------------------------------------------------------------------------
# Settings, simulate, statistics
# ---------------------------------------------------------------------------------------------


def test_scene_settings_defaults_and_corpus_file(tiny: WorldResult) -> None:
    definition = tiny.definition.runtime()
    defaults = EpisodeSettings()
    from_corpus = load_scene_settings("data/corpus/default.yaml", definition)
    assert from_corpus == defaults
    assert defaults.resolved()["event_type_weights"] == "uniform"
    settings = read_scene_settings(
        {
            "scene": {
                "size": [1, 2],
                "event_type_weights": {"EVENTTYPE1.1": 2.0, "EVENTTYPE2.1.1": 0.0},
                "policy": "uniform_event_type",
                "initial": "redraw",
            }
        },
        "test",
        definition,
    )
    assert settings.size == (1, 2)
    assert settings.event_type_weights == {"EVENTTYPE1.1": 2.0, "EVENTTYPE2.1.1": 0.0}
    assert settings.policy == "uniform_event_type" and settings.initial == "redraw"
    for scene, text in (
        ({"policy": "nothing"}, "policy"),
        ({"size": [3, 2]}, "size"),
        ({"event_type_weights": {"EVENTTYPE9.9": 1.0}}, "EVENTTYPE9.9"),
        ({"verb_weights": {"EVENTTYPE1.1": 1.0}}, r"verb_weights.*scene.event_type_weights"),
        ({"participant_weights": {"other": 1.0}}, "participant_weights.other"),
        ({"unknown": 1}, "unknown"),
        ({"initial": "fresh"}, "initial"),
    ):
        with pytest.raises(ConfigError, match=text):
            read_scene_settings({"scene": scene}, "test", definition)
    with pytest.raises(WorldError, match="not an event type"):
        EpisodeGenerator(
            definition,
            Relatedness.from_frames(definition, None),
            EpisodeSettings(event_type_weights={"EVENTTYPE9.9": 1.0}),
        )


def test_simulate_command_and_histories_file(
    tiny: WorldResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tiny.write(tmp_path / "tiny")
    rates = {f.label: f.initial_rate for f in tiny.fluents.base}
    assert load_definition(folder).initial_rates == rates
    assert tiny.definition.runtime().initial_rates == rates
    assert main(["simulate", str(folder), "--episodes", "4", "--legal"]) == 0
    out = capsys.readouterr().out
    assert "4 episodes" in out
    path = folder / "episodes.jsonl"
    histories = read_histories(path)
    assert len(histories) == 4 and all(h.policy == "uniform_event" for h in histories)
    definition = tiny.definition.runtime()
    for history in histories:
        replay(definition, history)
        assert all(step.legal is not None for step in history.steps)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["label"] for line in lines] == [
        "SCENE.1",
        "SCENE.2",
        "SCENE.3",
        "SCENE.4",
    ]
    assert "legal" in json.loads(lines[0])["steps"][0]
    # The same seed gives the same file; the run's seed is the default.
    _, same = simulate(
        folder, 4, seed=tiny.config.seed, record_legal=True, out=tmp_path / "again.jsonl"
    )
    assert same == histories
    # Without --legal, no legal counts; with a corpus configuration, its scene block.
    config = tmp_path / "corpus.yaml"
    config.write_text(
        yaml.safe_dump({"scene": {"policy": "uniform_event_type", "steps": [2, 2]}}),
        encoding="utf-8",
    )
    other = tmp_path / "other.jsonl"
    assert (
        main(
            [
                "simulate",
                str(folder),
                "--episodes",
                "2",
                "--seed",
                "3",
                "--config",
                str(config),
                "--out",
                str(other),
            ]
        )
        == 0
    )
    for history in read_histories(other):
        assert history.policy == "uniform_event_type"
        assert len(history.steps) <= 2 and all(step.legal is None for step in history.steps)
        replay(definition, history)
    round_trip = tmp_path / "copy.jsonl"
    write_histories(round_trip, histories)
    assert round_trip.read_bytes() == path.read_bytes()


def test_world_stats_report_episodes_redraws_and_never_legal(
    tiny: WorldResult, tmp_path: Path
) -> None:
    episodes = tiny.stats["episodes"]
    assert episodes["count"] == 1000
    performable = [et.label for et in tiny.event_types.leaves]
    assert list(episodes["legal_share"]) == performable
    assert list(episodes["occurrence_share"]) == performable
    for label in performable:
        assert 0.0 <= episodes["occurrence_share"][label] <= episodes["legal_share"][label] <= 1.0
    assert list(episodes["never_legal"]) == [
        label for label in performable if episodes["legal_share"][label] == 0
    ]
    assert set(episodes["never_legal"].values()) <= {"never_able", "preconditions"}
    assert len(episodes["warnings"]) == len(episodes["never_legal"])
    assert 0.0 <= episodes["quiescent_share"] <= 1.0
    assert episodes["mean_changes_per_event"] >= 0.0
    assert 0.0 < episodes["two_place_share"] < 1.0
    assert set(episodes["precondition_redraws"]) <= set(performable)
    folder = tiny.write(tmp_path / "tiny")
    stats = yaml.safe_load((folder / "world_stats.yaml").read_text(encoding="utf-8"))
    assert stats["fluents"] == {"base": 3, "derived": 1}
    assert stats["episodes"]["count"] == 1000
    assert stats["episodes"]["precondition_redraws"] == episodes["precondition_redraws"]
    assert stats["episodes"]["never_legal"] == episodes["never_legal"]
    # The statistics episodes are the default policy on the world:stats stream, with the legal
    # counts recorded; the statistics of the final definition carry the redraw counts over.
    generator = _generator(tiny)
    histories = generator.run(tiny.config.seed, 1000, STATS_STREAM)
    from semantic_world.world.runtime import able_table
    from semantic_world.world.stats import episode_stats

    able = able_table(generator.definition)
    never_able = [label for label in generator.performable if not able[label].any()]
    assert (
        episode_stats(
            histories, generator.performable, episodes["precondition_redraws"], never_able
        )
        == episodes
    )
    assert (
        statistics_episodes(
            tiny.definition, tiny.statics, tiny.config, episodes["precondition_redraws"]
        )
        == episodes
    )
