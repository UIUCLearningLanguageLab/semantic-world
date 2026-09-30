"""Stage 5 acceptance: reset, step, manifests, the action mask, observation shapes, props, and
agent_seed, through the Python API."""

import hashlib

import numpy as np
import pytest

import semantic_world as sw


def test_reset_gives_one_observation_per_agent(smoke_config):
    world = sw.make(smoke_config)
    obs, info = world.reset(seed=3)
    assert world.agents == ["human_0"]
    assert list(obs) == ["human_0"]
    assert isinstance(obs["human_0"], sw.Observation)
    assert info["tick"] == 0 and info["done"] is False
    assert info["agents"]["human_0"]["alive"] is True
    assert world.seed == 3
    assert world.agents_awaiting_action == ["human_0"]


def test_manifests_describe_every_block_and_action(smoke_config):
    world = sw.make(smoke_config)
    world.reset(seed=0)
    manifests = world.manifests()
    sensors = manifests["human_0"]["sensors"]
    names = [b["name"] for b in sensors["blocks"]]
    assert names == ["eyes", "interoception", "touch", "proprioception"]
    eyes, intero, touch, proprio = sensors["blocks"]
    assert eyes["shape"] == [64, 64, 3] and eyes["dtype"] == "uint8" and eyes["in_vector"] is False
    assert intero["shape"] == [5] and intero["offset"] == 0
    assert intero["labels"] == ["hunger", "thirst", "fatigue", "cold", "health"]
    assert touch["offset"] == 5 and touch["labels"] == ["intensity", "dir_x", "dir_y", "dir_z"]
    assert proprio["offset"] == 9 and len(proprio["labels"]) == 6
    assert sensors["vector_length"] == 15
    assert sensors["props"] is False
    actions = manifests["human_0"]["actions"]
    names = [a["name"] for a in actions["actions"]]
    assert names == ["noop", "move", "turn", "eat", "drink", "sleep"]
    move = actions["actions"][1]
    assert move["args"][0]["kind"] == "continuous" and move["args"][0]["name"] == "direction"
    target_slot = actions["actions"][3]["args"][0]
    assert target_slot == {"kind": "target", "name": "target", "has": "edible", "optional": True}


def test_observation_shapes_and_vector(smoke_config):
    world = sw.make(smoke_config)
    obs, _ = world.reset(seed=0)
    o = obs["human_0"]
    assert o.blocks["eyes"].shape == (64, 64, 3) and o.blocks["eyes"].dtype == np.uint8
    assert o.blocks["interoception"].shape == (5,) and o.blocks["interoception"].dtype == np.float32
    assert o.blocks["touch"].shape == (4,)
    assert o.blocks["proprioception"].shape == (6,)
    assert o.vector.shape == (15,) and o.vector.dtype == np.float32
    expected = np.concatenate(
        [o.blocks["interoception"], o.blocks["touch"], o.blocks["proprioception"]]
    )
    assert np.array_equal(o.vector, expected)
    # Needs start at 0, health at 1; touch is all zeros; the heading is a unit vector.
    np.testing.assert_allclose(o.blocks["interoception"], [0, 0, 0, 0, 1])
    assert not o.blocks["touch"].any()
    sin, cos = o.blocks["proprioception"][3:5]
    assert abs(sin**2 + cos**2 - 1) < 1e-5
    assert o.text is None


def test_mask_follows_the_option(smoke_config):
    world = sw.make(smoke_config)
    obs, _ = world.reset(seed=0)
    mask = obs["human_0"].mask
    assert mask is not None and mask.dtype == np.bool_ and mask.shape == (6,)
    assert mask[0], "noop is always available"
    assert mask[1] and mask[2], "move and turn are unmasked while awake"
    assert not mask[3] and not mask[4], "nothing to eat or drink at the start"
    world = sw.make(smoke_config, overrides={"options.impossible_actions": "attempt_and_fail"})
    obs, _ = world.reset(seed=0)
    assert obs["human_0"].mask is None


def test_props_when_the_sensor_is_on(smoke_config):
    world = sw.make(smoke_config)
    obs, _ = world.reset(seed=0)
    assert obs["human_0"].props is None
    world = sw.make(smoke_config, overrides={"views.propositional_sensor": True})
    obs, _ = world.reset(seed=0)
    props = obs["human_0"].props
    assert isinstance(props, list) and "(is human_0 human)" in props
    assert "(hunger human_0 0.00)" in props and "(health human_0 1.00)" in props
    assert world.manifests()["human_0"]["sensors"]["props"] is True


def test_agent_seed_is_the_named_stream(smoke_config):
    world = sw.make(smoke_config)
    world.reset(seed=5)
    seed = world.agent_seed("human_0")
    digest = hashlib.sha256((5).to_bytes(8, "little") + b"agent:human_0").digest()
    assert seed == int.from_bytes(digest, "little")
    world.reset(seed=6)
    assert world.agent_seed("human_0") != seed
    with pytest.raises(KeyError):
        world.agent_seed("tree_0")


def test_step_applies_actions_and_reports_events(smoke_config):
    world = sw.make(smoke_config)
    obs, _ = world.reset(seed=0)
    before = world.debug_state()["entities"]
    human = next(e for e in before if e["id"] == "human_0")
    obs, events, info = world.step({"human_0": {"type": "move", "direction": 0.0, "speed": 0.5}})
    assert info["tick"] == 2 and world.tick == 2 and abs(world.time_s - 0.2) < 1e-9
    kinds = [e["event"] for e in events]
    assert kinds == ["action_started", "action_completed"]
    assert events[0]["agent"] == "human_0" and events[0]["action"] == "move"
    after = next(e for e in world.debug_state()["entities"] if e["id"] == "human_0")
    moved = ((after["x"] - human["x"]) ** 2 + (after["z"] - human["z"]) ** 2) ** 0.5
    assert abs(moved - 1.4 * 0.5 / 0.6 * 0.2) < 1e-6
    # Proprioception reports the speed of the last tick, scaled by the run speed.
    speed = obs["human_0"].blocks["proprioception"][0]
    assert abs(speed - (1.4 * 0.5 / 0.6) / 3.5) < 1e-5
    # A failed action is an event, and proprioception reports it.
    world = sw.make(smoke_config, overrides={"options.impossible_actions": "attempt_and_fail"})
    world.reset(seed=0)
    obs, events, _ = world.step({"human_0": {"type": "eat"}})
    assert [e["event"] for e in events] == ["action_failed"]
    assert obs["human_0"].blocks["proprioception"][5] == 1.0


def test_step_rejects_bad_input(smoke_config):
    world = sw.make(smoke_config)
    with pytest.raises(RuntimeError):
        world.step({})
    world.reset(seed=0)
    with pytest.raises(ValueError):
        world.step({"tree_0": {"type": "noop"}})
    with pytest.raises(ValueError):
        world.step({"human_0": {"type": "move", "speed": 1.0}})
    with pytest.raises(ValueError):
        world.step({"human_0": {"type": "eat", "target": "berry_bush_0"}})
    with pytest.raises(ValueError):
        sw.make(smoke_config, overrides={"clock.mode": "realtime"})


def test_same_seed_and_actions_reproduce_the_state_hash(smoke_config):
    def run(seed: int) -> str:
        world = sw.make(smoke_config)
        world.reset(seed=seed)
        rng = np.random.default_rng(world.agent_seed("human_0"))
        for _ in range(200):
            action = {"type": "move", "direction": rng.uniform(-3, 3), "speed": rng.uniform(0, 1)}
            world.step({"human_0": action})
        return world.state_hash()

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_dead_agents_drop_out_of_observations(smoke_config):
    world = sw.make(smoke_config, overrides={"lifetime.default_days": 2})
    world.reset(seed=0)
    events = []
    while not world.done:
        _, step_events, _ = world.step({})
        events.extend(step_events)
    assert any(e["event"] == "died" for e in events)
    assert world.agents_awaiting_action == []
    with pytest.raises(ValueError, match="done"):
        world.step({})
