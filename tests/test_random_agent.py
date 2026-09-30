"""Stage 5 acceptance: the random agent runs for one day from Python."""

import time

import semantic_world as sw


def test_random_agent_runs_for_a_day(smoke_config):
    world = sw.make(smoke_config, overrides={"lifetime.default_days": 1})
    obs, info = world.reset(seed=3)
    manifests = world.manifests()
    brains = {
        aid: sw.agents.create("random", manifests[aid], seed=world.agent_seed(aid))
        for aid in world.agents
    }
    started = time.perf_counter()
    decisions = 0
    events = []
    while not world.done:
        actions = {aid: brains[aid].act(obs[aid]) for aid in world.agents_awaiting_action}
        obs, step_events, info = world.step(actions)
        events.extend(step_events)
        decisions += 1
    elapsed = time.perf_counter() - started
    assert info["tick"] == 14_400 or not info["agents"]["human_0"]["alive"]
    assert decisions > 100
    kinds = {e["event"] for e in events}
    assert "action_started" in kinds
    assert "action_failed" not in kinds, "with masking, the random agent never fails"
    assert elapsed < 60, f"one day took {elapsed:.1f} s"


def test_the_registry_knows_the_random_agent():
    assert "random" in sw.agents.names()
    try:
        sw.agents.create("nobody", {}, 0)
    except KeyError as e:
        assert "nobody" in str(e)
    else:
        raise AssertionError("unknown names must raise")
