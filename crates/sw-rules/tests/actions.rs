//! Acceptance (stage 4): unit tests for every action's precondition, effect, duration,
//! interruption, and failure.

mod common;

use common::{AGENT, Scene, TICKS_PER_HOUR, has_event, walk};
use sw_rules::{ActionChoice, EventKind, RuleError};

#[test]
fn the_manifest_lists_every_action_in_file_order() {
    let scene = Scene::masked();
    let manifest = scene.engine.manifest(AGENT).unwrap();
    let names: Vec<&str> = manifest.actions.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["noop", "move", "turn", "eat", "drink", "sleep"]);
    assert_eq!(manifest.actions[1].args.len(), 2);
    assert!(manifest.actions[3].durative && !manifest.actions[1].durative);
    assert_eq!(scene.mask().len(), 6);
}

#[test]
fn noop_does_nothing_but_pass_time() {
    let mut scene = Scene::masked();
    let before = scene.human();
    let events = scene.noop();
    assert!(events.is_empty());
    assert_eq!(scene.tick(), 2);
    let after = scene.human();
    assert_eq!(
        (after.x, after.z, after.yaw),
        (before.x, before.z, before.yaw)
    );
    assert!(!after.agent.unwrap().last_action_failed);
}

#[test]
fn move_walks_runs_and_turns() {
    let mut scene = Scene::masked();
    // Walking at speed 0.5: 1.4 × 0.5 / 0.6 m/s for 0.2 s.
    let events = scene.decide(walk(0.0, 0.5));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionStarted { action, .. } if action == "move")
    ));
    assert!(has_event(&events, |k| matches!(
        k,
        EventKind::ActionCompleted { .. }
    )));
    let h = scene.human();
    assert!((h.x - 1.4 * 0.5 / 0.6 * 0.2).abs() < 1e-9, "{}", h.x);
    // Running at speed 0.8: 3.5 m/s, and fatigue rises twice as fast.
    let fatigue_before = scene.need("fatigue");
    scene.decide(walk(std::f64::consts::FRAC_PI_2, 0.8));
    let h = scene.human();
    assert!((h.z - 0.7).abs() < 1e-9, "{}", h.z);
    assert!((h.yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    let running_rise = scene.need("fatigue") - fatigue_before;
    let fatigue_before = scene.need("fatigue");
    scene.decide(walk(0.0, 0.3));
    let walking_rise = scene.need("fatigue") - fatigue_before;
    assert!((running_rise - 2.0 * walking_rise).abs() < 1e-9);
    // Arguments are clamped to their ranges.
    scene.decide(walk(0.0, 5.0));
    assert!(scene.human().agent.unwrap().speed_mps <= 3.5);
    // A missing argument is a usage error, not a world failure.
    let mut choices = std::collections::BTreeMap::new();
    choices.insert(
        AGENT.to_string(),
        ActionChoice::named("move").with("speed", 1.0),
    );
    assert!(matches!(
        scene.engine.step(&mut scene.world, &choices),
        Err(RuleError::Usage(_))
    ));
}

#[test]
fn turn_rotates_within_a_quarter_turn() {
    let mut scene = Scene::masked();
    scene.decide(ActionChoice::named("turn").with("angle", 0.5));
    assert!((scene.human().yaw - 0.5).abs() < 1e-12);
    scene.decide(ActionChoice::named("turn").with("angle", -3.0));
    assert!(
        (scene.human().yaw - (0.5 - std::f64::consts::FRAC_PI_4)).abs() < 1e-12,
        "clamped to −π/4"
    );
}

#[test]
fn move_and_turn_need_the_agent_awake() {
    let mut scene = Scene::attempt_and_fail();
    scene.world.set_asleep(AGENT, true).unwrap();
    assert!(!scene.mask_of("move") && !scene.mask_of("turn") && scene.mask_of("noop"));
    let events = scene.decide(walk(0.0, 1.0));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionFailed { action, reason } if action == "move" && reason.contains("awake(self)"))
    ));
    let h = scene.human();
    assert_eq!(h.x, 0.0);
    assert!(h.agent.unwrap().last_action_failed);
    // A masked world reports the failure too, and says the action was masked.
    let mut scene = Scene::masked();
    scene.world.set_asleep(AGENT, true).unwrap();
    let events = scene.decide(ActionChoice::named("turn").with("angle", 0.1));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionFailed { reason, .. } if reason.starts_with("masked"))
    ));
}

/// A bush 0.8 m in front of the agent, with its surface gap under 1 m.
fn bush_in_front(scene: &mut Scene) {
    scene.place("berry_bush_0", 0.25 + 0.8 + 0.6, 0.0, 0.0);
}

#[test]
fn eat_takes_two_seconds_then_a_berry_lowers_hunger() {
    let mut scene = Scene::masked();
    scene.idle_ticks(12 * TICKS_PER_HOUR);
    assert!((scene.need("hunger") - 0.25).abs() < 1e-6);
    bush_in_front(&mut scene);
    assert!(scene.mask_of("eat"));
    let events = scene.decide(ActionChoice::named("eat"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionStarted { action, target } if action == "eat" && target.as_deref() == Some("berry_bush_0"))
    ));
    assert_eq!(scene.engine.current_action(AGENT).unwrap().action, "eat");
    // Nine more decisions: 20 ticks = 2 s in all. Nothing happens before the end.
    for _ in 0..8 {
        let events = scene.noop();
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(3));
    }
    let events = scene.noop();
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionCompleted { action } if action == "eat")
    ));
    assert!(scene.engine.current_action(AGENT).is_none());
    assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(2));
    // Hunger was 0.25, rose a little over the 2 s, and fell by 0.25 on completion.
    assert!(scene.need("hunger") < 0.001, "{}", scene.need("hunger"));
}

#[test]
fn interrupted_eating_has_no_effect() {
    let mut scene = Scene::masked();
    bush_in_front(&mut scene);
    scene.decide(ActionChoice::named("eat"));
    scene.noop();
    let events = scene.decide(ActionChoice::named("turn").with("angle", 0.0));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionInterrupted { action, by } if action == "eat" && by == "turn")
    ));
    scene.idle_ticks(40);
    assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(3));
    assert!(scene.engine.current_action(AGENT).is_none());
}

#[test]
fn eat_fails_when_not_facing_too_far_or_empty() {
    let mut scene = Scene::attempt_and_fail();
    // Too far: the gap is 2 m.
    scene.place("berry_bush_0", 0.25 + 2.0 + 0.6, 0.0, 0.0);
    assert!(!scene.mask_of("eat"));
    let events = scene.decide(ActionChoice::named("eat"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionFailed { action, reason } if action == "eat" && reason == "no target in reach")
    ));
    assert!(scene.human().agent.unwrap().last_action_failed);
    // Not facing: the bush is behind the agent.
    scene.place("berry_bush_0", -(0.25 + 0.8 + 0.6), 0.0, 0.0);
    assert!(!scene.mask_of("eat"));
    let events = scene.decide(ActionChoice::named("eat"));
    assert!(has_event(&events, |k| matches!(
        k,
        EventKind::ActionFailed { .. }
    )));
    // Facing it, three berries can be eaten, and then the bush is empty.
    scene.place(AGENT, 0.0, 0.0, std::f64::consts::PI);
    for berries_left in [2, 1, 0] {
        assert!(scene.mask_of("eat"));
        scene.decide(ActionChoice::named("eat"));
        scene.idle_ticks(18);
        assert_eq!(
            scene.entity("berry_bush_0").stock("berries"),
            Some(berries_left)
        );
    }
    assert!(!scene.mask_of("eat"), "an empty bush is not edible");
    let events = scene.decide(ActionChoice::named("eat"));
    assert!(has_event(&events, |k| matches!(
        k,
        EventKind::ActionFailed { .. }
    )));
}

#[test]
fn targets_by_id_need_the_propositional_sensor() {
    let mut scene = Scene::masked();
    bush_in_front(&mut scene);
    let mut choices = std::collections::BTreeMap::new();
    choices.insert(
        AGENT.to_string(),
        ActionChoice::named("eat").targeting("berry_bush_0"),
    );
    assert!(matches!(
        scene.engine.step(&mut scene.world, &choices),
        Err(RuleError::NotAllowed { .. })
    ));

    let mut scene = Scene::new(&[("views.propositional_sensor", serde_json::Value::from(true))]);
    bush_in_front(&mut scene);
    scene.place("berry_bush_1", 0.25 + 0.9 + 0.6, 0.3, 0.0);
    let events = scene.decide(ActionChoice::named("eat").targeting("berry_bush_1"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionStarted { target, .. } if target.as_deref() == Some("berry_bush_1"))
    ));
    // A target of the wrong kind, or one out of reach, fails.
    let events = scene.decide(ActionChoice::named("eat").targeting("pond_0"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionFailed { reason, .. } if reason.contains("not `edible`"))
    ));
    let events = scene.decide(ActionChoice::named("eat").targeting("berry_bush_5"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionFailed { reason, .. } if reason.contains("within"))
    ));
}

#[test]
fn egocentric_targeting_takes_the_nearest_qualifying_object() {
    let mut scene = Scene::masked();
    scene.place("berry_bush_3", 0.25 + 0.9 + 0.6, 0.0, 0.0);
    scene.place("berry_bush_1", 0.25 + 0.5 + 0.6, 0.2, 0.0);
    let events = scene.decide(ActionChoice::named("eat"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionStarted { target, .. } if target.as_deref() == Some("berry_bush_1"))
    ));
}

#[test]
fn drink_lowers_thirst_per_second_for_up_to_ten_seconds() {
    let mut scene = Scene::masked();
    scene.idle_ticks(12 * TICKS_PER_HOUR);
    let thirst = scene.need("thirst");
    assert!((thirst - 0.5).abs() < 1e-6);
    // The pond's radius is 3 m; its surface must be within 1 m.
    scene.place("pond_0", 0.25 + 0.5 + 3.0, 0.0, 0.0);
    assert!(scene.mask_of("drink") && !scene.mask_of("eat"));
    scene.decide(ActionChoice::named("drink"));
    scene.idle_ticks(18); // 2 s in all
    let per_second_rise = 1.0 / 1440.0;
    assert!(
        (scene.need("thirst") - (thirst - 0.2 + 2.0 * per_second_rise)).abs() < 1e-6,
        "{}",
        scene.need("thirst")
    );
    assert_eq!(scene.engine.current_action(AGENT).unwrap().action, "drink");
    let events = scene.idle_ticks(80);
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionCompleted { action } if action == "drink")
    ));
    assert!(
        scene.need("thirst") < 0.001,
        "thirst drained to 0 over the full 10 s"
    );
    // Interrupting stops the rate effect.
    scene.idle_ticks(2 * TICKS_PER_HOUR);
    let thirst = scene.need("thirst");
    scene.decide(ActionChoice::named("drink"));
    scene.decide(ActionChoice::named("turn").with("angle", 0.0));
    scene.idle_ticks(20);
    assert!(
        scene.need("thirst") > thirst - 0.03,
        "{}",
        scene.need("thirst")
    );
}

#[test]
fn sleep_lasts_until_fatigue_is_zero_or_the_agent_chooses_otherwise() {
    let mut scene = Scene::masked();
    scene.idle_ticks(6 * TICKS_PER_HOUR);
    assert!((scene.need("fatigue") - 0.25).abs() < 1e-6);
    let events = scene.decide(ActionChoice::named("sleep"));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionStarted { action, .. } if action == "sleep")
    ));
    assert!(scene.human().agent.unwrap().asleep);
    // Asleep, the agent still awaits an action, and every action is unmasked because choosing
    // one would wake it.
    assert_eq!(scene.world.agents_awaiting_action(), [AGENT]);
    assert!(scene.mask_of("move") && scene.mask_of("sleep"));
    // Fatigue falls at 4.0 per day: 0.25 takes 1.5 hours.
    let events = scene.idle_ticks((1.5 * TICKS_PER_HOUR as f64) as u64 + 2);
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionCompleted { action } if action == "sleep")
    ));
    let h = scene.human();
    assert!(
        h.need("fatigue").unwrap() < 0.001,
        "fatigue reached 0 and has just begun to rise again"
    );
    assert!(!h.agent.unwrap().asleep);
    // After two hours awake, sleeping again and then choosing to move wakes the agent and
    // moves it.
    scene.idle_ticks(2 * TICKS_PER_HOUR);
    scene.decide(ActionChoice::named("sleep"));
    scene.noop();
    let events = scene.decide(walk(0.0, 0.5));
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionInterrupted { action, by } if action == "sleep" && by == "move")
    ));
    let h = scene.human();
    assert!(!h.agent.as_ref().unwrap().asleep);
    assert!(h.x > 0.2);
}

#[test]
fn a_collapse_interrupts_the_action_and_is_reported() {
    let mut scene = Scene::new(&[(
        "population.0.overrides",
        serde_json::json!({"body": {"needs": {"thirst": {"rise_per_day": {"default": 0}}, "hunger": {"rise_per_day": {"default": 0}}, "cold": {"rise_per_day": {"default": 0}}}}}),
    )]);
    scene.place("pond_0", 0.25 + 0.5 + 3.0, 0.0, 0.0);
    scene.idle_ticks(24 * TICKS_PER_HOUR - 10);
    scene.decide(ActionChoice::named("drink"));
    let events = scene.idle_ticks(20);
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ActionInterrupted { action, by } if action == "drink" && by == "collapse")
    ));
    assert!(has_event(&events, |k| matches!(k, EventKind::Collapsed)));
    assert!(scene.world.agents_awaiting_action().is_empty());
    // The engine cannot be stepped with a choice for a collapsed agent; noop steps run.
    let empty = std::collections::BTreeMap::new();
    let mut events = Vec::new();
    while scene.tick() < 26 * TICKS_PER_HOUR + 2 {
        events.extend(scene.engine.step(&mut scene.world, &empty).unwrap());
    }
    assert!(has_event(&events, |k| matches!(
        k,
        EventKind::CollapseEnded
    )));
    assert_eq!(scene.world.agents_awaiting_action(), [AGENT]);
}

#[test]
fn death_is_reported_and_ends_the_agent() {
    let mut scene = Scene::masked();
    let mut events = Vec::new();
    while !scene.world.done() {
        events.extend(scene.noop());
    }
    let died = events
        .iter()
        .find(|e| matches!(e.kind, EventKind::Died))
        .unwrap();
    assert_eq!(died.agent.as_deref(), Some(AGENT));
    assert!(died.tick.abs_diff((22.9 * TICKS_PER_HOUR as f64) as u64) <= 2);
}
