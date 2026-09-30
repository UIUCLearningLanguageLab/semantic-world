//! Acceptance (stage 4): a berry bush regrows on schedule, and a masked action never appears
//! as available when its precondition fails.

mod common;

use common::{AGENT, Scene, has_event, walk};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sw_rules::{ActionChoice, EventKind};

#[test]
fn a_berry_bush_regrows_every_ten_calendar_hours() {
    let mut scene = Scene::masked();
    scene.place("berry_bush_0", 0.25 + 0.8 + 0.6, 0.0, 0.0);
    scene.decide(ActionChoice::named("eat"));
    scene.idle_ticks(18);
    assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(2));
    // 600 simulated seconds is 6,000 ticks; nothing before, one berry at the boundary.
    let events = scene.idle_ticks(6000 - scene.tick() - 2);
    assert!(!has_event(&events, |k| matches!(
        k,
        EventKind::ProcessFired { .. }
    )));
    assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(2));
    let events = scene.idle_ticks(2);
    assert!(has_event(
        &events,
        |k| matches!(k, EventKind::ProcessFired { process, entity } if process == "regrow" && entity == "berry_bush_0")
    ));
    assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(3));
    // Regrowth never passes the maximum.
    scene.idle_ticks(6000);
    assert_eq!(scene.entity("berry_bush_0").stock("berries"), Some(3));
    assert_eq!(scene.entity("berry_bush_7").stock("berries"), Some(3));
}

#[test]
fn the_mask_agrees_with_what_actually_succeeds() {
    // In attempt-and-fail mode, a random policy tries everything. Whenever an action fails,
    // the mask computed just before must have been false for it; whenever it starts, true.
    let mut scene = Scene::attempt_and_fail();
    // A cluster of objects around the agent so preconditions flip often.
    scene.place("berry_bush_0", 2.0, 0.0, 0.0);
    scene.place("berry_bush_1", -1.5, 1.5, 0.0);
    scene.place("pond_0", 0.0, -5.0, 0.0);
    scene.place("tree_0", 1.0, 3.0, 0.0);
    let manifest = scene.engine.manifest(AGENT).unwrap();
    let mut rng = ChaCha8Rng::from_seed([3; 32]);
    let mut failures = 0;
    let mut starts = 0;
    for _ in 0..3000 {
        if scene.world.done() {
            break;
        }
        if scene.world.agents_awaiting_action().is_empty() {
            let empty = std::collections::BTreeMap::new();
            scene.engine.step(&mut scene.world, &empty).unwrap();
            continue;
        }
        let mask = scene.mask();
        let i = rng.random_range(0..manifest.actions.len());
        let name = manifest.actions[i].name.as_str();
        let choice = match name {
            "move" => walk(rng.random_range(-3.1..3.1), rng.random_range(0.0..1.0)),
            "turn" => ActionChoice::named("turn").with("angle", rng.random_range(-0.78..0.78)),
            other => ActionChoice::named(other),
        };
        let events = scene.decide(choice);
        let failed = events
            .iter()
            .any(|e| matches!(&e.kind, EventKind::ActionFailed { action, .. } if action == name));
        let started = events
            .iter()
            .any(|e| matches!(&e.kind, EventKind::ActionStarted { action, .. } if action == name));
        if failed {
            failures += 1;
            assert!(!mask[i], "{name} failed but was unmasked: {events:?}");
        }
        if started {
            starts += 1;
            assert!(mask[i], "{name} started but was masked");
        }
        assert!(failed != started || name == "noop");
    }
    assert!(
        failures > 50 && starts > 50,
        "failures {failures}, starts {starts}"
    );
}

#[test]
fn every_masked_precondition_failure_is_reflected_in_the_mask() {
    let mut scene = Scene::masked();
    let manifest = scene.engine.manifest(AGENT).unwrap();
    // Nothing near: only noop, move, turn, and sleep are available.
    let mask = scene.mask();
    let available: Vec<&str> = manifest
        .actions
        .iter()
        .zip(&mask)
        .filter(|(_, m)| **m)
        .map(|(a, _)| a.name.as_str())
        .collect();
    assert_eq!(available, ["noop", "move", "turn", "sleep"]);
    // Eat becomes available only with a stocked, edible target in front and within reach.
    scene.place("berry_bush_0", 0.25 + 0.8 + 0.6, 0.0, 0.0);
    assert!(scene.mask_of("eat"));
    scene.place(AGENT, 0.0, 0.0, 1.0); // 57° off: outside the 45° cone
    assert!(!scene.mask_of("eat"));
    scene.place(AGENT, 0.0, 0.0, 0.7); // 40°: inside
    assert!(scene.mask_of("eat"));
}
