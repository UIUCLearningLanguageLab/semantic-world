//! Kinematic movement, blocking, shelters, and the agent states the needs depend on.

mod common;

use common::{TICKS_PER_DAY, TICKS_PER_HOUR, run_to};
use serde_json::json;
use sw_core::World;

fn human(w: &World) -> sw_core::EntitySnapshot {
    w.snapshot().entity("human_0").unwrap().clone()
}

/// A world with the agent standing alone at the origin, facing +x, with every object moved
/// far away except the ones a test puts back.
fn clear_world(seed: u64) -> World {
    let mut world = common::world(seed, 3.0);
    let ids = world.entity_ids();
    for (i, id) in ids.iter().enumerate() {
        if id != "human_0" {
            world
                .privileged_set_pose(
                    id,
                    -45.0 + (i as f64 % 20.0) * 4.5,
                    45.0 - (i as f64 / 20.0).floor() * 4.5,
                    0.0,
                )
                .unwrap();
        }
    }
    world.privileged_set_pose("human_0", 0.0, 0.0, 0.0).unwrap();
    world
}

#[test]
fn walking_covers_the_expected_distance() {
    let mut world = clear_world(0);
    // 1.4 m/s for one decision interval of 0.2 s.
    world.set_motion("human_0", 0.0, 1.4, false).unwrap();
    world.advance_decision();
    let h = human(&world);
    assert!((h.x - 0.28).abs() < 1e-9, "{}", h.x);
    assert_eq!(h.z, 0.0);
    assert!(!h.agent.as_ref().unwrap().blocked);
    // Motion lasts one decision interval only.
    world.advance_decision();
    assert!((human(&world).x - 0.28).abs() < 1e-9);
    // Turning changes the heading; the yaw is wrapped.
    world
        .set_motion("human_0", std::f64::consts::FRAC_PI_2, 3.5, true)
        .unwrap();
    world.advance_decision();
    let h = human(&world);
    assert!((h.z - 0.7).abs() < 1e-9);
    assert!((h.yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    world.set_facing("human_0", 7.0).unwrap();
    assert!((human(&world).yaw - (7.0 - std::f64::consts::TAU)).abs() < 1e-12);
}

#[test]
fn a_tree_blocks_and_touch_reports_it() {
    let mut world = clear_world(0);
    world.privileged_set_pose("tree_0", 2.0, 0.0, 0.0).unwrap();
    for _ in 0..20 {
        world.set_motion("human_0", 0.0, 3.5, true).unwrap();
        world.advance_decision();
    }
    let h = human(&world);
    let a = h.agent.as_ref().unwrap();
    assert!(
        (h.x - (2.0 - 0.3 - 0.25)).abs() < 1e-4,
        "stopped at contact: {}",
        h.x
    );
    assert!(a.blocked);
    assert!((a.contact_direction[0] - 1.0).abs() < 1e-6);
    // Walking away clears the contact.
    world
        .set_motion("human_0", std::f64::consts::PI, 1.4, false)
        .unwrap();
    world.advance_decision();
    assert!(!human(&world).agent.as_ref().unwrap().blocked);
}

#[test]
fn the_meadow_edge_blocks() {
    let mut world = clear_world(0);
    world
        .privileged_set_pose("human_0", 48.0, 0.0, 0.0)
        .unwrap();
    for _ in 0..20 {
        world.set_motion("human_0", 0.0, 3.5, true).unwrap();
        world.advance_decision();
    }
    let h = human(&world);
    assert!((h.x - 49.75).abs() < 1e-4, "{}", h.x);
    assert!(h.agent.as_ref().unwrap().blocked);
}

#[test]
fn a_pond_does_not_block() {
    let mut world = clear_world(0);
    world.privileged_set_pose("pond_0", 2.0, 0.0, 0.0).unwrap();
    for _ in 0..10 {
        world.set_motion("human_0", 0.0, 3.5, true).unwrap();
        world.advance_decision();
    }
    let h = human(&world);
    assert!((h.x - 7.0).abs() < 1e-6, "{}", h.x);
    assert!(!h.agent.as_ref().unwrap().blocked);
}

#[test]
fn inside_a_shelter_cold_does_not_rise_at_night() {
    let mut world = clear_world(0);
    // Shelter at (10, 0) facing +x: open toward +x. Walk in through the open side.
    world
        .privileged_set_pose("shelter_0", 10.0, 0.0, 0.0)
        .unwrap();
    world
        .privileged_set_pose("human_0", 14.0, 0.0, std::f64::consts::PI)
        .unwrap();
    for _ in 0..15 {
        world
            .set_motion("human_0", std::f64::consts::PI, 1.4, false)
            .unwrap();
        world.advance_decision();
    }
    let h = human(&world);
    assert!(h.agent.as_ref().unwrap().in_shelter, "{h:?}");
    assert!(h.x > 10.0 - 1.3, "stopped by the back wall: {}", h.x);
    // Through the night, cold stays at 0 and the agent survives.
    run_to(&mut world, TICKS_PER_DAY);
    let h = human(&world);
    assert_eq!(h.need("cold").unwrap(), 0.0);
    assert!(h.agent.as_ref().unwrap().alive);
    assert!(h.health.unwrap() > 0.9);
}

#[test]
fn fatigue_at_max_collapses_the_agent_for_two_hours() {
    // Keep thirst and hunger from rising so fatigue is the only need that reaches max.
    let resolved = common::smoke(&[
        ("lifetime.default_days", json!(3)),
        (
            "population.0.overrides",
            json!({"body": {"needs": {"thirst": {"rise_per_day": {"default": 0}}, "hunger": {"rise_per_day": {"default": 0}}}}}),
        ),
    ]);
    let mut world = World::new(&resolved, 0).unwrap();
    let ids = world.entity_ids();
    for (i, id) in ids.iter().enumerate() {
        if id != "human_0" {
            world
                .privileged_set_pose(
                    id,
                    -45.0 + (i as f64 % 20.0) * 4.5,
                    45.0 - (i as f64 / 20.0).floor() * 4.5,
                    0.0,
                )
                .unwrap();
        }
    }
    world
        .privileged_set_pose("shelter_0", 0.0, 0.0, 0.0)
        .unwrap();
    world.privileged_set_pose("human_0", 0.0, 0.0, 0.0).unwrap();
    // Fatigue rises 1.0/day while awake: at max at 24 hours.
    run_to(&mut world, TICKS_PER_DAY - 1);
    let h = human(&world);
    assert!(!h.agent.as_ref().unwrap().asleep);
    assert!(
        world.agents_awaiting_action().is_empty(),
        "not a decision point"
    );
    run_to(&mut world, TICKS_PER_DAY + 2);
    let h = human(&world);
    let a = h.agent.as_ref().unwrap();
    assert!(a.asleep && a.collapsed_until_tick.is_some(), "{a:?}");
    assert_eq!(
        a.collapsed_until_tick,
        Some(TICKS_PER_DAY - 1 + 2 * TICKS_PER_HOUR)
    );
    assert!(
        world.agents_awaiting_action().is_empty(),
        "a collapsed agent is not asked to act"
    );
    assert!(
        world.set_motion("human_0", 0.0, 1.0, false).is_err(),
        "a collapsed agent cannot move"
    );
    // Asleep in a shelter, fatigue falls 4.0/day × 2. After two hours the collapse ends.
    run_to(&mut world, TICKS_PER_DAY + 2 * TICKS_PER_HOUR + 2);
    let h = human(&world);
    let a = h.agent.as_ref().unwrap();
    assert!(!a.asleep && a.collapsed_until_tick.is_none(), "{a:?}");
    assert!(
        (h.need("fatigue").unwrap() - (1.0 - 8.0 * 2.0 / 24.0)).abs() < 1e-3,
        "{}",
        h.need("fatigue").unwrap()
    );
    assert_eq!(world.agents_awaiting_action(), ["human_0"]);
    assert!(a.alive);
}

#[test]
fn voluntary_sleep_lets_fatigue_fall_and_wakes_on_demand() {
    let mut world = clear_world(0);
    run_to(&mut world, 12 * TICKS_PER_HOUR);
    assert!((human(&world).need("fatigue").unwrap() - 0.5).abs() < 1e-6);
    world.set_asleep("human_0", true).unwrap();
    assert!(world.set_motion("human_0", 0.0, 1.0, false).is_err());
    run_to(&mut world, 13 * TICKS_PER_HOUR);
    let h = human(&world);
    assert!(
        (h.need("fatigue").unwrap() - (0.5 - 4.0 / 24.0)).abs() < 1e-6,
        "{}",
        h.need("fatigue").unwrap()
    );
    assert_eq!(
        world.agents_awaiting_action(),
        ["human_0"],
        "a sleeping agent is still asked, so it can wake"
    );
    world.set_asleep("human_0", false).unwrap();
    run_to(&mut world, 14 * TICKS_PER_HOUR);
    assert!(
        (human(&world).need("fatigue").unwrap() - (0.5 - 4.0 / 24.0 + 1.0 / 24.0)).abs() < 1e-6
    );
}

#[test]
fn agent_seed_is_the_agent_stream() {
    let world = common::world(7, 1.0);
    let seed = world.agent_seed("human_0").unwrap();
    assert_eq!(seed, sw_core::streams::stream_seed(7, "agent:human_0"));
    assert!(world.agent_seed("tree_0").is_err());
    assert!(world.agent_seed("nobody").is_err());
}
