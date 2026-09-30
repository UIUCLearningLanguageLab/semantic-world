//! Acceptance (stage 3): a headless run of 10 days with `noop` agents shows every need
//! following its configured rate, and the agent dying on the first night when cold drains
//! health. Cold reaches 1.0 about 21.9 calendar hours into the run, and death follows 1 hour
//! later.

mod common;

use common::{TICKS_PER_DAY, TICKS_PER_HOUR, run_to};

#[test]
fn needs_follow_their_rates_and_cold_kills_on_the_first_night() {
    let mut world = common::world(0, 10.0);
    assert_eq!(world.agent_ids(), ["human_0"]);
    let human = |w: &sw_core::World| w.snapshot().entity("human_0").unwrap().clone();

    // Hour 6: hunger 0.5/day, thirst 1.0/day, fatigue 1.0/day while awake, cold 0 by day.
    run_to(&mut world, 6 * TICKS_PER_HOUR);
    let h = human(&world);
    assert!((h.need("hunger").unwrap() - 0.125).abs() < 1e-6);
    assert!((h.need("thirst").unwrap() - 0.25).abs() < 1e-6);
    assert!((h.need("fatigue").unwrap() - 0.25).abs() < 1e-6);
    assert_eq!(h.need("cold").unwrap(), 0.0);
    assert_eq!(h.health, Some(1.0));
    assert!(h.agent.as_ref().unwrap().alive);

    // Hour 14: still daylight, cold 0.
    run_to(&mut world, 14 * TICKS_PER_HOUR);
    assert_eq!(human(&world).need("cold").unwrap(), 0.0);
    assert!(!world.snapshot().night);

    // Hour 18: night began at 14.4 h; cold rises 4.0/day × (1 − 0.2) = 0.1333/hour.
    run_to(&mut world, 18 * TICKS_PER_HOUR);
    let h = human(&world);
    assert!(world.snapshot().night);
    let expected_cold = (18.0 - 14.4) * 4.0 * 0.8 / 24.0;
    assert!(
        (h.need("cold").unwrap() - expected_cold).abs() < 1e-3,
        "{}",
        h.need("cold").unwrap()
    );
    assert_eq!(h.health, Some(1.0));

    // Cold reaches 1.0 at 14.4 + 7.5 = 21.9 hours.
    let cold_max_tick = loop {
        world.tick();
        if human(&world).need("cold").unwrap() >= 1.0 {
            break world.clock().tick;
        }
        assert!(
            world.clock().tick < 23 * TICKS_PER_HOUR,
            "cold never reached 1.0"
        );
    };
    let expected = (21.9 * TICKS_PER_HOUR as f64) as u64;
    assert!(
        cold_max_tick.abs_diff(expected) <= 2,
        "cold at max at tick {cold_max_tick}, expected about {expected}"
    );

    // Health drains 1.0 per hour from cold at max; death follows one hour later.
    let death_tick = loop {
        world.tick();
        let h = human(&world);
        if !h.agent.as_ref().unwrap().alive {
            break world.clock().tick;
        }
        assert!(
            world.clock().tick < 24 * TICKS_PER_HOUR,
            "the agent did not die"
        );
    };
    assert!(
        death_tick.abs_diff(cold_max_tick + TICKS_PER_HOUR) <= 2,
        "death at {death_tick}"
    );
    let h = human(&world);
    assert_eq!(h.health, Some(0.0));
    assert_eq!(h.agent.as_ref().unwrap().died_at_tick, Some(death_tick - 1));
    assert!(
        h.need("thirst").unwrap() < 1.0,
        "thirst had not yet reached 1.0"
    );
    assert!(world.done(), "the only agent is dead");
    assert!(world.agents_awaiting_action().is_empty());

    // The run can still be advanced to 10 days; the dead agent's state stays frozen.
    let frozen = human(&world);
    run_to(&mut world, 10 * TICKS_PER_DAY);
    assert_eq!(world.clock().day(), 10);
    assert_eq!(human(&world), frozen);
    assert!(world.clock().lifetime_over());
}

#[test]
fn the_world_has_the_specified_population() {
    let world = common::world(1, 1.0);
    let snapshot = world.snapshot();
    let count = |t: &str| snapshot.of_type(t).count();
    assert_eq!(count("human"), 1);
    assert_eq!(count("berry_bush"), 20);
    assert_eq!(count("pond"), 3);
    assert_eq!(count("shelter"), 2);
    assert_eq!(count("tree"), 30);
    assert_eq!(snapshot.entities.len(), 56);
    // IDs are the type name plus an index counting from 0 within the type.
    let bushes: Vec<&str> = snapshot
        .of_type("berry_bush")
        .map(|e| e.id.as_str())
        .collect();
    assert_eq!(bushes[0], "berry_bush_0");
    assert_eq!(bushes[19], "berry_bush_19");
    assert_eq!(
        snapshot.entity("berry_bush_3").unwrap().stock("berries"),
        Some(3)
    );
    // Snapshot order is entity-ID order: by type name, then index.
    let order: Vec<(&str, u32)> = snapshot
        .entities
        .iter()
        .map(|e| (e.type_name.as_str(), e.index))
        .collect();
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(order, sorted);
    // The creation order lists objects first, then agents.
    let ids = world.entity_ids();
    assert_eq!(ids[0], "berry_bush_0");
    assert_eq!(ids[55], "human_0");
}
