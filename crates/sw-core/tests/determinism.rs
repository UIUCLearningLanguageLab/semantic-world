//! Acceptance (stage 3): two runs with the same seed give identical state hashes at every day
//! boundary; two runs with different seeds differ.

mod common;

use common::{TICKS_PER_DAY, run_to};
use sw_core::{StateHash, World};

/// A fixed walking policy: at every decision point, walk toward a heading that turns with the
/// tick, so the agent covers ground and hits obstacles.
fn walk(world: &mut World) {
    if world.clock().is_decision_point() {
        for id in world.agents_awaiting_action() {
            let yaw = (world.clock().tick as f64) * 0.001;
            let running = world.clock().tick % 40 < 10;
            let speed = if running { 3.5 } else { 1.4 };
            world.set_motion(&id, yaw, speed, running).unwrap();
        }
    }
}

fn hashes_at_day_boundaries(seed: u64, days: u64) -> Vec<StateHash> {
    let mut world = common::world(seed, days as f64);
    let mut hashes = vec![world.state_hash()];
    for day in 1..=days {
        while world.clock().tick < day * TICKS_PER_DAY {
            walk(&mut world);
            world.tick();
        }
        hashes.push(world.state_hash());
    }
    hashes
}

#[test]
fn same_seed_same_hashes_at_every_day_boundary() {
    let a = hashes_at_day_boundaries(42, 3);
    let b = hashes_at_day_boundaries(42, 3);
    assert_eq!(a, b);
    assert_eq!(a.len(), 4);
    assert_ne!(a[0], a[1], "the state changes over a day");
}

#[test]
fn different_seeds_differ() {
    let a = hashes_at_day_boundaries(42, 1);
    let b = hashes_at_day_boundaries(43, 1);
    assert_ne!(a[0], b[0], "the map differs from tick 0");
    assert_ne!(a[1], b[1]);
}

#[test]
fn the_hash_covers_every_component_and_the_tick() {
    let mut world = common::world(5, 1.0);
    let h0 = world.state_hash();
    world.tick();
    assert_ne!(h0, world.state_hash(), "the tick number is covered");
    let mut other = common::world(5, 1.0);
    run_to(&mut other, 1);
    assert_eq!(world.state_hash(), other.state_hash());
    other.privileged_set_pose("tree_0", 1.0, 2.0, 0.0).unwrap();
    assert_ne!(
        world.state_hash(),
        other.state_hash(),
        "positions are covered"
    );
    let mut third = common::world(5, 1.0);
    run_to(&mut third, 1);
    third.set_asleep("human_0", true).unwrap();
    assert_ne!(
        world.state_hash(),
        third.state_hash(),
        "agent state is covered"
    );
    assert_eq!(world.state_hash().to_hex().len(), 64);
}

#[test]
fn individuals_are_sampled_from_the_traits_stream() {
    let a = common::world(9, 1.0);
    let b = common::world(9, 1.0);
    let c = common::world(10, 1.0);
    let mass = |w: &World| {
        *w.individual("human_0")
            .unwrap()
            .entity_type
            .body
            .mass
            .as_ref()
            .unwrap()
            .value()
    };
    assert_eq!(mass(&a), mass(&b));
    assert_ne!(mass(&a), mass(&c));
    assert_eq!(a.individual("human_0").unwrap().sampled.len(), 2);
    assert!(a.individual("tree_0").unwrap().sampled.is_empty());
}
