//! Map generation follows the placement rules of the meadow.

mod common;

use glam::DVec2;

#[test]
fn placements_respect_gaps_margins_and_clearance() {
    for seed in 0..5 {
        let world = common::world(seed, 1.0);
        let snapshot = world.snapshot();
        let objects: Vec<DVec2> = snapshot
            .entities
            .iter()
            .filter(|e| e.agent.is_none())
            .map(|e| DVec2::new(e.x, e.z))
            .collect();
        assert_eq!(objects.len(), 55);
        for (i, a) in objects.iter().enumerate() {
            assert!(
                a.x.abs() <= 47.0 && a.y.abs() <= 47.0,
                "object {i} is within 3 m of the edge"
            );
            for b in &objects[i + 1..] {
                assert!(a.distance(*b) >= 4.0, "objects closer than 4 m: {a} {b}");
            }
        }
        let human = snapshot.entity("human_0").unwrap();
        let p = DVec2::new(human.x, human.z);
        assert!(p.x.abs() <= 47.0 && p.y.abs() <= 47.0);
        for o in &objects {
            assert!(
                p.distance(*o) >= 5.0,
                "the agent starts within 5 m of an object"
            );
        }
        assert!(human.yaw > -std::f64::consts::PI && human.yaw <= std::f64::consts::PI);
    }
}

#[test]
fn shelters_get_a_rotation_and_a_footprint() {
    let world = common::world(3, 1.0);
    let snapshot = world.snapshot();
    let shelters: Vec<_> = snapshot.of_type("shelter").collect();
    assert_eq!(shelters.len(), 2);
    assert_ne!(shelters[0].yaw, shelters[1].yaw);
    let s = shelters[0];
    assert!(world.inside_any_walled(DVec2::new(s.x, s.z)));
    assert!(!world.inside_any_walled(DVec2::new(s.x + 10.0, s.z)));
}

#[test]
fn different_seeds_give_different_maps() {
    let a = common::world(1, 1.0).snapshot();
    let b = common::world(2, 1.0).snapshot();
    let pos =
        |s: &sw_core::StateSnapshot| (s.entity("tree_0").unwrap().x, s.entity("tree_0").unwrap().z);
    assert_ne!(pos(&a), pos(&b));
}

#[test]
fn an_unknown_generator_is_an_error() {
    let resolved = common::smoke(&[(
        "world",
        serde_json::json!({
            "world": "odd", "map": {"generator": "volcano", "size_m": [10, 10]},
            "calendar": {"minutes_per_day": 24, "daylight_fraction": 0.6},
            "light": {"twilight_hours": 1, "night_below": 0.5},
            "types": "../types", "rules": []
        }),
    )]);
    let err = sw_core::World::new(&resolved, 0).unwrap_err();
    assert!(
        err.to_string().contains("unknown map generator `volcano`"),
        "{err}"
    );
}
