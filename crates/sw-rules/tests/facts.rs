//! Detectors: the facts the propositional sensor reports.

mod common;

use common::{AGENT, Scene, TICKS_PER_HOUR};
use serde_json::Value;

fn facts(scene: &Scene) -> Vec<String> {
    scene
        .engine
        .facts(&scene.world, AGENT)
        .unwrap()
        .iter()
        .map(|f| f.to_string())
        .collect()
}

#[test]
fn facts_are_empty_when_the_sensor_is_off() {
    let scene = Scene::masked();
    assert!(facts(&scene).is_empty());
}

#[test]
fn facts_describe_the_agent_and_nearby_objects() {
    let mut scene = Scene::new(&[("views.propositional_sensor", Value::from(true))]);
    scene.place("berry_bush_0", 3.0, 0.0, 0.0); // gap 2.15 m: near
    scene.place("berry_bush_1", 15.0, 0.0, 0.0); // in range (20 m), visible, not near
    scene.place("tree_0", 30.0, 0.0, 0.0); // out of range
    scene.place("pond_0", -8.0, 0.0, 0.0); // behind: not visible
    let f = facts(&scene);
    assert!(
        f.contains(&"(near human_0 berry_bush_0)".to_string()),
        "{f:?}"
    );
    assert!(!f.contains(&"(near human_0 berry_bush_1)".to_string()));
    assert!(f.contains(&"(visible human_0 berry_bush_0)".to_string()));
    assert!(f.contains(&"(visible human_0 berry_bush_1)".to_string()));
    assert!(!f.contains(&"(visible human_0 pond_0)".to_string()));
    assert!(!f.iter().any(|s| s.contains("tree_0")));
    assert!(f.contains(&"(is human_0 human)".to_string()));
    assert!(f.contains(&"(is berry_bush_0 berry_bush)".to_string()));
    assert!(f.contains(&"(is pond_0 pond)".to_string()));
    assert!(f.contains(&"(berries berry_bush_0 3)".to_string()));
    assert!(f.contains(&"(hunger human_0 0.00)".to_string()));
    assert!(f.contains(&"(health human_0 1.00)".to_string()));
    assert!(!f.contains(&"(night)".to_string()));
    assert!(
        !f.iter()
            .any(|s| s.starts_with("(asleep") || s.starts_with("(inside"))
    );
    // Vocabulary order: predicates in the order the rules file declares them.
    let first_near = f.iter().position(|s| s.starts_with("(near")).unwrap();
    let first_is = f.iter().position(|s| s.starts_with("(is ")).unwrap();
    assert!(first_near < first_is);
}

#[test]
fn a_trunk_or_a_wall_blocks_the_line_of_sight_but_a_bush_does_not() {
    let mut scene = Scene::new(&[("views.propositional_sensor", Value::from(true))]);
    scene.place("berry_bush_0", 10.0, 0.0, 0.0);
    scene.place("tree_0", 5.0, 0.0, 0.0);
    let f = facts(&scene);
    assert!(f.contains(&"(visible human_0 tree_0)".to_string()));
    assert!(
        !f.contains(&"(visible human_0 berry_bush_0)".to_string()),
        "{f:?}"
    );
    scene.place("tree_0", 5.0, 2.0, 0.0);
    assert!(facts(&scene).contains(&"(visible human_0 berry_bush_0)".to_string()));
    // A bush in the way is lower than the camera and does not occlude.
    scene.place("berry_bush_1", 5.0, 0.0, 0.0);
    assert!(facts(&scene).contains(&"(visible human_0 berry_bush_0)".to_string()));
    // A shelter wall occludes.
    scene.place("berry_bush_1", 5.0, 3.0, 0.0);
    scene.place("shelter_0", 5.0, 0.0, std::f64::consts::FRAC_PI_2);
    assert!(!facts(&scene).contains(&"(visible human_0 berry_bush_0)".to_string()));
}

#[test]
fn state_facts_follow_the_agent() {
    let mut scene = Scene::new(&[("views.propositional_sensor", Value::from(true))]);
    scene.place("shelter_0", 0.0, 0.0, 0.0);
    scene.idle_ticks(15 * TICKS_PER_HOUR);
    scene.decide(sw_rules::ActionChoice::named("sleep"));
    let f = facts(&scene);
    assert!(
        f.contains(&"(inside human_0 shelter_0)".to_string()),
        "{f:?}"
    );
    assert!(f.contains(&"(asleep human_0)".to_string()));
    assert!(f.contains(&"(night)".to_string()));
    assert!(
        f.contains(&"(fatigue human_0 0.62)".to_string())
            || f.iter().any(|s| s.starts_with("(fatigue human_0 0.6")),
        "{f:?}"
    );
}
