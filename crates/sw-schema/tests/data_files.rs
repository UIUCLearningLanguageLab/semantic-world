//! Acceptance: all files in `data/` load and resolve, and the loaded content matches the
//! numbers in `docs/specs/MILESTONE_1.md`.

mod common;

use serde_json::json;
use sw_schema::entity::{AgentState, AtMax, Collision, Placeholder, RiseWhen, Shape};
use sw_schema::experiment::{ImpossibleActions, WorldRef};
use sw_schema::{load_experiment, load_experiment_with_overrides};

#[test]
fn both_experiments_resolve() {
    for name in ["m1_smoke", "m1_validation"] {
        let path = common::repo_root().join(format!("data/experiments/{name}.yaml"));
        let resolved = load_experiment(&path).unwrap();
        assert_eq!(resolved.experiment.experiment, name);
        assert!(matches!(resolved.experiment.world, WorldRef::Path(_)));
        assert_eq!(resolved.world.world.as_deref(), Some("meadow"));
        let types: Vec<&str> = resolved.types.keys().map(String::as_str).collect();
        assert_eq!(types, ["human", "berry_bush", "pond", "shelter", "tree"]);
    }
}

#[test]
fn every_type_file_loads_on_its_own() {
    let dir = common::repo_root().join("data/types");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "berry_bush.yaml",
            "human.yaml",
            "pond.yaml",
            "shelter.yaml",
            "tree.yaml"
        ]
    );
    for name in names {
        let path = dir.join(&name);
        let t: sw_schema::entity::EntityType = sw_schema::yaml::read(&path).unwrap();
        t.validate(&sw_schema::Ctx::new(&path)).unwrap();
        assert_eq!(format!("{}.yaml", t.name), name);
    }
}

#[test]
fn the_world_matches_the_specification() {
    let resolved =
        load_experiment(&common::repo_root().join("data/experiments/m1_smoke.yaml")).unwrap();
    // Time.
    assert_eq!(resolved.calendar.seconds_per_day(), 1440.0);
    assert_eq!(resolved.calendar.seconds_per_hour(), 60.0);
    assert_eq!(resolved.calendar.daylight_fraction, 0.6);
    assert_eq!(resolved.calendar.twilight_hours, 1.0);
    assert_eq!(resolved.calendar.night_below, 0.5);
    // Space.
    assert_eq!(resolved.world.map.size_m, [100.0, 100.0]);
    assert_eq!(
        resolved.world.map.params["objects"],
        json!({"berry_bush": 20, "pond": 3, "shelter": 2, "tree": 30})
    );
    // Entities.
    let human = &resolved.types["human"].body;
    assert_eq!(human.radius_m(), 0.25);
    assert_eq!(human.insulation_value(), 0.2);
    assert!(human.solid);
    let bush = &resolved.types["berry_bush"].body;
    assert_eq!(bush.radius_m(), 0.6);
    assert!(bush.has_tag("edible") && bush.has_tag("regrows"));
    assert_eq!(*bush.stock["berries"].initial.value(), 3);
    assert_eq!(*bush.stock["berries"].max.value(), 3);
    assert_eq!(*bush.provides["hunger"].value(), -0.25);
    match bush.placeholder.as_ref().unwrap() {
        Placeholder::Single(part) => {
            assert_eq!(part.shape, Shape::Sphere);
            assert_eq!(part.color_when_stocked.as_ref().unwrap().stock, "berries");
        }
        other => panic!("{other:?}"),
    }
    let pond = &resolved.types["pond"].body;
    assert_eq!(pond.radius_m(), 3.0);
    assert!(!pond.solid && pond.has_tag("drinkable"));
    let shelter = &resolved.types["shelter"].body;
    match &shelter.collision {
        Collision::Walls { thickness } => assert_eq!(*thickness.value(), 0.2),
        other => panic!("{other:?}"),
    }
    let tree = &resolved.types["tree"].body;
    assert_eq!(tree.radius_m(), 0.3);
    let tree_parts = tree.placeholder.as_ref().unwrap().parts();
    assert_eq!(tree_parts.len(), 2);
    assert_eq!(tree_parts[0].center(), [0.0, 1.5, 0.0]);
    assert_eq!(tree_parts[1].center(), [0.0, 4.25, 0.0]);
    // Needs.
    let needs = &human.needs;
    let names: Vec<&str> = needs.keys().map(String::as_str).collect();
    assert_eq!(names, ["hunger", "thirst", "fatigue", "cold"]);
    assert_eq!(*needs["hunger"].rise_per_day.value(), 0.5);
    assert!(matches!(&needs["hunger"].at_max, AtMax::HealthDrainPerHour(t) if *t.value() == 0.5));
    assert_eq!(*needs["thirst"].rise_per_day.value(), 1.0);
    assert!(matches!(&needs["thirst"].at_max, AtMax::HealthDrainPerHour(t) if *t.value() == 1.0));
    assert_eq!(*needs["fatigue"].rise_per_day.value(), 1.0);
    assert_eq!(*needs["fatigue"].rise_when.value(), RiseWhen::Awake);
    assert_eq!(
        *needs["fatigue"].rise_multipliers[&AgentState::Running].value(),
        2.0
    );
    assert_eq!(*needs["fatigue"].fall_per_day.value(), 4.0);
    assert_eq!(
        *needs["fatigue"].fall_multipliers[&AgentState::InShelter].value(),
        2.0
    );
    assert!(
        matches!(&needs["fatigue"].at_max, AtMax::Collapse { duration_hours } if *duration_hours.value() == 2.0)
    );
    assert_eq!(*needs["cold"].rise_per_day.value(), 4.0);
    assert_eq!(
        *needs["cold"].rise_when.value(),
        RiseWhen::NightOutsideShelter
    );
    assert!(*needs["cold"].insulation_reduces_rise.value());
    assert_eq!(*needs["cold"].fall_per_day.value(), 4.0);
    assert_eq!(
        *human.health.as_ref().unwrap().recover_per_hour.value(),
        0.05
    );
    // Rules.
    let actions: Vec<&str> = resolved.rules.actions.keys().map(String::as_str).collect();
    assert_eq!(actions, ["noop", "move", "turn", "eat", "drink", "sleep"]);
    let eat = &resolved.rules.actions["eat"];
    assert!(eat.is_durative());
    assert_eq!(*eat.duration.as_ref().unwrap().value(), 2.0);
    assert_eq!(eat.pre.len(), 5);
    assert_eq!(eat.effects[1].name, "change_need");
    assert!(!resolved.rules.actions["move"].is_durative());
    assert_eq!(
        resolved.rules.actions["move"].module.as_deref(),
        Some("kinematic_move")
    );
    assert!(resolved.rules.actions["sleep"].is_durative());
    assert!(resolved.rules.actions["sleep"].duration.is_none());
    assert_eq!(*resolved.rules.processes["regrow"].every.value(), 600.0);
    assert_eq!(resolved.rules.predicates.len(), 13);
    assert_eq!(
        resolved.rules.predicates["near"].params["max_gap_m"],
        json!(3.0)
    );
    // Sensors.
    let sensors = &resolved.types["human"].sensors;
    assert_eq!(
        sensors.present(),
        [
            "eyes",
            "interoception",
            "touch",
            "proprioception",
            "propositional"
        ]
    );
    assert_eq!(sensors.eyes.as_ref().unwrap().resolution, [64, 64]);
    assert_eq!(*sensors.eyes.as_ref().unwrap().fov_deg.value(), 90.0);
    assert_eq!(
        *sensors.propositional.as_ref().unwrap().range_m.value(),
        20.0
    );
}

#[test]
fn validation_conditions_cross_with_seeds() {
    let resolved =
        load_experiment(&common::repo_root().join("data/experiments/m1_validation.yaml")).unwrap();
    let conditions = &resolved.experiment.conditions;
    assert_eq!(conditions.len(), 2);
    assert_eq!(
        conditions["population.0.mind.module"],
        vec![json!("random"), json!("scripted_optimal")]
    );
    assert_eq!(resolved.experiment.master_seeds().len(), 10);
    assert_eq!(resolved.experiment.lifetime.default_days, 30.0);
    assert_eq!(
        resolved.experiment.options.impossible_actions,
        ImpossibleActions::Masked
    );
}

#[test]
fn overrides_apply_before_resolution() {
    let path = common::repo_root().join("data/experiments/m1_smoke.yaml");
    let resolved = load_experiment_with_overrides(
        &path,
        &[
            ("seeds".into(), json!([3])),
            ("population.0.count".into(), json!(4)),
            (
                "options.impossible_actions".into(),
                json!("attempt_and_fail"),
            ),
        ],
    )
    .unwrap();
    assert_eq!(resolved.experiment.master_seeds(), [3]);
    assert_eq!(resolved.population[0].count, 4);
    assert_eq!(
        resolved.experiment.options.impossible_actions,
        ImpossibleActions::AttemptAndFail
    );
}

#[test]
fn derived_types_extend_their_base() {
    let data = common::DataCopy::new("extends");
    data.write(
        "types/short_human.yaml",
        "type: short_human\nextends: human\nbody: {height: {default: 1.5}}\n",
    );
    data.replace(
        "experiments/m1_smoke.yaml",
        "- {type: human, count: 1, mind: {module: random}}",
        "- {type: short_human, count: 1, mind: {module: random}, overrides: {body: {insulation: {default: 0.4}}}}",
    );
    let resolved = load_experiment(&data.path("experiments/m1_smoke.yaml")).unwrap();
    let short = &resolved.types["short_human"];
    assert_eq!(short.extends.as_deref(), Some("human"));
    let height = short.body.height.as_ref().unwrap();
    assert_eq!(*height.value(), 1.5);
    assert_eq!(height.range, [1.4, 2.0], "the range comes from the base");
    assert!(height.varies(), "the variation comes from the base");
    assert_eq!(
        short.body.insulation_value(),
        0.2,
        "the type itself is unchanged"
    );
    assert_eq!(
        resolved.population[0].entity_type.body.insulation_value(),
        0.4
    );
    assert_eq!(resolved.population[0].entity_type.body.needs.len(), 4);
}

#[test]
fn inline_types_are_declared_in_the_experiment() {
    let data = common::DataCopy::new("inline");
    data.replace(
        "experiments/m1_smoke.yaml",
        "population:\n",
        "types:\n  - {type: tall_human, extends: human, body: {height: {default: 1.9}}}\npopulation:\n",
    );
    data.replace(
        "experiments/m1_smoke.yaml",
        "{type: human, count: 1",
        "{type: tall_human, count: 1",
    );
    let resolved = load_experiment(&data.path("experiments/m1_smoke.yaml")).unwrap();
    assert_eq!(
        *resolved.types["tall_human"]
            .body
            .height
            .as_ref()
            .unwrap()
            .value(),
        1.9
    );
}
