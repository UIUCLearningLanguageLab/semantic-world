//! Acceptance: a set of deliberately broken files each fail with the right error, naming the
//! file and the field.

mod common;

use sw_schema::{SchemaError, load_experiment};

/// Break one file in a fresh copy of `data/` and load the smoke experiment.
fn broken(label: &str, file: &str, from: &str, to: &str) -> SchemaError {
    let data = common::DataCopy::new(label);
    data.replace(file, from, to);
    load_experiment(&data.path("experiments/m1_smoke.yaml"))
        .err()
        .unwrap_or_else(|| panic!("{label}: the broken file loaded"))
}

fn expect_field(err: &SchemaError, file: &str, field: &str, message_part: &str) {
    match err {
        SchemaError::Field {
            file: f,
            field: fld,
            message,
        } => {
            assert!(f.ends_with(file), "{err}");
            assert_eq!(fld, field, "{err}");
            assert!(message.contains(message_part), "{err}");
        }
        other => panic!("expected a field error, got {other}"),
    }
}

fn expect_not_in_m1(err: &SchemaError, file: &str, field: &str) {
    match err {
        SchemaError::NotInMilestone1 {
            file: f,
            field: fld,
            ..
        } => {
            assert!(f.ends_with(file), "{err}");
            assert_eq!(fld, field, "{err}");
            assert!(err.to_string().ends_with("is not in milestone 1"), "{err}");
        }
        other => panic!("expected a not-in-milestone-1 error, got {other}"),
    }
}

#[test]
fn unknown_field() {
    let err = broken(
        "unknown-field",
        "types/human.yaml",
        "  plan: biped\n",
        "  plan: biped\n  colour: tan\n",
    );
    expect_field(
        &err,
        "types/human.yaml",
        "body.colour",
        "unknown field `colour`",
    );
}

#[test]
fn wrong_type() {
    let err = broken(
        "wrong-type",
        "types/berry_bush.yaml",
        "solid: true",
        "solid: yes",
    );
    expect_field(&err, "types/berry_bush.yaml", "body.solid", "invalid type");
}

#[test]
fn default_outside_range() {
    let err = broken(
        "out-of-range",
        "types/human.yaml",
        "insulation: {default: 0.2, range: [0, 1]}",
        "insulation: {default: 1.5, range: [0, 1]}",
    );
    expect_field(
        &err,
        "types/human.yaml",
        "body.insulation.default",
        "outside the range",
    );
}

#[test]
fn inverted_range() {
    let err = broken(
        "inverted-range",
        "types/human.yaml",
        "range: [40, 110], units: kg",
        "range: [110, 40], units: kg",
    );
    expect_field(
        &err,
        "types/human.yaml",
        "body.mass.range",
        "minimum 110 is above maximum 40",
    );
}

#[test]
fn stock_above_maximum() {
    let err = broken(
        "stock",
        "types/berry_bush.yaml",
        "initial: {default: 3, range: [0, 10]}",
        "initial: {default: 5, range: [0, 10]}",
    );
    expect_field(
        &err,
        "types/berry_bush.yaml",
        "body.stock.berries.initial",
        "above the maximum",
    );
}

#[test]
fn need_with_two_at_max_outcomes() {
    let err = broken(
        "at-max",
        "types/human.yaml",
        "at_max: {health_drain_per_hour: {default: 0.5, range: [0, 5], units: per_hour}}",
        "at_max: {health_drain_per_hour: 0.5, collapse: {duration_hours: 2}}",
    );
    expect_field(
        &err,
        "types/human.yaml",
        "body.needs.hunger.at_max",
        "expected",
    );
}

#[test]
fn unknown_rise_condition() {
    let err = broken(
        "rise-when",
        "types/human.yaml",
        "rise_when: night_outside_shelter",
        "rise_when: sometimes",
    );
    expect_field(
        &err,
        "types/human.yaml",
        "body.needs.cold.rise_when",
        "unknown variant `sometimes`",
    );
}

#[test]
fn bad_color() {
    let err = broken(
        "color",
        "types/pond.yaml",
        "color: \"#1e6fd9\"",
        "color: blue",
    );
    expect_field(
        &err,
        "types/pond.yaml",
        "body.placeholder.color",
        "not a color",
    );
}

#[test]
fn unknown_sensor_is_not_in_milestone_1() {
    let err = broken(
        "ears",
        "types/human.yaml",
        "  touch: {detail: whole_body}\n",
        "  touch: {detail: whole_body}\n  ears: {bands: 8}\n",
    );
    expect_not_in_m1(&err, "types/human.yaml", "sensors.ears");
}

#[test]
fn per_part_touch_is_not_in_milestone_1() {
    let err = broken(
        "per-part",
        "types/human.yaml",
        "touch: {detail: whole_body}",
        "touch: {detail: per_part}",
    );
    expect_not_in_m1(&err, "types/human.yaml", "sensors.touch.detail");
}

#[test]
fn unknown_actuator_is_not_in_milestone_1() {
    let err = broken(
        "wings",
        "types/human.yaml",
        "  mouth: {}\n",
        "  mouth: {}\n  wings: {}\n",
    );
    expect_not_in_m1(&err, "types/human.yaml", "actuators.wings");
}

#[test]
fn realtime_clock_is_not_in_milestone_1() {
    let err = broken(
        "realtime",
        "experiments/m1_smoke.yaml",
        "mode: synchronous",
        "mode: realtime",
    );
    expect_not_in_m1(&err, "experiments/m1_smoke.yaml", "clock.mode");
}

#[test]
fn checkpoints_are_not_in_milestone_1() {
    let err = broken(
        "checkpoints",
        "experiments/m1_smoke.yaml",
        "lifetime: {default_days: 1}\n",
        "lifetime: {default_days: 1}\ncheckpoints: {every_days: 10}\n",
    );
    expect_not_in_m1(&err, "experiments/m1_smoke.yaml", "checkpoints");
}

#[test]
fn persistent_society_is_not_in_milestone_1() {
    let err = broken(
        "death",
        "experiments/m1_smoke.yaml",
        "death: ends_lifetime",
        "death: persistent_society",
    );
    expect_not_in_m1(&err, "experiments/m1_smoke.yaml", "options.death");
}

#[test]
fn speech_as_symbols_is_not_in_milestone_1() {
    let err = broken(
        "speech",
        "experiments/m1_smoke.yaml",
        "speech_as_symbols: false",
        "speech_as_symbols: true",
    );
    expect_not_in_m1(&err, "experiments/m1_smoke.yaml", "views.speech_as_symbols");
}

#[test]
fn scheduled_changes_are_not_in_milestone_1() {
    let err = broken(
        "schedule",
        "worlds/meadow.yaml",
        "regime: scripted\n",
        "regime: scripted\nschedule:\n  - {at_day: 30, change: {bush_regrow_every_s: 1200}}\n",
    );
    expect_not_in_m1(&err, "worlds/meadow.yaml", "schedule");
}

#[test]
fn generative_regime_is_not_in_milestone_1() {
    let err = broken(
        "generative",
        "worlds/meadow.yaml",
        "regime: scripted",
        "regime: generative",
    );
    expect_not_in_m1(&err, "worlds/meadow.yaml", "regime");
}

#[test]
fn unknown_type_in_population() {
    let err = broken(
        "unknown-type",
        "experiments/m1_smoke.yaml",
        "{type: human, count: 1",
        "{type: wolf, count: 1",
    );
    match &err {
        SchemaError::Resolve { file, message } => {
            assert!(file.ends_with("experiments/m1_smoke.yaml"), "{err}");
            assert!(message.contains("unknown type `wolf`"), "{err}");
        }
        other => panic!("{other}"),
    }
}

#[test]
fn extends_cycle() {
    let data = common::DataCopy::new("cycle");
    data.write("types/a.yaml", "type: a\nextends: b\n");
    data.write("types/b.yaml", "type: b\nextends: a\n");
    data.replace(
        "experiments/m1_smoke.yaml",
        "{type: human, count: 1",
        "{type: a, count: 1",
    );
    let err = load_experiment(&data.path("experiments/m1_smoke.yaml")).unwrap_err();
    match &err {
        SchemaError::Resolve { message, .. } => {
            assert!(message.contains("cycle: a -> b -> a"), "{err}")
        }
        other => panic!("{other}"),
    }
}

#[test]
fn file_name_and_type_name_must_agree() {
    let err = broken("misnamed", "types/tree.yaml", "type: tree", "type: oak");
    expect_field(&err, "types/tree.yaml", "type", "declares type `oak`");
}

#[test]
fn duplicate_keys() {
    let err = broken(
        "duplicate",
        "types/pond.yaml",
        "  solid: false\n",
        "  solid: false\n  solid: true\n",
    );
    match &err {
        SchemaError::Yaml { file, message } => {
            assert!(file.ends_with("types/pond.yaml"), "{err}");
            assert!(message.contains("line"), "{err}");
        }
        other => panic!("{other}"),
    }
}

#[test]
fn bad_predicate_call() {
    let err = broken(
        "call",
        "rules/actions.yaml",
        "- facing(self, target, 45)\n      - has_tag(target, edible)",
        "- facing(self, target\n      - has_tag(target, edible)",
    );
    expect_field(
        &err,
        "rules/actions.yaml",
        "actions.eat.pre[2]",
        "missing closing parenthesis",
    );
}

#[test]
fn bad_effect_argument() {
    let err = broken(
        "effect-arg",
        "rules/actions.yaml",
        "by: target.provides.hunger",
        "by: nobody.provides.hunger",
    );
    expect_field(
        &err,
        "rules/actions.yaml",
        "actions.eat.effects[1]",
        "change_need.by",
    );
}

#[test]
fn slot_that_is_neither_range_nor_target() {
    let err = broken(
        "slot",
        "rules/actions.yaml",
        "speed: {range: [0, 1]}",
        "speed: {units: fraction}",
    );
    expect_field(
        &err,
        "rules/actions.yaml",
        "actions.move.args.speed",
        "needs `range` or `has`",
    );
}

#[test]
fn rule_declared_in_two_files() {
    let err = broken(
        "dup-rule",
        "rules/processes.yaml",
        "processes:\n",
        "actions:\n  noop: {}\nprocesses:\n",
    );
    expect_field(
        &err,
        "rules/processes.yaml",
        "actions.noop",
        "more than one rules file",
    );
}

#[test]
fn missing_calendar_field() {
    let err = broken(
        "calendar",
        "worlds/meadow.yaml",
        "  daylight_fraction: {default: 0.6, range: [0, 1]}\n",
        "",
    );
    expect_field(
        &err,
        "worlds/meadow.yaml",
        "calendar.daylight_fraction",
        "required for a run",
    );
}

#[test]
fn condition_path_must_exist() {
    let err = broken(
        "cond-path",
        "experiments/m1_smoke.yaml",
        "seeds: [0]\n",
        "seeds: [0]\nconditions:\n  options.gravity: [low, high]\n",
    );
    expect_field(
        &err,
        "experiments/m1_smoke.yaml",
        "conditions.options.gravity",
        "does not exist",
    );
}

#[test]
fn condition_setting_must_be_valid_in_place() {
    let err = broken(
        "cond-value",
        "experiments/m1_smoke.yaml",
        "seeds: [0]\n",
        "seeds: [0]\nconditions:\n  options.impossible_actions: [masked, sometimes]\n",
    );
    expect_field(
        &err,
        "experiments/m1_smoke.yaml",
        "conditions.options.impossible_actions.1: options.impossible_actions",
        "unknown variant `sometimes`",
    );
}

#[test]
fn propositional_view_needs_the_sensor() {
    let data = common::DataCopy::new("props");
    data.replace(
        "experiments/m1_smoke.yaml",
        "propositional_sensor: false",
        "propositional_sensor: true",
    );
    load_experiment(&data.path("experiments/m1_smoke.yaml")).unwrap();
    data.replace(
        "types/human.yaml",
        "  propositional: {range_m: {default: 20, range: [1, 100], units: m}}   # active only when views.propositional_sensor is on\n",
        "",
    );
    let err = load_experiment(&data.path("experiments/m1_smoke.yaml")).unwrap_err();
    expect_field(
        &err,
        "experiments/m1_smoke.yaml",
        "views.propositional_sensor",
        "declares no `propositional` sensor",
    );
}

#[test]
fn seed_and_seeds_together() {
    let err = broken(
        "seeds",
        "experiments/m1_smoke.yaml",
        "seeds: [0]\n",
        "seeds: [0]\nseed: 1\n",
    );
    expect_field(&err, "experiments/m1_smoke.yaml", "seed", "not both");
}

#[test]
fn missing_file() {
    let err = broken(
        "missing",
        "worlds/meadow.yaml",
        "../rules/processes.yaml",
        "../rules/nothing.yaml",
    );
    match &err {
        SchemaError::Io { file, .. } => assert!(file.ends_with("rules/nothing.yaml"), "{err}"),
        other => panic!("{other}"),
    }
}
