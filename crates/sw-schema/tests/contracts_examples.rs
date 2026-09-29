//! Acceptance: the YAML examples in `docs/CONTRACTS.md` (and the additions repeated in
//! `docs/specs/MILESTONE_1.md`) parse into the schema types. Parsing only: the examples name
//! types and modules that milestone 1 does not define, so they are not resolved or validated.

mod common;

use std::path::Path;

use indexmap::IndexMap;
use serde_json::Value;
use sw_schema::entity::{Body, EntityType, NeedSpec};
use sw_schema::experiment::ExperimentConfig;
use sw_schema::rules::RulesFile;
use sw_schema::task::TaskFile;
use sw_schema::yaml;

/// Every ```yaml fence in a Markdown file, with the line it starts on.
fn yaml_fences(markdown: &str) -> Vec<(usize, String)> {
    let mut fences = Vec::new();
    let mut current: Option<(usize, Vec<&str>)> = None;
    for (i, line) in markdown.lines().enumerate() {
        match &mut current {
            None if line.trim_start().starts_with("```yaml") => current = Some((i + 1, Vec::new())),
            Some((start, lines)) if line.trim_start().starts_with("```") => {
                fences.push((*start, lines.join("\n") + "\n"));
                current = None;
            }
            Some((_, lines)) => lines.push(line),
            None => {}
        }
    }
    fences
}

#[derive(Debug, PartialEq)]
enum Kind {
    EntityType,
    DerivedType,
    Rules,
    Experiment,
    Task,
    NeedsFragment,
    BodyFragment,
    SensorManifest,
}

fn classify(value: &Value) -> Kind {
    let has = |k: &str| value.get(k).is_some();
    let only_key = |k: &str| {
        value
            .as_object()
            .is_some_and(|m| m.len() == 1 && m.contains_key(k))
    };
    if has("agent") && has("blocks") {
        Kind::SensorManifest
    } else if has("experiment") {
        Kind::Experiment
    } else if has("actions") || has("processes") || has("predicates") {
        Kind::Rules
    } else if has("subskills") || has("task") {
        Kind::Task
    } else if has("type") && has("extends") {
        Kind::DerivedType
    } else if has("type") {
        Kind::EntityType
    } else if only_key("needs") {
        Kind::NeedsFragment
    } else if only_key("body") {
        Kind::BodyFragment
    } else {
        panic!("unclassified example: {value}")
    }
}

fn parse_fence(value: Value, kind: &Kind, file: &Path) {
    match kind {
        Kind::SensorManifest => {} // an output of `reset`, not a data file
        Kind::Experiment => {
            yaml::from_value::<ExperimentConfig>(value, file).unwrap();
        }
        Kind::Rules => {
            yaml::from_value::<RulesFile>(value, file).unwrap();
        }
        Kind::Task => {
            yaml::from_value::<TaskFile>(value, file).unwrap();
        }
        Kind::EntityType => {
            yaml::from_value::<EntityType>(value, file).unwrap();
        }
        Kind::DerivedType => {
            // A derived type is a patch onto its base. Merge it onto the milestone 1 human,
            // as the resolver would, and parse the result.
            let mut base =
                yaml::read_value(&common::repo_root().join("data/types/human.yaml")).unwrap();
            let base_name = value["extends"].as_str().unwrap().to_string();
            yaml::deep_merge(&mut base, value);
            let parsed: EntityType = yaml::from_value(base, file).unwrap();
            assert_eq!(parsed.extends.as_deref(), Some(base_name.as_str()));
        }
        Kind::NeedsFragment => {
            let needs = value.get("needs").unwrap().clone();
            yaml::from_value::<IndexMap<String, NeedSpec>>(needs, file).unwrap();
        }
        Kind::BodyFragment => {
            let mut body = value.get("body").unwrap().clone();
            yaml::deep_merge(&mut body, serde_json::json!({"plan": "example"}));
            yaml::from_value::<Body>(body, file).unwrap();
        }
    }
}

fn check_document(relative: &str) -> Vec<Kind> {
    let file = common::repo_root().join(relative);
    let markdown = std::fs::read_to_string(&file).unwrap();
    let mut kinds = Vec::new();
    for (line, text) in yaml_fences(&markdown) {
        // The contract examples abbreviate free-form parameters as `{...}`, which is not YAML.
        // Read the placeholder as an empty mapping.
        let text = text.replace("{...}", "{}");
        let value =
            yaml::parse_value(&text, &file).unwrap_or_else(|e| panic!("{relative}:{line}: {e}"));
        let kind = classify(&value);
        parse_fence(value, &kind, &file);
        kinds.push(kind);
    }
    kinds
}

#[test]
fn contracts_examples_parse() {
    let kinds = check_document("docs/CONTRACTS.md");
    for expected in [
        Kind::EntityType,
        Kind::DerivedType,
        Kind::Rules,
        Kind::SensorManifest,
        Kind::Task,
        Kind::Experiment,
        Kind::NeedsFragment,
        Kind::BodyFragment,
    ] {
        assert!(
            kinds.contains(&expected),
            "no {expected:?} example found in CONTRACTS.md"
        );
    }
}

#[test]
fn milestone_examples_parse() {
    let kinds = check_document("docs/specs/MILESTONE_1.md");
    assert!(kinds.contains(&Kind::NeedsFragment));
    assert!(kinds.contains(&Kind::BodyFragment));
}
