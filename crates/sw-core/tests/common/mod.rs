//! Shared helpers for the core crate's integration tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::Value;
use sw_core::World;
use sw_schema::{Resolved, load_experiment_with_overrides};

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The smoke experiment with dotted-path overrides applied.
pub fn smoke(overrides: &[(&str, Value)]) -> Resolved {
    let owned: Vec<(String, Value)> = overrides
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    load_experiment_with_overrides(&repo_root().join("data/experiments/m1_smoke.yaml"), &owned)
        .unwrap()
}

/// A world from the smoke experiment with a given lifetime.
pub fn world(seed: u64, lifetime_days: f64) -> World {
    let resolved = smoke(&[("lifetime.default_days", Value::from(lifetime_days))]);
    World::new(&resolved, seed).unwrap()
}

/// Ticks per calendar hour in the milestone 1 calendar.
pub const TICKS_PER_HOUR: u64 = 600;
pub const TICKS_PER_DAY: u64 = 14_400;

/// Run until the given tick.
pub fn run_to(world: &mut World, tick: u64) {
    while world.clock().tick < tick {
        world.tick();
    }
}
