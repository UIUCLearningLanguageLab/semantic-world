//! Shared helpers for the rule engine's tests: a world from the smoke experiment with every
//! object moved out of the way, so a test can place what it needs.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sw_core::{EntitySnapshot, World};
use sw_rules::{ActionChoice, Engine, Event};
use sw_schema::{Resolved, load_experiment_with_overrides};

pub const AGENT: &str = "human_0";
pub const TICKS_PER_HOUR: u64 = 600;

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn smoke(overrides: &[(&str, Value)]) -> Resolved {
    let owned: Vec<(String, Value)> = overrides
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    load_experiment_with_overrides(&repo_root().join("data/experiments/m1_smoke.yaml"), &owned)
        .unwrap()
}

pub struct Scene {
    pub world: World,
    pub engine: Engine,
}

impl Scene {
    /// A cleared scene: the agent at the origin facing +x, every object parked in a grid in
    /// the far corner, 3 days of lifetime, and the given overrides.
    pub fn new(overrides: &[(&str, Value)]) -> Scene {
        let mut all = vec![("lifetime.default_days", Value::from(3))];
        all.extend(overrides.iter().cloned());
        let resolved = smoke(&all);
        let mut world = World::new(&resolved, 0).unwrap();
        let ids = world.entity_ids();
        for (i, id) in ids.iter().enumerate() {
            if id != AGENT {
                let col = (i % 12) as f64;
                let row = (i / 12) as f64;
                world
                    .privileged_set_pose(id, -46.0 + col * 5.0, 46.0 - row * 5.0, 0.0)
                    .unwrap();
            }
        }
        world.privileged_set_pose(AGENT, 0.0, 0.0, 0.0).unwrap();
        let engine = Engine::new(&resolved, &world).unwrap();
        Scene { world, engine }
    }

    pub fn masked() -> Scene {
        Scene::new(&[("options.impossible_actions", Value::from("masked"))])
    }

    pub fn attempt_and_fail() -> Scene {
        Scene::new(&[(
            "options.impossible_actions",
            Value::from("attempt_and_fail"),
        )])
    }

    pub fn place(&mut self, id: &str, x: f64, z: f64, yaw: f64) {
        self.world.privileged_set_pose(id, x, z, yaw).unwrap();
    }

    pub fn human(&self) -> EntitySnapshot {
        self.world.snapshot().entity(AGENT).unwrap().clone()
    }

    pub fn entity(&self, id: &str) -> EntitySnapshot {
        self.world.snapshot().entity(id).unwrap().clone()
    }

    pub fn need(&self, name: &str) -> f64 {
        self.human().need(name).unwrap()
    }

    /// One decision for the agent, returning the interval's events.
    pub fn decide(&mut self, choice: ActionChoice) -> Vec<Event> {
        let mut choices = BTreeMap::new();
        choices.insert(AGENT.to_string(), choice);
        self.engine.step(&mut self.world, &choices).unwrap()
    }

    /// A noop decision, or an empty step while the agent is collapsed or dead.
    pub fn noop(&mut self) -> Vec<Event> {
        if self.world.agents_awaiting_action().is_empty() {
            let empty = BTreeMap::new();
            self.engine.step(&mut self.world, &empty).unwrap()
        } else {
            self.decide(ActionChoice::noop())
        }
    }

    /// Noop decisions for a number of ticks.
    pub fn idle_ticks(&mut self, ticks: u64) -> Vec<Event> {
        let mut events = Vec::new();
        let target = self.world.clock().tick + ticks;
        while self.world.clock().tick < target {
            events.extend(self.noop());
        }
        events
    }

    pub fn mask(&self) -> Vec<bool> {
        self.engine.mask(&self.world, AGENT).unwrap()
    }

    pub fn mask_of(&self, action: &str) -> bool {
        let manifest = self.engine.manifest(AGENT).unwrap();
        self.mask()[manifest.index_of(action).unwrap()]
    }

    pub fn tick(&self) -> u64 {
        self.world.clock().tick
    }
}

pub fn walk(direction: f64, speed: f64) -> ActionChoice {
    ActionChoice::named("move")
        .with("direction", direction)
        .with("speed", speed)
}

pub fn has_event(events: &[Event], f: impl Fn(&sw_rules::EventKind) -> bool) -> bool {
    events.iter().any(|e| f(&e.kind))
}
