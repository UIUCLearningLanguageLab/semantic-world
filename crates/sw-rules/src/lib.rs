//! The rule engine: actions, preconditions, effects, processes, and detectors, loaded from
//! `data/rules/` (contract 2, additions A3 and A4).
//!
//! Predicates, effects, action modules, and detectors are registered by name in
//! [`predicates`], [`effects`], [`modules`], and [`detectors`]. The rules files refer to the
//! names; new ones are added to the registries, never as special cases in the engine.

pub mod actions;
pub mod context;
pub mod detectors;
pub mod effects;
pub mod engine;
pub mod error;
pub mod events;
pub mod modules;
pub mod predicates;
pub mod processes;

pub use actions::{ActionChoice, ActionInfo, ActionManifest, ArgInfo, Ongoing};
pub use detectors::Fact;
pub use engine::Engine;
pub use error::{Result, RuleError};
pub use events::{Event, EventKind};

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-rules";
