//! The rule engine: actions, preconditions, effects, processes, and detectors, loaded from
//! `data/rules/`. New predicates and effects are registered modules, never engine special cases.
//!
//! Stage 4 of `docs/specs/MILESTONE_1.md` fills this crate in.

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-rules";
