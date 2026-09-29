//! Rust types for every data file, YAML loading, validation, and JSON Schema export.
//!
//! The schema crate knows nothing about the engine. Stage 2 of `docs/specs/MILESTONE_1.md`
//! fills this crate in. The formats it implements are defined in `docs/CONTRACTS.md`.

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-schema";
