//! Headless `wgpu` renderer for eyes, drawing placeholder shapes. Rendering is a view of the
//! canonical state: the simulation never depends on it, and it can be switched off entirely.
//!
//! Stage 6 of `docs/specs/MILESTONE_1.md` fills this crate in.

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-render";
