//! Parquet log writers (contract 7) and, behind the `rerun` Cargo feature, Rerun recording.
//!
//! Stage 7 of `docs/specs/MILESTONE_1.md` fills this crate in.

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-log";

/// Whether this build of the crate was compiled with the `rerun` feature.
pub const fn rerun_enabled() -> bool {
    cfg!(feature = "rerun")
}
