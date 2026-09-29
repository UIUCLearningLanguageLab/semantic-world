//! PyO3 bindings, compiled as the Python extension module `semantic_world._core`.
//!
//! Stage 5 of `docs/specs/MILESTONE_1.md` adds the `World` class and the observation types.
//! Stage 1 only proves that the module builds and imports.

use pyo3::prelude::*;

/// Version of the Rust crates, from the workspace `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Names of the engine crates linked into this module, in build-stage order.
pub const CRATES: [&str; 6] = [
    sw_schema::CRATE_NAME,
    sw_core::CRATE_NAME,
    sw_rules::CRATE_NAME,
    sw_sense::CRATE_NAME,
    sw_render::CRATE_NAME,
    sw_log::CRATE_NAME,
];

/// The version of the Rust crates behind this module.
#[pyfunction]
fn version() -> &'static str {
    VERSION
}

/// The names of the engine crates linked into this module.
#[pyfunction]
fn crates() -> Vec<&'static str> {
    CRATES.to_vec()
}

/// Whether the module was built with the `rerun` Cargo feature.
#[pyfunction]
fn rerun_enabled() -> bool {
    sw_log::rerun_enabled()
}

/// The `semantic_world._core` module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(crates, m)?)?;
    m.add_function(wrap_pyfunction!(rerun_enabled, m)?)?;
    Ok(())
}
