//! Rust types for every data file, YAML loading, validation, and JSON Schema export.
//!
//! The schema crate knows nothing about the engine. The formats it implements are defined in
//! `docs/CONTRACTS.md` and the additions in `docs/specs/MILESTONE_1.md`:
//!
//! - [`entity`]: entity type files (contract 1, additions A1 and A2);
//! - [`rules`]: actions, processes, and the proposition vocabulary (contract 2, addition A3);
//! - [`world`]: world settings, the map, the calendar, and light;
//! - [`experiment`]: experiment configurations (contract 10, addition A5);
//! - [`task`]: task files (contract 9), parsed only.
//!
//! Loading is strict ([`yaml`]): unknown fields, duplicate keys, and YAML 1.1 booleans are
//! errors, and every error names the file and the field. [`resolve`] follows the references
//! between files, applies `extends` and population overrides, validates everything, and
//! rejects features that are not in milestone 1.

pub mod entity;
pub mod error;
pub mod experiment;
pub mod expr;
pub mod resolve;
pub mod rules;
pub mod task;
pub mod traits;
pub mod world;
pub mod yaml;

use std::path::{Path, PathBuf};

pub use error::{Ctx, Result, SchemaError};
pub use resolve::{Resolved, ResolvedPopulation, load_experiment, load_experiment_with_overrides};
pub use traits::{Trait, TraitValue, Variation};

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-schema";

/// Version of the data-file formats this crate implements: the contracts' major version.
pub const CONTRACT_VERSION: u32 = 1;

/// The JSON Schemas for every data file, as `(file name, schema)`.
pub fn schemas() -> Vec<(&'static str, schemars::Schema)> {
    fn schema_for<T: schemars::JsonSchema>() -> schemars::Schema {
        schemars::SchemaGenerator::new(schemars::generate::SchemaSettings::draft2020_12())
            .into_root_schema_for::<T>()
    }
    vec![
        (
            "entity_type.schema.json",
            schema_for::<entity::EntityType>(),
        ),
        ("rules.schema.json", schema_for::<rules::RulesFile>()),
        ("world.schema.json", schema_for::<world::WorldSpec>()),
        (
            "experiment.schema.json",
            schema_for::<experiment::ExperimentConfig>(),
        ),
        ("task.schema.json", schema_for::<task::TaskFile>()),
    ]
}

/// The text of one schema file: pretty JSON with a trailing newline.
pub fn schema_text(schema: &schemars::Schema) -> String {
    let mut text = serde_json::to_string_pretty(schema.as_value()).expect("a schema serializes");
    text.push('\n');
    text
}

/// Write every JSON Schema into `dir`, creating it if needed. Returns the files written.
pub fn write_schemas(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut written = Vec::new();
    for (name, schema) in schemas() {
        let file = dir.join(name);
        std::fs::write(&file, schema_text(&schema))?;
        written.push(file);
    }
    Ok(written)
}
