//! Errors from building or running a world.

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error(transparent)]
    Schema(#[from] sw_schema::SchemaError),
    /// The map generator could not satisfy its constraints.
    #[error("map generation: {0}")]
    Map(String),
    /// A registered module name was not found.
    #[error("unknown {kind} `{name}`")]
    UnknownModule { kind: &'static str, name: String },
    /// A call named an entity that does not exist, or is not an agent.
    #[error("unknown agent `{0}`")]
    UnknownAgent(String),
    /// A type or parameter is missing something the engine needs.
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
