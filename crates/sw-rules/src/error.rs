//! Errors from compiling or running rules.

#[derive(Debug, thiserror::Error)]
pub enum RuleError {
    /// A rules file names something the registries do not have, or uses it wrongly. The
    /// message names the rule and the field.
    #[error("rules: {0}")]
    Invalid(String),
    /// A reference such as `target` is used where nothing is bound to it.
    #[error("`{0}` is not bound here")]
    Unbound(&'static str),
    /// An argument could not be resolved to the kind of value the rule needs.
    #[error("{0}")]
    Arg(String),
    /// An agent chose an action it may not use in this way.
    #[error("{agent}: {message}")]
    NotAllowed { agent: String, message: String },
    /// `step` was called between decision points, or for an unknown agent or action.
    #[error("{0}")]
    Usage(String),
    #[error(transparent)]
    Core(#[from] sw_core::CoreError),
}

pub type Result<T> = std::result::Result<T, RuleError>;
