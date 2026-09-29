//! Errors that name the file and the field.

use std::path::{Path, PathBuf};

/// An error from loading, validating, or resolving a data file.
///
/// Every variant names the file. Variants about a value also name the field, as a dotted path
/// such as `body.needs.hunger.rise_per_day`.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    /// The file could not be read.
    #[error("{file}: cannot read file: {source}")]
    Io {
        file: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The file is not valid YAML, or is valid YAML that the strict parser refuses
    /// (duplicate keys, YAML 1.1 booleans such as `no`, non-string keys).
    #[error("{file}: YAML error: {message}")]
    Yaml { file: PathBuf, message: String },
    /// A field is missing, unknown, of the wrong type, or has a value that fails validation.
    #[error("{file}: {field}: {message}")]
    Field {
        file: PathBuf,
        field: String,
        message: String,
    },
    /// The file uses a feature that is designed but not built in milestone 1.
    #[error("{file}: {field}: {feature} is not in milestone 1")]
    NotInMilestone1 {
        file: PathBuf,
        field: String,
        feature: String,
    },
    /// A reference between files cannot be resolved: an unknown type, a cycle of `extends`,
    /// a missing rules file, or a duplicate name across rules files.
    #[error("{file}: {message}")]
    Resolve { file: PathBuf, message: String },
}

impl SchemaError {
    pub fn field(file: &Path, field: impl Into<String>, message: impl Into<String>) -> Self {
        SchemaError::Field {
            file: file.to_path_buf(),
            field: field.into(),
            message: message.into(),
        }
    }

    pub fn not_in_milestone_1(
        file: &Path,
        field: impl Into<String>,
        feature: impl Into<String>,
    ) -> Self {
        SchemaError::NotInMilestone1 {
            file: file.to_path_buf(),
            field: field.into(),
            feature: feature.into(),
        }
    }

    pub fn resolve(file: &Path, message: impl Into<String>) -> Self {
        SchemaError::Resolve {
            file: file.to_path_buf(),
            message: message.into(),
        }
    }

    /// The file the error is about.
    pub fn file(&self) -> &Path {
        match self {
            SchemaError::Io { file, .. }
            | SchemaError::Yaml { file, .. }
            | SchemaError::Field { file, .. }
            | SchemaError::NotInMilestone1 { file, .. }
            | SchemaError::Resolve { file, .. } => file,
        }
    }

    /// The dotted field path, for the variants that have one.
    pub fn field_path(&self) -> Option<&str> {
        match self {
            SchemaError::Field { field, .. } | SchemaError::NotInMilestone1 { field, .. } => {
                Some(field)
            }
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, SchemaError>;

/// Where a validation check is happening: the file and the dotted path so far.
///
/// `Ctx` is cheap to clone, so nested checks call [`Ctx::child`] freely.
#[derive(Debug, Clone)]
pub struct Ctx {
    pub file: PathBuf,
    pub path: String,
}

impl Ctx {
    pub fn new(file: &Path) -> Self {
        Ctx {
            file: file.to_path_buf(),
            path: String::new(),
        }
    }

    /// The context of a field or element below this one.
    pub fn child(&self, segment: impl AsRef<str>) -> Ctx {
        let segment = segment.as_ref();
        let path = if self.path.is_empty() {
            segment.to_string()
        } else {
            format!("{}.{}", self.path, segment)
        };
        Ctx {
            file: self.file.clone(),
            path,
        }
    }

    pub fn error(&self, message: impl Into<String>) -> SchemaError {
        SchemaError::field(&self.file, self.path.clone(), message)
    }

    pub fn not_in_milestone_1(&self, feature: impl Into<String>) -> SchemaError {
        SchemaError::not_in_milestone_1(&self.file, self.path.clone(), feature)
    }

    /// Fail unless `condition` holds.
    pub fn check(&self, condition: bool, message: impl Into<String>) -> Result<()> {
        if condition {
            Ok(())
        } else {
            Err(self.error(message))
        }
    }
}
