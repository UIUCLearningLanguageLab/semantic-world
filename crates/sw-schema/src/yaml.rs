//! Strict YAML loading, in two steps.
//!
//! Step one parses the text into a generic [`Value`] with `serde-saphyr`, which rejects
//! duplicate keys and YAML 1.1 booleans (`yes`, `no`, `on`, `off`). Step two turns the value
//! into a typed struct, tracking the field path so an error can say `body.needs.hunger` rather
//! than "invalid type". The generic value in between is also where `extends` and overrides are
//! merged, before the typed parse.

use std::path::Path;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{Result, SchemaError};

/// The strict parser options used for every data file.
fn strict_options() -> serde_saphyr::Options {
    let mut options = serde_saphyr::Options::default();
    options.strict_booleans = true;
    options.duplicate_keys = serde_saphyr::DuplicateKeyPolicy::Error;
    options
}

/// Parse YAML text into a generic value, strictly.
pub fn parse_value(text: &str, file: &Path) -> Result<Value> {
    serde_saphyr::from_str_with_options::<Value>(text, strict_options()).map_err(|e| {
        SchemaError::Yaml {
            file: file.to_path_buf(),
            message: describe_yaml_error(&e),
        }
    })
}

fn describe_yaml_error(e: &serde_saphyr::Error) -> String {
    match e.location() {
        Some(loc) => format!("line {}, column {}: {}", loc.line(), loc.column(), e),
        None => e.to_string(),
    }
}

/// Read a file and parse it into a generic value, strictly.
pub fn read_value(file: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(file).map_err(|source| SchemaError::Io {
        file: file.to_path_buf(),
        source,
    })?;
    parse_value(&text, file)
}

/// Turn a generic value into a typed struct, reporting the field path on failure.
pub fn from_value<T: DeserializeOwned>(value: Value, file: &Path) -> Result<T> {
    serde_path_to_error::deserialize(value).map_err(|e| {
        let field = e.path().to_string();
        let field = if field == "." {
            String::from("(document)")
        } else {
            field
        };
        SchemaError::Field {
            file: file.to_path_buf(),
            field,
            message: e.into_inner().to_string(),
        }
    })
}

/// Parse YAML text straight into a typed struct.
pub fn from_str<T: DeserializeOwned>(text: &str, file: &Path) -> Result<T> {
    from_value(parse_value(text, file)?, file)
}

/// Read a file straight into a typed struct.
pub fn read<T: DeserializeOwned>(file: &Path) -> Result<T> {
    from_value(read_value(file)?, file)
}

/// Merge `patch` onto `base`: mappings merge key by key, recursively; anything else in the
/// patch replaces the base value. This is the `extends` mechanism of contract 1 and the
/// `overrides` mechanism of contract 10.
pub fn deep_merge(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base_map), Value::Object(patch_map)) => {
            for (key, patch_value) in patch_map {
                match base_map.get_mut(&key) {
                    Some(base_value) => deep_merge(base_value, patch_value),
                    None => {
                        base_map.insert(key, patch_value);
                    }
                }
            }
        }
        (base, patch) => *base = patch,
    }
}

/// Set the value at a dotted path such as `population.0.nervous_system.module`, creating
/// mappings along the way. A numeric segment indexes a list. Returns an error message when a
/// numeric segment is out of range.
pub fn set_path(root: &mut Value, path: &str, new_value: Value) -> std::result::Result<(), String> {
    let mut current = root;
    let segments: Vec<&str> = path.split('.').collect();
    for (i, segment) in segments.iter().enumerate() {
        let last = i + 1 == segments.len();
        match current {
            Value::Array(items) => {
                let index: usize = segment
                    .parse()
                    .map_err(|_| format!("`{segment}` is not an index into a list"))?;
                let item = items
                    .get_mut(index)
                    .ok_or_else(|| format!("index {index} is out of range"))?;
                if last {
                    *item = new_value;
                    return Ok(());
                }
                current = item;
            }
            Value::Object(map) => {
                if last {
                    map.insert(segment.to_string(), new_value);
                    return Ok(());
                }
                current = map
                    .entry(segment.to_string())
                    .or_insert_with(|| Value::Object(Default::default()));
            }
            _ => return Err(format!("`{segment}` is below a scalar value")),
        }
    }
    Ok(())
}

/// Join a relative path onto a base directory and remove `.` and `..` components lexically,
/// so error messages say `data/rules/actions.yaml` rather than `data/experiments/../rules/...`.
pub fn join_normalized(base: &Path, relative: &Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut parts: Vec<Component> = Vec::new();
    let joined = base.join(relative);
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match parts.last() {
                Some(Component::Normal(_)) => {
                    parts.pop();
                }
                _ => parts.push(component),
            },
            other => parts.push(other),
        }
    }
    parts.iter().collect()
}

/// Look up a dotted path, if every segment exists.
pub fn get_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        current = match current {
            Value::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
            Value::Object(map) => map.get(segment)?,
            _ => return None,
        };
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn file() -> std::path::PathBuf {
        "test.yaml".into()
    }

    #[test]
    fn duplicate_keys_are_rejected() {
        let err = parse_value("a: 1\na: 2\n", &file()).unwrap_err();
        assert!(matches!(err, SchemaError::Yaml { .. }), "{err}");
    }

    #[test]
    fn yaml_1_1_booleans_are_strings() {
        let value = parse_value("flag: no\n", &file()).unwrap();
        assert_eq!(value, json!({"flag": "no"}));
    }

    #[test]
    fn deep_merge_merges_mappings_and_replaces_scalars() {
        let mut base =
            json!({"body": {"height": {"default": 1.7, "range": [1.4, 2.0]}, "mass": 70}});
        deep_merge(
            &mut base,
            json!({"body": {"height": {"default": 1.5}}, "tags": ["x"]}),
        );
        assert_eq!(
            base,
            json!({"body": {"height": {"default": 1.5, "range": [1.4, 2.0]}, "mass": 70}, "tags": ["x"]})
        );
    }

    #[test]
    fn set_path_indexes_lists() {
        let mut root = json!({"population": [{"type": "human"}]});
        set_path(
            &mut root,
            "population.0.nervous_system.module",
            json!("random"),
        )
        .unwrap();
        assert_eq!(
            get_path(&root, "population.0.nervous_system.module"),
            Some(&json!("random"))
        );
        assert!(set_path(&mut root, "population.3.type", json!("x")).is_err());
    }

    #[test]
    fn join_normalized_removes_parent_components() {
        let joined = join_normalized(Path::new("data/experiments"), Path::new("../worlds/x.yaml"));
        assert_eq!(joined, Path::new("data/worlds/x.yaml"));
        let joined = join_normalized(Path::new("../a"), Path::new("../../b"));
        assert_eq!(joined, Path::new("../../b"));
    }

    #[test]
    fn typed_errors_name_the_field() {
        #[derive(Debug, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Outer {
            #[allow(dead_code)]
            inner: Inner,
        }
        #[derive(Debug, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Inner {
            #[allow(dead_code)]
            count: u32,
        }
        let err = from_str::<Outer>("inner: {count: -1}\n", &file()).unwrap_err();
        assert_eq!(err.field_path(), Some("inner.count"));
        let err = from_str::<Outer>("inner: {count: 1, extra: 2}\n", &file()).unwrap_err();
        assert_eq!(err.field_path(), Some("inner.extra"));
        assert!(err.to_string().contains("unknown field `extra`"), "{err}");
    }
}
