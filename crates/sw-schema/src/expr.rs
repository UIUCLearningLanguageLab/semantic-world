//! Rule expressions (addition A3): predicate calls, effect calls, and their arguments.
//!
//! An argument is a literal or a reference. A reference is `self`, `target`, or `this`, with an
//! optional field path (`target.provides.hunger`) and an optional leading minus sign. There is
//! no other arithmetic. The names of predicates and effects are not checked here: they resolve
//! against the registries in the rule engine, so new ones can be added as modules.

use std::borrow::Cow;
use std::fmt;

use indexmap::IndexMap;
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The entity a reference starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefRoot {
    /// The acting agent.
    SelfRef,
    /// The action's target.
    Target,
    /// The entity a process applies to.
    This,
}

impl RefRoot {
    fn parse(word: &str) -> Option<RefRoot> {
        match word {
            "self" => Some(RefRoot::SelfRef),
            "target" => Some(RefRoot::Target),
            "this" => Some(RefRoot::This),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RefRoot::SelfRef => "self",
            RefRoot::Target => "target",
            RefRoot::This => "this",
        }
    }
}

/// A reference to an entity, or to a field of an entity, optionally negated.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub root: RefRoot,
    /// The field path below the root, for example `["provides", "hunger"]`.
    pub path: Vec<String>,
    pub negated: bool,
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negated {
            write!(f, "-")?;
        }
        write!(f, "{}", self.root.as_str())?;
        for segment in &self.path {
            write!(f, ".{segment}")?;
        }
        Ok(())
    }
}

/// One argument of a predicate call or an effect.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Number(f64),
    Bool(bool),
    /// A bare word that is not a reference, such as a tag, a need, or a state name.
    Symbol(String),
    Ref(Reference),
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl Arg {
    /// Parse one argument written as text.
    pub fn parse(text: &str) -> Result<Arg, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("empty argument".into());
        }
        match text {
            "true" => return Ok(Arg::Bool(true)),
            "false" => return Ok(Arg::Bool(false)),
            _ => {}
        }
        if let Ok(n) = text.parse::<f64>()
            && n.is_finite()
        {
            return Ok(Arg::Number(n));
        }
        let (negated, body) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let mut segments = body.split('.');
        let head = segments.next().unwrap_or_default();
        if let Some(root) = RefRoot::parse(head) {
            let path: Vec<String> = segments.map(str::to_string).collect();
            if let Some(bad) = path.iter().find(|s| !is_identifier(s)) {
                return Err(format!("`{bad}` is not a valid field name in `{text}`"));
            }
            return Ok(Arg::Ref(Reference {
                root,
                path,
                negated,
            }));
        }
        if negated {
            return Err(format!(
                "`{text}`: a minus sign may only precede a number or a reference"
            ));
        }
        if is_identifier(body) {
            return Ok(Arg::Symbol(body.to_string()));
        }
        Err(format!(
            "`{text}` is not a number, a symbol, or a reference (self, target, this)"
        ))
    }

    /// Parse an argument from a YAML scalar.
    pub fn from_value(value: &Value) -> Result<Arg, String> {
        match value {
            Value::Number(n) => n
                .as_f64()
                .filter(|f| f.is_finite())
                .map(Arg::Number)
                .ok_or_else(|| format!("{n} is not a finite number")),
            Value::Bool(b) => Ok(Arg::Bool(*b)),
            Value::String(s) => Arg::parse(s),
            other => Err(format!(
                "expected a number, a symbol, or a reference, found {}",
                describe_value(other)
            )),
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            Arg::Number(n) => serde_json::json!(n),
            Arg::Bool(b) => Value::Bool(*b),
            Arg::Symbol(s) => Value::String(s.clone()),
            Arg::Ref(r) => Value::String(r.to_string()),
        }
    }
}

fn describe_value(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a mapping",
    }
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Number(n) => write!(f, "{n}"),
            Arg::Bool(b) => write!(f, "{b}"),
            Arg::Symbol(s) => write!(f, "{s}"),
            Arg::Ref(r) => write!(f, "{r}"),
        }
    }
}

impl Serialize for Arg {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Arg {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Arg::from_value(&value).map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for Arg {
    fn schema_name() -> Cow<'static, str> {
        "RuleArgument".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "A literal number or boolean, a bare symbol, or a reference: self, target, or this, optionally followed by a field path and optionally preceded by a minus sign.",
            "type": ["number", "boolean", "string"]
        })
    }
}

/// A call to a registered predicate, written as text: `within(self, target, 1.0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    pub args: Vec<Arg>,
}

impl Call {
    pub fn parse(text: &str) -> Result<Call, String> {
        let text = text.trim();
        let (name, rest) = match text.find('(') {
            Some(open) => {
                let rest = &text[open + 1..];
                let close = rest
                    .rfind(')')
                    .ok_or_else(|| format!("`{text}`: missing closing parenthesis"))?;
                if !rest[close + 1..].trim().is_empty() {
                    return Err(format!("`{text}`: text after the closing parenthesis"));
                }
                (&text[..open], &rest[..close])
            }
            None => (text, ""),
        };
        let name = name.trim();
        if !is_identifier(name) {
            return Err(format!("`{text}`: `{name}` is not a valid predicate name"));
        }
        let mut args = Vec::new();
        if !rest.trim().is_empty() {
            for piece in rest.split(',') {
                args.push(Arg::parse(piece).map_err(|m| format!("`{text}`: {m}"))?);
            }
        }
        Ok(Call {
            name: name.to_string(),
            args,
        })
    }
}

impl fmt::Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args: Vec<String> = self.args.iter().map(|a| a.to_string()).collect();
        write!(f, "{}({})", self.name, args.join(", "))
    }
}

impl Serialize for Call {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Call {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Call::parse(&text).map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for Call {
    fn schema_name() -> Cow<'static, str> {
        "PredicateCall".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "A call to a registered predicate, such as `within(self, target, 1.0)`.",
            "type": "string",
            "pattern": "^[A-Za-z_][A-Za-z0-9_]*(\\(.*\\))?$"
        })
    }
}

/// A call to a registered effect, written as a one-key mapping:
/// `{change_need: {entity: self, need: hunger, by: -0.25}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub name: String,
    pub args: IndexMap<String, Arg>,
}

impl Effect {
    /// The argument with this name, if given.
    pub fn arg(&self, name: &str) -> Option<&Arg> {
        self.args.get(name)
    }
}

impl Serialize for Effect {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut outer = serde_json::Map::new();
        let inner: serde_json::Map<String, Value> = self
            .args
            .iter()
            .map(|(k, v)| (k.clone(), v.to_value()))
            .collect();
        outer.insert(self.name.clone(), Value::Object(inner));
        Value::Object(outer).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Effect {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = Value::deserialize(deserializer)?;
        let Value::Object(outer) = value else {
            return Err(D::Error::custom(
                "an effect is a mapping with one key, the effect name",
            ));
        };
        if outer.len() != 1 {
            return Err(D::Error::custom(format!(
                "an effect is a mapping with exactly one key, the effect name; found {} keys",
                outer.len()
            )));
        }
        let (name, inner) = outer.into_iter().next().expect("one key");
        if !is_identifier(&name) {
            return Err(D::Error::custom(format!(
                "`{name}` is not a valid effect name"
            )));
        }
        let Value::Object(inner) = inner else {
            return Err(D::Error::custom(format!(
                "{name}: the effect's arguments must be a mapping"
            )));
        };
        let mut args = IndexMap::new();
        for (key, arg_value) in inner {
            let arg = Arg::from_value(&arg_value)
                .map_err(|m| D::Error::custom(format!("{name}.{key}: {m}")))?;
            args.insert(key, arg);
        }
        Ok(Effect { name, args })
    }
}

impl JsonSchema for Effect {
    fn schema_name() -> Cow<'static, str> {
        "Effect".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let arg = generator.subschema_for::<Arg>();
        json_schema!({
            "description": "A call to a registered effect: a mapping with one key, the effect name, whose value maps argument names to arguments.",
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": {"type": "object", "additionalProperties": arg}
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_literals_symbols_and_references() {
        assert_eq!(Arg::parse("1.0").unwrap(), Arg::Number(1.0));
        assert_eq!(Arg::parse("-0.25").unwrap(), Arg::Number(-0.25));
        assert_eq!(Arg::parse("true").unwrap(), Arg::Bool(true));
        assert_eq!(Arg::parse("edible").unwrap(), Arg::Symbol("edible".into()));
        assert_eq!(
            Arg::parse("-target.provides.hunger").unwrap(),
            Arg::Ref(Reference {
                root: RefRoot::Target,
                path: vec!["provides".into(), "hunger".into()],
                negated: true
            })
        );
        assert_eq!(
            Arg::parse("self").unwrap(),
            Arg::Ref(Reference {
                root: RefRoot::SelfRef,
                path: vec![],
                negated: false
            })
        );
        assert!(Arg::parse("-edible").is_err());
        assert!(Arg::parse("a b").is_err());
        assert!(Arg::parse("this.").is_err());
    }

    #[test]
    fn parses_calls() {
        let call = Call::parse("within(self, target, 1.0)").unwrap();
        assert_eq!(call.name, "within");
        assert_eq!(call.args.len(), 3);
        assert_eq!(call.to_string(), "within(self, target, 1)");
        assert_eq!(Call::parse("night").unwrap().args.len(), 0);
        assert_eq!(Call::parse("night()").unwrap().args.len(), 0);
        assert!(Call::parse("within(self").is_err());
        assert!(Call::parse("with in(self)").is_err());
        assert!(Call::parse("within(self) x").is_err());
    }

    #[test]
    fn effects_are_one_key_mappings() {
        let text = "change_need: {entity: self, need: hunger, by: target.provides.hunger}\n";
        let effect: Effect = crate::yaml::from_str(text, std::path::Path::new("t.yaml")).unwrap();
        assert_eq!(effect.name, "change_need");
        assert_eq!(effect.arg("need"), Some(&Arg::Symbol("hunger".into())));
        let round = serde_json::to_value(&effect).unwrap();
        assert_eq!(round["change_need"]["by"], "target.provides.hunger");
        let err = crate::yaml::from_str::<Effect>(
            "{change_need: {by: 1}, take_stock: {amount: 1}}\n",
            std::path::Path::new("t.yaml"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("exactly one key"), "{err}");
    }
}
