//! Traits: the declared form of every numeric or categorical setting (contract 1).
//!
//! A trait has a default, a range, optional units, and optional variation. A trait written as a
//! bare value is shorthand for a trait with that default and a range of that single value.
//! [`Trait::sample`] draws an individual's value from the variation and clips it to the range.

use std::borrow::Cow;
use std::fmt;

use rand::{Rng, RngExt};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Ctx, Result};

/// How individuals vary when sampled. Sampled values are clipped to the trait's range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Variation {
    /// Normal distribution centered on the default, with standard deviation `sd`.
    Normal { sd: f64 },
    /// Uniform distribution centered on the default, from `default - half_width` to
    /// `default + half_width`.
    Uniform { half_width: f64 },
}

impl Variation {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        match self {
            Variation::Normal { sd } => ctx.child("normal.sd").check(
                sd.is_finite() && *sd >= 0.0,
                "sd must be a finite number ≥ 0",
            ),
            Variation::Uniform { half_width } => ctx.child("uniform.half_width").check(
                half_width.is_finite() && *half_width >= 0.0,
                "half_width must be a finite number ≥ 0",
            ),
        }
    }

    /// Draw an offset from the default. Uses only the uniform generator, so the draw is the
    /// same on every platform for the same stream position.
    fn sample_offset<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Variation::Normal { sd } => {
                // Box–Muller. `1.0 - u1` keeps the logarithm finite, since `u1` is in [0, 1).
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let z = (-2.0_f64 * (1.0 - u1).ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                sd * z
            }
            Variation::Uniform { half_width } => {
                let u: f64 = rng.random();
                (2.0 * u - 1.0) * half_width
            }
        }
    }
}

/// A type that can be the value of a trait: numbers, categories, and flags.
pub trait TraitValue:
    Clone + fmt::Debug + PartialEq + Serialize + DeserializeOwned + JsonSchema + 'static
{
    /// The declared form of the allowed range: `[min, max]` for numbers, a list of options
    /// for categories.
    type Range: Clone + fmt::Debug + PartialEq + Serialize + DeserializeOwned + JsonSchema + 'static;

    /// Whether the type is numeric, so units and variation apply.
    const NUMERIC: bool;

    /// The range that contains only this value, for the bare-value shorthand.
    fn single_range(&self) -> Self::Range;

    /// Check that the range itself is well formed.
    fn check_range(range: &Self::Range) -> std::result::Result<(), String>;

    fn in_range(&self, range: &Self::Range) -> bool;

    /// Clip to the range. Categories are returned unchanged.
    fn clip(&self, range: &Self::Range) -> Self;

    /// Draw a value around `default`. Only numbers can vary.
    fn sample<R: Rng + ?Sized>(
        default: &Self,
        variation: &Variation,
        rng: &mut R,
    ) -> std::result::Result<Self, String>;

    fn describe(&self) -> String;

    fn describe_range(range: &Self::Range) -> String;
}

impl TraitValue for f64 {
    type Range = [f64; 2];
    const NUMERIC: bool = true;

    fn single_range(&self) -> Self::Range {
        [*self, *self]
    }

    fn check_range(range: &Self::Range) -> std::result::Result<(), String> {
        if !(range[0].is_finite() && range[1].is_finite()) {
            return Err("range bounds must be finite numbers".into());
        }
        if range[0] > range[1] {
            return Err(format!(
                "range minimum {} is above maximum {}",
                range[0], range[1]
            ));
        }
        Ok(())
    }

    fn in_range(&self, range: &Self::Range) -> bool {
        *self >= range[0] && *self <= range[1]
    }

    fn clip(&self, range: &Self::Range) -> Self {
        self.clamp(range[0], range[1])
    }

    fn sample<R: Rng + ?Sized>(
        default: &Self,
        variation: &Variation,
        rng: &mut R,
    ) -> std::result::Result<Self, String> {
        Ok(default + variation.sample_offset(rng))
    }

    fn describe(&self) -> String {
        self.to_string()
    }

    fn describe_range(range: &Self::Range) -> String {
        format!("[{}, {}]", range[0], range[1])
    }
}

impl TraitValue for i64 {
    type Range = [i64; 2];
    const NUMERIC: bool = true;

    fn single_range(&self) -> Self::Range {
        [*self, *self]
    }

    fn check_range(range: &Self::Range) -> std::result::Result<(), String> {
        if range[0] > range[1] {
            return Err(format!(
                "range minimum {} is above maximum {}",
                range[0], range[1]
            ));
        }
        Ok(())
    }

    fn in_range(&self, range: &Self::Range) -> bool {
        *self >= range[0] && *self <= range[1]
    }

    fn clip(&self, range: &Self::Range) -> Self {
        (*self).clamp(range[0], range[1])
    }

    fn sample<R: Rng + ?Sized>(
        default: &Self,
        variation: &Variation,
        rng: &mut R,
    ) -> std::result::Result<Self, String> {
        let value = *default as f64 + variation.sample_offset(rng);
        Ok(value.round() as i64)
    }

    fn describe(&self) -> String {
        self.to_string()
    }

    fn describe_range(range: &Self::Range) -> String {
        format!("[{}, {}]", range[0], range[1])
    }
}

/// Implement [`TraitValue`] for a categorical type: a string, a flag, or a unit enum.
/// The range is the list of allowed options.
#[macro_export]
macro_rules! categorical_trait {
    ($t:ty) => {
        impl $crate::traits::TraitValue for $t {
            type Range = Vec<$t>;
            const NUMERIC: bool = false;

            fn single_range(&self) -> Self::Range {
                vec![self.clone()]
            }

            fn check_range(range: &Self::Range) -> ::std::result::Result<(), String> {
                if range.is_empty() {
                    return Err("range must list at least one option".into());
                }
                Ok(())
            }

            fn in_range(&self, range: &Self::Range) -> bool {
                range.contains(self)
            }

            fn clip(&self, _range: &Self::Range) -> Self {
                self.clone()
            }

            fn sample<R: ::rand::Rng + ?Sized>(
                _default: &Self,
                _variation: &$crate::traits::Variation,
                _rng: &mut R,
            ) -> ::std::result::Result<Self, String> {
                Err("variation applies to numeric traits only".into())
            }

            fn describe(&self) -> String {
                ::serde_json::to_value(self)
                    .ok()
                    .and_then(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .or_else(|| Some(v.to_string()))
                    })
                    .unwrap_or_default()
            }

            fn describe_range(range: &Self::Range) -> String {
                let options: Vec<String> = range.iter().map(|o| o.describe()).collect();
                format!("[{}]", options.join(", "))
            }
        }
    };
}

categorical_trait!(String);
categorical_trait!(bool);

/// The full, written-out form of a trait. `range` may be left out, in which case the range is
/// the default value alone, as in the bare-value shorthand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "T: TraitValue")]
#[schemars(bound = "T: TraitValue")]
struct TraitFull<T: TraitValue> {
    default: T,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    range: Option<T::Range>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    units: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    variation: Option<Variation>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    heritable: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    mutable: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    visible: bool,
}

/// A declared setting: a default, a range, and how individuals vary (contract 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Trait<T: TraitValue> {
    pub default: T,
    pub range: T::Range,
    pub units: Option<String>,
    pub variation: Option<Variation>,
    pub heritable: bool,
    pub mutable: bool,
    pub visible: bool,
    /// Whether the trait was written as a bare value. Preserved so a file round-trips.
    pub shorthand: bool,
}

impl<T: TraitValue> Trait<T> {
    /// A trait fixed at one value, as the bare-value shorthand declares.
    pub fn fixed(value: T) -> Self {
        Trait {
            range: value.single_range(),
            default: value,
            units: None,
            variation: None,
            heritable: false,
            mutable: false,
            visible: false,
            shorthand: true,
        }
    }

    /// A trait with a default and a range, and nothing else.
    pub fn ranged(default: T, range: T::Range) -> Self {
        Trait {
            default,
            range,
            units: None,
            variation: None,
            heritable: false,
            mutable: false,
            visible: false,
            shorthand: false,
        }
    }

    /// The value this trait has for the individual it describes. After
    /// [`Trait::sample_in_place`], this is the sampled value.
    pub fn value(&self) -> &T {
        &self.default
    }

    /// Whether individuals vary on this trait.
    pub fn varies(&self) -> bool {
        self.variation.is_some()
    }

    /// Draw one individual's value: the default when there is no variation, otherwise a draw
    /// from the variation, clipped to the range.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        match &self.variation {
            None => self.default.clone(),
            Some(variation) => {
                let drawn = T::sample(&self.default, variation, rng)
                    .expect("validated traits vary only when numeric");
                drawn.clip(&self.range)
            }
        }
    }

    /// Replace the default with a sampled value and drop the variation, so the trait now
    /// describes one individual. Returns the sampled value.
    pub fn sample_in_place<R: Rng + ?Sized>(&mut self, rng: &mut R) -> T {
        let value = self.sample(rng);
        self.default = value.clone();
        self.variation = None;
        value
    }

    pub fn validate(&self, ctx: &Ctx) -> Result<()> {
        T::check_range(&self.range).map_err(|m| ctx.child("range").error(m))?;
        ctx.child("default").check(
            self.default.in_range(&self.range),
            format!(
                "default {} is outside the range {}",
                self.default.describe(),
                T::describe_range(&self.range)
            ),
        )?;
        if let Some(variation) = &self.variation {
            let vctx = ctx.child("variation");
            vctx.check(T::NUMERIC, "variation applies to numeric traits only")?;
            variation.validate(&vctx)?;
        }
        if self.units.is_some() {
            ctx.child("units")
                .check(T::NUMERIC, "units apply to numeric traits only")?;
        }
        Ok(())
    }
}

impl<T: TraitValue> From<TraitFull<T>> for Trait<T> {
    fn from(full: TraitFull<T>) -> Self {
        let range = full.range.unwrap_or_else(|| full.default.single_range());
        Trait {
            default: full.default,
            range,
            units: full.units,
            variation: full.variation,
            heritable: full.heritable,
            mutable: full.mutable,
            visible: full.visible,
            shorthand: false,
        }
    }
}

impl<T: TraitValue> Serialize for Trait<T> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        if self.shorthand {
            return self.default.serialize(serializer);
        }
        TraitFull {
            default: self.default.clone(),
            range: Some(self.range.clone()),
            units: self.units.clone(),
            variation: self.variation.clone(),
            heritable: self.heritable,
            mutable: self.mutable,
            visible: self.visible,
        }
        .serialize(serializer)
    }
}

impl<'de, T: TraitValue> Deserialize<'de> for Trait<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        if value.is_object() {
            let full: TraitFull<T> =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(full.into())
        } else {
            let bare: T = serde_json::from_value(value).map_err(|e| {
                serde::de::Error::custom(format!(
                    "{e} (a trait is a bare value, or a mapping with `default` and `range`)"
                ))
            })?;
            Ok(Trait::fixed(bare))
        }
    }
}

impl<T: TraitValue> JsonSchema for Trait<T> {
    fn schema_name() -> Cow<'static, str> {
        format!("Trait_for_{}", T::schema_name()).into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let bare = generator.subschema_for::<T>();
        let full = generator.subschema_for::<TraitFull<T>>();
        json_schema!({
            "description": "A trait: either a bare value, or a mapping with `default` and optional `range`, `units`, `variation`, `heritable`, `mutable`, and `visible`. Without `range`, the range is the default alone.",
            "anyOf": [bare, full]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yaml::from_str;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::path::PathBuf;

    fn file() -> PathBuf {
        "test.yaml".into()
    }

    #[test]
    fn bare_value_is_shorthand() {
        let t: Trait<f64> = from_str("70\n", &file()).unwrap();
        assert_eq!(t.default, 70.0);
        assert_eq!(t.range, [70.0, 70.0]);
        assert!(t.shorthand);
        assert_eq!(serde_json::to_value(&t).unwrap(), serde_json::json!(70.0));
    }

    #[test]
    fn full_form_round_trips() {
        let text = "{default: 70, range: [40, 110], units: kg, variation: {normal: {sd: 10}}, visible: true}\n";
        let t: Trait<f64> = from_str(text, &file()).unwrap();
        assert_eq!(t.units.as_deref(), Some("kg"));
        assert!(t.visible && !t.heritable);
        let json = serde_json::to_value(&t).unwrap();
        assert_eq!(
            json["variation"],
            serde_json::json!({"normal": {"sd": 10.0}})
        );
        assert!(json.get("heritable").is_none());
    }

    #[test]
    fn full_form_without_range_is_fixed_at_the_default() {
        let t: Trait<f64> = from_str("{default: 2.0, units: s}\n", &file()).unwrap();
        assert_eq!(t.range, [2.0, 2.0]);
        assert!(!t.shorthand);
        assert!(t.validate(&Ctx::new(&file())).is_ok());
    }

    #[test]
    fn unknown_trait_field_is_an_error() {
        let err = from_str::<Trait<f64>>("{default: 1, range: [0, 2], colour: red}\n", &file())
            .unwrap_err();
        assert!(err.to_string().contains("unknown field `colour`"), "{err}");
    }

    #[test]
    fn categorical_trait_uses_option_lists() {
        let t: Trait<String> =
            from_str("{default: always, range: [always, awake]}\n", &file()).unwrap();
        assert!(t.validate(&Ctx::new(&file())).is_ok());
        let bad: Trait<String> =
            from_str("{default: never, range: [always, awake]}\n", &file()).unwrap();
        let err = bad.validate(&Ctx::new(&file())).unwrap_err();
        assert_eq!(err.field_path(), Some("default"));
    }

    #[test]
    fn validation_catches_bad_ranges_and_defaults() {
        let t = Trait::ranged(5.0, [0.0, 1.0]);
        assert_eq!(
            t.validate(&Ctx::new(&file())).unwrap_err().field_path(),
            Some("default")
        );
        let t = Trait::ranged(0.5, [1.0, 0.0]);
        assert_eq!(
            t.validate(&Ctx::new(&file())).unwrap_err().field_path(),
            Some("range")
        );
        let mut t = Trait::ranged(String::from("a"), vec![String::from("a")]);
        t.variation = Some(Variation::Normal { sd: 1.0 });
        assert_eq!(
            t.validate(&Ctx::new(&file())).unwrap_err().field_path(),
            Some("variation")
        );
    }

    #[test]
    fn sampling_is_deterministic_and_clipped() {
        let mut t = Trait::ranged(70.0, [60.0, 80.0]);
        t.variation = Some(Variation::Normal { sd: 100.0 });
        let mut a = ChaCha8Rng::from_seed([7; 32]);
        let mut b = ChaCha8Rng::from_seed([7; 32]);
        let xs: Vec<f64> = (0..20).map(|_| t.sample(&mut a)).collect();
        let ys: Vec<f64> = (0..20).map(|_| t.sample(&mut b)).collect();
        assert_eq!(xs, ys);
        assert!(xs.iter().all(|x| (60.0..=80.0).contains(x)));
        assert!(
            xs.iter().any(|x| *x == 60.0 || *x == 80.0),
            "sd 100 should hit the clip"
        );
        let fixed = Trait::fixed(3_i64);
        assert_eq!(fixed.sample(&mut a), 3);
    }

    #[test]
    fn sample_in_place_freezes_the_individual() {
        let mut t = Trait::ranged(1.7, [1.4, 2.0]);
        t.variation = Some(Variation::Uniform { half_width: 0.1 });
        let mut rng = ChaCha8Rng::from_seed([1; 32]);
        let v = t.sample_in_place(&mut rng);
        assert_eq!(*t.value(), v);
        assert!(!t.varies());
    }
}
