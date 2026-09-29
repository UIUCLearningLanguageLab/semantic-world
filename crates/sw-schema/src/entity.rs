//! Entity type files (contract 1, with additions A1 and A2).
//!
//! An entity type names up to four components: body, sensors, actuators, and nervous system.
//! Every numeric or categorical setting is a [`Trait`]. The `extends` mechanism and population
//! overrides are applied on the generic YAML value before a file is parsed into these types
//! (see [`crate::resolve`]), so a parsed `EntityType` is always complete.

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::categorical_trait;
use crate::error::{Ctx, Result};
use crate::traits::Trait;

/// Free-form parameters of a registered module, checked by the module that registers them.
pub type Params = IndexMap<String, Value>;

fn one() -> u32 {
    1
}

fn is_one(v: &u32) -> bool {
    *v == 1
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Check that a name is a valid identifier: letters, digits, and underscores, not starting
/// with a digit.
pub fn check_identifier(ctx: &Ctx, name: &str) -> Result<()> {
    ctx.check(
        is_identifier(name),
        format!("`{name}` is not a valid name (letters, digits, and underscores only)"),
    )
}

/// One entity type: what the entity is, senses, does, and thinks with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EntityType {
    /// The type's name, which is also the prefix of its entity IDs (`human_0`).
    #[serde(rename = "type")]
    pub name: String,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub version: u32,
    /// The type this one was derived from, if any. Recorded for provenance; the derivation has
    /// already been applied when this struct exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,
    pub body: Body,
    #[serde(default, skip_serializing_if = "Sensors::is_empty")]
    pub sensors: Sensors,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub actuators: IndexMap<String, Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nervous_system: Option<NervousSystem>,
}

/// The body: what the entity is physically (contract 1, additions A1 and A2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Body {
    /// The body plan, a registered module that sets the parts list and the collision shapes.
    pub plan: String,
    /// Asset name. Optional in milestone 1 and not validated against an asset manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<Trait<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<Trait<f64>>,
    /// Fraction of cold rise removed, 0 to 1. Absent means 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insulation: Option<Trait<f64>>,
    /// Radius of the entity's circle on the ground plane, in meters, used for blocking and
    /// for surface distances. Absent means 0: a point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<Trait<f64>>,
    /// Whether the entity blocks movement.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub solid: bool,
    /// The blocking shape of a solid entity.
    #[serde(default, skip_serializing_if = "Collision::is_default")]
    pub collision: Collision,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub needs: IndexMap<String, NeedSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health: Option<HealthSpec>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub stock: IndexMap<String, StockSpec>,
    /// Need changes per unit consumed, by need name.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub provides: IndexMap<String, Trait<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<Placeholder>,
}

/// The blocking shape of a solid entity on the ground plane.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum Collision {
    /// A circle of the body's `radius`. The default.
    #[default]
    Circle,
    /// The walls of the placeholder footprint, open side excepted, `thickness` meters thick.
    Walls { thickness: Trait<f64> },
}

impl Collision {
    fn is_default(&self) -> bool {
        *self == Collision::Circle
    }
}

/// When a need rises (addition A1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RiseWhen {
    Always,
    NightOutsideShelter,
    /// While the agent is awake. Not in the A1 list; see `docs/proposals/`.
    Awake,
}
categorical_trait!(RiseWhen);

/// The agent states usable in rate multipliers (addition A1).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    Running,
    Asleep,
    InShelter,
}

/// One need: a generic mechanism, all of whose fields are traits (addition A1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NeedSpec {
    #[serde(default = "zero_trait")]
    pub initial: Trait<f64>,
    /// Rise rate in normal conditions, per calendar day.
    pub rise_per_day: Trait<f64>,
    #[serde(default = "always_trait")]
    pub rise_when: Trait<RiseWhen>,
    /// Multiply the rise rate while a state holds.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub rise_multipliers: IndexMap<AgentState, Trait<f64>>,
    /// Passive fall rate, per calendar day, when the rise condition does not hold.
    #[serde(default = "zero_trait")]
    pub fall_per_day: Trait<f64>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub fall_multipliers: IndexMap<AgentState, Trait<f64>>,
    /// If true, the rise rate is multiplied by (1 − body.insulation).
    #[serde(default = "false_trait")]
    pub insulation_reduces_rise: Trait<bool>,
    pub at_max: AtMax,
}

fn zero_trait() -> Trait<f64> {
    Trait::fixed(0.0)
}

fn always_trait() -> Trait<RiseWhen> {
    Trait::fixed(RiseWhen::Always)
}

fn false_trait() -> Trait<bool> {
    Trait::fixed(false)
}

/// What happens when a need is at 1.0 (addition A1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AtMax {
    /// Health drains at this rate per calendar hour.
    HealthDrainPerHour(Trait<f64>),
    /// Forced sleep that cannot be interrupted.
    Collapse { duration_hours: Trait<f64> },
}

/// Health (addition A1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HealthSpec {
    #[serde(default = "one_trait")]
    pub initial: Trait<f64>,
    /// Recovery per calendar hour while no need is at max.
    pub recover_per_hour: Trait<f64>,
}

fn one_trait() -> Trait<f64> {
    Trait::fixed(1.0)
}

/// A countable amount held by an entity (addition A2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StockSpec {
    pub initial: Trait<i64>,
    pub max: Trait<i64>,
}

/// Placeholder shapes (addition A2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    Capsule,
    Sphere,
    Cylinder,
    Cone,
    Box,
    Disc,
    /// A box with one open side. The open side faces the entity's facing direction.
    Shelter,
}

/// A color as `#rrggbb`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct Color(pub String);

impl Color {
    /// The red, green, and blue components, 0 to 255.
    pub fn rgb(&self) -> Option<[u8; 3]> {
        let hex = self.0.strip_prefix('#')?;
        if hex.len() != 6 {
            return None;
        }
        let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
        Some([byte(0)?, byte(2)?, byte(4)?])
    }

    fn validate(&self, ctx: &Ctx) -> Result<()> {
        ctx.check(
            self.rgb().is_some(),
            format!("`{}` is not a color of the form #rrggbb", self.0),
        )
    }
}

/// A different color while a stock is at least 1: a bush is red while berries remain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StockedColor {
    pub stock: String,
    pub color: Color,
}

/// One part of a placeholder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlaceholderPart {
    pub shape: Shape,
    /// Full extent in meters along x, y (up), and z.
    pub size: [f64; 3],
    pub color: Color,
    /// The part's center relative to the entity's origin on the ground. Absent means the part
    /// stands on the ground, centered: `[0, size[1] / 2, 0]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_when_stocked: Option<StockedColor>,
}

impl PlaceholderPart {
    /// The part's center relative to the entity's origin.
    pub fn center(&self) -> [f64; 3] {
        self.offset.unwrap_or([0.0, self.size[1] / 2.0, 0.0])
    }

    fn validate(&self, ctx: &Ctx, stock: &IndexMap<String, StockSpec>) -> Result<()> {
        ctx.child("size").check(
            self.size.iter().all(|s| s.is_finite() && *s > 0.0),
            "every size component must be a positive number",
        )?;
        self.color.validate(&ctx.child("color"))?;
        if let Some(stocked) = &self.color_when_stocked {
            let sctx = ctx.child("color_when_stocked");
            stocked.color.validate(&sctx.child("color"))?;
            sctx.child("stock").check(
                stock.contains_key(&stocked.stock),
                format!("`{}` is not a stock of this body", stocked.stock),
            )?;
        }
        Ok(())
    }
}

/// A placeholder: one part, or a list of parts with offsets (addition A2).
///
/// Written either as a single part (`{shape: sphere, size: [...], color: "#..."}`) or as
/// `{parts: [...]}`.
#[derive(Debug, Clone, PartialEq, JsonSchema)]
#[schemars(untagged)]
pub enum Placeholder {
    Single(PlaceholderPart),
    Parts { parts: Vec<PlaceholderPart> },
}

impl Placeholder {
    pub fn parts(&self) -> Vec<&PlaceholderPart> {
        match self {
            Placeholder::Single(part) => vec![part],
            Placeholder::Parts { parts } => parts.iter().collect(),
        }
    }

    fn validate(&self, ctx: &Ctx, stock: &IndexMap<String, StockSpec>) -> Result<()> {
        match self {
            Placeholder::Single(part) => part.validate(ctx, stock),
            Placeholder::Parts { parts } => {
                ctx.child("parts")
                    .check(!parts.is_empty(), "a placeholder needs at least one part")?;
                for (i, part) in parts.iter().enumerate() {
                    part.validate(&ctx.child("parts").child(i.to_string()), stock)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartsOnly {
    parts: Vec<PlaceholderPart>,
}

impl Serialize for Placeholder {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Placeholder::Single(part) => part.serialize(serializer),
            Placeholder::Parts { parts } => PartsOnly {
                parts: parts.clone(),
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Placeholder {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let is_parts = value.get("parts").is_some();
        if is_parts {
            let PartsOnly { parts } =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Placeholder::Parts { parts })
        } else {
            let part: PlaceholderPart =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Placeholder::Single(part))
        }
    }
}

/// The sensors an entity has. Each field is a registered sensor module with its parameters.
/// Sensors designed but not built in milestone 1 parse as free-form parameters and are
/// rejected at validation.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sensors {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eyes: Option<EyesSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interoception: Option<InteroceptionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub touch: Option<TouchSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proprioception: Option<ProprioceptionSpec>,
    /// The propositional sensor. Present means the agent may have it; the experiment's
    /// `views.propositional_sensor` switches it on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propositional: Option<PropositionalSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ears: Option<Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smell: Option<Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taste: Option<Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pain: Option<Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Params>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure: Option<Params>,
}

impl Sensors {
    pub fn is_empty(&self) -> bool {
        *self == Sensors::default()
    }

    /// Names of the sensors present, in the order the manifest lists them.
    pub fn present(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.eyes.is_some() {
            names.push("eyes");
        }
        if self.interoception.is_some() {
            names.push("interoception");
        }
        if self.touch.is_some() {
            names.push("touch");
        }
        if self.proprioception.is_some() {
            names.push("proprioception");
        }
        if self.propositional.is_some() {
            names.push("propositional");
        }
        names
    }
}

/// Eyes: a rendered view from a camera above the ground, along the agent's facing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EyesSpec {
    /// Width and height in pixels.
    pub resolution: [u32; 2],
    pub fov_deg: Trait<f64>,
    pub range_m: Trait<f64>,
    #[serde(default = "default_true")]
    pub color: bool,
    /// Camera height above the ground, in meters.
    #[serde(default = "default_eye_height")]
    pub height_m: Trait<f64>,
}

fn default_true() -> bool {
    true
}

fn default_eye_height() -> Trait<f64> {
    Trait::fixed(1.6)
}

/// Interoception: need levels, and health, as a float block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InteroceptionSpec {
    /// The needs reported, in block order.
    pub needs: Vec<String>,
    /// Whether health is reported after the needs.
    #[serde(default = "default_true")]
    pub health: bool,
    /// Standard deviation of Gaussian noise added to each value, clipped to 0 to 1.
    #[serde(default = "zero_trait")]
    pub noise_sd: Trait<f64>,
}

/// How finely touch is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TouchDetail {
    WholeBody,
    /// Per body part. Not in milestone 1.
    PerPart,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TouchSpec {
    pub detail: TouchDetail,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProprioceptionSpec {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PropositionalSpec {
    /// Facts are reported about entities within this distance, in meters.
    pub range_m: Trait<f64>,
}

/// The nervous system: a registered module name and its parameters. Modules resolve through
/// the registry in `semantic_world.agents`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NervousSystem {
    pub module: String,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub params: Params,
}

/// The actuators milestone 1 builds. Others parse and are rejected at validation.
pub const MILESTONE_1_ACTUATORS: [&str; 3] = ["legs", "arms_and_hands", "mouth"];

impl EntityType {
    /// Check every field, and reject features that are not in milestone 1.
    pub fn validate(&self, ctx: &Ctx) -> Result<()> {
        check_identifier(&ctx.child("type"), &self.name)?;
        ctx.child("version").check(
            self.version == 1,
            format!("unsupported version {}", self.version),
        )?;
        self.body.validate(&ctx.child("body"))?;
        self.sensors.validate(&ctx.child("sensors"), &self.body)?;
        for name in self.actuators.keys() {
            let actx = ctx.child("actuators").child(name);
            if !MILESTONE_1_ACTUATORS.contains(&name.as_str()) {
                return Err(actx.not_in_milestone_1(format!("the actuator `{name}`")));
            }
        }
        if let Some(ns) = &self.nervous_system {
            ns.validate(&ctx.child("nervous_system"))?;
        }
        Ok(())
    }

    /// Draw one individual: sample every trait that varies, in declaration order, and freeze
    /// the result. Returns the sampled trait paths and values, in the same order, for
    /// `individuals.parquet`.
    pub fn sample_individual<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> (EntityType, Vec<(String, Value)>) {
        let mut individual = self.clone();
        let mut sampled = Vec::new();
        individual.body.sample_in_place(rng, "body", &mut sampled);
        individual
            .sensors
            .sample_in_place(rng, "sensors", &mut sampled);
        (individual, sampled)
    }
}

fn sample_trait<T: crate::traits::TraitValue, R: rand::Rng + ?Sized>(
    t: &mut Trait<T>,
    rng: &mut R,
    path: String,
    sampled: &mut Vec<(String, Value)>,
) {
    if t.varies() {
        let value = t.sample_in_place(rng);
        sampled.push((path, serde_json::to_value(value).unwrap_or(Value::Null)));
    }
}

fn sample_opt<T: crate::traits::TraitValue, R: rand::Rng + ?Sized>(
    t: &mut Option<Trait<T>>,
    rng: &mut R,
    path: String,
    sampled: &mut Vec<(String, Value)>,
) {
    if let Some(t) = t {
        sample_trait(t, rng, path, sampled);
    }
}

impl Body {
    /// Whether this body is an agent's: it has needs.
    pub fn has_needs(&self) -> bool {
        !self.needs.is_empty()
    }

    /// The blocking and distance radius, 0 for a point.
    pub fn radius_m(&self) -> f64 {
        self.radius.as_ref().map_or(0.0, |r| *r.value())
    }

    pub fn insulation_value(&self) -> f64 {
        self.insulation.as_ref().map_or(0.0, |r| *r.value())
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    fn validate(&self, ctx: &Ctx) -> Result<()> {
        check_identifier(&ctx.child("plan"), &self.plan)?;
        for (name, t) in [
            ("mass", &self.mass),
            ("height", &self.height),
            ("insulation", &self.insulation),
            ("radius", &self.radius),
        ] {
            if let Some(t) = t {
                let tctx = ctx.child(name);
                t.validate(&tctx)?;
                tctx.check(t.range[0] >= 0.0, format!("{name} cannot be negative"))?;
            }
        }
        if let Some(insulation) = &self.insulation {
            ctx.child("insulation").check(
                insulation.range[1] <= 1.0,
                "insulation is a fraction, 0 to 1",
            )?;
        }
        if let Collision::Walls { thickness } = &self.collision {
            let cctx = ctx.child("collision").child("thickness");
            thickness.validate(&cctx)?;
            cctx.check(thickness.range[0] > 0.0, "wall thickness must be positive")?;
            ctx.child("collision").check(
                self.placeholder.is_some(),
                "wall collision follows the placeholder footprint, so a placeholder is required",
            )?;
        }
        for (i, tag) in self.tags.iter().enumerate() {
            check_identifier(&ctx.child("tags").child(i.to_string()), tag)?;
        }
        for (name, need) in &self.needs {
            let nctx = ctx.child("needs").child(name);
            check_identifier(&nctx, name)?;
            need.validate(&nctx)?;
        }
        match &self.health {
            Some(health) => health.validate(&ctx.child("health"))?,
            None => ctx.child("health").check(
                self.needs.is_empty(),
                "a body with needs must declare health",
            )?,
        }
        for (name, stock) in &self.stock {
            let sctx = ctx.child("stock").child(name);
            check_identifier(&sctx, name)?;
            stock.validate(&sctx)?;
        }
        for (name, t) in &self.provides {
            let pctx = ctx.child("provides").child(name);
            check_identifier(&pctx, name)?;
            t.validate(&pctx)?;
        }
        if let Some(placeholder) = &self.placeholder {
            placeholder.validate(&ctx.child("placeholder"), &self.stock)?;
        }
        Ok(())
    }

    fn sample_in_place<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        prefix: &str,
        sampled: &mut Vec<(String, Value)>,
    ) {
        sample_opt(&mut self.mass, rng, format!("{prefix}.mass"), sampled);
        sample_opt(&mut self.height, rng, format!("{prefix}.height"), sampled);
        sample_opt(
            &mut self.insulation,
            rng,
            format!("{prefix}.insulation"),
            sampled,
        );
        sample_opt(&mut self.radius, rng, format!("{prefix}.radius"), sampled);
        if let Collision::Walls { thickness } = &mut self.collision {
            sample_trait(
                thickness,
                rng,
                format!("{prefix}.collision.thickness"),
                sampled,
            );
        }
        for (name, need) in &mut self.needs {
            need.sample_in_place(rng, &format!("{prefix}.needs.{name}"), sampled);
        }
        if let Some(health) = &mut self.health {
            sample_trait(
                &mut health.initial,
                rng,
                format!("{prefix}.health.initial"),
                sampled,
            );
            sample_trait(
                &mut health.recover_per_hour,
                rng,
                format!("{prefix}.health.recover_per_hour"),
                sampled,
            );
        }
        for (name, stock) in &mut self.stock {
            sample_trait(
                &mut stock.initial,
                rng,
                format!("{prefix}.stock.{name}.initial"),
                sampled,
            );
            sample_trait(
                &mut stock.max,
                rng,
                format!("{prefix}.stock.{name}.max"),
                sampled,
            );
        }
        for (name, t) in &mut self.provides {
            sample_trait(t, rng, format!("{prefix}.provides.{name}"), sampled);
        }
    }
}

impl NeedSpec {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        for (name, t) in [
            ("initial", &self.initial),
            ("rise_per_day", &self.rise_per_day),
            ("fall_per_day", &self.fall_per_day),
        ] {
            let tctx = ctx.child(name);
            t.validate(&tctx)?;
            tctx.check(t.range[0] >= 0.0, format!("{name} cannot be negative"))?;
        }
        ctx.child("initial")
            .check(self.initial.range[1] <= 1.0, "needs range from 0 to 1")?;
        self.rise_when.validate(&ctx.child("rise_when"))?;
        for (state, t) in &self.rise_multipliers {
            let mctx = ctx.child("rise_multipliers").child(state_name(*state));
            t.validate(&mctx)?;
            mctx.check(t.range[0] >= 0.0, "multipliers cannot be negative")?;
        }
        for (state, t) in &self.fall_multipliers {
            let mctx = ctx.child("fall_multipliers").child(state_name(*state));
            t.validate(&mctx)?;
            mctx.check(t.range[0] >= 0.0, "multipliers cannot be negative")?;
        }
        self.insulation_reduces_rise
            .validate(&ctx.child("insulation_reduces_rise"))?;
        match &self.at_max {
            AtMax::HealthDrainPerHour(t) => {
                let actx = ctx.child("at_max").child("health_drain_per_hour");
                t.validate(&actx)?;
                actx.check(t.range[0] >= 0.0, "health drain cannot be negative")?;
            }
            AtMax::Collapse { duration_hours } => {
                let actx = ctx
                    .child("at_max")
                    .child("collapse")
                    .child("duration_hours");
                duration_hours.validate(&actx)?;
                actx.check(
                    duration_hours.range[0] > 0.0,
                    "collapse duration must be positive",
                )?;
            }
        }
        Ok(())
    }

    fn sample_in_place<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        prefix: &str,
        sampled: &mut Vec<(String, Value)>,
    ) {
        sample_trait(&mut self.initial, rng, format!("{prefix}.initial"), sampled);
        sample_trait(
            &mut self.rise_per_day,
            rng,
            format!("{prefix}.rise_per_day"),
            sampled,
        );
        for (state, t) in &mut self.rise_multipliers {
            sample_trait(
                t,
                rng,
                format!("{prefix}.rise_multipliers.{}", state_name(*state)),
                sampled,
            );
        }
        sample_trait(
            &mut self.fall_per_day,
            rng,
            format!("{prefix}.fall_per_day"),
            sampled,
        );
        for (state, t) in &mut self.fall_multipliers {
            sample_trait(
                t,
                rng,
                format!("{prefix}.fall_multipliers.{}", state_name(*state)),
                sampled,
            );
        }
        match &mut self.at_max {
            AtMax::HealthDrainPerHour(t) => sample_trait(
                t,
                rng,
                format!("{prefix}.at_max.health_drain_per_hour"),
                sampled,
            ),
            AtMax::Collapse { duration_hours } => sample_trait(
                duration_hours,
                rng,
                format!("{prefix}.at_max.collapse.duration_hours"),
                sampled,
            ),
        }
    }
}

/// The snake_case name of an agent state, as written in files.
pub fn state_name(state: AgentState) -> &'static str {
    match state {
        AgentState::Running => "running",
        AgentState::Asleep => "asleep",
        AgentState::InShelter => "in_shelter",
    }
}

impl HealthSpec {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        let ictx = ctx.child("initial");
        self.initial.validate(&ictx)?;
        ictx.check(
            self.initial.range[0] >= 0.0 && self.initial.range[1] <= 1.0,
            "health ranges from 0 to 1",
        )?;
        let rctx = ctx.child("recover_per_hour");
        self.recover_per_hour.validate(&rctx)?;
        rctx.check(
            self.recover_per_hour.range[0] >= 0.0,
            "recovery cannot be negative",
        )
    }
}

impl StockSpec {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        let ictx = ctx.child("initial");
        self.initial.validate(&ictx)?;
        ictx.check(self.initial.range[0] >= 0, "stock cannot be negative")?;
        let mctx = ctx.child("max");
        self.max.validate(&mctx)?;
        mctx.check(self.max.range[0] >= 0, "stock cannot be negative")?;
        ictx.check(
            self.initial.value() <= self.max.value(),
            format!(
                "initial stock {} is above the maximum {}",
                self.initial.value(),
                self.max.value()
            ),
        )
    }
}

impl Sensors {
    fn validate(&self, ctx: &Ctx, body: &Body) -> Result<()> {
        for (name, params) in [
            ("ears", &self.ears),
            ("smell", &self.smell),
            ("taste", &self.taste),
            ("pain", &self.pain),
            ("temperature", &self.temperature),
            ("pressure", &self.pressure),
        ] {
            if params.is_some() {
                return Err(ctx
                    .child(name)
                    .not_in_milestone_1(format!("the sensor `{name}`")));
            }
        }
        if let Some(eyes) = &self.eyes {
            let ectx = ctx.child("eyes");
            ectx.child("resolution").check(
                eyes.resolution.iter().all(|r| *r > 0),
                "resolution must be positive",
            )?;
            eyes.fov_deg.validate(&ectx.child("fov_deg"))?;
            ectx.child("fov_deg").check(
                eyes.fov_deg.range[0] > 0.0 && eyes.fov_deg.range[1] < 180.0,
                "the field of view is between 0 and 180 degrees",
            )?;
            eyes.range_m.validate(&ectx.child("range_m"))?;
            ectx.child("range_m")
                .check(eyes.range_m.range[0] > 0.0, "range must be positive")?;
            eyes.height_m.validate(&ectx.child("height_m"))?;
        }
        if let Some(intero) = &self.interoception {
            let ictx = ctx.child("interoception");
            for (i, need) in intero.needs.iter().enumerate() {
                ictx.child("needs").child(i.to_string()).check(
                    body.needs.contains_key(need),
                    format!("`{need}` is not a need of this body"),
                )?;
            }
            intero.noise_sd.validate(&ictx.child("noise_sd"))?;
            ictx.child("noise_sd").check(
                intero.noise_sd.range[0] >= 0.0,
                "noise_sd cannot be negative",
            )?;
        }
        if let Some(touch) = &self.touch
            && touch.detail == TouchDetail::PerPart
        {
            return Err(ctx
                .child("touch")
                .child("detail")
                .not_in_milestone_1("per-part touch"));
        }
        if let Some(prop) = &self.propositional {
            let pctx = ctx.child("propositional").child("range_m");
            prop.range_m.validate(&pctx)?;
            pctx.check(prop.range_m.range[0] > 0.0, "range must be positive")?;
        }
        Ok(())
    }

    fn sample_in_place<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        prefix: &str,
        sampled: &mut Vec<(String, Value)>,
    ) {
        if let Some(eyes) = &mut self.eyes {
            sample_trait(
                &mut eyes.fov_deg,
                rng,
                format!("{prefix}.eyes.fov_deg"),
                sampled,
            );
            sample_trait(
                &mut eyes.range_m,
                rng,
                format!("{prefix}.eyes.range_m"),
                sampled,
            );
            sample_trait(
                &mut eyes.height_m,
                rng,
                format!("{prefix}.eyes.height_m"),
                sampled,
            );
        }
        if let Some(intero) = &mut self.interoception {
            sample_trait(
                &mut intero.noise_sd,
                rng,
                format!("{prefix}.interoception.noise_sd"),
                sampled,
            );
        }
        if let Some(prop) = &mut self.propositional {
            sample_trait(
                &mut prop.range_m,
                rng,
                format!("{prefix}.propositional.range_m"),
                sampled,
            );
        }
    }
}

impl NervousSystem {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        check_identifier(&ctx.child("module"), &self.module)
    }
}
