//! Experiment configurations (contract 10, with addition A5).

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::entity::{NervousSystem, Params, check_identifier};
use crate::error::{Ctx, Result};
use crate::world::WorldSpec;

/// One run, or one batch of runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    pub experiment: String,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub version: u32,
    pub world: WorldRef,
    /// Types declared on the fly, each a named override of an existing type
    /// (`{type: short_human, extends: human, body: {...}}`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub types: Vec<Value>,
    pub population: Vec<PopulationEntry>,
    #[serde(default)]
    pub views: Views,
    #[serde(default)]
    pub options: Options,
    #[serde(default)]
    pub clock: Clock,
    pub lifetime: Lifetime,
    /// World checkpoints. Parsed for the contract's format; not in milestone 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoints: Option<Value>,
    #[serde(default)]
    pub logging: Logging,
    /// The agent-side reward function the adapters call (contract 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<RewardSpec>,
    /// The experimental design: each key is a dotted path into this configuration, and each
    /// value the list of settings to cross (addition A5).
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub conditions: IndexMap<String, Vec<Value>>,
    /// The master seeds, one run per seed per cell.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seeds: Vec<u64>,
    /// A single master seed, for one run. `seeds` and `seed` may not both be given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

fn one() -> u32 {
    1
}

fn is_one(v: &u32) -> bool {
    *v == 1
}

/// The world: a path to a world file, relative to the experiment file, or an inline world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum WorldRef {
    Path(std::path::PathBuf),
    Inline(Box<WorldSpec>),
}

/// One group of agents or objects placed in the world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PopulationEntry {
    #[serde(rename = "type")]
    pub type_name: String,
    pub count: u32,
    /// Replaces the type's nervous system for this group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nervous_system: Option<NervousSystem>,
    /// Trait overrides for this group, in the shape of a type file.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub overrides: Value,
}

/// Which views agents receive.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Views {
    #[serde(default)]
    pub propositional_sensor: bool,
    /// Heard speech as symbols. Not in milestone 1.
    #[serde(default)]
    pub speech_as_symbols: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImpossibleActions {
    #[default]
    Masked,
    AttemptAndFail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TouchDetailOption {
    #[default]
    WholeBody,
    PerPart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Death {
    #[default]
    EndsLifetime,
    /// Not in milestone 1.
    PersistentSociety,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Options {
    #[serde(default)]
    pub impossible_actions: ImpossibleActions,
    #[serde(default)]
    pub touch_detail: TouchDetailOption,
    #[serde(default)]
    pub death: Death,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClockMode {
    #[default]
    Synchronous,
    /// Not in milestone 1.
    Realtime,
}

/// The clock (contract 6). The defaults are the contract's: 0.1 s ticks, a decision every
/// 2 ticks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Clock {
    #[serde(default)]
    pub mode: ClockMode,
    #[serde(default = "default_tick")]
    pub tick_s: f64,
    #[serde(default = "default_decision")]
    pub decision_every_ticks: u32,
}

fn default_tick() -> f64 {
    0.1
}

fn default_decision() -> u32 {
    2
}

impl Default for Clock {
    fn default() -> Self {
        Clock {
            mode: ClockMode::Synchronous,
            tick_s: default_tick(),
            decision_every_ticks: default_decision(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lifetime {
    /// The run ends after this many calendar days, if any agent is still alive.
    pub default_days: f64,
}

/// How often something is logged: never, every decision or frame, or every `n`th.
///
/// Written as `off`, `all`, `every_decision`, or `every_<n>th` (`every_10th`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogRate {
    Off,
    Every,
    EveryNth(u32),
}

impl LogRate {
    pub fn parse(text: &str) -> std::result::Result<LogRate, String> {
        match text {
            "off" => return Ok(LogRate::Off),
            "all" | "every_decision" | "every_frame" => return Ok(LogRate::Every),
            _ => {}
        }
        if let Some(n) = text
            .strip_prefix("every_")
            .and_then(|s| s.strip_suffix("th"))
            .and_then(|s| s.parse::<u32>().ok())
            && n > 0
        {
            return Ok(LogRate::EveryNth(n));
        }
        Err(format!(
            "`{text}` is not a log rate (off, all, every_decision, or every_<n>th)"
        ))
    }
}

impl std::fmt::Display for LogRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogRate::Off => write!(f, "off"),
            LogRate::Every => write!(f, "all"),
            LogRate::EveryNth(n) => write!(f, "every_{n}th"),
        }
    }
}

impl Serialize for LogRate {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for LogRate {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        LogRate::parse(&text).map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for LogRate {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LogRate".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "off, all, every_decision, or every_<n>th (for example every_10th)",
            "type": "string",
            "pattern": "^(off|all|every_decision|every_frame|every_[1-9][0-9]*th)$"
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Logging {
    /// State snapshots every this many ticks.
    #[serde(default = "default_state_every")]
    pub state_every_ticks: u32,
    /// Rendered images: off by default.
    #[serde(default = "log_off")]
    pub images: LogRate,
    /// Agent internals.
    #[serde(default = "log_every")]
    pub internals: LogRate,
}

fn default_state_every() -> u32 {
    10
}

fn log_off() -> LogRate {
    LogRate::Off
}

fn log_every() -> LogRate {
    LogRate::Every
}

impl Default for Logging {
    fn default() -> Self {
        Logging {
            state_every_ticks: default_state_every(),
            images: LogRate::Off,
            internals: LogRate::Every,
        }
    }
}

/// The agent-side reward function (contract 4), registered by name in `semantic_world.reward`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RewardSpec {
    pub module: String,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub params: Params,
}

impl ExperimentConfig {
    /// Check the configuration's own fields and reject features that are not in milestone 1.
    /// References to types and to the world are checked in [`crate::resolve`].
    pub fn validate(&self, ctx: &Ctx) -> Result<()> {
        check_identifier(&ctx.child("experiment"), &self.experiment)?;
        ctx.child("version").check(
            self.version == 1,
            format!("unsupported version {}", self.version),
        )?;
        ctx.child("population").check(
            !self.population.is_empty(),
            "at least one population entry is needed",
        )?;
        for (i, entry) in self.population.iter().enumerate() {
            let ectx = ctx.child("population").child(i.to_string());
            check_identifier(&ectx.child("type"), &entry.type_name)?;
            ectx.child("count")
                .check(entry.count > 0, "count must be at least 1")?;
            if let Some(ns) = &entry.nervous_system {
                check_identifier(&ectx.child("nervous_system").child("module"), &ns.module)?;
            }
            ectx.child("overrides").check(
                entry.overrides.is_null() || entry.overrides.is_object(),
                "overrides must be a mapping in the shape of a type file",
            )?;
        }
        if self.views.speech_as_symbols {
            return Err(ctx
                .child("views")
                .child("speech_as_symbols")
                .not_in_milestone_1("speech as symbols"));
        }
        if self.options.touch_detail == TouchDetailOption::PerPart {
            return Err(ctx
                .child("options")
                .child("touch_detail")
                .not_in_milestone_1("per-part touch"));
        }
        if self.options.death == Death::PersistentSociety {
            return Err(ctx
                .child("options")
                .child("death")
                .not_in_milestone_1("the persistent-society death option"));
        }
        if self.clock.mode == ClockMode::Realtime {
            return Err(ctx
                .child("clock")
                .child("mode")
                .not_in_milestone_1("the real-time clock mode"));
        }
        let cctx = ctx.child("clock");
        cctx.child("tick_s").check(
            self.clock.tick_s.is_finite() && self.clock.tick_s > 0.0,
            "the tick must be positive",
        )?;
        cctx.child("decision_every_ticks")
            .check(self.clock.decision_every_ticks >= 1, "must be at least 1")?;
        ctx.child("lifetime").child("default_days").check(
            self.lifetime.default_days.is_finite() && self.lifetime.default_days > 0.0,
            "the lifetime must be positive",
        )?;
        if self.checkpoints.is_some() {
            return Err(ctx.child("checkpoints").not_in_milestone_1("checkpoints"));
        }
        ctx.child("logging")
            .child("state_every_ticks")
            .check(self.logging.state_every_ticks >= 1, "must be at least 1")?;
        if let Some(reward) = &self.reward {
            check_identifier(&ctx.child("reward").child("module"), &reward.module)?;
        }
        for (path, values) in &self.conditions {
            let pctx = ctx.child("conditions").child(path);
            pctx.check(!values.is_empty(), "a condition needs at least one setting")?;
        }
        if self.seed.is_some() && !self.seeds.is_empty() {
            return Err(ctx.child("seed").error("give `seed` or `seeds`, not both"));
        }
        ctx.child("seeds").check(
            self.seed.is_some() || !self.seeds.is_empty(),
            "a run needs `seed` or a non-empty `seeds` list",
        )?;
        Ok(())
    }

    /// The master seeds of this configuration's runs, in order.
    pub fn master_seeds(&self) -> Vec<u64> {
        match self.seed {
            Some(seed) => vec![seed],
            None => self.seeds.clone(),
        }
    }
}
