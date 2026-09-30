//! World settings: regime, map, calendar, light, and where the types and rules are.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::entity::{Params, check_identifier};
use crate::error::{Ctx, Result};
use crate::traits::Trait;

/// A world settings file, or the inline `world` section of an experiment configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorldSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world: Option<String>,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub version: u32,
    #[serde(default)]
    pub regime: Regime,
    /// Directory holding the entity type files `<name>.yaml`, relative to this file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<PathBuf>,
    /// Rules files, relative to this file. Their sections are merged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<PathBuf>,
    pub map: MapSpec,
    #[serde(default)]
    pub calendar: CalendarSpec,
    #[serde(default)]
    pub light: LightSpec,
    /// Scheduled world changes. Parsed for the contract's format; not in milestone 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<Vec<Value>>,
}

fn one() -> u32 {
    1
}

fn is_one(v: &u32) -> bool {
    *v == 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    #[default]
    Scripted,
    /// Physics-driven. Not in milestone 1.
    Generative,
}

/// The map generator, a registered module, and its parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MapSpec {
    pub generator: String,
    /// Extent in meters along x and z. The map is centered on the origin.
    pub size_m: [f64; 2],
    #[serde(default, skip_serializing_if = "indexmap::IndexMap::is_empty")]
    pub params: Params,
}

/// The calendar (contract 6): how simulated seconds map to days.
///
/// Fields are optional so that a partial inline calendar parses; a run needs all of them,
/// which [`WorldSpec::validate`] checks.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CalendarSpec {
    /// Simulated minutes per calendar day.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minutes_per_day: Option<Trait<f64>>,
    /// The fraction of the day, from dawn, that is daylight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub daylight_fraction: Option<Trait<f64>>,
}

/// Light: the dawn and dusk transitions and the night threshold.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LightSpec {
    /// Length of the linear rise at dawn and fall at dusk, in calendar hours, centered on
    /// the day–night boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub twilight_hours: Option<Trait<f64>>,
    /// It is night when the light level is below this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub night_below: Option<Trait<f64>>,
}

/// The calendar and light values a run uses, all present.
#[derive(Debug, Clone, PartialEq)]
pub struct Calendar {
    pub minutes_per_day: f64,
    pub daylight_fraction: f64,
    pub twilight_hours: f64,
    pub night_below: f64,
}

impl Calendar {
    pub fn seconds_per_day(&self) -> f64 {
        self.minutes_per_day * 60.0
    }

    pub fn seconds_per_hour(&self) -> f64 {
        self.seconds_per_day() / 24.0
    }
}

impl WorldSpec {
    /// Check every field and reject features that are not in milestone 1.
    pub fn validate(&self, ctx: &Ctx) -> Result<()> {
        if let Some(name) = &self.world {
            check_identifier(&ctx.child("world"), name)?;
        }
        ctx.child("version").check(
            self.version == 1,
            format!("unsupported version {}", self.version),
        )?;
        if self.regime == Regime::Generative {
            return Err(ctx
                .child("regime")
                .not_in_milestone_1("the generative regime"));
        }
        if self.schedule.is_some() {
            return Err(ctx
                .child("schedule")
                .not_in_milestone_1("scheduled world changes"));
        }
        check_identifier(&ctx.child("map").child("generator"), &self.map.generator)?;
        ctx.child("map").child("size_m").check(
            self.map.size_m.iter().all(|s| s.is_finite() && *s > 0.0),
            "the map size must be positive",
        )?;
        self.calendar_in(ctx)?;
        Ok(())
    }

    /// The calendar values, or an error naming the missing field. Without a file context the
    /// error names the file as `world`; [`WorldSpec::validate_and_calendar`] names the real
    /// file.
    pub fn calendar(&self) -> Result<Calendar> {
        let ctx = Ctx::new(std::path::Path::new("world"));
        self.calendar_in(&ctx)
    }

    fn calendar_in(&self, ctx: &Ctx) -> Result<Calendar> {
        fn required(ctx: &Ctx, field: &str, t: &Option<Trait<f64>>) -> Result<f64> {
            let fctx = ctx.child(field);
            let t = t.as_ref().ok_or_else(|| fctx.error("required for a run"))?;
            t.validate(&fctx)?;
            Ok(*t.value())
        }
        let cctx = ctx.child("calendar");
        let minutes_per_day = required(&cctx, "minutes_per_day", &self.calendar.minutes_per_day)?;
        cctx.child("minutes_per_day")
            .check(minutes_per_day > 0.0, "must be positive")?;
        let daylight_fraction =
            required(&cctx, "daylight_fraction", &self.calendar.daylight_fraction)?;
        cctx.child("daylight_fraction").check(
            (0.0..=1.0).contains(&daylight_fraction),
            "is a fraction, 0 to 1",
        )?;
        let lctx = ctx.child("light");
        let twilight_hours = required(&lctx, "twilight_hours", &self.light.twilight_hours)?;
        lctx.child("twilight_hours").check(
            (0.0..=12.0).contains(&twilight_hours),
            "is between 0 and 12 hours",
        )?;
        let night_below = required(&lctx, "night_below", &self.light.night_below)?;
        lctx.child("night_below").check(
            (0.0..=1.0).contains(&night_below),
            "is a light level, 0 to 1",
        )?;
        Ok(Calendar {
            minutes_per_day,
            daylight_fraction,
            twilight_hours,
            night_below,
        })
    }

    /// Validate with a file context and return the calendar.
    pub fn validate_and_calendar(&self, ctx: &Ctx) -> Result<Calendar> {
        self.validate(ctx)?;
        self.calendar_in(ctx)
    }
}
