//! Rule files (contract 2, with addition A3): actions, processes, and the proposition
//! vocabulary. Predicate and effect names are not checked here; the rule engine's registries
//! check them, so that new predicates and effects can be added as modules.

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::entity::{Params, check_identifier};
use crate::error::{Ctx, Result};
use crate::expr::{Call, Effect};
use crate::traits::Trait;

/// One rules file. A world may load several; their sections are merged by name.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RulesFile {
    /// Actions, in the order the action manifest lists them.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub actions: IndexMap<String, ActionSpec>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub processes: IndexMap<String, ProcessSpec>,
    /// The proposition vocabulary: each predicate and the generic detector that computes it.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub predicates: IndexMap<String, PredicateSpec>,
}

/// One action type (contract 2, addition A3, addition A4).
///
/// An action is durative when it has a `duration` or an `until` condition. Its `effects`
/// apply when the action completes; rate effects among them apply every tick while the action
/// lasts. `on_start` applies when the action starts, and `on_end` when the action ends for
/// any reason, completion or interruption, after `effects`. An action whose effect cannot be
/// written as data names a registered rule `module` and passes it `params`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ActionSpec {
    /// The actuator the action belongs to. An agent without the actuator lacks the action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actuator: Option<String>,
    /// Argument slots, in manifest order.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub args: IndexMap<String, ArgSlot>,
    /// Preconditions. All must hold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pre: Vec<Call>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub on_start: Vec<Effect>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<Effect>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub on_end: Vec<Effect>,
    /// Maximum duration in simulated seconds. Absent with no `until`: instantaneous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<Trait<f64>>,
    /// Ends the action early when all of these hold. Checked every tick.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub until: Vec<Call>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub params: Params,
    /// Energy cost, 0 in milestone 1: fatigue carries the cost of activity instead.
    #[serde(default = "zero_trait")]
    pub energy_cost: Trait<f64>,
    /// A sound event the action emits. Parsed for the contract's format; ears are not in
    /// milestone 1, so no sound is produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound: Option<SoundSpec>,
}

fn zero_trait() -> Trait<f64> {
    Trait::fixed(0.0)
}

impl Default for ActionSpec {
    fn default() -> Self {
        ActionSpec {
            actuator: None,
            args: IndexMap::new(),
            pre: Vec::new(),
            on_start: Vec::new(),
            effects: Vec::new(),
            on_end: Vec::new(),
            duration: None,
            until: Vec::new(),
            module: None,
            params: Params::new(),
            energy_cost: zero_trait(),
            sound: None,
        }
    }
}

impl ActionSpec {
    /// Whether the action lasts beyond its decision point.
    pub fn is_durative(&self) -> bool {
        self.duration.is_some() || !self.until.is_empty()
    }

    fn validate(&self, ctx: &Ctx) -> Result<()> {
        if let Some(actuator) = &self.actuator {
            check_identifier(&ctx.child("actuator"), actuator)?;
        }
        for (name, slot) in &self.args {
            let sctx = ctx.child("args").child(name);
            check_identifier(&sctx, name)?;
            slot.validate(&sctx)?;
        }
        if let Some(duration) = &self.duration {
            let dctx = ctx.child("duration");
            duration.validate(&dctx)?;
            dctx.check(duration.range[0] > 0.0, "duration must be positive")?;
        }
        if let Some(module) = &self.module {
            check_identifier(&ctx.child("module"), module)?;
        } else {
            ctx.child("params").check(
                self.params.is_empty(),
                "params belong to a module; this action names none",
            )?;
        }
        let ectx = ctx.child("energy_cost");
        self.energy_cost.validate(&ectx)?;
        ectx.check(
            self.energy_cost.range[0] >= 0.0,
            "energy cost cannot be negative",
        )?;
        if let Some(sound) = &self.sound {
            let sctx = ctx.child("sound");
            check_identifier(&sctx.child("event"), &sound.event)?;
            sctx.child("loudness").check(
                (0.0..=1.0).contains(&sound.loudness),
                "loudness is a fraction, 0 to 1",
            )?;
        }
        Ok(())
    }
}

/// One argument slot of an action: a continuous range, or a target.
///
/// Written as `{range: [lo, hi], units: rad}` for a continuous argument, or as
/// `{has: edible, optional: true}` for a target that must carry a tag.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArgSlot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
    /// For a target slot: the tag the target must carry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has: Option<String>,
    /// For a target slot: whether the target may be left out, in which case the nearest
    /// qualifying entity in front of the agent is used.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
}

impl ArgSlot {
    pub fn is_target(&self) -> bool {
        self.has.is_some()
    }

    fn validate(&self, ctx: &Ctx) -> Result<()> {
        match (&self.range, &self.has) {
            (Some(range), None) => {
                ctx.child("range").check(
                    range[0].is_finite() && range[1].is_finite() && range[0] < range[1],
                    "range must be [min, max] with min below max",
                )?;
                ctx.child("optional")
                    .check(!self.optional, "only target slots can be optional")
            }
            (None, Some(tag)) => {
                check_identifier(&ctx.child("has"), tag)?;
                ctx.child("units")
                    .check(self.units.is_none(), "a target slot has no units")
            }
            (Some(_), Some(_)) => Err(ctx.error("a slot is a range or a target, not both")),
            (None, None) => Err(ctx.error("a slot needs `range` or `has`")),
        }
    }
}

/// A sound event an action emits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SoundSpec {
    pub event: String,
    pub loudness: f64,
}

/// Something that happens in the world without any action, on a schedule (contract 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProcessSpec {
    pub applies_to: Selector,
    /// Period in simulated seconds, counted per entity from the start of the run.
    pub every: Trait<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<Effect>,
}

impl ProcessSpec {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        check_identifier(&ctx.child("applies_to").child("has"), &self.applies_to.has)?;
        let ectx = ctx.child("every");
        self.every.validate(&ectx)?;
        ectx.check(self.every.range[0] > 0.0, "the period must be positive")?;
        ctx.child("effects").check(
            !self.effects.is_empty(),
            "a process needs at least one effect",
        )
    }
}

/// Selects the entities a process applies to: those carrying a tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    pub has: String,
}

/// One predicate of the proposition vocabulary (contract 5), computed by a named detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PredicateSpec {
    /// Argument names, in PDDL order. A trailing numeric argument makes a fluent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// The registered detector that computes the predicate.
    pub detector: String,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub params: Params,
}

impl PredicateSpec {
    fn validate(&self, ctx: &Ctx) -> Result<()> {
        for (i, arg) in self.args.iter().enumerate() {
            check_identifier(&ctx.child("args").child(i.to_string()), arg)?;
        }
        check_identifier(&ctx.child("detector"), &self.detector)
    }
}

impl RulesFile {
    pub fn validate(&self, ctx: &Ctx) -> Result<()> {
        for (name, action) in &self.actions {
            let actx = ctx.child("actions").child(name);
            check_identifier(&actx, name)?;
            action.validate(&actx)?;
        }
        for (name, process) in &self.processes {
            let pctx = ctx.child("processes").child(name);
            check_identifier(&pctx, name)?;
            process.validate(&pctx)?;
        }
        for (name, predicate) in &self.predicates {
            let pctx = ctx.child("predicates").child(name);
            check_identifier(&pctx, name)?;
            predicate.validate(&pctx)?;
        }
        Ok(())
    }

    /// Merge another file's sections into this one. A name declared twice is an error,
    /// reported against `ctx`.
    pub fn merge(&mut self, other: RulesFile, ctx: &Ctx) -> Result<()> {
        for (name, action) in other.actions {
            if self.actions.insert(name.clone(), action).is_some() {
                return Err(ctx
                    .child("actions")
                    .child(&name)
                    .error("declared in more than one rules file"));
            }
        }
        for (name, process) in other.processes {
            if self.processes.insert(name.clone(), process).is_some() {
                return Err(ctx
                    .child("processes")
                    .child(&name)
                    .error("declared in more than one rules file"));
            }
        }
        for (name, predicate) in other.predicates {
            if self.predicates.insert(name.clone(), predicate).is_some() {
                return Err(ctx
                    .child("predicates")
                    .child(&name)
                    .error("declared in more than one rules file"));
            }
        }
        Ok(())
    }
}
