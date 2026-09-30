//! Actions compiled from the rules file: manifests, choices, targeting, masks, and the state
//! of a durative action.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Entity;
use serde::{Deserialize, Serialize};
use sw_core::World;
use sw_schema::expr::Call;
use sw_schema::rules::ActionSpec;

use crate::context::{Bindings, Ctx};
use crate::effects::{self, Timing};
use crate::error::{Result, RuleError};
use crate::modules;
use crate::predicates;

/// One action type, checked against the registries.
#[derive(Debug, Clone)]
pub struct ActionDef {
    pub name: String,
    pub spec: ActionSpec,
    /// Continuous argument slots, in manifest order, with their ranges.
    pub numbers: Vec<(String, [f64; 2])>,
    /// The target slot: its name, the tag a target must carry, and whether it is optional.
    pub target: Option<TargetSlot>,
    /// Maximum duration in ticks, for a durative action with a `duration`.
    pub duration_ticks: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetSlot {
    pub name: String,
    pub has: String,
    pub optional: bool,
}

impl ActionDef {
    /// Compile and check one action.
    pub fn compile(name: &str, spec: &ActionSpec, tick_s: f64) -> Result<ActionDef> {
        let where_ = format!("actions.{name}");
        for (i, call) in spec.pre.iter().enumerate() {
            predicates::check(call, &format!("{where_}.pre[{i}]"))?;
        }
        for (i, call) in spec.until.iter().enumerate() {
            predicates::check(call, &format!("{where_}.until[{i}]"))?;
        }
        for (list, label) in [(&spec.on_start, "on_start"), (&spec.on_end, "on_end")] {
            for (i, effect) in list.iter().enumerate() {
                let def = effects::check(effect, &format!("{where_}.{label}[{i}]"))?;
                if def.timing == Timing::Rate {
                    return Err(RuleError::Invalid(format!(
                        "{where_}.{label}[{i}]: `{}` is a rate effect and belongs in `effects`",
                        effect.name
                    )));
                }
            }
        }
        for (i, effect) in spec.effects.iter().enumerate() {
            let def = effects::check(effect, &format!("{where_}.effects[{i}]"))?;
            if def.timing == Timing::Rate && !spec.is_durative() {
                return Err(RuleError::Invalid(format!(
                    "{where_}.effects[{i}]: `{}` is a rate effect, but the action is instantaneous",
                    effect.name
                )));
            }
        }
        let mut numbers = Vec::new();
        let mut target = None;
        for (slot_name, slot) in &spec.args {
            if let Some(has) = &slot.has {
                if target.is_some() {
                    return Err(RuleError::Invalid(format!(
                        "{where_}.args: two target slots"
                    )));
                }
                target = Some(TargetSlot {
                    name: slot_name.clone(),
                    has: has.clone(),
                    optional: slot.optional,
                });
            } else if let Some(range) = slot.range {
                numbers.push((slot_name.clone(), range));
            }
        }
        if let Some(module) = &spec.module {
            let slots: Vec<&str> = numbers.iter().map(|(n, _)| n.as_str()).collect();
            modules::check(module, &spec.params, &slots, &where_)?;
            if spec.is_durative() || !spec.effects.is_empty() {
                return Err(RuleError::Invalid(format!(
                    "{where_}: a module action is instantaneous and has no data effects"
                )));
            }
        }
        let duration_ticks = spec
            .duration
            .as_ref()
            .map(|d| (d.value() / tick_s).round().max(1.0) as u64);
        Ok(ActionDef {
            name: name.to_string(),
            spec: spec.clone(),
            numbers,
            target,
            duration_ticks,
        })
    }

    pub fn is_durative(&self) -> bool {
        self.spec.is_durative()
    }

    /// Whether an agent with these actuators has this action.
    pub fn available_to(&self, actuators: &[&str]) -> bool {
        self.spec
            .actuator
            .as_ref()
            .is_none_or(|a| actuators.contains(&a.as_str()))
    }

    pub fn once_effects(&self) -> impl Iterator<Item = &sw_schema::expr::Effect> {
        self.spec
            .effects
            .iter()
            .filter(|e| effects::timing(e) == Timing::Once)
    }

    pub fn rate_effects(&self) -> impl Iterator<Item = &sw_schema::expr::Effect> {
        self.spec
            .effects
            .iter()
            .filter(|e| effects::timing(e) == Timing::Rate)
    }
}

/// One argument slot as the action manifest lists it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArgInfo {
    Continuous {
        name: String,
        range: [f64; 2],
        #[serde(skip_serializing_if = "Option::is_none")]
        units: Option<String>,
    },
    Target {
        name: String,
        has: String,
        optional: bool,
    },
}

/// One action type as the action manifest lists it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionInfo {
    pub name: String,
    pub durative: bool,
    pub args: Vec<ArgInfo>,
}

/// The action manifest of one agent (contract 3): every action type it has, in a fixed
/// order, with its argument slots and their ranges. The action mask follows this order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionManifest {
    pub agent: String,
    pub actions: Vec<ActionInfo>,
}

impl ActionManifest {
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.actions.iter().position(|a| a.name == name)
    }
}

impl ActionDef {
    pub fn info(&self) -> ActionInfo {
        let mut args = Vec::new();
        for (slot_name, slot) in &self.spec.args {
            if let Some(has) = &slot.has {
                args.push(ArgInfo::Target {
                    name: slot_name.clone(),
                    has: has.clone(),
                    optional: slot.optional,
                });
            } else if let Some(range) = slot.range {
                args.push(ArgInfo::Continuous {
                    name: slot_name.clone(),
                    range,
                    units: slot.units.clone(),
                });
            }
        }
        ActionInfo {
            name: self.name.clone(),
            durative: self.is_durative(),
            args,
        }
    }
}

/// What an agent chose: an action type, its continuous arguments, and optionally a target
/// by ID. Without a target ID, the target is egocentric: the nearest qualifying object in
/// front of the agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionChoice {
    pub name: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub numbers: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

impl ActionChoice {
    pub fn noop() -> Self {
        ActionChoice::named("noop")
    }

    pub fn named(name: &str) -> Self {
        ActionChoice {
            name: name.to_string(),
            numbers: BTreeMap::new(),
            target: None,
        }
    }

    pub fn with(mut self, arg: &str, value: f64) -> Self {
        self.numbers.insert(arg.to_string(), value);
        self
    }

    pub fn targeting(mut self, id: &str) -> Self {
        self.target = Some(id.to_string());
        self
    }
}

/// A durative action in progress.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ongoing {
    pub action: String,
    pub target: Option<String>,
    pub started_tick: u64,
    /// The tick at which the action completes by duration, if it has one.
    pub ends_at_tick: Option<u64>,
}

/// Why an action cannot start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The named precondition failed (with the target it was tried against, if any).
    Precondition(String),
    /// No target in reach meets the preconditions.
    NoTarget,
    /// The slot needs a target and none was given.
    TargetRequired,
    /// The given target ID does not exist or lacks the tag.
    BadTarget(String),
}

impl Refusal {
    pub fn reason(&self) -> String {
        match self {
            Refusal::Precondition(p) => format!("precondition failed: {p}"),
            Refusal::NoTarget => "no target in reach".into(),
            Refusal::TargetRequired => "a target is required".into(),
            Refusal::BadTarget(m) => m.clone(),
        }
    }
}

/// Resolve the target of an action for an agent and check the preconditions. Returns the
/// bound target, if the action has one.
///
/// Egocentric targeting takes the nearest entity carrying the slot's tag that satisfies every
/// precondition, by surface gap, ties broken by creation order.
pub fn resolve(
    world: &World,
    def: &ActionDef,
    agent: Entity,
    choice_target: Option<&str>,
    interrupting: bool,
) -> std::result::Result<Option<Entity>, Refusal> {
    let ctx_for = |target: Option<Entity>| {
        let ctx = Ctx::new(world, Bindings::agent(agent).with_target(target));
        if interrupting {
            ctx.interrupting()
        } else {
            ctx
        }
    };
    let check = |target: Option<Entity>| -> std::result::Result<(), Refusal> {
        match predicates::all_hold(&ctx_for(target), &def.spec.pre) {
            Ok(None) => Ok(()),
            Ok(Some(call)) => Err(Refusal::Precondition(call.to_string())),
            Err(e) => Err(Refusal::Precondition(e.to_string())),
        }
    };
    let Some(slot) = &def.target else {
        check(None)?;
        return Ok(None);
    };
    if let Some(id) = choice_target {
        let entity = world
            .entity(id)
            .ok_or_else(|| Refusal::BadTarget(format!("unknown target `{id}`")))?;
        let ctx = ctx_for(None);
        if !ctx.tags(entity).has(&slot.has) {
            return Err(Refusal::BadTarget(format!("`{id}` is not `{}`", slot.has)));
        }
        check(Some(entity))?;
        return Ok(Some(entity));
    }
    if !slot.optional {
        return Err(Refusal::TargetRequired);
    }
    let ctx = ctx_for(None);
    let mut best: Option<(f64, Entity)> = None;
    for &candidate in &world.registry().entities {
        if candidate == agent || !ctx.tags(candidate).has(&slot.has) {
            continue;
        }
        if check(Some(candidate)).is_err() {
            continue;
        }
        let gap = predicates::surface_gap(&ctx, agent, candidate);
        if best.is_none_or(|(g, _)| gap < g) {
            best = Some((gap, candidate));
        }
    }
    match best {
        Some((_, entity)) => Ok(Some(entity)),
        None => Err(Refusal::NoTarget),
    }
}

/// Whether the preconditions of an `until` list hold for an ongoing action.
pub fn until_holds(
    world: &World,
    def: &ActionDef,
    agent: Entity,
    target: Option<Entity>,
) -> Result<bool> {
    if def.spec.until.is_empty() {
        return Ok(false);
    }
    let ctx = Ctx::new(world, Bindings::agent(agent).with_target(target));
    Ok(predicates::all_hold(&ctx, &def.spec.until)?.is_none())
}

/// The calls of a precondition list, for tests and documentation.
pub fn preconditions(def: &ActionDef) -> &[Call] {
    &def.spec.pre
}
