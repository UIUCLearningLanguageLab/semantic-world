//! Registered effects (addition A3). New effects are added here, never as special cases
//! elsewhere.

use sw_core::World;
use sw_core::components::{Agent, Id, Needs, Stocks};
use sw_schema::expr::Effect;

use crate::context::{Bindings, boolean, number, symbol};
use crate::error::{Result, RuleError};

/// When an effect applies while an action lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timing {
    /// Once, when the action completes (or at once for an instantaneous action).
    Once,
    /// Every tick while the action lasts, scaled by the tick's duration.
    Rate,
}

pub struct EffectDef {
    pub name: &'static str,
    pub args: &'static [&'static str],
    pub timing: Timing,
}

pub const EFFECTS: &[EffectDef] = &[
    EffectDef {
        name: "change_need",
        args: &["entity", "need", "by"],
        timing: Timing::Once,
    },
    EffectDef {
        name: "change_need_rate",
        args: &["entity", "need", "per_second"],
        timing: Timing::Rate,
    },
    EffectDef {
        name: "take_stock",
        args: &["entity", "name", "amount"],
        timing: Timing::Once,
    },
    EffectDef {
        name: "add_stock",
        args: &["entity", "name", "amount", "up_to"],
        timing: Timing::Once,
    },
    EffectDef {
        name: "set_state",
        args: &["entity", "state", "value"],
        timing: Timing::Once,
    },
];

pub fn lookup(name: &str) -> Option<&'static EffectDef> {
    EFFECTS.iter().find(|e| e.name == name)
}

/// Check an effect against the registry: it exists and has every argument.
pub fn check(effect: &Effect, where_: &str) -> Result<&'static EffectDef> {
    let def = lookup(&effect.name)
        .ok_or_else(|| RuleError::Invalid(format!("{where_}: unknown effect `{}`", effect.name)))?;
    for arg in def.args {
        if effect.arg(arg).is_none() {
            return Err(RuleError::Invalid(format!(
                "{where_}: `{}` needs the argument `{arg}`",
                effect.name
            )));
        }
    }
    for arg in effect.args.keys() {
        if !def.args.contains(&arg.as_str()) {
            return Err(RuleError::Invalid(format!(
                "{where_}: `{}` has no argument `{arg}`",
                effect.name
            )));
        }
    }
    Ok(def)
}

pub fn timing(effect: &Effect) -> Timing {
    lookup(&effect.name).map_or(Timing::Once, |d| d.timing)
}

/// Apply one effect. `dt` is the tick's duration, used by rate effects.
pub fn apply(world: &mut World, bindings: &Bindings, effect: &Effect, dt: f64) -> Result<()> {
    let arg = |name: &str| {
        effect
            .arg(name)
            .ok_or_else(|| RuleError::Arg(format!("{}: missing `{name}`", effect.name)))
    };
    let entity_arg = arg("entity")?;
    let entity = bindings.resolve(match entity_arg {
        sw_schema::expr::Arg::Ref(r) if r.path.is_empty() => r.root,
        other => {
            return Err(RuleError::Arg(format!(
                "{}: `{other}` is not an entity",
                effect.name
            )));
        }
    })?;
    let id = world.ecs().get::<Id>(entity).expect("id").name.clone();
    match effect.name.as_str() {
        "change_need" | "change_need_rate" => {
            let need = symbol(arg("need")?)?.to_string();
            let delta = if effect.name == "change_need" {
                number(world, bindings, arg("by")?)?
            } else {
                number(world, bindings, arg("per_second")?)? * dt
            };
            let mut needs = world
                .ecs_mut()
                .get_mut::<Needs>(entity)
                .ok_or_else(|| RuleError::Arg(format!("{id} has no needs")))?;
            let state = needs
                .get_mut(&need)
                .ok_or_else(|| RuleError::Arg(format!("{id} has no need `{need}`")))?;
            state.value = (state.value + delta).clamp(0.0, 1.0);
        }
        "take_stock" | "add_stock" => {
            let name = symbol(arg("name")?)?.to_string();
            let amount = number(world, bindings, arg("amount")?)?.round() as i64;
            let up_to = if effect.name == "add_stock" {
                Some(number(world, bindings, arg("up_to")?)?.round() as i64)
            } else {
                None
            };
            let mut stocks = world.ecs_mut().get_mut::<Stocks>(entity).expect("stocks");
            let stock = stocks
                .0
                .get_mut(&name)
                .ok_or_else(|| RuleError::Arg(format!("{id} has no stock `{name}`")))?;
            stock.count = match up_to {
                None => (stock.count - amount).max(0),
                Some(limit) => (stock.count + amount)
                    .min(limit)
                    .max(stock.count.min(limit)),
            };
        }
        "set_state" => {
            let state = symbol(arg("state")?)?;
            let value = boolean(arg("value")?)?;
            match state {
                "asleep" => {
                    let collapsed = {
                        let mut agent = world
                            .ecs_mut()
                            .get_mut::<Agent>(entity)
                            .ok_or_else(|| RuleError::Arg(format!("{id} is not an agent")))?;
                        // A collapse is forced sleep that a rule cannot end.
                        agent.asleep = value || agent.collapsed();
                        agent.collapsed()
                    };
                    if value && !collapsed {
                        world.stop(&id)?;
                    }
                }
                other => {
                    return Err(RuleError::Arg(format!(
                        "set_state: `{other}` is not a settable state (only `asleep`)"
                    )));
                }
            }
        }
        other => return Err(RuleError::Invalid(format!("unknown effect `{other}`"))),
    }
    Ok(())
}
