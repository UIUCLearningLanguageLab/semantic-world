//! Registered action modules: actions whose effect cannot be written as data. Each module
//! declares the parameters it reads from the rules file and the arguments it takes.

use std::collections::BTreeMap;

use sw_core::World;
use sw_core::components::Facing;
use sw_core::map::param_f64;
use sw_schema::entity::Params;

use crate::error::{Result, RuleError};

pub struct ModuleDef {
    pub name: &'static str,
    pub params: &'static [&'static str],
    pub args: &'static [&'static str],
}

pub const MODULES: &[ModuleDef] = &[
    ModuleDef {
        name: "kinematic_move",
        params: &["walk_speed_mps", "walk_speed_max", "run_speed_mps"],
        args: &["direction", "speed"],
    },
    ModuleDef {
        name: "turn",
        params: &[],
        args: &["angle"],
    },
];

pub fn lookup(name: &str) -> Option<&'static ModuleDef> {
    MODULES.iter().find(|m| m.name == name)
}

/// Check that an action's module exists, has its parameters, and matches the action's
/// argument slots.
pub fn check(name: &str, params: &Params, slots: &[&str], where_: &str) -> Result<()> {
    let def = lookup(name)
        .ok_or_else(|| RuleError::Invalid(format!("{where_}: unknown action module `{name}`")))?;
    for p in def.params {
        param_f64(params, p, name).map_err(|e| RuleError::Invalid(format!("{where_}: {e}")))?;
    }
    if slots != def.args {
        return Err(RuleError::Invalid(format!(
            "{where_}: module `{name}` takes the arguments {:?}, but the action declares {:?}",
            def.args, slots
        )));
    }
    Ok(())
}

/// Run a module for an agent with resolved numeric arguments.
pub fn run(
    name: &str,
    params: &Params,
    world: &mut World,
    agent_id: &str,
    args: &BTreeMap<String, f64>,
) -> Result<()> {
    let arg = |a: &str| {
        args.get(a)
            .copied()
            .ok_or_else(|| RuleError::Arg(format!("{name}: missing argument `{a}`")))
    };
    let entity = world
        .entity(agent_id)
        .ok_or_else(|| sw_core::CoreError::UnknownAgent(agent_id.to_string()))?;
    let yaw = world.ecs().get::<Facing>(entity).expect("facing").yaw;
    match name {
        "kinematic_move" => {
            let walk_speed = param_f64(params, "walk_speed_mps", name)?;
            let walk_max = param_f64(params, "walk_speed_max", name)?;
            let run_speed = param_f64(params, "run_speed_mps", name)?;
            let speed = arg("speed")?;
            let direction = arg("direction")?;
            let (v, running) = if speed <= walk_max {
                (walk_speed * speed / walk_max, false)
            } else {
                (run_speed, true)
            };
            world.set_motion(agent_id, yaw + direction, v, running)?;
        }
        "turn" => {
            world.set_facing(agent_id, yaw + arg("angle")?)?;
        }
        other => {
            return Err(RuleError::Invalid(format!(
                "unknown action module `{other}`"
            )));
        }
    }
    Ok(())
}
