//! Registered detectors: the generic computations behind the proposition vocabulary
//! (contract 5). A rules file declares each predicate with the detector that computes it
//! and the detector's parameters.

use std::fmt;

use bevy_ecs::prelude::Entity;
use glam::DVec2;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sw_core::World;
use sw_core::components::{Health, Needs};
use sw_core::space::Obstacle;
use sw_schema::entity::Params;
use sw_schema::rules::PredicateSpec;

use crate::context::{Bindings, Ctx};
use crate::error::{Result, RuleError};
use crate::predicates::surface_gap;

/// One fact, in PDDL style: a predicate applied to arguments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub predicate: String,
    pub args: Vec<String>,
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}", self.predicate)?;
        for arg in &self.args {
            write!(f, " {arg}")?;
        }
        write!(f, ")")
    }
}

pub struct DetectorDef {
    pub name: &'static str,
    /// The arguments the predicate must declare, by position.
    pub arity: usize,
    pub params: &'static [&'static str],
    pub run: DetectorFn,
}

/// What a detector sees: the observing agent and the entities within the sensor's range,
/// in creation order, the agent included.
pub struct DetCtx<'w> {
    pub ctx: Ctx<'w>,
    pub agent: Entity,
    pub in_range: Vec<Entity>,
}

pub type DetectorFn = fn(&DetCtx, &Params) -> Result<Vec<Vec<String>>>;

pub const DETECTORS: &[DetectorDef] = &[
    DetectorDef {
        name: "surface_gap_at_most",
        arity: 2,
        params: &["max_gap_m"],
        run: surface_gap_at_most,
    },
    DetectorDef {
        name: "in_view",
        arity: 2,
        params: &[],
        run: in_view,
    },
    DetectorDef {
        name: "entity_type",
        arity: 2,
        params: &[],
        run: entity_type,
    },
    DetectorDef {
        name: "stock_count",
        arity: 2,
        params: &["stock"],
        run: stock_count,
    },
    DetectorDef {
        name: "inside_footprint",
        arity: 2,
        params: &["has"],
        run: inside_footprint,
    },
    DetectorDef {
        name: "state_is",
        arity: 1,
        params: &["state"],
        run: state_is,
    },
    DetectorDef {
        name: "is_night",
        arity: 0,
        params: &[],
        run: is_night,
    },
    DetectorDef {
        name: "need_level",
        arity: 2,
        params: &["need", "decimals"],
        run: need_level,
    },
    DetectorDef {
        name: "health_level",
        arity: 2,
        params: &["decimals"],
        run: health_level,
    },
];

pub fn lookup(name: &str) -> Option<&'static DetectorDef> {
    DETECTORS.iter().find(|d| d.name == name)
}

/// Check a predicate declaration against the registry.
pub fn check(name: &str, spec: &PredicateSpec) -> Result<()> {
    let where_ = format!("predicates.{name}");
    let def = lookup(&spec.detector).ok_or_else(|| {
        RuleError::Invalid(format!("{where_}: unknown detector `{}`", spec.detector))
    })?;
    if spec.args.len() != def.arity {
        return Err(RuleError::Invalid(format!(
            "{where_}: detector `{}` computes a predicate with {} arguments, not {}",
            def.name,
            def.arity,
            spec.args.len()
        )));
    }
    for p in def.params {
        if !spec.params.contains_key(*p) {
            return Err(RuleError::Invalid(format!(
                "{where_}: detector `{}` needs the parameter `{p}`",
                def.name
            )));
        }
    }
    Ok(())
}

/// Every fact of one predicate for one agent.
pub fn facts_of(name: &str, spec: &PredicateSpec, det: &DetCtx) -> Result<Vec<Fact>> {
    let def = lookup(&spec.detector)
        .ok_or_else(|| RuleError::Invalid(format!("unknown detector `{}`", spec.detector)))?;
    Ok((def.run)(det, &spec.params)?
        .into_iter()
        .map(|args| Fact {
            predicate: name.to_string(),
            args,
        })
        .collect())
}

/// The entities within `range_m` of the agent's surface, in creation order, the agent first.
pub fn entities_in_range(world: &World, agent: Entity, range_m: f64) -> Vec<Entity> {
    let ctx = Ctx::new(world, Bindings::agent(agent));
    world
        .registry()
        .entities
        .iter()
        .copied()
        .filter(|e| *e == agent || surface_gap(&ctx, agent, *e) <= range_m)
        .collect()
}

fn param_string<'a>(params: &'a Params, name: &str) -> Result<&'a str> {
    params
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| RuleError::Invalid(format!("detector parameter `{name}` must be a name")))
}

fn param_number(params: &Params, name: &str) -> Result<f64> {
    sw_core::map::param_f64(params, name, "detector").map_err(RuleError::Core)
}

fn round_to(value: f64, decimals: f64) -> String {
    format!("{:.*}", decimals.max(0.0) as usize, value)
}

fn surface_gap_at_most(det: &DetCtx, params: &Params) -> Result<Vec<Vec<String>>> {
    let max_gap = param_number(params, "max_gap_m")?;
    let me = det.ctx.id(det.agent);
    Ok(det
        .in_range
        .iter()
        .filter(|e| **e != det.agent && surface_gap(&det.ctx, det.agent, **e) <= max_gap)
        .map(|e| vec![me.to_string(), det.ctx.id(*e).to_string()])
        .collect())
}

/// The object's center is within the eyes' field of view and range, and the line of sight
/// from the camera to the center is not blocked by a solid entity taller than the camera.
fn in_view(det: &DetCtx, _params: &Params) -> Result<Vec<Vec<String>>> {
    let individual = det.ctx.individual(det.agent);
    let Some(eyes) = &individual.entity_type.sensors.eyes else {
        return Ok(Vec::new());
    };
    let fov = *eyes.fov_deg.value();
    let range = *eyes.range_m.value();
    let camera_height = *eyes.height_m.value();
    let me = det.ctx.id(det.agent);
    let origin = det.ctx.position(det.agent);
    let forward = det.ctx.facing(det.agent).forward();
    let occluders: Vec<(Entity, Obstacle)> = det
        .ctx
        .world
        .registry()
        .entities
        .iter()
        .copied()
        .filter(|e| *e != det.agent)
        .filter_map(|e| {
            let footprint = det.ctx.footprint(e);
            (footprint.solid && footprint.height_m > camera_height).then(|| {
                (
                    e,
                    Obstacle {
                        position: det.ctx.position(e),
                        yaw: det.ctx.facing(e).yaw,
                        footprint: *footprint,
                    },
                )
            })
        })
        .collect();
    let mut facts = Vec::new();
    for &e in &det.in_range {
        if e == det.agent {
            continue;
        }
        let center = det.ctx.position(e);
        let to = center - origin;
        let distance = to.length();
        if distance > range {
            continue;
        }
        if distance > 0.0 {
            let cos = (forward.dot(to) / distance).clamp(-1.0, 1.0);
            if cos.acos().to_degrees() > fov / 2.0 {
                continue;
            }
        }
        let blocked = occluders
            .iter()
            .any(|(o, obstacle)| *o != e && obstacle.blocks_segment(origin, center));
        if !blocked {
            facts.push(vec![me.to_string(), det.ctx.id(e).to_string()]);
        }
    }
    Ok(facts)
}

fn entity_type(det: &DetCtx, _params: &Params) -> Result<Vec<Vec<String>>> {
    Ok(det
        .in_range
        .iter()
        .map(|e| {
            let id = det
                .ctx
                .world
                .ecs()
                .get::<sw_core::components::Id>(*e)
                .expect("id");
            vec![id.name.clone(), id.type_name.clone()]
        })
        .collect())
}

fn stock_count(det: &DetCtx, params: &Params) -> Result<Vec<Vec<String>>> {
    let stock = param_string(params, "stock")?;
    Ok(det
        .in_range
        .iter()
        .filter_map(|e| {
            let s = det.ctx.stocks(*e).0.get(stock)?;
            Some(vec![det.ctx.id(*e).to_string(), s.count.to_string()])
        })
        .collect())
}

fn inside_footprint(det: &DetCtx, params: &Params) -> Result<Vec<Vec<String>>> {
    let tag = param_string(params, "has")?;
    let me = det.ctx.id(det.agent);
    let position: DVec2 = det.ctx.position(det.agent);
    Ok(det
        .in_range
        .iter()
        .filter(|e| **e != det.agent && det.ctx.tags(**e).has(tag))
        .filter(|e| {
            Obstacle {
                position: det.ctx.position(**e),
                yaw: det.ctx.facing(**e).yaw,
                footprint: *det.ctx.footprint(**e),
            }
            .contains(position)
        })
        .map(|e| vec![me.to_string(), det.ctx.id(*e).to_string()])
        .collect())
}

fn state_is(det: &DetCtx, params: &Params) -> Result<Vec<Vec<String>>> {
    let state = param_string(params, "state")?;
    let agent = det.ctx.agent(det.agent)?;
    let holds = match state {
        "asleep" => agent.asleep,
        "collapsed" => agent.collapsed(),
        "alive" => agent.alive,
        "in_shelter" => agent.in_shelter,
        other => {
            return Err(RuleError::Invalid(format!(
                "detector `state_is`: unknown state `{other}`"
            )));
        }
    };
    Ok(if holds {
        vec![vec![det.ctx.id(det.agent).to_string()]]
    } else {
        Vec::new()
    })
}

fn is_night(det: &DetCtx, _params: &Params) -> Result<Vec<Vec<String>>> {
    Ok(if det.ctx.world.clock().is_night() {
        vec![Vec::new()]
    } else {
        Vec::new()
    })
}

fn need_level(det: &DetCtx, params: &Params) -> Result<Vec<Vec<String>>> {
    let need = param_string(params, "need")?;
    let decimals = param_number(params, "decimals")?;
    let Some(needs) = det.ctx.world.ecs().get::<Needs>(det.agent) else {
        return Ok(Vec::new());
    };
    Ok(needs
        .get(need)
        .map(|n| {
            vec![vec![
                det.ctx.id(det.agent).to_string(),
                round_to(n.value, decimals),
            ]]
        })
        .unwrap_or_default())
}

fn health_level(det: &DetCtx, params: &Params) -> Result<Vec<Vec<String>>> {
    let decimals = param_number(params, "decimals")?;
    Ok(det
        .ctx
        .world
        .ecs()
        .get::<Health>(det.agent)
        .map(|h| {
            vec![vec![
                det.ctx.id(det.agent).to_string(),
                round_to(h.value, decimals),
            ]]
        })
        .unwrap_or_default())
}
