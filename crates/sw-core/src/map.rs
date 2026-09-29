//! Map generators, registered by name. A generator places every object and every agent
//! from the `map` random stream and returns the placements in creation order.

use glam::DVec2;
use indexmap::IndexMap;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use serde_json::Value;
use sw_schema::Trait;
use sw_schema::world::MapSpec;

use crate::error::{CoreError, Result};
use crate::space::MapBounds;

/// Where one entity goes.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub type_name: String,
    pub position: DVec2,
    pub yaw: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapLayout {
    pub bounds: MapBounds,
    /// Objects first, in the order the generator's `objects` parameter lists them, then
    /// agents in population order.
    pub placements: Vec<Placement>,
}

/// Run the generator the map names. `agents` lists the population's types and counts.
pub fn generate(
    spec: &MapSpec,
    agents: &[(String, u32)],
    rng: &mut ChaCha8Rng,
) -> Result<MapLayout> {
    match spec.generator.as_str() {
        "meadow" => meadow(spec, agents, rng),
        other => Err(CoreError::UnknownModule {
            kind: "map generator",
            name: other.to_string(),
        }),
    }
}

/// A numeric parameter of a module, written as a bare number or a full trait.
pub fn param_f64(params: &IndexMap<String, Value>, name: &str, module: &str) -> Result<f64> {
    let value = params
        .get(name)
        .ok_or_else(|| CoreError::Invalid(format!("{module}: missing parameter `{name}`")))?;
    let t: Trait<f64> = serde_json::from_value(value.clone())
        .map_err(|e| CoreError::Invalid(format!("{module}: parameter `{name}`: {e}")))?;
    Ok(*t.value())
}

const MAX_ATTEMPTS: u32 = 10_000;

/// The meadow: a flat square with objects scattered at random. Every object's center is at
/// least `min_object_gap_m` from every other object's center and at least `edge_margin_m`
/// from the edge. Agents start at least `agent_clearance_m` from any object, at least
/// `min_object_gap_m` from each other, with a random facing. Every entity gets a random
/// rotation, which matters for shelters: the open side faces the way the rotation points.
fn meadow(spec: &MapSpec, agents: &[(String, u32)], rng: &mut ChaCha8Rng) -> Result<MapLayout> {
    let module = "map generator `meadow`";
    let objects: IndexMap<String, u32> = spec
        .params
        .get("objects")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| CoreError::Invalid(format!("{module}: parameter `objects`: {e}")))?
        .unwrap_or_default();
    let min_gap = param_f64(&spec.params, "min_object_gap_m", module)?;
    let margin = param_f64(&spec.params, "edge_margin_m", module)?;
    let clearance = param_f64(&spec.params, "agent_clearance_m", module)?;
    let bounds = MapBounds {
        half_x: spec.size_m[0] / 2.0,
        half_z: spec.size_m[1] / 2.0,
    };
    let lo = DVec2::new(-bounds.half_x + margin, -bounds.half_z + margin);
    let hi = DVec2::new(bounds.half_x - margin, bounds.half_z - margin);
    if lo.x >= hi.x || lo.y >= hi.y {
        return Err(CoreError::Map("the edge margin leaves no room".into()));
    }

    let mut placements: Vec<Placement> = Vec::new();
    let mut object_centers: Vec<DVec2> = Vec::new();
    for (type_name, count) in &objects {
        for i in 0..*count {
            let position = place(rng, lo, hi, |p| {
                object_centers.iter().all(|c| c.distance(p) >= min_gap)
            })
            .ok_or_else(|| {
                CoreError::Map(format!(
                    "could not place {type_name} {i} with a gap of {min_gap} m after {MAX_ATTEMPTS} attempts"
                ))
            })?;
            let yaw = random_yaw(rng);
            object_centers.push(position);
            placements.push(Placement {
                type_name: type_name.clone(),
                position,
                yaw,
            });
        }
    }
    let mut agent_centers: Vec<DVec2> = Vec::new();
    for (type_name, count) in agents {
        for i in 0..*count {
            let position = place(rng, lo, hi, |p| {
                object_centers.iter().all(|c| c.distance(p) >= clearance)
                    && agent_centers.iter().all(|c| c.distance(p) >= min_gap)
            })
            .ok_or_else(|| {
                CoreError::Map(format!(
                    "could not place agent {type_name} {i} with a clearance of {clearance} m after {MAX_ATTEMPTS} attempts"
                ))
            })?;
            let yaw = random_yaw(rng);
            agent_centers.push(position);
            placements.push(Placement {
                type_name: type_name.clone(),
                position,
                yaw,
            });
        }
    }
    Ok(MapLayout { bounds, placements })
}

fn place(
    rng: &mut ChaCha8Rng,
    lo: DVec2,
    hi: DVec2,
    accept: impl Fn(DVec2) -> bool,
) -> Option<DVec2> {
    for _ in 0..MAX_ATTEMPTS {
        let x: f64 = rng.random_range(lo.x..hi.x);
        let z: f64 = rng.random_range(lo.y..hi.y);
        let p = DVec2::new(x, z);
        if accept(p) {
            return Some(p);
        }
    }
    None
}

fn random_yaw(rng: &mut ChaCha8Rng) -> f64 {
    rng.random_range(-std::f64::consts::PI..std::f64::consts::PI)
}
