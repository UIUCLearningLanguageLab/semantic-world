//! The simulation components of an entity. Every component here is covered by the state
//! hash, in the order [`crate::snapshot`] visits them.

use bevy_ecs::prelude::Component;
use glam::DVec2;
use indexmap::IndexMap;
use serde_json::Value;
use sw_schema::entity::{AgentState, EntityType, RiseWhen};

/// The entity's identity: the type name plus an index counting from 0 within the type, in
/// creation order (`human_0`, `berry_bush_3`).
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct Id {
    pub name: String,
    pub type_name: String,
    pub index: u32,
}

/// Position on the ground plane. The world is 3D with the y axis up; the ground is at
/// height 0 and every entity stands on it, so `x` and `z` are enough.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Position(pub DVec2);

/// Facing as a yaw in radians. Forward is `(cos yaw, sin yaw)` in `(x, z)`.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Facing {
    pub yaw: f64,
}

impl Facing {
    pub fn forward(&self) -> DVec2 {
        DVec2::new(self.yaw.cos(), self.yaw.sin())
    }

    /// Wrap a yaw into (-π, π].
    pub fn wrap(yaw: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        let mut y = yaw % tau;
        if y <= -std::f64::consts::PI {
            y += tau;
        } else if y > std::f64::consts::PI {
            y -= tau;
        }
        y
    }
}

/// The blocking shape of a solid entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Blocking {
    /// A circle of the footprint's radius.
    Circle,
    /// Three walls around a rectangular footprint, open on the side the entity faces.
    /// `half_forward` is half the extent along the facing, `half_side` half the extent
    /// across it, and `thickness` the wall thickness, all in meters.
    Walls {
        half_forward: f64,
        half_side: f64,
        thickness: f64,
    },
}

/// The entity's circle on the ground plane, used for blocking and surface distances, plus
/// its height, used for line-of-sight checks.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    pub radius: f64,
    pub solid: bool,
    pub blocking: Blocking,
    /// Height above the ground in meters: the body's `height` trait, or else the top of the
    /// placeholder. A solid entity taller than a camera occludes the camera's line of sight.
    pub height_m: f64,
}

/// Free-form labels that rules select on.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct Tags(pub Vec<String>);

impl Tags {
    pub fn has(&self, tag: &str) -> bool {
        self.0.iter().any(|t| t == tag)
    }
}

/// A countable amount held by the entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stock {
    pub count: i64,
    pub max: i64,
}

#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct Stocks(pub IndexMap<String, Stock>);

/// What happens when a need is at 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtMaxRule {
    /// Health drains at this rate per simulated second.
    DrainPerSecond(f64),
    /// Forced sleep for this many simulated seconds.
    Collapse { duration_s: f64 },
}

/// One need's rates, in per-second units, resolved from the type for this individual.
#[derive(Debug, Clone, PartialEq)]
pub struct NeedRates {
    pub rise_per_s: f64,
    pub rise_when: RiseWhen,
    pub rise_multipliers: Vec<(AgentState, f64)>,
    pub fall_per_s: f64,
    pub fall_multipliers: Vec<(AgentState, f64)>,
    pub insulation_reduces_rise: bool,
    pub at_max: AtMaxRule,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeedState {
    pub name: String,
    pub value: f64,
    pub rates: NeedRates,
}

impl NeedState {
    pub fn at_max(&self) -> bool {
        self.value >= 1.0
    }
}

/// The needs of an agent, in the order the type declares them, plus the body's insulation.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Needs {
    pub needs: Vec<NeedState>,
    pub insulation: f64,
}

impl Needs {
    pub fn get(&self, name: &str) -> Option<&NeedState> {
        self.needs.iter().find(|n| n.name == name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut NeedState> {
        self.needs.iter_mut().find(|n| n.name == name)
    }

    pub fn any_at_max(&self) -> bool {
        self.needs.iter().any(NeedState::at_max)
    }
}

#[derive(Component, Debug, Clone, PartialEq)]
pub struct Health {
    pub value: f64,
    /// Recovery per simulated second while no need is at max.
    pub recover_per_s: f64,
}

/// The state of an agent that the mechanisms read and write.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Agent {
    pub alive: bool,
    pub died_at_tick: Option<u64>,
    pub asleep: bool,
    /// While collapsed, the tick at which the collapse ends.
    pub collapsed_until_tick: Option<u64>,
    /// Whether the agent's center is within a shelter's footprint.
    pub in_shelter: bool,
    /// Whether the agent's last action failed.
    pub last_action_failed: bool,
}

impl Agent {
    pub fn collapsed(&self) -> bool {
        self.collapsed_until_tick.is_some()
    }
}

/// Kinematic motion set for a decision interval: the agent moves along its facing at
/// `speed_mps` for `ticks_left` more ticks.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Motion {
    pub speed_mps: f64,
    pub running: bool,
    pub ticks_left: u32,
    /// The speed the agent moved at during the last tick, 0 when it stood still. This is
    /// what proprioception reports as the current speed.
    pub last_tick_speed_mps: f64,
}

impl Motion {
    pub fn still() -> Self {
        Motion {
            speed_mps: 0.0,
            running: false,
            ticks_left: 0,
            last_tick_speed_mps: 0.0,
        }
    }

    pub fn moving(&self) -> bool {
        self.ticks_left > 0 && self.speed_mps > 0.0
    }
}

/// Whole-body contact: whether the agent was blocked by a solid this tick, and the direction
/// to the blocking surface in world coordinates (a unit vector, or zero).
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    pub blocked: bool,
    pub direction: DVec2,
}

impl Contact {
    pub fn none() -> Self {
        Contact {
            blocked: false,
            direction: DVec2::ZERO,
        }
    }
}

/// The individual's sampled type, kept for `individuals.parquet` and for rules that read
/// trait values (`target.provides.hunger`).
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Individual {
    pub entity_type: EntityType,
    /// The traits that were sampled, by path, in sampling order.
    pub sampled: Vec<(String, Value)>,
}
