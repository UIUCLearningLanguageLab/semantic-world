//! State snapshots and the state hash.
//!
//! The hash covers every simulation component of every entity, visited in entity-ID order
//! (type name, then index), plus the tick number. Floating-point values are hashed by their
//! bit patterns. Random-stream positions and rendering are not covered.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One agent's state in a snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub alive: bool,
    pub died_at_tick: Option<u64>,
    pub asleep: bool,
    pub collapsed_until_tick: Option<u64>,
    pub in_shelter: bool,
    pub last_action_failed: bool,
    pub speed_mps: f64,
    pub running: bool,
    pub blocked: bool,
    pub contact_direction: [f64; 2],
}

/// One entity's state in a snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub id: String,
    pub type_name: String,
    pub index: u32,
    pub x: f64,
    pub z: f64,
    pub yaw: f64,
    pub radius: f64,
    pub solid: bool,
    pub tags: Vec<String>,
    /// `(name, count, max)`.
    pub stocks: Vec<(String, i64, i64)>,
    /// `(name, value)`, in the type's order.
    pub needs: Vec<(String, f64)>,
    pub health: Option<f64>,
    pub agent: Option<AgentSnapshot>,
}

impl EntitySnapshot {
    pub fn need(&self, name: &str) -> Option<f64> {
        self.needs.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    pub fn stock(&self, name: &str) -> Option<i64> {
        self.stocks
            .iter()
            .find(|(n, ..)| n == name)
            .map(|(_, c, _)| *c)
    }
}

/// The canonical state at one tick, as the viewer, the logs, tests, and privileged agents
/// see it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub tick: u64,
    pub time_s: f64,
    pub day: u64,
    pub hour_of_day: f64,
    pub light_level: f64,
    pub night: bool,
    /// In entity-ID order: by type name, then index.
    pub entities: Vec<EntitySnapshot>,
}

impl StateSnapshot {
    pub fn entity(&self, id: &str) -> Option<&EntitySnapshot> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn of_type<'a>(
        &'a self,
        type_name: &'a str,
    ) -> impl Iterator<Item = &'a EntitySnapshot> + 'a {
        self.entities
            .iter()
            .filter(move |e| e.type_name == type_name)
    }

    /// The state hash of this snapshot.
    pub fn hash(&self) -> StateHash {
        let mut h = Sha256::new();
        h.update(self.tick.to_le_bytes());
        h.update((self.entities.len() as u64).to_le_bytes());
        for e in &self.entities {
            hash_str(&mut h, &e.id);
            hash_str(&mut h, &e.type_name);
            h.update(e.index.to_le_bytes());
            for v in [e.x, e.z, e.yaw, e.radius] {
                h.update(v.to_bits().to_le_bytes());
            }
            h.update([u8::from(e.solid)]);
            h.update((e.tags.len() as u64).to_le_bytes());
            for tag in &e.tags {
                hash_str(&mut h, tag);
            }
            h.update((e.stocks.len() as u64).to_le_bytes());
            for (name, count, max) in &e.stocks {
                hash_str(&mut h, name);
                h.update(count.to_le_bytes());
                h.update(max.to_le_bytes());
            }
            h.update((e.needs.len() as u64).to_le_bytes());
            for (name, value) in &e.needs {
                hash_str(&mut h, name);
                h.update(value.to_bits().to_le_bytes());
            }
            hash_opt_f64(&mut h, e.health);
            match &e.agent {
                None => h.update([0u8]),
                Some(a) => {
                    h.update([
                        1u8,
                        u8::from(a.alive),
                        u8::from(a.asleep),
                        u8::from(a.in_shelter),
                    ]);
                    h.update([
                        u8::from(a.last_action_failed),
                        u8::from(a.running),
                        u8::from(a.blocked),
                    ]);
                    h.update(a.died_at_tick.unwrap_or(u64::MAX).to_le_bytes());
                    h.update(a.collapsed_until_tick.unwrap_or(u64::MAX).to_le_bytes());
                    h.update(a.speed_mps.to_bits().to_le_bytes());
                    h.update(a.contact_direction[0].to_bits().to_le_bytes());
                    h.update(a.contact_direction[1].to_bits().to_le_bytes());
                }
            }
        }
        StateHash(h.finalize().into())
    }
}

fn hash_str(h: &mut Sha256, s: &str) {
    h.update((s.len() as u64).to_le_bytes());
    h.update(s.as_bytes());
}

fn hash_opt_f64(h: &mut Sha256, v: Option<f64>) {
    match v {
        None => h.update([0u8]),
        Some(v) => {
            h.update([1u8]);
            h.update(v.to_bits().to_le_bytes());
        }
    }
}

/// A 256-bit state hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateHash(pub [u8; 32]);

impl StateHash {
    /// The first 8 bytes as an integer, for compact logs.
    pub fn to_u64(self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().expect("8 bytes"))
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for StateHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for StateHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StateHash({})", self.to_hex())
    }
}
