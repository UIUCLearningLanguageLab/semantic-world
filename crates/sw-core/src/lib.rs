//! The world: ECS setup, clock, calendar, seeded random streams, building entities from types,
//! needs and health, kinematic movement, the step loop, and state hashing.
//!
//! Stage 3 of `docs/specs/MILESTONE_1.md` fills this crate in. The engine runs any world and
//! hard-codes none: entity types, numbers, actions, and rules live in `data/`.

pub mod streams;

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-core";
