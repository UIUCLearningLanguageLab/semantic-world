//! The world: ECS setup, clock, calendar, seeded random streams, building entities from
//! types, needs and health, kinematic movement, the step loop, and state hashing.
//!
//! The engine runs any world and hard-codes none: entity types, numbers, actions, and rules
//! live in `data/`. The world gives no reward. The core is single-threaded within one world,
//! and every loop that affects results visits entities in creation order.

pub mod build;
pub mod clock;
pub mod components;
pub mod error;
pub mod map;
pub mod needs;
pub mod snapshot;
pub mod space;
pub mod streams;
pub mod world;

pub use clock::Clock;
pub use error::{CoreError, Result};
pub use snapshot::{EntitySnapshot, StateHash, StateSnapshot};
pub use world::World;

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-core";
