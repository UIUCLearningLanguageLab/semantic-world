//! Sensors other than eyes: interoception, touch, proprioception, and the propositional
//! sensor, plus the sensor manifest (contract 3). Eyes are rendered by `sw-render`; until a
//! renderer is attached, the eyes block is black.
//!
//! Sensors read the canonical state and never change it. Sensor noise comes from the agent's
//! `sense:<id>` stream, so a run stays reproducible.

pub mod observation;
pub mod sensors;

pub use observation::{Block, BlockData, BlockInfo, Observation, SensorManifest};
pub use sensors::AgentSensors;

/// Name of this crate, used by the workspace smoke tests.
pub const CRATE_NAME: &str = "sw-sense";
