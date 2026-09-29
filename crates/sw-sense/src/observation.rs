//! Observations: named blocks, one per sensor, plus the flat vector (contract 3).

use serde::{Deserialize, Serialize};
use sw_rules::Fact;

/// The numbers of one block.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockData {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

impl BlockData {
    pub fn dtype(&self) -> &'static str {
        match self {
            BlockData::F32(_) => "float32",
            BlockData::U8(_) => "uint8",
        }
    }

    pub fn len(&self) -> usize {
        match self {
            BlockData::F32(v) => v.len(),
            BlockData::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: BlockData,
}

/// One agent's observation at a decision point.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// One block per sensor, in manifest order.
    pub blocks: Vec<Block>,
    /// All non-image blocks concatenated, in manifest order.
    pub vector: Vec<f32>,
    /// The propositional sensor's facts, when the sensor is on.
    pub props: Option<Vec<Fact>>,
    /// Heard speech as symbols. Not in milestone 1: always `None`.
    pub text: Option<String>,
    /// The action mask, in action-manifest order, when impossible actions are masked.
    pub mask: Option<Vec<bool>>,
}

impl Observation {
    pub fn block(&self, name: &str) -> Option<&Block> {
        self.blocks.iter().find(|b| b.name == name)
    }
}

/// One block as the sensor manifest lists it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    pub in_vector: bool,
    /// Position of the block's first element in the flat vector, when it is in the vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    /// A human-readable label for each element of a vector block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}

/// The sensor manifest of one agent (contract 3), published once at reset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorManifest {
    pub agent: String,
    pub blocks: Vec<BlockInfo>,
    pub vector_length: usize,
    /// Whether the propositional sensor is on for this agent.
    pub props: bool,
}
