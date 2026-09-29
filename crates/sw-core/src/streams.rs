//! Named random streams from one master seed (MILESTONE_1.md, "Determinism").
//!
//! A stream's seed is SHA-256 of the master seed (8 bytes, little-endian) followed by the
//! stream name (UTF-8). The 32-byte digest seeds a `ChaCha8Rng`. Adding a new stream never
//! changes an existing stream.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// The stream that samples trait variation when individuals are built.
pub const TRAITS: &str = "traits";
/// The stream that places entities on the map.
pub const MAP: &str = "map";
/// The stream that rules draw from.
pub const RULES: &str = "rules";

/// The 32-byte seed of a named stream.
pub fn stream_seed(master_seed: u64, name: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(master_seed.to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.finalize().into()
}

/// A fresh generator for a named stream.
pub fn stream(master_seed: u64, name: &str) -> ChaCha8Rng {
    ChaCha8Rng::from_seed(stream_seed(master_seed, name))
}

/// The name of an agent's stream, `agent:<id>`, whose seed `world.agent_seed(agent_id)`
/// hands to Python.
pub fn agent_stream_name(agent_id: &str) -> String {
    format!("agent:{agent_id}")
}

/// The name of the stream an agent's sensors draw noise from, `sense:<id>`.
pub fn sense_stream_name(agent_id: &str) -> String {
    format!("sense:{agent_id}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn seed_follows_the_specification() {
        // SHA-256 of the 8 little-endian bytes of 0 followed by "map", computed independently.
        let expected = Sha256::digest([0u8, 0, 0, 0, 0, 0, 0, 0, b'm', b'a', b'p']);
        assert_eq!(stream_seed(0, MAP), <[u8; 32]>::from(expected));
    }

    #[test]
    fn streams_are_independent_and_reproducible() {
        let mut a = stream(7, MAP);
        let mut b = stream(7, MAP);
        let mut c = stream(7, TRAITS);
        let mut d = stream(8, MAP);
        let xa: Vec<u32> = (0..4).map(|_| a.random()).collect();
        let xb: Vec<u32> = (0..4).map(|_| b.random()).collect();
        let xc: Vec<u32> = (0..4).map(|_| c.random()).collect();
        let xd: Vec<u32> = (0..4).map(|_| d.random()).collect();
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);
        assert_ne!(xa, xd);
    }
}
