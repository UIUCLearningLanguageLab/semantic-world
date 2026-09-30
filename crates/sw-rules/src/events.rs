//! Events: what happened in the world at a tick, for agents and logs.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub tick: u64,
    /// The agent the event is about, if any.
    pub agent: Option<String>,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    ActionStarted {
        action: String,
        target: Option<String>,
    },
    ActionCompleted {
        action: String,
    },
    ActionInterrupted {
        action: String,
        /// The action chosen instead, or `collapse`.
        by: String,
    },
    /// The action was impossible: its preconditions failed, or it was masked.
    ActionFailed {
        action: String,
        reason: String,
    },
    ProcessFired {
        process: String,
        entity: String,
    },
    Collapsed,
    CollapseEnded,
    Died,
}

impl Event {
    pub fn is_failure(&self) -> bool {
        matches!(self.kind, EventKind::ActionFailed { .. })
    }
}
