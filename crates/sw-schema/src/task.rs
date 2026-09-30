//! Task files (contract 9): subskills and a task composed of them. Milestone 1 parses tasks
//! and does not run them; goals stay as PDDL text.

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::entity::check_identifier;
use crate::error::{Ctx, Result};
use crate::traits::Trait;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskFile {
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub subskills: IndexMap<String, Subskill>,
    pub task: Task,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Subskill {
    /// The goal condition, as PDDL propositions.
    pub goal: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub name: String,
    /// The subskill whose goal is the task's goal.
    pub goal: String,
    /// Delays between a step that pays off and the payoff, keyed `from->to`, in seconds.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub delays: IndexMap<String, Trait<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<Success>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Success {
    /// Time allowed, in seconds.
    pub within: Trait<f64>,
}

impl TaskFile {
    pub fn validate(&self, ctx: &Ctx) -> Result<()> {
        for (name, subskill) in &self.subskills {
            let sctx = ctx.child("subskills").child(name);
            check_identifier(&sctx, name)?;
            for (i, req) in subskill.requires.iter().enumerate() {
                sctx.child("requires").child(i.to_string()).check(
                    self.subskills.contains_key(req),
                    format!("`{req}` is not a subskill"),
                )?;
            }
        }
        let tctx = ctx.child("task");
        check_identifier(&tctx.child("name"), &self.task.name)?;
        tctx.child("goal").check(
            self.subskills.contains_key(&self.task.goal),
            format!("`{}` is not a subskill", self.task.goal),
        )?;
        for (key, delay) in &self.task.delays {
            let dctx = tctx.child("delays").child(key);
            let (from, to) = key
                .split_once("->")
                .ok_or_else(|| dctx.error("a delay key is written `from->to`"))?;
            dctx.check(
                self.subskills.contains_key(from) && self.subskills.contains_key(to),
                format!("`{from}` and `{to}` must both be subskills"),
            )?;
            delay.validate(&dctx)?;
        }
        if let Some(success) = &self.task.success {
            success
                .within
                .validate(&tctx.child("success").child("within"))?;
        }
        Ok(())
    }
}
