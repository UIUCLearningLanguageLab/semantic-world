//! The rule engine: applies agents' choices at decision points, runs durative actions and
//! processes every tick, and reports events.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Entity;
use indexmap::IndexMap;
use sw_core::World;
use sw_core::components::{Agent, Id};
use sw_schema::Resolved;
use sw_schema::experiment::ImpossibleActions;
use sw_schema::rules::PredicateSpec;

use crate::actions::{self, ActionChoice, ActionDef, ActionManifest, Ongoing};
use crate::context::{Bindings, Ctx};
use crate::detectors::{self, DetCtx, Fact};
use crate::effects;
use crate::error::{Result, RuleError};
use crate::events::{Event, EventKind};
use crate::modules;
use crate::processes::ProcessDef;

/// What the engine knows about one agent.
#[derive(Debug, Clone)]
struct AgentInfo {
    entity: Entity,
    /// Indices into `Engine::actions`, in manifest order.
    actions: Vec<usize>,
    /// Whether the agent may name a target by ID: only agents with the propositional
    /// sensor switched on (contract 3).
    may_target_by_id: bool,
    /// The propositional sensor's range, when the sensor is on.
    prop_range_m: Option<f64>,
}

pub struct Engine {
    actions: Vec<ActionDef>,
    processes: Vec<ProcessDef>,
    predicates: IndexMap<String, PredicateSpec>,
    impossible: ImpossibleActions,
    agents: BTreeMap<String, AgentInfo>,
    ongoing: BTreeMap<String, Ongoing>,
    events: Vec<Event>,
}

impl Engine {
    /// Compile the rules of a resolved experiment against the registries, for a world built
    /// from it.
    pub fn new(resolved: &Resolved, world: &World) -> Result<Engine> {
        let tick_s = world.clock().tick_s;
        let mut actions = Vec::new();
        for (name, spec) in &resolved.rules.actions {
            actions.push(ActionDef::compile(name, spec, tick_s)?);
        }
        let mut processes = Vec::new();
        for (name, spec) in &resolved.rules.processes {
            processes.push(ProcessDef::compile(name, spec, tick_s)?);
        }
        for (name, spec) in &resolved.rules.predicates {
            detectors::check(name, spec)?;
        }
        let props_on = resolved.experiment.views.propositional_sensor;
        let mut agents = BTreeMap::new();
        for id in world.agent_ids() {
            let entity = world.entity(&id).expect("agent");
            let individual = world.individual(&id).expect("individual");
            let actuators: Vec<&str> = individual
                .entity_type
                .actuators
                .keys()
                .map(String::as_str)
                .collect();
            let available: Vec<usize> = actions
                .iter()
                .enumerate()
                .filter(|(_, a)| a.available_to(&actuators))
                .map(|(i, _)| i)
                .collect();
            let prop_range_m = individual
                .entity_type
                .sensors
                .propositional
                .as_ref()
                .filter(|_| props_on)
                .map(|p| *p.range_m.value());
            agents.insert(
                id,
                AgentInfo {
                    entity,
                    actions: available,
                    may_target_by_id: prop_range_m.is_some(),
                    prop_range_m,
                },
            );
        }
        Ok(Engine {
            actions,
            processes,
            predicates: resolved.rules.predicates.clone(),
            impossible: resolved.experiment.options.impossible_actions,
            agents,
            ongoing: BTreeMap::new(),
            events: Vec::new(),
        })
    }

    pub fn actions(&self) -> &[ActionDef] {
        &self.actions
    }

    pub fn processes(&self) -> &[ProcessDef] {
        &self.processes
    }

    pub fn impossible_actions(&self) -> ImpossibleActions {
        self.impossible
    }

    fn info(&self, agent: &str) -> Result<&AgentInfo> {
        self.agents
            .get(agent)
            .ok_or_else(|| RuleError::Usage(format!("unknown agent `{agent}`")))
    }

    /// The action manifest of an agent.
    pub fn manifest(&self, agent: &str) -> Result<ActionManifest> {
        let info = self.info(agent)?;
        Ok(ActionManifest {
            agent: agent.to_string(),
            actions: info
                .actions
                .iter()
                .map(|i| self.actions[*i].info())
                .collect(),
        })
    }

    /// The action mask of an agent: for each action type in manifest order, whether the
    /// action could start now. A durative action in progress counts as interrupted.
    pub fn mask(&self, world: &World, agent: &str) -> Result<Vec<bool>> {
        let info = self.info(agent)?;
        let interrupting = self.ongoing.contains_key(agent);
        Ok(info
            .actions
            .iter()
            .map(|i| {
                let def = &self.actions[*i];
                def.name == "noop"
                    || actions::resolve(world, def, info.entity, None, interrupting).is_ok()
            })
            .collect())
    }

    /// The durative action an agent is in, if any.
    pub fn current_action(&self, agent: &str) -> Option<&Ongoing> {
        self.ongoing.get(agent)
    }

    /// The facts the propositional sensor reports for an agent, in vocabulary order. Empty
    /// when the sensor is off for the agent.
    pub fn facts(&self, world: &World, agent: &str) -> Result<Vec<Fact>> {
        let info = self.info(agent)?;
        let Some(range) = info.prop_range_m else {
            return Ok(Vec::new());
        };
        let det = DetCtx {
            ctx: Ctx::new(world, Bindings::agent(info.entity)),
            agent: info.entity,
            in_range: detectors::entities_in_range(world, info.entity, range),
        };
        let mut facts = Vec::new();
        for (name, spec) in &self.predicates {
            facts.extend(detectors::facts_of(name, spec, &det)?);
        }
        Ok(facts)
    }

    /// Take the events reported since the last call.
    pub fn drain_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// One decision: apply every awaiting agent's choice, then run the ticks up to the next
    /// decision point. An awaiting agent without a choice continues with `noop`. Returns the
    /// events of the interval.
    pub fn step(
        &mut self,
        world: &mut World,
        choices: &BTreeMap<String, ActionChoice>,
    ) -> Result<Vec<Event>> {
        if !world.clock().is_decision_point() {
            return Err(RuleError::Usage(
                "step is only valid at a decision point".into(),
            ));
        }
        let awaiting = world.agents_awaiting_action();
        for agent in choices.keys() {
            if !awaiting.iter().any(|a| a == agent) {
                return Err(RuleError::Usage(format!(
                    "`{agent}` is not awaiting an action"
                )));
            }
        }
        let noop = ActionChoice::noop();
        for agent in &awaiting {
            let choice = choices.get(agent).unwrap_or(&noop);
            self.apply_choice(world, agent, choice)?;
        }
        let k = world.clock().decision_every_ticks;
        for _ in 0..k {
            self.tick(world)?;
        }
        Ok(self.drain_events())
    }

    /// One tick: rate effects of ongoing actions, the world's own tick, then completions,
    /// `until` conditions, processes, and the collapse and death events.
    pub fn tick(&mut self, world: &mut World) -> Result<()> {
        let dt = world.clock().tick_s;
        let before: Vec<(String, bool, bool)> = self
            .agents
            .iter()
            .map(|(id, info)| {
                let a = world.ecs().get::<Agent>(info.entity).expect("agent");
                (id.clone(), a.alive, a.collapsed())
            })
            .collect();
        // Rate effects, in agent creation order.
        for id in world.agent_ids() {
            if let Some(ongoing) = self.ongoing.get(&id).cloned() {
                let def = self.def(&ongoing.action)?.clone();
                let bindings = self.bindings(world, &id, ongoing.target.as_deref());
                for effect in def.rate_effects() {
                    effects::apply(world, &bindings, effect, dt)?;
                }
            }
        }
        world.tick();
        let now = world.clock().tick;
        // Completions and `until` conditions.
        for id in world.agent_ids() {
            let Some(ongoing) = self.ongoing.get(&id).cloned() else {
                continue;
            };
            let def = self.def(&ongoing.action)?.clone();
            let entity = self.info(&id)?.entity;
            let target = ongoing.target.as_deref().and_then(|t| world.entity(t));
            let due = ongoing.ends_at_tick.is_some_and(|t| now >= t)
                || actions::until_holds(world, &def, entity, target)?;
            if due {
                let bindings = self.bindings(world, &id, ongoing.target.as_deref());
                for effect in def.once_effects() {
                    effects::apply(world, &bindings, effect, dt)?;
                }
                for effect in &def.spec.on_end {
                    effects::apply(world, &bindings, effect, dt)?;
                }
                self.ongoing.remove(&id);
                self.events.push(Event {
                    tick: now,
                    agent: Some(id.clone()),
                    kind: EventKind::ActionCompleted {
                        action: ongoing.action.clone(),
                    },
                });
            }
        }
        // Processes.
        for process in self.processes.clone() {
            process.run(world, now, &mut self.events)?;
        }
        // Collapse and death, which end any action in progress.
        for (id, was_alive, was_collapsed) in before {
            let info = self.info(&id)?;
            let a = world
                .ecs()
                .get::<Agent>(info.entity)
                .expect("agent")
                .clone();
            if was_alive && !a.alive {
                self.ongoing.remove(&id);
                self.events.push(Event {
                    tick: now,
                    agent: Some(id.clone()),
                    kind: EventKind::Died,
                });
            } else if !was_collapsed && a.collapsed() {
                if let Some(ongoing) = self.ongoing.remove(&id) {
                    let def = self.def(&ongoing.action)?.clone();
                    let bindings = self.bindings(world, &id, ongoing.target.as_deref());
                    for effect in &def.spec.on_end {
                        effects::apply(world, &bindings, effect, dt)?;
                    }
                    self.events.push(Event {
                        tick: now,
                        agent: Some(id.clone()),
                        kind: EventKind::ActionInterrupted {
                            action: ongoing.action,
                            by: "collapse".into(),
                        },
                    });
                }
                self.events.push(Event {
                    tick: now,
                    agent: Some(id.clone()),
                    kind: EventKind::Collapsed,
                });
            } else if was_collapsed && !a.collapsed() && a.alive {
                self.events.push(Event {
                    tick: now,
                    agent: Some(id.clone()),
                    kind: EventKind::CollapseEnded,
                });
            }
        }
        Ok(())
    }

    fn def(&self, name: &str) -> Result<&ActionDef> {
        self.actions
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| RuleError::Usage(format!("unknown action `{name}`")))
    }

    fn bindings(&self, world: &World, agent: &str, target: Option<&str>) -> Bindings {
        let entity = self.agents[agent].entity;
        Bindings::agent(entity).with_target(target.and_then(|t| world.entity(t)))
    }

    fn set_failed(world: &mut World, entity: Entity, failed: bool) {
        world
            .ecs_mut()
            .get_mut::<Agent>(entity)
            .expect("agent")
            .last_action_failed = failed;
    }

    fn apply_choice(
        &mut self,
        world: &mut World,
        agent: &str,
        choice: &ActionChoice,
    ) -> Result<()> {
        let tick = world.clock().tick;
        let info = self.info(agent)?.clone();
        if choice.name == "noop" {
            Self::set_failed(world, info.entity, false);
            return Ok(());
        }
        let action_index = *info
            .actions
            .iter()
            .find(|i| self.actions[**i].name == choice.name)
            .ok_or_else(|| {
                RuleError::Usage(format!("`{agent}` has no action `{}`", choice.name))
            })?;
        let def = self.actions[action_index].clone();
        if choice.target.is_some() && !info.may_target_by_id {
            return Err(RuleError::NotAllowed {
                agent: agent.to_string(),
                message: "only agents with the propositional sensor may name a target by ID".into(),
            });
        }
        for (name, _) in &def.numbers {
            if !choice.numbers.contains_key(name) {
                return Err(RuleError::Usage(format!(
                    "`{}` needs the argument `{name}`",
                    choice.name
                )));
            }
        }
        // Choosing a new action interrupts the one in progress.
        let interrupting = self.ongoing.contains_key(agent);
        let resolved = actions::resolve(
            world,
            &def,
            info.entity,
            choice.target.as_deref(),
            interrupting,
        );
        if let Some(ongoing) = self.ongoing.remove(agent) {
            let old = self.def(&ongoing.action)?.clone();
            let bindings = self.bindings(world, agent, ongoing.target.as_deref());
            for effect in &old.spec.on_end {
                effects::apply(world, &bindings, effect, 0.0)?;
            }
            self.events.push(Event {
                tick,
                agent: Some(agent.to_string()),
                kind: EventKind::ActionInterrupted {
                    action: ongoing.action,
                    by: choice.name.clone(),
                },
            });
        }
        let target = match resolved {
            Ok(target) => target,
            Err(refusal) => {
                let reason = match self.impossible {
                    ImpossibleActions::Masked => format!("masked: {}", refusal.reason()),
                    ImpossibleActions::AttemptAndFail => refusal.reason(),
                };
                Self::set_failed(world, info.entity, true);
                self.events.push(Event {
                    tick,
                    agent: Some(agent.to_string()),
                    kind: EventKind::ActionFailed {
                        action: choice.name.clone(),
                        reason,
                    },
                });
                return Ok(());
            }
        };
        Self::set_failed(world, info.entity, false);
        let target_id = target.map(|t| world.ecs().get::<Id>(t).expect("id").name.clone());
        let bindings = Bindings::agent(info.entity).with_target(target);
        let mut numbers = BTreeMap::new();
        for (name, range) in &def.numbers {
            numbers.insert(name.clone(), choice.numbers[name].clamp(range[0], range[1]));
        }
        self.events.push(Event {
            tick,
            agent: Some(agent.to_string()),
            kind: EventKind::ActionStarted {
                action: choice.name.clone(),
                target: target_id.clone(),
            },
        });
        for effect in &def.spec.on_start {
            effects::apply(world, &bindings, effect, 0.0)?;
        }
        if def.is_durative() {
            self.ongoing.insert(
                agent.to_string(),
                Ongoing {
                    action: def.name.clone(),
                    target: target_id,
                    started_tick: tick,
                    ends_at_tick: def.duration_ticks.map(|d| tick + d),
                },
            );
            return Ok(());
        }
        if let Some(module) = &def.spec.module {
            modules::run(module, &def.spec.params, world, agent, &numbers)?;
        }
        for effect in def.once_effects() {
            effects::apply(world, &bindings, effect, 0.0)?;
        }
        for effect in &def.spec.on_end {
            effects::apply(world, &bindings, effect, 0.0)?;
        }
        self.events.push(Event {
            tick,
            agent: Some(agent.to_string()),
            kind: EventKind::ActionCompleted {
                action: choice.name.clone(),
            },
        });
        Ok(())
    }
}
