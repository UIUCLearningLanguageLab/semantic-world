//! One run of one world: the world, the rule engine, and every agent's sensors, stepped
//! together. Pure Rust; the Python bindings wrap this.

use std::collections::BTreeMap;

use serde_json::{Value, json};
use sw_core::World;
use sw_rules::{ActionChoice, ActionManifest, Engine, Event};
use sw_schema::Resolved;
use sw_sense::{AgentSensors, Observation, SensorManifest};

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error(transparent)]
    Schema(#[from] sw_schema::SchemaError),
    #[error(transparent)]
    Core(#[from] sw_core::CoreError),
    #[error(transparent)]
    Rules(#[from] sw_rules::RuleError),
    #[error("{0}")]
    Sense(String),
    #[error("{0}")]
    Usage(String),
}

pub type Result<T> = std::result::Result<T, SessionError>;

pub struct Session {
    world: World,
    engine: Engine,
    sensors: BTreeMap<String, AgentSensors>,
}

impl Session {
    /// Build a world, its rules, and its agents' sensors for a master seed.
    pub fn new(resolved: &Resolved, seed: u64) -> Result<Session> {
        let world = World::new(resolved, seed)?;
        let engine = Engine::new(resolved, &world)?;
        let mut sensors = BTreeMap::new();
        for agent in world.agent_ids() {
            let s = AgentSensors::new(resolved, &world, &agent).map_err(SessionError::Sense)?;
            sensors.insert(agent, s);
        }
        Ok(Session {
            world,
            engine,
            sensors,
        })
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// Every agent's ID, in creation order.
    pub fn agents(&self) -> Vec<String> {
        self.world.agent_ids()
    }

    pub fn agents_awaiting_action(&self) -> Vec<String> {
        self.world.agents_awaiting_action()
    }

    pub fn done(&self) -> bool {
        self.world.done()
    }

    /// The sensor and action manifests of an agent.
    pub fn manifests(&self, agent: &str) -> Result<(SensorManifest, ActionManifest)> {
        let sensors = self
            .sensors
            .get(agent)
            .ok_or_else(|| SessionError::Usage(format!("unknown agent `{agent}`")))?;
        Ok((sensors.manifest().clone(), self.engine.manifest(agent)?))
    }

    /// Observations for every living agent, in creation order.
    pub fn observe(&mut self) -> Result<BTreeMap<String, Observation>> {
        let mut observations = BTreeMap::new();
        for agent in self.world.agent_ids() {
            let entity = self.world.entity(&agent).expect("agent");
            let alive = self
                .world
                .ecs()
                .get::<sw_core::components::Agent>(entity)
                .is_some_and(|a| a.alive);
            if !alive {
                continue;
            }
            let sensors = self.sensors.get_mut(&agent).expect("sensors");
            let observation = sensors
                .observe(&self.world, &self.engine, None)
                .map_err(SessionError::Sense)?;
            observations.insert(agent, observation);
        }
        Ok(observations)
    }

    /// One decision for the world: apply the choices, advance to the next decision point,
    /// and observe. An awaiting agent without a choice continues with noop.
    pub fn step(
        &mut self,
        choices: &BTreeMap<String, ActionChoice>,
    ) -> Result<(BTreeMap<String, Observation>, Vec<Event>)> {
        if self.world.done() {
            return Err(SessionError::Usage("the run is done; call reset".into()));
        }
        let events = self.engine.step(&mut self.world, choices)?;
        let observations = self.observe()?;
        Ok((observations, events))
    }

    /// The run's clock and every agent's status.
    pub fn info(&self) -> Value {
        let snapshot = self.world.snapshot();
        let agents: serde_json::Map<String, Value> = self
            .world
            .agent_ids()
            .into_iter()
            .map(|id| {
                let e = snapshot.entity(&id).expect("agent");
                let a = e.agent.as_ref().expect("agent");
                let current = self.engine.current_action(&id).map(|o| o.action.clone());
                (
                    id,
                    json!({
                        "alive": a.alive,
                        "asleep": a.asleep,
                        "collapsed": a.collapsed_until_tick.is_some(),
                        "health": e.health,
                        "current_action": current,
                    }),
                )
            })
            .collect();
        json!({
            "tick": snapshot.tick,
            "time_s": snapshot.time_s,
            "day": snapshot.day,
            "hour_of_day": snapshot.hour_of_day,
            "light_level": snapshot.light_level,
            "night": snapshot.night,
            "done": self.world.done(),
            "agents": agents,
        })
    }
}
