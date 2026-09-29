//! The world: one canonical state, stepped one tick at a time.
//!
//! The core is single-threaded within one world. Every simulation loop visits entities in
//! creation order through the [`Registry`], never through hash-map iteration.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Entity;
use glam::DVec2;
use rand_chacha::ChaCha8Rng;
use sw_schema::Resolved;

use crate::clock::Clock;
use crate::components::{
    Agent, Blocking, Contact, Facing, Footprint, Health, Id, Individual, Motion, Needs, Position,
    Stocks, Tags,
};
use crate::error::{CoreError, Result};
use crate::map;
use crate::needs::{self, Conditions};
use crate::snapshot::{AgentSnapshot, EntitySnapshot, StateHash, StateSnapshot};
use crate::space::{self, MapBounds, Obstacle};
use crate::streams;

/// Every entity in creation order, plus lookups by ID.
#[derive(Debug, Default, Clone)]
pub struct Registry {
    /// All entities, in creation order.
    pub entities: Vec<Entity>,
    pub by_id: BTreeMap<String, Entity>,
    /// Agents, in creation order.
    pub agents: Vec<Entity>,
    /// Entities whose footprint is walls, in creation order.
    pub walled: Vec<Entity>,
}

pub struct World {
    ecs: bevy_ecs::world::World,
    registry: Registry,
    bounds: MapBounds,
    master_seed: u64,
    /// The `rules` stream, for rules that draw random numbers.
    rules_rng: ChaCha8Rng,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("tick", &self.clock().tick)
            .field("entities", &self.registry.entities.len())
            .field("agents", &self.registry.agents.len())
            .field("master_seed", &self.master_seed)
            .finish_non_exhaustive()
    }
}

impl World {
    /// Build a world from a resolved experiment and a master seed: place the map, sample the
    /// individuals, and start the clock at tick 0.
    pub fn new(resolved: &Resolved, master_seed: u64) -> Result<World> {
        let clock = Clock::new(
            resolved.experiment.clock.tick_s,
            resolved.experiment.clock.decision_every_ticks,
            resolved.calendar.clone(),
            resolved.experiment.lifetime.default_days,
        );
        let agents: Vec<(String, u32)> = resolved
            .population
            .iter()
            .map(|p| (p.type_name.clone(), p.count))
            .collect();
        let mut map_rng = streams::stream(master_seed, streams::MAP);
        let layout = map::generate(&resolved.world.map, &agents, &mut map_rng)?;

        let mut ecs = bevy_ecs::world::World::new();
        let mut registry = Registry::default();
        let mut traits_rng = streams::stream(master_seed, streams::TRAITS);
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        // Objects come from the type table; agents use their population entry's type, which
        // carries the entry's overrides and nervous system.
        let object_count =
            layout.placements.len() - agents.iter().map(|(_, c)| *c as usize).sum::<usize>();
        let mut population_types = resolved
            .population
            .iter()
            .flat_map(|p| std::iter::repeat_n(&p.entity_type, p.count as usize));
        for (i, placement) in layout.placements.iter().enumerate() {
            let entity_type = if i < object_count {
                resolved.types.get(&placement.type_name).ok_or_else(|| {
                    CoreError::Invalid(format!("unknown type `{}`", placement.type_name))
                })?
            } else {
                population_types
                    .next()
                    .expect("one type per agent placement")
            };
            let index = counts.entry(placement.type_name.clone()).or_insert(0);
            let entity = crate::build::spawn(
                &mut ecs,
                &clock,
                entity_type,
                *index,
                placement.position,
                placement.yaw,
                &mut traits_rng,
            )?;
            *index += 1;
            let id = ecs.get::<Id>(entity).expect("just spawned").name.clone();
            registry.by_id.insert(id, entity);
            registry.entities.push(entity);
            if ecs.get::<Agent>(entity).is_some() {
                registry.agents.push(entity);
            }
            if matches!(
                ecs.get::<Footprint>(entity).expect("footprint").blocking,
                Blocking::Walls { .. }
            ) {
                registry.walled.push(entity);
            }
        }
        ecs.insert_resource(clock);
        Ok(World {
            ecs,
            registry,
            bounds: layout.bounds,
            master_seed,
            rules_rng: streams::stream(master_seed, streams::RULES),
        })
    }

    pub fn master_seed(&self) -> u64 {
        self.master_seed
    }

    pub fn clock(&self) -> &Clock {
        self.ecs.resource::<Clock>()
    }

    pub fn bounds(&self) -> MapBounds {
        self.bounds
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The ECS itself, for the rule engine and the sensors. Not for agents.
    pub fn ecs(&self) -> &bevy_ecs::world::World {
        &self.ecs
    }

    pub fn ecs_mut(&mut self) -> &mut bevy_ecs::world::World {
        &mut self.ecs
    }

    /// The `rules` random stream.
    pub fn rules_rng(&mut self) -> &mut ChaCha8Rng {
        &mut self.rules_rng
    }

    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.registry.by_id.get(id).copied()
    }

    fn agent_entity(&self, id: &str) -> Result<Entity> {
        self.entity(id)
            .filter(|e| self.ecs.get::<Agent>(*e).is_some())
            .ok_or_else(|| CoreError::UnknownAgent(id.to_string()))
    }

    /// Every entity's ID, in creation order.
    pub fn entity_ids(&self) -> Vec<String> {
        self.registry
            .entities
            .iter()
            .map(|e| self.ecs.get::<Id>(*e).expect("id").name.clone())
            .collect()
    }

    /// Every agent's ID, in creation order.
    pub fn agent_ids(&self) -> Vec<String> {
        self.registry
            .agents
            .iter()
            .map(|e| self.ecs.get::<Id>(*e).expect("id").name.clone())
            .collect()
    }

    /// The seed Python agents draw their randomness from: the `agent:<id>` stream.
    pub fn agent_seed(&self, id: &str) -> Result<[u8; 32]> {
        self.agent_entity(id)?;
        Ok(streams::stream_seed(
            self.master_seed,
            &streams::agent_stream_name(id),
        ))
    }

    /// The agents that must act before the next step: the living, uncollapsed agents, at a
    /// decision point. Between decision points, none.
    pub fn agents_awaiting_action(&self) -> Vec<String> {
        if !self.clock().is_decision_point() {
            return Vec::new();
        }
        self.registry
            .agents
            .iter()
            .filter(|e| {
                let a = self.ecs.get::<Agent>(**e).expect("agent");
                a.alive && !a.collapsed()
            })
            .map(|e| self.ecs.get::<Id>(*e).expect("id").name.clone())
            .collect()
    }

    pub fn any_agent_alive(&self) -> bool {
        self.registry
            .agents
            .iter()
            .any(|e| self.ecs.get::<Agent>(*e).expect("agent").alive)
    }

    /// The run is over when every agent is dead or the lifetime has ended.
    pub fn done(&self) -> bool {
        self.clock().lifetime_over() || !self.any_agent_alive()
    }

    /// Set an agent's facing and its motion for the coming decision interval: it moves along
    /// the new facing at `speed_mps` for `decision_every_ticks` ticks. `running` marks the
    /// motion as running for the needs mechanism. A sleeping, collapsed, or dead agent
    /// cannot move.
    pub fn set_motion(&mut self, id: &str, yaw: f64, speed_mps: f64, running: bool) -> Result<()> {
        let entity = self.agent_entity(id)?;
        let agent = self.ecs.get::<Agent>(entity).expect("agent");
        if !agent.alive || agent.asleep {
            return Err(CoreError::Invalid(format!(
                "{id} cannot move: asleep or dead"
            )));
        }
        let ticks = self.clock().decision_every_ticks;
        let mut entity_mut = self.ecs.entity_mut(entity);
        entity_mut.get_mut::<Facing>().expect("facing").yaw = Facing::wrap(yaw);
        *entity_mut.get_mut::<Motion>().expect("motion") = Motion {
            speed_mps: speed_mps.max(0.0),
            running,
            ticks_left: ticks,
            last_tick_speed_mps: 0.0,
        };
        Ok(())
    }

    /// Turn an agent in place.
    pub fn set_facing(&mut self, id: &str, yaw: f64) -> Result<()> {
        let entity = self.agent_entity(id)?;
        self.ecs
            .entity_mut(entity)
            .get_mut::<Facing>()
            .expect("facing")
            .yaw = Facing::wrap(yaw);
        Ok(())
    }

    /// Stop an agent's motion.
    pub fn stop(&mut self, id: &str) -> Result<()> {
        let entity = self.agent_entity(id)?;
        *self
            .ecs
            .entity_mut(entity)
            .get_mut::<Motion>()
            .expect("motion") = Motion::still();
        Ok(())
    }

    /// Run the ticks up to the next decision point.
    pub fn advance_decision(&mut self) {
        let k = self.clock().decision_every_ticks;
        for _ in 0..k {
            self.tick();
        }
    }

    /// Advance the world by one tick: motion and blocking, then the needs and health of
    /// every living agent, in creation order, then the clock.
    pub fn tick(&mut self) {
        let clock = self.clock().clone();
        let dt = clock.tick_s;
        let night = clock.is_night();
        let agents = self.registry.agents.clone();
        for entity in agents {
            let agent = self.ecs.get::<Agent>(entity).expect("agent").clone();
            if !agent.alive {
                continue;
            }
            // Motion and blocking.
            let mut motion = self.ecs.get::<Motion>(entity).expect("motion").clone();
            let mut contact = Contact::none();
            let mut running = false;
            let mut speed = 0.0;
            if motion.ticks_left > 0 {
                motion.ticks_left -= 1;
                if motion.speed_mps > 0.0 {
                    let position = self.ecs.get::<Position>(entity).expect("position").0;
                    let facing = *self.ecs.get::<Facing>(entity).expect("facing");
                    let radius = self.ecs.get::<Footprint>(entity).expect("footprint").radius;
                    let obstacles = self.obstacles_except(entity);
                    let delta = facing.forward() * motion.speed_mps * dt;
                    let result = space::try_move(position, delta, radius, &obstacles, &self.bounds);
                    self.ecs
                        .entity_mut(entity)
                        .get_mut::<Position>()
                        .expect("position")
                        .0 = result.position;
                    contact = Contact {
                        blocked: result.blocked,
                        direction: result.contact_direction,
                    };
                    running = motion.running;
                    speed = motion.speed_mps;
                }
            }
            motion.last_tick_speed_mps = speed;
            let mut entity_mut = self.ecs.entity_mut(entity);
            *entity_mut.get_mut::<Motion>().expect("motion") = motion;
            *entity_mut.get_mut::<Contact>().expect("contact") = contact;

            // Collapse ends on its tick.
            let mut agent = agent;
            if let Some(until) = agent.collapsed_until_tick
                && clock.tick >= until
            {
                agent.collapsed_until_tick = None;
                agent.asleep = false;
            }
            // Shelter.
            let position = self.ecs.get::<Position>(entity).expect("position").0;
            agent.in_shelter = self.inside_any_walled(position);

            // Needs and health.
            let cond = Conditions {
                asleep: agent.asleep,
                running,
                in_shelter: agent.in_shelter,
                night,
            };
            if self.ecs.get::<Needs>(entity).is_some() {
                let mut needs = self.ecs.get::<Needs>(entity).expect("needs").clone();
                let mut health = self.ecs.get::<Health>(entity).expect("health").clone();
                needs::update(
                    &mut needs,
                    &mut health,
                    &mut agent,
                    cond,
                    dt,
                    clock.tick,
                    dt,
                );
                let mut entity_mut = self.ecs.entity_mut(entity);
                *entity_mut.get_mut::<Needs>().expect("needs") = needs;
                *entity_mut.get_mut::<Health>().expect("health") = health;
            }
            *self
                .ecs
                .entity_mut(entity)
                .get_mut::<Agent>()
                .expect("agent") = agent;
        }
        self.ecs.resource_mut::<Clock>().tick += 1;
    }

    /// Every solid entity except one, as obstacles.
    fn obstacles_except(&self, mover: Entity) -> Vec<Obstacle> {
        self.registry
            .entities
            .iter()
            .filter(|e| **e != mover)
            .filter_map(|e| {
                let footprint = self.ecs.get::<Footprint>(*e)?;
                if !footprint.solid {
                    return None;
                }
                Some(Obstacle::new(
                    self.ecs.get::<Position>(*e)?,
                    self.ecs.get::<Facing>(*e)?,
                    footprint,
                ))
            })
            .collect()
    }

    /// Whether a point is within the footprint of any walled entity (a shelter).
    pub fn inside_any_walled(&self, point: DVec2) -> bool {
        self.registry.walled.iter().any(|e| {
            Obstacle::new(
                self.ecs.get::<Position>(*e).expect("position"),
                self.ecs.get::<Facing>(*e).expect("facing"),
                self.ecs.get::<Footprint>(*e).expect("footprint"),
            )
            .contains(point)
        })
    }

    /// The canonical state at this tick. **Privileged:** only scripted agents, tests, logs,
    /// and viewers may read the full state. Learning agents see only their observations.
    pub fn snapshot(&self) -> StateSnapshot {
        let clock = self.clock();
        let mut entities: Vec<EntitySnapshot> = self
            .registry
            .entities
            .iter()
            .map(|e| self.entity_snapshot(*e))
            .collect();
        entities.sort_by(|a, b| (&a.type_name, a.index).cmp(&(&b.type_name, b.index)));
        StateSnapshot {
            tick: clock.tick,
            time_s: clock.time_s(),
            day: clock.day(),
            hour_of_day: clock.hour_of_day(),
            light_level: clock.light_level(),
            night: clock.is_night(),
            entities,
        }
    }

    fn entity_snapshot(&self, entity: Entity) -> EntitySnapshot {
        let id = self.ecs.get::<Id>(entity).expect("id");
        let position = self.ecs.get::<Position>(entity).expect("position").0;
        let facing = self.ecs.get::<Facing>(entity).expect("facing");
        let footprint = self.ecs.get::<Footprint>(entity).expect("footprint");
        let tags = self.ecs.get::<Tags>(entity).expect("tags");
        let stocks = self.ecs.get::<Stocks>(entity).expect("stocks");
        let needs = self.ecs.get::<Needs>(entity);
        let health = self.ecs.get::<Health>(entity);
        let agent = self.ecs.get::<Agent>(entity).map(|a| {
            let motion = self.ecs.get::<Motion>(entity).expect("motion");
            let contact = self.ecs.get::<Contact>(entity).expect("contact");
            AgentSnapshot {
                alive: a.alive,
                died_at_tick: a.died_at_tick,
                asleep: a.asleep,
                collapsed_until_tick: a.collapsed_until_tick,
                in_shelter: a.in_shelter,
                last_action_failed: a.last_action_failed,
                speed_mps: if motion.moving() {
                    motion.speed_mps
                } else {
                    0.0
                },
                running: motion.moving() && motion.running,
                blocked: contact.blocked,
                contact_direction: [contact.direction.x, contact.direction.y],
            }
        });
        EntitySnapshot {
            id: id.name.clone(),
            type_name: id.type_name.clone(),
            index: id.index,
            x: position.x,
            z: position.y,
            yaw: facing.yaw,
            radius: footprint.radius,
            height: footprint.height_m,
            solid: footprint.solid,
            tags: tags.0.clone(),
            stocks: stocks
                .0
                .iter()
                .map(|(n, s)| (n.clone(), s.count, s.max))
                .collect(),
            needs: needs
                .map(|n| n.needs.iter().map(|s| (s.name.clone(), s.value)).collect())
                .unwrap_or_default(),
            health: health.map(|h| h.value),
            agent,
        }
    }

    /// The state hash at this tick.
    pub fn state_hash(&self) -> StateHash {
        self.snapshot().hash()
    }

    /// The sampled type of an entity. **Privileged**, like [`World::snapshot`].
    pub fn individual(&self, id: &str) -> Option<&Individual> {
        self.entity(id).and_then(|e| self.ecs.get::<Individual>(e))
    }

    /// Put an entity somewhere. **Privileged:** for tests and scripted scenarios only.
    pub fn privileged_set_pose(&mut self, id: &str, x: f64, z: f64, yaw: f64) -> Result<()> {
        let entity = self
            .entity(id)
            .ok_or_else(|| CoreError::UnknownAgent(id.to_string()))?;
        let mut entity_mut = self.ecs.entity_mut(entity);
        entity_mut.get_mut::<Position>().expect("position").0 = DVec2::new(x, z);
        entity_mut.get_mut::<Facing>().expect("facing").yaw = Facing::wrap(yaw);
        Ok(())
    }

    /// Set an agent's sleep flag. Rules do this through the `set_state` effect; tests do it
    /// directly. **Privileged** for tests.
    pub fn set_asleep(&mut self, id: &str, asleep: bool) -> Result<()> {
        let entity = self.agent_entity(id)?;
        let mut entity_mut = self.ecs.entity_mut(entity);
        entity_mut.get_mut::<Agent>().expect("agent").asleep = asleep;
        if asleep {
            *entity_mut.get_mut::<Motion>().expect("motion") = Motion::still();
        }
        Ok(())
    }
}
