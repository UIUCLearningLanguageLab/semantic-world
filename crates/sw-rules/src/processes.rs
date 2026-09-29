//! Processes: things that happen without any action, on a schedule (contract 2).

use sw_core::World;
use sw_core::components::{Id, Tags};
use sw_schema::rules::ProcessSpec;

use crate::context::Bindings;
use crate::effects;
use crate::error::{Result, RuleError};
use crate::events::{Event, EventKind};

#[derive(Debug, Clone)]
pub struct ProcessDef {
    pub name: String,
    pub spec: ProcessSpec,
    /// The period in ticks. The process fires on every tick that is a whole number of
    /// periods from the start of the run, except tick 0.
    pub period_ticks: u64,
}

impl ProcessDef {
    pub fn compile(name: &str, spec: &ProcessSpec, tick_s: f64) -> Result<ProcessDef> {
        let where_ = format!("processes.{name}");
        for (i, effect) in spec.effects.iter().enumerate() {
            let def = effects::check(effect, &format!("{where_}.effects[{i}]"))?;
            if def.timing == effects::Timing::Rate {
                return Err(RuleError::Invalid(format!(
                    "{where_}.effects[{i}]: `{}` is a rate effect; processes apply once",
                    effect.name
                )));
            }
        }
        let period_ticks = (spec.every.value() / tick_s).round().max(1.0) as u64;
        Ok(ProcessDef {
            name: name.to_string(),
            spec: spec.clone(),
            period_ticks,
        })
    }

    pub fn fires_at(&self, tick: u64) -> bool {
        tick > 0 && tick % self.period_ticks == 0
    }

    /// Apply the process to every entity it applies to, if it fires at this tick.
    pub fn run(&self, world: &mut World, tick: u64, events: &mut Vec<Event>) -> Result<()> {
        if !self.fires_at(tick) {
            return Ok(());
        }
        let entities: Vec<_> = world
            .registry()
            .entities
            .iter()
            .copied()
            .filter(|e| {
                world
                    .ecs()
                    .get::<Tags>(*e)
                    .is_some_and(|t| t.has(&self.spec.applies_to.has))
            })
            .collect();
        for entity in entities {
            let bindings = Bindings::this(entity);
            for effect in &self.spec.effects {
                effects::apply(world, &bindings, effect, 0.0)?;
            }
            events.push(Event {
                tick,
                agent: None,
                kind: EventKind::ProcessFired {
                    process: self.name.clone(),
                    entity: world.ecs().get::<Id>(entity).expect("id").name.clone(),
                },
            });
        }
        Ok(())
    }
}
