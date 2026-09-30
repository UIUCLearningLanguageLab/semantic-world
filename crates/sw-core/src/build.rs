//! Building entities from types: sampling the individual, converting its traits to the
//! components the mechanisms read, and giving it an ID.

use bevy_ecs::prelude::Entity;
use glam::DVec2;
use indexmap::IndexMap;
use rand_chacha::ChaCha8Rng;
use sw_schema::entity::{Collision, EntityType};

use crate::clock::Clock;
use crate::components::{
    Agent, Blocking, Contact, Facing, Footprint, Health, Id, Individual, Motion, NeedRates,
    NeedState, Needs, Position, Stock, Stocks, Tags,
};
use crate::error::{CoreError, Result};

/// Sample one individual from a type and spawn it at a position.
pub fn spawn(
    ecs: &mut bevy_ecs::world::World,
    clock: &Clock,
    entity_type: &EntityType,
    index: u32,
    position: DVec2,
    yaw: f64,
    traits_rng: &mut ChaCha8Rng,
) -> Result<Entity> {
    let (individual, sampled) = entity_type.sample_individual(traits_rng);
    let body = &individual.body;
    let id = Id {
        name: format!("{}_{}", individual.name, index),
        type_name: individual.name.clone(),
        index,
    };
    let blocking = match &body.collision {
        Collision::Circle => Blocking::Circle,
        Collision::Walls { thickness } => {
            let part = body
                .placeholder
                .as_ref()
                .and_then(|p| p.parts().first().copied().cloned())
                .ok_or_else(|| {
                    CoreError::Invalid(format!(
                        "{}: wall collision needs a placeholder footprint",
                        id.name
                    ))
                })?;
            Blocking::Walls {
                half_forward: part.size[0] / 2.0,
                half_side: part.size[2] / 2.0,
                thickness: *thickness.value(),
            }
        }
    };
    let placeholder_top = body
        .placeholder
        .as_ref()
        .map(|p| {
            p.parts()
                .iter()
                .map(|part| part.center()[1] + part.size[1] / 2.0)
                .fold(0.0, f64::max)
        })
        .unwrap_or(0.0);
    let footprint = Footprint {
        radius: body.radius_m(),
        solid: body.solid,
        blocking,
        height_m: body.height.as_ref().map_or(placeholder_top, |h| *h.value()),
    };
    let stocks: IndexMap<String, Stock> = body
        .stock
        .iter()
        .map(|(name, s)| {
            (
                name.clone(),
                Stock {
                    count: *s.initial.value(),
                    max: *s.max.value(),
                },
            )
        })
        .collect();
    let is_agent = individual.nervous_system.is_some();
    let has_needs = body.has_needs();
    let needs = Needs {
        needs: body
            .needs
            .iter()
            .map(|(name, spec)| NeedState {
                name: name.clone(),
                value: *spec.initial.value(),
                rates: NeedRates::from_spec(
                    spec,
                    clock.seconds_per_day(),
                    clock.seconds_per_hour(),
                ),
            })
            .collect(),
        insulation: body.insulation_value(),
    };
    let health = body.health.as_ref().map(|h| Health {
        value: *h.initial.value(),
        recover_per_s: *h.recover_per_hour.value() / clock.seconds_per_hour(),
    });

    let mut spawned = ecs.spawn((
        id,
        Position(position),
        Facing { yaw },
        footprint,
        Tags(body.tags.clone()),
        Stocks(stocks),
        Individual {
            entity_type: individual.clone(),
            sampled,
        },
    ));
    if has_needs {
        let health = health.ok_or_else(|| {
            CoreError::Invalid(format!(
                "{}: a body with needs declares health",
                individual.name
            ))
        })?;
        spawned.insert((needs, health));
    }
    if is_agent {
        spawned.insert((
            Agent {
                alive: true,
                died_at_tick: None,
                asleep: false,
                collapsed_until_tick: None,
                in_shelter: false,
                last_action_failed: false,
            },
            Motion::still(),
            Contact::none(),
        ));
    }
    Ok(spawned.id())
}
