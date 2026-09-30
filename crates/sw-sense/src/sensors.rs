//! The sensor set of one agent, built from its type's `sensors` section.

use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use sw_core::components::{Agent, Contact, Facing, Health, Motion, Needs};
use sw_core::{DVec2, Entity, World, streams};
use sw_rules::Engine;
use sw_schema::Resolved;
use sw_schema::experiment::ImpossibleActions;

use crate::observation::{Block, BlockData, BlockInfo, Observation, SensorManifest};

/// The touch block's labels: contact intensity and direction in the agent's frame, where x
/// is forward, y is up, and z is to the right.
pub const TOUCH_LABELS: [&str; 4] = ["intensity", "dir_x", "dir_y", "dir_z"];
/// The proprioception block's labels.
pub const PROPRIOCEPTION_LABELS: [&str; 6] = [
    "speed",
    "asleep",
    "collapsed",
    "heading_sin",
    "heading_cos",
    "last_action_failed",
];

#[derive(Debug, Clone)]
struct Interoception {
    needs: Vec<String>,
    health: bool,
    noise_sd: f64,
}

/// One agent's sensors.
pub struct AgentSensors {
    agent: String,
    entity: Entity,
    manifest: SensorManifest,
    eyes: Option<(u32, u32, bool)>,
    interoception: Option<Interoception>,
    touch: bool,
    proprioception: bool,
    /// The speed that reads as 1.0 in proprioception: the fastest run speed any move module
    /// declares, or 1 m/s when none does.
    speed_scale: f64,
    masked: bool,
    rng: ChaCha8Rng,
}

impl AgentSensors {
    /// Build the sensors of an agent from its resolved type.
    pub fn new(resolved: &Resolved, world: &World, agent: &str) -> Result<AgentSensors, String> {
        let entity = world
            .entity(agent)
            .filter(|e| world.ecs().get::<Agent>(*e).is_some())
            .ok_or_else(|| format!("unknown agent `{agent}`"))?;
        let individual = world.individual(agent).ok_or("no individual")?;
        let sensors = &individual.entity_type.sensors;
        let needs = world.ecs().get::<Needs>(entity);
        let mut blocks = Vec::new();
        let mut offset = 0usize;
        let eyes = sensors
            .eyes
            .as_ref()
            .map(|e| (e.resolution[0], e.resolution[1], e.color));
        if let Some((w, h, color)) = eyes {
            blocks.push(BlockInfo {
                name: "eyes".into(),
                shape: vec![h as usize, w as usize, if color { 3 } else { 1 }],
                dtype: "uint8".into(),
                range: Some([0.0, 255.0]),
                in_vector: false,
                offset: None,
                labels: Vec::new(),
            });
        }
        let interoception = sensors.interoception.as_ref().map(|i| {
            let mut labels = i.needs.clone();
            if i.health {
                labels.push("health".into());
            }
            for need in &i.needs {
                assert!(
                    needs.is_some_and(|n| n.get(need).is_some()),
                    "validated: `{need}` is a need of the body"
                );
            }
            vector_block(
                &mut blocks,
                &mut offset,
                "interoception",
                labels,
                Some([0.0, 1.0]),
            );
            Interoception {
                needs: i.needs.clone(),
                health: i.health,
                noise_sd: *i.noise_sd.value(),
            }
        });
        let touch = sensors.touch.is_some();
        if touch {
            vector_block(
                &mut blocks,
                &mut offset,
                "touch",
                TOUCH_LABELS.iter().map(|s| s.to_string()).collect(),
                Some([-1.0, 1.0]),
            );
        }
        let proprioception = sensors.proprioception.is_some();
        if proprioception {
            vector_block(
                &mut blocks,
                &mut offset,
                "proprioception",
                PROPRIOCEPTION_LABELS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                Some([-1.0, 1.0]),
            );
        }
        let props =
            resolved.experiment.views.propositional_sensor && sensors.propositional.is_some();
        let speed_scale = resolved
            .rules
            .actions
            .values()
            .filter(|a| a.module.as_deref() == Some("kinematic_move"))
            .filter_map(|a| {
                sw_core::map::param_f64(&a.params, "run_speed_mps", "kinematic_move").ok()
            })
            .fold(0.0_f64, f64::max);
        let speed_scale = if speed_scale > 0.0 { speed_scale } else { 1.0 };
        let manifest = SensorManifest {
            agent: agent.to_string(),
            vector_length: offset,
            blocks,
            props,
        };
        Ok(AgentSensors {
            agent: agent.to_string(),
            entity,
            manifest,
            eyes,
            interoception,
            touch,
            proprioception,
            speed_scale,
            masked: resolved.experiment.options.impossible_actions == ImpossibleActions::Masked,
            rng: streams::stream(world.master_seed(), &streams::sense_stream_name(agent)),
        })
    }

    pub fn manifest(&self) -> &SensorManifest {
        &self.manifest
    }

    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// Whether the propositional sensor is on for this agent.
    pub fn props_on(&self) -> bool {
        self.manifest.props
    }

    /// Read every sensor. `eyes` is the rendered image for this agent, if a renderer produced
    /// one; otherwise the eyes block is black.
    pub fn observe(
        &mut self,
        world: &World,
        engine: &Engine,
        eyes: Option<Vec<u8>>,
    ) -> Result<Observation, String> {
        let ecs = world.ecs();
        let entity = self.entity;
        let mut blocks = Vec::new();
        let mut vector = Vec::with_capacity(self.manifest.vector_length);
        if let Some((w, h, color)) = self.eyes {
            let channels = if color { 3 } else { 1 };
            let len = (w * h) as usize * channels;
            let data = match eyes {
                Some(image) if image.len() == len => image,
                Some(image) => {
                    return Err(format!(
                        "eyes image has {} bytes, expected {len}",
                        image.len()
                    ));
                }
                None => vec![0u8; len],
            };
            blocks.push(Block {
                name: "eyes".into(),
                shape: vec![h as usize, w as usize, channels],
                data: BlockData::U8(data),
            });
        }
        if let Some(intero) = &self.interoception {
            let needs = ecs.get::<Needs>(entity).expect("needs");
            let mut values: Vec<f32> = intero
                .needs
                .iter()
                .map(|n| needs.get(n).map_or(0.0, |s| s.value) as f32)
                .collect();
            if intero.health {
                values.push(ecs.get::<Health>(entity).map_or(0.0, |h| h.value) as f32);
            }
            if intero.noise_sd > 0.0 {
                for v in &mut values {
                    let u1: f64 = self.rng.random();
                    let u2: f64 = self.rng.random();
                    let z =
                        (-2.0_f64 * (1.0 - u1).ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                    *v = (*v as f64 + intero.noise_sd * z).clamp(0.0, 1.0) as f32;
                }
            }
            vector.extend_from_slice(&values);
            blocks.push(Block {
                name: "interoception".into(),
                shape: vec![values.len()],
                data: BlockData::F32(values),
            });
        }
        if self.touch {
            let contact = ecs.get::<Contact>(entity).expect("contact");
            let facing = ecs.get::<Facing>(entity).expect("facing");
            let values = if contact.blocked {
                let forward = facing.forward();
                let right = glam_right(facing.yaw);
                [
                    1.0,
                    contact.direction.dot(forward) as f32,
                    0.0,
                    contact.direction.dot(right) as f32,
                ]
            } else {
                [0.0; 4]
            };
            vector.extend_from_slice(&values);
            blocks.push(Block {
                name: "touch".into(),
                shape: vec![4],
                data: BlockData::F32(values.to_vec()),
            });
        }
        if self.proprioception {
            let agent = ecs.get::<Agent>(entity).expect("agent");
            let motion = ecs.get::<Motion>(entity).expect("motion");
            let facing = ecs.get::<Facing>(entity).expect("facing");
            let values = [
                (motion.last_tick_speed_mps / self.speed_scale) as f32,
                f32::from(u8::from(agent.asleep)),
                f32::from(u8::from(agent.collapsed())),
                facing.yaw.sin() as f32,
                facing.yaw.cos() as f32,
                f32::from(u8::from(agent.last_action_failed)),
            ];
            vector.extend_from_slice(&values);
            blocks.push(Block {
                name: "proprioception".into(),
                shape: vec![6],
                data: BlockData::F32(values.to_vec()),
            });
        }
        let props = if self.manifest.props {
            Some(
                engine
                    .facts(world, &self.agent)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let mask = if self.masked {
            Some(engine.mask(world, &self.agent).map_err(|e| e.to_string())?)
        } else {
            None
        };
        Ok(Observation {
            blocks,
            vector,
            props,
            text: None,
            mask,
        })
    }
}

/// Append a float block that goes into the flat vector, at the next offset.
fn vector_block(
    blocks: &mut Vec<BlockInfo>,
    offset: &mut usize,
    name: &str,
    labels: Vec<String>,
    range: Option<[f64; 2]>,
) {
    let len = labels.len();
    blocks.push(BlockInfo {
        name: name.to_string(),
        shape: vec![len],
        dtype: "float32".into(),
        range,
        in_vector: true,
        offset: Some(*offset),
        labels,
    });
    *offset += len;
}

/// The agent's right-hand direction on the ground plane, for a facing yaw. With y up and
/// forward `(cos yaw, sin yaw)`, right is forward × up.
fn glam_right(yaw: f64) -> DVec2 {
    DVec2::new(-yaw.sin(), yaw.cos())
}
