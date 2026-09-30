//! What a rule sees when it runs: the world, and what `self`, `target`, and `this` refer to.

use bevy_ecs::prelude::Entity;
use sw_core::World;
use sw_core::components::{
    Agent, Facing, Footprint, Health, Id, Individual, Needs, Position, Stocks, Tags,
};
use sw_schema::expr::{Arg, RefRoot, Reference};

use crate::error::{Result, RuleError};

/// The entities bound to the three reference roots.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bindings {
    pub self_entity: Option<Entity>,
    pub target: Option<Entity>,
    pub this: Option<Entity>,
}

impl Bindings {
    pub fn agent(self_entity: Entity) -> Self {
        Bindings {
            self_entity: Some(self_entity),
            ..Default::default()
        }
    }

    pub fn with_target(mut self, target: Option<Entity>) -> Self {
        self.target = target;
        self
    }

    pub fn this(this: Entity) -> Self {
        Bindings {
            this: Some(this),
            ..Default::default()
        }
    }

    pub fn resolve(&self, root: RefRoot) -> Result<Entity> {
        match root {
            RefRoot::SelfRef => self.self_entity.ok_or(RuleError::Unbound("self")),
            RefRoot::Target => self.target.ok_or(RuleError::Unbound("target")),
            RefRoot::This => self.this.ok_or(RuleError::Unbound("this")),
        }
    }
}

/// A read-only view for predicates and detectors.
pub struct Ctx<'w> {
    pub world: &'w World,
    pub bindings: Bindings,
    /// True while judging a new choice or building the mask for an agent that is in a durative
    /// action. Choosing interrupts the action, so an agent asleep by choice counts as awake.
    pub interrupting: bool,
}

impl<'w> Ctx<'w> {
    pub fn new(world: &'w World, bindings: Bindings) -> Self {
        Ctx {
            world,
            bindings,
            interrupting: false,
        }
    }

    pub fn interrupting(mut self) -> Self {
        self.interrupting = true;
        self
    }

    /// The entity an argument refers to.
    pub fn entity(&self, arg: &Arg) -> Result<Entity> {
        match arg {
            Arg::Ref(Reference {
                root,
                path,
                negated,
            }) if path.is_empty() && !negated => self.bindings.resolve(*root),
            other => Err(RuleError::Arg(format!(
                "`{other}` is not an entity reference"
            ))),
        }
    }

    pub fn number(&self, arg: &Arg) -> Result<f64> {
        number(self.world, &self.bindings, arg)
    }

    pub fn symbol<'a>(&self, arg: &'a Arg) -> Result<&'a str> {
        symbol(arg)
    }

    pub fn id(&self, entity: Entity) -> &'w str {
        &self.world.ecs().get::<Id>(entity).expect("id").name
    }

    pub fn position(&self, entity: Entity) -> glam::DVec2 {
        self.world
            .ecs()
            .get::<Position>(entity)
            .expect("position")
            .0
    }

    pub fn facing(&self, entity: Entity) -> &'w Facing {
        self.world.ecs().get::<Facing>(entity).expect("facing")
    }

    pub fn footprint(&self, entity: Entity) -> &'w Footprint {
        self.world
            .ecs()
            .get::<Footprint>(entity)
            .expect("footprint")
    }

    pub fn tags(&self, entity: Entity) -> &'w Tags {
        self.world.ecs().get::<Tags>(entity).expect("tags")
    }

    pub fn stocks(&self, entity: Entity) -> &'w Stocks {
        self.world.ecs().get::<Stocks>(entity).expect("stocks")
    }

    pub fn agent(&self, entity: Entity) -> Result<&'w Agent> {
        self.world
            .ecs()
            .get::<Agent>(entity)
            .ok_or_else(|| RuleError::Arg(format!("{} is not an agent", self.id(entity))))
    }

    pub fn needs(&self, entity: Entity) -> Option<&'w Needs> {
        self.world.ecs().get::<Needs>(entity)
    }

    pub fn individual(&self, entity: Entity) -> &'w Individual {
        self.world
            .ecs()
            .get::<Individual>(entity)
            .expect("individual")
    }

    /// Whether the agent counts as awake for the rule being judged.
    pub fn is_awake(&self, agent: &Agent) -> bool {
        !agent.collapsed() && (!agent.asleep || self.interrupting)
    }
}

pub fn symbol(arg: &Arg) -> Result<&str> {
    match arg {
        Arg::Symbol(s) => Ok(s),
        other => Err(RuleError::Arg(format!("`{other}` is not a name"))),
    }
}

pub fn boolean(arg: &Arg) -> Result<bool> {
    match arg {
        Arg::Bool(b) => Ok(*b),
        other => Err(RuleError::Arg(format!("`{other}` is not true or false"))),
    }
}

/// A number: a literal, or a field of a bound entity such as `target.provides.hunger` or
/// `this.stock.berries.max`. Dynamic state (`needs.<name>`, `health`, `stock.<name>.count`)
/// is read from the components; everything else from the individual's type, where a trait
/// gives its value.
pub fn number(world: &World, bindings: &Bindings, arg: &Arg) -> Result<f64> {
    match arg {
        Arg::Number(n) => Ok(*n),
        Arg::Ref(reference) => {
            let entity = bindings.resolve(reference.root)?;
            let value = field(world, entity, &reference.path)?;
            Ok(if reference.negated { -value } else { value })
        }
        other => Err(RuleError::Arg(format!("`{other}` is not a number"))),
    }
}

fn field(world: &World, entity: Entity, path: &[String]) -> Result<f64> {
    let ecs = world.ecs();
    let id = &ecs.get::<Id>(entity).expect("id").name;
    let segments: Vec<&str> = path.iter().map(String::as_str).collect();
    let missing = || RuleError::Arg(format!("{id} has no field `{}`", path.join(".")));
    match segments.as_slice() {
        [] => Err(RuleError::Arg(format!("`{id}` is an entity, not a number"))),
        ["needs", name] => ecs
            .get::<Needs>(entity)
            .and_then(|n| n.get(name))
            .map(|n| n.value)
            .ok_or_else(missing),
        ["health"] => ecs
            .get::<Health>(entity)
            .map(|h| h.value)
            .ok_or_else(missing),
        ["stock", name, "count"] => ecs
            .get::<Stocks>(entity)
            .and_then(|s| s.0.get(*name))
            .map(|s| s.count as f64)
            .ok_or_else(missing),
        ["stock", name, "max"] => ecs
            .get::<Stocks>(entity)
            .and_then(|s| s.0.get(*name))
            .map(|s| s.max as f64)
            .ok_or_else(missing),
        _ => {
            let individual = ecs.get::<Individual>(entity).expect("individual");
            let body =
                serde_json::to_value(&individual.entity_type.body).expect("a body serializes");
            let mut node = &body;
            for segment in &segments {
                node = node.get(*segment).ok_or_else(missing)?;
            }
            // A trait in full form carries its value under `default`.
            if let Some(default) = node.get("default") {
                node = default;
            }
            node.as_f64()
                .ok_or_else(|| RuleError::Arg(format!("{id}.{} is not a number", path.join("."))))
        }
    }
}
