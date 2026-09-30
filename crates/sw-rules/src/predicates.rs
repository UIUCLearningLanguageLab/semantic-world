//! Registered predicates (addition A3). New predicates are added here, never as special
//! cases elsewhere.

use sw_schema::expr::{Arg, Call};

use crate::context::Ctx;
use crate::error::{Result, RuleError};

pub type PredicateFn = fn(&Ctx, &[Arg]) -> Result<bool>;

pub struct PredicateDef {
    pub name: &'static str,
    pub arity: usize,
    pub run: PredicateFn,
}

pub const PREDICATES: &[PredicateDef] = &[
    PredicateDef {
        name: "within",
        arity: 3,
        run: within,
    },
    PredicateDef {
        name: "facing",
        arity: 3,
        run: facing,
    },
    PredicateDef {
        name: "has_tag",
        arity: 2,
        run: has_tag,
    },
    PredicateDef {
        name: "stock_at_least",
        arity: 3,
        run: stock_at_least,
    },
    PredicateDef {
        name: "awake",
        arity: 1,
        run: awake,
    },
    PredicateDef {
        name: "asleep",
        arity: 1,
        run: asleep,
    },
    PredicateDef {
        name: "collapsed",
        arity: 1,
        run: collapsed,
    },
    PredicateDef {
        name: "need_at_most",
        arity: 3,
        run: need_at_most,
    },
];

pub fn lookup(name: &str) -> Option<&'static PredicateDef> {
    PREDICATES.iter().find(|p| p.name == name)
}

/// Check a call against the registry: the predicate exists and the arity matches.
pub fn check(call: &Call, where_: &str) -> Result<()> {
    let def = lookup(&call.name).ok_or_else(|| {
        RuleError::Invalid(format!("{where_}: unknown predicate `{}`", call.name))
    })?;
    if call.args.len() != def.arity {
        return Err(RuleError::Invalid(format!(
            "{where_}: `{}` takes {} arguments, not {}",
            call.name,
            def.arity,
            call.args.len()
        )));
    }
    Ok(())
}

/// Evaluate one call. An unbound reference (a missing target) makes the call false.
pub fn holds(ctx: &Ctx, call: &Call) -> Result<bool> {
    let def = lookup(&call.name)
        .ok_or_else(|| RuleError::Invalid(format!("unknown predicate `{}`", call.name)))?;
    match (def.run)(ctx, &call.args) {
        Err(RuleError::Unbound(_)) => Ok(false),
        other => other,
    }
}

/// Evaluate a list of calls: all must hold. Returns the first call that does not.
pub fn all_hold<'a>(ctx: &Ctx, calls: &'a [Call]) -> Result<Option<&'a Call>> {
    for call in calls {
        if !holds(ctx, call)? {
            return Ok(Some(call));
        }
    }
    Ok(None)
}

/// The gap between the surfaces of two entities on the ground plane.
pub fn surface_gap(ctx: &Ctx, a: bevy_ecs::prelude::Entity, b: bevy_ecs::prelude::Entity) -> f64 {
    let distance = ctx.position(a).distance(ctx.position(b));
    distance - ctx.footprint(a).radius - ctx.footprint(b).radius
}

fn within(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let a = ctx.entity(&args[0])?;
    let b = ctx.entity(&args[1])?;
    let d = ctx.number(&args[2])?;
    Ok(surface_gap(ctx, a, b) <= d)
}

fn facing(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let a = ctx.entity(&args[0])?;
    let b = ctx.entity(&args[1])?;
    let deg = ctx.number(&args[2])?;
    let to = ctx.position(b) - ctx.position(a);
    if to.length_squared() == 0.0 {
        return Ok(true);
    }
    let forward = ctx.facing(a).forward();
    let cos = (forward.dot(to) / to.length()).clamp(-1.0, 1.0);
    Ok(cos.acos().to_degrees() <= deg)
}

fn has_tag(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let x = ctx.entity(&args[0])?;
    let tag = ctx.symbol(&args[1])?;
    Ok(ctx.tags(x).has(tag))
}

fn stock_at_least(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let x = ctx.entity(&args[0])?;
    let name = ctx.symbol(&args[1])?;
    let n = ctx.number(&args[2])?;
    Ok(ctx
        .stocks(x)
        .0
        .get(name)
        .is_some_and(|s| s.count as f64 >= n))
}

fn awake(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let agent = ctx.agent(ctx.entity(&args[0])?)?;
    Ok(agent.alive && ctx.is_awake(agent))
}

fn asleep(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let agent = ctx.agent(ctx.entity(&args[0])?)?;
    Ok(agent.alive && !ctx.is_awake(agent))
}

fn collapsed(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let agent = ctx.agent(ctx.entity(&args[0])?)?;
    Ok(agent.collapsed())
}

fn need_at_most(ctx: &Ctx, args: &[Arg]) -> Result<bool> {
    let a = ctx.entity(&args[0])?;
    let need = ctx.symbol(&args[1])?;
    let v = ctx.number(&args[2])?;
    let needs = ctx
        .needs(a)
        .ok_or_else(|| RuleError::Arg(format!("{} has no needs", ctx.id(a))))?;
    let state = needs
        .get(need)
        .ok_or_else(|| RuleError::Arg(format!("{} has no need `{need}`", ctx.id(a))))?;
    Ok(state.value <= v)
}
