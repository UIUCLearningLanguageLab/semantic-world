//! Loading an experiment and everything it refers to: the world, the rules, and the types,
//! with `extends` and population overrides applied and everything validated.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_json::Value;

use crate::entity::{EntityType, NervousSystem};
use crate::error::{Ctx, Result, SchemaError};
use crate::experiment::{ExperimentConfig, WorldRef};
use crate::rules::RulesFile;
use crate::world::{Calendar, WorldSpec};
use crate::yaml;

/// A fully loaded and validated experiment: everything a run needs, with every reference
/// followed.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub experiment_path: PathBuf,
    pub experiment: ExperimentConfig,
    /// The experiment configuration as a generic value, after overrides. Conditions are
    /// applied to this value, and `run.yaml` is written from it.
    pub experiment_value: Value,
    /// The file the world settings came from: the world file, or the experiment file when the
    /// world is inline.
    pub world_path: PathBuf,
    pub world: WorldSpec,
    pub calendar: Calendar,
    /// All rules files merged.
    pub rules: RulesFile,
    /// Every type loaded, by name, with `extends` applied, in load order.
    pub types: IndexMap<String, EntityType>,
    /// One entry per population entry, with overrides applied.
    pub population: Vec<ResolvedPopulation>,
}

/// One population entry with its type fully resolved.
#[derive(Debug, Clone)]
pub struct ResolvedPopulation {
    pub index: usize,
    pub type_name: String,
    pub count: u32,
    pub nervous_system: Option<NervousSystem>,
    /// The entry's type after `extends` and `overrides`, validated.
    pub entity_type: EntityType,
}

/// Load an experiment file and resolve everything it refers to.
pub fn load_experiment(path: &Path) -> Result<Resolved> {
    load_experiment_with_overrides(path, &[])
}

/// Load an experiment file, apply dotted-path overrides (`seed`, `population.0.count`), and
/// resolve everything it refers to.
pub fn load_experiment_with_overrides(
    path: &Path,
    overrides: &[(String, Value)],
) -> Result<Resolved> {
    let mut value = yaml::read_value(path)?;
    for (dotted, new_value) in overrides {
        yaml::set_path(&mut value, dotted, new_value.clone()).map_err(|m| {
            SchemaError::field(path, dotted.clone(), format!("cannot override: {m}"))
        })?;
    }
    resolve_experiment_value(value, path)
}

/// Resolve an experiment configuration already read as a generic value. `path` is the file it
/// came from, which relative paths inside it are resolved against.
pub fn resolve_experiment_value(value: Value, path: &Path) -> Result<Resolved> {
    let experiment: ExperimentConfig = yaml::from_value(value.clone(), path)?;
    let ctx = Ctx::new(path);
    experiment.validate(&ctx)?;
    let experiment_dir = path.parent().unwrap_or_else(|| Path::new("."));

    // The world, from a file or inline.
    let (world_path, world) = match &experiment.world {
        WorldRef::Path(relative) => {
            let world_path = yaml::join_normalized(experiment_dir, relative);
            let world: WorldSpec = yaml::read(&world_path)?;
            (world_path, world)
        }
        WorldRef::Inline(world) => (path.to_path_buf(), (**world).clone()),
    };
    let world_ctx = match &experiment.world {
        WorldRef::Path(_) => Ctx::new(&world_path),
        WorldRef::Inline(_) => ctx.child("world"),
    };
    let calendar = world.validate_and_calendar(&world_ctx)?;
    let world_dir = world_path.parent().unwrap_or_else(|| Path::new("."));

    // The rules.
    let mut rules = RulesFile::default();
    for relative in &world.rules {
        let rules_path = yaml::join_normalized(world_dir, relative);
        let file: RulesFile = yaml::read(&rules_path)?;
        let rctx = Ctx::new(&rules_path);
        file.validate(&rctx)?;
        rules.merge(file, &rctx)?;
    }

    // The types: from the types directory, plus any declared inline in the experiment.
    let mut loader = TypeLoader::new(
        world
            .types
            .as_ref()
            .map(|dir| yaml::join_normalized(world_dir, dir)),
    );
    for (i, inline) in experiment.types.iter().enumerate() {
        let ictx = ctx.child("types").child(i.to_string());
        let name = inline.get("type").and_then(Value::as_str).ok_or_else(|| {
            ictx.child("type")
                .error("an inline type needs a `type` name")
        })?;
        if loader
            .inline
            .insert(name.to_string(), inline.clone())
            .is_some()
        {
            return Err(ictx
                .child("type")
                .error(format!("`{name}` is declared twice")));
        }
    }
    let mut types: IndexMap<String, EntityType> = IndexMap::new();
    let mut population = Vec::new();
    for (i, entry) in experiment.population.iter().enumerate() {
        let ectx = ctx.child("population").child(i.to_string());
        let (raw, source) = loader.load_raw(&entry.type_name, &mut Vec::new(), path)?;
        if !types.contains_key(&entry.type_name) {
            let entity_type: EntityType = yaml::from_value(raw.clone(), &source)?;
            entity_type.validate(&Ctx::new(&source))?;
            types.insert(entry.type_name.clone(), entity_type);
        }
        let mut merged = raw;
        if entry.overrides.is_object() {
            yaml::deep_merge(&mut merged, entry.overrides.clone());
        }
        if let Some(ns) = &entry.nervous_system {
            let ns_value = serde_json::to_value(ns).expect("a nervous system serializes");
            yaml::set_path(&mut merged, "nervous_system", ns_value).expect("root is a mapping");
        }
        let octx = ectx.child("overrides");
        let entity_type: EntityType =
            yaml::from_value(merged, path).map_err(|e| prefix_field(e, &octx.path))?;
        entity_type
            .validate(&octx)
            .map_err(|e| prefix_field(e, &octx.path))?;
        if experiment.views.propositional_sensor
            && entity_type.nervous_system.is_some()
            && entity_type.sensors.propositional.is_none()
        {
            return Err(ctx.child("views").child("propositional_sensor").error(format!(
                "the propositional sensor is on, but the type `{}` declares no `propositional` sensor",
                entry.type_name
            )));
        }
        population.push(ResolvedPopulation {
            index: i,
            type_name: entry.type_name.clone(),
            count: entry.count,
            nervous_system: entry.nervous_system.clone(),
            entity_type,
        });
    }
    // Types placed by the map generator are named in its parameters; load those too, so a
    // world file that names an unknown type fails here rather than at map generation.
    if let Some(objects) = world.map.params.get("objects").and_then(Value::as_object) {
        for name in objects.keys() {
            if !types.contains_key(name) {
                let (raw, source) = loader.load_raw(name, &mut Vec::new(), &world_path)?;
                let entity_type: EntityType = yaml::from_value(raw, &source)?;
                entity_type.validate(&Ctx::new(&source))?;
                types.insert(name.clone(), entity_type);
            }
        }
    }

    // Conditions: every path must exist, and every setting must parse in its place.
    for (dotted, settings) in &experiment.conditions {
        let cctx = ctx.child("conditions").child(dotted);
        cctx.check(
            yaml::get_path(&value, dotted).is_some(),
            "the path does not exist in this configuration",
        )?;
        for (j, setting) in settings.iter().enumerate() {
            let mut trial = value.clone();
            yaml::set_path(&mut trial, dotted, setting.clone())
                .map_err(|m| cctx.child(j.to_string()).error(m))?;
            let parsed: ExperimentConfig = yaml::from_value(trial, path)
                .map_err(|e| prefix_field(e, &cctx.child(j.to_string()).path))?;
            parsed
                .validate(&ctx)
                .map_err(|e| prefix_field(e, &cctx.child(j.to_string()).path))?;
        }
    }

    Ok(Resolved {
        experiment_path: path.to_path_buf(),
        experiment,
        experiment_value: value,
        world_path,
        world,
        calendar,
        rules,
        types,
        population,
    })
}

/// Put a path in front of an error's field, so an error inside a merged type says where the
/// merge happened.
fn prefix_field(error: SchemaError, prefix: &str) -> SchemaError {
    match error {
        SchemaError::Field {
            file,
            field,
            message,
        } => SchemaError::Field {
            file,
            field: format!("{prefix}: {field}"),
            message,
        },
        SchemaError::NotInMilestone1 {
            file,
            field,
            feature,
        } => SchemaError::NotInMilestone1 {
            file,
            field: format!("{prefix}: {field}"),
            feature,
        },
        other => other,
    }
}

/// Loads type files by name, following `extends`, with a cache.
struct TypeLoader {
    dir: Option<PathBuf>,
    /// Types declared inline in the experiment, by name.
    inline: IndexMap<String, Value>,
    /// Resolved raw values by name, with the file each came from.
    cache: IndexMap<String, (Value, PathBuf)>,
}

impl TypeLoader {
    fn new(dir: Option<PathBuf>) -> Self {
        TypeLoader {
            dir,
            inline: IndexMap::new(),
            cache: IndexMap::new(),
        }
    }

    /// The raw value of a type with `extends` applied, and the file to report errors against.
    /// `requester` is the file that asked for the type, for the error when it is not found.
    fn load_raw(
        &mut self,
        name: &str,
        stack: &mut Vec<String>,
        requester: &Path,
    ) -> Result<(Value, PathBuf)> {
        if let Some(cached) = self.cache.get(name) {
            return Ok(cached.clone());
        }
        if stack.iter().any(|n| n == name) {
            stack.push(name.to_string());
            return Err(SchemaError::resolve(
                requester,
                format!("`extends` cycle: {}", stack.join(" -> ")),
            ));
        }
        let (mut raw, source) = match self.inline.get(name) {
            Some(inline) => (inline.clone(), requester.to_path_buf()),
            None => {
                let Some(dir) = &self.dir else {
                    return Err(SchemaError::resolve(
                        requester,
                        format!("unknown type `{name}`: no types directory is configured"),
                    ));
                };
                let file = dir.join(format!("{name}.yaml"));
                if !file.exists() {
                    return Err(SchemaError::resolve(
                        requester,
                        format!("unknown type `{name}`: {} does not exist", file.display()),
                    ));
                }
                let raw = yaml::read_value(&file)?;
                let declared = raw.get("type").and_then(Value::as_str).unwrap_or_default();
                if declared != name {
                    return Err(SchemaError::field(
                        &file,
                        "type",
                        format!("the file is named `{name}` but declares type `{declared}`"),
                    ));
                }
                (raw, file)
            }
        };
        if let Some(base) = raw.get("extends").cloned() {
            let Some(base_name) = base.as_str() else {
                return Err(SchemaError::field(
                    &source,
                    "extends",
                    "must be a type name",
                ));
            };
            stack.push(name.to_string());
            let (mut merged, _) = self.load_raw(base_name, stack, &source)?;
            stack.pop();
            yaml::deep_merge(&mut merged, raw);
            raw = merged;
        }
        self.cache
            .insert(name.to_string(), (raw.clone(), source.clone()));
        Ok((raw, source))
    }
}
