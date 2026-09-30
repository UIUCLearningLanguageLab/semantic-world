//! Command-line tool: write the JSON Schemas, or check that an experiment loads and resolves.
//!
//! ```text
//! sw-schema schemas [DIR]         write JSON Schemas to DIR (default: schemas/)
//! sw-schema check EXPERIMENT.yaml load the experiment and everything it refers to
//! ```

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("schemas") => {
            let dir = args.get(1).map_or("schemas", String::as_str);
            match sw_schema::write_schemas(Path::new(dir)) {
                Ok(files) => {
                    for file in files {
                        println!("wrote {}", file.display());
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("check") if args.len() == 2 => match sw_schema::load_experiment(Path::new(&args[1])) {
            Ok(resolved) => {
                println!(
                    "{}: experiment `{}` resolves",
                    resolved.experiment_path.display(),
                    resolved.experiment.experiment
                );
                println!(
                    "  world: {} ({} m × {} m, {} min/day)",
                    resolved.world_path.display(),
                    resolved.world.map.size_m[0],
                    resolved.world.map.size_m[1],
                    resolved.calendar.minutes_per_day
                );
                println!(
                    "  rules: {} actions, {} processes, {} predicates",
                    resolved.rules.actions.len(),
                    resolved.rules.processes.len(),
                    resolved.rules.predicates.len()
                );
                let names: Vec<&str> = resolved.types.keys().map(String::as_str).collect();
                println!("  types: {}", names.join(", "));
                for p in &resolved.population {
                    let module = p
                        .entity_type
                        .nervous_system
                        .as_ref()
                        .map_or("-", |ns| ns.module.as_str());
                    println!(
                        "  population {}: {} × {} (nervous system: {})",
                        p.index, p.count, p.type_name, module
                    );
                }
                println!("  seeds: {:?}", resolved.experiment.master_seeds());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage:\n  sw-schema schemas [DIR]\n  sw-schema check EXPERIMENT.yaml");
            ExitCode::FAILURE
        }
    }
}
