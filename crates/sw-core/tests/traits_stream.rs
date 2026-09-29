//! Trait variation is sampled from the `traits` stream: the same master seed gives the same
//! individuals, and a different seed gives different ones.

use std::path::Path;

use sw_core::streams;
use sw_schema::load_experiment;

fn human() -> sw_schema::entity::EntityType {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let resolved = load_experiment(&root.join("data/experiments/m1_smoke.yaml")).unwrap();
    resolved.types["human"].clone()
}

#[test]
fn individuals_are_reproducible_from_the_master_seed() {
    let human = human();
    let sample = |seed: u64| {
        let mut rng = streams::stream(seed, streams::TRAITS);
        let (first, first_values) = human.sample_individual(&mut rng);
        let (second, second_values) = human.sample_individual(&mut rng);
        (first, first_values, second, second_values)
    };
    let (a1, v1, a2, v2) = sample(3);
    let (b1, w1, b2, w2) = sample(3);
    assert_eq!(a1, b1);
    assert_eq!(a2, b2);
    assert_eq!(v1, w1);
    assert_eq!(v2, w2);
    assert_ne!(a1, a2, "two individuals from one stream differ");
    let (c1, ..) = sample(4);
    assert_ne!(
        a1, c1,
        "a different master seed gives a different individual"
    );
}

#[test]
fn sampled_values_are_within_range_and_logged_by_path() {
    let human = human();
    let mut rng = streams::stream(11, streams::TRAITS);
    for _ in 0..50 {
        let (individual, sampled) = human.sample_individual(&mut rng);
        let paths: Vec<&str> = sampled.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["body.mass", "body.height"]);
        let mass = individual.body.mass.as_ref().unwrap();
        assert!(!mass.varies(), "the individual's traits are frozen");
        assert!((40.0..=110.0).contains(mass.value()));
        let height = individual.body.height.as_ref().unwrap();
        assert!((1.4..=2.0).contains(height.value()));
        // Traits without variation keep their defaults.
        assert_eq!(*individual.body.insulation.as_ref().unwrap().value(), 0.2);
    }
}
