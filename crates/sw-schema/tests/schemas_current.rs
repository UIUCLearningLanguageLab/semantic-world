//! The committed JSON Schemas in `schemas/` match the types. Regenerate them with
//! `cargo run -p sw-schema -- schemas`.

mod common;

#[test]
fn committed_schemas_are_current() {
    let dir = common::repo_root().join("schemas");
    for (name, schema) in sw_schema::schemas() {
        let file = dir.join(name);
        let committed = std::fs::read_to_string(&file).unwrap_or_else(|e| {
            panic!(
                "{}: {e}; run `cargo run -p sw-schema -- schemas`",
                file.display()
            )
        });
        assert_eq!(
            committed,
            sw_schema::schema_text(&schema),
            "{} is stale; run `cargo run -p sw-schema -- schemas`",
            file.display()
        );
    }
}

#[test]
fn schemas_are_draft_2020_12_with_titles() {
    for (name, schema) in sw_schema::schemas() {
        let value = schema.as_value();
        assert_eq!(
            value["$schema"], "https://json-schema.org/draft/2020-12/schema",
            "{name}"
        );
        assert!(value.get("title").is_some(), "{name} has no title");
    }
}
