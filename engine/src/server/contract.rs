//! Test helper: checks server JSON against `api/openapi.yaml`.
//!
//! The crate has no YAML parser, so the helper reads `testdata/openapi.json`.
//! `scripts/openapi_to_json.py` writes that file from the contract and records the hash of the YAML bytes.

use jsonschema::Validator;
use serde_json::{Value, json};
use std::sync::OnceLock;

const FIXTURE: &str = include_str!("testdata/openapi.json");

fn fixture() -> &'static Value {
    static PARSED: OnceLock<Value> = OnceLock::new();
    PARSED.get_or_init(|| serde_json::from_str(FIXTURE).expect("parse openapi.json"))
}

/// Compiles the schema at `pointer` in the contract document, for example `/components/schemas/Error`.
fn validator(pointer: &str) -> Validator {
    let document = fixture();
    let schema = json!({
        "$ref": format!("#{pointer}"),
        "components": document["components"],
        "paths": document["paths"],
    });
    jsonschema::validator_for(&schema).expect("compile contract schema")
}

fn errors(pointer: &str, instance: &Value) -> Vec<String> {
    validator(pointer)
        .iter_errors(instance)
        .map(|error| error.to_string())
        .collect()
}

/// Panics with every violation when `instance` does not match the component schema `name`.
pub(super) fn assert_schema(name: &str, instance: &Value) {
    let found = errors(&format!("/components/schemas/{name}"), instance);
    assert!(
        found.is_empty(),
        "{name} violations: {found:?} in {instance}"
    );
}

/// Panics with every violation when `instance` does not match the schema at `pointer`.
pub(super) fn assert_pointer(pointer: &str, instance: &Value) {
    let found = errors(pointer, instance);
    assert!(
        found.is_empty(),
        "{pointer} violations: {found:?} in {instance}"
    );
}

/// True when `instance` matches the component schema `name`.
pub(super) fn matches_schema(name: &str, instance: &Value) -> bool {
    errors(&format!("/components/schemas/{name}"), instance).is_empty()
}

#[cfg(test)]
mod tests {
    use super::{FIXTURE, assert_pointer, assert_schema, matches_schema};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    #[test]
    fn the_fixture_hash_matches_the_yaml_contract() {
        let yaml = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../api/openapi.yaml"))
            .expect("read api/openapi.yaml");
        let expected = format!("{:x}", Sha256::digest(&yaml));
        let fixture: Value = serde_json::from_str(FIXTURE).expect("parse openapi.json");
        assert_eq!(
            fixture["x-source-sha256"], expected,
            "stale fixture: run scripts/openapi_to_json.py"
        );
    }

    #[test]
    fn a_body_that_follows_the_contract_passes() {
        let body = json!({"error": {"message": "bad", "code": "bad_request"}});
        assert_schema("Error", &body);
        assert!(matches_schema("Error", &body));
    }

    #[test]
    fn a_body_that_breaks_the_contract_fails() {
        let missing_code = json!({"error": {"message": "bad"}});
        assert!(!matches_schema("Error", &missing_code));
        let unknown_code = json!({"error": {"message": "bad", "code": "teapot"}});
        assert!(!matches_schema("Error", &unknown_code));
    }

    #[test]
    fn a_path_pointer_selects_a_response_schema() {
        let pointer = "/paths/~1health/get/responses/200/content/application~1json/schema";
        assert_pointer(pointer, &json!({"status": "ok"}));
        assert!(
            super::errors(pointer, &json!({"status": "down"})).len() == 1,
            "an unknown status must fail"
        );
    }
}
