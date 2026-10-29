//! Family registry tests. The registry is a lookup — no starter runs here.

use super::config::EngineConfig;
use super::run::{StartedEngine, family_starter};
use crate::sched::worker::ReadyNote;

/// The registry is a lookup TABLE: two rows with the same name would
/// silently shadow one family with another's starter. The gate-linkage
/// test below covers membership; this pins uniqueness.
#[test]
fn registry_names_are_unique() {
    let names = super::run::family_names();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        names.len(),
        "duplicate registry row would shadow a family's starter"
    );
}

/// An unknown family resolves to nothing here — the config gate owns the
/// rejection text; the registry simply has no entry for it.
#[test]
fn registry_has_no_entry_for_unknown_families() {
    assert!(family_starter("gemma4-someday").is_none());
    assert!(family_starter("").is_none());
}

/// The starter signature is stable enough to name the type: one config, the
/// process-wide prefix_reuse flag and the ready note in, one StartedEngine
/// out.
#[test]
fn starter_signature_takes_config_prefix_reuse_and_ready_note() {
    let f: fn(&EngineConfig, bool, ReadyNote) -> Result<StartedEngine, crate::EngError> =
        family_starter("fake").expect("fake starter");
    let _ = f; // not called: signature pin only
}

/// No other test pins the nemotron35 ROW to the nemotron factory — a
/// wrong row would pass every other test. This drives the starter with a
/// gate-valid config whose `nemotron` options are absent and asserts the
/// factory's OWN refusal (factory.rs:89), so the row provably reaches
/// the nemotron start path and nothing else.
#[test]
fn nemotron35_row_pins_to_the_nemotron_factory_refusal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("enginesrv.json");
    std::fs::write(
        &path,
        r#"{
            "listen": "127.0.0.1:8100",
            "api_key_env": "ENGINESRV_KEY",
            "engines": [
                {"id": "e1", "model_dir": "m", "tokenizer_dir": "t",
                 "family": "nemotron35"}
            ]
        }"#,
    )
    .expect("write config");
    let config = crate::server::config::Config::from_file(&path)
        .expect("gate accepts nemotron35 without options");
    let entry = &config.engines[0];
    let starter = family_starter(&entry.family).expect("nemotron35 starter");
    match starter(entry, false, None) {
        Err(crate::EngError::BadRequest(message)) => assert_eq!(
            message, "nemotron35 config: this entry needs options",
            "the refusal must be the nemotron factory's own"
        ),
        other => panic!("expected the factory refusal, got Ok: {}", other.is_ok()),
    }
}

/// The config gate is the list of families a user can name. Every family
/// the gate accepts must resolve through the registry table — a gate entry
/// with no starter would fall through at startup. Covers the Qwen3.8
/// members (flash-next and dense) whatever their literal names.
#[test]
fn every_family_the_config_gate_accepts_resolves_through_the_registry() {
    for family in crate::server::config::FAMILIES {
        assert!(
            family_starter(family).is_some(),
            "config gate accepts {family} but the registry has no starter"
        );
    }
}

/// The unknown-family rejection the user sees is the CONFIG GATE's text —
/// pinned byte-identical here so the registry refactor (or any later
/// change) cannot silently alter the startup surface. The registry's own
/// "has no starter in this build" only fires if the gate misses, which
/// the previous test makes impossible for gate-listed families.
#[test]
fn unknown_family_error_text_is_the_config_gate_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("enginesrv.json");
    std::fs::write(
        &path,
        r#"{
            "listen": "127.0.0.1:8100",
            "api_key_env": "ENGINESRV_KEY",
            "engines": [
                {"id": "e1", "model_dir": "m", "tokenizer_dir": "t",
                 "family": "gemma4-someday"}
            ]
        }"#,
    )
    .expect("write config");
    let error = crate::server::config::Config::from_file(&path).expect_err("unknown family");
    let expected = format!(
        "engine e1 has family gemma4-someday, which this build does not support (supported: {})",
        crate::server::config::FAMILIES.join(", ")
    );
    assert_eq!(
        error.to_string(),
        expected,
        "the gate text must stay byte-identical"
    );
}
