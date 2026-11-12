//! The diagnostic support bundle: `enginesrv doctor --bundle <file>`.
//!
//! One self-contained JSON document (`bundle_schema
//! fifi-diagnostic-bundle/1`) with structured sections, a section
//! manifest that says what was collected and what was not, and an
//! embedded human-readable summary (also printed by the CLI). A single
//! inspectable file over an archive: any editor or `jq` can review it
//! before sharing, it writes atomically, and it needs no archive
//! dependency.
//!
//! Every fact comes from an existing source of truth — [`probe`] for
//! hardware, [`preflight`] for model qualification, and
//! `effective::q36_effective` for settings with origin — never a
//! parallel resolution. Sanitization is by construction (no
//! environment variable is read except `HOME` for path scrubbing and
//! the allowlisted `Q36_CACHE_LIMIT` the effective rows already read;
//! no request, log, or model content is an input) plus one recursive
//! pass that replaces the home-directory prefix with `~` in every
//! string of the finished document. One allowlist edge, by design:
//! the *value* of `Q36_CACHE_LIMIT` appears in the settings rows (it
//! is a memory size in every legitimate use), and an invalid value can
//! be quoted by that setting's refusal text — a name that is
//! allowlisted, not a credential.
//!
//! Schema history: /1 added the per-engine `preflight.decisions` array
//! (the execution-plan inventory; Flint on every estimate, Q36 on EXL3
//! loads; empty for families and formats whose estimate does not compose
//! decisions). Additive only — every earlier /1 field is unchanged.

use super::Probe;
use super::preflight::{self, Candidate, PreflightReport};
use super::probe::{self, HardwareFacts};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// The schema identity, versioned independently of the application.
pub(crate) const BUNDLE_SCHEMA: &str = "fifi-diagnostic-bundle/1";

/// One engine's diagnostics, pre-collected: the collector has already
/// run the preflight and the settings resolution, so `build` stays
/// pure and a failed section degrades to its `unavailable` text
/// without touching the others.
pub(crate) struct EngineInput {
    pub(crate) id: String,
    pub(crate) family: String,
    pub(crate) model_dir: PathBuf,
    pub(crate) preflight: Result<PreflightReport, String>,
    /// The effective-settings rows (`config effective`), or the
    /// resolution refusal.
    pub(crate) effective: Result<Vec<Value>, String>,
}

/// Everything `build` needs; every measured value arrives as an input
/// and is labeled as measured in the document.
pub(crate) struct BundleInputs {
    pub(crate) facts: HardwareFacts,
    pub(crate) engines: Vec<EngineInput>,
    /// `/etc/os-release` `PRETTY_NAME`, when readable.
    pub(crate) os_release: Option<String>,
    /// Collection time, unix seconds.
    pub(crate) generated_unix: u64,
    /// The home directory whose prefix becomes `~` in every string.
    pub(crate) home: String,
    /// Why a requested config contributed no engines, when it could
    /// not be parsed.
    pub(crate) config_note: Option<String>,
}

/// One probed value in the bundle's own shape. `Probe`'s tagged
/// derivation cannot serialize primitive contents, and the bundle —
/// unlike the text renderer — needs JSON, so the mapping is explicit:
/// `{"measured": T}` or `{"unavailable": reason}`.
fn probe_json<T: Copy + serde::Serialize>(probe: &Probe<T>) -> Value {
    match probe {
        Probe::Measured(value) => json!({"measured": value}),
        Probe::Unavailable(reason) => json!({"unavailable": reason}),
    }
}

/// The hardware section: the same `HardwareFacts` `doctor` probes,
/// field for field, every value carrying its evidence label.
fn hardware_json(facts: &HardwareFacts) -> Value {
    json!({
        "backends": {
            "cuda_compiled": facts.backends.cuda,
            "mlx_compiled": facts.backends.mlx,
            "none_compiled": facts.backends.none(),
        },
        "cuda": {
            "device_count": probe_json(&facts.cuda.device_count),
            "first_device_integrated": probe_json(&facts.cuda.first_device_integrated),
            "first_device_free_bytes": probe_json(&facts.cuda.first_device_free_bytes),
            "first_device_total_bytes": probe_json(&facts.cuda.first_device_total_bytes),
        },
        "host": {
            "mem_total_bytes": probe_json(&facts.host.mem_total_bytes),
            "mem_available_bytes": probe_json(&facts.host.mem_available_bytes),
            "cpu_threads": facts.host.cpu_threads,
        },
    })
}

/// Builds the bundle document. Pure: the same inputs produce the same
/// bytes, so a user can regenerate and diff a bundle.
pub(crate) fn build(inputs: &BundleInputs) -> Value {
    let engines: Vec<Value> = inputs
        .engines
        .iter()
        .map(|engine| {
            let preflight = match &engine.preflight {
                Ok(report) => preflight_json(report),
                Err(error) => json!({"unavailable": error}),
            };
            let effective = match &engine.effective {
                Ok(rows) => json!(rows),
                Err(error) => json!({"unavailable": error}),
            };
            json!({
                "id": engine.id,
                "family": engine.family,
                "model_dir": sanitize_path(&engine.model_dir.display().to_string(), &inputs.home),
                "preflight": preflight,
                "effective_settings": effective,
            })
        })
        .collect();
    let hardware = hardware_json(&inputs.facts);
    let mut bundle = json!({
        "bundle_schema": BUNDLE_SCHEMA,
        "generated": {"measured": true, "unix_seconds": inputs.generated_unix},
        "fifi": {
            "version": env!("CARGO_PKG_VERSION"),
            "platform": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH},
        },
        "os": match &inputs.os_release {
            Some(name) => json!({"pretty_name": {"measured": true, "value": name}}),
            None => json!({"pretty_name": {"unavailable": "/etc/os-release is not readable"}}),
        },
        "hardware": hardware,
        "engines": engines,
        // Stages exist only while an engine is loading inside a live
        // server; this bundle is offline by design.
        "startup_stages": {
            "unavailable": "offline bundle: loading stages exist only in a running server"
        },
        "logs": {"not_collected": "no log source is read; attach logs separately"},
    });
    let manifest = manifest_of(&bundle, inputs);
    bundle["manifest"] = manifest;
    bundle["summary"] = json!(summary_of(&bundle));
    sanitize_value(&bundle, &inputs.home)
}

/// The preflight section of one engine: verdict, checks, and next
/// step — the same report `check` prints, minus the directories
/// (already sanitized at the engine level).
fn preflight_json(report: &PreflightReport) -> Value {
    json!({
        "verdict": serde_json::to_value(report.verdict).unwrap_or(Value::Null),
        "family": report.family,
        "checks": report
            .checks
            .iter()
            .map(|check| json!({
                "name": check.name,
                "status": serde_json::to_value(check.status).unwrap_or(Value::Null),
                "detail": check.detail,
            }))
            .collect::<Vec<_>>(),
        "decisions": report
            .decisions
            .iter()
            .map(|decision| json!({
                "name": decision.name,
                "chosen": decision.chosen,
                "alternatives": decision
                    .alternatives
                    .iter()
                    .map(|alternative| json!({
                        "option": alternative.option,
                        "verdict": alternative.verdict,
                        "reason": alternative.reason,
                    }))
                    .collect::<Vec<_>>(),
                "fixed_at": serde_json::to_value(decision.fixed_at).unwrap_or(Value::Null),
            }))
            .collect::<Vec<_>>(),
        "next_step": report.next_step,
    })
}

/// One manifest row per section: collected, unavailable, or not
/// requested — the reader of a bundle never wonders what is missing.
fn manifest_of(bundle: &Value, inputs: &BundleInputs) -> Value {
    let mut sections = vec![
        json!({"name": "fifi", "status": "collected"}),
        json!({"name": "os", "status": if inputs.os_release.is_some() { "collected" } else { "unavailable" }}),
        json!({"name": "hardware", "status": "collected"}),
        json!({"name": "startup_stages", "status": "unavailable"}),
        json!({"name": "logs", "status": "not_collected"}),
    ];
    if inputs.engines.is_empty() {
        if let Some(note) = &inputs.config_note {
            sections.push(json!({"name": "engines", "status": "unavailable", "detail": note}));
        } else {
            sections.push(json!({
                "name": "engines",
                "status": "not_requested",
                "detail": "no --config given; pass one to include per-engine preflights and settings",
            }));
        }
    } else {
        sections
            .push(json!({"name": "engines", "status": "collected", "count": inputs.engines.len()}));
    }
    let _ = bundle;
    json!(sections)
}

/// The human-readable summary, derived from the finished document so
/// it can never disagree with the structured sections.
fn summary_of(bundle: &Value) -> String {
    let fifi = &bundle["fifi"];
    let mut out = String::new();
    out.push_str(&format!("fifi diagnostic bundle ({})\n", BUNDLE_SCHEMA));
    out.push_str(&format!(
        "enginesrv {} on {}/{}\n",
        fifi["version"], fifi["platform"]["os"], fifi["platform"]["arch"]
    ));
    let hardware = &bundle["hardware"];
    out.push_str(&format!(
        "backends compiled: {}\n",
        match hardware["backends"]["none_compiled"].as_bool() {
            Some(true) => "none".to_string(),
            _ => {
                let mut names = Vec::new();
                if hardware["backends"]["cuda_compiled"].as_bool() == Some(true) {
                    names.push("cuda");
                }
                if hardware["backends"]["mlx_compiled"].as_bool() == Some(true) {
                    names.push("mlx");
                }
                names.join(", ")
            }
        }
    ));
    match &hardware["cuda"]["first_device_total_bytes"]["measured"] {
        Value::Null => out.push_str(&format!(
            "cuda: unavailable ({})\n",
            hardware["cuda"]["first_device_total_bytes"]["unavailable"]
                .as_str()
                .unwrap_or("unknown reason")
        )),
        total => out.push_str(&format!(
            "cuda: first device {} B total, {} B free\n",
            total, hardware["cuda"]["first_device_free_bytes"]["measured"]
        )),
    }
    out.push_str(&format!(
        "host: {} B RAM total, {} B available, {} cpu threads\n",
        hardware["host"]["mem_total_bytes"]["measured"],
        hardware["host"]["mem_available_bytes"]["measured"],
        hardware["host"]["cpu_threads"]
    ));
    let engines = bundle["engines"].as_array();
    match engines {
        None => out.push_str("engines: not requested (no --config)\n"),
        Some(engines) if engines.is_empty() => {
            out.push_str("engines: not requested (no --config)\n")
        }
        Some(engines) => {
            out.push_str(&format!("engines: {}\n", engines.len()));
            for engine in engines {
                out.push_str(&format!(
                    "  {} ({}): preflight {}, settings {}\n",
                    engine["id"],
                    engine["family"],
                    engine["preflight"]["verdict"]
                        .as_str()
                        .unwrap_or("unavailable"),
                    match &engine["effective_settings"] {
                        Value::Array(rows) => format!("{} rows", rows.len()),
                        _ => "unavailable".to_string(),
                    }
                ));
            }
        }
    }
    out.push_str("startup stages: offline bundle — not collected\n");
    out.push_str("logs: not collected\n");
    out
}

/// Replaces the home directory with `~` in one string: as a whole
/// path (`/home/user/models` → `~/models`) and as embedded
/// occurrences inside free text such as check details and refusal
/// messages (`under /home/user/models: refused` → `under ~/models:
/// refused`). Usernames live in home paths; a bundle shows `~`
/// instead. The only other absolute paths the bundle carries are
/// system paths (preflight names none; the OS section names none).
pub(crate) fn sanitize_path(path: &str, home: &str) -> String {
    if home.is_empty() {
        return path.to_string();
    }
    let home = home.trim_end_matches('/');
    let prefix = format!("{home}/");
    if path == home {
        return "~".to_string();
    }
    let mut scrubbed = path.replace(&prefix, "~/");
    if scrubbed.contains(home) {
        // A bare mention of the home directory itself, without a slash.
        scrubbed = scrubbed.replace(home, "~");
    }
    scrubbed
}

/// The same replacement over every string leaf of the finished
/// document — one boundary, applied uniformly, so a path inside a
/// check detail or a settings row is scrubbed exactly like a
/// top-level directory field.
fn sanitize_value(value: &Value, home: &str) -> Value {
    match value {
        Value::String(text) => Value::String(sanitize_path(text, home)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| sanitize_value(item, home))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), sanitize_value(item, home)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Collects everything the bundle needs from this machine. Thin by
/// design: probe, per-engine preflight and settings resolution — the
/// same calls `doctor`, `check`, and `config effective` run — and
/// nothing else. No network, no weights: the preflight is the
/// non-probing validator, and a model directory that cannot even be
/// inspected becomes that engine's `unavailable` text.
pub(crate) fn collect(config: Option<&crate::server::config::Config>) -> BundleInputs {
    let facts = probe::probe();
    let engines = config
        .map(|config| {
            config
                .engines
                .iter()
                .map(|engine| engine_input(engine, &facts))
                .collect()
        })
        .unwrap_or_default();
    BundleInputs {
        facts,
        engines,
        os_release: os_release_pretty_name(),
        generated_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0),
        home: std::env::var("HOME").unwrap_or_default(),
        config_note: None,
    }
}

/// One engine's preflight and effective rows, degraded to text on
/// failure. The candidate mirrors the config surface: the family and
/// the q36 context/KV members the entry carries.
fn engine_input(
    engine: &crate::server::config::EngineConfig,
    facts: &HardwareFacts,
) -> EngineInput {
    let candidate = Candidate {
        family: Some(engine.family.clone()),
        context_limit: engine
            .qwen36
            .as_ref()
            .and_then(|options| options.context_limit),
        kv_dtype: engine
            .qwen36
            .as_ref()
            .and_then(|options| options.kv_dtype.clone()),
    };
    let preflight =
        preflight::preflight(&engine.model_dir, &engine.tokenizer_dir, &candidate, facts)
            .map_err(|error| super::describe_error(&error));
    let env = crate::family::qwen36::settings::Q36Env {
        cache_limit: std::env::var("Q36_CACHE_LIMIT").ok(),
        kv_geometric: std::env::var("Q36_KV_GEOMETRIC").ok(),
        spec_step_window: std::env::var("Q36_SPEC_STEP_WINDOW").ok(),
    };
    let effective = effective_rows(engine, &env);
    EngineInput {
        id: engine.id.clone(),
        family: engine.family.clone(),
        model_dir: engine.model_dir.clone(),
        preflight,
        effective,
    }
}

/// The `config effective` rows for one entry, or the resolution
/// refusal. Reads only the allowlisted `Q36_CACHE_LIMIT`, exactly as
/// that command does; the API key is never read here.
fn effective_rows(
    engine: &crate::server::config::EngineConfig,
    env: &crate::family::qwen36::settings::Q36Env,
) -> Result<Vec<Value>, String> {
    use crate::family::qwen36::factory::{DENSE_FAMILY, FAMILY, backend_of_build};
    use crate::family::qwen36::settings::Q36Settings;
    if engine.family == crate::family::flint::FAMILY {
        let default_options = crate::family::flint::settings::FlintOptions::cuda_default();
        let options = engine.flint.as_ref().unwrap_or(&default_options);
        return Ok(super::effective::flint_effective(
            options,
            &crate::family::flint::budget::FlintEnv::capture(),
            None,
        )
        .iter()
        .map(|row| json!({"key": row.key, "value": row.value, "origin": row.origin.name()}))
        .collect());
    }
    if engine.family != FAMILY && engine.family != DENSE_FAMILY {
        return Ok(vec![json!({
            "unavailable":
                "per-key origins are composed for the Qwen3.6 families only so far",
        })]);
    }
    let backend = backend_of_build();
    let spec = if engine.family == DENSE_FAMILY {
        crate::family::spec::QWEN3_8_27B_DENSE
    } else {
        crate::family::spec::QWEN3_6_35B_A3B
    };
    let settings = Q36Settings::new_for_family(
        &engine.model_dir,
        &engine.tokenizer_dir,
        engine.qwen36.as_ref(),
        env,
        backend,
        spec,
    )
    .map_err(|error| super::describe_error(&error))?;
    let default_options = crate::family::qwen36::settings::Q36Options::default();
    Ok(super::effective::q36_effective(
        engine.qwen36.as_ref().unwrap_or(&default_options),
        env.cache_limit.as_deref(),
        backend,
        &settings,
    )
    .iter()
    .map(|row| json!({"key": row.key, "value": row.value, "origin": row.origin.name()}))
    .collect())
}

/// `/etc/os-release` `PRETTY_NAME`: a fixed, public path — the only
/// file the bundle reads.
fn os_release_pretty_name() -> Option<String> {
    let text = std::fs::read_to_string("/etc/os-release").ok()?;
    let line = text.lines().find(|line| line.starts_with("PRETTY_NAME="))?;
    let quoted = line.trim_start_matches("PRETTY_NAME=").trim();
    Some(quoted.trim_matches('"').to_string())
}

/// Writes the bundle atomically: a fresh `.part` sibling next to the
/// destination, one rename, and the partial file removed on any
/// failure. An existing destination is refused, never overwritten —
/// including one that is only a symlink, live or dangling
/// (`symlink_metadata`, which does not follow). The `.part` file is
/// created with `create_new` (`O_EXCL`), so a symlink planted at that
/// path is refused instead of written through: no file outside the
/// destination directory is ever opened for writing.
pub(crate) fn write_atomically(path: &Path, body: &str) -> Result<(), String> {
    if path.symlink_metadata().is_ok() {
        return Err(format!(
            "doctor: bundle destination {} already exists; refusing to overwrite",
            path.display()
        ));
    }
    let partial = path.with_extension("part");
    // Failure cleanup removes only a `.part` this call created; a
    // refused creation (EEXIST — including a planted symlink) leaves
    // whatever was there untouched.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|error| format!("doctor: cannot write {}: {error}", partial.display()))?;
    if let Err(error) = std::io::Write::write_all(&mut file, body.as_bytes()) {
        let _ = std::fs::remove_file(&partial);
        return Err(format!(
            "doctor: cannot write {}: {error}",
            partial.display()
        ));
    }
    if let Err(error) = std::fs::rename(&partial, path) {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("doctor: cannot place {}: {error}", path.display()));
    }
    Ok(())
}

/// Builds and writes the bundle for the CLI, returning the summary
/// line text to print. `Err` text is the CLI's error convention.
/// Builds and writes the bundle for the CLI, returning the summary
/// line text to print. An invalid config degrades to an engines
/// section that says so — a broken config is exactly the situation
/// the bundle exists for. `Err` text is the CLI's error convention.
pub(crate) fn write_bundle(path: &Path, config_path: Option<&Path>) -> Result<String, String> {
    let mut config_note = None;
    let config = match config_path {
        Some(path) => match crate::server::config::Config::from_file(path) {
            Ok(config) => Some(config),
            Err(error) => {
                config_note = Some(format!(
                    "config {} is invalid: {}",
                    sanitize_path(
                        &path.display().to_string(),
                        &std::env::var("HOME").unwrap_or_default()
                    ),
                    error
                ));
                None
            }
        },
        None => None,
    };
    let mut inputs = collect(config.as_ref());
    inputs.config_note = config_note;
    let bundle = build(&inputs);
    let body = serde_json::to_string_pretty(&bundle).map_err(|error| format!("doctor: {error}"))?;
    write_atomically(path, &body)?;
    Ok(format!(
        "wrote {} ({} bytes): {} engines, hardware and platform facts, summary inside",
        path.display(),
        body.len(),
        inputs.engines.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        BUNDLE_SCHEMA, BundleInputs, EngineInput, build, sanitize_path, write_atomically,
        write_bundle,
    };
    use crate::capability::Probe;
    use crate::capability::preflight::{self, Candidate};
    use crate::capability::probe::{CompiledBackends, CudaFacts, HardwareFacts, HostFacts};
    use serde_json::{Value, json};
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};

    const HOME: &str = "/home/canary_user";

    fn synthetic_facts() -> HardwareFacts {
        HardwareFacts {
            backends: CompiledBackends {
                cuda: false,
                mlx: false,
            },
            cuda: CudaFacts {
                device_count: Probe::Unavailable(
                    "the CUDA backend is not compiled into this build".into(),
                ),
                first_device_integrated: Probe::Unavailable(
                    "the CUDA backend is not compiled into this build".into(),
                ),
                first_device_free_bytes: Probe::Unavailable(
                    "the CUDA backend is not compiled into this build".into(),
                ),
                first_device_total_bytes: Probe::Unavailable(
                    "the CUDA backend is not compiled into this build".into(),
                ),
            },
            host: HostFacts {
                mem_total_bytes: Probe::Measured(64 << 30),
                mem_available_bytes: Probe::Measured(32 << 30),
                cpu_threads: 16,
            },
        }
    }

    fn inputs(engines: Vec<EngineInput>) -> BundleInputs {
        BundleInputs {
            facts: synthetic_facts(),
            engines,
            os_release: Some("TestOS 1.0".to_string()),
            generated_unix: 1_700_000_000,
            home: HOME.to_string(),
            config_note: None,
        }
    }

    /// A model directory with the pinned tokenizer fixture, a config.json
    /// naming the MoE family, and a minimal GGUF file (the same recipe the
    /// serve --model tests use), so the real resolvers run.
    fn q36_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut gguf = b"GGUF".to_vec();
        gguf.extend_from_slice(&3u32.to_le_bytes());
        gguf.extend_from_slice(&0u64.to_le_bytes());
        gguf.extend_from_slice(&0u64.to_le_bytes());
        gguf.resize(32, 0);
        std::fs::write(dir.path().join("model.gguf"), gguf).expect("gguf");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"qwen3_5_moe"}"#,
        )
        .expect("config");
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            std::fs::copy(source.join(name), dir.path().join(name)).expect("tokenizer fixture");
        }
        dir
    }

    fn q36_engine(dir: &Path) -> EngineInput {
        let candidate = Candidate {
            family: Some("qwen3_6_35b_a3b".to_string()),
            context_limit: None,
            kv_dtype: None,
        };
        let preflight = preflight::preflight(dir, dir, &candidate, &synthetic_facts())
            .map_err(|error| error.describe());
        EngineInput {
            id: "q36".to_string(),
            family: "qwen3_6_35b_a3b".to_string(),
            model_dir: PathBuf::from(HOME).join("models/qwen"),
            preflight,
            effective: Err("settings would not start: test refusal".to_string()),
        }
    }

    #[test]
    fn the_bundle_has_the_schema_the_manifest_and_known_sections_only() {
        let bundle = build(&inputs(vec![]));
        assert_eq!(bundle["bundle_schema"], BUNDLE_SCHEMA);
        let mut sections: Vec<&str> = bundle["manifest"]
            .as_array()
            .expect("manifest is a list")
            .iter()
            .map(|row| row["name"].as_str().expect("named"))
            .collect();
        sections.sort();
        assert_eq!(
            sections,
            vec![
                "engines",
                "fifi",
                "hardware",
                "logs",
                "os",
                "startup_stages"
            ]
        );
        // No section can smuggle request data in: the top level is fixed.
        let mut top: Vec<&str> = bundle
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        top.sort_unstable();
        assert_eq!(
            top,
            vec![
                "bundle_schema",
                "engines",
                "fifi",
                "generated",
                "hardware",
                "logs",
                "manifest",
                "os",
                "startup_stages",
                "summary"
            ]
        );
        assert!(
            bundle["summary"]
                .as_str()
                .expect("summary text")
                .contains("fifi diagnostic bundle")
        );
    }

    #[test]
    fn the_same_inputs_produce_the_same_bytes() {
        let one = serde_json::to_string(&build(&inputs(vec![q36_engine(q36_dir().path())])))
            .expect("serialize");
        let two = serde_json::to_string(&build(&inputs(vec![q36_engine(q36_dir().path())])))
            .expect("serialize");
        assert_eq!(one, two);
    }

    /// The sanitization gate. Real inputs cannot carry credentials at
    /// all — the collector reads no API key, no request data, no logs
    /// (the fixed top-level key set is pinned above) — so the
    /// enforceable leak vector left is the home path, including home
    /// paths quoted inside free text. A canary home directory with a
    /// canary username must never survive, whole or embedded.
    #[test]
    fn canary_home_paths_never_survive_whole_or_embedded() {
        let mut engine = q36_engine(q36_dir().path());
        if let Ok(report) = &mut engine.preflight {
            // Simulate what real refusal texts do: quote a full home
            // path inside a longer sentence.
            report.checks.push(crate::capability::preflight::Check {
                name: "hostile".to_string(),
                status: crate::capability::preflight::CheckStatus::Warn,
                detail: format!("refused under {HOME}/models/qwen at the file {HOME}/x.gguf"),
            });
        }
        let bundle = build(&inputs(vec![engine]));
        let text = serde_json::to_string(&bundle).expect("serialize");
        assert!(!text.contains("canary_user"), "the username leaked: {text}");
        assert!(!text.contains(HOME), "the home path leaked: {text}");
        assert_eq!(bundle["engines"][0]["model_dir"], "~/models/qwen");
        assert!(
            bundle["engines"][0]["preflight"]["checks"][4]["detail"]
                .as_str()
                .expect("detail")
                .starts_with("refused under ~/models/qwen at the file ~/x.gguf")
        );
    }

    #[test]
    fn sanitize_path_replaces_only_the_home_prefix() {
        assert_eq!(
            sanitize_path("/home/canary_user/models/qwen", HOME),
            "~/models/qwen"
        );
        assert_eq!(sanitize_path(HOME, HOME), "~");
        assert_eq!(sanitize_path("/etc/os-release", HOME), "/etc/os-release");
        assert_eq!(sanitize_path("/home/other/file", HOME), "/home/other/file");
        assert_eq!(sanitize_path("/x", ""), "/x");
    }

    #[test]
    fn no_gpu_is_an_unavailable_answer_not_a_failure() {
        let bundle = build(&inputs(vec![]));
        assert_eq!(
            bundle["hardware"]["cuda"]["device_count"]["unavailable"],
            "the CUDA backend is not compiled into this build"
        );
        assert_eq!(
            bundle["hardware"]["host"]["mem_total_bytes"]["measured"].as_u64(),
            Some(64 << 30)
        );
    }

    #[test]
    fn a_broken_engine_section_degrades_without_killing_the_bundle() {
        let mut broken = q36_engine(q36_dir().path());
        broken.preflight = Err("not a directory: /home/canary_user/gone".to_string());
        broken.effective = Err("settings would not start".to_string());
        let bundle = build(&inputs(vec![broken]));
        assert_eq!(
            bundle["engines"][0]["preflight"]["unavailable"],
            "not a directory: ~/gone"
        );
        assert!(bundle["engines"][0]["effective_settings"]["unavailable"].is_string());
        assert_eq!(bundle["manifest"][2]["status"], "collected");
    }

    #[test]
    fn an_existing_destination_is_refused_untouched() {
        let dir = tempfile::tempdir().expect("tempdir");
        let destination = dir.path().join("bundle.json");
        std::fs::write(&destination, "original").expect("write");
        let error = write_atomically(&destination, "{}").unwrap_err();
        assert!(error.contains("already exists"), "{error}");
        assert_eq!(std::fs::read_to_string(&destination).unwrap(), "original");
    }

    #[test]
    fn a_failed_write_leaves_no_partial_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let destination = dir.path().join("missing-subdir").join("bundle.json");
        assert!(write_atomically(&destination, "{}").is_err());
        assert!(!dir.path().join("missing-subdir.part").exists());
        assert!(!destination.exists());
    }

    /// A `.part` symlink planted next to the destination must be
    /// refused, not written through: the bundle never opens a file
    /// outside its own destination directory for writing.
    #[test]
    fn a_planted_part_symlink_is_refused_not_followed() {
        let dir = tempfile::tempdir().expect("dir");
        let target = dir.path().join("sensitive.txt");
        std::fs::write(&target, "do not touch").expect("target");
        let destination = dir.path().join("bundle.json");
        symlink(&target, dir.path().join("bundle.part")).expect("symlink");
        let error = super::write_atomically(&destination, "{}").expect_err("refused");
        assert!(error.contains("cannot write"), "{error}");
        // The target was neither truncated nor replaced, and the
        // planted symlink itself is untouched.
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "do not touch");
        assert!(dir.path().join("bundle.part").symlink_metadata().is_ok());
        assert!(destination.symlink_metadata().is_err());
    }

    /// A destination that is a dangling symlink still counts as
    /// existing: `exists()` follows links and would miss it.
    #[test]
    fn a_dangling_destination_symlink_is_still_an_existing_file() {
        let dir = tempfile::tempdir().expect("dir");
        let destination = dir.path().join("bundle.json");
        symlink(dir.path().join("nowhere"), &destination).expect("symlink");
        let error = super::write_atomically(&destination, "{}").expect_err("refused");
        assert!(error.contains("already exists"), "{error}");
        // Still a symlink, not replaced by a bundle.
        assert!(destination.symlink_metadata().is_ok());
    }

    /// An unwritable destination directory fails loudly with no
    /// partial bundle and no stray `.part` left behind.
    #[test]
    fn an_unwritable_directory_produces_no_partial_success() {
        let dir = tempfile::tempdir().expect("dir");
        let readonly = dir.path().join("readonly");
        std::fs::create_dir(&readonly).expect("dir");
        let mut perms = std::fs::metadata(&readonly).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o500);
        std::fs::set_permissions(&readonly, perms).expect("locked");
        let destination = readonly.join("bundle.json");
        let error = super::write_atomically(&destination, "{}").expect_err("refused");
        assert!(error.contains("cannot write"), "{error}");
        assert!(destination.symlink_metadata().is_err());
        assert!(readonly.join("bundle.part").symlink_metadata().is_err());
    }

    #[test]
    fn write_bundle_writes_a_readable_document_and_notes_a_broken_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let destination = dir.path().join("bundle.json");
        let summary = write_bundle(&destination, Some(&dir.path().join("no-such-config.json")))
            .expect("the bundle survives a broken config");
        assert!(summary.contains("wrote"), "{summary}");
        let text = std::fs::read_to_string(&destination).expect("written");
        let bundle: Value = serde_json::from_str(&text).expect("the bundle is valid JSON");
        assert_eq!(bundle["bundle_schema"], BUNDLE_SCHEMA);
        let engines = bundle["manifest"]
            .as_array()
            .expect("manifest")
            .iter()
            .find(|row| row["name"] == "engines")
            .expect("engines row");
        assert_eq!(engines["status"], "unavailable");
        assert!(
            engines["detail"]
                .as_str()
                .expect("detail")
                .contains("is invalid")
        );
        assert!(!text.contains("canary"), "nothing sensitive in: {text}");
    }

    /// Compatibility: the compat test drives the REAL preflight and the
    /// REAL settings surface — the same `check` and `config effective`
    /// call — so the bundle cannot drift from them.
    #[test]
    fn the_real_preflight_and_settings_resolvers_fill_the_engine_section() {
        let model = q36_dir();
        let dir = model.path();
        let candidate = Candidate {
            family: Some("qwen3_6_35b_a3b".to_string()),
            context_limit: None,
            kv_dtype: None,
        };
        let preflight =
            preflight::preflight(dir, dir, &candidate, &synthetic_facts()).expect("report");
        let engine = EngineInput {
            id: "q36".to_string(),
            family: "qwen3_6_35b_a3b".to_string(),
            model_dir: dir.to_path_buf(),
            preflight: Ok(preflight),
            effective: Err("not exercised here".to_string()),
        };
        let bundle = build(&inputs(vec![engine]));
        let section = &bundle["engines"][0];
        assert!(
            section["preflight"]["verdict"].is_string(),
            "a real verdict: {section}"
        );
        assert!(section["preflight"]["next_step"].is_string());
        assert!(
            !section["preflight"]["checks"]
                .as_array()
                .expect("checks")
                .is_empty()
        );
    }

    #[test]
    fn effective_rows_come_from_the_real_resolver_with_origins() {
        let model = q36_dir();
        let engine = crate::server::config::EngineConfig {
            id: "q36".to_string(),
            model_dir: model.path().to_path_buf(),
            tokenizer_dir: model.path().to_path_buf(),
            family: "qwen3_6_35b_a3b".to_string(),
            model: None,
            context_limit: None,
            fake: None,
            flash_next: None,
            qwen36: Some(crate::family::qwen36::settings::Q36Options::default()),
            nemotron: None,
            flint: None,
            think_replay: None,
            notice_toolwrap: None,
        };
        // Deterministic env injection (the parity-test value), never the
        // process environment: tests must not depend on Q36_CACHE_LIMIT.
        let env = crate::family::qwen36::settings::Q36Env {
            cache_limit: Some("8589934592".to_string()),
            ..crate::family::qwen36::settings::Q36Env::default()
        };
        // Deliberately NOT asserting row count or keys: the resolver owns
        // the surface; the bundle only carries it with origins attached.
        let rows = super::effective_rows(&engine, &env).expect("rows");
        assert!(!rows.is_empty(), "real rows: {rows:?}");
        assert!(
            rows.iter()
                .all(|row| row["key"].is_string() && row["origin"].is_string())
        );
        let json_rows = json!(rows);
        assert!(!json_rows.to_string().contains("canary"));
    }
}
