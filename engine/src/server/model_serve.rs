//! `enginesrv serve --model <dir>`: the simplified startup.
//!
//! A front end, not a second loader. Discovery reads the directory the
//! way `capability::inspect` already does; the chosen settings go into
//! a normal in-memory `Config` that enters the exact `--config` path,
//! so validation, admission, and serving are the existing ones. The
//! GPU is never probed here — a failed diagnostic CUDA query latches
//! the backend terminal for the rest of the process
//! (`capability::probe`), so device feasibility is left to the
//! load-time admission ledger, which is authoritative anyway.
//!
//! Selection rules: the family comes from `config.json`'s `model_type`
//! matched against the family specs (or `--family` when there is no
//! evidence); the weight format is `gguf` when a GGUF is present
//! (the qualified default — an EXL3-only directory serves `exl3` with
//! its experimental acknowledgement printed, never silently). No
//! experimental feature is armed by this path: every other option is
//! the resolver's own documented default.

use std::path::PathBuf;

use crate::capability::probe::{CompiledBackends, HardwareFacts};
use crate::capability::{Probe, describe_error, inspect, preflight};
use crate::server::config::{Config, EngineConfig, StartupError};

/// The families `--model` can identify and start. One row per
/// detectable family; extending to Flint is a row here plus its
/// loader's own checks staying the authority.
const DETECTABLE_FAMILIES: &[(&str, &[&str])] = &[
    (
        crate::family::qwen36::factory::FAMILY,
        crate::family::QWEN3_6_35B_A3B.model_types,
    ),
    (
        crate::family::qwen36::factory::DENSE_FAMILY,
        crate::family::QWEN3_8_27B_DENSE.model_types,
    ),
];

/// The listen address of a simplified startup, matching the shipped
/// configurations (loopback; `listen_address` rejects anything else).
const DEFAULT_LISTEN: &str = "127.0.0.1:18791";
/// The key variable of a simplified startup, matching the shipped
/// configurations; the server refuses to serve without it set.
const DEFAULT_API_KEY_ENV: &str = "ENGINESRV_API_KEY";

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ModelServeArgs {
    pub(crate) model_dir: PathBuf,
    pub(crate) tokenizer_dir: Option<PathBuf>,
    pub(crate) family: Option<String>,
    pub(crate) context_limit: Option<u32>,
    pub(crate) kv_dtype: Option<String>,
    pub(crate) listen: Option<String>,
    pub(crate) print_config: bool,
}

/// Parse the arguments after `serve` when the first one is `--model`.
/// The flag set is the preflight candidate surface plus the serving
/// address and `--print-config`; anything else is a usage error, not a
/// guess.
pub(crate) fn parse_args(args: &[String]) -> Result<ModelServeArgs, StartupError> {
    let mut parsed = ModelServeArgs::default();
    let mut index = 0;
    let value = |flag: &str, args: &[String], index: usize| -> Result<String, StartupError> {
        args.get(index)
            .cloned()
            .ok_or_else(|| StartupError(format!("serve --model: {flag} needs a value\n{USAGE}")))
    };
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--model" => {
                index += 1;
                parsed.model_dir = PathBuf::from(value("--model", args, index)?);
            }
            "--tokenizer-dir" => {
                index += 1;
                parsed.tokenizer_dir = Some(PathBuf::from(value("--tokenizer-dir", args, index)?));
            }
            "--family" => {
                index += 1;
                parsed.family = Some(value("--family", args, index)?);
            }
            "--context" => {
                index += 1;
                let text = value("--context", args, index)?;
                parsed.context_limit = Some(text.parse().map_err(|_| {
                    StartupError(format!(
                        "serve --model: --context needs a token count, got {text:?}"
                    ))
                })?);
            }
            "--kv" => {
                index += 1;
                parsed.kv_dtype = Some(value("--kv", args, index)?);
            }
            "--listen" => {
                index += 1;
                parsed.listen = Some(value("--listen", args, index)?);
            }
            "--print-config" => parsed.print_config = true,
            other => {
                return Err(StartupError(format!(
                    "serve --model: unknown flag {other:?}\n{USAGE}"
                )));
            }
        }
        index += 1;
    }
    if parsed.model_dir.as_os_str().is_empty() {
        return Err(StartupError(format!(
            "serve --model: --model <dir> is required\n{USAGE}"
        )));
    }
    Ok(parsed)
}

const USAGE: &str = "usage: enginesrv serve --model <dir> [--tokenizer-dir <dir>] [--family <name>] [--context <n>] [--kv f32|q8] [--listen <addr>] [--print-config]";

/// What discovery chose: the family, the weight format, and the
/// options that carry them. Every other option stays the resolver's
/// own default — nothing experimental is armed here.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Discovered {
    pub(crate) family: String,
    pub(crate) weight_format: &'static str,
    pub(crate) options: crate::family::qwen36::settings::Q36Options,
    pub(crate) tokenizer_dir: PathBuf,
}

/// Inspect the directory and choose family, format, and options.
/// Pure filesystem evidence; no preflight, no probing.
fn discover(args: &ModelServeArgs) -> Result<Discovered, StartupError> {
    let tokenizer_dir = args
        .tokenizer_dir
        .clone()
        .unwrap_or_else(|| args.model_dir.clone());
    let inspection = inspect::inspect(&args.model_dir, &tokenizer_dir)
        .map_err(|error| StartupError(describe_error(&error)))?;

    let has_gguf = !inspection.gguf_files.is_empty();
    let has_safetensors = inspection.safetensors_files > 0;
    if !has_gguf && !has_safetensors {
        return Err(StartupError(format!(
            "serve --model: no .gguf or .safetensors weight files in {}",
            args.model_dir.display()
        )));
    }

    let family = select_family(args, &inspection)?;
    let weight_format = select_weight_format(has_gguf, has_safetensors);

    let options = crate::family::qwen36::settings::Q36Options {
        weight_format: Some(weight_format.to_string()),
        context_limit: args.context_limit,
        kv_dtype: args.kv_dtype.clone(),
        ..crate::family::qwen36::settings::Q36Options::default()
    };
    Ok(Discovered {
        family,
        weight_format,
        options,
        tokenizer_dir,
    })
}

/// Discover the model, run the non-probing preflight, and build the
/// in-memory `Config` for the existing serve path. Prints the chosen
/// configuration and any qualification acknowledgement before the
/// engines start.
pub(crate) fn build_config(args: &ModelServeArgs) -> Result<Config, StartupError> {
    let discovered = discover(args)?;

    // The preflight of a serving process: every check that needs no
    // device. CUDA facts stay deliberately unavailable (the module
    // docs); host facts are a /proc read the system endpoint already
    // does per request.
    let facts = HardwareFacts {
        backends: CompiledBackends::of_this_build(),
        cuda: no_cuda_facts(),
        host: crate::capability::probe::probe_host(),
    };
    let candidate = preflight::Candidate {
        family: Some(discovered.family.clone()),
        context_limit: args.context_limit,
        kv_dtype: args.kv_dtype.clone(),
    };
    let report = preflight::preflight(
        &args.model_dir,
        &discovered.tokenizer_dir,
        &candidate,
        &facts,
    )
    .map_err(|error| StartupError(describe_error(&error)))?;
    match report.verdict {
        preflight::Verdict::Unsupported => {
            return Err(StartupError(format!(
                "serve --model: this build cannot serve {}:\n{}next: {}",
                args.model_dir.display(),
                failed_checks(&report),
                report.next_step
            )));
        }
        preflight::Verdict::Experimental => {
            eprintln!(
                "qualification: EXPERIMENTAL — the quality gates for this configuration have not passed; serving it because serve --model prints this acknowledgement"
            );
            eprint!("{}", warned_checks(&report));
        }
        _ => eprint!("{}", warned_checks(&report)),
    }

    let backend = crate::family::qwen36::factory::backend_of_build();
    eprintln!(
        "model: {} — family {}, {} weights, backend {}",
        args.model_dir.display(),
        discovered.family,
        discovered.weight_format,
        backend_name(backend),
    );
    eprintln!(
        "settings: the family defaults for context and KV dtype (override with --context, --kv); memory feasibility is decided at load by the runtime admission ledger — 'enginesrv check {}' prints the full preflight",
        args.model_dir.display()
    );

    Ok(config_from(&discovered, args))
}

/// Assemble the in-memory `Config` — the document a `--config` file
/// would carry, headed for the identical serve path.
fn config_from(discovered: &Discovered, args: &ModelServeArgs) -> Config {
    Config {
        listen: args
            .listen
            .clone()
            .unwrap_or_else(|| DEFAULT_LISTEN.to_string()),
        api_key_env: DEFAULT_API_KEY_ENV.to_string(),
        prefix_reuse: false,
        sample_trace: false,
        web_dir: None,
        engines: vec![EngineConfig {
            id: "q36".to_string(),
            model_dir: args.model_dir.clone(),
            tokenizer_dir: discovered.tokenizer_dir.clone(),
            family: discovered.family.clone(),
            model: None,
            context_limit: None,
            think_replay: None,
            notice_toolwrap: None,
            fake: None,
            flash_next: None,
            qwen36: Some(discovered.options.clone()),
            nemotron: None,
            flint: None,
        }],
    }
}

/// The exact JSON of the built configuration: the same document a
/// `--config` file would carry, so a printed configuration starts
/// identically (`Config::from_file` reads it back).
pub(crate) fn config_json(config: &Config) -> String {
    use serde_json::json;
    let engine = &config.engines[0];
    let mut q36 = serde_json::Map::new();
    if let Some(options) = &engine.qwen36 {
        if let Some(value) = &options.weight_format {
            q36.insert("weight_format".into(), json!(value));
        }
        if let Some(value) = options.context_limit {
            q36.insert("context_limit".into(), json!(value));
        }
        if let Some(value) = &options.kv_dtype {
            q36.insert("kv_dtype".into(), json!(value));
        }
    }
    let document = json!({
        "listen": config.listen,
        "api_key_env": config.api_key_env,
        "prefix_reuse": config.prefix_reuse,
        "engines": [{
            "id": engine.id,
            "family": engine.family,
            "model_dir": engine.model_dir.display().to_string(),
            "tokenizer_dir": engine.tokenizer_dir.display().to_string(),
            "qwen36": q36,
        }],
    });
    serde_json::to_string_pretty(&document).expect("the built config is valid JSON")
}

fn select_family(
    args: &ModelServeArgs,
    inspection: &inspect::ModelInspection,
) -> Result<String, StartupError> {
    if let Some(family) = &args.family {
        if DETECTABLE_FAMILIES.iter().any(|(name, _)| name == family) {
            return Ok(family.clone());
        }
        return Err(StartupError(format!(
            "serve --model: family {family} has no simplified startup yet; write a --config file for it (engine/config/README.md). Detectable families: {}",
            detectable_names()
        )));
    }
    let mut matches: Vec<&str> = DETECTABLE_FAMILIES
        .iter()
        .filter(|(_, types)| {
            types
                .iter()
                .any(|t| inspection.config_model_types.iter().any(|seen| seen == t))
        })
        .map(|(name, _)| *name)
        .collect();
    match matches.pop() {
        Some(name) if matches.is_empty() => Ok(name.to_string()),
        _ => Err(StartupError(format!(
            "serve --model: cannot identify the family from config.json (model_type {:?}); pass --family. Detectable families: {}",
            inspection.config_model_types,
            detectable_names()
        ))),
    }
}

fn select_weight_format(has_gguf: bool, has_safetensors: bool) -> &'static str {
    if has_gguf {
        if has_safetensors {
            eprintln!(
                "weights: the directory holds GGUF and EXL3 shards; serving the qualified GGUF default (EXL3 is experimental — select it with --config and weight_format exl3)"
            );
        }
        "gguf"
    } else {
        "exl3"
    }
}

fn detectable_names() -> String {
    DETECTABLE_FAMILIES
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn backend_name(backend: crate::family::qwen36::settings::Backend) -> &'static str {
    match backend {
        crate::family::qwen36::settings::Backend::Mlx => "mlx",
        crate::family::qwen36::settings::Backend::Cuda => "cuda",
    }
}

fn no_cuda_facts() -> crate::capability::probe::CudaFacts {
    let reason = "simplified startup does not probe the GPU (a failed diagnostic query latches the CUDA backend); the load-time admission ledger decides";
    crate::capability::probe::CudaFacts {
        device_count: Probe::Unavailable(reason.to_string()),
        first_device_integrated: Probe::Unavailable(reason.to_string()),
        first_device_free_bytes: Probe::Unavailable(reason.to_string()),
        first_device_total_bytes: Probe::Unavailable(reason.to_string()),
    }
}

fn failed_checks(report: &preflight::PreflightReport) -> String {
    report
        .checks
        .iter()
        .filter(|check| check.status == preflight::CheckStatus::Fail)
        .map(|check| format!("  {}: {}\n", check.name, check.detail))
        .collect()
}

fn warned_checks(report: &preflight::PreflightReport) -> String {
    report
        .checks
        .iter()
        .filter(|check| {
            check.status != preflight::CheckStatus::Pass
                && check.status != preflight::CheckStatus::Fail
        })
        .map(|check| format!("note: {}: {}\n", check.name, check.detail))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ModelServeArgs, discover, parse_args};
    use crate::family::qwen36::settings::Q36Options;
    use crate::server::config::Config;
    use std::path::{Path, PathBuf};

    /// A GGUF v3 file with no metadata and no tensors: the header plus
    /// padding to the aligned data start, which is all `GgufHeader::parse`
    /// needs to accept the file as an inventory fact.
    fn minimal_gguf() -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(32, 0);
        bytes
    }

    fn copy_tokenizer(dir: &Path) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            std::fs::copy(source.join(name), dir.join(name)).expect("tokenizer fixture");
        }
    }

    /// A model directory with the given weights and `config.json`
    /// `model_type`, plus the pinned tokenizer fixture.
    fn model_dir(model_type: Option<&str>, gguf: bool, safetensors: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        if gguf {
            std::fs::write(dir.path().join("model.gguf"), minimal_gguf()).expect("gguf");
        }
        if safetensors {
            std::fs::write(dir.path().join("model-00001.safetensors"), vec![0u8; 16])
                .expect("safetensors");
        }
        if let Some(model_type) = model_type {
            std::fs::write(
                dir.path().join("config.json"),
                format!(r#"{{"model_type":"{model_type}"}}"#),
            )
            .expect("config.json");
        }
        copy_tokenizer(dir.path());
        dir
    }

    fn args_for(dir: &Path) -> ModelServeArgs {
        ModelServeArgs {
            model_dir: dir.to_path_buf(),
            ..ModelServeArgs::default()
        }
    }

    #[test]
    fn parse_args_reads_the_surface_and_refuses_unknowns() {
        let args: Vec<String> = [
            "--model",
            "/m",
            "--family",
            "qwen3_6_35b_a3b",
            "--context",
            "4096",
            "--kv",
            "q8",
            "--listen",
            "127.0.0.1:9000",
            "--tokenizer-dir",
            "/t",
            "--print-config",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_args(&args).expect("parses");
        assert_eq!(parsed.model_dir, PathBuf::from("/m"));
        assert_eq!(parsed.family.as_deref(), Some("qwen3_6_35b_a3b"));
        assert_eq!(parsed.context_limit, Some(4096));
        assert_eq!(parsed.kv_dtype.as_deref(), Some("q8"));
        assert_eq!(parsed.listen.as_deref(), Some("127.0.0.1:9000"));
        assert_eq!(parsed.tokenizer_dir, Some(PathBuf::from("/t")));
        assert!(parsed.print_config);

        let only_model = parse_args(&["--model".to_string(), "/m".to_string()]).expect("parses");
        assert_eq!(
            only_model,
            ModelServeArgs {
                model_dir: PathBuf::from("/m"),
                ..ModelServeArgs::default()
            }
        );

        let unknown = parse_args(&[
            "--model".to_string(),
            "/m".to_string(),
            "--turbo".to_string(),
        ])
        .unwrap_err();
        assert!(unknown.0.contains("unknown flag"), "{}", unknown.0);
        let bad_context = parse_args(&[
            "--model".to_string(),
            "/m".to_string(),
            "--context".to_string(),
            "huge".to_string(),
        ])
        .unwrap_err();
        assert!(bad_context.0.contains("token count"), "{}", bad_context.0);
        let missing_value = parse_args(&["--model".to_string()]).unwrap_err();
        assert!(
            missing_value.0.contains("needs a value"),
            "{}",
            missing_value.0
        );
    }

    #[test]
    fn a_gguf_directory_is_detected_with_the_qualified_default_options() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(discovered.family, "qwen3_6_35b_a3b");
        assert_eq!(discovered.weight_format, "gguf");
        // Only the discovered format is set: context and KV stay the
        // resolver's own defaults — nothing experimental is armed.
        assert_eq!(
            discovered.options,
            Q36Options {
                weight_format: Some("gguf".to_string()),
                ..Q36Options::default()
            }
        );
    }

    #[test]
    fn the_dense_model_type_maps_to_the_dense_family() {
        let dir = model_dir(Some("qwen3_5"), false, true);
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(discovered.family, "qwen3_8_27b_dense");
        assert_eq!(discovered.weight_format, "exl3");
    }

    #[test]
    fn an_exl3_only_directory_is_detected_as_exl3() {
        let dir = model_dir(Some("qwen3_5_moe"), false, true);
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(discovered.weight_format, "exl3");
    }

    #[test]
    fn an_explicit_family_replaces_missing_evidence() {
        let dir = model_dir(None, true, false);
        let mut args = args_for(dir.path());
        args.family = Some("qwen3_6_35b_a3b".to_string());
        let discovered = discover(&args).expect("discovers");
        assert_eq!(discovered.family, "qwen3_6_35b_a3b");
    }

    #[test]
    fn flags_reach_the_options_verbatim() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let mut args = args_for(dir.path());
        args.context_limit = Some(8192);
        args.kv_dtype = Some("q8".to_string());
        let discovered = discover(&args).expect("discovers");
        assert_eq!(discovered.options.context_limit, Some(8192));
        assert_eq!(discovered.options.kv_dtype.as_deref(), Some("q8"));
    }

    #[test]
    fn without_evidence_the_family_is_requested_not_guessed() {
        let dir = model_dir(None, true, false);
        let error = discover(&args_for(dir.path())).unwrap_err();
        assert!(error.0.contains("--family"), "{}", error.0);
        assert!(error.0.contains("qwen3_6_35b_a3b"), "{}", error.0);

        let unknown = model_dir(Some("llama"), true, false);
        let error = discover(&args_for(unknown.path())).unwrap_err();
        assert!(error.0.contains("--family"), "{}", error.0);
        assert!(error.0.contains("llama"), "{}", error.0);
    }

    #[test]
    fn flint_is_explicitly_not_simplified_yet() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let mut args = args_for(dir.path());
        args.family = Some("flint".to_string());
        let error = discover(&args).unwrap_err();
        assert!(error.0.contains("no simplified startup"), "{}", error.0);
        assert!(error.0.contains("--config"), "{}", error.0);
    }

    #[test]
    fn a_directory_without_weights_is_refused() {
        let dir = model_dir(Some("qwen3_5_moe"), false, false);
        let error = discover(&args_for(dir.path())).unwrap_err();
        assert!(error.0.contains("no .gguf or .safetensors"), "{}", error.0);
    }

    #[test]
    fn a_missing_directory_is_reported_by_the_inspection() {
        let args = ModelServeArgs {
            model_dir: PathBuf::from("/nonexistent/fifi-model"),
            ..ModelServeArgs::default()
        };
        let error = discover(&args).unwrap_err();
        assert!(error.0.contains("not a directory"), "{}", error.0);
    }

    // Fails if discovery stops refusing a network-mounted model directory:
    // `discover` calls the inspection, which refuses before it lists the
    // directory. `build_config` runs `discover` first, so its refusal here is
    // the same discovery refusal; the later preflight call is not what this
    // test exercises, and the engine start is covered by the `start_engines`
    // tests.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_model_directory_on_a_network_mount_is_refused_at_discovery() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let args = args_for(dir.path());
        let error = crate::storage_guard::seam::with_nfs_at(dir.path(), || {
            (
                discover(&args).map(|_| ()).unwrap_err(),
                super::build_config(&args).map(|_| ()).unwrap_err(),
            )
        });
        assert!(
            error.0.0.contains("must be on local storage"),
            "{}",
            error.0.0
        );
        assert!(
            error.1.0.contains("must be on local storage"),
            "{}",
            error.1.0
        );
        assert!(
            discover(&args).is_ok(),
            "the same directory is fine on a local filesystem"
        );
    }

    #[test]
    fn the_printed_config_round_trips_through_the_file_path() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let mut args = args_for(dir.path());
        args.listen = Some("127.0.0.1:9100".to_string());
        let discovered = discover(&args).expect("discovers");
        let config = super::config_from(&discovered, &args);
        let printed = super::config_json(&config);

        let file = dir.path().join("printed.json");
        std::fs::write(&file, &printed).expect("write printed config");
        let reread = Config::from_file(&file).expect("the printed config is a valid file");
        assert_eq!(reread.listen, "127.0.0.1:9100");
        assert_eq!(reread.api_key_env, "ENGINESRV_API_KEY");
        assert!(!reread.prefix_reuse);
        assert_eq!(reread.engines.len(), 1);
        let engine = &reread.engines[0];
        assert_eq!(engine.family, "qwen3_6_35b_a3b");
        assert_eq!(engine.model_dir, args.model_dir);
        assert_eq!(engine.qwen36, Some(discovered.options));
        // No extra fields can hide in the printed document: from_file
        // runs deny_unknown_fields, so the round-trip above is the
        // proof. The key variable is named (as in every config file);
        // its VALUE lives only in the environment.
        assert!(printed.contains("ENGINESRV_API_KEY"), "{printed}");
    }

    /// The refusal a user of this binary sees, quoted verbatim, with
    /// preflight's own next-step line. Each build picks its own arm below:
    /// the CPU build's backend check, the device builds' settings and
    /// tokenizer-pin refusals. The green zero-config path is the CUDA
    /// build's, exercised by the hardware lane; the settings the CPU
    /// build WOULD resolve are covered by the discovery tests above
    /// and the parity test of `run`.
    #[test]
    fn build_config_refusals_quote_preflight_verbatim() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let error = super::build_config(&args_for(dir.path())).unwrap_err();
        assert!(error.0.contains("cannot serve"), "{}", error.0);
        if cfg!(feature = "native-mlx") {
            // A native-mlx build reaches the settings validators before the backend
            // choice and refuses the default-args fixture (no cache limit) — also a
            // verbatim preflight quote, so this arm pins that refusal instead.
            assert!(
                error.0.contains("cache_limit_bytes is missing"),
                "{}",
                error.0
            );
        } else if cfg!(all(feature = "native-cuda", target_os = "linux")) {
            // A Linux native-cuda build picks the CUDA backend and reaches the
            // preflight validators: the fixture's tokenizer is not the pinned one,
            // so this arm pins that verbatim refusal instead of the CPU build's
            // "no GPU backend". (native-cuda is a no-op off Linux, engine/build.rs.)
            assert!(error.0.contains("tokenizer pin"), "{}", error.0);
        } else {
            assert!(error.0.contains("no GPU backend"), "{}", error.0);
        }
        assert!(error.0.contains("next:"), "{}", error.0);

        // The same refusal carries EXL3-only directories on every build
        // (detection itself is covered above): even a CUDA build refuses
        // them here, because the fixture's tokenizer fails the preflight
        // pin before the experimental verdict can speak.
        let exl3 = model_dir(Some("qwen3_5_moe"), false, true);
        let error = super::build_config(&args_for(exl3.path())).unwrap_err();
        if cfg!(all(feature = "native-cuda", target_os = "linux")) {
            // The EXL3-only directory's experimental qualification never
            // gets to speak here: its fixture tokenizer fails the same
            // preflight check first, and a failed check outranks the
            // experimental verdict, so this build refuses it too.
            assert!(error.0.contains("tokenizer pin"), "{}", error.0);
        } else {
            if cfg!(feature = "native-mlx") {
                assert!(
                    error.0.contains("cache_limit_bytes is missing"),
                    "{}",
                    error.0
                );
            } else {
                assert!(error.0.contains("no GPU backend"), "{}", error.0);
            }
        }
    }
}
