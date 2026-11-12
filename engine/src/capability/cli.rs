//! The diagnostic command line: `doctor`, `check`, and `config validate`.
//!
//! Parsing produces an `Action`; execution renders the same report
//! structs every other surface will consume, as text or `--json`. The
//! server path is deliberately not here: `server::run_cli` owns argv
//! dispatch and hands anything that is not a serve invocation to
//! `dispatch` below, so `--config` keeps its exact current meaning.

use std::path::PathBuf;

use super::{
    Probe, gib,
    preflight::{Candidate, PreflightReport, Verdict},
    probe::HardwareFacts,
};

pub const USAGE: &str = "\
usage:
  enginesrv --config <path>            serve (same as `serve --config <path>`)
  enginesrv serve --model <dir>        simplified startup: discover the model,
                    preflight it, and serve the qualified defaults
                    ([--tokenizer-dir <dir>] [--family <name>]
                    [--context <tokens>] [--kv <f32|bf16|q8>] [--listen <addr>]
                    [--print-config])
  enginesrv doctor [--config <path>] [--json] [--bundle <file>]
  enginesrv check <model_dir> [--tokenizer-dir <dir>] [--family <name>]
                    [--context <tokens>] [--kv <f32|bf16|q8>] [--json]
  enginesrv config validate <path>
  enginesrv config effective <path> [--json]
  enginesrv --help | --version

doctor  Report what this build can serve on this machine: compiled
        backends, CUDA device and free VRAM, host RAM and threads.
        With --config, also validate that configuration file.
        With --bundle <file>, write a sanitized diagnostic support
        bundle instead (one JSON file: platform, hardware, and — with
        --config — per-engine preflights and effective settings, plus
        a manifest of what was collected and a readable summary).
        Home paths become ~, no secrets or prompts are collected, and
        an existing file is never overwritten. Attach it to an issue.
check   Preflight one model directory: compatibility verdict, the
        checks behind it, and a memory estimate against the probed
        free memory. Exit 0 when servable, 1 otherwise.
config validate
        Parse and validate a configuration file, including that the
        environment variable named by api_key_env is set (its value is
        never printed).
config effective
        Print the resolved configuration with the origin of every
        principal setting (config file, environment, derived,
        default, ignored on this backend). Resolution is the same
        code the server starts with; a file that would not start is
        reported instead. The API key value is never printed.
";

#[derive(Debug, PartialEq)]
pub enum Action {
    Help,
    Version,
    Doctor {
        config: Option<PathBuf>,
        json: bool,
        bundle: Option<PathBuf>,
    },
    Check {
        model_dir: PathBuf,
        tokenizer_dir: Option<PathBuf>,
        candidate: Candidate,
        json: bool,
    },
    ConfigValidate(PathBuf),
    ConfigEffective(PathBuf, bool),
}

/// Parse the diagnostic commands. Returns `None` when argv is a serve
/// invocation (`serve …`, `--config …`, or anything else): the caller
/// keeps owning those, including their usage errors.
pub fn parse(args: &[String]) -> Option<Result<Action, String>> {
    match args.first().map(String::as_str) {
        Some("--help" | "-h" | "help") => Some(Ok(Action::Help)),
        Some("--version" | "-V" | "version") => Some(Ok(Action::Version)),
        Some("doctor") => Some(parse_doctor(&args[1..])),
        Some("check") => Some(parse_check(&args[1..])),
        Some("config") => Some(match args.get(1).map(String::as_str) {
            Some("validate") => match args.get(2) {
                Some(path) if args.len() == 3 => Ok(Action::ConfigValidate(PathBuf::from(path))),
                _ => Err("usage: enginesrv config validate <path>".to_string()),
            },
            Some("effective") => match (args.get(2), args.get(3)) {
                (Some(path), None) => Ok(Action::ConfigEffective(PathBuf::from(path), false)),
                (Some(path), Some(flag)) if flag == "--json" && args.len() == 4 => {
                    Ok(Action::ConfigEffective(PathBuf::from(path), true))
                }
                _ => Err("usage: enginesrv config effective <path> [--json]".to_string()),
            },
            _ => Err("usage: enginesrv config validate|effective <path>".to_string()),
        }),
        _ => None,
    }
}

fn parse_doctor(rest: &[String]) -> Result<Action, String> {
    let mut config = None;
    let mut json = false;
    let mut bundle = None;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--config" => {
                config = Some(PathBuf::from(
                    it.next()
                        .ok_or("doctor: --config needs a path".to_string())?,
                ))
            }
            "--bundle" => {
                bundle = Some(PathBuf::from(
                    it.next()
                        .ok_or("doctor: --bundle needs a destination path".to_string())?,
                ))
            }
            other => return Err(format!("doctor: unknown argument {other:?}")),
        }
    }
    if json && bundle.is_some() {
        return Err(
            "doctor: --bundle writes a file and --json prints one; pick either".to_string(),
        );
    }
    Ok(Action::Doctor {
        config,
        json,
        bundle,
    })
}

fn parse_check(rest: &[String]) -> Result<Action, String> {
    let mut positional = None;
    let mut tokenizer_dir = None;
    let mut candidate = Candidate::default();
    let mut json = false;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--tokenizer-dir" => {
                tokenizer_dir = Some(PathBuf::from(
                    it.next()
                        .ok_or("check: --tokenizer-dir needs a path".to_string())?,
                ))
            }
            "--family" => {
                candidate.family = Some(
                    it.next()
                        .ok_or("check: --family needs a name".to_string())?
                        .clone(),
                )
            }
            "--context" => {
                let value = it
                    .next()
                    .ok_or("check: --context needs a token count".to_string())?;
                candidate.context_limit = Some(
                    value
                        .parse()
                        .map_err(|_| format!("check: --context {value:?} is not a token count"))?,
                );
            }
            "--kv" => {
                candidate.kv_dtype = Some(
                    it.next()
                        .ok_or("check: --kv needs one of f32, bf16, q8".to_string())?
                        .clone(),
                )
            }
            other if !other.starts_with('-') && positional.is_none() => {
                positional = Some(PathBuf::from(other));
            }
            other => return Err(format!("check: unknown argument {other:?}")),
        }
    }
    match positional {
        Some(model_dir) => Ok(Action::Check {
            model_dir,
            tokenizer_dir,
            candidate,
            json,
        }),
        None => Err("usage: enginesrv check <model_dir> [options]".to_string()),
    }
}

/// Run one diagnostic action, printing to stdout. `Err` text is for the
/// caller's `enginesrv: <error>` line and exit 1.
pub fn run(action: &Action) -> Result<(), String> {
    match action {
        Action::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Action::Version => {
            println!("enginesrv {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Action::Doctor {
            config,
            json,
            bundle,
        } => {
            if let Some(destination) = bundle {
                println!(
                    "{}\ninspect the bundle before sharing it: it is one JSON file with a summary",
                    super::bundle::write_bundle(destination, config.as_deref())?
                );
                return Ok(());
            }
            let facts = super::probe::probe();
            let validation = config.as_deref().map(validate_config);
            if *json {
                let value = serde_json::json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "hardware": facts,
                    "config": validation.as_ref().map(|lines| {
                        serde_json::json!({"path": config, "result": lines})
                    }),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", render_doctor(&facts));
                if let (Some(path), Some(result)) = (config, &validation) {
                    println!("config {}:", path.display());
                    match result {
                        Ok(lines) => lines.iter().for_each(|line| println!("  {line}")),
                        Err(error) => println!("  INVALID: {error}"),
                    }
                }
            }
            match validation {
                Some(Err(error)) => Err(format!("config is invalid: {error}")),
                _ => Ok(()),
            }
        }
        Action::Check {
            model_dir,
            tokenizer_dir,
            candidate,
            json,
        } => {
            let tokenizer = tokenizer_dir.as_deref().unwrap_or(model_dir.as_path());
            let facts = super::probe::probe();
            let report = super::preflight::preflight(model_dir, tokenizer, candidate, &facts)
                .map_err(|error| super::describe_error(&error))?;
            if *json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", render_preflight(&report));
            }
            if report.verdict.servable() {
                Ok(())
            } else {
                Err(format!(
                    "check: verdict {} for {}",
                    report.verdict.name(),
                    report.model_dir.display()
                ))
            }
        }
        Action::ConfigEffective(path, json) => run_config_effective(path, *json),
        Action::ConfigValidate(path) => match validate_config(path) {
            Ok(lines) => {
                println!("config {} is valid:", path.display());
                lines.iter().for_each(|line| println!("  {line}"));
                Ok(())
            }
            Err(error) => Err(format!("config {} is invalid: {error}", path.display())),
        },
    }
}

/// The server's own validation (`Config::from_file` runs the full check),
/// plus the startup key presence rule. The key value is read exactly as
/// startup reads it and never returned or printed.
pub fn validate_config(path: &std::path::Path) -> Result<Vec<String>, String> {
    use crate::server::config::Config;
    let config = Config::from_file(path).map_err(|error| error.to_string())?;
    config
        .api_key(|name| std::env::var(name).ok())
        .map_err(|error| error.to_string())?;
    let mut lines = vec![format!(
        "listen {}, {} engine(s), api key variable is set",
        config.listen_address().map_err(|e| e.to_string())?,
        config.engines.len()
    )];
    for engine in &config.engines {
        lines.push(format!(
            "engine {}: family {}, model {}, model_dir {}",
            engine.id,
            engine.family,
            engine.model_name(),
            engine.model_dir.display()
        ));
    }
    Ok(lines)
}

/// The resolved configuration of one file, with origins. Resolution is
/// the factories' own (`Q36Settings::new_for_family` for the Qwen3.6
/// families), so what prints here is what the server would start with —
/// and a file that would refuse to start reports that refusal.
fn run_config_effective(path: &std::path::Path, json: bool) -> Result<(), String> {
    use crate::server::config::Config;
    let config = Config::from_file(path).map_err(|error| error.to_string())?;
    let key_set = config.api_key(|name| std::env::var(name).ok()).is_ok();
    let mut rendered = String::new();
    let mut json_engines = Vec::new();
    rendered.push_str(&format!(
        "listen: {}\napi_key_env: {} ({})\nprefix_reuse: {}\n",
        config.listen_address().map_err(|e| e.to_string())?,
        config.api_key_env,
        if key_set { "set" } else { "NOT SET" },
        config.prefix_reuse
    ));
    for engine in &config.engines {
        let is_q36_family = engine.family == crate::family::qwen36::factory::FAMILY
            || engine.family == crate::family::qwen36::factory::DENSE_FAMILY;
        if engine.family == crate::family::flint::FAMILY {
            let default_options = crate::family::flint::settings::FlintOptions::cuda_default();
            let rows = super::effective::flint_effective(
                engine.flint.as_ref().unwrap_or(&default_options),
                &crate::family::flint::budget::FlintEnv::capture(),
                None,
            );
            rendered.push_str(&format!(
                "engine {} ({}, model {}):\n",
                engine.id,
                engine.family,
                engine.model_name()
            ));
            for row in &rows {
                rendered.push_str(&format!(
                    "  {:32} {:>16}  ({})\n",
                    row.key,
                    row.value,
                    row.origin.name()
                ));
            }
            json_engines.push(serde_json::json!({
                "id": engine.id,
                "family": engine.family,
                "settings": rows.iter().map(|row| serde_json::json!({
                    "key": row.key, "value": row.value, "origin": row.origin.name()
                })).collect::<Vec<_>>(),
            }));
            continue;
        }
        if !is_q36_family {
            rendered.push_str(&format!(
                "engine {} ({}): per-key origins are composed for the Qwen3.6 families only so far; model_dir {}\n",
                engine.id,
                engine.family,
                engine.model_dir.display()
            ));
            json_engines.push(serde_json::json!({
                "id": engine.id, "family": engine.family, "settings": [],
            }));
            continue;
        }
        use crate::family::qwen36::factory::backend_of_build;
        use crate::family::qwen36::settings::{Q36Env, Q36Settings};
        let env = Q36Env {
            cache_limit: std::env::var("Q36_CACHE_LIMIT").ok(),
            kv_geometric: std::env::var("Q36_KV_GEOMETRIC").ok(),
            spec_step_window: std::env::var("Q36_SPEC_STEP_WINDOW").ok(),
        };
        let backend = backend_of_build();
        let spec = if engine.family == crate::family::qwen36::factory::DENSE_FAMILY {
            crate::family::spec::QWEN3_8_27B_DENSE
        } else {
            crate::family::spec::QWEN3_6_35B_A3B
        };
        let settings = Q36Settings::new_for_family(
            &engine.model_dir,
            &engine.tokenizer_dir,
            engine.qwen36.as_ref(),
            &env,
            backend,
            spec,
        )
        .map_err(|error| {
            format!(
                "engine {} would not start: {}",
                engine.id,
                super::describe_error(&error)
            )
        })?;
        if config.prefix_reuse {
            settings
                .check_prefix_reuse_cap(true)
                .map_err(|error| super::describe_error(&error))?;
        }
        let default_options = crate::family::qwen36::settings::Q36Options::default();
        let rows = super::effective::q36_effective(
            engine.qwen36.as_ref().unwrap_or(&default_options),
            env.cache_limit.as_deref(),
            backend,
            &settings,
        );
        rendered.push_str(&format!(
            "engine {} ({}, model {}):\n",
            engine.id,
            engine.family,
            engine.model_name()
        ));
        for row in &rows {
            rendered.push_str(&format!(
                "  {:32} {:>16}  ({})\n",
                row.key,
                row.value,
                row.origin.name()
            ));
        }
        json_engines.push(serde_json::json!({
            "id": engine.id, "family": engine.family, "settings": rows,
        }));
    }
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "listen": config.listen_address().map_err(|e| e.to_string())?.to_string(),
                "api_key_env": config.api_key_env,
                "api_key_set": key_set,
                "prefix_reuse": config.prefix_reuse,
                "engines": json_engines,
            }))
            .map_err(|e| e.to_string())?
        );
    } else {
        print!("{rendered}");
    }
    if key_set {
        Ok(())
    } else {
        Err(format!(
            "environment variable {} is not set or is empty",
            config.api_key_env
        ))
    }
}

fn probe_line<T>(label: &str, probe: &Probe<T>, render: impl Fn(&T) -> String) -> String {
    match probe {
        Probe::Measured(value) => format!("{label}: {}", render(value)),
        Probe::Unavailable(reason) => format!("{label}: unavailable ({reason})"),
    }
}

pub fn render_doctor(facts: &HardwareFacts) -> String {
    let mut out = String::new();
    out.push_str(&format!("enginesrv {}\n", env!("CARGO_PKG_VERSION")));
    let backends: Vec<&str> = [
        (facts.backends.cuda, "CUDA"),
        (facts.backends.mlx, "MLX (Apple Silicon)"),
    ]
    .into_iter()
    .filter_map(|(on, name)| on.then_some(name))
    .collect();
    out.push_str(&format!(
        "backends compiled in: {}\n",
        if backends.is_empty() {
            "none (fake family only)".to_string()
        } else {
            backends.join(", ")
        }
    ));
    out.push_str(&probe_line(
        "CUDA devices (measured now)",
        &facts.cuda.device_count,
        |n| n.to_string(),
    ));
    out.push('\n');
    out.push_str(&probe_line(
        "device 0 shares memory with host",
        &facts.cuda.first_device_integrated,
        |v| {
            if *v {
                "yes (integrated)".into()
            } else {
                "no".into()
            }
        },
    ));
    out.push('\n');
    out.push_str(&probe_line(
        "device 0 VRAM free (measured now)",
        &facts.cuda.first_device_free_bytes,
        |b| gib(*b),
    ));
    out.push('\n');
    out.push_str(&probe_line(
        "device 0 VRAM total",
        &facts.cuda.first_device_total_bytes,
        |b| gib(*b),
    ));
    out.push('\n');
    out.push_str(&probe_line(
        "host RAM available (measured now)",
        &facts.host.mem_available_bytes,
        |b| gib(*b),
    ));
    out.push('\n');
    out.push_str(&probe_line(
        "host RAM total",
        &facts.host.mem_total_bytes,
        |b| gib(*b),
    ));
    out.push('\n');
    out.push_str(&format!("host CPU threads: {}\n", facts.host.cpu_threads));
    if facts.cuda.device_count.as_ref().is_some_and(|n| *n > 1) {
        out.push_str(
            "note: memory above is device 0 only; a Qwen3.6 engine spans more devices only when its cuda_devices setting names them, under its own placement plan\n",
        );
    }
    out
}

pub fn render_preflight(report: &PreflightReport) -> String {
    use super::preflight::CheckStatus;
    let mut out = String::new();
    out.push_str(&format!("model_dir: {}\n", report.model_dir.display()));
    out.push_str(&format!(
        "family: {}\n",
        report.family.as_deref().unwrap_or("(not identified)")
    ));
    out.push_str(&format!(
        "compatibility: {:?} \u{b7} qualification: {:?} \u{b7} resource fit: {:?}\n",
        report.dimensions.compatibility,
        report.dimensions.qualification,
        report.dimensions.resource_fit
    ));
    for check in &report.checks {
        let mark = match check.status {
            CheckStatus::Pass => "PASS",
            CheckStatus::Warn => "WARN",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Info => "INFO",
        };
        out.push_str(&format!("[{mark}] {}: {}\n", check.name, check.detail));
    }
    for decision in &report.decisions {
        out.push_str(&format!(
            "decision {}: {} (fixed at {})\n",
            decision.name,
            decision.chosen,
            decision.fixed_at.name()
        ));
        for alternative in &decision.alternatives {
            out.push_str(&format!(
                "  alternative {}: {} — {}\n",
                alternative.option, alternative.verdict, alternative.reason
            ));
        }
    }
    if let Some(estimate) = &report.estimate {
        out.push_str(&format!(
            "estimate (derived; workspace and residency policy not included): checkpoint files {} + KV {} ({} B/position x {}) + fixed state {} = {}\n",
            gib(estimate.weight_bytes),
            gib(estimate.kv_bytes),
            estimate.kv_bytes_per_position,
            estimate.context_limit,
            super::mib(estimate.fixed_state_bytes),
            gib(estimate.total_bytes),
        ));
        if !estimate.admission_charge_terms.is_empty() {
            let terms: Vec<String> = estimate
                .admission_charge_terms
                .iter()
                .map(|(name, bytes)| format!("{name} {}", super::mib(*bytes)))
                .collect();
            out.push_str(&format!(
                "runtime admission charges (estimate): {} total = {}\n",
                super::mib(estimate.admission_charges_bytes),
                terms.join(" + ")
            ));
        }
        if let (Some(free), Some(limit)) = (estimate.budget_free_bytes, estimate.budget_limit_bytes)
        {
            let share = if estimate.budget_limit_is_policy_share {
                format!(
                    " ({} permille policy share)",
                    crate::memory::MemoryPolicy::default_share().share_permille()
                )
            } else {
                String::new()
            };
            out.push_str(&format!(
                "budget: {} = {} free, admission ceiling {}{}\n",
                estimate.budget_kind,
                gib(free),
                gib(limit),
                share,
            ));
        } else {
            out.push_str(&format!("budget: {}\n", estimate.budget_kind));
        }
        let fit_line = match (estimate.fit, estimate.borderline) {
            (super::preflight::ResourceFit::Fits, _) => "estimated fit: fits the measured budget\n",
            (super::preflight::ResourceFit::DoesNotFit, _) => {
                "estimated fit: exceeds the measured budget\n"
            }
            (super::preflight::ResourceFit::Unknown, true) => {
                "estimated fit: borderline (inside the measurement band); load-time admission decides\n"
            }
            (super::preflight::ResourceFit::Unknown, false) => {
                "estimated fit: unverified (no budget probed)\n"
            }
        };
        out.push_str(fit_line);
        if let Some(fitting) = estimate.largest_fitting_context {
            out.push_str(&format!(
                "largest fitting context (proposal only): {fitting}\n"
            ));
        }
    }
    out.push_str(&format!("verdict: {}\n", verdict_label(report.verdict)));
    out.push_str(&format!("next step: {}\n", report.next_step));
    out
}

fn verdict_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Ready => "READY",
        Verdict::ReadyWithAdjustments => "READY WITH ADJUSTMENTS",
        Verdict::ResourceConstrained => "RESOURCE-CONSTRAINED",
        Verdict::Experimental => "EXPERIMENTAL",
        Verdict::Unsupported => "UNSUPPORTED",
    }
}

#[cfg(test)]
mod tests {
    use super::{Action, parse};
    use std::path::PathBuf;

    fn args(text: &[&str]) -> Vec<String> {
        text.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn serve_invocations_are_not_diagnostic_actions() {
        assert!(parse(&args(&["--config", "f.json"])).is_none());
        assert!(parse(&args(&["serve", "--config", "f.json"])).is_none());
        assert!(parse(&args(&[])).is_none());
        assert!(parse(&args(&["--nonsense"])).is_none());
    }

    #[test]
    fn help_and_version_are_actions() {
        assert_eq!(parse(&args(&["--help"])), Some(Ok(Action::Help)));
        assert_eq!(parse(&args(&["--version"])), Some(Ok(Action::Version)));
    }

    #[test]
    fn check_parses_dir_and_overrides() {
        let parsed = parse(&args(&[
            "check",
            "/m",
            "--family",
            "qwen3_6_35b_a3b",
            "--context",
            "131072",
            "--kv",
            "q8",
            "--json",
        ]))
        .expect("action")
        .expect("ok");
        let Action::Check {
            model_dir,
            candidate,
            json,
            ..
        } = parsed
        else {
            panic!("expected check action");
        };
        assert_eq!(model_dir.to_str(), Some("/m"));
        assert_eq!(candidate.family.as_deref(), Some("qwen3_6_35b_a3b"));
        assert_eq!(candidate.context_limit, Some(131_072));
        assert_eq!(candidate.kv_dtype.as_deref(), Some("q8"));
        assert!(json);
    }

    #[test]
    fn check_refuses_missing_dir_and_bad_context() {
        assert!(parse(&args(&["check"])).expect("action").is_err());
        assert!(
            parse(&args(&["check", "/m", "--context", "lots"]))
                .expect("action")
                .is_err()
        );
        assert!(
            parse(&args(&["check", "/m", "--bogus"]))
                .expect("action")
                .is_err()
        );
    }

    #[test]
    fn doctor_and_config_validate_parse() {
        assert_eq!(
            parse(&args(&["doctor", "--json"])),
            Some(Ok(Action::Doctor {
                config: None,
                json: true,
                bundle: None
            }))
        );
        assert_eq!(
            parse(&args(&[
                "doctor", "--bundle", "out.json", "--config", "f.json"
            ])),
            Some(Ok(Action::Doctor {
                config: Some(PathBuf::from("f.json")),
                json: false,
                bundle: Some(PathBuf::from("out.json"))
            }))
        );
        // The two output modes are mutually exclusive: a file that
        // exists cannot be silently skipped for stdout.
        assert_eq!(
            parse(&args(&["doctor", "--bundle", "out.json", "--json"])),
            Some(Err(
                "doctor: --bundle writes a file and --json prints one; pick either".to_string()
            ))
        );
        assert!(matches!(
            parse(&args(&["config", "validate", "f.json"])),
            Some(Ok(Action::ConfigValidate(_)))
        ));
        assert!(matches!(
            parse(&args(&["config", "effective", "f.json"])),
            Some(Ok(Action::ConfigEffective(_, false)))
        ));
        assert!(matches!(
            parse(&args(&["config", "effective", "f.json", "--json"])),
            Some(Ok(Action::ConfigEffective(_, true)))
        ));
        assert!(
            parse(&args(&["config", "effective"]))
                .expect("action")
                .is_err()
        );
        assert!(parse(&args(&["config", "validate"])).expect("a").is_err());
        assert!(parse(&args(&["config", "print"])).expect("a").is_err());
    }
}
