//! `cuda-rd-run`: the qualification binary of the stage 1 gate (RD-10). Design guidance only, no code copied.
//!
//! Subcommands: `build-info` (prints the build record as JSON; touches no CUDA call) and `run` (flags in `RD-10a.md`).
//! Exit codes: 0 done, 2 STOP (bad arguments, unreadable input, not a release build), 3 REFUSED (the gate refused the run,
//! for example not enough free memory after the load; the raw file is still written).
use inference_engine::family::qwen36::cuda::gate::{GateSettings, run_step_gate};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const FLAGS: [&str; 14] = [
    "--layout",
    "--kv-mode",
    "--tiered",
    "--gpu-name",
    "--driver",
    "--mode",
    "--prompts",
    "--runs",
    "--capture",
    "--gguf",
    "--gguf-sha256",
    "--oracle",
    "--out",
    "--deadline-secs",
];

fn build_info() -> Value {
    json!({
        "profile": if env!("CUDA_RD_BUILD_PROFILE") == "release" && !cfg!(debug_assertions) { "release" } else { "debug" },
        "opt_level": env!("CUDA_RD_BUILD_OPT_LEVEL"),
        "debug": env!("CUDA_RD_BUILD_DEBUG") != "false",
        "debug_assertions": cfg!(debug_assertions),
        "commit": env!("CUDA_RD_COMMIT"),
    })
}

fn release_refusal(info: &Value) -> Option<String> {
    let bad = info["profile"] != "release"
        || info["opt_level"] == "0"
        || info["opt_level"] == "unset"
        || info["debug"] != false
        || info["debug_assertions"] != false;
    bad.then(|| format!("not a release build: {info}"))
}

/// `--key value` pairs. Unknown, empty and duplicate keys are refused.
fn parse_flags(args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(key) = it.next() {
        if !FLAGS.contains(&key.as_str()) {
            return Err(format!("unknown argument {key}"));
        }
        let value = it.next().ok_or(format!("missing value for {key}"))?.clone();
        if value.is_empty() || out.insert(key.clone(), value).is_some() {
            return Err(format!("empty or duplicate {key}"));
        }
    }
    Ok(out)
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// RD-10a 16c F2: the output file's parent must exist and be writable, checked BEFORE the gate runs (a gate run is a
/// paid GPU hold; a bad --out used to surface only as the raw io error after it). Probes with a .tmp file named from
/// the pid, created and removed again.
fn check_out_writable(out: &Path) -> Result<(), String> {
    let parent = out
        .parent()
        .ok_or_else(|| format!("the output path {} has no parent directory", out.display()))?;
    let probe = parent.join(format!(".cuda-rd-run-{}.tmp", std::process::id()));
    std::fs::write(&probe, b"")
        .map_err(|e| format!("cannot write in {}: {e}", parent.display()))?;
    std::fs::remove_file(&probe).map_err(|e| format!("cannot remove {}: {e}", probe.display()))
}

fn prompt_ids(oracle: &Path, name: &str) -> Result<Vec<u32>, String> {
    let text = String::from_utf8(read(&oracle.join(format!("prompts/{name}.ids")))?)
        .map_err(|e| e.to_string())?;
    text.split_whitespace()
        .map(|t| t.parse::<u32>().map_err(|e| format!("{name}.ids: {e}")))
        .collect()
}

/// The reference stream: the `token` of every step of the frozen cpu/clean run, and the SHA-256 of that file.
fn forced_stream(oracle: &Path, name: &str) -> Result<(Vec<u32>, String), String> {
    let bytes = read(&oracle.join(format!("cpu/clean/{name}/run.json")))?;
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{name} run.json: {e}"))?;
    let steps = doc["steps"].as_array().ok_or("run.json has no steps")?;
    let toks = steps
        .iter()
        .map(|s| {
            let t = s["token"].as_u64().ok_or("step without token")?;
            u32::try_from(t).map_err(|_| format!("{name} run.json: token {t} does not fit u32"))
        })
        .collect::<Result<_, _>>()?;
    Ok((toks, format!("{:x}", Sha256::digest(&bytes))))
}

/// The settings text the gate parses (`GateSettings: FromStr`); the gate types keep private fields.
fn settings_text(f: &BTreeMap<String, String>) -> Result<String, String> {
    let get = |k: &str| f.get(k).ok_or(format!("missing {k}"));
    let oracle = PathBuf::from(get("--oracle")?);
    let names: Vec<String> = get("--prompts")?.split(',').map(str::to_owned).collect();
    let forced_mode = match get("--mode")?.as_str() {
        "forced" => true,
        "free" => false,
        m => return Err(format!("bad --mode {m}")),
    };
    let capture = match get("--capture")?.as_str() {
        "graph" => true,
        "eager" => false,
        c => return Err(format!("bad --capture {c}")),
    };
    let (mut prompts, mut forced, mut shas) = (vec![], vec![], vec![]);
    for n in &names {
        prompts.push(prompt_ids(&oracle, n)?);
        if forced_mode {
            let (toks, sha) = forced_stream(&oracle, n)?;
            forced.push(toks);
            shas.push(sha);
        }
    }
    let number = |k: &str, default: u32| {
        get(k).map_or(Ok(default), |v| {
            v.parse::<u32>().map_err(|e| format!("{k}: {e}"))
        })
    };
    let mut settings = json!({
        "gguf": get("--gguf")?, "gguf_sha256": get("--gguf-sha256")?, "names": names, "prompts": prompts,
        "forced": forced, "forced_stream_sha256": shas, "gen_tokens": 128,
        "runs": get("--runs")?.parse::<u32>().map_err(|e| format!("--runs: {e}"))?,
        "capture": capture, "kv_mode": "fp16", "deadline_secs": number("--deadline-secs", 0)?,
    });
    // MATVEC-PAR S3b: the optional LOAD layout of the gate settings ("tiled" or "native");
    // absent keeps the gate's default (tiled).
    if let Some(layout) = f.get("--layout") {
        settings["layout"] = json!(layout);
    }
    // KV8-B B4: the optional `kv_mode` member of the gate settings ("fp16" or "int8g32"); absent
    // keeps the fp16 control already in the text. A bad value is refused naming the flag; the
    // gate checks the member itself, including its refusal with the Tiled layout (`gate.rs`).
    if let Some(kv_mode) = f.get("--kv-mode") {
        match kv_mode.as_str() {
            "fp16" | "int8g32" => settings["kv_mode"] = json!(kv_mode),
            other => return Err(format!("bad --kv-mode {other}")),
        }
    }
    // OFF-2: the optional `tiered` member of the gate settings (`{"slots": n, "threads": n, "timeout_ms": n,
    // "preload": bool}`), passed through as a JSON object; absent keeps the plain native step. The gate checks the
    // member itself (`gate.rs` tiered_member); here it only has to be JSON.
    if let Some(tiered) = f.get("--tiered") {
        settings["tiered"] = serde_json::from_str(tiered).map_err(|e| format!("--tiered: {e}"))?;
    }
    Ok(settings.to_string())
}

/// Adds the identities only the binary knows: its own SHA-256, commit, profile, process and start time.
fn add_identities(
    raw: &mut Value,
    flags: &BTreeMap<String, String>,
    started_unix_ns: u128,
) -> Result<(), String> {
    let (gguf_sha256, gpu, driver) = (
        &flags["--gguf-sha256"],
        &flags["--gpu-name"],
        &flags["--driver"],
    );
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let info = build_info();
    let id = raw
        .as_object_mut()
        .ok_or("the gate wrote a non-object root")?
        .get_mut("identities")
        .and_then(Value::as_object_mut)
        .ok_or("the gate wrote no identities")?;
    id.insert(
        "exe_sha256".into(),
        json!(format!("{:x}", Sha256::digest(read(&exe)?))),
    );
    id.insert("engine_commit".into(), info["commit"].clone());
    id.insert("gguf_sha256".into(), json!(gguf_sha256));
    id.insert("gpu".into(), json!(gpu));
    id.insert("driver".into(), json!(driver));
    for k in ["profile", "opt_level", "debug"] {
        id.insert(k.into(), info[k].clone());
    }
    id.insert("pid".into(), json!(std::process::id()));
    id.insert("started_unix_ns".into(), json!(started_unix_ns as u64));
    Ok(())
}

fn run(args: &[String]) -> Result<u8, String> {
    if let Some(why) = release_refusal(&build_info()) {
        return Err(why);
    }
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let flags = parse_flags(args)?;
    for k in ["--gpu-name", "--driver", "--out"] {
        flags.get(k).ok_or(format!("missing {k}"))?;
    }
    check_out_writable(Path::new(&flags["--out"]))?;
    let s: GateSettings = settings_text(&flags)?
        .parse()
        .map_err(|e| format!("gate: {e:?}"))?;
    let report = run_step_gate(&s).map_err(|e| format!("gate: {e:?}"))?;
    let mut raw: Value = serde_json::from_str(&report.to_string())
        .map_err(|e| format!("gate wrote invalid JSON: {e}"))?;
    add_identities(&mut raw, &flags, started)?;
    let out = PathBuf::from(&flags["--out"]);
    let tmp = out.with_extension("json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_string(&raw).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &out).map_err(|e| e.to_string())?;
    Ok(if raw["refused"].is_null() { 0 } else { 3 })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build-info") if args.len() == 1 => {
            println!("{}", build_info());
            ExitCode::SUCCESS
        }
        Some("run") => match run(&args[1..]) {
            Ok(code) => ExitCode::from(code),
            Err(e) => {
                println!("STOP: {e}");
                ExitCode::from(2)
            }
        },
        _ => {
            println!("STOP: usage: cuda-rd-run build-info | run --mode ... (see RD-10a.md)");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag map of one minimal free-mode call against a real one-prompt oracle directory
    /// (the settings reader loads the prompt ids from it), with or without the layout flag.
    fn flags(layout: Option<&str>) -> BTreeMap<String, String> {
        let dir = std::env::temp_dir().join("cuda-rd-run-layout-test");
        std::fs::create_dir_all(dir.join("prompts")).expect("the oracle dir");
        std::fs::write(dir.join("prompts/P1.ids"), "7 8 9").expect("the prompt ids");
        let mut f: BTreeMap<String, String> = [
            ("--gguf", "/m/q.gguf"),
            ("--gguf-sha256", "ab"),
            ("--oracle", dir.to_str().expect("utf8")),
            ("--prompts", "P1"),
            ("--mode", "free"),
            ("--capture", "eager"),
            ("--runs", "1"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        if let Some(layout) = layout {
            f.insert("--layout".to_owned(), layout.to_owned());
        }
        f
    }

    /// settings_text without the flag carries no layout key (the gate's tiled default), and
    /// with it the member is passed through verbatim (the value itself is the gate's to check).
    #[test]
    fn the_layout_flag_becomes_the_optional_settings_member() {
        let plain: serde_json::Value =
            serde_json::from_str(&settings_text(&flags(None)).expect("settings")).expect("json");
        assert!(plain.get("layout").is_none());
        let native: serde_json::Value =
            serde_json::from_str(&settings_text(&flags(Some("native"))).expect("settings"))
                .expect("json");
        assert_eq!(native["layout"], "native");
    }

    /// `--tiered` becomes the `tiered` settings member verbatim, the gate's own reader accepts it,
    /// the call without the flag carries no such key, and bad JSON is refused naming the flag.
    #[test]
    fn the_tiered_flag_becomes_the_optional_settings_member() {
        let mut f = flags(Some("native"));
        let plain: serde_json::Value =
            serde_json::from_str(&settings_text(&f).expect("settings")).expect("json");
        assert!(
            plain.get("tiered").is_none(),
            "tiered is never on without the flag"
        );
        f.insert(
            "--tiered".to_owned(),
            r#"{"slots": 40, "threads": 6, "preload": true}"#.to_owned(),
        );
        let text = settings_text(&f).expect("settings");
        let on: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(on["tiered"]["slots"], 40);
        assert_eq!(on["tiered"]["preload"], true);
        text.parse::<GateSettings>()
            .expect("the gate accepts the member");
        f.insert("--tiered".to_owned(), "{not json".to_owned());
        let err = settings_text(&f).expect_err("bad JSON");
        assert!(err.starts_with("--tiered:"), "{err}");
        assert!(parse_flags(&["--tiered".to_owned(), "{}".to_owned()]).is_ok());
    }

    /// `--kv-mode` becomes the `kv_mode` settings member: absent keeps the fp16 control,
    /// `int8g32` passes through and the text parses as GateSettings, any other value is
    /// refused naming the flag, and int8g32 with the Tiled layout is refused by the gate's
    /// own parse.
    #[test]
    fn the_kv_mode_flag_becomes_the_settings_member() {
        let plain: serde_json::Value =
            serde_json::from_str(&settings_text(&flags(None)).expect("settings")).expect("json");
        assert_eq!(plain["kv_mode"], "fp16");
        let mut f = flags(None);
        f.insert("--kv-mode".to_owned(), "int8g32".to_owned());
        let text = settings_text(&f).expect("settings");
        let q8: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(q8["kv_mode"], "int8g32");
        text.parse::<GateSettings>()
            .expect("the gate accepts int8g32 on its native default layout");
        f.insert("--layout".to_owned(), "tiled".to_owned());
        let tiled = settings_text(&f).expect("settings");
        assert!(
            tiled.parse::<GateSettings>().is_err(),
            "the gate refuses the Tiled Int8G32 pair"
        );
        f.insert("--kv-mode".to_owned(), "int8".to_owned());
        let err = settings_text(&f).expect_err("bad kv mode");
        assert!(err.starts_with("bad --kv-mode"), "{err}");
        assert!(parse_flags(&["--kv-mode".to_owned(), "fp16".to_owned()]).is_ok());
    }
}
