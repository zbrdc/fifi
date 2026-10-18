mod compare;
mod fixture;
mod run;

use std::{collections::BTreeMap, path::PathBuf};

struct Options {
    cases: PathBuf,
    manifest_sha256: String,
    output: Option<PathBuf>,
    integrity_only: bool,
    negative_control: bool,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let mut values = BTreeMap::new();
        let (mut integrity_only, mut negative_control) = (false, false);
        while let Some(key) = args.next() {
            if key == "--integrity-only" && !integrity_only {
                integrity_only = true;
            } else if key == "--negative-control" && !negative_control {
                negative_control = true;
            } else if matches!(key.as_str(), "--cases" | "--manifest-sha256" | "--output") {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {key}"))?;
                if values.insert(key.clone(), value).is_some() {
                    return Err(format!("duplicate argument {key}"));
                }
            } else {
                return Err(format!("unknown or duplicate argument {key}"));
            }
        }
        let manifest_sha256 = values
            .remove("--manifest-sha256")
            .ok_or("missing --manifest-sha256")?;
        let cases = values.remove("--cases").ok_or("missing --cases")?.into();
        let output = values.remove("--output").map(PathBuf::from);
        if output.is_none() && !integrity_only {
            return Err("missing --output".into());
        }
        if integrity_only && negative_control {
            return Err("--integrity-only excludes --negative-control".into());
        }
        Ok(Self {
            cases,
            manifest_sha256,
            output,
            integrity_only,
            negative_control,
        })
    }
}

fn run() -> Result<i32, String> {
    use std::io::Write;
    let options = Options::parse()?;
    if let Some(output) = &options.output
        && (output.exists() || output.is_symlink())
    {
        return Err("output already exists".into());
    }
    let suite = fixture::Suite::load(&options.cases, &options.manifest_sha256)?;
    if options.integrity_only {
        suite.packed_weights(None)?;
        println!(
            "{}",
            serde_json::json!({"status":"integrity_ok","cases":suite.manifest.cases.len(),"synthetic":suite.manifest.synthetic})
        );
        return Ok(0);
    }
    let report = run::execute(&suite, &options.manifest_sha256, options.negative_control);
    let output = options.output.expect("checked");
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temporary.write_all(&bytes).map_err(|e| e.to_string())?;
    temporary.as_file().sync_all().map_err(|e| e.to_string())?;
    temporary
        .persist_noclobber(&output)
        .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({"status":report.status,"qualified":report.qualified,"synthetic":report.synthetic,"negative_control":report.negative_control,"cases":report.cases.len(),"value_differences":report.value_differences,"bit_differences":report.bit_differences})
    );
    Ok(match report.status {
        "passed" | "negative_control_detected" => 0,
        "mismatch" | "negative_control_missed" => 1,
        _ => 2,
    })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
