mod cases;
mod run;

use std::{collections::BTreeMap, path::PathBuf};

struct Options {
    cases: PathBuf,
    manifest_sha256: String,
    output: PathBuf,
    fixture: bool,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let mut values = BTreeMap::new();
        let mut fixture = false;
        while let Some(key) = args.next() {
            if key == "--fixture" && !fixture {
                fixture = true;
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
        let mut required = |key: &str| values.remove(key).ok_or_else(|| format!("missing {key}"));
        let manifest_sha256 = required("--manifest-sha256")?;
        Ok(Self {
            cases: required("--cases")?.into(),
            manifest_sha256,
            output: required("--output")?.into(),
            fixture,
        })
    }
}

fn run() -> Result<i32, String> {
    use std::io::Write;
    let options = Options::parse()?;
    if options.output.exists() || options.output.is_symlink() {
        return Err("output already exists".into());
    }
    let suite = cases::Suite::load(&options.cases, &options.manifest_sha256, options.fixture)?;
    let report = run::execute(&suite, &options.manifest_sha256);
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temporary.write_all(&bytes).map_err(|e| e.to_string())?;
    temporary.as_file().sync_all().map_err(|e| e.to_string())?;
    temporary
        .persist_noclobber(&options.output)
        .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({"status":report.status, "qualified":report.qualified, "fixture":report.fixture, "matmul_calls":report.matmul_calls, "compared_values":report.compared_values, "value_differences":report.value_differences})
    );
    Ok(match report.status {
        "passed" => 0,
        "mismatch" => 1,
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
