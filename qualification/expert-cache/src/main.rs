//! Offline replay of the T0 route trace (S3 plan section 3.5).
//!
//! ```text
//! expert-cache self-test
//! expert-cache simulate --trace <trace.tsv> --out <dir> [--tokens <tokens.tsv>]
//!     [--planning-slots 193] [--t-fixed 0.149] [--bw <name>=<rate>]... [--bw-page-cache <rate>]
//!     [--bw-cold <name>] [--repetition-limit 0.5] [--threads <n>]
//! ```
//!
//! A rate is bytes per second, optionally with the suffix `GiB` (2^30) or `GB` (10^9).

mod cache;
mod pool;
mod report;
mod selftest;
mod sim;
mod trace;

use report::{Bandwidth, Settings};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("self-test") => self_test(),
        Some("simulate") => simulate(&args[1..]),
        _ => Err("usage: expert-cache self-test | simulate --trace <file> --out <dir> ...".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("expert-cache: {message}");
            ExitCode::FAILURE
        }
    }
}

fn self_test() -> Result<(), String> {
    let results = selftest::run_all();
    for (name, result) in &results {
        match result {
            Ok(()) => println!("pass {name}"),
            Err(message) => println!("FAIL {name}: {message}"),
        }
    }
    if results.iter().all(|(_, r)| r.is_ok()) {
        println!("self-test passed: {} cases", results.len());
        Ok(())
    } else {
        Err("self-test failed".into())
    }
}

/// Parses `7GiB`, `5GB`, or a plain number of bytes per second.
fn rate(text: &str) -> Result<f64, String> {
    let (number, scale) = if let Some(n) = text.strip_suffix("GiB") {
        (n, f64::from(1u32 << 30))
    } else if let Some(n) = text.strip_suffix("GB") {
        (n, 1e9)
    } else {
        (text, 1.0)
    };
    match number.parse::<f64>() {
        Ok(v) if v > 0.0 && v.is_finite() => Ok(v * scale),
        _ => Err(format!("rate {text} is not a positive number")),
    }
}

fn simulate(args: &[String]) -> Result<(), String> {
    let mut trace_path = None;
    let mut tokens_path = None;
    let mut out = None;
    let mut settings = Settings {
        planning_slots: 193,
        t_fixed: 0.149,
        bandwidths: Vec::new(),
        page_cache_bandwidth: rate("7GiB")?,
        cold: None,
        repetition_limit: 0.5,
        threads: std::thread::available_parallelism().map_or(1, usize::from),
    };
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let mut value = || {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--trace" => trace_path = Some(PathBuf::from(value()?)),
            "--tokens" => tokens_path = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--planning-slots" => {
                settings.planning_slots = value()?.parse().map_err(|_| "--planning-slots")?;
            }
            "--t-fixed" => settings.t_fixed = value()?.parse().map_err(|_| "--t-fixed")?,
            "--bw" => {
                let v = value()?;
                let (name, r) = v.split_once('=').ok_or("--bw needs <name>=<rate>")?;
                settings.bandwidths.push(Bandwidth {
                    name: name.to_string(),
                    bytes_per_second: rate(r)?,
                });
            }
            "--bw-page-cache" => settings.page_cache_bandwidth = rate(&value()?)?,
            "--bw-cold" => settings.cold = Some(value()?),
            "--repetition-limit" => {
                settings.repetition_limit = value()?.parse().map_err(|_| "--repetition-limit")?;
            }
            "--threads" => settings.threads = value()?.parse().map_err(|_| "--threads")?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if settings.bandwidths.is_empty() {
        // The warm read rates of the P1 profile and C5 (S3 plan section 3.5).
        for (name, r) in [("warm-7GiB", "7GiB"), ("warm-13GiB", "13GiB")] {
            settings.bandwidths.push(Bandwidth {
                name: name.to_string(),
                bytes_per_second: rate(r)?,
            });
        }
    }
    let trace_path = trace_path.ok_or("--trace is required")?;
    let out = out.ok_or("--out is required")?;
    let text = std::fs::read_to_string(&trace_path)
        .map_err(|e| format!("{}: {e}", trace_path.display()))?;
    let trace = trace::Trace::parse(&text)?;
    let tokens = match &tokens_path {
        Some(path) => Some(trace::parse_tokens(
            &std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?,
        )?),
        None => None,
    };
    if settings.planning_slots < trace.top_k || settings.planning_slots > trace.experts {
        return Err("--planning-slots must lie between top_k and the expert count".into());
    }
    let self_test = selftest::run_all();
    let files = report::report(&trace, tokens.as_ref(), &settings, &self_test)?;
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, body) in &files {
        std::fs::write(out.join(name), body).map_err(|e| format!("{name}: {e}"))?;
    }
    for name in ["checks.txt", "summary.txt"] {
        if let Some(body) = files.get(name) {
            print!("{body}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_take_binary_and_decimal_suffixes() {
        assert_eq!(rate("7GiB"), Ok(7.0 * 1_073_741_824.0));
        assert_eq!(rate("5GB"), Ok(5e9));
        assert_eq!(rate("1234.5"), Ok(1234.5));
        assert!(
            rate("-1").is_err() && rate("fast").is_err(),
            "invalid rates"
        );
    }

    #[test]
    fn simulate_writes_every_report_file() {
        let dir = std::env::temp_dir().join(format!("expert-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let trace_path = dir.join("trace.tsv");
        std::fs::write(&trace_path, selftest::to_tsv(&selftest::seeded())).unwrap();
        let out = dir.join("report");
        let args: Vec<String> = [
            "--trace",
            trace_path.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--planning-slots",
            "6",
            "--bw",
            "ssd=1GB",
            "--bw-cold",
            "ssd",
            "--threads",
            "2",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(simulate(&args), Ok(()));
        for name in [
            "results.tsv",
            "layers.tsv",
            "reuse.tsv",
            "distinct.tsv",
            "checks.txt",
            "summary.txt",
        ] {
            assert!(out.join(name).is_file(), "{name} missing");
        }
        let checks = std::fs::read_to_string(out.join("checks.txt")).unwrap();
        assert!(
            checks.contains("t0_p1_rows=pass rows=400 steps=200 layers=2"),
            "{checks}"
        );
        let summary = std::fs::read_to_string(out.join("summary.txt")).unwrap();
        assert!(summary.contains("bw_cold=ssd=1000000000"), "{summary}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
