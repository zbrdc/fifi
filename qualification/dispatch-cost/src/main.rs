//! Dispatch cost measurement command. It is a measurement command, not a cargo test seam.
#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]

mod host;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native;
mod plan;
mod stats;
mod synth;

use std::{path::PathBuf, process::ExitCode};

pub struct Options {
    pub layers: Vec<usize>,
    pub hidden: u32,
    pub inter: u32,
    pub seq: u32,
    pub reps: usize,
    pub warmup: usize,
    pub output: Option<PathBuf>,
    pub plan_only: bool,
}

fn parse() -> Result<Options, String> {
    let mut options = Options {
        layers: vec![1, 8],
        hidden: 1024,
        inter: 2048,
        seq: 256,
        reps: 50,
        warmup: 5,
        output: None,
        plan_only: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--plan" {
            options.plan_only = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        let number = |text: &str| text.parse::<u32>().map_err(|e| format!("{key}: {e}"));
        match key.as_str() {
            "--layers" => {
                options.layers = value
                    .split(',')
                    .map(|part| part.parse::<usize>().map_err(|e| format!("{key}: {e}")))
                    .collect::<Result<_, _>>()?;
                if options.layers.is_empty() || options.layers.contains(&0) {
                    return Err("--layers needs nonzero counts".into());
                }
            }
            "--hidden" => options.hidden = number(&value)?,
            "--inter" => options.inter = number(&value)?,
            "--seq" => options.seq = number(&value)?,
            "--reps" => options.reps = number(&value)? as usize,
            "--warmup" => options.warmup = number(&value)? as usize,
            "--output" => options.output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {key}")),
        }
    }
    if options.reps == 0 {
        return Err("--reps must be nonzero".into());
    }
    Ok(options)
}

fn main() -> ExitCode {
    let options = match parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("dispatch-cost: {message}");
            return ExitCode::from(2);
        }
    };
    if options.plan_only {
        return match plan::Config::new(options.hidden, options.inter, options.seq) {
            Ok(cfg) => {
                for line in cfg.describe() {
                    println!("{line}");
                }
                for &count in &options.layers {
                    println!("layers {count}: {} calls", cfg.calls_per_chain(count));
                }
                ExitCode::SUCCESS
            }
            Err(message) => {
                eprintln!("dispatch-cost: {message}");
                ExitCode::from(2)
            }
        };
    }
    measure(&options)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn measure(options: &Options) -> ExitCode {
    match native::run(options) {
        Ok(json) => {
            if let Some(path) = &options.output
                && let Err(error) = std::fs::write(path, &json)
            {
                eprintln!("dispatch-cost: write failed: {error}");
                return ExitCode::from(1);
            }
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("dispatch-cost: {message}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn measure(_options: &Options) -> ExitCode {
    eprintln!("dispatch-cost: the native measurement needs Apple Silicon; use --plan on this host");
    ExitCode::from(3)
}
