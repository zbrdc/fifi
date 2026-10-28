//! The `enginesrv` entry point. All behavior lives in `inference_engine::server`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match inference_engine::server::run_cli(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("enginesrv: {error}");
            ExitCode::FAILURE
        }
    }
}
