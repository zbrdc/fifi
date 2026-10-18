// Records how this binary was built (printed by `cuda-rd-run build-info`) and which commit it was built from.
// Cargo sets PROFILE, OPT_LEVEL and DEBUG for build scripts. The engine's build script (nvcc through cc) sees the same
// values for the same profile, but `check_release_build.py` does not rely on that: it also reads the cubins.
use std::process::Command;

/// The full 40-character commit, read with `safe.directory` opened so a root build of a host-owned worktree works.
fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(["-c", "safe.directory=*"])
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn commit() -> String {
    // RD-10a 16c F1: the build image has no git, so run_stage1.sh passes the host's commit in through the
    // environment (run-cpu.sh forwards only its listed variables). Unset keeps the git fallback for host builds.
    println!("cargo:rerun-if-env-changed=CUDA_RD_COMMIT");
    match std::env::var("CUDA_RD_COMMIT") {
        Ok(v) if v.len() == 40 && v.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) => v,
        Ok(v) => panic!(
            "CUDA_RD_COMMIT must be 40 lowercase hex characters (the host's git rev-parse HEAD), got '{v}'"
        ),
        Err(std::env::VarError::NotPresent) => git(&["rev-parse", "HEAD"]),
        Err(e) => panic!("CUDA_RD_COMMIT is set but not valid Unicode: {e}"),
    }
}

fn main() {
    for name in ["PROFILE", "OPT_LEVEL", "DEBUG"] {
        let value = std::env::var(name).unwrap_or_else(|_| "unset".to_owned());
        println!("cargo:rustc-env=CUDA_RD_BUILD_{name}={value}");
    }
    println!("cargo:rustc-env=CUDA_RD_COMMIT={}", commit());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../engine");
}
