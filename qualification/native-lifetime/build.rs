use std::{path::PathBuf, process::Command};

fn main() {
    assert_eq!(std::env::var("CARGO_CFG_TARGET_OS").unwrap(), "macos");
    assert_eq!(std::env::var("CARGO_CFG_TARGET_ARCH").unwrap(), "aarch64");
    println!("cargo:rerun-if-changed=observer.cpp");
    println!("cargo:rerun-if-env-changed=INFERENCE_MLX_SOURCE");
    let source = PathBuf::from(std::env::var_os("INFERENCE_MLX_SOURCE").unwrap());
    let revision = Command::new("git")
        .args(["-C"])
        .arg(&source)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(revision.status.success());
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        "64ea011cb65f14d9ce2737e60db9a4ae91ed7441"
    );
    let changes = Command::new("git")
        .args(["-C"])
        .arg(&source)
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .unwrap();
    assert!(changes.status.success() && changes.stdout.is_empty());
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .compiler("/usr/bin/clang++")
        .include(source)
        .file("observer.cpp")
        .warnings_into_errors(true)
        .compile("inference_lifetime_observer");
}
