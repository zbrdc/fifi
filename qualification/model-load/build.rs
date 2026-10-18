use std::process::Command;

fn git(args: &[&str]) -> String {
    git_optional(args).expect("Git source identity failed")
}

fn git_optional(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn main() {
    println!(
        "cargo:rustc-env=MODEL_SOURCE_COMMIT={}",
        git(&["rev-parse", "HEAD"])
    );
    println!(
        "cargo:rustc-env=MODEL_SOURCE_DIRTY={}",
        !git(&["status", "--porcelain"]).is_empty()
    );
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=tests");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=../../engine");
    println!(
        "cargo:rerun-if-changed={}",
        git(&["rev-parse", "--git-path", "HEAD"])
    );
    if let Some(reference) = git_optional(&["symbolic-ref", "HEAD"]) {
        println!(
            "cargo:rerun-if-changed={}",
            git(&["rev-parse", "--git-path", &reference])
        );
    }
}
