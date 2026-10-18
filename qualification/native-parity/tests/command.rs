use std::process::Command;

#[test]
fn requires_an_independent_manifest_digest() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("manifest.json"), b"{}").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .args(["--cases", directory.path().to_str().unwrap(), "--output"])
        .arg(directory.path().join("report.json"))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("manifest-sha256"));
    assert!(!directory.path().join("report.json").exists());
}

#[test]
fn rejects_a_changed_manifest_before_parsing_it() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("manifest.json"), b"untrusted bytes").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .arg("--cases")
        .arg(directory.path())
        .args(["--manifest-sha256", &"0".repeat(64), "--output"])
        .arg(directory.path().join("report.json"))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("manifest digest mismatch"));
    assert!(!directory.path().join("report.json").exists());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn compares_the_pinned_fixture_without_qualifying_real_groups() {
    use sha2::{Digest, Sha256};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = tempfile::tempdir().unwrap();
    let digest = format!(
        "{:x}",
        Sha256::digest(std::fs::read(root.join("manifest.json")).unwrap())
    );
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .arg("--cases")
        .arg(&root)
        .args(["--manifest-sha256", &digest, "--fixture", "--output"])
        .arg(output.path().join("report.json"))
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.path().join("report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "passed");
    assert_eq!(report["qualified"], false);
    assert_eq!(report["fixture"], true);
    assert_eq!(report["completed_groups"], 4);
    assert_eq!(report["matmul_calls"], 8);
    assert_eq!(report["compared_values"], 260);
    assert_eq!(report["value_differences"], 0);
}

fn fixture_copy() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for item in std::fs::read_dir(source).unwrap() {
        let item = item.unwrap();
        std::fs::copy(item.path(), directory.path().join(item.file_name())).unwrap();
    }
    directory
}

fn run_fixture(
    directory: &std::path::Path,
    manifest: &serde_json::Value,
) -> (std::process::Output, Option<serde_json::Value>) {
    use sha2::{Digest, Sha256};
    let data = serde_json::to_vec(manifest).unwrap();
    std::fs::write(directory.join("manifest.json"), &data).unwrap();
    let digest = format!("{:x}", Sha256::digest(&data));
    let output = tempfile::tempdir().unwrap();
    let report_path = output.path().join("report.json");
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .arg("--cases")
        .arg(directory)
        .args(["--manifest-sha256", &digest, "--fixture", "--output"])
        .arg(&report_path)
        .output()
        .unwrap();
    let report = std::fs::read(report_path)
        .ok()
        .map(|bytes| serde_json::from_slice(&bytes).unwrap());
    (result, report)
}

#[test]
fn rejects_duplicate_original_group_identities_before_native_execution() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    manifest["groups"][1] = manifest["groups"][0].clone();
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("duplicate group identity"));
    assert!(report.is_none());
}

#[test]
fn rejects_an_incomplete_group_set_before_native_execution() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    manifest["groups"].as_array_mut().unwrap().pop();
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("group count"));
    assert!(report.is_none());
}

#[test]
fn rejects_a_changed_sample_order_before_native_execution() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    manifest["groups"].as_array_mut().unwrap().swap(0, 1);
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("deterministic selection"));
    assert!(report.is_none());
}

#[test]
fn rejects_inconsistent_provenance_before_native_execution() {
    let directory = fixture_copy();
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    for (pointer, value) in [
        ("/version", serde_json::json!(2)),
        ("/reference_pin", serde_json::json!("0".repeat(64))),
        ("/versions/mlx", serde_json::json!("0.99")),
        ("/source_files/0/sha256", serde_json::json!("0".repeat(64))),
        ("/excluded_int8/0", serde_json::json!("dense")),
        (
            "/groups/0/shards/0",
            serde_json::json!("absent.safetensors"),
        ),
        ("/hardware", serde_json::json!("")),
    ] {
        let mut manifest = original.clone();
        *manifest.pointer_mut(pointer).unwrap() = value;
        let (result, report) = run_fixture(directory.path(), &manifest);
        assert_eq!(result.status.code(), Some(2), "{pointer}");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("provenance"),
            "{pointer}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(report.is_none(), "{pointer}");
    }
}

#[test]
fn rejects_false_negative_bias_coverage_before_native_execution() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    manifest["groups"][0]["negative_bias"] = false.into();
    manifest["negative_biases"] = 3.into();
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("negative bias"));
    assert!(report.is_none());
}

fn rewrite_tensor(
    directory: &std::path::Path,
    asset: &mut serde_json::Value,
    name: &str,
    replacement: &[u8],
) {
    use safetensors::{
        SafeTensors,
        tensor::{TensorView, serialize},
    };
    use sha2::{Digest, Sha256};
    let path = directory.join(asset["path"].as_str().unwrap());
    let original = std::fs::read(&path).unwrap();
    let source = SafeTensors::deserialize(&original).unwrap();
    let views: Vec<_> = source
        .tensors()
        .into_iter()
        .map(|(key, view)| {
            let value = if key == name {
                TensorView::new(view.dtype(), view.shape().to_vec(), replacement).unwrap()
            } else {
                view
            };
            (key, value)
        })
        .collect();
    let bytes = serialize(views, None).unwrap();
    asset["bytes"] = bytes.len().into();
    asset["sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn rejects_changed_probe_values_before_native_execution() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    rewrite_tensor(
        directory.path(),
        &mut manifest["activations"],
        "dense",
        &[0; 256],
    );
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("probe values"));
    assert!(report.is_none());
}

#[cfg(unix)]
#[test]
fn rejects_a_manifest_symlink() {
    use sha2::{Digest, Sha256};
    let directory = fixture_copy();
    let manifest = directory.path().join("manifest.json");
    let digest = format!("{:x}", Sha256::digest(std::fs::read(&manifest).unwrap()));
    std::fs::rename(&manifest, directory.path().join("outside.json")).unwrap();
    std::os::unix::fs::symlink(directory.path().join("outside.json"), &manifest).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .arg("--cases")
        .arg(directory.path())
        .args(["--manifest-sha256", &digest, "--fixture", "--output"])
        .arg(directory.path().join("report.json"))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("regular file"));
    assert!(!directory.path().join("report.json").exists());
}

#[test]
fn rejects_reused_prepared_files_before_native_execution() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    manifest["groups"][1]["weight"] = manifest["groups"][0]["weight"].clone();
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("artifact inventory"));
    assert!(report.is_none());
}

#[test]
fn preserves_provenance_in_a_complete_execution_report() {
    let directory = fixture_copy();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    let (result, report) = run_fixture(directory.path(), &manifest);
    let report = report.unwrap();
    assert_eq!(report["provenance"], manifest);
    assert_eq!(report["qualified"], false);
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert_eq!(result.status.code(), Some(0));
        assert_eq!(
            report["cases"][0]["actual_sha256"].as_str().unwrap().len(),
            64
        );
        use sha2::{Digest, Sha256};
        assert_eq!(
            report["cases"][0]["tensor_sha256"]["weight"],
            format!("{:x}", Sha256::digest([0_u8; 32]))
        );
        assert_eq!(
            report["cases"][0]["tensor_sha256"]["biases"],
            format!("{:x}", Sha256::digest([0x80_u8, 0xbf]))
        );
    } else {
        assert_eq!(result.status.code(), Some(2));
        assert_eq!(report["status"], "error");
        assert_eq!(report["completed_groups"], 0);
        assert!(report["error"].as_str().unwrap().contains("Apple Silicon"));
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn reports_a_finite_mismatch_with_original_group_identity() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    let mut expected: Vec<u8> = [-1_f32; 64].iter().flat_map(|v| v.to_le_bytes()).collect();
    expected[..4].copy_from_slice(&42_f32.to_le_bytes());
    rewrite_tensor(
        directory.path(),
        &mut manifest["groups"][0]["reference"],
        "basis",
        &expected,
    );
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(1));
    let report = report.unwrap();
    assert_eq!(report["status"], "mismatch");
    assert_eq!(report["qualified"], false);
    assert_eq!(report["completed_groups"], 4);
    assert_eq!(report["matmul_calls"], 8);
    assert_eq!(report["compared_values"], 260);
    assert_eq!(report["value_differences"], 1);
    let case = &report["cases"][0];
    assert_eq!(case["module"], "dense");
    assert_eq!(case["coordinates"], serde_json::json!([1, 1]));
    assert_eq!(case["probe"], "basis");
    assert_eq!(case["differences"][0]["expected"], 42.0);
    assert_eq!(case["differences"][0]["actual"], -1.0);
}

#[test]
fn rejects_nonfinite_reference_and_sidecar_values() {
    for (role, name, bytes) in [
        ("reference", "dense", f32::NAN.to_le_bytes().to_vec()),
        ("weight", "scales", vec![0x80, 0x7f]),
    ] {
        let directory = fixture_copy();
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
                .unwrap();
        rewrite_tensor(
            directory.path(),
            &mut manifest["groups"][0][role],
            name,
            &bytes,
        );
        let (result, report) = run_fixture(directory.path(), &manifest);
        assert_eq!(result.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&result.stderr).contains("nonfinite"));
        assert!(report.is_none());
    }
}

#[test]
fn rejects_missing_changed_escaping_or_excessive_artifacts() {
    for mode in ["missing", "changed", "escaping", "excessive"] {
        let directory = fixture_copy();
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
                .unwrap();
        let asset = &mut manifest["groups"][0]["reference"];
        match mode {
            "missing" => {
                std::fs::remove_file(directory.path().join(asset["path"].as_str().unwrap()))
                    .unwrap()
            }
            "changed" => std::fs::write(
                directory.path().join(asset["path"].as_str().unwrap()),
                b"changed",
            )
            .unwrap(),
            "escaping" => asset["path"] = "../outside.safetensors".into(),
            "excessive" => asset["bytes"] = 65537.into(),
            _ => unreachable!(),
        }
        let (result, report) = run_fixture(directory.path(), &manifest);
        assert_eq!(result.status.code(), Some(2), "{mode}");
        assert!(report.is_none(), "{mode}");
    }
}

#[test]
fn never_overwrites_an_existing_report() {
    let directory = fixture_copy();
    let destination = directory.path().join("previous.json");
    std::fs::write(&destination, b"previous result").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .arg("--cases")
        .arg(directory.path())
        .args(["--manifest-sha256", &"0".repeat(64), "--output"])
        .arg(&destination)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("output already exists"));
    assert_eq!(std::fs::read(destination).unwrap(), b"previous result");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn accepts_signed_zero_without_counting_it_as_a_value_difference() {
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    rewrite_tensor(
        directory.path(),
        &mut manifest["groups"][0]["weight"],
        "biases",
        &[0, 0],
    );
    let zeros: Vec<u8> = [-0.0_f32; 64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    rewrite_tensor(
        directory.path(),
        &mut manifest["groups"][0]["reference"],
        "basis",
        &zeros,
    );
    rewrite_tensor(
        directory.path(),
        &mut manifest["groups"][0]["reference"],
        "dense",
        &(-0.0_f32).to_le_bytes(),
    );
    manifest["groups"][0]["negative_bias"] = false.into();
    manifest["negative_biases"] = 3.into();
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report.unwrap();
    assert_eq!(report["value_differences"], 0);
    assert!(report["signed_zero_bit_differences"].as_u64().unwrap() > 0);
    assert_eq!(report["qualified"], false);
}

#[test]
fn rejects_fixture_identity_without_explicit_fixture_mode() {
    use sha2::{Digest, Sha256};
    let directory = fixture_copy();
    let digest = format!(
        "{:x}",
        Sha256::digest(std::fs::read(directory.path().join("manifest.json")).unwrap())
    );
    let result = Command::new(env!("CARGO_BIN_EXE_native-parity-qualification"))
        .arg("--cases")
        .arg(directory.path())
        .args(["--manifest-sha256", &digest, "--output"])
        .arg(directory.path().join("report.json"))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("fixture mode"));
    assert!(!directory.path().join("report.json").exists());
}

#[test]
fn rejects_a_reference_with_a_missing_probe() {
    use safetensors::{SafeTensors, tensor::serialize};
    use sha2::{Digest, Sha256};
    let directory = fixture_copy();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("manifest.json")).unwrap())
            .unwrap();
    let asset = &mut manifest["groups"][0]["reference"];
    let path = directory.path().join(asset["path"].as_str().unwrap());
    let original = std::fs::read(&path).unwrap();
    let source = SafeTensors::deserialize(&original).unwrap();
    let bytes = serialize(
        source
            .tensors()
            .into_iter()
            .filter(|(name, _)| name != "dense"),
        None,
    )
    .unwrap();
    asset["bytes"] = bytes.len().into();
    asset["sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    std::fs::write(path, bytes).unwrap();
    let (result, report) = run_fixture(directory.path(), &manifest);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("missing tensor dense"));
    assert!(report.is_none());
}
