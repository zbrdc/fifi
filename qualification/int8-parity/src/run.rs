//! Cases 1, 2, and 5 of decision 0031: native execution and bit comparison.

use serde::Serialize;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::compare::compare;
use crate::{
    compare::{PathResult, decide},
    fixture::Suite,
};

#[derive(Serialize)]
pub struct CaseReport {
    pub id: String,
    pub module: String,
    pub m: u32,
    pub expected_kernel: String,
    pub results: Vec<PathResult>,
}

#[derive(Serialize)]
pub struct Report {
    pub status: &'static str,
    pub qualified: bool,
    pub synthetic: bool,
    pub negative_control: bool,
    pub manifest_sha256: String,
    pub error: Option<String>,
    pub cpu_brand: String,
    pub cases: Vec<CaseReport>,
    pub bit_differences: u64,
    pub value_differences: u64,
    pub compared_values: u64,
    pub differing_modules: Vec<String>,
}

pub fn execute(suite: &Suite, manifest_sha256: &str, negative_control: bool) -> Report {
    let mut report = Report {
        status: "error",
        qualified: false,
        synthetic: suite.manifest.synthetic,
        negative_control,
        manifest_sha256: manifest_sha256.into(),
        error: None,
        cpu_brand: String::new(),
        cases: Vec::new(),
        bit_differences: 0,
        value_differences: 0,
        compared_values: 0,
        differing_modules: Vec::new(),
    };
    match execute_native(suite, negative_control, &mut report) {
        Ok(()) => {
            let verdict = decide(&report.cases, report.bit_differences, negative_control);
            report.differing_modules = verdict.differing_modules;
            report.status = verdict.status;
            report.qualified = verdict.status == "passed" && !report.synthetic;
        }
        Err(error) => report.error = Some(error),
    }
    report
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn execute_native(_: &Suite, _: bool, _: &mut Report) -> Result<(), String> {
    Err("native execution needs an Apple Silicon macOS host".into())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn execute_native(
    suite: &Suite,
    negative_control: bool,
    report: &mut Report,
) -> Result<(), String> {
    use inference_engine::{
        kernels::MlxKernels,
        tensor::{Dtype, Tensor},
    };

    use crate::fixture::{HIDDEN, tensor_bytes};
    if std::env::var("MLX_DISABLE_COMPILE").as_deref() != Ok("1") {
        return Err("MLX_DISABLE_COMPILE=1 is required".into());
    }
    let brand = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .map_err(|e| format!("{e:?}"))?;
    report.cpu_brand = String::from_utf8_lossy(&brand.stdout).trim().to_owned();
    let recorded = suite.manifest.device["cpu_brand"]
        .as_str()
        .unwrap_or_default();
    if report.cpu_brand != recorded && !suite.manifest.synthetic {
        return Err(format!(
            "host CPU {:?} differs from the fixture CPU {recorded:?}",
            report.cpu_brand
        ));
    }
    let mut modified;
    let shard_override = if negative_control {
        modified = suite.shard.clone();
        let name = "language_model.model.layers.0.mlp.gate.scales";
        let target = tensor_bytes(&suite.shard, name)?;
        let offset = target.as_ptr() as usize - suite.shard.as_ptr() as usize;
        modified[offset] ^= 0x01;
        Some(modified.as_slice())
    } else {
        None
    };
    let packed = suite.packed_weights(shard_override)?;
    let mut kernels = MlxKernels::try_new().map_err(|e| format!("{e:?}"))?;
    for case in &suite.manifest.cases {
        let weight = packed.get(&case.module).ok_or("module not loaded")?;
        let act_bytes = tensor_bytes(&suite.cases, &case.activation)?;
        let expected = tensor_bytes(&suite.cases, &case.reference)?;
        let activation =
            Tensor::try_from_host(Dtype::F32, vec![case.m, HIDDEN], act_bytes.to_vec())
                .map_err(|e| format!("{e:?}"))?;
        let mut results = Vec::new();
        let output = kernels
            .dequant_matmul(&activation, weight)
            .map_err(|e| format!("{e:?}"))?;
        results.push(compare(
            "dequant_matmul",
            expected,
            output.host_bytes().ok_or("no host bytes")?,
        ));
        if case.m == 1 {
            let resident = kernels
                .import_packed(weight)
                .map_err(|e| format!("{e:?}"))?;
            let input = kernels
                .import_tensor(&activation)
                .map_err(|e| format!("{e:?}"))?;
            let out = kernels
                .qmm(&input, &resident)
                .map_err(|e| format!("{e:?}"))?;
            let host = kernels.readback(&out).map_err(|e| format!("{e:?}"))?;
            results.push(compare(
                "resident_qmm",
                expected,
                host.host_bytes().ok_or("no host bytes")?,
            ));
        }
        for r in &results {
            report.bit_differences += r.bit_differences;
            report.value_differences += r.value_differences;
            report.compared_values += expected.len() as u64 / 4;
        }
        report.cases.push(CaseReport {
            id: case.id.clone(),
            module: case.module.clone(),
            m: case.m,
            expected_kernel: case.expected_kernel.clone(),
            results,
        });
    }
    Ok(())
}
