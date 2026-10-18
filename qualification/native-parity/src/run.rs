#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::cases::values;
use crate::cases::{Manifest, Suite};
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Difference {
    index: usize,
    expected: f32,
    actual: f32,
    expected_bits: u32,
    actual_bits: u32,
}

#[derive(Serialize)]
pub(super) struct CaseResult {
    module: String,
    coordinates: Vec<u64>,
    probe: &'static str,
    actual_sha256: String,
    tensor_sha256: std::collections::BTreeMap<&'static str, String>,
    compared_values: usize,
    differences: Vec<Difference>,
}

#[derive(Serialize)]
pub(super) struct Report<'a> {
    provenance: &'a Manifest,
    runner_os: &'static str,
    runner_arch: &'static str,
    pub status: &'static str,
    pub fixture: bool,
    pub qualified: bool,
    pub manifest_sha256: String,
    pub completed_groups: usize,
    pub matmul_calls: usize,
    pub compared_values: usize,
    pub value_differences: usize,
    pub signed_zero_bit_differences: usize,
    pub negative_biases: usize,
    pub cases: Vec<CaseResult>,
    pub error: Option<String>,
}

pub(super) fn execute<'a>(suite: &'a Suite, digest: &str) -> Report<'a> {
    let mut report = Report {
        provenance: &suite.manifest,
        runner_os: std::env::consts::OS,
        runner_arch: std::env::consts::ARCH,
        status: "error",
        fixture: suite.manifest.fixture,
        qualified: false,
        manifest_sha256: digest.into(),
        completed_groups: 0,
        matmul_calls: 0,
        compared_values: 0,
        value_differences: 0,
        signed_zero_bit_differences: 0,
        negative_biases: suite.manifest.negative_biases,
        cases: Vec::new(),
        error: None,
    };
    match execute_native(suite, &mut report) {
        Ok(()) => {
            report.status = if report.value_differences == 0 {
                "passed"
            } else {
                "mismatch"
            };
            report.qualified = !report.fixture && report.value_differences == 0;
        }
        Err(error) => report.error = Some(error),
    }
    report
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn execute_native(suite: &Suite, report: &mut Report<'_>) -> Result<(), String> {
    use inference_engine::kernels::MlxKernels;
    use sha2::{Digest, Sha256};
    let mut kernels = MlxKernels::try_new().map_err(|e| format!("native initialize: {e:?}"))?;
    for case in &suite.cases {
        for (index, probe) in ["basis", "dense"].into_iter().enumerate() {
            let output = kernels
                .dequant_matmul(&suite.activation[index], &case.weight)
                .map_err(|e| format!("{} {probe}: {e:?}", case.module))?;
            let shape = if index == 0 { [64, 1] } else { [1, 1] };
            let actual = values(&output, &shape)?;
            let expected = values(&case.expected[index], &shape)?;
            let mut differences = Vec::new();
            for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
                if expected != actual {
                    differences.push(Difference {
                        index,
                        expected,
                        actual,
                        expected_bits: expected.to_bits(),
                        actual_bits: actual.to_bits(),
                    });
                } else if expected.to_bits() != actual.to_bits() {
                    report.signed_zero_bit_differences += 1;
                }
            }
            report.matmul_calls += 1;
            report.compared_values += actual.len();
            report.value_differences += differences.len();
            let actual_sha256 = format!(
                "{:x}",
                Sha256::digest(output.host_bytes().ok_or("output has no host bytes")?)
            );
            let tensor_sha256 = [
                ("weight", case.weight.payload()),
                ("scales", case.weight.scales()),
                ("biases", case.weight.biases().ok_or("case has no biases")?),
                ("activation", &suite.activation[index]),
                ("reference", &case.expected[index]),
            ]
            .into_iter()
            .map(|(role, tensor)| {
                let bytes = tensor.host_bytes().ok_or("input has no host bytes")?;
                Ok((role, format!("{:x}", Sha256::digest(bytes))))
            })
            .collect::<Result<_, String>>()?;
            report.cases.push(CaseResult {
                actual_sha256,
                tensor_sha256,
                module: case.module.clone(),
                coordinates: case.coordinates.clone(),
                probe,
                compared_values: actual.len(),
                differences,
            });
        }
        report.completed_groups += 1;
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn execute_native(_: &Suite, _: &mut Report<'_>) -> Result<(), String> {
    Err("native parity requires Apple Silicon macOS".into())
}
