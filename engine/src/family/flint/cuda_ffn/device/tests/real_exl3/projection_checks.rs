//! Ideal FP64 matrix projections, not an operand-rounded or full-model oracle.

use super::*;
use crate::{
    kernels::cuda::DeviceStream,
    weights::{
        WeightSource,
        exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
    },
};

pub(super) struct CpuChecks {
    checkpoint: Checkpoint,
    pub(super) case: &'static str,
    pub(super) position: usize,
    pub(super) count: usize,
}

impl CpuChecks {
    pub(super) fn open(pack: &std::path::Path) -> Result<Self, EngError> {
        Ok(Self {
            checkpoint: Checkpoint::open(pack)?,
            case: "",
            position: 0,
            count: 0,
        })
    }

    pub(super) fn check(
        &mut self,
        stream: &DeviceStream,
        prefix: &str,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
        let keys = names.iter().map(String::as_str).collect::<Vec<_>>();
        let selected = self.checkpoint.source.read_selected(&keys, 1 << 30)?;
        let selected_bytes = usize::try_from(selected.resident_bytes())
            .map_err(|_| bad("ideal-FP64 selected payload size overflow"))?;
        let matrix = Matrix::load(&selected, prefix, &self.checkpoint.config)?;
        drop(selected);
        let [k, n] = matrix.shape();
        let working = k
            .checked_mul(n)
            .and_then(|v| v.checked_mul(10))
            .and_then(|v| {
                k.checked_add(n)
                    .and_then(|rows| rows.checked_mul(32))
                    .and_then(|rows| v.checked_add(rows))
            })
            .and_then(|v| v.checked_add(selected_bytes))
            .and_then(|v| v.checked_add(256 << 20))
            .ok_or_else(|| bad("ideal-FP64 projection reference size overflow"))?;
        if working > 4usize << 30 {
            return Err(EngError::MemoryPlan(
                "ideal-FP64 projection reference exceeds 4 GiB host budget".into(),
            ));
        }
        let x = super::decode::values(stream, input.range(0, input.bytes())?)?;
        let actual = super::decode::values(stream, output.range(0, output.bytes())?)?;
        if x.len() != k || actual.len() != n || x.iter().any(|v| !v.is_finite()) {
            return Err(bad("ideal-FP64 projection spans or input values"));
        }
        let x = x.into_iter().map(f64::from).collect::<Vec<_>>();
        let mut expected = vec![0.0; n];
        matrix.project_fp64_into(&x, 1, &mut expected)?;
        let (max_abs, reference_max, relative) = errors(&expected, &actual)?;
        self.count += 1;
        println!(
            "{}",
            serde_json::json!({
                "label":comparison::LABEL,"reference":"ideal-FP64 projection, NOT FP16-operand-rounded",
                "case":self.case,"position":self.position,"matrix":prefix,"shape":[k,n],
                "max_abs_error":max_abs,"reference_max_abs":reference_max,"relative_max_error":relative,
                "relative_max_is_finite_and_defined":relative.is_some(),
                "planned_reference_working_bytes_not_process_peak":working,"threshold":null,
            })
        );
        Ok(())
    }
}

fn errors(expected: &[f64], actual: &[f32]) -> Result<(f64, f64, Option<f64>), EngError> {
    if expected.is_empty()
        || expected.len() != actual.len()
        || expected.iter().any(|v| !v.is_finite())
        || actual.iter().any(|v| !v.is_finite())
    {
        return Err(bad("projection comparison dimensions or nonfinite values"));
    }
    let mut max_abs = 0.0f64;
    let mut reference_max = 0.0f64;
    for (&want, &got) in expected.iter().zip(actual) {
        max_abs = max_abs.max((f64::from(got) - want).abs());
        reference_max = reference_max.max(want.abs());
    }
    let relative = (reference_max != 0.0)
        .then(|| max_abs / reference_max)
        .filter(|v| v.is_finite());
    Ok((max_abs, reference_max, relative))
}

#[test]
fn projection_error_report_has_no_quality_threshold_and_keeps_zero_scale_explicit() {
    assert_eq!(
        errors(&[2.0, -4.0], &[3.0, -4.0]).unwrap(),
        (1.0, 4.0, Some(0.25))
    );
    assert_eq!(errors(&[0.0], &[1.0]).unwrap(), (1.0, 0.0, None));
    assert!(errors(&[0.0], &[]).is_err());
    assert!(errors(&[f64::NAN], &[0.0]).is_err());
    assert!(errors(&[0.0], &[f32::INFINITY]).is_err());
}
