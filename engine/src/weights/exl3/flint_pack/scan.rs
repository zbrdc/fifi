//! Payload passes of the audit: the small tensors of every module, and admitted samples.
//!
//! Only scale vectors, markers, norms, sinks and routers are read here, in budgeted batches
//! through `StreamedCheckpoint::read_selected`. A trellis is read only for a sample matrix.

use std::{collections::BTreeMap, num::FpCategory};

use half::{bf16, f16};

use super::{
    audit::{Admitted, Limits, ScaleCounts},
    schema::{Geometry, Kind, Plan, Spec},
};
use crate::{
    EngError,
    tensor::{Dtype, Tensor},
    weights::{
        StreamedCheckpoint, WeightSource,
        exl3::{Config, MUL1_MARKER, Matrix},
    },
};

/// Names kept for the report; the counts stay exact beyond this.
const DEAD_NAME_CAP: usize = 256;

#[derive(Debug, Default)]
pub(super) struct Scan {
    pub(super) scales: BTreeMap<String, ScaleCounts>,
    pub(super) dead_vectors: Vec<String>,
}

/// Reads every non-trellis, non-bulk tensor of `specs` in batches of at most `batch_bytes`.
pub(super) fn scan_small(
    source: &StreamedCheckpoint,
    specs: &[Spec],
    limits: &Limits,
) -> Result<Scan, EngError> {
    let mut scan = Scan::default();
    let mut batch: Vec<&Spec> = Vec::new();
    let mut batch_bytes = 0u64;
    for spec in specs
        .iter()
        .filter(|s| !matches!(s.kind, Kind::Trellis | Kind::Bulk))
    {
        if !batch.is_empty() && batch_bytes + spec.bytes() > limits.batch_bytes {
            scan.read_batch(source, &mut batch, limits)?;
            batch_bytes = 0;
        }
        batch.push(spec);
        batch_bytes += spec.bytes();
    }
    scan.read_batch(source, &mut batch, limits)?;
    Ok(scan)
}

impl Scan {
    fn read_batch(
        &mut self,
        source: &StreamedCheckpoint,
        batch: &mut Vec<&Spec>,
        limits: &Limits,
    ) -> Result<(), EngError> {
        let keys: Vec<&str> = batch.iter().map(|spec| spec.name.as_str()).collect();
        let resident = source.read_selected(&keys, limits.batch_bytes)?;
        for spec in batch.drain(..) {
            let tensor = resident
                .tensor(&spec.name)
                .ok_or_else(|| invalid(&format!("{}: not read", spec.name)))?;
            match spec.kind {
                Kind::Mul1 => check_marker(spec, &tensor)?,
                _ => self.count_halves(spec, &tensor)?,
            }
        }
        Ok(())
    }

    /// Counts zero and subnormal entries and refuses non-finite ones.
    fn count_halves(&mut self, spec: &Spec, tensor: &Tensor) -> Result<(), EngError> {
        let bytes = tensor
            .host_bytes()
            .ok_or_else(|| invalid(&format!("{}: requires host storage", spec.name)))?;
        let (mut zeros, mut subnormals, mut elements) = (0u64, 0u64, 0u64);
        for word in bytes.as_chunks::<2>().0 {
            let word = u16::from_le_bytes(*word);
            let category = match spec.dtype {
                Dtype::F16 => f16::from_bits(word).classify(),
                _ => bf16::from_bits(word).classify(),
            };
            match category {
                FpCategory::Nan | FpCategory::Infinite => {
                    return Err(invalid(&format!(
                        "{}: contains nonfinite values",
                        spec.name
                    )));
                }
                FpCategory::Zero => zeros += 1,
                FpCategory::Subnormal => subnormals += 1,
                FpCategory::Normal => {}
            }
            elements += 1;
        }
        let key = match spec.kind {
            Kind::Suh => format!("{}.suh", spec.role),
            Kind::Svh => format!("{}.svh", spec.role),
            _ => spec.role.to_string(),
        };
        let counts = self.scales.entry(key).or_default();
        counts.vectors += 1;
        counts.elements += elements;
        counts.zeros += zeros;
        counts.subnormals += subnormals;
        let dead = elements > 0 && zeros == elements;
        counts.zero_vectors += usize::from(dead);
        counts.subnormal_vectors += usize::from(elements > 0 && subnormals == elements);
        let is_scale = matches!(spec.kind, Kind::Suh | Kind::Svh);
        if dead && is_scale && self.dead_vectors.len() < DEAD_NAME_CAP {
            self.dead_vectors.push(spec.name.clone());
        }
        Ok(())
    }
}

/// The stored I32 must hold the unsigned marker word as its bit pattern.
fn check_marker(spec: &Spec, tensor: &Tensor) -> Result<(), EngError> {
    let word = tensor
        .host_bytes()
        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| {
            invalid(&format!(
                "{}: mul1 marker requires four host bytes",
                spec.name
            ))
        })?;
    if word != MUL1_MARKER {
        return Err(invalid(&format!(
            "{}: mul1 marker {word:#010x} is not {MUL1_MARKER:#010x}",
            spec.name
        )));
    }
    Ok(())
}

/// Admits each sample matrix through `Matrix::load` and records what the loader reports.
///
/// The header pass has already fixed shape and rate against the schema; the loader adds its own
/// group checks (marker against the config codebook, scale lengths against the trellis) and
/// proves the trellis payload of the sample can be fetched.
pub(super) fn admit_samples(
    source: &StreamedCheckpoint,
    config: &Config,
    plan: &Plan,
    geometry: &Geometry,
    limits: &Limits,
) -> Result<Vec<Admitted>, EngError> {
    let wanted = geometry.sample();
    let mut admitted = Vec::new();
    for module in plan.modules.iter().filter(|m| wanted.contains(&m.prefix)) {
        let specs = module.specs();
        let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        let resident = source.read_selected(&keys, limits.matrix_bytes)?;
        let matrix = Matrix::load(&resident, &module.prefix, config)?;
        admitted.push(Admitted {
            prefix: module.prefix.clone(),
            shape: matrix.shape(),
            bits_twice: matrix.bitrate_twice(),
            codebook: matrix.codebook(),
        });
    }
    Ok(admitted)
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 Flint pack: {message}"))
}
