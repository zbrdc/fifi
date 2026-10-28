use std::path::Path;

use half::{bf16, f16};

use super::super::{FlintResidentLimits, plan::plan_resident};
use super::{FlintExl3Resident, PackedMatrix, PackedRange, packing::Packing};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, plan},
    tensor::Dtype,
    weights::{
        WeightSource,
        exl3::{
            Checkpoint, Matrix,
            flint_pack::{Geometry, Plain},
        },
    },
};

fn cache_enabled(value: Result<String, std::env::VarError>) -> bool {
    crate::tokenizer::env_allows_value(Some(&value.unwrap_or_default()))
}

#[cfg(test)]
pub(super) fn load_cached_fixture(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    context.ensure_usable()?;
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::from_config(&checkpoint.model_config)?;
    super::host::load_cached(context, stream, path, checkpoint, geometry, limits)
}

pub(crate) fn load_flint_resident(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    load_with_prefill(
        context,
        stream,
        path,
        limits,
        crate::family::flint::settings::PREFILL_MAX_ROWS,
    )
}

pub(super) fn load_with_prefill(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
    rows: usize,
) -> Result<FlintExl3Resident, EngError> {
    context.ensure_usable()?;
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::published(&checkpoint.model_config)?;
    if cache_enabled(std::env::var("FIFI_FLINT_CUDA_EXPERT_CACHE")) {
        crate::family::flint::cuda_ffn::device::exl3::check_cache_prefill_rows(
            rows,
            super::super::cache::SLOTS,
        )?;
        return super::host::load_cached(context, stream, path, checkpoint, geometry, limits);
    }
    load_checked(context, stream, path, checkpoint, geometry, limits)
}

#[cfg(test)]
pub(crate) fn load_flint_fixture(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    context.ensure_usable()?;
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::from_config(&checkpoint.model_config)?;
    load_checked(context, stream, path, checkpoint, geometry, limits)
}

fn load_checked(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    checkpoint: Checkpoint,
    geometry: Geometry,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    let roles = plan_resident(path, &checkpoint, &geometry, limits)?;
    let packing = Packing::new(&roles.layout, &limits)?;
    plan::admit(
        &plan::Demand {
            tensors: packing.bytes as u64,
            kv_cache: limits.reserve.kv_cache,
            work: limits.reserve.work,
            headroom: limits.reserve.headroom,
        },
        u64::try_from(context.memory_info()?.free).unwrap_or(u64::MAX),
    )?;
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(roles.layout.modules.len())
        .map_err(|error| {
            EngError::MemoryPlan(format!("Flint EXL3 resident matrix descriptors: {error}"))
        })?;
    let buffer = context.allocate(packing.bytes)?;
    for (module, slots) in roles.layout.modules.iter().zip(&packing.matrices) {
        let specs = module.specs();
        let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        let selected = checkpoint
            .source
            .read_selected(&keys, limits.max_staging_bytes)?;
        let matrix = Matrix::load(&selected, &module.prefix, &checkpoint.config)?;
        let trellis = matrix.trellis().host_bytes().ok_or_else(|| {
            EngError::Checkpoint("Flint EXL3 trellis requires host storage".into())
        })?;
        if trellis.len() != slots[0].bytes || matrix.bias().is_some() {
            return Err(EngError::Checkpoint(
                "Flint EXL3 matrix disagrees with audited placement".into(),
            ));
        }
        buffer.write_at(stream, slots[0].offset, trellis)?;
        for (tensor, &slot) in [matrix.suh(), matrix.svh()].into_iter().zip(&slots[1..]) {
            let raw = tensor.host_bytes().ok_or_else(|| {
                EngError::Checkpoint("Flint EXL3 scales require host storage".into())
            })?;
            write_halves(&buffer, stream, slot, raw, tensor.dtype())?;
        }
        matrices.push(PackedMatrix {
            trellis: slots[0],
            suh: slots[1],
            svh: slots[2],
            bias: None,
            shape: matrix.shape(),
            bits_twice: matrix.bitrate_twice(),
            book: matrix.codebook(),
        });
    }
    for (plain, &slot) in roles.layout.plain.iter().zip(&packing.plain) {
        upload_plain(
            &checkpoint,
            &buffer,
            stream,
            plain,
            slot,
            limits.max_staging_bytes,
        )?;
    }
    Ok(FlintExl3Resident {
        buffer,
        host: None,
        matrices,
        plain: packing.plain,
        roles,
        model_config: checkpoint.model_config,
    })
}

pub(super) fn upload_plain(
    checkpoint: &Checkpoint,
    buffer: &DeviceBuffer,
    stream: &DeviceStream,
    plain: &Plain,
    slot: PackedRange,
    max_staging: u64,
) -> Result<(), EngError> {
    if !plain.bulk {
        let selected = checkpoint
            .source
            .read_selected(&[&plain.name], max_staging)?;
        let tensor = selected.tensor(&plain.name).ok_or_else(|| {
            EngError::Checkpoint(format!("Flint EXL3 missing auxiliary {}", plain.name))
        })?;
        let raw = tensor.host_bytes().ok_or_else(|| {
            EngError::Checkpoint("Flint EXL3 auxiliary requires host storage".into())
        })?;
        return write_halves(buffer, stream, slot, raw, tensor.dtype());
    }
    // The embedding is read one existing leading slice at a time, never staged whole.
    let rows = plain.shape[0];
    let row_bytes = slot.bytes / rows as usize;
    let mut raw = Vec::new();
    raw.try_reserve_exact(row_bytes / 2)
        .map_err(|error| EngError::MemoryPlan(format!("Flint EXL3 embedding row: {error}")))?;
    raw.resize(row_bytes / 2, 0);
    for row in 0..rows {
        checkpoint
            .source
            .read_slice_into(&plain.name, row, &mut raw)?;
        let row_slot = PackedRange {
            offset: slot.offset + row as usize * row_bytes,
            bytes: row_bytes,
        };
        write_halves(buffer, stream, row_slot, &raw, plain.dtype)?;
    }
    Ok(())
}

fn write_halves(
    buffer: &DeviceBuffer,
    stream: &DeviceStream,
    slot: PackedRange,
    raw: &[u8],
    dtype: Dtype,
) -> Result<(), EngError> {
    let converted = convert_halves(raw, dtype)?;
    if converted.len() != slot.bytes {
        return Err(EngError::Checkpoint(
            "Flint EXL3 F32 auxiliary disagrees with audited placement".into(),
        ));
    }
    buffer.write_at(stream, slot.offset, &converted)
}

pub(super) fn convert_halves(raw: &[u8], dtype: Dtype) -> Result<Vec<u8>, EngError> {
    if !matches!(dtype, Dtype::F16 | Dtype::Bf16) || !raw.len().is_multiple_of(2) {
        return Err(EngError::Checkpoint(
            "Flint EXL3 auxiliary requires complete F16/BF16 values".into(),
        ));
    }
    let bytes = raw
        .len()
        .checked_mul(2)
        .ok_or_else(|| EngError::MemoryPlan("Flint EXL3 F32 staging length overflows".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|error| EngError::MemoryPlan(format!("Flint EXL3 F32 staging: {error}")))?;
    for word in raw.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*word);
        let value = if dtype == Dtype::F16 {
            f16::from_bits(bits).to_f32()
        } else {
            bf16::from_bits(bits).to_f32()
        };
        if !value.is_finite() {
            return Err(EngError::Checkpoint(
                "Flint EXL3 auxiliary contains nonfinite values".into(),
            ));
        }
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_flag_is_strictly_opt_in_without_environment_mutation() {
        use std::env::VarError;
        for value in ["", "0", "false", "yes", "invalid"] {
            assert!(!cache_enabled(Ok(value.into())));
        }
        assert!(!cache_enabled(Err(VarError::NotPresent)));
        assert!(!cache_enabled(Err(VarError::NotUnicode(
            std::ffi::OsString::new()
        ))));
        for value in ["1", " true ", "ON", "TrUe"] {
            assert!(cache_enabled(Ok(value.into())));
        }
    }

    #[test]
    fn cache_prefill_refuses_configured_rows_above_active_capacity() {
        use crate::family::flint::cuda_ffn::device::exl3::check_cache_prefill_rows;
        let slots = super::super::super::cache::SLOTS;
        for rows in [1, 32, 37] {
            check_cache_prefill_rows(rows, slots).unwrap();
        }
        for rows in [0, 38, 64, usize::MAX] {
            assert!(
                check_cache_prefill_rows(rows, slots).is_err(),
                "rows={rows}"
            );
        }
    }

    #[test]
    fn raw_f32_conversion_preserves_norms_signed_zero_and_subnormals() {
        for (dtype, bits, expected) in [
            (Dtype::F16, 0x3c00u16, 1f32.to_bits()),
            (Dtype::Bf16, 0x3f80, 1f32.to_bits()),
            (Dtype::F16, 0x8000, (-0f32).to_bits()),
            (Dtype::F16, 0x0001, 0x33800000),
            (Dtype::Bf16, 0x0001, 0x00010000),
        ] {
            assert_eq!(
                convert_halves(&bits.to_le_bytes(), dtype).unwrap(),
                expected.to_le_bytes()
            );
        }
        for (dtype, bits) in [(Dtype::F16, 0x7c00u16), (Dtype::Bf16, 0x7f80)] {
            assert!(matches!(
                convert_halves(&bits.to_le_bytes(), dtype),
                Err(EngError::Checkpoint(_))
            ));
        }
        assert!(convert_halves(&[0], Dtype::F16).is_err());
        assert!(convert_halves(&[0, 0], Dtype::F32).is_err());
    }
}
