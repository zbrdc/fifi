//! Checked GGUF source views staged for Flint CUDA M2.
//! CPU qualification precedes registration with the canonical backend-op caller.

use std::ops::Range;

use crate::{
    EngError,
    weights::{GgufHeader, GgufTensor, GgufType, MetaValue},
};

pub(super) struct MatrixSource {
    pub(super) data: Range<u64>,
    pub(super) scale: Range<u64>,
    pub(super) shape: [usize; 2],
}

pub(super) struct Nvfp4View<'a> {
    pub(super) data: &'a [u8],
    pub(super) tensor_scale: f32,
    pub(super) shape: [usize; 2],
}

impl MatrixSource {
    pub(super) fn bind<'a>(
        &self,
        data: &'a [u8],
        scalar: &[u8],
    ) -> Result<Nvfp4View<'a>, EngError> {
        if data.len() as u64 != self.data.end - self.data.start {
            return Err(invalid(
                "expert payload length differs from its source range",
            ));
        }
        let scalar = <[u8; 4]>::try_from(scalar)
            .map_err(|_| invalid("tensor multiplier must contain exactly four bytes"))?;
        let tensor_scale = f32::from_le_bytes(scalar);
        if !tensor_scale.is_finite() {
            return Err(invalid("tensor multiplier must be finite"));
        }
        Ok(Nvfp4View {
            data,
            tensor_scale,
            shape: self.shape,
        })
    }
}

pub(super) struct ExpertSources {
    pub(super) gate: MatrixSource,
    pub(super) up: MatrixSource,
    pub(super) down: MatrixSource,
}

pub(super) fn expert_sources(
    header: &GgufHeader,
    layer: u32,
    expert: u32,
) -> Result<ExpertSources, EngError> {
    if !(2..49).contains(&layer) || expert >= 256 {
        return Err(EngError::BadRequest(
            "Flint routed expert requires layer 2..49 and expert 0..256".into(),
        ));
    }
    check_architecture(header)?;
    let stem = format!("blk.{layer}.ffn");
    Ok(ExpertSources {
        gate: part(header, &format!("{stem}_gate_exps"), [512, 2048], expert)?,
        up: part(header, &format!("{stem}_up_exps"), [512, 2048], expert)?,
        down: part(header, &format!("{stem}_down_exps"), [2048, 512], expert)?,
    })
}

pub(super) fn matrix_source(
    header: &GgufHeader,
    stem: &str,
    shape: [usize; 2],
) -> Result<MatrixSource, EngError> {
    check_architecture(header)?;
    let weight = required(
        header,
        &format!("{stem}.weight"),
        GgufType::Nvfp4,
        &[shape[1] as u64, shape[0] as u64],
    )?;
    let scale = required(header, &format!("{stem}.scale"), GgufType::F32, &[1])?;
    Ok(MatrixSource {
        data: weight.abs_range(),
        scale: scale.abs_range(),
        shape,
    })
}

/// GGUF stores direct BF16 rows, not the Apple checkpoint's affine INT8 embedding.
pub(super) fn embedding_row(header: &GgufHeader, token: u32) -> Result<Range<u64>, EngError> {
    if token >= 150144 {
        return Err(EngError::BadRequest(
            "Flint embedding token is outside 0..150144".into(),
        ));
    }
    check_architecture(header)?;
    let tensor = required(header, "token_embd.weight", GgufType::Bf16, &[2048, 150144])?;
    selected_range(tensor, token, 4096)
}

fn check_architecture(header: &GgufHeader) -> Result<(), EngError> {
    if !matches!(header.meta("general.architecture"), Some(MetaValue::String(value)) if value == "flint")
    {
        return Err(invalid("general.architecture must be flint"));
    }
    Ok(())
}

fn part(
    header: &GgufHeader,
    stem: &str,
    shape: [usize; 2],
    expert: u32,
) -> Result<MatrixSource, EngError> {
    let weight = required(
        header,
        &format!("{stem}.weight"),
        GgufType::Nvfp4,
        &[shape[1] as u64, shape[0] as u64, 256],
    )?;
    let scales = required(header, &format!("{stem}.scale"), GgufType::F32, &[256])?;
    let part_bytes = weight.byte_len() / 256;
    Ok(MatrixSource {
        data: selected_range(weight, expert, part_bytes)?,
        scale: selected_range(scales, expert, 4)?,
        shape,
    })
}

pub(super) fn required<'a>(
    header: &'a GgufHeader,
    name: &str,
    ty: GgufType,
    dims: &[u64],
) -> Result<&'a GgufTensor, EngError> {
    let tensor = header
        .tensor(name)
        .ok_or_else(|| invalid(&format!("missing {name}")))?;
    if tensor.ggml_type() != ty || tensor.dims() != dims {
        return Err(invalid(&format!("wrong dtype or dimensions for {name}")));
    }
    Ok(tensor)
}

fn selected_range(tensor: &GgufTensor, index: u32, bytes: u64) -> Result<Range<u64>, EngError> {
    let start = u64::from(index)
        .checked_mul(bytes)
        .and_then(|offset| tensor.abs_offset().checked_add(offset))
        .ok_or_else(|| invalid("selected byte offset overflows"))?;
    let end = start
        .checked_add(bytes)
        .filter(|end| *end <= tensor.abs_range().end)
        .ok_or_else(|| invalid("selected range exceeds its tensor"))?;
    Ok(start..end)
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("Flint GGUF source: {message}"))
}

#[cfg(test)]
#[path = "gguf_weights_tests.rs"]
mod tests;
