//! Resident sigmoid routing; numerical validation happens at the token boundary.
use crate::{
    EngError,
    kernels::{BinaryOp, BoundedIndices, MlxKernels, ReduceOp, UnaryOp},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

type Result<T> = std::result::Result<T, EngError>;
const VOCAB: u32 = 150_144;

pub(super) struct Route {
    pub(super) ids: BoundedIndices,
    pub(super) weights: ExecutionTensor,
    pub(super) health: ExecutionTensor,
}

pub(super) fn select(k: &mut MlxKernels, raw: &ExecutionTensor) -> Result<Route> {
    let &[rows, 256] = raw.shape() else {
        return Err(EngError::BadRequest(
            "Flint router requires [T,256] logits".into(),
        ));
    };
    if rows == 0 || !matches!(raw.dtype(), Dtype::F32 | Dtype::Bf16) {
        return Err(EngError::BadRequest(
            "Flint router requires positive floating rows".into(),
        ));
    }
    let pairs = rows
        .checked_mul(6)
        .ok_or_else(|| EngError::MemoryPlan("Flint routed pair count overflows".into()))?;
    let raw = k.cast(raw, Dtype::F32)?;
    let finite = k.finite_health(&raw)?;
    let scores = k.unary(UnaryOp::Sigmoid, &raw)?;
    // Sort scores, not logits: F32 sigmoid saturation introduces lower-ID ties.
    let negative = k.unary(UnaryOp::Neg, &scores)?;
    let order = k.argsort(&negative, 1)?;
    let ids = k.slice(&order, &[0, 0], &[rows, 6])?;
    let top = k.take_along_axis(&scores, &ids, 1)?;
    let ids = k.reshape(&ids, &[pairs])?;
    let ids = k.bound_indices(&ids, 256)?;
    let mass = k.reduce(ReduceOp::Sum, &top, 1, true)?;
    let weights = k.binary(BinaryOp::Div, &top, &mass)?;
    let weights = k.reshape(&weights, &[rows, 6, 1])?;
    let mass = k.reshape(&mass, &[rows])?;
    // A zero mass may build NaNs; no result becomes valid before read_step rejects it.
    let health = k.concatenate(&[&finite, &mass], 0)?;
    Ok(Route {
        ids,
        weights,
        health,
    })
}

pub(super) fn read_step(
    k: &mut MlxKernels,
    logits: Option<&ExecutionTensor>,
    health: &[ExecutionTensor],
) -> Result<Option<Vec<f32>>> {
    let health_width = health
        .first()
        .and_then(|h| h.shape().first())
        .copied()
        .unwrap_or(2);
    if health_width < 2
        || health
            .iter()
            .any(|h| h.dtype() != Dtype::F32 || h.shape() != [health_width])
    {
        return Err(EngError::BadRequest(
            "Flint router health must be matching F32[1+T] rows".into(),
        ));
    }
    let flat = if let Some(logits) = logits {
        if logits.shape() != [1, VOCAB] {
            return Err(invalid("head returned an unexpected shape"));
        }
        // Preserve the head's BF16 publication; only widen its already-rounded values.
        let logits = k.cast(logits, Dtype::F32)?;
        Some(k.reshape(&logits, &[VOCAB])?)
    } else {
        None
    };
    let mut parts: Vec<_> = flat.iter().collect();
    parts.extend(health.iter());
    if parts.is_empty() {
        return Ok(None);
    }
    let expected = u32::try_from(health.len())
        .ok()
        .and_then(|n| n.checked_mul(health_width))
        .and_then(|n| n.checked_add(if logits.is_some() { VOCAB } else { 0 }))
        .ok_or_else(|| EngError::MemoryPlan("Flint health export size overflows".into()))?;
    let combined = k.concatenate(&parts, 0)?;
    let host = k.readback(&combined)?;
    checked_export(&host, expected, logits.is_some(), health_width)
}

fn checked_export(
    host: &Tensor,
    expected: u32,
    logits: bool,
    health_width: u32,
) -> Result<Option<Vec<f32>>> {
    if host.dtype() != Dtype::F32 || host.shape() != [expected] {
        return Err(invalid("step returned an unexpected dtype or shape"));
    }
    let bytes = host
        .host_bytes()
        .ok_or_else(|| invalid("step readback has no host bytes"))?;
    let mut values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    let count = if logits { VOCAB as usize } else { 0 };
    for row in values[count..].chunks_exact(health_width as usize) {
        if row[0] != 1.0 {
            return Err(invalid("router logits must be finite before sigmoid"));
        }
        if row[1..]
            .iter()
            .any(|mass| !mass.is_finite() || *mass <= 0.0)
        {
            return Err(invalid("router selected mass must be finite and positive"));
        }
    }
    if logits {
        values.truncate(count);
        Ok(Some(values))
    } else {
        Ok(None)
    }
}

fn invalid(message: &str) -> EngError {
    EngError::Backend {
        operation: "flint step",
        message: message.into(),
    }
}

#[cfg(test)]
#[path = "mlx_route_tests.rs"]
mod tests;
