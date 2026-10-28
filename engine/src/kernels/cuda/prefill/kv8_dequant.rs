//! Checked KV8-prefix expansion into the time-major FP16 attention staging cache.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::kv8_dequant::{
    Kv8DequantInts, Kv8DequantRaw, kv8_dequant_launch,
};

const MAX_CAPACITY: usize = (1 << 30) - 1;

pub(crate) fn staging_bytes(
    capacity: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<usize, EngError> {
    capacity
        .checked_mul(n_kv_head)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(4))
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| {
            EngError::BadRequest("kv8_dequant: staging size is zero or overflows".into())
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Kv8DequantShape {
    pub(crate) capacity: usize,
    pub(crate) rows: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
}

pub(crate) struct Kv8DequantCall<'a> {
    pub(crate) shape: Kv8DequantShape,
    pub(crate) codes: DeviceRange<'a>,
    pub(crate) scales: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Kv8DequantNeeds {
    pub(crate) codes: usize,
    pub(crate) scales: usize,
    pub(crate) out: usize,
}

fn rule(detail: &str) -> EngError {
    EngError::BadRequest(format!("kv8_dequant: {detail}"))
}

fn check_shape(s: &Kv8DequantShape) -> Result<Kv8DequantInts, EngError> {
    if !(1..=MAX_CAPACITY).contains(&s.capacity) || s.rows == 0 || s.rows > s.capacity {
        return Err(rule("rows must lie within the positive cache capacity"));
    }
    if !(1..=64).contains(&s.n_kv_head) {
        return Err(rule("n_kv_head must be 1..=64"));
    }
    if !(32..=1024).contains(&s.head_dim) || !s.head_dim.is_multiple_of(32) {
        return Err(rule("head_dim must be a multiple of 32 in 32..=1024"));
    }
    // The preceding bounds put every value below i32::MAX.
    Ok(Kv8DequantInts {
        capacity: s.capacity as i32,
        rows: s.rows as i32,
        n_kv_head: s.n_kv_head as i32,
        head_dim: s.head_dim as i32,
    })
}

pub(crate) fn kv8_dequant_needs(s: &Kv8DequantShape) -> Option<Kv8DequantNeeds> {
    check_shape(s).ok()?;
    let codes = staging_bytes(s.capacity, s.n_kv_head, s.head_dim).ok()? / 4;
    let scales = (codes / 32).checked_mul(2)?;
    let out = staging_bytes(s.rows, s.n_kv_head, s.head_dim).ok()? / 2;
    Some(Kv8DequantNeeds { codes, scales, out })
}

fn check_source_origin(s: &Kv8DequantShape, origin: usize) -> Result<i32, EngError> {
    check_shape(s)?;
    if origin
        .checked_add(s.rows)
        .is_none_or(|end| end > s.capacity)
    {
        return Err(rule("source origin plus rows exceeds cache capacity"));
    }
    Ok(origin as i32)
}

impl CudaContext {
    /// Expands only the active prefix; source heads retain the full capacity stride.
    /// No allocation, synchronization or host transfer occurs in this launch.
    pub(crate) fn kv8_dequant(
        &self,
        stream: &DeviceStream,
        call: Kv8DequantCall<'_>,
    ) -> Result<(), EngError> {
        self.kv8_dequant_segment(stream, call, 0)
    }

    /// Expands an absolute source segment into stage row zero, retaining the source stride.
    /// Like the prefix launch, this allocates nothing and never synchronizes.
    pub(crate) fn kv8_dequant_segment(
        &self,
        stream: &DeviceStream,
        call: Kv8DequantCall<'_>,
        source_row_origin: usize,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let origin = check_source_origin(&call.shape, source_row_origin)?;
        let codes = self.raw_range(&call.codes)?;
        let scales = self.raw_range(&call.scales)?;
        let out = self.raw_range(&call.out)?;
        let ints = check_shape(&call.shape)?;
        let need = kv8_dequant_needs(&call.shape).ok_or_else(|| rule("byte extents overflow"))?;
        require_fit("codes", &call.codes, need.codes, 4)?;
        require_fit("scales", &call.scales, need.scales, 4)?;
        require_fit("out", &call.out, need.out, 16)?;
        require_apart(
            &Claim::of("out", &call.out, need.out),
            &Claim::of("codes", &call.codes, need.codes),
        )?;
        require_apart(
            &Claim::of("out", &call.out, need.out),
            &Claim::of("scales", &call.scales, need.scales),
        )?;
        self.launch_on(stream, move |raw| {
            kv8_dequant_launch(
                raw,
                Kv8DequantRaw {
                    codes,
                    scales,
                    out,
                    ints,
                },
                origin,
            )
        })
    }
}

#[cfg(test)]
mod tests;
