//! Contiguous dense NVFP4 MMQ over original GGUF blocks and D4 q8_1 activations.
//! Setup owns immutable FP32 column factors, including stream-k tail padding. Enqueue
//! borrows every payload, allocates nothing, and applies each factor once at writeback.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "explicit opt-in pending dense caller qualification"
    )
)]

use super::mmq_dense::{DenseInts, DenseNeeds, check_dense_mmq_rules};
use super::mmq_quant::{MmqPacking, MmqQuantizeCall};
use super::mmq_size::{MmqType, q8_1_bytes};
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::mmq_dense::{MmqDenseRaw, mmq_nvfp4_dense_launch};

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("mmq_nvfp4: {message}"))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Nvfp4DenseShape {
    pub(crate) ints: DenseInts,
    pub(crate) needs: DenseNeeds,
    pub(crate) factor_bytes: usize,
}

impl Nvfp4DenseShape {
    pub(crate) fn new(k: usize, n: usize, m: usize) -> Result<Self, EngError> {
        let ints = check_dense_mmq_rules(MmqType::Nvfp4, k, n, m, n, 0)?;
        let blocks = n
            .checked_mul(k / 64)
            .ok_or_else(|| bad("weight block count overflow"))?;
        if blocks > i32::MAX as usize {
            return Err(bad("weight blocks exceed native signed indexing"));
        }
        let w = blocks.checked_mul(36);
        let dst = m
            .checked_mul(n)
            .and_then(|elements| elements.checked_mul(4));
        Ok(Self {
            ints,
            needs: DenseNeeds {
                w: w.ok_or_else(|| bad("weight size overflow"))?,
                y: q8_1_bytes(k, m)?,
                dst: dst.ok_or_else(|| bad("destination size overflow"))?,
            },
            // Stream-k writeback reads a whole J tile, including columns past M.
            factor_bytes: m
                .checked_add(super::mmq_size::J_MAX - 1)
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| bad("factor size overflow"))?,
        })
    }

    fn factors(self, factor: f32) -> Result<Vec<u8>, EngError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(bad("projection factor must be finite and positive"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.factor_bytes)
            .map_err(|_| bad("factor allocation failed"))?;
        for _ in 0..self.factor_bytes / 4 {
            bytes.extend_from_slice(&factor.to_ne_bytes());
        }
        Ok(bytes)
    }
}

pub(crate) struct Nvfp4DensePlan<'a> {
    shape: Nvfp4DenseShape,
    weight: DeviceRange<'a>,
    factors: DeviceBuffer,
}

pub(crate) struct Nvfp4DenseCall<'a> {
    pub(crate) plan: &'a Nvfp4DensePlan<'a>,
    pub(crate) q8_1: DeviceRange<'a>,
    pub(crate) dst: DeviceRange<'a>,
    pub(crate) fixup: DeviceRange<'a>,
}

impl CudaContext {
    /// Setup only: `shape` is [K, N, M], `weight` is interleaved GGUF NVFP4,
    /// and `factor` is the matrix's calibrated global scale, not a doubled LUT correction.
    pub(crate) fn mmq_nvfp4_dense_plan<'a>(
        &self,
        weight: DeviceRange<'a>,
        shape: [usize; 3],
        factor: f32,
    ) -> Result<Nvfp4DensePlan<'a>, EngError> {
        self.ensure_usable()?;
        let shape = Nvfp4DenseShape::new(shape[0], shape[1], shape[2])?;
        require_fit("NVFP4 weight", &weight, shape.needs.w, 16)?;
        self.raw_range(&weight)?;
        let factors = self.upload(&shape.factors(factor)?)?;
        Ok(Nvfp4DensePlan {
            shape,
            weight,
            factors,
        })
    }

    pub(crate) fn mmq_nvfp4_dense_input(
        &self,
        stream: &DeviceStream,
        plan: &Nvfp4DensePlan<'_>,
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        self.raw_range(&plan.weight)?;
        let factors = plan.factors.range(0, plan.shape.factor_bytes)?;
        self.raw_range(&factors)?;
        let output = Claim::of("q8_1", &y, plan.shape.needs.y);
        require_apart(
            &output,
            &Claim::of("NVFP4 weight", &plan.weight, plan.shape.needs.w),
        )?;
        require_apart(
            &output,
            &Claim::of("projection factors", &factors, plan.shape.factor_bytes),
        )?;
        self.mmq_quantize(
            stream,
            MmqQuantizeCall {
                packing: MmqPacking::Vendor,
                ty: MmqType::Nvfp4,
                k: plan.shape.ints.k as usize,
                row_stride: plan.shape.ints.k as usize,
                ncols: plan.shape.ints.m as usize,
                x,
                gather: None,
                y,
            },
        )
    }

    pub(crate) fn mmq_nvfp4_dense(
        &self,
        stream: &DeviceStream,
        call: Nvfp4DenseCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let plan = call.plan;
        let need = plan.shape.needs;
        let floor = self.mmq_fixup_bytes()?;
        require_fit("NVFP4 weight", &plan.weight, need.w, 16)?;
        require_fit("q8_1", &call.q8_1, need.y, 16)?;
        require_fit("dst", &call.dst, need.dst, 4)?;
        require_fit("fixup", &call.fixup, floor, 16)?;
        let factors = plan.factors.range(0, plan.shape.factor_bytes)?;
        let reads = [
            Claim::of("NVFP4 weight", &plan.weight, need.w),
            Claim::of("q8_1", &call.q8_1, need.y),
            Claim::of("projection factors", &factors, plan.shape.factor_bytes),
        ];
        let dst = Claim::of("dst", &call.dst, need.dst);
        let fixup = Claim::of("fixup", &call.fixup, call.fixup.bytes());
        require_apart(&dst, &fixup)?;
        for read in &reads {
            require_apart(&dst, read)?;
            require_apart(&fixup, read)?;
        }
        let raw = MmqDenseRaw {
            w: self.raw_range(&plan.weight)?,
            y: self.raw_range(&call.q8_1)?,
            dst: self.raw_range(&call.dst)?,
            fixup: self.raw_range(&call.fixup)?,
            ints: plan.shape.ints,
        };
        let factors = self.raw_range(&factors)?;
        self.launch_on(stream, move |stream| {
            mmq_nvfp4_dense_launch(stream, raw, factors)
        })
    }
}

#[cfg(test)]
mod tests;
