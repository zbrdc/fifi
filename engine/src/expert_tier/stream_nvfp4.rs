//! The NVFP4 GPU callback over typed bank or hot-resident ranges.

use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceStream, Nvfp4WeightRange};

fn memory() -> EngError {
    EngError::MemoryPlan("NVFP4 routed batch buffer size or allocation exceeds capacity".into())
}

/// Returns one output row per supplied pair, in that exact order.
/// The caller admits pair I/O, immutable descriptors and status in addition to the banks.
pub(crate) fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &[Nvfp4WeightRange<'_>],
    inputs: &[f32],
) -> Result<Vec<f32>, EngError> {
    let shape = weights
        .first()
        .ok_or_else(|| EngError::BadRequest("NVFP4 routed batch needs a pair".into()))?
        .shape();
    let output_count = weights.len().checked_mul(shape[0]).ok_or_else(memory)?;
    let output = launch(ctx, stream, weights, inputs)?;
    read_f32_rows(ctx, stream, &output, output_count)
}

/// The grouped launch with host inputs; the output stays ON the device (the
/// activation between the up and down projections never crosses to the host).
pub(crate) fn launch_host_input(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &[Nvfp4WeightRange<'_>],
    inputs: &[f32],
) -> Result<crate::kernels::cuda::DeviceBuffer, EngError> {
    launch(ctx, stream, weights, inputs)
}

/// The grouped launch reading its input from a device buffer, output on the
/// device as well - the down projection of the streamed expert chunk.
pub(crate) fn launch_device_input(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &[Nvfp4WeightRange<'_>],
    inputs: &crate::kernels::cuda::DeviceBuffer,
    input_bytes: usize,
) -> Result<crate::kernels::cuda::DeviceBuffer, EngError> {
    ctx.ensure_usable()?;
    let shape = weights
        .first()
        .ok_or_else(|| EngError::BadRequest("NVFP4 routed batch needs a pair".into()))?
        .shape();
    let count = weights.len().checked_mul(shape[1]).ok_or_else(memory)?;
    if input_bytes != count.checked_mul(size_of::<f32>()).ok_or_else(memory)? {
        return Err(EngError::BadRequest(
            "NVFP4 routed batch device input has the wrong byte count".into(),
        ));
    }
    grouped_enqueue(ctx, stream, weights, inputs.range(0, input_bytes)?)
}

fn launch(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &[Nvfp4WeightRange<'_>],
    inputs: &[f32],
) -> Result<crate::kernels::cuda::DeviceBuffer, EngError> {
    ctx.ensure_usable()?;
    let shape = weights
        .first()
        .ok_or_else(|| EngError::BadRequest("NVFP4 routed batch needs a pair".into()))?
        .shape();
    let count = weights.len().checked_mul(shape[1]).ok_or_else(memory)?;
    if inputs.len() != count || inputs.iter().any(|value| !value.is_finite()) {
        return Err(EngError::BadRequest(
            "NVFP4 routed batch needs one finite input row per pair".into(),
        ));
    }
    let input_bytes = count.checked_mul(size_of::<f32>()).ok_or_else(memory)?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(input_bytes)
        .map_err(|_| memory())?;
    encoded.extend(inputs.iter().flat_map(|value| value.to_ne_bytes()));
    let input = ctx.upload(&encoded)?;
    grouped_enqueue(ctx, stream, weights, input.range(0, input_bytes)?)
}

fn grouped_enqueue(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &[Nvfp4WeightRange<'_>],
    input: crate::kernels::cuda::DeviceRange<'_>,
) -> Result<crate::kernels::cuda::DeviceBuffer, EngError> {
    let shape = weights
        .first()
        .ok_or_else(|| EngError::BadRequest("NVFP4 routed batch needs a pair".into()))?
        .shape();
    let output_count = weights.len().checked_mul(shape[0]).ok_or_else(memory)?;
    let output_bytes = output_count
        .checked_mul(size_of::<f32>())
        .ok_or_else(memory)?;
    let mut status = ctx.nvfp4_launch_status(stream)?;
    let plan = ctx.nvfp4_grouped_plan(weights)?;
    let output = ctx.allocate(output_bytes)?;
    let done = ctx.event()?;
    ctx.nvfp4_grouped_enqueue(
        stream,
        &plan,
        input,
        output.range(0, output_bytes)?,
        &mut status,
    )?;
    stream.record(&done)?;
    // A completed transfer alone cannot certify the kernel's numerical result.
    status.check(stream, &done)?;
    Ok(output)
}

pub(crate) fn read_f32_rows(
    _ctx: &CudaContext,
    stream: &DeviceStream,
    output: &crate::kernels::cuda::DeviceBuffer,
    output_count: usize,
) -> Result<Vec<f32>, EngError> {
    let output_bytes = output_count
        .checked_mul(size_of::<f32>())
        .ok_or_else(memory)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(output_bytes)
        .map_err(|_| memory())?;
    bytes.resize(output_bytes, 0);
    output.read_at(stream, 0, &mut bytes)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(output_count)
        .map_err(|_| memory())?;
    result.extend(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_ne_bytes(*v)),
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::project;
    use crate::EngError;
    use crate::expert_tier::{
        CudaStreamBanks, ExpertId, ExpertPayload, ExpertPayloadSource, Residency, StreamBankPlan,
    };
    use crate::kernels::cuda::CudaContext;

    struct Payloads(Vec<ExpertPayload>);

    impl ExpertPayloadSource for Payloads {
        fn parts(&self) -> usize {
            1
        }
        fn payload(&mut self, id: ExpertId, _: usize) -> Result<ExpertPayload, EngError> {
            self.0
                .get(id.expert as usize)
                .cloned()
                .ok_or_else(|| EngError::Checkpoint("unknown routed expert".into()))
        }
        fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
            Err(EngError::BadRequest(
                "whole-layer release is not used".into(),
            ))
        }
        fn release_experts(&mut self, _: &[ExpertId]) -> Result<(), EngError> {
            Ok(())
        }
    }

    #[test]
    fn bank_ranges_keep_the_routed_pair_order() {
        let observed = (|| -> Result<_, EngError> {
            let ctx = CudaContext::new()?;
            let mut banks = CudaStreamBanks::new(&ctx, StreamBankPlan::new(1024, 0, 4096)?)?;
            let mut source = Payloads(
                [0x11, 0x22]
                    .map(|code| ExpertPayload {
                        data: vec![code; 256],
                        scales: vec![0x38; 32],
                        tensor_scale: 1.0,
                        shape: [16, 32],
                    })
                    .to_vec(),
            );
            let pairs = [1, 0, 1].map(|expert| ExpertId { layer: 5, expert });
            let (bank, _) = banks.slots_mut();
            let lease = bank
                .stage(&ctx, &mut source, &Residency::default(), &pairs)?
                .ok_or_else(|| EngError::BadRequest("expected tail weights".into()))?;
            let weights = pairs
                .iter()
                .map(|&id| {
                    let part = lease.part(id, 0)?;
                    ctx.nvfp4_weight_range(
                        part.data,
                        part.scales
                            .ok_or_else(|| EngError::Checkpoint("missing NVFP4 scales".into()))?,
                        part.tensor_scale,
                        part.shape.map(|v| v as usize),
                    )
                })
                .collect::<Result<Vec<_>, EngError>>()?;
            let input: Vec<f32> = [0.125, 0.5, -0.25]
                .into_iter()
                .flat_map(|value| std::iter::repeat_n(value, 32))
                .collect();
            let output = project(&ctx, lease.stream(), &weights, &input)?;
            let mismatch = project(&ctx, lease.stream(), &weights, &input[..95]);
            drop(weights);
            lease.finish()?;
            ctx.ensure_usable()?;
            Ok((output, mismatch))
        })();
        let (output, mismatch) = observed.expect("NVFP4 projection from owned bank ranges");
        let expected: Vec<f32> = [4.0, 8.0, -8.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 16))
            .collect();
        assert_eq!(output, expected);
        assert!(matches!(mismatch, Err(EngError::BadRequest(_))));
    }
}
