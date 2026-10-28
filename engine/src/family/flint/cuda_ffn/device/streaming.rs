//! Staged mixed-residency Flint FFN; the serving factory is not wired here.

mod prepared;
mod source;
#[cfg(test)]
pub(super) use prepared::PreparedSource;
#[cfg(test)]
pub(super) use source::GgufSource;

use super::*;
use crate::expert_tier::{CudaStreamBanks, ExpertId, ExpertPayloadSource, StreamBankPlan};

const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 512;
const SHARED: usize = 1024;
const EXPERTS: usize = 256;
const PART_BYTES: u64 = 589_824;

fn block(rows: usize) -> Result<FfnBlock, EngError> {
    FfnBlock::new(
        rows,
        FfnShape::Moe {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            shared_intermediate: SHARED,
            experts: EXPERTS,
        },
    )
}

/// Reserve before hot placement; includes both banks and raw/repacked overlap in each slot.
/// The row workspace is an upper bound, not a claim of measured peak memory.
pub(super) fn plan(rows: usize) -> Result<StreamBankPlan, EngError> {
    let block = block(rows)?;
    let pair = (16 * HIDDEN + 32 * INTERMEDIATE) as u64 * 4 + 256;
    let shared = (16 * HIDDEN + 32 * SHARED) as u64 * 4;
    let device = 1_048_576
        + (EXPERTS * HIDDEN * 4) as u64
        + pair * block.pairs as u64
        + shared * rows as u64;
    let host = 2 * (2 * PART_BYTES + 65_536) + block.pairs as u64 * 128;
    StreamBankPlan::new(3 * PART_BYTES, device, host)
}

pub(super) struct MoeCall<'a, S> {
    pub(super) layer: u32,
    pub(super) rows: usize,
    pub(super) input: DeviceRange<'a>,
    pub(super) router: DeviceRange<'a>,
    pub(super) shared: ExpertWeights<'a>,
    pub(super) hot: &'a [(u32, ExpertWeights<'a>)],
    pub(super) source: &'a mut S,
    pub(super) banks: &'a mut CudaStreamBanks,
}

fn validate<S: ExpertPayloadSource>(call: &MoeCall<'_, S>) -> Result<(), EngError> {
    if !(2..49).contains(&call.layer) || call.source.parts() != 3 {
        return Err(bad(
            "fetch requires a Flint MoE layer and three expert parts",
        ));
    }
    if call.shared.shape()? != (HIDDEN, SHARED) {
        return Err(bad("fetch shared expert geometry differs"));
    }
    for (index, (id, weight)) in call.hot.iter().enumerate() {
        if *id >= EXPERTS as u32
            || call.hot[..index].iter().any(|(other, _)| other == id)
            || weight.shape()? != (HIDDEN, INTERMEDIATE)
        {
            return Err(bad(
                "fetch hot experts have duplicate IDs or invalid geometry",
            ));
        }
    }
    Ok(())
}

// The extra BF16 input publication is idempotent: x is the parent's published input.
fn evaluate(
    ctx: &CudaContext,
    stream: &DeviceStream,
    x: DeviceRange<'_>,
    tokens: &[i32],
    weights: ExpertWeights<'_>,
) -> Result<DeviceBuffer, EngError> {
    let mut frame = Frame::new(ctx, stream)?;
    let input_shape = RowShape {
        rows: x.bytes() / (HIDDEN * 4),
        columns: HIDDEN,
    };
    let input = frame.input(x, input_shape)?;
    let indices = frame.upload_indices(tokens)?;
    let gathered = frame.gather(
        input,
        indices,
        RowShape {
            rows: tokens.len(),
            columns: HIDDEN,
        },
        input_shape.rows,
    )?;
    let plan = ExpertPlan::new(ctx, &vec![weights; tokens.len()])?;
    let output = frame.expert(&plan, gathered)?;
    frame.finish(output)
}

/// Caller admits plan(rows), input/router/shared and hot weights before this staged entry.
/// Cold descriptors are consumed on their bank's stream and never outlive its lease.
pub(super) fn forward<S: ExpertPayloadSource>(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: MoeCall<'_, S>,
) -> Result<DeviceBuffer, EngError> {
    let block = block(call.rows)?;
    validate(&call)?;
    let shape = RowShape {
        rows: call.rows,
        columns: HIDDEN,
    };
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(call.input, shape)?;
    let routed = route(&mut frame, call.router, x, block)?;
    let ready = ctx.event()?;
    stream.record(&ready)?;
    let pairs = RowShape {
        rows: block.pairs,
        columns: HIDDEN,
    };
    let sorted = frame.allocate(pairs)?;
    let mut first = 0;
    while first < block.pairs {
        let expert = routed.ids[routed.grouping.order[first]] as u32;
        let mut end = first + 1;
        while end < block.pairs && routed.ids[routed.grouping.order[end]] as u32 == expert {
            end += 1;
        }
        let tokens = &routed.grouping.tokens[first..end];
        let output = if let Some((_, weights)) = call.hot.iter().find(|(id, _)| *id == expert) {
            evaluate(ctx, stream, frame.range(x)?, tokens, *weights)?
        } else {
            let id = ExpertId {
                layer: call.layer,
                expert,
            };
            let (bank, _) = call.banks.slots_mut();
            let lease = bank
                .stage_demand(ctx, call.source, &[id])?
                .ok_or_else(|| bad("cold fetch returned no bank lease"))?;
            lease.stream().wait(&ready)?;
            let output = {
                let mut parts = Vec::with_capacity(3);
                for index in 0..3 {
                    let part = lease.part(id, index)?;
                    let shape = part.shape.map(|n| n as usize);
                    parts.push(
                        ctx.nvfp4_weight_range(
                            part.data,
                            part.scales
                                .ok_or_else(|| bad("cold expert part has no scales"))?,
                            part.tensor_scale,
                            shape,
                        )?,
                    );
                }
                let weights = ExpertWeights {
                    gate: parts[0],
                    up: parts[1],
                    down: parts[2],
                };
                if weights.shape()? != (HIDDEN, INTERMEDIATE) {
                    return Err(bad("cold expert geometry differs"));
                }
                evaluate(ctx, lease.stream(), frame.range(x)?, tokens, weights)?
            };
            lease.finish()?;
            output
        };
        let bytes = output.bytes();
        let output = frame.keep(output, bytes);
        ctx.copy_device(
            stream,
            frame.range(output)?,
            frame.buffers[sorted].0.range(first * HIDDEN * 4, bytes)?,
        )?;
        first = end;
    }
    let inverse = frame.upload_indices(&routed.grouping.inverse)?;
    let down = frame.gather(sorted, inverse, pairs, block.pairs)?;
    let shared = ExpertPlan::new(ctx, &vec![call.shared; call.rows])?;
    let shared = frame.expert(&shared, x)?;
    let result = frame.combine(down, routed.gates, shared, block)?;
    frame.finish(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_plan_counts_both_raw_repack_slots_and_rejects_invalid_rows() {
        assert!(plan(0).is_err());
        assert!(plan(4097).is_err());
        let plan = plan(5).unwrap();
        assert_eq!(plan.bank_bytes(), 3 * PART_BYTES);
        assert!(plan.host_bytes() >= 2 * plan.bank_bytes() + 4 * PART_BYTES);
        assert!(plan.device_bytes() > 2 * plan.bank_bytes());
    }
}
