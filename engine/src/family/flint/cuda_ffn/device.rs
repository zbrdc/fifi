//! One block-wide Flint FFN. BF16 values use F32 backing storage; no residual here.
#[cfg(test)]
pub(crate) mod mmq;
#[cfg(test)]
pub(crate) mod route_trace;

use super::plan::{FfnBlock, FfnShape, Grouping, TOP_K};
use crate::EngError;
use crate::kernels::cuda::prefill::row_ops::RowGatherCall;
use crate::kernels::cuda::prefill::row_ops::{RowMulCall, RowShape, RowSumCall, RowUnaryCall};
use crate::kernels::cuda::resident::router::{RouterLogitsCall, RouterShape, RouterSigmoidCall};
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::resident::row_scale::Bf16PublishCall;
#[cfg(test)]
use crate::kernels::cuda::resident::row_scale::{RowPow2Call, RowPow2Mode};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};
#[cfg(test)]
use crate::kernels::cuda::{Nvfp4GroupedPlan, Nvfp4LaunchStatus, Nvfp4WeightRange};

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct ExpertWeights<'a> {
    pub(crate) gate: Nvfp4WeightRange<'a>,
    pub(crate) up: Nvfp4WeightRange<'a>,
    pub(crate) down: Nvfp4WeightRange<'a>,
}

#[cfg(test)]
pub(crate) enum FfnWeights<'a> {
    Dense(ExpertWeights<'a>),
    Moe {
        router: DeviceRange<'a>,
        experts: &'a [ExpertWeights<'a>],
        shared: ExpertWeights<'a>,
    },
}

#[cfg(test)]
pub(crate) struct FfnCall<'a, 'w> {
    pub(crate) weights: &'a FfnWeights<'w>,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) rows: usize,
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint CUDA FFN: {message}"))
}

#[cfg(test)]
impl ExpertWeights<'_> {
    fn shape(&self) -> Result<(usize, usize), EngError> {
        let [intermediate, hidden] = self.gate.shape();
        if self.up.shape() != [intermediate, hidden] || self.down.shape() != [hidden, intermediate]
        {
            return Err(bad("gate/up/down matrix shapes disagree"));
        }
        Ok((hidden, intermediate))
    }
}

#[cfg(test)]
impl FfnWeights<'_> {
    fn block(&self, rows: usize) -> Result<FfnBlock, EngError> {
        let shape = match self {
            Self::Dense(weight) => {
                let (hidden, intermediate) = weight.shape()?;
                FfnShape::Dense {
                    hidden,
                    intermediate,
                }
            }
            Self::Moe {
                experts, shared, ..
            } => {
                let first = experts.first().ok_or_else(|| bad("empty expert bank"))?;
                let (hidden, intermediate) = first.shape()?;
                for expert in *experts {
                    if expert.shape()? != (hidden, intermediate) {
                        return Err(bad("routed experts must have one common shape"));
                    }
                }
                let (shared_hidden, shared_intermediate) = shared.shape()?;
                if shared_hidden != hidden {
                    return Err(bad("shared expert hidden width differs"));
                }
                FfnShape::Moe {
                    hidden,
                    intermediate,
                    shared_intermediate,
                    experts: experts.len(),
                }
            }
        };
        FfnBlock::new(rows, shape)
    }
}

#[cfg(test)]
struct ExpertPlan<'a> {
    gate: Nvfp4GroupedPlan<'a>,
    up: Nvfp4GroupedPlan<'a>,
    down: Nvfp4GroupedPlan<'a>,
    rows: usize,
    hidden: usize,
    intermediate: usize,
}

#[cfg(test)]
impl<'a> ExpertPlan<'a> {
    fn new(ctx: &CudaContext, weights: &[ExpertWeights<'a>]) -> Result<Self, EngError> {
        let (hidden, intermediate) = weights
            .first()
            .ok_or_else(|| bad("empty projection plan"))?
            .shape()?;
        Ok(Self {
            gate: ctx.nvfp4_grouped_plan(&weights.iter().map(|w| w.gate).collect::<Vec<_>>())?,
            up: ctx.nvfp4_grouped_plan(&weights.iter().map(|w| w.up).collect::<Vec<_>>())?,
            down: ctx.nvfp4_grouped_plan(&weights.iter().map(|w| w.down).collect::<Vec<_>>())?,
            rows: weights.len(),
            hidden,
            intermediate,
        })
    }
}

enum FrameFlag<'a> {
    Owned(DeviceBuffer),
    Deferred(DeviceRange<'a>),
}

// Retain all scratch until late completion: dropping a CUDA buffer drains queued work.
struct Frame<'c, 's> {
    ctx: &'c CudaContext,
    stream: &'s DeviceStream,
    buffers: Vec<(DeviceBuffer, usize)>,
    flag: FrameFlag<'c>,
    #[cfg(test)]
    statuses: Vec<Nvfp4LaunchStatus<'s>>,
}

impl<'c, 's> Frame<'c, 's> {
    fn new(ctx: &'c CudaContext, stream: &'s DeviceStream) -> Result<Self, EngError> {
        Ok(Self::with_flag(
            ctx,
            stream,
            FrameFlag::Owned(ctx.upload(&[0; 4])?),
        ))
    }

    fn deferred(
        ctx: &'c CudaContext,
        stream: &'s DeviceStream,
        flag: DeviceRange<'c>,
    ) -> Result<Self, EngError> {
        if flag.bytes() != 4 {
            return Err(bad("deferred health requires one status word"));
        }
        Ok(Self::with_flag(ctx, stream, FrameFlag::Deferred(flag)))
    }

    fn with_flag(ctx: &'c CudaContext, stream: &'s DeviceStream, flag: FrameFlag<'c>) -> Self {
        Self {
            ctx,
            stream,
            buffers: Vec::new(),
            flag,
            #[cfg(test)]
            statuses: Vec::new(),
        }
    }

    fn keep(&mut self, buffer: DeviceBuffer, bytes: usize) -> usize {
        let index = self.buffers.len();
        self.buffers.push((buffer, bytes));
        index
    }

    fn allocate(&mut self, shape: RowShape) -> Result<usize, EngError> {
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("scratch shape exceeds row geometry"))?;
        Ok(self.keep(self.ctx.allocate(bytes)?, bytes))
    }

    fn range(&self, index: usize) -> Result<DeviceRange<'_>, EngError> {
        self.buffers[index].0.range(0, self.buffers[index].1)
    }

    fn flag(&self) -> Result<DeviceRange<'_>, EngError> {
        match &self.flag {
            FrameFlag::Owned(buffer) => buffer.range(0, 4),
            FrameFlag::Deferred(flag) => Ok(*flag),
        }
    }

    #[cfg(test)]
    fn test_set_health(&self, status: u32) -> Result<(), EngError> {
        let flag = self.flag()?;
        flag.buffer()
            .write_at(self.stream, flag.offset(), &status.to_ne_bytes())
    }

    fn input(&mut self, x: DeviceRange<'_>, shape: RowShape) -> Result<usize, EngError> {
        let out = self.allocate(shape)?;
        self.input_into(x, self.range(out)?, shape)?;
        Ok(out)
    }

    fn input_into(
        &self,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.ctx.publish_bf16_rows(
            self.stream,
            Bf16PublishCall {
                x,
                out,
                flag: self.flag()?,
                rows: shape.rows,
                columns: shape.columns,
            },
        )
    }

    fn publish(&mut self, x: usize, shape: RowShape) -> Result<usize, EngError> {
        let out = self.allocate(shape)?;
        self.ctx.publish_bf16_rows(
            self.stream,
            Bf16PublishCall {
                x: self.range(x)?,
                out: self.range(out)?,
                flag: self.flag()?,
                rows: shape.rows,
                columns: shape.columns,
            },
        )?;
        Ok(out)
    }

    fn sigmoid(&mut self, x: usize, shape: RowShape) -> Result<usize, EngError> {
        let out = self.allocate(shape)?;
        self.ctx.row_sigmoid(
            self.stream,
            RowUnaryCall {
                x: self.range(x)?,
                out: self.range(out)?,
                flag: self.flag()?,
                shape,
            },
        )?;
        self.publish(out, shape)
    }

    fn multiply(
        &mut self,
        x: usize,
        rhs: usize,
        shape: RowShape,
        rhs_columns: usize,
    ) -> Result<usize, EngError> {
        let out = self.allocate(shape)?;
        self.ctx.row_multiply(
            self.stream,
            RowMulCall {
                x: self.range(x)?,
                rhs: self.range(rhs)?,
                out: self.range(out)?,
                flag: self.flag()?,
                shape,
                rhs_columns,
            },
        )?;
        self.publish(out, shape)
    }

    // Kept for test-only legacy fixtures and the frozen-original oracle.
    #[cfg(test)]
    fn gather(
        &mut self,
        x: usize,
        indices: usize,
        shape: RowShape,
        source_rows: usize,
    ) -> Result<usize, EngError> {
        let out = self.allocate(shape)?;
        self.ctx.row_gather(
            self.stream,
            RowGatherCall {
                x: self.range(x)?,
                indices: self.range(indices)?,
                out: self.range(out)?,
                flag: self.flag()?,
                shape,
                source_rows,
            },
        )?;
        Ok(out)
    }

    fn upload_indices(&mut self, indices: &[i32]) -> Result<usize, EngError> {
        let bytes: Vec<_> = indices.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Ok(self.keep(self.ctx.upload(&bytes)?, bytes.len()))
    }

    #[cfg(test)]
    fn project(
        &mut self,
        plan: &Nvfp4GroupedPlan<'_>,
        x: usize,
        rows: usize,
        matrix: [usize; 2],
    ) -> Result<usize, EngError> {
        let raw = self.project_f32(plan, x, rows, matrix)?;
        self.publish(
            raw,
            RowShape {
                rows,
                columns: matrix[0],
            },
        )
    }

    #[cfg(test)]
    fn project_f32(
        &mut self,
        plan: &Nvfp4GroupedPlan<'_>,
        x: usize,
        rows: usize,
        matrix: [usize; 2],
    ) -> Result<usize, EngError> {
        let input = RowShape {
            rows,
            columns: matrix[1],
        };
        let output = RowShape {
            rows,
            columns: matrix[0],
        };
        let normalized = self.allocate(input)?;
        let scales = self.allocate(RowShape { rows, columns: 1 })?;
        let raw = self.allocate(output)?;
        let restored = self.allocate(output)?;
        self.ctx.row_pow2(
            self.stream,
            RowPow2Call {
                x: self.range(x)?,
                out: self.range(normalized)?,
                scales: self.range(scales)?,
                flag: self.flag()?,
                rows,
                columns: input.columns,
                mode: RowPow2Mode::Normalize,
            },
        )?;
        let mut status = self.ctx.nvfp4_launch_status(self.stream)?;
        #[cfg(test)]
        let mut timer = tests::profile::project_begin(self.ctx, self.stream)?;
        self.ctx.nvfp4_grouped_enqueue(
            self.stream,
            plan,
            self.range(normalized)?,
            self.range(raw)?,
            &mut status,
        )?;
        #[cfg(test)]
        tests::profile::project_end(self.stream, &mut timer)?;
        let projected = self.ctx.event()?;
        self.stream.record(&projected)?;
        status.check(self.stream, &projected)?;
        #[cfg(test)]
        tests::profile::project_complete(timer)?;
        // Retain the checked flag so its free cannot drain later row kernels.
        self.statuses.push(status);
        self.ctx.row_pow2(
            self.stream,
            RowPow2Call {
                x: self.range(raw)?,
                out: self.range(restored)?,
                scales: self.range(scales)?,
                flag: self.flag()?,
                rows,
                columns: output.columns,
                mode: RowPow2Mode::Restore,
            },
        )?;
        Ok(restored)
    }

    #[cfg(test)]
    fn expert(&mut self, plan: &ExpertPlan<'_>, x: usize) -> Result<usize, EngError> {
        let shape = RowShape {
            rows: plan.rows,
            columns: plan.intermediate,
        };
        let gate = self.project(&plan.gate, x, plan.rows, [plan.intermediate, plan.hidden])?;
        let up = self.project(&plan.up, x, plan.rows, [plan.intermediate, plan.hidden])?;
        let sig = self.sigmoid(gate, shape)?;
        let silu = self.multiply(gate, sig, shape, shape.columns)?;
        let hidden = self.multiply(silu, up, shape, shape.columns)?;
        self.project(
            &plan.down,
            hidden,
            plan.rows,
            [plan.hidden, plan.intermediate],
        )
    }

    fn combine(
        &mut self,
        down: usize,
        gates: usize,
        shared: usize,
        block: FfnBlock,
    ) -> Result<usize, EngError> {
        let gate_shape = RowShape {
            rows: block.rows,
            columns: TOP_K,
        };
        let gates = self.publish(gates, gate_shape)?;
        let pairs = RowShape {
            rows: block.pairs,
            columns: block.hidden,
        };
        let products = self.multiply(down, gates, pairs, 1)?;
        let shape = RowShape {
            rows: block.rows,
            columns: block.hidden,
        };
        let sum = self.allocate(shape)?;
        self.ctx.row_rank_sum(
            self.stream,
            RowSumCall {
                x: self.range(products)?,
                out: self.range(sum)?,
                flag: self.flag()?,
                shape,
                rank: TOP_K,
            },
        )?;
        let total = self.publish(sum, shape)?;
        let added = self.allocate(shape)?;
        self.ctx.row_add(
            self.stream,
            RowAddCall {
                x: self.range(total)?,
                y: self.range(shared)?,
                out: self.range(added)?,
                n: block.rows * block.hidden,
            },
        )?;
        self.publish(added, shape)
    }

    fn complete(&self) -> Result<(), EngError> {
        let FrameFlag::Owned(buffer) = &self.flag else {
            return Err(bad(
                "deferred output requires its attention health observation",
            ));
        };
        let event = self.ctx.event()?;
        self.stream.record(&event)?;
        event.synchronize()?;
        let mut flag = [0; 4];
        buffer.read_at(self.stream, 0, &mut flag)?;
        Self::validate_health(u32::from_ne_bytes(flag))
    }

    fn validate_health(status: u32) -> Result<(), EngError> {
        if status != 0 {
            // This flag also covers invalid indices derived from nonfinite activations.
            return Err(EngError::ModelOutput(
                "Flint CUDA FFN: late numeric/index health failed; output invalid".into(),
            ));
        }
        Ok(())
    }

    fn finish(mut self, output: usize) -> Result<DeviceBuffer, EngError> {
        self.complete()?;
        Ok(self.buffers.swap_remove(output).0)
    }
}

struct Routed {
    ids: Vec<i32>,
    grouping: Grouping,
    gates: usize,
    #[cfg(test)]
    logits: Option<(usize, usize)>,
}

enum PublishedRouter<'a> {
    Frame(usize),
    Scratch(DeviceRange<'a>),
}

#[cfg(test)]
fn route(
    frame: &mut Frame<'_, '_>,
    router: DeviceRange<'_>,
    x: usize,
    block: FfnBlock,
) -> Result<Routed, EngError> {
    route_reusing(frame, router, x, block, None)
}

fn route_reusing(
    frame: &mut Frame<'_, '_>,
    router: DeviceRange<'_>,
    x: usize,
    block: FfnBlock,
    scratch: Option<DeviceRange<'_>>,
) -> Result<Routed, EngError> {
    let shape = RouterShape {
        m: block.rows,
        hidden: block.hidden,
        n_expert: block.experts,
        top_k: TOP_K,
    };
    let matrix = RowShape {
        rows: block.experts,
        columns: block.hidden,
    };
    let router = match scratch {
        Some(out) => {
            frame.input_into(router, out, matrix)?;
            PublishedRouter::Scratch(out)
        }
        None => PublishedRouter::Frame(frame.input(router, matrix)?),
    };
    let logits_shape = RowShape {
        rows: block.rows,
        columns: block.experts,
    };
    let logits = frame.allocate(logits_shape)?;
    frame.ctx.router_logits_rows(
        frame.stream,
        RouterLogitsCall {
            shape,
            x: frame.range(x)?,
            w: match router {
                PublishedRouter::Frame(index) => frame.range(index)?,
                PublishedRouter::Scratch(out) => out,
            },
            logits: frame.range(logits)?,
        },
    )?;
    #[cfg(test)]
    let raw_logits = logits;
    let logits = frame.publish(logits, logits_shape)?;
    let picks = RowShape {
        rows: block.rows,
        columns: TOP_K,
    };
    let ids_bytes = picks
        .bytes()
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(|| bad("router ID and status bytes overflow"))?;
    let status_offset = ids_bytes - 4;
    let ids = frame.keep(frame.ctx.allocate(ids_bytes)?, ids_bytes);
    let gates = frame.allocate(picks)?;
    frame.ctx.router_sigmoid_select_rows(
        frame.stream,
        RouterSigmoidCall {
            shape,
            logits: frame.range(logits)?,
            ids: frame.range(ids)?,
            weights: frame.range(gates)?,
            status: frame.flag()?,
        },
    )?;
    frame.ctx.copy_device(
        frame.stream,
        frame.flag()?,
        frame.buffers[ids].0.range(status_offset, 4)?,
    )?;
    let mut bytes = vec![0; ids_bytes];
    // shortcut: one block-wide ID readback; replace with the shared NVFP4 indexed device-grouping plan.
    frame.buffers[ids].0.read_at(frame.stream, 0, &mut bytes)?;
    let (ids, status) = bytes.split_at(status_offset);
    // Sanitized logits can yield valid-looking IDs even while upstream status is poisoned.
    if status != [0; 4] {
        return Err(bad("status set at or before the router"));
    }
    let ids: Vec<_> = ids
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| i32::from_ne_bytes(*v))
        .collect();
    let grouping = Grouping::new(&ids, block.rows, block.experts)?;
    #[cfg(test)]
    route_trace::record(&ids, block.rows, block.experts)?;
    Ok(Routed {
        ids,
        grouping,
        gates,
        #[cfg(test)]
        logits: Some((raw_logits, logits)),
    })
}

#[cfg(test)]
fn moe(
    mut frame: Frame<'_, '_>,
    x: usize,
    block: FfnBlock,
    weights: &FfnWeights<'_>,
) -> Result<DeviceBuffer, EngError> {
    let FfnWeights::Moe { router, .. } = weights else {
        return Err(bad("expected MoE weights"));
    };
    let routed = route(&mut frame, *router, x, block)?;
    moe_routed(frame, x, block, weights, routed)
}

#[cfg(test)]
fn moe_routed(
    mut frame: Frame<'_, '_>,
    x: usize,
    block: FfnBlock,
    weights: &FfnWeights<'_>,
    routed: Routed,
) -> Result<DeviceBuffer, EngError> {
    let FfnWeights::Moe {
        experts, shared, ..
    } = weights
    else {
        return Err(bad("expected MoE weights"));
    };
    let selected: Vec<_> = routed
        .grouping
        .order
        .iter()
        .map(|&p| experts[routed.ids[p] as usize])
        .collect();
    let expert_plan = ExpertPlan::new(frame.ctx, &selected)?;
    let shared_plan = ExpertPlan::new(frame.ctx, &vec![*shared; block.rows])?;
    let tokens = frame.upload_indices(&routed.grouping.tokens)?;
    let inverse = frame.upload_indices(&routed.grouping.inverse)?;
    let shape = RowShape {
        rows: block.pairs,
        columns: block.hidden,
    };
    let gathered = frame.gather(x, tokens, shape, block.rows)?;
    #[cfg(test)]
    let routed_phase = tests::profile::phase(tests::profile::Phase::Routed);
    let sorted_down = frame.expert(&expert_plan, gathered)?;
    #[cfg(test)]
    drop(routed_phase);
    let down = frame.gather(sorted_down, inverse, shape, block.pairs)?;
    #[cfg(test)]
    let shared_phase = tests::profile::phase(tests::profile::Phase::Shared);
    let shared = frame.expert(&shared_plan, x)?;
    #[cfg(test)]
    drop(shared_phase);
    let result = frame.combine(down, routed.gates, shared, block)?;
    // Completion includes late BF16/combine kernels while descriptor owners are live.
    frame.finish(result)
}

/// Returns only completed, healthy BF16-precision rows. Each projection consumes
/// the whole block/pair group. Dense prefix layers bypass router/shared work.
#[cfg(test)]
pub(crate) fn forward(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: FfnCall<'_, '_>,
) -> Result<DeviceBuffer, EngError> {
    let block = call.weights.block(call.rows)?;
    let mut frame = Frame::new(ctx, stream)?;
    let shape = RowShape {
        rows: block.rows,
        columns: block.hidden,
    };
    match call.weights {
        FfnWeights::Dense(weight) => {
            #[cfg(test)]
            let _dense_phase = tests::profile::phase(tests::profile::Phase::Dense);
            let plan = ExpertPlan::new(ctx, &vec![*weight; block.rows])?;
            let x = frame.input(call.x, shape)?;
            let output = frame.expert(&plan, x)?;
            // finish checks late BF16 health before plan-owned descriptors may be freed.
            frame.finish(output)
        }
        FfnWeights::Moe { .. } => {
            let x = frame.input(call.x, shape)?;
            moe(frame, x, block, call.weights)
        }
    }
}

#[cfg(test)]
mod streaming;

pub(crate) mod exl3;
#[cfg(test)]
mod exl3_dispatch_tests;
pub(crate) mod runtime;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(super) use tests::decode;
