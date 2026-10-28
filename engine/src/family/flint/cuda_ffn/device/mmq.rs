//! Explicit FFN-only MMQ composition. The Frame entry and its projection method stay unchanged.
use super::*;
use crate::kernels::cuda::prefill::mmq_nvfp4::{Nvfp4DenseCall, Nvfp4DensePlan, Nvfp4DenseShape};

#[derive(Clone, Copy)]
pub(crate) struct MmqMatrix<'a> {
    pub(crate) data: DeviceRange<'a>,
    pub(crate) factor: f32,
    pub(crate) shape: [usize; 2],
}

pub(crate) struct MmqExpert<'a> {
    pub(crate) gate: MmqMatrix<'a>,
    pub(crate) up: MmqMatrix<'a>,
    pub(crate) down: MmqMatrix<'a>,
}

impl<'a> MmqMatrix<'a> {
    fn plan(
        self,
        ctx: &CudaContext,
        rows: usize,
        expected: [usize; 2],
    ) -> Result<Nvfp4DensePlan<'a>, EngError> {
        if self.shape != expected {
            return Err(bad("MMQ source and FFN matrix shapes disagree"));
        }
        ctx.mmq_nvfp4_dense_plan(self.data, [expected[1], expected[0], rows], self.factor)
    }
}

impl Frame<'_, '_> {
    fn mmq_project(
        &mut self,
        plan: &Nvfp4DensePlan<'_>,
        x: usize,
        rows: usize,
        matrix: [usize; 2],
    ) -> Result<usize, EngError> {
        let [n, k] = matrix;
        let shape = Nvfp4DenseShape::new(k, n, rows)?;
        let normalized = self.allocate(RowShape { rows, columns: k })?;
        let scales = self.allocate(RowShape { rows, columns: 1 })?;
        let output = RowShape { rows, columns: n };
        let raw = self.allocate(output)?;
        let restored = self.allocate(output)?;
        let q8 = self.keep(self.ctx.allocate(shape.needs.y)?, shape.needs.y);
        let fixup_bytes = self.ctx.mmq_fixup_bytes()?;
        let fixup = self.keep(self.ctx.allocate(fixup_bytes)?, fixup_bytes);
        self.ctx.row_pow2(
            self.stream,
            RowPow2Call {
                x: self.range(x)?,
                out: self.range(normalized)?,
                scales: self.range(scales)?,
                flag: self.flag()?,
                rows,
                columns: k,
                mode: RowPow2Mode::Normalize,
            },
        )?;
        self.ctx.mmq_nvfp4_dense_input(
            self.stream,
            plan,
            self.range(normalized)?,
            self.range(q8)?,
        )?;
        self.ctx.mmq_nvfp4_dense(
            self.stream,
            Nvfp4DenseCall {
                plan,
                q8_1: self.range(q8)?,
                dst: self.range(raw)?,
                fixup: self.range(fixup)?,
            },
        )?;
        self.ctx.row_pow2(
            self.stream,
            RowPow2Call {
                x: self.range(raw)?,
                out: self.range(restored)?,
                scales: self.range(scales)?,
                flag: self.flag()?,
                rows,
                columns: n,
                mode: RowPow2Mode::Restore,
            },
        )?;
        self.publish(restored, output)
    }

    fn mmq_expert(
        &mut self,
        plans: &[Nvfp4DensePlan<'_>; 3],
        x: usize,
        rows: usize,
        matrix: [usize; 2],
    ) -> Result<usize, EngError> {
        let shape = RowShape {
            rows,
            columns: matrix[0],
        };
        let gate = self.mmq_project(&plans[0], x, rows, matrix)?;
        let up = self.mmq_project(&plans[1], x, rows, matrix)?;
        let sigmoid = self.sigmoid(gate, shape)?;
        let silu = self.multiply(gate, sigmoid, shape, shape.columns)?;
        let hidden = self.multiply(silu, up, shape, shape.columns)?;
        self.mmq_project(&plans[2], hidden, rows, [matrix[1], matrix[0]])
    }

    #[cfg(test)]
    pub(super) fn bench_mmq_project(
        &mut self,
        plan: &Nvfp4DensePlan<'_>,
        x: usize,
        rows: usize,
        matrix: [usize; 2],
    ) -> Result<usize, EngError> {
        self.mmq_project(plan, x, rows, matrix)
    }
}

pub(crate) fn forward_mmq(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: FfnCall<'_, '_>,
    raw: MmqExpert<'_>,
) -> Result<DeviceBuffer, EngError> {
    let block = call.weights.block(call.rows)?;
    let (hidden, intermediate) = match call.weights {
        FfnWeights::Dense(weight) => weight.shape()?,
        FfnWeights::Moe { shared, .. } => shared.shape()?,
    };
    // Plan-owned factors must outlive completion, not drain the stream after each projection.
    let plans = [
        raw.gate.plan(ctx, block.rows, [intermediate, hidden])?,
        raw.up.plan(ctx, block.rows, [intermediate, hidden])?,
        raw.down.plan(ctx, block.rows, [hidden, intermediate])?,
    ];
    let mut frame = Frame::new(ctx, stream)?;
    let shape = RowShape {
        rows: block.rows,
        columns: hidden,
    };
    let x = frame.input(call.x, shape)?;
    let output = match call.weights {
        FfnWeights::Dense(_) => frame.mmq_expert(&plans, x, block.rows, [intermediate, hidden])?,
        FfnWeights::Moe {
            router, experts, ..
        } => {
            let routed = route(&mut frame, *router, x, block)?;
            let selected: Vec<_> = routed
                .grouping
                .order
                .iter()
                .map(|&p| experts[routed.ids[p] as usize])
                .collect();
            let plan = ExpertPlan::new(ctx, &selected)?;
            let tokens = frame.upload_indices(&routed.grouping.tokens)?;
            let inverse = frame.upload_indices(&routed.grouping.inverse)?;
            let pairs = RowShape {
                rows: block.pairs,
                columns: hidden,
            };
            let gathered = frame.gather(x, tokens, pairs, block.rows)?;
            let sorted = frame.expert(&plan, gathered)?;
            let down = frame.gather(sorted, inverse, pairs, block.pairs)?;
            let shared = frame.mmq_expert(&plans, x, block.rows, [intermediate, hidden])?;
            let result = frame.combine(down, routed.gates, shared, block)?;
            // Routed descriptors own device storage until the late combine has completed.
            return frame.finish(result);
        }
    };
    frame.finish(output)
}
