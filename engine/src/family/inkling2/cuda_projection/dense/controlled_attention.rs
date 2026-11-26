//! Cross-pass controllers and their attention projections share a staging transaction.

use super::{
    ControllerWeights, DenseGemvCall, DenseGemvShape, DenseStep, ResidentProjection, Slot, bad,
    dense_gemv_bytes, split_projections,
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceRange, resident::inkling_controller::Arithmetic},
};

#[cfg(test)]
mod tests;

pub(in crate::family::inkling2) struct ControlledAttentionInput<'a> {
    pub gate: &'a [f32],
    pub input: &'a [f32],
    pub message: &'a [f32],
    pub gamma: &'a [f32],
}

pub(in crate::family::inkling2::cuda_projection) type ProjectionTile<'a, const N: usize> =
    (Vec<[Vec<f32>; N]>, Option<DeviceRange<'a>>);

/// Controlled attention rows on the device: each role's [input, message] pairs, `rows` per role
/// (role-major), its norm weight, and health words for the gate preparation (two per pair).
#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct ControlledRows<'a> {
    pub pairs: DeviceRange<'a>,
    pub gammas: [DeviceRange<'a>; 3],
    pub health: DeviceRange<'a>,
    pub rows: usize,
    pub hidden: usize,
    /// The epsilon of the gate halves' RMS norm.
    pub gate_eps: f32,
}

/// One controlled attention tile: arena offsets, launches and the readback split.
struct ControlledPlan<'a, const N: usize> {
    m: usize,
    hidden: usize,
    /// Gate words, then raw triple words, from the start of the output slot.
    gates: usize,
    raw: usize,
    control: Vec<DenseGemvCall<'a>>,
    projections: Vec<DenseGemvCall<'a>>,
    workspace: DeviceRange<'a>,
    health: DeviceRange<'a>,
    readback: DeviceRange<'a>,
    side: Option<DeviceRange<'a>>,
    host_widths: [usize; N],
}

impl<'a, const N: usize> ControlledPlan<'a, N> {
    /// The tile's host projections from the readback (past the health words), and the side.
    fn tile(&self, values: Vec<f32>) -> ProjectionTile<'a, N> {
        let mut values = values.into_iter().skip(3 * self.m);
        let host = (0..self.m)
            .map(|_| {
                self.host_widths
                    .map(|width| values.by_ref().take(width).collect())
            })
            .collect();
        (host, self.side)
    }
}

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention_batch_capacity<
        const N: usize,
    >(
        &self,
        [rank, hidden]: [usize; 2],
        weights: [Option<ResidentProjection<'_>>; N],
    ) -> Result<Option<usize>, EngError> {
        if !(1..=4096).contains(&hidden) || !(1..=4096).contains(&rank) {
            return Ok(None);
        }
        let mut output = 0usize;
        for p in weights.iter().flatten() {
            if p.input >= 3 || p.weight.shape[1] != hidden || p.weight.shape[0] == 0 {
                return Ok(None);
            }
            let Some(sum) = output.checked_add(p.weight.shape[0]) else {
                return Ok(None);
            };
            output = sum;
        }
        if output == 0 {
            return Ok(None);
        }
        let Some(per_row) = Self::controlled_attention_row_words([rank, hidden], output) else {
            return Ok(None);
        };
        let capacity =
            self.arena.tensor(Slot::Output as usize)?.bytes() / size_of::<f32>() / per_row;
        Ok(Some(
            capacity.min(DenseGemvShape::MAX_ACTIVATION_ROWS).min(512),
        ))
    }

    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention_row_words(
        [rank, hidden]: [usize; 2],
        output: usize,
    ) -> Option<usize> {
        hidden
            .checked_mul(24)?
            .checked_add(rank.checked_mul(3)?)?
            .checked_add(3)?
            .checked_add(output)
    }

    // Fused transaction for grouped prefill; the caller selects controller arithmetic.
    #[allow(clippy::type_complexity)]
    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention_batch<
        'a,
        const N: usize,
    >(
        &'a self,
        context: &CudaContext,
        inputs: &[[ControlledAttentionInput<'_>; 3]],
        controllers: [ControllerWeights<'a>; 3],
        weights: [Option<ResidentProjection<'a>>; N],
        (eps, arithmetic, keep, scales): (f32, Arithmetic, &[usize], Option<&[[[f32; 2]; 3]]>),
    ) -> Result<Option<ProjectionTile<'a, N>>, EngError> {
        let Some(first) = inputs.first() else {
            return Ok(None);
        };
        let hidden = first[0].input.len();
        let m = inputs.len();
        if scales.is_some_and(|s| s.len() != m)
            || inputs.iter().flatten().any(|r| {
                r.gate.len() != 2 * hidden
                    || [r.input.len(), r.message.len(), r.gamma.len()] != [hidden; 3]
            })
        {
            return Err(bad("invalid batched CUDA controlled attention shape"));
        }
        let Some(plan) = self.controlled_plan([m, hidden], controllers, weights, keep)? else {
            return Ok(None);
        };
        let range = |offset, len| self.range(Slot::Output, offset, len);
        let (gates, raw) = (plan.gates, plan.raw);
        // Use the start of the gain scratch for scales; the first kernel reads
        // them before any controller GEMV writes coefficients into that region.
        let scale_words = 6 * m;
        let upload = if scales.is_some() {
            range(gates, raw + scale_words)?
        } else {
            range(0, gates + raw)?
        };
        let mut staged = Vec::with_capacity(12 * m);
        if scales.is_none() {
            for role in 0..3 {
                for row in inputs {
                    staged.push(row[role].gate);
                }
            }
        }
        for role in 0..3 {
            for row in inputs {
                staged.extend([row[role].input, row[role].message, row[role].gamma]);
            }
        }
        if let Some(scales) = scales {
            for role in 0..3 {
                staged.extend(scales.iter().map(|row| row[role].as_slice()));
            }
        }
        let rebuild = scales
            .map(|_| range(0, gates + raw + scale_words))
            .transpose()?;
        let values = self.run_rows(
            context,
            &self.stream,
            &staged,
            [upload, plan.readback],
            || {
                if let Some(workspace) = rebuild {
                    context.inkling_rebuild_gates(&self.stream, workspace, [hidden, 3 * m])?;
                }
                self.enqueue_controlled(context, &plan, (eps, arithmetic))
            },
        )?;
        self.capture_first_controller_tile(m, hidden)?;
        // run_rows fenced the producer; the caller must consume the borrowed tile
        // before running another dense step that reuses Slot::Output.
        Ok(Some(plan.tile(values)))
    }

    /// [`Self::controlled_attention_batch`] of rows `first..first + count` of device planes:
    /// the tile's gates and raw triples are prepared on the device from each role's
    /// [input, message] pairs and norm weight, with no host rows.
    #[allow(clippy::type_complexity)]
    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention_device_batch<
        'a,
        const N: usize,
    >(
        &'a self,
        context: &CudaContext,
        (rows, (first, count)): (&ControlledRows<'_>, (usize, usize)),
        controllers: [ControllerWeights<'a>; 3],
        weights: [Option<ResidentProjection<'a>>; N],
        (eps, arithmetic, keep): (f32, Arithmetic, &[usize]),
    ) -> Result<Option<ProjectionTile<'a, N>>, EngError> {
        let ControlledRows {
            pairs,
            gammas,
            health,
            rows: total,
            hidden,
            gate_eps,
        } = *rows;
        if count == 0
            || first.checked_add(count).is_none_or(|end| end > total)
            || pairs.bytes() != 3 * total * 2 * hidden * size_of::<f32>()
            || health.bytes() != 3 * total * 2 * size_of::<f32>()
            || gammas
                .iter()
                .any(|g| g.bytes() != hidden * size_of::<f32>())
        {
            return Err(bad("invalid device CUDA controlled attention rows"));
        }
        let Some(plan) = self.controlled_plan([count, hidden], controllers, weights, keep)? else {
            return Ok(None);
        };
        let m = count;
        let words = size_of::<f32>();
        let values = self.run_device_rows(context, &self.stream, plan.readback, || {
            for (role, gamma) in gammas.iter().enumerate() {
                let rows_at = role * total + first;
                context.inkling_prepare_controller(
                    &self.stream,
                    [
                        pairs.slice(rows_at * 2 * hidden * words, m * 2 * hidden * words)?,
                        *gamma,
                        self.range(
                            Slot::Output,
                            plan.gates + role * m * 3 * hidden,
                            m * 3 * hidden,
                        )?,
                        self.range(Slot::Output, role * m * 2 * hidden, m * 2 * hidden)?,
                        health.slice(rows_at * 2 * words, m * 2 * words)?,
                    ],
                    ([hidden, m], gate_eps),
                )?;
            }
            self.enqueue_controlled(context, &plan, (eps, arithmetic))
        })?;
        Ok(Some(plan.tile(values)))
    }

    /// The arena layout and launches of one controlled attention tile of `m` rows, or none
    /// when the tile does not fit the output slot.
    fn controlled_plan<'a, const N: usize>(
        &'a self,
        [m, hidden]: [usize; 2],
        controllers: [ControllerWeights<'a>; 3],
        weights: [Option<ResidentProjection<'a>>; N],
        keep: &[usize],
    ) -> Result<Option<ControlledPlan<'a, N>>, EngError> {
        let rank = controllers[0][0].0.shape[0];
        if !(1..=4096).contains(&hidden)
            || rank == 0
            || rank > 4096
            || m > DenseGemvShape::MAX_ACTIVATION_ROWS
            || m > 512
        {
            return Ok(None);
        }
        if controllers.iter().any(|w| {
            w.map(|(x, _)| x.shape) != [[rank, 2 * hidden], [hidden, rank], [hidden, rank]]
        }) || weights
            .iter()
            .flatten()
            .any(|p| p.input >= 3 || p.weight.shape[1] != hidden)
        {
            return Err(bad("invalid batched CUDA controlled attention shape"));
        }
        let widths = weights.map(|p| p.map_or(0, |p| p.weight.shape[0]));
        let output_width = widths
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .ok_or_else(|| bad("batched CUDA attention output size overflows"))?;
        let per_row = Self::controlled_attention_row_words([rank, hidden], output_width)
            .ok_or_else(|| bad("batched CUDA attention workspace overflows"))?;
        let capacity =
            self.arena.tensor(Slot::Output as usize)?.bytes() / size_of::<f32>() / per_row;
        if m > capacity {
            return Ok(None);
        }
        let gates = 6 * m * hidden;
        let raw = 9 * m * hidden;
        let norm = gates;
        let reduced = 24 * m * hidden;
        let health_at = reduced + 3 * m * rank;
        let out_at = health_at + 3 * m;
        let range = |offset, len| self.range(Slot::Output, offset, len);
        let workspace = range(norm, 18 * m * hidden)?;
        let health = range(health_at, 3 * m)?;
        let mut host_widths = widths;
        let mut side_widths = [0; N];
        for &i in keep {
            side_widths[i] = std::mem::take(&mut host_widths[i]);
        }
        let ld_host: usize = host_widths.iter().sum();
        let ld_side: usize = side_widths.iter().sum();
        let side_at = out_at + m * ld_host;
        let side = (ld_side != 0)
            .then(|| range(side_at, m * ld_side))
            .transpose()?;
        let readback = range(health_at, 3 * m + m * ld_host)?;
        let mut control = Vec::with_capacity(9);
        for (role, w) in controllers.into_iter().enumerate() {
            let gate = role * m * 2 * hidden;
            let mid = reduced + role * m * rank;
            let gain = gates + raw + role * m * 2 * hidden;
            control.push(self.batch_call(
                w[0],
                DenseGemvShape::packed(m, rank, 2 * hidden),
                [gate, mid],
            )?);
            let mut shape = DenseGemvShape::packed(m, hidden, rank);
            shape.ld_y = 2 * hidden;
            control.push(self.batch_call(w[1], shape, [mid, gain])?);
            control.push(self.batch_call(w[2], shape, [mid, gain + hidden])?);
        }
        let mut projections = Vec::with_capacity(N);
        for (i, p) in weights.into_iter().enumerate() {
            let Some(p) = p else { continue };
            let mut shape = DenseGemvShape::packed(m, p.weight.shape[0], hidden);
            let (base, widths, stride) = if side_widths[i] > 0 {
                (side_at, &side_widths, ld_side)
            } else {
                (out_at, &host_widths, ld_host)
            };
            shape.ld_y = stride;
            let x = gates + 15 * m * hidden + p.input * m * hidden;
            let y = base + widths[..i].iter().sum::<usize>();
            projections.push(self.batch_call((p.weight, p.dtype), shape, [x, y])?);
        }
        Ok(Some(ControlledPlan {
            m,
            hidden,
            gates,
            raw,
            control,
            projections,
            workspace,
            health,
            readback,
            side,
            host_widths,
        }))
    }

    /// The controller GEMVs, the controlled norm and the projections of a staged tile.
    fn enqueue_controlled<const N: usize>(
        &self,
        context: &CudaContext,
        plan: &ControlledPlan<'_, N>,
        (eps, arithmetic): (f32, Arithmetic),
    ) -> Result<(), EngError> {
        context.dense_gemv_group(&self.stream, &plan.control)?;
        context.inkling_controller_norm_with_arithmetic(
            &self.stream,
            plan.workspace,
            plan.health,
            ([plan.hidden, 3 * plan.m], eps),
            arithmetic,
        )?;
        self.project_prompt_group(context, &plan.projections, Some("controlled_qkv"))
    }

    fn capture_first_controller_tile(&self, rows: usize, hidden: usize) -> Result<(), EngError> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static DONE: AtomicBool = AtomicBool::new(false);
        let Some(dir) = std::env::var_os("FIFI_INKLING_DEBUG_CONTROLLER_CAPTURE") else {
            return Ok(());
        };
        if DONE.swap(true, Ordering::Relaxed) {
            return Ok(());
        }
        let dir = std::path::PathBuf::from(dir);
        let slot = self.arena.tensor(Slot::Output as usize)?;
        let mut bytes = vec![0u8; 24 * rows * hidden * size_of::<f32>()];
        slot.buffer()
            .read_at(&self.stream, slot.offset(), &mut bytes)?;
        std::fs::write(dir.join("gpu.f32"), bytes)
            .map_err(|e| bad(format!("capture GPU tile: {e}")))?;
        std::fs::write(
            dir.join("meta.json"),
            serde_json::json!({"rows":rows,"hidden":hidden}).to_string(),
        )
        .map_err(|e| bad(format!("capture GPU tile metadata: {e}")))?;
        Ok(())
    }

    fn batch_call<'a>(
        &'a self,
        (weight, dtype): (&'a super::Resident, super::DenseDtype),
        shape: DenseGemvShape,
        [x, y]: [usize; 2],
    ) -> Result<DenseGemvCall<'a>, EngError> {
        let need = dense_gemv_bytes(shape, dtype)?;
        Ok(DenseGemvCall {
            shape,
            dtype,
            w: weight.codes.range(0, need.w)?,
            x: self.range(Slot::Output, x, need.x / size_of::<f32>())?,
            y: self.range(Slot::Output, y, need.y / size_of::<f32>())?,
        })
    }

    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention<const N: usize>(
        &self,
        context: &CudaContext,
        inputs: [ControlledAttentionInput<'_>; 3],
        controllers: [ControllerWeights<'_>; 3],
        weights: [Option<ResidentProjection<'_>>; N],
        eps: f32,
    ) -> Result<Option<[Vec<f32>; N]>, EngError> {
        let hidden = inputs[0].input.len();
        if !(1..=4096).contains(&hidden) {
            return Ok(None);
        }
        // The idle logits slot packs 6 gate rows, 9 raw input/message/gamma
        // rows, 6 controller outputs, 3 normalized inputs, then the readback.
        let stage = 15 * hidden;
        let health_at = 24 * hidden;
        let out_at = health_at + 3;
        let rows: usize = weights.iter().flatten().map(|p| p.weight.shape[0]).sum();
        let Ok(readback) = self.range(Slot::Output, health_at, 3 + rows) else {
            return Ok(None);
        };
        if inputs.iter().any(|row| {
            row.gate.len() != 2 * hidden
                || [row.input, row.message, row.gamma].map(<[f32]>::len) != [hidden; 3]
        }) {
            return Err(bad("invalid CUDA controlled attention input shape"));
        }
        let upload = self.range(Slot::Output, 0, stage)?;
        let workspace = self.range(Slot::Output, 6 * hidden, 18 * hidden)?;
        let health = self.range(Slot::Output, health_at, 3)?;
        let control = self.controlled_attention_calls(hidden, controllers)?;
        let (projections, _) =
            self.group_calls([21, 22, 23].map(|i| i * hidden), out_at, weights)?;
        let staged: Vec<_> = inputs
            .iter()
            .map(|row| row.gate)
            .chain(
                inputs
                    .iter()
                    .flat_map(|row| [row.input, row.message, row.gamma]),
            )
            .collect();
        let values = self.run_rows(context, &self.stream, &staged, [upload, readback], || {
            context.dense_gemv_group(&self.stream, &control)?;
            context.inkling_controller_norm(&self.stream, workspace, health, (hidden, eps))?;
            self.project_prompt_group(context, &projections, None)
        })?;
        Ok(Some(split_projections(values.into_iter().skip(3), weights)))
    }

    pub(super) fn controlled_attention_calls<'a>(
        &'a self,
        hidden: usize,
        controllers: [ControllerWeights<'a>; 3],
    ) -> Result<Vec<DenseGemvCall<'a>>, EngError> {
        let reduced = self.range(Slot::Reduced, 0, controllers[0][0].0.shape[0])?;
        let mut calls = Vec::with_capacity(9);
        for (role, weights) in controllers.into_iter().enumerate() {
            let gate = self.range(Slot::Output, role * 2 * hidden, 2 * hidden)?;
            let gain = self.range(Slot::Output, (15 + 2 * role) * hidden, hidden)?;
            let forget = self.range(Slot::Output, (16 + 2 * role) * hidden, hidden)?;
            calls.extend(self.controller_calls(weights, [gate, reduced, gain, forget])?);
        }
        Ok(calls)
    }
}
