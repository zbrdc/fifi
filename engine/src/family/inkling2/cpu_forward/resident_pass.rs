//! The device-resident prefill pass (`FIFI_INKLING_CUDA_RESIDENT_PREFILL_PASS`, default off).
//!
//! One block of prompt rows becomes one device plan: every visit runs from device memory and
//! the host reads back once per plane at the end. A block that is not admitted falls through
//! to the flag-off path unchanged, so nothing here changes the host path or its interfaces.
//!
//! Host step by host step (`INVARIANTS.md` of the pass lane is the contract):
//!
//! | host step | device operation |
//! |---|---|
//! | embeddings | one upload |
//! | `Stream::mix` / `seal` / `add_current` | `inkling_pass_mix` / `_seal` / `_add` |
//! | `control` + `norm` (`attention_inputs_scalar`, `finish_mlp_input`) | `inkling_pass_gate`, the controller group, `inkling_pass_apply` |
//! | attention projections | `q8_row_reuse::project_group` into sidecar / raw / dynamic planes |
//! | `prepare_attention_row` (clip, convolutions, clip, head norms) | `relative_attention_clip_queries`, `inkling_pass_prep` |
//! | attention + ring rows | `relative_attention_tile` per `QUERY_TILE_LIMIT` rows |
//! | output projection + `attn_sconv` | `q8_row_reuse::project_output`, `inkling_output_conv` |
//! | router | F32 `dense_gemv` of `gate.weight`; the logits are read back and the host selects |
//! | routed experts | `prefill_device_experts_unfenced`: the GEMM lane's one call site |
//! | `mlp_norm` | inside `inkling_pass_apply` |
//! | `mlp_sconv` + `add_current` | `inkling_output_conv`, `inkling_pass_add` |
//! | convolution histories | tail planes per visit (`inkling_pass_tails`) |
//!
//! One fence ends the block (health, every status word, both flag words and the tails are
//! checked); the K/V rows follow from the visit rings. The host `VisitState` and the device
//! cursors both end the block holding the rows.
//!
//! A prompt is cut into blocks of at most [`BLOCK_ROWS`] rows; blocks of 1 to [`MAX_ROWS`] rows
//! share one plan and only the `rows` argument of each launch changes. A single row runs as a block of two (the token twice), because the attention tile
//! refuses one query row; the padded row is never committed (tails, ring cursors and the rows
//! published take the one). The router logits are the one readback inside a block: the
//! routed-expert operation takes host route ids.

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
use super::MoeRoute;
use std::ops::Range;

use super::super::cpu_attention::VisitState;
use super::super::cpu_weights::Projection;
use super::super::cuda_projection::gguf::ExpertPrefillRows;
use super::super::cuda_projection::gguf::attention::TileParts;
use super::super::cuda_projection::q8_row_reuse;
use super::{
    ATTENTION_ROLES, Attention, CpuForward, LayerVisit, MIX_EPS, ROUTE_NORMALIZATION_FLOOR, Role,
    Weights, attention_norm_name, bad, control_names,
};
use crate::EngError;
use crate::family::inkling2::cuda_projection::gguf::GgufCuda;
use crate::family::inkling2::gguf_weights::GgufTextWeights;
use crate::kernels::cuda::resident::dense_gemv::routed_prefill::{
    DeviceRoutedPrefill, routed_prefill_scratch_bytes,
};
use crate::kernels::cuda::resident::dense_gemv::{
    DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes,
};
use crate::kernels::cuda::resident::inkling_local_prep::OutputConvShape;
use crate::kernels::cuda::resident::inkling_pass::MAX_BLOCKS;
use crate::kernels::cuda::resident::inkling_pass::{
    PassAddCall, PassMixCall, PassPrepCall, PassSealCall, TAPS,
};
use crate::kernels::cuda::resident::relative_attention::{
    HEAD_DIM, QUERY_TILE_LIMIT, RelativeAttentionCall, RelativeAttentionTile,
};
use crate::kernels::cuda::resident::router::{
    RouterLogitsCall, RouterShape, RouterSigmoidBias, RouterSigmoidCall,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf};

/// Whether the pass is enabled; read once per process.
pub(super) fn enabled() -> bool {
    crate::family::inkling2::switches::Switch::ResidentPrefillPass.on()
}

/// Rows one block takes at most: the output-convolution, clip and router kernels bound their
/// row counts at 512.
pub(super) const MAX_ROWS: usize = 512;
/// Rows a prompt is cut into while it is longer. The weight-visit projection kernels of this tree
/// exist for 8, 12, 64, 128 and 256 rows (`inkling_prefetch_select`, `inkling_large_select`);
/// any other count falls back to a GEMV that reads the weights once per row. On the RTX 2080 Ti
/// (nsys traces `pass-nsys-pass-{256,512}-50a9e761-*`) a 512-row block takes 2.85 s of stream
/// time against 2 x 0.65 s for two 256-row blocks, and cuts smaller than 256 pay the expert
/// stage's per-block cost (about 250 ms) more often. Tests can override it with `FIFI_INKLING_PASS_BLOCK_ROWS`
/// (1..=MAX_ROWS); revisit once every row count has a weight-visit or tensor kernel.
pub(super) const BLOCK_ROWS: usize = 256;

const WORD: usize = 4;
const PAST: usize = TAPS - 1;
const ROLES: usize = 3;
const STREAM_BLOCKS: usize = MAX_BLOCKS;
/// The buffer index of the frozen prelude stream.
const PRELUDE: usize = 2;
/// Requests of `attention_requests`: q, k, v, relative, sinks, gates, k taps, v taps.
const REQUEST_INPUTS: [usize; 8] = [0, 1, 2, 0, 0, 0, 1, 2];

/// The pass lane's device plan: per-visit constants, the arena every visit overwrites, the
/// depth streams, and the tail planes of the previous block.
pub(super) struct PassWorkspace {
    visits: Vec<VisitPlan>,
    rows: usize,
    arena: DeviceBuffer,
    layout: Arena,
    constants: DeviceBuffer,
    /// The depth streams: the live stream, the bank of the current pass and the prelude.
    current: [DeviceBuffer; 3],
    sealed: [DeviceBuffer; 3],
    /// Raw convolution tails per visit, oldest first: K, V, attention, MLP (word ranges of the
    /// tail buffers). The device flips
    /// them at the end of a block; the host uploads them when they are stale.
    tails: [DeviceBuffer; 2],
    tails_from: usize,
    tails_valid: bool,
    readback: PinnedBuf,
    stream: DeviceStream,
}

struct VisitPlan {
    kind: usize,
    /// Tail planes of this visit: `[K, V, attention, MLP]`, `PAST` rows each.
    tails: Range<usize>,
    consts: VisitConsts,
    /// `[query heads, kv heads]`.
    heads: [usize; 2],
}

/// Byte ranges into the constants buffer of one visit.
struct VisitConsts {
    /// `attn_res.attn_queries` rows per role (the residual mix in front of attention).
    mix: [Range<usize>; ROLES],
    /// Control queries per role plus the MLP role.
    control: [Range<usize>; 4],
    norms: [Range<usize>; ROLES],
    clip: Range<usize>,
    conv_k: Range<usize>,
    conv_v: Range<usize>,
    attn_sconv: Range<usize>,
    mlp_mix: Range<usize>,
    mlp_norm: Range<usize>,
    mlp_sconv: Range<usize>,
    router_bias: Range<usize>,
}

/// Word offsets into the arena. Plane row `r` of a strided plane starts at `base + r * stride`.
#[derive(Clone, Copy)]
struct Arena {
    visits: usize,
    health: usize,
    status: usize,
    flags: usize,
    logits: usize,
    /// Device router picks per MLP visit: `[rows][top_k]` i32 ids and f32 weights.
    ids: usize,
    route_weights: usize,
    routed_scratch: usize,
    inputs: usize,
    messages: usize,
    gate: usize,
    reduced: usize,
    gain: usize,
    forget: usize,
    sidecar: [usize; 2],
    sidecar_stride: [usize; 2],
    raw: [usize; 2],
    raw_stride: [usize; 2],
    dynamic: [usize; 2],
    dynamic_stride: [usize; 2],
    side: [usize; 2],
    side_stride: [usize; 2],
    out: [usize; 2],
    attn_raw: usize,
    attn_conv: usize,
    mlp_input: usize,
    expert_out: usize,
    mlp_conv: usize,
    words: usize,
}

/// One attention kind's geometry.
#[derive(Clone, Copy)]
struct Geometry {
    query: usize,
    kv: usize,
    channels: usize,
    kv_channels: usize,
    relative: usize,
}

/// Whether the block report line is wanted (`FIFI_INKLING_RESIDENT_PASS_REPORT`).
fn reporting() -> bool {
    crate::family::inkling2::switches::Switch::ResidentPassReport.on()
}

impl CpuForward {
    /// Runs one prompt through the device plan in blocks of at most [`BLOCK_ROWS`] rows when
    /// admitted; `Ok(false)` leaves the sequence untouched for the flag-off path.
    pub(super) fn resident_pass(&mut self, tokens: &[u32]) -> Result<bool, EngError> {
        if !enabled() || self.draft_window_enabled() || !self.taps.is_empty() {
            return Ok(false);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(false);
        };
        let Some(cuda) = weights.cuda() else {
            return Ok(false);
        };
        if !self.pass_admitted(weights, cuda)? {
            return Ok(false);
        }
        let mut offset = 0;
        for block in tokens.chunks(Self::block_rows()) {
            let rows = block.len();
            let mut state = std::mem::take(&mut self.state);
            let started = std::time::Instant::now();
            let outcome = self.pass_workspace(cuda, MAX_ROWS).and_then(|()| {
                let plan_us = started.elapsed().as_micros() as u64;
                self.run_block(weights, cuda, block, offset, &mut state)
                    .inspect(|report| report.plan_us.set(plan_us))
            });
            self.state = state;
            match outcome {
                Ok(report) => {
                    self.position += rows;
                    offset += rows;
                    self.failed = false;
                    if reporting() {
                        eprintln!(
                            "Inkling CUDA resident pass {}",
                            serde_json::json!({"rows": rows, "launches": report.launches.get(),
                                "readbacks": report.readbacks.get(), "plan_us": report.plan_us.get(),
                                "launch_us": report.launch_us.get(), "fence_us": report.fence_us.get(),
                                "publish_us": report.publish_us.get(), "block_us": report.block_us.get()})
                        );
                    }
                }
                Err(error) => {
                    self.invalidate_device_attention();
                    return Err(error);
                }
            }
        }
        Ok(true)
    }

    /// Serving uses the measured block size, with no additional process-level tuning knob.
    #[cfg(not(test))]
    fn block_rows() -> usize {
        BLOCK_ROWS
    }

    /// Test override to exercise partial and multi-block tails.
    #[cfg(test)]
    fn block_rows() -> usize {
        static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *ROWS.get_or_init(|| {
            std::env::var("FIFI_INKLING_PASS_BLOCK_ROWS")
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|&rows| (1..=MAX_ROWS).contains(&rows))
                .unwrap_or(BLOCK_ROWS)
        })
    }

    /// Whether a visit reads the message bank and so runs the depth-mix controllers: the passes
    /// after the first over the core layers (the prelude pass and the suffix layers have none).
    fn pass_visit_banked(&self, slot: usize) -> bool {
        let core = &self.spec.core;
        match core.core_end.checked_sub(core.core_start) {
            Some(width) if width > 0 => {
                let end = core.core_start + width * core.num_passes.max(1);
                (core.core_start + width..end).contains(&slot)
            }
            _ => false,
        }
    }

    /// Names one refusal of the pass once per process when
    /// `FIFI_INKLING_RESIDENT_PASS_REPORT` is set, so a silent fall-through is diagnosable.
    fn pass_refused(&self, reason: std::fmt::Arguments<'_>) -> bool {
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            if reporting() {
                eprintln!("Inkling resident pass refused: {reason}");
            }
        });
        false
    }

    fn pass_admitted(&self, weights: &GgufTextWeights, cuda: &GgufCuda) -> Result<bool, EngError> {
        let core = &self.spec.core;
        let width = core.core_end.saturating_sub(core.core_start);
        let Some(end) = width
            .checked_mul(core.num_passes.max(1))
            .and_then(|n| n.checked_add(core.core_start))
        else {
            return Ok(self.pass_refused(format_args!("core geometry")));
        };
        let checks = [
            ("math", self.math.validate(self.context, &self.spec).is_ok()),
            (
                "core range",
                core.core_start < core.core_end && core.num_passes > 0,
            ),
            (
                "visit count",
                end <= self.visits.len() && self.state.len() == self.visits.len(),
            ),
            ("resident attention", cuda.resident_attention().is_some()),
        ];
        for (name, ok) in checks {
            if !ok {
                return Ok(self.pass_refused(format_args!("{name}")));
            }
        }
        for (slot, state) in self.state.iter().enumerate() {
            let certified = state.position() == self.position && !state.is_resident();
            let parts = cuda
                .resident_attention()
                .is_some_and(|(_, attention)| attention.tile_parts(slot).is_some());
            if !certified || !parts {
                return Ok(self.pass_refused(format_args!(
                    "visit {slot}: {}",
                    if !certified {
                        "state position, or resident rows"
                    } else {
                        "no ring"
                    }
                )));
            }
        }
        for (index, &visit) in self.visits.iter().enumerate() {
            if !self.pass_visit_admitted(weights, cuda, visit, index)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn pass_visit_admitted(
        &self,
        weights: &GgufTextWeights,
        cuda: &GgufCuda,
        visit: LayerVisit,
        index: usize,
    ) -> Result<bool, EngError> {
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let h = self.hidden();
        let heads = self.pass_heads(visit.attention);
        let dynamic = visit.attention == Attention::Global;
        let refused =
            |why: &str| self.pass_refused(format_args!("visit {index} (layer {physical}): {why}"));
        let Some(requests) = self.pass_requests(&layer, visit.attention, heads) else {
            return Ok(refused("projection geometry"));
        };
        let requests: [Option<Projection<'_>>; 8] = requests.each_ref().map(|r| {
            r.as_ref().map(|p| Projection {
                name: &p.name,
                input: p.input,
                rows: p.rows,
            })
        });
        let Some(Ok(projections)) = cuda.attention_group([h; 3], requests) else {
            return Ok(refused("attention weights not grouped"));
        };
        if projections.iter().flatten().count() != if dynamic { 8 } else { 6 } {
            return Ok(refused("attention projection shapes"));
        }
        let output = format!("{layer}.attn.wo_ud.weight");
        if !matches!(
            cuda.relative_attention_projection(&output, [h, heads[0] * HEAD_DIM]),
            Ok(Some(projection)) if projection.rows == h
        ) {
            return Ok(refused("attention output projection"));
        }
        if self.pass_visit_banked(visit.state_slot) {
            let rank = self.spec.core.gate_rank as usize;
            for role in [Role::Query, Role::Key, Role::Value, Role::Mlp] {
                let names = control_names(&layer, role);
                if !cuda
                    .controller_weights(names.each_ref().map(String::as_str), 2 * h, [rank, h])
                    .is_some_and(|result| result.is_ok())
                {
                    return Ok(refused(&format!(
                        "{} controller weights not resident",
                        role.name()
                    )));
                }
            }
        }
        let prefix = format!("{layer}.mlp");
        let experts = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let admitted = weights.admit_resident_prefill_experts(
            experts.each_ref().map(String::as_str),
            physical as u32,
            [
                MAX_ROWS,
                self.spec.num_experts_per_tok as usize,
                h,
                self.spec.n_routed_experts as usize,
            ],
            [self.math.mlp_gate_softcap, self.math.mlp_up_softcap],
        );
        if !matches!(admitted, Ok(true)) {
            return Ok(refused("expert window not admitted for 512 rows"));
        }
        if !cuda
            .router_weight(&format!("{prefix}.gate.weight"))
            .is_some_and(|(_, dtype, shape)| {
                dtype == DenseDtype::F32 && shape == [self.spec.n_routed_experts as usize, h]
            })
        {
            return Ok(refused("router weight"));
        }
        Ok(true)
    }

    fn pass_heads(&self, attention: Attention) -> [usize; 2] {
        match attention {
            Attention::Local => [
                self.spec.swa_num_attention_heads as usize,
                self.spec.swa_num_key_value_heads as usize,
            ],
            Attention::Global => [
                self.spec.num_attention_heads as usize,
                self.spec.num_key_value_heads as usize,
            ],
        }
    }

    /// The projection requests of one visit, in the order of `attention_requests`.
    fn pass_requests(
        &self,
        layer: &str,
        attention: Attention,
        heads: [usize; 2],
    ) -> Option<[Option<OwnedProjection>; 8]> {
        let taps = self.math.sconv_kernel_size.checked_mul(heads[1])?;
        let dynamic = attention == Attention::Global;
        let names = [
            "wq_du.weight",
            "wk_dv.weight",
            "wv_dv.weight",
            "wr_du.weight",
            "attn_sink_proj.weight",
            "gate_attn.weight",
            "k_sconv.weight_proj",
            "v_sconv.weight_proj",
        ];
        let rows = [
            heads[0] * HEAD_DIM,
            heads[1] * HEAD_DIM,
            heads[1] * HEAD_DIM,
            heads[0] * self.math.d_rel,
            heads[0],
            heads[0] * HEAD_DIM,
            taps,
            taps,
        ];
        Some(std::array::from_fn(|i| {
            (i < 6 || dynamic).then(|| OwnedProjection {
                name: format!("{layer}.attn.{}", names[i]),
                input: REQUEST_INPUTS[i],
                rows: rows[i],
            })
        }))
    }

    fn pass_workspace(&self, cuda: &GgufCuda, rows: usize) -> Result<(), EngError> {
        let mut pass = self.pass.borrow_mut();
        if pass.as_ref().is_none_or(|plan| plan.rows < rows) {
            *pass = Some(PassWorkspace::plan(self, cuda, rows)?);
        }
        Ok(())
    }

    /// Marks the device tails stale: the host state moved under them.
    pub(in crate::family::inkling2) fn pass_tails_stale(&self) {
        if let Some(plan) = self.pass.borrow_mut().as_mut() {
            plan.tails_valid = false;
        }
    }

    fn run_block(
        &self,
        weights: &GgufTextWeights,
        cuda: &GgufCuda,
        tokens: &[u32],
        offset: usize,
        state: &mut [VisitState],
    ) -> Result<Report, EngError> {
        let committed = tokens.len();
        // One row runs as two (the same embedding twice): the attention tile refuses a single
        // query row. Rows are independent inside a block (attention is causal), so the first
        // row is what a one-row block computes; the padded row is never committed.
        let rows = committed.max(2);
        let position = self.position;
        let h = self.hidden();
        let mut report = Report::default();
        let started = std::time::Instant::now();
        let mut pass = self.pass.borrow_mut();
        let plan = pass.as_mut().expect("admitted plan");
        let stream: &DeviceStream = &plan.stream;
        let readback: &mut PinnedBuf = &mut plan.readback;
        let arena = &plan.arena;
        let l = &plan.layout;
        let context = cuda.context();
        let mut block = Block {
            context,
            stream,
            arena,
            l,
            constants: &plan.constants,
            current: &plan.current,
            sealed: &plan.sealed,
            tails: [
                &plan.tails[plan.tails_from],
                &plan.tails[1 - plan.tails_from],
            ],
            visits: &plan.visits,
            rows,
            committed,
            offset,
            h,
            live: std::cell::Cell::new(0),
            report: &mut report,
        };
        let zero_words = rows + l.visits * rows + 2 * l.visits;
        context.inkling_pass_zero(stream, block.at(l.health, zero_words)?, zero_words)?;
        block.report.launches.set(block.report.launches.get() + 1);

        // Embeddings, and the tails of the previous block when the host state owns them.
        let mut embedded = Vec::with_capacity(rows * h);
        for id in tokens
            .iter()
            .chain(tokens.last().filter(|_| committed == 1))
        {
            embedded.extend_from_slice(self.embed(*id)?.current());
        }
        plan.current[0].write_at(stream, 0, &bytes(&embedded))?;
        if !plan.tails_valid {
            let mut tails = Vec::new();
            for (visit, consts) in self.visits.iter().zip(&plan.visits) {
                let histories = state[visit.state_slot].histories();
                let widths = [consts.heads[1] * HEAD_DIM, consts.heads[1] * HEAD_DIM, h, h];
                for (history, width) in histories.into_iter().zip(widths) {
                    tails.extend(tail_bytes(&history, width));
                }
            }
            plan.tails[plan.tails_from].write_at(stream, 0, &tails)?;
            plan.tails_valid = true;
        }
        for kind in 0..2 {
            context.inkling_pass_positions(
                stream,
                block.at(l.side[kind], rows * l.side_stride[kind])?,
                [
                    rows,
                    l.side_stride[kind],
                    2 * l.raw_stride[kind] / 2,
                    position,
                ],
            )?;
            block.report.launches.set(block.report.launches.get() + 1);
        }

        let core = &self.spec.core;
        let width = core.core_end - core.core_start;
        let end = core.core_start + width * core.num_passes;
        let mut live = 0usize;
        let blocks_pre = 0usize;
        let mut blocks = 0usize;
        let mut prelude: Option<(usize, usize)> = None;
        let _ = blocks_pre;
        let mut bank: Option<(usize, usize, usize)> = None;
        // With `FIFI_INKLING_RESIDENT_PASS_REPORT`, two timing events bracket the sweep: the
        // stream time from the first launch to the last kernel, read once the fence has completed.
        let block_start = if reporting() {
            let start = context.event_timing()?;
            stream.record_timing(&start)?;
            Some(start)
        } else {
            None
        };
        let attention = cuda.resident_attention().expect("admitted").1;
        for (index, &visit) in self.visits.iter().enumerate() {
            let slot = visit.state_slot;
            let physical = visit.weight_layer;
            let layer = format!("model.llm.layers.{physical}");
            if (core.core_start..end).contains(&slot)
                && (slot - core.core_start).is_multiple_of(width)
            {
                let pass_index = (slot - core.core_start) / width;
                if pass_index == 0 {
                    // The prelude is a frozen copy of the stream at the start of pass 0.
                    block.restore_stream(PRELUDE, live, blocks)?;
                    prelude = Some((PRELUDE, blocks));
                    block.report.launches.set(block.report.launches.get() + 2);
                } else {
                    let (prelude_buffer, prelude_blocks) = prelude.expect("pass 0 ran");
                    bank = Some((live, blocks, prelude_blocks + 1));
                    live = 3 - live - PRELUDE;
                    block.live.set(live);
                    blocks = prelude_blocks;
                    block.restore_stream(live, prelude_buffer, blocks)?;
                    block.report.launches.set(block.report.launches.get() + 2);
                }
            } else if slot == end {
                bank = None;
            }
            let attention = visit.attention;
            let Some(parts) = cuda
                .resident_attention()
                .and_then(|(_, a)| a.tile_parts(slot))
            else {
                return Err(bad("the resident pass lost a visit ring"));
            };
            self.pass_attention(
                &mut block, cuda, &parts, &layer, attention, index, position, live, blocks, bank,
            )?;
            let attn_conv = block.at(l.attn_conv, rows * h)?;
            let health = block.at(l.health, rows)?;
            let sealed = block.sealed_range(live, blocks + 1)?;
            let current = block.current_range(live)?;
            if physical.is_multiple_of(self.spec.n_layers_per_attn_res_block) {
                context.inkling_pass_seal(
                    stream,
                    PassSealCall {
                        sealed,
                        current,
                        next: attn_conv,
                        health,
                    },
                    [rows, blocks, h],
                    MIX_EPS,
                )?;
                blocks += 1;
            } else {
                context.inkling_pass_add(
                    stream,
                    PassAddCall {
                        current,
                        row: attn_conv,
                        health,
                    },
                    [rows, h],
                )?;
            }
            block.report.launches.set(block.report.launches.get() + 1);
            let mlp_input = block.at(l.mlp_input, rows * h)?;
            let mlp_mix = block.visits[index].consts.mlp_mix.clone();
            block.mix(live, blocks, 0, &mlp_mix, h, mlp_input)?;
            self.pass_mlp(
                &mut block, readback, weights, cuda, &layer, index, position, bank,
            )?;
        }
        let block_end = if block_start.is_some() {
            let end = context.event_timing()?;
            stream.record_timing(&end)?;
            Some(end)
        } else {
            None
        };
        block
            .report
            .launch_us
            .set(started.elapsed().as_micros() as u64);
        let fenced = std::time::Instant::now();
        self.pass_fence(&mut block, readback, rows)?;
        block
            .report
            .fence_us
            .set(fenced.elapsed().as_micros() as u64);
        if let (Some(start), Some(end)) = (block_start.as_ref(), block_end.as_ref()) {
            let us = start.elapsed_ms_since(end)? * 1e3;
            block.report.block_us.set(us.round() as u64);
        }
        let published = std::time::Instant::now();
        self.pass_publish(&mut block, attention, state, position)?;
        for slot in 0..self.visits.len() {
            cuda.resident_attention()
                .expect("admitted")
                .1
                .certify(slot, position + committed);
        }
        block
            .report
            .publish_us
            .set(published.elapsed().as_micros() as u64);
        plan.tails_from = 1 - plan.tails_from;
        Ok(report)
    }

    fn pass_fence(
        &self,
        block: &mut Block<'_>,
        readback: &mut PinnedBuf,
        rows: usize,
    ) -> Result<(), EngError> {
        let l = block.l;
        let health_bytes = rows * WORD;
        let status_words = l.visits * rows + 2 * l.visits;
        let status_bytes = status_words * WORD;
        if readback.len() < health_bytes + status_bytes {
            return Err(bad("the resident pass readback plane is too small"));
        }
        block
            .context
            .copy_to_pinned(block.stream, block.at(l.health, rows)?, readback, 0)?;
        block.context.copy_to_pinned(
            block.stream,
            block.at(l.status, status_words)?,
            readback,
            health_bytes,
        )?;
        readback.synchronize(block.stream)?;
        // Health words are f32 (zero when every row of every launch was finite); the status
        // and flag words are u32 read as bits. Any nonzero word invalidates the block.
        let mut words = vec![0f32; rows + status_words];
        readback.read_f32(0, &mut words)?;
        block.report.readbacks.set(block.report.readbacks.get() + 2);
        if words.iter().any(|w| w.to_bits() != 0) {
            return Err(bad("non-finite or invalid CUDA resident prefill pass"));
        }
        Ok(())
    }

    fn pass_publish(
        &self,
        block: &mut Block<'_>,
        attention: &crate::family::inkling2::cuda_projection::gguf::attention::ResidentAttention,
        state: &mut [VisitState],
        position: usize,
    ) -> Result<(), EngError> {
        let mut tails = vec![0u8; PAST * 2 * block.h * WORD];
        for (visit, consts) in self.visits.iter().zip(block.visits) {
            let slot = visit.state_slot;
            let (keys, values) =
                attention.download_ring(block.stream, slot, position + block.committed)?;
            let widths = [
                consts.heads[1] * HEAD_DIM,
                consts.heads[1] * HEAD_DIM,
                block.h,
                block.h,
            ];
            let mut planes: Vec<Vec<Vec<f32>>> = Vec::with_capacity(4);
            let mut at = consts.tails.start * WORD;
            for width in widths {
                let bytes = PAST * width * WORD;
                block.tails[1].read_at(block.stream, at, &mut tails[..bytes])?;
                // The tail plane holds its rows oldest first, one history row per `width` words.
                planes.push(
                    tails[..bytes]
                        .chunks_exact(width * WORD)
                        .map(words)
                        .collect(),
                );
                at += bytes;
            }
            let [k, v, attn, mlp] = <[Vec<Vec<f32>>; 4]>::try_from(planes).unwrap();
            state[slot].publish_pass(
                (keys, values),
                [k, v, attn, mlp],
                (
                    PAST.min(position + block.committed),
                    self.kv_capacity(visit.attention),
                    position + block.committed,
                ),
            );
            block.report.readbacks.set(block.report.readbacks.get() + 3);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn pass_attention(
        &self,
        block: &mut Block<'_>,
        cuda: &GgufCuda,
        parts: &TileParts<'_>,
        layer: &str,
        attention: Attention,
        index: usize,
        position: usize,
        live: usize,
        blocks: usize,
        bank: Option<(usize, usize, usize)>,
    ) -> Result<(), EngError> {
        let l = block.l;
        let rows = block.rows;
        let h = block.h;
        let kind = block.visits[index].kind;
        let consts = &block.visits[index].consts;
        let [query, kv] = block.visits[index].heads;
        let channels = query * HEAD_DIM;
        let kv_channels = kv * HEAD_DIM;
        let local = kind == 0;
        let eps = self.math.rms_norm_eps;
        let rank = self.spec.core.gate_rank as usize;

        // Attention inputs: residual mix, control, norm, per role.
        for (role, ((norm_range, mix), control)) in consts
            .norms
            .iter()
            .zip(&consts.mix)
            .zip(&consts.control)
            .enumerate()
        {
            if local && role > 0 && bank.is_none() {
                continue;
            }
            let input = block.at(l.inputs + role * rows * h, rows * h)?;
            let norm = block.constant(norm_range)?;
            let norm_only = || -> Result<(), EngError> {
                block.context.inkling_pass_apply(
                    block.stream,
                    [
                        input,
                        input,
                        input,
                        input,
                        norm,
                        input,
                        block.at(l.health, rows)?,
                    ],
                    [rows, h],
                    eps,
                    false,
                )?;
                block.report.launches.set(block.report.launches.get() + 1);
                Ok(())
            };
            // The first layer has no residual mix: its input is the normed current row. Every
            // other visit mixes first and takes the norm below (or the controller's, if banked).
            let from_current = mix.is_empty();
            if from_current {
                let current = block.current_range(live)?;
                block.context.inkling_pass_apply(
                    block.stream,
                    [
                        current,
                        current,
                        current,
                        current,
                        norm,
                        input,
                        block.at(l.health, rows)?,
                    ],
                    [rows, h],
                    eps,
                    false,
                )?;
                block.report.launches.set(block.report.launches.get() + 1);
            } else {
                block.mix(live, blocks, 0, mix, h, input)?;
            }
            if let Some((bank_buffer, bank_blocks, bank_first)) = bank {
                let message = block.at(l.messages + role * rows * h, rows * h)?;
                block.mix(bank_buffer, bank_blocks, bank_first, control, h, message)?;
                let gate = block.at(l.gate, rows * 2 * h)?;
                block.context.inkling_pass_gate(
                    block.stream,
                    [input, message, gate, block.at(l.health, rows)?],
                    [rows, h, 2 * h],
                    MIX_EPS,
                )?;
                let names = control_names(layer, ATTENTION_ROLES[role].0);
                self.pass_controller(
                    block,
                    cuda,
                    names.each_ref().map(String::as_str),
                    gate,
                    rank,
                )?;
                let gain = block.at(l.gain, rows * h)?;
                let forget = block.at(l.forget, rows * h)?;
                block.context.inkling_pass_apply(
                    block.stream,
                    [
                        input,
                        message,
                        gain,
                        forget,
                        norm,
                        input,
                        block.at(l.health, rows)?,
                    ],
                    [rows, h],
                    eps,
                    true,
                )?;
                block.report.launches.set(block.report.launches.get() + 2);
            } else if !from_current {
                norm_only()?;
            }
        }
        #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
        block.dump_control(index, bank)?;

        // Projections into the sidecar, raw and dynamic planes.
        let heads = [query, kv];
        let Some(requests) = self.pass_requests(layer, attention, heads) else {
            return Err(bad("invalid CUDA resident pass attention geometry"));
        };
        let requests: [Option<Projection<'_>>; 8] = requests.each_ref().map(|r| {
            r.as_ref().map(|p| Projection {
                name: &p.name,
                input: p.input,
                rows: p.rows,
            })
        });
        let projections = cuda
            .attention_group([h; 3], requests)
            .ok_or_else(|| bad("the resident pass needs grouped attention weights"))??;
        let sidecar_stride = l.sidecar_stride[kind];
        let raw_stride = l.raw_stride[kind];
        let dynamic_stride = l.dynamic_stride[kind];
        let relative = query * self.math.d_rel;
        let mut calls = Vec::with_capacity(8);
        for (i, (request, projection)) in requests.iter().zip(&projections).enumerate() {
            let (Some(request), Some(projection)) = (request, projection) else {
                continue;
            };
            let width = projection.weight.shape[0];
            let (y, ld_y) = match i {
                0 => (l.sidecar[kind], sidecar_stride),
                1 => (l.raw[kind], raw_stride),
                2 => (l.raw[kind] + kv_channels, raw_stride),
                3 => (l.sidecar[kind] + channels, sidecar_stride),
                4 => (l.sidecar[kind] + channels + relative, sidecar_stride),
                5 => (
                    l.sidecar[kind] + channels + relative + query,
                    sidecar_stride,
                ),
                6 => (l.dynamic[kind], dynamic_stride),
                _ => (l.dynamic[kind] + kv * TAPS, dynamic_stride),
            };
            let x_role = if local && bank.is_none() {
                0
            } else {
                request.input
            };
            let shape = DenseGemvShape {
                m: rows,
                rows: width,
                cols: h,
                ld_x: h,
                ld_y,
            };
            let need = dense_gemv_bytes(shape, projection.dtype)?;
            calls.push(DenseGemvCall {
                shape,
                dtype: projection.dtype,
                w: projection.weight.codes.range(0, need.w)?,
                x: block.at(l.inputs + x_role * rows * h, rows * h)?,
                y: block.at(y, (rows - 1) * ld_y + width)?,
            });
        }
        q8_row_reuse::project_group(block.context, block.stream, &calls, "pass_attention")?;
        block
            .report
            .launches
            .set(block.report.launches.get() + calls.len());
        // The raw K/V rows of this visit become convolution history for the next block.
        let past = PAST.min(position);
        for (source, which) in [(l.raw[kind], 0usize), (l.raw[kind] + kv_channels, 1)] {
            block.context.inkling_pass_tails(
                block.stream,
                [
                    block.at(source, rows * raw_stride)?,
                    block.tail(0, index, which)?,
                    block.tail(1, index, which)?,
                ],
                // The raw plane rows hold K and V side by side.
                [block.committed, raw_stride, kv_channels, past],
            )?;
        }

        // Clip the packed queries, then prepare K/V (convolutions, norms, clip).
        let status = block.at(l.status + index * rows, rows)?;
        block.context.relative_attention_clip_queries(
            block.stream,
            parts.shape,
            [rows, 0], // Raw K/V is in its own plane, not in the sidecar.
            [
                self.math.qk_l2_clip_threshold,
                self.math.rel_l2_clip_threshold,
            ],
            [
                block.at(l.sidecar[kind], rows * sidecar_stride)?,
                block.constant(&consts.clip)?,
                status,
            ],
        )?;
        let dynamic = kind == 1;
        block.context.inkling_pass_prep(
            block.stream,
            PassPrepCall {
                raw: block.at(l.raw[kind], rows * raw_stride)?,
                // The prep entry takes the exact extents it reads: the last row of `side` stops
                // after V (the position word is written by `inkling_pass_positions`).
                side: block.at(
                    l.side[kind],
                    (rows - 1) * l.side_stride[kind] + 2 * kv_channels,
                )?,
                history: [block.tail(0, index, 0)?, block.tail(0, index, 1)?],
                kernel: [
                    block.constant(&consts.conv_k)?,
                    block.constant(&consts.conv_v)?,
                ],
                dynamic: block.at(l.dynamic[kind], (rows - 1) * dynamic_stride + 2 * kv * TAPS)?,
                status,
            },
            rows,
            kv,
            [
                raw_stride,
                l.side_stride[kind],
                0,
                kv_channels,
                dynamic_stride,
            ],
            (self.math.qk_l2_clip_threshold, eps),
            dynamic,
        )?;
        block.report.launches.set(block.report.launches.get() + 3);
        #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
        block.dump_planes(index)?;

        // Attention in tiles of `QUERY_TILE_LIMIT` rows, each committing its ring rows.
        let side_stride = l.side_stride[kind];
        let mut tiles = 0;
        for first in (0..rows).step_by(QUERY_TILE_LIMIT) {
            let count = QUERY_TILE_LIMIT.min(rows - first);
            let tile = RelativeAttentionTile {
                first: position + first,
                count,
                input_stride: side_stride * WORD,
                query_stride: sidecar_stride * WORD,
                side_stride: sidecar_stride * WORD,
            };
            let at = |base: usize, stride: usize, width: usize| {
                block.at(base + first * stride, (count - 1) * stride + width)
            };
            let call = RelativeAttentionCall {
                shape: parts.shape,
                q: at(l.sidecar[kind], sidecar_stride, channels)?,
                k: at(l.side[kind], side_stride, kv_channels)?,
                v: at(l.side[kind] + kv_channels, side_stride, kv_channels)?,
                relative: at(l.sidecar[kind] + channels, sidecar_stride, relative)?,
                table: parts.table,
                sinks: at(l.sidecar[kind] + channels + relative, sidecar_stride, query)?,
                gates: at(
                    l.sidecar[kind] + channels + relative + query,
                    sidecar_stride,
                    channels,
                )?,
                k_cache: parts.k_cache,
                v_cache: parts.v_cache,
                out: block.at(l.out[kind] + first * channels, count * channels)?,
                pos: at(l.side[kind] + 2 * kv_channels, side_stride, 1)?,
                status: block.at(l.status + index * rows + first, count)?,
            };
            block
                .context
                .relative_attention_tile(block.stream, call, tile)?;
            tiles += 1;
        }
        block
            .report
            .launches
            .set(block.report.launches.get() + tiles);

        // Output projection and its short convolution.
        let output_name = format!("{layer}.attn.wo_ud.weight");
        let projection = cuda
            .relative_attention_projection(&output_name, [h, channels])?
            .ok_or_else(|| bad("the resident pass needs the attention output projection"))?;
        q8_row_reuse::project_output(
            block.context,
            block.stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(rows, projection.rows, channels),
                dtype: projection.dtype,
                w: projection.weight,
                x: block.at(l.out[kind], rows * channels)?,
                y: block.at(l.attn_raw, rows * h)?,
            },
        )?;
        block.context.inkling_output_conv(
            block.stream,
            [
                block.at(l.attn_raw, rows * h)?,
                block.tail(0, index, 2)?,
                block.constant(&consts.attn_sconv)?,
                block.at(l.attn_conv, rows * h)?,
                status,
            ],
            OutputConvShape {
                rows,
                width: h,
                history: past,
            },
        )?;
        block.context.inkling_pass_tails(
            block.stream,
            [
                block.at(l.attn_raw, rows * h)?,
                block.tail(0, index, 2)?,
                block.tail(1, index, 2)?,
            ],
            [block.committed, h, h, past],
        )?;
        block.report.launches.set(block.report.launches.get() + 3);
        Ok(())
    }

    /// The controller of one role: `down` of `[norm(in) | norm(msg)]`, then `gain` and `forget`
    /// of the reduced row; one launch group.
    fn pass_controller(
        &self,
        block: &Block<'_>,
        cuda: &GgufCuda,
        names: [&str; 3],
        gate: DeviceRange<'_>,
        rank: usize,
    ) -> Result<(), EngError> {
        let h = block.h;
        let rows = block.rows;
        // The `down` matrix maps one gate row `[norm(in) | norm(msg)]` of `2 * h` words.
        let weights = cuda
            .controller_weights(names, 2 * h, [rank, h])
            .ok_or_else(|| bad("the resident pass needs grouped controller weights"))??;
        let gain = block.at(block.l.gain, rows * h)?;
        let forget = block.at(block.l.forget, rows * h)?;
        let reduced = block.at(block.l.reduced, rows * rank)?;
        let outputs = [reduced, gain, forget];
        let mut calls = Vec::with_capacity(3);
        for (i, (weight, dtype)) in weights.iter().enumerate() {
            let (cols, ld_x) = if i == 0 { (2 * h, 2 * h) } else { (rank, rank) };
            let shape = DenseGemvShape {
                m: rows,
                rows: weight.shape[0],
                cols,
                ld_x,
                ld_y: weight.shape[0],
            };
            let need = dense_gemv_bytes(shape, *dtype)?;
            calls.push(DenseGemvCall {
                shape,
                dtype: *dtype,
                w: weight.codes.range(0, need.w)?,
                x: if i == 0 { gate } else { reduced },
                y: outputs[i],
            });
        }
        block.context.dense_gemv_group(block.stream, &calls)?;
        block
            .report
            .launches
            .set(block.report.launches.get() + calls.len());
        Ok(())
    }

    /// Test aid (`FIFI_INKLING_PASS_ROUTES_IN=<flag-off log>`): the expert routes an earlier flag-off
    /// run printed (`FIFI_INKLING_DEBUG_PREFILL_ROUTES=1`) take the place of this run's own
    /// selection, for visit `index` of the block over the block's rows of the prompt. The flag-off
    /// path prints its pieces one after another and, inside a piece, the layer's visits in order:
    /// the sets that open a piece tile the prompt, so the set is found from the block's first row.
    /// Two runs whose router logits differ by float noise disagree on near-tie rows only;
    /// replaying the routes removes that disagreement from the comparison.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    fn replayed_routes(
        &self,
        logits: &[f32],
        bias: &[f32],
        block: &Block<'_>,
        index: usize,
        layer: usize,
    ) -> Result<Option<Vec<MoeRoute>>, EngError> {
        // layer -> that layer's printed sets, in printed order
        type PrintedRoutes = Option<std::collections::HashMap<usize, Vec<Vec<Vec<usize>>>>>;
        static ROUTES: std::sync::OnceLock<PrintedRoutes> = std::sync::OnceLock::new();
        let (rows, offset) = (block.rows, block.offset);
        let of_layer = |i: usize| self.visits[i].weight_layer == layer;
        // The layer's visits in a piece, and this one's position among them.
        let per_piece = (0..self.visits.len()).filter(|&i| of_layer(i)).count();
        let ordinal = (0..index).filter(|&i| of_layer(i)).count();
        let routes = ROUTES.get_or_init(|| {
            let text =
                std::fs::read_to_string(std::env::var("FIFI_INKLING_PASS_ROUTES_IN").ok()?).ok()?;
            let mut out = std::collections::HashMap::<usize, Vec<Vec<Vec<usize>>>>::new();
            for line in text
                .lines()
                .filter(|line| line.contains("Inkling grouped route diagnostic "))
            {
                let value: serde_json::Value = line
                    .split("Inkling grouped route diagnostic ")
                    .nth(1)?
                    .trim()
                    .parse()
                    .ok()?;
                out.entry(value["layer"].as_u64()? as usize)
                    .or_default()
                    .push(serde_json::from_value::<Vec<Vec<usize>>>(value["routes"].clone()).ok()?);
            }
            Some(out)
        });
        let Some(sets) = routes.as_ref().and_then(|all| all.get(&layer)) else {
            return Ok(None);
        };
        let mut picked = None;
        let (mut piece, mut start) = (0, 0);
        while let Some(opening) = sets.get(piece * per_piece) {
            if offset < start + opening.len() {
                let Some(set) = sets.get(piece * per_piece + ordinal) else {
                    break;
                };
                if set.len() == opening.len() {
                    picked = set.get(offset - start..offset - start + rows);
                }
                break;
            }
            start += opening.len();
            piece += 1;
        }
        let Some(picked) = picked else {
            return Ok(None);
        };
        let mut replayed = Vec::with_capacity(rows);
        for (row, selected) in logits.chunks_exact(bias.len()).zip(picked) {
            let mut route = self.moe_route_from_logits(layer, row.to_vec(), bias)?;
            route.total = selected
                .iter()
                .map(|&id| route.scores[id])
                .sum::<f32>()
                .max(ROUTE_NORMALIZATION_FLOOR);
            route.selected = selected.clone();
            replayed.push(route);
        }
        Ok(Some(replayed))
    }

    #[allow(clippy::too_many_arguments)]
    fn pass_mlp(
        &self,
        block: &mut Block<'_>,
        readback: &mut PinnedBuf,
        weights: &GgufTextWeights,
        cuda: &GgufCuda,
        layer: &str,
        index: usize,
        position: usize,
        bank: Option<(usize, usize, usize)>,
    ) -> Result<(), EngError> {
        let l = block.l;
        let rows = block.rows;
        let h = block.h;
        let consts = &block.visits[index].consts;
        let scratch_words = routed_prefill_scratch_bytes(rows)
            .ok_or_else(|| bad("rows"))?
            .div_ceil(WORD);
        let eps = self.math.rms_norm_eps;
        let rank = self.spec.core.gate_rank as usize;
        let input = block.at(l.mlp_input, rows * h)?;
        let norm = block.constant(&consts.mlp_norm)?;
        let health = block.at(l.health, rows)?;
        if let Some((bank_buffer, bank_blocks, bank_first)) = bank {
            let message = block.at(l.messages + ROLES * rows * h, rows * h)?;
            block.mix(
                bank_buffer,
                bank_blocks,
                bank_first,
                &consts.control[ROLES],
                h,
                message,
            )?;
            let gate = block.at(l.gate, rows * 2 * h)?;
            block.context.inkling_pass_gate(
                block.stream,
                [input, message, gate, health],
                [rows, h, 2 * h],
                MIX_EPS,
            )?;
            let names = control_names(layer, Role::Mlp);
            self.pass_controller(
                block,
                cuda,
                names.each_ref().map(String::as_str),
                gate,
                rank,
            )?;
            let gain = block.at(l.gain, rows * h)?;
            let forget = block.at(l.forget, rows * h)?;
            block.context.inkling_pass_apply(
                block.stream,
                [input, message, gain, forget, norm, input, health],
                [rows, h],
                eps,
                true,
            )?;
            block.report.launches.set(block.report.launches.get() + 2);
        } else {
            block.context.inkling_pass_apply(
                block.stream,
                [input, input, input, input, norm, input, health],
                [rows, h],
                eps,
                false,
            )?;
            block.report.launches.set(block.report.launches.get() + 1);
        }

        // Router on the device: logits, then sigmoid top-k with the host's normalization.
        let experts = self.spec.n_routed_experts as usize;
        let top_k = self.spec.num_experts_per_tok as usize;
        let prefix = format!("{layer}.mlp");
        let (router, _, [router_rows, router_cols]) = cuda
            .router_weight(&format!("{prefix}.gate.weight"))
            .filter(|(_, dtype, _)| *dtype == DenseDtype::F32)
            .ok_or_else(|| bad("the resident pass needs an F32 router"))?;
        debug_assert_eq!((router_rows, router_cols), (experts, h));
        let shape = RouterShape {
            m: rows,
            hidden: h,
            n_expert: experts,
            top_k,
        };
        let logits = block.at(l.logits, rows * experts)?;
        let ids = block.at(l.ids, rows * top_k)?;
        let route_weights = block.at(l.route_weights, rows * top_k)?;
        let routed_status = block.at(l.flags + l.visits + index, 1)?;
        block.context.router_logits_rows(
            block.stream,
            RouterLogitsCall {
                shape,
                x: input,
                w: router.codes.range(0, router.codes.bytes())?,
                logits,
            },
        )?;
        block.context.router_sigmoid_select_rows(
            block.stream,
            RouterSigmoidCall {
                shape,
                bias: Some(RouterSigmoidBias::HostRounded {
                    values: block.constant(&consts.router_bias)?,
                    floor: ROUTE_NORMALIZATION_FLOOR,
                    scale: self.math.route_scale,
                }),
                logits,
                ids,
                weights: route_weights,
                status: routed_status,
            },
        )?;

        // The routed experts: the GEMM lane's device-routed stage when the layer has its
        // resident route, else its host-planned stage through this lane's launch-only adapter.
        let expert_out = block.at(l.expert_out, rows * h)?;
        let caps = [self.math.mlp_gate_softcap, self.math.mlp_up_softcap];
        let weight_layer = self.visits[index].weight_layer;
        let routed = weights.prefill_device_routed_experts(
            weight_layer as u32,
            block.stream,
            DeviceRoutedPrefill {
                rows,
                x: input,
                ids,
                weights: route_weights,
                out: expert_out,
                scratch: block.at(l.routed_scratch, scratch_words)?,
                status: routed_status,
            },
            caps,
        )?;
        if routed.is_none() {
            // The host adapter: the logits come back and the host selects as the flag-off
            // path does, then the host-planned window is enqueued launch-only.
            let logits_bytes = rows * experts * WORD;
            if readback.len() < logits_bytes {
                return Err(bad(
                    "the resident pass readback plane is too small for the router",
                ));
            }
            block
                .context
                .copy_to_pinned(block.stream, logits, readback, 0)?;
            readback.synchronize(block.stream)?;
            let mut logits = vec![0f32; rows * experts];
            readback.read_f32(0, &mut logits)?;
            block.report.readbacks.set(block.report.readbacks.get() + 2);
            let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            let routes = match self.replayed_routes(&logits, &bias, block, index, weight_layer)? {
                Some(routes) => routes,
                None => self.route_prefill_logits(weight_layer, &logits, &bias, rows)?,
            };
            #[cfg(not(all(test, feature = "native-cuda", target_os = "linux")))]
            let routes = self.route_prefill_logits(weight_layer, &logits, &bias, rows)?;
            let names = [
                format!("{prefix}.experts.w13_weight"),
                format!("{prefix}.experts.w2_weight"),
            ];
            let ids: Vec<&[usize]> = routes
                .iter()
                .map(|route| route.selected.as_slice())
                .collect();
            let batch = ExpertPrefillRows::Resident {
                ids: &ids,
                hidden: h,
                activations: input,
            };
            weights
                .prefill_device_experts_unfenced(
                    names.each_ref().map(String::as_str),
                    weight_layer as u32,
                    (batch, Some(expert_out)),
                    caps,
                    |row, id| routes[row].weight(id, self.math.route_scale),
                    (block.stream, block.at(l.flags + index, 1)?),
                )?
                .ok_or_else(|| bad("the resident pass expert window was not admitted"))?;
            block.report.readbacks.set(block.report.readbacks.get() + 1);
        }
        block.report.launches.set(block.report.launches.get() + 3);

        let past = PAST.min(position);
        let status = block.at(l.status + index * rows, rows)?;
        block.context.inkling_output_conv(
            block.stream,
            [
                block.at(l.expert_out, rows * h)?,
                block.tail(0, index, 3)?,
                block.constant(&consts.mlp_sconv)?,
                block.at(l.mlp_conv, rows * h)?,
                status,
            ],
            OutputConvShape {
                rows,
                width: h,
                history: past,
            },
        )?;
        block.context.inkling_pass_tails(
            block.stream,
            [
                block.at(l.expert_out, rows * h)?,
                block.tail(0, index, 3)?,
                block.tail(1, index, 3)?,
            ],
            [block.committed, h, h, past],
        )?;
        block.pass_add_to_current(block.at(l.mlp_conv, rows * h)?)?;
        block.report.launches.set(block.report.launches.get() + 3);
        Ok(())
    }
}

/// Everything one block reports.
#[derive(Default)]
struct Report {
    launches: std::cell::Cell<usize>,
    readbacks: std::cell::Cell<usize>,
    /// Wall-clock split of one block, microseconds: the plan (first block only), the host side of
    /// the launches (the stream keeps running), the fence wait, and the publish.
    plan_us: std::cell::Cell<u64>,
    launch_us: std::cell::Cell<u64>,
    fence_us: std::cell::Cell<u64>,
    publish_us: std::cell::Cell<u64>,
    /// The stream time of the sweep (first launch to the last kernel's end), microseconds.
    block_us: std::cell::Cell<u64>,
}

/// The device planes of one block, borrowed from the plan.
struct Block<'a> {
    context: &'a CudaContext,
    stream: &'a DeviceStream,
    arena: &'a DeviceBuffer,
    l: &'a Arena,
    constants: &'a DeviceBuffer,
    current: &'a [DeviceBuffer; 3],
    sealed: &'a [DeviceBuffer; 3],
    /// `[read, write]`: the previous block's tails and the ones this block writes.
    tails: [&'a DeviceBuffer; 2],
    visits: &'a [VisitPlan],
    /// Rows every launch shapes (2 for a one-row block: see [`CpuForward::run_block`]).
    rows: usize,
    /// Rows the block commits (tails, cursors, publication).
    committed: usize,
    /// The block's first row, counted from the prompt's first (test aid: the replayed routes).
    #[cfg_attr(
        not(all(test, feature = "native-cuda", target_os = "linux")),
        allow(dead_code)
    )]
    offset: usize,
    h: usize,
    /// The live depth stream's buffer index.
    live: std::cell::Cell<usize>,
    report: &'a mut Report,
}

impl Block<'_> {
    /// Test aid (`FIFI_INKLING_PASS_DUMP_VISIT=index:path[,index:path]`): after its K/V
    /// preparation, the
    /// visit's live stream, attention input, raw K/V and prepared side rows, as bits, for a host
    /// comparison.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    fn dump_planes(&self, index: usize) -> Result<(), EngError> {
        static DUMP: std::sync::OnceLock<Vec<(usize, std::path::PathBuf)>> =
            std::sync::OnceLock::new();
        let dumps = DUMP.get_or_init(|| {
            std::env::var("FIFI_INKLING_PASS_DUMP_VISIT")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|item| {
                            let (index, path) = item.split_once(':')?;
                            Some((index.parse().ok()?, std::path::PathBuf::from(path)))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        });
        let Some((_, path)) = dumps.iter().find(|(wanted, _)| *wanted == index) else {
            return Ok(());
        };
        let read = |buffer: &crate::kernels::cuda::DeviceBuffer, start: usize, words: usize| {
            let mut bytes = vec![0u8; words * WORD];
            buffer.read_at(self.stream, start * WORD, &mut bytes)?;
            Ok::<_, EngError>(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>(),
            )
        };
        let l = self.l;
        let rows = self.rows;
        let kind = self.visits[index].kind;
        let value = serde_json::json!({
            "current": read(&self.current[self.live.get()], 0, rows * self.h)?,
            "inputs": read(self.arena, l.inputs, 3 * rows * self.h)?,
            "input": read(self.arena, l.inputs, rows * self.h)?,
            "raw": read(self.arena, l.raw[kind], rows * l.raw_stride[kind])?,
            "side": read(self.arena, l.side[kind], rows * l.side_stride[kind])?,
            "raw_stride": l.raw_stride[kind],
            "side_stride": l.side_stride[kind],
        });
        std::fs::write(
            path,
            serde_json::to_vec(&value).map_err(|e| bad(format!("resident pass dump: {e}")))?,
        )
        .map_err(|e| bad(format!("resident pass dump: {e}")))?;
        eprintln!("Inkling resident pass dumped visit {index} after its K/V preparation");
        Ok(())
    }

    /// Test aid (`FIFI_INKLING_PASS_DUMP_CONTROL=index:path[,index:path]`): after the attention
    /// inputs, the three role inputs, the controller messages, the gate and the controller rows
    /// of the last role, as bits.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    fn dump_control(
        &self,
        index: usize,
        bank: Option<(usize, usize, usize)>,
    ) -> Result<(), EngError> {
        static DUMP: std::sync::OnceLock<Vec<(usize, std::path::PathBuf)>> =
            std::sync::OnceLock::new();
        let dumps = DUMP.get_or_init(|| {
            std::env::var("FIFI_INKLING_PASS_DUMP_CONTROL")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|item| {
                            let (index, path) = item.split_once(':')?;
                            Some((index.parse().ok()?, std::path::PathBuf::from(path)))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        });
        let Some((_, path)) = dumps.iter().find(|(wanted, _)| *wanted == index) else {
            return Ok(());
        };
        let read = |buffer: &crate::kernels::cuda::DeviceBuffer, start: usize, words: usize| {
            let mut bytes = vec![0u8; words * WORD];
            buffer.read_at(self.stream, start * WORD, &mut bytes)?;
            Ok::<_, EngError>(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>(),
            )
        };
        let l = self.l;
        let rows = self.rows;
        let h = self.h;
        let mut bank_rows = serde_json::Map::new();
        if let Some((buffer, blocks, _)) = bank {
            let sealed = &self
                .sealed
                .get(buffer)
                .ok_or_else(|| bad("resident pass dump: no bank buffer"))?;
            let current = &self
                .current
                .get(buffer)
                .ok_or_else(|| bad("resident pass dump: no bank buffer"))?;
            bank_rows.insert("sealed".into(), read(sealed, 0, blocks * rows * h)?.into());
            bank_rows.insert("current".into(), read(current, 0, rows * h)?.into());
        }
        let value = serde_json::json!({
            "inputs": read(self.arena, l.inputs, 3 * rows * h)?,
            "messages": read(self.arena, l.messages, 3 * rows * h)?,
            "gate": read(self.arena, l.gate, rows * 2 * h)?,
            "gain": read(self.arena, l.gain, rows * h)?,
            "forget": read(self.arena, l.forget, rows * h)?,
            "bank": bank,
            "bank_rows": bank_rows,
        });
        std::fs::write(
            path,
            serde_json::to_vec(&value).map_err(|e| bad(format!("resident pass dump: {e}")))?,
        )
        .map_err(|e| bad(format!("resident pass dump: {e}")))?;
        eprintln!("Inkling resident pass dumped the control of visit {index}");
        Ok(())
    }

    /// `Stream::add_current(row)` of the live stream.
    fn pass_add_to_current(&self, row: DeviceRange<'_>) -> Result<(), EngError> {
        let health = self.at(self.l.health, self.rows)?;
        self.context.inkling_pass_add(
            self.stream,
            PassAddCall {
                current: self.current_range(self.live.get())?,
                row,
                health,
            },
            [self.rows, self.h],
        )?;
        self.report.launches.set(self.report.launches.get() + 1);
        Ok(())
    }

    fn at(&self, start: usize, words: usize) -> Result<DeviceRange<'_>, EngError> {
        self.arena.range(start * WORD, words * WORD)
    }

    fn constant(&self, range: &Range<usize>) -> Result<DeviceRange<'_>, EngError> {
        self.constants.range(range.start, range.len())
    }

    fn current_range(&self, buffer: usize) -> Result<DeviceRange<'_>, EngError> {
        self.current[buffer].range(0, self.rows * self.h * WORD)
    }

    /// The first `blocks` sealed blocks; a stream without blocks shows one spare block.
    fn sealed_range(&self, buffer: usize, blocks: usize) -> Result<DeviceRange<'_>, EngError> {
        self.sealed[buffer].range(0, blocks.max(1) * self.rows * self.h * WORD)
    }

    /// The previous block's tail plane `which` of visit `index`, exactly `PAST * width` words
    /// (`tails` ranges count words; the buffers take bytes).
    fn tail(&self, from: usize, index: usize, which: usize) -> Result<DeviceRange<'_>, EngError> {
        let consts = &self.visits[index];
        let widths = self.tail_widths(index);
        let at = consts.tails.start * WORD + widths[..which].iter().sum::<usize>() * PAST * WORD;
        self.tails[from].range(at, PAST * widths[which] * WORD)
    }

    fn tail_widths(&self, index: usize) -> [usize; 4] {
        let kv = self.visits[index].heads[1] * HEAD_DIM;
        [kv, kv, self.h, self.h]
    }

    fn restore_stream(&self, live: usize, prelude: usize, blocks: usize) -> Result<(), EngError> {
        let row = self.rows * self.h * WORD;
        self.context.copy_device(
            self.stream,
            self.current[prelude].range(0, row)?,
            self.current[live].range(0, row)?,
        )?;
        if blocks > 0 {
            self.context.copy_device(
                self.stream,
                self.sealed[prelude].range(0, blocks * row)?,
                self.sealed[live].range(0, blocks * row)?,
            )?;
        }
        Ok(())
    }

    /// `Stream::mix(first, query)` of the depth stream `buffer` into `out` (`width` a row).
    fn mix(
        &self,
        buffer: usize,
        blocks: usize,
        first: usize,
        query: &Range<usize>,
        width: usize,
        out: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let first = first.min(blocks);
        self.context.inkling_pass_mix(
            self.stream,
            PassMixCall {
                sealed: self.sealed_range(buffer, blocks)?,
                current: self.current_range(buffer)?,
                query: self.constant(query)?,
                out,
                health: self.at(self.l.health, self.rows)?,
            },
            [self.rows, blocks, first, width, width],
            MIX_EPS,
        )?;
        self.report.launches.set(self.report.launches.get() + 1);
        Ok(())
    }
}

/// The per-visit constants buffer under construction: f32 rows appended in order.
#[derive(Default)]
struct Consts {
    bytes: Vec<u8>,
}

impl Consts {
    fn add(&mut self, values: &[f32]) -> Range<usize> {
        let start = self.bytes.len();
        self.bytes
            .extend(values.iter().flat_map(|v| v.to_ne_bytes()));
        start..self.bytes.len()
    }
}

struct OwnedProjection {
    name: String,
    input: usize,
    rows: usize,
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<WORD>()
        .0
        .iter()
        .map(|w| f32::from_ne_bytes(*w))
        .collect()
}

/// The `PAST` rows of a host history (newest first), oldest first, zero padded on the left.
fn tail_bytes(rows: &[Vec<f32>], width: usize) -> Vec<u8> {
    let mut out = vec![0u8; PAST * width * WORD];
    for (lag, row) in rows.iter().take(PAST).enumerate() {
        let at = (PAST - 1 - lag) * width * WORD;
        out[at..at + width * WORD].copy_from_slice(&bytes(&row[..width]));
    }
    out
}

impl CpuForward {
    /// The constants of one visit, looked up on the host by name: the depth-mix queries, the
    /// controller queries of a banked visit, the norms, clip scales, convolution taps and the router
    /// bias. Also used by the host-only test that checks every name against the checkpoint.
    fn visit_consts(
        host: &CpuForward,
        visit: LayerVisit,
        geometry: Geometry,
        consts: &mut Consts,
    ) -> Result<VisitConsts, EngError> {
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let prefix = format!("{layer}.attn");
        let kind = usize::from(visit.attention == Attention::Global);
        let dynamic = kind == 1;
        let h = host.hidden();
        let experts = host.spec.n_routed_experts as usize;
        let row = |name: &str, at: usize, len: usize| -> Result<Vec<f32>, EngError> {
            host.weights.row(name, at, len).map(|cow| cow.into_owned())
        };
        let vec = |name: &str, len: usize| -> Result<Vec<f32>, EngError> {
            host.weights.vec(name, len).map(|cow| cow.into_owned())
        };
        let mut mix: [Range<usize>; ROLES] = std::array::from_fn(|_| 0..0);
        if let Some(q) = physical.checked_sub(1) {
            for (role, range) in mix.iter_mut().enumerate() {
                *range = consts.add(&row(
                    &format!("model.llm.attn_res.attn_queries.{q}"),
                    if kind == 0 { 0 } else { role },
                    h,
                )?);
            }
        }
        let mut control: [Range<usize>; 4] = std::array::from_fn(|_| 0..0);
        if host.pass_visit_banked(visit.state_slot) {
            for (range, role) in
                control
                    .iter_mut()
                    .zip([Role::Query, Role::Key, Role::Value, Role::Mlp])
            {
                *range = consts.add(&host.control_query(physical, role)?);
            }
        }
        let mut norms: [Range<usize>; ROLES] = std::array::from_fn(|_| 0..0);
        for (i, range) in norms.iter_mut().enumerate() {
            *range = consts.add(&vec(&attention_norm_name(&layer, kind == 0, i), h)?);
        }
        let mut clip = vec(&format!("{prefix}.q_l2_clip.scale"), geometry.channels)?;
        clip.extend_from_slice(&vec(
            &format!("{prefix}.rel_l2_clip.scale"),
            geometry.relative,
        )?);
        let conv = |plane: &str| -> Result<Vec<f32>, EngError> {
            let name = if dynamic { "static_w" } else { "weight" };
            vec(
                &format!("{prefix}.{plane}_sconv.{name}"),
                TAPS * geometry.kv_channels,
            )
        };
        Ok(VisitConsts {
            mix,
            control,
            norms,
            clip: consts.add(&clip),
            conv_k: consts.add(&conv("k")?),
            conv_v: consts.add(&conv("v")?),
            attn_sconv: consts.add(&vec(&format!("{layer}.attn_sconv.weight"), TAPS * h)?),
            mlp_mix: consts.add(&vec(
                &format!("model.llm.attn_res.mlp_queries.{physical}"),
                h,
            )?),
            mlp_norm: consts.add(&vec(&format!("{layer}.mlp_norm.weight"), h)?),
            mlp_sconv: consts.add(&vec(&format!("{layer}.mlp_sconv.weight"), TAPS * h)?),
            router_bias: consts.add(&vec(&format!("{layer}.mlp.gate.bias"), experts)?),
        })
    }
}

impl PassWorkspace {
    fn plan(host: &CpuForward, cuda: &GgufCuda, rows: usize) -> Result<Self, EngError> {
        let h = host.hidden();
        let visits_n = host.visits.len();
        let experts = host.spec.n_routed_experts as usize;
        let top_k = host.spec.num_experts_per_tok as usize;
        let rank = host.spec.core.gate_rank as usize;
        let (_, attention) = cuda.resident_attention().ok_or_else(|| bad("admitted"))?;
        let mut consts = Consts::default();
        let mut geometries: [Option<Geometry>; 2] = [None, None];
        let mut tails_words = 0;
        let mut visits = Vec::with_capacity(visits_n);
        for visit in &host.visits {
            let heads = host.pass_heads(visit.attention);
            let kind = usize::from(visit.attention == Attention::Global);
            let parts = attention
                .tile_parts(visit.state_slot)
                .ok_or_else(|| bad("admitted"))?;
            let geometry = *geometries[kind].get_or_insert(Geometry {
                query: heads[0],
                kv: heads[1],
                channels: heads[0] * HEAD_DIM,
                kv_channels: heads[1] * HEAD_DIM,
                relative: heads[0] * host.math.d_rel,
            });
            let _ = parts;
            if geometry.query != heads[0] || geometry.kv != heads[1] {
                return Err(bad("two visits of one attention kind differ in head count"));
            }
            let consts = CpuForward::visit_consts(host, *visit, geometry, &mut consts)?;
            let tails = tails_words..tails_words + PAST * (2 * geometry.kv_channels + 2 * h);
            tails_words = tails.end;
            visits.push(VisitPlan {
                kind,
                tails,
                consts,
                heads,
            });
        }
        let geometry = |kind: usize| geometries[kind].ok_or_else(|| bad("admitted"));
        let constants = consts.bytes;
        let sidecar_stride = [
            geometry(0)?.channels
                + geometry(0)?.relative
                + geometry(0)?.query
                + geometry(0)?.channels,
            geometry(1)?.channels
                + geometry(1)?.relative
                + geometry(1)?.query
                + geometry(1)?.channels,
        ];
        let raw_stride = [2 * geometry(0)?.kv_channels, 2 * geometry(1)?.kv_channels];
        let dynamic_stride = [2 * geometry(0)?.kv * TAPS, 2 * geometry(1)?.kv * TAPS];
        let side_stride = [raw_stride[0] + 1, raw_stride[1] + 1];
        let mut at = 0usize;
        // Every plane starts on a 64-byte boundary: the projections take 16-byte aligned rows,
        // and the routed scratch is sized in bytes that need not be a multiple of four.
        let mut alloc = |words: usize| {
            let start = at;
            at += words.next_multiple_of(16);
            start
        };
        let layout = Arena {
            visits: visits_n,
            health: alloc(rows),
            status: alloc(visits_n * rows),
            flags: alloc(2 * visits_n),
            logits: alloc(rows * experts),
            ids: alloc(rows * top_k),
            route_weights: alloc(rows * top_k),
            routed_scratch: alloc(
                routed_prefill_scratch_bytes(rows)
                    .ok_or_else(|| bad("rows"))?
                    .div_ceil(WORD),
            ),
            inputs: alloc(ROLES * rows * h),
            messages: alloc(4 * rows * h),
            gate: alloc(rows * 2 * h),
            reduced: alloc(rows * rank),
            gain: alloc(rows * h),
            forget: alloc(rows * h),
            sidecar: [
                alloc(rows * sidecar_stride[0]),
                alloc(rows * sidecar_stride[1]),
            ],
            sidecar_stride,
            raw: [alloc(rows * raw_stride[0]), alloc(rows * raw_stride[1])],
            raw_stride,
            dynamic: [
                alloc(rows * dynamic_stride[0]),
                alloc(rows * dynamic_stride[1]),
            ],
            dynamic_stride,
            side: [alloc(rows * side_stride[0]), alloc(rows * side_stride[1])],
            side_stride,
            out: [
                alloc(rows * geometry(0)?.channels),
                alloc(rows * geometry(1)?.channels),
            ],
            attn_raw: alloc(rows * h),
            attn_conv: alloc(rows * h),
            mlp_input: alloc(rows * h),
            expert_out: alloc(rows * h),
            mlp_conv: alloc(rows * h),
            words: at,
        };
        let context = cuda.context();
        let stream = context.stream()?;
        let row = rows * h * WORD;
        Ok(Self {
            visits,
            rows,
            arena: context.allocate(layout.words * WORD)?,
            layout,
            constants: context.upload(&constants)?,
            current: [
                context.allocate(row)?,
                context.allocate(row)?,
                context.allocate(row)?,
            ],
            sealed: [
                context.allocate(STREAM_BLOCKS * row)?,
                context.allocate(STREAM_BLOCKS * row)?,
                context.allocate(STREAM_BLOCKS * row)?,
            ],
            tails: [
                context.allocate(tails_words * WORD)?,
                context.allocate(tails_words * WORD)?,
            ],
            tails_from: 0,
            tails_valid: false,
            readback: context.pinned_buf(
                (rows + visits_n * rows + 2 * visits_n + 1).max(rows * experts) * WORD,
            )?,
            stream,
        })
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod tests;
