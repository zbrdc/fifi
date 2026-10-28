//! One T-row Flint forward, shared by prefill chunks and one-row decode.

use super::{
    factory::context_limit,
    mlx_model::{Ffn, Layer, Model},
    mlx_route,
    mlx_state::{ChunkCachePlan, Plane, State, WINDOW},
    precision_native,
};
use crate::{
    EngError,
    kernels::{
        BinaryOp, BoundedIndices, MlxKernels,
        mlx::{KvBlock, SharedAttention, native::compile::FlintTailArgs},
    },
    tensor::ExecutionTensor,
};

/// Graph-cache slice 1 routing: the compiled tail serves only one-row chunks (decode steps
/// and single-row prefill pieces) with the flag armed. Multi-row prefill chunks keep the
/// eager tail unconditionally (their row counts would multiply the compile cache keys).
pub(in crate::family::flint) fn compiled_tail_routes(rows: u32) -> bool {
    crate::kernels::mlx::compile_layer_flagged() && rows == 1
}

#[cfg(test)]
use super::{
    mlx_math,
    mlx_qualification::{Span, capture, injected_input, wanted},
};
#[cfg(test)]
use crate::tensor::{Dtype, Tensor};

type Result<T> = std::result::Result<T, EngError>;

/// pp attribution probe (test builds): under FLINT_QUAL_STAGE_SYNC, fence
/// `tensor` (evaluate + synchronize) so the enclosing span's elapsed time
/// includes the GPU work instead of only host submission. Perturbs the
/// async chain on purpose — the two trace modes bracket dispatch vs GPU
/// per stage. No-op in every other test run and absent from production.
#[cfg(test)]
fn stage_fence(k: &mut MlxKernels, label: &str, tensor: &ExecutionTensor) -> Result<()> {
    if std::env::var_os("FLINT_QUAL_STAGE_SYNC").is_some() {
        use std::sync::Mutex;
        use std::time::Instant;
        static LAST: Mutex<Option<Instant>> = Mutex::new(None);
        let before = Instant::now();
        k.evaluate(&[tensor])?;
        let done = Instant::now();
        // Fence-to-fence delta: GPU wall for the sub-stage this fence closes.
        let mut last = LAST.lock().expect("fence lock");
        let delta = last.map_or(0, |l| done.duration_since(l).as_micros() as u64);
        *last = Some(done);
        drop(last);
        eprintln!(
            "flint_fence {} {} {}",
            label,
            done.duration_since(before).as_micros(),
            delta
        );
    }
    Ok(())
}

/// D2 async chain (design scrch/7c82644c/flint-async-chain/DESIGN.md): decode
/// layers per evaluate group when the chain is on. 49 layers -> 13 roots.
const GROUP_LAYERS: usize = 4;

/// Async-chain switch (D2, design scrch/7c82644c/flint-async-chain/DESIGN.md):
/// one-row chunks (decode steps and single-row prefill pieces) are grouped
/// into `GROUP_LAYERS`-layer evaluate chains. Default ON since the
/// 2026-10-25 A/B window (step 0.1038 -> 0.0774 s, bitwise-identical
/// logits); the fail-safe rollback is `FIFI_FLINT_MLX_ASYNC_CHAIN=0` (or
/// any non-enable value — see `tokenizer::env_allows_default_on`). Read
/// once per process, and the effective state is logged once at that read.
pub(in crate::family::flint) fn async_chain_flagged() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        let on = crate::tokenizer::env_allows_default_on("FIFI_FLINT_MLX_ASYNC_CHAIN");
        // The one load line an operator needs to confirm a rollback.
        eprintln!("flint: async chain {}", if on { "on" } else { "off" });
        on
    })
}

/// D1 residual-coalesce switch: same fail-safe default-on polarity and
/// once-per-process read as the async chain. Default on since the
/// 2026-10-24 A/B window (83.0 -> 72.1 ms/forward, bitwise-identical
/// trajectories); the rollback is any non-enable value, `=0` documented.
pub(in crate::family::flint) fn coalesce_residual_flagged() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        let on = crate::tokenizer::env_allows_default_on("FIFI_FLINT_MLX_COALESCE_RESIDUAL");
        eprintln!("flint: residual coalesce {}", if on { "on" } else { "off" });
        on
    })
}

/// FIFI_FLINT_MLX_RING_VIEW routes full-attention KV planes onto the Qwen
/// Q36-KV-INTEGRATE shape: the plane is appended in place and attention
/// consumes the whole plane (the kernel receives len = p0+1 separately from
/// the plane's row count), instead of materializing the chronological union
/// per step (~227k KV rows/step at a 4k prefix, window-12 measurement; the
/// history side of that union was already a view). SWA planes keep their
/// bounded two-slice union; safety rests on MLX donating in place only when
/// the handle is unique, never on any capacity change. Default ON since
/// the 2026-10-28 flip; the fail-safe rollback (`=0`, any non-enable
/// value, non-UTF-8) stays off so a mis-set rollback cannot fail
/// silently.
/// Pure mapping from the environment value to the flag, testable without
/// poisoning the process-wide OnceLock: absent keeps the default ON; once
/// set, only the explicit enable words stay on; non-UTF-8 and every other
/// value roll back OFF (the `env_allows_default_on` polarity, so a mis-set
/// rollback cannot silently fail).
pub(in crate::family::flint) fn ring_view_env_flag(
    value: std::result::Result<String, std::env::VarError>,
) -> bool {
    crate::tokenizer::env_result_allows_default_on(value)
}

/// Reads FIFI_FLINT_MLX_RING_VIEW once per process (fail-safe: absent is
/// the default ON; a set value survives only as the explicit enable words)
/// and logs the one load line an operator needs to confirm a rollback.
pub(in crate::family::flint) fn ring_view_flagged() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        let on = ring_view_env_flag(std::env::var("FIFI_FLINT_MLX_RING_VIEW"));
        // The one load line an operator needs to confirm a rollback. The
        // route additionally requires the chained commit, so report the
        // EFFECTIVE state: with ASYNC_CHAIN=0 the flag is armed but inert.
        // Reading the chain env directly (not via its OnceLock) keeps this
        // line side-effect-free and order-independent.
        let chain = crate::tokenizer::env_allows_default_on("FIFI_FLINT_MLX_ASYNC_CHAIN");
        let state = if !on {
            "off"
        } else if chain {
            "on"
        } else {
            "on (inert: async chain off)"
        };
        eprintln!("flint: ring view {state}");
        on
    })
}

/// The geometry a chunk must have before the ring-view route may arm.
/// Bundled (not five positional bools) because every field is part of one
/// safety argument: `chained` comes from the CALLER's commit choice, never
/// from the global chain flag, so a fenced caller cannot meet the route by
/// accident and then hit the guarded-commit rejection.
pub(in crate::family::flint) struct RingViewGeometry {
    pub(in crate::family::flint) full: bool,
    pub(in crate::family::flint) wrapped: bool,
    pub(in crate::family::flint) one_row: bool,
    pub(in crate::family::flint) chained: bool,
}

/// The pure routing predicate: arm only for a full-attention plane whose
/// one-row chunk does not wrap the ring, committed by the chained commit,
/// with the flag set. Pure in `flagged` so both polarities are unit-tested
/// without poisoning the process-wide OnceLock.
pub(in crate::family::flint) fn ring_view_routes_with(
    geo: RingViewGeometry,
    flagged: bool,
) -> bool {
    geo.chained && geo.one_row && geo.full && !geo.wrapped && flagged
}

/// Process-flag wrapper around [`ring_view_routes_with`].
pub(in crate::family::flint) fn ring_view_routes(geo: RingViewGeometry) -> bool {
    ring_view_routes_with(geo, ring_view_flagged())
}

/// Prefill-group chaining switch (window-34/35 attribution: the fenced
/// `Plane::commit_chunk` costs 8.4 ms x 392 on pp4096 while the chained
/// commit is graph-ordered and fence-free). Default OFF: multi-row chunks
/// keep the reviewed per-layer-fence path until the A/B proves >=+10%
/// pp4096 tok/s. Rollback is `FIFI_FLINT_MLX_PREFILL_CHAIN=0` (or any
/// non-enable value). Memory note: a chained multi-row group holds up to
/// GROUP_LAYERS layers of un-rooted graph (chunk-sized activations) before
/// the boundary evaluate retires them — the retire cap still bounds it.
pub(in crate::family::flint) fn prefill_chain_flagged() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        let on = match std::env::var("FIFI_FLINT_MLX_PREFILL_CHAIN") {
            Err(_) => false,
            Ok(value) => crate::tokenizer::env_allows_value(Some(&value)),
        };
        // The one load line an operator needs to confirm a rollback.
        eprintln!("flint: prefill chain {}", if on { "on" } else { "off" });
        on
    })
}

/// The chain routes one-row chunks (decode steps and single-row prefill
/// pieces) on the async-chain flag alone, and multi-row prefill chunks
/// only when the prefill-chain switch additionally arms them (default
/// OFF; the fenced per-layer commit stays the reviewed default for
/// prefill).
fn chain_routes_with(rows: u32, prefill_armed: bool) -> bool {
    rows == 1 || (prefill_armed && rows > 1)
}

fn chain_routes(rows: u32) -> bool {
    async_chain_flagged() && chain_routes_with(rows, prefill_chain_flagged())
}

impl Model {
    /// Family-private teacher-forcing seam. No sampling, template, or public debug API.
    pub(super) fn step(
        &mut self,
        state: &mut State,
        token: u32,
        logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        self.prefill_rows(state, &[token], logits)
    }

    pub(super) fn prefill_rows(
        &mut self,
        state: &mut State,
        tokens: &[u32],
        logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        if !state.valid {
            return Err(EngError::BadRequest(
                "Flint state needs reset after a failed forward".into(),
            ));
        }
        if tokens.iter().any(|&token| token >= 150_144) {
            return Err(EngError::BadRequest(
                "Flint token exceeds vocabulary".into(),
            ));
        }
        let rows = u32::try_from(tokens.len())
            .map_err(|_| EngError::BadRequest("Flint chunk row count exceeds u32".into()))?;
        ChunkCachePlan::new(0, state.pos, rows, context_limit())?;
        // Partial layer writes never become a usable state. Reset discards all planes.
        state.valid = false;
        let result = self.forward(state, tokens, logits);
        if result.is_ok() {
            state.valid = true;
        }
        result
    }

    fn forward(
        &mut self,
        state: &mut State,
        tokens: &[u32],
        logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let rows = tokens.len() as u32;
        // D2 chains only one-row chunks (see `chain_routes`):
        // multi-row chunks run the flag-off per-layer-fence path.
        let chain = chain_routes(rows);
        #[cfg(test)]
        let span = Span::begin("embedding_read", state.pos, None, None);
        let packed = self.source.embedding_rows(tokens)?;
        #[cfg(test)]
        span.finish(Some((packed.payload().dtype(), packed.payload().shape())));
        #[cfg(test)]
        let span = Span::begin(
            "embedding_dequantize",
            state.pos,
            None,
            Some((packed.payload().dtype(), packed.payload().shape())),
        );
        let embedding = super::precision::embedding(&packed)?;
        #[cfg(test)]
        span.finish(Some((embedding.dtype(), embedding.shape())));
        #[cfg(test)]
        let span = Span::begin(
            "embedding_import",
            state.pos,
            None,
            Some((embedding.dtype(), embedding.shape())),
        );
        let mut x = self.kernels.import_tensor(&embedding)?;
        #[cfg(test)]
        span.finish(Some((x.dtype(), x.shape())));
        // Expand only this chunk's requested rows, never the complete embedding table.
        drop(embedding);
        drop(packed);
        let mut health = Vec::new();
        // D1 first slice (147->98 evaluates): with the switch on, the per-layer residual root
        // is skipped for every layer that a later existing root completes anyway - the next
        // layer's attention-consumer evaluate in `commit_chunk` pulls this x through the graph,
        // and the head readback roots the last x on the head path. The no-head path keeps the
        // final layer's root: the health export alone may not execute the final FFN.
        // The switch itself is `coalesce_residual_flagged` above.
        let coalesce_residual = coalesce_residual_flagged();
        // D2 async chain (design scrch/7c82644c/flint-async-chain/DESIGN.md):
        // with the switch on, the per-layer commit fences are replaced by
        // graph-ordered appends (see `Plane::commit_chunk_chained`) and one
        // evaluate at each 4-layer group boundary roots the accumulated graph.
        // The group roots also carry the residual x, so the D1 per-layer root
        // above is skipped entirely when chained.
        let group = GROUP_LAYERS;
        for (index, layer) in self.layers.iter().enumerate() {
            #[cfg(test)]
            if rows == 1 {
                // A recorded one-row injection must never collapse a multirow chunk.
                if let Some(input) = injected_input((index, state.pos), &mut self.kernels)? {
                    x = input;
                }
            }
            #[cfg(test)]
            let span = Span::begin(
                "layer_attention",
                state.pos,
                Some(index),
                Some((x.dtype(), x.shape())),
            );
            // CTX32K-GROW: full planes carry the grown capacity, so the
            // plan's ring/origin math matches the planes that exist.
            state.ensure_full_capacity(&mut self.kernels, state.pos + rows)?;
            let plan = ChunkCachePlan::new(index, state.pos, rows, state.full_capacity)?;
            #[cfg(test)]
            capture((index, state.pos), "input", &mut self.kernels, &x)?;
            #[cfg(test)]
            stage_fence(&mut self.kernels, "attn_out", &x)?;
            let normalized = self.kernels.rms_norm(&x, &layer.input_norm, 1e-6)?;
            #[cfg(test)]
            capture(
                (index, state.pos),
                "norm-input",
                &mut self.kernels,
                &normalized,
            )?;
            let attended = attention(
                &mut self.kernels,
                layer,
                &normalized,
                &mut state.planes[index],
                &plan,
            )?;
            #[cfg(test)]
            capture(
                (index, state.pos),
                "attention-projection",
                &mut self.kernels,
                &attended,
            )?;
            #[cfg(test)]
            stage_fence(&mut self.kernels, "attn_commit", &attended)?;
            #[cfg(test)]
            span.finish(Some((attended.dtype(), attended.shape())));
            #[cfg(test)]
            let span = Span::begin(
                "layer_ffn",
                state.pos,
                Some(index),
                Some((x.dtype(), x.shape())),
            );
            #[cfg(test)]
            let layer_evaluate_span: Span;
            // The eager tail's temporaries must drop at iteration end, in the base's drop
            // order (ff first, then normalized — locals drop in reverse declaration order
            // and the tuple keeps that), exactly as in base main: their Resident drops
            // retire native planes, and a retire can synchronize the stream, so dropping
            // them earlier or in the other order would change flag-off synchronization
            // and retirement ordering (the else branch is flag-off; base holds both).
            let eager_tail_keepalive: (Option<ExecutionTensor>, Option<ExecutionTensor>);
            // Graph-cache slice 1: the flagged decode tail replays the eager op sequence
            // (residual add, post-attention RMS, dense SwiGLU, residual add) as one compiled
            // closure. Only the production decode shape routes here: one row, a dense FFN,
            // in the layer's activation dtype (F32 fixtures, BF16 on the served path).
            // MoE layers and every multi-row prefill chunk keep the eager tail.
            let compiled_tail = compiled_tail_routes(rows) && matches!(layer.ffn, Ffn::Dense(_));
            if compiled_tail {
                eager_tail_keepalive = (None, None);
                let Ffn::Dense(dense) = &layer.ffn else {
                    return Err(EngError::BadRequest(
                        "flint compiled tail needs a dense FFN".into(),
                    ));
                };
                let width = match x.shape() {
                    [_, width] => *width,
                    _ => {
                        return Err(EngError::BadRequest(
                            "flint compiled tail input must be two dimensional".into(),
                        ));
                    }
                };
                x = self.kernels.flint_compiled_tail(&FlintTailArgs {
                    x: &x,
                    attended: &attended,
                    gamma: &layer.post_norm,
                    packed: [&dense.gate, &dense.up, &dense.down],
                    hidden: width,
                    eps: 1e-6,
                })?;
                #[cfg(test)]
                span.finish(Some((x.dtype(), x.shape())));
                #[cfg(test)]
                {
                    layer_evaluate_span = Span::begin(
                        "layer_evaluate",
                        state.pos,
                        Some(index),
                        Some((x.dtype(), x.shape())),
                    );
                }
            } else {
                x = self.kernels.binary(BinaryOp::Add, &x, &attended)?;
                #[cfg(test)]
                capture(
                    (index, state.pos),
                    "post-attention-residual",
                    &mut self.kernels,
                    &x,
                )?;
                let normalized = self.kernels.rms_norm(&x, &layer.post_norm, 1e-6)?;
                #[cfg(test)]
                capture(
                    (index, state.pos),
                    "norm-post-attention",
                    &mut self.kernels,
                    &normalized,
                )?;
                let (ff, route_health) = ffn(&mut self.kernels, &layer.ffn, &normalized)?;
                if let Some(row) = route_health {
                    health.push(row);
                }
                #[cfg(test)]
                capture((index, state.pos), "ffn-output", &mut self.kernels, &ff)?;
                #[cfg(test)]
                stage_fence(&mut self.kernels, "ffn_out", &ff)?;
                #[cfg(test)]
                span.finish(Some((ff.dtype(), ff.shape())));
                #[cfg(test)]
                {
                    layer_evaluate_span = Span::begin(
                        "layer_evaluate",
                        state.pos,
                        Some(index),
                        Some((ff.dtype(), ff.shape())),
                    );
                }
                x = self.kernels.binary(BinaryOp::Add, &x, &ff)?;
                // (ff, normalized): tuple fields drop in declaration order, which must
                // match the base's reverse-local-order drop (ff was declared last).
                eager_tail_keepalive = (Some(ff), Some(normalized));
            }
            #[cfg(test)]
            capture((index, state.pos), "output", &mut self.kernels, &x)?;
            let final_layer = index + 1 == self.layers.len();
            if chain && ((index + 1) % group == 0 || final_layer) {
                // Root the whole group: the residual x plus every plane the
                // group appended into, so no fence is needed inside it.
                let mut roots: Vec<&ExecutionTensor> = vec![&x];
                // Group start for any boundary index, partial final groups
                // included (index - index % group roots every plane in the
                // group, not just the boundary layer's).
                let first = index - index % group;
                for plane in &state.planes[first..=index] {
                    if let Some(held) = plane.as_ref() {
                        roots.extend(held.append_roots());
                    }
                }
                self.kernels.evaluate(&roots)?;
            }
            let final_no_head = final_layer && !logits;
            if !chain && (!coalesce_residual || final_no_head) {
                self.kernels.evaluate(&[&x])?;
            }
            #[cfg(test)]
            layer_evaluate_span.finish(Some((x.dtype(), x.shape())));
            drop(eager_tail_keepalive);
        }
        state.pos += rows; // bounded by ChunkCachePlan before any layer write
        if !logits {
            return mlx_route::read_step(&mut self.kernels, None, &health);
        }
        #[cfg(test)]
        let span = Span::begin(
            "final_head",
            state.pos - 1,
            None,
            Some((x.dtype(), x.shape())),
        );
        // Only the last row reaches the vocabulary projection.
        let last = self.kernels.slice(&x, &[rows - 1, 0], &[rows, 2048])?;
        let normalized = self.kernels.rms_norm(&last, &self.norm, 1e-6)?;
        let logits = self.kernels.qmm(&normalized, &self.head)?;
        #[cfg(test)]
        span.finish(Some((logits.dtype(), logits.shape())));
        #[cfg(test)]
        let span = Span::begin(
            "head_readback",
            state.pos - 1,
            None,
            Some((logits.dtype(), logits.shape())),
        );
        let row = mlx_route::read_step(&mut self.kernels, Some(&logits), &health)?;
        #[cfg(test)]
        span.finish(Some((Dtype::F32, &[1, 150_144])));
        Ok(row)
    }
}

fn attention(
    k: &mut MlxKernels,
    w: &Layer,
    x: &ExecutionTensor,
    held: &mut Option<Plane>,
    plan: &ChunkCachePlan,
) -> Result<ExecutionTensor> {
    #[cfg(test)]
    let span = Span::begin(
        "attn_qkv",
        plan.p0,
        Some(plan.step.debug_layer),
        Some((x.dtype(), x.shape())),
    );
    let q = k.qmm(x, &w.q)?;
    let keys = k.qmm(x, &w.k)?;
    let values = k.qmm(x, &w.v)?;
    #[cfg(test)]
    {
        let coordinate = (plan.step.debug_layer, plan.p0);
        capture(coordinate, "q-projection", k, &q)?;
        capture(coordinate, "k-projection", k, &keys)?;
        capture(coordinate, "v-projection", k, &values)?;
    }
    let q = split_heads(k, &q, 32)?;
    let keys = split_heads(k, &keys, 4)?;
    let values = split_heads(k, &values, 4)?;
    #[cfg(test)]
    if wanted(plan.step.debug_layer, plan.p0) {
        let (pre_q, pre_k) =
            precision_native::normalize_qk(k, (&q, &keys), (&w.q_norm, &w.k_norm), None)?;
        capture((plan.step.debug_layer, plan.p0), "q-pre", k, &pre_q)?;
        capture((plan.step.debug_layer, plan.p0), "k-pre", k, &pre_k)?;
    }
    // The chunk plan stores its last row; RoPE starts at the FIRST row.
    let rope = plan.step.rope.map(|_| plan.p0);
    let (q, keys) = precision_native::normalize_qk(k, (&q, &keys), (&w.q_norm, &w.k_norm), rope)?;
    #[cfg(test)]
    {
        capture((plan.step.debug_layer, plan.p0), "q-post", k, &q)?;
        capture((plan.step.debug_layer, plan.p0), "k-post", k, &keys)?;
        capture((plan.step.debug_layer, plan.p0), "sink", k, &w.sink)?;
    }
    #[cfg(test)]
    span.finish(Some((keys.dtype(), keys.shape())));
    #[cfg(test)]
    let span = Span::begin(
        "attn_sdpa",
        plan.p0,
        Some(plan.step.debug_layer),
        Some((q.dtype(), q.shape())),
    );
    // The chained commit needs the union views this layer's attention reads;
    // keep them before the reshapes below shadow the bindings. Same routing
    // predicate as the layer loop: only one-row chunks chain, and the SAME
    // decision tells prepare_chunk which commit will consume the guard.
    let chain = chain_routes(plan.rows);
    #[cfg(test)]
    let sub = Span::begin(
        "attn_prepare",
        plan.p0,
        Some(plan.step.debug_layer),
        Some((keys.dtype(), keys.shape())),
    );
    let (union_keys, union_values, guard) =
        Plane::prepare_chunk(k, held, plan, &keys, &values, chain)?;
    #[cfg(test)]
    sub.finish(Some((union_keys.dtype(), union_keys.shape())));
    let slots = union_keys.shape()[1];
    let q = k.reshape(&q, &[1, 32, plan.rows, 128])?;
    let keys = k.reshape(&union_keys, &[1, 4, slots, 128])?;
    let values = k.reshape(&union_values, &[1, 4, slots, 128])?;
    let window = rope.map(|_| WINDOW);
    let key_origin = if window.is_some() {
        plan.p0.saturating_sub(WINDOW - 1)
    } else {
        0
    };
    let output = k.shared_attention(SharedAttention {
        q: &q,
        kv: KvBlock::Float {
            keys: &keys,
            values: &values,
        },
        query_start: plan.p0,
        key_origin,
        window,
        sinks: Some(&w.sink),
        scale: 1.0 / 128.0_f32.sqrt(),
    })?;
    let output = merge_heads(k, &output)?;
    #[cfg(test)]
    capture(
        (plan.step.debug_layer, plan.p0),
        "attention-out",
        k,
        &output,
    )?;
    #[cfg(test)]
    span.finish(Some((output.dtype(), output.shape())));
    #[cfg(test)]
    let span = Span::begin(
        "attn_oproj",
        plan.p0,
        Some(plan.step.debug_layer),
        Some((output.dtype(), output.shape())),
    );
    #[cfg(test)]
    let sub = Span::begin(
        "attn_oproj_qmm",
        plan.p0,
        Some(plan.step.debug_layer),
        Some((output.dtype(), output.shape())),
    );
    let projected = k.qmm(&output, &w.o)?;
    #[cfg(test)]
    sub.finish(Some((projected.dtype(), projected.shape())));
    #[cfg(test)]
    let sub = Span::begin(
        "attn_commit",
        plan.p0,
        Some(plan.step.debug_layer),
        Some((projected.dtype(), projected.shape())),
    );
    if chain {
        Plane::commit_chunk_chained(k, held, guard, &union_keys, &union_values)?;
    } else {
        Plane::commit_chunk(k, held, guard, &projected)?;
    }
    #[cfg(test)]
    sub.finish(Some((projected.dtype(), projected.shape())));
    #[cfg(test)]
    span.finish(Some((projected.dtype(), projected.shape())));
    Ok(projected)
}

fn split_heads(k: &mut MlxKernels, x: &ExecutionTensor, heads: u32) -> Result<ExecutionTensor> {
    let &[rows, _] = x.shape() else {
        return Err(EngError::BadRequest(
            "Flint projection must be [T,H*D]".into(),
        ));
    };
    let x = k.reshape(x, &[rows, heads, 128])?;
    k.transpose(&x, &[1, 0, 2])
}

fn merge_heads(k: &mut MlxKernels, x: &ExecutionTensor) -> Result<ExecutionTensor> {
    let &[1, 32, rows, 128] = x.shape() else {
        return Err(EngError::BadRequest(
            "Flint attention must be [1,32,T,128]".into(),
        ));
    };
    let x = k.reshape(x, &[32, rows, 128])?;
    let x = k.transpose(&x, &[1, 0, 2])?;
    k.reshape(&x, &[rows, 4096])
}

fn ffn(
    k: &mut MlxKernels,
    w: &Ffn,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, Option<ExecutionTensor>)> {
    match w {
        Ffn::Dense(w) => Ok((w.apply(k, x)?, None)),
        Ffn::Moe {
            router,
            routed,
            shared,
        } => {
            // pp attribution subs (test builds): ffn carries no position, so
            // these spans carry pos 0 + no layer; the analyzer keys them by
            // stage name only.
            #[cfg(test)]
            let span = Span::begin("ffn_router", 0, None, Some((x.dtype(), x.shape())));
            let logits = k.mm(x, router)?;
            let route = mlx_route::select(k, &logits)?;
            #[cfg(test)]
            stage_fence(k, "ffn_router", route.ids.ids())?;
            #[cfg(test)]
            span.finish(Some((logits.dtype(), logits.shape())));
            #[cfg(test)]
            let span = Span::begin("ffn_experts", 0, None, Some((x.dtype(), x.shape())));
            let rows = grouped_experts(k, routed, x, &route.ids)?;
            #[cfg(test)]
            span.finish(Some((rows.dtype(), rows.shape())));
            #[cfg(test)]
            let span = Span::begin("ffn_shared", 0, None, Some((x.dtype(), x.shape())));
            let shared = shared.apply(k, x)?;
            #[cfg(test)]
            stage_fence(k, "ffn_shared", &shared)?;
            #[cfg(test)]
            span.finish(Some((shared.dtype(), shared.shape())));
            let out = precision_native::combine(k, &rows, &route.weights, &shared)?;
            Ok((out, Some(route.health)))
        }
    }
}

fn grouped_experts(
    k: &mut MlxKernels,
    w: &super::mlx_model::SwiGlu,
    x: &ExecutionTensor,
    ids: &BoundedIndices,
) -> Result<ExecutionTensor> {
    let &[tokens, hidden] = x.shape() else {
        return Err(EngError::BadRequest(
            "Flint expert input must be [T,H]".into(),
        ));
    };
    let pairs = tokens
        .checked_mul(6)
        .ok_or_else(|| EngError::MemoryPlan("Flint expert pair count overflows".into()))?;
    if tokens == 0 || hidden == 0 || pairs > (1 << 24) || ids.ids().shape() != [pairs] {
        return Err(EngError::BadRequest(
            "Flint expert pair geometry is invalid".into(),
        ));
    }
    let keys = k.u32_to_f32(ids.ids())?;
    let order = k.argsort(&keys, 0)?;
    let sorted_ids = k.take_along_axis(ids.ids(), &order, 0)?;
    let sorted_ids = k.bound_indices(&sorted_ids, ids.bound())?;
    let repeated = k.repeat(x, 6, 0)?;
    let sorted_x = k.take(&repeated, &order, 0)?;
    let sorted_x = k.reshape(&sorted_x, &[pairs, 1, hidden])?;
    let gate = k.gather_qmm_sorted_bounded(&sorted_x, &w.gate, &sorted_ids)?;
    let up = k.gather_qmm_sorted_bounded(&sorted_x, &w.up, &sorted_ids)?;
    let gate = k.silu(&gate)?;
    let product = k.binary(BinaryOp::Mul, &gate, &up)?;
    #[cfg(test)]
    stage_fence(k, "ffn_gate_up", &product)?;
    let output = k.gather_qmm_sorted_bounded(&product, &w.down, &sorted_ids)?;
    let output = k.reshape(&output, &[pairs, hidden])?;
    #[cfg(test)]
    stage_fence(k, "ffn_down", &output)?;
    // Restore token/rank order before BF16 weighted reduction, not after it.
    let order = k.u32_to_f32(&order)?;
    let inverse = k.argsort(&order, 0)?;
    let ordered = k.take(&output, &inverse, 0)?;
    k.reshape(&ordered, &[tokens, 6, hidden])
}

#[cfg(test)]
pub(super) fn import_f32(
    k: &mut MlxKernels,
    shape: &[u32],
    values: &[f32],
) -> Result<ExecutionTensor> {
    let host = Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )?;
    k.import_tensor(&host)
}

#[cfg(test)]
#[path = "mlx_forward_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mlx_row_layout_tests.rs"]
mod row_layout_tests;
