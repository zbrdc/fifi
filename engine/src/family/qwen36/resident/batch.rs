//! M1 row-batched decode (the mistralrs design, coordinator-corrected): the
//! row-wise operations batch across the window's B slots (embedding, norms,
//! projections, router, MoE including the shared expert, final norm, head) -
//! the weight-read amortization - while attention and the DeltaNet recurrence
//! run PER SLOT on each slot's own KV ring and state, inside the layer. A
//! batched multi-sequence attention kernel is a later, profile-gated item only.
//!
//! `decode_rows` batches uncompiled host routing and explicitly prepared compiled
//! device routing for Generate/Logits. Other windows keep the solo path.

use super::decoder::{ResidentDecoder, StepMode, StepResult};
use super::state::ResidentState;
use super::weights::RFeedForward;
use crate::EngError;

#[cfg(test)]
#[path = "batch_real_tests.rs"]
mod real_identity_tests;

#[cfg(test)]
std::thread_local! {
    static BATCH_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl ResidentDecoder<'_> {
    /// Returns results in slot order. Host refusals change nothing; a failure after
    /// execution starts poisons the entire window, including any committed slots.
    /// Compiled windows with 3/5/6/7 rows retain solo fallback and per-row weight reads.
    pub(super) fn decode_rows(
        &mut self,
        states: &mut [&mut ResidentState],
        tokens: &[u32],
        mode: StepMode,
    ) -> Result<Vec<StepResult>, EngError> {
        if states.is_empty() || states.len() != tokens.len() {
            return Err(EngError::BadRequest(
                "decode_rows: the states and tokens must be the same non-empty window".into(),
            ));
        }
        let permits = self.preflight_rows(states, tokens)?;
        if states.len() > 1
            && matches!(mode, StepMode::Generate | StepMode::Logits)
            && (self.compiled_rows_ready(states.len())
                || (self.routing == super::moe::Routing::Host
                    && !super::compiled::step_requested(self.compile)))
        {
            return self.run_rows(states, tokens, mode, permits);
        }
        let olds: Vec<_> = states.iter_mut().map(|state| state.begin()).collect();
        let results: Result<Vec<_>, _> = states
            .iter_mut()
            .zip(olds)
            .zip(permits)
            .zip(tokens)
            .map(|(((state, old), permit), &token)| {
                super::step::run(self, state, old, permit, token, mode)
            })
            .collect();
        if results.is_err() {
            // No partial results escape, so no slot may silently resume past a lost token.
            for state in states {
                drop(state.begin());
            }
        }
        results
    }

    pub(super) fn step_rows(
        &mut self,
        states: &mut [&mut ResidentState],
        tokens: &[u32],
    ) -> Result<Vec<StepResult>, EngError> {
        self.decode_rows(states, tokens, StepMode::Generate)
    }
}

impl ResidentDecoder<'_> {
    fn preflight_rows(
        &mut self,
        states: &[&mut ResidentState],
        tokens: &[u32],
    ) -> Result<Vec<crate::sched::GrowthPermit>, EngError> {
        u32::try_from(states.len())
            .map_err(|_| EngError::BadRequest("batch row count exceeds u32".into()))?;
        for state in states {
            state.check_step(&self.weights, self.context_limit)?;
        }
        if tokens
            .iter()
            .any(|&token| token as usize >= self.weights.shape.vocab)
        {
            return Err(EngError::BadRequest(
                "token is outside the embedding table".into(),
            ));
        }
        let growth = super::state::kv_growth_bytes(&self.weights)?;
        (0..states.len())
            .map(|_| self.budget.acquire_growth(growth))
            .collect()
    }

    /// The caller supplies all permits after checking the whole window. A graph/readback
    /// error poisons the entire window until reset. Tokens and publication charges
    /// are checked for all slots before the first commit.
    fn run_rows(
        &mut self,
        states: &mut [&mut ResidentState],
        tokens: &[u32],
        mode: StepMode,
        permits: Vec<crate::sched::GrowthPermit>,
    ) -> Result<Vec<StepResult>, EngError> {
        use super::moe;
        use super::state::LayerState;
        use super::step::{check_dtype, check_state_dtypes};
        use super::weights::{RMixer, ResidentWeights};
        use crate::kernels::BinaryOp;
        use crate::tensor::Dtype;
        use crate::tensor::ExecutionTensor;

        #[cfg(test)]
        BATCH_CALLS.with(|count| count.set(count.get().saturating_add(1)));
        let rows = states.len();
        let compiled_rows = self.compiled_rows_ready(rows);
        let ckpt = self.ckpt;
        let weights: &ResidentWeights = &self.weights;
        let shape = &weights.shape;
        // A later refusal must not consume an earlier slot's layer states.
        let positions: Vec<_> = states.iter().map(|state| state.pos()).collect();
        let mut olds: Vec<_> = states
            .iter_mut()
            .map(|state| state.begin().into_iter())
            .collect();
        let mut ctx = super::ops::Ctx {
            kernels: &mut self.kernels,
            consts: &weights.consts,
            compile: compiled_rows,
        };
        // The stacked activations.
        let mut h = super::prefill::step::embed_rows(&mut ctx, ckpt, &weights.embed, tokens)?;
        check_dtype("embedding stack", &h, ctx.consts.act)?;
        // Per-slot next layer states, filled layer by layer.
        let mut nexts: Vec<Vec<LayerState>> = (0..rows).map(|_| Vec::new()).collect();
        let mut health: Option<ExecutionTensor> = None;
        for layer in weights.layers.iter() {
            // ONE input norm over all B rows.
            let normed = ctx.kernels.rms_norm(&h, &layer.input_norm, shape.eps)?;
            check_dtype("input norm", &normed, ctx.consts.act)?;
            // Dense projections batch across rows; attention caches and DeltaNet
            // recurrence/conv rings remain independent per slot.
            let mixed_stack = match &layer.mixer {
                RMixer::Attn(w) => {
                    let qp = super::ops::linear_rows(&mut ctx, &w.q, &normed)?;
                    let kp = super::ops::linear_rows(&mut ctx, &w.k, &normed)?;
                    let vp = super::ops::linear_rows(&mut ctx, &w.v, &normed)?;
                    let (qw, kw, vw) = (qp.shape()[1], kp.shape()[1], vp.shape()[1]);
                    let mut o_rows = Vec::with_capacity(rows);
                    for slot in 0..rows {
                        let old = olds[slot].next().ok_or_else(|| EngError::Backend {
                            operation: "qwen36 step_rows",
                            message: "a slot ran out of layer states".into(),
                        })?;
                        let LayerState::Attn(cache) = old else {
                            return Err(EngError::BadRequest(
                                "layer state kind does not match its weights".into(),
                            ));
                        };
                        let s = slot as u32;
                        let qp_s = ctx.kernels.slice(&qp, &[s, 0], &[s + 1, qw])?;
                        let kp_s = ctx.kernels.slice(&kp, &[s, 0], &[s + 1, kw])?;
                        let vp_s = ctx.kernels.slice(&vp, &[s, 0], &[s + 1, vw])?;
                        let (o_row, cache) = super::attn::step_projected(
                            &mut ctx,
                            shape,
                            w,
                            cache,
                            positions[slot],
                            [&qp_s, &kp_s, &vp_s],
                        )?;
                        nexts[slot].push(LayerState::Attn(cache));
                        o_rows.push(o_row);
                    }
                    let parts: Vec<&ExecutionTensor> = o_rows.iter().collect();
                    let o_stack = ctx.kernels.concatenate(&parts, 0)?;
                    super::ops::linear_rows(&mut ctx, &w.o, &o_stack)?
                }
                RMixer::Delta(w) if compiled_rows => {
                    let previous = olds
                        .iter_mut()
                        .map(|old| {
                            old.next().ok_or_else(|| EngError::Backend {
                                operation: "qwen36 step_rows",
                                message: "a slot ran out of layer states".into(),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let delta_states = previous
                        .iter()
                        .map(|old| match old {
                            LayerState::Delta { rec, ring } => Ok((rec, ring)),
                            _ => Err(EngError::BadRequest(
                                "layer state kind does not match its weights".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let got =
                        super::compiled::delta_rows(&mut ctx, shape, w, &delta_states, &normed)?;
                    if got.states.len() != rows {
                        return Err(EngError::Backend {
                            operation: "qwen36 step_rows",
                            message: "compiled DeltaNet returned the wrong state count".into(),
                        });
                    }
                    for (next, (rec, ring)) in nexts.iter_mut().zip(got.states) {
                        next.push(LayerState::Delta { rec, ring });
                    }
                    got.out
                }
                RMixer::Delta(w) => {
                    // Slice 3: the FOUR dense projections (qkv, z, b, a) meet
                    // their weights in ONE qmm each over all B rows; only the
                    // per-slot recurrence (ring, decay, gated delta) stays per
                    // slot, and `w.out` projects the stacked rows once.
                    let u = super::ops::linear_rows(&mut ctx, &w.qkv, &normed)?;
                    let z = super::ops::linear_rows(&mut ctx, &w.z, &normed)?;
                    let b = super::ops::linear_rows(&mut ctx, &w.b, &normed)?;
                    let a = super::ops::linear_rows(&mut ctx, &w.a, &normed)?;
                    let (uw, zw, bw, aw) = (u.shape()[1], z.shape()[1], b.shape()[1], a.shape()[1]);
                    let mut o_rows = Vec::with_capacity(rows);
                    for slot in 0..rows {
                        let old = olds[slot].next().ok_or_else(|| EngError::Backend {
                            operation: "qwen36 step_rows",
                            message: "a slot ran out of layer states".into(),
                        })?;
                        let LayerState::Delta { rec, ring } = old else {
                            return Err(EngError::BadRequest(
                                "layer state kind does not match its weights".into(),
                            ));
                        };
                        let s = slot as u32;
                        let u_s = ctx.kernels.slice(&u, &[s, 0], &[s + 1, uw])?;
                        let z_s = ctx.kernels.slice(&z, &[s, 0], &[s + 1, zw])?;
                        let b_s = ctx.kernels.slice(&b, &[s, 0], &[s + 1, bw])?;
                        let a_s = ctx.kernels.slice(&a, &[s, 0], &[s + 1, aw])?;
                        let (o_row, rec, ring) = super::delta::step_projected(
                            &mut ctx,
                            shape,
                            w,
                            &rec,
                            &ring,
                            [&u_s, &z_s, &b_s, &a_s],
                        )?;
                        nexts[slot].push(LayerState::Delta { rec, ring });
                        o_rows.push(o_row);
                    }
                    let parts: Vec<&ExecutionTensor> = o_rows.iter().collect();
                    let o_stack = ctx.kernels.concatenate(&parts, 0)?;
                    super::ops::linear_rows(&mut ctx, &w.out, &o_stack)?
                }
            };
            let mid = ctx.kernels.binary(BinaryOp::Add, &h, &mixed_stack)?;
            // ONE post norm over all B rows.
            let normed = ctx.kernels.rms_norm(&mid, &layer.post_norm, shape.eps)?;
            check_dtype("post norm", &normed, ctx.consts.act)?;
            // Slice 3: the SHARED expert runs ONCE over the whole row stack
            // (gate/up/down in one qmm each over [B, hidden]).
            // Slice 4: the routed experts batch too - ONE router readback and
            // ONE pair import for the window (route_rows), then ONE gathered
            // product per weight over all B * k (slot, expert) pairs
            // (combine_rows), the union-of-top-k device MoE.
            let total = match (&layer.feed_forward, compiled_rows) {
                // Dense layers carry no router: the plain SwiGLU path serves any row count.
                (RFeedForward::Dense(w), _) => moe::dense_rows(&mut ctx, w, &normed)?,
                (RFeedForward::Moe(w), true) => {
                    let (out, route) =
                        super::compiled::moe_step_device(&mut ctx, shape, w, &normed)?;
                    health = Some(match health.take() {
                        Some(sum) => ctx.kernels.binary(BinaryOp::Add, &sum, &route.health)?,
                        None => route.health,
                    });
                    out
                }
                (RFeedForward::Moe(w), false) => {
                    let shared_stack = moe::shared(&mut ctx, w, &normed)?;
                    let (ids, gates, top_k) = moe::route_rows(&mut ctx, shape, w, &normed)?;
                    let routed_stack =
                        moe::combine_rows(&mut ctx, shape, w, &normed, &ids, &gates, top_k)?;
                    let total = ctx
                        .kernels
                        .binary(BinaryOp::Add, &routed_stack, &shared_stack)?;
                    super::ops::to_act(&mut ctx, &total)?
                }
            };
            h = ctx.kernels.binary(BinaryOp::Add, &mid, &total)?;
            check_dtype("layer output", &h, ctx.consts.act)?;
        }
        // ONE head over all B rows: final norm, one lm_head matmul, one argmax.
        let final_h = ctx.kernels.rms_norm(&h, &weights.final_norm, shape.eps)?;
        let logits = super::ops::linear_rows(&mut ctx, &weights.lm_head, &final_h)?;
        let next = ctx.kernels.argmax(&logits, 1, false)?;
        let head = if compiled_rows {
            let sum = health.ok_or_else(|| EngError::Backend {
                operation: "qwen36 step_rows",
                message: "device routing produced no router health".into(),
            })?;
            let tokens = ctx.kernels.u32_to_f32(&next)?;
            ctx.kernels.concatenate(&[&tokens, &sum], 0)?
        } else {
            next
        };
        // The dtype invariants of the committed state, every slot.
        for layers in &nexts {
            check_state_dtypes(layers, ctx.consts.act)?;
        }
        // One evaluate of every slot's new state and the head output.
        let mut outputs: Vec<&ExecutionTensor> = Vec::new();
        for layers in &nexts {
            for layer in layers {
                match layer {
                    LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
                    LayerState::Attn(cache) => outputs.extend(cache.tensors()),
                }
            }
        }
        outputs.push(&head);
        if mode == StepMode::Logits {
            outputs.push(&logits);
        }
        ctx.kernels.evaluate(&outputs)?;
        // Validate every health row before any slot can publish its state.
        let next_tokens = if compiled_rows {
            device_head_tokens(&super::ops::values(ctx.kernels, &head)?, rows, shape.vocab)?
        } else {
            let host = ctx.kernels.readback(&head)?;
            let bytes = host
                .host_bytes()
                .filter(|_| host.dtype() == Dtype::U32)
                .ok_or_else(|| EngError::Backend {
                    operation: "qwen36 step_rows",
                    message: "the argmax readback is not U32".into(),
                })?;
            if bytes.len() != rows * 4 {
                return Err(EngError::Backend {
                    operation: "qwen36 step_rows",
                    message: format!("expected {rows} tokens, read {}", bytes.len() / 4),
                });
            }
            let next_tokens: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| u32::from_le_bytes(*bytes))
                .collect();
            if next_tokens
                .iter()
                .any(|&token| token as usize >= shape.vocab)
            {
                return Err(EngError::Backend {
                    operation: "qwen36 step_rows",
                    message: "argmax returned a token outside the vocabulary".into(),
                });
            }
            next_tokens
        };
        let mut logit_rows = if mode == StepMode::Logits {
            let flat = super::ops::values(ctx.kernels, &logits)?;
            if flat.len() != rows * shape.vocab {
                return Err(EngError::Backend {
                    operation: "qwen36 step_rows",
                    message: "logits readback has the wrong row count".into(),
                });
            }
            flat.chunks_exact(shape.vocab)
                .map(|row| Some(row.to_vec()))
                .collect::<Vec<_>>()
        } else {
            vec![None; rows]
        };
        for ((state, layers), permit) in states.iter().zip(&nexts).zip(&permits) {
            state.check_commit_rows(1, layers, permit)?;
        }
        let mut results = Vec::with_capacity(rows);
        for (slot, (state, permit)) in states.iter_mut().zip(permits).enumerate() {
            let layers = std::mem::take(&mut nexts[slot]);
            state.commit_rows(1, layers, permit)?;
            results.push(StepResult {
                token: Some(next_tokens[slot]),
                logits: logit_rows[slot].take(),
                layer_out: None,
                routes: None,
                probs: None,
            });
        }
        Ok(results)
    }
}

fn device_head_tokens(head: &[f32], rows: usize, vocab: usize) -> Result<Vec<u32>, EngError> {
    if head.len() != rows * 2 {
        return Err(EngError::Backend {
            operation: "qwen36 step_rows",
            message: "the device head did not return all tokens and health values".into(),
        });
    }
    let (tokens, health) = head.split_at(rows);
    if health.iter().any(|value| !value.is_finite()) {
        return Err(EngError::Backend {
            operation: "route_experts",
            message: "router probabilities must be finite and non-negative".into(),
        });
    }
    tokens
        .iter()
        .map(|&token| {
            let id = token as u32;
            if id as f32 != token || id as usize >= vocab {
                return Err(EngError::Backend {
                    operation: "qwen36 step_rows",
                    message: "argmax returned a token outside the vocabulary".into(),
                });
            }
            Ok(id)
        })
        .collect()
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::super::decoder::{ResidentDecoder, StepMode};

    #[test]
    fn device_head_checks_every_row_before_publication() {
        assert_eq!(
            super::device_head_tokens(&[2.0, 3.0, 1.0, 1.0], 2, 8).unwrap(),
            [2, 3]
        );
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(matches!(
                super::device_head_tokens(&[2.0, 3.0, 1.0, bad], 2, 8),
                Err(crate::EngError::Backend {
                    operation: "route_experts",
                    ..
                })
            ));
        }
        for bad in [f32::NAN, -1.0, 2.5, 8.0] {
            assert!(super::device_head_tokens(&[2.0, bad, 1.0, 1.0], 2, 8).is_err());
        }
        assert!(super::device_head_tokens(&[2.0, 3.0, 1.0], 2, 8).is_err());
    }
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;

    #[test]
    fn unsupported_compiled_row_counts_preserve_solo_outputs_history_and_state() {
        use super::super::{moe::Routing, state::LayerState};
        use crate::family::qwen36::{
            settings::{Activation, KvDtype},
            tiny::TINY_KV8,
        };

        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&TINY_KV8);
        let host = HostWeights::load(&ckpt, &TINY_KV8, &tiny::config(&TINY_KV8)).unwrap();
        let mut decoder = ResidentDecoder::new_with_kv(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            128,
            Activation::Bf16,
            KvDtype::Q8,
        )
        .unwrap();
        decoder.set_routing(Routing::Device);
        decoder.set_compile(true).unwrap();
        decoder.prepare_compiled_rows(&[2, 4, 8]).unwrap();
        assert!(
            [2, 4, 8]
                .into_iter()
                .all(|b| decoder.compiled_rows_ready(b))
        );
        super::BATCH_CALLS.with(|count| count.set(0));
        for rows in [3, 5, 6, 7] {
            assert!(!decoder.compiled_rows_ready(rows));
            let (mut solo, mut fallback) = (Vec::new(), Vec::new());
            for slot in 0..rows {
                let mut a = decoder.zero_state().unwrap();
                let mut b = decoder.zero_state().unwrap();
                for position in 0..slot {
                    let token = ((slot + position) % 8) as u32;
                    decoder.step(&mut a, token, StepMode::Generate).unwrap();
                    decoder.step(&mut b, token, StepMode::Generate).unwrap();
                }
                solo.push(a);
                fallback.push(b);
            }
            let mut solo_history = vec![Vec::new(); rows];
            let mut fallback_history = vec![Vec::new(); rows];
            for (step, mode) in [StepMode::Generate, StepMode::Logits]
                .into_iter()
                .enumerate()
            {
                let tokens: Vec<_> = (0..rows).map(|slot| ((slot + step) % 8) as u32).collect();
                let expected: Vec<_> = solo
                    .iter_mut()
                    .zip(&tokens)
                    .map(|(state, &token)| decoder.step(state, token, mode).unwrap())
                    .collect();
                let mut window: Vec<_> = fallback.iter_mut().collect();
                let actual = decoder.decode_rows(&mut window, &tokens, mode).unwrap();
                assert_eq!(super::BATCH_CALLS.with(std::cell::Cell::get), 0);
                assert_eq!(actual.len(), rows);
                for (slot, (got, want)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(got.token, want.token, "B={rows} slot={slot} step={step}");
                    assert_eq!(got.logits, want.logits, "B={rows} slot={slot} step={step}");
                    fallback_history[slot].push(got.token.unwrap());
                    solo_history[slot].push(want.token.unwrap());
                    assert_eq!(fallback_history[slot], solo_history[slot]);
                    assert_eq!(fallback[slot].pos(), solo[slot].pos());
                    assert_eq!(fallback[slot].pos() as usize, slot + step + 1);
                    assert!(!fallback[slot].is_poisoned());
                    let (got, want) = (fallback[slot].layers(), solo[slot].layers());
                    assert_eq!(got.len(), want.len());
                    for (got, want) in got.iter().zip(want) {
                        let (got, want, len) = match (got, want) {
                            (
                                LayerState::Delta { rec: a, ring: b },
                                LayerState::Delta { rec: c, ring: d },
                            ) => (vec![a, b], vec![c, d], None),
                            (LayerState::Attn(a), LayerState::Attn(b)) => {
                                assert_eq!((a.len, a.kv), (b.len, b.kv));
                                (a.tensors(), b.tensors(), Some(a.len as u32))
                            }
                            _ => panic!("fallback changed the layer state kind"),
                        };
                        assert_eq!(got.len(), want.len());
                        for (got, want) in got.into_iter().zip(want) {
                            // Compare only committed KV history, not unused capacity.
                            let prefix = |kernels: &mut MlxKernels, tensor: &crate::tensor::ExecutionTensor| {
                                if let Some(len) = len {
                                    let mut end = tensor.shape().to_vec();
                                    end[2] = len;
                                    kernels.slice(tensor, &[0; 4], &end).unwrap()
                                } else {
                                    tensor.share()
                                }
                            };
                            let got = prefix(&mut decoder.kernels, got);
                            let want = prefix(&mut decoder.kernels, want);
                            assert_eq!((got.dtype(), got.shape()), (want.dtype(), want.shape()));
                            let got = decoder.kernels.readback(&got).unwrap();
                            let want = decoder.kernels.readback(&want).unwrap();
                            assert_eq!(got.host_bytes().unwrap(), want.host_bytes().unwrap());
                        }
                    }
                }
            }
        }
    }

    /// The bf16 coverage of the identity gate: the batched path under the
    /// daily-engine activation must match its own solo loop exactly too - this
    /// is the dtype plumbing the F32 fixture does not reach (the embed stack's
    /// activation cast most of all).
    #[test]
    fn step_rows_matches_the_solo_loop_per_slot_bf16() {
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let config = tiny::config(&TINY_GQA);
        let make = || {
            let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
            let mut decoder = ResidentDecoder::new_with_activation(
                MlxKernels::try_new().unwrap(),
                &ckpt,
                host,
                1024,
                crate::family::qwen36::settings::Activation::Bf16,
            )
            .unwrap();
            decoder.routing = super::super::moe::Routing::Host;
            decoder
        };
        let mut solo = make();
        let mut solo_state = solo.zero_state().unwrap();
        let mut stream = Vec::with_capacity(6);
        let mut t = 3u32;
        for _ in 0..6 {
            let r = solo.step(&mut solo_state, t, StepMode::Generate).unwrap();
            t = r.token.unwrap();
            stream.push(t);
        }
        for b in [1usize, 2, 4, 8] {
            let mut dec = make();
            let mut states: Vec<_> = (0..b).map(|_| dec.zero_state().unwrap()).collect();
            let mut tokens = vec![3u32; b];
            for (i, want) in stream.iter().enumerate() {
                let mut window: Vec<&mut super::super::state::ResidentState> =
                    states.iter_mut().collect();
                let got = dec.step_rows(&mut window, &tokens).unwrap();
                for (slot, r) in got.iter().enumerate() {
                    assert_eq!(r.token, Some(*want), "bf16 B={b} slot {slot} step {i}");
                }
                tokens = got.iter().map(|r| r.token.unwrap()).collect();
            }
        }
    }

    /// The admission atomicity of `step_rows`: a later slot's refusal must
    /// leave EVERY state usable - no slot pays for another slot's
    /// context-limit breach. A state at its context limit refuses
    /// `check_step`; the healthy state must still step afterwards.
    #[test]
    fn step_rows_refusal_leaves_every_state_unchanged() {
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let config = tiny::config(&TINY_GQA);
        // A tiny context limit: this state reaches it in a few steps.
        let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
        let mut dec = ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host, 4).unwrap();
        dec.routing = super::super::moe::Routing::Host;
        let mut spent = dec.zero_state().unwrap();
        let mut t = 3u32;
        for _ in 0..4 {
            let r = dec.step(&mut spent, t, StepMode::Generate).unwrap();
            t = r.token.unwrap();
        }
        // `spent` sits at its limit; the healthy state has not stepped.
        let mut healthy = dec.zero_state().unwrap();
        {
            let mut window: Vec<&mut super::super::state::ResidentState> =
                vec![&mut healthy, &mut spent];
            let err = dec.step_rows(&mut window, &[7, 9]).unwrap_err();
            assert!(matches!(err, crate::EngError::ContextLimit(_)), "{err:?}");
        }
        // The refusal poisoned nothing: the healthy state steps on its own.
        let r = dec.step(&mut healthy, 7, StepMode::Generate).unwrap();
        assert!(r.token.is_some());
    }

    /// Real-weight diagnostic, run by name only. Checks greedy tokens at
    /// B=2/4/8 against solo streams with distinct initial tokens; it does not
    /// establish logit identity. Reports uncompiled BF16 host-routing decode
    /// timing and serial prefill service time, not integrated scheduler TTFT or
    /// a compiled daily-engine comparison. Ignored by default: set Q36_CKPT_DIR
    /// to the real checkpoint.
    #[test]
    #[ignore = "real weights: set Q36_CKPT_DIR"]
    fn real_weights_identity_distinct_prompts_and_timing() {
        let _serial = crate::native_test_lock();
        use super::super::real::{env_path, open};
        use crate::family::spec::QWEN3_6_35B_A3B;
        let ckpt_dir = env_path("Q36_CKPT_DIR");
        let real = open(&ckpt_dir);
        let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
        // This diagnostic uses BF16; it is not a compiled daily-engine baseline.
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &real.ckpt,
            host,
            2048,
            crate::family::qwen36::settings::Activation::Bf16,
        )
        .unwrap();
        decoder.routing = super::super::moe::Routing::Host;
        println!("S4 diagnostic: BF16, host routing, uncompiled; NOT baseline-comparable");
        const STEPS: usize = 6;
        // (1) Identity, distinct prompts per slot.
        for b in [2usize, 4, 8] {
            let mut solo = Vec::with_capacity(b);
            for s in 0..b {
                let mut state = decoder.zero_state().unwrap();
                let mut t = 3 + s as u32;
                let mut stream = Vec::with_capacity(STEPS);
                for _ in 0..STEPS {
                    let r = decoder.step(&mut state, t, StepMode::Generate).unwrap();
                    t = r.token.unwrap();
                    stream.push(t);
                }
                solo.push(stream);
            }
            let mut states: Vec<_> = (0..b).map(|_| decoder.zero_state().unwrap()).collect();
            let mut tokens: Vec<u32> = (0..b).map(|s| 3 + s as u32).collect();
            for (step_i, _) in solo[0].iter().enumerate() {
                let mut window: Vec<&mut super::super::state::ResidentState> =
                    states.iter_mut().collect();
                let got = decoder.step_rows(&mut window, &tokens).unwrap();
                for s in 0..b {
                    assert_eq!(
                        got[s].token,
                        Some(solo[s][step_i]),
                        "B={b} slot {s} step {step_i}"
                    );
                    tokens[s] = got[s].token.unwrap();
                }
            }
            println!(
                "identity B={b}: PASS ({} steps, distinct prompts, tokens exact)",
                STEPS
            );
        }
        // (2) Decode timing and serial prompt-feed service time.
        for n in [1usize, 2, 4, 8] {
            let mut states: Vec<_> = (0..n).map(|_| decoder.zero_state().unwrap()).collect();
            let mut tokens: Vec<u32> = (0..n).map(|s| 1000 + 10 * s as u32).collect();
            // Warm-up: 2 steps.
            for _ in 0..2 {
                let mut window: Vec<&mut super::super::state::ResidentState> =
                    states.iter_mut().collect();
                let got = decoder.step_rows(&mut window, &tokens).unwrap();
                for s in 0..n {
                    tokens[s] = got[s].token.unwrap();
                }
            }
            // Timed: 24 steps, wall clock.
            let start = std::time::Instant::now();
            let k = 24;
            for _ in 0..k {
                let mut window: Vec<&mut super::super::state::ResidentState> =
                    states.iter_mut().collect();
                let got = decoder.step_rows(&mut window, &tokens).unwrap();
                for s in 0..n {
                    tokens[s] = got[s].token.unwrap();
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            let agg = (n * k) as f64 / elapsed;
            println!(
                "timing N={n}: {:.1} tok/s aggregate ({:.1} tok/s per slot), {:.0} ms/step",
                agg,
                k as f64 / elapsed,
                elapsed / k as f64 * 1000.0
            );
            // Serial service time only; this does not exercise scheduler admission.
            let mut state = decoder.zero_state().unwrap();
            let prompt: Vec<u32> = (0..8).map(|i| 2000 + i).collect();
            let t0 = std::time::Instant::now();
            let mut t = prompt[0];
            for &tok in &prompt {
                let r = decoder.step(&mut state, tok, StepMode::Generate).unwrap();
                t = r.token.unwrap();
            }
            let first = decoder.step(&mut state, t, StepMode::Generate).unwrap();
            let ttft = t0.elapsed().as_secs_f64();
            assert!(first.token.is_some());
            println!(
                "timing N={n}: serial prompt-feed plus extra decode {:.2} s (8+1 calls; excludes queue/admission)",
                ttft
            );
        }
        // (3) Prefill service time while seven resident slots are paused. Prefill
        // already returns the first token; queue/admission time is not measured.
        {
            let mut states: Vec<_> = (0..7).map(|_| decoder.zero_state().unwrap()).collect();
            let mut tokens: Vec<u32> = (0..7).map(|s| 5000 + 10 * s as u32).collect();
            for _ in 0..4 {
                let mut window: Vec<&mut super::super::state::ResidentState> =
                    states.iter_mut().collect();
                let got = decoder.step_rows(&mut window, &tokens).unwrap();
                for s in 0..7 {
                    tokens[s] = got[s].token.unwrap();
                }
            }
            let t0 = std::time::Instant::now();
            let mut new_state = decoder.zero_state().unwrap();
            let prompt: Vec<u32> = (0..128).map(|i| 3000 + i).collect();
            let first = decoder
                .prefill(&mut new_state, &prompt, StepMode::Generate)
                .unwrap();
            let prefill_seconds = t0.elapsed().as_secs_f64();
            states.push(new_state);
            tokens.push(first.token.unwrap());
            let mut window: Vec<&mut super::super::state::ResidentState> =
                states.iter_mut().collect();
            let _got = decoder.step_rows(&mut window, &tokens).unwrap();
            println!(
                "newcomer prefill service time (128-token prompt, seven paused slots; excludes queue/admission): {:.2} s",
                prefill_seconds
            );
        }
        println!("S4 real-weights gate complete");
    }

    /// B=1 exercises the solo fallback. B=2/4/8 use the same token stream
    /// per slot and must match the solo greedy tokens through the batch core.
    #[test]
    fn step_rows_matches_the_solo_loop_per_slot() {
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let config = tiny::config(&TINY_GQA);
        let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
        let mut solo =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host, 1024).unwrap();
        // The solo reference stream: 8 Generate steps from token 3.
        let mut solo_state = solo.zero_state().unwrap();
        let mut solo_tokens = Vec::with_capacity(8);
        let mut t = 3u32;
        for _ in 0..8 {
            let r = solo.step(&mut solo_state, t, StepMode::Generate).unwrap();
            t = r.token.unwrap();
            solo_tokens.push(t);
        }
        // B=1: the exact solo match through the fallback.
        {
            let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
            let mut dec =
                ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host, 1024).unwrap();
            let mut state = dec.zero_state().unwrap();
            let mut t = 3u32;
            for (step_i, want) in solo_tokens.iter().enumerate() {
                let mut window: Vec<&mut super::super::state::ResidentState> = vec![&mut state];
                let before = super::BATCH_CALLS.with(std::cell::Cell::get);
                let got = dec.step_rows(&mut window, &[t]).unwrap().remove(0);
                assert_eq!(super::BATCH_CALLS.with(std::cell::Cell::get), before);
                assert_eq!(got.token, Some(*want), "B=1 step {step_i}");
                t = got.token.unwrap();
            }
        }
        // B=2/4/8: every slot steps the same stream; every slot must match solo.
        for b in [2usize, 4, 8] {
            let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
            let mut dec =
                ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host, 1024).unwrap();
            dec.routing = super::super::moe::Routing::Host;
            let mut states: Vec<_> = (0..b).map(|_| dec.zero_state().unwrap()).collect();
            let mut t = 3u32;
            for (step_i, want) in solo_tokens.iter().enumerate() {
                let tokens = vec![t; b];
                let mut window: Vec<&mut super::super::state::ResidentState> =
                    states.iter_mut().collect();
                let got = dec.step_rows(&mut window, &tokens).unwrap();
                for (slot, r) in got.iter().enumerate() {
                    assert_eq!(r.token, Some(*want), "B={b} slot {slot} step {step_i}");
                }
                t = got[0].token.unwrap();
            }
        }
    }
}
