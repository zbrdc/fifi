//! The host-side class partition of the one-token decode step: the Flint class-partition
//! method ported to Q36, because headless xctrace runs produced zero shader rows.
//!
//! One ignored Mac-only real-checkpoint test, `decode_class_partition`, builds the
//! decoder exactly like `decode_timing` and runs a TEST-ONLY REPLICA of `step.rs::run`'s
//! layer loop that calls the same resident functions with one counter-bracketed forced
//! evaluate at every op-class boundary. The bracket is the qualification counter pair the
//! phase harness already uses (`MlxKernels::qualification_reset_timing` /
//! `qualification_timing`), so a class's cost is its
//! `evaluate_ns`/`import_ns`/`readback_ns` sums, not a host wall. The classes sum to an
//! ATTRIBUTION, not a latency: the forced syncs serialize the graph. The empty-sync
//! control (a bracket holding only the forced evaluate) is measured in the same process
//! and subtracted per sync.
//!
//! The class boundaries split AROUND the compiled blocks, never inside them:
//! `delta_block` and `moe_block` time each compiled closure as one entry, and
//! the forced sync defeats fusion only across boundaries. The compiled
//! blocks' production cost comes from the non-split compile A/B already
//! banked in `decode_timing`; this run attributes the EAGER remainder and the
//! depth scaling of every class.
//!
//! Classes: `embed`, `input_norm`, `delta_block`, `attn_qkv_proj`,
//! `attn_core`, `attn_out_proj`, `residual`, `post_norm`, `moe_block`, `head`.
//!
//! Practices carried from the Flint method: 128 forwards per depth; all prefill and growth
//! before the timed window; the FIRST depth of a fresh process is JIT-cold, so a warm-up
//! depth pass precedes the reported ones; `evaluate_calls` is asserted against the
//! expected per-step count; when the summed class walls exceed the plain step wall the
//! overlap is reported, not hidden. Parity gate: the replica's greedy token chain must
//! equal the plain `ResidentDecoder::step` (`StepMode::Generate`) chain step for step — a
//! mismatch means the attribution is of a different computation and fails the run.
//!
//! Inputs by environment: the `Q36_*`/`DT_*` knobs of `decode_timing`
//! (`Q36_CKPT_DIR`, `Q36_OUT_DIR`, `Q36_PROMPT_IDS`, `DT_DEPTHS`,
//! `DT_TOKENS`) plus `QWEN36_G2A_COMPILE`. Report:
//! `decode_class_partition.json` in `Q36_OUT_DIR`.

use super::attn;
use super::decoder::ResidentDecoder;
use super::delta;
use super::moe;
use super::moe::Routing;
use super::ops::{Ctx, linear, short_import, to_act};
use super::state::{LayerState, ResidentState, kv_growth_bytes};
use super::step::read_head;
use super::weights::{RFeedForward, RMixer};
use crate::EngError;
use crate::family::flash_next::stored;
use crate::kernels::BinaryOp;
use crate::kernels::QualificationTiming;
use crate::tensor::ExecutionTensor;
use serde::Serialize;

/// The class names, in loop order.
const CLASSES: [&str; 12] = [
    "embed",
    "input_norm",
    "delta_block",
    "attn_qkv_proj",
    "attn_core",
    "attn_out_proj",
    "residual",
    "post_norm",
    "moe_block",
    "health",
    "residual_out",
    "head",
];

#[derive(Serialize)]
struct ClassRow {
    evaluate_ms_per_step: f64,
    import_ms_per_step: f64,
    readback_ms_per_step: f64,
    evaluate_ms_minus_empty: f64,
    forced_syncs_per_step: f64,
}

/// The per-class accumulator of one timed window.
#[derive(Default)]
struct ClassSums {
    evaluate_ns: u128,
    import_ns: u128,
    readback_ns: u128,
    syncs: u64,
}

impl ClassSums {
    fn row(&self, steps: f64, empty_ms: f64) -> ClassRow {
        let per = |ns: u128| ns as f64 / 1e6 / steps;
        ClassRow {
            evaluate_ms_per_step: per(self.evaluate_ns),
            import_ms_per_step: per(self.import_ns),
            readback_ms_per_step: per(self.readback_ns),
            evaluate_ms_minus_empty: per(self.evaluate_ns) - empty_ms * self.syncs as f64 / steps,
            forced_syncs_per_step: self.syncs as f64 / steps,
        }
    }
}

/// One forced evaluate of `sink` inside a counter bracket charged to `class`.
pub(super) struct Partitioner {
    sink: ExecutionTensor,
    sums: Vec<ClassSums>,
    empty_evaluate_ms: f64,
}

impl Partitioner {
    fn new(sink: ExecutionTensor) -> Self {
        Self {
            sink,
            sums: CLASSES.iter().map(|_| ClassSums::default()).collect(),
            empty_evaluate_ms: 0.0,
        }
    }

    fn index(class: &str) -> usize {
        CLASSES
            .iter()
            .position(|name| *name == class)
            .unwrap_or_else(|| panic!("unknown class {class}"))
    }

    /// Forces `outputs`' lazy graph to the device inside a counter bracket and
    /// charges the counters to `class`. The ancestors of `outputs` were forced
    /// by their own earlier marks, so the bracket holds this class's slice
    /// alone; the empty-sync control (the sink bracket of `calibrate`) prices
    /// the bare evaluate.
    fn mark(
        &mut self,
        ctx: &mut Ctx<'_>,
        class: &str,
        outputs: &[&ExecutionTensor],
    ) -> Result<(), EngError> {
        ctx.kernels.qualification_reset_timing();
        ctx.kernels.evaluate(outputs)?;
        let t: QualificationTiming = ctx.kernels.qualification_timing();
        let i = Self::index(class);
        let s = &mut self.sums[i];
        s.evaluate_ns += t.evaluate_ns as u128;
        s.import_ns += t.import_ns as u128;
        s.readback_ns += t.readback_ns as u128;
        s.syncs += 1;
        Ok(())
    }
}

/// Builds the partitioner's resident sink: token 0's embedding row, imported once.
/// Any resident tensor works as the forced-evaluate sink; this one needs no new API.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn partitioner_of(decoder: &mut ResidentDecoder<'_>) -> Partitioner {
    let row = stored::read_slice(decoder.ckpt, &decoder.weights.embed, 0)
        .expect("token 0 embedding row reads");
    let resident = decoder
        .kernels
        .import_packed_many(&[&row])
        .expect("sink import")
        .pop()
        .expect("one imported sink");
    let sink = decoder.kernels.dequantize(&resident).expect("sink dequant");
    Partitioner::new(sink)
}

/// One marked Generate step: the permit/begin/commit handling of `step::step`
/// around the marked replica.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn one_marked_token(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    token: u32,
    part: &mut Partitioner,
) -> Result<u32, EngError> {
    state.check_step(&decoder.weights, decoder.context_limit)?;
    let growth = kv_growth_bytes(&decoder.weights)?;
    let permit = decoder.budget.acquire_growth(growth)?;
    let old = state.begin();
    run_marked(decoder, state, old, permit, token, part)
}

impl Partitioner {
    /// Zeroes the per-class sums (after the warm-up tokens, before the timed
    /// window).
    pub(super) fn reset_sums(&mut self) {
        self.sums = CLASSES.iter().map(|_| ClassSums::default()).collect();
    }

    /// Prices the forced evaluate alone, inside `decoder`'s context.
    pub(super) fn calibrate(&mut self, decoder: &mut ResidentDecoder<'_>, rounds: usize) {
        let weights = &decoder.weights;
        let ctx = Ctx {
            kernels: &mut decoder.kernels,
            consts: &weights.consts,
            compile: decoder.compile,
        };
        let mut total = 0u128;
        for _ in 0..rounds {
            ctx.kernels.qualification_reset_timing();
            ctx.kernels.evaluate(&[&self.sink]).expect("sink evaluate");
            let t: QualificationTiming = ctx.kernels.qualification_timing();
            total += t.evaluate_ns as u128;
        }
        self.empty_evaluate_ms = total as f64 / 1e6 / rounds as f64;
    }

    /// The per-class rows of a timed window of `steps` marked tokens.
    pub(super) fn class_rows(&self, steps: usize) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (i, name) in CLASSES.iter().enumerate() {
            map.insert(
                (*name).to_string(),
                serde_json::to_value(self.sums[i].row(steps as f64, self.empty_evaluate_ms))
                    .unwrap(),
            );
        }
        serde_json::Value::Object(map)
    }
}

/// The replica of `step.rs::run` in Generate mode with one mark per class
/// occurrence. Calls the same resident functions as the real step; the layer
/// state handling mirrors it so the graphs are identical.
fn run_marked(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    old: Vec<LayerState>,
    permit: crate::sched::GrowthPermit,
    token: u32,
    part: &mut Partitioner,
) -> Result<u32, EngError> {
    let weights = &decoder.weights;
    let shape = &weights.shape;
    let eps = shape.eps;
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &weights.consts,
        compile: decoder.compile,
    };
    let mut h = {
        let row = stored::read_slice(decoder.ckpt, &weights.embed, token)?;
        let resident = ctx
            .kernels
            .import_packed_many(&[&row])?
            .pop()
            .ok_or_else(short_import)?;
        let row = ctx.kernels.dequantize(&resident)?;
        to_act(&mut ctx, &row)?
    };
    part.mark(&mut ctx, "embed", &[&h])?;
    // The device-routing health sum, accumulated layer by layer exactly as the
    // real step does so the marked chain keeps those reductions.
    let mut health = None;
    let mut layers = Vec::with_capacity(old.len());
    for (layer, old_layer) in weights.layers.iter().zip(old) {
        let normed = ctx.kernels.rms_norm(&h, &layer.input_norm, eps)?;
        part.mark(&mut ctx, "input_norm", &[&normed])?;
        let (mixed, next) = match (&layer.mixer, old_layer) {
            (RMixer::Delta(w), LayerState::Delta { rec, ring }) => {
                let s = delta::step(&mut ctx, shape, w, &rec, &ring, &normed)?;
                // The state update is part of this class: evaluating only `out`
                // would leave rec/ring lazy and charge them to the next class.
                part.mark(&mut ctx, "delta_block", &[&s.out, &s.rec, &s.ring])?;
                (
                    s.out,
                    LayerState::Delta {
                        rec: s.rec,
                        ring: s.ring,
                    },
                )
            }
            (RMixer::Attn(w), LayerState::Attn(cache)) => {
                let qp = linear(&mut ctx, &w.q, &normed)?;
                let kp = linear(&mut ctx, &w.k, &normed)?;
                let vp = linear(&mut ctx, &w.v, &normed)?;
                part.mark(&mut ctx, "attn_qkv_proj", &[&qp, &kp, &vp])?;
                let (o, cache) =
                    attn::step_projected(&mut ctx, shape, w, cache, state.pos(), [&qp, &kp, &vp])?;
                part.mark(&mut ctx, "attn_core", &[&o])?;
                let out = linear(&mut ctx, &w.o, &o)?;
                part.mark(&mut ctx, "attn_out_proj", &[&out])?;
                (out, LayerState::Attn(cache))
            }
            _ => {
                return Err(EngError::BadRequest(
                    "layer state kind does not match its weights".into(),
                ));
            }
        };
        layers.push(next);
        let mid = ctx.kernels.binary(BinaryOp::Add, &h, &mixed)?;
        part.mark(&mut ctx, "residual", &[&mid])?;
        let normed = ctx.kernels.rms_norm(&mid, &layer.post_norm, eps)?;
        part.mark(&mut ctx, "post_norm", &[&normed])?;
        let routed = match &layer.feed_forward {
            RFeedForward::Dense(w) => {
                let out = moe::dense_rows(&mut ctx, w, &normed)?;
                part.mark(&mut ctx, "moe_block", &[&out])?;
                out
            }
            RFeedForward::Moe(w) => {
                let (out, route) = moe::step_device(&mut ctx, shape, w, &normed)?;
                health = Some(match health.take() {
                    Some(sum) => ctx.kernels.binary(BinaryOp::Add, &sum, &route.health)?,
                    None => route.health,
                });
                // routed first: the router matmul chain charges to the block,
                // and the health mark then adds only this layer's reduction.
                part.mark(&mut ctx, "moe_block", &[&out])?;
                part.mark(
                    &mut ctx,
                    "health",
                    &[health.as_ref().expect("health just set")],
                )?;
                out
            }
        };
        h = ctx.kernels.binary(BinaryOp::Add, &mid, &routed)?;
        // Own class: without this mark the final add is charged to the next
        // layer's input_norm (or the head).
        part.mark(&mut ctx, "residual_out", &[&h])?;
    }
    // The Generate head: final norm, lm_head, argmax, the device-routing health
    // pair, token readback. The pair's readback lands after the marks, so the
    // readback count is its own class (a recorded limitation).
    let final_h = ctx.kernels.rms_norm(&h, &weights.final_norm, eps)?;
    let logits = linear(&mut ctx, &weights.lm_head, &final_h)?;
    let next = ctx.kernels.argmax(&logits, 1, false)?;
    let head_out = match health.as_ref() {
        Some(sum) => {
            let token = ctx.kernels.u32_to_f32(&next)?;
            Some(ctx.kernels.concatenate(&[&token, sum], 0)?)
        }
        None => None,
    };
    let head_outputs: [&ExecutionTensor; 1] = match head_out.as_ref() {
        Some(pair) => [pair],
        None => [&next],
    };
    part.mark(&mut ctx, "head", &head_outputs)?;
    let vocab = shape.vocab;
    let head_source = head_out.as_ref().unwrap_or(&next);
    // Production reads the head through `read_head`: the device pair's health
    // is validated (nonfinite -> route_experts error) before the token is
    // used; a dense-only head reads the argmax. Same call here.
    let routing = if health.is_some() {
        Routing::Device
    } else {
        Routing::Host
    };
    let token = read_head(ctx.kernels, head_source, routing, vocab)?;
    // Publish the step like the real close (the marks already forced every
    // slice).
    state.commit(layers, permit)?;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_names_are_unique() {
        let mut sorted = CLASSES;
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            assert_ne!(pair[0], pair[1], "duplicate class name");
        }
    }
}
