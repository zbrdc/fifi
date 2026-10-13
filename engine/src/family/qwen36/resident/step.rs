//! Resident Q36 decode step and head (packets Q36-D and Q36-DR).
//!
//! Synchronizations per token with `L` layers (`L = 40`). With `Routing::Device` (the default) the
//! token builds one graph: the token-start import 1, one `evaluate` 1, and the readbacks. `Prompt`
//! skips the head and the last layer's MoE, so it makes 2 synchronizations. `Generate` adds one
//! readback of the token and the router health: 3. `Logits` adds the logits readback: 4. `Record`
//! adds the layer rows and the expert ids: 6. With `Routing::Host` each layer's MoE adds 2
//! synchronizations (the probability readback and the index import): `Prompt` makes `2 L`,
//! `Generate` `2 L + 3`, `Logits` `2 L + 4`, and `Record` `2 L + 5`. The real model gives 80, 83,
//! 84, and 85. Each layer's old state drops as soon as its replacement graph exists. The state
//! stays poisoned from the first native call until `commit` (decision 0039 amendment of
//! 2026-10-09).
//!
//! Router health (decision 0046 amendment A2). Device routing cannot read a router row before it
//! picks experts. Each layer returns the sum of its whole probability row, and the step adds the
//! sums. The token readback carries the total next to the token. A NaN or infinite entry in any
//! row makes the total non-finite, and the step returns `Backend` with operation `route_experts`
//! before it commits. This is the failure that `route_experts` reports under host routing. `Prompt`
//! reads nothing back, so it does not check: the NaN stays in the state, and the next step that
//! reads back a token fails.
//!
//! Growth charge: after the host checks and before `begin`, `step` acquires one `GrowthPermit` for the
//! bytes that the step adds to the attention caches. A denial returns `MemoryPlan` and changes
//! nothing. The permit moves into `run` and then into `commit`. Any earlier return drops it.

use super::attn::KvBuffers;
use super::decoder::{ResidentDecoder, StepMode, StepResult};
use super::moe::{self, Routing};
use super::ops::{Ctx, cast_to, linear, short_import, to_act, values};
use super::state::{LayerState, ResidentState, kv_growth_bytes};
use super::weights::RMixer;
use super::{attn, delta};
use crate::family::qwen36::settings::KvDtype;
use crate::{
    EngError,
    family::flash_next::stored,
    kernels::BinaryOp,
    sched::GrowthPermit,
    tensor::{Dtype, ExecutionTensor},
};

/// Runs one token. The host checks and the growth permit run first and change nothing on error: a
/// denied permit returns `MemoryPlan`. After them, any error leaves the state poisoned until `reset`
/// and releases the step's permit.
pub(super) fn step(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    token: u32,
    mode: StepMode,
) -> Result<StepResult, EngError> {
    state.check_step(&decoder.weights, decoder.context_limit)?;
    let growth = kv_growth_bytes(&decoder.weights)?;
    let permit = decoder.budget.acquire_growth(growth)?;
    let old = state.begin();
    run(decoder, state, old, permit, token, mode)
}

/// Reads the embedding row of `token`, imports it packed, and dequantizes it to `[1, hidden]` in
/// the activation dtype (packet Q36-BF16-a; the dequantized row is F32 and one cast moves it to
/// BF16). One synchronization (one `import_packed_many` call). A token outside the table returns
/// `BadRequest` from the read.
fn embed_row(
    ctx: &mut Ctx<'_>,
    ckpt: &crate::weights::StreamedCheckpoint,
    embed: &stored::SlicedWeight,
    token: u32,
) -> Result<ExecutionTensor, EngError> {
    let row = stored::read_slice(ckpt, embed, token)?;
    let resident = ctx
        .kernels
        .import_packed_many(&[&row])?
        .pop()
        .ok_or_else(short_import)?;
    let row = ctx.kernels.dequantize(&resident)?;
    to_act(ctx, &row)
}

/// Builds the token graph, evaluates it once, reads back, and commits. `permit` moves into the
/// commit. Every earlier return drops it.
fn run(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    old: Vec<LayerState>,
    permit: GrowthPermit,
    token: u32,
    mode: StepMode,
) -> Result<StepResult, EngError> {
    let ckpt = decoder.ckpt;
    let routing = decoder.routing;
    let weights = &decoder.weights;
    let shape = &weights.shape;
    let pos = state.pos();
    let last = weights.layers.len().saturating_sub(1);
    // Packet GPU-GAP-COUNT: the test-only graph probe leaves the decoder before `ctx` borrows its
    // kernels. `None` in every run that sets no probe, so the step is what it was without this.
    #[cfg(test)]
    let mut graph_probe = decoder.graph_probe.take();
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &weights.consts,
    };
    let mut h = embed_row(&mut ctx, ckpt, &weights.embed, token)?;
    let mut layers = Vec::with_capacity(old.len());
    let mut records = Vec::new();
    // Device routing: the running sum of the row sums, and in `Record` the expert ids of each layer.
    let mut health: Option<ExecutionTensor> = None;
    let mut chosen: Vec<ExecutionTensor> = Vec::new();
    // Host routing in `Record`: the expert ids of each layer.
    let mut host_routes: Vec<Vec<u32>> = Vec::new();
    // Host routing in `Record`: the router probability row that each selection read (fix3 of
    // packet Q36-BF16-a). The readback already exists, so recording it adds no synchronization.
    let mut host_probs: Vec<Vec<f32>> = Vec::new();
    for (index, (layer, old_layer)) in weights.layers.iter().zip(old).enumerate() {
        let normed = ctx.kernels.rms_norm(&h, &layer.input_norm, shape.eps)?;
        check_dtype("input norm output", &normed, ctx.consts.act)?;
        // The old layer state moves into its arm: an attention cache is consumed by `kv_append`
        // (a growth replaces the buffer), and a DeltaNet state drops at the end of the arm, after
        // the replacement graph exists. The next synchronization frees the dropped arrays.
        let (mixed, next) = match (&layer.mixer, old_layer) {
            (RMixer::Delta(w), LayerState::Delta { rec, ring }) => {
                let s = delta::step(&mut ctx, shape, w, &rec, &ring, &normed)?;
                (
                    s.out,
                    LayerState::Delta {
                        rec: s.rec,
                        ring: s.ring,
                    },
                )
            }
            (RMixer::Attn(w), LayerState::Attn(cache)) => {
                let s = attn::step(&mut ctx, shape, w, cache, pos, &normed)?;
                (s.out, LayerState::Attn(s.cache))
            }
            _ => {
                return Err(EngError::BadRequest(
                    "layer state kind does not match its weights".into(),
                ));
            }
        };
        layers.push(next);
        let mid = ctx.kernels.binary(BinaryOp::Add, &h, &mixed)?;
        if mode == StepMode::Prompt && index == last {
            // The last layer's MoE output feeds only the head, which `Prompt` skips.
            h = mid;
            break;
        }
        let normed = ctx.kernels.rms_norm(&mid, &layer.post_norm, shape.eps)?;
        check_dtype("post norm output", &normed, ctx.consts.act)?;
        let routed = match routing {
            Routing::Host => {
                let (out, (selection, probs)) = moe::step(&mut ctx, shape, &layer.moe, &normed)?;
                if mode == StepMode::Record {
                    host_routes.push(selection.iter().map(|pick| pick.0).collect());
                    host_probs.push(probs);
                }
                out
            }
            Routing::Device => {
                let (out, route) = moe::step_device(&mut ctx, shape, &layer.moe, &normed)?;
                health = Some(match health.take() {
                    Some(sum) => ctx.kernels.binary(BinaryOp::Add, &sum, &route.health)?,
                    None => route.health,
                });
                if mode == StepMode::Record {
                    chosen.push(route.ids.ids().share());
                }
                out
            }
        };
        h = ctx.kernels.binary(BinaryOp::Add, &mid, &routed)?;
        // MLX promotes mixed operands, so a stray F32 operand would turn BF16 activations back to
        // F32 without an error. The check keeps a BF16 run honest (packet Q36-BF16-a).
        check_dtype("layer output", &h, ctx.consts.act)?;
        if mode == StepMode::Record {
            // The recorded rows read back as F32.
            records.push(cast_to(&mut ctx, &h, Dtype::F32)?);
        }
    }
    // The head. `Prompt` skips the norm, `lm_head`, and the argmax.
    let head = match mode {
        StepMode::Prompt => None,
        StepMode::Generate | StepMode::Logits | StepMode::Record => {
            let final_h = ctx.kernels.rms_norm(&h, &weights.final_norm, shape.eps)?;
            let logits = linear(&mut ctx, &weights.lm_head, &final_h)?;
            let next = ctx.kernels.argmax(&logits, 1, false)?;
            // `Host`: the U32 token `[1]`. `Device`: the F32 pair `[token, router health]`.
            let out = match (routing, &health) {
                (Routing::Host, _) => next,
                (Routing::Device, Some(sum)) => {
                    let token = ctx.kernels.u32_to_f32(&next)?;
                    ctx.kernels.concatenate(&[&token, sum], 0)?
                }
                (Routing::Device, None) => {
                    return Err(EngError::Backend {
                        operation: "qwen36",
                        message: "device routing produced no router health".into(),
                    });
                }
            };
            Some((logits, out))
        }
    };
    // `Record` with device routing: the expert ids of all layers as one F32 `[L * top_k]`.
    let route_rows = if mode == StepMode::Record && routing == Routing::Device {
        let parts = chosen
            .iter()
            .map(|ids| ctx.kernels.u32_to_f32(ids))
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<&ExecutionTensor> = parts.iter().collect();
        Some(ctx.kernels.concatenate(&refs, 0)?)
    } else {
        None
    };
    let rows = if mode == StepMode::Record {
        let parts: Vec<&ExecutionTensor> = records.iter().collect();
        Some(ctx.kernels.concatenate(&parts, 0)?)
    } else {
        None
    };
    // Packet Q36-BF16-a: the recurrent state and the convolution ring stay F32 whatever the
    // activation dtype. The key-value cache keeps its form's dtypes (packet KV8-INTEGRATE): the
    // F32 control holds F32 keys and values, and the Q8 form holds U32 codes beside scales and
    // biases of the activation dtype. The step refuses a graph that planned otherwise.
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => {
                check_dtype("recurrent state", rec, Dtype::F32)?;
                check_dtype("convolution ring", ring, Dtype::F32)?;
            }
            LayerState::Attn(cache) => match cache.kv {
                KvDtype::F32 => {
                    if let Some(KvBuffers::F32 { k, v }) = &cache.buffers {
                        check_dtype("key-value cache", k, Dtype::F32)?;
                        check_dtype("key-value cache", v, Dtype::F32)?;
                    }
                }
                KvDtype::Q8 => {
                    if let Some(KvBuffers::Q8 { k, v }) = &cache.buffers {
                        for (side, tensor) in [("key codes", &k.codes), ("value codes", &v.codes)] {
                            check_dtype(side, tensor, Dtype::U32)?;
                        }
                        for tensor in [&k.scales, &k.biases, &v.scales, &v.biases] {
                            check_dtype("key-value scales", tensor, ctx.consts.act)?;
                        }
                    }
                }
            },
        }
    }
    // One evaluate of the new state and the head outputs.
    let mut outputs: Vec<&ExecutionTensor> = Vec::new();
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
            LayerState::Attn(cache) => outputs.extend(cache.tensors()),
        }
    }
    if let Some((logits, out)) = &head {
        outputs.push(out);
        if matches!(mode, StepMode::Logits | StepMode::Record) {
            outputs.push(logits);
        }
    }
    if let Some(rows) = &rows {
        outputs.push(rows);
    }
    if let Some(route_rows) = &route_rows {
        outputs.push(route_rows);
    }
    // Packet GPU-GAP-COUNT: the test-only probe runs with the lazy outputs before the one
    // `evaluate` (it describes the graph, so it adds no synchronization and no counted fault).
    // With no probe set — every build except `decode_graph_primitives` — the block is a no-op and
    // the step is byte for byte what it is without it.
    #[cfg(test)]
    if let Some(probe) = graph_probe.as_mut() {
        probe(ctx.kernels, &outputs)?;
    }
    ctx.kernels.evaluate(&outputs)?;
    let token_out = match &head {
        Some((_, out)) => Some(read_head(ctx.kernels, out, routing, shape.vocab)?),
        None => None,
    };
    let logits_out = match (&head, mode) {
        (Some((logits, _)), StepMode::Logits | StepMode::Record) => {
            Some(values(ctx.kernels, logits)?)
        }
        _ => None,
    };
    let layer_out = match &rows {
        Some(rows) => Some(split_rows(values(ctx.kernels, rows)?, records.len())?),
        None => None,
    };
    let routes_out = match &route_rows {
        Some(rows) => Some(split_routes(
            values(ctx.kernels, rows)?,
            chosen.len(),
            shape.moe.experts,
        )?),
        None if mode == StepMode::Record => Some(host_routes),
        None => None,
    };
    // `Record` with host routing keeps the probability row of every layer (fix3 of packet
    // Q36-BF16-a): the near-tie rule of the route-parity gate reads its gaps from it.
    let probs_out = match (mode, routing) {
        (StepMode::Record, Routing::Host) => Some(host_probs),
        _ => None,
    };
    state.commit(layers, permit)?;
    Ok(StepResult {
        token: token_out,
        logits: logits_out,
        layer_out,
        routes: routes_out,
        probs: probs_out,
    })
}

/// Returns `Backend` when `tensor` does not have the dtype that the step plans for it (packet
/// Q36-BF16-a). MLX promotes mixed operands silently, so the dtype is the only honest witness of
/// a stray F32 operand in a BF16 run. The check never fires in an F32 run. The prefill chunk loop
/// runs the same guard (packet Q36-BF16-b).
pub(super) fn check_dtype(
    what: &str,
    tensor: &ExecutionTensor,
    want: Dtype,
) -> Result<(), EngError> {
    if tensor.dtype() == want {
        return Ok(());
    }
    Err(EngError::Backend {
        operation: "qwen36",
        message: format!("{what} has dtype {:?}, expected {want:?}", tensor.dtype()),
    })
}

/// Reads the head output and returns the token. One synchronization.
///
/// `Host`: the U32 `[1]` argmax (`read_token`). `Device`: the F32 pair `[token, health]`. A health
/// that is not finite means that a router row held a NaN or an infinite value. It returns `Backend`
/// with operation `route_experts`, the error of `route_experts`, before the token is used.
fn read_head(
    kernels: &mut crate::kernels::MlxKernels,
    out: &ExecutionTensor,
    routing: Routing,
    vocab: usize,
) -> Result<u32, EngError> {
    if routing == Routing::Host {
        return read_token(kernels, out, vocab);
    }
    let pair = values(kernels, out)?;
    let &[token, health] = pair.as_slice() else {
        return Err(EngError::Backend {
            operation: "qwen36",
            message: "the device head did not return a token and a health value".into(),
        });
    };
    if !health.is_finite() {
        return Err(EngError::Backend {
            operation: "route_experts",
            message: "router probabilities must be finite and non-negative".into(),
        });
    }
    // The token is an integer below 2^24, so the F32 value is exact.
    let id = token as u32;
    if id as f32 != token || usize::try_from(id).is_ok_and(|id| id >= vocab) {
        return Err(EngError::Backend {
            operation: "qwen36",
            message: "argmax did not return one token inside the vocabulary".into(),
        });
    }
    Ok(id)
}

/// Splits the `[layers * top_k]` readback of expert ids into one list per layer. Every value must be
/// an integer below `experts`.
fn split_routes(flat: Vec<f32>, layers: usize, experts: usize) -> Result<Vec<Vec<u32>>, EngError> {
    let fault = || EngError::Backend {
        operation: "qwen36",
        message: "recorded expert ids do not split by layer".into(),
    };
    if layers == 0 || !flat.len().is_multiple_of(layers) {
        return Err(fault());
    }
    let width = flat.len() / layers;
    flat.chunks_exact(width)
        .map(|row| {
            row.iter()
                .map(|&value| {
                    let id = value as u32;
                    (id as f32 == value && usize::try_from(id).is_ok_and(|id| id < experts))
                        .then_some(id)
                        .ok_or_else(fault)
                })
                .collect()
        })
        .collect()
}

/// Reads the U32 `[1]` argmax back as one token inside the vocabulary. One synchronization.
pub(super) fn read_token(
    kernels: &mut crate::kernels::MlxKernels,
    next: &ExecutionTensor,
    vocab: usize,
) -> Result<u32, EngError> {
    let host = kernels.readback(next)?;
    host.host_bytes()
        .filter(|_| host.dtype() == Dtype::U32)
        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
        .map(u32::from_le_bytes)
        .filter(|&t| usize::try_from(t).is_ok_and(|t| t < vocab))
        .ok_or_else(|| EngError::Backend {
            operation: "qwen36",
            message: "argmax did not return one token inside the vocabulary".into(),
        })
}

/// Splits the `[layers, H]` readback into one `[H]` row per layer.
pub(super) fn split_rows(flat: Vec<f32>, layers: usize) -> Result<Vec<Vec<f32>>, EngError> {
    if layers == 0 || !flat.len().is_multiple_of(layers) {
        return Err(EngError::Backend {
            operation: "qwen36",
            message: "recorded rows do not split by layer".into(),
        });
    }
    let width = flat.len() / layers;
    Ok(flat.chunks_exact(width).map(<[f32]>::to_vec).collect())
}

#[cfg(test)]
mod tests {
    use super::super::decoder::{DECODE_RETIRE_CAP, ResidentDecoder, StepMode, StepResult};
    use super::super::kv_plan::KV_STEP;
    use super::super::ops::{assert_parity, f32_tensor, values};
    use super::super::state::{LayerState, ResidentState, kv_storage_bytes};
    use super::*;
    use crate::family::qwen36::fixtures;
    use crate::family::qwen36::host::{self, HostState, LayerState as HostLayer};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::settings::Activation;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;
    use crate::sched::SessionBudget;
    use crate::weights::StreamedCheckpoint;

    /// Bytes that one step adds: two attention layers, 2 key-value heads of 32, keys and values, F32.
    const GROWTH: u64 = 2 * 2 * 2 * 32 * 4;

    fn setup() -> (StreamedCheckpoint, HostWeights) {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let host = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
        (ckpt, host)
    }

    /// A second host load of the fixture, for a decoder built with another activation dtype.
    fn host_load(ckpt: &StreamedCheckpoint) -> HostWeights {
        HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap()
    }

    /// A decoder over the four-layer tiny model, and a second host load for the oracle.
    fn decoder_and_oracle(
        ckpt: &StreamedCheckpoint,
        limit: u32,
    ) -> (ResidentDecoder<'_>, HostWeights) {
        let config = tiny::config(&TINY_GQA);
        let oracle = HostWeights::load(ckpt, &TINY_GQA, &config).unwrap();
        let host = HostWeights::load(ckpt, &TINY_GQA, &config).unwrap();
        let decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), ckpt, host, limit).unwrap();
        (decoder, oracle)
    }

    /// The lowest index of the largest logit: the device `argmax` tie rule.
    fn host_argmax(logits: &[f32]) -> u32 {
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Bytes of the stored caches, summed from the tensors themselves: the capacity buffers,
    /// slack included (packet Q36-KV-INTEGRATE).
    fn stored_kv_bytes(state: &ResidentState) -> u64 {
        kv_storage_bytes(state.layers())
    }

    /// The process-wide synchronization count in qualification builds.
    #[cfg(feature = "native-fault-qualification")]
    fn syncs(kernels: &MlxKernels) -> u64 {
        kernels.qualification_sync_count()
    }

    /// Token `t` of the test sequence: 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, ... (vocabulary 8).
    fn token(t: u32) -> u32 {
        (t + 1) % 8
    }

    /// The tiny GQA checkpoint with the MoE and DeltaNet output projections scaled by 2^-10, so that
    /// the dense attention layers supply most of the residual stream. The scales and biases of
    /// every `down_proj` and every `linear_attn.out_proj` are divided by 1024, which is exact in BF16.
    /// In the unscaled model the MoE output is about 1000 and the attention output is far smaller,
    /// so a change in which positions attention reads moves a row by only 4e-6 of its maximum.
    /// Scaled, the same change moves a row by 3e-3 to 6e-3 (measured on the host reference).
    fn attention_dominant_checkpoint() -> StreamedCheckpoint {
        fixtures::checkpoint(attention_dominant_entries())
    }

    /// The tensors of `attention_dominant_checkpoint`, before they enter a checkpoint.
    fn attention_dominant_entries() -> Vec<fixtures::Entry> {
        let mut entries = tiny::entries(&TINY_GQA);
        for (name, _, _, bytes) in &mut entries {
            let scaled = name.contains("down_proj") || name.contains("linear_attn.out_proj");
            if scaled && (name.ends_with(".scales") || name.ends_with(".biases")) {
                for pair in bytes.as_chunks_mut::<2>().0 {
                    let bits = u32::from(u16::from_le_bytes(*pair)) << 16;
                    *pair = fixtures::bf16(f32::from_bits(bits) / 1024.0);
                }
            }
        }
        entries
    }

    /// The attention-dominant checkpoint with every embedding row centred, so that the router
    /// inputs differ from token to token and the chosen experts change over a run.
    ///
    /// In `TINY_GQA` every embedding value is `scale * q + bias` with `q` in 0..16 and a small bias,
    /// so all rows share a large positive mean. After the RMS norm the router inputs of different
    /// tokens are almost equal, and every token chooses the same experts in every layer (1 distinct
    /// route list over `token(0..40)`, measured on the host reference). Here each bias is
    /// `-7.5 * scale`, which is exact in BF16 for the scales of the fixture (`k * 2^-9` for
    /// integer `k` of 8 to 16). The values become `scale * (q - 7.5)`, with a mean of zero. The
    /// attention-dominant scaling keeps the MoE output from swamping the residual stream in the
    /// later layers. On the host reference this gives 8 distinct route lists over `token(0..40)`,
    /// with 7, 2, 3, and 2 distinct expert pairs in layers 0 to 3.
    fn varied_route_checkpoint() -> StreamedCheckpoint {
        let mut entries = attention_dominant_entries();
        let scales: Vec<f32> = entries
            .iter()
            .find(|entry| entry.0 == "language_model.model.embed_tokens.scales")
            .expect("the embedding scales exist")
            .3
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
            .collect();
        for (name, _, _, bytes) in &mut entries {
            if name == "language_model.model.embed_tokens.biases" {
                assert_eq!(bytes.len(), scales.len() * 2, "one bias per scale");
                for (pair, scale) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(&scales) {
                    *pair = fixtures::bf16(-7.5 * scale);
                }
            }
        }
        fixtures::checkpoint(entries)
    }

    /// Token `t` of the window sequence: 24 tokens of 3, then 16 tokens of 5. Attention that reads
    /// only the last 16 positions sees no 3 at position 39, and the full attention sees 24 of them.
    fn block_token(t: u32) -> u32 {
        if t < 24 { 3 } else { 5 }
    }

    /// Runs `tokens` on a host state and on three resident states (one per mode sequence) and
    /// compares everything after the last token, and the recorded rows and logits after every token
    /// of the `Record` sequence.
    fn run_against_the_host(ckpt: &StreamedCheckpoint, tokens: &[u32]) {
        let count = tokens.len() as u32;
        let (mut decoder, oracle) = decoder_and_oracle(ckpt, 1024);
        let shape = oracle.shape().clone();
        let model = oracle.view();
        let mut cpu = CpuProjector::new(ckpt, 1);
        let mut host_state = HostState::new(&shape);
        let mut record = decoder.zero_state().unwrap();
        let mut generate = decoder.zero_state().unwrap();
        let mut prompt = decoder.zero_state().unwrap();
        for (t, &tok) in (0..count).zip(tokens) {
            let want = host::step(&mut cpu, &shape, &model, &mut host_state, tok, true).unwrap();
            let want_token = host_argmax(&want.logits);
            let last = t + 1 == count;
            let runs = [
                (&mut record, StepMode::Record),
                (
                    &mut generate,
                    if last {
                        StepMode::Record
                    } else {
                        StepMode::Generate
                    },
                ),
                (
                    &mut prompt,
                    if last {
                        StepMode::Record
                    } else {
                        StepMode::Prompt
                    },
                ),
            ];
            for (state, mode) in runs {
                let what = format!("token {t} {mode:?}");
                let stepped = decoder.step(state, tok, mode);
                assert!(
                    stepped.is_ok(),
                    "{what}: the step failed: {:?}",
                    stepped.as_ref().err()
                );
                let got = stepped.unwrap();
                assert_eq!(state.pos(), t + 1, "{what}: position");
                assert!(!state.is_poisoned(), "{what}: poison");
                match mode {
                    StepMode::Prompt => assert_eq!(
                        got,
                        StepResult {
                            token: None,
                            logits: None,
                            layer_out: None,
                            routes: None,
                            probs: None
                        },
                        "{what}"
                    ),
                    StepMode::Generate => assert_eq!(
                        got,
                        StepResult {
                            token: Some(want_token),
                            logits: None,
                            layer_out: None,
                            routes: None,
                            probs: None
                        },
                        "{what}"
                    ),
                    StepMode::Logits => unreachable!("not used here"),
                    StepMode::Record => {
                        assert_eq!(got.token, Some(want_token), "{what}: argmax");
                        assert_parity(
                            &got.logits.unwrap(),
                            &want.logits,
                            &format!("{what} logits"),
                        );
                        let rows = got.layer_out.unwrap();
                        assert_eq!(rows.len(), want.layer_outputs.len(), "{what}: layers");
                        for (l, (g, w)) in rows.iter().zip(&want.layer_outputs).enumerate() {
                            assert_parity(g, w, &format!("{what} layer {l}"));
                        }
                        let routes = got.routes.unwrap();
                        assert_eq!(routes.len(), 4, "{what}: route layers");
                        for (l, experts) in routes.iter().enumerate() {
                            assert!(
                                experts.len() == 2 && experts[0] != experts[1],
                                "{what} layer {l}: two distinct experts, got {experts:?}"
                            );
                            assert!(
                                experts.iter().all(|&e| e < 4),
                                "{what} layer {l}: {experts:?}"
                            );
                        }
                    }
                }
            }
        }
        // The committed recurrent state of every DeltaNet layer, compared directly after the last
        // token, for the sequences that ran `Generate` and `Prompt` before it.
        for (name, state) in [
            ("record", &record),
            ("generate", &generate),
            ("prompt", &prompt),
        ] {
            let mut compared = 0;
            for (l, pair) in host_state.layers().iter().zip(state.layers()).enumerate() {
                match pair {
                    (HostLayer::Delta(h), LayerState::Delta { rec, .. }) => {
                        let got = values(&mut decoder.kernels, rec).unwrap();
                        assert_parity(&got, h.rec(), &format!("{name} layer {l} rec"));
                        compared += 1;
                    }
                    (HostLayer::Attn(_), LayerState::Attn(cache)) => {
                        assert_eq!(cache.len, count as usize, "{name} layer {l} cache length");
                    }
                    _ => panic!("{name} layer {l}: state kinds differ"),
                }
            }
            assert_eq!(compared, 2, "{name}: both DeltaNet layers compared");
            // The logical charge follows the position; the storage is the exact hand-derived
            // capacity (review B finding 4): every count of this helper (8 and 40 positions) fits
            // one default quantum of 256, so each layer stores exactly `KV_STEP` positions.
            let logical = u64::from(count) * GROWTH;
            assert_eq!(
                stored_kv_bytes(state),
                KV_STEP as u64 * GROWTH,
                "{name}: the cache storage is exactly one quantum"
            );
            assert!(
                stored_kv_bytes(state) >= logical,
                "{name}: the buffers hold at least the valid rows"
            );
        }
    }

    #[test]
    fn a_new_decoder_holds_every_scale_and_bias_as_f32() {
        // The checkpoint stores BF16 scales and biases. `ResidentDecoder::new` widens all of them.
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (decoder, _) = decoder_and_oracle(&ckpt, 16);
        let census = decoder.weights.scale_census();
        // The tiny GQA model: the head, two DeltaNet layers of 5 weights, two attention layers of
        // 4 weights, and 6 MoE weights in each of the 4 layers.
        let weights = 1 + 2 * 5 + 2 * 4 + 4 * 6;
        assert_eq!(
            (
                census.weights,
                census.f32_parts,
                census.bf16_parts,
                census.other_parts
            ),
            (weights, 2 * weights, 0, 0)
        );
    }

    #[test]
    fn eight_tokens_match_the_host_step_in_every_mode() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let tokens: Vec<u32> = (0..8).map(token).collect();
        run_against_the_host(&ckpt, &tokens);
    }

    /// A longer run of the same checks. `step.rs` has no behavior that only 40 tokens reach: the
    /// ring and the cache concatenation belong to `delta.rs` and `attn.rs`, which run 5 and 7 steps.
    /// This run checks the cache growth charge and the committed state past 16 positions. It cannot
    /// see which attention positions are read: see the next test.
    #[test]
    fn forty_tokens_keep_parity_through_ring_and_cache_growth() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let tokens: Vec<u32> = (0..40).map(token).collect();
        run_against_the_host(&ckpt, &tokens);
    }

    /// Attention must read every cached position. In the unscaled tiny model the attention output is
    /// about 4e-6 of a row maximum, which hides a window of 16 positions behind the 1e-4 bound. This
    /// run scales the other contributions down, and its inputs make the early positions differ from
    /// the late ones. Measured on the host reference, a window of 16 moves layer 1 by 5.9e-3 and layer
    /// 3 by 3.1e-3 of the row maximum, so the 1e-4 bound of `assert_parity` rejects it by 30 times or
    /// more. Row 17 (attention over the last 16 positions) fails only here.
    #[test]
    fn forty_tokens_see_every_attention_position() {
        let _serial = crate::native_test_lock();
        let ckpt = attention_dominant_checkpoint();
        let tokens: Vec<u32> = (0..40).map(block_token).collect();
        run_against_the_host(&ckpt, &tokens);
    }

    #[test]
    fn logits_mode_returns_the_token_and_the_logits_without_layer_rows() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, oracle) = decoder_and_oracle(&ckpt, 1024);
        let shape = oracle.shape().clone();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut host_state = HostState::new(&shape);
        let mut state = decoder.zero_state().unwrap();
        let want = host::step(&mut cpu, &shape, &oracle.view(), &mut host_state, 3, false).unwrap();
        let got = decoder.step(&mut state, 3, StepMode::Logits).unwrap();
        assert_eq!(got.token, Some(host_argmax(&want.logits)));
        assert_parity(&got.logits.unwrap(), &want.logits, "logits");
        assert!(got.layer_out.is_none());
        assert!(got.routes.is_none());
    }

    #[test]
    fn each_committed_step_charges_the_logical_cache_growth() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        decoder.set_session_budget(SessionBudget::new(10 * GROWTH));
        let mut state = decoder.zero_state().unwrap();
        assert_eq!(decoder.session_used(), 0, "a zero state is not charged");
        for t in 1..=3u64 {
            let result = decoder.step(&mut state, token(t as u32), StepMode::Prompt);
            assert!(result.is_ok(), "step {t} must commit: {result:?}");
            assert_eq!(decoder.session_used(), t * GROWTH, "charge after step {t}");
            // The charge is logical; the storage is the exact hand-derived capacity (review B
            // finding 4): one to three positions fit the first default quantum of 256, so each
            // layer stores exactly `KV_STEP` positions.
            let stored = stored_kv_bytes(&state);
            assert_eq!(
                stored,
                KV_STEP as u64 * GROWTH,
                "step {t}: the cache storage is exactly one quantum"
            );
            assert!(
                stored >= t * GROWTH,
                "step {t}: the buffers hold at least the valid rows"
            );
        }
    }

    #[test]
    fn a_denied_growth_permit_returns_memory_plan_and_changes_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        // Cap 1,500: the first step charges 1,024 and fits; the second would reach 2,048.
        decoder.set_session_budget(SessionBudget::new(1_500));
        let mut state = decoder.zero_state().unwrap();
        decoder.step(&mut state, 1, StepMode::Prompt).unwrap();
        let bytes = state.byte_len().unwrap();
        let denied = decoder.step(&mut state, 2, StepMode::Generate);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m == "session growth requires 1024 bytes, available 476"),
            "{denied:?}"
        );
        assert_eq!(state.pos(), 1, "position after the denial");
        assert!(!state.is_poisoned(), "a denial does not poison");
        assert_eq!(
            state.byte_len().unwrap(),
            bytes,
            "the layer states stay in place"
        );
        assert_eq!(decoder.session_used(), GROWTH, "a denial charges nothing");
        decoder.reset(&mut state).unwrap();
        assert_eq!(decoder.session_used(), 0, "reset releases the permits");
    }

    #[test]
    fn a_failed_step_poisons_the_state_releases_its_permit_and_reset_recovers() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let mut state = decoder.zero_state().unwrap();
        let mut control = decoder.zero_state().unwrap();
        decoder.step(&mut state, 1, StepMode::Prompt).unwrap();
        // Token 8 is outside the 8-row embedding. The read fails on the host after `begin` has
        // poisoned the state, so the adapter stays usable.
        let failed = decoder.step(&mut state, 8, StepMode::Generate);
        assert!(
            matches!(
                failed,
                Err(EngError::BadRequest(_) | EngError::Checkpoint(_))
            ),
            "{failed:?}"
        );
        assert!(state.is_poisoned(), "a failed step poisons the state");
        assert_eq!(state.pos(), 1, "a failed step does not advance");
        assert_eq!(
            decoder.session_used(),
            GROWTH,
            "the failed step released its permit"
        );
        let again = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(
            matches!(&again, Err(EngError::BadRequest(m)) if m.contains("reset")),
            "{again:?}"
        );
        decoder.reset(&mut state).unwrap();
        assert!(
            !state.is_poisoned() && state.pos() == 0,
            "reset clears poison and position"
        );
        let got = decoder.step(&mut state, 1, StepMode::Record).unwrap();
        let want = decoder.step(&mut control, 1, StepMode::Record).unwrap();
        assert_parity(
            &got.logits.unwrap(),
            &want.logits.unwrap(),
            "logits after reset",
        );
    }

    #[test]
    fn the_context_limit_refuses_the_step_at_the_limit_without_poisoning() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 3);
        let mut state = decoder.zero_state().unwrap();
        for t in 0..3 {
            decoder
                .step(&mut state, token(t), StepMode::Prompt)
                .unwrap();
        }
        let refused = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(
            matches!(refused, Err(EngError::ContextLimit(_))),
            "{refused:?}"
        );
        assert_eq!(state.pos(), 3);
        assert!(!state.is_poisoned());
        assert_eq!(
            decoder.session_used(),
            3 * GROWTH,
            "no charge for the refused step"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn each_mode_uses_its_synchronization_budget() {
        // L = 4. Device routing: Prompt = 1 token start + 1 evaluate = 2. Generate adds the token
        // and health readback: 3. Logits adds the logits readback: 4. Record adds the layer rows
        // and the expert ids: 6.
        // Host routing: Prompt = 1 token start + 2 per layer except the last layer's MoE + 1
        // evaluate = 8. Generate: 1 + 2 * 4 + 1 + 1 token readback = 11. Logits: 12. Record: 13.
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let modes = [
            StepMode::Prompt,
            StepMode::Generate,
            StepMode::Logits,
            StepMode::Record,
        ];
        for (routing, budget) in [
            (Routing::Device, [2, 3, 4, 6]),
            (Routing::Host, [8, 11, 12, 13]),
        ] {
            decoder.set_routing(routing);
            for (mode, want) in modes.into_iter().zip(budget) {
                let mut state = decoder.zero_state().unwrap();
                let before = syncs(&decoder.kernels);
                decoder.step(&mut state, 1, mode).unwrap();
                assert_eq!(
                    syncs(&decoder.kernels) - before,
                    want,
                    "{routing:?} {mode:?}"
                );
            }
        }
    }

    /// The tiny GQA checkpoint with the embedding scales of row `token` set to NaN, so that the row
    /// dequantizes to NaN and every router row of a step on that token is NaN.
    fn nan_embedding_checkpoint(token: u32) -> StreamedCheckpoint {
        let mut entries = tiny::entries(&TINY_GQA);
        for (name, _, shape, bytes) in &mut entries {
            if name == "language_model.model.embed_tokens.scales" {
                let row_bytes = shape[1] as usize * 2;
                let at = token as usize * row_bytes;
                for pair in bytes[at..at + row_bytes].as_chunks_mut::<2>().0 {
                    *pair = fixtures::bf16(f32::NAN);
                }
            }
        }
        fixtures::checkpoint(entries)
    }

    #[test]
    fn device_and_host_routing_choose_the_same_experts_and_tokens() {
        let _serial = crate::native_test_lock();
        let ckpt = varied_route_checkpoint();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let mut host_state = decoder.zero_state().unwrap();
        let mut device_state = decoder.zero_state().unwrap();
        let mut distinct = std::collections::BTreeSet::new();
        for t in 0..40u32 {
            decoder.set_routing(Routing::Host);
            let host = decoder
                .step(&mut host_state, token(t), StepMode::Record)
                .unwrap();
            decoder.set_routing(Routing::Device);
            let device = decoder
                .step(&mut device_state, token(t), StepMode::Record)
                .unwrap();
            assert_eq!(device.routes, host.routes, "token {t}: experts per layer");
            assert_eq!(device.token, host.token, "token {t}: argmax");
            assert_parity(
                &device.logits.unwrap(),
                &host.logits.unwrap(),
                &format!("token {t} logits"),
            );
            distinct.insert(device.routes.unwrap());
        }
        assert!(
            distinct.len() >= 3,
            "the routes vary over the run: {} distinct",
            distinct.len()
        );
    }

    #[test]
    fn a_nan_router_row_fails_the_step_at_the_readback_and_a_reset_recovers() {
        let _serial = crate::native_test_lock();
        let ckpt = nan_embedding_checkpoint(7);
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        for routing in [Routing::Device, Routing::Host] {
            decoder.set_routing(routing);
            let mut state = decoder.zero_state().unwrap();
            let failed = decoder.step(&mut state, 7, StepMode::Generate);
            assert!(
                matches!(
                    &failed,
                    Err(EngError::Backend {
                        operation: "route_experts",
                        ..
                    })
                ),
                "{routing:?}: {failed:?}"
            );
            assert!(state.is_poisoned(), "{routing:?}: a failed step poisons");
            assert_eq!(state.pos(), 0, "{routing:?}: position");
            decoder.reset(&mut state).unwrap();
            let again = decoder.step(&mut state, 1, StepMode::Generate);
            assert!(again.is_ok(), "{routing:?}: after reset {again:?}");
            assert_eq!(state.pos(), 1, "{routing:?}: position after reset");
        }
    }

    #[test]
    fn a_nan_prompt_token_is_caught_by_the_next_step_that_reads_back() {
        // `Prompt` reads nothing back, so device routing cannot check it. The NaN stays in the
        // state and reaches every later router row.
        let _serial = crate::native_test_lock();
        let ckpt = nan_embedding_checkpoint(7);
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let mut state = decoder.zero_state().unwrap();
        let prompt = decoder.step(&mut state, 7, StepMode::Prompt);
        assert!(prompt.is_ok(), "Prompt reads nothing back: {prompt:?}");
        assert_eq!(state.pos(), 1);
        let next = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(
            matches!(
                &next,
                Err(EngError::Backend {
                    operation: "route_experts",
                    ..
                })
            ),
            "{next:?}"
        );
    }

    /// An F32 tensor `[2]` for `read_head`.
    fn head_pair(decoder: &mut ResidentDecoder<'_>, pair: [f32; 2]) -> ExecutionTensor {
        let host = f32_tensor(vec![2], &pair).unwrap();
        decoder.kernels.import_tensor(&host).unwrap()
    }

    #[test]
    fn read_head_returns_the_token_and_rejects_bad_pairs() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let mut read = |pair: [f32; 2]| {
            let out = head_pair(&mut decoder, pair);
            read_head(&mut decoder.kernels, &out, Routing::Device, 8)
        };
        assert_eq!(read([5.0, 4.0]).unwrap(), 5);
        let fault = |result: Result<u32, EngError>| match result {
            Err(EngError::Backend { operation, .. }) => operation,
            other => panic!("expected Backend, got {other:?}"),
        };
        assert_eq!(fault(read([5.0, f32::NAN])), "route_experts", "NaN health");
        assert_eq!(
            fault(read([5.0, f32::INFINITY])),
            "route_experts",
            "infinite health"
        );
        assert_eq!(fault(read([5.5, 4.0])), "qwen36", "a fraction is no token");
        assert_eq!(fault(read([-1.0, 4.0])), "qwen36", "a negative is no token");
        assert_eq!(fault(read([f32::NAN, 4.0])), "qwen36", "a NaN is no token");
        assert_eq!(
            fault(read([8.0, 4.0])),
            "qwen36",
            "token 8 is outside the vocabulary of 8"
        );
    }

    #[test]
    fn split_routes_checks_every_expert_id() {
        let ok = split_routes(vec![1.0, 3.0, 0.0, 2.0], 2, 4).unwrap();
        assert_eq!(ok, [vec![1, 3], vec![0, 2]]);
        for (flat, layers, what) in [
            (vec![1.0, 4.0], 1, "an id equal to the expert count"),
            (vec![1.0, 1.5], 1, "a fraction"),
            (vec![1.0, -1.0], 1, "a negative id"),
            (vec![1.0, 2.0, 3.0], 2, "a length that does not split"),
            (vec![1.0, 2.0], 0, "no layers"),
        ] {
            let result = split_routes(flat, layers, 4);
            assert!(result.is_err(), "{what}: {result:?}");
        }
    }

    #[test]
    fn a_new_decoder_raises_the_retire_cap_for_the_whole_token_graph() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let previous = decoder.kernels.set_retire_cap(1 << 20).unwrap();
        assert_eq!(previous, DECODE_RETIRE_CAP);
        assert_eq!(DECODE_RETIRE_CAP, 4 << 30, "4 GiB");
    }
    /// A BF16 decoder (packet Q36-BF16-a) steps the tiny model finitely and keeps the F32 argmax on
    /// every step whose F32 top-two logit gap exceeds 0.25, the BF16 rounding scale, so the two
    /// decoders read the same tokens only where the fixture decides clearly. The test also holds
    /// the dtype itself to account: `Consts::act` is `Bf16` and a converted layer norm reads BF16,
    /// so a decoder that silently stayed F32 fails here and not only on the finite check. The
    /// dtype guard inside `run` makes any stray F32 operand fail the step with `Backend`, so this
    /// test also proves the whole BF16 graph: embedding cast, converted norms, quantized matmuls,
    /// F32 cache and DeltaNet state.
    #[test]
    fn a_bf16_decoder_steps_finitely_and_keeps_the_f32_argmax_where_it_holds() {
        let _serial = crate::native_test_lock();
        let (ckpt, _) = setup();
        let mut f32 = ResidentDecoder::new(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host_load(&ckpt),
            1024,
        )
        .unwrap();
        let mut bf16 = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host_load(&ckpt),
            1024,
            Activation::Bf16,
        )
        .unwrap();
        // The hard dtype assertions: the decoder really runs BF16, not F32 with a finite readback.
        assert_eq!(
            bf16.weights.consts.act,
            Dtype::Bf16,
            "the decoder holds the BF16 activation dtype"
        );
        assert_eq!(
            bf16.weights.layers[0].input_norm.dtype(),
            Dtype::Bf16,
            "a converted layer norm reads BF16"
        );
        let mut state_f32 = f32.zero_state().unwrap();
        let mut state_bf16 = bf16.zero_state().unwrap();
        let mut token = token(0);
        let mut checked = 0u32;
        for step in 0..8u32 {
            let want = f32.step(&mut state_f32, token, StepMode::Logits).unwrap();
            let got = bf16.step(&mut state_bf16, token, StepMode::Logits).unwrap();
            let row = got.logits.as_ref().unwrap();
            assert!(
                row.iter().all(|value| value.is_finite()),
                "step {step}: every BF16 logit is finite"
            );
            assert_eq!(row.len(), 8, "step {step}: the vocabulary width");
            // The F32 top-two gap decides whether the argmax is a clear choice: only a gap above
            // the BF16 rounding scale of 0.25 binds.
            let mut ranked = want.logits.as_ref().unwrap().clone();
            ranked.sort_by(|a, b| b.total_cmp(a));
            let gap = ranked[0] - ranked[1];
            if gap > 0.25 {
                assert_eq!(
                    got.token, want.token,
                    "step {step}: BF16 keeps the F32 argmax (F32 gap {gap})"
                );
                checked += 1;
            }
            token = want.token.unwrap();
        }
        if checked == 0 {
            // The fixture has no position whose F32 top-two gap exceeds the BF16 rounding scale,
            // so only the finite-logit and dtype assertions bind on it (packet Q36-BF16-a fix).
            eprintln!(
                "no step of the tiny fixture has an F32 top-two gap above 0.25; \
                 the BF16 argmax comparison did not run"
            );
        }
    }

    /// The frozen expectation of one half of the bit-equality run (packet GAP-TRIM-A 4(b)): the
    /// tokens and the logit bits of eight `Record` steps of the tiny GQA model in one activation
    /// dtype. The literals come from a Mac run of the test below at the BASE commit of the trims
    /// (2253cbcc: one clamp per gather, a contiguous node after every product), taken twice with
    /// identical output. After an intended numeric change, run the test with `GAP_TRIM_FREEZE=1`
    /// (`--ignored` is not needed) and paste the printed block over each constant. A `u32::MAX`
    /// token marks an unfrozen half, and the test refuses it with these instructions.
    struct Frozen {
        tokens: [u32; 8],
        logits: [[f32; 8]; 8],
    }

    /// F32 expectation, frozen from a run at the base commit 2253cbcc (Mac, `GAP_TRIM_FREEZE=1`).
    const FROZEN_F32: Frozen = Frozen {
        tokens: [5, 5, 5, 5, 5, 5, 5, 5],
        logits: [
            [
                9.919306, 5.498854, 9.913164, 5.079399, 10.585416, 11.879209, 7.241288, 10.526949,
            ],
            [
                9.920084, 5.498866, 9.913519, 5.0806437, 10.586811, 11.880026, 7.2418394, 10.527744,
            ],
            [
                9.918437, 5.4970307, 9.916389, 5.0787616, 10.589268, 11.881268, 7.2431817,
                10.527388,
            ],
            [
                9.918474, 5.497011, 9.914625, 5.0805902, 10.588559, 11.878557, 7.2429037, 10.525873,
            ],
            [
                9.920477, 5.4977455, 9.913423, 5.081953, 10.58787, 11.880568, 7.241171, 10.526303,
            ],
            [
                9.920667, 5.4968534, 9.912812, 5.0828376, 10.586788, 11.879564, 7.2410135,
                10.526457,
            ],
            [
                9.920649, 5.4976397, 9.913076, 5.0823975, 10.588538, 11.880221, 7.2412996,
                10.527037,
            ],
            [
                9.92029, 5.4959135, 9.913987, 5.081272, 10.589256, 11.881017, 7.2413163, 10.525632,
            ],
        ],
    };

    /// BF16 expectation, frozen from a run at the base commit 2253cbcc (Mac, `GAP_TRIM_FREEZE=1`).
    const FROZEN_BF16: Frozen = Frozen {
        tokens: [5, 5, 5, 5, 5, 5, 5, 5],
        logits: [
            [
                9.919812, 5.4982963, 9.912594, 5.0796347, 10.587905, 11.876688, 7.2452645,
                10.524265,
            ],
            [
                9.919178, 5.501468, 9.915744, 5.078258, 10.586157, 11.878135, 7.241685, 10.5295105,
            ],
            [
                9.917215, 5.4937, 9.918845, 5.076762, 10.590743, 11.87645, 7.2530966, 10.5297575,
            ],
            [
                9.920153, 5.497995, 9.9168625, 5.0813932, 10.590816, 11.875647, 7.24703, 10.526147,
            ],
            [
                9.92337, 5.4974136, 9.913061, 5.0848355, 10.587328, 11.87897, 7.244747, 10.527159,
            ],
            [
                9.92108, 5.497634, 9.917196, 5.081452, 10.585882, 11.877356, 7.2483125, 10.528478,
            ],
            [
                9.921514, 5.5030937, 9.918383, 5.0804787, 10.58349, 11.877964, 7.248267, 10.531643,
            ],
            [
                9.921381, 5.500646, 9.918471, 5.0784984, 10.588705, 11.87885, 7.248251, 10.527467,
            ],
        ],
    };

    /// Renders one measured half as a paste-ready block for `Frozen`.
    fn frozen_block(name: &str, tokens: &[u32], logits: &[Vec<f32>]) -> String {
        let mut text = format!("    const {name}: Frozen = Frozen {{\n        tokens: [");
        for token in tokens {
            text.push_str(&format!("{token}, "));
        }
        text.push_str("],\n        logits: [\n");
        for row in logits {
            text.push_str("            [");
            for value in row {
                text.push_str(&format!("{value:?}, "));
            }
            text.push_str("],\n");
        }
        text.push_str("        ],\n    };\n");
        text
    }

    /// Runs eight `Record` steps of one activation dtype and returns the tokens and the logit
    /// rows (packet GAP-TRIM-A 4(b)). The step input is the frozen `token` sequence, so the run
    /// is deterministic in both dtypes.
    fn eight_record_steps(activation: Activation) -> (Vec<u32>, Vec<Vec<f32>>) {
        let (ckpt, _) = setup();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host_load(&ckpt),
            1024,
            activation,
        )
        .unwrap();
        let mut state = decoder.zero_state().unwrap();
        let mut tokens = Vec::new();
        let mut logits = Vec::new();
        for step in 0..8u32 {
            let got = decoder
                .step(&mut state, token(step), StepMode::Record)
                .unwrap();
            tokens.push(got.token.unwrap());
            logits.push(got.logits.unwrap());
        }
        (tokens, logits)
    }

    /// Packet GAP-TRIM-A 4(b): the end-to-end bit equality of a short decode. Eight `Record`
    /// steps of the tiny GQA model must return the SAME tokens and the SAME logit bits on the
    /// trimmed graph (one clamp per layer, no no-op contiguous nodes) as the frozen expectation
    /// at the base commit, in F32 and in BF16. The comparison is on `f32::to_bits`.
    #[test]
    fn eight_decode_steps_match_the_frozen_expectation_in_f32_and_bf16() {
        let freeze = std::env::var_os("GAP_TRIM_FREEZE").is_some();
        for (name, activation, frozen) in [
            ("FROZEN_F32", Activation::F32, &FROZEN_F32),
            ("FROZEN_BF16", Activation::Bf16, &FROZEN_BF16),
        ] {
            let (tokens, logits) = eight_record_steps(activation);
            if freeze {
                eprintln!(
                    "\npaste over the {name} constant:\n{}",
                    frozen_block(name, &tokens, &logits)
                );
                continue;
            }
            assert_ne!(
                frozen.tokens,
                [u32::MAX; 8],
                "{name} is unfrozen: run this test at the base commit with GAP_TRIM_FREEZE=1, \
                 paste the printed block, and remove the ignore"
            );
            assert_eq!(tokens.as_slice(), frozen.tokens, "{name}: the eight tokens");
            for (step, (got, want)) in logits.iter().zip(&frozen.logits).enumerate() {
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{name} step {step}: the vocabulary width"
                );
                for (index, (got, want)) in got.iter().zip(want).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{name} step {step} logit {index}: {got} against the frozen {want}"
                    );
                }
            }
        }
    }
}
