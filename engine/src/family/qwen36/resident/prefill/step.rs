//! Chunk step assembly (packet Q36-E6; decision 0044 items 1, 4, 8, and 9 without the page pool).
//!
//! `prefill` runs `T` prompt tokens. It splits them into chunks of at most `chunk_rows` tokens
//! (each part at most the depth-aware size of `kv_plan::depth_ranges`, so the score matrix of a deep
//! chunk stays within its budget) and runs each chunk layer by layer: layer 1 over all rows, then layer 2, and so on. A chunk of one
//! row runs the same blocks with `T = 1`. Every chunk but the last runs in `Prompt` mode, and the
//! last chunk runs in the requested mode. The head reads the last row only.
//!
//! Synchronizations of one chunk with `L` layers, `D` DeltaNet layers, and `S = ceil(T / s)` parts
//! of the recurrence (`delta::gated_delta_recurrence`): the token-start import 1, then the routing
//! mode of the decoder. `Routing::Host` (the reference) reads the probabilities and imports the
//! sorted pairs of every layer (2 per layer, none for the last layer of a `Prompt` chunk);
//! `Routing::Device` (the default, packet Q36-PF-DEVROUTE) makes none in a layer and instead
//! evaluates the layer output every `EVAL_CADENCE` layers (see below) plus the one health readback
//! of the chunk (the stacked router sums). Then `S` per DeltaNet layer when `S > 1` (one evaluation
//! of the operands and `S - 1` part boundaries) and none when `S = 1`, one final `evaluate`, and
//! the readbacks of the one-token step for the last row. So `Routing::Host` makes `2 L` in
//! `Prompt` and `2 L + 3`, `2 L + 4`, `2 L + 5` in `Generate`, `Logits`, and `Record` (8, 11, 12,
//! and 13 on the four-layer test model). `Routing::Device` makes `1 + cadence + 1 + 1` in
//! `Prompt` (6 on the test model with a cadence evaluate per layer) and one more per head readback
//! (7, 8, and 9 in `Generate`, `Logits`, and `Record`). The qualification build counts them
//! exactly.
//!
//! The retire cap of the adapter is a flat 512 MiB while a chunk runs and the previous cap (the
//! decoder's `DECODE_RETIRE_CAP`) afterwards, on every path. A chunk routes as its decoder holds:
//! `Routing::Host` calls the host-routed `moe::chunk` and `Routing::Device` the device-routed
//! `moe::chunk_device`; either way the result has `routes: None`. A chunk marks the state
//! poisoned before its first native call and publishes the new state only after the final
//! `evaluate` and the readbacks succeed. A host check that fails first changes nothing.
//!
//! The activation dtype (packet Q36-BF16-b): a chunk runs the same boundary casts as the decode
//! step of `step.rs` — the concatenated embedding rows, `bb`, the new keys and values, and the MoE
//! combine convert at the block edges — and the same dtype guard runs after each norm, after each
//! layer, and on every state tensor before the final `evaluate`; on the last layer of a `Prompt`
//! chunk, where the head and the post norm are skipped, the guard runs on the layer's `mid` itself
//! (fix1 of packet Q36-BF16-b). The DeltaNet state and the convolution ring stay F32 whatever
//! the activation dtype, the key-value cache keeps its form's dtypes (packet KV8-INTEGRATE: F32
//! keys and values, or U32 codes beside activation-dtype scales and biases), and the recorded
//! rows read back as F32. An F32 decoder with the F32 cache form builds the same graph as before
//! the packet: every cast shares the tensor it would have copied.

use super::super::attn::KvBuffers;
use super::super::chunk_plan::{PREFILL_RETIRE_CAP, delta_retire_per_token, delta_subchunk};
use super::super::decoder::{ResidentDecoder, StepMode, StepResult};
use super::super::kv_plan::{SCORE_BUDGET_BYTES, depth_ranges};
use super::super::moe::Routing;
use super::super::ops::{Ctx, cast_to, dim, linear, short_import, to_act, values};
use super::super::state::{LayerState, ResidentState, chunk_growth_bytes};
use super::super::step::{check_dtype, read_token, split_rows};
use super::super::weights::RMixer;
use super::{attn, delta, moe};
use crate::family::qwen36::settings::KvDtype;
use crate::{
    EngError,
    family::flash_next::stored::{self, SlicedWeight},
    family::route::check_health,
    kernels::{BinaryOp, PackedWeight},
    sched::GrowthPermit,
    tensor::{Dtype, ExecutionTensor},
    weights::StreamedCheckpoint,
};

/// Layers of a device-routed chunk between two evaluations of the layer output (packet
/// Q36-PF-DEVROUTE). The value is 1: without the host routing synchronizations the retired bytes
/// of the chunk would grow by one layer of MoE temporaries per layer (the `repeat`, the gathered
/// products, and the un-sorted rows are about 208 MB per 512-row layer, on top of the projections
/// of the mixer), and a wider cadence would cross the flat 512 MiB `PREFILL_RETIRE_CAP` of the
/// adapter at 512 rows and force retire flushes. One layer keeps the peak at about one layer of
/// temporaries (about 1.04 GB planned workspace at 256 rows, `ChunkCost::workspace`); the cost is
/// one `evaluate` per layer, which is a stream synchronization without a readback. The Mac gates
/// measure the allocator peak (`mlx_peak_bytes` of the prompt-speed report) against the plan at
/// 256 and 512 rows; a measured peak below the plan with a wider cadence would justify raising
/// this constant.
const EVAL_CADENCE: usize = 1;

fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// The token count of a call. An empty call is a `BadRequest`.
fn row_count(tokens: &[u32]) -> Result<u32, EngError> {
    u32::try_from(tokens.len())
        .ok()
        .filter(|&rows| rows > 0)
        .ok_or_else(|| bad("a prefill needs between one and u32::MAX tokens"))
}

/// The query repeats per key-value head (`QH / KVH`, 8 on the real model): the width of the
/// score matrix that the depth-aware chunk bound plans with (packet KV8-INTEGRATE item 6).
fn repeat_count(shape: &crate::family::qwen36::host::HostShape) -> Result<u64, EngError> {
    let (heads_q, heads_kv) = (shape.attn.heads_q, shape.attn.heads_kv);
    if heads_kv == 0 || heads_q % heads_kv != 0 {
        return Err(bad(
            "the query heads are not a multiple of the key-value heads",
        ));
    }
    u64::try_from(heads_q / heads_kv).map_err(|_| bad("the query repeat count exceeds u64"))
}

/// Runs `tokens` as a prefill and publishes the state after them. See `ResidentDecoder::prefill`.
///
/// The host checks of the whole prompt come first: a poisoned state, an empty prompt, and a prompt
/// that passes the context limit return an error and change nothing. Then the chunks run in order.
/// A failed host check or a denied growth permit (`MemoryPlan`) in a later chunk comes before
/// `begin`. It leaves the state clean, with the earlier chunks published and `state.pos()` after
/// the last of them. An error after `begin` poisons the state until `reset`. A chunk routes as its
/// decoder holds (`Routing::Host` or `Routing::Device`, packet Q36-PF-DEVROUTE) and its result
/// has `routes: None`.
pub(in super::super) fn prefill(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    tokens: &[u32],
    mode: StepMode,
) -> Result<StepResult, EngError> {
    let rows = row_count(tokens)?;
    state.check_chunk(&decoder.weights, decoder.context_limit, rows)?; // the whole prompt
    // The depth-aware split (packet KV8-INTEGRATE item 6): a deterministic function of the
    // prompt length, the start position, and the chunk setting, so snapshots and prefix reuse
    // see the same boundaries on a recompute. It applies to both KV forms.
    let parts = depth_ranges(
        state.pos(),
        rows,
        decoder.chunk_rows,
        repeat_count(&decoder.weights.shape)?,
        SCORE_BUDGET_BYTES,
    )?;
    let mut result = None;
    for (index, &(start, stop)) in parts.iter().enumerate() {
        let part = usize::try_from(start)
            .ok()
            .zip(usize::try_from(stop).ok())
            .and_then(|(start, stop)| tokens.get(start..stop))
            .ok_or_else(|| bad("a prefill chunk exceeds the platform size"))?;
        let part_mode = if index + 1 == parts.len() {
            mode
        } else {
            StepMode::Prompt
        };
        result = Some(chunk_with_cap(
            decoder,
            state,
            part,
            part_mode,
            PREFILL_RETIRE_CAP,
        )?);
    }
    result.ok_or_else(|| bad("a prefill needs at least one token"))
}

/// Runs `tokens` as one chunk with the retire cap `cap`. Tests use it to set a small cap. The gated
/// delta kernel ignores the sub-chunk size that the cap implies. See `prefill` for the checks and
/// the failure rules.
pub(in super::super) fn chunk_with_cap(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    tokens: &[u32],
    mode: StepMode,
    cap: u64,
) -> Result<StepResult, EngError> {
    let rows = row_count(tokens)?;
    state.check_chunk(&decoder.weights, decoder.context_limit, rows)?;
    let growth = chunk_growth_bytes(&decoder.weights, rows)?;
    let permit = decoder.budget.acquire_growth(growth)?;
    let old = state.begin();
    match decoder.kernels.set_retire_cap(cap) {
        Ok(previous) => {
            let result = run(decoder, state, old, permit, tokens, mode, cap);
            let restored = decoder.kernels.set_retire_cap(previous);
            result.and_then(|out| restored.map(|_| out))
        }
        Err(error) => Err(error),
    }
}

/// Reads the embedding row of every token, imports the rows packed with one synchronization, and
/// dequantizes them to `[T, hidden]` in the activation dtype (packet Q36-BF16-b; the dequantized
/// rows are F32 and one cast moves them to BF16). A token outside the table returns `BadRequest`
/// from the read.
fn embed_rows(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    embed: &SlicedWeight,
    tokens: &[u32],
) -> Result<ExecutionTensor, EngError> {
    let host = tokens
        .iter()
        .map(|&token| stored::read_slice(ckpt, embed, token))
        .collect::<Result<Vec<PackedWeight>, _>>()?;
    let refs: Vec<&PackedWeight> = host.iter().collect();
    let residents = ctx.kernels.import_packed_many(&refs)?;
    if residents.len() != host.len() {
        return Err(short_import());
    }
    let rows = residents
        .iter()
        .map(|row| ctx.kernels.dequantize(row))
        .collect::<Result<Vec<_>, _>>()?;
    let parts: Vec<&ExecutionTensor> = rows.iter().collect();
    let rows = ctx.kernels.concatenate(&parts, 0)?;
    to_act(ctx, &rows)
}

/// The last row of `h` `[T, H]` as `[1, H]`.
fn last_row(
    ctx: &mut Ctx<'_>,
    h: &ExecutionTensor,
    rows: u32,
    hidden: u32,
) -> Result<ExecutionTensor, EngError> {
    let last = rows
        .checked_sub(1)
        .ok_or_else(|| bad("a chunk needs a row"))?;
    ctx.kernels.slice(h, &[last, 0], &[rows, hidden])
}

/// Builds the chunk graph, evaluates it once, reads back, and commits. `permit` moves into the
/// commit. Every earlier return drops it.
fn run(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    old: Vec<LayerState>,
    permit: GrowthPermit,
    tokens: &[u32],
    mode: StepMode,
    cap: u64,
) -> Result<StepResult, EngError> {
    let rows = row_count(tokens)?;
    let ckpt = decoder.ckpt;
    let weights = &decoder.weights;
    let shape = &weights.shape;
    let pos = state.pos();
    let hidden = dim(shape.hidden)?;
    let per_token = delta_retire_per_token(
        dim(shape.delta.value_heads)?,
        dim(shape.delta.key_dim)?,
        dim(shape.delta.value_dim)?,
    )?;
    let sub = delta_subchunk(rows, cap, per_token)?;
    let last = weights.layers.len().saturating_sub(1);
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &weights.consts,
        compile: false,
    };
    let mut h = embed_rows(&mut ctx, ckpt, &weights.embed, tokens)?;
    let mut layers = Vec::with_capacity(old.len());
    let mut records = Vec::new();
    // The device routing of packet Q36-PF-DEVROUTE collects one health node per layer and checks
    // the stacked vector once at the chunk end. The host path checks every row inside
    // `route_experts` and holds no node.
    let mut healths: Vec<ExecutionTensor> = Vec::new();
    for (index, (layer, old_layer)) in weights.layers.iter().zip(old).enumerate() {
        let normed = ctx.kernels.rms_norm(&h, &layer.input_norm, shape.eps)?;
        // Packet Q36-BF16-b: the same dtype guard as the decode step. MLX promotes mixed operands,
        // so a stray F32 operand would turn BF16 activations back to F32 without an error.
        check_dtype("input norm output", &normed, ctx.consts.act)?;
        // The old layer state moves into its arm: an attention cache is consumed by `kv_append`
        // (a growth replaces the buffer), and a DeltaNet state drops at the end of the arm, after
        // the replacement graph exists. The next synchronization frees the dropped arrays.
        let (mixed, next) = match (&layer.mixer, old_layer) {
            (RMixer::Delta(w), LayerState::Delta { rec, ring }) => {
                let s = delta::chunk(&mut ctx, shape, w, &rec, &ring, &normed, sub)?;
                (
                    s.out,
                    LayerState::Delta {
                        rec: s.rec,
                        ring: s.ring,
                    },
                )
            }
            (RMixer::Attn(w), LayerState::Attn(cache)) => {
                let s = attn::chunk(&mut ctx, shape, w, cache, pos, &normed)?;
                (s.out, LayerState::Attn(s.cache))
            }
            _ => return Err(bad("layer state kind does not match its weights")),
        };
        layers.push(next);
        let mid = ctx.kernels.binary(BinaryOp::Add, &h, &mixed)?;
        if mode == StepMode::Prompt && index == last {
            // The last layer's MoE output feeds only the head, which `Prompt` skips, so no post
            // norm runs after this `mid` and the recorded rows are gone: without this guard a
            // dropped `to_act` at a mixer out-projection would pass unnoticed on the last layer
            // of a Prompt chunk (fix1 of packet Q36-BF16-b). Every other layer, and every other
            // mode, is already covered: its `mid` meets the guarded post norm below.
            check_dtype("last layer mid", &mid, ctx.consts.act)?;
            h = mid;
            break;
        }
        let normed = ctx.kernels.rms_norm(&mid, &layer.post_norm, shape.eps)?;
        check_dtype("post norm output", &normed, ctx.consts.act)?;
        let routed = match decoder.routing {
            Routing::Host => moe::chunk(&mut ctx, shape, &layer.moe, &normed)?.0,
            Routing::Device => {
                let device = moe::chunk_device(&mut ctx, shape, &layer.moe, &normed)?;
                healths.push(device.health);
                device.out
            }
        };
        h = ctx.kernels.binary(BinaryOp::Add, &mid, &routed)?;
        check_dtype("layer output", &h, ctx.consts.act)?;
        if mode == StepMode::Record {
            // The recorded last row reads back as F32.
            let row = last_row(&mut ctx, &h, rows, hidden)?;
            records.push(cast_to(&mut ctx, &row, Dtype::F32)?);
        }
        // The evaluation cadence of the device path (see `EVAL_CADENCE`): evaluating the layer
        // output frees the layer's retired temporaries at the next flush without reading anything
        // back, so the lazy graph of the chunk stays bounded without the host routing
        // synchronizations. The last layer feeds the final evaluate, so it needs no own evaluate.
        if decoder.routing == Routing::Device
            && index != last
            && (index + 1).is_multiple_of(EVAL_CADENCE)
        {
            ctx.kernels.evaluate(&[&h])?;
        }
    }
    // The head runs on the last row only. `Prompt` skips it.
    let head = match mode {
        StepMode::Prompt => None,
        StepMode::Generate | StepMode::Logits | StepMode::Record => {
            let row = last_row(&mut ctx, &h, rows, hidden)?;
            let final_h = ctx.kernels.rms_norm(&row, &weights.final_norm, shape.eps)?;
            let logits = linear(&mut ctx, &weights.lm_head, &final_h)?;
            let next = ctx.kernels.argmax(&logits, 1, false)?;
            Some((logits, next))
        }
    };
    let record_rows = if mode == StepMode::Record {
        let parts: Vec<&ExecutionTensor> = records.iter().collect();
        Some(ctx.kernels.concatenate(&parts, 0)?)
    } else {
        None
    };
    // Packet Q36-BF16-b: the recurrent state and the convolution ring stay F32 whatever the
    // activation dtype; the key-value cache keeps its form's dtypes (packet KV8-INTEGRATE), and
    // the chunk refuses a graph that planned otherwise.
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
    // One evaluate of the new state and the head outputs. A device-routed chunk also evaluates
    // the stacked health of its router blocks, so the readback of the stack that follows the
    // evaluate is the one extra synchronization per chunk (packet Q36-PF-DEVROUTE).
    let health_stack = if healths.is_empty() {
        None
    } else {
        let parts: Vec<&ExecutionTensor> = healths.iter().collect();
        Some(ctx.kernels.concatenate(&parts, 0)?)
    };
    let mut outputs: Vec<&ExecutionTensor> = Vec::new();
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
            LayerState::Attn(cache) => outputs.extend(cache.tensors()),
        }
    }
    if let Some((logits, next)) = &head {
        outputs.push(next);
        if matches!(mode, StepMode::Logits | StepMode::Record) {
            outputs.push(logits);
        }
    }
    if let Some(rows) = &record_rows {
        outputs.push(rows);
    }
    if let Some(health) = &health_stack {
        outputs.push(health);
    }
    ctx.kernels.evaluate(&outputs)?;
    // The health of every routed layer, read back once: a non-finite value is the same failure
    // that `route_experts` reports per row under host routing, checked before any head output is
    // used. This is after `begin`, so the failure poisons the state like a native failure.
    if let Some(health) = &health_stack {
        check_health(&values(ctx.kernels, health)?)?;
    }
    let token_out = match &head {
        Some((_, next)) => Some(read_token(ctx.kernels, next, shape.vocab)?),
        None => None,
    };
    let logits_out = match (&head, mode) {
        (Some((logits, _)), StepMode::Logits | StepMode::Record) => {
            Some(values(ctx.kernels, logits)?)
        }
        _ => None,
    };
    let layer_out = match &record_rows {
        Some(rows) => Some(split_rows(values(ctx.kernels, rows)?, records.len())?),
        None => None,
    };
    state.commit_rows(rows, layers, permit)?;
    Ok(StepResult {
        token: token_out,
        logits: logits_out,
        layer_out,
        routes: None,
        probs: None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::decoder::{
        DECODE_RETIRE_CAP, MAX_CHUNK_ROWS, ResidentDecoder, StepMode, StepResult,
    };
    use super::super::super::kv_plan::KV_STEP;
    use super::super::super::ops::{assert_parity, values};
    use super::super::super::parity::{cosine, top5};
    use super::super::super::state::{LayerState, kv_storage_bytes};
    use super::*;
    use crate::family::qwen36::host::{self, HostState, LayerState as HostLayer};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::resident::moe::Routing;
    use crate::family::qwen36::settings::Activation;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;
    use crate::sched::SessionBudget;
    use crate::weights::StreamedCheckpoint;

    /// Bytes that one position adds: two attention layers, 2 key-value heads of 32, keys and values, F32.
    const GROWTH: u64 = 2 * 2 * 2 * 32 * 4;

    /// The retire cap whose sub-chunk size is 2 tokens on the tiny model: one token retires
    /// `3 * 2 * 32 * 32 * 4 = 24,576` bytes, and the room above the 64 MiB that `delta_subchunk`
    /// reserves is two tokens. The gated delta kernel ignores the size. This 64 MiB is not the decoder's cap: that is
    /// `DECODE_RETIRE_CAP` (4 GiB), and the tests below read it back after a chunk.
    const CAP_FOR_TWO: u64 = (64 << 20) + 2 * 24_576;

    /// The value of a result that must succeed. A failure is an assertion failure that shows the
    /// error, not an `unwrap` panic.
    fn ok<T: std::fmt::Debug>(result: Result<T, EngError>, what: &str) -> T {
        assert!(result.is_ok(), "{what}: {result:?}");
        result.unwrap()
    }

    /// The value of an optional result that must be present.
    fn some<T>(value: Option<T>, what: &str) -> T {
        assert!(value.is_some(), "{what}: missing");
        value.unwrap()
    }

    fn setup() -> StreamedCheckpoint {
        tiny::checkpoint(&TINY_GQA)
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

    /// Token `t` of the test sequence: 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, ... (vocabulary 8).
    fn token(t: u32) -> u32 {
        (t + 1) % 8
    }

    /// The recurrent state of each DeltaNet layer of the host. The resident layout `[VH, VD, KD]`
    /// is the host layout.
    fn host_recs(host_state: &HostState) -> Vec<Vec<f32>> {
        host_state
            .layers()
            .iter()
            .filter_map(|layer| match layer {
                HostLayer::Delta(h) => Some(h.rec().to_vec()),
                HostLayer::Attn(_) => None,
            })
            .collect()
    }

    /// Runs `count` tokens through the host step, then through the resident decoder once per mode in
    /// `modes`, and compares them. `chunk_rows` sets the chunk size of `prefill`. With `cap` the
    /// tokens run as one `chunk_with_cap` instead. Each resident state is compared after the
    /// tokens (result, recurrent states, cache length and bytes) and again after one more
    /// `Record` step, which reads every part of the state.
    fn prefill_matches_the_host(count: u32, chunk_rows: u32, cap: Option<u64>, modes: &[StepMode]) {
        let ckpt = setup();
        let (mut decoder, oracle) = decoder_and_oracle(&ckpt, 1024);
        decoder.set_chunk_rows(chunk_rows).unwrap();
        let shape = oracle.shape().clone();
        let model = oracle.view();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut host_state = HostState::new(&shape);
        let tokens: Vec<u32> = (0..count).map(token).collect();
        let mut want = None;
        for &t in &tokens {
            want = Some(host::step(&mut cpu, &shape, &model, &mut host_state, t, true).unwrap());
        }
        let want = want.unwrap();
        let recs = host_recs(&host_state);
        let more = token(count);
        let after = host::step(&mut cpu, &shape, &model, &mut host_state, more, false).unwrap();
        for &mode in modes {
            let what = format!("{count} tokens {mode:?}");
            let mut state = decoder.zero_state().unwrap();
            let got = match cap {
                None => decoder.prefill(&mut state, &tokens, mode),
                Some(cap) => chunk_with_cap(&mut decoder, &mut state, &tokens, mode, cap),
            };
            let got = ok(got, &what);
            assert_eq!(state.pos(), count, "{what}: position");
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
                        token: Some(host_argmax(&want.logits)),
                        logits: None,
                        layer_out: None,
                        routes: None,
                        probs: None
                    },
                    "{what}"
                ),
                StepMode::Logits | StepMode::Record => {
                    assert_eq!(got.token, Some(host_argmax(&want.logits)), "{what}: argmax");
                    assert!(got.routes.is_none(), "{what}: a chunk never reports routes");
                    assert_parity(
                        &some(got.logits, &format!("{what} logits")),
                        &want.logits,
                        &format!("{what} logits"),
                    );
                    if mode == StepMode::Record {
                        let rows = some(got.layer_out, &format!("{what} layer rows"));
                        assert_eq!(rows.len(), want.layer_outputs.len(), "{what}: layers");
                        for (l, (g, w)) in rows.iter().zip(&want.layer_outputs).enumerate() {
                            assert_parity(g, w, &format!("{what} layer {l}"));
                        }
                    } else {
                        assert!(got.layer_out.is_none(), "{what}: no rows");
                    }
                }
            }
            let mut compared = 0;
            for (l, layer) in state.layers().iter().enumerate() {
                match layer {
                    LayerState::Delta { rec, .. } => {
                        let got = ok(values(&mut decoder.kernels, rec), &what);
                        assert_parity(&got, &recs[compared], &format!("{what} layer {l} rec"));
                        compared += 1;
                    }
                    LayerState::Attn(cache) => {
                        assert_eq!(cache.len, count as usize, "{what} layer {l} cache length");
                    }
                }
            }
            assert_eq!(compared, 2, "{what}: both DeltaNet layers compared");
            // The logical charge follows the position; the storage is the exact hand-derived
            // capacity (review B finding 4): every count of this helper (1, 8, and 40 positions)
            // fits one default quantum of 256, so each layer stores exactly `KV_STEP` positions.
            let logical = u64::from(count) * GROWTH;
            let stored = kv_storage_bytes(state.layers());
            assert_eq!(
                stored,
                KV_STEP as u64 * GROWTH,
                "{what}: the cache storage is exactly one quantum"
            );
            assert!(
                stored >= logical,
                "{what}: the buffers hold at least the valid rows"
            );
            assert_eq!(
                decoder.budget.used(),
                u64::from(count) * GROWTH,
                "{what}: charge"
            );
            let next = decoder.step(&mut state, more, StepMode::Record);
            let next = ok(next, &format!("{what}: next step"));
            assert_eq!(next.token, Some(host_argmax(&after.logits)), "{what}: next");
            assert_parity(
                &some(next.logits, &format!("{what} next logits")),
                &after.logits,
                &format!("{what} next logits"),
            );
        }
    }

    #[test]
    fn a_chunk_matches_the_host_step_in_every_mode() {
        let _serial = crate::native_test_lock();
        let modes = [
            StepMode::Record,
            StepMode::Generate,
            StepMode::Logits,
            StepMode::Prompt,
        ];
        prefill_matches_the_host(8, 512, None, &modes);
        // One row is a chunk too.
        prefill_matches_the_host(1, 512, None, &[StepMode::Record]);
    }

    /// Forty tokens in chunks of 16, 16, and 8: the second and third chunk start at cached positions,
    /// and the DeltaNet ring wraps.
    #[test]
    fn forty_tokens_in_three_chunks_keep_parity() {
        let _serial = crate::native_test_lock();
        prefill_matches_the_host(40, 16, None, &[StepMode::Record, StepMode::Prompt]);
    }

    /// The tiny GQA checkpoint with the embedding scales of row `token` set to NaN, so that the
    /// row dequantizes to NaN and every router block of a chunk on that token is NaN (the fixture
    /// of the decode tests, copied for the chunk tests).
    fn nan_embedding_checkpoint(token: u32) -> StreamedCheckpoint {
        let mut entries = tiny::entries(&TINY_GQA);
        for (name, _, shape, bytes) in &mut entries {
            if name == "language_model.model.embed_tokens.scales" {
                let row_bytes = shape[1] as usize * 2;
                let at = token as usize * row_bytes;
                for pair in bytes[at..at + row_bytes].as_chunks_mut::<2>().0 {
                    *pair = crate::family::qwen36::fixtures::bf16(f32::NAN);
                }
            }
        }
        crate::family::qwen36::fixtures::checkpoint(entries)
    }

    /// Packet Q36-PF-DEVROUTE: a chunk whose first row dequantizes to NaN has NaN router blocks.
    /// The device path cannot check a row before it picks experts, so it fails at the chunk-end
    /// health readback with the `Backend` error of `route_experts`, after `begin`, and the state
    /// is poisoned until `reset`; the growth permit is released and the retire cap restored, as
    /// for any failed chunk. The host path fails inside `route_experts` at the first layer with
    /// the same error.
    #[test]
    fn a_nan_router_row_fails_the_chunk_at_the_health_readback() {
        let _serial = crate::native_test_lock();
        let ckpt = nan_embedding_checkpoint(7);
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        for routing in [Routing::Device, Routing::Host] {
            decoder.set_routing(routing);
            let mut state = decoder.zero_state().unwrap();
            let failed = decoder.prefill(&mut state, &[7, 1, 2], StepMode::Generate);
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
            assert!(state.is_poisoned(), "{routing:?}: a failed chunk poisons");
            assert_eq!(state.pos(), 0, "{routing:?}: position");
            assert_eq!(
                decoder.session_used(),
                0,
                "{routing:?}: the failed chunk released its permit"
            );
            let cap = decoder.kernels.set_retire_cap(1 << 20).unwrap();
            assert_eq!(
                cap, DECODE_RETIRE_CAP,
                "{routing:?}: the failed chunk restored the cap of the decoder"
            );
            decoder.kernels.set_retire_cap(DECODE_RETIRE_CAP).unwrap();
            decoder.reset(&mut state).unwrap();
            assert!(
                !state.is_poisoned() && state.pos() == 0,
                "{routing:?}: reset recovers"
            );
            let again = decoder.prefill(&mut state, &[1], StepMode::Generate);
            assert!(again.is_ok(), "{routing:?}: after reset {again:?}");
            assert_eq!(state.pos(), 1, "{routing:?}: position after reset");
        }
    }

    /// Eight tokens as one chunk under a small cap. The kernel ignores the sub-chunk size, so the
    /// result equals the whole chunk.
    #[test]
    fn sub_chunks_of_the_delta_layers_keep_parity() {
        let _serial = crate::native_test_lock();
        prefill_matches_the_host(
            8,
            512,
            Some(CAP_FOR_TWO),
            &[StepMode::Record, StepMode::Prompt],
        );
    }

    /// Packet Q36-PF-DEVROUTE T6: the same 8-token chunk under both routings. Both select experts
    /// from the same F32 probability bits with the same tie rule, so every recorded layer row and
    /// the logits agree within 1e-6 relative (stated; the only differences are the F32 summation
    /// order of the gate renormalization and the place of the gate product). The device run goes
    /// through the whole chunk step - the device MoE, the cadence evaluates, the health stack and
    /// its readback - and the host run keeps the reference path.
    #[test]
    fn the_same_chunk_under_host_and_device_routing_agrees() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        decoder.set_chunk_rows(8).unwrap();
        let tokens: Vec<u32> = (0..8).map(token).collect();
        let mut run = |routing: Routing, what: &str| -> StepResult {
            decoder.set_routing(routing);
            let mut state = decoder.zero_state().unwrap();
            ok(decoder.prefill(&mut state, &tokens, StepMode::Record), what)
        };
        let host = run(Routing::Host, "the host-routed chunk");
        let device = run(Routing::Device, "the device-routed chunk");
        assert!(
            host.routes.is_none() && device.routes.is_none(),
            "a chunk never reports routes"
        );
        assert_eq!(host.token, device.token, "the argmax of both routings");
        let relative = |got: &[f32], want: &[f32]| {
            let (mut diff, mut top) = (0f64, 0f64);
            for (&g, &w) in got.iter().zip(want) {
                diff = diff.max((f64::from(g) - f64::from(w)).abs());
                top = top.max(f64::from(w).abs());
            }
            diff / top
        };
        let host_rows = some(host.layer_out, "the host layer rows");
        let device_rows = some(device.layer_out, "the device layer rows");
        assert_eq!(device_rows.len(), host_rows.len(), "the layer count");
        for (l, (d, h)) in device_rows.iter().zip(&host_rows).enumerate() {
            assert_eq!(d.len(), h.len(), "layer {l} width");
            let error = relative(d, h);
            assert!(
                error <= 1e-6,
                "layer {l}: device against host relative error {error}"
            );
        }
        let error = relative(
            &some(device.logits, "the device logits"),
            &some(host.logits, "the host logits"),
        );
        assert!(
            error <= 1e-6,
            "logits: device against host relative error {error}"
        );
    }

    #[test]
    fn a_prompt_past_the_context_limit_is_refused_before_any_chunk_runs() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 6);
        decoder.set_chunk_rows(3).unwrap();
        let mut state = decoder.zero_state().unwrap();
        // The first two chunks of 3 would fit; the whole prompt of 8 does not.
        let tokens: Vec<u32> = (0..8).map(token).collect();
        let refused = decoder.prefill(&mut state, &tokens, StepMode::Prompt);
        assert!(
            matches!(refused, Err(EngError::ContextLimit(_))),
            "{refused:?}"
        );
        assert_eq!(state.pos(), 0, "no chunk ran");
        assert!(!state.is_poisoned(), "a refusal does not poison");
        assert_eq!(decoder.budget.used(), 0, "no charge");
        let fits = decoder.prefill(&mut state, &tokens[..6], StepMode::Prompt);
        assert!(fits.is_ok(), "exactly the limit: {fits:?}");
        assert_eq!(state.pos(), 6);
    }

    #[test]
    fn a_denied_growth_permit_returns_memory_plan_and_publishes_the_earlier_chunks() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        // Cap 5 positions: a chunk of 8 is denied at once; chunks of 4 fit once and then are denied.
        decoder.set_session_budget(SessionBudget::new(5 * GROWTH));
        let tokens: Vec<u32> = (0..8).map(token).collect();
        let mut state = decoder.zero_state().unwrap();
        let denied = decoder.prefill(&mut state, &tokens, StepMode::Prompt);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m == "session growth requires 8192 bytes, available 5120"),
            "{denied:?}"
        );
        assert_eq!(state.pos(), 0, "nothing ran");
        assert!(!state.is_poisoned(), "a denial does not poison");
        assert_eq!(decoder.session_used(), 0, "a denial charges nothing");
        decoder.set_chunk_rows(4).unwrap();
        let second = decoder.prefill(&mut state, &tokens, StepMode::Prompt);
        assert!(
            matches!(&second, Err(EngError::MemoryPlan(m)) if m == "session growth requires 4096 bytes, available 1024"),
            "{second:?}"
        );
        assert_eq!(state.pos(), 4, "the first chunk stays published");
        assert!(!state.is_poisoned(), "a denial does not poison");
        assert_eq!(
            decoder.session_used(),
            4 * GROWTH,
            "the first chunk's charge"
        );
    }

    #[test]
    fn a_failed_chunk_poisons_the_state_releases_its_permit_restores_the_cap_and_reset_recovers() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let mut state = decoder.zero_state().unwrap();
        let mut control = decoder.zero_state().unwrap();
        // Token 8 is outside the 8-row embedding. The read fails on the host after `begin` has
        // poisoned the state, so the adapter stays usable.
        let failed = decoder.prefill(&mut state, &[1, 2, 8, 3], StepMode::Generate);
        assert!(
            matches!(
                failed,
                Err(EngError::BadRequest(_) | EngError::Checkpoint(_))
            ),
            "{failed:?}"
        );
        assert!(state.is_poisoned(), "a failed chunk poisons the state");
        assert_eq!(state.pos(), 0, "a failed chunk does not advance");
        assert_eq!(
            decoder.session_used(),
            0,
            "the failed chunk released its permit"
        );
        let again = decoder.prefill(&mut state, &[1], StepMode::Generate);
        assert!(
            matches!(&again, Err(EngError::BadRequest(m)) if m.contains("reset")),
            "{again:?}"
        );
        let cap = decoder.kernels.set_retire_cap(1 << 20).unwrap();
        assert_eq!(
            cap, DECODE_RETIRE_CAP,
            "the failed chunk restored the cap of the decoder"
        );
        decoder.kernels.set_retire_cap(DECODE_RETIRE_CAP).unwrap();
        decoder.reset(&mut state).unwrap();
        assert!(!state.is_poisoned() && state.pos() == 0, "reset recovers");
        let got = decoder.prefill(&mut state, &[1, 2], StepMode::Record);
        let got = ok(got, "after reset");
        let want = decoder.prefill(&mut control, &[1, 2], StepMode::Record);
        let want = ok(want, "control");
        assert_parity(
            &some(got.logits, "logits after reset"),
            &some(want.logits, "control logits"),
            "logits after reset",
        );
    }

    #[test]
    fn an_empty_prompt_and_a_bad_chunk_size_are_bad_requests() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let mut state = decoder.zero_state().unwrap();
        let empty = decoder.prefill(&mut state, &[], StepMode::Prompt);
        assert!(matches!(empty, Err(EngError::BadRequest(_))), "{empty:?}");
        for rows in [0, MAX_CHUNK_ROWS + 1] {
            let bad = decoder.set_chunk_rows(rows);
            assert!(
                matches!(bad, Err(EngError::BadRequest(_))),
                "{rows}: {bad:?}"
            );
        }
        assert_eq!(decoder.chunk_rows, 512, "a refused size changes nothing");
        assert!(decoder.set_chunk_rows(2_048).is_ok(), "the largest size");
        assert_eq!(state.pos(), 0);
        assert!(!state.is_poisoned());
    }

    /// The cap applies during the chunk and the previous cap returns afterwards. A cap of 1 byte
    /// forces a flush at every retire, so the chunk makes forced flushes. The default cap of
    /// `prefill` makes none on the tiny model.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn the_retire_cap_applies_during_the_chunk_and_returns_afterwards() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let tokens: Vec<u32> = (0..3).map(token).collect();
        let mut state = decoder.zero_state().unwrap();
        let f0 = decoder.kernels.qualification_retire_stats().0;
        let tight = chunk_with_cap(&mut decoder, &mut state, &tokens, StepMode::Prompt, 1);
        ok(tight, "a 1-byte cap");
        let flushes = decoder.kernels.qualification_retire_stats().0 - f0;
        assert!(flushes > 0, "a 1-byte cap forces flushes: {flushes}");
        let restored = decoder.kernels.set_retire_cap(1 << 20).unwrap();
        assert_eq!(
            restored, DECODE_RETIRE_CAP,
            "the chunk restored the previous cap"
        );
        decoder.kernels.set_retire_cap(DECODE_RETIRE_CAP).unwrap();
        let mut again = decoder.zero_state().unwrap();
        let f1 = decoder.kernels.qualification_retire_stats().0;
        let loose = decoder.prefill(&mut again, &tokens, StepMode::Prompt);
        ok(loose, "the prefill cap");
        assert_eq!(
            decoder.kernels.qualification_retire_stats().0 - f1,
            0,
            "no flush at the prefill cap"
        );
        let after = decoder.kernels.set_retire_cap(1 << 20).unwrap();
        assert_eq!(
            after, DECODE_RETIRE_CAP,
            "the prefill restored the previous cap"
        );
    }

    /// Synchronizations on the tiny model: 4 layers, 2 DeltaNet layers. Device routing (the
    /// default, packet Q36-PF-DEVROUTE): one chunk of 8 rows makes the token-start import 1, one
    /// cadence evaluate per layer except the last 3, the final evaluate 1, and the health readback
    /// 1: `Prompt` 6, `Generate` 7 (plus the token), `Logits` 8 (plus the logits), `Record` 9
    /// (plus the layer rows). The gated delta kernel makes no synchronization, so `CAP_FOR_TWO`
    /// adds none: the counts stay `Prompt` 6 and `Record` 9. Eight rows in chunks of 3,
    /// 3, and 2 run `Prompt`, `Prompt`, and the requested mode: `6 + 6 + 9`. Host routing (the
    /// reference) keeps the counts before the packet: `Prompt` 8 and `Generate`/`Logits`/`Record`
    /// 11/12/13.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn each_prefill_uses_its_planned_synchronizations() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let tokens: Vec<u32> = (0..8).map(token).collect();
        let syncs = |decoder: &ResidentDecoder<'_>| decoder.kernels.qualification_sync_count();
        for (routing, budget) in [
            (Routing::Device, [6, 7, 8, 9]),
            (Routing::Host, [8, 11, 12, 13]),
        ] {
            decoder.set_routing(routing);
            for (mode, want) in [
                (StepMode::Prompt, budget[0]),
                (StepMode::Generate, budget[1]),
                (StepMode::Logits, budget[2]),
                (StepMode::Record, budget[3]),
            ] {
                let mut state = decoder.zero_state().unwrap();
                let before = syncs(&decoder);
                ok(decoder.prefill(&mut state, &tokens, mode), "one chunk");
                assert_eq!(
                    syncs(&decoder) - before,
                    want,
                    "one chunk {routing:?} {mode:?}"
                );
            }
        }
        // The loop above ends on `Routing::Host`: the sub-chunk and three-chunk cases below count
        // the device-routed chunk, the production default, so they set it again.
        decoder.set_routing(Routing::Device);
        for (mode, want) in [(StepMode::Prompt, 6), (StepMode::Record, 9)] {
            let mut state = decoder.zero_state().unwrap();
            let before = syncs(&decoder);
            let split = chunk_with_cap(&mut decoder, &mut state, &tokens, mode, CAP_FOR_TWO);
            ok(split, "sub-chunks");
            assert_eq!(syncs(&decoder) - before, want, "sub-chunks {mode:?}");
        }
        decoder.set_chunk_rows(3).unwrap();
        let mut state = decoder.zero_state().unwrap();
        let before = syncs(&decoder);
        let three = decoder.prefill(&mut state, &tokens, StepMode::Record);
        ok(three, "three chunks");
        assert_eq!(syncs(&decoder) - before, 6 + 6 + 9, "three chunks");
    }

    /// `max|got - want| / max|want|`, the F32 gate metric, reported for the BF16 run.
    fn relative_error(got: &[f32], want: &[f32]) -> f64 {
        let (mut diff, mut top) = (0f64, 0f64);
        for (&g, &w) in got.iter().zip(want) {
            let (g, w) = (f64::from(g), f64::from(w));
            diff = diff.max((g - w).abs());
            top = top.max(w.abs());
        }
        if top > 0.0 { diff / top } else { f64::INFINITY }
    }

    /// Packet Q36-BF16-b: a BF16 decoder runs the chunk path, and the chunk equals the BF16
    /// one-token steps within the mode: every recorded layer row and the logits keep a cosine of
    /// at least 0.99999, the top 5 and the argmax are equal, and the next 8 greedy tokens are
    /// equal. The max relative error is reported, not gated: BF16 rounding moves a row by more
    /// than the 1e-4 of the F32 gates without changing the mode's decisions. The committed state
    /// is F32 in every part with the position and charge of the chunk itself (fix1: asserted
    /// right after the chunk, before the one-token steps and the greedy loop move them on), a
    /// `Prompt` chunk commits the same F32 state through the last-layer mid guard, and the chunk
    /// runs the default `Routing::Device` against `Host` one-token steps (the shape of the E8 g2
    /// gate after packet Q36-PF-DEVROUTE): both paths select experts from the same F32 probability
    /// bits with the same tie rule, so no device-routing near tie can flake the comparison.
    #[test]
    fn a_bf16_chunk_matches_the_bf16_one_token_steps_within_the_mode() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let config = tiny::config(&TINY_GQA);
        let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            1024,
            Activation::Bf16,
        )
        .unwrap();
        assert_eq!(
            decoder.weights.consts.act,
            Dtype::Bf16,
            "the decoder holds BF16"
        );
        decoder.set_chunk_rows(8).unwrap();
        let tokens: Vec<u32> = (0..16).map(token).collect();
        let mut chunked = decoder.zero_state().unwrap();
        let got = ok(
            decoder.prefill(&mut chunked, &tokens, StepMode::Record),
            "the BF16 chunk",
        );
        // The state of the chunk itself, before the one-token steps and the greedy loop below
        // move the position and the charge on (fix1 of packet Q36-BF16-b).
        assert_eq!(chunked.pos(), tokens.len() as u32, "position");
        assert!(!chunked.is_poisoned(), "the chunk committed");
        assert_eq!(
            decoder.budget.used(),
            u64::from(tokens.len() as u32) * GROWTH,
            "the charge of the chunk"
        );
        for (l, layer) in chunked.layers().iter().enumerate() {
            match layer {
                LayerState::Delta { rec, ring } => {
                    assert_eq!(rec.dtype(), Dtype::F32, "layer {l} rec dtype");
                    assert_eq!(ring.dtype(), Dtype::F32, "layer {l} ring dtype");
                }
                LayerState::Attn(cache) => {
                    for tensor in cache.tensors() {
                        assert_eq!(tensor.dtype(), Dtype::F32, "layer {l} cache dtype");
                    }
                }
            }
        }
        let mut stepped = decoder.zero_state().unwrap();
        let (last, before) = tokens.split_last().unwrap();
        // The chunk above ran the default device routing; the one-token path routes on the host
        // (fix1, as the real gate): both select from the same probability bits, so only the
        // accumulation of M > 1 against M = 1 differs, and no device near tie can flake the
        // top-5 and greedy equality below.
        decoder.set_routing(Routing::Host);
        for &t in before {
            ok(
                decoder.step(&mut stepped, t, StepMode::Prompt),
                "a BF16 one-token step",
            );
        }
        let want = ok(
            decoder.step(&mut stepped, *last, StepMode::Record),
            "the last BF16 step",
        );
        let mut max_rel = 0f64;
        for (l, (g, w)) in got
            .layer_out
            .as_ref()
            .unwrap()
            .iter()
            .zip(want.layer_out.as_ref().unwrap())
            .enumerate()
        {
            let cos = cosine(g, w);
            assert!(cos >= 0.99999, "layer {l}: cosine {cos}");
            max_rel = max_rel.max(relative_error(g, w));
        }
        let (g, w) = (got.logits.as_ref().unwrap(), want.logits.as_ref().unwrap());
        let logits_cos = cosine(g, w);
        assert!(logits_cos >= 0.99999, "logits cosine {logits_cos}");
        assert_eq!(top5(g), top5(w), "the top 5 of the mode");
        assert_eq!(got.token, want.token, "the argmax of the mode");
        max_rel = max_rel.max(relative_error(g, w));
        eprintln!(
            "bf16 chunk against steps: max relative error {max_rel:.3e} (reported, not gated)"
        );
        let (mut a, mut b) = (vec![got.token.unwrap()], vec![want.token.unwrap()]);
        for _ in 1..8 {
            let next = ok(
                decoder.step(&mut chunked, *a.last().unwrap(), StepMode::Generate),
                "greedy after the chunk",
            );
            a.push(next.token.unwrap());
            let next = ok(
                decoder.step(&mut stepped, *b.last().unwrap(), StepMode::Generate),
                "greedy after the steps",
            );
            b.push(next.token.unwrap());
        }
        assert_eq!(a, b, "the next 8 greedy tokens of the mode");
        // A `Prompt` chunk runs the same guards without the head and without the last layer's
        // MoE, so the last layer's `mid` carries the guard alone (fix1): it commits an F32 state
        // at the same position and returns nothing.
        let mut prompt = decoder.zero_state().unwrap();
        let got = ok(
            decoder.prefill(&mut prompt, &tokens, StepMode::Prompt),
            "a BF16 chunk in Prompt mode",
        );
        assert_eq!(
            got,
            StepResult {
                token: None,
                logits: None,
                layer_out: None,
                routes: None,
                probs: None
            },
            "a Prompt chunk returns nothing"
        );
        assert_eq!(prompt.pos(), tokens.len() as u32, "the Prompt position");
        assert!(!prompt.is_poisoned(), "the Prompt chunk committed");
        for (l, layer) in prompt.layers().iter().enumerate() {
            match layer {
                LayerState::Delta { rec, ring } => {
                    assert_eq!(rec.dtype(), Dtype::F32, "layer {l} rec dtype");
                    assert_eq!(ring.dtype(), Dtype::F32, "layer {l} ring dtype");
                }
                LayerState::Attn(cache) => {
                    for tensor in cache.tensors() {
                        assert_eq!(tensor.dtype(), Dtype::F32, "layer {l} cache dtype");
                    }
                }
            }
        }
    }

    /// Fix1 of packet Q36-BF16-b: the state guards are pinned separately from the casts. A BF16
    /// convolution ring handed to the chunk makes the mixed concatenate produce a BF16 new ring,
    /// and the chunk refuses it before the `evaluate` with the dtype guard text; the error leaves
    /// the state poisoned, because it happens after `begin`.
    #[test]
    fn a_chunk_refuses_a_bf16_ring_through_the_state_dtype_guard() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let config = tiny::config(&TINY_GQA);
        let host = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            1024,
            Activation::Bf16,
        )
        .unwrap();
        decoder.set_chunk_rows(8).unwrap();
        let mut state = decoder.zero_state().unwrap();
        // Take the committed layers out, as the chunk does, and cast one Delta ring to BF16: the
        // graph builder then plans a BF16 new ring, which only the guard refuses.
        let mut old = state.begin();
        let at = old
            .iter()
            .position(|layer| matches!(layer, LayerState::Delta { .. }))
            .unwrap();
        if let LayerState::Delta { ring, .. } = &old[at] {
            let bad = decoder.kernels.cast(ring, Dtype::Bf16).unwrap();
            if let LayerState::Delta { ring, .. } = &mut old[at] {
                *ring = bad;
            }
        }
        assert_eq!(
            old.iter()
                .filter(|layer| matches!(layer, LayerState::Delta { ring, .. } if ring.dtype() == Dtype::Bf16))
                .count(),
            1,
            "exactly one ring is BF16"
        );
        let tokens: Vec<u32> = (0..8).map(token).collect();
        let rows = row_count(&tokens).unwrap();
        let growth = chunk_growth_bytes(&decoder.weights, rows).unwrap();
        let permit = decoder.budget.acquire_growth(growth).unwrap();
        let refused = run(
            &mut decoder,
            &mut state,
            old,
            permit,
            &tokens,
            StepMode::Generate,
            PREFILL_RETIRE_CAP,
        );
        assert!(
            matches!(&refused, Err(EngError::Backend { message, .. }) if message.contains("has dtype")),
            "the dtype guard refuses the BF16 ring: {:?}",
            refused.as_ref().err()
        );
    }

    /// The Q8 form through the decoder's chunked prefill (packet KV8-INTEGRATE), in both
    /// activation dtypes: eight tokens in chunks of 3 rows (3, 3, 2) commit without a dtype-guard
    /// refusal, leave every attention cache in the Q8 form with U32 codes and scales and biases of
    /// the ACTIVATION dtype, hold all eight positions, charge the hand-derived Q8 bytes to the
    /// session budget, and return finite logits. On the tiny KV8 model (2 key-value heads of
    /// width 64, two attention layers) one layer and position takes 288 bytes with F32 scales and
    /// 272 with BF16 scales (codes 2 * 2 * 64 = 256, plus 2 sides * 2 arrays * 2 heads * 1 group
    /// * the scale width), so the charge is 2 * 8 * 288 = 4,608 or 2 * 8 * 272 = 4,352.
    fn q8_prefill_keeps_the_form_and_the_charge(
        activation: Activation,
        scale_dtype: Dtype,
        per_layer: u64,
    ) {
        use crate::family::qwen36::settings::KvDtype;
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&tiny::TINY_KV8);
        let host =
            HostWeights::load(&ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap();
        let mut decoder = ResidentDecoder::new_with_kv(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            activation,
            KvDtype::Q8,
        )
        .unwrap();
        decoder.set_chunk_rows(3).unwrap();
        let tokens: Vec<u32> = (0..8u32).map(|t| t % 7 + 1).collect();
        let mut state = decoder.zero_state().unwrap();
        let got = ok(
            decoder.prefill(&mut state, &tokens, StepMode::Logits),
            "the Q8 chunked prefill",
        );
        assert_eq!(state.pos(), 8, "position");
        assert!(!state.is_poisoned(), "the chunks committed");
        assert_eq!(
            decoder.budget.used(),
            2 * 8 * per_layer,
            "the Q8 charge of the eight positions"
        );
        let mut caches = 0;
        for layer in state.layers() {
            if let LayerState::Attn(cache) = layer {
                caches += 1;
                assert_eq!(cache.len, 8, "valid positions");
                for (index, tensor) in cache.tensors().iter().enumerate() {
                    let want = if index % 3 == 0 {
                        Dtype::U32
                    } else {
                        scale_dtype
                    };
                    assert_eq!(tensor.dtype(), want, "part {index} of the Q8 cache");
                }
            }
        }
        assert_eq!(caches, 2, "two attention layers");
        let logits = some(got.logits, "the logits of the last chunk");
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_q8_chunked_prefill_keeps_the_q8_form_and_charge_with_f32_scales() {
        q8_prefill_keeps_the_form_and_the_charge(Activation::F32, Dtype::F32, 288);
    }

    #[test]
    fn a_q8_chunked_prefill_keeps_the_q8_form_and_charge_with_bf16_scales() {
        q8_prefill_keeps_the_form_and_the_charge(Activation::Bf16, Dtype::Bf16, 272);
    }
}
