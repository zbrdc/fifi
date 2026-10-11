//! Chunk step assembly (packet Q36-E6; decision 0044 items 1, 4, 8, and 9 without the page pool).
//!
//! `prefill` runs `T` prompt tokens. It splits them into chunks of at most `chunk_rows` tokens and
//! runs each chunk layer by layer: layer 1 over all rows, then layer 2, and so on. A chunk of one
//! row runs the same blocks with `T = 1`. Every chunk but the last runs in `Prompt` mode, and the
//! last chunk runs in the requested mode. The head reads the last row only.
//!
//! Synchronizations of one chunk with `L` layers, `D` DeltaNet layers, and `S = ceil(T / s)` parts
//! of the recurrence (`delta::gated_delta_recurrence`): the token-start import 1, host routing 2
//! per layer (the probability readback and the index import), `S` per DeltaNet layer when `S > 1`
//! (one evaluation of the operands and `S - 1` part boundaries) and none when `S = 1`, one final
//! `evaluate`, and the readbacks of the one-token step for the last row. `Prompt` skips the last
//! layer's MoE, so it makes `2 L`, plus `D * S` when `S > 1`. `Generate`, `Logits`, and `Record`
//! make 3, 4, and 5 more than the `2 L`, as in `step.rs`. The qualification build counts them
//! exactly.
//!
//! The retire cap of the adapter is a flat 512 MiB while a chunk runs and the previous cap (the
//! decoder's `DECODE_RETIRE_CAP`) afterwards, on every path. A chunk routes on the host
//! (`moe::chunk`) whatever `Routing` the decoder holds, and its result has `routes: None`.
//! A chunk marks the state poisoned before its first native call and
//! publishes the new state only after the final `evaluate` and the readbacks succeed. A host check
//! that fails first changes nothing.

use super::super::chunk_plan::{
    PREFILL_RETIRE_CAP, delta_retire_per_token, delta_subchunk, ranges,
};
use super::super::decoder::{ResidentDecoder, StepMode, StepResult};
use super::super::ops::{Ctx, dim, linear, short_import, values};
use super::super::state::{LayerState, ResidentState, chunk_growth_bytes};
use super::super::step::{read_token, split_rows};
use super::super::weights::RMixer;
use super::{attn, delta, moe};
use crate::{
    EngError,
    family::flash_next::stored::{self, SlicedWeight},
    kernels::{BinaryOp, PackedWeight},
    sched::GrowthPermit,
    tensor::{Dtype, ExecutionTensor},
    weights::StreamedCheckpoint,
};

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

/// Runs `tokens` as a prefill and publishes the state after them. See `ResidentDecoder::prefill`.
///
/// The host checks of the whole prompt come first: a poisoned state, an empty prompt, and a prompt
/// that passes the context limit return an error and change nothing. Then the chunks run in order.
/// A decoder whose activation dtype is not F32 (packet Q36-BF16-a) returns `BadRequest` first,
/// before every other check, permit, and native call, and changes nothing: the chunk path is not
/// dtype aware until slice Q36-BF16-b. A failed host check or a denied growth permit
/// (`MemoryPlan`) in a later chunk comes before `begin`. It leaves the state clean, with the
/// earlier chunks published and `state.pos()` after the last of them. An error after `begin`
/// poisons the state until `reset`. A chunk routes on the host whatever `Routing` the decoder
/// holds, and its result has `routes: None`.
pub(in super::super) fn prefill(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    tokens: &[u32],
    mode: StepMode,
) -> Result<StepResult, EngError> {
    if decoder.weights.consts.act != Dtype::F32 {
        return Err(bad("the chunked prefill runs F32 activations only"));
    }
    let rows = row_count(tokens)?;
    state.check_chunk(&decoder.weights, decoder.context_limit, rows)?; // the whole prompt
    let parts = ranges(rows, decoder.chunk_rows)?;
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
/// dequantizes them to `[T, hidden]`. A token outside the table returns `BadRequest` from the read.
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
    ctx.kernels.concatenate(&parts, 0)
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
    };
    let mut h = embed_rows(&mut ctx, ckpt, &weights.embed, tokens)?;
    let mut layers = Vec::with_capacity(old.len());
    let mut records = Vec::new();
    for (index, (layer, old_layer)) in weights.layers.iter().zip(old).enumerate() {
        let normed = ctx.kernels.rms_norm(&h, &layer.input_norm, shape.eps)?;
        // The old layer state moves into its arm and drops at the end of the arm, after the
        // replacement graph exists. The next synchronization frees it.
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
                let s = attn::chunk(&mut ctx, shape, w, &cache, pos, &normed)?;
                (s.out, LayerState::Attn(s.cache))
            }
            _ => return Err(bad("layer state kind does not match its weights")),
        };
        layers.push(next);
        let mid = ctx.kernels.binary(BinaryOp::Add, &h, &mixed)?;
        if mode == StepMode::Prompt && index == last {
            // The last layer's MoE output feeds only the head, which `Prompt` skips.
            h = mid;
            break;
        }
        let normed = ctx.kernels.rms_norm(&mid, &layer.post_norm, shape.eps)?;
        let (routed, _routes) = moe::chunk(&mut ctx, shape, &layer.moe, &normed)?;
        h = ctx.kernels.binary(BinaryOp::Add, &mid, &routed)?;
        if mode == StepMode::Record {
            records.push(last_row(&mut ctx, &h, rows, hidden)?);
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
    // One evaluate of the new state and the head outputs.
    let mut outputs: Vec<&ExecutionTensor> = Vec::new();
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
            LayerState::Attn(cache) => outputs.extend([&cache.k, &cache.v].into_iter().flatten()),
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
    ctx.kernels.evaluate(&outputs)?;
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
    use super::super::super::ops::{assert_parity, values};
    use super::super::super::state::LayerState;
    use super::*;
    use crate::family::qwen36::host::{self, HostState, LayerState as HostLayer};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
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
                    assert!(got.routes.is_none(), "{what}: a chunk routes on the host");
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
            assert_eq!(
                state
                    .layers()
                    .iter()
                    .map(|layer| match layer {
                        LayerState::Attn(cache) => cache.byte_len(),
                        LayerState::Delta { .. } => 0,
                    })
                    .sum::<u64>(),
                u64::from(count) * GROWTH,
                "{what}: cache bytes"
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

    /// Synchronizations on the tiny model: 4 layers, 2 DeltaNet layers. One chunk of 8 rows makes
    /// `Prompt` 8, `Generate` 11, `Logits` 12, `Record` 13, as `step.rs`. The gated delta kernel
    /// makes no synchronization, so `CAP_FOR_TWO` adds none: the counts stay `Prompt` 8 and
    /// `Record` 13. Eight rows in chunks of 3, 3, and 2 run `Prompt`, `Prompt`, and the requested
    /// mode: `8 + 8 + 13`.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn each_prefill_uses_its_planned_synchronizations() {
        let _serial = crate::native_test_lock();
        let ckpt = setup();
        let (mut decoder, _) = decoder_and_oracle(&ckpt, 1024);
        let tokens: Vec<u32> = (0..8).map(token).collect();
        let syncs = |decoder: &ResidentDecoder<'_>| decoder.kernels.qualification_sync_count();
        for (mode, want) in [
            (StepMode::Prompt, 8),
            (StepMode::Generate, 11),
            (StepMode::Logits, 12),
            (StepMode::Record, 13),
        ] {
            let mut state = decoder.zero_state().unwrap();
            let before = syncs(&decoder);
            ok(decoder.prefill(&mut state, &tokens, mode), "one chunk");
            assert_eq!(syncs(&decoder) - before, want, "one chunk {mode:?}");
        }
        for (mode, want) in [(StepMode::Prompt, 8), (StepMode::Record, 13)] {
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
        assert_eq!(syncs(&decoder) - before, 8 + 8 + 13, "three chunks");
    }

    /// The chunked prefill is not dtype aware (packet Q36-BF16-a): a decoder that holds BF16
    /// activations is refused before any host check, any permit, or any native call, and the state
    /// stays clean. Slice Q36-BF16-b lifts the refusal.
    #[test]
    fn a_decoder_with_bf16_activations_is_refused_by_the_prefill() {
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
        let mut state = decoder.zero_state().unwrap();
        let refused = decoder.prefill(&mut state, &[1, 2, 3], StepMode::Generate);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("F32 activations only")),
            "{:?}",
            refused.as_ref().err()
        );
        assert_eq!(state.pos(), 0, "a refusal does not advance");
        assert!(!state.is_poisoned(), "a refusal does not poison");
        assert_eq!(decoder.session_used(), 0, "a refusal takes no permit");
    }
}
