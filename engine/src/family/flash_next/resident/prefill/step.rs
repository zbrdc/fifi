//! Chunk step assembly (decision 0044 items 1, 4, 5, 8, and 9; plan S4 section 3).
//!
//! `prefill` runs `T` prompt tokens. Up to the dense QSA limit it runs them as one chunk through
//! `chunk_with_cap`, layer by layer. Tokens past the limit run as one-token `Prompt` steps on the
//! sparse decode path, and the last one runs in the requested mode (decision 0044 amendment of
//! 2026-10-09). A chunk makes the synchronizations of one token: the token-start import, one
//! router readback per layer, and one `evaluate`. `Generate` and `Record` add the readbacks of the
//! one-token step, for the last row of the chunk only. Two groups add planned synchronizations on
//! top, and the qualification build counts them exactly:
//!
//! - DeltaNet sub-chunks. The unrolled recurrence retires about 9 MiB per token and layer at real
//!   dimensions, so a layer runs over sub-chunks of `s` tokens and evaluates its new state between
//!   them. That is `delta_layers * (ceil(T / s) - 1)` evaluations. `s` comes from
//!   `prefill_plan::delta_subchunk`.
//! - Expert waves (packet PF-7b): `ChunkStats::waves_extra`, the sum of `waves - 1` over the layers.
//!
//! The retire cap of the adapter is a flat 512 MiB while a chunk runs and the previous cap (64 MiB)
//! afterwards. The tail drops of `run` (the residual and the other chunk graphs) retire under the
//! 512 MiB cap, before the restore. A tail above 64 MiB, such as a `[2048, C, H]` residual, makes
//! the first retire of the next call force one flush. Packet PF-G measures that flush.
//! Qualification test builds can override both caps for a sweep (`PrefillCapOverride`).
//!
//! A chunk marks the state poisoned before its first native call and publishes the new
//! state only after the final `evaluate` and the readbacks succeed. The route recorder of packet
//! T0 sees no chunk.
//!
//! The prefill page sets belong to the decoder from `open` to `close`. `close` releases the
//! lent sets through `PrefillPool::finish`, so no caller can skip the release.

#[cfg(feature = "native-fault-qualification")]
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::ops::{Ctx, dim, linear, values};
use super::super::qsa::MAX_POSITIONS;
use super::super::state::{LayerState, ResidentState, chunk_growth_bytes};
use super::super::step::{read_token, split_rows};
use super::super::weights::{RAttn, RDelta};
use super::super::{ResidentDecoder, StepMode, StepResult, hyper as decode_hyper};
use super::pool::PrefillPool;
use super::qsa::dense_rows;
use super::{deltanet, hyper, moe, ple, qsa, start};
use crate::family::flash_next::deltanet::DeltaShape;
use crate::family::flash_next::prefill_plan::{
    PREFILL_RETIRE_CAP, delta_ranges, delta_retire_per_token, delta_subchunk, prefill_set_experts,
    wave_count,
};
use crate::{EngError, sched::GrowthPermit, tensor::ExecutionTensor};

fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// The open prefill page sets of a decoder and the largest chunk that they serve.
pub(in super::super) struct Open {
    /// Two page sets sized from `ChunkCost` (decision 0044 items 5 and 6).
    pool: PrefillPool,
    /// The chunk size that the sets were sized for.
    max_tokens: u32,
}

/// Counts that a finished chunk reports besides its result. Tests compare them with the
/// synchronizations that the qualification build counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in super::super) struct ChunkStats {
    /// Extra evaluations of the expert waves: the sum of `waves - 1` over the MoE layers.
    pub(in super::super) waves_extra: u64,
}

/// Maps the prefill page sets for chunks of up to `max_tokens` tokens (decision 0044 item 5).
/// The union of a chunk runs in waves of at most `wave_cap` experts; `None` means the expert count
/// of the layer, which is one wave. The sets hold `ChunkCost::with_wave_cap(wave_cap)
/// .set_experts(max_tokens)` experts (`prefill_plan::prefill_set_experts`). A set that is already
/// open is closed first. Call it once per prompt, before its first chunk.
pub(in super::super) fn open(
    decoder: &mut ResidentDecoder<'_>,
    max_tokens: u32,
    wave_cap: Option<u32>,
) -> Result<(), EngError> {
    if max_tokens == 0 {
        return Err(bad("prefill chunk size must exceed zero"));
    }
    close(decoder);
    let moe = &decoder.weights.shape.moe;
    let experts = u32::try_from(moe.experts()).map_err(|_| bad("expert count exceeds u32"))?;
    let top_k = u32::try_from(moe.top_k()).map_err(|_| bad("top-k exceeds u32"))?;
    let set_experts = prefill_set_experts(experts, top_k, wave_cap.unwrap_or(experts), max_tokens)?;
    let pairs = max_tokens
        .checked_mul(top_k)
        .ok_or_else(|| bad("prefill pairs exceed u32"))?;
    let layers = decoder.weights.layers.iter().map(|layer| &layer.moe);
    let pool = PrefillPool::new(decoder.ckpt, layers, set_experts, pairs)?;
    decoder.prefill_pages = Some(Open { pool, max_tokens });
    Ok(())
}

/// Releases and drops the prefill page sets. It does nothing when none are open. Call it after the
/// last chunk of a prompt.
pub(in super::super) fn close(decoder: &mut ResidentDecoder<'_>) {
    if let Some(open) = decoder.prefill_pages.take() {
        open.pool.finish();
    }
}

/// The token count of a chunk. An empty chunk is a `BadRequest`.
fn row_count(tokens: &[u32]) -> Result<u32, EngError> {
    u32::try_from(tokens.len())
        .ok()
        .filter(|&rows| rows > 0)
        .ok_or_else(|| bad("a prefill chunk needs between one and u32::MAX tokens"))
}

/// Checks that the page sets are open and serve a chunk of `rows` tokens.
fn check_open(decoder: &ResidentDecoder<'_>, rows: u32) -> Result<(), EngError> {
    let open = decoder
        .prefill_pages
        .as_ref()
        .ok_or_else(|| bad("the prefill page sets are not open"))?;
    if rows > open.max_tokens {
        return Err(bad("a prefill chunk exceeds the open page sets"));
    }
    Ok(())
}

/// Runs `tokens` as a prefill and publishes the state after them. See `ResidentDecoder::prefill`.
///
/// The checks that need no native call come first: a closed or too small set of pages, a poisoned
/// state, and a prompt that passes `qsa::MAX_POSITIONS` return an error and change nothing. The
/// tokens that fit the dense QSA limit run as one chunk. The rest run as one-token steps, and the
/// last token of the call runs in `mode`. An error in the one-token part leaves the earlier tokens
/// published.
pub(in super::super) fn prefill(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    tokens: &[u32],
    mode: StepMode,
) -> Result<StepResult, EngError> {
    let rows = row_count(tokens)?;
    check_open(decoder, rows)?;
    state.check_step(&decoder.weights)?;
    if state
        .pos()
        .checked_add(rows)
        .is_none_or(|end| end > MAX_POSITIONS)
    {
        return Err(EngError::ContextLimit(format!(
            "a prefill chunk passes the {MAX_POSITIONS} positions of the resident QSA path"
        )));
    }
    let fit = rows.min(dense_rows(&decoder.weights.shape.qsa, state.pos())?);
    let (chunked, steps) = usize::try_from(fit)
        .ok()
        .and_then(|fit| tokens.split_at_checked(fit))
        .ok_or_else(|| bad("a prefill chunk exceeds the platform size"))?;
    let chunk_mode = if steps.is_empty() {
        mode
    } else {
        StepMode::Prompt
    };
    let head = if chunked.is_empty() {
        None
    } else {
        Some(chunk_with_cap(decoder, state, chunked, chunk_mode, prefill_cap())?.0)
    };
    match steps.split_last() {
        None => head.ok_or_else(|| bad("a prefill chunk needs at least one token")),
        Some((&last, before)) => {
            for &token in before {
                decoder.step(state, token, StepMode::Prompt)?;
            }
            decoder.step(state, last, mode)
        }
    }
}

/// Qualification builds only: the retire cap that `PrefillCapOverride` sets, in bytes. Zero means
/// no override.
#[cfg(feature = "native-fault-qualification")]
static RETIRE_CAP_OVERRIDE: AtomicU64 = AtomicU64::new(0);
/// Qualification builds only: the sub-chunk cap that `PrefillCapOverride` sets, in bytes. Zero
/// means no override.
#[cfg(feature = "native-fault-qualification")]
static SUBCHUNK_CAP_OVERRIDE: AtomicU64 = AtomicU64::new(0);

/// The retire cap of a prefill chunk: `PREFILL_RETIRE_CAP`, unless a qualification test overrides it.
fn prefill_cap() -> u64 {
    #[cfg(feature = "native-fault-qualification")]
    {
        let over = RETIRE_CAP_OVERRIDE.load(Ordering::Relaxed);
        if over != 0 {
            return over;
        }
    }
    PREFILL_RETIRE_CAP
}

/// The cap that sizes the DeltaNet sub-chunks of a chunk run with the retire cap `cap`: `cap`,
/// unless a qualification test overrides it.
fn subchunk_cap(cap: u64) -> u64 {
    #[cfg(feature = "native-fault-qualification")]
    {
        let over = SUBCHUNK_CAP_OVERRIDE.load(Ordering::Relaxed);
        if over != 0 {
            return over;
        }
    }
    cap
}

/// Tests in qualification builds only: overrides the retire cap of `prefill` chunks and the cap
/// that sizes their DeltaNet sub-chunks, for cap sweeps (a flush-cost measurement). `None` keeps
/// the product value (`PREFILL_RETIRE_CAP`, and the retire cap for the sub-chunks). A value must
/// be above zero. Dropping the guard removes both overrides. The overrides are process-wide, so
/// the holder owns the process, as the serial native test lock requires.
#[cfg(all(test, feature = "native-fault-qualification"))]
pub(in super::super) struct PrefillCapOverride;

#[cfg(all(test, feature = "native-fault-qualification"))]
impl PrefillCapOverride {
    pub(in super::super) fn set(retire: Option<u64>, subchunk: Option<u64>) -> Self {
        assert!(
            retire != Some(0) && subchunk != Some(0),
            "a cap override must be above zero"
        );
        RETIRE_CAP_OVERRIDE.store(retire.unwrap_or(0), Ordering::Relaxed);
        SUBCHUNK_CAP_OVERRIDE.store(subchunk.unwrap_or(0), Ordering::Relaxed);
        Self
    }
}

#[cfg(all(test, feature = "native-fault-qualification"))]
impl Drop for PrefillCapOverride {
    fn drop(&mut self) {
        RETIRE_CAP_OVERRIDE.store(0, Ordering::Relaxed);
        SUBCHUNK_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    }
}

/// Runs the tokens of `tokens` as one dense chunk with the retire cap `cap`. Tests use it to
/// force sub-chunks and flushes. It returns `ContextLimit` when the last row passes the dense
/// QSA limit, before any native call.
pub(in super::super) fn chunk_with_cap(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    tokens: &[u32],
    mode: StepMode,
    cap: u64,
) -> Result<(StepResult, ChunkStats), EngError> {
    let rows = row_count(tokens)?;
    check_open(decoder, rows)?;
    state.check_chunk(&decoder.weights, rows)?;
    let growth = chunk_growth_bytes(&decoder.weights, rows)?;
    let permit = decoder.budget.acquire_growth(growth)?;
    let old = state.begin();
    let result = match decoder.kernels.set_retire_cap(cap) {
        Ok(previous) => {
            let result = run(decoder, state, old, permit, tokens, mode, cap);
            let restored = decoder.kernels.set_retire_cap(previous);
            result.and_then(|out| restored.map(|_| out))
        }
        Err(error) => Err(error),
    };
    if let Some(open) = decoder.prefill_pages.as_mut() {
        open.pool.release_lent();
    }
    result
}

/// The last row of the chunk residual `h` `[T, C, H]` as `[C, H]`.
fn last_row(
    ctx: &mut Ctx<'_>,
    h: &ExecutionTensor,
    rows: u32,
    branches: u32,
    hidden: u32,
) -> Result<ExecutionTensor, EngError> {
    let last = rows
        .checked_sub(1)
        .ok_or_else(|| bad("a chunk needs a row"))?;
    let row = ctx
        .kernels
        .slice(h, &[last, 0, 0], &[rows, branches, hidden])?;
    ctx.kernels.reshape(&row, &[branches, hidden])
}

/// The output rows and the new state of one DeltaNet layer.
struct DeltaLayer {
    out: ExecutionTensor,
    rec: ExecutionTensor,
    ring: ExecutionTensor,
}

/// Runs one DeltaNet layer over the rows of `x` `[T, H]` in sub-chunks of `sub` tokens. After each
/// sub-chunk except the last it evaluates the new state and the sub-chunk output. That is a
/// planned synchronization: it frees the state tensors that the unrolled recurrence dropped.
fn delta_layer(
    ctx: &mut Ctx<'_>,
    shape: &DeltaShape,
    w: &RDelta,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
    x: &ExecutionTensor,
    sub: u32,
) -> Result<DeltaLayer, EngError> {
    let &[rows, hidden] = x.shape() else {
        return Err(bad("a DeltaNet chunk input must be [T, H]"));
    };
    // `delta_ranges` rejects zero rows and a zero size, and every range is non-empty, so
    // `deltanet::chunk` never starts with an empty sub-range (it does not reject one itself).
    let ranges = delta_ranges(rows, sub)?;
    let (mut rec, mut ring) = (rec.share(), ring.share());
    let mut parts: Vec<ExecutionTensor> = Vec::with_capacity(ranges.len());
    for (start, stop) in ranges {
        let piece = if start == 0 && stop == rows {
            x.share()
        } else {
            ctx.kernels.slice(x, &[start, 0], &[stop, hidden])?
        };
        let next = deltanet::chunk(ctx, shape, w, &rec, &ring, &piece)?;
        if stop < rows {
            ctx.kernels.evaluate(&[&next.rec, &next.ring, &next.out])?;
        }
        parts.push(next.out);
        rec = next.rec;
        ring = next.ring;
    }
    let out = match parts.as_slice() {
        [only] => only.share(),
        many => {
            let refs: Vec<&ExecutionTensor> = many.iter().collect();
            ctx.kernels.concatenate(&refs, 0)?
        }
    };
    Ok(DeltaLayer { out, rec, ring })
}

/// Builds the chunk graph, evaluates it once, reads back, and commits.
fn run(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    old: Vec<LayerState>,
    permit: GrowthPermit,
    tokens: &[u32],
    mode: StepMode,
    cap: u64,
) -> Result<(StepResult, ChunkStats), EngError> {
    let rows = row_count(tokens)?;
    let ckpt = decoder.ckpt;
    let weights = &decoder.weights;
    let shape = &weights.shape;
    let pos = state.pos();
    let (branches, hidden) = (dim(shape.hyper.branches())?, dim(shape.hyper.hidden())?);
    let per_token = delta_retire_per_token(
        dim(shape.delta.value_heads())?,
        dim(shape.delta.key_dim())?,
        dim(shape.delta.value_dim())?,
    )?;
    let sub = delta_subchunk(rows, subchunk_cap(cap), per_token)?;
    let pool = &mut decoder
        .prefill_pages
        .as_mut()
        .ok_or_else(|| bad("the prefill page sets are not open"))?
        .pool;
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &weights.consts,
    };
    let begun = start::import_tokens(&mut ctx, ckpt, weights, state.history(), tokens)?;
    let mut h = begun.hyper;
    let mut ple_ring = None;
    let mut layers = Vec::with_capacity(old.len());
    let mut records = Vec::new();
    let mut stats = ChunkStats::default();
    for (index, (layer, old_layer)) in weights.layers.iter().zip(old).enumerate() {
        if index == shape.ple_layer {
            let out = ple::block_rows(
                &mut ctx,
                &weights.ple,
                shape,
                &begun.emb,
                &h,
                state.ple_ring(),
            )?;
            h = out.hyper;
            ple_ring = Some(out.ring);
        }
        let read = hyper::read_rows(&mut ctx, &shape.hyper, &layer.attn_hyper, &h)?;
        let gate = read
            .gate
            .ok_or_else(|| bad("attention hyper-connection needs inject"))?;
        let (block, next) = match (&layer.attn, old_layer) {
            (RAttn::Delta(w), LayerState::Delta { rec, ring }) => {
                let s = delta_layer(&mut ctx, &shape.delta, w, &rec, &ring, &read.mixed, sub)?;
                (
                    s.out,
                    LayerState::Delta {
                        rec: s.rec,
                        ring: s.ring,
                    },
                )
            }
            (RAttn::Qsa(w), cache @ LayerState::Qsa { .. }) => {
                let s = qsa::chunk(&mut ctx, &shape.qsa, w, &cache, pos, &read.mixed)?;
                (s.out, s.state)
            }
            _ => return Err(bad("layer state kind does not match its weights")),
        };
        layers.push(next);
        h = hyper::apply_rows(&mut ctx, &shape.hyper, &h, &gate, &block)?;
        let read = hyper::read_rows(&mut ctx, &shape.hyper, &layer.mlp_hyper, &h)?;
        let gate = read
            .gate
            .ok_or_else(|| bad("mlp hyper-connection needs inject"))?;
        let (out, route) =
            moe::step_chunk(&mut ctx, ckpt, &shape.moe, &layer.moe, pool, &read.mixed)?;
        let union =
            u32::try_from(route.distinct.len()).map_err(|_| bad("prefill union exceeds u32"))?;
        let waves = wave_count(union, pool.set_experts())?;
        stats.waves_extra = stats
            .waves_extra
            .checked_add(u64::from(waves.saturating_sub(1)))
            .ok_or_else(|| bad("prefill wave count overflows u64"))?;
        h = hyper::apply_rows(&mut ctx, &shape.hyper, &h, &gate, &out)?;
        if mode == StepMode::Record {
            records.push(last_row(&mut ctx, &h, rows, branches, hidden)?);
        }
    }
    let ple_ring = ple_ring.ok_or_else(|| bad("the PLE layer did not run"))?;
    // The head runs on the last row only. `Prompt` skips it.
    let head = match mode {
        StepMode::Prompt => None,
        StepMode::Generate | StepMode::Record | StepMode::Logits => {
            let row = last_row(&mut ctx, &h, rows, branches, hidden)?;
            let mixed = decode_hyper::mixer_read(&mut ctx, &shape.hyper, &weights.mixer, &row)?;
            let logits = linear(&mut ctx, &weights.lm_head, &mixed)?;
            // `Logits` builds no `argmax`: the host sampler chooses the token.
            let next = if mode == StepMode::Logits {
                None
            } else {
                Some(ctx.kernels.argmax(&logits, 1, false)?)
            };
            Some((logits, next))
        }
    };
    let record_rows = if mode == StepMode::Record {
        let parts: Vec<&ExecutionTensor> = records.iter().collect();
        Some(ctx.kernels.concatenate(&parts, 0)?)
    } else {
        None
    };
    let mut outputs: Vec<&ExecutionTensor> = Vec::new();
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
            LayerState::Qsa { k, v, ik, .. } => outputs.extend([k, v, ik].into_iter().flatten()),
        }
    }
    outputs.push(&ple_ring);
    if let Some((logits, next)) = &head {
        if let Some(next) = next {
            outputs.push(next);
        }
        if matches!(mode, StepMode::Record | StepMode::Logits) {
            outputs.push(logits);
        }
    }
    if let Some(rows) = &record_rows {
        outputs.push(rows);
    }
    ctx.kernels.evaluate(&outputs)?;
    let token_out = match &head {
        Some((_, Some(next))) => Some(read_token(ctx.kernels, next, shape.vocab)?),
        _ => None,
    };
    let logits_out = match (&head, mode) {
        (Some((logits, _)), StepMode::Record | StepMode::Logits) => {
            Some(values(ctx.kernels, logits)?)
        }
        _ => None,
    };
    if mode == StepMode::Logits && logits_out.as_ref().map(Vec::len) != Some(shape.vocab) {
        return Err(EngError::Backend {
            operation: "flash_next",
            message: "logits length differs from the vocabulary".into(),
        });
    }
    let layer_out = match &record_rows {
        Some(rows) => Some(split_rows(values(ctx.kernels, rows)?, records.len())?),
        None => None,
    };
    state.commit_rows(rows, layers, ple_ring, begun.history, permit)?;
    Ok((
        StepResult {
            token: token_out,
            logits: logits_out,
            layer_out,
        },
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::super::deltanet::step as decode_delta_step;
    use super::super::super::ops::f32_tensor;
    use super::super::{assert_parity, import_f32};
    use super::*;
    use crate::family::flash_next::{load::FlashWeights, resident::fixtures};
    use crate::kernels::{BinaryOp, MlxKernels};
    use crate::sched::SessionBudget;
    use crate::weights::StreamedCheckpoint;

    /// 64 MiB, the base retire cap.
    #[cfg(feature = "native-fault-qualification")]
    const BASE_CAP: u64 = 64 << 20;
    /// A cap that gives the fixture a sub-chunk of 3 tokens: the fixture DeltaNet layer retires
    /// 3 * 2 * 32 * 32 * 4 = 24,576 bytes per token, and `(cap - 64 MiB) / 24,576 = 3`.
    #[cfg(feature = "native-fault-qualification")]
    const CAP_FOR_THREE: u64 = BASE_CAP + 3 * 24_576;

    fn load() -> (StreamedCheckpoint, FlashWeights) {
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        (ckpt, flash)
    }

    /// A decoder with the prefill page sets open for chunks of up to 16 tokens, in one wave.
    fn decoder<'c>(ckpt: &'c StreamedCheckpoint, flash: &FlashWeights) -> ResidentDecoder<'c> {
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), ckpt, flash).unwrap();
        decoder.open_prefill(16).unwrap();
        decoder
    }

    /// Tokens `from..from + count` of the test prompt. The ids cycle through 1 to 7, because the
    /// fixture embedding has 8 rows.
    fn prompt(from: u32, count: u32) -> Vec<u32> {
        (from..from + count).map(|i| i % 7 + 1).collect()
    }

    /// The process-wide synchronization count in qualification builds.
    fn syncs(kernels: &MlxKernels) -> Option<u64> {
        #[cfg(feature = "native-fault-qualification")]
        {
            Some(kernels.qualification_sync_count())
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = kernels;
            None
        }
    }

    /// Runs `tokens` as one-token `Prompt` steps.
    fn steps(decoder: &mut ResidentDecoder<'_>, state: &mut ResidentState, tokens: &[u32]) {
        for &token in tokens {
            decoder.step(state, token, StepMode::Prompt).unwrap();
        }
    }

    /// Asserts equal position and history, and equal tensors within the G2 tolerance.
    fn assert_same_state(
        decoder: &mut ResidentDecoder<'_>,
        a: &ResidentState,
        b: &ResidentState,
        what: &str,
    ) {
        assert_eq!(a.pos(), b.pos(), "{what}: position");
        assert_eq!(a.history(), b.history(), "{what}: history");
        assert!(!a.is_poisoned() && !b.is_poisoned(), "{what}: poison");
        assert_eq!(a.layers().len(), b.layers().len(), "{what}: layer count");
        let mut read = |t: &ExecutionTensor| values(&mut decoder.kernels, t).unwrap();
        for (l, (x, y)) in a.layers().iter().zip(b.layers()).enumerate() {
            match (x, y) {
                (
                    LayerState::Delta { rec: r1, ring: g1 },
                    LayerState::Delta { rec: r2, ring: g2 },
                ) => {
                    assert_parity(&read(r1), &read(r2), &format!("{what} layer {l} rec"));
                    assert_parity(&read(g1), &read(g2), &format!("{what} layer {l} ring"));
                }
                (
                    LayerState::Qsa {
                        k: k1,
                        v: v1,
                        ik: i1,
                        len: n1,
                    },
                    LayerState::Qsa {
                        k: k2,
                        v: v2,
                        ik: i2,
                        len: n2,
                    },
                ) => {
                    assert_eq!(n1, n2, "{what} layer {l} length");
                    for ((p, q), name) in [(k1, k2), (v1, v2), (i1, i2)]
                        .into_iter()
                        .zip(["k", "v", "ik"])
                    {
                        match (p, q) {
                            (Some(p), Some(q)) => {
                                assert_parity(
                                    &read(p),
                                    &read(q),
                                    &format!("{what} layer {l} {name}"),
                                );
                            }
                            (None, None) => {}
                            _ => panic!("{what} layer {l} {name}: one cache is empty"),
                        }
                    }
                }
                _ => panic!("{what}: layer {l} kinds differ"),
            }
        }
        assert_parity(
            &read(a.ple_ring()),
            &read(b.ple_ring()),
            &format!("{what} ple ring"),
        );
    }

    /// G2. After a chunk of `T` tokens the state equals the state after `T` `Prompt` steps, for
    /// `P` cached positions of 0 and 5 and `T` of 1, 2, 3, 4, 7, and 10 (`P + T` stays within the
    /// 19 dense positions). A chunk of 10 replaces all 9 columns of the PLE ring. The next
    /// `Record` step of both states agrees as well.
    #[test]
    fn a_chunk_leaves_the_state_of_t_prompt_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        for p in [0u32, 5] {
            for t in [1u32, 2, 3, 4, 7, 10] {
                let what = format!("P {p} T {t}");
                let mut chunked = decoder.zero_state().unwrap();
                steps(&mut decoder, &mut chunked, &prompt(0, p));
                decoder
                    .prefill(&mut chunked, &prompt(p, t), StepMode::Prompt)
                    .unwrap();
                let mut stepped = decoder.zero_state().unwrap();
                steps(&mut decoder, &mut stepped, &prompt(0, p + t));
                assert_same_state(&mut decoder, &chunked, &stepped, &what);
                let got = decoder.step(&mut chunked, 3, StepMode::Record).unwrap();
                let want = decoder.step(&mut stepped, 3, StepMode::Record).unwrap();
                assert_eq!(got.token, want.token, "{what}: next token");
                assert_parity(
                    &got.logits.unwrap(),
                    &want.logits.unwrap(),
                    &format!("{what} next logits"),
                );
            }
        }
    }

    /// A chunk whose mode is `Logits` returns the logits row of the same chunk in `Record` mode,
    /// bit for bit, with exactly one vocabulary of values and no token. It builds no argmax.
    #[test]
    fn a_logits_chunk_returns_the_record_logits_and_no_token() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        for (p, t) in [(0u32, 1u32), (0, 4), (5, 3)] {
            let what = format!("P {p} T {t}");
            let mut recorded = decoder.zero_state().unwrap();
            steps(&mut decoder, &mut recorded, &prompt(0, p));
            let want = decoder
                .prefill(&mut recorded, &prompt(p, t), StepMode::Record)
                .unwrap();
            let mut sampled = decoder.zero_state().unwrap();
            steps(&mut decoder, &mut sampled, &prompt(0, p));
            let got = decoder
                .prefill(&mut sampled, &prompt(p, t), StepMode::Logits)
                .unwrap();
            assert_eq!(got.token, None, "{what}: token");
            assert!(got.layer_out.is_none(), "{what}: layer rows");
            let (got, want) = (got.logits.unwrap(), want.logits.unwrap());
            assert_eq!(got.len(), decoder.weights.shape.vocab, "{what}: length");
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{what}: logits bits");
            assert_same_state(&mut decoder, &sampled, &recorded, &what);
        }
    }

    /// G2, chunk-boundary independence: chunks of 3 and 5 equal one chunk of 8.
    #[test]
    fn two_chunks_equal_one_chunk() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let mut split = decoder.zero_state().unwrap();
        decoder
            .prefill(&mut split, &prompt(0, 3), StepMode::Prompt)
            .unwrap();
        decoder
            .prefill(&mut split, &prompt(3, 5), StepMode::Prompt)
            .unwrap();
        let mut whole = decoder.zero_state().unwrap();
        decoder
            .prefill(&mut whole, &prompt(0, 8), StepMode::Prompt)
            .unwrap();
        assert_same_state(&mut decoder, &split, &whole, "chunks (3, 5) and (8)");
    }

    /// G1. The head, the token, and the recorded layer rows of a chunk are those of the one-token
    /// `Record` step at the last token.
    #[test]
    fn the_last_row_of_a_chunk_matches_the_one_token_record_step() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        for p in [0u32, 5] {
            for t in [1u32, 2, 3, 4, 7, 10] {
                let what = format!("P {p} T {t}");
                let mut chunked = decoder.zero_state().unwrap();
                steps(&mut decoder, &mut chunked, &prompt(0, p));
                let got = decoder
                    .prefill(&mut chunked, &prompt(p, t), StepMode::Record)
                    .unwrap();
                let mut stepped = decoder.zero_state().unwrap();
                steps(&mut decoder, &mut stepped, &prompt(0, p + t - 1));
                let want = decoder
                    .step(&mut stepped, prompt(p + t - 1, 1)[0], StepMode::Record)
                    .unwrap();
                assert_eq!(got.token, want.token, "{what}: token");
                assert_parity(
                    &got.logits.unwrap(),
                    &want.logits.unwrap(),
                    &format!("{what} logits"),
                );
                let (got_rows, want_rows) = (got.layer_out.unwrap(), want.layer_out.unwrap());
                assert_eq!(got_rows.len(), want_rows.len(), "{what}: recorded layers");
                for (l, (g, w)) in got_rows.iter().zip(&want_rows).enumerate() {
                    assert_parity(g, w, &format!("{what} layer {l}"));
                }
            }
        }
    }

    /// G8. Ratio 4 and budget 16 give 19 dense positions. A dense chunk to position 20 returns
    /// `ContextLimit` and changes nothing: no synchronization, no poison, no charge. A chunk to
    /// position 19 runs. (`prefill` splits such a prompt instead; see the crossing test.)
    #[test]
    fn a_chunk_past_the_dense_limit_returns_context_limit_and_changes_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        decoder.open_prefill(20).unwrap();
        decoder.set_session_budget(SessionBudget::new(100_000));
        let mut state = decoder.zero_state().unwrap();
        steps(&mut decoder, &mut state, &prompt(0, 10));
        let bytes = state.byte_len().unwrap();
        let used = decoder.session_used();
        let before = syncs(&decoder.kernels);
        let past = chunk_with_cap(
            &mut decoder,
            &mut state,
            &prompt(10, 10),
            StepMode::Prompt,
            PREFILL_RETIRE_CAP,
        );
        assert!(matches!(past, Err(EngError::ContextLimit(_))), "{past:?}");
        assert_eq!(syncs(&decoder.kernels), before, "no synchronization");
        assert_eq!(state.pos(), 10, "position");
        assert!(!state.is_poisoned(), "a context limit does not poison");
        assert_eq!(
            state.byte_len().unwrap(),
            bytes,
            "the layer states stay in place"
        );
        assert_eq!(decoder.session_used(), used, "no charge");
        decoder
            .prefill(&mut state, &prompt(10, 8), StepMode::Prompt)
            .unwrap();
        assert_eq!(state.pos(), 18, "a chunk to 18 runs");
        decoder
            .prefill(&mut state, &prompt(18, 1), StepMode::Prompt)
            .unwrap();
        assert_eq!(state.pos(), 19, "the last dense position");
    }

    /// G8. A failure inside the chunk poisons the state, releases the permit, and restores the
    /// base retire cap. `reset` recovers. Token 8 is outside the 8-row embedding, so the
    /// token-start import fails on the host after `begin` and after the cap is raised.
    #[test]
    fn a_failed_chunk_poisons_the_state_until_reset() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        decoder.set_session_budget(SessionBudget::new(100_000));
        let mut state = decoder.zero_state().unwrap();
        let failed = decoder.prefill(&mut state, &[1, 8, 2], StepMode::Generate);
        assert!(
            matches!(
                failed,
                Err(EngError::BadRequest(_) | EngError::Checkpoint(_))
            ),
            "{failed:?}"
        );
        assert!(state.is_poisoned(), "a failed chunk poisons the state");
        assert_eq!(state.pos(), 0, "a failed chunk does not advance");
        assert_eq!(decoder.session_used(), 0, "the permit is released");
        // The chunk raised the retire cap before it failed. The restore must still run. (The
        // call also sets the base cap, which is the cap that the decoder already had.)
        assert_eq!(
            decoder.kernels.set_retire_cap(64 << 20).unwrap(),
            64 << 20,
            "a failed chunk restores the base retire cap"
        );
        let before = syncs(&decoder.kernels);
        let again = decoder.prefill(&mut state, &[1, 2, 3], StepMode::Generate);
        assert!(
            matches!(&again, Err(EngError::BadRequest(m)) if m.contains("reset")),
            "{again:?}"
        );
        assert_eq!(
            syncs(&decoder.kernels),
            before,
            "a poisoned chunk makes no synchronization"
        );
        decoder.reset(&mut state).unwrap();
        assert!(!state.is_poisoned(), "reset clears the poison");
        decoder
            .prefill(&mut state, &[1, 2, 3], StepMode::Prompt)
            .unwrap();
        let mut control = decoder.zero_state().unwrap();
        steps(&mut decoder, &mut control, &[1, 2, 3]);
        assert_same_state(&mut decoder, &state, &control, "after reset");
    }

    /// The fixture has one QSA layer, so a position adds (2 * 64 + 32) values of 4 bytes: 640
    /// bytes. A chunk of 3 charges 1,920. A chunk of 2 needs 1,280 and a cap of 1,000 denies it.
    #[test]
    fn a_chunk_charges_the_stored_qsa_growth() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        decoder.set_session_budget(SessionBudget::new(10_000));
        let mut state = decoder.zero_state().unwrap();
        decoder
            .prefill(&mut state, &prompt(0, 3), StepMode::Prompt)
            .unwrap();
        assert_eq!(decoder.session_used(), 1_920, "charge after the chunk");
        decoder.reset(&mut state).unwrap();
        assert_eq!(decoder.session_used(), 0, "reset releases the permit");

        // The budget changes only before any state exists.
        drop(state);
        decoder.set_session_budget(SessionBudget::new(1_000));
        let mut state = decoder.zero_state().unwrap();
        let before = syncs(&decoder.kernels);
        let denied = decoder.prefill(&mut state, &prompt(0, 2), StepMode::Prompt);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m.contains("1280")),
            "{denied:?}"
        );
        assert_eq!(syncs(&decoder.kernels), before, "no synchronization");
        assert_eq!(state.pos(), 0, "position");
        assert!(!state.is_poisoned(), "a denial does not poison");
        assert_eq!(decoder.session_used(), 0, "a denial charges nothing");
    }

    /// An empty chunk, a chunk without open page sets, and a chunk wider than the open sets return
    /// `BadRequest` and change nothing.
    #[test]
    fn an_empty_closed_or_oversized_chunk_is_rejected() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let mut state = decoder.zero_state().unwrap();
        let before = syncs(&decoder.kernels);
        let empty = decoder.prefill(&mut state, &[], StepMode::Prompt);
        assert!(matches!(empty, Err(EngError::BadRequest(_))), "{empty:?}");
        decoder.close_prefill();
        let closed = decoder.prefill(&mut state, &[1], StepMode::Prompt);
        assert!(
            matches!(&closed, Err(EngError::BadRequest(m)) if m.contains("not open")),
            "{closed:?}"
        );
        decoder.open_prefill(2).unwrap();
        let wide = decoder.prefill(&mut state, &[1, 2, 3], StepMode::Prompt);
        assert!(
            matches!(&wide, Err(EngError::BadRequest(m)) if m.contains("exceeds")),
            "{wide:?}"
        );
        assert_eq!(syncs(&decoder.kernels), before, "no synchronization");
        assert!(!state.is_poisoned(), "a rejected chunk does not poison");
    }

    /// A prompt that passes the 19 dense positions runs its dense part as one chunk and the rest as
    /// one-token `Prompt` steps on the sparse path, and the last token runs in `Generate`. The
    /// state and the token equal those of one-token steps. Rows (cached positions, tokens): a split
    /// of 9 and 6, a split of 15 and 1, and a prompt that starts at the limit and has no chunk.
    #[test]
    fn a_prompt_that_crosses_the_dense_limit_matches_one_token_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        for (p, t) in [(10u32, 15u32), (4, 16), (19, 3)] {
            let what = format!("P {p} T {t}");
            let mut chunked = decoder.zero_state().unwrap();
            steps(&mut decoder, &mut chunked, &prompt(0, p));
            let got = decoder
                .prefill(&mut chunked, &prompt(p, t), StepMode::Generate)
                .unwrap();
            let mut stepped = decoder.zero_state().unwrap();
            steps(&mut decoder, &mut stepped, &prompt(0, p + t - 1));
            let want = decoder
                .step(&mut stepped, prompt(p + t - 1, 1)[0], StepMode::Generate)
                .unwrap();
            assert_eq!(chunked.pos(), p + t, "{what}: position");
            assert!(got.token.is_some(), "{what}: the last token generates");
            assert_eq!(got.token, want.token, "{what}: token");
            assert_same_state(&mut decoder, &chunked, &stepped, &what);
        }
    }

    /// Qualification count of the same prompt (9 + 6): the dense part is one chunk of
    /// `layers + 2 = 4` synchronizations, the five `Prompt` steps make 4 each, and the `Generate`
    /// step makes 5. A prompt of one-token steps only would make 14 * 4 + 5 = 61 minus the chunk.
    /// The flat prefill retire cap gives one DeltaNet part, so the chunk adds no evaluation.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_crossing_prompt_makes_one_chunk_and_one_token_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let mut state = decoder.zero_state().unwrap();
        steps(&mut decoder, &mut state, &prompt(0, 10));
        let before = decoder.kernels.qualification_sync_count();
        decoder
            .prefill(&mut state, &prompt(10, 15), StepMode::Generate)
            .unwrap();
        let used = decoder.kernels.qualification_sync_count() - before;
        assert_eq!(used, 4 + 5 * 4 + 5, "one chunk of 9 and six steps");
    }

    /// G2 with expert waves. With a wave cap of 1 every layer runs one wave per distinct expert,
    /// so each chunk reads its experts in at least two waves. The state equals the state after `T`
    /// `Prompt` steps.
    #[test]
    fn waves_leave_the_state_of_t_prompt_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        decoder.open_prefill_waves(16, 1).unwrap();
        for (p, t) in [(0u32, 4u32), (0, 10), (5, 7)] {
            let what = format!("waves P {p} T {t}");
            let mut chunked = decoder.zero_state().unwrap();
            steps(&mut decoder, &mut chunked, &prompt(0, p));
            decoder
                .prefill(&mut chunked, &prompt(p, t), StepMode::Prompt)
                .unwrap();
            let mut stepped = decoder.zero_state().unwrap();
            steps(&mut decoder, &mut stepped, &prompt(0, p + t));
            assert_same_state(&mut decoder, &chunked, &stepped, &what);
        }
    }

    /// Four MoE calls depend on each other without a readback of their outputs: the output of one
    /// call, scaled by 1e-5, is the input of the next. Both sets start as NaN bytes. With a wave
    /// cap of 4 each call is one wave. The router readback of the next call evaluates the call
    /// before and releases its set. With a wave cap of 1 the first call has four waves (the
    /// asserted premise), and each wave before the last is evaluated and released at once. The
    /// result equals the same chain with every output read back and re-imported.
    ///
    /// This test proves the release rules: a set that is never released leaves no free set for
    /// the third call or the third wave, and a NaN byte of a set never reaches an output. It does
    /// not detect an early reuse of a set by itself. A set that a pending graph reads is not
    /// writable (`try_write` returns `None`), so an early release shows as set exhaustion.
    #[test]
    fn four_dependent_layers_chain_through_the_two_prefill_sets() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let input: Vec<f32> = (0..4u32).flat_map(delta_row).collect();
        for cap in [4u32, 1] {
            let what = format!("wave cap {cap}");
            let layers = || decoder.weights.layers.iter().map(|layer| &layer.moe);
            let mut chained = PrefillPool::new(&ckpt, layers(), cap, 8).unwrap();
            let mut reference = PrefillPool::new(&ckpt, layers(), cap, 8).unwrap();
            chained.poison(0xFF);
            reference.poison(0xFF);
            let moe_shape = &decoder.weights.shape.moe;
            let mut ctx = Ctx {
                kernels: &mut decoder.kernels,
                consts: &decoder.weights.consts,
            };
            let scale = import_f32(ctx.kernels, vec![1], &[1e-5]);
            let mut x = import_f32(ctx.kernels, vec![4, 64], &input);
            for layer in 0..4 {
                let moe_w = &decoder.weights.layers[layer % 2].moe;
                let (out, route) =
                    moe::step_chunk(&mut ctx, &ckpt, moe_shape, moe_w, &mut chained, &x).unwrap();
                if layer == 0 {
                    // The router logits of expert `e` are `64 e S + C`, where `S` is the sum of
                    // the row. The rows of `delta_row` have `S` of -0.375, 0, 0.375, and -0.0625,
                    // so the top two experts are {0, 1} and {3, 2} and the union holds all four.
                    assert!(
                        route.distinct.len() >= 3,
                        "{what}: the first call needs at least three experts: {:?}",
                        route.distinct
                    );
                }
                x = ctx.kernels.binary(BinaryOp::Mul, &out, &scale).unwrap();
            }
            let got = values(ctx.kernels, &x).unwrap();
            let mut want = input.clone();
            for layer in 0..4 {
                let moe_w = &decoder.weights.layers[layer % 2].moe;
                let x = import_f32(ctx.kernels, vec![4, 64], &want);
                let (out, _) =
                    moe::step_chunk(&mut ctx, &ckpt, moe_shape, moe_w, &mut reference, &x).unwrap();
                want = values(ctx.kernels, &out)
                    .unwrap()
                    .iter()
                    .map(|v| v * 1e-5)
                    .collect();
            }
            assert!(
                got.iter().all(|v| v.is_finite()) && got.iter().any(|&v| v != 0.0),
                "{what}: the chain gives finite nonzero values"
            );
            assert_parity(&got, &want, &what);
        }
    }

    /// Closing releases the lent sets and drops the pool, and a second close does nothing. A
    /// reopened prefill maps new sets and runs a chunk that equals one-token steps.
    #[test]
    fn closing_the_prefill_releases_the_sets_and_a_reopened_prefill_runs() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let mut first = decoder.zero_state().unwrap();
        decoder
            .prefill(&mut first, &prompt(0, 4), StepMode::Prompt)
            .unwrap();
        decoder.close_prefill();
        decoder.close_prefill();
        let closed = decoder.prefill(&mut first, &prompt(4, 1), StepMode::Prompt);
        assert!(
            matches!(&closed, Err(EngError::BadRequest(m)) if m.contains("not open")),
            "{closed:?}"
        );
        decoder.open_prefill(8).unwrap();
        decoder
            .prefill(&mut first, &prompt(4, 6), StepMode::Prompt)
            .unwrap();
        let mut stepped = decoder.zero_state().unwrap();
        steps(&mut decoder, &mut stepped, &prompt(0, 10));
        assert_same_state(&mut decoder, &first, &stepped, "reopened prefill");
    }

    /// A DeltaNet input row for the carry test.
    fn delta_row(pos: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((i + 7 * pos) % 13) as f32 - 6.0) * 0.0625)
            .collect()
    }

    /// The sub-chunk loop carries the recurrent state and the convolution ring from one sub-range
    /// to the next. Three decode steps first make both nonzero. Then 4 rows run as one chunk of
    /// sub-chunks 3 + 1 and 2 + 2, and as 4 decode steps. A sub-range of one token (the short last
    /// part of 3 + 1) starts from a nonzero ring. A zero size is `BadRequest` before any native call.
    #[test]
    fn delta_sub_chunks_carry_the_state_from_a_nonzero_ring() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let zero = decoder.zero_state().unwrap();
        let LayerState::Delta { rec, ring } = &zero.layers()[0] else {
            panic!("layer 0 is not DeltaNet");
        };
        let RAttn::Delta(w) = &decoder.weights.layers[0].attn else {
            panic!("resident layer 0 is not DeltaNet");
        };
        let shape = &decoder.weights.shape;
        let mut ctx = Ctx {
            kernels: &mut decoder.kernels,
            consts: &decoder.weights.consts,
        };
        let import = |ctx: &mut Ctx<'_>, rows: u32, first: u32| {
            let flat: Vec<f32> = (first..first + rows).flat_map(delta_row).collect();
            ctx.kernels
                .import_tensor(&f32_tensor(vec![rows, 64], &flat).unwrap())
                .unwrap()
        };
        // Warm-up: three decode steps from the zero state.
        let (mut rec0, mut ring0) = (rec.share(), ring.share());
        for pos in 100..103 {
            let x = import(&mut ctx, 1, pos);
            let s = decode_delta_step(&mut ctx, &shape.delta, w, &rec0, &ring0, &x).unwrap();
            (rec0, ring0) = (s.rec, s.ring);
        }
        // Reference: four decode steps from that state.
        let (mut r, mut g) = (rec0.share(), ring0.share());
        let mut want = Vec::new();
        for pos in 200..204 {
            let x = import(&mut ctx, 1, pos);
            let s = decode_delta_step(&mut ctx, &shape.delta, w, &r, &g, &x).unwrap();
            want.extend(values(ctx.kernels, &s.out).unwrap());
            (r, g) = (s.rec, s.ring);
        }
        let (want_rec, want_ring) = (
            values(ctx.kernels, &r).unwrap(),
            values(ctx.kernels, &g).unwrap(),
        );
        assert!(want_ring.iter().any(|&v| v != 0.0), "the ring is nonzero");
        let x = import(&mut ctx, 4, 200);
        for sub in [3u32, 2] {
            let what = format!("sub-chunks of {sub}");
            let got = delta_layer(&mut ctx, &shape.delta, w, &rec0, &ring0, &x, sub).unwrap();
            assert_parity(
                &values(ctx.kernels, &got.out).unwrap(),
                &want,
                &format!("{what} output"),
            );
            assert_parity(
                &values(ctx.kernels, &got.rec).unwrap(),
                &want_rec,
                &format!("{what} rec"),
            );
            assert_parity(
                &values(ctx.kernels, &got.ring).unwrap(),
                &want_ring,
                &format!("{what} ring"),
            );
        }
        let rejected = delta_layer(&mut ctx, &shape.delta, w, &rec0, &ring0, &x, 0);
        assert!(
            matches!(rejected, Err(EngError::BadRequest(_))),
            "zero size"
        );
    }

    /// G4 on the fixture: 2 layers, 1 DeltaNet layer. A `Prompt` chunk makes `2 + 2` synchronizations,
    /// `Generate` adds 1, `Record` adds 3, each DeltaNet layer adds `ceil(T / s) - 1`, and the
    /// expert waves add `ChunkStats::waves_extra`. A cap of 64 MiB leaves no room, so `s` is 1.
    /// `CAP_FOR_THREE` gives `s = 3`. The default `prefill` cap of 512 MiB gives `s = T`. Rows:
    /// tokens, cap, mode, expected synchronizations of one wave per layer. The second block uses a
    /// wave cap of 1, so every layer reads at least two waves.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn each_chunk_uses_its_planned_synchronizations_and_no_forced_flush() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let rows = [
            (3u32, BASE_CAP, StepMode::Prompt, 6u64),
            (3, CAP_FOR_THREE, StepMode::Prompt, 4),
            (4, CAP_FOR_THREE, StepMode::Prompt, 5),
            (7, CAP_FOR_THREE, StepMode::Prompt, 6),
            (7, CAP_FOR_THREE, StepMode::Generate, 7),
            (7, CAP_FOR_THREE, StepMode::Record, 9),
        ];
        let check = |decoder: &mut ResidentDecoder<'_>, row: (u32, u64, StepMode, u64), waves| {
            let (tokens, cap, mode, want) = row;
            let what = format!("T {tokens} cap {cap} {mode:?}");
            let mut state = decoder.zero_state().unwrap();
            let s0 = decoder.kernels.qualification_sync_count();
            let f0 = decoder.kernels.qualification_retire_stats().0;
            let (_, stats) =
                chunk_with_cap(decoder, &mut state, &prompt(0, tokens), mode, cap).unwrap();
            let used = decoder.kernels.qualification_sync_count() - s0;
            if waves {
                assert!(stats.waves_extra >= 2, "{what}: waves {stats:?}");
            } else {
                assert_eq!(stats.waves_extra, 0, "{what}: one wave per layer");
            }
            assert_eq!(used, want + stats.waves_extra, "{what}: synchronizations");
            let flushes = decoder.kernels.qualification_retire_stats().0 - f0;
            assert_eq!(flushes, 0, "{what}: forced flushes");
            let mut stepped = decoder.zero_state().unwrap();
            steps(decoder, &mut stepped, &prompt(0, tokens));
            if mode == StepMode::Prompt {
                assert_same_state(decoder, &state, &stepped, &what);
            }
        };
        for row in rows {
            check(&mut decoder, row, false);
        }
        decoder.open_prefill_waves(8, 1).unwrap();
        check(&mut decoder, (4, CAP_FOR_THREE, StepMode::Prompt, 5), true);
        check(
            &mut decoder,
            (7, CAP_FOR_THREE, StepMode::Generate, 7),
            true,
        );
        // The entry point sets the flat 512 MiB cap, so a layer runs in one part.
        decoder.open_prefill(8).unwrap();
        let mut state = decoder.zero_state().unwrap();
        let s0 = decoder.kernels.qualification_sync_count();
        decoder
            .prefill(&mut state, &prompt(0, 7), StepMode::Prompt)
            .unwrap();
        assert_eq!(
            decoder.kernels.qualification_sync_count() - s0,
            4,
            "the prefill cap leaves one DeltaNet part"
        );
    }

    /// The cap applies during the chunk and the previous cap returns afterwards. A cap of 1 byte
    /// forces a flush at every retire, so the chunk makes forced flushes. The next call of
    /// `set_retire_cap` returns the cap that the chunk restored.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn the_retire_cap_applies_during_the_chunk_and_returns_afterwards() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder(&ckpt, &flash);
        let mut state = decoder.zero_state().unwrap();
        let f0 = decoder.kernels.qualification_retire_stats().0;
        chunk_with_cap(&mut decoder, &mut state, &prompt(0, 3), StepMode::Prompt, 1).unwrap();
        let flushes = decoder.kernels.qualification_retire_stats().0 - f0;
        assert!(flushes > 0, "a 1-byte cap forces flushes: {flushes}");
        let restored = decoder.kernels.set_retire_cap(1 << 20).unwrap();
        assert_eq!(restored, BASE_CAP, "the chunk restored the previous cap");
        decoder.kernels.set_retire_cap(BASE_CAP).unwrap();
        let mut again = decoder.zero_state().unwrap();
        let f1 = decoder.kernels.qualification_retire_stats().0;
        decoder
            .prefill(&mut again, &prompt(0, 3), StepMode::Prompt)
            .unwrap();
        assert_eq!(
            decoder.kernels.qualification_retire_stats().0 - f1,
            0,
            "no flush at the default cap"
        );
    }

    /// Fault tests. A native failure is terminal for the process, so each runs in a child. This
    /// repeats the child launcher of the decode step tests with this module's test path.
    #[cfg(feature = "native-fault-qualification")]
    mod faults {
        use super::*;
        use std::{
            env,
            io::Read,
            process::{Command, Stdio},
            thread,
            time::{Duration, Instant},
        };

        const CHILD: &str = "INFERENCE_RESIDENT_CHUNK_CHILD";
        const DEADLINE: Duration = Duration::from_secs(180);

        fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = pipe.read_to_end(&mut bytes);
                String::from_utf8_lossy(&bytes).into_owned()
            })
        }

        /// In the parent, runs `name` in a child with the `fault` variable. Returns true in the parent.
        fn delegated_to_child(name: &str, fault: (&str, &str)) -> bool {
            if env::var_os(CHILD).is_some() {
                return false;
            }
            let path = module_path!().split_once("::").unwrap().1;
            let mut child = Command::new(env::current_exe().unwrap())
                .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
                .env(CHILD, "1")
                .env(fault.0, fault.1)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let stdout = drain(child.stdout.take().unwrap());
            let stderr = drain(child.stderr.take().unwrap());
            let started = Instant::now();
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break Some(status);
                }
                if started.elapsed() >= DEADLINE {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    break None;
                }
                thread::sleep(Duration::from_millis(20));
            };
            let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
            assert!(
                status.is_some_and(|status| status.success()) && captured.contains("1 passed"),
                "child did not pass {name}: {captured}"
            );
            true
        }

        #[test]
        fn an_injected_evaluation_failure_poisons_the_chunk_state() {
            let name = "an_injected_evaluation_failure_poisons_the_chunk_state";
            if delegated_to_child(name, ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "evaluate")) {
                return;
            }
            let (ckpt, flash) = load();
            let mut decoder = decoder(&ckpt, &flash);
            decoder.set_session_budget(SessionBudget::new(100_000));
            let mut state = decoder.zero_state().unwrap();
            let failed = decoder.prefill(&mut state, &prompt(0, 3), StepMode::Generate);
            assert!(
                matches!(
                    failed,
                    Err(EngError::Backend {
                        operation: "evaluate",
                        ..
                    })
                ),
                "{failed:?}"
            );
            assert!(state.is_poisoned(), "a failed evaluation poisons the state");
            assert_eq!(state.pos(), 0, "nothing is published");
            assert_eq!(decoder.session_used(), 0, "the permit is released");
            let before = decoder.kernels.qualification_sync_count();
            let again = decoder.prefill(&mut state, &prompt(0, 3), StepMode::Generate);
            assert!(
                matches!(&again, Err(EngError::BadRequest(m)) if m.contains("reset")),
                "{again:?}"
            );
            assert_eq!(
                decoder.kernels.qualification_sync_count(),
                before,
                "no synchronization"
            );
            // The failure is terminal (decision 0039 item 7), so reset cannot import a zero state.
            let reset = decoder.reset(&mut state);
            assert!(matches!(reset, Err(EngError::Backend { .. })), "{reset:?}");
            assert!(
                state.is_poisoned(),
                "a failed reset leaves the state poisoned"
            );
        }
    }
}
