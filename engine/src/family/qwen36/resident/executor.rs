//! The Q36 executor: `ExecutorOps` over the resident Q36 decoder.
//!
//! It follows the Flash-Next executor (`flash_next/resident/executor.rs`) and shares the token side
//! of the session with it (`family::session::DecodeSession`). `Q36Step` is the model side: a
//! `Prompt` step for each prompt token except the last, and a `Logits` step for each generated
//! token.
//!
//! Ownership: the engine thread builds the executor and its session inside the `make` closure of
//! `spawn_engine`. Nothing native crosses a thread boundary.
//!
//! Checkpoint lifetime: `ResidentDecoder<'c>` borrows its `StreamedCheckpoint`, and `spawn_engine`
//! needs an executor without a borrowed lifetime. `open` leaks one checkpoint with `Box::leak`, as
//! the Flash-Next executor does. Each call to `open` leaks one handle, so each engine reload leaks
//! one more. The checkpoint stays open until the process ends, which is later than any GPU work
//! that reads it.
//!
//! Context limit: `Q36Rules::context_limit` is the first position that the decoder refuses. A
//! prompt of that many tokens or more fails with `ContextLimit`, and a job ends with `Length` when
//! its next feed would need that position. The caller sets it: the default is 32,768
//! (`settings.rs`). A larger limit needs a larger `session_cap`: `from_parts` rejects a limit
//! whose cache growth (`bytes per position * context_limit`) exceeds the cap with `MemoryPlan`,
//! before it imports any weight. The decoder charges the same growth to its `SessionBudget`. With
//! concurrent sessions the cap is aggregate: other live contexts and snapshots can deny growth
//! below an individual session's context limit.
//!
//! Failure handling and teardown follow the Flash-Next executor: a native failure returns
//! `EngError::Backend` unchanged, and the worker marks the engine `NativeFailed`. The session
//! drops before the executor, and the decoder waits for GPU completion when it drops.
//!
//! Batched prefill: the chunk path is off until `plan_prefill` stores a chunk size, and then every
//! prompt row except the last token is fed in chunks through `ResidentDecoder::prefill` in `Prompt`
//! mode. Without a size, every row is a one-token `Prompt` step, as before, and the results are bit
//! for bit those of the earlier code. The executor owns the one chunk size. `plan_prefill` picks
//! the largest candidate whose workspace (`chunk_plan::ChunkCost`) fits the engine budget after the
//! weights and the session reserve, at most the largest chunk (`Q36Settings::prefill_chunk`). The
//! worker piece is the planned chunk, or 16 (`NO_BUDGET_PIECE`) when no chunk is planned, so one
//! `prefill` call never feeds more rows than a piece. It writes the size to the executor and to the
//! decoder (`ResidentDecoder::chunk_rows`) in one place, `apply_chunk`, so the decoder never splits
//! a chunk again and no other code sets either value. The decoder default of 512 rows serves
//! decoder tests only. A run of one row is a one-token step, as in the Flash-Next executor, and a
//! chunk equals the steps within the chunk-parity tolerance, not bit for bit. A chunk routes as its
//! decoder holds (`Routing::Device`, the default, routes on the device; `Routing::Host` keeps the
//! host-routed reference), and every decode step keeps device routing. A Q36 chunk has no page set,
//! so the Flash-Next rule that opens page sets once has no counterpart here.
//!
//! A failed prefill: a denied growth permit or a bad token in chunk k leaves chunks 1 to k - 1
//! published (the decoder rules), and the shared session has already taken its pending token. The
//! history and the state can then disagree, on the chunk path and on the one-token path alike. The
//! executor does not reset after an error. The caller must reset the session before it reuses it,
//! and the engine worker (`sched/worker.rs`) is that caller: it resets after every `prefill` error
//! that is not `Backend`. A `Backend` error ends the engine (the worker makes no further call), so
//! no native call follows it.
//!
//! Prefix reuse: `save_prefix` snapshots the state after the held prompt, and `restore_prefix`
//! gives it back after a reset. The executor keeps its entries in a `PrefixStore`, declared first
//! so its arrays drop before the decoder does. The store takes its limits from
//! `Q36Rules::prefix_store`: an engine started with `prefix_reuse` keeps the reuse set (two pinned
//! roots, six LRU entries), and every other engine keeps the tip store (zero roots, one entry), so
//! the worker saves the tip only there. Each entry carries the `PrefillPath` that built it, so an
//! entry restores only under its own path: a restore under `Chunked(n)` matches a full recompute
//! only within the chunk-parity tolerance, and only the `Steps` path is bitwise.
//! `DecodeSession::save_prefix` and `DecodeSession::restore_prefix` own the rules, and the executor
//! passes the two decoder calls `snapshot` and `restore`. A `MemoryPlan` from the save evicts
//! entries, retries, and reports `false` when no victim is left; a `MemoryPlan` from the restore
//! drops that entry and tries the next candidate.

use super::attn::growth_bytes;
use super::chunk_plan::{ChunkCost, PREFILL_CANDIDATES, PrefillPlan, PrefillTerms, admitted_chunk};
use super::decoder::{MAX_CHUNK_ROWS, ResidentDecoder, StepMode};
use super::state::{ResidentSnapshot, ResidentState};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        qwen36::{
            host::HostShape,
            host_load::HostWeights,
            settings::{Activation, KvDtype, chunked_prefill_allowed, shape_of},
        },
        session::{
            DecodePreparation, DecodeSession, Limits, PrefixSlot, SaveKind, ShadowLcp, StoreLimits,
            TokenStepper, check_prompt, sample_logits, shadow_block_lcp_enabled,
        },
        spec::{AttentionKind, FamilySpec, QWEN3_6_35B_A3B},
    },
    kernels::MlxKernels,
    sched::job::{PrefixHit, PrefixReport},
    sched::{GrowthPermit, SessionBudget},
    tokenizer::DecodeBatch,
    weights::{ShardIo, StreamedCheckpoint},
};
use std::{collections::BTreeSet, path::Path};

// The largest planned chunk is the most rows that the decoder accepts. The build fails if they differ.
const _: () = assert!(PREFILL_CANDIDATES[0] == MAX_CHUNK_ROWS);

type Result<T> = std::result::Result<T, EngError>;

/// The limits that the caller supplies.
pub(crate) struct Q36Rules {
    /// Every id that ends generation (`TokenizerBundle::eos_token_ids`: 248046 and 248044).
    pub(crate) eos: Vec<u32>,
    /// The number of logits that the sampler may choose from (`TokenizerBundle::vocab_size`). The
    /// model head can be wider than the tokenizer, and the extra ids have no text.
    /// `Q36Executor::from_parts` rejects 0 and any value above the width of the model head.
    pub(crate) sampler_vocab: usize,
    /// The first position that the decoder refuses. At least 2.
    pub(crate) context_limit: u32,
    /// The bytes of cache growth that this engine admits: the cap of the decoder's `SessionBudget`.
    pub(crate) session_cap: u64,
    /// The dtype of the decode activations, which also picks the scale storage at construction. The
    /// config member defaults to `bf16`.
    pub(crate) activation: Activation,
    /// The form of the attention KV cache: `F32`, the control, or `Q8`, the 8-bit codes with
    /// activation-dtype scales and biases. The config member `kv_dtype` defaults to `f32`.
    pub(crate) kv_dtype: KvDtype,
    /// `mlx_compile` on the DeltaNet and device-routed MoE decode blocks: the `qwen36.compile`
    /// member, default false. When true, the decoder warms both blocks once at load, so a trace
    /// failure fails the load instead of a request.
    pub(crate) compile: bool,
    /// The admitted concurrent sessions. One preserves the legacy solo lifecycle.
    pub(crate) max_seqs: usize,
    /// The limits of this executor's prefix store: the start maps the `prefix_reuse` flag to two
    /// pinned roots and six LRU entries - or the operator's exact `prefix_store_entries` override -
    /// and every other caller keeps the tip store (zero roots, one entry), so the existing tests
    /// keep their meaning. `from_parts` builds the store with these limits and `prefix_limits`
    /// reports them.
    pub(crate) prefix_store: StoreLimits,
}

/// The numeric path of a prefill. A saved prefix restores only under the path that built it: a
/// restore under `Chunked(n)` matches a full recompute only within the chunk-parity tolerance, and
/// only the `Steps` path is bitwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillPath {
    /// One-token steps (the default).
    Steps,
    /// Chunks of this many rows (`plan_prefill`).
    Chunked(u32),
}

impl PrefillPath {
    /// The path of a planned chunk size. A size of one is a one-token step, so it gives `Steps`.
    /// This is the one place that decides the path: `feed_rows` and `prefill_path` both use it.
    fn of(chunk: Option<u32>) -> Self {
        chunk
            .filter(|&size| size > 1)
            .map_or(Self::Steps, Self::Chunked)
    }
}

/// The executor of one Q36 model. It lives on the engine thread.
pub(crate) struct Q36Executor {
    /// The saved prefix. It is declared first, so its arrays drop before the decoder does: a
    /// Flash-parity invariant that no test can observe on this build, because every resident array
    /// shares the retire core, which holds the process lease and outlives the decoder.
    prefix: PrefixSlot<ResidentSnapshot, PrefillPath>,
    decoder: ResidentDecoder<'static>,
    eos: Vec<u32>,
    sampler_vocab: usize,
    context_limit: usize,
    max_seqs: usize,
    fixed_state_bytes: u64,
    /// The chunk size of the batched prefill, set only by `apply_chunk`. `None` means one-token steps.
    chunk: Option<u32>,
    /// The kind of the entry that the last `restore_prefix` call restored, for `prefix_report`.
    /// `Miss` until a restore takes an entry, and a reset clears it again, so the report of a job
    /// without a boundary is a miss.
    prefix_hit: PrefixHit,
    /// The `uncached_shared` count of the last `restore_prefix` call, for `prefix_report`;
    /// a reset clears it too.
    prefix_uncached: u32,
    /// The shadow block-LCP observation of the last `restore_prefix` call,
    /// for `prefix_report`; a reset clears it too.
    prefix_shadow: Option<ShadowLcp>,
    /// Tests only: the row count of each chunk that ran through `ResidentDecoder::prefill`, in
    /// order. It is shared so that an engine thread test can read it.
    #[cfg(test)]
    log: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
}

/// The mutable state of one session.
pub(crate) struct Q36Session {
    state: ResidentState,
    /// The committed tokens, the pending token, and the finish flag.
    core: DecodeSession,
    /// Opt-in live Delta state shares the aggregate budget with KV growth and snapshots.
    _fixed_state: Option<GrowthPermit>,
}

/// The model side of one call: the decoder, the native state of the session, and the chunk size
/// of the batched prefill. `Q36Executor::split` builds one value per call.
struct Q36Step<'a> {
    decoder: &'a mut ResidentDecoder<'static>,
    state: &'a mut ResidentState,
    /// The chunk size that `plan_prefill` stored, or `None` for one-token steps.
    chunk: Option<u32>,
    /// Tests only: the chunk log of the executor.
    #[cfg(test)]
    log: &'a std::sync::Mutex<Vec<usize>>,
}

impl TokenStepper for Q36Step<'_> {
    const OPERATION: &'static str = "qwen36";

    /// Feeds one prompt token. No head runs and nothing is read back.
    fn feed(&mut self, token: u32) -> Result<()> {
        self.decoder
            .step(self.state, token, StepMode::Prompt)
            .map(|_| ())
    }

    /// Feeds `tokens` as `Prompt` rows. With a planned chunk size, each run of that many rows is
    /// one chunk, and a single row is a one-token step. Without one, every row is a step.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let PrefillPath::Chunked(size) = PrefillPath::of(self.chunk) else {
            return tokens.iter().try_for_each(|&token| self.feed(token));
        };
        let width = usize::try_from(size)
            .map_err(|_| EngError::MemoryPlan("the chunk size exceeds the platform size".into()))?;
        for rows in tokens.chunks(width) {
            if let &[token] = rows {
                self.feed(token)?;
            } else {
                self.decoder.prefill(self.state, rows, StepMode::Prompt)?;
                #[cfg(test)]
                {
                    self.log.lock().unwrap().push(rows.len());
                }
            }
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decoder
            .step(self.state, token, StepMode::Logits)?
            .logits
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".to_string(),
            })
    }

    fn fed(&self) -> usize {
        fed_positions(self.state)
    }

    fn reset(&mut self) -> Result<()> {
        self.decoder.reset(self.state)
    }
}

/// The number of positions that the native state has fed.
fn fed_positions(state: &ResidentState) -> usize {
    usize::try_from(state.pos()).unwrap_or(usize::MAX)
}

fn window_output_error(message: &str) -> EngError {
    EngError::Backend {
        operation: "qwen36",
        message: message.into(),
    }
}

fn io_error(name: &str, error: &std::io::Error) -> EngError {
    EngError::Checkpoint(format!("{name}: {error}"))
}

/// Opens every shard of the checkpoint directory that the index names.
fn open_checkpoint(dir: &Path) -> Result<StreamedCheckpoint> {
    let name = "model.safetensors.index.json";
    let index = std::fs::read(dir.join(name)).map_err(|error| io_error(name, &error))?;
    let value: serde_json::Value = serde_json::from_slice(&index)
        .map_err(|error| EngError::Checkpoint(format!("{name}: {error}")))?;
    let names: BTreeSet<String> = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| EngError::Checkpoint(format!("{name} has no weight_map")))?
        .values()
        .filter_map(|shard| shard.as_str().map(str::to_string))
        .collect();
    let mut shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = Vec::with_capacity(names.len());
    for shard in names {
        let file =
            std::fs::File::open(dir.join(&shard)).map_err(|error| io_error(&shard, &error))?;
        let len = file
            .metadata()
            .map_err(|error| io_error(&shard, &error))?
            .len();
        shards.push((shard, Box::new(file) as Box<dyn ShardIo + Send>, len));
    }
    StreamedCheckpoint::open(&index, shards, 64 << 20)
}

/// The cache bytes that one position adds over all attention layers, in the cache form `kv` with
/// scales of the activation dtype `act`. The admission rule calls it with the rules' own form and
/// activation, so a Q8 engine admits its sessions by the 8-bit bytes (4,096 * layers F32, 1,088 *
/// layers at Q8/BF16 on the real model).
fn growth_per_position(shape: &HostShape, kv: KvDtype, act: Activation) -> Result<u64> {
    let layers = shape
        .kinds
        .iter()
        .filter(|kind| **kind == AttentionKind::Full)
        .count();
    let per_layer = growth_bytes(shape, kv, act)?;
    u64::try_from(layers)
        .ok()
        .and_then(|layers| per_layer.checked_mul(layers))
        .ok_or_else(|| EngError::MemoryPlan("cache growth bytes overflow u64".to_string()))
}

fn fixed_session_bytes(shape: &HostShape) -> Result<u64> {
    let delta = &shape.delta;
    let bytes = || {
        let values = delta.value_heads.checked_mul(delta.value_dim)?;
        let channels = delta
            .key_heads
            .checked_mul(delta.key_dim)?
            .checked_mul(2)?
            .checked_add(values)?;
        let cells = values
            .checked_mul(delta.key_dim)?
            .checked_add(channels.checked_mul(delta.conv_kernel.checked_sub(1)?)?)?;
        let layers = shape
            .kinds
            .iter()
            .filter(|kind| **kind == AttentionKind::DeltaNet)
            .count();
        u64::try_from(cells)
            .ok()?
            .checked_mul(4)?
            .checked_mul(u64::try_from(layers).ok()?)
    };
    bytes().ok_or_else(|| EngError::MemoryPlan("live DeltaNet state bytes overflow".into()))
}

/// Checks the rules against the model before any weight is imported.
fn admit(shape: &HostShape, rules: &Q36Rules) -> Result<()> {
    if shape.dense_inter.is_some() && rules.max_seqs > 1 {
        return Err(EngError::BadRequest(
            "the dense family runs one sequence at a time; batched rows require MoE device routing"
                .into(),
        ));
    }
    if rules.sampler_vocab == 0 || rules.sampler_vocab > shape.vocab {
        return Err(EngError::Checkpoint(format!(
            "sampler vocabulary {} is outside 1..={}, the width of the model head",
            rules.sampler_vocab, shape.vocab
        )));
    }
    if rules.context_limit < 2 {
        return Err(EngError::BadRequest(format!(
            "context limit {} is below 2 positions",
            rules.context_limit
        )));
    }
    let needed = growth_per_position(shape, rules.kv_dtype, rules.activation)?
        .checked_mul(u64::from(rules.context_limit))
        .ok_or_else(|| EngError::MemoryPlan("cache bytes at the limit overflow u64".to_string()))?;
    if needed > rules.session_cap {
        return Err(EngError::MemoryPlan(format!(
            "context limit {} needs {needed} bytes of cache; the session cap is {} bytes",
            rules.context_limit, rules.session_cap
        )));
    }
    if !(1..=8).contains(&rules.max_seqs) || (rules.max_seqs > 1 && !rules.compile) {
        return Err(EngError::BadRequest(
            "Qwen decode windows need max_seqs in 1..=8 and compile on above one".into(),
        ));
    }
    if rules.max_seqs > 1 {
        let required = fixed_session_bytes(shape)?
            .checked_mul(u64::try_from(rules.max_seqs).unwrap_or(u64::MAX))
            .and_then(|fixed| fixed.checked_add(needed))
            .ok_or_else(|| EngError::MemoryPlan("concurrent session bytes overflow".into()))?;
        if required > rules.session_cap {
            return Err(EngError::MemoryPlan(format!(
                "concurrent live states and one full context require {required} bytes; aggregate cap is {}",
                rules.session_cap
            )));
        }
    }
    Ok(())
}

impl Q36Executor {
    /// Loads the artifact in `artifact`, imports the weights, and creates the session. Call it on
    /// the engine thread. `cache_limit` is the MLX allocator cache limit in bytes.
    pub(crate) fn open(
        artifact: &Path,
        cache_limit: u64,
        rules: Q36Rules,
    ) -> Result<(Self, Q36Session)> {
        Self::open_for_family(artifact, cache_limit, rules, &QWEN3_6_35B_A3B)
    }

    /// Loads using the descriptor selected by the server, for both admission and weight loading.
    pub(crate) fn open_for_family(
        artifact: &Path,
        cache_limit: u64,
        rules: Q36Rules,
        spec: &FamilySpec,
    ) -> Result<(Self, Q36Session)> {
        let config = std::fs::read(artifact.join("config.json"))
            .map_err(|error| io_error("config.json", &error))?;
        // The rules are checked against the shape from `config.json` before any shard opens, so a
        // refused rule costs no load (the project rules: enforce the memory plan at engine load).
        admit(&shape_of(&config, spec)?, &rules)?;
        let ckpt: &'static StreamedCheckpoint = Box::leak(Box::new(open_checkpoint(artifact)?));
        let host = HostWeights::load(ckpt, spec, &config)?;
        let mut kernels = MlxKernels::try_new()?;
        kernels.set_cache_limit(cache_limit)?;
        Self::from_parts(kernels, ckpt, host, rules)
    }

    /// Builds the executor from loaded parts. The rules are checked again before the import, so a
    /// caller that skips `open` cannot skip them.
    pub(crate) fn from_parts(
        kernels: MlxKernels,
        ckpt: &'static StreamedCheckpoint,
        host: HostWeights,
        rules: Q36Rules,
    ) -> Result<(Self, Q36Session)> {
        admit(host.shape(), &rules)?;
        let fixed_state_bytes = if rules.max_seqs > 1 {
            fixed_session_bytes(host.shape())?
        } else {
            0
        };
        let context_limit = usize::try_from(rules.context_limit)
            .map_err(|_| EngError::BadRequest("context limit overflows".to_string()))?;
        let mut decoder = ResidentDecoder::new_with_kv(
            kernels,
            ckpt,
            host,
            rules.context_limit,
            rules.activation,
            rules.kv_dtype,
        )?;
        // The compile seam admits the whole model and warms one layer per distinct program HERE,
        // before the first state exists: a plan refusal or a trace failure is a typed load error,
        // never a failure in the middle of a request. The warm-up seconds of THIS engine land in
        // the decoder.
        if rules.compile {
            decoder.set_compile(true)?;
        }
        if rules.max_seqs > 1 {
            let sizes: Vec<usize> = [2, 4, 8]
                .into_iter()
                .filter(|&rows| rows <= rules.max_seqs)
                .collect();
            decoder.prepare_compiled_rows(&sizes)?;
        }
        // No state exists yet, so no permit charges the old budget.
        decoder.budget = SessionBudget::new(rules.session_cap);
        let fixed_state = if rules.max_seqs > 1 {
            Some(decoder.budget.acquire_growth(fixed_state_bytes)?)
        } else {
            None
        };
        let state = decoder.zero_state()?;
        let session = Q36Session {
            state,
            core: DecodeSession::default(),
            _fixed_state: fixed_state,
        };
        let mut executor = Self {
            prefix: PrefixSlot::with_limits(rules.prefix_store),
            decoder,
            eos: rules.eos,
            sampler_vocab: rules.sampler_vocab,
            context_limit,
            max_seqs: rules.max_seqs,
            fixed_state_bytes,
            chunk: None,
            prefix_hit: PrefixHit::Miss,
            prefix_uncached: 0,
            prefix_shadow: None,
            #[cfg(test)]
            log: std::sync::Arc::default(),
        };
        // The shadow block-LCP observation: off unless the environment
        // flag runs it; the observation changes no restore or save behavior.
        executor
            .prefix
            .set_shadow_enabled(shadow_block_lcp_enabled());
        Ok((executor, session))
    }

    /// The number of positions that the model accepts. A prompt of this many tokens or more is
    /// rejected with `ContextLimit`, and a job ends with `Length` when its next feed needs this
    /// position. The server uses it to check a prompt before admission.
    pub(crate) fn context_limit(&self) -> usize {
        self.context_limit
    }

    /// Reads the production counters of the compiled block calls of this executor's adapter: calls,
    /// the native trace counts, the cache hits, and the seconds of the first calls per block kind.
    /// The tests of the resident gate read them through the executor, which owns the adapter.
    pub(crate) fn compile_counters(&self) -> crate::kernels::mlx::compile_stats::Stats {
        self.decoder.kernels.compile_counters()
    }

    /// The seconds of THIS engine's load-time compile warm-up, delta first and moe second: the
    /// seconds per traced program of each block kind, empty vectors when `qwen36.compile` is off.
    /// They are measured inside this decoder's warm-up — not read from the process-wide counters —
    /// so the start line reports this engine alone and labels each block kind correctly.
    pub(crate) fn compile_warmup_seconds(&self) -> (&[f64], &[f64]) {
        (
            &self.decoder.compile_warmup.delta,
            &self.decoder.compile_warmup.moe,
        )
    }

    /// Splits the executor into the model side of one call and the limits of the engine. The two
    /// borrow different fields, so a caller holds both.
    fn split<'a>(&'a mut self, state: &'a mut ResidentState) -> (Q36Step<'a>, Limits<'a>) {
        let step = Q36Step {
            decoder: &mut self.decoder,
            state,
            chunk: self.chunk,
            #[cfg(test)]
            log: &self.log,
        };
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.sampler_vocab,
            context_limit: self.context_limit,
        };
        (step, limits)
    }

    /// The admission cost of one prefill chunk on this model (the admission rule in Q36 terms).
    fn prefill_cost(&self) -> Result<ChunkCost> {
        let heads_q = u64::try_from(self.decoder.weights.shape.attn.heads_q)
            .map_err(|_| EngError::MemoryPlan("the query head count exceeds u64".to_string()))?;
        ChunkCost::new(heads_q, self.decoder.context_limit)
    }

    /// Applies the chunk admission rule and stores the chunk size.
    ///
    /// `remaining = budget - resident weights - session_reserve`. The weights come from this
    /// executor. `session_reserve` is the sum that the caller owns: the session state at the
    /// context limit (the cache growth per position times the limit, plus the DeltaNet state), the
    /// MLX cache limit, and the pressure margin. A caller that passes 0 admits a larger chunk than
    /// the memory plan allows. Q36 has no decode pool and no expert cache, so no other term exists.
    /// `max_rows` is the largest chunk (`Q36Settings::prefill_chunk`): the chunk is never larger.
    /// The worker piece is the planned chunk, or 16 (`NO_BUDGET_PIECE`) when no chunk is planned.
    ///
    /// It returns the chunk size, or `None` when no size above one token fits and the prefill runs
    /// one-token steps, with the terms it charged. Call it once after construction and before the
    /// first job. `start::plan_prefill` calls it at engine start-up when the settings carry a
    /// budget. Both activation dtypes plan a chunk: the chunk path is dtype aware, so only the
    /// budget or `max_rows` can keep a chunk out.
    pub(crate) fn plan_prefill(
        &mut self,
        budget: u64,
        session_reserve: u64,
        max_rows: u32,
    ) -> Result<PrefillPlan> {
        let terms = PrefillTerms {
            budget,
            weights: self.decoder.weights.byte_len()?,
            session_reserve,
        };
        // `chunked_prefill_allowed` stays the one authority for the dtype rule: the chunk path is
        // dtype aware, so both activation dtypes reach `admitted_chunk`, and a dtype the rule does
        // not know still plans no chunk.
        let chunk = if chunked_prefill_allowed(self.decoder.weights.consts.act) {
            admitted_chunk(&self.prefill_cost()?, terms.remaining(), max_rows)?
        } else {
            None
        };
        self.apply_chunk(chunk)?;
        Ok(PrefillPlan { chunk, terms })
    }

    /// The numeric path that `feed_rows` takes now. A chunk size of one is a one-token step.
    fn prefill_path(&self) -> PrefillPath {
        PrefillPath::of(self.chunk)
    }

    /// The planned chunk size, or `None` when the prefill runs one-token steps.
    pub(crate) fn prefill_chunk(&self) -> Option<u32> {
        self.chunk
    }

    /// The only writer of the chunk size: it sets the decoder value first, so a size that the
    /// decoder refuses changes nothing.
    fn apply_chunk(&mut self, chunk: Option<u32>) -> Result<()> {
        if let Some(rows) = chunk {
            self.decoder.set_chunk_rows(rows)?;
        }
        self.chunk = chunk;
        Ok(())
    }

    /// Tests only: sets the chunk size without the admission rule, so that a speed run can sweep it.
    #[cfg(test)]
    fn force_chunk(&mut self, chunk: Option<u32>) {
        self.apply_chunk(chunk).unwrap();
    }
}

impl ExecutorOps for Q36Executor {
    type Session = Q36Session;

    fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    fn new_session(&mut self) -> Result<Q36Session> {
        let fixed_state = if self.max_seqs > 1 {
            Some(self.decoder.budget.acquire_growth(self.fixed_state_bytes)?)
        } else {
            None
        };
        Ok(Q36Session {
            state: self.decoder.zero_state()?,
            core: DecodeSession::default(),
            _fixed_state: fixed_state,
        })
    }

    fn decode_window(
        &mut self,
        sessions: &mut [&mut Q36Session],
        generations: &mut [&mut GenerationState],
    ) -> Result<Vec<Result<DecodeBatch>>> {
        if sessions.is_empty()
            || sessions.len() != generations.len()
            || sessions.len() > self.max_seqs
        {
            return Err(EngError::BadRequest(
                "Qwen decode window needs equal nonempty slices within max_seqs".into(),
            ));
        }
        if sessions.len() == 1 {
            return self
                .decode(sessions[0], generations[0])
                .map(|batch| vec![Ok(batch)]);
        }
        // Validate every input before cancellation or a decoder tail can mutate any row.
        for (session, generation) in sessions.iter().zip(generations.iter()) {
            let token = session.core.checked_pending()?;
            if generation.cancel.load(std::sync::atomic::Ordering::SeqCst)
                || generation.remaining_tokens() == Some(0)
            {
                continue;
            }
            session
                .state
                .check_step(&self.decoder.weights, self.decoder.context_limit)?;
            if token as usize >= self.decoder.weights.shape.vocab {
                return Err(EngError::BadRequest(
                    "token is outside the embedding table".into(),
                ));
            }
        }
        let mut results: Vec<Option<Result<DecodeBatch>>> =
            (0..sessions.len()).map(|_| None).collect();
        let mut active = Vec::new();
        for (slot, (session, generation)) in
            sessions.iter_mut().zip(generations.iter_mut()).enumerate()
        {
            let Q36Session { state, core, .. } = &mut **session;
            let (mut step, _) = self.split(state);
            match core.prepare_decode(&mut step, generation)? {
                DecodePreparation::Finished(batch) => results[slot] = Some(Ok(batch)),
                DecodePreparation::Token(_) => active.push(slot),
            }
        }
        let growth = super::state::kv_growth_bytes(&self.decoder.weights)?;
        let mut permits = Vec::new();
        let mut admitted = Vec::new();
        for slot in active {
            match self.decoder.budget.acquire_growth(growth) {
                Ok(permit) => {
                    permits.push(permit);
                    admitted.push(slot);
                }
                Err(error @ EngError::MemoryPlan(_)) => results[slot] = Some(Err(error)),
                Err(error) => return Err(error),
            }
        }
        // Select rows with cumulative charges; native decode recharges them on this engine thread.
        drop(permits);
        let active = admitted;
        if active.len() == 1 {
            let slot = active[0];
            results[slot] = Some(Ok(self.decode(sessions[slot], generations[slot])?));
        } else if !active.is_empty() {
            let tokens = active
                .iter()
                .map(|&slot| sessions[slot].core.checked_pending())
                .collect::<Result<Vec<_>>>()?;
            let fed_before: Vec<_> = active
                .iter()
                .map(|&slot| fed_positions(&sessions[slot].state))
                .collect();
            let mut states: Vec<_> = sessions
                .iter_mut()
                .enumerate()
                .filter_map(|(slot, session)| active.contains(&slot).then_some(&mut session.state))
                .collect();
            // Device argmax has no final-logit finite check; use the solo sampler policy.
            let stepped = self
                .decoder
                .decode_rows(&mut states, &tokens, StepMode::Logits);
            drop(states);
            // Record every consumed input before any result validation or sampler can fail.
            // The worker discards the entire window on error and never publishes partial rows.
            for (&slot, &before) in active.iter().zip(&fed_before) {
                let after = fed_positions(&sessions[slot].state);
                sessions[slot].core.record_feed(before, after);
            }
            let stepped = stepped?;
            if stepped.len() != active.len() {
                return Err(window_output_error(
                    "the decoder returned the wrong row count",
                ));
            }
            let mut sampled = Vec::with_capacity(active.len());
            for (&slot, row) in active.iter().zip(stepped) {
                let logits = row
                    .logits
                    .ok_or_else(|| window_output_error("the row returned no logits"))?;
                let token = sample_logits(
                    Q36Step::OPERATION,
                    &logits,
                    &mut generations[slot].sampler,
                    self.sampler_vocab,
                )?;
                if usize::try_from(token).map_or(true, |token| token >= self.sampler_vocab) {
                    return Err(window_output_error(
                        "the row returned a token outside the sampler vocabulary",
                    ));
                }
                sampled.push(token);
            }
            for (&slot, token) in active.iter().zip(sampled) {
                let Q36Session { state, core, .. } = &mut *sessions[slot];
                let (mut step, limits) = self.split(state);
                results[slot] = Some(Ok(core.finish_decode_token(
                    &mut step,
                    generations[slot],
                    &limits,
                    token,
                )?));
            }
        }
        results
            .into_iter()
            .map(|row| row.ok_or_else(|| window_output_error("the window omitted a result row")))
            .collect()
    }

    /// Refuses a prompt that cannot fit before any state changes. The limit is the same number
    /// that `prefill` enforces chunk by chunk (`Limits::context_limit`), so both checks refuse
    /// the same prompts.
    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        check_prompt(prompt_tokens, self.context_limit())
    }

    fn history<'s>(&self, st: &'s Q36Session) -> &'s [u32] {
        st.core.history()
    }

    fn prefill_piece(&self) -> Option<usize> {
        self.chunk.and_then(|rows| usize::try_from(rows).ok())
    }

    /// The limits of this executor's prefix store: the store this executor was built with
    /// (`Q36Rules::prefix_store` through `from_parts`), so a `prefix_reuse` engine reports two
    /// pinned roots and six LRU entries - or the operator's exact `prefix_store_entries` override.
    fn prefix_limits(&self) -> StoreLimits {
        self.prefix.limits()
    }

    /// The prefix-store report of this executor: the entries the store holds after the job's save
    /// points and their charged bytes — each `ResidentSnapshot` holds its own growth permit, so the
    /// executor sums `charged_bytes` over the stored states; the store never sizes a snapshot —
    /// beside the kind of the entry the job restored and the prompt tokens shared with a stored key
    /// beyond the restored prefix, both taken at the restore.
    fn prefix_report(&self) -> Option<PrefixReport> {
        let stored = self.prefix.roots_len() + self.prefix.entries_len();
        let entries = u32::try_from(stored).unwrap_or(u32::MAX);
        Some(PrefixReport {
            entries,
            bytes: self
                .prefix
                .snapshots()
                .map(|snapshot| snapshot.charged_bytes())
                .sum(),
            hit: self.prefix_hit,
            uncached_shared: self.prefix_uncached,
            shadow: self.prefix_shadow,
            restore_drops: self.prefix.restore_drop_count(),
            evictions: self.prefix.eviction_count(),
            save_side: self.prefix.save_side_totals(),
        })
    }

    fn reset(&mut self, st: &mut Q36Session) -> Result<()> {
        let Q36Session { state, core, .. } = st;
        let (mut step, _) = self.split(state);
        core.reset(&mut step)?;
        // The restore facts of the report do not survive a job: the worker resets every job
        // and skips the restore when the job has no boundary, so the report of such a job
        // must not carry the hit of the job before it.
        self.prefix_hit = PrefixHit::Miss;
        self.prefix_uncached = 0;
        self.prefix_shadow = None;
        Ok(())
    }

    /// Feeds the prompt rows of one piece. After any error the caller must reset the session
    /// before it reuses it: earlier chunks can stay published and the shared session can have
    /// taken its pending token, so the history and the state can disagree. The engine worker owns
    /// this recovery (`sched/worker.rs`). A `Backend` error is final, and no call follows it.
    fn prefill(&mut self, st: &mut Q36Session, toks: &[u32]) -> Result<()> {
        let Q36Session { state, core, .. } = st;
        let (mut step, limits) = self.split(state);
        core.prefill(&mut step, toks, limits.context_limit)
    }

    fn decode(
        &mut self,
        st: &mut Q36Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let Q36Session { state, core, .. } = st;
        let (mut step, limits) = self.split(state);
        core.decode(&mut step, generation, &limits)
    }

    /// No transient routing or draft state exists: expert routing is computed inside each step.
    fn boundary(&mut self, _st: &mut Q36Session) {}

    /// Shares the resident state as an entry of this executor's prefix store, saved as `kind`
    /// (a pinned root at the first system message or a tip at the last boundary; the worker
    /// passes the kind). The session rules decide whether the call is valid. The snapshot
    /// charges the session budget.
    fn save_prefix(&mut self, st: &Q36Session, key: &[u32], kind: SaveKind) -> Result<bool> {
        let path = self.prefill_path();
        let decoder = &mut self.decoder;
        let fed = fed_positions(&st.state);
        st.core
            .save_prefix(&mut self.prefix, key, path, fed, kind, || {
                decoder.snapshot(&st.state)
            })
    }

    /// Restores the saved prefix into the reset session when the session rules accept the prompt.
    fn restore_prefix(&mut self, st: &mut Q36Session, prompt: &[u32]) -> Result<usize> {
        let path = self.prefill_path();
        let decoder = &mut self.decoder;
        let Q36Session { state, core, .. } = st;
        let covered = core.restore_prefix(&mut self.prefix, prompt, path, |snapshot| {
            decoder.restore(state, snapshot)
        })?;
        // The restore facts of the report, read now: the store moves on with the job's saves,
        // and the report is taken only after the prefill.
        self.prefix_hit = self.prefix.restored_kind().into();
        self.prefix_uncached = self.prefix.uncached_shared(prompt, covered);
        self.prefix_shadow = self.prefix.shadow_report();
        Ok(covered)
    }

    /// Evicts ONE LRU entry of the engine's store per the incremental denial order of the radix
    /// compression: dead tips first (they never match the next turn), then redundant single-chain
    /// interior anchors, then anchor leaves, then branch points (a shared header), oldest within
    /// each tier, with the denied job's restored anchor (`protect`) last - so one denial cannot
    /// evict every live lineage - and reports the bytes its snapshot released from the session
    /// budget: a dropped snapshot's growth permit releases its charge, so a dead or idle
    /// conversation stops pinning the pool the moment the worker asks. `None` when no LRU entry
    /// remains (roots stay pinned), so an eviction loop that reaches `None` has freed everything it
    /// can and the denial stands for classification.
    fn evict_one_prefix_entry(&mut self, protect: Option<u64>) -> Option<u64> {
        let before = self.decoder.budget.used();
        if !self.prefix.evict_lru_one(protect) {
            return None;
        }
        Some(before.saturating_sub(self.decoder.budget.used()))
    }

    /// The denying session's restored anchor: the core session tracks it, so the eviction
    /// protects the conversation the denial belongs to, not whichever restored last.
    fn restored_anchor_of(&self, st: &Q36Session) -> Option<u64> {
        st.core.restored_anchor()
    }

    /// The first entry a restart of `prompt` would restore: the path is the executor's
    /// own prefill path, so it is known before any session exists.
    fn restore_candidate(&self, prompt: &[u32]) -> Option<u64> {
        self.prefix.restore_candidate(prompt, &self.prefill_path())
    }

    /// The pool-derived prompt ceiling: the free budget plus the session's own permits - which its
    /// next reset releases - over the per-position KV growth, minus the one position the first
    /// sampled token needs.
    ///
    /// shortcut: permits that a still-stored snapshot shares with the session (a pinned
    /// root this session saved) are counted as releasable, so the ceiling can overstate by
    /// that root's bytes; upgrade when a client compacting to exactly the ceiling still
    /// fails in practice, by tracking per-session shared charges.
    fn pool_prompt_ceiling(&self, st: &Q36Session) -> Option<u64> {
        let growth = super::state::kv_growth_bytes(&self.decoder.weights).ok()?;
        let session = st.state.charged_bytes().ok()?;
        Some((self.decoder.budget.free() + session).saturating_sub(growth) / growth.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::super::ops::assert_parity;
    use super::super::real::{activation_from_env, env_path, kv_dtype_from_env};
    use super::*;
    use crate::family::qwen36::settings::activation_name;
    use crate::family::qwen36::tiny::{self, TINY_GQA, TINY_KV8};
    use crate::family::session::SaveSideTotals;
    use crate::{
        family::sampler::{Sampler, SamplingParams},
        tokenizer::{DecodeFinish, IncrementalDecoder, Tokenizer},
    };

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

    /// The synthetic model `TINY_GQA`: vocabulary 8, four layers, two of them attention.
    const VOCAB: usize = 8;
    /// The context limit of the synthetic executor: positions 0 to 7 are fed.
    const LIMIT: u32 = 8;
    /// Cache bytes per position: 2 attention layers * 2 key-value heads * 32 * 2 (keys and values) * 4.
    const GROWTH: u64 = 2 * 2 * 32 * 2 * 4;
    const PROMPT: [u32; 3] = [1, 2, 3];

    /// The rules of the synthetic executor: the cap admits exactly `LIMIT` positions.
    fn rules(eos: Vec<u32>, sampler_vocab: usize) -> Q36Rules {
        Q36Rules {
            prefix_store: StoreLimits::default(),
            eos,
            sampler_vocab,
            context_limit: LIMIT,
            session_cap: u64::from(LIMIT) * GROWTH,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        }
    }

    /// The cache form of the rules reaches the decoder (the hop `from_parts` to `new_with_kv`),
    /// and the admission counts the bytes of that form. The tiny KV8 model has two attention
    /// layers of 2 key-value heads of width 64. With F32 scales one layer and position takes, per
    /// side, 2 * 64 code bytes plus 2 arrays * 2 heads * 1 group * 4 bytes = 144, so 288 for keys
    /// and values and 576 over both layers. The F32 form takes 2 * 64 * 2 * 4 = 1,024 per layer
    /// and 2,048 over both. A cap of exactly `LIMIT` positions of the Q8 bytes admits the Q8
    /// engine and refuses the F32 engine; one byte less refuses the Q8 engine.
    #[test]
    fn the_cache_form_of_the_rules_reaches_the_decoder_and_the_admission() {
        const Q8_GROWTH: u64 = 2 * 288;
        let _serial = crate::native_test_lock();
        let ckpt: &'static StreamedCheckpoint = Box::leak(Box::new(tiny::checkpoint(&TINY_KV8)));
        let host = || HostWeights::load(ckpt, &TINY_KV8, &tiny::config(&TINY_KV8)).unwrap();
        let with = |cap: u64, kv: KvDtype| Q36Rules {
            prefix_store: StoreLimits::default(),
            session_cap: cap,
            kv_dtype: kv,
            compile: false,
            ..rules(vec![], VOCAB)
        };
        let cap = u64::from(LIMIT) * Q8_GROWTH;
        let build = |rules: Q36Rules| {
            Q36Executor::from_parts(MlxKernels::try_new().unwrap(), ckpt, host(), rules)
        };
        let (executor, session) = build(with(cap, KvDtype::Q8)).unwrap();
        assert_eq!(
            executor.decoder.weights.kv,
            KvDtype::Q8,
            "the Q8 rules build a Q8 decoder"
        );
        drop(session);
        drop(executor);
        // BF16 scales take 2 * 272 bytes per position over both layers: the same rules in BF16
        // activation admit exactly that cap and refuse one byte less.
        let bf16 = |cap: u64| Q36Rules {
            prefix_store: StoreLimits::default(),
            activation: Activation::Bf16,
            ..with(cap, KvDtype::Q8)
        };
        let bf16_cap = u64::from(LIMIT) * 2 * 272;
        let (executor, session) = build(bf16(bf16_cap)).unwrap();
        assert_eq!(
            executor.decoder.weights.consts.act,
            crate::tensor::Dtype::Bf16,
            "the BF16 rules build a BF16 decoder"
        );
        drop(session);
        drop(executor);
        let refused = build(bf16(bf16_cap - 1)).err();
        assert!(
            matches!(&refused, Some(EngError::MemoryPlan(text)) if text.contains("needs")),
            "{refused:?}"
        );
        for (kv, cap) in [(KvDtype::F32, cap), (KvDtype::Q8, cap - 1)] {
            let refused = build(with(cap, kv)).err();
            assert!(
                matches!(&refused, Some(EngError::MemoryPlan(text)) if text.contains("needs")),
                "{kv:?} at {cap} bytes: {refused:?}"
            );
        }
        // A Q8 engine over a model whose head dimension is not a multiple of 64 is a BadRequest
        // that names the setting, not a cache-byte overflow.
        let gqa: &'static StreamedCheckpoint = Box::leak(Box::new(tiny::checkpoint(&TINY_GQA)));
        let refused = Q36Executor::from_parts(
            MlxKernels::try_new().unwrap(),
            gqa,
            HostWeights::load(gqa, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap(),
            with(u64::MAX, KvDtype::Q8),
        )
        .err();
        assert!(
            matches!(&refused, Some(EngError::BadRequest(text)) if text.contains("kv_dtype q8")),
            "{refused:?}"
        );
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap()
    }

    fn host(ckpt: &StreamedCheckpoint) -> HostWeights {
        HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap()
    }

    /// Builds the executor on the synthetic checkpoint. The checkpoint is leaked, as in `open`.
    fn build(rules: Q36Rules) -> (Q36Executor, Q36Session) {
        let ckpt: &'static StreamedCheckpoint = Box::leak(Box::new(tiny::checkpoint(&TINY_GQA)));
        Q36Executor::from_parts(MlxKernels::try_new().unwrap(), ckpt, host(ckpt), rules).unwrap()
    }

    /// Temperature 1 with a fixed seed. The sampled token depends on the whole logits row, so a
    /// decoder state that differs changes the draws.
    fn seeded_sampler() -> Sampler {
        Sampler::new(SamplingParams::new(1.0, 1.0, 0).unwrap(), 7)
    }

    fn seeded_generation(max_tokens: u32) -> GenerationState {
        GenerationState::with_sampler(IncrementalDecoder::new(&tokenizer()), seeded_sampler())
            .with_max_tokens(max_tokens)
    }

    #[test]
    fn a_prepared_window_matches_independent_solo_jobs_and_sampler_vocabularies() {
        let _serial = crate::native_test_lock();
        for activation in [Activation::F32, Activation::Bf16] {
            for seeded in [false, true] {
                for sampler_vocab in [VOCAB, VOCAB - 1] {
                    let (mut executor, first) = build(Q36Rules {
                        compile: true,
                        max_seqs: 4,
                        session_cap: 4 << 20,
                        activation,
                        ..rules(vec![], sampler_vocab)
                    });
                    let mut rows = [first, executor.new_session().unwrap()];
                    let mut solo = [
                        executor.new_session().unwrap(),
                        executor.new_session().unwrap(),
                    ];
                    let make_jobs = || {
                        (0..2)
                            .map(|slot| {
                                let sampler = if seeded {
                                    Sampler::new(
                                        SamplingParams::new(1.0, 1.0, 0).unwrap(),
                                        7 + slot,
                                    )
                                } else {
                                    Sampler::greedy()
                                };
                                GenerationState::with_sampler(
                                    IncrementalDecoder::new(&tokenizer()),
                                    sampler,
                                )
                                .with_max_tokens(3)
                            })
                            .collect::<Vec<_>>()
                    };
                    let (mut row_jobs, mut solo_jobs) = (make_jobs(), make_jobs());
                    for (slot, prompt) in [&PROMPT[..], &PROMPT[..2]].into_iter().enumerate() {
                        executor.prefill(&mut rows[slot], prompt).unwrap();
                        executor.prefill(&mut solo[slot], prompt).unwrap();
                    }
                    for _ in 0..3 {
                        let expected: Vec<_> = solo
                            .iter_mut()
                            .zip(&mut solo_jobs)
                            .map(|(state, job)| executor.decode(state, job).unwrap())
                            .collect();
                        let before = executor.compile_counters();
                        let actual = executor
                            .decode_window(
                                &mut rows.iter_mut().collect::<Vec<_>>(),
                                &mut row_jobs.iter_mut().collect::<Vec<_>>(),
                            )
                            .unwrap()
                            .into_iter()
                            .collect::<Result<Vec<_>>>()
                            .unwrap();
                        let after = executor.compile_counters();
                        assert_eq!(
                            actual, expected,
                            "{activation:?}, seeded={seeded}, vocab={sampler_vocab}"
                        );
                        assert_eq!(after.delta.calls - before.delta.calls, 2);
                        assert_eq!(after.moe.calls - before.moe.calls, 4);
                        assert_eq!(after.delta.traces, before.delta.traces);
                        assert_eq!(after.moe.traces, before.moe.traces);
                        for slot in 0..2 {
                            assert_eq!(rows[slot].core.history(), solo[slot].core.history());
                            assert_eq!(rows[slot].core.pending(), solo[slot].core.pending());
                            assert_eq!(rows[slot].state.pos(), solo[slot].state.pos());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn concurrent_state_budget_covers_initial_extra_reset_and_drop() {
        const FIXED: u64 = 19_456;
        let _serial = crate::native_test_lock();
        let ckpt = Box::leak(Box::new(tiny::checkpoint(&TINY_GQA)));
        let host = HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
        assert_eq!(fixed_session_bytes(host.shape()).unwrap(), FIXED);
        let mut admitted = Q36Rules {
            compile: true,
            max_seqs: 2,
            session_cap: 2 * FIXED + u64::from(LIMIT) * GROWTH - 1,
            ..rules(vec![], VOCAB)
        };
        assert!(matches!(
            admit(host.shape(), &admitted),
            Err(EngError::MemoryPlan(_))
        ));
        admitted.session_cap += 1;
        let (mut executor, mut first) =
            Q36Executor::from_parts(MlxKernels::try_new().unwrap(), ckpt, host, admitted).unwrap();
        assert_eq!(executor.decoder.session_used(), FIXED);
        let second = executor.new_session().unwrap();
        assert_eq!(executor.decoder.session_used(), 2 * FIXED);
        assert!(matches!(
            executor.new_session(),
            Err(EngError::MemoryPlan(_))
        ));
        assert_eq!(executor.decoder.session_used(), 2 * FIXED);
        drop(second);
        assert_eq!(executor.decoder.session_used(), FIXED);
        executor.reset(&mut first).unwrap();
        assert_eq!(executor.decoder.session_used(), FIXED);
        let replacement = executor.new_session().unwrap();
        assert_eq!(executor.decoder.session_used(), 2 * FIXED);
        drop(replacement);
        drop(first);
        assert_eq!(executor.decoder.session_used(), 0);
    }

    #[test]
    fn a_window_refuses_bad_shapes_before_feeding_and_filters_finished_rows() {
        use std::sync::atomic::Ordering;
        let _serial = crate::native_test_lock();
        let (mut executor, mut first) = build(Q36Rules {
            compile: true,
            max_seqs: 2,
            session_cap: 4 << 20,
            ..rules(vec![], VOCAB)
        });
        let mut second = executor.new_session().unwrap();
        executor.prefill(&mut first, &PROMPT).unwrap();
        executor.prefill(&mut second, &PROMPT[..2]).unwrap();
        let (mut a, mut b) = (seeded_generation(0), seeded_generation(0));
        a.cancel.store(true, Ordering::SeqCst);
        let before = executor.compile_counters();
        assert!(matches!(
            executor.decode_window(&mut [], &mut []),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            executor.decode_window(&mut [&mut first, &mut second], &mut [&mut a]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(first.core.pending(), Some(3));
        assert_eq!(second.core.pending(), Some(2));
        let result = executor
            .decode_window(&mut [&mut first, &mut second], &mut [&mut a, &mut b])
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(result[0].finish, Some(DecodeFinish::Cancelled));
        assert_eq!(result[1].finish, Some(DecodeFinish::Length));
        assert!(result.iter().all(|row| row.tokens.is_empty()));
        assert!(first.core.history().is_empty());
        assert!(second.core.is_finished());
        let after = executor.compile_counters();
        assert_eq!(after.delta.calls, before.delta.calls);
        assert_eq!(after.moe.calls, before.moe.calls);
    }

    #[test]
    fn a_bad_pending_window_row_returns_no_results_and_consumes_no_input() {
        use std::sync::atomic::Ordering;
        let _serial = crate::native_test_lock();
        for cancelled in [false, true] {
            let (mut executor, mut first) = build(Q36Rules {
                compile: true,
                max_seqs: 2,
                session_cap: 4 << 20,
                ..rules(vec![], VOCAB)
            });
            let mut second = executor.new_session().unwrap();
            executor.prefill(&mut first, &PROMPT).unwrap();
            // A one-token prompt holds the bad id pending without feeding it.
            executor.prefill(&mut second, &[VOCAB as u32]).unwrap();
            let positions = (first.state.pos(), second.state.pos());
            let (mut a, mut b) = (seeded_generation(3), seeded_generation(3));
            a.cancel.store(cancelled, Ordering::SeqCst);
            assert!(matches!(
                executor.decode_window(&mut [&mut first, &mut second], &mut [&mut a, &mut b]),
                Err(EngError::BadRequest(_))
            ));
            assert_eq!((first.state.pos(), second.state.pos()), positions);
            assert_eq!(first.core.pending(), Some(3));
            assert_eq!(second.core.pending(), Some(VOCAB as u32));
            assert_eq!(first.core.history(), PROMPT);
            assert_eq!(second.core.history(), [VOCAB as u32]);
            assert!(!first.core.is_finished());
            assert_eq!(a.remaining_tokens(), Some(3));
        }
    }

    #[test]
    fn greedy_windows_reject_non_finite_final_logits_like_solo() {
        let _serial = crate::native_test_lock();
        for non_finite in [f32::NAN, f32::INFINITY] {
            let (mut executor, mut first) = build(Q36Rules {
                compile: true,
                max_seqs: 3,
                session_cap: 4 << 20,
                ..rules(vec![], VOCAB)
            });
            let mut second = executor.new_session().unwrap();
            let mut solo = executor.new_session().unwrap();
            for session in [&mut first, &mut second, &mut solo] {
                executor.prefill(session, &PROMPT).unwrap();
            }
            let hidden = executor.decoder.weights.shape.hidden;
            let norm = super::super::ops::f32_tensor(
                vec![u32::try_from(hidden).unwrap()],
                &vec![non_finite; hidden],
            )
            .unwrap();
            let (_, mut imported) = executor
                .decoder
                .kernels
                .import_packed_many_with(&[], &[&norm])
                .unwrap();
            executor.decoder.weights.final_norm = imported.pop().unwrap();
            let generation =
                || GenerationState::new(IncrementalDecoder::new(&tokenizer())).with_max_tokens(2);
            let (mut a, mut b, mut reference) = (generation(), generation(), generation());
            let solo_error = executor.decode(&mut solo, &mut reference).unwrap_err();
            assert!(matches!(&solo_error, EngError::Backend { .. }));
            assert!(format!("{solo_error:?}").contains("finite"));
            let error = executor
                .decode_window(&mut [&mut first, &mut second], &mut [&mut a, &mut b])
                .unwrap_err();
            assert!(matches!(&error, EngError::Backend { .. }));
            assert!(format!("{error:?}").contains("finite"));
            for (session, job) in [(&first, &a), (&second, &b)] {
                assert_eq!(session.core.history(), PROMPT);
                assert_eq!(job.remaining_tokens(), Some(2));
            }
        }
    }

    #[test]
    fn window_growth_denial_leaves_only_the_refused_row_unfed_and_retryable() {
        const CAP: u64 = 4 << 20;
        let _serial = crate::native_test_lock();
        let (mut executor, mut first) = build(Q36Rules {
            compile: true,
            max_seqs: 2,
            session_cap: CAP,
            ..rules(vec![], VOCAB)
        });
        let mut second = executor.new_session().unwrap();
        executor.prefill(&mut first, &PROMPT).unwrap();
        executor.prefill(&mut second, &PROMPT).unwrap();
        let growth = super::super::state::kv_growth_bytes(&executor.decoder.weights).unwrap();
        let used = executor.decoder.session_used();
        let pressure = executor
            .decoder
            .budget
            .acquire_growth(CAP - used - growth)
            .unwrap();
        let positions = (first.state.pos(), second.state.pos());
        let (mut a, mut b) = (seeded_generation(3), seeded_generation(3));
        let rows = executor
            .decode_window(&mut [&mut first, &mut second], &mut [&mut a, &mut b])
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].as_ref().unwrap().tokens.len(), 1);
        assert!(matches!(&rows[1], Err(EngError::MemoryPlan(_))));
        assert_eq!(first.state.pos(), positions.0 + 1);
        assert_eq!(second.state.pos(), positions.1);
        assert_eq!(second.core.history(), PROMPT);
        assert_eq!(second.core.pending(), Some(3));
        assert_eq!(a.remaining_tokens(), Some(2));
        assert_eq!(b.remaining_tokens(), Some(3));
        assert_eq!(executor.decoder.session_used(), CAP);
        drop(pressure);
        let retry = executor
            .decode_window(&mut [&mut second], &mut [&mut b])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(retry.tokens.len(), 1);
        assert_eq!(second.state.pos(), positions.1 + 1);
        assert_eq!(first.state.pos(), positions.0 + 1);
        assert_eq!(b.remaining_tokens(), Some(2));
    }

    /// The tokens that a `Record` chain draws from `sampler`: a path that shares no code with the
    /// executor's `Logits` step. The decoder and its state drop before the caller builds an executor.
    fn oracle(prompt: &[u32], count: usize, sampler: &mut Sampler, limit: usize) -> Vec<u32> {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host(&ckpt), 1024).unwrap();
        let mut state = decoder.zero_state().unwrap();
        let (last, head) = prompt.split_last().unwrap();
        for &token in head {
            decoder.step(&mut state, token, StepMode::Prompt).unwrap();
        }
        let mut input = *last;
        let mut chain = Vec::new();
        for _ in 0..count {
            let logits = decoder
                .step(&mut state, input, StepMode::Record)
                .unwrap()
                .logits
                .unwrap();
            input = sampler.sample(&logits[..limit.min(logits.len())]).unwrap();
            chain.push(input);
        }
        chain
    }

    /// The full logits row that a fresh decoder gives for the next token after `prompt`.
    fn logits_alone(prompt: &[u32]) -> Vec<f32> {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host(&ckpt), 1024).unwrap();
        let mut state = decoder.zero_state().unwrap();
        let (last, head) = prompt.split_last().unwrap();
        for &token in head {
            decoder.step(&mut state, token, StepMode::Prompt).unwrap();
        }
        decoder
            .step(&mut state, *last, StepMode::Logits)
            .unwrap()
            .logits
            .unwrap()
    }

    fn bits(row: &[f32]) -> Vec<u32> {
        row.iter().map(|value| value.to_bits()).collect()
    }

    fn decode_all(
        executor: &mut Q36Executor,
        session: &mut Q36Session,
        generation: &mut GenerationState,
    ) -> Vec<DecodeBatch> {
        let mut batches = Vec::new();
        for _ in 0..100 {
            let batch = executor.decode(session, generation).unwrap();
            let done = batch.finish.is_some();
            batches.push(batch);
            if done {
                return batches;
            }
        }
        panic!("the job did not finish in 100 decode calls");
    }

    fn tokens_of(batches: &[DecodeBatch]) -> Vec<u32> {
        batches.iter().flat_map(|b| b.tokens.clone()).collect()
    }

    #[test]
    fn a_seeded_job_follows_the_record_chain_and_ends_at_max_tokens() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PROMPT, 5, &mut seeded_sampler(), VOCAB);
        assert!(
            want.iter().any(|&t| t != want[0]),
            "the chain is constant; pick another seed"
        );
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut job = seeded_generation(5);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let batches = decode_all(&mut executor, &mut session, &mut job);
        assert_eq!(tokens_of(&batches), want, "tokens");
        assert_eq!(batches.len(), 5, "one token per call");
        assert_eq!(batches[4].finish, Some(DecodeFinish::Length), "last finish");
        let mut history = PROMPT.to_vec();
        history.extend_from_slice(&want);
        assert_eq!(executor.history(&session), history.as_slice(), "history");
        // Three prompt tokens and four sampled tokens are fed. The fifth sampled token is pending.
        assert_eq!(session.state.pos(), 7, "fed positions");
        assert_eq!(session.core.pending(), Some(want[4]), "pending token");
    }

    /// The session A, B, A pattern on one executor. Each job starts with a reset, as the engine
    /// thread does. A kept decoder state shows in the fed position, in the logits row of the
    /// first decode step, and, through a seeded sampler, in the sampled tokens.
    #[test]
    fn each_job_after_a_reset_matches_the_same_job_run_alone_in_the_order_a_b_a() {
        const COUNT: usize = 4;
        let _serial = crate::native_test_lock();
        let prompt_a = PROMPT;
        let prompt_b = [6, 4, 7, 1];
        let logits_a = bits(&logits_alone(&prompt_a));
        let logits_b = bits(&logits_alone(&prompt_b));
        assert_ne!(
            logits_a, logits_b,
            "the two prompts must give different logits, or the test is vacuous"
        );
        let want_a = oracle(&prompt_a, COUNT, &mut seeded_sampler(), VOCAB);
        let want_b = oracle(&prompt_b, COUNT, &mut seeded_sampler(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        for (name, prompt, want_logits, want) in [
            ("A", &prompt_a[..], &logits_a, &want_a),
            ("B", &prompt_b[..], &logits_b, &want_b),
            ("A again", &prompt_a[..], &logits_a, &want_a),
        ] {
            // The logits of the first decode step, read through the decoder of the executor.
            executor.reset(&mut session).unwrap();
            assert_eq!(session.state.pos(), 0, "{name}: position after reset");
            assert!(executor.history(&session).is_empty(), "{name}: history");
            executor.prefill(&mut session, prompt).unwrap();
            let pending = session.core.pending().unwrap();
            let row = executor
                .decoder
                .step(&mut session.state, pending, StepMode::Logits)
                .unwrap()
                .logits
                .unwrap();
            assert_eq!(&bits(&row), want_logits, "{name}: first logits row");
            // The sampled tokens of a whole job.
            executor.reset(&mut session).unwrap();
            assert_eq!(session.state.pos(), 0, "{name}: position after reset");
            let mut job = seeded_generation(u32::try_from(COUNT).unwrap());
            executor.prefill(&mut session, prompt).unwrap();
            let got = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
            assert_eq!(&got, want, "{name}: tokens");
            // The prompt and all but the last sampled token are fed. The last one is pending.
            let fed = u32::try_from(prompt.len() + COUNT - 1).unwrap();
            assert_eq!(session.state.pos(), fed, "{name}: fed positions");
        }
    }

    /// The `config.json` of the pinned checkpoint (a committed copy).
    const REAL_CONFIG: &[u8] =
        include_bytes!("../../../../tests/fixtures/qwen3_6_35b_a3b_config.json");

    /// A directory that holds only `config.json`: any attempt to open a shard fails with a
    /// `Checkpoint` error that names `model.safetensors.index.json`.
    fn config_only_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), REAL_CONFIG).unwrap();
        dir
    }

    fn real_rules() -> Q36Rules {
        Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![248_046, 248_044],
            sampler_vocab: 248_077,
            context_limit: 32_768,
            session_cap: 40_960 * 32_768,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        }
    }

    #[test]
    fn open_refuses_a_cap_that_the_model_cannot_meet_before_it_opens_any_shard() {
        let dir = config_only_dir();
        let rules = Q36Rules {
            prefix_store: StoreLimits::default(),
            session_cap: 40_960 * 32_768 - 1,
            ..real_rules()
        };
        let error = Q36Executor::open(dir.path(), 1024, rules).err();
        assert_eq!(
            error,
            Some(EngError::MemoryPlan(
                "context limit 32768 needs 1342177280 bytes of cache; the session cap is 1342177279 bytes"
                    .to_string()
            ))
        );
    }

    #[test]
    fn open_refuses_a_vocabulary_wider_than_the_head_before_it_opens_any_shard() {
        let dir = config_only_dir();
        let rules = Q36Rules {
            prefix_store: StoreLimits::default(),
            sampler_vocab: 248_321,
            ..real_rules()
        };
        let error = Q36Executor::open(dir.path(), 1024, rules).err();
        assert_eq!(
            error,
            Some(EngError::Checkpoint(
                "sampler vocabulary 248321 is outside 1..=248320, the width of the model head"
                    .to_string()
            ))
        );
    }

    #[test]
    fn open_opens_the_shards_only_after_the_rules_pass() {
        let dir = config_only_dir();
        let error = Q36Executor::open(dir.path(), 1024, real_rules()).err();
        assert!(
            matches!(&error, Some(EngError::Checkpoint(m)) if m.contains("model.safetensors.index.json")),
            "{error:?}"
        );
    }

    #[test]
    fn a_prompt_at_the_limit_is_rejected_and_the_last_allowed_position_ends_with_length() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let long: Vec<u32> = (0..LIMIT).map(|i| i % 7 + 1).collect();
        let rejected = executor.prefill(&mut session, &long);
        assert!(
            matches!(rejected, Err(EngError::PromptTooLong { .. })),
            "{rejected:?}"
        );
        assert_eq!(session.state.pos(), 0, "no step ran");
        assert!(executor.history(&session).is_empty(), "history");
        // 7 prompt tokens fit: 6 fed, one pending. Two decodes feed positions 6 and 7. The cap
        // admits exactly 8 positions, so no step is denied.
        let mut job = seeded_generation(100);
        executor.prefill(&mut session, &long[..7]).unwrap();
        let first = executor.decode(&mut session, &mut job);
        assert!(matches!(&first, Ok(b) if b.finish.is_none()), "{first:?}");
        // The second decode feeds position 7. The next feed would need position 8, so the job
        // ends with Length here. A limit that is one too high gives None, and a third decode
        // would be refused by the decoder, so only this rule ends the job.
        let second = executor.decode(&mut session, &mut job);
        assert!(
            matches!(&second, Ok(b) if b.finish == Some(DecodeFinish::Length) && b.tokens.len() == 1),
            "{second:?}"
        );
        assert_eq!(session.state.pos(), 8, "fed positions");
    }

    #[test]
    fn a_prompt_that_cannot_fit_is_refused_before_any_decoder_step() {
        let _serial = crate::native_test_lock();
        let (executor, session) = build(rules(vec![], VOCAB));
        // The over-length form of `session::check_prompt`, written out by hand: a prompt
        // of 8 tokens against the limit of 8 names the largest prompt the gate admits.
        assert_eq!(
            executor.check_prompt(LIMIT as usize),
            Err(EngError::PromptTooLong {
                prompt_tokens: 8,
                limit_tokens: 7
            })
        );
        assert!(
            executor.check_prompt(LIMIT as usize - 1).is_ok(),
            "a prompt of 7 tokens leaves room for one sampled token"
        );
        assert_eq!(session.state.pos(), 0, "no decoder step ran");
        assert!(executor.history(&session).is_empty(), "history");
        assert!(take_log(&executor).is_empty(), "no chunk ran");
    }

    #[test]
    fn the_sampler_sees_only_the_tokenizer_vocabulary() {
        let _serial = crate::native_test_lock();
        const LIMIT_VOCAB: usize = 5;
        const COUNT: usize = 8;
        let open = oracle(&PROMPT, COUNT, &mut seeded_sampler(), VOCAB);
        let want = oracle(&PROMPT, COUNT, &mut seeded_sampler(), LIMIT_VOCAB);
        assert_ne!(
            want, open,
            "the limit must change the chain; pick another seed"
        );
        // The prompt feeds 2 positions and the 8 decode calls feed 8 more, so 10 positions are
        // fed. The limit of 16 positions and its cap leave room for them; the limit of `LIMIT`
        // would end the job with `Length` after 6 tokens.
        let rules = Q36Rules {
            prefix_store: StoreLimits::default(),
            context_limit: 2 * LIMIT,
            session_cap: u64::from(2 * LIMIT) * GROWTH,
            ..rules(vec![], LIMIT_VOCAB)
        };
        let (mut executor, mut session) = build(rules);
        let mut job = seeded_generation(u32::try_from(COUNT).unwrap());
        executor.prefill(&mut session, &PROMPT).unwrap();
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
        assert_eq!(got.len(), COUNT, "the job must reach max tokens");
        assert_eq!(got, want, "tokens");
        assert!(
            got.iter().all(|&t| (t as usize) < LIMIT_VOCAB),
            "a token is outside the vocabulary"
        );
    }

    #[test]
    fn an_eos_token_ends_the_job_with_stop() {
        let _serial = crate::native_test_lock();
        let chain = oracle(&PROMPT, 4, &mut seeded_sampler(), VOCAB);
        let eos = chain[2];
        let first = chain.iter().position(|&t| t == eos).unwrap();
        let (mut executor, mut session) = build(rules(vec![eos], VOCAB));
        let mut job = seeded_generation(100);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let batches = decode_all(&mut executor, &mut session, &mut job);
        assert_eq!(
            tokens_of(&batches),
            chain[..=first].to_vec(),
            "tokens up to EOS"
        );
        let last = batches.last().unwrap();
        assert_eq!(last.finish, Some(DecodeFinish::Stop), "finish");
        assert_eq!(last.text, "", "EOS adds no text");
        assert_eq!(session.core.pending(), None, "EOS is not fed");
    }

    #[test]
    fn a_cancelled_job_resets_the_session_and_the_next_job_is_clean() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PROMPT, 3, &mut seeded_sampler(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut cancelled = seeded_generation(100);
        executor.prefill(&mut session, &PROMPT).unwrap();
        executor.decode(&mut session, &mut cancelled).unwrap();
        cancelled
            .cancel
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let batch = executor.decode(&mut session, &mut cancelled).unwrap();
        assert_eq!(batch.finish, Some(DecodeFinish::Cancelled));
        assert!(executor.history(&session).is_empty(), "history");
        assert_eq!(session.state.pos(), 0, "position after cancel");
        let mut next = seeded_generation(3);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut next));
        assert_eq!(got, want, "tokens after a cancelled job");
    }

    #[test]
    fn rules_that_the_model_cannot_meet_are_rejected_before_the_import() {
        let _serial = crate::native_test_lock();
        let ckpt: &'static StreamedCheckpoint = Box::leak(Box::new(tiny::checkpoint(&TINY_GQA)));
        let cases = [
            (
                "sampler vocabulary 0",
                rules(vec![], 0),
                "sampler vocabulary",
            ),
            (
                "sampler vocabulary 9",
                rules(vec![], VOCAB + 1),
                "sampler vocabulary",
            ),
            (
                "context limit 1",
                Q36Rules {
                    prefix_store: StoreLimits::default(),
                    context_limit: 1,
                    ..rules(vec![], VOCAB)
                },
                "below 2",
            ),
            (
                "cap one byte short",
                Q36Rules {
                    prefix_store: StoreLimits::default(),
                    session_cap: u64::from(LIMIT) * GROWTH - 1,
                    ..rules(vec![], VOCAB)
                },
                "session cap",
            ),
        ];
        for (name, rules, want) in cases {
            let built =
                Q36Executor::from_parts(MlxKernels::try_new().unwrap(), ckpt, host(ckpt), rules);
            let message = match built.err() {
                Some(
                    EngError::Checkpoint(m) | EngError::BadRequest(m) | EngError::MemoryPlan(m),
                ) => m,
                other => panic!("{name}: expected a rule error, got {other:?}"),
            };
            assert!(message.contains(want), "{name}: {message}");
        }
    }

    #[test]
    fn the_rules_are_checked_in_the_order_vocabulary_limit_cap() {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let shape = host(&ckpt).shape().clone();
        let message = |rules: Q36Rules| match admit(&shape, &rules) {
            Err(EngError::Checkpoint(m) | EngError::BadRequest(m) | EngError::MemoryPlan(m)) => m,
            other => panic!("expected a rule error, got {other:?}"),
        };
        let bad = |sampler_vocab, context_limit, session_cap| Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab,
            context_limit,
            session_cap,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        };
        // Every rule is broken: the vocabulary is named first, then the limit, then the cap.
        assert!(message(bad(0, 1, 0)).contains("sampler vocabulary"));
        assert!(message(bad(VOCAB, 1, 0)).contains("below 2"));
        assert!(message(bad(VOCAB, LIMIT, 0)).contains("session cap"));
        // The smallest admitted limit is 2: a limit of 2 passes when the cap holds 2 positions.
        assert_eq!(admit(&shape, &bad(VOCAB, 2, 2 * GROWTH)), Ok(()));
        assert_eq!(
            admit(&shape, &bad(VOCAB, LIMIT, u64::from(LIMIT) * GROWTH)),
            Ok(())
        );
    }

    #[test]
    fn the_admitted_growth_equals_the_growth_that_the_decoder_charges() {
        let _serial = crate::native_test_lock();
        let (executor, _session) = build(rules(vec![], VOCAB));
        let per_step = super::super::state::kv_growth_bytes(&executor.decoder.weights).unwrap();
        assert_eq!(per_step, GROWTH, "bytes per step");
        let shape = executor.decoder.weights.shape.clone();
        assert_eq!(
            growth_per_position(&shape, KvDtype::F32, Activation::F32).unwrap(),
            per_step,
            "admission rule"
        );
    }

    #[test]
    fn the_decoder_budget_has_the_session_cap() {
        let _serial = crate::native_test_lock();
        let (mut executor, _session) = build(rules(vec![], VOCAB));
        let cap = u64::from(LIMIT) * GROWTH;
        let denied = executor.decoder.budget.acquire_growth(cap + 1);
        assert!(
            matches!(denied, Err(EngError::MemoryPlan(_))),
            "{:?}",
            denied.err()
        );
        let granted = executor.decoder.budget.acquire_growth(cap);
        assert!(granted.is_ok(), "the cap itself is admitted");
    }

    #[test]
    fn the_decoder_refuses_the_position_of_the_context_limit() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(executor.decoder.context_limit, LIMIT, "decoder limit");
        // The cap admits exactly `LIMIT` positions, so only the limit can refuse the next step.
        for token in 0..LIMIT {
            let step = executor
                .decoder
                .step(&mut session.state, token % 7 + 1, StepMode::Prompt);
            assert!(step.is_ok(), "position {token}: {:?}", step.err());
        }
        let refused = executor
            .decoder
            .step(&mut session.state, 1, StepMode::Prompt);
        assert!(
            matches!(refused, Err(EngError::ContextLimit(_))),
            "position {LIMIT}: {:?}",
            refused.err()
        );
    }

    /// The context limit of the chunk tests: 24 positions.
    const CHUNK_LIMIT: u32 = 24;

    /// The rules of the chunk tests: the cap admits exactly `CHUNK_LIMIT` positions.
    fn chunk_rules() -> Q36Rules {
        Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab: VOCAB,
            context_limit: CHUNK_LIMIT,
            session_cap: u64::from(CHUNK_LIMIT) * GROWTH,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        }
    }

    /// A prompt of `len` tokens: token `i` is `(i + 1) % 8`, the sequence of the chunk tests (the
    /// fixture has 8 ids). The host reference routes this sequence, followed by `FORCED`, with a
    /// gap of at least 0.0167 between the last chosen and the first rejected router probability in
    /// every layer, so a chunk and a step choose the same experts.
    fn cycle(len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| u32::try_from((i + 1) % 8).unwrap())
            .collect()
    }

    /// The row count of each chunk that ran since the last call. The log is cleared.
    fn take_log(executor: &Q36Executor) -> Vec<usize> {
        std::mem::take(&mut *executor.log.lock().unwrap())
    }

    /// The row counts of the chunks that one call makes of `rows` rows with chunks of `size`: a
    /// run of one row is a step and makes no chunk.
    fn expected_chunks(rows: usize, size: usize) -> Vec<usize> {
        let mut left = rows;
        let mut chunks = Vec::new();
        while left > 0 {
            let n = left.min(size);
            if n > 1 {
                chunks.push(n);
            }
            left -= n;
        }
        chunks
    }

    /// The bytes of the weights that the executor holds.
    fn held(executor: &Q36Executor) -> u64 {
        executor.decoder.weights.byte_len().unwrap()
    }

    /// The weights bytes of an executor built with `rules`: the executor, its session, and their
    /// native adapter lease all drop inside the helper, so the caller never holds a lease of a
    /// measuring build and a later `serve_with` stays within the two-lease bound of the native
    /// adapter.
    fn held_by(rules: Q36Rules) -> u64 {
        let (executor, _session) = build(rules);
        held(&executor)
    }

    /// Plans the chunk size `rows` through the admission rule: a budget that leaves exactly the
    /// workspace of `rows` rows after the weights, and no limit from the worker piece.
    fn plan_chunk(executor: &mut Q36Executor, rows: u32) -> Option<u32> {
        let workspace = executor.prefill_cost().unwrap().workspace(rows).unwrap();
        let budget = held(executor) + workspace;
        executor.plan_prefill(budget, 0, 2_048).unwrap().chunk
    }

    /// The index of the largest entry, the gap to the second largest, and the largest magnitude.
    fn top_two(row: &[f32]) -> (usize, f32, f32) {
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        let scale = row.iter().fold(0f32, |top, value| top.max(value.abs()));
        (order[0], row[order[0]] - row[order[1]], scale)
    }

    /// The tokens that both jobs of the chunk parity test feed after the prompt, so the two
    /// sessions see one chain whatever their logits are.
    const FORCED: [u32; 5] = [3, 5, 1, 6, 2];

    /// What a prefill leaves behind: the session after the prompt, the charge, and the logits row of
    /// each `Logits` step on the pending token and then on `FORCED`.
    struct Left {
        pos: u32,
        history: Vec<u32>,
        prompt_len: usize,
        pending: Option<u32>,
        used: u64,
        rows: Vec<Vec<f32>>,
    }

    /// Resets the session and prefills `prompt` in worker pieces of `worker` tokens.
    fn start_job(
        executor: &mut Q36Executor,
        session: &mut Q36Session,
        prompt: &[u32],
        worker: usize,
    ) {
        executor.reset(session).unwrap();
        executor.boundary(session);
        for piece in prompt.chunks(worker) {
            executor.prefill(session, piece).unwrap();
        }
    }

    /// Runs the prompt as the worker does, then feeds the pending token and `FORCED` through
    /// `Logits` steps and keeps every logits row.
    fn run_job(
        executor: &mut Q36Executor,
        session: &mut Q36Session,
        prompt: &[u32],
        worker: usize,
    ) -> Left {
        start_job(executor, session, prompt, worker);
        let pending = session.core.pending().unwrap();
        let (pos, history, prompt_len, used) = (
            session.state.pos(),
            session.core.history().to_vec(),
            session.core.prompt_len(),
            executor.decoder.session_used(),
        );
        let rows = std::iter::once(pending)
            .chain(FORCED)
            .map(|token| {
                executor
                    .decoder
                    .step(&mut session.state, token, StepMode::Logits)
                    .unwrap()
                    .logits
                    .unwrap()
            })
            .collect();
        Left {
            pos,
            history,
            prompt_len,
            pending: Some(pending),
            used,
            rows,
        }
    }

    /// Row 1: 10 tokens, chunks of 4, worker pieces of 5: the calls give the rows 4, then 4 and 1
    /// (the carried pending token leads the second call, and the single row is a step). Row 2: 14
    /// tokens, chunks of 8, worker pieces of 9: the calls give 8, then 5. A chunk equals the steps
    /// within the chunk-parity tolerance, not bit for bit. Both jobs get the same token chain, and
    /// every logits row is compared with `assert_parity`. The argmax is compared only where the
    /// stepped row's top two logits differ by more than twice the tolerance, because a smaller gap
    /// can flip inside the tolerance.
    #[test]
    fn a_chunked_prefill_leaves_the_session_of_one_token_steps() {
        let _serial = crate::native_test_lock();
        for (len, chunk, worker) in [(10usize, 4u32, 5usize), (14, 8, 9)] {
            let what = format!("{len} tokens, chunks of {chunk}, worker pieces of {worker}");
            let size = usize::try_from(chunk).unwrap();
            let prompt = cycle(len);
            let (mut executor, mut session) = build(chunk_rules());
            let stepped = run_job(&mut executor, &mut session, &prompt, worker);
            assert_eq!(
                executor.prefill_chunk(),
                None,
                "{what}: the default is steps"
            );
            assert!(take_log(&executor).is_empty(), "{what}: no chunk ran");
            assert_eq!(
                plan_chunk(&mut executor, chunk),
                Some(chunk),
                "{what}: plan"
            );
            let chunked = run_job(&mut executor, &mut session, &prompt, worker);
            let first = (worker - 1).min(len - 1);
            let mut want = expected_chunks(first, size);
            want.extend(expected_chunks(len - 1 - first, size));
            assert_eq!(take_log(&executor), want, "{what}: chunk row counts");
            let fed = u32::try_from(len - 1).unwrap();
            assert_eq!(stepped.pos, fed, "{what}: steps pos");
            assert_eq!(chunked.pos, stepped.pos, "{what}: fed positions");
            assert_eq!(chunked.used, u64::from(fed) * GROWTH, "{what}: charge");
            assert_eq!(chunked.used, stepped.used, "{what}: charge vs steps");
            assert_eq!(chunked.history, prompt, "{what}: history");
            assert_eq!(chunked.history, stepped.history, "{what}: history vs steps");
            assert_eq!(chunked.prompt_len, len, "{what}: held prompt length");
            assert_eq!(chunked.pending, prompt.last().copied(), "{what}: pending");
            assert_eq!(chunked.pending, stepped.pending, "{what}: pending vs steps");
            assert_eq!(chunked.rows.len(), 1 + FORCED.len(), "{what}: row count");
            for (n, (got, want)) in chunked.rows.iter().zip(&stepped.rows).enumerate() {
                assert_parity(got, want, &format!("{what}: logits row {n}"));
                let (best, margin, scale) = top_two(want);
                if margin > 2.0 * 1e-4 * scale {
                    assert_eq!(top_two(got).0, best, "{what}: argmax of row {n}");
                }
            }
            // A job that decodes after a chunked prompt still finishes.
            start_job(&mut executor, &mut session, &prompt, worker);
            let mut job = seeded_generation(6);
            let tokens = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
            assert_eq!(tokens.len(), 6, "{what}: decoded tokens");
        }
    }

    /// Each call is split at the planned size: a chunk never has more rows than the size, a last
    /// single row is a step, a size of 2 is a chunk, and a size of 1 makes no chunk at all.
    #[test]
    fn the_chunks_of_a_prompt_never_exceed_the_planned_size() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(chunk_rules());
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        let result = executor.prefill(&mut session, &cycle(10));
        assert!(result.is_ok(), "a prompt of 10 tokens: {result:?}");
        assert_eq!(
            take_log(&executor),
            vec![4, 4],
            "9 rows: two chunks of 4 and a step"
        );
        for (size, want) in [(2u32, vec![2, 2]), (1, vec![])] {
            executor.force_chunk(Some(size));
            executor.reset(&mut session).unwrap();
            let result = executor.prefill(&mut session, &cycle(6));
            assert!(result.is_ok(), "size {size}: {result:?}");
            assert_eq!(take_log(&executor), want, "size {size}: 5 rows");
        }
    }

    /// The plan reads the weights from the decoder, subtracts the reserve, and takes the largest
    /// candidate that fits and does not pass the worker piece. The planned size is the only value
    /// of the decoder chunk size. The workspace constants are the planning values of
    /// `chunk_plan.rs`: 1 MiB per token, two score copies of 4 heads, 4 bytes, and 24 positions
    /// (768 bytes), and 100,000,000 bytes plus the 512 MiB retire cap fixed.
    #[test]
    fn the_plan_follows_the_budget_the_weights_the_reserve_and_the_worker_piece() {
        let _serial = crate::native_test_lock();
        let (mut executor, _session) = build(chunk_rules());
        assert_eq!(executor.prefill_chunk(), None, "before planning");
        assert_eq!(executor.prefill_piece(), None, "no piece before planning");
        let cost = executor.prefill_cost().unwrap();
        assert_eq!(
            (cost.per_token, cost.fixed),
            (1_048_576 + 768, 100_000_000 + (512 << 20)),
            "the cost terms"
        );
        let w8 = cost.workspace(8).unwrap();
        assert_eq!(w8, 8 * 1_049_344 + 636_870_912, "workspace of 8 rows");
        let weights = held(&executor);
        let mut plan = |budget: u64, reserve: u64, max_rows: u32| {
            let plan = executor.plan_prefill(budget, reserve, max_rows).unwrap();
            assert_eq!(executor.prefill_chunk(), plan.chunk, "the plan is stored");
            assert_eq!(
                executor.prefill_piece(),
                plan.chunk.map(|rows| usize::try_from(rows).unwrap()),
                "the worker piece follows the plan"
            );
            if let Some(rows) = plan.chunk {
                assert_eq!(executor.decoder.chunk_rows, rows, "the decoder follows");
            }
            assert_eq!(
                (
                    plan.terms.budget,
                    plan.terms.weights,
                    plan.terms.session_reserve
                ),
                (budget, weights, reserve),
                "the plan records the terms"
            );
            plan.chunk
        };
        assert_eq!(plan(weights + w8, 0, 2_048), Some(8), "exactly 8 rows");
        assert_eq!(plan(weights + w8 - 1, 0, 2_048), Some(4), "one byte less");
        assert_eq!(plan(weights + w8 + 3_000, 3_000, 2_048), Some(8), "reserve");
        assert_eq!(
            plan(weights + w8 + 2_999, 3_000, 2_048),
            Some(4),
            "reserve, one less"
        );
        assert_eq!(plan(weights, 0, 2_048), None, "no room for a chunk");
        let huge = weights + (1 << 40);
        assert_eq!(plan(huge, 0, 3_000), Some(2_048), "the largest candidate");
        assert_eq!(
            plan(huge, 0, 6),
            Some(4),
            "the worker piece limits the size"
        );
        assert_eq!(plan(huge, 0, 8), Some(8), "a piece of exactly 8");
        assert_eq!(plan(huge, 0, 1), None, "a piece of one row");
        let refused = executor.plan_prefill(huge, 0, 0);
        assert!(
            matches!(refused, Err(EngError::BadRequest(_))),
            "a piece of zero rows: {refused:?}"
        );
        assert_eq!(
            executor.prefill_chunk(),
            None,
            "a refused plan changes nothing"
        );
    }

    /// Both activation dtypes plan a chunk. A BF16 executor plans the same chunk as the F32 control
    /// on the same budget (here the largest, 256), and its worker piece follows, so a BF16 engine
    /// serves its prompts in chunks like an F32 engine.
    #[test]
    fn plan_prefill_admits_the_same_chunk_for_both_activation_dtypes() {
        let _serial = crate::native_test_lock();
        let (mut f32_executor, _session) = build(chunk_rules());
        let huge = held(&f32_executor) + (1 << 40);
        let plan = f32_executor.plan_prefill(huge, 0, 256).unwrap();
        assert_eq!(
            plan.chunk,
            Some(256),
            "the F32 control plans the largest chunk"
        );
        assert_eq!(
            f32_executor.prefill_piece(),
            Some(256),
            "the F32 piece follows"
        );
        let mut rules = chunk_rules();
        rules.activation = Activation::Bf16;
        let (mut bf16_executor, _session) = build(rules);
        assert_eq!(
            bf16_executor.decoder.weights.consts.act,
            crate::tensor::Dtype::Bf16,
            "the BF16 executor holds BF16 activations"
        );
        let plan = bf16_executor.plan_prefill(huge, 0, 256).unwrap();
        assert_eq!(plan.chunk, Some(256), "a BF16 engine plans the same chunk");
        assert_eq!(
            bf16_executor.prefill_chunk(),
            Some(256),
            "the chunk is stored"
        );
        assert_eq!(
            bf16_executor.prefill_piece(),
            Some(256),
            "the BF16 piece follows like the F32 one"
        );
    }

    /// Without a plan, and with a plan that admits no chunk, every prompt row is a one-token
    /// step: the logits row is bit for bit that of the steps of a fresh decoder. A failure on the
    /// one-token path keeps its earlier behavior: the earlier rows stay fed and the executor does
    /// not reset (the worker does).
    #[test]
    fn the_chunk_path_is_off_until_a_budget_plans_it() {
        let _serial = crate::native_test_lock();
        let want = bits(&logits_alone(&PROMPT));
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        for planned in [false, true] {
            executor.reset(&mut session).unwrap();
            if planned {
                let weights = held(&executor);
                let plan = executor.plan_prefill(weights, 0, 2_048).unwrap();
                assert_eq!(plan.chunk, None, "a budget of the weights admits no chunk");
            }
            assert_eq!(executor.prefill_chunk(), None, "planned {planned}");
            executor.prefill(&mut session, &PROMPT).unwrap();
            assert!(
                take_log(&executor).is_empty(),
                "planned {planned}: no chunk"
            );
            let pending = session.core.pending().unwrap();
            let row = executor
                .decoder
                .step(&mut session.state, pending, StepMode::Logits)
                .unwrap()
                .logits
                .unwrap();
            assert_eq!(bits(&row), want, "planned {planned}: bit for bit the steps");
        }
        executor.reset(&mut session).unwrap();
        let failed = executor.prefill(&mut session, &[1, 2, 8, 3, 4]);
        assert!(failed.is_err(), "token 8 is outside the vocabulary");
        assert_eq!(session.state.pos(), 2, "the two earlier rows stay fed");
        assert!(
            executor.history(&session).is_empty(),
            "the history did not grow"
        );
    }

    /// The state of a session after a reset that followed a failed chunk.
    fn assert_clean_after(executor: &Q36Executor, session: &Q36Session, what: &str) {
        assert_eq!(session.state.pos(), 0, "{what}: position");
        assert!(!session.state.is_poisoned(), "{what}: poison");
        assert!(executor.history(session).is_empty(), "{what}: history");
        assert_eq!(session.core.pending(), None, "{what}: pending token");
        assert_eq!(
            executor.decoder.session_used(),
            0,
            "{what}: the permits went back"
        );
    }

    /// A growth permit that is denied in the second chunk. Room for 5 positions: the first chunk
    /// of 4 rows publishes, and the second is denied with `MemoryPlan`. The executor does not
    /// reset: the first chunk stays fed and the permits stay charged. The caller (the worker)
    /// resets through the public `reset`. Then the session is clean, and the next job matches the
    /// job before the failure.
    #[test]
    fn a_denied_growth_permit_in_a_later_chunk_is_cleared_by_the_reset_of_the_caller() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(chunk_rules());
        executor.force_chunk(Some(4));
        let prompt = cycle(10);
        let control = run_job(&mut executor, &mut session, &prompt, 10);
        executor.reset(&mut session).unwrap();
        take_log(&executor);
        executor
            .decoder
            .set_session_budget(SessionBudget::new(5 * GROWTH));
        let failed = executor.prefill(&mut session, &prompt);
        assert!(
            matches!(&failed, Err(EngError::MemoryPlan(m)) if m == "session growth requires 4096 bytes, available 1024"),
            "{failed:?}"
        );
        assert_eq!(take_log(&executor), vec![4], "the first chunk ran");
        assert_eq!(
            session.state.pos(),
            4,
            "the executor does not reset: the first chunk stays fed"
        );
        assert!(
            executor.history(&session).is_empty(),
            "the history did not grow"
        );
        executor.reset(&mut session).unwrap();
        assert_clean_after(&executor, &session, "after a denied permit and a reset");
        executor
            .decoder
            .set_session_budget(SessionBudget::new(u64::from(CHUNK_LIMIT) * GROWTH));
        let after = run_job(&mut executor, &mut session, &prompt, 10);
        assert_eq!(
            (after.pos, after.used),
            (control.pos, control.used),
            "state"
        );
        assert_eq!(after.history, control.history, "history");
        for (n, (got, want)) in after.rows.iter().zip(&control.rows).enumerate() {
            assert_parity(got, want, &format!("logits row {n} after the failure"));
        }
    }

    /// A token outside the vocabulary in the second chunk. The read fails after the chunk has
    /// poisoned the state, with the first chunk published. The executor does not reset. After the
    /// reset of the caller the session is clean, and the next job matches the job before the failure.
    #[test]
    fn a_bad_token_in_a_later_chunk_is_cleared_by_the_reset_of_the_caller() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(chunk_rules());
        executor.force_chunk(Some(4));
        let prompt = cycle(10);
        let control = run_job(&mut executor, &mut session, &prompt, 10);
        executor.reset(&mut session).unwrap();
        take_log(&executor);
        // Rows 1 to 4, then 5, 8, 6, 7: token 8 is outside the 8-row embedding.
        let failed = executor.prefill(&mut session, &[1, 2, 3, 4, 5, 8, 6, 7, 1, 2]);
        assert!(
            matches!(
                failed,
                Err(EngError::BadRequest(_) | EngError::Checkpoint(_))
            ),
            "{failed:?}"
        );
        assert_eq!(take_log(&executor), vec![4], "the first chunk ran");
        assert!(
            session.state.pos() > 0,
            "the executor does not reset: the first chunk stays fed"
        );
        assert_eq!(session.core.pending(), None, "the pending token was taken");
        executor.reset(&mut session).unwrap();
        assert_clean_after(&executor, &session, "after a bad token and a reset");
        let after = run_job(&mut executor, &mut session, &prompt, 10);
        assert_eq!(
            (after.pos, after.used),
            (control.pos, control.used),
            "state"
        );
        for (n, (got, want)) in after.rows.iter().zip(&control.rows).enumerate() {
            assert_parity(got, want, &format!("logits row {n} after the failure"));
        }
    }

    /// The engine thread feeds worker pieces of 4 tokens with the chunk size 4. A piece feeds its
    /// head and the carried pending token: 3, 4, and 2 rows, each one chunk. The size agrees with
    /// the piece, so no chunk is split or left short.
    #[test]
    fn an_engine_thread_feeds_each_worker_piece_as_one_chunk() {
        use crate::sched::{
            job::{Job, JobEvent, JobUsage, PrefixReport},
            worker::{EngineState, ShutdownReport, spawn_engine},
        };
        let _serial = crate::native_test_lock();
        let log: std::sync::Arc<std::sync::Mutex<Vec<usize>>> = std::sync::Arc::default();
        let thread_log = std::sync::Arc::clone(&log);
        let engine = spawn_engine(4, move || {
            let (mut executor, session) = build(chunk_rules());
            executor.force_chunk(Some(4));
            executor.log = thread_log;
            Ok((executor, session))
        })
        .unwrap();
        let (job, mut handle) = Job::new(cycle(10), seeded_generation(4)).unwrap();
        engine.handle().try_submit(job).unwrap();
        let mut tokens = Vec::new();
        let mut usage = None;
        while let Some(event) = handle.events.blocking_recv() {
            match event {
                JobEvent::Batch(batch) => tokens.extend(batch.tokens),
                JobEvent::Finished(done) => usage = Some(done),
                JobEvent::Error(error) => panic!("{error:?}"),
            }
        }
        assert_eq!(*log.lock().unwrap(), vec![3, 4, 2], "chunk rows");
        assert_eq!(tokens.len(), 4, "tokens");
        assert_eq!(
            usage,
            Some(JobUsage {
                prompt_tokens: 10,
                completion_tokens: 4,
                reused_tokens: 0,
                prefix: Some(PrefixReport::default())
            }),
            "usage"
        );
        assert_eq!(
            engine.shutdown(),
            ShutdownReport {
                state: EngineState::Stopped,
                failure: None
            }
        );
    }

    use crate::family::qwen36::{
        settings::{Backend, NO_BUDGET_PIECE, Q36Env, Q36Options, Q36Settings},
        start::plan_prefill,
    };

    /// The weights and the chunk cost of the synthetic executor: the terms of an exact budget.
    fn probe() -> (u64, ChunkCost) {
        let (executor, _session) = build(chunk_rules());
        (held(&executor), executor.prefill_cost().unwrap())
    }

    /// The settings of a serving entry on the synthetic model. The cache limit is a dummy, and
    /// `options` sets the rest.
    fn serving(options: Q36Options) -> Q36Settings {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..options
        };
        let env = Q36Env::default();
        let built = Q36Settings::new(
            Path::new("/m"),
            Path::new("/t"),
            Some(&options),
            &env,
            Backend::Mlx,
        );
        built.unwrap()
    }

    /// What one served job shows: the start-up line, the planned chunk, the chunk log, and the
    /// number of tokens that the job generated.
    struct Served {
        line: Option<String>,
        planned: Option<u32>,
        log: Vec<usize>,
        tokens: usize,
    }

    /// Starts an engine thread as `start::start` does, on the synthetic model: the thread builds
    /// the executor, calls `start::plan_prefill` with the settings, and the fallback worker piece
    /// is `NO_BUDGET_PIECE`, which the planned chunk replaces. It runs one job of 10 prompt tokens
    /// and 4 generated tokens.
    fn serve(settings: Q36Settings) -> Served {
        serve_with(settings, chunk_rules())
    }

    /// `serve` with the rules that the engine thread builds, so a test can serve another dtype.
    fn serve_with(settings: Q36Settings, rules: Q36Rules) -> Served {
        use crate::sched::{
            job::{Job, JobEvent},
            worker::{EngineState, ShutdownReport, spawn_engine},
        };
        let log: std::sync::Arc<std::sync::Mutex<Vec<usize>>> = std::sync::Arc::default();
        let thread_log = std::sync::Arc::clone(&log);
        let seen: std::sync::Arc<std::sync::Mutex<(Option<String>, Option<u32>)>> =
            std::sync::Arc::default();
        let thread_seen = std::sync::Arc::clone(&seen);
        let fallback = usize::try_from(NO_BUDGET_PIECE).unwrap();
        let engine = spawn_engine(fallback, move || {
            let (mut executor, session) = build(rules);
            let line = plan_prefill(&mut executor, &settings)?;
            *thread_seen.lock().unwrap() = (line, executor.prefill_chunk());
            executor.log = thread_log;
            Ok((executor, session))
        })
        .unwrap();
        let (job, mut handle) = Job::new(cycle(10), seeded_generation(4)).unwrap();
        engine.handle().try_submit(job).unwrap();
        let mut tokens = 0;
        while let Some(event) = handle.events.blocking_recv() {
            match event {
                JobEvent::Batch(batch) => tokens += batch.tokens.len(),
                JobEvent::Finished(_) => {}
                JobEvent::Error(error) => panic!("{error:?}"),
            }
        }
        assert_eq!(
            engine.shutdown(),
            ShutdownReport {
                state: EngineState::Stopped,
                failure: None
            }
        );
        let (line, planned) = std::mem::take(&mut *seen.lock().unwrap());
        let log = std::mem::take(&mut *log.lock().unwrap());
        Served {
            line,
            planned,
            log,
            tokens,
        }
    }

    /// A serving entry with a budget plans a chunk from its settings, and the worker feeds the
    /// prompt in chunks. The settings carry the explicit F32 control (`activation_dtype`), the
    /// qualification row of the planner; the BF16 entry has its own test below. The planned chunk
    /// sets the worker piece, whatever the largest chunk is: row 1 plans 4 with a largest chunk
    /// of 8, and row 2 plans 4 with a largest chunk of 4, so the pieces of 4 tokens give the
    /// chunks 3, 4, then 2 in both rows.
    #[test]
    fn a_serving_entry_with_a_budget_plans_a_chunk_and_the_prompt_runs_in_chunks() {
        let _serial = crate::native_test_lock();
        let (weights, cost) = probe();
        let exact = weights + cost.workspace(4).unwrap();
        let rows = [
            (exact, 8u32, 4u32, vec![3, 4, 2], cost.workspace(4).unwrap()),
            (weights + (1 << 40), 4, 4, vec![3, 4, 2], 1 << 40),
        ];
        for (budget, largest, planned, want, remaining) in rows {
            let served = serve(serving(Q36Options {
                prefill_chunk: Some(largest),
                prefill_budget_bytes: Some(budget),
                prefill_session_reserve_bytes: Some(0),
                activation_dtype: Some("f32".to_string()),
                ..Q36Options::default()
            }));
            let what = format!("budget {budget}, largest {largest}");
            assert_eq!(served.planned, Some(planned), "{what}: planned chunk");
            let line = served.line.expect("a budget gives a start-up line");
            let want_line = format!(
                "q36 prefill: chunk of {planned} rows planned (largest {largest}); budget_bytes={budget} weights_bytes={weights} session_reserve_bytes=0 remaining_bytes={remaining}"
            );
            assert_eq!(line, want_line, "{what}: start-up line");
            assert_eq!(served.log, want, "{what}: chunk rows");
            assert_eq!(served.tokens, 4, "{what}: tokens");
        }
    }

    /// A BF16 serving entry (the default of `activation_dtype`) plans a chunk like the F32 control
    /// when the budget admits the largest one: the start-up line is the chunk line of either dtype
    /// and its `weights_bytes` term is the byte count after `convert_norms`, the prompt runs in
    /// chunks, and the job decodes. The byte counts are measured inside `held_by`, so every lease
    /// of a measuring build drops before `serve_with` builds its own and at most one lease is live
    /// at any time.
    #[test]
    fn a_bf16_serving_entry_plans_a_chunk_and_runs_it() {
        let _serial = crate::native_test_lock();
        let (weights, _cost) = probe();
        let f32_held = held_by(chunk_rules());
        assert_eq!(f32_held, weights, "the F32 control holds the import bytes");
        let mut bf16_rules = chunk_rules();
        bf16_rules.activation = Activation::Bf16;
        // The weights bytes of the line are the decoder's byte count AFTER `convert_norms`. The
        // BF16 total is strictly below the F32 import because the import itself keeps the scales
        // and biases in BF16, which saves about 2.13 GB on the real model; `convert_norms` adds
        // only 337,920 B of savings, so this assert pins the dtype of the import, not
        // `convert_norms`. The mutation that removes `convert_norms` is caught by the served job
        // itself: F32 norms against BF16 activations fail the `input norm output has dtype` guard.
        // On the real model the term is the 19,286,549,004 B of the BF16 tool-forward log; on the
        // synthetic model the same rule shows as the smaller count.
        let bf16_weights = held_by(bf16_rules);
        assert!(
            bf16_weights < weights,
            "the BF16 import is below the F32 import: {bf16_weights} against {weights}"
        );
        let mut rules = chunk_rules();
        rules.activation = Activation::Bf16;
        let budget = weights + (1 << 40);
        let served = serve_with(
            serving(Q36Options {
                prefill_chunk: Some(4),
                prefill_budget_bytes: Some(budget),
                prefill_session_reserve_bytes: Some(0),
                ..Q36Options::default()
            }),
            rules,
        );
        assert_eq!(served.planned, Some(4), "the chunk is planned");
        let line = served.line.expect("a budget gives a start-up line");
        assert_eq!(
            line,
            format!(
                "q36 prefill: chunk of 4 rows planned (largest 4); budget_bytes={budget} \
                 weights_bytes={bf16_weights} session_reserve_bytes=0 remaining_bytes={}",
                budget - bf16_weights
            ),
            "the start-up line carries the post-conversion weights bytes"
        );
        assert_eq!(served.log, vec![3, 4, 2], "the prompt runs in chunks");
        assert_eq!(served.tokens, 4, "the job still decodes");
    }

    /// Without a budget the start-up code plans nothing: no line, no chunk size, and every prompt
    /// row is a one-token step, whatever the worker piece is. The settings carry the explicit F32
    /// control, the qualification row; with either dtype the plan needs a budget first.
    #[test]
    fn a_serving_entry_without_a_budget_keeps_the_one_token_path() {
        let _serial = crate::native_test_lock();
        let served = serve(serving(Q36Options {
            prefill_chunk: Some(4),
            activation_dtype: Some("f32".to_string()),
            ..Q36Options::default()
        }));
        assert_eq!(served.line, None, "start-up line");
        assert_eq!(served.planned, None, "planned chunk");
        assert_eq!(served.log, Vec::<usize>::new(), "chunk rows");
        assert_eq!(served.tokens, 4, "tokens");
    }

    /// A plan that admits no chunk does not fail the engine. The path stays off, the job runs, and
    /// the line names the terms that were charged. The settings carry the explicit F32 control
    /// (`activation_dtype`), the qualification row of the same budget arithmetic for both dtypes.
    /// Row 1: the budget of 4 rows with a reserve of 2 GiB plus 1024 bytes leaves nothing. Row 2:
    /// one byte below the workspace of 4 rows. Row 3: a largest chunk of 3 is below the smallest
    /// candidate.
    #[test]
    fn a_budget_that_admits_no_chunk_keeps_the_one_token_path_and_says_why() {
        let _serial = crate::native_test_lock();
        let (weights, cost) = probe();
        let exact = weights + cost.workspace(4).unwrap();
        let large_reserve = (2u64 << 30) + 1024;
        let rows = [
            (exact, 8u32, large_reserve, 0u64),
            (exact - 1, 8, 0, cost.workspace(4).unwrap() - 1),
            (weights + (1 << 40), 3, 0, 1 << 40),
        ];
        for (budget, largest, charged, remaining) in rows {
            let served = serve(serving(Q36Options {
                prefill_chunk: Some(largest),
                prefill_budget_bytes: Some(budget),
                prefill_session_reserve_bytes: Some(charged),
                activation_dtype: Some("f32".to_string()),
                ..Q36Options::default()
            }));
            let what = format!("budget {budget}, largest {largest}");
            assert_eq!(served.planned, None, "{what}: planned chunk");
            let want = format!(
                "q36 prefill: no chunk above one token fits (largest {largest}); prompts run one token at a time; budget_bytes={budget} weights_bytes={weights} session_reserve_bytes={charged} remaining_bytes={remaining}"
            );
            assert_eq!(served.line, Some(want), "{what}: start-up line");
            assert_eq!(served.log, Vec::<usize>::new(), "{what}: chunk rows");
            assert_eq!(served.tokens, 4, "{what}: tokens");
        }
    }

    /// The synthetic model has 4 layers. A one-token `Prompt` step makes 2 synchronizations under
    /// device routing (the executor's default), so 9 rows make 18. A chunk in `Prompt` mode makes
    /// 8 on this backend. The chunk's sync sites are shared code (`prefill/step.rs`), but this
    /// count is the MLX native counter's own granularity, which fires two more than the CUDA-side
    /// derivation of `layers + 2` = 6 that 29fbd00cb wrote from the site anatomy — an expectation
    /// that never executed, because this test only builds under `native-fault-qualification` (an
    /// MLX gate) while that commit's checks ran the default feature set. The 8 here is the
    /// observed count (the baaa86ceb expectation, verified on the Mac). A prompt of 10 tokens has
    /// 9 rows: one chunk of 8 and a step for the last row make 8 + 2, which also pins the mode.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_chunked_prefill_makes_the_synchronizations_of_one_chunk_per_run_of_rows() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(chunk_rules());
        let count = |executor: &Q36Executor| executor.decoder.kernels.qualification_sync_count();
        let before = count(&executor);
        executor.prefill(&mut session, &cycle(10)).unwrap();
        assert_eq!(count(&executor) - before, 18, "nine steps");
        assert_eq!(plan_chunk(&mut executor, 8), Some(8), "plan");
        executor.reset(&mut session).unwrap();
        let before = count(&executor);
        executor.prefill(&mut session, &cycle(10)).unwrap();
        assert_eq!(take_log(&executor), vec![8], "one chunk of 8 rows");
        assert_eq!(count(&executor) - before, 8 + 2, "chunk, then step");
    }

    static_assertions::assert_not_impl_any!(Q36Executor: Send, Sync);
    static_assertions::assert_not_impl_any!(Q36Session: Send, Sync);

    /// Pins the step modes through the executor by the synchronization count, under both routings.
    /// The synthetic model has 4 layers. Device routing is the default and the only routing that
    /// the executor uses in production: a Prompt step costs 2, a Logits step costs 4. Host routing
    /// is the parity reference: a Prompt step costs 8, a Logits step costs 12 (`step.rs`,
    /// `each_mode_uses_its_synchronization_budget`). A prompt of 3 tokens feeds 2 tokens by Prompt
    /// steps and keeps the last pending: twice the Prompt cost. Each decode feeds one pending token
    /// by a Logits step. Each routing gives a different count for each mode, so a wrong mode fails
    /// under both routings.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn the_executor_uses_prompt_steps_for_the_prompt_and_logits_steps_for_decode() {
        use super::super::moe::Routing;
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(
            executor.decoder.routing,
            Routing::Device,
            "the executor starts with device routing"
        );
        let count = |executor: &Q36Executor| executor.decoder.kernels.qualification_sync_count();
        for (routing, prompt_cost, logits_cost) in [(Routing::Device, 2, 4), (Routing::Host, 8, 12)]
        {
            executor.decoder.set_routing(routing);
            executor.reset(&mut session).unwrap();
            let mut job = seeded_generation(3);
            let before = count(&executor);
            executor.prefill(&mut session, &PROMPT).unwrap();
            assert_eq!(
                count(&executor) - before,
                2 * prompt_cost,
                "{routing:?} prefill: 2 Prompt steps"
            );
            for call in 0..3 {
                let before = count(&executor);
                let batch = executor.decode(&mut session, &mut job);
                assert!(
                    batch.is_ok(),
                    "{routing:?} decode {call}: the Logits step must return logits: {:?}",
                    batch.as_ref().err()
                );
                assert_eq!(
                    count(&executor) - before,
                    logits_cost,
                    "{routing:?} decode {call}: one Logits step"
                );
            }
        }
    }

    #[test]
    fn an_engine_thread_builds_the_executor_runs_two_jobs_and_shuts_down() {
        use crate::sched::{
            job::{Job, JobEvent, JobUsage, PrefixReport},
            worker::{EngineState, ShutdownReport, spawn_engine},
        };
        let _serial = crate::native_test_lock();
        const COUNT: usize = 4;
        let prompt_b = vec![6, 4, 7, 1];
        let want_a = oracle(&PROMPT, COUNT, &mut seeded_sampler(), VOCAB);
        let want_b = oracle(&prompt_b, COUNT, &mut seeded_sampler(), VOCAB);
        let engine = spawn_engine(2, || Ok(build(rules(vec![], VOCAB)))).unwrap();
        for (round, (prompt, want)) in [
            (PROMPT.to_vec(), &want_a),
            (prompt_b.clone(), &want_b),
            (PROMPT.to_vec(), &want_a),
        ]
        .into_iter()
        .enumerate()
        {
            let prompt_tokens = u32::try_from(prompt.len()).unwrap();
            let (job, mut handle) = Job::new(prompt, seeded_generation(4)).unwrap();
            engine.handle().try_submit(job).unwrap();
            let mut tokens = Vec::new();
            let mut usage = None;
            while let Some(event) = handle.events.blocking_recv() {
                match event {
                    JobEvent::Batch(batch) => tokens.extend(batch.tokens),
                    JobEvent::Finished(done) => usage = Some(done),
                    JobEvent::Error(error) => panic!("round {round}: {error:?}"),
                }
            }
            assert_eq!(&tokens, want, "round {round}: tokens");
            assert_eq!(
                usage,
                Some(JobUsage {
                    prompt_tokens,
                    completion_tokens: 4,
                    reused_tokens: 0,
                    prefix: Some(PrefixReport::default())
                }),
                "round {round}: usage"
            );
        }
        assert_eq!(
            engine.shutdown(),
            ShutdownReport {
                state: EngineState::Stopped,
                failure: None
            }
        );
    }

    /// The real checkpoint on the Mac: the executor with a greedy sampler must give the 32 greedy
    /// ids of the frozen host-reference evidence for its 5-token prompt. The coordinator runs it
    /// with `--ignored --exact` under the lock. Environment: `Q36_CKPT_DIR` (the pinned checkpoint
    /// with `tokenizer.json`), `Q36_REF_DIR` (the directory of `host-ref.json`), `Q36_CACHE_LIMIT`
    /// (bytes), `Q36_EXECUTOR_REPORT` (the report file).
    #[test]
    #[ignore = "needs the pinned Q36 checkpoint and the frozen host reference on the Mac"]
    fn the_executor_serves_the_real_checkpoint_with_the_frozen_greedy_ids() {
        let _serial = crate::native_test_lock();
        let (ckpt_dir, ref_dir) = (env_path("Q36_CKPT_DIR"), env_path("Q36_REF_DIR"));
        let report = env_path("Q36_EXECUTOR_REPORT");
        let cache_limit: u64 = std::env::var("Q36_CACHE_LIMIT")
            .expect("Q36_CACHE_LIMIT is not set")
            .parse()
            .expect("Q36_CACHE_LIMIT is not a byte count");
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(ref_dir.join("host-ref.json")).unwrap()).unwrap();
        let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(reference["greedy_ids"].clone()).unwrap();
        assert_eq!((ids.len(), want.len()), (5, 32), "the frozen lengths");
        let tokenizer =
            Tokenizer::from_bytes(&std::fs::read(ckpt_dir.join("tokenizer.json")).unwrap())
                .unwrap();
        // No EOS ids: the frozen chain is compared id for id.
        let rules = Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab: 248_320,
            context_limit: 256,
            session_cap: 1 << 30,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        };
        let (mut executor, mut session) = Q36Executor::open(&ckpt_dir, cache_limit, rules).unwrap();
        let mut job = GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(32);
        let started = std::time::Instant::now();
        executor.prefill(&mut session, &ids).unwrap();
        let prefill_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let batches = decode_all(&mut executor, &mut session, &mut job);
        let decode_seconds = started.elapsed().as_secs_f64();
        let got = tokens_of(&batches);
        let join = |tokens: &[u32]| {
            tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        std::fs::write(
            &report,
            format!(
                "executor_tokens={}\nfrozen_tokens={}\nprefill_seconds={prefill_seconds}\n\
                 decode_seconds={decode_seconds}\ndecode_calls={}\n",
                join(&got),
                join(&want),
                batches.len(),
            ),
        )
        .unwrap();
        assert_eq!(
            got, want,
            "executor tokens differ from the frozen greedy ids"
        );
        assert_eq!(batches.len(), 32, "decode calls");
        assert_eq!(
            batches.last().and_then(|b| b.finish),
            Some(DecodeFinish::Length),
            "the last batch must end with Length"
        );
    }

    /// A byte count from the environment variable `name`, or `default` when it is unset.
    fn env_u64(name: &str, default: Option<u64>) -> u64 {
        std::env::var(name).map_or_else(
            |_| default.unwrap_or_else(|| panic!("set {name}")),
            |value| {
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("{name} is not a number"))
            },
        )
    }

    /// The `session_reserve` of a real run: the session state at `positions` (the DeltaNet state
    /// plus the cache growth of each position), the MLX cache limit, and a pressure margin of 1
    /// GiB.
    fn real_reserve(
        executor: &Q36Executor,
        session: &Q36Session,
        positions: u64,
        cache_limit: u64,
    ) -> u64 {
        // The form and the activation of the run's own decoder: an F32 control run and a Q8 run
        // each reserve their own cache bytes.
        let act = Activation::of_dtype(executor.decoder.weights.consts.act).unwrap();
        let growth = growth_per_position(
            &executor.decoder.weights.shape,
            executor.decoder.weights.kv,
            act,
        )
        .unwrap();
        session.state.byte_len().unwrap() + growth * positions + cache_limit + (1 << 30)
    }

    /// The real checkpoint through the executor with the chunk path off and then on. The 5-token
    /// prompt of the frozen host-reference evidence feeds 4 rows. With the path off the 32 greedy
    /// ids must equal the frozen ones. With a planned size the 4 rows are one chunk, and all 32
    /// greedy ids must equal the frozen ones, the rule that the Flash-Next executor test uses. The
    /// report names the first difference and the number of equal ids. The coordinator runs it with
    /// `--ignored --exact` under the lock. Environment: `Q36_CKPT_DIR` (with `tokenizer.json`),
    /// `Q36_REF_DIR` (with `host-ref.json`), `Q36_CACHE_LIMIT` (bytes), `Q36_BUDGET` (bytes,
    /// default 32,212,254,720), `Q36_ACTIVATION` (`f32` | `bf16`, unset the F32 control; read
    /// through `real::activation_from_env`, the one definition), and `Q36_EXECUTOR_REPORT` (the
    /// report file).
    #[test]
    #[ignore = "needs the pinned Q36 checkpoint and the frozen host reference on the Mac"]
    fn the_chunked_prefill_serves_the_real_checkpoint_like_the_one_token_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt_dir, ref_dir) = (env_path("Q36_CKPT_DIR"), env_path("Q36_REF_DIR"));
        let report = env_path("Q36_EXECUTOR_REPORT");
        let cache_limit = env_u64("Q36_CACHE_LIMIT", None);
        let budget = env_u64("Q36_BUDGET", Some(32_212_254_720));
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(ref_dir.join("host-ref.json")).unwrap()).unwrap();
        let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(reference["greedy_ids"].clone()).unwrap();
        assert_eq!((ids.len(), want.len()), (5, 32), "the frozen lengths");
        let tokenizer =
            Tokenizer::from_bytes(&std::fs::read(ckpt_dir.join("tokenizer.json")).unwrap())
                .unwrap();
        // No EOS ids: the frozen chain is compared id for id.
        let rules = Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab: 248_320,
            context_limit: 256,
            session_cap: 1 << 30,
            max_seqs: 1,
            activation: activation_from_env(),
            kv_dtype: kv_dtype_from_env(),
            compile: false,
        };
        let dtype = activation_name(rules.activation);
        let (mut executor, mut session) = Q36Executor::open(&ckpt_dir, cache_limit, rules).unwrap();
        let run = |executor: &mut Q36Executor, session: &mut Q36Session| {
            executor.reset(session).unwrap();
            let mut job =
                GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(32);
            let started = std::time::Instant::now();
            executor.prefill(session, &ids).unwrap();
            let prefill_seconds = started.elapsed().as_secs_f64();
            (
                tokens_of(&decode_all(executor, session, &mut job)),
                prefill_seconds,
            )
        };
        assert_eq!(executor.prefill_chunk(), None, "the path starts off");
        let (off, off_seconds) = run(&mut executor, &mut session);
        assert!(take_log(&executor).is_empty(), "no chunk with the path off");
        let reserve = real_reserve(&executor, &session, 256, cache_limit);
        let plan = executor.plan_prefill(budget, reserve, 16).unwrap();
        let (on, on_seconds) = run(&mut executor, &mut session);
        let chunks = take_log(&executor);
        let join = |tokens: &[u32]| {
            tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let first_difference = on.iter().zip(&want).position(|(a, b)| a != b);
        let equal = on.iter().zip(&want).filter(|(a, b)| a == b).count();
        std::fs::write(
            &report,
            format!(
                "activation_dtype={dtype}\n{}planned_chunk={:?}\nfrozen_tokens={}\n\
                 off_tokens={}\non_tokens={}\n\
                 off_prefill_seconds={off_seconds}\non_prefill_seconds={on_seconds}\n\
                 on_chunk_rows={chunks:?}\non_first_difference={first_difference:?}\n\
                 on_equal_ids={equal}\n",
                plan.terms,
                plan.chunk,
                join(&want),
                join(&off),
                join(&on),
            ),
        )
        .unwrap();
        assert_eq!(off, want, "the steps differ from the frozen greedy ids");
        assert_eq!(plan.chunk, Some(16), "the budget admits the largest piece");
        assert_eq!(chunks, vec![4], "the four prompt rows are one chunk");
        assert_eq!(on.len(), 32, "decoded ids");
        assert_eq!(
            on, want,
            "the chunk path differs from the frozen greedy ids"
        );
    }

    /// SHA-256 of the first 2,048 prompt ids of the reference run as little-endian words
    /// (`q36-dr4-prompt-ids.py`).
    const PROMPT_2048_SHA256: &str =
        "a2176ac355b06d384c0b8b02807d4e0f9dfcc3659c43531148e75dfe70c98180";

    /// The prefill rate of the frozen 2,048-token prompt through the executor (the prompt gate is
    /// 60 prompt tokens per second). The test runs the prompt once for each chunk size in
    /// `Q36_CHUNKS` (default `256,512`; `0` is the one-token path and is slow) in worker pieces of
    /// that size, and reports seconds, tokens per second, chunk runs, the allocator peak against
    /// the planned workspace, and in a qualification build the synchronizations, the forced
    /// flushes, and the peak retired bytes. It asserts no speed. Every size must be admitted by the
    /// plan of `Q36_BUDGET` and at most 512 (the largest worker piece that the settings accept).
    /// The coordinator runs it with `--ignored --exact` under the exclusive lock. Environment:
    /// `Q36_CKPT_DIR` (with `tokenizer.json`), `Q36_PROMPT_IDS` (the file of little-endian `u32`
    /// ids that `q36-dr4-prompt-ids.py` writes), `Q36_CACHE_LIMIT`, `Q36_BUDGET` (default
    /// 32,212,254,720), `Q36_CHUNKS`, `Q36_ACTIVATION` (`f32` | `bf16`, unset the F32 control;
    /// `real::activation_from_env`, the one definition), and `Q36_EXECUTOR_REPORT`.
    #[test]
    #[ignore = "needs the pinned Q36 checkpoint and the frozen 2,048-token prompt on the Mac"]
    fn the_chunked_prefill_reports_the_rate_on_the_frozen_2048_token_prompt() {
        use sha2::{Digest, Sha256};
        let _serial = crate::native_test_lock();
        let ckpt_dir = env_path("Q36_CKPT_DIR");
        let report = env_path("Q36_EXECUTOR_REPORT");
        let cache_limit = env_u64("Q36_CACHE_LIMIT", None);
        let budget = env_u64("Q36_BUDGET", Some(32_212_254_720));
        let sizes: Vec<u32> = std::env::var("Q36_CHUNKS")
            .unwrap_or_else(|_| "256,512".to_string())
            .split(',')
            .map(|size| {
                size.trim()
                    .parse()
                    .expect("Q36_CHUNKS is not a number list")
            })
            .collect();
        let file = std::fs::read(env_path("Q36_PROMPT_IDS")).unwrap();
        let bytes = file
            .get(..2_048 * 4)
            .expect("the prompt file holds fewer than 2,048 ids");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            PROMPT_2048_SHA256,
            "the first 2,048 ids differ from the reference prompt"
        );
        let prompt: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let tokenizer =
            Tokenizer::from_bytes(&std::fs::read(ckpt_dir.join("tokenizer.json")).unwrap())
                .unwrap();
        let rules = Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab: 248_320,
            context_limit: 4_096,
            session_cap: 4_096 * 40_960,
            max_seqs: 1,
            activation: activation_from_env(),
            kv_dtype: kv_dtype_from_env(),
            compile: false,
        };
        let dtype = activation_name(rules.activation);
        let (mut executor, mut session) = Q36Executor::open(&ckpt_dir, cache_limit, rules).unwrap();
        let reserve = real_reserve(&executor, &session, 4_096, cache_limit);
        let plan = executor.plan_prefill(budget, reserve, 512).unwrap();
        let mut lines = format!(
            "activation_dtype={dtype}\n{}planned_chunk={:?}\nprompt_sha256={PROMPT_2048_SHA256}\n\
             prompt_tokens={}\n\
             note=tokens_per_second divides all prompt tokens by the time, but the last token stays \
             pending, so the run feeds one row fewer\n",
            plan.terms,
            plan.chunk,
            prompt.len()
        );
        // Write the plan before any failure: a plan that admits no chunk is a result to report.
        std::fs::write(&report, &lines).unwrap();
        for &size in &sizes {
            if size == 0 {
                executor.force_chunk(None);
            } else {
                assert!(size <= 512, "chunk {size} passes the largest worker piece");
                assert!(
                    plan.chunk.is_some_and(|planned| size <= planned),
                    "chunk {size} passes the plan {:?}",
                    plan.chunk
                );
                executor.force_chunk(Some(size));
            }
            executor.reset(&mut session).unwrap();
            take_log(&executor);
            let kernels = &mut executor.decoder.kernels;
            kernels.reset_peak_memory().unwrap();
            let active_before = kernels.memory_stats().unwrap().active;
            #[cfg(feature = "native-fault-qualification")]
            let (syncs, flushes) = {
                kernels.qualification_reset_retire_peak();
                (
                    kernels.qualification_sync_count(),
                    kernels.qualification_retire_stats().0,
                )
            };
            let piece = usize::try_from(if size == 0 { 16 } else { size }).unwrap();
            let started = std::time::Instant::now();
            for part in prompt.chunks(piece) {
                executor.prefill(&mut session, part).unwrap();
            }
            let seconds = started.elapsed().as_secs_f64();
            assert_eq!(session.core.history(), prompt.as_slice(), "{size}: history");
            assert_eq!(session.state.pos(), 2_047, "{size}: fed positions");
            assert_eq!(
                executor.decoder.session_used(),
                2_047 * 40_960,
                "{size}: charge"
            );
            let runs = take_log(&executor);
            let memory = executor.decoder.kernels.memory_stats().unwrap();
            let workspace = if size > 1 {
                executor.prefill_cost().unwrap().workspace(size).unwrap()
            } else {
                0
            };
            lines.push_str(&format!(
                "chunk_{size}_seconds={seconds}\nchunk_{size}_tokens_per_second={}\n\
                 chunk_{size}_chunk_runs={}\nchunk_{size}_first_runs={:?}\n\
                 chunk_{size}_mlx_peak_bytes={}\nchunk_{size}_mlx_active_bytes={}\n\
                 chunk_{size}_mlx_cache_bytes={}\nchunk_{size}_mlx_peak_over_start_bytes={}\n\
                 chunk_{size}_planned_workspace_bytes={workspace}\n",
                prompt.len() as f64 / seconds,
                runs.len(),
                runs.iter().take(3).collect::<Vec<_>>(),
                memory.peak,
                memory.active,
                memory.cache,
                memory.peak.saturating_sub(active_before),
            ));
            #[cfg(feature = "native-fault-qualification")]
            {
                let kernels = &executor.decoder.kernels;
                let (total, peak) = kernels.qualification_retire_stats();
                lines.push_str(&format!(
                    "chunk_{size}_syncs={}\nchunk_{size}_forced_flushes={}\n\
                     chunk_{size}_peak_retired_bytes={peak}\n",
                    kernels.qualification_sync_count() - syncs,
                    total - flushes,
                ));
            }
            // Four greedy ids after the prompt, so a run can be compared with the other sizes.
            let mut job =
                GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(4);
            let next = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
            lines.push_str(&format!("chunk_{size}_next_tokens={next:?}\n"));
            std::fs::write(&report, &lines).unwrap();
        }
    }

    /// A prompt of 6 tokens whose first 4 tokens form the saved key.
    const PREFIX_PROMPT: [u32; 6] = [1, 2, 3, 4, 5, 6];
    /// The key length of the prefix tests.
    const KEY: usize = 4;
    /// The context limit of the prefix tests: a 6-token prompt and 6 logits steps (the pending
    /// token and `FORCED`) feed 11 positions, and one more stays free.
    const PREFIX_LIMIT: u32 = 12;
    /// The DeltaNet bytes that a snapshot charges and the steps do not: per DeltaNet layer the
    /// recurrent state `[2, 32, 32]` is 8,192 bytes and the convolution ring `[128, 3]` is 1,536
    /// (the fixture numbers of `state.rs`).
    const FIXED_BYTES: u64 = 2 * (2 * 32 * 32 * 4 + 128 * 3 * 4);
    /// An entry charges the fixed DeltaNet bytes plus the page-table allowance of its alias - one
    /// 16-byte page covers every fixture position count here.
    const TABLE: u64 = 16;

    /// The rules of the prefix tests: `PREFIX_LIMIT` positions and room for a live session and one
    /// saved entry (an entry charges `FIXED_BYTES` plus the page-table `TABLE` allowance).
    fn prefix_rules() -> Q36Rules {
        Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab: VOCAB,
            context_limit: PREFIX_LIMIT,
            session_cap: 100_000,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        }
    }

    /// The rules of the budget-denial tests: a cap of 28,000 bytes. It admits the save of
    /// `save_key` (3 * GROWTH + FIXED_BYTES + TABLE = 22,544) and the fed positions that the
    /// tests add around it, but not the denied call that each test names.
    fn denial_rules() -> Q36Rules {
        Q36Rules {
            prefix_store: StoreLimits::default(),
            eos: vec![],
            sampler_vocab: VOCAB,
            context_limit: PREFIX_LIMIT,
            session_cap: 25_000,
            max_seqs: 1,
            activation: Activation::F32,
            kv_dtype: KvDtype::F32,
            compile: false,
        }
    }

    /// Saves `PREFIX_PROMPT[..KEY]` in a reset session: prefill the key, then save it.
    fn save_key(executor: &mut Q36Executor, session: &mut Q36Session) -> bool {
        executor.reset(session).unwrap();
        executor.prefill(session, &PREFIX_PROMPT[..KEY]).unwrap();
        executor
            .save_prefix(session, &PREFIX_PROMPT[..KEY], SaveKind::Tip)
            .unwrap()
    }

    /// The largest absolute difference of two equal-length rows.
    fn max_gap(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn a_restored_prefix_continues_like_a_full_prefill() {
        let _serial = crate::native_test_lock();
        // The seeded sampler reads the whole logits row, so a restored state that differs
        // changes the draws. The greedy chain of the synthetic model is constant.
        let want = oracle(&PREFIX_PROMPT, 4, &mut seeded_sampler(), VOCAB);
        let (mut executor, mut session) = build(prefix_rules());
        assert!(
            save_key(&mut executor, &mut session),
            "the save was refused"
        );
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        let mut first = seeded_generation(4);
        assert_eq!(
            tokens_of(&decode_all(&mut executor, &mut session, &mut first)),
            want,
            "tokens of the first job"
        );
        executor.reset(&mut session).unwrap();
        let covered = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(covered, KEY, "covered tokens");
        assert_eq!(
            executor.history(&session),
            &PREFIX_PROMPT[..KEY],
            "restored history"
        );
        assert_eq!(session.core.pending(), Some(4), "restored pending token");
        // Three of the four key tokens are fed. The fourth is pending.
        assert_eq!(session.state.pos(), 3, "fed positions");
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        let mut second = seeded_generation(4);
        assert_eq!(
            tokens_of(&decode_all(&mut executor, &mut session, &mut second)),
            want,
            "tokens of the restored job"
        );
    }

    #[test]
    fn a_restored_prefix_gives_bitwise_equal_logits_after_the_suffix() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(prefix_rules());
        let want = run_job(&mut executor, &mut session, &PREFIX_PROMPT, 6);
        // A prompt with other tokens gives other rows, so equal rows below cannot come from
        // state that ignores the tokens.
        let other = run_job(&mut executor, &mut session, &[6, 4, 7, 1, 5, 3], 6);
        assert_ne!(
            bits(&want.rows[0]),
            bits(&other.rows[0]),
            "the two prompts must give different logits, or the test is vacuous"
        );
        let gap = max_gap(&want.rows[0], &other.rows[0]);
        assert!(
            gap > 1e-4,
            "the two keys differ by only {gap} in the logits; the fixture needs a \
             token-dependent embedding table"
        );
        assert!(
            save_key(&mut executor, &mut session),
            "the save was refused"
        );
        executor.reset(&mut session).unwrap();
        let covered = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(covered, KEY, "covered tokens");
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        let pending = session.core.pending().unwrap();
        // One-token steps are the path with bitwise equality.
        for (n, token) in std::iter::once(pending).chain(FORCED).enumerate() {
            let row = executor
                .decoder
                .step(&mut session.state, token, StepMode::Logits)
                .unwrap()
                .logits
                .unwrap();
            assert_eq!(bits(&row), bits(&want.rows[n]), "logits row {n}");
        }
    }

    #[test]
    fn a_denied_save_returns_false_and_keeps_no_entry() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(prefix_rules());
        // Cap 10,000. Three fed positions hold 3,072 bytes. The snapshot needs
        // 3,072 + FIXED_BYTES = 22,528 and the cap denies it.
        executor
            .decoder
            .set_session_budget(SessionBudget::new(10_000));
        assert!(
            !save_key(&mut executor, &mut session),
            "a denied charge reports false"
        );
        assert_eq!(executor.prefix.key_len(), None, "no entry");
        assert_eq!(
            executor.decoder.session_used(),
            3 * GROWTH,
            "only the session is charged"
        );
    }

    #[test]
    fn a_second_save_fits_only_because_the_first_entry_drops_first() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(prefix_rules());
        // Cap 30,000. The second save needs 5,120 + 19,472 = 24,592 bytes after the first
        // entry drops. With the first entry kept it needs 24,592 + 19,472 = 44,064 and the cap
        // denies it.
        executor
            .decoder
            .set_session_budget(SessionBudget::new(30_000));
        assert!(save_key(&mut executor, &mut session), "first save");
        // Three fed positions: session 3,072 + snapshot (FIXED_BYTES + TABLE).
        assert_eq!(
            executor.decoder.session_used(),
            3 * GROWTH + FIXED_BYTES + TABLE,
            "after the first save"
        );
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        // Five fed positions: session 5,120 + the first snapshot 22,528.
        assert_eq!(
            executor.decoder.session_used(),
            5 * GROWTH + FIXED_BYTES + TABLE,
            "before the second save"
        );
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT, SaveKind::Tip)
                .unwrap(),
            "second save"
        );
        assert_eq!(
            executor.decoder.session_used(),
            5 * GROWTH + FIXED_BYTES + TABLE,
            "after the second save"
        );
        assert_eq!(
            executor.prefix.key_len(),
            Some(6),
            "the entry holds the new key"
        );
    }

    /// A restore that the session budget denies drops the entry and changes nothing. The denial
    /// comes from the cap, never from a `SessionBudget` swap under live state: a swap after a save
    /// orphans the entry's permit and breaks the rollback re-acquire of the target's charge, which
    /// then leaves the state with its layers but empty permits and the next commit fails its charge
    /// check.
    ///
    /// The cap (25,000, `denial_rules`) is fixed before the save. The save needs 22,544 bytes
    /// (the saver's three key positions 3 * GROWTH plus the entry FIXED_BYTES + TABLE) and
    /// the target's pre-held permit takes 1,024 more (23,568). The restore then wants the key
    /// cache (3 * GROWTH) beside the entry (26,640) and is denied, while the rollback of the
    /// target's 1,024 bytes and one more decode step still fit.
    ///
    /// Production reachability: the reason for the denial (the entry's charge plus its restored
    /// cache do not fit the free session cap) is the reason a real restore is denied when
    /// `session_cap_bytes` is too small. The arrangement that reaches the branch here with a
    /// competing live charge. Default single-session serving cannot hold this second session;
    /// opt-in concurrent sessions can hold competing charges.
    #[test]
    fn a_restore_denied_by_the_budget_drops_the_entry_and_changes_nothing() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut saver) = build(denial_rules());
        assert!(save_key(&mut executor, &mut saver), "the save was refused");
        // Three fed positions (3 * GROWTH) hold the saver beside the entry
        // (FIXED_BYTES + TABLE).
        assert_eq!(
            executor.decoder.session_used(),
            3 * GROWTH + FIXED_BYTES + TABLE,
            "the save charges the entry beside the saver"
        );
        // The restore target is a second session constructed with a pre-held permit: one fed
        // position (GROWTH) charged to the same budget and kept through the restore attempt.
        let mut target = Q36Session {
            state: executor.decoder.zero_state().unwrap(),
            core: DecodeSession::default(),
            _fixed_state: None,
        };
        executor.prefill(&mut target, &[5, 6]).unwrap();
        assert_eq!(
            executor.decoder.session_used(),
            3 * GROWTH + FIXED_BYTES + TABLE + GROWTH,
            "the target holds its pre-held permit"
        );
        let covered = executor
            .restore_prefix(&mut target, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(covered, 0, "covered tokens");
        assert_eq!(
            executor.prefix.key_len(),
            None,
            "the denied restore drops the entry"
        );
        assert_eq!(target.state.pos(), 1, "the target keeps its position");
        assert_eq!(
            executor.history(&target),
            &[5u32, 6][..],
            "the target keeps its history"
        );
        assert_eq!(
            target.core.pending(),
            Some(6),
            "the target keeps its pending token"
        );
        // The rollback re-acquired the target's 1,024 bytes, so its permits still equal its cache.
        assert_eq!(
            executor.decoder.session_used(),
            3 * GROWTH + GROWTH,
            "the denial drops the entry and keeps the two live charges"
        );
        // The next decode step on the unchanged state still commits.
        let row = executor
            .decoder
            .step(&mut target.state, 7, StepMode::Logits)
            .unwrap();
        assert!(row.logits.is_some(), "the decode step returns logits");
        assert_eq!(target.state.pos(), 2, "the decode step committed");
    }

    /// A save of a new key makes room first (the documented rule of `DecodeSession::save_prefix`:
    /// with one entry place the earlier entry drops before the snapshot), and a denial then
    /// evicts the remaining victims, so the store is empty afterwards and the next job recomputes
    /// its whole prompt instead of restoring anything.
    #[test]
    fn a_denied_save_while_an_older_entry_exists_leaves_the_slot_empty_and_the_next_job_recomputes()
    {
        let _serial = crate::native_test_lock();
        // Page sharing collapsed the old short-prompt arrangement: an entry charges the same fixed
        // bytes plus page allowance as the snapshot replacing it, so no cap can fit the
        // prompt-with-entry peak and still deny the save after the entry drops - the two peaks are
        // equal. The one reachable budget denial that empties a store sits at a page boundary
        // (PAGE_POSITIONS = 2,048): the longer key crosses into a second page-table row, so its
        // snapshot (FIXED_BYTES + 2 * 16 = 19,488) is strictly bigger than the older entry's
        // (FIXED_BYTES + 16 = 19,472). The long prompt feeds 2,053 positions (2,053 * GROWTH =
        // 2,102,272 live): with the older entry the peak is 2,121,744 and the cap of 2,121,750 fits
        // it, while the tip save after the retry evicts the older entry needs 2,102,272 + 19,488 =
        // 2,121,760 and is denied - the exact 16-byte page step is the whole window.
        let long: Vec<u32> = (0..2_054u32).map(|i| i % 8).collect();
        let mut denial = denial_rules();
        denial.session_cap = 2_121_750;
        denial.context_limit = 2_060;
        let (mut executor, mut session) = build(denial);
        executor.prefill(&mut session, &long[..KEY]).unwrap();
        assert!(
            executor
                .save_prefix(&session, &long[..KEY], SaveKind::Tip)
                .unwrap(),
            "the first save"
        );
        assert_eq!(
            executor.prefix.key_len(),
            Some(KEY),
            "the first entry is stored"
        );
        executor.prefill(&mut session, &long[KEY..]).unwrap();
        assert!(
            !executor
                .save_prefix(&session, &long, SaveKind::Tip)
                .unwrap(),
            "the second save is denied"
        );
        assert_eq!(
            executor.prefix.key_len(),
            None,
            "the store is empty: the older entry dropped first"
        );
        assert_eq!(
            executor.decoder.session_used(),
            2_053 * GROWTH,
            "the denied save charges nothing"
        );
        // The next job restores nothing and recomputes the whole prompt.
        executor.reset(&mut session).unwrap();
        let covered = executor.restore_prefix(&mut session, &long).unwrap();
        assert_eq!(covered, 0, "covered tokens");
        assert!(
            executor.history(&session).is_empty(),
            "the reset job holds no tokens"
        );
        executor.prefill(&mut session, &long).unwrap();
        assert_eq!(
            session.state.pos(),
            2_053,
            "the next job recomputed the whole prompt"
        );
    }

    /// At the real executor, stored entries charge the budget through their snapshots' permits, and
    /// each `evict_one_prefix_entry` step drops one LRU entry - the restored one included - and
    /// releases exactly their charges, so a dead conversation stops pinning the pool. A pinned root
    /// keeps its charge. The tip the LIVE session saved releases only its fixed and page-table
    /// bytes: its KV permits are shared with the session, which still holds them.
    #[test]
    fn evicting_the_store_releases_the_entries_charges_and_keeps_roots() {
        let _serial = crate::native_test_lock();
        let rules = Q36Rules {
            prefix_store: StoreLimits {
                roots: 1,
                entries: 2,
            },
            ..prefix_rules()
        };
        let (mut executor, mut session) = build(rules);
        executor
            .decoder
            .set_session_budget(SessionBudget::new(200_000));
        // A root of the first two tokens, then two divergent tips.
        executor.reset(&mut session).unwrap();
        executor.prefill(&mut session, &PREFIX_PROMPT[..2]).unwrap();
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT[..2], SaveKind::Root)
                .unwrap(),
            "the root save"
        );
        assert!(save_key(&mut executor, &mut session), "the first tip");
        executor.reset(&mut session).unwrap();
        let second = [7u32, 7, 7, 7];
        executor.prefill(&mut session, &second).unwrap();
        assert!(
            executor
                .save_prefix(&session, &second, SaveKind::Tip)
                .unwrap(),
            "the second tip"
        );
        let before = executor.decoder.session_used();
        // The incremental denial order empties the LRU one entry at a time; the total over
        // the loop is the old clear-all's single report.
        let mut freed = 0;
        while let Some(released) = executor.evict_one_prefix_entry(None) {
            freed += released;
        }
        // The dead conversation's tip releases everything: its key holds 4 tokens = 3 fed
        // positions of KV (3 * GROWTH) plus fixed and page-table bytes. The live session's
        // tip shares its KV permits with the session, so only its fixed and page-table
        // bytes release.
        assert_eq!(
            freed,
            3 * GROWTH + FIXED_BYTES + TABLE + FIXED_BYTES + TABLE,
            "the dead tip releases fully; the live session's tip keeps its shared KV"
        );
        assert_eq!(
            executor.decoder.session_used(),
            before - freed,
            "the root and the live session keep their charges"
        );
        assert_eq!(executor.prefix.roots_len(), 1, "the root stays pinned");
        assert_eq!(executor.prefix.entries_len(), 0, "the LRU entries are gone");
    }

    /// A planned chunk of one row is the one-token path, not `Chunked(1)`.
    #[test]
    fn a_planned_chunk_of_one_token_takes_the_steps_path() {
        assert_eq!(
            PrefillPath::of(Some(1)),
            PrefillPath::Steps,
            "the path of a chunk of one token"
        );
    }

    /// No planned chunk is the one-token path.
    #[test]
    fn no_planned_chunk_takes_the_steps_path() {
        assert_eq!(
            PrefillPath::of(None),
            PrefillPath::Steps,
            "the path without a planned chunk"
        );
    }

    #[test]
    fn a_path_or_chunk_size_mismatch_refuses_the_restore_and_keeps_the_entry() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(prefix_rules());
        // The one-token path saves the entry.
        assert!(
            save_key(&mut executor, &mut session),
            "the steps save was refused"
        );
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        executor.reset(&mut session).unwrap();
        let chunked = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(chunked, 0, "the chunk path refuses a steps entry");
        assert!(executor.history(&session).is_empty(), "history");
        assert_eq!(executor.prefix.key_len(), Some(KEY), "the entry stays");
        // The chunk path saves its own entry under the size 4.
        assert!(
            save_key(&mut executor, &mut session),
            "the chunk save was refused"
        );
        for (name, chunk) in [
            ("a smaller size", Some(3)),
            ("a larger size", Some(5)),
            ("one-token steps", None),
        ] {
            executor.force_chunk(chunk);
            executor.reset(&mut session).unwrap();
            let covered = executor
                .restore_prefix(&mut session, &PREFIX_PROMPT)
                .unwrap();
            assert_eq!(covered, 0, "{name}: covered tokens");
            assert!(executor.history(&session).is_empty(), "{name}: history");
            assert_eq!(
                executor.prefix.key_len(),
                Some(KEY),
                "{name}: the entry stays"
            );
        }
        executor.force_chunk(Some(4));
        let same = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(
            same, KEY,
            "covered tokens under the size that built the entry"
        );
    }

    #[test]
    fn a_save_is_refused_when_the_model_is_not_at_the_position_of_the_key() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(prefix_rules());
        executor.reset(&mut session).unwrap();
        executor
            .prefill(&mut session, &PREFIX_PROMPT[..KEY])
            .unwrap();
        // One more position reaches the native state without reaching the session, as a failed
        // prefill that fed part of its rows would leave it.
        executor
            .decoder
            .step(&mut session.state, 5, StepMode::Prompt)
            .unwrap();
        let refused = executor.save_prefix(&session, &PREFIX_PROMPT[..KEY], SaveKind::Tip);
        assert!(
            matches!(refused, Err(EngError::BadRequest(_))),
            "{refused:?}"
        );
        assert_eq!(executor.prefix.key_len(), None, "no entry");
        // Each committed step charges one position of cache growth (GROWTH bytes in the fixture).
        // The key prefill fed KEY - 1 = 3 positions (the last key token stays pending) and the
        // extra step fed one more, so 4 steps are charged. The refused save took no snapshot.
        assert_eq!(
            executor.decoder.session_used(),
            KEY as u64 * GROWTH,
            "four fed positions and no snapshot"
        );
    }

    /// The default rules keep the tip store: every executor built without the `prefix_reuse` flag
    /// reports zero roots and one entry, so the worker saves only the tip. The literals pin the
    /// value, not the `Default` under test.
    #[test]
    fn the_default_rules_keep_the_addendum_one_store_limits() {
        let _serial = crate::native_test_lock();
        let (executor, _session) = build(rules(vec![], VOCAB));
        assert_eq!(
            executor.prefix_limits(),
            StoreLimits {
                roots: 0,
                entries: 1
            },
            "zero roots and one entry"
        );
    }

    /// A `prefix_reuse` engine holds the reuse set: the rules carry the limits and the executor
    /// reports them, so the worker emits the root save point and the store keeps two pinned roots
    /// beside six LRU entries.
    ///
    /// Mutation check (the limits-ignored patch): building the store with
    /// `PrefixSlot::default()` in `from_parts` makes the report read `(0, 1)` and fails.
    #[test]
    fn a_prefix_reuse_engine_reports_two_roots_and_six_entries() {
        let _serial = crate::native_test_lock();
        let (executor, _session) = build(Q36Rules {
            prefix_store: StoreLimits {
                roots: 2,
                entries: 6,
            },
            ..prefix_rules()
        });
        assert_eq!(
            executor.prefix_limits(),
            StoreLimits {
                roots: 2,
                entries: 6
            },
            "the executor holds the reuse set"
        );
    }

    /// A root is best effort inside the session cap. A denied root save skips only that save: the
    /// store stays empty, the denial charges nothing, and the job runs on to its tip save and its
    /// decode like a job without a root point. The charge grows with the position — the tip
    /// snapshot of a longer key needs strictly more bytes than the root snapshot beside the same
    /// live session — so no single cap can deny the earlier root and admit the later tip; the test
    /// raises the cap between the two saves, after the reset that the `set_session_budget` seam
    /// requires, at a point where no entry is stored (the root was skipped).
    #[test]
    fn a_denied_root_save_skips_only_that_save_and_the_job_still_succeeds() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PREFIX_PROMPT, 4, &mut seeded_sampler(), VOCAB);
        let (mut executor, mut session) = build(Q36Rules {
            prefix_store: StoreLimits {
                roots: 2,
                entries: 6,
            },
            ..prefix_rules()
        });
        // The tight cap under page sharing: the root snapshot charges the fixed DeltaNet bytes plus
        // a page-table allowance (never the KV of its positions), so the exact charge moves with
        // the allowance. The denial is anchored on the invariant the the sharing keeps: fixed_bytes
        // alone (19,456 for this fixture) exceeds the whole 10,000 cap. The continuing job still
        // fits - its 2 prompt + 4 decode positions bring the session to 9 * GROWTH = 9,216 - so the
        // root save is denied whatever the page allowance is.
        executor
            .decoder
            .set_session_budget(SessionBudget::new(10_000));
        executor.reset(&mut session).unwrap();
        executor
            .prefill(&mut session, &PREFIX_PROMPT[..KEY])
            .unwrap();
        assert!(
            !executor
                .save_prefix(&session, &PREFIX_PROMPT[..KEY], SaveKind::Root)
                .unwrap(),
            "the denied root save reports false"
        );
        assert!(executor.prefix.key_len().is_none(), "the store stays empty");
        assert_eq!(
            executor.decoder.session_used(),
            3 * GROWTH,
            "the denial charged nothing"
        );
        // The job still succeeds: the rest of the prompt and the whole decode are untouched.
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        let mut job = seeded_generation(4);
        assert_eq!(
            tokens_of(&decode_all(&mut executor, &mut session, &mut job)),
            want,
            "the tokens after a denied root save equal those of a full run"
        );
        // The tip is still saved when it fits: the raised cap (after the reset, with no stored
        // entry) admits both points of the same prompt, and the tip lands beside the root.
        executor.reset(&mut session).unwrap();
        executor
            .decoder
            .set_session_budget(SessionBudget::new(100_000));
        executor
            .prefill(&mut session, &PREFIX_PROMPT[..KEY])
            .unwrap();
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT[..KEY], SaveKind::Root)
                .unwrap(),
            "the root fits under the raised cap"
        );
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT, SaveKind::Tip)
                .unwrap(),
            "the tip is saved when it fits"
        );
        assert_eq!(
            executor.prefix.keys(),
            vec![(KEY, true), (PREFIX_PROMPT.len(), false)],
            "the pinned root and the tip are stored"
        );
    }

    /// The report of the executor: the entries and the charged bytes of the store after the job's
    /// save points, and the kind of the entry a later job restored. Every number is hand-derived
    /// from the fixture: an entry of `n` key tokens holds `n - 1` fed positions, so it charges
    /// `FIXED_BYTES + TABLE` bytes.
    ///
    /// Mutation checks: the hit-always-Miss patch fails the restored rows; the byte-sum-dropped
    /// patch (bytes 0) fails the bytes rows.
    #[test]
    fn the_prefix_report_counts_entries_bytes_and_the_restored_kind() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(Q36Rules {
            prefix_store: StoreLimits {
                roots: 2,
                entries: 6,
            },
            ..prefix_rules()
        });
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 0,
                bytes: 0,
                hit: PrefixHit::Miss,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: SaveSideTotals::default()
            }),
            "a fresh store reports all zero"
        );
        // The worker's flow of a first job: prefill to the root point, save the root, prefill
        // to the tip, save the tip. No restore ran, so the hit stays a miss.
        executor.reset(&mut session).unwrap();
        executor
            .prefill(&mut session, &PREFIX_PROMPT[..KEY])
            .unwrap();
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT[..KEY], SaveKind::Root)
                .unwrap(),
            "the root is saved"
        );
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT, SaveKind::Tip)
                .unwrap(),
            "the tip is saved"
        );
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 2,
                bytes: 2 * (FIXED_BYTES + TABLE),
                hit: PrefixHit::Miss,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: SaveSideTotals::default()
            }),
            "the root of 4 and the tip of 6 key tokens"
        );
        // A second conversation diverges after the root: it restores the root. The stored tip
        // shares its first four tokens with the divergent prompt — the longest common prefix —
        // which equals the restored root, so no token is shared beyond the restore.
        executor.reset(&mut session).unwrap();
        let divergent = [1, 2, 3, 4, 9, 9];
        assert_eq!(
            executor.restore_prefix(&mut session, &divergent).unwrap(),
            KEY,
            "the root covers the shared system prompt"
        );
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 2,
                bytes: 2 * (FIXED_BYTES + TABLE),
                hit: PrefixHit::Root,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: SaveSideTotals::default()
            }),
            "the root hit, and the tip's common prefix equals the restored root"
        );
        // A prompt that extends the tip restores the tip.
        executor.reset(&mut session).unwrap();
        let extends = [1, 2, 3, 4, 5, 6, 7];
        assert_eq!(
            executor.restore_prefix(&mut session, &extends).unwrap(),
            PREFIX_PROMPT.len(),
            "the tip covers six tokens"
        );
        assert_eq!(
            executor.prefix_report().unwrap().hit,
            PrefixHit::Tip,
            "the tip hit"
        );
    }

    /// The `uncached_shared` count of the report: a stored key that the prompt covers but cannot
    /// restore — here one of another numeric path — leaves its tokens shared but uncached. The
    /// chunk path refuses the steps entry, nothing restores, and the whole stored key counts.
    ///
    /// Mutation check: returning 0 from `uncached_shared` fails the count row.
    #[test]
    fn the_prefix_report_counts_tokens_shared_with_a_key_that_cannot_restore() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(prefix_rules());
        assert!(
            save_key(&mut executor, &mut session),
            "the steps path saves the key"
        );
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan the chunk path");
        executor.reset(&mut session).unwrap();
        assert_eq!(
            executor
                .restore_prefix(&mut session, &PREFIX_PROMPT)
                .unwrap(),
            0,
            "the chunk path refuses the steps entry"
        );
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 1,
                bytes: FIXED_BYTES + TABLE,
                hit: PrefixHit::Miss,
                uncached_shared: KEY as u32,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: SaveSideTotals::default()
            }),
            "the stored key covers four prompt tokens that no checkpoint served"
        );
    }

    #[test]
    fn an_engine_reuses_a_saved_prefix_between_two_jobs_and_never_across_other_prompts() {
        use crate::sched::{
            job::{Job, JobEvent, JobUsage, PrefixReport},
            worker::{EngineState, ShutdownReport, spawn_engine},
        };
        let _serial = crate::native_test_lock();
        const COUNT: usize = 4;
        let prompt_a = PREFIX_PROMPT.to_vec();
        let prompt_b = vec![6, 4, 7, 1, 5, 3];
        // The first four tokens of A: the same length as the saved key.
        let prompt_e = prompt_a[..KEY].to_vec();
        assert_ne!(
            bits(&logits_alone(&prompt_a[..KEY])),
            bits(&logits_alone(&prompt_b[..KEY])),
            "the two keys must give different logits, or the test is vacuous"
        );
        let chain = |prompt: &[u32]| oracle(prompt, COUNT, &mut seeded_sampler(), VOCAB);
        let (want_a, want_b, want_e) = (chain(&prompt_a), chain(&prompt_b), chain(&prompt_e));
        // The worker piece of 3 splits the 4 key tokens into two pieces.
        let engine = spawn_engine(3, move || Ok(build(prefix_rules()))).unwrap();
        // The report of a store with the one tip entry of `n` key tokens, which holds `n - 1` fed
        // positions and charges `FIXED_BYTES + TABLE` bytes. The save-side lifetime totals: the
        // one-entry store (tip-store limits) makes every later cross-key save spend the single
        // slot - a room removal when the job restored nothing, a restored-ancestor replace when the
        // tip extends the entry this job restored - so they accumulate across the rounds.
        let tip_of = |_key: usize, hit: PrefixHit, uncached: u32, save_side: SaveSideTotals| {
            Some(PrefixReport {
                entries: 1,
                bytes: FIXED_BYTES + TABLE,
                hit,
                uncached_shared: uncached,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side,
            })
        };
        // Each round: the prompt, the boundary of the job, the expected tokens, the expected
        // reused count, and the expected prefix report.
        let rounds = [
            // The first A saves the key of 4 tokens.
            (
                &prompt_a,
                KEY,
                &want_a,
                0u32,
                tip_of(KEY, PrefixHit::Miss, 0, SaveSideTotals::default()),
            ),
            // The next two A restore it, so the entry survives a restore. The restore
            // cursor equals the tip, so no save point runs: the totals stay zero.
            (
                &prompt_a,
                KEY,
                &want_a,
                4,
                tip_of(KEY, PrefixHit::Tip, 0, SaveSideTotals::default()),
            ),
            (
                &prompt_a,
                KEY,
                &want_a,
                4,
                tip_of(KEY, PrefixHit::Tip, 0, SaveSideTotals::default()),
            ),
            // A prompt that equals the key restores nothing — the restore needs a strictly
            // shorter key — but the stored key covers all four of its tokens, so the report
            // counts them as shared and uncached. The job saves a key of 3 tokens: the
            // one-entry list spends the stored tip to admit it (a room removal).
            (
                &prompt_e,
                KEY - 1,
                &want_e,
                0,
                tip_of(
                    KEY - 1,
                    PrefixHit::Miss,
                    KEY as u32,
                    SaveSideTotals {
                        room_removals: 1,
                        ..SaveSideTotals::default()
                    },
                ),
            ),
            // The key of 3 tokens is a prefix of A: 3 tokens are covered. The tip of 4 tokens
            // extends the restored entry, so it replaces it although the store holds one entry
            // (clarification 1: the restored entry is the victim of last resort).
            (
                &prompt_a,
                KEY,
                &want_a,
                3,
                tip_of(
                    KEY,
                    PrefixHit::Tip,
                    0,
                    SaveSideTotals {
                        room_removals: 1,
                        restored_replaces: 1,
                        ..SaveSideTotals::default()
                    },
                ),
            ),
            // The tip of 4 tokens was stored after that restore, so this A covers all 4 key
            // tokens. Had the tip not been stored, the entry of 3 tokens would restore: 3.
            // The restore cursor equals the tip: no save, the totals carry over.
            (
                &prompt_a,
                KEY,
                &want_a,
                4,
                tip_of(
                    KEY,
                    PrefixHit::Tip,
                    0,
                    SaveSideTotals {
                        room_removals: 1,
                        restored_replaces: 1,
                        ..SaveSideTotals::default()
                    },
                ),
            ),
            // B replaces the entry (this job restored nothing, so the room spends it),
            // so the last A restores nothing - and A's own save then spends B's entry.
            (
                &prompt_b,
                KEY,
                &want_b,
                0,
                tip_of(
                    KEY,
                    PrefixHit::Miss,
                    0,
                    SaveSideTotals {
                        room_removals: 2,
                        restored_replaces: 1,
                        ..SaveSideTotals::default()
                    },
                ),
            ),
            (
                &prompt_a,
                KEY,
                &want_a,
                0,
                tip_of(
                    KEY,
                    PrefixHit::Miss,
                    0,
                    SaveSideTotals {
                        room_removals: 3,
                        restored_replaces: 1,
                        ..SaveSideTotals::default()
                    },
                ),
            ),
        ];
        for (round, (prompt, boundary, want, reused, report)) in rounds.into_iter().enumerate() {
            let (job, mut handle) = Job::new(prompt.clone(), seeded_generation(4)).unwrap();
            let job = job.with_prefix_boundary(boundary).unwrap();
            engine.handle().try_submit(job).unwrap();
            let mut tokens = Vec::new();
            let mut usage = None;
            let mut error = None;
            while let Some(event) = handle.events.blocking_recv() {
                match event {
                    JobEvent::Batch(batch) => tokens.extend(batch.tokens),
                    JobEvent::Finished(done) => usage = Some(done),
                    JobEvent::Error(failure) => error = Some(format!("{failure:?}")),
                }
            }
            assert_eq!(error, None, "round {round}: error");
            assert_eq!(&tokens, want, "round {round}: tokens");
            assert_eq!(
                usage,
                Some(JobUsage {
                    prompt_tokens: u32::try_from(prompt.len()).unwrap(),
                    completion_tokens: 4,
                    reused_tokens: reused,
                    prefix: report,
                }),
                "round {round}: usage"
            );
        }
        assert_eq!(
            engine.shutdown(),
            ShutdownReport {
                state: EngineState::Stopped,
                failure: None
            }
        );
    }
}
