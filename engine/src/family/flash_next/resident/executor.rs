//! The Flash-Next executor: `ExecutorOps` over `ResidentDecoder` (decision 0033; packet H8).
//!
//! Ownership: the engine thread builds the executor and its session inside the `make` closure of
//! `spawn_engine`. Nothing native crosses a thread boundary. `ResidentDecoder` and
//! `MlxKernels` are thread-confined by compile-time checks (decisions 0019 and 0024).
//!
//! Checkpoint lifetime: `ResidentDecoder<'c>` borrows its `StreamedCheckpoint`, and `spawn_engine`
//! needs an executor without a borrowed lifetime. `open` leaks one checkpoint with `Box::leak`.
//! The leaked checkpoint stays open until the process ends, which is later than any GPU work that
//! reads it. Known limit: each call to `open` leaks one checkpoint handle, so each engine reload
//! leaks one more. A decoder that owns an `Arc<StreamedCheckpoint>` removes the leak and is a
//! later packet (it edits the files of `resident/` and waits for the snapshot work).
//!
//! Context limit: the executor reads the number of positions from the decoder's own authority,
//! `qsa::MAX_POSITIONS`, which `qsa::check_cache` applies to every step through
//! `ResidentState::check_step`. With sparse QSA (packet C6) the decoder accepts positions 0 to
//! `MAX_POSITIONS - 1` and returns `ContextLimit` at `MAX_POSITIONS`. The prompt check, the
//! finish rule, and `context_limit()` use that one value. No formula is copied here. The Mac test
//! `the_context_limit_is_the_first_position_that_the_decoder_rejects` pins the value to
//! `check_cache`.
//!
//! Prompt length: `prefill` rejects a prompt only when the chunk that crosses the limit arrives.
//! Earlier chunks are already fed by then, and each fed token costs device time. The server (H4)
//! must therefore compare the prompt length with `context_limit()` before it admits the job. A
//! prompt of `context_limit()` tokens or more fails with `ContextLimit`. During generation the job
//! ends with `Length` when the next feed would need position `context_limit()`.
//!
//! Failure handling and the open native-failure gate (decision 0022): a native failure returns
//! `EngError::Backend` unchanged from `prefill` or `decode`. The worker then marks the engine
//! `NativeFailed` and makes no further executor call. The holder that keeps native state alive
//! after such a failure is not built. The unwind drops the state instead, so this gate stays open.
//! An error that is not `Backend` (for example `ContextLimit`, or a failed token decode) fails only
//! the job: the executor resets the session where it can and the engine stays `Ready`.
//!
//! Teardown: `work` in `sched/worker.rs` drops the session before the executor. The session drops
//! its resident state first, then the executor drops the decoder, whose fields wait for GPU
//! completion before the pool and the adapter go (decisions 0032 and 0039).
//!
//! Call order: `reset`, `boundary`, `prefill` for each chunk, then `decode` until a finish. A
//! prompt feeds every token except the last one as `Prompt` rows. The last token stays pending.
//! Each `decode` feeds the pending token through a `Logits` step, draws the next token from the
//! job's sampler, and keeps that token pending. One token leaves per call.
//!
//! Batched prefill (decision 0044, packet H8b): `plan_prefill` applies the admission rule of item
//! 6 to the engine budget and stores the chunk size. Until it runs, and when no chunk size fits,
//! every `Prompt` row is a one-token step. With a chunk size, `prefill` runs its rows through
//! `ResidentDecoder::prefill` in chunks of at most that size. A chunk of one row is a one-token
//! step, and the decoder runs the rows past the dense QSA limit as one-token steps itself. The
//! page sets of the chunks open at the first chunk of a prompt and close at the first `decode` or
//! at a `reset`. A chunk equals the one-token steps within the tolerance of gate G2, not bit for
//! bit, so a state saved by one path is not a state of the other (decision 0036 addendum 1). The
//! worker's `prefill_chunk` should be at least the planned size, or the chunks stay small.
//!
//! The token side of the session (history, pending token, finish rules, EOS, output and context
//! limits, cancellation) lives in `family::session::DecodeSession`, which the Q36 executor shares
//! (packet Q36-F1). `FlashStep` is the model side: it runs the `Prompt` and `Logits` steps, owns
//! the chunk path of the batched prefill, and closes the prefill page sets at a reset.
//!
//! Prefix reuse (decision 0036, addendum 1; packet PR1c): `save_prefix` snapshots the state after
//! the held prompt, and `restore_prefix` gives it back after a reset. The executor keeps its
//! entries in a `PrefixStore` with the addendum 1 limits (one tip entry, saved by the worker); the
//! roots and the periodic points of addendum 2 come with the SERVE-2 packets.
//! `DecodeSession::save_prefix` and `DecodeSession::restore_prefix` own the rules (the held
//! prompt, the strict extension, the denied restore), and the executor passes the two calls that
//! share the resident state. The entry carries the `PrefillPath` that built it. The one-token
//! steps and each chunk size are separate paths, because a chunk matches the steps only within
//! the tolerance of gate G2, so an entry restores only under its own path.

use super::super::{
    load::FlashWeights,
    prefill_plan::{self, ChunkCost, PREFILL_WAVE_CAP, PrefillPlan, PrefillTerms},
};
use super::{
    ResidentDecoder, StepMode,
    qsa::MAX_POSITIONS,
    slots::SetLayout,
    state::{ResidentSnapshot, ResidentState},
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, PrefixSlot, SaveKind, TokenStepper, check_prompt},
    },
    kernels::{HostPages, MlxKernels},
    tokenizer::DecodeBatch,
    weights::{ShardIo, StreamedCheckpoint},
};
use std::{collections::BTreeSet, path::Path};

type Result<T> = std::result::Result<T, EngError>;

/// The limits that the caller reads from the tokenizer bundle. The decoder supplies the position
/// limit (`FlashExecutor::context_limit`).
pub(crate) struct FlashRules {
    /// Every id that ends generation (`TokenizerBundle::eos_token_ids`).
    pub(crate) eos: Vec<u32>,
    /// The number of logits that the sampler may choose from (`TokenizerBundle::vocab_size`). The
    /// model head can be wider than the tokenizer, and the extra ids have no text.
    /// `FlashExecutor::from_parts` rejects 0 and any value above the width of the model head.
    pub(crate) sampler_vocab: usize,
}

/// The numeric path of a prefill. A saved prefix restores only under the path that built it.
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

/// The executor of one Flash-Next model. It lives on the engine thread.
pub(crate) struct FlashExecutor {
    /// The saved prefix. It is declared first, so its arrays drop before the decoder does.
    prefix: PrefixSlot<ResidentSnapshot, PrefillPath>,
    decoder: ResidentDecoder<'static>,
    rules: FlashRules,
    /// The number of positions that the decoder accepts, `qsa::MAX_POSITIONS`.
    context_limit: usize,
    /// The chunk size of the batched prefill, set by `plan_prefill`. `None` means one-token steps.
    chunk: Option<u32>,
    /// The chunk size that the open prefill page sets serve, or 0 when none are open.
    open_tokens: u32,
    /// Tests only: what the chunk path did, shared so that an engine thread test can read it.
    #[cfg(test)]
    log: std::sync::Arc<std::sync::Mutex<ChunkLog>>,
}

/// Tests only: the chunk runs and the page set opens of an executor.
#[cfg(test)]
#[derive(Debug, Default, PartialEq, Eq)]
struct ChunkLog {
    /// The row count of each chunk that ran through `ResidentDecoder::prefill`, in order.
    chunks: Vec<usize>,
    /// The number of times the prefill page sets opened.
    opens: usize,
}

/// The mutable state of one session.
pub(crate) struct FlashSession {
    state: ResidentState,
    /// The committed tokens, the pending token, and the finish flag.
    core: DecodeSession,
}

/// The model side of one call: the decoder, the native state of the session, and the chunk path
/// of the batched prefill. `FlashExecutor::split` builds one value per call from the fields of
/// the executor, so the fields stay on the executor and the tests read them there.
struct FlashStep<'a> {
    decoder: &'a mut ResidentDecoder<'static>,
    state: &'a mut ResidentState,
    /// The chunk size that `plan_prefill` stored, or `None` for one-token steps.
    chunk: Option<u32>,
    /// The chunk size that the open prefill page sets serve, or 0 when none are open.
    open_tokens: &'a mut u32,
    /// Tests only: the chunk log of the executor.
    #[cfg(test)]
    log: &'a std::sync::Mutex<ChunkLog>,
}

impl FlashStep<'_> {
    /// Opens the prefill page sets for `tokens` rows per chunk unless they are open already.
    fn open_prefill(&mut self, tokens: u32) -> Result<()> {
        if *self.open_tokens != tokens {
            *self.open_tokens = 0;
            self.decoder.open_prefill_waves(tokens, PREFILL_WAVE_CAP)?;
            *self.open_tokens = tokens;
            #[cfg(test)]
            {
                self.log.lock().unwrap().opens += 1;
            }
        }
        Ok(())
    }
}

impl TokenStepper for FlashStep<'_> {
    const OPERATION: &'static str = "flash_next";

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
                self.open_prefill(size)?;
                self.decoder.prefill(self.state, rows, StepMode::Prompt)?;
                #[cfg(test)]
                {
                    self.log.lock().unwrap().chunks.push(rows.len());
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

    /// Closes the prefill page sets, then clears the native state.
    fn reset(&mut self) -> Result<()> {
        close_pages(self.decoder, self.open_tokens);
        self.decoder.reset(self.state)
    }
}

/// The number of positions that the native state has fed.
fn fed_positions(state: &ResidentState) -> usize {
    usize::try_from(state.pos()).unwrap_or(usize::MAX)
}

/// Releases and drops the prefill page sets. Nothing happens when none are open.
fn close_pages(decoder: &mut ResidentDecoder<'static>, open_tokens: &mut u32) {
    decoder.close_prefill();
    *open_tokens = 0;
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

impl FlashExecutor {
    /// Loads the artifact in `artifact`, imports the weights, and creates the session. Call it on
    /// the engine thread. `cache_limit` is the MLX allocator cache limit in bytes.
    pub(crate) fn open(
        artifact: &Path,
        cache_limit: u64,
        rules: FlashRules,
    ) -> Result<(Self, FlashSession)> {
        let config = std::fs::read(artifact.join("config.json"))
            .map_err(|error| io_error("config.json", &error))?;
        let ckpt: &'static StreamedCheckpoint = Box::leak(Box::new(open_checkpoint(artifact)?));
        let flash = FlashWeights::load(ckpt, &config)?;
        let mut kernels = MlxKernels::try_new()?;
        kernels.set_cache_limit(cache_limit)?;
        Self::from_parts(kernels, ckpt, &flash, rules)
    }

    /// Builds the executor from loaded parts. The caller may drop `flash` afterwards.
    pub(crate) fn from_parts(
        kernels: MlxKernels,
        ckpt: &'static StreamedCheckpoint,
        flash: &FlashWeights,
        rules: FlashRules,
    ) -> Result<(Self, FlashSession)> {
        let head_vocab = flash.shape().vocab;
        if rules.sampler_vocab == 0 || rules.sampler_vocab > head_vocab {
            return Err(EngError::Checkpoint(format!(
                "sampler vocabulary {} is outside 1..={head_vocab}, the width of the model head",
                rules.sampler_vocab
            )));
        }
        let context_limit = usize::try_from(MAX_POSITIONS)
            .map_err(|_| EngError::BadRequest("context limit overflows".to_string()))?;
        let mut decoder = ResidentDecoder::new(kernels, ckpt, flash)?;
        let state = decoder.zero_state()?;
        let session = FlashSession {
            state,
            core: DecodeSession::default(),
        };
        let executor = Self {
            prefix: PrefixSlot::default(),
            decoder,
            rules,
            context_limit,
            chunk: None,
            open_tokens: 0,
            #[cfg(test)]
            log: std::sync::Arc::default(),
        };
        Ok((executor, session))
    }

    /// The number of positions that the model accepts. A prompt of this many tokens or more is
    /// rejected with `ContextLimit`, and a job ends with `Length` when its next feed needs this
    /// position. The server uses it to check a prompt before admission.
    pub(crate) fn context_limit(&self) -> usize {
        self.context_limit
    }

    /// Splits the executor into the model side of one call and the limits of the engine. The two
    /// borrow different fields, so a caller holds both.
    fn split<'a>(&'a mut self, state: &'a mut ResidentState) -> (FlashStep<'a>, Limits<'a>) {
        let step = FlashStep {
            decoder: &mut self.decoder,
            state,
            chunk: self.chunk,
            open_tokens: &mut self.open_tokens,
            #[cfg(test)]
            log: &self.log,
        };
        let limits = Limits {
            eos: &self.rules.eos,
            sampler_vocab: self.rules.sampler_vocab,
            context_limit: self.context_limit,
        };
        (step, limits)
    }

    /// The admission cost of one prefill chunk on this model (decision 0044 item 6). One expert
    /// costs the page-rounded set of all experts of the largest layer, divided by the expert count.
    fn prefill_cost(&self) -> Result<ChunkCost> {
        let too_big =
            |what: &str| EngError::MemoryPlan(format!("{what} exceeds the platform size"));
        let moe = &self.decoder.weights.shape.moe;
        let experts = u32::try_from(moe.experts())
            .ok()
            .filter(|&experts| experts > 0)
            .ok_or_else(|| too_big("the expert count"))?;
        let top_k = u32::try_from(moe.top_k()).map_err(|_| too_big("top-k"))?;
        let page = HostPages::page_size();
        let mut set_bytes = 0;
        for layer in &self.decoder.weights.layers {
            let layout = SetLayout::new(self.decoder.ckpt, &layer.moe, experts, page)?;
            set_bytes = set_bytes.max(layout.end);
        }
        let set_bytes = u64::try_from(set_bytes).map_err(|_| too_big("the expert set"))?;
        let page = u64::try_from(page).map_err(|_| too_big("the page size"))?;
        prefill_plan::admission_cost(experts, top_k, set_bytes.div_ceil(u64::from(experts)), page)
    }

    /// Applies the admission rule of decision 0044 item 6 and stores the chunk size.
    ///
    /// Item 6 is `remaining = B - resident weights - session state - S3 cache - MLX cache limit -
    /// pressure margin`. The terms of this call, all in bytes:
    ///
    /// - `budget` is `B`, the engine memory budget (32,212,254,720 B in the decision).
    /// - The resident weights come from this executor (`FlashWeights` byte length).
    /// - `s3_cache` is the S3 cache (0 while no S3 cache exists).
    /// - `session_reserve` holds three terms that the caller sums: the session state (QSA,
    ///   DeltaNet, and rings; 245,471,984 B at 2,304 positions in the decision), the MLX cache
    ///   limit (1 GiB in the decision), and the pressure margin (1 GiB in the decision). The
    ///   executor does not know the position count, the cache limit setting, or the margin, so a
    ///   caller that passes 0 admits a larger chunk than the memory plan allows.
    /// - The decode pool is an extra term. Item 6 counts it as idle during prefill. This call
    ///   charges it anyway, which is conservative by 83,214,336 B on the real model (the
    ///   coordinator decision of packet H8b).
    ///
    /// It returns the chunk size, or `None` when no size above one token fits and the prefill
    /// runs one-token steps, together with the terms it charged (item 6: the executor records
    /// which terms it charged). Call it once after construction and before the first job. It
    /// closes open prefill pages.
    pub(crate) fn plan_prefill(
        &mut self,
        budget: u64,
        s3_cache: u64,
        session_reserve: u64,
    ) -> Result<PrefillPlan> {
        let terms = PrefillTerms {
            budget,
            weights: self.decoder.weights.byte_len()?,
            decode_pool: self.decoder.pool_bytes(),
            s3_cache,
            session_reserve,
        };
        let chunk = prefill_plan::admitted_chunk(&self.prefill_cost()?, terms.remaining())?;
        self.close_prefill();
        self.chunk = chunk;
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

    /// Tests only: sets the chunk size without the admission rule, so that a speed run can sweep it.
    #[cfg(test)]
    fn force_chunk(&mut self, chunk: Option<u32>) {
        self.close_prefill();
        self.chunk = chunk;
    }

    /// Releases and drops the prefill page sets. Nothing happens when none are open.
    fn close_prefill(&mut self) {
        close_pages(&mut self.decoder, &mut self.open_tokens);
    }
}

impl ExecutorOps for FlashExecutor {
    type Session = FlashSession;

    /// Refuses a prompt that cannot fit before any state changes: a prompt of `context_limit()`
    /// tokens or more fails with `ContextLimit`, the Flash-Next rule that `prefill` enforces
    /// chunk by chunk on the same number.
    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        check_prompt(prompt_tokens, self.context_limit())
    }

    fn history<'s>(&self, st: &'s FlashSession) -> &'s [u32] {
        st.core.history()
    }

    fn reset(&mut self, st: &mut FlashSession) -> Result<()> {
        let FlashSession { state, core } = st;
        let (mut step, _) = self.split(state);
        core.reset(&mut step)
    }

    fn prefill(&mut self, st: &mut FlashSession, toks: &[u32]) -> Result<()> {
        let FlashSession { state, core } = st;
        let (mut step, limits) = self.split(state);
        core.prefill(&mut step, toks, limits.context_limit)
    }

    fn decode(
        &mut self,
        st: &mut FlashSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        self.close_prefill();
        let FlashSession { state, core } = st;
        let (mut step, limits) = self.split(state);
        core.decode(&mut step, generation, &limits)
    }

    /// No transient routing or draft state exists: the page sets return at the end of each step.
    fn boundary(&mut self, _st: &mut FlashSession) {}

    /// Shares the resident state as a tip entry of this executor's prefix store. The session
    /// rules decide whether the call is valid. The snapshot charges the session budget.
    fn save_prefix(&mut self, st: &FlashSession, key: &[u32]) -> Result<bool> {
        let path = self.prefill_path();
        let decoder = &mut self.decoder;
        let fed = fed_positions(&st.state);
        st.core
            .save_prefix(&mut self.prefix, key, path, fed, SaveKind::Tip, || {
                decoder.snapshot(&st.state)
            })
    }

    /// Restores the saved prefix into the reset session when the session rules accept the prompt.
    fn restore_prefix(&mut self, st: &mut FlashSession, prompt: &[u32]) -> Result<usize> {
        let path = self.prefill_path();
        let decoder = &mut self.decoder;
        let FlashSession { state, core } = st;
        core.restore_prefix(&mut self.prefix, prompt, path, |snapshot| {
            decoder.restore(state, snapshot)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{self, Embedding};
    use super::super::prefill::assert_parity;
    use super::*;
    use crate::{
        family::sampler::{Sampler, SamplingParams},
        tokenizer::{DecodeFinish, IncrementalDecoder, Tokenizer},
    };
    use std::sync::atomic::Ordering;

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

    /// The synthetic model: vocabulary 8. `LIMIT` is a lowered executor limit that the finish
    /// tests set, because the decoder accepts 32,768 positions.
    const VOCAB: usize = 8;
    const LIMIT: usize = 19;
    const PROMPT: [u32; 3] = [1, 2, 3];

    fn rules(eos: Vec<u32>, sampler_vocab: usize) -> FlashRules {
        FlashRules { eos, sampler_vocab }
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap()
    }

    fn generation() -> GenerationState {
        GenerationState::new(IncrementalDecoder::new(&tokenizer()))
    }

    /// Temperature 2.0 spreads the draws over the vocabulary of 8.
    fn params() -> SamplingParams {
        SamplingParams::new(2.0, 1.0, 0).unwrap()
    }

    /// Builds the executor on the synthetic checkpoint. The checkpoint is leaked, as in `open`.
    fn build(rules: FlashRules) -> (FlashExecutor, FlashSession) {
        build_with(Embedding::Flat, rules)
    }

    /// Builds the executor on the synthetic checkpoint with the given embedding table.
    fn build_with(embedding: Embedding, rules: FlashRules) -> (FlashExecutor, FlashSession) {
        let ckpt: &'static StreamedCheckpoint =
            Box::leak(Box::new(fixtures::checkpoint_with(embedding)));
        let flash = FlashWeights::load(ckpt, &fixtures::config()).unwrap();
        FlashExecutor::from_parts(MlxKernels::try_new().unwrap(), ckpt, &flash, rules).unwrap()
    }

    /// The tokens that a `Record` chain draws from `sampler`: a path that shares no code with the
    /// executor's `Logits` step. The decoder and its state drop before the caller builds an executor.
    fn oracle(prompt: &[u32], count: usize, sampler: &mut Sampler, limit: usize) -> Vec<u32> {
        oracle_with(Embedding::Flat, prompt, count, sampler, limit)
    }

    /// `oracle` on the synthetic checkpoint with the given embedding table.
    fn oracle_with(
        embedding: Embedding,
        prompt: &[u32],
        count: usize,
        sampler: &mut Sampler,
        limit: usize,
    ) -> Vec<u32> {
        let ckpt = fixtures::checkpoint_with(embedding);
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
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

    /// Temperature 1 with a fixed seed. The sampled token depends on the whole logits row, so a
    /// decoder state that differs changes the draws. The greedy chain of the synthetic model is
    /// constant and shows no state.
    fn seeded_sampler() -> Sampler {
        Sampler::new(SamplingParams::new(1.0, 1.0, 0).unwrap(), 7)
    }

    fn seeded_generation(max_tokens: u32) -> GenerationState {
        GenerationState::with_sampler(IncrementalDecoder::new(&tokenizer()), seeded_sampler())
            .with_max_tokens(max_tokens)
    }

    /// The full logits row that a fresh decoder gives for the next token after `prompt`. It runs
    /// the same step modes as the executor: `Prompt` for all but the last token, then `Logits`.
    fn logits_alone(prompt: &[u32]) -> Vec<f32> {
        logits_alone_with(Embedding::Flat, prompt)
    }

    /// `logits_alone` on the synthetic checkpoint with the given embedding table.
    fn logits_alone_with(embedding: Embedding, prompt: &[u32]) -> Vec<f32> {
        let ckpt = fixtures::checkpoint_with(embedding);
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
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

    /// A prompt of `len` tokens. The ids cycle through 1 to 7, because the fixture has 8 ids.
    fn cycle(len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| u32::try_from(i % 7).unwrap() + 1)
            .collect()
    }

    /// Plans the chunk size `tokens` through the admission rule: a budget that leaves exactly the
    /// workspace of `tokens` rows after the weights and the decode pool.
    fn plan_chunk(executor: &mut FlashExecutor, tokens: u32) -> Option<u32> {
        let workspace = executor.prefill_cost().unwrap().workspace(tokens).unwrap();
        let held = executor.decoder.weights.byte_len().unwrap() + executor.decoder.pool_bytes();
        executor.plan_prefill(held + workspace, 0, 0).unwrap().chunk
    }

    /// The tokens that both sessions of the chunk parity test feed after the prompt, so the two
    /// sessions see one chain whatever their logits are.
    const FORCED: [u32; 5] = [3, 5, 1, 6, 2];

    /// What a prefill leaves behind: the session after the prompt, and the logits row of each
    /// `Logits` step on the pending token and then on `FORCED`.
    struct Left {
        pages_open: bool,
        pos: u32,
        history: Vec<u32>,
        prompt_len: usize,
        pending: Option<u32>,
        rows: Vec<Vec<f32>>,
    }

    /// Resets the session and prefills `prompt` in worker pieces of `worker_chunk` tokens.
    fn start_job(
        executor: &mut FlashExecutor,
        session: &mut FlashSession,
        prompt: &[u32],
        worker_chunk: usize,
    ) {
        executor.reset(session).unwrap();
        executor.boundary(session);
        for piece in prompt.chunks(worker_chunk) {
            executor.prefill(session, piece).unwrap();
        }
    }

    /// Runs the prompt as the worker does, then feeds the pending token and `FORCED` through
    /// `Logits` steps and keeps every logits row.
    fn run_job(
        executor: &mut FlashExecutor,
        session: &mut FlashSession,
        prompt: &[u32],
        worker_chunk: usize,
    ) -> Left {
        start_job(executor, session, prompt, worker_chunk);
        let pages_open = executor.decoder.prefill_pages.is_some();
        let pending = session.core.pending().unwrap();
        let (pos, history, prompt_len) = (
            session.state.pos(),
            session.core.history().to_vec(),
            session.core.prompt_len(),
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
            pages_open,
            pos,
            history,
            prompt_len,
            pending: Some(pending),
            rows,
        }
    }

    /// The chunk runs and page opens of the executor since the last call. The log is cleared.
    fn take_log(executor: &FlashExecutor) -> (Vec<usize>, usize) {
        let log = std::mem::take(&mut *executor.log.lock().unwrap());
        (log.chunks, log.opens)
    }

    /// The row counts of the chunks that `prefill` makes of `rows` rows with chunks of `size`: a
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

    #[test]
    fn the_expected_chunks_skip_a_single_row_and_split_at_the_size() {
        assert_eq!(expected_chunks(9, 4), vec![4, 4], "a last single row");
        assert_eq!(expected_chunks(10, 4), vec![4, 4, 2], "a last pair");
        assert_eq!(expected_chunks(3, 8), vec![3], "fewer rows than the size");
        assert_eq!(expected_chunks(1, 4), Vec::<usize>::new(), "one row");
        assert_eq!(expected_chunks(0, 4), Vec::<usize>::new(), "no row");
    }

    /// The session state at the positions of decision 0044 item 6 (2,304) in bytes, as the decision
    /// documents it. A run prints it beside the measured value.
    const DOCUMENTED_STATE_BYTES: u64 = 245_471_984;

    /// The positions at which decision 0044 item 6 counts the session state.
    const STATE_POSITIONS: u64 = 2_304;

    /// The terms of `session_reserve` (decision 0044 item 6): the session state, the MLX cache
    /// limit, and the pressure margin. The state is the bytes of the zero state that the decoder
    /// holds plus the QSA growth per position times `positions`. A margin with no value is `None`.
    struct Reserve {
        zero_state: u64,
        growth_per_position: u64,
        positions: u64,
        mlx_cache_limit: u64,
        pressure_margin: Option<u64>,
    }

    impl Reserve {
        fn measure(
            executor: &FlashExecutor,
            session: &FlashSession,
            positions: u64,
            mlx_cache_limit: u64,
            pressure_margin: Option<u64>,
        ) -> Self {
            Self {
                zero_state: session.state.byte_len().unwrap(),
                growth_per_position: super::super::state::qsa_growth_bytes(
                    &executor.decoder.weights,
                )
                .unwrap(),
                positions,
                mlx_cache_limit,
                pressure_margin,
            }
        }

        fn session_state(&self) -> u64 {
            self.zero_state + self.growth_per_position * self.positions
        }

        /// The value for `plan_prefill`. A margin that is not set adds 0.
        fn total(&self) -> u64 {
            self.session_state() + self.mlx_cache_limit + self.pressure_margin.unwrap_or(0)
        }

        /// One `name=value` line per term. A margin that is not set says so.
        fn report(&self) -> String {
            let margin = self.pressure_margin.map_or_else(
                || {
                    "not set (no value recorded, charged as 0, so remaining is overstated)"
                        .to_string()
                },
                |bytes| bytes.to_string(),
            );
            format!(
                "session_state_zero_bytes={}\nsession_state_growth_bytes_per_position={}\n\
                 session_state_positions={}\nsession_state_bytes={}\n\
                 session_state_documented_bytes={DOCUMENTED_STATE_BYTES}\n\
                 mlx_cache_limit_bytes={}\npressure_margin_bytes={margin}\n\
                 session_reserve_total_bytes={}\n",
                self.zero_state,
                self.growth_per_position,
                self.positions,
                self.session_state(),
                self.mlx_cache_limit,
                self.total()
            )
        }

        /// The reserve of a real run: the state at 2,304 positions, the configured MLX cache
        /// limit, and `FLASH_NEXT_EXECUTOR_PRESSURE_MARGIN` (bytes, optional).
        fn of_real_run(
            executor: &FlashExecutor,
            session: &FlashSession,
            mlx_cache_limit: u64,
        ) -> Self {
            let margin = std::env::var("FLASH_NEXT_EXECUTOR_PRESSURE_MARGIN")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FLASH_NEXT_EXECUTOR_PRESSURE_MARGIN is not a number")
                });
            Self::measure(executor, session, STATE_POSITIONS, mlx_cache_limit, margin)
        }
    }

    /// Each term of the reserve changes the total by its own bytes, and a margin that is not set
    /// is named in the report.
    #[test]
    fn the_session_reserve_sums_the_state_the_cache_limit_and_the_margin() {
        let _serial = crate::native_test_lock();
        let (executor, session) = build(rules(vec![], VOCAB));
        let base = Reserve::measure(&executor, &session, 0, 0, None);
        assert_eq!(
            base.total(),
            session.state.byte_len().unwrap(),
            "state only"
        );
        assert!(base.growth_per_position > 0, "the QSA layers grow");
        let ten = Reserve::measure(&executor, &session, 10, 0, None);
        assert_eq!(
            ten.total() - base.total(),
            10 * base.growth_per_position,
            "ten positions"
        );
        let limit = Reserve::measure(&executor, &session, 0, 1_000, None);
        assert_eq!(limit.total() - base.total(), 1_000, "the cache limit");
        let margin = Reserve::measure(&executor, &session, 0, 0, Some(7));
        assert_eq!(margin.total() - base.total(), 7, "the margin");
        assert!(
            base.report().contains("pressure_margin_bytes=not set"),
            "{}",
            base.report()
        );
        assert!(
            margin.report().contains("pressure_margin_bytes=7\n"),
            "{}",
            margin.report()
        );
        assert!(
            limit.report().contains("mlx_cache_limit_bytes=1000\n"),
            "{}",
            limit.report()
        );
    }

    fn argmax(row: &[f32]) -> usize {
        (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap()
    }

    /// The index of the largest logit, the gap to the second largest, and the largest magnitude.
    fn top_two(row: &[f32]) -> (usize, f32, f32) {
        let best = argmax(row);
        let second = row
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != best)
            .map(|(_, &v)| v)
            .max_by(f32::total_cmp)
            .unwrap();
        let scale = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        (best, row[best] - second, scale)
    }

    #[test]
    fn the_top_two_gap_is_the_distance_to_the_second_logit() {
        assert_eq!(top_two(&[1.0, -4.0, 3.5]), (2, 2.5, 4.0), "gap");
        assert_eq!(top_two(&[2.0, 2.0]).1, 0.0, "a tie has no gap");
    }

    fn decode_all(
        executor: &mut FlashExecutor,
        session: &mut FlashSession,
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

    /// The text that a fresh decoder gives for `tokens`, with its tail.
    fn text_of(tokens: &[u32]) -> String {
        let mut decoder = IncrementalDecoder::new(&tokenizer());
        let mut text = String::new();
        for &token in tokens {
            text.push_str(&decoder.push(token).unwrap());
        }
        text.push_str(&decoder.finish().unwrap());
        text
    }

    #[test]
    fn greedy_decode_follows_the_record_chain_and_ends_at_max_tokens() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PROMPT, 5, &mut Sampler::greedy(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut job = generation().with_max_tokens(5);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let batches = decode_all(&mut executor, &mut session, &mut job);
        assert_eq!(tokens_of(&batches), want, "tokens");
        assert_eq!(batches.len(), 5, "one token per call");
        assert!(
            batches[..4].iter().all(|b| b.finish.is_none()),
            "early finish"
        );
        assert_eq!(batches[4].finish, Some(DecodeFinish::Length), "last finish");
        let text: String = batches.iter().map(|b| b.text.as_str()).collect();
        assert_eq!(text, text_of(&want), "text");
        let mut history = PROMPT.to_vec();
        history.extend_from_slice(&want);
        assert_eq!(executor.history(&session), history.as_slice(), "history");
        // Three prompt tokens and four sampled tokens are fed. The fifth sampled token is pending.
        assert_eq!(session.state.pos(), 7, "fed positions");
        assert_eq!(session.core.pending(), Some(want[4]), "pending token");
    }

    #[test]
    fn a_prompt_in_two_chunks_gives_the_same_tokens_as_one_chunk() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PROMPT, 4, &mut Sampler::greedy(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut first = generation().with_max_tokens(4);
        executor.prefill(&mut session, &PROMPT).unwrap();
        assert_eq!(
            tokens_of(&decode_all(&mut executor, &mut session, &mut first)),
            want
        );
        executor.reset(&mut session).unwrap();
        assert!(executor.history(&session).is_empty(), "history after reset");
        let mut second = generation().with_max_tokens(4);
        executor.prefill(&mut session, &PROMPT[..1]).unwrap();
        executor.prefill(&mut session, &PROMPT[1..]).unwrap();
        assert_eq!(
            executor.history(&session),
            PROMPT.as_slice(),
            "history after chunks"
        );
        assert_eq!(session.state.pos(), 2, "fed positions after chunks");
        assert_eq!(
            session.core.prompt_len(),
            3,
            "held prompt length after chunks"
        );
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut second));
        assert_eq!(got, want, "tokens after a reset and two chunks");
    }

    #[test]
    fn an_eos_token_ends_the_job_with_stop_and_adds_no_text() {
        let _serial = crate::native_test_lock();
        let chain = oracle(&PROMPT, 3, &mut Sampler::greedy(), VOCAB);
        let eos = chain[2];
        let first = chain.iter().position(|&t| t == eos).unwrap();
        let (mut executor, mut session) = build(rules(vec![eos], VOCAB));
        let mut job = generation();
        executor.prefill(&mut session, &PROMPT).unwrap();
        let batches = decode_all(&mut executor, &mut session, &mut job);
        assert_eq!(
            tokens_of(&batches),
            chain[..=first].to_vec(),
            "tokens up to EOS"
        );
        let last = batches.last().unwrap();
        assert_eq!(last.finish, Some(DecodeFinish::Stop), "finish");
        assert_eq!(last.tokens, vec![eos], "the EOS token is in the batch");
        assert_eq!(last.text, "", "EOS adds no text");
        assert_eq!(session.core.pending(), None, "EOS is not fed");
        assert_eq!(job.remaining_tokens(), None, "no output limit was set");
        let again = executor.decode(&mut session, &mut job);
        assert!(matches!(again, Err(EngError::BadRequest(_))), "{again:?}");
    }

    #[test]
    fn a_limit_of_zero_ends_before_any_step() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut job = generation().with_max_tokens(0);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let batch = executor.decode(&mut session, &mut job).unwrap();
        assert_eq!(batch.finish, Some(DecodeFinish::Length));
        assert!(batch.tokens.is_empty(), "no token");
        assert_eq!(session.state.pos(), 2, "the pending token was not fed");
    }

    #[test]
    fn a_cancelled_job_resets_the_session_and_the_next_job_is_clean() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PROMPT, 3, &mut Sampler::greedy(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut cancelled = generation();
        executor.prefill(&mut session, &PROMPT).unwrap();
        executor.decode(&mut session, &mut cancelled).unwrap();
        cancelled.cancel.store(true, Ordering::SeqCst);
        let batch = executor.decode(&mut session, &mut cancelled).unwrap();
        assert_eq!(batch.finish, Some(DecodeFinish::Cancelled));
        assert!(batch.tokens.is_empty(), "no token after the cancel");
        assert!(
            executor.history(&session).is_empty(),
            "history after cancel"
        );
        assert_eq!(session.core.pending(), None, "pending after cancel");
        assert_eq!(session.state.pos(), 0, "position after cancel");
        let mut next = generation().with_max_tokens(3);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut next));
        assert_eq!(got, want, "tokens after a cancelled job");
    }

    /// The session A, B, A pattern on one executor. Each job starts with a reset, as the engine
    /// thread does. A kept decoder state shows in the fed position, in the logits row of the
    /// first decode step, and, through a seeded sampler, in the sampled tokens.
    #[test]
    fn each_job_after_a_reset_matches_the_same_job_run_alone_in_the_order_a_b_a() {
        const COUNT: usize = 8;
        let _serial = crate::native_test_lock();
        let prompt_a = PROMPT;
        let prompt_b = [6, 4, 7, 1, 5];
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
            assert!(executor.history(&session).is_empty(), "{name}: history");
            let mut job = seeded_generation(u32::try_from(COUNT).unwrap());
            executor.prefill(&mut session, prompt).unwrap();
            let got = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
            assert_eq!(&got, want, "{name}: tokens");
            // The prompt and all but the last sampled token are fed. The last one is pending.
            let fed = u32::try_from(prompt.len() + COUNT - 1).unwrap();
            assert_eq!(session.state.pos(), fed, "{name}: fed positions");
        }
    }

    #[test]
    fn a_prompt_at_the_limit_is_rejected_and_the_last_allowed_position_ends_with_length() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        // Lower the executor limit: the decoder accepts 32,768 positions, which a test cannot feed.
        executor.context_limit = LIMIT;
        let long: Vec<u32> = (0..LIMIT as u32).map(|i| i % 7 + 1).collect();
        let rejected = executor.prefill(&mut session, &long);
        assert!(
            matches!(rejected, Err(EngError::ContextLimit(_))),
            "{rejected:?}"
        );
        assert_eq!(session.state.pos(), 0, "no step ran");
        assert!(
            executor.history(&session).is_empty(),
            "history after the rejection"
        );
        // 18 prompt tokens fit: 17 fed, one pending. Two decodes feed positions 17 and 18.
        let mut job = generation();
        executor.prefill(&mut session, &long[..LIMIT - 1]).unwrap();
        let first = executor.decode(&mut session, &mut job);
        assert!(matches!(&first, Ok(b) if b.finish.is_none()), "{first:?}");
        // The second decode feeds position 18. The next feed would need position 19, so the job
        // ends with Length here. A limit that is one too high gives None, and a third decode
        // would feed position 19, which the decoder accepts, so only this rule ends the job.
        let second = executor.decode(&mut session, &mut job);
        assert!(
            matches!(&second, Ok(b) if b.finish == Some(DecodeFinish::Length) && b.tokens.len() == 1),
            "{second:?}"
        );
        assert_eq!(session.state.pos(), 19, "fed positions");
    }

    #[test]
    fn the_context_limit_is_the_first_position_that_the_decoder_rejects() {
        let _serial = crate::native_test_lock();
        let (executor, _session) = build(rules(vec![], VOCAB));
        let limit = executor.context_limit();
        assert_eq!(
            limit, MAX_POSITIONS as usize,
            "the limit is the decoder constant"
        );
        // `check_cache` is the check that `ResidentState::check_step` applies to every QSA layer.
        let last = u32::try_from(limit - 1).unwrap();
        assert!(
            super::super::qsa::check_cache(limit - 1, last).is_ok(),
            "position {last} must be accepted"
        );
        let first_past = super::super::qsa::check_cache(limit, last + 1);
        assert!(
            matches!(first_past, Err(EngError::ContextLimit(_))),
            "position {limit}: {first_past:?}"
        );
    }

    /// Row 1: 10 tokens, chunks of 4, worker pieces of 5: calls give the rows 4, then 4 and 1 (the
    /// carried pending token leads the second call, and the single row is a step). Row 2: 25
    /// tokens, chunks of 8, worker pieces of 15: the second call starts at position 14, so its
    /// chunk of 8 straddles the 19 dense positions of the fixture, and the last call runs steps.
    /// A chunk equals the steps within the gate G2 tolerance (not bit for bit). Both sessions get
    /// the same token chain, and every logits row is compared with `assert_parity`. The argmax is
    /// compared only where the stepped row's top two logits differ by more than twice the G2
    /// tolerance, because a smaller gap can flip inside the tolerance. A seeded sampler is not
    /// compared for the same reason.
    #[test]
    fn a_chunked_prefill_leaves_the_session_of_one_token_steps() {
        let _serial = crate::native_test_lock();
        for (len, chunk, worker) in [(10usize, 4u32, 5usize), (25, 8, 15)] {
            let what = format!("{len} tokens, chunks of {chunk}, worker pieces of {worker}");
            let prompt = cycle(len);
            let (mut executor, mut session) = build(rules(vec![], VOCAB));
            let stepped = run_job(&mut executor, &mut session, &prompt, worker);
            assert!(
                !stepped.pages_open,
                "{what}: the default is one-token steps"
            );
            assert_eq!(take_log(&executor), (vec![], 0), "{what}: no chunk ran");
            assert_eq!(
                plan_chunk(&mut executor, chunk),
                Some(chunk),
                "{what}: plan"
            );
            let chunked = run_job(&mut executor, &mut session, &prompt, worker);
            assert!(chunked.pages_open, "{what}: the chunk path did not run");
            let (chunks, opens) = take_log(&executor);
            let first = (worker - 1).min(len - 1);
            let mut want = expected_chunks(first, chunk as usize);
            want.extend(expected_chunks(len - 1 - first, chunk as usize));
            assert_eq!(chunks, want, "{what}: chunk row counts");
            assert_eq!(opens, 1, "{what}: one page open for the prompt");
            assert_eq!(
                stepped.pos,
                u32::try_from(len - 1).unwrap(),
                "{what}: steps pos"
            );
            assert_eq!(chunked.pos, stepped.pos, "{what}: fed positions");
            assert_eq!(chunked.history, prompt, "{what}: history");
            assert_eq!(chunked.history, stepped.history, "{what}: history vs steps");
            assert_eq!(chunked.prompt_len, len, "{what}: held prompt length");
            assert_eq!(
                chunked.pending,
                prompt.last().copied(),
                "{what}: pending token"
            );
            assert_eq!(chunked.pending, stepped.pending, "{what}: pending vs steps");
            assert_eq!(chunked.rows.len(), 1 + FORCED.len(), "{what}: row count");
            for (n, (got, want)) in chunked.rows.iter().zip(&stepped.rows).enumerate() {
                assert_parity(got, want, &format!("{what}: logits row {n}"));
                let (best, margin, scale) = top_two(want);
                if margin > 2.0 * 1e-4 * scale {
                    assert_eq!(argmax(got), best, "{what}: argmax of row {n}");
                }
            }
            // The decode after a chunked prompt still works and closes the pages.
            start_job(&mut executor, &mut session, &prompt, worker);
            let mut job = seeded_generation(6);
            let tokens = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
            assert_eq!(tokens.len(), 6, "{what}: decoded tokens");
            assert!(
                executor.decoder.prefill_pages.is_none(),
                "{what}: pages closed at the first decode"
            );
        }
    }

    /// Each prompt is split at the planned size: a chunk never has more rows than the size, a
    /// last single row is a step, a size of 2 is a chunk, and a size of 1 makes no chunk at all.
    #[test]
    fn the_chunks_of_a_prompt_never_exceed_the_planned_size() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        let result = executor.prefill(&mut session, &cycle(10));
        assert!(result.is_ok(), "a prompt of 10 tokens: {result:?}");
        assert_eq!(
            take_log(&executor),
            (vec![4, 4], 1),
            "9 rows: two chunks of 4 and a step"
        );
        for (size, want) in [(2u32, vec![2, 2]), (1, vec![])] {
            executor.force_chunk(Some(size));
            executor.reset(&mut session).unwrap();
            let result = executor.prefill(&mut session, &cycle(6));
            assert!(result.is_ok(), "size {size}: {result:?}");
            let (chunks, opens) = take_log(&executor);
            assert_eq!(chunks, want, "size {size}: 5 rows");
            assert_eq!(opens, usize::from(!want.is_empty()), "size {size}: opens");
        }
    }

    /// The plan reads the weights and the decode pool from the decoder and subtracts the cache and
    /// the reserve. A budget of exactly the workspace of 8 rows gives 8, one byte less gives 4
    /// (the next candidate), and a budget that holds only the weights and the pool gives `None`.
    #[test]
    fn the_plan_follows_the_budget_the_weights_the_cache_and_the_reserve() {
        let _serial = crate::native_test_lock();
        let (mut executor, _session) = build(rules(vec![], VOCAB));
        assert_eq!(executor.prefill_chunk(), None, "before planning");
        let workspace = |executor: &FlashExecutor, rows| {
            executor.prefill_cost().unwrap().workspace(rows).unwrap()
        };
        let held = executor.decoder.weights.byte_len().unwrap() + executor.decoder.pool_bytes();
        // The fixture has 4 experts and top-k 2 (`fixtures.rs`). One expert costs the page-rounded
        // set of all 4 experts of the largest layer, divided by 4 and rounded up.
        let cost = executor.prefill_cost().unwrap();
        let page = HostPages::page_size();
        let largest = executor
            .decoder
            .weights
            .layers
            .iter()
            .map(|l| {
                SetLayout::new(executor.decoder.ckpt, &l.moe, 4, page)
                    .unwrap()
                    .end
            })
            .max()
            .unwrap();
        assert_eq!((cost.experts, cost.top_k), (4, 2), "experts and top-k");
        assert_eq!(
            cost.expert_bytes,
            u64::try_from(largest).unwrap().div_ceil(4),
            "bytes of one expert"
        );
        let w8 = workspace(&executor, 8);
        let mut plan = |budget, cache, reserve| {
            let plan = executor.plan_prefill(budget, cache, reserve).unwrap();
            assert_eq!(executor.prefill_chunk(), plan.chunk, "the plan is stored");
            let terms = plan.terms;
            assert_eq!(
                (terms.budget, terms.s3_cache, terms.session_reserve),
                (budget, cache, reserve),
                "the plan records the terms that the caller passed"
            );
            assert_eq!(
                terms.weights + terms.decode_pool,
                held,
                "the plan records the weights and the decode pool of the executor"
            );
            plan.chunk
        };
        assert_eq!(plan(held + w8, 0, 0), Some(8), "exactly 8 rows");
        assert_eq!(plan(held + w8 - 1, 0, 0), Some(4), "one byte less");
        assert_eq!(
            plan(held + w8 + 3_000, 1_000, 2_000),
            Some(8),
            "cache and reserve"
        );
        assert_eq!(
            plan(held + w8 + 2_999, 1_000, 2_000),
            Some(4),
            "one byte less"
        );
        assert_eq!(plan(held, 0, 0), None, "no room for a chunk");
    }

    /// Fixture: 2 layers, so a one-token `Prompt` step makes 4 synchronizations and a chunk makes
    /// `layers + 2` = 4 (the flat cap leaves one DeltaNet part, one wave per layer). A prompt of 10
    /// tokens has 9 rows: chunks of 4, 4, and a step for the last row make 12. Steps alone make 36.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_chunked_prefill_makes_the_synchronizations_of_one_chunk_per_run_of_rows() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        let before = executor.decoder.kernels.qualification_sync_count();
        executor.prefill(&mut session, &cycle(10)).unwrap();
        let used = executor.decoder.kernels.qualification_sync_count() - before;
        assert_eq!(used, 4 + 4 + 4, "chunk, chunk, step");
    }

    /// The page sets open at the first chunk of a prompt, stay open between worker calls, and close
    /// at the first `decode` and at a `reset`. A cancelled or failed job ends in a `reset`.
    #[test]
    fn the_prefill_pages_open_with_the_first_chunk_and_close_at_the_first_decode_and_at_a_reset() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        let open = |executor: &FlashExecutor| {
            (
                executor.decoder.prefill_pages.is_some(),
                executor.open_tokens,
            )
        };
        assert_eq!(open(&executor), (false, 0), "before the prompt");
        // Two tokens leave one row to feed: a one-token step, so no chunk and no pages.
        executor.prefill(&mut session, &cycle(2)).unwrap();
        assert_eq!(open(&executor), (false, 0), "a single row is a step");
        executor.reset(&mut session).unwrap();
        executor.prefill(&mut session, &cycle(6)).unwrap();
        assert_eq!(open(&executor), (true, 4), "after the first chunk");
        executor.prefill(&mut session, &cycle(3)).unwrap();
        assert_eq!(open(&executor), (true, 4), "kept between worker calls");
        assert_eq!(
            take_log(&executor),
            (vec![4, 3], 1),
            "the second worker call chunks without a second open"
        );
        executor.decode(&mut session, &mut generation()).unwrap();
        assert_eq!(open(&executor), (false, 0), "closed at the first decode");
        executor.reset(&mut session).unwrap();
        executor.prefill(&mut session, &cycle(6)).unwrap();
        assert_eq!(open(&executor), (true, 4), "reopened by the next prompt");
        executor.reset(&mut session).unwrap();
        assert_eq!(open(&executor), (false, 0), "closed by a reset");
        executor.prefill(&mut session, &cycle(6)).unwrap();
        assert_eq!(open(&executor), (true, 4), "open before a new plan");
        assert_eq!(plan_chunk(&mut executor, 8), Some(8), "a new plan");
        assert_eq!(open(&executor), (false, 0), "closed by a new plan");
    }

    #[test]
    fn the_sampler_sees_only_the_tokenizer_vocabulary() {
        let _serial = crate::native_test_lock();
        let open = oracle(&PROMPT, 6, &mut Sampler::greedy(), VOCAB);
        let limit = usize::try_from(*open.iter().max().unwrap()).unwrap();
        assert!(
            limit > 0,
            "the fixture chain is all token 0; the test cannot discriminate"
        );
        let want = oracle(&PROMPT, 6, &mut Sampler::greedy(), limit);
        assert_ne!(want, open, "the limit must change the chain");
        let (mut executor, mut session) = build(rules(vec![], limit));
        let mut job = generation().with_max_tokens(6);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
        assert_eq!(got, want, "tokens");
        assert!(
            got.iter().all(|&t| (t as usize) < limit),
            "a token is outside the vocabulary"
        );
    }

    #[test]
    fn a_sampling_job_draws_from_its_own_sampler() {
        let _serial = crate::native_test_lock();
        let greedy = oracle(&PROMPT, 6, &mut Sampler::greedy(), VOCAB);
        let want = oracle(&PROMPT, 6, &mut Sampler::new(params(), 42), VOCAB);
        assert_ne!(
            want, greedy,
            "the sampled chain equals the greedy chain; pick another seed"
        );
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let sampler = Sampler::new(params(), 42);
        let mut job = GenerationState::with_sampler(IncrementalDecoder::new(&tokenizer()), sampler)
            .with_max_tokens(6);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut job));
        assert_eq!(got, want, "tokens");
    }

    #[test]
    fn a_decode_before_prefill_and_an_empty_chunk_are_rejected() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut job = generation();
        let early = executor.decode(&mut session, &mut job);
        assert!(matches!(early, Err(EngError::BadRequest(_))), "{early:?}");
        let empty = executor.prefill(&mut session, &[]);
        assert!(matches!(empty, Err(EngError::BadRequest(_))), "{empty:?}");
    }

    #[test]
    fn a_prefill_after_a_finish_needs_a_reset() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let mut job = generation().with_max_tokens(1);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let batch = executor.decode(&mut session, &mut job).unwrap();
        assert_eq!(batch.finish, Some(DecodeFinish::Length), "finish");
        let held = executor.history(&session).to_vec();
        let fed = session.state.pos();
        let again = executor.prefill(&mut session, &PROMPT);
        assert!(
            matches!(&again, Err(EngError::BadRequest(m)) if m.contains("after finish")),
            "{again:?}"
        );
        assert_eq!(executor.history(&session), held.as_slice(), "history kept");
        assert_eq!(session.state.pos(), fed, "no step ran");
        executor.reset(&mut session).unwrap();
        executor.prefill(&mut session, &PROMPT).unwrap();
        assert_eq!(executor.history(&session), PROMPT.as_slice(), "after reset");
    }

    /// The decoder of the rewriting fixture of `tokenizer/tests.rs`: pushing ids 0 and 1 and then
    /// calling `finish` fails, because the final text no longer extends the text already sent.
    const REWRITING_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Replace","pattern":{"String":"ab"},"content":"X"}]},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;

    #[test]
    fn a_failed_decoder_tail_resets_the_session_and_the_next_job_is_clean() {
        let _serial = crate::native_test_lock();
        let want = oracle(&PROMPT, 3, &mut Sampler::greedy(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let rewriting = Tokenizer::from_bytes(REWRITING_TOKENIZER_JSON.as_bytes()).unwrap();
        let mut decoder = IncrementalDecoder::new(&rewriting);
        assert_eq!(decoder.push(0).unwrap(), "a");
        assert_eq!(decoder.push(1).unwrap(), "");
        // A limit of zero ends the job in `end` before any step, and `finish` then fails.
        let mut job = GenerationState::new(decoder).with_max_tokens(0);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let failed = executor.decode(&mut session, &mut job);
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(m)) if m.contains("prefix")),
            "{failed:?}"
        );
        assert!(executor.history(&session).is_empty(), "history");
        assert_eq!(session.state.pos(), 0, "position");
        assert_eq!(session.core.pending(), None, "pending");
        assert!(
            !session.core.is_finished(),
            "the session is reset, not finished"
        );
        let mut next = generation().with_max_tokens(3);
        executor.prefill(&mut session, &PROMPT).unwrap();
        let got = tokens_of(&decode_all(&mut executor, &mut session, &mut next));
        assert_eq!(got, want, "tokens after the failed job");
    }

    /// The synthetic tokenizer cut to its first `size` ids.
    fn tokenizer_with_vocab(size: usize) -> Tokenizer {
        let full = r#""vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}"#;
        let ids: Vec<String> = "abcdefgh"
            .chars()
            .take(size)
            .enumerate()
            .map(|(id, c)| format!("\"{c}\":{id}"))
            .collect();
        let json = TOKENIZER_JSON.replace(full, &format!("\"vocab\":{{{}}}", ids.join(",")));
        assert_ne!(json, TOKENIZER_JSON, "the vocabulary was not replaced");
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    #[test]
    fn a_token_that_the_text_decoder_rejects_resets_the_session() {
        let _serial = crate::native_test_lock();
        let chain = oracle(&PROMPT, 6, &mut Sampler::greedy(), VOCAB);
        let size = usize::try_from(*chain.iter().max().unwrap()).unwrap();
        assert!(
            size > 0,
            "the fixture chain is all token 0; the test cannot discriminate"
        );
        // The first sampled token with the largest id is outside the cut tokenizer.
        let bad = chain.iter().position(|&t| t as usize >= size).unwrap();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        let decoder = IncrementalDecoder::new(&tokenizer_with_vocab(size));
        let mut job = GenerationState::new(decoder);
        executor.prefill(&mut session, &PROMPT).unwrap();
        for call in 0..bad {
            let batch = executor.decode(&mut session, &mut job);
            assert!(batch.is_ok(), "call {call}: {batch:?}");
        }
        let failed = executor.decode(&mut session, &mut job);
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(_))),
            "{failed:?}"
        );
        assert!(executor.history(&session).is_empty(), "history");
        assert_eq!(session.state.pos(), 0, "position");
        assert_eq!(session.core.pending(), None, "pending");
    }

    #[test]
    fn a_sampler_vocabulary_outside_the_model_head_is_rejected() {
        let _serial = crate::native_test_lock();
        for sampler_vocab in [0, VOCAB + 1] {
            let ckpt: &'static StreamedCheckpoint = Box::leak(Box::new(fixtures::checkpoint()));
            let flash = FlashWeights::load(ckpt, &fixtures::config()).unwrap();
            let kernels = MlxKernels::try_new().unwrap();
            let built =
                FlashExecutor::from_parts(kernels, ckpt, &flash, rules(vec![], sampler_vocab));
            assert!(
                matches!(built.as_ref().err(), Some(EngError::Checkpoint(m)) if m.contains("sampler vocabulary")),
                "sampler_vocab {sampler_vocab}: {:?}",
                built.err()
            );
        }
    }

    /// Pins the step modes through the executor by the synchronization count (decision 0041
    /// item 3). `step.rs` gives `base = layers + 2` and: Prompt `base`, Logits `base + 1`, Record
    /// `base + 3`. The synthetic model has 2 layers, so base is 4, Prompt costs 4, Logits costs 5,
    /// and Record costs 7. A prompt of 3 tokens feeds 2 tokens by Prompt steps and keeps the last
    /// one pending: 2 * 4 = 8. Each decode feeds one pending token by a Logits step: 5. A
    /// `Generate` step would also cost 5, but it reads back a token and no logits, so the token
    /// chain tests in this module cover it.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn the_executor_uses_prompt_steps_for_the_prompt_and_logits_steps_for_decode() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(
            executor.decoder.weights.layers.len(),
            2,
            "synthetic layer count"
        );
        let mut job = generation().with_max_tokens(3);
        let count = |executor: &FlashExecutor| executor.decoder.kernels.qualification_sync_count();
        let before = count(&executor);
        executor.prefill(&mut session, &PROMPT).unwrap();
        assert_eq!(
            count(&executor) - before,
            8,
            "prefill of 3 tokens: 2 Prompt steps"
        );
        for call in 0..3 {
            let before = count(&executor);
            executor.decode(&mut session, &mut job).unwrap();
            assert_eq!(
                count(&executor) - before,
                5,
                "decode {call}: one Logits step"
            );
        }
    }

    static_assertions::assert_not_impl_any!(FlashExecutor: Send, Sync);
    static_assertions::assert_not_impl_any!(FlashSession: Send, Sync);

    /// A prompt of 6 tokens whose first 4 tokens form the saved key.
    const PREFIX_PROMPT: [u32; 6] = [1, 2, 3, 4, 5, 6];
    /// The key length of the prefix tests.
    const KEY: usize = 4;
    /// Bytes that one fed QSA position adds in the fixture (resident `state.rs`).
    const QSA_PER_POSITION: u64 = 640;

    /// Saves `PREFIX_PROMPT[..KEY]` in a reset session: prefill the key, then save it.
    fn save_key(executor: &mut FlashExecutor, session: &mut FlashSession) -> bool {
        executor.reset(session).unwrap();
        executor.prefill(session, &PREFIX_PROMPT[..KEY]).unwrap();
        executor
            .save_prefix(session, &PREFIX_PROMPT[..KEY])
            .unwrap()
    }

    #[test]
    fn a_restored_prefix_continues_like_a_full_prefill() {
        let _serial = crate::native_test_lock();
        // The seeded sampler reads the whole logits row, so a restored state that differs
        // changes the draws. The greedy chain of the synthetic model is constant.
        let want = oracle(&PREFIX_PROMPT, 4, &mut seeded_sampler(), VOCAB);
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
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
    fn a_restored_prefix_gives_bitwise_equal_logits_on_a_token_dependent_model() {
        let _serial = crate::native_test_lock();
        let table = Embedding::TokenDependent;
        let (mut executor, mut session) = build_with(table, rules(vec![], VOCAB));
        let want = run_job(&mut executor, &mut session, &PREFIX_PROMPT, 6);
        // A prompt with other tokens gives other rows, so equal rows below cannot come from
        // state that ignores the tokens.
        let other = run_job(&mut executor, &mut session, &[6, 4, 7, 1, 5, 3], 6);
        assert_ne!(
            bits(&want.rows[0]),
            bits(&other.rows[0]),
            "the two prompts must give different logits, or the test is vacuous"
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
        // One-token steps are the path with bitwise equality (addendum item 8).
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
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        // Cap 10,000. Three fed positions hold 1,920 bytes. The snapshot needs 1,920 + 14,336 = 16,256.
        executor
            .decoder
            .set_session_budget(crate::sched::SessionBudget::new(10_000));
        assert!(
            !save_key(&mut executor, &mut session),
            "a denied charge reports false"
        );
        assert_eq!(executor.prefix.key_len(), None, "no entry");
        assert_eq!(
            executor.decoder.session_used(),
            3 * QSA_PER_POSITION,
            "only the session is charged"
        );
    }

    #[test]
    fn a_second_save_fits_only_because_the_first_entry_drops_first() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        // Cap 25,000. The second save needs 3,200 + 17,536 = 20,736 bytes after the first entry
        // drops. With the first entry kept it needs 19,456 + 17,536 = 36,992 and the cap denies it.
        executor
            .decoder
            .set_session_budget(crate::sched::SessionBudget::new(25_000));
        assert!(save_key(&mut executor, &mut session), "first save");
        // Three fed positions: session 1,920 + snapshot (1,920 + 14,336).
        assert_eq!(
            executor.decoder.session_used(),
            18_176,
            "after the first save"
        );
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        // Five fed positions: session 3,200 + the first snapshot 16,256.
        assert_eq!(
            executor.decoder.session_used(),
            19_456,
            "before the second save"
        );
        assert!(
            executor.save_prefix(&session, &PREFIX_PROMPT).unwrap(),
            "second save"
        );
        assert_eq!(
            executor.decoder.session_used(),
            20_736,
            "after the second save"
        );
        assert_eq!(
            executor.prefix.key_len(),
            Some(6),
            "the entry holds the new key"
        );
    }

    #[test]
    fn a_restore_denied_by_the_budget_drops_the_entry_and_changes_nothing() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        // Cap 19,000: session 1,920 + snapshot 16,256 = 18,176 fit. A restore into the live
        // session needs 1,920 more (20,096) and is denied. The worker always resets first; this
        // test skips the reset to reach the denial.
        executor
            .decoder
            .set_session_budget(crate::sched::SessionBudget::new(19_000));
        assert!(
            save_key(&mut executor, &mut session),
            "the save was refused"
        );
        assert_eq!(executor.decoder.session_used(), 18_176, "after the save");
        let covered = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(covered, 0, "covered tokens");
        assert_eq!(
            executor.prefix.key_len(),
            None,
            "the denied restore drops the entry"
        );
        assert_eq!(
            executor.decoder.session_used(),
            3 * QSA_PER_POSITION,
            "only the session stays charged"
        );
        assert_eq!(executor.history(&session), &PREFIX_PROMPT[..KEY], "history");
        assert_eq!(session.core.pending(), Some(4), "pending token");
        assert_eq!(session.state.pos(), 3, "fed positions");
    }

    #[test]
    fn a_prefix_saved_on_one_prefill_path_is_not_restored_on_another() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert!(
            save_key(&mut executor, &mut session),
            "the save was refused"
        );
        executor.reset(&mut session).unwrap();
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        let chunked = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(chunked, 0, "covered tokens on the chunk path");
        assert!(executor.history(&session).is_empty(), "history");
        assert_eq!(executor.prefix.key_len(), Some(KEY), "the entry stays");
        executor.force_chunk(None);
        let steps = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(
            steps, KEY,
            "covered tokens on the path that built the entry"
        );
    }

    #[test]
    fn a_prefix_saved_and_restored_on_the_chunk_path_matches_a_full_chunked_prefill() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        let want = run_job(&mut executor, &mut session, &PREFIX_PROMPT, 6);
        // The same prompt through a saved key of 4 tokens: the key is one chunk of 3 rows, and
        // the rest is one chunk of 2 rows (the pending token of the key and the token 5).
        executor.reset(&mut session).unwrap();
        executor
            .prefill(&mut session, &PREFIX_PROMPT[..KEY])
            .unwrap();
        assert!(
            executor
                .save_prefix(&session, &PREFIX_PROMPT[..KEY])
                .unwrap(),
            "the save was refused"
        );
        executor.reset(&mut session).unwrap();
        take_log(&executor);
        let covered = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(covered, KEY, "covered tokens");
        executor
            .prefill(&mut session, &PREFIX_PROMPT[KEY..])
            .unwrap();
        assert_eq!(take_log(&executor), (vec![2], 1), "chunks of the rest");
        assert_eq!(session.state.pos(), want.pos, "fed positions");
        assert_eq!(session.core.history(), want.history, "history");
        assert_eq!(session.core.pending(), want.pending, "pending token");
        let rows: Vec<Vec<f32>> = std::iter::once(session.core.pending().unwrap())
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
        assert_eq!(rows.len(), want.rows.len(), "row count");
        // The chunk boundaries differ, so the rows match within the tolerance of gate G2.
        for (n, (got, want)) in rows.iter().zip(&want.rows).enumerate() {
            assert_parity(got, want, &format!("logits row {n}"));
        }
    }

    #[test]
    fn an_engine_reuses_a_saved_prefix_between_two_jobs_and_never_across_other_prompts() {
        use crate::sched::{
            job::{Job, JobEvent, JobUsage},
            worker::{EngineState, ShutdownReport, spawn_engine},
        };
        let _serial = crate::native_test_lock();
        const COUNT: usize = 4;
        // The token-dependent table: with the flat table the logits of two keys differ only in
        // the rounding bits (see `Embedding`).
        let table = Embedding::TokenDependent;
        let prompt_a = PREFIX_PROMPT.to_vec();
        let prompt_b = vec![6, 4, 7, 1, 5, 3];
        // The first four tokens of A: the same length as the saved key.
        let prompt_e = prompt_a[..KEY].to_vec();
        assert_ne!(
            bits(&logits_alone_with(table, &prompt_a[..KEY])),
            bits(&logits_alone_with(table, &prompt_b[..KEY])),
            "the two keys must give different logits, or the test is vacuous"
        );
        let chain =
            |prompt: &[u32]| oracle_with(table, prompt, COUNT, &mut seeded_sampler(), VOCAB);
        let (want_a, want_b, want_e) = (chain(&prompt_a), chain(&prompt_b), chain(&prompt_e));
        // The prefill chunk of 3 splits the 4 key tokens into two pieces.
        let engine = spawn_engine(3, move || Ok(build_with(table, rules(vec![], VOCAB)))).unwrap();
        // Each round: the prompt, the boundary of the job, the expected tokens, and the expected
        // reused count.
        let rounds = [
            // The first A saves the key of 4 tokens.
            (&prompt_a, KEY, &want_a, 0u32),
            // The next two A restore it, so the entry survives a restore.
            (&prompt_a, KEY, &want_a, 4),
            (&prompt_a, KEY, &want_a, 4),
            // A prompt that equals the key restores nothing. It saves a key of 3 tokens.
            (&prompt_e, KEY - 1, &want_e, 0),
            // The key of 3 tokens is a prefix of A: 3 tokens are covered, and the key of 4 follows.
            (&prompt_a, KEY, &want_a, 3),
            // B replaces the entry, so the last A restores nothing.
            (&prompt_b, KEY, &want_b, 0),
            (&prompt_a, KEY, &want_a, 0),
        ];
        for (round, (prompt, boundary, want, reused)) in rounds.into_iter().enumerate() {
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
                    reused_tokens: reused
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

    #[test]
    fn a_prefix_restores_only_under_the_chunk_size_that_built_it() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        assert_eq!(plan_chunk(&mut executor, 4), Some(4), "plan");
        assert!(
            save_key(&mut executor, &mut session),
            "the save was refused"
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
    fn a_chunk_size_of_one_is_the_one_token_path() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
        executor.force_chunk(Some(1));
        assert!(
            save_key(&mut executor, &mut session),
            "the save was refused"
        );
        assert_eq!(
            take_log(&executor),
            (vec![], 0),
            "a size of one makes no chunk"
        );
        executor.force_chunk(None);
        executor.reset(&mut session).unwrap();
        let covered = executor
            .restore_prefix(&mut session, &PREFIX_PROMPT)
            .unwrap();
        assert_eq!(covered, KEY, "covered tokens under one-token steps");
    }

    #[test]
    fn a_save_is_refused_when_the_model_is_not_at_the_position_of_the_key() {
        let _serial = crate::native_test_lock();
        let (mut executor, mut session) = build(rules(vec![], VOCAB));
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
        let refused = executor.save_prefix(&session, &PREFIX_PROMPT[..KEY]);
        assert!(
            matches!(refused, Err(EngError::BadRequest(_))),
            "{refused:?}"
        );
        assert_eq!(executor.prefix.key_len(), None, "no entry");
        // Each committed step charges one position of QSA growth (640 bytes in the fixture). The
        // key prefill fed KEY - 1 = 3 positions (the last key token stays pending) and the extra
        // step fed one more, so 4 steps are charged. The refused save took no snapshot.
        assert_eq!(
            executor.decoder.session_used(),
            KEY as u64 * QSA_PER_POSITION,
            "four fed positions and no snapshot"
        );
    }

    /// The real artifact: a saved prefix of 250 of 300 prompt tokens gives the same 32 greedy tokens
    /// as a full prefill. The prompt tokens are synthetic ids, so the text is meaningless; the test
    /// compares ids. The coordinator runs it with `--ignored --exact` under the lock (about 4 minutes).
    /// Environment: `FLASH_NEXT_ARTIFACT`, `FLASH_NEXT_ARTIFACT_MANIFEST`, `FLASH_NEXT_CACHE_LIMIT`,
    /// `FLASH_NEXT_PREFIX_REPORT` (the report file). `FLASH_NEXT_PREFIX_CHUNK` (rows, optional) runs
    /// the batched prefill of that chunk size on both runs. Without it the run uses one-token steps,
    /// the path with bitwise equality (addendum item 8). With it, the chunk boundaries of the two
    /// runs differ, and the result is the evidence for the tolerance gate that the addendum requires.
    #[test]
    #[ignore = "needs the Flash-Next artifact on the Mac"]
    fn a_restored_prefix_gives_the_same_32_greedy_tokens_on_the_real_artifact() {
        use super::super::super::artifact::tokenizer_pin;
        use super::super::super::mlx::tests::env_path;
        use crate::tokenizer::TokenizerBundle;
        let _serial = crate::native_test_lock();
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let manifest = std::fs::read(env_path("FLASH_NEXT_ARTIFACT_MANIFEST")).unwrap();
        let report = env_path("FLASH_NEXT_PREFIX_REPORT");
        let cache_limit: u64 = std::env::var("FLASH_NEXT_CACHE_LIMIT")
            .expect("FLASH_NEXT_CACHE_LIMIT is not set")
            .parse()
            .expect("FLASH_NEXT_CACHE_LIMIT is not a byte count");
        let chunk: Option<u32> = std::env::var("FLASH_NEXT_PREFIX_CHUNK").ok().map(|rows| {
            rows.parse()
                .expect("FLASH_NEXT_PREFIX_CHUNK is not a number")
        });
        let pin = tokenizer_pin(&manifest, 248_077).unwrap();
        let bundle = TokenizerBundle::load(&artifact, &pin).unwrap();
        // No EOS id, so that every job runs the full 32 tokens.
        let rules = FlashRules {
            eos: Vec::new(),
            sampler_vocab: usize::try_from(bundle.vocab_size()).unwrap(),
        };
        let (mut executor, mut session) =
            FlashExecutor::open(&artifact, cache_limit, rules).unwrap();
        executor.force_chunk(chunk);
        let path = executor.prefill_path();
        let prompt: Vec<u32> = (0..300u32).map(|i| 1_000 + (i * 7_919) % 50_000).collect();
        let key = &prompt[..250];
        let decode32 = |executor: &mut FlashExecutor, session: &mut FlashSession| {
            let mut state = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()))
                .with_max_tokens(32);
            let mut tokens = Vec::new();
            for _ in 0..40 {
                let batch = executor.decode(session, &mut state).unwrap();
                tokens.extend(batch.tokens);
                if batch.finish.is_some() {
                    return tokens;
                }
            }
            panic!("the job did not finish in 40 decode calls");
        };
        // Run A: one full prefill of 300 tokens.
        executor.reset(&mut session).unwrap();
        let started = std::time::Instant::now();
        executor.prefill(&mut session, &prompt).unwrap();
        let full_prefill_seconds = started.elapsed().as_secs_f64();
        let full = decode32(&mut executor, &mut session);
        // Run B: prefill the key, save, reset, restore, prefill the last 50 tokens.
        executor.reset(&mut session).unwrap();
        executor.prefill(&mut session, key).unwrap();
        assert!(
            executor.save_prefix(&session, key).unwrap(),
            "the save was denied"
        );
        executor.reset(&mut session).unwrap();
        let started = std::time::Instant::now();
        let covered = executor.restore_prefix(&mut session, &prompt).unwrap();
        assert_eq!(covered, 250, "covered tokens");
        executor.prefill(&mut session, &prompt[250..]).unwrap();
        let restored_prefill_seconds = started.elapsed().as_secs_f64();
        let reused = decode32(&mut executor, &mut session);
        std::fs::write(
            &report,
            format!(
                "prompt_tokens=300\nkey_tokens=250\nprefill_path={path:?}\nfull_prefill_seconds={full_prefill_seconds}\nrestored_prefill_seconds={restored_prefill_seconds}\nfull_tokens={full:?}\nreused_tokens={reused:?}\n"
            ),
        )
        .unwrap();
        assert_eq!(full.len(), 32, "token count of the full run");
        assert_eq!(reused, full, "32 greedy tokens with and without reuse");
    }

    #[test]
    fn an_engine_thread_builds_the_executor_runs_two_jobs_and_shuts_down() {
        three_jobs_on_an_engine_thread(2, None, (&[], 0));
    }

    /// The same flow with the batched prefill planned for chunks of 4 and worker pieces of 4. The
    /// tokens must equal the one-token oracle, and the executor log must show that the chunk path
    /// ran: prompt A (3 tokens) feeds one chunk of 2 rows. Prompt B (5 tokens) feeds one chunk of
    /// 3 rows in its first piece and a single step row in its second piece. Prompt A again feeds
    /// one chunk of 2 rows. Each job opens the page sets once, because a reset closes them.
    #[test]
    fn an_engine_thread_with_a_planned_chunk_gives_the_one_token_tokens() {
        three_jobs_on_an_engine_thread(4, Some(4), (&[2, 3, 2], 3));
    }

    /// `want_log` is the chunk row counts and the page opens that the executor must log.
    fn three_jobs_on_an_engine_thread(
        worker_chunk: usize,
        planned: Option<u32>,
        want_log: (&[usize], usize),
    ) {
        use crate::sched::{
            job::{Job, JobEvent, JobUsage},
            worker::{EngineState, ShutdownReport, spawn_engine},
        };
        let _serial = crate::native_test_lock();
        // The prompts A, B, A with a seeded sampler: The test checks the A, B, A flow, the tokens,
        // and the usage through the engine thread. The reset proof is in
        // each_job_after_a_reset_matches_the_same_job_run_alone_in_the_order_a_b_a. The prefill
        // chunk of 2 splits each prompt into chunks.
        const COUNT: usize = 8;
        let prompt_b = vec![6, 4, 7, 1, 5];
        assert_ne!(
            bits(&logits_alone(&PROMPT)),
            bits(&logits_alone(&prompt_b)),
            "the two prompts must give different logits, or the test is vacuous"
        );
        let want_a = oracle(&PROMPT, COUNT, &mut seeded_sampler(), VOCAB);
        let want_b = oracle(&prompt_b, COUNT, &mut seeded_sampler(), VOCAB);
        let log = std::sync::Arc::new(std::sync::Mutex::new(ChunkLog::default()));
        let thread_log = std::sync::Arc::clone(&log);
        let engine = spawn_engine(worker_chunk, move || {
            let (mut executor, session) = build(rules(vec![], VOCAB));
            executor.log = thread_log;
            if let Some(rows) = planned {
                assert_eq!(plan_chunk(&mut executor, rows), planned, "plan");
            }
            Ok((executor, session))
        })
        .unwrap();
        for (round, (prompt, want)) in [
            (PROMPT.to_vec(), &want_a),
            (prompt_b.clone(), &want_b),
            (PROMPT.to_vec(), &want_a),
        ]
        .into_iter()
        .enumerate()
        {
            let prompt_tokens = u32::try_from(prompt.len()).unwrap();
            let (job, mut handle) = Job::new(prompt, seeded_generation(8)).unwrap();
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
                    completion_tokens: 8,
                    reused_tokens: 0
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
        let log = log.lock().unwrap();
        assert_eq!(
            (log.chunks.as_slice(), log.opens),
            want_log,
            "the chunk runs and page opens of the engine thread"
        );
    }

    /// The real artifact on the Mac. The coordinator runs it with `--ignored --exact` under the lock.
    /// Environment: `FLASH_NEXT_ARTIFACT`, `FLASH_NEXT_ARTIFACT_MANIFEST`, `FLASH_NEXT_CACHE_LIMIT`,
    /// `FLASH_NEXT_EXECUTOR_REPORT` (the report file). `FLASH_NEXT_EXECUTOR_BUDGET` (bytes, optional)
    /// plans the batched prefill (packet H8b): the run then uses chunks, and the report names the
    /// chunk size. Without it the run uses one-token steps, as before.
    #[test]
    #[ignore = "needs the Flash-Next artifact on the Mac"]
    fn the_executor_serves_the_real_artifact_with_the_device_token_chain() {
        use super::super::super::artifact::tokenizer_pin;
        use super::super::super::mlx::tests::{PROMPT as C5_PROMPT, env_path};
        use crate::tokenizer::TokenizerBundle;
        let _serial = crate::native_test_lock();
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let manifest = std::fs::read(env_path("FLASH_NEXT_ARTIFACT_MANIFEST")).unwrap();
        let report = env_path("FLASH_NEXT_EXECUTOR_REPORT");
        let cache_limit: u64 = std::env::var("FLASH_NEXT_CACHE_LIMIT")
            .expect("FLASH_NEXT_CACHE_LIMIT is not set")
            .parse()
            .expect("FLASH_NEXT_CACHE_LIMIT is not a byte count");
        // The tokenizer files come from the artifact directory and are checked against the manifest.
        let pin = tokenizer_pin(&manifest, 248_077).unwrap();
        let bundle = TokenizerBundle::load(&artifact, &pin).unwrap();
        assert_eq!(bundle.eos_token_ids(), [248_046, 248_044], "EOS ids");
        let count = 32;
        let config = std::fs::read(artifact.join("config.json")).unwrap();
        // The device chain: `Generate` steps on a decoder that drops before the executor loads.
        let device: Vec<u32> = {
            let ckpt = open_checkpoint(&artifact).unwrap();
            let flash = FlashWeights::load(&ckpt, &config).unwrap();
            let mut kernels = MlxKernels::try_new().unwrap();
            kernels.set_cache_limit(cache_limit).unwrap();
            let mut decoder = ResidentDecoder::new(kernels, &ckpt, &flash).unwrap();
            drop(flash);
            let mut state = decoder.zero_state().unwrap();
            let mut next = 0;
            let mut chain = Vec::new();
            for n in 0..C5_PROMPT.len() - 1 + count {
                let input = C5_PROMPT.get(n).copied().unwrap_or(next);
                let mode = if n + 1 < C5_PROMPT.len() {
                    StepMode::Prompt
                } else {
                    StepMode::Generate
                };
                let out = decoder.step(&mut state, input, mode).unwrap();
                if let Some(token) = out.token {
                    chain.push(token);
                    next = token;
                }
            }
            chain
        };
        let rules = FlashRules {
            eos: bundle.eos_token_ids().to_vec(),
            sampler_vocab: usize::try_from(bundle.vocab_size()).unwrap(),
        };
        let (mut executor, mut session) =
            FlashExecutor::open(&artifact, cache_limit, rules).unwrap();
        let context_limit = executor.context_limit();
        let reserve = Reserve::of_real_run(&executor, &session, cache_limit);
        let (chunk, plan_report) = match std::env::var("FLASH_NEXT_EXECUTOR_BUDGET") {
            Ok(budget) => {
                let budget = budget
                    .parse()
                    .expect("FLASH_NEXT_EXECUTOR_BUDGET is not a number");
                let plan = executor.plan_prefill(budget, 0, reserve.total()).unwrap();
                (plan.chunk, format!("{}{}", plan.terms, reserve.report()))
            }
            Err(_) => (None, "no budget: one-token steps\n".to_string()),
        };
        let mut job = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()))
            .with_max_tokens(u32::try_from(count).unwrap());
        let started = std::time::Instant::now();
        executor.prefill(&mut session, &C5_PROMPT).unwrap();
        let prefill_seconds = started.elapsed().as_secs_f64();
        let pages_open = executor.decoder.prefill_pages.is_some();
        let (chunk_rows, _) = take_log(&executor);
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
                "executor_tokens={}\ndevice_tokens={}\nprefill_seconds={prefill_seconds}\n\
                 decode_seconds={decode_seconds}\ncontext_limit={context_limit}\ndecode_calls={}\n\
                 finish={:?}\nprefill_chunk={chunk:?}\nprefill_chunk_rows={chunk_rows:?}\n\
                 prefill_pages_open_after_prefill={pages_open}\n{plan_report}",
                join(&got),
                join(&device),
                batches.len(),
                batches.last().and_then(|b| b.finish),
            ),
        )
        .unwrap();
        if std::env::var("FLASH_NEXT_EXECUTOR_BUDGET").is_ok() {
            assert!(chunk.is_some(), "the budget admits no chunk");
        }
        if let Some(size) = chunk {
            let want = expected_chunks(C5_PROMPT.len() - 1, usize::try_from(size).unwrap());
            assert_eq!(chunk_rows, want, "the chunk path did not run as planned");
            assert!(
                pages_open || want.is_empty(),
                "prefill pages were not opened after the prefill"
            );
        }
        assert_eq!(got, device, "executor tokens differ from the device chain");
        assert_eq!(batches.len(), count, "decode calls");
        assert_eq!(
            batches.last().and_then(|b| b.finish),
            Some(DecodeFinish::Length),
            "the last batch must end with Length"
        );
        assert!(
            got.iter().any(|&t| t != got[0]),
            "the token chain is constant, so a one-token shift would pass"
        );
    }

    /// The counters of a qualification build at one instant: the forced flush window, the
    /// synchronizing calls, and the expert reads and page imports. All are cumulative from the
    /// start of the run.
    #[cfg(feature = "native-fault-qualification")]
    type Counters = (
        crate::kernels::QualificationFlushStats,
        crate::kernels::QualificationTiming,
        crate::family::flash_next::stored::ExpertReadStats,
    );

    #[cfg(feature = "native-fault-qualification")]
    fn counters(executor: &FlashExecutor) -> Counters {
        let kernels = &executor.decoder.kernels;
        (
            kernels.qualification_flush_stats(),
            kernels.qualification_timing(),
            crate::family::flash_next::stored::qualification_expert_reads(),
        )
    }

    /// One worker piece of the rate test: its rows and wall seconds, and in a qualification build
    /// the counters before and after it. A piece of the planned chunk size is one chunk.
    struct PieceRun {
        rows: usize,
        seconds: f64,
        #[cfg(feature = "native-fault-qualification")]
        before: Counters,
        #[cfg(feature = "native-fault-qualification")]
        after: Counters,
    }

    #[cfg(feature = "native-fault-qualification")]
    impl PieceRun {
        /// The report lines of piece `n`: wall seconds and the parts that the counters cover.
        /// `gpu_wait` is the time blocked in `evaluate` and `readback`. `other` is what is left of
        /// the wall time after the flush, the expert reads, the page imports, and the GPU wait
        /// (host graph building, token imports). It can be negative because a forced flush that
        /// runs inside a timed call is counted twice.
        fn report(&self, run: &str, n: usize) -> String {
            let (b, a) = (&self.before, &self.after);
            let secs = |after: u64, before: u64| (after - before) as f64 / 1e9;
            let flush = secs(a.0.sync_ns, b.0.sync_ns);
            let free = secs(a.0.free_ns, b.0.free_ns);
            let (import, readback, evaluate) = (
                secs(a.1.import_ns, b.1.import_ns),
                secs(a.1.readback_ns, b.1.readback_ns),
                secs(a.1.evaluate_ns, b.1.evaluate_ns),
            );
            let (read, page_import) = (
                secs(a.2.read_ns, b.2.read_ns),
                secs(a.2.import_ns, b.2.import_ns),
            );
            let gpu_wait = evaluate + readback;
            let other = self.seconds - flush - free - read - import - gpu_wait;
            let p = format!("{run}_piece{n}");
            format!(
                "{p}_rows={}\n{p}_seconds={}\n{p}_forced_flushes={}\n\
                 {p}_forced_flush_seconds={flush}\n{p}_forced_flush_free_seconds={free}\n\
                 {p}_expert_read_seconds={read}\n{p}_expert_read_bytes={}\n\
                 {p}_expert_slice_reads={}\n{p}_expert_page_imports={}\n\
                 {p}_import_seconds={import}\n{p}_expert_page_import_seconds={page_import}\n\
                 {p}_gpu_wait_seconds={gpu_wait}\n{p}_evaluate_seconds={evaluate}\n\
                 {p}_readback_seconds={readback}\n{p}_other_seconds={other}\n",
                self.rows,
                self.seconds,
                a.0.flushes - b.0.flushes,
                a.2.read_bytes - b.2.read_bytes,
                a.2.slice_reads - b.2.slice_reads,
                a.2.page_imports - b.2.page_imports,
            )
        }
    }

    /// The prefill rate of a long prompt on the real artifact (packet H8b; the gate of decision 0044
    /// is 60 prompt tokens per second on 2,048 tokens, measured cold). The coordinator evicts the
    /// checkpoint pages first (`prefill-speed coldprep`). The test runs the prompt twice with
    /// worker pieces of the planned chunk size: `first` is the cold run, `repeat` the warm one.
    /// Environment: `FLASH_NEXT_ARTIFACT`, `FLASH_NEXT_CACHE_LIMIT`, `FLASH_NEXT_EXECUTOR_PROMPT`
    /// (a fixture JSON with `tokens`), `FLASH_NEXT_EXECUTOR_BUDGET`, `FLASH_NEXT_EXECUTOR_S3_CACHE`
    /// and `FLASH_NEXT_EXECUTOR_RESERVE` (bytes, default 0), `FLASH_NEXT_EXECUTOR_CHUNK` (optional,
    /// forces the chunk size for a sweep), `FLASH_NEXT_EXECUTOR_REPORT`. The report names every
    /// term of the plan, and in a qualification build the forced flushes and the peak retired
    /// bytes of each run (decision 0044 review, F4).
    ///
    /// Each run also reports the MLX allocator peak, active, and cache bytes (process-wide; the
    /// peak starts at the active bytes after the session reset). A qualification build also
    /// reports the wall time of the forced flushes, the retired bytes at each flush, and the
    /// synchronizing calls of each run (the flush-cost measurement). Two optional variables
    /// override the prefill caps for a sweep, and only in a qualification build:
    /// `FLASH_NEXT_EXECUTOR_PREFILL_RETIRE_CAP` (bytes; the retire cap of the chunk, which also
    /// sizes the DeltaNet sub-chunks) and `FLASH_NEXT_EXECUTOR_PREFILL_SUBCHUNK_CAP` (bytes; the
    /// sub-chunk sizing only). The admission plan keeps `PREFILL_RETIRE_CAP`, so a cap above it
    /// runs outside the planned memory: the report prints both caps and the peak retired bytes.
    #[test]
    #[ignore = "needs the Flash-Next artifact on the Mac"]
    fn the_executor_reports_the_prefill_rate_of_a_long_prompt() {
        use super::super::super::mlx::tests::env_path;
        let _serial = crate::native_test_lock();
        let env = |name: &str, default: Option<u64>| -> u64 {
            std::env::var(name).map_or_else(
                |_| default.unwrap_or_else(|| panic!("{name} is not set")),
                |v| {
                    v.parse()
                        .unwrap_or_else(|_| panic!("{name} is not a number"))
                },
            )
        };
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let fixture = std::fs::read(env_path("FLASH_NEXT_EXECUTOR_PROMPT")).unwrap();
        let tokens: Vec<u32> =
            serde_json::from_slice::<serde_json::Value>(&fixture).unwrap()["tokens"]
                .as_array()
                .expect("the fixture has no tokens")
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
        let rules = FlashRules {
            eos: vec![],
            sampler_vocab: 248_077,
        };
        let (mut executor, mut session) =
            FlashExecutor::open(&artifact, env("FLASH_NEXT_CACHE_LIMIT", None), rules).unwrap();
        let cache_limit = env("FLASH_NEXT_CACHE_LIMIT", None);
        let (budget, s3) = (
            env("FLASH_NEXT_EXECUTOR_BUDGET", None),
            env("FLASH_NEXT_EXECUTOR_S3_CACHE", Some(0)),
        );
        let reserve = Reserve::of_real_run(&executor, &session, cache_limit);
        let plan = executor.plan_prefill(budget, s3, reserve.total()).unwrap();
        let forced = std::env::var("FLASH_NEXT_EXECUTOR_CHUNK").ok().map(|v| {
            v.parse::<u32>()
                .expect("FLASH_NEXT_EXECUTOR_CHUNK is not a number")
        });
        if forced.is_some() {
            executor.force_chunk(forced);
        }
        let report = env_path("FLASH_NEXT_EXECUTOR_REPORT");
        let mut lines = format!(
            "{}{}planned_chunk={:?}\nchunk={:?}\nprompt_tokens={}\n\
             note=tokens_per_second divides all prompt tokens by the time, but the last token stays \
             pending, so the run feeds one row fewer (about 0.05 percent at 2,048 tokens)\n",
            plan.terms,
            reserve.report(),
            plan.chunk,
            executor.prefill_chunk(),
            tokens.len()
        );
        #[cfg(feature = "native-fault-qualification")]
        let _caps = {
            let cap = |name: &str| -> Option<u64> {
                std::env::var(name).ok().map(|v| {
                    v.parse()
                        .unwrap_or_else(|_| panic!("{name} is not a number"))
                })
            };
            let retire = cap("FLASH_NEXT_EXECUTOR_PREFILL_RETIRE_CAP");
            let subchunk = cap("FLASH_NEXT_EXECUTOR_PREFILL_SUBCHUNK_CAP");
            lines.push_str(&format!(
                "prefill_retire_cap_plan_bytes={}\nprefill_retire_cap_used_bytes={}\n\
                 prefill_subchunk_cap_used_bytes={}\n",
                prefill_plan::PREFILL_RETIRE_CAP,
                retire.unwrap_or(prefill_plan::PREFILL_RETIRE_CAP),
                subchunk
                    .or(retire)
                    .unwrap_or(prefill_plan::PREFILL_RETIRE_CAP),
            ));
            super::super::prefill::step::PrefillCapOverride::set(retire, subchunk)
        };
        #[cfg(not(feature = "native-fault-qualification"))]
        assert!(
            std::env::var("FLASH_NEXT_EXECUTOR_PREFILL_RETIRE_CAP").is_err()
                && std::env::var("FLASH_NEXT_EXECUTOR_PREFILL_SUBCHUNK_CAP").is_err(),
            "the prefill cap overrides need the native-fault-qualification build"
        );
        // Write the terms before any failure: a plan that admits no chunk is a result to report.
        std::fs::write(&report, &lines).unwrap();
        let chunk = executor
            .prefill_chunk()
            .expect("no chunk size fits the budget");
        let width = usize::try_from(chunk).unwrap();
        let mut mismatches = Vec::new();
        for run in ["first", "repeat"] {
            executor.reset(&mut session).unwrap();
            // The windows start after `reset`, so they cover the timed prefill only.
            executor.decoder.kernels.reset_peak_memory().unwrap();
            #[cfg(feature = "native-fault-qualification")]
            executor.decoder.kernels.qualification_reset_retire_peak();
            #[cfg(feature = "native-fault-qualification")]
            executor.decoder.kernels.qualification_reset_flush_stats();
            #[cfg(feature = "native-fault-qualification")]
            executor.decoder.kernels.qualification_reset_timing();
            #[cfg(feature = "native-fault-qualification")]
            crate::family::flash_next::stored::qualification_reset_expert_reads();
            #[cfg(feature = "native-fault-qualification")]
            let flushes = executor.decoder.kernels.qualification_retire_stats().0;
            let started = std::time::Instant::now();
            let mut pieces = Vec::new();
            for piece in tokens.chunks(width) {
                #[cfg(feature = "native-fault-qualification")]
                let before = counters(&executor);
                let piece_started = std::time::Instant::now();
                executor.prefill(&mut session, piece).unwrap();
                pieces.push(PieceRun {
                    rows: piece.len(),
                    seconds: piece_started.elapsed().as_secs_f64(),
                    #[cfg(feature = "native-fault-qualification")]
                    before,
                    #[cfg(feature = "native-fault-qualification")]
                    after: counters(&executor),
                });
            }
            let seconds = started.elapsed().as_secs_f64();
            assert_eq!(session.core.history(), tokens, "{run}: history");
            let (chunk_rows, opens) = take_log(&executor);
            let want: Vec<usize> = tokens
                .chunks(width)
                .enumerate()
                .flat_map(|(n, piece)| expected_chunks(usize::from(n > 0) + piece.len() - 1, width))
                .collect();
            lines.push_str(&format!(
                "{run}_chunk_runs={}\n{run}_page_opens={opens}\n",
                chunk_rows.len()
            ));
            if chunk_rows != want {
                mismatches.push(format!("{run}: chunk rows {chunk_rows:?}, want {want:?}"));
            }
            let join = |values: Vec<String>| values.join(",");
            lines.push_str(&format!(
                "{run}_piece_rows={}\n{run}_piece_seconds={}\n",
                join(pieces.iter().map(|p| p.rows.to_string()).collect()),
                join(pieces.iter().map(|p| p.seconds.to_string()).collect()),
            ));
            let memory = executor.decoder.kernels.memory_stats().unwrap();
            lines.push_str(&format!(
                "{run}_seconds={seconds}\n{run}_tokens_per_second={}\n\
                 {run}_mlx_peak_bytes={}\n{run}_mlx_active_bytes={}\n{run}_mlx_cache_bytes={}\n",
                tokens.len() as f64 / seconds,
                memory.peak,
                memory.active,
                memory.cache
            ));
            #[cfg(feature = "native-fault-qualification")]
            {
                let kernels = &executor.decoder.kernels;
                let (total, peak) = kernels.qualification_retire_stats();
                let (flush, timing) = (
                    kernels.qualification_flush_stats(),
                    kernels.qualification_timing(),
                );
                let secs = |ns: u64| ns as f64 / 1e9;
                let (sync, free) = (secs(flush.sync_ns), secs(flush.free_ns));
                let mean = flush.bytes_sum.checked_div(flush.flushes).unwrap_or(0);
                lines.push_str(&format!(
                    "{run}_forced_flushes={}\n{run}_peak_retired_bytes={peak}\n\
                     {run}_forced_flush_seconds={sync}\n{run}_forced_flush_free_seconds={free}\n\
                     {run}_forced_flush_share_of_run={}\n\
                     {run}_forced_flush_and_free_share_of_run={}\n\
                     {run}_flush_bytes_min={}\n{run}_flush_bytes_max={}\n{run}_flush_bytes_mean={mean}\n\
                     {run}_import_seconds={}\n{run}_import_calls={}\n\
                     {run}_readback_seconds={}\n{run}_readback_calls={}\n\
                     {run}_evaluate_seconds={}\n{run}_evaluate_calls={}\n",
                    total - flushes,
                    sync / seconds,
                    (sync + free) / seconds,
                    flush.bytes_min,
                    flush.bytes_max,
                    secs(timing.import_ns),
                    timing.import_calls,
                    secs(timing.readback_ns),
                    timing.readback_calls,
                    secs(timing.evaluate_ns),
                    timing.evaluate_calls,
                ));
                let reads = crate::family::flash_next::stored::qualification_expert_reads();
                lines.push_str(&format!(
                    "{run}_expert_read_seconds={}\n{run}_expert_read_bytes={}\n\
                     {run}_expert_slice_reads={}\n{run}_expert_page_imports={}\n\
                     {run}_expert_page_import_seconds={}\n",
                    secs(reads.read_ns),
                    reads.read_bytes,
                    reads.slice_reads,
                    reads.page_imports,
                    secs(reads.import_ns),
                ));
                for (n, piece) in pieces.iter().enumerate() {
                    lines.push_str(&piece.report(run, n));
                }
                // A chunk reads experts and imports pages, and the pieces cover the whole run.
                let piece_bytes: u64 = pieces
                    .iter()
                    .map(|p| p.after.2.read_bytes - p.before.2.read_bytes)
                    .sum();
                if reads.read_bytes == 0 || reads.slice_reads == 0 || reads.page_imports == 0 {
                    mismatches.push(format!("{run}: no expert read or page import counted"));
                }
                if piece_bytes != reads.read_bytes {
                    mismatches.push(format!(
                        "{run}: pieces read {piece_bytes} bytes, the run {}",
                        reads.read_bytes
                    ));
                }
                // The two flush counters must agree, and the flush sizes sit below the peak.
                if flush.flushes != total - flushes {
                    mismatches.push(format!(
                        "{run}: flush window {} differs from the forced flush count {}",
                        flush.flushes,
                        total - flushes
                    ));
                }
                if flush.bytes_min > flush.bytes_max || flush.bytes_max > peak {
                    mismatches.push(format!(
                        "{run}: flush bytes min {} max {} peak {peak}",
                        flush.bytes_min, flush.bytes_max
                    ));
                }
            }
        }
        std::fs::write(&report, lines).unwrap();
        assert!(
            mismatches.is_empty(),
            "the chunk path did not run as planned: {mismatches:?}"
        );
    }
}
