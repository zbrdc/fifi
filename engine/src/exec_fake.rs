//! The scripted test fake of the executor seam.
//!
//! The trait and `GenerationState` live in `crate::family::executor`.
//!
//! The fake keeps the real prefix rules: its session holds a real `DecodeSession` for the token
//! side, and `save_prefix` / `restore_prefix` run through `DecodeSession::save_prefix` /
//! `restore_prefix` and a real `PrefixStore`, the same code path the resident executors use. The
//! saved state is `()`: the fake has no model, and the store never inspects a state.

use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::session::{
    DecodeSession, PrefixStore, SaveKind, ShadowLcp, StoreLimits, TokenStepper,
};
use crate::sched::job::{PrefixHit, PrefixReport};
use crate::tokenizer::{DecodeBatch, DecodeFinish};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

type Result<T> = std::result::Result<T, EngError>;

/// Locks the shared store. A poisoned lock means a test thread panicked while saving; the
/// store is still readable, so the poison is cleared like every other lock in this crate.
fn lock_store(
    store: &Mutex<PrefixStore<FakeSnapshot, u32>>,
) -> MutexGuard<'_, PrefixStore<FakeSnapshot, u32>> {
    store.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The saved state of a fake prefix entry: it holds the entry's budget permit, like the resident
/// executors' snapshots, so dropping an entry releases its charge. Without a budget it holds
/// nothing.
pub(crate) struct FakeSnapshot {
    #[allow(dead_code)]
    charge: Option<TestPermit>,
}

/// The fake's own test budget: unlike the production `SessionBudget`, whose `Rc` cells confine it
/// to one thread from construction, this one is built with atomics so the fake can be created on
/// the test thread and moved into the engine's worker thread. The worker drives it single-threaded,
/// so the check-then-add of `acquire` sees no races.
struct TestBudget {
    cap: u64,
    used: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// One charge against the fake's test budget. Dropping it releases the bytes.
struct TestPermit {
    used: std::sync::Arc<std::sync::atomic::AtomicU64>,
    bytes: u64,
}

impl TestBudget {
    fn new(cap: u64) -> Self {
        Self {
            cap,
            used: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    fn used(&self) -> u64 {
        self.used.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn free(&self) -> u64 {
        self.cap.saturating_sub(self.used())
    }

    fn acquire(&self, bytes: u64) -> Result<TestPermit> {
        let used = self.used();
        let next = used
            .checked_add(bytes)
            .ok_or_else(|| crate::EngError::MemoryPlan("session growth overflows u64".into()))?;
        if next > self.cap {
            return Err(crate::EngError::MemoryPlan(format!(
                "session growth requires {bytes} bytes, available {}",
                self.free()
            )));
        }
        self.used.store(next, std::sync::atomic::Ordering::Relaxed);
        Ok(TestPermit {
            used: std::sync::Arc::clone(&self.used),
            bytes,
        })
    }
}

impl TestPermit {
    fn retained_bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for TestPermit {
    fn drop(&mut self) {
        self.used
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The fake's model side: it counts the fed positions and has no logits. It lives in the
/// session, like the model state of the resident executors, so its positions reset with the
/// session instead of surviving it.
#[derive(Default)]
struct CountingStepper {
    fed: usize,
}

impl TokenStepper for CountingStepper {
    const OPERATION: &'static str = "fake";

    fn feed(&mut self, _token: u32) -> Result<()> {
        self.fed += 1;
        Ok(())
    }

    fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
        Err(EngError::BadRequest("the fake has no logits".to_string()))
    }

    fn fed(&self) -> usize {
        self.fed
    }

    fn reset(&mut self) -> Result<()> {
        self.fed = 0;
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct FakeSession {
    /// Every committed token: the prompt and the scripted tokens, like the history of a real
    /// session after decode.
    history: Vec<u32>,
    /// The token side of the prompt, driven through the real `DecodeSession` prefill and
    /// prefix rules. The fake's scripted `decode` cannot enter it, because the real `decode`
    /// samples one token per call from logits, so the prompt tokens stay here and in `history`.
    core: DecodeSession,
    /// The model side of the session: the count of fed positions.
    step: CountingStepper,
    /// The growth permits of the committed tokens: one per prefilled or decoded token batch, like
    /// the resident state's permits. `reset` and drop release them.
    permits: Vec<TestPermit>,
    finished: bool,
}

/// Returns scripted token batches and applies the `decode` finish rules: EOS, output limit, and
/// cancellation. It never matches stop strings. The output limit is the lower of the constructor
/// cap and the job limit in `GenerationState`. Its prefix store is a real `PrefixStore` with the
/// default limits (zero roots, one entry) or the limits that `with_prefix_limits` sets.
pub(crate) struct FakeExecutor {
    script: VecDeque<Vec<u32>>,
    eos: Vec<u32>,
    remaining: u32,
    /// The limits of the store, reported by `prefix_limits`.
    limits: StoreLimits,
    /// The real prefix store behind a lock, so a test can read it while the fake runs on the
    /// engine thread.
    prefix: Arc<Mutex<PrefixStore<FakeSnapshot, u32>>>,
    revision: u32,
    deny_saves: bool,
    /// Denies the pool charge for a snapshot at the executor level, before the store's
    /// graceful snapshot loop runs. This models an executor that charges pool memory in
    /// `save_prefix` itself: the denial surfaces as a `MemoryPlan` error, unlike
    /// `deny_saves`, which the store degrades to a skipped save.
    deny_save_charge: bool,
    /// Denies the charge of the first `n` save calls only, so a test can deny one save point of a
    /// job (a root) and let the later ones (the tip) run. A save call that the store serves as a
    /// use of a duplicate key never reaches the snapshot closure but still counts.
    deny_first_saves: usize,
    /// The number of save calls so far: one per `save_prefix` call, counted before the store
    /// decides whether a snapshot is needed. A save call that the store serves as a use of a
    /// duplicate key never reaches the snapshot closure but still counts.
    save_calls: usize,
    /// Fails the next snapshot with this error, before the store sees its result.
    fail_snapshot: Option<EngError>,
    /// The kind of the entry that the last `restore_prefix` call restored, for `prefix_report`. A
    /// reset clears it, so the report of a job without a boundary is a miss.
    prefix_hit: PrefixHit,
    /// The `uncached_shared` count of the last `restore_prefix` call, for `prefix_report`;
    /// a reset clears it too.
    prefix_uncached: u32,
    /// The session budget that prefill, decode, and prefix saves charge. `None` (the default) keeps
    /// the uncharged behavior of earlier tests.
    budget: Option<TestBudget>,
    /// The bytes one prefilled or decoded token charges when a budget is set.
    token_bytes: u64,
    /// The bytes one stored prefix entry charges when a budget is set.
    entry_bytes: u64,
    /// Whether the shadow block-LCP observation runs. A test turns it on with
    /// `with_shadow_block_lcp`, which runs before the conversation, because it rebuilds the store
    /// like `with_prefix_limits`.
    shadow_block_lcp: bool,
    /// The shadow observation of the last `restore_prefix` call, for `prefix_report`; a reset
    /// clears it, like `prefix_hit`.
    prefix_shadow: Option<ShadowLcp>,
    /// The block prefix-match spacing: `None` keeps the save lattice unchanged; `Some(tokens)` cuts
    /// one rolling intermediate entry per lineage. A test sets it with `with_block_match` before
    /// the conversation.
    block_spacing: Option<usize>,
    /// The reported `max_seqs`: the default 1 keeps the single-worker path - a test that wants its
    /// real store to run through the BATCH worker sets this above one before spawning the engine.
    max_seqs: usize,
    /// `decode_window` calls so far: the single worker never advances a window, so a positive count
    /// proves the batch branch ran.
    window_calls: Arc<std::sync::atomic::AtomicUsize>,
    /// Denies the next restore's adoption once. Set through `deny_restore_once`; consumed by the
    /// first restore whose closure runs.
    deny_restore_once: bool,
}

impl FakeExecutor {
    pub(crate) fn new(script: Vec<Vec<u32>>, eos: &[u32], max_tokens: u32) -> Self {
        Self {
            script: script.into(),
            eos: eos.to_vec(),
            remaining: max_tokens,
            limits: StoreLimits::default(),
            prefix: Arc::new(Mutex::new(PrefixStore::default())),
            revision: 1,
            deny_saves: false,
            deny_save_charge: false,
            deny_first_saves: 0,
            save_calls: 0,
            fail_snapshot: None,
            prefix_hit: PrefixHit::Miss,
            prefix_uncached: 0,
            budget: None,
            token_bytes: 0,
            entry_bytes: 0,
            shadow_block_lcp: false,
            prefix_shadow: None,
            block_spacing: None,
            max_seqs: 1,
            window_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            deny_restore_once: false,
        }
    }

    /// Runs the engine on the BATCH worker path with this fake's real store: any value above one
    /// makes `spawn_engine` select `batch_worker::work`, so a real-store fixture exercises the
    /// batch pre-pass and restart wiring.
    pub(crate) fn with_max_seqs(&mut self, max_seqs: usize) {
        self.max_seqs = max_seqs;
    }

    /// The shared `decode_window` counter, so a test can ASSERT the batch branch ran even after the
    /// fake moves into the engine factory: the single worker never advances a window.
    pub(crate) fn window_counter(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.window_calls)
    }

    /// Denies the NEXT restore's adoption once: the restore closure returns one `MemoryPlan` denial
    /// before succeeding, exercising the store's evict-one-victim-and-retry-same-candidate step
    /// through the full worker path.
    pub(crate) fn deny_restore_once(&mut self) {
        self.deny_restore_once = true;
    }

    /// Turns the block prefix-match on with the given spacing: the worker's save lattice then cuts
    /// one rolling intermediate entry per lineage at the last block boundary below the tip. Runs
    /// before the conversation, like `with_prefix_limits`.
    pub(crate) fn with_block_match(&mut self, tokens: usize) {
        self.block_spacing = Some(tokens);
    }

    /// Installs a session budget with the given cap and charges: every prefilled and decoded token
    /// charges `token_bytes`, every stored prefix entry `entry_bytes`. The charges model the
    /// resident executor's pre-mutation rule: the permit is acquired before any session state
    /// changes, so a denial leaves the session unchanged.
    pub(crate) fn with_session_budget(&mut self, cap: u64, token_bytes: u64, entry_bytes: u64) {
        self.budget = Some(TestBudget::new(cap));
        self.token_bytes = token_bytes;
        self.entry_bytes = entry_bytes;
    }

    /// Acquires `bytes` of growth from the installed budget, if any. A denial returns the
    /// memory-plan error unchanged.
    fn charge(&mut self, bytes: u64) -> Result<Option<TestPermit>> {
        match &self.budget {
            Some(budget) => Ok(Some(budget.acquire(bytes)?)),
            None => Ok(None),
        }
    }

    /// Sets the limits of the prefix store. The default limits are zero roots and one entry.
    pub(crate) fn with_prefix_limits(&mut self, roots: usize, entries: usize) {
        self.limits = StoreLimits { roots, entries };
        *lock_store(&self.prefix) = PrefixStore::with_limits(self.limits);
    }

    /// Turns the shadow block-LCP observation on. Like `with_prefix_limits`, it runs before
    /// the conversation holds entries, because it rebuilds the store.
    pub(crate) fn with_shadow_block_lcp(&mut self) {
        self.shadow_block_lcp = true;
        let mut store = PrefixStore::with_limits(self.limits);
        store.set_shadow_enabled(true);
        *lock_store(&self.prefix) = store;
    }

    /// The handle through which a test reads the store while the fake runs on its engine
    /// thread.
    pub(crate) fn prefix_store(&self) -> Arc<Mutex<PrefixStore<FakeSnapshot, u32>>> {
        Arc::clone(&self.prefix)
    }

    /// The stored prefix keys as `(length, is_root)`, roots first, for test assertions.
    pub(crate) fn stored_prefixes(&self) -> Vec<(usize, bool)> {
        lock_store(&self.prefix).keys()
    }

    /// Sets the numeric path revision. A saved prefix restores only under the revision that built it.
    pub(crate) fn set_numeric_revision(&mut self, revision: u32) {
        self.revision = revision;
    }

    /// Makes every later snapshot report a denied memory charge (`MemoryPlan`). The real store
    /// applies its own rules to the denial: it drops victims before each retry and skips the
    /// save when none is left, as it does for the resident executors.
    pub(crate) fn deny_saves(&mut self) {
        self.deny_saves = true;
    }

    /// Denies the pool charge for a snapshot at the executor level, before the store's
    /// graceful snapshot loop runs: the denial surfaces as a `MemoryPlan` error, unlike
    /// `deny_saves`, which the store degrades to a skipped save.
    pub(crate) fn deny_save_charge(&mut self) {
        self.deny_save_charge = true;
    }

    /// Denies the charge of the first `n` save calls only, so a test can deny one save point of a
    /// job (a root) while the later ones (the tip) still run. A save call that the store serves as
    /// a use of a duplicate key never reaches the snapshot closure but still counts.
    pub(crate) fn deny_first_saves(&mut self, n: usize) {
        self.deny_first_saves = n;
    }

    /// Fails the next snapshot with `error`, so the save fails like a real snapshot failure.
    /// The failure waits for a snapshot call: a save that the store refuses or serves as a use
    /// without a snapshot leaves it armed.
    pub(crate) fn fail_snapshots(&mut self, error: EngError) {
        self.fail_snapshot = Some(error);
    }
}

impl ExecutorOps for FakeExecutor {
    type Session = FakeSession;

    fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// One independent session per batch slot: the scripted steps live in the fake's `script`
    /// queue, and a default `FakeSession` starts each new conversation empty - exactly what the
    /// batch worker needs to admit more slots.
    fn new_session(&mut self) -> Result<FakeSession> {
        Ok(FakeSession::default())
    }

    /// Row-by-row window advance, with every error surfaced as an OUTER error: the window contract
    /// permits an inner, row-local error only BEFORE that row feeds, and this fake cannot classify
    /// where a row's decode failed - an inner wrap would claim a clean refusal for a row that had
    /// already advanced. Outer is always contract-legal: it invalidates every affected session
    /// instead of lying about one.
    fn decode_window(
        &mut self,
        sessions: &mut [&mut FakeSession],
        generations: &mut [&mut GenerationState],
    ) -> std::result::Result<Vec<std::result::Result<DecodeBatch, EngError>>, EngError> {
        self.window_calls.fetch_add(1, Ordering::SeqCst);
        let mut batches = Vec::with_capacity(sessions.len());
        for (st, generation) in sessions.iter_mut().zip(generations) {
            batches.push(self.decode(st, generation)?);
        }
        Ok(batches.into_iter().map(Ok).collect())
    }

    fn history<'s>(&self, st: &'s FakeSession) -> &'s [u32] {
        &st.history
    }

    fn reset(&mut self, st: &mut FakeSession) -> Result<()> {
        st.core.reset(&mut st.step)?;
        st.history.clear();
        st.permits.clear();
        st.finished = false;
        // The restore facts of the report do not survive a job: the worker resets every job and
        // skips the restore when the job has no boundary, so the report of such a job must not
        // carry the hit of the job before it.
        self.prefix_hit = PrefixHit::Miss;
        self.prefix_uncached = 0;
        self.prefix_shadow = None;
        Ok(())
    }

    fn prefill(&mut self, st: &mut FakeSession, toks: &[u32]) -> Result<()> {
        // The fake models no context limit, so every prompt fits. The growth charge runs before any
        // session change, like the resident executor's pre-mutation rule.
        let bytes = u64::try_from(toks.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(self.token_bytes);
        let permit = self.charge(bytes)?;
        if let Some(permit) = permit {
            st.permits.push(permit);
        }
        st.core.prefill(&mut st.step, toks, usize::MAX)?;
        st.history.extend_from_slice(toks);
        Ok(())
    }

    fn decode(
        &mut self,
        st: &mut FakeSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        if st.finished {
            return Err(EngError::BadRequest("decode after finish".to_string()));
        }
        if st.history.is_empty() {
            return Err(EngError::BadRequest("decode before prefill".to_string()));
        }
        // The growth charge for the batch runs before any history change, like the resident
        // executor's pre-mutation rule.
        let permit = self.charge(self.token_bytes)?;
        if let Some(permit) = permit {
            st.permits.push(permit);
        }
        let mut tokens = Vec::new();
        let mut text = String::new();
        let mut finish = None;
        if generation.cancel.load(Ordering::SeqCst) {
            finish = Some(DecodeFinish::Cancelled);
        } else if self.remaining == 0 || generation.remaining_tokens() == Some(0) {
            finish = Some(DecodeFinish::Length);
        } else {
            let ids = self
                .script
                .pop_front()
                .ok_or_else(|| EngError::BadRequest("script exhausted".to_string()))?;
            for id in ids {
                tokens.push(id);
                st.history.push(id);
                if self.eos.contains(&id) {
                    // The EOS id joins the consumed prefix and adds no text.
                    finish = Some(DecodeFinish::Stop);
                    break;
                }
                match generation.decoder.push(id) {
                    Ok(piece) => text.push_str(&piece),
                    Err(error) => {
                        // An error resets the session.
                        let _ = self.reset(st);
                        return Err(error);
                    }
                }
                self.remaining = self.remaining.saturating_sub(1);
                generation.record_generated(1);
                if self.remaining == 0 || generation.remaining_tokens() == Some(0) {
                    finish = Some(DecodeFinish::Length);
                    break;
                }
            }
        }
        if finish.is_some() {
            match generation.decoder.finish() {
                Ok(tail) => text.push_str(&tail),
                Err(error) => {
                    let _ = self.reset(st);
                    return Err(error);
                }
            }
            if finish == Some(DecodeFinish::Cancelled) {
                // A cancelled job resets the session.
                let _ = self.reset(st);
            } else {
                st.finished = true;
            }
        }
        Ok(DecodeBatch {
            tokens,
            text,
            finish,
        })
    }

    fn boundary(&mut self, _st: &mut FakeSession) {}

    fn prefix_limits(&self) -> StoreLimits {
        self.limits
    }

    /// The test knob's spacing; the default reads the environment.
    fn prefix_block_spacing(&self) -> Option<usize> {
        self.block_spacing
    }

    /// The report of the fake's real store: the entries it holds, charged at a flat 1,000 bytes per
    /// entry (the fake's saved state is `()`, so the charge is synthetic, but the plumbing that
    /// sums and logs it is real), beside the kind of the entry the last restore took and the shared
    /// tokens of that restore.
    fn prefix_report(&self) -> Option<PrefixReport> {
        let store = lock_store(&self.prefix);
        let entries = u32::try_from(store.roots_len() + store.entries_len()).unwrap_or(u32::MAX);
        Some(PrefixReport {
            entries,
            bytes: u64::from(entries) * 1_000,
            hit: self.prefix_hit,
            uncached_shared: self.prefix_uncached,
            shadow: self.prefix_shadow,
            restore_drops: store.restore_drop_count(),
            evictions: store.eviction_count(),
            save_side: store.save_side_totals(),
        })
    }

    /// Evicts one LRU entry of the fake's real store per the incremental denial order and reports
    /// the bytes its permits released. Without an installed budget the eviction still runs; it
    /// frees nothing to report. `None` when the store holds no LRU entry.
    fn evict_one_prefix_entry(&mut self, protect: Option<u64>) -> Option<u64> {
        let before = self
            .budget
            .as_ref()
            .map(|budget| budget.used())
            .unwrap_or(0);
        if !lock_store(&self.prefix).evict_lru_one(protect) {
            return None;
        }
        let after = self
            .budget
            .as_ref()
            .map(|budget| budget.used())
            .unwrap_or(0);
        Some(before.saturating_sub(after))
    }

    /// The denying fake session's restored anchor, from its real core session: the eviction
    /// protects the conversation the denial belongs to.
    fn restored_anchor_of(&self, st: &FakeSession) -> Option<u64> {
        st.core.restored_anchor()
    }

    /// The fake's own revision is its path, so the candidate is known without a session.
    fn restore_candidate(&self, prompt: &[u32]) -> Option<u64> {
        lock_store(&self.prefix).restore_candidate(prompt, &self.revision)
    }

    /// The fake's pool-derived ceiling: the free budget plus the session's own permits - which its
    /// next reset releases - over the per-token charge, minus the one token the first sampled step
    /// needs. Without a budget there is no ceiling to report.
    fn pool_prompt_ceiling(&self, st: &FakeSession) -> Option<u64> {
        let budget = self.budget.as_ref()?;
        let session = st
            .permits
            .iter()
            .map(|permit| permit.retained_bytes())
            .fold(0u64, u64::saturating_add);
        Some((budget.free() + session).saturating_sub(self.token_bytes) / self.token_bytes.max(1))
    }

    fn save_prefix(&mut self, st: &FakeSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        // The committed tokens include every scripted token, so a save after a decode is
        // refused before the session rules run.
        if st.history != key {
            return Err(EngError::BadRequest(
                "prefix key differs from the committed tokens".to_string(),
            ));
        }
        if self.deny_save_charge {
            return Err(EngError::MemoryPlan(
                "the fake denies the pool charge for the snapshot".to_string(),
            ));
        }
        let fed = st.step.fed();
        let path = self.revision;
        let deny = self.deny_saves;
        let first_denied = self.deny_first_saves;
        let call = self.save_calls;
        self.save_calls += 1;
        let failing = &mut self.fail_snapshot;
        let budget = &mut self.budget;
        let entry_bytes = self.entry_bytes;
        let mut store = lock_store(&self.prefix);
        st.core.save_prefix(&mut *store, key, path, fed, kind, || {
            if let Some(error) = failing.take() {
                return Err(error);
            }
            if deny || call < first_denied {
                return Err(EngError::MemoryPlan(
                    "the fake denies the prefix charge".to_string(),
                ));
            }
            let charge = match budget {
                Some(budget) => Some(budget.acquire(entry_bytes)?),
                None => None,
            };
            Ok(FakeSnapshot { charge })
        })
    }

    fn restore_prefix(&mut self, st: &mut FakeSession, prompt: &[u32]) -> Result<usize> {
        let path = self.revision;
        // The deny-once knob: deny the adoption ONCE - the executor-side permit the restore path
        // charges - so the store's evict-one-victim-and-retry-same-candidate step runs exactly as
        // in the resident executor, whose restore charges the pool for the whole restored extent.
        // The flag rides an atomic the closure swaps only when a candidate is actually tried: a
        // candidate-less restore (a first turn) must NOT consume the denial. Copy (not take): a
        // candidate-less restore - a first turn - leaves the knob armed; only a denial the closure
        // actually fires consumes it, written back below.
        let deny_once =
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(self.deny_restore_once));
        let closure_deny = Arc::clone(&deny_once);
        let mut store = lock_store(&self.prefix);
        let covered = st
            .core
            .restore_prefix(&mut *store, prompt, path, move |_state| {
                if closure_deny.swap(false, Ordering::SeqCst) {
                    return Err(EngError::MemoryPlan("adoption denied".to_string()));
                }
                Ok(())
            })?;
        self.deny_restore_once = deny_once.load(Ordering::SeqCst);
        // The restore facts of the report, read while the store is locked.
        self.prefix_hit = store.restored_kind().into();
        self.prefix_uncached = store.uncached_shared(prompt, covered);
        self.prefix_shadow = store.shadow_report();
        drop(store);
        if covered > 0 {
            // The restored state fed every key token but the pending one, and the scripted
            // tokens of an earlier decode never survive a restore.
            st.step.fed = covered - 1;
            st.history = st.core.history().to_vec();
            st.finished = false;
        }
        Ok(covered)
    }
}

#[cfg(test)]
mod tests {
    use super::FakeExecutor;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0},"merges":[]}}"#;

    /// The EOS id of these tests. The vocabulary holds only token 0, the text "a".
    const EOS: u32 = 7;

    fn state() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    /// A fake that holds `script` and a session that has prefilled one prompt token.
    fn prefilled(
        script: Vec<Vec<u32>>,
        cap: u32,
    ) -> (FakeExecutor, <FakeExecutor as ExecutorOps>::Session) {
        let mut executor = FakeExecutor::new(script, &[EOS], cap);
        let mut session = Default::default();
        executor.prefill(&mut session, &[1]).expect("prefill");
        (executor, session)
    }

    fn decode(
        executor: &mut FakeExecutor,
        session: &mut <FakeExecutor as ExecutorOps>::Session,
        generation: &mut GenerationState,
    ) -> DecodeBatch {
        executor.decode(session, generation).expect("decode")
    }

    #[test]
    fn a_job_limit_below_the_executor_cap_ends_with_length() {
        let (mut executor, mut session) = prefilled(vec![vec![0], vec![0], vec![0]], 100);
        let mut generation = state().with_max_tokens(2);
        let first = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(first.tokens, [0], "first tokens");
        assert_eq!(first.finish, None, "first finish");
        let second = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(second.tokens, [0], "second tokens");
        assert_eq!(second.text, "a", "second text");
        assert_eq!(second.finish, Some(DecodeFinish::Length), "second finish");
        assert_eq!(generation.remaining_tokens(), Some(0), "limit used up");
    }

    #[test]
    fn a_job_without_a_limit_follows_the_executor_cap() {
        let (mut executor, mut session) = prefilled(vec![vec![0], vec![0], vec![0]], 2);
        let mut generation = state();
        let first = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(first.finish, None, "first finish");
        let second = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(second.tokens, [0], "second tokens");
        assert_eq!(second.finish, Some(DecodeFinish::Length), "second finish");
        assert_eq!(generation.remaining_tokens(), None, "still no job limit");
    }

    #[test]
    fn a_job_limit_of_zero_ends_before_any_token() {
        let (mut executor, mut session) = prefilled(vec![vec![0]], 100);
        let mut generation = state().with_max_tokens(0);
        let batch = decode(&mut executor, &mut session, &mut generation);
        assert!(batch.tokens.is_empty(), "no token: {:?}", batch.tokens);
        assert_eq!(batch.text, "", "text");
        assert_eq!(batch.finish, Some(DecodeFinish::Length), "finish");
        assert_eq!(
            executor.history(&session),
            &[1],
            "history keeps the prompt only"
        );
    }

    #[test]
    fn an_eos_token_does_not_count_toward_the_job_limit() {
        let (mut executor, mut session) = prefilled(vec![vec![0], vec![EOS]], 100);
        let mut generation = state().with_max_tokens(3);
        let first = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(first.finish, None, "first finish");
        assert_eq!(generation.remaining_tokens(), Some(2), "after one token");
        let second = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(second.tokens, [EOS], "EOS joins the batch");
        assert_eq!(second.finish, Some(DecodeFinish::Stop), "finish");
        assert_eq!(generation.remaining_tokens(), Some(2), "EOS is not counted");
    }

    #[test]
    fn a_decode_window_failure_is_outer_not_row_local() {
        // The window contract permits an inner, row-local error only BEFORE that row feeds. This
        // fake cannot classify where a row's decode failed, so `decode_window` must surface every
        // failure as an OUTER error - an inner wrap would claim a clean row refusal for a row that
        // may already have advanced (for example a mid-row vocab failure after consuming token 0).
        // Two rows, the second un-prefilled: the result is one outer `Err`, never
        // `Ok([Ok(..), Err(..)])`.
        use crate::family::executor::ExecutorOps;
        let (mut executor, mut first) = prefilled(vec![vec![0]], 100);
        let mut second = <FakeExecutor as ExecutorOps>::Session::default();
        let (mut g1, mut g2) = (state().with_max_tokens(2), state().with_max_tokens(2));
        let window =
            executor.decode_window(&mut [&mut first, &mut second], &mut [&mut g1, &mut g2]);
        let error = window.expect_err("the un-prefilled row's refusal is OUTER");
        assert!(
            matches!(error, crate::EngError::BadRequest(ref message) if message.contains("decode before prefill")),
            "the outer error carries the failing row's refusal: {error:?}"
        );
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::{FakeExecutor, FakeSession};
    use crate::EngError;
    use crate::family::executor::ExecutorOps;
    use crate::family::session::{SaveKind, ShadowReason};
    use crate::sched::job::{PrefixHit, PrefixReport};

    /// A fake with an empty script, and a session that holds `tokens` as its prefilled prompt.
    fn fake_with_prefilled(tokens: &[u32]) -> (FakeExecutor, FakeSession) {
        let mut executor = FakeExecutor::new(Vec::new(), &[9], 10);
        let mut session = FakeSession::default();
        executor.prefill(&mut session, tokens).expect("prefill");
        (executor, session)
    }

    #[test]
    fn a_saved_prefix_restores_into_a_reset_session_and_reports_its_length() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3, 4]);
        assert!(
            executor
                .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Tip)
                .expect("save")
        );
        let mut fresh = FakeSession::default();
        let covered = executor
            .restore_prefix(&mut fresh, &[1, 2, 3, 4, 5, 6])
            .expect("restore");
        assert_eq!(covered, 4, "covered tokens");
        assert_eq!(executor.history(&fresh), &[1, 2, 3, 4], "restored history");
    }

    #[test]
    fn a_saved_prefix_restores_only_for_a_longer_prompt_that_starts_with_the_key() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3, 4]);
        executor
            .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Tip)
            .expect("save");
        let cases: [(&str, &[u32]); 3] = [
            ("equal prompt", &[1, 2, 3, 4]),
            ("shorter prompt", &[1, 2, 3]),
            ("divergent prompt", &[1, 2, 9, 4, 5]),
        ];
        for (name, prompt) in cases {
            let mut fresh = FakeSession::default();
            let covered = executor.restore_prefix(&mut fresh, prompt).expect(name);
            assert_eq!(covered, 0, "{name}: covered tokens");
            assert!(
                executor.history(&fresh).is_empty(),
                "{name}: session changed"
            );
        }
    }

    #[test]
    fn a_changed_numeric_revision_hides_the_saved_prefix() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2]);
        executor
            .save_prefix(&session, &[1, 2], SaveKind::Tip)
            .expect("save");
        executor.set_numeric_revision(2);
        let mut fresh = FakeSession::default();
        let covered = executor
            .restore_prefix(&mut fresh, &[1, 2, 3])
            .expect("restore");
        assert_eq!(covered, 0, "covered tokens under another revision");
    }

    #[test]
    fn a_new_save_replaces_the_old_one_and_a_denied_save_leaves_none() {
        let (mut executor, first) = fake_with_prefilled(&[1, 2]);
        executor
            .save_prefix(&first, &[1, 2], SaveKind::Tip)
            .expect("save first");
        let mut second = FakeSession::default();
        executor.prefill(&mut second, &[3, 4]).expect("prefill");
        executor
            .save_prefix(&second, &[3, 4], SaveKind::Tip)
            .expect("save second");
        let mut fresh = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut fresh, &[1, 2, 5])
                .expect("restore"),
            0,
            "the first key survived the second save"
        );
        assert_eq!(
            executor
                .restore_prefix(&mut fresh, &[3, 4, 5])
                .expect("restore"),
            2,
            "the second key is missing"
        );
        // A job start restores before it saves. This restore matches nothing, so the entry
        // [3, 4] that the last restore protected is a plain entry again.
        let mut third = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut third, &[5, 6, 7])
                .expect("restore"),
            0,
            "nothing stored matches the third prompt"
        );
        executor.prefill(&mut third, &[5, 6]).expect("prefill");
        executor.deny_saves();
        // The default store holds one entry, so the save drops [3, 4] to make room. The
        // snapshot is then denied and no victim is left: the save is skipped.
        let saved = executor
            .save_prefix(&third, &[5, 6], SaveKind::Tip)
            .expect("denied save");
        assert!(!saved, "a denied save reports false");
        assert!(
            executor.stored_prefixes().is_empty(),
            "the store rules dropped the old entry and the denied save added none"
        );
        let mut other = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut other, &[3, 4, 5])
                .expect("restore"),
            0,
            "a denied save left an entry that restores"
        );
    }

    #[test]
    fn a_denied_save_does_not_inherit_another_sessions_protected_entry() {
        // Protection belongs to the restoring session, not another session's later save.
        let (mut executor, first) = fake_with_prefilled(&[1, 2]);
        executor
            .save_prefix(&first, &[1, 2], SaveKind::Tip)
            .expect("save first");
        let mut restored = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut restored, &[1, 2, 5])
                .expect("restore"),
            2,
            "the first key restores"
        );
        let mut unrelated = FakeSession::default();
        executor.prefill(&mut unrelated, &[7, 8]).expect("prefill");
        executor.deny_saves();
        let saved = executor
            .save_prefix(&unrelated, &[7, 8], SaveKind::Tip)
            .expect("denied save");
        assert!(!saved, "a denied save reports false");
        assert!(
            executor.stored_prefixes().is_empty(),
            "the unrelated session has no ancestor to protect"
        );
        assert_eq!(executor.history(&restored), [1, 2]);
    }

    #[test]
    fn an_armed_snapshot_failure_waits_for_a_snapshot_call() {
        // The second save repeats a stored key, so the store serves it as a use and never calls
        // the snapshot. The armed failure must survive it and fail the next real snapshot.
        let (mut executor, first) = fake_with_prefilled(&[1, 2]);
        executor
            .save_prefix(&first, &[1, 2], SaveKind::Tip)
            .expect("save first");
        executor.fail_snapshots(EngError::Checkpoint("snapshot boom".to_string()));
        assert!(
            executor
                .save_prefix(&first, &[1, 2], SaveKind::Tip)
                .expect("a repeated key is a use"),
            "the repeat counts as done"
        );
        let mut second = FakeSession::default();
        executor.prefill(&mut second, &[3, 4]).expect("prefill");
        let failed = executor.save_prefix(&second, &[3, 4], SaveKind::Tip);
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(_))),
            "got {failed:?}"
        );
    }

    #[test]
    fn a_save_rejects_a_key_that_differs_from_the_committed_tokens() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3]);
        let rejected = executor.save_prefix(&session, &[1, 2], SaveKind::Tip);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "got {rejected:?}"
        );
    }

    /// The report of the fake: the entries of its real store, each charged a flat 1,000 bytes, and
    /// the kind of the entry the last restore took. A prompt that diverges from the stored tip
    /// restores the root; the tip still shares its first four tokens with the prompt, which equals
    /// the restored root, so nothing is uncached.
    ///
    /// Mutation check (the hit-always-Miss patch): the hit rows fail; zeroing the bytes fails
    /// the bytes rows.
    #[test]
    fn the_fake_reports_the_store_entries_and_the_restored_kind() {
        let (mut executor, mut session) = fake_with_prefilled(&[1, 2, 3, 4]);
        executor.with_prefix_limits(2, 6);
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
                save_side: crate::family::session::SaveSideTotals::default()
            }),
            "a fresh store reports all zero"
        );
        assert!(
            executor
                .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Root)
                .expect("save the root")
        );
        executor.prefill(&mut session, &[5, 6]).expect("prefill");
        assert!(
            executor
                .save_prefix(&session, &[1, 2, 3, 4, 5, 6], SaveKind::Tip)
                .expect("save the tip")
        );
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 2,
                bytes: 2_000,
                hit: PrefixHit::Miss,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: crate::family::session::SaveSideTotals::default()
            }),
            "the root and the tip, 1,000 charged bytes each, no restore yet"
        );
        let mut second = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut second, &[1, 2, 3, 4, 9, 9])
                .expect("restore"),
            4,
            "the root covers the divergent conversation"
        );
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 2,
                bytes: 2_000,
                hit: PrefixHit::Root,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: crate::family::session::SaveSideTotals::default()
            }),
            "the root hit; the tip shares exactly the restored four tokens"
        );
        // A reset clears the restore facts: the report of the next job starts from a miss even
        // though the job before it restored a root.
        executor.reset(&mut second).expect("reset");
        assert_eq!(
            executor.prefix_report(),
            Some(PrefixReport {
                entries: 2,
                bytes: 2_000,
                hit: PrefixHit::Miss,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: crate::family::session::SaveSideTotals::default()
            }),
            "the reset cleared the hit of the restore"
        );
    }

    #[test]
    fn the_fake_saves_a_root_and_a_tip_through_the_real_store() {
        // The limits hold two roots and six entries. A Root save of the key of a stored tip
        // promotes that entry to a pinned root, so the store ends with one pinned root and no
        // entry.
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3, 4]);
        executor.with_prefix_limits(2, 6);
        executor
            .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Tip)
            .expect("save tip");
        assert_eq!(
            executor.stored_prefixes(),
            vec![(4, false)],
            "the tip alone"
        );
        assert!(
            executor
                .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Root)
                .expect("save root")
        );
        assert_eq!(
            executor.stored_prefixes(),
            vec![(4, true)],
            "the root replaced the tip of the same key"
        );
    }

    #[test]
    fn a_failing_snapshot_fails_the_save_and_leaves_the_store_unchanged() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3]);
        executor.fail_snapshots(EngError::Checkpoint("snapshot boom".to_string()));
        let failed = executor.save_prefix(&session, &[1, 2, 3], SaveKind::Tip);
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(_))),
            "got {failed:?}"
        );
        assert!(executor.stored_prefixes().is_empty(), "the store changed");
    }

    /// The shadow knob: with the knob on, a restore stashes the shadow and the report carries
    /// it; the restore itself is unchanged. The knob, like `with_prefix_limits`, resets the
    /// store, so it runs before the conversation.
    #[test]
    fn the_shadow_knob_carries_the_shadow_into_the_report() {
        let (mut executor, mut session) = fake_with_prefilled(&[1]);
        executor.with_shadow_block_lcp();
        executor
            .save_prefix(&session, &[1], SaveKind::Tip)
            .expect("save");
        executor.reset(&mut session).expect("reset");
        let covered = executor
            .restore_prefix(&mut session, &[1, 0])
            .expect("restore");
        assert_eq!(covered, 1, "the restore is unchanged by the shadow");
        let shadow = executor
            .prefix_report()
            .expect("the fake reports a prefix")
            .shadow
            .expect("the shadow ran");
        assert_eq!(shadow.usable_checkpoint_tokens, 1);
        assert_eq!(shadow.divergence_offset, 1);
        assert_eq!(shadow.reason, ShadowReason::SameAsActual);
    }

    /// The default is off: the report carries no shadow until the knob runs.
    #[test]
    fn the_report_has_no_shadow_by_default() {
        let (mut executor, mut session) = fake_with_prefilled(&[1]);
        executor
            .save_prefix(&session, &[1], SaveKind::Tip)
            .expect("save");
        executor.reset(&mut session).expect("reset");
        assert_eq!(
            executor
                .restore_prefix(&mut session, &[1, 0])
                .expect("restore"),
            1
        );
        assert_eq!(
            executor
                .prefix_report()
                .expect("the fake reports a prefix")
                .shadow,
            None,
            "off by default"
        );
    }
}
