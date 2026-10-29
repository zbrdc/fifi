//! Single-session CUDA adapter. The shared session owns generation and publication.

use super::{
    cuda_ffn::device::runtime::{Decoder, State, VOCAB},
    cuda_prefix::{CudaPrefixEnv, CudaPrefixPath, CudaPrefixSnapshot},
    cuda_sample::SamplingScratch,
    exl3_resident::{FlintExl3Resident, FlintResidentLimits},
    settings::{FlintOptions, late_health_selected},
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        project::cuda::{Reserve, served_admission},
        sampler::Sampler,
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
        },
    },
    kernels::cuda::{CudaContext, DeviceStream, plan},
    sched::job::{PrefixHit, PrefixReport},
    tokenizer::DecodeBatch,
};
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

const ALLOC_CACHE_VAR: &str = "FIFI_FLINT_CUDA_ALLOC_CACHE";

fn alloc_cache_selected() -> bool {
    crate::tokenizer::env_allows_default_on(ALLOC_CACHE_VAR)
}

/// The frozen effective selectors the prefix fingerprint binds: one read per
/// process, shared by every save and restore,
/// so the fingerprint cannot drift when the environment changes mid-run
/// and the three previously-unbound selectors (packed-pointwise, midrows
/// legacy, expert cache) are part of the numeric-path identity.
fn frozen_prefix_env() -> CudaPrefixEnv {
    static ENV: std::sync::OnceLock<CudaPrefixEnv> = std::sync::OnceLock::new();
    ENV.get_or_init(|| CudaPrefixEnv {
        alloc_cache: alloc_cache_selected(),
        late_health: late_health_selected(std::env::var_os("FIFI_FLINT_LATE_HEALTH").as_deref()),
        packed_pointwise: crate::family::flint::settings::packed_pointwise_selected(
            std::env::var_os("FIFI_FLINT_CUDA_PACKED_POINTWISE").as_deref(),
        ),
        exl3_legacy: std::env::var_os("FIFI_EXL3_LEGACY").as_deref()
            == Some(std::ffi::OsStr::new("1")),
        expert_cache: crate::tokenizer::env_allows_value(Some(
            &std::env::var("FIFI_FLINT_CUDA_EXPERT_CACHE").unwrap_or_default(),
        )),
    })
    .clone()
}

#[cfg(test)]
#[path = "cuda_executor_route_trace_tests.rs"]
mod route_trace_tests;
#[cfg(test)]
#[path = "cuda_executor_tests.rs"]
mod tests;

pub(super) struct FlintExecutor {
    // Declared first so the host-plane snapshot store drops BEFORE the
    // device/kernel state (Flash and Q36 declare it first).
    prefix: PrefixStore<CudaPrefixSnapshot, CudaPrefixPath>,
    /// The restore fact of the CURRENT job for `prefix_report` (mirror
    /// of the MLX rule): cleared on reset so a job without a
    /// restore never reports the job before it.
    prefix_hit: PrefixHit,
    /// The last restore's uncached-shared observation, taken at the
    /// restore that saw the prompt (no false zeros).
    prefix_uncached_shared: u32,
    /// Lifetime count of saves the EXECUTOR declined before the store
    /// (the zero-committed boundary: a 1-token key): rides
    /// `save_refusals` in the report next to the store's own budget
    /// refusals — the two counters never overlap.
    prefix_zero_committed_saves: u64,
    ctx: CudaContext,
    stream: DeviceStream,
    model: FlintExl3Resident,
    sampling: SamplingScratch,
    eos: Vec<u32>,
    vocab: usize,
    context_limit: usize,
    prefill_chunk_size: usize,
    root: std::path::PathBuf,
    options: FlintOptions,
}

pub(super) struct FlintSession {
    state: State,
    core: DecodeSession,
}

struct Step<'a> {
    executor: &'a FlintExecutor,
    state: &'a mut State,
}

impl TokenStepper for Step<'_> {
    const OPERATION: &'static str = "flint_cuda";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.executor
            .decoder()?
            .step(self.state, token, |_, _| Ok(()))
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let rows = self.executor.prefill_chunk_size;
        if rows == 1 {
            for &token in tokens {
                self.feed(token)?;
            }
        } else {
            let decoder = self.executor.decoder()?;
            for chunk in tokens.chunks(rows) {
                decoder.feed_rows(self.state, chunk)?;
            }
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.executor
            .decoder()?
            .step(self.state, token, |_, logits| Ok(logits.to_vec()))
    }

    fn next_token(&mut self, token: u32, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let executor = self.executor;
        executor.decoder()?.step(self.state, token, |head, logits| {
            executor.sampling.sample(
                &executor.ctx,
                &executor.stream,
                head,
                logits,
                sampler,
                vocab,
            )
        })
    }

    fn fed(&self) -> usize {
        self.state.fed()
    }

    fn reset(&mut self) -> Result<()> {
        self.state.reset(&self.executor.stream)?;
        self.executor.ctx.drain_alloc_cache()?;
        self.executor
            .model
            .recover_expert_cache(&self.executor.ctx, &self.executor.stream)?;
        Ok(())
    }
}

fn work_reserve(plan: &super::kv_plan::KvPlan, chunk_rows: u32, cache_bytes: u64) -> Result<u64> {
    // Six routed + one shared expert retain <16 MiB per extra row at width <=32768.
    ((512 + 16 * (u64::from(chunk_rows) - 1)) << 20)
        .checked_add(State::scratch_bytes(plan)?)
        .and_then(|bytes| bytes.checked_add(cache_bytes))
        .ok_or_else(|| EngError::MemoryPlan("Flint workspace reserve overflows".into()))
}

impl FlintExecutor {
    pub(super) fn open(
        root: &Path,
        options: &FlintOptions,
        eos: Vec<u32>,
        vocab: usize,
    ) -> Result<(Self, FlintSession)> {
        if vocab == 0
            || vocab > VOCAB
            || eos.is_empty()
            || eos.iter().any(|&id| id as usize >= vocab)
        {
            return Err(EngError::Checkpoint(
                "Flint tokenizer vocabulary or EOS set is invalid".into(),
            ));
        }
        let kv = options.plan()?;
        let cache_enabled = alloc_cache_selected();
        let ctx = CudaContext::with_admission_cache(&served_admission(), cache_enabled)?;
        let stream = ctx.stream()?;
        let free = ctx.memory_info()?.free as u64;
        let cache_bytes = CudaContext::alloc_cache_reserve(cache_enabled);
        let reserve = Reserve {
            kv_cache: kv.bytes(kv.storage().row_bytes())? as u64,
            work: work_reserve(&kv, options.prefill_chunk_size, cache_bytes)?,
            headroom: 512 << 20,
        };
        let model = FlintExl3Resident::load_with_prefill(
            &ctx,
            &stream,
            root,
            FlintResidentLimits {
                max_staging_bytes: 2 << 30,
                max_weight_bytes: free,
                reserve,
            },
            options.prefill_chunk_size as usize,
        )?;
        // Weights are now live; recheck the remaining demand before session allocation.
        plan::admit(
            &plan::Demand {
                tensors: 0,
                kv_cache: reserve.kv_cache,
                work: reserve.work,
                headroom: reserve.headroom,
            },
            ctx.memory_info()?.free as u64,
        )?;
        Decoder::new(&ctx, &stream, &model)?;
        let state = State::new(&ctx, kv)?;
        let sampling = SamplingScratch::new(&ctx)?;
        eprintln!(
            "Flint CUDA V/{}-write {}: context={} kv={} kv_bytes={} work_reserve={} headroom={} weights={} alloc_cache={} late_health={}",
            options.kv_profile.to_ascii_uppercase(),
            if options.kv_profile == "q8" {
                "native"
            } else {
                "bridge"
            },
            options.context_limit,
            options.kv_profile,
            reserve.kv_cache,
            reserve.work,
            reserve.headroom,
            model.device_weight_bytes(),
            if cache_enabled { "on" } else { "off" },
            if late_health_selected(std::env::var_os("FIFI_FLINT_LATE_HEALTH").as_deref()) {
                "on"
            } else {
                "off"
            }
        );
        // Resolve the prefix opt-in against the kv profile at startup.
        // This fires the one-time load-line note when the env is set but
        // the profile refuses, instead of failing the first boundary
        // request.
        let _ = super::prefix_reuse_enabled_for(&options.kv_profile);
        Ok((
            Self {
                ctx,
                stream,
                model,
                sampling,
                eos,
                vocab,
                context_limit: options.context_limit as usize,
                prefill_chunk_size: options.prefill_chunk_size as usize,
                root: root.to_path_buf(),
                // The one-entry tip store (the MLX mirror); serving default-off is
                // the family gate in run.rs (flag + FIFI_FLINT_PREFIX_REUSE).
                prefix: PrefixStore::with_limits(StoreLimits::default()),
                // The restore facts of the CURRENT job (mirror of the MLX
                // twin: no false zeros — the report shows
                // what the last restore observed, not hardcoded 0s).
                prefix_hit: PrefixHit::Miss,
                prefix_uncached_shared: 0,
                prefix_zero_committed_saves: 0,
                options: options.clone(),
            },
            FlintSession {
                state,
                core: DecodeSession::default(),
            },
        ))
    }

    /// The store fingerprint computed from the CURRENT frozen selectors —
    /// the effective values the numeric path reads, never raw env.
    /// All five selectors freeze together, once per process, so a save and
    /// a later restore can never see different spellings of one state.
    fn prefix_path(&self) -> CudaPrefixPath {
        CudaPrefixPath::now(&self.root, &self.options, frozen_prefix_env())
    }

    /// The store arms only when the opt-in resolves against this engine's
    /// kv profile default OFF stays bit-for-bit unobservable.
    fn prefix_armed(&self) -> bool {
        super::prefix_reuse_enabled_for(&self.options.kv_profile)
    }

    fn decoder(&self) -> Result<Decoder<'_>> {
        Decoder::new(&self.ctx, &self.stream, &self.model)
    }

    /// One logits row without the sampler, for the qualification greedy
    /// walk and the bitwise boundary-row comparisons.
    #[cfg(test)]
    pub(super) fn qual_logits(&self, session: &mut FlintSession, token: u32) -> Result<Vec<f32>> {
        Step {
            executor: self,
            state: &mut session.state,
        }
        .logits(token)
    }
}

impl FlintSession {
    /// The boundary token a fed prompt left pending. Visible to the prefix
    /// tests because `core` is private to this module.
    #[cfg(test)]
    pub(super) fn pending_token(&self) -> Option<u32> {
        self.core.pending()
    }

    /// Rows the ring has fed. The wrap gate asserts this reaches the tip so
    /// a silently truncated wrap arm cannot degrade into a fresh comparison.
    #[cfg(test)]
    pub(super) fn fed(&self) -> usize {
        self.state.fed()
    }
}

impl ExecutorOps for FlintExecutor {
    type Session = FlintSession;

    fn check_prompt(&self, tokens: usize) -> Result<()> {
        check_prompt(tokens, self.context_limit)
    }

    fn history<'s>(&self, session: &'s FlintSession) -> &'s [u32] {
        session.core.history()
    }

    fn reset(&mut self, session: &mut FlintSession) -> Result<()> {
        // A job without a restore never reports the job before it:
        // the restore facts clear with the session.
        self.prefix_hit = PrefixHit::Miss;
        self.prefix_uncached_shared = 0;
        session.core.reset(&mut Step {
            executor: self,
            state: &mut session.state,
        })
    }

    fn prefill(&mut self, session: &mut FlintSession, tokens: &[u32]) -> Result<()> {
        session.core.prefill(
            &mut Step {
                executor: self,
                state: &mut session.state,
            },
            tokens,
            self.context_limit,
        )
    }

    fn decode(
        &mut self,
        session: &mut FlintSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.vocab,
            context_limit: self.context_limit,
        };
        session.core.decode(
            &mut Step {
                executor: self,
                state: &mut session.state,
            },
            generation,
            &limits,
        )
    }

    fn boundary(&mut self, _session: &mut FlintSession) {}

    fn prefill_piece(&self) -> Option<usize> {
        Some(self.prefill_chunk_size)
    }

    /// Saves the session's committed KV planes as the store's tip entry
    /// (slice P1). An unarmed engine (opt-in off, or resolved off by a
    /// non-f16 profile) reports "not saved" WITHOUT error before the
    /// profile gate, so a q8 boundary request serves cold;
    /// the refusal is a planned no-op, never a BadRequest.
    fn save_prefix(&mut self, st: &FlintSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        if !self.prefix_armed() {
            return Ok(false);
        }
        super::cuda_prefix::profile_allows_prefix(&self.options.kv_profile)?;
        // A zero-committed save: a 1-token key
        // leaves nothing committed, and the committed-boundary check
        // would fail the REQUEST. Decline to cache instead — the boundary
        // request serves cold with no error — and count the decline
        // executor-side so `save_refusals` shows it (the store never sees
        // these, so the two counters cannot double-count).
        if st.state.fed() == 0 {
            self.prefix_zero_committed_saves = self.prefix_zero_committed_saves.saturating_add(1);
            return Ok(false);
        }
        let path = self.prefix_path();
        let fed = st.state.fed();
        let ctx = &self.ctx;
        let stream = &self.stream;
        st.core
            .save_prefix(&mut self.prefix, key, path, fed, kind, || {
                st.state.snapshot_prefix(ctx, stream)
            })
    }

    /// Replaces the reset session's device state with the saved tip when the
    /// prompt extends it and the numeric path matches. Unarmed engines
    /// reuse 0 without error, same rule as the save.
    fn restore_prefix(&mut self, st: &mut FlintSession, prompt: &[u32]) -> Result<usize> {
        if !self.prefix_armed() {
            return Ok(0);
        }
        super::cuda_prefix::profile_allows_prefix(&self.options.kv_profile)?;
        let path = self.prefix_path();
        let core = &mut st.core;
        let state = &mut st.state;
        let ctx = &self.ctx;
        let stream = &self.stream;
        let restored = core.restore_prefix(&mut self.prefix, prompt, path, |snapshot| {
            state.restore_prefix(ctx, stream, snapshot)
        })?;
        // Capture the restore facts THIS job observed:
        // the hit kind rides the restored entry's kind, and the uncached
        // shared prefix is measured against the store as it was for this
        // prompt — not a stale 0 in the report.
        self.prefix_hit = if restored > 0 {
            match self.prefix.restored_kind() {
                Some(SaveKind::Root) => PrefixHit::Root,
                Some(SaveKind::Periodic) => PrefixHit::Periodic,
                Some(SaveKind::Tip) | Some(SaveKind::Block) | None => PrefixHit::Tip,
            }
        } else {
            PrefixHit::Miss
        };
        self.prefix_uncached_shared = self.prefix.uncached_shared(prompt, restored);
        Ok(restored)
    }

    /// The store report: entries,
    /// the host bytes the snapshots pin, the kind of the last restore, and
    /// the store's lifetime counters. The shadow stays off for CUDA (the
    /// block-match flag is not lifted for this store yet).
    /// The store limits: the armed store (0 roots, 1 entry) only
    /// when armed; otherwise a closed store (0, 0) so default OFF is
    /// unobservable.
    fn prefix_limits(&self) -> StoreLimits {
        if self.prefix_armed() {
            StoreLimits::default()
        } else {
            StoreLimits {
                roots: 0,
                entries: 0,
            }
        }
    }

    fn prefix_report(&self) -> Option<PrefixReport> {
        if !self.prefix_armed() {
            return None;
        }
        let stored = self.prefix.roots_len() + self.prefix.entries_len();
        let hit = self.prefix_hit;
        let mut save_side = self.prefix.save_side_totals();
        // Executor-side declines (zero-committed saves) ride the same
        // counter next to the store's own budget refusals; the two never
        // overlap, so the sum is exact.
        save_side.save_refusals = save_side
            .save_refusals
            .saturating_add(self.prefix_zero_committed_saves);
        Some(PrefixReport {
            entries: u32::try_from(stored).unwrap_or(u32::MAX),
            bytes: self.prefix.snapshots().map(|s| s.host_bytes()).sum(),
            hit,
            uncached_shared: self.prefix_uncached_shared,
            shadow: None,
            save_side,
            restore_drops: self.prefix.restore_drop_count(),
            evictions: self.prefix.eviction_count(),
        })
    }
}

#[cfg(test)]
impl FlintExecutor {
    /// Qualification-window constructor: the FLINT_QUAL_MODEL root (or
    /// FLINT_QUAL_TOKENIZER), resolved straight on the open() path.
    pub(super) fn open_qualification(limit: u32, profile: &str) -> (Self, FlintSession) {
        let root = std::env::var_os("FLINT_QUAL_MODEL")
            .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
        let tokenizer_dir =
            std::env::var_os("FLINT_QUAL_TOKENIZER").map_or_else(|| root.clone(), Into::into);
        let options = FlintOptions {
            context_limit: limit,
            kv_profile: profile.to_owned(),
            prefill_chunk_size: 8,
        };
        let bundle =
            super::factory::load_tokenizer(&tokenizer_dir).expect("the qualification tokenizer");
        let eos = bundle.eos_token_ids().to_vec();
        let vocab = bundle.vocab_size() as usize;
        Self::open(&root, &options, eos, vocab).expect("the qualification executor")
    }

    /// Envelope-gate probe: (free, total) device bytes for the 40 GiB
    /// guard assertion and the envelope record.
    pub(super) fn qual_memory_info(&self) -> (u64, u64) {
        let info = self.ctx.memory_info().expect("device memory info");
        (info.free as u64, info.total as u64)
    }
}
