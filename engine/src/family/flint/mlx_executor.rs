//! Worker adapter. Shared DecodeSession owns EOS, sampling, cancellation and token publication.

use super::{
    factory::context_limit, mlx_model::Model, mlx_state::PrefixSnapshot, mlx_state::State,
};
use crate::sched::job::{PrefixHit, PrefixReport};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
        },
    },
    tokenizer::DecodeBatch,
};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, EngError>;

/// The store shape this executor pins: zero roots, one tip entry (the worker
/// plans exactly one tip save per job). A named constant so the store-shape
/// test pins what the EXECUTOR builds, not the default.
pub(super) const PREFIX_STORE_LIMITS: StoreLimits = StoreLimits {
    roots: 0,
    entries: 1,
};

pub(super) struct FlintExecutor {
    // Declared first so the STORE drops before the model and its kernels
    // (the Flash/Q36 field order).
    prefix: PrefixStore<PrefixSnapshot, FlintPath>,
    model: Model,
    eos: Vec<u32>,
    pub(super) vocab: usize,
    root: PathBuf,
    /// The restore fact of the CURRENT job for `prefix_report` (cleared on
    /// reset so a job without a restore never reports the job before it).
    prefix_hit: PrefixHit,
    /// The last restore's uncached-shared observation: set by
    /// `restore_prefix`, cleared with `prefix_hit` in `reset`.
    prefix_uncached_shared: u32,
    /// Lifetime count of saves refused by the retention budget; rides
    /// `save_refusals` in the report.
    prefix_budget_refusals: u64,
    /// The retention cap this executor admits saves against: the explicit
    /// env value, or the derived one. Always finite - `open` refuses to
    /// start when neither can be established.
    pub(super) prefix_retention_cap: u64,
}

pub(super) struct FlintSession {
    state: State,
    core: DecodeSession,
}

impl FlintSession {
    /// The boundary token a fed prompt left pending. Visible to the prefix
    /// tests because `core` is private to this module.
    #[cfg(test)]
    pub(super) fn pending_token(&self) -> Option<u32> {
        self.core.pending()
    }

    /// Takes a plane snapshot WITHOUT the prefix store: the save-then-
    /// continue variant gate drops it immediately, isolating the refcount
    /// effect of a stored snapshot from the capture itself.
    #[cfg(test)]
    pub(super) fn snapshot_prefix_now(&self) -> super::mlx_state::PrefixSnapshot {
        self.state.snapshot_prefix()
    }
}

pub(super) struct Step<'a> {
    model: &'a mut Model,
    state: &'a mut State,
}

impl TokenStepper for Step<'_> {
    const OPERATION: &'static str = "flint";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.model.step(self.state, token, false).map(|_| ())
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        self.model
            .prefill_rows(self.state, tokens, false)
            .map(|_| ())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.model
            .step(self.state, token, true)?
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".into(),
            })
    }

    fn fed(&self) -> usize {
        self.state.pos as usize
    }

    fn reset(&mut self) -> Result<()> {
        self.state.reset();
        Ok(())
    }
}

impl FlintExecutor {
    /// Tests only: a stepper over this executor's model and one session's
    /// state, for identity gates that must feed rows and read logits back.
    #[cfg(test)]
    pub(super) fn stepper<'a>(&'a mut self, st: &'a mut FlintSession) -> Step<'a> {
        Step {
            model: &mut self.model,
            state: &mut st.state,
        }
    }

    pub(super) fn open(root: &Path, eos: Vec<u32>, vocab: usize) -> Result<(Self, FlintSession)> {
        if vocab == 0
            || vocab > 150_144
            || eos.is_empty()
            || eos.iter().any(|&id| id as usize >= vocab)
        {
            return Err(EngError::Checkpoint(
                "Flint tokenizer vocabulary or EOS set is invalid".into(),
            ));
        }
        // Retention cap: computed ONLY when the reuse opt-in is armed - a
        // default-OFF engine must load unchanged (no sysctl probe, no env
        // validation, no new failure modes) - and before the model load so
        // a cap failure costs nothing. Opt-in OFF admits a cap of 0: every
        // save refused. The probes are CLOSURES so the opt-in gate is not
        // merely branched around but never executed when OFF.
        let retention_cap = super::retention_cap_for(
            super::prefix_reuse_opt_in(),
            std::env::var_os("FIFI_FLINT_PREFIX_BUDGET_BYTES").as_deref(),
            physical_memory_bytes,
            || weights_charge(root),
        )
        .map_err(EngError::Checkpoint)?;
        if retention_cap > 0 {
            eprintln!("flint prefix retention cap {retention_cap} bytes");
        }
        let model = Model::open(root)?;
        Ok((
            Self {
                model,
                eos,
                vocab,
                root: root.to_path_buf(),
                // The store: one tip entry, no roots.
                prefix: PrefixStore::with_limits(PREFIX_STORE_LIMITS),
                prefix_hit: PrefixHit::Miss,
                prefix_uncached_shared: 0,
                prefix_budget_refusals: 0,
                prefix_retention_cap: retention_cap,
            },
            FlintSession {
                state: State::new(),
                core: DecodeSession::default(),
            },
        ))
    }
}

impl ExecutorOps for FlintExecutor {
    type Session = FlintSession;

    fn check_prompt(&self, tokens: usize) -> Result<()> {
        check_prompt(tokens, context_limit() as usize)
    }

    fn history<'s>(&self, st: &'s FlintSession) -> &'s [u32] {
        st.core.history()
    }

    fn reset(&mut self, st: &mut FlintSession) -> Result<()> {
        // The job-scoped restore fact dies with the job, so a job without
        // a restore never reports the job before it.
        self.prefix_hit = PrefixHit::Miss;
        self.prefix_uncached_shared = 0;
        st.core.reset(&mut Step {
            model: &mut self.model,
            state: &mut st.state,
        })
    }

    fn prefill(&mut self, st: &mut FlintSession, tokens: &[u32]) -> Result<()> {
        st.core.prefill(
            &mut Step {
                model: &mut self.model,
                state: &mut st.state,
            },
            tokens,
            context_limit() as usize,
        )
    }

    fn decode(
        &mut self,
        st: &mut FlintSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.vocab,
            context_limit: context_limit() as usize,
        };
        st.core.decode(
            &mut Step {
                model: &mut self.model,
                state: &mut st.state,
            },
            generation,
            &limits,
        )
    }

    fn boundary(&mut self, _st: &mut FlintSession) {}

    fn prefill_piece(&self) -> Option<usize> {
        Some(super::factory::prefill_chunk_rows(
            super::factory::os_setting(std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref()),
        ))
    }

    fn prefix_limits(&self) -> StoreLimits {
        // The store shape is zero roots, one tip entry, so the worker plans
        // exactly one tip save per job. Overridden so the worker's plan and
        // the store's limits agree by construction, not by both happening to
        // default to (0, 1).
        PREFIX_STORE_LIMITS
    }

    fn save_prefix(&mut self, st: &FlintSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        // Retention admission (enforced by default): the worst case a save
        // commits - the new entry, the transient overlap with the entry it
        // displaces, and the COW the continuing session needs on the
        // shared planes - must fit the cap BEFORE the store is touched.
        // The check runs before `core.save_prefix` so an over-cap save is
        // a planned Ok(false) at zero store work (the one accepted
        // tradeoff: a duplicate-use re-save under a tight cap counts as a
        // refusal); the release is the Rc drop when the store drops the
        // entry. The refusal is counted in save_refusals and logged once
        // per refused save (at most one per job).
        let charge = st.state.snapshot_prefix().charged_bytes();
        let retained: u64 = self
            .prefix
            .snapshots()
            .map(|snapshot| snapshot.charged_bytes())
            .sum();
        if !super::retention_admitted(charge, retained, self.prefix_retention_cap) {
            self.prefix_budget_refusals += 1;
            eprintln!(
                "flint prefix save refused: charge {charge} + retained {retained} exceeds the retention cap {}",
                self.prefix_retention_cap
            );
            return Ok(false);
        }
        let path = FlintPath::now(&self.root);
        let fed = st.state.pos as usize;
        st.core
            .save_prefix(&mut self.prefix, key, path, fed, kind, || {
                Ok(st.state.snapshot_prefix())
            })
    }

    fn restore_prefix(&mut self, st: &mut FlintSession, prompt: &[u32]) -> Result<usize> {
        let path = FlintPath::now(&self.root);
        let state = &mut st.state;
        let restored = st
            .core
            .restore_prefix(&mut self.prefix, prompt, path, |snapshot| {
                state.restore_prefix(snapshot);
                Ok(())
            });
        if let Ok(n) = restored {
            // The store holds only tips (one-tip limits), so a non-zero
            // restore is a Tip hit. The uncached-shared observation is
            // taken HERE, where the prompt is in hand, and rides the
            // report of this job only.
            self.prefix_hit = if n > 0 {
                PrefixHit::Tip
            } else {
                PrefixHit::Miss
            };
            self.prefix_uncached_shared = self.prefix.uncached_shared(prompt, n);
        }
        restored
    }

    /// The prefix-store facts of one job: entries and their charged KV bytes
    /// after the job's save points, beside the job's own restore hit and its
    /// uncached-shared observation (taken at the restore that saw the prompt,
    /// not a stale 0).
    fn prefix_report(&self) -> Option<PrefixReport> {
        let stored = self.prefix.roots_len() + self.prefix.entries_len();
        let entries = u32::try_from(stored).unwrap_or(u32::MAX);
        let mut save_side = self.prefix.save_side_totals();
        // An executor-side retention-budget refusal is also a save that
        // returned without storing, so it rides the same counter instead
        // of a new report field.
        save_side.save_refusals = save_side
            .save_refusals
            .saturating_add(self.prefix_budget_refusals);
        Some(PrefixReport {
            entries,
            bytes: self
                .prefix
                .snapshots()
                .map(|snapshot| snapshot.charged_bytes())
                .sum(),
            hit: self.prefix_hit,
            uncached_shared: self.prefix_uncached_shared,
            shadow: None,
            restore_drops: self.prefix.restore_drop_count(),
            evictions: self.prefix.eviction_count(),
            save_side,
        })
    }
}

/// Physical memory in bytes via `/usr/sbin/sysctl` -n hw.memsize` - the
/// ABSOLUTE path (a PATH lookup breaks under a restricted PATH or sandbox,
/// which would refuse engine loads; the absolute binary needs no PATH at
/// all). `None` when the probe fails, which the cap seam treats as
/// underivable (fail closed, opt-in only).
fn physical_memory_bytes() -> Option<u64> {
    std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
}

/// The loaded-weights charge: the safetensors bytes under the model root,
/// charged the way the loader actually opens them (weights.rs): a
/// non-recursive root listing, each shard OPENED so a symlinked shard
/// charges its target's true length (a no-follow walk would charge symlink
/// length and overstate headroom), every error propagated (fail closed,
/// never a swallowed undercount), sums checked. File size is the resident
/// proxy for the MLX load; no tensors touched.
fn weights_charge(root: &Path) -> std::result::Result<u64, String> {
    let entries = std::fs::read_dir(root)
        .map_err(|e| format!("weights charge: cannot read {root:?}: {e}"))?;
    let mut total = 0u64;
    for entry in entries {
        let entry = entry.map_err(|e| format!("weights charge: bad dir entry: {e}"))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            let file = std::fs::File::open(&path)
                .map_err(|e| format!("weights charge: open {}: {e}", path.display()))?;
            let len = file
                .metadata()
                .map_err(|e| format!("weights charge: stat {}: {e}", path.display()))?
                .len();
            total = total
                .checked_add(len)
                .ok_or_else(|| "weights charge: byte count overflow".to_string())?;
        }
    }
    Ok(total)
}

/// The environment flags a [`FlintPath`] binds as their FROZEN EFFECTIVE
/// values, not the raw env strings: the numerics depend only on the
/// effective flag, so "1", "true", and "on" collapse to one fingerprint, and
/// unset and garbage that resolve to the same default are the same path.
/// Every flag that can move prefill or decode numerics is bound by its
/// effective value; a future flag whose raw string matters beyond its
/// effective value must not be added here as a raw string.
/// `FIFI_FLINT_MLX_COMPILE_MOE` is deliberately NOT bound: nothing on the
/// prefill/decode path reads it. `FIFI_FLINT_MLX_ASYNC_CHAIN` and
/// `FIFI_FLINT_MLX_COALESCE_RESIDUAL` ARE bound (the chain flag gates the
/// ring-view and prefill-chain routes, so it moves numerics). Grouped so the
/// pure constructor stays at three arguments (clippy::too_many_arguments).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FlintEnv {
    pub(super) batch_prefill: usize,
    pub(super) decode_attn: bool,
    pub(super) ring_view: bool,
    pub(super) prefill_chain: bool,
    pub(super) compile_layer: bool,
    pub(super) t1_flash: bool,
    pub(super) async_chain: bool,
    pub(super) coalesce_residual: bool,
}

/// The numeric-path fingerprint a stored Flint prefix binds: the context
/// limit, the frozen EFFECTIVE env flag values above, plus the model root.
/// A restore under any other fingerprint is refused by the store's path
/// comparison, so a snapshot can never serve numerics it was not built
/// under.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FlintPath {
    limit: u32,
    env: FlintEnv,
    root: std::path::PathBuf,
}

impl FlintPath {
    pub(super) fn now(root: &Path) -> Self {
        use crate::kernels::mlx;
        Self::from_env_values(
            root,
            context_limit(),
            FlintEnv {
                // The same resolvers the serving path uses, so the
                // fingerprint can never disagree with the route.
                batch_prefill: super::factory::prefill_chunk_rows(super::factory::os_setting(
                    std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref(),
                )),
                decode_attn: mlx::decode_attn_flagged(),
                ring_view: super::mlx_forward::ring_view_flagged(),
                prefill_chain: super::mlx_forward::prefill_chain_flagged(),
                compile_layer: mlx::compile_layer_flagged(),
                t1_flash: mlx::t1_flash_flagged(),
                async_chain: super::mlx_forward::async_chain_flagged(),
                coalesce_residual: super::mlx_forward::coalesce_residual_flagged(),
            },
        )
    }

    /// The pure half of [`FlintPath::now`] - every fingerprint input is an
    /// explicit value, so the fingerprint contract is testable without
    /// mutating the process environment (real-env coverage stays with the
    /// per-arm test-bin processes of the qualification window).
    pub(super) fn from_env_values(root: &Path, limit: u32, env: FlintEnv) -> Self {
        Self {
            limit,
            env,
            root: root.to_path_buf(),
        }
    }
}
