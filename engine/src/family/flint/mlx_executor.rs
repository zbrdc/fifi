//! Worker adapter. Shared DecodeSession owns EOS, sampling, cancellation and token publication.

use super::{
    factory::{PrefillPairPolicy, context_limit, prefill_pair_policy},
    mlx_forward::{PairPlan, PairStaging},
    mlx_model::{Model, expert_staging_bytes},
    mlx_state::PrefixSnapshot,
    mlx_state::State,
};
use crate::sched::job::{PrefixHit, PrefixReport};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState, PrefillPairOutcome},
        mlx_report::{MlxExecutionPlan, MlxPrefillMemory, PrefillReason},
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
        },
    },
    tokenizer::DecodeBatch,
};
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

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
    prefill_piece: usize,
    mlx_plan: MlxExecutionPlan,
    request_sized_kv: bool,
    request_prefill: bool,
    flash_prefill: bool,
    prefill_pair: PrefillPairPolicy,
    cancel_prefill_pair: bool,
    eos: Vec<u32>,
    pub(super) vocab: usize,
    root: PathBuf,
    /// The pinned chat-template digest threaded from the factory: the
    /// park-identity `template` field.
    #[allow(dead_code)] // read by the park hooks once they exist
    template_sha: [u8; 32],
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
    request_prefill_piece: Option<NonZeroUsize>,
    prefill_pair: bool,
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

impl<'a> Step<'a> {
    fn new(model: &'a mut Model, state: &'a mut State) -> Self {
        Self { model, state }
    }
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
        Step::new(&mut self.model, &mut st.state)
    }

    pub(super) fn open(
        root: &Path,
        eos: Vec<u32>,
        vocab: usize,
        template_sha: [u8; 32],
    ) -> Result<(Self, FlintSession)> {
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
        let device = crate::family::mlx_budget::DeviceBudget::detect()?;
        let reserve = device
            .map(|device| {
                let charges = device_session_charges(retention_cap)?;
                let weights = device.remaining("Flint MLX", &charges)?;
                Ok::<_, EngError>((device, charges, weights))
            })
            .transpose()?;
        let model = Model::open_for_budget(root, reserve.as_ref().map(|(_, _, bytes)| *bytes))?;
        let max_rows = super::factory::prefill_chunk_rows(super::factory::os_setting(
            std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref(),
        ));
        let (prefill_piece, memory, flash_prefill) = if let Some((device, charges, _)) = reserve {
            let (piece, memory, flash_prefill) =
                device_prefill_piece(device, &model, max_rows, &charges)?;
            (piece, Some(memory), flash_prefill)
        } else {
            (max_rows, None, false)
        };
        let (placement, placement_reason) = if model.streamed_expert_bytes != 0 {
            ("disk_experts", "full expert banks exceed the device budget")
        } else {
            ("resident", "routed expert banks are imported at startup")
        };
        let reason = if max_rows == 1 {
            PrefillReason::RowCeiling
        } else if memory.is_none() {
            PrefillReason::Configured
        } else if prefill_piece == 1 {
            PrefillReason::Workspace
        } else {
            PrefillReason::Admitted
        };
        let request_sized_kv = memory.is_some()
            && super::factory::request_sized_kv_enabled(
                std::env::var_os("FIFI_FLINT_MLX_REQUEST_KV").as_deref(),
            );
        let request_prefill = memory.is_some()
            && super::factory::request_prefill_enabled(
                std::env::var_os("FIFI_FLINT_MLX_REQUEST_PREFILL").as_deref(),
            );
        let prefill_pair = if request_prefill
            && model.streamed_expert_bytes != 0
            && super::mlx_model::expert_pipeline_flagged()
        {
            prefill_pair_policy(std::env::var_os("FIFI_FLINT_MLX_PREFILL_PAIR").as_deref())
        } else {
            PrefillPairPolicy::Off
        };
        let mlx_plan = MlxExecutionPlan::new(
            placement,
            placement_reason,
            max_rows,
            prefill_piece,
            reason,
            memory,
        );
        Ok((
            Self {
                model,
                prefill_piece,
                mlx_plan,
                request_sized_kv,
                request_prefill,
                flash_prefill,
                prefill_pair,
                cancel_prefill_pair: std::env::var_os("FIFI_FLINT_MLX_CANCEL_PREFILL_PAIR")
                    .is_some_and(|value| value == "1"),
                eos,
                vocab,
                root: root.to_path_buf(),
                template_sha,
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
                request_prefill_piece: None,
                prefill_pair: false,
            },
        ))
    }
}

/// Ladder prefill widths for the device-budget path: the SHARED
/// prefill::mlx::PREFILL_CANDIDATES ladder (2048 down to 1) capped by the
/// switch ceiling, the same list Qwen3.6 and Flash-Next size their rows
/// from, so the next rung change reaches all three with one edit. The
/// [T, H] batch forward serves ANY width (its only bound is that a chunk
/// plus its position fits the context), and the FlintPath fingerprint
/// records the value the planner picks; the 512/1024 list the env switch
/// recognizes is the QUALIFIED set for hand-rolled values, not a kernel
/// constraint, and the switch's fail-safe (unrecognized -> 512) does not
/// apply to planner results.
///
/// Flash prefill pricing, measured on one M3 Ultra host with 128 GB
/// (medians of three fresh boots per width, 2026-11-19). At a 3,601-token
/// prompt the allocator peak over the resident base grows a ~182 MB base
/// plus ~260-264 KB per row, linear from 1 to 1,024 rows. At a 28,842-token
/// prompt the same widths grow 1,224 MB (512 rows) and 1,350 MB (1,024
/// rows), but ~864 MB of that is the KV planes the prefill materializes,
/// which the session charges price (session_state_bytes), not this ladder;
/// net of the planes the flash terms grow only +44 MB (512 rows) and
/// +37 MB (1,024 rows) between the two depths (~316/449 MB shallow,
/// ~360/486 MB deep), so the charge carries no context term. Both
/// terms carry a 2x margin: the measurement is one device and the margin
/// covers kernel and driver variance across widths and machines; even at 2x
/// the flash charge stays ~24x below the composite per-row charge at 32K
/// context.
///
/// Chunked prefill is not bit-identical across row widths: greedy
/// decoding is deterministic within a width (three runs per width,
/// identical), and identical across widths for roughly the first 69
/// tokens before the chunk-boundary numerics diverge — so a user's
/// output can depend on the row width the budget selects. No
/// teacher-forced quality arm (NLL, top-1) across widths exists on
/// record yet; that comparison is a listed gap.
///
/// Infeasible case: below the base term (~384 MiB charged, ~182 MB
/// measured) no multi-row piece fits, and the ladder floors at one row —
/// a one-row piece still materializes the base, so the tightest budgets
/// overshoot their remainder by ~176 MB at the minimum rung. The base
/// belongs in the resident plan; that move is deliberately not part of
/// this commit.
/// The query dtype of the sink-bearing float-flash prefill route; priced
/// here with the head dim the checkpoint load check pins
/// (config::PREFILL_FLASH_HEAD_DIM), so the ladder buys the route the
/// forward dispatches.
pub(crate) const PREFILL_FLASH_DTYPE: crate::tensor::Dtype = crate::tensor::Dtype::Bf16;
pub(crate) const FLASH_PREFILL_BASE_BYTES: u64 = 384 * 1024 * 1024;
pub(crate) const FLASH_PREFILL_ROW_BYTES: u64 = 512 * 1024;

/// Pricing: two routes price the ladder. The kernel that runs for a
/// multi-row chunk on a device that can dispatch it is the sink-bearing
/// float-flash prefill, charged by the measured base-plus-per-row terms
/// above. The composite attention fallback (the explicit [T, S] additive
/// mask tail in `shared_attention`, reached when the device refuses the
/// flash threadgroup staging, and the small-block vendor SDPA route that
/// serves chunks the flash predicate skips) materializes 3 x
/// [rows*32, context] F32 score+mask+softmax workspaces and keeps the
/// 3*32*context*4 per-row charge. Without the sub-512 rungs the composite
/// charge cliffs to ONE row whenever under ~6.4 GiB remain at 32K context
/// (the slowest path there is), which is why the ladder walks down to 4
/// instead.
fn fitting_prefill_piece(
    max_rows: usize,
    context: u32,
    remaining: u64,
    streamed_expert_bytes: u64,
    flash_prefill: bool,
) -> Result<usize> {
    let composite_bytes_per_row = 3 * 32 * u64::from(context) * 4;
    let charge = |rows: u32| {
        let kernel = if flash_prefill {
            FLASH_PREFILL_BASE_BYTES + FLASH_PREFILL_ROW_BYTES * u64::from(rows)
        } else {
            u64::from(rows) * composite_bytes_per_row
        };
        Ok(kernel + expert_staging_bytes(streamed_expert_bytes, rows))
    };
    let max_rows = u32::try_from(max_rows)
        .map_err(|_| EngError::BadRequest("Flint prefill ceiling exceeds u32".into()))?;
    let rows = crate::family::prefill::mlx::admitted_chunk(max_rows, remaining, charge)?;
    Ok(rows.unwrap_or(1) as usize)
}

fn pair_workspace(first: u32, second: u32, per_expert: u64, staging: PairStaging) -> u64 {
    let (first, second) = (u64::from(first), u64::from(second));
    let total = first + second;
    // Price both attention workspaces, one maximally populated stage, and
    // the hidden, projection and routed-pair buffers across both pieces.
    let scores = 3 * 32 * 4 * (first * first + second * total);
    let stage = 2 * per_expert * staging.slots() as u64;
    // Waves keep projected pairs while concatenating outputs and restoring rank order.
    let restore = match staging {
        PairStaging::Shared => 0,
        PairStaging::Waves64 => 12 * total * 2048 * 2 + (256 << 10),
    };
    scores + stage + 32 * total * 2048 * 2 + restore
}

fn admitted_pair_plan(
    prompt_rows: usize,
    first_piece: usize,
    remaining_bytes: u64,
    per_expert: u64,
    policy: PrefillPairPolicy,
) -> Option<PairPlan> {
    let second = prompt_rows.checked_sub(first_piece)?;
    if !(3..=1024).contains(&first_piece) || !(2..=first_piece).contains(&second) {
        return None;
    }
    let split = first_piece - 1;
    let candidates: &[PairStaging] = match policy {
        PrefillPairPolicy::Off => &[],
        PrefillPairPolicy::Shared => &[PairStaging::Shared],
        PrefillPairPolicy::Auto => &[PairStaging::Shared, PairStaging::Waves64],
        PrefillPairPolicy::Waves64 => &[PairStaging::Waves64],
    };
    candidates
        .iter()
        .copied()
        .find(|&staging| {
            pair_workspace(split as u32, second as u32, per_expert, staging) <= remaining_bytes
        })
        .map(|staging| PairPlan { split, staging })
}

#[test]
fn prefill_pair_admission_accounts_for_both_pieces_before_mutation() {
    let expert = 1_769_472;
    let shared = pair_workspace(1023, 1017, expert, PairStaging::Shared);
    let waves = pair_workspace(1023, 1017, expert, PairStaging::Waves64);
    let plan = |budget, policy| admitted_pair_plan(2041, 1024, budget, expert, policy);
    assert_eq!(
        plan(shared, PrefillPairPolicy::Auto),
        Some(PairPlan {
            split: 1023,
            staging: PairStaging::Shared
        }),
    );
    assert_eq!(
        plan(waves, PrefillPairPolicy::Auto),
        Some(PairPlan {
            split: 1023,
            staging: PairStaging::Waves64
        }),
    );
    assert_eq!(plan(waves - 1, PrefillPairPolicy::Auto), None);
    assert_eq!(plan(shared - 1, PrefillPairPolicy::Shared), None);
    assert_eq!(
        plan(u64::MAX, PrefillPairPolicy::Waves64).unwrap().staging,
        PairStaging::Waves64
    );
    assert_eq!(plan(u64::MAX, PrefillPairPolicy::Off), None);
    for first in [512, 1024] {
        let plan =
            |rows| admitted_pair_plan(rows, first, u64::MAX, expert, PrefillPairPolicy::Auto);
        for rows in [first - 1, first, first + 1, 2 * first + 1, 3 * first] {
            assert_eq!(plan(rows), None, "first piece {first}, prompt {rows}");
        }
        for rows in [first + 2, 2 * first] {
            assert_eq!(
                plan(rows),
                Some(PairPlan {
                    split: first - 1,
                    staging: PairStaging::Shared,
                }),
                "first piece {first}, prompt {rows}",
            );
        }
    }
}

fn device_session_charges(retention_cap: u64) -> Result<[(&'static str, u64); 5]> {
    use crate::family::{
        kv_format::{KvFormat, KvRowShape},
        mlx_budget::PRESSURE_MARGIN,
    };

    let context = context_limit();
    let full = super::config::FULL_LAYERS.len() as u64;
    let sliding = super::config::LAYERS as u64 - full;
    let row_bytes = KvFormat::Bf16.layer_position_bytes(KvRowShape {
        heads_kv: 4,
        head_dim: 128,
    })?;
    let full_plane_bytes = u64::from(context) * row_bytes;
    let state = full * full_plane_bytes + sliding * u64::from(super::mlx_state::WINDOW) * row_bytes;
    let retained = retention_cap.min(state * 3);
    let cache = super::mlx_model::cache_limit_bytes();
    Ok([
        ("session_state_bytes", state),
        ("prefix_reserve_bytes", retained),
        ("mlx_cache_limit_bytes", cache),
        ("one_full_layer_growth_bytes", full_plane_bytes),
        ("pressure_margin_bytes", PRESSURE_MARGIN),
    ])
}

fn device_prefill_piece(
    device: crate::family::mlx_budget::DeviceBudget,
    model: &Model,
    max_rows: usize,
    session: &[(&'static str, u64)],
) -> Result<(usize, MlxPrefillMemory, bool)> {
    // Model::open_for_budget has imported its resident arrays. The allocator
    // count includes those arrays; the prefill reserve separately prices the
    // whole session including the fixed-size rings.
    let weights = model.kernels.memory_stats()?.active;
    let per_expert = model.streamed_expert_bytes;
    let mut charges = session.to_vec();
    charges.push(("resident_active_bytes", weights));
    if model.cached_expert_bytes != 0 {
        charges.push(("resident_expert_cache_bytes", model.cached_expert_bytes));
    }
    charges.push((
        "decode_expert_staging_bytes",
        expert_staging_bytes(per_expert, 1),
    ));
    device.remaining("Flint MLX", &charges)?;
    // Each batched candidate charges the whole staging set for its own row count.
    charges.pop();
    let remaining = device.remaining("Flint MLX", &charges)?;
    // Price by the route the chunk will take: the sink-bearing float-flash
    // kernel unless the live device refuses its threadgroup staging.
    let flash_prefill = model.kernels.float_flash_prefill_available(
        PREFILL_FLASH_DTYPE,
        super::config::PREFILL_FLASH_HEAD_DIM,
    )?;
    let rows = fitting_prefill_piece(
        max_rows,
        context_limit(),
        remaining,
        per_expert,
        flash_prefill,
    )?;
    let terms = charges
        .iter()
        .map(|(name, bytes)| format!("{name}={bytes}"))
        .collect::<Vec<_>>()
        .join(" ");
    let staging = expert_staging_bytes(per_expert, rows as u32);
    eprintln!(
        "flint: MLX device budget selects {rows} prefill rows (ceiling {max_rows}); {device} {terms} expert_staging_bytes={} remaining_bytes={remaining}",
        staging,
    );
    Ok((
        rows,
        MlxPrefillMemory {
            prefill_budget_bytes: device.bytes(),
            remaining_bytes: remaining,
            device: Some(device.ceiling_report()),
            base_charges: charges.into_iter().collect(),
            workspace: [("expert_staging_bytes", staging)].into(),
        },
        flash_prefill,
    ))
}

#[cfg(test)]
#[test]
fn device_prefill_walks_the_ladder_and_respects_the_ceiling() {
    let context = 32768;
    let row = 3 * 32 * u64::from(context) * 4;
    // Composite pricing (flash unavailable): each rung is the largest whose
    // whole-row charge fits; the rung below takes over where the old
    // two-rung ladder cliffed to one row.
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 512 - 1, 0, false).unwrap(),
        256
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 512, 0, false).unwrap(),
        512
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 1024, 0, false).unwrap(),
        1024
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 256 - 1, 0, false).unwrap(),
        128
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 4 - 1, 0, false).unwrap(),
        // Below four rows' worth of workspace no multi-row rung fits; the
        // ladder has no 2 rung, so the planner falls to one row.
        1
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 4, 0, false).unwrap(),
        4
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row - 1, 0, false).unwrap(),
        1
    );
    assert_eq!(
        fitting_prefill_piece(512, context, u64::MAX, 0, false).unwrap(),
        512
    );
    assert_eq!(
        fitting_prefill_piece(1, context, u64::MAX, 0, false).unwrap(),
        1
    );
    assert_eq!(
        fitting_prefill_piece(1024, context / 4, row * 512, 0, false).unwrap(),
        1024
    );
    // Streamed experts add a per-row staging charge to every candidate.
    let expert = 1 << 20;
    let staged = expert_staging_bytes(expert, 512);
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 512 + staged - 1, expert, false).unwrap(),
        256
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, row * 512 + staged, expert, false).unwrap(),
        512
    );
}

#[cfg(test)]
#[test]
fn device_prefill_flash_pricing_uses_the_measured_charge() {
    let context = 32768;
    let base = FLASH_PREFILL_BASE_BYTES;
    let row = FLASH_PREFILL_ROW_BYTES;
    // Flash pricing: the same ladder over the measured base+per-row charge;
    // the ~12 MiB composite row becomes ~0.5 MiB, so the same remaining
    // budget that bought 256 composite rows buys the ceiling.
    assert_eq!(
        fitting_prefill_piece(1024, context, base + row * 1024, 0, true).unwrap(),
        1024
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, base + row * 1024 - 1, 0, true).unwrap(),
        512
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, base + row * 4, 0, true).unwrap(),
        4
    );
    // Below the base term even one row does not fit; the planner still
    // returns one row (the floor), never zero.
    assert_eq!(
        fitting_prefill_piece(1024, context, base - 1, 0, true).unwrap(),
        1
    );
    // At a 30,064 MiB device ceiling the session and resident charges
    // leave ~8.86 GB; composite pricing bought 512 rows there, flash
    // pricing buys the 1024 ceiling.
    let remaining = 8_859_732_416_u64;
    assert_eq!(
        fitting_prefill_piece(1024, context, remaining, 0, true).unwrap(),
        1024
    );
    assert_eq!(
        fitting_prefill_piece(1024, context, remaining, 0, false).unwrap(),
        512
    );
}

#[cfg(test)]
#[test]
fn device_prefill_does_not_cliff_on_the_measured_working_set() {
    // The Mac Studio on-leg numbers from the S1 serve proof: a 28 GiB
    // FIFI_MLX_WORKING_SET_MIB cap left exactly 5,638,506,944 bytes after
    // resident weights, session state, reserves and the pressure margin.
    // The old two-rung ladder selected ONE row there (512 rows cost
    // 6,442,450,944); the walked ladder must not.
    let context = 32768;
    let remaining = 5_638_506_944_u64;
    let selected = fitting_prefill_piece(1024, context, remaining, 0, false).unwrap();
    assert!(selected > 1, "cliffed to one row: {selected}");
    assert_eq!(selected, 256);
    // Flash pricing on the same remaining budget selects the ceiling: the
    // measured charge (base + 512 KiB/row) fits 1,024 rows in 5.6 GB many
    // times over.
    let selected = fitting_prefill_piece(1024, context, remaining, 0, true).unwrap();
    assert_eq!(selected, 1024);
}

#[test]
fn request_span_admits_a_wider_piece_at_the_same_budget() {
    let expert = 1_769_472;
    let remaining = 5_453_957_568;
    assert_eq!(
        fitting_prefill_piece(1024, 32768, remaining, expert, false).unwrap(),
        256
    );
    assert_eq!(
        fitting_prefill_piece(1024, 2041, remaining, expert, false).unwrap(),
        1024
    );
    let just_fits = 3 * 32 * 4 * 2041 * 1024 + expert_staging_bytes(expert, 1024);
    assert_eq!(
        fitting_prefill_piece(1024, 2041, just_fits, expert, false).unwrap(),
        1024
    );
    assert_eq!(
        fitting_prefill_piece(1024, 2041, just_fits - 1, expert, false).unwrap(),
        512
    );
    assert_eq!(
        fitting_prefill_piece(1024, 2041, remaining, expert, true).unwrap(),
        fitting_prefill_piece(1024, 32768, remaining, expert, true).unwrap(),
        "flash workspace is independent of prompt span"
    );
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
        st.request_prefill_piece = None;
        st.prefill_pair = false;
        st.core
            .reset(&mut Step::new(&mut self.model, &mut st.state))
    }

    fn reset_for_request(
        &mut self,
        st: &mut FlintSession,
        prompt_tokens: usize,
        completion_limit: Option<u32>,
        has_prefix_boundary: bool,
    ) -> Result<()> {
        self.reset(st)?;
        st.prefill_pair = self.prefill_pair != PrefillPairPolicy::Off && !has_prefix_boundary;
        if self.request_prefill {
            let mut rows = self.prefill_piece;
            // Prefix snapshots keep their established chunk tiling.
            if !has_prefix_boundary && let Some(memory) = &self.mlx_plan.memory {
                let prompt_rows = prompt_tokens.min(context_limit() as usize);
                rows = rows.max(fitting_prefill_piece(
                    self.mlx_plan.configured_prefill_rows.min(prompt_rows),
                    prompt_rows as u32,
                    memory.remaining_bytes,
                    self.model.streamed_expert_bytes,
                    self.flash_prefill,
                )?);
            }
            st.request_prefill_piece = NonZeroUsize::new(rows);
        }
        // Prefix jobs keep geometric growth and its existing snapshot charges.
        if self.request_sized_kv && !has_prefix_boundary {
            st.state.full_capacity = super::mlx_state::initial_request_capacity(
                prompt_tokens,
                completion_limit,
                context_limit(),
            );
        }
        Ok(())
    }

    fn prefill(&mut self, st: &mut FlintSession, tokens: &[u32]) -> Result<()> {
        st.core.prefill(
            &mut Step::new(&mut self.model, &mut st.state),
            tokens,
            context_limit() as usize,
        )
    }

    fn try_prefill_pair(
        &mut self,
        st: &mut FlintSession,
        tokens: &[u32],
        first_piece: usize,
        cancel: &AtomicBool,
    ) -> Result<PrefillPairOutcome> {
        if !st.prefill_pair || !st.core.history().is_empty() || st.state.pos != 0 {
            return Ok(PrefillPairOutcome::Declined);
        }
        let (Some(piece), Some(memory)) = (st.request_prefill_piece, &self.mlx_plan.memory) else {
            return Ok(PrefillPairOutcome::Declined);
        };
        if first_piece != piece.get() {
            return Ok(PrefillPairOutcome::Declined);
        }
        let Some(plan) = admitted_pair_plan(
            tokens.len(),
            first_piece,
            memory.remaining_bytes,
            self.model.streamed_expert_bytes,
            self.prefill_pair,
        ) else {
            return Ok(PrefillPairOutcome::Declined);
        };
        // DecodeSession leaves the last token pending. The split preserves
        // the boundary just before the first piece's former pending token.
        let cancel = self.cancel_prefill_pair.then_some(cancel);
        let forward = st
            .core
            .prefill_with(tokens, context_limit() as usize, |rows| {
                self.model.prefill_pair(&mut st.state, rows, plan, cancel)
            })?;
        if forward.is_break() {
            Ok(PrefillPairOutcome::Cancelled)
        } else {
            Ok(PrefillPairOutcome::Completed)
        }
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
            &mut Step::new(&mut self.model, &mut st.state),
            generation,
            &limits,
        )
    }

    fn boundary(&mut self, _st: &mut FlintSession) {}

    fn prefill_piece(&self) -> Option<usize> {
        if self.request_prefill {
            Some(self.mlx_plan.configured_prefill_rows)
        } else {
            Some(self.prefill_piece)
        }
    }

    fn request_prefill_piece(&self, st: &FlintSession) -> Option<NonZeroUsize> {
        st.request_prefill_piece
    }

    fn mlx_execution_plan(&self) -> Option<&MlxExecutionPlan> {
        Some(&self.mlx_plan)
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
        let path = FlintPath::now(&self.root, self.prefill_piece);
        let fed = st.state.pos as usize;
        st.core
            .save_prefix(&mut self.prefix, key, path, fed, kind, || {
                Ok(st.state.snapshot_prefix())
            })
    }

    fn restore_prefix(&mut self, st: &mut FlintSession, prompt: &[u32]) -> Result<usize> {
        let path = FlintPath::now(&self.root, self.prefill_piece);
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
    pub(super) compact_kv: bool,
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
    pub(super) fn now(root: &Path, prefill_piece: usize) -> Self {
        use crate::kernels::mlx;
        Self::from_env_values(
            root,
            context_limit(),
            FlintEnv {
                // The same resolvers the serving path uses, so the
                // fingerprint can never disagree with the route.
                batch_prefill: prefill_piece,
                decode_attn: mlx::decode_attn_flagged(),
                ring_view: super::mlx_forward::ring_view_flagged(),
                prefill_chain: super::mlx_forward::prefill_chain_flagged(),
                compile_layer: mlx::compile_layer_flagged(),
                t1_flash: mlx::t1_flash_flagged(),
                async_chain: super::mlx_forward::async_chain_flagged(),
                coalesce_residual: super::mlx_forward::coalesce_residual_flagged(),
                compact_kv: super::mlx_state::compact_kv_flagged(),
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
