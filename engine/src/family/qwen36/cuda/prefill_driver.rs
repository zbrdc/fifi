//! The chunked-prefill driver of the resident step: the admission inputs of the session arena, the
//! pre-arena half of the chunk gate, the zero spans of the arena fill, and the arena owner itself
//! with its single zero fill at session build.
//!
//! The pure pieces are host arithmetic only, no CUDA call: [`AdmitFacts`] and [`admit_m_sess`] feed
//! the prefill plan's `Admission` the REAL state term, guarded by the planned-against-allocated
//! equality; [`needs_arena`] is the first four conditions of `chunk_verdict`, the half an
//! arena-owning session must pass before any device query; [`zero_spans`] cuts the one reused zero
//! buffer of the arena fill. The device half is [`PrefillArena`]: `QwenResidentStep::admit_prefill`
//! allocates and zero-fills it once per session, and the per-chunk protocol (the chunk body of
//! `prefill.rs` and the driver around it) reads its tensors. The `io` counter (test builds only)
//! pins the I/O rule: no per-chunk allocation, no D2H of intermediate buffers, one status read per
//! chunk, and the zero fill once at build.

use super::{
    dims::{KvMode, StepDims},
    prefill_plan::{
        Admission, ChunkCost, ChunkSlot, Extra, M_LADDER, PART_CHARGE, PrefillStep, SPARE_MIN,
        chunk_list_with_depth_tier, chunk_tensors, exl3_chunk_tensors,
    },
};
use crate::{
    EngError,
    family::{
        prefill::arena::{ArenaHooks, allocate_zeroed, drive_steps},
        project::cuda::WeightLayout,
    },
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream,
        prefill::{
            attn_prefill::CPF_ATTN_PART_CAP, kv8_dequant::staging_bytes, mmq_facts::MmqFacts,
        },
    },
};

// The family-neutral halves moved to the shared prefill core (`family::prefill`); these
// re-exports keep the old paths alive for the step and the driver tests.
pub(super) use crate::family::prefill::admission::AdmitFacts;
pub(super) use crate::family::prefill::arena::check_chunk_start;
#[cfg(test)]
pub(super) use crate::family::prefill::arena::{ZERO_CHUNK, zero_spans};

/// Free VRAM after the weight load and before state/decode allocation, less the selected
/// headroom. Extras include caller-owned allocations such as the device prefix tip.
/// Fixed admission also checks fresh free VRAM before allocating its remaining buffers.
pub(super) struct PrefillBudget {
    pub(super) device_free: usize,
    pub(super) extras: Vec<Extra>,
}

/// Test-only prefill I/O counters for the I/O rule of the module doc.
/// These counters do not measure every stream synchronization. KV8 also uses blocking
/// causal-length writes before each attention tile. Neither path copies intermediate
/// buffers to host or allocates per chunk. The arena zero fill runs once at session build.
///
/// Like `mmq_facts`, these counters use std atomics. A device test reads them after construction
/// and after a chunked run. Production does not use these counters.
#[cfg(test)]
pub(super) mod io {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Session allocations: chunk arena, part and optional KV8 staging.
    pub(in super::super) static ALLOCS: AtomicUsize = AtomicUsize::new(0);
    /// Zero-fill span writes of the arena build (27-93 for the ladder's arenas).
    pub(in super::super) static SPAN_WRITES: AtomicUsize = AtomicUsize::new(0);
    /// Scribble writes of the golden capture arm (0 without `FIFI_GOLDEN_SCRIBBLE`): the whole
    /// fresh arena in [`ZERO_CHUNK`] pieces, so the arm can assert the scribble ran at all.
    pub(in super::super) static SCRIBBLES: AtomicUsize = AtomicUsize::new(0);

    pub(in super::super) fn reset() {
        ALLOCS.store(0, Ordering::Relaxed);
        SPAN_WRITES.store(0, Ordering::Relaxed);
    }

    pub(super) fn bump_alloc() {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn bump_span_write() {
        SPAN_WRITES.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn bump_scribble() {
        SCRIBBLES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Feeds the prefill plan's [`Admission`] with the REAL resident state: the planned term must equal
/// the allocated layout, so a figure from anywhere else — the KV bytes alone, about 62.8 MiB short
/// at the 16K window — is a `Checkpoint` refusal BEFORE anything else. [`Admission::admit`] then
/// picks the largest ladder size at most `cap` whose device sum leaves the spare minimum; `None`
/// means no size fits and the session keeps the per-token path. The Qwen3.6 constants bind here.
pub(super) fn admit_m_sess(
    facts: &AdmitFacts,
    extras: &[Extra],
    costs: &[ChunkCost],
) -> Result<Option<usize>, EngError> {
    crate::family::prefill::admission::admit_m_sess(
        facts,
        extras,
        costs,
        PART_CHARGE,
        SPARE_MIN,
        &M_LADDER,
        "prefill_chunk",
    )
}

/// One reusable layer's K/V half planes, bounded independently of stored KV capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Kv8StagingPlan {
    pub(super) capacity: usize,
    pub(super) bytes: usize,
}

impl Kv8StagingPlan {
    pub(super) fn new(capacity: usize, heads: usize, dim: usize) -> Result<Self, EngError> {
        Self::for_context(capacity.min(65_536), heads, dim)
    }

    pub(super) fn for_context(capacity: usize, heads: usize, dim: usize) -> Result<Self, EngError> {
        Ok(Self {
            capacity,
            bytes: staging_bytes(capacity, heads, dim)?,
        })
    }

    pub(super) fn covers(self, end: usize) -> bool {
        (1..=self.capacity).contains(&end)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kv8Staging<'a> {
    pub(super) plan: Kv8StagingPlan,
    pub(super) segmented: bool,
    /// Time-major K then V, separate from the attention partial workspace.
    pub(super) planes: [DeviceRange<'a>; 2],
}

/// Prefer staged attention; keep the legacy all-or-none rule through 64K.
/// Extended requests shrink to the largest prefix accepted by the same budget predicate.
pub(super) fn admit_with_staging(
    facts: &AdmitFacts,
    extras: &[Extra],
    costs: &[ChunkCost],
    staging: Option<Kv8StagingPlan>,
) -> Result<(Option<usize>, Option<Kv8StagingPlan>), EngError> {
    let Some(plan) = staging else {
        return Ok((admit_m_sess(facts, extras, costs)?, None));
    };
    let mut charged = extras.to_vec();
    let staging_index = charged.len();
    charged.push(Extra {
        name: "kv8_fp16_staging",
        fixed: plan.bytes,
        bytes_per_token: 0,
    });
    if let Some(m) = admit_m_sess(facts, &charged, costs)? {
        return Ok((Some(m), Some(plan)));
    }
    let fallback = admit_m_sess(facts, extras, costs)?;
    if plan.capacity <= 65_536 || fallback.is_none() {
        return Ok((fallback, None));
    }
    let row_bytes = plan.bytes / plan.capacity;
    let mut low = 1;
    let mut high = plan.capacity - 1;
    let mut best = None;
    while low <= high {
        let capacity = low + (high - low) / 2;
        // Shrinking the checked allocation cannot overflow its proportional byte count.
        let bytes = capacity * row_bytes;
        charged[staging_index].fixed = bytes;
        if let Some(m) = admit_m_sess(facts, &charged, costs)? {
            best = Some((m, Kv8StagingPlan { capacity, bytes }));
            low = capacity + 1;
        } else {
            high = capacity - 1;
        }
    }
    Ok(best.map_or((fallback, None), |(m, plan)| (Some(m), Some(plan))))
}

pub(super) const FIXED_ADMISSION_MARGIN: usize =
    crate::family::qwen36::admission_sketch::FIXED_ADMISSION_MARGIN_BYTES;

pub(super) fn fixed_chunk_rows(capacity: usize, cap: usize) -> usize {
    // Below the stage tier the whole context fits one stage; above it the
    // smaller tail keeps the fixed reserve sane. The tier bound is
    // FULL_STAGE_TIER_ROWS: the wide candidate and the tier are one boundary.
    cap.min(if capacity <= FULL_STAGE_TIER_ROWS {
        1024
    } else {
        512
    })
}

pub(super) fn fixed_staging_rows(capacity: usize) -> usize {
    crate::family::qwen36::admission_sketch::fixed_staging_rows(capacity)
}

pub(super) fn admit_fixed(
    facts: &AdmitFacts,
    extras: &[Extra],
    costs: &[ChunkCost],
    staging: Kv8StagingPlan,
) -> Result<ChunkCost, EngError> {
    let cost = costs
        .iter()
        .find(|cost| cost.m == facts.cap)
        .ok_or_else(|| {
            EngError::BadRequest("fixed prefill admission: unsupported row tier".into())
        })?;
    let mut charged = extras.to_vec();
    charged.push(Extra {
        name: "kv8_fp16_staging",
        fixed: staging.bytes,
        bytes_per_token: 0,
    });
    if admit_m_sess(facts, &charged, std::slice::from_ref(cost))? != Some(cost.m) {
        return Err(EngError::MemoryPlan(format!(
            "fixed prefill admission: {} rows and {} staging bytes do not fit the budget",
            cost.m, staging.bytes
        )));
    }
    Ok(*cost)
}

/// The full-stage tier bound: the maximum rows a single default stage carries AND
/// the boundary above which `fixed_staging_rows` shrinks to the 32K tail — a
/// staging tier bound, not a context cap. The wide CPF candidate is defined from
/// this tier; retuning the candidate means retuning the tier deliberately.
/// Canonical value and rule live in `admission_sketch`, shared with preflight.
pub(super) const FULL_STAGE_TIER_ROWS: usize =
    crate::family::qwen36::admission_sketch::FULL_STAGE_TIER_ROWS;

/// The wide candidate applies only where it is actually wider than today's stage.
pub(super) fn wide_stage_candidate(capacity: usize, wide_flag: bool) -> bool {
    wide_flag && capacity > FULL_STAGE_TIER_ROWS
}

/// Names the boot line's stage provenance for the OFF/ON A/B receipts.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum StageChoice {
    Wide,
    Default,
}

impl StageChoice {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            StageChoice::Wide => "wide",
            StageChoice::Default => "default",
        }
    }
}

/// Selects the fixed-prefill stage under one live snapshot. The wide candidate (131072
/// rows) is tried first where it applies; any MemoryPlan from it falls back to the default
/// stage, and only then does the tight-snapshot ladder run: a budget or live-room refusal
/// halves the stage while the rows halve exactly, down to `fixed_stage_min_rows(kv_capacity)`
/// — the split bound composed with the performance floor — where `kv_capacity` is the KV
/// context capacity (not a floor). Above 131,072 rows the ladder adds up to
/// 48 MiB (64 MiB stage -> 16 MiB floor) of headroom over the 32,768-row default
/// stage, 240 MiB when the wide fallback ran first; at or below 131,072 rows it runs
/// from the full-context stage down to the floor instead (240 MiB at 131,072,
/// 112 MiB at 65,536). A snapshot too tight for every rung down to the
/// floor refuses the boot with the ORIGINAL full-stage error text; odd stages keep that
/// refusal too because bytes cannot halve exactly. Every MemoryPlan from an attempt
/// shrinks — a budget or live-room refusal, a charge overflow (a smaller stage can
/// un-overflow the sum), or live VRAM below the margin — every other error surfaces
/// unchanged.
pub(super) fn admit_fixed_staged(
    facts: &AdmitFacts,
    extras: &[Extra],
    costs: &[ChunkCost],
    default_plan: Kv8StagingPlan,
    wide_plan: Option<Kv8StagingPlan>,
    free: usize,
    kv_capacity: usize,
) -> Result<(ChunkCost, Kv8StagingPlan, StageChoice), EngError> {
    // The floor is computed here, from the KV capacity, so every caller —
    // production and tests alike — gates the ladder with the same expression;
    // the floor-admit tests exercise exactly this computation.
    let min_capacity = fixed_stage_min_rows(kv_capacity);
    let attempt = |plan: Kv8StagingPlan| {
        admit_fixed(facts, extras, costs, plan)
            .and_then(|cost| require_fixed_room(free, &cost, plan).map(|_| cost))
    };
    if let Some(wide) = wide_plan {
        match attempt(wide) {
            Ok(cost) => return Ok((cost, wide, StageChoice::Wide)),
            // Every MemoryPlan from the wide attempt falls back to the default
            // stage — budget, live room, or arithmetic overflow alike: a plan
            // whose charge overflows does not fit, and the default stage is the
            // safe answer. Row-tier and checkpoint errors propagate from the
            // wide attempt itself. A non-Memory error cannot differ between the
            // two attempts — both read the same facts/costs and the row tier is
            // looked up before any plan — so propagating it is identical for
            // every input (mutation-tested equivalent).
            Err(EngError::MemoryPlan(_)) => {}
            Err(other) => return Err(other),
        }
    }
    match attempt(default_plan) {
        Ok(cost) => Ok((cost, default_plan, StageChoice::Default)),
        Err(first) if matches!(first, EngError::MemoryPlan(_)) => {
            // Overflowing charge sums also arrive as MemoryPlan and are retried
            // deliberately: a smaller stage can un-overflow the sum, and the floor
            // bounds the ladder, so the worst case refuses with `first`.
            let mut plan = default_plan;
            while let Some(next) = shrink_once(plan, min_capacity) {
                match attempt(next) {
                    Ok(cost) => return Ok((cost, next, StageChoice::Default)),
                    Err(EngError::MemoryPlan(_)) => plan = next,
                    Err(other) => return Err(other),
                }
            }
            Err(first)
        }
        Err(other) => Err(other),
    }
}

/// The architectural stage floor: the segmented-attention split bound for the context,
/// so every rung the ladder may keep satisfies `capacity.div_ceil(rung) <= CPF_ATTN_MAX_SPLITS`.
/// Pinned by tests: 262144 -> 2048, 180224 -> 1408.
pub(super) fn fixed_stage_floor(kv_capacity: usize) -> usize {
    kv_capacity.div_ceil(crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_MAX_SPLITS)
}

/// The performance floor: the lowest rung whose cold-prefill TTFT stays within 1.25x
/// the full stage. Deep rungs re-dequantize the whole prefix per query subtile, so a
/// silently much slower boot is worse than a clear refusal; rungs below this floor
/// refuse with the original full-stage text. 8192 is the last rung measured inside
/// the 1.25x cold-TTFT budget on one 24 GiB card: one cold 131,030-token prefill per
/// rung at the 180224 profile gave 48.9 s at the wide stage, 58.44 s at 8192 and
/// 78.67 s at 4096. That establishes the property only at that point (n = 1 per
/// rung, single prompt depth); at other profiles or prompt depths the sub-floor cost
/// is unmeasured. The split bound stays the floor for contexts above 1,048,576 rows.
pub(super) const FIXED_STAGE_PERF_FLOOR_ROWS: usize = 8_192;

/// The admission inputs and stage candidates of one boot, as the gates saw them:
/// every field feeds one figure of [`ledger_line`].
pub(super) struct LedgerFigures<'a> {
    pub(super) budget_post_margin: usize,
    pub(super) live_free: usize,
    pub(super) state_allocated: usize,
    pub(super) decode_arena: usize,
    pub(super) chunk_rows: usize,
    /// The cap-row chunk cost's part and arena terms (the cost the gates charge).
    pub(super) part: usize,
    pub(super) arena: usize,
    pub(super) extras: &'a [Extra],
    pub(super) staging: &'a Kv8StagingPlan,
    pub(super) wide: Option<&'a Kv8StagingPlan>,
}

/// The boot ledger line as one string: every charge the admission gates
/// compute, printed on accepted and refused boots alike, so a refusal can be
/// closed by subtraction instead of guessed from the outside. `part_charge`
/// is the charged law itself (`PART_CHARGE.max(part)`); `extras_fixed` sums
/// the flat extras — Admission::passes additionally adds each extra's
/// `bytes_per_token * m` via `extras_bytes`, which no flat figure can
/// capture; the only production extra (the device-prefix-delta, start.rs)
/// carries `bytes_per_token` 0, so the printed sum is the whole extras
/// charge today.
pub(super) fn ledger_line(f: &LedgerFigures) -> String {
    format!(
        "q36 cuda: fixed admission ledger: budget_post_margin={} \
live_free={} live_margin={} state={} decode_arena={} \
arena={} part_charge={} extras_fixed={} spare_min={} default_staging={} \
default_rows={} chunk_rows={}{}",
        f.budget_post_margin,
        f.live_free,
        FIXED_ADMISSION_MARGIN,
        f.state_allocated,
        f.decode_arena,
        f.arena,
        PART_CHARGE.max(f.part),
        f.extras.iter().map(|extra| extra.fixed).sum::<usize>(),
        SPARE_MIN,
        f.staging.bytes,
        f.staging.capacity,
        f.chunk_rows,
        f.wide.map_or(String::new(), |w| {
            format!(" wide_staging={} wide_rows={}", w.bytes, w.capacity)
        }),
    )
}

/// The boot-line provenance label: a stage below the full default stage came
/// from the shrink ladder; anything else names the choice that admitted it.
/// A wide boot (131072 rows over a 32768-row default) is NOT shrunk.
pub(super) fn stage_label(choice: StageChoice, rows: usize, full_rows: usize) -> &'static str {
    if rows < full_rows {
        "shrunk"
    } else {
        choice.as_str()
    }
}

/// The last rung the ladder may reach for a context: the segmented-attention
/// split bound composed with the performance floor, whichever is higher.
pub(super) fn fixed_stage_min_rows(kv_capacity: usize) -> usize {
    fixed_stage_floor(kv_capacity).max(FIXED_STAGE_PERF_FLOOR_ROWS)
}

/// Halves the stage while the capacity stays even and the half stays at or
/// above the floor. Odd stages cannot halve with exact bytes, so they keep the
/// full-stage refusal; every shipped profile stage is even (asserted in the tests).
fn shrink_once(plan: Kv8StagingPlan, min_capacity: usize) -> Option<Kv8StagingPlan> {
    let next = plan.capacity / 2;
    (next >= min_capacity && next * 2 == plan.capacity).then_some(Kv8StagingPlan {
        capacity: next,
        bytes: plan.bytes / 2,
    })
}

pub(super) fn require_fixed_room(
    free: usize,
    cost: &ChunkCost,
    staging: Kv8StagingPlan,
) -> Result<(), EngError> {
    let budget = free.checked_sub(FIXED_ADMISSION_MARGIN).ok_or_else(|| {
        EngError::MemoryPlan("fixed prefill admission: live VRAM is below the margin".into())
    })?;
    // State, decode and prefix owners already exist; charge only the remaining allocations.
    let admission = Admission {
        device_extras: vec![Extra {
            name: "kv8_fp16_staging",
            fixed: staging.bytes,
            bytes_per_token: 0,
        }],
        ..Admission::qwen36(budget)
    };
    if !admission.passes(0, cost)? {
        return Err(EngError::MemoryPlan(format!(
            "fixed prefill admission: {free} live bytes cannot retain {} rows, staging and the margin",
            cost.m
        )));
    }
    Ok(())
}

/// The pre-arena gate accepts a Native or EXL3 layout, `Fp16` or opted-in `Int8G32`, a non-tiered
/// session, and no Q6_K MMQ weight. These conditions match the first four `chunk_verdict`
/// checks. A rejected session needs no arena. `admit_prefill` runs this check before
/// device queries or arena allocation.
pub(super) fn needs_arena(
    layout: WeightLayout,
    kv_mode: KvMode,
    tiered: bool,
    mmq_q6k: bool,
    kv8_chunked_prefill: bool,
) -> bool {
    let supported_kv =
        kv_mode == KvMode::Fp16 || (kv_mode == KvMode::Int8G32 && kv8_chunked_prefill);
    layout != WeightLayout::Tiled && supported_kv && !tiered && !mmq_q6k
}

/// The two session-build guards of `admit_prefill` as a pure seam (condition 8a): the
/// arena is admitted before the first token (the mode freezes there, the form of
/// `set_capture_config`'s refusal) and at most once. `admit_prefill` calls this before
/// anything else, so the host rows pin both texts with the other guard satisfied.
pub(super) fn admit_guard(started: bool, admitted: bool) -> Result<(), EngError> {
    if started {
        return Err(EngError::BadRequest(
            "step: the prefill arena is admitted before the first token".to_owned(),
        ));
    }
    if admitted {
        return Err(EngError::BadRequest(
            "step: the prefill arena is already admitted".to_owned(),
        ));
    }
    Ok(())
}

/// The prefill arena of one session: the chunk tensors of the ADMITTED `M_sess` in one device
/// arena, the attention `part` buffer at [`CPF_ATTN_PART_CAP`], and the facts and budget that
/// admitted them. Built and zero-filled ONCE per session by [`PrefillArena::allocate`] through
/// `QwenResidentStep::admit_prefill` — never per request or per chunk — and held by the step for
/// its life; the step's field is declared before the state, so the arena drops first.
pub(super) struct PrefillArena {
    /// `chunk_tensors(dims, m_sess, facts.nsm)`: the tensor list of the admitted size.
    list: Vec<(ChunkSlot, ArenaTensor)>,
    /// The one allocation that holds the chunk tensors.
    arena: DeviceArena,
    /// The attention `part` buffer, at `CPF_ATTN_PART_CAP` and never at `ChunkCost::part`: a
    /// smaller chunk launched with its own split plan can need more than the `M_sess` plan, and the
    /// flat cap covers both.
    part: DeviceBuffer,
    kv8_staging: Option<(Kv8StagingPlan, DeviceBuffer, bool)>,
    m_sess: usize,
    /// The device facts of the admission (`cc`, `nsm`, `smpbo`), kept for the chunk builds.
    facts: MmqFacts,
    /// The admission budget, for the run report.
    budget: usize,
}

impl PrefillArena {
    /// Builds and zero-fills the arena of `m_sess` rows: the tensor list, ONE device arena for it,
    /// the WHOLE arena (the fixup range included) zeroed through one reused [`ZERO_CHUNK`] host
    /// buffer at every [`zero_spans`] span — 27-93 synchronous writes, about 9-78 ms, once per
    /// session — and then the `part` buffer at [`CPF_ATTN_PART_CAP`], which is NOT zero-filled (the
    /// attention kernel writes what it reads). `part` is allocated last, so any refusal drops the
    /// whole partly built arena before it returns.
    pub(super) fn allocate(
        ctx: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        m_sess: usize,
        facts: MmqFacts,
        budget: usize,
    ) -> Result<Self, EngError> {
        let list = chunk_tensors(dims, m_sess, facts.nsm)?;
        Self::allocate_list(ctx, stream, list, (m_sess, facts, budget))
    }

    /// The same for an EXL3 pack: the tensor list of [`exl3_chunk_tensors`], which has no MMQ
    /// scratch. `facts` keeps the device facts the attention plan reads.
    pub(super) fn allocate_exl3(
        ctx: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        m_sess: usize,
        facts: MmqFacts,
        budget: usize,
    ) -> Result<Self, EngError> {
        let list = exl3_chunk_tensors(dims, m_sess)?;
        Self::allocate_list(ctx, stream, list, (m_sess, facts, budget))
    }

    fn allocate_list(
        ctx: &CudaContext,
        stream: &DeviceStream,
        list: Vec<(ChunkSlot, ArenaTensor)>,
        (m_sess, facts, budget): (usize, MmqFacts, usize),
    ) -> Result<Self, EngError> {
        let tensors: Vec<ArenaTensor> = list.iter().map(|&(_, tensor)| tensor).collect();
        // The fill itself is the shared core's (`family::prefill::arena`): one arena, a whole-arena
        // 0xA5 scribble when the golden capture arm asks for it, then the one zero fill. The
        // test-only I/O counters ride along as hooks so the shared core stays family-neutral.
        #[cfg(test)]
        let (scribble, hooks) = (
            std::env::var_os("FIFI_GOLDEN_SCRIBBLE").is_some(),
            ArenaHooks {
                on_alloc: Some(io::bump_alloc),
                on_scribble: Some(io::bump_scribble),
                on_span_write: Some(io::bump_span_write),
            },
        );
        #[cfg(not(test))]
        let (scribble, hooks) = (false, ArenaHooks::default());
        let arena = allocate_zeroed(ctx, stream, &tensors, scribble, hooks)?;
        let part = ctx.allocate(CPF_ATTN_PART_CAP)?;
        #[cfg(test)]
        io::bump_alloc();
        Ok(Self {
            list,
            arena,
            part,
            kv8_staging: None,
            m_sess,
            facts,
            budget,
        })
    }

    /// The admitted chunk size, one of `M_LADDER` (never the raw cap).
    pub(super) fn m_sess(&self) -> usize {
        self.m_sess
    }

    /// Chunk arena plus optional staging bytes, excluding the separately reported part.
    pub(super) fn bytes(&self) -> usize {
        self.arena.bytes()
            + self
                .kv8_staging
                .as_ref()
                .map_or(0, |(plan, _, _)| plan.bytes)
    }

    pub(super) fn attach_kv8_staging(
        &mut self,
        ctx: &CudaContext,
        plan: Kv8StagingPlan,
        segmented: bool,
    ) -> Result<(), EngError> {
        if self.kv8_staging.is_some() {
            return Err(EngError::BadRequest("KV8 staging already attached".into()));
        }
        let buffer = ctx.allocate(plan.bytes)?;
        #[cfg(test)]
        io::bump_alloc();
        self.kv8_staging = Some((plan, buffer, segmented));
        Ok(())
    }

    pub(super) fn kv8_staging(&self) -> Result<Option<Kv8Staging<'_>>, EngError> {
        self.kv8_staging
            .as_ref()
            .map(|(plan, buffer, segmented)| {
                let half = plan.bytes / 2;
                Ok(Kv8Staging {
                    plan: *plan,
                    segmented: *segmented,
                    planes: [buffer.range(0, half)?, buffer.range(half, half)?],
                })
            })
            .transpose()
    }

    pub(super) fn segmented(&self) -> bool {
        self.kv8_staging.as_ref().is_some_and(|(_, _, mode)| *mode)
    }

    pub(super) fn part_bytes(&self) -> usize {
        self.part.bytes()
    }

    pub(super) fn list(&self) -> &[(ChunkSlot, ArenaTensor)] {
        &self.list
    }

    pub(super) fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    pub(super) fn part(&self) -> &DeviceBuffer {
        &self.part
    }

    pub(super) fn facts(&self) -> MmqFacts {
        self.facts
    }

    pub(super) fn budget(&self) -> usize {
        self.budget
    }
}

// `check_chunk_start` moved to the shared prefill core (`family::prefill::arena`); the old
// path lives as the re-export at the top of this file.

/// Drives the prompt rows of one feed_rows call through the prefill plan: each step gets its own
/// rows, a feed step exactly one. The start position is the committed row count at the call, so a
/// second prefill call or a prefix import continues at its own position. The list that ran is
/// returned.
pub(super) fn drive_chunks(
    start: usize,
    tokens: &[u32],
    m_sess: usize,
    fixed_depth: bool,
    run: impl FnMut(PrefillStep, &[u32]) -> Result<(), EngError>,
) -> Result<Vec<PrefillStep>, EngError> {
    let steps = chunk_list_with_depth_tier(start, tokens.len(), m_sess, fixed_depth)?;
    drive_steps(start, tokens, steps, run)
}

/// Preserves the per-token vocabulary refusal before checking the signed kernel index.
pub(super) fn chunk_ids(rows: &[u32], sampler_width: usize) -> Result<Vec<i32>, EngError> {
    rows.iter()
        .map(|&token| {
            super::rules::check_token_id(token, sampler_width)?;
            i32::try_from(token).map_err(|_| {
                EngError::BadRequest(format!("prefill chunk: token id {token} does not fit i32"))
            })
        })
        .collect()
}

// ---- the shared chunk-family surface ---------------------------------------------------------

use super::moe::Launch;
use super::prefill::{ChunkBufs, ChunkModel, ChunkRows, ChunkSparse};
use super::state::ResidentState;
use crate::family::prefill::chunk_family::ChunkFamily;

/// What the chunk body captures today, handed to the trait opaquely: the state, the launch, the
/// host-checked ids with their arena range, the prebuilt model, buffers, rows and sparse
/// projection. The ids are a PREFLIGHT product (`chunk_ids` runs before the window opens), so a
/// refused token id changes nothing on the device.
pub(super) struct Q36ChunkCtx<'a, 'w> {
    pub(super) state: &'a mut ResidentState,
    pub(super) launch: Launch<'a>,
    pub(super) ids: &'a [i32],
    pub(super) ids_range: DeviceRange<'a>,
    pub(super) model: &'a ChunkModel<'a, 'w>,
    pub(super) bufs: &'a ChunkBufs<'a>,
    pub(super) chunk_rows: ChunkRows,
    pub(super) sparse: Option<ChunkSparse<'a>>,
}

/// The Qwen3.6 [`ChunkFamily`]: every method calls the site it replaced verbatim; the
/// instruction order inside a chunk is unchanged. `run_layers` stores the run's part count in
/// `segmented_parts` for the step's report (a Qwen3.6-only figure, so it stays here and not in
/// the trait).
pub(super) struct Q36ChunkFamily<'a, 'w> {
    bank: Option<&'a std::cell::RefCell<super::exl3_bank::Exl3SlotResolver<'w>>>,
    #[cfg(test)]
    kernel_class: super::prefill::ChunkKernelClass,
    pub(super) segmented_parts: usize,
}

impl<'a, 'w> Q36ChunkFamily<'a, 'w> {
    pub(super) fn new(
        bank: Option<&'a std::cell::RefCell<super::exl3_bank::Exl3SlotResolver<'w>>>,
        #[cfg(test)] kernel_class: super::prefill::ChunkKernelClass,
    ) -> Self {
        Self {
            bank,
            #[cfg(test)]
            kernel_class,
            segmented_parts: 0,
        }
    }
}

impl<'a, 'w> ChunkFamily for Q36ChunkFamily<'a, 'w> {
    type Ctx = Q36ChunkCtx<'a, 'w>;

    fn begin_window(&mut self, ctx: &mut Self::Ctx) -> Result<(), EngError> {
        ctx.state.begin_prefill()
    }

    fn clear_chunk_status(&mut self, ctx: &mut Self::Ctx) -> Result<(), EngError> {
        ctx.state.clear_prefill_status(ctx.launch.stream)
    }

    fn write_chunk_inputs(
        &mut self,
        ctx: &mut Self::Ctx,
        _p0: usize,
        _m: usize,
    ) -> Result<(), EngError> {
        let bytes: Vec<u8> = ctx.ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        ctx.ids_range
            .buffer()
            .write_at(ctx.launch.stream, ctx.ids_range.offset(), &bytes)
    }

    fn run_layers(&mut self, ctx: &mut Self::Ctx, _p0: usize, _m: usize) -> Result<(), EngError> {
        let run = super::prefill::ChunkRun::new(&*ctx.state, ctx.model, ctx.bufs, ctx.chunk_rows)?;
        let run = run.with_bank(self.bank).with_sparse(ctx.sparse.take());
        #[cfg(test)]
        let run = run.with_kernel_class(self.kernel_class);
        super::prefill::run_chunk_body(&ctx.launch, &run)?;
        self.segmented_parts = run.segmented_parts();
        Ok(())
    }

    fn read_health(&mut self, ctx: &mut Self::Ctx) -> Result<i32, EngError> {
        ctx.state.read_prefill_status(ctx.launch.stream)
    }

    fn commit_chunk(
        &mut self,
        ctx: &mut Self::Ctx,
        p0: usize,
        m: usize,
        status: i32,
    ) -> Result<(), EngError> {
        ctx.state.revalidate(ctx.launch.stream, p0 + m, status)
    }

    fn reset(&mut self, ctx: &mut Self::Ctx) -> Result<(), EngError> {
        ctx.state.reset(ctx.launch.stream)
    }

    fn bank_misses(&self) -> Option<u64> {
        self.bank.as_ref().map(|bank| bank.borrow().stats().1)
    }
}
