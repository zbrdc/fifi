//! The chunked-prefill driver of the resident step (decision 0049, packet CPF-7b-C): the
//! admission inputs of the session arena, the pre-arena half of the chunk gate, the zero
//! spans of the arena fill, and the arena owner itself with its single zero fill at
//! session build.
//!
//! The pure pieces are host arithmetic only, no CUDA call: [`AdmitFacts`] and
//! [`admit_m_sess`] feed CPF-1's `Admission` the REAL state term, guarded by the
//! planned-against-allocated equality; [`needs_arena`] is the first four conditions of
//! `chunk_verdict`, the half an arena-owning session must pass before any device query;
//! [`zero_spans`] cuts the one reused zero buffer of the arena fill (ruling (c), plan
//! §C.3). The device half is [`PrefillArena`]: `QwenResidentStep::admit_prefill` (C2)
//! allocates and zero-fills it once per session, and the per-chunk protocol of plan §C.4
//! (CPF-7a part 2's body plus CPF-7b-C C3/C4) reads its tensors. The `io` counter (test
//! builds only) pins protection 2: no per-chunk allocation, no D2H of intermediate
//! buffers, one status read per chunk, and the zero fill once at build.

use super::{
    dims::{KvMode, StepDims},
    prefill_plan::{
        Admission, ChunkCost, ChunkSlot, Extra, PrefillStep, chunk_list, chunk_tensors,
    },
};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceStream,
        prefill::{attn_prefill::CPF_ATTN_PART_CAP, mmq_facts::MmqFacts},
    },
};

/// The measured budget of one session build (CPF-7b-C B.1): the caller — `make` of
/// `start.rs`, once the switch commit wires it (C5) — reads free VRAM ONCE after the weight
/// load and before the state and decode-arena allocations, less the 1 GiB headroom, and
/// passes it here with its extras (`make` passes none; CPF-7c's probe is seam I4's
/// [`Extra`]).
pub(super) struct PrefillBudget {
    /// Free device bytes after the load, less the headroom.
    pub(super) device_free: usize,
    /// The admission's device extras (seam I4).
    pub(super) extras: Vec<Extra>,
}

/// The per-chunk I/O counter of protection 2 (lane S, OK'd by git-db 2026-10-14): the
/// driver adds no host synchronisation per chunk beyond the single prefill-status read, no
/// D2H of intermediate buffers and no per-chunk allocation, and the arena zero fill runs
/// ONCE at session build, never per request or per chunk. Test builds only (the
/// `mmq_facts` query-counter form, std atomics only): the binding device row (DV-B9) reads
/// the counters after the build and after a chunked run; production is untouched.
#[cfg(test)]
pub(super) mod io {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Device allocations the driver made: the arena buffer and `part`, once at build.
    pub(super) static ALLOCS: AtomicUsize = AtomicUsize::new(0);
    /// Zero-fill span writes of the arena build (27-93 for the ladder's arenas).
    pub(super) static SPAN_WRITES: AtomicUsize = AtomicUsize::new(0);
    /// Prefill-status reads, exactly one per chunk at chunk close (protection 2).
    pub(super) static STATUS_READS: AtomicUsize = AtomicUsize::new(0);
    /// Device-to-host reads of intermediate buffers, which protection 2 bans per chunk.
    pub(super) static D2H_READS: AtomicUsize = AtomicUsize::new(0);

    /// Zeroes the counters. Only the build-side counters exist yet; the chunk-side ones
    /// (the status reads, the D2H reads) join when their bump sites land with the chunk
    /// protocol (C3b/C4), which also deletes their staging expects above.
    pub(super) fn reset() {
        ALLOCS.store(0, Ordering::Relaxed);
        SPAN_WRITES.store(0, Ordering::Relaxed);
    }

    /// One device allocation of the build side (the arena buffer, then `part`).
    pub(super) fn bump_alloc() {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }

    /// One span write of the zero fill.
    pub(super) fn bump_span_write() {
        SPAN_WRITES.fetch_add(1, Ordering::Relaxed);
    }
}

/// The admission inputs the step reads at session build (CPF-7b-C B.1): the budget the
/// caller measured, the decode arena, the state term in its two forms, and the cap of
/// `M_sess`. Pure host values, so the host rows pin the rule without a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AdmitFacts {
    /// `PrefillBudget::device_free`: free VRAM measured ONCE after the weight load and
    /// before the state and decode-arena allocations, less the 1 GiB headroom (`start.rs`).
    pub(super) budget: usize,
    /// `res.arena.bytes()` of the decode plan (`DeviceArena::bytes`).
    pub(super) decode_arena: usize,
    /// `state_bytes(dims, kv_mode, capacity, delta_slots)`: the PLANNED state term.
    pub(super) state_planned: usize,
    /// `self.state.layout().bytes()`: the state the step really allocated. The KV mode, the
    /// KV capacity and the DeltaNet slots all flow through the layout (`ResidentState::allocate`).
    pub(super) state_allocated: usize,
    /// `self.m_sess_cap as usize` (`set_m_sess_cap`).
    pub(super) cap: usize,
}

/// Feeds CPF-1's [`Admission`] with the REAL resident state: the planned term must equal the
/// allocated layout, so a figure from anywhere else — the KV bytes alone, about 62.8 MiB
/// short at the 16K window (plan §C.3) — is a `Checkpoint` refusal BEFORE anything else.
/// `Admission::admit` then picks the largest ladder size at most `cap` whose device sum
/// leaves the spare minimum; `None` means no size fits and the session keeps the per-token
/// path.
pub(super) fn admit_m_sess(
    facts: &AdmitFacts,
    extras: &[Extra],
    costs: &[ChunkCost],
) -> Result<Option<usize>, EngError> {
    if facts.state_planned != facts.state_allocated {
        return Err(EngError::Checkpoint(format!(
            "prefill admission: the planned state {} B is not the allocated state {} B",
            facts.state_planned, facts.state_allocated
        )));
    }
    let admission = Admission {
        device_budget: facts.budget,
        decode_arena: facts.decode_arena,
        device_extras: extras.to_vec(),
        ..Admission::new(facts.budget)
    };
    admission.admit(facts.state_planned, costs, Some(facts.cap))
}

/// The pre-arena half of the chunk gate: true only when the session could run the chunk
/// path at all — the Native layout, the `Fp16` KV mode, a plain (non-tiered) session and no
/// Q6_K MMQ weight of the chunk (the first four conditions of `chunk_verdict`, in their
/// order). A session that fails any one of them never takes VRAM for a path it cannot run,
/// so `admit_prefill` checks this BEFORE any device fact query and before the arena
/// allocation.
pub(super) fn needs_arena(
    layout: WeightLayout,
    kv_mode: KvMode,
    tiered: bool,
    mmq_q6k: bool,
) -> bool {
    layout == WeightLayout::Native && kv_mode == KvMode::Fp16 && !tiered && !mmq_q6k
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

/// The one reused zero buffer of the arena fill (ruling (c), plan §C.3): about 8 MiB per
/// synchronous `write_at`, 27-93 spans for the 213.6-741 MiB arenas of the ladder
/// (`zero_spans`, ESTIMATE).
pub(super) const ZERO_CHUNK: usize = 8 << 20;

/// The spans that cover `[0, total)` exactly once, each `1..=chunk` bytes long, contiguous
/// from 0: the zero fill of a whole arena (the fixup range included) runs through one reused
/// host buffer of at most [`ZERO_CHUNK`] bytes. Refuses a zero-byte span.
pub(super) fn zero_spans(total: usize, chunk: usize) -> Result<Vec<(usize, usize)>, EngError> {
    if chunk == 0 {
        return Err(EngError::BadRequest(format!(
            "prefill arena: a zero fill of {total} B cannot write zero-byte spans"
        )));
    }
    let mut spans = Vec::new();
    let mut at = 0;
    while at < total {
        let len = chunk.min(total - at);
        spans.push((at, len));
        at += len;
    }
    Ok(spans)
}

/// The prefill arena of one session (plan §C.3, ruling (c); CPF-7b-C B.3): the chunk
/// tensors of the ADMITTED `M_sess` in one device arena, the attention `part` buffer at
/// [`CPF_ATTN_PART_CAP`], and the facts and budget that admitted them. Built and
/// zero-filled ONCE per session by [`PrefillArena::allocate`] through
/// `QwenResidentStep::admit_prefill` — never per request or per chunk (protection 2) — and
/// held by the step for its life; the step's field is declared before the state, so the
/// arena drops first.
pub(super) struct PrefillArena {
    /// `chunk_tensors(dims, m_sess, facts.nsm)`: the tensor list of the admitted size.
    list: Vec<(ChunkSlot, ArenaTensor)>,
    /// The one allocation that holds the chunk tensors.
    arena: DeviceArena,
    /// The attention `part` buffer, at `CPF_ATTN_PART_CAP` and never at `ChunkCost::part`:
    /// a smaller chunk launched with its own split plan can need more than the `M_sess`
    /// plan (part 2 carry item 11), and the flat cap covers both.
    part: DeviceBuffer,
    m_sess: usize,
    /// The device facts of the admission (`cc`, `nsm`, `smpbo`), kept for the chunk builds.
    facts: MmqFacts,
    /// The admission budget, for the run report (CPF-8a §E.2).
    budget: usize,
}

impl PrefillArena {
    /// Builds and zero-fills the arena of `m_sess` rows: the tensor list, ONE device arena
    /// for it, the WHOLE arena (the fixup range included, plan §C.3 ruling (c)) zeroed
    /// through one reused [`ZERO_CHUNK`] host buffer at every [`zero_spans`] span — 27-93
    /// synchronous writes, about 9-78 ms, once per session — and then the `part` buffer at
    /// [`CPF_ATTN_PART_CAP`], which is NOT zero-filled (ruling (c) names the arena; the
    /// attention kernel writes what it reads). `part` is allocated last, so any refusal
    /// drops the whole partly built arena before it returns.
    pub(super) fn allocate(
        ctx: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        m_sess: usize,
        facts: MmqFacts,
        budget: usize,
    ) -> Result<Self, EngError> {
        let list = chunk_tensors(dims, m_sess, facts.nsm)?;
        let tensors: Vec<ArenaTensor> = list.iter().map(|&(_, tensor)| tensor).collect();
        let arena = ctx.arena(&tensors)?;
        #[cfg(test)]
        io::bump_alloc();
        let zeros = vec![0u8; ZERO_CHUNK.min(arena.bytes())];
        for (at, len) in zero_spans(arena.bytes(), ZERO_CHUNK)? {
            arena.buffer().write_at(stream, at, &zeros[..len])?;
            #[cfg(test)]
            io::bump_span_write();
        }
        let part = ctx.allocate(CPF_ATTN_PART_CAP)?;
        #[cfg(test)]
        io::bump_alloc();
        Ok(Self {
            list,
            arena,
            part,
            m_sess,
            facts,
            budget,
        })
    }

    /// The admitted chunk size, one of `M_LADDER` (never the raw cap).
    pub(super) fn m_sess(&self) -> usize {
        self.m_sess
    }

    /// Bytes of the arena allocation.
    pub(super) fn bytes(&self) -> usize {
        self.arena.bytes()
    }

    /// Bytes of the `part` allocation (`CPF_ATTN_PART_CAP`).
    pub(super) fn part_bytes(&self) -> usize {
        self.part.bytes()
    }

    /// The tensor list of the admitted size, for the chunk-parts build (C3a).
    pub(super) fn list(&self) -> &[(ChunkSlot, ArenaTensor)] {
        &self.list
    }

    /// The arena itself, for the chunk views (C3a).
    pub(super) fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    /// The `part` buffer, for the chunk buffers (C3a).
    pub(super) fn part(&self) -> &DeviceBuffer {
        &self.part
    }

    /// The device facts of the admission, for the chunk buffers (C3a).
    pub(super) fn facts(&self) -> MmqFacts {
        self.facts
    }

    /// The admission budget, for the run report (C4b).
    pub(super) fn budget(&self) -> usize {
        self.budget
    }
}

/// Checks the continuation position before the chunk changes device state.
pub(super) fn check_chunk_start(p0: usize, committed: usize) -> Result<(), EngError> {
    if p0 != committed {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: p0 {p0} is not the committed rows {committed}"
        )));
    }
    Ok(())
}

/// Drives the prompt rows of one feed_rows call through the prefill plan (CPF-7b-C B.5):
/// each step gets its own rows, a feed step exactly one. The start position is the committed
/// row count at the call, so a second prefill call or a prefix import continues at its own
/// position. The list that ran is returned.
pub(super) fn drive_chunks(
    start: usize,
    tokens: &[u32],
    m_sess: usize,
    mut run: impl FnMut(PrefillStep, &[u32]) -> Result<(), EngError>,
) -> Result<Vec<PrefillStep>, EngError> {
    let steps = chunk_list(start, tokens.len(), m_sess)?;
    for step in &steps {
        let (at, rows) = match *step {
            PrefillStep::Chunk { p0, rows } => (p0, rows),
            PrefillStep::Feed { p0 } => (p0, 1),
        };
        let offset = at.checked_sub(start).ok_or_else(|| {
            EngError::BadRequest(format!(
                "prefill chunk: step at {at} precedes the start row {start}"
            ))
        })?;
        let end = offset.checked_add(rows).ok_or_else(|| {
            EngError::BadRequest("prefill chunk: the row window overflows".to_owned())
        })?;
        let slice = tokens.get(offset..end).ok_or_else(|| {
            EngError::BadRequest(format!(
                "prefill chunk: the plan names rows {offset}..{end} of {} tokens",
                tokens.len()
            ))
        })?;
        run(*step, slice)?;
    }
    Ok(steps)
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
