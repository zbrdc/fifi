//! The CUDA arena-fill and chunk-driver primitives of the chunked prefill, shared by every family
//! that arenas prompt chunks (promoted from the first chunked family).
//!
//! [`zero_spans`] and [`ZERO_CHUNK`] are the one reused zero buffer of the arena fill;
//! [`allocate_zeroed`] is the fill itself — one arena, a whole-arena scribble when the golden
//! capture arm asks for it, then one zero fill. The family's test-only I/O counters ride along as
//! [`ArenaHooks`] fn pointers, so no family name travels through here. [`PrefillStep`] and
//! [`drive_steps`] are the family-neutral chunk walk: the family builds its own step list, this
//! drives it.

use crate::{
    EngError,
    kernels::cuda::{ArenaTensor, CudaContext, DeviceArena, DeviceStream},
};

/// The one reused zero buffer of the arena fill: about 8 MiB per synchronous `write_at`, 27-93
/// spans for the 213.6-741 MiB arenas of the promoting ladder (`zero_spans`, ESTIMATE).
pub(crate) const ZERO_CHUNK: usize = 8 << 20;

/// The spans that cover `[0, total)` exactly once, each `1..=chunk` bytes long, contiguous
/// from 0: the zero fill of a whole arena (the fixup range included) runs through one reused
/// host buffer of at most [`ZERO_CHUNK`] bytes. Refuses a zero-byte span.
pub(crate) fn zero_spans(total: usize, chunk: usize) -> Result<Vec<(usize, usize)>, EngError> {
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

/// The family's test-only I/O counters of the fill: fn pointers so the shared fill cannot name a
/// family module. `NONE` in production; the family binds its own counters in test builds.
#[derive(Clone, Copy, Default)]
pub(crate) struct ArenaHooks {
    pub(crate) on_alloc: Option<fn()>,
    pub(crate) on_scribble: Option<fn()>,
    pub(crate) on_span_write: Option<fn()>,
}

/// Builds ONE device arena for `tensors` and zero-fills the WHOLE arena (the fixup range included)
/// through one reused [`ZERO_CHUNK`] host buffer at every [`zero_spans`] span — 27-93 synchronous
/// writes, about 9-78 ms, once per session. `scribble` (the golden capture arm) writes a 0xA5
/// pattern over the whole fresh allocation first, so a span `zero_spans` misses cannot hide behind
/// the allocator.
pub(crate) fn allocate_zeroed(
    ctx: &CudaContext,
    stream: &DeviceStream,
    tensors: &[ArenaTensor],
    scribble: bool,
    hooks: ArenaHooks,
) -> Result<DeviceArena, EngError> {
    let arena = ctx.arena(tensors)?;
    if let Some(on_alloc) = hooks.on_alloc {
        on_alloc();
    }
    if scribble {
        let pattern = vec![0xA5u8; ZERO_CHUNK.min(arena.bytes())];
        let mut at = 0;
        while at < arena.bytes() {
            let len = ZERO_CHUNK.min(arena.bytes() - at);
            arena.buffer().write_at(stream, at, &pattern[..len])?;
            if let Some(on_scribble) = hooks.on_scribble {
                on_scribble();
            }
            at += len;
        }
    }
    let zeros = vec![0u8; ZERO_CHUNK.min(arena.bytes())];
    for (at, len) in zero_spans(arena.bytes(), ZERO_CHUNK)? {
        arena.buffer().write_at(stream, at, &zeros[..len])?;
        if let Some(on_span_write) = hooks.on_span_write {
            on_span_write();
        }
    }
    Ok(arena)
}

/// Checks the continuation position before the chunk changes device state.
pub(crate) fn check_chunk_start(p0: usize, committed: usize) -> Result<(), EngError> {
    if p0 != committed {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: p0 {p0} is not the committed rows {committed}"
        )));
    }
    Ok(())
}

/// One step of the family's chunk list: `rows >= 2` rows from position `p0` through the chunk
/// path, or a run of exactly one row at `p0` through the per-token path, so a 1-row tail never
/// needs a chunk plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillStep {
    Chunk { p0: usize, rows: usize },
    Feed { p0: usize },
}

/// Drives `steps` over the prompt rows of one feed_rows call: each step gets its own rows, a feed
/// step exactly one. The list is the family's own; this walk only slices `tokens` at each step
/// and runs it. The list that ran is returned.
pub(crate) fn drive_steps(
    start: usize,
    tokens: &[u32],
    steps: Vec<PrefillStep>,
    mut run: impl FnMut(PrefillStep, &[u32]) -> Result<(), EngError>,
) -> Result<Vec<PrefillStep>, EngError> {
    for &step in &steps {
        let (at, rows) = match step {
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
        run(step, slice)?;
    }
    Ok(steps)
}
