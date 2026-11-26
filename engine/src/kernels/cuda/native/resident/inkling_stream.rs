//! Launch-only FFI for the resident Inkling depth stream
//! (`native/resident/inkling_stream.cu`, contract in `inkling_stream.h`).

use std::ffi::{c_int, c_void};
use std::mem::size_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest `blocks` of a stream state: the mix reads this many sealed rows and the current row.
pub(crate) const MAX_BLOCKS: usize = 63;
const MAX_WIDTH: usize = 8192;

unsafe extern "C" {
    fn inference_cuda_inkling_stream_mix(
        state: *const c_void,
        query: *const c_void,
        gamma: *const c_void,
        out: *mut c_void,
        health: *mut c_void,
        first_block: c_int,
        blocks: c_int,
        width: c_int,
        mix_eps: f32,
        norm_eps: f32,
        rows: c_int,
        stride: c_int,
        out_stride: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_mix_jobs(
        rows: *const *const c_void,
        extents: *const c_int,
        count: c_int,
        health: *mut c_void,
        width: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_seal(
        state: *mut c_void,
        next: *const c_void,
        health: *mut c_void,
        blocks: c_int,
        width: c_int,
        eps: f32,
        rows: c_int,
        stride: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_add(
        state: *mut c_void,
        row: *const c_void,
        health: *mut c_void,
        width: c_int,
        rows: c_int,
        stride: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_norm_rows(
        ranges: *const *const c_void,
        count: c_int,
        width: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The most states one launch takes (one CUDA block each).
pub(crate) const MAX_STATES: usize = 65535;

/// `rows` stream states, each starting `stride` rows of the width after the previous one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct States {
    pub(crate) rows: usize,
    pub(crate) stride: usize,
}

impl States {
    /// One state of `used` rows.
    pub(crate) fn one(used: usize) -> Self {
        Self {
            rows: 1,
            stride: used,
        }
    }

    /// The bytes from the first state's start to the end of the last state's `used` rows, when
    /// the count and stride are valid.
    pub(crate) fn bytes(self, used: usize, width: usize) -> Option<usize> {
        if !(1..=MAX_STATES).contains(&self.rows)
            || self.stride < used
            || c_int::try_from(self.stride).is_err()
        {
            return None;
        }
        ((self.rows - 1) * self.stride)
            .checked_add(used)?
            .checked_mul(width)?
            .checked_mul(size_of::<f32>())
    }
}

fn shape(blocks: usize, width: usize, eps: f32) -> bool {
    blocks <= MAX_BLOCKS && (1..=MAX_WIDTH).contains(&width) && eps.is_finite() && eps > 0.0
}

fn sized(range: &RawRange<'_>, bytes: usize) -> bool {
    range.bytes >= bytes && range.offset.is_multiple_of(4)
}

/// Largest number of jobs of one [`mix_jobs`] launch.
pub(crate) const MAX_MIX_JOBS: usize = 8;

/// One job of [`mix_jobs`]: a state of `blocks + 1` rows, its query and its output row.
pub(in crate::kernels::cuda) struct MixJobRaw<'a> {
    pub(in crate::kernels::cuda) state: RawRange<'a>,
    pub(in crate::kernels::cuda) query: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) first_block: usize,
    pub(in crate::kernels::cuda) blocks: usize,
}

/// [`mix`] of each job in one launch, a block per job; the jobs share `health`. Every output and
/// `health` share no byte with a row any job reads or with each other.
pub(in crate::kernels::cuda) fn mix_jobs(
    stream: &RawStream,
    jobs: &[MixJobRaw<'_>],
    health: RawRange<'_>,
    (width, eps): (usize, f32),
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    if !(1..=MAX_MIX_JOBS).contains(&jobs.len()) || !sized(&health, health_bytes) {
        return Err(invalid_argument());
    }
    let mut reads = Vec::with_capacity(2 * jobs.len());
    let mut writes = vec![Span::of(&health, health_bytes)];
    for job in jobs {
        if !(job.first_block <= job.blocks && shape(job.blocks, width, eps)) {
            return Err(invalid_argument());
        }
        let state_bytes = (job.blocks + 1) * row;
        if !(sized(&job.state, state_bytes) && sized(&job.query, row) && sized(&job.out, row)) {
            return Err(invalid_argument());
        }
        let (state, query) = (Span::of(&job.state, state_bytes), Span::of(&job.query, row));
        if !state.apart(&query) {
            return Err(invalid_argument());
        }
        reads.extend([state, query]);
        writes.push(Span::of(&job.out, row));
    }
    let apart = writes.iter().enumerate().all(|(at, write)| {
        reads.iter().all(|read| write.apart(read))
            && writes[at + 1..].iter().all(|other| write.apart(other))
    });
    if !apart {
        return Err(invalid_argument());
    }
    let rows: Vec<*const c_void> = jobs
        .iter()
        .flat_map(|job| {
            [
                job.state.address() as *const c_void,
                job.query.address() as *const c_void,
                job.out.address() as *const c_void,
            ]
        })
        .collect();
    let extents: Vec<c_int> = jobs
        .iter()
        .flat_map(|job| [job.first_block as c_int, job.blocks as c_int])
        .collect();
    let mut stage = STAGE_NONE;
    // SAFETY: each job's ranges are aligned and hold the extents above; the outputs and the
    // health word share no byte with a read row or with each other. The C entry copies both
    // lists into the kernel's parameters before it returns.
    let status = unsafe {
        inference_cuda_inkling_stream_mix_jobs(
            rows.as_ptr(),
            extents.as_ptr(),
            jobs.len() as c_int,
            health.address(),
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Ranges of a mix: `state` holds the current row and `blocks` sealed rows of each state; `out`
/// takes `Stream::mix(first_block, query)` of each. `query` absent takes the raw current row,
/// `gamma` present normalizes the mix and scales it (`CpuForward::norm`).
pub(in crate::kernels::cuda) struct MixRaw<'a> {
    pub(in crate::kernels::cuda) state: RawRange<'a>,
    pub(in crate::kernels::cuda) query: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) gamma: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) health: RawRange<'a>,
}

/// The mix of each of `states` (see [`MixRaw`]), output rows `out_stride` floats apart.
pub(in crate::kernels::cuda) fn mix(
    stream: &RawStream,
    call: MixRaw<'_>,
    [first_block, blocks, width, out_stride]: [usize; 4],
    [mix_eps, norm_eps]: [f32; 2],
    states: States,
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let state_bytes = states
        .bytes(blocks + 1, width)
        .ok_or_else(invalid_argument)?;
    let out_bytes = ((states.rows - 1) * out_stride + width) * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    let valid = first_block <= blocks
        && out_stride >= width
        && c_int::try_from(out_stride).is_ok()
        && shape(
            blocks,
            width,
            if call.query.is_some() { mix_eps } else { 1.0 },
        )
        && shape(
            blocks,
            width,
            if call.gamma.is_some() { norm_eps } else { 1.0 },
        )
        && sized(&call.state, state_bytes)
        && call.query.as_ref().is_none_or(|q| sized(q, row))
        && call.gamma.as_ref().is_none_or(|g| sized(g, row))
        && sized(&call.out, out_bytes)
        && sized(&call.health, health_bytes);
    if !valid {
        return Err(invalid_argument());
    }
    let out = Span::of(&call.out, out_bytes);
    let health = Span::of(&call.health, health_bytes);
    let mut reads = vec![Span::of(&call.state, state_bytes)];
    reads.extend(
        call.query
            .iter()
            .chain(&call.gamma)
            .map(|r| Span::of(r, row)),
    );
    if !out.apart(&health) || reads.iter().any(|r| !out.apart(r) || !health.apart(r)) {
        return Err(invalid_argument());
    }
    let address = |r: &Option<RawRange<'_>>| {
        r.as_ref()
            .map_or(std::ptr::null(), |r| r.address() as *const c_void)
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and the outputs share no byte with
    // any other range; absent operands are null, which the entry accepts.
    let status = unsafe {
        inference_cuda_inkling_stream_mix(
            call.state.address() as *const c_void,
            address(&call.query),
            address(&call.gamma),
            call.out.address(),
            call.health.address(),
            first_block as c_int,
            blocks as c_int,
            width as c_int,
            mix_eps,
            norm_eps,
            states.rows as c_int,
            states.stride as c_int,
            out_stride as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `sealed[blocks] = current * rms_scale(current)`, then `current = next` (`Stream::seal(next)`).
/// `state` must hold the new sealed row: `blocks + 2` rows.
pub(in crate::kernels::cuda) fn seal(
    stream: &RawStream,
    state: RawRange<'_>,
    next: RawRange<'_>,
    health: RawRange<'_>,
    [blocks, width]: [usize; 2],
    eps: f32,
    states: States,
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let state_bytes = states
        .bytes(blocks + 2, width)
        .ok_or_else(invalid_argument)?;
    let next_bytes = states.rows * row;
    let health_bytes = size_of::<f32>();
    if !shape(blocks, width, eps)
        || !sized(&state, state_bytes)
        || !sized(&next, next_bytes)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (state_span, next_span, health_span) = (
        Span::of(&state, state_bytes),
        Span::of(&next, next_bytes),
        Span::of(&health, health_bytes),
    );
    if !state_span.apart(&next_span)
        || !health_span.apart(&state_span)
        || !health_span.apart(&next_span)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_seal(
            state.address(),
            next.address() as *const c_void,
            health.address(),
            blocks as c_int,
            width as c_int,
            eps,
            states.rows as c_int,
            states.stride as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `current = current + row` (`Stream::add_current(row)`).
pub(in crate::kernels::cuda) fn add(
    stream: &RawStream,
    state: RawRange<'_>,
    row: RawRange<'_>,
    health: RawRange<'_>,
    width: usize,
    states: States,
) -> Result<(), DeviceFailure> {
    let state_bytes = states.bytes(1, width).ok_or_else(invalid_argument)?;
    let row_bytes = states.rows * width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    if !shape(0, width, 1.0)
        || !sized(&state, state_bytes)
        || !sized(&row, row_bytes)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (state_span, row_span, health_span) = (
        Span::of(&state, state_bytes),
        Span::of(&row, row_bytes),
        Span::of(&health, health_bytes),
    );
    if !state_span.apart(&row_span)
        || !health_span.apart(&state_span)
        || !health_span.apart(&row_span)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_add(
            state.address(),
            row.address() as *const c_void,
            health.address(),
            width as c_int,
            states.rows as c_int,
            states.stride as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Largest number of rows of one [`norm_rows`] launch.
pub(crate) const MAX_NORM_ROWS: usize = 8;

/// One row of [`norm_rows`]: `dst = src * rms_scale(src)`, health into `health`.
pub(in crate::kernels::cuda) struct NormRowRaw<'a> {
    pub(in crate::kernels::cuda) src: RawRange<'a>,
    pub(in crate::kernels::cuda) dst: RawRange<'a>,
    pub(in crate::kernels::cuda) health: RawRange<'a>,
}

/// `dst = src * rms_scale(src)` of each row in one launch, a block per row. Every output shares no byte with a
/// source or another output, and every health word none with a source or an output; rows may
/// share one health word.
pub(in crate::kernels::cuda) fn norm_rows(
    stream: &RawStream,
    rows: &[NormRowRaw<'_>],
    width: usize,
    eps: f32,
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    let sized_rows = rows
        .iter()
        .all(|r| sized(&r.src, row) && sized(&r.dst, row) && sized(&r.health, health_bytes));
    if !(1..=MAX_NORM_ROWS).contains(&rows.len()) || !shape(0, width, eps) || !sized_rows {
        return Err(invalid_argument());
    }
    let spans: Vec<[Span<'_>; 3]> = rows
        .iter()
        .map(|r| {
            [
                Span::of(&r.src, row),
                Span::of(&r.dst, row),
                Span::of(&r.health, health_bytes),
            ]
        })
        .collect();
    let apart = spans.iter().enumerate().all(|(at, [_, dst, health])| {
        spans
            .iter()
            .enumerate()
            .all(|(other_at, [src, other_dst, _])| {
                dst.apart(src)
                    && (at == other_at || dst.apart(other_dst))
                    && health.apart(src)
                    && health.apart(other_dst)
            })
    });
    if !apart {
        return Err(invalid_argument());
    }
    let ranges: Vec<*const c_void> = rows
        .iter()
        .flat_map(|r| {
            [
                r.src.address() as *const c_void,
                r.dst.address() as *const c_void,
                r.health.address() as *const c_void,
            ]
        })
        .collect();
    let mut stage = STAGE_NONE;
    // SAFETY: every range is aligned and holds its extent; the outputs and health words share no
    // byte with a source or an output. The C entry copies the list into the kernel's parameters
    // before it returns.
    let status = unsafe {
        inference_cuda_inkling_stream_norm_rows(
            ranges.as_ptr(),
            rows.len() as c_int,
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
