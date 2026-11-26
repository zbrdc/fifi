//! Launch-only FFI for the resident Inkling depth stream
//! (`native/resident/inkling_stream.cu`, contract in `inkling_stream.h`).

use std::ffi::{c_int, c_void};
use std::mem::size_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest `blocks` of a stream state: the mix reads this many sealed rows and the current row.
pub(crate) const MAX_BLOCKS: usize = 63;
const MAX_WIDTH: usize = 8192;

unsafe extern "C" {
    fn inference_cuda_inkling_stream_mix_jobs(
        ranges: *const *const c_void,
        shape: *const c_int,
        norm_eps: *const f32,
        count: c_int,
        width: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_mix(
        state: *const c_void,
        query: *const c_void,
        out: *mut c_void,
        health: *mut c_void,
        first_block: c_int,
        blocks: c_int,
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

    fn inference_cuda_inkling_zero_words(
        rows: *const *const c_void,
        words: *const c_int,
        count: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_copy_words(
        source: *const *const c_void,
        destination: *const *mut c_void,
        words: *const c_int,
        count: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_add(
        state: *mut c_void,
        row: *const c_void,
        health: *mut c_void,
        width: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

}

fn shape(blocks: usize, width: usize, eps: f32) -> bool {
    blocks <= MAX_BLOCKS && (1..=MAX_WIDTH).contains(&width) && eps.is_finite() && eps > 0.0
}

fn sized(range: &RawRange<'_>, bytes: usize) -> bool {
    range.bytes >= bytes && range.offset.is_multiple_of(4)
}

/// Every range of one mix with the bytes the launch reads or writes.
type MixSpans<'a> = [(Span<'a>, usize); 4];

/// The spans of one valid mix: `(state, query, out, health)`, in that order.
fn mix_spans<'a>(
    ranges: &[RawRange<'a>; 4],
    [first_block, blocks, width]: [usize; 3],
    eps: f32,
) -> Result<MixSpans<'a>, DeviceFailure> {
    let row = width * size_of::<f32>();
    let state_bytes = (blocks + 1) * row;
    let health_bytes = size_of::<f32>();
    let valid = first_block <= blocks
        && shape(blocks, width, eps)
        && sized(&ranges[0], state_bytes)
        && sized(&ranges[1], row)
        && sized(&ranges[2], row)
        && sized(&ranges[3], health_bytes);
    if !valid {
        return Err(invalid_argument());
    }
    let spans: MixSpans<'_> = [
        (Span::of(&ranges[0], state_bytes), state_bytes),
        (Span::of(&ranges[1], row), row),
        (Span::of(&ranges[2], row), row),
        (Span::of(&ranges[3], health_bytes), health_bytes),
    ];
    let [state, query, out, health] = &spans;
    if !out.0.apart(&state.0)
        || !out.0.apart(&query.0)
        || !health.0.apart(&state.0)
        || !health.0.apart(&query.0)
        || !health.0.apart(&out.0)
        || !state.0.apart(&query.0)
    {
        return Err(invalid_argument());
    }
    Ok(spans)
}

/// `state` holds the current row and `blocks` sealed rows; `out` takes the weighted mix of
/// `Stream::mix(first_block, query)`.
pub(in crate::kernels::cuda) fn mix(
    stream: &RawStream,
    ranges: [RawRange<'_>; 4],
    [first_block, blocks, width]: [usize; 3],
    eps: f32,
) -> Result<(), DeviceFailure> {
    mix_spans(&ranges, [first_block, blocks, width], eps)?;
    let pointers = ranges.map(|r| r.address());
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_mix(
            pointers[0] as *const c_void,
            pointers[1] as *const c_void,
            pointers[2],
            pointers[3],
            first_block as c_int,
            blocks as c_int,
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Largest number of mixes of one `mix_jobs` launch.
pub(in crate::kernels::cuda) const MAX_MIX_JOBS: usize = 8;

/// One mix of a `mix_jobs` launch: `(state, query, out, health)` with `[first_block, blocks]`,
/// an optional addend row and an optional norm epilogue `(gamma, normed, eps)`.
pub(in crate::kernels::cuda) struct MixJobRaw<'a> {
    pub(in crate::kernels::cuda) ranges: [RawRange<'a>; 4],
    pub(in crate::kernels::cuda) addend: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) norm: Option<([RawRange<'a>; 2], f32)>,
    pub(in crate::kernels::cuda) shape: [usize; 2],
}

/// The spans of one checked job.
struct JobSpans<'a> {
    state: Span<'a>,
    query: Span<'a>,
    out: Span<'a>,
    health: Span<'a>,
    addend: Option<Span<'a>>,
    norm: Option<[Span<'a>; 2]>,
}

fn copy_span<'a>(s: &Span<'a>) -> Span<'a> {
    Span {
        device: s.device,
        start: s.start,
        len: s.len,
    }
}

impl JobSpans<'_> {
    /// The spans the job writes, each with whether it is the current row of its state: the
    /// output, the health word, the normalized row and (with an addend) that current row.
    fn writes(&self) -> Vec<(Span<'_>, bool)> {
        let mut writes = vec![
            (copy_span(&self.out), false),
            (copy_span(&self.health), false),
        ];
        if let Some(n) = &self.norm {
            writes.push((copy_span(&n[1]), false));
        }
        if self.addend.is_some() {
            writes.push((
                Span {
                    device: self.state.device,
                    start: self.state.start,
                    len: self.query.len,
                },
                true,
            ));
        }
        writes
    }

    /// Every span the job reads: the state, the query, the addend and the gamma.
    fn reads(&self) -> Vec<Span<'_>> {
        [
            Some(copy_span(&self.state)),
            Some(copy_span(&self.query)),
            self.addend.as_ref().map(copy_span),
            self.norm.as_ref().map(|n| copy_span(&n[0])),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

fn same_span(a: &Span<'_>, b: &Span<'_>) -> bool {
    std::ptr::eq(a.device, b.device) && a.start == b.start && a.len == b.len
}

/// Independent mixes over one `width` in one launch, one block each. Each job passes the checks
/// of [`mix`]; its optional addend row (added into the current row of its state first) and its
/// optional norm epilogue (gamma read, normed row written) hold the row and share no byte with
/// the job. No span a job writes shares a byte with any span of another job, unless two jobs
/// name the same health word (they would store the same NaN).
pub(in crate::kernels::cuda) fn mix_jobs(
    stream: &RawStream,
    jobs: &[MixJobRaw<'_>],
    (width, eps): (usize, f32),
) -> Result<(), DeviceFailure> {
    if jobs.is_empty() || jobs.len() > MAX_MIX_JOBS {
        return Err(invalid_argument());
    }
    let row = width * size_of::<f32>();
    let mut spans = Vec::with_capacity(jobs.len());
    for job in jobs {
        let [first_block, blocks] = job.shape;
        let [state, query, out, health] =
            mix_spans(&job.ranges, [first_block, blocks, width], eps)?;
        let addend = job.addend.map(|r| Span::of(&r, row));
        let norm = job
            .norm
            .map(|(ranges, eps)| {
                if !eps.is_finite() || eps <= 0.0 {
                    return Err(invalid_argument());
                }
                Ok([Span::of(&ranges[0], row), Span::of(&ranges[1], row)])
            })
            .transpose()?;
        let own = JobSpans {
            state: state.0,
            query: query.0,
            out: out.0,
            health: health.0,
            addend,
            norm,
        };
        let writes = own.writes();
        let reads = own.reads();
        for (i, (w, _)) in writes.iter().enumerate() {
            for (r, _) in &writes[i + 1..] {
                if !w.apart(r) {
                    return Err(invalid_argument());
                }
            }
        }
        for (w, current) in &writes {
            for r in &reads {
                // The current row a job adds into is part of the state it reads.
                if *current && same_span(r, &own.state) {
                    continue;
                }
                if !w.apart(r) {
                    return Err(invalid_argument());
                }
            }
        }
        spans.push(own);
    }
    for (i, a) in spans.iter().enumerate() {
        for b in &spans[i + 1..] {
            let shared_health = same_span(&a.health, &b.health);
            let a_writes = a.writes();
            let b_writes = b.writes();
            for (x, x_writes, x_health) in [(a, &a_writes, &a.health), (b, &b_writes, &b.health)] {
                let (y, y_writes, y_health) = if std::ptr::eq(x, a) {
                    (b, &b_writes, &b.health)
                } else {
                    (a, &a_writes, &a.health)
                };
                for (w, _) in x_writes {
                    for r in y.reads().iter().chain(y_writes.iter().map(|(r, _)| r)) {
                        if shared_health && same_span(w, x_health) && same_span(r, y_health) {
                            continue;
                        }
                        if !w.apart(r) {
                            return Err(invalid_argument());
                        }
                    }
                }
            }
        }
    }
    let mut pointers = [std::ptr::null::<c_void>(); 7 * MAX_MIX_JOBS];
    let mut shape = [0; 2 * MAX_MIX_JOBS];
    let mut norm_eps = [0.0f32; MAX_MIX_JOBS];
    let mut any_norm = false;
    for (i, job) in jobs.iter().enumerate() {
        for (slot, range) in pointers[7 * i..7 * i + 4].iter_mut().zip(&job.ranges) {
            *slot = range.address() as *const c_void;
        }
        if let Some(a) = &job.addend {
            pointers[7 * i + 4] = a.address() as *const c_void;
        }
        if let Some((ranges, eps)) = &job.norm {
            pointers[7 * i + 5] = ranges[0].address() as *const c_void;
            pointers[7 * i + 6] = ranges[1].address() as *const c_void;
            norm_eps[i] = *eps;
            any_norm = true;
        }
        shape[2 * i] = job.shape[0] as c_int;
        shape[2 * i + 1] = job.shape[1] as c_int;
    }
    let mut stage = STAGE_NONE;
    // SAFETY: every job passed the checks of a single mix and no two jobs share a written byte.
    let status = unsafe {
        inference_cuda_inkling_stream_mix_jobs(
            pointers.as_ptr(),
            shape.as_ptr(),
            if any_norm {
                norm_eps.as_ptr()
            } else {
                std::ptr::null()
            },
            jobs.len() as c_int,
            width as c_int,
            eps,
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
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let state_bytes = (blocks + 2) * row;
    let health_bytes = size_of::<f32>();
    if !shape(blocks, width, eps)
        || !sized(&state, state_bytes)
        || !sized(&next, row)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (state_span, next_span, health_span) = (
        Span::of(&state, state_bytes),
        Span::of(&next, row),
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
) -> Result<(), DeviceFailure> {
    let row_bytes = width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    if !shape(0, width, 1.0)
        || !sized(&state, row_bytes)
        || !sized(&row, row_bytes)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (state_span, row_span, health_span) = (
        Span::of(&state, row_bytes),
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
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Largest number of rows of one `norm_rows` launch.
pub(in crate::kernels::cuda) const MAX_NORM_ROWS: usize = 8;

/// `dst = src * rms_scale(src)` rows in one launch, one block per row: every row holds
/// `(src, dst, health)` and shares `width` and `eps`. A row's output and health word share no
/// byte with its source; no row's output or health word shares a byte with another row's
/// ranges, unless rows name the same health word (they would store the same NaN).
pub(in crate::kernels::cuda) fn norm_rows(
    stream: &RawStream,
    rows: &[[RawRange<'_>; 3]],
    (width, eps): (usize, f32),
) -> Result<(), DeviceFailure> {
    if rows.is_empty() || rows.len() > MAX_NORM_ROWS {
        return Err(invalid_argument());
    }
    let row = width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    let spans: Vec<[(Span<'_>, usize); 3]> = rows
        .iter()
        .map(|ranges| {
            let sized_ok = shape(0, width, eps)
                && sized(&ranges[0], row)
                && sized(&ranges[1], row)
                && sized(&ranges[2], health_bytes);
            if !sized_ok {
                return Err(invalid_argument());
            }
            let spans = [
                (Span::of(&ranges[0], row), row),
                (Span::of(&ranges[1], row), row),
                (Span::of(&ranges[2], health_bytes), health_bytes),
            ];
            let [src, dst, health] = &spans;
            if !dst.0.apart(&src.0) || !health.0.apart(&src.0) || !health.0.apart(&dst.0) {
                return Err(invalid_argument());
            }
            Ok(spans)
        })
        .collect::<Result<_, _>>()?;
    for (r, [src, dst, health]) in spans.iter().enumerate() {
        for [other_src, other_dst, other_health] in &spans[r + 1..] {
            let same_health = std::ptr::eq(health.0.device, other_health.0.device)
                && health.0.start == other_health.0.start
                && health.0.len == other_health.0.len;
            let health_apart = same_health
                || (dst.0.apart(&other_health.0)
                    && other_dst.0.apart(&health.0)
                    && health.0.apart(&other_health.0)
                    && health.0.apart(&other_src.0)
                    && other_health.0.apart(&src.0));
            if !(dst.0.apart(&other_src.0)
                && other_dst.0.apart(&src.0)
                && dst.0.apart(&other_dst.0)
                && health_apart)
            {
                return Err(invalid_argument());
            }
        }
    }
    let mut pointers = [std::ptr::null::<c_void>(); 3 * MAX_NORM_ROWS];
    for (slot, range) in pointers[..3 * rows.len()]
        .iter_mut()
        .zip(rows.iter().flatten())
    {
        *slot = range.address() as *const c_void;
    }
    let mut stage = STAGE_NONE;
    // SAFETY: every row passed the checks of a single norm and no two rows share a written byte.
    let status = unsafe {
        inference_cuda_inkling_stream_norm_rows(
            pointers.as_ptr(),
            rows.len() as c_int,
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Largest number of rows of one `zero_words` or `copy_words` launch.
pub(in crate::kernels::cuda) const MAX_WORD_ROWS: usize = 64;

fn word_rows(ranges: &[RawRange<'_>]) -> Result<Vec<c_int>, DeviceFailure> {
    if ranges.is_empty() || ranges.len() > MAX_WORD_ROWS {
        return Err(invalid_argument());
    }
    ranges
        .iter()
        .map(|range| {
            (range.bytes.is_multiple_of(4) && range.bytes > 0)
                .then_some((range.bytes / 4) as c_int)
                .ok_or_else(invalid_argument)
        })
        .collect()
}

/// Zero every row of `ranges` in one launch, one block per row.
pub(in crate::kernels::cuda) fn zero_words(
    stream: &RawStream,
    ranges: &[RawRange<'_>],
) -> Result<(), DeviceFailure> {
    let words = word_rows(ranges)?;
    let pointers: Vec<*const c_void> = ranges
        .iter()
        .map(|r| r.address() as *const c_void)
        .collect();
    let mut stage = STAGE_NONE;
    // SAFETY: every range is aligned to the word and holds its bytes.
    let status = unsafe {
        inference_cuda_inkling_zero_words(
            pointers.as_ptr(),
            words.as_ptr(),
            ranges.len() as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Copy every `(source, destination)` pair of equal size in one launch, one block per pair.
pub(in crate::kernels::cuda) fn copy_words(
    stream: &RawStream,
    pairs: &[(RawRange<'_>, RawRange<'_>)],
) -> Result<(), DeviceFailure> {
    let sources: Vec<RawRange<'_>> = pairs.iter().map(|(a, _)| *a).collect();
    let words = word_rows(&sources)?;
    let destinations: Vec<RawRange<'_>> = pairs.iter().map(|(_, b)| *b).collect();
    if word_rows(&destinations)? != words {
        return Err(invalid_argument());
    }
    for (source, destination) in pairs {
        if !Span::of(source, source.bytes).apart(&Span::of(destination, destination.bytes)) {
            return Err(invalid_argument());
        }
    }
    let sources: Vec<*const c_void> = sources
        .iter()
        .map(|r| r.address() as *const c_void)
        .collect();
    let destinations: Vec<*mut c_void> = destinations.iter().map(|r| r.address()).collect();
    let mut stage = STAGE_NONE;
    // SAFETY: every range is aligned to the word, holds its bytes and shares none with its pair.
    let status = unsafe {
        inference_cuda_inkling_copy_words(
            sources.as_ptr(),
            destinations.as_ptr(),
            words.as_ptr(),
            pairs.len() as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
