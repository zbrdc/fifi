//! Launch-only FFI for the packed Inkling controller workspace.

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest role count of `launch_rows`: the rows travel by value as kernel arguments.
pub(crate) const MAX_ROW_ROLES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum Arithmetic {
    Generic = 0,
    CpuParity = 1,
}

unsafe extern "C" {
    fn inference_cuda_inkling_rebuild_gates(
        workspace: *mut c_void,
        width: c_int,
        roles: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_prepare_controller(
        pairs: *const c_void,
        gamma: *const c_void,
        triples: *mut c_void,
        gates: *mut c_void,
        health: *mut c_void,
        width: c_int,
        roles: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_controller_norm(
        workspace: *mut c_void,
        health: *mut c_void,
        width: c_int,
        roles: c_int,
        eps: f32,
        cpu_parity: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_controller_norm_rows(
        rows: *const *const c_void,
        health: *mut c_void,
        width: c_int,
        roles: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn rebuild_gates(
    stream: &RawStream,
    workspace: RawRange<'_>,
    [width, roles]: [usize; 2],
) -> Result<(), DeviceFailure> {
    if !(1..=4096).contains(&width)
        || !(1..=1536).contains(&roles)
        || workspace.bytes() < roles * (5 * width + 2) * size_of::<f32>()
        || !workspace.offset.is_multiple_of(4)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the aligned range contains disjoint output, raw and scale segments.
    let status = unsafe {
        inference_cuda_inkling_rebuild_gates(
            workspace.address(),
            width as c_int,
            roles as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn prepare(
    stream: &RawStream,
    ranges: [RawRange<'_>; 5],
    ([width, roles], eps): ([usize; 2], f32),
) -> Result<(), DeviceFailure> {
    if !(1..=4096).contains(&width)
        || !(1..=1536).contains(&roles)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid_argument());
    }
    let row = width * size_of::<f32>();
    let bytes = [
        2 * roles * row,
        row,
        3 * roles * row,
        2 * roles * row,
        2 * roles * size_of::<f32>(),
    ];
    let spans = std::array::from_fn::<_, 5, _>(|i| Span::of(&ranges[i], bytes[i]));
    for (i, range) in ranges.iter().enumerate() {
        if range.bytes() != bytes[i]
            || !range.offset.is_multiple_of(4)
            || spans[..i].iter().any(|prev| !prev.apart(&spans[i]))
        {
            return Err(invalid_argument());
        }
    }
    let [pairs, gamma, triples, gates, health] = ranges.map(|range| range.address());
    let mut stage = STAGE_NONE;
    // SAFETY: all five live ranges are aligned, sized, and disjoint.
    let status = unsafe {
        inference_cuda_inkling_prepare_controller(
            pairs,
            gamma,
            triples,
            gates,
            health,
            width as c_int,
            roles as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    workspace: RawRange<'_>,
    health: RawRange<'_>,
    [width, roles]: [usize; 2],
    eps: f32,
    arithmetic: Arithmetic,
) -> Result<(), DeviceFailure> {
    if !(1..=4096).contains(&width)
        || !(1..=1536).contains(&roles)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid_argument());
    }
    let bytes = 6 * roles * width * size_of::<f32>();
    let health_bytes = roles * size_of::<f32>();
    if workspace.bytes() < bytes
        || health.bytes() < health_bytes
        || !workspace.offset.is_multiple_of(4)
        || !health.offset.is_multiple_of(4)
        || !Span::of(&workspace, bytes).apart(&Span::of(&health, health_bytes))
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: both live ranges hold the accessed words and are aligned and disjoint.
    // The device frees allocations only after queued work completes.
    let status = unsafe {
        inference_cuda_inkling_controller_norm(
            workspace.address(),
            health.address(),
            width as c_int,
            roles as c_int,
            eps,
            arithmetic as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Up to 16 controller roles over the caller's rows (the pointers travel as kernel arguments), `[input, message, gamma, gain, forget,
/// out]` per role, `width` floats each: every output row shares no byte with any row of any
/// role, and `health` (`roles` floats) with none of them.
pub(in crate::kernels::cuda) fn launch_rows(
    stream: &RawStream,
    rows: &[RawRange<'_>],
    roles: usize,
    health: RawRange<'_>,
    (width, eps): (usize, f32),
) -> Result<(), DeviceFailure> {
    if !(1..=4096).contains(&width)
        || !(1..=1536).contains(&roles)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid_argument());
    }
    let row = width * size_of::<f32>();
    let health_bytes = roles * size_of::<f32>();
    if rows.len() != 6 * roles || health.bytes() != health_bytes || !health.offset.is_multiple_of(4)
    {
        return Err(invalid_argument());
    }
    let spans = rows
        .iter()
        .map(|range| Span::of(range, row))
        .collect::<Vec<_>>();
    let status = Span::of(&health, health_bytes);
    for (i, range) in rows.iter().enumerate() {
        if range.bytes() != row || !range.offset.is_multiple_of(4) || !status.apart(&spans[i]) {
            return Err(invalid_argument());
        }
        // Output rows (index 5 of a role) share no byte with any row but themselves.
        for (j, other) in spans.iter().enumerate() {
            if (i % 6 == 5 || j % 6 == 5) && i != j && !spans[i].apart(other) {
                return Err(invalid_argument());
            }
        }
    }
    let pointers = rows
        .iter()
        .map(|range| range.address() as *const c_void)
        .collect::<Vec<_>>();
    let mut stage = STAGE_NONE;
    // SAFETY: the rows and the health words are aligned, sized, and every output row shares
    // no byte with any other row.
    let code = unsafe {
        inference_cuda_inkling_controller_norm_rows(
            pointers.as_ptr(),
            health.address(),
            width as c_int,
            roles as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(code, stage)
}
