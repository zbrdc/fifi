//! Preflight of the page-local control; the KV8 composite queues both checked entries.
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange,
        context::{Claim, require_apart, require_fit},
        native::resident::paged_position::{PositionRaw, geometry_ok},
    },
};

pub(crate) struct PagedPositionCall<'a> {
    pub(crate) params: DeviceRange<'a>,
    pub(crate) local: DeviceRange<'a>,
    pub(crate) expected: usize,
    pub(crate) base: usize,
    pub(crate) capacity: usize,
}

pub(super) fn raw_call<'a>(
    context: &CudaContext,
    call: &PagedPositionCall<'a>,
    cells: usize,
) -> Result<PositionRaw<'a>, EngError> {
    let bad = || EngError::BadRequest("paged append position or geometry is invalid".to_owned());
    let expected = i32::try_from(call.expected).map_err(|_| bad())?;
    let base = i32::try_from(call.base).map_err(|_| bad())?;
    let cells = i32::try_from(cells).map_err(|_| bad())?;
    let capacity = i32::try_from(call.capacity).map_err(|_| bad())?;
    if !geometry_ok(expected, base, cells, capacity) {
        return Err(bad());
    }
    require_fit("paged params", &call.params, 32, 4)?;
    require_fit("paged local", &call.local, 4, 4)?;
    require_apart(
        &Claim::of("paged params", &call.params, 32),
        &Claim::of("paged local", &call.local, 4),
    )?;
    Ok(PositionRaw {
        params: context.raw_range(&call.params)?,
        local: context.raw_range(&call.local)?,
        expected,
        base,
        cells,
        capacity,
    })
}
