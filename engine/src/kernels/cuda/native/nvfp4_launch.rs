//! Native boundary for the existing NVFP4 grouped launch; no kernel or ABI changes.
//! Pointer encoding stays here. Context owns the uploaded tables, keeps payloads
//! borrowed, and serializes each flag's reuse until its checked completion.
//! Allocation/free synchronizes before reclaiming any queued allocation, as for
//! the other resident launches. Call only under the context dispatch guard.

use super::resident::{RawRange, RawStream, Span};
use super::{DeviceFailure, RawDevice, STAGE_NONE, WorkspaceRaw, checked, invalid_argument};
use std::ffi::{c_int, c_void};

unsafe extern "C" {
    #[link_name = "inference_cuda_nvfp4_grouped_gemm_launch"]
    #[allow(clippy::too_many_arguments, reason = "existing C ABI")]
    fn launch_raw(
        packed: *const *const c_void,
        packed_bytes: usize,
        scales: *const *const c_void,
        scales_bytes: usize,
        factors: *const f32,
        factors_bytes: usize,
        n: usize,
        rows: usize,
        columns: usize,
        workspace: *const WorkspaceRaw,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

/// Setup only: upload pointers derived from live checked ranges, not user bytes.
pub(in crate::kernels::cuda) fn upload_pointers(
    ranges: &[RawRange<'_>],
) -> Result<RawDevice, DeviceFailure> {
    if ranges.is_empty() {
        return Err(invalid_argument());
    }
    let mut bytes = Vec::new();
    let len = ranges
        .len()
        .checked_mul(size_of::<usize>())
        .ok_or_else(invalid_argument)?;
    bytes
        .try_reserve_exact(len)
        .map_err(|_| invalid_argument())?;
    for range in ranges {
        bytes.extend_from_slice(&(range.address() as usize).to_ne_bytes());
    }
    super::upload(&bytes)
}

/// Canonical byte geometry plus the grouped kernel's signed grid-x bound.
pub(in crate::kernels::cuda) fn geometry(
    n: usize,
    shape: [usize; 2],
) -> Option<(usize, usize, usize, super::Activations)> {
    let (packed, scales, _) = super::nvfp4_matvec_bytes(shape)?;
    let (pointers, io) = super::nvfp4_batch_bytes(n, shape)?;
    if (shape[0] - 1) / 64 + 1 > c_int::MAX as usize {
        return None;
    }
    n.checked_mul(size_of::<f32>())?;
    pointers
        .checked_mul(2)?
        .checked_add(n.checked_mul(size_of::<f32>())?)?;
    Some((packed, scales, pointers, io))
}

pub(in crate::kernels::cuda) struct Launch<'a> {
    pub(in crate::kernels::cuda) packed: RawRange<'a>,
    pub(in crate::kernels::cuda) scales: RawRange<'a>,
    pub(in crate::kernels::cuda) factors: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) shape: [usize; 2],
}

/// Enqueue only: one asynchronous flag clear and the existing grouped kernel.
/// Context proves descriptor provenance/payload bounds before this boundary.
pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    call: &Launch<'_>,
) -> Result<(), DeviceFailure> {
    let (_, _, pointers, io) = geometry(call.n, call.shape).ok_or_else(invalid_argument)?;
    let ranges = [
        call.packed,
        call.scales,
        call.factors,
        call.x,
        call.y,
        call.flag,
    ];
    let needs = [
        pointers,
        pointers,
        call.n * size_of::<f32>(),
        io.x,
        io.y,
        io.xf,
    ];
    let aligns = [align_of::<usize>(), align_of::<usize>(), 4, 4, 4, 4];
    for i in 0..ranges.len() {
        let start = ranges[i].address() as usize;
        if ranges[i].bytes() < needs[i] || !start.is_multiple_of(aligns[i]) {
            return Err(invalid_argument());
        }
        start.checked_add(needs[i]).ok_or_else(invalid_argument)?;
        for j in 0..i {
            if !Span::of(&ranges[i], needs[i]).apart(&Span::of(&ranges[j], needs[j])) {
                return Err(invalid_argument());
            }
        }
    }
    let work = WorkspaceRaw {
        x: call.x.address(),
        x_bytes: io.x,
        xf: call.flag.address(),
        xf_bytes: io.xf,
        y: call.y.address(),
        y_bytes: io.y,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: ranges above are aligned, bounded, pairwise disjoint. Context
    // privately owns immutable descriptor tables built by upload_pointers from
    // checked live weight ranges and retains those owners; it also prevents
    // clearing an unchecked flag. All frees wait for queued use to finish.
    let status = unsafe {
        launch_raw(
            call.packed.address().cast(),
            pointers,
            call.scales.address().cast(),
            pointers,
            call.factors.address().cast(),
            needs[2],
            call.n,
            call.shape[0],
            call.shape[1],
            &work,
            stream.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::geometry;
    #[test]
    fn pair_axis_is_not_weight_rows() {
        let (p, s, table, io) = geometry(22, [65, 32]).unwrap();
        assert_eq!((p, s, table), (1040, 130, 22 * size_of::<usize>()));
        assert_eq!((io.x, io.y, io.xf), (22 * 32 * 4, 22 * 65 * 4, 4));
        assert!(geometry(1, [1, 16]).is_some());
    }
    #[test]
    fn rejects_grid_and_arithmetic_overflow() {
        assert!(geometry(0, [1, 16]).is_none());
        assert!(geometry(1, [1, 15]).is_none());
        assert!(geometry(usize::MAX, [1, 16]).is_none());
        assert!(geometry(1, [usize::MAX, 16]).is_none());
        if usize::BITS == 64 {
            assert!(geometry(1, [(i32::MAX as usize) * 64 + 1, 16]).is_none());
        }
    }
}
