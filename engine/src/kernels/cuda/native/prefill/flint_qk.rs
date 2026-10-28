//! Private D128 Q/K FFI. Checked live ranges enter through launch_on;
//! C repeats geometry, alignment and overlap checks before touching the stream.

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

pub(in crate::kernels::cuda) fn qk_geometry(
    rows: usize,
    position: Option<u32>,
) -> Option<[usize; 2]> {
    if !(1..=4096).contains(&rows) {
        return None;
    }
    if let Some(position) = position {
        position.checked_add(u32::try_from(rows - 1).ok()?)?;
    }
    Some([
        rows.checked_mul(32 * 128 * 4)?,
        rows.checked_mul(4 * 128 * 4)?,
    ])
}

#[repr(C)]
struct QkArgs {
    q: *const f32,
    q_bytes: usize,
    k: *const f32,
    k_bytes: usize,
    gamma_q: *const f32,
    gamma_q_bytes: usize,
    gamma_k: *const f32,
    gamma_k_bytes: usize,
    out_q: *mut f32,
    out_q_bytes: usize,
    out_k: *mut f32,
    out_k_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    position: u32,
    reserved: u32,
}

const _: () = assert!(
    size_of::<QkArgs>() == 128
        && offset_of!(QkArgs, q) == 0
        && offset_of!(QkArgs, k) == 16
        && offset_of!(QkArgs, gamma_q) == 32
        && offset_of!(QkArgs, gamma_k) == 48
        && offset_of!(QkArgs, out_q) == 64
        && offset_of!(QkArgs, out_k) == 80
        && offset_of!(QkArgs, flag) == 96
        && offset_of!(QkArgs, rows) == 112
        && offset_of!(QkArgs, position) == 120
        && offset_of!(QkArgs, reserved) == 124
);

unsafe extern "C" {
    fn inference_cuda_flint_qk_norm(
        args: *const QkArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_flint_qk_rope(
        args: *const QkArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct QkRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) gamma: [RawRange<'a>; 2],
    pub(in crate::kernels::cuda) out: [RawRange<'a>; 2],
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) position: u32,
}

fn arguments(call: &QkRaw<'_>) -> Result<QkArgs, DeviceFailure> {
    let [q_bytes, k_bytes] =
        qk_geometry(call.rows, Some(call.position)).ok_or_else(invalid_argument)?;
    let regions = [
        (&call.q, q_bytes, false),
        (&call.k, k_bytes, false),
        (&call.gamma[0], 512, false),
        (&call.gamma[1], 512, false),
        (&call.out[0], q_bytes, true),
        (&call.out[1], k_bytes, true),
        (&call.flag, 4, true),
    ];
    for (index, &(range, bytes, written)) in regions.iter().enumerate() {
        if range.bytes() < bytes || !range.offset().is_multiple_of(4) {
            return Err(invalid_argument());
        }
        for &(other, needed, other_written) in &regions[..index] {
            if (written || other_written) && !Span::of(range, bytes).apart(&Span::of(other, needed))
            {
                return Err(invalid_argument());
            }
        }
    }
    Ok(QkArgs {
        q: call.q.address().cast::<f32>().cast_const(),
        q_bytes: call.q.bytes(),
        k: call.k.address().cast::<f32>().cast_const(),
        k_bytes: call.k.bytes(),
        gamma_q: call.gamma[0].address().cast::<f32>().cast_const(),
        gamma_q_bytes: call.gamma[0].bytes(),
        gamma_k: call.gamma[1].address().cast::<f32>().cast_const(),
        gamma_k_bytes: call.gamma[1].bytes(),
        out_q: call.out[0].address().cast(),
        out_q_bytes: call.out[0].bytes(),
        out_k: call.out[1].address().cast(),
        out_k_bytes: call.out[1].bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        position: call.position,
        reserved: 0,
    })
}

pub(in crate::kernels::cuda) fn qk_launch(
    stream: &RawStream,
    call: &QkRaw<'_>,
    rotary: bool,
) -> Result<(), DeviceFailure> {
    let args = arguments(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: live ranges, checked geometry/alignment and nonoverlapping writes;
    // launch_on supplies the owned explicit stream and holds the dispatch guard.
    let status = unsafe {
        if rotary {
            inference_cuda_flint_qk_rope(&args, stream.as_ptr(), &mut stage)
        } else {
            inference_cuda_flint_qk_norm(&args, stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::qk_geometry;

    #[test]
    fn d128_qk_four_rows_are_admitted_without_rope() {
        assert_eq!(qk_geometry(4, None), Some([65_536, 8_192]));
    }

    #[test]
    fn d128_qk_row_and_position_boundaries() {
        assert_eq!(qk_geometry(4096, None), Some([67_108_864, 8_388_608]));
        for rows in [0, 4097, usize::MAX] {
            assert_eq!(qk_geometry(rows, None), None);
        }
        assert!(qk_geometry(1, Some(u32::MAX)).is_some());
        assert!(qk_geometry(4096, Some(u32::MAX - 4095)).is_some());
        assert_eq!(qk_geometry(4096, Some(u32::MAX - 4094)), None);
        assert_eq!(qk_geometry(2, Some(u32::MAX)), None);
    }
}

#[cfg(test)]
mod native_tests;
