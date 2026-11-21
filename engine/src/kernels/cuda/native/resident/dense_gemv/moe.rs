//! FFI for the indexed packed GGUF expert sequence in dense_gemv.cu.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DenseDtype, GEMV_MAX_COLS, GEMV_MAX_ROWS, GemvDims, typed_gemv_need};
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, checked};

pub(crate) const MAX_GGUF_MOE_PICKS: usize = 8;

#[derive(Clone, Copy)]
pub(crate) struct GgufMoeShape {
    pub(crate) hidden: usize,
    pub(crate) middle: usize,
    pub(crate) experts: usize,
}

impl GgufMoeShape {
    pub(crate) fn scratch_bytes(self, count: usize) -> Option<usize> {
        if (1..=GEMV_MAX_COLS).contains(&self.hidden)
            && (1..=GEMV_MAX_COLS).contains(&self.middle)
            && (1..=GEMV_MAX_ROWS).contains(&self.experts)
            && (1..=MAX_GGUF_MOE_PICKS).contains(&count)
        {
            Some(count * (self.hidden + 3 * self.middle) * size_of::<f32>())
        } else {
            None
        }
    }

    pub(in crate::kernels::cuda) fn stack_bytes(
        self,
        dtypes: [DenseDtype; 2],
    ) -> Option<[usize; 2]> {
        let matrix = |rows, cols, dtype: DenseDtype| {
            if !dtype.is_packed_gguf() {
                return None;
            }
            let dims = GemvDims {
                m: 1,
                rows,
                cols,
                ld_x: cols,
                ld_y: rows,
            };
            typed_gemv_need(dims, dtype)?.w.checked_mul(self.experts)
        };
        Some([
            matrix(self.middle.checked_mul(2)?, self.hidden, dtypes[0])?,
            matrix(self.hidden, self.middle, dtypes[1])?,
        ])
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct GgufMoePick {
    pub(crate) expert: u32,
    pub(crate) gate_scale: f32,
    pub(crate) down_scale: f32,
    pub(crate) weight: f32,
}

pub(in crate::kernels::cuda) struct GgufMoeRaw<'a> {
    pub(in crate::kernels::cuda) shape: GgufMoeShape,
    pub(in crate::kernels::cuda) dtypes: [DenseDtype; 2],
    pub(in crate::kernels::cuda) caps: [f32; 2],
    pub(in crate::kernels::cuda) picks: &'a [GgufMoePick],
    pub(in crate::kernels::cuda) stacks: [RawRange<'a>; 2],
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) scratch: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
}

#[repr(C)]
struct MoeArgs {
    gate: *const c_void,
    down: *const c_void,
    x: *const f32,
    scratch: *mut f32,
    y: *mut f32,
    hidden: c_int,
    middle: c_int,
    experts: c_int,
    count: c_int,
    gate_dtype: c_int,
    down_dtype: c_int,
    gate_cap: f32,
    up_cap: f32,
    picks: [GgufMoePick; MAX_GGUF_MOE_PICKS],
}

const _: () = assert!(
    size_of::<MoeArgs>() == 200
        && size_of::<GgufMoePick>() == 16
        && offset_of!(MoeArgs, hidden) == 40
        && offset_of!(MoeArgs, gate_cap) == 64
        && offset_of!(MoeArgs, picks) == 72
);

unsafe extern "C" {
    fn inference_cuda_gguf_moe_launch(
        args: *const MoeArgs,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn launch_gguf_moe(
    stream: &RawStream,
    call: GgufMoeRaw<'_>,
) -> Result<(), DeviceFailure> {
    let mut picks = [GgufMoePick::default(); MAX_GGUF_MOE_PICKS];
    picks[..call.picks.len()].copy_from_slice(call.picks);
    let args = MoeArgs {
        gate: call.stacks[0].address().cast_const(),
        down: call.stacks[1].address().cast_const(),
        x: call.x.address().cast(),
        scratch: call.scratch.address().cast(),
        y: call.y.address().cast(),
        hidden: call.shape.hidden as c_int,
        middle: call.shape.middle as c_int,
        experts: call.shape.experts as c_int,
        count: call.picks.len() as c_int,
        gate_dtype: call.dtypes[0].code(),
        down_dtype: call.dtypes[1].code(),
        gate_cap: call.caps[0],
        up_cap: call.caps[1],
        picks,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: The parent CUDA wrapper checked the adapter, dimensions, ordered picks,
    // extents, alignment and aliases of all live ranges. The native entry reads the
    // local argument struct during the call and copies scalar parameters into launches.
    let status = unsafe { inference_cuda_gguf_moe_launch(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
