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

    pub(in crate::kernels::cuda) fn matrix_bytes(
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
            Some(typed_gemv_need(dims, dtype)?.w)
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

pub(in crate::kernels::cuda) enum GgufMoeRawWeights<'a> {
    Stacks([RawRange<'a>; 2]),
    Separate(Vec<[RawRange<'a>; 2]>),
}

pub(in crate::kernels::cuda) enum GgufMoeRawPicks<'a> {
    Host(&'a [GgufMoePick]),
    Device {
        count: usize,
        ids: RawRange<'a>,
        weights: RawRange<'a>,
        scales: [RawRange<'a>; 2],
    },
}

pub(in crate::kernels::cuda) struct GgufMoeRaw<'a> {
    pub(in crate::kernels::cuda) shape: GgufMoeShape,
    pub(in crate::kernels::cuda) dtypes: [DenseDtype; 2],
    pub(in crate::kernels::cuda) caps: [f32; 2],
    pub(in crate::kernels::cuda) picks: GgufMoeRawPicks<'a>,
    pub(in crate::kernels::cuda) weights: GgufMoeRawWeights<'a>,
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
    device_ids: *const c_int,
    device_weights: *const f32,
    gate_scales: *const f32,
    down_scales: *const f32,
}

#[repr(C)]
struct MoeMatrices {
    gate: [*const c_void; MAX_GGUF_MOE_PICKS],
    down: [*const c_void; MAX_GGUF_MOE_PICKS],
}

const _: () = assert!(
    size_of::<MoeArgs>() == 232
        && size_of::<MoeMatrices>() == 128
        && size_of::<GgufMoePick>() == 16
        && offset_of!(MoeArgs, hidden) == 40
        && offset_of!(MoeArgs, gate_cap) == 64
        && offset_of!(MoeArgs, picks) == 72
        && offset_of!(MoeArgs, device_ids) == 200
        && offset_of!(MoeArgs, device_weights) == 208
        && offset_of!(MoeArgs, gate_scales) == 216
        && offset_of!(MoeArgs, down_scales) == 224
);

unsafe extern "C" {
    fn inference_cuda_gguf_moe_launch(
        args: *const MoeArgs,
        matrices: *const MoeMatrices,
        stream: *mut c_void,
        out_stage: *mut c_int,
        schedule: *mut c_int,
    ) -> c_int;
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum GgufGate {
    #[default]
    Packed = 0,
    GroupedIq2 = 1,
    IntegerDot = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufMoeSchedule {
    pub(crate) gate: GgufGate,
    pub(crate) exact_q2_down: bool,
    /// Both products on the q8_1 window matvec: the expert input and the middle rows are
    /// quantized to q8_1 first, so the output is not bit-identical to the f32 products. The
    /// wrapper takes it with device picks over stacks when both matrix types have a window or
    /// vendored case for their row lengths; the native entry never sees it.
    pub(crate) q8_window: bool,
}

impl GgufMoeSchedule {
    // INFERENCE_CUDA_GGUF_MOE_* in dense_gemv.h.
    const GATE_MASK: c_int = 3;
    const EXACT_Q2_DOWN: c_int = 4;
    const DECODE_SPLIT: c_int = 8;
    const ACTIVATE_ONLY: c_int = 16;
    const SUM_ONLY: c_int = 32;

    fn bits(self) -> c_int {
        self.gate as c_int
            | if self.exact_q2_down {
                Self::EXACT_Q2_DOWN
            } else {
                0
            }
    }

    fn from_bits(bits: c_int) -> Self {
        let gate = match bits & Self::GATE_MASK {
            2 => GgufGate::IntegerDot,
            1 => GgufGate::GroupedIq2,
            _ => GgufGate::Packed,
        };
        Self {
            gate,
            exact_q2_down: bits & Self::EXACT_Q2_DOWN != 0,
            q8_window: false,
        }
    }
}

/// One stage of the expert sequence, launched alone between products a caller queues itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::kernels::cuda) enum GgufMoePhase {
    /// The clipped activation of the gate and up products at the start of the scratch.
    Activate,
    /// The weighted sum of the down products after the middle rows.
    Sum,
}

/// Queues only `phase` of `call`'s expert sequence.
pub(in crate::kernels::cuda) fn launch_gguf_moe_phase(
    stream: &RawStream,
    call: GgufMoeRaw<'_>,
    phase: GgufMoePhase,
) -> Result<(), DeviceFailure> {
    let bits = match phase {
        GgufMoePhase::Activate => GgufMoeSchedule::ACTIVATE_ONLY,
        GgufMoePhase::Sum => GgufMoeSchedule::SUM_ONLY,
    };
    queue(stream, call, bits).map(|_| ())
}

pub(in crate::kernels::cuda) fn launch_gguf_moe(
    stream: &RawStream,
    call: GgufMoeRaw<'_>,
    schedule: GgufMoeSchedule,
    decode_split: bool,
) -> Result<GgufMoeSchedule, DeviceFailure> {
    let bits = schedule.bits()
        | if decode_split {
            GgufMoeSchedule::DECODE_SPLIT
        } else {
            0
        };
    queue(stream, call, bits).map(GgufMoeSchedule::from_bits)
}

/// Queues `call` with the request `bits`; returns the schedule bits the native entry reports.
fn queue(
    stream: &RawStream,
    call: GgufMoeRaw<'_>,
    mut bits: c_int,
) -> Result<c_int, DeviceFailure> {
    let mut picks = [GgufMoePick::default(); MAX_GGUF_MOE_PICKS];
    let (count, device_ids, device_weights, [gate_scales, down_scales]) = match call.picks {
        GgufMoeRawPicks::Host(host) => {
            picks[..host.len()].copy_from_slice(host);
            (
                host.len(),
                std::ptr::null(),
                std::ptr::null(),
                [std::ptr::null(); 2],
            )
        }
        GgufMoeRawPicks::Device {
            count,
            ids,
            weights,
            scales,
        } => (
            count,
            ids.address().cast_const().cast(),
            weights.address().cast_const().cast(),
            scales.map(|range| range.address().cast_const().cast()),
        ),
    };
    let mut matrices = None;
    let [gate, down] = match call.weights {
        GgufMoeRawWeights::Stacks(stacks) => stacks.map(|range| range.address().cast_const()),
        GgufMoeRawWeights::Separate(ranges) => {
            let [mut gate, mut down] = [[std::ptr::null(); MAX_GGUF_MOE_PICKS]; 2];
            for (slot, [up, projection]) in ranges.into_iter().enumerate() {
                gate[slot] = up.address().cast_const();
                down[slot] = projection.address().cast_const();
            }
            matrices = Some(MoeMatrices { gate, down });
            [std::ptr::null(); 2]
        }
    };
    let args = MoeArgs {
        gate,
        down,
        x: call.x.address().cast(),
        scratch: call.scratch.address().cast(),
        y: call.y.address().cast(),
        hidden: call.shape.hidden as c_int,
        middle: call.shape.middle as c_int,
        experts: call.shape.experts as c_int,
        count: count as c_int,
        gate_dtype: call.dtypes[0].code(),
        down_dtype: call.dtypes[1].code(),
        gate_cap: call.caps[0],
        up_cap: call.caps[1],
        picks,
        device_ids,
        device_weights,
        gate_scales,
        down_scales,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: The parent CUDA wrapper checked the adapter, dimensions, ordered picks,
    // extents, alignment and aliases of all live ranges. The native entry reads the
    // local argument struct during the call and copies scalar parameters into launches.
    let matrices = matrices
        .as_ref()
        .map_or(std::ptr::null(), std::ptr::from_ref);
    let request = if bits == 0 {
        std::ptr::null_mut()
    } else {
        &raw mut bits
    };
    let status = unsafe {
        inference_cuda_gguf_moe_launch(&args, matrices, stream.0.as_ptr(), &mut stage, request)
    };
    checked(status, stage)?;
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_q2_down_does_not_change_the_gate_request() {
        for gate in [GgufGate::Packed, GgufGate::GroupedIq2, GgufGate::IntegerDot] {
            let baseline = GgufMoeSchedule {
                gate,
                exact_q2_down: false,
                q8_window: false,
            };
            let candidate = GgufMoeSchedule {
                exact_q2_down: true,
                ..baseline
            };
            assert_eq!(baseline.bits(), gate as c_int);
            assert_eq!(candidate.bits(), gate as c_int | 4);
            assert_eq!(GgufMoeSchedule::from_bits(baseline.bits()), baseline);
            assert_eq!(GgufMoeSchedule::from_bits(candidate.bits()), candidate);
        }
        assert_eq!(GgufMoeSchedule::default().bits(), 0);
    }
}
