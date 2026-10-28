//! Raw FFI to inference_cuda_sample_gemma_launch of native/resident/sample_gemma.cu.
//!
//! Safety invariants: RawRange::new proved each range in bounds; this module
//! re-proves sizes, alignment, parameter ranges and the separation of the three
//! outputs and the scratch row from the logits; the C entry repeats them.
//! Launch-only, under the dispatch guard (decisions 0023 and 0024).

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_sample_gemma_launch"]
    fn sample_gemma_launch_raw(
        args: *const SampleGemmaArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_sample_gemma_args, field for field.
#[repr(C)]
struct SampleGemmaArgsRepr {
    token: *mut c_void,
    logprob: *mut c_void,
    status: *mut c_void,
    probs: *mut c_void,
    logits: *const c_void,
    n: c_int,
    temperature: f32,
    top_k: c_int,
    top_p: f32,
    min_p: f32,
    seed_lo: u32,
    seed_hi: u32,
    pos_lo: u32,
    pos_hi: u32,
}

const _: () = assert!(
    size_of::<SampleGemmaArgsRepr>() == 80
        && offset_of!(SampleGemmaArgsRepr, token) == 0
        && offset_of!(SampleGemmaArgsRepr, logprob) == 8
        && offset_of!(SampleGemmaArgsRepr, status) == 16
        && offset_of!(SampleGemmaArgsRepr, probs) == 24
        && offset_of!(SampleGemmaArgsRepr, logits) == 32
        && offset_of!(SampleGemmaArgsRepr, n) == 40
        && offset_of!(SampleGemmaArgsRepr, temperature) == 44
        && offset_of!(SampleGemmaArgsRepr, top_k) == 48
        && offset_of!(SampleGemmaArgsRepr, top_p) == 52
        && offset_of!(SampleGemmaArgsRepr, min_p) == 56
        && offset_of!(SampleGemmaArgsRepr, seed_lo) == 60
        && offset_of!(SampleGemmaArgsRepr, seed_hi) == 64
        && offset_of!(SampleGemmaArgsRepr, pos_lo) == 68
        && offset_of!(SampleGemmaArgsRepr, pos_hi) == 72
);

/// One sampler launch.
pub(in crate::kernels::cuda) struct SampleGemmaRaw<'a> {
    pub(in crate::kernels::cuda) token: RawRange<'a>,
    pub(in crate::kernels::cuda) logprob: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) probs: RawRange<'a>,
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) temperature: f32,
    pub(in crate::kernels::cuda) top_k: usize,
    pub(in crate::kernels::cuda) top_p: f32,
    pub(in crate::kernels::cuda) min_p: f32,
    pub(in crate::kernels::cuda) seed: u64,
    pub(in crate::kernels::cuda) position: u64,
}

fn check(call: &SampleGemmaRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized = call.probs.bytes >= bytes && call.logits.bytes >= bytes;
    let small = |r: &RawRange<'_>| r.bytes >= 4 && r.offset.is_multiple_of(4);
    let aligned = call.probs.offset.is_multiple_of(4) && call.logits.offset.is_multiple_of(4);
    let logits = Span::of(&call.logits, bytes);
    let probs = Span::of(&call.probs, bytes);
    let outputs = [&call.token, &call.logprob, &call.status];
    let mut apart = probs.apart(&logits);
    for (index, range) in outputs.iter().enumerate() {
        let out = Span::of(range, 4);
        apart = apart && out.apart(&logits) && out.apart(&probs);
        for other in outputs.iter().skip(index + 1) {
            apart = apart && out.apart(&Span::of(other, 4));
        }
    }
    let params = (32..=4_194_304).contains(&call.n)
        && call.n.is_multiple_of(32)
        && call.temperature.is_finite()
        && call.top_k <= call.n
        && call.top_p > 0.0
        && call.top_p <= 1.0
        && call.min_p >= 0.0
        && call.min_p < 1.0;
    if !(sized
        && aligned
        && small(&call.token)
        && small(&call.logprob)
        && small(&call.status)
        && apart
        && params)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues one sample on stream.
pub(in crate::kernels::cuda) fn sample_gemma_launch(
    stream: &RawStream,
    call: &SampleGemmaRaw<'_>,
) -> Result<(), DeviceFailure> {
    check(call)?;
    let args = SampleGemmaArgsRepr {
        token: call.token.address(),
        logprob: call.logprob.address(),
        status: call.status.address(),
        probs: call.probs.address(),
        logits: call.logits.address().cast_const(),
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
        temperature: call.temperature,
        top_k: c_int::try_from(call.top_k).map_err(|_| invalid_argument())?,
        top_p: call.top_p,
        min_p: call.min_p,
        seed_lo: call.seed as u32,
        seed_hi: (call.seed >> 32) as u32,
        pos_lo: call.position as u32,
        pos_hi: (call.position >> 32) as u32,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check proved the extents and the parameter ranges; the adapter repeats them.
    let status = unsafe { sample_gemma_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
