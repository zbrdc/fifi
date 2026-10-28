//! Raw FFI to the device-facts entry of `native/prefill/mmq_facts.cu` (declared in
//! `native/prefill/mmq_facts.h`, CPF-V4 V4-0): the facts every MMQ glue launch needs (the
//! built architecture, the SM count, the opt-in shared memory per block). Child module of
//! `native/prefill.rs`; the safety invariants are the parent's list — this entry is not a
//! launch: it takes no stream, enqueues nothing and reads only the CUDA device attributes.
use std::ffi::c_int;

use super::{DeviceFailure, STAGE_NONE};

/// Mirror of `inference_cuda_mmq_facts` in `mmq_facts.h`, field for field (the reserved
/// word included, so the layout assert pins it).
#[repr(C)]
struct MmqFactsArgsRaw {
    device: c_int,
    cc: c_int,
    nsm: c_int,
    reserved: c_int,
    smpbo: usize,
}

const _: () = assert!(
    std::mem::size_of::<MmqFactsArgsRaw>() == 24
        && std::mem::offset_of!(MmqFactsArgsRaw, nsm) == 8
        && std::mem::offset_of!(MmqFactsArgsRaw, smpbo) == 16
);

unsafe extern "C" {
    #[link_name = "inference_cuda_mmq_device_facts"]
    fn mmq_device_facts_raw(out: *mut MmqFactsArgsRaw) -> c_int;
}

/// The facts as the wrapper reads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct MmqFactsRaw {
    pub(in crate::kernels::cuda) cc: c_int,
    pub(in crate::kernels::cuda) nsm: c_int,
    pub(in crate::kernels::cuda) smpbo: usize,
}

/// Queries the device facts of the current device. `Err` carries the C entry's status: an
/// argument refusal (unreachable here), a CUDA error code of a failed query, or the native
/// exception status.
pub(in crate::kernels::cuda) fn mmq_facts() -> Result<MmqFactsRaw, DeviceFailure> {
    let mut out = MmqFactsArgsRaw {
        device: 0,
        cc: 0,
        nsm: 0,
        reserved: 0,
        smpbo: 0,
    };
    // SAFETY: `out` is a live, correctly aligned local of the mirrored struct; the entry
    // only writes through it and copies nothing out of it. It reads no device memory and
    // needs no stream.
    let status = unsafe { mmq_device_facts_raw(&mut out) };
    if status == 0 {
        return Ok(MmqFactsRaw {
            cc: out.cc,
            nsm: out.nsm,
            smpbo: out.smpbo,
        });
    }
    Err(DeviceFailure {
        status,
        stage: STAGE_NONE,
    })
}
