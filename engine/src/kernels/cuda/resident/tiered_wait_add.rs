//! The tiered wait-add of one layer of the resident decode step: the kernel that waits, with no
//! host synchronization, for the CPU server's answer through the mailbox and adds its partial into
//! the residual row.
//!
//! Original code; the kernel and its semantics are documented in
//! `native/resident/tiered_wait_add.h`.
//!
//! [`CudaContext::tiered_wait_add`] queues one kernel on a [`DeviceStream`] and returns. It
//! allocates nothing, copies nothing and never synchronizes; a fault in the queued work surfaces at
//! the next synchronize of the stream. The caller owns the order: `ctrl` must hold the layer's miss
//! count and expected sequence before the call is queued (classify writes them on the same stream),
//! the mailbox must belong to this context's tiered runtime, and `resid` and `word` are read only
//! behind the stream. The kernel blocks its own block until the server answers or the timeout
//! passes, so it must never be queued on a stream the host must free mid-token.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a rejection from
//! the native layer latches the backend terminal. In order: the backend is usable; every range
//! belongs to this context; the shape (see [`check_shape`]); the `hidden` of the call equals the
//! mailbox's; each caller range holds what the call uses and starts at its alignment; every range
//! the kernel writes (`resid`, `word`) overlaps no other range; and last, when the launch is
//! queued, the stream belongs to this context. The two mailbox ranges are exact and aligned by
//! construction (`Mailbox::answer_range`, `Mailbox::partial_range`), and the native entry and the
//! C entry repeat every check on the addresses. A refused call changes nothing.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the tiered decode step")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, Mailbox, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::tiered_wait_add::{WaitAddRaw, tiered_wait_add_launch};

const MAX_HIDDEN: usize = 65536;
/// The longest wait the kernel accepts, 120 seconds; the server's own timeout fires first in
/// production (the design adds a margin to `TieredPlan.timeout_ms`).
pub(crate) const MAX_TIMEOUT_NS: u64 = 120_000_000_000;
/// Bytes of the control block the call reads: u64 `[0]` miss count, `[1]` expected sequence.
pub(crate) const CTRL_BYTES: usize = 64;

/// The sizes of one wait-add call. Qwen3.6 is `hidden` 2048; the timeout is the tiered plan's plus
/// a margin.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WaitAddShape {
    /// Width of the row, a multiple of 4 from 4 to 65536; it must equal the mailbox's.
    pub(crate) hidden: usize,
    /// The wait budget in nanoseconds, 1 to [`MAX_TIMEOUT_NS`].
    pub(crate) timeout_ns: u64,
}

/// One wait-add call. The layout of every range is in `native/resident/tiered_wait_add.h`.
pub(crate) struct WaitAddCall<'b> {
    /// Width of the row in floats, a multiple of 4 from 4 to 65536.
    pub(crate) hidden: usize,
    /// [`CTRL_BYTES`], 8-byte aligned: the layer's control block, written by classify.
    pub(crate) ctrl: DeviceRange<'b>,
    /// i32, 4-byte aligned, written: the tiered word; the kernel only ORs bits into it.
    pub(crate) word: DeviceRange<'b>,
    /// f32 `[hidden]`, 16-byte aligned, written in place: the residual row.
    pub(crate) resid: DeviceRange<'b>,
    /// The mailbox whose answer line and partial the kernel waits for; its `hidden` must match.
    pub(crate) mailbox: &'b Mailbox,
    /// The wait budget in nanoseconds, 1 to [`MAX_TIMEOUT_NS`].
    pub(crate) timeout_ns: u64,
}

/// `BadRequest` unless every integer is inside its limit. Pure: no device is needed.
pub(crate) fn check_shape(shape: &WaitAddShape) -> Result<(), EngError> {
    let WaitAddShape { hidden, timeout_ns } = *shape;
    let inside = (4..=MAX_HIDDEN).contains(&hidden)
        && hidden.is_multiple_of(4)
        && (1..=MAX_TIMEOUT_NS).contains(&timeout_ns);
    if inside {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA tiered wait-add takes hidden a multiple of 4 from 4 to {MAX_HIDDEN} and a timeout \
         from 1 to {MAX_TIMEOUT_NS} nanoseconds; got hidden {hidden}, timeout {timeout_ns} \
         nanoseconds"
    )))
}

/// One range of a call: its name, the bytes it must hold, its alignment, and whether the kernel
/// writes it.
struct Part<'x, 'b> {
    name: &'static str,
    range: &'x DeviceRange<'b>,
    bytes: usize,
    align: usize,
    written: bool,
}

/// The three caller ranges of a call with what the kernel needs of each. The two mailbox ranges
/// are not here: `Mailbox::answer_range` and `Mailbox::partial_range` hand out exactly the bytes
/// the kernel reads (128 and the mailbox's `hidden * 4`, which the hidden refusal already matched
/// to the call) on 128-byte-aligned lines inside the mapping, and the mapping is an allocation of
/// its own, so it shares no byte with any device buffer; the native entry proves both again on
/// the addresses.
fn parts<'x, 'b>(call: &'x WaitAddCall<'b>) -> [Part<'x, 'b>; 3] {
    let row = call.hidden * 4;
    let part = |name, range, bytes, align, written| Part {
        name,
        range,
        bytes,
        align,
        written,
    };
    [
        part("ctrl", &call.ctrl, CTRL_BYTES, 8, false),
        part("word", &call.word, 4, 4, true),
        part("resid", &call.resid, row, 16, true),
    ]
}

/// Every range holds what the call uses at its alignment, and every range the kernel writes
/// (`resid`, `word`) overlaps no other range (the written one named first in the refusal).
fn check_ranges(call: &WaitAddCall<'_>) -> Result<(), EngError> {
    let parts = parts(call);
    for part in &parts {
        require_fit(part.name, part.range, part.bytes, part.align)?;
    }
    for (i, a) in parts.iter().enumerate() {
        for b in &parts[i + 1..] {
            if !(a.written || b.written) {
                continue;
            }
            let (first, second) = if a.written { (a, b) } else { (b, a) };
            require_apart(
                &Claim::of(first.name, first.range, first.bytes),
                &Claim::of(second.name, second.range, second.bytes),
            )?;
        }
    }
    Ok(())
}

fn to_int(value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value)
        .map_err(|_| EngError::BadRequest("the CUDA tiered wait-add hidden exceeds i32".to_owned()))
}

impl CudaContext {
    /// Queues the wait-add of one layer on `stream`. See the module doc and
    /// `native/resident/tiered_wait_add.h` for the semantics.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a shape outside
    /// [`check_shape`], a `hidden` that differs from the mailbox's, a range that is short or
    /// misaligned, and a written range that overlaps another range.
    pub(crate) fn tiered_wait_add(
        &self,
        stream: &DeviceStream,
        call: WaitAddCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_wait_add(&call)?;
        self.launch_on(stream, |stream_raw| {
            tiered_wait_add_launch(stream_raw, &raw)
        })
    }

    /// Every check of a wait-add call, then the native view of its ranges and ints.
    fn plan_wait_add<'b>(&self, call: &WaitAddCall<'b>) -> Result<WaitAddRaw<'b>, EngError> {
        call.mailbox.check_adapter(self)?;
        let (ctrl, word, resid) = (
            self.raw_range(&call.ctrl)?,
            self.raw_range(&call.word)?,
            self.raw_range(&call.resid)?,
        );
        let (mbox_ans, mbox_partial) =
            (call.mailbox.answer_range()?, call.mailbox.partial_range()?);
        check_shape(&WaitAddShape {
            hidden: call.hidden,
            timeout_ns: call.timeout_ns,
        })?;
        let mailbox_hidden = call.mailbox.hidden();
        if call.hidden != mailbox_hidden {
            return Err(EngError::BadRequest(format!(
                "the CUDA tiered wait-add takes the hidden width of its mailbox, {mailbox_hidden}; \
                 got {}",
                call.hidden
            )));
        }
        check_ranges(call)?;
        Ok(WaitAddRaw {
            ctrl,
            word,
            mbox_ans,
            mbox_partial,
            resid,
            hidden: to_int(call.hidden)?,
            timeout_ns: call.timeout_ns,
        })
    }
}

#[cfg(test)]
mod tests;
