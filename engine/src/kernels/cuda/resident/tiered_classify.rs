//! The tiered classification of one layer of the resident decode step: the router's picks mapped
//! through the layer's slot table row, in one launch (TCAP C2, `ledger/plans/TCAP-DESIGN.md`
//! section 3.2). The host oracle is `family::qwen36::tiered::classify::classify_reference`.
//!
//! Original code; the kernel and its semantics are documented in
//! `native/resident/tiered_classify.h`.
//!
//! [`CudaContext::tiered_classify`] queues one kernel on a [`DeviceStream`] and returns. It
//! allocates nothing, copies nothing and never synchronizes; a fault in the queued work surfaces
//! at the next synchronize of the stream. The caller owns the order: `ids`, `weights`,
//! `slot_row` and `norm_row` must hold their data before the call is queued (the stream or an
//! event), and the ranges the kernel writes are read only behind the stream.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a rejection from
//! the native layer latches the backend terminal. In order: the backend is usable; every range
//! belongs to this context; the shape (see [`check_shape`]); each range holds what the call uses
//! and starts at its alignment; every range the call writes overlaps no other range; and last,
//! when the launch is queued, the stream belongs to this context. The native entry and the C
//! entry repeat the checks on the ranges. A refused call changes nothing.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for TCAP C6 (ledger/plans/TCAP-DESIGN.md)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::tiered_classify::{
    ClassifyRaw, tiered_classify_launch,
};

const MAX_EXPERT: usize = 4096;
const MAX_TOP_K: usize = 8;
const MAX_HIDDEN: usize = 65536;
const MAX_LAYER: usize = 1023;
/// Bytes of the control block: u64 `[0]` miss count, `[1]` expected sequence, `[2]` sequence counter.
pub(crate) const CTRL_BYTES: usize = 64;
/// Bytes of the request line of the mailbox.
pub(crate) const REQUEST_BYTES: usize = 128;
/// Bytes of one logged pick: an i32 id and the f32 bits of its weight.
pub(crate) const LOG_BYTES_PER_PICK: usize = 8;

/// The sizes of one classification call. Qwen3.6 is `n_expert` 256, `top_k` 8 and `hidden` 2048;
/// `slots` is the bank size of the layer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClassifyShape {
    /// Experts of the layer, 1 to 4096.
    pub(crate) n_expert: usize,
    /// Bank slots per layer, 1 to `n_expert`.
    pub(crate) slots: usize,
    /// Picks per token, 1 to 8 (the request line holds eight).
    pub(crate) top_k: usize,
    /// Width of the normed row, a multiple of 4 from 4 to 65536.
    pub(crate) hidden: usize,
    /// The layer index, 0 to 1023; layer 0 starts a token (it clears the tiered word).
    pub(crate) layer: usize,
}

/// One classification call. The layout of every range is in `native/resident/tiered_classify.h`.
pub(crate) struct ClassifyCall<'b> {
    /// The sizes of the call.
    pub(crate) shape: ClassifyShape,
    /// i32 `[top_k]`, 4-byte aligned: the router's picks.
    pub(crate) ids: DeviceRange<'b>,
    /// f32 `[top_k]`, 4-byte aligned: the router's weights.
    pub(crate) weights: DeviceRange<'b>,
    /// i32 `[n_expert]`, 4-byte aligned: the layer's slot table row.
    pub(crate) slot_row: DeviceRange<'b>,
    /// f32 `[hidden]`, 16-byte aligned: the normed row.
    pub(crate) norm_row: DeviceRange<'b>,
    /// i32 `[top_k]`, 4-byte aligned, written: the slot of each pick.
    pub(crate) slot_ids_out: DeviceRange<'b>,
    /// f32 `[top_k]`, 4-byte aligned, written: the combine weight of each pick.
    pub(crate) weights_out: DeviceRange<'b>,
    /// `8 * top_k` bytes, 8-byte aligned, written: the route log of the layer.
    pub(crate) log_out: DeviceRange<'b>,
    /// [`CTRL_BYTES`], 8-byte aligned, written: the control block.
    pub(crate) ctrl: DeviceRange<'b>,
    /// i32, 4-byte aligned, written: the tiered word.
    pub(crate) word: DeviceRange<'b>,
    /// [`REQUEST_BYTES`], 8-byte aligned, written: the request line of the mailbox.
    pub(crate) mbox_req: DeviceRange<'b>,
    /// f32 `[hidden]`, 16-byte aligned, written: the row of the mailbox.
    pub(crate) mbox_row: DeviceRange<'b>,
}

/// `BadRequest` unless every integer is inside its limit. Pure: no device is needed.
pub(crate) fn check_shape(shape: &ClassifyShape) -> Result<(), EngError> {
    let ClassifyShape {
        n_expert,
        slots,
        top_k,
        hidden,
        layer,
    } = *shape;
    let inside = (1..=MAX_EXPERT).contains(&n_expert)
        && (1..=n_expert).contains(&slots)
        && (1..=MAX_TOP_K).contains(&top_k)
        && (4..=MAX_HIDDEN).contains(&hidden)
        && hidden.is_multiple_of(4)
        && layer <= MAX_LAYER;
    if inside {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA tiered classify takes n_expert 1 to {MAX_EXPERT}, slots 1 to n_expert, top_k 1 to \
         {MAX_TOP_K}, hidden a multiple of 4 from 4 to {MAX_HIDDEN} and layer 0 to {MAX_LAYER}; got \
         n_expert {n_expert}, slots {slots}, top_k {top_k}, hidden {hidden}, layer {layer}"
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

/// The eleven ranges of a call with what the kernel needs of each.
fn parts<'x, 'b>(call: &'x ClassifyCall<'b>) -> [Part<'x, 'b>; 11] {
    let s = &call.shape;
    let (top, row) = (s.top_k * 4, s.hidden * 4);
    let part = |name, range, bytes, align, written| Part {
        name,
        range,
        bytes,
        align,
        written,
    };
    [
        part("ids", &call.ids, top, 4, false),
        part("weights", &call.weights, top, 4, false),
        part("slot_row", &call.slot_row, s.n_expert * 4, 4, false),
        part("norm_row", &call.norm_row, row, 16, false),
        part("slot_ids_out", &call.slot_ids_out, top, 4, true),
        part("weights_out", &call.weights_out, top, 4, true),
        part(
            "log_out",
            &call.log_out,
            s.top_k * LOG_BYTES_PER_PICK,
            8,
            true,
        ),
        part("ctrl", &call.ctrl, CTRL_BYTES, 8, true),
        part("word", &call.word, 4, 4, true),
        part("mbox_req", &call.mbox_req, REQUEST_BYTES, 8, true),
        part("mbox_row", &call.mbox_row, row, 16, true),
    ]
}

/// Every range holds what the call uses at its alignment, and every range the kernel writes
/// overlaps no other range (the written one named first in the refusal).
fn check_ranges(call: &ClassifyCall<'_>) -> Result<(), EngError> {
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

fn to_int(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value)
        .map_err(|_| EngError::BadRequest(format!("the CUDA tiered classify {name} exceeds i32")))
}

impl CudaContext {
    /// Queues the classification of one layer on `stream`. See the module doc and
    /// `native/resident/tiered_classify.h` for the semantics.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a shape outside
    /// [`check_shape`], a range that is short or misaligned, and a written range that overlaps
    /// another range.
    pub(crate) fn tiered_classify(
        &self,
        stream: &DeviceStream,
        call: ClassifyCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_classify(&call)?;
        self.launch_on(stream, |stream_raw| {
            tiered_classify_launch(stream_raw, &raw)
        })
    }

    /// Every check of a classification call, then the native view of its ranges and ints.
    fn plan_classify<'b>(&self, call: &ClassifyCall<'b>) -> Result<ClassifyRaw<'b>, EngError> {
        let raw = |range: &DeviceRange<'b>| self.raw_range(range);
        let ranges = (
            raw(&call.ids)?,
            raw(&call.weights)?,
            raw(&call.slot_row)?,
            raw(&call.norm_row)?,
            raw(&call.slot_ids_out)?,
            raw(&call.weights_out)?,
            raw(&call.log_out)?,
            raw(&call.ctrl)?,
            raw(&call.word)?,
            raw(&call.mbox_req)?,
            raw(&call.mbox_row)?,
        );
        check_shape(&call.shape)?;
        check_ranges(call)?;
        let ClassifyShape {
            n_expert,
            slots,
            top_k,
            hidden,
            layer,
        } = call.shape;
        Ok(ClassifyRaw {
            ids: ranges.0,
            weights: ranges.1,
            slot_row: ranges.2,
            norm_row: ranges.3,
            slot_ids_out: ranges.4,
            weights_out: ranges.5,
            log_out: ranges.6,
            ctrl: ranges.7,
            word: ranges.8,
            mbox_req: ranges.9,
            mbox_row: ranges.10,
            n_expert: to_int("n_expert", n_expert)?,
            slots: to_int("slots", slots)?,
            top_k: to_int("top_k", top_k)?,
            hidden: to_int("hidden", hidden)?,
            layer: to_int("layer", layer)?,
        })
    }
}

#[cfg(test)]
mod tests;
