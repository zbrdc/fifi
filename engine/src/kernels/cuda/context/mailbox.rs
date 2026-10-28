//! The CPU-expert mailbox of the tiered decode step (TCAP C1, `ledger/plans/TCAP-DESIGN.md` sections 3.1 and
//! 3.3): one page-locked, device-mapped block through which a layer's kernels hand the CPU server a request and
//! the server hands back an answer, with no host synchronization inside the token.
//!
//! ```text
//!   0    request line, 128 B   seq u64 (the release word) | layer u32 | count u32 | ids i32[8] @16 | weights f32[8] @48
//!   128  row, 4 * hidden B     the normed row (the device writes it before it releases `seq`)
//!   ...  answer line, 128 B    done u64 (the release word, written by the host) | code u32 @8
//!   ...  partial, 4 * hidden B the CPU experts' partial sum (the host writes it before it releases `done`)
//! ```
//!
//! WHAT: [`Mailbox`] owns the mapping and is `!Send`; [`MailboxPort`] is the `Send + Sync` handle a server thread
//! polls and answers with, and holds no CUDA type. WHY a port: the server must run without touching the context.
//!
//! Protocol, one request in flight: the device writes row and header, then releases `seq`; the port acquires
//! `seq`, reads header and row, writes the partial and `code`, then releases `done = seq`. The one-in-flight
//! invariant is stream order: the next layer's classify runs after the wait kernel (C3) of the previous request
//! has returned, and a layer without a miss publishes nothing, so no word is read while it is written. Until C3
//! exists nothing on the device enforces it and the callers serialize on the host (the stress test does). A wait
//! that times out poisons the token and the mailbox is not reused before the reset (design 3.6).
//! Soundness (KUBDAN-MSG-84): the memory is written by the device, so no `&[f32]` or `&mut [f32]` is ever formed
//! over it; every access goes through the atomic word accessors of `native::resident::mailbox`, which export no
//! slice or pointer. Both ends are real atomics, because the Jetson hosts are aarch64 and not TSO.
//! Memory model: the host's `Release` and `Acquire` are CPU-thread-scope atomics and std has no scope operand,
//! while the device uses system scope (`st.release.sys`, `ld.acquire.sys`). The pairing rests on the platform
//! (x86 TSO posted writes; aarch64 `STLR`/`LDAR` observed through the coherent fabric), not on a proof in the
//! PTX model; C3 reviews the device loads against these accessors on the Jetson, and the Xavier window kills the
//! host-side ordering mutants that the x86 hosts cannot show.
//!
//! There is no read-modify-write on the mapping, ever: the Tegra pinned pages are uncached without I/O coherence, so an
//! atomic RMW there is not guaranteed, and the protocol needs only loads and stores.
//!
//! Dropping the [`Mailbox`] frees the mapping (after a device wait, decision 0022) only when no port is alive;
//! otherwise the pages are leaked, because a server thread may still read them.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for TCAP C6 (ledger/plans/TCAP-DESIGN.md)")
)]

use std::{array, marker::PhantomData, rc::Rc, sync::Arc, sync::atomic::Ordering};

use super::graph::{ensure_closed, run_or_defer};
use super::stream::{release, same_adapter};
use super::{CudaContext, LIVE_PINNED, device_error};
use crate::EngError;
use crate::kernels::cuda::guard::Lease;
use crate::kernels::cuda::native::resident::RawRange;
use crate::kernels::cuda::native::resident::mailbox::{self as raw, RawMapped};
use crate::kernels::cuda::resident::tiered_classify::REQUEST_BYTES;

/// Bytes of the answer line; every line starts on a 128-byte boundary.
const ANSWER_BYTES: usize = 128;
const LINE: usize = 128;
const MAX_HIDDEN: usize = 65536;
/// Byte offsets inside the request line.
const LAYER_AT: usize = 8;
const IDS_AT: usize = 16;
/// Words of the request header that follow `seq`: layer, count, 8 ids, 8 weights.
const HEADER_WORDS: usize = 18;
/// Offset of `code` inside the answer line.
const CODE_AT: usize = 8;

fn refuse(what: impl Into<String>) -> EngError {
    EngError::BadRequest(what.into())
}

/// Where each part lives in a mailbox of `hidden`-wide rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MailboxLayout {
    hidden: usize,
}

impl MailboxLayout {
    /// `BadRequest` unless `hidden` is a multiple of 4 from 4 to 65536 (the limits of the classify kernel).
    pub(crate) fn new(hidden: usize) -> Result<Self, EngError> {
        if !(4..=MAX_HIDDEN).contains(&hidden) || !hidden.is_multiple_of(4) {
            return Err(refuse(format!(
                "the CUDA mailbox takes a hidden width that is a multiple of 4 from 4 to {MAX_HIDDEN}; got {hidden}"
            )));
        }
        Ok(Self { hidden })
    }

    fn row_bytes(&self) -> usize {
        self.hidden * 4
    }

    /// Byte offset of the row.
    pub(crate) fn row_offset(&self) -> usize {
        REQUEST_BYTES
    }

    /// Byte offset of the answer line.
    pub(crate) fn answer_offset(&self) -> usize {
        REQUEST_BYTES + self.row_bytes().next_multiple_of(LINE)
    }

    /// Byte offset of the partial.
    pub(crate) fn partial_offset(&self) -> usize {
        self.answer_offset() + ANSWER_BYTES
    }

    /// Bytes of the whole mapping.
    pub(crate) fn total_bytes(&self) -> usize {
        self.partial_offset() + self.row_bytes().next_multiple_of(LINE)
    }
}

/// What the owner and its ports share. The mapping is freed only when the owner holds the last handle.
struct Shared {
    raw: RawMapped,
    layout: MailboxLayout,
}

/// The device-mapped block, owned by a [`CudaContext`].
pub(crate) struct Mailbox {
    shared: Option<Arc<Shared>>,
    layout: MailboxLayout,
    lease: Rc<Lease>,
    _thread_confined: PhantomData<*const ()>,
}

/// One request the device published.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Request {
    /// The sequence number the device released, from 1 and increasing by 1 per published request.
    pub(crate) seq: u64,
    /// The layer that asked.
    pub(crate) layer: u32,
    /// Number of valid entries of `ids` and `weights`; a value above 8 is a device bug the server must refuse.
    pub(crate) count: u32,
    /// The expert ids of the misses in pick order, -1 after the first `count`.
    pub(crate) ids: [i32; 8],
    /// The router weights of the misses in pick order, 0.0 after the first `count`.
    pub(crate) weights: [f32; 8],
}

/// The server's handle on a mailbox: poll, read the row, answer. Holds no CUDA type.
#[derive(Clone)]
pub(crate) struct MailboxPort {
    shared: Arc<Shared>,
}

impl CudaContext {
    /// Allocates a zeroed mailbox for rows of `hidden` floats.
    ///
    /// Returns `BadRequest` for a bad `hidden`, a call inside a capture window, and a device that cannot map
    /// host memory or has no unified addressing (the caller falls back to option B there).
    pub(crate) fn mailbox(&self, hidden: usize) -> Result<Mailbox, EngError> {
        self.ensure_usable()?;
        ensure_closed("a mailbox allocation")?;
        let layout = MailboxLayout::new(hidden)?;
        let _dispatch = self.enter()?;
        let caps = raw::mailbox_caps().map_err(device_error)?;
        if !(caps.can_map_host && caps.unified_addressing) {
            return Err(refuse(
                "this CUDA device cannot map host memory with unified addressing: the mailbox is unavailable",
            ));
        }
        let mapped = raw::mapped_alloc(layout.total_bytes()).map_err(device_error)?;
        LIVE_PINNED.fetch_add(layout.total_bytes(), Ordering::AcqRel);
        Ok(Mailbox {
            shared: Some(Arc::new(Shared {
                raw: mapped,
                layout,
            })),
            layout,
            lease: Rc::clone(&self.lease),
            _thread_confined: PhantomData,
        })
    }
}

impl Mailbox {
    fn shared(&self) -> Result<&Arc<Shared>, EngError> {
        self.shared
            .as_ref()
            .ok_or_else(|| refuse("the CUDA mailbox was already freed"))
    }

    /// A handle for a server thread. The mapping is freed only when no port is alive: dropping the mailbox while one is alive leaks it.
    pub(crate) fn port(&self) -> Result<MailboxPort, EngError> {
        Ok(MailboxPort {
            shared: Arc::clone(self.shared()?),
        })
    }

    /// `BadRequest` unless this mailbox was made by `ctx` (the adapter that owns its mapping).
    pub(in crate::kernels::cuda) fn check_adapter(
        &self,
        ctx: &CudaContext,
    ) -> Result<(), EngError> {
        same_adapter(&ctx.lease, &self.lease, "mailbox")
    }

    /// Where each part of this mailbox lives.
    pub(crate) fn layout(&self) -> MailboxLayout {
        self.layout
    }

    /// Width of the row and the partial in floats.
    pub(crate) fn hidden(&self) -> usize {
        self.layout.hidden
    }

    /// The request line as a device range, for the classify launch's `mbox_req`.
    pub(in crate::kernels::cuda) fn request_range(&self) -> Result<RawRange<'_>, EngError> {
        self.range(0, REQUEST_BYTES)
    }

    /// The row as a device range, for the classify launch's `mbox_row`.
    pub(in crate::kernels::cuda) fn row_range(&self) -> Result<RawRange<'_>, EngError> {
        self.range(self.layout.row_offset(), self.layout.row_bytes())
    }

    /// The answer line as a device range, for the wait-add launch's `mbox_ans` (TCAP C3).
    pub(in crate::kernels::cuda) fn answer_range(&self) -> Result<RawRange<'_>, EngError> {
        self.range(self.layout.answer_offset(), ANSWER_BYTES)
    }

    /// The partial as a device range, for the wait-add launch's `mbox_partial` (TCAP C3).
    pub(in crate::kernels::cuda) fn partial_range(&self) -> Result<RawRange<'_>, EngError> {
        self.range(self.layout.partial_offset(), self.layout.row_bytes())
    }

    fn range(&self, offset: usize, bytes: usize) -> Result<RawRange<'_>, EngError> {
        RawRange::new(self.shared()?.raw.device(), offset, bytes)
            .ok_or_else(|| refuse("the CUDA mailbox range lies outside the mapping"))
    }
}

impl MailboxPort {
    /// Width of the row in floats.
    pub(crate) fn hidden(&self) -> usize {
        self.shared.layout.hidden
    }

    /// The request published after sequence `after`, or `None` while the device has published none. The
    /// header is read after an acquire load of `seq`, so it is the one the device wrote before releasing it.
    pub(crate) fn poll(&self, after: u64) -> Option<Request> {
        let raw = &self.shared.raw;
        let seq = raw.load_acquire(0).filter(|&seq| seq > after)?;
        let mut words = [0u32; HEADER_WORDS];
        raw.read_words(LAYER_AT, &mut words).then_some(())?;
        Some(Request {
            seq,
            layer: words[0],
            count: words[1],
            ids: array::from_fn(|i| words[(IDS_AT - LAYER_AT) / 4 + i] as i32),
            weights: array::from_fn(|i| f32::from_bits(words[10 + i])),
        })
    }

    /// Copies the row of the request just polled into `out` (`hidden` floats). `false` for another length.
    pub(crate) fn read_row(&self, out: &mut [f32]) -> bool {
        let layout = self.shared.layout;
        out.len() == layout.hidden && self.shared.raw.read_f32s(layout.row_offset(), out)
    }

    /// Answers request `seq`: the partial (`hidden` floats) or an error `code` (nonzero), then the release of
    /// `done = seq`. `false`, with nothing written, for a partial of another length or a zero error code.
    pub(crate) fn answer(&self, seq: u64, result: Result<&[f32], u32>) -> bool {
        let (raw, layout) = (&self.shared.raw, self.shared.layout);
        let code = match result {
            Ok(partial) => {
                if partial.len() != layout.hidden
                    || !raw.write_f32s(layout.partial_offset(), partial)
                {
                    return false;
                }
                0
            }
            Err(0) => return false,
            Err(code) => code,
        };
        raw.write_words(layout.answer_offset() + CODE_AT, &[code])
            && raw.store_release(layout.answer_offset(), seq)
    }
}

impl Drop for Mailbox {
    fn drop(&mut self) {
        let Some(shared) = self.shared.take() else {
            return;
        };
        if let Ok(Shared {
            raw: mapped,
            layout,
        }) = Arc::try_unwrap(shared)
        {
            let bytes = layout.total_bytes();
            run_or_defer(move || release(mapped, raw::mapped_free, &LIVE_PINNED, bytes));
        }
    }
}

#[cfg(test)]
mod tests;
