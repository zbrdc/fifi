//! The pinned result word of a resident step: eight bytes of page-locked host
//! memory that a device-to-host copy on a stream fills with a token and a status.
//!
//! The host may read the word only after an event recorded behind the copy has
//! completed. [`PinnedWord::read_after`] enforces it with the pending-work
//! sequence of [`super::stream`]: the copy remembers the stream id and the
//! sequence number it was enqueued with, an event remembers the stream id and
//! the count at its record, and the read is refused unless the event is on the
//! same stream and was recorded at or after the copy. A read of memory that a
//! copy may still be writing is therefore a typed error, not a convention.
//!
//! Inside a capture window the word can be neither allocated nor copied into, and `read_after`
//! refuses an event that was last recorded inside a window (the host wait is a CUDA error: status 1 on the 12.2 runtime, 907 in its documentation). A
//! resident step replays its graph first and queues the word copy, the event record and the host
//! wait eagerly afterwards.
//!
//! The word is `!Send` and `!Sync`, shares the lease of its context, and follows
//! the drop rule of [`super::stream::release`].

use std::{cell::Cell, marker::PhantomData, rc::Rc, sync::atomic::Ordering};

use super::graph::{ensure_closed, run_or_defer};
use super::stream::{DeviceEvent, DeviceStream, release, same_adapter};
use super::{CudaContext, DeviceRange, LIVE_PINNED, device_error};
use crate::EngError;
use crate::kernels::cuda::guard::Lease;
use crate::kernels::cuda::native::resident::{self, RawPinned, RawRange};
use crate::kernels::cuda::{TERMINAL, ensure_usable};

/// Bytes of the word: a little-endian `i32` token, then an `i32` status.
const WORD_BYTES: usize = 8;

/// Eight bytes of page-locked host memory owned by a [`CudaContext`].
pub(crate) struct PinnedWord {
    raw: Option<RawPinned>,
    lease: Rc<Lease>,
    /// Stream id and sequence number of the copy that has not been read yet.
    pending: Cell<Option<(u64, u64)>>,
    _thread_confined: PhantomData<*const ()>,
}

/// The two values a resident step writes into the word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WordValue {
    pub(crate) token: i32,
    pub(crate) status: i32,
}

impl CudaContext {
    /// Allocates the pinned word.
    pub(crate) fn pinned_word(&self) -> Result<PinnedWord, EngError> {
        self.ensure_usable()?;
        ensure_closed("a pinned word allocation")?;
        let _dispatch = self.enter()?;
        let raw = resident::pinned_alloc(WORD_BYTES).map_err(device_error)?;
        LIVE_PINNED.fetch_add(WORD_BYTES, Ordering::AcqRel);
        Ok(PinnedWord {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            pending: Cell::new(None),
            _thread_confined: PhantomData,
        })
    }

    /// Queues a copy of the 8 bytes at the start of `src` into `word` on
    /// `stream`. The copy has not finished when the call returns.
    ///
    /// Returns `BadRequest` when `src` holds fewer than 8 bytes or starts at an
    /// offset that is not a multiple of 4, when an earlier copy was not read yet,
    /// and for a stream, buffer or word of another adapter.
    pub(crate) fn copy_word_to_host(
        &self,
        stream: &DeviceStream,
        src: DeviceRange<'_>,
        word: &PinnedWord,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        // The bookkeeping below (`pending`) would advance for a copy that did not run.
        ensure_closed("a word copy")?;
        same_adapter(&self.lease, &word.lease, "pinned word")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        let device = self.owned(src.buffer())?;
        if src.bytes() < WORD_BYTES || !src.offset().is_multiple_of(4) {
            return Err(EngError::BadRequest(format!(
                "the CUDA word copy needs 8 bytes at a 4-byte aligned offset, got {} bytes at offset {}",
                src.bytes(),
                src.offset()
            )));
        }
        if word.pending.get().is_some() {
            return Err(EngError::BadRequest(
                "the CUDA pinned word still holds a copy that was not read".to_owned(),
            ));
        }
        let range = RawRange::new(device, src.offset(), WORD_BYTES)
            .ok_or_else(|| EngError::BadRequest("the CUDA word range is not valid".to_owned()))?;
        let (stream_raw, word_raw) = (stream.raw()?, word.raw()?);
        let _dispatch = self.enter()?;
        resident::copy_d2h_async(word_raw, range, stream_raw).map_err(device_error)?;
        word.pending.set(Some((stream.id(), stream.note_enqueue())));
        Ok(())
    }
}

impl PinnedWord {
    /// Waits for `event` and reads the word.
    ///
    /// Returns `BadRequest` unless a copy is pending and `event` was last
    /// recorded on the stream of that copy at or after the copy was enqueued.
    /// The word is then free for the next copy.
    pub(crate) fn read_after(&self, event: &DeviceEvent) -> Result<WordValue, EngError> {
        ensure_usable(&TERMINAL)?;
        same_adapter(&self.lease, event.lease(), "event")?;
        let pending = self.pending.get().ok_or_else(|| {
            EngError::BadRequest("the CUDA pinned word holds no copy to read".to_owned())
        })?;
        let event_mark = event
            .mark()
            .ok_or_else(|| EngError::BadRequest("the CUDA event was never recorded".to_owned()))?;
        if event_mark.0 != pending.0 {
            return Err(EngError::BadRequest(
                "the CUDA event was recorded on another stream than the word copy".to_owned(),
            ));
        }
        if event_mark.1 < pending.1 {
            return Err(EngError::BadRequest(
                "the CUDA event was recorded before the word copy was enqueued".to_owned(),
            ));
        }
        event.synchronize()?;
        let bytes = resident::pinned_read(self.raw()?);
        self.pending.set(None);
        let [t0, t1, t2, t3, s0, s1, s2, s3] = bytes;
        Ok(WordValue {
            token: i32::from_le_bytes([t0, t1, t2, t3]),
            status: i32::from_le_bytes([s0, s1, s2, s3]),
        })
    }

    /// The 8 bytes of the word right now, without waiting for any copy or event.
    /// Tests only: the typed `read_after` is the one way for production code, and
    /// a test uses this to observe what a stream or event synchronize guaranteed.
    #[cfg(test)]
    pub(super) fn peek(&self) -> Result<[u8; 8], EngError> {
        Ok(resident::pinned_read(self.raw()?))
    }

    fn raw(&self) -> Result<&RawPinned, EngError> {
        self.raw.as_ref().ok_or_else(|| {
            EngError::BadRequest("the CUDA pinned word was already freed".to_owned())
        })
    }
}

impl Drop for PinnedWord {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            run_or_defer(move || release(raw, resident::pinned_free, &LIVE_PINNED, WORD_BYTES));
        }
    }
}
