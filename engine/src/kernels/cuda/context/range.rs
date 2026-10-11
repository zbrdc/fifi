//! A checked byte range of one device allocation.
//!
//! The pinned-word copy takes a [`DeviceRange`] instead of a raw pointer, so
//! the bounds are proved once, in safe code, when the range is made. The
//! native layer checks them again (`native::resident::RawRange`).
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use super::DeviceBuffer;
use crate::EngError;

/// A checked byte range of one device allocation.
#[derive(Clone, Copy)]
pub(crate) struct DeviceRange<'b> {
    buffer: &'b DeviceBuffer,
    offset: usize,
    bytes: usize,
}

impl DeviceBuffer {
    /// The `bytes` bytes of this allocation from `offset` on.
    ///
    /// Returns `BadRequest` for an empty range, a range past the end of the
    /// allocation, or a buffer that was already freed.
    pub(crate) fn range(&self, offset: usize, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
        let Some(raw) = self.raw.as_ref() else {
            return Err(EngError::BadRequest(
                "the CUDA buffer was already freed".to_owned(),
            ));
        };
        let inside = offset
            .checked_add(bytes)
            .is_some_and(|end| end <= raw.bytes());
        if bytes == 0 || !inside {
            return Err(EngError::BadRequest(format!(
                "the CUDA range of {bytes} bytes at offset {offset} does not fit a buffer of {} bytes",
                raw.bytes()
            )));
        }
        Ok(DeviceRange {
            buffer: self,
            offset,
            bytes,
        })
    }
}

impl<'b> DeviceRange<'b> {
    /// Byte offset of the range inside its buffer.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    /// Bytes of the range; never 0.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer the range lies in.
    pub(crate) fn buffer(&self) -> &'b DeviceBuffer {
        self.buffer
    }
}
