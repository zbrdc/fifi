//! Owned page-locked storage for stream-ordered device/host transfers (OFF-2 D1).
//!
//! WHAT: [`PinnedBuf`] generalises [`super::pinned::PinnedWord`] from 8 bytes to any size and from
//! one copy to several copies into disjoint ranges before one event. WHY: the tiered step
//! reads the normed row, the router ids and the router weights of every layer back to the host
//! (about 8 KB) and must do it without a caller convention that makes a safe API unsound.
//!
//! Soundness is enforced, not documented. Both pinned copy entries take `&mut PinnedBuf`
//! and record the stream id and sequence of the LAST transfer. Host reads and writes refuse
//! while a transfer is outstanding. [`PinnedBuf::wait`] clears the outstanding state only for an event
//! recorded on that stream at or after the last copy, after synchronizing it. All copies in
//! flight must share one stream (a copy on another stream is refused), so one event proves
//! them all. Dropping the buffer follows decision 0022 through [`super::stream::release`]: the
//! native free waits for the whole device first, and when that wait fails nothing is freed
//! (the pages are leaked) and the backend latches terminal, exactly as for the pinned word.
//!
//! The memory is zeroed at allocation, so a read of an untouched range returns zeros
//! rather than indeterminate bytes. Inside a capture window the buffer can neither be allocated
//! nor copied into. The buffer is `!Send` and `!Sync`.

use std::{marker::PhantomData, rc::Rc, sync::atomic::Ordering};

use super::graph::{ensure_closed, run_or_defer};
use super::stream::{DeviceEvent, DeviceStream, release, same_adapter};
use super::{CudaContext, DeviceRange, LIVE_PINNED, device_error};
use crate::EngError;
use crate::kernels::cuda::guard::Lease;
use crate::kernels::cuda::native::resident::{self, RawPinned, RawRange};
use crate::kernels::cuda::{TERMINAL, ensure_usable};

/// Page-locked host memory of `bytes` bytes owned by a [`CudaContext`].
pub(crate) struct PinnedBuf {
    raw: Option<RawPinned>,
    lease: Rc<Lease>,
    bytes: usize,
    /// Stream id and sequence of the last transfer that no waited event has covered yet.
    pending: Option<(u64, u64)>,
    _thread_confined: PhantomData<*const ()>,
}

fn refuse(what: impl Into<String>) -> EngError {
    EngError::BadRequest(what.into())
}

impl CudaContext {
    /// Allocates `bytes` (at least one) of zeroed page-locked host memory.
    pub(crate) fn pinned_buf(&self, bytes: usize) -> Result<PinnedBuf, EngError> {
        self.ensure_usable()?;
        ensure_closed("a pinned buffer allocation")?;
        if bytes == 0 {
            return Err(refuse("the CUDA pinned buffer needs at least one byte"));
        }
        let _dispatch = self.enter()?;
        let raw = resident::pinned_alloc(bytes).map_err(device_error)?;
        resident::pinned_zero(&raw);
        LIVE_PINNED.fetch_add(bytes, Ordering::AcqRel);
        Ok(PinnedBuf {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            bytes,
            pending: None,
            _thread_confined: PhantomData,
        })
    }

    /// Queues a copy of `src` into `dst` at byte `dst_offset` on `stream`, behind the work
    /// queued earlier on it. The copy has not finished when the call returns.
    ///
    /// Returns `BadRequest` for an empty `src`, a range that ends past `dst`, a stream, buffer
    /// or pinned buffer of another adapter, a call inside a capture window, and a copy while an
    /// earlier copy into `dst` is outstanding on ANOTHER stream (one event could not cover
    /// both).
    pub(crate) fn copy_to_pinned(
        &self,
        stream: &DeviceStream,
        src: DeviceRange<'_>,
        dst: &mut PinnedBuf,
        dst_offset: usize,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        // The bookkeeping below (`pending`) would advance for a copy that did not run.
        ensure_closed("a pinned copy")?;
        same_adapter(&self.lease, &dst.lease, "pinned buffer")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        let device = self.owned(src.buffer())?;
        let fits = dst_offset
            .checked_add(src.bytes())
            .is_some_and(|end| end <= dst.bytes);
        if src.bytes() == 0 || !fits {
            return Err(refuse(format!(
                "the CUDA pinned copy of {} bytes at offset {dst_offset} lies outside a pinned buffer of {} bytes",
                src.bytes(),
                dst.bytes
            )));
        }
        if dst.pending.is_some_and(|(id, _)| id != stream.id()) {
            return Err(refuse(
                "the CUDA pinned buffer has a copy outstanding on another stream",
            ));
        }
        stream.next_enqueue()?;
        let range = RawRange::new(device, src.offset(), src.bytes())
            .ok_or_else(|| refuse("the CUDA pinned copy range is not valid"))?;
        let (stream_raw, dst_raw) = (stream.raw()?, dst.raw()?);
        let _dispatch = self.enter()?;
        resident::copy_d2h_async_at(dst_raw, dst_offset, range, stream_raw)
            .map_err(device_error)?;
        dst.pending = Some((stream.id(), stream.note_enqueue()));
        Ok(())
    }

    /// Queues pinned-host bytes into a device range; a waited event must cover the transfer.
    pub(crate) fn copy_from_pinned(
        &self,
        stream: &DeviceStream,
        src: &mut PinnedBuf,
        src_offset: usize,
        dst: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        ensure_closed("a pinned copy")?;
        same_adapter(&self.lease, &src.lease, "pinned buffer")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        let device = self.owned(dst.buffer())?;
        let fits = src_offset
            .checked_add(dst.bytes())
            .is_some_and(|end| end <= src.bytes);
        if dst.bytes() == 0 || !fits {
            return Err(refuse(
                "the CUDA pinned import range lies outside the host buffer",
            ));
        }
        if src.pending.is_some_and(|(id, _)| id != stream.id()) {
            return Err(refuse(
                "the CUDA pinned buffer has a copy outstanding on another stream",
            ));
        }
        stream.next_enqueue()?;
        let range = RawRange::new(device, dst.offset(), dst.bytes())
            .ok_or_else(|| refuse("the CUDA pinned import device range is not valid"))?;
        let _dispatch = self.enter()?;
        resident::copy_h2d_async_at(range, src.raw()?, src_offset, stream.raw()?)
            .map_err(device_error)?;
        src.pending = Some((stream.id(), stream.note_enqueue()));
        Ok(())
    }
}

impl PinnedBuf {
    /// Writes host bytes only when no transfer can still access this storage.
    pub(crate) fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        ensure_closed("a pinned write")?;
        if self.pending.is_some() {
            return Err(refuse(
                "the CUDA pinned buffer has a copy outstanding: wait for an event recorded behind it first",
            ));
        }
        if bytes.is_empty() {
            return Err(refuse("the CUDA pinned write of an empty slice"));
        }
        resident::pinned_write_from(self.raw()?, offset, bytes).map_err(|_| {
            refuse(format!(
                "the CUDA pinned write of {} bytes at offset {offset} lies outside a pinned buffer of {} bytes",
                bytes.len(),
                self.bytes
            ))
        })
    }

    /// Size of the buffer in bytes.
    pub(crate) fn len(&self) -> usize {
        self.bytes
    }

    /// Whether a transfer is outstanding (not yet covered by a waited event). Tests only: the
    /// production paths learn the same from the refusals of `read` and `wait`.
    #[cfg(test)]
    pub(crate) fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Proves completion on the copy's owning stream, including when no copy is pending.
    /// A failed synchronization leaves the obligation intact; this does not forge an event mark.
    pub(crate) fn synchronize(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        ensure_closed("a pinned synchronize")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        if self.pending.is_some_and(|pending| pending.0 != stream.id()) {
            return Err(refuse("the pinned copy belongs to another stream"));
        }
        stream.synchronize()?;
        self.pending = None;
        Ok(())
    }

    /// Waits for `event` and marks every transfer of the buffer as finished.
    ///
    /// Returns `BadRequest` unless a copy is outstanding and `event` was last recorded on the
    /// stream of that copy at or after the last copy was enqueued, and for an event of another
    /// adapter. A refusal leaves the copy outstanding, so a correct event can still release it.
    pub(crate) fn wait(&mut self, event: &DeviceEvent) -> Result<(), EngError> {
        self.check_event(event)?;
        event.synchronize()?;
        self.pending = None;
        Ok(())
    }

    /// Validates every outstanding transfer before one fence releases the whole group.
    pub(crate) fn wait_all(buffers: &mut [&mut Self], event: &DeviceEvent) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        for buffer in buffers.iter().filter(|buffer| buffer.pending.is_some()) {
            buffer.check_event(event)?;
        }
        if buffers.iter().all(|buffer| buffer.pending.is_none()) {
            return Err(refuse("the CUDA pinned buffers hold no copy to wait for"));
        }
        event.synchronize()?;
        for buffer in buffers {
            buffer.pending = None;
        }
        Ok(())
    }

    fn check_event(&self, event: &DeviceEvent) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        same_adapter(&self.lease, event.lease(), "event")?;
        let pending = self
            .pending
            .ok_or_else(|| refuse("the CUDA pinned buffer holds no copy to wait for"))?;
        let mark = event
            .mark()
            .ok_or_else(|| refuse("the CUDA event was never recorded"))?;
        if mark.0 != pending.0 {
            return Err(refuse(
                "the CUDA event was recorded on another stream than the pinned copy",
            ));
        }
        if mark.1 < pending.1 {
            return Err(refuse(
                "the CUDA event was recorded before the pinned copy was enqueued",
            ));
        }
        Ok(())
    }

    /// Copies `out.len()` bytes at `offset` into `out`.
    ///
    /// Returns `BadRequest` while a copy is outstanding (call [`PinnedBuf::wait`] first), for an
    /// empty `out` and for a range outside the buffer. Never reads memory a copy may be writing.
    pub(crate) fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        if self.pending.is_some() {
            return Err(refuse(
                "the CUDA pinned buffer has a copy outstanding: wait for an event recorded behind it first",
            ));
        }
        if out.is_empty() {
            return Err(refuse("the CUDA pinned read of an empty slice"));
        }
        resident::pinned_read_into(self.raw()?, offset, out).map_err(|_| {
            refuse(format!(
                "the CUDA pinned read of {} bytes at offset {offset} lies outside a pinned buffer of {} bytes",
                out.len(),
                self.bytes
            ))
        })
    }

    pub(in crate::kernels::cuda) fn raw(&self) -> Result<&RawPinned, EngError> {
        self.raw
            .as_ref()
            .ok_or_else(|| refuse("the CUDA pinned buffer was already freed"))
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        let bytes = self.bytes;
        if let Some(raw) = self.raw.take() {
            run_or_defer(move || release(raw, resident::pinned_free, &LIVE_PINNED, bytes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::CapturePlan;
    use crate::kernels::cuda::resident::matvec_q::tests::assert_bad_request;
    use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
    use crate::kernels::cuda::testkit;

    #[test]
    #[ignore = "requires native CUDA; run explicitly under the GPU lease"]
    fn pinned_buf_import_roundtrip_has_one_stream_fence() {
        assert!(testkit::device_present(
            "pinned_buf_import_roundtrip_has_one_stream_fence"
        ));
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let side = ctx.stream().expect("side stream");
        let (early, done) = (ctx.event().expect("early"), ctx.event().expect("done"));
        let data = pattern(64, 19);
        let device = testkit::upload_bytes(&ctx, &data);
        let output = ctx.allocate(64).expect("output");
        let mut host = ctx.pinned_buf(72).expect("pinned");
        ctx.copy_to_pinned(&stream, device.range(0, 64).expect("source"), &mut host, 8)
            .expect("D2H");
        stream.record(&early).expect("early record");
        assert_bad_request(
            ctx.copy_from_pinned(&side, &mut host, 8, output.range(0, 32).expect("range")),
            "cross-stream import",
            "another stream",
        );
        for offset in [0, 32] {
            ctx.copy_from_pinned(
                &stream,
                &mut host,
                8 + offset,
                output.range(offset, 32).expect("range"),
            )
            .expect("H2D");
        }
        assert_bad_request(host.wait(&early), "early event", "before the pinned copy");
        stream.record(&done).expect("record all copies");
        host.wait(&done).expect("one fence");
        let mut got = vec![0u8; 64];
        output.read_at(&stream, 0, &mut got).expect("read output");
        assert_eq!(got, data);
        println!("PINNED_IMPORT_ROUNDTRIP_PASS");
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    struct OwnedPinnedRoundTrip {
        expected: Vec<u8>,
        got: Vec<u8>,
        pending_after_queue: bool,
        pending_after_wait: bool,
        read_during_copy: Result<(), EngError>,
        write_during_copy: Result<(), EngError>,
        leading: [u8; 8],
        trailing: [u8; 8],
    }

    fn owned_pinned_round_trip() -> Result<OwnedPinnedRoundTrip, EngError> {
        let data = pattern(4096 + 64 + 32, 19);
        let zeros = vec![0u8; data.len()];
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let event = ctx.event()?;
        let device = ctx.allocate(data.len())?;
        let mut buf = ctx.pinned_buf(data.len() + 16)?;
        eprintln!("PKV_PINNED_H2D_REAL_DEVICE_READY bytes={}", buf.len());
        buf.write(8, &data)?;
        let dst = device.range(0, data.len())?;
        ctx.copy_from_pinned(&stream, &mut buf, 8, dst)?;
        let pending_after_queue = buf.is_pending();
        let mut sample = [0u8; 16];
        let read_during_copy = buf.read(8, &mut sample);
        let write_during_copy = buf.write(8, &zeros);
        stream.record(&event)?;
        buf.wait(&event)?;
        let pending_after_wait = buf.is_pending();
        buf.write(8, &zeros)?;
        let src = device.range(0, data.len())?;
        ctx.copy_to_pinned(&stream, src, &mut buf, 8)?;
        stream.record(&event)?;
        buf.wait(&event)?;
        let mut got = vec![0u8; data.len()];
        buf.read(8, &mut got)?;
        let mut leading = [0xFFu8; 8];
        let mut trailing = [0xFFu8; 8];
        buf.read(0, &mut leading)?;
        buf.read(data.len() + 8, &mut trailing)?;
        Ok(OwnedPinnedRoundTrip {
            expected: data,
            got,
            pending_after_queue,
            pending_after_wait,
            read_during_copy,
            write_during_copy,
            leading,
            trailing,
        })
    }

    #[test]
    fn owned_pinned_h2d_waits_before_host_reuse_and_round_trips() {
        if !testkit::device_present("owned_pinned_h2d_waits_before_host_reuse_and_round_trips") {
            return;
        }
        let observed = owned_pinned_round_trip().expect("owned pinned roundtrip device phase");
        assert!(observed.pending_after_queue);
        assert_bad_request(observed.read_during_copy, "read during H2D", "outstanding");
        assert_bad_request(
            observed.write_during_copy,
            "write during H2D",
            "outstanding",
        );
        assert!(!observed.pending_after_wait);
        assert_eq!(
            observed.got, observed.expected,
            "host reuse cannot change the device payload"
        );
        assert_eq!(observed.leading, [0u8; 8]);
        assert_eq!(observed.trailing, [0u8; 8]);
    }

    #[test]
    fn pinned_buf_round_trips_a_device_pattern() {
        if !testkit::device_present("pinned_buf_round_trips_a_device_pattern") {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let (stream, event) = (ctx.stream().expect("stream"), ctx.event().expect("event"));
        let data = pattern(4096 + 64 + 32, 7);
        let device = testkit::upload_bytes(&ctx, &data);
        let mut buf = ctx.pinned_buf(data.len() + 8).expect("pinned buffer");
        // Three copies into disjoint ranges, one event after all of them.
        for (from, len) in [(0usize, 4096usize), (4096, 64), (4160, 32)] {
            let src = device.range(from, len).expect("source range");
            ctx.copy_to_pinned(&stream, src, &mut buf, from)
                .expect("queue the copy");
        }
        assert!(buf.is_pending());
        stream.record(&event).expect("record behind the copies");
        buf.wait(&event).expect("the event covers the copies");
        assert!(!buf.is_pending());
        let mut got = vec![0u8; data.len()];
        buf.read(0, &mut got).expect("read the whole pattern");
        assert_eq!(got, data);
        let mut slack = [0xFFu8; 8];
        buf.read(data.len(), &mut slack).expect("read the slack");
        assert_eq!(slack, [0u8; 8], "memory no copy filled stays zeroed");
    }

    #[test]
    fn a_read_while_a_copy_is_outstanding_is_refused() {
        if !testkit::device_present("a_read_while_a_copy_is_outstanding_is_refused") {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let (early, late, never) = (
            ctx.event().expect("early"),
            ctx.event().expect("late"),
            ctx.event().expect("never recorded"),
        );
        let device = testkit::upload_bytes(&ctx, &pattern(256, 3));
        let mut buf = ctx.pinned_buf(256).expect("pinned buffer");
        let mut out = [0u8; 16];
        stream.record(&early).expect("record before the copy");
        let src = device.range(0, 256).expect("source");
        ctx.copy_to_pinned(&stream, src, &mut buf, 0).expect("copy");
        assert_bad_request(buf.read(0, &mut out), "a read during a copy", "outstanding");
        assert_bad_request(
            buf.wait(&early),
            "an event before the copy",
            "before the pinned copy",
        );
        assert_bad_request(
            buf.wait(&never),
            "an event never recorded",
            "never recorded",
        );
        assert!(buf.is_pending(), "the refusals left the copy outstanding");
        stream.record(&late).expect("record behind the copy");
        buf.wait(&late).expect("the late event releases the buffer");
        buf.read(0, &mut out).expect("the read after the wait");
        assert_eq!(out.to_vec(), pattern(256, 3)[..16].to_vec());
        assert_bad_request(
            buf.wait(&late),
            "a wait with nothing outstanding",
            "no copy to wait for",
        );
    }

    /// The stream guard of `wait` (MSG-79 soundness condition): an event recorded on ANOTHER stream
    /// proves nothing about a copy queued on this one, however many launches the other stream has seen,
    /// so it is refused and the copy stays outstanding; so is an event of another context. The event of
    /// the other stream is recorded five times so that its sequence number is past the copy's: without
    /// the stream check the sequence check would let it through, and the test would see a release.
    #[test]
    fn a_wait_on_an_event_of_another_stream_or_context_is_refused_and_keeps_the_copy_outstanding() {
        if !testkit::device_present(
            "a_wait_on_an_event_of_another_stream_or_context_is_refused_and_keeps_the_copy_outstanding",
        ) {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let other = CudaContext::new().expect("second context");
        let (copy_stream, side_stream) =
            (ctx.stream().expect("stream"), ctx.stream().expect("side"));
        let (on_side, on_copy) = (
            ctx.event().expect("side event"),
            ctx.event().expect("copy event"),
        );
        let foreign = other.event().expect("event of the other context");
        let foreign_stream = other.stream().expect("stream of the other context");
        let device = testkit::upload_bytes(&ctx, &pattern(256, 5));
        let mut buf = ctx.pinned_buf(256).expect("pinned buffer");
        let src = device.range(0, 256).expect("source");
        ctx.copy_to_pinned(&copy_stream, src, &mut buf, 0)
            .expect("copy");
        for _ in 0..5 {
            side_stream
                .record(&on_side)
                .expect("record on the side stream");
        }
        foreign_stream
            .record(&foreign)
            .expect("record on the foreign stream");
        assert!(
            on_side.mark().expect("recorded").1 > 1,
            "the side event's sequence is past the copy's, so only the stream check can refuse it"
        );
        assert_bad_request(
            buf.wait(&on_side),
            "an event of another stream",
            "another stream than the pinned copy",
        );
        assert_bad_request(
            buf.wait(&foreign),
            "an event of another context",
            "belongs to another",
        );
        assert!(buf.is_pending(), "both refusals left the copy outstanding");
        copy_stream
            .record(&on_copy)
            .expect("record behind the copy");
        buf.wait(&on_copy)
            .expect("the event of the copy's own stream releases the buffer");
        let mut out = [0u8; 16];
        buf.read(0, &mut out).expect("the read after the wait");
        assert_eq!(out.to_vec(), pattern(256, 5)[..16].to_vec());
    }

    #[test]
    fn pinned_buf_refuses_ranges_outside_either_buffer() {
        if !testkit::device_present("pinned_buf_refuses_ranges_outside_either_buffer") {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let (stream, other) = (ctx.stream().expect("stream"), ctx.stream().expect("other"));
        let device = testkit::upload_bytes(&ctx, &pattern(128, 1));
        let mut buf = ctx.pinned_buf(64).expect("pinned buffer");
        assert_bad_request(
            ctx.pinned_buf(0).map(|_| ()),
            "an empty buffer",
            "at least one byte",
        );
        let big = device.range(0, 65).expect("65 bytes");
        assert_bad_request(
            ctx.copy_to_pinned(&stream, big, &mut buf, 0),
            "a copy past the end",
            "lies outside",
        );
        let past = device.range(0, 64).expect("64 bytes");
        assert_bad_request(
            ctx.copy_to_pinned(&stream, past, &mut buf, 1),
            "a copy at a late offset",
            "lies outside",
        );
        assert_bad_request(
            buf.read(60, &mut [0u8; 8]),
            "a read past the end",
            "lies outside",
        );
        assert_bad_request(buf.read(0, &mut []), "an empty read", "empty slice");
        assert!(!buf.is_pending(), "no refusal queued anything");
        // One event covers one stream: a second stream cannot join an outstanding copy.
        let first = device.range(0, 32).expect("first");
        ctx.copy_to_pinned(&stream, first, &mut buf, 0)
            .expect("copy on the first stream");
        let second = device.range(32, 32).expect("second");
        assert_bad_request(
            ctx.copy_to_pinned(&other, second, &mut buf, 32),
            "a copy on another stream",
            "another stream",
        );
        stream.synchronize().expect("drain");
    }

    #[test]
    fn pinned_buf_copy_and_allocation_are_refused_inside_a_capture_window() {
        if !testkit::device_present(
            "pinned_buf_copy_and_allocation_are_refused_inside_a_capture_window",
        ) {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let mut buf = ctx.pinned_buf(64).expect("pinned buffer");
        let src = testkit::upload_bytes(&ctx, &pattern(64, 5));
        let x = testkit::upload_f32(&ctx, &[0.5f32; 256]);
        let q8_bytes = q8_1_bytes(256, 1).expect("size");
        let q8 = testkit::upload_filled(&ctx, 0, q8_bytes);
        let quantize = || {
            let call = QuantizeCall {
                x: x.range(0, 1024).expect("x"),
                y: q8.range(0, q8_bytes).expect("y"),
                k: 256,
                ncols: 1,
            };
            ctx.quantize_q8_1(&stream, call)
        };
        quantize().expect("warm the kernel: a first launch loads its module");
        stream.synchronize().expect("drain");
        let answers = std::cell::RefCell::new(Vec::new());
        let plan = CapturePlan {
            pins: &[],
            holds: Vec::new(),
        };
        let captured = ctx.capture(&stream, plan, || {
            quantize()?;
            let mut log = answers.borrow_mut();
            let copy = ctx.copy_to_pinned(&stream, src.range(0, 64)?, &mut buf, 0);
            log.push(format!("{copy:?}"));
            log.push(format!("{:?}", ctx.pinned_buf(8).map(|_| ())));
            Ok(())
        });
        captured.expect("the capture itself succeeds");
        for answer in answers.into_inner() {
            assert!(
                answer.contains("not allowed inside a capture window"),
                "{answer}"
            );
        }
        assert!(!buf.is_pending(), "the refused copy advanced nothing");
    }

    #[test]
    fn dropping_a_buffer_with_a_copy_in_flight_is_clean() {
        if !testkit::device_present("dropping_a_buffer_with_a_copy_in_flight_is_clean") {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let big = 32 << 20;
        let device = testkit::upload_filled(&ctx, 0x5A, big);
        let before = ctx.live_owners();
        let mut buf = ctx.pinned_buf(big).expect("pinned buffer");
        let src = device.range(0, big).expect("source");
        ctx.copy_to_pinned(&stream, src, &mut buf, 0)
            .expect("queue the big copy");
        // The native free waits for the whole device first, so the drop cannot free memory
        // the copy is still writing; lane L runs this test once under compute-sanitizer.
        drop(buf);
        assert_eq!(
            ctx.live_owners().pinned,
            before.pinned,
            "the bytes were released"
        );
        stream.synchronize().expect("the context is still usable");
        assert!(ctx.pinned_buf(8).is_ok(), "and still allocates");
    }

    #[test]
    fn pinned_buf_is_released_on_drop() {
        if !testkit::device_present("pinned_buf_is_released_on_drop") {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let (before, destroys) = (ctx.live_owners(), ctx.destroy_calls());
        let buf = ctx.pinned_buf(1000).expect("pinned buffer");
        assert_eq!(ctx.live_owners().pinned, before.pinned + 1000);
        assert_eq!(buf.len(), 1000);
        drop(buf);
        assert_eq!(ctx.live_owners(), before, "the owner released its count");
        assert_eq!(ctx.destroy_calls(), destroys + 1, "one native free");
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod h2d_tests;
#[cfg(all(test, feature = "native-cuda"))]
mod sync_tests;
#[cfg(all(test, feature = "native-cuda"))]
mod transfer_bench;
