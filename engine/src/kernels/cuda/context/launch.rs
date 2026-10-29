//! Launch-only pack and matmul entries on an owned stream.
//!
//! [`CudaContext::pack`] and [`CudaContext::matmul`] queue the vendored kernels
//! on a [`DeviceStream`] and return when they are queued. They allocate nothing,
//! copy nothing and never synchronize; a fault in the queued work surfaces at the
//! next synchronize of the stream (or of an event behind it) and latches the
//! backend there. The caller owns the order: `x` must hold its data before the
//! pack is queued, `xf` is written by the pack and read by the matmuls queued
//! behind it, and `y` is read only after the stream drained. The stream is always
//! explicit; there is no default-stream fallback.
//!
//! What each call queues, in order, on the stream it is given:
//! - `pack`: for `m <= 16` one `cudaMemsetAsync` that zeroes the whole
//!   fragment-order `xf` (the kernels read its padding positions and its last 16
//!   bytes), then the pack kernel. For `m > 16` the row-major `xf` is fully written
//!   by the kernel and is not zeroed.
//! - `matmul`: when the shape splits along k, one `cudaMemset2DAsync` that zeroes
//!   `y` for the atomic partial sums, then the kernel (for `m > 16` the tensor-core
//!   gemm).
//! - `matmul` of a Q5_0 weight: the own kernel of
//!   `native/resident/matmul_q5_0.cu`, one launch, no split along k, 1 to 16 tokens
//!   (more is a `BadRequest`; Q5_0 prefill is not supported). Its `xf` is the
//!   fragment-order buffer of a row rounded up to the 256 grid with a zero tail, the
//!   width `native::activation_k` gives, so `pack` takes that width as `k` and `x`
//!   holds `m` rows of it. The kernel never reads `xf` beyond the weight's `k`.
//!
//! Every range is a checked [`DeviceRange`]. Before any native call the wrappers
//! prove that the stream and every buffer belong to this context, that the shape
//! is one the kernels accept, that each range holds the bytes the call uses and is
//! aligned (x 4, xf 16, y 4, weight tiles 16), and that the range a launch writes
//! overlaps none it reads. `native::resident` and the C entries repeat these
//! checks, so a rejection is always a `BadRequest` here and never a terminal
//! adapter failure. A refused call changes nothing.

use super::stream::{DeviceStream, same_adapter};
use super::{
    CudaContext, DeviceBuffer, DeviceRange, TiledWeight, check_q5_0_tokens, device_error, row_range,
};
use crate::EngError;
use crate::kernels::cuda::native::resident::matmul_q5_0::matmul_q5_0_launch;
use crate::kernels::cuda::native::resident::{self, MatmulRaw, PackRaw, RawRange, RawStream};
use crate::kernels::cuda::native::{
    self, DeviceFailure, MAX_DIM, TILE_COLS, TileVec, Tiled, TiledType,
};
use crate::kernels::cuda::{ADAPTER_OPERATION, TERMINAL, native_error};

/// `x` ([m][k] f32) to the fp16 buffer `xf`.
pub(crate) struct PackCall<'b> {
    pub(crate) x: DeviceRange<'b>,
    pub(crate) xf: DeviceRange<'b>,
    pub(crate) m: usize,
    pub(crate) k: usize,
}

/// Rows `first_row..first_row + rows` of a tiled weight. A whole expert of a
/// stack is such a run.
pub(crate) struct Projection<'a> {
    pub(crate) weight: &'a TiledWeight,
    pub(crate) first_row: usize,
    pub(crate) rows: usize,
}

/// The packed input and the output of a matmul of `m` tokens. `y` holds `m` rows
/// of `rows` values (`ldy` is `rows`).
pub(crate) struct MatmulIo<'b> {
    pub(crate) xf: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) m: usize,
}

impl CudaContext {
    /// Queues the pack of `call.x` into `call.xf` on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for `m`
    /// outside `1..=32768`, for `k` that is not a multiple of 256 in
    /// `256..=32768`, for a range that is too short or misaligned, and for `x`
    /// and `xf` ranges that overlap.
    pub(crate) fn pack(&self, stream: &DeviceStream, call: PackCall<'_>) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_pack(&call)?;
        self.launch_on(stream, |stream_raw| resident::pack_launch(stream_raw, &raw))
    }

    /// Queues the matmul of `projection` on `stream`, reading `io.xf` and writing
    /// `io.y`.
    ///
    /// Returns `BadRequest` for the conditions of [`Self::pack`] that apply, for
    /// rows that are not whole 8-row groups inside the weight, and for a `y` range
    /// that overlaps the weight tiles or `xf`.
    pub(crate) fn matmul(
        &self,
        stream: &DeviceStream,
        projection: Projection<'_>,
        io: MatmulIo<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_matmul(&projection, &io)?;
        // The weight type picks the kernel; the checks above are the same for every type.
        self.launch_on(stream, |stream_raw| match raw.tiled.ty {
            TiledType::Q50 => matmul_q5_0_launch(stream_raw, &raw),
            TiledType::Q4K | TiledType::Q6K | TiledType::Q80 => {
                resident::matmul_launch(stream_raw, &raw)
            }
        })
    }

    /// Runs one native launch on `stream`, for the callers that queue work on it:
    /// the entries above and the kernel wrappers of `kernels::cuda::resident`.
    ///
    /// In order: the terminal check, the check that `stream` belongs to this
    /// context, the dispatch guard, `launch` with the raw stream, then one enqueue
    /// counted on the stream if the launch succeeded. A failed launch is mapped
    /// like every other native failure and counts nothing. The raw handle stays
    /// inside the native tree: `launch` borrows it for the call and cannot keep it.
    ///
    /// `launch` must queue its work and return, as every launch entry does, and
    /// the caller must have proved the arguments of the call before this one; a
    /// rejection from the native side is an adapter failure and latches terminal.
    pub(in crate::kernels::cuda) fn launch_on<R>(
        &self,
        stream: &DeviceStream,
        launch: impl FnOnce(&RawStream) -> Result<R, DeviceFailure>,
    ) -> Result<R, EngError> {
        self.ensure_usable()?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        let stream_raw = stream.raw()?;
        let _dispatch = self.enter()?;
        let queued = launch(stream_raw).map_err(device_error)?;
        stream.note_enqueue();
        Ok(queued)
    }

    /// Checks a stream without enqueueing a command or exposing its raw handle.
    pub(crate) fn check_launch_stream(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.ensure_usable()?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        stream.raw()?;
        Ok(())
    }

    /// The native view of `range`, for the kernel wrappers of
    /// `kernels::cuda::resident`.
    ///
    /// Returns `BadRequest` for a buffer of another adapter or one already freed.
    pub(in crate::kernels::cuda) fn raw_range<'b>(
        &self,
        range: &DeviceRange<'b>,
    ) -> Result<RawRange<'b>, EngError> {
        let device = self.owned(range.buffer())?;
        RawRange::new(device, range.offset(), range.bytes())
            .ok_or_else(|| EngError::BadRequest("the CUDA launch range is not valid".to_owned()))
    }

    /// Every check of a pack, then the native view of its ranges.
    fn plan_pack<'b>(&self, call: &PackCall<'b>) -> Result<PackRaw<'b>, EngError> {
        let x_raw = self.raw_range(&call.x)?;
        let xf_raw = self.raw_range(&call.xf)?;
        check_tokens(call.m)?;
        if !(1..=MAX_DIM).contains(&call.k) || !call.k.is_multiple_of(TILE_COLS) {
            return Err(EngError::BadRequest(format!(
                "the CUDA pack needs k as a multiple of {TILE_COLS} up to {MAX_DIM}, got {}",
                call.k
            )));
        }
        let (x_need, xf_need) = native::pack_bytes(call.m, call.k)
            .ok_or_else(|| EngError::BadRequest("the CUDA pack sizes overflow".to_owned()))?;
        require_fit("x", &call.x, x_need, 4)?;
        require_fit("xf", &call.xf, xf_need, 16)?;
        require_apart(
            &Claim::of("x", &call.x, x_need),
            &Claim::of("xf", &call.xf, xf_need),
        )?;
        Ok(PackRaw {
            x: x_raw,
            xf: xf_raw,
            m: call.m,
            k: call.k,
        })
    }

    /// Every check of a matmul, then the native view of its weight and ranges.
    fn plan_matmul<'b>(
        &self,
        projection: &Projection<'b>,
        io: &MatmulIo<'b>,
    ) -> Result<MatmulRaw<'b>, EngError> {
        let weight: &'b TiledWeight = projection.weight;
        let weight_device = self.owned(&weight.buffer)?;
        let xf_raw = self.raw_range(&io.xf)?;
        let y_raw = self.raw_range(&io.y)?;
        let (view, offset) = row_range(weight.tiled, projection.first_row, projection.rows)?;
        check_tokens(io.m)?;
        check_q5_0_tokens(view.ty, io.m)?;
        let k_x = native::activation_k(view.ty, view.k);
        let need = native::activation_bytes(view.n_rows, k_x, io.m)
            .map_err(|status| native_error(&TERMINAL, ADAPTER_OPERATION, status))?;
        let tile_bytes = view
            .tile_vecs()
            .and_then(|vecs| vecs.checked_mul(size_of::<TileVec>()))
            .ok_or_else(|| EngError::BadRequest("the CUDA weight size overflows".to_owned()))?;
        require_fit("xf", &io.xf, need.xf, 16)?;
        require_fit("y", &io.y, need.y, 4)?;
        let tiles = Claim {
            name: "weight",
            buffer: &weight.buffer,
            start: offset,
            bytes: tile_bytes,
        };
        let (xf, y) = (
            Claim::of("xf", &io.xf, need.xf),
            Claim::of("y", &io.y, need.y),
        );
        require_apart(&y, &xf)?;
        require_apart(&y, &tiles)?;
        require_apart(&xf, &tiles)?;
        Ok(MatmulRaw {
            tiled: view,
            weight: weight_device.at(offset),
            xf: xf_raw,
            y: y_raw,
            m: io.m,
        })
    }
}

impl TiledWeight {
    /// The tile shape of the weight, for the kernel wrappers of `kernels::cuda::resident`.
    pub(in crate::kernels::cuda) fn layout(&self) -> Tiled {
        self.tiled
    }
}

impl CudaContext {
    /// Test only: overwrites part of a device buffer and waits for the copy, so the ids of a
    /// launch can change between replays. `DeviceBuffer::write_at` is to replace it.
    #[cfg(test)]
    pub(in crate::kernels::cuda) fn overwrite(
        &self,
        range: DeviceRange<'_>,
        host: &[u8],
    ) -> Result<(), EngError> {
        if host.len() > range.bytes() {
            return Err(EngError::BadRequest(format!(
                "the CUDA overwrite of {} bytes does not fit a range of {} bytes",
                host.len(),
                range.bytes()
            )));
        }
        let raw = self.owned(range.buffer())?;
        let _dispatch = self.enter()?;
        native::upload_at(raw, range.offset(), host).map_err(device_error)
    }
}

fn check_tokens(m: usize) -> Result<(), EngError> {
    if (1..=MAX_DIM).contains(&m) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA launch takes 1 to {MAX_DIM} tokens, got {m}"
    )))
}

/// `BadRequest` unless `range` holds the `need` bytes the call uses and starts at
/// a multiple of `align`. Every allocation starts 256-byte aligned, so the offset
/// decides the alignment of the address.
pub(in crate::kernels::cuda) fn require_fit(
    name: &str,
    range: &DeviceRange<'_>,
    need: usize,
    align: usize,
) -> Result<(), EngError> {
    if range.bytes() < need {
        return Err(EngError::BadRequest(format!(
            "the CUDA {name} range holds {} bytes but the call needs {need}",
            range.bytes()
        )));
    }
    if !range.offset().is_multiple_of(align) {
        return Err(EngError::BadRequest(format!(
            "the CUDA {name} range starts at offset {}, which is not a multiple of {align}",
            range.offset()
        )));
    }
    Ok(())
}

/// Bytes of one allocation that a call reads or writes.
///
/// The kernel wrappers of `kernels::cuda::resident` build their claims with [`Claim::of`] and
/// compare them with [`require_apart`]; they must not keep a local copy of the overlap rule.
/// (The native children use `native::resident::Span` instead.)
pub(in crate::kernels::cuda) struct Claim<'a> {
    name: &'static str,
    buffer: &'a DeviceBuffer,
    start: usize,
    bytes: usize,
}

impl<'a> Claim<'a> {
    /// The claim on `bytes` bytes of `range`, named `name` in the rejection text.
    pub(in crate::kernels::cuda) fn of(
        name: &'static str,
        range: &DeviceRange<'a>,
        bytes: usize,
    ) -> Self {
        Self {
            name,
            buffer: range.buffer(),
            start: range.offset(),
            bytes,
        }
    }
}

/// `BadRequest` when the two claims share a byte of one allocation.
pub(in crate::kernels::cuda) fn require_apart(
    a: &Claim<'_>,
    b: &Claim<'_>,
) -> Result<(), EngError> {
    let apart = !std::ptr::eq(a.buffer, b.buffer)
        || a.start.saturating_add(a.bytes) <= b.start
        || b.start.saturating_add(b.bytes) <= a.start;
    if apart {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA {} and {} ranges overlap",
        a.name, b.name
    )))
}

#[cfg(test)]
mod tests;
