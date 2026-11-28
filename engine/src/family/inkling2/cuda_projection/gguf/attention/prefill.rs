//! Batch transfer and fences, with opt-in immutable-snapshot query tiles.

use super::*;
use crate::family::inkling2::switches::Switch;
use crate::{
    family::inkling2::cuda_projection::{DEVICE_HEADROOM, q8_row_reuse},
    family::step::{attend::relative, conv},
    kernels::cuda::resident::{
        relative_attention::{LogScale, QUERY_TILE_LIMIT, RelativeAttentionTile},
        relative_prep::{KvPrepShape, OutputConvShape, PrepShape},
    },
};

const ARENA_BYTES: usize = 32 << 20;
/// The boundary the output projection's input rows start on: the tensor-core prompt GEMM reads
/// its activation rows from 16-byte boundaries, and the status words before them are 4 bytes a row.
const PROJECTION_ALIGN: usize = 16;

pub(in crate::family::inkling2) fn enabled() -> bool {
    Switch::BatchRelativeAttention.on()
}

pub(super) fn query_tiles_enabled(context: &CudaContext) -> bool {
    enabled()
        && Switch::PrefillQueryTiles.on()
        // The kernels use SM75 tensor-core instructions.
        && context
            .device_attributes()
            .is_ok_and(|attrs| attrs.compute_capability >= (7, 5))
}

pub(in crate::family::inkling2) fn device_side_enabled() -> bool {
    Switch::PrefillAttentionSide.on()
}

/// Extends the ordinary batched relative-attention path; does not enable
/// batching, grouped controllers, or the resident attention cache on its own.
pub(in crate::family::inkling2) fn resident_query_enabled() -> bool {
    Switch::PrefillResidentQuery.on()
}

pub(in crate::family::inkling2) fn host_reserve() -> u64 {
    if enabled() {
        (2 * ARENA_BYTES) as u64
    } else {
        0
    }
}

pub(super) struct Workspace {
    device: DeviceBuffer,
    transfer: PinnedBuf,
    encoded: Vec<u8>,
}

impl Workspace {
    pub(super) fn allocate(context: &CudaContext) -> Result<Option<Self>, EngError> {
        if (context.memory_info()?.free as u64).saturating_sub(ARENA_BYTES as u64) < DEVICE_HEADROOM
        {
            return Ok(None);
        }
        let device = match context.allocate(ARENA_BYTES) {
            Ok(device) => device,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        if (context.memory_info()?.free as u64) < DEVICE_HEADROOM {
            return Ok(None);
        }
        let transfer = match context.pinned_buf(ARENA_BYTES) {
            Ok(transfer) => transfer,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(Self {
            device,
            transfer,
            encoded: Vec::with_capacity(ARENA_BYTES),
        }))
    }
}

#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct QueryClip<'a> {
    pub scales: [&'a [f32]; 2],
    pub limits: [f32; 2],
    /// The scale by position of a visit that attends over the whole context.
    pub log: Option<LogScale>,
}

#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct Sidecar<'a> {
    pub range: DeviceRange<'a>,
    pub query: Option<QueryClip<'a>>,
    /// Set when the rows hold raw K and V, which the device prepares before attention.
    pub kv: Option<&'a KvPrep<'a>>,
}

/// The host's inputs to the device preparation of a sub-tile's K and V, and its results.
pub(in crate::family::inkling2) struct KvPrep<'a> {
    pub kv_heads: usize,
    /// Raw history of each plane, newest first, at most three rows.
    pub history: [&'a VecDeque<Vec<f32>>; 2],
    /// Static taps channel-major, or lag-major static weights when `coefficients` is set.
    pub weights: [std::borrow::Cow<'a, [f32]>; 2],
    /// Per row, the K taps then the V taps, four for each head.
    pub coefficients: Option<Vec<f32>>,
    pub limit: f32,
    pub eps: f32,
    /// Whether the prepared rows come back to the host; the raw tail always does.
    pub read_rows: bool,
    pub prepared: std::cell::Cell<Option<KvPrepared>>,
}

pub(in crate::family::inkling2) struct KvPrepared {
    /// The prepared K and V of each row, empty unless `KvPrep::read_rows` asked for them.
    pub rows: Vec<[Vec<f32>; 2]>,
    /// The raw K and V of the last rows, at most three, oldest first.
    pub raw_tail: Vec<[Vec<f32>; 2]>,
}

#[derive(Clone, Copy)]
pub(super) struct ResidentOutput<'a> {
    pub raw: DeviceRange<'a>,
    pub convolution: Option<OutputConvolution<'a>>,
}

#[derive(Clone, Copy)]
pub(super) struct OutputConvolution<'a> {
    pub history: usize,
    // Raw history (right-aligned, oldest first), channel-major weights, output.
    pub planes: [DeviceRange<'a>; 3],
    pub read_convolved: bool,
}

/// The four buffers of a [`PrefillConv`]: raw rows after their history, convolved rows, weights
/// and row status. Every visit of a prompt chunk convolves planes of the same shape, so a finished
/// convolution hands its buffers to the next one instead of returning them to the allocator, whose
/// return synchronizes the device, and the next admission skips its allocations and memory queries.
pub(super) type ConvBuffers = [DeviceBuffer; 4];

/// Shapes of convolution buffers a prompt keeps at once: an attention and an expert plane are open
/// together, and a chunk of another size can follow.
pub(super) const CONV_POOL: usize = 4;

/// Owns the admitted whole visit. Adjacent raw tiles become convolution
/// history by slicing the same buffer, with no intermediate copies.
pub(in crate::family::inkling2) struct PrefillConv {
    raw: DeviceBuffer,
    convolved: DeviceBuffer,
    weights: DeviceBuffer,
    // One word per row, for the whole-plane convolution of an expert plane.
    status: DeviceBuffer,
    stride: usize,
    initial: usize,
    rows: usize,
    written: usize,
}

impl PrefillConv {
    const PAST: usize = PrepShape::TAPS - 1;

    /// Borrows only complete, status-validated convolved rows. The caller must
    /// finish consuming the plane before publishing the raw history tail.
    pub(in crate::family::inkling2) fn convolved_plane(&self) -> Result<DeviceRange<'_>, EngError> {
        if self.written != self.rows {
            return Err(bad("incomplete CUDA prefill convolution visit"));
        }
        self.convolved.range(0, self.convolved.bytes())
    }

    /// Full chronological raw plane, excluding the fixed history prefix.
    /// The caller must finish writing and checking every expert window first.
    pub(in crate::family::inkling2) fn expert_destination(
        &self,
    ) -> Result<DeviceRange<'_>, EngError> {
        Ok(self.destination(self.rows, false)?.raw)
    }

    fn destination(
        &self,
        count: usize,
        read_convolved: bool,
    ) -> Result<ResidentOutput<'_>, EngError> {
        if count == 0 || count > self.rows - self.written {
            return Err(bad("CUDA prefill convolution exceeds admitted rows"));
        }
        Ok(ResidentOutput {
            raw: self.raw.range(
                (Self::PAST + self.written) * self.stride,
                count * self.stride,
            )?,
            convolution: Some(OutputConvolution {
                history: (self.initial + self.written).min(Self::PAST),
                planes: [
                    self.raw
                        .range(self.written * self.stride, Self::PAST * self.stride)?,
                    self.weights.range(0, self.weights.bytes())?,
                    self.convolved
                        .range(self.written * self.stride, count * self.stride)?,
                ],
                read_convolved,
            }),
        })
    }
}

impl ResidentOutput<'_> {
    fn validate(self, work: &Workspace, layout: Layout, rows: usize) -> Result<(), EngError> {
        let Some(conv) = self.convolution else {
            if layout.projected
                && rows.checked_mul(layout.output) == Some(self.raw.bytes())
                && !std::ptr::eq(self.raw.buffer(), &work.device)
            {
                return Ok(());
            }
            return Err(bad("incorrect resident CUDA attention output destination"));
        };
        let shape = OutputConvShape {
            rows,
            width: layout.output / 4,
            history: conv.history,
        };
        if let Some([raw, past, weight, out, _]) = shape.words()
            && layout.projected
            && [self.raw, conv.planes[0], conv.planes[1], conv.planes[2]]
                .into_iter()
                .zip([raw, past, weight, out])
                .all(|(r, words)| {
                    r.bytes() == 4 * words
                        && r.offset().is_multiple_of(4)
                        && !std::ptr::eq(r.buffer(), &work.device)
                })
        {
            return Ok(());
        }
        Err(bad(
            "incorrect resident CUDA attention convolution destination",
        ))
    }
}

/// Appends the K/V preparation's raw history (right-aligned, oldest first, K then V), weights and
/// per-row taps after the query scales. Returns the rows of history.
fn encode_kv_prep(
    bytes: &mut Vec<u8>,
    kv: &KvPrep<'_>,
    layout: Layout,
    count: usize,
) -> Result<usize, EngError> {
    let width = layout.resident[1] / size_of::<f32>();
    let past = PrepShape::TAPS - 1;
    let history = kv.history[0].len();
    let heads_taps = 2 * PrepShape::TAPS * kv.kv_heads;
    if !layout.raw_kv
        || kv.kv_heads * PrepShape::WIDTH != width
        || kv.history[1].len() != history
        || history > past
        || kv
            .history
            .iter()
            .flat_map(|h| h.iter())
            .any(|row| row.len() != width)
        || kv
            .weights
            .iter()
            .any(|w| w.len() != PrepShape::TAPS * width)
        || kv
            .coefficients
            .as_ref()
            .is_some_and(|c| c.len() != count * heads_taps)
    {
        return Err(bad("incorrect CUDA K/V preparation inputs"));
    }
    for plane in kv.history {
        bytes.resize(bytes.len() + 4 * width * (past - history), 0);
        for row in plane.iter().rev() {
            extend_words(bytes, row);
        }
    }
    for weights in &kv.weights {
        extend_words(bytes, weights);
    }
    if let Some(coefficients) = &kv.coefficients {
        extend_words(bytes, coefficients);
    }
    Ok(history)
}

/// The prepared K and V of every row and the raw tail, read from the pinned readback.
fn read_kv_prep(
    transfer: &PinnedBuf,
    layout: Layout,
    offsets: &[[usize; 8]],
    read_rows: bool,
) -> Result<KvPrepared, EngError> {
    let count = offsets.len();
    let width = layout.resident[1] / size_of::<f32>();
    let rows_at = count * (4 + layout.output);
    let tail_at = rows_at + count * layout.input;
    let read = |at: usize| {
        let mut row = vec![0.0; width];
        transfer.read_f32(at, &mut row)?;
        if row.iter().all(|x| x.is_finite()) {
            Ok(row)
        } else {
            Err(bad("non-finite CUDA K/V preparation row"))
        }
    };
    let rows = offsets
        .iter()
        .filter(|_| read_rows)
        .map(|row| Ok([read(rows_at + row[1])?, read(rows_at + row[2])?]))
        .collect::<Result<_, EngError>>()?;
    let tail = count.min(PrepShape::TAPS - 1);
    let stride = layout.side_stride();
    let k = layout.resident[0];
    let raw_tail = (0..tail)
        .map(|j| {
            let at = tail_at + j * stride + k;
            Ok([read(at)?, read(at + layout.resident[1])?])
        })
        .collect::<Result<_, EngError>>()?;
    Ok(KvPrepared { rows, raw_tail })
}

/// Rows of `stride` bytes of native-endian floats, refused when any value is not finite.
fn finite_rows(bytes: &[u8], stride: usize, error: &str) -> Result<Vec<Vec<f32>>, EngError> {
    bytes
        .chunks_exact(stride)
        .map(|row| {
            let row: Vec<_> = row
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_ne_bytes(*w))
                .collect();
            row.iter()
                .all(|x| x.is_finite())
                .then_some(row)
                .ok_or_else(|| bad(error))
        })
        .collect()
}

/// The projection outputs that stay on the device: sinks and gates always, Q and relative
/// coefficients when the query is clipped there, raw K and V when the device prepares them.
pub(in crate::family::inkling2) fn projection_side(query: bool, kv: bool) -> &'static [usize] {
    match (query, kv) {
        (true, true) => &[0, 1, 2, 3, 4, 5],
        (true, false) => &[0, 3, 4, 5],
        (false, true) => &[1, 2, 4, 5],
        (false, false) => &[4, 5],
    }
}

#[derive(Clone, Copy)]
struct Layout {
    input: usize,
    scales: usize,
    resident: [usize; 6],
    attention: usize,
    output: usize,
    projected: bool,
    // The side rows hold raw K and V; the host rows keep slots for the prepared ones.
    raw_kv: bool,
}

impl Layout {
    fn new(
        shape: RelativeAttentionShape,
        output_rows: Option<usize>,
        keep: &[usize],
    ) -> Option<Self> {
        let [attention, _, _, _, q, k, v, relative, _, sinks, gates, pos] = shape.region_bytes()?;
        let widths = [q, k, v, relative, sinks, gates];
        let mut resident = [0; 6];
        for &i in keep {
            resident[i] = widths[i];
        }
        let raw_kv = resident[1] != 0 && resident[2] != 0;
        if raw_kv != (resident[1] != 0 || resident[2] != 0) {
            return None;
        }
        let slots = if raw_kv { k + v } else { 0 };
        Some(Self {
            input: widths.iter().sum::<usize>() + pos - resident.iter().sum::<usize>() + slots,
            scales: resident[0] + resident[3],
            resident,
            attention,
            output: output_rows.map_or(Some(attention), |n| n.checked_mul(4))?,
            projected: output_rows.is_some(),
            raw_kv,
        })
    }

    fn side_stride(self) -> usize {
        self.resident.iter().sum()
    }

    /// Which inputs the host rows leave out because the side rows carry them for attention.
    fn side_slots(self) -> [bool; 6] {
        std::array::from_fn(|i| self.resident[i] != 0 && !(self.raw_kv && (i == 1 || i == 2)))
    }

    /// Words of the K/V preparation's history and weights, and of each row's taps.
    fn kv_prep_words(self) -> [usize; 2] {
        if !self.raw_kv {
            return [0, 0];
        }
        let kv = self.resident[1] / size_of::<f32>();
        let taps = PrepShape::TAPS;
        [2 * (2 * taps - 1) * kv, 2 * taps * (kv / PrepShape::WIDTH)]
    }

    fn row_bytes(self) -> Option<usize> {
        let projected = if self.projected { self.attention } else { 0 };
        let taps = 4 * self.kv_prep_words()[1];
        (self.input + 4 + projected + taps).checked_add(self.output)
    }

    /// Bytes every tile needs besides its rows: query scales, the K/V preparation's history
    /// and weights, the readback of the last three raw side rows, and the pad that puts the
    /// projection's input rows on [`PROJECTION_ALIGN`].
    fn fixed_bytes(self) -> usize {
        let tail = if self.raw_kv {
            (PrepShape::TAPS - 1) * self.side_stride()
        } else {
            0
        };
        let pad = if self.projected { PROJECTION_ALIGN } else { 0 };
        self.scales + 4 * self.kv_prep_words()[0] + tail + pad
    }

    /// Where the attention rows of a `count`-row tile start, after its `count` output rows at
    /// `out_at`: on [`PROJECTION_ALIGN`] when a projection reads them, else at `out_at`, where the
    /// rows are the output.
    fn attention_at(self, out_at: usize, count: usize) -> usize {
        if self.projected {
            (out_at + self.output * count).next_multiple_of(PROJECTION_ALIGN)
        } else {
            out_at
        }
    }

    fn capacity(self, query_tiles: bool) -> usize {
        let Some(row_bytes) = self.row_bytes() else {
            return 0;
        };
        // Keep the combined tip's batching when resident queries are off.
        let limit = if query_tiles || self.scales == 0 {
            QUERY_TILE_LIMIT
        } else {
            512 // resident clip launch limit
        };
        limit
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS)
            .min(ARENA_BYTES.saturating_sub(self.fixed_bytes()) / row_bytes)
    }

    fn row_inputs<'a>(
        self,
        work: &'a DeviceBuffer,
        offsets: [usize; 8],
        widths: [usize; 6],
        side: Option<&Sidecar<'a>>,
        row: usize,
    ) -> Result<[DeviceRange<'a>; 6], EngError> {
        let mut side_at = side.map_or(0, |s| s.range.offset()) + row * self.side_stride();
        let side_slots = self.side_slots();
        let mut next = |i: usize| {
            if side_slots[i] {
                let side = side.ok_or_else(|| bad("missing CUDA attention sidecar"))?;
                let relative_at = side_at - side.range.offset();
                if relative_at
                    .checked_add(widths[i])
                    .is_none_or(|end| end > side.range.bytes())
                {
                    return Err(bad("incorrect CUDA attention sidecar extent"));
                }
                let range = side.range.buffer().range(side_at, widths[i]);
                side_at += self.resident[i];
                range
            } else {
                // Raw K and V columns of the side rows are the preparation's input only.
                side_at += self.resident[i];
                work.range(offsets[i], widths[i])
            }
        };
        Ok([next(0)?, next(1)?, next(2)?, next(3)?, next(4)?, next(5)?])
    }
}

impl ResidentAttention {
    pub(in crate::family::inkling2) fn prefill_capacity(
        &self,
        context: &CudaContext,
        slot: usize,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<usize, EngError> {
        self.prefill_tile(context, slot, projection, &[])
            .map(|(capacity, _)| capacity)
    }

    pub(in crate::family::inkling2) fn prefill_tile(
        &self,
        context: &CudaContext,
        slot: usize,
        projection: Option<&OutputProjection<'_>>,
        keep: &[usize],
    ) -> Result<(usize, usize), EngError> {
        if !enabled() {
            return Ok((0, 0));
        }
        let Some(layout) = Layout::new(self.visits[slot].shape, projection.map(|p| p.rows), keep)
        else {
            return Ok((0, 0));
        };
        let mut workspace = self.prefill.borrow_mut();
        if workspace.is_none() {
            *workspace = Workspace::allocate(context)?;
        }
        Ok((
            workspace
                .as_ref()
                .map_or(0, |_| layout.capacity(self.query_tiles)),
            layout.side_stride(),
        ))
    }

    pub(in crate::family::inkling2) fn step_prefill(
        &self,
        context: &CudaContext,
        slot: usize,
        inputs: (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.step_prefill_dispatch(context, slot, inputs, history, projection, self.query_tiles)
    }

    pub(in crate::family::inkling2) fn admit_prefill_conv(
        &self,
        context: &CudaContext,
        shape: [usize; 2],
        prior: &VecDeque<Vec<f32>>,
        weights: &[f32],
    ) -> Result<Option<PrefillConv>, EngError> {
        let [rows, width] = shape;
        let initial = prior.len();
        let Some([plane, past, taps, _, _]) = (OutputConvShape {
            rows,
            width,
            history: initial,
        })
        .words() else {
            return Ok(None);
        };
        if weights.len() != taps || prior.iter().any(|row| row.len() != width) {
            return Err(bad("invalid CUDA prefill convolution weights or history"));
        }
        let sizes = [4 * (plane + past), 4 * plane, 4 * taps, 4 * rows];
        let Some([raw, convolved, kernel, status]) =
            (self.conv_pool).take_or_allocate(context, sizes, DEVICE_HEADROOM)?
        else {
            return Ok(None);
        };
        let mut upload = vec![0; 4 * (past - initial * width)];
        for row in prior.iter().rev() {
            extend_words(&mut upload, row);
        }
        // Queued: only kernels on this stream read the history and the weights, and a pageable
        // source is consumed before the call returns.
        raw.write_at_enqueue(&self.stream, 0, &upload)?;
        upload.clear();
        extend_words(&mut upload, weights);
        kernel.write_at_enqueue(&self.stream, 0, &upload)?;
        Ok(Some(PrefillConv {
            raw,
            convolved,
            weights: kernel,
            status,
            stride: 4 * width,
            initial,
            rows,
            written: 0,
        }))
    }

    pub(in crate::family::inkling2) fn step_prefill_conv(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: &OutputProjection<'_>,
        (conv, read_convolved): (&mut PrefillConv, bool),
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let destination = conv.destination(rows.len(), read_convolved)?;
        let output = self.step_prefill_to(
            context,
            slot,
            (rows, side),
            history,
            Some(projection),
            (self.query_tiles, Some(destination)),
        )?;
        conv.written += rows.len();
        Ok(output)
    }

    /// Consume a checked, fully written raw expert plane. Publish nothing on failure.
    pub(in crate::family::inkling2) fn finish_prefill_expert_conv(
        &self,
        context: &CudaContext,
        conv: &mut PrefillConv,
    ) -> Result<(), EngError> {
        let status = conv.status.range(0, conv.status.bytes())?;
        let destination = conv.destination(conv.rows, false)?;
        let convolution = destination
            .convolution
            .expect("prefill convolution destination");
        let [past, weight, output] = convolution.planes;
        let mut health = vec![0; conv.rows * 4];
        if status.bytes() != health.len() {
            return Err(bad("incorrect CUDA expert convolution status extent"));
        }
        status
            .buffer()
            .zero_at(&self.stream, status.offset(), health.len())?;
        conv::rows(
            context,
            &self.stream,
            conv::Variant::Residual,
            OutputConvShape {
                rows: conv.rows,
                width: conv.stride / 4,
                history: convolution.history,
            },
            conv::Rows::from_planes([destination.raw, past, weight, output, status]),
        )?;
        status
            .buffer()
            .read_at(&self.stream, status.offset(), &mut health)?;
        if health.iter().any(|&byte| byte != 0) {
            return Err(bad("non-finite CUDA Inkling expert convolution"));
        }
        conv.written = conv.rows;
        Ok(())
    }

    /// [`Self::finish_prefill_expert_conv`], then the convolved rows in order. Publish nothing
    /// on failure.
    pub(in crate::family::inkling2) fn finish_prefill_expert_conv_rows(
        &self,
        context: &CudaContext,
        conv: &mut PrefillConv,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.finish_prefill_expert_conv(context, conv)?;
        let mut bytes = vec![0; conv.rows * conv.stride];
        conv.convolved.read_at(&self.stream, 0, &mut bytes)?;
        finite_rows(
            &bytes,
            conv.stride,
            "non-finite CUDA Inkling expert convolution",
        )
    }

    /// The raw rows the next visit's convolution takes as history, newest first. The caller must
    /// have synchronized every stream other than this one's that reads the visit's planes: the
    /// buffers go back to the pool for the next admission.
    pub(in crate::family::inkling2) fn prefill_conv_tail(
        &self,
        conv: PrefillConv,
    ) -> Result<VecDeque<Vec<f32>>, EngError> {
        if conv.written != conv.rows {
            return Err(bad("incomplete CUDA prefill convolution visit"));
        }
        let count = (conv.written + conv.initial).min(PrefillConv::PAST);
        let mut bytes = vec![0; count * conv.stride];
        let first = PrefillConv::PAST + conv.written - count;
        conv.raw
            .read_at(&self.stream, first * conv.stride, &mut bytes)?;
        let rows = finite_rows(
            &bytes,
            conv.stride,
            "non-finite CUDA prefill convolution history",
        )?;
        // The read above drained this stream and the caller synchronized every other stream that
        // read the planes, so no queued work still touches the buffers the next visit reuses.
        (self.conv_pool).put([conv.raw, conv.convolved, conv.weights, conv.status]);
        Ok(rows.into_iter().rev().collect())
    }

    pub(super) fn step_prefill_dispatch(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
        query_tiles: bool,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.step_prefill_to(
            context,
            slot,
            (rows, side),
            history,
            projection,
            (query_tiles, None),
        )
    }

    /// A supplied destination is owned by the visit, not the reused attention tile.
    /// The kernel validates convolved output through the attention status plane.
    /// Raw-only consumers must check finite values before publishing the visit.
    pub(super) fn step_prefill_to(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
        (query_tiles, destination): (bool, Option<ResidentOutput<'_>>),
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let visit = self.visits[slot];
        let keep = side.as_ref().map_or(&[][..], |s| {
            projection_side(s.query.is_some(), s.kv.is_some())
        });
        let layout = Layout::new(visit.shape, projection.map(|p| p.rows), keep)
            .ok_or_else(|| bad("invalid CUDA relative attention batch layout"))?;
        if let Some(side) = &side
            && rows.len().checked_mul(layout.side_stride()) != Some(side.range.bytes())
        {
            return Err(bad("incorrect CUDA attention sidecar extent"));
        }
        let count = rows.len();
        let position = history.position;
        if count == 0
            || count > layout.capacity(query_tiles)
            || position
                .checked_add(count - 1)
                .is_none_or(|last| last > POS_MAX as usize)
        {
            return Err(bad("invalid CUDA relative attention batch extent"));
        }
        // No additional workspace. Require resident output projection so the same
        // fenced transaction covers it; singleton tails use the ordinary launch.
        let tile = if query_tiles && projection.is_some() {
            let tile = RelativeAttentionTile {
                first: position,
                count,
                input_stride: layout.input,
                query_stride: if layout.scales != 0 {
                    layout.side_stride()
                } else {
                    layout.input
                },
                side_stride: if side.is_some() {
                    layout.side_stride()
                } else {
                    layout.input
                },
            };
            tile.region_bytes_for(visit.shape, visit.kv_format)
                .map(|bytes| (tile, bytes))
        } else {
            None
        };
        if tile.is_some() {
            context.ensure_uncaptured("an Inkling attention prefill tile")?;
            context.check_launch_stream(&self.stream)?;
        }
        let mut workspace = self.prefill.borrow_mut();
        let work = workspace
            .as_mut()
            .ok_or_else(|| bad("CUDA relative attention batch lacks workspace"))?;
        if let Some(dest) = destination {
            dest.validate(work, layout, count)?;
        }
        work.encoded.clear();
        let kv = side.as_ref().and_then(|s| s.kv);
        // With device K/V preparation and resident queries the rows hold no host bytes: the
        // preparation writes their K/V and positions, and the upload starts at the scales.
        let mut offsets = Vec::with_capacity(count);
        if kv.is_some() {
            for (i, row) in rows.iter().enumerate() {
                match row.device_offsets(visit.shape, i * layout.input, layout.side_slots())? {
                    Some(row) => offsets.push(row),
                    None => break,
                }
            }
        }
        let device_rows = offsets.len() == count;
        if !device_rows {
            offsets.clear();
            for (i, row) in rows.iter().enumerate() {
                offsets.push(row.append_inputs(
                    visit.shape,
                    (position + i) as i32,
                    &mut work.encoded,
                    (layout.side_slots(), kv.is_some()),
                )?);
            }
        }
        if let Some(query) = side.as_ref().and_then(|s| s.query) {
            for (scale, i) in query.scales.into_iter().zip([0, 3]) {
                if scale.len() != layout.resident[i] / size_of::<f32>() {
                    return Err(bad("incorrect CUDA attention query scale extent"));
                }
                extend_words(&mut work.encoded, scale);
            }
        }
        let kv_history = match kv {
            Some(kv) => Some(encode_kv_prep(&mut work.encoded, kv, layout, count)?),
            None => None,
        };
        self.invalidate_local_aux(slot);
        let mut state = self.state.borrow_mut();
        let cursor = state.next_position[slot].take();
        // Ordinary grouped prefill never runs on a speculative clone. Reset and
        // prefix replacement invalidate the visit cursors before replacing host
        // state. Within this lineage an equal cursor is certified; never pass a
        // mismatched one to restore_rows's incremental rewind path here.
        self.restore_rows(
            visit,
            &history,
            cursor.filter(|&cursor| cursor == position),
            &mut state.encoded,
        )?;
        let result = self.execute_prefill(
            context,
            visit,
            layout,
            work,
            (
                &offsets,
                side,
                kv.zip(kv_history),
                device_rows.then_some(position),
                position,
            ),
            (projection, tile, destination),
        )?;
        // The ring is independent of the projection output. Publish its cursor
        // after the fence and attention status (and host output check, if read).
        state.next_position[slot] = Some(position + count);
        if tile.is_some() {
            static TILED: std::sync::Once = std::sync::Once::new();
            TILED.call_once(|| eprintln!("Inkling CUDA used relative attention query tiles"));
        }
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used batched relative attention"));
        Ok(result)
    }

    #[allow(clippy::type_complexity)]
    fn execute_prefill(
        &self,
        context: &CudaContext,
        visit: Visit,
        layout: Layout,
        work: &mut Workspace,
        (offsets, side, kv, first, position): (
            &[[usize; 8]],
            Option<Sidecar<'_>>,
            Option<(&KvPrep<'_>, usize)>,
            Option<usize>,
            usize,
        ),
        (projection, tile, destination): (
            Option<&OutputProjection<'_>>,
            Option<(RelativeAttentionTile, [usize; 12])>,
            Option<ResidentOutput<'_>>,
        ),
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let count = offsets.len();
        let convolved_readback = destination
            .and_then(|d| d.convolution)
            .filter(|c| c.read_convolved)
            .map(|c| c.planes[2]);
        let scales_at = layout.input * count;
        let prep_at = scales_at + layout.scales;
        let [fixed_words, tap_words] = layout.kv_prep_words();
        let prep_words = if kv.is_some_and(|(kv, _)| kv.coefficients.is_some()) {
            fixed_words + count * tap_words
        } else {
            fixed_words
        };
        let status_at = prep_at + 4 * prep_words;
        let out_at = status_at + 4 * count;
        let attention_at = layout.attention_at(out_at, count);
        // `first` when the device writes the rows: the host image starts at the scales.
        let uploaded = if first.is_some() { scales_at } else { 0 };
        debug_assert_eq!(uploaded + work.encoded.len(), status_at);
        work.encoded.resize(out_at - uploaded, 0);
        work.transfer.write(0, &work.encoded)?;
        let device = &work.device;
        let score_row = |i: usize, bytes: [usize; 12]| {
            let [out, _, _, status, q, k, v, rel, _, sinks, gates, pos] = bytes;
            Ok::<_, EngError>(relative::ScoreRow {
                rows: relative::Scored::from_order(layout.row_inputs(
                    device,
                    offsets[i],
                    [q, k, v, rel, sinks, gates],
                    side.as_ref(),
                    i,
                )?),
                pos: device.range(offsets[i][6], pos)?,
                status: device.range(status_at + 4 * i, status)?,
                out: device.range(attention_at + i * layout.attention, out)?,
            })
        };
        let queued = (|| {
            let status_all = device.range(status_at, 4 * count)?;
            context.copy_from_pinned(
                &self.stream,
                &mut work.transfer,
                0,
                device.range(uploaded, out_at - uploaded)?,
            )?;
            let query = side
                .as_ref()
                .and_then(|s| s.query.map(|q| (s.range, q)))
                .map(|(side, q)| {
                    Ok::<_, EngError>(relative::QueryClipRows {
                        shape: visit.shape,
                        rows: count,
                        raw_kv_bytes: layout.resident[1] + layout.resident[2],
                        limits: q.limits,
                        log: q.log.map(|log| (log, position)),
                        side,
                        scales: device.range(scales_at, layout.scales)?,
                        status: status_all,
                    })
                })
                .transpose()?;
            let kv_prep = kv
                .map(|(kv, history)| {
                    let side = side
                        .as_ref()
                        .ok_or_else(|| bad("K/V preparation lacks side rows"))?;
                    let kv_bytes = layout.resident[1];
                    let [k_at, v_at, pos_at] = [offsets[0][1], offsets[0][2], offsets[0][6]];
                    if offsets.iter().enumerate().any(|(i, r)| {
                        [r[1], r[2], r[6]] != [k_at, v_at, pos_at].map(|at| at + i * layout.input)
                    }) {
                        return Err(bad("CUDA K/V preparation rows are not evenly spaced"));
                    }
                    let past = 2 * 3 * kv_bytes;
                    Ok(relative::KvPrepRows {
                        shape: KvPrepShape {
                            rows: count,
                            kv_heads: kv.kv_heads,
                            history,
                            dynamic: kv.coefficients.is_some(),
                            raw_stride: layout.side_stride() / 4,
                            raw: [layout.resident[0] / 4, (layout.resident[0] + kv_bytes) / 4],
                            out_stride: layout.input / 4,
                            out: [k_at / 4, v_at / 4],
                            positions: first.map(|first| (pos_at / 4, first)),
                            limit: kv.limit,
                            eps: kv.eps,
                        },
                        raw: side.range,
                        history: device.range(prep_at, past)?,
                        taps: device.range(prep_at + past, 4 * prep_words - past)?,
                        out: device.range(0, count * layout.input)?,
                        status: status_all,
                    })
                })
                .transpose()?;
            relative::prep_rows(
                context,
                &self.stream,
                relative::PrepRows { query, kv: kv_prep },
            )?;
            // Prep ORs into the same status checked by snapshot commit. No
            // resident sidecar is replayed or re-prepared after enqueue.
            let scored = match tile {
                Some((tile, bytes)) => relative::ScoredRows::Tile {
                    tile,
                    row: Box::new(score_row(0, bytes)?),
                },
                None => {
                    let bytes = visit.shape.region_bytes_for(visit.kv_format).unwrap();
                    let calls = (0..count)
                        .map(|i| score_row(i, bytes))
                        .collect::<Result<_, _>>()?;
                    relative::ScoredRows::PerRow(calls)
                }
            };
            let rings = visit.rings(&self.arena)?;
            relative::attend_rows(
                context,
                &self.stream,
                relative::AttendRows { rings, scored },
            )?;
            if let Some(p) = projection {
                let y = match destination {
                    Some(dest) => dest.raw,
                    None => device.range(out_at, count * layout.output)?,
                };
                q8_row_reuse::project_prompt(
                    context,
                    &self.stream,
                    &self.tensor_core,
                    &[DenseGemvCall {
                        shape: DenseGemvShape::packed(count, p.rows, layout.attention / 4),
                        dtype: p.dtype,
                        w: p.weight,
                        x: device.range(attention_at, count * layout.attention)?,
                        y,
                    }],
                    q8_row_reuse::Exact::Output,
                )?;
            }
            if let Some(dest) = destination
                && let Some(c) = dest.convolution
            {
                let [past, weight, output] = c.planes;
                let shape = OutputConvShape {
                    rows: count,
                    width: layout.output / 4,
                    history: c.history,
                };
                conv::rows(
                    context,
                    &self.stream,
                    conv::Variant::Residual,
                    shape,
                    conv::Rows::from_planes([dest.raw, past, weight, output, status_all]),
                )?;
            }
            if let Some(convolved) = convolved_readback {
                context.copy_to_pinned(&self.stream, convolved, &mut work.transfer, 4 * count)?;
            }
            let readback = if destination.is_some() {
                4 * count
            } else {
                count * (4 + layout.output)
            };
            context.copy_to_pinned(
                &self.stream,
                work.device.range(status_at, readback)?,
                &mut work.transfer,
                0,
            )?;
            if let Some((kv, _)) = kv {
                let side = side
                    .as_ref()
                    .ok_or_else(|| bad("K/V preparation lacks side rows"))?;
                let rows_at = count * (4 + layout.output);
                if kv.read_rows {
                    context.copy_to_pinned(
                        &self.stream,
                        work.device.range(0, count * layout.input)?,
                        &mut work.transfer,
                        rows_at,
                    )?;
                }
                let tail = count.min(PrepShape::TAPS - 1);
                let stride = layout.side_stride();
                context.copy_to_pinned(
                    &self.stream,
                    side.range.slice((count - tail) * stride, tail * stride)?,
                    &mut work.transfer,
                    rows_at + count * layout.input,
                )?;
            }
            Ok(())
        })();
        // A later enqueue can fail after queries or commit were accepted. Drain
        // before the borrowed sidecar or pinned workspace can be reused. Do not
        // retry serially on a possibly dirty ring.
        work.transfer
            .synchronize_after_enqueue(&self.stream, queued)?;
        let mut status = vec![0; 4 * count];
        work.transfer.read(0, &mut status)?;
        if status.into_iter().any(|byte| byte != 0) {
            return Err(bad("invalid CUDA relative attention batch state"));
        }
        if let Some((kv, _)) = kv {
            kv.prepared.set(Some(read_kv_prep(
                &work.transfer,
                layout,
                offsets,
                kv.read_rows,
            )?));
        }
        if destination.is_some() && convolved_readback.is_none() {
            return Ok(Vec::new());
        }
        (0..count)
            .map(|i| {
                let mut output = vec![0.0; layout.output / 4];
                work.transfer
                    .read_f32(4 * count + layout.output * i, &mut output)?;
                if !output.iter().all(|v| v.is_finite()) {
                    return Err(bad("non-finite CUDA relative attention batch output"));
                }
                Ok(output)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_query_layout_leaves_only_kv_and_position_in_host_rows() {
        let shape = RelativeAttentionShape {
            query_heads: 64,
            kv_heads: 8,
            relative_dims: 32,
            relative_extent: 1024,
            capacity: 4096,
        };
        let words = |n| n * size_of::<f32>();
        let q = words(64 * 128);
        let kv = words(8 * 128);
        let relative = words(64 * 32);
        let sinks = words(64);
        let pos = size_of::<i32>();
        let projected = Some(16384);
        let host = Layout::new(shape, projected, &[]).unwrap();
        let side = Layout::new(shape, projected, projection_side(false, false)).unwrap();
        let query = Layout::new(shape, projected, projection_side(true, false)).unwrap();
        assert_eq!((host.side_stride(), host.scales), (0, 0));
        assert_eq!((side.side_stride(), side.scales), (sinks + q, 0));
        assert_eq!(query.resident, [q, 0, 0, relative, sinks, q]);
        assert_eq!(query.input, 2 * kv + pos);
        assert_eq!(query.scales, q + relative);
        assert_eq!(query.input + query.side_stride(), host.input);
        for layout in [host, side, query] {
            for tiled in [false, true] {
                let limit = if tiled || layout.scales == 0 {
                    QUERY_TILE_LIMIT
                } else {
                    512
                };
                let capacity = layout.capacity(tiled);
                let row_bytes = layout.row_bytes().unwrap();
                assert!(layout.fixed_bytes() + capacity * row_bytes <= ARENA_BYTES);
                assert!(
                    capacity == limit.min(DenseGemvShape::MAX_ACTIVATION_ROWS)
                        || layout.fixed_bytes() + (capacity + 1) * row_bytes > ARENA_BYTES
                );
            }
        }
        assert!(query.capacity(false) > side.capacity(false));
        assert!(side.capacity(false) > host.capacity(false));
    }

    #[test]
    fn projected_attention_rows_start_on_the_gemm_boundary() {
        let shape = RelativeAttentionShape {
            query_heads: 24,
            kv_heads: 8,
            relative_dims: 16,
            relative_extent: 1024,
            capacity: 4096,
        };
        let layout = Layout::new(shape, Some(2304), projection_side(true, true)).unwrap();
        assert!(layout.projected);
        // A 511-row prompt runs as tiles of 171, 171 and 169 rows: 4 status bytes a row put the
        // output rows, and the attention rows after them, off a 16-byte boundary.
        for count in [1usize, 169, 170, 171, 172] {
            let prep_words = layout.kv_prep_words()[0] + count * layout.kv_prep_words()[1];
            let status_at = layout.input * count + layout.scales + 4 * prep_words;
            let out_at = status_at + 4 * count;
            let at = layout.attention_at(out_at, count);
            assert!(at.is_multiple_of(PROJECTION_ALIGN), "{count} rows: {at}");
            assert!(
                at >= out_at + layout.output * count && at < out_at + layout.output * count + 16
            );
            assert!(
                at + count * layout.attention
                    <= layout.fixed_bytes() + count * layout.row_bytes().unwrap()
            );
        }
        let plain = Layout::new(shape, None, projection_side(true, true)).unwrap();
        assert_eq!(plain.attention_at(1234, 171), 1234);
    }

    #[test]
    fn raw_kv_layouts_keep_kv_slots_and_fit_their_preparation() {
        let shape = RelativeAttentionShape {
            query_heads: 64,
            kv_heads: 8,
            relative_dims: 32,
            relative_extent: 1024,
            capacity: 4096,
        };
        let kv_bytes = 8 * 128 * size_of::<f32>();
        let projected = Some(16384);
        for query in [false, true] {
            let plain = Layout::new(shape, projected, projection_side(query, false)).unwrap();
            let kv = Layout::new(shape, projected, projection_side(query, true)).unwrap();
            assert!(kv.raw_kv && !plain.raw_kv);
            // The host rows keep slots for the prepared K and V; the side rows add the raw ones.
            assert_eq!(kv.input, plain.input);
            assert_eq!(kv.side_stride(), plain.side_stride() + 2 * kv_bytes);
            assert_eq!(kv.side_slots()[1..3], [false, false]);
            assert_eq!(plain.kv_prep_words(), [0, 0]);
            assert_eq!(kv.kv_prep_words(), [2 * 7 * 1024, 2 * 4 * 8]);
            assert_eq!(
                kv.fixed_bytes(),
                kv.scales + 4 * 2 * 7 * 1024 + 3 * kv.side_stride() + PROJECTION_ALIGN
            );
            for tiled in [false, true] {
                let capacity = kv.capacity(tiled);
                let row_bytes = kv.row_bytes().unwrap();
                assert!(capacity > 0);
                assert!(kv.fixed_bytes() + capacity * row_bytes <= ARENA_BYTES);
                assert!(
                    capacity == plain.capacity(tiled)
                        || kv.fixed_bytes() + (capacity + 1) * row_bytes > ARENA_BYTES
                );
            }
        }
        // K and V are raw together or not at all.
        assert!(Layout::new(shape, projected, &[1, 4, 5]).is_none());
        assert!(Layout::new(shape, projected, &[2, 4, 5]).is_none());
    }
}
