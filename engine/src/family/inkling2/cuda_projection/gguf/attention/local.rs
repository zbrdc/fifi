//! Local attention with host-authoritative raw convolution and processed cache rows.

use super::{
    super::super::dense::{LocalInputs, LocalProjection, LocalTargets},
    *,
};
use crate::family::inkling2::switches::Switch;
use crate::{
    family::inkling2::cuda_projection::{DEVICE_HEADROOM, upload_tensor},
    kernels::cuda::{
        resident::inkling_conv::ConvCall,
        resident::inkling_local_prep::{PrepMath, PrepShape},
    },
};

use std::borrow::Cow;

use super::super::resident_local::{LocalStatus, ResidentLocalDecode, local_status};

pub(in crate::family::inkling2) type LocalReadWeights<'a> =
    dyn Fn([usize; 4]) -> Result<[Cow<'a, [f32]>; 4], EngError> + 'a;

pub(super) struct AuxStore {
    buffer: DeviceBuffer,
    transfer: PinnedBuf,
    slot_bytes: usize,
    scratch_at: usize,
    next_position: Vec<Option<usize>>,
}

impl AuxStore {
    fn allocate(
        context: &CudaContext,
        visits: &[Visit],
        (device, host): (usize, usize),
    ) -> Result<Option<Self>, EngError> {
        let slot_bytes = visits
            .iter()
            .map(|v| Layout::new(v.shape, None, false).aux_bytes())
            .max()
            .unwrap_or(0);
        let Some(scratch_at) = slot_bytes.checked_mul(visits.len()) else {
            return Ok(None);
        };
        // The local projection row uses aligned offsets relative to this base.
        let scratch_at = if decode_split_enabled() {
            let Some(at) = scratch_at.checked_next_multiple_of(16) else {
                return Ok(None);
            };
            at
        } else {
            scratch_at
        };
        let Some(bytes) = scratch_at.checked_add(device) else {
            return Ok(None);
        };
        if bytes == 0
            || (context.memory_info()?.free as u64).saturating_sub(bytes as u64) < DEVICE_HEADROOM
        {
            return Ok(None);
        }
        let buffer = match context.allocate(bytes) {
            Ok(buffer) => buffer,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        if (context.memory_info()?.free as u64) < DEVICE_HEADROOM {
            return Ok(None);
        }
        let transfer = match context.pinned_buf(host) {
            Ok(transfer) => transfer,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(Self {
            buffer,
            transfer,
            slot_bytes,
            scratch_at,
            next_position: vec![None; visits.len()],
        }))
    }

    pub(super) fn invalidate(&mut self) {
        self.next_position.fill(None);
    }
}

#[cfg(test)]
mod resident_tests;
#[cfg(test)]
mod tests;

pub(in crate::family::inkling2) struct LocalPrep<'a> {
    pub math: PrepMath,
    // Query scale, relative scale, channel-major K and V kernels.
    pub read_weights: &'a LocalReadWeights<'a>,
    pub raw: [&'a VecDeque<Vec<f32>>; 2],
}

pub(in crate::family::inkling2) struct LocalResult {
    pub output: Vec<f32>,
    pub projected: bool,
    // Raw K/V followed by clipped K and RMS-normalized V.
    pub rows: [Vec<f32>; 4],
}

mod baseline;

fn decode_split_enabled() -> bool {
    Switch::DecodeSplit.on()
}

struct Layout {
    prep: PrepShape,
    shape: RelativeAttentionShape,
    prep_at: [usize; 10],
    controller_at: usize,
    output_at: usize,
    output_rows: usize,
    readback: usize,
    attention_at: usize,
    end: usize,
}

impl Layout {
    fn new(shape: RelativeAttentionShape, output_rows: Option<usize>, controlled: bool) -> Self {
        let prep = PrepShape {
            queries: shape.query_heads,
            kv: shape.kv_heads,
            dims: shape.relative_dims,
        };
        let words = prep.words().map(|n| n * size_of::<f32>());
        let mut prep_at = [0; 10];
        // Status + prep health, controller health, final output and the four host rows.
        let controller_at = words[9];
        let output_at = controller_at + usize::from(controlled) * 3 * size_of::<f32>();
        let mut end = output_at + output_rows.map_or(words[0], |rows| rows * size_of::<f32>());
        for i in [2, 3, 4, 5, 0, 1, 6, 7, 8] {
            prep_at[i] = end;
            end += words[i];
        }
        let attention_at = if output_rows.is_some() {
            // A restored prefix takes the arena path on its first visit. Match
            // that path's aligned projection input (and hence its arithmetic).
            if decode_split_enabled() {
                end = end.next_multiple_of(16);
            }
            let at = end;
            end += words[0];
            at
        } else {
            output_at
        };
        Self {
            prep,
            shape,
            prep_at,
            controller_at,
            output_at,
            output_rows: output_rows.unwrap_or(prep.queries * PrepShape::WIDTH),
            readback: prep_at[0],
            attention_at,
            end,
        }
    }

    fn aux_bytes(&self) -> usize {
        self.prep.words()[8] * size_of::<f32>()
    }

    fn ranges<'a>(
        &self,
        arena: &'a DeviceBuffer,
        base: usize,
    ) -> Result<[DeviceRange<'a>; 10], EngError> {
        let words = self.prep.words();
        let [a, b, c, d, e, f, g, h, i, j] = std::array::from_fn(|n| {
            arena.range(base + self.prep_at[n], words[n] * size_of::<f32>())
        });
        Ok([a?, b?, c?, d?, e?, f?, g?, h?, i?, j?])
    }

    fn read(&self, transfer: &PinnedBuf) -> Result<LocalResult, EngError> {
        let mut health = vec![0; self.output_at];
        transfer.read(0, &mut health)?;
        if health.into_iter().any(|byte| byte != 0) {
            return Err(bad("invalid CUDA local attention preparation or state"));
        }
        let read = |at, words| {
            let mut values = vec![0.0; words];
            transfer.read_f32(at, &mut values)?;
            if !values.iter().all(|v| v.is_finite()) {
                return Err(bad("non-finite CUDA local attention output or history"));
            }
            Ok(values)
        };
        let [k0, v0, k, v] = [2, 3, 4, 5].map(|i| read(self.prep_at[i], self.prep.words()[i]));
        Ok(LocalResult {
            output: read(self.output_at, self.output_rows)?,
            projected: self.attention_at != self.output_at,
            rows: [k0?, v0?, k?, v?],
        })
    }
}

pub(super) fn workspace(
    shape: RelativeAttentionShape,
    projected_rows: Option<usize>,
    hidden: usize,
) -> (usize, usize) {
    let projected = Layout::new(shape, projected_rows, true);
    (
        projected.end,
        projected
            .readback
            .max(projected.aux_bytes() + 15 * hidden * size_of::<f32>()),
    )
}

impl LocalPrep<'_> {
    fn encode(
        &self,
        layout: &Layout,
        position: i32,
        cached: bool,
        buf: &mut Vec<u8>,
    ) -> Result<usize, EngError> {
        buf.clear();
        buf.extend_from_slice(&position.to_ne_bytes());
        let count = (position as usize).min(PrepShape::TAPS - 1);
        if cached {
            return Ok(count);
        }
        let [q, rel, kv, ..] = layout.prep.words();
        let widths = [q, rel, kv * PrepShape::TAPS, kv * PrepShape::TAPS];
        let weights = (self.read_weights)(widths)?;
        if weights.each_ref().map(|row| row.len()) != widths
            || self
                .raw
                .iter()
                .any(|raw| raw.len() != count || raw.iter().any(|row| row.len() != kv))
        {
            return Err(bad(
                "wrong CUDA local attention convolution history or weights",
            ));
        }
        for weights in &weights {
            extend_words(buf, weights);
        }
        for raw in self.raw {
            for row in raw {
                extend_words(buf, row);
            }
            buf.resize(
                buf.len() + (PrepShape::TAPS - 1 - count) * kv * size_of::<f32>(),
                0,
            );
        }
        debug_assert_eq!(buf.len(), layout.aux_bytes());
        Ok(count)
    }
}

impl ResidentAttention {
    pub(in crate::family::inkling2) fn can_chain_local(
        &self,
        slot: usize,
        position: usize,
    ) -> bool {
        self.local_workspace.is_some()
            && position <= POS_MAX as usize
            && self.state.borrow().next_position.get(slot) == Some(&Some(position))
    }

    pub(in crate::family::inkling2::cuda_projection) fn step_local(
        &self,
        context: &CudaContext,
        (slot, position): (usize, usize),
        prep: LocalPrep<'_>,
        input: LocalProjection<'_>,
        projection: Option<OutputProjection<'_>>,
    ) -> Result<Option<LocalResult>, EngError> {
        if !self.can_chain_local(slot, position) {
            return Ok(None);
        }
        let visit = self.visits[slot];
        let controlled = matches!(input.inputs, LocalInputs::Controlled(..));
        let layout = Layout::new(visit.shape, projection.as_ref().map(|p| p.rows), controlled);
        let mut store = self.local_aux.borrow_mut();
        if store.is_none()
            && let Some(sizes) = self.local_workspace
        {
            *store = AuxStore::allocate(context, &self.visits, sizes)?;
        }
        let Some(store) = store.as_mut() else {
            return Ok(None);
        };
        let cached = store.next_position[slot].take() == Some(position);
        let aux = store
            .buffer
            .range(slot * store.slot_bytes, layout.aux_bytes())?;
        let mut state = self.state.borrow_mut();
        let count = prep.encode(&layout, position as i32, cached, &mut state.encoded)?;
        state.next_position[slot] = None;
        let transfer = &mut store.transfer;
        transfer.write(0, &state.encoded)?;
        let queued = self.enqueue_local(
            context,
            visit,
            &layout,
            (&store.buffer, store.scratch_at, transfer),
            (count, prep.math, aux, state.encoded.len()),
            input,
            projection,
        );
        // Device attention can write the cache even when the health check fails.
        // Drain every accepted operation and publish the cursor only on success.
        let complete = transfer.synchronize(&self.stream);
        queued?;
        complete?;
        let result = layout.read(transfer)?;
        state.next_position[slot] = Some(position + 1);
        store.next_position[slot] = Some(position + 1);
        if cached {
            static REUSED: std::sync::Once = std::sync::Once::new();
            REUSED.call_once(|| eprintln!("Inkling CUDA reused local decode aux"));
        }
        if result.projected {
            state.mark_chained_projection();
        }
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used chained local decode attention"));
        Ok(Some(result))
    }

    pub(super) fn invalidate_local_aux(&self, slot: usize) {
        if let Some(store) = self.local_aux.borrow_mut().as_mut() {
            store.next_position[slot] = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_local(
        &self,
        context: &CudaContext,
        visit: Visit,
        layout: &Layout,
        (scratch, base, transfer): (&DeviceBuffer, usize, &mut PinnedBuf),
        (count, math, aux, aux_uploaded): (usize, PrepMath, DeviceRange<'_>, usize),
        input: LocalProjection<'_>,
        projection: Option<OutputProjection<'_>>,
    ) -> Result<(), EngError> {
        let mut ranges = layout.ranges(scratch, base)?;
        ranges[8] = aux;
        let [q, relative, raw_k, raw_v, k, v, sinks, gates, _, _] = ranges;
        let stream = &self.stream;
        input.dense.enqueue_local(
            context,
            input.inputs,
            input.weights,
            LocalTargets {
                projections: [q, raw_k, raw_v, relative, sinks, gates],
                controller_health: (layout.controller_at < layout.output_at)
                    .then(|| scratch.range(base + layout.controller_at, 3 * size_of::<f32>()))
                    .transpose()?,
                stream,
                transfer,
                upload_at: aux_uploaded,
            },
        )?;
        let position = aux.buffer().range(aux.offset(), size_of::<i32>())?;
        context.copy_from_pinned(
            stream,
            transfer,
            0,
            aux.buffer().range(aux.offset(), aux_uploaded)?,
        )?;
        context.inkling_local_prep(stream, ranges, layout.prep, count, math)?;
        let attention_out = scratch.range(base + layout.attention_at, q.bytes())?;
        context.relative_attention(
            stream,
            RelativeAttentionCall {
                shape: layout.shape,
                q,
                k,
                v,
                relative,
                sinks,
                gates,
                out: attention_out,
                table: self.arena.range(visit.table_at, visit.table_bytes)?,
                pos: position,
                status: scratch.range(base, size_of::<u32>())?,
                k_cache: self.arena.range(visit.cache_at, visit.cache_bytes)?,
                v_cache: self
                    .arena
                    .range(visit.cache_at + visit.cache_bytes, visit.cache_bytes)?,
            },
        )?;
        if let Some(projection) = projection {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, projection.rows, q.bytes() / size_of::<f32>()),
                    dtype: projection.dtype,
                    w: projection.weight,
                    x: attention_out,
                    y: scratch.range(base + layout.output_at, f32_bytes(projection.rows)?)?,
                },
            )?;
        }
        context.copy_to_pinned(stream, scratch.range(base, layout.readback)?, transfer, 0)
    }
}

/// Scratch reused by successive local visits on one stream; the caller owns the
/// persistent histories and weights.
pub(super) struct ResidentWorkspace {
    aux: DeviceBuffer,
    scratch: DeviceBuffer,
    zeros: Option<DeviceBuffer>,
}

/// The status sub-ranges the dense side writes into, with the zero gain and forget rows
/// of a visit without a bank.
pub(in crate::family::inkling2) struct ResidentStatus<'a> {
    pub effective: [DeviceRange<'a>; 2],
    pub controller_health: DeviceRange<'a>,
    /// The six gate half health words; `None` without a bank.
    pub gate_health: Option<DeviceRange<'a>>,
    /// `2 * hidden` zero words for identity control, only needed without a bank.
    pub zeros: Option<DeviceRange<'a>>,
}

/// Geometry the chain needs besides the request.
pub(in crate::family::inkling2) struct ResidentParts<'a> {
    pub shape: LocalShape,
    pub projection: OutputProjection<'a>,
    pub hidden: usize,
}

/// Word offsets and sizes inside one slot's aux, the layout `inkling_local_prep` reads:
/// the position word (unused here), four weight rows, then the K and V histories, which
/// the kernel shifts in place.
#[derive(Clone, Copy)]
struct AuxLayout {
    at: [usize; 6],
    sizes: [usize; 6],
}

/// Word offsets inside the shared scratch: query, relative, sinks, gates, the attention
/// output before its projection, the projected row. Each row's width follows from the shape.
#[derive(Clone, Copy)]
struct ScratchLayout {
    at: [usize; 6],
    words: usize,
}

/// What the output half of the chain reads: the scratch rows the dense side wrote, the
/// log's processed K and V, the attention status word and the two output scratch rows.
struct PrepRows<'a> {
    q: DeviceRange<'a>,
    relative: DeviceRange<'a>,
    sinks: DeviceRange<'a>,
    gates: DeviceRange<'a>,
    k: DeviceRange<'a>,
    v: DeviceRange<'a>,
    status: DeviceRange<'a>,
    attention_out: DeviceRange<'a>,
    projected: DeviceRange<'a>,
}

const WORD: usize = size_of::<f32>();

fn words_range(
    buffer: &DeviceBuffer,
    at: usize,
    words: usize,
) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(at * WORD, words * WORD)
}

impl AuxLayout {
    fn new(shape: LocalShape) -> Self {
        let [q, rel, kv] = [
            shape.query_channels(),
            shape.relative_words(),
            shape.kv_channels(),
        ];
        let taps = PrepShape::TAPS;
        let sizes = [
            q,
            rel,
            kv * taps,
            kv * taps,
            kv * (taps - 1),
            kv * (taps - 1),
        ];
        let mut at = [1; 6];
        for index in 1..6 {
            at[index] = at[index - 1] + sizes[index - 1];
        }
        Self { at, sizes }
    }

    fn words(self) -> usize {
        self.at[5] + self.sizes[5]
    }

    fn range(self, index: usize, aux: DeviceRange<'_>) -> Result<DeviceRange<'_>, EngError> {
        aux.slice(self.at[index] * WORD, self.sizes[index] * WORD)
    }
}

impl ScratchLayout {
    fn new(shape: LocalShape, hidden: usize) -> Self {
        let q = shape.query_channels();
        let sizes = [q, shape.relative_words(), shape.query_heads, q, q, hidden];
        let mut at = [0; 6];
        for index in 1..6 {
            at[index] = at[index - 1] + sizes[index - 1];
        }
        Self {
            at,
            words: at[5] + sizes[5],
        }
    }
}

impl ResidentWorkspace {
    fn allocate(
        context: &CudaContext,
        (shape, hidden): (LocalShape, usize),
    ) -> Result<Self, EngError> {
        Ok(Self {
            aux: context.allocate(AuxLayout::new(shape).words() * WORD)?,
            scratch: context.allocate(ScratchLayout::new(shape, hidden).words * WORD)?,
            zeros: None,
        })
    }
}

impl ResidentAttention {
    pub(in crate::family::inkling2::cuda_projection::gguf) fn prepare_resident_workspace(
        &self,
        context: &CudaContext,
        geometry: (LocalShape, usize),
        banked: bool,
    ) -> Result<(), EngError> {
        let mut store = self.resident.borrow_mut();
        if store.is_none() {
            *store = Some(ResidentWorkspace::allocate(context, geometry)?);
        }
        let store = store.as_mut().expect("resident workspace allocated");
        if !banked && store.zeros.is_none() {
            store.zeros = Some(upload_tensor(context, &vec![0; 2 * geometry.1 * WORD])?);
        }
        Ok(())
    }

    /// Publish the ring cursor only after the caller fenced and checked every stage.
    pub(in crate::family::inkling2) fn publish_resident_cursor(
        &self,
        slot: usize,
        position: usize,
    ) {
        let mut state = self.state.borrow_mut();
        debug_assert!(state.next_position[slot].is_none());
        state.next_position[slot] = Some(position + 1);
    }

    /// Launch-only form of `step_local` for rows that stay on the device. `dense` queues
    /// the controller and projection GEMVs into four scratch rows and the caller's log;
    /// the prep then shifts the caller's K/V histories through the aux scratch, the
    /// attention step appends to the ring, and the output projection and `attn_sconv`
    /// convolution write `out`. The slot's cursor is unknown until completion is published.
    pub(in crate::family::inkling2) fn enqueue_resident_local<'a>(
        &'a self,
        context: &CudaContext,
        stream: &DeviceStream,
        request: &'a ResidentLocalDecode<'a>,
        parts: ResidentParts<'_>,
        dense: impl FnOnce([DeviceRange<'_>; 4], ResidentStatus<'_>) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let ResidentParts {
            shape,
            projection,
            hidden,
        } = parts;
        let slot = request.visit.state_slot;
        if !self.can_chain_local(slot, request.position) {
            return Err(bad("resident local decode lost the device cursor"));
        }
        let store = self.resident.borrow();
        let store = store
            .as_ref()
            .filter(|w| request.messages.is_some() || w.zeros.is_some())
            .ok_or_else(|| bad("resident local decode workspace was not prepared"))?;
        self.invalidate_local_aux(slot);
        self.state.borrow_mut().next_position[slot] = None;
        let status = local_status(shape, request.messages.is_some());
        let rows = self.enqueue_resident_prep(
            context,
            stream,
            store,
            request,
            (shape, hidden, status),
            dense,
        )?;
        self.enqueue_resident_output(context, stream, request, (shape, projection), rows)
    }

    /// The dense closure, then the prep launch over the caller's own weight rows and K/V
    /// histories: nothing is staged into the aux scratch, which only holds the effective taps
    /// of a global visit. The kernel shifts the caller's histories in place.
    fn enqueue_resident_prep<'a>(
        &'a self,
        context: &CudaContext,
        stream: &DeviceStream,
        store: &'a ResidentWorkspace,
        request: &'a ResidentLocalDecode<'a>,
        (shape, hidden, status): (LocalShape, usize, LocalStatus),
        dense: impl FnOnce([DeviceRange<'_>; 4], ResidentStatus<'_>) -> Result<(), EngError>,
    ) -> Result<PrepRows<'a>, EngError> {
        if !context.dense_group_policy().decode_glue {
            return self.enqueue_prep_baseline(
                context,
                stream,
                store,
                request,
                (shape, hidden, status),
                dense,
            );
        }
        let aux = store.aux.range(0, store.aux.bytes())?;
        let scratch = ScratchLayout::new(shape, hidden);
        let at = scratch.at;
        let row = |index: usize, words: usize| words_range(&store.scratch, at[index], words);
        let q = shape.query_channels();
        let [qr, relative, sinks, gates] = [
            row(0, q)?,
            row(1, shape.relative_words())?,
            row(2, shape.query_heads)?,
            row(3, q)?,
        ];
        let health = Health::split(request.status, status)?;
        let layout = AuxLayout::new(shape);
        let effective = [layout.range(2, aux)?, layout.range(3, aux)?];
        dense(
            [qr, relative, sinks, gates],
            ResidentStatus {
                effective,
                controller_health: health.controller,
                gate_health: health.gates,
                zeros: store
                    .zeros
                    .as_ref()
                    .map(|b| b.range(0, b.bytes()))
                    .transpose()?,
            },
        )?;
        let kv = shape.kv_channels();
        let [raw_k, raw_v, k, v] =
            [0, 1, 2, 3].map(|i| request.log.slice(i * kv * WORD, kv * WORD));
        let [raw_k, raw_v, k, v] = [raw_k?, raw_v?, k?, v?];
        let prep = PrepShape {
            queries: shape.query_heads,
            kv: shape.kv_heads,
            dims: shape.relative_dims,
        };
        // A global visit's dense closure wrote the effective taps; a local one convolves with
        // the static kernels.
        let [kernel_k, kernel_v] = if request.visit.attention == Attention::Global {
            effective
        } else {
            [request.weights.key_kernel, request.weights.value_kernel]
        };
        context.inkling_local_prep_direct(
            stream,
            [
                qr,
                relative,
                raw_k,
                raw_v,
                k,
                v,
                sinks,
                gates,
                request.weights.query_scale,
                request.weights.relative_scale,
                kernel_k,
                kernel_v,
                request.raw_history[0],
                request.raw_history[1],
                health.prep,
            ],
            prep,
            PrepShape::TAPS - 1,
            request.math,
        )?;
        Ok(PrepRows {
            q: qr,
            relative,
            sinks,
            gates,
            k,
            v,
            status: health.attention,
            attention_out: row(4, q)?,
            projected: row(5, hidden)?,
        })
    }

    /// Attention over the resident ring, then the output projection and the `attn_sconv`
    /// convolution into the caller's row.
    fn enqueue_resident_output(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        request: &ResidentLocalDecode<'_>,
        (shape, projection): (LocalShape, OutputProjection<'_>),
        rows: PrepRows<'_>,
    ) -> Result<(), EngError> {
        let PrepRows {
            q,
            relative,
            sinks,
            gates,
            k,
            v,
            status,
            attention_out,
            projected,
        } = rows;
        let visit = self.visits[request.visit.state_slot];
        let ring = |at, bytes| self.arena.range(at, bytes);
        context.relative_attention(
            stream,
            RelativeAttentionCall {
                shape: visit.shape,
                q,
                k,
                v,
                relative,
                sinks,
                gates,
                out: attention_out,
                table: ring(visit.table_at, visit.table_bytes)?,
                pos: request.pos,
                status,
                k_cache: ring(visit.cache_at, visit.cache_bytes)?,
                v_cache: ring(visit.cache_at + visit.cache_bytes, visit.cache_bytes)?,
            },
        )?;
        context.dense_gemv(
            stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(1, projection.rows, shape.query_channels()),
                dtype: projection.dtype,
                w: projection.weight,
                x: attention_out,
                y: projected,
            },
        )?;
        let status = local_status(shape, request.messages.is_some());
        let health = Health::split(request.status, status)?;
        context.inkling_conv(
            stream,
            ConvCall {
                input: projected,
                kernel: request.attn_conv.kernel,
                history: request.attn_conv.history,
                out: request.out,
                state: None,
                health: health.conv,
            },
            (request.out.bytes() / WORD, PrepShape::TAPS),
        )
    }
}

/// The request's status range by consumer.
struct Health<'a> {
    /// The prep flags: one shared status word, then one flag per head.
    prep: DeviceRange<'a>,
    /// The shared status word of the prep and the attention step.
    attention: DeviceRange<'a>,
    controller: DeviceRange<'a>,
    gates: Option<DeviceRange<'a>>,
    conv: DeviceRange<'a>,
}

impl<'a> Health<'a> {
    fn split(status: DeviceRange<'a>, layout: LocalStatus) -> Result<Self, EngError> {
        let at = |index: usize| status.slice(index * WORD, WORD);
        let controller = status.slice(layout.prep_words * WORD, layout.roles * WORD)?;
        let gates_at = layout.prep_words + layout.roles;
        let gates = (layout.gates > 0)
            .then(|| status.slice(gates_at * WORD, layout.gates * WORD))
            .transpose()?;
        Ok(Self {
            prep: status.slice(0, layout.prep_words * WORD)?,
            attention: at(0)?,
            controller,
            gates,
            conv: at(gates_at + layout.gates)?,
        })
    }
}
