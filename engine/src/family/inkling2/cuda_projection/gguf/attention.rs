//! GGUF attention with resident, per-visit K/V and reusable staging.
//!
//! Host K/V rows back prefix snapshots and speculative verifier undo.
//! A known device cursor is valid for the current continuation. Prefix restore
//! invalidates every cursor. Rejection repairs only overwritten visible rows.

use crate::family::inkling2::switches::Switch;
use crate::kernels::cuda::set_pool::SetPool;
use std::{cell::RefCell, collections::VecDeque, sync::Arc};

use crate::{
    EngError,
    expert_tier::VRAM_HEADROOM_BYTES,
    family::inkling2::{
        Attention, LayerVisit, TextSpec, gguf_weights::GgufTextWeights,
        resident_history::upload_rows,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        resident::{
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape},
            relative_attention::{
                HEAD_DIM, RelativeAttentionCall, RelativeAttentionShape, RelativeKvFormat,
            },
            step_params::POS_MAX,
        },
    },
};

#[cfg(test)]
use super::DEVICE_HEADROOM;
use super::{super::state_headroom, bad};

mod local;
pub(in crate::family::inkling2) mod prefill;
pub(in crate::family::inkling2) use local::{
    LocalPrep, LocalReadWeights, LocalResult, ResidentParts, ResidentStatus,
};

#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct Config {
    pub context: usize,
    pub relative_dims: usize,
    pub relative_extent: usize,
}

#[derive(Clone, Copy)]
struct Visit {
    shape: RelativeAttentionShape,
    kv_format: RelativeKvFormat,
    table_at: usize,
    table_bytes: usize,
    cache_at: usize,
    cache_bytes: usize,
}

/// The local visit geometry as the resident decode chain uses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family::inkling2) struct LocalShape {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub relative_dims: usize,
}

impl LocalShape {
    pub(in crate::family::inkling2) fn query_channels(self) -> usize {
        self.query_heads * HEAD_DIM
    }
    pub(in crate::family::inkling2) fn kv_channels(self) -> usize {
        self.kv_heads * HEAD_DIM
    }
    pub(in crate::family::inkling2) fn relative_words(self) -> usize {
        self.query_heads * self.relative_dims
    }
}

pub(in crate::family::inkling2) struct Plan {
    visits: Vec<Visit>,
    tables: Vec<u8>,
    scratch_bytes: usize,
    local_workspace: Option<(usize, usize)>,
    projected_rows: Option<usize>,
}

pub(in crate::family::inkling2) struct Row<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub relative: &'a [f32],
    pub sinks: &'a [f32],
    pub gates: &'a [f32],
}

pub(in crate::family::inkling2) struct OutputProjection<'a> {
    pub(super) weight: DeviceRange<'a>,
    pub(super) dtype: DenseDtype,
    pub(super) rows: usize,
}

pub(in crate::family::inkling2) struct History<'a> {
    pub position: usize,
    pub keys: &'a VecDeque<Arc<Vec<f32>>>,
    pub values: &'a VecDeque<Arc<Vec<f32>>>,
}

pub(in crate::family::inkling2) struct ResidentAttention {
    visits: Vec<Visit>,
    arena: DeviceBuffer,
    scratch_at: usize,
    local_workspace: Option<(usize, usize)>,
    projected_rows: Option<usize>,
    align_projected_input: bool,
    projection_headroom: u64,
    reserve_credit: u64,
    stream: DeviceStream,
    query_tiles: bool,
    state: RefCell<Runtime>,
    prefill: RefCell<Option<prefill::Workspace>>,
    /// The device buffers of finished prompt convolutions, kept for the next visit's.
    conv_pool: SetPool<prefill::ConvBuffers>,
    local_aux: RefCell<Option<local::AuxStore>>,
    resident: RefCell<Option<local::ResidentWorkspace>>,
    /// The MMQ GEMM state of prompt output projections.
    tensor_core: RefCell<crate::kernels::cuda::prefill::dense_chunk_gemm::PlaneGemm>,
}

struct Runtime {
    transfer: PinnedBuf,
    encoded: Vec<u8>,
    next_position: Vec<Option<usize>>,
    chained_projection_used: bool,
}

impl Plan {
    pub(in crate::family::inkling2) fn new(
        weights: &GgufTextWeights,
        spec: &TextSpec,
        config: Config,
    ) -> Result<Option<Self>, EngError> {
        let table_words = config
            .relative_dims
            .checked_mul(config.relative_extent)
            .ok_or_else(|| bad("CUDA relative attention table overflows"))?;
        let table_bytes = f32_bytes(table_words)?;
        let mut tables = Vec::new();
        for layer in 0..spec.stored_layers() {
            let table = weights.vec(
                &format!("model.llm.layers.{layer}.attn.rel_logits_proj.proj"),
                table_words,
            )?;
            extend_words(&mut tables, &table);
        }

        let hidden = spec.hidden_size as usize;
        let projected_rows = (Switch::ChainAttentionOutput.on()).then_some(hidden);
        let local_enabled = (1..=4096).contains(&hidden) && Switch::ChainLocalDecode.on();
        let mut local_workspace = None::<(usize, usize)>;
        let mut visits = Vec::new();
        let mut cache_at = tables.len();
        let mut scratch_bytes = 0;
        for visit in spec.visits() {
            let Some(shape) = shape(spec, config, visit) else {
                return Ok(None);
            };
            let Some(
                [
                    out,
                    cache_bytes,
                    _,
                    status,
                    q,
                    k,
                    v,
                    relative,
                    _,
                    sinks,
                    gates,
                    pos,
                ],
            ) = shape.region_bytes()
            else {
                return Ok(None);
            };
            scratch_bytes =
                scratch_bytes.max(q + k + v + relative + sinks + gates + pos + status + out);
            if local_enabled && visit.attention == Attention::Local {
                let (device, host) = local::workspace(shape, projected_rows, hidden);
                let (old_device, old_host) = local_workspace.unwrap_or_default();
                local_workspace = Some((device.max(old_device), host.max(old_host)));
            }
            visits.push(Visit {
                shape,
                kv_format: RelativeKvFormat::F32,
                table_at: visit.weight_layer * table_bytes,
                table_bytes,
                cache_at,
                cache_bytes,
            });
            cache_at = cache_at
                .checked_add(cache_bytes)
                .and_then(|at| at.checked_add(cache_bytes))
                .ok_or_else(|| bad("CUDA relative attention cache overflows"))?;
        }
        if let Some(rows) = projected_rows {
            scratch_bytes = scratch_bytes
                .checked_add(f32_bytes(rows)?)
                .ok_or_else(|| bad("CUDA relative attention projection scratch overflows"))?;
        }
        Ok(Some(Self {
            visits,
            tables,
            scratch_bytes,
            local_workspace,
            projected_rows,
        }))
    }

    fn arena_bytes_for(
        &self,
        kv_format: RelativeKvFormat,
        align_projected_input: bool,
    ) -> Result<usize, EngError> {
        let mut cache_at = self.tables.len();
        for visit in &self.visits {
            let [_, cache_bytes, ..] = visit
                .shape
                .region_bytes_for(kv_format)
                .ok_or_else(|| bad("CUDA relative attention cache overflows"))?;
            cache_at = cache_at
                .checked_add(cache_bytes)
                .and_then(|at| at.checked_add(cache_bytes))
                .ok_or_else(|| bad("CUDA relative attention cache overflows"))?;
        }
        cache_at
            .checked_add(self.scratch_bytes)
            .and_then(|n| n.checked_add(if align_projected_input { 12 } else { 0 }))
            .ok_or_else(|| bad("CUDA relative attention arena overflows"))
    }

    fn apply_kv_format(&mut self, kv_format: RelativeKvFormat) -> Result<usize, EngError> {
        let mut cache_at = self.tables.len();
        for visit in &mut self.visits {
            let [_, cache_bytes, ..] = visit
                .shape
                .region_bytes_for(kv_format)
                .ok_or_else(|| bad("CUDA relative attention cache overflows"))?;
            visit.kv_format = kv_format;
            visit.cache_at = cache_at;
            visit.cache_bytes = cache_bytes;
            cache_at = cache_at
                .checked_add(cache_bytes)
                .and_then(|at| at.checked_add(cache_bytes))
                .ok_or_else(|| bad("CUDA relative attention cache overflows"))?;
        }
        Ok(cache_at)
    }

    pub(super) fn load(
        mut self,
        context: &CudaContext,
        charge_reserve: bool,
    ) -> Result<Option<ResidentAttention>, EngError> {
        if self.visits.is_empty() {
            return Err(bad("CUDA attention requires at least one visit"));
        }
        let before = context.memory_info()?.free as u64;
        // Keep status and the returned projection contiguous. Only the internal
        // attention row, consumed by the single-row weight kernel, needs padding.
        let align_projected_input = self.projected_rows.is_some() && Switch::DecodeSplit.on();
        let requested = kv_dtype_override()?;
        let candidates: &[RelativeKvFormat] = match &requested {
            Some(fmt) => std::slice::from_ref(fmt),
            None => &[
                RelativeKvFormat::F32,
                RelativeKvFormat::F16,
                RelativeKvFormat::Int8,
            ],
        };
        let sizes: Vec<(RelativeKvFormat, usize)> = candidates
            .iter()
            .map(|&fmt| Ok((fmt, self.arena_bytes_for(fmt, align_projected_input)?)))
            .collect::<Result<_, EngError>>()?;
        let Some((chosen_format, bytes, credit_reserve, min_headroom)) =
            select_kv_format(before, charge_reserve, &sizes)
        else {
            let smallest = sizes.last().map_or(0, |&(_, b)| b);
            let spare = before.saturating_sub(state_headroom(if charge_reserve {
                smallest as u64
            } else {
                0
            }));
            eprintln!(
                "Inkling CUDA relative attention not admitted: needs {smallest} bytes, {spare} available"
            );
            return Ok(None);
        };
        let scratch_at = self.apply_kv_format(chosen_format)?;
        let proposed_credit = if credit_reserve { bytes as u64 } else { 0 };
        let arena = context.allocate(bytes)?;
        let after = context.memory_info()?.free as u64;
        let device_bytes = before.saturating_sub(after);
        // Only this allocation's requested bytes count against the reserve.
        // CUDA padding still consumes the physical free-memory margin.
        let credit = device_bytes.min(proposed_credit);
        let projection_headroom = state_headroom(credit).max(min_headroom);
        if after < projection_headroom {
            eprintln!(
                "Inkling CUDA relative attention not admitted after device allocation rounding"
            );
            return Ok(None);
        }
        let stream = context.stream()?;
        arena.write_at(&stream, 0, &self.tables)?;
        let transfer = context.pinned_buf(self.scratch_bytes)?;
        if credit_reserve {
            let free = context.memory_info()?.free as u64;
            if free < projection_headroom {
                eprintln!("Inkling CUDA relative attention not admitted after staging setup");
                return Ok(None);
            }
            eprintln!(
                "Inkling CUDA relative attention reserve: before {before}, allocated {device_bytes}, credited {credit}, free {free}, projection floor {projection_headroom} bytes"
            );
        }
        let next_position = vec![None; self.visits.len()];
        eprintln!(
            "Inkling CUDA relative attention admitted {} visits, {} device bytes (kv={})",
            self.visits.len(),
            bytes,
            chosen_format.name(),
        );
        Ok(Some(ResidentAttention {
            visits: self.visits,
            arena,
            scratch_at,
            local_workspace: self.local_workspace,
            projected_rows: self.projected_rows,
            align_projected_input,
            projection_headroom,
            reserve_credit: credit,
            stream,
            query_tiles: self.projected_rows.is_some() && prefill::query_tiles_enabled(context),
            prefill: RefCell::default(),
            conv_pool: SetPool::new(prefill::CONV_POOL),
            local_aux: RefCell::default(),
            resident: RefCell::default(),
            tensor_core: RefCell::default(),
            state: RefCell::new(Runtime {
                transfer,
                encoded: Vec::with_capacity(self.scratch_bytes),
                next_position,
                chained_projection_used: false,
            }),
        }))
    }
}

fn shape(spec: &TextSpec, config: Config, visit: LayerVisit) -> Option<RelativeAttentionShape> {
    let (query_heads, kv_heads, width, capacity) = match visit.attention {
        Attention::Local => (
            spec.swa_num_attention_heads,
            spec.swa_num_key_value_heads,
            spec.swa_head_dim,
            config.context.min(spec.sliding_window_size as usize),
        ),
        Attention::Global => (
            spec.num_attention_heads,
            spec.num_key_value_heads,
            spec.head_dim,
            config.context,
        ),
    };
    (width == HEAD_DIM as u64).then_some(RelativeAttentionShape {
        query_heads: query_heads as usize,
        kv_heads: kv_heads as usize,
        relative_dims: config.relative_dims,
        relative_extent: config.relative_extent,
        capacity,
    })
}

impl ResidentAttention {
    pub(super) fn projection_headroom(&self) -> u64 {
        self.projection_headroom
    }

    pub(in crate::family::inkling2) fn reserve_credit(&self) -> u64 {
        self.reserve_credit
    }

    /// The geometry of a visit slot; `None` for an unknown slot.
    pub(super) fn local_shape(&self, slot: usize) -> Option<LocalShape> {
        let visit = self.visits.get(slot)?;
        Some(LocalShape {
            query_heads: visit.shape.query_heads,
            kv_heads: visit.shape.kv_heads,
            relative_dims: visit.shape.relative_dims,
        })
    }

    /// The taps of every local short convolution, fixed by the prep kernel.
    pub(super) fn local_taps(&self) -> usize {
        crate::kernels::cuda::resident::relative_prep::PrepShape::TAPS
    }

    pub(super) fn projected_rows(&self) -> Option<usize> {
        self.projected_rows
    }

    pub(in crate::family::inkling2) fn invalidate(&self) {
        self.state.borrow_mut().next_position.fill(None);
        if let Some(aux) = self.local_aux.borrow_mut().as_mut() {
            aux.invalidate();
        }
    }

    /// [`Self::download_ring`] behind this attention's own stream, which appends the rows of a
    /// device prompt visit.
    #[allow(clippy::type_complexity)]
    pub(in crate::family::inkling2) fn download_prompt_ring(
        &self,
        slot: usize,
        position: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), EngError> {
        self.download_ring(&self.stream, slot, position)
    }

    /// The K and V rows of one visit's ring at `position`, oldest last, read behind the work
    /// queued on `stream`, which must be the stream that appended the rows. Only explicit host
    /// events call it; a chain token keeps these rows on the device.
    #[allow(clippy::type_complexity)]
    pub(in crate::family::inkling2) fn download_ring(
        &self,
        stream: &DeviceStream,
        slot: usize,
        position: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), EngError> {
        let visit = self
            .visits
            .get(slot)
            .ok_or_else(|| bad("unknown visit slot"))?;
        let capacity = visit.shape.capacity;
        let row_bytes = visit.cache_bytes / capacity;
        let first = position.saturating_sub(capacity);
        let rows = position - first;
        let mut planes = Vec::with_capacity(2);
        for plane in 0..2 {
            let base = visit.cache_at + plane * visit.cache_bytes;
            let mut bytes = vec![0u8; rows * row_bytes];
            let mut read = 0;
            for (at, count) in ring_splits(first % capacity, rows, capacity) {
                if count > 0 {
                    self.arena.read_at(
                        stream,
                        base + at * row_bytes,
                        &mut bytes[read..read + count * row_bytes],
                    )?;
                    read += count * row_bytes;
                }
            }
            planes.push(
                bytes
                    .chunks_exact(row_bytes)
                    .map(|row| visit.kv_format.decode_row(row))
                    .collect::<Vec<Vec<f32>>>(),
            );
            debug_assert_eq!(planes[plane].len(), rows);
        }
        let [keys, values]: [Vec<Vec<f32>>; 2] = planes.try_into().unwrap();
        Ok((keys, values))
    }

    pub(in crate::family::inkling2) fn step(
        &self,
        context: &CudaContext,
        state_slot: usize,
        row: Row<'_>,
        history: History<'_>,
        projection: Option<OutputProjection<'_>>,
    ) -> Result<Vec<f32>, EngError> {
        let position = i32::try_from(history.position)
            .ok()
            .filter(|&position| position <= POS_MAX)
            .ok_or_else(|| bad("CUDA relative attention position overflows"))?;
        let visit = self.visits[state_slot];
        self.invalidate_local_aux(state_slot);
        let mut state = self.state.borrow_mut();
        // The kernel can publish a cache row even when its output or a
        // later transfer fails. Leave this cursor unknown until completion.
        let device_cursor = state.next_position[state_slot].take();
        self.restore_rows(visit, &history, device_cursor, &mut state.encoded)?;

        let ordered = row.encode(visit.shape, position, &mut state.encoded)?;
        let input_bytes = state.encoded.len();
        let attention_bytes = f32_bytes(row.query.len())?;
        let output_rows = projection.as_ref().map_or(row.query.len(), |p| p.rows);
        let output_bytes = f32_bytes(output_rows)?;
        let Runtime {
            transfer, encoded, ..
        } = &mut *state;
        transfer.write(0, encoded)?;
        let [q, k, v, relative, sinks, gates, pos, status] =
            ordered.map(|offset| offset + self.scratch_at);
        let out = self.scratch_at + input_bytes;
        // Put the final result immediately after the status word for one readback.
        let chained = projection.is_some();
        let attention_at = out + if chained { output_bytes } else { 0 };
        let attention_at = if chained && self.align_projected_input {
            attention_at.next_multiple_of(16)
        } else {
            attention_at
        };
        let attention_out = self.arena.range(attention_at, attention_bytes)?;
        let call = RelativeAttentionCall {
            shape: visit.shape,
            kv_format: visit.kv_format,
            q: self.arena.range(q, k - q)?,
            k: self.arena.range(k, v - k)?,
            v: self.arena.range(v, relative - v)?,
            relative: self.arena.range(relative, sinks - relative)?,
            table: self.arena.range(visit.table_at, visit.table_bytes)?,
            sinks: self.arena.range(sinks, gates - sinks)?,
            gates: self.arena.range(gates, pos - gates)?,
            pos: self.arena.range(pos, size_of::<u32>())?,
            status: self.arena.range(status, size_of::<u32>())?,
            out: attention_out,
            k_cache: self.arena.range(visit.cache_at, visit.cache_bytes)?,
            v_cache: self
                .arena
                .range(visit.cache_at + visit.cache_bytes, visit.cache_bytes)?,
        };
        let queued = (|| {
            context.copy_from_pinned(
                &self.stream,
                transfer,
                0,
                self.arena.range(self.scratch_at, input_bytes)?,
            )?;
            context.relative_attention(&self.stream, call)?;
            if let Some(projection) = projection {
                context.dense_gemv(
                    &self.stream,
                    DenseGemvCall {
                        shape: DenseGemvShape::packed(1, projection.rows, row.query.len()),
                        dtype: projection.dtype,
                        w: projection.weight,
                        x: attention_out,
                        y: self.arena.range(out, output_bytes)?,
                    },
                )?;
            }
            context.copy_to_pinned(
                &self.stream,
                self.arena.range(status, size_of::<u32>() + output_bytes)?,
                transfer,
                0,
            )
        })();
        // Finish any accepted transfer before shared staging can be used again.
        let complete = transfer.synchronize(&self.stream);
        queued?;
        complete?;
        let mut health = [0; size_of::<u32>()];
        transfer.read(0, &mut health)?;
        if u32::from_ne_bytes(health) != 0 {
            return Err(bad("invalid CUDA relative attention state"));
        }
        let mut result = vec![0.0; output_rows];
        transfer.read_f32(size_of::<u32>(), &mut result)?;
        if !result.iter().all(|v| v.is_finite()) {
            return Err(bad(
                "non-finite CUDA relative attention or projection output",
            ));
        }
        state.next_position[state_slot] = Some(history.position + 1);
        if chained {
            state.mark_chained_projection();
        }
        Ok(result)
    }

    fn restore_rows(
        &self,
        visit: Visit,
        history: &History<'_>,
        device_cursor: Option<usize>,
        bytes: &mut Vec<u8>,
    ) -> Result<(), EngError> {
        let capacity = visit.shape.capacity;
        let upload = upload_rows(history.position, device_cursor, capacity);
        if upload.is_empty() {
            return Ok(());
        }
        let row_bytes = visit.cache_bytes / capacity;
        let row_words = visit.shape.kv_heads * HEAD_DIM;
        let splits = ring_splits(upload.start % capacity, upload.len(), capacity);
        let skip = history.position - upload.end;
        let rewind = device_cursor.is_some_and(|cursor| cursor > history.position);
        for (plane, values) in [history.keys, history.values].into_iter().enumerate() {
            if values.len() < skip + upload.len() {
                return Err(bad("missing host rows for CUDA relative attention restore"));
            }
            bytes.clear();
            for row in values.iter().skip(skip).take(upload.len()).rev() {
                if row.len() != row_words {
                    return Err(bad("invalid host row for CUDA relative attention restore"));
                }
                visit.kv_format.encode_row(row, bytes);
            }
            let base = visit.cache_at + plane * visit.cache_bytes;
            let mut read = 0;
            for (at, count) in splits {
                if count == 0 {
                    continue;
                }
                let len = count * row_bytes;
                let offset = base + at * row_bytes;
                let slice = &bytes[read..read + len];
                if rewind {
                    // The following attention launch uses the same stream.
                    self.arena.write_at_enqueue(&self.stream, offset, slice)?;
                } else {
                    self.arena.write_at(&self.stream, offset, slice)?;
                }
                read += len;
            }
        }
        Ok(())
    }
}

impl Runtime {
    fn mark_chained_projection(&mut self) {
        if !self.chained_projection_used {
            eprintln!("Inkling CUDA used chained relative attention output projection");
            self.chained_projection_used = true;
        }
    }
}

impl Row<'_> {
    fn encode(
        &self,
        shape: RelativeAttentionShape,
        position: i32,
        bytes: &mut Vec<u8>,
    ) -> Result<[usize; 8], EngError> {
        bytes.clear();
        let offsets = self.append_inputs(shape, position, bytes, ([false; 6], false))?;
        bytes.extend_from_slice(&0_u32.to_ne_bytes());
        Ok(offsets)
    }

    /// The bytes of the row's six input slots: none for a slot marked `resident`. With
    /// `device_kv` an empty K or V row is a zero slot the device's K/V preparation fills.
    fn input_sizes(
        &self,
        shape: RelativeAttentionShape,
        (resident, device_kv): ([bool; 6], bool),
    ) -> Result<InputSlots<'_>, EngError> {
        let slices = [
            self.query,
            self.key,
            self.value,
            self.relative,
            self.sinks,
            self.gates,
        ];
        let q = shape.query_heads * HEAD_DIM;
        let kv = shape.kv_heads * HEAD_DIM;
        let mut sizes = [
            q,
            kv,
            kv,
            shape.query_heads * shape.relative_dims,
            shape.query_heads,
            q,
        ];
        for (size, resident) in sizes.iter_mut().zip(resident) {
            if resident {
                *size = 0;
            }
        }
        let filled =
            std::array::from_fn(|i| device_kv && (i == 1 || i == 2) && slices[i].is_empty());
        if (0..6).any(|i| slices[i].len() != sizes[i] && !filled[i]) {
            return Err(bad("incorrect CUDA relative attention input shape"));
        }
        Ok((slices, sizes.map(|words| words * size_of::<f32>()), filled))
    }

    /// Appends the row's host inputs (see [`Self::input_sizes`]) and its position.
    fn append_inputs(
        &self,
        shape: RelativeAttentionShape,
        position: i32,
        bytes: &mut Vec<u8>,
        slots: ([bool; 6], bool),
    ) -> Result<[usize; 8], EngError> {
        let (slices, sizes, filled) = self.input_sizes(shape, slots)?;
        let mut offsets = [0; 8];
        for i in 0..6 {
            offsets[i] = bytes.len();
            if filled[i] {
                bytes.resize(bytes.len() + sizes[i], 0);
            } else {
                extend_words(bytes, slices[i]);
            }
        }
        offsets[6] = bytes.len();
        bytes.extend_from_slice(&position.to_ne_bytes());
        offsets[7] = bytes.len();
        Ok(offsets)
    }

    /// The offsets [`Self::append_inputs`] would give a row at byte `at`, for a row the
    /// device fills whole: every slot is resident or a K/V slot its preparation writes, and the
    /// preparation writes the position too. `None` when the row has host bytes.
    fn device_offsets(
        &self,
        shape: RelativeAttentionShape,
        at: usize,
        resident: [bool; 6],
    ) -> Result<Option<[usize; 8]>, EngError> {
        let (_, sizes, filled) = self.input_sizes(shape, (resident, true))?;
        if (0..6).any(|i| sizes[i] != 0 && !filled[i]) {
            return Ok(None);
        }
        let mut offsets = [0; 8];
        let mut end = at;
        for (offset, size) in offsets.iter_mut().zip(sizes) {
            *offset = end;
            end += size;
        }
        offsets[6] = end;
        offsets[7] = end + size_of::<i32>();
        Ok(Some(offsets))
    }
}

/// A row's six input slots ([`Row::input_sizes`]): the host values, the bytes each takes in the
/// upload, and whether the device's K/V preparation fills it.
type InputSlots<'a> = ([&'a [f32]; 6], [usize; 6], [bool; 6]);

/// Minimum free VRAM kept after crediting the 3 GiB state reserve for a long-context ring,
/// leaving room for 512-row prefill workspaces (~450 MiB).
const LONG_CONTEXT_WORKSPACE_FLOOR: u64 = 768 * 1024 * 1024;

fn kv_dtype_override() -> Result<Option<RelativeKvFormat>, EngError> {
    const NAME: &str = "FIFI_INKLING_CUDA_KV_DTYPE";
    let Some(raw) = std::env::var_os(NAME) else {
        return Ok(None);
    };
    let text = raw
        .to_str()
        .ok_or_else(|| bad(format!("{NAME} must be valid UTF-8")))?;
    RelativeKvFormat::parse_override(text).ok_or_else(|| {
        bad(format!(
            "{NAME} must be auto, f32, f16 or q8 (got {text:?})"
        ))
    })
}

/// Picks the first `(format, bytes)` that fits in `before` free VRAM. Pass 1 honors the requested
/// reserve policy (`VRAM_HEADROOM_BYTES` unless `charge_reserve` is set). If nothing fits and
/// `charge_reserve` was off, Pass 2 credits the state reserve while keeping at least
/// [`LONG_CONTEXT_WORKSPACE_FLOOR`] free for prefill workspaces.
fn select_kv_format(
    before: u64,
    charge_reserve: bool,
    candidates: &[(RelativeKvFormat, usize)],
) -> Option<(RelativeKvFormat, usize, bool, u64)> {
    if before >= VRAM_HEADROOM_BYTES {
        for &(fmt, bytes) in candidates {
            let headroom = state_headroom(if charge_reserve { bytes as u64 } else { 0 });
            if (bytes as u64) <= before.saturating_sub(headroom) {
                return Some((fmt, bytes, charge_reserve, 0));
            }
        }
    }
    if !charge_reserve && before >= LONG_CONTEXT_WORKSPACE_FLOOR {
        for &(fmt, bytes) in candidates {
            let headroom = state_headroom(bytes as u64).max(LONG_CONTEXT_WORKSPACE_FLOOR);
            if (bytes as u64) <= before.saturating_sub(headroom) {
                return Some((fmt, bytes, true, LONG_CONTEXT_WORKSPACE_FLOOR));
            }
        }
    }
    None
}

fn ring_splits(first_slot: usize, rows: usize, capacity: usize) -> [(usize, usize); 2] {
    let until_wrap = rows.min(capacity - first_slot);
    [(first_slot, until_wrap), (0, rows - until_wrap)]
}

fn extend_words(bytes: &mut Vec<u8>, words: &[f32]) {
    bytes.extend(words.iter().flat_map(|word| word.to_ne_bytes()));
}

fn f32_bytes(words: usize) -> Result<usize, EngError> {
    words
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| bad("CUDA relative attention byte count overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_credit_preserves_workspace_headroom() {
        assert_eq!(state_headroom(0), VRAM_HEADROOM_BYTES);
        assert_eq!(state_headroom(32 << 20), VRAM_HEADROOM_BYTES - (32 << 20));
        assert_eq!(state_headroom(VRAM_HEADROOM_BYTES), DEVICE_HEADROOM);
    }

    #[test]
    fn auto_selects_narrower_kv_ring_when_free_vram_tightens() {
        assert_eq!(RelativeKvFormat::parse_override("auto"), Some(None));
        assert_eq!(
            RelativeKvFormat::parse_override("f32"),
            Some(Some(RelativeKvFormat::F32))
        );
        assert_eq!(
            RelativeKvFormat::parse_override("f16"),
            Some(Some(RelativeKvFormat::F16))
        );
        assert_eq!(
            RelativeKvFormat::parse_override("q8"),
            Some(Some(RelativeKvFormat::Int8))
        );
        assert_eq!(RelativeKvFormat::parse_override("bogus"), None);

        let gib = 1usize << 30;
        let sizes_256k = [
            (RelativeKvFormat::F32, 16 * gib),
            (RelativeKvFormat::F16, 8 * gib),
            (RelativeKvFormat::Int8, 4 * gib + (gib / 4)),
        ];
        // 14.7 GiB free (IQ1_M at 256K): F16 fits while keeping the full 3 GiB reserve.
        let picked = select_kv_format(14 * gib as u64 + 700 * (1 << 20), false, &sizes_256k);
        assert_eq!(picked, Some((RelativeKvFormat::F16, 8 * gib, false, 0)));
        // 5.2 GiB free (17 GiB quant at 256K): Int8 fits by crediting the state reserve.
        let picked = select_kv_format(5 * gib as u64 + 200 * (1 << 20), false, &sizes_256k);
        assert_eq!(
            picked,
            Some((
                RelativeKvFormat::Int8,
                4 * gib + (gib / 4),
                true,
                LONG_CONTEXT_WORKSPACE_FLOOR
            ))
        );
    }

    #[test]
    fn restore_remaps_visible_rows_across_the_ring_boundary() {
        assert_eq!(ring_splits(0, 2, 4), [(0, 2), (0, 0)]);
        assert_eq!(ring_splits(3, 4, 4), [(3, 1), (0, 3)]);
        assert_eq!(ring_splits(2, 3, 4), [(2, 2), (0, 1)]);
    }

    #[test]
    fn restored_attention_replaces_a_divergent_ring_at_the_same_position() {
        if !crate::kernels::cuda::device_present("Inkling CUDA restored relative attention") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let shape = RelativeAttentionShape {
            query_heads: 1,
            kv_heads: 1,
            relative_dims: 1,
            relative_extent: 1,
            capacity: 3,
        };
        let [
            out,
            cache_bytes,
            _,
            status,
            q,
            k,
            v,
            relative,
            table_bytes,
            sinks,
            gates,
            pos,
        ] = shape.region_bytes().unwrap();
        let attention = Plan {
            visits: vec![Visit {
                shape,
                kv_format: RelativeKvFormat::F32,
                table_at: 0,
                table_bytes,
                cache_at: table_bytes,
                cache_bytes,
            }],
            tables: vec![0; table_bytes],
            scratch_bytes: q
                + k
                + v
                + relative
                + sinks
                + gates
                + pos
                + status
                + out
                + f32_bytes(2).unwrap(),
            local_workspace: None,
            projected_rows: Some(2),
        }
        .load(&ctx, false)
        .unwrap()
        .unwrap();
        let zero = [0.0; HEAD_DIM];
        let stream = ctx.stream().unwrap();
        let weight = ctx.allocate(f32_bytes(2 * HEAD_DIM).unwrap()).unwrap();
        let mut weights = [0.0; 2 * HEAD_DIM];
        weights[0] = 2.0;
        weights[HEAD_DIM] = -1.0;
        let mut encoded = Vec::new();
        extend_words(&mut encoded, &weights);
        weight.write_at(&stream, 0, &encoded).unwrap();
        let run = |position, value, history: &[f32], project: bool| {
            let keys = history.iter().map(|_| Arc::new(zero.to_vec())).collect();
            let values = history
                .iter()
                .map(|&v| Arc::new(vec![v; HEAD_DIM]))
                .collect();
            attention.step(
                &ctx,
                0,
                Row {
                    query: &zero,
                    key: &zero,
                    value: &[value; HEAD_DIM],
                    relative: &[0.0],
                    sinks: &[0.0],
                    gates: &zero,
                },
                History {
                    position,
                    keys: &keys,
                    values: &values,
                },
                project.then(|| OutputProjection {
                    weight: weight.range(0, weight.bytes()).unwrap(),
                    dtype: DenseDtype::F32,
                    rows: 2,
                }),
            )
        };
        assert_eq!(run(0, 2.0, &[], false).unwrap(), [0.5; HEAD_DIM]);
        assert_eq!(run(1, 4.0, &[2.0], true).unwrap(), [2.0, -1.0]);
        attention.invalidate();
        assert_eq!(run(2, 2.0, &[40.0, 20.0], true).unwrap(), [15.5, -7.75]);
        assert_eq!(
            run(3, 6.0, &[2.0, 40.0, 20.0], false).unwrap(),
            [6.0; HEAD_DIM]
        );

        // Abandon two speculative rows after they overwrite a visible value.
        run(4, 100.0, &[6.0, 2.0, 40.0], true).unwrap();
        run(5, 200.0, &[100.0, 6.0, 2.0], false).unwrap();
        assert_eq!(run(4, 8.0, &[6.0, 2.0, 40.0], true).unwrap(), [4.0, -2.0]);
        let base = [6.0, 2.0, 40.0];
        let mut expected = Vec::new();
        let mut serial_history = base.to_vec();
        for (position, value) in (4..).zip([8.0, 9.0, 10.0, 11.0]) {
            expected.push(run(position, value, &serial_history, true).unwrap());
            serial_history.insert(0, value);
            serial_history.truncate(3);
        }
        *attention.prefill.borrow_mut() = prefill::Workspace::allocate(&ctx).unwrap();
        let values = [8.0, 9.0, 10.0, 11.0].map(|value| [value; HEAD_DIM]);
        let rows = values.each_ref().map(|value| Row {
            query: &zero,
            key: &zero,
            value,
            relative: &[0.0],
            sinks: &[0.0],
            gates: &zero,
        });
        let keys = base.iter().map(|_| Arc::new(zero.to_vec())).collect();
        let values = base.iter().map(|&v| Arc::new(vec![v; HEAD_DIM])).collect();
        let projection = OutputProjection {
            weight: weight.range(0, weight.bytes()).unwrap(),
            dtype: DenseDtype::F32,
            rows: 2,
        };
        let actual = attention
            .step_prefill(
                &ctx,
                0,
                (&rows, None),
                History {
                    position: 4,
                    keys: &keys,
                    values: &values,
                },
                Some(&projection),
            )
            .unwrap();
        assert_eq!(actual, expected);

        let side_bytes = 4 * (HEAD_DIM + 1) * size_of::<f32>();
        let side = ctx.allocate(side_bytes).unwrap();
        let side_in = side.range(0, side_bytes).unwrap();
        let side_rows: [[f32; HEAD_DIM + 1]; 4] = std::array::from_fn(|r| {
            std::array::from_fn(|i| (i * (r + 1) % 11) as f32 * 0.125 - 0.5)
        });
        let host_rows = std::array::from_fn::<_, 4, _>(|i| Row {
            sinks: &side_rows[i][..1],
            gates: &side_rows[i][1..],
            ..rows[i]
        });
        let device_rows = rows.map(|row| Row {
            sinks: &[],
            gates: &[],
            ..row
        });
        let cursor = || attention.state.borrow().next_position[0];
        let batch = |position,
                     rows: &[Row<'_>],
                     side: Option<prefill::Sidecar<'_>>,
                     values: &VecDeque<_>,
                     tiled| {
            attention.step_prefill_dispatch(
                &ctx,
                0,
                (rows, side),
                History {
                    position,
                    keys: &keys,
                    values,
                },
                Some(&projection),
                tiled,
            )
        };
        let mut staged = Vec::new();
        extend_words(&mut staged, side_rows.as_flattened());
        side.write_at(&stream, 0, &staged).unwrap();
        let device_side = Some(prefill::Sidecar {
            range: side_in,
            query: None,
            kv: None,
        });
        let ring = || {
            let mut bytes = vec![0; 2 * cache_bytes];
            attention
                .arena
                .read_at(&attention.stream, table_bytes, &mut bytes)
                .unwrap();
            bytes
        };
        let bits = |outputs: Vec<Vec<f32>>| {
            outputs
                .into_iter()
                .flatten()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let tail_values = [11.0, 10.0, 9.0]
            .map(|v| Arc::new(vec![v; HEAD_DIM]))
            .into();
        let evaluate = |tiled, side: Option<prefill::Sidecar<'_>>| {
            // Manufacture an equal cursor from a different lineage before entering
            // each arm. Invalidate first, as prefix restore does in production.
            run(3, 100.0, &[200.0, 300.0, 400.0], true).unwrap();
            assert_eq!(cursor(), Some(4));
            attention.invalidate();
            let rows = if side.is_some() {
                &device_rows
            } else {
                &host_rows
            };
            let outputs = batch(4, rows, side, &values, tiled).unwrap();
            assert_eq!(cursor(), Some(8));
            let committed = ring();
            let tail = batch(8, &host_rows[..1], None, &tail_values, tiled).unwrap();
            assert_eq!(cursor(), Some(9));
            (bits(outputs), committed, bits(tail), ring())
        };
        let reference = evaluate(false, None);
        assert_eq!(evaluate(false, device_side), reference);
        assert_eq!(evaluate(true, None), reference);
        assert_eq!(evaluate(true, device_side), reference);
        assert!(batch(4, &device_rows[..3], device_side, &values, true).is_err());
        assert_eq!(cursor(), Some(9)); // rejected extent is pre-mutation

        let resident_bytes = host_rows.len() * projection.rows * size_of::<f32>();
        let resident = ctx.allocate(resident_bytes).unwrap();
        let publish = |len, convolution| {
            attention.step_prefill_to(
                &ctx,
                0,
                (&host_rows, None),
                History {
                    position: 4,
                    keys: &keys,
                    values: &values,
                },
                Some(&projection),
                (
                    true,
                    Some(prefill::ResidentOutput {
                        raw: resident.range(0, len).unwrap(),
                        convolution,
                    }),
                ),
            )
        };
        assert!(publish(resident_bytes - 4, None).is_err());
        assert_eq!(cursor(), Some(9));
        assert!(publish(resident_bytes, None).unwrap().is_empty());
        assert_eq!(cursor(), Some(8));
        let mut resident_output = vec![0; resident_bytes];
        resident.read_at(&stream, 0, &mut resident_output).unwrap();
        assert_eq!(
            resident_output,
            reference
                .0
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(ring(), reference.1);

        let past = [4.0, 5.0, 3.0, -2.5, 2.0, -0.0];
        let taps = [0.5, -0.25, 0.125, 2.0, -1.0, 0.25, -0.75, 0.0];
        let upload = |input: &[f32]| {
            let mut bytes = Vec::new();
            extend_words(&mut bytes, input);
            let buf = ctx.allocate(bytes.len()).unwrap();
            buf.write_at(&stream, 0, &bytes).unwrap();
            buf
        };
        let history = upload(&past);
        let filters = upload(&taps);
        let convolved = ctx.allocate(resident_bytes).unwrap();
        let conv = prefill::OutputConvolution {
            history: 3,
            planes: [&history, &filters, &convolved].map(|b| b.range(0, b.bytes()).unwrap()),
            read_convolved: false,
        };
        assert!(publish(resident_bytes, Some(conv)).unwrap().is_empty());
        assert_eq!(cursor(), Some(8));
        assert_eq!(ring(), reference.1);
        let raw: Vec<_> = reference.0.iter().copied().map(f32::from_bits).collect();
        let mut expected = Vec::new();
        for at in 0..raw.len() {
            let mut sum = 0.0;
            for lag in 0..4 {
                let source = if at / 2 >= lag {
                    raw[at - 2 * lag]
                } else {
                    past[2 * (3 + at / 2 - lag) + at % 2]
                };
                sum += taps[4 * (at % 2) + 3 - lag] * source;
            }
            expected.push((sum + raw[at]) * 5.0f32.sqrt().recip());
        }
        let mut expected_bytes = Vec::new();
        extend_words(&mut expected_bytes, &expected);
        let mut got = vec![0; resident_bytes];
        convolved.read_at(&stream, 0, &mut got).unwrap();
        assert_eq!(got, expected_bytes);
        resident.read_at(&stream, 0, &mut got).unwrap();
        assert_eq!(got, resident_output); // next decode consumes raw, not convolved, history

        // A singleton first tile forces the second tile to combine uploaded
        // history with the previous device tile without copying either.
        let run_convolution = |read_convolved| {
            attention.invalidate();
            let prior = past
                .as_chunks::<2>()
                .0
                .iter()
                .rev()
                .map(|pair| pair.as_slice().to_vec())
                .collect();
            let mut visit_conv = attention
                .admit_prefill_conv(&ctx, [4, 2], &prior, &taps)
                .unwrap()
                .unwrap();
            let mut outputs = Vec::new();
            for (span, position) in [(0..1, 4), (1..4, 5)] {
                assert!(visit_conv.convolved_plane().is_err());
                outputs.extend(
                    attention
                        .step_prefill_conv(
                            &ctx,
                            0,
                            (&host_rows[span], None),
                            History {
                                position,
                                keys: &keys,
                                values: &values,
                            },
                            &projection,
                            (&mut visit_conv, read_convolved),
                        )
                        .unwrap(),
                );
            }
            if read_convolved {
                assert_eq!(
                    bits(outputs),
                    expected
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
            } else {
                assert!(outputs.is_empty());
            }
            let tail = attention.prefill_conv_tail(visit_conv).unwrap();
            assert_eq!(
                bits(tail.into()),
                reference
                    .0
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .rev()
                    .take(3)
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
            );
            assert_eq!(ring(), reference.1);
        };
        run_convolution(true);
        run_convolution(false);

        filters
            .write_at(&stream, 0, &f32::NAN.to_ne_bytes())
            .unwrap();
        assert_eq!(
            publish(resident_bytes, Some(conv)),
            Err(bad("invalid CUDA relative attention batch state"))
        );
        assert_eq!(cursor(), None);

        // Both rewind and host-ahead cursor gaps must repair before reading.
        assert_eq!(
            bits(batch(4, &host_rows, None, &values, true).unwrap()),
            reference.0
        );
        attention.invalidate();
        run(2, 100.0, &[200.0, 300.0], true).unwrap();
        assert_eq!(cursor(), Some(3));
        assert_eq!(
            batch(4, &host_rows, None, &VecDeque::new(), true),
            Err(bad("missing host rows for CUDA relative attention restore"))
        );
        assert_eq!(cursor(), None);

        side.write_at(&stream, size_of::<f32>(), &f32::INFINITY.to_ne_bytes())
            .unwrap();
        run(3, 6.0, &[2.0, 40.0, 20.0], true).unwrap();
        let snapshot = ring();
        assert_eq!(
            batch(4, &device_rows, device_side, &values, true),
            Err(bad("invalid CUDA relative attention batch state"))
        );
        assert_eq!(cursor(), None);
        assert_eq!(ring(), snapshot); // attention status prevents physical commit

        weight
            .write_at(&stream, 0, &f32::NAN.to_ne_bytes())
            .unwrap();
        assert_eq!(
            batch(4, &host_rows, None, &values, true),
            Err(bad("non-finite CUDA relative attention batch output"))
        );
        assert_eq!(cursor(), None); // projection failed after physical commit
        assert_eq!(ring(), reference.1);
        weight.write_at(&stream, 0, &encoded).unwrap();
        assert_eq!(
            bits(batch(4, &host_rows, None, &values, true).unwrap()),
            reference.0
        );

        let q_scale = std::array::from_fn::<_, HEAD_DIM, _>(|i| (i % 4 + 1) as f32 * 0.25);
        let rel_scale = [0.75];
        let query = prefill::QueryClip {
            scales: [&q_scale, &rel_scale],
            limits: [1.25, 0.5],
            log: None,
        };
        let raw_q = std::array::from_fn::<_, 4, _>(|r| {
            std::array::from_fn::<_, HEAD_DIM, _>(|i| {
                ((i * 17 + r * 13) % 29) as f32 * 0.125 - 1.75
            })
        });
        let raw_rel = [[0.25], [2.5], [-1.0], [1.25]];
        let mut host_q = raw_q;
        let mut host_rel = raw_rel;
        for (q, rel) in host_q.iter_mut().zip(&mut host_rel) {
            for (q, s) in q.iter_mut().zip(q_scale) {
                *q *= s;
            }
            rel[0] *= rel_scale[0];
            crate::family::inkling2::cpu_attention::clip_rows(q, HEAD_DIM, query.limits[0]);
            crate::family::inkling2::cpu_attention::clip_rows(rel, 1, query.limits[1]);
        }
        let mut packed = Vec::new();
        let mut want = Vec::new();
        for i in 0..4 {
            extend_words(&mut packed, &raw_q[i]);
            extend_words(&mut packed, &raw_rel[i]);
            extend_words(&mut packed, &side_rows[i]);
            extend_words(&mut want, &host_q[i]);
            extend_words(&mut want, &host_rel[i]);
            extend_words(&mut want, &side_rows[i]);
        }
        // Exercise a borrowed sub-tile that starts at a nonzero byte offset.
        let start = packed.len() / 4;
        let resident = ctx.allocate(start + packed.len()).unwrap();
        resident.write_at(&stream, start, &packed).unwrap();
        let resident_side = || {
            Some(prefill::Sidecar {
                range: resident.range(start, packed.len()).unwrap(),
                query: Some(query),
                kv: None,
            })
        };
        let host_rows = std::array::from_fn::<_, 4, _>(|i| Row {
            query: &host_q[i],
            key: &side_rows[i][1..],
            relative: &host_rel[i],
            ..host_rows[i]
        });
        let device_rows = std::array::from_fn::<_, 4, _>(|i| Row {
            query: &[],
            relative: &[],
            key: host_rows[i].key,
            ..device_rows[i]
        });
        // A tiled prompt scores its queries on tensor cores and a row-by-row one in f32, so each
        // path compares the resident side with host rows on that same path.
        for tiled in [false, true] {
            attention.invalidate();
            let expected = bits(batch(4, &host_rows, None, &values, tiled).unwrap());
            let committed = ring();
            attention.invalidate();
            resident.write_at(&stream, start, &packed).unwrap();
            assert_eq!(
                bits(batch(4, &device_rows, resident_side(), &values, tiled).unwrap()),
                expected
            );
            assert_eq!(ring(), committed);
            assert_eq!(cursor(), Some(8));
            let mut got = vec![0; want.len()];
            resident.read_at(&stream, start, &mut got).unwrap();
            assert_eq!(
                got, want,
                "prepared query side is not bitwise equal to host clip"
            );
        }
        // Reject the resident subrange before touching the certified cursor.
        assert!(batch(4, &device_rows[..3], resident_side(), &values, true).is_err());
        assert_eq!(cursor(), Some(8));
        // Raw inf/NaN must be rejected even if clipping or sigmoid masks it.
        for (word, value) in [
            (0, f32::INFINITY),
            (HEAD_DIM, f32::NAN),
            (HEAD_DIM + 1, f32::INFINITY),
            (HEAD_DIM + 2, f32::NEG_INFINITY),
        ] {
            for tiled in [false, true] {
                run(3, 6.0, &[2.0, 40.0, 20.0], true).unwrap();
                let snapshot = ring();
                resident.write_at(&stream, start, &packed).unwrap();
                resident
                    .write_at(
                        &stream,
                        start + word * size_of::<f32>(),
                        &value.to_ne_bytes(),
                    )
                    .unwrap();
                assert_eq!(
                    batch(4, &device_rows, resident_side(), &values, tiled),
                    Err(bad("invalid CUDA relative attention batch state"))
                );
                assert_eq!(cursor(), None);
                if tiled {
                    assert_eq!(ring(), snapshot); // prep/health status prevents commit
                }
            }
        }
        assert!(run(5, f32::NAN, &[8.0, 6.0, 2.0], true).is_err());
        assert_eq!(
            run(POS_MAX as usize + 1, 1.0, &[], true),
            Err(bad("CUDA relative attention position overflows"))
        );
    }
}
