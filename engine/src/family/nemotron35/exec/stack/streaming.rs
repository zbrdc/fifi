//! Model-specific borrowed payloads for the shared NVMe transport.
//! Unlike the CPU mapper, this owner can unmap before targeted cache release.

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use super::WeightSource;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use super::{ExpertIndex, ExpertProjection};
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use crate::EngError;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use crate::expert_tier::{
    ExpertId, ExpertPayload, ExpertPayloadSource, ExpertPayloadView, Placement, Residency,
};
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use crate::family::nemotron35::loader::{ExpertLayerMapping, ProjectionRanges};
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use crate::family::qwen36::tiered::mapped_file::MappedFile;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::kernels::cuda::resident::expert_activation::ReluSquaredCall;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::kernels::cuda::resident::row_scale::{RowPow2Call, RowPow2Mode};
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use crate::kernels::nvfp4_cpu::Nvfp4Matrix;

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn stream_error(message: &str) -> EngError {
    EngError::MemoryPlan(format!("chunk expert stream: {message}"))
}

/// Mapped checkpoints are sized from headers; in-memory fixtures need only expert zero.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(super) fn stream_plan<S: WeightSource>(
    source: &mut S,
    layer: u32,
    experts: u32,
    max_pairs: usize,
) -> Result<crate::family::nemotron35::admission::ChunkStreamPlan, EngError> {
    let shape = if source.has_mapped_experts() {
        let mapping = source.expert_layer_mapping(layer)?;
        if mapping.experts.len() != experts as usize || experts == 0 {
            return Err(stream_error("mapped expert count differs from the model"));
        }
        let first = &mapping.experts[0].up;
        let shape = [first.rows, first.columns];
        for expert in &mapping.experts {
            for (part, expected) in [(&expert.up, shape), (&expert.down, [shape[1], shape[0]])] {
                let elements = expected[0]
                    .checked_mul(expected[1])
                    .ok_or_else(|| stream_error("projection size overflow"))?;
                if [part.rows, part.columns] != expected
                    || part.packed.len() != elements / 2
                    || part.scales.len() != elements / 16
                    || part.weight_scale_2.len() != 4
                    || [&part.packed, &part.scales, &part.weight_scale_2]
                        .iter()
                        .any(|range| range.start > range.end || range.end as u64 > mapping.file_len)
                {
                    return Err(stream_error("inconsistent mapped projection geometry"));
                }
            }
        }
        shape
    } else {
        let up = source.expert_packed(layer, ExpertIndex::Routed(0), ExpertProjection::Up)?;
        let down = source.expert_packed(layer, ExpertIndex::Routed(0), ExpertProjection::Down)?;
        if up.shape[0] != down.shape[1] || up.shape[1] != down.shape[0] {
            return Err(stream_error(
                "up/down projections are not transpose-compatible",
            ));
        }
        [up.shape[0] as usize, up.shape[1] as usize]
    };
    stream_memory_plan(experts, shape, max_pairs)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn stream_memory_plan(
    experts: u32,
    up_shape: [usize; 2],
    max_pairs: usize,
) -> Result<crate::family::nemotron35::admission::ChunkStreamPlan, EngError> {
    use crate::expert_tier::{StreamBankPlan, gpu_stream::MAX_STREAM_PAIRS};
    let [rows, columns] = up_shape;
    if experts == 0
        || !(1..=MAX_STREAM_PAIRS).contains(&max_pairs)
        || [rows, columns]
            .iter()
            .any(|&n| n == 0 || n > (1 << 22) || !n.is_multiple_of(16))
    {
        return Err(stream_error("unsupported stream geometry"));
    }
    if [rows, columns].iter().any(|&n| {
        n.checked_mul(max_pairs)
            .is_none_or(|v| v as u64 > (1u64 << 32))
    }) {
        return Err(stream_error("projection exceeds the row guard value limit"));
    }
    let overflow = || stream_error("stream storage size overflow");
    let elements = (rows as u64)
        .checked_mul(columns as u64)
        .ok_or_else(overflow)?;
    let part = (elements / 2)
        .checked_add(elements / 16)
        .ok_or_else(overflow)?;
    let parts = u64::from(experts).checked_mul(2).ok_or_else(overflow)?;
    let bank = part
        .checked_add(2 * 256)
        .and_then(|n| n.checked_mul(parts))
        .ok_or_else(overflow)?;
    // Three live projection buffers per direction.
    // shortcut: metadata is bounded at 1 KiB/pair; recalculate if grouped descriptor storage grows.
    let workspace = (rows as u64)
        .checked_add(columns as u64)
        .and_then(|n| n.checked_mul(3 * 4))
        .and_then(|n| n.checked_add(1024))
        .and_then(|n| n.checked_mul(max_pairs as u64))
        .and_then(|n| n.checked_add(12))
        .ok_or_else(overflow)?;
    let scratch = part
        .checked_add(2 * 256 + (1 << 20))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(workspace))
        .ok_or_else(overflow)?;
    Ok(crate::family::nemotron35::admission::ChunkStreamPlan {
        banks: StreamBankPlan::new(bank, workspace, scratch)?,
        max_pairs,
        up_shape,
    })
}

/// Reserve before context creation and hot placement; the unset/default path is unchanged.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) fn reserve_streaming<S: WeightSource>(
    spec: &super::NemotronExecSpec,
    source: &mut S,
    capacity: u32,
    admission: &mut crate::family::nemotron35::admission::cuda::CudaAdmission,
) -> Result<(), EngError> {
    if admission.stream_plan().is_some()
        || std::env::var_os("NEM35_CHUNK_EXPERTS").as_deref() != Some(std::ffi::OsStr::new("gpu"))
    {
        return Ok(());
    }
    let rows = super::chunk::parse_rows(std::env::var_os("NEM35_CUDA_CHUNK_PREFILL").as_deref())?
        .unwrap_or(0)
        .min(capacity as usize);
    let pairs = rows.saturating_mul(spec.moe.experts_per_token as usize);
    if !crate::expert_tier::gpu_stream::wants_device_stream(rows, pairs, Some("gpu")) {
        return Ok(());
    }
    let expected = [
        spec.moe.moe_intermediate_size as usize,
        spec.moe.latent_size as usize,
    ];
    let mut plan = None;
    for (layer, kind) in spec.blocks.iter().enumerate() {
        if *kind != crate::family::nemotron35::NemotronBlockKind::Moe {
            continue;
        }
        let next = stream_plan(source, layer as u32, spec.moe.routed_experts, pairs)?;
        if next.up_shape != expected || plan.is_some_and(|old| old != next) {
            return Err(stream_error("stream layout differs across model layers"));
        }
        plan = Some(next);
    }
    if let Some(plan) = plan {
        admission.reserve_streaming(plan)?;
    }
    Ok(())
}

/// The prefill-chunk device arm: stream the layer's routed experts over PCIe
/// once per layer per chunk and run the grouped NVFP4 kernel per (row, expert)
/// pair — up, guarded device relu-squared, down, then the host weighted sum. The
/// result contract matches the CPU pool's `routed_batch`: one latent row per
/// input row. The banks are sized once (one layer's routed bytes) and reused
/// for every later layer and chunk.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) fn gpu_stream_routed<S: WeightSource>(
    cuda: &mut crate::family::nemotron35::cuda_projection::CudaProjector,
    hybrid: &mut super::HybridRuntime,
    source: &mut S,
    layer: u32,
    experts: u32,
    xs: &[f32],
    picks: &[Vec<crate::expert_tier::ExpertPick>],
) -> Result<Vec<f32>, EngError> {
    use crate::expert_tier::{CudaStreamBanks, ExpertId, VRAM_HEADROOM_BYTES};

    let bad = |message: &str| EngError::BadRequest(format!("chunk expert stream: {message}"));
    let rows = picks.len();
    let Some(width) = xs.len().checked_div(rows) else {
        return Err(bad("a chunk needs at least one row"));
    };
    if width == 0 || !xs.len().is_multiple_of(rows) {
        return Err(bad("activation rows do not match the pick sets"));
    }
    // The pair list is row-major: row t's picks in selection order.
    let mut demand: Vec<ExpertId> = Vec::with_capacity(rows * 2);
    let mut pairs: Vec<(ExpertId, f32, usize)> = Vec::with_capacity(rows * 2);
    for (row, picks) in picks.iter().enumerate() {
        if picks.is_empty() {
            return Err(bad("every row needs at least one pick"));
        }
        for pick in picks {
            if pick.expert >= experts || !pick.weight.is_finite() {
                return Err(bad("invalid expert selection or nonfinite gate"));
            }
            let id = ExpertId {
                layer,
                expert: pick.expert,
            };
            demand.push(id);
            pairs.push((id, pick.weight, row));
        }
    }
    if pairs.len() > crate::expert_tier::gpu_stream::MAX_STREAM_PAIRS {
        return Err(bad("pair count exceeds the row guard limit"));
    }
    demand.sort_unstable();
    demand.dedup();

    let admitted = cuda
        .stream_plan()
        .ok_or_else(|| stream_error("streaming was not admitted"))?;
    if pairs.len() > admitted.max_pairs || width != admitted.up_shape[1] {
        return Err(stream_error(
            "chunk exceeds its admitted projection workspace",
        ));
    }
    let ctx = cuda.context();
    if hybrid.stream_banks.is_none() {
        let plan = admitted.banks;
        let free = u64::try_from(ctx.memory_info()?.free)
            .map_err(|_| bad("free device bytes overflow"))?;
        let required = plan
            .device_bytes()
            .checked_add(VRAM_HEADROOM_BYTES)
            .ok_or_else(|| bad("device reservation overflow"))?;
        eprintln!(
            "nem35_stream_admission experts={experts} bank_bytes={} device_bytes={} host_bytes={} \
             free_vram_bytes={free} required_vram_bytes={required}",
            plan.bank_bytes(),
            plan.device_bytes(),
            plan.host_bytes(),
        );
        if free < required {
            return Err(stream_error(
                "admitted banks/workspace no longer fit device headroom",
            ));
        }
        // Retained banks stay charged until the context's proven native cleanup.
        hybrid.stream_banks = Some(CudaStreamBanks::new(ctx, plan)?);
    }
    let banks = hybrid.stream_banks.as_mut().expect("just initialized");
    let (bank, _) = banks.slots_mut();
    let mut stream_source = Nvfp4StreamSource::new_any(source, layer)?;
    let lease = bank
        .stage_demand(ctx, &mut stream_source, &demand)?
        .ok_or_else(|| bad("the chunk demanded no experts"))?;

    // Up: one grouped launch over every pair; the output stays on the device.
    // (lease.stream() is taken per launch: the lease is finished below.)
    let up_shape = lease.part(demand[0], 0)?.shape;
    let up = [
        usize::try_from(up_shape[0]).map_err(|_| bad("up rows"))?,
        usize::try_from(up_shape[1]).map_err(|_| bad("up columns"))?,
    ];
    if up != admitted.up_shape || up[1] != width {
        return Err(bad(
            "up geometry differs from the admitted latent projection",
        ));
    }
    let mut up_ranges = Vec::with_capacity(pairs.len());
    let mut up_inputs = Vec::with_capacity(pairs.len() * up[1]);
    for (id, _, row) in &pairs {
        let part = lease.part(*id, 0)?;
        if part.shape != up_shape {
            return Err(bad("inconsistent streamed up geometry"));
        }
        let scales = part.scales.ok_or_else(|| bad("up part has no scales"))?;
        up_ranges.push(ctx.nvfp4_weight_range(part.data, scales, part.tensor_scale, up)?);
        up_inputs.extend_from_slice(&xs[row * width..(row + 1) * width]);
    }
    let up_out = crate::expert_tier::project_nvfp4_batch_launch(
        ctx,
        lease.stream(),
        &up_ranges,
        &up_inputs,
    )?;
    drop(up_ranges);

    // Device-side activation between the launches: relu-squared stays on
    // the device (no per-layer host round trip), then the shared row guard
    // (resident::row_scale) normalizes each pair row by an exact power of
    // two so the F16-operand down input stays inside range; the same scales
    // restore the down output before the readback (#5's ruled pattern:
    // guard -> grouped down -> undo, the scales never cross to the host).
    let intermediate = up[0];
    let values = pairs
        .len()
        .checked_mul(intermediate)
        .ok_or_else(|| bad("pair values overflow"))?;
    let value_bytes = values
        .checked_mul(4)
        .ok_or_else(|| bad("pair bytes overflow"))?;
    // The row guard's buffer geometry: rows x columns values, one f32 scale
    // per row, the same 1..=65536 x 1..=2^22 bounds it checks itself.
    let guard_bounds = |rows: usize, columns: usize| -> Result<usize, EngError> {
        let values = rows
            .checked_mul(columns)
            .filter(|values| *values <= (1usize << 32))
            .ok_or_else(|| bad("the row guard rejects this geometry"))?;
        if !(1..=crate::expert_tier::gpu_stream::MAX_STREAM_PAIRS).contains(&rows)
            || !(1..=(1 << 22)).contains(&columns)
        {
            return Err(bad("the row guard rejects this geometry"));
        }
        Ok(values)
    };
    let guard_values = guard_bounds(pairs.len(), intermediate)?;
    let guard_scales_bytes = pairs.len() * 4;
    let act = ctx.allocate(value_bytes)?;
    ctx.relu_squared(
        lease.stream(),
        ReluSquaredCall {
            out: act.range(0, value_bytes)?,
            x: up_out.range(0, value_bytes)?,
            values,
        },
    )?;
    let normalized = ctx.allocate(guard_values * 4)?;
    let guard_scales = ctx.allocate(guard_scales_bytes)?;
    // The guard's completion flag: initialized once, bit0 set on an invalid
    // row; checked after the stream drains.
    let guard_flag = ctx.upload(&0u32.to_ne_bytes())?;
    ctx.row_pow2(
        lease.stream(),
        RowPow2Call {
            x: act.range(0, value_bytes)?,
            out: normalized.range(0, guard_values * 4)?,
            scales: guard_scales.range(0, guard_scales_bytes)?,
            flag: guard_flag.range(0, 4)?,
            rows: pairs.len(),
            columns: intermediate,
            mode: RowPow2Mode::Normalize,
        },
    )?;

    // Down: one grouped launch reading the guarded activation on the device.
    let down_shape = lease.part(demand[0], 1)?.shape;
    let down = [
        usize::try_from(down_shape[0]).map_err(|_| bad("down rows"))?,
        usize::try_from(down_shape[1]).map_err(|_| bad("down columns"))?,
    ];
    if down != [width, intermediate] {
        return Err(bad("down geometry does not match the up projection"));
    }
    let mut down_ranges = Vec::with_capacity(pairs.len());
    for (id, _, _) in &pairs {
        let part = lease.part(*id, 1)?;
        if part.shape != down_shape {
            return Err(bad("inconsistent streamed down geometry"));
        }
        let scales = part.scales.ok_or_else(|| bad("down part has no scales"))?;
        down_ranges.push(ctx.nvfp4_weight_range(part.data, scales, part.tensor_scale, down)?);
    }
    let down_gpu = crate::expert_tier::project_nvfp4_batch_device(
        ctx,
        lease.stream(),
        &down_ranges,
        &normalized,
        value_bytes,
    )?;
    drop(down_ranges);

    // Undo the guard on the device (the same scales, the down output's own
    // column count), then one readback rides the lease's stream before
    // finish (which consumes the lease and waits for its bank).
    let restore_values = guard_bounds(pairs.len(), width)?;
    let restored = ctx.allocate(restore_values * 4)?;
    ctx.row_pow2(
        lease.stream(),
        RowPow2Call {
            x: down_gpu.range(0, restore_values * 4)?,
            out: restored.range(0, restore_values * 4)?,
            scales: guard_scales.range(0, guard_scales_bytes)?,
            flag: guard_flag.range(0, 4)?,
            rows: pairs.len(),
            columns: width,
            mode: RowPow2Mode::Restore,
        },
    )?;
    let down_out =
        crate::expert_tier::project_nvfp4_read(ctx, lease.stream(), &restored, restore_values)?;
    let mut guard_flag_bytes = [0_u8; 4];
    guard_flag.read_at(lease.stream(), 0, &mut guard_flag_bytes)?;
    lease.finish()?;
    if u32::from_ne_bytes(guard_flag_bytes) & 1 != 0 {
        return Err(bad("the row guard rejected a non-finite activation row"));
    }

    // The weighted per-row sum: the CPU pool's accumulation contract (the
    // guard's undo already ran on the device).
    let mut routed = vec![0.0_f32; xs.len()];
    for (index, (_, weight, row)) in pairs.iter().enumerate() {
        let output = &down_out[index * width..(index + 1) * width];
        let target = &mut routed[row * width..(row + 1) * width];
        for (value, target) in output.iter().zip(target) {
            *target += weight * value;
        }
    }
    Ok(routed)
}

/// Resolves checkpoint keys in pair order without downloading or cloning weight bytes.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) fn resident_ranges<'a>(
    projector: &'a crate::family::nemotron35::cuda_projection::CudaProjector,
    keys: &[u64],
    shape: [u32; 2],
) -> Result<Vec<crate::kernels::cuda::Nvfp4WeightRange<'a>>, EngError> {
    use crate::kernels::nvfp4::GROUP_SIZE;
    let bad = || EngError::BadRequest("Nemotron GPU expert view shape or payload mismatch".into());
    let shape = [
        usize::try_from(shape[0]).map_err(|_| bad())?,
        usize::try_from(shape[1]).map_err(|_| bad())?,
    ];
    if shape[0] == 0 || shape[1] == 0 || !shape[1].is_multiple_of(GROUP_SIZE) {
        return Err(bad());
    }
    let elements = shape[0].checked_mul(shape[1]).ok_or_else(bad)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(keys.len())
        .map_err(|_| EngError::MemoryPlan("allocate Nemotron GPU expert views".into()))?;
    for &key in keys {
        let view = projector.packed_expert_view(key).ok_or_else(|| {
            EngError::Checkpoint(format!("GPU expert key {key} has no borrowed packed view"))
        })?;
        if view.packed.bytes() != elements / 2 || view.scale.bytes() != elements / GROUP_SIZE {
            return Err(bad());
        }
        weights.push(projector.context().nvfp4_weight_range(
            view.packed.range(0, view.packed.bytes())?,
            view.scale.range(0, view.scale.bytes())?,
            view.weight_scale_2,
            shape,
        )?);
    }
    Ok(weights)
}

/// Keeps model admission and timing around the shared format kernel.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) fn project_grouped(
    projector: &crate::family::nemotron35::cuda_projection::CudaProjector,
    stream: &crate::kernels::cuda::DeviceStream,
    weights: &[crate::kernels::cuda::Nvfp4WeightRange<'_>],
    inputs: &[f32],
) -> Result<Vec<f32>, EngError> {
    use crate::family::nemotron35::transport_accounting::{self, Category};
    let shape = weights
        .first()
        .ok_or_else(|| EngError::BadRequest("NVFP4 routed batch needs a pair".into()))?
        .shape();
    let [Ok(rows), Ok(columns)] = shape.map(u32::try_from) else {
        return Err(EngError::BadRequest(
            "NVFP4 shape exceeds model limits".into(),
        ));
    };
    let _call = projector.begin_projection([rows, columns])?;
    let output =
        crate::expert_tier::project_nvfp4_batch(projector.context(), stream, weights, inputs)?;
    projector
        .grouped_launches
        .set(projector.grouped_launches.get() + 1);
    for _ in weights {
        transport_accounting::record_resident_call(Category::Routed);
    }
    Ok(output)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(super) struct Nvfp4StreamSource<'a, S: WeightSource> {
    source: &'a mut S,
    layer: u32,
    /// None: the prefill stream serves every routed expert of the layer
    /// (placement tiers are untouched — this is transport, not residency).
    residency: Option<&'a Residency>,
    /// None: the packed mode serves `expert_packed` views directly (no shard
    /// mapping) — in-memory fixtures and the mapped checkpoint both work.
    mapping: Option<ExpertLayerMapping>,
    mapped: Option<MappedFile>,
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn checkpoint(error: impl std::fmt::Display) -> EngError {
    EngError::Checkpoint(format!("Nemotron NVMe source: {error}"))
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn projection(part: usize) -> Result<ExpertProjection, EngError> {
    match part {
        0 => Ok(ExpertProjection::Up),
        1 => Ok(ExpertProjection::Down),
        _ => Err(EngError::BadRequest(
            "NVFP4 expert part must be up or down".into(),
        )),
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
impl<'a, S: WeightSource> Nvfp4StreamSource<'a, S> {
    /// Every routed expert of the layer, regardless of placement (the
    /// prefill chunk stream). The packed mode serves the source's borrowed
    /// `expert_packed` views — no shard mapping, so in-memory fixtures and
    /// the mapped checkpoint both work.
    #[cfg_attr(
        not(all(feature = "native-cuda", target_os = "linux")),
        allow(dead_code)
    )]
    pub(super) fn new_any(source: &'a mut S, layer: u32) -> Result<Self, EngError> {
        let mapping = if source.has_mapped_experts() {
            Some(source.expert_layer_mapping(layer)?)
        } else {
            None
        };
        Ok(Self {
            source,
            layer,
            residency: None,
            mapping,
            mapped: None,
        })
    }

    #[cfg(test)]
    pub(super) fn new(
        source: &'a mut S,
        layer: u32,
        residency: &'a Residency,
    ) -> Result<Self, EngError> {
        let mapping = source.expert_layer_mapping(layer)?;
        Ok(Self {
            source,
            layer,
            residency: Some(residency),
            mapping: Some(mapping),
            mapped: None,
        })
    }

    fn index(&self, id: ExpertId) -> Result<usize, EngError> {
        let index = usize::try_from(id.expert).map_err(checkpoint)?;
        if id.layer != self.layer {
            return Err(checkpoint("unknown routed expert"));
        }
        if let Some(mapping) = self.mapping.as_ref()
            && index >= mapping.experts.len()
        {
            return Err(checkpoint("unknown routed expert"));
        }
        if let Some(residency) = self.residency
            && residency.placement(id) != Placement::Nvme
        {
            return Err(EngError::BadRequest(
                "stream source only serves NVMe experts".into(),
            ));
        }
        Ok(index)
    }

    fn experts(&self, ids: &[ExpertId]) -> Result<Vec<u32>, EngError> {
        let mut experts = Vec::new();
        experts.try_reserve_exact(ids.len()).map_err(|error| {
            EngError::MemoryPlan(format!("allocate streamed expert ids: {error}"))
        })?;
        for &id in ids {
            self.index(id)?;
            experts.push(id.expert);
        }
        Ok(experts)
    }

    fn open(&mut self) -> Result<(), EngError> {
        if let Some(mapping) = self.mapping.as_ref()
            && self.mapped.is_none()
        {
            self.mapped =
                Some(MappedFile::open(&mapping.path, mapping.file_len).map_err(checkpoint)?);
        }
        Ok(())
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn mapped_matrix<'a>(
    bytes: &'a [u8],
    ranges: &ProjectionRanges,
) -> Result<Nvfp4Matrix<'a>, EngError> {
    let slice = |range: &std::ops::Range<usize>| {
        bytes
            .get(range.clone())
            .ok_or_else(|| checkpoint("expert range exceeds the shard"))
    };
    let scale: [u8; 4] = slice(&ranges.weight_scale_2)?
        .try_into()
        .map_err(|_| checkpoint("weight_scale_2 must be four bytes"))?;
    Nvfp4Matrix::new(
        slice(&ranges.packed)?,
        slice(&ranges.scales)?,
        f32::from_le_bytes(scale),
        ranges.rows,
        ranges.columns,
    )
    .map_err(checkpoint)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
impl<S: WeightSource> ExpertPayloadSource for Nvfp4StreamSource<'_, S> {
    fn parts(&self) -> usize {
        2
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        self.index(id)?;
        let packed = self.source.expert_packed(
            self.layer,
            ExpertIndex::Routed(id.expert),
            projection(part)?,
        )?;
        Ok(ExpertPayload {
            data: packed.packed,
            scales: packed.scale,
            tensor_scale: packed.weight_scale_2,
            shape: packed.shape,
        })
    }

    fn with_payload<R>(
        &mut self,
        id: ExpertId,
        part: usize,
        consume: impl FnOnce(ExpertPayloadView<'_>) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        let index = self.index(id)?;
        let which = projection(part)?;
        self.open()?;
        let Some(mapping) = self.mapping.as_ref() else {
            // The packed mode: the same borrowed views `payload` uses.
            let packed =
                self.source
                    .expert_packed(self.layer, ExpertIndex::Routed(id.expert), which)?;
            return consume(ExpertPayloadView {
                data: &packed.packed,
                scales: &packed.scale,
                tensor_scale: packed.weight_scale_2,
                shape: packed.shape,
            });
        };
        let mapped = self.mapped.as_ref().expect("open initialized the mapping");
        mapped.check_unchanged().map_err(checkpoint)?;
        let ranges = match which {
            ExpertProjection::Up => &mapping.experts[index].up,
            ExpertProjection::Down => &mapping.experts[index].down,
        };
        let matrix = mapped_matrix(mapped.as_ref(), ranges)?;
        let shape = [
            u32::try_from(matrix.rows()).map_err(checkpoint)?,
            u32::try_from(matrix.cols()).map_err(checkpoint)?,
        ];
        let output = consume(ExpertPayloadView {
            data: matrix.packed(),
            scales: matrix.scales(),
            tensor_scale: matrix.weight_scale_2(),
            shape,
        })?;
        mapped.check_unchanged().map_err(checkpoint)?;
        Ok(output)
    }

    fn prefetch_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
        let experts = self.experts(ids)?;
        if experts.is_empty() {
            return Ok(());
        }
        self.source.prefetch_routed_experts(self.layer, &experts)?;
        self.open()
    }

    fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
        let experts = self.experts(ids)?;
        if experts.is_empty() {
            return Ok(());
        }
        // Borrowed bytes are already copied into the bank; DMA no longer reads the mapping.
        drop(self.mapped.take());
        if self.mapping.is_none() {
            // Packed mode borrows in-memory views; there is nothing to unmap.
            return Ok(());
        }
        self.source.release_routed_experts(self.layer, &experts)
    }

    fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
        Err(EngError::BadRequest(
            "NVMe source requires expert-granular release".into(),
        ))
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_tests {
    use super::{project_grouped, resident_ranges};
    use crate::{
        EngError,
        family::nemotron35::{
            admission::{
                self, ACTIVATION_WORKSPACE_BYTES,
                cuda::{CudaPlanInput, admit_cuda_plan},
            },
            cuda_projection::CudaProjector,
        },
        memory::MemoryPolicy,
    };

    fn projector() -> Result<CudaProjector, EngError> {
        let largest = 16 * 32 * 4;
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes: largest * 4 + ACTIVATION_WORKSPACE_BYTES + 4096,
                largest_tensor_bytes: largest,
                packed_pair_bytes: 256 + 32,
                max_shape: [16, 32],
                device_weight_cache_bytes: 4096,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes()
                .ok_or_else(|| EngError::MemoryPlan("host availability unavailable".into()))?,
        )?;
        CudaProjector::new(admission)
    }

    fn loaded_projector() -> Result<CudaProjector, EngError> {
        let mut projector = projector()?;
        projector.load_resident_expert(7, &[0x11; 256], &[0x38; 32], 1.0, [16, 32])?;
        projector.ensure_packed_resident(9, &[0x22; 256], &[0x38; 32], 1.0, [16, 32])?;
        projector.ensure_packed_resident(11, &[0x11; 256], &[0x38; 32], 1.0, [16, 32])?;
        Ok(projector)
    }

    #[test]
    #[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
    fn published_view_covers_permanent_and_cached_experts() {
        let observed = (|| -> Result<_, EngError> {
            let mut projector = loaded_projector()?;
            let control = projector
                .resident_nvfp4_matvec(7, [16, 32], &[0.5; 32])
                .ok_or_else(|| {
                    EngError::Checkpoint("fixture GPU expert is not resident".into())
                })??;
            let permanent = projector.packed_expert_view(7).is_some();
            let cached = projector.packed_expert_view(9).is_some();
            Ok((control, permanent, cached))
        })();
        let (control, permanent, cached) =
            observed.expect("resident projection and published views");
        assert_eq!(control, vec![8.0; 16]);
        assert!(cached, "positive LRU cache control has no view");
        assert!(permanent, "a preloaded GPU-hot key has no borrowed view");
    }

    #[test]
    #[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
    fn permanent_view_wins_collision_and_reports_exact_bytes() {
        let observed = (|| -> Result<_, EngError> {
            let mut projector = loaded_projector()?;
            // Legacy overlapping keys must still select permanent residency.
            projector.load_resident_expert(9, &[0x11; 256], &[0x38; 32], 2.0, [16, 32])?;
            let permanent = projector
                .packed_expert_view(9)
                .ok_or_else(|| EngError::Checkpoint("missing colliding permanent view".into()))?;
            let scale = permanent.weight_scale_2;
            let permanent_bytes = permanent.bytes;
            let cached_bytes = projector
                .packed_expert_view(11)
                .ok_or_else(|| EngError::Checkpoint("missing cached control view".into()))?
                .bytes;
            let weights = resident_ranges(&projector, &[9, 11], [16, 32])?;
            let stream = projector.context().stream()?;
            let output = project_grouped(&projector, &stream, &weights, &[0.5; 64])?;
            projector.context().ensure_usable()?;
            Ok((output, scale, permanent_bytes, cached_bytes))
        })();
        let (output, scale, permanent_bytes, cached_bytes) =
            observed.expect("collision projection");
        assert_eq!(scale, 2.0);
        assert_eq!((permanent_bytes, cached_bytes), (288, 288));
        let expected: Vec<f32> = [16.0, 8.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 16))
            .collect();
        assert_eq!(output, expected);
    }

    fn assert_callback(keys: &[u64]) {
        let observed = (|| -> Result<_, EngError> {
            let projector = loaded_projector()?;
            assert!(matches!(
                resident_ranges(&projector, keys, [8, 32]),
                Err(EngError::BadRequest(_))
            ));
            let weights = resident_ranges(&projector, keys, [16, 32])?;
            let stream = projector.context().stream()?;
            let inputs: Vec<f32> = [0.125, 0.5, -0.25]
                .into_iter()
                .flat_map(|value| std::iter::repeat_n(value, 32))
                .collect();
            let before = projector.grouped_launch_count();
            let before_clock = projector.stats();
            let bad = project_grouped(&projector, &stream, &weights, &inputs[..95]);
            let after_bad = projector.grouped_launch_count();
            let output = project_grouped(&projector, &stream, &weights, &inputs)?;
            let after_good = projector.grouped_launch_count();
            projector.context().ensure_usable()?;
            Ok((
                output,
                bad,
                before,
                after_bad,
                after_good,
                before_clock,
                projector.stats(),
            ))
        })();
        let (output, bad, before, after_bad, after_good, before_clock, after_clock) =
            observed.expect("borrowed-view callback");
        let expected: Vec<f32> = [4.0, 8.0, -8.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 16))
            .collect();
        assert_eq!(output, expected);
        assert!(matches!(bad, Err(EngError::BadRequest(_))));
        assert_eq!(after_bad, before);
        assert_eq!(after_good, before + 1);
        assert_eq!(after_clock.calls, before_clock.calls + 2);
        assert!(after_clock.call > before_clock.call);
    }

    #[test]
    #[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
    fn projection_guard_preserves_shape_admission_and_clock() {
        let observed = (|| -> Result<_, EngError> {
            let projector = projector()?;
            let before = projector.stats();
            let guard = projector.begin_projection([16, 32])?;
            let while_live = projector.stats();
            drop(guard);
            let after_valid = projector.stats();
            let refused = matches!(
                projector.begin_projection([17, 32]),
                Err(EngError::MemoryPlan(_))
            );
            Ok((before, while_live, after_valid, projector.stats(), refused))
        })();
        let (before, while_live, after_valid, after_refusal, refused) =
            observed.expect("projection guard");
        assert_eq!(while_live, before);
        assert_eq!(after_valid.calls, before.calls + 1);
        assert!(refused, "oversized shape bypassed admitted limits");
        assert_eq!(after_refusal.calls, after_valid.calls + 1);
        assert!(after_refusal.call >= after_valid.call);
    }

    #[test]
    #[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
    fn grouped_callback_keeps_pair_order_and_counts_success_only() {
        assert_callback(&[9, 7, 9]);
    }

    #[test]
    #[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
    fn cached_callback_is_a_positive_control_for_the_shared_kernel() {
        assert_callback(&[9, 11, 9]);
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::{Nvfp4StreamSource, stream_plan};
    use crate::EngError;
    use crate::expert_tier::{ExpertId, ExpertPayloadSource, Placement, Residency};
    use crate::family::nemotron35::{
        exec::{ExpertIndex, ExpertProjection, WeightSource},
        loader::{ExpertLayerMapping, ExpertRanges, ProjectionRanges},
    };
    use crate::tensor::Tensor;

    struct Source {
        _file: tempfile::NamedTempFile,
        mapping: ExpertLayerMapping,
        prefetched: Vec<(u32, Vec<u32>)>,
        released: Vec<(u32, Vec<u32>)>,
    }

    impl WeightSource for Source {
        fn dense(&mut self, _: &str) -> Result<Tensor, EngError> {
            Err(EngError::Checkpoint("unexpected dense read".into()))
        }
        fn expert(
            &mut self,
            _: u32,
            _: ExpertIndex,
            _: ExpertProjection,
        ) -> Result<Tensor, EngError> {
            Err(EngError::Checkpoint(
                "unexpected decoded expert read".into(),
            ))
        }
        fn has_packed_experts(&self) -> bool {
            true
        }
        fn has_mapped_experts(&self) -> bool {
            true
        }
        fn expert_layer_mapping(&mut self, layer: u32) -> Result<ExpertLayerMapping, EngError> {
            if layer != 5 {
                return Err(EngError::Checkpoint("unknown mapped layer".into()));
            }
            Ok(self.mapping.clone())
        }
        fn prefetch_routed_experts(&mut self, layer: u32, experts: &[u32]) -> Result<(), EngError> {
            self.prefetched.push((layer, experts.to_vec()));
            Ok(())
        }
        fn release_routed_experts(&mut self, layer: u32, experts: &[u32]) -> Result<(), EngError> {
            self.released.push((layer, experts.to_vec()));
            Ok(())
        }
    }

    fn fixture() -> Source {
        let mut bytes = Vec::new();
        let mut experts = Vec::new();
        for expert in 0..3u8 {
            let mut parts = Vec::new();
            for part in 0..2u8 {
                let start = bytes.len();
                bytes.extend(std::iter::repeat_n(expert * 16 + part + 1, 256));
                let packed = start..bytes.len();
                let start = bytes.len();
                bytes.extend(std::iter::repeat_n(0x38, 32));
                let scales = start..bytes.len();
                let start = bytes.len();
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
                parts.push(ProjectionRanges {
                    packed,
                    scales,
                    weight_scale_2: start..bytes.len(),
                    rows: 16,
                    columns: 32,
                });
            }
            let down = parts.pop().unwrap();
            let up = parts.pop().unwrap();
            experts.push(ExpertRanges { up, down });
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        Source {
            mapping: ExpertLayerMapping {
                path: file.path().to_owned(),
                file_len: bytes.len() as u64,
                experts,
            },
            _file: file,
            prefetched: Vec::new(),
            released: Vec::new(),
        }
    }

    #[test]
    fn stream_sizing_uses_mapped_metadata_without_reading_payloads() {
        let mut source = fixture();
        for expert in &mut source.mapping.experts {
            expert.down.rows = 32;
            expert.down.columns = 16;
        }
        let plan = stream_plan(&mut source, 5, 3, 16).unwrap();
        assert_eq!(plan.up_shape, [16, 32]);
        assert_eq!(plan.max_pairs, 16);
        assert_eq!(plan.banks.bank_bytes(), 4800);
        assert_eq!(plan.banks.device_bytes(), 35_212);
        assert_eq!(plan.banks.host_bytes(), 2_133_964);
        assert!(stream_plan(&mut source, 5, 4, 16).is_err());
        assert!(stream_plan(&mut source, 5, 3, 65_537).is_err());
        source.mapping.experts[1].down.rows = 16;
        assert!(stream_plan(&mut source, 5, 3, 16).is_err());
        source.mapping.experts[1].down.rows = 32;
        source.mapping.experts[2].up.packed.end -= 1;
        assert!(stream_plan(&mut source, 5, 3, 16).is_err());
    }

    #[test]
    fn stream_workspace_rejects_zero_overflow_and_row_guard_extents() {
        for (experts, shape, pairs) in [
            (0, [32, 32], 16),
            (3, [32, 32], 0),
            (3, [31, 32], 16),
            (3, [1 << 22, 1 << 22], 65_536),
            (u32::MAX, [1 << 22, 1 << 22], 1),
        ] {
            assert!(super::stream_memory_plan(experts, shape, pairs).is_err());
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    #[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
    fn mapped_chunk_stream_releases_exact_demand_without_residency() {
        use crate::family::nemotron35::{
            admission::cuda::{CudaPlanInput, admit_cuda_plan},
            cuda_projection::CudaProjector,
        };
        let _gpu = super::super::gpu_test_lock();
        let mut source = fixture();
        for expert in &mut source.mapping.experts {
            expert.down.rows = 32;
            expert.down.columns = 16;
        }
        let rows = 12;
        let xs = vec![0.125; rows * 32];
        let picks = vec![
            vec![
                crate::expert_tier::ExpertPick {
                    expert: 0,
                    weight: 0.25
                },
                crate::expert_tier::ExpertPick {
                    expert: 2,
                    weight: 0.75
                },
            ];
            rows
        ];
        let mut want = vec![0.0; xs.len()];
        {
            let mut view = Nvfp4StreamSource::new_any(&mut source, 5).unwrap();
            for (input, output) in xs
                .as_chunks::<32>()
                .0
                .iter()
                .zip(want.as_chunks_mut::<32>().0.iter_mut())
            {
                for (expert, weight) in [(0, 0.25), (2, 0.75)] {
                    let id = ExpertId { layer: 5, expert };
                    let mut up = [0.0; 16];
                    view.with_payload(id, 0, |p| {
                        let matrix =
                            super::Nvfp4Matrix::new(p.data, p.scales, p.tensor_scale, 16, 32)
                                .unwrap();
                        matrix
                            .matvec(crate::kernels::nvfp4_cpu::Isa::Scalar, input, &mut up)
                            .unwrap();
                        Ok(())
                    })
                    .unwrap();
                    for value in &mut up {
                        *value = value.max(0.0).powi(2);
                    }
                    let mut down = [0.0; 32];
                    view.with_payload(id, 1, |p| {
                        let matrix =
                            super::Nvfp4Matrix::new(p.data, p.scales, p.tensor_scale, 32, 16)
                                .unwrap();
                        matrix
                            .matvec(crate::kernels::nvfp4_cpu::Isa::Scalar, &up, &mut down)
                            .unwrap();
                        Ok(())
                    })
                    .unwrap();
                    for (out, value) in output.iter_mut().zip(down) {
                        *out += weight * value;
                    }
                }
            }
        }
        let plan = stream_plan(&mut source, 5, 3, rows * 2).unwrap();
        let mut ticket = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes: 1 << 20,
                largest_tensor_bytes: 4096,
                packed_pair_bytes: 288,
                max_shape: [32, 32],
                device_weight_cache_bytes: 0,
            },
            crate::memory::MemoryPolicy::default_share(),
            u64::MAX,
        )
        .unwrap();
        ticket.reserve_streaming(plan).unwrap();
        let mut cuda = CudaProjector::new(ticket).unwrap();
        let cpu = crate::family::nemotron35::cpu_experts::new_cpu_experts(1).unwrap();
        let mut runtime = super::super::HybridRuntime::new(cpu, false);
        let got = super::gpu_stream_routed(&mut cuda, &mut runtime, &mut source, 5, 3, &xs, &picks)
            .unwrap();
        assert_eq!(got, want);
        assert_eq!(source.prefetched, [(5, vec![0, 2])]);
        assert_eq!(source.released, [(5, vec![0, 2])]);
    }

    #[test]
    fn in_memory_packed_source_does_not_require_file_mapping() {
        struct Packed;
        impl WeightSource for Packed {
            fn dense(&mut self, _: &str) -> Result<Tensor, EngError> {
                unreachable!()
            }
            fn expert(
                &mut self,
                _: u32,
                _: ExpertIndex,
                _: ExpertProjection,
            ) -> Result<Tensor, EngError> {
                unreachable!()
            }
            fn has_packed_experts(&self) -> bool {
                true
            }
            fn expert_packed(
                &mut self,
                _: u32,
                _: ExpertIndex,
                projection: ExpertProjection,
            ) -> Result<crate::family::nemotron35::exec::PackedExpert, EngError> {
                Ok(crate::family::nemotron35::exec::PackedExpert {
                    packed: vec![0x11; 256],
                    scale: vec![0x38; 32],
                    weight_scale_2: 1.0,
                    input_scale: 1.0,
                    shape: if projection == ExpertProjection::Up {
                        [16, 32]
                    } else {
                        [32, 16]
                    },
                })
            }
        }
        let mut source = Packed;
        assert!(stream_plan(&mut source, 5, 3, 24).is_ok());
        let mut view = Nvfp4StreamSource::new_any(&mut source, 5).unwrap();
        view.with_payload(
            ExpertId {
                layer: 5,
                expert: 2,
            },
            0,
            |p| {
                assert_eq!(p.shape, [16, 32]);
                assert_eq!(p.data, [0x11; 256]);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn mapping_errors_are_not_packed_fallback() {
        let mut source = fixture();
        assert!(matches!(
            Nvfp4StreamSource::new_any(&mut source, 4),
            Err(EngError::Checkpoint(_))
        ));
        assert!(source.prefetched.is_empty());
        assert!(source.released.is_empty());
    }

    #[test]
    fn prefill_mapping_releases_only_its_demand_without_residency() {
        let mut source = fixture();
        let ids = [0, 2].map(|expert| ExpertId { layer: 5, expert });
        let mut view = Nvfp4StreamSource::new_any(&mut source, 5).unwrap();
        for (id, expected) in ids.into_iter().zip([0x01, 0x21]) {
            assert_eq!(
                view.with_payload(id, 0, |part| Ok(part.data[0])).unwrap(),
                expected
            );
        }
        assert!(view.mapped.is_some());
        view.release_experts(&ids).unwrap();
        assert!(view.mapped.is_none());
        drop(view);
        assert_eq!(source.released, [(5, vec![0, 2])]);
    }

    #[test]
    fn mapped_tail_borrows_both_parts_and_releases_only_its_ids() {
        let mut source = fixture();
        let ids = [0, 1, 2].map(|expert| ExpertId { layer: 5, expert });
        let mut residency = Residency::default();
        residency
            .set_placement(ids[0], 288, Placement::Cpu)
            .unwrap();
        residency.mark_expert(5, 2, 288);
        let mut view = Nvfp4StreamSource::new(&mut source, 5, &residency).unwrap();
        view.prefetch_experts(&[ids[1]]).unwrap();
        let up = view
            .with_payload(ids[1], 0, |part| {
                Ok((
                    part.data[0],
                    part.data.len(),
                    part.scales[0],
                    part.tensor_scale,
                    part.shape,
                ))
            })
            .unwrap();
        let down = view
            .with_payload(ids[1], 1, |part| Ok(part.data[0]))
            .unwrap();
        assert_eq!(up, (0x11, 256, 0x38, 1.0, [16, 32]));
        assert_eq!(down, 0x12);
        assert!(view.release_experts(&[ids[0]]).is_err());
        assert!(
            view.mapped.is_some(),
            "an invalid release must not discard the current map"
        );
        view.release_experts(&[ids[1]]).unwrap();
        assert!(
            view.mapped.is_none(),
            "the temporary map must drop before cache release"
        );
        let again = view
            .with_payload(ids[1], 0, |part| Ok(part.data[0]))
            .unwrap();
        assert_eq!(again, 0x11);
        view.release_experts(&[ids[1]]).unwrap();
        for id in [
            ids[0],
            ids[2],
            ExpertId {
                layer: 4,
                expert: 1,
            },
            ExpertId {
                layer: 5,
                expert: 9,
            },
        ] {
            assert!(view.with_payload(id, 0, |_| Ok(())).is_err());
            assert!(view.prefetch_experts(&[id]).is_err());
            assert!(view.release_experts(&[id]).is_err());
        }
        assert!(view.with_payload(ids[1], 2, |_| Ok(())).is_err());
        assert!(view.release_layer(5).is_err());
        drop(view);
        assert_eq!(source.prefetched, [(5, vec![1])]);
        assert_eq!(source.released, [(5, vec![1]), (5, vec![1])]);
    }
}
