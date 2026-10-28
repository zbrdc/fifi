//! One immutable pinned source and one authoritative cache map per model.
//!
//! Six routed experts per row can demand at most 37 rows in 224 active slots:
//! C32 needs 192 slots; C64 needs 384 and is refused before weight allocation.

use std::{
    cell::{Cell, RefCell},
    path::Path,
};

use super::super::{
    FfnRoles, FlintResidentLimits,
    cache::{EXPERTS, HostBudget, HostStore, LayerCache, MissCopy},
    plan::plan_resident,
};
use super::{
    FlintExl3Resident, FlintExpertBank, FlintExpertView, PackedMatrix, PackedRange,
    packing::{CachedPacking, Packing},
};
use crate::{
    EngError,
    family::project::cuda::Exl3MatrixView,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, PinnedBuf, plan},
    weights::exl3::{Checkpoint, Matrix, flint_pack::Geometry},
};

pub(super) struct HostOwner {
    state: RefCell<HostState>,
    poisoned: Cell<bool>,
    layers: Vec<usize>,
    arena_base: usize,
}

struct HostState {
    pinned: PinnedBuf,
    store: HostStore,
    caches: Vec<LayerCache>,
    stream: u64,
}

fn refused(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint host cache: {message}"))
}

fn overflow() -> EngError {
    EngError::MemoryPlan("Flint host cache byte count overflows".into())
}

fn admit_cached(
    context: &CudaContext,
    cached: &CachedPacking,
    limits: FlintResidentLimits,
    config_bytes: usize,
) -> Result<(), EngError> {
    let packing = &cached.packing;
    // Charge full source, bounded acquisition/conversion, and descriptor storage.
    let metadata = packing
        .matrices
        .len()
        .checked_mul(std::mem::size_of::<PackedMatrix>() + 2048)
        .and_then(|bytes| {
            bytes.checked_add(
                cached
                    .layers
                    .len()
                    .checked_mul(std::mem::size_of::<LayerCache>() + 1024)?,
            )
        })
        .and_then(|bytes| bytes.checked_add(config_bytes))
        .ok_or_else(overflow)?;
    let required = u64::try_from(cached.layout.host_bytes())
        .map_err(|_| overflow())?
        .checked_add(packing.staging.checked_mul(2).ok_or_else(overflow)?)
        .and_then(|bytes| bytes.checked_add(metadata as u64))
        .ok_or_else(overflow)?;
    HostBudget::probe()?.admit(required)?;
    plan::admit(
        &plan::Demand {
            tensors: packing.bytes as u64,
            kv_cache: limits.reserve.kv_cache,
            work: limits.reserve.work,
            headroom: limits.reserve.headroom,
        },
        u64::try_from(context.memory_info()?.free).map_err(|_| overflow())?,
    )?;
    Ok(())
}

fn enqueue(
    context: &CudaContext,
    stream: &DeviceStream,
    pinned: &mut PinnedBuf,
    buffer: &DeviceBuffer,
    arena_base: usize,
    copies: &[MissCopy],
) -> Result<(), EngError> {
    for copy in copies {
        let destination = arena_base
            .checked_add(copy.destination.offset)
            .ok_or_else(overflow)?;
        if copy.source.bytes != copy.destination.bytes {
            return Err(refused("copy sizes disagree"));
        }
        context.copy_from_pinned(
            stream,
            pinned,
            copy.source.offset,
            buffer.range(destination, copy.destination.bytes)?,
        )?;
    }
    Ok(())
}

pub(super) fn load_cached(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    checkpoint: Checkpoint,
    geometry: Geometry,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    let mut audit_limits = limits;
    audit_limits.max_weight_bytes = u64::MAX;
    let roles = plan_resident(path, &checkpoint, &geometry, audit_limits)?;
    let cached = Packing::cached(&roles, &limits)?;
    admit_cached(context, &cached, limits, checkpoint.model_config.len())?;
    let packing = cached.packing;
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(roles.layout.modules.len())
        .map_err(|error| EngError::MemoryPlan(format!("Flint matrix descriptors: {error}")))?;
    descriptors.resize(roles.layout.modules.len(), None);
    let mut pinned = context.pinned_buf(cached.layout.host_bytes())?;
    let buffer = context.allocate(packing.bytes)?;
    for (index, module) in roles.layout.modules.iter().enumerate() {
        if cached.routed[index].is_some() {
            continue;
        }
        let specs = module.specs();
        let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        descriptors[index] = Some(fill_matrix(
            &checkpoint,
            &module.prefix,
            &keys,
            packing.matrices[index],
            limits.max_staging_bytes,
            |slot, bytes| buffer.write_at(stream, slot.offset, bytes),
        )?);
    }
    let store = HostStore::build(cached.layout, |layer, expert, range| {
        let FfnRoles::Moe { experts, .. } = &roles.layers[cached.layers[layer]].ffn else {
            return Err(refused("routed layer changed during loading"));
        };
        for &index in &experts[usize::from(expert)] {
            let module = &roles.layout.modules[index];
            let specs = module.specs();
            let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
            descriptors[index] = Some(fill_matrix(
                &checkpoint,
                &module.prefix,
                &keys,
                packing.matrices[index],
                limits.max_staging_bytes,
                |slot, bytes| {
                    let end = slot.offset.checked_add(bytes.len()).ok_or_else(overflow)?;
                    if end > range.bytes {
                        return Err(refused("source component exceeds expert stride"));
                    }
                    pinned.write(
                        range.offset.checked_add(slot.offset).ok_or_else(overflow)?,
                        bytes,
                    )
                },
            )?);
        }
        // pinned_buf zeroed every padding byte before any component was written.
        Ok(range.bytes)
    })?;
    for (plain, &slot) in roles.layout.plain.iter().zip(&packing.plain) {
        super::load::upload_plain(
            &checkpoint,
            &buffer,
            stream,
            plain,
            slot,
            limits.max_staging_bytes,
        )?;
    }
    let mut caches = Vec::new();
    caches
        .try_reserve_exact(cached.layers.len())
        .map_err(|error| EngError::MemoryPlan(format!("Flint layer caches: {error}")))?;
    for layer in 0..cached.layers.len() {
        let mut core = LayerCache::new(&store, layer, stream.id())?;
        core.publish_initial(stream.id(), |copies| {
            enqueue(
                context,
                stream,
                &mut pinned,
                &buffer,
                cached.arena_base,
                copies,
            )?;
            pinned.synchronize(stream)
        })?;
        caches.push(core);
    }
    stream.synchronize()?;
    let matrices = descriptors
        .into_iter()
        .map(|descriptor| descriptor.ok_or_else(|| refused("incomplete logical matrix coverage")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FlintExl3Resident {
        buffer,
        host: Some(HostOwner {
            state: RefCell::new(HostState {
                pinned,
                store,
                caches,
                stream: stream.id(),
            }),
            poisoned: Cell::new(false),
            layers: cached.layers,
            arena_base: cached.arena_base,
        }),
        matrices,
        plain: packing.plain,
        roles,
        model_config: checkpoint.model_config,
    })
}

fn fill_matrix(
    checkpoint: &Checkpoint,
    prefix: &str,
    keys: &[&str],
    slots: [PackedRange; 3],
    max_staging: u64,
    mut write: impl FnMut(PackedRange, &[u8]) -> Result<(), EngError>,
) -> Result<PackedMatrix, EngError> {
    // The audited plan supplies names; Matrix::load preserves each logical matrix's metadata.
    let selected = checkpoint.source.read_selected(keys, max_staging)?;
    let matrix = Matrix::load(&selected, prefix, &checkpoint.config)?;
    if matrix.bias().is_some() {
        return Err(refused("routed matrix bias is unsupported"));
    }
    for (part, (tensor, slot)) in [matrix.trellis(), matrix.suh(), matrix.svh()]
        .into_iter()
        .zip(slots)
        .enumerate()
    {
        let raw = tensor
            .host_bytes()
            .ok_or_else(|| refused("matrix requires host storage"))?;
        if part == 0 {
            if raw.len() != slot.bytes {
                return Err(refused("trellis disagrees with placement"));
            }
            write(slot, raw)?;
        } else {
            let converted = super::load::convert_halves(raw, tensor.dtype())?;
            if converted.len() != slot.bytes {
                return Err(refused("scale disagrees with placement"));
            }
            write(slot, &converted)?;
        }
    }
    Ok(PackedMatrix {
        trellis: slots[0],
        suh: slots[1],
        svh: slots[2],
        bias: None,
        shape: matrix.shape(),
        bits_twice: matrix.bitrate_twice(),
        book: matrix.codebook(),
    })
}

impl FlintExl3Resident {
    pub(crate) fn load_with_prefill(
        context: &CudaContext,
        stream: &DeviceStream,
        path: &Path,
        limits: FlintResidentLimits,
        rows: usize,
    ) -> Result<Self, EngError> {
        super::load::load_with_prefill(context, stream, path, limits, rows)
    }

    #[cfg(test)]
    pub(crate) fn test_host_cache_byte(&self) -> Result<u8, EngError> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| refused("model has no pinned source"))?;
        let state = host
            .state
            .try_borrow()
            .map_err(|_| refused("model cache is already borrowed"))?;
        let mut byte = [0];
        state.pinned.read(0, &mut byte)?;
        Ok(byte[0])
    }

    #[cfg(test)]
    pub(crate) fn load_cached_fixture(
        context: &CudaContext,
        stream: &DeviceStream,
        path: &Path,
        limits: FlintResidentLimits,
    ) -> Result<Self, EngError> {
        super::load::load_cached_fixture(context, stream, path, limits)
    }

    pub(crate) fn recover_expert_cache(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<(), EngError> {
        let Some(host) = &self.host else {
            return Ok(());
        };
        if !host.poisoned.get() {
            return Ok(());
        }
        context.ensure_usable()?;
        let mut state = host
            .state
            .try_borrow_mut()
            .map_err(|_| refused("model cache is already borrowed"))?;
        if state.stream != stream.id() {
            return Err(refused("recovery stream differs from model stream"));
        }
        stream.synchronize()?;
        state.pinned.synchronize(stream)?;
        let HostState { pinned, caches, .. } = &mut *state;
        for core in caches {
            core.poison();
            core.reset_after_drain(stream.id(), || {
                stream.synchronize()?;
                pinned.synchronize(stream)
            })?;
            core.publish_initial(stream.id(), |copies| {
                enqueue(
                    context,
                    stream,
                    pinned,
                    &self.buffer,
                    host.arena_base,
                    copies,
                )?;
                pinned.synchronize(stream)
            })?;
        }
        stream.synchronize()?;
        host.poisoned.set(false);
        Ok(())
    }

    pub(super) fn prepared_expert(
        &self,
        indices: [usize; 3],
        base: usize,
    ) -> Result<FlintExpertView<'_>, EngError> {
        let matrix = |index: usize| -> Result<Exl3MatrixView<'_>, EngError> {
            let logical = self
                .matrices
                .get(index)
                .ok_or_else(|| refused("unknown matrix"))?;
            let range = |part: PackedRange| {
                self.range(PackedRange {
                    offset: base.checked_add(part.offset).ok_or_else(overflow)?,
                    bytes: part.bytes,
                })
            };
            Ok(Exl3MatrixView {
                trellis: range(logical.trellis)?,
                suh: range(logical.suh)?,
                svh: range(logical.svh)?,
                bias: None,
                shape: logical.shape,
                bits_twice: logical.bits_twice,
                book: logical.book,
            })
        };
        Ok(FlintExpertView {
            gate: matrix(indices[0])?,
            up: matrix(indices[1])?,
            down: matrix(indices[2])?,
        })
    }
}

impl FlintExpertBank<'_> {
    /// The callback must finish all routed, shared, and combine work with Frame::finish.
    /// Only a completed owned output may escape this scope, never live weight views.
    pub(crate) fn with_prepared(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        ids: &[i32],
        consume: impl for<'a> FnOnce(&FlintExpertBank<'a>) -> Result<DeviceBuffer, EngError>,
    ) -> Result<DeviceBuffer, EngError> {
        let Some(host) = &self.owner.host else {
            return consume(&FlintExpertBank {
                owner: self.owner,
                indices: self.indices,
                layer: self.layer,
                prepared: None,
            });
        };
        if host.poisoned.replace(true) {
            return Err(refused("model cache is poisoned"));
        }
        context.ensure_usable()?;
        let mut state = host
            .state
            .try_borrow_mut()
            .map_err(|_| refused("model cache is already borrowed"))?;
        if state.stream != stream.id() {
            return Err(refused("consumer stream differs from model stream"));
        }
        let ordinal = host
            .layers
            .iter()
            .position(|&layer| layer == self.layer)
            .ok_or_else(|| refused("unknown routed layer"))?;
        let HostState {
            pinned,
            store,
            caches,
            ..
        } = &mut *state;
        let core = &mut caches[ordinal];
        let max_copy_bytes = store
            .layout()
            .layer(ordinal)?
            .slot(0)?
            .bytes
            .checked_mul(super::super::cache::SLOTS)
            .ok_or_else(overflow)?;
        let pass = core.prepare(ids, max_copy_bytes)?;
        let ready = core.copy_ready(pass, stream.id(), |copies| {
            enqueue(
                context,
                stream,
                pinned,
                &self.owner.buffer,
                host.arena_base,
                copies,
            )
        })?;
        let mut snapshot = [None; EXPERTS];
        for &id in ids {
            let expert = u16::try_from(id).map_err(|_| refused("invalid expert ID"))?;
            let slot = core.slot(ready, expert)?;
            let relative = store.layout().layer(ordinal)?.slot(slot)?.offset;
            snapshot[usize::from(expert)] =
                Some(host.arena_base.checked_add(relative).ok_or_else(overflow)?);
        }
        let output = core.release(ready, stream.id(), || {
            consume(&FlintExpertBank {
                owner: self.owner,
                indices: self.indices,
                layer: self.layer,
                prepared: Some(&snapshot),
            })
        })?;
        host.poisoned.set(false);
        Ok(output)
    }
}
