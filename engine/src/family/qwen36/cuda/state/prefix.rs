//! Engine prefix payload: resident raw compatibility or charged DeltaNet plus aliased pages.
use super::super::{
    paging_store::{HostCharge, PageSnapshot, PagedKv},
    state_layout::DeltaPool,
};
use super::{
    KvStorage, ResidentState, StateIdentity,
    export::{ExportedState, IMPORT_REWRITES_WORDS},
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream},
};
use std::{
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

#[cfg(all(test, feature = "cuda-fault-qualification"))]
mod fault_tests;

#[derive(Clone)]
pub(crate) enum PrefixState {
    Resident(Arc<ExportedState>),
    Paged(Arc<PagedPrefix>),
}

struct DeltaBytes {
    recurrent: Vec<u8>,
    conv: Vec<u8>,
}

pub(crate) struct PagedPrefix {
    identity: StateIdentity,
    tokens: usize,
    delta: Vec<DeltaBytes>,
    pages: PageSnapshot,
    _charge: HostCharge,
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<PrefixState>();
};

fn memory(what: impl std::fmt::Display) -> EngError {
    EngError::MemoryPlan(format!("state prefix: {what}"))
}

fn allocation_bytes(pool: DeltaPool) -> Result<usize, EngError> {
    let values = pool
        .rec_values
        .checked_add(pool.conv_values)
        .ok_or_else(|| memory("values overflow"))?;
    let state = values
        .checked_mul(4)
        .and_then(|n| n.checked_mul(pool.layers))
        .ok_or_else(|| memory("state bytes overflow"))?;
    let meta = pool
        .layers
        .checked_mul(size_of::<DeltaBytes>())
        .and_then(|n| n.checked_add(size_of::<PagedPrefix>()))
        .and_then(|n| n.checked_add(2 * size_of::<AtomicUsize>()))
        .ok_or_else(|| memory("metadata overflows"))?;
    state
        .checked_add(meta)
        .ok_or_else(|| memory("snapshot size overflows"))
}

struct Reader<'a> {
    context: &'a CudaContext,
    stream: &'a DeviceStream,
    device: &'a DeviceBuffer,
    store: &'a mut PagedKv,
    charge: &'a mut HostCharge,
}

impl Reader<'_> {
    fn read(&mut self, offset: usize, len: usize) -> Result<Vec<u8>, EngError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(memory)?;
        self.charge.grow(bytes.capacity() - len)?;
        bytes.resize(len, 0);
        if len != 0 {
            self.store.read_state(
                self.context,
                self.stream,
                self.device.range(offset, len)?,
                &mut bytes,
            )?;
        }
        Ok(bytes)
    }
}

impl ResidentState {
    pub(crate) fn export_prefix(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<PrefixState, EngError> {
        context.ensure_uncaptured("state prefix export")?;
        context.check_launch_stream(stream)?;
        self.book.begin_export(stream.id())?;
        let KvStorage::Paged { store, .. } = &self.kv else {
            return self
                .export_state(stream)
                .map(|saved| PrefixState::Resident(Arc::new(saved)));
        };
        let mut store = store
            .try_borrow_mut()
            .map_err(|_| EngError::BadRequest("paged state is already borrowed".to_owned()))?;
        let pages = store.snapshot(context, stream, self.tokens())?;
        let pool = self.layout.delta;
        let mut charge = pages.charge_host(allocation_bytes(pool)?)?;
        let mut delta = Vec::new();
        delta.try_reserve_exact(pool.layers).map_err(memory)?;
        charge.grow((delta.capacity() - pool.layers) * size_of::<DeltaBytes>())?;
        {
            let mut reader = Reader {
                context,
                stream,
                device: &self.delta,
                store: &mut store,
                charge: &mut charge,
            };
            for layer in 0..pool.layers {
                delta.push(DeltaBytes {
                    recurrent: reader.read(pool.rec_offset(0, layer)?, pool.rec_values * 4)?,
                    conv: reader.read(pool.conv_offset(0, layer)?, pool.conv_values * 4)?,
                });
            }
        }
        Ok(PrefixState::Paged(Arc::new(PagedPrefix {
            identity: self.identity.clone(),
            tokens: self.tokens(),
            delta,
            pages,
            _charge: charge,
        })))
    }

    pub(crate) fn import_prefix(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        prefix: &PrefixState,
    ) -> Result<(), EngError> {
        context.ensure_uncaptured("state prefix import")?;
        context.check_launch_stream(stream)?;
        let saved = match prefix {
            PrefixState::Resident(saved) => return self.import_state(stream, saved),
            PrefixState::Paged(saved) => saved,
        };
        let pool = self.layout.delta;
        if saved.identity != self.identity
            || saved.tokens > self.layout.kv.capacity
            || saved.pages.tokens() != saved.tokens
            || saved.delta.len() != pool.layers
            || saved.delta.iter().any(|layer| {
                layer.recurrent.len() != pool.rec_values * 4
                    || layer.conv.len() != pool.conv_values * 4
            })
        {
            return Err(EngError::Checkpoint(
                "state prefix identity, depth or DeltaNet lengths differ".to_owned(),
            ));
        }
        let KvStorage::Paged { store, .. } = &self.kv else {
            return Err(EngError::Checkpoint(
                "paged prefix needs paged storage".to_owned(),
            ));
        };
        let mut store = store
            .try_borrow_mut()
            .map_err(|_| EngError::BadRequest("paged state is already borrowed".to_owned()))?;
        store.check_restore(stream, &saved.pages)?;
        self.book.begin_import(stream.id(), IMPORT_REWRITES_WORDS)?;
        store.invalidate(stream)?;
        self.clear_words(stream)?;
        self.clear_delta(stream, 1)?;
        for (layer, data) in saved.delta.iter().enumerate() {
            for (offset, bytes) in [
                (pool.rec_offset(0, layer)?, data.recurrent.as_slice()),
                (pool.conv_offset(0, layer)?, data.conv.as_slice()),
            ] {
                if !bytes.is_empty() {
                    store.write_state(
                        context,
                        stream,
                        bytes,
                        self.delta.range(offset, bytes.len())?,
                    )?;
                }
            }
        }
        store.restore(stream, &saved.pages)?;
        self.book.finish_overwrite(saved.tokens);
        Ok(())
    }
}
