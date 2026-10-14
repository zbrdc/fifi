//! `export_state` and `import_state`: the host value of the committed session state (decision
//! 0047 item 3, packet RD-9b1).
//!
//! [`ExportedState`] is a plain host value (`Send`) for the checkpoint store of decision 0036. It
//! carries the device element orders unchanged (see `state_layout.rs` for the orders and for the
//! transposes against llama.cpp and the Q36 host reference), the committed position count, and an
//! identity that includes the KV mode. `import_state` refuses a value whose identity differs and
//! checks every length before the first device write. Export reads slot 0 and the first `tokens` KV
//! positions only and never the words. `import_state` writes the same range from the snapshot and
//! does the rest of reset's work (the words block rewritten with zeros, the other slots cleared), so
//! it may leave a poisoned state; the host writes `StepParams` (`write_step_params`) before the
//! next step.
//!
//! Each copy is stream ordered (`DeviceBuffer::read_at` / `write_at`): it runs behind the work
//! queued on the given stream and waits for that stream only. The caller passes the stream that
//! runs the step, has joined any other stream into it, and has committed every enqueued step.

use super::{
    super::state_layout::{KvPool, StateLayout},
    F32, ResidentState, StateIdentity,
};
use crate::{EngError, kernels::cuda::DeviceStream};

/// One DeltaNet layer: recurrent `[value head][key row][value column]` and convolution
/// `[tap][channel]`, both f32.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeltaExport {
    pub(crate) recurrent: Vec<f32>,
    pub(crate) conv: Vec<f32>,
}

/// One attention layer: its planes in the order of `KvPool::planes`, each holding the first
/// `tokens` positions with no capacity padding. Fp16 values are little-endian halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KvExport {
    pub(crate) planes: Vec<Vec<u8>>,
}

/// A checkpoint of the committed session state, in layer order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExportedState {
    pub(crate) identity: StateIdentity,
    pub(crate) tokens: usize,
    pub(crate) delta: Vec<DeltaExport>,
    pub(crate) kv: Vec<KvExport>,
}

// The checkpoint store keeps exports on another thread than the device owner.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<ExportedState>();
};

/// What `import_state` passes to `StateBook::begin_import`: it does ALL of reset's work (words
/// rewritten, status zeroed, the other slots cleared), so a `Poisoned` state may be imported
/// (REPLY-76 Q1).
pub(crate) const IMPORT_REWRITES_WORDS: bool = true;

/// Refuses an export that does not fit `layout` and `identity`; reads no device memory.
///
/// **Contract (lane C, Xavier XS-1 and depth timings).** It validates STRUCTURE only: the layout
/// tag, the KV mode and the geometry (the identity), the DeltaNet layer count and vector lengths,
/// the attention layer count, the plane lengths for `tokens` positions, and `tokens <= capacity`.
/// It accepts ANY finite snapshot of that shape at any depth `N <= capacity` (the start block of
/// such a state is `pos = N`, `kv_len = N + 1`, written by `write_step_params`), and it never
/// compares values against a real run. It does not inspect values at all: this packet promises NO
/// finiteness check, so a snapshot holding NaN or infinity also passes (a non-finite state is
/// caught by the status word of the first step, which poisons). A depth past the capacity is
/// refused as `ContextLimit` before any plane length is looked at. A state imported at
/// `N = capacity` is accepted but cannot take a decode step; decoding at depth `D` needs
/// `capacity >= D + 1` (lane C).
pub(crate) fn check_import(
    layout: &StateLayout,
    identity: &StateIdentity,
    exported: &ExportedState,
) -> Result<(), EngError> {
    let bad = |what: &str| EngError::Checkpoint(format!("import_state: {what}"));
    if exported.identity != *identity {
        return Err(bad("the identity (layout, KV mode or geometry) differs"));
    }
    if exported.tokens > layout.kv.capacity {
        return Err(EngError::ContextLimit(format!(
            "import_state: {} positions exceed the KV capacity {}",
            exported.tokens, layout.kv.capacity
        )));
    }
    let pool = layout.delta;
    let delta_fits = exported.delta.len() == pool.layers
        && exported.delta.iter().all(|layer| {
            layer.recurrent.len() == pool.rec_values && layer.conv.len() == pool.conv_values
        });
    if !delta_fits {
        return Err(bad("a DeltaNet layer has the wrong length"));
    }
    if exported.kv.len() != layout.kv.layers {
        return Err(bad("the attention layer count differs"));
    }
    for (ordinal, layer) in exported.kv.iter().enumerate() {
        let lengths: Vec<usize> = layer.planes.iter().map(Vec::len).collect();
        if lengths != plane_sizes(&layout.kv, ordinal, exported.tokens)? {
            return Err(bad("a KV plane has the wrong length"));
        }
    }
    Ok(())
}

/// Bytes of each plane of one attention layer for `tokens` positions, in export order.
pub(crate) fn plane_sizes(
    kv: &KvPool,
    ordinal: usize,
    tokens: usize,
) -> Result<Vec<usize>, EngError> {
    let ranges = kv.prefix_ranges(ordinal, tokens)?;
    Ok(kv
        .planes()
        .iter()
        .map(|&plane| {
            ranges
                .iter()
                .filter(|range| range.plane == plane)
                .map(|range| range.bytes)
                .sum()
        })
        .collect())
}

/// A zeroed host buffer of `len` bytes. `try_reserve_exact` returns an error (a capacity
/// overflow or an allocation failure) where `vec![0u8; len]` and `Vec::resize` abort the
/// process, so a huge or denied export becomes a `MemoryPlan` refusal -- which a prefix save
/// turns into `Ok(false)` (decision 0036 addendum 1 item 7) instead of killing the engine.
/// `f32_from_le` and `f32_to_le` stay as they are: they allocate at most one layer's vectors
/// (about 2 MB per layer) from bytes the caller already holds.
pub(crate) fn host_zeroed(len: usize) -> Result<Vec<u8>, EngError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|error| {
        EngError::MemoryPlan(format!("export_state: {len} host bytes: {error}"))
    })?;
    bytes.resize(len, 0);
    Ok(bytes)
}

pub(crate) fn f32_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<F32>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

pub(crate) fn f32_to_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

impl ResidentState {
    /// Copies the committed state (slot 0 and the first `tokens` KV positions) to the host.
    ///
    /// Every copy runs behind the work queued on `stream`, so no launch of the step on that
    /// stream is still writing the pools.
    pub(crate) fn export_state(&self, stream: &DeviceStream) -> Result<ExportedState, EngError> {
        self.book.begin_export(stream.id())?;
        let pool = self.layout.delta;
        let delta = (0..pool.layers)
            .map(|ordinal| {
                Ok(DeltaExport {
                    recurrent: self.read_f32(
                        stream,
                        pool.rec_offset(0, ordinal)?,
                        pool.rec_values,
                    )?,
                    conv: self.read_f32(stream, pool.conv_offset(0, ordinal)?, pool.conv_values)?,
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let kv = (0..self.layout.kv.layers)
            .map(|ordinal| self.export_kv(stream, ordinal))
            .collect::<Result<Vec<_>, EngError>>()?;
        Ok(ExportedState {
            identity: self.identity.clone(),
            tokens: self.book.tokens(),
            delta,
            kv,
        })
    }
    /// Replaces the committed state with `exported`, doing ALL of reset's work: the words block
    /// is rewritten with zeros (so `StepParams` and `status` are zeroed and a start block is owed:
    /// `write_step_params` must follow), the DeltaNet slots above 0 are cleared, slot 0 and the
    /// first `tokens` KV positions come from the snapshot. That is what lets it leave a `Poisoned`
    /// state (lane M REPLY-76 Q1); the book would refuse a poisoned state otherwise.
    ///
    /// Every check runs before the first device write, so a refused value changes nothing. A
    /// device failure after `check_import` leaves the state `Poisoned` (never a partly imported
    /// `Valid` state) until the next reset or successful import.
    pub(crate) fn import_state(
        &mut self,
        stream: &DeviceStream,
        exported: &ExportedState,
    ) -> Result<(), EngError> {
        check_import(&self.layout, &self.identity, exported)?;
        self.book.begin_import(stream.id(), IMPORT_REWRITES_WORDS)?;
        self.clear_words(stream)?;
        self.clear_delta(stream, 1)?;
        let pool = self.layout.delta;
        for (ordinal, layer) in exported.delta.iter().enumerate() {
            self.write_f32(stream, pool.rec_offset(0, ordinal)?, &layer.recurrent)?;
            self.write_f32(stream, pool.conv_offset(0, ordinal)?, &layer.conv)?;
        }
        for (ordinal, layer) in exported.kv.iter().enumerate() {
            self.import_kv(stream, ordinal, exported.tokens, layer)?;
        }
        self.book.finish_overwrite(exported.tokens);
        Ok(())
    }
    fn export_kv(&self, stream: &DeviceStream, ordinal: usize) -> Result<KvExport, EngError> {
        let ranges = self.layout.kv.prefix_ranges(ordinal, self.book.tokens())?;
        let mut planes = Vec::new();
        for &plane in self.layout.kv.planes() {
            // One fallible allocation per plane: the runs fill it at their offsets.
            let total: usize = ranges
                .iter()
                .filter(|range| range.plane == plane)
                .map(|range| range.bytes)
                .sum();
            let mut bytes = host_zeroed(total)?;
            let mut at = 0;
            for range in ranges.iter().filter(|range| range.plane == plane) {
                if range.bytes > 0 {
                    self.kv.read_at(
                        stream,
                        range.device_offset,
                        &mut bytes[at..at + range.bytes],
                    )?;
                }
                at += range.bytes;
            }
            planes.push(bytes);
        }
        Ok(KvExport { planes })
    }
    fn import_kv(
        &self,
        stream: &DeviceStream,
        ordinal: usize,
        tokens: usize,
        layer: &KvExport,
    ) -> Result<(), EngError> {
        let ranges = self.layout.kv.prefix_ranges(ordinal, tokens)?;
        for (&plane, bytes) in self.layout.kv.planes().iter().zip(&layer.planes) {
            let mut at = 0;
            for range in ranges.iter().filter(|range| range.plane == plane) {
                if range.bytes > 0 {
                    let part = &bytes[at..at + range.bytes];
                    self.kv.write_at(stream, range.device_offset, part)?;
                }
                at += range.bytes;
            }
        }
        Ok(())
    }
    fn read_f32(
        &self,
        stream: &DeviceStream,
        offset: usize,
        values: usize,
    ) -> Result<Vec<f32>, EngError> {
        let byte_len = values.checked_mul(F32).ok_or_else(|| {
            EngError::MemoryPlan(format!("export_state: {values} f32 values overflow usize"))
        })?;
        let mut bytes = host_zeroed(byte_len)?;
        self.delta.read_at(stream, offset, &mut bytes)?;
        Ok(f32_from_le(&bytes))
    }
    fn write_f32(
        &self,
        stream: &DeviceStream,
        offset: usize,
        values: &[f32],
    ) -> Result<(), EngError> {
        self.delta.write_at(stream, offset, &f32_to_le(values))
    }
}
