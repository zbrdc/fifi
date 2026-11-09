//! The prefill expert ring: streams every expert a big chunk needs from the pinned store onto
//! the device behind the compute stream, one copy-stream entry per expert in layer-major
//! order, so later layers' experts arrive while earlier layers compute. This is the overlap
//! Strata measured at ~1.2k tok/s prefill on a 12 GB card: PCIe becomes background traffic.
//!
//! Ordering discipline (the same shape as the bank's twice-reviewed copy tail): a ring slot's
//! next H2D copy on the copy stream waits the `used` event the compute stream recorded after
//! the slot's last read, and the compute stream waits the slot's `copied` event before its
//! first read. The bank is frozen read-only for the whole chunk, so the host places the plan
//! reads are stable.

use super::bank::NO_PLACE;

/// An expert the chunk needs that the bank does not hold: streamed into a ring slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StreamEntry {
    pub(super) layer: u16,
    pub(super) expert: u16,
    /// The pinned-store place the record reads from, stable while the bank is frozen.
    pub(super) place: u32,
    /// The ring slot the record lands in (`index % ring`).
    pub(super) slot: u16,
}

/// Every entry in stream order: layer-major, expert id ascending, skipping experts the bank
/// already holds (they read from bank slots on the compute stream, no copy). Inclusive store
/// keeps a place for every expert, Exclusive only for the non-banked ones, which is exactly
/// what the ring needs either way.
pub(super) fn plan_from_rows(
    slot_rows: &[&[i32]],
    place_rows: &[&[u32]],
    experts: usize,
    ring: usize,
) -> Vec<StreamEntry> {
    let mut entries = Vec::new();
    for (layer, (slots, places)) in slot_rows.iter().zip(place_rows).enumerate() {
        for expert in 0..experts {
            let (slot, place) = (
                slots.get(expert).copied().unwrap_or(-1),
                places.get(expert).copied().unwrap_or(NO_PLACE),
            );
            if slot >= 0 || place == NO_PLACE {
                continue;
            }
            entries.push(StreamEntry {
                layer: layer as u16,
                expert: expert as u16,
                place,
                slot: (entries.len() % ring.max(1)) as u16,
            });
        }
    }
    entries
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) mod device {
    use super::{NO_PLACE, StreamEntry, plan_from_rows};
    use crate::{
        EngError,
        family::flash_next::cuda::bank::{BankGeometry, device::DeviceBank},
        kernels::cuda::{CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream},
    };

    /// The ring's device side: `ring` record slots laid out like one bank layer, plus one copy
    /// stream whose H2D copies run behind the compute stream via per-slot events.
    pub(in crate::family::flash_next) struct PrefillRing {
        slots: DeviceBuffer,
        copy: DeviceStream,
        copied: Vec<DeviceEvent>,
        used: Vec<DeviceEvent>,
        live: Vec<bool>,
        entries: Vec<StreamEntry>,
        /// Boundary per layer: entries[layer_start[l]..layer_start[l + 1]] are layer l's.
        layer_start: Vec<usize>,
        /// The first layer whose `free_layer` has not run yet (the consumed point).
        freed: usize,
        issued: usize,
        record: usize,
        ring: usize,
    }

    impl PrefillRing {
        /// Plans the stream from the frozen bank (every non-resident expert, layer-major) and
        /// allocates the ring. Strata measured 384 slots as the optimum on a 24 GB card,
        /// ~1.2 GiB at flash-next's 3.23 MiB records.
        pub(in crate::family::flash_next) fn build(
            ctx: &CudaContext,
            bank: &DeviceBank,
            ring: usize,
        ) -> Result<Self, EngError> {
            if ring == 0 {
                return Err(EngError::BadRequest(
                    "the prefill ring needs at least one slot".into(),
                ));
            }
            let geometry: BankGeometry = *bank.geometry();
            let layers = geometry.layers as usize;
            let experts = geometry.experts as usize;
            let slot_rows: Vec<&[i32]> = (0..layers)
                .map(|l| bank.slot_row(l))
                .collect::<Result<Vec<_>, _>>()?;
            let place_rows: Vec<Vec<u32>> = (0..layers)
                .map(|l| {
                    (0..experts)
                        .map(|e| {
                            bank.state()
                                .places()
                                .place_of(l, e as u16)
                                .unwrap_or(NO_PLACE)
                        })
                        .collect()
                })
                .collect();
            let place_refs: Vec<&[u32]> = place_rows.iter().map(|v| v.as_slice()).collect();
            let entries = plan_from_rows(&slot_rows, &place_refs, experts, ring);
            let mut layer_start = vec![entries.len(); layers + 1];
            for (i, entry) in entries.iter().enumerate() {
                let l = usize::from(entry.layer);
                if layer_start[l] == entries.len() {
                    layer_start[l] = i;
                }
            }
            // A layer's grouped pass reads every streamed expert of the layer at once, so the
            // ring must hold one full layer while the next streams in behind it.
            let per_layer = layer_start
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap_or(0);
            if ring < 2 * per_layer {
                return Err(EngError::BadRequest(format!(
                    "the prefill ring needs {}/{} slots to hold a layer and the next, has {ring}",
                    per_layer,
                    2 * per_layer
                )));
            }
            Ok(Self {
                slots: ctx.allocate(ring * geometry.record_bytes)?,
                copy: ctx.stream()?,
                copied: (0..ring)
                    .map(|_| ctx.event())
                    .collect::<Result<Vec<_>, _>>()?,
                used: (0..ring)
                    .map(|_| ctx.event())
                    .collect::<Result<Vec<_>, _>>()?,
                live: vec![false; ring],
                entries,
                layer_start,
                freed: 0,
                issued: 0,
                record: geometry.record_bytes,
                ring,
            })
        }

        /// Enqueues H2D copies until `ahead` entries past the consumed point (the driver calls
        /// this between layers, typically with the whole next layer in flight). The horizon
        /// never crosses a wrap onto a slot whose tenant is not freed yet: the copy would wait
        /// an unrecorded `used` event and the bytes it overwrites could still be read.
        pub(in crate::family::flash_next) fn issue_until(
            &mut self,
            ctx: &CudaContext,
            bank: &mut DeviceBank,
            ahead: usize,
        ) -> Result<(), EngError> {
            let start = self.layer_start[self.freed];
            let mut target = (start + ahead).min(self.entries.len());
            let mut held = vec![false; self.ring];
            for entry in &self.entries[start..self.issued] {
                held[usize::from(entry.slot)] = true;
            }
            for (i, entry) in self.entries[self.issued..target].iter().enumerate() {
                let slot = usize::from(entry.slot);
                if held[slot] {
                    target = self.issued + i;
                    break;
                }
                held[slot] = true;
            }
            while self.issued < target {
                let entry = self.entries[self.issued];
                let slot = usize::from(entry.slot);
                if self.live[slot] {
                    self.copy.wait(&self.used[slot])?;
                }
                bank.stream_place(
                    ctx,
                    &self.copy,
                    usize::from(entry.layer),
                    entry.place,
                    self.range(slot)?,
                )?;
                self.copy.record(&self.copied[slot])?;
                self.live[slot] = true;
                self.issued += 1;
            }
            Ok(())
        }

        /// Makes `stream` wait every copy of `layer`'s entries, so the grouped launches that
        /// follow read settled bytes.
        pub(in crate::family::flash_next) fn wait_layer(
            &self,
            stream: &DeviceStream,
            layer: usize,
        ) -> Result<(), EngError> {
            for entry in self.layer_entries(layer) {
                stream.wait(&self.copied[usize::from(entry.slot)])?;
            }
            Ok(())
        }

        /// Records `used` on `stream` for every slot of `layer`, letting later copies overwrite
        /// them once this layer's reads finish. Call after the layer's grouped launches are on
        /// the compute stream. Layers must be freed in order; the freed point is what
        /// [`Self::issue_until`] issues past.
        pub(in crate::family::flash_next) fn free_layer(
            &mut self,
            stream: &DeviceStream,
            layer: usize,
        ) -> Result<(), EngError> {
            if layer != self.freed {
                return Err(EngError::BadRequest(format!(
                    "the prefill ring frees layers in order, layer {layer} after {}",
                    self.freed
                )));
            }
            for entry in self.layer_entries(layer) {
                stream.record(&self.used[usize::from(entry.slot)])?;
            }
            self.freed += 1;
            Ok(())
        }

        /// The ring slot an expert of a layer landed in, for the grouped pass's slot ids.
        pub(in crate::family::flash_next) fn slot_of(
            &self,
            layer: usize,
            expert: u16,
        ) -> Option<u16> {
            self.layer_entries(layer)
                .iter()
                .find(|e| e.expert == expert)
                .map(|e| e.slot)
        }

        /// The ring buffer, laid out like one bank layer for the grouped launch.
        pub(in crate::family::flash_next) fn slots(&self) -> &DeviceBuffer {
            &self.slots
        }

        fn layer_entries(&self, layer: usize) -> &[StreamEntry] {
            self.entries
                .get(
                    self.layer_start.get(layer).copied().unwrap_or(0)
                        ..self.layer_start.get(layer + 1).copied().unwrap_or(0),
                )
                .unwrap_or(&[])
        }

        fn range(&self, slot: usize) -> Result<DeviceRange<'_>, EngError> {
            self.slots.range(slot * self.record, self.record)
        }
    }
}
