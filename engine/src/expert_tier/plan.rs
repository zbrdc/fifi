//! Placement: which MoE layers keep their routed experts on the device, fitted to the free VRAM.

use super::{ExpertId, residency::Residency};
use crate::EngError;

/// VRAM the tier leaves free after its resident experts and the family's dense weight cache.
pub const VRAM_HEADROOM_BYTES: u64 = 3 << 30;

/// How many MoE layers keep their routed experts on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuLayerPlan {
    /// As many as the free VRAM allows after the dense cache, leaving [`VRAM_HEADROOM_BYTES`].
    Auto,
    /// Exactly this many (zero means every MoE layer runs on the CPU).
    Fixed(usize),
}

/// How many whole MoE layers fit: `free` VRAM minus what the dense weight cache will still take
/// and the headroom, divided by one layer's measured footprint, capped at `max_layers`.
pub fn auto_fit_gpu_layers(
    free_bytes: u64,
    reserved_bytes: u64,
    layer_bytes: u64,
    max_layers: usize,
) -> usize {
    let spare = free_bytes
        .saturating_sub(reserved_bytes)
        .saturating_sub(VRAM_HEADROOM_BYTES);
    if layer_bytes == 0 {
        return 0;
    }
    usize::try_from(spare / layer_bytes).map_or(max_layers, |fit| fit.min(max_layers))
}

/// One part of one routed expert as the checkpoint stores it: its quantized data, the group
/// scales that go with it (empty for formats without them) and one per-tensor scale.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertPayload {
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
    pub tensor_scale: f32,
    /// `[rows, columns]` of the matrix.
    pub shape: [u32; 2],
}

/// A read-only expert part borrowed from a payload or an immutable mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertPayloadView<'a> {
    pub data: &'a [u8],
    pub scales: &'a [u8],
    pub tensor_scale: f32,
    pub shape: [u32; 2],
}

impl ExpertPayload {
    pub fn as_view(&self) -> ExpertPayloadView<'_> {
        ExpertPayloadView {
            data: &self.data,
            scales: &self.scales,
            tensor_scale: self.tensor_scale,
            shape: self.shape,
        }
    }
}

/// Expert id to weight bytes, the family's side of the device upload.
pub trait ExpertPayloadSource {
    /// Parts per expert (two for up and down, three for gate, up and down).
    fn parts(&self) -> usize;

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError>;

    /// Called once a layer's experts are on the device: the family drops the host page cache of
    /// their bytes so the expert is in exactly one place.
    fn release_layer(&mut self, layer: u32) -> Result<(), EngError>;

    /// The slices remain valid for the callback only. Mapped sources can override
    /// this owned-payload default to avoid copying the checkpoint bytes.
    fn with_payload<R>(
        &mut self,
        id: ExpertId,
        part: usize,
        consume: impl FnOnce(ExpertPayloadView<'_>) -> Result<R, EngError>,
    ) -> Result<R, EngError>
    where
        Self: Sized,
    {
        let payload = self.payload(id, part)?;
        consume(payload.as_view())
    }

    /// An optional hint; it must not change demand order or expert placement.
    fn prefetch_experts(&mut self, _ids: &[ExpertId]) -> Result<(), EngError> {
        Ok(())
    }

    /// Release only these experts' cached pages, never their whole mixed layer.
    /// The default refuses rather than evicting unrelated CPU-resident experts.
    fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
        if ids.is_empty() {
            return Ok(());
        }
        Err(EngError::BadRequest(
            "expert-granular release is unsupported by this source".into(),
        ))
    }
}

/// The family's device-side store of resident experts.
pub trait GpuExpertStore {
    /// Keeps one expert part on the device for the whole run, with no host copy.
    fn upload(
        &mut self,
        id: ExpertId,
        part: usize,
        payload: &ExpertPayload,
    ) -> Result<(), EngError>;

    /// Free device memory as the driver reports it.
    fn free_bytes(&self) -> Result<u64, EngError>;

    /// Device bytes the family still needs for its dense weight cache and staging; resident
    /// experts must leave them free. [`crate::expert_tier::StreamBankPlan::reserve`]
    /// includes pending streaming banks here until their allocation consumes the reservation.
    fn reserved_bytes(&self) -> u64;

    /// The placement is final: close the per-token upload paths and hold the dense cache to the
    /// `free_bytes` that are left.
    fn finish_placement(&mut self, free_bytes: u64) -> Result<(), EngError>;
}

/// Uploads whole MoE layers once, last layer first (as llama.cpp's `--n-cpu-moe` keeps its GPU
/// layers), and returns which experts are now resident. `nominal_layer_bytes` is the stored size of
/// one layer's experts; later layers are budgeted from what the first one really cost, allocation
/// rounding included. The policy is whole-layer, the result is expert-granular.
pub fn place_gpu_layers<G: GpuExpertStore, S: ExpertPayloadSource>(
    plan: GpuLayerPlan,
    moe_layers: &[u32],
    experts_per_layer: u32,
    nominal_layer_bytes: u64,
    gpu: &mut G,
    source: &mut S,
) -> Result<Residency, EngError> {
    let wanted = match plan {
        GpuLayerPlan::Fixed(count) if count > moe_layers.len() => {
            return Err(EngError::BadRequest(format!(
                "{count} GPU-resident MoE layers requested, the model has {}",
                moe_layers.len()
            )));
        }
        GpuLayerPlan::Fixed(count) => count,
        GpuLayerPlan::Auto => moe_layers.len(),
    };
    let mut residency = Residency::default();
    let mut layer_bytes = nominal_layer_bytes;
    for &layer in moe_layers.iter().rev().take(wanted) {
        let before = gpu.free_bytes()?;
        if auto_fit_gpu_layers(before, gpu.reserved_bytes(), layer_bytes, 1) == 0 {
            match plan {
                GpuLayerPlan::Auto => break,
                GpuLayerPlan::Fixed(_) => {
                    return Err(EngError::MemoryPlan(format!(
                        "{wanted} GPU-resident MoE layers do not fit: {before} bytes free, {} \
                         reserved for the dense cache and staging, {VRAM_HEADROOM_BYTES} headroom, \
                         {layer_bytes} per layer",
                        gpu.reserved_bytes()
                    )));
                }
            }
        }
        for expert in 0..experts_per_layer {
            let id = ExpertId { layer, expert };
            for part in 0..source.parts() {
                let payload = source.payload(id, part)?;
                gpu.upload(id, part, &payload)?;
            }
            residency.mark_expert(layer, expert, experts_per_layer);
        }
        source.release_layer(layer)?;
        layer_bytes = before.saturating_sub(gpu.free_bytes()?).max(layer_bytes);
    }
    let free = gpu.free_bytes()?;
    gpu.finish_placement(free)?;
    Ok(residency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_tier::{Coverage, StreamBankPlan};

    const GIB: u64 = 1 << 30;
    const EXPERTS: u32 = 4;
    const PAYLOAD: u64 = 100;
    /// 4 experts x 2 parts x 100 bytes.
    const LAYER: u64 = 800;

    struct Gpu {
        free: u64,
        reserved: u64,
        uploads: Vec<(ExpertId, usize)>,
        finished: Option<u64>,
    }

    impl GpuExpertStore for Gpu {
        fn upload(&mut self, id: ExpertId, part: usize, p: &ExpertPayload) -> Result<(), EngError> {
            self.free = self.free.saturating_sub(p.data.len() as u64);
            self.uploads.push((id, part));
            Ok(())
        }

        fn free_bytes(&self) -> Result<u64, EngError> {
            Ok(self.free)
        }

        fn reserved_bytes(&self) -> u64 {
            self.reserved
        }

        fn finish_placement(&mut self, free_bytes: u64) -> Result<(), EngError> {
            self.finished = Some(free_bytes);
            Ok(())
        }
    }

    #[derive(Default)]
    struct Source {
        released: Vec<u32>,
        fail_layer: Option<u32>,
    }

    impl ExpertPayloadSource for Source {
        fn parts(&self) -> usize {
            2
        }

        fn payload(&mut self, id: ExpertId, _part: usize) -> Result<ExpertPayload, EngError> {
            if self.fail_layer == Some(id.layer) {
                return Err(EngError::Checkpoint("unreadable".to_owned()));
            }
            Ok(ExpertPayload {
                data: vec![0; PAYLOAD as usize],
                scales: Vec::new(),
                tensor_scale: 1.0,
                shape: [1, 1],
            })
        }

        fn release_layer(&mut self, layer: u32) -> Result<(), EngError> {
            self.released.push(layer);
            Ok(())
        }
    }

    fn gpu(free: u64, reserved: u64) -> Gpu {
        Gpu {
            free,
            reserved,
            uploads: Vec::new(),
            finished: None,
        }
    }

    const MOE: [u32; 5] = [1, 3, 5, 8, 10];

    #[test]
    fn stream_bank_reservation_reaches_placement_and_the_final_cache_budget() {
        let plan = StreamBankPlan::new(LAYER, 64, 16).unwrap();
        assert_eq!(plan.bank_bytes(), LAYER);
        assert_eq!(plan.device_bytes(), 2 * LAYER + 64);
        assert_eq!(plan.host_bytes(), 2 * LAYER + 16);
        let mut device = gpu(VRAM_HEADROOM_BYTES + 5 * LAYER, 0);
        let mut reserved = plan.reserve(&mut device, 4096).unwrap();
        assert_eq!(reserved.reserved_bytes(), plan.device_bytes());
        let residency = place_gpu_layers(
            GpuLayerPlan::Auto,
            &MOE,
            EXPERTS,
            LAYER,
            &mut reserved,
            &mut Source::default(),
        )
        .unwrap();
        assert_eq!(residency.full_layers(), vec![8, 10]);
        drop(reserved);
        assert_eq!(device.finished, Some(device.free - plan.device_bytes()));
    }

    #[test]
    fn stream_bank_admission_refuses_overflow_and_short_budgets_before_upload() {
        assert!(matches!(
            StreamBankPlan::new(0, 0, 0),
            Err(EngError::BadRequest(_))
        ));
        for (bank, device_aux, scratch) in [(u64::MAX, 0, 0), (1, u64::MAX, 0), (1, 0, u64::MAX)] {
            assert!(matches!(
                StreamBankPlan::new(bank, device_aux, scratch),
                Err(EngError::MemoryPlan(_))
            ));
        }
        let plan = StreamBankPlan::new(LAYER, 64, 16).unwrap();
        for (free, reserved_bytes, host_budget) in [
            (
                VRAM_HEADROOM_BYTES + plan.device_bytes(),
                1,
                plan.host_bytes(),
            ),
            (
                VRAM_HEADROOM_BYTES + plan.device_bytes(),
                0,
                plan.host_bytes() - 1,
            ),
        ] {
            let mut device = gpu(free, reserved_bytes);
            assert!(matches!(
                plan.reserve(&mut device, host_budget),
                Err(EngError::MemoryPlan(_))
            ));
            assert!(device.uploads.is_empty());
            assert!(device.finished.is_none());
        }
    }

    #[test]
    fn stream_bank_reservation_rechecks_actual_free_bytes_before_finishing() {
        let plan = StreamBankPlan::new(100, 40, 0).unwrap();
        let mut device = gpu(VRAM_HEADROOM_BYTES + plan.device_bytes() + 100, 0);
        let mut reserved = plan.reserve(&mut device, plan.host_bytes()).unwrap();
        reserved
            .upload(
                ExpertId {
                    layer: 7,
                    expert: 1,
                },
                0,
                &ExpertPayload {
                    data: vec![0; 200],
                    scales: Vec::new(),
                    tensor_scale: 1.0,
                    shape: [1, 1],
                },
            )
            .unwrap();
        let free = reserved.free_bytes().unwrap();
        assert!(matches!(
            reserved.finish_placement(free),
            Err(EngError::MemoryPlan(_))
        ));
        drop(reserved);
        assert!(device.finished.is_none());
    }

    #[test]
    fn legacy_sources_support_views_without_coarse_expert_release() {
        let mut source = Source::default();
        let id = ExpertId {
            layer: 7,
            expert: 1,
        };
        let viewed = source
            .with_payload(id, 0, |view| {
                Ok((
                    view.data.len(),
                    view.scales.len(),
                    view.tensor_scale,
                    view.shape,
                ))
            })
            .unwrap();
        assert_eq!(viewed, (PAYLOAD as usize, 0, 1.0, [1, 1]));
        source.prefetch_experts(&[id]).unwrap();
        assert!(matches!(
            source.release_experts(&[id]),
            Err(EngError::BadRequest(_))
        ));
        source.release_experts(&[]).unwrap();
        assert!(source.released.is_empty());
        let legacy: &mut dyn ExpertPayloadSource = &mut source;
        legacy.release_layer(9).unwrap();
        assert_eq!(source.released, vec![9]);
    }

    #[test]
    fn borrowed_source_hooks_keep_views_borrowed_and_releases_targeted() {
        use crate::expert_tier::ExpertPayloadView;

        struct Borrowed {
            bytes: [u8; 4],
            hinted: Vec<ExpertId>,
            released: Vec<ExpertId>,
        }
        impl ExpertPayloadSource for Borrowed {
            fn parts(&self) -> usize {
                1
            }
            fn payload(&mut self, _: ExpertId, _: usize) -> Result<ExpertPayload, EngError> {
                Err(EngError::Checkpoint("owned payload was requested".into()))
            }
            fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
                Err(EngError::BadRequest(
                    "whole-layer release was requested".into(),
                ))
            }
            fn with_payload<R>(
                &mut self,
                _: ExpertId,
                _: usize,
                consume: impl FnOnce(ExpertPayloadView<'_>) -> Result<R, EngError>,
            ) -> Result<R, EngError> {
                consume(ExpertPayloadView {
                    data: &self.bytes,
                    scales: &[],
                    tensor_scale: 1.0,
                    shape: [1, 4],
                })
            }
            fn prefetch_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
                self.hinted.extend_from_slice(ids);
                Ok(())
            }
            fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
                self.released.extend_from_slice(ids);
                Ok(())
            }
        }
        let mut source = Borrowed {
            bytes: [1, 2, 3, 4],
            hinted: Vec::new(),
            released: Vec::new(),
        };
        let ids = [
            ExpertId {
                layer: 7,
                expert: 1,
            },
            ExpertId {
                layer: 7,
                expert: 3,
            },
        ];
        let address = source.bytes.as_ptr();
        source.prefetch_experts(&ids).unwrap();
        let observed = source
            .with_payload(ids[0], 0, |view| Ok((view.data.as_ptr(), view.data.len())))
            .unwrap();
        assert_eq!(observed, (address, 4));
        source.release_experts(&ids).unwrap();
        assert_eq!(source.hinted, ids);
        assert_eq!(source.released, ids);
    }

    #[test]
    fn auto_places_the_last_layers_that_fit_and_releases_each_after_its_upload() {
        // Room for 3 layers and a half after the reserve and the headroom.
        let mut device = gpu(VRAM_HEADROOM_BYTES + 50 + 3 * LAYER + 400, 50);
        let mut source = Source::default();
        let residency = place_gpu_layers(
            GpuLayerPlan::Auto,
            &MOE,
            EXPERTS,
            LAYER,
            &mut device,
            &mut source,
        )
        .unwrap();
        assert_eq!(residency.full_layers(), vec![5, 8, 10]);
        assert_eq!(residency.resident_experts(), 3 * EXPERTS as usize);
        assert_eq!(
            source.released,
            vec![10, 8, 5],
            "last layer first, each once"
        );
        assert_eq!(device.uploads.len(), 3 * EXPERTS as usize * 2);
        assert_eq!(device.finished, Some(device.free));
        assert_eq!(residency.coverage(8, &[0, 3]), Coverage::Device);
        assert_eq!(residency.coverage(3, &[0]), Coverage::Host);
    }

    #[test]
    fn a_fixed_count_places_exactly_that_many_or_refuses() {
        let mut device = gpu(VRAM_HEADROOM_BYTES + 10 * LAYER, 0);
        let mut source = Source::default();
        let residency = place_gpu_layers(
            GpuLayerPlan::Fixed(2),
            &MOE,
            EXPERTS,
            LAYER,
            &mut device,
            &mut source,
        )
        .unwrap();
        assert_eq!(residency.full_layers(), vec![8, 10]);

        let mut small = gpu(VRAM_HEADROOM_BYTES + LAYER + 50, 0);
        let refused = place_gpu_layers(
            GpuLayerPlan::Fixed(2),
            &MOE,
            EXPERTS,
            LAYER,
            &mut small,
            &mut Source::default(),
        );
        assert!(
            matches!(refused, Err(EngError::MemoryPlan(_))),
            "{refused:?}"
        );
        assert!(
            small.finished.is_none(),
            "a refused placement does not finish"
        );

        let too_many = place_gpu_layers(
            GpuLayerPlan::Fixed(6),
            &MOE,
            EXPERTS,
            LAYER,
            &mut gpu(100 * GIB, 0),
            &mut Source::default(),
        );
        assert!(matches!(too_many, Err(EngError::BadRequest(_))));
    }

    #[test]
    fn zero_layers_is_valid_and_still_finishes_the_placement() {
        let mut device = gpu(5 * GIB, 0);
        let residency = place_gpu_layers(
            GpuLayerPlan::Fixed(0),
            &MOE,
            EXPERTS,
            LAYER,
            &mut device,
            &mut Source::default(),
        )
        .unwrap();
        assert!(residency.full_layers().is_empty());
        assert_eq!(device.finished, Some(5 * GIB));
    }

    #[test]
    fn a_source_failure_stops_the_placement_before_anything_is_released_for_that_layer() {
        let mut device = gpu(100 * GIB, 0);
        let mut source = Source {
            fail_layer: Some(8),
            ..Source::default()
        };
        let outcome = place_gpu_layers(
            GpuLayerPlan::Auto,
            &MOE,
            EXPERTS,
            LAYER,
            &mut device,
            &mut source,
        );
        assert!(matches!(outcome, Err(EngError::Checkpoint(_))));
        assert_eq!(source.released, vec![10]);
        assert!(device.finished.is_none());
    }

    #[test]
    fn auto_fit_leaves_the_cache_and_the_vram_headroom() {
        let layer = 1512 * (1 << 20);
        // 23 GiB free, a 12 GiB dense cache still to come, 3 GiB headroom: 8 GiB / 1512 MiB = 5.
        assert_eq!(auto_fit_gpu_layers(23 * GIB, 12 * GIB, layer, 40), 5);
        // The same free memory with an 8 GiB stored-dtype dense set fits eight.
        assert_eq!(auto_fit_gpu_layers(23 * GIB, 8 * GIB, layer, 40), 8);
        // Never more than the model has, never negative, and a zero-size layer fits nothing.
        assert_eq!(auto_fit_gpu_layers(1000 * GIB, 0, layer, 40), 40);
        assert_eq!(auto_fit_gpu_layers(GIB / 2, 0, layer, 40), 0);
        assert_eq!(auto_fit_gpu_layers(23 * GIB, 30 * GIB, layer, 40), 0);
        assert_eq!(auto_fit_gpu_layers(23 * GIB, 0, 0, 40), 0);
    }
}
