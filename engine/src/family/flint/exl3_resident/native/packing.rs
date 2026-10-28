use super::PackedRange;
use crate::{
    EngError,
    family::flint::exl3_resident::FlintResidentLimits,
    kernels::cuda::{ARENA_ALIGN, CHUNK_LIMIT},
    weights::exl3::flint_pack::Plan,
};

pub(super) struct Packing {
    pub(super) matrices: Vec<[PackedRange; 3]>,
    pub(super) plain: Vec<PackedRange>,
    pub(super) bytes: usize,
    pub(super) staging: u64,
}

pub(super) struct CachedPacking {
    pub(super) packing: Packing,
    pub(super) layout: super::super::cache::HostLayout,
    pub(super) arena_base: usize,
    pub(super) routed: Vec<Option<(usize, u16)>>,
    pub(super) layers: Vec<usize>,
}

fn audit_layer_stride(
    packing: &mut Packing,
    routed: &mut [Option<(usize, u16)>],
    experts: &[[usize; 3]],
    ordinal: usize,
) -> Result<usize, EngError> {
    if experts.len() != super::super::cache::EXPERTS {
        return Err(EngError::Checkpoint(
            "Flint host cache requires all 256 experts".into(),
        ));
    }
    let mut stride = None;
    for (id, projections) in experts.iter().enumerate() {
        let mut cursor = 0;
        for &index in projections {
            if routed[index].replace((ordinal, id as u16)).is_some() {
                return Err(EngError::Checkpoint(
                    "Flint duplicate routed matrix role".into(),
                ));
            }
            for slot in &mut packing.matrices[index] {
                *slot = place(&mut cursor, slot.bytes as u64)?;
            }
        }
        let padded = cursor
            .checked_next_multiple_of(ARENA_ALIGN)
            .ok_or_else(overflow)?;
        if stride.is_some_and(|expected| expected != padded) {
            return Err(EngError::Checkpoint(
                "Flint routed experts have incompatible padded strides".into(),
            ));
        }
        stride = Some(padded);
    }
    stride.ok_or_else(overflow)
}

impl Packing {
    pub(super) fn cached(
        roles: &super::ResidentPlan,
        limits: &FlintResidentLimits,
    ) -> Result<CachedPacking, EngError> {
        use super::super::{FfnRoles, cache::HostLayout};
        let mut audit_limits = *limits;
        audit_limits.max_weight_bytes = u64::MAX;
        let mut packing = Self::new(&roles.layout, &audit_limits)?;
        let mut routed = vec![None; packing.matrices.len()];
        let mut layers = Vec::new();
        let mut strides = Vec::new();
        for (layer, role) in roles.layers.iter().enumerate() {
            let FfnRoles::Moe { experts, .. } = &role.ffn else {
                continue;
            };
            let stride = audit_layer_stride(&mut packing, &mut routed, experts, layers.len())?;
            layers.push(layer);
            strides.push(stride);
        }
        let layout = HostLayout::new(&strides)?;
        let mut cursor = 0;
        for (index, slots) in packing.matrices.iter_mut().enumerate() {
            if routed[index].is_none() {
                for slot in slots {
                    *slot = place(&mut cursor, slot.bytes as u64)?;
                }
            }
        }
        for slot in &mut packing.plain {
            *slot = place(&mut cursor, slot.bytes as u64)?;
        }
        let arena_base = cursor
            .checked_next_multiple_of(ARENA_ALIGN)
            .ok_or_else(overflow)?;
        packing.bytes = arena_base
            .checked_add(layout.slot_bytes())
            .ok_or_else(overflow)?;
        if packing.bytes as u64 > limits.max_weight_bytes {
            return Err(EngError::MemoryPlan(format!(
                "Flint reduced weights require {} bytes; budget {}",
                packing.bytes, limits.max_weight_bytes
            )));
        }
        Ok(CachedPacking {
            packing,
            layout,
            arena_base,
            routed,
            layers,
        })
    }

    pub(super) fn new(layout: &Plan, limits: &FlintResidentLimits) -> Result<Self, EngError> {
        let mut result = Self {
            matrices: Vec::new(),
            plain: Vec::new(),
            bytes: 0,
            staging: 0,
        };
        result
            .matrices
            .try_reserve_exact(layout.modules.len())
            .map_err(allocation)?;
        result
            .plain
            .try_reserve_exact(layout.plain.len())
            .map_err(allocation)?;
        for module in &layout.modules {
            let specs = module.specs();
            let raw = specs.iter().try_fold(0u64, |sum, spec| {
                sum.checked_add(spec.bytes()).ok_or_else(overflow)
            })?;
            let trellis = specs[0].bytes();
            let suh = specs[1].bytes().checked_mul(2).ok_or_else(overflow)?;
            let svh = specs[2].bytes().checked_mul(2).ok_or_else(overflow)?;
            let copying = trellis.max(converted_copy(suh)?).max(converted_copy(svh)?);
            result.staging = result
                .staging
                .max(raw.checked_add(copying).ok_or_else(overflow)?);
            result.matrices.push([
                place(&mut result.bytes, trellis)?,
                place(&mut result.bytes, suh)?,
                place(&mut result.bytes, svh)?,
            ]);
        }
        for plain in &layout.plain {
            let raw = plain
                .shape
                .iter()
                .try_fold(plain.dtype.byte_width(), |bytes, &d| {
                    bytes.checked_mul(u64::from(d)).ok_or_else(overflow)
                })?;
            let live_raw = if plain.bulk {
                raw / u64::from(plain.shape[0])
            } else {
                raw
            };
            let converted = live_raw.checked_mul(2).ok_or_else(overflow)?;
            result.staging = result.staging.max(
                live_raw
                    .checked_add(converted_copy(converted)?)
                    .ok_or_else(overflow)?,
            );
            result.plain.push(place(
                &mut result.bytes,
                raw.checked_mul(2).ok_or_else(overflow)?,
            )?);
        }
        result.bytes = result
            .bytes
            .checked_next_multiple_of(ARENA_ALIGN)
            .ok_or_else(overflow)?;
        if result.bytes as u64 > limits.max_weight_bytes
            || result.staging > limits.max_staging_bytes
        {
            return Err(EngError::MemoryPlan(format!(
                "Flint EXL3 packed weights require {} device bytes and {} staging bytes; budgets {} and {}",
                result.bytes, result.staging, limits.max_weight_bytes, limits.max_staging_bytes
            )));
        }
        Ok(result)
    }
}

fn place(cursor: &mut usize, bytes: u64) -> Result<PackedRange, EngError> {
    let bytes = usize::try_from(bytes).map_err(|_| overflow())?;
    if bytes == 0 {
        return Err(overflow());
    }
    let offset = cursor
        .checked_next_multiple_of(ARENA_ALIGN)
        .ok_or_else(overflow)?;
    *cursor = offset.checked_add(bytes).ok_or_else(overflow)?;
    Ok(PackedRange { offset, bytes })
}

fn copy_bytes(bytes: u64) -> u64 {
    bytes.min(CHUNK_LIMIT as u64)
}

fn converted_copy(bytes: u64) -> Result<u64, EngError> {
    bytes.checked_add(copy_bytes(bytes)).ok_or_else(overflow)
}

fn overflow() -> EngError {
    EngError::MemoryPlan("Flint EXL3 resident byte count is zero or overflows".into())
}

fn allocation(error: std::collections::TryReserveError) -> EngError {
    EngError::MemoryPlan(format!("Flint EXL3 resident descriptors: {error}"))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::{
        family::project::cuda::Reserve,
        weights::exl3::flint_pack::{Geometry, PackBuilder},
    };

    #[test]
    fn published_pack_reports_expected_resident_admission_bytes() {
        let config = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/flint-exl3/config-4.0-e7ab1124.json"
        ));
        let geometry = Geometry::published(config).unwrap();
        let layout = geometry.plan();
        let mut limits = FlintResidentLimits {
            max_staging_bytes: u64::MAX,
            max_weight_bytes: u64::MAX,
            reserve: Reserve::default(),
        };
        let packing = Packing::new(&layout, &limits).unwrap();
        let payload: usize = packing
            .matrices
            .iter()
            .flatten()
            .chain(&packing.plain)
            .map(|range| range.bytes)
            .sum();
        assert!(packing.bytes >= payload && payload > 0);
        assert_eq!(packing.bytes % ARENA_ALIGN, 0);
        let head_specs = layout.modules[0].specs();
        let head_raw: u64 = head_specs.iter().map(|spec| spec.bytes()).sum();
        let acquisition_peak = head_raw + head_specs[0].bytes();
        // read_selected retains scales while Tensor copies the head's Vec into an Arc.
        assert!(packing.staging >= acquisition_peak);
        limits.max_weight_bytes = packing.bytes as u64;
        limits.max_staging_bytes = packing.staging;
        assert!(Packing::new(&layout, &limits).is_ok());
        // Keep the receipt visible without changing the existing test runner's capture flags.
        writeln!(
            std::io::stdout().lock(),
            "EXPECTED_CPU_PLAN device_weight_bytes={} staging_bytes={} arena_align={} modules={} plain={}; NOT measured VRAM peak",
            packing.bytes, packing.staging, ARENA_ALIGN, layout.modules.len(), layout.plain.len()
        ).unwrap();
    }

    #[test]
    fn packed_ranges_are_all_live_aligned_and_both_caps_include_copy_staging() {
        let pack = PackBuilder::tiny();
        let geometry = Geometry::from_config(&serde_json::to_vec(&pack.config).unwrap()).unwrap();
        let layout = geometry.plan();
        let mut limits = FlintResidentLimits {
            max_staging_bytes: u64::MAX,
            max_weight_bytes: u64::MAX,
            reserve: Reserve::default(),
        };
        let packing = Packing::new(&layout, &limits).unwrap();
        let mut end = 0;
        for range in packing.matrices.iter().flatten().chain(&packing.plain) {
            assert_eq!(range.offset % ARENA_ALIGN, 0);
            assert!(range.offset >= end && range.bytes > 0);
            end = range.offset + range.bytes;
        }
        assert!(end <= packing.bytes);
        let head_raw: u64 = layout.modules[0]
            .specs()
            .iter()
            .map(|spec| spec.bytes())
            .sum();
        assert!(packing.staging >= head_raw + copy_bytes(layout.modules[0].specs()[0].bytes()));
        limits.max_staging_bytes = packing.staging;
        limits.max_weight_bytes = packing.bytes as u64;
        assert!(Packing::new(&layout, &limits).is_ok());
        limits.max_weight_bytes -= 1;
        assert!(matches!(
            Packing::new(&layout, &limits),
            Err(EngError::MemoryPlan(_))
        ));
        limits.max_weight_bytes += 1;
        limits.max_staging_bytes -= 1;
        assert!(matches!(
            Packing::new(&layout, &limits),
            Err(EngError::MemoryPlan(_))
        ));
        let mut cursor = usize::MAX;
        assert!(place(&mut cursor, 1).is_err());
    }

    fn published_roles() -> super::super::ResidentPlan {
        use super::super::super::{FfnRoles, LayerRoles, ResidentPlan};
        let config = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/flint-exl3/config-4.0-e7ab1124.json"
        ));
        let geometry = Geometry::published(config).unwrap();
        let layout = geometry.plan();
        let names: std::collections::HashMap<_, _> = layout
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| (module.prefix.as_str(), index))
            .collect();
        let layers = (0..geometry.layers)
            .map(|layer| {
                let expert = |prefix: String| {
                    ["gate", "up", "down"]
                        .map(|part| names[format!("{prefix}.{part}_proj").as_str()])
                };
                let prefix = format!("model.layers.{layer}.mlp");
                let ffn = if geometry.dense_layers.contains(&layer) {
                    FfnRoles::Dense(expert(prefix))
                } else {
                    FfnRoles::Moe {
                        router: 0,
                        experts: (0..256)
                            .map(|id| expert(format!("{prefix}.experts.{id}")))
                            .collect(),
                        shared: expert(format!("{prefix}.shared_expert")),
                        routing_weight_multiplier: 128.0,
                    }
                };
                LayerRoles {
                    qkvo: [0; 4],
                    norms: [0; 4],
                    sinks: 0,
                    ffn,
                }
            })
            .collect();
        ResidentPlan {
            layout,
            layers,
            embedding: 0,
            final_norm: 0,
            head: 0,
            stored_payload_bytes: 0,
            converted_payload_bytes: 0,
        }
    }

    #[test]
    fn cached_placement_reduces_real_allocation_and_enforces_exact_cap() {
        let roles = published_roles();
        let mut limits = FlintResidentLimits {
            max_weight_bytes: u64::MAX,
            max_staging_bytes: u64::MAX,
            reserve: Reserve::default(),
        };
        let off = Packing::new(&roles.layout, &limits).unwrap();
        let cached = Packing::cached(&roles, &limits).unwrap();
        assert_eq!(
            off.bytes - cached.packing.bytes,
            cached.layout.host_bytes() - cached.layout.slot_bytes()
        );
        assert_eq!(
            cached.packing.bytes,
            cached.arena_base + cached.layout.slot_bytes()
        );
        for (index, routed) in cached.routed.iter().enumerate() {
            for range in cached.packing.matrices[index] {
                if let Some((layer, _)) = routed {
                    assert!(
                        range.offset + range.bytes
                            <= cached
                                .layout
                                .layer(*layer)
                                .unwrap()
                                .source(255)
                                .unwrap()
                                .bytes
                    );
                } else {
                    assert!(range.offset + range.bytes <= cached.arena_base);
                }
            }
        }
        limits.max_weight_bytes = cached.packing.bytes as u64;
        assert!(Packing::cached(&roles, &limits).is_ok());
        limits.max_weight_bytes -= 1;
        assert!(matches!(
            Packing::cached(&roles, &limits),
            Err(EngError::MemoryPlan(_))
        ));
        limits.max_weight_bytes = u64::MAX;
        let off_again = Packing::new(&roles.layout, &limits).unwrap();
        assert_eq!(off.bytes, off_again.bytes);
        assert_eq!(off.staging, off_again.staging);
        for (before, after) in off
            .matrices
            .iter()
            .flatten()
            .zip(off_again.matrices.iter().flatten())
        {
            assert_eq!((before.offset, before.bytes), (after.offset, after.bytes));
        }
    }

    #[test]
    fn cached_placement_audits_nonzero_experts_in_every_projection() {
        for projection in 0..3 {
            let mut roles = published_roles();
            let super::super::super::FfnRoles::Moe { experts, .. } = &roles.layers[2].ffn else {
                panic!("routed layer required")
            };
            let index = experts[193][projection];
            roles.layout.modules[index].words += 16;
            let limits = FlintResidentLimits {
                max_weight_bytes: u64::MAX,
                max_staging_bytes: u64::MAX,
                reserve: Reserve::default(),
            };
            assert!(matches!(
                Packing::cached(&roles, &limits),
                Err(EngError::Checkpoint(_))
            ));
        }
    }
}
