//! Overlap the upload of the next routed expert with projection on the current bank.
use std::collections::BTreeMap;

use super::{GgufCuda, bad, cache::LayerCache};
use crate::{
    EngError,
    expert_tier::{CudaBankLease, ExpertId, ExpertPayloadSource, Placement, StreamBankPlan},
    kernels::cuda::{DeviceRange, DeviceStream, resident::dense_gemv::DenseDtype},
};

pub(super) struct Matrix<'a> {
    pub(super) data: DeviceRange<'a>,
    pub(super) shape: [usize; 2],
    pub(super) scale: f32,
}

pub(in crate::family::inkling2) struct RoutedExperts<'a> {
    pub ids: &'a [ExpertId],
    pub dtypes: [DenseDtype; 2],
    pub input: &'a [f32],
}

pub(in crate::family::inkling2) struct ResidentRoute<'a> {
    pub layer: u32,
    pub ids: &'a [usize],
    pub input: &'a [f32],
    pub caps: [f32; 2],
}

pub(in crate::family::inkling2) struct PrefillRoute<'a> {
    pub ids: &'a [usize],
    pub input: &'a [f32],
}

#[derive(Default)]
struct StreamTransfers {
    hits: usize,
    uploads: usize,
    bytes: u64,
}

struct ExpertMatrices<'a> {
    stream: Option<&'a DeviceStream>,
    parts: [Matrix<'a>; 2],
}

impl<'a> ExpertMatrices<'a> {
    fn staged(lease: &'a CudaBankLease<'_>, id: ExpertId) -> Result<Self, EngError> {
        let part = |index| -> Result<Matrix<'a>, EngError> {
            let p = lease.part(id, index)?;
            Ok(Matrix {
                data: p.data,
                shape: p.shape.map(|d| d as usize),
                scale: p.tensor_scale,
            })
        };
        Ok(Self {
            stream: Some(lease.stream()),
            parts: [part(0)?, part(1)?],
        })
    }
}

pub(super) fn plan(hidden: usize, ff: usize) -> Result<StreamBankPlan, EngError> {
    // BF16 gate/up and down are the largest pair. Part starts are 256-byte aligned.
    let matrix = hidden
        .checked_mul(ff)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| bad("GGUF stream matrix size overflows"))? as u64;
    let bank = matrix
        .checked_mul(3)
        .and_then(|n| n.checked_add(255))
        .ok_or_else(|| bad("GGUF stream bank size overflows"))?;
    // The transport validates the largest borrowed part and metadata per slot.
    let scratch = matrix
        .checked_mul(4)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| bad("GGUF stream scratch size overflows"))?;
    StreamBankPlan::new(bank, 0, scratch)
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn streams_layer(&self, layer: u32) -> bool {
        self.stream_banks.is_some() && self.resident.placement().has_cold(layer)
    }

    pub(in crate::family::inkling2) fn stream_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        route: RoutedExperts<'_>,
        activate: impl Fn(f32, f32) -> f32,
        mut combine: impl FnMut(usize, &[f32]),
    ) -> Result<bool, EngError> {
        let Some(layer) = route.ids.first().map(|id| id.layer) else {
            return Ok(false);
        };
        if !self.can_stream(route.ids) {
            return Ok(false);
        }
        let mut caches = self.cache.borrow_mut();
        let mut cache = caches.get_mut(&layer);
        if let Some(c) = cache.as_deref_mut() {
            c.observe(route.ids)?;
        }
        let transfers =
            self.visit_experts(source, route.ids, cache.as_deref_mut(), |id, matrices| {
                let out = self.project_pair(&matrices, route.dtypes, route.input, &activate)?;
                combine(id.expert as usize, &out);
                Ok(())
            })?;
        if let Some(c) = cache {
            c.record(layer, route.ids.len(), transfers.hits);
        }
        self.record_transfer(&transfers, std::iter::repeat_n(1, route.ids.len()));
        Ok(true)
    }

    pub(in crate::family::inkling2) fn has_stream_banks(&self) -> bool {
        self.stream_banks.is_some()
    }

    fn can_stream(&self, ids: &[ExpertId]) -> bool {
        self.has_stream_banks()
            && ids
                .iter()
                .all(|&id| self.resident.placement().placement(id) == Placement::Nvme)
    }

    pub(in crate::family::inkling2) fn prefill_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        routes: &[PrefillRoute<'_>],
        dtypes: [DenseDtype; 2],
        activate: impl Fn(f32, f32) -> f32,
    ) -> Result<Option<ExpertRows>, EngError> {
        let groups = group_routes(layer, routes);
        let ids = groups.keys().copied().collect::<Vec<_>>();
        if !self.can_stream(&ids) {
            return Ok(None);
        }
        let mut cache = self.cache.borrow_mut();
        let mut cache = cache.get_mut(&layer);
        if let Some(c) = cache.as_deref_mut() {
            for route in routes {
                let ids = route
                    .ids
                    .iter()
                    .map(|&expert| ExpertId {
                        layer,
                        expert: expert as u32,
                    })
                    .collect::<Vec<_>>();
                c.observe(&ids)?;
            }
        }
        // Store each result at its routing rank so each token adds contributions
        // in the original order, irrespective of the order we stage experts.
        let mut rows: Vec<_> = routes
            .iter()
            .map(|route| vec![Vec::new(); route.ids.len()])
            .collect();
        let transfers = self.visit_experts(source, &ids, cache, |id, matrices| {
            for &(row, rank) in &groups[&id] {
                rows[row][rank] =
                    self.project_pair(&matrices, dtypes, routes[row].input, &activate)?;
            }
            Ok(())
        })?;
        self.record_transfer(&transfers, groups.values().map(Vec::len));
        Ok(Some(rows))
    }

    pub(in crate::family::inkling2) fn prefill_bank_report(
        &self,
    ) -> Option<crate::sched::job::PrefillBankReport> {
        self.prefill_report
            .as_ref()
            .map(|report| report.borrow().clone())
    }

    fn record_transfer(
        &self,
        transfers: &StreamTransfers,
        routed_rows: impl Iterator<Item = usize>,
    ) {
        let Some(report) = &self.prefill_report else {
            return;
        };
        let mut report = report.borrow_mut();
        report.passes += 1;
        report.misses += transfers.uploads as u64;
        report.upload_bytes += transfers.bytes;
        if transfers.uploads > 0 {
            note_bucket(&mut report.miss_histogram, transfers.uploads);
        }
        for rows in routed_rows {
            note_bucket(&mut report.route_histogram, rows);
        }
    }

    fn visit_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        ids: &[ExpertId],
        mut cache: Option<&mut LayerCache>,
        mut execute: impl FnMut(ExpertId, ExpertMatrices<'_>) -> Result<(), EngError>,
    ) -> Result<StreamTransfers, EngError> {
        let mut banks = self
            .stream_banks
            .as_ref()
            .ok_or_else(|| bad("no GGUF stream banks"))?
            .borrow_mut();
        let (first, next) = banks.slots_mut();
        let mut transfers = StreamTransfers::default();
        for pair in ids.chunks(2) {
            let hit = |id: &ExpertId| cache.as_ref().and_then(|c| c.hit(id.expert));
            let slots = [hit(&pair[0]), pair.get(1).and_then(hit)];
            let a = if slots[0].is_some() {
                None
            } else {
                first.stage_demand(&self.context, source, &pair[..1])?
            };
            let b = if slots[1].is_some() || pair.len() == 1 {
                None
            } else {
                next.stage_demand(&self.context, source, &pair[1..])?
            };
            for ((&id, lease), slot) in pair.iter().zip([a, b]).zip(slots) {
                if let Some(lease) = lease {
                    execute(id, ExpertMatrices::staged(&lease, id)?)?;
                    transfers.uploads += 1;
                    transfers.bytes += lease.bytes() as u64;
                    match cache.as_deref_mut() {
                        Some(c) => c.finish(&self.context, lease, id)?,
                        None => lease.finish()?,
                    }
                } else {
                    let (c, slot) = cache
                        .as_ref()
                        .zip(slot)
                        .ok_or_else(|| bad("missing GGUF expert bank"))?;
                    execute(
                        id,
                        ExpertMatrices {
                            stream: None,
                            parts: [c.part(slot, 0)?, c.part(slot, 1)?],
                        },
                    )?;
                    transfers.hits += 1;
                }
            }
        }
        Ok(transfers)
    }

    fn project_pair(
        &self,
        weights: &ExpertMatrices<'_>,
        dtypes: [DenseDtype; 2],
        input: &[f32],
        activate: &impl Fn(f32, f32) -> f32,
    ) -> Result<Vec<f32>, EngError> {
        let project = |part: usize, x: &[f32]| {
            let view = &weights.parts[part];
            self.step.project_gguf(
                &self.context,
                weights.stream,
                (dtypes[part], view.data, view.shape, view.scale),
                x,
            )
        };
        let gates = project(0, input)?;
        // The GGUF stores complete gate rows followed by complete up rows.
        let (gate, up) = gates.split_at(gates.len() / 2);
        let mid: Vec<_> = gate.iter().zip(up).map(|(&g, &u)| activate(g, u)).collect();
        project(1, &mid)
    }
}

fn note_bucket(histogram: &mut Vec<u64>, count: usize) {
    let bucket = count.next_power_of_two().ilog2().min(12) as usize;
    if histogram.len() <= bucket {
        histogram.resize(bucket + 1, 0);
    }
    histogram[bucket] += 1;
}

pub(in crate::family::inkling2) type ExpertRows = Vec<Vec<Vec<f32>>>;

fn group_routes(
    layer: u32,
    routes: &[PrefillRoute<'_>],
) -> BTreeMap<ExpertId, Vec<(usize, usize)>> {
    let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for (row, route) in routes.iter().enumerate() {
        for (rank, &expert) in route.ids.iter().enumerate() {
            groups
                .entry(ExpertId {
                    layer,
                    expert: expert as u32,
                })
                .or_default()
                .push((row, rank));
        }
    }
    groups
}

#[test]
fn grouped_routes_keep_the_rank_of_each_expert_for_each_row() {
    let a = PrefillRoute {
        ids: &[7, 4],
        input: &[],
    };
    let b = PrefillRoute {
        ids: &[5, 7],
        input: &[],
    };
    let groups = group_routes(3, &[a, b]);
    assert_eq!(groups.len(), 3);
    assert_eq!(
        groups[&ExpertId {
            layer: 3,
            expert: 7
        }],
        [(0, 0), (1, 1)]
    );
}
