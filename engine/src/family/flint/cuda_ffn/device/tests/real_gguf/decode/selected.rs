//! Selected-only diagnostic consumer of the existing GPU fetch transport.

use super::*;
use crate::expert_tier::{CudaStreamBanks, ExpertId, ExpertPayload, ExpertPayloadSource};
use crate::family::flint::cuda_ffn::device::streaming::{self, GgufSource, MoeCall};
use std::{io::Write, time::Instant};

const PREPARED_HOST_BUDGET: u64 = 28 << 30;

fn prepared_flag(value: Result<String, std::env::VarError>) -> Result<bool, EngError> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        _ => Err(bad("FIFI_FLINT_PREPARED_EXPERTS must be absent, 0 or 1")),
    }
}

fn prepared_ids() -> Vec<ExpertId> {
    // There are 47 routed layers: the upper bound 49 is exclusive.
    (2..49)
        .flat_map(|layer| (0..256).map(move |expert| ExpertId { layer, expert }))
        .collect()
}

fn prepare_marker(progress: &mut impl Write, record: serde_json::Value) -> Result<(), EngError> {
    let line = format!("SELECTED_PREPARE {record}\n");
    progress
        .write_all(line.as_bytes())
        .and_then(|()| progress.flush())
        .map_err(|error| EngError::Checkpoint(format!("Flint prepare progress: {error}")))
}

pub(super) fn first_step_marker(progress: &mut impl Write) -> Result<(), EngError> {
    progress
        .write_all(b"SELECTED_STEP_BEGIN {\"position\":0}\n")
        .and_then(|()| progress.flush())
        .map_err(|error| EngError::Checkpoint(format!("Flint first STEP progress: {error}")))
}

fn prepare_experts(
    source: (&File, &GgufHeader),
    ids: &[ExpertId],
    host_budget: u64,
    progress: &mut impl Write,
) -> Result<streaming::PreparedSource, EngError> {
    prepare_marker(
        progress,
        serde_json::json!({
            "event": "begin", "prepared_enabled": true,
            "experts": ids.len(), "host_budget_bytes": host_budget,
        }),
    )?;
    let start = Instant::now();
    let prepared = streaming::PreparedSource::new(source.0, source.1, ids, host_budget)?;
    let prepare_ns = start.elapsed().as_nanos();
    prepare_marker(
        progress,
        serde_json::json!({
            "event": "end", "prepared_enabled": true,
            "experts": ids.len(), "host_budget_bytes": host_budget,
            "retained_bytes": prepared.retained_host_bytes(), "prepare_ns": prepare_ns,
        }),
    )?;
    Ok(prepared)
}

#[derive(Default)]
pub(in crate::family::flint::cuda_ffn) struct FetchStats {
    pub(in crate::family::flint::cuda_ffn) resident_experts: usize,
    pub(in crate::family::flint::cuda_ffn) cold_experts: Vec<ExpertId>,
    pub(in crate::family::flint::cuda_ffn) payload_bytes: usize,
    pub(in crate::family::flint::cuda_ffn) kv_device_bytes: usize,
    pub(in crate::family::flint::cuda_ffn) retained_device_bytes: usize,
    pub(in crate::family::flint::cuda_ffn) bank_device_bytes: u64,
    pub(in crate::family::flint::cuda_ffn) workspace_device_reservation: u64,
    pub(in crate::family::flint::cuda_ffn) host_stream_reservation: u64,
    pub(in crate::family::flint::cuda_ffn) device_free_before: usize,
    pub(in crate::family::flint::cuda_ffn) device_free_after: usize,
}

pub(super) enum SelectedSource<'a> {
    Gguf(GgufSource<'a>),
    Prepared(streaming::PreparedSource),
}

impl<'a> SelectedSource<'a> {
    fn new(
        source: (&'a File, &'a GgufHeader),
        enabled: bool,
        progress: &mut impl Write,
    ) -> Result<Self, EngError> {
        if enabled {
            return Ok(Self::Prepared(prepare_experts(
                source,
                &prepared_ids(),
                PREPARED_HOST_BUDGET,
                progress,
            )?));
        }
        let source = GgufSource::new(source.0, source.1)?;
        prepare_marker(
            progress,
            serde_json::json!({
                "event": "disabled", "prepared_enabled": false,
                "experts": 0, "retained_bytes": 0, "prepare_ns": 0,
            }),
        )?;
        Ok(Self::Gguf(source))
    }

    pub(super) fn prepared_enabled(&self) -> bool {
        matches!(self, Self::Prepared(_))
    }
}

impl ExpertPayloadSource for SelectedSource<'_> {
    fn parts(&self) -> usize {
        match self {
            Self::Gguf(source) => source.parts(),
            Self::Prepared(source) => source.parts(),
        }
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        match self {
            Self::Gguf(source) => source.payload(id, part),
            Self::Prepared(source) => source.payload(id, part),
        }
    }

    fn with_payload<R>(
        &mut self,
        id: ExpertId,
        part: usize,
        consume: impl FnOnce(crate::expert_tier::ExpertPayloadView<'_>) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        match self {
            Self::Gguf(source) => source.with_payload(id, part, consume),
            Self::Prepared(source) => source.with_payload(id, part, consume),
        }
    }

    fn release_layer(&mut self, layer: u32) -> Result<(), EngError> {
        match self {
            Self::Gguf(source) => source.release_layer(layer),
            Self::Prepared(source) => source.release_layer(layer),
        }
    }

    fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
        match self {
            Self::Gguf(source) => source.release_experts(ids),
            Self::Prepared(source) => source.release_experts(ids),
        }
    }
}

pub(super) struct Selected<'a> {
    pub(super) layers: Vec<weights::Layer>,
    pub(super) source: SelectedSource<'a>,
    pub(super) banks: CudaStreamBanks,
    pub(super) stats: FetchStats,
}

pub(super) fn validate_request(
    hot_n: usize,
    mmq: Result<String, std::env::VarError>,
) -> Result<(), EngError> {
    match mmq {
        Err(std::env::VarError::NotPresent) => {}
        Ok(value) if value == "0" => {}
        Ok(value) if value == "1" => {
            return Err(bad("selected decode does not wire FLINT_CUDA_FFN_MMQ=1"));
        }
        _ => {
            return Err(bad(
                "FLINT_CUDA_FFN_MMQ must be absent or 0 for selected decode",
            ));
        }
    }
    // shortcut: cold pilot only; admit fixed hot sets after reviewing its measured headroom.
    if hot_n != 0 {
        return Err(bad(
            "selected cold pilot requires hot_n=0; hot placement is not admitted",
        ));
    }
    Ok(())
}

impl<'a> Selected<'a> {
    pub(super) fn new(
        ctx: &CudaContext,
        source: (&'a File, &'a GgufHeader),
        kv: &crate::family::flint::kv_plan::KvPlan,
        kv_device_bytes: usize,
    ) -> Result<Self, EngError> {
        // This source lives with the decoder across all steps; preparation is not a STEP phase.
        let selected_source = SelectedSource::new(
            source,
            prepared_flag(std::env::var("FIFI_FLINT_PREPARED_EXPERTS"))?,
            &mut std::io::stderr().lock(),
        )?;
        let free_before = ctx.memory_info()?.free;
        let plan = streaming::plan(1)?;
        let bank_device_bytes = 2 * plan.bank_bytes();
        let workspace_device_reservation = plan.device_bytes() - bank_device_bytes;
        let host_stream_reservation = plan.host_bytes();
        let banks = CudaStreamBanks::new(ctx, plan)?;
        let layers = (0..49)
            .map(|index| {
                weights::Layer::load_selected(ctx, source, index as u32, kv.cache_rows(index)?)
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let resident_experts = layers.iter().map(weights::Layer::resident_experts).sum();
        let retained_device_bytes = layers.iter().map(weights::Layer::owned_device_bytes).sum();
        let free_after = ctx.memory_info()?.free;
        Ok(Self {
            layers,
            source: selected_source,
            banks,
            stats: FetchStats {
                resident_experts,
                cold_experts: Vec::new(),
                payload_bytes: 0,
                kv_device_bytes,
                retained_device_bytes,
                bank_device_bytes,
                workspace_device_reservation,
                host_stream_reservation,
                device_free_before: free_before,
                device_free_after: free_after,
            },
        })
    }
}

pub(super) struct Fetch<'a, 'model> {
    pub(super) layer: u32,
    pub(super) source: &'a mut SelectedSource<'model>,
    pub(super) banks: &'a mut CudaStreamBanks,
    pub(super) stats: &'a mut FetchStats,
}

impl Fetch<'_, '_> {
    pub(super) fn forward(
        self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        layer: &weights::Layer,
        input: &DeviceBuffer,
    ) -> Result<DeviceBuffer, EngError> {
        let weights::Ffn::Moe {
            router,
            experts,
            shared,
        } = &layer.ffn
        else {
            return layer.ffn(ctx, stream, input);
        };
        let hot = experts
            .iter()
            .enumerate()
            .map(|(id, expert)| Ok((id as u32, expert.view(ctx)?)))
            .collect::<Result<Vec<_>, EngError>>()?;
        let mut source = Traced {
            source: self.source,
            stats: self.stats,
        };
        streaming::forward(
            ctx,
            stream,
            MoeCall {
                layer: self.layer,
                rows: row_shape(input, WIDTH)?.rows,
                input: input.range(0, input.bytes())?,
                router: router.range(0, router.bytes())?,
                shared: shared.view(ctx)?,
                hot: &hot,
                source: &mut source,
                banks: self.banks,
            },
        )
    }
}

struct Traced<'a, S> {
    source: &'a mut S,
    stats: &'a mut FetchStats,
}

impl<S: ExpertPayloadSource> ExpertPayloadSource for Traced<'_, S> {
    fn parts(&self) -> usize {
        self.source.parts()
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        let _fetch = super::super::super::profile::wall("fetch");
        let payload = self.source.payload(id, part)?;
        if part == 0 {
            self.stats.cold_experts.push(id);
        }
        self.stats.payload_bytes += payload.data.len() + payload.scales.len();
        Ok(payload)
    }

    fn with_payload<R>(
        &mut self,
        id: ExpertId,
        part: usize,
        consume: impl FnOnce(crate::expert_tier::ExpertPayloadView<'_>) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        let fetch = super::super::super::profile::wall("fetch");
        self.source.with_payload(id, part, |view| {
            if part == 0 {
                self.stats.cold_experts.push(id);
            }
            self.stats.payload_bytes += view.data.len() + view.scales.len();
            drop(fetch);
            consume(view)
        })
    }

    fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
        self.source.release_experts(ids)
    }

    fn release_layer(&mut self, layer: u32) -> Result<(), EngError> {
        self.source.release_layer(layer)
    }
}

#[cfg(test)]
#[path = "selected_prepared_tests.rs"]
mod prepared_tests;

#[test]
fn traced_borrows_once_and_propagates_source_and_consumer_errors() {
    struct Borrowed;
    impl ExpertPayloadSource for Borrowed {
        fn parts(&self) -> usize {
            2
        }
        fn payload(&mut self, _: ExpertId, _: usize) -> Result<ExpertPayload, EngError> {
            panic!("borrowed source must not request an owned copy")
        }
        fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
            Ok(())
        }
        fn with_payload<R>(
            &mut self,
            _: ExpertId,
            part: usize,
            consume: impl FnOnce(crate::expert_tier::ExpertPayloadView<'_>) -> Result<R, EngError>,
        ) -> Result<R, EngError> {
            if part >= 2 {
                return Err(bad("source part"));
            }
            consume(crate::expert_tier::ExpertPayloadView {
                data: &[1, 2],
                scales: &[3],
                tensor_scale: 1.0,
                shape: [1, 2],
            })
        }
    }
    let id = ExpertId {
        layer: 2,
        expert: 7,
    };
    let mut source = Borrowed;
    let mut stats = FetchStats::default();
    let mut traced = Traced {
        source: &mut source,
        stats: &mut stats,
    };
    let capture = super::super::super::profile::Capture::start();
    traced
        .with_payload(id, 0, |view| {
            assert_eq!(view.data, [1, 2]);
            assert!(capture.finish().wall_ns.contains_key("fetch"));
            Ok(())
        })
        .unwrap();
    assert_eq!(traced.stats.payload_bytes, 3);
    assert_eq!(traced.stats.cold_experts, [id]);
    traced.with_payload(id, 1, |_| Ok(())).unwrap();
    assert_eq!(traced.stats.payload_bytes, 6);
    assert_eq!(traced.stats.cold_experts, [id]);
    let failed: Result<(), EngError> =
        traced.with_payload(id, 2, |_| panic!("invalid part reached consumer"));
    assert!(
        matches!(failed, Err(EngError::BadRequest(message)) if message.contains("source part"))
    );
    assert_eq!(traced.stats.payload_bytes, 6);
    let failed: Result<(), EngError> = traced.with_payload(id, 0, |_| Err(bad("consumer")));
    assert!(matches!(failed, Err(EngError::BadRequest(message)) if message.contains("consumer")));
    assert_eq!(traced.stats.payload_bytes, 9);
    assert_eq!(traced.stats.cold_experts, [id, id]);
}

#[test]
fn selected_request_never_ignores_mmq_or_unadmitted_hot_placement() {
    assert!(validate_request(0, Err(std::env::VarError::NotPresent)).is_ok());
    assert!(validate_request(0, Ok("0".into())).is_ok());
    assert!(matches!(validate_request(0, Ok("1".into())),
        Err(EngError::BadRequest(message)) if message.contains("does not wire FLINT_CUDA_FFN_MMQ=1")));
    for value in ["", "2", "false"] {
        assert!(matches!(validate_request(0, Ok(value.into())),
            Err(EngError::BadRequest(message)) if message.contains("must be absent or 0")));
    }
    assert!(matches!(
        validate_request(0, Err(std::env::VarError::NotUnicode("invalid".into()))),
        Err(EngError::BadRequest(message)) if message.contains("must be absent or 0")));
    for hot_n in [1, 257] {
        assert!(matches!(validate_request(hot_n, Ok("0".into())),
            Err(EngError::BadRequest(message)) if message.contains("requires hot_n=0")));
    }
}
