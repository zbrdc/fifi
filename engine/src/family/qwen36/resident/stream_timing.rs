//! Existing MLX wait counters and host-side staging measurements for a streamed forward.

use std::time::Instant;

use super::{decoder::StepMode, weights::ExpertPages, weights::ResidentWeights};
use crate::{EngError, kernels::MlxKernels};

#[cfg(feature = "native-fault-qualification")]
use crate::family::{
    mlx_stream_stats::StreamSnapshot,
    stored::{ExpertReadStats, qualification_expert_reads},
};

pub(super) struct ForwardTiming {
    #[cfg(feature = "native-fault-qualification")]
    counters: StreamSnapshot,
    #[cfg(feature = "native-fault-qualification")]
    position: u32,
    #[cfg(feature = "native-fault-qualification")]
    dense_reads: Option<ExpertReadStats>,
    #[cfg(feature = "native-fault-qualification")]
    expert_trace: Option<Vec<serde_json::Value>>,
    host_route_ns: u64,
    stages: u32,
    stage_fill_ns: u64,
    stage_lifetime_ns: u64,
    peak_stage_bytes: u64,
    peak_stage_experts: u32,
}

fn ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

impl ForwardTiming {
    pub(super) fn begin(k: &MlxKernels, weights: &ResidentWeights, position: u32) -> Option<Self> {
        #[cfg(feature = "native-fault-qualification")]
        {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let enabled = *ENABLED.get_or_init(|| {
                std::env::var_os("FIFI_Q36_MLX_STREAM_TIMING").is_some_and(|value| value == "1")
            });
            static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let trace_enabled = *TRACE.get_or_init(|| {
                std::env::var_os("FIFI_Q36_MLX_EXPERT_TRACE").is_some_and(|value| value == "1")
            });
            let dense = enabled && weights.stages_dense();
            let streamed = weights.streams_experts();
            let trace = trace_enabled && streamed;
            (dense || trace || (enabled && streamed)).then(|| Self {
                counters: StreamSnapshot::begin(k),
                position,
                dense_reads: dense.then(qualification_expert_reads),
                expert_trace: trace.then(Vec::new),
                host_route_ns: 0,
                stages: 0,
                stage_fill_ns: 0,
                stage_lifetime_ns: 0,
                peak_stage_bytes: 0,
                peak_stage_experts: 0,
            })
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = (k, weights, position);
            None
        }
    }

    pub(super) fn host_route<T>(timing: Option<&mut Self>, route: impl FnOnce() -> T) -> T {
        let Some(timing) = timing else {
            return route();
        };
        let started = Instant::now();
        let value = route();
        timing.host_route_ns += ns(started);
        value
    }

    /// Trace the exact checkpoint IDs passed to a successful stage, with its logical I/O.
    /// Stages are appended in model layer order. Prompt forwards omit the final MoE.
    pub(super) fn load_stage(
        timing: Option<&mut Self>,
        experts: &[u32],
        coalesced: bool,
        load: impl FnOnce() -> Result<ExpertPages, EngError>,
    ) -> Result<ExpertPages, EngError> {
        #[cfg(feature = "native-fault-qualification")]
        if let Some(trace) = timing.and_then(|timing| timing.expert_trace.as_mut()) {
            let before = qualification_expert_reads();
            let pages = load()?;
            let after = qualification_expert_reads();
            trace.push(serde_json::json!({
                "layer": trace.len(),
                "experts": experts,
                "coalesced": coalesced,
                "slice_reads": after.slice_reads - before.slice_reads,
                "read_bytes": after.read_bytes - before.read_bytes,
                "mapped_bytes": pages.page_bytes()?,
            }));
            return Ok(pages);
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        let _ = (timing, experts, coalesced);
        load()
    }

    pub(super) fn stage_ready(
        timing: Option<&mut Self>,
        started: Option<Instant>,
        pages: Option<&ExpertPages>,
    ) -> Result<(), EngError> {
        if let (Some(timing), Some(started), Some(pages)) = (timing, started, pages) {
            timing.stages += 1;
            timing.stage_fill_ns += ns(started);
            timing.peak_stage_bytes = timing.peak_stage_bytes.max(pages.page_bytes()?);
            timing.peak_stage_experts =
                timing.peak_stage_experts.max(pages.down.logical_shape()[0]);
        }
        Ok(())
    }

    pub(super) fn stage_end(timing: Option<&mut Self>, started: Option<Instant>) {
        if let (Some(timing), Some(started)) = (timing, started) {
            timing.stage_lifetime_ns += ns(started);
        }
    }

    pub(super) fn finish(self, k: &MlxKernels, rows: u32, mode: StepMode) {
        #[cfg(feature = "native-fault-qualification")]
        {
            let mut row = self.counters.finish(k);
            row["position"] = self.position.into();
            row["rows"] = rows.into();
            row["mode"] = format!("{mode:?}").into();
            row["host_route_ns"] = self.host_route_ns.into();
            row["stages"] = self.stages.into();
            row["stage_fill_ns"] = self.stage_fill_ns.into();
            row["stage_lifetime_ns"] = self.stage_lifetime_ns.into();
            row["peak_stage_bytes"] = self.peak_stage_bytes.into();
            row["peak_stage_experts"] = self.peak_stage_experts.into();
            if let Some(before) = self.dense_reads {
                let after = qualification_expert_reads();
                row["dense_ffn_read_bytes"] =
                    after.read_bytes.saturating_sub(before.read_bytes).into();
                row["dense_ffn_slice_reads"] =
                    after.slice_reads.saturating_sub(before.slice_reads).into();
            }
            if let Some(trace) = self.expert_trace {
                row["expert_trace"] = trace.into();
            }
            eprintln!("qwen36_stream_timing {row}");
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        let _ = (k, rows, mode);
    }
}
