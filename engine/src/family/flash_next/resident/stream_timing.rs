use super::StepMode;
use crate::{
    family::mlx_stream_stats::StreamSnapshot,
    kernels::{MlxKernels, mlx::native::qualification_process_usage},
};

pub(super) struct ForwardTiming {
    counters: StreamSnapshot,
    position: u32,
}

fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_FLASH_NEXT_MLX_STREAM_TIMING").is_some_and(|value| value == "1")
    })
}

pub(super) fn begin(kernels: &MlxKernels, position: u32) -> Option<ForwardTiming> {
    enabled().then(|| ForwardTiming {
        counters: StreamSnapshot::begin(kernels),
        position,
    })
}

pub(super) fn record_prefix_restore(position: u32, charged_bytes: u64) {
    if enabled() {
        let event = serde_json::json!({"position": position, "charged_bytes": charged_bytes});
        eprintln!("flash_next_prefix_restore {event}");
    }
}

pub(super) fn record_prefix_budget_refusal(position: u32, charged_bytes: u64, limit_bytes: u64) {
    if enabled() {
        let event = serde_json::json!({
            "position": position, "charged_bytes": charged_bytes, "limit_bytes": limit_bytes,
        });
        eprintln!("flash_next_prefix_budget_refusal {event}");
    }
}

pub(super) fn prefill_memory(kernels: &MlxKernels) -> Option<serde_json::Value> {
    if !enabled() {
        return None;
    }
    let footprint = qualification_process_usage().map(|(_, bytes)| bytes);
    let allocator = kernels.memory_stats().ok();
    Some(serde_json::json!({
        "process_footprint_bytes": footprint,
        "mlx_active_bytes": allocator.map(|stats| stats.active),
        "mlx_cache_bytes": allocator.map(|stats| stats.cache),
    }))
}

pub(super) fn record_prefill_release(
    kernels: &MlxKernels,
    mapped_bytes: u64,
    before: serde_json::Value,
) {
    let event = serde_json::json!({
        "mapped_pool_bytes": mapped_bytes,
        "before": before,
        "after": prefill_memory(kernels),
    });
    eprintln!("flash_next_prefill_release {event}");
}

pub(super) fn record_decode_cache(position: u32, stats: super::decode_cache::CopyStats) {
    let event = serde_json::json!({
        "position": position,
        "hit_bytes": stats.hit_bytes,
        "placement_bytes": stats.placement_bytes,
        "copy_ns": stats.copy_ns,
    });
    eprintln!("flash_next_decode_cache {event}");
}

impl ForwardTiming {
    pub(super) fn finish(self, kernels: &MlxKernels, rows: u32, mode: StepMode, route: &str) {
        let mut row = self.counters.finish(kernels);
        row["position"] = self.position.into();
        row["rows"] = rows.into();
        row["mode"] = format!("{mode:?}").into();
        row["forward_path"] = route.into();
        if mode == StepMode::Logits {
            let usage = qualification_process_usage();
            row["process_read_bytes"] = usage.map(|(bytes, _)| bytes).into();
            row["process_footprint_bytes"] = usage.map(|(_, bytes)| bytes).into();
        }
        eprintln!("flash_next_stream_timing {row}");
    }
}

/// Observe the compact expert unions already computed for dense chunks.
pub(super) fn capture_dense_unions() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_FLASH_NEXT_MLX_EXPERT_TRACE").is_some_and(|value| value == "1")
    })
}

pub(super) fn record_dense_unions(position: u32, rows: u32, set_experts: u32, layers: &[Vec<u32>]) {
    let event = serde_json::json!({
        "position": position,
        "rows": rows,
        "set_experts": set_experts,
        "layers": layers,
    });
    eprintln!("flash_next_dense_expert_trace {event}");
}

/// Capture the experts selected by successful one-row sampling forwards.
pub(super) fn capture_decode_routes() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_FLASH_NEXT_MLX_DECODE_EXPERT_TRACE")
            .is_some_and(|value| value == "1")
    })
}

pub(super) fn record_decode_routes(position: u32, layers: &[Vec<u32>]) {
    let event = serde_json::json!({"position": position, "layers": layers});
    eprintln!("flash_next_decode_expert_trace {event}");
}
