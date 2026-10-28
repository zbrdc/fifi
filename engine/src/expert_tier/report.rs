//! Where each decoded token spent its time, and what the tier did.

use std::time::Duration;

/// Where one decoded token spent its time. Parts overlap only where stated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenSplit {
    /// The whole `step`, embedding to greedy head.
    pub wall: Duration,
    /// CPU routed-expert kernel, summed over the CPU layers (runs beside `shared`).
    pub cpu_routed: Duration,
    /// Time the main thread waited for the CPU kernel after its shared expert finished.
    pub cpu_exposed: Duration,
    /// Shared expert through the device projector, summed over the CPU layers.
    pub shared: Duration,
    /// Routed experts of the device-resident layers.
    pub gpu_routed: Duration,
    /// Device projection calls of the whole token, uploads excluded.
    pub device_calls: Duration,
    /// Weight uploads (allocation plus host-to-device copy) of the whole token.
    pub uploads: Duration,
    pub upload_bytes: u64,
}

/// What the tier did, for a harness report.
#[derive(Clone, Debug, Default)]
pub struct TierReport {
    pub cpu_threads: usize,
    pub cpu_layers: Vec<u32>,
    /// Layers whose every routed expert is device-resident.
    pub gpu_layers: Vec<u32>,
    /// Routed experts that are device-resident, whole layers or not.
    pub gpu_resident_experts: usize,
    pub gpu_resident_bytes: u64,
    pub load: Duration,
    pub tokens: Vec<TokenSplit>,
}
