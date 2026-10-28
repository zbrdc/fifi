//! The shared expert tier: routed experts split between VRAM, the host CPU and the page cache,
//! each expert in exactly one place, for any large mixture-of-experts model.
//!
//! WHAT lives here, with no model family named anywhere:
//! - [`Residency`]: which experts of which layer are device-resident (one assignment per expert) and how a
//!   layer's picks split ([`Coverage`]). Whole-layer placement fills it today; a popularity-chosen
//!   hot set can fill it later without touching the decode path.
//! - [`place_gpu_layers`]: uploads the chosen experts once, fits the count to the free VRAM, and
//!   drops the host page cache of what it uploaded so no expert counts twice.
//! - [`CpuTier`]: the CPU worker pool plus the zero-copy read-only mapped views of the experts
//!   that stay on the host, behind a format-specific [`CpuExpertKernel`].
//! - [`run_overlapped`]: the hand-off that runs the CPU experts on a scoped thread while the
//!   caller keeps the device busy with the shared expert.
//! - [`TokenSplit`] and [`TierReport`]: where each decoded token spent its time.
//!
//! WHAT a model family supplies, the whole narrow interface:
//! 1. expert id to weight views: [`ExpertPayloadSource`] reads one expert's bytes for the device
//!    upload, and the family maps the host-served layers read-only ([`map_read_only`]) into
//!    whatever view type its CPU kernel takes;
//! 2. a format-specific CPU kernel: [`CpuExpertKernel`] (Nemotron: NVFP4 on AVX-512/AVX2);
//! 3. a GPU expert kernel: [`GpuExpertStore`] uploads an expert part and reports free VRAM; the
//!    family's decode step calls its own device kernel on resident keys.
//!
//! The CPU pool and mapping follow the patterns of `family::qwen36::tiered` (`cpu_pool`,
//! `mapped_file`, `host`), whose [`MappedFile`](crate::family::qwen36::tiered::mapped_file) this
//! reuses unchanged; extracting that module onto this interface is the next step.

mod cpu;
pub(crate) mod gpu_stream;
mod overlap;
mod plan;
mod report;
mod residency;
mod stream;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod stream_cuda;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
pub(crate) mod stream_cuda_profile;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod stream_nvfp4;

pub use cpu::{CpuExpertKernel, CpuTier, ExpertPick, map_read_only};
pub use overlap::{Overlap, run_overlapped};
pub use plan::{
    ExpertPayload, ExpertPayloadSource, ExpertPayloadView, GpuExpertStore, GpuLayerPlan,
    VRAM_HEADROOM_BYTES, auto_fit_gpu_layers, place_gpu_layers,
};
pub use report::{TierReport, TokenSplit};
pub use residency::{Coverage, Placement, Residency};
pub use stream::StreamBankPlan;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use stream_cuda::CudaStreamBanks;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use stream_nvfp4::launch_device_input as project_nvfp4_batch_device;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use stream_nvfp4::launch_host_input as project_nvfp4_batch_launch;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use stream_nvfp4::project as project_nvfp4_batch;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use stream_nvfp4::read_f32_rows as project_nvfp4_read;

/// One routed expert of one MoE layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}
