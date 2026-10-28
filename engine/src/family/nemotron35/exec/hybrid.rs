//! The Nemotron configuration of the shared expert tier (`crate::expert_tier`).
//!
//! `NEM35_CPU_EXPERTS=1` turns the hybrid on: the routed experts of most MoE layers run on the
//! host CPU from read-only mapped shards, a few layers stay resident on the GPU, and nothing is
//! uploaded per token for a routed expert. This file holds only what is Nemotron's: the
//! environment variables and the runtime state the stack carries; placement, residency, the CPU
//! pool and the CPU/GPU hand-off are the tier's.

use std::time::Duration;

pub use crate::expert_tier::{GpuLayerPlan, TierReport as HybridReport, TokenSplit};
use crate::{
    EngError,
    expert_tier::{Residency, TierReport},
    family::nemotron35::cpu_experts::{CpuExperts, DEFAULT_THREADS},
};

/// Environment-selected hybrid configuration (`NEM35_CPU_EXPERTS=1` turns it on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridConfig {
    pub cpu_threads: usize,
    pub gpu_layers: GpuLayerPlan,
    /// Run the GPU-resident layers through the batched NVFP4 kernel (`NEM35_GPU_BATCH=1`). Off by
    /// default: that kernel returns zeros at production shapes on the branch this was cut from.
    pub gpu_batch: bool,
}

impl HybridConfig {
    /// Reads `NEM35_CPU_EXPERTS`, `NEM35_GPU_MOE_LAYERS` and `NEM35_CPU_THREADS`.
    /// `None` means the flag is off and the sequential path runs unchanged.
    pub fn from_env() -> Result<Option<Self>, EngError> {
        let read = |name: &str| std::env::var(name).ok();
        Self::parse(
            read("NEM35_CPU_EXPERTS").as_deref(),
            read("NEM35_GPU_MOE_LAYERS").as_deref(),
            read("NEM35_CPU_THREADS").as_deref(),
            read("NEM35_GPU_BATCH").as_deref(),
        )
    }

    pub(crate) fn parse(
        enabled: Option<&str>,
        gpu_layers: Option<&str>,
        threads: Option<&str>,
        gpu_batch: Option<&str>,
    ) -> Result<Option<Self>, EngError> {
        match enabled {
            None | Some("" | "0") => return Ok(None),
            Some("1") => {}
            Some(other) => {
                return Err(bad(format!(
                    "NEM35_CPU_EXPERTS must be 0 or 1, not {other:?}"
                )));
            }
        }
        let number = |name: &str, value: &str| {
            value
                .parse::<usize>()
                .map_err(|_| bad(format!("{name} must be a whole number, not {value:?}")))
        };
        let gpu_layers = match gpu_layers {
            None | Some("" | "auto") => GpuLayerPlan::Auto,
            Some(value) => GpuLayerPlan::Fixed(number("NEM35_GPU_MOE_LAYERS", value)?),
        };
        let cpu_threads = match threads {
            None | Some("") => DEFAULT_THREADS,
            Some(value) => number("NEM35_CPU_THREADS", value)?,
        };
        if cpu_threads == 0 {
            return Err(bad("NEM35_CPU_THREADS must be at least 1"));
        }
        let gpu_batch = match gpu_batch {
            None | Some("" | "0") => false,
            Some("1") => true,
            Some(other) => {
                return Err(bad(format!(
                    "NEM35_GPU_BATCH must be 0 or 1, not {other:?}"
                )));
            }
        };
        Ok(Some(Self {
            cpu_threads,
            gpu_layers,
            gpu_batch,
        }))
    }
}

pub(crate) struct HybridRuntime {
    pub(super) cpu: CpuExperts,
    pub(super) residency: Residency,
    pub(super) gpu_batch: bool,
    pub(super) gpu_resident_bytes: u64,
    pub(super) load: Duration,
    pub(super) current: TokenSplit,
    pub(super) tokens: Vec<TokenSplit>,
    /// Two PCIe banks planned before placement and allocated on the first streamed chunk.
    /// Their host/device reservation lasts until proven context cleanup.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) stream_banks: Option<crate::expert_tier::CudaStreamBanks>,
}

impl HybridRuntime {
    pub(super) fn new(cpu: CpuExperts, gpu_batch: bool) -> Self {
        Self {
            cpu,
            residency: Residency::default(),
            gpu_batch,
            gpu_resident_bytes: 0,
            load: Duration::ZERO,
            current: TokenSplit::default(),
            tokens: Vec::new(),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            stream_banks: None,
        }
    }

    pub(super) fn report(&self) -> TierReport {
        TierReport {
            cpu_threads: self.cpu.threads(),
            cpu_layers: self.cpu.layer_ids(),
            gpu_layers: self.residency.full_layers(),
            gpu_resident_experts: self.residency.resident_experts(),
            gpu_resident_bytes: self.gpu_resident_bytes,
            load: self.load,
            tokens: self.tokens.clone(),
        }
    }
}

fn bad(message: impl Into<String>) -> EngError {
    EngError::BadRequest(format!("Nemotron hybrid: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_is_off_unless_it_is_exactly_one() {
        for off in [None, Some(""), Some("0")] {
            assert_eq!(
                HybridConfig::parse(off, Some("3"), None, None).unwrap(),
                None
            );
        }
        assert!(HybridConfig::parse(Some("yes"), None, None, None).is_err());
        assert!(HybridConfig::parse(Some("2"), None, None, None).is_err());
    }

    #[test]
    fn defaults_are_auto_layers_and_twelve_threads() {
        let config = HybridConfig::parse(Some("1"), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(config.gpu_layers, GpuLayerPlan::Auto);
        assert_eq!(config.cpu_threads, 12);
        assert!(!config.gpu_batch, "the batched kernel is opt-in");
        assert!(HybridConfig::parse(Some("1"), None, None, Some("yes")).is_err());
        let again = HybridConfig::parse(Some("1"), Some("auto"), Some(""), Some(""))
            .unwrap()
            .unwrap();
        assert_eq!(again, config);
    }

    #[test]
    fn the_environment_overrides_layers_and_threads() {
        let config = HybridConfig::parse(Some("1"), Some("9"), Some("8"), Some("1"))
            .unwrap()
            .unwrap();
        assert_eq!(config.gpu_layers, GpuLayerPlan::Fixed(9));
        assert_eq!(config.cpu_threads, 8);
        assert!(config.gpu_batch);
        let none = HybridConfig::parse(Some("1"), Some("0"), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(none.gpu_layers, GpuLayerPlan::Fixed(0));
        for (layers, threads) in [
            (Some("-1"), None),
            (Some("x"), None),
            (None, Some("0")),
            (None, Some("many")),
        ] {
            assert!(HybridConfig::parse(Some("1"), layers, threads, None).is_err());
        }
    }
}
