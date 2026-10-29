//! Host-only tier policy and residency tables of the Q36 expert banks.
//!
//! Qwen3.6-35B-A3B decode keeps the expert weights in the OS page cache and holds a
//! small per-layer VRAM bank of the hot experts; the rest is served on the CPU. This
//! module is the pure-host brain of that split: the config and per-token report, the
//! tier policy of the bank ([`policy`]) and the residency ledger of its slots
//! ([`residency`]). There is no CUDA here and no clock read outside [`server`]'s
//! thread and the pool's bounded `finish` (whose waits time themselves); time
//! otherwise arrives as a parameter, and the only file access is the read-only
//! mapping of [`mapped_file`]. The tiered step uses it through `cuda/tiered_step.rs`
//! and `cuda/tiered_load.rs`.

pub mod budget;
pub mod classify;
pub mod cpu_pool;
pub mod guess;
pub mod host;
pub mod locator;
pub mod mapped_file;
pub mod policy;
pub mod report;
pub mod residency;
pub mod server;
pub mod watchdog;

/// How the expert bank chooses its residents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyMode {
    /// The bank holds a fixed profile for the whole run and never swaps.
    Static,
    /// The bank starts empty and chases the routing popularity, credit-limited.
    Adaptive,
}

/// The shape and the rates of the tiered expert banks, plain data so it can travel
/// between threads. `check` validates the invariants before any consumer runs.
#[derive(Clone, Debug, PartialEq)]
pub struct TieredConfig {
    /// MoE layer count (40 for Qwen3.6-35B-A3B); layers are indexed `0..layers`.
    pub layers: usize,
    /// Experts per MoE layer (256 for Qwen3.6-35B-A3B); ids are `0..experts`.
    pub experts: usize,
    /// Experts routed per token and layer (top-8 for Qwen3.6-35B-A3B).
    pub top_k: usize,
    /// VRAM bank slots per layer; a layer bank holds at most this many experts.
    pub bank_slots_per_layer: usize,
    /// The bank policy: a fixed profile or the adaptive popularity chase.
    pub mode: PolicyMode,
    /// Swap credit added per token; `floor` of the credit is the swap budget of a
    /// token boundary. Units: swaps per token (0.25 by default).
    pub swaps_per_token: f64,
    /// Popularity decay applied per token, in `(0, 1]` (0.999 keeps roughly a
    /// thousand-token horizon).
    pub decay: f64,
}

impl TieredConfig {
    /// The Qwen3.6-35B-A3B decode shape: 40 MoE layers, 256 experts, top-8 routing,
    /// the given bank size and mode, and the defaults of 0.25 swaps per token with a
    /// 0.999 decay.
    pub fn qwen36(bank_slots_per_layer: usize, mode: PolicyMode) -> Self {
        Self {
            layers: 40,
            experts: 256,
            top_k: 8,
            bank_slots_per_layer,
            mode,
            swaps_per_token: 0.25,
            decay: 0.999,
        }
    }

    /// Validates the invariants every consumer relies on: nonzero dimensions, a
    /// routing and a bank that fit the expert count, a decay in `(0, 1]` and a
    /// non-negative, finite swap rate. `Err` names the broken invariant.
    pub fn check(&self) -> Result<(), String> {
        if self.layers == 0 {
            return Err(String::from("tiered config: layers must be nonzero"));
        }
        if self.experts == 0 {
            return Err(String::from("tiered config: experts must be nonzero"));
        }
        // Expert ids and slot indices are stored as u16 by the policy and the
        // residency table; a larger layer would truncate them silently.
        if self.experts > usize::from(u16::MAX) + 1 {
            return Err(format!(
                "tiered config: {experts} experts exceed the 65,536 a u16 id can name",
                experts = self.experts
            ));
        }
        if self.bank_slots_per_layer == 0 {
            return Err(String::from(
                "tiered config: bank_slots_per_layer must be nonzero",
            ));
        }
        if self.top_k > self.experts {
            return Err(format!(
                "tiered config: top_k {top_k} exceeds the {experts} experts",
                top_k = self.top_k,
                experts = self.experts
            ));
        }
        if self.bank_slots_per_layer > self.experts {
            return Err(format!(
                "tiered config: bank of {bank} slots exceeds the {experts} experts",
                bank = self.bank_slots_per_layer,
                experts = self.experts
            ));
        }
        if !(0.0 < self.decay && self.decay <= 1.0) {
            return Err(format!(
                "tiered config: decay {decay} is outside (0, 1]",
                decay = self.decay
            ));
        }
        if !self.swaps_per_token.is_finite() || self.swaps_per_token < 0.0 {
            return Err(format!(
                "tiered config: swaps_per_token {rate} is negative or not finite",
                rate = self.swaps_per_token
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PolicyMode, TieredConfig};

    #[test]
    fn tiered_config_defaults_and_checks() {
        let cfg = TieredConfig::qwen36(61, PolicyMode::Adaptive);
        assert_eq!(
            (cfg.layers, cfg.experts, cfg.top_k, cfg.bank_slots_per_layer),
            (40, 256, 8, 61)
        );
        assert_eq!((cfg.swaps_per_token, cfg.decay), (0.25, 0.999));
        assert_eq!(cfg.mode, PolicyMode::Adaptive);
        assert_eq!(TieredConfig::qwen36(4, PolicyMode::Static).check(), Ok(()));
        for bad in [
            TieredConfig {
                layers: 0,
                ..cfg.clone()
            },
            TieredConfig {
                experts: 0,
                ..cfg.clone()
            },
            TieredConfig {
                bank_slots_per_layer: 0,
                ..cfg.clone()
            },
            TieredConfig {
                top_k: 257,
                ..cfg.clone()
            },
            TieredConfig {
                bank_slots_per_layer: 257,
                ..cfg.clone()
            },
            TieredConfig {
                decay: 0.0,
                ..cfg.clone()
            },
            TieredConfig {
                decay: 1.001,
                ..cfg.clone()
            },
            TieredConfig {
                swaps_per_token: -0.1,
                ..cfg.clone()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?} must be refused");
        }
        // Boundary values that stay legal: decay of exactly 1 and a zero rate.
        assert_eq!(
            TieredConfig {
                decay: 1.0,
                ..cfg.clone()
            }
            .check(),
            Ok(())
        );
        assert_eq!(
            TieredConfig {
                swaps_per_token: 0.0,
                ..cfg
            }
            .check(),
            Ok(())
        );
    }

    #[test]
    fn tiered_config_refuses_more_experts_than_a_u16_id_names() {
        let wide = |experts: usize| TieredConfig {
            layers: 1,
            experts,
            top_k: 1,
            bank_slots_per_layer: experts,
            mode: PolicyMode::Adaptive,
            swaps_per_token: 0.25,
            decay: 0.999,
        };
        assert_eq!(wide(65_536).check(), Ok(()));
        assert!(wide(65_537).check().is_err());
        assert!(wide(70_000).check().is_err());
    }
}
