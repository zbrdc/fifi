//! Startup working-set budget for an opted-in Apple Silicon model.

use std::{ffi::OsStr, fmt};

use crate::{EngError, family::device_budget::DeviceReserve};

const CEILING_MIB: &str = "FIFI_MLX_WORKING_SET_MIB";

/// Family reserves leave this much room for memory outside their tensor estimates.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) const PRESSURE_MARGIN: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceBudget {
    recommended: u64,
    ceiling: Option<u64>,
}

impl DeviceBudget {
    pub(crate) fn new(bytes: u64) -> Result<Self, EngError> {
        if bytes == 0 {
            return Err(EngError::MemoryPlan(
                "Metal did not report a recommended working set".into(),
            ));
        }
        Ok(Self {
            recommended: bytes,
            ceiling: None,
        })
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn detect() -> Result<Option<Self>, EngError> {
        if enabled() {
            let ceiling = configured_ceiling()?;
            let device = Self::new(crate::kernels::MlxKernels::recommended_working_set_bytes())?;
            Ok(Some(Self { ceiling, ..device }))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn bytes(self) -> u64 {
        self.recommended.min(self.ceiling.unwrap_or(u64::MAX))
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn ceiling_report(self) -> super::mlx_report::MlxDeviceCeiling {
        super::mlx_report::MlxDeviceCeiling {
            metal_working_set_bytes: self.recommended,
            effective_device_budget_bytes: self.bytes(),
            configured_device_ceiling_bytes: self.ceiling,
        }
    }

    /// Constrain an admission check by a separate budget, leaving the original device
    /// budget available for reporting.
    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(super) fn capped_for_admission(self, cap: u64) -> Self {
        Self {
            ceiling: Some(self.ceiling.map_or(cap, |ceiling| ceiling.min(cap))),
            ..self
        }
    }

    /// A configured upper bound can make the working set smaller.
    pub(crate) fn prefill(self, configured: Option<u64>) -> u64 {
        self.bytes().min(configured.unwrap_or(u64::MAX))
    }

    /// Refuse a model when its mandatory packed payloads alone exceed the
    /// budget left for weights. The loader still charges the full resident
    /// tensors and staging pages after import.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn check_packed_payload_floor(
        self,
        family: &str,
        ckpt: &crate::weights::StreamedCheckpoint,
        reserve: u64,
        is_resident: impl Fn(&str) -> bool,
    ) -> Result<(), EngError> {
        let weights = ckpt
            .names()
            .filter(|key| key.ends_with(".weight") && is_resident(key))
            .try_fold(0u64, |total, key| {
                total.checked_add(ckpt.stored_byte_len(key)?)
            })
            .ok_or_else(|| {
                EngError::MemoryPlan(format!("{family} packed payload bytes overflow"))
            })?;
        self.remaining(
            family,
            &[
                ("packed_weight_payload_floor_bytes", weights),
                ("session_reserve_bytes", reserve),
            ],
        )?;
        Ok(())
    }

    /// The space left after the resident charges and the family's reserve.
    pub(crate) fn remaining(
        self,
        family: &str,
        charges: &[(&'static str, u64)],
    ) -> Result<u64, EngError> {
        let reserve = charges
            .iter()
            .fold(DeviceReserve::new(0), |plan, &(label, bytes)| {
                plan.push(label, bytes)
            });
        let total = reserve.checked_total().ok_or_else(|| {
            EngError::MemoryPlan(format!("{family} MLX startup charges overflow u64"))
        })?;
        self.bytes().checked_sub(total).ok_or_else(|| {
            let terms = reserve.charges().iter().map(|(name, bytes)| format!("{name}={bytes}"))
                .collect::<Vec<_>>().join(" ");
            EngError::MemoryPlan(format!(
                "{family} requires {total} bytes for the resident model and session reserve. Metal recommends {} bytes, with {} bytes admitted ({terms})",
                self.recommended,
                self.bytes(),
            ))
        })
    }
}

impl fmt::Display for DeviceBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "metal_working_set_bytes={} effective_device_budget_bytes={}",
            self.recommended,
            self.bytes()
        )?;
        if let Some(ceiling) = self.ceiling {
            write!(f, " configured_device_ceiling_bytes={ceiling}")?;
        }
        Ok(())
    }
}

fn parse_ceiling(value: &OsStr) -> Result<u64, EngError> {
    value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|mib| mib.checked_mul(1 << 20))
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| {
            EngError::BadRequest(format!(
                "{CEILING_MIB} must be a positive whole number of MiB that fits in u64 bytes"
            ))
        })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn enabled() -> bool {
    std::env::var_os("FIFI_MLX_AUTO_BUDGET").is_some_and(|value| value == "1")
        || std::env::var_os(CEILING_MIB).is_some()
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn configured_ceiling() -> Result<Option<u64>, EngError> {
    std::env::var_os(CEILING_MIB)
        .as_deref()
        .map(parse_ceiling)
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_working_set_caps_explicit_budget_and_checks_base_charges() {
        let small = DeviceBudget::new(12 << 30).unwrap();
        let large = DeviceBudget::new(24 << 30).unwrap();
        assert_eq!(small.prefill(None), small.bytes());
        assert_eq!(small.prefill(Some(16 << 30)), 12 << 30);
        assert_eq!(large.prefill(Some(16 << 30)), 16 << 30);
        let charges = [("weights", 11 << 30), ("session", 2 << 30)];
        assert!(matches!(
            small.remaining("fixture", &charges),
            Err(EngError::MemoryPlan(_))
        ));
        assert_eq!(large.remaining("fixture", &charges).unwrap(), 11 << 30);
        assert!(
            large
                .remaining("fixture", &[("one", u64::MAX), ("two", 1)])
                .is_err()
        );
        assert!(DeviceBudget::new(0).is_err());
    }
    #[test]
    fn explicit_ceiling_bounds_every_family_without_overriding_metal() {
        let ceiling = |mib| Some(parse_ceiling(OsStr::new(mib)).unwrap());
        let device = DeviceBudget::new(12 << 30).unwrap();
        let constrained = DeviceBudget {
            ceiling: ceiling("4096"),
            ..device
        };
        assert_eq!(constrained.bytes(), 4 << 30);
        assert_eq!(constrained.prefill(Some(2 << 30)), 2 << 30);
        assert_eq!(constrained.prefill(Some(8 << 30)), 4 << 30);
        assert_eq!(constrained.capped_for_admission(8 << 30).bytes(), 4 << 30);
        assert_eq!(constrained.capped_for_admission(3 << 30).bytes(), 3 << 30);
        assert_eq!(device.capped_for_admission(16 << 30).bytes(), 12 << 30);
        assert!(
            constrained
                .remaining("fixture", &[("weights", 5 << 30)])
                .is_err()
        );
        let larger = DeviceBudget {
            ceiling: ceiling("16384"),
            ..device
        };
        assert_eq!(larger.bytes(), device.bytes());
        for invalid in ["", "0", "-1", "4.5", "17592186044416"] {
            assert!(parse_ceiling(OsStr::new(invalid)).is_err(), "{invalid}");
        }
    }
}
