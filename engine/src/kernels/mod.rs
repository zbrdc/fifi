// Native callers consume the checks; portable builds compile them for unit tests only.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) mod compile_key;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod elementwise;
pub mod expert_cpu;
pub mod expert_cpu_avx2;
pub mod expert_cpu_q8g;
pub mod expert_cpu_q8k;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod gated_delta;
mod packed;
mod rope;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod shape;

pub mod decision;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub mod cuda;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod mlx;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub use mlx::{
    BoundedIndices, GatedDeltaOutput, HostPages, MemoryStats, MlxKernels, PackedPages, PageRange,
    QuantizedKv, ResidentPackedWeight,
};

#[cfg(all(
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub use mlx::{QualificationFlushStats, QualificationTiming};

pub use packed::PackedWeight;
pub use rope::{RopeSpec, RopeStep};

use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Affine,
    Mxfp4,
    Mxfp8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationSpec {
    pub mode: QuantizationMode,
    pub bits: u32,
    pub group_size: u32,
}

impl QuantizationSpec {
    /// Checks the width and group size against the declared modes.
    ///
    /// Affine weights use 2, 3, 4, 5, 6, or 8 bits in groups of 32, 64, or 128.
    /// MXFP4 uses 4 bits and MXFP8 uses 8 bits, both in groups of 32.
    /// Metadata parsing and packed-weight construction share this one rule.
    pub(crate) fn validate(&self) -> Result<(), EngError> {
        let bits_ok = match self.mode {
            QuantizationMode::Affine => matches!(self.bits, 2..=6 | 8),
            QuantizationMode::Mxfp4 => self.bits == 4,
            QuantizationMode::Mxfp8 => self.bits == 8,
        };
        if !bits_ok {
            return Err(EngError::Checkpoint(format!(
                "bits {} are unsupported for {:?} weights",
                self.bits, self.mode
            )));
        }
        let group_ok = match self.mode {
            QuantizationMode::Affine => matches!(self.group_size, 32 | 64 | 128),
            QuantizationMode::Mxfp4 | QuantizationMode::Mxfp8 => self.group_size == 32,
        };
        if !group_ok {
            return Err(EngError::Checkpoint(format!(
                "group_size {} is unsupported for {:?} weights",
                self.group_size, self.mode
            )));
        }
        Ok(())
    }
}

/// Resident unary elementwise operation (decision 0039).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Sigmoid,
    Exp,
    Sqrt,
    Rsqrt,
    Neg,
    Abs,
    Sign,
}

/// Resident binary elementwise operation with broadcasting (decision 0039).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    LogAddExp,
}

/// Resident reduction over one axis (decision 0039).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
}

#[cfg(test)]
mod tests {
    use super::{QuantizationMode, QuantizationSpec};
    use crate::EngError;

    fn spec(mode: QuantizationMode, bits: u32, group_size: u32) -> QuantizationSpec {
        QuantizationSpec {
            mode,
            bits,
            group_size,
        }
    }

    #[test]
    fn validate_accepts_every_declared_width_and_group() {
        for bits in [2, 3, 4, 5, 6, 8] {
            for group in [32, 64, 128] {
                spec(QuantizationMode::Affine, bits, group)
                    .validate()
                    .unwrap();
            }
        }
        spec(QuantizationMode::Mxfp4, 4, 32).validate().unwrap();
        spec(QuantizationMode::Mxfp8, 8, 32).validate().unwrap();
    }

    #[test]
    fn validate_names_the_rejected_field() {
        use QuantizationMode::{Affine, Mxfp4, Mxfp8};
        for (candidate, field) in [
            (spec(Affine, 0, 64), "bits"),
            (spec(Affine, 7, 64), "bits"),
            (spec(Affine, u32::MAX, 64), "bits"),
            (spec(Mxfp4, 8, 32), "bits"),
            (spec(Mxfp8, 4, 32), "bits"),
            (spec(Affine, 4, 0), "group_size"),
            (spec(Affine, 4, 16), "group_size"),
            (spec(Affine, 4, 256), "group_size"),
            (spec(Mxfp4, 4, 64), "group_size"),
            (spec(Mxfp8, 8, 128), "group_size"),
        ] {
            let Err(EngError::Checkpoint(message)) = candidate.validate() else {
                panic!("{candidate:?} must fail");
            };
            assert!(message.contains(field), "{candidate:?}: {message}");
            let other = if field == "bits" {
                "group_size"
            } else {
                "bits"
            };
            assert!(!message.contains(other), "{candidate:?}: {message}");
        }
    }
}
