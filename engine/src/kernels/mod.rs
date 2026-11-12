#[cfg(test)]
pub(crate) mod attention_reference;
#[cfg(test)]
pub(crate) mod gdn_reference;

#[cfg(test)]
mod evaluate_wall;

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
pub mod exl3_cpu;
mod exl3_cpu_avx2;
mod exl3_cpu_avx512;
mod exl3_cpu_pool;
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
// The CUDA sampler will consume the request encoding; until then
// portable builds compile it for unit tests only.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod sample_canon;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod shape;

pub mod decision;
pub mod f8;
pub mod nvfp4;
pub mod nvfp4_cpu;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod row_launches;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod single_row;

/// Device arena placement alignment: every packed tensor region starts at a
/// multiple of this. Defined here, above the backend trees, because
/// host-side planning (the Flint EXL3 pack placement audit) places tensors
/// with the same rule without a device.
#[cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]
pub(crate) const ARENA_ALIGN: usize = 256;

/// The largest buffer of repacked tiles held on the host at once. Defined
/// here, above the backend trees, for the same reason as [`ARENA_ALIGN`]:
/// host-side pack planning bounds its staging with the same limit.
#[cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]
pub(crate) const CHUNK_LIMIT: usize = 64 << 20;

/// Host time blocked in the stream-synchronise of a completed pageable copy
/// (`write_at` / `read_at`), split by direction. Armed by the same
/// `INFERENCE_FLASH_NEXT_PROFILE` switch as the chunk phase profile: the split decides
/// whether the routing host thread's block sits in the uploads or the readbacks.
#[cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]
pub(crate) mod page_io_block {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    static ENABLED: OnceLock<bool> = OnceLock::new();
    pub(crate) static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
    pub(crate) static READBACK_NS: AtomicU64 = AtomicU64::new(0);

    /// Arms the counters; the chunk profile's own switch calls this, so every copy after the
    /// first step is recorded.
    pub(crate) fn arm() -> bool {
        *ENABLED
            .get_or_init(|| std::env::var("INFERENCE_FLASH_NEXT_PROFILE").is_ok_and(|v| v != "0"))
    }

    /// Whether any copy is being recorded; reading this never arms the switch. The
    /// pageable-copy path checks it BEFORE taking any clock read, per the shared-path rule.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[inline]
    pub(crate) fn armed() -> bool {
        ENABLED.get().copied().unwrap_or(false)
    }

    /// Records `ns` spent blocked; the disarmed path is one load and no store. Only the
    /// native page-io wrappers call this, so it exists where they do.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[inline]
    pub(crate) fn note(upload: bool, ns: u64) {
        if ENABLED.get().copied().unwrap_or(false) {
            (if upload { &UPLOAD_NS } else { &READBACK_NS }).fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// Both counters and a reset; `None` when nothing was recorded.
    pub(crate) fn take() -> Option<(u64, u64)> {
        let up = UPLOAD_NS.swap(0, Ordering::Relaxed);
        let down = READBACK_NS.swap(0, Ordering::Relaxed);
        (up != 0 || down != 0).then_some((up, down))
    }
}

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

/// Per-adapter budget of distinct baked keys, not tapes or bytes. Settings need this
/// checked value on every backend, even when the native compile registry is absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProgramLimit(usize);

impl Default for ProgramLimit {
    fn default() -> Self {
        Self(8)
    }
}

impl ProgramLimit {
    pub(crate) fn new(value: usize) -> Result<Self, EngError> {
        // Vec<u64> must be addressable even if the registry eventually fills the budget.
        if value == 0 || value > isize::MAX as usize / size_of::<u64>() {
            return Err(EngError::BadRequest(
                "compile_program_limit must be positive and fit the registry address space".into(),
            ));
        }
        Ok(Self(value))
    }

    pub(crate) fn get(self) -> usize {
        self.0
    }
}

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

/// Resident unary elementwise operation.
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

/// Resident binary elementwise operation with broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    LogAddExp,
}

/// Resident reduction over one axis.
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
