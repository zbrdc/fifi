//! The shared K/V byte arithmetic: one format type that states, per plane and per layer,
//! what one cache position costs. The families keep their own enums, parsers and
//! allocation layouts; this module carries only the numbers, so every family that maps
//! its format onto this type computes the same bytes from the same shape.
//!
//! Shape table (bytes of ONE plane of one layer for ONE position — `layer_position_bytes`
//! doubles it for K and V):
//!
//! | variant | payload | scales/params |
//! |---|---|---|
//! | `F16` | `heads*dim*2` | — |
//! | `F32` | `heads*dim*4` | — |
//! | `Bf16` | `heads*dim*2` | — |
//! | `Int8 { group, scale_bytes }` | `heads*dim` | `heads*(dim/group)*scale_bytes` |
//! | `Affine8 { group, param_bytes }` | `heads*dim` | `heads*(dim/group)*param_bytes*2` |
//!
//! The two Int8 shapes are byte-equal at `{ group: 32, scale_bytes: 2 }` vs
//! `{ group: 64, param_bytes: 1 }`-style coincidences; the affine form carries a scale
//! AND a bias per group, so `param_bytes` counts one of the two.

use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvFormat {
    /// 2-byte elements, no scales.
    F16,
    /// 4-byte elements, no scales.
    F32,
    /// 2-byte elements appended with a cast.
    Bf16,
    /// 1-byte codes, `scale_bytes` of scale per group of `group` elements.
    Int8 { group: u32, scale_bytes: u32 },
    /// 1-byte affine codes, a scale AND a bias of `param_bytes` each per group.
    Affine8 { group: u32, param_bytes: u32 },
}

/// One K/V plane's row shape (per layer, per position). No `planes` field: mixed-plane
/// formats exist, so bytes are computed per plane and the caller doubles for K+V.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KvRowShape {
    pub heads_kv: u64,
    pub head_dim: u64,
}

fn checked_mul2(a: u64, b: u64) -> Result<u64, EngError> {
    a.checked_mul(b)
        .ok_or_else(|| EngError::MemoryPlan("kv format: position bytes overflow u64".to_owned()))
}

fn checked_add2(a: u64, b: u64) -> Result<u64, EngError> {
    a.checked_add(b)
        .ok_or_else(|| EngError::MemoryPlan("kv format: position bytes overflow u64".to_owned()))
}

impl KvFormat {
    /// Payload+scale bytes of one plane of one layer for one position, checked.
    pub(crate) fn plane_position_bytes(self, row: KvRowShape) -> Result<u64, EngError> {
        let elements = checked_mul2(row.heads_kv, row.head_dim)?;
        match self {
            Self::F16 => checked_mul2(elements, 2),
            Self::F32 => checked_mul2(elements, 4),
            Self::Bf16 => checked_mul2(elements, 2),
            Self::Int8 { group, scale_bytes } => {
                let group = u64::from(group).max(1);
                if !row.head_dim.is_multiple_of(group) {
                    return Err(EngError::BadRequest(format!(
                        "kv format: head_dim {} is not a multiple of the group {group}",
                        row.head_dim
                    )));
                }
                let groups = checked_mul2(row.heads_kv, row.head_dim / group)?;
                let scales = checked_mul2(groups, u64::from(scale_bytes))?;
                checked_add2(elements, scales)
            }
            Self::Affine8 { group, param_bytes } => {
                let group = u64::from(group).max(1);
                if !row.head_dim.is_multiple_of(group) {
                    return Err(EngError::BadRequest(format!(
                        "kv format: head_dim {} is not a multiple of the group {group}",
                        row.head_dim
                    )));
                }
                let groups = checked_mul2(row.heads_kv, row.head_dim / group)?;
                let params = checked_mul2(checked_mul2(groups, 2)?, u64::from(param_bytes))?;
                checked_add2(elements, params)
            }
        }
    }

    /// Same, doubled for K and V.
    pub(crate) fn layer_position_bytes(self, row: KvRowShape) -> Result<u64, EngError> {
        checked_mul2(self.plane_position_bytes(row)?, 2)
    }
}
