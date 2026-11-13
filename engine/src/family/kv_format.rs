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
//! The two Int8 shapes are byte-equal at `{ group: 32, scale_bytes: 1 }` vs
//! `{ group: 64, param_bytes: 1 }`-style coincidences; the affine form carries a scale
//! AND a bias per group, so `param_bytes` counts one of the two.

use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvFormat {
    /// 2-byte elements, no scales.
    F16,
    /// 4-byte elements, no scales.
    F32,
    /// 2-byte elements with reduced mantissa (bfloat16); same payload size as F16.
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
    /// Payload bytes (the coded elements themselves) of one plane of one layer for one
    /// position, checked.
    pub(crate) fn payload_bytes(self, row: KvRowShape) -> Result<u64, EngError> {
        let elements = checked_mul2(row.heads_kv, row.head_dim)?;
        let element_bytes = match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
            Self::Int8 { .. } | Self::Affine8 { .. } => 1,
        };
        checked_mul2(elements, element_bytes)
    }

    /// Scale (or scale-and-bias) bytes of one plane of one layer for one position:
    /// zero for the float forms. Checked, and a zero group is a `BadRequest`.
    pub(crate) fn scale_bytes(self, row: KvRowShape) -> Result<u64, EngError> {
        match self {
            Self::F16 | Self::F32 | Self::Bf16 => Ok(0),
            Self::Int8 { group, scale_bytes } => {
                let span = group_span(group, row.head_dim)?;
                let groups = checked_mul2(row.heads_kv, row.head_dim / span)?;
                checked_mul2(groups, u64::from(scale_bytes))
            }
            Self::Affine8 { group, param_bytes } => {
                let span = group_span(group, row.head_dim)?;
                let groups = checked_mul2(row.heads_kv, row.head_dim / span)?;
                checked_mul2(checked_mul2(groups, 2)?, u64::from(param_bytes))
            }
        }
    }

    /// Payload+scale bytes of one plane of one layer for one position, checked.
    pub(crate) fn plane_position_bytes(self, row: KvRowShape) -> Result<u64, EngError> {
        checked_add2(self.payload_bytes(row)?, self.scale_bytes(row)?)
    }

    /// Same, doubled for K and V.
    pub(crate) fn layer_position_bytes(self, row: KvRowShape) -> Result<u64, EngError> {
        checked_mul2(self.plane_position_bytes(row)?, 2)
    }
}

/// The byte span of one group of an 8-bit arm: refused at zero and when the head
/// dimension does not divide into whole groups.
fn group_span(group: u32, head_dim: u64) -> Result<u64, EngError> {
    let span = u64::from(group);
    if span == 0 || !head_dim.is_multiple_of(span) {
        return Err(EngError::BadRequest(format!(
            "kv format: head_dim {head_dim} is not a multiple of the group {span}"
        )));
    }
    Ok(span)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three figures the families pin at the Flash-Next geometry (12 cached layers,
    /// 2 KV heads, 256 head dim, a 128-float indexer key over 4 positions): the F16
    /// position bytes, Int8 with its per-head f32 scales (one group the width of the
    /// head dimension), and the payload-only raw-element form the tier arithmetic maps
    /// 1-byte codes onto.
    #[test]
    fn the_pinned_family_figures_hold() {
        let row = KvRowShape {
            heads_kv: 2,
            head_dim: 256,
        };
        let indexer = 128 * 4 / 4;
        let f16 = KvFormat::F16.layer_position_bytes(row).unwrap();
        let int8 = KvFormat::Int8 {
            group: 256,
            scale_bytes: 4,
        }
        .layer_position_bytes(row)
        .unwrap();
        let payload_only = KvFormat::Int8 {
            group: 1,
            scale_bytes: 0,
        }
        .layer_position_bytes(row)
        .unwrap();
        assert_eq!(12 * (f16 + indexer), 26_112);
        assert_eq!(12 * (int8 + indexer), 14_016);
        assert_eq!(12 * (payload_only + indexer), 13_824);
    }

    /// A zero group is refused, not silently treated as one.
    #[test]
    fn a_zero_group_is_refused() {
        let row = KvRowShape {
            heads_kv: 2,
            head_dim: 256,
        };
        assert!(
            KvFormat::Int8 {
                group: 0,
                scale_bytes: 4,
            }
            .scale_bytes(row)
            .is_err()
        );
        assert!(
            KvFormat::Affine8 {
                group: 0,
                param_bytes: 1,
            }
            .scale_bytes(row)
            .is_err()
        );
    }
}
