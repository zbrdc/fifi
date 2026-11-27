//! SenseNova Loop 9B (`sensenova_loop` GGUF architecture).
//!
//! A Qwen3-Next-lineage hybrid: gated delta net layers interleaved with
//! full-attention layers, partial YaRN MRoPE, and two structural extensions
//! the vendor graph adds on top:
//!
//! * **Loop**: the layer range `[loop.start, loop.end)` is executed
//!   `loop.count` times per pass; every revisit owns its own recurrent and
//!   K/V state while sharing the weights of the first visit.
//! * **mHC** (multi-hop carry): layers whose physical index falls in
//!   `[mhc.start, mhc.end)` replace the residual stream with two routed
//!   streams merged by a Sinkhorn-normalized mixing matrix.
//!
//! The math mirrors the vendor implementation at
//! `OpenSenseNova/llama.cpp@24ea7bb7` (`src/models/sensenova-loop.cpp`,
//! `qwen35.cpp`, `delta-net-base.cpp`, `ggml` rope/YaRN).

pub(crate) mod cuda;
pub(crate) mod serve;
mod step;
#[cfg(test)]
mod tests;
mod weights;
#[cfg(feature = "native-cuda")]
pub(crate) use weights::LoopShape;

/// The family name the config and the server registry use.
pub(crate) const FAMILY: &str = "sensenova";
pub(crate) const ARCHITECTURE: &str = "sensenova_loop";

use crate::{
    EngError,
    weights::{ArrayMeta, GgufHeader, GgufWeights, MetaValue},
};

/// Logical trunk visits. The last GGUF block is a separate MTP draft head.
pub(crate) const TRUNK_LAYERS: usize = 56;
/// Unique trunk weight sets after removing the 24 loop revisits.
pub(crate) const PHYSICAL_LAYERS: usize = 32;

/// Layer geometry and the Loop/mHC execution plan.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LoopPlan {
    /// First loop layer (logical), inclusive.
    pub(crate) loop_start: usize,
    /// One past the last loop layer (logical), exclusive.
    pub(crate) loop_end: usize,
    /// Passes over the loop range, at least one.
    pub(crate) loop_count: usize,
    /// First mHC layer (physical), inclusive.
    pub(crate) mhc_start: usize,
    /// One past the last mHC layer (physical), exclusive.
    pub(crate) mhc_end: usize,
    /// Sinkhorn iterations for the mixing matrix.
    pub(crate) mhc_iterations: usize,
    /// Scale applied to the routed input.
    pub(crate) mhc_pre_scale: f32,
    /// Whether the mixing matrix is fixed to the identity (4 router rows).
    pub(crate) mhc_fixed_identity: bool,
    /// Per-logical-trunk-layer attention kind: `true` = gated delta net.
    pub(crate) recurrent: Vec<bool>,
}

impl LoopPlan {
    pub(crate) fn from_header(
        weights: &GgufWeights,
        header: &GgufHeader,
        arch: &str,
    ) -> Result<Self, EngError> {
        let u32_of = |key: &str| -> Result<u32, EngError> {
            scalar(header, key).and_then(|value| match value {
                MetaValue::U32(v) => Ok(v),
                _ => Err(reject(format!("{key} is not a u32"))),
            })
        };
        let block_count = u32_of(&format!("{arch}.block_count"))? as usize;
        let nextn = u32_of(&format!("{arch}.nextn_predict_layers"))? as usize;
        if block_count != TRUNK_LAYERS + 1 || nextn != 1 {
            return Err(reject("expected 56 trunk visits and one MTP block"));
        }
        let loop_start = u32_of(&format!("{arch}.loop.start"))? as usize;
        let loop_end = u32_of(&format!("{arch}.loop.end"))? as usize;
        let loop_count = u32_of(&format!("{arch}.loop.count"))? as usize;
        let mhc_start = u32_of(&format!("{arch}.mhc.start"))? as usize;
        let mhc_end = u32_of(&format!("{arch}.mhc.end"))? as usize;
        let mhc_iterations = u32_of(&format!("{arch}.mhc.iterations"))? as usize;
        let mhc_streams = u32_of(&format!("{arch}.mhc.streams"))? as usize;
        if mhc_streams != 2 {
            return Err(reject("mhc.streams must be 2"));
        }
        let mhc_pre_scale = match scalar(header, &format!("{arch}.mhc.pre_scale"))? {
            MetaValue::F32(v) => v,
            _ => return Err(reject("mhc.pre_scale is not an f32")),
        };
        let mhc_fixed_identity = match scalar(header, &format!("{arch}.mhc.fixed_identity"))? {
            MetaValue::Bool(v) => v,
            _ => return Err(reject("mhc.fixed_identity is not a bool")),
        };
        let recurrent = bool_array(
            weights,
            header,
            &format!("{arch}.attention.recurrent_layers"),
            block_count,
        )?;
        // The MTP block must be full attention; only the trunk prefix is kept.
        if recurrent[TRUNK_LAYERS] {
            return Err(reject("the MTP block must be a full-attention layer"));
        }
        let plan = Self {
            loop_start,
            loop_end,
            loop_count,
            mhc_start,
            mhc_end,
            mhc_iterations,
            mhc_pre_scale,
            mhc_fixed_identity,
            recurrent: recurrent[..TRUNK_LAYERS].to_vec(),
        };
        plan.validate()?;
        for il in 0..TRUNK_LAYERS {
            if plan.recurrent[il] != plan.recurrent[plan.block(plan.physical(il))] {
                return Err(reject("loop visits have different attention kinds"));
            }
        }
        Ok(plan)
    }

    fn validate(&self) -> Result<(), EngError> {
        let width = self.loop_end.checked_sub(self.loop_start);
        if self.loop_count == 0 || width.is_none_or(|w| w == 0 || self.loop_end > TRUNK_LAYERS) {
            return Err(reject("invalid SenseNova Loop layer range or count"));
        }
        let width = self.loop_end - self.loop_start;
        let extra = (self.loop_count - 1)
            .checked_mul(width)
            .ok_or_else(|| reject("loop geometry overflows"))?;
        if TRUNK_LAYERS - self.loop_end < extra {
            return Err(reject(
                "SenseNova Loop execution plan exceeds the block count",
            ));
        }
        if TRUNK_LAYERS - extra != PHYSICAL_LAYERS {
            return Err(reject(
                "loop geometry disagrees with the physical layer count",
            ));
        }
        if self.mhc_start >= self.mhc_end
            || self.mhc_end > PHYSICAL_LAYERS
            || self.mhc_iterations == 0
            || self.mhc_iterations > 20
            || !self.mhc_pre_scale.is_finite()
            || self.mhc_pre_scale <= 0.0
        {
            return Err(reject("unsupported SenseNova mHC configuration"));
        }
        if !(self.mhc_end <= self.loop_start
            || (self.mhc_start >= self.loop_start && self.mhc_end <= self.loop_end)
            || (self.mhc_start >= self.loop_end))
        {
            return Err(reject("SenseNova mHC range crosses a Loop boundary"));
        }
        Ok(())
    }

    /// Physical layer index of logical layer `il`, mirroring the vendor
    /// `physical_layer`. Valid for `il < TRUNK_LAYERS`.
    pub(crate) fn physical(&self, il: usize) -> usize {
        let width = self.loop_end - self.loop_start;
        let extra = (self.loop_count - 1) * width;
        if il < self.loop_start {
            il
        } else if il < self.loop_end + extra {
            self.loop_start + (il - self.loop_start) % width
        } else {
            il - extra
        }
    }

    /// GGUF block index that stores the weights of physical layer `p`.
    /// The tail uses its logical index; revisits use the first pass's block.
    pub(crate) fn block(&self, p: usize) -> usize {
        if p < self.loop_end {
            p
        } else {
            p + (self.loop_count - 1) * (self.loop_end - self.loop_start)
        }
    }

    /// The trunk as the family builder takes it: one visit per logical layer.
    #[cfg(feature = "native-cuda")]
    pub(crate) fn visits(&self) -> Vec<crate::family::qwen::recipe::Visit> {
        use crate::family::qwen::recipe::{Streams, Visit};
        (0..TRUNK_LAYERS)
            .map(|il| {
                let p = self.physical(il);
                let block = self.block(p);
                Visit {
                    block,
                    recurrent: self.recurrent[block],
                    streams: if self.mhc_active(il) {
                        Streams::Two {
                            fork: p == self.mhc_start,
                        }
                    } else {
                        Streams::One
                    },
                }
            })
            .collect()
    }

    /// The routing of the two-stream range, as the step's site takes it.
    #[cfg(feature = "native-cuda")]
    pub(crate) fn routing(&self) -> cuda::step::Routing {
        cuda::step::Routing {
            pre_scale: self.mhc_pre_scale,
            iterations: self.mhc_iterations,
            fixed_identity: self.mhc_fixed_identity,
        }
    }

    /// Whether logical layer `il` routes through mHC streams.
    pub(crate) fn mhc_active(&self, il: usize) -> bool {
        let p = self.physical(il);
        (self.mhc_start..self.mhc_end).contains(&p)
    }
}

/// YaRN-scaled MRoPE geometry of the full-attention layers.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RopePlan {
    /// Rotary dimension count (even), the first dimensions of each head.
    pub(crate) dims: usize,
    /// Section sizes in pair units: temporal, height, width, extra.
    pub(crate) sections: [usize; 4],
    pub(crate) freq_base: f32,
    /// YaRN scale factor (`rope.scaling.factor`).
    pub(crate) factor: f32,
    /// Context the frequencies were trained at.
    pub(crate) original_context: usize,
}

impl RopePlan {
    pub(crate) fn from_header(
        weights: &GgufWeights,
        header: &GgufHeader,
        arch: &str,
    ) -> Result<Self, EngError> {
        let dims = match scalar(header, &format!("{arch}.rope.dimension_count"))? {
            MetaValue::U32(v) => v as usize,
            _ => return Err(reject("rope.dimension_count is not a u32")),
        };
        if dims == 0 || !dims.is_multiple_of(2) {
            return Err(reject("rope.dimension_count must be even and positive"));
        }
        if header
            .meta(&format!("{arch}.rope.scaling.attn_factor"))
            .is_some_and(|value| value != &MetaValue::F32(1.0))
        {
            return Err(reject("non-default rope attention scaling is unsupported"));
        }
        let scaling = format!("{arch}.rope.scaling.type");
        match scalar(header, &scaling)? {
            MetaValue::String(kind) if kind == "yarn" => {}
            _ => return Err(reject("only yarn rope scaling is supported")),
        }
        let factor = match scalar(header, &format!("{arch}.rope.scaling.factor"))? {
            MetaValue::F32(v) => v,
            _ => return Err(reject("rope.scaling.factor is not an f32")),
        };
        let original = match scalar(
            header,
            &format!("{arch}.rope.scaling.original_context_length"),
        )? {
            MetaValue::U32(v) => v as usize,
            _ => return Err(reject("rope.scaling.original_context_length is not a u32")),
        };
        let freq_base = match scalar(header, &format!("{arch}.rope.freq_base"))? {
            MetaValue::F32(v) => v,
            _ => return Err(reject("rope.freq_base is not an f32")),
        };
        let meta = match header.meta(&format!("{arch}.rope.dimension_sections")) {
            Some(MetaValue::Array(meta)) => meta,
            _ => return Err(reject("rope.dimension_sections is missing or not an array")),
        };
        if meta.len != 4 || !matches!(meta.elem_type, 4 | 5) {
            return Err(reject("rope.dimension_sections must hold four sections"));
        }
        let bytes = meta_array_bytes(weights, meta, 16)?;
        let mut sections = [0usize; 4];
        for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            sections[i] = u32::from_le_bytes(*chunk) as usize;
        }
        if sections.iter().sum::<usize>() != dims / 2 {
            return Err(reject(
                "rope.dimension_sections must sum to half the rotary dimensions",
            ));
        }
        if !(factor.is_finite() && factor > 1.0 && freq_base.is_finite() && freq_base > 1.0) {
            return Err(reject("rope scaling parameters are out of range"));
        }
        if original == 0 {
            return Err(reject("rope original context must be positive"));
        }
        Ok(Self {
            dims,
            sections,
            freq_base,
            factor,
            original_context: original,
        })
    }
}

fn scalar(header: &GgufHeader, key: &str) -> Result<MetaValue, EngError> {
    header
        .meta(key)
        .cloned()
        .ok_or_else(|| reject(format!("metadata key {key} is missing")))
        .and_then(|value| {
            if matches!(value, MetaValue::Array(_)) {
                Err(reject(format!("metadata key {key} must be a scalar")))
            } else {
                Ok(value)
            }
        })
}

fn meta_array_bytes(
    weights: &GgufWeights,
    meta: &ArrayMeta,
    len: usize,
) -> Result<Vec<u8>, EngError> {
    let mut buf = vec![0u8; len];
    weights.read_header_range(meta.byte_range.start, &mut buf)?;
    Ok(buf)
}

fn bool_array(
    weights: &GgufWeights,
    header: &GgufHeader,
    key: &str,
    len: usize,
) -> Result<Vec<bool>, EngError> {
    let meta = match header.meta(key) {
        Some(MetaValue::Array(meta)) => meta,
        Some(_) => return Err(reject(format!("metadata key {key} is not an array"))),
        None => return Err(reject(format!("metadata key {key} is missing"))),
    };
    // GGUF BOOL arrays carry element type 7; uint8 (0) is accepted for
    // writers that store flags that way.
    if meta.elem_type != 7 && meta.elem_type != 0 {
        return Err(reject(format!("metadata key {key} is not a bool array")));
    }
    if meta.len != len as u64 {
        return Err(reject(format!("metadata key {key} has the wrong length")));
    }
    let bytes = meta_array_bytes(weights, meta, len)?;
    Ok(bytes.into_iter().map(|b| b != 0).collect())
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("sensenova: {}", message.into()))
}
