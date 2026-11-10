//! The compile-cache keys, the checked block dimensions, and the compiled-program registry of
//! the MLX compile seam. Everything here is pure integer arithmetic over
//! plain structs, so it carries the same cfg as `shape` and its unit tests run on Linux: the MLX
//! arm builds the keys from its checked arguments and admits them into the per-adapter registry
//! before any native call.
//!
//! The cache key (`fun_id`) is the FNV-1a mix of ALL baked constants of a block, not only the
//! dtypes: the tag, the activation dtype, the scale dtype of EACH packed weight, the
//! quantization triple (mode, group, bits) of EACH packed weight, the hidden width, every
//! dimension of `dims`, `f32::to_bits(eps)` for the DeltaNet block, and for the MoE block
//! `experts`, `top_k`, `renorm`, and the clamp bound. Two engines whose baked constants differ
//! never read each other's compiled program.

use crate::EngError;
use crate::tensor::Dtype;

/// The cache-key tag of the DeltaNet decode closure: the initial state of the mix, distinct from
/// every other block tag, so a DeltaNet key and a MoE key never collide for equal field values.
pub(crate) const DELTA_TAG: u64 = 0x0036_5147_a5a0_0002;
pub(crate) const MOE_TAG: u64 = 0x0036_5147_a5a0_0003;
pub(crate) const FLINT_TAIL_TAG: u64 = 0x0036_5147_a5a0_0004;
pub(crate) const DELTA_LAYER_TAG: u64 = 0x0036_5147_a5a0_0007;
/// The cache-key tag of the attention pre-RoPE closure (projections, q/gate split, q/k norms).
pub(crate) const ATTN_PRE_TAG: u64 = 0x0036_5147_a5a0_0005;
pub(crate) const ATTN_POST_TAG: u64 = 0x0036_5147_a5a0_0006;

/// The activation dtype and the scale dtype of every packed weight are part of the compile
/// cache key, so the F32 and BF16 decoders never read each other's trace. Every dtype is its
/// own key word.
pub(crate) fn dtype_key(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::Bf16 => 1,
        Dtype::F16 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I64 => 5,
        Dtype::I16 => 6,
        Dtype::I32 => 7,
    }
}

/// Folds `parts` into one deterministic cache key (FNV-1a over the little-endian bytes, started
/// at `tag`): one compiled program per block kind and set of baked constants, so runs with
/// the same constants share one program and runs whose constants differ never read a stale trace.
pub(crate) fn mix_fun_id(tag: u64, parts: &[u64]) -> u64 {
    let mut hash: u64 = tag;
    for part in parts {
        for byte in part.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    hash
}

/// The checked dimension set of one compiled DeltaNet decode step. `dims` holds `kh, kd, vh,
/// vd, cd, taps`; the result is the full `dims` array of the native seam: the same six values
/// plus `kw = kh * kd` and `kw2 = 2 * kw`. Every input dimension, every product (`vh * vd`,
/// `kh * kd`, `2 * kw`), and the fused-projection identity `cd == 2 * kw + vh * vd` — the
/// q, k, v slices of the closure body cover `kw`, `kw`, and `vh * vd` channels — must fit an
/// i32, the width the native seam bakes; a violation is a `BadRequest` before any native call.
pub(crate) fn checked_delta_dims(dims: [u32; 6]) -> Result<[u32; 8], EngError> {
    let [kh, kd, vh, vd, cd, taps] = dims;
    let native = |name: &'static str, value: u64| -> Result<u32, EngError> {
        let narrow = i32::try_from(value)
            .map_err(|_| EngError::BadRequest(format!("compile seam {name} exceeds i32")))?;
        u32::try_from(narrow)
            .map_err(|_| EngError::BadRequest(format!("compile seam {name} is negative")))
    };
    for (name, value) in [
        ("kh", kh),
        ("kd", kd),
        ("vh", vh),
        ("vd", vd),
        ("cd", cd),
        ("taps", taps),
    ] {
        native(name, u64::from(value))?;
    }
    let vh_vd = u64::from(vh)
        .checked_mul(u64::from(vd))
        .ok_or_else(|| EngError::BadRequest("compile seam vh * vd exceeds i32".into()))?;
    let kw = u64::from(kh)
        .checked_mul(u64::from(kd))
        .ok_or_else(|| EngError::BadRequest("compile seam kh * kd exceeds i32".into()))?;
    let kw2 = kw
        .checked_mul(2)
        .ok_or_else(|| EngError::BadRequest("compile seam 2 * kw exceeds i32".into()))?;
    let (kw, kw2) = (native("kw", kw)?, native("kw2", kw2)?);
    native("vh * vd", vh_vd)?;
    let fused = u64::from(kw2)
        .checked_add(vh_vd)
        .ok_or_else(|| EngError::BadRequest("compile seam cd overflows".into()))?;
    if u64::from(cd) != fused {
        return Err(EngError::BadRequest(format!(
            "compile seam cd {cd} must equal 2 * kw + vh * vd = {fused}"
        )));
    }
    Ok([kh, kd, vh, vd, cd, taps, kw, kw2])
}

/// The baked constants of one compiled DeltaNet decode step: a
/// small plain struct of integers, so the key compiles and tests on Linux. The field order is the
/// mix order; every field is a baked constant of the closure body in `guard.cpp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeltaBaked {
    pub(crate) hidden: u32,
    /// `kh, kd, vh, vd, cd, taps` (conv_kernel - 1), `kw` (kh * kd), `kw2` (2 * kw): every
    /// dimension of the native seam's `dims`, as checked by [`checked_delta_dims`].
    pub(crate) dims: [u32; 8],
    /// 1 when the pre-recurrence middle runs as the ONE `delta_middle` Metal kernel
    /// (`Q36_FUSED_DECODE=1` with three taps): a different closure body, so a different key.
    pub(crate) fused: u32,
    /// `f32::to_bits(eps)`: the RMS norm epsilon of the block output norm.
    pub(crate) eps_bits: u32,
    /// The activation dtype key of the step (`dtype_key`): the dtype of the layer input and of
    /// the block output, and the target of the block's final `to_act` cast.
    pub(crate) act: u64,
    /// The scale dtype key of EACH packed weight, in the seam's weight order qkv, z, b, a, out.
    pub(crate) scale_keys: [u64; 5],
    /// The quantization triple (mode, group, bits; group or bits -1 when the mode has none) of
    /// EACH packed weight, in the same order.
    pub(crate) quant: [[i32; 3]; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MoeBaked {
    pub(crate) hidden: u32,
    pub(crate) experts: u32,
    pub(crate) top_k: u32,
    /// 1 when the gates renormalize over the selection, 0 otherwise.
    pub(crate) renorm: u32,
    /// The clamp bound of the expert ids, `experts - 1`: the ONE `mlx_minimum` clamp per layer.
    pub(crate) clamp: u32,
    /// `f32::to_bits(eps)`: the RMS norm epsilon of the folded post-attention norm.
    pub(crate) eps_bits: u32,
    pub(crate) act: u64,
    /// The scale dtype key of EACH packed weight, in the seam's weight order shared gate, up,
    /// down, then the stacked gate/up expert operand and the down stack.
    pub(crate) scale_keys: [u64; 5],
    /// The quantization triple of EACH packed weight, in the same order.
    pub(crate) quant: [[i32; 3]; 5],
}

pub(crate) fn delta_fun_id(baked: &DeltaBaked) -> u64 {
    let mut parts = Vec::with_capacity(32);
    parts.push(u64::from(baked.hidden));
    parts.extend(baked.dims.iter().map(|&dim| u64::from(dim)));
    parts.push(u64::from(baked.fused));
    parts.push(u64::from(baked.eps_bits));
    parts.push(baked.act);
    parts.extend(baked.scale_keys);
    for triple in &baked.quant {
        parts.extend(triple.iter().map(|&value| i64::from(value) as u64));
    }
    mix_fun_id(DELTA_TAG, &parts)
}

/// The baked constants of one compiled whole-layer DeltaNet chunk step (B1): the input
/// norm, the four projections, the fused prep/recurrence/post kernels, the mixer residual,
/// and the folded device-routed MoE rows sequence. T (the row count) is NOT baked: the
/// nonshapeless MLX cache specializes it under the one key, like the B axis of the decode
/// blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeltaLayerBaked {
    pub(crate) hidden: u32,
    /// `kh, kd, vh, vd, cd, taps` (conv_kernel - 1), `kw` (kh * kd), `kw2` (2 * kw).
    pub(crate) dims: [u32; 8],
    pub(crate) experts: u32,
    pub(crate) top_k: u32,
    /// 1 when the gates renormalize over the selection, 0 otherwise.
    pub(crate) renorm: u32,
    /// `f32::to_bits(eps)`: the RMS norm epsilon of the input and post norms.
    pub(crate) eps_bits: u32,
    /// The activation dtype key of the layer input and output.
    pub(crate) act: u64,
    /// The scale dtype key of EACH packed weight, in the seam's weight order qkv, z, b, a,
    /// out, shared gate, shared up, shared down, gate/up stack, down stack.
    pub(crate) scale_keys: [u64; 10],
    /// The quantization triple of EACH packed weight, in the same order.
    pub(crate) quant: [[i32; 3]; 10],
}

pub(crate) fn delta_layer_fun_id(baked: &DeltaLayerBaked) -> u64 {
    let mut parts = Vec::with_capacity(48);
    parts.push(u64::from(baked.hidden));
    parts.extend(baked.dims.iter().map(|&dim| u64::from(dim)));
    parts.push(u64::from(baked.experts));
    parts.push(u64::from(baked.top_k));
    parts.push(u64::from(baked.renorm));
    parts.push(u64::from(baked.eps_bits));
    parts.push(baked.act);
    parts.extend(baked.scale_keys);
    for triple in &baked.quant {
        parts.extend(triple.iter().map(|&value| i64::from(value) as u64));
    }
    mix_fun_id(DELTA_LAYER_TAG, &parts)
}

/// The baked constants of one compiled Flint decode tail: the
/// post-attention residual add, the RMS norm, and the dense SwiGLU, all in one closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlintTailBaked {
    /// The row count of the tail inputs: the closure compiles per row count, and the flag
    /// routes only the production decode shape (rows == 1).
    pub(crate) rows: u32,
    /// The hidden width `H` of the layer (the flat width of x and attended).
    pub(crate) hidden: u32,
    /// `f32::to_bits(eps)`: the RMS norm epsilon of the post-attention norm.
    pub(crate) eps_bits: u32,
    /// The activation dtype key of the tail (`dtype_key`): the dtype of x, attended, and the
    /// output.
    pub(crate) act: u64,
    /// The scale dtype key of EACH packed weight, in the seam's weight order gate, up, down.
    pub(crate) scale_keys: [u64; 3],
    /// The quantization triple (mode, group, bits; group or bits -1 when the mode has none) of
    /// EACH packed weight, in the same order.
    pub(crate) quant: [[i32; 3]; 3],
}

pub(crate) fn flint_tail_fun_id(baked: &FlintTailBaked) -> u64 {
    let mut parts = Vec::with_capacity(16);
    parts.push(u64::from(baked.rows));
    parts.push(u64::from(baked.hidden));
    parts.push(u64::from(baked.eps_bits));
    parts.push(baked.act);
    parts.extend(baked.scale_keys);
    for triple in &baked.quant {
        parts.extend(triple.iter().map(|&value| i64::from(value) as u64));
    }
    mix_fun_id(FLINT_TAIL_TAG, &parts)
}

/// The baked constants of one compiled attention pre-RoPE step: the q/k/v projections, the
/// interleaved q/gate split, and the q/k RMS norms, up to (but excluding) the position-scaled
/// RoPE, which stays eager because `mlx_fast_rope` takes the offset as a C++ int and a baked
/// offset would retrace every position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttnPreBaked {
    /// The hidden width `H` of the layer.
    pub(crate) hidden: u32,
    /// The query head count.
    pub(crate) heads: u32,
    /// The key/value head count (grouped-query attention).
    pub(crate) kv_heads: u32,
    /// The head width `D`.
    pub(crate) width: u32,
    /// `f32::to_bits(eps)`: the RMS norm epsilon of the q and k norms.
    pub(crate) eps_bits: u32,
    /// The activation dtype key of the step.
    pub(crate) act: u64,
    /// The scale dtype key of EACH packed weight, in the seam's weight order q, k, v.
    pub(crate) scale_keys: [u64; 3],
    /// The quantization triple of EACH packed weight, in the same order.
    pub(crate) quant: [[i32; 3]; 3],
}

pub(crate) fn attn_pre_fun_id(baked: &AttnPreBaked) -> u64 {
    let mut parts = Vec::with_capacity(16);
    parts.push(u64::from(baked.hidden));
    parts.push(u64::from(baked.heads));
    parts.push(u64::from(baked.kv_heads));
    parts.push(u64::from(baked.width));
    parts.push(u64::from(baked.eps_bits));
    parts.push(baked.act);
    parts.extend(baked.scale_keys);
    for triple in &baked.quant {
        parts.extend(triple.iter().map(|&value| i64::from(value) as u64));
    }
    mix_fun_id(ATTN_PRE_TAG, &parts)
}

/// The baked constants of one compiled attention post-core step: the sigmoid output gate, the
/// gated multiply, the cast to the activation dtype, and the output projection, from the
/// attention core output to the projected row. The cache append and the attention core stay
/// eager because they read the position-dependent prefix length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttnPostBaked {
    /// The query head count.
    pub(crate) heads: u32,
    /// The head width `D`.
    pub(crate) width: u32,
    /// The hidden width `H` of the output projection.
    pub(crate) hidden: u32,
    /// The activation dtype key of the step.
    pub(crate) act: u64,
    /// The dtype key of the attention core output operand: it is F32 over the F32 cache and
    /// the query dtype over the quantized cache, so it joins the key even though the body
    /// relies on MLX's runtime promotion instead of branching on it.
    pub(crate) att: u64,
    /// The scale dtype key of the packed output weight.
    pub(crate) scale_key: u64,
    /// The quantization triple of the packed output weight.
    pub(crate) quant: [i32; 3],
}

pub(crate) fn attn_post_fun_id(baked: &AttnPostBaked) -> u64 {
    let parts = [
        u64::from(baked.heads),
        u64::from(baked.width),
        u64::from(baked.hidden),
        baked.act,
        baked.att,
        baked.scale_key,
        i64::from(baked.quant[0]) as u64,
        i64::from(baked.quant[1]) as u64,
        i64::from(baked.quant[2]) as u64,
    ];
    mix_fun_id(ATTN_POST_TAG, &parts)
}

pub(crate) fn moe_fun_id(baked: &MoeBaked) -> u64 {
    let mut parts = Vec::with_capacity(32);
    parts.push(u64::from(baked.hidden));
    parts.push(u64::from(baked.experts));
    parts.push(u64::from(baked.top_k));
    parts.push(u64::from(baked.renorm));
    parts.push(u64::from(baked.clamp));
    parts.push(u64::from(baked.eps_bits));
    parts.push(baked.act);
    parts.extend(baked.scale_keys);
    for triple in &baked.quant {
        parts.extend(triple.iter().map(|&value| i64::from(value) as u64));
    }
    mix_fun_id(MOE_TAG, &parts)
}

pub(crate) use super::ProgramLimit;

/// The compiled-program registry of ONE adapter context:
/// the MLX compile cache is thread-local and dies with its thread, so the registry lives in the
/// adapter state and dies with it — a fresh adapter starts with an empty registry and its own
/// limit. The seam consults it BEFORE any native call, so a refused key compiles nothing.
#[derive(Debug, Default)]
pub(crate) struct ProgramRegistry {
    seen: Vec<u64>,
    limit: ProgramLimit,
}

impl ProgramRegistry {
    /// Configuration is allowed only before any key is admitted; it never resets deduplication.
    pub(crate) fn set_limit(&mut self, limit: ProgramLimit) -> Result<(), EngError> {
        if !self.seen.is_empty() {
            return Err(EngError::BadRequest(
                "compile_program_limit must be configured before compiling any program".into(),
            ));
        }
        self.limit = limit;
        Ok(())
    }

    pub(crate) fn limit(&self) -> ProgramLimit {
        self.limit
    }

    pub(crate) fn admit(&mut self, id: u64) -> Result<bool, EngError> {
        if self.seen.contains(&id) {
            return Ok(false);
        }
        if self.seen.len() >= self.limit.get() {
            return Err(EngError::BadRequest(format!(
                "at most {} compiled programs per adapter; cache key {id:#x} is a \
                 further distinct set of baked constants",
                self.limit.get()
            )));
        }
        self.seen.push(id);
        Ok(true)
    }

    /// The distinct ids registered so far, in admission order. The load-time warm-up plan and
    /// the tests read it.
    #[cfg(test)]
    pub(crate) fn seen(&self) -> &[u64] {
        &self.seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DeltaNet constant set of the real model (hidden 2048, kh 16, kd 128, vh 32, vd 128,
    /// cd 8192, taps 3, kw 2048, kw2 4096) with affine 4-bit weights in groups of 64.
    fn delta() -> DeltaBaked {
        DeltaBaked {
            hidden: 2048,
            dims: [16, 128, 32, 128, 8192, 3, 2048, 4096],
            fused: 0,
            eps_bits: 1e-6f32.to_bits(),
            act: dtype_key(Dtype::F32),
            scale_keys: [1, 1, 1, 1, 1],
            quant: [[0, 64, 4]; 5],
        }
    }

    /// A MoE constant set of the real model (hidden 2048, 256 experts, top 8, renorm,
    /// clamp 255) with the same affine quantization.
    fn moe() -> MoeBaked {
        MoeBaked {
            hidden: 2048,
            experts: 256,
            top_k: 8,
            renorm: 1,
            clamp: 255,
            eps_bits: 1.0e-6f32.to_bits(),
            act: dtype_key(Dtype::F32),
            scale_keys: [1, 1, 1, 1, 1],
            quant: [[0, 64, 4]; 5],
        }
    }

    /// A compiled Flint tail constant set of the served decode shape (one row, hidden
    /// 2048, eps 1e-6, BF16 activation, BF16 scales, affine 4-bit groups of 64).
    fn flint_tail() -> FlintTailBaked {
        FlintTailBaked {
            rows: 1,
            hidden: 2048,
            eps_bits: 1e-6f32.to_bits(),
            act: dtype_key(Dtype::Bf16),
            scale_keys: [1, 1, 1],
            quant: [[0, 64, 4]; 3],
        }
    }

    /// An attention pre-RoPE constant set of the real model (hidden 2048, 32 query heads,
    /// 8 key/value heads, width 128, eps 1e-6) with the same affine quantization.
    fn attn() -> AttnPreBaked {
        AttnPreBaked {
            hidden: 2048,
            heads: 32,
            kv_heads: 8,
            width: 128,
            eps_bits: 1e-6f32.to_bits(),
            act: dtype_key(Dtype::F32),
            scale_keys: [1, 1, 1],
            quant: [[0, 64, 4]; 3],
        }
    }

    /// An attention post-core constant set of the same model: the output projection of
    /// 32 heads of width 128 back to hidden 2048, F32 step, F32 attention core output.
    fn attn_post() -> AttnPostBaked {
        AttnPostBaked {
            heads: 32,
            width: 128,
            hidden: 2048,
            act: dtype_key(Dtype::F32),
            att: dtype_key(Dtype::F32),
            scale_key: 1,
            quant: [0, 64, 4],
        }
    }

    #[test]
    fn equal_constants_give_equal_keys() {
        assert_eq!(delta_fun_id(&delta()), delta_fun_id(&delta()));
        assert_eq!(moe_fun_id(&moe()), moe_fun_id(&moe()));
        assert_eq!(attn_pre_fun_id(&attn()), attn_pre_fun_id(&attn()));
        assert_eq!(
            attn_post_fun_id(&attn_post()),
            attn_post_fun_id(&attn_post())
        );
        assert_eq!(
            flint_tail_fun_id(&flint_tail()),
            flint_tail_fun_id(&flint_tail())
        );
    }

    /// The attention keys of the three block kinds never collide, and every scalar field,
    /// scale key, and quantization value of the attention block changes its key.
    #[test]
    fn every_single_attn_pre_constant_changes_the_key() {
        let base = attn();
        let rows: Vec<(&str, AttnPreBaked)> = vec![
            (
                "hidden",
                AttnPreBaked {
                    hidden: 2049,
                    ..base
                },
            ),
            ("heads", AttnPreBaked { heads: 33, ..base }),
            (
                "kv_heads",
                AttnPreBaked {
                    kv_heads: 9,
                    ..base
                },
            ),
            ("width", AttnPreBaked { width: 129, ..base }),
            (
                "eps_bits",
                AttnPreBaked {
                    eps_bits: base.eps_bits ^ 1,
                    ..base
                },
            ),
            (
                "act",
                AttnPreBaked {
                    act: dtype_key(Dtype::Bf16),
                    ..base
                },
            ),
            (
                "scale_keys[0]",
                AttnPreBaked {
                    scale_keys: [0, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[2]",
                AttnPreBaked {
                    scale_keys: [1, 1, 0],
                    ..base
                },
            ),
            (
                "quant[1] mode",
                AttnPreBaked {
                    quant: [[0, 64, 4], [1, 64, 4], [0, 64, 4]],
                    ..base
                },
            ),
            (
                "quant[2] bits",
                AttnPreBaked {
                    quant: [[0, 64, 4], [0, 64, 4], [0, 64, 8]],
                    ..base
                },
            ),
        ];
        let base_id = attn_pre_fun_id(&base);
        for (name, variant) in rows {
            assert_ne!(
                attn_pre_fun_id(&variant),
                base_id,
                "changing {name} must change the attention key"
            );
        }
        assert_ne!(
            base_id,
            delta_fun_id(&delta()),
            "distinct from the delta key"
        );
        assert_ne!(base_id, moe_fun_id(&moe()), "distinct from the moe key");
        assert_ne!(
            base_id,
            flint_tail_fun_id(&flint_tail()),
            "distinct from the flint tail key"
        );
    }

    /// Every scalar field, the two dtype keys, the scale key, and every quantization value of
    /// the attention post-core block changes its key, and the post key never collides with
    /// the keys of the other block kinds.
    #[test]
    fn every_single_attn_post_constant_changes_the_key() {
        let base = attn_post();
        let rows: Vec<(&str, AttnPostBaked)> = vec![
            ("heads", AttnPostBaked { heads: 33, ..base }),
            ("width", AttnPostBaked { width: 129, ..base }),
            (
                "hidden",
                AttnPostBaked {
                    hidden: 2049,
                    ..base
                },
            ),
            (
                "act",
                AttnPostBaked {
                    act: dtype_key(Dtype::Bf16),
                    ..base
                },
            ),
            (
                "att",
                AttnPostBaked {
                    att: dtype_key(Dtype::Bf16),
                    ..base
                },
            ),
            (
                "scale_key",
                AttnPostBaked {
                    scale_key: 0,
                    ..base
                },
            ),
            (
                "quant mode",
                AttnPostBaked {
                    quant: [1, 64, 4],
                    ..base
                },
            ),
            (
                "quant group",
                AttnPostBaked {
                    quant: [0, 32, 4],
                    ..base
                },
            ),
            (
                "quant bits",
                AttnPostBaked {
                    quant: [0, 64, 8],
                    ..base
                },
            ),
        ];
        let base_id = attn_post_fun_id(&base);
        for (name, variant) in rows {
            assert_ne!(
                attn_post_fun_id(&variant),
                base_id,
                "changing {name} must change the attention post key"
            );
        }
        assert_ne!(
            base_id,
            attn_pre_fun_id(&attn()),
            "distinct from the attention pre key"
        );
        assert_ne!(
            base_id,
            delta_fun_id(&delta()),
            "distinct from the delta key"
        );
        assert_ne!(base_id, moe_fun_id(&moe()), "distinct from the moe key");
        assert_ne!(
            base_id,
            flint_tail_fun_id(&flint_tail()),
            "distinct from the flint tail key"
        );
    }

    /// The row count of the delta table below: one row per scalar field, per dimension, per
    /// scale key, and per quantization value.
    const DELTA_ROWS: usize = 1 + 1 + 8 + 1 + 1 + 5 + 5 * 3;
    const MOE_ROWS: usize = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 5 + 5 * 3;
    /// The row count of the Flint tail table below: one row per scalar field, per scale
    /// key, and per quantization value.
    const FLINT_ROWS: usize = 1 + 1 + 1 + 1 + 3 + 3 * 3;

    #[test]
    fn every_single_flint_tail_constant_changes_the_key() {
        let base = flint_tail();
        // One row per scalar field, per scale key, and per quantization value: changing
        // exactly one baked constant must change the cache key, and the Flint tail key must
        // never collide with the DeltaNet or MoE keys of this file's other fixtures.
        let rows: Vec<(&str, FlintTailBaked)> = vec![
            ("rows", FlintTailBaked { rows: 2, ..base }),
            (
                "hidden",
                FlintTailBaked {
                    hidden: 2049,
                    ..base
                },
            ),
            (
                "eps_bits",
                FlintTailBaked {
                    eps_bits: base.eps_bits ^ 1,
                    ..base
                },
            ),
            (
                "act",
                FlintTailBaked {
                    act: dtype_key(Dtype::F32),
                    ..base
                },
            ),
            (
                "scale_keys[0]",
                FlintTailBaked {
                    scale_keys: [0, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[1]",
                FlintTailBaked {
                    scale_keys: [1, 0, 1],
                    ..base
                },
            ),
            (
                "scale_keys[2]",
                FlintTailBaked {
                    scale_keys: [1, 1, 0],
                    ..base
                },
            ),
        ]
        .into_iter()
        .chain((0..3).flat_map(|weight| {
            (0..3).map(move |field| {
                let names = ["mode", "group", "bits"];
                let name = names[field];
                let mut quant = base.quant;
                quant[weight][field] += 1;
                (name, FlintTailBaked { quant, ..base })
            })
        }))
        .collect();
        // The exact row count: deleting a row of the table must fail here, not pass silently.
        assert_eq!(rows.len(), FLINT_ROWS, "the table covers every field");
        for (name, changed) in rows {
            assert_ne!(
                flint_tail_fun_id(&changed),
                flint_tail_fun_id(&base),
                "changing {name} must change the cache key"
            );
        }
        assert_ne!(
            flint_tail_fun_id(&base),
            delta_fun_id(&delta()),
            "the Flint tail key never collides with the DeltaNet key"
        );
        assert_ne!(
            flint_tail_fun_id(&base),
            moe_fun_id(&moe()),
            "the Flint tail key never collides with the MoE key"
        );
    }

    #[test]
    fn every_single_delta_constant_changes_the_key() {
        let base = delta();
        // One row per scalar field, per dimension, per scale key, and per quantization value:
        // changing exactly one baked constant must change the cache key.
        let rows: Vec<(&str, DeltaBaked)> = vec![
            (
                "hidden",
                DeltaBaked {
                    hidden: 2049,
                    ..base
                },
            ),
            ("fused", DeltaBaked { fused: 1, ..base }),
            (
                "dims[0] kh",
                DeltaBaked {
                    dims: [17, 128, 32, 128, 8192, 3, 2048, 4096],
                    ..base
                },
            ),
            (
                "dims[1] kd",
                DeltaBaked {
                    dims: [16, 129, 32, 128, 8192, 3, 2048, 4096],
                    ..base
                },
            ),
            (
                "dims[2] vh",
                DeltaBaked {
                    dims: [16, 128, 33, 128, 8192, 3, 2048, 4096],
                    ..base
                },
            ),
            (
                "dims[3] vd",
                DeltaBaked {
                    dims: [16, 128, 32, 129, 8192, 3, 2048, 4096],
                    ..base
                },
            ),
            (
                "dims[4] cd",
                DeltaBaked {
                    dims: [16, 128, 32, 128, 8193, 3, 2048, 4096],
                    ..base
                },
            ),
            (
                "dims[5] taps",
                DeltaBaked {
                    dims: [16, 128, 32, 128, 8192, 4, 2048, 4096],
                    ..base
                },
            ),
            (
                "dims[6] kw",
                DeltaBaked {
                    dims: [16, 128, 32, 128, 8192, 3, 2049, 4096],
                    ..base
                },
            ),
            (
                "dims[7] kw2",
                DeltaBaked {
                    dims: [16, 128, 32, 128, 8192, 3, 2048, 4097],
                    ..base
                },
            ),
            (
                "eps_bits",
                DeltaBaked {
                    eps_bits: base.eps_bits ^ 1,
                    ..base
                },
            ),
            (
                "act",
                DeltaBaked {
                    act: dtype_key(Dtype::Bf16),
                    ..base
                },
            ),
            (
                "scale_keys[0]",
                DeltaBaked {
                    scale_keys: [0, 1, 1, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[1]",
                DeltaBaked {
                    scale_keys: [1, 0, 1, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[2]",
                DeltaBaked {
                    scale_keys: [1, 1, 0, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[3]",
                DeltaBaked {
                    scale_keys: [1, 1, 1, 0, 1],
                    ..base
                },
            ),
            (
                "scale_keys[4]",
                DeltaBaked {
                    scale_keys: [1, 1, 1, 1, 0],
                    ..base
                },
            ),
        ]
        .into_iter()
        .chain((0..5).flat_map(|weight| {
            (0..3).map(move |field| {
                let names = ["mode", "group", "bits"];
                let name = names[field];
                let mut quant = base.quant;
                quant[weight][field] += 1;
                (name, DeltaBaked { quant, ..base })
            })
        }))
        .collect();
        // The exact row count: deleting a row of the table must fail here, not pass silently.
        assert_eq!(rows.len(), DELTA_ROWS, "the table covers every field");
        for (name, baked) in rows {
            let key = delta_fun_id(&baked);
            assert_ne!(
                key,
                delta_fun_id(&base),
                "changing {name} must change the cache key"
            );
        }
    }

    #[test]
    fn every_single_moe_constant_changes_the_key() {
        let base = moe();
        let rows: Vec<(&str, MoeBaked)> = vec![
            (
                "hidden",
                MoeBaked {
                    hidden: 2049,
                    ..base
                },
            ),
            (
                "experts",
                MoeBaked {
                    experts: 257,
                    ..base
                },
            ),
            ("top_k", MoeBaked { top_k: 9, ..base }),
            ("renorm", MoeBaked { renorm: 0, ..base }),
            ("clamp", MoeBaked { clamp: 256, ..base }),
            (
                "eps_bits",
                MoeBaked {
                    eps_bits: 1.0e-5f32.to_bits(),
                    ..base
                },
            ),
            (
                "act",
                MoeBaked {
                    act: dtype_key(Dtype::Bf16),
                    ..base
                },
            ),
            (
                "scale_keys[0]",
                MoeBaked {
                    scale_keys: [0, 1, 1, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[1]",
                MoeBaked {
                    scale_keys: [1, 0, 1, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[2]",
                MoeBaked {
                    scale_keys: [1, 1, 0, 1, 1],
                    ..base
                },
            ),
            (
                "scale_keys[3]",
                MoeBaked {
                    scale_keys: [1, 1, 1, 0, 1],
                    ..base
                },
            ),
            (
                "scale_keys[4]",
                MoeBaked {
                    scale_keys: [1, 1, 1, 1, 0],
                    ..base
                },
            ),
        ]
        .into_iter()
        .chain((0..5).flat_map(|weight| {
            (0..3).map(move |field| {
                let names = ["mode", "group", "bits"];
                let name = names[field];
                let mut quant = base.quant;
                quant[weight][field] += 1;
                (name, MoeBaked { quant, ..base })
            })
        }))
        .collect();
        // The exact row count: deleting a row of the table must fail here, not pass silently.
        assert_eq!(rows.len(), MOE_ROWS, "the table covers every field");
        for (name, baked) in rows {
            let key = moe_fun_id(&baked);
            assert_ne!(
                key,
                moe_fun_id(&base),
                "changing {name} must change the cache key"
            );
        }
    }

    #[test]
    fn the_delta_and_moe_keys_never_collide_for_equal_field_values() {
        // Equal hidden width, activation dtype, scale dtypes, and quantization triples: the two
        // keys still differ, because the tag is the initial state of the mix.
        let d = delta();
        let m = MoeBaked {
            hidden: d.hidden,
            experts: 256,
            top_k: 8,
            renorm: 1,
            clamp: 255,
            eps_bits: 1.0e-6f32.to_bits(),
            act: d.act,
            scale_keys: [d.scale_keys[0], 1, 1, 1, 1],
            quant: [d.quant[0]; 5],
        };
        assert_ne!(
            delta_fun_id(&d),
            moe_fun_id(&m),
            "the block tags keep the two key spaces apart"
        );
        // The same holds for identical part lists: only the tag differs.
        let parts = [u64::from(d.hidden), d.act, 7];
        assert_ne!(mix_fun_id(DELTA_TAG, &parts), mix_fun_id(MOE_TAG, &parts));
    }

    /// Every dtype is its own key word: two different dtypes never share a cache-key word, so a
    /// scale stored in one dtype can never reuse the trace of another.
    #[test]
    fn every_dtype_is_its_own_key_word() {
        let words = [
            dtype_key(Dtype::F32),
            dtype_key(Dtype::Bf16),
            dtype_key(Dtype::F16),
            dtype_key(Dtype::U8),
            dtype_key(Dtype::U32),
            dtype_key(Dtype::I64),
            dtype_key(Dtype::I16),
            dtype_key(Dtype::I32),
        ];
        assert_eq!(words, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// The checked dimension set of the real model and the tiny model: the six block dimensions
    /// widen to the full native `dims` with `kw` and `kw2`.
    #[test]
    fn checked_delta_dims_widens_the_dimensions_of_both_models() {
        assert_eq!(
            checked_delta_dims([16, 128, 32, 128, 8192, 3]).unwrap(),
            [16, 128, 32, 128, 8192, 3, 2048, 4096],
            "the real model"
        );
        assert_eq!(
            checked_delta_dims([1, 32, 2, 32, 128, 3]).unwrap(),
            [1, 32, 2, 32, 128, 3, 32, 64],
            "the tiny model: cd = 2 * kw + vh * vd = 128"
        );
    }

    /// Every refusal of the checked dimension set: an input dimension, each product, and the
    /// fused-projection identity itself must refuse with `BadRequest` before any native call.
    #[test]
    fn checked_delta_dims_refuses_every_violation() {
        let refuse = |dims: [u32; 6], want: &str| {
            let EngError::BadRequest(message) = checked_delta_dims(dims).unwrap_err() else {
                panic!("the violation must be a BadRequest: {dims:?}");
            };
            assert!(
                message.contains(want),
                "the refusal names {want}: {message}"
            );
        };
        refuse([16, 1 << 31, 1, 1, 4, 1], "kd exceeds i32");
        refuse([1, 1, 1 << 16, 1 << 16, 4, 1], "vh * vd exceeds i32");
        refuse([1 << 16, 1 << 16, 1, 1, 4, 1], "kw exceeds i32");
        // 2 * kw beyond i32 (kh * kd = 2^30, so kw2 = 2^31).
        refuse([1 << 15, 1 << 15, 1, 1, 4, 1], "kw2 exceeds i32");
        refuse([1, 32, 2, 32, 129, 3], "must equal 2 * kw + vh * vd");
        refuse([1, 32, 2, 32, 127, 3], "must equal 2 * kw + vh * vd");
    }

    /// Default adapters admit eight distinct ids and deduplicate repeats even when full.
    #[test]
    fn explicit_program_budget_is_bounded_and_deduplicates() {
        let mut registry = ProgramRegistry::default();
        registry.set_limit(ProgramLimit::new(12).unwrap()).unwrap();
        assert_eq!(registry.limit().get(), 12);
        for id in 0..12 {
            assert_eq!(registry.admit(id), Ok(true));
        }
        assert_eq!(registry.admit(0), Ok(false));
        assert!(registry.admit(12).is_err());
        assert_eq!(registry.seen().len(), 12);
        assert!(registry.set_limit(ProgramLimit::new(13).unwrap()).is_err());
        assert_eq!(registry.limit().get(), 12);
        assert_eq!(ProgramRegistry::default().limit().get(), 8);
    }

    #[test]
    fn program_budget_rejects_zero_and_address_space_overflow() {
        assert!(ProgramLimit::new(0).is_err());
        assert!(ProgramLimit::new(usize::MAX).is_err());
        assert!(ProgramLimit::new(isize::MAX as usize / size_of::<u64>() + 1).is_err());
        assert_eq!(ProgramLimit::new(1).unwrap().get(), 1);
    }

    #[test]
    fn a_local_registry_holds_the_same_rule() {
        let mut registry = ProgramRegistry::default();
        assert!(registry.seen().is_empty(), "a fresh adapter holds none");
        for id in 7u64..15 {
            assert_eq!(registry.admit(id), Ok(true));
        }
        assert_eq!(registry.seen().len(), 8);
        // A repeat of a registered id is fine: it is a cache hit, not a new program.
        assert_eq!(registry.admit(8), Ok(false));
        let ninth = registry.admit(16);
        let EngError::BadRequest(message) = ninth.unwrap_err() else {
            panic!("the limit refusal is a BadRequest");
        };
        assert!(
            message.contains("at most 8 compiled programs per adapter"),
            "the refusal names the limit: {message}"
        );
        // The refused id was not registered, so the registry still holds exactly the limit and
        // stays refused for a further distinct id.
        assert_eq!(registry.seen().len(), 8);
        assert!(registry.admit(24).is_err(), "the limit stays refused");
    }
}
